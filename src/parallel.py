"""
Parallel execution of cross-validation folds across multiple GPUs.

Why fold-level
--------------
Nested cross-validation multiplies the work by the inner search: 10 outer folds
x ~30 transformer trials x 3 inner folds is ~900 network fits per experiment.
The outer folds are completely independent -- each one scales, selects
features, tunes and fits using only its own training rows -- so they are the
natural unit to distribute, and distributing them requires no change to the
statistics.

The alternative, parallelising Optuna trials within a fold, needs shared study
storage and gives a messier audit trail. Fold-level keeps each search
self-contained and reproducible: fold k always sees the same data and the same
seed whether it ran alone or alongside three others.

With 10 folds on 4 GPUs the schedule is 4 + 4 + 2, so expect roughly a 2.5x
wall-clock speedup rather than 4x. Workers pull from a queue as they free up,
so a slow fold does not stall the others.

CPU oversubscription
--------------------
XGBoost defaults to n_jobs=-1. Four concurrent workers each grabbing every core
is slower than one, so `cpu_threads_per_worker()` divides the machine's cores
by the worker count and the CV engine passes that to XGBoost and the BLAS
libraries.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Callable, Iterable, Optional


# ---------------------------------------------------------------------------
# CUDA discovery, out of process
# ---------------------------------------------------------------------------
# IMPORTANT: this module must never `import torch` in the parent process.
#
# On macOS (and on some Linux installs) torch ships its own libomp, and a
# process that has loaded torch will SEGFAULT when it later runs multithreaded
# XGBoost. It is silent, immediate, and reproducible: torch + XGBoost with
# OMP_NUM_THREADS > 1 crashes; either alone is fine. KMP_DUPLICATE_LIB_OK does
# not help.
#
# The pipeline therefore keeps the two apart by process. XGBoost folds run in
# workers that never import torch; transformer folds import torch and (with the
# default SelectKBest selector) never touch XGBoost. Discovering how many GPUs
# exist would break that rule in the parent, so it is done in a throwaway
# subprocess instead.

_CUDA_CACHE = None

_PROBE_SCRIPT = """
import json, sys
try:
    import torch
    if torch.cuda.is_available():
        out = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({"index": i, "name": p.name, "total_memory": p.total_memory})
        print(json.dumps(out))
    else:
        print("[]")
except Exception:
    print("[]")
"""


def probe_cuda_devices(force=False):
    """CUDA devices visible to torch, discovered in a subprocess.

    Returns a list of {index, name, total_memory}; empty when torch is missing
    or no GPU is present. Cached, because spawning a python interpreter is not
    free and the answer does not change during a run.
    """
    global _CUDA_CACHE
    if _CUDA_CACHE is not None and not force:
        return _CUDA_CACHE

    import json
    import subprocess

    try:
        result = subprocess.run([sys.executable, "-c", _PROBE_SCRIPT],
                                capture_output=True, text=True, timeout=120)
        _CUDA_CACHE = json.loads(result.stdout.strip() or "[]")
    except Exception:                                          # noqa: BLE001
        _CUDA_CACHE = []
    return _CUDA_CACHE


def configure_openmp(n_threads):
    """Pin every OpenMP-based library to `n_threads`.

    Must run BEFORE torch or xgboost is imported: the OpenMP runtime reads these
    at load time and ignores later changes.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n_threads)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_devices(spec: Optional[str] = None, verbose: bool = True) -> list[str]:
    """Turn a --gpus specification into a list of torch device strings.

    Accepted forms
        None / "auto"  -> every visible CUDA device, else ["cpu"]
        "0,1,2,3"      -> ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        "cuda:1"       -> ["cuda:1"]
        "cpu"          -> ["cpu"]
        "0,0"          -> ["cuda:0", "cuda:0"]  (two workers sharing one GPU)

    Requested devices that do not exist are dropped with a warning rather than
    crashing three hours into a run.
    """
    n_cuda = len(probe_cuda_devices())

    if spec is None or str(spec).strip().lower() in ("auto", ""):
        devices = [f"cuda:{i}" for i in range(n_cuda)] or ["cpu"]
        if verbose:
            print(f"[Parallel] auto-detected {len(devices)} device(s): {devices}")
        return devices

    raw = [tok.strip() for tok in str(spec).split(",") if tok.strip()]
    devices: list[str] = []
    for tok in raw:
        low = tok.lower()
        if low == "cpu":
            devices.append("cpu")
            continue
        index = int(low.replace("cuda:", "")) if low.startswith("cuda:") else int(low)
        if index >= n_cuda:
            print(f"[Parallel] WARNING: cuda:{index} requested but only {n_cuda} "
                  f"CUDA device(s) visible — dropping it.")
            continue
        devices.append(f"cuda:{index}")

    if not devices:
        print("[Parallel] WARNING: no requested device is available; falling back to CPU.")
        devices = ["cpu"]

    if verbose:
        print(f"[Parallel] using {len(devices)} worker device(s): {devices}")
    return devices


def cpu_threads_per_worker(n_workers: int, reserve: int = 1) -> int:
    """Cores each worker may use without the pool oversubscribing the machine."""
    total = os.cpu_count() or 1
    return max(1, (total - reserve) // max(1, n_workers))


def describe_devices(devices: list[str]) -> str:
    """Human-readable device inventory, for the run header and the log."""
    props = {d["index"]: d for d in probe_cuda_devices()}

    seen = {}
    for dev in devices:
        seen[dev] = seen.get(dev, 0) + 1

    lines = [f"{len(devices)} worker(s)"]
    for dev, count in seen.items():
        if dev == "cpu":
            lines.append(f"  cpu x{count}")
            continue
        idx = int(dev.split(":")[1])
        info = props.get(idx)
        if info:
            lines.append(f"  {dev} x{count}  {info['name']}  "
                         f"{info['total_memory'] / 1e9:.1f} GB")
        else:
            lines.append(f"  {dev} x{count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------

_WORKER_DEVICE: Optional[str] = None
_WORKER_THREADS: int = 1


def _init_worker(device_queue, threads: int) -> None:
    """Claim one device from the shared queue, once, at worker startup.

    Each pool worker pops a device and keeps it for its whole life, so a worker
    never migrates between GPUs mid-run and CUDA contexts are created once.

    Deliberately does NOT import torch: a worker running XGBoost must stay
    torch-free (see the note above probe_cuda_devices). The transformer path
    imports torch itself and calls torch.cuda.set_device() there.
    """
    global _WORKER_DEVICE, _WORKER_THREADS
    _WORKER_DEVICE = device_queue.get()
    _WORKER_THREADS = threads
    configure_openmp(threads)

    print(f"[Parallel] worker pid={os.getpid()} claimed {_WORKER_DEVICE} "
          f"({threads} cpu thread(s))", flush=True)


def get_worker_device() -> str:
    """The device this worker owns. 'cpu' when running unparallelised."""
    return _WORKER_DEVICE or "cpu"


def get_worker_threads() -> int:
    """CPU threads this worker may use (pass to XGBoost's n_jobs)."""
    return _WORKER_THREADS


def _call(payload):
    """Top-level trampoline so the submitted callable stays picklable."""
    fn, kwargs = payload
    try:
        return {"ok": True, "result": fn(device=get_worker_device(),
                                         n_threads=get_worker_threads(), **kwargs)}
    except Exception as exc:                                   # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "kwargs": {k: v for k, v in kwargs.items()
                           if isinstance(v, (str, int, float, bool, type(None)))}}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_tasks(worker_fn: Callable, task_kwargs: Iterable[dict],
              devices: list[str], sequential: bool = False,
              label: str = "task") -> list[Any]:
    """Run `worker_fn(device=..., n_threads=..., **kwargs)` for each task.

    Results come back **in task order**, not completion order, so downstream
    code can index by fold number without re-sorting.

    Parameters
    ----------
    worker_fn : must be importable by name in a fresh interpreter (a
        module-level function, not a closure or a lambda) because the pool uses
        the 'spawn' start method -- required for CUDA, which cannot be
        initialised in a forked child.
    devices : one entry per worker; repeat an entry to put two workers on one GPU.
    sequential : run in-process instead. Use only for debugging -- exceptions
        keep their real traceback and pdb works, but the torch/XGBoost isolation
        is lost, so run_pipeline.py pins OpenMP to one thread in that mode.

    A failing task does not kill the pool. Every failure is collected and
    re-raised together at the end, with the child's traceback, so one bad fold
    does not silently drop out of a 10-fold result.
    """
    tasks = list(task_kwargs)
    if not tasks:
        return []

    # In-process execution ONLY when explicitly requested.
    #
    # It is tempting to skip the pool when there is a single device, and an
    # earlier version did. That is a trap: with one device, every experiment
    # runs in the parent, so an XGBoost experiment loads XGBoost's OpenMP and a
    # later transformer experiment loads torch's into the same process. The two
    # then deadlock (0% CPU, no traceback, forever) or segfault. Spawning a
    # one-worker pool costs about a second per experiment and guarantees the
    # two never meet.
    if sequential:
        device = devices[0]
        threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
        print(f"[Parallel] --sequential: running {len(tasks)} {label}(s) in-process "
              f"on {device} with {threads} thread(s).")
        results = []
        for i, kwargs in enumerate(tasks, 1):
            print(f"\n[Parallel] {label} {i}/{len(tasks)} on {device}", flush=True)
            results.append(worker_fn(device=device, n_threads=threads, **kwargs))
        return results

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = mp.get_context("spawn")
    n_workers = max(1, min(len(devices), len(tasks)))
    threads = cpu_threads_per_worker(n_workers)

    device_queue = ctx.Queue()
    for i in range(n_workers):
        device_queue.put(devices[i % len(devices)])

    print(f"\n[Parallel] dispatching {len(tasks)} {label}(s) over {n_workers} worker(s): "
          f"{devices[:n_workers]}")
    print(f"[Parallel] {threads} cpu thread(s) per worker "
          f"(of {os.cpu_count()} cores) to avoid oversubscription")

    results: list[Any] = [None] * len(tasks)
    failures: list[dict] = []

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_init_worker,
                             initargs=(device_queue, threads)) as pool:
        futures = {pool.submit(_call, (worker_fn, kwargs)): i
                   for i, kwargs in enumerate(tasks)}
        done = 0
        for future in _as_completed(futures):
            i = futures[future]
            payload = future.result()
            done += 1
            if payload["ok"]:
                results[i] = payload["result"]
                print(f"[Parallel] {label} {i + 1} finished ({done}/{len(tasks)} complete)",
                      flush=True)
            else:
                failures.append({"index": i, **payload})
                print(f"[Parallel] {label} {i + 1} FAILED: {payload['error']}", flush=True)

    if failures:
        detail = "\n\n".join(
            f"--- {label} index {f['index']} ---\n{f['traceback']}" for f in failures
        )
        raise RuntimeError(
            f"{len(failures)} of {len(tasks)} {label}(s) failed.\n\n{detail}\n"
            f"Stopping rather than aggregating over an incomplete set of folds: "
            f"a 10-fold mean computed from 8 folds is not the reported estimand."
        )

    return results


def _as_completed(futures):
    from concurrent.futures import as_completed
    return as_completed(futures)
