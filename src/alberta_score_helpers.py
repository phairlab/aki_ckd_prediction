"""
Deprecated module name, kept so existing scripts and notebooks keep importing.

The manuscript calls this score the James score (James et al., JAMA 2017);
earlier versions of this codebase called it the Alberta score. The
implementation now lives in `james_score_helpers`, which also carries the
corrected unit handling. Import from there in new code.
"""

import warnings

from james_score_helpers import *          # noqa: F401,F403
from james_score_helpers import (          # noqa: F401
    age_mapping, stage_mapping,
    baseline_creatinine_mapping, discharge_creatinine_mapping,
    albuminuria_status_mapping,
    get_baseline_creatinine, get_discharge_creatinine, get_peak_creatinine,
    get_albuminuria_status,
    creatinine_to_mg_dl, report_unknown_units, UNKNOWN_UNITS_SEEN,
)

warnings.warn(
    "alberta_score_helpers is deprecated; import from james_score_helpers instead.",
    DeprecationWarning, stacklevel=2,
)
