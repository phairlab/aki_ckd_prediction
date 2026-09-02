"""
SUPERSEDED -- pre-refactor duplicate, kept only for provenance.

This module predates the refactor into config.py / run_pipeline.py / src/{models,
analysis,evaluation}. It is not on the analysis path and is not maintained. The
live equivalents are:

    get_results_xgboost.py        -> src/cross_validation.py + src/models/xgboost_model.py
    get_transformer_results.py    -> src/cross_validation.py + src/models/transformer_*.py
    transformer_model_helpers.py  -> src/models/transformer_model.py
    shap_analysis.py              -> src/analysis/shap_analysis.py (plotting) and
                                     src/cross_validation.py::_fold_shap (out-of-fold values)

Do not use for the resubmission: none of these carry the fold-local imputation,
the nested hyperparameter search, the stratified split, or the out-of-fold SHAP.
"""

"""
Shared SHAP Analysis Module for AKI-CKD Prediction Models

This module provides standardized SHAP analysis functions for both XGBoost and
transformer models used in the AKI-CKD prediction project.

Functions:
- perform_shap_analysis_tree: For tree-based models (XGBoost, Random Forest, etc.)
- perform_shap_analysis_kernel: For any model using KernelExplainer (transformers, neural nets)
- setup_plot_style: Configure matplotlib for publication-quality figures
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap


def setup_plot_style():
    """Set up matplotlib style to match paper figures."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })


def get_custom_colormap():
    """Create custom colormap matching paper style (blue to orange)."""
    colors_custom = ['#1f77b4', '#f0f0f0', '#ff7f0e']  # Blue -> Light gray -> Orange
    return LinearSegmentedColormap.from_list('paper_style', colors_custom)


# Feature name mapping for cleaner display in plots
FEATURE_NAME_MAPPING = {
    # Creatinine features
    'discharge_creatinine_raw': 'Discharge Creatinine',
    'baseline_creatinine_raw': 'Baseline Creatinine',
    'discharge_creatinine_points': 'Discharge Creatinine (points)',
    'baseline_creatinine_points': 'Baseline Creatinine (points)',
    
    # Sex (0 = Male, 1 = Female)
    'sex_0': 'Sex: Male',
    'sex_1': 'Sex: Female',
    'sex': 'Sex',
    'sex_points': 'Sex (points)',
    
    # Age
    'age_admit': 'Age at Admission',
    'age_admit_points': 'Age (points)',
    
    # Albuminuria status
    'albuminuria_status_raw_unmeasured': 'Albuminuria: Unmeasured',
    'albuminuria_status_raw_mild': 'Albuminuria: Mild',
    'albuminuria_status_raw_normal': 'Albuminuria: Normal',
    'albuminuria_status_raw_heavy': 'Albuminuria: Heavy',
    'albuminuria_status_raw': 'Albuminuria Status',
    'albuminuria_status_points': 'Albuminuria (points)',
    
    # AKI Stage
    'highest_stage': 'Highest AKI Stage',
    'highest_stage_1': 'AKI Stage 1',
    'highest_stage_2': 'AKI Stage 2',
    'highest_stage_3': 'AKI Stage 3',
    'highest_stage_points': 'AKI Stage (points)',
    
    # Alberta score
    'alberta_score': 'Alberta Score',
}


def clean_feature_names(feature_names):
    """
    Apply the feature name mapping to make names more readable.
    
    Parameters:
    -----------
    feature_names : array-like
        Original feature names
    
    Returns:
    --------
    list
        Cleaned feature names
    """
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    
    cleaned = []
    for name in feature_names:
        # Use mapping if available, otherwise keep original
        cleaned_name = FEATURE_NAME_MAPPING.get(name, name)
        cleaned.append(cleaned_name)
    
    return cleaned


def perform_shap_analysis_tree(classifier, X_data, feature_names, output_dir, max_display=20,
                                beeswarm_title='SHAP Feature Impact', bar_title='SHAP Feature Importance'):
    """
    Perform SHAP analysis on a tree-based classifier (XGBoost, Random Forest, etc.)
    using TreeExplainer for fast, exact SHAP values.
    
    Parameters:
    -----------
    classifier : trained model
        The trained tree-based classifier (XGBoost, LightGBM, Random Forest, etc.)
    X_data : np.ndarray or pd.DataFrame
        Feature data to explain
    feature_names : array-like
        Names of the features
    output_dir : str
        Directory to save the SHAP plots and values
    max_display : int, default=20
        Maximum number of features to display (rest are summarized)
    beeswarm_title : str, default='SHAP Feature Impact'
        Title for the beeswarm plot
    bar_title : str, default='SHAP Feature Importance'
        Title for the bar plot
    
    Returns:
    --------
    shap_values : shap.Explanation
        The computed SHAP values
    """
    print("Performing SHAP analysis (TreeExplainer)...")
    
    # Set up matplotlib style
    setup_plot_style()
    
    # Clean feature names for display
    cleaned_names = clean_feature_names(feature_names)
    
    # Create a DataFrame for SHAP analysis with cleaned names
    shap_df = pd.DataFrame(X_data, columns=cleaned_names)
    
    # Initialize the SHAP explainer and compute values
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(shap_df)
    
    # Get custom colormap
    cmap_custom = get_custom_colormap()
    
    # Generate and save the beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(
        shap_values, 
        max_display=max_display,
        color=cmap_custom,
        show=False,
        plot_size=None  # Let matplotlib handle sizing
    )
    
    # Adjust plot styling
    ax = plt.gca()
    ax.set_title(beeswarm_title, fontweight='bold', fontsize=14)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/shap_beeswarm.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Also create a bar plot for mean absolute SHAP values
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(
        shap_values,
        max_display=max_display,
        show=False
    )
    
    ax = plt.gca()
    ax.set_title(bar_title, fontweight='bold', fontsize=14)
    ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/shap_bar.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Save SHAP values for future analysis
    with open(f"{output_dir}/shap_values.pkl", 'wb') as f:
        pickle.dump(shap_values, f)
    
    print(f"SHAP analysis completed and plots saved to {output_dir}/")
    
    return shap_values


def perform_shap_analysis_kernel(model_predict_fn, X_data, feature_names, output_dir,
                                  background_samples=100, samples_to_explain=100,
                                  max_display=20, verbose=True,
                                  beeswarm_title='SHAP Feature Impact', bar_title='SHAP Feature Importance'):
    """
    Perform SHAP analysis using KernelExplainer for any model type.
    This is suitable for neural networks, transformers, and other non-tree models.
    
    Parameters:
    -----------
    model_predict_fn : callable
        A function that takes X_data and returns predicted probabilities.
        Should return shape (n_samples, n_classes) for classification.
    X_data : np.ndarray
        Feature data to explain (scaled/preprocessed as needed by the model)
    feature_names : array-like
        Names of the features
    output_dir : str
        Directory to save the SHAP plots and values
    background_samples : int, default=100
        Number of background samples to use for the explainer
    samples_to_explain : int, default=100
        Number of samples to compute SHAP values for
    max_display : int, default=20
        Maximum number of features to display (rest are summarized)
    verbose : bool, default=True
        Whether to print debugging information
    beeswarm_title : str, default='SHAP Feature Impact'
        Title for the beeswarm plot
    bar_title : str, default='SHAP Feature Importance'
        Title for the bar plot
    
    Returns:
    --------
    shap_values : np.ndarray
        The computed SHAP values
    """
    print("Performing SHAP analysis (KernelExplainer)...")
    
    # Set up matplotlib style
    setup_plot_style()
    
    # Create background dataset
    background = X_data[:background_samples]
    sample_to_explain = X_data[:samples_to_explain]
    
    if verbose:
        print(f"Background shape: {background.shape}")
        print(f"Sample to explain shape: {sample_to_explain.shape}")
        print(f"Feature count: {len(feature_names)}")
        
        # Test model_predict function
        print("Testing model_predict function...")
        try:
            test_output = model_predict_fn(sample_to_explain[:1])
            print(f"Model output shape: {test_output.shape}")
            print(f"Model output sample: {test_output}")
        except Exception as e:
            print(f"Error during model_predict test: {e}")
            raise
    
    # Initialize the SHAP explainer
    print("Initializing KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict_fn, background)
    
    print("Computing SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(sample_to_explain)
    
    # Clean feature names for display
    feature_names_list = clean_feature_names(feature_names)
    
    # Create a SHAP summary plot (beeswarm-style for positive class)
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values[1],  # SHAP values for positive class
        sample_to_explain, 
        feature_names=feature_names_list,
        max_display=max_display,
        show=False
    )
    ax = plt.gca()
    ax.set_title(beeswarm_title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Create a SHAP bar plot of feature importance
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values[1],
        sample_to_explain, 
        feature_names=feature_names_list,
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    ax = plt.gca()
    ax.set_title(bar_title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_bar.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Save SHAP values for future analysis
    with open(f"{output_dir}/shap_values.pkl", 'wb') as f:
        pickle.dump(shap_values, f)
    
    # Save the X_data that was explained (needed for regenerating plots)
    with open(f"{output_dir}/shap_X_data.pkl", 'wb') as f:
        pickle.dump(sample_to_explain, f)
    
    # Save feature names list
    with open(f"{output_dir}/shap_feature_names.pkl", 'wb') as f:
        pickle.dump(feature_names_list, f)
    
    print(f"SHAP analysis completed and plots saved to {output_dir}/")
    
    return shap_values


def regenerate_shap_plots_tree(shap_values, feature_names, output_dir, max_display=20,
                                beeswarm_title='SHAP Feature Impact', bar_title='SHAP Feature Importance'):
    """
    Regenerate SHAP plots from previously computed SHAP values (tree-based models).
    
    Parameters:
    -----------
    shap_values : shap.Explanation
        Previously computed SHAP values (loaded from shap_values.pkl)
    feature_names : array-like
        Names of the features
    output_dir : str
        Directory to save the regenerated plots
    max_display : int, default=20
        Maximum number of features to display
    beeswarm_title : str, default='SHAP Feature Impact'
        Title for the beeswarm plot
    bar_title : str, default='SHAP Feature Importance'
        Title for the bar plot
    
    Returns:
    --------
    None
    """
    print("Regenerating SHAP plots from saved values...")
    
    # Set up matplotlib style
    setup_plot_style()
    
    # Clean feature names embedded in shap_values
    if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
        shap_values.feature_names = clean_feature_names(shap_values.feature_names)
    
    # Get custom colormap
    cmap_custom = get_custom_colormap()
    
    # Generate and save the beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(
        shap_values, 
        max_display=max_display,
        color=cmap_custom,
        show=False,
        plot_size=None
    )
    
    ax = plt.gca()
    ax.set_title(beeswarm_title, fontweight='bold', fontsize=14)
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/shap_beeswarm.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Create bar plot for mean absolute SHAP values
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(
        shap_values,
        max_display=max_display,
        show=False
    )
    
    ax = plt.gca()
    ax.set_title(bar_title, fontweight='bold', fontsize=14)
    ax.set_xlabel('mean(|SHAP value|)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/shap_bar.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"SHAP plots regenerated and saved to {output_dir}/")


def regenerate_shap_plots_kernel(shap_values, X_data, feature_names, output_dir, max_display=20,
                                  beeswarm_title='SHAP Feature Impact', bar_title='SHAP Feature Importance'):
    """
    Regenerate SHAP plots from previously computed SHAP values (kernel explainer).
    
    Parameters:
    -----------
    shap_values : np.ndarray
        Previously computed SHAP values (loaded from shap_values.pkl)
    X_data : np.ndarray
        Feature data that was explained (needed for summary plot)
    feature_names : array-like
        Names of the features
    output_dir : str
        Directory to save the regenerated plots
    max_display : int, default=20
        Maximum number of features to display
    beeswarm_title : str, default='SHAP Feature Impact'
        Title for the beeswarm plot
    bar_title : str, default='SHAP Feature Importance'
        Title for the bar plot
    
    Returns:
    --------
    None
    """
    print("Regenerating SHAP plots from saved values...")
    
    # Set up matplotlib style
    setup_plot_style()
    
    # Clean feature names for display
    feature_names_list = clean_feature_names(feature_names)
    
    # Create a SHAP summary plot (beeswarm-style for positive class)
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values[1],  # SHAP values for positive class
        X_data, 
        feature_names=feature_names_list,
        max_display=max_display,
        show=False
    )
    ax = plt.gca()
    ax.set_title(beeswarm_title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    # Create a SHAP bar plot of feature importance
    fig = plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values[1],
        X_data, 
        feature_names=feature_names_list,
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    ax = plt.gca()
    ax.set_title(bar_title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_bar.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"SHAP plots regenerated and saved to {output_dir}/")


def load_shap_values(filepath):
    """
    Load previously computed SHAP values from a pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to the shap_values.pkl file
    
    Returns:
    --------
    shap_values : shap.Explanation or np.ndarray
        The loaded SHAP values
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
