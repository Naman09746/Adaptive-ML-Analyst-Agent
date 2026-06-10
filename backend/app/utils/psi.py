# ama2/backend/app/utils/psi.py

import numpy as np
import pandas as pd

def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between an expected (baseline) and an actual dataset.
    Handles both numerical and categorical/low-cardinality columns.
    
    PSI rule of thumb:
    - PSI < 0.1: No significant change / stable
    - 0.1 <= PSI < 0.2: Moderate change / warning
    - PSI >= 0.2: Significant shift / drift detected
    """
    # Drop missing values for distribution check
    expected_clean = pd.Series(expected).dropna()
    actual_clean = pd.Series(actual).dropna()
    
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0

    is_numeric = pd.api.types.is_numeric_dtype(expected_clean.dtype)
    
    if is_numeric and expected_clean.nunique() > bins:
        # Numeric binning using percentiles of the expected distribution
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(expected_clean, percentiles)
        bin_edges = np.unique(bin_edges)  # ensure unique boundaries to avoid empty bins
        
        if len(bin_edges) < 2:
            is_numeric = False
        else:
            # Expand boundaries slightly to prevent float boundary exclusions
            bin_edges[0] -= 1e-5
            bin_edges[-1] += 1e-5
            
            expected_counts, _ = np.histogram(expected_clean, bins=bin_edges)
            actual_counts, _ = np.histogram(actual_clean, bins=bin_edges)
            
    if not is_numeric or expected_clean.nunique() <= bins:
        # Categorical or low-cardinality binning: group by unique categories
        all_cats = np.union1d(expected_clean.unique(), actual_clean.unique())
        
        expected_counts = expected_clean.value_counts().reindex(all_cats, fill_value=0).values
        actual_counts = actual_clean.value_counts().reindex(all_cats, fill_value=0).values

    # Convert counts to proportions
    expected_props = expected_counts / len(expected_clean)
    actual_props = actual_counts / len(actual_clean)
    
    # Handle zero proportions by replacing them with epsilon
    eps = 1e-4
    expected_props = np.where(expected_props == 0, eps, expected_props)
    actual_props = np.where(actual_props == 0, eps, actual_props)
    
    # Re-normalize after epsilon injection
    expected_props = expected_props / np.sum(expected_props)
    actual_props = actual_props / np.sum(actual_props)
    
    # Compute PSI formula
    psi_value = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    return float(psi_value)
