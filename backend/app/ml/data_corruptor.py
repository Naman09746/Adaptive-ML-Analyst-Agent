# ama2/backend/app/ml/data_corruptor.py

import numpy as np
import pandas as pd

class DataCorruptor:
    """
    Injects known data quality issues for demo reproducibility, testing pipeline resiliency,
    and triggering automated agent flags/halts.
    """

    def missing_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Injects blank headers, triggering the MISSING_COLUMN_NAMES flag."""
        corrupted = df.copy()
        cols = list(corrupted.columns)
        if cols:
            cols[0] = ""
            corrupted.columns = cols
        return corrupted

    def duplicate_rows(self, df: pd.DataFrame, frac: float = 0.25) -> pd.DataFrame:
        """Duplicates a fraction of rows to trigger the HIGH_DUPLICATE_RATIO flag."""
        corrupted = df.copy()
        n_dups = int(len(df) * frac)
        if n_dups > 0:
            # Sample with replacement
            dups = corrupted.sample(n=n_dups, replace=True, random_state=42)
            corrupted = pd.concat([corrupted, dups], ignore_index=True)
        return corrupted

    def target_leakage_inject(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Injects columns with 100% correlation or label suffixes to trigger LEAKAGE_SUSPECTED."""
        corrupted = df.copy()
        if target_column in corrupted.columns:
            corrupted[f"{target_column}_outcome"] = corrupted[target_column]
            # Also inject numeric duplicate
            if pd.api.types.is_numeric_dtype(corrupted[target_column].dtype):
                corrupted["leakage_val"] = corrupted[target_column] * 2.0
            else:
                corrupted["leakage_val"] = corrupted[target_column]
        return corrupted

    def dtype_change(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Changes numeric column to string object to trigger SCHEMA_DTYPE_MISMATCH."""
        corrupted = df.copy()
        if col in corrupted.columns:
            corrupted[col] = corrupted[col].astype(str)
        return corrupted

    def null_burst(self, df: pd.DataFrame, col: str, frac: float = 0.4) -> pd.DataFrame:
        """Injects a burst of nulls into a column to test PreprocessingAgent's KNN imputer."""
        corrupted = df.copy()
        if col in corrupted.columns:
            n_null = int(len(df) * frac)
            if n_null > 0:
                indices = corrupted.sample(n=n_null, random_state=42).index
                corrupted.loc[indices, col] = np.nan
        return corrupted

    def unseen_categories(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Injects unseen category labels to test PreprocessingAgent's OHE unknown handling."""
        corrupted = df.copy()
        if col in corrupted.columns:
            # Modify first 5 values to a completely new category
            corrupted.loc[corrupted.index[:5], col] = "UNSEEN_CAT_XYZ"
        return corrupted

    def extreme_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Injects extreme numeric outliers to test outlier profiling."""
        corrupted = df.copy()
        if col in corrupted.columns and pd.api.types.is_numeric_dtype(corrupted[col].dtype):
            max_val = corrupted[col].max()
            if pd.isna(max_val) or np.isinf(max_val):
                max_val = 1.0
            # Set top values to 1000x of max_val
            corrupted.loc[corrupted.index[:5], col] = (max_val + 1) * 1000
        return corrupted

    def changed_column_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverses columns to trigger schema fingerprint mismatch."""
        cols = list(df.columns)
        return df[cols[::-1]]
