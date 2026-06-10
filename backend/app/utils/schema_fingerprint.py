# ama2/backend/app/utils/schema_fingerprint.py

import hashlib
import pandas as pd

def compute_fingerprint(df: pd.DataFrame) -> str:
    """
    Computes a deterministic SHA-256 hash of the DataFrame's sorted columns and their string dtypes.
    """
    if df is None:
        raise ValueError("DataFrame cannot be None when computing fingerprint")
    
    # Generate sorted representation of col:dtype
    pairs = [f"{col}:{str(df[col].dtype)}" for col in sorted(df.columns)]
    raw_str = ",".join(pairs)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

def get_schema_meta(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """
    Extracts column name lists and dtype strings for storing in database schema tracking.
    """
    columns = {col: str(df[col].dtype) for col in df.columns}
    dtypes = {str(dtype): [col for col in df.columns if df[col].dtype == dtype] 
              for dtype in df.dtypes.unique()}
    return columns, dtypes
