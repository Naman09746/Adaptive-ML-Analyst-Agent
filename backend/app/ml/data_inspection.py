# ama2/backend/app/ml/data_inspection.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:  # pragma: no cover - optional dependency fallback
    pd = None
    HAS_PANDAS = False

from ..core.constants import (
    CLASS_IMBALANCE,
    CONSTANT_COLUMN,
    HIGH_DUPLICATE_RATIO,
    HIGH_MISSING_RATE,
    MISSING_COLUMN_NAMES,
    PROBLEM_CLASSIFICATION,
    PROBLEM_REGRESSION,
    PROBLEM_TIMESERIES,
    TINY_DATASET,
)

EMPTY_COLUMN_PATTERN = re.compile(r"^\s*$")
LEAKY_NAME_PATTERN = re.compile(r"(_label$|_target$|^actual_|^final_|leak|outcome)", re.IGNORECASE)
GROUP_COLUMN_PATTERN = re.compile(r"(^|_)(user|account|customer|client|store|session|group|id)$", re.IGNORECASE)
TIMESERIES_PATTERN = re.compile(r"(date|time|timestamp|event_time|created_at|updated_at)", re.IGNORECASE)
TARGET_NAME_PATTERN = re.compile(r"(target|label|outcome|response|class)", re.IGNORECASE)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    if not HAS_PANDAS:
        raise ImportError("pandas is required to load datasets")

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    raise ValueError(f"Unsupported dataset format: {suffix or '<no extension>'}")


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    if not HAS_PANDAS:
        raise ImportError("pandas is required to profile datasets")

    missing_rates = df.isna().mean().sort_values(ascending=False)
    duplicate_ratio = float(df.duplicated().mean()) if len(df) else 0.0

    constant_cols = [column for column in df.columns if df[column].nunique(dropna=False) <= 1]
    mixed_dtype_cols = [
        column
        for column in df.columns
        if df[column].dtype == "object" and df[column].dropna().map(type).nunique() > 1
    ]

    outlier_scores: dict[str, float] = {}
    for column in df.select_dtypes(include=["number"]).columns:
        series = df[column].dropna()
        if series.empty or series.nunique() < 4:
            outlier_scores[column] = 0.0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_scores[column] = 0.0
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_scores[column] = float(((series < lower_bound) | (series > upper_bound)).mean())

    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_rates": {column: float(rate) for column, rate in missing_rates.items()},
        "duplicate_ratio": duplicate_ratio,
        "constant_cols": constant_cols,
        "mixed_dtype_cols": mixed_dtype_cols,
        "outlier_scores": outlier_scores,
        "numeric_columns": list(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category", "bool"]).columns),
    }


def infer_target_column(df: pd.DataFrame, problem_statement: str | None = None) -> str | None:
    if not HAS_PANDAS:
        raise ImportError("pandas is required to infer a target column")

    if not len(df.columns):
        return None

    statement = (problem_statement or "").lower()
    if any(keyword in statement for keyword in ("target", "label", "predict", "forecast")):
        for column in df.columns:
            if TARGET_NAME_PATTERN.search(str(column)):
                return str(column)

    for column in df.columns:
        if TARGET_NAME_PATTERN.search(str(column)):
            return str(column)

    return str(df.columns[-1])


def infer_problem_type(df: pd.DataFrame, target_column: str | None, problem_statement: str | None = None) -> str:
    if not HAS_PANDAS:
        raise ImportError("pandas is required to infer problem type")

    statement = (problem_statement or "").lower()
    if any(keyword in statement for keyword in ("forecast", "time series", "timeseries", "sequence")):
        return PROBLEM_TIMESERIES

    if target_column and target_column in df.columns:
        target_series = df[target_column]
        if pd.api.types.is_datetime64_any_dtype(target_series):
            return PROBLEM_TIMESERIES
        if pd.api.types.is_numeric_dtype(target_series):
            distinct_values = target_series.nunique(dropna=True)
            if distinct_values <= max(20, int(len(df) * 0.05)):
                return PROBLEM_CLASSIFICATION
            return PROBLEM_REGRESSION
        return PROBLEM_CLASSIFICATION

    return PROBLEM_CLASSIFICATION


def detect_group_column(df: pd.DataFrame) -> str | None:
    if not HAS_PANDAS:
        raise ImportError("pandas is required to detect group columns")

    for column in df.columns:
        if GROUP_COLUMN_PATTERN.search(str(column)):
            return str(column)
    return None


def detect_timeseries_column(df: pd.DataFrame) -> str | None:
    if not HAS_PANDAS:
        raise ImportError("pandas is required to detect timeseries columns")

    for column in df.columns:
        if TIMESERIES_PATTERN.search(str(column)) or pd.api.types.is_datetime64_any_dtype(df[column]):
            return str(column)
    return None


def detect_leakage_suspects(df: pd.DataFrame, target_column: str | None) -> list[str]:
    if not HAS_PANDAS:
        raise ImportError("pandas is required to detect leakage")

    suspects: list[str] = []
    target_series = df[target_column] if target_column and target_column in df.columns else None

    for column in df.columns:
        if EMPTY_COLUMN_PATTERN.match(str(column)):
            suspects.append(str(column))
            continue

        if LEAKY_NAME_PATTERN.search(str(column)):
            suspects.append(str(column))
            continue

        if target_series is not None and column != target_column:
            candidate = df[column]
            if pd.api.types.is_numeric_dtype(candidate) and pd.api.types.is_numeric_dtype(target_series):
                aligned = pd.concat([candidate, target_series], axis=1).dropna()
                if len(aligned) >= 3:
                    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    if pd.notna(correlation) and abs(correlation) > 0.95:
                        suspects.append(str(column))
                        continue

            feature_values = df[column].fillna("__missing__").astype(str)
            target_values = target_series.fillna("__missing__").astype(str)
            if len(feature_values) > 0 and (feature_values == target_values).mean() > 0.99:
                suspects.append(str(column))

    return sorted(set(suspects))


def infer_cv_strategy(problem_type: str, group_column: str | None, timeseries_column: str | None) -> str:
    if group_column:
        return "GroupKFold"
    if problem_type == PROBLEM_TIMESERIES or timeseries_column:
        return "TimeSeriesSplit"
    if problem_type == PROBLEM_REGRESSION:
        return "KFold"
    return "StratifiedKFold"


def detect_cv_strategy(problem_type: str, group_column: str | None, timeseries_column: str | None) -> str:
    return infer_cv_strategy(problem_type, group_column, timeseries_column)


def build_risk_signals(df: pd.DataFrame, target_column: str | None) -> list[dict[str, Any]]:
    if not HAS_PANDAS:
        raise ImportError("pandas is required to build risk signals")

    profile = profile_dataframe(df)
    risk_signals: list[dict[str, Any]] = []

    if any(not str(column).strip() for column in df.columns):
        risk_signals.append(
            {"level": "critical", "code": MISSING_COLUMN_NAMES, "feature": None, "reason": "One or more column headers are blank."}
        )

    if profile["duplicate_ratio"] > 0.2:
        risk_signals.append(
            {"level": "warning", "code": HIGH_DUPLICATE_RATIO, "feature": None, "reason": "Duplicate rows exceed the safe threshold."}
        )

    if len(df) < 100:
        risk_signals.append(
            {"level": "warning", "code": TINY_DATASET, "feature": None, "reason": "Dataset is too small for high-confidence automation."}
        )

    for column, rate in profile["missing_rates"].items():
        if rate > 0.5:
            risk_signals.append(
                {"level": "warning", "code": HIGH_MISSING_RATE, "feature": column, "reason": "Column has more than 50% missing values."}
            )

    for column in profile["constant_cols"]:
        risk_signals.append(
            {"level": "info", "code": CONSTANT_COLUMN, "feature": column, "reason": "Column has no useful variation."}
        )

    if target_column and target_column in df.columns and len(df) < 100:
        risk_signals.append(
            {"level": "warning", "code": CLASS_IMBALANCE, "feature": target_column, "reason": "Tiny datasets require explicit review of class balance."}
        )

    return risk_signals
