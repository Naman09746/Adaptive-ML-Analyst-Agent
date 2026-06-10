# ama2/backend/app/agents/preprocessing.py

from __future__ import annotations

import inspect
from typing import Any
import numpy as np
import pandas as pd

from ..core.pipeline_state import PipelineState
from .base import BaseAgent
from ..core.constants import RANDOM_SEED
from ..ml.data_inspection import load_dataset, profile_dataframe

try:
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    SimpleImputer = KNNImputer = OneHotEncoder = StandardScaler = FunctionTransformer = ColumnTransformer = Pipeline = None
    train_test_split = None
    HAS_SKLEARN = False


class PreprocessingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="preprocessing")

    def _execute(self, state: PipelineState) -> PipelineState:
        if state.df is None:
            try:
                state.df = load_dataset(state.dataset_path)
            except Exception as e:
                self._log_decision(state, "preprocessing", False, f"unable to load dataset: {e}")
                return state

        profile = profile_dataframe(state.df)
        state.data_profile.update(profile)

        if not HAS_SKLEARN:
            self._log_decision(
                state,
                "preprocessing_available",
                False,
                "scikit-learn is not installed; skipping pipeline build"
            )
            return state

        # 1. Detect ID-like columns
        n_rows = len(state.df)
        id_cols = []
        if n_rows > 0:
            for col in state.df.columns:
                if col != state.target_column:
                    nunique = state.df[col].nunique(dropna=True)
                    if nunique / n_rows > 0.95 and not pd.api.types.is_float_dtype(state.df[col].dtype):
                        id_cols.append(col)

        # 2. Drop leakage suspects and ID columns from predictor set
        dropped_cols = sorted(list(set(state.leakage_suspects + id_cols)))
        if dropped_cols:
            self._log_decision(
                state,
                "dropped_columns",
                dropped_cols,
                f"Dropped columns due to leakage or high cardinality/ID patterns: {dropped_cols}"
            )

        features_df = state.df.drop(columns=[state.target_column] + dropped_cols, errors="ignore")
        target_series = state.df[state.target_column]

        # 3. Perform Train-Test Split (test_size=0.2, stratify if classification)
        is_classification = (state.problem_type == "classification")
        stratify = target_series if is_classification else None
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, target_series, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
            )
        except Exception:
            # Fallback if stratification is impossible due to class count variance
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, target_series, test_size=0.2, random_state=RANDOM_SEED, stratify=None
            )

        state.X_train = X_train
        state.X_test = X_test
        state.y_train = y_train
        state.y_test = y_test

        # Identify types of columns in the feature training set
        numeric_cols = [c for c in X_train.select_dtypes(include=["number"]).columns]
        categorical_cols = [c for c in X_train.select_dtypes(exclude=["number"]).columns]

        plan: dict[str, Any] = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "imputation": {},
            "scaling": {},
            "encoding": "onehot",
            "remainder": "drop",
        }

        # Safe parameter lookup for OneHotEncoder (sklearn compatibility helper)
        ohe_params = {"handle_unknown": "ignore"}
        if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
            ohe_params["sparse_output"] = False
        else:
            ohe_params["sparse"] = False

        transformers = []

        # 4. Build numerical pipelines with advanced rules
        for col in numeric_cols:
            missing_rate = X_train[col].isna().mean()
            
            # Rule: Missing between 5% and 30% -> KNNImputer + indicator flag
            if 0.05 < missing_rate <= 0.30:
                imputer = KNNImputer(n_neighbors=5, add_indicator=True)
                imputer_name = "knn_impute"
                plan["imputation"][col] = "knn_imputer_5"
            else:
                add_ind = missing_rate > 0
                imputer = SimpleImputer(strategy="median", add_indicator=add_ind)
                imputer_name = "median_impute"
                plan["imputation"][col] = "median" if not add_ind else "median_with_indicator"

            steps: list[tuple[str, Any]] = [(imputer_name, imputer)]

            # Rule: High skewness (|skew| > 1.5) and minimum >= 0 -> log1p scaling
            is_skewed = False
            if X_train[col].min() >= 0:
                skew = X_train[col].skew()
                if pd.notna(skew) and abs(skew) > 1.5:
                    steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))
                    is_skewed = True
                    plan["scaling"][col] = "log1p_and_standard"

            if not is_skewed:
                plan["scaling"][col] = "standard"

            steps.append(("scale", StandardScaler()))
            col_pipeline = Pipeline(steps)
            transformers.append((f"num_{col}", col_pipeline, [col]))

        # 5. Build categorical pipelines
        for col in categorical_cols:
            cat_pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(**ohe_params)),
            ])
            transformers.append((f"cat_{col}", cat_pipeline, [col]))

        # Construct final ColumnTransformer
        preprocessor = ColumnTransformer(transformers, remainder="drop")
        state.sklearn_pipeline = preprocessor
        state.preprocessing_plan = plan

        self._log_decision(
            state,
            "preprocessing_plan",
            plan,
            "Constructed advanced sklearn ColumnTransformer and split training/testing sets."
        )

        return state

