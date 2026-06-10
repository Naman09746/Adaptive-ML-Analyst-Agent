# ama2/backend/app/agents/model_strategy.py

from __future__ import annotations

from typing import Any
import pandas as pd
from ..core.pipeline_state import PipelineState
from .base import BaseAgent

from ..ml.model_registry import ModelRegistry
from ..ml.data_inspection import profile_dataframe
from ..core.constants import RANDOM_SEED

try:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    HAS_SKLEARN_DUMMY = True
except ImportError:
    DummyClassifier = None
    DummyRegressor = None
    HAS_SKLEARN_DUMMY = False


class ModelStrategyAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="model_strategy")

    def _execute(self, state: PipelineState) -> PipelineState:
        if state.df is None:
            self._log_decision(state, "model_candidates", [], "No dataframe available to generate model candidates")
            return state

        profile = profile_dataframe(state.df)
        n_samples = profile.get("shape", {}).get("rows", 0)
        problem_type = state.problem_type or "classification"

        # Determine target type (classification or regression)
        is_classification = (problem_type == "classification")
        if problem_type == "timeseries" and state.target_column in state.df.columns:
            target_series = state.df[state.target_column]
            if pd.api.types.is_numeric_dtype(target_series):
                if target_series.nunique(dropna=True) <= max(20, int(len(state.df) * 0.05)):
                    is_classification = True
                else:
                    is_classification = False
            else:
                is_classification = True

        candidates: list[dict[str, Any]] = []

        # 1. Add dual dummy baselines
        if HAS_SKLEARN_DUMMY:
            if is_classification:
                candidates.extend([
                    {
                        "name": "dummy_most_frequent",
                        "estimator": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
                        "param_grid": {},
                        "tier": 0,
                    },
                    {
                        "name": "dummy_stratified",
                        "estimator": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
                        "param_grid": {},
                        "tier": 0,
                    }
                ])
            else:
                candidates.extend([
                    {
                        "name": "dummy_mean",
                        "estimator": DummyRegressor(strategy="mean"),
                        "param_grid": {},
                        "tier": 0,
                    },
                    {
                        "name": "dummy_median",
                        "estimator": DummyRegressor(strategy="median"),
                        "param_grid": {},
                        "tier": 0,
                    }
                ])

        # Check for class imbalance > 10:1
        is_imbalanced = False
        if is_classification and state.target_column in state.df.columns:
            counts = state.df[state.target_column].value_counts()
            if len(counts) >= 2:
                ratio = counts.iloc[0] / counts.iloc[-1]
                if ratio > 10:
                    is_imbalanced = True

        # 2. Add eligible models from registry
        for name in ModelRegistry.available_names():
            # Hard rule: n < 100 -> cap at logistic/linear only
            if n_samples < 100 and name != "logistic":
                continue

            strategy = ModelRegistry._registry.get(name)
            if strategy is None:
                continue

            if n_samples < getattr(strategy, "min_samples", 0):
                continue

            try:
                estimator = strategy.build(problem_type)
            except Exception as e:
                self._log_decision(state, f"model_strategy_skip_{name}", False, f"could not build estimator: {e}")
                continue

            # Enforce class weight if imbalanced
            if is_imbalanced and hasattr(estimator, "class_weight"):
                estimator.set_params(class_weight="balanced")

            candidates.append({
                "name": name,
                "estimator": estimator,
                "param_grid": strategy.get_param_grid(),
                "tier": getattr(strategy, "tier", 99),
            })

        # sort by tier
        candidates = sorted(candidates, key=lambda c: c.get("tier", 99))

        state.model_candidates = candidates
        self._log_decision(state, "model_candidates", [c["name"] for c in candidates], "Selected model strategies based on dataset size and problem type")

        return state

