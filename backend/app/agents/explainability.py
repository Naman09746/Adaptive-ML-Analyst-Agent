# ama2/backend/app/agents/explainability.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from ..core.constants import RISK_INFO
from ..core.pipeline_state import PipelineState, RiskFlag
from .base import BaseAgent
from ..ml.explainer import ModelExplainer


class ExplainabilityAgent(BaseAgent):
    """
    Analyzes model transparency and global/local feature importances.
    Detects highly correlated feature clusters and builds a structured, grounded business narrative.
    """

    def __init__(self):
        super().__init__(name="explainability")

    def _execute(self, state: PipelineState) -> PipelineState:
        if state.best_model is None:
            self.logger.warning("no_model_available_for_explainability_skipping")
            return state

        if state.X_train is None or state.X_test is None or state.y_test is None:
            raise ValueError("Training and test datasets must be present for explainability computation.")

        self.logger.info("running_explainability_workflow", model_name=state.best_model_name)

        # 1. Audit highly correlated features (|r| > 0.7)
        numeric_df = state.X_train.select_dtypes(include=["number"])
        correlated_pairs = []

        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            for col in upper_tri.columns:
                for idx in upper_tri.index:
                    r_val = upper_tri.loc[idx, col]
                    if pd.notna(r_val) and r_val > 0.7:
                        correlated_pairs.append((idx, col, float(r_val)))

        if correlated_pairs:
            pair_descriptions = [f"{p[0]} <-> {p[1]} (r={p[2]:.3f})" for p in correlated_pairs[:5]]
            state.risk_flags.append(
                RiskFlag(
                    level=RISK_INFO,
                    code="CORRELATED_FEATURES",
                    feature=None,
                    description=f"Highly correlated features detected: {'; '.join(pair_descriptions)}",
                    recommended_action="SHAP splits importance across correlated features; keep this in mind during interpretation.",
                    requires_human_approval=False
                )
            )

        # 2. Run SHAP computations
        explainer = ModelExplainer(problem_type=state.problem_type or "classification")
        shap_results = explainer.explain(
            model_pipeline=state.best_model,
            X_train=state.X_train,
            X_test=state.X_test,
            y_test=state.y_test
        )

        state.shap_values = shap_results

        # 3. Compile a grounded, rule-based business narrative
        global_imp = shap_results.get("global_importance", [])
        top_features = [item["feature"] for item in global_imp[:3]]
        
        narrative_parts = [
            f"The trained model '{state.best_model_name}' has successfully modeled the target variable '{state.target_column}'.",
            f"Global Feature Analysis: The top features driving the model predictions are {', '.join(top_features)}."
        ]

        local_cases = shap_results.get("local_explanations", {})
        hcc = local_cases.get("highest_confidence_correct")
        fc = local_cases.get("failure_case")

        if hcc or fc:
            narrative_parts.append("\nRepresentative Case Analysis:")
            if hcc:
                # Format to limit feature values shown
                feat_sample = ", ".join(f"{k}={v}" for k, v in list(hcc["features"].items())[:3])
                narrative_parts.append(
                    f"- Correct Case: Sample index {hcc['test_index']} predicted {hcc['predicted']} correctly (Actual: {hcc['actual']}). "
                    f"Sample values: ({feat_sample})."
                )
            if fc:
                feat_sample = ", ".join(f"{k}={v}" for k, v in list(fc["features"].items())[:3])
                narrative_parts.append(
                    f"- Failure Case: Sample index {fc['test_index']} was incorrectly predicted as {fc['predicted']} (Actual: {fc['actual']}). "
                    f"Sample values: ({feat_sample})."
                )

        state.business_narrative = " ".join(narrative_parts)

        self._log_decision(
            state,
            "explainability_narrative",
            {
                "top_features": top_features,
                "correlated_pairs_count": len(correlated_pairs)
            },
            "Generated SHAP explanations and compiled business narrative."
        )

        return state
