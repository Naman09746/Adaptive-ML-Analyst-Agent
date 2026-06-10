# ama2/backend/app/agents/risk_failure.py

from __future__ import annotations

import pandas as pd
from ..core.constants import (
    TINY_DATASET,
    LEAKAGE_SUSPECTED,
    RISK_CRITICAL,
    RISK_WARNING,
    RISK_INFO,
    CONSTANT_COLUMN,
    HIGH_MISSING_RATE,
    SUSPICIOUS_AUC,
    DOMINANT_FEATURE,
    UNSTABLE_CV,
    DRIFT_DETECTED,
    SCHEMA_DTYPE_MISMATCH,
)
from ..core.pipeline_state import PipelineState, RiskFlag
from .base import BaseAgent
from ..utils.schema_fingerprint import compute_fingerprint
from ..utils.psi import compute_psi


class RiskFailureAgent(BaseAgent):
    """
    Evaluates system safety at three distinct checkpoints:
    1. Pre-training: checks shape, leakage, target variation, and general missing rates.
    2. Post-training: checks baseline improvements, CV stability, suspicious AUC, and dominant features.
    3. Schema drift: checks data schema mismatches and distribution shifts (PSI) against a reference run.
    """

    def __init__(self):
        super().__init__("risk_failure")

    def _execute(self, state: PipelineState) -> PipelineState:
        """Runs the pre-training or post-training risk check depending on pipeline stage."""
        if state.best_model is not None or state.eval_metrics:
            self.logger.info("running_post_training_risk_checks")
            state = self._post_training_check(state)
        else:
            self.logger.info("running_pre_training_risk_checks")
            state = self._pre_training_check(state)
        return state

    def _pre_training_check(self, state: PipelineState) -> PipelineState:
        """Evaluates datasets and problem parameters before training begins."""
        # 1. Dataset size check
        if state.df is not None and len(state.df) < 100:
            if "tiny_dataset_proceed" not in state.human_approvals:
                self.logger.warning("tiny_dataset_detected", rows=len(state.df))
                state.risk_flags.append(
                    RiskFlag(
                        level=RISK_CRITICAL,
                        code=TINY_DATASET,
                        feature=None,
                        description="Dataset too small for reliable training (n < 100).",
                        recommended_action="Provide more samples or confirm proceeding at your own risk.",
                        requires_human_approval=True
                    )
                )

        # 2. Target leakage features check
        if state.leakage_suspects:
            if "leakage_feature_drop" not in state.human_approvals:
                self.logger.warning("target_leakage_suspects_found", suspects=state.leakage_suspects)
                state.risk_flags.append(
                    RiskFlag(
                        level=RISK_CRITICAL,
                        code=LEAKAGE_SUSPECTED,
                        feature=", ".join(state.leakage_suspects),
                        description=f"Potential target leakage features detected: {state.leakage_suspects}",
                        recommended_action="Approve dropping leakage columns to ensure a valid model.",
                        requires_human_approval=True
                    )
                )

        # 3. High missing rates check (all features have > 50% missing values)
        missing_rates = state.data_profile.get("missing_rates", {})
        feature_missing_rates = {col: rate for col, rate in missing_rates.items() if col != state.target_column}
        if feature_missing_rates and all(rate > 0.5 for rate in feature_missing_rates.values()):
            self.logger.error("all_features_mostly_missing")
            state.halt = True
            state.halt_reason = "All features have > 50% missing values."
            state.risk_flags.append(
                RiskFlag(
                    level=RISK_CRITICAL,
                    code=HIGH_MISSING_RATE,
                    feature=None,
                    description="All features have missing rates above 50%. Pipeline aborted.",
                    recommended_action="Re-upload a dataset containing populated predictor columns.",
                    requires_human_approval=False
                )
            )

        # 4. Target variation check
        if state.df is not None and state.target_column is not None and state.target_column in state.df.columns:
            target_series = state.df[state.target_column].dropna()
            if target_series.nunique() <= 1:
                self.logger.error("constant_target_detected", column=state.target_column)
                state.halt = True
                state.halt_reason = "Target column has no learnable variation."
                state.risk_flags.append(
                    RiskFlag(
                        level=RISK_CRITICAL,
                        code=CONSTANT_COLUMN,
                        feature=state.target_column,
                        description="Target column has 1 or 0 unique values. Training is impossible.",
                        recommended_action="Re-upload dataset with varied target labels.",
                        requires_human_approval=False
                    )
                )

        return state

    def _post_training_check(self, state: PipelineState) -> PipelineState:
        """Evaluates final model statistics and logs risk parameters after evaluation."""
        metrics = state.eval_metrics

        # 1. Improved baseline gate
        if not metrics.get("beats_dummy_baseline", True):
            self.logger.error("model_failed_baseline_improvement")
            state.halt = True
            state.halt_reason = "Model does not beat dummy baseline by the required 5% relative threshold."

        # 2. Unstable cross-validation check
        cv_std = metrics.get("cv_std", 0.0)
        if cv_std > 0.15:
            self.logger.warning("unstable_cv_score", cv_std=cv_std)
            state.risk_flags.append(
                RiskFlag(
                    level=RISK_WARNING,
                    code=UNSTABLE_CV,
                    feature=None,
                    description=f"Model cross-validation is highly unstable (std dev = {cv_std:.4f} > 0.15).",
                    recommended_action="Ensure the training data partition is clean or choose a simpler algorithm.",
                    requires_human_approval=False
                )
            )

        # 3. Suspiciously high score (e.g. perfect AUC indicative of leakage)
        roc_auc = metrics.get("roc_auc", 0.0)
        # Apply checks if it's classification (for regression, ROC-AUC might be 0.0 or missing)
        if roc_auc > 0.99:
            if "suspicious_auc" not in state.human_approvals:
                self.logger.warning("suspiciously_high_auc", auc=roc_auc)
                state.risk_flags.append(
                    RiskFlag(
                        level=RISK_CRITICAL,
                        code=SUSPICIOUS_AUC,
                        feature=None,
                        description=f"Model reached a suspicious ROC-AUC score ({roc_auc:.4f} > 0.99). Possibility of target leakage.",
                        recommended_action="Confirm that this performance is not caused by target leakage.",
                        requires_human_approval=True
                    )
                )

        # 4. Dominant feature check
        dominant_feature_ratio = metrics.get("dominant_feature_ratio", 0.0)
        if dominant_feature_ratio > 0.5:
            feature_name = metrics.get("dominant_feature_name", "Unknown")
            self.logger.warning("dominant_feature_detected", feature=feature_name, ratio=dominant_feature_ratio)
            state.risk_flags.append(
                RiskFlag(
                    level=RISK_WARNING,
                    code=DOMINANT_FEATURE,
                    feature=feature_name,
                    description=f"Single feature '{feature_name}' represents {dominant_feature_ratio:.2%} of model importance.",
                    recommended_action="Verify this feature is valid and not a proxy for the target column.",
                    requires_human_approval=False
                )
            )

        return state

    def _schema_drift_check(self, state: PipelineState, baseline_df: pd.DataFrame) -> PipelineState:
        """
        Runs a comprehensive schema & distribution drift audit of state.df against a baseline dataset.
        Appends risk flags representing schema dtype mismatches or columns added/removed.
        """
        if state.df is None or baseline_df is None:
            return state

        baseline_fp = compute_fingerprint(baseline_df)
        current_fp = compute_fingerprint(state.df)

        if baseline_fp != current_fp:
            self.logger.warning("schema_fingerprint_mismatch", baseline=baseline_fp, current=current_fp)
            baseline_cols = set(baseline_df.columns)
            current_cols = set(state.df.columns)

            new_cols = current_cols - baseline_cols
            deleted_cols = baseline_cols - current_cols
            common_cols = baseline_cols & current_cols

            dtype_mismatches = []
            for col in common_cols:
                b_dtype = baseline_df[col].dtype
                c_dtype = state.df[col].dtype
                if b_dtype != c_dtype:
                    # Enforce critical flag on numeric -> categorical shifts
                    if pd.api.types.is_numeric_dtype(b_dtype) and not pd.api.types.is_numeric_dtype(c_dtype):
                        dtype_mismatches.append(f"{col} changed from {b_dtype} (numeric) to {c_dtype} (categorical)")
                    else:
                        dtype_mismatches.append(f"{col} dtype changed from {b_dtype} to {c_dtype}")

            if dtype_mismatches:
                if "schema_drift_retrain" not in state.human_approvals:
                    state.risk_flags.append(
                        RiskFlag(
                            level=RISK_CRITICAL,
                            code=SCHEMA_DTYPE_MISMATCH,
                            feature=None,
                            description=f"Schema data type mismatch: {'; '.join(dtype_mismatches)}",
                            recommended_action="Approve schema update/retraining with the new data types.",
                            requires_human_approval=True
                        )
                    )
            else:
                desc_parts = []
                if new_cols:
                    desc_parts.append(f"Added columns: {list(new_cols)}")
                if deleted_cols:
                    desc_parts.append(f"Removed columns: {list(deleted_cols)}")

                if "schema_drift_retrain" not in state.human_approvals:
                    state.risk_flags.append(
                        RiskFlag(
                            level=RISK_WARNING,
                            code=DRIFT_DETECTED,
                            feature=None,
                            description=f"Column schema changed. {'. '.join(desc_parts)}",
                            recommended_action="Approve retraining the pipeline with the updated column structure.",
                            requires_human_approval=True
                        )
                    )

        # 5. Population Stability Index (PSI) checking on numerical predictors
        common_numeric = [
            col for col in baseline_df.columns 
            if col in state.df.columns 
            and pd.api.types.is_numeric_dtype(baseline_df[col]) 
            and col != state.target_column
        ]
        
        drifted_features = []
        for col in common_numeric:
            psi = compute_psi(baseline_df[col], state.df[col])
            if psi >= 0.2:
                drifted_features.append(f"{col} (PSI={psi:.3f})")

        if drifted_features:
            self.logger.warning("statistical_drift_detected", columns=drifted_features)
            state.risk_flags.append(
                RiskFlag(
                    level=RISK_WARNING,
                    code=DRIFT_DETECTED,
                    feature=None,
                    description=f"Significant statistical drift detected in columns: {', '.join(drifted_features)}",
                    recommended_action="Monitor downstream predictions; statistical distribution has shifted.",
                    requires_human_approval=False
                )
            )

        return state
