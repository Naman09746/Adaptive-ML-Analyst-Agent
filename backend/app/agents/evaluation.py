# ama2/backend/app/agents/evaluation.py

from __future__ import annotations

from ..core.constants import RISK_CRITICAL, RISK_WARNING
from ..core.pipeline_state import PipelineState, RiskFlag
from .base import BaseAgent
from ..ml.evaluator import Evaluator


class EvaluationAgent(BaseAgent):
    """
    Evaluates the trained model against holdout test data. Computes statistics,
    compares train vs test performance (overfitting gap), checks slice-level accuracy/AUC,
    validates probability calibration error (ECE), and enforces quality gating.
    """

    def __init__(self):
        super().__init__(name="evaluation")

    def _execute(self, state: PipelineState) -> PipelineState:
        if state.best_model is None:
            self.logger.warning("no_model_available_for_evaluation")
            state.eval_metrics["pass_gate"] = False
            return state

        if state.X_test is None or state.y_test is None:
            raise ValueError("Test data (X_test, y_test) must be present in PipelineState for evaluation.")

        self.logger.info("running_evaluator_holdout_check", model_name=state.best_model_name)

        # 1. Compute holdout and slice metrics
        evaluator = Evaluator(
            problem_type=state.problem_type or "classification",
            target_column=state.target_column
        )

        test_metrics = evaluator.evaluate(
            model=state.best_model,
            X_test=state.X_test,
            y_test=state.y_test,
            df_full=state.df
        )

        # Merge new holdout metrics into PipelineState.eval_metrics
        state.eval_metrics.update(test_metrics)

        # 2. Extract primary metrics for validation checks
        cv_mean = state.eval_metrics.get("cv_mean", 0.0)
        
        is_classification = (state.problem_type == "classification")
        if is_classification:
            holdout_score = state.eval_metrics.get("roc_auc", 0.5)
        else:
            holdout_score = state.eval_metrics.get("r2", 0.0)

        # 3. Validation Gate A: Train-validation (CV vs Holdout) gap check
        # Standard overfitting rule: relative difference
        gap = cv_mean - holdout_score
        rel_gap = gap / max(abs(cv_mean), 1e-5)

        if rel_gap > 0.25:
            state.eval_metrics["pass_gate"] = False
            state.risk_flags.append(
                RiskFlag(
                    level=RISK_CRITICAL,
                    code="OVERFITTING_FAIL",
                    feature=None,
                    description=f"Overfitting detected. CV-Holdout gap is too large ({rel_gap:.2%} > 25%).",
                    recommended_action="Regularize hyperparameters, try a simpler model, or gather more data.",
                    requires_human_approval=False
                )
            )
        elif rel_gap > 0.10:
            state.risk_flags.append(
                RiskFlag(
                    level=RISK_WARNING,
                    code="OVERFITTING_WARN",
                    feature=None,
                    description=f"Potential overfitting. CV-Holdout gap is moderate ({rel_gap:.2%} > 10%).",
                    recommended_action="Monitor holdout performance closely; model may not generalize perfectly.",
                    requires_human_approval=False
                )
            )

        # 4. Validation Gate B: Slice performance check
        slice_analysis = state.eval_metrics.get("slice_analysis", {})
        failed_slices = [
            slice_name for slice_name, details in slice_analysis.items()
            if not details.get("pass_gate", True)
        ]

        if failed_slices:
            state.eval_metrics["pass_gate"] = False
            state.risk_flags.append(
                RiskFlag(
                    level=RISK_CRITICAL,
                    code="SLICE_FAILURE",
                    feature=", ".join(failed_slices),
                    description=f"Subpopulation performance failure on slices: {failed_slices}.",
                    recommended_action="Review feature balance or train slice-specific models.",
                    requires_human_approval=False
                )
            )

        # 5. Validation Gate C: Calibration error check (classification only)
        if is_classification:
            ece = state.eval_metrics.get("ece", 0.0)
            if ece > 0.15:
                state.eval_metrics["pass_gate"] = False
                state.risk_flags.append(
                    RiskFlag(
                        level=RISK_CRITICAL,
                        code="HIGH_CALIBRATION_ERROR",
                        feature=None,
                        description=f"Model calibration error is too high (ECE = {ece:.4f} > 0.15).",
                        recommended_action="Apply calibration techniques (temperature scaling / Platt / isotonic).",
                        requires_human_approval=False
                    )
                )

        self._log_decision(
            state,
            "evaluation_complete",
            {
                "holdout_score": holdout_score,
                "cv_holdout_gap": rel_gap,
                "pass_gate": state.eval_metrics["pass_gate"],
                "failed_slices_count": len(failed_slices),
            },
            f"Holdout evaluation completed. Pass gate status: {state.eval_metrics['pass_gate']}."
        )

        return state
