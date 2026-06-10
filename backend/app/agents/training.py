# ama2/backend/app/agents/training.py

from __future__ import annotations

import time
import tracemalloc
import numpy as np
import pandas as pd
from typing import Any

from ..core.pipeline_state import PipelineState
from .base import BaseAgent, mlflow
from ..ml.trainer import Trainer


class TrainingAgent(BaseAgent):
    """
    Coordinates model selection and tuning. Runs cross-validation and hyperparameter optimization,
    evaluates baseline improvement gates, and logs parameters/metrics to MLflow.
    """

    def __init__(self):
        super().__init__(name="training")

    def _execute(self, state: PipelineState) -> PipelineState:
        if state.X_train is None or state.y_train is None:
            raise ValueError("Training data (X_train, y_train) must be present in PipelineState before training.")

        if not state.model_candidates:
            raise ValueError("No model candidates found in PipelineState. Run model_strategy first.")

        self.logger.info("starting_training_agent_execution", candidates=[c["name"] for c in state.model_candidates])

        # 1. Initialize and run the ML Trainer
        trainer = Trainer(
            problem_type=state.problem_type or "classification",
            cv_strategy=state.cv_strategy or "StratifiedKFold",
            target_column=state.target_column,
            group_column=state.group_column
        )

        res = trainer.run(
            X_train=state.X_train,
            y_train=state.y_train,
            model_candidates=state.model_candidates,
            preprocessor_pipeline=state.sklearn_pipeline,
            df_full=state.df
        )

        best_cand = res["best_candidate"]
        state.eval_metrics.update({
            "cv_mean": best_cand["cv_mean"],
            "cv_std": best_cand["cv_std"],
            "beats_dummy_baseline": res["beats_dummy_baseline"],
            "pass_gate": res["pass_gate"],
        })

        # Record all results for comparison reports
        state.eval_metrics["all_candidates"] = [
            {"name": r["name"], "cv_mean": r["cv_mean"], "cv_std": r["cv_std"]} 
            for r in res["all_results"]
        ]

        if not res["pass_gate"]:
            self.logger.warning("training_did_not_pass_gate", message="No model beat the dummy baseline by 5%")
            self._log_decision(
                state, "training_failure", False, "No models outperformed the dummy baseline."
            )
            return state

        # 2. Fit the best pipeline (preprocessor + tuned estimator) on full X_train
        from sklearn.pipeline import Pipeline
        from sklearn.base import clone

        # Tuned, unfit estimator returned by trainer
        tuned_estimator = best_cand["estimator"]
        
        best_pipeline = Pipeline([
            ("preprocessor", state.sklearn_pipeline),
            ("model", tuned_estimator)
        ])

        self.logger.info("fitting_final_best_model", name=best_cand["name"])

        # Start wall-clock timing and memory tracking
        t0 = time.perf_counter()
        tracemalloc.start()

        # Fit full pipeline
        best_pipeline.fit(state.X_train, state.y_train)

        # Retrieve performance stats
        fit_duration = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024.0 * 1024.0)

        # Write best model and stats to state
        state.best_model = best_pipeline
        state.best_model_name = best_cand["name"]
        state.eval_metrics["fit_duration_s"] = fit_duration
        state.eval_metrics["peak_memory_mb"] = peak_mb

        # 3. Post-CV dominant feature check
        fitted_model = best_pipeline.named_steps["model"]
        importances = None
        if hasattr(fitted_model, "feature_importances_"):
            importances = fitted_model.feature_importances_
        elif hasattr(fitted_model, "coef_"):
            importances = np.abs(fitted_model.coef_)
            if len(importances.shape) > 1:
                importances = np.mean(importances, axis=0)

        if importances is not None:
            try:
                feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
            except Exception:
                feature_names = [f"f_{i}" for i in range(state.X_train.shape[1])]

            if len(importances) == len(feature_names):
                total_imp = float(np.sum(importances))
                if total_imp > 0:
                    rel_importances = importances / total_imp
                    max_idx = np.argmax(rel_importances)
                    max_ratio = float(rel_importances[max_idx])
                    
                    if max_ratio > 0.5:
                        state.eval_metrics["dominant_feature_name"] = str(feature_names[max_idx])
                        state.eval_metrics["dominant_feature_ratio"] = max_ratio

        # 4. Set current MLflow run details
        if HAS_MLFLOW:
            active_run = mlflow.active_run()
            if active_run:
                state.mlflow_run_id = active_run.info.run_id
                mlflow.log_metric("fit_duration_s", fit_duration)
                mlflow.log_metric("peak_memory_mb", peak_mb)

        self._log_decision(
            state,
            "best_model_selection",
            {
                "model_name": state.best_model_name,
                "cv_mean": state.eval_metrics["cv_mean"],
                "cv_std": state.eval_metrics["cv_std"],
                "fit_duration_s": fit_duration,
                "peak_memory_mb": peak_mb,
            },
            f"Successfully tuned and trained best candidate model: {state.best_model_name}"
        )

        return state
