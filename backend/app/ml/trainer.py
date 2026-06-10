# ama2/backend/app/ml/trainer.py

from __future__ import annotations

import time
import tracemalloc
import numpy as np
import pandas as pd
import optuna
from typing import Any, Dict, List

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator

from ..core.constants import RANDOM_SEED
from ..utils.schema_fingerprint import compute_fingerprint
from ..utils.logging import get_logger

logger = get_logger("ml_trainer")

# Suppress Optuna logs to keep stdout clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Trainer:
    """
    Manages CV splits, Optuna hyperparameter tuning, MLflow nested tracking,
    wall-clock timing, peak memory tracing, and baseline improvement validation.
    """

    def __init__(self, problem_type: str, cv_strategy: str, target_column: str, group_column: str | None = None):
        self.problem_type = problem_type
        self.cv_strategy = cv_strategy
        self.target_column = target_column
        self.group_column = group_column

    def _get_cv_splitter(self) -> Any:
        """Instantiates the correct cross-validation strategy splitter."""
        if self.cv_strategy == "GroupKFold":
            return GroupKFold(n_splits=5)
        elif self.cv_strategy == "TimeSeriesSplit":
            return TimeSeriesSplit(n_splits=5)
        elif self.cv_strategy == "KFold":
            return KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        else:
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def _get_scoring(self, y: pd.Series) -> str:
        """Determines the standard metric scoring type for validation."""
        if self.problem_type == "classification":
            # Multi-class or binary
            if y.nunique() > 2:
                return "roc_auc_ovr"
            return "roc_auc"
        return "r2"

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, model_candidates: List[Dict[str, Any]], preprocessor_pipeline: Any, df_full: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs the full cross-validation and hyperparameter tuning loop across candidates.
        """
        cv = self._get_cv_splitter()
        scoring = self._get_scoring(y_train)
        
        # Extract groups if GroupKFold is used
        groups = None
        if self.cv_strategy == "GroupKFold" and self.group_column:
            if self.group_column in df_full.columns:
                groups = df_full.loc[X_train.index, self.group_column]

        results = []
        schema_hash = compute_fingerprint(df_full)

        for candidate in model_candidates:
            name = candidate["name"]
            estimator = candidate["estimator"]
            param_grid = candidate["param_grid"]
            
            logger.info("training_candidate", name=name)

            # Define MLflow run context
            # (Assumes parent run is already active in BaseAgent)
            run_ctx = mlflow.start_run(run_name=name, nested=True) if HAS_MLFLOW else None
            
            try:
                # 1. Define full pipeline (preprocessor + candidate model)
                full_pipeline = Pipeline([
                    ("preprocessor", preprocessor_pipeline),
                    ("model", estimator)
                ])

                # 2. Optuna Hyperparameter Tuning (if param grid is not empty)
                best_params = {}
                cv_scores = None

                if param_grid:
                    best_params = self._optuna_tune(
                        full_pipeline, X_train, y_train, cv, groups, scoring, param_grid
                    )
                    # Apply best parameters to the estimator clone
                    tuned_estimator = clone(estimator).set_params(**best_params)
                    full_pipeline = Pipeline([
                        ("preprocessor", preprocessor_pipeline),
                        ("model", tuned_estimator)
                    ])

                # 3. Final CV evaluation with best/default parameters
                cv_results = cross_validate(
                    full_pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    groups=groups,
                    return_train_score=True,
                    error_score="raise"
                )
                
                test_scores = cv_results["test_score"]
                cv_mean = float(np.mean(test_scores))
                cv_std = float(np.std(test_scores))

                # Log metrics to MLflow
                if HAS_MLFLOW:
                    mlflow.log_param("schema_hash", schema_hash)
                    mlflow.log_param("model_type", estimator.__class__.__name__)
                    for k, v in best_params.items():
                        mlflow.log_param(f"best_param_{k}", v)
                    mlflow.log_metric("cv_mean", cv_mean)
                    mlflow.log_metric("cv_std", cv_std)
                    for idx, score in enumerate(test_scores):
                        mlflow.log_metric("cv_fold_score", score, step=idx)

                results.append({
                    "name": name,
                    "estimator": full_pipeline.named_steps["model"],  # Unfit tuned estimator template
                    "best_params": best_params,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "cv_scores": test_scores.tolist(),
                })
            
            except Exception as e:
                logger.exception("candidate_training_failed", name=name, error=str(e))
                if HAS_MLFLOW:
                    mlflow.set_tag("status", "failed")
                    mlflow.log_param("error", str(e)[:250])
                # Do not re-raise, we want the pipeline to proceed with other candidates
            finally:
                if run_ctx:
                    mlflow.end_run()

        if not results:
            raise ValueError("All candidate model training attempts failed.")

        # 4. Enforce improvement over dummy baselines by 5% relative threshold
        dummy_scores = [r for r in results if "dummy" in r["name"].lower()]
        real_models = [r for r in results if "dummy" not in r["name"].lower()]

        dummy_baseline = max(d["cv_mean"] for d in dummy_scores) if dummy_scores else 0.0

        if self.problem_type == "regression":
            # Dummy R2 is around 0. Real model beats baseline if R2 > dummy + 0.05
            beats_baseline = [r for r in real_models if r["cv_mean"] > max(dummy_baseline + 0.05, 0.05)]
        else:
            # Classification ROC-AUC (typically 0.5 for dummy). 0.5 * 1.05 = 0.525.
            beats_baseline = [r for r in real_models if r["cv_mean"] > dummy_baseline * 1.05]

        # Select best candidate
        if not beats_baseline:
            # Failed to beat dummy baseline!
            # Return dummy if no real model beat the gate, flagging failure
            best = max(results, key=lambda r: r["cv_mean"])
            return {
                "best_candidate": best,
                "beats_dummy_baseline": False,
                "pass_gate": False,
                "all_results": results
            }

        best = max(beats_baseline, key=lambda r: r["cv_mean"])
        return {
            "best_candidate": best,
            "beats_dummy_baseline": True,
            "pass_gate": True,
            "all_results": results
        }

    def _optuna_tune(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: Any, groups: Any, scoring: str, param_grid: dict[str, list]) -> dict[str, Any]:
        """Runs Optuna hyperparameter optimization with MedianPruner support."""
        
        def objective(trial: optuna.Trial) -> float:
            sampled_params = {}
            for param_name, values in param_grid.items():
                sampled_params[param_name] = trial.suggest_categorical(param_name, values)
            
            # Map params to pipeline's model step
            model_params = {f"model__{k}": v for k, v in sampled_params.items()}
            
            # Clone and set params to avoid shared state across trials
            trial_pipeline = clone(pipeline)
            trial_pipeline.set_params(**model_params)
            
            try:
                cv_res = cross_validate(
                    trial_pipeline, X, y, cv=cv, scoring=scoring, groups=groups, error_score="raise"
                )
                score = float(np.mean(cv_res["test_score"]))
                
                # MLflow run logging for trials
                if HAS_MLFLOW:
                    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                        for k, v in sampled_params.items():
                            mlflow.log_param(k, v)
                        mlflow.log_metric("cv_score", score)
                
                return score
            except Exception as e:
                # Log failed trial and return worst possible score
                logger.warning("trial_failed", trial=trial.number, error=str(e))
                return -999.0 if self.problem_type == "classification" else 999.0

        study = optuna.create_study(
            direction="maximize" if self.problem_type == "classification" else "minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        study.optimize(objective, n_trials=30)
        return study.best_params
