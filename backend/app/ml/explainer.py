# ama2/backend/app/ml/explainer.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

from ..utils.logging import get_logger

logger = get_logger("ml_explainer")


class ModelExplainer:
    """
    Computes SHAP values, builds global feature importance lists,
    and extracts local explanations for representative correct, borderline, and failure cases.
    """

    def __init__(self, problem_type: str):
        self.problem_type = problem_type

    def _get_explainer(self, model: Any, X_background: np.ndarray) -> Any:
        """Selects the best SHAP explainer strategy based on estimator properties."""
        if not HAS_SHAP or shap is None:
            raise ImportError("SHAP is not installed or failed to import.")

        # 1. Tree Explainer for tree-based models
        if hasattr(model, "feature_importances_"):
            try:
                return shap.TreeExplainer(model)
            except Exception:
                pass

        # 2. Linear Explainer for linear models
        if hasattr(model, "coef_"):
            try:
                return shap.LinearExplainer(model, X_background)
            except Exception:
                pass

        # 3. Kernel Explainer as general fallback (uses predict_proba for class, predict for reg)
        # We cap background data size to keep execution under budget
        background_subset = X_background[:min(20, len(X_background))]
        if hasattr(model, "predict_proba"):
            return shap.KernelExplainer(model.predict_proba, background_subset)
        return shap.KernelExplainer(model.predict, background_subset)

    def explain(self, model_pipeline: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Runs the explanation workflow and returns structured results."""
        y_test_arr = np.array(y_test)
        preprocessor = model_pipeline.named_steps["preprocessor"]
        estimator = model_pipeline.named_steps["model"]

        # Transform inputs into preprocessed feature space
        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        # Retrieve preprocessed feature names
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_names = [f"f_{i}" for i in range(X_train_proc.shape[1])]

        # Ensure numeric arrays
        if isinstance(X_train_proc, pd.DataFrame):
            X_train_proc = X_train_proc.values
        if isinstance(X_test_proc, pd.DataFrame):
            X_test_proc = X_test_proc.values

        # Fallback if SHAP is absent
        if not HAS_SHAP:
            logger.warning("shap_not_available_using_surrogate_importances")
            return self._build_surrogate_explanation(estimator, feature_names)

        try:
            # Enforce budget: explain on at most 200 samples
            n_explain = min(200, len(X_test_proc))
            X_explain = X_test_proc[:n_explain]

            explainer = self._get_explainer(estimator, X_train_proc)
            
            # Compute raw SHAP values
            if isinstance(explainer, shap.KernelExplainer):
                raw_shap = explainer.shap_values(X_explain)
            else:
                raw_shap = explainer(X_explain)
                if hasattr(raw_shap, "values"):
                    raw_shap = raw_shap.values

            # Align SHAP dimensionality for classification/regression
            # (Binary class TreeExplainer can output a list of 2 arrays, one per class)
            if isinstance(raw_shap, list):
                # For classification, default to explaining the positive or predicted class
                shap_values = np.array(raw_shap[-1])
            elif len(raw_shap.shape) == 3:
                # Multiclass SHAP [samples, features, classes]
                shap_values = raw_shap[:, :, -1]
            else:
                shap_values = raw_shap

            # Calculate global feature importance (mean absolute SHAP)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            global_importance = []
            for col_idx, importance in enumerate(mean_abs_shap):
                global_importance.append({
                    "feature": feature_names[col_idx],
                    "importance": float(importance)
                })
            # Sort descending
            global_importance = sorted(global_importance, key=lambda x: x["importance"], reverse=True)

            # Compute local explanations for 3 representative samples
            local_explanations = self._extract_representative_cases(
                model_pipeline, X_test, y_test_arr, shap_values, feature_names
            )

            return {
                "global_importance": global_importance,
                "local_explanations": local_explanations
            }

        except Exception as e:
            logger.exception("shap_computation_failed_falling_back", error=str(e))
            return self._build_surrogate_explanation(estimator, feature_names)

    def _build_surrogate_explanation(self, estimator: Any, feature_names: List[str]) -> Dict[str, Any]:
        """Provides fallback explanation structure when SHAP is unavailable."""
        global_importance = []
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            for idx, imp in enumerate(importances):
                global_importance.append({"feature": feature_names[idx], "importance": float(imp)})
        elif hasattr(estimator, "coef_"):
            coefs = np.abs(estimator.coef_)
            if len(coefs.shape) > 1:
                coefs = np.mean(coefs, axis=0)
            for idx, c in enumerate(coefs):
                global_importance.append({"feature": feature_names[idx], "importance": float(c)})
        else:
            # Equal importance fallback
            for name in feature_names:
                global_importance.append({"feature": name, "importance": 1.0 / len(feature_names)})

        global_importance = sorted(global_importance, key=lambda x: x["importance"], reverse=True)
        return {
            "global_importance": global_importance,
            "local_explanations": {
                "highest_confidence_correct": None,
                "borderline": None,
                "failure_case": None
            }
        }

    def _extract_representative_cases(
        self, model_pipeline: Any, X_test: pd.DataFrame, y_test: np.ndarray, shap_values: np.ndarray, feature_names: List[str]
    ) -> Dict[str, Any]:
        """Selects three key sample indices representing correct, borderline, and incorrect predictions."""
        y_pred = model_pipeline.predict(X_test)
        y_pred_arr = np.array(y_pred)
        
        n_samples = len(y_pred_arr)
        # Map back to test indices
        indices = np.arange(min(n_samples, len(shap_values)))
        
        correct_mask = (y_pred_arr[:len(shap_values)] == y_test[:len(shap_values)])

        # Default indices
        idx_correct = idx_borderline = idx_failure = 0

        if self.problem_type == "classification" and hasattr(model_pipeline, "predict_proba"):
            y_prob = model_pipeline.predict_proba(X_test)[:len(shap_values)]
            if y_prob.shape[1] == 2:
                # Binary: prob of predicted class
                pred_prob = np.where(y_pred_arr[:len(shap_values)] == 1, y_prob[:, 1], y_prob[:, 0])
            else:
                pred_prob = np.max(y_prob, axis=1)

            # Correct predictions
            correct_indices = indices[correct_mask]
            if len(correct_indices) > 0:
                idx_correct = int(correct_indices[np.argmax(pred_prob[correct_indices])])

            # Borderline prediction (closest confidence to 0.5)
            idx_borderline = int(np.argmin(np.abs(pred_prob - 0.5)))

            # Failure cases
            incorrect_indices = indices[~correct_mask]
            if len(incorrect_indices) > 0:
                idx_failure = int(incorrect_indices[np.argmax(pred_prob[incorrect_indices])])
            else:
                idx_failure = idx_correct
        else:
            # Regression or classification without probability
            # Distance from true value
            errors = np.abs(y_test[:len(shap_values)] - y_pred_arr[:len(shap_values)])
            
            # Highest confidence correct -> smallest error
            idx_correct = int(np.argmin(errors))
            
            # Borderline -> median error
            idx_borderline = int(np.argsort(errors)[len(errors) // 2])
            
            # Failure case -> largest error
            idx_failure = int(np.argmax(errors))

        cases = {}
        for name, idx in [
            ("highest_confidence_correct", idx_correct),
            ("borderline", idx_borderline),
            ("failure_case", idx_failure)
        ]:
            # Slice actual feature row (raw strings/numeric values)
            raw_row = X_test.iloc[idx].to_dict()
            
            # Slice SHAP values for this index
            row_shaps = shap_values[idx]
            shap_dict = {feature_names[i]: float(row_shaps[i]) for i in range(len(feature_names))}

            cases[name] = {
                "test_index": int(X_test.index[idx]),
                "actual": float(y_test[idx]) if isinstance(y_test[idx], (int, float, np.integer, np.floating)) else str(y_test[idx]),
                "predicted": float(y_pred_arr[idx]) if isinstance(y_pred_arr[idx], (int, float, np.integer, np.floating)) else str(y_pred_arr[idx]),
                "features": raw_row,
                "shap_values": shap_dict
            }
            
        return cases
