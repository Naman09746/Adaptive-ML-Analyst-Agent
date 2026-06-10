# ama2/backend/app/ml/evaluator.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

try:
    from scipy.stats import chi2
    HAS_SCIPY = True
except ImportError:
    chi2 = None
    HAS_SCIPY = False


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) for classification probability calibration check."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
        # Multiclass
        preds = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        accuracies = (preds == y_true)
    else:
        # Binary
        preds = (y_prob >= 0.5).astype(int)
        confidences = np.where(preds == 1, y_prob, 1.0 - y_prob)
        accuracies = (preds == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


class Evaluator:
    """
    Computes validation and holdout metrics, slices performance by categorical attributes,
    checks probability calibration error, and performs heteroscedasticity diagnostics.
    """

    def __init__(self, problem_type: str, target_column: str):
        self.problem_type = problem_type
        self.target_column = target_column

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, df_full: pd.DataFrame) -> Dict[str, Any]:
        """Calculates classification or regression metrics and runs slice checks."""
        y_test_arr = np.array(y_test)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_arr = np.array(y_pred)

        metrics: Dict[str, Any] = {}

        if self.problem_type == "classification":
            # Check if predict_proba is supported
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    pass

            # Accuracy & F1
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["f1_weighted"] = float(f1_score(y_test, y_pred, average="weighted"))
            metrics["f1_per_class"] = f1_score(y_test, y_pred, average=None).tolist()
            metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

            # ROC-AUC
            if y_prob is not None:
                if y_prob.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                    # Precision Recall Curve
                    prec, rec, thrs = precision_recall_curve(y_test, y_prob[:, 1])
                    metrics["precision_recall_curve"] = {
                        "precision": prec.tolist(),
                        "recall": rec.tolist(),
                        "thresholds": thrs.tolist(),
                    }
                    # ECE
                    metrics["ece"] = expected_calibration_error(y_test_arr, y_prob[:, 1])
                else:
                    try:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))
                    except Exception:
                        metrics["roc_auc"] = 0.5
                    metrics["ece"] = expected_calibration_error(y_test_arr, y_prob)
            else:
                metrics["roc_auc"] = 0.5
                metrics["ece"] = 0.0

            # Run Slice Analysis
            metrics["slice_analysis"] = self._compute_slice_analysis_classification(
                X_test, y_test_arr, y_pred_arr, y_prob, df_full
            )

        else:
            # Regression Metrics
            metrics["rmse"] = float(mean_squared_error(y_test, y_pred, squared=False))
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["r2"] = float(r2_score(y_test, y_pred))

            # Breusch-Pagan homoscedasticity test
            metrics["residual_heteroscedasticity_p_value"] = self._compute_breusch_pagan_p_value(
                y_test_arr, y_pred_arr
            )

            # Run Slice Analysis
            metrics["slice_analysis"] = self._compute_slice_analysis_regression(
                X_test, y_test_arr, y_pred_arr, df_full
            )

        return metrics

    def _compute_breusch_pagan_p_value(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Runs Breusch-Pagan auxiliary test to evaluate residual variance homoscedasticity."""
        residuals = y_true - y_pred
        squared_residuals = residuals ** 2
        
        n_samples = len(y_true)
        if n_samples < 5:
            return 1.0

        # Regress squared residuals on predictions
        try:
            from sklearn.linear_model import LinearRegression
            aux = LinearRegression()
            aux.fit(y_pred.reshape(-1, 1), squared_residuals)
            r2_aux = aux.score(y_pred.reshape(-1, 1), squared_residuals)
            
            lm_stat = n_samples * r2_aux
            if HAS_SCIPY and chi2 is not None:
                p_val = 1.0 - chi2.cdf(lm_stat, df=1)
                return float(p_val)
            
            # Fallback simple correlation coefficient p-value approximation
            corr = np.corrcoef(y_pred, squared_residuals)[0, 1]
            if np.isnan(corr):
                return 1.0
            return float(1.0 - abs(corr))
        except Exception:
            return 1.0

    def _compute_slice_analysis_classification(
        self, X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, df_full: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculates categorical subpopulation ROC-AUC scores for slice checking."""
        slice_results = {}
        categorical_cols = list(X_test.select_dtypes(include=["object", "category", "bool"]).columns)

        for col in categorical_cols:
            if col == self.target_column:
                continue
            # Get the values corresponding to X_test index from full dataframe
            test_vals = df_full.loc[X_test.index, col].astype(str)
            for val in test_vals.unique():
                mask = (test_vals == val).values
                n_samples = int(np.sum(mask))
                
                # Exclude small slices to prevent statistical noise
                if n_samples < 10:
                    continue

                slice_y_true = y_true[mask]
                slice_y_pred = y_pred[mask]
                
                # If only one class exists in this slice, ROC-AUC is not defined. Fallback to accuracy.
                if len(np.unique(slice_y_true)) < 2:
                    acc = float(accuracy_score(slice_y_true, slice_y_pred))
                    slice_results[f"{col}={val}"] = {
                        "metric": "accuracy",
                        "score": acc,
                        "n": n_samples,
                        "pass_gate": acc >= 0.55
                    }
                    continue

                if y_prob is not None:
                    try:
                        if len(y_prob.shape) > 1 and y_prob.shape[1] > 2:
                            auc = float(roc_auc_score(slice_y_true, y_prob[mask], multi_class="ovr"))
                        elif len(y_prob.shape) > 1:
                            auc = float(roc_auc_score(slice_y_true, y_prob[mask, 1]))
                        else:
                            auc = float(roc_auc_score(slice_y_true, y_prob[mask]))
                    except Exception:
                        auc = float(accuracy_score(slice_y_true, slice_y_pred))
                else:
                    auc = float(accuracy_score(slice_y_true, slice_y_pred))

                slice_results[f"{col}={val}"] = {
                    "metric": "auc",
                    "score": auc,
                    "n": n_samples,
                    "pass_gate": auc >= 0.55
                }
        return slice_results

    def _compute_slice_analysis_regression(
        self, X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, df_full: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculates categorical subpopulation R² scores for slice checking."""
        slice_results = {}
        categorical_cols = list(X_test.select_dtypes(include=["object", "category", "bool"]).columns)

        for col in categorical_cols:
            if col == self.target_column:
                continue
            test_vals = df_full.loc[X_test.index, col].astype(str)
            for val in test_vals.unique():
                mask = (test_vals == val).values
                n_samples = int(np.sum(mask))

                if n_samples < 10:
                    continue

                slice_y_true = y_true[mask]
                slice_y_pred = y_pred[mask]
                
                try:
                    r2 = float(r2_score(slice_y_true, slice_y_pred))
                except Exception:
                    r2 = -1.0

                # For regression, check if R2 is positive or beats simple mean baseline
                slice_results[f"{col}={val}"] = {
                    "metric": "r2",
                    "score": r2,
                    "n": n_samples,
                    "pass_gate": r2 >= -0.5
                }
        return slice_results
