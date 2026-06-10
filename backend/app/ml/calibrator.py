# ama2/backend/app/ml/calibrator.py

from __future__ import annotations

from typing import Any
from sklearn.calibration import CalibratedClassifierCV

class ModelCalibrator:
    """
    Applies probability calibration (Platt scaling / Sigmoid or Isotonic regression)
    to a classification model to improve Expected Calibration Error (ECE).
    """

    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("Method must be 'sigmoid' or 'isotonic'")
        self.method = method

    def calibrate(self, model_pipeline: Any, X_val: Any, y_val: Any, cv: str | int = "prefit") -> Any:
        """
        Calibrates the classification pipeline using validation data.
        Returns a calibrated pipeline wrapper.
        """
        # CalibratedClassifierCV works on classifiers. 
        # If it is a full pipeline, CalibratedClassifierCV wraps the entire pipeline.
        calibrated_model = CalibratedClassifierCV(
            estimator=model_pipeline,
            method=self.method,
            cv=cv
        )
        # Fit calibration on validation data
        calibrated_model.fit(X_val, y_val)
        return calibrated_model
