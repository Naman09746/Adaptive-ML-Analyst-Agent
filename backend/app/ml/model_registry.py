# ama2/backend/app/ml/model_registry.py

from abc import ABC, abstractmethod
from typing import Any, ClassVar, List, Type

try:
    from sklearn.base import BaseEstimator
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency fallback
    BaseEstimator = Any  # type: ignore[assignment]
    RandomForestClassifier = RandomForestRegressor = LogisticRegression = Ridge = None
    HAS_SKLEARN = False

from ..core.constants import RANDOM_SEED

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False

class ModelStrategy(ABC):
    tier: int
    min_samples: int = 0

    @abstractmethod
    def build(self, problem_type: str) -> BaseEstimator:
        """Create a new estimator instance."""
        ...

    @abstractmethod
    def get_param_grid(self) -> dict:
        """Parameters for hyperparameter optimization."""
        ...

class ModelRegistry:
    _registry: ClassVar[dict[str, ModelStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: Type[ModelStrategy]):
            cls._registry[name] = klass()
            return klass
        return decorator

    @classmethod
    def get_eligible(cls, n_samples: int, problem_type: str) -> List[ModelStrategy]:
        """Return strategies eligible for this dataset size and type, sorted by hierarchy (tier)."""
        return sorted(
            [s for s in cls._registry.values() if n_samples >= s.min_samples],
            key=lambda s: s.tier
        )

    @classmethod
    def available_names(cls) -> list[str]:
        return sorted(cls._registry.keys())

@ModelRegistry.register("logistic")
class LogisticStrategy(ModelStrategy):
    tier = 1
    def build(self, problem_type: str):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required to build model strategies")
        if problem_type == "classification":
            return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
        return Ridge(random_state=RANDOM_SEED)

    def get_param_grid(self):
        return {"C": [0.01, 0.1, 1, 10]}

@ModelRegistry.register("random_forest")
class RandomForestStrategy(ModelStrategy):
    tier = 2
    min_samples = 200
    def build(self, problem_type: str):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required to build model strategies")
        if problem_type == "classification":
            return RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        return RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)

    def get_param_grid(self):
        return {"n_estimators": [100, 200, 500], "max_depth": [None, 10, 20]}

if HAS_XGBOOST:

    @ModelRegistry.register("xgboost")
    class XGBoostStrategy(ModelStrategy):
        tier = 3
        min_samples = 500

        def build(self, problem_type: str):
            if problem_type == "classification":
                return XGBClassifier(
                    n_estimators=300,
                    random_state=RANDOM_SEED,
                    eval_metric="logloss",
                )
            return XGBRegressor(n_estimators=300, random_state=RANDOM_SEED)

        def get_param_grid(self):
            return {"n_estimators": [100, 300, 500], "learning_rate": [0.01, 0.1, 0.3]}
