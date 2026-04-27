# ama2/backend/app/core/pipeline_state.py

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Union
from uuid import UUID
import pandas as pd
from .constants import RANDOM_SEED

@dataclass
class TraceEntry:
    agent: str
    decision_key: str
    decision_value: Any
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class RiskFlag:
    level: str          # critical | warning | info
    code: str           # enum: LEAKAGE_SUSPECTED | TINY_DATASET | DRIFT_DETECTED | etc.
    feature: Optional[str]
    description: str
    recommended_action: str
    requires_human_approval: bool

@dataclass
class PipelineState:
    session_id: UUID
    user_id: str
    dataset_path: str
    problem_statement: str

    df: Optional[pd.DataFrame] = None
    data_profile: dict = field(default_factory=dict)
    target_column: Optional[str] = None
    problem_type: Optional[str] = None     # classification | regression | timeseries
    confidence_level: Optional[str] = None # safe | uncertain | unsafe
    cv_strategy: Optional[str] = None      # StratifiedKFold | KFold | TimeSeriesSplit | GroupKFold
    group_column: Optional[str] = None     # detected group col (user_id, store_id etc.)
    leakage_suspects: list[str] = field(default_factory=list)
    preprocessing_plan: dict = field(default_factory=dict)
    sklearn_pipeline: Any = None           # fitted ColumnTransformer (never raw steps)
    X_train: Any = None
    X_test: Any = None
    y_train: Any = None
    y_test: Any = None
    model_candidates: list[dict] = field(default_factory=list)
    best_model: Any = None
    best_model_name: Optional[str] = None
    eval_metrics: dict = field(default_factory=dict)
    shap_values: Any = None
    business_narrative: Optional[str] = None
    risk_flags: list[RiskFlag] = field(default_factory=list)
    pending_approval_gates: list[str] = field(default_factory=list)
    human_approvals: dict[str, dict] = field(default_factory=dict)
    trace_log: list[TraceEntry] = field(default_factory=list)
    mlflow_run_id: Optional[str] = None
    report_path: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    halt: bool = False
    halt_reason: Optional[str] = None
