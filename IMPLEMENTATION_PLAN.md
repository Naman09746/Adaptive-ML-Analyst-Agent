# AMA² — Adaptive ML Analyst Agent  
## Refined Product Definition + Architecture

---

## Product Definition

**What it is:** A multi-agent ML operations system that takes a raw tabular dataset + a plain-English problem statement and autonomously profiles the data, infers the ML task type, selects preprocessing and models, evaluates with slice analysis, generates explainable reports, and emits full execution traces — with hard safety gates that block on genuine risk.

**What makes it non-trivial (vs. AutoML demos):**
1. **Structured agent contracts** — every agent-to-agent handoff is a typed JSON schema, never free text
2. **3-checkpoint Risk Agent** — runs pre-training, post-training, and at inference; not a single pass
3. **Graph-level retry routing** — LangGraph routes back to `model_strategy` on eval failure; no try/except hacks
4. **LangGraph interrupt gates** — pipeline literally halts and waits for human decision with mandatory written reason
5. **Audit-first design** — every decision in `agent_decisions` PostgreSQL table + MLflow child run; report is regeneratable from trace alone
6. **DataCorruptor** — dirty data injection module for demo and regression testing

**Users:** Data scientists or ML engineers who upload a CSV and receive a production-quality analysis with full rationale, explainability, and risk assessment.

---

## Architecture Overview

```
[User] → FastAPI → Celery Worker → LangGraph Graph
                                        │
         ┌──────────────────────────────┤ PipelineState (single shared state)
         │                              │
         ▼                              ▼
  PostgreSQL (6 tables)        Node sequence:
  Redis (broker/cache)         1. DataUnderstanding
  MLflow (tracking)            2. ProblemFraming
  FAISS (vector memory)        3. RiskCheck (pre-training)
                               4. ─── human_review gate? ───
                               5. Preprocessing
                               6. ModelStrategy
                               7. Training (CV + Optuna + MLflow)
                               8. Evaluation + RiskCheck (post-training)
                               9. ─── retry → 6 | human_review | halt ───
                              10. Explainability
                              11. ReportGenerator
                              12. END
```

---

## 1. Folder Structure

```
ama2/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app factory + lifespan
│   │   ├── config.py                  # Pydantic Settings (all from env)
│   │   ├── dependencies.py            # DB session, current_user, redis DI
│   │   ├── core/
│   │   │   ├── pipeline_state.py      # PipelineState dataclass (single truth)
│   │   │   ├── orchestrator.py        # LangGraph graph builder
│   │   │   ├── agent_factory.py       # AgentFactory (Factory pattern)
│   │   │   ├── exceptions.py          # Domain exception hierarchy
│   │   │   └── constants.py           # RANDOM_SEED=42, thresholds
│   │   ├── agents/
│   │   │   ├── base.py                # BaseAgent ABC (Template Method)
│   │   │   ├── data_understanding.py
│   │   │   ├── problem_framing.py
│   │   │   ├── risk_failure.py        # 3-checkpoint risk agent
│   │   │   ├── preprocessing.py
│   │   │   ├── model_strategy.py
│   │   │   ├── training.py
│   │   │   ├── evaluation.py
│   │   │   ├── explainability.py
│   │   │   ├── report_generator.py
│   │   │   └── human_review.py
│   │   ├── ml/
│   │   │   ├── preprocessors.py       # Adaptive pipeline builders
│   │   │   ├── model_registry.py      # ModelRegistry + ModelStrategy ABC
│   │   │   ├── trainer.py             # CV strategy selector + Optuna loop
│   │   │   ├── evaluator.py           # Metrics + slice analysis + pass/fail gate
│   │   │   ├── explainer.py           # SHAP wrappers (Tree/Linear/Kernel)
│   │   │   ├── calibrator.py          # Platt/isotonic calibration
│   │   │   └── data_corruptor.py      # DataCorruptor injection module
│   │   ├── db/
│   │   │   ├── base.py                # SQLAlchemy DeclarativeBase
│   │   │   ├── session.py             # async_sessionmaker
│   │   │   ├── models/                # ORM: session, schema_version, agent_decision,
│   │   │   │                          #      model_run, risk_flag, human_approval
│   │   │   └── repositories/          # GenericRepository[T] + per-entity repos
│   │   ├── api/v1/
│   │   │   ├── router.py
│   │   │   ├── auth.py
│   │   │   ├── sessions.py
│   │   │   ├── pipeline.py            # /run + /status (SSE stream)
│   │   │   ├── approvals.py           # /pending + /submit
│   │   │   ├── reports.py             # /pdf | /json | /html
│   │   │   └── traces.py
│   │   ├── schemas/                   # Pydantic I/O contracts per agent
│   │   ├── services/                  # auth, file, pipeline, report, drift
│   │   └── utils/
│   │       ├── logging.py             # structlog JSON renderer
│   │       ├── tracing.py             # MLflow child-run wrappers
│   │       ├── schema_fingerprint.py  # SHA-256 of sorted(col:dtype)
│   │       ├── psi.py                 # PSI per feature (10-bin histogram)
│   │       └── retry.py               # Exponential backoff decorator
│   ├── alembic/
│   ├── tests/
│   │   ├── unit/agents/ ml/ services/ utils/
│   │   ├── integration/
│   │   └── e2e/                       # 5 blueprint demo scenarios
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── pages/  Dashboard, SessionDetail, ApprovalGate, ReportViewer
│   │   ├── components/  TraceTimeline, RiskPanel, ModelTable, ShapChart, ApprovalForm
│   │   ├── hooks/
│   │   ├── api/         # Typed client (axios + zod)
│   │   └── store/       # Zustand
│   └── Dockerfile
├── infra/
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   └── nginx/nginx.conf
└── .github/workflows/  ci.yml, deploy.yml
```

---

## 2. PipelineState — Central Data Contract

```python
# backend/app/core/pipeline_state.py
RANDOM_SEED = 42   # enforced globally in all model constructors

@dataclass
class TraceEntry:
    agent: str; decision_key: str; decision_value: Any
    rationale: str; timestamp: str  # ISO8601

@dataclass
class RiskFlag:
    level: str          # critical | warning | info
    code: str           # enum: LEAKAGE_SUSPECTED | TINY_DATASET | DRIFT_DETECTED |
                        #       SCHEMA_DTYPE_MISMATCH | UNSTABLE_CV | SUSPICIOUS_AUC |
                        #       DOMINANT_FEATURE | HIGH_MISSING_RATE | CLASS_IMBALANCE |
                        #       CONSTANT_COLUMN | HIGH_DUPLICATE_RATIO | MEMORY_BUDGET
    feature: Optional[str]
    description: str
    recommended_action: str
    requires_human_approval: bool

@dataclass
class PipelineState:
    session_id: UUID; user_id: str
    dataset_path: str; problem_statement: str

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
    X_train: Any = None; X_test: Any = None
    y_train: Any = None; y_test: Any = None
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
    retry_count: int = 0; max_retries: int = 3
    halt: bool = False; halt_reason: Optional[str] = None
```

> **Law:** Agents only write to their own designated fields. No agent overwrites another's output.

---

## 3. OOP Design Patterns

### 3.1 BaseAgent (Template Method)

```python
class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)

    @abstractmethod
    def _execute(self, state: PipelineState) -> PipelineState: ...

    def run(self, state: PipelineState) -> PipelineState:
        """Template method: wraps _execute with logging, MLflow child run, error capture."""
        t0 = time.perf_counter()
        structlog.contextvars.bind_contextvars(session_id=str(state.session_id))
        with mlflow.start_run(run_name=self.name, nested=True) as run:
            try:
                state = self._execute(state)
                mlflow.log_metric("latency_s", time.perf_counter() - t0)
                mlflow.set_tag("status", "success")
            except Exception as e:
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e)[:250])
                raise
        return state

    def _log_decision(self, state, key, value, rationale):
        entry = TraceEntry(agent=self.name, decision_key=key,
                           decision_value=value, rationale=rationale,
                           timestamp=datetime.now(timezone.utc).isoformat())
        state.trace_log.append(entry)
        mlflow.log_param(key, str(value)[:250])
```

### 3.2 ModelStrategy (Strategy Pattern)

```python
class ModelStrategy(ABC):
    tier: int; min_samples: int = 0
    @abstractmethod
    def build(self, problem_type: str) -> BaseEstimator: ...
    @abstractmethod
    def get_param_grid(self) -> dict: ...

class ModelRegistry:
    _registry: ClassVar[dict[str, type[ModelStrategy]]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass): cls._registry[name] = klass(); return klass
        return decorator

    @classmethod
    def get_eligible(cls, n: int, problem_type: str) -> list[ModelStrategy]:
        return sorted(
            [s for s in cls._registry.values() if n >= s.min_samples],
            key=lambda s: s.tier
        )

@ModelRegistry.register("logistic")
class LogisticStrategy(ModelStrategy):
    tier = 1
    def build(self, problem_type):
        return LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_SEED) if problem_type == "classification" \
               else Ridge(random_state=RANDOM_SEED)
    def get_param_grid(self): return {"model__C": [0.01, 0.1, 1, 10]}

@ModelRegistry.register("random_forest")
class RandomForestStrategy(ModelStrategy):
    tier = 2; min_samples = 200
    def build(self, problem_type):
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED) \
               if problem_type == "classification" \
               else RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED)

@ModelRegistry.register("xgboost")
class XGBoostStrategy(ModelStrategy):
    tier = 3; min_samples = 500
    def build(self, problem_type):
        return XGBClassifier(n_estimators=300, random_state=RANDOM_SEED, eval_metric="logloss")
```

### 3.3 AgentFactory (Factory Pattern)

```python
class AgentFactory:
    _agents: ClassVar[dict[str, type[BaseAgent]]] = {
        "data_understanding": DataUnderstandingAgent,
        "problem_framing": ProblemFramingAgent,
        "risk_failure": RiskFailureAgent,
        "preprocessing": PreprocessingAgent,
        "model_strategy": ModelStrategyAgent,
        "training": TrainingAgent,
        "evaluation": EvaluationAgent,
        "explainability": ExplainabilityAgent,
        "report_generator": ReportGeneratorAgent,
        "human_review": HumanReviewAgent,
    }

    @classmethod
    def create(cls, name: str) -> BaseAgent:
        if name not in cls._agents:
            raise ValueError(f"Unknown agent: {name}")
        return cls._agents[name]()
```

### 3.4 GenericRepository (Repository Pattern)

```python
class GenericRepository(Generic[T]):
    def __init__(self, model: type[T], session: AsyncSession):
        self.model = model; self.session = session

    async def get(self, id: UUID) -> Optional[T]:
        return await self.session.get(self.model, id)

    async def create(self, **kwargs) -> T:
        obj = self.model(**kwargs)
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def filter(self, **kwargs) -> list[T]:
        stmt = select(self.model).filter_by(**kwargs)
        return (await self.session.execute(stmt)).scalars().all()
```

---

## 4. Agent Specifications (Refined)

### Agent 1 — Data Understanding Agent

**Profile fields written to `state.data_profile`:**
- shape, dtypes, missing_rates (per column), duplicate_ratio
- constant_cols, mixed_dtype_cols, outlier_scores (IQR-based)
- class_distribution + imbalance_ratio (if target column available)
- inferred_target (from LLM + problem_statement)

**Risk flags emitted:**
| Code | Condition |
|------|-----------|
| `MISSING_COLUMN_NAMES` | Any column header is empty string |
| `HIGH_MISSING_RATE` | Any column > 50% null |
| `CONSTANT_COLUMN` | nunique ≤ 1 |
| `HIGH_DUPLICATE_RATIO` | duplicates > 20% |
| `TINY_DATASET` | n < 100 → `requires_human_approval=True` |
| `CLASS_IMBALANCE` | minority class < 10% → requires strategy confirmation |

---

### Agent 2 — Problem Framing Agent

**LLM call:** OpenAI structured output → `FramingDecision` (Pydantic model). Fallback: deterministic heuristics on column names + dtypes.

**Leakage detection (both checks mandatory):**
1. **Correlation check:** any feature with |ρ| > 0.95 with target
2. **Name-pattern check:** regex against leak patterns (`_label`, `_target`, `actual_`, `final_`)
3. **Temporal check:** datetime column values post-dating label generation date
4. **Encoded target check:** column that appears to be a direct encoding of the target

If any leakage: `confidence_level = 'unsafe'`. Hard stop. `LEAKAGE_SUSPECTED` flag. Human approval required before training.

**CV strategy selection written to `state.cv_strategy`:**
- Detects group columns (`user_id`, `store_id`, `customer_id`) → `GroupKFold`
- Detects timeseries (sorted datetime index + framing decision) → `TimeSeriesSplit`
- Classification default → `StratifiedKFold(n_splits=5)`, bump to 10 if n < 1000
- Regression default → `KFold(n_splits=5)`

---

### Agent 3 — Preprocessing Agent

**All preprocessing inside a sklearn Pipeline. Zero exceptions.**

**Numeric column decision logic:**
| Missing Rate | Strategy |
|---|---|
| < 5% | `SimpleImputer(strategy="median")` |
| 5–30% | `KNNImputer(n_neighbors=5)` + add `col_was_missing` binary flag column |
| > 30% | Emit `HIGH_MISSING_RATE` warning flag; recommend drop; require human confirmation |

**High skewness (|skew| > 1.5):** apply `FunctionTransformer(np.log1p)`. Guard: if min ≤ 0, use `log1p(x - min + 1)`.

**Near-perfect uniqueness (ID-like columns):** drop with trace log entry. Threshold: nunique/n > 0.95.

**Categorical column decision logic:**
| Cardinality | Strategy |
|---|---|
| ≤ 10 unique values | `OneHotEncoder(handle_unknown='ignore')` |
| 10–50 unique values | `OneHotEncoder(drop='if_binary', handle_unknown='ignore')` |
| > 50 unique values | Cross-validated target encoding (inside CV loop only — no leakage) |

At inference: unseen categories route to a pre-defined `unknown` fallback bucket (never raises).

SMOTE: apply only if imbalance_ratio > 10:1 AND human confirmed OR if `confidence_level == 'safe'`. Applied after split, never before.

---

### Agent 4 — Model Strategy Agent

**Dual dummy baselines (non-negotiable):**
- Classification: `DummyClassifier(strategy='most_frequent')` + `DummyClassifier(strategy='stratified')`
- Regression: `DummyRegressor(strategy='mean')` + `DummyRegressor(strategy='median')`

**Hard rules:**
- n < 100: cap at Logistic Regression only; no tree models; risk flag raised
- imbalance > 10:1: `class_weight='balanced'` enforced OR SMOTE (human confirms which)
- Training exceeds time budget (configurable, default 300s): fall back to prior tier; log timeout

---

### Agent 5 — Training Agent

**`RANDOM_SEED = 42` enforced in every model constructor. No exceptions.**

**MLflow per-run logging (child run per model candidate):**
- Input schema hash (col names + dtypes)
- Preprocessing pipeline hash (detects mid-run pipeline changes)
- Model type + all hyperparameter values
- CV fold scores: mean + std per fold (not just aggregate)
- Training wall-clock time + peak memory (via `tracemalloc`)
- Each Optuna trial logged as child run within parent MLflow run

**Optuna pruning:** `MedianPruner(n_startup_trials=5)` kills unpromising HP trials early.

**Post-CV leakage heuristic:** if any single feature importance > 50% of total → `DOMINANT_FEATURE` warning.

---

### Agent 6 — Evaluation Agent

**Pass/Fail gate (ALL must pass to avoid retry):**
1. Beats BOTH dummy baselines by meaningful margin (>5% relative)
2. Train-val gap: relative gap > 10% → overfitting warning; > 25% → fail gate
3. No catastrophic slice failures (any slice AUC < 0.55 for classification)
4. Calibration acceptable (ECE < 0.15)

**Classification metrics:** Accuracy, F1 (weighted + per-class), ROC-AUC (macro), ECE, confusion matrix, precision-recall curve.

**Regression metrics:** RMSE, MAE, R², residual plot (heteroscedasticity check).

**Slice analysis:** per-categorical-column, top-N values. Flags fairness/robustness issues invisible in aggregate. Logs to `eval_metrics["slice_analysis"]`.

**Post-evaluation risk checks (delegated to RiskFailureAgent):**
- AUC > 0.99 → `SUSPICIOUS_AUC` → `requires_human_approval=True`
- CV std > 0.15 → `UNSTABLE_CV`
- Peak memory > production budget (configurable) → `MEMORY_BUDGET`
- Dominant feature > 50% → `DOMINANT_FEATURE`
- Model ≤ dummy baseline → halt with `halt_reason`

---

### Agent 7 — Explainability Agent

**Explainer selection:**
- `TreeExplainer` for any `feature_importances_` model (fast, exact)
- `LinearExplainer` for linear models
- `KernelExplainer` only as last resort (slow, approximate — log warning)

**SHAP computed on `min(200, n_samples)` rows for latency budget.**

**Global importance:** mean |SHAP| per feature, top-10, sorted descending.

**Local explanations for 3 representative samples:**
1. High-confidence correct prediction
2. Low-confidence prediction (borderline case)
3. Incorrect prediction (model failure case)

**Correlated feature warning:** any pair with |ρ| > 0.7 → emit info flag noting SHAP splits importance across correlated features. Combined interpretation required.

**Business narrative:** LLM call using structured SHAP facts + problem_statement. No hallucination — prompt is fully grounded in state fields.

---

### Agent 8 — Risk & Failure Agent (3 Checkpoints)

**Checkpoint 1 — Pre-training:**
- n < 100: refuse; confidence_level = 'unsafe'; human gate
- imbalance > 20:1: warn; require explicit strategy confirmation
- Leakage detected: hard stop; list suspect features + reasoning
- All features > 50% missing: refuse; dataset unusable
- Constant target: refuse; no learnable signal
- Duplicates > 20%: warn; recommend deduplication

**Checkpoint 2 — Post-training (after evaluation):**
- Model ≤ dummy baseline: halt; problem framing issue
- CV std > 0.15: `UNSTABLE_CV`
- AUC > 0.99: `SUSPICIOUS_AUC`; human escalation
- Single feature > 50% importance: `DOMINANT_FEATURE`
- Peak memory > budget: `MEMORY_BUDGET`

**Checkpoint 3 — Schema drift (on every new dataset):**
- Schema fingerprint = `SHA-256(sorted(col:dtype pairs))`
- PSI per feature (10-bin histogram comparison vs prior run)
- New columns added: warn; log adaptation
- Columns removed: warn; check if removed column had high importance
- Dtype change (numeric → categorical): hard flag `SCHEMA_DTYPE_MISMATCH`; cannot adapt silently
- PSI > 0.2 for any feature: `DRIFT_DETECTED`; recommend retraining

**Risk flag schema:**
```python
{
  "level": "critical" | "warning" | "info",
  "code": "LEAKAGE_SUSPECTED" | "TINY_DATASET" | "DRIFT_DETECTED" | ...,
  "feature": "column_name",            # null if dataset-level
  "description": "Plain English",
  "recommended_action": "...",
  "requires_human_approval": True | False
}
```

---

### Agent 9 — Report Generator Agent

**Content sourced exclusively from `state.trace_log` + `state.*` fields. No LLM hallucination.**

Report includes:
- Executive summary (LLM-generated from structured facts only)
- Data profile table
- Problem framing decision + rationale
- Preprocessing decisions with per-step justification
- Model comparison table (all candidates + all metrics)
- Best model dashboard (calibration, confusion matrix, slice heatmap)
- SHAP top-10 + local examples + business narrative
- Risk flags with resolution status
- Deployment recommendation: `yes | no | conditional` + specific conditions
- What to do next checklist + model card

**Exports:** JSON (always), PDF (`reportlab`/`weasyprint`), HTML — all regeneratable from trace alone.

---

### Agent 10 — Human Review Agent (LangGraph Interrupt)

Interrupted by LangGraph `interrupt_before=["human_review"]`. Pipeline literally pauses.

**Approval gates (with required context):**
| Gate | Required fields |
|------|---|
| `leakage_feature_drop` | which feature, why suspect, impact assessment |
| `tiny_dataset_proceed` | user confirms data is correct + complete |
| `imbalance_strategy` | user confirms class_weight vs SMOTE for business context |
| `low_confidence_deploy` | user acknowledges risk level |
| `schema_drift_retrain` | user confirms schema change is intentional |
| `max_retries_exhausted` | user provides new strategy direction |

**Mandatory reason field:** approval submitted without `reason` field is rejected by system. Not optional.

---

## 5. LangGraph Orchestration

```python
def build_graph(checkpointer: PostgresSaver) -> CompiledGraph:
    graph = StateGraph(PipelineState)

    # Nodes
    for name in ["data_understanding","problem_framing","risk_check",
                 "human_review","preprocessing","model_strategy",
                 "training","evaluation","explainability","report_generator","halt"]:
        graph.add_node(name, AgentFactory.create(name).run)

    # Edges
    graph.set_entry_point("data_understanding")
    graph.add_edge("data_understanding", "problem_framing")
    graph.add_edge("problem_framing", "risk_check")

    graph.add_conditional_edges("risk_check", route_after_risk, {
        "human_review": "human_review",
        "preprocessing": "preprocessing",
        "halt": "halt",
    })
    graph.add_edge("human_review", "preprocessing")
    graph.add_edge("preprocessing", "model_strategy")
    graph.add_edge("model_strategy", "training")
    graph.add_edge("training", "evaluation")

    graph.add_conditional_edges("evaluation", route_after_eval, {
        "retry": "model_strategy",        # graph-level retry; not try/except
        "explainability": "explainability",
        "human_review": "human_review",
        "halt": "halt",
    })
    graph.add_edge("explainability", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]  # pipeline pauses; API resumes after approval
    )

def route_after_risk(state: PipelineState) -> str:
    if state.halt: return "halt"
    if any(f.requires_human_approval for f in state.risk_flags): return "human_review"
    return "preprocessing"

def route_after_eval(state: PipelineState) -> str:
    if state.halt: return "halt"
    if any(f.requires_human_approval for f in state.risk_flags): return "human_review"
    if state.eval_metrics.get("pass_gate") is False:
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            return "retry"
        return "human_review"
    return "explainability"
```

**Checkpoint persistence:** `PostgresSaver` saves state after every node. If worker crashes, graph resumes from last checkpoint on restart.

---

## 6. Database Schema + ORM

```python
# All tables use UUID PK, CASCADE deletes on session_id FK

class SessionORM(Base):
    __tablename__ = "sessions"
    id: UUID PK; created_at: TIMESTAMP; user_id: TEXT
    dataset_path: TEXT; problem_statement: TEXT

class SchemaVersionORM(Base):
    __tablename__ = "schema_versions"
    # Index: (session_id, recorded_at DESC) — fetch latest for PSI
    id: UUID PK; session_id: FK(sessions)
    fingerprint: TEXT; columns: JSONB; dtypes: JSONB
    psi_scores: JSONB; recorded_at: TIMESTAMP

class AgentDecisionORM(Base):
    __tablename__ = "agent_decisions"
    # Index: (session_id, agent_name) — trace timeline queries
    id: UUID PK; session_id: FK; agent_name: TEXT
    decision_key: TEXT; decision_value: JSONB
    rationale: TEXT; timestamp: TIMESTAMP

class ModelRunORM(Base):
    __tablename__ = "model_runs"
    # Index: (session_id, is_selected) — find winning model
    id: UUID PK; session_id: FK; mlflow_run_id: TEXT
    model_type: TEXT; hyperparameters: JSONB
    cv_scores: JSONB; eval_metrics: JSONB; is_selected: BOOLEAN

class RiskFlagORM(Base):
    __tablename__ = "risk_flags"
    # Index: (session_id, level) — sort by severity in UI
    id: UUID PK; session_id: FK; level: TEXT; code: TEXT
    description: TEXT; resolved: BOOLEAN; resolution: TEXT

class HumanApprovalORM(Base):
    __tablename__ = "human_approvals"
    id: UUID PK; session_id: FK; gate_name: TEXT
    approved: BOOLEAN; reason: TEXT (NOT NULL); approved_by: TEXT; approved_at: TIMESTAMP
```

**Migrations:** Alembic autogenerate. Always run `alembic upgrade head` in CI before tests.

**Query optimization:**
- All FK columns indexed
- JSONB columns use GIN indexes for `->` operator queries
- `agent_decisions` has composite index on `(session_id, agent_name)`
- Paginate `agent_decisions` in trace timeline API (cursor-based, not offset)

---

## 7. API Endpoints

| Method | Path | Rate Limit | Description |
|--------|------|-----------|-------------|
| POST | `/api/v1/auth/login` | 10/min | JWT login |
| POST | `/api/v1/auth/refresh` | 30/min | Refresh access token |
| POST | `/api/v1/sessions` | 20/min | Create session + upload CSV/Parquet |
| GET | `/api/v1/sessions` | 60/min | List user sessions |
| POST | `/api/v1/pipeline/run/{session_id}` | 5/min | Trigger async pipeline |
| GET | `/api/v1/pipeline/status/{session_id}` | 60/min | Poll status (or SSE stream) |
| GET | `/api/v1/approvals/pending/{session_id}` | 60/min | Pending gates with context |
| POST | `/api/v1/approvals/submit/{session_id}` | 20/min | Submit decision + resume graph |
| GET | `/api/v1/traces/{session_id}` | 60/min | Full trace timeline (paginated) |
| GET | `/api/v1/reports/{session_id}/json` | 30/min | JSON report |
| GET | `/api/v1/reports/{session_id}/pdf` | 10/min | PDF download |
| GET | `/api/v1/reports/{session_id}/html` | 30/min | HTML report |

**Pipeline status uses Server-Sent Events (SSE)** for real-time progress — no polling loop on client.

---

## 8. Security

| Threat | Control |
|--------|---------|
| SQL injection | SQLAlchemy ORM parameterized queries only |
| CSV formula injection | First-byte check for `=`, `+`, `-`, `@` |
| Path traversal | Uploads stored in `/uploads/{session_id}/dataset{ext}` — no user filename |
| Unauthorized cross-session access | `session_id + user_id` ownership check on every request |
| LLM prompt injection | All prompts use structured outputs (Pydantic); no raw user text injected |
| Brute force | Rate limits on auth endpoints; account lockout after 10 failures |
| Secrets | `.env` never committed; Pydantic Settings reads from env only |
| File type abuse | Allowlist: `.csv`, `.parquet` only; MIME type verified |

---

## 9. Phase-Wise Build Order

| Phase | Deliverable | Gate |
|-------|------------|------|
| 1 | Project scaffold, PipelineState, BaseAgent, DB migrations, structlog | All imports resolve; `alembic upgrade head` passes |
| 2 | DataUnderstanding + ProblemFraming + RiskFailure (pre-training) | Unit tests pass; all 6 DataAgent flags emit correctly |
| 3 | Preprocessing + ModelStrategy + Training | Sklearn pipeline serializable; MLflow run created per model |
| 4 | Evaluation (pass/fail gate) + RiskFailure (post-training) | Retry routing works in isolation; dummy baseline always present |
| 5 | LangGraph graph wiring + HumanReview interrupt | Graph halts at gate; resumes correctly after approval |
| 6 | Explainability + ReportGenerator | JSON report regeneratable from trace alone |
| 7 | FastAPI API + Celery + SSE | `/pipeline/run` returns task_id; `/status` streams progress |
| 8 | Frontend (TraceTimeline + ApprovalForm + SHAP panel) | All 5 demo scenarios visible in UI |
| 9 | DataCorruptor, E2E tests, Docker, CI/CD | 80% unit coverage; all 5 E2E scenarios pass |

---

## 10. Testing Strategy

### Unit Tests (per agent, no external services)
```bash
pytest tests/unit/ -v --cov=app --cov-report=term-missing --cov-fail-under=80
```
- Each agent tested with mocked `PipelineState` and fixture DataFrames
- All 8 `DataCorruptor` methods tested individually
- PSI and schema fingerprint utilities tested with known inputs

### DataCorruptor Integration Matrix
| Method | Expected Flag | Expected Agent |
|--------|--------------|----------------|
| `missing_column_names` | `MISSING_COLUMN_NAMES` | DataUnderstanding |
| `duplicate_rows(frac=0.25)` | `HIGH_DUPLICATE_RATIO` | DataUnderstanding |
| `target_leakage_inject` | `LEAKAGE_SUSPECTED` | ProblemFraming |
| `dtype_change` | `SCHEMA_DTYPE_MISMATCH` | RiskFailure |
| `null_burst(frac=0.4)` | handled (no halt) | Preprocessing |
| `unseen_categories` | handled (unknown bucket) | Preprocessing |
| `extreme_outliers` | outlier score logged | DataUnderstanding |
| `changed_column_order` | fingerprint mismatch | RiskFailure |

### E2E Scenarios
```bash
pytest tests/e2e/ -v --timeout=120
```
1. Clean classification → full pipeline → PDF generated, no flags
2. Messy dataset (40% null) → KNNImputer selected → no halt
3. Time-series tabular → `TimeSeriesSplit` enforced → random split refused
4. Schema-changed dataset → PSI alert → human gate triggered
5. Tiny dataset (n=50) → `TINY_DATASET` → `confidence_level=unsafe` → human gate

---

## 11. Deployment

```yaml
# infra/docker-compose.yml (dev)
services:
  api:       # FastAPI on :8000 with --reload
  worker:    # Celery 4-concurrent workers
  postgres:  # pg:16-alpine, volume-persisted
  redis:     # redis:7-alpine
  mlflow:    # mlflow:2.11 → pg backend
  frontend:  # React dev server :3000
```

**CI/CD (GitHub Actions):**
1. `ci.yml`: ruff, mypy, unit tests (pg + redis services), coverage gate ≥ 80%
2. `deploy.yml`: build + push Docker images on `main` merge; roll out to target env

**Observability stack:**
- Structlog → stdout → Loki aggregation
- Prometheus metrics on `/metrics` (HTTP latency, queue depth, error rate)
- Grafana dashboards for above
- Sentry for error tracking with `session_id` as tag
- Flower UI for Celery task monitoring

**Scaling:**
- Add Celery worker replicas horizontally (stateless, all state in PostgreSQL + Redis)
- LangGraph checkpoint in PostgreSQL enables worker crash recovery
- Redis caches PSI/fingerprint results per `dataset_hash` (TTL 1h)
- FAISS index for similar-experiment retrieval (embeddings of past `data_profile` dicts)

---

> [!IMPORTANT]
> **Non-negotiables from blueprint — must exist at launch:**
> sklearn Pipeline for ALL preprocessing · Dual dummy baselines always run · 3-checkpoint Risk Agent · Graph-level retry (not try/except) · LangGraph interrupt gate with mandatory reason · MLflow child run per agent · RANDOM_SEED=42 everywhere · Structured typed agent outputs (Pydantic) · Slice analysis in Evaluation · DataCorruptor module · Trace regeneratable report
