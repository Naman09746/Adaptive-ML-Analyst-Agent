# AMA² — Adaptive ML Analyst Agent
## Complete Implementation Plan (v2.0)

> **Status:** Scaffold complete (Phase 1 done). This document is the authoritative roadmap from scaffold → production-ready demo.

---

## Table of Contents

1. [Project Goal & North Star](#1-project-goal--north-star)
2. [What Is Already Built](#2-what-is-already-built)
3. [Gap Analysis](#3-gap-analysis)
4. [Folder Structure (Target State)](#4-folder-structure-target-state)
5. [Central Data Contract (PipelineState)](#5-central-data-contract-pipelinestate)
6. [OOP Design Patterns](#6-oop-design-patterns)
7. [Agent Specifications](#7-agent-specifications)
8. [LangGraph Orchestration](#8-langgraph-orchestration)
9. [Database Schema & ORM](#9-database-schema--orm)
10. [API Endpoints](#10-api-endpoints)
11. [Security Controls](#11-security-controls)
12. [Testing Strategy](#12-testing-strategy)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Phased Build Order](#14-phased-build-order)
15. [Non-Negotiables at Launch](#15-non-negotiables-at-launch)

---

## 1. Project Goal & North Star

**What AMA² is:** A multi-agent ML operations system that takes a raw tabular CSV/Parquet dataset and a plain-English problem statement, then autonomously:
1. Profiles the data and raises risk flags
2. Infers the ML task type (classification / regression / timeseries)
3. Selects and builds an adaptive preprocessing pipeline
4. Trains and tunes a ranked set of model candidates with Optuna + MLflow
5. Evaluates with slice analysis, calibration checks, and pass/fail gates
6. Generates SHAP explanations and a business narrative
7. Emits a full JSON/HTML/PDF report — regeneratable from trace alone
8. Provides a human approval gate for every safety-critical decision

**What makes this non-trivial:**
- Typed, schema-validated agent handoffs (Pydantic) — never free text
- Three-checkpoint Risk Agent: pre-training, post-training, schema drift
- Graph-level retry routing via LangGraph — not try/except hacks
- LangGraph interrupt gates that literally pause the pipeline for human decisions
- Audit-first design: every decision in PostgreSQL + MLflow; report regeneratable from trace
- DataCorruptor module for demo reproducibility and regression testing

---

## 2. What Is Already Built

| File | Status | Notes |
|---|---|---|
| `backend/app/core/pipeline_state.py` | ✅ Complete | Typed PipelineState dataclass |
| `backend/app/agents/base.py` | ✅ Complete | BaseAgent with MLflow + structlog |
| `backend/app/agents/data_understanding.py` | ✅ Complete | Calls real data_inspection logic |
| `backend/app/agents/problem_framing.py` | ✅ Complete | Infers target, CV strategy, leakage |
| `backend/app/agents/preprocessing.py` | ✅ Complete | Builds sklearn ColumnTransformer |
| `backend/app/agents/model_strategy.py` | ✅ Complete | ModelRegistry + tiered candidate selection |
| `backend/app/ml/data_inspection.py` | ✅ Complete | profiling, risk signals, CV strategy, leakage |
| `backend/app/ml/model_registry.py` | ✅ Complete | Logistic, RandomForest, XGBoost strategies |
| `backend/app/db/models/models.py` | ✅ Complete | Sessions, Decisions, ModelRuns, RiskFlags, Approvals |
| `backend/app/db/repositories/generic.py` | ✅ Complete | Async GenericRepository[T] |
| `backend/app/core/agent_factory.py` | ✅ Complete | Dynamic import + alias resolution |
| `backend/app/core/constants.py` | ✅ Complete | All risk codes, problem types, seeds |
| `backend/app/config.py` | ✅ Complete | Pydantic Settings + production validation |
| `backend/app/utils/logging.py` | ✅ Complete | structlog JSON renderer |
| `backend/app/main.py` | ⚠️ Scaffold | Only /health endpoint. Needs real routers |

---

## 3. Gap Analysis

These are the **exact gaps** between current state and a complete, demo-ready system:

### Missing Agents (Code Not Written)
- `agents/training.py` — CV loop, Optuna tuning, MLflow per-run
- `agents/evaluation.py` — Metrics, pass/fail gate, slice analysis
- `agents/risk_failure.py` — 3-checkpoint risk logic (pre, post, drift)
- `agents/explainability.py` — SHAP tree/linear/kernel selector
- `agents/report_generator.py` — JSON/HTML/PDF report from trace
- `agents/human_review.py` — LangGraph interrupt gate stub

### Missing ML Modules
- `ml/trainer.py` — CV strategy factory + Optuna loop
- `ml/evaluator.py` — Metrics computation + slice analysis + pass/fail gate
- `ml/explainer.py` — SHAP wrapper (TreeExplainer / LinearExplainer / KernelExplainer)
- `ml/calibrator.py` — Platt/isotonic calibration
- `ml/data_corruptor.py` — 8 injection methods for test scenarios

### Missing Core Modules
- `core/orchestrator.py` — LangGraph graph + routing functions
- `core/exceptions.py` — Domain exception hierarchy

### Missing Database & Persistence
- `db/session.py` — async_sessionmaker factory
- `db/repositories/` — Per-entity repositories (session, model_run, etc.)
- `alembic/` — Migration files; no `alembic upgrade head` works yet

### Missing API Layer
- `api/v1/router.py` — Aggregate router
- `api/v1/sessions.py` — Create session, upload dataset
- `api/v1/pipeline.py` — /run + /status (SSE stream)
- `api/v1/approvals.py` — /pending + /submit endpoints
- `api/v1/traces.py` — Paginated trace timeline
- `api/v1/reports.py` — JSON/PDF/HTML download
- `schemas/` — Pydantic request/response schemas
- `dependencies.py` — DB session DI, current_user, Redis DI

### Missing Utilities
- `utils/schema_fingerprint.py` — SHA-256 of sorted col:dtype pairs
- `utils/psi.py` — Population Stability Index (10-bin histogram)
- `utils/retry.py` — Exponential backoff decorator
- `utils/tracing.py` — MLflow child-run helpers

### Missing Infrastructure
- `Dockerfile` (backend)
- `infra/docker-compose.yml`
- `infra/docker-compose.prod.yml`
- `.github/workflows/ci.yml`
- `.env.example`

### Missing Tests
- `tests/unit/` — Agent, ML module, utility tests
- `tests/integration/` — DB repository, state transition tests
- `tests/e2e/` — 5 blueprint demo scenarios end-to-end

### Missing Frontend
- React dashboard with TraceTimeline, RiskPanel, ApprovalForm, ShapChart

---

## 4. Folder Structure (Target State)

```
ama2/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app factory + lifespan [EXISTS - needs routers]
│   │   ├── config.py                  # Pydantic Settings               [EXISTS - complete]
│   │   ├── dependencies.py            # DB session, current_user DI     [MISSING]
│   │   ├── core/
│   │   │   ├── pipeline_state.py      # PipelineState dataclass         [EXISTS - complete]
│   │   │   ├── orchestrator.py        # LangGraph graph builder         [MISSING]
│   │   │   ├── agent_factory.py       # AgentFactory                    [EXISTS - complete]
│   │   │   ├── exceptions.py          # Domain exception hierarchy      [MISSING]
│   │   │   └── constants.py           # Risk codes, seeds, thresholds   [EXISTS - complete]
│   │   ├── agents/
│   │   │   ├── base.py                # BaseAgent ABC                   [EXISTS - complete]
│   │   │   ├── data_understanding.py  #                                 [EXISTS - complete]
│   │   │   ├── problem_framing.py     #                                 [EXISTS - complete]
│   │   │   ├── risk_failure.py        # 3-checkpoint risk agent         [MISSING]
│   │   │   ├── preprocessing.py       # sklearn ColumnTransformer       [EXISTS - complete]
│   │   │   ├── model_strategy.py      # ModelRegistry integration       [EXISTS - complete]
│   │   │   ├── training.py            # CV + Optuna + MLflow            [MISSING]
│   │   │   ├── evaluation.py          # Metrics + pass/fail gate        [MISSING]
│   │   │   ├── explainability.py      # SHAP selector                   [MISSING]
│   │   │   ├── report_generator.py    # JSON/HTML/PDF from trace        [MISSING]
│   │   │   └── human_review.py        # LangGraph interrupt stub        [MISSING]
│   │   ├── ml/
│   │   │   ├── data_inspection.py     # Profile, risk signals, leakage  [EXISTS - complete]
│   │   │   ├── model_registry.py      # ModelStrategy ABC + impls       [EXISTS - complete]
│   │   │   ├── trainer.py             # CV strategy factory + Optuna    [MISSING]
│   │   │   ├── evaluator.py           # Metrics + slice analysis        [MISSING]
│   │   │   ├── explainer.py           # SHAP wrappers                   [MISSING]
│   │   │   ├── calibrator.py          # Platt/isotonic calibration      [MISSING]
│   │   │   └── data_corruptor.py      # 8 injection methods             [MISSING]
│   │   ├── db/
│   │   │   ├── base.py                # SQLAlchemy DeclarativeBase       [EXISTS]
│   │   │   ├── session.py             # async_sessionmaker factory       [MISSING]
│   │   │   ├── models/
│   │   │   │   └── models.py          # All ORM tables                   [EXISTS - complete]
│   │   │   └── repositories/
│   │   │       ├── generic.py         # GenericRepository[T]            [EXISTS - complete]
│   │   │       ├── session_repo.py    # SessionRepository               [MISSING]
│   │   │       ├── model_run_repo.py  # ModelRunRepository              [MISSING]
│   │   │       └── risk_flag_repo.py  # RiskFlagRepository              [MISSING]
│   │   ├── api/v1/
│   │   │   ├── router.py              # Aggregate all sub-routers        [MISSING]
│   │   │   ├── sessions.py            # POST /sessions, GET /sessions    [MISSING]
│   │   │   ├── pipeline.py            # /run, /status (SSE)             [MISSING]
│   │   │   ├── approvals.py           # /pending, /submit               [MISSING]
│   │   │   ├── reports.py             # /json, /pdf, /html              [MISSING]
│   │   │   └── traces.py              # Paginated trace timeline         [MISSING]
│   │   ├── schemas/
│   │   │   ├── session.py             # SessionCreate, SessionOut        [MISSING]
│   │   │   ├── pipeline.py            # RunRequest, StatusResponse       [MISSING]
│   │   │   ├── approval.py            # ApprovalRequest, GateContext     [MISSING]
│   │   │   └── report.py              # ReportOut schema                 [MISSING]
│   │   ├── services/
│   │   │   ├── pipeline_service.py    # Celery task dispatch             [MISSING]
│   │   │   ├── report_service.py      # Report file generation          [MISSING]
│   │   │   └── file_service.py        # Upload + path sanitization       [MISSING]
│   │   └── utils/
│   │       ├── logging.py             # structlog JSON renderer          [EXISTS - complete]
│   │       ├── schema_fingerprint.py  # SHA-256 col:dtype fingerprint   [MISSING]
│   │       ├── psi.py                 # PSI (Population Stability Index) [MISSING]
│   │       └── retry.py               # Exponential backoff decorator    [MISSING]
│   ├── alembic/                       # DB migrations                    [MISSING]
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── agents/               # Per-agent unit tests             [MISSING]
│   │   │   ├── ml/                   # ML module unit tests             [MISSING]
│   │   │   └── utils/                # Utility unit tests               [MISSING]
│   │   ├── integration/               # DB + state transition tests      [MISSING]
│   │   └── e2e/                       # 5 blueprint demo scenarios       [MISSING]
│   ├── Dockerfile                     #                                  [MISSING]
│   ├── .env.example                   #                                  [MISSING]
│   └── pyproject.toml                 # [EXISTS - dependencies defined]
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx          # Session list + create            [MISSING]
│   │   │   ├── SessionDetail.tsx      # Pipeline status + trace          [MISSING]
│   │   │   ├── ApprovalGate.tsx       # Human approval UI               [MISSING]
│   │   │   └── ReportViewer.tsx       # Rendered report                  [MISSING]
│   │   ├── components/
│   │   │   ├── TraceTimeline.tsx      # Chronological decision log       [MISSING]
│   │   │   ├── RiskPanel.tsx          # Risk flags with severity         [MISSING]
│   │   │   ├── ModelTable.tsx         # Model comparison table           [MISSING]
│   │   │   ├── ShapChart.tsx          # SHAP summary bar chart          [MISSING]
│   │   │   └── ApprovalForm.tsx       # Gate form with required reason   [MISSING]
│   │   ├── hooks/                     # useSSE, useSession, useApproval  [MISSING]
│   │   ├── api/                       # Typed axios client + zod        [MISSING]
│   │   └── store/                     # Zustand state management         [MISSING]
│   └── Dockerfile                     #                                  [MISSING]
├── infra/
│   ├── docker-compose.yml             # Dev: api, worker, pg, redis, mlflow, frontend
│   ├── docker-compose.prod.yml        # Prod: replicas + nginx
│   └── nginx/nginx.conf               # Reverse proxy config
└── .github/workflows/
    ├── ci.yml                         # ruff, mypy, pytest, coverage gate
    └── deploy.yml                     # Docker build + push on main merge
```

---

## 5. Central Data Contract (PipelineState)

The `PipelineState` dataclass is the **single source of truth** shared across all agents. It is already implemented and complete. All agents must obey the **write ownership** rule below.

```
Agent                 | Writes to
----------------------|--------------------------------------------------
DataUnderstanding     | df, data_profile, risk_flags, confidence_level
ProblemFraming        | target_column, problem_type, cv_strategy,
                      | group_column, leakage_suspects, risk_flags,
                      | confidence_level
RiskFailure           | risk_flags, halt, halt_reason,
                      | pending_approval_gates, confidence_level
Preprocessing         | preprocessing_plan, sklearn_pipeline,
                      | X_train, X_test, y_train, y_test
ModelStrategy         | model_candidates
Training              | best_model, best_model_name, eval_metrics,
                      | mlflow_run_id, retry_count
Evaluation            | eval_metrics (extends), risk_flags (post-training)
Explainability        | shap_values, business_narrative
ReportGenerator       | report_path
HumanReview           | human_approvals, pending_approval_gates (clears)
```

> **Law:** No agent overwrites another agent's designated fields. Violations must fail hard.

---

## 6. OOP Design Patterns

### 6.1 BaseAgent — Template Method (ALREADY IMPLEMENTED)

```python
class BaseAgent(ABC):
    def run(self, state: PipelineState) -> PipelineState:
        """Template: wraps _execute with structlog, MLflow child run, error capture."""
        t0 = time.perf_counter()
        bind_contextvars(session_id=str(state.session_id))
        with mlflow.start_run(run_name=self.name, nested=True):
            try:
                state = self._execute(state)
                mlflow.log_metric("latency_s", time.perf_counter() - t0)
                mlflow.set_tag("status", "success")
            except Exception as e:
                mlflow.set_tag("status", "failed")
                raise
        return state

    @abstractmethod
    def _execute(self, state: PipelineState) -> PipelineState: ...
```

### 6.2 ModelStrategy — Strategy Pattern (ALREADY IMPLEMENTED)

Logistic (tier 1), RandomForest (tier 2, n≥200), XGBoost (tier 3, n≥500).
Extendable by registering a new `@ModelRegistry.register("name")` class.

### 6.3 AgentFactory — Factory Pattern (ALREADY IMPLEMENTED)

Dynamic import with module path + class name. Raises `NotImplementedError` for missing agents gracefully.

### 6.4 GenericRepository — Repository Pattern (ALREADY IMPLEMENTED)

Async SQLAlchemy repository with `get_by_id`, `create`, `filter`, `delete`.

---

## 7. Agent Specifications

### Agent 1 — DataUnderstandingAgent ✅ IMPLEMENTED

Reads: `dataset_path`, `problem_statement`
Writes: `df`, `data_profile`, `risk_flags`, `confidence_level`

Risk flags emitted:
| Code | Trigger |
|---|---|
| `MISSING_COLUMN_NAMES` | Any column header is blank |
| `HIGH_DUPLICATE_RATIO` | Duplicate rows > 20% |
| `TINY_DATASET` | n < 100 → `requires_human_approval=True` |
| `HIGH_MISSING_RATE` | Any column > 50% null |
| `CONSTANT_COLUMN` | nunique ≤ 1 |
| `CLASS_IMBALANCE` | Minority class < 10% |

---

### Agent 2 — ProblemFramingAgent ✅ IMPLEMENTED

Reads: `df`, `problem_statement`, `target_column` (optional)
Writes: `target_column`, `problem_type`, `cv_strategy`, `group_column`, `leakage_suspects`, `confidence_level`, `risk_flags`

Leakage detection:
1. Correlation check: |ρ| > 0.95 with target
2. Name-pattern check: regex against `_label$`, `_target$`, `actual_`, `final_`, `leak`, `outcome`
3. Encoded target check: column values match target > 99%

CV strategy selection:
- Group column detected → `GroupKFold`
- Datetime column / timeseries framing → `TimeSeriesSplit`
- Classification → `StratifiedKFold(n_splits=5)`
- Regression → `KFold(n_splits=5)`

---

### Agent 3 — RiskFailureAgent ❌ MISSING

**File to create:** `backend/app/agents/risk_failure.py`

Reads: `state.*` (read-only across all fields)
Writes: `risk_flags`, `halt`, `halt_reason`, `pending_approval_gates`, `confidence_level`

**Checkpoint 1 — Pre-training (called after ProblemFraming):**
```python
def _pre_training_check(self, state):
    if len(state.df) < 100:
        state.halt = True
        state.halt_reason = "Dataset too small for reliable training (n < 100)"
        state.risk_flags.append(RiskFlag(level="critical", code=TINY_DATASET, ...))

    if state.leakage_suspects:
        state.halt = True
        state.halt_reason = "Leakage features detected — cannot train safely"

    if all(r > 0.5 for r in state.data_profile["missing_rates"].values()):
        state.halt = True
        state.halt_reason = "All features have > 50% missing values"

    if state.data_profile.get("constant_target"):
        state.halt = True
        state.halt_reason = "Target column has no learnable variation"
```

**Checkpoint 2 — Post-training (called after Evaluation):**
```python
def _post_training_check(self, state):
    metrics = state.eval_metrics

    if not metrics.get("beats_dummy_baseline", True):
        state.halt = True
        state.halt_reason = "Model does not beat dummy baseline"

    if metrics.get("cv_std", 0) > 0.15:
        state.risk_flags.append(RiskFlag(level="warning", code="UNSTABLE_CV", ...))

    if metrics.get("roc_auc", 0) > 0.99:
        state.risk_flags.append(RiskFlag(level="critical", code="SUSPICIOUS_AUC",
                                          requires_human_approval=True, ...))

    if metrics.get("dominant_feature_ratio", 0) > 0.5:
        state.risk_flags.append(RiskFlag(level="warning", code="DOMINANT_FEATURE", ...))
```

**Checkpoint 3 — Schema drift (on new dataset vs prior run):**
- Compute `SHA-256(sorted(col:dtype))` fingerprint.
- Compute PSI per feature vs prior run baseline.
- Dtype change (numeric → categorical): `SCHEMA_DTYPE_MISMATCH` → critical flag.
- PSI > 0.2 for any feature: `DRIFT_DETECTED` → warning flag.
- New/removed columns: warn + log.

---

### Agent 4 — PreprocessingAgent ✅ IMPLEMENTED

Reads: `df`, `data_profile`, `target_column`
Writes: `preprocessing_plan`, `sklearn_pipeline`, `X_train`, `X_test`, `y_train`, `y_test`

**Enhancement needed:** Currently builds a simple `ColumnTransformer` but does not yet perform the train/test split. The agent must:
1. Drop leakage suspects from feature set.
2. Perform `train_test_split(test_size=0.2, stratify=y if classification)`.
3. Build the `ColumnTransformer` on `X_train` column list.
4. Write `X_train`, `X_test`, `y_train`, `y_test` to state.

Advanced preprocessing rules (see IMPLEMENTATION_PLAN v1):
- Missing > 5%–30%: use `KNNImputer` + add `col_was_missing` binary flag
- High skewness (|skew| > 1.5): `log1p` transform
- ID-like columns (nunique/n > 0.95): drop with trace entry
- Cardinality > 50: target encoding (inside CV only)

---

### Agent 5 — ModelStrategyAgent ✅ IMPLEMENTED

Reads: `df`, `problem_type`, `model_candidates` (none yet)
Writes: `model_candidates`

**Enhancement needed:**
- Always include dual dummy baselines as mandatory first candidates:
  - Classification: `DummyClassifier(strategy='most_frequent')` + `DummyClassifier(strategy='stratified')`
  - Regression: `DummyRegressor(strategy='mean')` + `DummyRegressor(strategy='median')`
- Hard rule: n < 100 → cap at Logistic only; no tree models.
- Imbalance > 10:1 → enforce `class_weight='balanced'` on all candidate estimators.

---

### Agent 6 — TrainingAgent ❌ MISSING

**File to create:** `backend/app/agents/training.py`
**Helper to create:** `backend/app/ml/trainer.py`

Reads: `X_train`, `y_train`, `model_candidates`, `cv_strategy`, `problem_type`, `sklearn_pipeline`
Writes: `best_model`, `best_model_name`, `eval_metrics` (cv scores), `mlflow_run_id`

**Trainer logic (`ml/trainer.py`):**
```python
class Trainer:
    def run(self, state: PipelineState) -> dict:
        """Runs CV + Optuna tuning for each candidate. Returns best metrics."""
        cv = self._get_cv_splitter(state.cv_strategy)
        results = []

        for candidate in state.model_candidates:
            with mlflow.start_run(run_name=candidate["name"], nested=True) as run:
                # Log input schema hash
                mlflow.log_param("schema_hash", compute_fingerprint(state.df))

                # Build full pipeline: preprocessing + estimator
                full_pipeline = Pipeline([
                    ("preprocessor", state.sklearn_pipeline),
                    ("model", candidate["estimator"])
                ])

                # Cross-validation
                cv_scores = cross_validate(
                    full_pipeline, X, y, cv=cv,
                    scoring=self._get_scoring(state.problem_type),
                    return_train_score=True
                )

                # Optuna hyperparameter tuning
                best_params = self._optuna_tune(
                    full_pipeline, X, y, cv,
                    candidate["param_grid"],
                    n_trials=30
                )

                # Log per-fold scores (not just aggregate)
                for fold_i, score in enumerate(cv_scores["test_score"]):
                    mlflow.log_metric("cv_score", score, step=fold_i)
                mlflow.log_metric("cv_mean", cv_scores["test_score"].mean())
                mlflow.log_metric("cv_std", cv_scores["test_score"].std())

                results.append({
                    "name": candidate["name"],
                    "cv_mean": cv_scores["test_score"].mean(),
                    "cv_std": cv_scores["test_score"].std(),
                    "best_params": best_params,
                    "mlflow_run_id": run.info.run_id,
                })

        # Select best by cv_mean, excluding candidates that don't beat both dummies
        dummy_scores = [r for r in results if "dummy" in r["name"].lower()]
        dummy_baseline = max(d["cv_mean"] for d in dummy_scores) if dummy_scores else 0.0

        real_models = [r for r in results if "dummy" not in r["name"].lower()]
        beats_baseline = [r for r in real_models if r["cv_mean"] > dummy_baseline * 1.05]

        if not beats_baseline:
            return {"pass_gate": False, "beats_dummy_baseline": False, "all_results": results}

        best = max(beats_baseline, key=lambda r: r["cv_mean"])
        return {**best, "pass_gate": True, "beats_dummy_baseline": True, "all_results": results}
```

**MLflow logging requirements per training run:**
- Input schema hash (`SHA-256(sorted(col:dtype))`)
- Preprocessing pipeline hash
- Model type + all hyperparameter values
- CV fold scores: mean + std per fold
- Training wall-clock time + peak memory (`tracemalloc`)
- Each Optuna trial logged as MLflow child run
- `RANDOM_SEED = 42` enforced in every model constructor

**Optuna pruning:** `MedianPruner(n_startup_trials=5)` kills unpromising HP trials early.

**Post-CV leakage heuristic:** If any single feature importance > 50% of total → emit `DOMINANT_FEATURE` warning.

---

### Agent 7 — EvaluationAgent ❌ MISSING

**File to create:** `backend/app/agents/evaluation.py`
**Helper to create:** `backend/app/ml/evaluator.py`

Reads: `best_model`, `X_test`, `y_test`, `problem_type`, `df`, `target_column`
Writes: `eval_metrics` (extended with hold-out metrics + slice analysis)

**Evaluator logic (`ml/evaluator.py`):**

**Classification metrics:**
```python
{
  "accuracy": accuracy_score(y_test, y_pred),
  "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
  "f1_per_class": f1_score(y_test, y_pred, average=None).tolist(),
  "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr"),
  "ece": expected_calibration_error(y_test, y_prob),
  "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
  "precision_recall_curve": {...},
  "cv_mean": <from training>,
  "cv_std": <from training>,
}
```

**Regression metrics:**
```python
{
  "rmse": mean_squared_error(y_test, y_pred, squared=False),
  "mae": mean_absolute_error(y_test, y_pred),
  "r2": r2_score(y_test, y_pred),
  "residual_heteroscedasticity": <Breusch-Pagan p-value>,
}
```

**Slice analysis (mandatory):**
```python
def compute_slice_analysis(df_test, y_test, y_pred, categorical_cols):
    """Per-column, per-value slice metrics. Flags AUC < 0.55 on any slice."""
    slice_results = {}
    for col in categorical_cols:
        for val in df_test[col].unique():
            mask = df_test[col] == val
            if mask.sum() < 10:
                continue
            slice_auc = roc_auc_score(y_test[mask], y_pred[mask])
            slice_results[f"{col}={val}"] = {"auc": slice_auc, "n": int(mask.sum())}
            if slice_auc < 0.55:
                # Emit SLICE_FAILURE risk flag
                ...
    return slice_results
```

**Pass/Fail Gate (ALL must pass to proceed):**
1. Beats BOTH dummy baselines by > 5% relative margin.
2. Train-validation gap: > 10% → overfitting warning; > 25% → fail gate.
3. No slice AUC < 0.55 for classification (any slice failure = fail gate).
4. Calibration ECE < 0.15.
5. If fail gate → increment `state.retry_count`; if exhausted → human gate.

---

### Agent 8 — ExplainabilityAgent ❌ MISSING

**File to create:** `backend/app/agents/explainability.py`
**Helper to create:** `backend/app/ml/explainer.py`

Reads: `best_model`, `X_train`, `X_test`, `problem_type`
Writes: `shap_values`, `business_narrative`

**Explainer selection logic:**
```python
def get_explainer(model, X_background):
    if hasattr(model, "feature_importances_"):
        return shap.TreeExplainer(model)
    elif hasattr(model, "coef_"):
        return shap.LinearExplainer(model, X_background)
    else:
        logger.warning("Using KernelExplainer — this is slow")
        return shap.KernelExplainer(model.predict_proba, X_background[:50])
```

**SHAP computation:** On `min(200, n_test)` rows for latency budget.

**Global importance:** `mean(|SHAP values|)` per feature, top-10, sorted descending.

**Local explanations for 3 representative samples:**
1. Highest confidence correct prediction.
2. Lowest confidence prediction (borderline).
3. An incorrect prediction (model failure case).

**Correlated feature warning:** Any pair with |ρ| > 0.7 → emit info flag: "SHAP splits importance across correlated features."

**Business narrative:** Template-based string constructed from structured SHAP facts + state fields. No LLM hallucination risk — fully grounded. (Optional: LLM call with strict structured prompt if OpenAI key provided.)

---

### Agent 9 — ReportGeneratorAgent ❌ MISSING

**File to create:** `backend/app/agents/report_generator.py`
**Helper to create:** `backend/app/services/report_service.py`

Reads: `state.*` — all fields
Writes: `report_path`

**Report structure:**
```
1. Executive Summary
   - Dataset: {rows} rows × {columns} columns
   - Problem type: {classification/regression/timeseries}
   - Best model: {name} with CV AUC {cv_mean:.3f} ± {cv_std:.3f}
   - Risk level: {confidence_level}
   - Report generated: {timestamp}

2. Data Profile Table
   - Shape, dtypes, missing rates per column, duplicate ratio

3. Problem Framing
   - Target column + rationale
   - CV strategy + rationale
   - Leakage suspects (if any) + resolution

4. Preprocessing Decisions
   - Per-column: imputation strategy, encoding, scaling
   - Dropped columns + reason

5. Model Comparison Table
   - All candidates, CV mean ± std, hold-out metrics, training time
   - Dummy baseline highlighted as reference

6. Best Model Dashboard
   - Classification: confusion matrix, ROC curve, calibration plot
   - Regression: residual plot, actual vs predicted

7. Slice Analysis Heatmap
   - Per-categorical column AUC across top-N categories

8. Explainability
   - SHAP top-10 global importance bar chart
   - 3 local explanations (waterfall)
   - Business narrative

9. Risk Flags
   - All emitted flags, level, description, resolution status
   - Human approvals granted (if any)

10. Deployment Recommendation
    - "Recommended" / "Conditional" / "Not recommended"
    - Specific conditions for production use

11. Model Card
    - Training data summary, feature list, limitations, monitoring checklist
```

**Export formats:**
- `JSON`: Always generated first (machine-readable, reproducible from trace).
- `HTML`: Jinja2 template with inline CSS + base64-encoded charts.
- `PDF`: WeasyPrint from HTML (avoids LaTeX dependency).

**Invariant:** Report must be fully regeneratable from `state.trace_log` + `state.eval_metrics` alone, without re-running the pipeline.

---

### Agent 10 — HumanReviewAgent ❌ MISSING

**File to create:** `backend/app/agents/human_review.py`

This agent is a LangGraph interrupt stub — it does not execute logic itself. When the graph routes to this node, LangGraph calls `interrupt_before=["human_review"]` which pauses the graph. The FastAPI approval endpoint resumes it by injecting a decision into the checkpoint.

```python
class HumanReviewAgent(BaseAgent):
    def _execute(self, state: PipelineState) -> PipelineState:
        """
        This node is reached only after LangGraph resumes from interrupt.
        It validates the human approval and clears the pending gate.
        """
        for gate in list(state.pending_approval_gates):
            approval = state.human_approvals.get(gate)
            if approval is None:
                raise ValueError(f"Gate '{gate}' requires human approval but none provided")
            if not approval.get("reason"):
                raise ValueError(f"Approval for '{gate}' must include a written reason")
            state.pending_approval_gates.remove(gate)

        self._log_decision(state, "human_review_complete",
                           list(state.human_approvals.keys()),
                           "Human approvals validated; pipeline resuming.")
        return state
```

**Approval gates that trigger this node:**
| Gate Name | Trigger |
|---|---|
| `leakage_feature_drop` | Leakage suspect detected |
| `tiny_dataset_proceed` | n < 100 |
| `imbalance_strategy` | Imbalance > 10:1 |
| `suspicious_auc` | AUC > 0.99 post-training |
| `schema_drift_retrain` | Schema fingerprint changed |
| `max_retries_exhausted` | Eval retry limit hit |
| `low_confidence_deploy` | confidence_level = 'uncertain' at report time |

**Mandatory:** The `reason` field in every approval is validated as non-empty. The API rejects submissions without it.

---

## 8. LangGraph Orchestration

**File to create:** `backend/app/core/orchestrator.py`

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from .pipeline_state import PipelineState
from .agent_factory import AgentFactory
from .constants import CONFIDENCE_UNSAFE

def build_graph(checkpointer: PostgresSaver) -> CompiledGraph:
    graph = StateGraph(PipelineState)

    # Register all nodes
    for name in [
        "data_understanding", "problem_framing", "risk_check",
        "human_review", "preprocessing", "model_strategy",
        "training", "evaluation", "explainability", "report_generator", "halt"
    ]:
        graph.add_node(name, AgentFactory.create(name).run)

    # Edge wiring
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
        "retry": "model_strategy",          # graph-level retry; NOT try/except
        "explainability": "explainability",
        "human_review": "human_review",
        "halt": "halt",
    })
    graph.add_edge("explainability", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],  # pipeline pauses here; API resumes
    )


def route_after_risk(state: PipelineState) -> str:
    if state.halt:
        return "halt"
    if any(f.requires_human_approval for f in state.risk_flags):
        state.pending_approval_gates.extend(
            [f.code for f in state.risk_flags if f.requires_human_approval]
        )
        return "human_review"
    return "preprocessing"


def route_after_eval(state: PipelineState) -> str:
    if state.halt:
        return "halt"
    if any(f.requires_human_approval for f in state.risk_flags
           if f.code == "SUSPICIOUS_AUC"):
        return "human_review"
    if state.eval_metrics.get("pass_gate") is False:
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            return "retry"
        state.pending_approval_gates.append("max_retries_exhausted")
        return "human_review"
    return "explainability"
```

**Checkpoint persistence:** `PostgresSaver` saves full state after every node. If Celery worker crashes, the graph resumes from last checkpoint automatically on restart.

---

## 9. Database Schema & ORM

All tables already defined in `db/models/models.py`. The following additions are needed:

### New Table: SchemaVersionORM

```python
class SchemaVersionORM(Base):
    __tablename__ = "schema_versions"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    fingerprint: Mapped[str] = mapped_column(String)                 # SHA-256 of col:dtype pairs
    columns: Mapped[dict] = mapped_column(JSONB)
    dtypes: Mapped[dict] = mapped_column(JSONB)
    psi_scores: Mapped[dict] = mapped_column(JSONB, nullable=True)   # vs prior run
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
```

### Missing Repositories to Implement

| File | Purpose |
|---|---|
| `db/session.py` | `async_sessionmaker` factory |
| `db/repositories/session_repo.py` | `create_session`, `get_user_sessions` |
| `db/repositories/model_run_repo.py` | `create_run`, `mark_selected`, `get_by_session` |
| `db/repositories/risk_flag_repo.py` | `create_flags`, `resolve_flag` |

### Alembic Migrations

```bash
# Setup (run once)
cd backend
alembic init alembic
# Configure alembic.ini: sqlalchemy.url = postgresql+psycopg2://...

# Generate and apply migrations
alembic revision --autogenerate -m "initial_schema"
alembic upgrade head
```

**Index strategy:**
- All FK columns indexed (already in ORM models).
- JSONB columns use GIN indexes for `->` operator queries.
- `agent_decisions`: composite index on `(session_id, agent_name)`.
- `schema_versions`: composite index on `(session_id, recorded_at DESC)`.
- Paginate `agent_decisions` with cursor-based pagination (not offset).

---

## 10. API Endpoints

### FastAPI Router Structure

```
POST   /api/v1/sessions                        # Create session + upload CSV/Parquet
GET    /api/v1/sessions                        # List user's sessions (paginated)
GET    /api/v1/sessions/{session_id}           # Session detail + status

POST   /api/v1/pipeline/run/{session_id}       # Dispatch Celery task
GET    /api/v1/pipeline/status/{session_id}    # SSE stream: real-time stage progress

GET    /api/v1/approvals/pending/{session_id}  # Pending gates with full context
POST   /api/v1/approvals/submit/{session_id}   # Submit decision + reason → resume graph

GET    /api/v1/traces/{session_id}             # Full trace timeline (cursor-paginated)

GET    /api/v1/reports/{session_id}/json       # JSON report download
GET    /api/v1/reports/{session_id}/pdf        # PDF download
GET    /api/v1/reports/{session_id}/html       # HTML report

GET    /health                                 # [EXISTS] Health check
```

### Rate Limits

| Endpoint group | Limit |
|---|---|
| Session creation | 20/min |
| Pipeline trigger | 5/min |
| Approval submit | 20/min |
| Report download | 10/min (PDF), 30/min (JSON/HTML) |
| Read endpoints | 60/min |

### SSE Pipeline Status Stream

```python
@router.get("/pipeline/status/{session_id}")
async def pipeline_status(session_id: UUID, current_user=Depends(get_current_user)):
    async def event_generator():
        while True:
            state = await get_graph_checkpoint(session_id)
            yield {
                "event": "pipeline_update",
                "data": json.dumps({
                    "current_node": state.current_node,
                    "confidence_level": state.confidence_level,
                    "risk_flag_count": len(state.risk_flags),
                    "pending_approvals": state.pending_approval_gates,
                    "completed": state.report_path is not None or state.halt,
                })
            }
            if state.report_path or state.halt:
                break
            await asyncio.sleep(1)
    return EventSourceResponse(event_generator())
```

### Pydantic Schemas (to create in `schemas/`)

```python
# schemas/session.py
class SessionCreate(BaseModel):
    problem_statement: str = Field(..., min_length=10, max_length=2000)

class SessionOut(BaseModel):
    id: UUID
    user_id: str
    problem_statement: str
    created_at: datetime

# schemas/approval.py
class ApprovalSubmit(BaseModel):
    gate_name: str
    approved: bool
    reason: str = Field(..., min_length=10)   # mandatory; min 10 chars
```

---

## 11. Security Controls

| Threat | Control |
|---|---|
| SQL injection | SQLAlchemy ORM parameterized queries only. No raw SQL. |
| CSV formula injection | First-byte check for `=`, `+`, `-`, `@` before pandas read |
| Path traversal | Uploads stored in `/uploads/{session_id}/dataset{ext}` — user filename ignored |
| Cross-session access | `(session_id, user_id)` ownership check on every request handler |
| LLM prompt injection | All LLM prompts use structured outputs (Pydantic); no raw user text injected |
| Brute force | Rate limits on all write endpoints; 429 responses logged |
| Secrets | `.env` never committed; `Settings` reads from env only; production `SECRET_KEY` validated |
| File type abuse | Allowlist: `.csv`, `.parquet` only; MIME type verified; max size enforced |
| Approval bypass | `reason` field enforced non-empty in API schema AND agent validation |

---

## 12. Testing Strategy

### Unit Tests (per agent, no external services)

Each agent tested with:
- Mocked `PipelineState` and fixture DataFrames
- Assertions on written state fields
- Assertions that risk flags are emitted under correct conditions
- MLflow patched to no-op

```bash
pytest tests/unit/ -v --cov=app --cov-report=term-missing --cov-fail-under=80
```

**High-value unit test targets:**
- `PipelineState` invariants (all fields have correct defaults)
- `BaseAgent` logging and failure handling (exception re-raised)
- Every risk flag code emitted by `DataUnderstandingAgent` and `ProblemFramingAgent`
- `ModelRegistry` tier ordering and `min_samples` filtering
- `TrainingAgent` pass/fail gate logic (mocked cv scores)
- `EvaluationAgent` slice analysis flag emission
- `RiskFailureAgent` all 3 checkpoint scenarios
- Schema fingerprint utility with known col:dtype inputs
- PSI utility with known input distributions
- Approval gate rejection when `reason` is empty

### DataCorruptor Integration Matrix

**File to create:** `backend/app/ml/data_corruptor.py`

```python
class DataCorruptor:
    """Injects known data quality issues for demo reproducibility and regression testing."""

    def missing_column_names(self, df):    # → MISSING_COLUMN_NAMES flag
    def duplicate_rows(self, df, frac):    # → HIGH_DUPLICATE_RATIO flag
    def target_leakage_inject(self, df):   # → LEAKAGE_SUSPECTED flag
    def dtype_change(self, df, col):       # → SCHEMA_DTYPE_MISMATCH flag
    def null_burst(self, df, frac):        # Handled by preprocessing (no halt)
    def unseen_categories(self, df, col):  # Handled by OHE unknown bucket
    def extreme_outliers(self, df, col):   # → outlier_score logged in profile
    def changed_column_order(self, df):    # → fingerprint mismatch (schema drift)
```

| Method | Expected Flag | Expected Agent |
|---|---|---|
| `missing_column_names` | `MISSING_COLUMN_NAMES` | DataUnderstanding |
| `duplicate_rows(frac=0.25)` | `HIGH_DUPLICATE_RATIO` | DataUnderstanding |
| `target_leakage_inject` | `LEAKAGE_SUSPECTED` | ProblemFraming |
| `dtype_change` | `SCHEMA_DTYPE_MISMATCH` | RiskFailure (drift) |
| `null_burst(frac=0.4)` | handled (no halt) | Preprocessing |
| `unseen_categories` | handled (unknown bucket) | Preprocessing |
| `extreme_outliers` | outlier_score logged | DataUnderstanding |
| `changed_column_order` | fingerprint mismatch | RiskFailure (drift) |

### End-to-End Scenarios

```bash
pytest tests/e2e/ -v --timeout=180
```

**5 mandatory demo scenarios:**
1. **Clean classification** → full pipeline → JSON + HTML report generated, 0 risk flags.
2. **Messy dataset (40% null)** → KNNImputer selected → pipeline completes, no halt.
3. **Time-series tabular** → `TimeSeriesSplit` enforced → random split refused with flag.
4. **Schema-changed dataset** → PSI alert → `schema_drift_retrain` human gate triggered.
5. **Tiny dataset (n=50)** → `TINY_DATASET` → `confidence_level=unsafe` → `tiny_dataset_proceed` human gate.

### Integration Tests

```bash
pytest tests/integration/ -v --timeout=60
```

- Session creation and retrieval from PostgreSQL (test DB)
- `GenericRepository` CRUD operations
- LangGraph graph checkpoint save and resume
- Approval gate halt and resume (inject approval → graph continues)

---

## 13. Deployment Architecture

### Development (docker-compose.yml)

```yaml
services:
  api:
    build: ./backend
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    depends_on: [postgres, redis, mlflow]
    env_file: .env

  worker:
    build: ./backend
    command: celery -A app.services.pipeline_service worker -c 4 -l INFO
    depends_on: [postgres, redis]
    env_file: .env

  postgres:
    image: postgres:16-alpine
    volumes: [pg_data:/var/lib/postgresql/data]
    environment:
      POSTGRES_DB: ama2
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres

  redis:
    image: redis:7-alpine

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.11.0
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql://...
    ports: ["5000:5000"]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on: [api]
```

### CI/CD (GitHub Actions)

**`ci.yml`** — runs on every push and PR:
```yaml
jobs:
  lint:
    - ruff check .
    - mypy backend/app --ignore-missing-imports
  test:
    services: postgres, redis
    - alembic upgrade head
    - pytest tests/unit tests/integration --cov=app --cov-fail-under=80
  e2e:
    services: postgres, redis, mlflow
    - pytest tests/e2e --timeout=180
```

**`deploy.yml`** — runs on merge to `main`:
1. Build and push Docker images for API and worker.
2. Run `alembic upgrade head` migration.
3. Rolling restart of API and worker containers.

### Observability Stack

| Tool | Purpose |
|---|---|
| structlog | Structured JSON logs with `session_id` bound to every log line |
| Prometheus | `/metrics` endpoint: HTTP latency, Celery queue depth, error rate |
| Grafana | Dashboards for pipeline stage timing and failure rates |
| Sentry | Exception tracking with `session_id` + `agent_name` as tags |
| Flower | Celery task monitoring UI |
| MLflow UI | Experiment tracking, model comparison, run history |

### Scaling

- Celery worker replicas are stateless → scale horizontally.
- LangGraph checkpoints in PostgreSQL → worker crash recovery is automatic.
- Redis caches PSI/fingerprint results per `dataset_hash` (TTL: 1h).
- FAISS index for similar-experiment retrieval (embeddings of past `data_profile` dicts).

---

## 14. Phased Build Order

Each phase has a clear deliverable and a hard gate that must pass before the next phase begins.

### Phase 2 — Missing Agent Logic *(NEXT)*

**Goal:** All 10 agents implemented with real logic.

| Task | File | Gate |
|---|---|---|
| Implement `RiskFailureAgent` (3 checkpoints) | `agents/risk_failure.py` | All 3 checkpoint scenarios unit-tested |
| Implement `TrainingAgent` + `Trainer` | `agents/training.py`, `ml/trainer.py` | CV loop runs; MLflow run created per model |
| Implement `EvaluationAgent` + `Evaluator` | `agents/evaluation.py`, `ml/evaluator.py` | Pass/fail gate triggers retry correctly |
| Implement `ExplainabilityAgent` + `Explainer` | `agents/explainability.py`, `ml/explainer.py` | SHAP values computed; top-10 stored in state |
| Implement `ReportGeneratorAgent` | `agents/report_generator.py`, `services/report_service.py` | JSON report written to disk from trace |
| Implement `HumanReviewAgent` | `agents/human_review.py` | Agent validates approval fields and clears gates |
| Add dummy baselines to `ModelStrategyAgent` | `agents/model_strategy.py` | Dummies always present in candidates |
| Add train/test split to `PreprocessingAgent` | `agents/preprocessing.py` | `X_train/X_test/y_train/y_test` written to state |
| Implement `DataCorruptor` | `ml/data_corruptor.py` | All 8 methods cause expected flag emissions |

---

### Phase 3 — LangGraph Wiring & Graph Routing

**Goal:** Full pipeline runs end-to-end from dataset → report for a clean classification CSV.

| Task | File | Gate |
|---|---|---|
| Implement `build_graph()` with all 11 nodes | `core/orchestrator.py` | Graph compiles without error |
| Implement `route_after_risk()` | `core/orchestrator.py` | Routes to `human_review` when flag requires approval |
| Implement `route_after_eval()` | `core/orchestrator.py` | Retry increments and routes back to `model_strategy` |
| Implement `halt` terminal node | `core/orchestrator.py` | Pipeline halts cleanly with `halt_reason` logged |
| Wire PostgresSaver checkpoint | `core/orchestrator.py` | State recoverable after simulated worker crash |
| Implement `domain exceptions` | `core/exceptions.py` | All agents raise typed exceptions |

**E2E Smoke Test Gate:**
```python
# Demo scenario 1: Clean classification
state = PipelineState(session_id=uuid4(), user_id="test", dataset_path="tests/fixtures/iris.csv", problem_statement="Classify iris species")
result = graph.invoke(state)
assert result.report_path is not None
assert result.halt is False
assert len(result.trace_log) > 10
```

---

### Phase 4 — Persistence & Migrations

**Goal:** All pipeline decisions persisted to PostgreSQL; graph checkpointed.

| Task | File | Gate |
|---|---|---|
| Implement `db/session.py` async sessionmaker | `db/session.py` | Session connects to test DB |
| Implement per-entity repositories | `db/repositories/session_repo.py` etc. | CRUD tests pass |
| Add `SchemaVersionORM` model | `db/models/models.py` | Migration generated cleanly |
| Generate and apply Alembic migrations | `alembic/` | `alembic upgrade head` passes in CI |
| Write integration tests for repositories | `tests/integration/` | All CRUD operations verified |
| Implement `schema_fingerprint.py` | `utils/schema_fingerprint.py` | SHA-256 deterministic for same schema |
| Implement `psi.py` | `utils/psi.py` | PSI = 0 for identical distributions; > 0.2 for drifted |

---

### Phase 5 — FastAPI Layer & Celery

**Goal:** Working API that a frontend or `curl` can drive end-to-end.

| Task | File | Gate |
|---|---|---|
| Implement `dependencies.py` (DB DI, user DI) | `dependencies.py` | DB session injected in test client |
| Wire all routers in `main.py` | `main.py`, `api/v1/router.py` | All endpoints return 200/201/422 correctly |
| Implement `sessions.py` router | `api/v1/sessions.py` | POST with CSV upload → session_id returned |
| Implement `pipeline.py` router + SSE | `api/v1/pipeline.py` | `/run` dispatches Celery; `/status` streams events |
| Implement `approvals.py` router | `api/v1/approvals.py` | Submit approval → graph resumes |
| Implement `traces.py` router | `api/v1/traces.py` | Paginated trace returned; cursor works |
| Implement `reports.py` router | `api/v1/reports.py` | JSON/HTML/PDF download all work |
| Implement `pipeline_service.py` (Celery task) | `services/pipeline_service.py` | Task dispatched; worker picks it up |
| Implement `file_service.py` | `services/file_service.py` | Upload sanitized; CSV formula chars blocked |
| Write all Pydantic schemas | `schemas/` | Schemas validate correctly in tests |
| Add rate limiting (slowapi) | `main.py` | 429 returned when limit exceeded |

---

### Phase 6 — Utilities, Robustness & Security

**Goal:** Production-safe, observable, and auditable system.

| Task | File | Gate |
|---|---|---|
| Implement `retry.py` exponential backoff | `utils/retry.py` | Decorated function retries on transient error |
| Add `RANDOM_SEED = 42` assertion in all model builders | `ml/model_registry.py` | Unit test verifies seed present |
| Validate secrets on startup (production guard) | `config.py` | App refuses to start with weak `SECRET_KEY` in prod |
| Add upload file size guard (MAX_UPLOAD_SIZE_MB) | `services/file_service.py` | 413 returned when exceeded |
| Add CSV formula injection check | `services/file_service.py` | Files with `=SUM(...)` cells raise 422 |
| Implement `prometheus_client` `/metrics` endpoint | `main.py` | Metrics visible at `/metrics` |
| Configure Sentry SDK with `session_id` tag | `main.py` | Errors appear in Sentry with correct tag |
| Add `.env.example` | root | All required variables documented |

---

### Phase 7 — Unit Test Suite

**Goal:** 80%+ unit test coverage across all agents and ML modules.

| Test file | Coverage target |
|---|---|
| `tests/unit/agents/test_data_understanding.py` | All 6 risk flag codes |
| `tests/unit/agents/test_problem_framing.py` | All CV strategies + leakage detection |
| `tests/unit/agents/test_risk_failure.py` | All 3 checkpoint scenarios |
| `tests/unit/agents/test_preprocessing.py` | Train/test split; sklearn pipeline built |
| `tests/unit/agents/test_model_strategy.py` | Tier ordering; dummy baselines included |
| `tests/unit/agents/test_training.py` | CV runs; pass/fail gate; MLflow logged |
| `tests/unit/agents/test_evaluation.py` | Metrics computed; slice analysis flags |
| `tests/unit/agents/test_explainability.py` | TreeExplainer selected; top-10 stored |
| `tests/unit/agents/test_report_generator.py` | JSON report regeneratable from trace |
| `tests/unit/agents/test_human_review.py` | Empty reason rejected; gates cleared |
| `tests/unit/ml/test_data_corruptor.py` | All 8 methods tested individually |
| `tests/unit/utils/test_schema_fingerprint.py` | Deterministic; sensitive to dtype change |
| `tests/unit/utils/test_psi.py` | PSI = 0 for identical; > 0.2 for drifted |

---

### Phase 8 — E2E Tests & Docker

**Goal:** All 5 demo scenarios pass in a containerized environment.

| Task | Gate |
|---|---|
| Write 5 E2E test scenarios | All pass with `--timeout=180` |
| Create `backend/Dockerfile` | Image builds; app starts; `/health` returns 200 |
| Create `infra/docker-compose.yml` | `docker compose up` starts all 6 services |
| Create `infra/docker-compose.prod.yml` | Prod compose with nginx + replicas |
| Configure `nginx/nginx.conf` | Reverse proxy to API on port 80 |
| Write `ci.yml` GitHub Actions | lint + unit + integration pass on PR |
| Write `deploy.yml` GitHub Actions | Docker push + deploy on main merge |

---

### Phase 9 — Frontend (React Dashboard)

**Goal:** Visual interface that makes the system feel inspectable and trustworthy.

| Component | Purpose |
|---|---|
| `Dashboard.tsx` | Session list with status badges; "New Session" upload form |
| `SessionDetail.tsx` | Pipeline stage progress bar; live SSE updates |
| `TraceTimeline.tsx` | Chronological agent decision log with rationale |
| `RiskPanel.tsx` | Risk flags sorted by severity; resolution status |
| `ApprovalGate.tsx` | Context-rich approval form; reason field required |
| `ModelTable.tsx` | All candidates vs dummy baseline; metrics highlighted |
| `ShapChart.tsx` | Top-10 SHAP bar chart + 3 local waterfall explanations |
| `ReportViewer.tsx` | Embedded HTML report + PDF download button |

**Tech stack:** React + TypeScript + Zustand + Axios + Zod + Recharts.

---

## 15. Non-Negotiables at Launch

These must exist for the system to be credible. No exceptions.

- [ ] `sklearn Pipeline` wraps ALL preprocessing — no bare transforms outside the pipeline
- [ ] Dual dummy baselines always present in every training run
- [ ] Three-checkpoint `RiskFailureAgent` (pre-training, post-training, schema drift)
- [ ] Graph-level retry routing via LangGraph conditional edges — no try/except retry hacks
- [ ] LangGraph interrupt gate with mandatory non-empty `reason` field
- [ ] MLflow child run created per agent via `BaseAgent.run()` template method
- [ ] `RANDOM_SEED = 42` passed to every model constructor. Assert in unit tests.
- [ ] Every agent output is a typed Pydantic model or writes to a declared `PipelineState` field
- [ ] Slice analysis in `EvaluationAgent` — aggregate AUC alone is insufficient
- [ ] `DataCorruptor` module with all 8 methods — required for E2E scenario reproducibility
- [ ] JSON report regeneratable from `state.trace_log` alone — no re-running allowed
- [ ] `alembic upgrade head` runs cleanly in CI before every test run
- [ ] 80%+ unit test coverage enforced in CI (coverage gate fails build)
- [ ] `SECRET_KEY` validation fails startup in production with a weak value
- [ ] Every approval submission requires a `reason` field (API schema + agent validation)

---

> **Last Updated:** Phase 1 complete (scaffold). Phase 2 is the active sprint.
> **Target state:** All 9 phases complete = production-ready, fully testable, demo-able system.
