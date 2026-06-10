# ama2/backend/app/agents/base.py

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

try:
    import mlflow

    HAS_MLFLOW = True
except Exception:  # pragma: no cover - optional dependency fallback
    mlflow = None
    HAS_MLFLOW = False

from ..core.pipeline_state import PipelineState, TraceEntry
from ..utils.logging import bind_contextvars, get_logger


class _NoopMLflow:
    @staticmethod
    def start_run(*args, **kwargs):
        return nullcontext()

    @staticmethod
    def log_metric(*args, **kwargs):
        return None

    @staticmethod
    def set_tag(*args, **kwargs):
        return None

    @staticmethod
    def log_param(*args, **kwargs):
        return None

    @staticmethod
    def end_run(*args, **kwargs):
        return None


mlflow = mlflow if HAS_MLFLOW else _NoopMLflow()

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)

    @abstractmethod
    def _execute(self, state: PipelineState) -> PipelineState:
        """Core logic of the agent, implemented by subclasses."""
        ...

    def run(self, state: PipelineState) -> PipelineState:
        """
        Template method: wraps _execute with logging, MLflow child run, and error capture.
        """
        t0 = time.perf_counter()
        # Bind session ID for structural logging
        bind_contextvars(session_id=str(state.session_id))
        
        self.logger.info("agent_start", agent=self.name)
        
        # Start MLflow child run
        # Note: A parent run must be active for this nested call to work in a real setup.
        with mlflow.start_run(run_name=self.name, nested=True):
            try:
                state = self._execute(state)
                latency = time.perf_counter() - t0
                mlflow.log_metric("latency_s", latency)
                mlflow.set_tag("status", "success")
                self.logger.info("agent_complete", agent=self.name, latency_s=latency)
            except Exception as e:
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e)[:250])
                self.logger.exception("agent_failed", agent=self.name, error=str(e))
                raise
        return state

    def _log_decision(self, state: PipelineState, key: str, value: Any, rationale: str):
        """Standard method for logging agent decisions into the TraceLog and MLflow."""
        entry = TraceEntry(
            agent=self.name,
            decision_key=key,
            decision_value=value,
            rationale=rationale,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        state.trace_log.append(entry)
        
        # Log to MLflow for visibility
        mlflow.log_param(f"{self.name}_{key}", str(value)[:250])
        self.logger.info("agent_decision", agent=self.name, key=key, value=value, rationale=rationale)
