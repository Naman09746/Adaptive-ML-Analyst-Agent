# ama2/backend/app/agents/data_understanding.py

from __future__ import annotations

from ..core.constants import CONFIDENCE_UNCERTAIN
from ..core.pipeline_state import PipelineState, RiskFlag
from ..ml.data_inspection import build_risk_signals, load_dataset, profile_dataframe
from .base import BaseAgent


class DataUnderstandingAgent(BaseAgent):
    def __init__(self):
        super().__init__("data_understanding")

    def _execute(self, state: PipelineState) -> PipelineState:
        """Load and profile the dataset, then record structural risks."""
        self.logger.info("profiling_dataset", path=state.dataset_path)

        if state.df is None:
            state.df = load_dataset(state.dataset_path)

        profile = profile_dataframe(state.df)
        state.data_profile = profile
        state.confidence_level = state.confidence_level or CONFIDENCE_UNCERTAIN

        for signal in build_risk_signals(state.df, state.target_column):
            state.risk_flags.append(
                RiskFlag(
                    level=signal["level"],
                    code=signal["code"],
                    feature=signal.get("feature"),
                    description=signal["reason"],
                    recommended_action="Review dataset quality before advancing.",
                    requires_human_approval=signal["level"] == "critical",
                )
            )

        self._log_decision(
            state,
            "profile_summary",
            {
                "rows": profile["shape"]["rows"],
                "columns": profile["shape"]["columns"],
                "duplicate_ratio": profile["duplicate_ratio"],
            },
            "Captured a deterministic dataset profile for downstream agents.",
        )

        self._log_decision(state, "status", "complete", "Successfully profiled dataset.")
        return state
