# ama2/backend/app/agents/problem_framing.py

from __future__ import annotations

from ..core.constants import (
    CONFIDENCE_SAFE,
    CONFIDENCE_UNCERTAIN,
    CONFIDENCE_UNSAFE,
    LEAKAGE_SUSPECTED,
    PROBLEM_CLASSIFICATION,
    PROBLEM_REGRESSION,
    PROBLEM_TIMESERIES,
)
from ..core.pipeline_state import PipelineState, RiskFlag
from ..ml.data_inspection import (
    detect_cv_strategy,
    detect_group_column,
    detect_leakage_suspects,
    detect_timeseries_column,
    infer_problem_type,
    infer_target_column,
)
from .base import BaseAgent


class ProblemFramingAgent(BaseAgent):
    def __init__(self):
        super().__init__("problem_framing")

    def _execute(self, state: PipelineState) -> PipelineState:
        if state.df is None:
            raise ValueError("Problem framing requires a loaded dataframe")

        target_column = state.target_column or infer_target_column(state.df, state.problem_statement)
        problem_type = infer_problem_type(state.df, target_column, state.problem_statement)
        group_column = detect_group_column(state.df)
        timeseries_column = detect_timeseries_column(state.df)
        cv_strategy = detect_cv_strategy(problem_type, group_column, timeseries_column)
        leakage_suspects = detect_leakage_suspects(state.df, target_column)

        state.target_column = target_column
        state.problem_type = problem_type
        state.group_column = group_column
        state.cv_strategy = cv_strategy
        state.leakage_suspects = leakage_suspects

        if leakage_suspects:
            state.confidence_level = CONFIDENCE_UNSAFE
            state.risk_flags.append(
                RiskFlag(
                    level="critical",
                    code=LEAKAGE_SUSPECTED,
                    feature=leakage_suspects[0],
                    description="Potential leakage features were detected during problem framing.",
                    recommended_action="Remove or validate suspicious columns before training.",
                    requires_human_approval=True,
                )
            )
        elif state.confidence_level is None:
            if problem_type in {PROBLEM_CLASSIFICATION, PROBLEM_REGRESSION, PROBLEM_TIMESERIES}:
                state.confidence_level = CONFIDENCE_SAFE
            else:
                state.confidence_level = CONFIDENCE_UNCERTAIN

        self._log_decision(
            state,
            "framing_summary",
            {
                "target_column": target_column,
                "problem_type": problem_type,
                "group_column": group_column,
                "timeseries_column": timeseries_column,
                "cv_strategy": cv_strategy,
                "leakage_suspects": leakage_suspects,
            },
            "Derived target, task type, CV strategy, and leakage signals from the dataset.",
        )

        self._log_decision(state, "status", "complete", "Successfully framed the problem.")
        return state
