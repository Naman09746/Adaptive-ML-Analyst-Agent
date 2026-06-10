# ama2/backend/app/agents/report_generator.py

from __future__ import annotations

from ..core.constants import CONFIDENCE_SAFE, CONFIDENCE_UNCERTAIN, CONFIDENCE_UNSAFE
from ..core.pipeline_state import PipelineState
from .base import BaseAgent
from ..services.report_service import ReportService


class ReportGeneratorAgent(BaseAgent):
    """
    Finalizes execution by assessing cumulative risk flags to set the system confidence level,
    and calls ReportService to export JSON/HTML/PDF artifacts.
    """

    def __init__(self):
        super().__init__(name="report_generator")

    def _execute(self, state: PipelineState) -> PipelineState:
        self.logger.info("generating_pipeline_reports")

        # 1. Determine the final system confidence level based on risk flag severity
        criticals = [f for f in state.risk_flags if f.level == "critical"]
        warnings = [f for f in state.risk_flags if f.level == "warning"]

        if criticals:
            state.confidence_level = CONFIDENCE_UNSAFE
        elif warnings:
            state.confidence_level = CONFIDENCE_UNCERTAIN
        else:
            state.confidence_level = CONFIDENCE_SAFE

        self.logger.info(
            "determined_confidence_level",
            confidence=state.confidence_level,
            critical_count=len(criticals),
            warning_count=len(warnings)
        )

        # 2. Invoke the Report Service
        # We store reports in 'backend/static/reports' or local 'reports' based on config
        service = ReportService(base_reports_dir="reports")
        report_html_path = service.write_reports(state)

        # 3. Store final report path in PipelineState
        state.report_path = report_html_path

        self._log_decision(
            state,
            "report_generation_complete",
            {
                "report_path": state.report_path,
                "confidence_level": state.confidence_level,
                "total_risk_flags": len(state.risk_flags)
            },
            f"HTML/JSON executive reports successfully written to {state.report_path}."
        )

        return state
