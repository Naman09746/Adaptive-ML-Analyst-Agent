# ama2/backend/app/agents/human_review.py

from __future__ import annotations

from ..core.pipeline_state import PipelineState
from .base import BaseAgent


class HumanReviewAgent(BaseAgent):
    """
    Acts as a validation gate after the LangGraph execution is resumed from interrupt.
    Ensures that human review decisions contain non-empty justifications before clearing gates.
    """

    def __init__(self):
        super().__init__(name="human_review")

    def _execute(self, state: PipelineState) -> PipelineState:
        """
        Validates approval entries present in state.human_approvals against
        the checklist in state.pending_approval_gates.
        """
        self.logger.info("resuming_from_interrupt_validating_approvals", pending=state.pending_approval_gates)

        # We copy to avoid modifying list during iteration
        for gate in list(state.pending_approval_gates):
            approval = state.human_approvals.get(gate)
            
            if approval is None:
                raise ValueError(f"Gate '{gate}' requires human approval but no decision was submitted.")

            # Validate that a reason was supplied
            reason = approval.get("reason", "").strip()
            if not reason or len(reason) < 10:
                raise ValueError(
                    f"Approval for gate '{gate}' must include a justification of at least 10 characters."
                )

            # Check if user rejected the gate
            if not approval.get("approved", False):
                self.logger.warning("gate_rejected_by_human", gate=gate, reason=reason)
                state.halt = True
                state.halt_reason = f"Gate '{gate}' was rejected by human operator: {reason}"
                break

            # Clear the gate since it is approved
            state.pending_approval_gates.remove(gate)

        self._log_decision(
            state,
            "human_review_complete",
            {
                "approved_gates": list(state.human_approvals.keys()),
                "halted": state.halt,
                "halt_reason": state.halt_reason
            },
            "Human approvals validated successfully. Resume pipeline."
        )

        return state
