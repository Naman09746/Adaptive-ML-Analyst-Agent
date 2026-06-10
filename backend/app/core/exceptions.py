# ama2/backend/app/core/exceptions.py

class PipelineError(Exception):
    """Base class for all pipeline-related exceptions."""
    pass

class PipelineHaltError(PipelineError):
    """Raised when a critical error or risk flag halts pipeline execution."""
    pass

class AgentContractViolationError(PipelineError):
    """Raised when an agent violates write ownership or state invariants."""
    pass

class ApprovalRequiredError(PipelineError):
    """Raised when a risk flag requires human approval and pauses execution."""
    pass

class SchemaFingerprintMismatchError(PipelineError):
    """Raised when the schema fingerprint of a new run does not match the baseline."""
    pass
