# ama2/backend/app/core/agent_factory.py

from typing import ClassVar, Type
from ..agents.base import BaseAgent
from ..agents.data_understanding import DataUnderstandingAgent
from ..agents.problem_framing import ProblemFramingAgent
from ..agents.risk_failure import RiskFailureAgent
from ..agents.preprocessing import PreprocessingAgent
from ..agents.model_strategy import ModelStrategyAgent
from ..agents.training import TrainingAgent
from ..agents.evaluation import EvaluationAgent
from ..agents.explainability import ExplainabilityAgent
from ..agents.report_generator import ReportGeneratorAgent
from ..agents.human_review import HumanReviewAgent

class AgentFactory:
    _agents: ClassVar[dict[str, Type[BaseAgent]]] = {
        "data_understanding": DataUnderstandingAgent,
        "problem_framing": ProblemFramingAgent,
        "risk_check": RiskFailureAgent, # Note: plan uses risk_failure and risk_check interchangeably
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
