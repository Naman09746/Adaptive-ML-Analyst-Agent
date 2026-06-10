# ama2/backend/app/core/agent_factory.py

from importlib import import_module
from typing import ClassVar, Type

from ..agents.base import BaseAgent

class AgentFactory:
    _agents: ClassVar[dict[str, tuple[str, str]]] = {
        "data_understanding": ("backend.app.agents.data_understanding", "DataUnderstandingAgent"),
        "problem_framing": ("backend.app.agents.problem_framing", "ProblemFramingAgent"),
        "risk_failure": ("backend.app.agents.risk_failure", "RiskFailureAgent"),
        "preprocessing": ("backend.app.agents.preprocessing", "PreprocessingAgent"),
        "model_strategy": ("backend.app.agents.model_strategy", "ModelStrategyAgent"),
        "training": ("backend.app.agents.training", "TrainingAgent"),
        "evaluation": ("backend.app.agents.evaluation", "EvaluationAgent"),
        "explainability": ("backend.app.agents.explainability", "ExplainabilityAgent"),
        "report_generator": ("backend.app.agents.report_generator", "ReportGeneratorAgent"),
        "human_review": ("backend.app.agents.human_review", "HumanReviewAgent"),
    }

    _aliases: ClassVar[dict[str, str]] = {
        "risk_check": "risk_failure",
    }

    @classmethod
    def create(cls, name: str) -> BaseAgent:
        canonical_name = cls._aliases.get(name, name)

        if canonical_name not in cls._agents:
            raise ValueError(f"Unknown agent: {name}")

        module_path, class_name = cls._agents[canonical_name]

        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:
            raise NotImplementedError(
                f"Agent '{canonical_name}' is registered but not implemented yet."
            ) from exc

        agent_cls = getattr(module, class_name, None)
        if agent_cls is None:
            raise NotImplementedError(
                f"Agent '{canonical_name}' is missing the expected class '{class_name}'."
            )

        return agent_cls()

    @classmethod
    def available_agents(cls) -> list[str]:
        available: list[str] = []
        for name, (module_path, class_name) in cls._agents.items():
            try:
                module = import_module(module_path)
                if getattr(module, class_name, None) is not None:
                    available.append(name)
            except ModuleNotFoundError:
                continue
        return sorted(available)
