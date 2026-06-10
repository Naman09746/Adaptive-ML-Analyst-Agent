# ama2/backend/app/core/orchestrator.py
"""
LangGraph orchestrator for the AMA² pipeline.

Constructs a state machine with all agents as nodes, conditional routing,
and human approval gates (interrupts) at critical checkpoints.
"""

from __future__ import annotations

import pickle
from typing import Any
from uuid import UUID

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    StateGraph = None
    END = None
    MemorySaver = None
    HAS_LANGGRAPH = False

from .pipeline_state import PipelineState
from .agent_factory import AgentFactory
from ..utils.logging import get_logger

logger = get_logger("orchestrator")


def make_agent_node(agent_name: str) -> Any:
    """Creates a lazy-loaded wrapper function for an agent to act as a LangGraph node."""
    def node_fn(state: PipelineState) -> PipelineState:
        # Resolve canonical name and run the agent template method
        agent = AgentFactory.create(agent_name)
        return agent.run(state)
    node_fn.__name__ = agent_name
    return node_fn


def halt_node(state: PipelineState) -> PipelineState:
    """Terminal node representing an explicit pipeline halt."""
    logger.warning("pipeline_halted_execution", reason=state.halt_reason)
    state.halt = True
    return state


def route_after_risk(state: PipelineState) -> str:
    """Routes the graph after the risk check node based on safety flags."""
    if state.halt:
        return "halt"
        
    # Find any critical risk flags requiring manual intervention
    critical_gates = [f.code for f in state.risk_flags if f.requires_human_approval]
    
    if critical_gates:
        # Inject outstanding gates into the pending list
        for gate in critical_gates:
            if gate not in state.pending_approval_gates:
                state.pending_approval_gates.append(gate)
        return "human_review"
        
    return "preprocessing"


def route_after_eval(state: PipelineState) -> str:
    """Routes the graph after model evaluation to trigger retries, human review, or explanations."""
    if state.halt:
        return "halt"

    # 1. Suspicious ROC-AUC gate
    suspicious = [f.code for f in state.risk_flags if f.code == "SUSPICIOUS_AUC" and f.requires_human_approval]
    if suspicious:
        for gate in suspicious:
            if gate not in state.pending_approval_gates:
                state.pending_approval_gates.append(gate)
        return "human_review"

    # 2. Performance quality check
    if state.eval_metrics.get("pass_gate") is False:
        # Check if we can retry with a different model strategy tier
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            logger.info("initiating_pipeline_retry", attempt=state.retry_count, max=state.max_retries)
            return "retry"
            
        # Max retries hit, escalate to human gate
        if "max_retries_exhausted" not in state.pending_approval_gates:
            state.pending_approval_gates.append("max_retries_exhausted")
        return "human_review"

    return "explainability"


class PickleSerde:
    """A serializer that uses python's pickle module to handle arbitrary python objects."""
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return "pickle", pickle.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        return pickle.loads(data[1])


def build_graph(checkpointer: Any = None) -> Any:
    """
    Constructs, wires, and compiles the LangGraph StateGraph state machine.
    Enforces a strict interrupt checkpoint before reaching the human review node.
    
    Raises:
        ImportError: If langgraph is not installed
    """
    if not HAS_LANGGRAPH:
        raise ImportError(
            "langgraph is required to build the orchestration graph. "
            "Install with: pip install langgraph"
        )
    
    # Use in-memory checkpointer with pickle serialization as default fallback
    if checkpointer is None:
        checkpointer = MemorySaver(serde=PickleSerde())

    graph = StateGraph(PipelineState)

    # 1. Register all nodes
    graph.add_node("data_understanding", make_agent_node("data_understanding"))
    graph.add_node("problem_framing", make_agent_node("problem_framing"))
    graph.add_node("risk_check", make_agent_node("risk_check"))
    graph.add_node("human_review", make_agent_node("human_review"))
    graph.add_node("preprocessing", make_agent_node("preprocessing"))
    graph.add_node("model_strategy", make_agent_node("model_strategy"))
    graph.add_node("training", make_agent_node("training"))
    graph.add_node("evaluation", make_agent_node("evaluation"))
    graph.add_node("explainability", make_agent_node("explainability"))
    graph.add_node("report_generator", make_agent_node("report_generator"))
    graph.add_node("halt", halt_node)

    # 2. Configure edges and flow paths
    graph.set_entry_point("data_understanding")
    graph.add_edge("data_understanding", "problem_framing")
    graph.add_edge("problem_framing", "risk_check")

    # Conditional routing after the initial risk checks
    graph.add_conditional_edges(
        "risk_check",
        route_after_risk,
        {
            "human_review": "human_review",
            "preprocessing": "preprocessing",
            "halt": "halt"
        }
    )

    # Transition from human review to preprocessing
    graph.add_edge("human_review", "preprocessing")
    graph.add_edge("preprocessing", "model_strategy")
    graph.add_edge("model_strategy", "training")
    graph.add_edge("training", "evaluation")

    # Conditional routing after evaluation: supports model retries
    graph.add_conditional_edges(
        "evaluation",
        route_after_eval,
        {
            "retry": "model_strategy",
            "explainability": "explainability",
            "human_review": "human_review",
            "halt": "halt"
        }
    )

    # Finalize pipeline
    graph.add_edge("explainability", "report_generator")
    graph.add_edge("report_generator", END)
    graph.add_edge("halt", END)

    # Compile the graph with checkpointer and human gate interrupt before human review
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )
