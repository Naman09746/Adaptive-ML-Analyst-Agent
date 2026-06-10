# ama2/backend/tests/test_smoke.py

from uuid import uuid4
import pytest
from pathlib import Path

from backend.app.core.pipeline_state import PipelineState
from backend.app.core.orchestrator import build_graph

def test_clean_classification_pipeline():
    # 1. Locate dataset
    iris_path = Path(__file__).parent / "fixtures" / "iris.csv"
    assert iris_path.exists(), "Iris fixture CSV must exist before running smoke test"

    # 2. Build pipeline state
    state = PipelineState(
        session_id=uuid4(),
        user_id="test_user",
        dataset_path=str(iris_path),
        problem_statement="Classify iris species based on measurements"
    )

    # 3. Build and run LangGraph workflow
    graph = build_graph()
    
    # Configure thread context required by LangGraph checkpointer
    config = {"configurable": {"thread_id": str(state.session_id)}}
    
    # Execute graph: it should run data_understanding, problem_framing, and risk_check,
    # then pause because the risk_check flags a target leakage risk (petal_width).
    result = graph.invoke(state, config=config)

    # 4. Confirm the graph is paused at 'human_review'
    state_snapshot = graph.get_state(config)
    assert "human_review" in state_snapshot.next, "Graph should be paused before human_review"
    assert "LEAKAGE_SUSPECTED" in state_snapshot.values["pending_approval_gates"], "Gate list should contain LEAKAGE_SUSPECTED"

    # 5. Simulate FastAPI endpoint submitting approval to resume
    graph.update_state(
        config,
        {
            "human_approvals": {
                "LEAKAGE_SUSPECTED": {"approved": True, "reason": "Approved dropping the leakage column to proceed safely."}
            }
        },
        as_node="human_review"
    )

    # 6. Resume execution by invoking graph with None
    final_result = graph.invoke(None, config=config)

    # 7. Verify final deliverables
    assert final_result["report_path"] is not None, "HTML report should be generated and recorded"
    assert final_result["halt"] is False, "Pipeline should complete successfully without halting"
    assert len(final_result["trace_log"]) >= 10, "Trace log should contain entries from all nodes"
    assert final_result["best_model"] is not None, "Best model pipeline should be fitted and saved"
    assert final_result["best_model_name"] is not None, "Best model name should be saved"
    
    # Safe checks on output files
    report_html_path = Path(final_result["report_path"])
    assert report_html_path.exists(), f"Report file not found on disk: {report_html_path}"
    
    report_json_path = report_html_path.parent / "report.json"
    assert report_json_path.exists(), "JSON report should be generated on disk"
    
    # Clean up generated report files to keep the directory clean
    try:
        report_html_path.unlink()
        report_json_path.unlink()
        report_html_path.parent.rmdir()
    except Exception:
        pass
