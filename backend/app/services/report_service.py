# ama2/backend/app/services/report_service.py

from __future__ import annotations

import os
import json
import base64
import io
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    sns = None
    HAS_MATPLOTLIB = False

try:
    from weasyprint import HTML as WPHTML
    HAS_WEASYPRINT = True
except ImportError:
    WPHTML = None
    HAS_WEASYPRINT = False

from jinja2 import Template
from ..utils.logging import get_logger

logger = get_logger("report_service")


class ReportService:
    """
    Handles file-system paths, generates structured JSON summaries,
    renders rich HTML with base64 embedded charts, and outputs PDF copies.
    """

    def __init__(self, base_reports_dir: str = "reports"):
        self.base_dir = Path(base_reports_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(self, state: Any) -> Dict[str, Any]:
        """Assembles a machine-readable JSON structure containing all pipeline details."""
        # Convert RiskFlags to serializable dicts
        serialized_flags = [
            {
                "level": f.level,
                "code": f.code,
                "feature": f.feature,
                "description": f.description,
                "recommended_action": f.recommended_action,
                "requires_human_approval": f.requires_human_approval
            } for f in state.risk_flags
        ]

        # Convert TraceEntries to serializable dicts
        serialized_trace = [
            {
                "agent": t.agent,
                "decision_key": t.decision_key,
                "decision_value": t.decision_value,
                "rationale": t.rationale,
                "timestamp": t.timestamp
            } for t in state.trace_log
        ]

        report_data = {
            "session_id": str(state.session_id),
            "user_id": state.user_id,
            "dataset_path": state.dataset_path,
            "problem_statement": state.problem_statement,
            "problem_type": state.problem_type,
            "target_column": state.target_column,
            "cv_strategy": state.cv_strategy,
            "group_column": state.group_column,
            "confidence_level": state.confidence_level,
            "best_model_name": state.best_model_name,
            "data_profile": state.data_profile,
            "preprocessing_plan": state.preprocessing_plan,
            "eval_metrics": {k: v for k, v in state.eval_metrics.items() if k != "all_candidates"},
            "candidate_history": state.eval_metrics.get("all_candidates", []),
            "shap_values": state.shap_values,
            "business_narrative": state.business_narrative,
            "risk_flags": serialized_flags,
            "trace_log": serialized_trace,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Remove non-serializable objects (like dataframes) from inner structures if any
        if "shape" in report_data.get("data_profile", {}):
            # Safe copy
            pass
            
        return report_data

    def build_charts(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Generates matplotlib charts as base64-encoded strings for inline HTML embedding."""
        charts = {}
        if not HAS_MATPLOTLIB or plt is None:
            logger.warning("matplotlib_not_available_skipping_visual_charts")
            return charts

        try:
            # 1. SHAP global feature importance chart
            shap_data = report_data.get("shap_values", {})
            global_imp = shap_data.get("global_importance", []) if isinstance(shap_data, dict) else []
            
            if global_imp:
                top_n = global_imp[:10]
                features = [item["feature"] for item in top_n]
                importances = [item["importance"] for item in top_n]

                plt.figure(figsize=(7, 4))
                # Accent color styling
                sns.barplot(x=importances, y=features, palette="viridis")
                plt.title("SHAP Global Feature Importance (Top 10)")
                plt.xlabel("Mean |SHAP Value|")
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150)
                plt.close()
                buf.seek(0)
                charts["shap_importance"] = base64.b64encode(buf.read()).decode("utf-8")

            # 2. Confusion Matrix Heatmap (for Classification)
            metrics = report_data.get("eval_metrics", {})
            conf_matrix = metrics.get("confusion_matrix")
            if conf_matrix:
                plt.figure(figsize=(5, 4))
                sns.heatmap(np.array(conf_matrix), annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.title("Holdout Confusion Matrix")
                plt.ylabel("Actual Label")
                plt.xlabel("Predicted Label")
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150)
                plt.close()
                buf.seek(0)
                charts["confusion_matrix"] = base64.b64encode(buf.read()).decode("utf-8")

        except Exception as e:
            logger.exception("failed_to_build_visual_charts", error=str(e))
            
        return charts

    def render_html_report(self, report_data: Dict[str, Any], charts: Dict[str, str]) -> str:
        """Renders the HTML report template using Jinja2 with modern Glassmorphism aesthetics."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AMA² Executive Model Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-purple: #8b5cf6;
            --accent-indigo: #6366f1;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 1000px;
        }

        .header-card {
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple));
            border-radius: 24px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
        }

        .header-card h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.8rem;
            margin: 0 0 10px 0;
            font-weight: 800;
        }

        .header-meta {
            display: flex;
            gap: 20px;
            font-size: 0.95rem;
            opacity: 0.9;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(16px);
        }

        h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.6rem;
            color: var(--text-primary);
            margin-top: 0;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
            display: flex;
            align-items: center;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .metric-tile {
            background: rgba(15, 23, 42, 0.4);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent-purple);
        }

        .label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        th {
            text-align: left;
            color: var(--text-secondary);
            font-weight: 600;
            padding: 10px;
            border-bottom: 2px solid var(--border-color);
        }

        td {
            padding: 12px 10px;
            border-bottom: 1px solid var(--border-color);
        }

        .flag-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .flag-critical { background: rgba(239, 68, 68, 0.2); color: var(--danger-color); }
        .flag-warning { background: rgba(245, 158, 11, 0.2); color: var(--warning-color); }
        .flag-info { background: rgba(99, 102, 241, 0.2); color: var(--accent-indigo); }

        .business-narrative {
            font-size: 1.1rem;
            line-height: 1.6;
            color: #e2e8f0;
            background: rgba(99, 102, 241, 0.1);
            border-left: 4px solid var(--accent-indigo);
            padding: 20px;
            border-radius: 0 12px 12px 0;
        }

        .chart-box {
            text-align: center;
            padding: 15px 0;
        }

        .chart-img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }

        .deployment-badge {
            display: inline-block;
            padding: 12px 24px;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 700;
            text-align: center;
        }

        .deploy-recommended { background: rgba(16, 185, 129, 0.2); color: var(--success-color); }
        .deploy-conditional { background: rgba(245, 158, 11, 0.2); color: var(--warning-color); }
        .deploy-not-recommended { background: rgba(239, 68, 68, 0.2); color: var(--danger-color); }
    </style>
</head>
<body>
    <div class="container">
        
        <!-- Header -->
        <div class="header-card">
            <h1>AMA² Model Report</h1>
            <div class="header-meta">
                <div><strong>Session:</strong> {{ session_id }}</div>
                <div><strong>Target:</strong> {{ target_column }}</div>
                <div><strong>Problem Type:</strong> {{ problem_type }}</div>
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="card">
            <h2>Executive Summary</h2>
            <div class="grid-2">
                <div class="metric-tile">
                    <div class="label">Best Model</div>
                    <div class="metric-value">{{ best_model_name }}</div>
                </div>
                <div class="metric-tile">
                    <div class="label">Confidence Score</div>
                    <div class="metric-value" style="color: {% if confidence_level == 'safe' %}var(--success-color){% elif confidence_level == 'uncertain' %}var(--warning-color){% else %}var(--danger-color){% endif %}">
                        {{ confidence_level | upper }}
                    </div>
                </div>
            </div>
            
            <h3 style="margin-top: 30px;">Deployment Decision</h3>
            {% if confidence_level == 'safe' %}
                <div class="deployment-badge deploy-recommended">RECOMMENDED FOR PRODUCTION</div>
            {% elif confidence_level == 'uncertain' %}
                <div class="deployment-badge deploy-conditional">CONDITIONAL APPROVAL REQUIRED</div>
            {% else %}
                <div class="deployment-badge deploy-not-recommended">NOT RECOMMENDED / BLOCKED</div>
            {% endif %}
        </div>

        <!-- Business Narrative -->
        <div class="card">
            <h2>Business Interpretation</h2>
            <div class="business-narrative">
                {{ business_narrative }}
            </div>
        </div>

        <!-- Performance Dashboard -->
        <div class="card">
            <h2>Model Performance Dashboard</h2>
            <div class="grid-2">
                <div class="metric-tile">
                    <div class="label">Cross-Validation Mean Score</div>
                    <div class="metric-value">{{ "%.4f"|format(eval_metrics.cv_mean) if eval_metrics.cv_mean is not none else "N/A" }}</div>
                </div>
                <div class="metric-tile">
                    {% if problem_type == 'classification' %}
                        <div class="label">Holdout ROC-AUC</div>
                        <div class="metric-value">{{ "%.4f"|format(eval_metrics.roc_auc) if eval_metrics.roc_auc is not none else "N/A" }}</div>
                    {% else %}
                        <div class="label">Holdout R² Score</div>
                        <div class="metric-value">{{ "%.4f"|format(eval_metrics.r2) if eval_metrics.r2 is not none else "N/A" }}</div>
                    {% endif %}
                </div>
            </div>

            {% if charts.confusion_matrix %}
                <div class="chart-box" style="margin-top: 30px;">
                    <img class="chart-img" src="data:image/png;base64,{{ charts.confusion_matrix }}" alt="Confusion Matrix">
                </div>
            {% endif %}
        </div>

        <!-- Explainability -->
        {% if charts.shap_importance %}
            <div class="card">
                <h2>Explainability & Feature Attributions</h2>
                <div class="chart-box">
                    <img class="chart-img" src="data:image/png;base64,{{ charts.shap_importance }}" alt="SHAP Feature Importance">
                </div>
            </div>
        {% endif %}

        <!-- Preprocessing -->
        <div class="card">
            <h2>Preprocessing Pipeline Details</h2>
            <table>
                <thead>
                    <tr>
                        <th>Numeric Features</th>
                        <th>Categorical Features</th>
                        <th>Remainder Strategy</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{{ preprocessing_plan.numeric_columns | length }} Columns</td>
                        <td>{{ preprocessing_plan.categorical_columns | length }} Columns</td>
                        <td>{{ preprocessing_plan.remainder | upper }}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Risk Flags -->
        <div class="card">
            <h2>Risk Audit Log</h2>
            {% if risk_flags %}
                <table>
                    <thead>
                        <tr>
                            <th>Severity</th>
                            <th>Code</th>
                            <th>Target Column/Feature</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for flag in risk_flags %}
                            <tr>
                                <td>
                                    <span class="flag-pill flag-{{ flag.level }}">
                                        {{ flag.level | upper }}
                                    </span>
                                </td>
                                <td><strong>{{ flag.code }}</strong></td>
                                <td>{{ flag.feature or 'Global' }}</td>
                                <td>{{ flag.description }}</td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% else %}
                <div style="color: var(--success-color); font-weight: 600;">✓ No critical or warning risks detected in this pipeline execution.</div>
            {% endif %}
        </div>

    </div>
</body>
</html>
"""
        template = Template(template_str)
        return template.render(
            session_id=report_data["session_id"],
            target_column=report_data["target_column"],
            problem_type=report_data["problem_type"],
            best_model_name=report_data["best_model_name"],
            confidence_level=report_data["confidence_level"],
            business_narrative=report_data["business_narrative"],
            eval_metrics=report_data["eval_metrics"],
            preprocessing_plan=report_data["preprocessing_plan"],
            risk_flags=report_data["risk_flags"],
            charts=charts
        )

    def write_reports(self, state: Any) -> str:
        """
        Main entrypoint: builds JSON, HTML, and PDF reports.
        Returns the absolute filepath to the HTML report.
        """
        session_id = str(state.session_id)
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate and save JSON
        report_data = self.generate_json_report(state)
        json_path = session_dir / "report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info("json_report_saved", path=str(json_path))

        # 2. Build charts & render HTML
        charts = self.build_charts(report_data)
        html_content = self.render_html_report(report_data, charts)
        html_path = session_dir / "report.html"
        with open(html_path, "w") as f:
            f.write(html_content)
        logger.info("html_report_saved", path=str(html_path))

        # 3. Export PDF via WeasyPrint if available
        pdf_path = session_dir / "report.pdf"
        if HAS_WEASYPRINT and WPHTML is not None:
            try:
                WPHTML(string=html_content).write_pdf(target=pdf_path)
                logger.info("pdf_report_saved", path=str(pdf_path))
            except Exception as e:
                logger.exception("weasyprint_pdf_rendering_failed", error=str(e))
        else:
            logger.info("weasyprint_not_available_pdf_skipped")

        return str(html_path)

    def regenerate_from_json(self, json_file_path: str) -> str:
        """Regenerates the HTML/PDF reports using only a previously saved JSON report file."""
        with open(json_file_path, "r") as f:
            report_data = json.load(f)
            
        charts = self.build_charts(report_data)
        html_content = self.render_html_report(report_data, charts)
        
        # Save to the same directory as the JSON file
        json_path = Path(json_file_path)
        session_dir = json_path.parent
        
        html_path = session_dir / "report.html"
        with open(html_path, "w") as f:
            f.write(html_content)
            
        pdf_path = session_dir / "report.pdf"
        if HAS_WEASYPRINT and WPHTML is not None:
            try:
                WPHTML(string=html_content).write_pdf(target=pdf_path)
            except Exception as e:
                logger.warning(f"Could not regenerate PDF: {e}")
                
        return str(html_path)
