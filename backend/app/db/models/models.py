# ama2/backend/app/db/models/models.py
from datetime import datetime, timezone
from uuid import UUID, uuid4
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, JSON, Boolean, Text
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from ..base import Base

class SessionORM(Base):
    __tablename__ = "sessions"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String, index=True)
    dataset_path: Mapped[str] = mapped_column(String)
    problem_statement: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class AgentDecisionORM(Base):
    __tablename__ = "agent_decisions"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    agent_name: Mapped[str] = mapped_column(String, index=True)
    decision_key: Mapped[str] = mapped_column(String)
    decision_value: Mapped[dict] = mapped_column(JSONB)
    rationale: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ModelRunORM(Base):
    __tablename__ = "model_runs"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    mlflow_run_id: Mapped[str] = mapped_column(String, unique=True)
    model_type: Mapped[str] = mapped_column(String)
    hyperparameters: Mapped[dict] = mapped_column(JSONB)
    cv_scores: Mapped[dict] = mapped_column(JSONB)
    eval_metrics: Mapped[dict] = mapped_column(JSONB)
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False)

class RiskFlagORM(Base):
    __tablename__ = "risk_flags"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    level: Mapped[str] = mapped_column(String) # critical | warning | info
    code: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    feature: Mapped[str] = mapped_column(String, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution: Mapped[str] = mapped_column(Text, nullable=True)

class HumanApprovalORM(Base):
    __tablename__ = "human_approvals"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    gate_name: Mapped[str] = mapped_column(String)
    approved: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str] = mapped_column(Text)
    approved_by: Mapped[str] = mapped_column(String)
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
