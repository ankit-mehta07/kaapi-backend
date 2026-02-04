from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel

from app.core.util import now

if TYPE_CHECKING:
    from .organization import Organization
    from .project import Project


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class JobType(str, Enum):
    RESPONSE = "RESPONSE"
    LLM_API = "LLM_API"


class Job(SQLModel, table=True):
    """Database model for tracking async jobs."""

    __tablename__ = "job"

    id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"comment": "Unique identifier for the job"},
    )
    task_id: str | None = Field(
        nullable=True,
        description="Celery task ID returned when job is queued.",
        sa_column_kwargs={"comment": "Celery task ID returned when job is queued"},
    )
    trace_id: str | None = Field(
        default=None,
        description="Tracing ID for correlating logs and traces.",
        sa_column_kwargs={"comment": "Tracing ID for correlating logs and traces"},
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if the job fails.",
        sa_column_kwargs={"comment": "Error details if the job fails"},
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Current state of the job.",
        sa_column_kwargs={
            "comment": "Current state of the job (PENDING, PROCESSING, SUCCESS, FAILED)"
        },
    )
    job_type: JobType = Field(
        description="Type of job being executed (e.g., response, ingestion).",
        sa_column_kwargs={
            "comment": "Type of job being executed (e.g., RESPONSE, LLM_API)"
        },
    )

    # Foreign keys
    organization_id: int = Field(
        foreign_key="organization.id",
        nullable=False,
        ondelete="CASCADE",
        index=True,
        sa_column_kwargs={"comment": "Reference to the organization"},
    )
    project_id: int = Field(
        foreign_key="project.id",
        nullable=False,
        ondelete="CASCADE",
        index=True,
        sa_column_kwargs={"comment": "Reference to the project"},
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=now,
        sa_column_kwargs={"comment": "Timestamp when the job was created"},
    )
    updated_at: datetime = Field(
        default_factory=now,
        sa_column_kwargs={"comment": "Timestamp when the job was last updated"},
    )

    # Relationships
    organization: Optional["Organization"] = Relationship()
    project: Optional["Project"] = Relationship()


class JobUpdate(SQLModel):
    status: JobStatus | None = None
    error_message: str | None = None
    task_id: str | None = None
