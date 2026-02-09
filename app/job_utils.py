from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models import Job


def start_job(session: Session, job_name: str, commit: bool = False) -> str:
    job_id = uuid4().hex
    job = Job(
        job_id=job_id,
        job_name=job_name,
        status="running",
        started_at=datetime.utcnow(),
    )
    session.add(job)
    session.flush()
    if commit:
        session.commit()
    return job_id


def finish_job(
    session: Session,
    job_id: str,
    status: str,
    error_text: Optional[str] = None,
    logs: Optional[Dict[str, Any]] = None,
) -> None:
    session.query(Job).filter(Job.job_id == job_id).update(
        {
            Job.status: status,
            Job.ended_at: datetime.utcnow(),
            Job.error_text: error_text,
            Job.logs_json: logs,
        }
    )
    session.flush()


def update_job(
    session: Session,
    job_id: str,
    status: Optional[str] = None,
    error_text: Optional[str] = None,
    logs: Optional[Dict[str, Any]] = None,
    commit: bool = False,
) -> None:
    payload: Dict[Any, Any] = {}
    if status is not None:
        payload[Job.status] = status
    if error_text is not None:
        payload[Job.error_text] = error_text
    if logs is not None:
        payload[Job.logs_json] = logs
    if not payload:
        return
    session.query(Job).filter(Job.job_id == job_id).update(payload)
    session.flush()
    if commit:
        session.commit()
