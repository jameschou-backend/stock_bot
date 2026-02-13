from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models import Job


STALE_RUNNING_JOB_ERROR = "stale_running_job: auto-closed"


def cleanup_stale_running_jobs(
    session: Session,
    stale_minutes: int = 120,
    commit: bool = False,
) -> int:
    """將長時間卡在 running 的 job 自動收斂為 failed。"""
    cutoff = datetime.utcnow() - timedelta(minutes=max(stale_minutes, 1))
    running_rows = (
        session.query(Job)
        .filter(Job.status == "running")
        .all()
    )
    if not running_rows:
        return 0

    stale_rows = []
    for row in running_rows:
        heartbeat_at = None
        if isinstance(row.logs_json, dict):
            raw = row.logs_json.get("_heartbeat_at")
            if isinstance(raw, str):
                try:
                    heartbeat_at = datetime.fromisoformat(raw)
                except Exception:
                    heartbeat_at = None
        ref_time = heartbeat_at or row.started_at
        if ref_time is not None and ref_time < cutoff:
            stale_rows.append(row)

    if not stale_rows:
        return 0

    now = datetime.utcnow()
    for row in stale_rows:
        row.status = "failed"
        row.ended_at = now
        row.error_text = STALE_RUNNING_JOB_ERROR
    session.flush()
    if commit:
        session.commit()
    return len(stale_rows)


def start_job(session: Session, job_name: str, commit: bool = False) -> str:
    cleanup_stale_running_jobs(session, stale_minutes=120, commit=False)
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
        if isinstance(logs, dict):
            logs = dict(logs)
            logs["_heartbeat_at"] = datetime.utcnow().isoformat(timespec="seconds")
        payload[Job.logs_json] = logs
    if not payload:
        return
    session.query(Job).filter(Job.job_id == job_id).update(payload)
    session.flush()
    if commit:
        session.commit()
