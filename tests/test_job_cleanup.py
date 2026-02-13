from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.job_utils import STALE_RUNNING_JOB_ERROR, cleanup_stale_running_jobs
from app.models import Base, Job


def test_cleanup_stale_running_jobs_marks_old_running_as_failed():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        stale = Job(
            job_id="stale-job",
            job_name="ingest_prices",
            status="running",
            started_at=datetime.utcnow() - timedelta(hours=5),
        )
        fresh = Job(
            job_id="fresh-job",
            job_name="ingest_prices",
            status="running",
            started_at=datetime.utcnow() - timedelta(minutes=10),
        )
        done = Job(
            job_id="done-job",
            job_name="ingest_prices",
            status="success",
            started_at=datetime.utcnow() - timedelta(hours=1),
            ended_at=datetime.utcnow() - timedelta(minutes=30),
        )
        session.add_all([stale, fresh, done])
        session.flush()

        cleaned = cleanup_stale_running_jobs(session, stale_minutes=120, commit=False)
        assert cleaned == 1

        stale_ref = session.query(Job).filter(Job.job_id == "stale-job").one()
        fresh_ref = session.query(Job).filter(Job.job_id == "fresh-job").one()
        done_ref = session.query(Job).filter(Job.job_id == "done-job").one()

        assert stale_ref.status == "failed"
        assert stale_ref.error_text == STALE_RUNNING_JOB_ERROR
        assert stale_ref.ended_at is not None
        assert fresh_ref.status == "running"
        assert done_ref.status == "success"


def test_cleanup_uses_heartbeat_over_started_at():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        with_heartbeat = Job(
            job_id="hb-job",
            job_name="ingest_prices",
            status="running",
            started_at=datetime.utcnow() - timedelta(hours=10),
            logs_json={"_heartbeat_at": (datetime.utcnow() - timedelta(minutes=2)).isoformat(timespec="seconds")},
        )
        session.add(with_heartbeat)
        session.flush()

        cleaned = cleanup_stale_running_jobs(session, stale_minutes=30, commit=False)
        assert cleaned == 0
        row = session.query(Job).filter(Job.job_id == "hb-job").one()
        assert row.status == "running"
