from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import Pick


REPORTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "reports"


@dataclass(frozen=True)
class ReportResult:
    rows: int
    dir: str
    csv_path: str
    html_path: str
    pick_date: str


def _resolve_pick_date(config, db_session: Session, pick_date: Optional[date]) -> date:
    if pick_date:
        return pick_date
    latest = db_session.query(func.max(Pick.pick_date)).scalar()
    if latest is None:
        return datetime.now(ZoneInfo(config.tz)).date()
    return latest


def _normalize_reason(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def _format_reason_preview(reason: Dict, max_items: int = 3) -> str:
    items = list(reason.items())[:max_items]
    return ", ".join(f"{k}: {v}" for k, v in items)


def export_latest_report(config, db_session: Session, pick_date: Optional[date] = None) -> ReportResult:
    target_date = _resolve_pick_date(config, db_session, pick_date)
    report_dir = REPORTS_DIR / target_date.isoformat()
    report_dir.mkdir(parents=True, exist_ok=True)

    picks = (
        db_session.query(Pick)
        .filter(Pick.pick_date == target_date)
        .order_by(Pick.score.desc())
        .limit(config.topn)
        .all()
    )

    rows: List[Dict] = []
    for row in picks:
        reason = _normalize_reason(row.reason_json)
        rows.append(
            {
                "stock_id": row.stock_id,
                "score": float(row.score) if row.score is not None else None,
                "model_id": row.model_id,
                "pick_date": row.pick_date.isoformat(),
                "reason_json": json.dumps(reason, ensure_ascii=False),
                "reason_preview": _format_reason_preview(reason),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["stock_id", "score", "model_id", "pick_date", "reason_json", "reason_preview"])

    csv_path = report_dir / "candidates.csv"
    html_path = report_dir / "report.html"
    df[["stock_id", "score", "model_id", "pick_date", "reason_json"]].to_csv(csv_path, index=False)

    html_rows = df[["stock_id", "score", "model_id", "reason_preview"]].to_html(
        index=False, escape=True, border=1
    )
    html_body = (
        f"<html><head><meta charset='utf-8'></head><body>"
        f"<h2>Daily Picks ({target_date.isoformat()})</h2>"
        f"{html_rows if len(df) else '<p>No picks for this date.</p>'}"
        f"</body></html>"
    )
    html_path.write_text(html_body, encoding="utf-8")

    return ReportResult(
        rows=len(picks),
        dir=str(report_dir),
        csv_path=str(csv_path),
        html_path=str(html_path),
        pick_date=target_date.isoformat(),
    )


def run(config, db_session: Session, pick_date: Optional[date] = None, **kwargs) -> Dict:
    job_id = start_job(db_session, "export_report")
    try:
        result = export_latest_report(config, db_session, pick_date=pick_date)
        logs = {
            "rows": result.rows,
            "dir": result.dir,
            "csv_path": result.csv_path,
            "html_path": result.html_path,
            "pick_date": result.pick_date,
        }
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    except Exception as exc:  # pragma: no cover - exercised by pipeline
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs={"error": str(exc)})
        raise
