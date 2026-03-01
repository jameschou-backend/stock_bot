#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


@dataclass
class DatasetReport:
    dataset: str
    min_date: str | None
    max_date: str | None
    symbol_coverage_ratio_last_30d: float
    missing_dates_last_90d: List[str]
    null_rate: Dict[str, float]
    outlier_rate: Dict[str, float]
    fail_reasons: List[str]


def _to_date(value):
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


def _open_trading_days(session, start_date: date, end_date: date) -> List[date]:
    rows = session.execute(
        text(
            """
            SELECT trading_date
            FROM trading_calendar
            WHERE trading_date BETWEEN :start_date AND :end_date
              AND is_open = 1
            ORDER BY trading_date
            """
        ),
        {"start_date": start_date, "end_date": end_date},
    ).fetchall()
    days = [_to_date(row[0]) for row in rows]
    if days:
        return days
    cursor = start_date
    out = []
    while cursor <= end_date:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _closed_days(session, start_date: date, end_date: date) -> List[str]:
    rows = session.execute(
        text(
            """
            SELECT trading_date
            FROM trading_calendar
            WHERE trading_date BETWEEN :start_date AND :end_date
              AND is_open = 0
            ORDER BY trading_date
            """
        ),
        {"start_date": start_date, "end_date": end_date},
    ).fetchall()
    return [_to_date(row[0]).isoformat() for row in rows if _to_date(row[0]) is not None]


def _dataset_range(session, table_name: str, date_col: str = "trading_date") -> Tuple[str | None, str | None]:
    row = session.execute(
        text(f"SELECT MIN({date_col}) AS min_date, MAX({date_col}) AS max_date FROM {table_name}")
    ).fetchone()
    if not row:
        return None, None
    min_val = _to_date(row[0])
    max_val = _to_date(row[1])
    min_date = min_val.isoformat() if min_val else None
    max_date = max_val.isoformat() if max_val else None
    return min_date, max_date


def _daily_universe_count(session) -> int:
    return int(
        session.execute(
            text(
                """
                SELECT COUNT(*) FROM stocks
                WHERE is_listed = 1 AND security_type = 'stock'
                """
            )
        ).scalar()
        or 0
    )


def _symbol_coverage_ratio_last_30d(
    session,
    table_name: str,
    trading_days: List[date],
    denominator: int,
    id_col: str = "stock_id",
) -> float:
    if denominator <= 0 or not trading_days:
        return 0.0
    days = trading_days[-30:]
    if not days:
        return 0.0
    rows = session.execute(
        text(
            f"""
            SELECT trading_date, COUNT(DISTINCT {id_col}) AS cnt
            FROM {table_name}
            WHERE trading_date BETWEEN :start_date AND :end_date
            GROUP BY trading_date
            """
        ),
        {"start_date": min(days), "end_date": max(days)},
    ).fetchall()
    coverage = {_to_date(row[0]): int(row[1]) for row in rows}
    ratios = [min(coverage.get(d, 0) / denominator, 1.0) for d in days]
    return float(sum(ratios) / len(ratios)) if ratios else 0.0


def _missing_dates_last_90d(session, table_name: str, trading_days: List[date]) -> List[str]:
    days = trading_days[-90:]
    if not days:
        return []
    rows = session.execute(
        text(
            f"""
            SELECT DISTINCT trading_date
            FROM {table_name}
            WHERE trading_date BETWEEN :start_date AND :end_date
            """
        ),
        {"start_date": min(days), "end_date": max(days)},
    ).fetchall()
    has_data = {_to_date(row[0]) for row in rows}
    return [d.isoformat() for d in days if d not in has_data]


def _null_and_outlier_rates(session, table_name: str, key_cols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    null_rate: Dict[str, float] = {}
    outlier_rate: Dict[str, float] = {}
    total_rows = int(session.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)
    for col in key_cols:
        if total_rows == 0:
            null_rate[col] = 1.0
            outlier_rate[col] = 0.0
            continue
        null_cnt = int(
            session.execute(text(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL")).scalar() or 0
        )
        null_rate[col] = float(null_cnt / total_rows)
        # outlier: 以負值為異常（價格/量/買賣超）
        out_cnt = int(
            session.execute(text(f"SELECT COUNT(*) FROM {table_name} WHERE {col} < 0")).scalar() or 0
        )
        outlier_rate[col] = float(out_cnt / total_rows)
    return null_rate, outlier_rate


def _generate_report_in_session(days: int, asof: date, config, session) -> Dict:
    start_date = asof - timedelta(days=days)
    trading_days = _open_trading_days(session, start_date, asof)
    universe = _daily_universe_count(session)
    closed_dates_last_90d = _closed_days(session, asof - timedelta(days=90), asof)

    datasets = {
        "raw_prices": {"key_cols": ["open", "high", "low", "close", "volume"], "id_col": "stock_id"},
        "raw_institutional": {
            "key_cols": ["foreign_buy", "foreign_sell", "foreign_net", "trust_net", "dealer_net"],
            "id_col": "stock_id",
        },
        "raw_margin_short": {"key_cols": ["margin_purchase_balance", "short_sale_balance"], "id_col": "stock_id"},
        "raw_fundamentals": {"key_cols": ["revenue_mom", "revenue_yoy"], "id_col": "stock_id"},
        "raw_theme_flow": {"key_cols": ["turnover_amount", "turnover_ratio", "theme_return_20", "hot_score"], "id_col": "theme_id"},
    }
    reports: List[DatasetReport] = []

    for table_name, spec in datasets.items():
        key_cols = spec["key_cols"]
        id_col = spec["id_col"]
        min_date, max_date = _dataset_range(session, table_name)
        ratio = _symbol_coverage_ratio_last_30d(session, table_name, trading_days, universe, id_col=id_col)
        missing_dates = _missing_dates_last_90d(session, table_name, trading_days)
        null_rate, outlier_rate = _null_and_outlier_rates(session, table_name, key_cols)
        fail_reasons: List[str] = []
        if table_name == "raw_prices" and ratio < config.dq_coverage_ratio_prices:
            fail_reasons.append(
                f"coverage_ratio {ratio:.2%} < threshold {config.dq_coverage_ratio_prices:.0%}"
            )
        if table_name == "raw_institutional" and ratio < config.dq_coverage_ratio_institutional:
            fail_reasons.append(
                f"coverage_ratio {ratio:.2%} < threshold {config.dq_coverage_ratio_institutional:.0%}"
            )
        if table_name == "raw_margin_short" and ratio < config.dq_coverage_ratio_margin:
            fail_reasons.append(
                f"coverage_ratio {ratio:.2%} < threshold {config.dq_coverage_ratio_margin:.0%}"
            )
        reports.append(
            DatasetReport(
                dataset=table_name,
                min_date=min_date,
                max_date=max_date,
                symbol_coverage_ratio_last_30d=ratio,
                missing_dates_last_90d=missing_dates,
                null_rate=null_rate,
                outlier_rate=outlier_rate,
                fail_reasons=fail_reasons,
            )
        )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "asof": asof.isoformat(),
        "days": days,
        "data_quality_mode": getattr(config, "data_quality_mode", "strict"),
        "daily_universe_count": universe,
        "trading_days_count": len(trading_days),
        "closed_dates_last_90d": closed_dates_last_90d,
        "datasets": [asdict(r) for r in reports],
    }


def generate_report(days: int, asof: date) -> Dict:
    config = load_config()
    with get_session() as session:
        return _generate_report_in_session(days=days, asof=asof, config=config, session=session)


def write_outputs(payload: Dict) -> Tuple[Path, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_DIR / "data_quality_report.json"
    md_path = ARTIFACT_DIR / "data_quality_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Data Quality Report",
        "",
        f"- asof: `{payload['asof']}`",
        f"- days: `{payload['days']}`",
        f"- mode: `{payload['data_quality_mode']}`",
        f"- daily_universe_count: `{payload['daily_universe_count']}`",
        f"- trading_days_count: `{payload['trading_days_count']}`",
        f"- closed_dates_last_90d_count: `{len(payload['closed_dates_last_90d'])}`",
        "",
    ]
    for ds in payload["datasets"]:
        lines.extend(
            [
                f"## {ds['dataset']}",
                f"- date_coverage: `{ds['min_date']} ~ {ds['max_date']}`",
                f"- symbol_coverage_ratio_last_30d: `{ds['symbol_coverage_ratio_last_30d']:.2%}`",
                f"- missing_dates_last_90d_count: `{len(ds['missing_dates_last_90d'])}`",
                f"- fail_reasons: `{'; '.join(ds['fail_reasons']) if ds['fail_reasons'] else 'none'}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="產出資料品質診斷報表")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--asof", type=str, default=None)
    args = parser.parse_args()

    asof = date.fromisoformat(args.asof) if args.asof else date.today()
    payload = generate_report(args.days, asof)
    json_path, md_path = write_outputs(payload)
    print(f"written: {json_path}")
    print(f"written: {md_path}")


if __name__ == "__main__":
    main()
