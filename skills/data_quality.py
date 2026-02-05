"""Data Quality Check 模組

每日 pipeline 必跑的資料品質檢查，若不達標則 fail-fast 並記錄原因。

檢查項目：
1. raw_prices: 最新 trading_date 接近最近交易日，最近 10 交易日每天 distinct stock_id > 門檻
2. raw_institutional: 最新 trading_date 不落後 raw_prices 超過 1 日，
   最近 10 交易日每天 distinct stock_id > 門檻，0050 最近 30 交易日筆數 >= 20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import RawInstitutional, RawPrice


# ---------- 常數設定 ----------
# 最近 N 個交易日的檢查視窗
CHECK_DAYS = 10
# 每日 distinct stock_id 的最低門檻
MIN_DAILY_STOCKS_PRICES = 1200
MIN_DAILY_STOCKS_INSTITUTIONAL = 800
# 0050 資料完整度檢查
BENCHMARK_STOCK_ID = "0050"
BENCHMARK_CHECK_DAYS = 30
BENCHMARK_MIN_ROWS = 20
# 最新資料允許落後的天數上限 (日曆天，考慮週末)
MAX_STALE_CALENDAR_DAYS = 5


@dataclass
class QualityIssue:
    """資料品質問題描述"""

    category: str  # 'prices_stale', 'prices_coverage', 'inst_stale', 'inst_coverage', 'inst_benchmark'
    message: str
    severity: str = "error"  # 'error' or 'warning'


@dataclass
class QualityReport:
    """資料品質檢查結果"""

    passed: bool
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, object] = field(default_factory=dict)

    @property
    def error_text(self) -> str:
        """產生 error_text 供 job 紀錄"""
        errors = [i for i in self.issues if i.severity == "error"]
        if not errors:
            return ""
        return "; ".join(f"[{i.category}] {i.message}" for i in errors)


def _get_taiwan_trading_days(
    reference_date: date, lookback_days: int = CHECK_DAYS
) -> List[date]:
    """估算最近 N 個交易日（排除週六日）
    
    注意：此為近似值，未考慮台股國定假日。
    若需精確台股日曆，可改用第三方套件或 API。
    """
    days = []
    cursor = reference_date
    while len(days) < lookback_days:
        if cursor.weekday() < 5:  # 週一到週五
            days.append(cursor)
        cursor -= timedelta(days=1)
    return days


def _check_prices_freshness(
    session: Session, today: date, max_stale_days: int = MAX_STALE_CALENDAR_DAYS
) -> tuple[Optional[date], Optional[QualityIssue]]:
    """檢查 raw_prices 最新日期是否過時"""
    max_date = session.query(func.max(RawPrice.trading_date)).scalar()
    if max_date is None:
        return None, QualityIssue(
            category="prices_empty",
            message="raw_prices 表為空，請先執行 bootstrap_history 或 ingest_prices",
        )
    
    stale_days = (today - max_date).days
    if stale_days > max_stale_days:
        return max_date, QualityIssue(
            category="prices_stale",
            message=f"raw_prices 最新資料為 {max_date.isoformat()}，已落後 {stale_days} 天 (上限 {max_stale_days} 天)",
        )
    return max_date, None


def _check_prices_coverage(
    session: Session, trading_days: List[date], min_stocks: int = MIN_DAILY_STOCKS_PRICES
) -> tuple[Dict[date, int], Optional[QualityIssue]]:
    """檢查 raw_prices 在最近 N 個交易日的股票數覆蓋率"""
    if not trading_days:
        return {}, None

    stmt = (
        select(RawPrice.trading_date, func.count(func.distinct(RawPrice.stock_id)))
        .where(RawPrice.trading_date.in_(trading_days))
        .group_by(RawPrice.trading_date)
    )
    rows = session.execute(stmt).fetchall()
    coverage = {row[0]: row[1] for row in rows}
    
    # 找出不達標的日期
    low_coverage_days = []
    for d in trading_days:
        count = coverage.get(d, 0)
        if count < min_stocks:
            low_coverage_days.append((d, count))
    
    if low_coverage_days:
        details = ", ".join(f"{d.isoformat()}({c})" for d, c in low_coverage_days[:5])
        return coverage, QualityIssue(
            category="prices_coverage",
            message=f"raw_prices 最近 {len(trading_days)} 交易日中，{len(low_coverage_days)} 天股票數低於 {min_stocks}：{details}",
        )
    return coverage, None


def _check_institutional_freshness(
    session: Session, prices_max_date: Optional[date]
) -> tuple[Optional[date], Optional[QualityIssue]]:
    """檢查 raw_institutional 最新日期是否與 raw_prices 同步"""
    max_date = session.query(func.max(RawInstitutional.trading_date)).scalar()
    if max_date is None:
        return None, QualityIssue(
            category="inst_empty",
            message="raw_institutional 表為空，請先執行 bootstrap_history 或 ingest_institutional",
        )
    
    if prices_max_date is not None:
        lag_days = (prices_max_date - max_date).days
        if lag_days > 1:
            return max_date, QualityIssue(
                category="inst_stale",
                message=f"raw_institutional 最新資料 {max_date.isoformat()} 落後 raw_prices ({prices_max_date.isoformat()}) {lag_days} 天",
            )
    return max_date, None


def _check_institutional_coverage(
    session: Session,
    trading_days: List[date],
    min_stocks: int = MIN_DAILY_STOCKS_INSTITUTIONAL,
) -> tuple[Dict[date, int], Optional[QualityIssue]]:
    """檢查 raw_institutional 在最近 N 個交易日的股票數覆蓋率"""
    if not trading_days:
        return {}, None

    stmt = (
        select(
            RawInstitutional.trading_date,
            func.count(func.distinct(RawInstitutional.stock_id)),
        )
        .where(RawInstitutional.trading_date.in_(trading_days))
        .group_by(RawInstitutional.trading_date)
    )
    rows = session.execute(stmt).fetchall()
    coverage = {row[0]: row[1] for row in rows}
    
    low_coverage_days = []
    for d in trading_days:
        count = coverage.get(d, 0)
        if count < min_stocks:
            low_coverage_days.append((d, count))
    
    if low_coverage_days:
        details = ", ".join(f"{d.isoformat()}({c})" for d, c in low_coverage_days[:5])
        return coverage, QualityIssue(
            category="inst_coverage",
            message=f"raw_institutional 最近 {len(trading_days)} 交易日中，{len(low_coverage_days)} 天股票數低於 {min_stocks}：{details}",
        )
    return coverage, None


def _check_institutional_benchmark(
    session: Session,
    trading_days: List[date],
    stock_id: str = BENCHMARK_STOCK_ID,
    check_days: int = BENCHMARK_CHECK_DAYS,
    min_rows: int = BENCHMARK_MIN_ROWS,
) -> tuple[int, Optional[QualityIssue]]:
    """檢查指標股票 (如 0050) 的資料完整度
    
    用途：驗證 FinMind API 回傳的 dataset 是否為日頻
          若是月/週頻，0050 近 30 交易日筆數會 < 20
    """
    # 擴展 trading_days 到 check_days
    all_days = _get_taiwan_trading_days(trading_days[0] if trading_days else date.today(), check_days)
    
    stmt = (
        select(func.count())
        .where(RawInstitutional.stock_id == stock_id)
        .where(RawInstitutional.trading_date.in_(all_days))
    )
    row_count = session.execute(stmt).scalar() or 0
    
    if row_count < min_rows:
        return row_count, QualityIssue(
            category="inst_benchmark",
            message=f"法人資料可能非日頻：{stock_id} 近 {check_days} 交易日僅 {row_count} 筆 (需 >= {min_rows})",
        )
    return row_count, None


def check_data_quality(session: Session, tz: str = "Asia/Taipei") -> QualityReport:
    """執行完整的資料品質檢查
    
    Args:
        session: SQLAlchemy DB session
        tz: 時區
    
    Returns:
        QualityReport 包含是否通過、問題列表、指標數據
    """
    today = datetime.now(ZoneInfo(tz)).date()
    trading_days = _get_taiwan_trading_days(today, CHECK_DAYS)
    
    issues: List[QualityIssue] = []
    metrics: Dict[str, object] = {
        "check_date": today.isoformat(),
        "trading_days_checked": len(trading_days),
    }
    
    # 1. 檢查 raw_prices 新鮮度
    prices_max_date, issue = _check_prices_freshness(session, today)
    metrics["prices_max_date"] = prices_max_date.isoformat() if prices_max_date else None
    if issue:
        issues.append(issue)
    
    # 2. 檢查 raw_prices 覆蓋率
    prices_coverage, issue = _check_prices_coverage(session, trading_days)
    metrics["prices_coverage"] = {d.isoformat(): c for d, c in prices_coverage.items()}
    if issue:
        issues.append(issue)
    
    # 3. 檢查 raw_institutional 新鮮度
    inst_max_date, issue = _check_institutional_freshness(session, prices_max_date)
    metrics["inst_max_date"] = inst_max_date.isoformat() if inst_max_date else None
    if issue:
        issues.append(issue)
    
    # 4. 檢查 raw_institutional 覆蓋率
    inst_coverage, issue = _check_institutional_coverage(session, trading_days)
    metrics["inst_coverage"] = {d.isoformat(): c for d, c in inst_coverage.items()}
    if issue:
        issues.append(issue)
    
    # 5. 檢查 benchmark 股票（驗證是否日頻）
    benchmark_rows, issue = _check_institutional_benchmark(session, trading_days)
    metrics["inst_benchmark_rows"] = benchmark_rows
    if issue:
        issues.append(issue)
    
    # 判斷是否通過
    has_error = any(i.severity == "error" for i in issues)
    
    return QualityReport(passed=not has_error, issues=issues, metrics=metrics)


def run(config, db_session: Session, **kwargs) -> Dict:
    """Pipeline skill entry point
    
    若資料品質檢查不通過，會將 job 狀態設為 failed 並拋出 ValueError
    """
    job_id = start_job(db_session, "data_quality_check")
    logs: Dict[str, object] = {}
    
    try:
        report = check_data_quality(db_session, config.tz)
        logs.update(report.metrics)
        logs["issues"] = [{"category": i.category, "message": i.message, "severity": i.severity} for i in report.issues]
        logs["passed"] = report.passed
        
        if not report.passed:
            error_text = report.error_text
            finish_job(db_session, job_id, "failed", error_text=error_text, logs=logs)
            raise ValueError(f"Data quality check failed: {error_text}")
        
        finish_job(db_session, job_id, "success", logs=logs)
        return logs
    
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover
        logs["error"] = str(exc)
        finish_job(db_session, job_id, "failed", error_text=str(exc), logs=logs)
        raise
