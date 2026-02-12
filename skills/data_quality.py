"""Data Quality Check 模組

每日 pipeline 必跑的資料品質檢查，若不達標則 fail-fast 並記錄原因。

檢查項目：
1. raw_prices: 新鮮度與覆蓋率（比例 + 最小值雙門檻）
2. raw_institutional: 新鮮度與覆蓋率（比例 + 最小值雙門檻）
3. raw_margin_short: 新鮮度與覆蓋率（比例 + 最小值雙門檻）
4. 落後交易日數檢查（可配置門檻）

門檻設定從 config 讀取，支援以下參數：
- dq_min_stocks_prices: prices 每日最小股票數
- dq_min_stocks_institutional: institutional 每日最小股票數
- dq_min_stocks_margin: margin 每日最小股票數
- dq_coverage_ratio_prices: prices 覆蓋率門檻
- dq_coverage_ratio_institutional: institutional 覆蓋率門檻
- dq_coverage_ratio_margin: margin 覆蓋率門檻
- dq_max_lag_trading_days: 允許最大落後交易日數
- dq_max_stale_calendar_days: 允許最大落後日曆天數
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.job_utils import finish_job, start_job
from app.models import RawInstitutional, RawMarginShort, RawPrice


# ---------- 常數設定（預設值，可被 config 覆蓋）----------
CHECK_DAYS = 10  # 檢查最近 N 個交易日
BENCHMARK_STOCK_ID = "2330"  # 台積電（普通股，確保在 --listed-only 資料集中）
BENCHMARK_CHECK_DAYS = 30
BENCHMARK_MIN_ROWS = 20


@dataclass
class QualityIssue:
    """資料品質問題描述"""

    category: str
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
    """估算最近 N 個交易日（排除週六日）"""
    days = []
    cursor = reference_date
    while len(days) < lookback_days:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return days


def _calculate_lag_trading_days(
    session: Session,
    db_max_date: date,
    reference_date: date,
) -> int:
    """計算落後的交易日數"""
    if db_max_date >= reference_date:
        return 0
    
    # 嘗試從 raw_prices 取得交易日
    stmt = (
        select(func.count(func.distinct(RawPrice.trading_date)))
        .where(RawPrice.trading_date > db_max_date)
        .where(RawPrice.trading_date <= reference_date)
    )
    count = session.execute(stmt).scalar() or 0
    
    if count > 0:
        return count
    
    # 若 raw_prices 沒有資料，用估算
    est_days = _get_taiwan_trading_days(reference_date, 30)
    return len([d for d in est_days if d > db_max_date])


def _check_table_freshness(
    session: Session,
    model,
    table_name: str,
    today: date,
    max_stale_days: int,
    max_lag_trading_days: int,
) -> Tuple[Optional[date], List[QualityIssue]]:
    """檢查表的資料新鮮度"""
    issues = []
    max_date = session.query(func.max(model.trading_date)).scalar()
    
    if max_date is None:
        issues.append(QualityIssue(
            category=f"{table_name}_empty",
            message=f"{table_name} 表為空，請先執行資料回補",
        ))
        return None, issues
    
    # 日曆天落後檢查
    stale_days = (today - max_date).days
    if stale_days > max_stale_days:
        issues.append(QualityIssue(
            category=f"{table_name}_stale",
            message=f"{table_name} 最新資料為 {max_date.isoformat()}，已落後 {stale_days} 日曆天 (上限 {max_stale_days})",
        ))
    
    # 交易日落後檢查
    lag_trading = _calculate_lag_trading_days(session, max_date, today)
    if lag_trading > max_lag_trading_days:
        issues.append(QualityIssue(
            category=f"{table_name}_lag",
            message=f"{table_name} 最新資料為 {max_date.isoformat()}，落後 {lag_trading} 個交易日 (上限 {max_lag_trading_days})",
        ))
    
    return max_date, issues


def _check_table_coverage(
    session: Session,
    model,
    table_name: str,
    trading_days: List[date],
    min_stocks: int,
    coverage_ratio: float,
) -> Tuple[Dict[date, int], List[QualityIssue]]:
    """檢查表在最近 N 個交易日的覆蓋率
    
    雙門檻：覆蓋率 >= coverage_ratio 且股票數 >= min_stocks
    """
    issues = []
    
    if not trading_days:
        return {}, issues

    stmt = (
        select(model.trading_date, func.count(func.distinct(model.stock_id)))
        .where(model.trading_date.in_(trading_days))
        .group_by(model.trading_date)
    )
    rows = session.execute(stmt).fetchall()
    coverage = {row[0]: row[1] for row in rows}
    
    # 計算覆蓋天數比例
    covered_days = len([d for d in trading_days if coverage.get(d, 0) >= min_stocks])
    actual_ratio = covered_days / len(trading_days) if trading_days else 0
    
    # 找出不達標的日期
    low_coverage_days = []
    for d in trading_days:
        count = coverage.get(d, 0)
        if count < min_stocks:
            low_coverage_days.append((d, count))
    
    # 雙門檻檢查
    if actual_ratio < coverage_ratio:
        details = ", ".join(f"{d.isoformat()}({c})" for d, c in low_coverage_days[:5])
        issues.append(QualityIssue(
            category=f"{table_name}_coverage",
            message=f"{table_name} 覆蓋率 {actual_ratio:.1%} < {coverage_ratio:.0%}，"
                   f"最近 {len(trading_days)} 交易日中 {len(low_coverage_days)} 天低於 {min_stocks}：{details}",
        ))
    
    return coverage, issues


def _check_institutional_benchmark(
    session: Session,
    trading_days: List[date],
    stock_id: str = BENCHMARK_STOCK_ID,
    check_days: int = BENCHMARK_CHECK_DAYS,
    min_rows: int = BENCHMARK_MIN_ROWS,
) -> Tuple[int, Optional[QualityIssue]]:
    """檢查指標股票的資料完整度（驗證是否日頻）"""
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


def check_data_quality(session: Session, config) -> QualityReport:
    """執行完整的資料品質檢查
    
    Args:
        session: SQLAlchemy DB session
        config: AppConfig（從中讀取門檻設定）
    
    Returns:
        QualityReport 包含是否通過、問題列表、指標數據
    """
    today = datetime.now(ZoneInfo(config.tz)).date()
    trading_days = _get_taiwan_trading_days(today, CHECK_DAYS)
    
    issues: List[QualityIssue] = []
    metrics: Dict[str, object] = {
        "check_date": today.isoformat(),
        "trading_days_checked": len(trading_days),
    }
    
    # 取得門檻設定
    max_stale_days = config.dq_max_stale_calendar_days
    max_lag_trading = config.dq_max_lag_trading_days
    
    # === 1. raw_prices 檢查 ===
    prices_max_date, price_issues = _check_table_freshness(
        session, RawPrice, "raw_prices", today, max_stale_days, max_lag_trading
    )
    metrics["prices_max_date"] = prices_max_date.isoformat() if prices_max_date else None
    issues.extend(price_issues)
    
    if prices_max_date:
        prices_coverage, cov_issues = _check_table_coverage(
            session, RawPrice, "raw_prices", trading_days,
            config.dq_min_stocks_prices, config.dq_coverage_ratio_prices
        )
        metrics["prices_coverage"] = {d.isoformat(): c for d, c in prices_coverage.items()}
        issues.extend(cov_issues)
    
    # === 2. raw_institutional 檢查 ===
    inst_max_date, inst_issues = _check_table_freshness(
        session, RawInstitutional, "raw_institutional", today, max_stale_days, max_lag_trading
    )
    metrics["inst_max_date"] = inst_max_date.isoformat() if inst_max_date else None
    issues.extend(inst_issues)
    
    if inst_max_date:
        # 檢查是否與 prices 同步
        if prices_max_date and inst_max_date:
            lag = (prices_max_date - inst_max_date).days
            if lag > 1:
                issues.append(QualityIssue(
                    category="inst_sync",
                    message=f"raw_institutional ({inst_max_date}) 落後 raw_prices ({prices_max_date}) {lag} 天",
                ))
        
        inst_coverage, cov_issues = _check_table_coverage(
            session, RawInstitutional, "raw_institutional", trading_days,
            config.dq_min_stocks_institutional, config.dq_coverage_ratio_institutional
        )
        metrics["inst_coverage"] = {d.isoformat(): c for d, c in inst_coverage.items()}
        issues.extend(cov_issues)
        
        # Benchmark 檢查
        benchmark_rows, bench_issue = _check_institutional_benchmark(session, trading_days)
        metrics["inst_benchmark_rows"] = benchmark_rows
        if bench_issue:
            issues.append(bench_issue)
    
    # === 3. raw_margin_short 檢查 ===
    margin_max_date, margin_issues = _check_table_freshness(
        session, RawMarginShort, "raw_margin_short", today, max_stale_days, max_lag_trading
    )
    metrics["margin_max_date"] = margin_max_date.isoformat() if margin_max_date else None
    
    # margin 資料可能不是必要的，降級為 warning
    for issue in margin_issues:
        if issue.category == "raw_margin_short_empty":
            issue.severity = "warning"
            issue.message += "（融資融券資料為選用）"
    issues.extend(margin_issues)
    
    if margin_max_date:
        margin_coverage, cov_issues = _check_table_coverage(
            session, RawMarginShort, "raw_margin_short", trading_days,
            config.dq_min_stocks_margin, config.dq_coverage_ratio_margin
        )
        metrics["margin_coverage"] = {d.isoformat(): c for d, c in margin_coverage.items()}
        # margin 覆蓋率不足也降級為 warning
        for issue in cov_issues:
            issue.severity = "warning"
        issues.extend(cov_issues)
    
    # === 4. 計算整體指標 ===
    metrics["total_issues"] = len(issues)
    metrics["error_count"] = len([i for i in issues if i.severity == "error"])
    metrics["warning_count"] = len([i for i in issues if i.severity == "warning"])
    
    # 判斷是否通過（只看 error）
    has_error = any(i.severity == "error" for i in issues)
    
    return QualityReport(passed=not has_error, issues=issues, metrics=metrics)


def run(config, db_session: Session, **kwargs) -> Dict:
    """Pipeline skill entry point
    
    若資料品質檢查不通過（有 error），會將 job 狀態設為 failed 並拋出 ValueError
    """
    job_id = start_job(db_session, "data_quality_check")
    logs: Dict[str, object] = {}
    
    try:
        report = check_data_quality(db_session, config)
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
