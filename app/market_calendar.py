"""Market Calendar 模組

提供交易日判定與估算功能。

兩種模式：
1. DB 有資料時：從 raw_prices 的 distinct trading_date 取得實際交易日
2. DB 空時：用 estimate_trading_days 估算（排除週末）
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Optional, Set

from sqlalchemy import func, select
from sqlalchemy.orm import Session


def estimate_trading_days(
    start_date: date,
    end_date: date,
    exclude_weekends: bool = True,
) -> List[date]:
    """估算日期範圍內的交易日（排除週末）
    
    注意：此為近似值，未考慮台股國定假日。
    
    Args:
        start_date: 開始日期
        end_date: 結束日期
        exclude_weekends: 是否排除週末
    
    Returns:
        估算的交易日列表（從新到舊排序）
    """
    days = []
    cursor = end_date
    while cursor >= start_date:
        if exclude_weekends:
            if cursor.weekday() < 5:  # 週一到週五
                days.append(cursor)
        else:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return days  # 從新到舊


def estimate_trading_days_count(
    start_date: date,
    end_date: date,
) -> int:
    """估算日期範圍內的交易日數量
    
    Args:
        start_date: 開始日期
        end_date: 結束日期
    
    Returns:
        估算的交易日數量
    """
    total_days = (end_date - start_date).days + 1
    # 約略估算：每 7 天有 5 個交易日
    return int(total_days * 5 / 7)


def get_trading_days_from_db(
    session: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: Optional[int] = None,
) -> List[date]:
    """從 DB 取得實際交易日
    
    Args:
        session: SQLAlchemy session
        start_date: 開始日期（可選）
        end_date: 結束日期（可選）
        limit: 最多回傳筆數
    
    Returns:
        交易日列表（從新到舊排序）
    """
    from app.models import RawPrice
    
    stmt = select(RawPrice.trading_date).distinct().order_by(RawPrice.trading_date.desc())
    
    if start_date:
        stmt = stmt.where(RawPrice.trading_date >= start_date)
    if end_date:
        stmt = stmt.where(RawPrice.trading_date <= end_date)
    if limit:
        stmt = stmt.limit(limit)
    
    rows = session.execute(stmt).fetchall()
    return [row[0] for row in rows]


def get_latest_trading_day(session: Session) -> Optional[date]:
    """取得 DB 中最新的交易日
    
    Args:
        session: SQLAlchemy session
    
    Returns:
        最新交易日，若 DB 空則回傳 None
    """
    from app.models import RawPrice
    
    return session.query(func.max(RawPrice.trading_date)).scalar()


def get_trading_days_set(
    session: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Set[date]:
    """取得交易日集合（用於快速查詢）
    
    Args:
        session: SQLAlchemy session
        start_date: 開始日期
        end_date: 結束日期
    
    Returns:
        交易日 Set
    """
    days = get_trading_days_from_db(session, start_date, end_date)
    return set(days)


def get_recent_trading_days(
    session: Session,
    reference_date: date,
    count: int = 10,
) -> List[date]:
    """取得參考日期之前（含）的最近 N 個交易日
    
    若 DB 空則用估算。
    
    Args:
        session: SQLAlchemy session
        reference_date: 參考日期
        count: 要取得的交易日數量
    
    Returns:
        交易日列表（從新到舊）
    """
    # 先嘗試從 DB 取得
    db_days = get_trading_days_from_db(session, end_date=reference_date, limit=count)
    
    if len(db_days) >= count:
        return db_days[:count]
    
    # DB 資料不足，用估算補足
    # 往前估算 count * 2 天（因為有週末）
    start_estimate = reference_date - timedelta(days=count * 2)
    estimated = estimate_trading_days(start_estimate, reference_date)
    
    # 合併並去重
    all_days = list(set(db_days) | set(estimated))
    all_days.sort(reverse=True)
    
    return all_days[:count]


def calculate_lag_trading_days(
    session: Session,
    db_max_date: date,
    reference_date: date,
) -> int:
    """計算落後的交易日數
    
    Args:
        session: SQLAlchemy session
        db_max_date: DB 中的最新日期
        reference_date: 參考日期（通常是今天）
    
    Returns:
        落後的交易日數（0 表示沒有落後）
    """
    if db_max_date >= reference_date:
        return 0
    
    # 取得 db_max_date 到 reference_date 之間的交易日
    trading_days = get_trading_days_from_db(
        session,
        start_date=db_max_date + timedelta(days=1),
        end_date=reference_date,
    )
    
    if trading_days:
        return len(trading_days)
    
    # DB 沒有資料，用估算
    estimated = estimate_trading_days(
        db_max_date + timedelta(days=1),
        reference_date,
    )
    return len(estimated)
