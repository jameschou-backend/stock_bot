"""skills.feature_utils.align_news_to_trading_day 測試。

驗證消除同日 lookahead 的對齊邏輯：盤中→當日、盤後(>=13:30)→次一交易日、
週末→下週一、超出範圍→NaT、順序保留。
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from skills.feature_utils import align_news_to_trading_day


# 模擬交易日（週一~週五 + 下週一），6/13、6/14 為週末
TRADING_DAYS = [
    date(2026, 6, 8), date(2026, 6, 9), date(2026, 6, 10),
    date(2026, 6, 11), date(2026, 6, 12), date(2026, 6, 15),
]


def test_intraday_news_maps_to_same_day():
    # 週一 10:00（盤中 <13:30）→ T 日收盤時已知 → 當日
    out = align_news_to_trading_day([datetime(2026, 6, 8, 10, 0)], TRADING_DAYS)
    assert out.iloc[0] == date(2026, 6, 8)


def test_after_close_news_maps_to_next_trading_day():
    # 週一 14:00（盤後 >=13:30）→ 次一交易日（週二）
    out = align_news_to_trading_day([datetime(2026, 6, 8, 14, 0)], TRADING_DAYS)
    assert out.iloc[0] == date(2026, 6, 9)


def test_exactly_at_close_counts_as_after():
    # 正好 13:30 視為盤後 → 次日
    out = align_news_to_trading_day([datetime(2026, 6, 8, 13, 30)], TRADING_DAYS)
    assert out.iloc[0] == date(2026, 6, 9)


def test_friday_after_close_maps_to_monday():
    # 週五 15:00 盤後 → 下週一（6/13、6/14 非交易日）
    out = align_news_to_trading_day([datetime(2026, 6, 12, 15, 0)], TRADING_DAYS)
    assert out.iloc[0] == date(2026, 6, 15)


def test_weekend_news_maps_to_monday():
    # 週六上午新聞 → 下週一
    out = align_news_to_trading_day([datetime(2026, 6, 13, 9, 0)], TRADING_DAYS)
    assert out.iloc[0] == date(2026, 6, 15)


def test_beyond_calendar_returns_nat():
    # 最後一個交易日盤後 → 次交易日尚未到 → NaT（避免對齊到不存在的未來）
    out = align_news_to_trading_day([datetime(2026, 6, 15, 14, 0)], TRADING_DAYS)
    assert pd.isna(out.iloc[0])


def test_order_preserved():
    news = [datetime(2026, 6, 9, 10, 0), datetime(2026, 6, 8, 14, 0)]
    out = align_news_to_trading_day(news, TRADING_DAYS)
    assert out.iloc[0] == date(2026, 6, 9)   # 盤中→當日
    assert out.iloc[1] == date(2026, 6, 9)   # 週一盤後→週二


def test_empty_input():
    out = align_news_to_trading_day([], TRADING_DAYS)
    assert len(out) == 0
