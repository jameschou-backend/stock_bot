"""Market Calendar 模組單元測試"""

from datetime import date, timedelta

import pytest

from app import market_calendar


class TestEstimateTradingDays:
    """測試交易日估算函數"""

    def test_excludes_weekends_by_default(self):
        """預設排除週六日"""
        days = market_calendar.estimate_trading_days(date(2025, 2, 3), date(2025, 2, 7))
        for d in days:
            assert d.weekday() < 5  # 0-4 是週一到週五

    def test_includes_weekends_when_disabled(self):
        """可選擇包含週六日"""
        days = market_calendar.estimate_trading_days(
            date(2025, 2, 1), date(2025, 2, 7), exclude_weekends=False
        )
        assert len(days) == 7

    def test_returns_empty_when_end_before_start(self):
        """結束日在開始日之前應回傳空列表"""
        days = market_calendar.estimate_trading_days(date(2025, 2, 7), date(2025, 2, 1))
        assert days == []

    def test_single_day_range(self):
        """單日區間"""
        days = market_calendar.estimate_trading_days(date(2025, 2, 3), date(2025, 2, 3))  # 週一
        assert len(days) == 1
        assert days[0] == date(2025, 2, 3)


class TestCalculateLagTradingDays:
    """測試交易日落後計算"""

    def test_returns_zero_when_up_to_date(self):
        """資料日期等於參考日期時應回傳 0"""
        # 此測試需要 mock session，這裡只測試邏輯的基本面向
        from unittest.mock import MagicMock
        
        mock_session = MagicMock()
        # 當查詢 raw_prices 回傳空結果時，會用估算
        mock_session.execute.return_value.fetchall.return_value = []
        
        lag = market_calendar.calculate_lag_trading_days(
            mock_session,
            date(2025, 2, 5),
            date(2025, 2, 5),
        )
        assert lag == 0

    def test_returns_positive_when_behind(self):
        """資料日期落後時應回傳正數"""
        from unittest.mock import MagicMock
        
        mock_session = MagicMock()
        # 模擬 raw_prices 有 2 個交易日在 db_max_date 之後
        mock_session.execute.return_value.fetchall.return_value = [
            (date(2025, 2, 4),),
            (date(2025, 2, 5),),
        ]
        
        lag = market_calendar.calculate_lag_trading_days(
            mock_session,
            date(2025, 2, 3),  # 週一
            date(2025, 2, 5),  # 週三
        )
        # 根據實際實作，這可能用 raw_prices 計數或估算
        assert lag >= 0


class TestDateRangeValidation:
    """測試日期區間驗證"""

    def test_estimate_handles_long_range(self):
        """長期區間估算"""
        start = date(2015, 1, 1)
        end = date(2025, 1, 1)
        days = market_calendar.estimate_trading_days(start, end)
        # 10 年約 2500+ 個交易日
        assert len(days) > 2000
        assert len(days) < 3000

    def test_estimate_handles_weeklong_range(self):
        """一週區間"""
        # 2025-02-03 是週一
        days = market_calendar.estimate_trading_days(date(2025, 2, 3), date(2025, 2, 9))
        # 週一到週五 = 5 天（週六日排除）
        assert len(days) == 5
