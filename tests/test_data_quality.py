"""Data Quality Check 模組單元測試"""

from datetime import date, timedelta

import pytest

from skills import data_quality


class TestGetTaiwanTradingDays:
    """測試交易日估算函數"""

    def test_returns_correct_count(self):
        """應回傳指定數量的交易日"""
        days = data_quality._get_taiwan_trading_days(date(2025, 2, 5), lookback_days=10)
        assert len(days) == 10

    def test_excludes_weekends(self):
        """應排除週六日"""
        days = data_quality._get_taiwan_trading_days(date(2025, 2, 5), lookback_days=5)
        for d in days:
            assert d.weekday() < 5  # 0-4 是週一到週五

    def test_days_are_descending(self):
        """應由近到遠排序"""
        days = data_quality._get_taiwan_trading_days(date(2025, 2, 5), lookback_days=5)
        for i in range(len(days) - 1):
            assert days[i] > days[i + 1]


class TestQualityIssue:
    """測試 QualityIssue 資料類別"""

    def test_default_severity_is_error(self):
        issue = data_quality.QualityIssue(
            category="test_category",
            message="test message",
        )
        assert issue.severity == "error"

    def test_custom_severity(self):
        issue = data_quality.QualityIssue(
            category="test_category",
            message="test message",
            severity="warning",
        )
        assert issue.severity == "warning"


class TestQualityReport:
    """測試 QualityReport 資料類別"""

    def test_error_text_empty_when_passed(self):
        report = data_quality.QualityReport(passed=True, issues=[], metrics={})
        assert report.error_text == ""

    def test_error_text_includes_errors_only(self):
        issues = [
            data_quality.QualityIssue("cat1", "error msg 1", "error"),
            data_quality.QualityIssue("cat2", "warning msg", "warning"),
            data_quality.QualityIssue("cat3", "error msg 2", "error"),
        ]
        report = data_quality.QualityReport(passed=False, issues=issues, metrics={})
        error_text = report.error_text
        assert "error msg 1" in error_text
        assert "error msg 2" in error_text
        assert "warning msg" not in error_text

    def test_error_text_format(self):
        issues = [
            data_quality.QualityIssue("prices_stale", "too old", "error"),
        ]
        report = data_quality.QualityReport(passed=False, issues=issues, metrics={})
        assert "[prices_stale]" in report.error_text
        assert "too old" in report.error_text


class TestBootstrapHistoryDiagnosis:
    """測試 bootstrap_history 的診斷邏輯"""

    def test_should_backfill_when_empty(self):
        from skills import bootstrap_history

        status = bootstrap_history._should_backfill(None, None, None, None, required_days=365)
        assert status.needs_backfill is True
        assert "empty" in status.reason
        assert status.reason_category == "empty"

    def test_should_backfill_when_span_insufficient(self):
        from skills import bootstrap_history

        status = bootstrap_history._should_backfill(
            price_min=date(2025, 1, 1),
            price_max=date(2025, 6, 1),
            inst_min=date(2025, 1, 1),
            inst_max=date(2025, 6, 1),
            required_days=365,
        )
        assert status.needs_backfill is True
        assert status.reason_category == "insufficient"

    def test_should_skip_when_span_ok(self):
        from skills import bootstrap_history

        status = bootstrap_history._should_backfill(
            price_min=date(2024, 1, 1),
            price_max=date(2025, 2, 1),
            inst_min=date(2024, 1, 1),
            inst_max=date(2025, 2, 1),
            required_days=365,
        )
        assert status.needs_backfill is False
        assert status.reason_category == "ok"

    def test_reason_category_for_inst_empty(self):
        from skills import bootstrap_history

        status = bootstrap_history._should_backfill(
            price_min=date(2024, 1, 1),
            price_max=date(2025, 2, 1),
            inst_min=None,
            inst_max=None,
            required_days=365,
        )
        assert status.needs_backfill is True
        assert status.reason_category == "empty"
        assert "raw_institutional empty" in status.reason
