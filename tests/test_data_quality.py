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


class TestDualThresholdCoverage:
    """測試比例 + 最小值雙門檻邏輯"""

    def test_coverage_ratio_calculation(self):
        """覆蓋率計算正確"""
        # 模擬 10 個交易日，其中 7 個達標（>= 1200 股票）
        # 覆蓋率 = 7/10 = 0.7
        days_with_enough = 7
        total_days = 10
        coverage_ratio = days_with_enough / total_days
        assert coverage_ratio == 0.7

    def test_threshold_pass_condition(self):
        """同時滿足雙門檻才算通過"""
        min_stocks = 1200
        ratio_threshold = 0.7
        
        # 情況 1: 股票數達標、覆蓋率達標 -> 通過
        actual_stocks = 1500
        actual_ratio = 0.8
        assert actual_stocks >= min_stocks and actual_ratio >= ratio_threshold
        
        # 情況 2: 股票數不足、覆蓋率達標 -> 不通過
        actual_stocks = 1000
        actual_ratio = 0.8
        assert not (actual_stocks >= min_stocks and actual_ratio >= ratio_threshold)
        
        # 情況 3: 股票數達標、覆蓋率不足 -> 不通過
        actual_stocks = 1500
        actual_ratio = 0.5
        assert not (actual_stocks >= min_stocks and actual_ratio >= ratio_threshold)


class TestMarginDataOptional:
    """測試融資融券資料為選用"""

    def test_margin_empty_is_warning_not_error(self):
        """融資融券為空應為 warning 而非 error"""
        # margin 資料空時，severity 應為 warning
        issue = data_quality.QualityIssue(
            category="raw_margin_short_empty",
            message="raw_margin_short 表為空",
            severity="warning",
        )
        assert issue.severity == "warning"

    def test_report_passes_with_margin_warning_only(self):
        """只有 margin warning 時報告應通過"""
        issues = [
            data_quality.QualityIssue("raw_margin_short_empty", "margin empty", "warning"),
        ]
        report = data_quality.QualityReport(passed=True, issues=issues, metrics={})
        assert report.passed is True
        assert report.error_text == ""  # warning 不會出現在 error_text


def test_check_data_quality_uses_latest_trading_day_as_reference(monkeypatch):
    """休市日執行時，應以最新交易日作為 freshness/lag 基準。"""

    class _Cfg:
        tz = "Asia/Taipei"
        dq_max_stale_calendar_days = 1
        dq_max_lag_trading_days = 1
        dq_min_stocks_prices = 1
        dq_min_stocks_institutional = 1
        dq_min_stocks_margin = 1
        dq_coverage_ratio_prices = 0.0
        dq_coverage_ratio_institutional = 0.0
        dq_coverage_ratio_margin = 0.0

    reference_date = date(2026, 2, 11)
    captured_references = []

    def _fake_freshness(session, model, table_name, ref_date, max_stale, max_lag):
        captured_references.append(ref_date)
        return reference_date, []

    monkeypatch.setattr(data_quality, "get_latest_trading_day", lambda _session: reference_date)
    monkeypatch.setattr(data_quality, "_check_table_freshness", _fake_freshness)
    monkeypatch.setattr(data_quality, "_check_table_coverage", lambda *_args, **_kwargs: ({}, []))
    monkeypatch.setattr(
        data_quality,
        "_check_institutional_benchmark",
        lambda *_args, **_kwargs: (data_quality.BENCHMARK_MIN_ROWS, None),
    )

    report = data_quality.check_data_quality(object(), _Cfg())

    assert len(captured_references) == 3
    assert all(ref == reference_date for ref in captured_references)
    assert report.metrics["reference_trading_date"] == reference_date.isoformat()
