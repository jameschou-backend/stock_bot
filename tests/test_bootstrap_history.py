from datetime import date

from skills import bootstrap_history


def test_should_backfill_when_empty():
    status = bootstrap_history._should_backfill(None, None, None, None, required_days=365)
    assert status.needs_backfill is True
    assert "raw_prices empty" in status.reason


def test_should_backfill_when_span_insufficient():
    status = bootstrap_history._should_backfill(
        price_min=date(2025, 1, 1),
        price_max=date(2025, 6, 1),
        inst_min=date(2025, 1, 1),
        inst_max=date(2025, 6, 1),
        required_days=365,
    )
    assert status.needs_backfill is True


def test_should_skip_when_span_ok():
    status = bootstrap_history._should_backfill(
        price_min=date(2024, 1, 1),
        price_max=date(2025, 2, 1),
        inst_min=date(2024, 1, 1),
        inst_max=date(2025, 2, 1),
        required_days=365,
    )
    assert status.needs_backfill is False
