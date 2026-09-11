import pandas as pd
import pytest
from app.finmind import FinMindError
from skills.holding_validation import TIERS, aggregate
from skills.ingest_holding_dist import _aggregate_holding


def sample():
    rows = [dict(stock_id='2492', date='2026-04-02', HoldingSharesLevel=t.replace('1000001+', 'more than 1,000,001'),
                 people=1, unit=100, percent=round(100/15, 2)) for t in TIERS]
    rows += [dict(stock_id='2492', date='2026-04-02', HoldingSharesLevel='total', people=15, unit=1500, percent=100)]
    return pd.DataFrame(rows)


def test_total_not_investor_and_highest_tier():
    row = _aggregate_holding(sample()).iloc[0]
    assert row.holder_count == 15
    assert row.large_holder_pct == row.top_level_pct == .0667
    assert row.small_holder_pct == .9333
    assert str(row.available_date) == '2026-04-10'


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unknown', 'percent', 'unit', 'people', 'nan'])
def test_bad_evidence_fails_closed(kind):
    raw = sample()
    if kind == 'missing': raw = raw.iloc[1:]
    elif kind == 'duplicate': raw = pd.concat([raw, raw.iloc[[0]]])
    elif kind == 'unknown': raw.loc[0, 'HoldingSharesLevel'] = 'mystery'
    elif kind == 'nan': raw.loc[0, 'percent'] = float('nan')
    else: raw.loc[0, kind] = 99
    with pytest.raises(FinMindError): aggregate(raw)


def test_stock_validation_and_empty_allowlist():
    assert aggregate(sample(), set()).empty
    raw = sample().assign(stock_id='2492A')
    assert aggregate(raw).empty
