from copy import deepcopy

import pytest

from skills.historical_universe_followup import (
    apply_followup, parse_industry_changes, resolve_followup,
)


@pytest.fixture
def evidence():
    base = dict(episodes=[dict(stock_id='1507', market='TWSE', start=None, end='2022-04-21',
                               category='unconfirmed', start_evidence=None)])
    additions = [dict(stock_id='1507', market='TWSE', start='1989-11-09', end='2022-04-21',
                      category='股票')]
    exclusions = [dict(stock_id='1507', market='TWSE', start='2022-04-14', end='2022-04-21')]
    return base, additions, exclusions


@pytest.mark.parametrize('stamp,status', [
    ('1989-11-08', 'outside_verified_intervals'),
    ('1989-11-09', 'identified'),
    ('1989-11-10', 'identified'),
    ('2022-04-12', 'identified'),
    ('2022-04-13', 'identified'),
    ('2022-04-14', 'official_trading_suspension'),
    ('2022-04-15', 'official_trading_suspension'),
    ('2022-04-20', 'official_trading_suspension'),
    ('2022-04-21', 'outside_verified_intervals'),
    ('2022-04-22', 'outside_verified_intervals'),
])
def test_listing_and_suspension_boundaries(evidence, stamp, status):
    before = deepcopy(evidence)
    episodes, intervals = apply_followup(*evidence)
    result = resolve_followup(dict(episodes=episodes, trading_exclusions=intervals), '1507', stamp)
    assert result['status'] == status
    assert result.get('tradable') is not True
    if status == 'official_trading_suspension':
        assert result['tradable'] is False
    assert evidence == before


@pytest.mark.parametrize('field,value', [
    ('start', '1989-11-08'), ('start', '2022-04-21'),
    ('end', '2022-04-20'), ('end', '2022-04-22'), ('stock_id', '1506'),
])
def test_rejects_wrong_exclusion_episode(evidence, field, value):
    base, rows, intervals = evidence
    intervals[0][field] = value
    with pytest.raises(ValueError):
        apply_followup(base, rows, intervals)


def test_industry_reclassification_does_not_remove_prior_listing():
    base = dict(episodes=[dict(stock_id='3313', market='TPEx', start='2026-06-01', end=None,
                               category='股票', start_evidence='current_official_ISIN')])
    rows = [dict(stock_id='3313', market='TPEx', start='2006-05-29', category='股票',
                 replaces_snapshot_start='2026-06-01')]
    episodes, intervals = apply_followup(base, rows, [])
    result = dict(episodes=episodes, trading_exclusions=intervals)
    assert resolve_followup(result, '3313', '2006-05-28')['status'] == 'outside_verified_intervals'
    for stamp in ('2006-05-29', '2006-05-30', '2026-05-31', '2026-06-01', '2026-06-02'):
        assert resolve_followup(result, '3313', stamp)['status'] == 'identified'


def test_2358_exact_start_does_not_use_public_issue_date():
    base = dict(episodes=[dict(stock_id='2358', market='TWSE', start=None, end='2024-11-19',
                               category='unconfirmed', start_evidence=None)])
    episodes, intervals = apply_followup(base, [dict(stock_id='2358', market='TWSE',
        start='1996-12-18', end='2024-11-19', category='股票')], [])
    result = dict(episodes=episodes, trading_exclusions=intervals)
    for stamp in ('1991-12-30', '1996-12-16', '1996-12-17'):
        assert resolve_followup(result, '2358', stamp)['status'] == 'outside_verified_intervals'
    for stamp in ('1996-12-18', '1996-12-19', '1996-12-20'):
        assert resolve_followup(result, '2358', stamp)['status'] == 'identified'


def test_original_2025_typographical_parenthesis_does_not_drop_row():
    text = ('114年6月2日 證券代號不予變更 '
            '萬潤（股票代號：6187由「其他電子業」調整為「半導體業」。'
            '松崗（股票代號：6240）由「其他」」調整為「資訊服務業」。')
    rows = parse_industry_changes(text, year=2025, expected_count=2)
    assert [r['stock_id'] for r in rows] == ['6187', '6240']
    with pytest.raises(ValueError):
        parse_industry_changes(text, year=2025, expected_count=3)
    with pytest.raises(ValueError):
        parse_industry_changes(text + text, year=2025, expected_count=4)
    with pytest.raises(ValueError):
        parse_industry_changes(text, year=2026, expected_count=2)


def test_changed_primary_digest_is_rejected(tmp_path, monkeypatch):
    import scripts.audit_identity_continuation as primary
    import scripts.audit_historical_universe_followup as followup
    monkeypatch.setattr(primary, 'ROOT', tmp_path)
    path = tmp_path / 'source.html'
    path.write_text('replacement')
    with pytest.raises(ValueError, match='missing or changed'):
        primary.checked_file('source.html', '0' * 64, {})
    with pytest.raises(ValueError, match='Unreviewed'):
        followup.verify_bound_source({'source_url': 'https://example.com/unverified'}, {})
