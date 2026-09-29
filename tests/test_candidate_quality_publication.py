import pytest
from scripts.export_candidate_quality import ROOT, period_stats, publication_path, verify_runs


def test_relative_publication_path_is_canonical(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert publication_path('artifacts/forward_simulation/example') == ROOT/'artifacts/forward_simulation/example'
    with pytest.raises(ValueError):
        publication_path(ROOT.parent/'outside-publication')


def test_period_returns_use_opening_nav_and_restart_drawdown_peak():
    rows = [dict(date='2020-12-31', opening_nav=1, nav=200),
            dict(date='2021-01-04', opening_nav=200, nav=250),
            dict(date='2021-01-05', opening_nav=250, nav=225),
            dict(date='2022-01-03', opening_nav=225, nav=50)]
    p = period_stats(rows, '2021-01-01', '2021-12-31')
    assert p['total_return'] == .125
    assert p['max_drawdown'] == pytest.approx(-.1)
    assert p['start_nav'] == 200


def test_publication_rejects_same_run_even_through_alias(tmp_path):
    with pytest.raises(ValueError, match='independent'):
        verify_runs(tmp_path, tmp_path/'.')


def test_empty_period_is_not_zero_return():
    with pytest.raises(ValueError, match='Empty'):
        period_stats([], '2020-01-01', '2020-12-31')
