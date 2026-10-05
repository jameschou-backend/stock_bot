"""Differential serializer oracle frozen before the numpy lookup optimization.

The old scan entry point is retained only as an offline oracle. Features are
computed by the same unchanged engine; the comparison covers every emitted
value/order/type rather than relying on a small hand-picked return fixture.
"""
import json
import time

import numpy as np
import pandas as pd
import pytest

from skills.strategy_scanner import engine


LEGACY_SCAN = 'def scan_market(bars, calendar, *, start, end, names=None, original_signals=(),\n                poc=None, provenance=None, strategies=None):\n    """Return one outcome for every stock/date/active strategy, no account state.\n\n    `original_signals` must be the complete hash-bound candidate ledger, not\n    trades. Without that evidence original adapters return unknown, not False.\n    Caller supplies the full market calendar and at least 420 prior sessions.\n    Publication timestamps for optional external evidence belong in its adapter.\n    """\n    start, end = _day(start), _day(end)\n    if start>end:\n        raise ValueError(\'Scan start must not exceed end\')\n    f, days, ids = _prepare(bars, calendar, end)\n    if end not in days or start not in days:\n        raise ValueError(\'Scan endpoints must be observed market sessions\')\n    provenance = dict(provenance or {})\n    if provenance.get(\'source_end\') and end>_day(provenance[\'source_end\']):\n        raise ValueError(\'Scan exceeds frozen source coverage\')\n    z, masks, catalog, evidence = _compile_rules(f, days, ids,\n        original_signals=original_signals, poc=poc, provenance=provenance)\n    active = {s[\'id\']:s for s in catalog if s[\'status\']==\'active\'}\n    chosen = list(ACTIVE_IDS if strategies is None else strategies)\n    if len(set(chosen)) != len(chosen) or not chosen or set(chosen)-set(active):\n        raise ValueError(\'Select unique registered active strategies\')\n    priorities, pstate, before, after, red = (evidence[k] for k in\n        (\'priorities\', \'pstate\', \'before\', \'after\', \'red\'))\n    eligible = f[\'eligible\'].eq(True).fillna(False)\n    dates = []\n    stock_ids = [sid for sid in ids if not sid.startswith(\'0\')]\n    date_index = {d:i for i,d in enumerate(days)}\n    for d in days[(days>=start)&(days<=end)]:\n        i=date_index[d]; stocks=[]; counts=Counter()\n        market=\'unknown\'\n        if \'0050\' in ids and z[\'regime_known\'].at[d,\'0050\']:\n            market=\'trend_up\' if z[\'uptrend\'].at[d,\'0050\'] else \'trend_down\' if z[\'downtrend\'].at[d,\'0050\'] else \'range\'\n        for sid in stock_ids:\n            regime=\'unknown\'\n            if z[\'regime_known\'].at[d,sid]:\n                regime=\'trend_up\' if z[\'uptrend\'].at[d,sid] else \'trend_down\' if z[\'downtrend\'].at[d,sid] else \'range\'\n            results={}\n            for key in chosen:\n                match, known, fields, rule = masks[key]\n                metrics={field:_number(z[field].at[d,sid]) for field in fields}\n                first=None\n                if f[\'eligible\'].at[d,sid] is False or (pd.notna(f[\'eligible\'].at[d,sid]) and not bool(f[\'eligible\'].at[d,sid])):\n                    state=\'ineligible\'; reasons=[\'該日市場身分／原資料資格不符合個股掃描範圍\']\n                elif not eligible.at[d,sid] or not known.at[d,sid]:\n                    state=\'unknown\'; reasons=[\'資料不足、品質衝突或歷史窗口不完整，未判定為不符合\']\n                else:\n                    state=\'matched\' if match.at[d,sid] else \'not_matched\'\n                    reasons=[rule if state==\'matched\' else \'未同時符合：\'+rule]\n                    if state==\'matched\' and i>0 and known.iloc[i-1][sid] and eligible.iloc[i-1][sid]:\n                        first=not bool(match.iloc[i-1][sid])\n                if key in (\'original_breakout\',\'original_red\',\'poc_red_priority\',\'poc_up_red\'):\n                    metrics[\'original_priority\']=_number(priorities.at[d,sid])\n                if key in (\'poc_red_priority\',\'poc_up_red\'):\n                    metrics.update(poc_before=_number(before.at[d,sid]),poc_after=_number(after.at[d,sid]),poc_status=pstate.at[d,sid])\n                    if red.at[d,sid] and pstate.at[d,sid] not in (\'up\',\'down\'):\n                        reasons.append(\'POC資料不足；不能宣稱籌碼成本已上移\')\n                preferred=active[key].get(\'preferred_regimes\',[])\n                results[key]=dict(status=state,reasons=reasons,metrics=metrics,first_signal=first,\n                    regime_fit=None if regime==\'unknown\' else (regime in preferred if preferred else True))\n                counts[state]+=1\n            stocks.append(dict(stock_id=sid,name=(names or {}).get(sid,sid),regime=regime,results=results))\n        dates.append(dict(date=str(d.date()),market_regime=market,stocks=stocks,counts=dict(counts)))\n    return dict(schema=\'multi_strategy_scan_v1\',start=str(start.date()),end=str(end.date()),\n        source_end=provenance.get(\'source_end\',str(days[-1].date())),strategies=catalog,days=dates,\n        evaluated_strategy_ids=chosen,provenance=provenance,live_qualified=False,\n        account_independent=True,signal_timing=\'T_close_confirmed__earliest_next_market_session\',\n        execution_model=None,portfolio_model=None,returns_inherited=False,\n        context_policy=\'regime_fit_is_descriptive_not_a_validated_strategy_router\')\n'


def old_scan():
    namespace = dict(vars(engine))
    exec(compile(LEGACY_SCAN, "<pre_numpy_scan_market>", "exec"), namespace)
    return namespace["scan_market"]


def fixture(n=430, stock_count=4):
    days = pd.bdate_range("2024-01-02", periods=n)
    ids = ["0050", *[str(1101+i) for i in range(stock_count-1)]]
    records = []
    for j, sid in enumerate(ids):
        for i, day in enumerate(days):
            close = 100+i*.17+np.sin(i*.27+j)*4
            records.append(dict(date=day, stock_id=sid, open=close-.3,
                high=close+.8, low=close-.9, close=close,
                adjusted_close=close*.973, volume=1000000+(i%9)*100000,
                amount=close*1000000, quality=True, eligible=True))
    bars = pd.DataFrame(records)
    bars['eligible'] = bars['eligible'].astype('boolean')
    events = []
    poc = []
    for offset, state in zip(range(-5, 0), ['up', 'down', 'unknown', 'pending_data', 'up']):
        idx = len(days)+offset
        events.append(dict(signal_date=str(days[idx].date()), members=['1101'], priority=.314))
        row = dict(signal_date=str(days[idx].date()), stock_id='1101', status=state)
        if state in ('up', 'down'):
            row.update(source_date_end=str(days[idx-1].date()), prior_dates=[str(d.date()) for d in days[idx-20:idx]],
                       poc_before=100., poc_after=110. if state=='up' else 100., available=True)
        poc.append(row)
    kwargs = dict(start=str(days[-5].date()), end=str(days[-1].date()),
        names={'1101':'中文股票'}, original_signals=events, poc=poc,
        provenance=dict(source_end=str(days[-1].date()), original_candidates_complete=True))
    return bars, days, kwargs


def exact(a, b):
    # No tolerance: including all numbers, None, booleans, keys and list order.
    assert a == b
    assert json.dumps(a, ensure_ascii=False, allow_nan=False) == json.dumps(b, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize('nullable_identity', [False, True])
def test_complete_old_new_scan_calls_have_exact_json_equality(nullable_identity):
    bars, days, kwargs = fixture()
    bars.loc[(bars.stock_id=='1102') & bars.date.eq(days[-2]), 'eligible'] = False
    bars.loc[(bars.stock_id=='1103') & bars.date.eq(days[-3]), 'quality'] = False
    bars = bars[~(bars.stock_id.eq('1103') & bars.date.eq(days[-4]))].copy()
    if nullable_identity:
        bars.loc[(bars.stock_id=='1102') & bars.date.eq(days[-1]), 'eligible'] = pd.NA
    expected = old_scan()(bars, days, **kwargs)
    got = engine.scan_market(bars, days, **kwargs)
    exact(got, expected)
    states = {r['status'] for d in got['days'] for s in d['stocks'] for r in s['results'].values()}
    assert states == {'matched', 'not_matched', 'unknown', 'ineligible'}
    first_values = {r['first_signal'] for d in got['days'] for s in d['stocks'] for r in s['results'].values()}
    assert first_values == {True, False, None}


def test_subset_output_order_and_first_calendar_day_are_unchanged():
    bars, days, kwargs = fixture(30)
    kwargs.update(start=str(days[0].date()), strategies=['poc_up_red','original_red','ma20_60_cross'])
    exact(engine.scan_market(bars, days, **kwargs), old_scan()(bars, days, **kwargs))
    result = engine.scan_market(bars, days, **kwargs)
    assert list(result['days'][0]['stocks'][0]['results']) == kwargs['strategies']
    assert all(r['first_signal'] is None for s in result['days'][0]['stocks'] for r in s['results'].values())


def freeze_matrices(monkeypatch, stock_count=4):
    bars, days, kwargs = fixture(150, stock_count=stock_count)
    prepared = engine._prepare(bars, days, pd.Timestamp(kwargs['end']))
    f, full_days, ids = prepared
    compiled = engine._compile_rules(f, full_days, ids, original_signals=kwargs['original_signals'],
                                    poc=kwargs['poc'], provenance=kwargs['provenance'])
    monkeypatch.setattr(engine, '_prepare', lambda *args, **kw: prepared)
    monkeypatch.setattr(engine, '_compile_rules', lambda *args, **kw: compiled)
    return bars, days, kwargs, prepared, compiled


def test_serializer_has_no_pandas_scalar_or_row_indexer_calls(monkeypatch):
    bars, days, kwargs, _, _ = freeze_matrices(monkeypatch)
    expected = old_scan()(bars, days, **kwargs)
    def forbidden(self):
        raise AssertionError('Serialization must not perform a pandas per-stock scalar/row lookup')
    monkeypatch.setattr(pd.DataFrame, 'at', property(forbidden))
    monkeypatch.setattr(pd.DataFrame, 'iloc', property(forbidden))
    exact(engine.scan_market(bars, days, **kwargs), expected)


def test_precompiled_matrices_are_not_mutated_and_serialization_is_differentially_timed(monkeypatch):
    bars, days, kwargs, prepared, compiled = freeze_matrices(monkeypatch, stock_count=32)
    f, _, _ = prepared
    z, masks, _, evidence = compiled
    matrices = [*f.values(), *z.values(), *evidence.values(), *[x for mask in masks.values() for x in mask[:2]]]
    copies = [x.copy(deep=True) for x in matrices]
    start = time.perf_counter(); expected = old_scan()(bars, days, **kwargs); old_seconds = time.perf_counter()-start
    start = time.perf_counter(); got = engine.scan_market(bars, days, **kwargs); new_seconds = time.perf_counter()-start
    exact(got, expected)
    for before, after in zip(copies, matrices):
        pd.testing.assert_frame_equal(before, after)
    # Informative timing only; no flaky CPU/wall-clock threshold in the test.
    print(f'precompiled serialize: old={old_seconds:.4f}s new={new_seconds:.4f}s ratio={old_seconds/new_seconds:.2f}x')
