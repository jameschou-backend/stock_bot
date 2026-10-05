"""Research adapters: numeric boundaries, causal windows and explicit unknowns."""
import numpy as np
import pandas as pd
import pytest

from skills.strategy_scanner.engine import _features, _prepare
from skills.strategy_scanner.research_rules import (
    RESEARCH_CATALOG, RESEARCH_IDS, VERSION, add_research_rules,
)


def inputs(n=430, close=None, volume=None):
    days = pd.bdate_range('2024-01-02', periods=n)
    close = np.full(n, 100.) if close is None else np.asarray(close, dtype=float)
    volume = np.full(n, 1_000_000.) if volume is None else np.asarray(volume, dtype=float)
    rows = []
    for sid in ('0050', '2330'):
        prices = np.full(n, 100.) if sid == '0050' else close
        for i, (day, c) in enumerate(zip(days, prices)):
            v = volume[i] if sid == '2330' else 1_000_000.
            rows.append(dict(date=day, stock_id=sid, open=c-.2, high=c+.5,
                low=c-.5, close=c, volume=v, amount=c*v, adjusted_close=c,
                quality=True, eligible=True))
    return pd.DataFrame(rows), days


def calculate(bars, days):
    f, _, _ = _prepare(bars, days, days[-1])
    z = _features(f)
    rules = {}
    def add(identifier, match, fields, rule, *, known=None):
        assert identifier not in rules
        available = f['valid'] & f['eligible'].eq(True).fillna(False)
        for key in fields:
            available &= np.isfinite(z[key])
        if known is not None:
            available &= known
        rules[identifier] = dict(match=match, known=available.fillna(False), fields=fields, rule=rule)
    add_research_rules(f, z, add)
    return rules, z


def state(rules, identifier, offset=-1, sid='2330'):
    row = rules[identifier]
    if not row['known'].iloc[offset][sid]:
        return 'unknown'
    return 'matched' if row['match'].iloc[offset][sid] else 'not_matched'


def replace_stock(bars, days, index, **values):
    mask = bars.stock_id.eq('2330') & bars.date.eq(days[index])
    for key, value in values.items():
        bars.loc[mask, key] = value


def pulse_inputs():
    n=220
    close=np.full(n,100.);volume=np.full(n,1_000_000.)
    for i in (140,151,162,183):
        close[i]=103.;volume[i]=6_000_000.
    bars,days=inputs(n,close,volume)
    for i in (140,151,162,183):
        replace_stock(bars,days,i,open=100.,high=103.5,low=99.5)
    return bars,days


def test_twelve_registered_rules_have_complete_research_metadata():
    assert tuple(x['id'] for x in RESEARCH_CATALOG) == RESEARCH_IDS
    assert len(RESEARCH_IDS) == len(set(RESEARCH_IDS)) == 12
    required = {'id','name','family','kind','status','description','required_data',
        'preferred_regimes','version','source_paths','source_urls','variants','data_gaps',
        'reusable_interfaces','live_qualified','returns_inherited'}
    for row in RESEARCH_CATALOG:
        assert required <= row.keys()
        assert row['status']=='active' and row['version']==VERSION
        assert row['live_qualified'] is False and row['returns_inherited'] is False
        assert row['kind'] == ('filter' if row['id'].startswith(('entry_', 'liquidity_')) else 'entry')
    b,d=inputs();rules,_=calculate(b,d)
    assert tuple(rules)==RESEARCH_IDS
    assert all('amount20' in rule['fields'] for rule in rules.values())


def test_legacy_momentum_uses_inclusive_volume_and_ten_percent_return():
    close=np.r_[np.full(350,100.),np.linspace(100.,200.,80)]
    volume=np.r_[np.full(410,1_000_000.),np.full(20,2_000_000.)]
    b,d=inputs(close=close,volume=volume);rules,z=calculate(b,d)
    assert state(rules,'legacy_momentum_trend')=='matched'
    assert z['research_volume20_inclusive'].iloc[-1]['2330']==2_000_000.
    b.loc[b.stock_id.eq('2330'),'volume']=1_000_000.
    b.loc[b.stock_id.eq('2330'),'amount']=b.loc[b.stock_id.eq('2330'),'close']*1_000_000.
    rules,_=calculate(b,d)
    assert state(rules,'legacy_momentum_trend')=='not_matched'  # Equal averages do not pass.


def test_legacy_mean_reversion_simple_rsi_and_sample_bollinger():
    close=np.r_[np.full(425,100.),[99.,98.,97.,96.,80.]]
    b,d=inputs(close=close);rules,z=calculate(b,d)
    assert state(rules,'legacy_mean_reversion')=='matched'
    assert z['research_rsi14_simple'].iloc[-1]['2330']==0.
    expected=close[-20:].mean()-2*close[-20:].std(ddof=1)
    assert z['research_bb_lower_sample'].iloc[-1]['2330']==pytest.approx(expected)


def test_rsi_flat_or_missing_changes_are_not_filled_with_zero():
    b,d=inputs();rules,z=calculate(b,d)
    assert np.isnan(z['research_rsi14_simple'].iloc[-1]['2330'])
    assert state(rules,'legacy_mean_reversion')=='unknown'
    replace_stock(b,d,-5,quality=False)
    _,z=calculate(b,d)
    assert np.isnan(z['research_rsi14_simple'].iloc[-1]['2330'])


def test_course_breakout_includes_today_but_requires_400_days_and_500m():
    b,d=inputs(close=np.linspace(100.,200.,430),volume=np.full(430,4_000_000.))
    rules,z=calculate(b,d)
    assert state(rules,'legacy_course_breakout')=='matched'
    assert z['research_highest_close400'].iloc[-1]['2330']==200.
    b.loc[b.stock_id.eq('2330'),'amount']/=10
    rules,_=calculate(b,d)
    assert state(rules,'legacy_course_breakout')=='not_matched'
    b,d=inputs(399,close=np.linspace(100.,200.,399),volume=np.full(399,4_000_000.))
    rules,_=calculate(b,d)
    assert state(rules,'legacy_course_breakout')=='unknown'


def test_first_bar_is_pure_price_and_cooldown_counts_suppressed_candidates():
    b,d=pulse_inputs();rules,z=calculate(b,d)
    assert [state(rules,'first_volume_bar_price',i) for i in (140,151,162,183)] == [
        'matched','not_matched','not_matched','matched']
    assert z['research_prior_first_candidate_count20'].iloc[162]['2330']==1.
    assert state(rules,'first_volume_bar_price',138)=='unknown'
    assert state(rules,'first_volume_bar_price',139)=='not_matched'


def test_prior_pulse_even_without_base_setup_blocks_next_first_bar():
    b,d=pulse_inputs()
    # A pulse 5 days before the candidate is known, even though that pulse's
    # first-bar/cooldown conditions need not pass.
    replace_stock(b,d,178,open=100.,high=103.5,low=99.5,close=103.,adjusted_close=103.,
                  volume=6_000_000.,amount=618_000_000.)
    rules,z=calculate(b,d)
    assert z['research_prior_burst_count10'].iloc[183]['2330']==1.
    assert state(rules,'first_volume_bar_price',183)=='not_matched'


def test_first_bar_gap_keeps_unknown_cooldown_instead_of_resetting():
    b,d=pulse_inputs();replace_stock(b,d,150,quality=False)
    rules,z=calculate(b,d)
    assert state(rules,'first_volume_bar_price',183)=='unknown'
    assert np.isnan(z['research_prior_first_candidate_count20'].iloc[183]['2330'])


def test_early_rotation_and_turnover_use_nonoverlapping_amount_windows():
    close=np.r_[np.full(410,100.),np.linspace(100.,120.,20)]
    volume=np.r_[np.full(425,1_000_000.),np.full(5,3_000_000.)]
    b,d=inputs(close=close,volume=volume);rules,z=calculate(b,d)
    expected=(close[-5:]*volume[-5:]).mean()/(close[-25:-5]*volume[-25:-5]).mean()
    assert z['research_turnover_heat'].iloc[-1]['2330']==pytest.approx(expected)
    assert state(rules,'early_rotation')=='matched'
    assert state(rules,'launch_turnover_heat')=='matched'
    assert state(rules,'launch_breakout_strength')=='matched'
    # Neither future winners nor an original-candidate ledger are inputs.
    assert z['research_relative20'].iloc[-1]['2330']==pytest.approx(.20)


def test_missing_benchmark_breaks_relative_signals_but_not_turnover_only():
    b,d=inputs(close=np.linspace(100.,120.,430),volume=np.r_[np.full(425,1e6),np.full(5,3e6)])
    b.loc[b.stock_id.eq('0050') & b.date.eq(d[-10]),'quality']=False
    rules,_=calculate(b,d)
    assert state(rules,'early_rotation')=='unknown'
    assert state(rules,'launch_breakout_strength')=='unknown'
    assert state(rules,'launch_turnover_heat')=='matched'
    rules,_=calculate(b[b.stock_id.ne('0050')],d)
    assert state(rules,'early_rotation')=='unknown'


def test_not_extended_is_filter_and_does_not_claim_deep_fall_is_buy_signal():
    b,d=inputs();replace_stock(b,d,-1,open=50.,high=51.,low=49.,close=50.,adjusted_close=50.,amount=50e6)
    rules,_=calculate(b,d)
    assert state(rules,'entry_not_extended')=='matched'
    item=next(x for x in RESEARCH_CATALOG if x['id']=='entry_not_extended')
    assert item['kind']=='filter' and '深跌' in ''.join(item['data_gaps'])


def test_strong_close_requires_strict_red_and_nonzero_range():
    b,d=inputs();replace_stock(b,d,-1,open=99.,high=100.,low=99.)
    rules,_=calculate(b,d)
    assert state(rules,'entry_strong_close')=='matched'
    replace_stock(b,d,-1,open=100.)
    rules,_=calculate(b,d)
    assert state(rules,'entry_strong_close')=='not_matched'
    replace_stock(b,d,-1,high=100.,low=100.)
    rules,_=calculate(b,d)
    assert state(rules,'entry_strong_close')=='unknown'


def test_liquidity_variants_cannot_be_satisfied_by_one_day_spike():
    volume=np.full(430,400_000.);volume[-1]=4_000_000.
    b,d=inputs(volume=volume);rules,z=calculate(b,d)
    assert z['amount20'].iloc[-1]['2330']>=50_000_000
    for sid in ('liquidity_median50m','liquidity_prior50m','liquidity_persistent50m'):
        assert state(rules,sid)=='not_matched'
    b,d=inputs(volume=np.full(430,500_000.));rules,_=calculate(b,d)
    assert all(state(rules,sid)=='matched' for sid in RESEARCH_IDS if sid.startswith('liquidity_'))


@pytest.mark.parametrize('invalid', ['missing', 'ineligible', 'source_conflict'])
def test_missing_market_session_breaks_windows_and_does_not_compress(invalid):
    b,d=inputs(close=np.linspace(100.,200.,430),volume=np.full(430,4e6))
    mask=b.stock_id.eq('2330') & b.date.eq(d[-5])
    if invalid=='missing': b=b.loc[~mask]
    elif invalid=='ineligible': b.loc[mask,'eligible']=False
    else: b['source_disagreement']=False; b.loc[mask,'source_disagreement']=True
    rules,_=calculate(b,d)
    assert all(state(rules,sid)=='unknown' for sid in RESEARCH_IDS)


def test_all_rules_require_minimum_average_amount_and_do_not_inherit_returns():
    b,d=pulse_inputs();b['amount']=1_000_000.
    rules,_=calculate(b,d)
    assert all(state(rules,sid,183)!='matched' for sid in RESEARCH_IDS)


def test_future_append_or_mutation_cannot_change_past_rules_or_features():
    b,d=pulse_inputs();cutoff=d[183]
    before,z_before=calculate(b[b.date<=cutoff],d[:184])
    b.loc[b.date>cutoff,['open','high','low','close','adjusted_close']]*=7
    b.loc[b.date>cutoff,['volume','amount']]*=50
    after,z_after=calculate(b,d)
    for key in RESEARCH_IDS:
        for part in ('match','known'):
            pd.testing.assert_frame_equal(before[key][part],after[key][part].loc[:cutoff])
    for key in z_before:
        if key.startswith('research_'):
            pd.testing.assert_frame_equal(z_before[key],z_after[key].loc[:cutoff])


def test_bad_matrix_alignment_and_duplicate_registration_fail_explicitly():
    b,d=inputs();f,_,_=_prepare(b,d,d[-1]);z=_features(f)
    f['a']=f['a'].iloc[1:]
    with pytest.raises(ValueError,match='identical ordered'):
        add_research_rules(f,z,lambda *a,**kw:None)
    f,_,_=_prepare(b,d,d[-1]);z=_features(f)
    add_research_rules(f,z,lambda *a,**kw:None)
    with pytest.raises(ValueError,match='Duplicate research feature'):
        add_research_rules(f,z,lambda *a,**kw:None)
