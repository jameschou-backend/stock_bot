from copy import deepcopy

import pandas as pd
import pytest

from skills.verified_volume_midpoint_audit import audit_verified_volume_midpoint, ordinary_inputs
from test_cash_allocation_replay import fixture, ENTRY
from test_verified_volume_midpoint import run, SafeExitReplay, SafeReplay


def checked(**kwargs):
    engine, account, days = run(**kwargs)
    _, _, args, _ = fixture()
    def audit(value=account):
        return audit_verified_volume_midpoint(value, engine.ticks, engine.odd_feeds,
            engine.markets, args[0], days, engine.corporate, engine.feeds,
            engine.ordinary_volumes, lambda d, s: 'TWSE')
    return engine, account, audit


@pytest.mark.parametrize('missing', [None, ENTRY, ENTRY-20])
def test_full_account_reconciles_actual_session_caps_and_missing_evidence(missing):
    _, _, audit = checked(missing=missing)
    result = audit()
    assert result['daily_capacity_rebuilt']
    assert result['all_attempted_ordinary_capacity_verified'] is (missing is None)


@pytest.mark.parametrize('mutation', ['volume', 'average', 'capacity', 'cash', 'scope', 'coverage', 'trade'])
def test_independent_audit_rejects_changed_capacity_inputs_and_claims(mutation):
    _, account, audit = checked()
    bad = deepcopy(account)
    row = next(r for r in bad['orders'] if r['channel'] == 'board')
    if mutation == 'volume': row['source_volume'] = row['source_total_volume']
    if mutation == 'average': row['capacity_prior_ordinary_volume20'] = row['prior_avg_volume20']
    if mutation == 'capacity': row['capacity_qty'] += 1000
    if mutation == 'cash': bad['daily'][-1]['cash'] += 1000
    if mutation == 'scope': row['volume_scope'] = 'all_daily_sessions'
    if mutation == 'coverage': bad['ordinary_volume_evidence']['blocked_board_children'] = 1
    if mutation == 'trade': bad['trades'][0]['source_volume'] += 1
    with pytest.raises(ValueError): audit(bad)


def test_audit_cannot_promote_skipped_data_to_complete_execution():
    _, account, audit = checked(missing=ENTRY)
    bad = deepcopy(account)
    bad['ordinary_volume_evidence']['all_requested_board_capacity_observed'] = True
    with pytest.raises(ValueError, match='coverage'): audit(bad)


def test_scalar_audit_uses_each_historical_venue_and_excludes_future_volume():
    days = pd.bdate_range('2025-01-01', periods=22)
    matrices = {'TWSE': pd.DataFrame(200_000., index=days, columns=['5314']),
                'TPEX': pd.DataFrame(100_000., index=days, columns=['5314'])}
    resolver = lambda d, s: 'TPEX' if d < days[10] else 'TWSE'
    assert ordinary_inputs(matrices, resolver, days, days[20], '5314') == (200_000., 150_000., True)
    matrices['TWSE'].loc[days[21], '5314'] = 100e6
    assert ordinary_inputs(matrices, resolver, days, days[20], '5314') == (200_000., 150_000., True)


@pytest.mark.parametrize('field,value', [
    ('stock_id','9999'), ('source_high',777.), ('source_low',1.), ('limit_price',999.),
    ('signal_date','2020-01-01'), ('requested_qty',1_000_000), ('volume_policy','legacy_total_research'),
    ('prior_avg_volume20',123.), ('capacity_qty',999_000), ('sequence',99),
])
def test_trade_must_copy_all_source_order_identity_and_evidence(field, value):
    _, account, audit = checked()
    bad = deepcopy(account)
    row = next(t for t in bad['trades'] if t['channel']=='board')
    row[field] = value
    with pytest.raises(ValueError): audit(bad)


@pytest.mark.parametrize('mutation', ['verified_count','block_missing','wrong_gap','wrong_stockdays','policy','qualification'])
def test_coverage_manifest_matches_independently_rebuilt_gaps(mutation):
    _, account, audit = checked(missing=ENTRY)
    bad = deepcopy(account)
    evidence = bad['ordinary_volume_evidence']
    if mutation == 'verified_count': evidence['verified_board_children'] += 1
    elif mutation == 'block_missing': evidence['blocks'] = []
    elif mutation == 'wrong_gap':
        evidence['blocks'] = deepcopy(evidence['blocks'])
        evidence['blocks'][0]['missing'][0]['date'] = '2020-01-01'
    elif mutation == 'wrong_stockdays': evidence['missing_stock_days'] = []
    elif mutation == 'policy': evidence['policy'] = 'legacy_total_research'
    else: evidence['live_qualified'] = True
    with pytest.raises(ValueError, match='coverage'): audit(bad)


def test_changing_order_and_summary_gap_together_cannot_hide_actual_missing_day():
    _, account, audit = checked(missing=ENTRY)
    bad = deepcopy(account)
    row = next(r for r in bad['orders'] if r.get('failure')=='ordinary_volume_evidence_missing')
    row['ordinary_volume_gaps'][0]['date'] = '2020-01-01'
    with pytest.raises(ValueError, match='Missing ordinary'): audit(bad)


@pytest.mark.parametrize('mutation', ['fake_liquidation','empty_positions','wrong_quantity','valuation','settings'])
def test_ending_inventory_must_match_final_mark_to_market_holdings(mutation):
    _, account, audit = checked(end=ENTRY+1)
    bad = deepcopy(account)
    ending = bad['ending_inventory']
    if mutation == 'fake_liquidation': ending['automatically_liquidated'] = True
    elif mutation == 'empty_positions': ending['positions'] = []; ending['remaining_positions'] = 0
    elif mutation == 'wrong_quantity': ending['positions'][0]['qty'] += 1000
    elif mutation == 'valuation': ending['valuation'] = 'realized_cash'
    else: bad['settings']['forced_end_liquidation'] = True
    with pytest.raises(ValueError, match='Ending inventory'): audit(bad)


def test_unknown_dated_identity_cannot_supply_ordinary_history():
    days = pd.bdate_range('2025-01-01',periods=21)
    matrices = {'TWSE':pd.DataFrame(1_000_000.,index=days,columns=['5314'])}
    resolver = lambda d,s: dict(status='unidentified',market='TWSE')
    assert ordinary_inputs(matrices,resolver,days,days[-1],'5314') == (None,None,False)


@pytest.mark.parametrize('mode', ['all_blocked', 'one_entire_attempt', 'zero_requested', 'fake_halt'])
def test_nonzero_frozen_plans_prevent_deleting_failed_orders_and_promoting_coverage(mode):
    _, baseline, days = run(replay_class=SafeExitReplay)
    exit_day = next(r['date'] for r in baseline['trades'] if r['side']=='sell' and r['channel']=='board')
    _, account, audit = checked(missing=days.get_loc(pd.Timestamp(exit_day)), replay_class=SafeExitReplay)
    audit()  # The visible, unfilled sell attempts are a valid incomplete result.
    bad = deepcopy(account)
    blocked = [r for r in bad['orders'] if r.get('failure')=='ordinary_volume_evidence_missing']
    assert len(blocked) == 3
    if mode == 'all_blocked':
        bad['orders'] = [r for r in bad['orders'] if r not in blocked]
    elif mode == 'one_entire_attempt':
        # This later date has only a board child, so no sibling odd order can
        # expose the omission. The precommitted plan must still require it.
        target = blocked[-1]
        bad['orders'] = [r for r in bad['orders'] if r['date'] != target['date']]
    elif mode == 'zero_requested':
        for row in blocked: row['requested_qty'] = 0
    else:
        for row in blocked:
            row.update(channel='event',failure='official_full_session_halt')
    boards = [r for r in bad['orders'] if r['channel']=='board' and r['requested_qty']]
    bad['ordinary_volume_evidence'].update(blocks=[],missing_stock_days=[],blocked_board_children=0,
        requested_board_children=len(boards),verified_board_children=len(boards),all_requested_board_capacity_observed=True)
    with pytest.raises(ValueError, match='[Nn]onzero frozen child'):
        audit(bad)


def test_zero_sized_frozen_plan_does_not_require_a_fabricated_order():
    class BelowOneShareReplay(SafeReplay):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,initial_cash=10.,**kwargs)
    _,account,audit=checked(replay_class=BelowOneShareReplay,end=ENTRY+1)
    assert account['tick_plans'] and all(p['planned_qty']==0 for p in account['tick_plans'])
    assert not account['orders'] and not account['trades']
    assert audit()['nonzero_planned_children_reconciled']


def halted_case():
    class HaltedReplay(SafeReplay):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            day, resume = self.days[ENTRY], self.days[ENTRY+1]
            self.full_halts.append(dict(stock_id='1101',market='TWSE',kind='trading_suspension',
                start=str(day.date()),end=str(resume.date())))
            for key in ('open','high','low','close','volume'):
                self.fields[key].at[day,'1101'] = 0.
            self.ordinary_volumes['TWSE'].at[day,'1101'] = 0.
    engine,account,days = run(replay_class=HaltedReplay,end=ENTRY+1)
    _,_,args,_ = fixture()
    quotes=args[0].copy()
    quotes.loc[quotes.date.eq(days[ENTRY]) & quotes.stock_id.eq('1101'),['open','high','low','close','volume']]=0.
    def audit(value=account, halts=None, source=quotes):
        return audit_verified_volume_midpoint(value,engine.ticks,engine.odd_feeds,engine.markets,
            source,days,engine.corporate,engine.feeds,engine.ordinary_volumes,lambda d,s:'TWSE',
            verified_halts=engine.full_halts if halts is None else halts)
    return engine,account,audit,quotes


def test_proven_full_session_halt_event_replaces_nonzero_children_without_inventing_fills():
    _,account,audit,_ = halted_case()
    assert account['tick_plans'][0]['planned_qty'] > 0
    assert account['orders'][0]['failure'] == 'official_full_session_halt'
    result=audit()
    assert result['full_session_halt_plans']==1
    assert result['nonzero_planned_children_reconciled']
    assert result['all_attempted_ordinary_capacity_verified'] is False


@pytest.mark.parametrize('mutation',['missing_notice','wrong_date','intraday_only','positive_quote','positive_ordinary','removed_event'])
def test_halt_exception_requires_correct_full_session_source_and_event(mutation):
    engine,account,audit,quotes=halted_case()
    bad,halts=deepcopy(account),deepcopy(engine.full_halts)
    if mutation=='missing_notice':halts=[]
    elif mutation=='wrong_date':halts[0]['end']=halts[0]['start']
    elif mutation=='intraday_only':halts[0].update(kind='information_halt',source_row=['','','','','09:00','','10:00'])
    elif mutation=='positive_quote':quotes.loc[quotes.date.eq(pd.Timestamp(halts[0]['start'])) & quotes.stock_id.eq('1101'),'volume']=1000.
    elif mutation=='positive_ordinary':engine.ordinary_volumes['TWSE'].at[pd.Timestamp(halts[0]['start']),'1101']=1000.
    else:bad['orders']=[]
    with pytest.raises(ValueError,match='[Nn]onzero frozen child|[Hh]alt conflicts'):
        audit(bad,halts=halts,source=quotes)
