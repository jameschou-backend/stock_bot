"""Checks for branch-flow diagnostics; no API or historical performance rerun."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skills.broker_branch_diagnostics import NAMED_BRANCHES, branch_snapshot
from scripts import research_broker_branch_diagnostics as research


def raw_frame(day='2026-08-03', stock='2330'):
    # B01's two prices must aggregate above B02. Largest absolute flow is a seller.
    quantities = [('B01', 60, 5), ('B01', 40, 5), ('B02', 90, 10),
                  ('B03', 80, 10), ('B04', 70, 10), ('B05', 60, 10),
                  ('B06', 50, 10), ('SELL', 10, 400)]
    return pd.DataFrame([dict(date=day, stock_id=stock, securities_trader_id=b,
        securities_trader=b, price=20+i, buy=buy, sell=sell)
        for i, (b, buy, sell) in enumerate(quantities)])


def test_price_levels_aggregate_and_top5_does_not_use_absolute_net():
    raw = raw_frame()
    result = branch_snapshot(raw, '2330', '2026-08-03')
    assert result['known']
    assert result['raw_rows'] == 8 and result['branch_count'] == 7
    assert [b['securities_trader_id'] for b in result['top5']] == ['B01','B02','B03','B04','B05']
    assert result['top5'][0]['buy'] == 100 and result['top5'][0]['sell'] == 10
    assert result['top5'][0]['net'] == 90
    assert result['market_net_shares'] == 0
    assert result['total_buy_shares'] == 460
    assert result['top5_net_share'] == pytest.approx(350/460)
    assert result['top5_directional_ratio'] == pytest.approx(350/450)
    assert all(b['net'] > 0 for b in result['top5'])
    # Preserve every execution row; changing source row order cannot change a score.
    shuffled = raw.sample(frac=1, random_state=11)
    assert branch_snapshot(shuffled, '2330', '2026-08-03') == result


def test_tied_buyers_have_deterministic_broker_code_order():
    raw = raw_frame()
    raw.loc[raw.securities_trader_id.eq('B02'), 'buy'] -= 10
    raw.loc[raw.securities_trader_id.eq('B03'), 'sell'] += 10
    raw.loc[raw.securities_trader_id.eq('SELL'), 'sell'] -= 20
    # B02 net70, B03 net60 -> set both to60 preserving balance.
    raw.loc[raw.securities_trader_id.eq('B02'), 'buy'] -= 10
    raw.loc[raw.securities_trader_id.eq('SELL'), 'sell'] -= 10
    result = branch_snapshot(raw, '2330', '2026-08-03')
    assert result['known']
    codes = [x['securities_trader_id'] for x in result['top5']]
    assert codes.index('B02') < codes.index('B03')


@pytest.mark.parametrize('value', [-1, .5, np.nan, np.inf, -np.inf, 2**53, 'bad'])
def test_invalid_quantities_cannot_be_unknown_or_zero(value):
    raw = raw_frame()
    raw['buy'] = raw.buy.astype(object)
    raw.loc[0, 'buy'] = value
    with pytest.raises((ValueError, TypeError)):
        branch_snapshot(raw, '2330', '2026-08-03')


def test_sum_rejected_before_int64_overflow():
    n = 1025
    raw = pd.DataFrame(dict(date=['2026-08-03']*n, stock_id=['2330']*n,
        securities_trader_id=['ONE']*n, buy=[2**53-1]*n, sell=[2**53-1]*n))
    with pytest.raises(ValueError, match='exact quantity range'):
        branch_snapshot(raw, '2330', '2026-08-03')


@pytest.mark.parametrize('column,value', [('stock_id','2317'),('date','2026-08-04'),
                                         ('securities_trader_id',None),('securities_trader_id',' ')])
def test_wrong_identity_or_future_row_rejected(column, value):
    raw = raw_frame()
    raw.loc[0, column] = value
    with pytest.raises(ValueError, match='identity'):
        branch_snapshot(raw, '2330', '2026-08-03')


def test_unbalanced_empty_and_insufficient_buyers_have_explicit_unknown():
    assert branch_snapshot(pd.DataFrame(), '2330', '2026-08-03') == dict(known=False, reason='missing_raw_source')
    raw = raw_frame()
    raw.loc[0,'buy'] += 20
    assert branch_snapshot(raw, '2330', '2026-08-03') == dict(known=False, reason='raw_market_imbalance')
    raw = raw_frame()
    raw['securities_trader_id'] = raw.securities_trader_id.replace({'B05':'B04','B06':'B04'})
    assert branch_snapshot(raw, '2330', '2026-08-03')['reason'] == 'fewer_than_five_positive_branches'


def test_unobserved_named_branch_quantities_are_not_zero_or_holdings():
    result = branch_snapshot(raw_frame(), '2330', '2026-08-03')
    assert not result['named_positive']
    for branch in result['named'].values():
        assert branch['observed'] is False
        assert branch['buy'] is branch['sell'] is branch['net'] is branch['positive_rank'] is None
    raw = raw_frame().replace({'securities_trader_id': {'B01':'9A9g','SELL':'9853'}})
    result = branch_snapshot(raw, '2330', '2026-08-03')
    assert result['named']['9A9g']['observed'] and result['named']['9A9g']['positive_rank'] == 1
    assert result['named']['9A9g']['net'] == 90
    assert result['named']['9853']['observed'] and result['named']['9853']['net'] == -390
    assert result['named']['9853']['positive_rank'] is None
    assert 'holding_shares' not in result['named']['9A9g']


def diagnostic_row(index, *, branch_known=True, persist_known=True, passed=True,
                   concentrated=True, status='closed', net=.1, year='2026'):
    b = branch_snapshot(raw_frame(), '2330', '2026-08-03') if branch_known else {'known':False,'reason':'missing_raw_source'}
    if branch_known: b['concentrated_directional'] = concentrated
    return dict(event_id=str(index), stock_id='2330', signal_date=year+'-08-03', branch=b,
        persistence5=dict(known=True,passed=passed) if persist_known else dict(known=False,reason='missing_branch_market_day'),
        outcome=dict(status=status, net_return=net if status=='closed' else None,
                     reason='loss12' if net<0 else 'time63',peak_close_return=.3))


def test_comparisons_match_required_coverage_and_keep_unknown_outcomes_separate():
    rows = [diagnostic_row(0,net=.2), diagnostic_row(1,passed=False,net=-.1),
            diagnostic_row(2,persist_known=False,net=.9), diagnostic_row(3,branch_known=False),
            diagnostic_row(4,status='open'),diagnostic_row(5,status='unknown'),
            diagnostic_row(6,concentrated=False,net=.05,year='2025')]
    result = research.comparisons(rows)
    persistence = result['persistent5']
    assert persistence['known_signals'] == 5 and persistence['unknown_signals'] == 2
    assert persistence['passed']['signals'] == 4 and persistence['other']['signals'] == 1
    assert persistence['passed']['closed'] == 2
    assert persistence['passed']['statuses'] == {'closed':2,'open':1,'unknown':1}
    assert persistence['passed']['mean_net_return'] == pytest.approx(.125)
    assert persistence['other']['mean_net_return'] == pytest.approx(-.1)
    assert result['combined']['known_signals'] == 5
    assert result['combined']['passed']['signals'] == 3
    assert result['concentrated_directional']['known_signals'] == 6
    assert result['named_positive']['other_definition'] == 'no_positive_named_flow_observed_not_verified_zero_trading'
    # Missing named rows count as no observed event, never asserted zero trading.
    assert result['named_positive']['passed']['signals'] == 0
    assert result['named_positive']['other']['signals'] == 6
    assert persistence['annual']['2025']['passed']['signals'] == 1


def test_group_membership_does_not_use_future_outcome_values():
    rows = [diagnostic_row(0),diagnostic_row(1,passed=False),diagnostic_row(2,persist_known=False)]
    before = research.comparisons(rows)
    changed = deepcopy(rows)
    for r in changed: r['outcome'] = dict(status='open',net_return=None,reason=None,peak_close_return=999.)
    after = research.comparisons(changed)
    for group in before:
        assert after[group]['known_signals'] == before[group]['known_signals']
        assert after[group]['unknown_signals'] == before[group]['unknown_signals']
        for arm in ('passed','other'):
            assert after[group][arm]['signals'] == before[group][arm]['signals']


@pytest.fixture
def offline_run_fixture(tmp_path, monkeypatch):
    """Exercise real run ordering/hash guards on 458 synthetic, source-bound events."""
    root = tmp_path/'repo';root.mkdir()
    constants = {'ROOT':root,'SPEC':root/'docs/spec.md','INPUT':root/'inputs',
                 'OLD':root/'old','SIGNALS':root/'signals.json','POC':root/'explorer/payload.json',
                 '__file__':str(root/'scripts/runner.py')}
    for key,value in constants.items():monkeypatch.setattr(research,key,value)
    def put(path, content):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(content) if not isinstance(content,str) else content)
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [constants['SPEC'],Path(constants['__file__']),*[root/'skills'/n for n in
        ('broker_branch_diagnostics.py','independent_signals.py','exit_policy.py','million_replay.py','trial_registry.py')],
        root/'tests/test_broker_branch_diagnostics.py',root/'scripts/research_exit_scenarios.py']:
        put(path,'source fixture')
    calendar=pd.bdate_range(end='2026-09-09',periods=230)
    ids=['2317','2330']
    events=[dict(event_id=f'{sid}-{day:%Y-%m-%d}',members=[sid],signal_date=f'{day:%Y-%m-%d}')
            for day in calendar[:-1] for sid in ids]
    put(constants['SIGNALS'],{'entries':list(reversed(events))})
    persistence={e['event_id']:{'5':dict(known=True,passed=True)} for e in events}
    put(constants['OLD']/'signals.json',persistence)
    put(constants['OLD']/'manifest.json',{'files_sha256':{str(p.relative_to(root)):digest(p)
        for p in (constants['SIGNALS'],constants['OLD']/'signals.json')}})
    frames={}
    for name in ('close-official.parquet','close-quality.parquet','eligibility.parquet'):
        frames[constants['INPUT']/name]=pd.DataFrame({'date':calendar,**{sid:np.full(len(calendar),True if name=='eligibility.parquet' else 100.) for sid in ids}})
    frames[constants['INPUT']/'quotes-unmasked.parquet']=pd.DataFrame([
        dict(date=d,stock_id=sid,close=100.,high=101.,low=99.,volume=1000.) for d in calendar for sid in ids])
    frames[constants['INPUT']/'companies.parquet']=pd.DataFrame(dict(stock_id=ids,name=ids))
    for path in frames:put(path,str(path.name))
    put(constants['INPUT']/'manifest.json',{'files_sha256':{p.name:digest(p) for p in frames}})
    rawrefs={}
    for e in events:
        sid,day=e['members'][0],e['signal_date']
        path=root/f'.cache/chip-inputs/raw/broker/{sid}_{day}_{day}.parquet'
        frames[path]=raw_frame(day,sid);put(path,e['event_id'])
        rawrefs[str(path.relative_to(root))]=digest(path)
    put(root/'.cache/chip-inputs/manifest.json',{'files_sha256':rawrefs})
    put(constants['POC'],{'signals':[dict(stock_id=e['members'][0],signal_date=e['signal_date']) for e in events]})
    put(constants['POC'].with_name('receipt.json'),{'output_sha256':{str(constants['POC'].relative_to(root)):digest(constants['POC'])}})
    monkeypatch.setattr(research.pd,'read_parquet',lambda path,**kwargs:frames[Path(path)].copy())
    calls=[]
    def observe(path,index):
        calls.append(index)
        assert 1 <= index < len(path.days)
        return dict(status='closed',reason='time63',net_return=(1 if index==1 else -1)*.1,peak_close_return=.3)
    monkeypatch.setattr(research,'observe',observe)
    return root,events,calls,constants


def test_actual_runner_first_per_stock_and_tplus1_are_outcome_independent(offline_run_fixture,monkeypatch):
    root,events,calls,constants=offline_run_fixture
    research.run(root/'first')
    rows=json.loads((root/'first/rows.json').read_text())
    first=[r for r in rows if r['first_for_stock']]
    assert len(first)==2 and {r['signal_date'] for r in first} == {events[0]['signal_date']}
    assert [r['stock_id'] for r in first]==['2317','2330']
    assert calls[:4]==[1,1,2,2] and len(calls)==458
    assert all(r['entry_date']>r['signal_date'] for r in rows)
    # Reversing all future results cannot choose a different first signal or branch group.
    monkeypatch.setattr(research,'observe',lambda path,index:dict(status='closed',reason='loss12',net_return=-999.,peak_close_return=0.))
    research.run(root/'changed_future_outcomes')
    changed=json.loads((root/'changed_future_outcomes/rows.json').read_text())
    assert [r['event_id'] for r in changed if r['first_for_stock']]==[r['event_id'] for r in first]
    assert [r['branch'] for r in changed]==[r['branch'] for r in rows]
    assert [r['persistence5'] for r in changed]==[r['persistence5'] for r in rows]


def test_actual_runner_stops_on_hash_changed_raw_before_observing(offline_run_fixture):
    root,events,calls,_=offline_run_fixture
    e=events[0];sid=e['members'][0];day=e['signal_date']
    path=root/f'.cache/chip-inputs/raw/broker/{sid}_{day}_{day}.parquet'
    path.write_text('tampered after manifest')
    with pytest.raises(ValueError,match='Frozen input changed'):
        research.run(root/'rejected')
    assert calls==[]
    assert not (root/'rejected').exists()
