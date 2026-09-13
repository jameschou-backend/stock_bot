from dataclasses import asdict
import copy
import hashlib
import json
import pytest
from skills.contingent_execution import Plan,OPEN,CUTOFF
from skills.contingent_replay import Tape,replay_day,cumulative_bill
from scripts.replay_contingent_day import load_tape,run
from skills.replay_market_feeds import ReplayDataUnavailable


def spec(odd=False):
    plans=[Plan('sell','2330','sell','board',1000,1000,0,'2026-09-11'),
           Plan('buy','2317','buy','board',1000,900,910000,'2026-09-11')]
    holdings={'2330':1000}
    if odd:
        plans.insert(1,Plan('odd','2330','sell','odd',1,1000,0,'2026-09-11'))
        holdings['2330']+=1
    return dict(date='2026-09-14',decision_date='2026-09-11',prior_data_date='2026-09-11',
        calendar=['2026-09-11','2026-09-14'],plans=[asdict(p) for p in plans],holdings=holdings,
        available_cents=20000,slots=1,credit_delay_us=0,slippage_bps=45,
        prior_avg_volume_shares={'2330':200000,'2317':200000})


def tape(sid,channel,rows):
    return Tape(sid,channel,'TWSE','2026-09-14',tuple(rows),'synthetic fixture','a'*64,True)


def tapes():
    return {('2330','board'):tape('2330','board',[(OPEN+10,1010,100000,True)]),
            ('2317','board'):tape('2317','board',[(OPEN+5,890,999000,True),
                                               (OPEN+10,890,100000,True),
                                               (OPEN+20,890,100000,True)])}


def test_integrated_sell_credit_buy_uses_only_post_submission_volume():
    r=replay_day(spec(),tapes())
    assert r['completed'] and r['synthetic'] and r['total_return'] is None
    assert [(f['side'],f['at']) for f in r['fills']]==[('sell',OPEN+10),('buy',OPEN+20)]
    assert r['fills'][1]['eligible_shares']==100000
    assert r['holdings']=={'2317':1000} and r['audit']['cash_conserved']


def test_credit_latency_cannot_use_earlier_buy_volume():
    s=spec();s['credit_delay_us']=11
    r=replay_day(s,tapes())
    assert [f['side'] for f in r['fills']]==['sell']
    assert r['unfilled'][0]['submitted'] and r['unfilled'][0]['qty']==1000


def test_no_reuse_keeps_sale_receivable_and_buy_unsubmitted():
    s=spec();s['credit_delay_us']=None
    r=replay_day(s,tapes())
    assert r['receivable_cents']>0 and not r['unfilled'][0]['submitted']


def test_missing_odd_tape_blocks_entire_day_before_any_fill():
    r=replay_day(spec(odd=True),tapes())
    assert not r['completed'] and not r['fills'] and not r['events']
    assert r['missing']==[dict(stock_id='2330',channel='odd')]


def test_trial_odd_does_not_clear_slot_and_negative_net_sale_can_clear_it():
    ts=tapes();s=spec(odd=True)
    ts[('2330','odd')]=tape('2330','odd',[(OPEN+11,1010,10000,False),(OPEN+15,1010,20,True)])
    r=replay_day(s,ts)
    assert [(f['channel'],f['at']) for f in r['fills']]==[('board',OPEN+10),('odd',OPEN+15),('board',OPEN+20)]
    assert r['fills'][1]['cash_cents']<0
    assert r['audit']['cash_conserved'] and r['holdings']=={'2317':1000}


def test_adv_and_strict_price_prevent_same_price_or_small_volume_fills():
    ts=tapes();s=spec();s['prior_avg_volume_shares']['2330']=99999
    assert not replay_day(s,ts)['fills']
    ts[('2330','board')]=tape('2330','board',[(OPEN+10,1000,10000000,True)])
    assert not replay_day(spec(),ts)['fills']


def test_cutoff_price_cannot_fill_and_partial_order_commission_charged_once():
    s=spec();s['plans']=[asdict(Plan('buy','2317','buy','odd',2,10000,23000,'2026-09-11'))]
    s['holdings']={};s['available_cents']=23000
    ts={('2317','odd'):tape('2317','odd',[(OPEN+1,9990,20,True),(OPEN+2,9990,20,True),(CUTOFF,9990,9999,True)])}
    r=replay_day(s,ts)
    assert len(r['fills'])==2 and sum(f['costs_cents']['commission'] for f in r['fills'])==2000
    assert sum(f['costs_cents']['total'] for f in r['fills'])==cumulative_bill(20000,'buy','2317',45)['total']


def test_unfunded_negative_sale_does_not_invent_cash():
    s=spec(odd=True);s['plans']=[s['plans'][1]];s['holdings']={'2330':1};s['available_cents']=0
    ts={('2330','odd'):tape('2330','odd',[(OPEN+1,1010,20,True)])}
    r=replay_day(s,ts)
    assert not r['fills'] and r['execution_blocks'][0]['reason']=='residual_sale_cost_unfunded'
    assert r['holdings']=={'2330':1}


@pytest.mark.parametrize('change',[{'prior_data_date':'2026-09-14'}, {'decision_date':'2026-09-14'}, {'credit_delay_us':-1}])
def test_future_prior_data_and_invalid_clock_rejected(change):
    s=spec();s.update(change)
    with pytest.raises(ValueError):replay_day(s,tapes())


def test_multiple_orders_cannot_reuse_same_tape_capacity():
    s=spec();s['plans'].append(dict(s['plans'][0],order_id='duplicate'))
    with pytest.raises(ValueError,match='volume reuse'):replay_day(s,tapes())


def test_illegal_limit_tick_is_not_treated_as_executable_order():
    s=spec();s['plans'][0]['limit_cents']=10311
    with pytest.raises(ValueError,match='tick size'):replay_day(s,tapes())


def test_equal_time_other_stock_order_does_not_depend_on_mapping_order():
    ts=tapes()
    assert replay_day(spec(),ts)==replay_day(spec(),dict(reversed(list(ts.items()))))


def test_invalid_tape_identity_and_quantity_unit_rejected():
    ts=tapes();ts[('2330','board')]=tape('2330','board',[(OPEN+1,1010,100,True)])
    with pytest.raises(ValueError,match='normalized'):replay_day(spec(),ts)
    ts=tapes();ts[('2330','board')]=tape('wrong','board',[(OPEN+1,1010,100000,True)])
    with pytest.raises(ValueError,match='identity'):replay_day(spec(),ts)


def source_file(tmp_path,**changes):
    data=dict(schema='normalized_auction_v1',timezone='Asia/Taipei',quantity_unit='shares',
        price_unit='TWD_cents',channel='odd',date='2026-09-14',stock_id='2330',market='TWSE',
        source_url='synthetic fixture',synthetic=True,session_complete=True,
        rows=[dict(time_us=OPEN+1,price_cents=1010,shares=20,record_type='trade')])
    data.update(changes);path=tmp_path/'odd.json';path.write_text(json.dumps(data))
    return dict(format='normalized_auction_v1',path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                stock_id='2330',date='2026-09-14',market='TWSE')


@pytest.mark.parametrize('changes',[{'quantity_unit':'lots'},{'timezone':'UTC'},{'price_unit':'unknown'}])
def test_ambiguous_auction_units_rejected(tmp_path,changes):
    item=source_file(tmp_path,**changes)
    with pytest.raises(ValueError):load_tape(item,tmp_path)


def test_incomplete_or_empty_auction_cannot_be_silent_no_fill(tmp_path):
    for change in ({'session_complete':False},{'rows':[]}):
        with pytest.raises(ReplayDataUnavailable):load_tape(source_file(tmp_path,**change),tmp_path)


def test_source_hash_detects_tampering(tmp_path):
    item=source_file(tmp_path);(tmp_path/item['path']).write_text('{}')
    with pytest.raises(ValueError,match='hash'):load_tape(item,tmp_path)


def test_cli_seals_missing_source_as_blocked_and_reproduces(tmp_path):
    plan=tmp_path/'plan.json';out=tmp_path/'out.json'
    plan.write_text(json.dumps(dict(spec=spec(),sources=[])))
    r=run(plan,out);assert not r['completed'] and r['total_return'] is None
    assert run(plan,out,True)==r


def test_cli_loads_both_markets_channels_and_replays_synthetic_sequence(tmp_path):
    ts=tapes();ts[('2330','odd')]=tape('2330','odd',[(OPEN+15,1010,20,True)])
    sources=[]
    for (sid,channel),t in ts.items():
        data=dict(schema='normalized_auction_v1',timezone='Asia/Taipei',quantity_unit='shares',
            price_unit='TWD_cents',channel=channel,date=t.day,stock_id=sid,market=t.market,
            source_url='synthetic fixture',synthetic=True,session_complete=True,
            rows=[dict(time_us=at,price_cents=p,shares=q,record_type='trade') for at,p,q,_ in t.rows])
        path=tmp_path/f'{sid}-{channel}.json';path.write_text(json.dumps(data))
        sources.append(dict(format='normalized_auction_v1',path=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),date=t.day,stock_id=sid,market=t.market))
    plan=tmp_path/'plan.json';out=tmp_path/'out.json'
    plan.write_text(json.dumps(dict(spec=spec(odd=True),sources=sources)))
    r=run(plan,out)
    assert r['completed'] and r['synthetic'] and len(r['fills'])==3
    assert r['holdings']=={'2317':1000} and r['live_qualified'] is False
    assert run(plan,out,True)==r
