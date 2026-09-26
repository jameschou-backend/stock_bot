"""Use the sealed benchmark's fee, dividend and funding ledger on earlier inputs."""
import pandas as pd
from skills.board_only_verified_replay import BoardOnlyVerifiedBenchmark,audit_verified_board_only
from skills.execution_resources import audit_resources
from skills.index_earlier_inputs import EarlierBenchmarkCorporate,EarlierBenchmarkFeeds
from skills.backtest_contract import validate_completed_account
from scripts.research_exit_scenarios import summarize

def audit_benchmark(value,data):
    account=value['account']
    validate_completed_account(account,data['days'],data['days'][0],data['days'][-1])
    result=audit_resources(account,value['resource_plans'],opening_cash_only=True,lock_slots=False,lock_unused=True)
    result.update(audit_verified_board_only(account,value['board_decisions'],value['resource_plans']))
    expected={x['date']:x for x in data['dividends']}
    entitlements={}
    for action in account['corporate_actions']:
        if action['kind']=='cash_dividend':
            item=expected.get(action['date'])
            if not item or any(action[k]!=item[k] for k in ('stock_id','action_id','cash_per_share','pay_date')):
                raise ValueError('Benchmark distribution detached from source')
            entitlements[action['date']]=action
        elif action['kind']=='payment':
            item=expected.get(action['ex_date'])
            payment_day=next((d for d in data['days'] if item and d>=item['pay_date']),None)
            if not item or action['date']!=payment_day or action['action_id']!=item['action_id']:
                raise ValueError('Benchmark payment detached from source')
        else:raise ValueError('Unexpected earlier benchmark action')
    previous_qty=0;needed=set();held={r['date']:r['qty'] for r in account['holdings']}
    for day in data['days']:
        if previous_qty and day in expected:needed.add(day)
        previous_qty=held.get(day,0)
    if set(entitlements)!=needed:raise ValueError('Missing or extra dividend entitlements')
    quotes=data['benchmark_quotes'].set_index('date')
    for trade in account['trades']:
        if trade['stock_id']!='0050' or trade['side']!='buy':raise ValueError('Benchmark is buy and hold')
        if float(quotes.at[trade['date'],'close'])!=trade['reference_price']:raise ValueError('Benchmark raw fill price differs')
    result.update(dividend_source_binding=True,raw_quote_source_binding=True,announcement_used_for_signals=False)
    return result

def run_benchmark(data,double_slippage):
    engine=BoardOnlyVerifiedBenchmark(data['benchmark_quotes'],
        pd.DataFrame([dict(stock_id='0050',name='元大台灣50',market='TWSE')]),data['calendar'],[],
        EarlierBenchmarkFeeds(data['benchmark_limits']),EarlierBenchmarkCorporate(data['dividends']),
        start=data['days'][0],end=data['days'][-1],stress_mode='slip90' if double_slippage else 'control')
    account=engine.run()
    value=dict(completed=True,account=account,summary=summarize(account),
        config=dict(benchmark=True,board_only=True,factor_mask=int(double_slippage)),
        resource_plans=engine.resource_plans,board_decisions=engine.board_decisions,
        live_qualified=False,unseen_validation=False)
    value['audit']=audit_benchmark(value,data)
    return value
