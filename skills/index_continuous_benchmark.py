"""Continuous 0050 ledger with source-bound rights, including the 2025 split."""
from collections import Counter
import pandas as pd
from skills.index_continuous_inputs import ContinuousCorporate, EarlierBenchmarkFeeds
from skills.board_only_verified_replay import BoardOnlyVerifiedBenchmark,audit_verified_board_only
from skills.execution_resources import audit_resources
from skills.backtest_contract import validate_completed_account
from scripts.research_exit_scenarios import summarize

def audit_benchmark(value,data):
    account=value['account'];days=data['days']
    validate_completed_account(account,days,days[0],days[-1])
    result=audit_resources(account,value['resource_plans'],opening_cash_only=True,lock_slots=False,lock_unused=True)
    result.update(audit_verified_board_only(account,value['board_decisions'],value['resource_plans']))
    sources={a['action_id']:a for a in data['actions']};seen=Counter();needed=Counter();previous_qty=0
    held={r['date']:r['qty'] for r in account['holdings']}
    for day in days:
        for a in data['actions']:
            if previous_qty and a['date']==day:needed[a['action_id']]+=1
        previous_qty=held.get(day,0)
    paid=Counter()
    for action in account['corporate_actions']:
        source=sources.get(action['action_id'])
        if not source:raise ValueError('Corporate action has no source')
        if action['kind']=='payment':
            expected=next((d for d in days if d>=source['pay_date']),None)
            if action['date']!=expected or action['ex_date']!=source['date']:raise ValueError('Payment date differs')
            paid[action['action_id']]+=1
        else:
            if any(action[k]!=v for k,v in source.items()):raise ValueError('Corporate terms differ')
            seen[action['action_id']]+=1
    if seen!=needed:raise ValueError('Missing or duplicated corporate entitlement')
    expected_paid=Counter({k:v for k,v in needed.items() if sources[k]['kind']=='cash_dividend' and sources[k]['pay_date']<=days[-1]})
    if paid!=expected_paid:raise ValueError('Missing or duplicated dividend payment')
    raw=data['benchmark_quotes'].set_index('date')
    for trade in account['trades']:
        if trade['stock_id']!='0050' or trade['side']!='buy' or trade['reference_price']!=float(raw.at[trade['date'],'close']):
            raise ValueError('Benchmark trade detached from buy-and-hold raw quote')
    result.update(corporate_source_binding=True,raw_quote_source_binding=True)
    return result

def run_benchmark(data,stress):
    engine=BoardOnlyVerifiedBenchmark(data['benchmark_quotes'],pd.DataFrame([dict(stock_id='0050',name='元大台灣50',market='TWSE')]),
        data['calendar'],[],EarlierBenchmarkFeeds(data['benchmark_limits']),ContinuousCorporate(data['actions']),
        start=data['days'][0],end=data['days'][-1],stress_mode='slip90' if stress else 'control')
    account=engine.run()
    value=dict(completed=True,config=dict(benchmark=True,board_only=True,factor_mask=int(stress)),account=account,
        summary=summarize(account),resource_plans=engine.resource_plans,board_decisions=engine.board_decisions,
        live_qualified=False,unseen_validation=False)
    value['audit']=audit_benchmark(value,data);return value
