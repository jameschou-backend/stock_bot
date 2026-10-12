#!/usr/bin/env python3
"""Publish every fixed legacy replay with verified sources and benchmark parity."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_poc_broker_account import Sources, account_metrics
from scripts.summarize_all_entry_accounts import guard_certificate_inventory
from scripts.summarize_top3_validation import windows, prior_publication, select_attempt
from skills.all_entry_accounts import digest, write
from skills.legacy_return_comparison import ARMS, NAMES, BASE, SPEC
from skills.top3_validation import MODES, verify_costs

JOURNALS=('daily','trades','orders','cash_ledger','holdings','cohorts','corporate_actions','receivables')


def check_prices(account):
    errors=[]
    for trade in account['trades']:
        fraction=.5 if trade['side']=='buy' else .3
        errors.append(abs(trade['reference_price']-(trade['source_low']+fraction*(trade['source_high']-trade['source_low']))))
        if trade['side']=='buy' and not trade['signal_date']<trade['date']:
            raise ValueError('Buy lacks a prior-session signal')
    if max(errors,default=0)>1e-8: raise ValueError('Trade price differs from registered range')
    return dict(trades_checked=len(errors),maximum_error=max(errors,default=0),buy_hl2=True,sell30=True)


def compare_paths(old,new):
    """Descriptive funded-event changes; not a single-factor causal attribution."""
    def keys(account):
        return {(c['stock_id'],c['signal_date']) for c in account['cohorts'] if c['entry_date']>='2024-01-02'}
    def focus(account):
        rows=[t for t in account['trades'] if t['stock_id']=='5386' and t['date']>='2026-01-01']
        rejected=[r for r in account['orders'] if r['stock_id']=='5386' and r['date']>='2026-01-01' and r.get('filled_qty')==0]
        return dict(trades=[{k:t.get(k) for k in ('date','signal_date','side','qty','reference_price','reason','channel','total_cost')} for t in rows],
            trade_cash_change=sum(t['cash_change'] for t in rows),
            rejected_orders=[{k:t.get(k) for k in ('date','signal_date','failure','reason')} for t in rejected],
            note='僅此股2026年買賣現金淨流量，不含權息、其他現金項目或未平倉市值，並非完整損益。')
    a,b=keys(old),keys(new)
    old_open=next(r for r in old['daily'] if r['date']=='2024-01-02')
    earlier=[r['date'] for r in old['daily'] if r['date']<'2024-01-02']
    carry=[{k:r[k] for k in ('stock_id','qty','market_value')} for r in old['holdings']
           if earlier and r['date']==max(earlier)]
    missed=[]
    for day in sorted({t['date'] for t in old['trades'] if t['stock_id']=='5386' and t['side']=='buy' and t['date']>='2026-01-01'}):
        if any(t['stock_id']=='5386' and t['side']=='buy' and t['date']==day for t in new['trades']):continue
        prior=max(r['date'] for r in new['daily'] if r['date']<day)
        missed.append(dict(date=day,new_orders=[{k:o.get(k) for k in ('stock_id','signal_date','failure','filled_qty')}
            for o in new['orders'] if o['stock_id']=='5386' and o['side']=='buy' and o['date']==day],
            opening_holdings=[{k:h[k] for k in ('stock_id','qty','market_value','event_id')} for h in new['holdings'] if h['date']==prior],
            opening_cash=next(r['cash'] for r in new['daily'] if r['date']==prior)))
    before,after=stock_contributions(old),stock_contributions(new)
    contributions=[dict(stock_id=s,old_profit=before.get(s,0),new_profit=after.get(s,0),
        difference=after.get(s,0)-before.get(s,0)) for s in set(before)|set(after)]
    contributions.sort(key=lambda r:abs(r['difference']),reverse=True)
    return dict(old_funded_events=len(a),new_funded_events=len(b),common_funded_events=len(a&b),
        old_opening_nav_2024=old_open['opening_nav'],old_carry_positions=carry,
        new_opening_nav_2024=new['daily'][0]['opening_nav'],
        only_old=[dict(stock_id=s,signal_date=d) for s,d in sorted(a-b)],
        only_new=[dict(stock_id=s,signal_date=d) for s,d in sorted(b-a)],
        focus_5386=dict(old=focus(old),new=focus(new)),missed_5386_legacy_entries=missed,
        stock_profit_differences=contributions,
        stock_profit_note='各自完整期間：所有買賣、權息現金加期末持股及應收款；換股歸回原進場股票。加總核對帳戶總損益。跨版本差額不是單一條件因果。')


def stock_contributions(account):
    """Reconcile cash plus closing assets to whole-account profit by entry stock."""
    from collections import defaultdict
    by_event={c['event_id']:c['stock_id'] for c in account['cohorts']}
    totals=defaultdict(float)
    def sid(row):return by_event.get(row.get('event_id'),row.get('stock_id'))
    for row in account['cash_ledger']:
        if row['kind']=='initial_deposit':continue
        if not sid(row):raise ValueError('Profit attribution lacks investment identity')
        totals[sid(row)]+=row['cash_change']
    final=account['daily'][-1]
    for row in account['holdings']:
        if row['date']==final['date']:totals[sid(row)]+=row['market_value']
    for row in account['receivables']:
        if row['kind']!='cash':raise ValueError('Share receivable attribution needs independent terminal valuation')
        totals[sid(row)]+=row['amount']
    if abs(sum(totals.values())-(final['nav']-account['settings']['initial_cash']))>.011:
        raise ValueError('Stock attribution does not sum to account profit')
    return dict(totals)


def summarize(paths,output,replacements=()):
    sources=Sources(ROOT);cases={};accounts={};reports=[];trials=[]
    for supplied,repair in [(p,False) for p in paths]+[(p,True) for p in replacements]:
        path=Path(supplied).resolve();relative=str(path.relative_to(ROOT))
        expected=path.with_suffix('.sha256').read_text().strip()
        report=json.loads(sources.bind(relative,expected).read_text())
        sources.closure(report,relative)
        model=report['legacy_comparison'];mode=model['mode']
        if (mode not in MODES or (report['start'],report['end'],report['initial_cash'])!=('2024-01-02','2026-10-02',1_000_000)
            or (model['commission'],model['slippage'])!=MODES[mode]
            or (model['buy_fraction'],model['sell_fraction'])!=(.5,.3)):
            raise ValueError('Legacy comparison scope differs')
        trial_path=path.parent/'trials.jsonl'
        trial_hash=digest(trial_path)
        rows=[json.loads(line) for line in sources.bind(str(trial_path.relative_to(ROOT)),trial_hash).read_text().splitlines() if line]
        if (len(rows)!=len(report['cases']) or {r['params']['arm'] for r in rows}!=set(report['cases'])
            or any(r.get('legacy_mode')!=mode for r in rows)):
            raise ValueError('All attempted cases must retain their trial record')
        trials.extend(rows)
        reports.append(dict(path=relative,sha256=expected,mode=mode,replacement=repair,
            trial_path=str(trial_path.relative_to(ROOT)),trial_sha256=trial_hash))
        for arm,descriptor in report['cases'].items():
            if arm not in ARMS: raise ValueError('Unexpected legacy strategy')
            case=json.loads(sources.bind(descriptor['path'],descriptor['sha256']).read_text())
            if case['legacy_comparison']!=model: raise ValueError('Case configuration differs')
            row=dict(id=arm,name=NAMES[arm],mode=mode,completed=case['completed'],
                path=descriptor['path'],sha256=descriptor['sha256'],summary=case.get('summary'),
                reason=case.get('reason'),metrics=None,data_gap_exclusions=len(case.get('data_gap_exclusions',[])),
                selection_fallback_days=case.get('selection_fallback_days',0),
                family_rules=case['family_rules'],settlement_evidence_notes=case.get('settlement_evidence_notes',[]))
            if case['completed']:
                account=case['account'];guard_certificate_inventory(case);verify_costs(account,mode)
                if any(a['kind']=='share_exchange' for a in account['corporate_actions']):
                    import pandas as pd
                    from skills.legacy_merger_audit import audit_exchange
                    from skills.legacy_merger_settlement import load_terms, NOTE
                    _,merger_refs=load_terms()
                    for name,sha in merger_refs.items():sources.bind(name,sha)
                    quotes=pd.read_parquet(ROOT/'.cache/poc-latest-20261003/inputs-v1/quotes-unmasked.parquet')
                    proof=audit_exchange(account,quotes)
                    if case['audit'].get('share_exchange')!=proof:
                        raise ValueError('Published exchange proof differs from independent inventory')
                    row['merger_inventory']=proof
                    if NOTE not in row['settlement_evidence_notes']:row['settlement_evidence_notes'].append(NOTE)
                metrics=account_metrics(case)
                if metrics['trading_days']!=666 or (metrics['start'],metrics['end'])!=('2024-01-02','2026-10-02'):
                    raise ValueError('Incomplete date range')
                row.update(metrics=metrics,total_return=metrics['total_return'],max_drawdown=metrics['max_drawdown'],
                    price_audit=check_prices(account),
                    sell_reasons=dict(Counter(t['reason'] for t in account['trades'] if t['side']=='sell')),
                    gross_turnover=sum(t['reference_price']*t['qty'] for t in account['trades']))
                accounts[f'{arm}:{mode}']=account
            select_attempt(cases,f'{arm}:{mode}',row,repair)
    if set(cases)!={f'{arm}:{mode}' for arm in ARMS for mode in MODES}:
        raise ValueError('All twelve cases must have an attempted result')
    top3,top3_ref=prior_publication(sources,'top3_validation_20261012')
    parity={};diagnostics={}
    for mode in MODES:
        key=f'benchmark:{mode}';row=cases[key]
        old=top3['cases'][key]
        if row['completed']:
            previous=json.loads(sources.bind(old['path'],old['sha256']).read_text())['account']
            same={name:accounts[key][name]==previous[name] for name in JOURNALS}
            if not all(same.values()):raise ValueError('Same-cost 0050 financial journals changed: '+mode)
            parity[mode]=dict(all_exact=True,journals=same,previous_case=dict(path=old['path'],sha256=old['sha256']))
        for arm in ARMS:
            candidate=cases[f'{arm}:{mode}'];candidate['benchmark_comparable']=row['completed'] and candidate['completed']
            if candidate['benchmark_comparable']:
                candidate['excess_return']=candidate['total_return']-row['total_return']
                candidate['annual_excess']=[dict(year=a['year'],total_return=a['total_return'],
                    benchmark_return=b['total_return'],excess=a['total_return']-b['total_return'])
                    for a,b in zip(candidate['metrics']['annual'],row['metrics']['annual'])]
                candidate['rolling252']=windows(accounts[f'{arm}:{mode}']['daily'],accounts[key]['daily'])
    for arm in ARMS[:-1]:
        rows=[cases[f'{arm}:{mode}'] for mode in MODES if cases[f'{arm}:{mode}']['benchmark_comparable']]
        diagnostics[arm]=dict(name=NAMES[arm],compared_stresses=len(rows),
            benchmark_wins=sum(r['excess_return']>0 for r in rows),
            positive_stresses=sum(r['total_return']>0 for r in rows),
            worst_return=min((r['total_return'] for r in rows),default=None),
            worst_drawdown=min((r['max_drawdown'] for r in rows),default=None))
    universe,universe_ref=prior_publication(sources,'stock_universe_2019_20260929',
        '17bd7f2f0ff6ebb23fbd8b21ebad48be64ee346f21929d8650f61ee666c9b8a7')
    poc,poc_ref=prior_publication(sources,'poc_broker_account_20261004')
    legacy_paths={}
    old_sources={
        'liquid_universe':ROOT/'.cache/stock-universe-2019-20260929/final-a/liquid_universe.json',
        'poc_priority':ROOT/'.cache/poc-broker-account-20261004/offline-v3/poc_red.json',
    }
    # The old raw accounts are provenance-bound by their original run reports.
    for arm,old_path in old_sources.items():
        parent=old_path.parent/'report.json'
        parent_name=str(parent.relative_to(ROOT))
        parent_hash=(next(r['sha256'] for r in universe['source_reports'] if r['path']==parent_name)
            if arm=='liquid_universe' else poc['source_sha256'][parent_name])
        parent_data=json.loads(sources.bind(parent_name,parent_hash).read_text())
        descriptor=parent_data['cases']['liquid_universe' if arm=='liquid_universe' else 'poc_red']
        if descriptor['path']!=str(old_path.relative_to(ROOT)):
            raise ValueError('Legacy path differs from its original run')
        previous=json.loads(sources.bind(descriptor['path'],descriptor['sha256']).read_text())
        old_return=next(r['total_return'] for r in universe['comparison'] if r['arm']=='liquid_universe') if arm=='liquid_universe' else poc['arms']['poc_red']['metrics']['total_return']
        if abs(previous['summary']['total_return']-old_return)>1e-10:
            raise ValueError('Old account does not explain the cited old return')
        if f'{arm}:baseline' in accounts:
            legacy_paths[arm]=dict(compare_paths(previous['account'],accounts[f'{arm}:baseline']),
                old_account=dict(path=descriptor['path'],sha256=descriptor['sha256']))
    candidate_path=BASE/'candidates-v1/candidates.json'
    candidate=json.loads(sources.bind(str(candidate_path.relative_to(ROOT)),digest(candidate_path)).read_text())
    failed_setups=[]
    for p in sorted(BASE.glob('*/failure.json')):
        record=json.loads(sources.bind(str(p.relative_to(ROOT)),digest(p)).read_text())
        failed_setups.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),failure=record))
    result=dict(schema='legacy_return_comparison_v1',created_at=datetime.now(timezone.utc).isoformat(),
        start='2024-01-02',end='2026-10-02',initial_cash=1_000_000,position_count=3,
        buy_fraction=.5,sell_fraction=.3,modes={k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()},
        cases=cases,diagnostics=diagnostics,completed_cases=sum(r['completed'] for r in cases.values()),
        expected_cases=12,all_completed=all(r['completed'] for r in cases.values()),
        live_qualified=False,unseen_validation=False,actual_fill_verified=False,paper_enabled=False,
        benchmark_parity=parity,previous_publication=top3_ref,
        hard_filter_reference={mode:top3['cases'][f'poc_up_red:{mode}'] for mode in MODES},
        candidate_differences=candidate['old_candidate_comparison'],candidate_count=len(candidate['entries']),
        candidate_prefix_checks=candidate['prefix_checks'],path_differences=legacy_paths,
        old_versions=[dict(id='liquid_universe',name=NAMES['liquid_universe'],start=universe['start'],end=universe['end'],
            total_return=next(r['total_return'] for r in universe['comparison'] if r['arm']=='liquid_universe'),publication=universe_ref),
            dict(id='poc_priority',name=NAMES['poc_priority'],start=poc['start'],end=poc['end'],
            total_return=poc['arms']['poc_red']['metrics']['total_return'],publication=poc_ref)],
        trial_accounting=dict(total_recorded_attempts=len(trials)+len(failed_setups),
            incomplete_attempts=sum(r['completed'] is False for r in trials)+len(failed_setups),setup_failures=failed_setups),
        differences=[
            '949%原期間2019/1/2～2026/9/9；本輪2024/1/2重置100萬開始，終點延長至10/2。不能切舊曲線冒充新帳戶。',
            '擴大股池是日均成交金額至少5000萬；POC原版還有20日成交金額中位數至少5000萬、紅K與POC優先排序。',
            '擴大股池保留12%停損／63日期限；POC優先版另有三黑K出場。POC優先不是POC硬篩選。',
            '兩套舊版買賣皆HL2。本輪买HL2、卖30%，並統一完整普通盤與零股容量、權息及資源核帳。',
            'POC永久品質未知時依舊版明示政策恢復當日原排序；取得資料失敗仍停止。此行為與硬篩選排除未知不同。',
            '手續費／滑價改變可買股數與後續交易路徑；跨版本差額不是單一條件的因果貢獻。'],
        limitations=['全部為已看過歷史，沒有新增真正樣本外紀錄。',
            'HL2須全日結束才知，是事後成交代理；未核實實際委託、排隊及成交。',
            '歷史全市場身分、首次發布與修訂資料尚未完整認證；資料排除明示保留。'],
        reports=reports,checked_source_count=len(sources.used),
        reporting_sources={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),SPEC,ROOT/'skills/legacy_return_comparison.py')})
    write(output,result);Path(output).with_suffix('.sha256').write_text(digest(output)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',action='append',type=Path,required=True)
    p.add_argument('--replacement-report',action='append',type=Path,default=[])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=summarize(a.report,a.output,a.replacement_report)
    print(json.dumps({k:r[k] for k in ('completed_cases','checked_source_count','diagnostics')}))
