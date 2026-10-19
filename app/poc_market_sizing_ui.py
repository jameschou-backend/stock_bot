"""Strict scope and provenance reader for the isolated market sizing comparison."""
import hashlib
import json
import math
from pathlib import Path

from skills.top3_validation import MODES

IDENTIFIER='poc_market_sizing_20261019'
VARIANT='market120_half'


def validate(report):
    expected={f'{arm}:{variant}:{m}' for m in MODES for arm,variant in
        [('poc_up_red','original'),('benchmark','original'),('poc_up_red',VARIANT)]}
    if (report.get('schema')!='poc_market_sizing_publication_v1'
            or (report.get('start'),report.get('end'),report.get('initial_cash'),report.get('position_count'))
                !=('2024-01-02','2026-10-02',1000000,3)
            or report.get('variant')!=VARIANT or set(report.get('cases',{}))!=expected
            or (report.get('completed_new_accounts'),report.get('reused_accounts'))!=(4,8)
            or report.get('modes')!={k:dict(commission=v[0],slippage=v[1]) for k,v in MODES.items()}
            or any(report.get(k) is not False for k in ('actual_fill_verified','unseen_validation','paper_enabled','live_qualified'))):
        raise ValueError('大盤轉弱新買減半比較範圍或資格標示不符')
    rule=report.get('rule',{})
    if (rule.get('market'),rule.get('price'),rule.get('window_valid_observations'),
            rule.get('weak_condition'),rule.get('weak_new_buy_factor'),rule.get('normal_new_buy_factor'),
            rule.get('execution_lag_sessions'),rule.get('existing_holdings')) != (
            '0050','adjusted_close',120,'close_strictly_below_MA120',.5,1.,1,
            'original_exits_only_no_trim_no_top_up'):
        raise ValueError('大盤轉弱新買減半門檻不符')
    overlap=report.get('candidate_market_overlap',{})
    counts=overlap.get('candidate_states',{})
    if (set(counts)!={'normal','weak','unknown'} or any(type(v) is not int or v<0 for v in counts.values())
            or sum(counts.values())!=overlap.get('candidate_count') or not overlap.get('candidate_count')
            or overlap.get('scope')!='original_red_candidates_before_POC_hard_filter'):
        raise ValueError('缺少候選與市場狀態交集核對')
    for key,row in report['cases'].items():
        if key!=f"{row['id']}:{row['variant']}:{row['mode']}" or row.get('completed') is not True:
            raise ValueError('大盤轉弱新買減半案例缺件')
        metrics=row['metrics']
        if metrics['trading_days']!=666 or any(not isinstance(metrics.get(k),(int,float))
                or not math.isfinite(metrics[k]) or abs(metrics[k]-row['summary'][k])>1e-7
                for k in ('total_return','max_drawdown','final_nav')):
            raise ValueError('大盤轉弱新買減半績效摘要不一致')
        if row['variant']==VARIANT and (row.get('market_sizing_audit',{}).get('passed') is not True
                or row.get('unchanged_before_first_trigger') is not True or row.get('reused') is not False):
            raise ValueError('大盤轉弱新買減半缺少獨立核對')
        if row['variant']=='original' and row.get('reused') is not True:
            raise ValueError('原版與0050必須重用')
    for mode in MODES:
        new=report['cases'][f'poc_up_red:{VARIANT}:{mode}']['metrics']
        for target,arm in [('original','poc_up_red'),('benchmark','benchmark')]:
            old=report['cases'][f'{arm}:original:{mode}']['metrics'];d=report['comparisons'][mode][target]
            for key,value in dict(return_change=new['total_return']-old['total_return'],
                    drawdown_improvement=new['max_drawdown']-old['max_drawdown'],
                    profit_change=new['final_nav']-old['final_nav'],
                    net_profit_retained=new['profit']/old['profit'] if old['profit'] else None).items():
                if d.get(key)!=value:raise ValueError('大盤轉弱新買減半使用了不同成本或錯誤比較')
    return report


def case_detail(publication,case_id,root):
    validate(publication)
    row=next((r for k,r in publication['cases'].items() if k.replace(':','__')==case_id),None)
    if not row:raise ValueError('找不到大盤轉弱新買減半案例')
    root=Path(root).resolve();path=(root/row['path']).resolve()
    if not path.is_relative_to(root) or path.suffix!='.json':raise ValueError('案例路徑錯誤')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('案例指紋不符')
    case=json.loads(raw);a=case['account']
    if (case.get('completed') is not True or case['summary']!=row['summary']
            or case.get('live_qualified') is not False or case.get('actual_fill_verified') is not False):
        raise ValueError('案例摘要或資格不符')
    settings=a['settings']
    if (len(a['daily'])!=666 or settings['initial_cash']!=1000000
            or (settings['commission'],settings['slippage'])!=MODES[row['mode']]
            or (a['daily'][0]['date'],a['daily'][-1]['date'])!=(publication['start'],publication['end'])):
        raise ValueError('案例期間、本金或成本不符')
    if row['variant']==VARIANT:
        meta=case.get('risk_controls',{})
        if (meta.get('arm'),meta.get('mode'),meta.get('variant'),meta.get('preregistration'))!=(
                row['id'],row['mode'],VARIANT,publication['preregistration']):
            raise ValueError('案例與事前登記不符')
    name='0050' if row['id']=='benchmark' else 'POC硬篩＋紅K・'+('大盤轉弱新買減半' if row['variant']==VARIANT else '原版')
    return dict(id=case_id,name=name,summary=case['summary'],initial_cash=1000000,position_count=1 if row['id']=='benchmark' else 3,
        position_scope='active_stock_slots',daily=[{k:r.get(k) for k in ('date','nav','cash','market_value','receivable','drawdown','holdings')} for r in a['daily']],
        trades=[dict({k:r.get(k) for k in ('date','signal_date','stock_id','name','side','qty','reason','total_cost','cash_after')},price=r['reference_price']) for r in a['trades']],
        live_qualified=False,actual_fill_verified=False,source_sha256=row['sha256'],
        note='已見歷史研究；買HL2／賣30%日區間估算，未驗證實際成交。')
