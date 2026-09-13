#!/usr/bin/env python3
"""Publish compact evidence only after all paired accounts reproduce offline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read, write, sha, encoded
from scripts.research_observed_risk import OUTPUT, AUDIT, identity

TARGET = ROOT / 'artifacts/forward_simulation/observed_risk_delivery_20260913.json'


def export(output=OUTPUT, target=TARGET):
    if target.exists():
        raise ValueError('Use a new publication path')
    verified = read(output/'offline-verification.json')
    if (verified['all_identical'] is not True or verified['network_calls'] != 0 or
        verified['manifest_sha256'] != sha(output/'manifest.json') or
        verified['source_identity_sha256'] != sha(output/'identity.json') or
        identity() != read(output/'identity.json')):
        raise ValueError('Offline verification is stale or incomplete')
    for name, digest in read(output/'manifest.json')['output_sha256'].items():
        if sha(output/name) != digest:
            raise ValueError('Published evidence changed: '+name)
    summary = read(output/'summary.json')
    cases = summary['cases']
    if not summary['all_completed'] or len(cases) != 14 or not all(r['completed'] for r in cases.values()):
        raise ValueError('All fourteen cases must complete')

    def row(name, label):
        result = cases[name]
        stats = result['summary']
        benchmark = cases['benchmark_'+result['config']['stress']]['summary']
        return {'情境':label, '扣成本總報酬':f"{stats['total_return']*100:+.2f}%",
                '最大回撤':f"{stats['max_drawdown']*100:.2f}%", '期末資產':round(stats['final_nav'],2),
                '同條件0050':f"{benchmark['total_return']*100:+.2f}%",
                '超額百分點':round((stats['total_return']-benchmark['total_return'])*100,2),
                '成交筆數':stats['trade_count'], '缺風控訊號天數':result['unknown_risk_days']}
    audit = read(AUDIT)
    unchanged = {}
    for stress in ('control','combined'):
        a = read(output/f'cases/original_{stress}.json')['account']
        b = read(output/f'cases/quarantine_{stress}.json')['account']
        unchanged[stress] = encoded(a) == encoded(b)
    result = dict(offline_identical=True, offline_seconds=verified['elapsed_seconds'],
        research_code_sha256={k:v for k,v in identity().items() if k.endswith('.py')},
        source_manifest_sha256=sha(output/'manifest.json'), audit_sha256=sha(AUDIT),
        case_count=len(cases), request_counts=summary['request_counts'], data_probe_requests=3,
        quarantine_account_unchanged=unchanged, live_qualified=False, unseen_validation=False,
        comparison={
            '原策略與隔離結果':[row(f'{version}_{stress}', label+'／'+condition)
                for version,label in [('original','原現金版'),('quarantine','行情隔離版')]
                for stress,condition in [('control','原成交條件'),('combined','合併成交壓力')]],
            '風控窗口比較':[row(f'{risk}_{window}_{stress}', label+'／'+window_label+'／'+condition)
                for risk,label in [('trend60','60日均線'),('shock','急跌與波動')]
                for window,window_label in [('market','原市場日窗口'),('observed','有效收盤窗口')]
                for stress,condition in [('control','原成交條件'),('combined','合併成交壓力')]],
            '錯日行情盤點':[{'日期':r['date'],'來源衝突筆數':len(r['conflicts']),
                '完全吻合舊行情筆數':len(r['exact_older_ohlcv_matches']),
                '對照舊日期':r['older_date'],'回測池內衝突股票':','.join(r['frozen_pool_conflicts'])}
                for r in audit['rows']]},
        conclusion='新急跌風控在原成交條件下報酬589.98%，合併壓力後113.75%，低於原現金版211.66%及同條件0050的239.73%。缺訊號期縮至真正缺價的5天，但這輪不替換正式策略，也不是新樣本外優勢。',
        remaining=[
            '三天158筆來源衝突僅隔離於研究副本；原DB保留，責任寫入程式與其餘歷史尚待追查。',
            '固定458候選的帳戶重播不能取代完整月度族群重建、歷史股票名冊與還原價獨立對帳。',
            '新窗口是有效收盤觀察，停牌日仍不新增；不是停牌期間每日波動的估計。',
            '未將新風控套用到正式策略；升息預期、戰爭事件時點及真正前向成交仍未完成。'])
    write(target, result)
    return result


if __name__ == '__main__':
    with file_lock(OUTPUT/'run.lock', timeout=0):
        result = export()
    print('Published', result['case_count'], 'verified cases')
