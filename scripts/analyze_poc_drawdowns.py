#!/usr/bin/env python3
"""Publish a read-only attribution of already completed POC/0050 ledgers."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.cohort_drawdown import analyze
from app.poc_drawdown_ui import validate

PARENT = 'artifacts/forward_simulation/poc_market_sizing_20261019.json'
OUTPUT = 'artifacts/forward_simulation/poc_drawdown_attribution_20261019.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(root=ROOT):
    root = Path(root)
    raw = (root/PARENT).read_bytes()
    if sha(raw) != (root/PARENT).with_suffix('.sha256').read_text().strip():
        raise ValueError('Parent publication fingerprint differs')
    parent = json.loads(raw)
    sources = {PARENT: sha(raw)}
    cases = []
    for key in ('poc_up_red:original:baseline', 'benchmark:original:baseline'):
        ref = parent['cases'][key]
        path = (root/ref['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Account source outside repository')
        raw = path.read_bytes()
        if sha(raw) != ref['sha256']:
            raise ValueError('Frozen account fingerprint differs')
        case = json.loads(raw)
        if case.get('completed') is not True or case['summary'] != ref['summary']:
            raise ValueError('Account completion/summary differs')
        if any(case.get(k) is not False for k in ('live_qualified', 'actual_fill_verified', 'unseen_validation')):
            raise ValueError('Account qualification differs')
        sources[ref['path']] = sha(raw)
        cases.append(case)
    original, benchmark = cases
    a, b = original['account'], benchmark['account']
    for account in (a, b):
        settings = account['settings']
        if (settings['initial_cash'], settings['commission'], settings['slippage']) != (1000000, .001425, .0045):
            raise ValueError('Original baseline account configuration differs')
    result = analyze(a, b['daily'], top=5)
    for name in ('scripts/analyze_poc_drawdowns.py', 'skills/cohort_drawdown.py'):
        sources[name] = sha((root/name).read_bytes())
    report = dict(schema='poc_drawdown_attribution_v1', strategy_id='poc_up_red', mode='baseline',
        start=original['summary']['start'], end=original['summary']['end'], initial_cash=1000000,
        position_count=3, summary=original['summary'], benchmark_summary=benchmark['summary'],
        **result, sources=sources, backtests_run=0, new_accounts=0, reused_accounts=2,
        finmind_requests=0, strategy_changed=False, descriptive_only=True, counterfactual_backtest=False,
        actual_fill_verified=False, unseen_validation=False, paper_enabled=False, live_qualified=False,
        method=dict(selection='five_deepest_disjoint_high_watermark_episodes', window='peak_exclusive_trough_inclusive',
            winner='cohort_net_pnl_at_peak_positive_including_prior_cash_flows_and_unpaid_rights',
            new_entry='first_funded_entry_after_account_peak',
            pnl_identity='holding_value_change + cash_change + receivable_change',
            cost_separation='cohort_gross_pnl = net_pnl + sealed_window_cost; subtract cost once',
            benchmark='0050 return over exactly the same peak-to-trough dates, not its maximum drawdown'),
        limitations=[
            '歷史日區間估算成交；未證明盤中可成交，帳本內應收權益沿用原估值。',
            '舊贏家按回撤高點當時的批次累計淨獲利分類；跌價可超過既有獲利，超過部分另列。',
            '新買批次的損益只算到本段低點；可能仍持有，不以後來的最終輸贏分類。',
            '費稅與滑價自批次損益拆出，沒有重複扣款；加回成本只是固定持股路徑的會計拆解，非零成本回測。',
            '回撤高低點為事後辨識，不能直接當事前交易訊號，也未證明任何新風控能保留報酬。',
            '五段按回撤幅度排名且不重疊；跨年期間不拆開，未回復高點的期間列為尚未修復。'],
        findings=[
            '五段中四段的最大負向項是高點時獲利的舊持股跌價；最深一段也伴隨大量接替新買虧損。',
            '最深一段0050同期仍上漲，僅用大盤均線無法充分反映所持個股轉弱。',
            '優先研究既有持股同步轉弱時的曝險與出場；接替新買的失敗累積其次。成本是放大因素，並非所有回撤的主因。',
            '先用原帳本診斷轉弱前兆與回吐時序，再另訂單一規則驗證；本次不新增停利、減碼或新買限制。'])
    return validate(report)


def main():
    report = build()
    path = ROOT/OUTPUT
    raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('Preserve existing attribution publication; use an explicit new version')
    path.write_bytes(raw)
    path.with_suffix('.sha256').write_text(sha(raw)+'\n')
    print(json.dumps(dict(output=OUTPUT, episodes=len(report['episodes']), backtests_run=0, sha256=sha(raw))))


if __name__ == '__main__':
    main()
