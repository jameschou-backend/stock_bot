#!/usr/bin/env python3
"""Publish close-warning diagnostics from existing ledgers, without replay."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from skills.poc_weakness import RULES, diagnose
from skills.three_black_exit import ThreeBlackSignals, audit_three_black

PARENT = 'artifacts/forward_simulation/poc_drawdown_attribution_20261019.json'
ACCOUNT = '.cache/top3-validation-20261012/baseline-v3/poc_up_red.json'
BUNDLE = '.cache/poc-latest-20261003/inputs-v1'
OUTPUT = 'artifacts/forward_simulation/poc_weakness_20261019.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root=ROOT):
    root = Path(root).resolve()
    sources = {}

    def bind(name, expected=None):
        path = (root/name).resolve()
        if not path.is_relative_to(root) or (expected and digest(path) != expected):
            raise ValueError('Warning source fingerprint differs: '+name)
        sources[name] = digest(path)
        return path

    parent = json.loads(bind(PARENT, (root/PARENT).with_suffix('.sha256').read_text().strip()).read_text())
    original = json.loads(bind(ACCOUNT, parent['sources'][ACCOUNT]).read_text())
    if original.get('completed') is not True or original['summary'] != parent['summary']:
        raise ValueError('Original account differs from published attribution')
    for name, sha in parent['sources'].items():
        bind(name, sha)
    manifest_name = BUNDLE+'/manifest.json'
    manifest = json.loads(bind(manifest_name, (root/manifest_name).with_suffix('.sha256').read_text().strip()).read_text())
    a = original['account']
    ids = sorted({c['stock_id'] for c in a['cohorts']})
    path = bind(BUNDLE+'/close-official.parquet', manifest['files_sha256']['close-official.parquet'])
    close = pd.read_parquet(path, columns=['date', *ids]).set_index('date')
    path = bind(BUNDLE+'/quotes-unmasked.parquet', manifest['files_sha256']['quotes-unmasked.parquet'])
    quotes = pd.read_parquet(path, filters=[('stock_id', 'in', ids)])
    black = ThreeBlackSignals(close, quotes, close.index)
    audit = audit_three_black(a, black)
    # Check that the ledger marks share the raw price source used for warnings.
    for row in a['holdings']:
        value = black.raw['close'].at[pd.Timestamp(row['mark_date']), row['stock_id']]
        if abs(value-row['price']) > 1e-7:
            raise ValueError('Ledger mark differs from frozen warning quotes')
    result = diagnose(a, close, quotes, parent['episodes'])
    for name in ('scripts/analyze_poc_weakness.py', 'skills/poc_weakness.py',
                 'skills/three_black_exit.py', 'docs/poc_weakness_diagnostic_20261019.md'):
        bind(name)
    report = dict(schema='poc_weakness_v1', strategy_id='poc_up_red', mode='baseline',
        start=parent['start'], end=parent['end'], initial_cash=parent['initial_cash'],
        summary=parent['summary'], benchmark_summary=parent['benchmark_summary'],
        **result, sources=sources, rules=RULES,
        original_black_audit=dict(decisions_checked=audit['decisions_rebuilt'], exits=audit['exits']),
        backtests_run=0, new_accounts=0, reused_accounts=1, finmind_requests=0,
        descriptive_only=True, strategy_changed=False, counterfactual_backtest=False,
        actual_fill_verified=False, unseen_validation=False, paper_enabled=False, live_qualified=False,
        method=dict(first_warning='first_profitable_signal_since_entry_before_original_exit_decision',
            profit='adjusted_close_above_entry_day_adjusted_close_not_net_trade_profit',
            timing='completed_close_T_to_earliest_next_market_session_T_plus_1',
            episode_selection='reuse_five_sealed_drawdown_episodes_without_resetting_warning_at_peak',
            controls='all_73_original_funded_cohorts_including_non_drawdown_periods',
            outcome='close_path_until_original_first_exit_decision_or_censored_period_end'),
        candidate_rules=[dict(id=k, name=v, status='diagnosed_not_backtested',
            action='隔日減原持有股數三分之一；每批一次、保留其餘約三分之二，原版完整出場優先') for k,v in RULES.items()],
        limitations=[
            '這是原持股路徑的警訊診斷，沒有減碼後的新帳戶，不能換算少跌多少或少賺多少。',
            '價格獲利採還原收盤相對進場日收盤，未扣費，也不是實際買進成交價損益。',
            '第一次警訊從進場開始找；發生於回撤高點前也照列，不能事後挑選最接近大跌的一次。',
            '後續價格與是否創新高僅作事後誤報診斷，不參與警訊；全部樣本已看過，沒有樣本外資格。',
            '均線包含當天及前19個市場交易日，缺資料不補值；所有訊號只用當日及更早資料。',
            '提前天數對照原版首次出場決策日，排除原指令已出但整股或零股未成交的等待期。',
            '隔日是最早可提出減碼的日期，日行情及原帳本不能證明新委託成交價、量或成交時點。',
            '未出場批次為右設限；五段低點後的警訊不能解釋該次回撤，另標註日期位置。'])
    for name, sha in sources.items():
        if digest(root/name) != sha:
            raise ValueError('Source changed during warning diagnosis')
    return report


def main():
    from app.poc_weakness_ui import validate
    output = ROOT/OUTPUT
    if output.exists():
        if digest(output) != output.with_suffix('.sha256').read_text().strip():
            raise ValueError('Existing warning publication fingerprint differs')
        report = validate(json.loads(output.read_text()))
        for name, sha in report['sources'].items():
            if digest(ROOT/name) != sha:
                raise ValueError('Published warning source changed')
        print(json.dumps(dict(output=OUTPUT, reused=True, backtests_run=0)))
        return
    report = validate(build())
    raw = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    output.write_bytes(raw)
    output.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    print(json.dumps(dict(output=OUTPUT, backtests_run=0, cohorts=report['total_cohorts'],
        summaries=report['rule_summaries']), ensure_ascii=False))


if __name__ == '__main__':
    main()
