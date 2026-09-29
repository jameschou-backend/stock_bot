#!/usr/bin/env python3
"""Publish two identical, complete historical accounts and their audit trail."""
from pathlib import Path
import sys
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_partial_risk_period import verify
from scripts.research_exit_scenarios import read, write, sha


def export():
    base = ROOT/'.cache/partial-risk-2019-20260929'
    output = ROOT/'artifacts/forward_simulation/partial_risk_2019_20260929'
    document = ROOT/'docs/research_partial_risk_2019_20260929.md'
    publication = output.with_suffix('.json')
    if output.exists() or document.exists() or publication.exists():
        raise ValueError('Preserve existing publication; choose a new experiment version')
    reports = [read(base/name/'report.json') for name in ('final-a', 'final-b')]
    if any(not r.get('validated') or r['start'] != '2019-01-02' or r['end'] != '2026-09-09'
           for r in reports):
        raise ValueError('Historical period or independent audit is incomplete')
    report = verify(base/'final-a', base/'final-b', base/'publication-verified.json')
    prefix_path = base/'signal-prefix-final.json'
    prefix = read(prefix_path)
    if set(prefix['checks']) != {'2020-12-31', '2021-12-30'} or any(
            not row['same_selected_stocks'] or not row['future_prices_and_events_removed']
            for row in prefix['checks'].values()):
        raise ValueError('Historical signal prefix audit missing')
    for path, digest in prefix['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Signal prefix audit source changed')
    report['signal_prefix_audit'] = prefix
    report['source_sha256'].update(prefix['source_sha256'])
    report['source_sha256'][str(prefix_path.relative_to(ROOT))] = sha(prefix_path)
    report['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    output.mkdir()
    names = {'original': '原版 loss12/time63', 'cap40': '單股超過 40% 減至約三分之一',
             'benchmark': '0050 股息再投入'}
    results, annual = {}, []
    for arm in names:
        case = read(base/'final-a'/(arm+'.json'))
        results[arm] = case['summary']
        for name, key in [('trades', 'trades'), ('daily', 'daily'), ('cash', 'cash_ledger'),
                          ('orders', 'orders'), ('holdings', 'holdings')]:
            pd.DataFrame(case['account'][key]).to_csv(output/f'{arm}-{name}.csv', index=False, encoding='utf-8-sig')
        for row in case['summary']['annual']:
            annual.append(dict(arm=arm, **row))
    pd.DataFrame(annual).to_csv(output/'annual.csv', index=False, encoding='utf-8-sig')
    def pct(value): return f'{value*100:+.2f}%'
    lines = ['# 2019 年起固定策略回測結果', '',
        '**已完成 2019-01-02 至 2026-09-09，1,867 個交易日、两次獨立離線完整回放。**', '',
        '本金 100 萬元、複利。原版與 cap40 只買個股，閒錢持現金；0050 僅作基準。',
        '收盤訊號隔一個交易日執行；不使用額外一天延遲。', '',
        '| 固定版本 | 累積淨報酬 | 年化報酬 | 最大回撤 | 期末資產 | 成交成本 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, name in names.items():
        s = results[arm]
        lines.append(f"| {name} | {pct(s['total_return'])} | {pct(s['cagr'])} | {pct(s['max_drawdown'])} | {s['final_nav']:,.2f} | {s['costs']['total_cost']:,.0f} |")
    lines += ['', '**拉長到 2019 年，兩個個股版本都輸給 0050。不能把 2022 年起勝出擴大解讀成跨時期穩定優勢。**', '',
              '| 年度（連續帳戶） | 原版 | cap40 | 0050 |', '|---|---:|---:|---:|']
    for year in range(2019, 2027):
        values = [next(row['total_return'] for row in results[a]['annual'] if row['year'] == str(year)) for a in names]
        lines.append('| '+('2026 至 9/9' if year == 2026 else str(year))+' | '+' | '.join(map(pct, values))+' |')
    lines += ['', '2019–2021 年原版只由 100 萬變成約 102 萬，cap40 約 101 萬，0050 約 215 萬。',
        '策略優勢集中在 2023、2025 年；2024 年又明顯落後。cap40 提高全期報酬，',
        '但兩個個股版本全期最大回撤同為約 36.03%，發生在 2020-06-08。',
        '因此這次沒有證據說單靠比重限制能解決跨時期回撤，或已穩健跑贏大盤。', '',
        '## 本次補齊與核對', '',
        '- 延伸 2018 年暖機及 2019 年起行情、官方除權息、歷史停牌、退市與轉板身分。',
        '- 重建 667 筆選股訊號；2022 年起 454 筆股票與訊號日和先前名單一致。',
        '- 刪除 2020 年末、2021 年末以後的行情及公司行動，重建早期選股；114、213 筆訊號一致。',
        '- 補齊柏文 2019、彰銀 2019、潤泰新 2021 的股票股利比例、交付日及碎股折現條款。',
        '- 2020-10-26 以前採當時的盤後零股單次成交資料；以後用盤中零股日資料。',
        '- 原版 424 筆成交、cap40 442 筆、0050 54 筆；兩次所有帳本完全一致。',
        '- 獨立重算交易成本、逐日現金／持股／NAV、先前日配置、渠道成交量、減碼規則與交易時序。', '',
        '## 解讀範圍', '',
        '整股及 2020-10-26 起的零股採使用者指定高低價中點；較早盤後零股採官方單次成交價。',
        '分渠道計入手續費、交易稅、每邊 0.45% 滑價及 1% 成交量上限。這些是成交代理假設，',
        '沒有把日高低中點稱為可以事先知道或一定成交的價格。', '',
        '已完成的是這組固定策略與已重建股票群體的歷史回放，不是宣稱全市場資料毫無遺漏。',
        '身分群體尚不能證明完全消除存活者偏誤；舊分類與資料修訂、零股分配順位仍有限制。',
        '三筆新增配股的碎股現金以新股交付日結清，是明列的入帳日假設。',
        '`live_qualified=false`、`actual_fill_verified=false`、`unseen_validation=false`。', '',
        '## 檔案與重跑', '',
        '- `artifacts/forward_simulation/partial_risk_2019_20260929.json`：完整來源指紋與比較結果。',
        '- 同名資料夾：全部買賣（含價格、理由、股數、稅費）、未成交委託、每日資產、持倉、現金流水、年度表。',
        '- `python scripts/research_partial_risk_2019.py --output <新的資料夾名稱>`：使用封存輸入离線回測。',
        '- `scripts/audit_partial_risk_2019_prefix.py`：刪除未來資料的訊號一致性核對。',
        '- `docs/partial_risk_2019_corporate_terms.json`：新增配股條款與原始證據指紋。', '',
        '驗收：4,132 項測試通過；`make pipeline` 完成；本機 API health/picks/models/jobs 均為 HTTP 200。', '']
    document.write_text('\n'.join(lines))
    report['export_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in [document, *output.iterdir()]}
    write(publication, report)
    publication.with_suffix('.sha256').write_text(sha(publication)+'\n')
    print(json.dumps({a: {k: s[k] for k in ('total_return', 'max_drawdown', 'final_nav')}
                      for a, s in results.items()}, ensure_ascii=False))


if __name__ == '__main__':
    export()
