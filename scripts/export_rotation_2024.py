#!/usr/bin/env python3
"""Publish only complete, independently reproducible 2024 rotation comparisons."""
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from scripts.export_midpoint_2025_report import verify_cash

ARMS=('original','relaxed','stagnant','stronger','combined','benchmark')


def verify_rotation(left,right,output):
    left,right,output=Path(left).resolve(),Path(right).resolve(),Path(output).resolve()
    if left==right or output.exists():
        raise ValueError('Require independent runs and a new output')
    reports=[read(folder/'report.json') for folder in (left,right)]
    for report in reports:
        if (set(report['cases'])!=set(ARMS) or not report['all_completed'] or not report['validated']
                or report['preparation'] or report['start']!='2019-01-02' or report['end']!='2024-12-31'
                or report['initial_cash']!=1_000_000 or report['live_qualified'] is not False):
            raise ValueError('Missing or invalid fixed rotation comparison')
    if reports[0]['source_sha256']!=reports[1]['source_sha256']:
        raise ValueError('Independent run source sets differ')
    refs=dict(reports[0]['source_sha256'])
    for arm in ARMS:
        cases=[read(folder/(arm+'.json')) for folder in (left,right)]
        if cases[0]!=cases[1]:raise ValueError('Independent rotation accounts differ: '+arm)
        for folder,report,case in zip((left,right),reports,cases):
            path=folder/(arm+'.json');digest=sha(path)
            if not case['completed'] or digest!=report['cases'][arm]['sha256']:
                raise ValueError('Rotation case changed: '+arm)
            verify_cash(case)
            refs[str(path.relative_to(ROOT))]=digest
            refs[str((folder/'report.json').relative_to(ROOT))]=sha(folder/'report.json')
    for rel,digest in refs.items():
        if sha(ROOT/rel)!=digest:raise ValueError('Publication source changed: '+rel)
    report=dict(reports[0],source_sha256=refs,offline_identical=True)
    write(output,report)
    return report


def main():
    base=ROOT/'.cache/rotation-2024-20260929'
    target=ROOT/'artifacts/forward_simulation/rotation_2024_20260929'
    document=ROOT/'docs/research_rotation_2024_20260929.md'
    if target.exists() or target.with_suffix('.json').exists() or document.exists():
        raise ValueError('Preserve published experiment')
    names=dict(original='原版',relaxed='只放寬族群',stagnant='原選股＋停滯退出與候補',
        stronger='原選股＋強弱換股與候補',combined='放寬族群＋兩種換股與候補',benchmark='0050 股息再投入')
    for name in ('final-a','final-b'):
        report=read(base/name/'report.json')
        if (not report['validated'] or not report['all_completed'] or set(report['cases'])!=set(names)
                or report['start']!='2019-01-02' or report['end']!='2024-12-31'):
            raise ValueError('Incomplete fixed comparison')
    report=verify_rotation(base/'final-a',base/'final-b',base/'publication-verified.json')
    report['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    target.mkdir()
    rows=[]
    for arm in names:
        case=read(base/'final-a'/(arm+'.json'));account=case['account']
        annual=next(row for row in case['summary']['annual'] if row['year']=='2024')
        trades=[t for t in account['trades'] if t['date'].startswith('2024')]
        bought=sorted({t['stock_id'] for t in trades if t['side']=='buy'})
        rows.append(dict(arm=arm,label=names[arm],**annual,cost=sum(t['total_cost'] for t in trades),
            trade_count=len(trades),stocks_bought=len(bought),bought=bought,
            rotation_instructions=len(account.get('rotation_decisions',[]))))
        for key in ('daily','trades','orders','holdings','cash_ledger'):
            records=[r for r in account[key] if r['date'].startswith('2024')]
            pd.DataFrame(records).to_csv(target/f'{arm}-{key}.csv',index=False,encoding='utf-8-sig')
        if arm!='benchmark':
            write(target/(arm+'-rotation-decisions.json'),account.get('rotation_decisions',[]))
    baseline=next(r for r in rows if r['arm']=='original')['total_return']
    benchmark=next(r for r in rows if r['arm']=='benchmark')['total_return']
    for row in rows:
        row.update(excess_original=row['total_return']-baseline,excess_0050=row['total_return']-benchmark)
    pd.DataFrame(rows).to_csv(target/'comparison.csv',index=False,encoding='utf-8-sig')
    pct=lambda x:f'{x*100:+.2f}%'
    lines=['# 2024 族群放寬與換股回測', '',
        '固定五組個股策略與 0050；全數完整跑完，兩次離線帳本一致。',
        '個股帳戶延續同一份 2019–2023 帳本，僅 2024 改規則。以下均為 2024 年度，',
        '不是 2019 起累積績效，也不是 2024 年重新投入 100 萬的帳戶。', '',
        '| 版本 | 2024 淨報酬 | 年內最大回撤 | 年初 NAV | 年末 NAV | 年內交易成本 | 成交筆數 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['label']} | {pct(r['total_return'])} | {pct(r['max_drawdown'])} | {r['start_nav']:,.2f} | {r['end_nav']:,.2f} | {r['cost']:,.0f} | {r['trade_count']} |")
    weaker=[r for r in rows if r['arm'] not in ('original','benchmark') and r['total_return']<baseline]
    if len(weaker)==4:
        lines+=['', '**這四個固定改法都未改善原版的 2024 年淨報酬；不採用其中任何一版替換原策略。**',
            '放寬限制確實讓部分漏選股進入帳戶，但也改變資金占用與後續買入次序。',
            '換股版本增加交易成本；本次候補／換股與原版是整個組合路徑比較，',
            '不能只看某檔新增贏家，就推論帳戶報酬會上升。']
    lines+=['', '## 規則', '',
        '- 原版保留 12% 停損、63 個交易日期限、三個持股名額；閒錢持現金。',
        '- 放寬族群：取消群組大小、同組走強不超過 40%、每組每月僅一次領漲限制。',
        '  保留月初合格前 300 名、5,000 萬成交金額門檻、126 日歷史品質、突破、量增與大盤開關。',
        '- 停滯：持有滿 20 個交易日，最近 20 日漲幅小於 3%，且落後 0050，次日退出。',
        '- 強弱換股：名額滿，舊股持有至少 10 日且最近 20 日落後或持平大盤；',
        '  新候選 20 日超額報酬至少高出 10 個百分點，才退出最弱一檔。',
        '- 換股版本包含 5 個交易日候補，每日以當時收盤重新確認，明日執行；',
        '  這組比較同時包含候補機制，不能把改善全部歸因於退出本身。',
        '- 保持前一日資料判斷、開盤現金與名額鎖定。賣出後的名額／資金在後續交易日',
        '  才可支持換入，候補仍可能失效、無量或無法成交。', '',
        '## 買到哪些 2024 年回顧贏家', '',
        '此名單只用於事後歸因，從未用於新策略選股或排序。', '',
        '| 版本 | 光聖 | 聯鈞 | 所羅門 | 羅昇 | 皇昌 | 和椿 |',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        if r['arm']=='benchmark':continue
        lines.append('| '+r['label']+' | '+' | '.join('有買進' if s in r['bought'] else '未買進'
                     for s in ('6442','3450','2359','8374','2543','6215'))+' |')
    lines+=['', '## 驗證與限制', '',
        '- 原版與 0050 的 2019–2024 逐日帳戶、交易、委託、持股、現金與前次封存完全相同。',
        '- 所有修改版到 2023 年底的上述帳本完全相同，排除起始部位差異。',
        '- 刪除 2024 年後半年輸入後，前半年所有版本選股／候補名單一致。',
        '- 換股指令逐筆以實際進場日、前一日收盤報酬及候選強弱重新核對；',
        '  帳本沿用獨立費稅、現金、股數、成交容量及來源價格核算。',
        '- 買賣採分渠道日高低價中點代理價、原費稅及每邊 0.45% 滑價；不保證真實成交。',
        '- 選定 2024 年是因為已知道原版落後，因此屬樣本內研究；任何勝出都不是未見驗證。',
        '- 已知歷史股票群體仍非全市場逐日無遺漏名單，舊資料修訂與存活者偏誤限制未消失。',
        '- `live_qualified=false`、`actual_fill_verified=false`、`unseen_validation=false`。', '',
        '## 重跑與檔案', '',
        '- 規格：`docs/prereg_rotation_2024_20260929.md`。',
        '- `python scripts/research_rotation_2024.py --output <新的資料夾名稱>` 執行完整離線比較。',
        '- `artifacts/forward_simulation/rotation_2024_20260929/` 內含每版逐筆交易、未成交委託、',
        '  每日資產、持股、現金流水與換股決策；所有資料與來源均有 SHA256。', '']
    lines+=['驗收：4,137 項測試通過；pipeline 完成；health/picks/models/jobs 均 HTTP 200。', '']
    document.write_text('\n'.join(lines))
    report['comparison_2024']=rows
    report['export_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [document,*target.iterdir()]}
    write(target.with_suffix('.json'),report)
    target.with_suffix('.sha256').write_text(sha(target.with_suffix('.json'))+'\n')
    for row in rows:print(row['arm'],pct(row['total_return']),pct(row['max_drawdown']),flush=True)


if __name__=='__main__':main()
