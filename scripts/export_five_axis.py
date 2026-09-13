#!/usr/bin/env python3
"""Publish verified paired results and clearly separate incomplete data qualifications."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from app.file_lock import file_lock
from scripts.research_exit_scenarios import read,write,sha,encoded
from scripts.prepare_five_axis import OUTPUT
from scripts.research_five_axis import identity

TARGET=ROOT/'artifacts/forward_simulation/five_axis_delivery_20260913.json'
LABELS={'benchmark':'0050同條件基準','control':'原現金版','limit3':'限價等待3天','retest3':'回測突破位後進場',
    'capacity':'前20日成交金額排序','capacity_vol':'成交金額除以波動排序',
    'group_one':'訊號族群不重疊','slots5':'分散至5檔','staged':'先買一半再確認加碼',
    'joint':'限價＋成交金額排序＋族群限制',
    'revenue_covered':'營收同覆蓋對照','surprise':'營收超預期10%',
    'quality_covered':'營收及財報同覆蓋對照','surprise_quality':'營收超預期＋毛利改善及獲利為正'}


def export(output=OUTPUT,target=TARGET):
    if target.exists():raise ValueError('Use a new publication path')
    verification=read(output/'offline-verification.json')
    if (verification['all_identical'] is not True or verification['network_calls']!=0 or
        verification['identity_sha256']!=sha(output/'execution-identity.json') or
        verification['manifest_sha256']!=sha(output/'execution-manifest.json') or
        identity(output)!=read(output/'execution-identity.json')):
        raise ValueError('Offline validation is missing, incomplete, or stale')
    for name,value in read(output/'execution-manifest.json')['files_sha256'].items():
        if sha(output/name)!=value:raise ValueError('Changed execution evidence: '+name)
    summary=read(output/'summary.json');cases=summary['cases']
    if len(cases)!=38 or not summary['all_completed']:raise ValueError('All 38 cases must complete')
    def paired(arm,label=None):
        a,b=(cases[arm+'_'+stress]['summary'] for stress in ('control','combined'))
        benchmark=cases['benchmark_combined']['summary']
        return {'規則':label or LABELS[arm],'原條件總報酬':f"{a['total_return']*100:+.2f}%",
                '原條件最大回撤':f"{a['max_drawdown']*100:.2f}%",
                '壓力總報酬':f"{b['total_return']*100:+.2f}%",
                '壓力最大回撤':f"{b['max_drawdown']*100:.2f}%",
                '壓力期末資產（元）':round(b['final_nav'],2),
                '壓力相對0050百分點':round((b['total_return']-benchmark['total_return'])*100,2),
                '壓力成交筆數':b['trade_count'],'壓力交易成本':b['costs']['total_cost']}
    reconstruction=read(output/'rebuild/summary.json')
    facts=[]
    for lag in (15,30):
        for mode in ('revenue_covered','surprise','quality_covered','surprise_quality'):
            name=f'{mode}_{lag}'
            row=paired(name,LABELS[mode]+f'（營收+{lag}日／季報+{120 if lag==15 else 150}日假設）')
            row['入選候選']=cases[name+'_control']['candidate_count']
            if mode in ('surprise','surprise_quality'):
                counterpart=f'{"revenue_covered" if mode=="surprise" else "quality_covered"}_{lag}_combined'
                row['壓力相對同覆蓋百分點']=round((cases[name+'_combined']['summary']['total_return']-
                    cases[counterpart]['summary']['total_return'])*100,2)
            facts.append(row)
    unchanged={}
    for stress in ('control','combined'):
        a=read(output/f'cases/original_{stress}.json')['account']
        b=read(output/f'cases/control_{stress}.json')['account']
        unchanged[stress]={key:encoded(a[key])==encoded(b[key]) for key in
                            ('daily','trades','cash_ledger','holdings','corporate_actions')}
    forecast=read(output/'revenue-forecast.json')
    for name,value in {**forecast['input_sha256'],**forecast['code_sha256']}.items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen forecast source changed: '+name)
    companies=pd.read_parquet(ROOT/'.cache/million-replay-inputs/companies.parquet')
    names=dict(zip(companies.stock_id,companies.name))
    forecast['rows']=[dict(r,name=names.get(r['stock_id'],'')) for r in forecast['rows']]
    statuses=[
        {'項目':'1 資料與候選重建','已完成':'57個月份重新分群，458個候選不增不減；17個附帶證據更新',
         '尚缺':'50筆下市紀錄的上市區間與完整歷史母體／還原價資格'},
        {'項目':'2 進場等待','已完成':'限價3日、突破位回測，均配原條件與成交壓力','尚缺':'日級模型尚非實際盤中排隊成交'},
        {'項目':'3 候選排序','已完成':'成交金額、成交金額除以波動兩種排序配對','尚缺':'未見資料及前向實際成交'},
        {'項目':'4 資金配置','已完成':'族群重疊限制、5檔、分批加碼及組合配對','尚缺':'未見資料及前向實際成交'},
        {'項目':'5 營收與獲利','已完成':'280檔候選損益表、兩種時間假設及同覆蓋對照；273檔下期預估封存',
         '尚缺':'歷史精確公布時間與修訂版本，及未來實際公布結果'}]
    annual=[]
    for original,capacity,benchmark in zip(cases['control_combined']['summary']['annual'],
                                          cases['capacity_combined']['summary']['annual'],
                                          cases['benchmark_combined']['summary']['annual']):
        if len({row['year'] for row in (original,capacity,benchmark)})!=1:
            raise ValueError('Annual comparison dates differ')
        annual.append({'年份':capacity['year']+('（至9/9）' if capacity['partial_year'] else ''),
            '原現金版':f"{original['total_return']*100:+.2f}%",
            '成交金額排序':f"{capacity['total_return']*100:+.2f}%",
            '0050':f"{benchmark['total_return']*100:+.2f}%",
            '排序相對0050百分點':round((capacity['total_return']-benchmark['total_return'])*100,2)})
    result=dict(offline_identical=True,offline_seconds=verification['elapsed_seconds'],case_count=38,
        research_code_sha256={k:v for k,v in identity(output).items() if k.endswith('.py')},
        caption='2022/1/3–2026/9/9，本金100萬元複利、閒錢保留現金；基準3檔，5檔組另列。總報酬已扣成本。合併壓力：每邊0.90%滑價費用、零股對手報價與數量限制，進出場再延後1個市場日。',
        warning='歷史探索，尚未取得實盤資格。營收／財報結果使用公布延遲假設，不能稱為當時已知的真實公告回測。',
        comparison={'五項研究進度':statuses,
            '進場規則':[paired(k) for k in ('benchmark','control','limit3','retest3')],
            '候選排序':[paired(k) for k in ('benchmark','control','capacity','capacity_vol')],
            '資金配置與組合':[paired(k) for k in ('benchmark','control','group_one','slots5','staged','joint')],
            '營收與獲利（時間假設診斷）':facts,
            '逐年對照（合併成交壓力）':annual},
        cases=cases,reconstruction=reconstruction,unchanged_after_rebuild=unchanged,
        future_revenue_forecast=forecast,source_manifest_sha256=sha(output/'execution-manifest.json'),
        requests={'financial':summary['financial_requests'],**summary['execution_requests']},
        conclusion='成交金額排序值得繼續驗證，但超額報酬集中在2026年，2023至2025年逐年落後0050。38組都來自反覆使用的歷史，尚未切換正式策略。',
        remaining=['完整歷史股票母體、獨立還原價與價格修訂尚未全部核實。',
            '營收與財報尚缺歷史精確公布時刻；15/30日與120/150日是研究假設。',
            '成交壓力仍使用日級／零股日末證據；缺乏實際排隊與券商成交驗證。',
            '容量排序應接受固定規則的後續驗證，不能用本輪資料再次調參後宣稱樣本外成功。'],
        live_qualified=False,unseen_validation=False)
    write(target,result)
    return result


if __name__=='__main__':
    with file_lock(OUTPUT/'run.lock',timeout=0):r=export()
    print('Published',r['case_count'],'verified paired cases')
