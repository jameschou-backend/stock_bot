#!/usr/bin/env python3
"""Publish only six independently audited, exactly reproduced account cases."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_intraday_limit import OUTPUT,CODE
from scripts.research_exit_scenarios import read,write,sha

TARGET=ROOT/'artifacts/forward_simulation/intraday_limit_delivery_20260914.json'


def export(output=OUTPUT,target=TARGET):
    verification=read(output/'offline-verification.json')
    report=read(output/'summary.json')
    if not verification['identical'] or verification['requests']!=0 or len(report['cases'])!=6:
        raise ValueError('Six identical offline cases required')
    identity=read(output/'identity.json')
    for name,digest in identity.items():
        path=ROOT/'.cache/five-axis-20260913/execution-manifest.json' if name=='parent_execution_manifest' else ROOT/name
        if sha(path)!=digest:
            raise ValueError('Research lineage changed: '+name)
    for name,digest in read(output/'manifest.json')['files_sha256'].items():
        if sha(output/name)!=digest:
            raise ValueError('Execution source changed: '+name)
    cases={}
    for name,item in report['cases'].items():
        path=output/'cases'/f'{name}.json'
        if sha(path)!=item['sha256'] or verification['case_sha256'][name]!=item['sha256']:
            raise ValueError('Case/offline verification mismatch')
        cases[name]=dict(item,path=str(path.relative_to(ROOT)))
    result=dict(report,cases=cases,offline=verification,
        code_sha256={name:sha(ROOT/name) for name in CODE},
        caption='2022/1/3–2026/9/9，100萬元複利；策略閒錢保留現金。整張逐筆成交診斷，並非原零股版本。',
        warning='這是整張診斷：配股零股無法驗證成交，會持續占用名額並改變選股路徑。原零股策略尚未完成此項驗證。',
        rules=['前日固定股票、限價、股數、現金及名額；當日成交結果不改買別檔。',
               '只計09:01之後、13:25之前穿過限價的逐筆量；同價位成交不算。',
               '普通條件：合格量及前20日均量各1%，另計每邊45bp成本緩衝；壓力：各0.5%、90bp。',
               '部分成交如實入帳；買單餘量當日取消；出場未成交股數隔日重掛。',
               '公司行動產生的零股保留估值，沒有零股逐筆就不偽造成交。'],
        limitations=['缺歷史零股逐筆與真實排隊證據；不能用本結果宣稱已可實戰。',
                     '歷史候選池、轉板身分、還原價格及逐筆完整性仍未全面獨立核准。',
                     '已研究區間，不是未見測試集；成本緩衝為假設，非券商實際滑價。',
                     '名額整天預留及整張限制會改變選股路徑，不能將與舊報酬的差額全部歸因於成交高估。'])
    if target.exists() and read(target)!=result:
        raise ValueError('Publication exists; use a new path')
    write(target,result)
    print(str(target))
    return result


if __name__=='__main__':export()
