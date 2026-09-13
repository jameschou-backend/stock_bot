#!/usr/bin/env python3
"""Publish verified paired diagnostics without assigning an intraday return."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_reservation_bridge import OUTPUT, CODE
from scripts.research_exit_scenarios import read, write, sha

TARGET = ROOT/'artifacts/forward_simulation/reservation_bridge_delivery_20260914.json'


def export(output=OUTPUT, target=TARGET):
    report, offline = read(output/'summary.json'), read(output/'offline.json')
    if not report['all_completed'] or len(report['cases']) != 8 or not offline['identical'] or offline['network_calls'] != 0:
        raise ValueError('Eight complete, identical paired accounts required')
    if offline.get('manifest_sha256') != sha(output/'manifest.json'):
        raise ValueError('Offline verification does not match current manifest')
    for name, digest in read(output/'identity.json').items():
        if sha(ROOT/name) != digest:
            raise ValueError('Bridge code or source changed: '+name)
    for name, digest in read(output/'manifest.json').items():
        if sha(output/name) != digest:
            raise ValueError('Bridge account or evidence changed: '+name)
    result = dict(report, offline=offline, code_sha256={name:sha(ROOT/name) for name in CODE},
        verdict='日資料假設下的勝出候選；原零股策略逐筆驗證尚未完成。',
        strict_intraday_completed=False, strict_intraday_return=None,
        warning='這輪保留零股，只比較現金與名額事前預留。成交仍用日資料假設，不能認定能實戰跑贏0050。',
        sources={
            'TWSE盤中零股歷史資料商品':'https://eshop.twse.com.tw/zh/product/detail/0000000080da7fa70182334eb932009d',
            'FinMind逐筆資料規格':'https://finmind.github.io/tutor/TaiwanMarket/Technical/',
            'TPEx揭示檔規格':'https://eshop.tpex.org.tw/zh/product/detail/2c92e0139984eab70199894054740008'})
    # The verdict above describes the legacy winner, never promotes a new rule.
    if target.exists() and read(target) != result:
        raise ValueError('Existing bridge publication is immutable')
    write(target, result)
    print(target)


if __name__ == '__main__':
    export()
