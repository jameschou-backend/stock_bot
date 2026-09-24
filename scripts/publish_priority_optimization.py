#!/usr/bin/env python3
"""Publish verified research results without replaying, fetching, or promoting a strategy."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(board_output):
    evidence = {}

    def read(path):
        path = Path(path)
        evidence[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    causal = read(ROOT / 'artifacts/forward_simulation/current_causality_20260925.json')
    odd = read(ROOT / '.cache/oddlot_scope_20260925/summary.json')
    board = read(board_output / 'summary.json')
    replay = read(board_output / 'offline.json')
    if not replay['all_cases_identical'] or not board['parent_controls_identical']:
        raise ValueError('Board-only controls or offline replay failed')
    if replay['manifest_sha256'] != sha(board_output / 'manifest.json'):
        raise ValueError('Offline replay does not bind the current board-only manifest')
    for name, digest in read(board_output / 'manifest.json')['files_sha256'].items():
        if sha(board_output / name) != digest:
            raise ValueError('Board-only evidence changed: ' + name)
    for name, digest in read(board_output / 'identity.json').items():
        if sha(ROOT / name) != digest:
            raise ValueError('Board-only input or code changed: ' + name)
    odd_manifest = read(ROOT / '.cache/oddlot_scope_20260925/manifest.json')
    for name, digest in odd_manifest['input_files_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Odd-lot input or code changed: ' + name)
    for name, digest in odd_manifest['output_files_sha256'].items():
        if sha(ROOT / '.cache/oddlot_scope_20260925' / name) != digest:
            raise ValueError('Odd-lot demand evidence changed: ' + name)
    demand_csv = '.cache/oddlot_scope_20260925/union_stock_date_side.csv'
    evidence[demand_csv] = odd_manifest['output_files_sha256']['union_stock_date_side.csv']
    for name, digest in causal['source_and_code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Causality input or code changed: ' + name)
    cases = {name: {key: value for key, value in case.items()
                   if key in ('completed', 'config', 'summary', 'reason', 'execution', 'same_policy', 'mixed_reference')}
             for name, case in board['cases'].items()}
    return dict(schema='priority_optimization_v1', live_qualified=False, unseen_validation=False,
        evidence_sha256=evidence, causality=dict(passed=causal['passed'], complete=causal['complete'],
            accepted_candidates=causal['accepted_candidates'], case_count=len(causal['cases']),
            cutoff_count=len({case['cutoff'] for case in causal['cases']}), elapsed_seconds=causal['elapsed_seconds']),
        oddlot=odd['union'], oddlot_demand_csv=demand_csv, board_cases=cases,
        next_direction='優先研究可執行的委託與殘股處理，再評估選股優勢；未取得完整來源與跨期間勝出證據前，不採用舊版高報酬作實戰依據。',
        financial_publication_note='FinMind月營收create_time自2026/4/21起記錄入庫日期，並非正式公告時間；初始回填及較早歷史不能用它認證公告先後。',
        financial_publication_source='https://finmind.github.io/tutor/TaiwanMarket/Fundamental/#taiwanstockmonthrevenue',
        schedules_restarted=False, broker_orders_sent=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board-output', type=Path, default=ROOT / '.cache/board-only-20260925')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/forward_simulation/priority_optimization_20260925.json')
    args = parser.parse_args()
    if args.output.exists():
        sys.exit('Choose a new output; published results are immutable')
    result = build(args.board_output.resolve())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    args.output.with_suffix('.sha256').write_text(sha(args.output) + '\n')
    print(args.output)
