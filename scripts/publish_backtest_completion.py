#!/usr/bin/env python3
"""Publish a local, integrity-bound index after each research report is sealed."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha


def descriptor(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Report must remain inside this project')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def account_report(path):
    from skills.backtest_case_cache import file_identities
    path = Path(path).resolve()
    manifest = read(path.parent / 'manifest.json')
    for name, digest in manifest['files_sha256'].items():
        source = (path.parent / name).resolve()
        if not source.is_relative_to(path.parent) or sha(source) != digest:
            raise ValueError('Account report artifact changed: ' + name)
    if manifest['files_sha256'].get(path.name) != sha(path):
        raise ValueError('Report is absent from the account manifest')
    identity = read(path.parent / 'identity.json')
    refs = identity.get('source_sha256', identity)
    if file_identities([ROOT / name for name in refs], ROOT) != refs:
        raise ValueError('Account source/code closure changed since execution')
    value = read(path)
    if value.get('live_qualified') is not False or value.get('unseen_validation') is not False:
        raise ValueError('Only exploratory reports may be published')
    return value


def run(corporate, sector, data, output):
    from skills.backtest_data_evidence import verify_report
    corp, sec = account_report(corporate), account_report(sector)
    evidence = verify_report(data, ROOT)
    if evidence.get('live_qualified') is not False:
        raise ValueError('Execution evidence cannot qualify a strategy for live use')
    from app.backtest_completion_ui import validate_data_scope
    validate_data_scope(corp, sec, evidence)
    complete = sum(row['completed'] for row in corp['cases'].values())
    sector_complete = sum(row['status'] == 'completed_daily' for row in sec['case_rows'])
    coverage = evidence['coverage_totals']
    ordinary, odd = coverage['ordinary'], coverage['odd_lot']
    index = dict(format='backtest_completion_v1', live_qualified=False, unseen_validation=False,
        reports=dict(corporate=descriptor(corporate), sector=descriptor(sector), data=descriptor(data)),
        data_coverage=coverage,
        disposition=[
            dict(title='公司行動帳務', status=f'{complete}/{len(corp["cases"])} 組完成日資料帳戶',
                detail='新股按核實日期交付；未驗證的畸零淨款與付款日保持應收，不當作可用現金。'),
            dict(title='族群策略完整回測', status=f'{sector_complete}/{len(sec["case_rows"])} 組完成',
                detail='相對強勢、加成交升溫、0050；各測整張／零股與一般／加嚴成本，含現金和每日資產。'),
            dict(title='真實成交序列',
                status=f'普通盤待補{ordinary["missing_unique_sessions"]}股日；零股待補{odd["missing_unique_sessions"]}股日',
                detail=f'普通盤已有{ordinary["local_format_valid_unique_sessions"]}/{ordinary["required_unique_sessions"]}個必要股日的本機格式可用資料。'
                       '完整場次及委託可成交性仍需獨立核對，日量或格式樣本不替代證據。'),
            dict(title='歷史可得時間', status='尚未完整',
                detail='已補可核實的產業異動；目前族群快照及不完整公告修訂史仍阻擋嚴格PIT驗證。'),
        ], schedules_restarted=False, broker_orders_sent=False)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Completion index must be a project artifact')
    write(output, index)
    output.with_suffix('.sha256').write_text(sha(output) + '\n')
    from app.backtest_completion_ui import load
    load(output, ROOT)
    print(json.dumps(dict(corporate_completed=complete, sector_completed=sector_complete,
                         output=str(output.relative_to(ROOT))), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corporate', type=Path, required=True)
    parser.add_argument('--sector', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT/'artifacts/forward_simulation/backtest_data_completion_20260925.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/backtest_completion_20260925.json')
    args = parser.parse_args()
    run(args.corporate, args.sector, args.data, args.output)
