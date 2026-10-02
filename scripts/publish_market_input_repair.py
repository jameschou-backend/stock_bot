#!/usr/bin/env python3
"""Seal fixed-strategy input repair and repeated research/strict replay evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.ordinary_volume_bundle import bind, digest


def load_capacity_conflicts(root, path, refs):
    path = bind(root, refs, str(path), digest(root/path))
    value = json.loads(path.read_text())
    parent = bind(root, refs, value['source_report'], value['source_sha256'])
    evidence = json.loads(parent.read_text())
    if (value['schema'] != 'ordinary_capacity_conflicts_v1'
            or evidence['schema'] != 'ordinary_capacity_evidence_v1'):
        raise ValueError('Unsupported ordinary-capacity evidence')
    for field in ('source_sha256', 'output_sha256', 'code_sha256'):
        for name, expected in evidence[field].items():
            bind(root, refs, name, expected)
    conflicts = [r for r in evidence['rows'] if r['fill_check'] == 'capacity_conflict']
    if (value['rows'] != conflicts or value['count'] != len(conflicts)
            or evidence['counts']['capacity_conflict'] != len(conflicts)):
        raise ValueError('Original ordinary-capacity conflicts differ from their source report')
    return value


def verify_run_scopes(reports, bundle):
    required = {'three_black', 'benchmark'}
    scope = None
    for report in reports:
        if report['input_bundle'] != str(bundle) or report['data_revision'] != 'repaired' or report['preparation']:
            raise ValueError('Report is not an offline replay of the repaired bundle')
        if set(report['cases']) != required:
            raise ValueError('Both strategy and benchmark arms are required')
        current = (report['start'], report['end'], report['initial_cash'])
        if scope is None: scope = current
        if current != scope or report['initial_cash'] != 1_000_000:
            raise ValueError('Replay periods or initial cash differ')
        if (any(type(c['completed']) is not bool for c in report['cases'].values())
                or report['all_completed'] is not all(c['completed'] for c in report['cases'].values())):
            raise ValueError('Replay completion summary differs from both cases')
        if report['live_qualified'] or report['actual_fill_verified']:
            raise ValueError('Research result was incorrectly promoted')


def publish(bundle, research, repeat, strict, destination):
    refs = {}
    def load(path, *, closure=True):
        path = (ROOT/path).resolve()
        bind(ROOT, refs, str(path.relative_to(ROOT)), digest(path))
        value = json.loads(path.read_text())
        if closure:
            for name, expected in value.get('source_sha256', {}).items():
                bind(ROOT, refs, name, expected)
        return value
    manifest = load(bundle/'manifest.json')
    if not manifest['baseline_reproduced'] or manifest['frozen_inputs_changed']:
        raise ValueError('The frozen candidate baseline was not preserved')
    for name, expected in manifest['files_sha256'].items():
        path = bundle/name
        bind(ROOT, refs, str(path), expected)
    control = load(Path('.cache/market-input-repair-20261002/frozen-control-c/report.json'), closure=False)
    # The control finished before the subsequent strict-only resolver repair.
    # Preserve its exact script bytes; do not rewrite the old report or pretend
    # it ran today's version. All its other sources must still match in place.
    control_versions = {'scripts/replay_repaired_market_inputs.py':
                        '.cache/market-input-repair-20261002/frozen-control-c/replay_source.py'}
    for name, expected in control['source_sha256'].items():
        bind(ROOT, refs, control_versions.get(name, name), expected)
    if control['data_revision'] != 'frozen_control' or not control['all_completed'] or control['preparation']:
        raise ValueError('Exact frozen control account reproduction is missing')
    if set(control['cases']) != {'three_black', 'benchmark'}:
        raise ValueError('Frozen control is missing a comparison arm')
    for case in control['cases'].values():
        path = bind(ROOT, refs, case['path'], case['sha256'])
        if case['completed'] is not True or json.loads(path.read_text())['completed'] is not True:
            raise ValueError('A frozen control account is incomplete')
    first, second, strict_report = (load(p/'report.json') for p in (research, repeat, strict))
    verify_run_scopes((first, second, strict_report), bundle)
    if first['volume_policy'] != 'legacy_total_research' or second['volume_policy'] != first['volume_policy']:
        raise ValueError('Repeated research policies differ')
    if strict_report['volume_policy'] != 'strict':
        raise ValueError('Independent ordinary-volume check was not attempted')
    cases, repeated = {}, True
    for arm in ('three_black', 'benchmark'):
        a = load(Path(first['cases'][arm]['path']), closure=False)
        b = load(Path(second['cases'][arm]['path']), closure=False)
        if digest(ROOT/first['cases'][arm]['path']) != first['cases'][arm]['sha256']:
            raise ValueError('Research account differs from its run manifest')
        if digest(ROOT/second['cases'][arm]['path']) != second['cases'][arm]['sha256']:
            raise ValueError('Repeated account differs from its run manifest')
        if not a['completed'] or not b['completed'] or a['account'] != b['account'] or a['summary'] != b['summary']:
            repeated = False
        cases[arm] = dict(completed=a['completed'], summary=a['summary'] if a['completed'] else None,
                          path=first['cases'][arm]['path'], reason=a.get('reason'))
    if first['all_completed'] and not repeated:
        raise ValueError('Repeated offline account replay changed')
    strict_cases, blocked = {}, 0
    for arm, item in strict_report['cases'].items():
        value = load(Path(item['path']), closure=False)
        if digest(ROOT/item['path']) != item['sha256']:
            raise ValueError('Strict account differs from its run manifest')
        count = (value.get('ordinary_evidence_blocked_orders', 0) if value['completed']
                 else len(value.get('ordinary_evidence_blocks', [])))
        blocked += count
        strict_cases[arm] = dict(completed=value['completed'], reason=value.get('reason'),
            completed_sessions=value.get('completed_sessions'), last_date=value.get('last_date'),
            blocked_board_orders=count, ordinary_capacity_complete=value.get('ordinary_capacity_complete', False),
            path=item['path'])
    capacity_conflicts = load_capacity_conflicts(ROOT,
        Path('.cache/ordinary-volume-evidence-20261002/diagnostic-v2/capacity-conflicts.json'), refs)
    diff = manifest['median50m_candidate_diff']
    capacity_complete = strict_report['all_completed'] and blocked == 0 and all(
        c['ordinary_capacity_complete'] for c in strict_cases.values())
    result = dict(schema='market_input_repair_replay_v1', created_at=datetime.now(timezone.utc).isoformat(),
        start=first['start'], end=first['end'], initial_cash=1_000_000,
        input_bundle=str(bundle), repairs=dict(quotes_added=manifest['quote_repairs'],
            independently_aligned=manifest['directly_aligned_adjusted_repairs'],
            derived_adjusted=manifest['derived_adjusted_repairs'],
            original_candidates=diff['original_candidate_count'], repaired_candidates=diff['repaired_candidate_count'],
            added_candidates=len(diff['added_candidates']), removed_candidates=len(diff['removed_candidates']),
            unresolved_quote_rows=manifest['unresolved_quote_rows'],
            unresolved_quote_rows_in_strategy_scope=manifest['unresolved_quote_rows_in_strategy_scope'],
            candidate_prefix_checks=manifest['candidate_prefix_checks'],
            candidate_prefix_exact=manifest['candidate_prefix_exact'], roster_issue_count=manifest['roster_issue_count']),
        research=dict(completed=first['all_completed'], repeat_identical=repeated,
                      volume_policy=first['volume_policy'], cases=cases),
        strict=dict(completed=strict_report['all_completed'], capacity_complete=capacity_complete,
                    blocked_board_orders=blocked, cases=strict_cases),
        limitations=[
            '買賣價仍為各渠道日高低中點的研究假設，沒有聲稱當時一定能成交。',
            f"研究對照仍用全日量估算普通盤容量；原帳戶已有{capacity_conflicts['count']}筆成交超過核對後的普通盤1%容量，不能作實戰績效證明。",
            '嚴格模式將缺少普通盤當日或前20日證據的委託列為資料阻擋，保留現金與持股；不得把這種跳過資料的帳戶當成完整策略驗證。',
            '完整歷史可交易股票名單與未見期間績效尚未認證；市場日表和已觀測名單一致並不等於兩項認證通過。'],
        return_recomputed=first['all_completed'] and repeated,
        complete_verified_data=False, live_qualified=False, actual_fill_verified=False, unseen_validation=False,
        control_source_versions=control_versions)
    for path in (Path(__file__), ROOT/'skills/ordinary_volume_bundle.py'):
        bind(ROOT, refs, str(path.relative_to(ROOT)), digest(path))
    result['source_sha256'] = refs
    destination = (ROOT/destination).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError('Published report must remain inside the repository')
    if destination.exists():
        raise ValueError('Refuse to replace a published repair report')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    destination.with_suffix('.sha256').write_text(digest(destination)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-bundle', type=Path, required=True)
    parser.add_argument('--research-run', type=Path, required=True)
    parser.add_argument('--repeat-run', type=Path, required=True)
    parser.add_argument('--strict-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/forward_simulation/market_input_repair_20261002.json'))
    args = parser.parse_args()
    result = publish(args.input_bundle, args.research_run, args.repeat_run, args.strict_run, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))
