"""Read sealed event diagnostics; never infer strategy validity from returns."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = {'scripts/prepare_event_groups.py', 'scripts/research_event_groups.py',
        'scripts/research_flow.py', 'skills/event_group_research.py', 'skills/news_radar.py',
        'skills/official_adj_factors.py', 'skills/rule_research.py', 'skills/flow_research.py'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overview():
    folder = ROOT/'.cache/event-group-research'
    path = folder/'report.summary.json'
    if not path.exists():
        return {'available': False, 'live_qualified': False, 'note': '尚無事件研究診斷，請先準備資料與訊號，再執行 make diagnose-event-groups'}
    try:
        report = json.loads(path.read_text())
        if (report['schema'] != 1 or report['experiment'] != 'event_groups_20260909'
                or report['research_only'] is not True or report['live_qualified'] is not False
                or report['diagnostic_only'] is not True or report['valid_strategy_evidence'] is not False):
            raise ValueError('Unsupported event diagnostic')
        if set(report['code_sha256']) != CODE:
            raise ValueError('Missing code provenance')
        for name, expected in report['code_sha256'].items():
            if sha(ROOT/name) != expected:
                raise ValueError('Research code changed')
        if sha(ROOT/'docs/prereg_event_groups_20260909.md') != report['preregistration_sha256']:
            raise ValueError('Preregistration changed')
        if sha(ROOT/'docs/event_news_source_audit_20260909.json') != report['source_audit_sha256']:
            raise ValueError('Source audit changed')
        if sha(folder/'signal-inputs.json') != report['signal_manifest_sha256']:
            raise ValueError('Signal manifest changed')
        expected = {(rule, horizon, scenario, basis, delay)
                    for rule in ('event', 'group', 'combined') for horizon in (63,126)
                    for scenario in ('base','stress') for basis in ('official','snapshot')
                    for delay in ((0,1) if basis == 'official' and scenario == 'stress' else (0,))}
        rows = report['results']
        if len(rows) != 30 or {(r['rule'],r['horizon'],r['scenario'],r['basis'],r['delay']) for r in rows} != expected:
            raise ValueError('Incomplete contrasts')
        audit = json.loads((ROOT/'docs/event_news_source_audit_20260909.json').read_text())
        if audit != report['source_audit'] or audit['passed'] is not False:
            raise ValueError('Source audit mismatch')
        return {**report, 'available': True}
    except (OSError, ValueError, KeyError, TypeError):
        return {'available': False, 'live_qualified': False, 'note': '事件診斷不完整或輸入／程式已變更，請重新準備並執行 make diagnose-event-groups'}
