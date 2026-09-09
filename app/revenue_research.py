"""Read completed revenue evidence without running network/DB/backtest work."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = {'scripts/research_revenue.py', 'scripts/research_flow.py', 'skills/revenue_research.py',
        'skills/flow_research.py', 'skills/rule_research.py'}


def overview():
    path = ROOT/'.cache/revenue-research/report.summary.json'
    if not path.exists():
        return {'available': False, 'live_qualified': False, 'note': '尚無營收對照，先執行 make research-revenue'}
    try:
        report = json.loads(path.read_text())
        if (report['schema'] != 1 or report['experiment'] != 'revenue_20260909'
                or report['research_only'] is not True or report['live_qualified'] is not False):
            raise ValueError('Unsupported revenue report')
        if set(report['code_sha256']) != CODE:
            raise ValueError('Missing code provenance')
        for name, expected in report['code_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Research code changed')
        prereg = ROOT/'docs/prereg_revenue_20260909.md'
        if hashlib.sha256(prereg.read_bytes()).hexdigest() != report['preregistration_sha256']:
            raise ValueError('Preregistration changed')
        expected = {(rule, lag, scenario, market)
                    for rule in ('price', 'covered', 'growth', 'growth_trust', 'revenue_first')
                    for lag in (45, 60) for scenario in ('base', 'stress')
                    for market in (('ALL', 'TWSE', 'TPEX') if lag == 45 and scenario == 'stress' else ('ALL',))}
        rows = report['results']
        if len(rows) != 30 or {(r['rule'], r['lag_days'], r['scenario'], r['market']) for r in rows} != expected:
            raise ValueError('Incomplete contrasts')
        if report['revenue_inputs']['audit']['passed'] is not True:
            raise ValueError('Source audit failed')
        # Verify the small manifest. Large input hashes are checked by each CLI run.
        manifest = json.loads((ROOT/'.cache/revenue-research/inputs.json').read_text())
        if manifest != report['revenue_inputs']:
            raise ValueError('Input manifest changed')
        return {**report, 'available': True}
    except (OSError, ValueError, KeyError, TypeError):
        return {'available': False, 'live_qualified': False, 'note': '營收報告不完整或程式已變更，請執行 make research-revenue 重算'}
