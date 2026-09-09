"""Small sealed pilot report; reading the workbench never starts research or HTTP."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = {'scripts/research_guidance.py', 'skills/guidance_signals.py', 'skills/guidance_research.py'}
SOURCES = {'docs/guidance_quarterly_sources_20260910.json', 'docs/guidance_annual_sources_20260910.json'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overview():
    path = ROOT/'.cache/guidance-research/report.summary.json'
    missing = {'available':False, 'live_qualified':False,
               'note':'尚無完整的官方指引試驗；先執行 make prepare-guidance，再執行 make research-guidance。'}
    if not path.exists():
        return missing
    try:
        report = json.loads(path.read_text())
        if (report['schema'] != 1 or report['experiment'] != 'guidance_20260910'
                or report['research_only'] is not True or report['live_qualified'] is not False
                or report['valid_strategy_evidence'] is not False):
            raise ValueError('Unsupported pilot')
        for key, expected in (('code_sha256', CODE), ('source_sha256', SOURCES)):
            if set(report[key]) != expected:
                raise ValueError('Incomplete provenance')
            for name, fingerprint in report[key].items():
                if sha(ROOT/name) != fingerprint:
                    raise ValueError('Changed research source or code')
        if sha(ROOT/'docs/prereg_guidance_20260910.md') != report['preregistration_sha256']:
            raise ValueError('Changed protocol')
        if sha(ROOT/'.cache/guidance-research/inputs.json') != report['input_manifest_sha256']:
            raise ValueError('Changed price manifest')
        manifest = json.loads((ROOT/'.cache/guidance-research/inputs.json').read_text())
        if report['inputs'] != manifest or set(manifest['files_sha256']) != {
                'close-official.parquet','close-snapshot.parquet','trade-flags.parquet','raw.parquet'}:
            raise ValueError('Incomplete price provenance')
        for name, fingerprint in manifest['files_sha256'].items():
            if sha(ROOT/'.cache/guidance-research'/name) != fingerprint:
                raise ValueError('Changed frozen price slice')
        expected = {(r,b,c,d) for r in ('beat','beat_confirm','guidance_up','combined')
                    for b in ('official','snapshot') for c in ('base','stress')
                    for d in ((0,1) if (b,c)==('official','stress') else (0,))}
        if (len(report['results']) != 20
                or {(r['rule'],r['basis'],r['scenario'],r['delay']) for r in report['results']} != expected):
            raise ValueError('Incomplete contrasts')
        for row in report['results']:
            if row['timing_diagnostic'] != (row['rule'] in ('guidance_up','combined')):
                raise ValueError('Missing timing warning')
        baseline_keys = {(b,c,m) for b in ('official','snapshot') for c in ('base','stress') for m in ('benchmark','static_mix')}
        if len(report['baselines']) != 8 or {(r['basis'],r['scenario'],r['mode']) for r in report['baselines']} != baseline_keys:
            raise ValueError('Incomplete baseline controls')
        if set(report['events']) != {'official','snapshot'} or any(len(v)!=13 for v in report['events'].values()):
            raise ValueError('Incomplete consecutive source coverage')
        return {**report, 'available':True}
    except (OSError, ValueError, KeyError, TypeError):
        return {**missing, 'note':'官方指引試驗不完整，或來源／程式已變更；請重新執行 make research-guidance。'}
