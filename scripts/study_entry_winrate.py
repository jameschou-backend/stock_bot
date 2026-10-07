#!/usr/bin/env python3
"""Offline, bounded entry-only win-rate screen with all trials retained."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.entry_winrate import build_conditions, summarize_entry_winrate, summarize_candidates
from skills.trial_registry import append_trial_registry


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def run(output):
    started = time.perf_counter()
    output = Path(output).resolve()
    output.relative_to(ROOT / '.cache/entry-winrate-20261007')
    if output.exists():
        raise ValueError('Use a new output directory; prior trials must remain intact')
    source = ROOT / '.cache/rally-context-20261007/run-v2/report.json'
    expected = 'fc302ce2e30e057071b14bce4ef1659e1ad617a77b2359e4e0a6b78688c5ebfc'
    if digest(source) != expected:
        raise ValueError('Frozen context report changed')
    prior = json.loads(source.read_text())
    sources = dict(prior['source_sha256'])
    sources[str(source.relative_to(ROOT))] = expected
    for field in ('features', 'labelled_events'):
        sources[prior[field]['path']] = prior[field]['sha256']
    for path in (Path(__file__).resolve(), ROOT / 'skills/entry_winrate.py',
                 ROOT / 'docs/prereg_entry_winrate_20261007.md'):
        sources[str(path.relative_to(ROOT))] = digest(path)
    catalog = ROOT / '.cache/rally-precursors-20261006/run-v4/report.json'
    sources[str(catalog.relative_to(ROOT))] = 'f22ffe0ff6c6e809f1ee16355276e09d697a511729bb16ea45c2ba41c834e66d'
    for name, sha in sources.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Frozen source changed: ' + name)

    output.mkdir(parents=True)
    features = pd.read_parquet(ROOT / prior['features']['path'])
    conditions, definitions = build_conditions(features)
    if len(definitions) != 118 or definitions.get('all_entries') != []:
        raise ValueError('Unexpected predeclared condition registry')
    decisions_path = output / 'entry-conditions.parquet'
    conditions.to_parquet(decisions_path, index=False)
    decisions_sha = digest(decisions_path)
    # The entire decision matrix is sealed before reading any future outcomes.
    labels = pd.read_parquet(ROOT / prior['labelled_events']['path'])
    keys = ['cohort', 'event_id', 'stock_id', 'signal_date', 'signal_index']
    if labels.duplicated(keys + ['horizon']).any():
        raise ValueError('Duplicated labelled identity')
    if set(labels[keys].itertuples(index=False, name=None)) != set(
            features[keys].itertuples(index=False, name=None)):
        raise ValueError('Feature and outcome identities differ')
    if not labels.groupby(keys).horizon.agg(lambda values: set(values) == {20, 60}).all():
        raise ValueError('Every feature requires both fixed labelled horizons')
    labelled = labels.merge(conditions[keys + list(definitions)], on=keys,
                            how='left', validate='many_to_one')
    labelled_path = output / 'labelled-conditions.parquet'
    labelled.to_parquet(labelled_path, index=False)
    statistics = summarize_entry_winrate(labelled, definitions)
    candidates = summarize_candidates(statistics['records'])
    for name, sha in {**sources, str(decisions_path.relative_to(ROOT)): decisions_sha}.items():
        if digest(ROOT / name) != sha:
            raise ValueError('Source or sealed decisions changed during evaluation: ' + name)
    catalog_data = json.loads(catalog.read_text())
    report = dict(
        schema='entry_winrate_screen_v1', generated_at=datetime.now(timezone.utc).isoformat(),
        start=prior['start'], end=prior['end'], definitions=definitions,
        cost_model=prior['cost_model'], source_sha256=sources,
        files={p.name: dict(path=str(p.relative_to(ROOT)), sha256=digest(p))
               for p in (decisions_path, labelled_path)},
        statistics=statistics, candidates=candidates, hypothesis_count=117 * 2 * 2,
        feature_events=len(features), labelled_events=len(labelled),
        existing_catalog=dict(
            source=str(catalog.relative_to(ROOT)), source_sha256=digest(catalog),
            start=catalog_data['start'], end=catalog_data['end'],
            active_entry_rules=catalog_data['active_entry_rules'],
            research_prototypes=catalog_data['research_prototypes'],
            all_period_records=[r for r in catalog_data['summary'] if r['year'] == 'all'],
            interpretation='Previously evaluated raw first signals; no cooldown. Do not merge denominators.'),
        network_requests=0, finmind_requests=0, database_mutations=False, model_training=False,
        account_backtest=False, live_qualified=False, unseen_validation=False,
        multiple_testing_adjusted=False, candidate_selection_uses_historical_outcomes=True,
        elapsed_seconds=time.perf_counter() - started,
        limitations=[
            'Bounded exploratory selection among 468 hypotheses on already researched history.',
            'Sixty percent observed win rate is not a guarantee or probability for a future trade.',
            'Signals use next adjusted open and fixed horizon close proxies, not broker fills.',
            'No shared capital, position count, early exit, reinvestment or account drawdown.',
            'Each condition has its own evidence-known subset; unknowns are explicit.',
            'Price-correlated peers are not verified news themes; flow publication lag is assumed.',
            'Same-stock cooldown does not remove cross-stock or calendar correlation.',
            'All-period may include cross-year holdings; yearly rows require same-year endpoints.',
            'Historical market identity and first-publication archive remain incompletely certified.'])
    path = output / 'report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    report_sha = digest(path)
    path.with_suffix('.sha256').write_text(report_sha + '\n')
    for cohort in prior['cohorts']:
        for horizon in (20, 60):
            for condition, atoms in definitions.items():
                if condition == 'all_entries':
                    continue
                trial = dict(timestamp=report['generated_at'], source='entry_winrate_screen_20261007',
                    params=dict(cohort=cohort, horizon=horizon, condition=condition, atoms=atoms),
                    result_path=str(path.relative_to(ROOT)), report_sha256=report_sha,
                    account_backtest=False, live_qualified=False, unseen_validation=False)
                append_trial_registry(trial, registry_path=output / 'trials.jsonl')
                append_trial_registry(trial)
    print(json.dumps(dict(output=str(output), hypotheses=report['hypothesis_count'],
                         elapsed_seconds=report['elapsed_seconds'], finmind_requests=0)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
