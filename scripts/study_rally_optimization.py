#!/usr/bin/env python3
"""Frozen ranking and post-entry diagnostics; no network, training or orders."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.study_rally_context import digest
from skills.rally_checkpoint import build_checkpoints, CHECKPOINT_ACTIONS, CHECKPOINT_AGES
from skills.rally_ranking import build_rankings, RANKING_DEFINITIONS
from skills.rally_context_stats import summarize_filters
from skills.rally_optimization_stats import (rank_selections, ranking_day_pairs,
    build_exit_policies, summarize_exit_policies)
from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.engine import _prepare
from skills.trial_registry import append_trial_registry


def run(args):
    started = time.perf_counter()
    output = args.output.resolve();output.relative_to(ROOT)
    if output.exists():
        raise ValueError('Use a new output directory to preserve all trials')
    output.mkdir(parents=True)
    source = ROOT/'.cache/rally-context-20261007/run-v2/report.json'
    expected = 'fc302ce2e30e057071b14bce4ef1659e1ad617a77b2359e4e0a6b78688c5ebfc'
    if digest(source)!=expected:
        raise ValueError('Frozen context report changed')
    previous = json.loads(source.read_text())
    sources = dict(previous['source_sha256'])
    sources[str(source.relative_to(ROOT))] = expected
    for key in ['features','labelled_events']:
        ref=previous[key];sources[ref['path']]=ref['sha256']
    for p in [Path(__file__).resolve(), ROOT/'skills/rally_ranking.py', ROOT/'skills/rally_checkpoint.py',
              ROOT/'skills/rally_optimization_stats.py',ROOT/'docs/prereg_rally_optimization_20261007.md']:
        sources[str(p.relative_to(ROOT))]=digest(p)
    for path,sha in sources.items():
        if digest(ROOT/path)!=sha:
            raise ValueError('Source hash changed: '+path)
    features = pd.read_parquet(ROOT/previous['features']['path'])
    # Ranking is finalized and persisted before reading future labels.
    rankings, conditions = rank_selections(build_rankings(features))
    rankings.to_parquet(output/'rankings.parquet',index=False)
    data = load_bundle(ROOT/previous['source_provenance']['bundle'], previous['start'], previous['end'])
    f,days,ids = _prepare(data['bars'],data['calendar'],pd.Timestamp(previous['end']))
    for row in features.itertuples(index=False):
        if str(days[row.signal_index].date())!=row.signal_date or ids[row.column_index]!=row.stock_id:
            raise ValueError('Feature coordinates differ from reloaded scanner matrices')
    checkpoints = build_checkpoints(features,f,days,ids)
    checkpoints.to_parquet(output/'checkpoints.parquet',index=False)
    decision_sha = {str((output/name).relative_to(ROOT)):digest(output/name)
        for name in ['rankings.parquet','checkpoints.parquet']}
    # Future labels are accessed only after both decision tables are saved.
    labelled = pd.read_parquet(ROOT/previous['labelled_events']['path'])
    extra = [c for c in rankings if c not in features]+['cohort','event_id']
    ranked_labels = labelled.merge(rankings[extra],on=['cohort','event_id'],how='left',validate='many_to_one')
    ranked_labels.to_parquet(output/'ranked-labels.parquet',index=False)
    ranking_results = summarize_filters(ranked_labels,conditions)
    day_pairs = ranking_day_pairs(ranked_labels)
    policies = build_exit_policies(labelled,checkpoints,f,days,ids)
    policies.to_parquet(output/'exit-policy-events.parquet',index=False)
    exit_results = summarize_exit_policies(policies,labelled)
    for path,sha in {**sources,**decision_sha}.items():
        if digest(ROOT/path)!=sha:
            raise ValueError('Source or frozen decisions changed while evaluating: '+path)
    files = {name:dict(path=str((output/name).relative_to(ROOT)),sha256=digest(output/name)) for name in
        ['rankings.parquet','checkpoints.parquet','ranked-labels.parquet','exit-policy-events.parquet']}
    report = dict(schema='rally_optimization_study_v1',generated_at=datetime.now(timezone.utc).isoformat(),
        start=previous['start'],end=previous['end'],source_sha256=sources,files=files,
        cost_model=previous['cost_model'],source_provenance=data['provenance'],
        ranking_definitions=RANKING_DEFINITIONS,ranking_results=ranking_results,
        ranking_day_pairs=day_pairs,exit_results=exit_results,
        feature_events=len(features),checkpoint_rows=len(checkpoints),policy_rows=len(policies),
        ranking_coverage=rankings.groupby('cohort').ranking_known.agg(['size','sum']).rename(
            columns={'size':'events','sum':'context_known'}).astype(int).to_dict('index'),
        checkpoint_issues={str(age):group.checkpoint_issue.fillna('known').value_counts().to_dict()
            for age,group in checkpoints.groupby('checkpoint_age')},
        elapsed_seconds=time.perf_counter()-started,finmind_requests=0,network_requests=0,
        database_mutations=False,model_training=False,account_backtest=False,
        live_qualified=False,unseen_validation=False,multiple_testing_adjusted=False,
        ranking_hypotheses=len(conditions)*2*2,exit_hypotheses=2*2*len(CHECKPOINT_AGES)*len(CHECKPOINT_ACTIONS),
        limitations=['Previously researched history; no new unseen evidence.',
            'Signal-level equal-unit outcomes are not a shared cash account or capacity guarantee.',
            'All ranking variants compare only the same common-known external-context subset.',
            'Future-incomplete selected ranks never promote a lower-ranked candidate.',
            'Same-date fully observed group comparisons are ex-post diagnostics, not tradable selection.',
            'Day3/day5 rules are separate one-time policies, not combined into earliest exit.',
            'Early exits remain cash to the original horizon; no reinvestment or reentry.',
            'Unknown checkpoint or original endpoint remains unknown, not an avoided loss.',
            'Flow publication lag assumed; historical universe and news archive incomplete.'])
    path = output/'report.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    report_sha=digest(path);path.with_suffix('.sha256').write_text(report_sha+'\n')
    for cohort in previous['cohorts']:
        for horizon in [20,60]:
            configurations = [dict(type='rank_selection',condition=c) for c in conditions]
            configurations += [dict(type='checkpoint_exit',age=age,condition=c)
                for age in CHECKPOINT_AGES for c in CHECKPOINT_ACTIONS]
            for config in configurations:
                append_trial_registry(dict(timestamp=report['generated_at'],source='rally_optimization_20261007',
                    params=dict(cohort=cohort,horizon=horizon,**config),result_path=str(path.relative_to(ROOT)),
                    report_sha256=report_sha,account_backtest=False,live_qualified=False,unseen_validation=False))
    print(json.dumps(dict(output=str(output),ranking_hypotheses=report['ranking_hypotheses'],
        exit_hypotheses=report['exit_hypotheses'],elapsed_seconds=report['elapsed_seconds'],finmind_requests=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
