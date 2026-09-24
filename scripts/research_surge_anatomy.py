#!/usr/bin/env python3
"""Study preregistered sudden-gainer patterns and same-date counterexamples offline."""
from pathlib import Path
import argparse
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from app.file_lock import file_lock
from scripts.audit_current_causality_20260925 import provenance, matrices, BASE, ORIGINAL, verify_hashes
from scripts.research_exit_scenarios import write, sha
from skills.verified_backtest_tool import offline_only
from skills.surge_anatomy import cohort_table, causal_check, RULES, NUMERIC_FEATURES
from skills.surge_statistics import rule_statistics, matched_controls

SPEC = ROOT / 'docs/prereg_surge_anatomy_20260925.md'
OUTPUT = ROOT / '.cache/surge-anatomy-20260925'


def clean(value):
    if value is pd.NA: return None
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def run(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Keep prior results; select a new immutable output directory')
    if not output.is_relative_to(ROOT / '.cache'): raise ValueError('Output must be inside the project cache')
    tick = time.perf_counter()
    with file_lock(ROOT / '.cache/surge-anatomy.lock', timeout=0), offline_only():
        expected = provenance()
        for path in (SPEC, Path(__file__), ROOT / 'skills/surge_anatomy.py', ROOT / 'skills/surge_statistics.py',
                     ROOT / 'tests/test_surge_anatomy.py', ROOT / 'tests/test_surge_statistics.py'):
            expected[path] = sha(path)
        close, quality, raw, volume = matrices(BASE)
        companies = pd.read_parquet(ORIGINAL / 'companies.parquet')
        table, coverage = cohort_table(close, quality, raw, volume, companies)
        # Calendar-selected boundaries, not selected by which stocks later rose.
        cutoffs = [str(close.index[close.index <= pd.Timestamp(day)][-1].date())
                   for day in ('2022-12-31', '2024-12-31', '2026-06-30')]
        checks = causal_check(close, raw, volume, companies, cutoffs)
        statistics = rule_statistics(table, RULES)
        pairs, matched = matched_controls(table, NUMERIC_FEATURES)
        known = table[table.event.notna() & table.phase.ne('boundary')]
        events = known[known.event.eq(True)]
        fields = ['stock_id', 'name', 'signal_date', 'entry_date', 'exit_date', 'phase',
                  'forward_return', 'excess_return', *NUMERIC_FEATURES, *RULES[1:]]
        examples = events.sort_values(['forward_return', 'stock_id'], ascending=[False, True]).drop_duplicates('stock_id').head(12)
        false_positives = known[known.A_breakout_volume_strength & known.event.eq(False)].sort_values('forward_return').head(8)
        missed = events[events.A_breakout_volume_strength.eq(False)].sort_values('forward_return', ascending=False).drop_duplicates('stock_id').head(8)
        output.mkdir(parents=True)
        files = {}
        for name, frame in (('observations', table), ('anchor_coverage', coverage), ('matched_controls', pairs),
                            ('rule_statistics', pd.DataFrame(statistics)), ('surge_examples', examples[fields]),
                            ('false_positive_examples', false_positives[fields]), ('missed_examples', missed[fields])):
            path = output / f'{name}.csv'
            frame.to_csv(path, index=False, encoding='utf-8-sig')
            files[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(frame))
        verify_hashes(expected)
        report = clean(dict(format='surge_anatomy_v1', completed=True, live_qualified=False, unseen_validation=False,
            portfolio_returns_computed=False, strategy_net_return=None,
            source_start=str(close.index[0].date()), source_end=str(close.index[-1].date()),
            first_signal=table.signal_date.min(), last_signal=table.signal_date.max(), last_exit=table.exit_date.max(),
            companies=len(companies), eligible_observations=len(table), observed_labels=int(table.event.notna().sum()),
            unknown_labels=int(table.event.isna().sum()), boundary_observations=int(table.phase.eq('boundary').sum()),
            unknown_label_groups=table.loc[table.event.isna()].groupby(
                ['phase', 'signal_date', 'label_reason']).size().reset_index(name='count').to_dict('records'),
            surge_events=len(events), distinct_surge_stocks=int(events.stock_id.nunique()),
            rule_statistics=statistics, matched_controls=matched, examples=examples[fields].to_dict('records'),
            false_positive_examples=false_positives[fields].to_dict('records'), missed_examples=missed[fields].to_dict('records'),
            causality_checks=checks, artifacts=files, source_sha256={str(p.relative_to(ROOT)): d for p, d in expected.items()},
            elapsed_seconds=round(time.perf_counter()-tick, 3), finmind_requests=0, network_calls=0, database_writes=0,
            institutional_included=False, fundamentals_included=False, news_included=False,
            limitations=[
                'Fixed current-company cohort, incomplete historical universe/listing identity and current industry classifications.',
                'Fixed 21-session anchors do not enumerate every rally onset; prices are adjustment-based proxies, not verified total return.',
                'No orders, transaction costs, liquidity capacity or capital allocation were simulated; event rates are not account returns.',
                'Entire interval previously researched; temporal replication is not an untouched test.',
                'Future missing/anomalous/disputed prices create unresolved labels, not failed predictions.',
                'Institutional full-cohort snapshot has source discrepancies; verified subset is selected by prior signals and is not a valid broad cohort.',
                'Historical financial and news publication/revision timestamps are not fully validated; not used as leading signals.',
                'Matched controls use replacement and date bootstrap retains sector/stock dependence across dates; findings are exploratory.',
            ]))
        write(output / 'report.json', report)
        write(output / 'manifest.json', dict(report_sha256=sha(output / 'report.json'), files=files,
                                           source_sha256=report['source_sha256']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({k: result[k] for k in ('eligible_observations', 'unknown_labels', 'surge_events',
                      'distinct_surge_stocks', 'elapsed_seconds', 'finmind_requests')}, ensure_ascii=False))
