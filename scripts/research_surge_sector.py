#!/usr/bin/env python3
"""Offline, retrospective sector confirmation on the sealed surge observations."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np

from app.file_lock import file_lock
from scripts.audit_current_causality_20260925 import matrices, BASE, ORIGINAL, verify_hashes
from scripts.research_exit_scenarios import sha, write
from scripts.research_surge_anatomy import clean
from skills.surge_anatomy import features
from skills.surge_sector import sector_features
from skills.surge_statistics import rule_statistics
from skills.verified_backtest_tool import offline_only

PREVIOUS = ROOT / 'artifacts/surge_anatomy_20260925.json'
MEMBERS = ROOT / '.cache/chain-flow-research/members.parquet'
SPEC = ROOT / 'docs/prereg_surge_sector_20260925.md'
OUTPUT = ROOT / '.cache/surge-sector-20260925'
FLAGS = ['breadth_confirmed', 'turnover_confirmed', 'sector_confirmed']
RULES = ['all_eligible', 'relative_strength', 'A_breakout_volume_strength',
         'strength_with_breadth', 'strength_with_turnover', 'strength_with_sector']


def nullable_any(values):
    """Unknown groups cannot be silently interpreted as failed groups."""
    if values.eq(True).any():
        return True
    return pd.NA if values.isna().any() else False


def attach_groups(observations, sectors):
    """OR already-combined sector results; never combine two different groups."""
    keys = ['signal_date', 'stock_id']
    if sectors.duplicated([*keys, 'industry']).any():
        raise ValueError('Duplicate stock/date/industry result')
    groups = sectors.groupby(keys, sort=True)
    aggregated = groups[FLAGS].agg(nullable_any).astype('boolean')
    aggregated['group_count'] = groups.size()
    aggregated['confirmed_groups'] = groups.sector_confirmed.apply(
        lambda values: int(values.fillna(False).sum()))
    table = observations.merge(aggregated.reset_index(), on=keys, how='left', validate='one_to_one')
    for column in FLAGS:
        table[column] = table[column].astype('boolean')
    table['has_membership'] = table.group_count.notna()
    table['common_peer_observation'] = table.has_membership & table[FLAGS].notna().all(axis=1)
    for flag, target in zip(FLAGS, RULES[-3:]):
        table[target] = table.relative_strength.astype('boolean') & table[flag]
    return table


def coverage(table):
    rows = []
    for phase in ('discovery', 'replication', 'boundary'):
        part = table[table.phase.eq(phase)]
        rows.append(dict(phase=phase, observations=len(part),
                         unmapped=int((~part.has_membership).sum()),
                         peer_unknown=int((part.has_membership & ~part.common_peer_observation).sum()),
                         common_observations=int(part.common_peer_observation.sum()),
                         common_unknown_labels=int((part.common_peer_observation & part.event.isna()).sum())))
    return rows


def incremental_comparisons(table):
    """Same-observation, same-date bootstrap of added filters versus strength."""
    results = []
    for phase in ('discovery', 'replication'):
        rows = table[table.phase.eq(phase)]
        for rule in RULES[-3:]:
            paired = rows[rows[['event', 'relative_strength', rule]].notna().all(axis=1)]
            base = paired.relative_strength.to_numpy(dtype=bool)
            new = paired[rule].to_numpy(dtype=bool)
            event = paired.event.to_numpy(dtype=bool)
            if (new & ~base).any():
                raise ValueError('An added confirmation must remain a subset of relative strength')
            dates, codes = np.unique(paired.signal_date.to_numpy(), return_inverse=True)
            blocks = np.column_stack([
                np.bincount(codes, weights=mask.astype(float), minlength=len(dates))
                for mask in (base & event, base, new & event, new)])
            base_rate = float((base & event).sum() / base.sum()) if base.any() else None
            new_rate = float((new & event).sum() / new.sum()) if new.any() else None
            item = dict(phase=phase, rule=rule, paired_known_observations=len(paired),
                        excluded_unknown=int(len(rows)-len(paired)),
                        baseline_precision=base_rate, new_precision=new_rate,
                        precision_difference=None if base_rate is None or new_rate is None else new_rate-base_rate,
                        difference_ci_low=None, difference_ci_high=None, bootstrap_unit='signal_date',
                        bootstrap_seed=20260925, bootstrap_replicates=1000, bootstrap_valid_replicates=0)
            if len(dates) >= 5:
                weights = np.random.default_rng(20260925).multinomial(
                    len(dates), np.full(len(dates), 1/len(dates)), size=1000)
                base_tp, base_n, new_tp, new_n = (weights @ blocks).T
                valid = (base_n > 0) & (new_n > 0)
                differences = new_tp[valid]/new_n[valid] - base_tp[valid]/base_n[valid]
                item['bootstrap_valid_replicates'] = int(valid.sum())
                if valid.sum() >= 950:
                    item['difference_ci_low'], item['difference_ci_high'] = map(
                        float, np.quantile(differences, [.025, .975]))
            results.append(item)
    return results


def causal_checks(close, raw, volume, companies, members, baseline):
    # Fixed calendar cut, not chosen from which stocks subsequently rose.
    cutoff = pd.Timestamp('2024-12-31')
    dates = sorted(d for d in baseline.signal_date.unique() if d <= str(cutoff.date()))
    expected = baseline[baseline.signal_date.isin(dates)].reset_index(drop=True)
    checks = []
    for mode in ('truncate', 'mutate'):
        frames = []
        for frame in (close, raw, volume):
            changed = frame.loc[:cutoff].copy() if mode == 'truncate' else frame.copy()
            if mode == 'mutate':
                changed.loc[changed.index > cutoff] *= 7
            frames.append(changed)
        actual = sector_features(*frames, companies, members, dates)
        pd.testing.assert_frame_equal(expected, actual.reset_index(drop=True))
        checks.append(dict(cutoff=str(cutoff.date()), mode=mode, passed=True))
    return checks


def run(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(ROOT / '.cache'):
        raise ValueError('Select a new output directory inside the project cache; never overwrite prior evidence')
    started = time.perf_counter()
    with file_lock(ROOT / '.cache/surge-sector.lock', timeout=0), offline_only():
        previous = json.loads(PREVIOUS.read_text())
        expected = {ROOT / p: value for p, value in previous['source_sha256'].items()}
        observation = previous['artifacts']['observations']
        expected[ROOT / observation['path']] = observation['sha256']
        own = [PREVIOUS, MEMBERS, MEMBERS.with_suffix('.meta.json'), SPEC, Path(__file__),
               ROOT / 'skills/surge_sector.py', ROOT / 'tests/test_surge_sector.py',
               ROOT / 'tests/test_surge_sector_driver.py']
        expected.update({p: sha(p) for p in own})
        verify_hashes(expected)
        close, _, raw, volume = matrices(BASE)
        companies = pd.read_parquet(ORIGINAL / 'companies.parquet')
        provider_members = pd.read_parquet(MEMBERS)
        excluded_ids = sorted(set(provider_members.stock_id) - set(companies.stock_id))
        members = provider_members[provider_members.stock_id.isin(companies.stock_id)].copy()
        observations = pd.read_csv(ROOT / observation['path'], dtype={'stock_id': str})
        dates = sorted(observations.signal_date.unique())
        sectors = sector_features(close, raw, volume, companies, members, dates)
        table = attach_groups(observations, sectors)
        common = table[table.common_peer_observation].copy()
        passive_sectors = sectors[sectors.industry.eq('被動元件')]
        passive_ids = set(members.loc[members.industry.eq('被動元件'), 'stock_id'])
        passive = attach_groups(observations[observations.stock_id.isin(passive_ids)], passive_sectors)
        passive_common = passive[passive.common_peer_observation].copy()
        statistics = {}
        for name, part in [('all_observations', table), ('common_observations', common),
                           ('passive_common_observations', passive_common)]:
            statistics[name] = rule_statistics(part, RULES)
        checks = causal_checks(close, raw, volume, companies, members, sectors)
        case_dates = close.index[(close.index >= '2026-04-02') & (close.index <= '2026-05-11')]
        case = sector_features(close, raw, volume, companies,
                               members[members.industry.eq('被動元件')], case_dates)
        case = case[case.stock_id.isin(['2492', '2327'])].copy()
        individual = features(close, raw, volume, companies)
        for column, matrix in {**individual['numeric'], **individual['rules'],
                                'stock_eligible': individual['eligible']}.items():
            case[column] = [matrix.loc[pd.Timestamp(row.signal_date), row.stock_id]
                            for row in case.itertuples()]
        case['strength_with_sector'] = case.relative_strength.astype('boolean') & case.sector_confirmed
        case['earliest_execution_date'] = [
            str(close.index[close.index.get_loc(pd.Timestamp(day)) + 1].date()) for day in case.signal_date]
        verify_hashes(expected)
        output.mkdir(parents=True)
        files = {}
        stats_csv = pd.DataFrame([dict(comparison=key, **item) for key, rows in statistics.items() for item in rows])
        for name, frame in [('observations', table), ('sector_features', sectors),
                            ('passive_observations', passive), ('rule_statistics', stats_csv), ('case_daily', case)]:
            saved = frame.copy()
            if name == 'sector_features':
                # Repeated peer lists compress well in Parquet. Avoid a huge
                # CSV while retaining the exact membership for every row.
                path = output / f'{name}.parquet'
                saved.to_parquet(path, index=False)
            else:
                if 'peer_ids' in saved:
                    saved['peer_ids'] = saved.peer_ids.map(lambda ids: json.dumps(ids, ensure_ascii=False))
                path = output / f'{name}.csv'
                saved.to_csv(path, index=False, encoding='utf-8-sig')
            files[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(frame))
        result = clean(dict(format='surge_sector_v1', completed=True, membership_point_in_time=False,
            live_qualified=False, unseen_validation=False, portfolio_returns_computed=False, strategy_net_return=None,
            current_groups=int(members.industry.nunique()), mapped_companies=int(members.stock_id.nunique()),
            excluded_members_outside_fixed_cohort=excluded_ids,
            membership_metadata=json.loads(MEMBERS.with_suffix('.meta.json').read_text()),
            coverage=coverage(table), passive_coverage=coverage(passive), rule_statistics=statistics,
            incremental_comparisons=incremental_comparisons(common),
            case_asof_20260511=case[case.signal_date.eq('2026-05-11')].to_dict('records'),
            causality_checks=checks, artifacts=files, source_sha256={str(p.relative_to(ROOT)): value for p, value in expected.items()},
            elapsed_seconds=round(time.perf_counter()-started, 3), finmind_requests=0, network_calls=0, database_writes=0,
            news_included_in_quantitative_signals=False,
            limitations=[
                'Current provider membership is retrospective context, not a historically available classification ledger.',
                'Industry-chain groups include suppliers, distributors and diversified companies; common product exposure is not equal.',
                'The fixed company cohort has incomplete historical listings and delistings; out-of-cohort ids are disclosed.',
                'Turnover is raw close times shares; changes in trading share do not measure net inflows or prove executable fills.',
                'Both periods and named case stocks were previously seen; no untouched validation or causal news inference.',
                'Forward price labels and unknown outcomes inherit the previous study; no fee, holding or capital model is run.',
                'Whole-date bootstrap does not remove dependence across dates and is not multiple-testing-adjusted.',
            ]))
        write(output / 'report.json', result)
        write(output / 'manifest.json', dict(report_sha256=sha(output / 'report.json'), files=files,
                                            source_sha256=result['source_sha256']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    report = run(parser.parse_args().output)
    print(json.dumps({k: report[k] for k in ('current_groups', 'mapped_companies', 'elapsed_seconds',
                                           'finmind_requests', 'coverage')}, ensure_ascii=False))
