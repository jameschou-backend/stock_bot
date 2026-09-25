#!/usr/bin/env python3
"""Month-boundary future-isolation checks on all three corrected selector arms."""
from copy import deepcopy
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts.research_exit_scenarios import read, write, sha
from scripts.audit_current_causality_20260925 import cutoff_plan, deterministic
from scripts.audit_signal_causality import transform, digest
from skills.historical_selector_replay import build_signals
from skills.verified_backtest_tool import offline_only
from skills.backtest_case_cache import file_identities


def as_of(result, cutoff):
    groups = []
    for original in result['diffusion']['groups']:
        if original['month'] <= cutoff[:7]:
            group = deepcopy(original)
            group['leader_rejections'] = [r for r in group['leader_rejections'] if r['date'] <= cutoff]
            groups.append(group)
    return dict(entries=[r for r in result['entries'] if r['signal_date'] <= cutoff],
                rejections=[r for r in result['rejections'] if r['signal_date'] <= cutoff], groups=groups)


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Keep prior causality evidence; choose a new output')
    started = time.monotonic()
    names = ('close-official', 'close-quality', 'raw-close', 'raw-volume')
    manifest = read(source / 'manifest.json')
    refs = {str((source / name).relative_to(ROOT)): value for name, value in manifest['files_sha256'].items()
            if any(name.startswith(arm + '/') for arm in ('identity', 'omitted', 'combined'))}
    refs.update(read(source / 'identity.json'))
    for p in (Path(__file__), source / 'manifest.json', ROOT / 'scripts/audit_signal_causality.py',
              ROOT / 'scripts/audit_current_causality_20260925.py'):
        refs[str(p.relative_to(ROOT))] = sha(p)
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Causality sources changed')
    result = dict(schema='historical_selector_causality_v1', passed=False, complete=False,
        cases=[], source_sha256=refs, network_calls=0, live_qualified=False,
        scope='Fixed-snapshot algorithmic causality; historical source publication availability remains unproven')
    with offline_only():
        for arm in ('identity', 'omitted', 'combined'):
            frames = {name: pd.read_parquet(source / arm / (name + '.parquet')).set_index('date') for name in names}
            for frame in frames.values():
                frame.index = pd.to_datetime(frame.index)
            companies = pd.read_parquet(source / arm / 'companies.parquet')
            mask = pd.read_parquet(source / arm / 'eligibility.parquet').set_index('date')
            mask.index = pd.to_datetime(mask.index)
            frozen = read(source / arm / 'signals.json')
            full = build_signals(frames, companies, mask)
            if deterministic(full) != deterministic(frozen):
                raise ValueError('Full selector cannot be reproduced: ' + arm)
            plan = cutoff_plan(frames['raw-close'].index, '2022-01-03', '2026-09-08')
            result['expected_checks'] = len(plan) * 2 * 3
            for planned in plan:
                cutoff = planned['cutoff']
                target = as_of(full, cutoff)
                for mode in ('truncate', 'mutate'):
                    altered = dict(zip(names, transform(list(frames.values()), cutoff, mode)))
                    future_mask = mask.reindex(altered['raw-close'].index).copy()
                    if mode == 'mutate':
                        # Mutate future legal eligibility too; historical rows must be untouched.
                        future_mask.loc[future_mask.index > pd.Timestamp(cutoff)] = ~future_mask.loc[future_mask.index > pd.Timestamp(cutoff)]
                    rebuilt = build_signals(altered, companies, future_mask, signal_end=cutoff)
                    actual = as_of(rebuilt, cutoff)
                    passed = actual == target
                    result['cases'].append(dict(arm=arm, cutoff=cutoff, mode=mode, passed=passed,
                        expected_sha256=digest(target), actual_sha256=digest(actual)))
                    result['elapsed_seconds'] = round(time.monotonic()-started, 3)
                    write(output, result)
                    if not passed:
                        raise ValueError(f'Future data changed prior decisions: {arm} {cutoff} {mode}')
                print(arm, cutoff, 'PASS', flush=True)
    if file_identities([ROOT / p for p in refs], ROOT) != refs:
        raise ValueError('Source/code changed during causality checks')
    result.update(passed=len(result['cases']) == result['expected_checks'], complete=True,
                  elapsed_seconds=round(time.monotonic()-started, 3))
    write(output, result)
    output.with_suffix('.sha256').write_text(sha(output) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.output)
    print('passed', result['passed'], 'checks', len(result['cases']))
