#!/usr/bin/env python3
"""One candidate-validity contrast across eight sealed execution stresses."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.file_lock import file_lock
from app.residual_slots_ui import load, REPORT
from scripts import research_residual_slots as parent
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize, TrackedCorporateActions
from skills.backtest_case_cache import file_identities
from skills.backtest_contract import validate_completed_account
from skills.candidate_queue_replay import CandidateQueueReplay, audit_queue
from skills.residual_slot_replay import audit_residual_slots
from skills.execution_factorial import flags, stock_pnl, load_capital_terms
from skills.replay_market_feeds import ReplayMarketFeeds, ReplayDataUnavailable
from skills.million_replay import UnresolvedAction
from skills.trial_registry import append_trial_registry
from skills.verified_backtest_tool import offline_only

OUTPUT = ROOT / '.cache/candidate-queue-20260927'
CODE = [Path(__file__), ROOT/'skills/candidate_queue_replay.py',
        ROOT/'skills/trial_registry.py', ROOT/'docs/prereg_candidate_queue_20260927.md']


def case(data, inputs, identity, additions, mask, validity):
    config = dict(factor_mask=mask, factors=flags(mask), validity_sessions=validity,
                  benchmark=False, board_only=True, position_count=5)
    feeds = ReplayMarketFeeds(inputs/'execution-feeds', offline=True)
    overrides = (read(parent.parent.parent.sealed.parent.OVERRIDES)['overrides'] |
        read(parent.parent.parent.sealed.parent.ADDITIONS)['overrides'] |
        read(ROOT/'docs/intraday_corporate_additions_20260914.json')['overrides'] | additions)
    corp = TrackedCorporateActions(data.events, inputs/'dividends', None, offline=True, overrides=overrides)
    actions = list(zip(data.events.stock_id, data.events.event_date))
    engine = CandidateQueueReplay(data.quotes, data.companies, data.days, data.entries, feeds, corp,
        start=data.start, end=data.end, identity_report=identity, factor_mask=mask,
        validity_sessions=validity, exit_signals=data.features, action_dates=actions)
    try:
        account = engine.run()
        validate_completed_account(account, [str(d.date()) for d in data.days], data.start, data.end)
        audit = audit_residual_slots(account, engine.resource_plans, engine.slot_decisions,
            engine.board_decisions, engine.residual_days, data.quotes)
        queue = audit_queue(account, engine.queue_decisions, data.entries, data.days, data.quotes,
                            actions, validity, mask)
        return dict(completed=True, config=config, account=account, summary=summarize(account),
            audit=audit, queue_audit=queue, queue_decisions=engine.queue_decisions,
            stock_pnl=stock_pnl(account, engine.marks), residual_days=engine.residual_days,
            resource_plans=engine.resource_plans, slot_decisions=engine.slot_decisions,
            board_decisions=engine.board_decisions, identity_decisions=engine.identity_decisions,
            live_qualified=False, unseen_validation=False)
    except (ReplayDataUnavailable, UnresolvedAction) as exc:
        return dict(completed=False, config=config, reason=str(exc), partial_account=dict(
            daily=engine.daily, trades=engine.trades, orders=engine.orders), live_qualified=False)
    except ValueError as exc:
        if not str(exc).startswith('Frozen dividend source missing:'):
            raise
        return dict(completed=False, config=config, reason=str(exc), live_qualified=False)


def metrics(value, benchmark):
    a, b = value['account']['daily'], benchmark['account']['daily']
    if [r['date'] for r in a] != [r['date'] for r in b]:
        raise ValueError('Benchmark calendar differs')
    rolling = [(a[i]['nav']/a[i-251]['opening_nav']-1) -
               (b[i]['nav']/b[i-251]['opening_nav']-1) for i in range(251, len(a))]
    outcomes = value['queue_audit']['candidate_outcomes']
    retry_ids = {r['event_id'] for r in value['queue_decisions'] if r['attempt_number']==2 and r['reason']=='eligible'}
    return dict(benchmark_return=benchmark['summary']['total_return'],
        excess_return=value['summary']['total_return']-benchmark['summary']['total_return'],
        rolling252_count=len(rolling), rolling252_win_rate=sum(x>0 for x in rolling)/len(rolling),
        rolling252_worst_excess=min(rolling),
        annual_excess=[dict(year=x['year'], excess=x['total_return']-y['total_return'])
                      for x,y in zip(value['summary']['annual'], benchmark['summary']['annual'])],
        traded_notional=sum(t['gross'] for t in value['account']['trades']),
        candidate_outcomes=dict(Counter(r['outcome'] for r in outcomes)),
        retries_filled=sum(r['event_id'] in retry_ids and r['outcome']=='filled' for r in outcomes))


def run(output):
    tick = time.monotonic(); output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(OUTPUT) or output==OUTPUT:
        raise ValueError('Choose a new immutable candidate-queue directory')
    original, selector = load(), parent.load_selector()
    refs = dict(original['source_sha256'])
    refs.update(file_identities([REPORT, REPORT.with_suffix('.sha256'), *CODE], ROOT))
    write(output/'identity.json', refs)
    rows, trials = {}, []
    with offline_only():
        data, inputs, identity = parent.parent.load_data(selector)
        additions = parent.parent.parent.load_corporate_completion(ROOT) | load_capital_terms(ROOT)[0]
        for validity in (1, 2):
            for mask in range(8):
                name = f'valid{validity}_{mask}'
                print('running', name, flush=True)
                value = None
                try:
                    value = case(data, inputs, identity, additions, mask, validity)
                    write(output/'cases'/(name+'.json'), value)
                    if validity==1:
                        sealed = read(ROOT/original['cases']['release_'+str(mask)]['result']['path'])
                        if not value['completed'] or encoded(value['account']) != encoded(sealed['account']):
                            raise ValueError('One-session control does not reproduce '+name)
                finally:
                    record = dict(timestamp=datetime.now(timezone.utc).isoformat(), source='candidate_queue_20260927',
                        command=' '.join(sys.argv), case=name, factor_mask=mask, validity_sessions=validity,
                        result_path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),
                        status='completed' if value and value['completed'] else 'blocked_or_failed')
                    append_trial_registry(record)
                    trials.append(record); write(output/'trials.json', trials)
                row = dict(completed=value['completed'], config=value['config'], summary=value.get('summary'),
                    reason=value.get('reason'), result=dict(path=str((output/'cases'/(name+'.json')).relative_to(ROOT)),
                                                         sha256=sha(output/'cases'/(name+'.json'))))
                if value['completed']:
                    benchmark = read(ROOT/original['cases']['benchmark_combined' if mask&1 else 'benchmark_control']['result']['path'])
                    row['metrics'] = metrics(value, benchmark)
                rows[name] = row
                print(name, value.get('summary', {}).get('total_return', value.get('reason')), flush=True)
    if file_identities([ROOT/p for p in refs], ROOT) != refs:
        raise ValueError('Source changed during queue research')
    contrasts = {str(m):dict(return_change=rows[f'valid2_{m}']['summary']['total_return']-rows[f'valid1_{m}']['summary']['total_return'],
        final_nav_change=rows[f'valid2_{m}']['summary']['final_nav']-rows[f'valid1_{m}']['summary']['final_nav'])
        for m in range(8) if rows[f'valid2_{m}']['completed']}
    report = dict(schema='candidate_queue_research_v1', start=data.start, end=data.end, initial_cash=1_000_000,
        candidate_count=len(data.entries), all_completed=all(r['completed'] for r in rows.values()),
        cases=rows, contrasts=contrasts, elapsed_seconds=round(time.monotonic()-tick, 3),
        trial_count=len(trials), network_calls=0, database_writes=0,
        live_qualified=False, unseen_validation=False, strict_data_ready=False)
    write(output/'report.json', report)
    write(output/'manifest.json', dict(files_sha256={str(p.relative_to(output)):sha(p)
        for p in output.rglob('*.json') if p.name!='manifest.json'}))
    return report


def verify(left, right, output):
    left, right, output = (Path(p).resolve() for p in (left, right, output))
    if left==right or output.exists():
        raise ValueError('Use separate runs and a new proof file')
    reports = []
    for folder in (left, right):
        for name,digest in read(folder/'manifest.json')['files_sha256'].items():
            if sha(folder/name)!=digest:
                raise ValueError('Research artifact changed '+name)
        refs=read(folder/'identity.json')
        if file_identities([ROOT/p for p in refs], ROOT)!=refs:
            raise ValueError('Research source changed')
        reports.append(read(folder/'report.json'))
    if read(left/'identity.json')!=read(right/'identity.json'):
        raise ValueError('Run source identities differ')
    expected={f'valid{v}_{m}' for v in (1,2) for m in range(8)}
    if any(set(r['cases'])!=expected for r in reports):
        raise ValueError('Sixteen recorded cases required, including blocked cases')
    for name in expected:
        if read(left/'cases'/(name+'.json'))!=read(right/'cases'/(name+'.json')):
            raise ValueError('Full case did not reproduce '+name)
    if reports[0]['contrasts']!=reports[1]['contrasts']:
        raise ValueError('Contrasts did not reproduce')
    proof=dict(schema='candidate_queue_reproduction_v1', passed=True, compared_cases=16,
        all_completed=all(r['all_completed'] for r in reports), network_calls=0,
        runs=[dict(path=str((f/'manifest.json').relative_to(ROOT)), sha256=sha(f/'manifest.json')) for f in (left,right)],
        live_qualified=False)
    write(output, proof); output.with_suffix('.sha256').write_text(sha(output)+'\n')
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path, nargs=2)
    args=parser.parse_args()
    with file_lock(OUTPUT/'.run.lock', timeout=0):
        report=verify(*args.compare, args.output) if args.compare else run(args.output)
    print('complete', report['all_completed'], flush=True)
