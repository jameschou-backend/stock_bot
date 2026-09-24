"""One offline entry point for the frozen cash strategy and matched 0050 accounts.

This adapter reuses the sealed accounting engines without changing their rules.
Only complete daily-model accounts receive performance statistics. Sequence
execution cannot be inferred from daily volume, even when a daily account passes.
"""
from contextlib import ExitStack, contextmanager
from importlib.metadata import version
from pathlib import Path
import csv
import io
import platform
import re
import socket
import time
from unittest.mock import patch

from app.file_lock import file_lock
from skills.backtest_case_cache import CaseStore, content_digest, file_identities
from skills.backtest_contract import validate_signals, validate_completed_account, validate_comparison
from scripts import research_board_only_supplement as sealed
from scripts.research_exit_scenarios import read, write, sha, encoded, summarize
from skills.execution_resources import audit_resources
from skills.slot_reuse_replay import audit_slots

ROOT = Path(__file__).resolve().parents[1]
SEAL = ROOT / '.cache/board-only-supplement-verified-r2-20260925'
RUNS = ROOT / '.cache/backtest-tool/runs'
CAUSALITY = ROOT / 'artifacts/forward_simulation/current_causality_20260925.json'
CODE = ('skills/verified_backtest_tool.py', 'skills/backtest_case_cache.py',
        'skills/backtest_contract.py', 'scripts/run_verified_backtest.py',
        'docs/backtest_tool.md')
LIMITATIONS = [
    '完整日資料回測仍是成交估算；日量、收盤價、最後報價不能證明委託送出後可成交。',
    '歷史公司名冊、公告實際可得時間與資料修訂版本尚未全部核實。',
    '2022 至 2026 年已多次研究，不是未見樣本；通過帳務檢查不代表未來可獲利。',
    '期末淨值包括未賣持股與應收，不能全部視為可動用現金。',
    '預檢列出固定資料版本下已知的缺口；不宣稱覆蓋所有未走到的交易路徑。',
]


@contextmanager
def offline_only():
    def denied(*args, **kwargs):
        raise RuntimeError('Backtest tool is offline: implicit network access is prohibited')
    with ExitStack() as stack:
        for target, name in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                             (socket.socket, 'sendto'), (socket, 'getaddrinfo'),
                             (socket, 'create_connection')):
            stack.enter_context(patch.object(target, name, denied))
        yield


def source_context():
    """Hash each declared source once per boundary, including all transitive code.

    The sealed identity already contains the recursive parent inventory. Calling
    each parent's inventory again would read the same multi-GB inputs repeatedly.
    Copied inputs inside SEAL are not used; the source supplement is authoritative.
    """
    manifest = read(SEAL / 'manifest.json')
    offline = read(SEAL / 'offline.json')
    if (offline.get('all_cases_identical') is not True or offline.get('parent_controls_identical') is not True
            or offline['manifest_sha256'] != sha(SEAL / 'manifest.json')):
        raise ValueError('Sealed offline verification is not bound to this manifest')
    needed = ['identity.json', 'summary.json'] + [f'cases/{name}.json' for name, _ in sealed.parent.configurations()]
    for name in needed:
        if sha(SEAL / name) != manifest['files_sha256'][name]:
            raise ValueError('Sealed case/reference changed: ' + name)
    expected = read(SEAL / 'identity.json')
    causal = read(CAUSALITY)
    if not all(causal[key] is True for key in ('passed', 'complete', 'sources_unchanged_after_run')):
        raise ValueError('Current signal causality audit is not complete')
    for name, digest in causal['source_and_code_sha256'].items():
        if name in expected and expected[name] != digest:
            raise ValueError('Causality evidence and account source versions disagree: ' + name)
        expected[name] = digest
    current = file_identities([ROOT / name for name in expected], ROOT)
    if current != expected:
        changed = sorted(name for name in expected if current.get(name) != expected[name])
        raise ValueError('Frozen source changed; prepare a new reviewed dataset: ' + ', '.join(changed[:5]))
    extra = [ROOT / p for p in CODE] + [CAUSALITY, SEAL / 'manifest.json', SEAL / 'offline.json']
    extra += [SEAL / name for name in needed]
    identity = dict(recipe='conservative_cash_five_slots_v1', source_sha256=current | file_identities(extra, ROOT),
                    runtime=dict(python=platform.python_version(), pandas=version('pandas'), numpy=version('numpy')),
                    initial_cash=1_000_000, start='2022-01-03', end='2026-09-09', candidate_count=458)
    priors = {name: read(SEAL / f'cases/{name}.json') for name, _ in sealed.parent.configurations()}
    return identity, priors


def configurations(policy, stress):
    if policy not in ('all', 'mixed', 'board_only') or stress not in ('all', 'control', 'combined'):
        raise ValueError('Unknown execution policy or cost scenario')
    return [(name, config) for name, config in sealed.parent.configurations()
            if (policy == 'all' or config['board_only'] == (policy == 'board_only'))
            and (stress == 'all' or config['stress'] == stress)]


def label(config):
    return ('0050' if config['benchmark'] else '策略 5 檔') + ' · ' + (
        '只買賣整張' if config['board_only'] else '整張加零股') + ' · ' + (
        '一般成本' if config['stress'] == 'control' else '加嚴成交')


def known_issue(result):
    reason = result.get('reason', '')
    if reason.startswith('Stock dividend data missing or invalid:'):
        detail = reason.split(':', 1)[1].strip().split(';')[0]
        return '股票股利缺少已核實的配股比例或交付日：' + detail
    if 'price-limit' in reason:
        return '缺少當日漲跌停價格證據：' + reason
    return reason


def preflight(selected, priors, mode):
    issues, ready = [], 0
    for name, config in selected:
        if mode == 'strict':
            issues.append(dict(code='sequence_evidence_required', case_name=name,
                message='尚未建立覆蓋此完整帳戶的普通盤及所需零股逐筆成交證據；日資料不能替代。'))
        elif not priors[name]['completed']:
            issues.append(dict(code='known_source_gap', case_name=name, message=known_issue(priors[name])))
        else:
            ready += 1
    return dict(issues=issues, ready_cases=ready, blocked_cases=len(selected)-ready,
                scope='來源雜湊與已封存案例的已知缺口檢查，不是全市場資料完整性證明。')


def verify_result(result, config, calendar, identity):
    if result.get('config') != config or result.get('live_qualified') is not False or result.get('unseen_validation') is not False:
        raise ValueError('Case configuration or research scope is invalid')
    if result.get('completed') is not True:
        if not result.get('reason') or result.get('summary'):
            raise ValueError('Blocked case must state a reason and cannot carry a full-period return')
        return
    account = result['account']
    if account['settings']['initial_cash'] != identity['initial_cash']:
        raise ValueError('Account capital differs from the experiment')
    validate_completed_account(account, calendar, identity['start'], identity['end'])
    if config['board_only']:
        sealed.audit_account(account, result['resource_plans'], result['slot_decisions'],
                             result['board_decisions'], config['benchmark'])
    elif config['benchmark']:
        audit_resources(account, result['resource_plans'], opening_cash_only=True, lock_unused=True, lock_slots=False)
    else:
        audit_slots(account, result['resource_plans'], result['slot_decisions'], opening_cash_only=True,
                    lock_unused=True, lock_opening_slots=True, lock_failed_slots=True)
    if encoded(summarize(account)) != encoded(result['summary']):
        raise ValueError('Reported performance differs from the audited account')


def export_csv(path, rows):
    # Canonical cache JSON sorts keys; preserve the same column order on first
    # execution, resume and forced replay regardless of dictionary insertion.
    keys = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue().encode('utf-8-sig')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError('An immutable account export changed')
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def report_row(name, config, result=None, *, status=None, reason='', cache_hit=False, store=None):
    complete = result is not None and result['completed'] is True
    row = dict(name=name, label=label(config), status=status or ('completed_daily' if complete else 'blocked'),
               config=config, total_return=None, max_drawdown=None, reason=reason,
               cache_hit=cache_hit, result_path=None, result_sha256=None, artifact_paths={})
    if result is not None:
        row['reason'] = '' if complete else known_issue(result)
    if complete:
        result_path = store.directory / 'cases' / name / 'result.json'
        row.update(total_return=result['summary']['total_return'], max_drawdown=result['summary']['max_drawdown'],
                   result_path=str(result_path.relative_to(ROOT)), result_sha256=sha(result_path),
                   final_nav=result['summary']['final_nav'], costs=result['summary']['costs'],
                   annual=result['summary']['annual'])
        row['artifact_paths'] = {key: export_csv(store.directory / 'exports' / f'{name}-{key}.csv', result['account'][key])
                                 for key in ('trades', 'daily')}
    return row


def run(*, output, mode='daily', policy='all', stress='all', preflight_only=False, fresh=False):
    if mode not in ('daily', 'strict'):
        raise ValueError('Unknown evidence mode')
    output = Path(output).expanduser().resolve()
    # A report may not overwrite inputs, evidence, code, or another case receipt.
    report_root = ROOT / '.cache/backtest-tool'
    tool_report = report_root in output.parents and 'runs' not in output.relative_to(report_root).parts
    worker_report = (output.parent == ROOT / '.cache/workbench/jobs'
                     and re.fullmatch(r'[0-9a-f]{32}\.result\.json', output.name))
    if output.suffix != '.json' or not (tool_report or worker_report):
        raise ValueError('Use a JSON report under .cache/backtest-tool outside runs, or a workbench job result')
    if output.exists():
        prior = read(output)
        if prior.get('format') != 'backtest_tool_v1':
            raise ValueError('Refusing to overwrite a non-tool file')
    tick = time.monotonic()
    selected = configurations(policy, stress)
    with file_lock(ROOT / '.cache/backtest-tool.lock', timeout=0), offline_only():
        print('[TIMER] source_validation start', flush=True)
        identity, priors = source_context()
        checked = time.monotonic()
        flight = preflight(selected, priors, mode)
        digest = content_digest(identity)
        rows, results, executed, reused = [], {}, 0, 0
        report = dict(format='backtest_tool_v1', status='blocked', mode=mode,
                      recipe=identity['recipe'], start=identity['start'], end=identity['end'],
                      initial_cash=identity['initial_cash'], candidate_count=identity['candidate_count'],
                      input_identity=digest, preflight=flight, case_rows=rows, comparisons=[],
                      live_qualified=False, unseen_validation=False, limitations=LIMITATIONS,
                      schedules_restarted=False, broker_orders_sent=False)
        if mode == 'strict' or preflight_only:
            by_case = {r['case_name']: r['message'] for r in flight['issues']}
            rows.extend(report_row(name, config, status='blocked' if name in by_case else 'ready',
                                   reason=by_case.get(name, '來源核對通過；尚未執行本次帳戶重算')) for name, config in selected)
            report['status'] = 'blocked' if flight['blocked_cases'] else 'preflight_ready'
        else:
            print('[TIMER] prepare start', flush=True)
            data, _ = sealed.parent.source.inputs()
            calendar = [str(day.date()) for day in data.days]
            if len(data.entries) != identity['candidate_count'] or str(data.start)[:10] != identity['start'] or str(data.end)[:10] != identity['end']:
                raise ValueError('Fixed recipe data interval/candidate count changed')
            report['signal_contract'] = validate_signals(data.entries, calendar)
            store = CaseStore(RUNS / digest, identity)
            additions = read(sealed.SOURCES / 'overrides.json')['overrides']
            # Board variants depend on the four exact mixed-account controls.
            required = dict(selected)
            if any(config['board_only'] for _, config in selected):
                required = {name: config for name, config in sealed.parent.configurations()
                            if not config['board_only'] or name in required}
            for name, config in required.items():
                print('[TIMER] case start ' + name, flush=True)
                cached = store.load(name, config)
                result = None if fresh else cached
                hit = result is not None
                if result is None:
                    result = sealed.run_case(data, config, sealed.SOURCES / 'inputs', additions)
                    if not result['completed'] and 'partial_account' in result:
                        control = f"{'benchmark' if config['benchmark'] else 'capacity'}_{config['stress']}_mixed"
                        result['partial_audit'] = sealed.partial_audit(result, results[control])
                    # New infrastructure must reproduce every sealed case, including
                    # blocked partial journals. It does not redefine earlier returns.
                    if encoded(result) != encoded(priors[name]):
                        raise ValueError('Recomputed account differs from the frozen reference: ' + name)
                    if cached is not None and encoded(result) != encoded(cached):
                        raise ValueError('Fresh computation differs from its cached result: ' + name)
                    executed += 1
                else:
                    reused += 1
                verify_result(result, config, calendar, identity)
                if not config['board_only'] and not (result['completed'] and result.get('parent_account_identical')):
                    raise ValueError('Mixed control did not reproduce; board-only comparisons are prohibited')
                store.save(name, config, result)
                results[name] = result
                if name in dict(selected):
                    rows.append(report_row(name, config, result, cache_hit=hit, store=store))
            for name, config in selected:
                if config['benchmark'] or not results[name]['completed']:
                    continue
                base = f"benchmark_{config['stress']}_{'board_only' if config['board_only'] else 'mixed'}"
                if results[base]['completed']:
                    pair = validate_comparison(results[name], results[base])
                    report['comparisons'].append(dict(strategy=name, benchmark=base, audit=pair,
                        excess_return=results[name]['summary']['total_return']-results[base]['summary']['total_return']))
            report['status'] = 'exploratory' if all(row['status'] == 'completed_daily' for row in rows) else 'blocked'
        print('[TIMER] source_validation end_check', flush=True)
        # Warm runs and preflight also use evidence after the initial snapshot.
        # Check bytes again before publication, even when no case was recomputed.
        if file_identities([ROOT / p for p in identity['source_sha256']], ROOT) != identity['source_sha256']:
            raise ValueError('Sources changed during replay; results are not publishable')
        report['metrics'] = dict(elapsed_seconds=round(time.monotonic()-tick, 3),
            source_validation_seconds=round(checked-tick, 3), executed_cases=executed,
            reused_cases=reused, source_files=len(identity['source_sha256']), network_calls=0,
            finmind_requests=0, database_writes=0)
        write(output, report)
    return report
