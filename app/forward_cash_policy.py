"""Versioned cash allocation overlay on the unchanged forward accounting engine."""
from datetime import datetime
from pathlib import Path
import hashlib
import shutil
import uuid
from app import forward_journal as j, forward_portfolio as p, forward_simulation as sim
from app import forward_comparison as comparison, forward_portfolio_service as service, forward_halts as halts
from app.file_lock import file_lock

ROOT = j.ROOT / '.cache/forward-simulation/cash-v1'
RULES = dict(version='cash-allocation-forward-v1', idle_asset='cash', cash_interest='0',
             stock_slots=3, stock_budget='prior_close_nav_divided_by_3', benchmark='0050_independent_account',
             override='Only remove strategy-account 0050 allocation; preserve inherited signal, exit and matching rules.',
             live_qualified=False)
CODE = ('app/forward_cash_policy.py', 'app/forward_cash_automation.py', 'scripts/run_cash_forward.py')


def hashes():
    return {name: hashlib.sha256((j.ROOT/name).read_bytes()).hexdigest() for name in CODE}


def verify(path, role=None):
    if not Path(path).is_file():
        raise ValueError('尚未初始化現金版帳本')
    with j.connection(path) as con:
        rows = sim.verify(con)
    seal = next((r for r in rows if r['kind'] == 'cash_allocation_protocol'), None)
    if (not seal or seal['body']['rules'] != RULES or seal['body']['code_sha256'] != hashes()
            or seal['body']['role'] not in ('strategy', 'benchmark')
            or (role is not None and seal['body']['role'] != role)):
        raise ValueError('現金版規則或程式不符；不可沿用本帳本')
    if seal['body']['role'] == 'strategy':
        state = p.state(rows)
        active = [o for o in state['orders'].values() if not o['closed'] and o['filled'] < o['qty']]
        if '0050' in state['holdings'] or any(o['stock_id'] == '0050' for o in active):
            raise ValueError('現金版不可持有0050或保留其有效委託')
        if any(r['seq'] > seal['seq'] and r['kind'] in ('order', 'funding_intent')
               and r['body']['stock_id'] == '0050' for r in rows):
            raise ValueError('現金版出現違反配置規則的0050委託')
    return rows


def initialize(root=ROOT, strategy=p.PATH, benchmark=comparison.BENCHMARK, signals=j.DEFAULT_PATH, clock=j.now):
    root = Path(root)
    with file_lock(root.parent/'.cash-init.lock', timeout=0):
        if root.exists():
            for role in ('strategy', 'benchmark'):
                verify(root/(role+'.sqlite3'), role)
            return root
        staging = root.parent/('.cash-init-'+uuid.uuid4().hex)
        try:
            sim.initialize(staging, strategy, benchmark, clock, signals)
            for role in ('strategy', 'benchmark'):
                path = staging/(role+'.sqlite3')
                with j.connection(path) as con:
                    rows = sim.verify(con)
                    j.append(con, 'cash_allocation_protocol', 'cash_allocation_protocol',
                             dict(rules=RULES, code_sha256=hashes(), role=role, inherited_head=rows[-1]['hash']), clock)
                    if role == 'strategy':
                        for o in p.state(rows)['orders'].values():
                            if o['stock_id'] == '0050' and not o['closed'] and o['filled'] < o['qty']:
                                p.submit(con, dict(kind='cancel', id='cash-policy:'+o['order_id'],
                                         order_id=o['order_id'], reason='使用者選擇閒置現金；僅在新版本取消繼承的未成交0050計畫'), clock)
                verify(path, role)
            staging.rename(root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return root


def save_plans(path, signal_path=j.DEFAULT_PATH, benchmark=False, clock=j.now):
    rows = verify(path, 'benchmark' if benchmark else 'strategy')
    if benchmark:
        return comparison.reinvest_dividends(path, clock)
    state = p.state(rows)
    if state['mark']:
        next_day = p.next_session(state, state['mark']['body']['date'])
        if halts.estimated(rows) or halts.active(rows, next_day):
            return halts.save_plans(path, signal_path, clock=clock)
    exits = comparison.save_exits(path, clock)
    try:
        proposals = service.proposals(path, signal_path, clock)
        saved = []
        with j.connection(path) as con:
            sim.verify(con)
            for command in proposals:
                # Keep the original stock sizing/priority/funding-intent logic.
                # Exits are already committed independently above.
                if command['stock_id'] != '0050' and command['kind'] in ('order', 'funding_intent'):
                    saved.append(p.submit(con, command, clock))
        verify(path, 'strategy')
        return dict(exits=exits, new_plans=saved, entry_block=None, idle_asset='cash')
    except (ValueError, KeyError, OSError) as exc:
        reason=f'{type(exc).__name__}: {exc}'
        with j.connection(path) as con:
            body=dict(date=str(clock().astimezone(p.TZ).date()), reason=reason)
            j.append(con, 'entry_gap:'+j.digest(body), 'entry_gap', body, clock)
        return dict(exits=exits, new_plans=[], entry_block=reason, idle_asset='cash')
