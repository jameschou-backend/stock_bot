"""Prospective, separately sealed freshness guard around the unchanged matcher.

The inherited protocol and all earlier fills remain intact. This guard can only
reject an observation; it cannot change a price, quantity, rule or source TTL.
"""
from datetime import timedelta
from pathlib import Path
import hashlib
import uuid

from app import capacity_forward as policy, capacity_forward_runner as runner
from app import forward_corporate_audit as corporate, forward_journal as j
from app import forward_simulation as sim
from app.file_lock import file_lock

NAME = 'source-guard-v1.sqlite3'
CODE = ('app/capacity_source_guard.py', 'app/capacity_guard_runner.py',
        'scripts/run_capacity_guarded.py', 'docs/prereg_capacity_source_guard_v1.md')
RULES = dict(version='capacity-source-guard-v1', purpose='reject_expired_sources_before_each_match',
             retrospective_attestation=False, inherited_rules_unchanged=True, live_qualified=False)


def hashes():
    return {name: hashlib.sha256((j.ROOT / name).read_bytes()).hexdigest() for name in CODE}


def verify(root):
    paths = runner.books(root)  # Retain ALL inherited fingerprint checks.
    path = Path(root) / NAME
    if not path.is_file():
        raise ValueError('逐筆來源檢查尚未啟用；請先執行 guarded activate')
    with j.connection(path) as con:
        rows = j.read_events(con)
    seal = rows[0] if rows else None
    if not seal or seal['kind'] != 'source_guard_protocol' or seal['body']['rules'] != RULES or seal['body']['code_sha256'] != hashes():
        raise ValueError('來源檢查版本不符；保留紀錄，不得重封指紋')
    completed = {r['body']['check_hash'] for r in rows if r['kind'] == 'source_guard_result'}
    if any(r['kind'] == 'source_guard_check' and r['hash'] not in completed for r in rows):
        raise ValueError('上次逐筆來源檢查中斷，需核對帳本；不得直接重跑撮合')
    for role, path in paths.items():
        parent = policy.verify(path, role)
        head = seal['body']['inherited_heads'][role]
        if not any(row['hash'] == head for row in parent):
            raise ValueError('來源檢查與繼承帳本不符')
    return paths


def activate(root=policy.ROOT, clock=j.now):
    root = Path(root)
    with file_lock(root / '.run.lock', timeout=0):
        if (root / NAME).exists():
            verify(root)
            return dict(status='already_active')
        paths = runner.books(root)
        rows = {role: policy.verify(path, role) for role, path in paths.items()}
        body = dict(rules=RULES, code_sha256=hashes(),
                    inherited_heads={role: rs[-1]['hash'] for role, rs in rows.items()},
                    inherited_fill_counts={role: sum(r['kind'] == 'fill' for r in rs) for role, rs in rows.items()})
        with j.connection(root / NAME) as con:
            seal = j.append(con, 'source_guard_protocol', 'source_guard_protocol', body, clock)
        return dict(status='activated', effective_at=seal['recorded_at'], hash=seal['hash'])


def _record(root, key, kind, body, clock):
    with j.connection(Path(root) / NAME) as con:
        return j.append(con, key, kind, body, clock)


def match(root, role, observation, clock=j.now, evidence_path=corporate.PATH):
    path = verify(root)[role]
    before = policy.verify(path, role)
    checked_at = clock()
    audit = corporate.inspect(path, evidence_path, lambda: checked_at, rows=before)
    token = uuid.uuid4().hex
    start = _record(root, token + ':start', 'source_guard_check', dict(
        role=role, account_head=before[-1]['hash'], quote_hash=observation['hash'],
        checked_at=checked_at.isoformat(), blocked=audit['blocked'],
        sources=audit['sources'], issues=audit['issues']), clock)
    if audit['blocked']:
        result = dict(status='blocked', fills=[], reasons=['公司行動來源缺漏、過期或待核對'])
    else:
        deadlines = [j.timestamp(s['retrieved_at']) + timedelta(seconds=corporate.TTL) for s in audit['sources']]
        deadline = min(deadlines) if deadlines else checked_at + timedelta(seconds=corporate.TTL)
        last_clock, invalid_clock = checked_at, False

        def protected_clock():
            nonlocal last_clock, invalid_clock
            now = clock()
            invalid_clock = invalid_clock or now < last_clock or now > deadline
            last_clock = now
            if invalid_clock:
                raise ValueError('撮合期間公司行動來源已過期或時鐘倒退；整筆觀察不入帳')
            return now

        try:
            # The frozen matcher uses this clock for proposal, fill and journal
            # append. A deadline crossing raises inside its transaction, rolling
            # back ALL fills/observations from this call, including partial work.
            protected_clock()
            raw = sim.match(path, observation, protected_clock)
        except ValueError as exc:
            result = dict(status='blocked', fills=[], reasons=[str(exc)])
        else:
            # If verification fails AFTER a committed match, keep the start
            # unresolved for review rather than falsely recording zero fills.
            after = policy.verify(path, role)
            new_fills = [r['hash'] for r in after[len(before):] if r['kind'] == 'fill']
            result = dict(status='observation', fills=new_fills, reasons=raw.get('reasons', []))
    _record(root, token + ':result', 'source_guard_result', dict(
        role=role, check_hash=start['hash'], **result), clock)
    return result
