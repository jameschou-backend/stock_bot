"""Read-only temporal reconstruction; never a retroactive execution attestation."""
from datetime import timedelta
from pathlib import Path

from app import capacity_forward as policy, forward_corporate_audit as corporate
from app import forward_journal as j
from app import capacity_source_guard as guard


def at_time(rows, history, at):
    sources = {tuple(r['body']['query']): r for r in history if j.timestamp(r['recorded_at']) <= at}
    required = corporate.specs(corporate.scope(rows, lambda: at))
    checks = []
    for query in required:
        source = sources.get(query)
        retrieved = j.timestamp(source['body']['retrieved_at']) if source else None
        age = (at - retrieved).total_seconds() if source else None
        ok = bool(source and source['body']['status'] == 'ok' and 0 <= age <= corporate.TTL)
        checks.append(dict(dataset=corporate.LABELS[query[0]], stock_id=query[1] or '全市場',
                           hash=source['hash'] if source else None,
                           recorded_at=source['recorded_at'] if source else None,
                           retrieved_at=retrieved.isoformat() if retrieved else None,
                           expires_at=(retrieved + timedelta(seconds=corporate.TTL)).isoformat() if retrieved else None,
                           age_seconds=age, fresh=ok))
    return dict(source_freshness_passed=bool(checks) and all(s['fresh'] for s in checks), sources=checks)


def review(root=policy.ROOT, clock=j.now):
    at = clock()
    history = corporate.source_history()
    books = {}
    for role in policy.ROLES:
        rows = policy.verify(Path(root) / (role + '.sqlite3'), role)
        fills = []
        for index, row in enumerate(rows):
            if row['kind'] != 'fill':
                continue
            b = row['body']
            executed = j.timestamp(b['executed_at'])
            if executed > at or j.timestamp(row['recorded_at']) > at:
                continue
            fills.append(dict(fill_hash=row['hash'], executed_at=b['executed_at'], qty=b['qty'],
                              **at_time(rows[:index], history, executed)))
        books[role] = dict(fills=fills, current=at_time(rows, history, at))
    guard_info = dict(active=False, pending_checks=0)
    path = Path(root) / guard.NAME
    if path.exists():
        guard.verify(root)
        with j.connection(path) as con:
            events = j.read_events(con)
        done = {r['body']['check_hash'] for r in events if r['kind'] == 'source_guard_result'}
        pending = [r['hash'] for r in events if r['kind'] == 'source_guard_check' and r['hash'] not in done]
        guard_info = dict(active=True, effective_at=events[0]['recorded_at'], pending_checks=len(pending),
                          pending_hashes=pending, head=events[-1]['hash'])
    return dict(observed_at=at.isoformat(), books=books, guard=guard_info, live_qualified=False,
                note='依當時已記錄來源重建有效期限；不是事後補簽撮合檢查，也不代表公司行動完整性或券商成交已驗證。')
