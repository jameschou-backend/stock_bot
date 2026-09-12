"""Operational evidence beside sealed engines; never approves or records a fill."""
from pathlib import Path
from decimal import Decimal as D
from app import forward_cash_policy as cash, forward_corporate_audit as corporate
from app import forward_automation as auto, forward_journal as j, forward_portfolio as p
from app.file_lock import file_lock

STAGES = ('observe', 'expire_strategy', 'expire_benchmark', 'pipeline', 'signals',
          'close_strategy', 'close_benchmark', 'plan_strategy', 'plan_benchmark')


def execution(rows):
    state = p.state(rows)
    result = []
    for order in state['orders'].values():
        fills = [r['body'] for r in rows if r['kind'] == 'fill' and r['body']['order_id'] == order['order_id']]
        qty = sum(f['qty'] for f in fills)
        value = sum((D(f['price']) * f['qty'] for f in fills), D(0))
        average = value / qty if qty else None
        limit = D(order['limit_price'])
        result.append(dict(order_id=order['order_id'], stock_id=order['stock_id'], channel=order['channel'],
                           side=order['side'], session=order['session'], planned=order['qty'], filled=qty,
                           unfilled=order['qty']-qty, cancelled=order['closed'],
                           fill_ratio=qty/order['qty'], average_price=str(average) if average is not None else None,
                           adverse_vs_limit_bps=float((average/limit-1)*10000*(1 if order['side']=='buy' else -1)) if average is not None else None,
                           fees_and_tax=str(sum((D(f['fee'])+D(f['tax']) for f in fills), D(0)))))
    return result


def workflow(runs):
    days = {}
    for row in runs:
        b = row['body']
        days.setdefault(b['date'], {})[b['stage']] = b
    out = []
    for day, stages in sorted(days.items()):
        passed = []
        for name in STAGES:
            stage = stages.get(name, {})
            detail = stage.get('detail')
            nested_block = (isinstance(detail, dict) and bool(detail.get('entry_block')) or
                            isinstance(detail, list) and any(x.get('status') == 'blocked' for x in detail))
            # Both books must reach collection. No active orders legitimately need no quotes.
            observed = name != 'observe' or isinstance(detail, list) and all(
                any(x.get('role') == role and x.get('status') == 'collect' for x in detail)
                for role in ('strategy', 'benchmark'))
            if stage.get('status') == 'ok' and not nested_block and observed:
                passed.append(name)
        out.append(dict(date=day, completed=len(passed)==len(STAGES),
                        missing=[name for name in STAGES if name not in passed]))
    return out


def inspect(root=cash.ROOT, evidence_path=corporate.PATH, clock=j.now):
    root = Path(root)
    books = {}
    for role in ('strategy', 'benchmark'):
        path = root/(role+'.sqlite3')
        rows = cash.verify(path, role)
        audit = corporate.inspect(path, evidence_path, clock, rows=rows)
        state = p.state(rows)
        approved = False
        if not audit['blocked']:
            signature = auto.approval_signature(rows, evidence_path, clock)
            approved = any(r['kind']=='simulation_review' and
                           r['body']['date']==str(clock().astimezone(p.TZ).date()) and
                           r['body']['source_fingerprint']==signature for r in rows)
        books[role] = dict(head=rows[-1]['hash'], source_blocked=audit['blocked'],
                           sources=audit['sources'],
                           issues=audit['issues'], events=audit['events'], scope=audit['scope'],
                           official_coverage_complete=audit['coverage_complete'],
                           manual_close_review_required=bool(state['holdings']), review_current=approved,
                           execution=execution(rows), simulated_fills=sum(r['kind']=='fill' for r in rows))
    days = workflow([row for row in auto._runs(root) if j.timestamp(row['recorded_at']) <= clock()
                     and row['body']['date'] <= str(clock().astimezone(p.TZ).date())])
    return dict(observed_at=clock().isoformat(), policy=cash.RULES['version'], books=books, days=days,
                completed_days=sum(d['completed'] for d in days), live_qualified=False,
                broker_execution_verified=False,
                next_action='更新公司行動來源' if any(b['source_blocked'] for b in books.values()) else
                '核對今日官方公告並保存核對' if any(b['manual_close_review_required'] and not b['review_current'] for b in books.values()) else
                '等待交易時段觀察；有成交持股後仍需當日公告核對',
                research_gaps=['現金版未累積未見區間績效', '歷史還原價與資料可用時間仍需完整對帳',
                               '歷史股票名冊與存活者偏差仍需驗證'])


def refresh(root=cash.ROOT, evidence_path=corporate.PATH, clock=j.now, budget=12):
    """One bounded pass for BOTH books, same lock as the scheduled cash runner."""
    if type(budget) is not int or not 1 <= budget <= 12:
        raise ValueError('兩份帳本合計查詢上限須介於1至12')
    root = Path(root)
    if not root.is_dir():
        raise ValueError('現金版帳本尚未初始化')
    with file_lock(root/'.run.lock', timeout=0):
        for role in ('strategy', 'benchmark'):
            cash.verify(root/(role+'.sqlite3'), role)
        calls = reused = 0
        for role in ('strategy', 'benchmark'):
            if calls >= budget:
                break
            result = corporate.refresh(root/(role+'.sqlite3'), evidence_path, clock, request_budget=budget-calls)
            calls += result['calls']
            reused += result['reused']
        return dict(calls=calls, reused=reused, report=inspect(root, evidence_path, clock))
