"""Read-only closing checklist for the THREE capacity simulation books."""
from datetime import time
from decimal import Decimal as D
from pathlib import Path

from app import capacity_forward as policy, capacity_source_guard as guard
from app import forward_automation as auto, forward_corporate_audit as corporate
from app import forward_journal as j, forward_portfolio as p
from app.forward_readiness import execution
from app.file_lock import file_lock


def cash_check(rows, day):
    before = [r for r in rows if str(j.timestamp(r['recorded_at']).astimezone(p.TZ).date()) < day]
    opening = p.state(before)['cash']
    orders = p.state(rows)['orders']
    rights = p.state(rows)['rights']
    bought = sold = fees = received = D(0)
    count = 0
    for r in rows:
        if str(j.timestamp(r['recorded_at']).astimezone(p.TZ).date()) != day:
            continue
        b = r['body']
        if r['kind'] == 'fill':
            count += 1
            value = D(b['price']) * b['qty']
            fees += D(b['fee']) + D(b['tax'])
            if orders[b['order_id']]['side'] == 'buy':
                bought += value
            else:
                sold += value
        elif r['kind'] == 'delivery' and rights[b['action_id']]['action_type'] == 'cash':
            received += D(rights[b['action_id']]['amount'])
    expected = opening - bought + sold - fees + received
    actual = p.state(rows)['cash']
    return dict(opening_cash=str(opening), gross_buys=str(bought), gross_sells=str(sold),
                fees_and_tax=str(fees), delivered_cash=str(received), closing_cash_so_far=str(actual),
                difference=str(actual - expected), reconciled=actual == expected, fills=count)


def load_prices(day, ids):
    if not ids:
        return {}
    from sqlalchemy import select
    from app.db import get_session
    from app.models import RawPrice
    with get_session() as db:
        return {sid: str(price) for sid, price in db.execute(select(RawPrice.stock_id, RawPrice.close).where(
            RawPrice.stock_id.in_(ids), RawPrice.trading_date == day)) if price is not None and price > 0}


def inspect(root=policy.ROOT, clock=j.now, evidence_path=corporate.PATH, market_status=None, prices=None):
    root = Path(root)
    with file_lock(root / '.run.lock', timeout=0):
        paths = guard.verify(root)
        at = clock(); day = str(at.astimezone(p.TZ).date())
        calendar_error = None
        try:
            is_open = auto.calendar_day(at.astimezone(p.TZ).date())
        except ValueError as exc:
            is_open = None; calendar_error = str(exc)
        if market_status is None:
            from app.workbench_service import data_status
            market_status = data_status()
        records = {role: policy.verify(path, role) for role, path in paths.items()}
        records = {role: [r for r in rows if j.timestamp(r['recorded_at']) <= at] for role, rows in records.items()}
        ids = sorted({sid for rows in records.values() for sid in p.state(rows)['holdings']})
        if prices is None:
            prices = load_prices(day, ids)
        reports = {}
        evening = at.astimezone(p.TZ).time() >= time(18)
        for role, path in paths.items():
            rows = records[role]; state = p.state(rows)
            audit = corporate.inspect(path, evidence_path, lambda: at, rows=rows)
            reviewed = False
            if not audit['blocked']:
                signature = auto.approval_signature(rows, evidence_path, lambda: at)
                reviewed = any(r['kind'] == 'simulation_review' and r['body']['date'] == day
                               and r['body']['source_fingerprint'] == signature for r in rows)
            pending = [o for o in state['orders'].values() if o['session'] <= day
                       and not o['closed'] and o['filled'] < o['qty']]
            valid_prices = {sid for sid, value in prices.items()
                            if value is not None and D(str(value)).is_finite() and D(str(value)) > 0}
            missing = sorted(set(state['holdings']) - valid_prices)
            market_ready = bool(market_status['data_ready'] and market_status['price_date'] == day and not missing)
            cash = cash_check(rows, day)
            checks = [
                dict(item='官方交易日曆', status='通過' if is_open else '休市' if is_open is False else '缺件',
                     detail=calendar_error or day),
                dict(item='結算時段', status='通過' if evening else '等待', detail='18:00 後才執行每日結算'),
                dict(item='未成交委託', status='等待' if pending and at.astimezone(p.TZ).time() < time(13, 30)
                     else '待取消' if pending else '通過', detail=f'{len(pending)} 筆；13:30 後取消剩餘'),
                dict(item='今日收盤行情', status='通過' if market_ready else '缺件' if evening else '等待',
                     detail=f"行情日 {market_status['price_date']}；缺持股價格：{','.join(missing) or '無'}"),
                dict(item='公司行動來源', status='缺件' if audit['blocked'] else '通過',
                     detail='；'.join(i['message'] for i in audit['issues'] if i['blocking']) or '未發現已知衝突；非完整性保證'),
                dict(item='今日人工核對', status='通過' if reviewed else '不需持股核對' if not state['holdings'] else '需你核對',
                     detail='有持股時須核對官方公告、停復牌及權益；本清單不會代勾'),
                dict(item='帳列現金收支', status='通過' if cash['reconciled'] else '差異', detail='差額 '+cash['difference']+' 元'),
            ]
            closed = any(r['kind'] == 'close' and r['body']['date'] == day for r in rows)
            cancelled_today = {r['body']['order_id'] for r in rows if r['kind'] == 'cancel'
                               and str(j.timestamp(r['recorded_at']).astimezone(p.TZ).date()) == day}
            orders = [o for o in execution(rows) if o['session'] == day
                      and (o['filled'] or not o['cancelled'] or o['order_id'] in cancelled_today)]
            reports[role] = dict(head=rows[-1]['hash'], checks=checks, cash=cash, orders=orders,
                source_hashes=[s['hash'] for s in audit['sources']], review_current=reviewed,
                holdings=[dict(stock_id=sid, qty=h['qty']) for sid,h in state['holdings'].items()],
                already_closed=closed, preconditions_passed=bool(is_open and evening and not pending and market_ready
                    and not audit['blocked'] and (reviewed or not state['holdings']) and cash['reconciled']))
        return dict(observed_at=at.isoformat(), date=day, books=reports, source_requests=0,
                    classification='simulation_closing_preview_not_settlement', live_qualified=False,
                    note='這是讀取當下的結算前清單；不寫核對、成交、股息、淨值或隔日訊號。正式結算仍重跑完整檢查。')
