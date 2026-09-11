"""Versioned, append-only corporate evidence beside the frozen paper engines.

A successful/empty provider response is not proof of complete market coverage.
Only known discrepancies block automatically; manual official review remains
required. This module never creates entitlements, deliveries, orders or fills.
"""
from datetime import date, datetime, timezone
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import re

from app import forward_journal as j, forward_portfolio as p
from app.finmind import fetch_dataset, FinMindError, FinMindQuotaError

PATH = j.ROOT / '.cache/forward-validation/corporate-audit.sqlite3'
VERSION = 'corporate-review-v2'
TTL = 3600
POLICY = 'TaiwanStockDividend'
RESULT = 'TaiwanStockDividendResult'
SPLIT = 'TaiwanStockSplitPrice'
REDUCTION = 'TaiwanStockCapitalReductionReferencePrice'
PAR = 'TaiwanStockParValueChange'
DATASETS = (POLICY, RESULT, SPLIT, REDUCTION, PAR)
LABELS = dict(zip(DATASETS, ('股利政策', '除權息結果', '分割參考價', '減資參考價', '面額變更')))
LIMITATION = '同一供應商的交叉檢查，並非完整官方公告覆蓋；空資料不代表沒有事件，仍需人工核對停復牌、權益條款及實際交付。'


def account_rows(path, clock=j.now):
    if not Path(path).exists():
        raise ValueError('前向帳本尚未建立')
    with j.connection(path) as con:
        p.initialize(con, clock)
        return j.read_events(con)


def scope(rows, clock=j.now):
    today = clock().astimezone(p.TZ).date()
    s = p.state(rows)
    closes = [r['body']['date'] for r in rows if r['kind'] == 'close']
    start = min(closes) if closes else str(today)
    # Include sold positions: selling on/after ex-date does not remove entitlement.
    ids = set(s['holdings']) | {r['stock_id'] for r in s['rights'].values()}
    ids |= {s['orders'][r['body']['order_id']]['stock_id'] for r in rows if r['kind'] == 'fill'}
    ids |= {o['stock_id'] for o in s['orders'].values() if not o['closed'] and o['filled'] < o['qty']}
    for sid in ids:
        if not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid):
            raise ValueError('公司行動只接受四碼台股代號')
    future = [o['session'] for o in s['orders'].values() if not o['closed'] and o['filled'] < o['qty']]
    return dict(stock_ids=sorted(ids), start=start, end=max([str(today)] + future),
                query_start=f'{date.fromisoformat(start).year - 1}-01-01', query_end=str(today))


def specs(sc):
    return [(ds, sid if ds != PAR else '', sc['query_start'], sc['query_end'])
            for ds in DATASETS for sid in (sc['stock_ids'] if ds != PAR else ([''] if sc['stock_ids'] else []))]


def source_history(evidence_path=PATH):
    if not Path(evidence_path).exists():
        return []
    with j.connection(evidence_path) as con:
        rows = j.read_events(con)
    return [r for r in rows if r['kind'] == 'corporate_source']


def latest(evidence_path=PATH):
    return {tuple(r['body']['query']): r for r in source_history(evidence_path)}


def fresh(event, clock=j.now, ttl=TTL):
    if not event:
        return False
    age = (clock() - j.timestamp(event['body']['retrieved_at'])).total_seconds()
    return 0 <= age <= ttl


def refresh(path=p.PATH, evidence_path=PATH, clock=j.now, request_budget=12):
    """At most 12 sequential shared-quota calls; persist each response for resume.

    Par-value changes use the documented all-market query (data_id gives HTTP400).
    Raw dates/rows are retained; no provider dates become historical observations.
    """
    if type(request_budget) is not int or not 1 <= request_budget <= 12:
        raise ValueError('每批查詢上限必須介於1至12')
    from app.config import load_config
    sc = scope(account_rows(path, clock), clock)
    cached = latest(evidence_path)
    calls, reused = 0, 0
    token = None
    for query in specs(sc):
        old = cached.get(query)
        if fresh(old, clock, TTL if old and old['body']['status'] == 'ok' else 60):
            reused += 1
            continue
        if calls >= request_budget:
            break
        if token is None:
            token = load_config().finmind_token
        ds, sid, start, end = query
        body = dict(version=VERSION, query=list(query), status='ok',
                    retrieved_at=clock().isoformat(), rows=[], error=None)
        quota_error = None
        try:
            frame = fetch_dataset(ds, date.fromisoformat(start), date.fromisoformat(end),
                token=token, data_id=sid or None, max_retries=0, timeout=20, cache_ttl=TTL)
            body['rows'] = json.loads(frame.to_json(orient='records', date_format='iso'))
            body['retrieved_at'] = datetime.fromtimestamp(frame.attrs['retrieved_at'], timezone.utc).isoformat()
        except FinMindError as exc:
            body.update(status='error', error=str(exc))
            if isinstance(exc, FinMindQuotaError):
                quota_error = exc
        calls += 1
        with j.connection(evidence_path) as con:
            j.append(con, 'corporate_source:' + j.digest(body), 'corporate_source', body, clock)
        if quota_error:
            raise quota_error
    return dict(calls=calls, reused=reused, report=inspect(path, evidence_path, clock))


def number(value):
    if value is None or value == '' or isinstance(value, bool):
        raise ValueError('缺少必要數字')
    result = D(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError('數字須有限且非負')
    return result


def day(value, optional=False):
    if optional and value in (None, ''):
        return None
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('日期格式錯誤')
    date.fromisoformat(value)
    return value


def before_ex(rows, ex):
    """Account just before ex-date, including previously delivered split shares."""
    prior = []
    for r in rows:
        b, kind = r['body'], r['kind']
        effective = (str(j.timestamp(b['executed_at']).astimezone(p.TZ).date()) if kind == 'fill'
                     else b['ex_date'] if kind == 'entitlement' else b['date'] if kind == 'delivery' else None)
        if effective is None or effective < ex:
            prior.append(r)
    return p.state(prior)


def inspect(path=p.PATH, evidence_path=PATH, clock=j.now, *, rows=None, resolve=True):
    rows = account_rows(path, clock) if rows is None else rows
    sc, s = scope(rows, clock), p.state(rows)
    history = source_history(evidence_path)
    sources = {tuple(r['body']['query']): r for r in history}
    status, issues, events = [], [], []
    today = str(clock().astimezone(p.TZ).date())
    parsed = {sid: {ds: [] for ds in DATASETS} for sid in sc['stock_ids']}

    def issue(sid, ex, code, message, blocking=True):
        item = dict(stock_id=sid, date=ex, code=code, message=message, blocking=blocking)
        if item not in issues:
            issues.append(item)

    for query in specs(sc):
        ds, sid, _, _ = query
        event = sources.get(query)
        ok = bool(event and fresh(event, clock) and event['body']['status'] == 'ok')
        state = ('尚未抓取' if not event else '已過期' if not fresh(event, clock)
                 else '查詢失敗' if event['body']['status'] != 'ok' else '已取得（非完整性保證）')
        status.append(dict(dataset=LABELS[ds], stock_id=sid or '全市場', status=state,
                           rows=len(event['body']['rows']) if event else None,
                           retrieved_at=event['body']['retrieved_at'] if event else None,
                           hash=event['hash'] if event else None))
        if not ok:
            issue(sid or '全市場', today, 'source_unavailable', LABELS[ds] + '：' + state)
            continue
        raw = list(event['body']['rows'])
        # Retain previously observed terms even when today's provider deletes or
        # revises them. Corrections need reconciliation, not silent replacement.
        for old in history:
            q, b = old['body']['query'], old['body']
            if (q[:3] == list(query[:3]) and q[3] <= query[3] and b['status'] == 'ok'
                    and j.timestamp(b['retrieved_at']) <= clock()):
                for item in b['rows']:
                    if item not in raw:
                        raw.append(item)
        try:
            for row in raw:
                if not isinstance(row, dict) or not isinstance(row.get('stock_id'), str):
                    raise ValueError('缺少股票代號')
                if sid and row['stock_id'] != sid:
                    raise ValueError('回傳其他股票')
                if row['stock_id'] not in parsed:
                    continue  # Documented full-market par-value response, including non-four-digit ETFs.
                observed = day(row.get('date'))
                if not query[2] <= observed <= query[3]:
                    raise ValueError('來源日期超出查詢範圍')
                parsed[row['stock_id']][ds].append(row)
        except (ValueError, TypeError, KeyError) as exc:
            issue(sid or '全市場', today, 'invalid_schema', LABELS[ds] + '：' + str(exc))

    known_cash = set()
    for sid, data in parsed.items():
        cash, results = {}, {}
        try:
            for row in data[POLICY]:
                amount = number(row.get('CashEarningsDistribution')) + number(row.get('CashStatutorySurplus'))
                stock = number(row.get('StockEarningsDistribution')) + number(row.get('StockStatutorySurplus'))
                capital = number(row.get('TotalNumberOfCashCapitalIncrease'))
                ex, pay = day(row.get('CashExDividendTradingDate'), True), day(row.get('CashDividendPaymentDate'), True)
                stock_ex = day(row.get('StockExDividendTradingDate'), True)
                if amount and not ex:
                    issue(sid, '', 'undated_policy', '現金股利尚缺除息日，需確認是否影響持有期間')
                if ex and sc['start'] <= ex <= sc['end'] and amount:
                    if pay and pay < ex:
                        raise ValueError('股息發放日早於除息日')
                    value = (amount, pay)
                    if ex in cash and cash[ex] != value:
                        issue(sid, ex, 'policy_revision', '同日股利有不同版本，不自行選用最新一筆')
                    cash[ex] = value
                if stock or capital:
                    if not stock_ex or sc['start'] <= stock_ex <= sc['end']:
                        issue(sid, stock_ex or '', 'unsupported_rights', '股票股利或增資需完整條款；目前不自動處理')
            for row in data[RESULT]:
                ex = day(row['date'])
                if not sc['start'] <= ex <= sc['end']:
                    continue
                kind = row.get('stock_or_cache_dividend')
                if kind not in ('息', '除息'):
                    issue(sid, ex, 'unsupported_result', '除權或混合權益結果，需人工對帳')
                    continue
                amount = number(row.get('stock_and_cache_dividend'))
                if ex in results and results[ex] != amount:
                    issue(sid, ex, 'result_revision', '同日除息結果有不同版本')
                results[ex] = amount
            for ex in sorted(cash.keys() | results.keys()):
                known_cash.add((sid, ex))
                amount, pay = cash.get(ex, (None, None))
                eligible = before_ex(rows, ex)['holdings'].get(sid, {}).get('qty', 0) if ex <= today else None
                match = next((r for r in s['rights'].values() if r['stock_id'] == sid and r['ex_date'] == ex and r['action_type'] == 'cash'), None)
                event = dict(stock_id=sid, ex_date=ex, cash_per_share=str(amount) if amount is not None else None,
                             payment_date=pay, eligible_qty=eligible, paid=bool(match and match['paid']))
                events.append(event)
                if amount is None or (ex <= today and ex not in results):
                    issue(sid, ex, 'unpaired_dividend', '股利政策與除息結果尚未成對')
                elif ex in results and abs(amount - results[ex]) > D('0.000001'):
                    issue(sid, ex, 'amount_conflict', '股利政策與除息結果金額不一致')
                if not pay:
                    issue(sid, ex, 'payment_unknown', '股息發放日未確認')
                if eligible and amount is not None:
                    if not match:
                        issue(sid, ex, 'missing_entitlement', '應收股息漏登；需先對帳，不可補造過去事件')
                    elif (number(match['cash_per_share']) != amount or match['eligible_qty'] != eligible
                          or number(match['amount']) != amount * eligible or match['delivery_date'] != pay):
                        issue(sid, ex, 'ledger_conflict', '帳本股息、股數或交付日與來源不符')
                elif eligible == 0 and match:
                    issue(sid, ex, 'unexpected_entitlement', '除息前未持有卻登錄了股息')
                if match and not match['paid'] and pay and pay <= today:
                    issue(sid, ex, 'payment_due', '已到公告發放日，等待實際交付證據；不自動增加現金', False)
                if ex > today:
                    issue(sid, ex, 'upcoming_action', '已知將除息，下一交易日委託參考價需另行核對', False)
            for ds in (SPLIT, REDUCTION, PAR):
                for row in data[ds]:
                    ex = day(row['date'])
                    if sc['start'] <= ex <= sc['end']:
                        # Prices do not establish exact share ratio, payable date, or fractional settlement.
                        issue(sid, ex, 'structural_action', LABELS[ds] + '有事件；須官方條款核對，不由價差推算股數')
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            issue(sid, '', 'invalid_terms', '公司行動欄位無法核對：' + str(exc))
    for r in s['rights'].values():
        if r['action_type'] == 'cash' and (r['stock_id'], r['ex_date']) not in known_cash:
            issue(r['stock_id'], r['ex_date'], 'unmatched_right', '已登錄股息缺少對應來源')
    report = dict(version=VERSION, scope=sc, sources=status, events=events, issues=issues,
                blocked=any(i['blocking'] for i in issues), coverage_complete=False,
                manual_review_required=bool(sc['stock_ids']), limitation=LIMITATION,
                account_head=rows[-1]['hash'] if rows else None)
    if resolve:
        from app.forward_corporate_resolution import apply
        return apply(report, rows, evidence_path, clock)
    return report


def capture_close(path=p.PATH, actions_reviewed=False, clock=j.now, evidence_path=PATH):
    """Guard and seal in the SAME account transaction. No frozen engine edits."""
    from sqlalchemy import select
    from app.db import get_session
    from app.models import RawPrice, TradingCalendar
    from app.workbench_service import data_status
    from app import forward_comparison as comparison
    status = data_status()
    today = str(clock().astimezone(p.TZ).date())
    if not status['data_ready'] or status['price_date'] != today:
        raise ValueError('今日市場資料尚未完整，不能補写前向結算')
    with j.connection(path) as con:
        p.initialize(con, clock)
        rows = j.read_events(con)
        if any(r['kind'] == 'comparison_anchor' for r in rows):
            comparison.protocol(con, 'benchmark', clock)
        existing = next((r for r in rows if r['kind'] == 'close' and r['body']['date'] == today), None)
        if existing:
            return existing
        audit = inspect(path, evidence_path, clock, rows=rows)
        if audit['blocked']:
            raise ValueError('公司行動待核對：' + '；'.join(i['message'] for i in audit['issues'] if i['blocking']))
        if audit['manual_review_required'] and not actions_reviewed:
            raise ValueError('自動資料不代表完整覆蓋，仍需核對官方公告及實際權益')
        s = p.state(rows)
        with get_session() as db:
            sessions = [str(d) for d in db.scalars(select(TradingCalendar.trading_date).where(
                TradingCalendar.is_open.is_(True)).order_by(TradingCalendar.trading_date))]
            prices = {str(sid): str(price) for sid, price in db.execute(select(RawPrice.stock_id, RawPrice.close).where(
                RawPrice.stock_id.in_(list(s['holdings']) + ['0050']), RawPrice.trading_date == today)) if price is not None and price > 0}
        p.submit(con, dict(kind='calendar', id=today, sessions=sessions, source='DB/trading_calendar'), clock)
        body = dict(date=today, prices=prices, actions_reviewed=True,
                    source='DB/raw_prices; corporate-audit:' + j.digest(audit))
        nav = p.valuation({**s, 'mark': {'body': body}})
        if nav is None:
            raise ValueError('缺少持股價格或有未交付分割，不能封存完整資產')
        j.append(con, 'corporate_review:' + today, 'corporate_review', dict(
            version=VERSION, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            report=audit, manual_review=actions_reviewed), clock)
        return p.submit(con, dict(kind='close', id=today, **body, nav=str(nav)), clock)
