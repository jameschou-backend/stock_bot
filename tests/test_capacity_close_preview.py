from datetime import timedelta

import pytest

from app import capacity_close_preview as preview, capacity_forward as cap
from app import capacity_source_guard as guard, forward_automation as auto
from app import forward_corporate_audit as corporate, forward_journal as j
from app.file_lock import file_lock
from tests.test_capacity_source_guard import setup, quote
from tests.test_forward_corporate_audit import sources
from tests.test_forward_portfolio import clock


def filled(tmp_path):
    root, _, evidence, now = setup(tmp_path)
    guard.match(root, 'strategy', quote(), lambda: now, evidence)
    guard.match(root, 'strategy', quote(20, 21000), lambda: now + timedelta(seconds=20), evidence)
    return root, evidence, now


def read(root, evidence, at, prices=None):
    return preview.inspect(root, lambda: at, evidence,
                           dict(data_ready=True, price_date='2026-09-14'),
                           {'1101': '100'} if prices is None else prices)


def status(book, item):
    return next(c['status'] for c in book['checks'] if c['item'] == item)


def test_intraday_is_read_only_and_excludes_future_fills(tmp_path, monkeypatch):
    root, evidence, now = filled(tmp_path)
    monkeypatch.setattr(auto, 'calendar_day', lambda day: True)
    monkeypatch.setattr(corporate, 'refresh', lambda *a, **k: pytest.fail('preview cannot fetch'))
    before = {role: cap.verify(root / (role + '.sqlite3')) for role in cap.ROLES}
    old = read(root, evidence, now)['books']['strategy']
    assert old['cash']['fills'] == 0 and not old['holdings']
    result = read(root, evidence, now + timedelta(seconds=20))
    book = result['books']['strategy']
    assert book['cash']['fills'] == 1 and book['cash']['reconciled']
    assert status(book, '今日人工核對') == '需你核對'
    assert status(book, '結算時段') == '等待' and not book['preconditions_passed']
    assert result['source_requests'] == 0 and not result['live_qualified']
    assert before == {role: cap.verify(root / (role + '.sqlite3')) for role in cap.ROLES}


def test_evening_needs_cancel_prices_and_matching_human_review(tmp_path, monkeypatch):
    root, evidence, _ = filled(tmp_path)
    at = clock('2026-09-14', 18)()
    monkeypatch.setattr(auto, 'calendar_day', lambda day: True)
    path = root / 'strategy.sqlite3'
    sources(path, evidence, at=lambda: at)
    book = read(root, evidence, at)['books']['strategy']
    assert status(book, '未成交委託') == '待取消'
    auto._cancel_day(path, lambda: at)
    auto.approve(path, 'fixture reviewer', 'synthetic test only; not a real review', evidence, lambda: at)
    book = read(root, evidence, at)['books']['strategy']
    assert book['preconditions_passed'] and book['review_current']
    for invalid in ({}, {'1101': 0}, {'1101': 'NaN'}):
        book = read(root, evidence, at, invalid)['books']['strategy']
        assert status(book, '今日收盤行情') == '缺件' and not book['preconditions_passed']


def test_expired_sources_cannot_keep_review_green(tmp_path, monkeypatch):
    root, evidence, _ = filled(tmp_path)
    at = clock('2026-09-14', 18)()
    monkeypatch.setattr(auto, 'calendar_day', lambda day: True)
    path = root / 'strategy.sqlite3'
    auto._cancel_day(path, lambda: at)
    sources(path, evidence, at=lambda: at)
    auto.approve(path, 'fixture', 'synthetic test only; not real approval', evidence, lambda: at)
    book = read(root, evidence, at + timedelta(seconds=3601))['books']['strategy']
    assert status(book, '公司行動來源') == '缺件'
    assert not book['review_current'] and not book['preconditions_passed']


def test_missing_calendar_and_busy_runner_are_not_success(tmp_path, monkeypatch):
    root, evidence, now = filled(tmp_path)
    def missing(day):
        raise ValueError('缺少官方交易日曆')
    monkeypatch.setattr(auto, 'calendar_day', missing)
    book = read(root, evidence, now)['books']['strategy']
    assert status(book, '官方交易日曆') == '缺件' and not book['preconditions_passed']
    with file_lock(root / '.run.lock'):
        with pytest.raises(TimeoutError):
            read(root, evidence, now)


def test_cash_reconciles_sales_and_only_delivered_dividends():
    def event(kind, body, day):
        return dict(kind=kind, body=body, recorded_at=clock(day)().isoformat(), hash=kind)
    rows = [
        event('order', dict(order_id='buy', stock_id='1101', side='buy'), '2026-09-11'),
        event('fill', dict(order_id='buy', price='100', qty=1000, fee='20', tax='0',
                           executed_at=clock('2026-09-11')().isoformat()), '2026-09-11'),
        event('entitlement', dict(action_id='div', stock_id='1101', action_type='cash',
                                 cash_per_share='2', amount='2000'), '2026-09-14'),
        event('order', dict(order_id='sell', stock_id='1101', side='sell'), '2026-09-14'),
        event('fill', dict(order_id='sell', price='110', qty=500, fee='20', tax='165',
                           executed_at=clock('2026-09-14')().isoformat()), '2026-09-14'),
    ]
    pending = preview.cash_check(rows, '2026-09-14')
    assert pending['delivered_cash'] == '0' and pending['closing_cash_so_far'] == '954795'
    rows.append(event('delivery', dict(action_id='div'), '2026-09-14'))
    paid = preview.cash_check(rows, '2026-09-14')
    assert paid['closing_cash_so_far'] == '956795' and paid['fees_and_tax'] == '185'
    assert paid['reconciled'] and paid['delivered_cash'] == '2000'
