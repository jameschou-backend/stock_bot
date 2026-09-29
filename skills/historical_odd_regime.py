"""Explicit pre-2020 after-hours auction evidence for retrospective replay.

The single auction price is an execution proxy with a volume cap. It is not
evidence that a particular retail order won allocation in that auction.
"""
from copy import deepcopy
import math
import re

from skills.replay_market_feeds import ReplayDataUnavailable, _number

INTRADAY_START = '2020-10-26'
AFTER_URLS = {
    'twse': 'https://www.twse.com.tw/rwd/zh/afterTrading/TWT53U',
    'tpex': 'https://www.tpex.org.tw/www/zh-tw/afterTrading/odd',
}


def parse_after_hours(record, market, day):
    """Reject wrong-day, wrong-market, intraday and incomplete tables."""
    if market not in AFTER_URLS or not '2019-01-02' <= day < INTRADAY_START:
        raise ReplayDataUnavailable('Outside registered historical odd regime')
    params = dict(date=day.replace('-', '' if market == 'twse' else '/'),
                  response='json', type='ALL' if market == 'twse' else 'Daily')
    expected = dict(day=day, provider=market, url=AFTER_URLS[market],
                    params=params, http_status=200)
    if any(record.get(k) != v for k, v in expected.items()):
        raise ReplayDataUnavailable('After-hours source provenance differs')
    data = record['payload']
    if data.get('date') != day.replace('-', '') or str(data.get('stat')).lower() != 'ok':
        raise ReplayDataUnavailable('After-hours date or status differs')
    if market == 'twse':
        table = data
        title = f'{int(day[:4])-1911}年{day[5:7]}月{day[8:]}日 盤後零股交易行情單'
        fields = ['證券代號', '證券名稱', '成交股數', '成交筆數', '成交金額', '成交價',
                  '最後揭示買價', '最後揭示買量', '最後揭示賣價', '最後揭示賣量']
        if data.get('type') != 'ALL':
            raise ReplayDataUnavailable('Incomplete TWSE stock scope')
        count = data.get('total')
    else:
        if len(data.get('tables', [])) != 1 or data.get('template') != '/template/afterTrading/odd':
            raise ReplayDataUnavailable('Wrong TPEx after-hours table')
        table = data['tables'][0]
        if table.get('date') != f'{int(day[:4])-1911}/{day[5:7]}/{day[8:]}':
            raise ReplayDataUnavailable('TPEx inner table date differs')
        title = '盤後零股每日收盤行情'
        fields = ['代號', '名稱', '成交股數', '成交筆數', '成交金額', '成交價格(元)',
                  '未成交買價', '未成交買量', '未成交賣價', '未成交賣量']
        count = table.get('totalCount')
    if table.get('title') != title or table.get('fields') != fields or count != len(table['data']):
        raise ReplayDataUnavailable('After-hours schema or row count differs')
    rows = {}
    for raw in table['data']:
        if len(raw) != len(fields):
            raise ReplayDataUnavailable('After-hours row width differs')
        sid = str(raw[0]).strip()
        if not re.fullmatch(r'[0-9]{4}', sid):
            continue  # Scope is individual four-digit stocks plus benchmark 0050.
        qty = _number(raw[2], quantity=True)
        price = _number(raw[5], missing=True)
        if (qty > 0 and (price is None or price <= 0)) or (qty == 0 and price is not None):
            raise ReplayDataUnavailable('After-hours price and volume conflict')
        if sid in rows:
            raise ReplayDataUnavailable('Duplicate after-hours security')
        rows[sid] = dict(odd_high=price, odd_low=price, odd_shares=qty,
                         source_date=day, market=market, after_hours=True)
    return rows


class HistoricalOddEra:
    """Keep prior-session sizing, but label the actual dated execution window."""
    def _plan(self, day, *args, **kwargs):
        super()._plan(day, *args, **kwargs)
        if str(day.date()) < INTRADAY_START:
            plan = self.tick_plans[-1]
            plan.update(odd_order_time='13:40:00', odd_expires_at='14:30:00')
            self.day_plans[(plan['event_id'], plan['side'])].update(
                odd_order_time='13:40:00', odd_expires_at='14:30:00')

    def _execute_order(self, day, *args, **kwargs):
        ni, nt = len(self.orders), len(self.trades)
        result = super()._execute_order(day, *args, **kwargs)
        if str(day.date()) < INTRADAY_START:
            for row in [*self.orders[ni:], *self.trades[nt:]]:
                if row['channel'] == 'odd':
                    row.update(order_time='13:40:00', expires_at='14:30:00',
                               execution_evidence='after_hours_single_auction_proxy')
        return result

    def run(self):
        result = super().run()
        result['settings']['historical_odd_regime'] = dict(
            intraday_start=INTRADAY_START, earlier='after_hours_single_auction_proxy',
            sizing='previous_session_inputs_and_opening_cash_only')
        return result


def normalized_era_account(account):
    """Validate real historical times before adapting legacy algebra-only audits.

    Source prices/volumes, quantities, limits, cash and costs are never changed.
    The legacy checker knows only the 2022+ window and checks sizing algebra.
    """
    view = deepcopy(account)
    checked = 0
    for plan in view['tick_plans']:
        if plan['date'] < INTRADAY_START:
            if (plan['odd_order_time'], plan['odd_expires_at']) != ('13:40:00', '14:30:00'):
                raise ValueError('Historical plan uses unavailable intraday odd market')
            plan.update(odd_order_time='09:00:00', odd_expires_at='13:30:00')
    for key in ('orders', 'trades'):
        for row in view[key]:
            if row['channel'] != 'odd' or row['date'] >= INTRADAY_START:
                continue
            if (row['order_time'], row['expires_at'], row['execution_evidence']) != (
                    '13:40:00', '14:30:00', 'after_hours_single_auction_proxy'):
                raise ValueError('Historical odd execution window differs')
            if row['source_high'] != row['source_low']:
                raise ValueError('After-hours single auction has inconsistent price')
            row.update(order_time='09:00:00', expires_at='13:30:00',
                       execution_evidence='daily_high_low_midpoint_proxy')
            checked += key == 'orders'
    # Opening-cash sizing prevents later auction sales from financing earlier
    # board buys. Independently check chronological channel ordering too.
    for i, day in enumerate(account['daily']):
        if day['date'] >= INTRADAY_START:
            continue
        cash = account['daily'][i-1]['cash'] if i else account['settings']['initial_cash']
        cash += sum(r['cash_change'] for r in account['cash_ledger'] if r['date'] == day['date']
                    and r['kind'] not in ('initial_deposit', 'buy', 'sell'))
        rows = [r for r in account['trades'] if r['date'] == day['date']]
        for row in sorted(rows, key=lambda r: (r['channel'] == 'odd', r['sequence'])):
            cash = round(cash + row['cash_change'], 2)
            if cash < 0:
                raise ValueError('After-hours proceeds financed an earlier board trade')
        if not math.isclose(cash, day['cash'], abs_tol=.02):
            raise ValueError('Historical chronological cash failed reconciliation')
    return view, dict(after_hours_orders_checked=checked, chronological_cash_verified=True)
