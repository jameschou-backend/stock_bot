from skills.replay_market_feeds import ReplayDataUnavailable
from copy import deepcopy
import pytest

from skills.historical_odd_regime import (
    AFTER_URLS, parse_after_hours, normalized_era_account,
)


def source(market):
    tw = market == 'twse'
    fields = (['證券代號', '證券名稱', '成交股數', '成交筆數', '成交金額', '成交價',
               '最後揭示買價', '最後揭示買量', '最後揭示賣價', '最後揭示賣量'] if tw else
              ['代號', '名稱', '成交股數', '成交筆數', '成交金額', '成交價格(元)',
               '未成交買價', '未成交買量', '未成交賣價', '未成交賣量'])
    table = dict(fields=fields, data=[['2330', 'Test', '1,234', '3', '123,400', '100', '', '', '', '']],
                 title='108年01月02日 盤後零股交易行情單' if tw else '盤後零股每日收盤行情')
    if tw:
        payload = dict(table, date='20190102', stat='OK', type='ALL', total=1)
    else:
        payload = dict(date='20190102', stat='ok', template='/template/afterTrading/odd',
                       tables=[dict(table, date='108/01/02', totalCount=1)])
    return dict(day='2019-01-02', provider=market, url=AFTER_URLS[market], http_status=200,
                params=dict(date='20190102' if tw else '2019/01/02', response='json',
                            type='ALL' if tw else 'Daily'), payload=payload)


@pytest.mark.parametrize('market', ['twse', 'tpex'])
def test_single_auction_not_invented_intraday_range(market):
    result = parse_after_hours(source(market), market, '2019-01-02')['2330']
    assert result['odd_high'] == result['odd_low'] == 100
    assert result['odd_shares'] == 1234


@pytest.mark.parametrize('field,value', [
    ('day', '2019-01-03'), ('provider', 'tpex'), ('url', 'https://example.com'),
    ('params', {'date': '20190102'}), ('http_status', 403),
])
def test_wrong_source_cannot_masquerade_as_auction(field, value):
    data = source('twse'); data[field] = value
    with pytest.raises(ReplayDataUnavailable):
        parse_after_hours(data, 'twse', '2019-01-02')


@pytest.mark.parametrize('mutation', ['date', 'title', 'count', 'width', 'duplicate', 'no_price'])
@pytest.mark.parametrize('market', ['twse', 'tpex'])
def test_malformed_or_wrong_table_fails_closed(market, mutation):
    data = source(market); payload = data['payload']
    table = payload if market == 'twse' else payload['tables'][0]
    if mutation == 'date': payload['date'] = '20190103'
    elif mutation == 'title': table['title'] = '盤中零股'
    elif mutation == 'count': table['total' if market == 'twse' else 'totalCount'] = 9
    elif mutation == 'width': table['data'][0].pop()
    elif mutation == 'duplicate':
        table['data'].append(deepcopy(table['data'][0]))
        table['total' if market == 'twse' else 'totalCount'] = 2
    elif mutation == 'no_price': table['data'][0][5] = '--'
    with pytest.raises(ReplayDataUnavailable): parse_after_hours(data, market, '2019-01-02')


def account():
    row = dict(date='2019-01-02', channel='odd', order_time='13:40:00', expires_at='14:30:00',
               execution_evidence='after_hours_single_auction_proxy', source_high=100, source_low=100,
               cash_change=-10, sequence=1)
    return dict(settings=dict(initial_cash=100), tick_plans=[dict(date='2019-01-02',
                odd_order_time='13:40:00', odd_expires_at='14:30:00')], orders=[row], trades=[dict(row)],
                daily=[dict(date='2019-01-02', cash=90)], cash_ledger=[])


def test_normalization_only_changes_legacy_window_annotations():
    data = account(); before = deepcopy(data)
    result, audit = normalized_era_account(data)
    assert data == before
    assert result['trades'][0]['source_high'] == 100
    assert result['trades'][0]['cash_change'] == -10
    assert audit['after_hours_orders_checked'] == 1


def test_impossible_intraday_2019_timestamp_is_rejected():
    data = account(); data['orders'][0]['order_time'] = '09:00:00'
    with pytest.raises(ValueError): normalized_era_account(data)


def test_later_auction_proceeds_cannot_finance_earlier_board_buy():
    data = account(); data['trades'][0]['cash_change'] = 100
    data['trades'].append(dict(data['trades'][0], channel='board', cash_change=-150, sequence=2))
    data['daily'][0]['cash'] = 50
    with pytest.raises(ValueError, match='financed'): normalized_era_account(data)
