from pathlib import Path
import pytest
from skills.odd_daily_cache import OddDailyCache
from skills.replay_market_feeds import ReplayDataUnavailable
from test_replay_market_feeds import feeds,odd_record,Response


def build(root,name,high='102.00'):
    source=odd_record();source['payload']['data'][0][source['payload']['fields'].index('當日最高價')]=high
    target=root/'.cache'/name
    provider=feeds(target,http_get=lambda *a,**k:Response(source['payload']))
    provider.get_odd('2022-01-04','0050','TWSE')
    return target


def test_own_market_prices_reused_and_raw_changes_rejected(tmp_path):
    primary=build(tmp_path,'one')
    cache=OddDailyCache(tmp_path,primary)
    assert cache.get_odd('2022-01-04','0050','TWSE')['odd_shares']==1200
    raw=primary/'odd-twse-2022-01-04.raw.json';raw.write_text('{}')
    with pytest.raises(ReplayDataUnavailable):OddDailyCache(tmp_path,primary).get_odd('2022-01-04','0050','TWSE')


def test_conflict_or_missing_stock_is_not_silently_zero(tmp_path):
    primary=build(tmp_path,'one');build(tmp_path,'two',high='103.00')
    cache=OddDailyCache(tmp_path,primary)
    with pytest.raises(ReplayDataUnavailable,match='Conflicting'):cache.get_odd('2022-01-04','0050','TWSE')
    with pytest.raises(ReplayDataUnavailable,match='missing'):cache.get_odd('2022-01-04','9999','TWSE')
