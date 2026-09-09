import numpy as np
import pandas as pd
import pytest

from skills.event_group_research import (classify_event, extract_news, peer_confirmation,
                                        build_signals, simulate_signals)
from skills.flow_research import portfolio_diagnostics


def test_event_requires_change_and_rejects_forecast_price_hype_and_denial():
    assert classify_event('南亞科虧損縮小 毛利率回升')[1] == 'accepted'
    assert classify_event('國巨正式量產出貨')[1] == 'accepted'
    for title in ['國巨出貨消息', '國巨可望取得訂單', '國巨澄清漲價傳聞',
                  '國巨量產出貨股價飆漲', '【即時新聞】國巨量產出貨',
                  '精誠內湖總部取得ISO四項認證', '联詠出貨拚年增五成']:
        assert classify_event(title)[1] != 'accepted'
    assert classify_event('崇越漲價、中鋼虧損', '中鋼')[1] != 'accepted'


def test_news_associations_and_cooldown_do_not_use_later_stock_tags():
    raw = pd.DataFrame([
        ['2330', '2022-01-03 15:00', '2026-01-01', '台積電與華邦電記憶體量產出貨', 'A', 'https://example.com/1'],
        ['2344', '2022-02-04 15:00', '2026-01-01', '台積電與華邦電記憶體量產出貨', 'A', 'https://example.com/1'],
        ['2330', '2022-01-04 16:00', '2026-01-01', '台積電記憶體正式量產', 'B', 'https://example.com/2'],
    ], columns=['stock_id', 'news_datetime', 'created_at', 'title', 'source', 'link'])
    mentions, events, rejected, _ = extract_news(raw, {'2330': '台積電', '2344': '華邦電'})
    assert mentions[mentions.stock_id.eq('2344')].available_date.min() == pd.Timestamp('2022-02-05')
    assert len(events) == 2
    assert rejected[0]['reason'] == 'same_event_kind_30d'


def test_candidate_cannot_confirm_its_own_theme_and_missing_peers_fail():
    args = ([0, 1, 2, 3], np.ones(4, dtype=bool), np.array([9., .1, .1, .1]),
            np.ones(4, dtype=bool), np.ones(4)*.02, np.ones(4)*.01, .2)
    detail = peer_confirmation(*args)[0]
    assert not detail['confirmed']
    assert detail['median_excess_20d'] == pytest.approx(-.1)
    args[1][1] = False
    assert peer_confirmation(*args)[0]['reason'] == 'insufficient_peers'


def test_future_prices_and_mentions_do_not_change_past_signals():
    days = pd.bdate_range('2021-01-01', periods=240)
    ids = ['0050', '2330', '2344', '2408', '2337', '2603']
    close = pd.DataFrame({s: 100*(1+(.0002 if s in ('0050', '2603') else .002))**np.arange(len(days)) for s in ids}, index=days)
    fields = {'adj_close': close, 'raw_close': close.copy(), 'raw_volume': close*0+10_000_000}
    mentions = pd.DataFrame([{'available_date': days[70], 'stock_id': s, 'theme': 'memory'} for s in ids[1:5]])
    event = {'stock_id': '2330', 'available_date': str(days[90].date())}
    before = build_signals(fields, mentions, [event])[0]
    for values in fields.values():
        values.iloc[150:] *= 5
    future = pd.concat([mentions, pd.DataFrame([{'available_date': days[151], 'stock_id': '2603', 'theme': 'memory'}])])
    after = build_signals(fields, future, [event])[0]
    for key in before:
        pd.testing.assert_frame_equal(before[key].iloc[:150], after[key].iloc[:150])
    assert np.isfinite(before['combined'].loc[days[90], '2330'])


def test_signals_fill_next_day_and_stop_retries_until_tradable():
    days = pd.bdate_range('2022-01-03', periods=9)
    close = pd.DataFrame({'2330': [100., 100., 80., 50., 60., 60., 60., 60., 60.]}, index=days)
    scores = close*np.nan; scores.iloc[0, 0] = 1
    flags = close.notna(); flags.iloc[3, 0] = False
    run, trades = simulate_signals(close, flags, scores, horizon=63, slippage=.0045, start=days[1])
    assert run.trades[0]['date'] == str(days[1].date())
    assert run.trades[1]['date'] == str(days[4].date())
    assert run.trades[1]['reason'] == 'entry_stop'
    assert run.summary['blocked_orders'] == 1
    assert trades[0]['net_return'] < -.4  # A gap may exceed a 15% stop.
    companies = pd.DataFrame({'stock_id': ['2330'], 'name': ['台積電'], 'industry': ['24']})
    assert portfolio_diagnostics(run, close, companies)['per_stock_pnl_sum'] == pytest.approx(run.summary['total_return'])


def test_blocked_preselected_buy_is_not_replaced_and_horizon_is_fixed():
    days = pd.bdate_range('2022-01-03', periods=70)
    close = pd.DataFrame(100., index=days, columns=[str(s) for s in range(2300, 2311)])
    scores = close*np.nan; scores.iloc[0] = np.arange(11.)
    flags = close.notna(); flags.iloc[1, 10] = False
    run, completed = simulate_signals(close, flags, scores, horizon=63, slippage=.003, start=days[1])
    buys = [r for r in run.trades if r['side'] == 'buy']
    assert len(buys) == 9 and all(r['stock_id'] != '2300' for r in buys)
    assert all(r['holding_sessions'] == 63 for r in completed)
    assert run.curve.cash.min() >= 0 and run.curve.positions.max() <= 10


def test_event_diagnostics_retain_losses_and_reject_bad_endpoint_prices():
    from scripts.research_event_groups import independent_event_outcomes, require_diagnostic_mode
    with pytest.raises(ValueError, match='only --diagnostic-only'):
        require_diagnostic_mode(False)
    days = pd.bdate_range('2022-01-03', periods=70)
    close = pd.DataFrame({'0050': 100., '2330': 100.}, index=days)
    close.iloc[64,1] = 80.
    event = {'signal_date': str(days[0].date()), 'stock_id': '2330', 'price_eligible': True}
    flags = close.notna()
    outcome = independent_event_outcomes(close, flags, [event])[0]
    assert outcome['stock_net'] < -.2 and outcome['excess'] < 0
    assert outcome['entry_date'] == str(days[1].date())
    close.iloc[64,1] = 0.
    outcome = independent_event_outcomes(close, flags, [event])[0]
    assert outcome['outcome_status'] == 'endpoint_untradable' and outcome['stock_net'] is None


def test_official_benchmark_handles_split_across_suspended_sessions():
    from scripts.research_event_groups import official_close
    days = pd.to_datetime(['2025-06-10','2025-06-11','2025-06-18'])
    raw = pd.DataFrame({'0050':[188.65,np.nan,48.], '2330':[100.,0.,102.]},index=days)
    events = pd.DataFrame(columns=['stock_id','event_date','ratio'])
    adjusted = official_close({'raw_close':raw}, events)
    assert adjusted['0050'].dropna().iloc[1]/adjusted['0050'].dropna().iloc[0]-1 == pytest.approx(48./(188.65/4)-1)
    assert pd.isna(adjusted.loc[days[1],'2330'])
    duplicate = pd.DataFrame([{'stock_id':'0050','event_date':'2025-06-18','ratio':.25}])
    with pytest.raises(ValueError,match='reconcile'):
        official_close({'raw_close':raw},duplicate)
