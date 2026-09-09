"""Past-news theme membership and explicit event hypotheses, without future labels."""
from __future__ import annotations

from collections import Counter
import hashlib
import re
import numpy as np
import pandas as pd

from skills.news_radar import PATTERNS, PRICE, clean_title, title_key, safe_link
from skills.rule_research import Simulation, metrics

RULES = {'event': '營運事件', 'group': '題材群轉強', 'combined': '營運事件＋題材群'}
EVENT_PATTERNS = {
    'pricing': re.compile(r'(?:報價|售價).{0,8}(?:回升|上漲|調漲|走揚|攀升)|調漲|漲價'),
    'orders': re.compile(r'(?:接獲|取得|拿下|獲得|獲|接).{0,10}(?:訂單|大單)|(?:訂單|接單).{0,10}(?:成長|增加|倍增|滿載)|(?:通過|取得).{0,10}客戶.{0,6}認證|客戶認證.{0,4}通過'),
    'production': re.compile(r'(?:開始|正式|啟動|進入|已).{0,6}(?:量產|投產)|量產出貨|(?:出貨|拉貨).{0,10}(?:增|成長|回升|倍)|(?:量產|投產).{0,6}(?:啟動|開始)'),
    'earnings': re.compile(r'轉盈|轉虧為盈|虧損.{0,8}(?:收斂|縮小|減少)|毛利(?:率)?.{0,8}(?:回升|改善|增加|成長|上升)'),
}
VETO = re.compile(r'下滑|衰退|取消|延後|延期|否認|澄清|未量產|未出貨|未獲單|尚未|傳聞|傳言|傳出|預期|可望|有望|將|估|預料|看好|可能|目標價|恐|拚|擬|規劃|有機會')
TEMPLATE = re.compile(r'即時新聞|投資快訊|盘前|盤前要聞|盤前掃瞄|必看財經|新聞摘要|新聞彙整|熱門股|焦點股|周漲|週漲|連\d*紅')


def classify_event(title, subject=None):
    # Conservative subject binding: another firm's event in a compound headline
    # must not be assigned to this stock. This sacrifices some headline recall.
    clauses = re.split(r'[，,；;、]', title)
    context = ' '.join(c for c in clauses if subject is None or subject in c)
    kinds = [key for key, pattern in EVENT_PATTERNS.items() if pattern.search(context)]
    reason = ('no_operating_change' if not kinds else 'template' if TEMPLATE.search(title)
              else 'expectation_or_negative' if VETO.search(title)
              else 'price_commentary' if PRICE.search(title) else 'accepted')
    return kinds, reason


def extract_news(rows, names):
    """Never merge a later stock tag into an earlier article's membership.

    A title match is only a research association, not an independently verified
    business event. Duplicate/cooldown decisions use chronological text only.
    """
    rows = rows.copy()
    rows['news_datetime'] = pd.to_datetime(rows.news_datetime)
    rows = rows.sort_values(['news_datetime', 'stock_id', 'title', 'source'], kind='stable')
    mentions, events, rejected = [], [], []
    counts, recent, seen = Counter(), {}, {}
    aliases = {s: re.sub(r'\*|-KY$', '', name) for s, name in names.items()}
    columns = ['stock_id', 'news_datetime', 'created_at', 'title', 'source', 'link']
    for sid, stamp, recorded, raw_title, source, link in rows[columns].itertuples(index=False, name=None):
        counts['prefiltered_rows'] += 1
        if sid not in aliases or pd.isna(stamp):
            counts['outside_current_cohort'] += 1
            continue
        title = clean_title(raw_title, source)
        alias = aliases[sid]
        if not ((len(alias) >= 2 and alias in title) or re.search(rf'(?<!\d){sid}(?!\d)', title)):
            counts['not_explicitly_named'] += 1
            continue
        day = stamp.normalize()
        key = (sid, title_key(title))
        if key in seen and (day-seen[key]).days <= 30:
            counts['duplicate_title_30d'] += 1
            continue
        seen[key] = day
        available = day + pd.Timedelta(days=1)
        themes = [k for k, pattern in PATTERNS.items() if pattern.search(title)]
        for theme in themes:
            mentions.append({'available_date': available, 'stock_id': sid, 'theme': theme})
        kinds, reason = classify_event(title, alias if len(alias) >= 2 and alias in title else sid)
        if not kinds:
            counts[reason] += 1
            continue
        item = {'id': hashlib.sha256(f'{sid}|{stamp}|{key[1]}'.encode()).hexdigest()[:20],
                'stock_id': sid, 'source_date': str(day.date()), 'provider_datetime': str(stamp),
                'available_date': str(available.date()), 'first_recorded_at': str(recorded),
                'title': title, 'source': str(source), 'link': safe_link(link), 'kinds': kinds,
                'headline_themes': themes, 'evidence_level': 'headline_clue_unverified'}
        if reason == 'accepted':
            fresh = [k for k in kinds if (sid, k) not in recent or (day-recent[(sid, k)]).days > 30]
            if not fresh:
                reason = 'same_event_kind_30d'
            else:
                item['kinds'] = fresh
                for kind in fresh:
                    recent[(sid, kind)] = day
        counts[reason] += 1
        if reason == 'accepted':
            events.append(item)
        else:
            rejected.append({**item, 'reason': reason})
    return pd.DataFrame(mentions, columns=['available_date', 'stock_id', 'theme']), events, rejected, dict(counts)


def peer_confirmation(member_ids, valid, ret20, above_ma, share5, share20, benchmark):
    """Return leave-one-out peer evidence. Candidate cannot confirm its own group."""
    result = {}
    good = [j for j in member_ids if valid[j]]
    for j in member_ids:
        peers = [k for k in good if k != j]
        denominator = len(member_ids)-1
        coverage = len(peers)/denominator if denominator else 0.
        if len(peers) < 3 or coverage < .8 or not np.isfinite(benchmark):
            result[j] = {'confirmed': False, 'reason': 'insufficient_peers',
                         'peer_count': len(peers), 'peer_coverage': coverage}
            continue
        excess = float(np.median(ret20[peers])-benchmark)
        breadth = float(np.mean(above_ma[peers]))
        recent, prior = float(share5[peers].sum()), float(share20[peers].sum())
        passed = excess > 0 and breadth >= .6 and recent > prior
        result[j] = {'confirmed': bool(passed), 'reason': 'confirmed' if passed else 'peers_not_strong',
                     'peer_count': len(peers), 'peer_coverage': coverage,
                     'median_excess_20d': excess, 'above_ma_fraction': breadth,
                     'share_recent5': recent, 'share_previous20': prior}
    return result


def build_signals(fields, mentions, events, *, signal_end='2025-11-30'):
    close = fields['adj_close']
    days, ids = close.index, close.columns
    if '0050' not in ids:
        raise ValueError('0050 benchmark is required')
    pos = {sid: i for i, sid in enumerate(ids)}
    themes = list(PATTERNS)
    tpos = {theme: i for i, theme in enumerate(themes)}
    turnover = fields['raw_close'] * fields['raw_volume']
    share = turnover.div(turnover.drop(columns='0050').sum(axis=1, min_count=1).replace(0, np.nan), axis=0)
    share5 = share.rolling(5, min_periods=5).mean().to_numpy()
    share20 = share.shift(5).rolling(20, min_periods=20).mean().to_numpy()
    ret = close.div(close.shift(20))-1
    above = close.gt(close.rolling(60, min_periods=60).mean()).to_numpy()
    valid = (close.rolling(60, min_periods=60).count().eq(60)
             & turnover.rolling(25, min_periods=25).count().eq(25)).to_numpy()
    liquid = turnover.rolling(20, min_periods=20).mean().ge(50_000_000).to_numpy()
    relative = ret.sub(ret['0050'], axis=0).to_numpy()
    ever_quoted = close.notna().cummax().to_numpy()
    returns = ret.to_numpy()
    benchmark = ret['0050'].to_numpy()
    updates, event_days = {}, {}
    for row in mentions.itertuples(index=False):
        if row.stock_id not in pos:
            continue
        i = int(days.searchsorted(pd.Timestamp(row.available_date)))
        updates.setdefault(i, []).append((tpos[row.theme], pos[row.stock_id], pd.Timestamp(row.available_date)))
    for event in events:
        if event['stock_id'] not in pos:
            continue
        i = int(days.searchsorted(pd.Timestamp(event['available_date'])))
        event_days.setdefault(i, []).append(event)
    last_seen = np.full((len(themes), len(ids)), np.datetime64('NaT'), dtype='datetime64[ns]')
    group_state = np.zeros(len(ids), dtype=bool)
    last_group_pulse = np.full(len(ids), np.datetime64('1900-01-01'), dtype='datetime64[ns]')
    matrices = {key: np.full(close.shape, np.nan) for key in RULES}
    annotated, group_evidence = [], []
    for i, day in enumerate(days):
        if day > pd.Timestamp(signal_end):
            break
        for t, j, stamp in updates.get(i, []):
            last_seen[t, j] = stamp.to_datetime64()
        member = (last_seen >= (day-pd.Timedelta(days=180)).to_datetime64())
        member[:, ~ever_quoted[i]] = False
        member[:, pos['0050']] = False
        confirmed = np.zeros(len(ids), dtype=bool)
        best = {}
        for t, theme in enumerate(themes):
            members = np.flatnonzero(member[t]).tolist()
            if not members:
                continue
            evidence = peer_confirmation(members, valid[i], returns[i], above[i], share5[i], share20[i], benchmark[i])
            for j, detail in evidence.items():
                detail = {**detail, 'theme': theme}
                confirmed[j] |= detail['confirmed']
                if j not in best or (detail['confirmed'], detail.get('median_excess_20d', -np.inf)) > (best[j]['confirmed'], best[j].get('median_excess_20d', -np.inf)):
                    best[j] = detail
        common = liquid[i] & above[i] & valid[i] & np.isfinite(relative[i])
        common[pos['0050']] = False
        pulse = confirmed & ~group_state & common & ((day.to_datetime64()-last_group_pulse) > np.timedelta64(30, 'D'))
        matrices['group'][i, pulse] = relative[i, pulse]
        for j in np.flatnonzero(pulse):
            group_evidence.append({'signal_date': str(day.date()), 'stock_id': str(ids[j]), **best[j]})
        last_group_pulse[pulse] = day.to_datetime64()
        group_state = confirmed
        for event in event_days.get(i, []):
            j = pos[event['stock_id']]
            detail = best.get(j, {'confirmed': False, 'reason': 'no_prior_theme', 'peer_count': 0, 'peer_coverage': 0.})
            accepted = bool(common[j])
            annotated.append({**event, 'signal_date': str(day.date()), 'price_eligible': accepted,
                              'common_exclusion': None if accepted else 'price_history_trend_or_liquidity',
                              'peer_evidence': detail})
            if accepted:
                matrices['event'][i, j] = relative[i, j]
                if confirmed[j]:
                    matrices['combined'][i, j] = relative[i, j]
    return ({key: pd.DataFrame(value, index=days, columns=ids) for key, value in matrices.items()},
            annotated, group_evidence)


def simulate_signals(close, can_trade, scores, *, horizon, slippage, start='2022-01-03'):
    """Fixed-capacity daily entry pulses; stop decisions execute next session.

    Entry selection and budget use the prior close. Failed selected buys are not
    replaced using execution-day information; pending exits keep their slots.
    """
    if horizon not in (63, 126) or not 0 <= slippage < 1:
        raise ValueError('Invalid preregistered holding/cost policy')
    if (not close.index.is_unique or not close.index.is_monotonic_increasing
            or not close.index.equals(scores.index) or not close.columns.equals(scores.columns)
            or not close.index.equals(can_trade.index) or not close.columns.equals(can_trade.columns)):
        raise ValueError('Unique, ordered, aligned matrices required')
    days, ids = close.index, close.columns
    first = int(days.searchsorted(pd.Timestamp(start)))
    if first < 1 or first >= len(days)-1:
        raise ValueError('Insufficient history')
    px, score = close.to_numpy(float), scores.to_numpy(float)
    flags = can_trade.to_numpy(bool) & np.isfinite(px) & (px > 0)
    mark = close.iloc[:first].ffill().iloc[-1].to_numpy(float)
    mark[~np.isfinite(mark)] = 0.
    positions, pending = {}, {}
    cash, cost, notional = 1., 0., 0.
    buy_cost, sell_cost = .001425+slippage, .001425+slippage+.003
    trades, decisions, completed = [], [], []
    blocked, missing = 0, 0
    curve = [{'date': days[first-1], 'equity': 1., 'cash': 1., 'positions': 0}]
    for i in range(first, len(days)):
        due = {j for j, p in positions.items() if j in pending or i >= p['entry_index']+horizon or i == len(days)-1}
        slots = 10-len(positions)+len(due)
        eligible = [j for j in np.flatnonzero(np.isfinite(score[i-1])) if j not in positions and ids[j] != '0050']
        chosen = sorted(eligible, key=lambda j: (-score[i-1, j], ids[j]))[:slots] if i < len(days)-1 else []
        for j in set(eligible)-set(chosen):
            decisions.append({'signal_date': str(days[i-1].date()), 'execution_date': str(days[i].date()),
                              'stock_id': str(ids[j]), 'result': 'capacity_or_rank_not_selected'})
        budget = curve[-1]['equity']*.1
        observed = np.isfinite(px[i]) & (px[i] > 0)
        missing += sum(not observed[j] for j in positions)
        mark[observed] = px[i, observed]
        for j in sorted(due):
            p = positions[j]
            reason = pending.setdefault(j, 'scheduled_exit')
            if not flags[i, j]:
                blocked += 1
                continue
            amount = p['units']*px[i, j]
            cash += amount*(1-sell_cost)
            cost += amount*sell_cost; notional += amount
            trades.append({'date': str(days[i].date()), 'stock_id': str(ids[j]), 'side': 'sell',
                           'notional_initial_equity': amount, 'cost_initial_equity': amount*sell_cost,
                           'reason': reason})
            completed.append({'stock_id': str(ids[j]), 'signal_date': p['signal_date'],
                              'entry_date': str(days[p['entry_index']].date()), 'exit_date': str(days[i].date()),
                              'holding_sessions': i-p['entry_index'], 'reason': reason,
                              'net_return': amount*(1-sell_cost)/p['initial_cash']-1,
                              'pnl_initial_equity': amount*(1-sell_cost)-p['initial_cash']})
            del positions[j]; del pending[j]
        for j in chosen:
            reason = ('capacity_blocked' if len(positions) >= 10 else 'buy_untradable' if not flags[i, j]
                      else 'no_cash' if cash <= 1e-12 else 'entered')
            decisions.append({'signal_date': str(days[i-1].date()), 'execution_date': str(days[i].date()),
                              'stock_id': str(ids[j]), 'result': reason})
            if reason != 'entered':
                blocked += 1
                continue
            cash_spent = min(cash, budget)
            amount = cash_spent/(1+buy_cost)
            positions[j] = {'units': amount/px[i, j], 'entry_price': px[i, j], 'peak': px[i, j],
                            'entry_index': i, 'signal_date': str(days[i-1].date()), 'initial_cash': cash_spent}
            cash -= cash_spent; cost += amount*buy_cost; notional += amount
            trades.append({'date': str(days[i].date()), 'stock_id': str(ids[j]), 'side': 'buy',
                           'notional_initial_equity': amount, 'cost_initial_equity': amount*buy_cost})
        for j, p in positions.items():
            if not observed[j]:
                continue
            p['peak'] = max(p['peak'], px[i, j])
            if px[i, j] <= p['entry_price']*.85:
                pending.setdefault(j, 'entry_stop')
            elif px[i, j] <= p['peak']*.8:
                pending.setdefault(j, 'trailing_stop')
        nav = cash+sum(p['units']*mark[j] for j, p in positions.items())
        if cash < -1e-10 or len(positions) > 10:
            raise AssertionError('Portfolio borrowed or exceeded capacity')
        curve.append({'date': days[i], 'equity': nav, 'cash': cash, 'positions': len(positions)})
    frame = pd.DataFrame(curve)
    summary = metrics(frame)
    summary.update(trades=len(trades), traded_notional_initial_equity=notional, fees_initial_equity=cost,
                   average_positions=float(frame.positions.iloc[1:].mean()), blocked_orders=blocked,
                   missing_hold_days=int(missing), unliquidated_positions=len(positions),
                   completed_trades=len(completed),
                   average_holding_sessions=float(np.mean([c['holding_sessions'] for c in completed])) if completed else None,
                   completed_win_fraction=float(np.mean([c['net_return'] > 0 for c in completed])) if completed else None,
                   annual_returns={str(y): metrics(frame, f'{y}-01-01', f'{y}-12-31')['total_return']
                                   for y in sorted(set(days[first:].year))})
    return Simulation(summary, frame, trades, decisions), completed
