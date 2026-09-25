"""Dated universe overlays and execution guards, without changing sealed research."""
from collections import defaultdict
from copy import deepcopy

import pandas as pd
from scripts.prepare_million_signals import _liquidity
from skills.historical_diffusion_signals import build_diffusion
from skills.regime_state import build_trend, gate_events
from skills.historical_universe_completion import resolve_completion
from skills.board_only_verified_replay import BoardOnlyVerifiedReplay
from skills.replay_market_feeds import ReplayDataUnavailable


def eligibility_matrix(report, companies, days):
    """Known legal identities and explicit exclusions, not proof of full PIT coverage.

    Missing price rows never alter this expected-stock denominator. Unknown
    identities are a preparation error, not an excuse to shrink the universe.
    """
    ids = list(companies.stock_id) + ['0050']
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate historical cohort')
    days = pd.DatetimeIndex(days)
    if (days.min() < pd.Timestamp(report['coverage_start'])
            or days.max() > pd.Timestamp(report['coverage_end'])):
        raise ValueError('Calendar is outside verified identity coverage')
    episodes = defaultdict(list)
    for row in report['episodes']:
        episodes[row['stock_id']].append(row)
    mask = pd.DataFrame(False, index=days, columns=ids)
    venue_masks = {}
    for sid in ids:
        if not episodes[sid] or any(e['start'] is None for e in episodes[sid]):
            raise ValueError('Historical identity unresolved: ' + sid)
        for episode in episodes[sid]:
            if episode['category'] != ('ETF' if sid == '0050' else '股票'):
                continue
            start, end = pd.Timestamp(episode['start']), episode['end']
            valid = (days >= start) & (True if end is None else days < pd.Timestamp(end))
            if episode.get('snapshot_date'):
                valid &= days <= pd.Timestamp(episode['snapshot_date'])
            if (mask[sid].to_numpy() & valid).any():
                raise ValueError('Overlapping historical identities: ' + sid)
            mask.loc[valid, sid] = True
            key = (sid, episode['market'].upper())
            venue_masks[key] = venue_masks.get(key, pd.Series(False, index=days)) | valid
    for row in report['trading_exclusions']:
        sid = row['stock_id']
        if sid not in mask:
            continue
        valid = (days >= pd.Timestamp(row['start'])) & (
            True if row['end'] is None else days < pd.Timestamp(row['end']))
        valid &= venue_masks.get((sid, row['market'].upper()), pd.Series(False, index=days)).to_numpy()
        mask.loc[valid, sid] = False
    return mask


def build_signals(frames, companies, eligibility=None, *, start='2022-01-03', signal_end='2026-09-08'):
    close, other = frames['close-official'], frames['close-quality']
    raw, volume = frames['raw-close'], frames['raw-volume'].copy()
    volume.attrs['unit'] = 'shares'
    turnover = raw * volume
    turnover.attrs['unit'] = 'TWD'
    result = build_diffusion(close, other, volume, turnover, companies,
                            start=start, signal_end=signal_end, eligibility=eligibility)
    accepted, rejected = gate_events(result['entries']['leader_now'], build_trend(close['0050']))
    originals = {row['event_id']: row for row in result['events']}
    for batch in (accepted, rejected):
        for row in batch:
            original = originals[row['event_id']]
            for key in ('group_id', 'group_cutoff_date'):
                row[key] = original[key]
            row['group_members'] = deepcopy(original['members'])
            row['selection_reason'] = ('60-session breakout, positive excess20, volume expansion, '
                                       'low peer breadth; original signal trend ON required')
            row['leader_evidence'] = {key: deepcopy(original[key]) for key in (
                'leader_peer_breadth', 'leader_return20', 'benchmark_return20',
                'leader_volume_ratio', 'leader_turnover_share')}
            sid = row['members'][0]
            i = close.index.get_loc(pd.Timestamp(row['entry_date']))
            row['liquidity_at_signal'] = _liquidity(volume, raw, row['signal_date'], sid)
            row['liquidity_before_entry'] = _liquidity(volume, raw, close.index[i - 1], sid)
    return dict(entries=accepted, rejections=rejected, diffusion=result,
                cohort_limitation='known_reconstructed_cohort_not_complete_historical_market',
                live_qualified=False, unseen_validation=False)


class HistoricalBoardReplay(BoardOnlyVerifiedReplay):
    """Reject known suspended orders; never liquidate a vanished identity by assumption."""
    def __init__(self, *args, identity_report, identity_stock_ids=None, **kwargs):
        self.identity_stock_ids = None if identity_stock_ids is None else set(identity_stock_ids)
        self.identity_by_sid = defaultdict(lambda: dict(episodes=[], trading_exclusions=[],
            coverage_start=identity_report['coverage_start'], coverage_end=identity_report['coverage_end']))
        for row in identity_report['episodes']:
            self.identity_by_sid[row['stock_id']]['episodes'].append(row)
        for row in identity_report['trading_exclusions']:
            self.identity_by_sid[row['stock_id']]['trading_exclusions'].append(row)
        self.identity_decisions = []
        super().__init__(*args, **kwargs)

    def identity(self, day, sid):
        return resolve_completion(self.identity_by_sid[sid], sid, str(day.date()))

    def corporate_day(self, day):
        for sid, holding in self.holdings.items():
            if not holding['qty']:
                continue
            if self.identity_stock_ids is not None and sid not in self.identity_stock_ids:
                continue
            identity = self.identity(day, sid)
            if identity['status'] not in ('identified', 'official_trading_suspension', 'not_general_board'):
                raise ReplayDataUnavailable(f'Held stock lacks dated settlement/identity: {sid} {day.date()}')
            self.markets[sid] = identity['market'].upper()
        return super().corporate_day(day)

    def _execute_order(self, day, sid, side, qty, reason, event_id, signal_date=None):
        if self.identity_stock_ids is not None and sid not in self.identity_stock_ids:
            return super()._execute_order(day, sid, side, qty, reason, event_id, signal_date)
        identity = self.identity(day, sid)
        if identity['status'] in ('official_trading_suspension', 'not_general_board'):
            # Preserve the parent resource reservation and board rounding audit.
            # Quotes for known halts are masked during preparation, so parent
            # execution records a zero fill. Never silently ignore a quote here.
            if self.raw(day, sid) is not None:
                raise ValueError('Known ineligible day still has an executable quote')
            self.identity_decisions.append(dict(date=str(day.date()), stock_id=sid,
                event_id=event_id, side=side, status=identity['status'], filled_qty=0))
        elif identity['status'] != 'identified' or identity['category'] != ('ETF' if sid == '0050' else '股票'):
            raise ReplayDataUnavailable(f'Order identity unresolved/ineligible: {sid} {day.date()}')
        self.markets[sid] = identity['market'].upper()
        return super()._execute_order(day, sid, side, qty, reason, event_id, signal_date)
