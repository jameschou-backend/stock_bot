"""Fixed candidate hypotheses; inputs end at the decision close, never the fill day."""
from copy import deepcopy
import math

import numpy as np
import pandas as pd

from skills.diffusion_signals import LOOKBACK, MIN_COMMON, MIN_TURNOVER, _returns, _rolling
from skills.regime_state import build_trend

ARMS = ('original', 'cap40', 'queue', 'not_extended', 'contraction', 'failed_base', 'benchmark')


def failed_base(age, return20, relative20, below_two):
    return bool(age >= 20 and all(math.isfinite(v) for v in (return20, relative20))
                and return20 < .03 and relative20 < 0 and below_two)


def candidate_features(frames, companies):
    close, other, raw, volume = [frames[n] for n in
        ('close-official', 'close-quality', 'raw-close', 'raw-volume')]
    if not close.index.is_unique or not close.index.is_monotonic_increasing or not close.columns.is_unique:
        raise ValueError('Unique ordered candidate calendar and stock ids required')
    if any(not f.index.equals(close.index) or not f.columns.equals(close.columns) for f in frames.values()):
        raise ValueError('Candidate frames must have identical axes')
    companies = companies.set_index('stock_id')
    days, ids = close.index, list(close.columns)
    listing = np.ones(close.shape, dtype=bool)
    for j, sid in enumerate(ids):
        if sid != '0050':
            listing[:, j] = days >= companies.at[sid, 'listed_date']
    eligibility = frames['eligibility']
    if eligibility.isna().any().any():
        raise ValueError('Eligibility cannot be missing')
    listing &= eligibility.to_numpy(dtype=bool)
    price, other_price, vol, amount = [f.to_numpy(dtype=float, copy=True)
        for f in (close, other, volume, raw*volume)]
    for values in (price, other_price, vol, amount):
        values[(values <= 0) | ~listing] = np.nan
    amount[~np.isfinite(vol)] = np.nan
    benchmark = ids.index('0050')
    ret, ret_other = _returns(price), _returns(other_price)
    common = np.isfinite(ret[:, benchmark]) & np.isfinite(ret_other[:, benchmark])
    count = _rolling(common[:, None].astype(float), LOOKBACK, 'sum')[:, 0]
    incomplete = _rolling((common[:, None] & ~(np.isfinite(ret) & np.isfinite(ret_other))).astype(float), LOOKBACK, 'sum') > 0
    anomaly = _rolling(((abs(ret) > .2) | (abs(ret_other) > .2) | (abs(ret-ret_other) > .005)).astype(float), LOOKBACK, 'sum') > 0
    mature = np.zeros(close.shape, dtype=bool)
    for j, sid in enumerate(ids):
        mature[LOOKBACK:, j] = True if sid == '0050' else days[:-LOOKBACK] >= companies.at[sid, 'listed_date']
    enough = (np.arange(len(days)) >= LOOKBACK) & (count >= MIN_COMMON)
    liquid = _rolling(amount, 20)
    quality = enough[:, None] & mature & ~incomplete & ~anomaly & np.isfinite(liquid) & (liquid >= MIN_TURNOVER)
    quality &= (enough & ~anomaly[:, benchmark])[:, None] & listing
    c = pd.DataFrame(price, index=days, columns=ids)
    ma20 = c.rolling(20, min_periods=20).mean()
    r20 = c/c.shift(20)-1
    r5 = c/c.shift(5)-1
    previous = c.shift(1)
    span10 = previous.rolling(10, min_periods=10).max()/previous.rolling(10, min_periods=10).min()-1
    span40 = previous.rolling(40, min_periods=40).max()/previous.rolling(40, min_periods=40).min()-1
    return dict(close=c, ma20=ma20, relative20=r20.sub(r20['0050'], axis=0),
        quality=pd.DataFrame(quality, index=days, columns=ids),
        trend=build_trend(close['0050']).state,
        not_extended=c.le(ma20*1.15) & r5.le(.15),
        contraction=span10.le(span40*.6) & span40.gt(0))


def generate_candidates(frames, companies, entries, cutoff):
    f = candidate_features(frames, companies)
    days = f['close'].index
    original = [deepcopy(e) for e in entries if e['signal_date'] <= cutoff]
    result = {'original': original, 'cap40': deepcopy(original), 'failed_base': deepcopy(original)}
    for arm in ('not_extended', 'contraction'):
        result[arm] = [deepcopy(e) for e in original if bool(f[arm].at[pd.Timestamp(e['signal_date']), e['members'][0]])]
    queued = {}
    for e in sorted(original, key=lambda item: (item['signal_date'], item['event_id'])):
        sid = e['members'][0]
        if sid == '0050' or len(sid) != 4 or not sid.isdigit():
            raise ValueError('Only individual stock candidates permitted')
        origin = days.get_loc(pd.Timestamp(e['signal_date']))
        if origin+1 >= len(days) or str(days[origin+1].date()) != e['entry_date']:
            raise ValueError('Original signal must precede entry by one session')
        for offset in range(5):
            i = origin+offset
            if i+1 >= len(days) or str(days[i].date()) > cutoff:
                break
            day = days[i]
            if offset and not (f['quality'].at[day, sid] and f['trend'].at[day] == 'ON'
                and f['close'].at[day, sid] > f['ma20'].at[day, sid]
                and f['relative20'].at[day, sid] > 0
                and f['close'].at[day, sid] >= f['close'].iloc[origin][sid]*.95):
                continue
            item = deepcopy(e)
            if offset:
                item.update(event_id=e['event_id']+'-renew-'+str(day.date()),
                    origin_event_id=e['event_id'], origin_signal_date=e['signal_date'],
                    signal_date=str(day.date()), entry_date=str(days[i+1].date()),
                    priority=float(f['relative20'].at[day, sid]),
                    selection_reason='five-session renewed candidate; no forced rotation')
                # Prior liquidity metadata belongs to the original event. The
                # account computes current-session sizing independently.
                for key in ('liquidity_at_signal', 'liquidity_before_entry', 'trend_decision_date', 'trend_state'):
                    item.pop(key, None)
            queued[(item['entry_date'], sid)] = item
    result['queue'] = sorted(queued.values(), key=lambda e: (e['entry_date'], -e['priority'], e['event_id']))
    return result


class CandidateQuality:
    def __init__(self, *args, candidate_arm, **kwargs):
        if candidate_arm not in ARMS:
            raise ValueError('Unregistered candidate research arm')
        self.candidate_arm = candidate_arm
        self.candidate_decisions = []
        super().__init__(*args, **kwargs)
        c = self.exit_signals.adjusted_close
        self.candidate_return20 = c/c.shift(20)-1

    def corporate_day(self, day):
        income = super().corporate_day(day)
        if self.candidate_arm != 'failed_base':
            return income
        i = self.positions[day]
        previous = self.days[i-1]
        for sid, h in self.holdings.items():
            state = self.exit_states.get(h['event_id'])
            if not state or not h['qty'] or state['trigger_reason']:
                continue
            context = dict(age=i-state['entry_index'],
                return20=float(self.candidate_return20.at[previous, sid]),
                relative20=float(self.exit_signals.relative20.at[previous, sid]),
                below_two=bool(self.exit_signals.below_two.at[previous, sid]))
            if not failed_base(**context):
                continue
            signal = str(previous.date())
            state.update(trigger_reason='failed_base', signal_date=signal,
                target_date=str(day.date()), target_index=i)
            h['due_index'] = i
            self._plan(day, sid, 'sell', h['event_id'], signal,
                h['qty']//1000*1000, 0., self.opening_limit, None)
            self.candidate_decisions.append(dict(date=str(day.date()), signal_date=signal,
                stock_id=sid, event_id=h['event_id'], reason='failed_base', **context))
        return income

    def run(self):
        account = super().run()
        if self.candidate_arm != 'original':
            account['candidate_decisions'] = self.candidate_decisions
            account['settings']['candidate_arm'] = self.candidate_arm
        return account
