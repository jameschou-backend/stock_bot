"""Causal chip features and account hooks for the fixed September experiment."""
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
from skills.technical_replay import TechnicalReplay

FILTERS = ('trust', 'foreign', 'price', 'trust_price', 'foreign_price',
           'holder', 'margin', 'holder_margin', 'holder14_margin', 'sbl', 'broker', 'group_flow')


def holder_rows(raw):
    """Discard totals/adjustments, require all 15 disjoint reported buckets."""
    records = []
    if raw.empty:
        return pd.DataFrame(columns=['date', 'stock_id', 'large_pct', 'valid'])
    for (day, sid), group in raw.groupby(['date', 'stock_id']):
        buckets, valid = {}, True
        for row in group.to_dict('records'):
            label = str(row['HoldingSharesLevel']).replace(',', '').strip().lower()
            if label == 'total' or label.startswith('差異數調整'):
                continue
            match = re.fullmatch(r'(\d+)-(\d+)', label)
            upper = re.fullmatch(r'(?:more than|over)\s*(\d+)', label)
            if not match and not upper:
                valid = False
                continue
            low = int((match or upper).group(1))
            pct = float(row['percent'])
            if low in buckets or not math.isfinite(pct) or not 0 <= pct <= 100:
                valid = False
            buckets[low] = pct
        expected = {1, 1000, 5001, 10001, 15001, 20001, 30001, 40001,
                    50001, 100001, 200001, 400001, 600001, 800001, 1000001}
        valid = valid and set(buckets) == expected and 99.5 <= sum(buckets.values()) <= 100.5
        records.append(dict(date=pd.Timestamp(day), stock_id=sid, valid=valid,
            large_pct=sum(v for k, v in buckets.items() if k >= 400001) if valid else np.nan))
    return pd.DataFrame(records)


def broker_concentration(raw):
    """Net by branch first, never count multiple price rows as multiple branches."""
    if raw.empty:
        return None
    needed = {'securities_trader_id', 'buy', 'sell'}
    if not needed.issubset(raw) or raw.securities_trader_id.isna().any():
        return None
    values = raw[['buy', 'sell']].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(values).all().all() or (values < 0).any().any():
        return None
    grouped = values.groupby(raw.securities_trader_id).sum()
    buy, sell = grouped.buy.sum(), grouped.sell.sum()
    if buy <= 0 or abs(buy-sell) > max(1, buy*.001):
        return None
    net = grouped.buy-grouped.sell
    return float(net.clip(lower=0).nlargest(5).sum()/buy)


def tri_and(*values):
    # All required features must be observed even if another condition is false.
    return None if any(v is None for v in values) else all(values)


class ChipSignals:
    def __init__(self, data, directory):
        self.days, self.technical = data.days, data.features
        columns = data.features.adjusted_close.columns
        directory = Path(directory)
        self.matrices = {}
        def matrix(frame, field):
            if frame.empty:
                return pd.DataFrame(np.nan, index=self.days, columns=columns)
            if frame.duplicated(['date', 'stock_id']).any():
                raise ValueError('Duplicate chip daily observation')
            return frame.pivot(index='date', columns='stock_id', values=field).reindex(index=self.days, columns=columns).astype(float)
        inst = pd.read_parquet(directory/'institutional_verified.parquet')
        margin = pd.read_parquet(directory/'margin_verified.parquet')
        volume = data.features.raw_fields['volume'].where(data.features.valid_volume)
        v5 = volume.rolling(5, min_periods=5).sum().replace(0, np.nan)
        v20 = volume.rolling(20, min_periods=20).sum().shift(5).replace(0, np.nan)
        flows = {}
        for actor in ('trust', 'foreign'):
            net = matrix(inst, actor+'_net')
            net = net.where(np.isfinite(net))
            recent = net.rolling(5, min_periods=5).sum()/v5
            prior = net.rolling(20, min_periods=20).sum().shift(5)/v20
            # Missing days cannot become zero buying days.
            buys = net.gt(0).astype(float).where(net.notna()).rolling(5, min_periods=5).sum()
            self.matrices[actor+'_ratio5'] = recent
            self.matrices[actor+'_ratio20_prior'] = prior
            self.matrices[actor] = ((recent >= .01) & (recent > prior) & (buys >= 3)).astype(float).where(recent.notna() & prior.notna() & buys.notna())
            flows[actor] = net
        combined = flows['trust']+flows['foreign']
        recent = combined.rolling(5, min_periods=5).sum()/v5
        prior = combined.rolling(20, min_periods=20).sum().shift(5)
        self.matrices['selling'] = ((recent <= -.01) & (prior > 0)).astype(float).where(recent.notna() & prior.notna())
        close = data.features.adjusted_close
        extension = close/close.rolling(20, min_periods=20).mean()
        self.matrices['extension20'] = extension
        self.matrices['price'] = (extension <= 1.10).astype(float).where(extension.notna())
        factor = data.features.adjustment_factor
        ratio = factor/factor.shift(20)
        unchanged_units = (ratio-1).abs() <= .02
        def declining(frame):
            frame = frame.where(np.isfinite(frame) & frame.ge(0))
            valid = frame.rolling(21, min_periods=21).count().eq(21) & unchanged_units
            return (frame < frame.shift(20)).astype(float).where(valid)
        self.matrices['margin'] = declining(matrix(margin, 'margin_purchase_balance'))
        sbl_frames = [pd.read_parquet(p) for p in (directory/'raw/TaiwanDailyShortSaleBalances').glob('*.parquet')]
        sbl = pd.concat(sbl_frames, ignore_index=True) if sbl_frames else pd.DataFrame()
        if not sbl.empty:
            sbl['date'] = pd.to_datetime(sbl.date)
        self.matrices['sbl'] = declining(matrix(sbl, 'SBLShortSalesCurrentDayBalance'))
        self.holders, self.holder_audit = {}, []
        for path in (directory/'raw/TaiwanStockHoldingSharesPer').glob('*.parquet'):
            rows = holder_rows(pd.read_parquet(path)).sort_values('date')
            self.holder_audit.extend(rows.to_dict('records'))
            if not rows.empty:
                rows['delta4'] = rows.large_pct-rows.large_pct.shift(4)
                gap = (rows.date-rows.date.shift(4)).dt.days
                rows['delta4'] = rows.delta4.where(gap.between(21, 35) & rows.valid & rows.valid.shift(4, fill_value=False))
                self.holders[str(rows.stock_id.iloc[0])] = rows
        self.brokers = {}
        for path in (directory/'raw/broker').glob('*.parquet'):
            sid, day, _ = path.stem.split('_')
            self.brokers[(sid, day)] = broker_concentration(pd.read_parquet(path))
        self.events = {e['event_id']: e for e in data.entries}

    def context(self, index, sid, event_id=None):
        if not 1 <= index < len(self.days):
            raise ValueError('Chip decision needs prior market session')
        j, result = index-1, {}
        for name, frame in self.matrices.items():
            value = float(frame.iat[j, frame.columns.get_loc(sid)])
            result[name] = None if not math.isfinite(value) else (bool(value) if name in (*FILTERS, 'selling') else value)
        day = self.days[j]
        result['signal_date'] = str(day.date())
        for lag, key in ((7, 'holder'), (14, 'holder14')):
            rows = self.holders.get(sid)
            eligible = rows.loc[rows.date+pd.Timedelta(days=lag) <= day] if rows is not None else pd.DataFrame()
            value, observation = None, None
            if not eligible.empty:
                row = eligible.iloc[-1]
                observation = str(row.date.date())
                if (day-row.date).days <= 21 and math.isfinite(row.delta4):
                    value = bool(row.delta4 >= .5)
            result[key], result[key+'_observation'] = value, observation
        result['broker_concentration'] = self.brokers.get((sid, str(day.date())))
        result['broker'] = None if result['broker_concentration'] is None else result['broker_concentration'] >= .10
        result['group_flow'] = None
        if event_id is not None:
            event = self.events[event_id]
            if event['signal_date'] != str(day.date()) or event['members'] != [sid]:
                raise ValueError('Candidate chip context date mismatch')
            share = event['leader_evidence']['leader_turnover_share']
            result['group_flow'] = bool(share['rising']) if share['valid'] else None
        for left, right in (('trust', 'price'), ('foreign', 'price'), ('holder', 'margin'), ('holder14', 'margin')):
            result[left+'_'+right] = tri_and(result[left], result[right])
        return result


class ChipReplay(TechnicalReplay):
    def __init__(self, *args, chip_signals, chip_mode, **kwargs):
        base = chip_mode.removeprefix('available_')
        if base not in ('control', 'sell_full', 'sell_half', *FILTERS):
            raise ValueError('Unknown chip experiment')
        super().__init__(*args, mode='control', **kwargs)
        self.chip_signals, self.chip_mode = chip_signals, chip_mode
        self.chip_decisions, self.half_states = [], {}

    def run(self):
        return self._run_experiment()

    def _entry_plan(self, day, event, opening_nav, previous_price, budget):
        qty, budget, row = super()._entry_plan(day, event, opening_nav, previous_price, budget)
        mode = self.chip_mode
        if mode in ('control', 'sell_full', 'sell_half'):
            return qty, budget, row
        context = self.chip_signals.context(self.positions[day], event['members'][0], event['event_id'])
        value = context[mode.removeprefix('available_')]
        passed = value is not None if mode.startswith('available_') else value is True
        self.chip_decisions.append(dict(date=str(day.date()), stock_id=event['members'][0],
            event_id=event['event_id'], action='entry_filter', passed=passed, **context))
        if not passed:
            row.update(failure='chip_unknown' if value is None else 'chip_filter_rejected', requested_qty=0)
            qty = 0
        return qty, budget, row

    def corporate_day(self, day):
        income = super().corporate_day(day)
        if self.chip_mode != 'sell_full':
            return income
        index = self.positions[day]
        for event_id, state in self.exit_states.items():
            sid = state['stock_id']
            if state['trigger_reason'] or sid not in self.holdings or self.holdings[sid]['event_id'] != event_id:
                continue
            context = self.chip_signals.context(index, sid)
            if context['selling'] is True:
                state.update(trigger_reason='chip_sell_full', signal_date=context['signal_date'],
                             target_date=str(day.date()), target_index=index)
                self.holdings[sid]['due_index'] = index
                self.chip_decisions.append(dict(date=str(day.date()), stock_id=sid,
                    event_id=event_id, action='exit_full', **context))
        return income

    def _add_positions(self, day, opening_nav):
        # Existing daily loop calls this after scheduled exits and entries;
        # no re-entry-day selling, no extra cash allocated, all original caps apply.
        if self.chip_mode != 'sell_half':
            return
        for sid, holding in list(self.holdings.items()):
            identity = holding['event_id']
            if sid == '0050' or holding['qty'] <= 0:
                continue
            state = self.exit_states.get(identity)
            if state is None or state['trigger_reason']:
                continue
            context = self.chip_signals.context(self.positions[day], sid)
            if identity not in self.half_states and context['selling'] is True:
                self.half_states[identity] = dict(remaining=holding['qty']//2,
                    initial_qty=holding['qty'], signal_date=context['signal_date'])
            half = self.half_states.get(identity)
            if half and half['remaining'] > 0:
                requested = min(half['remaining'], holding['qty'])
                filled = self.order(day, sid, 'sell', requested, 'chip_sell_half', identity, half['signal_date'])
                half['remaining'] -= filled
                self.chip_decisions.append(dict(date=str(day.date()), stock_id=sid,
                    event_id=identity, action='exit_half', requested_qty=requested,
                    filled_qty=filled, first_signal_date=half['signal_date'], **context))
