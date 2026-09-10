"""Causal support, risk distance and one preregistered breakout pattern.

Execution row ``i`` can inspect signals only through row ``i - 1``. Historical
windows count audited market rows, not whichever observations happen to exist.
Missing bars therefore remain unknown instead of being forward filled.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from skills.scenario_exit_replay import ExitSignals, _finite


class TechnicalSignals(ExitSignals):
    """Add adjusted OHLC features while preserving the exit-signal contract.

    Each raw bar is adjusted by that same row's adjusted-close/raw-close ratio.
    The supplied adjusted-close history is never interpolated. Bad individual
    observations are retained as unavailable features with diagnostics; invalid
    schema, duplicate bars and ambiguous dates fail explicitly.
    """

    def __init__(self, adjusted_close: pd.DataFrame, raw_quotes: pd.DataFrame,
                 days: pd.DatetimeIndex):
        super().__init__(adjusted_close, days)
        if not isinstance(raw_quotes, pd.DataFrame):
            raise ValueError('raw_quotes must be a DataFrame')
        fields = ('open', 'high', 'low', 'close', 'volume')
        required = {'date', 'stock_id', *fields}
        missing = required.difference(raw_quotes.columns)
        if missing:
            raise ValueError('Missing raw quote columns: ' + ', '.join(sorted(missing)))
        if not raw_quotes.columns.is_unique:
            raise ValueError('Raw quote columns must be unique')
        q = raw_quotes.loc[:, ['date', 'stock_id', *fields]].copy()
        if not q.stock_id.map(lambda sid: isinstance(sid, str)).all():
            raise ValueError('Raw stock IDs must be strings')
        try:
            q['date'] = pd.to_datetime(q.date, errors='raise')
            dates = pd.DatetimeIndex(q.date)
        except (ValueError, TypeError) as exc:
            raise ValueError('Raw quote dates must be valid naive market dates') from exc
        if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError('Raw quote dates must be valid naive market dates')
        if q.duplicated(['date', 'stock_id']).any():
            raise ValueError('Duplicate raw technical bars')
        for field in fields:
            dtype = q[field].dtype
            if (pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype)
                    or not pd.api.types.is_numeric_dtype(dtype)):
                raise ValueError('Raw OHLC and volume must be real numbers or explicitly missing')
        columns = self.adjusted_close.columns
        self.raw_fields = {
            field: q.pivot(index='date', columns='stock_id', values=field)
                    .reindex(index=self.days, columns=columns).astype(float)
            for field in fields
        }
        observed = q.assign(_observed=True).pivot(index='date', columns='stock_id', values='_observed')
        self.observed = observed.reindex(index=self.days, columns=columns).notna()
        raw = self.raw_fields
        self.raw_ohlc_complete = raw['open'].notna()
        raw_positive = np.isfinite(raw['open']) & raw['open'].gt(0)
        for field in ('high', 'low', 'close'):
            self.raw_ohlc_complete &= raw[field].notna()
            raw_positive &= np.isfinite(raw[field]) & raw[field].gt(0)
        geometry = (raw['high'].ge(raw['open']) & raw['high'].ge(raw['close'])
                    & raw['low'].le(raw['open']) & raw['low'].le(raw['close'])
                    & raw['high'].ge(raw['low']))
        self.valid_raw_ohlc = raw_positive & geometry
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            factor = self.adjusted_close / raw['close']
            high, low = raw['high'] * factor, raw['low'] * factor
        self.adjustment_factor = factor.where(np.isfinite(factor) & factor.gt(0))
        self.valid_ohlc = (self.valid_raw_ohlc & self.adjustment_factor.notna()
                           & np.isfinite(high) & high.gt(0)
                           & np.isfinite(low) & low.gt(0))
        self.adjusted_high = high.where(self.valid_ohlc)
        self.adjusted_low = low.where(self.valid_ohlc)
        self.support20 = self.adjusted_low.rolling(20, min_periods=20).min().shift(1)
        self.resistance20 = self.adjusted_high.rolling(20, min_periods=20).max().shift(1)
        self.history20_count = self.valid_ohlc.astype(int).rolling(20, min_periods=1).sum().shift(1)
        recent_high = self.adjusted_high.rolling(10, min_periods=10).max().shift(1)
        recent_low = self.adjusted_low.rolling(10, min_periods=10).min().shift(1)
        earlier_high = self.adjusted_high.rolling(10, min_periods=10).max().shift(11)
        earlier_low = self.adjusted_low.rolling(10, min_periods=10).min().shift(11)
        self.range_recent10 = recent_high - recent_low
        self.range_preceding10 = earlier_high - earlier_low
        volume = raw['volume']
        self.valid_volume = np.isfinite(volume) & volume.ge(0)
        # Volume means require complete, valid OHLC bars too. A known zero is
        # allowed; an entirely zero baseline cannot establish volume expansion.
        self.volume20 = volume.where(self.valid_volume & self.valid_ohlc).rolling(20, min_periods=20).mean().shift(1)

    def technical_context(self, index: int, sid: str) -> dict:
        """Return the prior session's signal, with explicit unknown states.

        ``risk_fraction`` is a price distance, not a guaranteed loss or a
        cost-inclusive stop estimate. Nonpositive distance cannot size an entry.
        ``support_raw`` uses the same prior session's adjustment ratio; a later
        split must never change the raw price selected for this decision.
        """
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.days):
            raise ValueError('Execution index must identify an audited market row')
        if sid not in self.adjusted_close:
            raise ValueError('Technical signal stock missing from adjusted matrix: ' + str(sid))
        j = int(index) - 1
        result = dict(signal_index=j, signal_date=str(self.days[j].date()) if j >= 0 else None,
                      adjusted_close=None, raw_close=None, adjustment_factor=None,
                      support20=None, support_raw=None, support_available=False,
                      support_failure=None, risk_fraction=None, risk_available=False,
                      resistance20=None, range_recent10=None, range_preceding10=None,
                      raw_volume=None, volume20=None, breakout20=None, contraction10=None,
                      volume_expansion=None, pattern_available=False, pattern_pass=None,
                      valid_history20=0, diagnostics=[])
        if j < 0:
            result['diagnostics'].append('no_prior_market_session')
            return result

        value = lambda frame: _finite(frame.iloc[j][sid])
        result.update(adjusted_close=self.price(j, sid), raw_close=value(self.raw_fields['close']),
                      adjustment_factor=value(self.adjustment_factor), support20=value(self.support20),
                      resistance20=value(self.resistance20), range_recent10=value(self.range_recent10),
                      range_preceding10=value(self.range_preceding10), raw_volume=value(self.raw_fields['volume']),
                      volume20=value(self.volume20), valid_history20=int(value(self.history20_count) or 0))
        issues = result['diagnostics']
        if not self.observed.iloc[j][sid]:
            issues.append('raw_bar_missing')
        elif not self.raw_ohlc_complete.iloc[j][sid]:
            issues.append('raw_ohlc_missing')
        elif not self.valid_raw_ohlc.iloc[j][sid]:
            issues.append('raw_ohlc_invalid')
        if result['adjusted_close'] is None:
            issues.append('adjusted_close_missing_or_invalid')
        if not self.valid_ohlc.iloc[j][sid] and not issues:
            issues.append('adjusted_bar_invalid')
        if result['support20'] is None:
            issues.append('support_history_incomplete')
        current_valid = bool(self.valid_ohlc.iloc[j][sid])
        if current_valid and result['support20'] is not None:
            close, support = result['adjusted_close'], result['support20']
            result.update(support_available=True, support_failure=bool(close < support),
                          support_raw=support / result['adjustment_factor'])
            distance = (close - support) / close
            if distance > 0:
                result.update(risk_fraction=float(distance), risk_available=True)
            else:
                issues.append('support_distance_not_positive')

        if current_valid and result['resistance20'] is not None:
            result['breakout20'] = bool(result['adjusted_close'] > result['resistance20'])
        if current_valid and result['range_recent10'] is not None and result['range_preceding10'] is not None:
            if result['range_preceding10'] > 0:
                result['contraction10'] = bool(result['range_recent10'] <= .75 * result['range_preceding10'])
            else:
                issues.append('preceding_range_not_positive')
        if not bool(self.valid_volume.iloc[j][sid]):
            issues.append('signal_volume_missing_or_invalid')
        if result['volume20'] is None:
            issues.append('volume_history_incomplete')
        elif result['volume20'] <= 0:
            issues.append('volume_baseline_not_positive')
        elif current_valid and bool(self.valid_volume.iloc[j][sid]):
            result['volume_expansion'] = bool(result['raw_volume'] >= 1.5 * result['volume20'])
        components = [result[key] for key in ('breakout20', 'contraction10', 'volume_expansion')]
        if all(component is not None for component in components):
            result.update(pattern_available=True, pattern_pass=bool(all(components)))
        return result

    def entry_context(self, index: int, sid: str) -> dict:
        """Alias for the same strictly lagged information used to size entries."""
        return self.technical_context(index, sid)
