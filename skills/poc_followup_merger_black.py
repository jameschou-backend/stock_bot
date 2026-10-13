"""Preserve holding clocks across the evidenced 2867 exchange, with new ordinary candles."""
import inspect
from pathlib import Path
import textwrap
from types import FunctionType

import pandas as pd

from skills.legacy_merger_settlement import MergerSettlement, OLD, NEW, EXCHANGE, ORDINARY
from skills import legacy_merger_audit
from skills.three_black_exit import audit_three_black


class ExchangeBlack:
    def __init__(self, source, account, required):
        self.source, self.account, self.required = source, account, required
        self.days = source.days

    def __getattr__(self, name): return getattr(self.source, name)

    def exits(self, execution_index, entry_index, sid):
        holding = self.account.holdings.get(sid, {})
        if (sid == NEW and self.account.merger_event_id
                and holding.get('event_id') == self.account.merger_event_id):
            first = self.days.get_loc(pd.Timestamp(ORDINARY))
            if execution_index-first < self.required:
                return False
        return self.source.exits(execution_index, entry_index, sid)


def _exchange_method():
    code = textwrap.dedent(inspect.getsource(MergerSettlement.exchange_units))
    old = "if self.drawdown_arm != 'control':"
    if code.count(old) != 1:
        raise ValueError('Frozen exchange validation seam changed')
    code = code.replace(old, "if self.drawdown_arm != 'control' and not isinstance(self.black_signals, ExchangeBlack):")
    namespace = dict(MergerSettlement.exchange_units.__globals__, ExchangeBlack=ExchangeBlack)
    exec(compile(code, str(Path(__file__)), 'exec'), namespace)
    return namespace['exchange_units']


def _stock_id(cohort, day):
    return (NEW if cohort.get('stock_id') == OLD and cohort.get('share_exchange_date') == EXCHANGE
            and day >= EXCHANGE else cohort['stock_id'])


def _ordinary_candles(cohort, days, execution_index, required):
    if cohort.get('stock_id') != OLD or cohort.get('share_exchange_date') != EXCHANGE:
        return True
    if str(days[execution_index].date()) < EXCHANGE:
        return True
    return execution_index-days.get_loc(pd.Timestamp(ORDINARY)) >= required


def independent_black_audit(required):
    code = inspect.getsource(audit_three_black)
    replacements = [("sid != cohorts[eid]['stock_id']", "sid != _stock_id(cohorts[eid], row['date'])")]
    if required == 4:
        replacements += [('i >= 4 and i-entry >= 3', 'i >= 5 and i-entry >= 4'),
                         ('range(i-3, i)', 'range(i-4, i)')]
    elif required != 3:
        raise ValueError('Unregistered bearish-candle requirement')
    for old, new in replacements:
        if code.count(old) != 1: raise ValueError('Independent black audit seam changed')
        code = code.replace(old, new)
    site = '        if expected:\n'
    if code.count(site) != 2: raise ValueError('Independent black audit flow changed')
    code = code.replace(site,
        '        expected &= _ordinary_candles(cohorts[eid], days, i, _required)\n'+site, 1)
    namespace = dict(audit_three_black.__globals__, _stock_id=_stock_id,
                     _ordinary_candles=_ordinary_candles, _required=required)
    exec(compile(code, str(Path(__file__)), 'exec'), namespace)
    return namespace['audit_three_black']


def install(runner, mode, variant):
    required = 4 if variant == 'exit_confirm' else 3
    class CorporateBlackMerger(MergerSettlement):
        exchange_units = _exchange_method()
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.benchmark:
                self.black_signals = ExchangeBlack(self.black_signals, self, required)
        def run(self):
            result = super().run()
            result['settings']['merger_black_policy'] = 'new_ordinary_candles_original_holding_clock_v1'
            return result
    private = FunctionType(legacy_merger_audit.install.__code__,
        dict(legacy_merger_audit.install.__globals__, MergerSettlement=CorporateBlackMerger))
    refs = private(runner, mode)
    factory = runner.engine_types
    def stocks(*args, **kwargs):
        stock, benchmark = factory(*args, **kwargs)
        if benchmark.__bases__[0] is not CorporateBlackMerger:
            raise ValueError('Unexpected merger benchmark seam')
        return stock, benchmark.__bases__[1]
    runner.engine_types = stocks
    runner.audit_three_black = independent_black_audit(required)
    return refs
