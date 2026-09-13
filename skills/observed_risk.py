"""Research-only observation windows: gaps block action but do not poison MA60."""
import numpy as np
import pandas as pd

from skills.cash_risk_replay import CashRiskReplay, risk_schedule
from skills.regime_state import _validate_index


def observed_risk_schedule(close, mode):
    _validate_index(close.index)
    if mode == 'none':
        return risk_schedule(close, mode)
    valid = close.where(np.isfinite(close) & close.gt(0))
    # Only the desired caps are reused; recovery is evaluated on market dates so
    # a missing close cannot silently count toward consecutive recovery sessions.
    if mode not in ('trend60', 'shock'):
        raise ValueError('Unknown account risk mode')
    available = valid.dropna()
    known = (risk_schedule(available, mode) if len(available) else
             pd.DataFrame(columns=['desired_cap', 'reason'])).reindex(
        [str(day.date()) for day in close.index])
    rows, cap, recovery = [], 1., []
    for day, row in known.iterrows():
        desired = row.desired_cap
        if pd.isna(desired):
            recovery = []
            effective, desired, reason = None, None, 'missing_risk_data'
        else:
            reason = row.reason
            if desired <= cap:
                cap, recovery = desired, []
            else:
                recovery.append(desired)
                if len(recovery) >= 5:
                    cap, recovery = min(recovery[-5:]), []
            effective = cap
        rows.append(dict(date=day, cap=effective, desired_cap=desired,
                         reason=reason, recovery_sessions=len(recovery)))
    return pd.DataFrame(rows, columns=['date', 'cap', 'desired_cap', 'reason',
                                       'recovery_sessions']).set_index('date')


class ObservedRiskReplay(CashRiskReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk = observed_risk_schedule(self.exit_signals.adjusted_close['0050'], self.risk_mode)
