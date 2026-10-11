"""Exact indexed equivalent of the frozen corporate opening-reference lookup.

This private class adapter changes no event, price, exception or calendar rule.
The frozen runner's CorpLoader still applies rescheduled opening references
before calling this method. No imported module or production class is mutated.
"""
import math
from types import SimpleNamespace

import pandas as pd

from skills.million_replay import UnresolvedAction
from skills.replay_corporate_actions import CorporateActions


def indexed_type(base):
    if base.reference_price is not CorporateActions.reference_price:
        raise ValueError('Corporate reference method changed; index needs renewed parity review')

    class IndexedReference(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._opening_reference_rows = {}
            for row in self.events[['stock_id','event_date','opening_ref','ref_price']].itertuples(index=False):
                key = (row.stock_id,row.event_date)
                self._opening_reference_rows.setdefault(key,[]).append((row.opening_ref,row.ref_price))

        def reference_price(self, sid, day, prior):
            rows = self._opening_reference_rows.get((sid,day),())
            if len(rows)>1:
                raise UnresolvedAction(f'Multiple corporate opening references: {sid} {day}')
            if rows:
                opening, reference = rows[0]
                ref = opening if pd.notna(opening) else reference
                if ref and math.isfinite(ref) and ref>0:
                    return float(ref)
                raise UnresolvedAction(f'Corporate opening reference missing: {sid} {day}')
            return prior/4 if sid=='0050' and day=='2025-06-18' else prior

    return IndexedReference


def install_private_index(runner):
    # Copy only the module namespaces used for the nested CorpLoader base.
    # All other frozen functions, classes and their globals remain unchanged.
    study = SimpleNamespace(**vars(runner.study))
    study.old = SimpleNamespace(**vars(runner.study.old))
    study.old.TrackedCorporateActions = indexed_type(runner.study.old.TrackedCorporateActions)
    runner.study = study
