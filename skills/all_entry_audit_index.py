"""Index quote reads in a private copy of the unchanged independent auditor.

Only three boolean row selections are replaced. Source hashes and exact match
counts guard the transformation; every financial assertion is retained. The
auditor owns its copied, immutable quote frames, so an index lives only as long
as its frame. No imported module or shared runner is patched.
"""
from hashlib import sha256
import inspect
from pathlib import Path
from types import ModuleType
import weakref

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().with_name('poc_range_audit.py')
SOURCE_SHA = 'df5c1968dde29c9a9e7dee871ad865b38033c5812be3e56e391a320ac2e1a641'


class QuoteRows:
    """Return exactly the original rows, including order, duplicates and misses."""

    def __init__(self):
        self._frames = {}

    def __call__(self, source, day, sid):
        day = pd.Timestamp(day)
        if pd.isna(day) or pd.isna(sid):
            return source.iloc[:0]
        key = id(source)
        cached = self._frames.get(key)
        if cached is None or cached[0]() is not source:
            eligible = source.date.notna() & source.stock_id.notna()
            positions = np.flatnonzero(eligible.to_numpy())
            coordinates = pd.MultiIndex.from_arrays([source.date.iloc[positions], source.stock_id.iloc[positions]])
            coordinates, order = coordinates.sort_values(return_indexer=True)
            positions = positions[order]
            frame = weakref.ref(source, lambda ref: self._frames.pop(key, None))
            cached = frame, coordinates, positions
            self._frames[key] = cached
        _, coordinates, positions = cached
        try:
            found = coordinates.get_loc((day, sid))
        except KeyError:
            return source.iloc[:0]
        # Boolean filtering preserves the source's order even for duplicate keys.
        rows = np.sort(np.atleast_1d(positions[found]))
        return source.iloc[rows]


def private_auditor(source=SOURCE):
    raw = Path(source).read_bytes()
    if sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('Frozen independent auditor changed; index needs renewed parity review')
    code = raw.decode('utf-8')
    replacements = (
        ("source.loc[source.date.eq(pd.Timestamp(day)) & source.stock_id.eq(sid)]",
         "_indexed_quote_rows(source, day, sid)", 2),
        ("source.loc[source.date.eq(pd.Timestamp(p['reference_date'])) & source.stock_id.eq(sid)]",
         "_indexed_quote_rows(source, p['reference_date'], sid)", 1),
    )
    for old, new, count in replacements:
        if code.count(old) != count:
            raise ValueError('Frozen quote-read sites differ')
        code = code.replace(old, new)
    module = ModuleType('_all_entry_independent_audit')
    module.__file__ = str(source)
    module._indexed_quote_rows = QuoteRows()
    exec(compile(code, str(source), 'exec'), module.__dict__)
    return module


def install_audit_index(runner):
    from scripts.research_all_entry_accounts import FROZEN, FROZEN_SHA
    if sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA:
        raise ValueError('Frozen financial runner changed before audit adapter')
    code = inspect.getsource(runner._run)
    old = 'from skills.poc_range_audit import audit_range_execution'
    if code.count(old) != 1 or runner._run.__globals__ is not runner.__dict__:
        raise ValueError('Private runner audit binding differs')
    auditor = private_auditor()
    runner._all_entry_indexed_audit = auditor.audit_range_execution
    code = code.replace(old, 'audit_range_execution = _all_entry_indexed_audit')
    exec(compile(code, str(FROZEN), 'exec'), runner.__dict__)
