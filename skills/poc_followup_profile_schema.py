"""Explicit adapter for the two sealed POC profile schemas; no rule changes."""
from copy import deepcopy
import math
from types import FunctionType

from skills import poc_followup as frozen

VARIANTS = frozen.VARIANTS


def strength_inputs(value):
    formats = []
    pairs = []
    if 'poc_before' in value or 'poc_after' in value:
        formats.append('flat_half_poc_prices')
        pairs.append((value.get('poc_before'), value.get('poc_after')))
    if 'profile' in value:
        profile = value['profile']
        if profile.get('available') is not True:
            raise ValueError('Available POC result has unavailable nested profile')
        halves = [profile.get(k, {}) for k in ('first_half', 'second_half')]
        if any(h.get('available') is not True for h in halves):
            raise ValueError('Available POC result has unavailable half profile')
        formats.append('nested_half_poc_prices')
        pairs.append(tuple(h.get('poc_price') for h in halves))
    if not pairs or any(not isinstance(x, (int, float)) or isinstance(x, bool)
            or not math.isfinite(x) or x <= 0 for pair in pairs for x in pair):
        raise ValueError('Available POC profile lacks finite strength inputs')
    if any(pair != pairs[0] for pair in pairs[1:]):
        raise ValueError('Conflicting POC profile schemas')
    return pairs[0], formats


def strength_order(events, provider, planner):
    originals, formats = {}, {}
    def normalized(event):
        value = provider(deepcopy(event))
        originals[event['event_id']] = deepcopy(value)
        if not value.get('available'):
            return value
        pair, schema = strength_inputs(value)
        formats[event['event_id']] = schema
        return dict(deepcopy(value), poc_before=pair[0], poc_after=pair[1])
    rows, evidence, _ = frozen.strength_order(events, normalized, planner)
    evidence['profile_input_formats'] = formats
    return rows, evidence, originals


# A private globals table leaves the frozen v1 implementation and other runs intact.
install = FunctionType(frozen.install.__code__, dict(frozen.install.__globals__,
    strength_order=strength_order), name='install')
