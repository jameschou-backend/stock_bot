"""Preserve event identity when recording a real reservation planner result."""
from types import FunctionType

from skills import poc_followup_profile_schema as schema
from skills.poc_followup import CorrelationPlanner as FrozenCorrelationPlanner

VARIANTS = schema.VARIANTS
POLICY = 'correlation_reservation_identity_v1'


class CorrelationPlanner(FrozenCorrelationPlanner):
    def consider(self, event):
        expected = self.probe(event)
        if expected.get('event_id', event['event_id']) != event['event_id']:
            raise ValueError('Correlation reservation event identity changed')
        if expected['attempted']:
            actual = self.source.consider(event)
            if any(actual[k] != expected[k] for k in actual):
                raise ValueError('Correlation reservation changed original cash/quantity')
        self.log.append(dict(expected, event_id=event['event_id']))
        return expected


# Keep every historical implementation intact, including profile normalization.
install = FunctionType(schema.install.__code__, dict(schema.install.__globals__,
    CorrelationPlanner=CorrelationPlanner), name='install')
