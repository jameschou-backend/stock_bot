from copy import deepcopy

import pytest

from skills.all_entry_merger_halt_publication import verify_published_halt
from tests.test_all_entry_merger_halt import planner, plan


def old_rejection():
    p = plan(planner())
    p.update(rejection='opening_slots_locked', sizing_budget=0., reserved_cash=0.)
    return dict(completed=True, account=dict(holdings=[], trades=[], tick_plans=[p],
                                            orders=[], cash_ledger=[]))


def test_old_zero_budget_slot_rejection_needs_no_new_financial_replay(monkeypatch):
    monkeypatch.setattr('skills.all_entry_merger_halt_publication.load_halt', lambda *_: ({}, {}))
    verify_published_halt(old_rejection())


@pytest.mark.parametrize('key,value', [('sizing_budget', 1), ('reserved_cash', 1),
                                    ('planned_qty', 1), ('rejection', 'unknown')])
def test_old_rejection_cannot_hide_funded_or_unknown_plan(monkeypatch, key, value):
    monkeypatch.setattr('skills.all_entry_merger_halt_publication.load_halt', lambda *_: ({}, {}))
    case = old_rejection(); case['account']['tick_plans'][0][key] = value
    with pytest.raises(ValueError): verify_published_halt(case)


def test_old_zero_budget_rejection_cannot_move_cash(monkeypatch):
    monkeypatch.setattr('skills.all_entry_merger_halt_publication.load_halt', lambda *_: ({}, {}))
    case = old_rejection(); p = case['account']['tick_plans'][0]
    case['account']['cash_ledger'].append(dict(date=p['date'], event_id=p['event_id'], cash_change=-1))
    with pytest.raises(ValueError, match='moved'): verify_published_halt(case)
