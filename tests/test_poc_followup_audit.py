from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from skills.poc_followup_audit import audit, _correlation


def inputs(tmp_path, variant):
    events = [dict(event_id=s, members=[s], signal_date='2024-04-01', priority=p)
              for s, p in [('1111', 9), ('2222', 8)]]
    candidate = tmp_path/'.cache/all-entry-accounts-20261011/inputs-v3/poc_up_red.json'
    candidate.parent.mkdir(parents=True)
    candidate.write_text(json.dumps(dict(entries=events)))
    settings = dict(followup_variant=variant, horizon=63, protective_loss12_unchanged=True,
                    time63_unchanged=True, buy_fraction=.5, sell_fraction=.3)
    report = dict(input_bundle='inputs', source_sha256={str(candidate.relative_to(tmp_path)): 'bound'},
                  shared_profile_results={})
    row = dict(date='2024-04-02', original_ids=['1111', '2222'], applied=True)
    context = dict(date=row['date'], original_event_ids=row['original_ids'],
                   actual_reservations_verified=True, certificate={})
    case = dict(followup=dict(arm='poc_up_red', variant=variant),
                account=dict(settings=settings, followup_log=[row], selection_decisions=[context]))
    return case, report, row, context


def test_strength_independently_checks_sealed_inputs_ties_and_cutoff(tmp_path):
    case, report, row, context = inputs(tmp_path, 'poc_strength')
    for eid, after in [('1111', 101), ('2222', 120)]:
        report['shared_profile_results'][eid] = dict(available=True, stock_id=eid,
            signal_date='2024-04-01', prior_dates=['2024-03-29'], poc_before=100, poc_after=after)
    row.update(scores={'1111': .01, '2222': .2}, original_priorities={'1111': 9, '2222': 8},
               ordered_ids=['2222', '1111'])
    context['certificate']['original_ordered_ids'] = row['ordered_ids']
    assert audit(case, report, tmp_path)['decisions_checked'] == 2
    bad = deepcopy(case); bad['account']['followup_log'][0]['scores']['2222'] = .3
    with pytest.raises(ValueError, match='score differs'):
        audit(bad, report, tmp_path)
    report['shared_profile_results']['1111']['prior_dates'] = ['2024-04-01']
    with pytest.raises(ValueError, match='noncausal'):
        audit(case, report, tmp_path)


def test_correlation_admission_is_rebuilt_and_future_prices_are_ignored(tmp_path):
    case, report, row, context = inputs(tmp_path, 'correlation60')
    rng = np.random.default_rng(72)
    dates = pd.date_range('2024-02-01', periods=70)
    close = pd.DataFrame(100*np.cumprod(1+rng.normal(0, .02, (70,2)), axis=0),
                         index=dates, columns=['1111', '2222'])
    close.index.name = 'date'
    (tmp_path/'inputs').mkdir()
    close.reset_index().to_parquet(tmp_path/'inputs/close-official.parquet')
    state = dict(held=['1111'], occupied=['1111'])
    rho = _correlation(close, '2222', '1111', '2024-04-01')
    evidence = dict(attempted=True, correlation_pairs={'1111': rho}, correlation_cutoff='2024-04-01')
    context['certificate'].update(initial_state=state, decisions=[dict(event_id='2222',
        profile_status='known_true', selection_status='reserved', state_before=state,
        probe=deepcopy(evidence), reservation=deepcopy(evidence))])
    assert audit(case, report, tmp_path)['correlation_pairs_checked'] == 2
    close.loc['2024-04-02':, '1111'] = -1e10
    close.reset_index().to_parquet(tmp_path/'inputs/close-official.parquet')
    assert audit(case, report, tmp_path)['passed']
    context['certificate']['decisions'][0]['reservation']['correlation_pairs']['1111'] = .99
    with pytest.raises(ValueError, match='independent 60-return'):
        audit(case, report, tmp_path)
