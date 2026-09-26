"""Bind replay metadata back to the frozen candidate ledger, not another log."""
import pandas as pd


def audit_candidate_binding(case, candidates, calendar):
    source = {entry['event_id']:entry for entry in candidates}
    if len(source) != len(candidates):
        raise ValueError('Candidate identities are not unique')
    positions = {str(day.date()):i for i,day in enumerate(pd.DatetimeIndex(calendar))}
    dates = list(positions)
    mask = case['config']['factor_mask']
    if type(mask) is not int or mask not in range(8):
        raise ValueError('Unknown candidate execution scenario')
    account = case['account'] if case['completed'] else case.get('partial_account',{})
    checked = 0
    for field in ('technical_entries','cohorts','trades','orders'):
        records = case.get(field,[]) if field=='technical_entries' else account.get(field,[])
        for row in records:
            if field in ('trades','orders') and row['side']!='buy':
                continue
            original = source.get(row['event_id'])
            if original is None:
                raise ValueError('Unregistered candidate entered the account')
            expected = positions[original['entry_date']] + bool(mask&2)
            day = row['entry_date'] if field=='cohorts' else row['date']
            if (expected >= len(dates) or day!=dates[expected]
                    or row['signal_date']!=original['signal_date']
                    or row['stock_id']!=original['members'][0]):
                raise ValueError('Entry changed the original candidate, signal date or delayed execution date')
            checked += 1
    return dict(candidate_metadata_bound_to_source=True, checked_records=checked,
                candidate_count=len(source), entries_reanchored_after_delay=False)
