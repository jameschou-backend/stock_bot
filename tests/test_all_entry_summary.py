import json
import pytest

from scripts import summarize_all_entry_accounts as summary
from skills.all_entry_accounts import registry, write
from app.all_entry_account_ui import validate


def test_delayed_share_delivery_is_counted_outside_active_entry_slots():
    account = dict(holdings=[dict(date='2026-09-01',stock_id=str(2300+i),qty=1)
                            for i in range(4)] + [dict(date='2026-09-02',stock_id='2300',qty=0)])
    assert summary.holding_scope(account,3) == dict(maximum_held_symbols=4,days_above_active_slot_count=1)


@pytest.mark.parametrize('sid,day', [('5876', '2026-07-28'), ('6446', '2026-07-09'), ('2801', '2026-08-05')])
def test_certificate_exposure_is_not_a_qualified_completed_account(sid, day):
    case = {'account':{'corporate_actions':[dict(stock_id=sid, date=day, kind='stock_dividend')]}}
    with pytest.raises(summary.UnverifiedCertificateInventory):summary.guard_certificate_inventory(case)
    case['account']['corporate_actions'][0]['kind'] = 'cash_dividend'
    summary.guard_certificate_inventory(case)


def test_invalid_instrument_keeps_its_evidence_but_cannot_rank(tmp_path, monkeypatch):
    for name in ('scripts/summarize_all_entry_accounts.py','skills/all_entry_cohort_reporting.py',
                 'scripts/summarize_strategy_account_comparison.py','skills/account_cohort_attribution.py'):
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((summary.ROOT/name).read_bytes())
    monkeypatch.setattr(summary, 'ROOT', tmp_path)
    cases = {}
    for arm in [r['id'] for r in registry()] + ['benchmark']:
        value = dict(completed=arm == 'original_breakout', summary=None,
            live_qualified=False, actual_fill_verified=False, unseen_validation=False)
        if value['completed']:
            value.update(summary={'total_return':99}, account=dict(daily=[],corporate_actions=[
                dict(stock_id='5876',date='2026-07-28',kind='stock_dividend')]))
        else:value['reason'] = 'Missing data'
        path = tmp_path / (arm+'.json')
        cases[arm] = dict(path=path.name,sha256=write(path,value),completed=value['completed'])
    source_sha = write(tmp_path/'inputs.json',dict(fixture='certificate_validation'))
    report = dict(schema='all_entry_accounts_v1',cases=cases,start=summary.START,end=summary.END,
        initial_cash=1_000_000,candidate_manifest={},source_sha256={'inputs.json':source_sha})
    path = tmp_path/'report.json'
    path.with_suffix('.sha256').write_text(write(path,report))
    result = summary.summarize(path,tmp_path/'publication.json')
    row = result['cases']['original_breakout']
    assert row['replay_completed'] is True and row['completed'] is False
    assert row['metrics'] is None and row['rank'] is None
    assert row['sha256'] == cases['original_breakout']['sha256']
    assert result['ranking'] == [] and result['best_is_global'] is False
    validate(result)
