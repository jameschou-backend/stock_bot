import json
from pathlib import Path

import pytest

from skills.all_entry_accounts import digest, write
from skills.poc_risk_evidence import assemble, evidence_type
from skills.poc_followup_data_completion import CompletedProfileData
from skills.poc_followup_reuse import ReusedFollowupData
from skills.poc_followup_shared_acquisition import SynchronizedFollowupData
from scripts.summarize_poc_broker_account import Sources
from scripts.run_poc_risk_matrix import account_output


def donor_fixture(root):
    spec = root/'docs/prereg_poc_followup_20261014.json'
    write(spec, {'rules':'fixed'})
    (root/'raw.txt').write_text('original data')
    (root/'model.py').write_text('new code')
    (root/'frozen.snapshot').write_text('old code')
    report = dict(all_completed=True, start='2024-01-02', end='2026-10-02',
        followup=dict(preregistration=dict(sha256=digest(spec))),
        shared_profile_results={'event':dict(event_id='event',available=True)},
        source_sha256={'raw.txt':digest(root/'raw.txt'),'model.py':digest(root/'frozen.snapshot')},
        source_snapshots={'model.py':dict(path='frozen.snapshot',sha256=digest(root/'frozen.snapshot'))})
    write(root/'report.json',report)
    manifest=dict(schema='poc_followup_profile_donors_v1',reports=[dict(path='report.json',sha256=digest(root/'report.json'))])
    write(root/'donors.json',manifest)
    return report


def test_plain_donor_assembly_preserves_the_exact_original_source_selection(tmp_path):
    report = donor_fixture(tmp_path)
    package = tmp_path/'prepared.json'
    expected = assemble(tmp_path/'donors.json',package,root=tmp_path)
    data = json.loads(package.read_text())
    original = Sources(tmp_path)
    original.bind('donors.json',digest(tmp_path/'donors.json'))
    original.bind('report.json',digest(tmp_path/'report.json'))
    original.closure(report,'report.json')
    assert data['source_sha256']==original.used
    assert data['profiles']==report['shared_profile_results']
    assert data['origins']=={'event':'report.json'}
    assert digest(package)==expected
    Sources(tmp_path).closure(data,'prepared.json')
    (tmp_path/'raw.txt').write_text('corrupted later')
    with pytest.raises(ValueError,match='Source hash mismatch'):
        Sources(tmp_path).closure(data,'prepared.json')


def test_changed_donor_report_is_rejected_before_assembly(tmp_path):
    donor_fixture(tmp_path)
    (tmp_path/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='Donor report changed'):
        assemble(tmp_path/'donors.json',tmp_path/'prepared.json',root=tmp_path)


def test_fresh_provider_keeps_frozen_verification_acquisition_and_profile_methods():
    root=Path(__file__).resolve().parents[1]
    provider=evidence_type(root/'.cache/prepared.json','expected',root/'scripts/run_poc_risk_matrix.py')
    assert provider.verify_sources is CompletedProfileData.verify_sources
    assert provider.profile is CompletedProfileData.profile
    assert provider.__init__ is CompletedProfileData.__init__
    mro=provider.mro()
    loader=next(c for c in mro if c.__name__=='DataOnlyDonors')
    assert mro.index(SynchronizedFollowupData)<mro.index(loader)<mro.index(ReusedFollowupData)


def test_every_account_has_an_independent_frozen_run_lock(tmp_path):
    parents = [account_output(tmp_path,v,m,a).parent for v in ('chase3','vol20')
               for m in ('baseline','fee2','slip2','combined') for a in ('poc_up_red','poc_priority')]
    assert len(parents)==len(set(parents))==16
