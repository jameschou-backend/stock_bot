"""Assemble data-only POC donors; frozen accounts still verify every source.

No initialized clients, locks, credentials, or financial objects cross processes.
This changes only when donor reports are decoded. The unchanged frozen runner
calls verify_sources before creating the account and again after its last day.
"""
from copy import deepcopy
import json
from pathlib import Path

from skills.all_entry_accounts import digest, write
from skills.poc_followup_corporate import CompletedFollowupData
from skills.poc_followup_reuse import ReusedFollowupData
from skills.poc_followup_shared_acquisition import SynchronizedFollowupData
from skills.poc_followup_data_completion import CompletedProfileData

ROOT = Path(__file__).resolve().parents[1]


def merge(refs, name, expected):
    if name in refs and refs[name] != expected:
        raise ValueError('Conflicting donor source '+name)
    refs[name] = expected


def assemble(manifest, output, root=ROOT):
    root, manifest, output = Path(root).resolve(), Path(manifest).resolve(), Path(output).resolve()
    manifest.relative_to(root); output.relative_to(root)
    spec_sha = digest(root/'docs/prereg_poc_followup_20261014.json')
    manifest_sha = digest(manifest)
    content = json.loads(manifest.read_text())
    if content.get('schema') != 'poc_followup_profile_donors_v1':
        raise ValueError('Unknown donor manifest')
    refs = {str(manifest.relative_to(root)): manifest_sha}
    profiles, origins = {}, {}
    for ref in content['reports']:
        path = (root/ref['path']).resolve(); path.relative_to(root)
        if digest(path) != ref['sha256']:
            raise ValueError('Donor report changed')
        report = json.loads(path.read_text())
        if (report.get('all_completed') is not True or (report['start'], report['end']) != ('2024-01-02', '2026-10-02')
                or report.get('followup', {}).get('preregistration', {}).get('sha256') != spec_sha
                or not report.get('source_sha256')):
            raise ValueError('Incomplete or unregistered donor')
        merge(refs, ref['path'], ref['sha256'])
        for name, expected in report['source_sha256'].items():
            snapshot = report.get('source_snapshots', {}).get(name)
            # Exactly the selection policy of Sources.closure, without claiming
            # the assembly step has checked the underlying bytes yet.
            if snapshot and Path(name).suffix in ('.py', '.md'):
                if snapshot['sha256'] != expected:
                    raise ValueError('Donor snapshot differs')
                name = snapshot['path']
            merge(refs, name, expected)
        for eid, value in report['shared_profile_results'].items():
            if eid != value['event_id'] or (eid in profiles and profiles[eid] != value):
                raise ValueError('Conflicting donor profile')
            profiles[eid] = value
            origins.setdefault(eid, ref['path'])
    value = dict(schema='poc_risk_data_only_donors_v1', manifest=dict(path=str(manifest.relative_to(root)), sha256=manifest_sha),
        profiles=profiles, origins=origins, source_sha256=refs,
        source_verification='unchanged_frozen_runner_before_first_and_after_last_account_action')
    write(output, value)
    return digest(output)


def evidence_type(package, expected, controller):
    package, controller = Path(package).resolve(), Path(controller).resolve()
    package.relative_to(ROOT); controller.relative_to(ROOT)

    class DataOnlyDonors(ReusedFollowupData):
        def __init__(self, *, online=False, donor_manifest):
            # Initialize fresh native clients and locks, without decoding and
            # re-binding the same 13 donor reports in every new account.
            CompletedFollowupData.__init__(self, online=online)
            self._bind(package, expected)
            value = json.loads(package.read_text())
            donor = Path(donor_manifest).resolve()
            if (value['schema'] != 'poc_risk_data_only_donors_v1'
                    or value['manifest'] != dict(path=str(donor.relative_to(ROOT)), sha256=digest(donor))):
                raise ValueError('Prepared donor manifest differs')
            self.followup_donors = dict(value['origins'])
            for eid, profile in value['profiles'].items():
                if eid != profile['event_id'] or (eid in self.profile_memo and self.profile_memo[eid] != profile):
                    raise ValueError('Prepared profile differs from frozen data')
                self.profile_memo[eid] = deepcopy(profile)
            for name, sha in value['source_sha256'].items():
                merge(self.refs, name, sha)
            self._bind(Path(__file__)); self._bind(controller)

    # Retain the original acquisition guard, shared quota, completion budget,
    # schema checks and profile method. Only its donor-loading constructor is
    # replaced in this explicit MRO; the frozen verify_sources remains intact.
    class Guarded(SynchronizedFollowupData, DataOnlyDonors):
        pass

    class Prepared(CompletedProfileData, Guarded):
        pass

    return Prepared
