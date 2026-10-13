"""Reuse sealed account-independent profiles across fixed cost cases explicitly."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.poc_followup_corporate import CompletedFollowupData
from skills.strategy_comparison_data import require_profile_evidence

ROOT = Path(__file__).resolve().parents[1]


class ReusedFollowupData(CompletedFollowupData):
    def __init__(self, *, online=False, donor_manifest):
        super().__init__(online=online)
        path = Path(donor_manifest).resolve()
        path.relative_to(ROOT/'.cache/poc-followup-20261014')
        sources = Sources(ROOT)
        manifest = json.loads(sources.bind(str(path.relative_to(ROOT)), digest(path)).read_text())
        if manifest.get('schema') != 'poc_followup_profile_donors_v1':
            raise ValueError('Unknown explicit profile reuse manifest')
        self.followup_donors = {}
        for ref in manifest['reports']:
            report = json.loads(sources.bind(ref['path'], ref['sha256']).read_text())
            if (report.get('all_completed') is not True
                    or (report['start'], report['end']) != ('2024-01-02', '2026-10-02')
                    or report.get('followup', {}).get('preregistration', {}).get('sha256')
                       != digest(ROOT/'docs/prereg_poc_followup_20261014.json')):
                raise ValueError('Profile donor is not a complete registered account')
            sources.closure(report, ref['path'])
            for eid, value in report['shared_profile_results'].items():
                if eid != value['event_id']:
                    raise ValueError('Profile donor identity differs')
                if eid in self.profile_memo and self.profile_memo[eid] != value:
                    raise ValueError('Conflicting sealed profile evidence: '+eid)
                self.profile_memo[eid] = deepcopy(value)
                self.followup_donors.setdefault(eid, ref['path'])
        for name, sha in sources.used.items():
            self._bind(ROOT/name, sha)
        self._bind(Path(__file__))

    def profile(self, arm, event):
        result = super().profile(arm, event)
        if event['event_id'] in self.followup_donors:
            require_profile_evidence(result, event)
            self.used[arm][event['event_id']] = deepcopy(result)
            self.profile_origins[event['event_id']] = 'sealed_followup_profile:'+self.followup_donors[event['event_id']]
        return result
