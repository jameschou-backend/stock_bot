from datetime import datetime, timezone
import json

import pytest

from scripts import research_poc_optimization_completion as subject
from skills.all_entry_accounts import digest


def test_combined_attempt_ceiling_keeps_the_old_budget(tmp_path, monkeypatch):
    old, new = tmp_path/'old', tmp_path/'new'
    monkeypatch.setattr(subject, 'OLD_BASE', old)
    for folder, count in [(old/'odd-v1/attempts', 99), (new/'attempts', 1)]:
        folder.mkdir(parents=True)
        for i in range(count):
            (folder/f'{i}.json').write_text('{}')
    client = subject.OptimizationOdd.__new__(subject.OptimizationOdd)
    client.cache = new
    monkeypatch.setattr(subject.OddData, '_dispatch_permission', lambda *args: pytest.fail('Must not dispatch'))
    with pytest.raises(subject.ComparisonDataRequired, match='Combined original'):
        client._dispatch_permission({}, {}, None, None)


def test_remaining_budget_still_calls_original_origin_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, 'OLD_BASE', tmp_path/'old')
    client = subject.OptimizationOdd.__new__(subject.OptimizationOdd)
    client.cache = tmp_path/'new'
    seen = []
    def guarded(self, *args):
        seen.append(args)
        raise subject.ComparisonDataRequired('shared origin stop')
    monkeypatch.setattr(subject.OddData, '_dispatch_permission', guarded)
    with pytest.raises(subject.ComparisonDataRequired, match='shared origin stop'):
        client._dispatch_permission({'url': 'original'}, {'in_flight': True}, 'hold', 'proof')
    assert seen == [({'url': 'original'}, {'in_flight': True}, 'hold', 'proof')]


def authorization_client(tmp_path, age):
    spec = tmp_path/'spec.json'
    spec.write_text('{}')
    client = subject.OptimizationOdd.__new__(subject.OptimizationOdd)
    client.root, client.refs = tmp_path, {}
    client.auth = tmp_path/'authorization.json'
    client.expected_auth = dict(schema='poc_optimization_odd_authorization_v1',
        prereg_path='spec.json', prereg_sha256=digest(spec))
    now = 1800000000
    client.clock = lambda: now
    client.auth.write_text(json.dumps(dict(client.expected_auth,
        created_at=datetime.fromtimestamp(now-age, timezone.utc).isoformat())))
    return client


def test_new_scope_does_not_disable_expiry_or_rewrite_old_evidence(tmp_path):
    client = authorization_client(tmp_path, 86401)
    before = client.auth.read_bytes()
    with pytest.raises(subject.ComparisonDataRequired, match='authorization expired'):
        client._authorization(fresh=True)
    assert client.auth.read_bytes() == before
    assert client._authorization(fresh=False)['sha256'] == digest(client.auth)


def test_new_scope_rejects_altered_authorization(tmp_path):
    client = authorization_client(tmp_path, 1)
    value = json.loads(client.auth.read_text())
    value['schema'] = 'unregistered'
    client.auth.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='authorization scope differs'):
        client._authorization(fresh=True)


def test_new_scope_binds_registered_spec(tmp_path):
    client = authorization_client(tmp_path, 1)
    (tmp_path/'spec.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='source hash differs'):
        client._authorization(fresh=True)
