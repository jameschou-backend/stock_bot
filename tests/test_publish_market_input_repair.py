from copy import deepcopy
import json
from pathlib import Path
import pytest
from scripts.publish_market_input_repair import verify_run_scopes, load_capacity_conflicts
from skills.ordinary_volume_bundle import digest


def fixture():
    return dict(input_bundle='inputs-v2',data_revision='repaired',preparation=False,
                start='2019-01-02',end='2026-09-09',initial_cash=1_000_000,
                cases={arm:dict(completed=True) for arm in ('three_black','benchmark')},
                all_completed=True,live_qualified=False,actual_fill_verified=False)


@pytest.mark.parametrize('mutation', ['missing_arm','wrong_end','cash','complete','preparation','promotion'])
def test_mismatched_strict_or_research_scope_cannot_be_published(mutation):
    original = fixture(); bad = deepcopy(original)
    if mutation == 'missing_arm': bad['cases'].pop('three_black')
    if mutation == 'wrong_end': bad['end'] = '2025-12-31'
    if mutation == 'cash': bad['initial_cash'] = 2_000_000
    if mutation == 'complete': bad['cases']['benchmark']['completed'] = False
    if mutation == 'preparation': bad['preparation'] = True
    if mutation == 'promotion': bad['live_qualified'] = True
    with pytest.raises(ValueError): verify_run_scopes([original, original, bad], 'inputs-v2')


def test_incomplete_strict_run_is_allowed_only_as_an_explicit_incomplete_result():
    original = fixture(); strict = deepcopy(original)
    strict['all_completed'] = strict['cases']['three_black']['completed'] = False
    verify_run_scopes([original, original, strict], 'inputs-v2')


def capacity_fixture(root):
    raw = root/'raw.json'; raw.write_text('original')
    row = dict(fill_check='capacity_conflict', sequence=1, cumulative_qty=10000, verified_capacity=9000)
    parent = root/'parent.json'
    parent.write_text(json.dumps(dict(schema='ordinary_capacity_evidence_v1', rows=[row],
        counts=dict(capacity_conflict=1), source_sha256={raw.name:digest(raw)}, output_sha256={}, code_sha256={})))
    child = root/'conflicts.json'
    value = dict(schema='ordinary_capacity_conflicts_v1', source_report=parent.name,
                 source_sha256=digest(parent), rows=[row], count=1)
    child.write_text(json.dumps(value))
    return raw, child, value


def test_capacity_conflicts_follow_parent_digest_and_bind_underlying_evidence(tmp_path):
    _, child, _ = capacity_fixture(tmp_path)
    refs = {}
    assert load_capacity_conflicts(tmp_path, Path(child.name), refs)['count'] == 1
    assert set(refs) == {'conflicts.json', 'parent.json', 'raw.json'}


@pytest.mark.parametrize('mutation', ['source', 'dropped_conflict', 'parent_digest'])
def test_capacity_conflict_publication_rejects_changed_evidence(tmp_path, mutation):
    raw, child, value = capacity_fixture(tmp_path)
    if mutation == 'source': raw.write_text('changed')
    elif mutation == 'dropped_conflict': value.update(rows=[], count=0)
    else: value['source_sha256'] = '0'*64
    child.write_text(json.dumps(value))
    with pytest.raises(ValueError): load_capacity_conflicts(tmp_path, Path(child.name), {})
