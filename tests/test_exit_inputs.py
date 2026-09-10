"""Synthetic provenance/clone tests; no pandas, database, API or returns."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import prepare_exit_inputs as module


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    code = write(root, 'skills/replay_market_feeds.py', b'synthetic parser')
    spec = write(root, module.SPEC, b'synthetic predeclared protocol')
    overrides = write(root, module.OVERRIDES, {'overrides': {}})
    raw_files = {name:write(root, module.INPUT/name, ('synthetic '+name).encode()) for name in
                 ('quotes.parquet','calendar.parquet','companies.parquet','events.parquet')}
    raw_hash = write(root, module.INPUT/'manifest.json', {'schema':1, 'files_sha256':raw_files})
    nested = module.SIGNALS/'adjraw'
    adjusted_hash = write(root, nested/'manifest.json', {'schema':1,
        'files_sha256':{'raw.json':write(root,nested/'raw.json',[])},
        'plan_sha256':write(root,nested/'plan.json',{'dates':[]}),
        'attempts_sha256':write(root,nested/'attempts.json',[])})
    old = Path('.cache/older/inputs.json')
    old_hash = write(root, old, {'schema':1, 'parent_files_sha256':{
        str(module.INPUT/'quotes.parquet'):raw_files['quotes.parquet']}})
    signal_files = {name:write(root,module.SIGNALS/name,('synthetic '+name).encode())
                    for name in ('close-official.parquet','signals.json')}
    signal_hash = write(root,module.SIGNALS/'manifest.json',{'schema':1,'files_sha256':signal_files,
        'source_files_sha256':{str(module.INPUT/'manifest.json'):raw_hash,
            str(nested/'manifest.json'):adjusted_hash,str(old):old_hash},
        'spec_path':str(module.SPEC),'spec_sha256':spec})
    feed = module.INPUT/'execution-feeds'
    state = {'schema':1,'parser_sha256':code,'entries':{'odd:twse:2022-01-03':{
        'raw_file':'one.raw.json','rows_file':'one.rows.json'}},
        'files_sha256':{'one.raw.json':write(root,feed/'one.raw.json',{'payload':'one'}),
                        'one.rows.json':write(root,feed/'one.rows.json',{'rows':{}})},
        'request_counters':{'official_http_requests':7,'finmind_requests':2},
        'limitations':['Volume is not a fill guarantee']}
    index_hash = write(root,feed/'index.json',state)
    div_hash = write(root,module.INPUT/'dividends/0050.parquet',b'synthetic dividend policy')
    write(root,module.INPUT/'dividends/4123.parquet',b'downloaded policy unused by old ledger')
    report_files = {name:write(root,module.REPORT/name,{'synthetic':True})
                    for name in ('report.json','summary.json','prepared-accounts.json')}
    write(root,module.REPORT/'manifest.json',{'schema':1,'offline_identical':True,'live_qualified':False,
        'code_sha256':{'skills/replay_market_feeds.py':code},'files_sha256':report_files,
        'source_files_sha256':{str(module.INPUT/'manifest.json'):raw_hash,
            str(module.SIGNALS/'manifest.json'):signal_hash,str(module.SPEC):spec,str(module.OVERRIDES):overrides},
        'execution_feeds':{**state,'manifest_sha256':index_hash},
        'corporate_sources':{'files_sha256':{'0050.parquet':div_hash}}})
    return root


def snapshot(folder):
    return {str(p.relative_to(folder)):(p.read_bytes(),p.stat().st_mtime_ns)
            for p in folder.rglob('*') if p.is_file()}


def child_index(root):
    path = root/module.DESTINATION/'execution-feeds/index.json'
    return path,json.loads(path.read_text())


def test_clone_preserves_parent_and_references_large_files_without_links(parent):
    before = snapshot(parent)
    data = module.prepare(root=parent)
    child = parent/module.DESTINATION
    assert {name:(parent/name).read_bytes() for name in before} == {name:pair[0] for name,pair in before.items()}
    assert all((parent/name).stat().st_mtime_ns == pair[1] for name,pair in before.items())
    assert not (child/'quotes.parquet').exists()
    assert data['references']['quotes']['path'] == str(module.INPUT/'quotes.parquet')
    assert data['finmind_requests'] == data['official_http_requests'] == 0
    assert data['seed_execution_index']['request_counters']['official_http_requests'] == 7
    assert data['parent_report_dividend_files'] == ['0050.parquet']
    assert set(data['seed_dividend_hashes']) == {'0050.parquet','4123.parquet'}
    for name,original in data['clone_parent_paths'].items():
        assert (child/name).read_bytes() == (parent/original).read_bytes()
        assert not os.path.samefile(child/name,parent/original)
    assert not os.path.samefile(child/'execution-feeds/index.json', parent/module.INPUT/'execution-feeds/index.json')


def test_repeated_prepare_is_byte_and_mtime_idempotent(parent):
    first = module.prepare(root=parent)
    before = snapshot(parent)
    assert module.prepare(root=parent) == first
    assert module.verify(root=parent) == first
    assert snapshot(parent) == before


def test_child_can_append_execution_and_dividend_without_losing_new_records(parent):
    module.prepare(root=parent)
    path,state = child_index(parent)
    child = parent/module.DESTINATION
    raw = write(child,'execution-feeds/two.raw.json',{'payload':'two'})
    normalized = write(child,'execution-feeds/two.rows.json',{'rows':{}})
    state['files_sha256'].update({'two.raw.json':raw,'two.rows.json':normalized})
    state['entries']['odd:twse:2022-01-04'] = {'raw_file':'two.raw.json','rows_file':'two.rows.json'}
    state['request_counters']['official_http_requests'] += 1
    write(child,'execution-feeds/index.json',state)
    write(child,'dividends/1101.parquet',b'newly prepared policy')
    before = snapshot(parent)
    module.prepare(root=parent)
    assert snapshot(parent) == before
    assert json.loads(path.read_text())['request_counters']['official_http_requests'] == 8


@pytest.mark.parametrize('source',[module.INPUT/'quotes.parquet',module.SIGNALS/'close-official.parquet',
    module.REPORT/'report.json',module.SPEC,module.OVERRIDES,
    module.SIGNALS/'adjraw/plan.json',module.INPUT/'execution-feeds/one.raw.json',
    module.INPUT/'dividends/0050.parquet',module.INPUT/'dividends/4123.parquet'])
def test_changed_parent_source_is_rejected_and_child_is_untouched(parent, source):
    module.prepare(root=parent)
    child_before = snapshot(parent/module.DESTINATION)
    (parent/source).write_bytes(b'changed after seal')
    with pytest.raises(module.InputChanged, match='changed'):
        module.prepare(root=parent)
    assert snapshot(parent/module.DESTINATION) == child_before


@pytest.mark.parametrize('mutation',['entry','hash','counter','parser','seed_file','seed_manifest','index_hardlink'])
def test_inherited_child_evidence_cannot_change(parent, mutation):
    module.prepare(root=parent)
    path,state = child_index(parent)
    child = parent/module.DESTINATION
    if mutation == 'entry':
        state['entries']['odd:twse:2022-01-03']['raw_file'] = 'different.json'
    elif mutation == 'hash':
        del state['files_sha256']['one.raw.json']
    elif mutation == 'counter':
        state['request_counters']['official_http_requests'] = 0
    elif mutation == 'parser':
        state['parser_sha256'] = 'a'*64
    elif mutation == 'seed_file':
        (child/'dividends/0050.parquet').write_bytes(b'overwritten')
    elif mutation == 'seed_manifest':
        manifest = json.loads((child/'manifest.json').read_text())
        del manifest['seed_files_sha256']['dividends/0050.parquet']
        write(child,'manifest.json',manifest)
    elif mutation == 'index_hardlink':
        path.unlink()
        os.link(parent/module.INPUT/'execution-feeds/index.json',path)
    if mutation in {'entry','hash','counter','parser'}:
        write(child,'execution-feeds/index.json',state)
    with pytest.raises(module.InputChanged):
        module.verify(root=parent)


def test_missing_parent_manifest_is_not_rebuilt_or_fetched(parent):
    (parent/module.REPORT/'manifest.json').unlink()
    with pytest.raises(module.InputChanged, match='missing'):
        module.prepare(root=parent)
    assert not (parent/module.DESTINATION).exists()


@pytest.mark.parametrize('destination',[module.INPUT,module.INPUT/'new-child',Path('.cache'),module.REPORT])
def test_destination_cannot_overlap_old_evidence(parent,destination):
    before = snapshot(parent)
    with pytest.raises(module.InputChanged, match='overlaps'):
        module.prepare(root=parent,destination=destination)
    assert snapshot(parent) == before


def test_source_symlinks_are_rejected(parent):
    path = parent/module.INPUT/'quotes.parquet'
    copy = parent/'same-bytes.parquet'
    path.rename(copy)
    path.symlink_to(copy)
    with pytest.raises(module.InputChanged, match='Symlink'):
        module.prepare(root=parent)


def test_failed_copy_does_not_publish_partial_cache_or_modify_parent(parent,monkeypatch):
    before = snapshot(parent)
    def fail(*args,**kwargs):
        raise OSError('synthetic disk failure')
    monkeypatch.setattr(module.shutil,'copyfile',fail)
    with pytest.raises(OSError, match='disk failure'):
        module.prepare(root=parent)
    assert not (parent/module.DESTINATION).exists()
    assert snapshot(parent) == before
    assert not list((parent/'.cache').glob('.exit-research-inputs-*'))


def test_large_parent_file_is_hashed_once_even_with_multiple_manifest_references(parent,monkeypatch):
    counts = Counter()
    original = module._sha
    def counted(path):
        counts[Path(path)] += 1
        return original(path)
    monkeypatch.setattr(module,'_sha',counted)
    module.prepare(root=parent)
    assert counts[parent/module.INPUT/'quotes.parquet'] == 1
    counts.clear()
    module.verify(root=parent)
    assert counts[parent/module.INPUT/'quotes.parquet'] == 1


def test_parent_change_during_copy_is_detected(parent,monkeypatch):
    original = module.shutil.copyfile
    changed = False
    def mutate(source,dest):
        nonlocal changed
        result = original(source,dest)
        if not changed:
            changed = True
            (parent/module.INPUT/'quotes.parquet').write_bytes(b'concurrent change')
        return result
    monkeypatch.setattr(module.shutil,'copyfile',mutate)
    with pytest.raises(module.InputChanged, match='changed during'):
        module.prepare(root=parent)
    assert not (parent/module.DESTINATION).exists()


def test_parent_extra_dividend_inventory_changes_are_not_silently_reseeded(parent):
    module.prepare(root=parent)
    write(parent,module.INPUT/'dividends/1101.parquet',b'new parent policy')
    with pytest.raises(module.InputChanged, match='inventory changed'):
        module.prepare(root=parent)


def test_reference_cannot_be_redirected_to_another_valid_parent_file(parent):
    module.prepare(root=parent)
    child = parent/module.DESTINATION
    info = json.loads((child/'manifest.json').read_text())
    info['references']['quotes'] = info['references']['close_official']
    write(child,'manifest.json',info)
    with pytest.raises(module.InputChanged, match='reference changed'):
        module.verify(root=parent)


def test_prepare_and_verify_need_no_network(parent,monkeypatch):
    import socket
    def forbidden(*args,**kwargs):
        pytest.fail('Local namespace preparation must never access the network')
    monkeypatch.setattr(socket,'create_connection',forbidden)
    module.prepare(root=parent)
    module.verify(root=parent)


def test_malicious_parent_relative_path_is_rejected_before_copy(parent):
    path = parent/module.REPORT/'manifest.json'
    report = json.loads(path.read_text())
    report['files_sha256']['../../outside'] = 'a'*64
    write(parent,module.REPORT/'manifest.json',report)
    with pytest.raises(module.InputChanged, match='Unsafe'):
        module.prepare(root=parent)
    assert not (parent/module.DESTINATION).exists()
