import pytest

from scripts.summarize_poc_broker_account import Sources
from scripts.summarize_poc_risk_controls import PublicationSources
from skills.all_entry_accounts import digest


def test_shared_source_closures_keep_the_original_bindings_and_snapshot_policy(tmp_path):
    (tmp_path/'data').write_text('prices')
    (tmp_path/'model.py').write_text('current version')
    (tmp_path/'snapshot').write_text('sealed version')
    report = dict(source_sha256={'data':digest(tmp_path/'data'),'model.py':digest(tmp_path/'snapshot')},
        source_snapshots={'model.py':dict(path='snapshot',sha256=digest(tmp_path/'snapshot'))})
    old, new = Sources(tmp_path), PublicationSources(tmp_path)
    for name in ('one','two'):
        old.closure(report,name)
        new.closure(report,name)
    assert not new.used  # No claim of byte verification before finish.
    new.finish()
    assert new.used==old.used and new.snapshot_resolutions==old.snapshot_resolutions


def test_mutation_after_collection_or_direct_read_prevents_publication(tmp_path):
    path=tmp_path/'data';path.write_text('original')
    for direct in (False,True):
        path.write_text('original')
        sources=PublicationSources(tmp_path)
        if direct:sources.bind('data',digest(path))
        else:sources.closure(dict(source_sha256={'data':digest(path)}),'report')
        path.write_text('changed')
        with pytest.raises(ValueError,match='Source hash mismatch'):
            sources.finish()


def test_conflicting_versions_are_rejected_even_before_final_byte_verification(tmp_path):
    path=tmp_path/'data';path.write_text('original')
    for direct in (False,True):
        sources=PublicationSources(tmp_path)
        if direct:sources.bind('data',digest(path))
        else:sources.closure(dict(source_sha256={'data':digest(path)}),'one')
        with pytest.raises(ValueError,match='Conflicting publication source'):
            sources.closure(dict(source_sha256={'data':'0'*64}),'two')


def test_collected_source_cannot_escape_root_through_a_symlink(tmp_path):
    root=tmp_path/'root';root.mkdir()
    external=tmp_path/'external';external.write_text('external data')
    (root/'escape').symlink_to(external)
    sources=PublicationSources(root)
    sources.closure(dict(source_sha256={'escape':digest(external)}),'report')
    with pytest.raises(ValueError):sources.finish()
