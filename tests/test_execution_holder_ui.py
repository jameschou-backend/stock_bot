import hashlib
import pytest
from app import execution_holder_ui as ui
from scripts.research_exit_scenarios import write


def test_unsealed_or_incomplete_sources_never_show_results(tmp_path,monkeypatch):
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    path=tmp_path/'.cache/execution-stress/manifest.json'
    write(path,dict(offline_identical=False,live_qualified=False))
    with pytest.raises(ValueError,match='離線'):
        ui.signature('execution-stress')
    write(path,dict(offline_identical=True,live_qualified=False,files_sha256={}))
    with pytest.raises(ValueError,match='來源索引'):
        ui.signature('execution-stress')


def test_changed_child_engine_invalidates_cached_result(tmp_path,monkeypatch):
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    ui.verified.cache_clear()
    names=['.cache/execution-stress/summary.json','docs/prereg_execution_holder_20260911.md',
        'skills/execution_stress.py','scripts/research_execution_stress.py',
        '.cache/execution-stress/cases/control.json','.cache/execution-stress/cases/control_benchmark.json']
    for name in names:
        write(tmp_path/name,dict(live_qualified=False))
    files={n:hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in names}
    write(tmp_path/'.cache/execution-stress/manifest.json',dict(offline_identical=True,live_qualified=False,files_sha256=files))
    old=ui.signature('execution-stress')
    ui.verified('execution-stress',old)
    (tmp_path/'skills/execution_stress.py').write_text('changed')
    new=ui.signature('execution-stress')
    assert old!=new
    with pytest.raises(ValueError,match='已變更'):
        ui.verified('execution-stress',new)
    ui.verified.cache_clear()
