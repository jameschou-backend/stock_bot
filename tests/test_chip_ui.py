from copy import deepcopy
import hashlib

import pytest
from app import chip_research_ui as ui
from scripts.research_exit_scenarios import write


def test_rows_compare_same_coverage_and_do_not_fake_blocked_returns():
    summary=dict(benchmark=dict(total_return=2.),cases={})
    for mode in ui.LABELS:
        summary['cases'][mode]=dict(completed=True,summary=dict(final_nav=4e6,total_return=3.,max_drawdown=-.3))
    summary['cases']['available_trust']=dict(completed=True,summary=dict(total_return=2.5))
    summary['cases']['broker']=dict(completed=False,reason='missing delivery date')
    rows=ui.comparison_rows(summary)
    trust=next(r for r in rows if r['方法']==ui.LABELS['trust'])
    assert trust['總淨報酬（%）']==300
    assert trust['比0050多（百分點）']==100
    assert trust['比同資料對照多（百分點）']==50
    blocked=next(r for r in rows if r['方法']==ui.LABELS['broker'])
    assert '總淨報酬（%）' not in blocked


def test_source_edit_invalidates_cached_display(tmp_path,monkeypatch):
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    ui.verified.cache_clear()
    names=[str(ui.CACHE/'summary.json'),str(ui.CACHE/'cases/control.json'),
        'scripts/research_chip.py','skills/chip_research.py','docs/prereg_chip_20260911.md',
        '.cache/chip-inputs/manifest.json']
    for name in names:
        write(tmp_path/name,dict(live_qualified=False,unseen_validation=False))
    files={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in names}
    write(tmp_path/ui.CACHE/'manifest.json',dict(offline_identical=True,live_qualified=False,files_sha256=files))
    first=ui.signature()
    ui.verified(first)
    (tmp_path/'skills/chip_research.py').write_text('changed')
    second=ui.signature()
    assert second != first
    with pytest.raises(ValueError,match='已變更'):
        ui.verified(second)
    ui.verified.cache_clear()
