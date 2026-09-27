import json
import pytest
from scripts.prepare_theme_chips import load_complete
from scripts.research_exit_scenarios import sha


def fixture(root):
    day='2026-04-02';path=root/f'{day}.parquet';path.write_bytes(b'frozen snapshot')
    plan=dict(schema='theme_chip_inputs_v1',dates=[day])
    meta=dict(date=day,rows=17,sha256=sha(path),retrieved_at=123,cache_hit=False)
    (root/f'{day}.json').write_text(json.dumps(meta))
    saved=dict(**plan,files={day:dict(meta,reused=False)},requests_this_run=1,elapsed_seconds=2)
    (root/'manifest.json').write_text(json.dumps(saved))
    return plan


def test_completed_collection_reuses_without_rewriting_any_evidence(tmp_path):
    plan=fixture(tmp_path);before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    result=load_complete(tmp_path,plan)
    assert result['requests_this_run']==0 and result['reused_complete'] is True
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()}==before


@pytest.mark.parametrize('defect',['data','metadata','plan'])
def test_completed_cache_corruption_fails_closed(tmp_path,defect):
    plan=fixture(tmp_path)
    if defect=='data':(tmp_path/'2026-04-02.parquet').write_bytes(b'changed')
    elif defect=='metadata':(tmp_path/'2026-04-02.json').write_text('{}')
    else:plan['dates']=[]
    with pytest.raises(ValueError):load_complete(tmp_path,plan)
