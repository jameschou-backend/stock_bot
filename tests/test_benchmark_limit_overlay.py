from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from skills import benchmark_limit_overlay as overlay
from skills.midpoint_replay import midpoint_match


def evidence():
    return json.loads((Path(__file__).resolve().parents[1]/overlay.EVIDENCE_PATH).read_text())


def fixture_sources(tmp_path, monkeypatch):
    """Small synthetic copies exercise the loader without external cache files."""
    parent = tmp_path/'sources'; parent.mkdir()
    limit = parent/'0050-TaiwanStockPriceLimit.parquet'
    pd.DataFrame([dict(stock_id='0050',date=overlay.DAY,reference_price=146.2,
                       limit_down=132.,limit_up=160.5)]).to_parquet(limit,index=False)
    meta = dict(evidence()['provider_query'],sha256=overlay._sha(limit))
    limit.with_suffix('.json').write_text(json.dumps(meta))
    pd.DataFrame(columns=['stock_id','event_date']).to_parquet(parent/'events.parquet',index=False)
    pd.DataFrame([dict(stock_id='0050',CashExDividendTradingDate='2025-01-17',
        StockExDividendTradingDate='')]).to_parquet(parent/'0050-TaiwanStockDividend.parquet',index=False)
    for day,close,change in [(overlay.PRIOR,'146.20','7.05'),(overlay.DAY,'160.80','14.60')]:
        raw = tmp_path/'test/acquisition/raw'/(day+'.json');raw.parent.mkdir(parents=True,exist_ok=True)
        raw.write_text(json.dumps(dict(date=day.replace('-',''),stat='OK',tables=[dict(
            fields=['證券代號','開盤價','最高價','最低價','收盤價','漲跌價差'],
            data=[['0050',close,close,close,close,change]])])))
        receipt = tmp_path/'test/acquisition/receipts'/(day+'.json');receipt.parent.mkdir(parents=True,exist_ok=True)
        receipt.write_text(json.dumps(dict(date=day,market='TWSE',http_status=200,accepted=True,
            params=dict(date=day.replace('-',''),response='json',type='ALLBUT0999'),
            raw_path=str(raw.relative_to(tmp_path)),raw_sha256=overlay._sha(raw))))
    doc=tmp_path/overlay.EVIDENCE_PATH;doc.parent.mkdir()
    value=evidence()
    def seal():
        value['source_sha256']={str(p.relative_to(tmp_path)):overlay._sha(p)
            for p in tmp_path.rglob('*') if p.is_file() and p != doc}
        doc.write_text(json.dumps(value))
        monkeypatch.setattr(overlay,'EVIDENCE_SHA256',overlay._sha(doc))
    seal()
    return parent,doc,seal


def test_narrow_overlay_preserves_provider_other_dates_and_other_stocks():
    x=overlay.BenchmarkLimitOverlay(evidence(),{'some/source':'sha'})
    original={overlay.DAY:dict(lower=132.,upper=160.5),
              '2025-04-07':dict(lower=158.5,upper=193.5),
              '2026-07-31':dict(lower=1.,upper=2.)}
    before=deepcopy(original)
    fixed=x.apply('0050',original)
    assert original==before
    assert fixed[overlay.DAY]==dict(lower=131.6,upper=160.8)
    assert fixed['2025-04-07']==original['2025-04-07']
    assert fixed['2026-07-31']==original['2026-07-31']
    assert x.apply('2330',original)==original
    assert x.apply('0050',original)==fixed and len(x.audit_rows)==1
    assert x.audit_rows[0]['provider']==dict(lower=132.,upper=160.5)
    assert x.audit_rows[0]['observed_official_daily_limits'] is False


@pytest.mark.parametrize('wrong', [{}, {overlay.DAY:dict(lower=131.6,upper=160.8)},
                                  {overlay.DAY:dict(lower=132.,upper=160.6)}])
def test_unexpected_original_limits_fail_instead_of_widening(wrong):
    x=overlay.BenchmarkLimitOverlay(evidence(),{})
    with pytest.raises(ValueError,match='exact original'):
        x.apply('0050',wrong)


def test_no_new_dates_or_official_limit_promotion_can_be_inserted():
    value=evidence();value['patches'][0]['date']='2026-07-31'
    with pytest.raises(ValueError,match='identity'):
        overlay.BenchmarkLimitOverlay(value,{})
    value=evidence();value['patches'][0]['observed_official_daily_limits']=True
    with pytest.raises(ValueError,match='identity'):
        overlay.BenchmarkLimitOverlay(value,{})


def test_positive_query_and_hash_bound_load(tmp_path,monkeypatch):
    _,_,_=fixture_sources(tmp_path,monkeypatch)
    loaded=overlay.load_overlay(tmp_path)
    assert loaded.refs[overlay.EVIDENCE_PATH]==overlay.EVIDENCE_SHA256
    assert loaded.apply('0050',{overlay.DAY:overlay.PROVIDER})[overlay.DAY]==overlay.DERIVED


def test_source_and_manifest_mutations_fail(tmp_path,monkeypatch):
    parent,doc,_=fixture_sources(tmp_path,monkeypatch)
    meta=parent/'0050-TaiwanStockPriceLimit.json'
    original=meta.read_text();meta.write_text(original+' ')
    with pytest.raises(ValueError,match='source hash'):
        overlay.load_overlay(tmp_path)
    meta.write_text(original);doc.write_text(doc.read_text()+' ')
    with pytest.raises(ValueError,match='evidence hash'):
        overlay.load_overlay(tmp_path)


def test_query_change_fails_even_with_self_consistent_new_hashes(tmp_path,monkeypatch):
    parent,_,seal=fixture_sources(tmp_path,monkeypatch)
    meta=parent/'0050-TaiwanStockPriceLimit.json'
    value=json.loads(meta.read_text());value['stock_id']='2330';meta.write_text(json.dumps(value));seal()
    with pytest.raises(ValueError,match='query identity'):
        overlay.load_overlay(tmp_path)


def test_reference_change_fails_even_when_metadata_hash_matches(tmp_path,monkeypatch):
    parent,_,seal=fixture_sources(tmp_path,monkeypatch)
    raw=parent/'0050-TaiwanStockPriceLimit.parquet'
    frame=pd.read_parquet(raw);frame.loc[0,'reference_price']=146.3;frame.to_parquet(raw,index=False)
    meta=raw.with_suffix('.json');value=json.loads(meta.read_text());value['sha256']=overlay._sha(raw);meta.write_text(json.dumps(value));seal()
    with pytest.raises(ValueError,match='reference/limits'):
        overlay.load_overlay(tmp_path)


def test_conflicting_corporate_event_fails(tmp_path,monkeypatch):
    parent,_,seal=fixture_sources(tmp_path,monkeypatch)
    pd.DataFrame([dict(stock_id='0050',event_date='2025-04-10')]).to_parquet(parent/'events.parquet',index=False);seal()
    with pytest.raises(ValueError,match='corporate event'):
        overlay.load_overlay(tmp_path)


def test_correct_upper_limit_does_not_imply_a_limit_up_buy_fill():
    value=midpoint_match(160.8,160.8,1000000,1000000,1000,'buy',160.8,131.6,160.8,'board')
    assert value['filled_qty']==0 and value['failure']=='midpoint_limit_not_crossed'
