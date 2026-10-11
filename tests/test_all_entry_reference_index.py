from types import SimpleNamespace
import pandas as pd
import pytest

from skills.all_entry_reference_index import indexed_type,install_private_index
from skills.replay_corporate_actions import CorporateActions


def outcome(provider,sid,day,prior):
    try:return ('value',provider.reference_price(sid,day,prior))
    except Exception as exc:return (type(exc).__name__,str(exc))


def test_index_matches_original_errors_missing_and_split_without_future_lookup(tmp_path):
    rows=[]
    for i,(opening,reference) in enumerate([(100,99),(None,90),(float('nan'),80),
        (0,70),(-1,60),(float('inf'),50),(None,0),(None,float('nan'))]):
        rows.append(dict(stock_id=str(2300+i),event_date='2024-09-12',opening_ref=opening,ref_price=reference))
    rows += [dict(stock_id='2310',event_date='2024-09-12',opening_ref=1,ref_price=1)]*2
    frame=pd.DataFrame(rows)
    original=CorporateActions(frame,tmp_path/'old',offline=True)
    indexed=indexed_type(CorporateActions)(frame,tmp_path/'new',offline=True)
    for sid in ['0050','2330',*frame.stock_id]:
        for day in ['2024-09-11','2024-09-12','2024-09-13','2025-06-18']:
            for price in [None,0.,123.45]:
                assert outcome(indexed,sid,day,price)==outcome(original,sid,day,price)
    # A valid event next day must not be used as today's opening reference.
    assert indexed.reference_price('2300','2024-09-11',7.)==7.


def test_private_adapter_does_not_mutate_modules_and_preserves_calendar_override(tmp_path):
    study=SimpleNamespace(old=SimpleNamespace(TrackedCorporateActions=CorporateActions))
    runner=SimpleNamespace(study=study)
    install_private_index(runner)
    assert study.old.TrackedCorporateActions is CorporateActions
    assert runner.study is not study and runner.study.old is not study.old
    class Loader(runner.study.old.TrackedCorporateActions):
        def reference_price(self,sid,day,prior):
            return 12. if day=='2024-09-13' else super().reference_price(sid,day,prior)
    frame=pd.DataFrame([dict(stock_id='2330',event_date='2024-09-12',opening_ref=10.,ref_price=9.)])
    provider=Loader(frame,tmp_path,offline=True)
    assert provider.reference_price('2330','2024-09-13',11.)==12.
    assert provider.reference_price('2330','2024-09-12',11.)==10.


def test_new_scalar_semantics_require_review_instead_of_silent_override():
    class Changed(CorporateActions):
        def reference_price(self,*args):return 1.
    with pytest.raises(ValueError,match='renewed parity'):indexed_type(Changed)
