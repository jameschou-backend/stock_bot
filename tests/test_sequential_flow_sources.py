import numpy as np
import pandas as pd
from scripts.prepare_sequential_flows_20261003 import combine_versions


def rows(a,b):
    return pd.DataFrame([dict(date='2025-01-02',stock_id='2330',foreign=a,trust=b)])


def test_conflict_never_latest_wins_and_survives_checkpoint():
    result=combine_versions([rows(1,2),rows(3,2)])
    assert result.conflict.iloc[0] and pd.isna(result.foreign.iloc[0])
    again=combine_versions([result,rows(3,2)])
    assert again.conflict.iloc[0] and pd.isna(again.trust.iloc[0])


def test_missing_category_is_not_zero_or_dropped():
    result=combine_versions([rows(np.nan,2),rows(0,2)])
    assert result.conflict.iloc[0] and pd.isna(result.foreign.iloc[0])
    same=combine_versions([rows(0,2),rows(0,2)])
    assert not same.conflict.iloc[0] and same.foreign.iloc[0]==0
