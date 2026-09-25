import json

import pandas as pd
import pytest

from scripts.reconcile_provider_tape_conflicts import compare, verify
from skills.board_tape_reconciliation import summarize_ticks
from skills.replay_market_feeds import ReplayDataUnavailable


def frame(prices=(10.,11.),volumes=(2,3),times=('09:01:01','09:02:01')):
    return pd.DataFrame(dict(date=['2024-01-02']*len(prices),stock_id=['1234']*len(prices),
        deal_price=prices,volume=volumes,Time=times,TickType=['2']*len(prices)))


IDENTITY=dict(stock_id='1234',date='2024-01-02',market='TPEX')


def official(source):
    return summarize_ticks(source,'1234','2024-01-02','TPEX')


def test_refreshed_provider_data_can_resolve_aggregate_missing_record():
    full=frame()
    old=full.iloc[:1]
    result=compare(old,full,IDENTITY,official(full))
    assert result['added_records']==1
    assert result['result']['same_scope_aggregate_matched'] is True
    assert result['result']['tick_sequence_complete'] is False
    assert result['result']['accepted_for_strict_replay'] is False


def test_unchanged_missing_trade_stays_a_conflict_without_imputation():
    full=frame(); partial=full.iloc[:1]
    result=compare(partial,partial.copy(),IDENTITY,official(full))
    assert result['payload_unchanged'] is True
    assert result['added_records']==result['removed_records']==0
    assert result['result']['quarantined'] is True
    assert result['result']['differences']['shares']==dict(tape=2000,official=5000)


def test_same_quantity_with_price_difference_is_not_tolerated():
    full=frame(); wrong=frame(prices=(10.,10.95))
    result=compare(wrong,wrong.copy(),IDENTITY,official(full))
    assert result['result']['same_scope_aggregate_matched'] is False
    assert 'amount_cents' in result['result']['differences']


def test_equal_timestamp_reordering_is_a_payload_change():
    first=frame(times=('09:01:01','09:01:01'))
    second=first.iloc[::-1].reset_index(drop=True)
    result=compare(first,second,IDENTITY,official(first))
    assert result['same_record_multiset'] is True
    assert result['payload_unchanged'] is False


def test_wrong_date_cannot_enter_the_comparison():
    full=frame(); wrong=frame(); wrong['date']='2024-01-03'
    with pytest.raises(ReplayDataUnavailable,match='identity'):
        compare(full,wrong,IDENTITY,official(full))


def test_modified_report_is_rejected_before_replay(tmp_path):
    path=tmp_path/'report.json'
    path.write_text(json.dumps({'summary':{'conflicts':0}}))
    path.with_suffix('.sha256').write_text('a'*64)
    with pytest.raises(ValueError,match='hash'):
        verify(path)
