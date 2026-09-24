import hashlib
import json

import pytest

from scripts.audit_oddlot_scope_20260925 import DailyEvidence, csv_bytes, demand


def order(day='2022-01-04', sid='0050', side='buy', request=800, **extra):
    return dict(date=day, stock_id=sid, side=side, requested_qty=request,
                filled_qty=0, channel='odd', failure='capacity_or_cash_zero', **extra)


def test_rejected_demand_is_preserved_and_duplicate_side_is_grouped():
    account = dict(orders=[order(), order(request=200), order(side='sell', request=1),
                           dict(order(), channel='board', requested_qty=1000)],
                   trades=[dict(date='2022-01-04', stock_id='0050', side='buy', channel='odd',
                                qty=100, gross=10000., sequence=4)])
    rows = demand(account, 'case')
    assert len(rows) == 2
    buy, sell = rows
    assert (buy['order_count'], buy['trade_count'], buy['requested_shares'], buy['filled_shares']) == (2, 1, 1000, 100)
    assert buy['order_indices'] == [0, 1] and buy['trade_sequences'] == [4]
    assert sell['order_count'] == 1 and sell['filled_shares'] == 0
    assert sell['failures'] == ['capacity_or_cash_zero']


def test_trade_without_matching_order_is_an_error():
    with pytest.raises(ValueError, match='ledger mismatch'):
        demand(dict(orders=[], trades=[dict(date='2022-01-04', stock_id='0050', side='buy',
                    channel='odd', qty=10, gross=100., sequence=1)]), 'case')


def test_daily_record_never_implies_a_sequence_tape(tmp_path, monkeypatch):
    import scripts.audit_oddlot_scope_20260925 as scope
    monkeypatch.setattr(scope, 'ROOT', tmp_path)
    source = tmp_path/'source'
    feeds = source/'inputs/execution-feeds'
    feeds.mkdir(parents=True)
    raw_name, rows_name = 'odd-twse-2022-01-04.raw.json', 'odd-twse-2022-01-04.rows.json'
    raw = b'{"data": "a daily aggregate"}'
    (feeds/raw_name).write_bytes(raw)
    parsed = dict(raw_sha256=hashlib.sha256(raw).hexdigest(), rows={'0050': dict(
        source_date='2022-01-04', market='twse', odd_shares=1000, odd_bid=100, odd_ask=101, bid_qty=3, ask_qty=4)})
    (feeds/rows_name).write_text(json.dumps(parsed))
    hashes = {name: scope.sha(feeds/name) for name in (raw_name, rows_name)}
    (feeds/'index.json').write_text(json.dumps(dict(files_sha256=hashes, entries={
        'odd:twse:2022-01-04': dict(raw_file=raw_name, rows_file=rows_name)})))
    manifest = {str(p.relative_to(source)):scope.sha(p) for p in feeds.iterdir()}
    observed = DailyEvidence(source, manifest).get('2022-01-04', '0050')
    assert observed['daily_record_present'] and observed['daily_volume_shares'] == 1000
    assert 'accepted_sequence_tape_present' not in observed
    (feeds/raw_name).write_text('{}')
    with pytest.raises(ValueError, match='Sealed daily evidence changed'):
        DailyEvidence(source, manifest).get('2022-01-04', '0050')


def test_csv_has_quoted_list_evidence_and_preserves_leading_zero_text():
    text = csv_bytes([dict(stock_id='0050', failures=['one', 'two'])]).decode()
    assert '0050' in text and '"[""one"", ""two""]"' in text
