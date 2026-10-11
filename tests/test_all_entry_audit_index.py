import pandas as pd
import pytest

from skills.all_entry_audit_index import QuoteRows, SOURCE, private_auditor
from skills import poc_range_audit as original
from test_poc_executable_replay import audit_case  # noqa: F401
from test_poc_gap_audit import gap_case  # noqa: F401
from test_poc_range_audit import range_case, range_gap_case  # noqa: F401


def test_quote_index_preserves_order_duplicates_empty_keys_and_frame_lifetime():
    lookup = QuoteRows()
    source = pd.DataFrame(dict(date=pd.to_datetime(['2024-01-02', '2024-01-01', '2024-01-02', None]),
                               stock_id=['2330', '2317', '2330', '2330'], close=[3., 1., 2., 4.]),
                          index=[8, 5, 8, 7])
    for day, sid in [('2024-01-02', '2330'), ('2024-01-01', '2317'), ('2024-01-01', '2330'),
                     ('2024-01-02', '9999'), (pd.NaT, '2330')]:
        expected = source.loc[source.date.eq(pd.Timestamp(day)) & source.stock_id.eq(sid)]
        pd.testing.assert_frame_equal(lookup(source, day, sid), expected)
    assert len(lookup._frames) == 1
    del source
    assert not lookup._frames


@pytest.mark.parametrize('fixture', ['range_case', 'range_gap_case'])
def test_complete_and_gap_audit_reports_match_the_original(request, fixture):
    args = request.getfixturevalue(fixture)
    before = original.audit_range_execution
    indexed = private_auditor()
    assert indexed.audit_range_execution(*args) == original.audit_range_execution(*args)
    assert original.audit_range_execution is before
    assert indexed.audit_range_execution.__globals__ is not original.__dict__


@pytest.mark.parametrize('change', ['cash', 'fill', 'quote_duplicate', 'gap_reason'])
def test_same_invalid_evidence_is_rejected_with_the_same_error(request, change):
    args = request.getfixturevalue('range_gap_case' if change == 'gap_reason' else 'range_case')
    account = args[0]
    if change == 'cash':
        account['daily'][0]['cash'] += 1
    elif change == 'fill':
        account['orders'][0]['filled_qty'] += 1000
    elif change == 'quote_duplicate':
        args = (*args[:4], pd.concat([args[4], args[4].iloc[:1]]), *args[5:])
    else:
        account['data_gap_exclusions'][0]['failure_reason'] = 'invented failure'
    with pytest.raises(Exception) as expected:
        original.audit_range_execution(*args)
    with pytest.raises(type(expected.value)) as actual:
        private_auditor().audit_range_execution(*args)
    assert str(actual.value) == str(expected.value)


def test_unknown_audit_source_is_rejected(tmp_path):
    changed = tmp_path / 'audit.py'
    changed.write_bytes(SOURCE.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='changed'):
        private_auditor(changed)
