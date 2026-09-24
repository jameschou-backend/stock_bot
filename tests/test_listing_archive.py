import pytest
from scripts.audit_listing_archive import parse_table

HEADER='Company Name Date of Listing Code IPO Price Capital Stock\n'


@pytest.mark.parametrize('day',['20110704','2011/7/4','2011.07.04','100/07/04'])
def test_listing_formats(day):
    assert parse_table(HEADER+day+'\n1234\n20\n100,000,000',2011)=={'1234':'2011-07-04'}


@pytest.mark.parametrize('body',[
    'Emerging Stock Market\n20110704 1234',
    '20120704 1234',
    '20110704 1234\n20110705 1234',
    '20110230 1234',
    '20110704 123456',
])
def test_wrong_market_year_duplicates_invalid_date_or_warrant_blocked(body):
    with pytest.raises(ValueError):parse_table(HEADER+body,2011)


def test_daily_quote_is_not_listing_evidence():
    with pytest.raises(ValueError):parse_table('date close\n20110704 1234',2011)
