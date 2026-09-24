"""Synthetic protocol checks; real vendor sample remains a local research file."""
import pytest

from skills.tpex_mth import FORMAT_ID, inspect_mth_sample


def record(side, *, trade="00000001", at="091007334514", kind="2", qty=16, price="0142.50"):
    return (f"202309153105  {side}{kind}{at}{trade}t00Mc{price}{qty:09d}20  0J6010").encode()


def pair(**kwargs):
    return record("B", **kwargs) + b"\r\n" + record("S", **kwargs) + b"\r\n"


def inspect(raw, **kwargs):
    return inspect_mth_sample(raw, format_id=kwargs.get("format_id", FORMAT_ID))


def test_paired_sides_count_shares_once_and_do_not_qualify_tape():
    result = inspect(pair() + pair(trade="00000002", at="143000000000", qty=20))
    assert result["raw_rows"] == 4 and result["paired_trades"] == 2
    odd = result["by_trade_type"]["2"]
    assert odd["shares_once"] == 36
    assert odd["value_cents_once"] == 14250 * 36
    assert odd["raw_shares_both_sides"] == 72
    assert odd["raw_side_shares"] == {"B": 36, "S": 36}
    assert result["clock_buckets_inferred"] is True
    assert odd["raw_clock_buckets"]["0910_to_1330"]["shares_once"] == 16
    assert odd["raw_clock_buckets"]["143000000000"]["shares_once"] == 20
    assert not any(result[key] for key in (
        "historical_session_complete", "execution_tape_accepted", "source_authenticated", "live_qualified"))
    assert "rows" not in result  # Cannot accidentally serve as a normalized auction tape.


@pytest.mark.parametrize("raw", [record("B"), pair() + record("B"), record("B") + b"\n" + record("B")])
def test_missing_or_duplicate_side_is_not_silently_summed(raw):
    with pytest.raises(ValueError, match="matching buy/sell pair"):
        inspect(raw)


@pytest.mark.parametrize("change", [{"qty": 17}, {"price": "0143.00"}, {"at": "091107334514"}])
def test_pair_price_quantity_and_time_must_match(change):
    with pytest.raises(ValueError, match="matching buy/sell pair"):
        inspect(record("B") + b"\n" + record("S", **change))


@pytest.mark.parametrize("raw,match", [
    (b"", "nonempty"), (pair() + b"\n", "67 printable"),
    (pair().replace(b"3105  ", b"123456"), "four-digit"),
    (pair(kind="9"), "trade type"), (pair(qty=0), "shares"),
    (pair(at="246000000000"), "MTH line"),
    (pair(price="0000000"), "price encoding"),
    (pair().replace(b"20230915", b"20230230"), "MTH line"),
    (pair().replace(b"\r\n", b"\x0b"), "67 printable"),
])
def test_malformed_or_unknown_input_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        inspect(raw)


def test_format_must_be_explicit_and_known():
    with pytest.raises(ValueError, match="Unsupported"):
        inspect(pair(), format_id="mth-future-version")


def test_regular_trading_is_not_labeled_odd_and_other_times_remain_visible():
    result = inspect(pair(kind="0", at="090014019624", qty=1000))
    assert set(result["by_trade_type"]) == {"0"}
    assert result["by_trade_type"]["0"]["raw_clock_buckets"]["other"]["shares_once"] == 1000


def test_same_serial_in_different_security_or_date_is_not_merged():
    raw = pair() + pair().replace(b"3105  ", b"2330  ") + pair().replace(b"20230915", b"20230918")
    result = inspect(raw)
    assert result["paired_trades"] == 3
    assert result["by_trade_type"]["2"]["shares_once"] == 48
    assert result["stock_ids"] == ["2330", "3105"]
    assert result["dates"] == ["20230915", "20230918"]
