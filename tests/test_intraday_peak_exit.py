import json
import subprocess
import sys
from copy import deepcopy

import pytest

from skills.intraday_peak_exit import IntradayPeakExit, replay


def position(venue='board', qty=2000):
    return dict(stock_id='6197', qty=qty, venue=venue, peak=1000.,
                asof='2026-07-09T13:30:00+08:00', basis='verified_raw', source='known holding')


def session(day='2026-07-13'):
    return dict(date=day, known_at=day+'T08:59:00+08:00', lower=700., upper=1100.,
                prior_adv_shares=1000000, basis='verified_raw', source='official limits + prior ADV')


def group(price, seconds=1, shares=100000, venue='board', day='2026-07-13'):
    return dict(at=f'{day}T09:00:{seconds:02d}+08:00', stock_id='6197', venue=venue,
                basis='verified_raw', prints=[dict(price=price, shares=shares)])


def engine(venue='board', qty=2000):
    e = IntradayPeakExit(**position(venue, qty)); e.start_session(**session()); return e


def test_trigger_is_immediate_but_trigger_print_cannot_fill():
    e = engine()
    first = e.observe(**group(850))
    assert first['intent']['limit_price'] == 700
    assert first['fill'] is None and e.remaining == 2000
    e.observe(**group(900, 2))  # Rebound does not cancel exit.
    assert e.remaining == 1000 and e.fills[0]['price'] == 900
    e.observe(**group(890, 3))
    r = e.report()
    assert r['status'] == 'closed_proxy' and len(r['orders']) == 1
    assert all(f['at'] > r['trigger']['at'] for f in r['fills'])
    assert r['net_sell_proceeds'] < 1790000


def test_gap_and_locked_limit_wait_for_later_volume_and_carry_overnight():
    e = engine(); e.observe(**group(700)); e.observe(**group(700, 2, 999000))
    assert e.remaining == 2000
    e.start_session(**session('2026-07-14'))
    e.observe(**group(800, 1, day='2026-07-14'))
    assert e.remaining == 1000 and len(e.orders) == 2
    assert e.orders[1]['trigger_at'] == e.orders[0]['trigger_at']


def test_odd_volume_cannot_be_borrowed_from_board_market():
    e = engine('odd', 24)
    with pytest.raises(ValueError): e.observe(**group(850))
    e.observe(**group(850, venue='odd', shares=100))
    e.observe(**group(849, 2, shares=1000, venue='odd'))
    assert e.remaining == 14


def test_tied_timestamp_ambiguity_rejected_without_mutation():
    e = engine(); before = deepcopy(e.__dict__)
    g = group(900); g['prints'].append(dict(price=1100, shares=1000))
    with pytest.raises(ValueError, match='Ambiguous'): e.observe(**g)
    assert e.__dict__ == before


def test_new_peak_needs_no_profit_arm_and_invalid_basis_stops_processing():
    e = engine(); e.observe(**group(1050)); e.observe(**group(892.5, 2))
    assert e.trigger['threshold'] == 892.5
    g = group(800, 3); g['basis'] = 'unresolved_ex_dividend'
    with pytest.raises(ValueError): e.observe(**g)
    assert e.remaining == 2000


def test_adv_cap_prevents_accumulating_unlimited_fills():
    e = engine(); e.session['adv'] = 100000
    e.observe(**group(850))
    for i in range(2, 5): e.observe(**group(849, i, shares=900000))
    assert e.remaining == 1000


@pytest.mark.parametrize('change', [dict(shares=-1), dict(shares=1), dict(price=float('nan')), dict(price=1200)])
def test_invalid_prints_fail_closed(change):
    e = engine(); g = group(850); g['prints'][0].update(change)
    with pytest.raises(ValueError): e.observe(**g)
    assert e.trigger is None


def test_cli_and_replay_are_deterministic_and_preserve_existing_output(tmp_path):
    payload = dict(position=position(), events=[dict(kind='session', **session()),
        dict(kind='prints', **group(850)), dict(kind='prints', **group(849, 2))])
    assert replay(payload) == replay(json.loads(json.dumps(payload)))
    inp, out = tmp_path/'input.json', tmp_path/'output.json'
    inp.write_text(json.dumps(payload))
    cmd = [sys.executable, 'scripts/replay_intraday_stop.py', '--input', str(inp), '--output', str(out)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    result = json.loads(out.read_text())
    assert result['remaining_qty'] == 1000 and not result['broker_submitted']
    assert subprocess.run(cmd, capture_output=True).returncode != 0


def test_no_reordered_groups_or_same_timestamp_duplicates():
    e = engine(); e.observe(**group(900))
    with pytest.raises(ValueError): e.observe(**group(850))
    assert e.trigger is None
