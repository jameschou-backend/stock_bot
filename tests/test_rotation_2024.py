from skills.rotation_2024 import stagnant, stronger


def test_stagnation_needs_age_and_underperformance():
    assert stagnant(20, .02, -.04)
    assert not stagnant(19, .02, -.04)
    assert not stagnant(20, .03, -.04)
    assert not stagnant(20, .02, .01)
    assert not stagnant(20, float('nan'), -.04)


def test_rotation_does_not_sell_a_young_or_outperforming_winner():
    assert stronger(10, -.05, .20)
    assert not stronger(9, -.05, .20)
    assert not stronger(10, .05, .30)
    assert not stronger(10, -.01, .05)
    assert not stronger(10, -.01, float('nan'))
