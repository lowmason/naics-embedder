'''
The paired two-stage bootstrap (Req 5): units shared by every arm, seeds nested and drawn per arm.
'''

import hashlib

import numpy as np
import polars as pl
import pytest

from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    stream,
    unit_draws,
)

pytestmark = pytest.mark.unit

def _frame(units, items, values):
    return pl.DataFrame({'unit': units, 'item': items, 'value': values})

def test_a_stream_is_the_first_sixteen_hex_digits_of_the_names_sha256():
    assert stream('outcome') == int(hashlib.sha256(b'outcome').hexdigest()[:16], 16)
    assert stream('outcome') != stream('regressor_seen')

def test_unit_draws_depend_on_the_panel_alone_and_seed_draws_on_the_arm_too():
    units = unit_draws('outcome', 7, 50, 20260924)

    assert units.shape == (50, 7)
    assert (units.sum(axis=1) == 7).all()
    np.testing.assert_array_equal(units, unit_draws('outcome', 7, 50, 20260924))
    assert not np.array_equal(units, unit_draws('regressor_seen', 7, 50, 20260924))
    seeds = seed_draws('outcome', 'A', 5, 50, 20260924)
    assert seeds.shape == (50, 5)
    assert (seeds.sum(axis=1) == 5).all()
    np.testing.assert_array_equal(seeds, seed_draws('outcome', 'A', 5, 50, 20260924))
    assert not np.array_equal(seeds, seed_draws('outcome', 'B', 5, 50, 20260924))

def test_a_replicate_is_the_mean_over_drawn_seeds_of_the_item_mean_over_drawn_units():
    # Unit a holds items 1 and 2, unit b item 3; two seeds
    items = PanelItems.from_values(_frame(['b', 'a', 'a'], ['3', '2', '1'], [0.0, 0.0, 0.0]))
    values = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 8.0]])
    sums = items.sums(values)
    units = np.array([[2, 0], [0, 2], [1, 1]])
    seeds = np.array([[2, 0], [1, 1], [0, 2]])

    replicates = replicate_statistics(sums, items.sizes, units, seeds)

    assert items.items == ('1', '2', '3')
    np.testing.assert_array_equal(sums, [[4.0, 5.0], [4.0, 8.0]])
    # Seed 0 on unit a; both seeds on unit b; seed 1 on both units
    np.testing.assert_allclose(replicates, [2.0, 6.5, 4.0])
    assert point_statistic(sums, items.sizes) == pytest.approx((9 / 3 + 12 / 3) / 2)

def test_pairing_cancels_what_the_arms_share_whatever_the_units_drawn():
    rng = np.random.default_rng(0)
    units = [f'u{index // 3:02d}' for index in range(60)]
    base = rng.uniform(0.0, 10.0, size=60)
    items = PanelItems.from_values(_frame(units, [str(index) for index in range(60)], base))
    a = items.sums(np.tile(base, (5, 1)) + 0.1)
    b = items.sums(np.tile(base, (5, 1)))
    draws = unit_draws('outcome', len(items.units), 200, 1)

    delta = replicate_statistics(
        a, items.sizes, draws, seed_draws('outcome', 'A', 5, 200, 1)
    ) - replicate_statistics(b, items.sizes, draws, seed_draws('outcome', 'B', 5, 200, 1))
    unpaired = replicate_statistics(
        a, items.sizes, draws, seed_draws('outcome', 'A', 5, 200, 1)
    ) - replicate_statistics(
        b, items.sizes, unit_draws('other', len(items.units), 200, 1),
        seed_draws('outcome', 'B', 5, 200, 1)
    )

    np.testing.assert_allclose(delta, 0.1)
    assert unpaired.std() > 0.1

def test_paired_arms_must_share_items_and_units():
    items = PanelItems.from_values(_frame(['a', 'b'], ['1', '2'], [0.0, 0.0]))

    with pytest.raises(ValueError, match='same items'):
        items.values(_frame(['a', 'b'], ['1', '3'], [0.0, 0.0]))
    with pytest.raises(ValueError, match='share units'):
        items.values(_frame(['a', 'a'], ['1', '2'], [0.0, 0.0]))
    with pytest.raises(ValueError, match='expected 2'):
        items.values(_frame(['a'], ['1'], [0.0]))
    with pytest.raises(ValueError, match='repeats'):
        PanelItems.from_values(_frame(['a', 'b'], ['1', '1'], [0.0, 0.0]))

def test_the_interval_is_two_sided_percentiles():
    assert percentile_interval(np.arange(101.0), 0.9) == pytest.approx((5.0, 95.0))
    lower, upper = percentile_interval(np.arange(101.0), 1 - 0.05 / 3)
    assert (lower, upper) == pytest.approx((100 * 0.05 / 6, 100 - 100 * 0.05 / 6))
