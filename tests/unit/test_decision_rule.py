'''
Req 5's rule: 95 % non-inferiority, superiority at 1 − 0.05/n over a decision's n panels (98⅓ %
over D8's three, Req 24), adoption, the non-dominated set and the tie order (D11 breaks the last
tie).
'''

import numpy as np
import pytest
from pydantic import ValidationError

from naics_embedder.decision.records import ArmSpec
from naics_embedder.decision.rule import (
    NONINFERIORITY_LEVEL,
    TieUnresolvedError,
    compare,
    compare_panel,
    non_dominated,
    superiority_level,
    tie_order,
)

pytestmark = pytest.mark.unit

# D8's three panels
THREE_PANELS = superiority_level(3)

def _spec(name, components=1, dimension=16, geometry='hyperbolic'):
    return ArmSpec(
        name=name,
        components=components,
        dimension=dimension,
        geometry=geometry,
        backbone='b',
        backbone_revision='r',
        descriptions_sha256='d',
        summaries_sha256=None,
        max_length=8,
    )

def _panel(panel, low, high, margin=1.0):
    # Evenly spread replicates: every percentile is known exactly
    replicates = np.linspace(low, high, 12001)
    return compare_panel(panel, (low + high) / 2, replicates, margin, superiority=THREE_PANELS)

def test_superiority_splits_the_error_rate_evenly_over_a_decisions_panels():
    '''Req 24: 1 − 0.05/n, so D8's three panels still read 98⅓ % and two growth panels 97.5 %.'''

    assert NONINFERIORITY_LEVEL == 0.95
    assert superiority_level(3) == 1.0 - 0.05 / 3
    assert superiority_level(3) == pytest.approx(0.98333333)
    assert superiority_level(2) == 0.975
    with pytest.raises(ValueError, match='at least one panel, not 0'):
        superiority_level(0)

def test_superiority_reads_the_level_the_decision_gives():
    # Replicates spread evenly over [-0.04, 3.96]: the lower bound is 0.01 at 97.5 % and
    # -0.0067 at 98⅓ %
    replicates = np.linspace(-0.04, 3.96, 12001)

    two = compare_panel('growth_seen', 1.96, replicates, 1.0, superiority=superiority_level(2))
    three = compare_panel('growth_seen', 1.96, replicates, 1.0, superiority=THREE_PANELS)

    assert two.superiority_interval[0] == pytest.approx(0.01)
    assert two.superior
    assert three.superiority_interval[0] == pytest.approx(-0.04 + 4 / 120)
    assert not three.superior

def test_non_inferiority_reads_the_95_interval_and_superiority_the_98_one_third_interval():
    # Replicates spread evenly over [-1, 3]: the 95 % interval is (-0.9, 2.9) and the 98⅓ %
    # interval (-0.9667, 2.9667)
    within = _panel('outcome', -1.0, 3.0, margin=0.95)
    outside = _panel('outcome', -1.0, 3.0, margin=0.85)

    assert within.noninferiority_interval == pytest.approx((-0.9, 2.9))
    assert within.superiority_interval == pytest.approx((-1 + 4 / 120, 3 - 4 / 120))
    assert within.non_inferior and not within.superior
    assert not outside.non_inferior
    assert _panel('outcome', 0.1, 2.0).superior
    # Above zero at 95 % (0.05) but not at 98⅓ % (-0.0167)
    borderline = _panel('outcome', -0.05, 3.95)
    assert borderline.noninferiority_interval[0] > 0 and not borderline.superior
    # A non-finite delta must be refused before a record can ever be written (fix round 1)
    with pytest.raises(ValidationError):
        compare_panel(
            'outcome', float('nan'), np.linspace(-1.0, 3.0, 12001), 0.95, superiority=THREE_PANELS
        )

def test_adoption_needs_non_inferiority_everywhere_and_superiority_somewhere():
    superior = _panel('outcome', 0.5, 1.5)
    level = _panel('regressor_seen', -0.5, 0.5)
    inferior = _panel('regressor_heldout', -3.0, -1.5)

    assert compare('A', 'B', [superior, level, level]).adopted
    assert not compare('A', 'B', [level, level, level]).adopted
    assert not compare('A', 'B', [superior, level, inferior]).adopted

def _comparison(a, b, adopted):
    panel = _panel('outcome', 0.5, 1.5) if adopted else _panel('outcome', -0.5, 0.5)
    return compare(a, b, [panel])

def test_the_survivors_are_the_arms_no_other_arm_is_adopted_over():
    comparisons = [_comparison('A', 'B', True), _comparison('B', 'C', False)]

    assert non_dominated(['A', 'B', 'C'], comparisons) == (['A', 'C'], False)

def test_when_dominance_cycles_every_arm_survives():
    comparisons = [
        _comparison('A', 'B', True),
        _comparison('B', 'C', True),
        _comparison('C', 'A', True),
    ]

    assert non_dominated(['A', 'B', 'C'], comparisons) == (['A', 'B', 'C'], True)

def test_the_tie_order_prefers_fewer_components_then_lower_dimension_then_flat_geometry():
    specs = [
        _spec('two-stage', components=2, dimension=8, geometry='euclidean'),
        _spec('wide', dimension=32, geometry='euclidean'),
        _spec('small-hyperbolic', dimension=16, geometry='hyperbolic'),
        _spec('small-flat', dimension=16, geometry='spherical'),
    ]
    gain = {'two-stage': 9.0, 'wide': 9.0, 'small-hyperbolic': 9.0, 'small-flat': 0.0}

    assert tie_order(specs, gain) == ['small-flat', 'small-hyperbolic', 'wide', 'two-stage']

def test_the_last_tie_goes_to_the_higher_held_out_gain():
    specs = [_spec('euclidean', geometry='euclidean'), _spec('spherical', geometry='spherical')]

    assert tie_order(specs, {'euclidean': 0.01, 'spherical': 0.02}) == ['spherical', 'euclidean']
    with pytest.raises(TieUnresolvedError, match='euclidean and spherical'):
        tie_order(specs, {'euclidean': 0.02, 'spherical': 0.02})
