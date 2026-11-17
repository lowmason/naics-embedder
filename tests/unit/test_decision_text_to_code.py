'''
Text→code's reference choice and beat-by-δ verdict (plan 17, Task 6; spec text-to-code-ita-panel,
4.5).

The reference is the arm with the higher validation headline MRR, ``lexical-stub`` on a tie. A
candidate beats it when the lower bound of the 95 % interval on Δ = candidate − reference exceeds
+δ. Δ's replicates come from the paired two-stage bootstrap: description groups are the units, the
headline pairs the items, and each arm's seeds are nested in the shared unit draw. Every result is
scored on the fixture panel of ``tests/fixtures/ita.py``, with its log under ``tmp_path``.
'''

import dataclasses
import math
from typing import Mapping, Sequence

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.text_to_code import (
    EUCLIDEAN_D16,
    LEXICAL_STUB,
    REFERENCE_ARMS,
    ItaVerdict,
    arm_replicates,
    beat_by_margin,
    choose_reference,
    compare_to_reference,
    headline_values,
)
from naics_embedder.panels.ita import ItaPanel
from naics_embedder.panels.ita_scoring import ItaResult
from naics_embedder.panels.selection_log import SelectionLog
from tests.fixtures.ita import (
    ASSIGNED,
    ITA_CANDIDATES,
    ITA_SETTINGS,
    AssignedStubEncoder,
    ita_tables,
)

pytestmark = pytest.mark.unit

REPLICATES = 400
BOOTSTRAP_SEED = 20260924

# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------

class ReassignedStub(AssignedStubEncoder):
    '''The stub arm with some queries decoded to other candidates, so its ranks differ.'''

    def __init__(self, changes: Mapping[str, str]):
        self.assigned = {**ASSIGNED, **changes}

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self._one_hot([ITA_CANDIDATES.index(self.assigned[text]) for text in texts])

# Three candidate seeds: p03 decoded right, p07 decoded wrong, p01 decoded wrong
CANDIDATE_CHANGES = (
    {
        'Oil and gas wells': '211120'
    },
    {
        'Oil and gas wells': '211120',
        'General contractor': '621210'
    },
    {
        'We make dog food': '111110'
    },
)

def _panel(tmp_path) -> ItaPanel:
    pairs, codes = ita_tables()
    return ItaPanel(
        pairs, codes, ITA_CANDIDATES, ITA_SETTINGS, SelectionLog(tmp_path / 'selection_log.jsonl')
    )

def _score(tmp_path, encoder) -> ItaResult:
    return _panel(tmp_path).score(encoder, 'validation', 'check')

def _stub(tmp_path) -> ItaResult:
    return _score(tmp_path, AssignedStubEncoder())

def _candidate_seeds(tmp_path) -> list:
    panel = _panel(tmp_path)
    return [
        panel.score(ReassignedStub(changes), 'validation', 'check') for changes in CANDIDATE_CHANGES
    ]

def _without(result: ItaResult, pair_id: str) -> ItaResult:
    return dataclasses.replace(
        result, per_pair=result.per_pair.filter(pl.col('pair_id') != pair_id)
    )

def _compare(candidate_seeds, reference_seeds, margin=0.0, candidate='c') -> ItaVerdict:
    return compare_to_reference(
        candidate,
        candidate_seeds,
        LEXICAL_STUB,
        reference_seeds,
        margin,
        replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

# -------------------------------------------------------------------------------------------------
# Reference choice
# -------------------------------------------------------------------------------------------------

def test_the_reference_arms_are_the_stub_and_euclidean_d16():
    assert (LEXICAL_STUB, EUCLIDEAN_D16) == ('lexical-stub', 'euclidean-d16')
    assert REFERENCE_ARMS == ('lexical-stub', 'euclidean-d16')

def test_the_reference_is_the_higher_mrr_and_a_tie_goes_to_the_stub():
    assert choose_reference({'lexical-stub': 0.30, 'euclidean-d16': 0.42}) == 'euclidean-d16'
    assert choose_reference({'lexical-stub': 0.42, 'euclidean-d16': 0.42}) == 'lexical-stub'

def test_the_stub_is_the_reference_when_its_mrr_is_higher():
    assert choose_reference({'euclidean-d16': 0.30, 'lexical-stub': 0.42}) == 'lexical-stub'

@pytest.mark.parametrize(
    'mrr',
    [
        {
            'lexical-stub': 0.3
        },
        {
            'euclidean-d16': 0.3
        },
        {},
        {
            'lexical-stub': 0.3,
            'euclidean-d16': 0.4,
            'euclidean-d32': 0.5
        },
        {
            'lexical-stub': 0.3,
            'euclidean-d32': 0.4
        },
    ],
    ids=['stub-only', 'd16-only', 'none', 'a-third-arm', 'another-arm'],
)
def test_choose_reference_refuses_other_arms(mrr):
    with pytest.raises(ValueError):
        choose_reference(mrr)

@pytest.mark.parametrize('arm', REFERENCE_ARMS)
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf], ids=['nan', 'inf', '-inf'])
def test_choose_reference_refuses_a_value_that_is_not_finite(arm, value):
    mrr = {'lexical-stub': 0.3, 'euclidean-d16': 0.4, arm: value}
    with pytest.raises(ValueError):
        choose_reference(mrr)

def test_choose_reference_refuses_other_arms_or_a_nan():
    with pytest.raises(ValueError):
        choose_reference({'lexical-stub': 0.3, 'euclidean-d32': 0.4})
    with pytest.raises(ValueError):
        choose_reference({'lexical-stub': math.nan, 'euclidean-d16': 0.4})

# -------------------------------------------------------------------------------------------------
# Beat by δ
# -------------------------------------------------------------------------------------------------

def test_a_superior_candidate_that_does_not_beat_by_delta_does_not_go_forward():
    replicates = np.linspace(0.02, 0.04, 1001)
    assert beat_by_margin('c', 'lexical-stub', 0.03, replicates, 0.01).beats is True
    verdict = beat_by_margin('c', 'lexical-stub', 0.03, replicates, 0.025)
    assert verdict.interval[0] > 0 and verdict.beats is False and verdict.level == 0.95

def test_the_verdict_reads_the_two_sided_95_percent_interval():
    replicates = np.linspace(0.0, 1.0, 1001)
    verdict = beat_by_margin('c', 'euclidean-d16', 0.5, replicates, 0.02)
    assert verdict.interval == percentile_interval(replicates, 0.95)
    assert verdict.interval == pytest.approx((0.025, 0.975), abs=1e-12)
    # 0.025 > 0.02 at 95 %; a wider interval would put the lower bound below δ
    assert verdict.beats is True
    assert verdict == ItaVerdict(
        candidate='c',
        reference='euclidean-d16',
        delta=0.5,
        interval=verdict.interval,
        level=0.95,
        margin=0.02,
        beats=True,
    )

def test_a_lower_bound_equal_to_delta_does_not_beat():
    verdict = beat_by_margin('c', 'lexical-stub', 0.03, np.full(100, 0.03), 0.03)
    assert verdict.interval == (0.03, 0.03)
    assert verdict.beats is False

def test_a_lower_bound_at_or_below_zero_never_beats_a_positive_delta():
    assert beat_by_margin('c', 'lexical-stub', 0.0, np.zeros(100), 0.01).beats is False

# -------------------------------------------------------------------------------------------------
# Headline values and replicates
# -------------------------------------------------------------------------------------------------

def test_headline_values_are_the_non_copied_pairs_by_description_group(tmp_path):
    result = _stub(tmp_path)
    values = headline_values(result).sort('item')
    assert values.columns == ['unit', 'item', 'value']
    headline = result.per_pair.filter(~pl.col('copied')).sort('pair_id')
    assert values.get_column('item').to_list() == headline.get_column('pair_id').to_list()
    assert 'p05' not in values.get_column('item').to_list()
    assert values.get_column('unit').to_list() == headline.get_column('description_key').to_list()
    assert values.get_column('value').to_list() == headline.get_column('reciprocal_rank').to_list()

def test_the_point_statistic_is_the_headline_mrr(tmp_path):
    result = _stub(tmp_path)
    values = headline_values(result)
    items = PanelItems.from_values(values)
    point = point_statistic(items.sums(items.values(values)), items.sizes)
    assert point == result.headline['mrr']

def test_arm_replicates_draw_units_on_the_panel_and_seeds_on_the_arm(tmp_path):
    seeds = _candidate_seeds(tmp_path)
    items = PanelItems.from_values(headline_values(seeds[0]))
    point, replicates = arm_replicates(
        items, 'c', seeds, replicates=REPLICATES, bootstrap_seed=BOOTSTRAP_SEED
    )
    sums = items.sums(np.stack([items.values(headline_values(seed)) for seed in seeds]))
    units = unit_draws('ita_text', len(items.units), REPLICATES, BOOTSTRAP_SEED)
    seed_counts = seed_draws('ita_text', 'c', 3, REPLICATES, BOOTSTRAP_SEED)
    assert point == point_statistic(sums, items.sizes)
    np.testing.assert_array_equal(
        replicates, replicate_statistics(sums, items.sizes, units, seed_counts)
    )

# -------------------------------------------------------------------------------------------------
# Comparison with the reference
# -------------------------------------------------------------------------------------------------

def test_compare_to_reference_is_reproducible_and_paired(tmp_path):
    reference = [_stub(tmp_path)]
    candidate = _candidate_seeds(tmp_path)
    first = _compare(candidate, reference, margin=0.01)
    assert first == _compare(candidate, reference, margin=0.01)
    candidate_mrr = math.fsum(seed.headline['mrr'] for seed in candidate) / len(candidate)
    assert first.delta == pytest.approx(candidate_mrr - reference[0].headline['mrr'], abs=1e-12)
    assert (first.candidate, first.reference, first.margin, first.level) == (
        'c', LEXICAL_STUB, 0.01, 0.95
    )

def test_delta_replicates_are_the_paired_difference_of_the_arms(tmp_path):
    reference = [_stub(tmp_path)]
    candidate = _candidate_seeds(tmp_path)
    verdict = _compare(candidate, reference, margin=0.01)
    items = PanelItems.from_values(headline_values(reference[0]))
    settings = {'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED}
    candidate_point, candidate_draws = arm_replicates(items, 'c', candidate, **settings)
    reference_point, reference_draws = arm_replicates(items, LEXICAL_STUB, reference, **settings)
    expected = beat_by_margin(
        'c',
        LEXICAL_STUB,
        candidate_point - reference_point,
        candidate_draws - reference_draws,
        0.01,
    )
    assert verdict == expected

def test_an_arm_against_a_copy_of_itself_differs_by_zero_on_every_replicate(tmp_path):
    # Paired arms share each unit draw, so one-seed arms with equal scores have Δ = 0 throughout
    stub = _stub(tmp_path)
    verdict = _compare([stub], [stub], margin=0.0, candidate='copy')
    assert verdict.delta == 0.0
    assert verdict.interval == (0.0, 0.0)
    assert verdict.beats is False

def test_arms_scored_on_other_pairs_are_refused(tmp_path):
    reference = [_stub(tmp_path)]
    candidate = _candidate_seeds(tmp_path)
    candidate[1] = _without(candidate[1], 'p03')
    with pytest.raises(ValueError):
        _compare(candidate, reference)

def test_a_later_reference_seed_on_other_pairs_is_refused(tmp_path):
    stub = _stub(tmp_path)
    with pytest.raises(ValueError):
        _compare([stub], [stub, _without(stub, 'p03')])

def test_the_first_candidate_seed_on_other_pairs_is_refused(tmp_path):
    stub = _stub(tmp_path)
    with pytest.raises(ValueError):
        _compare([_without(stub, 'p03')], [stub])

@pytest.mark.parametrize('side', ['candidate', 'reference'])
def test_an_arm_with_no_seed_is_refused(tmp_path, side):
    stub = _stub(tmp_path)
    seeds = {'candidate': [stub], 'reference': [stub]}
    seeds[side] = []
    with pytest.raises(ValueError, match='has no seed'):
        _compare(seeds['candidate'], seeds['reference'])
