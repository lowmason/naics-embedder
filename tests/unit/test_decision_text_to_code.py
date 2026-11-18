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
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import numpy as np
import polars as pl
import pytest
import torch
from pydantic import ValidationError

from naics_embedder.decision.decide import read_time
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
    ItaMarginRecord,
    ItaVerdict,
    arm_replicates,
    beat_by_margin,
    check_ita_margin_first,
    choose_reference,
    compare_to_reference,
    fix_ita_margin,
    headline_values,
)
from naics_embedder.panels.ita import ItaPanel
from naics_embedder.panels.ita_scoring import ItaResult
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
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

# -------------------------------------------------------------------------------------------------
# Margin
# -------------------------------------------------------------------------------------------------

PURPOSE = 'fix the text→code margin'

def _margin_arms() -> dict:
    # The stub as lexical-stub; euclidean-d16's two seeds decode p03 right, so its MRR is higher
    return {
        LEXICAL_STUB: [(None, AssignedStubEncoder())],
        EUCLIDEAN_D16: [
            (1, ReassignedStub(CANDIDATE_CHANGES[0])),
            (2, ReassignedStub(CANDIDATE_CHANGES[0])),
        ],
    }

def _read(panel: ItaPanel, role=None, part='validation', purpose='a stage 5 read') -> dict:
    detail = {'arm': 'c', 'seed': 1, 'run': 'stage5/c'}
    if role is not None:
        detail['role'] = role
    if part == 'sealed':
        panel.open_sealed('the decision')
    _, record = panel.score_logged(ReassignedStub(CANDIDATE_CHANGES[1]), part, purpose, detail)
    return record

def _fix(panel: ItaPanel) -> ItaMarginRecord:
    return fix_ita_margin(panel, _margin_arms(), 0.05, 'ita-margins', purpose=PURPOSE)

def test_the_margin_is_r_times_the_chosen_references_validation_mrr(tmp_path):
    panel = _panel(tmp_path)
    arms = _margin_arms()

    record = fix_ita_margin(panel, arms, 0.05, 'ita-margins', purpose='fix the text→code margin')

    assert record.reference == choose_reference(record.reference_mrr)
    assert record.margin == pytest.approx(0.05 * record.reference_mrr[record.reference])
    assert record.split_fingerprint == panel.fingerprint
    assert {r['detail']['role'] for r in record.log_records} == {'reference'}
    assert len(record.log_records) == 3

def test_the_margin_records_each_arms_mean_mrr_and_its_reads(tmp_path):
    panel = _panel(tmp_path)
    stub = panel.score(AssignedStubEncoder(), 'validation', 'check').headline['mrr']
    better = panel.score(ReassignedStub(CANDIDATE_CHANGES[0]), 'validation', 'check').headline['mrr']
    assert better > stub

    record = _fix(panel)

    assert record.kind == 'ita_margin'
    assert record.name == 'ita-margins' and record.r == 0.05
    assert record.reference_mrr == {LEXICAL_STUB: stub, EUCLIDEAN_D16: better}
    assert record.reference == EUCLIDEAN_D16
    assert record.margin == 0.05 * better
    assert record.data_fingerprint == panel.data_fingerprint('validation')
    assert record.run_id.startswith('ita-margin/') and len(record.run_id) == len('ita-margin/') + 32
    assert record.fixed_at.utcoffset() is not None
    reads = [r for r in panel.log.records() if r['detail'].get('run') == record.run_id]
    assert record.log_records == reads
    assert [(r['detail']['arm'], r['detail'].get('seed')) for r in reads] == [
        (LEXICAL_STUB, None),
        (EUCLIDEAN_D16, 1),
        (EUCLIDEAN_D16, 2),
    ]
    assert 'seed' not in reads[0]['detail']
    for read in reads:
        assert (read['event'], read['panel'], read['split'], read['purpose']) == (
            'read',
            'ita_text',
            'validation',
            PURPOSE,
        )
        assert read['detail']['role'] == 'reference'

def test_each_margin_reads_under_a_fresh_run_id(tmp_path):
    panel = _panel(tmp_path)
    assert _fix(panel).run_id != _fix(panel).run_id

def test_the_stub_is_the_reference_when_d16_reads_worse(tmp_path):
    panel = _panel(tmp_path)
    arms = {
        LEXICAL_STUB: [(None, ReassignedStub(CANDIDATE_CHANGES[0]))],
        EUCLIDEAN_D16: [(1, AssignedStubEncoder()), (2, ReassignedStub(CANDIDATE_CHANGES[1]))],
    }
    record = fix_ita_margin(panel, arms, 0.05, 'ita-margins', purpose=PURPOSE)
    seeds = [
        panel.score(AssignedStubEncoder(), 'validation', 'check').headline['mrr'],
        panel.score(ReassignedStub(CANDIDATE_CHANGES[1]), 'validation', 'check').headline['mrr'],
    ]
    assert record.reference_mrr[EUCLIDEAN_D16] == math.fsum(seeds) / 2
    assert record.reference == LEXICAL_STUB
    assert record.margin == 0.05 * record.reference_mrr[LEXICAL_STUB]

def test_a_tie_between_the_reference_arms_goes_to_the_stub(tmp_path):
    panel = _panel(tmp_path)
    arms = {
        LEXICAL_STUB: [(None, AssignedStubEncoder())],
        EUCLIDEAN_D16: [(1, AssignedStubEncoder())],
    }
    record = fix_ita_margin(panel, arms, 0.05, 'ita-margins', purpose=PURPOSE)
    assert record.reference_mrr[LEXICAL_STUB] == record.reference_mrr[EUCLIDEAN_D16]
    assert record.reference == LEXICAL_STUB

def test_the_margin_scales_with_r(tmp_path):
    panel = _panel(tmp_path)
    record = fix_ita_margin(panel, _margin_arms(), 0.2, 'ita-margins', purpose=PURPOSE)
    assert record.r == 0.2
    assert record.margin == 0.2 * record.reference_mrr[EUCLIDEAN_D16]

@pytest.mark.parametrize(
    'r, arms',
    [
        (0.0, None),
        (-0.05, None),
        (math.nan, None),
        (math.inf, None),
        (0.05, {
            LEXICAL_STUB: [(None, AssignedStubEncoder())]
        }),
        (
            0.05,
            {
                LEXICAL_STUB: [(None, AssignedStubEncoder())],
                'euclidean-d32': [(1, AssignedStubEncoder())],
            },
        ),
        (
            0.05,
            {
                LEXICAL_STUB: [(None, AssignedStubEncoder())],
                EUCLIDEAN_D16: [(1, AssignedStubEncoder())],
                'euclidean-d32': [(1, AssignedStubEncoder())],
            },
        ),
        (0.05, {
            LEXICAL_STUB: [(None, AssignedStubEncoder())],
            EUCLIDEAN_D16: []
        }),
    ],
    ids=['zero', 'negative', 'nan', 'inf', 'one-arm', 'another-arm', 'a-third-arm', 'no-seed'],
)
def test_fix_ita_margin_refuses_a_bad_r_or_other_arms_before_reading(tmp_path, r, arms):
    panel = _panel(tmp_path)
    with pytest.raises(ValueError):
        fix_ita_margin(
            panel, _margin_arms() if arms is None else arms, r, 'ita-margins', purpose=PURPOSE
        )
    assert not panel.log.path.exists()

def test_a_margin_after_a_candidate_read_is_refused(tmp_path):
    panel = _panel(tmp_path)
    read = _read(panel, role='candidate', purpose='an early candidate read')
    margin = _fix(panel)
    with pytest.raises(ValueError, match='an early candidate read') as refused:
        check_ita_margin_first(margin, panel.log)
    assert read['time'] in str(refused.value)

def test_a_margin_after_a_read_with_no_role_is_refused(tmp_path):
    panel = _panel(tmp_path)
    read = _read(panel, role=None, purpose='a monitor read')
    assert 'role' not in read['detail']
    margin = _fix(panel)
    with pytest.raises(ValueError, match='a monitor read') as refused:
        check_ita_margin_first(margin, panel.log)
    assert read['time'] in str(refused.value)

def test_earlier_reference_reads_and_later_candidate_reads_pass(tmp_path):
    panel = _panel(tmp_path)
    _read(panel, role='reference', purpose='stage 4 gap evidence')
    margin = _fix(panel)
    _read(panel, role='candidate')
    _read(panel, role=None)
    check_ita_margin_first(margin, panel.log)

def test_earlier_sealed_reads_and_other_panels_reads_do_not_count(tmp_path):
    panel = _panel(tmp_path)
    _read(panel, role='candidate', part='sealed')
    panel.log.append(
        SelectionEvent.READ,
        panel='outcome',
        split='validation',
        purpose='a stage 7 read',
        fingerprint='f' * 64,
        n_queries=3,
        detail={'run': 'stage7/x'},
    )
    margin = _fix(panel)
    check_ita_margin_first(margin, panel.log)

def test_a_candidate_read_before_fixed_at_is_refused_when_fixed_at_is_moved_later(tmp_path):
    panel = _panel(tmp_path)
    margin = _fix(panel)
    read = _read(panel, role='candidate', purpose='a later candidate read')
    later = margin.model_copy(update={'fixed_at': read_time(read, 'read') + timedelta(seconds=1)})
    with pytest.raises(ValueError, match='a later candidate read'):
        check_ita_margin_first(later, panel.log)

def test_a_naive_fixed_at_or_a_read_after_fixed_at_is_refused(tmp_path):
    panel = _panel(tmp_path)
    margin = _fix(panel)
    naive = margin.model_copy(update={'fixed_at': margin.fixed_at.replace(tzinfo=None)})
    with pytest.raises(ValueError, match='timezone'):
        check_ita_margin_first(naive, panel.log)
    first = read_time(margin.log_records[0], 'read')
    early = margin.model_copy(update={'fixed_at': first - timedelta(seconds=1)})
    with pytest.raises(ValueError, match='before'):
        check_ita_margin_first(early, panel.log)
    check_ita_margin_first(margin, panel.log)

def test_the_record_round_trips_through_json(tmp_path):
    margin = _fix(_panel(tmp_path))
    assert ItaMarginRecord.model_validate_json(margin.model_dump_json()) == margin

def test_the_record_refuses_extra_keys_a_bad_r_and_a_nan():
    fields = {
        'name': 'ita-margins',
        'r': 0.05,
        'run_id': 'ita-margin/x',
        'reference_mrr': {
            LEXICAL_STUB: 0.5,
            EUCLIDEAN_D16: 0.6
        },
        'reference': EUCLIDEAN_D16,
        'margin': 0.03,
        'split_fingerprint': 'a' * 64,
        'data_fingerprint': 'b' * 64,
        'log_records': [],
        'fixed_at': datetime(2026, 11, 17, tzinfo=timezone.utc),
    }
    record = ItaMarginRecord(**fields)
    with pytest.raises(ValidationError):
        record.margin = 0.1
    for bad in ({'extra': 1}, {'r': 0.0}, {'margin': math.nan}, {'kind': 'relative_margins'}):
        with pytest.raises(ValidationError):
            ItaMarginRecord(**{**fields, **bad})
