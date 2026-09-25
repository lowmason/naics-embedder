'''
Margins and decisions on synthetic arms with known effects (roadmap Stage 4 Exit).

Effects are in units of ``SIGMA``, and every arm's across-seed standard deviation is ``SD`` =
``SIGMA`` × √2.5, so a margin multiple of 2 gives δ ≈ 3.2 ``SIGMA``. Paired Δ is the effect plus
the difference of two means of five seed offsets drawn with replacement; its 95 % and 98⅓ %
intervals reach about 1.8 and 2.1 ``SIGMA`` either side. An effect of 5 is superior, 0 is not,
and −5 is not non-inferior.
'''

import json

import pytest

from naics_embedder.decision.decide import decide, fix_margins
from naics_embedder.decision.records import ArmRecord, DecisionRecord, read_record, write_record
from naics_embedder.decision.rule import SUPERIORITY_LEVEL
from naics_embedder.decision.scores import PANELS
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.decoding import METRIC_NAMES
from tests.fixtures.decision import (
    SD,
    SEED_OFFSETS,
    SIGMA,
    spec,
    synthetic_arm,
    write_text_only,
)

pytestmark = pytest.mark.unit

REPLICATES = 4000
BOOTSTRAP_SEED = 20260924

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

@pytest.fixture
def reference(store, tmp_path):
    return synthetic_arm(store, tmp_path, spec('reference'), {})

@pytest.fixture
def margins(reference, store):
    return fix_margins(reference, 2.0, 'fixture margins', store, min_seeds=5)

def _decide(arms, margins, store):
    return decide(
        'fixture decision',
        'which arm?',
        arms,
        margins,
        store,
        replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        min_seeds=5,
    )

def _comparison(record, a, b):
    return next(item for item in record.comparisons if (item.a, item.b) == (a, b))

def _panel(comparison, panel):
    return next(item for item in comparison.panels if item.panel == panel)

def _edited(arm, edit):
    '''The arm record after ``edit`` changes its JSON form.'''

    data = json.loads(arm.model_dump_json())
    edit(data)
    return ArmRecord.model_validate(data)

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

def test_each_margin_is_the_multiple_times_the_references_across_seed_sd(margins):
    assert [entry.panel for entry in margins.margins] == list(PANELS)
    for entry in margins.margins:
        assert len(entry.per_seed) == 5
        assert entry.sd == pytest.approx(SD)
        assert entry.margin == pytest.approx(2 * SD)

def test_a_margin_needs_a_positive_multiple_and_a_reference_that_varies(store, tmp_path, reference):
    with pytest.raises(ValueError, match='positive'):
        fix_margins(reference, 0.0, 'none', store, min_seeds=5)
    with pytest.raises(ValueError, match='at least 5 seeds'):
        fix_margins(reference, 2.0, 'few', store, min_seeds=4)
    flat = synthetic_arm(store, tmp_path, spec('flat'), {}, offsets=(0.0, ) * 5)
    with pytest.raises(ValueError, match='does not vary'):
        fix_margins(flat, 2.0, 'flat', store, min_seeds=5)

# -------------------------------------------------------------------------------------------------
# The rule on known effects
# -------------------------------------------------------------------------------------------------

def test_an_arm_superior_on_one_panel_and_level_on_the_others_is_adopted(
    store, tmp_path, reference, margins
):
    better = synthetic_arm(store, tmp_path, spec('better', dimension=32), {'outcome': 5})

    record = _decide([better, reference], margins, store)

    adopted = _comparison(record, 'better', 'reference')
    assert adopted.adopted
    assert _panel(adopted, 'outcome').delta == pytest.approx(5 * SIGMA)
    assert _panel(adopted, 'outcome').superior
    for panel in ('regressor_seen', 'regressor_heldout'):
        assert _panel(adopted, panel).non_inferior
        assert not _panel(adopted, panel).superior
    assert not _comparison(record, 'reference', 'better').adopted
    # Adopted over the simpler arm: the tie order never runs between them
    assert (record.non_dominated, record.cycle, record.chosen) == (['better'], False, 'better')

def test_superiority_on_one_panel_does_not_rescue_an_inferior_one(
    store, tmp_path, reference, margins
):
    mixed = synthetic_arm(
        store, tmp_path, spec('mixed', dimension=32), {
            'outcome': -5,
            'regressor_seen': 5
        }
    )

    record = _decide([mixed, reference], margins, store)

    rejected = _comparison(record, 'mixed', 'reference')
    assert _panel(rejected, 'regressor_seen').superior
    assert not _panel(rejected, 'outcome').non_inferior
    assert not rejected.adopted
    assert not _comparison(record, 'reference', 'mixed').adopted
    assert record.non_dominated == ['mixed', 'reference']
    assert record.chosen == 'reference'

@pytest.mark.parametrize('dimension, chosen', [(32, 'reference'), (8, 'level')])
def test_without_superiority_the_simpler_arm_stands(
    store, tmp_path, reference, margins, dimension, chosen
):
    level = synthetic_arm(store, tmp_path, spec('level', dimension=dimension), {})

    record = _decide([level, reference], margins, store)

    assert not any(item.adopted for item in record.comparisons)
    assert record.chosen == chosen

def test_a_dominance_cycle_leaves_every_arm_to_the_tie_order(store, tmp_path, reference):
    # δ = 5 SD ≈ 7.9 SIGMA: each arm is non-inferior where it is 5 SIGMA worse
    wide = fix_margins(reference, 5.0, 'wide margins', store, min_seeds=5)
    arms = [
        synthetic_arm(store, tmp_path, spec('outcome-arm', dimension=32), {'outcome': 5}),
        synthetic_arm(store, tmp_path, spec('seen-arm', dimension=16), {'regressor_seen': 5}),
        synthetic_arm(store, tmp_path, spec('heldout-arm', dimension=24), {'regressor_heldout': 5}),
    ]

    record = _decide(arms, wide, store)

    assert all(item.adopted for item in record.comparisons)
    assert record.cycle
    assert record.non_dominated == ['outcome-arm', 'seen-arm', 'heldout-arm']
    assert record.tie_order == ['seen-arm', 'heldout-arm', 'outcome-arm']
    assert record.chosen == 'seen-arm'

def test_the_held_out_gain_over_ancestors_breaks_the_last_tie(store, tmp_path, reference, margins):
    # Half a SIGMA lower held-out error: too small to be superior, enough to break the tie
    arms = [
        synthetic_arm(store, tmp_path, spec('spherical', geometry='spherical'), {}),
        synthetic_arm(
            store, tmp_path, spec('euclidean', geometry='euclidean'), {'regressor_heldout': 0.5}
        ),
    ]

    record = _decide(arms, margins, store)

    assert not any(item.adopted for item in record.comparisons)
    assert record.heldout_gain['euclidean'] == pytest.approx(0.03 + 0.5 * SIGMA)
    assert record.heldout_gain['spherical'] == pytest.approx(0.03)
    assert record.tie_order == ['euclidean', 'spherical']

# -------------------------------------------------------------------------------------------------
# The record (Verification "Decision records")
# -------------------------------------------------------------------------------------------------

def test_the_record_carries_every_field_verification_lists(store, tmp_path, reference, margins):
    better = synthetic_arm(store, tmp_path, spec('better'), {'outcome': 5})
    path = write_record(_decide([better, reference], margins, store), tmp_path / 'decision.json')

    record = read_record(path, DecisionRecord)

    # Its arms, at least 5 seeds each
    assert [arm.spec.name for arm in record.arms] == ['better', 'reference']
    assert all(len(arm.runs) >= 5 for arm in record.arms)
    # The δ per panel, fixed before the runs
    assert [entry.panel for entry in record.margins.margins] == list(PANELS)
    reads = [log['time'] for run in record.arms[0].runs for log in run.log_records]
    assert min(reads) >= record.margins.fixed_at.isoformat()
    # The 95 % non-inferiority and 98⅓ % superiority intervals, on every panel
    assert record.settings.noninferiority_level == 0.95
    assert record.settings.superiority_level == SUPERIORITY_LEVEL
    for comparison in record.comparisons:
        assert [panel.panel for panel in comparison.panels] == list(PANELS)
        for panel in comparison.panels:
            low, high = panel.noninferiority_interval
            wide_low, wide_high = panel.superiority_interval
            assert wide_low <= low <= high <= wide_high
    # The non-dominated set
    assert record.non_dominated == ['better']
    # The selection-log records of the runs and every artifact reference
    for arm in record.arms:
        store.resolve(arm.text_only.table)
        store.resolve(arm.text_only.provenance)
        for run in arm.runs:
            assert sorted(log['panel'] for log in run.log_records) == sorted(PANELS)
            for reference_ in (
                run.checkpoint, run.table, run.scores, run.decoding, run.predictions
            ):
                store.resolve(reference_)
    # What D10 reports besides the rule
    report = record.reports[0]
    assert set(report.outcome_metrics) == set(METRIC_NAMES)
    assert set(report.comparator_mse['regressor_seen']) >= {
        'covariates', 'covariates+embedding', 'covariates+one_hot'
    }
    assert report.gain['regressor_seen'].point == pytest.approx(0.02)
    assert report.gain['regressor_heldout'].point == pytest.approx(0.03)
    assert set(report.heldout_by_feature_year) == {'2022', '2023'}
    assert report.heldout_by_feature_year['2023']['gain'] == pytest.approx(0.03)

# -------------------------------------------------------------------------------------------------
# Guards
# -------------------------------------------------------------------------------------------------

def test_every_arm_needs_five_seeds(store, tmp_path, reference, margins):
    short = synthetic_arm(store, tmp_path, spec('short'), {}, offsets=SEED_OFFSETS[:4])

    with pytest.raises(ValueError, match='fewer than 5'):
        _decide([short, reference], margins, store)

def test_a_run_that_read_before_the_margins_were_fixed_is_refused(store, tmp_path, reference):
    early = synthetic_arm(store, tmp_path, spec('early'), {})
    margins = fix_margins(reference, 2.0, 'late margins', store, min_seeds=5)

    with pytest.raises(ValueError, match='before the margins were fixed'):
        _decide([early, reference], margins, store)

def _first_read(data):
    return data['runs'][0]['log_records'][0]

@pytest.mark.parametrize(
    'edit, message',
    [
        (lambda data: _first_read(data).update(split='test'), 'validation splits only'),
        (lambda data: _first_read(data).update(event='open'), 'validation splits only'),
        (lambda data: _first_read(data)['detail'].update(run='other'), 'another run'),
        (lambda data: _first_read(data)['detail'].update(table='other'), "another \\['table'\\]"),
        (lambda data: _first_read(data).update(fingerprint='other'), 'another'),
        (lambda data: data['runs'][0]['log_records'].pop(), 'not each of'),
    ],
)
def test_a_runs_log_records_must_be_its_own_validation_reads(
    store, tmp_path, reference, margins, edit, message
):
    arm = _edited(synthetic_arm(store, tmp_path, spec('edited'), {}), edit)

    with pytest.raises(ValueError, match=message):
        _decide([arm, reference], margins, store)

def test_a_regressor_read_must_name_the_runs_tables(store, tmp_path, reference, margins):
    arm = _edited(
        synthetic_arm(store, tmp_path, spec('edited'), {}),
        lambda data: data['runs'][0]['log_records'][1]['detail'].update(text_only='other'),
    )

    with pytest.raises(ValueError, match="another \\['text_only'\\]"):
        _decide([arm, reference], margins, store)

@pytest.mark.parametrize(
    'edit, message',
    [
        (lambda panels: panels['fit_settings'].update(folds=3), "\\['fit_settings'\\]"),
        (lambda panels: panels.update(outcome_data='other'), "\\['outcome_data'\\]"),
        (lambda panels: panels.update(regressor_data='other'), "\\['regressor_data'\\]"),
    ],
)
def test_paired_arms_must_read_the_same_panels(store, tmp_path, reference, margins, edit, message):
    arm = _edited(
        synthetic_arm(store, tmp_path, spec('other-panels'), {}),
        lambda data: edit(data['panels']),
    )

    with pytest.raises(ValueError, match=f'other panels or fit settings .*differing in {message}'):
        _decide([arm, reference], margins, store)

def test_the_text_only_table_must_come_from_the_arms_backbone_and_text(
    store, tmp_path, reference, margins
):
    stale = write_text_only(tmp_path / 'stale', revision='an-older-revision')
    arm = synthetic_arm(store, tmp_path, spec('stale'), {}, text_only_table=stale)

    with pytest.raises(ValueError, match='D9'):
        _decide([arm, reference], margins, store)

def test_a_changed_artifact_is_refused(store, tmp_path, reference, margins):
    arm = synthetic_arm(store, tmp_path, spec('changed'), {})
    store.resolve(arm.runs[2].scores).write_bytes(b'not the scores')

    with pytest.raises(ValueError, match='changed since it was stored'):
        _decide([arm, reference], margins, store)
