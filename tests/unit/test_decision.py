'''
Margins and decisions on synthetic arms with known effects (roadmap Stage 4 Exit).

Effects are in units of ``SIGMA``, and every arm's across-seed standard deviation is ``SD`` =
``SIGMA`` × √2.5, so a margin multiple of 2 gives δ ≈ 3.2 ``SIGMA``. Paired Δ is the effect plus
the difference of two means of five seed offsets drawn with replacement; its 95 % and 98⅓ %
intervals reach about 1.8 and 2.1 ``SIGMA`` either side. An effect of 5 is superior, 0 is not,
and −5 is not non-inferior.
'''

import json
import re
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from naics_embedder.decision.decide import check_arm, check_seed_distance, decide, fix_margins
from naics_embedder.decision.records import (
    ArmRecord,
    DecisionRecord,
    MarginRecord,
    read_record,
    write_record,
)
from naics_embedder.decision.rule import SUPERIORITY_LEVEL
from naics_embedder.decision.scores import PANELS
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.decoding import METRIC_NAMES
from tests.fixtures.decision import (
    REVISION,
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

def _json_edited(record, edit):
    '''A record's JSON form after ``edit`` changes it.'''

    data = json.loads(record.model_dump_json())
    edit(data)
    return data

def _edited(arm, edit):
    '''The arm record after ``edit`` changes its JSON form.'''

    return ArmRecord.model_validate(_json_edited(arm, edit))

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

def test_each_margin_is_the_multiple_times_the_references_across_seed_sd(margins):
    assert [entry.panel for entry in margins.margins] == list(PANELS)
    for entry in margins.margins:
        assert len(entry.per_seed) == 5
        assert entry.sd == pytest.approx(SD)
        assert entry.margin == pytest.approx(2 * SD)

def _without_heldout(data):
    data['margins'] = [entry for entry in data['margins'] if entry['panel'] != 'regressor_heldout']

def test_a_panel_without_a_margin_is_named(margins):
    partial = MarginRecord.model_validate(_json_edited(margins, _without_heldout))

    assert partial.margin('outcome') == margins.margin('outcome')
    with pytest.raises(ValueError, match='no δ for regressor_heldout'):
        partial.margin('regressor_heldout')

@pytest.mark.parametrize(
    'edit',
    [
        _without_heldout,
        lambda data: data['margins'][0].update(statistic='top1'),
        lambda data: data['margins'].append(data['margins'][0]),
    ],
    ids=['a panel without one', 'another statistic', 'a panel twice'],
)
def test_a_decision_needs_one_margin_per_panel_for_its_statistic(
    store, tmp_path, reference, margins, edit
):
    edited = MarginRecord.model_validate(_json_edited(margins, edit))
    candidate = synthetic_arm(store, tmp_path, spec('candidate', dimension=32), {})

    with pytest.raises(ValueError, match='not one δ per panel'):
        _decide([candidate, reference], edited, store)

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
        (
            lambda data: _first_read(data)['detail'].update(distance='cosine'),
            "another \\['distance'\\]",
        ),
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

@pytest.mark.parametrize(
    'edit, key',
    [
        (lambda read: read['detail'].update(text_only='other'), 'text_only'),
        (lambda read: read['detail'].update(arm='other'), 'arm'),
        (lambda read: read['detail'].update(dimension=16), 'dimension'),
        (lambda read: read.update(fingerprint='other'), 'fingerprint'),
    ],
    ids=['text_only', 'arm', 'dimension', 'fingerprint'],
)
def test_a_regressor_read_must_name_the_runs_tables_and_its_panel(
    store, tmp_path, reference, margins, edit, key
):
    arm = _edited(
        synthetic_arm(store, tmp_path, spec('edited', dimension=32), {}),
        lambda data: edit(data['runs'][0]['log_records'][1]),
    )

    with pytest.raises(ValueError, match=f"another \\['{key}'\\]"):
        _decide([arm, reference], margins, store)

@pytest.mark.parametrize(
    'geometry, distance',
    [('euclidean', 'euclidean'), ('spherical', 'cosine'), ('hyperbolic', 'lorentz')],
)
def test_a_seed_decodes_by_the_distance_of_its_arms_geometry(geometry, distance):
    '''Req 12: each arm decodes by its own distance, and no other.'''

    arm_spec = spec('arm', geometry=geometry)

    check_seed_distance(arm_spec, 3, distance)
    for other in sorted({'euclidean', 'cosine', 'lorentz'} - {distance}):
        message = (
            f"^arm seed 3: the encoder decodes by '{other}', but a {geometry} arm decodes by "
            f"'{distance}' \\(Req 12\\)$"
        )
        with pytest.raises(ValueError, match=message):
            check_seed_distance(arm_spec, 3, other)

def test_a_decision_compares_at_least_two_arms(store, reference, margins):
    with pytest.raises(ValueError, match='at least two arms'):
        _decide([reference], margins, store)

def test_two_arms_may_not_share_a_name(store, tmp_path, reference, margins):
    # One name at two dimensions, so that without the refusal the tie order would pick one
    twins = [
        synthetic_arm(store, tmp_path, spec('twin', dimension=dimension), {})
        for dimension in (16, 32)
    ]

    with pytest.raises(ValueError, match='arm names repeat'):
        _decide(twins, margins, store)

@pytest.mark.parametrize(
    'edit',
    [
        lambda data: data['runs'][5].update(seed=4),
        lambda data: data['runs'][5].update(run_id=data['runs'][4]['run_id']),
    ],
    ids=['seed', 'run id'],
)
def test_a_seed_or_run_id_may_not_repeat_within_an_arm(store, tmp_path, reference, margins, edit):
    # Six runs, so five distinct seeds remain when one repeats
    arm = _edited(
        synthetic_arm(
            store, tmp_path, spec('repeated', dimension=32), {}, offsets=SEED_OFFSETS + (0.0, )
        ),
        edit,
    )

    with pytest.raises(ValueError, match='a seed or run id repeats'):
        _decide([arm, reference], margins, store)

def test_the_margin_reference_must_have_read_the_arms_panels(store, tmp_path, reference, margins):
    other = MarginRecord.model_validate(
        _json_edited(
            margins, lambda data: data['reference']['panels'].update(outcome_data='other')
        )
    )
    candidate = synthetic_arm(store, tmp_path, spec('candidate', dimension=32), {})

    with pytest.raises(ValueError, match="the margin reference reference .*\\['outcome_data'\\]"):
        _decide([candidate, reference], other, store)

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

def test_the_text_only_table_must_read_the_arms_summaries(store, tmp_path, reference, margins):
    # Built before the arm's summaries: on truncated text
    stale = write_text_only(tmp_path / 'stale', summaries=None)
    arm = synthetic_arm(store, tmp_path, spec('stale'), {}, text_only_table=stale)

    with pytest.raises(ValueError, match="'e{64}'.*D9"):
        _decide([arm, reference], margins, store)

def test_the_text_only_check_reads_the_stored_provenance(store, tmp_path, reference, margins):
    stale = write_text_only(tmp_path / 'stale', revision='an-older-revision')
    # The record's copy claims the arm's revision; the stored provenance names the older one
    arm = _edited(
        synthetic_arm(store, tmp_path, spec('stale', dimension=32), {}, text_only_table=stale),
        lambda data: data['text_only'].update(revision=REVISION),
    )

    with pytest.raises(
        ValueError, match="fields \\['revision'\\] differ from its stored provenance"
    ):
        _decide([arm, reference], margins, store)

def test_a_changed_artifact_is_refused(store, tmp_path, reference, margins):
    arm = synthetic_arm(store, tmp_path, spec('changed'), {})
    store.resolve(arm.runs[2].scores).write_bytes(b'not the scores')

    with pytest.raises(ValueError, match='changed since it was stored'):
        _decide([arm, reference], margins, store)

# -------------------------------------------------------------------------------------------------
# Monitor records (Req 4; spec 4.4)
# -------------------------------------------------------------------------------------------------

# Every trained seed's monitor MRR by epoch: epochs 1 and 2 tie for the highest
MONITOR_MRRS = (0.2, 0.5, 0.5, 0.4)

@pytest.fixture
def trained(store, tmp_path):
    return synthetic_arm(
        store, tmp_path, spec('trained', dimension=32), {}, monitor_mrrs=MONITOR_MRRS
    )

def _run(data, seed=3):
    return data['runs'][seed]

def _monitor_record(data, seed=3, epoch=2):
    return _run(data, seed)['monitor_records'][epoch]

def _monitor_read(data, seed=3, epoch=2):
    return _monitor_record(data, seed, epoch)['read']

def test_a_trained_run_keeps_its_earliest_epoch_with_the_highest_monitor_mrr(
    store, reference, margins, trained
):
    '''Spec 4.4: the kept checkpoint is the earliest epoch with the highest MRR, ties included.'''

    for run in trained.runs:
        assert run.training_run == f'trained-training-{run.seed}'
        assert [record['mrr'] for record in run.monitor_records] == list(MONITOR_MRRS)
        assert run.checkpoint_epoch == 1

    check_arm(trained, store, min_seeds=5)
    # Neither arm is adopted, and the lower dimension stands
    assert _decide([trained, reference], margins, store).chosen == 'reference'

def test_a_monitor_read_may_name_a_table_other_than_the_exported_one(store, trained):
    '''
    A read names its epoch's code cache, which equals the exported table only when both were
    encoded on the CPU (spec 4.4's agreement test); the campaign trains on CUDA.
    '''

    for run in trained.runs:
        selected = run.monitor_records[run.checkpoint_epoch]['read']['detail']
        assert selected['table'] != run.table.matrix_fingerprint

    check_arm(trained, store, min_seeds=5)

def _shuffle_the_records(data):
    '''Seed 3's records at epochs 2, 0, 1 and 3, in that order.'''

    records = _run(data)['monitor_records']
    records[:] = [records[epoch] for epoch in (2, 0, 1, 3)]

def test_the_earliest_best_epoch_is_read_from_the_epochs_not_the_records_order(store, trained):
    '''
    The first record is epoch 2, tied for the highest MRR with epoch 1, so neither the first best
    record's epoch (2) nor its position (0) is the earliest best epoch.
    '''

    arm = _edited(trained, _shuffle_the_records)

    shuffled = arm.runs[3]
    epochs = [record['read']['detail']['epoch'] for record in shuffled.monitor_records]
    assert (epochs, shuffled.checkpoint_epoch) == ([2, 0, 1, 3], 1)
    check_arm(arm, store, min_seeds=5)

def _repeat_an_epoch(data):
    _run(data)['monitor_records'].append(_monitor_record(data, epoch=1))

# What follows 'trained seed 3: ' in each refusal
NO_TRAINING_RUN = 'on a run that names no training run'
MALFORMED = 'a monitor record is malformed: '
NOT_A_MONITOR_READ = "; the monitor reads the outcome panel's validation split \\(Req 4\\)"

@pytest.mark.parametrize(
    'edit, message',
    [
        (
            lambda data: _run(data).update(training_run=None),
            f'monitor records and a checkpoint epoch {NO_TRAINING_RUN}',
        ),
        (
            lambda data: _run(data).update(training_run=None, checkpoint_epoch=None),
            f'monitor records {NO_TRAINING_RUN}',
        ),
        (
            lambda data: _run(data).update(training_run=None, monitor_records=[]),
            f'a checkpoint epoch {NO_TRAINING_RUN}',
        ),
        (
            lambda data: _run(data).update(monitor_records=[]),
            'training run trained-training-3 has no monitor records',
        ),
        (
            lambda data: _monitor_record(data).update(extra=1),
            f"{MALFORMED}it must be an object with exactly 'mrr' and 'read'",
        ),
        (
            lambda data: _monitor_record(data).update(mrr='0.5'),
            f'{MALFORMED}its mrr is not a finite number',
        ),
        (
            lambda data: _monitor_record(data).update(mrr=True),
            f'{MALFORMED}its mrr is not a finite number',
        ),
        (
            lambda data: _monitor_record(data).update(mrr=float('inf')),
            f'{MALFORMED}its mrr is not a finite number',
        ),
        (
            lambda data: _monitor_record(data).update(mrr=float('nan')),
            f'{MALFORMED}its mrr is not a finite number',
        ),
        (
            lambda data: _monitor_read(data)['detail'].pop('epoch'),
            f'{MALFORMED}its read names no non-negative integer epoch',
        ),
        (
            lambda data: _monitor_read(data)['detail'].update(epoch=-1),
            f'{MALFORMED}its read names no non-negative integer epoch',
        ),
        (
            # On epoch 1's record: True == 1, so only the type check tells them apart
            lambda data: _monitor_read(data, epoch=1)['detail'].update(epoch=True),
            f'{MALFORMED}its read names no non-negative integer epoch',
        ),
        (
            lambda data: _monitor_read(data).update(panel='regressor_seen'),
            "a monitor record logs 'read' on the regressor_seen panel's validation split"
            f'{NOT_A_MONITOR_READ}',
        ),
        (
            lambda data: _monitor_read(data).update(split='test'),
            f"a monitor record logs 'read' on the outcome panel's test split{NOT_A_MONITOR_READ}",
        ),
        (
            lambda data: _monitor_read(data).update(event='open'),
            "a monitor record logs 'open' on the outcome panel's validation split"
            f'{NOT_A_MONITOR_READ}',
        ),
        (
            lambda data: _monitor_read(data)['detail'].update(training_run='another-run'),
            "a monitor read names another \\['training_run'\\]",
        ),
        (
            lambda data: _monitor_read(data)['detail'].update(seed=4),
            "a monitor read names another \\['seed'\\]",
        ),
        (
            lambda data: _monitor_read(data).update(fingerprint='other-roles'),
            "a monitor read names another \\['fingerprint'\\]",
        ),
        (
            lambda data: _monitor_read(data)['detail'].update(distance='cosine'),
            "a monitor read names another \\['distance'\\]",
        ),
        (_repeat_an_epoch, 'the monitor records repeat the epochs \\[1\\]'),
    ],
    ids=[
        'records-and-an-epoch-without-a-training-run',
        'records-without-a-training-run',
        'an-epoch-without-a-training-run',
        'a-training-run-without-records',
        'another-key',
        'a-text-mrr',
        'a-boolean-mrr',
        'an-infinite-mrr',
        'a-nan-mrr',
        'no-epoch',
        'a-negative-epoch',
        'a-boolean-epoch',
        'another-panel',
        'a-test-split',
        'an-open-event',
        'another-training-run',
        'another-seed',
        'another-fingerprint',
        'another-distance',
        'a-repeated-epoch',
    ],
)
def test_a_runs_monitor_records_must_be_its_own_outcome_validation_reads(
    store, trained, edit, message
):
    '''Spec §5: monitor records that are not the run's outcome validation reads are refused.'''

    arm = _edited(trained, edit)

    with pytest.raises(ValueError, match=f'^trained seed 3: {message}'):
        check_arm(arm, store, min_seeds=5)

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_a_flat_arms_reads_are_checked_against_its_own_distance(store, tmp_path, geometry):
    '''Req 12: a flat arm's outcome and monitor reads decode by its distance, not the Lorentz.'''

    flat = synthetic_arm(
        store, tmp_path, spec('flat', geometry=geometry), {}, monitor_mrrs=MONITOR_MRRS
    )

    check_arm(flat, store, min_seeds=5)
    for edit in (
        lambda data: _first_read(data)['detail'].update(distance='lorentz'),
        lambda data: _monitor_read(data)['detail'].update(distance='lorentz'),
    ):
        with pytest.raises(ValueError, match="another \\['distance'\\]"):
            check_arm(_edited(flat, edit), store, min_seeds=5)

@pytest.mark.parametrize('epoch', [None, 0, 2, 3], ids=['none', 'earlier', 'a-later-tie', 'later'])
def test_a_checkpoint_from_any_other_epoch_than_the_earliest_best_is_refused(store, trained, epoch):
    '''Spec §5: the run's earliest epoch with the highest MRR must be its checkpoint's, ties too.'''

    arm = _edited(trained, lambda data: _run(data).update(checkpoint_epoch=epoch))

    with pytest.raises(
        ValueError,
        match=(
            f'^trained seed 3: the earliest epoch with the highest monitor MRR, 0.5, is 1, but '
            f'the checkpoint is from epoch {epoch}'
        ),
    ):
        check_arm(arm, store, min_seeds=5)

@pytest.mark.parametrize(
    'field, value, error',
    [('training_run', '', 'string_too_short'), ('checkpoint_epoch', -1, 'greater_than_equal')],
    ids=['a-blank-training-run', 'a-negative-checkpoint-epoch'],
)
def test_a_run_with_a_blank_training_run_or_a_negative_epoch_does_not_load(
    trained, field, value, error
):
    with pytest.raises(ValidationError, match=f'runs\\.3\\.{field}\\n.*{error}'):
        _edited(trained, lambda data: _run(data).update({field: value}))

def test_a_monitor_read_before_the_margins_were_fixed_is_refused(
    store, tmp_path, reference, margins
):
    early = (margins.fixed_at - timedelta(minutes=1)).isoformat()
    arm = _edited(
        synthetic_arm(store, tmp_path, spec('early', dimension=32), {}, monitor_mrrs=MONITOR_MRRS),
        lambda data: _monitor_read(data).update(time=early),
    )
    # Its decision reads all came after the margins
    reads = [datetime.fromisoformat(log['time']) for run in arm.runs for log in run.log_records]
    assert min(reads) >= margins.fixed_at

    with pytest.raises(
        ValueError,
        match=f'^early seed 3 read at {re.escape(early)}, before the margins were fixed',
    ):
        _decide([arm, reference], margins, store)

def test_a_decision_record_carries_every_trained_runs_monitor_reads(store, tmp_path):
    '''
    Spec 4.4: every validation read that selected anything reaches the decision record. The
    reference's monitor reads predate its margins, and its runs are exempt from the check.
    '''

    reference = synthetic_arm(store, tmp_path, spec('reference'), {}, monitor_mrrs=MONITOR_MRRS)
    margins = fix_margins(reference, 2.0, 'trained margins', store, min_seeds=5)
    candidate = synthetic_arm(
        store, tmp_path, spec('candidate', dimension=32), {}, monitor_mrrs=MONITOR_MRRS
    )
    monitored = [
        datetime.fromisoformat(record['read']['time']) for run in reference.runs
        for record in run.monitor_records
    ]
    assert max(monitored) < margins.fixed_at
    path = write_record(_decide([candidate, reference], margins, store), tmp_path / 'decision.json')

    record = read_record(path, DecisionRecord)

    carried = [*record.arms, record.margins.reference]
    for written, arm in zip(carried, [candidate, reference, reference]):
        assert written.runs == arm.runs
        for run in written.runs:
            assert len(run.monitor_records) == len(MONITOR_MRRS)

MONITOR_FIELDS = ('training_run', 'checkpoint_epoch', 'monitor_records')

def test_a_record_written_before_runs_carried_monitor_reads_still_loads(
    store, tmp_path, reference, margins
):
    candidate = synthetic_arm(store, tmp_path, spec('candidate', dimension=32), {})
    path = write_record(_decide([candidate, reference], margins, store), tmp_path / 'decision.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    for arm in [*data['arms'], data['margins']['reference']]:
        for run in arm['runs']:
            for field in MONITOR_FIELDS:
                del run[field]
    old = tmp_path / 'written-before.json'
    old.write_text(json.dumps(data), encoding='utf-8')

    record = read_record(old, DecisionRecord)

    for arm in [*record.arms, record.margins.reference]:
        for run in arm.runs:
            assert (run.training_run, run.checkpoint_epoch, run.monitor_records) == (None, None, [])
        check_arm(arm, store, min_seeds=5)
