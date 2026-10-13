'''
The economic decision on the fixture growth panel (Req 24; roadmap Stage 2 Exit).

The margin read scores the reference, ``covariates+ancestors``, beside the other comparators that
read no candidate. Every candidate's read fits the same rows, folds and penalties, so the
reference's errors pair exactly with each candidate's.
'''

import hashlib
import json
import platform
import re
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from naics_embedder.decision.economic import (
    GROWTH_PANELS,
    REFERENCE_COMPARATOR,
    check_economic_arm,
    check_reference_pairing,
    decide_economic,
    fix_relative_margins,
    read_deterministic_arm,
)
from naics_embedder.decision.records import (
    EconomicArmRecord,
    EconomicDecisionRecord,
    RelativeMarginRecord,
    read_record,
    write_record,
)
from naics_embedder.decision.resampling import percentile_interval, stream
from naics_embedder.decision.scores import panel_statistic, regressor_scores, statistic_values
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import (
    Regime,
    RegressorPanel,
    comparators,
    reference_comparators,
)
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from tests.fixtures.decision import write_text_only
from tests.fixtures.economic import candidate_spec, write_candidate
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, SETTINGS, coordinate_table

pytestmark = pytest.mark.unit

PURPOSE = 'fixture economic decision'
# U13
R = 0.05

@pytest.fixture
def log(tmp_path):
    return SelectionLog(tmp_path / 'selection_log.jsonl')

@pytest.fixture
def growth_panel(regressor_growth_rows, log):
    return RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

@pytest.fixture
def lock(tmp_path):
    path = tmp_path / 'uv.lock'
    path.write_text('a fixture lock\n')
    return path

@pytest.fixture
def margins(growth_panel, store, lock):
    return fix_relative_margins(
        growth_panel, R, 'fixture margins', store, purpose=PURPOSE, lock=lock
    )

# -------------------------------------------------------------------------------------------------
# The margin read and the relative margins (Verification "Margin first, reference only")
# -------------------------------------------------------------------------------------------------

def _reference_mse(predictions, panel):
    '''The reference's MSE on a panel, each row's squared error averaged over its repeats first.'''

    # yapf: disable
    rows = (
        predictions
        .filter((pl.col('panel') == panel) & (pl.col('comparator') == REFERENCE_COMPARATOR))
        .group_by('code', 'feature_year')
        .agg(error=((pl.col('prediction') - pl.col('outcome'))**2).mean())
    )
    # yapf: enable
    return rows.get_column('error').mean()

def test_each_margin_is_r_times_the_references_validation_mse(margins, store):
    predictions = store.read_frame(margins.read.predictions)

    assert (margins.r, margins.reference) == (R, 'covariates+ancestors')
    assert [entry.panel for entry in margins.margins] == list(GROWTH_PANELS)
    for entry in margins.margins:
        assert entry.statistic == REFERENCE_COMPARATOR
        assert entry.reference_mse == pytest.approx(
            _reference_mse(predictions, entry.panel), rel=1e-12
        )
        assert entry.margin == R * entry.reference_mse
        assert margins.margin(entry.panel) == entry.margin

def test_the_margin_read_logs_only_comparators_that_read_no_candidate(margins, store, log):
    records = log.records()
    predictions = store.read_frame(margins.read.predictions)

    assert margins.read.log_records == records
    assert [(record['event'], record['panel'], record['split']) for record in records] == [
        ('read', 'growth_seen', 'validation'),
        ('read', 'growth_heldout', 'validation'),
    ]
    for record, regime in zip(records, Regime):
        names = list(reference_comparators(regime, 6))
        assert record['detail'] == {'level': 6, 'comparators': names, 'run': margins.read.run_id}
        assert datetime.fromisoformat(record['time']) <= margins.fixed_at
        read = predictions.filter(pl.col('panel') == record['panel'])
        assert read.get_column('comparator').unique(maintain_order=True).to_list() == names

def test_the_margins_name_the_read_its_panel_and_where_it_ran(
    margins, growth_panel, store, lock, tmp_path
):
    read = margins.read

    assert read.panels.regressor == growth_panel.fingerprint
    assert read.panels.regressor_data == growth_panel.data_fingerprint(6)
    assert read.panels.fit_settings['repeats'] == SETTINGS.repeats
    assert read.environment.platform == platform.platform()
    assert read.environment.python == platform.python_version()
    assert read.environment.lock_sha256 == hashlib.sha256(lock.read_bytes()).hexdigest()
    store.resolve(read.scores)
    path = write_record(margins, tmp_path / 'margins.json')
    assert read_record(path, RelativeMarginRecord) == margins

def test_the_margins_read_growth_only(regressor_rows, log, store, lock):
    level = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)

    with pytest.raises(
        ValueError, match='^the margins: the economic decision reads growth, not level'
    ):
        fix_relative_margins(level, R, 'level margins', store, purpose=PURPOSE, lock=lock)
    assert log.records() == []

@pytest.mark.parametrize('r', [0.0, -0.05])
def test_r_must_be_positive(growth_panel, log, store, lock, r):
    with pytest.raises(ValueError, match='r must be positive'):
        fix_relative_margins(growth_panel, r, 'none', store, purpose=PURPOSE, lock=lock)
    assert log.records() == []

def test_the_margins_name_a_lock_that_exists(growth_panel, log, store, tmp_path):
    with pytest.raises(FileNotFoundError, match='no uv.lock'):
        fix_relative_margins(
            growth_panel, R, 'unlocked', store, purpose=PURPOSE, lock=tmp_path / 'missing'
        )
    assert log.records() == []

# -------------------------------------------------------------------------------------------------
# A deterministic candidate (Verification "Deterministic arms")
# -------------------------------------------------------------------------------------------------

TREE = candidate_spec('tree')

@pytest.fixture
def text_only(tmp_path):
    return write_text_only(tmp_path / 'text', codes=CODEBOOK)

@pytest.fixture
def candidate(tmp_path):
    return write_candidate(tmp_path / 'candidates', 'tree', coordinate_table(CODEBOOK))

def _read(table, panel, text_only, margins, store, lock, arm_spec=TREE):
    return read_deterministic_arm(
        arm_spec,
        table,
        regressor_panel=panel,
        text_only_table=text_only,
        margins=margins,
        store=store,
        purpose=PURPOSE,
        lock=lock,
    )

@pytest.fixture
def arm(candidate, growth_panel, text_only, margins, store, lock):
    return _read(candidate, growth_panel, text_only, margins, store, lock)

def test_a_deterministic_candidate_is_one_replicate_read_once_on_each_growth_panel(
    arm, margins, store, log
):
    '''Req 24: a single replicate, with no seed, checkpoint or monitor reads.'''

    (run, ) = arm.runs
    reads = log.records()[len(margins.read.log_records):]

    assert run.seed is None
    assert run.log_records == reads
    assert [(read['panel'], read['split']) for read in reads] == [
        ('growth_seen', 'validation'),
        ('growth_heldout', 'validation'),
    ]
    for read, regime in zip(reads, Regime):
        assert read['detail'] == {
            'level': 6,
            'comparators': list(comparators(regime, 6)),
            'arm': run.table.matrix_fingerprint,
            'text_only': arm.text_only.table.matrix_fingerprint,
            'dimension': 3,
            'run': run.run_id,
            'arm_name': 'tree',
        }
    scores = store.read_frame(run.scores)
    assert run.statistics == {
        panel: panel_statistic(scores, panel, 'covariates+embedding')
        for panel in GROWTH_PANELS
    }
    assert (arm.panels, arm.environment) == (margins.read.panels, margins.read.environment)

def test_the_record_keeps_the_tables_provenance_verbatim(arm, candidate, store, tmp_path):
    '''Req 23: the record shows what the candidate was built from, as its builder wrote it.'''

    (run, ) = arm.runs
    written = provenance_path(candidate)

    assert run.provenance.content == json.loads(written.read_text())
    assert store.resolve(run.provenance.file).read_bytes() == written.read_bytes()
    path = write_record(arm, tmp_path / 'arm.json')
    assert read_record(path, EconomicArmRecord) == arm

def test_a_single_replicate_record_validates_and_pairs_with_the_reference(arm, margins, store):
    '''Verification "Deterministic arms": the reference predicts as in the margin read.'''

    check_economic_arm(arm, store)
    check_reference_pairing(margins, arm, store)

    def reference(predictions):
        return store.read_frame(predictions).filter(pl.col('comparator') == REFERENCE_COMPARATOR)

    assert reference(arm.runs[0].predictions).equals(reference(margins.read.predictions))

def test_a_candidate_is_read_on_the_growth_panel_the_margins_read(
    candidate, regressor_rows, regressor_growth_rows, log, text_only, margins, store, lock
):
    level = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)
    refit = RegressorPanel(
        regressor_growth_rows,
        HELDOUT_GROUPS,
        log,
        replace(SETTINGS, alphas=(0.1, 1.0)),
        outcome=RegressorOutcome.GROWTH,
    )

    with pytest.raises(ValueError, match='^tree: the economic decision reads growth, not level'):
        _read(candidate, level, text_only, margins, store, lock)
    with pytest.raises(
        ValueError,
        match="^tree read other panels or fit settings than the margins 'fixture margins', "
        "differing in \\['fit_settings'\\]",
    ):
        _read(candidate, refit, text_only, margins, store, lock)
    assert log.records() == margins.read.log_records

@pytest.mark.parametrize(
    'edit, error, message',
    [
        (lambda table: provenance_path(table).unlink(), FileNotFoundError, 'has no provenance'),
        (
            lambda table: coordinate_table(CODEBOOK, seed=8).write_parquet(table),
            ValueError,
            'describes another file than the table',
        ),
    ],
    ids=['missing', 'another-file'],
)
def test_a_candidate_is_read_only_with_provenance_that_describes_it(
    candidate, growth_panel, text_only, margins, store, lock, log, edit, error, message
):
    edit(candidate)

    with pytest.raises(error, match=message):
        _read(candidate, growth_panel, text_only, margins, store, lock)
    assert log.records() == margins.read.log_records

def test_a_candidate_reads_its_own_text_only_table_at_its_dimension(
    candidate, growth_panel, text_only, margins, store, lock, log, tmp_path
):
    other = write_text_only(tmp_path / 'other', revision='other', codes=CODEBOOK)
    wide = candidate_spec('tree', dimension=4)

    with pytest.raises(ValueError, match='^tree: the text-only table was built from .*D9'):
        _read(candidate, growth_panel, other, margins, store, lock)
    with pytest.raises(ValueError, match="^tree: the table has dimension 3, not the spec's 4"):
        _read(candidate, growth_panel, text_only, margins, store, lock, arm_spec=wide)
    assert log.records() == margins.read.log_records

@pytest.mark.parametrize(
    'field, value',
    [
        ('checkpoint', {
            'path': 'objects/a',
            'sha256': 'a' * 64,
            'bytes': 1
        }),
        ('training_run', 'tree-training-0'),
        ('monitor_records', []),
    ],
)
def test_an_imported_run_with_a_checkpoint_or_monitor_reads_does_not_load(
    arm, tmp_path, field, value
):
    '''Req 24: a deterministic candidate is recorded without checkpoint or monitor reads.'''

    data = json.loads(arm.model_dump_json())
    data['runs'][0][field] = value
    path = tmp_path / 'imported.json'
    path.write_text(json.dumps(data))

    with pytest.raises(ValidationError, match=f'runs\\.0\\.{field}\\n  Extra inputs'):
        read_record(path, EconomicArmRecord)

def test_an_economic_arm_is_one_replicate_with_no_seed(arm, store):
    (run, ) = arm.runs
    seeded = arm.model_copy(update={'runs': [run.model_copy(update={'seed': 0})]})
    twice = arm.model_copy(update={'runs': [run, run]})

    for record in (seeded, twice):
        with pytest.raises(ValueError, match='^tree: an economic arm is one deterministic'):
            check_economic_arm(record, store)

def test_a_run_whose_reads_name_another_table_is_refused(arm, store):
    (run, ) = arm.runs
    reads = json.loads(json.dumps(run.log_records))
    reads[0]['detail']['arm'] = 'another table'
    edited = arm.model_copy(update={'runs': [run.model_copy(update={'log_records': reads})]})

    with pytest.raises(ValueError, match="^tree: a growth_seen read names another \\['arm'\\]"):
        check_economic_arm(edited, store)

def test_a_provenance_edited_after_it_was_stored_is_refused(arm, store):
    (run, ) = arm.runs
    content = {**run.provenance.content, 'inputs': {'qcew': 'national slices'}}
    provenance = run.provenance.model_copy(update={'content': content})
    edited = arm.model_copy(update={'runs': [run.model_copy(update={'provenance': provenance})]})

    with pytest.raises(ValueError, match="^tree: the record's provenance differs from its stored"):
        check_economic_arm(edited, store)

def test_reference_predictions_that_differ_from_the_margin_reads_do_not_pair(arm, margins, store):
    '''Req 24: only the predictions decide; the environments name where each read ran.'''

    (run, ) = arm.runs
    moved = arm.environment.model_copy(update={'platform': 'another platform'})
    reference = pl.col('comparator') == REFERENCE_COMPARATOR
    predictions = store.read_frame(run.predictions).with_columns(
        prediction=pl.when(reference).then(pl.col('prediction') + 1e-12).otherwise('prediction')
    )
    refit = run.model_copy(
        update={'predictions': store.put_frame(predictions, 'predictions.parquet')}
    )

    check_reference_pairing(margins, arm.model_copy(update={'environment': moved}), store)
    with pytest.raises(
        ValueError,
        match='^tree: its covariates\\+ancestors predictions differ .* another platform'
    ):
        check_reference_pairing(
            margins, arm.model_copy(update={
                'environment': moved,
                'runs': [refit]
            }), store
        )

# -------------------------------------------------------------------------------------------------
# The economic decision (Verification "Economic decision record", "Margin first, reference only")
# -------------------------------------------------------------------------------------------------

REPLICATES = 2000
BOOTSTRAP_SEED = 20260924
GROWTH_SEEN, GROWTH_HELDOUT = GROWTH_PANELS

def _both(factor):
    return dict.fromkeys(GROWTH_PANELS, factor)

def _scaled(arm, factors, store):
    '''
    The arm, rescored with each ``covariates+embedding`` prediction moved to the outcome plus
    ``factors[panel]`` times the reference's error on its row and repeat. Every row's squared
    error is then factor² times the reference's, so Δ on each item is (1 − factor²) times the
    reference's error, whatever the draw.
    '''

    (run, ) = arm.runs
    predictions = store.read_frame(run.predictions)
    keys = ['panel', 'repeat', 'code', 'feature_year']
    reference = predictions.filter(pl.col('comparator') == REFERENCE_COMPARATOR
                                   ).select(*keys, reference='prediction')
    factor = pl.when(pl.col('panel') == GROWTH_SEEN).then(factors[GROWTH_SEEN]).otherwise(
        factors[GROWTH_HELDOUT]
    )
    error = pl.col('reference') - pl.col('outcome')
    moved = predictions.join(reference, on=keys).with_columns(
        prediction=pl.when(pl.col('comparator') == 'covariates+embedding').then(
            pl.col('outcome') + factor * error
        ).otherwise('prediction')
    ).select(predictions.columns)
    scores = regressor_scores(moved, SETTINGS.repeats, GROWTH_PANELS)
    scaled = run.model_copy(
        update={
            'predictions': store.put_frame(moved, 'predictions.parquet'),
            'scores': store.put_frame(scores, 'scores.parquet'),
            'statistics': {
                panel: panel_statistic(scores, panel, 'covariates+embedding')
                for panel in GROWTH_PANELS
            },
        }
    )
    return arm.model_copy(update={'runs': [scaled]})

@pytest.fixture
def candidates(growth_panel, text_only, margins, store, lock, tmp_path):
    '''Reads a new candidate, after the margins, and scales its errors (``_scaled``).'''

    def read(name, factors, **overrides):
        table = write_candidate(tmp_path / 'candidates', name, coordinate_table(CODEBOOK))
        arm_spec = candidate_spec(name, **overrides)
        arm = _read(table, growth_panel, text_only, margins, store, lock, arm_spec)
        return _scaled(arm, factors, store)

    return read

def _decide(arms, margins, store):
    return decide_economic(
        'fixture economic decision',
        'which candidate, if any, is the economic artifact?',
        arms,
        margins,
        store,
        replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

def test_a_candidate_adopted_over_the_reference_is_chosen(candidates, margins, store):
    record = _decide([candidates('better', _both(0.5))], margins, store)

    (comparison, ) = record.comparisons
    assert (comparison.a, comparison.b, comparison.adopted) == (
        'better', REFERENCE_COMPARATOR, True
    )
    assert [(panel.non_inferior, panel.superior)
            for panel in comparison.panels] == [(True, True)] * 2
    assert (record.eligible, record.non_dominated, record.cycle) == (['better'], ['better'], False)
    assert (record.tie_order, record.chosen) == (['better'], 'better')

@pytest.mark.parametrize(
    'factor, verdicts', [(1.0, (True, False)), (1.5, (False, False))], ids=['equal', 'worse']
)
def test_with_no_candidate_adopted_the_reference_stands(
    candidates, margins, store, factor, verdicts
):
    '''Req 24: with none eligible the reference stands, and there is no economic artifact.'''

    record = _decide([candidates('candidate', _both(factor))], margins, store)

    (comparison, ) = record.comparisons
    assert [(panel.non_inferior, panel.superior) for panel in comparison.panels] == [verdicts] * 2
    assert (record.eligible, record.non_dominated, record.cycle) == ([], [], False)
    assert (record.tie_order, record.chosen) == ([], None)

def test_only_candidates_adopted_over_the_reference_are_eligible(candidates, margins, store):
    '''Req 24: the rule picks among the eligible, compared with each other only.'''

    arms = [
        candidates('better', _both(0.5)),
        candidates('best', _both(0.3)),
        candidates('equal', _both(1.0)),
    ]

    record = _decide(arms, margins, store)

    assert [(c.a, c.b, c.adopted) for c in record.comparisons] == [
        ('better', REFERENCE_COMPARATOR, True),
        ('best', REFERENCE_COMPARATOR, True),
        ('equal', REFERENCE_COMPARATOR, False),
        ('better', 'best', False),
        ('best', 'better', True),
    ]
    assert record.eligible == ['better', 'best']
    assert (record.non_dominated, record.tie_order, record.chosen) == (['best'], ['best'], 'best')

def test_the_tie_order_prefers_fewer_components_among_the_eligible(candidates, margins, store):
    stacked = candidates('stacked', _both(0.5), components=2)
    simple = candidates('simple', _both(0.5))

    record = _decide([stacked, simple], margins, store)

    # Equal errors: neither is adopted over the other
    assert record.non_dominated == ['stacked', 'simple']
    assert (record.tie_order, record.chosen) == (['simple', 'stacked'], 'simple')

def test_the_higher_heldout_gain_over_the_reference_breaks_a_remaining_tie(
    candidates, margins, store
):
    '''D11, for growth: each candidate is superior on one panel and inferior on the other.'''

    seen = candidates('seen', {GROWTH_SEEN: 0.5, GROWTH_HELDOUT: 0.7})
    heldout = candidates('heldout', {GROWTH_SEEN: 0.7, GROWTH_HELDOUT: 0.5})

    record = _decide([seen, heldout], margins, store)

    assert record.eligible == record.non_dominated == ['seen', 'heldout']
    assert record.heldout_gain['heldout'] > record.heldout_gain['seen'] > 0
    assert (record.tie_order, record.chosen) == (['heldout', 'seen'], 'heldout')

def test_the_economic_decision_record_shows_what_verification_lists(
    candidates, margins, store, tmp_path
):
    record = _decide([candidates('better', _both(0.5))], margins, store)
    path = write_record(record, tmp_path / 'decision.json')

    assert read_record(path, EconomicDecisionRecord) == record
    # Two panels, with their 95 % non-inferiority and 97.5 % superiority intervals
    assert (record.settings.noninferiority_level, record.settings.superiority_level) == (
        0.95, 0.975
    )
    assert [panel.panel for panel in record.comparisons[0].panels] == list(GROWTH_PANELS)
    # δ = 0.05 × the reference's MSE per panel, and the reference
    scores = store.read_frame(margins.read.scores)
    for panel, entry in zip(GROWTH_PANELS, record.margins.margins):
        assert (entry.panel, entry.margin) == (panel, 0.05 * entry.reference_mse)
        assert entry.reference_mse == pytest.approx(
            panel_statistic(scores, panel, REFERENCE_COMPARATOR), rel=1e-12
        )
    assert record.reference == record.margins.reference == 'covariates+ancestors'
    # Each candidate's provenance, which names no employment-data input
    assert record.arms[0].runs[0].provenance.content['inputs'] == {
        'codebook': 'the fixture codebook'
    }
    # The chosen arm
    assert record.chosen == 'better'
    # One-hot is still reported, where the seen regime scores it
    (report, ) = record.reports
    assert report.statistics == pytest.approx(record.arms[0].runs[0].statistics, rel=1e-12)
    assert 'covariates+one_hot' in report.comparator_mse[GROWTH_SEEN]
    assert 'covariates+one_hot' not in report.comparator_mse[GROWTH_HELDOUT]

def test_a_single_replicate_is_resampled_by_its_units_alone(candidates, margins, store):
    '''
    Req 24: a deterministic candidate is one replicate, so each bootstrap replicate's Δ is the
    reference's MSE minus the candidate's on a draw of four-digit groups alone, shared by both.
    '''

    arm = candidates('better', _both(0.5))

    record = _decide([arm], margins, store)

    for panel, comparison in zip(GROWTH_PANELS, record.comparisons[0].panels):
        reference = statistic_values(
            store.read_frame(margins.read.scores), panel, REFERENCE_COMPARATOR
        )
        candidate = statistic_values(
            store.read_frame(arm.runs[0].scores), panel, 'covariates+embedding'
        )
        # yapf: disable
        by_unit = (
            reference
            .join(candidate, on=['unit', 'item'], suffix='_candidate')
            .group_by('unit')
            .agg(delta=(pl.col('value') - pl.col('value_candidate')).sum(), n=pl.len())
            .sort('unit')
        )
        # yapf: enable
        units = by_unit.height
        counts = np.random.default_rng([BOOTSTRAP_SEED, stream(panel)]).multinomial(
            units, np.full(units, 1.0 / units), size=REPLICATES
        )
        replicates = counts @ by_unit['delta'].to_numpy() / (counts @ by_unit['n'].to_numpy())
        assert comparison.noninferiority_interval == pytest.approx(
            percentile_interval(replicates, 0.95), rel=1e-12
        )
        assert comparison.superiority_interval == pytest.approx(
            percentile_interval(replicates, 0.975), rel=1e-12
        )

def test_a_candidate_read_before_the_margins_were_fixed_is_refused(
    candidates, growth_panel, store, lock
):
    '''Verification "Margin first": margins fixed after a candidate's read cannot judge it.'''

    arm = candidates('tree', _both(0.5))
    later = fix_relative_margins(
        growth_panel, R, 'later margins', store, purpose=PURPOSE, lock=lock
    )
    first = min(datetime.fromisoformat(read['time']) for read in arm.runs[0].log_records)

    with pytest.raises(
        ValueError,
        match=f'^tree read at {re.escape(first.isoformat())}, before the margins were fixed',
    ):
        _decide([arm], later, store)

def test_margins_fixed_before_their_own_read_are_refused(candidates, margins, store):
    '''Verification "Margin first": δ is recorded from a reference-only read that precedes it.'''

    arm = candidates('tree', _both(0.5))
    early = margins.model_copy(update={'fixed_at': margins.fixed_at - timedelta(hours=1)})

    with pytest.raises(
        ValueError,
        match="^the margins 'fixture margins' were fixed at .*, before their growth_seen read",
    ):
        _decide([arm], early, store)

def _margin_read(margins, **update):
    return margins.model_copy(update={'read': margins.read.model_copy(update=update)})

def test_margins_whose_read_was_not_reference_only_are_refused(candidates, margins, store):
    '''Verification "Margin first, reference only": decide refuses a margin read of a candidate.'''

    arm = candidates('tree', _both(0.5))
    (run, ) = arm.runs
    reads = json.loads(json.dumps(margins.read.log_records))
    reads[0]['detail']['arm'] = run.table.matrix_fingerprint
    candidate_read = _margin_read(margins, predictions=run.predictions, scores=run.scores)

    with pytest.raises(
        ValueError,
        match="^the margins 'fixture margins': the growth_seen read is not a validation read of "
        'the reference comparators alone',
    ):
        _decide([arm], _margin_read(margins, log_records=reads), store)
    with pytest.raises(
        ValueError,
        match=
        "^the margins 'fixture margins': the growth_seen read scored .*'covariates\\+embedding'",
    ):
        _decide([arm], candidate_read, store)

def test_margins_whose_delta_is_not_r_times_the_reference_mse_are_refused(
    candidates, margins, store
):
    arm = candidates('tree', _both(0.5))
    first = margins.margins[0]
    doubled = first.model_copy(update={'margin': 2 * first.margin})
    rescaled = first.model_copy(
        update={
            'reference_mse': 2 * first.reference_mse,
            'margin': R * 2 * first.reference_mse
        }
    )

    for entry in (doubled, rescaled):
        edited = margins.model_copy(update={'margins': [entry, *margins.margins[1:]]})
        with pytest.raises(
            ValueError,
            match="^the margins 'fixture margins': the growth_seen δ is not r × the reference's",
        ):
            _decide([arm], edited, store)
    with pytest.raises(ValueError, match="name the reference 'covariates\\+one_hot'"):
        _decide([arm], margins.model_copy(update={'reference': 'covariates+one_hot'}), store)

def test_the_decision_refuses_a_candidate_whose_reference_predicts_otherwise(
    candidates, margins, store
):
    arm = candidates('tree', _both(0.5))
    (run, ) = arm.runs
    reference = pl.col('comparator') == REFERENCE_COMPARATOR
    predictions = store.read_frame(run.predictions).with_columns(
        prediction=pl.when(reference).then(pl.col('prediction') + 1e-12).otherwise('prediction')
    )
    scores = regressor_scores(predictions, SETTINGS.repeats, GROWTH_PANELS)
    refit = run.model_copy(
        update={
            'predictions': store.put_frame(predictions, 'predictions.parquet'),
            'scores': store.put_frame(scores, 'scores.parquet'),
        }
    )

    with pytest.raises(ValueError, match='^tree: its covariates\\+ancestors predictions differ'):
        _decide([arm.model_copy(update={'runs': [refit]})], margins, store)

def test_a_candidate_whose_statistics_are_not_its_predictions_is_refused(candidates, store):
    '''The decision scores the predictions the pairing guard compared, and the record must agree.'''

    arm = candidates('tree', _both(0.5))
    (run, ) = arm.runs
    shifted = store.read_frame(run.predictions).with_columns(prediction=pl.col('prediction') + 1.0)
    moved = run.model_copy(update={'predictions': store.put_frame(shifted, 'predictions.parquet')})

    with pytest.raises(ValueError, match="^tree: the record's statistics are not those of its"):
        check_economic_arm(arm.model_copy(update={'runs': [moved]}), store)

def test_a_decision_reads_distinct_candidates_none_named_for_the_reference(
    candidates, margins, store
):
    better = candidates('better', _both(0.5))
    named = candidates(REFERENCE_COMPARATOR, _both(0.5))

    with pytest.raises(ValueError, match='at least one candidate'):
        _decide([], margins, store)
    for arms in ([better, better], [named]):
        with pytest.raises(ValueError, match='^candidate names must be distinct'):
            _decide(arms, margins, store)
