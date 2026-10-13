'''
The economic decision on the fixture growth panel (Req 24; roadmap Stage 2 Exit).

The margin read scores the reference, ``covariates+ancestors``, beside the other comparators that
read no candidate. Every candidate's read fits the same rows, folds and penalties, so the
reference's errors pair exactly with each candidate's.
'''

import hashlib
import json
import platform
from dataclasses import replace
from datetime import datetime

import polars as pl
import pytest
from pydantic import ValidationError

from naics_embedder.decision.economic import (
    GROWTH_PANELS,
    REFERENCE_COMPARATOR,
    check_economic_arm,
    check_reference_pairing,
    fix_relative_margins,
    read_deterministic_arm,
)
from naics_embedder.decision.records import (
    EconomicArmRecord,
    RelativeMarginRecord,
    read_record,
    write_record,
)
from naics_embedder.decision.scores import panel_statistic
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
