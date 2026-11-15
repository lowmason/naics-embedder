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
import zlib
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from naics_embedder.decision.decide import check_arm
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
    ArmRecord,
    ArmSpec,
    ArtifactRef,
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
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.tree_model.builders import build_candidate
from tests.fixtures.decision import spec, synthetic_arm, write_text_only
from tests.fixtures.economic import (
    CODEBOOK_INPUT,
    candidate_spec,
    list_for,
    scaled_arm,
    write_candidate,
    write_candidate_list,
    write_codebook,
)
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, SETTINGS, coordinate_table

pytestmark = pytest.mark.unit

PURPOSE = 'fixture economic decision'
# U13
R = 0.05
# The column that binds a read's predictions to its table (P4: economic.BINDING_COLUMN)
BINDING = 'table_fingerprint'

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

@pytest.mark.parametrize('r', [0.0, -0.05, float('nan'), float('inf')])
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

def test_a_single_replicate_record_validates_and_pairs_with_the_reference(
    arm, margins, store, text_only
):
    '''Verification "Deterministic arms": the reference predicts as in the margin read.'''

    check_economic_arm(arm, store, list_for([arm], store, text_only_sha256=sha256_file(text_only)))
    check_reference_pairing(margins, arm, store)

    def reference(predictions):
        columns = store.read_frame(margins.read.predictions).columns
        return store.read_frame(predictions).select(columns).filter(
            pl.col('comparator') == REFERENCE_COMPARATOR
        )

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
    '''D9 holds for a text-only candidate, which reads the backbone; a tree candidate reads none.'''

    other = write_text_only(tmp_path / 'other', revision='other', codes=CODEBOOK)
    table = build_candidate(
        'text-only',
        3,
        write_codebook(tmp_path / 'cb.parquet'),
        tmp_path / 'c' / 'text-only-d3.parquet',
        text_only=text_only,
    )
    text = spec('text-only-d3', dimension=3, geometry='euclidean')
    wide = candidate_spec('tree', dimension=4)

    with pytest.raises(ValueError, match='^text-only-d3: the text-only table was built from .*D9'):
        _read(table, growth_panel, other, margins, store, lock, arm_spec=text)
    with pytest.raises(ValueError, match="^tree: the table has dimension 3, not the spec's 4"):
        _read(candidate, growth_panel, text_only, margins, store, lock, arm_spec=wide)
    assert log.records() == margins.read.log_records

@pytest.mark.parametrize('run_id', ['', '   '], ids=['empty', 'blank'])
def test_an_imported_run_with_a_blank_run_id_does_not_load(arm, tmp_path, run_id):
    '''A run's id names its selection-log records, so an id of spaces names nothing.'''

    data = json.loads(arm.model_dump_json())
    data['runs'][0]['run_id'] = run_id
    path = tmp_path / 'imported.json'
    path.write_text(json.dumps(data))

    with pytest.raises(ValidationError, match='runs\\.0\\.run_id'):
        read_record(path, EconomicArmRecord)

def test_an_economic_arm_is_one_replicate_with_no_seed(arm, store, text_only):
    (run, ) = arm.runs
    seeded = arm.model_copy(update={'runs': [run.model_copy(update={'seed': 0})]})
    twice = arm.model_copy(update={'runs': [run, run]})

    for record in (seeded, twice):
        with pytest.raises(ValueError, match='^tree: an economic arm is one deterministic'):
            check_economic_arm(
                record, store, list_for([arm], store, text_only_sha256=sha256_file(text_only))
            )

def test_a_run_whose_reads_name_another_table_is_refused(arm, store, text_only):
    (run, ) = arm.runs
    reads = json.loads(json.dumps(run.log_records))
    reads[0]['detail']['arm'] = 'another table'
    edited = arm.model_copy(update={'runs': [run.model_copy(update={'log_records': reads})]})

    with pytest.raises(ValueError, match="^tree: a growth_seen read names another \\['arm'\\]"):
        check_economic_arm(
            edited, store, list_for([arm], store, text_only_sha256=sha256_file(text_only))
        )

def test_a_provenance_edited_after_it_was_stored_is_refused(arm, store, text_only):
    (run, ) = arm.runs
    content = {**run.provenance.content, 'inputs': {'qcew': 'national slices'}}
    provenance = run.provenance.model_copy(update={'content': content})
    edited = arm.model_copy(update={'runs': [run.model_copy(update={'provenance': provenance})]})

    with pytest.raises(ValueError, match="^tree: the record's provenance differs from its stored"):
        check_economic_arm(
            edited, store, list_for([arm], store, text_only_sha256=sha256_file(text_only))
        )

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

def test_an_economic_arm_with_no_runs_pairs_nothing(arm, margins, store):
    '''
    A loop over no runs would pass an arm that read nothing, so the check refuses it first. It
    compares every run, so two runs that each predict as the margin read pair.
    '''

    (run, ) = arm.runs
    none = arm.model_copy(update={'runs': []})
    twice = arm.model_copy(update={'runs': [run, run]})

    with pytest.raises(
        ValueError, match='^tree: an economic arm with no runs pairs nothing \\(Req 24\\)$'
    ):
        check_reference_pairing(margins, none, store)
    check_reference_pairing(margins, twice, store)

def test_every_run_of_an_arm_pairs_with_the_margin_read(arm, margins, store):
    '''§4.4: the pairing compares each run's reference predictions, not the first run's alone.'''

    (run, ) = arm.runs
    reference = pl.col('comparator') == REFERENCE_COMPARATOR
    predictions = store.read_frame(run.predictions).with_columns(
        prediction=pl.when(reference).then(pl.col('prediction') + 1e-12).otherwise('prediction')
    )
    moved = run.model_copy(
        update={'predictions': store.put_frame(predictions, 'predictions.parquet')}
    )

    with pytest.raises(
        ValueError,
        match="^tree: its covariates\\+ancestors predictions differ from the margin read's"
    ):
        check_reference_pairing(margins, arm.model_copy(update={'runs': [run, moved]}), store)

# -------------------------------------------------------------------------------------------------
# A deterministic candidate against the frozen list (§4.3, §4.4, §4.5, §4.8)
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def frozen(arm, store, text_only):
    '''The frozen list of exactly the fixture candidate, pinned to its table.'''

    return list_for([arm], store, text_only_sha256=sha256_file(text_only))

@pytest.fixture
def text_table(text_only, tmp_path):
    '''``text-only-d3`` as its builder writes it, with the codebook the builder read.'''

    codebook = write_codebook(tmp_path / 'codebook' / 'naics_codebook.parquet')
    table = build_candidate(
        'text-only', 3, codebook, tmp_path / 'built' / 'text-only-d3.parquet', text_only=text_only
    )
    return table, codebook

TEXT_ONLY_D3 = spec('text-only-d3', dimension=3, geometry='euclidean')

def _with_run(arm, **update):
    '''The arm with its one run updated.'''

    (run, ) = arm.runs
    return arm.model_copy(update={'runs': [run.model_copy(update=update)]})

def _first_row(column, value):
    '''``column`` with its first row set to ``value`` and every other row kept.'''

    first = pl.int_range(pl.len()) == 0
    return pl.when(first).then(value).otherwise(pl.col(column)).alias(column)

def test_a_text_only_candidates_embedding_predicts_as_its_text_only_comparator(
    growth_panel, text_only, margins, store, lock, tmp_path
):
    '''§6 "D9 consistency": text-only-dN's covariates+embedding is its covariates+text_only.'''
    table = build_candidate(
        'text-only',
        3,
        write_codebook(tmp_path / 'cb.parquet'),
        tmp_path / 'c' / 'text-only-d3.parquet',
        text_only=text_only
    )
    arm = _read(
        table,
        growth_panel,
        text_only,
        margins,
        store,
        lock,
        arm_spec=spec('text-only-d3', dimension=3, geometry='euclidean')
    )
    keys = ['panel', 'repeat', 'code', 'feature_year']
    predictions = store.read_frame(arm.runs[0].predictions)
    embedding = predictions.filter(pl.col('comparator') == 'covariates+embedding').sort(keys)
    text = predictions.filter(pl.col('comparator') == 'covariates+text_only').sort(keys)
    assert embedding.get_column('prediction').equals(text.get_column('prediction'))

def test_a_reads_predictions_are_bound_to_its_table(arm, margins, store):
    '''P4: each stored prediction names its table; the margin read's carry no binding.'''

    (run, ) = arm.runs
    predictions = store.read_frame(run.predictions)

    assert predictions.get_column(BINDING).unique().to_list() == [run.table.matrix_fingerprint]
    assert BINDING not in store.read_frame(margins.read.predictions).columns

def test_a_binding_stored_with_the_predictions_does_not_break_the_pairing(arm, margins, store):
    '''§4.8: the reference comparison reads only the margin read's own columns.'''

    (run, ) = arm.runs
    assert BINDING in store.read_frame(run.predictions).columns

    check_reference_pairing(margins, arm, store)

def test_a_text_only_candidate_passes_its_checks_and_is_d9_checked(
    text_table, growth_panel, text_only, margins, store, lock
):
    '''A text-only candidate reads the backbone, so its record's spec is D9-checked too.'''

    table, codebook = text_table
    arm = _read(table, growth_panel, text_only, margins, store, lock, arm_spec=TEXT_ONLY_D3)
    frozen = list_for(
        [arm],
        store,
        text_only_sha256=sha256_file(text_only),
        codebook_sha256=sha256_file(codebook),
    )
    other = TEXT_ONLY_D3.model_copy(update={'backbone_revision': 'other'})

    check_economic_arm(arm, store, frozen)
    check_reference_pairing(margins, arm, store)
    with pytest.raises(ValueError, match='^text-only-d3: the text-only table was built from .*D9'):
        check_economic_arm(arm.model_copy(update={'spec': other}), store, frozen)

def test_a_text_only_candidate_reads_the_backbone(
    text_table, growth_panel, text_only, margins, store, lock
):
    '''A spec that reads no backbone is read without D9, and refused as a text-only candidate's.'''

    table, codebook = text_table
    arm = _read(
        table,
        growth_panel,
        text_only,
        margins,
        store,
        lock,
        arm_spec=candidate_spec('text-only-d3'),
    )
    frozen = list_for(
        [arm],
        store,
        text_only_sha256=sha256_file(text_only),
        codebook_sha256=sha256_file(codebook),
    )

    with pytest.raises(
        ValueError,
        match="^text-only-d3: a text-only candidate reads the text-only table's backbone"
    ):
        check_economic_arm(arm, store, frozen)

def test_a_smoothed_candidate_reads_the_backbone(
    text_table, growth_panel, text_only, margins, store, lock, tmp_path
):
    '''A smoothed candidate reads the text-only table too: every kind but the tree reads one.'''

    table, codebook = text_table
    text = _read(table, growth_panel, text_only, margins, store, lock, arm_spec=TEXT_ONLY_D3)
    smoothed_table = build_candidate(
        'smoothed', 3, codebook, tmp_path / 'built' / 'smoothed-d3.parquet', text_only=text_only
    )
    smoothed = _read(
        smoothed_table,
        growth_panel,
        text_only,
        margins,
        store,
        lock,
        arm_spec=candidate_spec('smoothed-d3'),
    )
    frozen = list_for(
        [text, smoothed],
        store,
        text_only_sha256=sha256_file(text_only),
        codebook_sha256=sha256_file(codebook),
    )

    with pytest.raises(
        ValueError, match="^smoothed-d3: a smoothed candidate reads the text-only table's backbone"
    ):
        check_economic_arm(smoothed, store, frozen)

def test_a_tree_candidate_reads_no_backbone_and_only_the_shared_text_only_table(
    arm, candidate, growth_panel, text_only, margins, store, lock
):
    '''P7: a tree candidate's spec names no backbone, and the list pins its text-only table.'''

    backboned = _read(
        candidate,
        growth_panel,
        text_only,
        margins,
        store,
        lock,
        arm_spec=spec('tree', dimension=3, geometry='euclidean'),
    )

    with pytest.raises(ValueError, match='^tree: a tree candidate reads no backbone'):
        check_economic_arm(
            backboned,
            store,
            list_for([backboned], store, text_only_sha256=sha256_file(text_only)),
        )
    with pytest.raises(ValueError, match='^tree: .*the shared text-only table'):
        check_economic_arm(arm, store, list_for([arm], store, text_only_sha256='f' * 64))

def test_a_tree_candidate_is_not_d9_checked(
    candidate, growth_panel, margins, store, lock, frozen, tmp_path
):
    '''
    A tree candidate reads no backbone, so a text-only table from another revision is read beside
    it, and its check reads that table's bytes alone: here they are the shared table's.
    '''

    other = write_text_only(tmp_path / 'other', revision='other', codes=CODEBOOK)

    arm = _read(candidate, growth_panel, other, margins, store, lock)

    assert arm.text_only.revision == 'other'
    check_economic_arm(arm, store, frozen)

def test_a_recorded_text_only_field_that_differs_from_its_stored_provenance_is_refused(
    arm, store, frozen
):
    edited = arm.model_copy(
        update={'text_only': arm.text_only.model_copy(update={'max_length': 99})}
    )

    with pytest.raises(
        ValueError, match="^tree: the record's text-only fields \\['max_length'\\] differ"
    ):
        check_economic_arm(edited, store, frozen)

@pytest.mark.parametrize(
    'row, message',
    [
        ({
            'dimension': 4
        }, 'its frozen-list row'),
        ({
            'components': 2
        }, 'its frozen-list row'),
        ({
            'geometry': 'spherical'
        }, 'its frozen-list row'),
        ({
            'table_sha256': None
        }, 'pins no table'),
        ({
            'table_sha256': 'f' * 64
        }, 'differs from its pin'),
    ],
    ids=['dimension', 'components', 'geometry', 'unpinned', 'another-sha'],
)
def test_a_candidate_off_its_list_row_is_refused(arm, store, frozen, row, message):
    (entry, ) = frozen.candidates
    edited = frozen.model_copy(update={'candidates': [entry.model_copy(update=row)]})

    with pytest.raises(ValueError, match=f'^tree: .*{message}'):
        check_economic_arm(arm, store, edited)

def test_an_arm_whose_name_is_not_on_the_list_is_refused(arm, store, frozen):
    '''The arm's row is the one that bears its name, even beside a row of its exact shape.'''

    (entry, ) = frozen.candidates
    renamed = frozen.model_copy(update={'candidates': [entry.model_copy(update={'name': 'other'})]})

    with pytest.raises(ValueError, match="^'tree' is not on the frozen candidate list"):
        check_economic_arm(arm, store, renamed)

def test_a_candidate_whose_provenance_breaks_the_rule_is_refused(
    growth_panel, text_only, margins, store, lock, tmp_path
):
    '''Req 29: a provenance that names a QCEW file describes its table, so it is read; then refused.'''

    table = write_candidate(
        tmp_path / 'qcew',
        'tree',
        coordinate_table(CODEBOOK),
        qcew={
            'path': 'qcew.parquet',
            'sha256': 'f' * 64
        },
    )
    arm = _read(table, growth_panel, text_only, margins, store, lock)

    with pytest.raises(ValueError, match='^tree: entries: .*\\(Req 29'):
        check_economic_arm(
            arm, store, list_for([arm], store, text_only_sha256=sha256_file(text_only))
        )

@pytest.mark.parametrize(
    'target, update',
    [
        ('run', {
            'seed': 0
        }),
        ('run', {
            'checkpoint': ArtifactRef(path='objects/a', sha256='a' * 64, bytes=1)
        }),
        ('run', {
            'training_run': 't-0'
        }),
        ('run', {
            'checkpoint_epoch': 0
        }),
        ('run', {
            'monitor_records': [{
                'mrr': 0.5,
                'read': {}
            }]
        }),
        ('arm', {
            'source': ArtifactRef(path='records/x.json', sha256='b' * 64, bytes=1)
        }),
    ],
    ids=['seed', 'checkpoint', 'training-run', 'epoch', 'monitor-records', 'source'],
)
def test_a_deterministic_run_with_selection_evidence_is_refused(arm, store, frozen, target, update):
    '''§4.4: a deterministic run keeps no seed and no selection evidence, and names no source.'''

    edited = _with_run(arm, **update) if target == 'run' else arm.model_copy(update=update)

    with pytest.raises(ValueError, match='^tree: .*one seedless run'):
        check_economic_arm(edited, store, frozen)

@pytest.mark.parametrize('count', [0, 2], ids=['none', 'two'])
def test_a_deterministic_arm_holds_exactly_one_run(arm, store, frozen, count):
    (run, ) = arm.runs

    with pytest.raises(ValueError, match='^tree: .*one seedless run'):
        check_economic_arm(arm.model_copy(update={'runs': [run] * count}), store, frozen)

@pytest.mark.parametrize(
    'edit',
    [
        lambda frame: frame.with_columns(_first_row(BINDING, pl.lit('another table'))),
        lambda frame: frame.drop(BINDING),
    ],
    ids=['two-tables', 'unbound'],
)
def test_predictions_bound_to_two_tables_or_to_none_are_refused(arm, store, frozen, edit):
    (run, ) = arm.runs
    predictions = edit(store.read_frame(run.predictions))
    edited = _with_run(arm, predictions=store.put_frame(predictions, 'predictions.parquet'))

    with pytest.raises(ValueError, match='^tree: its predictions belong to another table'):
        check_economic_arm(edited, store, frozen)

@pytest.mark.parametrize(
    'edit',
    [
        lambda statistics: {
            'growth_seen': statistics['growth_seen']
        },
        lambda statistics: {
            **statistics, 'regressor_seen': statistics['growth_seen']
        },
    ],
    ids=['dropped', 'extra'],
)
def test_a_record_whose_statistics_name_other_panels_is_refused(arm, store, frozen, edit):
    '''Deferred :650: the statistics' keys are exactly the growth panels.'''

    (run, ) = arm.runs
    edited = _with_run(arm, statistics=edit(run.statistics))

    with pytest.raises(ValueError, match="^tree: the record's statistics are not those of its"):
        check_economic_arm(edited, store, frozen)

@pytest.mark.parametrize(
    'field, value',
    [('dimension', 3.0), ('level', True), ('level', 6.0)],
    ids=['dimension-float', 'level-bool', 'level-float'],
)
def test_a_logged_identity_of_another_type_is_refused(arm, store, frozen, field, value):
    '''Deferred :650: a logged identity is the named one and of its type (``_same``).'''

    (run, ) = arm.runs
    reads = json.loads(json.dumps(run.log_records))
    reads[0]['detail'][field] = value
    edited = _with_run(arm, log_records=reads)

    with pytest.raises(
        ValueError, match=f"^tree: a growth_seen read names another \\['{field}'\\]"
    ):
        check_economic_arm(edited, store, frozen)

def _stored_scores(arm, store, edit):
    '''The arm with its stored scores edited and stored again.'''

    (run, ) = arm.runs
    return _with_run(
        arm, scores=store.put_frame(edit(store.read_frame(run.scores)), 'scores.parquet')
    )

def test_a_record_whose_stored_scores_differ_from_its_predictions_is_refused(arm, store, frozen):
    '''Deferred :633: one stored value raised by 1e-6 is not the rescore of the predictions.'''

    edited = _stored_scores(
        arm, store, lambda scores: scores.with_columns(_first_row('value',
                                                                  pl.col('value') + 1e-6))
    )

    with pytest.raises(ValueError, match='^tree: .*stored scores'):
        check_economic_arm(edited, store, frozen)

def test_stored_scores_within_the_tolerance_are_accepted(arm, store, frozen):
    '''A stored value one part in 1e13 from the rescore agrees with it to float summation order.'''

    (run, ) = arm.runs
    scores = store.read_frame(run.scores)
    nudged = scores.with_columns(_first_row('value', pl.col('value') * (1 + 1e-13)))
    assert nudged.get_column('value')[0] != scores.get_column('value')[0]

    check_economic_arm(_stored_scores(arm, store, lambda _: nudged), store, frozen)

def test_stored_scores_beyond_the_tolerance_are_refused(arm, store, frozen):
    '''One part in 1e10 is beyond STATISTIC_TOLERANCE, though within math.isclose's default.'''

    edited = _stored_scores(
        arm,
        store,
        lambda scores: scores.with_columns(_first_row('value',
                                                      pl.col('value') * (1 + 1e-10))),
    )

    with pytest.raises(ValueError, match='^tree: .*stored scores'):
        check_economic_arm(edited, store, frozen)

@pytest.mark.parametrize(
    'edit',
    [
        lambda scores: scores.slice(1),
        lambda scores: pl.concat([scores, scores.head(1)]),
        # A trailing space keeps the row where the sort puts it, so only its key differs
        lambda scores: scores.with_columns(_first_row('item',
                                                      pl.col('item') + ' ')),
    ],
    ids=['dropped', 'repeated', 'renamed'],
)
def test_stored_scores_hold_exactly_the_rescores_rows(arm, store, frozen, edit):
    with pytest.raises(ValueError, match='^tree: .*stored scores'):
        check_economic_arm(_stored_scores(arm, store, edit), store, frozen)

# -------------------------------------------------------------------------------------------------
# Records (§4.3, §4.4: the backbone-less spec and a trained candidate's selection evidence)
# -------------------------------------------------------------------------------------------------

def test_a_stage_8_record_loads_and_dumps_unchanged(tmp_path):
    '''§4.3: ArmSpec's change keeps Stage 8's records loading and validating unchanged.'''
    store = ArtifactStore(tmp_path / 'store')
    arm = synthetic_arm(
        store,
        tmp_path,
        spec('euclidean-d16', dimension=16, geometry='euclidean'), {},
        monitor_mrrs=(0.1, 0.3, 0.2)
    )
    text = arm.model_dump_json(indent=2)
    assert ArmRecord.model_validate_json(text).model_dump_json(indent=2) == text
    check_arm(ArmRecord.model_validate_json(text), store, 5)

NO_BACKBONE = dict.fromkeys(
    ('backbone', 'backbone_revision', 'descriptions_sha256', 'summaries_sha256', 'max_length')
)

def test_a_spec_reads_its_backbone_or_no_backbone():
    tree = spec('tree-d3', dimension=3, geometry='euclidean', **NO_BACKBONE)
    assert not tree.reads_backbone
    assert spec('text-only-d3').reads_backbone
    for partial in ({'backbone': None}, {'max_length': None}, {'descriptions_sha256': None}):
        with pytest.raises(ValidationError, match='reads its backbone or no backbone'):
            spec('partial', **partial)

def test_a_trained_run_with_its_selection_evidence_loads(arm, tmp_path):
    '''Replaces test_an_imported_run_with_a_checkpoint_or_monitor_reads_does_not_load: the
    fields exist now, and check_economic_arm refuses them on a deterministic run (Task 5).'''
    data = json.loads(arm.model_dump_json())
    data['runs'][0].update(
        seed=1,
        checkpoint={
            'path': 'objects/a',
            'sha256': 'a' * 64,
            'bytes': 1
        },
        training_run='t-1',
        checkpoint_epoch=2,
        monitor_records=[{
            'mrr': 0.5,
            'read': {}
        }]
    )
    data['source'] = {'path': 'records/x.json', 'sha256': 'b' * 64, 'bytes': 1}
    (tmp_path / 'trained.json').write_text(json.dumps(data))
    loaded = read_record(tmp_path / 'trained.json', EconomicArmRecord)
    assert (loaded.runs[0].training_run, loaded.runs[0].checkpoint_epoch) == ('t-1', 2)
    assert loaded.source.path == 'records/x.json'

@pytest.mark.parametrize(
    'field, value', [
        ('checkpoint_epoch', 1.0), ('checkpoint_epoch', '1'), ('seed', 1.0), ('training_run', '  ')
    ]
)
def test_imported_evidence_identities_are_strict(arm, tmp_path, field, value):
    data = json.loads(arm.model_dump_json())
    data['runs'][0][field] = value
    (tmp_path / 'x.json').write_text(json.dumps(data))
    with pytest.raises(ValidationError, match=f'runs\\.0\\.{field}'):
        read_record(tmp_path / 'x.json', EconomicArmRecord)

# One value for each of a spec's five D9 fields, to set on a spec that reads no backbone
D9_VALUES = {
    'backbone': 'tiny-backbone',
    'backbone_revision': 'abc123',
    'descriptions_sha256': 'd' * 64,
    'summaries_sha256': 'e' * 64,
    'max_length': 16,
}

@pytest.mark.parametrize('field', sorted(D9_VALUES))
def test_a_spec_with_one_backbone_field_set_and_the_rest_null_is_refused(field):
    '''No backbone means all five D9 fields null: a revision or a window names none.'''

    with pytest.raises(ValidationError, match='reads its backbone or no backbone'):
        spec('half', **{**NO_BACKBONE, field: D9_VALUES[field]})

@pytest.mark.parametrize('pinned', ['backbone_revision', 'summaries_sha256'])
def test_a_spec_reads_a_backbone_with_no_pinned_revision_or_summaries(pinned):
    '''The revision and the summaries are the two D9 fields a backbone may lack.'''

    unpinned = spec('unpinned', **{pinned: None})
    both = spec('both-unpinned', backbone_revision=None, summaries_sha256=None)
    assert unpinned.reads_backbone and both.reads_backbone
    assert ArmSpec.model_validate_json(both.model_dump_json()) == both

def test_a_spec_that_reads_no_backbone_dumps_five_null_keys_and_loads_back():
    tree = spec('tree-d3', dimension=3, geometry='euclidean', **NO_BACKBONE)
    dumped = json.loads(tree.model_dump_json())
    assert {field: dumped[field] for field in NO_BACKBONE} == NO_BACKBONE
    assert ArmSpec.model_validate_json(tree.model_dump_json()) == tree

@pytest.mark.parametrize('field', sorted(NO_BACKBONE))
def test_every_backbone_field_of_a_spec_is_still_a_required_key(field):
    '''Null is a value, not an absent key: the JSON shape of Stage 8's specs does not change.'''

    data = json.loads(spec('text-only-d3').model_dump_json())
    del data[field]
    with pytest.raises(ValidationError, match=f'{field}\\n  Field required'):
        ArmSpec.model_validate(data)

@pytest.mark.parametrize('max_length', [0, -1])
def test_a_spec_that_reads_a_backbone_has_a_window_of_at_least_one(max_length):
    with pytest.raises(
        ValidationError, match='max_length\\n  Input should be greater than or equal'
    ):
        spec('narrow', max_length=max_length)

def test_a_spec_that_reads_a_backbone_may_have_a_window_of_exactly_one():
    '''One is the smallest window: the bound is ``ge=1``, not ``gt=1``.'''

    narrowest = spec('narrowest', max_length=1)
    assert narrowest.max_length == 1 and narrowest.reads_backbone
    assert ArmSpec.model_validate_json(narrowest.model_dump_json()) == narrowest

def test_a_deterministic_run_carries_no_selection_evidence(arm, tmp_path):
    '''The four fields and the source default to none, so such a record needs no key for them.'''

    (run, ) = arm.runs
    assert run.seed is None
    assert (run.checkpoint, run.training_run, run.checkpoint_epoch) == (None, None, None)
    assert run.monitor_records == []
    assert arm.source is None

    data = json.loads(arm.model_dump_json())
    for key in ('checkpoint', 'training_run', 'checkpoint_epoch', 'monitor_records'):
        del data['runs'][0][key]
    del data['source']
    (tmp_path / 'bare.json').write_text(json.dumps(data))
    assert read_record(tmp_path / 'bare.json', EconomicArmRecord) == arm

def test_a_trained_runs_evidence_and_source_survive_a_round_trip(arm, tmp_path):
    checkpoint = ArtifactRef(path='objects/a', sha256='a' * 64, bytes=7)
    source = ArtifactRef(path='records/x.json', sha256='b' * 64, bytes=9)
    monitor = [{'mrr': 0.5, 'read': {'epoch': 0}}, {'mrr': 0.7, 'read': {'epoch': 1}}]
    (run, ) = arm.runs
    trained_run = run.model_copy(
        update={
            'seed': 0,
            'checkpoint': checkpoint,
            'training_run': 't-0',
            'checkpoint_epoch': 0,
            'monitor_records': monitor,
        }
    )
    trained = arm.model_copy(update={'runs': [trained_run], 'source': source})
    write_record(trained, tmp_path / 'trained.json')

    loaded = read_record(tmp_path / 'trained.json', EconomicArmRecord)
    assert loaded == trained
    (loaded_run, ) = loaded.runs
    assert (loaded_run.seed, loaded_run.checkpoint_epoch, loaded_run.training_run) == (0, 0, 't-0')
    assert loaded_run.checkpoint == checkpoint
    assert loaded_run.monitor_records == monitor
    assert loaded.source == source

def _refusal(arm, tmp_path, field, value):
    '''The (location, error type) of each error the run's field set to ``value`` raises.'''

    data = json.loads(arm.model_dump_json())
    data['runs'][0][field] = value
    (tmp_path / 'x.json').write_text(json.dumps(data))
    with pytest.raises(ValidationError) as refused:
        read_record(tmp_path / 'x.json', EconomicArmRecord)
    return [(item['loc'], item['type']) for item in refused.value.errors()]

@pytest.mark.parametrize(
    'field, value, error',
    [
        ('checkpoint_epoch', 1.0, 'int_type'),
        ('checkpoint_epoch', '1', 'int_type'),
        ('checkpoint_epoch', True, 'int_type'),
        ('checkpoint_epoch', -1, 'greater_than_equal'),
        ('training_run', '  ', 'string_pattern_mismatch'),
        ('training_run', '', 'string_too_short'),
        ('training_run', 7, 'string_type'),
        ('checkpoint', 'objects/a', 'model_type'),
    ],
)
def test_imported_evidence_is_refused_for_what_it_is_not_for_being_there(
    arm, tmp_path, field, value, error
):
    '''The fields exist now, so the refusal is their own check and no longer "Extra inputs".'''

    assert _refusal(arm, tmp_path, field, value) == [(('runs', 0, field), error)]

@pytest.mark.parametrize('seed', [1.0, True])
def test_a_trained_runs_seed_is_an_integer(arm, tmp_path, seed):
    '''A float or a bool would load as a seed, and a seed names a training run's stream.'''

    assert _refusal(arm, tmp_path, 'seed', seed) == [(('runs', 0, 'seed'), 'int_type')]

@pytest.mark.parametrize(
    'source', [
        {
            'path': 'records/x.json'
        }, {
            'path': 'records/x.json',
            'sha256': 'b' * 64,
            'bytes': -1
        }
    ]
)
def test_a_source_names_a_stored_file_by_path_sha256_and_size(arm, tmp_path, source):
    data = json.loads(arm.model_dump_json())
    data['source'] = source
    (tmp_path / 'x.json').write_text(json.dumps(data))
    with pytest.raises(ValidationError, match='source\\.'):
        read_record(tmp_path / 'x.json', EconomicArmRecord)

# -------------------------------------------------------------------------------------------------
# The economic decision (Verification "Economic decision record", "Margin first, reference only")
# -------------------------------------------------------------------------------------------------

REPLICATES = 2000
BOOTSTRAP_SEED = 20260924
GROWTH_SEEN, GROWTH_HELDOUT = GROWTH_PANELS

def _both(factor):
    return dict.fromkeys(GROWTH_PANELS, factor)

@pytest.fixture
def candidates(growth_panel, text_only, margins, store, lock, tmp_path):
    '''
    Reads a new candidate, after the margins, and scales its errors (``scaled_arm``). Each name
    has its own table, so two candidates' predictions are bound to two tables.
    '''

    def read(name, factors, **overrides):
        coordinates = coordinate_table(CODEBOOK, seed=zlib.crc32(name.encode()))
        table = write_candidate(tmp_path / 'candidates', name, coordinates)
        arm_spec = candidate_spec(name, **overrides)
        arm = _read(table, growth_panel, text_only, margins, store, lock, arm_spec)
        return scaled_arm(arm, factors, store)

    return read

def _decide(arms, margins, store):
    '''
    The decision over the arms, reading a frozen list of exactly their candidates, each once,
    written beside the store and pinned to their tables and their text-only table.
    '''

    distinct = list({arm.spec.name: arm for arm in arms}.values())
    # The arms share one text-only table; a list over no arms pins one that no check reads
    (text_only_sha256, ) = {arm.text_only.table.sha256 for arm in arms} or {'0' * 64}
    listed = write_candidate_list(
        store.root.parent / 'candidates.yaml', distinct, store, text_only_sha256=text_only_sha256
    )
    return decide_economic(
        'fixture economic decision',
        'which candidate, if any, is the economic artifact?',
        arms,
        margins,
        store,
        candidates=listed,
        replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

def test_a_record_whose_predictions_belong_to_another_table_is_refused(
    candidates, store, text_only
):
    a, b = candidates('a', _both(1.0)), candidates('b', _both(1.0))
    swapped = a.model_copy(
        update={'runs': [a.runs[0].model_copy(update={'predictions': b.runs[0].predictions})]}
    )
    frozen = list_for([a, b], store, text_only_sha256=sha256_file(text_only))
    with pytest.raises(ValueError, match='^a: its predictions belong to another table'):
        check_economic_arm(swapped, store, frozen)

def test_the_decision_reads_exactly_the_frozen_lists_candidates(
    candidates, margins, store, tmp_path, text_only
):
    a, b = candidates('a', _both(0.5)), candidates('b', _both(1.0))
    listed = write_candidate_list(
        tmp_path / 'list.yaml', [a, b], store, text_only_sha256=sha256_file(text_only)
    )
    settings = {'candidates': listed, 'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED}
    with pytest.raises(ValueError, match="'b'.*missing"):
        decide_economic('d', 'q', [a], margins, store, **settings)
    c = candidates('c', _both(1.0))
    with pytest.raises(ValueError, match="'c' is not on the frozen candidate list"):
        decide_economic('d', 'q', [a, b, c], margins, store, **settings)
    record = decide_economic('d', 'q', [a, b], margins, store, **settings)
    assert record.candidates.sha256 == sha256_file(listed)

def test_the_decision_reads_the_lists_names_before_any_other_check(
    candidates, margins, store, tmp_path, text_only
):
    '''An off-list or missing candidate is refused before the margins are checked.'''

    a, c = candidates('a', _both(0.5)), candidates('c', _both(0.5))
    broken = margins.model_copy(update={'reference': 'covariates+one_hot'})
    settings = {'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED}
    only_a = write_candidate_list(
        tmp_path / 'a.yaml', [a], store, text_only_sha256=sha256_file(text_only)
    )
    a_and_c = write_candidate_list(
        tmp_path / 'ac.yaml', [a, c], store, text_only_sha256=sha256_file(text_only)
    )

    with pytest.raises(ValueError, match="^'c' is not on the frozen candidate list"):
        decide_economic('d', 'q', [a, c], broken, store, candidates=only_a, **settings)
    with pytest.raises(ValueError, match="'c'.*missing"):
        decide_economic('d', 'q', [a], broken, store, candidates=a_and_c, **settings)

def test_the_decision_checks_each_candidate_against_the_list_it_reads(
    candidates, margins, store, tmp_path
):
    '''The list the decision loads is the one each candidate is checked against.'''

    arm = candidates('tree', _both(0.5))
    listed = write_candidate_list(tmp_path / 'list.yaml', [arm], store, text_only_sha256='f' * 64)

    with pytest.raises(ValueError, match='^tree: .*the shared text-only table'):
        decide_economic(
            'd',
            'q',
            [arm],
            margins,
            store,
            candidates=listed,
            replicates=REPLICATES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )

def test_the_decision_records_the_list_as_stored(candidates, margins, store, tmp_path):
    '''§4.4: the record names the list file's bytes in the store, and they load as the list.'''

    record = _decide([candidates('tree', _both(0.5))], margins, store)

    stored = store.resolve(record.candidates)
    assert stored.read_bytes() == (tmp_path / 'candidates.yaml').read_bytes()
    assert yaml.safe_load(stored.read_text())['candidates'][0]['name'] == 'tree'

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
    assert record.arms[0].runs[0].provenance.content['inputs'] == {'codebook': CODEBOOK_INPUT}
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

def test_margins_whose_stored_scores_differ_from_their_read_are_refused(candidates, margins, store):
    '''Deferred :633: the margin read's stored scores are the rescore of its predictions.'''

    arm = candidates('tree', _both(0.5))
    scores = store.read_frame(margins.read.scores)
    raised = scores.with_columns(_first_row('value', pl.col('value') + 1e-6))
    edited = _margin_read(margins, scores=store.put_frame(raised, 'scores.parquet'))

    with pytest.raises(ValueError, match="^the margins 'fixture margins': .*stored scores"):
        _decide([arm], edited, store)

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

def test_a_candidate_whose_statistics_are_not_its_predictions_is_refused(
    candidates, store, text_only
):
    '''The decision scores the predictions the pairing guard compared, and the record must agree.'''

    arm = candidates('tree', _both(0.5))
    (run, ) = arm.runs
    shifted = store.read_frame(run.predictions).with_columns(prediction=pl.col('prediction') + 1.0)
    moved = run.model_copy(update={'predictions': store.put_frame(shifted, 'predictions.parquet')})

    with pytest.raises(ValueError, match="^tree: the record's statistics are not those of its"):
        check_economic_arm(
            arm.model_copy(update={'runs': [moved]}),
            store,
            list_for([arm], store, text_only_sha256=sha256_file(text_only)),
        )

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
