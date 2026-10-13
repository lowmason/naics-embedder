'''
The regressor panel (Req 2; Req 4; Req 5; roadmap Stage 3 Exit; D1, D7, D8, D9).

Synthetic QCEW rows (``tests/fixtures/regressor_panel.py``) and a stub arm stand in for the real
data and a trained arm. Every outer set opened here is a fixture's: no test reads a real one.
'''

import hashlib
import json
import platform
import re
import sys

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels import regressor, regressor_splits
from naics_embedder.panels.outcome import SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.qcew_rows import (
    RegressorOutcome,
    level_cells,
    panel_rows,
    population,
)
from naics_embedder.panels.regressor import (
    LEVELS,
    OUTER_SPLITS,
    PANEL_NAMES,
    PREDICTION_COLUMNS,
    READ_NAMES,
    ArmTables,
    FitSettings,
    Regime,
    RegressorPanel,
    comparators,
    coordinate_matrix,
    feature_matrix,
    group_folds,
    heldout_validation_plan,
    load_regressor_panel,
    outer_plan,
    reference_comparators,
    seen_validation_plan,
    summarize,
    table_fingerprint,
    verify_branch_record,
)
from naics_embedder.panels.regressor_splits import (
    RegressorSplit,
    assign_splits,
    codes_fingerprint,
    group_table_fingerprint,
    write_group_table,
)
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import text_only_fingerprint
from naics_embedder.utils.config import RegressorBranchRecord, RegressorPanelConfig
from tests.fixtures.regressor_panel import (
    BRANCH_RECORD,
    CODEBOOK,
    HELDOUT_GROUPS,
    POPULATION,
    SETTINGS,
    coordinate_table,
    panel_rows_d61f49b,
    text_only_table,
    write_qcew_slices,
)

pytestmark = pytest.mark.unit

SEEN_COMPARATORS = (
    'covariates',
    'embedding',
    'covariates+embedding',
    'one_hot',
    'covariates+one_hot',
    'ancestors',
    'covariates+ancestors',
    'text_only',
    'covariates+text_only',
)
PURPOSE = 'exit test: fixture panel only'
# What a fit decides for a scored row: its penalty and its prediction
DECIDED = ['comparator', 'repeat', 'fold', 'code', 'feature_year', 'alpha', 'prediction']

@pytest.fixture
def log(tmp_path):
    return SelectionLog(tmp_path / 'selection_log.jsonl')

@pytest.fixture
def panel(regressor_rows, log):
    return RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)

def _keys(frame):
    return set(zip(frame.get_column('code').to_list(), frame.get_column('feature_year').to_list()))

def _split(rows, *splits):
    names = [split.value for split in splits]
    return assign_splits(rows, HELDOUT_GROUPS).filter(pl.col('split').is_in(names))

# -------------------------------------------------------------------------------------------------
# Regimes and comparators (Exit: reported separately, one-hot only in the seen regime)
# -------------------------------------------------------------------------------------------------

def test_one_hot_runs_only_in_the_seen_regime_and_ancestors_only_above_the_sector():
    without_one_hot = tuple(name for name in SEEN_COMPARATORS if 'one_hot' not in name)
    without_ancestors = tuple(name for name in SEEN_COMPARATORS if 'ancestors' not in name)

    assert comparators(Regime.SEEN, 6) == SEEN_COMPARATORS
    assert comparators(Regime.HELDOUT, 6) == without_one_hot
    assert comparators(Regime.SEEN, 2) == without_ancestors

def test_each_regime_is_its_own_panel_and_scores_every_comparator(
    panel, regressor_rows, regressor_arm
):
    remainder = _split(regressor_rows[6], RegressorSplit.REMAINDER)
    scored = {
        Regime.SEEN: remainder.filter(pl.col('feature_year') == 2023).height,
        Regime.HELDOUT: remainder.height,
    }

    for regime, name in ((Regime.SEEN, 'regressor_seen'), (Regime.HELDOUT, 'regressor_heldout')):
        predictions = panel.validation(regime, 6, regressor_arm, PURPOSE)

        assert predictions.columns == list(PREDICTION_COLUMNS)
        assert predictions.get_column('panel').unique().to_list() == [name]
        names = predictions.get_column('comparator').unique(maintain_order=True)
        assert tuple(names) == comparators(regime, 6)
        rows = predictions.group_by('comparator').len().get_column('len')
        assert rows.to_list() == [scored[regime] * SETTINGS.repeats] * len(rows)
        assert predictions.get_column('prediction').is_finite().all()

# -------------------------------------------------------------------------------------------------
# Tuning inside the remainder (Exit: the penalty is tuned inside the remainder, the outer set is
# read once)
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('regime', list(Regime))
def test_no_scored_row_tunes_its_own_penalty(regressor_rows, regime):
    frame = _split(regressor_rows[6], RegressorSplit.REMAINDER)
    build = seen_validation_plan if regime is Regime.SEEN else heldout_validation_plan
    groups = frame.get_column('group').to_list()
    codes = frame.get_column('code').to_list()

    plan = build(frame, 6, SETTINGS)

    assert len(plan) == SETTINGS.repeats * SETTINGS.folds
    for task in plan:
        scored = set(task.score)
        assert all(scored.isdisjoint({*fit, *tuned}) for fit, tuned in task.tuning)
        # Folds are grouped: no row of a scored group is scored while tuning the penalty
        scored_groups = {groups[row] for row in scored}
        assert all(
            scored_groups.isdisjoint(groups[row] for row in tuned) for _, tuned in task.tuning
        )
        if regime is Regime.SEEN:
            # Seen: every scored code has earlier rows in the fit set
            assert {codes[row] for row in scored} <= {codes[row] for row in task.fit}
        else:
            # Held-out: a scored group leaves the fit set and every tuning split
            assert {groups[row] for row in scored}.isdisjoint(groups[row] for row in task.fit)
            for fit, tuned in task.tuning:
                assert {*fit, *tuned} <= set(task.fit)
                assert {groups[row] for row in tuned}.isdisjoint(groups[row] for row in fit)

@pytest.mark.parametrize('regime', list(Regime))
def test_the_outer_set_is_scored_once_by_a_penalty_tuned_inside_the_remainder(
    regressor_rows, regime
):
    frame = _split(regressor_rows[6], RegressorSplit.REMAINDER, OUTER_SPLITS[regime])
    split = frame.get_column('split').to_list()
    remainder = {row for row, name in enumerate(split) if name == RegressorSplit.REMAINDER.value}

    [task] = outer_plan(frame, regime, 6, SETTINGS)

    assert (task.repeat, task.fold) == (0, -1)
    assert set(task.fit) == remainder
    assert sorted(task.score) == sorted(set(range(frame.height)) - remainder)
    assert task.tuning
    assert all({*fit, *tuned} <= remainder for fit, tuned in task.tuning)

def test_the_test_split_scores_each_outer_row_once_per_comparator(
    panel, regressor_rows, regressor_arm, log
):
    for regime in Regime:
        panel.open_outer(regime, PURPOSE)
        predictions = panel.test(regime, 6, regressor_arm, PURPOSE)

        outer = _keys(_split(regressor_rows[6], OUTER_SPLITS[regime]))
        assert set(predictions.get_column('fold')) == {-1}
        for _, rows in predictions.group_by('comparator'):
            assert rows.height == len(outer)
            assert _keys(rows) == outer

    assert [(r['event'], r['panel'], r['split']) for r in log.records()] == [
        ('open', 'regressor_seen', 'test'),
        ('read', 'regressor_seen', 'test'),
        ('open', 'regressor_heldout', 'test'),
        ('read', 'regressor_heldout', 'test'),
    ]

def test_validation_reads_only_the_remainder_and_logs_each_read(
    panel, regressor_rows, regressor_arm, log
):
    remainder = _keys(_split(regressor_rows[6], RegressorSplit.REMAINDER))

    seen = panel.validation(Regime.SEEN, 6, regressor_arm, PURPOSE)
    heldout = panel.validation(Regime.HELDOUT, 6, regressor_arm, PURPOSE)

    assert _keys(seen) <= remainder
    assert _keys(heldout) == remainder
    records = log.records()
    assert [(r['event'], r['panel'], r['split'], r['n_queries']) for r in records] == [
        ('read', 'regressor_seen', 'validation', len(remainder)),
        ('read', 'regressor_heldout', 'validation', len(remainder)),
    ]
    assert records[0]['fingerprint'] == panel.fingerprint
    assert records[0]['detail']['level'] == 6
    assert records[0]['detail']['arm'] == regressor_arm.fingerprint
    assert records[0]['detail']['text_only'] == regressor_arm.text_only_fingerprint
    assert records[0]['detail']['comparators'] == list(SEEN_COMPARATORS)

def test_a_read_carries_the_callers_detail_beside_the_panels_own(panel, regressor_arm, log):
    run = {'run': 'arm-a/seed-0', 'seed': 0}

    panel.validation(Regime.SEEN, 6, regressor_arm, PURPOSE, detail=run)
    panel.open_outer(Regime.SEEN, PURPOSE)
    panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE, detail=run)

    reads = [r for r in log.records() if r['event'] == 'read']
    assert [r['split'] for r in reads] == ['validation', 'test']
    for record in reads:
        assert record['detail']['run'] == 'arm-a/seed-0'
        assert record['detail']['seed'] == 0
        assert record['detail']['arm'] == regressor_arm.fingerprint
        assert record['detail']['text_only'] == regressor_arm.text_only_fingerprint

@pytest.mark.parametrize('key', ['arm', 'text_only', 'level', 'comparators', 'dimension'])
def test_a_read_cannot_replace_what_the_panel_logs(panel, regressor_arm, log, key):
    with pytest.raises(ValueError, match=key):
        panel.validation(Regime.SEEN, 6, regressor_arm, PURPOSE, detail={key: 'other'})
    panel.open_outer(Regime.SEEN, PURPOSE)
    with pytest.raises(ValueError, match=key):
        panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE, detail={key: 'other'})

    assert [r['event'] for r in log.records()] == ['open']

def test_the_logged_table_names_are_their_matrix_fingerprints(regressor_arm):
    coordinates = coordinate_table(CODEBOOK)
    text = text_only_table(CODEBOOK)

    assert regressor_arm.fingerprint == table_fingerprint(coordinates)
    assert regressor_arm.text_only_fingerprint == text_only_fingerprint(text)
    # Row order does not change a name
    assert table_fingerprint(coordinates.reverse()) == regressor_arm.fingerprint
    assert text_only_fingerprint(text.reverse()) == regressor_arm.text_only_fingerprint

def _shift_outcomes(rows, where):
    '''The panel rows, same order, with 10 added to the six-digit outcomes ``where`` picks.'''

    frame = assign_splits(rows[6], HELDOUT_GROUPS).sort('code', 'feature_year')
    outcome = pl.when(where).then(pl.col('outcome') + 10.0).otherwise(pl.col('outcome'))
    return {**rows, 6: frame.with_columns(outcome=outcome).select(rows[6].columns)}

def _read(rows, log_path, regime, split, arm):
    '''One read of a fresh panel over ``rows``; a test read opens the outer set first.'''

    panel = RegressorPanel(rows, HELDOUT_GROUPS, SelectionLog(log_path), SETTINGS)
    if split == 'validation':
        return panel.validation(regime, 6, arm, PURPOSE)
    panel.open_outer(regime, PURPOSE)
    return panel.test(regime, 6, arm, PURPOSE)

def test_no_outcome_outside_the_remainder_reaches_a_test_prediction(
    regressor_rows, regressor_arm, tmp_path
):
    # Both outer sets move; a test read fits and tunes on the remainder's outcomes only
    shifted = _shift_outcomes(regressor_rows, pl.col('split') != RegressorSplit.REMAINDER.value)

    for regime in Regime:
        base = _read(regressor_rows, tmp_path / 'base.jsonl', regime, 'test', regressor_arm)
        moved = _read(shifted, tmp_path / 'shifted.jsonl', regime, 'test', regressor_arm)

        assert (moved.get_column('outcome') != base.get_column('outcome')).all()
        assert moved.select(DECIDED).equals(base.select(DECIDED))

@pytest.mark.parametrize(
    ('regime', 'where'),
    [
        # A seen task fits the 2022 rows and tunes on the other folds' 2023 rows
        (Regime.SEEN, (pl.col('group') == '1111') & (pl.col('feature_year') == 2023)),
        # A held-out task leaves its scored groups out of its fit and every tuning split
        (Regime.HELDOUT, pl.col('group') == '1111'),
    ],
)
def test_no_scored_outcome_reaches_its_own_validation_prediction(
    regressor_rows, regressor_arm, tmp_path, regime, where
):
    remainder = pl.col('split') == RegressorSplit.REMAINDER.value
    shifted = _shift_outcomes(regressor_rows, where & remainder)

    base = _read(regressor_rows, tmp_path / 'base.jsonl', regime, 'validation', regressor_arm)
    moved = _read(shifted, tmp_path / 'shifted.jsonl', regime, 'validation', regressor_arm)

    # The tasks that score a shifted row: the group's fold in each repeat
    task = pl.col('repeat') * SETTINGS.folds + pl.col('fold')
    changed = moved.get_column('outcome') != base.get_column('outcome')
    scoring = base.filter(changed).select(task).to_series().unique().to_list()
    assert len(scoring) == SETTINGS.repeats
    scored = task.is_in(scoring)
    assert moved.filter(scored).select(DECIDED).equals(base.filter(scored).select(DECIDED))

# -------------------------------------------------------------------------------------------------
# The committed draw (Exit: the panel reads the committed outer groups, never a fresh draw)
# -------------------------------------------------------------------------------------------------

def _from_sources(tmp_path, cells, groups, **overrides):
    qcew = tmp_path / 'qcew'
    pins = write_qcew_slices(qcew, cells)
    table = tmp_path / 'regressor_heldout_groups.csv'
    fingerprint = write_group_table(groups, table)
    arguments = {
        'qcew_dir': qcew,
        'qcew_sha256': pins,
        'codebook_codes': CODEBOOK,
        'heldout_groups_csv': table,
        'heldout_groups_sha256': fingerprint,
        'log_path': tmp_path / 'selection_log.jsonl',
        'settings': SETTINGS,
        'branch_record': BRANCH_RECORD,
        'levels': (4, 6),
    }
    return RegressorPanel.from_sources(**{**arguments, **overrides}), table

def test_the_panel_reads_the_committed_groups_and_never_draws(
    tmp_path, monkeypatch, regressor_cells
):

    def no_draw(*args, **kwargs):
        raise AssertionError('the panel drew held-out groups')

    monkeypatch.setattr(regressor_splits, 'draw_heldout_groups', no_draw)

    panel, table = _from_sources(tmp_path, regressor_cells, ['1111'])

    assert not hasattr(regressor, 'draw_heldout_groups')
    assert panel.heldout_groups == ('1111', )
    assert panel.fingerprint == hashlib.sha256(table.read_bytes()).hexdigest()
    assert panel.levels == (4, 6)
    # 111111 and 111112 in three feature years, and 1111 itself in three
    assert panel.split_counts(6)['heldout_outer'] == 6
    assert panel.split_counts(4)['heldout_outer'] == 3

def test_from_sources_reads_no_groups_but_the_pinned_draw(tmp_path, regressor_cells):
    # An edited or redrawn table would get a new fingerprint, under which the log counts none of
    # the pinned draw's openings
    pinned = group_table_fingerprint(HELDOUT_GROUPS)

    with pytest.raises(ValueError, match='not the pinned draw'):
        _from_sources(tmp_path, regressor_cells, ['1111'], heldout_groups_sha256=pinned)
    assert not (tmp_path / 'selection_log.jsonl').exists()

def test_from_sources_refuses_data_the_branch_record_does_not_name(tmp_path, regressor_cells):
    record = {**BRANCH_RECORD, 'excluded_codes': []}

    with pytest.raises(ValueError, match='branch record mismatch'):
        _from_sources(tmp_path, regressor_cells, HELDOUT_GROUPS, branch_record=record)

def test_the_branch_record_must_name_this_panels_data(regressor_cells):
    six = population(level_cells(regressor_cells, CODEBOOK, 6))
    verify_branch_record(BRANCH_RECORD, CODEBOOK, six)

    for change, message in [
        ({
            'branch': 'B'
        }, 'branch A'),
        ({
            'time_respecting_outcome': False
        }, 'time-respecting'),
        ({
            'seen_regime': False
        }, 'seen-code regime'),
        ({
            'reference_years': [2021, 2022, 2023, 2024]
        }, 'reference years'),
        ({
            'ownership': '0'
        }, 'national private'),
        ({
            'excluded_codes': []
        }, 'six-digit population'),
        ({
            'population_heldout': 33
        }, 'populations'),
    ]:
        with pytest.raises(ValueError, match=message):
            verify_branch_record({**BRANCH_RECORD, **change}, CODEBOOK, six)

# -------------------------------------------------------------------------------------------------
# Sealing (Exit: neither regime's outer set can be read without a logged opening)
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('regime', list(Regime))
def test_neither_outer_set_is_read_without_a_logged_opening(panel, regressor_arm, log, regime):
    other = Regime.HELDOUT if regime is Regime.SEEN else Regime.SEEN

    with pytest.raises(SealedSplitError, match='sealed'):
        panel.test(regime, 6, regressor_arm, PURPOSE)
    assert log.records() == []

    panel.open_outer(regime, PURPOSE)

    with pytest.raises(SealedSplitError, match='sealed'):
        panel.test(other, 6, regressor_arm, PURPOSE)
    panel.test(regime, 6, regressor_arm, PURPOSE)
    assert [(r['event'], r['split']) for r in log.records()] == [('open', 'test'), ('read', 'test')]

def test_every_panel_object_opens_for_itself_and_a_reopening_needs_a_reason(
    regressor_rows, regressor_arm, log
):
    first = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)
    first.open_outer(Regime.SEEN, 'first opening')
    second = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)

    with pytest.raises(SealedSplitError):
        second.test(Regime.SEEN, 6, regressor_arm, PURPOSE)
    with pytest.raises(SplitAlreadyOpenedError, match='reopen_reason'):
        second.open_outer(Regime.SEEN, 'second opening')
    # The same check, alone, logs nothing
    with pytest.raises(SplitAlreadyOpenedError, match='reopen_reason'):
        second.require_openable(Regime.SEEN)
    second.require_openable(Regime.SEEN, reopen_reason='a fixture rerun')
    second.require_openable(Regime.HELDOUT)
    assert len(log.records()) == 1
    second.open_outer(Regime.SEEN, 'second opening', reopen_reason='a fixture rerun')
    second.open_outer(Regime.HELDOUT, 'first held-out opening')

    records = log.records()
    assert [(r['event'], r['panel']) for r in records] == [
        ('open', 'regressor_seen'),
        ('reopen', 'regressor_seen'),
        ('open', 'regressor_heldout'),
    ]
    assert records[1]['detail']['reason'] == 'a fixture rerun'
    # An opening counts the six-digit outer rows, and records every loaded level's
    assert records[0]['n_queries'] == 26
    assert records[0]['detail']['rows_by_level'] == {'2': 1, '3': 5, '4': 14, '5': 14, '6': 26}
    assert records[2]['n_queries'] == 18

def test_openings_are_counted_per_held_out_draw(regressor_rows, regressor_growth_rows, log):
    growth = RegressorOutcome.GROWTH
    RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS).open_outer(Regime.SEEN, 'first')

    RegressorPanel(regressor_rows, ['1111'], log, SETTINGS).open_outer(Regime.SEEN, 'other draw')
    third = RegressorPanel(regressor_growth_rows, ['1112'], log, SETTINGS, outcome=growth)
    third.open_outer(Regime.SEEN, 'a third draw')

    assert [r['event'] for r in log.records()] == ['open', 'open', 'open']

@pytest.mark.parametrize(
    ('first', 'second'),
    [
        (RegressorOutcome.LEVEL, RegressorOutcome.GROWTH),
        (RegressorOutcome.GROWTH, RegressorOutcome.LEVEL),
    ],
)
def test_an_opening_under_either_outcome_seals_the_split_for_both(
    regressor_rows, regressor_growth_rows, log, first, second
):
    # Req 22: 2025 employment underlies both outcomes' outer rows, so the split is the key
    rows = {RegressorOutcome.LEVEL: regressor_rows, RegressorOutcome.GROWTH: regressor_growth_rows}
    opener = RegressorPanel(rows[first], HELDOUT_GROUPS, log, SETTINGS, outcome=first)
    other = RegressorPanel(rows[second], HELDOUT_GROUPS, log, SETTINGS, outcome=second)
    opened = READ_NAMES[first][Regime.SEEN]

    opener.open_outer(Regime.SEEN, 'first opening')

    with pytest.raises(SplitAlreadyOpenedError, match=f'as {opened} for'):
        other.require_openable(Regime.SEEN)
    with pytest.raises(SplitAlreadyOpenedError, match='reopen_reason'):
        other.open_outer(Regime.SEEN, 'second opening')
    # Each regime's outer set is its own split
    other.require_openable(Regime.HELDOUT)
    other.open_outer(Regime.SEEN, 'second opening', reopen_reason='a fixture rerun')

    records = log.records()
    assert [(r['event'], r['panel']) for r in records] == [
        ('open', opened),
        ('reopen', READ_NAMES[second][Regime.SEEN]),
    ]
    assert records[1]['detail']['reason'] == 'a fixture rerun'
    assert {r['fingerprint'] for r in records} == {opener.fingerprint}

def test_the_data_fingerprint_names_the_rows_a_read_at_a_level_scores(panel, regressor_rows, log):
    moved = _shift_outcomes(regressor_rows, pl.col('group') == '1111')
    shifted = RegressorPanel(moved, HELDOUT_GROUPS, log, SETTINGS)
    reordered = RegressorPanel(
        {
            level: rows.reverse().select(rows.columns[::-1])
            for level, rows in regressor_rows.items()
        },
        HELDOUT_GROUPS,
        log,
        SETTINGS,
    )

    # One draw can hold other outcomes: only the data fingerprint tells the two apart
    assert shifted.fingerprint == panel.fingerprint
    assert shifted.data_fingerprint(6) != panel.data_fingerprint(6)
    # A read at level 6 scores no other level's rows, and neither row nor column order is data
    assert shifted.data_fingerprint(5) == panel.data_fingerprint(5)
    assert reordered.data_fingerprint(6) == panel.data_fingerprint(6)
    assert log.records() == []

def test_an_arm_missing_a_panel_code_is_refused_before_anything_is_logged(
    panel, regressor_arm, log
):
    codes = [code for code in CODEBOOK if code != POPULATION[0]]
    arm = ArmTables.from_tables(coordinate_table(codes), text_only_table(codes))

    with pytest.raises(ValueError, match='no coordinates'):
        panel.require_arm(arm)
    with pytest.raises(ValueError, match='no coordinates'):
        panel.validation(Regime.SEEN, 6, arm, PURPOSE)
    assert log.records() == []

    panel.open_outer(Regime.SEEN, PURPOSE)
    with pytest.raises(ValueError, match='no coordinates'):
        panel.test(Regime.SEEN, 6, arm, PURPOSE)
    # A refused read logs nothing, so the opening still serves a complete arm
    panel.require_arm(regressor_arm)
    panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE)
    assert [r['event'] for r in log.records()] == ['open', 'read']

# -------------------------------------------------------------------------------------------------
# Dating (Exit: every row's features are dated before its outcome, D7)
# -------------------------------------------------------------------------------------------------

def test_every_rows_features_are_dated_before_its_outcome(panel, regressor_rows, regressor_arm):
    panel.open_outer(Regime.SEEN, PURPOSE)
    panel.open_outer(Regime.HELDOUT, PURPOSE)

    predictions = pl.concat(
        [
            panel.validation(Regime.SEEN, 6, regressor_arm, PURPOSE),
            panel.validation(Regime.HELDOUT, 6, regressor_arm, PURPOSE),
            panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE),
            panel.test(Regime.HELDOUT, 6, regressor_arm, PURPOSE),
        ]
    )

    for frame in (*regressor_rows.values(), predictions):
        assert (frame.get_column('outcome_year') == frame.get_column('feature_year') + 1).all()

@pytest.mark.parametrize('split', ['validation', 'test'])
def test_the_seen_regime_fits_no_outcome_after_a_scored_rows_feature_year(regressor_rows, split):
    if split == 'validation':
        frame = _split(regressor_rows[6], RegressorSplit.REMAINDER)
        plan = seen_validation_plan(frame, 6, SETTINGS)
    else:
        frame = _split(regressor_rows[6], RegressorSplit.REMAINDER, RegressorSplit.SEEN_OUTER)
        plan = outer_plan(frame, Regime.SEEN, 6, SETTINGS)
    feature = frame.get_column('feature_year').to_list()
    outcome = frame.get_column('outcome_year').to_list()

    for task in plan:
        for fit, scored in ((task.fit, task.score), *task.tuning):
            assert max(outcome[row] for row in fit) <= min(feature[row] for row in scored)

# -------------------------------------------------------------------------------------------------
# Pairing (Req 5) and folds
# -------------------------------------------------------------------------------------------------

def test_folds_do_not_depend_on_the_arm(panel, regressor_arm):
    other = ArmTables.from_tables(
        coordinate_table(CODEBOOK, dimension=4, seed=99), text_only_table(CODEBOOK, seed=5)
    )
    keys = ['comparator', 'repeat', 'fold', 'code', 'feature_year']

    for regime in Regime:
        first = panel.validation(regime, 6, regressor_arm, PURPOSE)
        second = panel.validation(regime, 6, other, PURPOSE)

        assert first.select(keys).equals(second.select(keys))
        covariates = pl.col('comparator') == 'covariates'
        assert first.filter(covariates).equals(second.filter(covariates))

def test_group_folds_keep_a_groups_rows_together_whatever_the_row_order():
    groups = ['a', 'b', 'a', 'c', 'd', 'b', 'e']
    seed = (20260924, 0, 6, 0)

    folds = group_folds(groups, 2, seed)

    assert set(folds.tolist()) == {0, 1}
    assert all(len({f for g, f in zip(groups, folds) if g == group}) == 1 for group in groups)
    assert dict(zip(groups, folds)) == dict(zip(groups[::-1], group_folds(groups[::-1], 2, seed)))
    with pytest.raises(ValueError, match='cannot fill'):
        group_folds(['a', 'b'], 3, seed)

@pytest.mark.parametrize(
    'arguments',
    [
        {
            'alphas': ()
        },
        {
            'alphas': (1.0, 0.1)
        },
        {
            'alphas': (0.0, 1.0)
        },
        {
            'alphas': (1.0, 1.0)
        },
        {
            'alphas': (1.0, ),
            'folds': 1
        },
        {
            'alphas': (1.0, ),
            'repeats': 0
        },
        {
            'alphas': (1.0, ),
            'min_groups': 9
        },
    ],
)
def test_fit_settings_are_checked(arguments):
    with pytest.raises(ValueError):
        FitSettings(**arguments)

# -------------------------------------------------------------------------------------------------
# Levels
# -------------------------------------------------------------------------------------------------

def test_a_cell_without_enough_remainder_groups_is_undefined_not_scored(panel, regressor_arm, log):
    assert panel.cell_status(Regime.SEEN, 2) == '1 remainder groups, fewer than 4'
    assert panel.cell_status(Regime.HELDOUT, 3) == (
        'no four-digit parent: the held-out regime runs at levels 4-6'
    )
    assert panel.cell_status(Regime.SEEN, 3) is None
    assert panel.cell_status(Regime.HELDOUT, 4) is None

    with pytest.raises(ValueError, match='undefined at level 2'):
        panel.validation(Regime.SEEN, 2, regressor_arm, PURPOSE)
    assert log.records() == []

def test_the_multi_level_variant_scores_level_codes_outside_the_held_out_groups(
    panel, regressor_arm
):
    predictions = panel.validation(Regime.SEEN, 3, regressor_arm, PURPOSE)

    assert set(predictions.get_column('code')) <= {'112', '238', '311', '332', '522'}
    assert set(predictions.get_column('level')) == {3}

def test_each_comparator_has_its_own_columns(regressor_rows, regressor_arm):
    frame = regressor_rows[6]
    rows, codes = frame.height, frame.get_column('code').n_unique()

    assert feature_matrix('covariates', frame, 6, regressor_arm).shape == (rows, 2)
    assert feature_matrix('embedding', frame, 6, regressor_arm).shape == (rows, 3)
    assert feature_matrix('text_only', frame, 6, regressor_arm).shape == (rows, 3)
    assert feature_matrix('one_hot', frame, 6, regressor_arm).shape == (rows, codes)
    assert feature_matrix('covariates+one_hot', frame, 6, regressor_arm).shape == (rows, 2 + codes)
    # One indicator per ancestor level, 2 through 5
    assert (feature_matrix('ancestors', frame, 6, regressor_arm).sum(axis=1) == 4).all()
    with pytest.raises(ValueError, match='unknown comparator'):
        feature_matrix('covariates+nothing', frame, 6, regressor_arm)

# -------------------------------------------------------------------------------------------------
# An arm's tables (D9)
# -------------------------------------------------------------------------------------------------

def test_lorentz_points_are_refused_and_the_export_form_is_read():
    codes = ['11', '21', '22', '23']
    tangent = np.random.default_rng(0).normal(size=(4, 3))
    time = np.sqrt(1.0 + (tangent**2).sum(axis=1))
    lorentz = pl.DataFrame(
        {
            'code': codes,
            'x0': time,
            **{
                f'x{i + 1}': tangent[:, i]
                for i in range(3)
            }
        }
    )
    exported = pl.DataFrame(
        {
            'code': codes,
            'index': range(4),
            'level': [2] * 4,
            **{
                f'e{i}': tangent[:, i]
                for i in range(3)
            }
        }
    )

    with pytest.raises(ValueError, match='Lorentz points'):
        coordinate_matrix(lorentz)
    read_codes, matrix = coordinate_matrix(exported)
    assert read_codes == tuple(codes)
    np.testing.assert_array_equal(matrix, tangent)

def test_the_lorentz_refusal_names_the_export_form_alone():
    # tools diagnostics raises it too, so it names no panel
    tangent = np.random.default_rng(1).normal(size=(4, 3))
    time = np.sqrt(1.0 + (tangent**2).sum(axis=1))
    lorentz = pl.DataFrame(
        {
            'code': ['11', '21', '22', '23'],
            'x0': time,
            **{
                f'x{i + 1}': tangent[:, i]
                for i in range(3)
            }
        }
    )

    with pytest.raises(ValueError, match='hyperboloid, not the export form') as excinfo:
        coordinate_matrix(lorentz)

    assert 'regressor panel' not in str(excinfo.value)

def _float32_lorentz_export(radius, dimension, curvature, rng):
    '''Points at these radii, computed in float32 and stored in Float64 columns, as exported.'''

    direction = rng.normal(size=(radius.size, dimension))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    root = np.float32(np.sqrt(curvature))
    scaled = (np.sqrt(curvature) * radius[:, None]).astype(np.float32)
    points = np.hstack(
        [np.cosh(scaled) / root,
         np.sinh(scaled) / root * direction.astype(np.float32)]
    )
    return pl.DataFrame(
        {
            'index': range(radius.size),
            'level': [6] * radius.size,
            'code': [f'{index:06d}' for index in range(radius.size)],
            **{
                f'hyp_e{i}': points[:, i].astype(np.float64)
                for i in range(dimension + 1)
            },
        }
    )

@pytest.mark.parametrize('curvature', [0.5, 1.0, 2.0])
@pytest.mark.parametrize('dimension', [8, 16, 32])
@pytest.mark.parametrize(
    'layout',
    [
        [(2125, 0.0, 8.0)],
        # A few codes near the origin and the rest far out, where a median level would sit
        [(20, 0.3, 1.0), (2105, 8.0, 11.0)],
        # Every code far out, where rounding can push x0^2 - |x|^2 below zero
        [(2125, 9.0, 12.0)],
    ],
    ids=['spread', 'near-and-far', 'far'],
)
def test_a_float32_lorentz_export_is_refused_at_any_radius(layout, dimension, curvature):
    # Rounding error in x0^2 - |x|^2 grows with x0^2, so neither a tolerance fixed relative to
    # 1/c nor a level taken from far rows serves every radius
    for seed in range(3):
        rng = np.random.default_rng(seed)
        radius = np.concatenate([rng.uniform(low, high, size=rows) for rows, low, high in layout])

        with pytest.raises(ValueError, match='Lorentz points'):
            coordinate_matrix(_float32_lorentz_export(radius, dimension, curvature, rng))

@pytest.mark.parametrize(
    ('table', 'message'),
    [
        (pl.DataFrame({'e0': [1.0]}), 'no code column'),
        (pl.DataFrame({
            'code': ['11'],
            'index': [0]
        }), 'no coordinate columns'),
        (pl.DataFrame({
            'code': ['11', '11'],
            'e0': [1.0, 2.0]
        }), 'repeats a code'),
        (pl.DataFrame({
            'code': ['11', '21'],
            'e0': [1.0, float('nan')]
        }), 'not finite'),
        # A log map at the origin keeps a zero time column, which would count as a dimension
        (
            pl.DataFrame({
                'code': ['11', '21'],
                'e0': [0.0, 0.0],
                'e1': [1.0, 2.0]
            }), r"constant columns \['e0'\]"
        ),
    ],
)
def test_a_malformed_coordinate_table_is_refused(table, message):
    with pytest.raises(ValueError, match=message):
        coordinate_matrix(table)

def test_the_text_only_table_is_reduced_to_the_arms_dimension_code_by_code(regressor_arm):
    shuffled = text_only_table(CODEBOOK).reverse()

    arm = ArmTables.from_tables(coordinate_table(CODEBOOK), shuffled)

    assert regressor_arm.text_only.shape == (len(CODEBOOK), regressor_arm.dimension)
    np.testing.assert_allclose(arm.text_only, regressor_arm.text_only, atol=1e-10)
    assert arm.text_only_fingerprint == regressor_arm.text_only_fingerprint
    with pytest.raises(ValueError, match='different codes'):
        ArmTables.from_tables(coordinate_table(CODEBOOK), text_only_table(CODEBOOK[1:]))
    with pytest.raises(ValueError, match='no coordinates'):
        regressor_arm.lookup(['999999'])

# -------------------------------------------------------------------------------------------------
# From config, and the summary
# -------------------------------------------------------------------------------------------------

def test_the_config_builds_the_panel_over_a_codebook(tmp_path, regressor_cells):
    qcew = tmp_path / 'qcew'
    pins = write_qcew_slices(qcew, regressor_cells)
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    table = tmp_path / 'groups.csv'
    cfg = RegressorPanelConfig(
        qcew_dir=str(qcew),
        qcew_sha256=pins,
        codebook_codes_sha256=codes_fingerprint(CODEBOOK),
        heldout_groups_csv=str(table),
        heldout_groups_sha256=write_group_table(HELDOUT_GROUPS, table),
        selection_log=str(tmp_path / 'selection_log.jsonl'),
        alphas=list(SETTINGS.alphas),
        folds=2,
        repeats=2,
        inner_folds=2,
        min_groups=4,
        branch_record=RegressorBranchRecord(**BRANCH_RECORD),
    )

    panel = load_regressor_panel(cfg, codebook, levels=(6, ))
    growth = load_regressor_panel(
        cfg.model_copy(update={'outcome': 'growth'}), codebook, levels=(6, )
    )

    assert panel.settings == SETTINGS
    assert panel.levels == (6, )
    assert (panel.outcome, growth.outcome) == (RegressorOutcome.LEVEL, RegressorOutcome.GROWTH)
    assert growth.data_fingerprint(6) != panel.data_fingerprint(6)
    assert panel.log.path == tmp_path / 'selection_log.jsonl'
    other = load_regressor_panel(cfg, codebook, log_path=tmp_path / 'other.jsonl', levels=(6, ))
    assert other.log.path == tmp_path / 'other.jsonl'
    with pytest.raises(ValueError, match='no branch_record'):
        load_regressor_panel(cfg.model_copy(update={'branch_record': None}), codebook)
    with pytest.raises(ValueError, match='not the pinned draw'):
        load_regressor_panel(cfg.model_copy(update={'heldout_groups_sha256': '0' * 64}), codebook)

def test_the_summary_pools_rows_per_panel_split_level_and_comparator():
    predictions = pl.DataFrame(
        {
            'panel': ['regressor_seen'] * 4,
            'split': ['validation'] * 4,
            'level': [6] * 4,
            'comparator': ['covariates', 'covariates', 'one_hot', 'one_hot'],
            'outcome': [1.0, 3.0, 1.0, 3.0],
            'prediction': [1.0, 3.0, 2.0, 2.0],
            'alpha': [1.0, 1.0, 10.0, 100.0],
        }
    )

    first, second = summarize(predictions).iter_rows(named=True)

    assert (first['comparator'], first['rows'], first['rmse'], first['r2']) == (
        'covariates', 2, 0.0, 1.0
    )
    assert (second['rmse'], second['r2'], second['median_alpha']) == (1.0, 0.0, 55.0)

# -------------------------------------------------------------------------------------------------
# Levels unchanged (Req 21): level mode's values, pinned at d61f49b before the growth outcome
# -------------------------------------------------------------------------------------------------

# Captured at d61f49b on macOS arm64, where Python 3.12 (numpy 2.3.4) and Python 3.10 (numpy 2.2.6)
# agree bit for bit under polars 1.35.1
LEVEL_DRAW = 'b47d16873ea7065cdd9a1b9718268d9d5095dfc2b5879c3e966710e3bec9c78d'
# Each level's rows per split (remainder, seen outer set, held-out outer set), and the digest of
# every row's code, feature year, outcome year, group and split, sorted by code and feature year
LEVEL_PARTITIONS = {
    2: ((2, 1, 9), 'ae050a881bfaaeeea3588e545a50f6930afed9f912e808b2ff3868c1bfa41e97'),
    3: ((10, 5, 9), '2cf28b0b3ed39230fa250133b3269d2c081065a4e62b0d72519d973d21d34de7'),
    4: ((28, 14, 9), '40ee441f5f6fc808275c21555925f4247a4cabd90d0b2402b65b2b5e69afaa24'),
    5: ((28, 14, 9), '536780bc9117e2e5b7551d7f8169f655b5223d91fbb7ce5ab584428529e945be'),
    6: ((52, 26, 18), '1f719ca447c6ac66307846f5b869e396ffa04db423c355710288c5fae639d1f7'),
}
# The log of ``level_reads``: each defined cell's validation read, then each regime's opening and
# its outer-set reads
LEVEL_LOG = [
    ('read', 'regressor_seen', 'validation', 10),
    ('read', 'regressor_seen', 'validation', 28),
    ('read', 'regressor_seen', 'validation', 28),
    ('read', 'regressor_seen', 'validation', 52),
    ('read', 'regressor_heldout', 'validation', 28),
    ('read', 'regressor_heldout', 'validation', 28),
    ('read', 'regressor_heldout', 'validation', 52),
    ('open', 'regressor_seen', 'test', 26),
    ('read', 'regressor_seen', 'test', 5),
    ('read', 'regressor_seen', 'test', 14),
    ('read', 'regressor_seen', 'test', 14),
    ('read', 'regressor_seen', 'test', 26),
    ('open', 'regressor_heldout', 'test', 18),
    ('read', 'regressor_heldout', 'test', 9),
    ('read', 'regressor_heldout', 'test', 9),
    ('read', 'regressor_heldout', 'test', 18),
]
# Each read's digest of its key columns and chosen penalties, the sums of its predictions and of
# their squared errors, and the digest of every prediction column
LEVEL_READS = {
    (Regime.SEEN, 'validation', 3): (
        '0d3fe43ef5a84011a6c630421057bacfa43924b7c494e6c5cd6a8a4747467791',
        667.4451283013116,
        5.130531432910643,
        '650a98f3f27e0bd8ba0ab1302ff0dc3ca4b7c904b90a4abfbdb84d78f8ba2761',
    ),
    (Regime.SEEN, 'validation', 4): (
        '2a0db0140205fced1794f7e7d1fba820bc7e90833078443ba2e88a722d889f86',
        1771.3781051987503,
        79.44066289871006,
        '6fe8812179def263a314bc78e935eb9c73103981689d749db49d9cdb20ce42b5',
    ),
    (Regime.SEEN, 'validation', 5): (
        '348a229977df37a13fecec449db1c4fc59a9f3041e01de54395f47233c659705',
        1672.4597586951015,
        162.28358979019288,
        '178f31610ecabfca5a6acdfcb2f2041704d12cb0d2408a1ce9e5005114683b12',
    ),
    (Regime.SEEN, 'validation', 6): (
        'd6a1914d785a674c188d50e7c1eddd738bd079ee50e11ce15a83320dc54e1446',
        3127.5068050704963,
        131.82012311086612,
        'bd48f3a7b2a5e128eeb2cd3f2bc300013455ae238483c804731c532bf051adac',
    ),
    (Regime.HELDOUT, 'validation', 4): (
        'd4ef9be4090fa25a61cc0f756f1adddd2616a22830d483361c43d9e9894bd658',
        2762.234709277887,
        311.1588055545941,
        '3331930a7ef3034fa708b88e148125084527b552ec5a1277e43e8e5cd42f6b95',
    ),
    (Regime.HELDOUT, 'validation', 5): (
        '8a5720352cf749805294ab4dd431122f86c5e01c7fbe117b4d352451aaa2189f',
        2603.8437776456385,
        902.0747609652292,
        '21dd412769a514eda499d2413c0656ec608a3365c4ffe5ba80e409041098c734',
    ),
    (Regime.HELDOUT, 'validation', 6): (
        'fa2b27d4692a5fc28c294a71c58283f73f9eed0a7f626b4fccf14df7f5890d91',
        4890.906428128532,
        438.6849114831849,
        '6173effe96fa20085250cb6f42fb2647e76574cc68cf18401ab7c378d3c287fc',
    ),
    (Regime.SEEN, 'test', 3): (
        '858a694a6dd83a2763f5d1832d9e43d5fdf916a7026c858e1c91714d67bedfb3',
        335.91313347500005,
        1.141657267795871,
        'f1662bb791d10ef9f03d194defe2cc3efcd34e35567ddbda8915e9bdb6ae6871',
    ),
    (Regime.SEEN, 'test', 4): (
        '3aa1a3896e4548f68397f51f5da22fe5e9b913527af8c0fd02f302b7797e25d2',
        889.6516122009634,
        33.50275136228421,
        '5d7dd4b34c774e42b1de017a5135b05dc917e66a7641fd47973c6df0ecb5b3ca',
    ),
    (Regime.SEEN, 'test', 5): (
        'de3c54f5b8e38ec881a3f7f1f0d91434c77980a741cba24535169ab17b875c46',
        839.5593902615201,
        70.59059817108438,
        'cc3f2ed072f133b27fb22a8168d5f3a21bdc25a090c4198067ac6b72f0f66942',
    ),
    (Regime.SEEN, 'test', 6): (
        '4557c61fe0967915cddc93cd3fc2f3735c01d8ebf62b33445815c260ce50ec8b',
        1572.8628775819825,
        69.21606532040232,
        'c4b2af96f52d1192f35dcfa3860455d9be47bfe516c5407baf176560e3bcb8c5',
    ),
    (Regime.HELDOUT, 'test', 4): (
        '96a88a55e35dde451daa931a1c32be5fc46e0dd547fd74bbd5cc34da21cc5f30',
        437.0580054753184,
        116.30107865000055,
        'de2f101d3a16831cf7d3ea39a3ae31752fcd6f3abe85914bf57af52a51d3d0c5',
    ),
    (Regime.HELDOUT, 'test', 5): (
        'ae4b294c299114c6984dc2366ae0174841bf0456068ecb6953c145c190f95c33',
        384.49482348133023,
        165.4899488131396,
        '3b6a5b59e1e81e715ae3dd6db70c15a13a67c9eea98d10956129c44ccc1a4f10',
    ),
    (Regime.HELDOUT, 'test', 6): (
        '8b25d80ee6fe623a979501877b9ce3fa2229ee99baa36d6138141bf9a272a303',
        857.1212167419905,
        98.41731977089233,
        '497f48b7e809d2984b20762684e67b6f4c07a90f2ae3e174f5937ebc42b794e8',
    ),
}
# Level 6's validation folds: each regime's and repeat's groups in fold 0, then in fold 1
LEVEL_SIX_FOLDS = {
    (Regime.SEEN, 0): (
        ('1111', '1121', '1122', '2381', '3111', '3112', '5221'), (
            '1112', '1114', '3212', '3321', '3322', '5222', '5232'
        )
    ),
    (Regime.SEEN, 1): (
        ('1111', '1112', '1114', '2381', '3111', '3322', '5222'), (
            '1121', '1122', '3112', '3212', '3321', '5221', '5232'
        )
    ),
    (Regime.HELDOUT, 0): (
        ('1114', '2381', '3111', '3212', '3321', '3322', '5221'), (
            '1111', '1112', '1121', '1122', '3112', '5222', '5232'
        )
    ),
    (Regime.HELDOUT, 1): (
        ('1111', '1114', '1122', '2381', '3112', '3322', '5221'), (
            '1112', '1121', '3111', '3212', '3321', '5222', '5232'
        )
    ),
}
LEVEL_DATA_FINGERPRINTS = {
    2: 'eca3bfb9319d9c28ba0e48b8cca2279e56aac8711113833bb24d03be96f7881a',
    3: '1b3f9104506d9f6866131731d0f5d7f2c4422538fb0172eae80d12e3b2906e0d',
    4: 'cba83a33960191032af7cff326c02d37c17ec087cd8c206d722bd729a793160c',
    5: '221c729fbd73c45fe07a136cc8c5ca904a022d69390b6bc4dfed5c0a59ce9443',
    6: '9501d616de750922fb899f9310be2ab00587bc9487dc7542b3e1f77ada6821c7',
}
PARTITION_COLUMNS = ['code', 'feature_year', 'outcome_year', 'group', 'split']
KEY_COLUMNS = [name for name in PREDICTION_COLUMNS if name not in ('outcome', 'prediction')]
# Literal float pins hold where they were captured. Another libm can round a log a last bit
# differently: 7 of level 6's cell values lie within glibc's 0.519-ULP error bound of a rounding
# midpoint, the closest at 0.0019 ULP. Another BLAS can round a ridge fit differently. The d61f49b
# oracle and the float-free digests hold on every platform.
# The literals also depend on the lock's numpy and polars and the BLAS they call (Accelerate here).
# A CAPTURED_HERE pin failing while the d61f49b oracle and the digest pins pass means that
# environment changed, not the rows or the fit: review the change, then re-capture the literals.
CAPTURED_HERE = pytest.mark.skipif(
    (sys.platform, platform.machine()) != ('darwin', 'arm64'),
    reason='pinned where captured, on macOS arm64: another libm or BLAS can differ in a last bit',
)

def _digest(frame, columns):
    '''SHA-256 of the columns' values as JSON, row by row in the frame's order.'''

    return hashlib.sha256(json.dumps(frame.select(columns).rows()).encode('utf-8')).hexdigest()

@pytest.fixture(scope='module')
def level_reads(regressor_rows, regressor_arm, tmp_path_factory):
    '''
    Every read of the fixture panel in level mode: each defined cell's validation split, then each
    regime's outer set, opened once, at each defined level. Returns the predictions by (regime,
    split, level), and the log's records.
    '''

    log = SelectionLog(tmp_path_factory.mktemp('level') / 'selection_log.jsonl')
    panel = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)
    cells = [
        (regime, level) for regime in Regime for level in panel.levels
        if panel.cell_status(regime, level) is None
    ]
    reads = {
        (regime, 'validation', level): panel.validation(regime, level, regressor_arm, PURPOSE)
        for regime, level in cells
    }
    for regime in Regime:
        panel.open_outer(regime, PURPOSE)
        for level in [level for other, level in cells if other is regime]:
            reads[regime, 'test', level] = panel.test(regime, level, regressor_arm, PURPOSE)
    return reads, log.records()

def test_level_rows_are_d61f49bs_at_every_level(regressor_cells, regressor_rows):
    for level in LEVELS:
        cells = level_cells(regressor_cells, CODEBOOK, level)
        codes = population(cells)
        oracle = panel_rows_d61f49b(cells, codes)

        assert panel_rows(cells, codes).equals(oracle)
        assert regressor_rows[level].equals(oracle)

def test_level_partitions_and_data_fingerprints_are_pinned(panel, regressor_rows):
    assert panel.fingerprint == LEVEL_DRAW
    for level, (counts, digest) in LEVEL_PARTITIONS.items():
        frame = assign_splits(regressor_rows[level], HELDOUT_GROUPS).sort('code', 'feature_year')
        named = frame.select(sorted(frame.columns))
        payload = json.dumps({'columns': named.columns, 'rows': named.rows()}).encode('utf-8')

        assert tuple(panel.split_counts(level).values()) == counts
        assert _digest(frame, PARTITION_COLUMNS) == digest
        # The data fingerprint hashes every column of those rows, in name order
        assert panel.data_fingerprint(level) == hashlib.sha256(payload).hexdigest()

@CAPTURED_HERE
def test_level_data_fingerprints_are_the_captured_ones(panel):
    assert {level: panel.data_fingerprint(level) for level in LEVELS} == LEVEL_DATA_FINGERPRINTS

def test_level_reads_and_folds_are_pinned(level_reads):
    reads, records = level_reads

    assert [(r['event'], r['panel'], r['split'], r['n_queries']) for r in records] == LEVEL_LOG
    assert list(reads) == list(LEVEL_READS)
    for key, (digest, prediction_sum, error_sum, _) in LEVEL_READS.items():
        sums = reads[key].select(
            predictions=pl.col('prediction').sum(),
            squared_errors=((pl.col('prediction') - pl.col('outcome'))**2).sum(),
        ).row(0)
        assert _digest(reads[key], KEY_COLUMNS) == digest
        assert sums == pytest.approx((prediction_sum, error_sum), rel=1e-9)
    for (regime, repeat), folds in LEVEL_SIX_FOLDS.items():
        rows = reads[regime, 'validation', 6].filter(
            (pl.col('comparator') == 'covariates') & (pl.col('repeat') == repeat)
        )
        assigned = tuple(
            tuple(sorted(rows.filter(pl.col('fold') == fold).get_column('group').unique()))
            for fold in range(SETTINGS.folds)
        )
        assert assigned == folds

@CAPTURED_HERE
def test_level_predictions_are_the_captured_ones_bit_for_bit(level_reads):
    reads, _ = level_reads

    for key, (*_, digest) in LEVEL_READS.items():
        assert _digest(reads[key], list(PREDICTION_COLUMNS)) == digest

# -------------------------------------------------------------------------------------------------
# The growth outcome (Req 21): level mode's rows, partition, draw and folds, under its own names
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def growth_panel(regressor_growth_rows, log):
    return RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )

def test_each_outcome_reads_under_its_own_names():
    assert READ_NAMES == {
        RegressorOutcome.LEVEL: {
            Regime.SEEN: 'regressor_seen',
            Regime.HELDOUT: 'regressor_heldout'
        },
        RegressorOutcome.GROWTH: {
            Regime.SEEN: 'growth_seen',
            Regime.HELDOUT: 'growth_heldout'
        },
    }
    # The levels history's names, which D8's decision reads
    assert PANEL_NAMES == READ_NAMES[RegressorOutcome.LEVEL]

def test_growth_keeps_level_modes_partition_and_draw(
    panel, growth_panel, regressor_rows, regressor_growth_rows
):
    assert (panel.outcome, growth_panel.outcome) == (
        RegressorOutcome.LEVEL, RegressorOutcome.GROWTH
    )
    assert growth_panel.fingerprint == panel.fingerprint
    for level in LEVELS:
        level_frame = assign_splits(regressor_rows[level], HELDOUT_GROUPS)
        growth_frame = assign_splits(regressor_growth_rows[level], HELDOUT_GROUPS)

        assert growth_frame.drop('outcome').equals(level_frame.drop('outcome'))
        assert growth_panel.split_counts(level) == panel.split_counts(level)
        # Only the outcome differs, and the data fingerprint names it
        assert growth_panel.data_fingerprint(level) != panel.data_fingerprint(level)

def test_growth_reads_score_every_comparator_on_level_modes_folds_under_their_own_names(
    panel, growth_panel, regressor_arm, log
):
    keys = ['comparator', 'repeat', 'fold', 'code', 'group', 'feature_year', 'outcome_year']

    for regime in Regime:
        level = panel.validation(regime, 6, regressor_arm, PURPOSE)
        growth = growth_panel.validation(regime, 6, regressor_arm, PURPOSE)

        assert growth.columns == list(PREDICTION_COLUMNS)
        name = READ_NAMES[RegressorOutcome.GROWTH][regime]
        assert growth.get_column('panel').unique().to_list() == [name]
        assert growth.select(keys).equals(level.select(keys))
        assert growth.get_column('prediction').is_finite().all()

    records = log.records()
    assert [(r['panel'], r['split'], r['n_queries']) for r in records] == [
        ('regressor_seen', 'validation', 52),
        ('growth_seen', 'validation', 52),
        ('regressor_heldout', 'validation', 52),
        ('growth_heldout', 'validation', 52),
    ]
    # Under the draw's fingerprint, with the level read's detail
    assert {r['fingerprint'] for r in records} == {panel.fingerprint}
    assert records[1]['detail'] == records[0]['detail']

@pytest.mark.parametrize('regime', list(Regime))
def test_a_growth_read_scores_each_rows_log_employment_change(
    regressor_cells, growth_panel, regressor_arm, regime
):
    '''Every scored outcome is log E(t + 1) - log E(t), read from the cells of t and t + 1.'''

    employment = level_cells(regressor_cells, CODEBOOK, 6).select(
        'code', 'year', log_emp=pl.col('emp').cast(pl.Float64).log()
    )
    current = employment.rename({'year': 'feature_year', 'log_emp': 'current'})
    following = employment.rename({'year': 'outcome_year', 'log_emp': 'following'})
    validation = growth_panel.validation(regime, 6, regressor_arm, PURPOSE)
    growth_panel.open_outer(regime, PURPOSE)
    test = growth_panel.test(regime, 6, regressor_arm, PURPOSE)

    for read in (validation, test):
        scored = read.join(current, on=['code', 'feature_year'])
        scored = scored.join(following, on=['code', 'outcome_year'])
        assert scored.height == read.height > 0
        assert scored.get_column('outcome').equals(
            scored.get_column('following') - scored.get_column('current'), check_names=False
        )

def test_a_growth_opening_and_its_reads_are_logged_under_the_growth_names(
    growth_panel, regressor_arm, log
):
    growth_panel.open_outer(Regime.HELDOUT, PURPOSE)
    predictions = growth_panel.test(Regime.HELDOUT, 6, regressor_arm, PURPOSE)

    assert predictions.get_column('panel').unique().to_list() == ['growth_heldout']
    assert [(r['event'], r['panel'], r['split'], r['n_queries']) for r in log.records()] == [
        ('open', 'growth_heldout', 'test', 18),
        ('read', 'growth_heldout', 'test', 18),
    ]
    with pytest.raises(SealedSplitError, match='growth_seen outer set is sealed'):
        growth_panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE)

def test_from_sources_builds_the_rows_of_its_outcome(
    tmp_path, regressor_cells, regressor_growth_rows
):
    built, _ = _from_sources(
        tmp_path, regressor_cells, HELDOUT_GROUPS, outcome=RegressorOutcome.GROWTH
    )

    expected = RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, built.log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )
    assert built.outcome is RegressorOutcome.GROWTH
    for level in built.levels:
        assert built.data_fingerprint(level) == expected.data_fingerprint(level)

# -------------------------------------------------------------------------------------------------
# The reference-only read (Req 24): the margin read reads no arm's tables
# -------------------------------------------------------------------------------------------------

def test_the_reference_comparators_are_those_that_read_no_arms_tables():
    assert reference_comparators(Regime.SEEN, 6) == (
        'covariates', 'one_hot', 'covariates+one_hot', 'ancestors', 'covariates+ancestors'
    )
    assert reference_comparators(Regime.HELDOUT, 6) == (
        'covariates', 'ancestors', 'covariates+ancestors'
    )
    assert reference_comparators(Regime.SEEN, 2) == ('covariates', 'one_hot', 'covariates+one_hot')

@pytest.mark.parametrize('regime', list(Regime))
def test_the_reference_read_logs_and_scores_only_comparators_that_read_no_arm(
    growth_panel, log, regime
):
    predictions = growth_panel.reference_validation(regime, 6, PURPOSE, detail={'run': 'margins'})

    names = reference_comparators(regime, 6)
    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert tuple(predictions.get_column('comparator').unique(maintain_order=True)) == names
    (record, ) = log.records()
    name = READ_NAMES[RegressorOutcome.GROWTH][regime]
    assert (record['event'], record['panel'], record['split']) == ('read', name, 'validation')
    assert record['fingerprint'] == growth_panel.fingerprint
    # No arm, text-only table or dimension: the read names no candidate
    assert record['detail'] == {'level': 6, 'comparators': list(names), 'run': 'margins'}

@pytest.mark.parametrize('regime', list(Regime))
def test_each_reference_comparator_predicts_as_in_any_arms_read(
    growth_panel, regressor_arm, regime
):
    '''Req 24: the reference reads no candidate, so its errors pair exactly with each one's.'''

    reference = growth_panel.reference_validation(regime, 6, PURPOSE)
    read = growth_panel.validation(regime, 6, regressor_arm, PURPOSE)

    names = list(reference_comparators(regime, 6))
    assert reference.equals(read.filter(pl.col('comparator').is_in(names)))

def test_a_comparator_that_reads_an_arms_table_needs_an_arm(regressor_rows):
    for comparator in ('embedding', 'covariates+text_only'):
        with pytest.raises(ValueError, match=f"{re.escape(comparator)}' reads an arm's table"):
            feature_matrix(comparator, regressor_rows[6], 6, None)
