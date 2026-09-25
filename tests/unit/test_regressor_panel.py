'''
The regressor panel (Req 2; Req 4; Req 5; roadmap Stage 3 Exit; D1, D7, D8, D9).

Synthetic QCEW rows (``tests/fixtures/regressor_panel.py``) and a stub arm stand in for the real
data and a trained arm. Every outer set opened here is a fixture's: no test reads a real one.
'''

import hashlib

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels import regressor, regressor_splits
from naics_embedder.panels.outcome import SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.qcew_rows import level_cells, population
from naics_embedder.panels.regressor import (
    OUTER_SPLITS,
    PREDICTION_COLUMNS,
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
    seen_validation_plan,
    summarize,
    verify_branch_record,
)
from naics_embedder.panels.regressor_splits import (
    RegressorSplit,
    assign_splits,
    codes_fingerprint,
    write_group_table,
)
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.utils.config import RegressorBranchRecord, RegressorPanelConfig
from tests.fixtures.regressor_panel import (
    BRANCH_RECORD,
    CODEBOOK,
    HELDOUT_GROUPS,
    POPULATION,
    SETTINGS,
    coordinate_table,
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

# -------------------------------------------------------------------------------------------------
# The committed draw (Exit: the panel reads the committed outer groups, never a fresh draw)
# -------------------------------------------------------------------------------------------------

def _from_sources(tmp_path, cells, groups, **overrides):
    qcew = tmp_path / 'qcew'
    pins = write_qcew_slices(qcew, cells)
    table = tmp_path / 'regressor_heldout_groups.csv'
    write_group_table(groups, table)
    arguments = {
        'qcew_dir': qcew,
        'qcew_sha256': pins,
        'codebook_codes': CODEBOOK,
        'heldout_groups_csv': table,
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

def test_openings_are_counted_per_held_out_draw(regressor_rows, log):
    RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS).open_outer(Regime.SEEN, 'first')

    RegressorPanel(regressor_rows, ['1111'], log, SETTINGS).open_outer(Regime.SEEN, 'other draw')

    assert [r['event'] for r in log.records()] == ['open', 'open']

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

@pytest.mark.parametrize('curvature', [0.5, 1.0, 2.0])
def test_a_float32_lorentz_export_is_refused_at_any_radius(curvature):
    # The train export writes float32 points into Float64 columns. Rounding error in
    # x0^2 - |x|^2 grows with x0^2, so a tolerance fixed relative to 1/c passes far points
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(2125, 16))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    root = np.float32(np.sqrt(curvature))
    scaled = (np.sqrt(curvature) * rng.uniform(0.0, 8.0, size=(2125, 1))).astype(np.float32)
    points = np.hstack(
        [np.cosh(scaled) / root,
         np.sinh(scaled) / root * direction.astype(np.float32)]
    )
    exported = pl.DataFrame(
        {
            'index': range(2125),
            'level': [6] * 2125,
            'code': [f'{index:06d}' for index in range(2125)],
            **{
                f'hyp_e{i}': points[:, i].astype(np.float64)
                for i in range(17)
            },
        }
    )

    with pytest.raises(ValueError, match='Lorentz points'):
        coordinate_matrix(exported)

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
    write_group_table(HELDOUT_GROUPS, table)
    cfg = RegressorPanelConfig(
        qcew_dir=str(qcew),
        qcew_sha256=pins,
        codebook_codes_sha256=codes_fingerprint(CODEBOOK),
        heldout_groups_csv=str(table),
        selection_log=str(tmp_path / 'selection_log.jsonl'),
        alphas=list(SETTINGS.alphas),
        folds=2,
        repeats=2,
        inner_folds=2,
        min_groups=4,
        branch_record=RegressorBranchRecord(**BRANCH_RECORD),
    )

    panel = load_regressor_panel(cfg, codebook, levels=(6, ))

    assert panel.settings == SETTINGS
    assert panel.levels == (6, )
    assert panel.log.path == tmp_path / 'selection_log.jsonl'
    other = load_regressor_panel(cfg, codebook, log_path=tmp_path / 'other.jsonl', levels=(6, ))
    assert other.log.path == tmp_path / 'other.jsonl'
    with pytest.raises(ValueError, match='no branch_record'):
        load_regressor_panel(cfg.model_copy(update={'branch_record': None}), codebook)

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
