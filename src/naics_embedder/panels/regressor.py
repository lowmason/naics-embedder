'''
The regressor panel (Req 2; Req 1, regressor estimand; Req 4; roadmap Stage 3; D1, D7, D8, D9).

An arm's coordinates enter a ridge regression as regressors, and the panel returns out-of-sample
predictions per row, keyed by code, year and group, so that Stage 4 can compute whichever
statistic it settles on and resample by four-digit group.

- **Rows (D7).** A code in a feature year t (2022–2024): covariates from year t, outcome log
  employment in t + 1 (``qcew_rows``).
- **Regimes (D8: each is a panel).** ``seen``: every scored code has earlier rows in the fit set;
  one-hot is a real competitor. ``heldout``: every row of a held-out four-digit group leaves the
  fit set; it runs at levels 4–6, where four-digit parents exist.
- **Partition.** One partition serves both regimes (``regressor_splits``): validation reads only
  the remainder, feature years 2022 and 2023 of codes outside the held-out groups.
- **Comparators (Req 2).** Covariates alone (log establishments and log wages, D1), and each
  representation alone and with the covariates: the arm's coordinates, six-digit one-hot (level-L
  one-hot in the multi-level variant; seen regime only), ancestor indicators at levels 2 to L − 1,
  and the text-only table reduced by PCA to the arm's dimension (D9).
- **Fitting.** Ridge on standardized features, the penalty tuned by nested cross-validation
  inside the remainder only (``FitTask``). Fold assignments depend on the regime, level, repeat
  and group ids alone, never on the arm, so every arm is scored on the same folds (Req 5).
- **Sealing.** Each regime's outer set is read only after a logged opening by the same panel
  object; a second opening of the same split (the same held-out draw, by fingerprint) needs a
  stated reason. Every read goes to the selection log, whose ``n_queries`` field counts rows here.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import polars as pl

from naics_embedder.panels.outcome import SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.qcew_rows import (
    PRIVATE,
    WINDOW_YEARS,
    level_cells,
    load_national_cells,
    panel_rows,
    population,
)
from naics_embedder.panels.regressor_splits import (
    GROUP_LEVEL,
    REMAINDER_FEATURE_YEARS,
    SECTOR_LEVEL,
    RegressorSplit,
    ancestor_at,
    assign_splits,
    check_partition,
    group_table_fingerprint,
    read_codebook_codes,
    read_group_table,
    split_counts,
)
from naics_embedder.panels.ridge import (
    best_alpha_index,
    squared_errors,
    standardized_ridge_path,
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.panels.text_only import TEXT_ONLY_PREFIX, pca_reduce
from naics_embedder.utils.config import RegressorBranchRecord, RegressorPanelConfig

class Regime(str, Enum):
    '''The two regressor regimes, each its own panel under Req 5 (D8).'''

    SEEN = 'seen'
    HELDOUT = 'heldout'

PANEL_NAMES = {Regime.SEEN: 'regressor_seen', Regime.HELDOUT: 'regressor_heldout'}
OUTER_SPLITS = {
    Regime.SEEN: RegressorSplit.SEEN_OUTER,
    Regime.HELDOUT: RegressorSplit.HELDOUT_OUTER
}
REGIME_STREAMS = {Regime.SEEN: 0, Regime.HELDOUT: 1}
# The test fit's inner folds draw from their own stream; validation repeats never reach it
TEST_STREAM = 1000
LEVELS = (2, 3, 4, 5, 6)
# D8's two regressor panels are the six-digit regimes; levels 2–5 are the multi-level variant
DECISION_LEVEL = 6
VALIDATION = 'validation'
TEST = 'test'

COVARIATES = 'covariates'
REPRESENTATIONS = ('embedding', 'one_hot', 'ancestors', 'text_only')
COVARIATE_COLUMNS = ('log_estabs', 'log_wages')
METADATA_COLUMNS = ('code', 'index', 'level')
PREDICTION_COLUMNS = (
    'panel',
    'split',
    'level',
    'comparator',
    'repeat',
    'fold',
    'code',
    'group',
    'feature_year',
    'outcome_year',
    'alpha',
    'outcome',
    'prediction',
)

# -------------------------------------------------------------------------------------------------
# Settings, comparators and folds
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FitSettings:
    '''
    How the panel tunes and resamples.

    Args:
        alphas: Ridge penalties, ascending.
        folds: Validation folds per repeat (grouped).
        repeats: Validation repeats, each with its own fold assignment.
        inner_folds: Tuning folds inside a held-out fit set (grouped).
        fold_seed: Base seed of every fold assignment.
        min_groups: Fewest remainder groups a regime needs at a level; fewer is undefined.
    '''

    alphas: Tuple[float, ...]
    folds: int = 5
    repeats: int = 5
    inner_folds: int = 5
    fold_seed: int = 20260924
    min_groups: int = 10

    def __post_init__(self) -> None:
        grid = list(self.alphas)
        if not grid or any(alpha <= 0 for alpha in grid) or grid != sorted(set(grid)):
            raise ValueError('alphas must be distinct positive penalties in ascending order')
        if self.folds < 2 or self.inner_folds < 2 or self.repeats < 1:
            raise ValueError('folds and inner_folds must be at least 2, and repeats at least 1')
        if self.min_groups < 2 * max(self.folds, self.inner_folds):
            raise ValueError('min_groups must be at least twice the larger fold count')

def comparators(regime: Regime, level: int) -> Tuple[str, ...]:
    '''
    Every comparator the regime scores at a level, covariates alone first.

    One-hot runs only in the seen regime, where it can predict more than the intercept; ancestor
    indicators need a level above 2.
    '''

    names = [COVARIATES]
    for representation in REPRESENTATIONS:
        if representation == 'one_hot' and regime is Regime.HELDOUT:
            continue
        if representation == 'ancestors' and level <= SECTOR_LEVEL:
            continue
        names.extend([representation, f'{COVARIATES}+{representation}'])
    return tuple(names)

def group_folds(groups: Sequence[str], n_folds: int, seed: Sequence[int]) -> np.ndarray:
    '''
    A fold per row: the distinct groups, sorted, permuted by ``np.random.default_rng(seed)`` and
    dealt into ``n_folds`` folds in turn, so a group's rows share a fold.

    Raises:
        ValueError: If there are fewer groups than folds.
    '''

    distinct = sorted(set(groups))
    if len(distinct) < n_folds:
        raise ValueError(f'{len(distinct)} groups cannot fill {n_folds} folds')
    order = np.random.default_rng(list(seed)).permutation(len(distinct))
    fold_of = {distinct[index]: position % n_folds for position, index in enumerate(order)}
    return np.array([fold_of[group] for group in groups], dtype=np.int64)

# -------------------------------------------------------------------------------------------------
# Plans: which rows fit, which are scored, which tune the penalty
# -------------------------------------------------------------------------------------------------

Rows = Tuple[int, ...]

@dataclass(frozen=True)
class FitTask:
    '''
    One out-of-sample prediction: fit on ``fit`` and predict ``score`` at the penalty with the
    least summed squared error over the ``tuning`` pairs (fit rows, scored rows).

    Row numbers index the frame the plan was built for; ``fold`` is -1 for the outer set.
    '''

    repeat: int
    fold: int
    fit: Rows
    score: Rows
    tuning: Tuple[Tuple[Rows, Rows], ...]

# One read's fit tasks: every repeat and fold of a validation read, or the outer set's single task
Plan = List[FitTask]

def _rows(mask: np.ndarray) -> Rows:
    return tuple(int(row) for row in np.flatnonzero(mask))

def seen_validation_plan(frame: pl.DataFrame, level: int, settings: FitSettings) -> Plan:
    '''
    Fit on the remainder's 2022 rows and predict its 2023 rows, forward in time (D7).

    The 2023 rows fall into grouped folds; each fold's penalty is chosen on the other folds'
    2023 rows, so no scored row tunes its own penalty. Repeats redraw the folds.
    '''

    years = frame.get_column('feature_year').to_numpy()
    groups = frame.get_column('group').to_list()
    fit = _rows(years == REMAINDER_FEATURE_YEARS[0])
    scored = np.flatnonzero(years == REMAINDER_FEATURE_YEARS[1])
    tasks = []
    for repeat in range(settings.repeats):
        seed = (settings.fold_seed, REGIME_STREAMS[Regime.SEEN], level, repeat)
        folds = group_folds([groups[row] for row in scored], settings.folds, seed)
        for fold in range(settings.folds):
            inside = tuple(int(row) for row in scored[folds == fold])
            outside = tuple(int(row) for row in scored[folds != fold])
            tasks.append(FitTask(repeat, fold, fit, inside, ((fit, outside), )))
    return tasks

def _inner_tuning(
    fit: np.ndarray, groups: Sequence[str], settings: FitSettings, seed: Tuple[int, ...]
) -> Tuple[Tuple[Rows, Rows], ...]:
    inner = group_folds([groups[row] for row in fit], settings.inner_folds, seed)
    return tuple(
        (
            tuple(int(row)
                  for row in fit[inner != fold]), tuple(int(row) for row in fit[inner == fold])
        ) for fold in range(settings.inner_folds)
    )

def heldout_validation_plan(frame: pl.DataFrame, level: int, settings: FitSettings) -> Plan:
    '''
    Repeated grouped folds over the remainder; each fold's penalty is tuned by grouped folds
    inside the rest of the remainder (nested), so a scored group never tunes its own penalty.
    '''

    groups = frame.get_column('group').to_list()
    rows = np.arange(frame.height)
    stream = REGIME_STREAMS[Regime.HELDOUT]
    tasks = []
    for repeat in range(settings.repeats):
        folds = group_folds(groups, settings.folds, (settings.fold_seed, stream, level, repeat))
        for fold in range(settings.folds):
            fit = rows[folds != fold]
            seed = (settings.fold_seed, stream, level, repeat, fold + 1)
            tuning = _inner_tuning(fit, groups, settings, seed)
            tasks.append(FitTask(repeat, fold, _rows(folds != fold), _rows(folds == fold), tuning))
    return tasks

def outer_plan(frame: pl.DataFrame, regime: Regime, level: int, settings: FitSettings) -> Plan:
    '''
    Fit on the whole remainder and predict the regime's outer set once.

    The penalty is tuned inside the remainder only: in the seen regime on the forward split
    (2022 rows fit, 2023 rows scored), in the held-out regime by grouped folds.
    '''

    split = frame.get_column('split').to_numpy()
    remainder = split == RegressorSplit.REMAINDER.value
    outer = split == OUTER_SPLITS[regime].value
    if regime is Regime.SEEN:
        years = frame.get_column('feature_year').to_numpy()
        tuning = (
            (
                _rows(remainder & (years == REMAINDER_FEATURE_YEARS[0])),
                _rows(remainder & (years == REMAINDER_FEATURE_YEARS[1])),
            ),
        )
    else:
        seed = (settings.fold_seed, REGIME_STREAMS[regime], level, TEST_STREAM)
        tuning = _inner_tuning(
            np.flatnonzero(remainder),
            frame.get_column('group').to_list(), settings, seed
        )
    return [FitTask(0, -1, _rows(remainder), _rows(outer), tuning)]

# -------------------------------------------------------------------------------------------------
# An arm's representations and the feature matrices
# -------------------------------------------------------------------------------------------------

def looks_lorentz(matrix: np.ndarray, rtol: float = 1e-3) -> bool:
    '''
    Whether every row lies on one hyperboloid ``x0^2 - |x|^2 = 1/c`` with ``x0 > 0``.

    A float32 export's rounding error in ``x0^2 - |x|^2`` grows with ``x0^2``, so each row's
    tolerance scales with its ``x0^2``, and the level ``1/c`` is read from the row nearest the
    origin, whose error is least; a level set by far rows (their median, say) would fail the
    near ones. The level's sign is not checked: when even the nearest row is far out, rounding
    can push its value below zero.
    '''

    if matrix.shape[1] < 2:
        return False
    time = matrix[:, 0]
    if (time <= 0).any():
        return False
    norm = time**2 - (matrix[:, 1:]**2).sum(axis=1)
    level = norm[np.argmin(time)]
    return bool((np.abs(norm - level) <= rtol * time**2).all())

def coordinate_matrix(table: pl.DataFrame) -> Tuple[Tuple[str, ...], np.ndarray]:
    '''
    The codes and coordinates of an arm's table in the export form Req 2 names.

    Every column other than ``code`` and the export's ``index`` and ``level`` metadata is a
    coordinate.

    Raises:
        ValueError: If codes repeat, a coordinate is not finite, or the rows lie on a hyperboloid:
            Lorentz points (the train prompt's ``hyp_e*`` export) are not the export form, which
            is tangent coordinates at the origin for a hyperbolic arm (Stage 6's export). Also if
            a column is constant: it carries nothing and would count toward the arm's dimension,
            which the text-only comparator is reduced to (D9). A log map at the origin keeps such
            a column, its zero time coordinate.
    '''

    if 'code' not in table.columns:
        raise ValueError('the coordinate table has no code column')
    columns = [name for name in table.columns if name not in METADATA_COLUMNS]
    if not columns:
        raise ValueError('the coordinate table has no coordinate columns')
    codes = tuple(table.get_column('code').cast(pl.Utf8).to_list())
    if len(set(codes)) != len(codes):
        raise ValueError('the coordinate table repeats a code')
    matrix = np.array(table.select(columns).to_numpy(), dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('the coordinate table has a coordinate that is not finite')
    if looks_lorentz(matrix):
        raise ValueError(
            'the coordinate table holds Lorentz points on a hyperboloid; the regressor panel '
            'takes the export form (tangent coordinates at the origin for a hyperbolic arm)'
        )
    constant = [name for name, spread in zip(columns, np.ptp(matrix, axis=0)) if spread == 0]
    if constant:
        raise ValueError(
            f'the coordinate table has constant columns {constant}: they would count toward '
            "the arm's dimension; export the tangent coordinates without a log map's zero time "
            'coordinate'
        )
    return codes, matrix

def matrix_fingerprint(codes: Sequence[str], matrix: np.ndarray) -> str:
    '''SHA-256 of the codes and their float64 values, in code order.'''

    order = np.argsort(np.asarray(codes))
    digest = hashlib.sha256('\n'.join(codes[index] for index in order).encode('utf-8'))
    digest.update(np.ascontiguousarray(matrix[order], dtype=np.float64).tobytes())
    return digest.hexdigest()

@dataclass(frozen=True)
class ArmTables:
    '''One arm's coordinates and the text-only table reduced to the arm's dimension (D9).'''

    codes: Tuple[str, ...]
    coordinates: np.ndarray
    text_only: np.ndarray
    fingerprint: str
    text_only_fingerprint: str

    @classmethod
    def from_tables(cls, coordinates: pl.DataFrame, text_only: pl.DataFrame) -> 'ArmTables':
        '''
        Pair an arm's coordinate table with the text-only table of the same codes.

        Raises:
            ValueError: If the two tables cover different codes.
        '''

        codes, matrix = coordinate_matrix(coordinates)
        text_columns = [name for name in text_only.columns if name.startswith(TEXT_ONLY_PREFIX)]
        text_codes = text_only.get_column('code').to_list()
        if set(text_codes) != set(codes) or len(text_codes) != len(codes):
            raise ValueError('the coordinate and text-only tables cover different codes')
        text_matrix = np.array(text_only.select(text_columns).to_numpy(), dtype=np.float64)
        reduced = pca_reduce(text_matrix, matrix.shape[1])
        position = {code: row for row, code in enumerate(text_codes)}
        aligned = reduced[[position[code] for code in codes]]
        return cls(
            codes=codes,
            coordinates=matrix,
            text_only=aligned,
            fingerprint=matrix_fingerprint(codes, matrix),
            text_only_fingerprint=matrix_fingerprint(text_codes, text_matrix),
        )

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[1])

    def lookup(self, codes: Sequence[str]) -> np.ndarray:
        '''Row numbers of the codes in this arm's tables.'''

        position = {code: row for row, code in enumerate(self.codes)}
        missing = sorted(set(codes) - set(position))
        if missing:
            raise ValueError(f'the arm has no coordinates for {len(missing)} codes: {missing[:5]}')
        return np.array([position[code] for code in codes], dtype=np.int64)

def _require_codes(arm: ArmTables, frame: pl.DataFrame) -> None:
    '''Raise before a read is logged if the arm lacks a code of ``frame`` (``ArmTables.lookup``).'''

    arm.lookup(frame.get_column('code').unique().to_list())

def indicators(values: Sequence[str]) -> np.ndarray:
    '''One column per distinct value (sorted), one 1 per row.'''

    categories = sorted(set(values))
    column = {value: index for index, value in enumerate(categories)}
    matrix = np.zeros((len(values), len(categories)), dtype=np.float64)
    matrix[np.arange(len(values)), [column[value] for value in values]] = 1.0
    return matrix

def feature_matrix(comparator: str, frame: pl.DataFrame, level: int, arm: ArmTables) -> np.ndarray:
    '''
    The comparator's features for every row of ``frame``.

    Indicator columns come from the codes in ``frame``, never from outcomes.
    '''

    codes = frame.get_column('code').to_list()
    blocks: List[np.ndarray] = []
    for part in comparator.split('+'):
        if part == COVARIATES:
            blocks.append(np.array(frame.select(COVARIATE_COLUMNS).to_numpy(), dtype=np.float64))
        elif part == 'embedding':
            blocks.append(arm.coordinates[arm.lookup(codes)])
        elif part == 'text_only':
            blocks.append(arm.text_only[arm.lookup(codes)])
        elif part == 'one_hot':
            blocks.append(indicators(codes))
        elif part == 'ancestors':
            blocks.extend(
                indicators([ancestor_at(code, ancestor) for code in codes])
                for ancestor in range(SECTOR_LEVEL, level)
            )
        else:
            raise ValueError(f'unknown comparator part {part!r}')
    return np.hstack(blocks)

def verify_branch_record(
    record: Mapping[str, Any],
    codebook_codes: Collection[str],
    six_digit_population: Collection[str],
) -> None:
    '''
    Require the panel's data to be what Stage 1's finding dictates.

    Args:
        record: The branch record (``RegressorBranchRecord`` fields).
        codebook_codes: The codebook's codes.
        six_digit_population: The six-digit codes with a usable cell in every window year.

    Raises:
        ValueError: If the record names another branch, years, ownership or grain than this
            panel reads, or the population is not the codebook's six-digit codes without the
            record's excluded codes.
    '''

    problems = []
    if record['branch'] != 'A' or not record['time_respecting_outcome']:
        problems.append('the panel implements branch A, with a time-respecting outcome')
    if not record['seen_regime']:
        problems.append('the panel runs the seen-code regime')
    if tuple(record['reference_years']) != WINDOW_YEARS:
        problems.append(f'reference years {record["reference_years"]} are not {WINDOW_YEARS}')
    if record['ownership'] != PRIVATE or record['grain'] != 'national':
        problems.append('the panel reads national private ownership')
    six_digit = set(code for code in codebook_codes if len(code) == 6)
    expected = sorted(six_digit - set(record['excluded_codes']))
    if sorted(six_digit_population) != expected:
        problems.append(
            f'the six-digit population has {len(six_digit_population):,} codes, not the '
            f'{len(expected):,} codebook codes outside the excluded list'
        )
    counts = {record['population_seen'], record['population_heldout']}
    if counts != {len(six_digit_population)}:
        problems.append(f'the record names populations {sorted(counts)}')
    if problems:
        raise ValueError('branch record mismatch: ' + '; '.join(problems))

def run_plan(tasks: Sequence[FitTask], x: np.ndarray, y: np.ndarray,
             alphas: Sequence[float]) -> List[Tuple[FitTask, float, np.ndarray]]:
    '''Each task's chosen penalty and its predictions for the scored rows.'''

    results = []
    for task in tasks:
        errors = np.zeros(len(alphas))
        for fit, scored in task.tuning:
            fit_rows, scored_rows = list(fit), list(scored)
            path = standardized_ridge_path(x[fit_rows], y[fit_rows], x[scored_rows], alphas)
            errors += squared_errors(y[scored_rows], path)
        alpha = float(alphas[best_alpha_index(errors)])
        fit_rows, scored_rows = list(task.fit), list(task.score)
        predictions = standardized_ridge_path(x[fit_rows], y[fit_rows], x[scored_rows], [alpha])
        results.append((task, alpha, predictions[:, 0]))
    return results

# -------------------------------------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------------------------------------

class RegressorPanel:
    '''
    The panel rows at each level, the committed held-out draw, and the log every read goes to.

    Args:
        rows_by_level: Panel rows per level (``qcew_rows.panel_rows``).
        heldout_groups: The committed held-out four-digit groups.
        log: The selection log.
        settings: Tuning and resampling settings.
    '''

    def __init__(
        self,
        rows_by_level: Mapping[int, pl.DataFrame],
        heldout_groups: Collection[str],
        log: SelectionLog,
        settings: FitSettings,
    ):
        self._rows: Dict[int, pl.DataFrame] = {}
        for level, rows in rows_by_level.items():
            if level not in LEVELS:
                raise ValueError(f'level must be one of {LEVELS}, got {level}')
            frame = assign_splits(rows, heldout_groups)
            check_partition(frame, rows.get_column('code').unique().to_list())
            self._rows[level] = frame
        self.heldout_groups: Tuple[str, ...] = tuple(sorted(heldout_groups))
        self.fingerprint = group_table_fingerprint(self.heldout_groups)
        self.log = log
        self.settings = settings
        self._open: Set[Regime] = set()

    @classmethod
    def from_sources(
        cls,
        *,
        qcew_dir: Union[str, Path],
        qcew_sha256: Mapping[str, str],
        codebook_codes: Sequence[str],
        heldout_groups_csv: Union[str, Path],
        log_path: Union[str, Path],
        settings: FitSettings,
        branch_record: Mapping[str, Any],
        levels: Sequence[int] = LEVELS,
    ) -> 'RegressorPanel':
        '''
        The panel from the pinned QCEW slices, a codebook and the committed held-out groups.

        Raises:
            ValueError: If the data are not the population ``branch_record`` names
                (``verify_branch_record``).
        '''

        cells = load_national_cells(Path(qcew_dir), qcew_sha256)
        verify_branch_record(
            branch_record, codebook_codes,
            population(level_cells(cells, codebook_codes, DECISION_LEVEL))
        )
        rows_by_level = {}
        for level in levels:
            cells_at_level = level_cells(cells, codebook_codes, level)
            rows_by_level[level] = panel_rows(cells_at_level, population(cells_at_level))
        return cls(
            rows_by_level,
            read_group_table(Path(heldout_groups_csv)),
            SelectionLog(Path(log_path)),
            settings,
        )

    @property
    def levels(self) -> Tuple[int, ...]:
        return tuple(sorted(self._rows))

    def split_counts(self, level: int) -> Dict[str, int]:
        '''Rows per split at a level (counts only; reading rows goes through the log).'''

        return split_counts(self._frame(level))

    def cell_status(self, regime: Regime, level: int) -> Optional[str]:
        '''None if the regime is defined at the level, otherwise the reason it is not.'''

        regime = Regime(regime)
        if level not in self._rows:
            return f'level {level} is not loaded'
        if regime is Regime.HELDOUT and level < GROUP_LEVEL:
            return 'no four-digit parent: the held-out regime runs at levels 4-6'
        remainder = self._remainder(level)
        if regime is Regime.SEEN:
            remainder = remainder.filter(pl.col('feature_year') == REMAINDER_FEATURE_YEARS[1])
        n_groups = remainder.get_column('group').n_unique()
        if n_groups < self.settings.min_groups:
            return f'{n_groups} remainder groups, fewer than {self.settings.min_groups}'
        return None

    def require_arm(self, arm: ArmTables) -> None:
        '''
        Require the arm to cover every code of every loaded level; logs nothing.

        Check before opening an outer set: a test read that failed after the opening would use
        the opening up. Codes are not sealed; only rows are.

        Raises:
            ValueError: If the arm has no coordinates for a panel code.
        '''

        for level in self.levels:
            _require_codes(arm, self._frame(level))

    def validation(self, regime: Regime, level: int, arm: ArmTables, purpose: str) -> pl.DataFrame:
        '''Out-of-sample predictions for the remainder rows, logging the read.'''

        regime = Regime(regime)
        self._require_defined(regime, level)
        frame = self._remainder(level)
        if regime is Regime.SEEN:
            plan = seen_validation_plan(frame, level, self.settings)
        else:
            plan = heldout_validation_plan(frame, level, self.settings)
        _require_codes(arm, frame)
        self._log_read(regime, VALIDATION, level, arm, purpose, frame.height)
        return self._predict(regime, VALIDATION, level, frame, plan, arm)

    def open_outer(
        self, regime: Regime, purpose: str, *, reopen_reason: Optional[str] = None
    ) -> None:
        '''
        Open one regime's sealed outer set for this panel object, logging the opening.

        Raises:
            SplitAlreadyOpenedError: If the log already records an opening of this split and no
                ``reopen_reason`` is given.
        '''

        regime = Regime(regime)
        panel = PANEL_NAMES[regime]
        prior = self.log.openings(panel, self.fingerprint)
        reason = (reopen_reason or '').strip()
        if prior and not reason:
            first = prior[0]
            raise SplitAlreadyOpenedError(
                f'the {panel} outer set was opened at {first["time"]} for {first["purpose"]!r}; '
                'opening it again needs reopen_reason'
            )
        outer = {
            level: frame.filter(pl.col('split') == OUTER_SPLITS[regime].value).height
            for level, frame in sorted(self._rows.items())
        }
        detail: Dict[str, Any] = {'rows_by_level': {str(level): n for level, n in outer.items()}}
        if reason:
            detail['reason'] = reason
        self.log.append(
            SelectionEvent.REOPEN if prior else SelectionEvent.OPEN,
            panel=panel,
            split=TEST,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=outer.get(DECISION_LEVEL, sum(outer.values())),
            detail=detail,
        )
        self._open.add(regime)

    def test(self, regime: Regime, level: int, arm: ArmTables, purpose: str) -> pl.DataFrame:
        '''
        Predictions for the regime's outer set from a fit on the whole remainder, logging the read.

        Raises:
            SealedSplitError: If this panel object has not opened the regime's outer set.
        '''

        regime = Regime(regime)
        self._require_defined(regime, level)
        if regime not in self._open:
            raise SealedSplitError(
                f'the {PANEL_NAMES[regime]} outer set is sealed: call open_outer(regime, purpose) '
                'first, which is logged'
            )
        splits = [RegressorSplit.REMAINDER.value, OUTER_SPLITS[regime].value]
        frame = self._frame(level).filter(pl.col('split').is_in(splits))
        plan = outer_plan(frame, regime, level, self.settings)
        n_outer = len(plan[0].score)
        _require_codes(arm, frame)
        self._log_read(regime, TEST, level, arm, purpose, n_outer)
        return self._predict(regime, TEST, level, frame, plan, arm)

    def _frame(self, level: int) -> pl.DataFrame:
        if level not in self._rows:
            raise ValueError(f'level {level} is not loaded')
        return self._rows[level]

    def _remainder(self, level: int) -> pl.DataFrame:
        return self._frame(level).filter(pl.col('split') == RegressorSplit.REMAINDER.value)

    def _require_defined(self, regime: Regime, level: int) -> None:
        reason = self.cell_status(regime, level)
        if reason is not None:
            raise ValueError(f'the {regime.value} regime is undefined at level {level}: {reason}')

    def _log_read(
        self, regime: Regime, split: str, level: int, arm: ArmTables, purpose: str, n_rows: int
    ) -> None:
        self.log.append(
            SelectionEvent.READ,
            panel=PANEL_NAMES[regime],
            split=split,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=n_rows,
            detail={
                'level': level,
                'comparators': list(comparators(regime, level)),
                'arm': arm.fingerprint,
                'text_only': arm.text_only_fingerprint,
                'dimension': arm.dimension,
            },
        )

    def _predict(
        self,
        regime: Regime,
        split: str,
        level: int,
        frame: pl.DataFrame,
        plan: Sequence[FitTask],
        arm: ArmTables,
    ) -> pl.DataFrame:
        y = frame.get_column('outcome').to_numpy()
        keys = frame.select('code', 'group', 'feature_year', 'outcome_year', 'outcome')
        parts = []
        for comparator in comparators(regime, level):
            x = feature_matrix(comparator, frame, level, arm)
            for task, alpha, predictions in run_plan(plan, x, y, self.settings.alphas):
                scored = keys[list(task.score)]
                parts.append(
                    scored.with_columns(
                        panel=pl.lit(PANEL_NAMES[regime]),
                        split=pl.lit(split),
                        level=pl.lit(level, dtype=pl.Int32),
                        comparator=pl.lit(comparator),
                        repeat=pl.lit(task.repeat, dtype=pl.Int32),
                        fold=pl.lit(task.fold, dtype=pl.Int32),
                        alpha=pl.lit(alpha, dtype=pl.Float64),
                        prediction=pl.Series(predictions, dtype=pl.Float64),
                    )
                )
        return pl.concat(parts).select(PREDICTION_COLUMNS)

# -------------------------------------------------------------------------------------------------
# From config
# -------------------------------------------------------------------------------------------------

def require_branch_record(cfg: RegressorPanelConfig) -> RegressorBranchRecord:
    '''
    The configured branch record.

    Raises:
        ValueError: If the config has none.
    '''

    if cfg.branch_record is None:
        raise ValueError(
            'the regressor panel config has no branch_record (conf/data/regressor_panel.yaml)'
        )
    return cfg.branch_record

def fit_settings(cfg: RegressorPanelConfig) -> FitSettings:
    '''The config's tuning and resampling settings.'''

    return FitSettings(
        alphas=tuple(cfg.alphas),
        folds=cfg.folds,
        repeats=cfg.repeats,
        inner_folds=cfg.inner_folds,
        fold_seed=cfg.fold_seed,
        min_groups=cfg.min_groups,
    )

def load_regressor_panel(
    cfg: RegressorPanelConfig,
    codebook_path: Union[str, Path],
    *,
    log_path: Optional[Union[str, Path]] = None,
    levels: Sequence[int] = LEVELS,
) -> RegressorPanel:
    '''The panel the config describes, over a codebook, logging to ``log_path`` or the config's.'''

    return RegressorPanel.from_sources(
        qcew_dir=cfg.qcew_dir,
        qcew_sha256=cfg.qcew_sha256,
        codebook_codes=read_codebook_codes(Path(codebook_path), cfg.codebook_codes_sha256),
        heldout_groups_csv=cfg.heldout_groups_csv,
        log_path=log_path or cfg.selection_log,
        settings=fit_settings(cfg),
        branch_record=require_branch_record(cfg).model_dump(),
        levels=levels,
    )

def summarize(predictions: pl.DataFrame) -> pl.DataFrame:
    '''
    Descriptive fit per panel, split, level and comparator: rows, RMSE and R² pooled over repeats.

    A report for reading, not the decision statistic (Stage 4 settles that).
    '''

    keys = ['panel', 'split', 'level', 'comparator']
    # yapf: disable
    return (
        predictions
        .with_columns(
            residual=pl.col('prediction') - pl.col('outcome'),
            centered=pl.col('outcome') - pl.col('outcome').mean().over(keys),
        )
        .group_by(keys, maintain_order=True)
        .agg(
            rows=pl.len(),
            rmse=(pl.col('residual')**2).mean().sqrt(),
            r2=1 - (pl.col('residual')**2).sum() / (pl.col('centered')**2).sum(),
            median_alpha=pl.col('alpha').median(),
        )
    )
    # yapf: enable
