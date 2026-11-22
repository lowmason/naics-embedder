'''
The relatedness judging panel (spec ``relatedness-artifact``): Stage 10's co-ownership tables read
into the panel's gains, and the panel that reads an arm on them.

Inputs
    The panel reads two of Stage 10's tables, ``ita_co_ownership`` and ``co_ownership_marginals``,
    and the bundle's six-digit codes. From the target rows it reads ``family``, ``measure``, the two
    unit types, the two units and ``n_unweighted``, which is J(a, b), the employers with
    establishments in both industries. From the marginal rows it reads ``node_id`` and
    ``n_multi_industry_employers``, which is m(a). Each node id has the form
    ``naics:2022:national_industry:<code>``. An input that disagrees with the spec is refused with a
    ``ValueError`` that names the table and the first offending row, never dropped. The checks run
    in a fixed order: a column that is missing or of another dtype, a null in a column read, a
    family, measure or unit type that is not accepted, a node id of another system, vintage or
    level, a code outside the codebook, a repeated marginal node, a target row whose industry has no
    marginal row, an industry paired with itself, a repeated ordered pair, a J below 1, a pair in
    one direction only, a pair whose two directions hold different J, and a J above m of either
    industry. A codebook entry that is not six digits is refused before any table is read. The gain
    of a candidate b for an anchor a is J(a, b) / m(b), and 0 when no row joins them.

Panel
    ``RelatednessPanel`` mirrors ``ItaPanel`` under its own panel name,
    ``relatedness_co_ownership``, and shares no class with it. The candidates are the codes with a
    marginal row. A code falls in the part that its salted hash gives, and an anchor is a candidate
    with at least one partner. Every read of a part appends a ``read`` record to the selection log
    before the arm is asked for a similarity. A role outside ``READ_ROLES``, a sealed read before
    ``open_sealed`` and a part with no anchor are refused first, and log nothing. The sealed part
    is read only after ``open_sealed`` on the same panel object. Openings are keyed on the split
    fingerprint, which a rebuilt table does not change, so a second opening needs a stated reason
    and is logged as ``reopen``. The data fingerprint names what a read scores: the part's anchors
    with their partners and gains, the candidates and the two tables' SHA-256 values. Computing it
    is not a read. ``from_files`` checks each table's SHA-256 against its pin before it reads a
    table.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import dataclasses
import hashlib
import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

from naics_embedder.panels.ita import CODE_PATTERN, node_code
from naics_embedder.panels.outcome import SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.relatedness_arms import RelatednessArm
from naics_embedder.panels.relatedness_scoring import (
    RelatednessResult,
    score_anchors,
    summarize_relatedness,
)
from naics_embedder.panels.relatedness_split import (
    RELATEDNESS_PANEL,
    RelatednessPanelSettings,
    RelatednessPart,
    load_relatedness_settings,
    split_fingerprint,
    split_part,
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog, merge_read_detail
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file

# -------------------------------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------------------------------

# The columns of ``ita_co_ownership`` the panel reads, with Stage 10's dtypes. ``n_unweighted`` is J
TARGET_COLUMNS: Dict[str, pl.DataType] = {
    'family': pl.Utf8,
    'measure': pl.Utf8,
    'unit_a_type': pl.Utf8,
    'unit_a': pl.Utf8,
    'unit_b_type': pl.Utf8,
    'unit_b': pl.Utf8,
    'n_unweighted': pl.UInt64,
}

# The columns of ``co_ownership_marginals`` the panel reads. ``n_multi_industry_employers`` is m
MARGINAL_COLUMNS: Dict[str, pl.DataType] = {
    'node_id': pl.Utf8,
    'n_multi_industry_employers': pl.UInt64,
}

# The only family, measure and unit type a target row may hold
CO_OWNERSHIP_FAMILY = 'co_ownership'
CO_OWNERSHIP_MEASURE = 'employer_share'
INDUSTRY_UNIT = 'naics_national_industry'

# The two tables' names, which a refusal gives
TABLES = ('ita_co_ownership', 'co_ownership_marginals')
TARGETS_TABLE, MARGINALS_TABLE = TABLES

# A target column with one accepted value, and what a refusal calls it
_ACCEPTED = (
    ('family', 'family', CO_OWNERSHIP_FAMILY),
    ('measure', 'measure', CO_OWNERSHIP_MEASURE),
    ('unit_a_type', 'unit type', INDUSTRY_UNIT),
    ('unit_b_type', 'unit type', INDUSTRY_UNIT),
)

# The columns that name a target row and a marginal row in a refusal
_TARGET_ROW = ('unit_a', 'unit_b')
_MARGINAL_ROW = ('node_id', )

# One target row as the panel reads it: the codes of its two industries, then J
Pair = Tuple[str, str, int]

@dataclasses.dataclass(frozen=True)
class CoOwnership:
    '''
    Stage 10's co-ownership tables, read into the numbers a gain needs.

    Attributes:
        candidates: The six-digit codes that have a marginal row, sorted.
        multi: m for each candidate: its employers that also operate in another industry.
        shared: J for each ordered pair of codes with a target row. Both directions of a pair are
            keys, with one J.
    '''

    candidates: Tuple[str, ...]
    multi: Mapping[str, int]
    shared: Mapping[Tuple[str, str], int]

    @cached_property
    def _partner_index(self) -> Dict[str, Tuple[str, ...]]:
        partners: Dict[str, List[str]] = {}
        for code_a, code_b in self.shared:
            partners.setdefault(code_a, []).append(code_b)
        return {code: tuple(sorted(codes)) for code, codes in partners.items()}

    def partners(self, code: str) -> Tuple[str, ...]:
        '''
        The codes b that have a target row (``code``, b).

        Args:
            code: A six-digit code.

        Returns:
            The codes, sorted. The tuple is empty for a code with no row.
        '''

        return self._partner_index.get(code, ())

    def gain(self, anchor: str, candidate: str) -> float:
        '''
        The gain of a candidate for an anchor: of the candidate's multi-industry employers, the
        share that also operate in the anchor, J(anchor, candidate) / m(candidate).

        Args:
            anchor: The anchor's six-digit code.
            candidate: The candidate's six-digit code.

        Returns:
            The gain, in (0, 1] for a pair with a row and 0.0 for any other pair.
        '''

        shared = self.shared.get((anchor, candidate))
        return 0.0 if shared is None else shared / self.multi[candidate]

def _require_columns(table: pl.DataFrame, name: str, expected: Dict[str, pl.DataType]) -> None:
    for column, dtype in expected.items():
        if column not in table.columns:
            raise ValueError(f'{name} lacks column {column!r}')
        if table.schema[column] != dtype:
            raise ValueError(
                f'{name} column {column!r} has type {table.schema[column]}, expected {dtype}'
            )

def _row_text(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    return '(' + ', '.join(str(row[column]) for column in columns) + ')'

def _refuse_nulls(
    table: pl.DataFrame,
    name: str,
    columns: Sequence[str],
    row_columns: Sequence[str],
) -> None:
    '''Refuse the first row, in table order, with a null in one of the columns read.'''

    bad = table.filter(pl.any_horizontal([pl.col(column).is_null() for column in columns]))
    if not bad.is_empty():
        row = bad.row(0, named=True)
        column = next(column for column in columns if row[column] is None)
        raise ValueError(f'{name} has a null {column!r} in row {_row_text(row, row_columns)}')

def _refuse_unaccepted_values(targets: pl.DataFrame) -> None:
    '''Refuse the first row, in table order, whose family, measure or unit type is not accepted.'''

    bad = targets.filter(
        pl.any_horizontal([pl.col(column) != accepted for column, _, accepted in _ACCEPTED])
    )
    if not bad.is_empty():
        row = bad.row(0, named=True)
        column, label, accepted = next(
            (column, label, accepted) for column, label, accepted in _ACCEPTED
            if row[column] != accepted
        )
        raise ValueError(
            f'{TARGETS_TABLE} row {_row_text(row, _TARGET_ROW)} has {label} {row[column]!r} in '
            f'column {column!r}, which is not accepted: expected {accepted!r}'
        )

def _code(node_id: str, name: str, column: str, row: str = '') -> str:
    '''The code of a node id, or a refusal that names the table, the column and the row.'''

    try:
        return node_code(node_id)
    except ValueError as error:
        where = f'{name} row {row} column {column!r}' if row else f'{name} column {column!r}'
        raise ValueError(f'{where}: {error}') from None

def _pairs(targets: pl.DataFrame) -> List[Pair]:
    '''Each target row's two codes and J, in table order.'''

    pairs = []
    for unit_a, unit_b, j in targets.select('unit_a', 'unit_b', 'n_unweighted').iter_rows():
        row = f'({unit_a}, {unit_b})'
        pairs.append(
            (
                _code(unit_a, TARGETS_TABLE, 'unit_a', row),
                _code(unit_b, TARGETS_TABLE, 'unit_b', row),
                j,
            )
        )
    return pairs

def _nodes(marginals: pl.DataFrame) -> List[Tuple[str, int]]:
    '''Each marginal row's code and m, in table order.'''

    return [
        (_code(node_id, MARGINALS_TABLE, 'node_id'), m)
        for node_id, m in marginals.select('node_id', 'n_multi_industry_employers').iter_rows()
    ]

def _codebook(codebook_codes: Sequence[str]) -> FrozenSet[str]:
    for code in codebook_codes:
        if not CODE_PATTERN.fullmatch(code):
            raise ValueError(f'codebook_codes holds {code!r}, which is not a six-digit code')
    return frozenset(codebook_codes)

def _refuse_codes_outside(
    codebook: FrozenSet[str],
    pairs: Sequence[Pair],
    nodes: Sequence[Tuple[str, int]],
) -> None:
    for code_a, code_b, _ in pairs:
        for code in (code_a, code_b):
            if code not in codebook:
                raise ValueError(
                    f'{TARGETS_TABLE} row ({code_a}, {code_b}) has {code}, '
                    'which is not a codebook code'
                )
    for code, _ in nodes:
        if code not in codebook:
            raise ValueError(f'{MARGINALS_TABLE} has {code}, which is not a codebook code')

def _marginal_counts(nodes: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    multi: Dict[str, int] = {}
    for code, m in nodes:
        if code in multi:
            raise ValueError(f'{MARGINALS_TABLE} repeats node {code}')
        multi[code] = m
    return multi

def _refuse_industries_without_marginals(pairs: Sequence[Pair], multi: Mapping[str, int]) -> None:
    for code_a, code_b, _ in pairs:
        for code in (code_a, code_b):
            if code not in multi:
                raise ValueError(
                    f'{TARGETS_TABLE} row ({code_a}, {code_b}) has no marginal row for {code}'
                )

def _refuse_self_pairs(pairs: Sequence[Pair]) -> None:
    for code_a, code_b, _ in pairs:
        if code_a == code_b:
            raise ValueError(f'{TARGETS_TABLE} pairs {code_a} with itself')

def _shared_counts(pairs: Sequence[Pair]) -> Dict[Tuple[str, str], int]:
    shared: Dict[Tuple[str, str], int] = {}
    for code_a, code_b, j in pairs:
        if (code_a, code_b) in shared:
            raise ValueError(f'{TARGETS_TABLE} repeats the pair ({code_a}, {code_b})')
        shared[(code_a, code_b)] = j
    return shared

def _refuse_counts_below_one(shared: Mapping[Tuple[str, str], int]) -> None:
    for (code_a, code_b), j in shared.items():
        if j < 1:
            raise ValueError(f'{TARGETS_TABLE} row ({code_a}, {code_b}) has J {j}, not at least 1')

def _refuse_one_way_pairs(shared: Mapping[Tuple[str, str], int]) -> None:
    for code_a, code_b in shared:
        if (code_b, code_a) not in shared:
            raise ValueError(
                f'{TARGETS_TABLE} has the pair ({code_a}, {code_b}) in one direction only: '
                f'no row ({code_b}, {code_a})'
            )

def _refuse_unequal_mirrors(shared: Mapping[Tuple[str, str], int]) -> None:
    for (code_a, code_b), j in shared.items():
        if shared[(code_b, code_a)] != j:
            raise ValueError(
                f'{TARGETS_TABLE} row ({code_a}, {code_b}) has J {j}, but its mirror '
                f'({code_b}, {code_a}) has J {shared[(code_b, code_a)]}'
            )

def _refuse_counts_above_marginals(
    shared: Mapping[Tuple[str, str], int],
    multi: Mapping[str, int],
) -> None:
    for (code_a, code_b), j in shared.items():
        for code in (code_a, code_b):
            if j > multi[code]:
                raise ValueError(
                    f'{TARGETS_TABLE} row ({code_a}, {code_b}) has J {j}, which exceeds m of '
                    f'{code} ({multi[code]})'
                )

def co_ownership_inputs(
    targets: pl.DataFrame,
    marginals: pl.DataFrame,
    codebook_codes: Sequence[str],
) -> CoOwnership:
    '''
    Stage 10's co-ownership tables as the numbers a gain needs.

    Only the columns in ``TARGET_COLUMNS`` and ``MARGINAL_COLUMNS`` are read. Row order and other
    columns do not matter.

    Args:
        targets: ``ita_co_ownership``.
        marginals: ``co_ownership_marginals``.
        codebook_codes: The bundle's six-digit codes.

    Returns:
        The candidates (the codes with a marginal row), m for each, and J for each ordered pair.

    Raises:
        ValueError: When a codebook entry is not six digits. Then, in this order, when a column the
            panel reads is missing or of another dtype, a column it reads holds a null, a family,
            measure or unit type is not ``co_ownership``, ``employer_share`` or
            ``naics_national_industry``, a node id is of another system, vintage or level (see
            ``node_code``), a code is outside ``codebook_codes``, a marginal node repeats, a target
            row's industry has no marginal row, an industry is paired with itself, an ordered pair
            repeats, a J is below 1, a pair has a row in one direction only, a pair's two directions
            hold different J, or a J exceeds m of either industry. The message names the table and
            the first offending row.
    '''

    codebook = _codebook(codebook_codes)
    _require_columns(targets, TARGETS_TABLE, TARGET_COLUMNS)
    _require_columns(marginals, MARGINALS_TABLE, MARGINAL_COLUMNS)
    _refuse_nulls(targets, TARGETS_TABLE, list(TARGET_COLUMNS), _TARGET_ROW)
    _refuse_nulls(marginals, MARGINALS_TABLE, list(MARGINAL_COLUMNS), _MARGINAL_ROW)
    _refuse_unaccepted_values(targets)
    pairs = _pairs(targets)
    nodes = _nodes(marginals)
    _refuse_codes_outside(codebook, pairs, nodes)
    multi = _marginal_counts(nodes)
    _refuse_industries_without_marginals(pairs, multi)
    _refuse_self_pairs(pairs)
    shared = _shared_counts(pairs)
    _refuse_counts_below_one(shared)
    _refuse_one_way_pairs(shared)
    _refuse_unequal_mirrors(shared)
    _refuse_counts_above_marginals(shared, multi)
    return CoOwnership(candidates=tuple(sorted(multi)), multi=multi, shared=shared)

# -------------------------------------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------------------------------------

# The roles a caller's read may name: a reference arm, or a candidate judged against one
READ_ROLES = ('reference', 'candidate')

# A table's SHA-256, as a pin and as the panel keeps it
SHA256_PATTERN = re.compile(r'[0-9a-f]{64}')

def _table_hashes(hashes: Mapping[str, str], name: str) -> Dict[str, str]:
    '''
    The two tables' SHA-256 values, in ``TABLES`` order.

    Raises:
        ValueError: When the keys are not exactly ``TABLES``, or a value is not 64 lowercase
            hexadecimal characters. The message names ``name`` and, for a value, the table.
    '''

    if set(hashes) != set(TABLES):
        raise ValueError(
            f'{name} must name exactly the tables {list(TABLES)}, not {sorted(map(str, hashes))}'
        )
    for table in TABLES:
        value = hashes[table]
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise ValueError(
                f'{name} of {table} must be 64 lowercase hexadecimal characters, not {value!r}'
            )
    return {table: hashes[table] for table in TABLES}

def _refuse_unfit_similarity(
    similarity: np.ndarray,
    anchors: Sequence[str],
    candidates: Sequence[str],
    arm_type: str,
) -> None:
    '''Refuse an arm's matrix of another shape, then the first anchor with a non-finite value.'''

    wanted = (len(anchors), len(candidates))
    if similarity.shape != wanted:
        raise ValueError(
            f'{arm_type} gave a similarity of shape {similarity.shape}, expected {wanted}: one row '
            'for each anchor and one column for each candidate'
        )
    bad = np.flatnonzero(~np.isfinite(similarity).all(axis=1))
    if len(bad):
        raise ValueError(
            f'{arm_type} gave a similarity that is not finite for anchor {anchors[bad[0]]}'
        )

class RelatednessPanel:
    '''
    The co-ownership gains in their two parts, the candidates, and the log every read goes to.

    Args:
        targets: ``ita_co_ownership``.
        marginals: ``co_ownership_marginals``.
        codebook_codes: The bundle's six-digit codes.
        settings: The panel's settings, ``conf/data/relatedness_panel.yaml``.
        log: The selection log.
        table_sha256: Each table's SHA-256, keyed by ``TABLES``. It enters the data fingerprint
            only.

    Attributes:
        name: The panel's name in the selection log, ``relatedness_co_ownership``.
        co_ownership: The tables read into gains (``co_ownership_inputs``).
        candidates: The codes with a marginal row, sorted. Every anchor is ranked over them.
        settings: The panel's settings.
        log: The selection log.
        fingerprint: The split fingerprint, which keys the sealed part's openings.
        table_sha256: Each table's SHA-256, keyed by ``TABLES``.

    Raises:
        ValueError: When ``table_sha256`` does not name exactly ``TABLES`` with 64 lowercase
            hexadecimal characters each, or ``co_ownership_inputs`` refuses the tables. Nothing is
            logged.
    '''

    name = RELATEDNESS_PANEL

    def __init__(
        self,
        targets: pl.DataFrame,
        marginals: pl.DataFrame,
        codebook_codes: Sequence[str],
        settings: RelatednessPanelSettings,
        log: SelectionLog,
        *,
        table_sha256: Mapping[str, str],
    ):
        self.table_sha256 = _table_hashes(table_sha256, 'table_sha256')
        self.co_ownership = co_ownership_inputs(targets, marginals, codebook_codes)
        self.candidates: Tuple[str, ...] = self.co_ownership.candidates
        self.settings = settings
        self.log = log
        self.fingerprint = split_fingerprint(settings)
        self._sealed_open = False

    @classmethod
    def from_files(
        cls,
        targets_parquet: Union[str, Path],
        marginals_parquet: Union[str, Path],
        bundle: ValidatedSupervisionBundle,
        settings_path: Union[str, Path],
        log_path: Union[str, Path],
        *,
        expected_sha256: Mapping[str, str],
    ) -> 'RelatednessPanel':
        '''
        The panel from Stage 10's two tables, each checked against its pin, and a bundle's codebook.

        Args:
            targets_parquet: ``ita_co_ownership.parquet``.
            marginals_parquet: ``co_ownership_marginals.parquet``.
            bundle: The supervision bundle whose codebook's six-digit codes the tables must use.
            settings_path: The panel's settings file, ``conf/data/relatedness_panel.yaml``.
            log_path: The selection log's file.
            expected_sha256: Each table's pinned SHA-256, keyed by ``TABLES``.

        Returns:
            The panel, its ``table_sha256`` the files' SHA-256 values.

        Raises:
            ValueError: When ``expected_sha256`` does not name exactly ``TABLES`` with 64 lowercase
                hexadecimal characters each, or a file's SHA-256 differs from its pin. Both come
                before either table is read, and the second names the table, the pin and the
                SHA-256 found. Then, as the constructor. Nothing is logged.
        '''

        expected = _table_hashes(expected_sha256, 'expected_sha256')
        paths = dict(zip(TABLES, (Path(targets_parquet), Path(marginals_parquet))))
        found = {table: sha256_file(path) for table, path in paths.items()}
        for table in TABLES:
            if found[table] != expected[table]:
                raise ValueError(
                    f'{table} at {paths[table]} has SHA-256 {found[table]}, but its pin is '
                    f'{expected[table]}'
                )
        codebook = pl.read_parquet(bundle.artifact_path('codebook'))
        codes = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        return cls(
            pl.read_parquet(paths[TARGETS_TABLE]),
            pl.read_parquet(paths[MARGINALS_TABLE]),
            codes.to_list(),
            load_relatedness_settings(settings_path),
            SelectionLog(Path(log_path)),
            table_sha256=found,
        )

    def part_of(self, code: str) -> RelatednessPart:
        '''
        The part a code falls in, from its salted hash alone.

        Args:
            code: A six-digit code.

        Returns:
            The part.
        '''

        return split_part(code, self.settings.salt)

    def anchors(self, part: Union[RelatednessPart, str]) -> Tuple[str, ...]:
        '''
        The part's anchors: its candidates with at least one partner.

        Args:
            part: ``validation`` or ``sealed``.

        Returns:
            The codes, sorted.

        Raises:
            ValueError: When the part is neither ``validation`` nor ``sealed``.
        '''

        part = RelatednessPart(part)
        return tuple(
            code for code in self.candidates
            if self.co_ownership.partners(code) and self.part_of(code) is part
        )

    def data_fingerprint(self, part: Union[RelatednessPart, str]) -> str:
        '''
        SHA-256 of what a read of the part scores.

        The payload is canonical JSON of the candidates, each of the part's anchors with its
        partners and their gains, and the two tables' SHA-256 values (P6). Anchors and partners are
        sorted by code. An anchor's candidates are the candidates less the anchor, and a candidate
        that is not a partner has a gain of 0. ``fingerprint`` names the split alone, so two panels
        can share it and score other gains. This returns no score, so it is not a read and logs
        nothing, even for the sealed part.

        Args:
            part: ``validation`` or ``sealed``.

        Returns:
            The hexadecimal digest.
        '''

        payload = {
            'candidates': list(self.candidates),
            'anchors': [
                [
                    anchor,
                    [
                        [partner, self.co_ownership.gain(anchor, partner)]
                        for partner in self.co_ownership.partners(anchor)
                    ],
                ] for anchor in self.anchors(part)
            ],
            'tables': dict(self.table_sha256),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    def open_sealed(self, purpose: str, *, reopen_reason: Optional[str] = None) -> None:
        '''
        Open the sealed part for this panel object, logging the opening.

        Args:
            purpose: Why the part is opened.
            reopen_reason: Why it is opened again, needed when the log already records an opening
                under the split fingerprint.

        Raises:
            SplitAlreadyOpenedError: If the log already records an opening of this split and no
                non-blank ``reopen_reason`` is given.
        '''

        prior = self.log.openings(RELATEDNESS_PANEL, self.fingerprint)
        reason = (reopen_reason or '').strip()
        if prior and not reason:
            first = prior[0]
            raise SplitAlreadyOpenedError(
                f'the relatedness sealed part was opened at {first["time"]} for '
                f'{first["purpose"]!r}; opening it again needs reopen_reason'
            )
        self.log.append(
            SelectionEvent.REOPEN if prior else SelectionEvent.OPEN,
            panel=RELATEDNESS_PANEL,
            split=RelatednessPart.SEALED.value,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=len(self.anchors(RelatednessPart.SEALED)),
            detail={'reason': reason} if reason else None,
        )
        self._sealed_open = True

    def score(
        self,
        arm: RelatednessArm,
        part: Union[RelatednessPart, str],
        purpose: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> RelatednessResult:
        '''
        Score every anchor of the part against every candidate with the arm, logging the read.

        Args:
            arm: The arm: the similarity of each anchor to each candidate.
            part: ``validation`` or ``sealed``.
            purpose: Why the part is read.
            detail: The caller's detail joined to the read's logged detail: the arm (``arm``), its
                seed, its role in ``READ_ROLES`` (``role``) and the run (``run``).

        Returns:
            The part's ``RelatednessResult``, its ``data_fingerprint`` the part's as the read logged
            it. ``score_logged`` also returns the record the read logged.
        '''

        result, _ = self.score_logged(arm, part, purpose, detail=detail)
        return result

    def score_logged(
        self,
        arm: RelatednessArm,
        part: Union[RelatednessPart, str],
        purpose: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[RelatednessResult, Dict[str, Any]]:
        '''
        ``score``, also returning the record the read appended to the selection log.

        The refusals come first and log nothing. The record is then appended before the arm is
        asked for a similarity, so a read that fails while it scores is logged all the same, and an
        arm's own error passes unchanged. Its detail holds the arm's type name and the part's data
        fingerprint, joined with the caller's ``detail``.

        Args:
            arm: The arm: the similarity of each anchor to each candidate.
            part: ``validation`` or ``sealed``.
            purpose: Why the part is read.
            detail: The caller's detail: the arm (``arm``), its seed, its role in ``READ_ROLES``
                (``role``) and the run (``run``). A missing role is allowed.

        Returns:
            The part's ``RelatednessResult``, its ``data_fingerprint`` the part's as the read logged
            it, and the read's record as the log appended it.

        Raises:
            ValueError: Before the read is logged, when the part is neither ``validation`` nor
                ``sealed``, ``detail`` names a role outside ``READ_ROLES``, the part has no anchor
                or ``detail`` names a key the panel logs itself. After it is logged, when the arm's
                matrix is not one row for each anchor and one column for each candidate, or a
                similarity is not finite: the arm's type and the first such anchor are named.
            SealedSplitError: When the part is sealed and this panel object has not opened it,
                after the role check and before the anchor check.
        '''

        part = RelatednessPart(part)
        if detail is not None and 'role' in detail and detail['role'] not in READ_ROLES:
            raise ValueError(
                f'a read\'s role must be one of {list(READ_ROLES)}, not {detail["role"]!r}'
            )
        if part is RelatednessPart.SEALED and not self._sealed_open:
            raise SealedSplitError(
                'the relatedness sealed part is sealed: call open_sealed(purpose) first, which is '
                'logged'
            )
        anchors = self.anchors(part)
        if not anchors:
            raise ValueError(
                f'the relatedness {part.value} part has no anchor: no candidate in it has a partner'
            )
        arm_type = type(arm).__name__
        logged = {'arm_type': arm_type, 'data_fingerprint': self.data_fingerprint(part)}
        merged = merge_read_detail(logged, detail)
        record = self.log.append(
            SelectionEvent.READ,
            panel=RELATEDNESS_PANEL,
            split=part.value,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=len(anchors),
            detail=merged,
        )
        similarity = np.asarray(arm.similarity(anchors, self.candidates), dtype=np.float64)
        _refuse_unfit_similarity(similarity, anchors, self.candidates, arm_type)
        per_anchor = score_anchors(
            similarity, self._gains(anchors), anchors, self.candidates, part.value
        )
        result = summarize_relatedness(
            per_anchor,
            panel=RELATEDNESS_PANEL,
            part=part.value,
            counts=self._counts(part, anchors),
        )
        return dataclasses.replace(result, data_fingerprint=logged['data_fingerprint']), record

    def _gains(self, anchors: Sequence[str]) -> np.ndarray:
        '''Each anchor's gain for each candidate, (anchors, candidates), 0 off its partners.'''

        column_of = {code: column for column, code in enumerate(self.candidates)}
        gains = np.zeros((len(anchors), len(self.candidates)), dtype=np.float64)
        for row, anchor in enumerate(anchors):
            for partner in self.co_ownership.partners(anchor):
                gains[row, column_of[partner]] = self.co_ownership.gain(anchor, partner)
        return gains

    def _counts(self, part: RelatednessPart, anchors: Sequence[str]) -> Dict[str, int]:
        '''The part's counts (4.4), keyed by ``COUNT_KEYS``.'''

        industries = [code for code in self.candidates if self.part_of(code) is part]
        crossing = [
            (code_a, code_b) for code_a, code_b in self.co_ownership.shared
            if code_a < code_b and self.part_of(code_a) is not self.part_of(code_b)
        ]
        return {
            'n_industries': len(industries),
            'n_anchors': len(anchors),
            'n_no_partner': len(industries) - len(anchors),
            'n_candidates': len(self.candidates),
            'n_relevant': sum(len(self.co_ownership.partners(anchor)) for anchor in anchors),
            'n_crossing': len(crossing),
        }
