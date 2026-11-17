'''
The ITA text-to-code panel (spec ``text-to-code-ita-panel``): Stage 10's pair tables read into the
panel's pairs.

Inputs
    The panel reads two of Stage 10's tables, ``ita_text_pairs`` and ``ita_text_pair_codes``. It
    keeps the pairs with ``flag_headline``. A pair's reported codes are the distinct codes of its
    ``flag_headline_basis`` rows, each read from a ``naics_2022_node_id`` of the form
    ``naics:2022:national_industry:<code>``. One node listed under several filing years is one
    reported code. Only the columns in ``PAIR_COLUMNS`` and ``PAIR_CODE_COLUMNS`` are read, and
    only a pair's own headline rows are checked beyond their columns. An input that disagrees with
    the spec is refused with a ``ValueError``, never dropped: a column that is missing or of
    another dtype, a repeated pair, a null in a column read, a node id of another system, vintage
    or level, a code outside the candidates, a headline pair with no reported code, and a conflict
    flag that disagrees with the reported codes.

Panel
    ``ItaPanel`` mirrors ``OutcomePanel`` under its own panel name, ``ita_text``, and shares no
    class with it. Every read of a part appends a ``read`` record to the selection log before any
    query is encoded. The sealed part is read only after ``open_sealed`` on the same panel object.
    Openings are keyed on the split fingerprint, which a rebuilt pair table does not change, so a
    second opening needs a stated reason and is logged as ``reopen``. The data fingerprint names
    what a read scores: each panel pair's id, query, reported codes and copied flag, and the
    candidates. Computing it is not a read.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import polars as pl

from naics_embedder.panels.decoding import resolve_distance
from naics_embedder.panels.ita_scoring import ItaResult, score_shortlist, summarize_ita
from naics_embedder.panels.ita_split import (
    ITA_PANEL,
    ItaPanelSettings,
    ItaPart,
    load_ita_settings,
    split_fingerprint,
    split_part,
)
from naics_embedder.panels.outcome import (
    QueryCodeEncoder,
    SealedSplitError,
    SplitAlreadyOpenedError,
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog, merge_read_detail
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle

# -------------------------------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------------------------------

# The columns of ``ita_text_pairs`` the panel reads, with Stage 10's dtypes
PAIR_COLUMNS: Dict[str, pl.DataType] = {
    'pair_id': pl.Utf8,
    'description_key': pl.Utf8,
    'representative_string': pl.Utf8,
    'flag_headline': pl.Boolean,
    'flag_headline_codes_conflict': pl.Boolean,
    'flag_copied_title_any': pl.Boolean,
    'flag_copied_index_any': pl.Boolean,
    'flag_suspected_mojibake': pl.Boolean,
    'flag_employer_company_fallback': pl.Boolean,
}

# The columns of ``ita_text_pair_codes`` the panel reads, with Stage 10's dtypes
PAIR_CODE_COLUMNS: Dict[str, pl.DataType] = {
    'pair_id': pl.Utf8,
    'naics_2022_node_id': pl.Utf8,
    'flag_headline_basis': pl.Boolean,
}

# A reported code's node id is these three parts, then the code
NODE_PREFIX = ('naics', '2022', 'national_industry')
CODE_PATTERN = re.compile(r'[0-9]{6}')
NODE_FORM = ':'.join((*NODE_PREFIX, '<six digits>'))

PANEL_PAIR_SCHEMA: Dict[str, pl.DataType] = {
    'pair_id': pl.Utf8,
    'description_key': pl.Utf8,
    'query': pl.Utf8,
    'codes': pl.List(pl.Utf8),
    'copied': pl.Boolean,
    'conflict': pl.Boolean,
    'mojibake': pl.Boolean,
    'company_fallback': pl.Boolean,
    'part': pl.Utf8,
}

# The pair columns a headline pair may not leave null, and the two whose null no pair can carry
_PAIR_KEY_COLUMNS = ('pair_id', 'flag_headline')
_HEADLINE_PAIR_COLUMNS = tuple(name for name in PAIR_COLUMNS if name not in _PAIR_KEY_COLUMNS)

def node_code(node_id: str) -> str:
    '''
    The six-digit code of a NAICS 2022 national-industry node id.

    Args:
        node_id: A node id, ``naics:2022:national_industry:<code>``.

    Returns:
        The code.

    Raises:
        ValueError: When the node id is of another system, vintage or level, has another number of
            parts, or its code is not six digits.
    '''

    parts = node_id.split(':')
    code = parts[-1]
    if tuple(parts[:-1]) != NODE_PREFIX or not CODE_PATTERN.fullmatch(code):
        raise ValueError(f'{node_id!r} is not a {NODE_FORM} node id')
    return code

def _require_columns(table: pl.DataFrame, name: str, expected: Dict[str, pl.DataType]) -> None:
    for column, dtype in expected.items():
        if column not in table.columns:
            raise ValueError(f'{name} lacks column {column!r}')
        if table.schema[column] != dtype:
            raise ValueError(
                f'{name} column {column!r} has type {table.schema[column]}, expected {dtype}'
            )

def _first_null_pair(table: pl.DataFrame, column: str) -> Optional[str]:
    '''The id of the first pair whose value in a column is null, or None when none is.'''

    nulls = table.filter(pl.col(column).is_null())
    return None if nulls.is_empty() else nulls.get_column('pair_id').item(0)

def _reported_codes(
    basis: pl.DataFrame,
    candidates: Sequence[str],
) -> Dict[str, List[str]]:
    '''Each pair's distinct reported codes, sorted, from its headline-basis rows.'''

    allowed = set(candidates)
    reported: Dict[str, set] = {}
    for pair_id, node_id in basis.select('pair_id', 'naics_2022_node_id').iter_rows():
        if node_id is None:
            raise ValueError(
                f"pair {pair_id} has a headline-basis row with a null 'naics_2022_node_id'"
            )
        code = node_code(node_id)
        if code not in allowed:
            raise ValueError(f'pair {pair_id} reports code {code}, which is not a candidate')
        reported.setdefault(pair_id, set()).add(code)
    return {pair_id: sorted(codes) for pair_id, codes in reported.items()}

def panel_pairs(
    pairs: pl.DataFrame,
    pair_codes: pl.DataFrame,
    candidates: Sequence[str],
    salt: str,
) -> pl.DataFrame:
    '''
    Stage 10's pair tables as the panel's pairs: the headline pairs and their reported codes.

    Args:
        pairs: ``ita_text_pairs``.
        pair_codes: ``ita_text_pair_codes``.
        candidates: The bundle's six-digit codes.
        salt: The split's salt.

    Returns:
        One row for each headline pair, sorted by ``pair_id``, in ``PANEL_PAIR_SCHEMA``. ``codes``
        are the pair's distinct reported codes, sorted. ``copied`` is a copied title or a copied
        index (U38). ``part`` is the pair's part, ``validation`` or ``sealed``.

    Raises:
        ValueError: When a column the panel reads is missing or of another dtype in either table,
            a ``pair_id`` repeats, ``pair_id`` or ``flag_headline`` is null, a headline pair has
            a null in a column read or no ``flag_headline_basis`` row, a headline-basis row has a
            null or a malformed node id (see ``node_code``) or a code outside ``candidates``, or
            ``flag_headline_codes_conflict`` disagrees with the number of reported codes.
    '''

    _require_columns(pairs, 'pair table', PAIR_COLUMNS)
    _require_columns(pair_codes, 'pair-code table', PAIR_CODE_COLUMNS)

    for column in _PAIR_KEY_COLUMNS:
        if pairs.get_column(column).null_count():
            raise ValueError(f'pair table has a null {column!r}')
    repeated = pairs.filter(pl.col('pair_id').is_duplicated()).get_column('pair_id')
    if not repeated.is_empty():
        raise ValueError(f'pair table repeats pair_id: {repeated.unique().sort().to_list()}')

    headline = pairs.filter(pl.col('flag_headline')).sort('pair_id')
    for column in _HEADLINE_PAIR_COLUMNS:
        pair_id = _first_null_pair(headline, column)
        if pair_id is not None:
            raise ValueError(f'headline pair {pair_id} has a null {column!r}')

    rows = pair_codes.filter(pl.col('pair_id').is_in(headline.get_column('pair_id')))
    pair_id = _first_null_pair(rows, 'flag_headline_basis')
    if pair_id is not None:
        raise ValueError(f"headline pair {pair_id} has a null 'flag_headline_basis'")
    reported = _reported_codes(rows.filter(pl.col('flag_headline_basis')), candidates)

    panel = []
    for pair in headline.iter_rows(named=True):
        pair_id = pair['pair_id']
        codes = reported.get(pair_id)
        if not codes:
            raise ValueError(f'headline pair {pair_id} has no headline-basis row')
        if (len(codes) > 1) != pair['flag_headline_codes_conflict']:
            raise ValueError(
                f'pair {pair_id} has flag_headline_codes_conflict '
                f'{pair["flag_headline_codes_conflict"]} but reports {codes}'
            )
        panel.append(
            {
                'pair_id': pair_id,
                'description_key': pair['description_key'],
                'query': pair['representative_string'],
                'codes': codes,
                'copied': pair['flag_copied_title_any'] or pair['flag_copied_index_any'],
                'conflict': pair['flag_headline_codes_conflict'],
                'mojibake': pair['flag_suspected_mojibake'],
                'company_fallback': pair['flag_employer_company_fallback'],
                'part': split_part(pair['description_key'], salt).value,
            }
        )
    return pl.DataFrame(panel, schema=PANEL_PAIR_SCHEMA)

# -------------------------------------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------------------------------------

# The roles a caller's read may name: a reference arm, or a candidate judged against one (P5)
READ_ROLES = ('reference', 'candidate')

# The distance an encoder without a ``distance`` attribute is read under: the stub's (P4)
DEFAULT_DISTANCE = 'cosine'

class ItaPanel:
    '''
    The ITA pairs in their two parts, the six-digit candidates, and the log every read goes to.

    Args:
        pairs: ``ita_text_pairs``.
        pair_codes: ``ita_text_pair_codes``.
        candidates: The six-digit codes a query is ranked over.
        settings: The panel's settings, ``conf/data/ita_panel.yaml``.
        log: The selection log.

    Attributes:
        pairs: The panel pairs (``panel_pairs``), sorted by ``pair_id``.
        candidates: The candidates, sorted.
        settings: The panel's settings.
        log: The selection log.
        fingerprint: The split fingerprint, which keys the sealed part's openings.

    Raises:
        ValueError: When the candidates are not distinct six-digit codes, or ``panel_pairs``
            refuses the tables. Nothing is logged.
    '''

    def __init__(
        self,
        pairs: pl.DataFrame,
        pair_codes: pl.DataFrame,
        candidates: Sequence[str],
        settings: ItaPanelSettings,
        log: SelectionLog,
    ):
        candidates = [str(code) for code in candidates]
        if len(set(candidates)) != len(candidates) or any(
            not CODE_PATTERN.fullmatch(code) for code in candidates
        ):
            raise ValueError('candidates must be distinct six-digit codes')
        self.pairs = panel_pairs(pairs, pair_codes, candidates, settings.salt)
        self.candidates: Tuple[str, ...] = tuple(sorted(candidates))
        self.settings = settings
        self.log = log
        self.fingerprint = split_fingerprint(settings)
        self._sealed_open = False

    @classmethod
    def from_files(
        cls,
        pairs_parquet: Union[str, Path],
        pair_codes_parquet: Union[str, Path],
        bundle: ValidatedSupervisionBundle,
        settings_path: Union[str, Path],
        log_path: Union[str, Path],
    ) -> 'ItaPanel':
        '''
        The panel from Stage 10's two tables and a validated bundle's codebook.

        Args:
            pairs_parquet: ``ita_text_pairs.parquet``.
            pair_codes_parquet: ``ita_text_pair_codes.parquet``.
            bundle: The supervision bundle whose codebook's six-digit codes are the candidates.
            settings_path: The panel's settings file, ``conf/data/ita_panel.yaml``.
            log_path: The selection log's file.

        Returns:
            The panel.
        '''

        codebook = pl.read_parquet(bundle.artifact_path('codebook'))
        candidates = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        return cls(
            pl.read_parquet(pairs_parquet),
            pl.read_parquet(pair_codes_parquet),
            candidates.to_list(),
            load_ita_settings(settings_path),
            SelectionLog(Path(log_path)),
        )

    def data_fingerprint(self, part: Union[ItaPart, str]) -> str:
        '''
        SHA-256 of what a read of the part scores.

        The payload is each panel pair's ``pair_id``, query, reported codes and copied flag, in
        ``pair_id`` order, and the candidates, as canonical JSON. ``fingerprint`` names the split
        alone, so two panels can share it and score other pairs. This returns no rows, so it is
        not a read and logs nothing, even for the sealed part.

        Args:
            part: ``validation`` or ``sealed``.

        Returns:
            The hexadecimal digest.
        '''

        pairs = self._part(ItaPart(part)).sort('pair_id')
        rows = pairs.select('pair_id', 'query', 'codes', 'copied').rows()
        payload = {
            'pairs': [list(row) for row in rows],
            'candidates': list(self.candidates),
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

        prior = self.log.openings(ITA_PANEL, self.fingerprint)
        reason = (reopen_reason or '').strip()
        if prior and not reason:
            first = prior[0]
            raise SplitAlreadyOpenedError(
                f'the ITA sealed part was opened at {first["time"]} for {first["purpose"]!r}; '
                f'opening it again needs reopen_reason'
            )
        self.log.append(
            SelectionEvent.REOPEN if prior else SelectionEvent.OPEN,
            panel=ITA_PANEL,
            split=ItaPart.SEALED.value,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=self._part(ItaPart.SEALED).height,
            detail={'reason': reason} if reason else None,
        )
        self._sealed_open = True

    def score(
        self,
        encoder: QueryCodeEncoder,
        part: Union[ItaPart, str],
        purpose: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> ItaResult:
        '''
        Rank every candidate for each of the part's queries with the encoder, logging the read.

        Args:
            encoder: The arm: queries and codes in one space. Its ``distance`` attribute names the
                distance it is read under; without one, cosine.
            part: ``validation`` or ``sealed``.
            purpose: Why the part is read.
            detail: The caller's detail joined to the read's logged detail: the arm (``arm``), its
                seed, its role in ``READ_ROLES`` (``role``) and the run (``run``).

        Returns:
            The part's ``ItaResult``. ``score_logged`` also returns the record the read logged.
        '''

        result, _ = self.score_logged(encoder, part, purpose, detail=detail)
        return result

    def score_logged(
        self,
        encoder: QueryCodeEncoder,
        part: Union[ItaPart, str],
        purpose: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ItaResult, Dict[str, Any]]:
        '''
        ``score``, also returning the record the read appended to the selection log.

        The refusals come first and log nothing. The record is then appended before any query is
        encoded, so a read that fails while it scores is logged all the same. Its detail holds the
        encoder's type name, the distance's name and the part's data fingerprint, joined with the
        caller's ``detail``.

        Args:
            encoder: The arm: queries and codes in one space. Its ``distance`` attribute names the
                distance it is read under; without one, cosine.
            part: ``validation`` or ``sealed``.
            purpose: Why the part is read.
            detail: The caller's detail: the arm (``arm``), its seed, its role in ``READ_ROLES``
                (``role``) and the run (``run``). A missing role is allowed.

        Returns:
            The part's ``ItaResult``, and the read's record as the log appended it.

        Raises:
            ValueError: When the part is neither ``validation`` nor ``sealed``, ``detail`` names a
                role outside ``READ_ROLES`` or a key the panel logs itself.
            SealedSplitError: When the part is sealed and this panel object has not opened it.
        '''

        part = ItaPart(part)
        if detail is not None and 'role' in detail and detail['role'] not in READ_ROLES:
            raise ValueError(
                f'a read\'s role must be one of {list(READ_ROLES)}, not {detail["role"]!r}'
            )
        if part is ItaPart.SEALED and not self._sealed_open:
            raise SealedSplitError(
                'the ITA sealed part is sealed: call open_sealed(purpose) first, which is logged'
            )
        distance = getattr(encoder, 'distance', DEFAULT_DISTANCE)
        name, _ = resolve_distance(distance)
        logged = {
            'encoder': type(encoder).__name__,
            'distance': name,
            'data_fingerprint': self.data_fingerprint(part),
        }
        merged = merge_read_detail(logged, detail)
        pairs = self._part(part)
        record = self.log.append(
            SelectionEvent.READ,
            panel=ITA_PANEL,
            split=part.value,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=pairs.height,
            detail=merged,
        )
        scores = score_shortlist(
            encoder.encode_queries(pairs.get_column('query').to_list()),
            pairs.get_column('codes').to_list(),
            encoder.encode_codes(list(self.candidates)),
            self.candidates,
            pairs.get_column('pair_id').to_list(),
            distance=distance,
        )
        return summarize_ita(scores, pairs, name), record

    def _part(self, part: ItaPart) -> pl.DataFrame:
        return self.pairs.filter(pl.col('part') == part.value)
