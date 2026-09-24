'''
The outcome panel (Req 3; Req 4): text-to-code decoding over sealed query splits.

Validation queries may be read at any time, and every read is logged. The test split is sealed:
reading it needs a logged opening by the same panel object, and opening the same split (the same
role assignment, by fingerprint) a second time needs a stated reason. Training queries are
training data, not a selection, so reading them is not logged.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, Union

import polars as pl
import torch

from naics_embedder.panels.decoding import (
    DecodingResult,
    DistanceFn,
    resolve_distance,
    score_decoding,
)
from naics_embedder.panels.index_roles import role_table_fingerprint, verify_examples_channel
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.supervision.artifacts import INDEX_ROLE_COLUMNS, validate_index_role_table
from naics_embedder.supervision.schema import IndexRole

OUTCOME_PANEL = 'outcome'

# -------------------------------------------------------------------------------------------------
# Errors and the encoder interface
# -------------------------------------------------------------------------------------------------

class SealedSplitError(RuntimeError):
    '''The test split was read before a logged opening.'''

class SplitAlreadyOpenedError(RuntimeError):
    '''The test split was opened again without a stated reason.'''

class QueryCodeEncoder(Protocol):
    '''What the panel needs from an arm: queries and codes embedded in one space.'''

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        ...

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        ...

# -------------------------------------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------------------------------------

class OutcomePanel:
    '''
    The index-entry query splits, the six-digit candidates, and the log every read goes to.

    Args:
        role_rows: Every index entry with its text and role (``entry_id``, ``code``, ``text``,
            ``role``).
        candidates: The six-digit codes queries decode to, entry-less codes included.
        log: The selection log.
    '''

    def __init__(self, role_rows: pl.DataFrame, candidates: Sequence[str], log: SelectionLog):
        candidates = [str(code) for code in candidates]
        if len(set(candidates)) != len(candidates) or any(len(code) != 6 for code in candidates):
            raise ValueError('candidates must be distinct six-digit codes')
        validate_index_role_table(role_rows, candidates)
        self._rows = role_rows.select(INDEX_ROLE_COLUMNS).sort('entry_id')
        self.candidates: Tuple[str, ...] = tuple(sorted(candidates))
        self.log = log
        self.fingerprint = role_table_fingerprint(self._rows)
        self._test_open = False

    @classmethod
    def from_files(
        cls,
        index_roles_parquet: Union[str, Path],
        descriptions_parquet: Union[str, Path],
        log_path: Union[str, Path],
    ) -> 'OutcomePanel':
        '''
        The panel from ``data preprocess`` outputs: the index roles and the descriptions.

        Raises:
            ValueError: If the descriptions' examples channel is not built from the examples-role
                entries only. A descriptions file from before the roles existed holds every
                entry, held-out queries included, in its examples channel.
        '''

        roles = pl.read_parquet(index_roles_parquet)
        descriptions = pl.read_parquet(descriptions_parquet)
        verify_examples_channel(descriptions, roles)
        candidates = descriptions.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        return cls(roles, candidates.to_list(), SelectionLog(Path(log_path)))

    @property
    def entryless_candidates(self) -> Tuple[str, ...]:
        '''Candidates without index entries: decoded to, never queried (112130 and 541120).'''

        with_entries = set(self._rows.get_column('code').to_list())
        return tuple(code for code in self.candidates if code not in with_entries)

    def training_queries(self) -> pl.DataFrame:
        '''Training queries (``entry_id``, ``code``, ``text``); training data, so not logged.'''

        return self._split(IndexRole.TRAINING)

    def validation_queries(self, purpose: str) -> pl.DataFrame:
        '''Validation queries, logging the read.'''

        return self._read(IndexRole.VALIDATION, purpose)

    def open_test(self, purpose: str, *, reopen_reason: Optional[str] = None) -> None:
        '''
        Open the sealed test split for this panel object, logging the opening.

        Raises:
            SplitAlreadyOpenedError: If the log already records an opening of this split and
                no ``reopen_reason`` is given.
        '''

        prior = self.log.openings(OUTCOME_PANEL, self.fingerprint)
        reason = (reopen_reason or '').strip()
        if prior and not reason:
            first = prior[0]
            raise SplitAlreadyOpenedError(
                f'the outcome test split was opened at {first["time"]} for '
                f'{first["purpose"]!r}; opening it again needs reopen_reason'
            )
        self.log.append(
            SelectionEvent.REOPEN if prior else SelectionEvent.OPEN,
            panel=OUTCOME_PANEL,
            split=IndexRole.TEST.value,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=self._split(IndexRole.TEST).height,
            detail={'reason': reason} if reason else None,
        )
        self._test_open = True

    def test_queries(self, purpose: str) -> pl.DataFrame:
        '''
        Test queries, logging the read.

        Raises:
            SealedSplitError: If this panel object has not opened the test split.
        '''

        return self._read(IndexRole.TEST, purpose)

    def score(
        self,
        encoder: QueryCodeEncoder,
        split: Union[IndexRole, str],
        purpose: str,
        distance: Union[str, DistanceFn] = 'cosine',
    ) -> DecodingResult:
        '''Decode one split's queries over every candidate with the encoder, logging the read.'''

        name, _ = resolve_distance(distance)
        detail = {'encoder': type(encoder).__name__, 'distance': name}
        queries = self._read(IndexRole(split), purpose, detail)
        return score_decoding(
            encoder.encode_queries(queries.get_column('text').to_list()),
            queries.get_column('code').to_list(),
            encoder.encode_codes(list(self.candidates)),
            self.candidates,
            distance=distance,
            query_ids=queries.get_column('entry_id').to_list(),
        )

    def _split(self, role: IndexRole) -> pl.DataFrame:
        return self._rows.filter(pl.col('role') == role.value).select('entry_id', 'code', 'text')

    def _read(
        self,
        role: IndexRole,
        purpose: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> pl.DataFrame:
        if role not in (IndexRole.VALIDATION, IndexRole.TEST):
            raise ValueError(
                f'only validation and test splits are read as selections, not {role.value!r}'
            )
        if role is IndexRole.TEST and not self._test_open:
            raise SealedSplitError(
                'the outcome test split is sealed: call open_test(purpose) first, which is logged'
            )
        queries = self._split(role)
        self.log.append(
            SelectionEvent.READ,
            panel=OUTCOME_PANEL,
            split=role.value,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=queries.height,
            detail=detail,
        )
        return queries
