'''
Text→code's reference reads on the ITA validation part (redesign roadmap Stage 4, plan 20).

``read_reference`` reads one reference arm, ``lexical-stub`` or one seed of ``euclidean-d16``, on
the ITA panel's validation part with role ``reference`` (spec ``text-to-code-ita-panel``, 4.4 and
4.5). Its logged detail names the arm, the role and the run, and for ``euclidean-d16`` the seed,
the checkpoint's SHA-256 and the table's matrix fingerprint. On request it also reads the outcome
panel's validation split with the same encoder and the same detail, for the gap's Census-phrase
MRR (D1, P4). It stores the per-pair frame, and the outcome read's per-query scores, in the
artifact store, and returns an ``ItaReferenceRead``. It writes no other file.

Its refusals come before any log record: an arm outside ``REFERENCE_ARMS``, a seed on the stub, a
missing or non-positive seed on ``euclidean-d16``, a blank run or purpose, and a
``euclidean-d16`` encoder without its checkpoint or table identity. Stage 4's reads are reference
reads of reference arms, so Stage 5's ``check_ita_margin_first`` does not count them as candidate
reads. Nothing here opens the sealed part or chooses a reference.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from naics_embedder.decision.records import ArtifactRef
from naics_embedder.decision.scores import outcome_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.text_to_code import EUCLIDEAN_D16, LEXICAL_STUB, REFERENCE_ARMS
from naics_embedder.panels.ita import DEFAULT_DISTANCE, ItaPanel
from naics_embedder.panels.ita_evidence import ItaEvidence
from naics_embedder.panels.ita_split import ItaPart
from naics_embedder.panels.outcome import OutcomePanel, QueryCodeEncoder
from naics_embedder.supervision.schema import IndexRole

# Every Stage 4 read is a reference read (spec 4.5)
READ_ROLE = 'reference'

# The stored frames' file names
ITA_PER_PAIR_FILE = 'ita_per_pair.parquet'
OUTCOME_SCORES_FILE = 'outcome_scores.parquet'

# The identities a euclidean-d16 encoder carries, as ArmEncoder does, and their logged keys
_D16_IDENTITIES = {'checkpoint_sha256': 'checkpoint', 'table_fingerprint': 'table'}

# -------------------------------------------------------------------------------------------------
# Records
# -------------------------------------------------------------------------------------------------

class OutcomeBeside(BaseModel):
    '''
    The outcome panel's validation read beside an ITA read: the arm's Census-phrase scores (D1).

    Attributes:
        fingerprint: The outcome panel's role-table fingerprint.
        data_fingerprint: The validation split's data fingerprint.
        summary: The read's ``DecodingResult.summary``.
        scores: The stored ``outcome_scores`` frame of its per-query scores.
    '''

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    fingerprint: str
    data_fingerprint: str
    summary: Dict[str, Any]
    scores: ArtifactRef

class ItaReferenceRead(BaseModel):
    '''
    One reference arm's read of the ITA validation part, and of the outcome validation split beside
    it when asked (plan 20, P3).

    Attributes:
        kind: The record's kind.
        arm: ``lexical-stub`` or ``euclidean-d16``.
        seed: The ``euclidean-d16`` seed; None for the stub.
        run: The run id every Stage 4 read is logged under.
        purpose: Why the read happens, as the selection log records it.
        evidence: The pin the tables were checked against (C1).
        split_fingerprint: The ITA panel's split fingerprint.
        data_fingerprint: The validation part's data fingerprint: the table build read.
        headline: ``ItaResult.headline``: the pairs that are not copied.
        copied: ``ItaResult.copied``.
        phrasing: ``ItaResult.phrasing``.
        counts: ``ItaResult.counts``.
        per_pair: The stored ``ItaResult.per_pair`` frame.
        checkpoint_sha256: The ``euclidean-d16`` checkpoint's SHA-256; None for the stub.
        table_fingerprint: The ``euclidean-d16`` table's matrix fingerprint; None for the stub.
        outcome: The outcome read beside, or None.
        log_records: The records this read appended to the selection log, in order.
        created_at: When the record was made, in UTC.
    '''

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    kind: Literal['ita_reference_read'] = 'ita_reference_read'
    arm: Literal['lexical-stub', 'euclidean-d16']
    seed: Optional[int] = Field(ge=1)
    run: str
    purpose: str
    evidence: ItaEvidence
    split_fingerprint: str
    data_fingerprint: str
    headline: Dict[str, Any]
    copied: Dict[str, Any]
    phrasing: Dict[str, Any]
    counts: Dict[str, int]
    per_pair: ArtifactRef
    checkpoint_sha256: Optional[str]
    table_fingerprint: Optional[str]
    outcome: Optional[OutcomeBeside]
    log_records: List[Dict[str, Any]]
    created_at: datetime

# -------------------------------------------------------------------------------------------------
# The read
# -------------------------------------------------------------------------------------------------

def _check_request(
    encoder: QueryCodeEncoder,
    arm: str,
    seed: Optional[int],
    run: str,
    purpose: str,
) -> None:
    '''Refuse a read that would log under another arm, seed or identity, before any record.'''

    if arm not in REFERENCE_ARMS:
        raise ValueError(
            f'a Stage 4 read reads a reference arm, one of {list(REFERENCE_ARMS)}, not {arm!r}: '
            'check_ita_margin_first counts any other arm as a candidate read'
        )
    if arm == LEXICAL_STUB and seed is not None:
        raise ValueError(f'the stub has no seed, got seed {seed}')
    if arm == EUCLIDEAN_D16 and (seed is None or seed < 1):
        raise ValueError(f'a euclidean-d16 read needs a seed of 1 or more, got {seed}')
    for name, value in (('run', run), ('purpose', purpose)):
        if not value.strip():
            raise ValueError(f'a read needs a {name} that is not blank')
    if arm == EUCLIDEAN_D16:
        for attribute in _D16_IDENTITIES:
            if not getattr(encoder, attribute, None):
                raise ValueError(
                    f'a euclidean-d16 encoder must carry its {attribute}, which the read logs'
                )

def read_reference(
    panel: ItaPanel,
    encoder: QueryCodeEncoder,
    store: ArtifactStore,
    evidence: ItaEvidence,
    *,
    arm: str,
    seed: Optional[int],
    run: str,
    purpose: str,
    outcome_panel: Optional[OutcomePanel] = None,
) -> ItaReferenceRead:
    '''
    Read one reference arm on the ITA validation part, and on request the outcome validation split.

    Args:
        panel: The ITA panel, on the pinned run's tables.
        encoder: The arm: the stub's ``LexicalTrigramEncoder``, or one ``euclidean-d16`` seed's
            ``ArmEncoder``, which carries ``checkpoint_sha256`` and ``table_fingerprint``.
        store: The artifact store the per-pair and per-query frames go to.
        evidence: The pin the tables were checked against (C1), which the record carries.
        arm: ``lexical-stub`` or ``euclidean-d16``.
        seed: The ``euclidean-d16`` seed, 1 or more; None for the stub.
        run: The run id every Stage 4 read is logged under.
        purpose: Why the read happens; the selection log records it.
        outcome_panel: The outcome panel, when its validation split is read beside, on the same
            selection log.

    Returns:
        The record. Its ``log_records`` are the ITA read's record, then the outcome read's.

    Raises:
        ValueError: Before any log record, when the arm is not a reference arm, the stub has a
            seed, ``euclidean-d16`` has none or one below 1, the run or purpose is blank, or a
            ``euclidean-d16`` encoder lacks ``checkpoint_sha256`` or ``table_fingerprint``.
    '''

    _check_request(encoder, arm, seed, run, purpose)
    detail: Dict[str, Any] = {'arm': arm, 'role': READ_ROLE, 'run': run}
    if arm == EUCLIDEAN_D16:
        detail['seed'] = seed
        for attribute, key in _D16_IDENTITIES.items():
            detail[key] = getattr(encoder, attribute)
    result, ita_record = panel.score_logged(encoder, ItaPart.VALIDATION, purpose, detail=detail)
    log_records = [ita_record]
    per_pair = store.put_frame(result.per_pair, ITA_PER_PAIR_FILE)
    outcome = None
    if outcome_panel is not None:
        decoding, outcome_record = outcome_panel.score_logged(
            encoder,
            IndexRole.VALIDATION,
            purpose,
            distance=getattr(encoder, 'distance', DEFAULT_DISTANCE),
            detail=detail,
        )
        log_records.append(outcome_record)
        outcome = OutcomeBeside(
            fingerprint=outcome_panel.fingerprint,
            data_fingerprint=outcome_panel.data_fingerprint(IndexRole.VALIDATION),
            summary=dict(decoding.summary),
            scores=store.put_frame(outcome_scores(decoding.per_query), OUTCOME_SCORES_FILE),
        )
    return ItaReferenceRead(
        arm=arm,
        seed=seed,
        run=run,
        purpose=purpose,
        evidence=evidence,
        split_fingerprint=panel.fingerprint,
        data_fingerprint=result.data_fingerprint,
        headline=dict(result.headline),
        copied=dict(result.copied),
        phrasing=dict(result.phrasing),
        counts=dict(result.counts),
        per_pair=per_pair,
        checkpoint_sha256=detail.get('checkpoint'),
        table_fingerprint=detail.get('table'),
        outcome=outcome,
        log_records=log_records,
        created_at=datetime.now(timezone.utc),
    )
