'''
Relatedness's decision on the co-ownership panel (spec ``relatedness-artifact``, 4.6).

- **Reference.** The reference arms are ``tree``, ``text-only`` and ``text-only-centered``: the
  tree, and the text-only table under plain and centered cosine. None has seeds, so each is one
  read. The reference is the arm with the highest validation headline NDCG@10 as a point estimate.
  A tie goes to the arm earlier in ``REFERENCE_ARMS``, the simpler one.
- **Margin.** δ = r × the chosen reference's validation headline, with r = 0.05, fixed from the
  margin record's own fresh reference reads under a new run id, before any candidate read. The
  record holds the family's panel name (P4), so the check keys on the record and a second family's
  record is judged on its own panel's reads. ``check_relatedness_margin`` refuses a record that does
  not agree with itself: its arms, its reference, its δ and its reads, each reference arm read
  exactly once (P9). ``check_relatedness_margin_first`` runs that check, then refuses a record when
  the log holds a validation read of its panel made before it was fixed that is not a reference
  read: one with role ``candidate``, with no role, or with role ``reference`` but an arm outside
  ``REFERENCE_ARMS``. Training-monitor reads count as candidate reads. Reads of the sealed part,
  of another panel and records that are not reads do not count.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from naics_embedder.decision.decide import read_time
from naics_embedder.decision.sweep import _log_records
from naics_embedder.panels.relatedness import RelatednessPanel
from naics_embedder.panels.relatedness_arms import RelatednessArm
from naics_embedder.panels.relatedness_split import RelatednessPart
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog

TREE = 'tree'
TEXT_ONLY = 'text-only'
TEXT_ONLY_CENTERED = 'text-only-centered'

# The reference arms, which is also the tie order: the simpler arm comes first (4.6)
REFERENCE_ARMS = (TREE, TEXT_ONLY, TEXT_ONLY_CENTERED)

# An embedding-data commit: 7 to 40 lowercase hexadecimal characters
COMMIT_PATTERN = r'^[0-9a-f]{7,40}$'

# -------------------------------------------------------------------------------------------------
# Reference and margin
# -------------------------------------------------------------------------------------------------

def choose_reference(headlines: Mapping[str, float]) -> str:
    '''
    The reference arm: the highest validation headline, the earlier arm of ``REFERENCE_ARMS`` on a
    tie.

    Args:
        headlines: Each reference arm's validation headline NDCG@10, keyed by exactly
            ``REFERENCE_ARMS``.

    Returns:
        The arm with the highest headline. Of arms with the same highest headline, the one that
        comes first in ``REFERENCE_ARMS``.

    Raises:
        ValueError: If the keys are not exactly ``REFERENCE_ARMS``, or a headline is not finite.
    '''

    if set(headlines) != set(REFERENCE_ARMS):
        raise ValueError(
            f'the reference arms are {list(REFERENCE_ARMS)}, not {sorted(map(str, headlines))}'
        )
    for arm in REFERENCE_ARMS:
        if not math.isfinite(headlines[arm]):
            raise ValueError(f'arm {arm!r} has a headline that is not finite: {headlines[arm]}')
    return max(REFERENCE_ARMS, key=lambda arm: headlines[arm])

class RelatednessMarginRecord(BaseModel):
    '''
    One judging family's δ, fixed from the reference arms' own validation reads before any
    candidate read (spec 4.6).

    Attributes:
        panel: The family's panel name in the selection log, which the margin-first check keys on.
        name: Names the margin in the verdicts that use it.
        r: δ as a multiple of the reference's headline.
        run_id: The run the margin's reads are logged under.
        reference_headlines: Each reference arm's validation headline NDCG@10.
        reference: The chosen reference arm (``choose_reference``).
        margin: δ, ``r`` times the reference's headline.
        split_fingerprint: The panel's split fingerprint.
        data_fingerprint: The validation part's data fingerprint.
        embedding_data_commit: The embedding-data commit the tables come from.
        table_sha256: The two tables' SHA-256 values.
        log_records: The margin's reads, as the selection log appended them.
        fixed_at: When δ was fixed, after its reads.
    '''

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    kind: Literal['relatedness_margin'] = 'relatedness_margin'
    panel: str
    name: str
    r: float = Field(gt=0)
    run_id: str
    reference_headlines: Dict[str, float]
    reference: str
    margin: float
    split_fingerprint: str
    data_fingerprint: str
    embedding_data_commit: str = Field(pattern=COMMIT_PATTERN)
    table_sha256: Dict[str, str]
    log_records: List[Dict[str, Any]]
    fixed_at: datetime

def fix_relatedness_margin(
    panel: RelatednessPanel,
    reference_arms: Mapping[str, RelatednessArm],
    r: float,
    name: str,
    *,
    embedding_data_commit: str,
    purpose: str,
) -> RelatednessMarginRecord:
    '''
    Read each reference arm once on the validation part, and fix δ as ``r`` times the chosen
    reference's headline (spec 4.6).

    Every read is logged under one fresh run id with role ``reference``. An arm's headline is the
    ``ndcg_10`` of its validation result.

    Args:
        panel: The relatedness panel.
        reference_arms: Each reference arm, keyed by exactly ``REFERENCE_ARMS``.
        r: δ as a multiple of the reference's headline (spec 4.6: 0.05).
        name: Names the margin in the verdicts that use it.
        embedding_data_commit: The embedding-data commit the panel's tables come from.
        purpose: Why the reads happen, which the selection log records.

    Returns:
        The margin record, fixed after its reads.

    Raises:
        ValueError: If ``r`` is not a positive finite number, the arms are not exactly
            ``REFERENCE_ARMS``, ``embedding_data_commit`` is not 7 to 40 lowercase hexadecimal
            characters, or ``purpose`` is blank, before anything is read.
    '''

    if not (math.isfinite(r) and r > 0):
        raise ValueError(f'the margin multiple r must be positive and finite, got {r}')
    if set(reference_arms) != set(REFERENCE_ARMS):
        raise ValueError(
            f'the reference arms are {list(REFERENCE_ARMS)}, '
            f'not {sorted(map(str, reference_arms))}'
        )
    if not re.fullmatch(COMMIT_PATTERN, embedding_data_commit):
        raise ValueError(
            'embedding_data_commit must be 7 to 40 lowercase hexadecimal characters, '
            f'got {embedding_data_commit!r}'
        )
    run_id = f'relatedness-margin/{uuid.uuid4().hex}'
    reference_headlines: Dict[str, float] = {}
    for arm in REFERENCE_ARMS:
        detail = {'arm': arm, 'role': 'reference', 'run': run_id}
        result = panel.score(
            reference_arms[arm], RelatednessPart.VALIDATION, purpose, detail=detail
        )
        reference_headlines[arm] = result.headline['ndcg_10']
    reference = choose_reference(reference_headlines)
    return RelatednessMarginRecord(
        panel=panel.name,
        name=name,
        r=r,
        run_id=run_id,
        reference_headlines=reference_headlines,
        reference=reference,
        margin=r * reference_headlines[reference],
        split_fingerprint=panel.fingerprint,
        data_fingerprint=panel.data_fingerprint(RelatednessPart.VALIDATION),
        embedding_data_commit=embedding_data_commit,
        table_sha256=dict(panel.table_sha256),
        log_records=_log_records([panel.log], run_id),
        fixed_at=datetime.now(timezone.utc),
    )

def check_relatedness_margin(margin: RelatednessMarginRecord) -> None:
    '''
    Require the margin record to agree with itself (spec 4.6).

    Its headlines are keyed by exactly ``REFERENCE_ARMS``, its reference is the one
    ``choose_reference`` picks from them, and δ is exactly ``r`` times the reference's headline. It
    holds at least one read. Each read is a read of the validation part of ``margin.panel`` under
    its run id, with role ``reference``, an arm in ``REFERENCE_ARMS``, its split fingerprint and its
    data fingerprint. Each reference arm is read exactly once (P9). The record holds no per-anchor
    score, so each headline is not recomputed.

    Args:
        margin: The margin record.

    Raises:
        ValueError: If the headlines are not keyed by exactly ``REFERENCE_ARMS`` or one is not
            finite; if the reference is not the one they choose; if δ is not ``r`` times the
            reference's headline; if the record holds no read; if a read is not a read of the
            validation part of ``margin.panel``, or names another run, a role other than
            ``reference``, an arm outside ``REFERENCE_ARMS``, another split fingerprint or another
            data fingerprint; or if a reference arm is read other than once. The message names the
            record and what disagrees.
    '''

    name = f'the margin {margin.name!r}'
    if set(margin.reference_headlines) != set(REFERENCE_ARMS):
        raise ValueError(
            f'{name} holds headlines for {sorted(map(str, margin.reference_headlines))}, not the '
            f'reference arms {list(REFERENCE_ARMS)}'
        )
    chosen = choose_reference(margin.reference_headlines)
    if margin.reference != chosen:
        raise ValueError(
            f'{name} names the reference {margin.reference!r}, but its headlines choose {chosen!r}'
        )
    expected = margin.r * margin.reference_headlines[margin.reference]
    if margin.margin != expected:
        raise ValueError(
            f"{name} holds δ = {margin.margin}, not r × the reference's headline, {margin.r} × "
            f'{margin.reference_headlines[margin.reference]} = {expected}'
        )
    if not margin.log_records:
        raise ValueError(f'{name} holds no read')
    for position, record in enumerate(margin.log_records):
        detail = record.get('detail', {})
        read = f'{name}: its read {position}'
        logged = (record.get('event'), record.get('panel'), record.get('split'))
        if logged != (SelectionEvent.READ.value, margin.panel, RelatednessPart.VALIDATION.value):
            raise ValueError(
                f'{read} is not a read of the validation part of {margin.panel!r}: {logged}'
            )
        if detail.get('run') != margin.run_id:
            raise ValueError(
                f"{read} was logged under run {detail.get('run')!r}, not {margin.run_id!r}"
            )
        if detail.get('role') != 'reference':
            raise ValueError(f"{read} has role {detail.get('role')!r}, not 'reference'")
        if detail.get('arm') not in REFERENCE_ARMS:
            raise ValueError(
                f"{read} reads arm {detail.get('arm')!r}, not one of {list(REFERENCE_ARMS)}"
            )
        if record.get('fingerprint') != margin.split_fingerprint:
            raise ValueError(
                f"{read} was read on split fingerprint {record.get('fingerprint')}, not "
                f'{margin.split_fingerprint}'
            )
        if detail.get('data_fingerprint') != margin.data_fingerprint:
            raise ValueError(
                f"{read} was read on data fingerprint {detail.get('data_fingerprint')}, not "
                f'{margin.data_fingerprint}'
            )
    reads = Counter(record['detail']['arm'] for record in margin.log_records)
    for arm in REFERENCE_ARMS:
        if reads[arm] != 1:
            raise ValueError(f'{name}: arm {arm!r} is read {reads[arm]} times, not once')

def check_relatedness_margin_first(margin: RelatednessMarginRecord, log: SelectionLog) -> None:
    '''
    Require the margin's reads to precede it, and no candidate read of its panel's validation part
    to precede it (spec 4.6).

    The record must first pass ``check_relatedness_margin``. A read counts as a candidate read
    unless its role is ``reference`` and its arm is in ``REFERENCE_ARMS``: a read with role
    ``candidate``, with no role, or with role ``reference`` and another arm or none counts, and
    training-monitor reads count. Reads of the sealed part, reads of other panels and records that
    are not reads do not count, so a second family's record is judged on its own panel's reads. A
    read at the instant ``fixed_at`` is not before it.

    Args:
        margin: The margin record.
        log: The selection log the reads went to.

    Raises:
        ValueError: If the record fails ``check_relatedness_margin``; if ``fixed_at`` names no
            timezone; if one of the margin's reads is later than ``fixed_at``; or if the log holds
            a validation read of ``margin.panel`` before ``fixed_at`` that is not a reference read
            of a reference arm, naming its purpose and time.
    '''

    check_relatedness_margin(margin)
    name = f'the margin {margin.name!r}'
    if margin.fixed_at.utcoffset() is None:
        raise ValueError(f'{name} was fixed at {margin.fixed_at.isoformat()}, with no timezone')
    for record in margin.log_records:
        time = read_time(record, f'{name}: its read of {record.get("detail", {}).get("arm")}')
        if time > margin.fixed_at:
            raise ValueError(
                f'{name} was fixed at {margin.fixed_at.isoformat()}, before its read at '
                f'{time.isoformat()} (spec 4.6)'
            )
    for record in log.records():
        if (record.get('event'), record.get('panel'), record.get('split')) != (
            SelectionEvent.READ.value, margin.panel, RelatednessPart.VALIDATION.value
        ):
            continue
        detail = record.get('detail', {})
        if detail.get('role') == 'reference' and detail.get('arm') in REFERENCE_ARMS:
            continue
        time = read_time(record, f'the read {record.get("purpose")!r}')
        if time < margin.fixed_at:
            raise ValueError(
                f'{name} was fixed at {margin.fixed_at.isoformat()}, after the candidate read '
                f'{record.get("purpose")!r} at {time.isoformat()}: the margin comes first '
                '(spec 4.6)'
            )
