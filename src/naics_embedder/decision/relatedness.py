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
- **Test.** For each judging family, Δ = the candidate's headline minus the reference's, and a
  candidate beats the reference on the family when the lower bound of the interval on Δ exceeds +δ.
  It goes forward when it beats on every family, at the level ``superiority_level(n)`` = 1 − 0.05/n
  for n the number of judging families when the test runs. The interval is the paired two-stage
  bootstrap of ``decision/resampling.py``: anchors are the units, with one item each, a candidate's
  seeds are nested within each resample, and the reference arms are deterministic, so each is one
  read. A family's unit draws are on the stream of its panel name and an arm's seed draws on
  ``(panel, arm)``, so a second family has streams of its own and the first family's draws do not
  change when it joins. ``relatedness_test`` takes the families from the shared keys of its
  mappings, in sorted order, and refuses before any draw: a ``min_seeds`` below 1; empty or
  differently keyed mappings; a margin that is not its family's or does not agree with itself; fewer
  than ``min_seeds`` seeds on a family; a result of another panel; results of more than one part;
  and a seed read on another table build than its family's reference. The same test runs on the
  sealed part with δ fixed on validation, so it compares neither the part nor the reference's build
  with the margin's.
- **Provenance.** ``check_relatedness_provenance`` enforces the first anti-goal in code, as
  ``decision/provenance.py`` does for economic candidates. A table's provenance may name only the
  codebook, the text-only table and the training families (``ita_area_mix``, optional) as inputs, so
  a provenance that names co-ownership, the very data the panel judges on, is refused. The rule is
  a whitelist of inputs and a fixed set of entries. It describes the provenance and checks no file:
  it reads nothing and imports nothing beyond ``typing``. Eight checks run in a fixed order, each
  refusing with a ``ValueError`` that names its check first and ends with the rule's tag: ``writer``,
  ``entries``, ``inputs``, ``codebook``, ``text_only``, ``runs``, ``checkpoint`` and ``seed``.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Literal, Mapping, NoReturn, Sequence, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from naics_embedder.decision.decide import read_time
from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.rule import superiority_level
from naics_embedder.decision.sweep import _log_records
from naics_embedder.panels.relatedness import RelatednessPanel
from naics_embedder.panels.relatedness_arms import RelatednessArm
from naics_embedder.panels.relatedness_scoring import RelatednessResult
from naics_embedder.panels.relatedness_split import RelatednessPart
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog

TREE = 'tree'
TEXT_ONLY = 'text-only'
TEXT_ONLY_CENTERED = 'text-only-centered'

# The reference arms, which is also the tie order: the simpler arm comes first (4.6)
REFERENCE_ARMS = (TREE, TEXT_ONLY, TEXT_ONLY_CENTERED)

# An embedding-data commit: 7 to 40 lowercase hexadecimal characters
COMMIT_PATTERN = r'^[0-9a-f]{7,40}$'

# The writer a relatedness table's provenance names, and the entries it writes (4.6)
RELATEDNESS_EXPORT = 'relatedness_export'
PROVENANCE_ENTRIES: FrozenSet[str] = frozenset(
    {
        'writer',
        'codes',
        'dimension',
        'inputs',
        'runs',
        'configuration',
        'seed',
        'checkpoint',
        'table_sha256',
        'generated_at',
        'library_versions',
    }
)

# The inputs a table may read: the codebook and the text-only table, which are required, and the
# training families of 4.1, which are optional (after G1: the area mix)
CODEBOOK_INPUTS = ('codebook', 'text_only')
TRAINING_FAMILY_INPUTS = ('ita_area_mix', )

_RULE = '(relatedness §4.6: codebook, text-only table and training families only)'
_FILE_ENTRIES = frozenset({'path', 'sha256'})

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
            ``REFERENCE_ARMS``, or ``embedding_data_commit`` is not 7 to 40 lowercase hexadecimal
            characters, before anything is read. A blank ``purpose`` is refused by the first read's
            log append, which writes nothing.
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

# -------------------------------------------------------------------------------------------------
# Test
# -------------------------------------------------------------------------------------------------

def headline_values(result: RelatednessResult) -> pl.DataFrame:
    '''
    The headline's values for the bootstrap: one row per anchor.

    Args:
        result: A part's ``RelatednessResult``.

    Returns:
        ``unit`` and ``item`` (both the anchor: anchors are the units, with one item each) and
        ``value`` (its ``ndcg_10``).
    '''

    return result.per_anchor.select(
        pl.col('anchor').alias('unit'),
        pl.col('anchor').alias('item'),
        pl.col('ndcg_10').alias('value'),
    )

def arm_replicates(
    items: PanelItems,
    panel: str,
    arm: str,
    seeds: Sequence[RelatednessResult],
    *,
    replicates: int,
    bootstrap_seed: int,
) -> Tuple[float, np.ndarray]:
    '''
    One arm's headline NDCG@10 as a point estimate and on each paired resample.

    Unit draws are on the stream ``panel``, shared by every arm of the family, and the arm's seed
    draws on ``(panel, arm)``. A second family has its own panel name, so its streams are its own.

    Args:
        items: The headline's items, shared by every arm compared.
        panel: The family's panel name.
        arm: The arm's name.
        seeds: The arm's results, one per seed.
        replicates: The number of resamples.
        bootstrap_seed: The bootstrap's seed.

    Returns:
        The mean over seeds of each seed's NDCG@10, and the NDCG@10 on each resample,
        (replicates,).

    Raises:
        ValueError: If the arm has no seed, or a seed's headline holds other items than ``items``.
    '''

    if not seeds:
        raise ValueError(f'arm {arm!r} has no seed')
    values = np.stack([items.values(headline_values(seed)) for seed in seeds])
    sums = items.sums(values)
    units = unit_draws(panel, len(items.units), replicates, bootstrap_seed)
    seed_counts = seed_draws(panel, arm, len(seeds), replicates, bootstrap_seed)
    return (
        point_statistic(sums, items.sizes),
        replicate_statistics(sums, items.sizes, units, seed_counts),
    )

@dataclass(frozen=True)
class FamilyDelta:
    '''
    A candidate's Δ on one judging family, before its interval is read.

    Attributes:
        panel: The family's panel name.
        reference: The family's reference arm.
        delta: Δ, the candidate's headline minus the reference's, as point estimates.
        replicates: Δ on each paired resample, oriented the same way.
        margin: The family's δ.
    '''

    panel: str
    reference: str
    delta: float
    replicates: np.ndarray
    margin: float

@dataclass(frozen=True)
class FamilyVerdict:
    '''
    A candidate's verdict on one judging family.

    Attributes:
        panel: The family's panel name.
        reference: The family's reference arm.
        delta: Δ, the candidate's headline minus the reference's, as point estimates.
        interval: The two-sided percentile interval on Δ at the verdict's level.
        margin: The family's δ.
        beats: Whether the interval's lower bound exceeds δ.
    '''

    panel: str
    reference: str
    delta: float
    interval: Tuple[float, float]
    margin: float
    beats: bool

@dataclass(frozen=True)
class RelatednessVerdict:
    '''
    A candidate's verdict against the reference on every judging family.

    Attributes:
        candidate: The candidate arm.
        level: The intervals' level, ``superiority_level(n)`` for n judging families.
        families: Each family's verdict, in the order the families were given.
        beats: Whether the candidate beats the reference by δ on every family: only then does it
            go forward.
    '''

    candidate: str
    level: float
    families: Tuple[FamilyVerdict, ...]
    beats: bool

def beat_by_margin(candidate: str, deltas: Sequence[FamilyDelta]) -> RelatednessVerdict:
    '''
    Whether the candidate beats the reference by δ on every judging family (spec 4.6).

    The number of judging families is ``len(deltas)``, so the level is ``superiority_level(n)``
    and is the same on every family. A family beats when the lower bound of its interval exceeds its
    own δ, and the candidate beats when every family does.

    Args:
        candidate: The candidate arm.
        deltas: Each judging family's Δ, its replicates and its δ.

    Returns:
        The verdict, its families in the order given.

    Raises:
        ValueError: If there is no family, or a panel repeats.
    '''

    if not deltas:
        raise ValueError('beat_by_margin needs at least one judging family, got none')
    panels = [delta.panel for delta in deltas]
    repeated = sorted({name for name in panels if panels.count(name) > 1})
    if repeated:
        raise ValueError(f'a judging family repeats: {repeated}')
    level = superiority_level(len(deltas))
    families = []
    for delta in deltas:
        interval = percentile_interval(delta.replicates, level)
        families.append(
            FamilyVerdict(
                panel=delta.panel,
                reference=delta.reference,
                delta=float(delta.delta),
                interval=interval,
                margin=float(delta.margin),
                beats=bool(interval[0] > delta.margin),
            )
        )
    return RelatednessVerdict(
        candidate=candidate,
        level=level,
        families=tuple(families),
        beats=all(family.beats for family in families),
    )

def relatedness_test(
    candidate: str,
    candidate_seeds: Mapping[str, Sequence[RelatednessResult]],
    references: Mapping[str, RelatednessResult],
    margins: Mapping[str, RelatednessMarginRecord],
    *,
    min_seeds: int,
    replicates: int,
    bootstrap_seed: int,
) -> RelatednessVerdict:
    '''
    A candidate's verdict against the reference on paired resamples of every judging family (spec
    4.6).

    The families are the mappings' shared keys, each a panel name, taken in sorted order so that
    the verdict does not depend on the mappings' order. A family's reference arm is its margin's
    ``reference``, its items come from the reference's result, and Δ is the candidate's headline
    minus the reference's, the reference being one deterministic read. The same test runs on the
    sealed part with δ fixed on validation, so it compares neither the part nor the reference's
    table build with the margin's.

    Args:
        candidate: The candidate arm.
        candidate_seeds: Its results on each family, one per seed, keyed by panel name.
        references: Each family's reference result, keyed by panel name.
        margins: Each family's margin record, keyed by panel name.
        min_seeds: The fewest seeds a candidate has on each family (spec 4.6: 5).
        replicates: The number of resamples.
        bootstrap_seed: The bootstrap's seed.

    Returns:
        The verdict, its families in sorted order of their panel names.

    Raises:
        ValueError: Before any draw, in this order: if ``min_seeds`` is below 1; if any mapping is
            empty, or the three have not the same keys; if a margin is for another panel than its
            key, or fails ``check_relatedness_margin``; if a family has fewer than ``min_seeds``
            candidate seeds, naming the family, the count and the floor; if a result is of another
            panel than its key; if the results are of more than one part; or if a candidate seed
            was read on another data fingerprint than its family's reference, naming the family and
            the seed's position. Then, if a seed is scored on other anchors than the reference.
    '''

    if min_seeds < 1:
        raise ValueError(f'min_seeds must be at least 1, got {min_seeds}')
    inputs = {'candidate_seeds': candidate_seeds, 'references': references, 'margins': margins}
    empty = [name for name, mapping in inputs.items() if not mapping]
    if empty:
        raise ValueError(
            f"the test needs at least one judging family, but no family is in {', '.join(empty)}"
        )
    if not set(candidate_seeds) == set(references) == set(margins):
        listed = ', '.join(
            f'{name} {sorted(map(str, mapping))}' for name, mapping in inputs.items()
        )
        raise ValueError(
            f'candidate_seeds, references and margins must name the same families: {listed}'
        )
    keys = sorted(candidate_seeds)
    for key in keys:
        margin = margins[key]
        if margin.panel != key:
            raise ValueError(
                f'the margin {margin.name!r} is for panel {margin.panel!r}, not for the family {key!r}'
            )
        check_relatedness_margin(margin)
    for key in keys:
        if len(candidate_seeds[key]) < min_seeds:
            raise ValueError(
                f'the family {key!r} has {len(candidate_seeds[key])} candidate seeds, fewer than '
                f'the {min_seeds} that the test needs'
            )
    for key in keys:
        results = [('reference', references[key])]
        results += [
            (f'candidate seed {position}', seed)
            for position, seed in enumerate(candidate_seeds[key])
        ]
        for what, result in results:
            if result.panel != key:
                raise ValueError(
                    f'the {what} of the family {key!r} was read on panel {result.panel!r}'
                )
    parts = {references[key].part for key in keys}
    parts |= {seed.part for key in keys for seed in candidate_seeds[key]}
    if len(parts) > 1:
        raise ValueError(f'the results are of more than one part: {sorted(parts)}')
    for key in keys:
        expected = references[key].data_fingerprint
        for position, seed in enumerate(candidate_seeds[key]):
            if seed.data_fingerprint != expected:
                raise ValueError(
                    f'the family {key!r} candidate seed {position} was read on data fingerprint '
                    f"{seed.data_fingerprint}, not the reference's {expected}: the test needs the "
                    'same table build'
                )
    settings = {'replicates': replicates, 'bootstrap_seed': bootstrap_seed}
    deltas = []
    for key in keys:
        reference = references[key]
        arm = margins[key].reference
        items = PanelItems.from_values(headline_values(reference))
        candidate_point, candidate_draws = arm_replicates(
            items, key, candidate, candidate_seeds[key], **settings
        )
        reference_point, reference_draws = arm_replicates(items, key, arm, [reference], **settings)
        deltas.append(
            FamilyDelta(
                panel=key,
                reference=arm,
                delta=candidate_point - reference_point,
                replicates=candidate_draws - reference_draws,
                margin=margins[key].margin,
            )
        )
    return beat_by_margin(candidate, deltas)

# -------------------------------------------------------------------------------------------------
# Provenance
# -------------------------------------------------------------------------------------------------

def check_relatedness_provenance(
    content: Mapping[str, Any], *, codebook_sha256: str, text_only_sha256: str
) -> None:
    '''
    Require a relatedness table's provenance to name only the codebook, the text-only table and the
    training families as inputs (spec 4.6).

    The rule is a whitelist, so an input that names co-ownership, or any other input outside
    ``CODEBOOK_INPUTS`` and ``TRAINING_FAMILY_INPUTS``, is refused by that fact. It describes the
    provenance and checks no file. The checks run in this order, and each refusal names the check
    that refused (``writer``, ``entries``, ``inputs``, ``codebook``, ``text_only``, ``runs``,
    ``checkpoint`` or ``seed``):

    1. ``writer``: ``content`` is a mapping whose ``writer`` is ``RELATEDNESS_EXPORT``.
    2. ``entries``: the entries are exactly ``PROVENANCE_ENTRIES``. An unknown entry, such as
       ``co_ownership``, is refused by name.
    3. ``inputs``: a mapping that holds every name of ``CODEBOOK_INPUTS``, holds no name outside
       ``CODEBOOK_INPUTS`` and ``TRAINING_FAMILY_INPUTS``, and maps each name to exactly a string
       ``path`` and a string ``sha256``.
    4. ``codebook`` and ``text_only``: each input's ``sha256`` equals its argument.
    5. ``runs``: a mapping keyed by exactly the training-family inputs present in ``inputs``, each
       value a string that holds a character other than whitespace.
    6. ``checkpoint``: exactly a string ``path`` and a string ``sha256``.
    7. ``seed``: an ``int`` that is not a ``bool``.

    Args:
        content: The provenance's JSON content.
        codebook_sha256: The SHA-256 of the codebook the decision reads.
        text_only_sha256: The SHA-256 of the text-only table the decision reads.

    Raises:
        ValueError: At the first check that fails, with the check's name first and
            ``(relatedness §4.6: codebook, text-only table and training families only)`` last.
            An earlier problem is always the one reported, and a later one never hides behind it.
    '''

    if not isinstance(content, Mapping):
        _refuse('writer', f'the provenance is not a mapping, so it has no writer: {content!r}')
    if 'writer' not in content:
        _refuse(
            'writer',
            f"the provenance has no 'writer' entry, and the writer must be {RELATEDNESS_EXPORT!r}"
        )
    if content['writer'] != RELATEDNESS_EXPORT:
        _refuse('writer', f"the writer is {content['writer']!r}, not {RELATEDNESS_EXPORT!r}")
    _check_entries(content)
    inputs = _check_inputs(content['inputs'])
    if inputs['codebook']['sha256'] != codebook_sha256:
        _refuse(
            'codebook', f"the codebook input has sha256 {inputs['codebook']['sha256']!r}, not the "
            f'{codebook_sha256!r} that the decision reads'
        )
    if inputs['text_only']['sha256'] != text_only_sha256:
        _refuse(
            'text_only', f"the text_only input has sha256 {inputs['text_only']['sha256']!r}, not "
            f'the {text_only_sha256!r} that the decision reads'
        )
    _check_runs(content['runs'], [name for name in TRAINING_FAMILY_INPUTS if name in inputs])
    _check_file(content['checkpoint'], 'checkpoint', 'the checkpoint')
    seed = content['seed']
    if isinstance(seed, bool) or not isinstance(seed, int):
        _refuse('seed', f'the seed is {seed!r}, not an int')

def _check_entries(content: Mapping[str, Any]) -> None:
    '''Require exactly ``PROVENANCE_ENTRIES``, naming each entry that is extra or missing.'''

    extra = sorted(map(str, set(content) - PROVENANCE_ENTRIES))
    missing = sorted(PROVENANCE_ENTRIES - set(content))
    if extra or missing:
        problems = []
        if extra:
            problems.append(f'the entries {extra}, which the writer does not write')
        if missing:
            problems.append(f'no entry {missing}, which the writer writes')
        _refuse('entries', f'the provenance has {" and has ".join(problems)}')

def _check_inputs(inputs: Any) -> Mapping[str, Any]:
    '''
    Require the inputs to be the codebook's, the text-only table's and optional training families,
    each a path and a sha256. Returns them.
    '''

    if not isinstance(inputs, Mapping):
        _refuse('inputs', f'inputs is not a mapping of input names to files: {inputs!r}')
    allowed = CODEBOOK_INPUTS + TRAINING_FAMILY_INPUTS
    extra = sorted(map(str, set(inputs) - set(allowed)))
    missing = sorted(set(CODEBOOK_INPUTS) - set(inputs))
    if extra or missing:
        problems = []
        if extra:
            problems.append(f'the inputs {extra}, which are outside {list(allowed)}')
        if missing:
            problems.append(f'no input {missing}, which the table must read')
        _refuse('inputs', f'the provenance names {" and names ".join(problems)}')
    for name in allowed:
        if name in inputs:
            _check_file(inputs[name], 'inputs', f'the input {name!r}')
    return inputs

def _check_runs(runs: Any, present: List[str]) -> None:
    '''Require the runs to be keyed by exactly the ``present`` training families, each non-blank.'''

    if not isinstance(runs, Mapping):
        _refuse('runs', f'runs is not a mapping of training-family inputs to runs: {runs!r}')
    if set(runs) != set(present):
        _refuse(
            'runs', f'the runs are keyed by {sorted(map(str, runs))}, but the training-family '
            f'inputs present are {present}'
        )
    for name in present:
        if not isinstance(runs[name], str) or not runs[name].strip():
            _refuse('runs', f'the run of {name!r} is not a non-blank string: {runs[name]!r}')

def _check_file(value: Any, check: str, what: str) -> None:
    '''Require a file reference: a mapping of a ``path`` and a ``sha256``, both strings.'''

    if (
        not isinstance(value, Mapping) or set(value) != _FILE_ENTRIES or not all(
            isinstance(value[field], str) for field in _FILE_ENTRIES
        )
    ):
        _refuse(check, f"{what} is not exactly a string 'path' and a string 'sha256'")

def _refuse(check: str, detail: str) -> NoReturn:
    raise ValueError(f'{check}: {detail} {_RULE}')
