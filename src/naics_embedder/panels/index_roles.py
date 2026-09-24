'''
Index-entry roles for the outcome panel (Req 3; roadmap D4).

Every entry of the Census NAICS index file holds exactly one role: examples-channel text, or a
training, validation or test query. The assignment is made once, frozen in a committed table
(``conf/data/index_roles.csv``), and consumed by hash afterwards; regeneration is never how the
sealed splits are preserved.

- **Eligibility.** Validation and test queries are drawn only from entries that no training text
  can leak: an entry is withheld from both held-out splits when it matches, exactly or as a
  near-duplicate (``leakage``), any title, description, exclusion text, fallback examples text
  or any other index entry. Checking against every other entry, not only those that end up as
  training text, makes the held-out splits leak-free whatever the assignment.
- **Quotas.** Per code, role counts are largest-remainder quotas of the configured fractions,
  with remainder ties broken by a seeded draw. Every code keeps at least ``examples_floor``
  examples-channel entries, and held-out quotas beyond the code's eligible entries move to
  training, one at a time from the larger of the two held-out quotas (test on a tie).
- **Seeds.** Each code draws from its own generator, seeded by (seed, code), so a code's roles do
  not depend on any other code.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import polars as pl

from naics_embedder.panels.leakage import (
    EXAMPLES_SEPARATOR,
    NEAR_DUPLICATE_MIN_JACCARD,
    find_leakage,
    find_leakage_within,
    training_text_segments,
)
from naics_embedder.supervision.artifacts import INDEX_ROLE_COLUMNS
from naics_embedder.supervision.schema import IndexRole

ROLE_ORDER = (IndexRole.EXAMPLES, IndexRole.TRAINING, IndexRole.VALIDATION, IndexRole.TEST)
ROLE_TABLE_SCHEMA = {'entry_id': pl.Int64, 'code': pl.Utf8, 'role': pl.Utf8}

# -------------------------------------------------------------------------------------------------
# Fractions and per-code quotas
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleFractions:
    '''Exact role fractions; non-negative and summing to one.'''

    examples: Fraction
    training: Fraction
    validation: Fraction
    test: Fraction

    def __post_init__(self) -> None:
        values = [self.of(role) for role in ROLE_ORDER]
        if any(value < 0 for value in values) or sum(values) != 1:
            raise ValueError(
                'role fractions must be non-negative and sum to 1, got '
                f'{[str(value) for value in values]}'
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> 'RoleFractions':
        '''Exact fractions from decimal config values (0.35 becomes 7/20, not a binary float).'''

        return cls(**{role.value: Fraction(str(mapping[role.value])) for role in ROLE_ORDER})

    def of(self, role: IndexRole) -> Fraction:
        return getattr(self, role.value)

def allocate_role_counts(
    n_entries: int,
    n_eligible: int,
    fractions: RoleFractions,
    tie_break: Sequence[float],
    examples_floor: int = 1,
) -> Dict[IndexRole, int]:
    '''
    Role counts for one code with ``n_entries`` entries, ``n_eligible`` of them held-out-eligible.

    Args:
        n_entries: The code's index entries.
        n_eligible: Entries that may become validation or test queries.
        fractions: Target role fractions.
        tie_break: One draw per role, in ``ROLE_ORDER``; the smaller draw wins a remainder tie.
        examples_floor: Minimum examples-role entries (capped at ``n_entries``).

    Returns:
        Counts per role, summing to ``n_entries``.
    '''

    if not 0 <= n_eligible <= n_entries:
        raise ValueError(f'eligible count {n_eligible} must lie in [0, {n_entries}]')
    if len(tie_break) != len(ROLE_ORDER):
        raise ValueError(f'tie_break needs one draw per role, got {len(tie_break)}')

    exact = [fractions.of(role) * n_entries for role in ROLE_ORDER]
    counts = [math.floor(value) for value in exact]
    remainders = [value - count for value, count in zip(exact, counts)]
    order = sorted(range(len(ROLE_ORDER)), key=lambda k: (-remainders[k], tie_break[k]))
    for k in order[:n_entries - sum(counts)]:
        counts[k] += 1
    by_role = dict(zip(ROLE_ORDER, counts))

    while by_role[IndexRole.EXAMPLES] < min(examples_floor, n_entries):
        if by_role[IndexRole.TRAINING]:
            donor = IndexRole.TRAINING
        elif by_role[IndexRole.TEST] >= by_role[IndexRole.VALIDATION]:
            donor = IndexRole.TEST
        else:
            donor = IndexRole.VALIDATION
        by_role[donor] -= 1
        by_role[IndexRole.EXAMPLES] += 1

    while by_role[IndexRole.VALIDATION] + by_role[IndexRole.TEST] > n_eligible:
        if by_role[IndexRole.TEST] >= by_role[IndexRole.VALIDATION]:
            by_role[IndexRole.TEST] -= 1
        else:
            by_role[IndexRole.VALIDATION] -= 1
        by_role[IndexRole.TRAINING] += 1
    return by_role

# -------------------------------------------------------------------------------------------------
# Eligibility and assignment
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EligibilityReport:
    '''
    Which entries may be held out, and why the others are withheld.

    The ``*_static`` and ``*_entry`` counts overlap; ``withheld_exact`` (any exact match) and
    ``withheld_near_duplicate`` (a near-duplicate and no exact match) partition ``withheld``.
    '''

    eligible: np.ndarray
    counts: Dict[str, int]

def held_out_eligibility(
    entries: pl.DataFrame,
    static_segments: Sequence[str],
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> EligibilityReport:
    '''
    Flag the entries that no training text can leak.

    Args:
        entries: Index entries (``entry_id``, ``code``, ``text``).
        static_segments: Every code's title, description, exclusion and fallback examples
            segments (``leakage.training_text_segments`` without index-derived examples).
        min_jaccard: Near-duplicate threshold.
    '''

    texts = entries.get_column('text').to_list()
    static = find_leakage(texts, static_segments, min_jaccard=min_jaccard)
    within = find_leakage_within(texts, min_jaccard=min_jaccard)
    exact = static.exact | within.exact
    near_duplicate_only = (static.near_duplicate | within.near_duplicate) & ~exact
    eligible = ~(exact | near_duplicate_only)
    counts = {
        'entries': len(texts),
        'exact_static': int(static.exact.sum()),
        'near_duplicate_static': int(static.near_duplicate.sum()),
        'exact_entry': int(within.exact.sum()),
        'near_duplicate_entry': int(within.near_duplicate.sum()),
        'withheld_exact': int(exact.sum()),
        'withheld_near_duplicate': int(near_duplicate_only.sum()),
        'withheld': int((~eligible).sum()),
        'eligible': int(eligible.sum()),
    }
    return EligibilityReport(eligible=eligible, counts=counts)

def assign_index_roles(
    entries: pl.DataFrame,
    eligible: np.ndarray,
    fractions: RoleFractions,
    seed: int,
    examples_floor: int = 1,
) -> pl.DataFrame:
    '''
    Give every entry exactly one role, per code and stratified.

    Args:
        entries: Index entries (``entry_id``, ``code``, ...), one row per entry.
        eligible: Held-out eligibility, aligned with ``entries``.
        fractions: Target role fractions.
        seed: Base seed; each code draws from ``np.random.default_rng([seed, int(code)])``.
        examples_floor: Minimum examples-role entries per code.

    Returns:
        ``entry_id``, ``code``, ``role`` sorted by ``entry_id``.
    '''

    if len(eligible) != entries.height:
        raise ValueError(f'eligible has {len(eligible)} flags for {entries.height} entries')
    frame = entries.select('entry_id', 'code').with_columns(
        eligible=pl.Series(np.asarray(eligible, dtype=bool))
    )
    assigned: List[Tuple[int, str, str]] = []
    for (code, ), group in frame.sort('entry_id').group_by('code', maintain_order=True):
        rng = np.random.default_rng([seed, int(code)])
        tie_break = rng.random(len(ROLE_ORDER))
        entry_ids = group.get_column('entry_id').to_numpy()
        flags = group.get_column('eligible').to_numpy()
        counts = allocate_role_counts(
            len(entry_ids), int(flags.sum()), fractions, tie_break, examples_floor
        )
        held_out = rng.permutation(entry_ids[flags])
        n_validation = counts[IndexRole.VALIDATION]
        n_held_out = n_validation + counts[IndexRole.TEST]
        rest = rng.permutation(np.concatenate([held_out[n_held_out:], entry_ids[~flags]]))
        n_examples = counts[IndexRole.EXAMPLES]
        for ids, role in (
            (held_out[:n_validation], IndexRole.VALIDATION),
            (held_out[n_validation:n_held_out], IndexRole.TEST),
            (rest[:n_examples], IndexRole.EXAMPLES),
            (rest[n_examples:], IndexRole.TRAINING),
        ):
            assigned.extend((int(entry_id), str(code), role.value) for entry_id in ids)
    return pl.DataFrame(assigned, schema=ROLE_TABLE_SCHEMA, orient='row').sort('entry_id')

# -------------------------------------------------------------------------------------------------
# The frozen role table
# -------------------------------------------------------------------------------------------------

def _role_table_csv(roles: pl.DataFrame) -> bytes:
    return roles.select(list(ROLE_TABLE_SCHEMA)).sort('entry_id').write_csv().encode('utf-8')

def role_table_fingerprint(roles: pl.DataFrame) -> str:
    '''
    SHA-256 of the role assignment's canonical CSV (``entry_id``, ``code``, ``role``).

    It equals the committed table's file hash, so a split is identified by its assignment
    whichever file it was read from.
    '''

    return hashlib.sha256(_role_table_csv(roles)).hexdigest()

def write_role_table(roles: pl.DataFrame, path: Path) -> str:
    '''Write the canonical CSV and return its sha256 (``role_table_fingerprint``).'''

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_role_table_csv(roles))
    return role_table_fingerprint(roles)

def read_role_table(path: Path) -> pl.DataFrame:
    '''Read the frozen role table, keeping codes as strings.'''

    return pl.read_csv(Path(path), schema=ROLE_TABLE_SCHEMA)

def attach_role_text(roles: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame:
    '''
    Join the frozen role table to the index entries it was built from.

    Returns:
        ``entry_id``, ``code``, ``text``, ``role`` sorted by ``entry_id``.

    Raises:
        ValueError: If the two list different entries, or give an entry different codes.
    '''

    joined = entries.select('entry_id', 'code', 'text').join(
        roles.select('entry_id',
                     pl.col('code').alias('role_code'), 'role'),
        on='entry_id',
        how='full',
        coalesce=True,
    )
    if joined.filter(pl.col('code').is_null() | pl.col('role').is_null()).height:
        raise ValueError('the role table and the index file list different entries')
    if joined.filter(pl.col('code') != pl.col('role_code')).height:
        raise ValueError('the role table gives an entry a different code than the index file')
    return joined.select(INDEX_ROLE_COLUMNS).sort('entry_id')

# -------------------------------------------------------------------------------------------------
# Consistency with the descriptions built from the roles
# -------------------------------------------------------------------------------------------------

def examples_channel_by_code(role_rows: pl.DataFrame) -> pl.DataFrame:
    '''Each code's examples channel: its examples-role entries in index-file order, joined.'''

    # yapf: disable
    return (
        role_rows
        .filter(pl.col('role') == IndexRole.EXAMPLES.value)
        .sort('entry_id')
        .group_by('code', maintain_order=True)
        .agg(examples=pl.col('text').str.join(EXAMPLES_SEPARATOR))
    )
    # yapf: enable

def verify_examples_channel(descriptions: pl.DataFrame, role_rows: pl.DataFrame) -> None:
    '''
    Require every code with index entries to carry exactly its examples-role entries.

    Raises:
        ValueError: If a code's examples channel differs from its examples-role entries.
    '''

    expected = role_rows.select('code').unique().join(
        examples_channel_by_code(role_rows), on='code', how='left'
    )
    checked = expected.join(
        descriptions.select('code', actual=pl.col('examples')), on='code', how='left'
    )
    mismatched = checked.filter(pl.col('examples').ne_missing(pl.col('actual')))
    if mismatched.height:
        raise ValueError(
            f'{mismatched.height:,} codes have an examples channel other than their '
            f'examples-role entries, e.g. {mismatched.get_column("code").sort().to_list()[:5]}'
        )

def verify_role_leakage(
    descriptions: pl.DataFrame,
    role_rows: pl.DataFrame,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> Dict[str, Dict[str, int]]:
    '''
    Match every validation and test query against all training text; fail on any match.

    Training text is every code's title, description, examples channel and exclusion text in
    ``descriptions``, plus every training-role query.

    Returns:
        Exact and near-duplicate match counts per held-out split (all zero on success).
    '''

    training = role_rows.filter(pl.col('role') == IndexRole.TRAINING.value)
    corpus = training_text_segments(descriptions, extra_texts=training.get_column('text'))
    report: Dict[str, Dict[str, int]] = {}
    for role in (IndexRole.VALIDATION, IndexRole.TEST):
        queries = role_rows.filter(pl.col('role') == role.value).get_column('text').to_list()
        matches = find_leakage(queries, corpus, min_jaccard=min_jaccard)
        report[role.value] = {
            'exact': int(matches.exact.sum()),
            'near_duplicate': int(matches.near_duplicate.sum()),
        }
    leaked = {split: found for split, found in report.items() if any(found.values())}
    if leaked:
        raise ValueError(f'held-out queries match training text: {leaked}')
    return report
