'''
Req 11's task queries (spec 4.1(i)), built once from a validated supervision bundle.

A query is a distinct (text, level) pair with a target set T of codes at that level and a set N of
forced negatives. Two bundle members give them:

- The role table's training entries: each is a query at its code's level, 6, with T its code and
  N empty.
- The redirection table's activity phrases. A phrase row's destinations are its named codes that
  are not lineal to its referencing code, so a row whose named codes are all lineal stays text
  only (Req 8), and a withheld row carries no phrase. Rows group by (phrase, destination level): T
  is the group's destinations at that level, and N its rows' referencing codes, whatever their
  level, so a cross-reference query always scores its referencing code (Req 8(b)).

An index entry and a phrase with the same exact text at level 6 are one query, whose T and N are
the unions; at another level the two stay apart. A query whose T and N overlap is refused: no code
can be both where an activity is classified and the code that sends it elsewhere.

Only training text is read: the role table's training rows, and the phrases of the redirection
rows that are not withheld, without the rows' texts.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import polars as pl

from naics_embedder.supervision.artifacts import (
    INDEX_ROLES_ARTIFACT,
    REDIRECTIONS_ARTIFACT,
    ValidatedSupervisionBundle,
)
from naics_embedder.supervision.schema import IndexRole

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# The query
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskQuery:
    '''
    One task query (spec 4.1(i)).

    Attributes:
        text: The query's text: an index entry's or an activity phrase, exactly as stored.
        level: The level of every target, 2-6.
        targets: T, the codes the text is classified in, sorted.
        negatives: N, the forced negatives: the referencing codes of the phrase rows that send the
            text away, sorted.
    '''

    text: str
    level: int
    targets: Tuple[str, ...]
    negatives: Tuple[str, ...]

# -------------------------------------------------------------------------------------------------
# The two sources
# -------------------------------------------------------------------------------------------------

def _training_entries(bundle: ValidatedSupervisionBundle) -> pl.DataFrame:
    '''The role table's training entries (``code``, ``text``); no held-out row is collected.'''

    # yapf: disable
    return (
        pl.scan_parquet(bundle.artifact_path(INDEX_ROLES_ARTIFACT))
        .filter(pl.col('role') == IndexRole.TRAINING.value)
        .select('code', 'text')
        .collect()
    )
    # yapf: enable

def _phrase_rows(bundle: ValidatedSupervisionBundle) -> pl.DataFrame:
    '''The redirection rows that carry a phrase and are not withheld, without their texts.'''

    # yapf: disable
    return (
        pl.scan_parquet(bundle.artifact_path(REDIRECTIONS_ARTIFACT))
        .filter(~pl.col('withheld') & pl.col('activity').is_not_null())
        .select('code', 'activity', 'named_codes', 'lineal_codes')
        .collect()
    )
    # yapf: enable

# -------------------------------------------------------------------------------------------------
# The queries
# -------------------------------------------------------------------------------------------------

QueryKey = Tuple[str, int]

def build_task_queries(bundle: ValidatedSupervisionBundle) -> List[TaskQuery]:
    '''
    Every task query of the bundle, sorted by level, then text (spec 4.1(i)).

    Args:
        bundle: A validated supervision bundle, whose ``index_roles`` and ``redirections``
            members give the queries.

    Returns:
        One query per distinct (text, level) pair, with its targets and forced negatives sorted.

    Raises:
        ValueError: If a query's targets and forced negatives overlap.
    '''

    targets: Dict[QueryKey, Set[str]] = defaultdict(set)
    negatives: Dict[QueryKey, Set[str]] = defaultdict(set)
    from_entries: Set[QueryKey] = set()
    from_phrases: Set[QueryKey] = set()
    for code, text in _training_entries(bundle).iter_rows():
        key = (text, len(code))
        targets[key].add(code)
        from_entries.add(key)
    for code, phrase, named, lineal in _phrase_rows(bundle).iter_rows():
        for destination in named:
            if destination in lineal:
                continue
            key = (phrase, len(destination))
            targets[key].add(destination)
            negatives[key].add(code)
            from_phrases.add(key)

    queries = [
        TaskQuery(
            text=text,
            level=level,
            targets=tuple(sorted(targets[text, level])),
            negatives=tuple(sorted(negatives[text, level])),
        ) for text, level in sorted(targets, key=lambda key: (key[1], key[0]))
    ]
    for query in queries:
        overlap = sorted(set(query.targets) & set(query.negatives))
        if overlap:
            raise ValueError(
                f'task query {query.text!r} at level {query.level} names {overlap} as a target '
                'and a forced negative'
            )

    by_level = Counter(query.level for query in queries)
    levels = ', '.join(f'{level}: {count:,}' for level, count in sorted(by_level.items()))
    logger.info(
        f'Task queries: {len(queries):,} ({len(from_entries):,} from index entries, '
        f'{len(from_entries & from_phrases):,} merged with a phrase, '
        f'{len(from_phrases - from_entries):,} phrase-only); by level: {levels}'
    )
    return queries
