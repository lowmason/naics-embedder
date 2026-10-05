'''
The redirection table (Req 8): every Census cross-reference once, with where it sends an activity.

A cross-reference reroutes an activity rather than asserting that two codes are unrelated:
"Growing soybeans--are classified in Industry 111110" sends soybean growing to 111110. The table
holds one row per cross-reference row, in file order, then one per "Excluded" paragraph harvested
from a description, in code order. Its columns:

- ``reference_id``: the row's position in the table, so cross-reference rows keep their file
  positions.
- ``source``: ``cross_reference`` or ``description``.
- ``code`` and ``text``: the referencing code and the row's text.
- ``activity``: the text before ``--`` or before " are/is classified" (or "included"), on a
  cross-reference row that names a code (``supervision.activity.activity_phrase``, which this
  module re-exports). Stage 7 trains on these phrases as queries, and the bundle loader
  recomputes each one.
- ``named_codes``: the codebook codes the text names other than its own code, in order of first
  appearance. On a cross-reference row these are its destinations.
- ``lineal_codes``: the named codes that are the row's code's ancestors or descendants. Lineal
  references stay text only and never act as negatives.
- ``withheld``: a held-out query leaks into one of the text's segments or into its activity
  phrase (Req 3), so the text leaves the exclusion channel and the activity phrase is dropped.
  The row stays in the table, and its named codes still count as exclusions.

The exclusion channel is built from the table: each code's texts that are not withheld, once
each, joined in table order.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import re
from fractions import Fraction
from typing import List, Optional, Sequence, Set

import polars as pl

from naics_embedder.panels.leakage import (
    NEAR_DUPLICATE_MIN_JACCARD,
    leaking_texts,
    normalize_text,
    text_segments,
)
from naics_embedder.supervision.activity import activity_phrase
from naics_embedder.supervision.artifacts import (
    CROSS_REFERENCE_SOURCE,
    DESCRIPTION_SOURCE,
    REDIRECTIONS_SCHEMA,
)
from naics_embedder.utils.naics_hierarchy import code_lineage

logger = logging.getLogger(__name__)

_CODE_REFERENCE = re.compile(r' (\d{2,6})')

# -------------------------------------------------------------------------------------------------
# One row's parts
# -------------------------------------------------------------------------------------------------

def named_codes(code: str, text: str, codes: Set[str]) -> List[str]:
    '''The codebook codes ``text`` names other than ``code``, in order of first appearance.'''

    named: List[str] = []
    for number in _CODE_REFERENCE.findall(text):
        if number in codes and number != code and number not in named:
            named.append(number)
    return named

def lineal_codes(code: str, named: Sequence[str]) -> List[str]:
    '''The named codes that are ``code``'s ancestors or descendants.'''

    return [other for other in named if other in code_lineage(code) or code in code_lineage(other)]

def _activity_segments(activity: Optional[str]) -> List[str]:
    normalized = normalize_text(activity or '')
    return [normalized] if normalized else []

# -------------------------------------------------------------------------------------------------
# The table and the channel
# -------------------------------------------------------------------------------------------------

def build_redirections(
    references: pl.DataFrame,
    paragraphs: pl.DataFrame,
    codes: Set[str],
    held_out_queries: Sequence[str] = (),
    *,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> pl.DataFrame:
    '''
    The redirection table: ``references`` rows in order, then ``paragraphs`` rows in order.

    Args:
        references: The cross-reference file's rows (``code``, ``text``), in file order.
        paragraphs: The "Excluded" paragraphs harvested from descriptions (``code``, ``text``),
            in code order.
        codes: The codebook's codes.
        held_out_queries: Validation and test queries; a row any of them leaks into is withheld.
        min_jaccard: The near-duplicate threshold of the leakage check.

    Returns:
        One row per input row, with the columns of ``REDIRECTIONS_SCHEMA``.
    '''

    rows = []
    for source, frame in ((CROSS_REFERENCE_SOURCE, references), (DESCRIPTION_SOURCE, paragraphs)):
        for code, text in frame.select('code', 'text').iter_rows():
            named = named_codes(code, text, codes)
            activity = activity_phrase(text) if source == CROSS_REFERENCE_SOURCE and named else None
            rows.append((len(rows), source, code, text, activity, named, lineal_codes(code, named)))

    in_text = leaking_texts(
        [text_segments(row[3]) for row in rows], held_out_queries, min_jaccard=min_jaccard
    )
    in_activity = leaking_texts(
        [_activity_segments(row[4]) for row in rows], held_out_queries, min_jaccard=min_jaccard
    )
    withheld = in_text | in_activity
    table = pl.DataFrame(
        [
            (*row[:4], None if flag else row[4], *row[5:], bool(flag))
            for row, flag in zip(rows, withheld)
        ],
        schema=REDIRECTIONS_SCHEMA,
        orient='row',
    )

    naming = table.filter(pl.col('named_codes').list.len() > 0).height
    activities = table.get_column('activity').drop_nulls().len()
    lineal = int(table.get_column('lineal_codes').list.len().sum())
    withheld_ids = table.filter('withheld').get_column('reference_id').to_list()
    logger.info('Redirection table:')
    logger.info(f'  Cross-reference rows: {references.height: ,}')
    logger.info(f'  Excluded paragraphs from descriptions: {paragraphs.height: ,}')
    logger.info(f'  Rows naming a code: {naming: ,}')
    logger.info(f'  Activity phrases: {activities: ,}')
    logger.info(f'  Lineal references: {lineal: ,}')
    logger.info(f'  Withheld rows: {withheld_ids}\n')
    return table

def exclusion_channel(redirections: pl.DataFrame) -> pl.DataFrame:
    '''
    Each code's exclusion channel from the redirection table (Req 8(a)).

    Returns:
        One row per code with a redirection row, sorted by code: ``excluded``, the texts of its
        rows that are not withheld, each once, joined by one space in table order (null when all
        are withheld), and ``excluded_codes``, the codes its rows name, withheld rows included,
        each once in order of first appearance (null when none).
    '''

    # yapf: disable
    return (
        redirections
        .sort('reference_id')
        .group_by('code', maintain_order=True)
        .agg(
            excluded=pl.col('text').filter(~pl.col('withheld')),
            excluded_codes=pl.col('named_codes').flatten().drop_nulls().unique(maintain_order=True),
        )
        .select(
            code=pl.col('code'),
            excluded=pl.when(pl.col('excluded').list.len() > 0).then(
                pl.col('excluded').list.join(' ')
            ),
            excluded_codes=pl.when(pl.col('excluded_codes').list.len() > 0).then(
                pl.col('excluded_codes')
            ),
        )
        .sort('code')
    )
    # yapf: enable
