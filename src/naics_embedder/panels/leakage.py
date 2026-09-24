'''
Leakage between held-out index-entry queries and training text (Req 3, "Leakage").

Texts are compared after ``normalize_text``: lowercase ASCII letters and digits, with every other
run of characters collapsed to one space. A query leaks into a training segment in two ways:

- exact: the query occurs in the segment as whole words (equality included);
- near-duplicate: the character-trigram Jaccard similarity of query and segment reaches the
  threshold, 9/10 by default. Trigrams are taken inside word boundaries, as scikit-learn's
  ``char_wb`` analyzer builds them, so reordered words score as the same text.

Jaccard is compared in integers (``denominator * shared >= numerator * union``), so no pair sits
on a floating-point boundary. The check compares whole queries with whole segments, so a query
whose words appear reordered inside a longer segment is not caught.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

NEAR_DUPLICATE_MIN_JACCARD = Fraction(9, 10)
EXAMPLES_SEPARATOR = '; '
ACTIVITY_SEPARATOR = '--'
TEXT_COLUMNS = ('title', 'description', 'examples', 'excluded')

_NON_ALNUM = re.compile(r'[^a-z0-9]+')
_SENTENCE_BREAK = re.compile(r'(?<=[.;])\s+')
_CHUNK_ROWS = 256

# -------------------------------------------------------------------------------------------------
# Normalization and segmentation
# -------------------------------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    '''Lowercase ASCII letters and digits, every other run of characters one space.'''

    return _NON_ALNUM.sub(' ', text.lower()).strip()

def text_segments(text: Optional[str]) -> List[str]:
    '''
    Normalized sentences of a description or exclusion text.

    A cross-reference sentence ("Growing soybeans--are classified in Industry 111110") also
    yields its activity phrase, the part before ``--``, which later stages train on as a query.
    '''

    if not text:
        return []
    pieces: List[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        pieces.append(sentence)
        if ACTIVITY_SEPARATOR in sentence:
            pieces.append(sentence.split(ACTIVITY_SEPARATOR, 1)[0])
    return [segment for segment in map(normalize_text, pieces) if segment]

def training_text_segments(
    descriptions: pl.DataFrame,
    extra_texts: Iterable[str] = (),
) -> List[str]:
    '''
    Sorted, de-duplicated normalized segments of every code's training text, plus extra texts.

    Titles are one segment each, descriptions and exclusion texts are split into sentences, and
    the examples channel is split into its ``'; '``-joined entries.
    '''

    missing = [name for name in TEXT_COLUMNS if name not in descriptions.columns]
    if missing:
        raise ValueError(f'descriptions lack text columns: {missing}')
    segments = set()
    for title, description, examples, excluded in descriptions.select(TEXT_COLUMNS).iter_rows():
        title_text = normalize_text(title or '')
        if title_text:
            segments.add(title_text)
        segments.update(text_segments(description))
        segments.update(text_segments(excluded))
        segments.update(
            text for text in map(normalize_text, (examples or '').split(EXAMPLES_SEPARATOR)) if text
        )
    segments.update(text for text in map(normalize_text, extra_texts) if text)
    return sorted(segments)

# -------------------------------------------------------------------------------------------------
# Matching
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LeakageMatches:
    '''Per-query leakage flags, one boolean per query in input order.'''

    exact: np.ndarray
    near_duplicate: np.ndarray

    @property
    def leaked(self) -> np.ndarray:
        return self.exact | self.near_duplicate

def _check_threshold(min_jaccard: Fraction) -> None:
    if not 0 < min_jaccard <= 1:
        raise ValueError(f'min_jaccard must lie in (0, 1], got {min_jaccard}')

def _normalized_queries(queries: Sequence[str]) -> List[str]:
    normalized = [normalize_text(query) for query in queries]
    empty = [query for query, text in zip(queries, normalized) if not text]
    if empty:
        raise ValueError(f'{len(empty):,} queries have no letters or digits, e.g. {empty[0]!r}')
    return normalized

def _containing_row_counts(patterns: Sequence[str], texts: Sequence[str]) -> Dict[str, int]:
    '''For each pattern found, the number of texts that contain it as whole words.'''

    if not patterns or not texts:
        return {}
    padded = [f' {pattern} ' for pattern in patterns]
    # yapf: disable
    found = (
        pl.DataFrame({'text': [f' {text} ' for text in texts]})
        .select(match=pl.col('text').str.extract_many(padded, overlapping=True).list.unique())
        .explode('match')
        .drop_nulls()
        .group_by('match')
        .len()
    )
    # yapf: enable
    return {match[1:-1]: count for match, count in found.iter_rows()}

def _near_duplicate_flags(
    queries: Sequence[str],
    corpus: Sequence[str],
    min_jaccard: Fraction,
    *,
    same_list: bool,
) -> np.ndarray:
    flags = np.zeros(len(queries), dtype=bool)
    if not queries or not corpus:
        return flags
    vectorizer = CountVectorizer(
        analyzer='char_wb', ngram_range=(3, 3), binary=True, lowercase=False, dtype=np.int32
    ).fit(list(queries) + list(corpus))
    query_grams = vectorizer.transform(queries)
    corpus_grams = vectorizer.transform(corpus)
    query_sizes = np.asarray(query_grams.sum(axis=1)).ravel()
    corpus_sizes = np.asarray(corpus_grams.sum(axis=1)).ravel()
    numerator, denominator = min_jaccard.numerator, min_jaccard.denominator
    for start in range(0, len(queries), _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, len(queries))
        shared = (query_grams[start:stop] @ corpus_grams.T).toarray()
        union = query_sizes[start:stop, None] + corpus_sizes[None, :] - shared
        similar = denominator * shared >= numerator * union
        if same_list:
            rows = np.arange(stop - start)
            similar[rows, rows + start] = False
        flags[start:stop] = similar.any(axis=1)
    return flags

def find_leakage(
    queries: Sequence[str],
    corpus: Sequence[str],
    *,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> LeakageMatches:
    '''
    Flag each query that occurs in, or near-duplicates, any corpus text.

    Raises:
        ValueError: If a query has no letters or digits, or the threshold is outside (0, 1].
    '''

    _check_threshold(min_jaccard)
    normalized = _normalized_queries(queries)
    targets = [text for text in map(normalize_text, corpus) if text]
    counts = _containing_row_counts(normalized, targets)
    exact = np.array([counts.get(text, 0) > 0 for text in normalized], dtype=bool)
    near = _near_duplicate_flags(normalized, targets, min_jaccard, same_list=False)
    return LeakageMatches(exact=exact, near_duplicate=near)

def find_leakage_within(
    texts: Sequence[str],
    *,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> LeakageMatches:
    '''
    Flag each text that occurs in, or near-duplicates, another text of the same list.

    A text is never matched against itself, but two identical texts flag each other.
    '''

    _check_threshold(min_jaccard)
    normalized = _normalized_queries(texts)
    counts = _containing_row_counts(normalized, normalized)
    exact = np.array([counts.get(text, 0) > 1 for text in normalized], dtype=bool)
    near = _near_duplicate_flags(normalized, normalized, min_jaccard, same_list=True)
    return LeakageMatches(exact=exact, near_duplicate=near)
