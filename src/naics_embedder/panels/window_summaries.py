'''
Window-fitting summaries of over-long channel texts (Req 9, "Input windows"; roadmap Stage 6b).

A channel text whose marked form (``'<field>: <text>'``, special tokens included) is over the
backbone's trained window is read as its summary: whole units of the text, in source order, chosen
once by ``naics-embedder data summaries`` (``data/window_summaries.py``) and committed. Every unit
boundary is a boundary of the leakage segmenter (``panels/leakage.py``), so a summary's segments
are a subset of its text's, and leakage needs no sealed read (spec 4.4).

``WINDOW_SUMMARIES`` pins each backbone's artifact by sha256, and ``summaries_identity`` is the
sha256 every identity site records: the token cache's sidecar, the checkpoint contract, the export
and text-only provenances, and the decision store. ``resolve_channel_texts`` is the only place
summaries enter. The token cache and the text-only builder call it, and it checks the artifact's
invariants against the descriptions on every call that finds an over-window text. The module loads
no model and imports no torch.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import polars as pl

from naics_embedder.panels.leakage import (
    EXAMPLES_SEPARATOR,
    SENTENCE_BREAK,
    normalize_text,
    text_segments,
)
from naics_embedder.text_model.fields import CHANNELS, marked_text, marker

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# The pin
# -------------------------------------------------------------------------------------------------

WINDOW_SUMMARIES_PATH = 'conf/data/window_summaries.csv'

@dataclass(frozen=True)
class SummariesPin:
    '''
    A committed summaries artifact.

    Attributes:
        path: The artifact, relative to the repository root.
        sha256: The artifact's sha256, which every identity site records.
        window: The trained window the summaries fit.
    '''

    path: str
    sha256: str
    window: int

# Keyed by backbone, like TRAINED_WINDOWS (utils/input_window.py). Each entry is a reviewed
# change, committed with the artifact it pins; `naics-embedder data summaries` prints it.
WINDOW_SUMMARIES: Dict[str, SummariesPin] = {
    'sentence-transformers/all-MiniLM-L6-v2': SummariesPin(
        path=WINDOW_SUMMARIES_PATH,
        sha256='dd425eb5ef9a7fa2be1b2e821c02f1b036f74f6ec503ff2fa70256ea7808a9a0',
        window=128,
    ),
}

def summaries_identity(backbone: str) -> Optional[str]:
    '''
    The sha256 of the backbone's pinned summaries, or None when it has no pin.

    Args:
        backbone: A Hugging Face model or tokenizer name.

    Returns:
        The pin's sha256, or None.
    '''

    pin = WINDOW_SUMMARIES.get(backbone)
    return None if pin is None else pin.sha256

# -------------------------------------------------------------------------------------------------
# Token counts
# -------------------------------------------------------------------------------------------------

TokenCounter = Callable[[Sequence[str]], List[int]]

def token_counter(tokenizer: Any, *, special_tokens: bool = True) -> TokenCounter:
    '''
    Token counts under ``tokenizer``, without truncation.

    The torch-free twin of ``utils.input_window.token_counter``, which this module cannot import:
    ``naics_embedder.utils`` imports torch when the package loads.

    Args:
        tokenizer: A Hugging Face tokenizer.
        special_tokens: Count ``[CLS]`` and ``[SEP]``, as the model reads a text.

    Returns:
        A function from texts to their token counts.
    '''

    def count(texts: Sequence[str]) -> List[int]:
        if not texts:
            return []
        encoded = tokenizer(list(texts), add_special_tokens=special_tokens, truncation=False)
        return [len(ids) for ids in encoded['input_ids']]

    return count

# -------------------------------------------------------------------------------------------------
# Units
# -------------------------------------------------------------------------------------------------

SUMMARY_CHANNELS = ('description', 'examples', 'excluded')
UNIT_RULE = 'sentence-clause-piece-v1'
# At levels 1 and 2 a piece that ends in an abbreviation or a list numeral closes no unit (spec 4.2)
NO_BREAK_PATTERN = r'(?:\bU\.S|\bi\.e|\be\.g|\betc|\bNo|\bvs|(?:^|\s)\d{1,2}|\(\d{1,2}\))[.;]$'
_NO_BREAK = re.compile(NO_BREAK_PATTERN)

Span = Tuple[int, int]

def summary_budget(count_marked: TokenCounter, channel: str, window: int) -> int:
    '''
    The tokens a channel's text may take: the window less its marker and special tokens.

    Args:
        count_marked: Token counts, special tokens included.
        channel: A channel name.
        window: The trained window.

    Returns:
        The budget; 124 for every channel under MiniLM at 128.
    '''

    return window - count_marked([marker(channel)])[0]

def _pieces(text: str, start: int, end: int) -> List[Span]:
    '''The segmenter's pieces of ``text[start:end]``, as spans of ``text``; empty ones dropped.'''

    spans: List[Span] = []
    for match in SENTENCE_BREAK.finditer(text, start, end):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, end))
    return [(left, right) for left, right in spans if right > left]

def _closes_sentence(channel: str) -> Callable[[str], bool]:
    # An exclusion text's cross-references end in ';', so each is one unit
    endings = ('.', ';') if channel == 'excluded' else ('.', )
    return lambda piece: piece.endswith(endings)

def _closes_clause(piece: str) -> bool:
    return piece.endswith(('.', ';'))

def _merge(text: str, pieces: List[Span], closes: Callable[[str], bool]) -> List[Span]:
    '''
    Consecutive pieces merged into units: a piece closes its unit when ``closes`` accepts it, the
    unit's parentheses balance and the piece does not end in an abbreviation or list numeral.
    '''

    units: List[Span] = []
    first: Optional[int] = None
    for index, (start, end) in enumerate(pieces):
        first = start if first is None else first
        piece, unit = text[start:end], text[first:end]
        balanced = unit.count('(') == unit.count(')')
        if index == len(pieces) - 1 or (closes(piece) and balanced and not _NO_BREAK.search(piece)):
            units.append((first, end))
            first = None
    return units

def text_units(channel: str, text: str, count: TokenCounter, budget: int) -> List[str]:
    '''
    A channel text's units, in source order, each a verbatim span of ``text`` (spec 4.2).

    Every unit boundary is a boundary of the leakage segmenter. A title is one unit; the examples
    channel's units are its entries. A description or exclusion text's units are its sentences
    (level 1); a sentence over the budget is re-split at its clauses (level 2), and a clause over
    the budget at the segmenter's pieces (level 3).

    Args:
        channel: ``title``, ``description``, ``examples`` or ``excluded``.
        text: The channel text.
        count: Token counts without special tokens.
        budget: The channel's budget (``summary_budget``).

    Returns:
        The units.

    Raises:
        ValueError: If a title, an examples entry or a level-3 piece is over the budget, or the
            channel is none of the four.
    '''

    if channel == 'title':
        if count([text])[0] > budget:
            raise ValueError(f'a title over the window cannot be summarized: {text!r}')
        return [text]
    if channel == 'examples':
        entries = [entry for entry in text.split(EXAMPLES_SEPARATOR) if entry.strip()]
        over = [entry for entry, tokens in zip(entries, count(entries)) if tokens > budget]
        if over:
            raise ValueError(f'an examples entry is over the {budget}-token budget: {over[0]!r}')
        return entries
    if channel not in ('description', 'excluded'):
        raise ValueError(f'no units are defined for channel {channel!r}')

    def fits(span: Span) -> bool:
        return count([text[span[0]:span[1]]])[0] <= budget

    units: List[Span] = []
    for sentence in _merge(text, _pieces(text, 0, len(text)), _closes_sentence(channel)):
        if fits(sentence):
            units.append(sentence)
            continue
        for clause in _merge(text, _pieces(text, *sentence), _closes_clause):
            if fits(clause):
                units.append(clause)
                continue
            # Level 3: the segmenter's pieces, with no balance or no-break guard
            for piece in _pieces(text, *clause):
                if not fits(piece):
                    raise ValueError(
                        f'a {channel} piece is over the {budget}-token budget: '
                        f'{text[piece[0]:piece[1]]!r}'
                    )
                units.append(piece)
    return [text[start:end] for start, end in units]

# -------------------------------------------------------------------------------------------------
# The artifact
# -------------------------------------------------------------------------------------------------

SUMMARIES_SCHEMA = {
    'code': pl.Utf8,
    'channel': pl.Utf8,
    'source_sha256': pl.Utf8,
    'window': pl.Int64,
    'summary': pl.Utf8,
    'source_tokens': pl.Int64,
    'summary_tokens': pl.Int64,
    'units_kept': pl.Int64,
    'units_total': pl.Int64,
}

def text_sha256(text: str) -> str:
    '''The sha256 of a text's UTF-8 bytes.'''

    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def write_window_summaries(rows: pl.DataFrame, path: Path) -> str:
    '''
    Write summaries rows as the artifact: UTF-8 CSV, sorted by channel, then code.

    Args:
        rows: Rows with the columns of ``SUMMARIES_SCHEMA``.
        path: The CSV to write.

    Returns:
        The sha256 of the bytes written.
    '''

    # yapf: disable
    data = (
        rows
        .select([pl.col(name).cast(dtype) for name, dtype in SUMMARIES_SCHEMA.items()])
        .sort('channel', 'code')
        .write_csv()
        .encode('utf-8')
    )
    # yapf: enable
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()

def _parse_window_summaries(data: bytes, path: Path) -> pl.DataFrame:
    rows = pl.read_csv(io.BytesIO(data), schema=SUMMARIES_SCHEMA)
    channels = sorted(set(rows.get_column('channel').to_list()) - set(SUMMARY_CHANNELS))
    if channels:
        raise ValueError(f'{path} names a channel outside {SUMMARY_CHANNELS}: {channels[0]}')
    pairs = Counter(rows.select('code', 'channel').iter_rows())
    repeated = sorted(pair for pair, count in pairs.items() if count > 1)
    if repeated:
        code, channel = repeated[0]
        raise ValueError(f"{path} summarizes code {code}'s {channel} more than once")
    return rows

def read_window_summaries(path: Path) -> pl.DataFrame:
    '''
    Read a summaries artifact with ``SUMMARIES_SCHEMA``, so that ``code`` stays a string.

    Raises:
        ValueError: If a row names a channel other than the three, or a ``(code, channel)``
            repeats.
    '''

    return _parse_window_summaries(Path(path).read_bytes(), Path(path))

# -------------------------------------------------------------------------------------------------
# The resolver
# -------------------------------------------------------------------------------------------------

class _PinDefault(Enum):
    DEFAULT = 'default'

# The resolver's default pin: the backbone's entry in WINDOW_SUMMARIES, looked up at call time
DEFAULT = _PinDefault.DEFAULT

def over_window(
    descriptions: pl.DataFrame,
    count_marked: TokenCounter,
    window: int,
) -> List[Tuple[str, str]]:
    '''
    The present channel texts whose marked form is over the window, sorted by channel, then code.

    Args:
        descriptions: Codes and their four channel texts; a null or blank text is absent.
        count_marked: Token counts, special tokens included.
        window: The window.

    Returns:
        ``(code, channel)`` pairs.
    '''

    codes = descriptions.get_column('code').to_list()
    over: List[Tuple[str, str]] = []
    for channel in CHANNELS:
        texts = descriptions.get_column(channel).to_list()
        present = [row for row, text in enumerate(texts) if text is not None and text.strip()]
        counts = count_marked([marked_text(channel, texts[row]) for row in present])
        over.extend(
            (codes[row], channel) for row, tokens in zip(present, counts) if tokens > window
        )
    return sorted(over, key=lambda pair: (pair[1], pair[0]))

def _raw_pieces(channel: str, text: str) -> List[str]:
    if channel == 'examples':
        return text.split(EXAMPLES_SEPARATOR)
    return SENTENCE_BREAK.split(text)

def _segments(channel: str, text: str) -> Set[str]:
    '''The text's segments, as ``training_text_segments`` cuts them.'''

    if channel == 'examples':
        return {
            segment
            for segment in map(normalize_text, text.split(EXAMPLES_SEPARATOR)) if segment
        }
    return set(text_segments(text))

def _is_extract(channel: str, summary: str, source: str) -> bool:
    '''
    S5: the summary's raw pieces are an in-order subsequence of the source's, each source piece
    used at most once, and its segments are a subset of the source's (spec 4.4).
    '''

    source_pieces = _raw_pieces(channel, source)
    position = 0
    for piece in _raw_pieces(channel, summary):
        while position < len(source_pieces) and source_pieces[position] != piece:
            position += 1
        if position == len(source_pieces):
            return False
        position += 1
    return _segments(channel, summary) <= _segments(channel, source)

def resolve_channel_texts(
    descriptions: pl.DataFrame,
    tokenizer: Any,
    backbone: str,
    max_length: int,
    *,
    pin: Union[SummariesPin, None, _PinDefault] = DEFAULT,
) -> pl.DataFrame:
    '''
    The descriptions, each over-window channel text replaced by its pinned summary (spec 4.7).

    Steps, raising at the first failure: (1) find the texts over the window, and return the
    descriptions unchanged when there are none; (2) require a pin for the window; (3) read the
    artifact and require the pin's sha256; (4) require one row per over-window text and no other;
    (5) require each row's source sha256 and window; (6) require each summary to be an extract of
    its source; (7) require every marked channel text to fit after substitution.

    Args:
        descriptions: Codes and their four channel texts.
        tokenizer: The backbone's tokenizer, which counts tokens as the backbone reads them.
        backbone: The backbone, whose pin is the default.
        max_length: The window texts are tokenized at.
        pin: The summaries to read: ``DEFAULT`` looks up ``WINDOW_SUMMARIES[backbone]`` at call
            time, and None means no pin.

    Returns:
        The descriptions with over-window texts replaced.

    Raises:
        ValueError: At the first step that fails, naming the code and channel where there is one.
    '''

    count_marked = token_counter(tokenizer)
    over = over_window(descriptions, count_marked, max_length)
    if not over:
        return descriptions
    if pin is DEFAULT:
        pin = WINDOW_SUMMARIES.get(backbone)
    if pin is None:
        code, channel = over[0]
        raise ValueError(
            f"code {code}'s {channel} is over the {max_length}-token window, and no window "
            f'summaries are pinned for {backbone} ({len(over)} texts are over)'
        )
    if pin.window != max_length:
        raise ValueError(
            f'the window summaries pinned for {backbone} fit a {pin.window}-token window, but '
            f'texts are tokenized at {max_length}'
        )

    path = Path(pin.path)
    if not path.is_file():
        raise ValueError(
            f'the window summaries pinned for {backbone} are missing: {path.resolve()}'
        )
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != pin.sha256:
        raise ValueError(f'{path} has sha256 {sha256}, but the pin names {pin.sha256}')
    rows = _parse_window_summaries(data, path)

    texts = {
        (code, channel): text
        for channel in CHANNELS
        for code, text in descriptions.select('code', channel).iter_rows()
        if text is not None and text.strip()
    }
    named = set(rows.select('code', 'channel').iter_rows())
    for code, channel in over:
        if (code, channel) not in named:
            raise ValueError(f"code {code}'s {channel} is over the window but has no summary")
    for code, channel in sorted(named - set(over)):
        if (code, channel) not in texts:
            raise ValueError(f"code {code}'s {channel} is not in the descriptions")
        raise ValueError(f"code {code}'s {channel} fits the window and needs no summary")

    summaries = {(row['code'], row['channel']): row for row in rows.iter_rows(named=True)}
    for (code, channel), row in summaries.items():
        if row['source_sha256'] != text_sha256(texts[(code, channel)]):
            raise ValueError(
                f"code {code}'s {channel}: the summary was built from another source text"
            )
        if row['window'] != max_length:
            raise ValueError(
                f"code {code}'s {channel}: the summary fits a {row['window']}-token window, not "
                f'{max_length}'
            )
    for (code, channel), row in summaries.items():
        if not _is_extract(channel, row['summary'], texts[(code, channel)]):
            raise ValueError(f"code {code}'s {channel}: the summary is not an extract of its text")

    resolved = descriptions.with_columns(
        [
            pl.Series(
                channel,
                [
                    summaries[(code, channel)]['summary'] if (code, channel) in summaries else text
                    for code, text in descriptions.select('code', channel).iter_rows()
                ],
                dtype=pl.Utf8,
            ) for channel in SUMMARY_CHANNELS
        ]
    )
    still_over = over_window(resolved, count_marked, max_length)
    if still_over:
        code, channel = still_over[0]
        raise ValueError(
            f"code {code}'s {channel} does not fit the {max_length}-token window after "
            'substitution'
        )
    replaced = {
        channel: sum(pair[1] == channel for pair in summaries)
        for channel in SUMMARY_CHANNELS
    }
    logger.info(f'Window summaries for {backbone} replaced channel texts: {replaced}')
    return resolved
