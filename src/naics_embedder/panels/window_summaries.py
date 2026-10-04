'''
Window-fitting summaries of over-long channel texts (Req 9, "Input windows"; roadmap Stage 6b).

``WINDOW_SUMMARIES`` pins, per backbone, the committed artifact of extractive summaries, and
``summaries_identity`` is the sha256 every identity site records: the token cache's sidecar, the
checkpoint contract, the export and text-only provenances, and the decision store.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from naics_embedder.panels.leakage import EXAMPLES_SEPARATOR, SENTENCE_BREAK
from naics_embedder.text_model.fields import marker

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

# Keyed by backbone. Plan 9's Exit adds MiniLM's entry together with the artifact it pins.
WINDOW_SUMMARIES: Dict[str, SummariesPin] = {}

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
        ValueError: If a title, an examples entry or a level-3 piece is over the budget.
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
