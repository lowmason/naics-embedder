'''
The backbone's trained input window (Req 9, "Input windows"; Verification "Backbone input window").

Inputs fit the window the backbone was trained on, recorded here from the backbone's own
documentation. For sentence-transformers/all-MiniLM-L6-v2, the model card at revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 says that in training "the sequence length was limited
to 128 tokens". Its 256 (``sentence_bert_config.json``'s ``max_seq_length``, the truncation it
applies at inference) and 512 (``config.json``'s ``max_position_embeddings``) are not the trained
window. Every tokenizing path truncates to the window and refuses a longer ``max_length``, and
the supervision bundle records each channel's share of texts beyond it.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

# Tokens per text, [CLS] and [SEP] included
TRAINED_WINDOWS: Dict[str, int] = {'sentence-transformers/all-MiniLM-L6-v2': 128}

# -------------------------------------------------------------------------------------------------
# The window
# -------------------------------------------------------------------------------------------------

def trained_window(backbone: str) -> int:
    '''
    The backbone's trained input window, in tokens.

    Raises:
        ValueError: If no window is recorded for the backbone.
    '''

    if backbone not in TRAINED_WINDOWS:
        raise ValueError(
            f'no trained input window is recorded for {backbone!r}; record it in '
            "utils/input_window.py from the backbone's documentation"
        )
    return TRAINED_WINDOWS[backbone]

def check_window(backbone: str, max_length: Optional[int]) -> int:
    '''
    The length to tokenize at: ``max_length``, or the trained window when it is None.

    Raises:
        ValueError: If ``max_length`` exceeds the backbone's trained window, or none is recorded.
    '''

    window = trained_window(backbone)
    if max_length is None:
        return window
    if max_length > window:
        raise ValueError(
            f'max_length {max_length} exceeds the trained input window of {backbone} '
            f'({window} tokens)'
        )
    return max_length

# -------------------------------------------------------------------------------------------------
# Texts beyond the window
# -------------------------------------------------------------------------------------------------

def token_counter(tokenizer: Any) -> Callable[[List[str]], List[int]]:
    '''Token counts under ``tokenizer``, special tokens included, without truncation.'''

    def count(texts: List[str]) -> List[int]:
        return [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]

    return count

def overflow_shares(
    texts: Mapping[str, Sequence[Optional[str]]],
    count_tokens: Callable[[List[str]], List[int]],
    window: int,
) -> Dict[str, Dict[str, Any]]:
    '''
    Each channel's share of present texts longer than ``window`` tokens.

    Args:
        texts: Each channel's texts; a null or blank text is absent and not counted.
        count_tokens: Token counts of a list of texts, special tokens included.
        window: The trained window.

    Returns:
        Per channel: ``present`` texts, ``over`` (longer than the window) and their ``share``.
    '''

    shares: Dict[str, Dict[str, Any]] = {}
    for channel, values in texts.items():
        present = [text for text in values if text is not None and text.strip()]
        over = sum(count > window for count in count_tokens(present)) if present else 0
        shares[channel] = {
            'present': len(present),
            'over': int(over),
            'share': over / len(present) if present else 0.0,
        }
    return shares
