'''
The text fields the shared encoder reads, and their markers (Req 14; spec R13).

A code has four channels: title, description, excluded and examples. A query is a fifth field. A
present text is marked with its field's name, ``'<field>: <text>'``, so one backbone can tell the
fields apart. An absent text (null or blank) is the empty string with ``present`` False and no
marker, so fusion can mask it (Req 9).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, Dict, Optional

FIELDS = ('title', 'description', 'excluded', 'examples', 'query')
CHANNELS = FIELDS[:4]
QUERY = 'query'

# -------------------------------------------------------------------------------------------------
# Markers
# -------------------------------------------------------------------------------------------------

def marker(field: str) -> str:
    '''
    The prefix that marks a field's text.

    Raises:
        ValueError: If the field is outside the marker set.
    '''

    if field not in FIELDS:
        raise ValueError(f'unknown field {field!r}; the marker set is {list(FIELDS)}')
    return f'{field}: '

def marked_text(field: str, text: str) -> str:
    '''A present text with its field's marker, ``'<field>: <text>'``.'''

    return marker(field) + text

# -------------------------------------------------------------------------------------------------
# Tokenization
# -------------------------------------------------------------------------------------------------

def tokenize_field(
    tokenizer: Any,
    field: str,
    text: Optional[str],
    max_length: int,
) -> Dict[str, Any]:
    '''
    Tokenize one field text as the tokenization cache stores it.

    A present text is tokenized with its marker. An absent one (null or blank) is the empty
    string, ``[CLS] [SEP]``, with ``present`` False and no marker. Either is truncated and padded
    to ``max_length``.

    Args:
        tokenizer: The backbone's tokenizer.
        field: One of ``FIELDS``.
        text: The field's text; None or blank is absent.
        max_length: Tokens kept, at most the backbone's trained window.

    Returns:
        ``input_ids`` and ``attention_mask`` of shape ``(max_length,)``, and ``present``.

    Raises:
        ValueError: If the field is outside the marker set.
    '''

    prefix = marker(field)
    present = bool((text or '').strip())
    encoded = tokenizer(
        prefix + text if present else '',
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return {
        'input_ids': encoded['input_ids'].squeeze(0),
        'attention_mask': encoded['attention_mask'].squeeze(0),
        'present': present,
    }
