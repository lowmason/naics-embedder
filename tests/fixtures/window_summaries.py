'''
Stubs for the window-fitting summaries tests (roadmap Stage 6b): a tokenizer that counts words,
and summaries rows written as a pinned artifact.
'''

from pathlib import Path
from typing import Any, Dict, List, Sequence

import polars as pl

from naics_embedder.panels.window_summaries import (
    SUMMARIES_SCHEMA,
    SummariesPin,
    write_window_summaries,
)

class WordTokenizer:
    '''A stub tokenizer: one token per whitespace-separated word, plus [CLS] and [SEP].'''

    def __call__(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> Dict[str, List[List[int]]]:
        extra = 2 if add_special_tokens else 0
        return {'input_ids': [[0] * (len(text.split()) + extra) for text in texts]}

def words(texts: Sequence[str]) -> List[int]:
    '''A stub token counter: one token per whitespace-separated word.'''

    return [len(text.split()) for text in texts]

def pin_artifact(path: Path, rows: List[Dict[str, Any]], *, window: int) -> SummariesPin:
    '''Write summaries rows as an artifact at ``path``, and pin it.'''

    sha256 = write_window_summaries(pl.DataFrame(rows, schema=SUMMARIES_SCHEMA), path)
    return SummariesPin(path=str(path), sha256=sha256, window=window)
