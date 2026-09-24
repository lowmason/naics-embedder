'''
A training-free lexical encoder for the outcome panel.

Texts are hashed character trigrams of their normalized form (``leakage.normalize_text``),
L2-normalized, so cosine distance decodes a query to the code whose text shares most of its
trigrams. It is the stub arm that exercises the panel end to end on the real validation split
until a trained encoder can embed a query (roadmap Stage 6), and a floor for trained arms. It is
never a candidate configuration.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Dict, Iterable, Mapping, Sequence

import polars as pl
import torch
from sklearn.feature_extraction.text import HashingVectorizer

from naics_embedder.panels.leakage import normalize_text

CODE_TEXT_COLUMNS = ('title', 'description', 'examples')

# -------------------------------------------------------------------------------------------------
# Encoder
# -------------------------------------------------------------------------------------------------

def code_texts_from_descriptions(descriptions: pl.DataFrame) -> Dict[str, str]:
    '''
    Each code's title, description and examples channel, joined by spaces.

    Exclusion text is left out: it names activities the code does not cover.
    '''

    texts = {}
    for row in descriptions.select('code', *CODE_TEXT_COLUMNS).iter_rows(named=True):
        texts[row['code']] = ' '.join(row[name] for name in CODE_TEXT_COLUMNS if row[name])
    return texts

class LexicalTrigramEncoder:
    '''
    Hashed character-trigram vectors for codes (from their texts) and queries.

    Args:
        code_texts: The text each code is embedded from.
        n_features: Hash buckets per vector.
    '''

    def __init__(self, code_texts: Mapping[str, str], n_features: int = 4096):
        self.code_texts = dict(code_texts)
        self.vectorizer = HashingVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 3),
            n_features=n_features,
            alternate_sign=False,
            norm='l2',
            lowercase=False,
            preprocessor=normalize_text,
        )

    def _embed(self, texts: Iterable[str]) -> torch.Tensor:
        return torch.from_numpy(self.vectorizer.transform(list(texts)).toarray()).to(torch.float32)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        missing = [code for code in codes if code not in self.code_texts]
        if missing:
            raise ValueError(f'{len(missing):,} codes have no text, e.g. {missing[:5]}')
        return self._embed(self.code_texts[code] for code in codes)

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self._embed(texts)
