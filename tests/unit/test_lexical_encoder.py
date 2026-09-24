'''The training-free lexical stub encoder for the outcome panel.'''

import polars as pl
import pytest
import torch

from naics_embedder.panels.lexical_encoder import (
    LexicalTrigramEncoder,
    code_texts_from_descriptions,
)

pytestmark = pytest.mark.unit

def test_code_texts_join_title_description_and_examples_but_not_exclusions():
    descriptions = pl.DataFrame(
        {
            'code': ['111110', '112130'],
            'title': ['Soybean Farming', 'Dual-Purpose Cattle Ranching'],
            'description': ['This industry grows soybeans.', 'This industry raises cattle.'],
            'examples': ['Soybean farming, field', None],
            'excluded': ['Growing corn--are classified elsewhere.', None],
        }
    )

    assert code_texts_from_descriptions(descriptions) == {
        '111110': 'Soybean Farming This industry grows soybeans. Soybean farming, field',
        '112130': 'Dual-Purpose Cattle Ranching This industry raises cattle.',
    }

def test_vectors_are_unit_length_and_ignore_case_and_punctuation():
    encoder = LexicalTrigramEncoder({'111110': 'Soybean farming'}, n_features=256)

    codes = encoder.encode_codes(['111110'])
    queries = encoder.encode_queries(['SOYBEAN -- farming!', 'Tobacco'])

    assert codes.shape == (1, 256)
    assert codes.dtype == torch.float32
    torch.testing.assert_close(queries.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(queries[0], codes[0])

def test_codes_without_text_are_rejected():
    encoder = LexicalTrigramEncoder({'111110': 'Soybean farming'})

    with pytest.raises(ValueError, match='no text'):
        encoder.encode_codes(['111110', '112130'])
