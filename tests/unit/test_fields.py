'''
Field markers (spec R13) and tokenizing one field text as the cache stores it.
'''

import pytest
import torch
from transformers import AutoTokenizer

from naics_embedder.text_model.fields import (
    CHANNELS,
    FIELDS,
    QUERY,
    marked_text,
    marker,
    tokenize_field,
)

pytestmark = pytest.mark.unit

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'

@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(MINILM)

def test_the_five_fields_and_their_markers():
    assert FIELDS == ('title', 'description', 'excluded', 'examples', 'query')
    assert CHANNELS == FIELDS[:4]
    assert QUERY == 'query'
    assert marker('excluded') == 'excluded: '
    assert marked_text('title', 'Soybean Farming') == 'title: Soybean Farming'

def test_a_field_outside_the_marker_set_is_refused(tokenizer):
    with pytest.raises(ValueError, match='unknown field'):
        marked_text('summary', 'text')
    with pytest.raises(ValueError, match='unknown field'):
        tokenize_field(tokenizer, 'summary', None, 16)

def test_a_present_text_is_tokenized_with_its_marker(tokenizer):
    encoded = tokenize_field(tokenizer, 'title', 'Soybean Farming', 16)

    expected = tokenizer(
        'title: Soybean Farming',
        padding='max_length',
        truncation=True,
        max_length=16,
        return_tensors='pt',
    )
    assert encoded['present'] is True
    assert torch.equal(encoded['input_ids'], expected['input_ids'][0])
    assert torch.equal(encoded['attention_mask'], expected['attention_mask'][0])

@pytest.mark.parametrize('text', [None, '', '   '])
def test_an_absent_text_is_the_unmarked_empty_string(tokenizer, text):
    encoded = tokenize_field(tokenizer, 'examples', text, 16)

    assert encoded['present'] is False
    assert encoded['input_ids'].shape == (16, )
    assert int(encoded['attention_mask'].sum()) == 2  # [CLS] [SEP]
