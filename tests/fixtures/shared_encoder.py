'''
A tiny backbone for the shared encoder's tests, so they download nothing (spec §6).

``tiny_bert`` builds a one-layer BERT whose vocabulary is MiniLM's, so token rows from the real
tokenizer fit it. ``tiny_backbone`` makes every ``SharedEncoder`` a test builds load it in place of
MiniLM.
'''

import pytest
import torch
from transformers import BertConfig, BertModel

TINY_HIDDEN = 8

def tiny_bert(name: str = 'tiny-bert') -> BertModel:
    '''A seeded one-layer BERT of width 8 over MiniLM's 30,522-token vocabulary.'''

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=30522,
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=512,
    )
    return BertModel(config)

@pytest.fixture
def tiny_backbone(monkeypatch):
    '''Every ``SharedEncoder`` built in the test loads ``tiny_bert`` instead of MiniLM.'''

    monkeypatch.setattr('naics_embedder.text_model.shared_encoder.load_base_model', tiny_bert)
    return tiny_bert
