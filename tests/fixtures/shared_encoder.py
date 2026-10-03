'''
A tiny backbone for the shared encoder's tests, so they download nothing (spec §6), and a
five-code arm built on it.

``tiny_bert`` builds a one-layer BERT whose vocabulary is MiniLM's, so token rows from the real
tokenizer fit it. ``tiny_backbone`` makes every ``SharedEncoder`` a test builds load it in place of
MiniLM.

The arm fixtures train nothing. ``shared_model`` is a d = 16 model of the five-code supervision
bundle (``tests/fixtures/supervision.py``) on the tiny backbone, and ``shared_checkpoint`` saves it
as Lightning would.
'''

from pathlib import Path
from typing import Any, Dict, List

import polars as pl
import pytest
import pytorch_lightning as pyl
import torch
from transformers import BertConfig, BertModel

from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import TokenizationConfig

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

# -------------------------------------------------------------------------------------------------
# A shared-encoder arm of the five-code bundle
# -------------------------------------------------------------------------------------------------

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
# The five-code bundle's codebook, in code_id order
FIVE_CODES = ('111111', '111112', '111113', '222222', '333333')
ARM_DIMENSION = 16
TOKEN_WINDOW = 32

def lightning_checkpoint(model: NAICSContrastiveModel) -> Dict[str, Any]:
    '''The dict Lightning saves for ``model``, its checkpoint contract included.'''

    checkpoint = {
        'state_dict': model.state_dict(),
        'hyper_parameters': dict(model.hparams),
        'pytorch-lightning_version': pyl.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    return checkpoint

def five_code_token_rows(token_config: TokenizationConfig,
                         bundle: ValidatedSupervisionBundle) -> List[Dict[str, Any]]:
    '''The five codes' cached token rows, in codebook order (the descriptions' ``index``).'''

    cache = tokenization_cache(
        token_config,
        description_fingerprint=bundle.manifest.description_fingerprint,
        codebook_fingerprint=bundle.manifest.codebook_fingerprint,
    )
    return [cache[index] for index in range(len(FIVE_CODES))]

@pytest.fixture
def five_code_descriptions_parquet(tmp_path, text_descriptions_fixture) -> Path:
    '''The five-code descriptions with their text channels and a ``level``, under ``tmp_path``.'''

    path = tmp_path / 'naics_descriptions.parquet'
    text_descriptions_fixture.with_columns(level=pl.lit(6)).write_parquet(path)
    return path

@pytest.fixture
def five_code_token_config(tmp_path, five_code_descriptions_parquet) -> TokenizationConfig:
    '''The five codes' token cache: MiniLM's tokenizer, a 32-token window, under ``tmp_path``.'''

    return TokenizationConfig(
        descriptions_parquet=str(five_code_descriptions_parquet),
        tokenizer_name=MINILM,
        max_length=TOKEN_WINDOW,
        output_path=str(tmp_path / 'token_cache' / 'token_cache.pt'),
    )

@pytest.fixture
def shared_model(tiny_backbone, generated_bundle) -> NAICSContrastiveModel:
    '''A d = 16 masked-mean model of the five-code bundle on the tiny backbone, in eval mode.'''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        fusion='masked_mean',
        dimension=ARM_DIMENSION,
        curvature=1.0,
        supervision_manifest_path=str(generated_bundle),
    )
    return model.eval()

@pytest.fixture
def shared_checkpoint(tmp_path, shared_model) -> Path:
    '''``shared_model`` saved as a Lightning checkpoint.'''

    path = tmp_path / 'arm.ckpt'
    torch.save(lightning_checkpoint(shared_model), path)
    return path
