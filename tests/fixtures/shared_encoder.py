'''
A tiny backbone for the shared encoder's tests, so they download nothing (spec §6), and a
five-code arm built on it.

``tiny_bert`` builds a one-layer BERT whose vocabulary is MiniLM's, so token rows from the real
tokenizer fit it. ``tiny_backbone`` makes every ``SharedEncoder`` a test builds load it in place of
MiniLM.

The arm fixtures train nothing. ``shared_model`` is a d = 16 model of the five-code supervision
bundle (``tests/fixtures/supervision.py``) on the tiny backbone, and ``shared_checkpoint`` saves it
as Lightning would. ``text_only_comparator_table`` is a table a read can be pointed at by mistake:
the text-only comparator's, written by its own builder.
'''

from pathlib import Path
from typing import Any, Dict, List

import polars as pl
import pytest
import pytorch_lightning as pyl
import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from naics_embedder.panels.text_only import build_text_only_table
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import export_code_table
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import TokenizationConfig

TINY_HIDDEN = 8

def tiny_bert(name: str = 'tiny-bert') -> BertModel:
    '''A seeded one-layer BERT of width 8 over MiniLM's 30,522-token vocabulary, in eval mode.'''

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=30522,
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=512,
    )
    # from_pretrained returns the real backbone in eval mode; a train-mode stand-in would hide it
    return BertModel(config).eval()

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
    '''
    A d = 16 masked-mean model of the five-code bundle on the tiny backbone, in eval mode.

    It records MiniLM's summaries, as training does: under the test seam, the dummy pin's sha256.
    '''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        fusion='masked_mean',
        dimension=ARM_DIMENSION,
        curvature=1.0,
        supervision_manifest_path=str(generated_bundle),
        summaries=summaries_identity(MINILM),
    )
    return model.eval()

@pytest.fixture
def shared_checkpoint(tmp_path, shared_model) -> Path:
    '''``shared_model`` saved as a Lightning checkpoint.'''

    path = tmp_path / 'arm.ckpt'
    torch.save(lightning_checkpoint(shared_model), path)
    return path

@pytest.fixture
def truncated_checkpoint(tmp_path, shared_model) -> Path:
    '''``shared_model`` saved as a checkpoint trained before Stage 6b: it records no summaries.'''

    checkpoint = lightning_checkpoint(shared_model)
    del checkpoint['stage3_supervision']['summaries']
    del checkpoint['hyper_parameters']['summaries']
    path = tmp_path / 'truncated.ckpt'
    torch.save(checkpoint, path)
    return path

@pytest.fixture
def exported_table(tmp_path, shared_checkpoint, validated_bundle, five_code_token_config) -> Path:
    '''``shared_checkpoint``'s code table, exported on the CPU, with its provenance beside it.'''

    return export_code_table(
        shared_checkpoint, validated_bundle, five_code_token_config, tmp_path / 'arm_table.parquet'
    )

@pytest.fixture
def text_only_comparator_table(tmp_path, five_code_descriptions_parquet) -> Path:
    '''
    The five codes' text-only comparator table, with its provenance beside it.

    ``build_text_only_table`` writes both, on the tiny backbone and MiniLM's tokenizer. The
    provenance has the table's hash and window, as an export's does, but names no checkpoint.
    '''

    return build_text_only_table(
        five_code_descriptions_parquet,
        tmp_path / 'text_only.parquet',
        backbone=MINILM,
        max_length=TOKEN_WINDOW,
        model=tiny_bert(),
        tokenizer=AutoTokenizer.from_pretrained(MINILM),
    )
