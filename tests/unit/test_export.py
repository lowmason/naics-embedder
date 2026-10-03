'''
Encoding token rows through an arm's model, the HGCN feeder built on it, and the code-table
export (spec 4.3).
'''

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.cli.commands import training as training_cli
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.export import code_token_config, encode_token_rows
from naics_embedder.utils.config import Config
from tests.fixtures.shared_encoder import (
    ARM_DIMENSION,
    FIVE_CODES,
    MINILM,
    TOKEN_WINDOW,
    five_code_token_rows,
)

pytestmark = pytest.mark.unit

# -------------------------------------------------------------------------------------------------
# Encoding token rows
# -------------------------------------------------------------------------------------------------

def test_code_token_config_is_the_cache_training_reads(tmp_path, monkeypatch):
    # The default cache path is relative to the working directory: keep it under tmp_path
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = '/data/descriptions.parquet'
    cfg.data_loader.streaming.max_length = 64
    # The window the feeder used to read, which training never did
    cfg.data_loader.tokenization.max_length = 32

    token_config = code_token_config(cfg)

    assert token_config.descriptions_parquet == '/data/descriptions.parquet'
    assert token_config.tokenizer_name == MINILM
    assert token_config.max_length == 64
    assert token_config.output_path == './data/token_cache/token_cache.pt'

def test_rows_encode_to_float64_cpu_tensors_in_row_order(
    shared_model, five_code_token_config, validated_bundle
):
    rows = five_code_token_rows(five_code_token_config, validated_bundle)

    encoded = encode_token_rows(shared_model, rows, batch_size=2)

    assert encoded['tangent'].shape == (5, ARM_DIMENSION)
    assert encoded['embedding'].shape == (5, ARM_DIMENSION + 1)
    for tensor in encoded.values():
        assert tensor.dtype == torch.float64
        assert tensor.device.type == 'cpu'
    # Three batches give what one forward pass over all five rows gives
    with torch.no_grad():
        whole = shared_model(stack_text_inputs(rows))
    assert torch.allclose(encoded['tangent'], whole['tangent'].to(torch.float64), atol=1e-6)
    assert torch.allclose(encoded['embedding'], whole['embedding'].to(torch.float64), atol=1e-6)

def test_rows_encode_in_eval_mode(shared_model, five_code_token_config, validated_bundle):
    rows = five_code_token_rows(five_code_token_config, validated_bundle)
    # BERT's dropout would make two training-mode passes differ
    shared_model.train()

    first = encode_token_rows(shared_model, rows)['tangent']

    assert not shared_model.training
    assert torch.equal(first, encode_token_rows(shared_model, rows)['tangent'])

def test_no_rows_are_refused(shared_model):
    with pytest.raises(ValueError, match='no token rows'):
        encode_token_rows(shared_model, [])

# -------------------------------------------------------------------------------------------------
# The HGCN feeder
# -------------------------------------------------------------------------------------------------

def test_the_hgcn_feeder_writes_d_plus_one_lorentz_columns(
    monkeypatch, tmp_path, shared_checkpoint, validated_bundle, five_code_descriptions_parquet
):
    # The fixture bundle's description fingerprint hashes the frame, not the file, so the real
    # gate would refuse it
    monkeypatch.setattr(
        training_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle
    )
    monkeypatch.setattr(training_cli, 'pick_device', lambda *_args: torch.device('cpu'))
    # code_token_config keeps the default ./data/token_cache path; the descriptions path is
    # absolute, so it survives the move
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(five_code_descriptions_parquet)
    cfg.data_loader.streaming.max_length = TOKEN_WINDOW
    output = tmp_path / 'encodings.parquet'

    # No cache exists yet: the feeder builds it (P19), where it used to fail fast
    training_cli.generate_embeddings_from_checkpoint(
        str(shared_checkpoint), cfg, str(output), batch_size=2
    )

    table = pl.read_parquet(output)
    columns = [f'hyp_e{index}' for index in range(ARM_DIMENSION + 1)]
    assert table.columns == ['index', 'level', 'code', *columns]
    assert table.schema['index'] == pl.Int64
    assert table.schema['level'] == pl.Int64
    assert table.get_column('code').to_list() == list(FIVE_CODES)
    points = table.select(columns).to_numpy()
    # Each row lies on the hyperboloid: -x0^2 + |x|^2 = -1
    assert np.allclose(-points[:, 0]**2 + (points[:, 1:]**2).sum(axis=1), -1.0, atol=1e-4)
