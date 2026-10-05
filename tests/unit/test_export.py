'''
Encoding token rows through an arm's model, the HGCN feeder built on it, and the code-table
export (spec 4.3).
'''

import json

import numpy as np
import polars as pl
import pytest
import torch
from transformers import AutoTokenizer

from naics_embedder.cli.commands import training as training_cli
from naics_embedder.panels.regressor import coordinate_matrix, table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.supervision.checkpoints import contract_for_bundle, shared_encoder_architecture
from naics_embedder.text_model import export as encoding
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.export import (
    COORDINATES,
    code_token_config,
    encode_token_rows,
    export_code_table,
    load_arm_model,
)
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.utils.config import Config
from tests.fixtures.shared_encoder import (
    ARM_DIMENSION,
    FIVE_CODES,
    MINILM,
    PRE_STAGE_7_REFUSAL,
    TOKEN_WINDOW,
    five_code_token_rows,
    forbid_model_loads,
    lightning_checkpoint,
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

def test_rows_encode_each_rows_radius_and_direction_too(
    shared_model, five_code_token_config, validated_bundle
):
    '''The training cache reads r and û from the export's own encode (spec 4.3).'''

    rows = five_code_token_rows(five_code_token_config, validated_bundle)

    encoded = encode_token_rows(shared_model, rows)

    assert set(encoded) == {'tangent', 'embedding', 'radius', 'direction'}
    assert encoded['radius'].shape == (5, )
    assert encoded['direction'].shape == (5, ARM_DIMENSION)
    # Five rows are one batch, so each output is the whole forward's float32 value, in float64
    with torch.no_grad():
        whole = shared_model(stack_text_inputs(rows))
    for name in ('radius', 'direction'):
        assert (encoded[name].dtype, encoded[name].device.type) == (torch.float64, 'cpu')
        assert torch.equal(encoded[name], whole[name].to(torch.float64))

def test_query_texts_encode_marked_through_the_model(shared_model):
    '''One query path for the arm encoder and the training monitor (spec 4.4).'''

    tokenizer = AutoTokenizer.from_pretrained(MINILM)
    texts = ['Edamame farming', 'Lignite mining', 'Coal mining']

    tangent = encoding.encode_query_texts(shared_model, tokenizer, texts, TOKEN_WINDOW)

    rows = [{QUERY: tokenize_field(tokenizer, QUERY, text, TOKEN_WINDOW)} for text in texts]
    with torch.no_grad():
        whole = shared_model(stack_text_inputs(rows, fields=(QUERY, )))
    assert (tangent.dtype, tangent.device.type) == (torch.float64, 'cpu')
    assert tangent.shape == (3, ARM_DIMENSION)
    assert torch.equal(tangent, whole['tangent'].to(torch.float64))

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

@pytest.mark.parametrize(
    'device', [
        'cpu',
        pytest.param(
            'mps',
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason='MPS is unavailable'
            )
        ),
    ]
)
def test_the_hgcn_feeder_writes_d_plus_one_lorentz_columns(
    monkeypatch, tmp_path, shared_checkpoint, validated_bundle, five_code_descriptions_parquet,
    device
):
    # The fixture bundle's description fingerprint hashes the frame, not the file, so the real
    # gate would refuse it
    monkeypatch.setattr(
        training_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle
    )
    monkeypatch.setattr(training_cli, 'pick_device', lambda *_args: torch.device(device))
    saved = torch.load(shared_checkpoint, map_location='cpu', weights_only=False)
    saved['callbacks'] = {
        'ModelCheckpoint': {
            'best_model_score': torch.tensor(0.25, dtype=torch.float64)
        }
    }
    torch.save(saved, shared_checkpoint)
    original_load = training_cli.NAICSContrastiveModel.load_from_checkpoint

    def cpu_load(*args, **kwargs):
        assert kwargs['map_location'] == 'cpu'
        return original_load(*args, **kwargs)

    monkeypatch.setattr(training_cli.NAICSContrastiveModel, 'load_from_checkpoint', cpu_load)
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

def test_the_hgcn_feeder_refuses_a_checkpoint_trained_on_truncated_text(
    monkeypatch, tmp_path, truncated_checkpoint, validated_bundle, five_code_descriptions_parquet
):
    monkeypatch.setattr(
        training_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle
    )
    # The refusal comes first. A regression past it would pick a device and build the default
    # ./data/token_cache: keep that on the CPU and under tmp_path
    monkeypatch.setattr(training_cli, 'pick_device', lambda *_args: torch.device('cpu'))
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(five_code_descriptions_parquet)
    output = tmp_path / 'encodings.parquet'

    with pytest.raises(ValueError, match="exact resume contract mismatch .*'summaries'"):
        training_cli.generate_embeddings_from_checkpoint(
            str(truncated_checkpoint), cfg, str(output)
        )
    assert not output.exists()

def test_the_hgcn_feeder_refuses_a_pre_stage_7_checkpoint_before_its_model_loads(
    monkeypatch, tmp_path, pre_stage7_checkpoint, validated_bundle, five_code_descriptions_parquet
):
    '''Spec 4.5: on its objective, as exact resume refuses it, and nothing migrates it (D2).'''

    monkeypatch.setattr(
        training_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle
    )
    # The refusal comes first. A regression past it would pick a device and build the default
    # ./data/token_cache: keep that on the CPU and under tmp_path
    monkeypatch.setattr(training_cli, 'pick_device', lambda *_args: torch.device('cpu'))
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(five_code_descriptions_parquet)
    output = tmp_path / 'encodings.parquet'
    forbid_model_loads(monkeypatch)

    with pytest.raises(ValueError, match=PRE_STAGE_7_REFUSAL):
        training_cli.generate_embeddings_from_checkpoint(
            str(pre_stage7_checkpoint), cfg, str(output)
        )
    assert not output.exists()

# -------------------------------------------------------------------------------------------------
# The code-table export
# -------------------------------------------------------------------------------------------------

COORDINATE_COLUMNS = [f'e{index}' for index in range(ARM_DIMENSION)]

def test_the_table_is_in_reqs_export_form(exported_table):
    '''Spec §6: code, index, level and e0 … e15 as float64, readable by coordinate_matrix.'''

    table = pl.read_parquet(exported_table)

    assert table.columns == ['code', 'index', 'level', *COORDINATE_COLUMNS]
    assert dict(table.schema) == {
        'code': pl.Utf8,
        'index': pl.Int64,
        'level': pl.Int64,
        **{
            column: pl.Float64
            for column in COORDINATE_COLUMNS
        },
    }
    # The bundle's codebook order
    assert table.get_column('code').to_list() == list(FIVE_CODES)
    assert table.get_column('index').to_list() == [0, 1, 2, 3, 4]
    codes, matrix = coordinate_matrix(table)
    assert codes == FIVE_CODES
    assert matrix.shape == (5, ARM_DIMENSION)

def test_the_table_holds_each_codes_bounded_tangent(
    exported_table, shared_checkpoint, validated_bundle, five_code_token_config
):
    model, _ = load_arm_model(
        shared_checkpoint, validated_bundle, summaries=summaries_identity(MINILM)
    )
    rows = five_code_token_rows(five_code_token_config, validated_bundle)
    tangent = encode_token_rows(model, rows)['tangent']

    table = pl.read_parquet(exported_table)

    assert np.array_equal(table.select(COORDINATE_COLUMNS).to_numpy(), tangent.numpy())
    # The head bounds every radius below R before its exp map; the table keeps the bounded vector
    bound = model.encoder.head.radius_bound
    assert bound == 8.0
    assert (np.linalg.norm(tangent.numpy(), axis=1) <= bound).all()
    provenance = json.loads(provenance_path(exported_table).read_text())
    assert provenance['coordinates'] == COORDINATES
    assert COORDINATES.startswith('the bounded tangent vector at the origin')

def test_a_read_rebuilds_the_head_at_the_checkpoints_radius_bound(
    tmp_path, shared_model, validated_bundle
):
    # The bound is a saved hyperparameter, so export and reads never fall back to the default R
    checkpoint = lightning_checkpoint(shared_model)
    checkpoint['hyper_parameters']['radius_bound'] = 5.0
    path = tmp_path / 'bound.ckpt'
    torch.save(checkpoint, path)

    model, _ = load_arm_model(path, validated_bundle, summaries=summaries_identity(MINILM))

    assert model.encoder.head.radius_bound == 5.0

@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param(
            'mps',
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason='MPS is unavailable'
            )
        ),
    ],
)
def test_a_load_keeps_float64_callback_scores_off_the_model_device(
    tmp_path, shared_model, validated_bundle, device
):
    checkpoint = lightning_checkpoint(shared_model)
    checkpoint['callbacks'] = {
        'ModelCheckpoint': {
            'best_model_score': torch.tensor(0.25, dtype=torch.float64)
        },
    }
    path = tmp_path / 'callback-score.ckpt'
    torch.save(checkpoint, path)

    model, contract = load_arm_model(
        path, validated_bundle, summaries=summaries_identity(MINILM), device=device
    )

    assert next(model.parameters()).device.type == device
    assert not model.training
    assert contract.objective == 'req11-v1'
    for name, tensor in shared_model.state_dict().items():
        assert torch.equal(model.state_dict()[name].cpu(), tensor.cpu())
    saved = torch.load(path, map_location='cpu', weights_only=False)
    score = saved['callbacks']['ModelCheckpoint']['best_model_score']
    assert score.dtype == torch.float64
    assert score.item() == 0.25

def test_the_provenance_names_the_table_and_the_checkpoint(
    exported_table, shared_checkpoint, validated_bundle, five_code_descriptions_parquet
):
    provenance = json.loads(provenance_path(exported_table).read_text())

    assert provenance['checkpoint'] == {
        'path': str(shared_checkpoint),
        'sha256': sha256_file(shared_checkpoint),
    }
    expected = contract_for_bundle(
        validated_bundle.manifest,
        encoder=shared_encoder_architecture(
            fusion='masked_mean', dimension=ARM_DIMENSION, backbone=MINILM
        ),
        summaries=summaries_identity(MINILM),
    )
    assert provenance['contract'] == expected.model_dump(mode='json')
    assert provenance['backbone'] == MINILM
    # The tiny backbone has no Hugging Face snapshot
    assert provenance['revision'] is None
    assert provenance['max_length'] == TOKEN_WINDOW
    assert provenance['descriptions'] == {
        'path': str(five_code_descriptions_parquet),
        'sha256': sha256_file(five_code_descriptions_parquet),
    }
    # The seam's dummy pin for MiniLM (tests/conftest.py)
    assert provenance['summaries'] == summaries_identity(MINILM)
    assert provenance['summaries'] is not None
    assert provenance['tokenizer'] == MINILM
    assert (provenance['codes'], provenance['dimension']) == (5, ARM_DIMENSION)
    assert provenance['table_sha256'] == sha256_file(exported_table)
    assert provenance['matrix_fingerprint'] == table_fingerprint(pl.read_parquet(exported_table))
    assert set(provenance['library_versions']) == {'peft', 'polars', 'torch', 'transformers'}

def test_a_load_refuses_a_pre_stage_7_checkpoint_before_its_model_loads(
    monkeypatch, pre_stage7_checkpoint, validated_bundle
):
    '''
    Spec 4.5: its hyperparameters name no radius bound, so a load would rebuild the head at the
    default R and read the old objective's weights as this one's. Its contract's objective refuses
    it first, and nothing migrates it (D2).
    '''

    forbid_model_loads(monkeypatch)

    with pytest.raises(ValueError, match=PRE_STAGE_7_REFUSAL):
        load_arm_model(
            pre_stage7_checkpoint, validated_bundle, summaries=summaries_identity(MINILM)
        )

def test_the_export_refuses_a_pre_stage_7_checkpoint(
    monkeypatch, tmp_path, pre_stage7_checkpoint, validated_bundle, five_code_token_config
):
    '''Spec 4.5: on its objective, before its model loads, so no table or provenance is written.'''

    output = tmp_path / 'table.parquet'
    forbid_model_loads(monkeypatch)

    with pytest.raises(ValueError, match=PRE_STAGE_7_REFUSAL):
        export_code_table(pre_stage7_checkpoint, validated_bundle, five_code_token_config, output)
    assert not output.exists()
    assert not provenance_path(output).exists()

def test_a_checkpoint_of_another_bundle_is_refused(
    tmp_path, shared_model, validated_bundle, five_code_token_config
):
    checkpoint = lightning_checkpoint(shared_model)
    checkpoint['stage3_supervision']['bundle_id'] = 'bundle-b'
    path = tmp_path / 'other-bundle.ckpt'
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match='supervision contract mismatch'):
        export_code_table(path, validated_bundle, five_code_token_config, tmp_path / 't.parquet')

def test_a_load_that_omits_the_summaries_is_a_type_error(shared_checkpoint, validated_bundle):
    with pytest.raises(TypeError):
        load_arm_model(shared_checkpoint, validated_bundle)

def test_a_checkpoint_trained_on_truncated_text_is_refused(
    tmp_path, truncated_checkpoint, validated_bundle, five_code_token_config
):
    output = tmp_path / 'table.parquet'

    with pytest.raises(ValueError, match="supervision contract mismatch .*'summaries'"):
        export_code_table(truncated_checkpoint, validated_bundle, five_code_token_config, output)
    assert not output.exists()

def test_a_four_copy_checkpoint_is_refused_with_d2(
    tmp_path, shared_model, validated_bundle, five_code_token_config
):
    checkpoint = lightning_checkpoint(shared_model)
    # Contracts saved before Stage 6 carry no encoder record, and their hyperparameters predate
    # fusion and dimension
    del checkpoint['stage3_supervision']['encoder']
    for name in ('fusion', 'dimension'):
        del checkpoint['hyper_parameters'][name]
    path = tmp_path / 'four-copy.ckpt'
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match='D2'):
        export_code_table(path, validated_bundle, five_code_token_config, tmp_path / 't.parquet')

def test_descriptions_that_are_not_the_codebook_are_refused(
    tmp_path, shared_checkpoint, validated_bundle, five_code_token_config, text_descriptions_fixture
):
    other = tmp_path / 'other_descriptions.parquet'
    text_descriptions_fixture.with_columns(
        level=pl.lit(6), code=pl.col('code').str.replace('333333', '333334', literal=True)
    ).write_parquet(other)
    token_config = five_code_token_config.model_copy(update={'descriptions_parquet': str(other)})

    with pytest.raises(ValueError, match='codebook'):
        export_code_table(shared_checkpoint, validated_bundle, token_config, tmp_path / 't.parquet')
