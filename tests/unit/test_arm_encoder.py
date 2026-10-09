'''
The arm encoder: queries through the checkpoint's model, codes from its exported table
(spec 4.3).
'''

import inspect
import json

import polars as pl
import pytest
import torch

import naics_embedder.text_model.arm_encoder as arm_encoder_module
from naics_embedder.panels.decoding import GEOMETRY_DISTANCES, lorentz_distances
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.export import (
    encode_query_texts,
    encode_token_rows,
    export_code_table,
)
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.text_model.hyperbolic import HyperbolicHead, exp_map_origin
from tests.fixtures.shared_encoder import (
    ARM_DIMENSION,
    FIVE_CODES,
    PRE_STAGE_7_REFUSAL,
    TOKEN_WINDOW,
    five_code_token_rows,
    forbid_model_loads,
    lightning_checkpoint,
)

pytestmark = pytest.mark.unit

QUERIES = ['Edamame farming', 'Lignite mining']

@pytest.fixture
def arm(shared_checkpoint, exported_table, validated_bundle, five_code_token_config) -> ArmEncoder:
    return ArmEncoder.from_files(
        shared_checkpoint, exported_table, validated_bundle, five_code_token_config
    )

@pytest.fixture
def no_model_load(monkeypatch):
    '''Fails the test if a read loads the model: every provenance check comes first.'''

    def never(*_args, **_kwargs):
        # AssertionError, so a test's pytest.raises(ValueError) cannot swallow an unwanted load
        raise AssertionError('the model loaded before the provenance was refused')

    monkeypatch.setattr('naics_embedder.text_model.arm_encoder.load_arm_model', never)

def _table_tangent(table_path) -> torch.Tensor:
    table = pl.read_parquet(table_path)
    return torch.tensor(table.select(pl.exclude('code', 'index', 'level')).to_numpy())

# -------------------------------------------------------------------------------------------------
# The exp map at the origin
# -------------------------------------------------------------------------------------------------

def test_the_exp_map_lands_on_the_hyperboloid_as_the_heads_does():
    # The head bounds every vector, so its bounded tangent is what the export writes and the exp
    # map reads. Float64, so radii up to the bound compare exactly: x0 is about 1,490 at r = 8
    directions = torch.randn(6, ARM_DIMENSION, dtype=torch.float64)
    directions = directions / directions.norm(dim=1, keepdim=True)
    norms = torch.tensor([0.0, 0.1, 1.0, 5.0, 20.0, 100.0], dtype=torch.float64)
    head_points = HyperbolicHead()(norms.unsqueeze(1) * directions)

    points = exp_map_origin(head_points.tangent)

    assert points.dtype == torch.float64
    assert points.shape == (6, ARM_DIMENSION + 1)
    lorentz_norm = -points[:, 0]**2 + (points[:, 1:]**2).sum(dim=1)
    # |<x, x>_L + 1| within 1e-9 * x0^2, the "Radius" check's bound
    assert ((lorentz_norm + 1.0).abs() <= 1e-9 * points[:, 0]**2).all()
    origin = torch.zeros(ARM_DIMENSION + 1, dtype=torch.float64)
    origin[0] = 1.0
    assert torch.equal(points[0], origin)
    torch.testing.assert_close(points, head_points.embedding, rtol=1e-12, atol=1e-12)

# -------------------------------------------------------------------------------------------------
# Queries and codes
# -------------------------------------------------------------------------------------------------

def test_a_query_embeds_through_the_same_forward_as_a_code(arm):
    '''Spec §6: encode_queries([T]) is the float64 exp map of the forward's {'query': [T]}.'''

    tokens = tokenize_field(arm.tokenizer, QUERY, 'Edamame farming', TOKEN_WINDOW)
    with torch.no_grad():
        output = arm.model(stack_text_inputs([{QUERY: tokens}], fields=(QUERY, )))

    assert torch.equal(arm.encode_queries(['Edamame farming']), exp_map_origin(output['tangent']))

def test_queries_encode_through_the_query_path_the_training_monitor_shares(arm, monkeypatch):
    '''One query path, so a live read and a read of the export put a query at one point.'''

    shared = arm_encoder_module.encode_query_texts
    calls = []

    def spy(*args, **kwargs):
        calls.append(inspect.signature(shared).bind(*args, **kwargs).arguments)
        return shared(*args, **kwargs)

    monkeypatch.setattr(arm_encoder_module, 'encode_query_texts', spy)

    queries = arm.encode_queries(QUERIES)

    assert calls == [
        {
            'model': arm.model,
            'tokenizer': arm.tokenizer,
            'texts': QUERIES,
            'max_length': arm.max_length,
            'batch_size': arm.batch_size,
        }
    ]
    expected = shared(arm.model, arm.tokenizer, QUERIES, arm.max_length)
    assert torch.equal(queries, exp_map_origin(expected))

def test_codes_decode_from_the_table_in_the_order_asked(arm, exported_table):
    tangent = _table_tangent(exported_table)

    decoded = arm.encode_codes(['222222', '111111'])

    assert torch.equal(decoded, exp_map_origin(tangent[[3, 0]]))

def test_table_decoded_distances_match_the_live_forward(
    arm, five_code_token_config, validated_bundle
):
    '''Spec §6: decoding from the table matches the live forward within float32 tolerance.'''

    rows = five_code_token_rows(five_code_token_config, validated_bundle)
    # The head's own float32 points, from the forward the export ran
    live = encode_token_rows(arm.model, rows)['embedding']
    queries = arm.encode_queries(QUERIES)

    decoded = arm.encode_codes(list(FIVE_CODES))

    assert torch.allclose(
        lorentz_distances(queries, decoded), lorentz_distances(queries, live), atol=1e-4
    )

def test_an_unknown_code_is_refused(arm):
    with pytest.raises(ValueError, match='has no row for'):
        arm.encode_codes(['111111', '999999'])

def test_the_distance_is_the_heads(arm):
    assert arm.distance == arm.model.encoder.head.distance == 'lorentz'

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_a_flat_arm_reads_its_exported_points_as_they_are_under_its_own_distance(
    tmp_path, geometry_checkpoint, validated_bundle, five_code_token_config, geometry
):
    '''Req 12: each arm decodes by its own distance, and a flat arm's read map is the identity.'''

    checkpoint = geometry_checkpoint(geometry)
    table = export_code_table(
        checkpoint, validated_bundle, five_code_token_config, tmp_path / f'{geometry}.parquet'
    )
    flat = ArmEncoder.from_files(checkpoint, table, validated_bundle, five_code_token_config)
    log_path = tmp_path / 'selection_log.jsonl'

    read_outcome_validation(
        flat, OutcomePanel.from_bundle(validated_bundle, log_path), 'plan 12 fixture read'
    )

    assert flat.distance == GEOMETRY_DISTANCES[geometry]
    assert torch.equal(flat.encode_codes(['222222', '111111']), _table_tangent(table)[[3, 0]])
    queries = encode_query_texts(flat.model, flat.tokenizer, QUERIES, flat.max_length)
    assert torch.equal(flat.encode_queries(QUERIES), queries)
    [record] = SelectionLog(log_path).records()
    assert record['detail']['distance'] == GEOMETRY_DISTANCES[geometry]

def test_the_logged_names_are_the_tables_and_the_checkpoints(
    arm, exported_table, shared_checkpoint
):
    provenance = json.loads(provenance_path(exported_table).read_text())

    assert arm.table_fingerprint == provenance['matrix_fingerprint']
    # The name tools regressor-panel logs the same table by
    assert arm.table_fingerprint == table_fingerprint(pl.read_parquet(exported_table))
    assert arm.checkpoint_sha256 == sha256_file(shared_checkpoint)

# -------------------------------------------------------------------------------------------------
# Refusals
# -------------------------------------------------------------------------------------------------

def test_a_table_of_another_checkpoint_is_refused(
    tmp_path, shared_model, exported_table, validated_bundle, five_code_token_config
):
    checkpoint = lightning_checkpoint(shared_model)
    # The same weights in a file with other bytes, as another run's would be
    checkpoint['note'] = 'another run'
    other = tmp_path / 'other.ckpt'
    torch.save(checkpoint, other)

    with pytest.raises(ValueError, match='another checkpoint'):
        ArmEncoder.from_files(other, exported_table, validated_bundle, five_code_token_config)

def test_an_edited_table_is_refused(
    exported_table, shared_checkpoint, validated_bundle, five_code_token_config
):
    pl.read_parquet(exported_table).with_columns(pl.col('e0') * 2).write_parquet(exported_table)

    with pytest.raises(ValueError, match='not the table its provenance names'):
        ArmEncoder.from_files(
            shared_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_a_pre_stage_7_checkpoint_is_refused_on_read_before_its_model_loads(
    monkeypatch, pre_stage7_checkpoint, exported_table, validated_bundle, five_code_token_config
):
    '''Spec 4.5: the outcome read refuses it on its objective, and nothing migrates it (D2).'''

    # The provenance names the pre-Stage-7 checkpoint, so its checks pass and only the
    # checkpoint's contract can refuse it
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    provenance['checkpoint']['sha256'] = sha256_file(pre_stage7_checkpoint)
    path.write_text(json.dumps(provenance))
    forbid_model_loads(monkeypatch)

    with pytest.raises(ValueError, match=PRE_STAGE_7_REFUSAL):
        ArmEncoder.from_files(
            pre_stage7_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_a_text_only_table_is_refused_before_any_model_loads(
    no_model_load, shared_checkpoint, text_only_comparator_table, validated_bundle,
    five_code_token_config
):
    '''Its provenance has a table hash and a window, like an export's, but names no checkpoint.'''

    with pytest.raises(
        ValueError, match='is not an exported arm table: its provenance names no checkpoint$'
    ) as refusal:
        ArmEncoder.from_files(
            shared_checkpoint, text_only_comparator_table, validated_bundle, five_code_token_config
        )

    assert text_only_comparator_table.name in str(refusal.value)

@pytest.mark.parametrize('missing', ['checkpoint', 'table_sha256', 'max_length'])
def test_a_provenance_missing_an_entry_is_refused_before_any_model_loads(
    no_model_load, missing, exported_table, shared_checkpoint, validated_bundle,
    five_code_token_config
):
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    del provenance[missing]
    path.write_text(json.dumps(provenance))

    with pytest.raises(
        ValueError, match=f'is not an exported arm table: its provenance names no {missing}$'
    ):
        ArmEncoder.from_files(
            shared_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_a_provenance_naming_no_checkpoint_hash_is_refused_before_any_model_loads(
    no_model_load, exported_table, shared_checkpoint, validated_bundle, five_code_token_config
):
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    del provenance['checkpoint']['sha256']
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=r'its provenance names no checkpoint\.sha256$'):
        ArmEncoder.from_files(
            shared_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_a_table_exported_at_another_window_is_refused_before_any_model_loads(
    no_model_load, exported_table, shared_checkpoint, validated_bundle, five_code_token_config
):
    '''One preprocessing contract: queries take the window the table's codes were encoded at.'''

    wider = five_code_token_config.model_copy(update={'max_length': 2 * TOKEN_WINDOW})

    with pytest.raises(
        ValueError,
        match=f'exported at a {TOKEN_WINDOW}-token window.*tokenizes queries at {2 * TOKEN_WINDOW}',
    ) as refusal:
        ArmEncoder.from_files(shared_checkpoint, exported_table, validated_bundle, wider)

    # The key the read's window comes from (code_token_config), so the remedy points at it
    assert 'data_loader.streaming.max_length' in str(refusal.value)

@pytest.mark.parametrize('missing', ['summaries', 'tokenizer'])
def test_a_table_exported_before_stage_6b_is_refused_before_any_model_loads(
    no_model_load, missing, exported_table, shared_checkpoint, validated_bundle,
    five_code_token_config
):
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    del provenance[missing]
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=f'before Stage 6b: its provenance records no {missing};'):
        ArmEncoder.from_files(
            shared_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

@pytest.mark.parametrize(
    ('entry', 'value', 'refusal'),
    [
        ('tokenizer', 'other/tokenizer', 'exported with the tokenizer other/tokenizer'),
        ('summaries', None, 'exported under the summaries None'),
        ('summaries', 'f' * 64, f"exported under the summaries {'f' * 64}"),
    ],
)
def test_a_table_read_under_another_tokenizer_or_summaries_is_refused_before_any_model_loads(
    no_model_load, entry, value, refusal, exported_table, shared_checkpoint, validated_bundle,
    five_code_token_config
):
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    provenance[entry] = value
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=refusal):
        ArmEncoder.from_files(
            shared_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_a_checkpoint_trained_on_truncated_text_is_refused_on_read(
    truncated_checkpoint, exported_table, validated_bundle, five_code_token_config
):
    # The provenance names the truncated checkpoint, so only its contract can refuse it
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    provenance['checkpoint']['sha256'] = sha256_file(truncated_checkpoint)
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="supervision contract mismatch .*'summaries'"):
        ArmEncoder.from_files(
            truncated_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_queries_are_tokenized_at_the_tables_window(arm, exported_table):
    provenance = json.loads(provenance_path(exported_table).read_text())

    assert arm.max_length == provenance['max_length'] == TOKEN_WINDOW

def test_a_missing_checkpoint_is_a_file_not_found_before_any_model_loads(
    no_model_load, tmp_path, exported_table, validated_bundle, five_code_token_config
):
    with pytest.raises(FileNotFoundError):
        ArmEncoder.from_files(
            tmp_path / 'missing.ckpt', exported_table, validated_bundle, five_code_token_config
        )

# -------------------------------------------------------------------------------------------------
# Devices
# -------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs an MPS device')
def test_queries_and_codes_on_mps_come_back_float64_on_the_cpu(
    arm, shared_checkpoint, exported_table, validated_bundle, five_code_token_config
):
    '''Spec §6: on MPS, encode_queries and encode_codes return float64 CPU tensors.'''

    on_mps = ArmEncoder.from_files(
        shared_checkpoint, exported_table, validated_bundle, five_code_token_config, device='mps'
    )

    queries = on_mps.encode_queries(QUERIES)
    codes = on_mps.encode_codes(list(FIVE_CODES))

    for vectors in (queries, codes):
        assert vectors.dtype == torch.float64
        assert vectors.device.type == 'cpu'
    assert torch.allclose(queries, arm.encode_queries(QUERIES), atol=1e-4)
    assert torch.equal(codes, arm.encode_codes(list(FIVE_CODES)))

# -------------------------------------------------------------------------------------------------
# The outcome read
# -------------------------------------------------------------------------------------------------

def test_a_validation_read_logs_the_table_it_decodes_against(
    tmp_path, arm, exported_table, validated_bundle
):
    '''Spec §6: a read on a fixture panel logs table equal to the table's matrix_fingerprint.'''

    log_path = tmp_path / 'selection_log.jsonl'
    panel = OutcomePanel.from_bundle(validated_bundle, log_path)

    result = read_outcome_validation(arm, panel, 'plan 8 fixture read')

    # The five-code bundle's one validation entry: 'Edamame farming', for 111111
    [record] = SelectionLog(log_path).records()
    assert (record['event'], record['split'], record['n_queries']) == ('read', 'validation', 1)
    assert record['purpose'] == 'plan 8 fixture read'
    assert record['detail'] == {
        'encoder': 'ArmEncoder',
        'distance': 'lorentz',
        'table': table_fingerprint(pl.read_parquet(exported_table)),
        'checkpoint': arm.checkpoint_sha256,
    }
    assert (result.summary['n_queries'], result.summary['n_candidates']) == (1, 5)
