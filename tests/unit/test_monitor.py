'''
D6's monitor on the live model (spec 4.4): the code cache, the live encoder, the logged read and
the run's ``monitor_reads.jsonl``.

The five-code arm (``tests/fixtures/shared_encoder.py``) carries the unit tests; its bundle has one
validation entry. The agreement test reads the reference bundle (``tests/fixtures/supervision.py``),
whose validation split has an entry for each of its four six-digit codes, so the two reads it
compares rank each query against the others.
'''

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import pytest
import torch
from transformers import AutoTokenizer

from naics_embedder.panels.decoding import GEOMETRY_DISTANCES, score_decoding
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.text_model import monitor
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import (
    ENCODE_BATCH_SIZE,
    encode_query_texts,
    encode_token_rows,
    export_code_table,
)
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.hyperbolic import exp_map_origin
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import TokenizationConfig
from tests.fixtures.shared_encoder import (
    ARM_DIMENSION,
    FIVE_CODES,
    MINILM,
    TOKEN_WINDOW,
    five_code_token_rows,
    lightning_checkpoint,
)
from tests.fixtures.supervision import REFERENCE_INDEX_ROLE_ROWS

pytestmark = pytest.mark.unit

PURPOSE = 'D6 monitor fixture read'
QUERIES = ['Edamame farming', 'Lignite mining']
# The reference bundle's longest marked channel text has 44 tokens; the dummy pin fits 128
REFERENCE_WINDOW = 128

@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(MINILM)

@pytest.fixture
def code_rows(five_code_token_config, validated_bundle) -> List[Dict[str, Any]]:
    return five_code_token_rows(five_code_token_config, validated_bundle)

@pytest.fixture
def cache(shared_model, code_rows) -> 'monitor.CodeCache':
    return monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)

@pytest.fixture
def panel(tmp_path, validated_bundle) -> OutcomePanel:
    return OutcomePanel.from_bundle(validated_bundle, tmp_path / 'logs' / 'selection_log.jsonl')

@pytest.fixture
def outcome_monitor(tmp_path, panel, tokenizer) -> 'monitor.OutcomeMonitor':
    records = tmp_path / 'checkpoints' / monitor.MONITOR_RECORDS
    return monitor.OutcomeMonitor(panel, tokenizer, TOKEN_WINDOW, records, PURPOSE)

@pytest.fixture
def arm(shared_checkpoint, exported_table, validated_bundle, five_code_token_config) -> ArmEncoder:
    return ArmEncoder.from_files(
        shared_checkpoint, exported_table, validated_bundle, five_code_token_config
    )

def _flags(model: torch.nn.Module) -> Dict[str, bool]:
    return {name: module.training for name, module in model.named_modules()}

def _mixed_training_flags(model: torch.nn.Module) -> Dict[str, bool]:
    '''Training mode, but one submodule in eval mode, which a plain ``train()`` would not keep.'''

    model.train()
    model.encoder.projection.eval()
    return _flags(model)

def _synthetic_read(epoch: int, mrr: float = 0.5) -> 'monitor.MonitorRead':
    '''A read shaped as the monitor logs one, without a model.'''

    record = {
        'detail': {
            'distance': 'lorentz',
            'encoder': 'LiveEncoder',
            'epoch': epoch,
            'seed': 0,
            'table': 'f' * 64,
            'training_run': 'run-a',
        },
        'event': 'read',
        'fingerprint': 'a' * 64,
        'n_queries': 1,
        'panel': 'outcome',
        'purpose': PURPOSE,
        'split': 'validation',
        'time': f'2026-10-04T00:00:{epoch:02d}+00:00',
    }
    return monitor.MonitorRead(epoch=epoch, mrr=mrr, record=record)

def _recorded_epochs(path: Path) -> List[int]:
    return [record['read']['detail']['epoch'] for record in monitor.read_monitor_records(path)]

# -------------------------------------------------------------------------------------------------
# Training flags
# -------------------------------------------------------------------------------------------------

def test_the_training_flags_are_restored_on_exit_even_after_an_error():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5), torch.nn.Linear(2, 2))
    model.train()
    model[2].eval()
    before = [module.training for module in model.modules()]
    assert before == [True, True, True, False]

    with monitor.preserve_training_flags(model):
        model.eval()
        assert not any(module.training for module in model.modules())

    assert [module.training for module in model.modules()] == before
    with pytest.raises(RuntimeError, match='inside the context'):
        with monitor.preserve_training_flags(model):
            model.eval()
            raise RuntimeError('inside the context')
    assert [module.training for module in model.modules()] == before

# -------------------------------------------------------------------------------------------------
# The code cache
# -------------------------------------------------------------------------------------------------

def test_a_refresh_holds_every_codes_point_in_codebook_order(shared_model, code_rows):
    cache = monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)
    # Five codes are one batch, so the refresh is the model's whole forward, in eval mode
    with torch.no_grad():
        head = shared_model(stack_text_inputs(code_rows))

    assert cache.codes == FIVE_CODES
    assert cache.radius.shape == (5, )
    assert cache.direction.shape == cache.tangent.shape == (5, ARM_DIMENSION)
    assert (cache.radius.dtype, cache.direction.dtype) == (torch.float32, torch.float32)
    assert cache.radius.device == cache.direction.device == next(shared_model.parameters()).device
    assert (cache.tangent.dtype, cache.tangent.device.type) == (torch.float64, 'cpu')
    # The head's own float32 values, which survive the float64 round trip exactly
    assert torch.equal(cache.radius, head['radius'])
    assert torch.equal(cache.direction, head['direction'])
    assert torch.equal(cache.tangent, head['tangent'].to(torch.float64))

def test_a_refresh_runs_in_eval_mode_and_restores_the_training_flags(shared_model, code_rows):
    evaluated = monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)
    flags = _mixed_training_flags(shared_model)
    # BERT's dropout makes a training-mode pass differ, so the equality below is not vacuous
    with torch.no_grad():
        trained = shared_model(stack_text_inputs(code_rows))['tangent'].to(torch.float64)
    assert not torch.equal(trained, evaluated.tangent)

    refreshed = monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)

    assert _flags(shared_model) == flags
    assert torch.equal(refreshed.tangent, evaluated.tangent)
    assert torch.equal(refreshed.radius, evaluated.radius)

def test_a_refresh_takes_no_gradient(shared_model, code_rows):
    cache = monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)

    assert torch.is_grad_enabled()
    for tensor in (cache.radius, cache.direction, cache.tangent):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None

def test_a_refresh_under_bf16_autocast_is_the_float32_refresh(shared_model, code_rows):
    '''Spec 4.3: the refresh runs in float32, the backbone included, with autocast off.'''

    plain = monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        refreshed = monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)
        # Without the refresh's guard, the backbone runs in bf16 under this autocast
        with torch.no_grad():
            reduced = shared_model(stack_text_inputs(code_rows))['tangent'].to(torch.float64)

    assert not torch.equal(reduced, plain.tangent)
    for name in ('radius', 'direction', 'tangent'):
        assert torch.equal(getattr(refreshed, name), getattr(plain, name))

@pytest.mark.parametrize(
    ('row_count', 'codes', 'refusal'),
    [
        (4, FIVE_CODES, '4 token rows for 5 codes'),
        (5, FIVE_CODES[:4] + FIVE_CODES[:1], "repeats the code '111111'"),
    ],
    ids=['a-row-short', 'a-repeated-code'],
)
def test_a_refresh_refuses_rows_that_are_not_one_per_code(
    shared_model, code_rows, row_count, codes, refusal
):
    with pytest.raises(ValueError, match=refusal):
        monitor.refresh_code_cache(shared_model, code_rows[:row_count], codes)

def test_the_anchors_rows_are_their_live_points_and_every_other_row_is_the_caches(cache):
    saved_radius, saved_direction = cache.radius.clone(), cache.direction.clone()
    ids = torch.tensor([3, 1])
    live_radius = torch.tensor([0.5, 1.5], requires_grad=True)
    live_direction = torch.nn.functional.normalize(torch.randn(2, ARM_DIMENSION), dim=1)
    live_direction.requires_grad_()

    radius, direction = cache.with_live(ids, live_radius, live_direction)

    assert torch.equal(radius[ids], live_radius.detach())
    assert torch.equal(direction[ids], live_direction.detach())
    others = torch.tensor([0, 2, 4])
    assert torch.equal(radius[others], saved_radius[others])
    assert torch.equal(direction[others], saved_direction[others])
    # Out of place: the cache itself is unchanged, and no row of it takes gradient
    assert torch.equal(cache.radius, saved_radius)
    assert torch.equal(cache.direction, saved_direction)
    assert not (cache.radius.requires_grad or cache.direction.requires_grad)
    # Gradient reaches the anchors' live points, and through their rows only
    weights = torch.arange(1.0, 6.0)
    ((radius * weights).sum() + (direction * weights[:, None]).sum()).backward()
    assert torch.equal(live_radius.grad, weights[ids])
    assert torch.equal(live_direction.grad, weights[ids][:, None].expand(2, ARM_DIMENSION))

@pytest.mark.parametrize(
    ('ids', 'radius', 'direction'),
    [
        (torch.tensor([[3, 1]]), torch.zeros(2), torch.zeros(2, ARM_DIMENSION)),
        (torch.tensor([3, 1]), torch.zeros(2, 1), torch.zeros(2, ARM_DIMENSION)),
        (torch.tensor([3, 1]), torch.zeros(3), torch.zeros(2, ARM_DIMENSION)),
        (torch.tensor([3, 1]), torch.zeros(2), torch.zeros(2, ARM_DIMENSION + 1)),
    ],
    ids=['two-dimensional-ids', 'radii-that-are-not-1-d', 'another-count', 'another-width'],
)
def test_live_rows_that_do_not_fit_the_cache_are_refused(cache, ids, radius, direction):
    with pytest.raises(ValueError, match=r'with_live takes anchor ids \(A,\), radii \(A,\)'):
        cache.with_live(ids, radius, direction)

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs an MPS device')
def test_on_mps_the_points_stay_on_the_device_and_the_tangents_come_to_the_cpu(
    shared_model, code_rows
):
    on_cpu = monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)
    shared_model.to('mps')

    cache = monitor.refresh_code_cache(shared_model, code_rows, FIVE_CODES)

    assert cache.radius.device.type == cache.direction.device.type == 'mps'
    assert (cache.radius.dtype, cache.direction.dtype) == (torch.float32, torch.float32)
    assert (cache.tangent.dtype, cache.tangent.device.type) == (torch.float64, 'cpu')
    assert torch.allclose(cache.tangent, on_cpu.tangent, atol=1e-4)
    radius, direction = cache.with_live(
        torch.tensor([0], device='mps'), cache.radius[:1] + 1.0, cache.direction[:1]
    )
    assert radius.device.type == direction.device.type == 'mps'
    assert torch.equal(radius[1:].cpu(), cache.radius[1:].cpu())

# -------------------------------------------------------------------------------------------------
# The live encoder
# -------------------------------------------------------------------------------------------------

def test_live_queries_go_through_the_model_as_the_arm_encoders_do(
    shared_model, cache, tokenizer, arm
):
    live = monitor.LiveEncoder(shared_model, cache, tokenizer, TOKEN_WINDOW)

    queries = live.encode_queries(QUERIES)

    assert (queries.dtype, queries.device.type) == (torch.float64, 'cpu')
    assert queries.shape == (2, ARM_DIMENSION + 1)
    assert torch.equal(queries, arm.encode_queries(QUERIES))
    assert torch.equal(
        queries, exp_map_origin(encode_query_texts(shared_model, tokenizer, QUERIES, TOKEN_WINDOW))
    )

def test_live_queries_run_in_eval_mode_without_autocast_and_restore_the_flags(
    shared_model, cache, tokenizer
):
    live = monitor.LiveEncoder(shared_model, cache, tokenizer, TOKEN_WINDOW)
    plain = live.encode_queries(QUERIES)
    flags = _mixed_training_flags(shared_model)

    with torch.autocast('cpu', dtype=torch.bfloat16):
        queries = live.encode_queries(QUERIES)

    assert _flags(shared_model) == flags
    assert torch.equal(queries, plain)

def test_live_codes_are_their_cached_tangents_through_the_exp_map(shared_model, cache, tokenizer):
    live = monitor.LiveEncoder(shared_model, cache, tokenizer, TOKEN_WINDOW)

    assert torch.equal(
        live.encode_codes(['222222', '111111']), exp_map_origin(cache.tangent[[3, 0]])
    )
    assert live.distance == 'lorentz'
    with pytest.raises(ValueError, match=r"has no row for \['999999'\]"):
        live.encode_codes(['111111', '999999'])

# -------------------------------------------------------------------------------------------------
# The monitor's read
# -------------------------------------------------------------------------------------------------

def test_a_read_is_logged_as_an_outcome_validation_read_naming_the_run_seed_epoch_and_table(
    panel, outcome_monitor, shared_model, cache, tokenizer
):
    '''Spec §6: the read is an outcome validation read with the training id, seed, epoch, table.'''

    read = outcome_monitor.read(shared_model, cache, training_run='run-a', seed=3, epoch=2)

    [record] = panel.log.records()
    assert (record['event'], record['panel'], record['split']) == ('read', 'outcome', 'validation')
    assert record['purpose'] == PURPOSE
    assert record['fingerprint'] == panel.fingerprint
    # The five-code bundle's one validation entry: 'Edamame farming', for 111111
    assert record['n_queries'] == 1
    assert record['detail'] == {
        'encoder': 'LiveEncoder',
        'distance': 'lorentz',
        'training_run': 'run-a',
        'seed': 3,
        'epoch': 2,
        'table': matrix_fingerprint(FIVE_CODES, cache.tangent.numpy()),
    }
    assert read.record == record
    assert read.epoch == 2
    # The MRR is the live model's decoding of the split, one float
    live = monitor.LiveEncoder(shared_model, cache, tokenizer, TOKEN_WINDOW)
    decoded = score_decoding(
        live.encode_queries(['Edamame farming']),
        ['111111'],
        live.encode_codes(list(FIVE_CODES)),
        FIVE_CODES,
        distance='lorentz',
    )
    assert type(read.mrr) is float
    assert read.mrr == decoded.summary['mrr']

def test_a_read_records_integer_seeds_and_epochs_of_any_integer_type(
    panel, outcome_monitor, shared_model, cache
):
    read = outcome_monitor.read(
        shared_model, cache, training_run='run-a', seed=np.int64(3), epoch=torch.tensor(2)
    )

    [record] = panel.log.records()
    assert (type(record['detail']['seed']), type(record['detail']['epoch'])) == (int, int)
    assert (record['detail']['seed'], record['detail']['epoch'], read.epoch) == (3, 2, 2)

@pytest.mark.parametrize(
    ('training_run', 'seed', 'epoch', 'error', 'refusal'),
    [
        ('', 0, 0, ValueError, 'names its training run'),
        ('   ', 0, 0, ValueError, 'names its training run'),
        (None, 0, 0, ValueError, 'names its training run'),
        ('run-a', 0, -1, ValueError, 'never negative'),
        ('run-a', 1.5, 0, TypeError, 'integer'),
        ('run-a', 0, 2.0, TypeError, 'integer'),
    ],
    ids=[
        'a-blank-run', 'a-space-run', 'no-run', 'a-negative-epoch', 'a-float-seed', 'a-float-epoch'
    ],
)
def test_a_read_refuses_what_it_cannot_record_before_anything_is_logged(
    panel, outcome_monitor, shared_model, cache, training_run, seed, epoch, error, refusal
):
    with pytest.raises(error, match=refusal):
        outcome_monitor.read(shared_model, cache, training_run=training_run, seed=seed, epoch=epoch)

    assert not panel.log.path.exists()

def test_a_read_refuses_a_cache_without_a_candidate_before_anything_is_logged(
    panel, outcome_monitor, shared_model, code_rows
):
    partial = monitor.refresh_code_cache(shared_model, code_rows[:4], FIVE_CODES[:4])

    with pytest.raises(ValueError, match=r"no row for the candidates \['333333'\]"):
        outcome_monitor.read(shared_model, partial, training_run='run-a', seed=0, epoch=0)

    assert not panel.log.path.exists()

@pytest.mark.parametrize('purpose', ['', '   '], ids=['empty', 'blank'])
def test_a_monitor_needs_a_purpose(tmp_path, panel, tokenizer, purpose):
    with pytest.raises(ValueError, match='needs a purpose'):
        monitor.OutcomeMonitor(
            panel, tokenizer, TOKEN_WINDOW, tmp_path / monitor.MONITOR_RECORDS, purpose
        )

# -------------------------------------------------------------------------------------------------
# The records file
# -------------------------------------------------------------------------------------------------

def test_each_read_is_appended_as_one_mrr_and_read_line(
    panel, outcome_monitor, shared_model, cache
):
    '''Spec §6: monitor_reads.jsonl is written, one {mrr, read} line per read.'''

    path = outcome_monitor.records_path
    assert path.name == monitor.MONITOR_RECORDS == 'monitor_reads.jsonl'
    outcome_monitor.start(resumed_epoch=None)
    # A fresh start writes nothing until the first read
    assert not path.exists()

    reads = [
        outcome_monitor.read(shared_model, cache, training_run='run-a', seed=0, epoch=epoch)
        for epoch in (0, 1)
    ]
    for read in reads:
        outcome_monitor.append(read)

    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines == [
        json.dumps({
            'mrr': read.mrr,
            'read': read.record
        }, sort_keys=True) for read in reads
    ]
    records = monitor.read_monitor_records(path)
    # Each line's read is the selection log's record, as logged
    assert [record['read'] for record in records] == panel.log.records()
    assert [record['mrr'] for record in records] == [read.mrr for read in reads]

@pytest.mark.parametrize(
    'content', ['', '{"mrr": 0.5, "read": {}}\n'], ids=['empty', 'with-a-line']
)
def test_a_fresh_start_refuses_an_existing_records_file(outcome_monitor, content):
    path = outcome_monitor.records_path
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding='utf-8')

    with pytest.raises(ValueError, match='already holds monitor records'):
        outcome_monitor.start(resumed_epoch=None)

    assert path.read_text(encoding='utf-8') == content

@pytest.mark.parametrize(('resumed_epoch', 'kept'), [(0, 1), (1, 2), (3, 4)])
def test_a_resume_keeps_the_records_through_its_epoch_and_drops_later_ones(
    panel, tokenizer, outcome_monitor, resumed_epoch, kept
):
    '''Spec §6: exact resume continues the file, dropping a record past the restored epoch.'''

    path = outcome_monitor.records_path
    outcome_monitor.start(resumed_epoch=None)
    for epoch in range(4):
        outcome_monitor.append(_synthetic_read(epoch, mrr=0.1 * (epoch + 1)))
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    # A restarted process builds its monitor again
    resumed = monitor.OutcomeMonitor(panel, tokenizer, TOKEN_WINDOW, path, PURPOSE)

    resumed.start(resumed_epoch=resumed_epoch)

    # The kept lines are the lines as written, byte for byte, and nothing is left beside them
    assert path.read_text(encoding='utf-8') == ''.join(lines[:kept])
    assert sorted(entry.name for entry in path.parent.iterdir()) == [monitor.MONITOR_RECORDS]
    resumed.append(_synthetic_read(resumed_epoch + 1))
    assert _recorded_epochs(path) == list(range(resumed_epoch + 2))

def test_a_resume_needs_its_records_file(outcome_monitor):
    with pytest.raises(ValueError, match=r'from epoch 2 .* does not exist'):
        outcome_monitor.start(resumed_epoch=2)

    assert not outcome_monitor.records_path.exists()

def test_a_resume_from_a_negative_epoch_is_refused(outcome_monitor):
    outcome_monitor.start(resumed_epoch=None)
    outcome_monitor.append(_synthetic_read(0))

    with pytest.raises(ValueError, match='never negative'):
        outcome_monitor.start(resumed_epoch=-1)

    assert _recorded_epochs(outcome_monitor.records_path) == [0]

@pytest.mark.parametrize('epoch', [1, 0], ids=['the-same-epoch', 'an-earlier-epoch'])
def test_an_append_refuses_an_epoch_the_file_already_records(outcome_monitor, epoch):
    '''Each epoch of the surviving run appears once, in order (spec 4.4).'''

    path = outcome_monitor.records_path
    outcome_monitor.start(resumed_epoch=None)
    for recorded in (0, 1):
        outcome_monitor.append(_synthetic_read(recorded))
    content = path.read_text(encoding='utf-8')

    with pytest.raises(ValueError, match=f'already records epoch 1: a read of epoch {epoch}'):
        outcome_monitor.append(_synthetic_read(epoch))

    assert path.read_text(encoding='utf-8') == content

@pytest.mark.parametrize(
    ('line', 'problem'),
    [
        ('not json', 'is not JSON'),
        ('[0.5]', "exactly 'mrr' and 'read'"),
        ('{"mrr": 0.5}', "exactly 'mrr' and 'read'"),
        ('{"extra": 1, "mrr": 0.5, "read": {"detail": {"epoch": 1}}}', "exactly 'mrr' and 'read'"),
        ('{"mrr": "high", "read": {"detail": {"epoch": 1}}}', 'its mrr is not a number'),
        ('{"mrr": true, "read": {"detail": {"epoch": 1}}}', 'its mrr is not a number'),
        ('{"mrr": 0.5, "read": {"detail": {}}}', 'no non-negative integer epoch'),
        ('{"mrr": 0.5, "read": {"detail": {"epoch": "1"}}}', 'no non-negative integer epoch'),
        ('{"mrr": 0.5, "read": {"detail": {"epoch": true}}}', 'no non-negative integer epoch'),
        ('{"mrr": 0.5, "read": {"detail": {"epoch": -1}}}', 'no non-negative integer epoch'),
        ('{"mrr": 0.5, "read": "a read"}', 'no non-negative integer epoch'),
    ],
    ids=[
        'not-json',
        'not-an-object',
        'no-read',
        'another-key',
        'a-text-mrr',
        'a-boolean-mrr',
        'no-epoch',
        'a-text-epoch',
        'a-boolean-epoch',
        'a-negative-epoch',
        'a-read-that-is-not-an-object',
    ],
)
def test_reading_the_records_refuses_a_line_that_is_not_a_monitor_record(tmp_path, line, problem):
    path = tmp_path / monitor.MONITOR_RECORDS
    good = json.dumps({'mrr': 0.25, 'read': {'detail': {'epoch': 0}}})
    path.write_text(f'{good}\n{line}\n', encoding='utf-8')

    with pytest.raises(ValueError, match=f'line 2 .*{problem}'):
        monitor.read_monitor_records(path)

def test_reading_the_records_skips_blank_lines(tmp_path):
    path = tmp_path / monitor.MONITOR_RECORDS
    first = {'mrr': 0.25, 'read': {'detail': {'epoch': 0}}}
    second = {'mrr': 0.5, 'read': {'detail': {'epoch': 1}}}
    path.write_text(f'{json.dumps(first)}\n\n{json.dumps(second)}\n  \n', encoding='utf-8')

    assert monitor.read_monitor_records(path) == [first, second]

# -------------------------------------------------------------------------------------------------
# One batch size (spec 4.4's agreement)
# -------------------------------------------------------------------------------------------------

def test_every_encode_defaults_to_one_batch_size():
    '''The cache and the monitor encode in the export's and the arm encoder's batches.'''

    encoders = {
        'encode_token_rows': encode_token_rows,
        'encode_query_texts': encode_query_texts,
        'export_code_table': export_code_table,
        'ArmEncoder': ArmEncoder.__init__,
        'ArmEncoder.from_files': ArmEncoder.from_files,
        'refresh_code_cache': monitor.refresh_code_cache,
        'LiveEncoder': monitor.LiveEncoder.__init__,
    }

    defaults = {
        name: inspect.signature(function).parameters['batch_size'].default
        for name, function in encoders.items()
    }

    assert ENCODE_BATCH_SIZE == 32
    assert defaults == {name: ENCODE_BATCH_SIZE for name in encoders}

# -------------------------------------------------------------------------------------------------
# Agreement: the live read and the read of the export (spec 4.4)
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def reference_token_config(tmp_path, reference_bundle) -> TokenizationConfig:
    parameters = reference_bundle.manifest.generation_parameters
    return TokenizationConfig(
        descriptions_parquet=parameters['descriptions_parquet'],
        tokenizer_name=MINILM,
        max_length=REFERENCE_WINDOW,
        output_path=str(tmp_path / 'reference_cache' / 'token_cache.pt'),
    )

@pytest.fixture
def reference_model(
    request, tiny_backbone, reference_manifest, reference_bundle
) -> NAICSContrastiveModel:
    '''
    A d = 16 masked-mean model of the reference bundle on the tiny backbone, in eval mode, in the
    geometry arm an indirect parameter names (Req 12).
    '''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        fusion='masked_mean',
        dimension=ARM_DIMENSION,
        geometry=request.param,
        supervision_manifest_path=str(reference_manifest),
        summaries=summaries_identity(MINILM),
        supervision_bundle=reference_bundle,
    )
    return model.eval()

def _codebook_codes(bundle: ValidatedSupervisionBundle) -> Tuple[str, ...]:
    codebook = pl.read_parquet(bundle.artifact_path('codebook')).sort('code_id')
    return tuple(codebook.get_column('code').to_list())

def _code_rows(bundle: ValidatedSupervisionBundle, config: TokenizationConfig) -> List[Dict]:
    '''Every code's cached token rows, in codebook order.'''

    cache = tokenization_cache(
        config,
        description_fingerprint=bundle.manifest.description_fingerprint,
        codebook_fingerprint=bundle.manifest.codebook_fingerprint,
    )
    return [cache[code_id] for code_id in range(len(cache))]

@pytest.mark.parametrize('reference_model', GEOMETRIES, indirect=True)
def test_the_live_read_is_the_read_of_the_export_of_its_cache(
    tmp_path, tokenizer, reference_model, reference_bundle, reference_token_config
):
    '''
    Spec 4.4's agreement, on the CPU and exactly, in every geometry arm: the live encoder on the
    model and its cache, and the arm encoder on the model's checkpoint and the table exported
    from it, decode the validation split alike, under the arm's own distance (Req 12).
    '''

    rows = _code_rows(reference_bundle, reference_token_config)
    cache = monitor.refresh_code_cache(reference_model, rows, _codebook_codes(reference_bundle))
    checkpoint = tmp_path / 'reference.ckpt'
    torch.save(lightning_checkpoint(reference_model), checkpoint)
    table_path = export_code_table(
        checkpoint, reference_bundle, reference_token_config, tmp_path / 'reference_table.parquet'
    )
    arm = ArmEncoder.from_files(checkpoint, table_path, reference_bundle, reference_token_config)
    live_panel = OutcomePanel.from_bundle(reference_bundle, tmp_path / 'live_log.jsonl')
    outcome_monitor = monitor.OutcomeMonitor(
        live_panel, tokenizer, REFERENCE_WINDOW, tmp_path / monitor.MONITOR_RECORDS, PURPOSE
    )

    read = outcome_monitor.read(reference_model, cache, training_run='run-a', seed=0, epoch=0)
    live_encoder = monitor.LiveEncoder(reference_model, cache, tokenizer, REFERENCE_WINDOW)
    distance = GEOMETRY_DISTANCES[reference_model.encoder.head.geometry]
    assert live_encoder.distance == arm.distance == distance
    assert read.record['detail']['distance'] == distance
    live = live_panel.score(live_encoder, 'validation', PURPOSE, distance=distance)
    exported = read_outcome_validation(
        arm, OutcomePanel.from_bundle(reference_bundle, tmp_path / 'arm_log.jsonl'), PURPOSE
    )

    # The table is the export of the cache: the same tangents, bit for bit, in codebook order
    table = pl.read_parquet(table_path)
    assert tuple(table.get_column('code').to_list()) == cache.codes
    coordinates = table.select(pl.exclude('code', 'index', 'level')).to_numpy()
    assert np.array_equal(coordinates, cache.tangent.numpy())
    provenance = json.loads(provenance_path(table_path).read_text())
    assert read.record['detail']['table'] == provenance['matrix_fingerprint']
    assert read.record['detail']['table'] == arm.table_fingerprint
    # The same points, exactly: the validation queries and the candidates
    validation = [text for _, _, text, role in REFERENCE_INDEX_ROLE_ROWS if role == 'validation']
    candidates = list(live_panel.candidates)
    assert torch.equal(live_encoder.encode_queries(validation), arm.encode_queries(validation))
    assert torch.equal(live_encoder.encode_codes(candidates), arm.encode_codes(candidates))
    # The same decoding, exactly
    assert live.per_query.equals(exported.per_query)
    assert live.summary == exported.summary
    assert read.mrr == exported.summary['mrr']
    # Four queries over four candidates, not all at rank 1: each query is ranked against the others
    assert (exported.summary['n_queries'], exported.summary['n_candidates']) == (4, 4)
    assert exported.per_query.get_column('rank').max() > 1
