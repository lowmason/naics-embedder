'''
The shared encoder (Req 14; spec 4.1), on a one-layer BERT so these tests download nothing.

One backbone serves codes and queries; an absent channel never reaches the output; exactly one
affine map sits between fusion and the point; a field's texts go to the backbone in bounded chunks
that leave every output and every gradient unchanged up to float noise; a new encoder is in train
mode, so its dropout and gradient checkpointing engage, and a checkpointed backward replays the
forward's dropout masks, on MPS too; a step reaches every adapter and the projection; and under
autocast only the backbone runs in reduced precision.
'''

import contextlib
from collections import Counter
from typing import List
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer
from transformers import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer

from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.fields import CHANNELS, QUERY
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.hyperbolic import HyperbolicHead, check_lorentz_manifold_validity
from naics_embedder.text_model.shared_encoder import (
    DIMENSIONS,
    MAX_TEXTS_PER_CALL,
    SharedEncoder,
    _replay_mps_rng,
)
from tests.fixtures.shared_encoder import TINY_HIDDEN, tiny_bert

pytestmark = pytest.mark.unit

WIDTH = 8

def _text(ids):
    '''A present text: these token ids, right-padded to WIDTH.'''

    input_ids = torch.zeros(WIDTH, dtype=torch.long)
    input_ids[:len(ids)] = torch.tensor(ids)
    attention_mask = torch.zeros(WIDTH, dtype=torch.long)
    attention_mask[:len(ids)] = 1
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'present': True}

def _absent():
    '''An absent channel as the cache stores it: ``[CLS] [SEP]`` with ``present`` False.'''

    return {**_text([101, 102]), 'present': False}

def _code(**texts):
    '''A code whose named channels hold these token ids; its other channels are absent.'''

    return {
        channel: _text(texts[channel]) if channel in texts else _absent()
        for channel in CHANNELS
    }

# Code 0 has a title and examples; code 1 has a description only
CODES = [
    _code(title=[101, 2001, 2002, 102], examples=[101, 2003, 2004, 2005, 102]),
    _code(description=[101, 2006, 2007, 102]),
]

def _tokens(row, field, length):
    '''Token ids that belong to this (row, field) alone: ``[CLS]``, length - 2 ids, ``[SEP]``.'''

    start = 3000 + 100 * row + 10 * CHANNELS.index(field)
    return [101, *range(start, start + length - 2), 102]

def _mixed_code(row, counts):
    '''A code whose present channels have these token counts, with ids that no other code shares.'''

    return _code(**{field: _tokens(row, field, length) for field, length in counts.items()})

# Per row, the token count of each present channel; the last code has none. The title is present
# in rows 0, 2, 3 and 4, so a bound of 2 splits it into the chunks (0, 2) and (3, 4)
MIXED_PRESENCE = [
    dict(title=4, examples=6),
    dict(description=5),
    dict(title=7, description=3, excluded=4),
    dict(title=3, examples=5),
    dict(title=5, excluded=6, examples=4),
    dict(),
]
# Every text has ids of its own, so a pooled vector in another row's slot changes an output
MIXED_CODES = [_mixed_code(row, counts) for row, counts in enumerate(MIXED_PRESENCE)]

@pytest.fixture
def make_encoder(tiny_backbone):
    '''Build a ``SharedEncoder`` on the tiny backbone, with these settings overridden.'''

    def make(**overrides):
        settings = {
            'base_model_name': 'tiny-bert',
            'lora_r': 2,
            'lora_alpha': 4,
            'lora_dropout': 0.0,
            'fusion': 'masked_mean',
            'dimension': 8,
            'num_experts': 2,
            'top_k': 1,
            'moe_hidden_dim': 4,
            'use_gradient_checkpointing': False,
        }
        return SharedEncoder(**{**settings, **overrides})

    return make

def _record_backbone_calls(encoder, monkeypatch) -> List[torch.Tensor]:
    '''The ``input_ids`` of every backbone call the encoder makes, in order.'''

    calls = []
    forward = encoder.backbone.forward

    def spy(**inputs):
        calls.append(inputs['input_ids'].clone())
        return forward(**inputs)

    monkeypatch.setattr(encoder.backbone, 'forward', spy)
    return calls

def _encode_with_slots(encoder, batch):
    '''The encoder's output for this batch, and the ``(pooled, present)`` it gave its fusion.'''

    seen = []
    handle = encoder.fusion.register_forward_hook(lambda module, args, output: seen.append(args))
    try:
        output = encoder(batch)
    finally:
        handle.remove()
    pooled, present = seen[0]
    return output, pooled, present

def _count_layer_runs(encoder):
    '''
    Per backbone layer, by name: how often it ran, and how often it ran through its checkpointing
    function, which is the checkpointed branch of ``GradientCheckpointingLayer``.
    '''

    ran, checkpointed = Counter(), Counter()

    def watch(name, layer):
        through = layer._gradient_checkpointing_func

        def count_and_call(function, *args, **kwargs):
            checkpointed[name] += 1
            return through(function, *args, **kwargs)

        layer._gradient_checkpointing_func = count_and_call
        layer.register_forward_hook(lambda module, args, output: ran.update([name]))

    for name, module in encoder.backbone.named_modules():
        if isinstance(module, GradientCheckpointingLayer):
            watch(name, module)
    return ran, checkpointed

def _switch_off_dropout(encoder):
    '''
    Turn off every dropout in the encoder, so a chunk bound cannot change the random draws.

    Setting each ``nn.Dropout``'s ``p`` is not enough: under SDPA, BERT's attention dropout reads a
    ``dropout_prob`` attribute and draws from the RNG inside the attention kernel.
    '''

    for module in encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if hasattr(module, 'dropout_prob'):
            module.dropout_prob = 0.0

def _weighted_loss(output):
    '''A fixed random-weighted sum of ``tangent``, and of ``gate_probs`` under ``moe``.'''

    loss = 0
    for seed, key in enumerate(('tangent', 'gate_probs')):
        if key in output:
            weights = torch.randn(output[key].shape, generator=torch.Generator().manual_seed(seed))
            loss = loss + (output[key] * weights.to(output[key].device)).sum()
    return loss

DEVICES = [
    'cpu',
    pytest.param(
        'mps',
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason='needs an MPS device'
        ),
    ),
]

def _to_device(batch, device):
    '''A ``stack_text_inputs`` batch with every tensor on this device.'''

    return {
        field: {
            name: tensor.to(device)
            for name, tensor in inputs.items()
        }
        for field, inputs in batch.items()
    }

def _seeded_step(encoder, batch, device):
    '''
    One forward and backward from a fixed seed, with the encoder on this device. It returns the
    tangent, every gradient, and the random state the step leaves behind, all as CPU tensors.
    '''

    encoder.to(device).train()
    torch.manual_seed(1234)
    output = encoder(batch)
    _weighted_loss(output).backward()
    gradients = {
        name: parameter.grad.cpu()
        for name, parameter in encoder.named_parameters() if parameter.grad is not None
    }
    # The state that dropout on this device draws from
    state = torch.mps.get_rng_state() if device == 'mps' else torch.get_rng_state()
    return output['tangent'].detach().cpu(), gradients, state

# -------------------------------------------------------------------------------------------------
# Structure
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fusion', FUSIONS)
def test_one_backbone_then_fusion_one_affine_map_and_a_parameter_free_head(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion, dimension=16)

    # From the fused vector to the point: the projection, then the head
    assert [name for name, _ in encoder.named_children()] == [
        'backbone', 'fusion', 'projection', 'head'
    ]
    assert sum(isinstance(module, PreTrainedModel) for module in encoder.modules()) == 1
    assert isinstance(encoder.projection, nn.Linear)
    assert (encoder.projection.in_features, encoder.projection.out_features) == (TINY_HIDDEN, 16)
    assert isinstance(encoder.head, HyperbolicHead)
    assert list(encoder.head.parameters()) == []
    assert encoder.fusion_name == fusion

def test_one_lora_adapter_wraps_every_linear_layer_of_the_backbone(make_encoder):
    encoder = make_encoder()
    linear_layers = [module for module in tiny_bert().modules() if isinstance(module, nn.Linear)]
    adapted = [module for module in encoder.backbone.modules() if isinstance(module, LoraLayer)]

    assert list(encoder.backbone.peft_config) == ['default']
    # The pooler's dense layer too, which mean pooling never reads (P8)
    assert len(adapted) == len(linear_layers) == 7

def test_the_encoder_takes_no_curvature(make_encoder):
    # Spec 4.2: curvature is fixed at 1, with no parameter in the text stage
    with pytest.raises(TypeError):
        make_encoder(curvature=1.0)

def test_the_radius_bound_reaches_the_head(make_encoder):
    assert make_encoder(radius_bound=5.0).head.radius_bound == 5.0
    assert make_encoder().head.radius_bound == 8.0

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_the_geometry_picks_the_head_and_the_width_of_the_point(make_encoder, geometry):
    '''Req 12: every arm shares the encoder; the head and the point's width follow the geometry.'''

    encoder = make_encoder(geometry=geometry).eval()

    with torch.no_grad():
        output = encoder(stack_text_inputs(CODES))

    assert encoder.head.geometry == geometry
    assert [name for name, _ in encoder.named_children()][-1] == 'head'
    assert output['tangent'].shape == (2, 8)
    assert output['embedding'].shape == (2, 9 if geometry == 'hyperbolic' else 8)

def test_an_unknown_geometry_is_refused_before_the_backbone_loads(make_encoder, monkeypatch):

    def never(_name):
        raise AssertionError('the backbone loaded before the geometry was refused')

    monkeypatch.setattr('naics_embedder.text_model.shared_encoder.load_base_model', never)

    with pytest.raises(ValueError, match="unknown geometry 'poincare'"):
        make_encoder(geometry='poincare')

def test_an_unknown_fusion_or_dimension_is_refused(make_encoder):
    assert DIMENSIONS == (8, 16, 32)
    with pytest.raises(ValueError, match='unknown fusion'):
        make_encoder(fusion='concatenate')
    with pytest.raises(ValueError, match='unknown dimension'):
        make_encoder(dimension=12)

@pytest.mark.parametrize('bound', [0, -1])
def test_a_chunk_bound_below_one_is_refused(make_encoder, bound):
    with pytest.raises(ValueError, match='max_texts_per_call must be at least 1'):
        make_encoder(max_texts_per_call=bound)

# -------------------------------------------------------------------------------------------------
# One encoder for codes and queries
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fusion', FUSIONS)
def test_a_one_field_batch_encodes_as_a_code_with_only_that_channel(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion).eval()
    text = [101, 2001, 2002, 2003, 102]
    one_code = stack_text_inputs([_code(excluded=text)])
    one_field = stack_text_inputs([{'excluded': _text(text)}], fields=('excluded', ))
    query = stack_text_inputs([{QUERY: _text(text)}], fields=(QUERY, ))

    with torch.no_grad():
        expected, *others = [encoder(batch) for batch in (one_code, one_field, query)]

    for output in others:
        assert output.keys() == expected.keys()
        for key in output:
            assert torch.equal(output[key], expected[key])

# -------------------------------------------------------------------------------------------------
# Masking
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fusion', FUSIONS)
@pytest.mark.parametrize('max_texts_per_call', (1, MAX_TEXTS_PER_CALL))
def test_perturbing_an_absent_channel_leaves_the_output_bit_identical(
    make_encoder, fusion, max_texts_per_call
):
    encoder = make_encoder(fusion=fusion, max_texts_per_call=max_texts_per_call).eval()
    batch = stack_text_inputs(CODES)
    perturbed = stack_text_inputs(CODES)
    for channel in CHANNELS:
        absent = ~perturbed[channel]['present']
        perturbed[channel]['input_ids'][absent] = 2999
        perturbed[channel]['attention_mask'][absent] = 1

    with torch.no_grad():
        clean, noisy = encoder(batch), encoder(perturbed)

    assert clean.keys() == noisy.keys()
    for key in clean:
        assert torch.equal(clean[key], noisy[key])

def test_absent_texts_and_padding_never_enter_the_backbone(make_encoder, monkeypatch):
    encoder = make_encoder().eval()
    calls = _record_backbone_calls(encoder, monkeypatch)

    with torch.no_grad():
        encoder(stack_text_inputs(CODES))

    # One call per field with a present text, in field order (no code has an excluded text), each
    # trimmed to its own longest text
    assert [call.tolist() for call in calls] == [
        [[101, 2001, 2002, 102]],
        [[101, 2006, 2007, 102]],
        [[101, 2003, 2004, 2005, 102]],
    ]

@pytest.mark.parametrize('fusion', FUSIONS)
def test_a_code_with_no_present_channel_encodes_finitely(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion)

    output = encoder(stack_text_inputs([CODES[0], _code()]))
    output['embedding'].sum().backward()

    assert torch.isfinite(output['embedding']).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )

def test_a_batch_with_no_present_text_never_calls_the_backbone(make_encoder, monkeypatch):
    encoder = make_encoder().eval()
    monkeypatch.setattr(encoder.backbone, 'forward', Mock(side_effect=AssertionError('backbone')))

    with torch.no_grad():
        output = encoder(stack_text_inputs([_code(), _code()]))

    assert output['embedding'].shape == (2, 9)
    assert torch.isfinite(output['embedding']).all()

# -------------------------------------------------------------------------------------------------
# Chunked backbone calls
# -------------------------------------------------------------------------------------------------

def test_each_fields_texts_go_in_chunks_trimmed_to_their_own_longest_text(
    make_encoder, monkeypatch
):
    encoder = make_encoder(max_texts_per_call=2).eval()
    calls = _record_backbone_calls(encoder, monkeypatch)
    codes = [
        _code(title=[101, 2001, 102]),
        _code(title=[101, 2002, 2003, 2004, 102]),
        _code(title=[101, 2005, 2006, 102]),
    ]

    with torch.no_grad():
        encoder(stack_text_inputs(codes))

    # Rows 0 and 1 share a chunk, trimmed to the longer (five tokens); row 2 is trimmed to its own
    assert [call.tolist() for call in calls] == [
        [[101, 2001, 102, 0, 0], [101, 2002, 2003, 2004, 102]],
        [[101, 2005, 2006, 102]],
    ]

@pytest.mark.parametrize('fusion', FUSIONS)
def test_the_chunk_bound_leaves_the_outputs_unchanged(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion).eval()
    batch = stack_text_inputs(MIXED_CODES)

    outputs = []
    for bound in (1, 2, MAX_TEXTS_PER_CALL):
        encoder.max_texts_per_call = bound
        with torch.no_grad():
            outputs.append(encoder(batch))

    reference, *others = outputs
    for output in others:
        assert output.keys() == reference.keys()
        torch.testing.assert_close(output['embedding'], reference['embedding'])
        torch.testing.assert_close(output['tangent'], reference['tangent'])
        if 'gate_probs' in output:
            torch.testing.assert_close(output['gate_probs'], reference['gate_probs'])
            assert torch.equal(output['top_k_indices'], reference['top_k_indices'])

@pytest.mark.parametrize('fusion', FUSIONS)
def test_each_row_of_a_batch_encodes_as_that_code_alone(make_encoder, fusion):
    # A bound of 2 puts rows of the same field in one chunk, so a vector can land in another row
    encoder = make_encoder(fusion=fusion, max_texts_per_call=2).eval()

    with torch.no_grad():
        together, pooled, present = _encode_with_slots(encoder, stack_text_inputs(MIXED_CODES))
        for row, code in enumerate(MIXED_CODES):
            alone = encoder(stack_text_inputs([code]))
            assert alone.keys() == together.keys()
            for key in alone:
                torch.testing.assert_close(together[key][row], alone[key][0])

            # Every slot holds its own text, pooled on its own, or zeros if the text is absent. A
            # misplacement that is the same for every batch composition shows only here
            for column, field in enumerate(CHANNELS):
                slot = pooled[row, column]
                if present[row, column]:
                    one_field = stack_text_inputs([{field: code[field]}], fields=(field, ))
                    _, text_alone, _ = _encode_with_slots(encoder, one_field)
                    torch.testing.assert_close(slot, text_alone[0, 0])
                else:
                    assert torch.equal(slot, torch.zeros_like(slot))

@pytest.mark.parametrize('fusion', FUSIONS)
def test_the_chunk_bound_leaves_the_gradients_unchanged(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion, use_gradient_checkpointing=True).train()
    _switch_off_dropout(encoder)
    # PEFT starts every lora_B at zero, which leaves every lora_A gradient exactly zero (P9)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(0.0, 0.1, generator=generator)
    batch = stack_text_inputs(MIXED_CODES)

    gradients = {}
    for bound in (1, 2, MAX_TEXTS_PER_CALL):
        encoder.max_texts_per_call = bound
        encoder.zero_grad()
        _weighted_loss(encoder(batch)).backward()
        gradients[bound] = {
            name: parameter.grad.clone()
            for name, parameter in encoder.named_parameters() if parameter.grad is not None
        }

    reference = gradients[1]
    lora_a = [gradient for name, gradient in reference.items() if 'lora_A' in name]
    assert lora_a and all(gradient.abs().sum() > 0 for gradient in lora_a)
    for bound in (2, MAX_TEXTS_PER_CALL):
        assert gradients[bound].keys() == reference.keys()
        for name, gradient in gradients[bound].items():
            torch.testing.assert_close(
                gradient, reference[name], msg=lambda text: f'{name}: {text}'
            )

# -------------------------------------------------------------------------------------------------
# Outputs and refusals
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fusion', FUSIONS)
def test_only_the_moe_fusion_emits_gates(make_encoder, fusion):
    with torch.no_grad():
        output = make_encoder(fusion=fusion).eval()(stack_text_inputs(CODES))

    gates = {'gate_probs', 'top_k_indices'} if fusion == 'moe' else set()
    assert set(output) == {'embedding', 'tangent', 'radius', 'direction'} | gates

@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_the_point_is_the_exp_map_of_the_bounded_tangent(make_encoder, dimension):
    with torch.no_grad():
        output = make_encoder(dimension=dimension).eval()(stack_text_inputs(CODES))

    assert output['tangent'].shape == (2, dimension)
    assert output['embedding'].shape == (2, dimension + 1)
    assert output['radius'].shape == (2, )
    assert output['direction'].shape == (2, dimension)
    # One radial coordinate: r is the tangent's norm, below the bound, and the tangent is r · û
    torch.testing.assert_close(output['tangent'].norm(dim=1), output['radius'])
    assert (output['radius'] <= 8.0).all()
    torch.testing.assert_close(
        output['tangent'], output['radius'].unsqueeze(1) * output['direction']
    )
    expected = torch.cat(
        [
            torch.cosh(output['radius']).unsqueeze(1),
            torch.sinh(output['radius']).unsqueeze(1) * output['direction'],
        ],
        dim=1,
    )
    torch.testing.assert_close(output['embedding'], expected)
    is_valid, _, _ = check_lorentz_manifold_validity(output['embedding'], curvature=1.0)
    assert is_valid

@pytest.mark.parametrize('fusion', FUSIONS)
def test_under_bf16_autocast_only_the_backbone_runs_in_reduced_precision(make_encoder, fusion):
    '''Spec 4.2 and §6 "Precision": from fusion on, through the projection and the head, the
    encoder runs in float32 with autocast off.'''

    encoder = make_encoder(fusion=fusion).eval()
    dtypes = {}

    def record(name):
        # A forward hook that returns a value replaces the output, so this one returns None
        def hook(_module, _inputs, output):
            dtypes.setdefault(name, output.dtype)

        return hook

    backbone_linear = next(
        module for module in encoder.backbone.modules() if isinstance(module, nn.Linear)
    )
    backbone_linear.register_forward_hook(record('backbone'))
    encoder.projection.register_forward_hook(record('projection'))

    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        output = encoder(stack_text_inputs(CODES))

    assert dtypes == {'backbone': torch.bfloat16, 'projection': torch.float32}
    floating = {name for name, value in output.items() if value.is_floating_point()}
    assert floating >= {'embedding', 'tangent', 'radius', 'direction'}
    assert {output[name].dtype for name in floating} == {torch.float32}

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_under_bf16_autocast_a_flat_head_gives_float32_points(make_encoder, geometry):
    encoder = make_encoder(geometry=geometry).eval()

    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        output = encoder(stack_text_inputs(CODES))

    names = ('embedding', 'tangent', 'radius', 'direction')
    assert {output[name].dtype for name in names} == {torch.float32}

def test_a_malformed_batch_is_refused(make_encoder):
    encoder = make_encoder()
    batch = stack_text_inputs(CODES)
    del batch['title']['present']

    with pytest.raises(ValueError, match='no present flag'):
        encoder(batch)
    with pytest.raises(ValueError, match='unknown field'):
        encoder({'summary': stack_text_inputs(CODES)['title']})
    with pytest.raises(ValueError, match='at least one field'):
        encoder({})

# -------------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------------

def test_a_new_encoder_is_in_train_mode_throughout(make_encoder):
    # The pretrained backbone arrives in eval mode (the fixture mirrors from_pretrained), and
    # Lightning never calls .train(), so the encoder has to build every module in train mode
    encoder = make_encoder()

    assert {module.training for module in encoder.modules()} == {True}
    encoder.eval()
    assert {module.training for module in encoder.modules()} == {False}
    encoder.train()
    assert {module.training for module in encoder.modules()} == {True}

def test_train_mode_runs_the_backbone_layers_checkpointed(make_encoder):
    encoder = make_encoder(use_gradient_checkpointing=True)
    ran, checkpointed = _count_layer_runs(encoder)
    batch = stack_text_inputs(CODES)

    # As built, with no .train() call: dropout and checkpointing key on each layer's own mode
    encoder(batch)
    assert ran and checkpointed == ran

    ran.clear()
    checkpointed.clear()
    encoder.eval()
    with torch.no_grad():
        encoder(batch)
    assert ran and not checkpointed

@pytest.mark.parametrize('device', DEVICES)
def test_checkpointing_replays_the_dropout_masks(make_encoder, device):
    # Dropout is on in train mode, and a checkpointed layer's backward reruns its forward. The rerun
    # has to draw the forward's masks, or its gradients belong to another forward. torch restores
    # the CPU and CUDA random states for it, but not MPS's
    settings = dict(lora_dropout=0.1, max_texts_per_call=2)
    checkpointed = make_encoder(use_gradient_checkpointing=True, **settings)
    plain = make_encoder(use_gradient_checkpointing=False, **settings)
    # PEFT starts every lora_B at zero, which leaves every lora_A gradient exactly zero (P9)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, parameter in checkpointed.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(0.0, 0.1, generator=generator)
    plain.load_state_dict(checkpointed.state_dict())
    batch = _to_device(stack_text_inputs(MIXED_CODES), device)

    plain_tangent, plain_gradients, plain_state = _seeded_step(plain, batch, device)
    tangent, gradients, state = _seeded_step(checkpointed, batch, device)

    # Dropout is live, so equal gradients come from replayed masks: another seed, another forward
    with torch.no_grad():
        torch.manual_seed(4321)
        other_tangent = plain(batch)['tangent'].cpu()
    assert not torch.allclose(other_tangent, plain_tangent)
    torch.testing.assert_close(tangent, plain_tangent)
    assert gradients.keys() == plain_gradients.keys()
    lora_a = [gradient for name, gradient in plain_gradients.items() if 'lora_A' in name]
    assert lora_a and all(gradient.abs().sum() > 0 for gradient in lora_a)
    for name, gradient in gradients.items():
        torch.testing.assert_close(
            gradient, plain_gradients[name], msg=lambda text: f'{name}: {text}'
        )
    # The recompute also leaves the random stream where an uncheckpointed step leaves it
    assert torch.equal(state, plain_state)

def test_off_mps_the_replay_contexts_do_nothing(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    read_state, set_state = Mock(), Mock()
    monkeypatch.setattr(torch.mps, 'get_rng_state', read_state)
    monkeypatch.setattr(torch.mps, 'set_rng_state', set_state)

    forward_context, recompute_context = _replay_mps_rng()
    with forward_context, recompute_context:
        pass

    assert isinstance(forward_context, contextlib.nullcontext)
    assert isinstance(recompute_context, contextlib.nullcontext)
    read_state.assert_not_called()
    set_state.assert_not_called()

@pytest.mark.parametrize('checkpointing', [False, True])
@pytest.mark.parametrize('max_texts_per_call', (1, MAX_TEXTS_PER_CALL))
def test_a_step_reaches_every_adapter_and_the_projection(
    make_encoder, checkpointing, max_texts_per_call
):
    encoder = make_encoder(
        use_gradient_checkpointing=checkpointing, max_texts_per_call=max_texts_per_call
    ).train()

    encoder(stack_text_inputs(CODES))['embedding'].sum().backward()

    assert encoder.projection.weight.grad.abs().sum() > 0
    adapters = {
        name: parameter
        for name, parameter in encoder.backbone.named_parameters() if 'lora_B' in name
    }
    assert adapters
    for name, parameter in adapters.items():
        if '.pooler.' in name:
            # Mean pooling never reads the pooler, so its adapter gets no gradient (P8)
            assert parameter.grad is None, name
        else:
            # PEFT starts lora_B at zero, so lora_A's first gradient is exactly zero (P9)
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
