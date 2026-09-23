import pytest
import torch
from torch import nn

from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    contract_for_bundle,
    load_weights_only,
    validate_checkpoint_contract,
    validate_exact_resume,
)


@pytest.fixture
def runtime_contract() -> CheckpointContract:
    return CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
    )


@pytest.fixture
def tiny_repaired_model() -> nn.Module:
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    model.current_curriculum_flags = {}
    return model


def test_matching_new_checkpoint_can_exact_resume(tmp_path, runtime_contract):
    path = tmp_path / 'new.ckpt'
    torch.save({'stage3_supervision': runtime_contract.model_dump()}, path)

    validate_exact_resume(path, runtime_contract)


@pytest.mark.parametrize(
    'checkpoint_metadata',
    [
        None,
        {'contract_version': 'legacy'},
        {'bundle_id': 'other-bundle'},
        {'codebook_fingerprint': 'f' * 64},
        {'structural_preference_loss_version': 'other-loss'},
        {'mining_contract_version': 'other-mining'},
    ],
)
def test_legacy_or_mismatched_checkpoint_cannot_exact_resume(
    tmp_path, runtime_contract, checkpoint_metadata
):
    path = tmp_path / 'checkpoint.ckpt'
    payload = {}
    if checkpoint_metadata is not None:
        payload['stage3_supervision'] = {
            **runtime_contract.model_dump(),
            **checkpoint_metadata,
        }
    torch.save(payload, path)

    with pytest.raises(ValueError, match='exact resume'):
        validate_exact_resume(path, runtime_contract)


def test_in_memory_contract_check_names_every_mismatched_field(runtime_contract):
    saved = {**runtime_contract.model_dump(), 'bundle_id': 'bundle-b', 'supervision_mode': 'x'}

    with pytest.raises(ValueError, match='bundle_id') as excinfo:
        validate_checkpoint_contract(saved, runtime_contract)

    assert 'supervision_mode' in str(excinfo.value)


def test_contract_for_bundle_reads_manifest_identity(validated_bundle):
    contract = contract_for_bundle(validated_bundle.manifest)

    assert contract.supervision_mode == 'repaired'
    assert contract.bundle_id == 'bundle-a'
    assert contract.contract_version == 'stage3-supervision-v1'
    assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint


def test_weights_only_loads_allowlisted_encoder_and_resets_training_state(
    tmp_path, tiny_repaired_model
):
    path = tmp_path / 'legacy.ckpt'
    encoder_key = next(
        name
        for name in tiny_repaired_model.state_dict()
        if name.startswith('encoder.')
    )
    state = {
        encoder_key: torch.ones_like(tiny_repaired_model.state_dict()[encoder_key]),
        'lambdarank_loss_fn.tree_distances': torch.ones((3, 3)),
        'unexpected.weight': torch.ones(1),
    }
    torch.save(
        {
            'state_dict': state,
            'optimizer_states': [{'state': {'x': 1}}],
            'epoch': 9,
            'global_step': 123,
        },
        path,
    )

    with pytest.raises(ValueError, match='unexpected.weight'):
        load_weights_only(tiny_repaired_model, path)


def test_weights_only_reports_loaded_skipped_and_missing_without_restoring_state(
    tmp_path, tiny_repaired_model
):
    path = tmp_path / 'legacy.ckpt'
    target = tiny_repaired_model.state_dict()
    encoder_keys = sorted(name for name in target if name.startswith('encoder.'))
    loaded_key = encoder_keys[0]
    initial_flags = dict(tiny_repaired_model.current_curriculum_flags)
    torch.save(
        {
            'state_dict': {
                loaded_key: torch.full_like(target[loaded_key], 0.25),
                'loss_fn.legacy_buffer': torch.ones(1),
            },
            'optimizer_states': [{'state': {'legacy': 1}}],
            'epoch': 9,
            'global_step': 123,
        },
        path,
    )

    report = load_weights_only(tiny_repaired_model, path)

    assert report.loaded == (loaded_key,)
    assert report.skipped == ('loss_fn.legacy_buffer',)
    assert report.missing == tuple(encoder_keys[1:])
    assert report.unexpected == ()
    assert tiny_repaired_model.current_curriculum_flags == initial_flags
    assert torch.equal(
        tiny_repaired_model.state_dict()[loaded_key],
        torch.full_like(target[loaded_key], 0.25),
    )


def test_weights_only_rejects_a_checkpoint_with_no_encoder_weights(tmp_path, tiny_repaired_model):
    path = tmp_path / 'loss_only.ckpt'
    torch.save({'state_dict': {'loss_fn.legacy_buffer': torch.ones(1)}}, path)

    with pytest.raises(ValueError, match='no allowlisted encoder parameters'):
        load_weights_only(tiny_repaired_model, path)


def test_weights_only_rejects_shape_mismatched_encoder_weights(tmp_path, tiny_repaired_model):
    path = tmp_path / 'legacy.ckpt'
    torch.save({'state_dict': {'encoder.0.weight': torch.ones((5, 5))}}, path)

    with pytest.raises(ValueError, match='encoder.0.weight'):
        load_weights_only(tiny_repaired_model, path)
