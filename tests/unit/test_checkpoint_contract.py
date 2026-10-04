import pytest
import torch
from torch import nn

from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.checkpoints import (
    D2_REFUSAL,
    LEGACY_ENCODER,
    CheckpointContract,
    EncoderArchitecture,
    containment_contract,
    contract_for_bundle,
    load_weights_only,
    saved_encoder,
    shared_encoder_architecture,
    validate_checkpoint_contract,
    validate_exact_resume,
    validate_supervision_contract,
)

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
SHARED = shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone=MINILM)

@pytest.fixture
def summaries() -> str:
    '''MiniLM's summaries: under the test seam, the dummy pin's sha256.'''

    return summaries_identity(MINILM)

@pytest.fixture
def runtime_contract(summaries) -> CheckpointContract:
    return CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
        encoder=SHARED,
        summaries=summaries,
    )

@pytest.fixture
def tiny_repaired_model() -> nn.Module:
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    model.current_curriculum_flags = {}
    return model

def _save(path, contract=None, **payload):
    '''Save a checkpoint holding ``payload``, under ``contract`` when one is given.'''

    if contract is not None:
        payload['stage3_supervision'] = contract.model_dump()
    torch.save(payload, path)
    return path

# -------------------------------------------------------------------------------------------------
# Exact resume
# -------------------------------------------------------------------------------------------------

def test_matching_new_checkpoint_can_exact_resume(tmp_path, runtime_contract):
    validate_exact_resume(_save(tmp_path / 'new.ckpt', runtime_contract), runtime_contract)

@pytest.mark.parametrize(
    'checkpoint_metadata',
    [
        None,
        {
            'contract_version': 'legacy'
        },
        {
            'bundle_id': 'other-bundle'
        },
        {
            'codebook_fingerprint': 'f' * 64
        },
        {
            'structural_preference_loss_version': 'other-loss'
        },
        {
            'mining_contract_version': 'other-mining'
        },
        # Trained before Stage 6b, on truncated text, or under other summaries
        {
            'summaries': None
        },
        {
            'summaries': 'f' * 64
        },
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
    # A supervision mismatch is not the architecture refusal
    assert D2_REFUSAL not in str(excinfo.value)

def test_a_checkpoint_trained_under_the_exclusion_quota_cannot_exact_resume(
    tmp_path, runtime_contract
):
    # negative-selection-v1 reserved a slot for an explicit exclusion; v2 never selects one
    path = tmp_path / 'quota.ckpt'
    saved = {**runtime_contract.model_dump(), 'mining_contract_version': 'negative-selection-v1'}
    torch.save({'stage3_supervision': saved}, path)

    assert runtime_contract.mining_contract_version == 'negative-selection-v2'
    with pytest.raises(ValueError, match='exact resume'):
        validate_exact_resume(path, runtime_contract)

def test_contract_for_bundle_reads_manifest_identity(validated_bundle, summaries):
    contract = contract_for_bundle(validated_bundle.manifest, encoder=SHARED, summaries=summaries)

    assert contract.supervision_mode == 'repaired'
    assert contract.bundle_id == 'bundle-a'
    assert contract.contract_version == 'stage3-supervision-v2'
    assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
    assert contract.encoder == SHARED
    assert contract.summaries == summaries

# -------------------------------------------------------------------------------------------------
# The summaries (Stage 6b spec, 4.8)
# -------------------------------------------------------------------------------------------------

def test_a_contract_saved_before_stage_6b_reads_as_null_summaries(runtime_contract):
    saved = runtime_contract.model_dump()
    del saved['summaries']

    assert CheckpointContract.model_validate(saved).summaries is None

def test_a_containment_checkpoint_under_other_summaries_cannot_exact_resume(tmp_path, summaries):
    runtime = containment_contract(encoder=SHARED, summaries=summaries)
    path = _save(
        tmp_path / 'containment.ckpt', containment_contract(encoder=SHARED, summaries=None)
    )

    with pytest.raises(ValueError, match='exact resume') as refusal:
        validate_exact_resume(path, runtime)

    assert f"'summaries': (None, '{summaries}')" in str(refusal.value)

def test_the_supervision_check_refuses_other_summaries_naming_the_field(
    validated_bundle, summaries
):
    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED, summaries=None)

    with pytest.raises(ValueError, match="supervision contract mismatch .*'summaries'"):
        validate_supervision_contract(saved.model_dump(), manifest, summaries=summaries)

def test_a_caller_that_omits_the_summaries_is_a_type_error(validated_bundle, summaries):
    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED, summaries=summaries).model_dump()

    with pytest.raises(TypeError):
        validate_supervision_contract(saved, manifest)
    with pytest.raises(TypeError):
        contract_for_bundle(manifest, encoder=SHARED)
    with pytest.raises(TypeError):
        containment_contract(encoder=SHARED)

# -------------------------------------------------------------------------------------------------
# The encoder record (spec 4.4)
# -------------------------------------------------------------------------------------------------

def test_an_absent_encoder_record_reads_as_the_legacy_four_copy_layout(runtime_contract):
    saved = runtime_contract.model_dump()
    del saved['encoder']

    assert CheckpointContract.model_validate(saved).encoder == LEGACY_ENCODER
    assert LEGACY_ENCODER == EncoderArchitecture(layout='four-copy')
    assert saved_encoder(saved) == LEGACY_ENCODER
    assert saved_encoder(None) == LEGACY_ENCODER

@pytest.mark.parametrize(
    'record',
    [
        {
            'layout': 'shared',
            'fusion': 'masked_mean',
            'dimension': 16
        },
        {
            'layout': 'four-copy',
            'dimension': 16
        },
        {
            'layout': 'concatenated'
        },
        {
            'layout': 'shared',
            'fusion': 'masked_mean',
            'dimension': 16,
            'backbone': MINILM,
            'x': 1
        },
    ],
)
def test_a_malformed_encoder_record_is_refused(record):
    with pytest.raises(ValueError):
        EncoderArchitecture(**record)

def test_the_encoder_record_survives_a_save_round_trip(tmp_path, runtime_contract):
    path = _save(tmp_path / 'shared.ckpt', runtime_contract)

    saved = torch.load(path, weights_only=False)['stage3_supervision']

    assert saved['encoder'] == {
        'layout': 'shared',
        'fusion': 'masked_mean',
        'dimension': 16,
        'backbone': MINILM,
    }
    assert CheckpointContract.model_validate(saved) == runtime_contract

@pytest.mark.parametrize(
    'encoder',
    [
        LEGACY_ENCODER,
        shared_encoder_architecture(fusion='masked_mean', dimension=8, backbone=MINILM),
        shared_encoder_architecture(fusion='moe', dimension=16, backbone=MINILM),
        shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone='other/model'),
    ],
)
def test_another_encoder_architecture_cannot_exact_resume(tmp_path, runtime_contract, encoder):
    path = _save(tmp_path / 'other.ckpt', runtime_contract.model_copy(update={'encoder': encoder}))

    with pytest.raises(ValueError, match='exact resume') as excinfo:
        validate_exact_resume(path, runtime_contract)

    assert 'encoder' in str(excinfo.value)
    assert D2_REFUSAL in str(excinfo.value)

def test_a_contract_saved_before_stage_6_meets_the_d2_refusal(tmp_path, runtime_contract):
    saved = runtime_contract.model_dump()
    del saved['encoder']
    path = tmp_path / 'four-copy.ckpt'
    torch.save({'stage3_supervision': saved}, path)

    with pytest.raises(ValueError, match='four-copy') as excinfo:
        validate_exact_resume(path, runtime_contract)

    assert D2_REFUSAL in str(excinfo.value)

def test_a_checkpoint_without_a_contract_cites_d2(tmp_path, runtime_contract):
    with pytest.raises(ValueError, match='cannot exact resume') as excinfo:
        validate_exact_resume(_save(tmp_path / 'legacy.ckpt'), runtime_contract)

    assert D2_REFUSAL in str(excinfo.value)
    assert 'weights_only' not in str(excinfo.value)

# -------------------------------------------------------------------------------------------------
# Export and reads compare the supervision fields only
# -------------------------------------------------------------------------------------------------

def test_the_supervision_check_takes_the_encoder_record_from_the_checkpoint(
    validated_bundle, summaries
):
    manifest = validated_bundle.manifest
    other = shared_encoder_architecture(fusion='attention', dimension=8, backbone=MINILM)
    saved = contract_for_bundle(manifest, encoder=other, summaries=summaries)

    assert validate_supervision_contract(saved.model_dump(), manifest, summaries=summaries) == saved

@pytest.mark.parametrize(
    'update',
    [
        {
            'bundle_id': 'other-bundle'
        },
        {
            'codebook_fingerprint': 'f' * 64
        },
        {
            'supervision_mode': 'legacy_containment'
        },
    ],
)
def test_the_supervision_check_refuses_another_bundle(validated_bundle, summaries, update):
    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED, summaries=summaries)

    with pytest.raises(ValueError, match='supervision contract mismatch'):
        validate_supervision_contract(
            saved.model_copy(update=update).model_dump(), manifest, summaries=summaries
        )

def test_the_supervision_check_refuses_a_checkpoint_without_a_contract(validated_bundle):
    with pytest.raises(ValueError, match='no Stage-3 contract') as excinfo:
        validate_supervision_contract(None, validated_bundle.manifest, summaries=None)

    assert D2_REFUSAL in str(excinfo.value)

# -------------------------------------------------------------------------------------------------
# Weights-only migration
# -------------------------------------------------------------------------------------------------

def test_weights_only_loads_allowlisted_encoder_and_resets_training_state(
    tmp_path, runtime_contract, tiny_repaired_model
):
    encoder_key = next(
        name for name in tiny_repaired_model.state_dict() if name.startswith('encoder.')
    )
    path = _save(
        tmp_path / 'checkpoint.ckpt',
        runtime_contract,
        state_dict={
            encoder_key: torch.ones_like(tiny_repaired_model.state_dict()[encoder_key]),
            'lambdarank_loss_fn.tree_distances': torch.ones((3, 3)),
            'unexpected.weight': torch.ones(1),
        },
        optimizer_states=[{
            'state': {
                'x': 1
            }
        }],
        epoch=9,
        global_step=123,
    )

    with pytest.raises(ValueError, match='unexpected.weight'):
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

def test_weights_only_reports_loaded_skipped_and_missing_without_restoring_state(
    tmp_path, runtime_contract, tiny_repaired_model
):
    target = tiny_repaired_model.state_dict()
    encoder_keys = sorted(name for name in target if name.startswith('encoder.'))
    loaded_key = encoder_keys[0]
    initial_flags = dict(tiny_repaired_model.current_curriculum_flags)
    # The same architecture under another bundle, which weights-only still serves (D2)
    path = _save(
        tmp_path / 'other-bundle.ckpt',
        runtime_contract.model_copy(update={'bundle_id': 'bundle-b'}),
        state_dict={
            loaded_key: torch.full_like(target[loaded_key], 0.25),
            'loss_fn.legacy_buffer': torch.ones(1),
        },
        optimizer_states=[{
            'state': {
                'legacy': 1
            }
        }],
        epoch=9,
        global_step=123,
    )

    report = load_weights_only(tiny_repaired_model, path, encoder=SHARED)

    assert report.loaded == (loaded_key, )
    assert report.skipped == ('loss_fn.legacy_buffer', )
    assert report.missing == tuple(encoder_keys[1:])
    assert report.unexpected == ()
    assert tiny_repaired_model.current_curriculum_flags == initial_flags
    assert torch.equal(
        tiny_repaired_model.state_dict()[loaded_key],
        torch.full_like(target[loaded_key], 0.25),
    )

def test_weights_only_rejects_a_checkpoint_with_no_encoder_weights(
    tmp_path, runtime_contract, tiny_repaired_model
):
    path = _save(
        tmp_path / 'loss_only.ckpt',
        runtime_contract,
        state_dict={'loss_fn.legacy_buffer': torch.ones(1)},
    )

    with pytest.raises(ValueError, match='no allowlisted encoder parameters'):
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

def test_weights_only_loads_a_checkpoint_trained_under_other_summaries(
    tmp_path, runtime_contract, tiny_repaired_model
):
    # Weights-only compares the encoder record alone, so a pre-6b checkpoint can seed a run
    key = sorted(name for name in tiny_repaired_model.state_dict()
                 if name.startswith('encoder.'))[0]
    path = _save(
        tmp_path / 'truncated.ckpt',
        runtime_contract.model_copy(update={'summaries': None}),
        state_dict={key: torch.zeros_like(tiny_repaired_model.state_dict()[key])},
    )

    report = load_weights_only(tiny_repaired_model, path, encoder=SHARED)

    assert report.loaded == (key, )

def test_weights_only_rejects_shape_mismatched_encoder_weights(
    tmp_path, runtime_contract, tiny_repaired_model
):
    path = _save(
        tmp_path / 'mismatched.ckpt',
        runtime_contract,
        state_dict={'encoder.0.weight': torch.ones((5, 5))},
    )

    with pytest.raises(ValueError, match='encoder.0.weight'):
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

@pytest.mark.parametrize(
    'saved',
    [
        None,
        LEGACY_ENCODER,
        shared_encoder_architecture(fusion='masked_mean', dimension=8, backbone=MINILM),
    ],
)
def test_weights_only_refuses_another_encoder_before_reading_any_parameter(
    tmp_path, runtime_contract, tiny_repaired_model, saved
):
    target = tiny_repaired_model.state_dict()
    key = sorted(name for name in target if name.startswith('encoder.'))[0]
    before = target[key].clone()
    contract = None if saved is None else runtime_contract.model_copy(update={'encoder': saved})
    path = _save(tmp_path / 'other.ckpt', contract, state_dict={key: torch.full_like(before, 0.25)})

    with pytest.raises(ValueError, match='weights-only encoder mismatch') as excinfo:
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

    assert D2_REFUSAL in str(excinfo.value)
    assert torch.equal(tiny_repaired_model.state_dict()[key], before)
