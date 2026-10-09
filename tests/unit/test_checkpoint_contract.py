import importlib
import importlib.util

import pytest
import torch

from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision import schema
from naics_embedder.supervision.checkpoints import (
    D2_REFUSAL,
    LEGACY_ENCODER,
    CheckpointContract,
    EncoderArchitecture,
    contract_for_bundle,
    shared_encoder_architecture,
    validate_checkpoint_contract,
    validate_exact_resume,
    validate_supervision_contract,
)
from tests.fixtures.shared_encoder import (
    PRE_STAGE_7_FIELDS,
    PRE_STAGE_7_REFUSAL,
    pre_stage_7_contract,
)

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
SHARED = shared_encoder_architecture(
    fusion='masked_mean', dimension=16, backbone=MINILM, geometry='hyperbolic'
)
# Every field of a contract saved since Stage 7 (spec 4.5)
CONTRACT_FIELDS = {
    'contract_version',
    'bundle_id',
    'codebook_fingerprint',
    'objective',
    'encoder',
    'summaries',
}

@pytest.fixture
def summaries() -> str:
    '''MiniLM's summaries: under the test seam, the dummy pin's sha256.'''

    return summaries_identity(MINILM)

@pytest.fixture
def runtime_contract(summaries) -> CheckpointContract:
    return CheckpointContract(
        objective='req11-v1',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
        encoder=SHARED,
        summaries=summaries,
    )

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
    saved = {
        **runtime_contract.model_dump(),
        'bundle_id': 'bundle-b',
        'codebook_fingerprint': 'f' * 64,
    }

    with pytest.raises(ValueError, match='bundle_id') as excinfo:
        validate_checkpoint_contract(saved, runtime_contract)

    assert 'codebook_fingerprint' in str(excinfo.value)
    # A supervision mismatch is not the architecture refusal
    assert D2_REFUSAL not in str(excinfo.value)

def test_contract_for_bundle_reads_manifest_identity(validated_bundle, summaries):
    contract = contract_for_bundle(validated_bundle.manifest, encoder=SHARED, summaries=summaries)

    assert contract.objective == 'req11-v1'
    assert contract.bundle_id == 'bundle-a'
    assert contract.contract_version == 'stage3-supervision-v2'
    assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
    assert contract.encoder == SHARED
    assert contract.summaries == summaries
    # Spec 4.5: no field of the six-term objective's contract is left
    assert set(CheckpointContract.model_fields) == CONTRACT_FIELDS
    assert set(contract.model_dump()) == CONTRACT_FIELDS

# -------------------------------------------------------------------------------------------------
# The objective (spec 4.5, P23): a checkpoint trained before Stage 7 never loads (D2)
# -------------------------------------------------------------------------------------------------

def test_the_objective_and_its_legacy_marker_are_the_specs():
    '''
    Checkpoints save both strings, so neither may change: req11-v1 names Req 11's three terms, the
    radial form and the bound, and every older contract reads as pre-req11.
    '''

    assert (schema.OBJECTIVE, schema.LEGACY_OBJECTIVE) == ('req11-v1', 'pre-req11')

def _stage_6b(saved):
    '''Saved from Stage 6b until Stage 7: every field.'''

def _stage_6(saved):
    '''Saved before Stage 6b: no summaries.'''

    del saved['summaries']

def _stage_3(saved):
    '''Saved before Stage 6: no encoder record either.'''

    del saved['encoder']
    del saved['summaries']

def _legacy_containment(saved):
    '''Saved under legacy containment, before D2 deleted it.'''

    saved.update(
        supervision_mode='legacy_containment',
        bundle_id='legacy-containment',
        codebook_fingerprint='unversioned',
    )

def _exclusion_quota(saved):
    '''Trained before Stage 6 under negative-selection-v1, which kept a slot for an exclusion.'''

    _stage_3(saved)
    saved['mining_contract_version'] = 'negative-selection-v1'

PRE_STAGE_7_SHAPES = pytest.mark.parametrize(
    'shape',
    [_stage_6b, _stage_6, _stage_3, _legacy_containment, _exclusion_quota],
    ids=['stage-6b', 'stage-6', 'stage-3', 'legacy-containment', 'exclusion-quota'],
)

@PRE_STAGE_7_SHAPES
def test_a_contract_saved_before_stage_7_parses_and_reads_as_the_old_objective(
    runtime_contract, shape
):
    '''The three fields Stage 7 dropped are ignored, and the absent objective reads as pre-req11.'''

    saved = pre_stage_7_contract(runtime_contract.model_dump())
    shape(saved)

    contract = CheckpointContract.model_validate(saved)

    assert contract.objective == 'pre-req11'
    assert set(contract.model_dump()) == CONTRACT_FIELDS
    assert not set(PRE_STAGE_7_FIELDS) & set(contract.model_dump())
    assert (contract.bundle_id, contract.codebook_fingerprint) == (
        saved['bundle_id'], saved['codebook_fingerprint']
    )

def test_the_contract_still_refuses_a_field_it_never_had(runtime_contract):
    '''Only the three fields Stage 7 dropped are ignored: the contract forbids any other.'''

    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
        CheckpointContract.model_validate({**runtime_contract.model_dump(), 'loss_version': 'x'})

@PRE_STAGE_7_SHAPES
def test_exact_resume_refuses_a_pre_stage_7_checkpoint_on_its_objective_first(
    tmp_path, runtime_contract, shape
):
    '''
    The objective is checked before any other field, so each pre-Stage-7 contract meets the
    objective's refusal, whatever else of it differs: its encoder record, its summaries or its
    bundle.
    '''

    saved = pre_stage_7_contract(runtime_contract.model_dump())
    shape(saved)
    path = tmp_path / 'pre-stage-7.ckpt'
    torch.save({'stage3_supervision': saved}, path)

    with pytest.raises(ValueError, match=PRE_STAGE_7_REFUSAL) as refusal:
        validate_exact_resume(path, runtime_contract)

    assert 'contract mismatch' not in str(refusal.value)

@PRE_STAGE_7_SHAPES
def test_the_supervision_check_refuses_a_pre_stage_7_checkpoint_on_its_objective_first(
    validated_bundle, summaries, shape
):
    '''Export and reads refuse it as exact resume does, before comparing any other field.'''

    manifest = validated_bundle.manifest
    configured = contract_for_bundle(manifest, encoder=SHARED, summaries=summaries)
    saved = pre_stage_7_contract(configured.model_dump())
    shape(saved)

    with pytest.raises(ValueError, match=PRE_STAGE_7_REFUSAL) as refusal:
        validate_supervision_contract(saved, manifest, summaries=summaries)

    assert 'contract mismatch' not in str(refusal.value)

def test_a_checkpoint_of_any_other_objective_is_refused_naming_it(
    tmp_path, runtime_contract, validated_bundle, summaries
):
    '''Not only the legacy marker: a later objective is refused too, by name.'''

    refusal = r'objective req11-v2, not req11-v1 .*nothing migrates \(D2\)'
    later = runtime_contract.model_copy(update={'objective': 'req11-v2'})
    manifest = validated_bundle.manifest
    configured = contract_for_bundle(manifest, encoder=SHARED, summaries=summaries)
    saved = configured.model_copy(update={'objective': 'req11-v2'}).model_dump()

    with pytest.raises(ValueError, match=refusal):
        validate_exact_resume(_save(tmp_path / 'later.ckpt', later), runtime_contract)
    with pytest.raises(ValueError, match=refusal):
        validate_supervision_contract(saved, manifest, summaries=summaries)

# -------------------------------------------------------------------------------------------------
# The summaries (Stage 6b spec, 4.8)
# -------------------------------------------------------------------------------------------------

def test_a_contract_saved_before_stage_6b_reads_as_null_summaries(runtime_contract):
    saved = runtime_contract.model_dump()
    del saved['summaries']

    assert CheckpointContract.model_validate(saved).summaries is None

def test_a_checkpoint_under_other_summaries_cannot_exact_resume_naming_the_field(
    tmp_path, runtime_contract, summaries
):
    path = _save(
        tmp_path / 'truncated.ckpt', runtime_contract.model_copy(update={'summaries': None})
    )

    with pytest.raises(ValueError, match='exact resume') as refusal:
        validate_exact_resume(path, runtime_contract)

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

# -------------------------------------------------------------------------------------------------
# The encoder record (spec 4.4)
# -------------------------------------------------------------------------------------------------

def test_an_absent_encoder_record_reads_as_the_legacy_four_copy_layout(runtime_contract):
    saved = runtime_contract.model_dump()
    del saved['encoder']

    assert CheckpointContract.model_validate(saved).encoder == LEGACY_ENCODER
    assert LEGACY_ENCODER == EncoderArchitecture(layout='four-copy')

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
        {
            'layout': 'shared',
            'fusion': 'masked_mean',
            'dimension': 16,
            'backbone': MINILM,
            'geometry': 'poincare'
        },
        {
            'layout': 'four-copy',
            'geometry': 'spherical'
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
        'geometry': 'hyperbolic',
    }
    assert CheckpointContract.model_validate(saved) == runtime_contract

def test_an_encoder_record_saved_before_stage_8_reads_as_hyperbolic(tmp_path, runtime_contract):
    '''P7: no record saved before Stage 8 names a geometry, and each of them is hyperbolic.'''

    saved = runtime_contract.model_dump()
    del saved['encoder']['geometry']
    path = tmp_path / 'pre-stage-8.ckpt'
    torch.save({'stage3_supervision': saved}, path)

    assert CheckpointContract.model_validate(saved).encoder == SHARED
    validate_exact_resume(path, runtime_contract)

def test_a_shared_encoder_record_cannot_leave_out_its_geometry():
    assert SHARED.geometry == 'hyperbolic'
    with pytest.raises(TypeError, match='geometry'):
        shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone=MINILM)

@pytest.mark.parametrize(
    'encoder',
    [
        LEGACY_ENCODER,
        shared_encoder_architecture(
            fusion='masked_mean', dimension=8, backbone=MINILM, geometry='hyperbolic'
        ),
        shared_encoder_architecture(
            fusion='moe', dimension=16, backbone=MINILM, geometry='hyperbolic'
        ),
        shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone='other/model', geometry='hyperbolic'
        ),
        shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone=MINILM, geometry='euclidean'
        ),
        shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone=MINILM, geometry='spherical'
        ),
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
# Export and reads compare the supervision fields and the summaries, not the encoder record
# -------------------------------------------------------------------------------------------------

def test_the_supervision_check_takes_the_encoder_record_from_the_checkpoint(
    validated_bundle, summaries
):
    manifest = validated_bundle.manifest
    other = shared_encoder_architecture(
        fusion='attention', dimension=8, backbone=MINILM, geometry='spherical'
    )
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
# D2: legacy containment and the weights-only migration are gone
# -------------------------------------------------------------------------------------------------

def test_the_contract_helpers_take_no_supervision_mode(validated_bundle, summaries):
    '''No contract records a mode now that every run is repaired, so neither helper takes one.'''

    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED, summaries=summaries)

    with pytest.raises(TypeError):
        contract_for_bundle(manifest, 'repaired', encoder=SHARED, summaries=summaries)
    with pytest.raises(TypeError):
        validate_supervision_contract(saved.model_dump(), manifest, 'repaired', summaries=summaries)
    assert 'supervision_mode' not in saved.model_dump()

D2_DELETIONS = [
    'naics_embedder.supervision.mode',
    'naics_embedder.supervision.checkpoints:containment_contract',
    'naics_embedder.supervision.checkpoints:LEGACY_CONTAINMENT_BUNDLE_ID',
    'naics_embedder.supervision.checkpoints:UNVERSIONED_CODEBOOK_FINGERPRINT',
    'naics_embedder.supervision.checkpoints:load_weights_only',
    'naics_embedder.supervision.checkpoints:MigrationReport',
    'naics_embedder.supervision.checkpoints:saved_encoder',
    'naics_embedder.supervision.checkpoints:WEIGHTS_ONLY_ALLOWED_PREFIXES',
    'naics_embedder.supervision.checkpoints:WEIGHTS_ONLY_EXCLUDED_PREFIXES',
    'naics_embedder.cli.commands.training:announce_legacy_containment',
    'naics_embedder.cli.commands.training:log_migration_report',
    'naics_embedder.text_model.dataloader.datamodule:legacy_token_fingerprints',
    'naics_embedder.text_model.dataloader.datamodule:_collate_legacy',
    'naics_embedder.text_model.dataloader.datamodule:SUPERVISION_MODES',
    'naics_embedder.text_model.naics_model:NAICSContrastiveModel._legacy_containment_training_step',
    'naics_embedder.text_model.naics_model:NAICSContrastiveModel._load_ground_truth_distances',
    'naics_embedder.utils.config:CheckpointLoadMode.WEIGHTS_ONLY',
]

@pytest.mark.parametrize('path', D2_DELETIONS)
def test_d2_deleted_containment_and_the_weights_only_migration(path):
    '''Roadmap D2: nothing contains a legacy run, and nothing migrates a checkpoint's weights.'''

    module, _, attribute = path.partition(':')
    if not attribute:
        assert importlib.util.find_spec(module) is None
        return
    *owners, name = attribute.split('.')
    owner = importlib.import_module(module)
    for part in owners:
        owner = getattr(owner, part)
    assert not hasattr(owner, name)
