'''
Unit tests for NAICSContrastiveModel (PyTorch Lightning module).

Tests cover:
- Model initialization and configuration (repaired supervision bundle)
- Forward pass through encoder
- Repaired training step over one checked selection
- Validation step and embedding storage
- Optimizer and scheduler configuration
- Curriculum integration
- Post-selection false negative masking
- Selection health counters
- Distributed training utilities
- Checkpoint loading

Distributed candidate gathering is covered by the Gloo tests in
``tests/integration/test_distributed_supervision.py``.
'''

import logging
from unittest.mock import Mock, patch

import pytest
import pytorch_lightning as pyl
import torch

from naics_embedder.text_model.dataloader.datamodule import collate_fn
from naics_embedder.text_model.naics_model import (
    NAICSContrastiveModel,
    gather_embeddings_global,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def model_config(generated_bundle):
    '''Minimal model configuration for fast testing, backed by the five-code bundle fixture.'''

    return {
        'base_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'num_experts': 4,
        'top_k': 2,
        'moe_hidden_dim': 512,
        'temperature': 0.07,
        'curvature': 1.0,
        'hierarchy_weight': 0.0,  # Disabled for basic tests
        'radius_reg_weight': 0.01,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'use_warmup_cosine': False,
        'load_balancing_coef': 0.01,
        'fn_curriculum_start_epoch': 5,
        'fn_cluster_every_n_epochs': 3,
        'fn_num_clusters': 50,
        'eval_every_n_epochs': 1,
        'eval_sample_size': 100,
        'supervision_manifest_path': str(generated_bundle),
    }

@pytest.fixture
def naics_model(model_config, test_device):
    '''Create NAICSContrastiveModel instance for testing.'''

    model = NAICSContrastiveModel(**model_config)
    model.to(test_device)
    model.eval()
    return model

@pytest.fixture
def sample_training_batch(test_device, batch_size=4, k_negatives=8):
    '''Create sample training batch with anchor, positive, and negative samples.'''

    seq_length = 32
    channels = ['title', 'description', 'excluded', 'examples']

    def create_channel_inputs(batch_size):
        return {
            channel: {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=test_device),
                'attention_mask': torch.ones(batch_size, seq_length, device=test_device),
            }
            for channel in channels
        }

    batch = {
        'anchor': create_channel_inputs(batch_size),
        'positive': create_channel_inputs(batch_size),
        'negatives': create_channel_inputs(batch_size * k_negatives),
        'batch_size': batch_size,
        'k_negatives': k_negatives,
        'anchor_code': [f'{i:02d}111' for i in range(batch_size)],
        'positive_code': [f'{i:02d}1111' for i in range(batch_size)],
        'negative_codes': [[f'{j:02d}999' for j in range(k_negatives)] for _ in range(batch_size)],
        'positive_levels': [len(f'{i:02d}1111') for i in range(batch_size)],
    }

    return batch

# Five-code bundle fixture (tests/fixtures/supervision.py): code ID -> code. Anchor '111111' (0)
# explicitly excludes '111113' (2); '222222' (3) excludes '111112' (1).
BUNDLE_CODES = ('111111', '111112', '111113', '222222', '333333')
CHANNELS = ('title', 'description', 'excluded', 'examples')


def _tokens(code_id: int, seq_length: int = 32) -> dict:
    '''Deterministic per-code token inputs, so repeated codes encode identically.'''

    generator = torch.Generator().manual_seed(1000 + code_id)
    return {
        channel: {
            'input_ids': torch.randint(0, 1000, (seq_length, ), generator=generator),
            'attention_mask': torch.ones(seq_length, dtype=torch.long),
        }
        for channel in CHANNELS
    }


def _repaired_item(
    anchor_id: int,
    positive_id: int,
    positive_distance: float,
    positive_relation_id: int,
    pool: list,
    selection_k: int = 3,
) -> dict:
    return {
        'anchor_code_id': anchor_id,
        'anchor_code': BUNDLE_CODES[anchor_id],
        'anchor_embedding': _tokens(anchor_id),
        'positive_code_id': positive_id,
        'positive_code': BUNDLE_CODES[positive_id],
        'positive_embedding': _tokens(positive_id),
        'positive_structural_distance': positive_distance,
        'positive_structural_relation_id': positive_relation_id,
        'candidate_pool': [
            {
                'negative_code_id': code_id,
                'negative_code': BUNDLE_CODES[code_id],
                'negative_embedding': _tokens(code_id),
                'sampling_role_id': 2,
                'sampling_provenance_id': 2,
            }
            for code_id in pool
        ],
        'difficulty_proposal_indices': list(range(len(pool))),
        'selection_k': selection_k,
    }


@pytest.fixture
def repaired_training_batch():
    '''
    Two repaired rows with uneven pools: row 0 (anchor 0) holds its exclusion (2) and one padding
    row; row 1 (anchor 2) repeats code 4 and has no exclusion in its pool.
    '''

    return collate_fn(
        [
            _repaired_item(0, 1, 0.5, 1, [2, 3, 4]),
            _repaired_item(2, 0, 2.0, 2, [1, 3, 4, 4]),
        ],
        supervision_mode='repaired',
    )

# -------------------------------------------------------------------------------------------------
# Test: Initialization
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestModelInitialization:
    '''Test NAICSContrastiveModel initialization and configuration.'''

    def test_model_creation(self, naics_model, model_config):
        '''Test that model is created successfully with all components.'''

        assert isinstance(naics_model, pyl.LightningModule)
        assert hasattr(naics_model, 'encoder')
        assert hasattr(naics_model, 'loss_fn')
        assert hasattr(naics_model, 'hard_negative_miner')
        assert hasattr(naics_model, 'embedding_eval')

    def test_hyperparameters_saved(self, naics_model, model_config):
        '''Test that hyperparameters are saved correctly.'''

        # PyTorch Lightning saves hparams
        assert hasattr(naics_model, 'hparams')
        assert naics_model.hparams['learning_rate'] == model_config['learning_rate']
        assert naics_model.hparams['temperature'] == model_config['temperature']
        assert naics_model.hparams['curvature'] == model_config['curvature']

    def test_encoder_configuration(self, naics_model, model_config):
        '''Test that encoder is configured correctly.'''

        encoder = naics_model.encoder
        assert encoder.curvature == model_config['curvature']
        assert encoder.moe.num_experts == model_config['num_experts']
        assert encoder.moe.top_k == model_config['top_k']

    def test_loss_function_configuration(self, naics_model, model_config):
        '''Test that loss function is configured correctly.'''

        loss_fn = naics_model.loss_fn
        assert loss_fn.temperature == model_config['temperature']
        assert loss_fn.curvature == model_config['curvature']

    def test_hard_negative_miner_configuration(self, naics_model, model_config):
        '''Test that hard negative miner is configured correctly.'''

        miner = naics_model.hard_negative_miner
        assert miner.curvature == model_config['curvature']

    def test_ground_truth_distances_come_from_bundle(self, naics_model):
        '''Evaluation ground truth is the validated structural matrix in codebook order.'''

        index = naics_model.supervision_index
        assert naics_model.code_to_idx == index.code_to_id
        assert torch.equal(naics_model.ground_truth_distances, index.structural_distance)
        assert naics_model.ground_truth_distances[0, 2] == 2.0
        assert naics_model.hierarchy_loss_fn is None  # Weight is 0 in config

    def test_hierarchy_loss_with_weight(self, model_config, test_device):
        '''Hierarchy loss uses the bundle structural matrix when its weight is positive.'''

        model_config['hierarchy_weight'] = 0.1
        model = NAICSContrastiveModel(**model_config).to(test_device)

        assert model.hierarchy_loss_fn is not None
        assert torch.equal(
            model.hierarchy_loss_fn.tree_distances.cpu(),
            model.supervision_index.structural_distance,
        )

    def test_structural_preference_loss_configuration(self, model_config, test_device):
        '''Structural preference replaces LambdaRank and carries its configured weight.'''

        model_config['structural_preference_weight'] = 0.2
        model_config['structural_preference_margin'] = 0.3
        model = NAICSContrastiveModel(**model_config).to(test_device)

        loss_fn = model.structural_preference_loss_fn
        assert loss_fn.weight == 0.2
        assert loss_fn.margin == 0.3
        assert not hasattr(model, 'lambdarank_loss_fn')

    def test_evaluation_hierarchy_comes_from_bundle(self, naics_model):
        '''The evaluation hierarchy reads the bundle relations artifact (child ID 1).'''

        assert naics_model.naics_hierarchy is not None
        assert naics_model.naics_hierarchy.get_parent('111112') == '111111'

    def test_repaired_mode_requires_manifest(self, model_config):
        '''Repaired training fails closed without a supervision manifest.'''

        model_config.pop('supervision_manifest_path')
        with pytest.raises(ValueError, match='supervision_manifest_path'):
            NAICSContrastiveModel(**model_config)

    def test_repaired_mode_rejects_legacy_ground_truth_inputs(self, model_config, tmp_path):
        '''Legacy distance and relation files cannot mix with the bundle authority.'''

        with pytest.raises(ValueError, match='distance_matrix_path'):
            NAICSContrastiveModel(
                **model_config,
                distance_matrix_path=str(tmp_path / 'naics_distance_matrix.parquet'),
            )
        with pytest.raises(ValueError, match='relations_parquet_path'):
            NAICSContrastiveModel(
                **model_config,
                relations_parquet_path=str(tmp_path / 'naics_relations.parquet'),
            )

    def test_rejects_mismatched_contract_version(self, model_config):
        '''The bundle must match the expected supervision contract.'''

        with pytest.raises(ValueError, match='contract'):
            NAICSContrastiveModel(
                **model_config,
                supervision_contract_version='stage3-supervision-v0',
            )

    def test_rejects_unknown_supervision_mode(self, model_config):
        with pytest.raises(ValueError, match='supervision mode'):
            NAICSContrastiveModel(**model_config, supervision_mode='mystery')

# -------------------------------------------------------------------------------------------------
# Test: Forward Pass
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestForwardPass:
    '''Test model forward pass.'''

    def test_forward_basic(self, naics_model, sample_training_batch):
        '''Test basic forward pass through encoder.'''

        with torch.no_grad():
            output = naics_model(sample_training_batch['anchor'])

        assert 'embedding' in output
        assert 'embedding_euc' in output
        assert 'gate_probs' in output
        assert 'top_k_indices' in output

    def test_forward_output_shapes(self, naics_model, sample_training_batch):
        '''Test forward pass output shapes.'''

        batch_size = sample_training_batch['batch_size']

        with torch.no_grad():
            output = naics_model(sample_training_batch['anchor'])

        # Hyperbolic embedding: (batch_size, embedding_dim + 1)
        assert output['embedding'].shape == (batch_size, naics_model.encoder.embedding_dim + 1)

        # Euclidean embedding: (batch_size, embedding_dim)
        assert output['embedding_euc'].shape == (batch_size, naics_model.encoder.embedding_dim)

# -------------------------------------------------------------------------------------------------
# Test: Training Step
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestTrainingStep:
    '''Test the repaired training step.'''

    def test_training_step_basic(self, naics_model, repaired_training_batch):
        '''Test basic training step runs without errors.'''

        naics_model.train()
        loss = naics_model.training_step(repaired_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_training_step_gradient_flow(self, naics_model, repaired_training_batch):
        '''Uneven pools train with finite gradients; padding rows never reach the encoder.'''

        naics_model.train()
        loss = naics_model.training_step(repaired_training_batch, batch_idx=0)
        loss.backward()

        grads = [p.grad for p in naics_model.parameters() if p.requires_grad]
        assert any(grad is not None for grad in grads)
        assert all(grad is None or torch.isfinite(grad).all() for grad in grads)

    def test_forward_candidate_pool_encodes_only_valid_rows(
        self, naics_model, repaired_training_batch
    ):
        '''Padding rows get zero outputs and source slot -1; valid UIDs are (rank, row, slot).'''

        encoded_rows = []
        original_forward = naics_model.encoder.forward

        def spy_forward(channel_inputs):
            encoded_rows.append(channel_inputs['title']['input_ids'].shape[0])
            return original_forward(channel_inputs)

        naics_model.encoder.forward = spy_forward
        with torch.no_grad():
            output, uid = naics_model._forward_candidate_pool(repaired_training_batch)

        assert encoded_rows == [7]  # 3 + 4 valid rows of the 2 x 4 pool
        assert torch.count_nonzero(output['embedding'][3]) == 0
        assert uid[0, 3].tolist() == [0, 0, -1]
        assert uid[1].tolist() == [[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3]]

    def test_training_step_with_curriculum(self, naics_model, repaired_training_batch):
        '''Test training step with curriculum scheduler.'''

        from naics_embedder.text_model.curriculum import CurriculumScheduler

        # Initialize curriculum scheduler
        naics_model.curriculum_scheduler = CurriculumScheduler(
            max_epochs=15, phase1_end=0.33, phase2_end=0.67, phase3_end=1.0
        )

        naics_model.train()
        loss = naics_model.training_step(repaired_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_training_step_false_negative_masking(
        self, naics_model, repaired_training_batch, monkeypatch
    ):
        '''Pseudo-related masking happens after selection and never clears an exclusion.'''

        naics_model.current_curriculum_flags = {'enable_clustering': True}
        monkeypatch.setattr(naics_model, '_update_curriculum_state', lambda *_args: None)
        # '111111' shares a cluster with its exclusion '111113' and with '222222'
        naics_model.code_to_pseudo_label = {'111111': 1, '111113': 1, '222222': 1}

        captured = {}
        original = naics_model._compute_contrastive_loss

        def spy(anchor_emb, positive_emb, selected, effective_mask):
            captured['codes'] = selected.code_id[0].tolist()
            captured['mask'] = effective_mask[0].tolist()
            return original(anchor_emb, positive_emb, selected, effective_mask)

        monkeypatch.setattr(naics_model, '_compute_contrastive_loss', spy)
        naics_model.train()
        loss = naics_model.training_step(repaired_training_batch, batch_idx=0)

        flags = dict(zip(captured['codes'], captured['mask']))
        assert flags == {2: False, 3: True, 4: False}
        assert not torch.isnan(loss)

    def test_selection_health_counters_are_epoch_sums(
        self, naics_model, repaired_training_batch, monkeypatch
    ):
        '''Integrity counters are logged per batch as epoch sums without identities.'''

        log = Mock()
        monkeypatch.setattr(naics_model, 'log', log)
        naics_model.train()
        naics_model.training_step(repaired_training_batch, batch_idx=1)

        counters = {
            call.args[0]: call.args[1].item()
            for call in log.call_args_list
            if call.args[0].startswith('train/integrity/')
        }
        assert counters == {
            'train/integrity/anchors_with_exclusions': 1.0,
            'train/integrity/quota_selections': 1.0,
            'train/integrity/invalid_candidates_ignored': 1.0,
            'train/integrity/deterministic_backfills': 0.0,
            'train/integrity/duplicate_candidates_removed': 1.0,
        }
        for call in log.call_args_list:
            if call.args[0].startswith('train/integrity/'):
                assert call.kwargs['reduce_fx'] == 'sum'
                assert call.kwargs['on_epoch'] is True
                assert call.kwargs['on_step'] is False

    def test_training_step_logs_structural_preference(
        self, naics_model, repaired_training_batch, monkeypatch
    ):
        '''The structural preference term is logged under its own key; LambdaRank is gone.'''

        log = Mock()
        monkeypatch.setattr(naics_model, 'log', log)
        naics_model.train()
        naics_model.training_step(repaired_training_batch, batch_idx=1)

        keys = {call.args[0] for call in log.call_args_list}
        assert 'train/structural_preference_loss' in keys
        assert not any('lambdarank' in key for key in keys)

    def test_training_step_load_balancing_loss(self, naics_model, repaired_training_batch):
        '''Test that load balancing loss is computed.'''

        naics_model.train()
        naics_model.load_balancing_coef = 0.01

        loss = naics_model.training_step(repaired_training_batch, batch_idx=0)

        # Loss should include load balancing component
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_combine_loss_terms_scales_load_balancing(self, naics_model):
        '''Ensure load balancing term is scaled before contributing to total loss.'''

        naics_model.load_balancing_coef = 0.25

        contrastive_loss = torch.tensor(1.0)
        load_balancing_loss = torch.tensor(2.0)
        hierarchy_loss = torch.tensor(0.3)
        structural_preference_loss = torch.tensor(0.2)
        radius_reg_loss = torch.tensor(0.1)
        level_radius_loss = torch.tensor(0.05)

        total_loss, scaled_load_balancing = naics_model._combine_loss_terms(
            contrastive_loss,
            load_balancing_loss,
            hierarchy_loss,
            structural_preference_loss,
            radius_reg_loss,
            level_radius_loss,
        )

        expected_scaled = load_balancing_loss * naics_model.load_balancing_coef
        expected_total = (
            contrastive_loss + expected_scaled + hierarchy_loss + structural_preference_loss +
            radius_reg_loss + level_radius_loss
        )

        assert torch.isclose(scaled_load_balancing, expected_scaled)
        assert torch.isclose(total_loss, expected_total)

# -------------------------------------------------------------------------------------------------
# Test: Validation Step
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestValidationStep:
    '''Test validation step functionality.'''

    def test_validation_step_basic(self, naics_model, repaired_training_batch):
        '''Test basic validation step runs without errors.'''

        naics_model.eval()
        with torch.no_grad():
            loss = naics_model.validation_step(repaired_training_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_validation_step_uses_no_selection(
        self, naics_model, repaired_training_batch, monkeypatch
    ):
        '''Validation scores the whole valid pool; mining and selection never run.'''

        monkeypatch.setattr(
            naics_model.selection_coordinator,
            'select',
            Mock(side_effect=AssertionError('selection during validation')),
        )
        naics_model.current_curriculum_flags = {
            'enable_hard_negative_mining': True,
            'enable_router_guided_sampling': True,
            'enable_clustering': True,
        }
        naics_model.eval()
        with torch.no_grad():
            loss = naics_model.validation_step(repaired_training_batch, batch_idx=0)

        assert torch.isfinite(loss)

    def test_validation_step_embedding_storage(self, naics_model, repaired_training_batch):
        '''Test that validation step stores embeddings.'''

        naics_model.eval()
        naics_model.validation_embeddings = {}
        naics_model.validation_codes = []

        with torch.no_grad():
            naics_model.validation_step(repaired_training_batch, batch_idx=0)

        # Check that embeddings were stored
        assert len(naics_model.validation_embeddings) > 0
        assert len(naics_model.validation_codes) > 0

        # Check that embeddings match codes
        for code in naics_model.validation_codes:
            assert code in naics_model.validation_embeddings
            embedding = naics_model.validation_embeddings[code]
            assert embedding.shape[0] == naics_model.encoder.embedding_dim + 1  # Lorentz

    def test_validation_step_no_duplicate_codes(self, naics_model, repaired_training_batch):
        '''Test that validation step doesn\'t store duplicate codes.'''

        naics_model.eval()
        naics_model.validation_embeddings = {}
        naics_model.validation_codes = []

        # Run validation step twice with same batch
        with torch.no_grad():
            naics_model.validation_step(repaired_training_batch, batch_idx=0)
            initial_count = len(naics_model.validation_codes)

            naics_model.validation_step(repaired_training_batch, batch_idx=1)
            final_count = len(naics_model.validation_codes)

        # Should not add duplicates
        assert final_count == initial_count

# -------------------------------------------------------------------------------------------------
# Test: Optimizer Configuration
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestOptimizerConfiguration:
    '''Test optimizer and scheduler configuration.'''

    def test_configure_optimizers_basic(self, naics_model):
        '''Test basic optimizer configuration.'''

        optimizer_config = naics_model.configure_optimizers()

        # Should return optimizer
        assert 'optimizer' in optimizer_config or isinstance(
            optimizer_config, torch.optim.Optimizer
        )

    def test_adamw_optimizer(self, naics_model, model_config):
        '''Test that AdamW optimizer is configured correctly.'''

        optimizer_config = naics_model.configure_optimizers()

        # Extract optimizer
        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config['optimizer']
        else:
            optimizer = optimizer_config

        assert isinstance(optimizer, torch.optim.AdamW)

        # Check learning rate
        assert optimizer.param_groups[0]['lr'] == model_config['learning_rate']

        # Check weight decay
        assert optimizer.param_groups[0]['weight_decay'] == model_config['weight_decay']

    def test_warmup_cosine_scheduler(self, model_config, test_device):
        '''Test warmup + cosine scheduler configuration.'''

        model_config['use_warmup_cosine'] = True
        model_config['warmup_steps'] = 100

        model = NAICSContrastiveModel(**model_config).to(test_device)
        optimizer_config = model.configure_optimizers()

        # Should return dict with optimizer and lr_scheduler
        assert isinstance(optimizer_config, dict)
        assert 'optimizer' in optimizer_config
        assert 'lr_scheduler' in optimizer_config

        # Check scheduler configuration
        lr_scheduler_config = optimizer_config['lr_scheduler']
        assert 'scheduler' in lr_scheduler_config
        assert 'interval' in lr_scheduler_config

# -------------------------------------------------------------------------------------------------
# Test: Distributed Utilities
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestDistributedUtilities:
    '''Test distributed training utility functions.'''

    def test_gather_embeddings_single_gpu(self, test_device):
        '''Test gather_embeddings_global with single GPU (no-op).'''

        embeddings = torch.randn(16, 385, device=test_device)

        # Without distributed environment, should return input unchanged
        with patch('torch.distributed.is_initialized', return_value=False):
            gathered = gather_embeddings_global(embeddings)

            assert gathered is embeddings
            torch.testing.assert_close(gathered, embeddings)

    def test_gather_embeddings_world_size_one(self, test_device):
        '''Test gather_embeddings_global with world_size=1.'''

        embeddings = torch.randn(16, 385, device=test_device)

        with (
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=1),
        ):
            gathered = gather_embeddings_global(embeddings)

            assert gathered is embeddings

    @patch('torch.distributed.all_gather')
    def test_gather_embeddings_multi_gpu(self, mock_all_gather, test_device):
        '''Test gather_embeddings_global with multiple GPUs (mocked).'''

        local_embeddings = torch.randn(8, 385, device=test_device)
        world_size = 4

        # Mock all_gather to simulate gathering from multiple GPUs
        def mock_gather_fn(gathered_list, tensor):
            # Simulate gathering: each rank contributes the same tensor
            for i in range(len(gathered_list)):
                gathered_list[i] = tensor.clone()

        mock_all_gather.side_effect = mock_gather_fn

        with (
            patch('torch.distributed.is_initialized', return_value=True),
            patch('torch.distributed.get_world_size', return_value=world_size),
        ):
            gathered = gather_embeddings_global(local_embeddings)

            # Should concatenate world_size copies of local embeddings
            assert gathered.shape[0] == local_embeddings.shape[0] * world_size
            assert gathered.shape[1] == local_embeddings.shape[1]

# -------------------------------------------------------------------------------------------------
# Test: Curriculum Integration
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCurriculumIntegration:
    '''Test curriculum scheduler integration.'''

    def test_curriculum_flags_update(self, naics_model, repaired_training_batch):
        '''Test that curriculum flags are updated during training.'''

        from naics_embedder.text_model.curriculum import CurriculumScheduler

        naics_model.curriculum_scheduler = CurriculumScheduler(
            max_epochs=15, phase1_end=0.33, phase2_end=0.67, phase3_end=1.0
        )

        # Set epoch to phase 2
        naics_model.trainer = Mock()
        naics_model.trainer.current_epoch = 7

        naics_model.train()
        naics_model.training_step(repaired_training_batch, batch_idx=0)

        # Check that curriculum flags were updated
        assert len(naics_model.current_curriculum_flags) > 0

    def test_curriculum_phase_transition(self, naics_model, repaired_training_batch):
        '''Test curriculum phase transition logging.'''

        from naics_embedder.text_model.curriculum import CurriculumScheduler

        naics_model.curriculum_scheduler = CurriculumScheduler(
            max_epochs=15, phase1_end=0.33, phase2_end=0.67, phase3_end=1.0
        )

        # Transition from phase 1 to phase 2
        naics_model.trainer = Mock()
        naics_model.trainer.current_epoch = 5
        naics_model.previous_phase = 1

        naics_model.train()
        naics_model.training_step(repaired_training_batch, batch_idx=0)

        # Previous phase should be updated
        current_phase = naics_model.curriculum_scheduler.get_phase(naics_model.current_epoch)
        assert naics_model.previous_phase == current_phase

# -------------------------------------------------------------------------------------------------
# Test: Checkpoint Loading
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointLoading:
    '''Test model checkpoint saving and loading.'''

    def test_state_dict_save_load(self, naics_model, test_device):
        '''Test that model state dict can be saved and loaded.'''

        # Save state dict
        state_dict = naics_model.state_dict()

        # Create new model
        new_model = NAICSContrastiveModel(**naics_model.hparams).to(test_device)

        # Load state dict
        new_model.load_state_dict(state_dict)

        # Check that parameters match
        for (name1, param1), (name2, param2) in zip(
            naics_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)

    def test_hparams_save_load(self, naics_model):
        '''Test that hyperparameters are saved correctly.'''

        hparams = naics_model.hparams

        # Hyperparameters should be accessible
        assert 'learning_rate' in hparams
        assert 'curvature' in hparams
        assert 'temperature' in hparams

# -------------------------------------------------------------------------------------------------
# Test: Supervision checkpoint contract
# -------------------------------------------------------------------------------------------------

def _lightning_checkpoint(model) -> dict:
    checkpoint = {
        'state_dict': model.state_dict(),
        'hyper_parameters': dict(model.hparams),
        'pytorch-lightning_version': pyl.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    return checkpoint

@pytest.mark.unit
class TestCheckpointContract:
    '''The supervision contract travels with checkpoints and gates every restore.'''

    def test_model_contract_matches_its_bundle(self, naics_model, validated_bundle):
        contract = naics_model.checkpoint_contract

        assert contract.supervision_mode == 'repaired'
        assert contract.bundle_id == validated_bundle.manifest.bundle_id
        assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint

    def test_on_save_checkpoint_writes_contract(self, naics_model):
        checkpoint = {}
        naics_model.on_save_checkpoint(checkpoint)

        assert checkpoint['stage3_supervision'] == naics_model.checkpoint_contract.model_dump()

    def test_on_load_checkpoint_rejects_legacy_and_mismatched_contracts(self, naics_model):
        contract = naics_model.checkpoint_contract.model_dump()

        with pytest.raises(ValueError, match='exact resume'):
            naics_model.on_load_checkpoint({})
        with pytest.raises(ValueError, match='exact resume'):
            naics_model.on_load_checkpoint(
                {'stage3_supervision': {**contract, 'bundle_id': 'other-bundle'}}
            )
        naics_model.on_load_checkpoint({'stage3_supervision': contract})

    def test_runtime_contract_must_match_the_loaded_bundle(self, model_config):
        from naics_embedder.supervision.checkpoints import CheckpointContract

        other = CheckpointContract(
            supervision_mode='repaired',
            bundle_id='other-bundle',
            codebook_fingerprint='f' * 64,
        )
        with pytest.raises(ValueError, match='bundle'):
            NAICSContrastiveModel(**model_config, checkpoint_contract=other)

    def test_prevalidated_bundle_is_reused_and_kept_out_of_hparams(
        self, model_config, validated_bundle, monkeypatch
    ):
        import naics_embedder.text_model.naics_model as model_module

        monkeypatch.setattr(
            model_module,
            'load_validated_bundle',
            Mock(side_effect=AssertionError('bundle validated twice')),
        )
        model = NAICSContrastiveModel(**model_config, supervision_bundle=validated_bundle)

        assert model.supervision_bundle_id == validated_bundle.manifest.bundle_id
        assert 'supervision_bundle' not in model.hparams
        assert 'checkpoint_contract' not in model.hparams

    def test_prevalidated_bundle_must_be_the_configured_manifest(
        self, model_config, validated_bundle, tmp_path
    ):
        model_config['supervision_manifest_path'] = str(tmp_path / 'other' / 'manifest.json')

        with pytest.raises(ValueError, match='manifest'):
            NAICSContrastiveModel(**model_config, supervision_bundle=validated_bundle)

    def test_load_from_checkpoint_round_trips_the_contract(self, naics_model, tmp_path):
        path = tmp_path / 'repaired.ckpt'
        torch.save(_lightning_checkpoint(naics_model), path)

        restored = NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

        assert restored.checkpoint_contract == naics_model.checkpoint_contract

    def test_load_from_checkpoint_rejects_a_legacy_checkpoint(self, naics_model, tmp_path):
        checkpoint = _lightning_checkpoint(naics_model)
        del checkpoint['stage3_supervision']
        path = tmp_path / 'legacy.ckpt'
        torch.save(checkpoint, path)

        with pytest.raises(ValueError, match='exact resume'):
            NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

# -------------------------------------------------------------------------------------------------
# Test: Explicit legacy containment
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def runtime_contract():
    from naics_embedder.supervision.checkpoints import CheckpointContract

    return CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
    )

@pytest.fixture
def legacy_model(model_config):
    return NAICSContrastiveModel(
        **model_config,
        supervision_mode='legacy_containment',
    )

@pytest.fixture
def legacy_batch(sample_training_batch):
    return sample_training_batch

def test_legacy_containment_disables_contaminated_and_reordering_paths(
    legacy_model, legacy_batch, monkeypatch
):
    monkeypatch.setattr(legacy_model, 'log', Mock())
    forbidden = [
        ('hard_negative_miner', 'propose'),
        ('router_guided_miner', 'propose'),
        ('structural_preference_loss_fn', 'forward'),
        ('hierarchy_loss_fn', 'forward'),
    ]
    for name, method in forbidden:
        value = getattr(legacy_model, name, None)
        if value is not None:
            monkeypatch.setattr(value, method, Mock(side_effect=AssertionError(name)))

    monkeypatch.setattr(
        legacy_model,
        '_build_selected_pseudo_related_mask',
        Mock(side_effect=AssertionError('pseudo-related handling')),
    )

    loss = legacy_model.training_step(legacy_batch, 0)

    assert torch.isfinite(loss)
    assert legacy_model.checkpoint_contract.supervision_mode == 'legacy_containment'

def test_containment_checkpoint_cannot_resume_repaired(runtime_contract):
    containment = runtime_contract.model_copy(
        update={
            'supervision_mode': 'legacy_containment',
            'bundle_id': 'legacy-containment',
            'codebook_fingerprint': 'unversioned',
        }
    )

    assert containment != runtime_contract

def test_legacy_containment_disables_structural_losses_even_with_old_weights(model_config):
    model_config['hierarchy_weight'] = 0.45
    model = NAICSContrastiveModel(**model_config, supervision_mode='legacy_containment')

    assert model.hierarchy_loss_fn is None
    assert model.structural_preference_loss_fn is None
    assert model.checkpoint_contract.bundle_id == 'legacy-containment'
    assert model.checkpoint_contract.codebook_fingerprint == 'unversioned'

def test_legacy_containment_logs_integrity_tag(legacy_model, legacy_batch, monkeypatch):
    log = Mock()
    monkeypatch.setattr(legacy_model, 'log', log)

    legacy_model.training_step(legacy_batch, 0)

    tags = [
        call for call in log.call_args_list
        if call.args[0] == 'train/integrity/legacy_containment'
    ]
    assert tags and tags[0].args[1] == 1.0

def test_legacy_containment_validation_uses_local_negatives(legacy_model, legacy_batch):
    legacy_model.eval()
    with torch.no_grad():
        loss = legacy_model.validation_step(legacy_batch, batch_idx=0)

    assert torch.isfinite(loss)

def test_containment_checkpoint_never_restores_into_repaired_model(naics_model, legacy_model):
    checkpoint = {}
    legacy_model.on_save_checkpoint(checkpoint)

    with pytest.raises(ValueError, match='exact resume'):
        naics_model.on_load_checkpoint(checkpoint)
    legacy_model.on_load_checkpoint(checkpoint)

# -------------------------------------------------------------------------------------------------
# Test: Numerical Stability
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestNumericalStability:
    '''Test model numerical stability.'''

    def test_no_nan_in_training(self, naics_model, repaired_training_batch):
        '''Test that training produces no NaN values.'''

        naics_model.train()
        loss = naics_model.training_step(repaired_training_batch, batch_idx=0)

        assert not torch.isnan(loss)

        # Check gradients
        loss.backward()
        for name, param in naics_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f'NaN gradient in {name}'

    def test_no_inf_in_training(self, naics_model, repaired_training_batch):
        '''Test that training produces no Inf values.'''

        naics_model.train()
        loss = naics_model.training_step(repaired_training_batch, batch_idx=0)

        assert not torch.isinf(loss)

    def test_extreme_temperature(self, model_config, repaired_training_batch, test_device):
        '''Test model with extreme temperature values.'''

        for temperature in [0.01, 1.0]:
            model_config['temperature'] = temperature
            model = NAICSContrastiveModel(**model_config).to(test_device)

            model.train()
            loss = model.training_step(repaired_training_batch, batch_idx=0)

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

# -------------------------------------------------------------------------------------------------
# Test: Logging and Metrics
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLoggingAndMetrics:
    '''Test logging and metric tracking.'''

    def test_to_python_scalar(self, naics_model):
        '''Test _to_python_scalar conversion utility.'''

        # Test tensor conversion
        tensor_val = torch.tensor(3.14)
        assert isinstance(naics_model._to_python_scalar(tensor_val), float)

        # Test bool conversion
        bool_val = True
        assert isinstance(naics_model._to_python_scalar(bool_val), int)

        # Test float conversion
        float_val = 2.718
        assert isinstance(naics_model._to_python_scalar(float_val), float)

    def test_metrics_file_path(self, naics_model, tmp_path):
        '''Test metrics file path generation.'''

        # Mock logger with log_dir
        mock_logger = Mock()
        mock_logger.log_dir = str(tmp_path / 'logs')

        naics_model.trainer = Mock()
        naics_model.trainer.logger = mock_logger

        metrics_path = naics_model._get_metrics_file_path()

        assert metrics_path is not None
        assert 'evaluation_metrics.json' in str(metrics_path)

# -------------------------------------------------------------------------------------------------
# Test: Edge Cases
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    '''Test edge cases and error handling.'''

    def test_batch_size_one(self, naics_model):
        '''Test model handles batch size of 1.'''

        batch = collate_fn(
            [_repaired_item(0, 1, 0.5, 1, [2, 3, 4])],
            supervision_mode='repaired',
        )

        naics_model.train()
        loss = naics_model.training_step(batch, batch_idx=0)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_selection_capacity_failure_is_fatal(self, naics_model):
        '''An anchor without K selectable codes aborts the step with its code ID.'''

        batch = collate_fn(
            [_repaired_item(0, 1, 0.5, 1, [2, 3], selection_k=3)],
            supervision_mode='repaired',
        )

        naics_model.train()
        with pytest.raises(ValueError, match='anchor code ID 0'):
            naics_model.training_step(batch, batch_idx=0)
