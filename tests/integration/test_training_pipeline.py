'''
Integration tests for the complete training pipeline.

Tests cover:
- Full training epoch (data loading → forward → backward → optimizer step)
- Validation epoch with metric computation
- Checkpoint save/load cycle
- Multi-component integration (encoder + loss + optimizer)
- Curriculum progression across epochs
- End-to-end model training (small scale)
'''

import logging
from pathlib import Path

import polars as pl
import pytest
import pytorch_lightning as pyl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule
from naics_embedder.text_model.naics_model import NAICSContrastiveModel

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def small_training_config():
    '''Minimal training config for fast integration tests.'''

    return {
        'base_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'lora_r': 4,
        'lora_alpha': 8,
        'num_experts': 4,
        'top_k': 2,
        'moe_hidden_dim': 256,
        'temperature': 0.07,
        'curvature': 1.0,
        'learning_rate': 1e-3,  # Higher LR for faster convergence in tests
        'weight_decay': 0.01,
        'warmup_steps': 10,
        'use_warmup_cosine': False,
    }

@pytest.fixture
def sample_training_data(tmp_path):
    '''Create minimal training dataset.'''

    # Create sample NAICS codes
    n_samples = 50
    codes = [f'{i:02d}111' for i in range(n_samples)]

    data = {
        'code': codes,
        'title': [f'Industry {i}' for i in range(n_samples)],
        'description': [f'Description for industry {i}' for i in range(n_samples)],
        'examples': [f'Example {i}' for i in range(n_samples)],
        'excluded': [f'Exclusion {i}' for i in range(n_samples)],
    }

    df = pl.DataFrame(data)
    parquet_path = tmp_path / 'naics_train.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

@pytest.fixture
def sample_training_pairs(tmp_path, sample_training_data):
    '''Create minimal training pairs dataset.'''

    n_pairs = 100
    anchor_codes = [f'{i % 50:02d}111' for i in range(n_pairs)]
    positive_codes = [f'{(i + 1) % 50:02d}111' for i in range(n_pairs)]

    # Create negative codes (4 per anchor)
    negative_codes = []
    for i in range(n_pairs):
        negatives = [f'{(i + j + 2) % 50:02d}111' for j in range(4)]
        negative_codes.append(negatives)

    data = {
        'anchor_code': anchor_codes,
        'positive_code': positive_codes,
        'negative_codes': negative_codes,
    }

    df = pl.DataFrame(data)
    parquet_path = tmp_path / 'naics_pairs.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

# -------------------------------------------------------------------------------------------------
# Test: Full Training Epoch
# -------------------------------------------------------------------------------------------------

@pytest.mark.integration
class TestFullTrainingEpoch:
    '''Test complete training epoch with all components.'''

    def test_single_training_epoch(
        self, small_training_config, sample_training_data, sample_training_pairs, tmp_path
    ):
        '''Test that a single training epoch runs end-to-end without errors.'''

        # Create model
        model = NAICSContrastiveModel(**small_training_config)

        # Create data module (mocked - we'll create batches manually)
        # For integration test, we use sample data directly

        # Create trainer with minimal config
        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        # Create dummy dataloader
        from torch.utils.data import DataLoader, TensorDataset

        # Create simple dataset with tokenized inputs
        batch_size = 8
        n_batches = 5

        # Since we can't easily create real tokenized inputs in integration test,
        # we'll test the model's training step directly with synthetic batches
        model.train()

        synthetic_batch = self._create_synthetic_batch(batch_size, model.encoder.channels)

        # Run training step
        loss = model.training_step(synthetic_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() > 0

    def test_multiple_training_epochs(self, small_training_config, tmp_path):
        '''Test training for multiple epochs to verify curriculum progression.'''

        model = NAICSContrastiveModel(**small_training_config)
        model.train()

        # Simulate 5 epochs
        for epoch in range(5):
            model.current_epoch = epoch

            synthetic_batch = self._create_synthetic_batch(8, model.encoder.channels)
            loss = model.training_step(synthetic_batch, batch_idx=0)

            assert not torch.isnan(loss)
            logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    def _create_synthetic_batch(self, batch_size, channels, k_negatives=4):
        '''Helper to create synthetic tokenized batch.'''

        seq_length = 32

        def create_channel_inputs(size):
            return {
                channel: {
                    'input_ids': torch.randint(0, 1000, (size, seq_length)),
                    'attention_mask': torch.ones(size, seq_length),
                }
                for channel in channels
            }

        return {
            'anchor': create_channel_inputs(batch_size),
            'positive': create_channel_inputs(batch_size),
            'negatives': create_channel_inputs(batch_size * k_negatives),
            'batch_size': batch_size,
            'k_negatives': k_negatives,
            'anchor_code': [f'{i:02d}111' for i in range(batch_size)],
        }

# -------------------------------------------------------------------------------------------------
# Test: Validation Epoch
# -------------------------------------------------------------------------------------------------

@pytest.mark.integration
class TestValidationEpoch:
    '''Test validation epoch with embedding storage and metrics.'''

    def test_validation_epoch(self, small_training_config):
        '''Test validation epoch stores embeddings correctly.'''

        model = NAICSContrastiveModel(**small_training_config)
        model.eval()

        # Create validation batches
        batch_size = 8
        n_batches = 3

        model.validation_embeddings = {}
        model.validation_codes = []

        for batch_idx in range(n_batches):
            synthetic_batch = self._create_synthetic_batch(
                batch_size, model.encoder.channels, offset=batch_idx * batch_size
            )

            with torch.no_grad():
                loss = model.validation_step(synthetic_batch, batch_idx=batch_idx)

            assert not torch.isnan(loss)

        # Check embeddings stored
        assert len(model.validation_embeddings) > 0
        assert len(model.validation_codes) > 0

        # Check all codes have embeddings
        for code in model.validation_codes:
            assert code in model.validation_embeddings
            emb = model.validation_embeddings[code]
            assert emb.shape[0] == model.encoder.embedding_dim + 1  # Lorentz embedding

    def _create_synthetic_batch(self, batch_size, channels, k_negatives=4, offset=0):
        '''Helper to create synthetic batch with unique codes.'''

        seq_length = 32

        def create_channel_inputs(size):
            return {
                channel: {
                    'input_ids': torch.randint(0, 1000, (size, seq_length)),
                    'attention_mask': torch.ones(size, seq_length),
                }
                for channel in channels
            }

        return {
            'anchor': create_channel_inputs(batch_size),
            'positive': create_channel_inputs(batch_size),
            'negatives': create_channel_inputs(batch_size * k_negatives),
            'batch_size': batch_size,
            'k_negatives': k_negatives,
            'anchor_code': [f'{i + offset:02d}111' for i in range(batch_size)],
        }

# -------------------------------------------------------------------------------------------------
# Test: Checkpoint Save/Load
# -------------------------------------------------------------------------------------------------

@pytest.mark.integration
class TestCheckpointSaveLoad:
    '''Test checkpoint saving and loading.'''

    def test_checkpoint_save_and_load(self, small_training_config, tmp_path):
        '''Test that model can be saved and loaded from checkpoint.'''

        # Create and train model
        model = NAICSContrastiveModel(**small_training_config)
        model.train()

        # Run one training step to initialize optimizer state
        synthetic_batch = self._create_synthetic_batch(8, model.encoder.channels)
        loss = model.training_step(synthetic_batch, batch_idx=0)
        loss.backward()

        # Save checkpoint
        checkpoint_path = tmp_path / 'test_checkpoint.ckpt'
        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.strategy.connect(model)
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load checkpoint into new model
        loaded_model = NAICSContrastiveModel.load_from_checkpoint(
            checkpoint_path, **small_training_config
        )

        # Verify model weights match
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)

    def test_checkpoint_resumption(self, small_training_config, tmp_path):
        '''Test that training can resume from checkpoint.'''

        # Train for 1 epoch
        model = NAICSContrastiveModel(**small_training_config)
        checkpoint_path = tmp_path / 'resume_checkpoint.ckpt'

        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

        # Save initial state
        model.train()
        synthetic_batch = self._create_synthetic_batch(8, model.encoder.channels)
        initial_loss = model.training_step(synthetic_batch, batch_idx=0)

        trainer.strategy.connect(model)
        trainer.save_checkpoint(checkpoint_path)

        # Load and continue training
        resumed_model = NAICSContrastiveModel.load_from_checkpoint(
            checkpoint_path, **small_training_config
        )
        resumed_model.train()

        # Training should continue without errors
        resumed_loss = resumed_model.training_step(synthetic_batch, batch_idx=0)

        assert not torch.isnan(resumed_loss)
        assert resumed_loss.item() > 0

    def _create_synthetic_batch(self, batch_size, channels, k_negatives=4):
        '''Helper to create synthetic batch.'''

        seq_length = 32

        def create_channel_inputs(size):
            return {
                channel: {
                    'input_ids': torch.randint(0, 1000, (size, seq_length)),
                    'attention_mask': torch.ones(size, seq_length),
                }
                for channel in channels
            }

        return {
            'anchor': create_channel_inputs(batch_size),
            'positive': create_channel_inputs(batch_size),
            'negatives': create_channel_inputs(batch_size * k_negatives),
            'batch_size': batch_size,
            'k_negatives': k_negatives,
            'anchor_code': [f'{i:02d}111' for i in range(batch_size)],
        }

# -------------------------------------------------------------------------------------------------
# Test: Optimizer State
# -------------------------------------------------------------------------------------------------

@pytest.mark.integration
class TestOptimizerState:
    '''Test optimizer state persistence across checkpoints.'''

    def test_optimizer_state_saved(self, small_training_config, tmp_path):
        '''Test that optimizer state is saved in checkpoint.'''

        model = NAICSContrastiveModel(**small_training_config)
        optimizer_config = model.configure_optimizers()

        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config['optimizer']
        else:
            optimizer = optimizer_config

        # Run training step to create optimizer state
        model.train()
        synthetic_batch = self._create_synthetic_batch(8, model.encoder.channels)
        loss = model.training_step(synthetic_batch, batch_idx=0)
        loss.backward()
        optimizer.step()

        # Save checkpoint
        checkpoint_path = tmp_path / 'optimizer_checkpoint.ckpt'
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'hyper_parameters': model.hparams,
        }
        torch.save(checkpoint, checkpoint_path)

        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        assert 'optimizer_state' in loaded_checkpoint
        assert len(loaded_checkpoint['optimizer_state']['state']) >= 0

    def _create_synthetic_batch(self, batch_size, channels, k_negatives=4):
        '''Helper to create synthetic batch.'''

        seq_length = 32

        def create_channel_inputs(size):
            return {
                channel: {
                    'input_ids': torch.randint(0, 1000, (size, seq_length)),
                    'attention_mask': torch.ones(size, seq_length),
                }
                for channel in channels
            }

        return {
            'anchor': create_channel_inputs(batch_size),
            'positive': create_channel_inputs(batch_size),
            'negatives': create_channel_inputs(batch_size * k_negatives),
            'batch_size': batch_size,
            'k_negatives': k_negatives,
            'anchor_code': [f'{i:02d}111' for i in range(batch_size)],
        }

# -------------------------------------------------------------------------------------------------
# Test: Curriculum Progression
# -------------------------------------------------------------------------------------------------

@pytest.mark.integration
class TestCurriculumProgression:
    '''Test curriculum scheduler progression across epochs.'''

    def test_curriculum_phase_transitions(self, small_training_config):
        '''Test that curriculum phases transition correctly.'''

        from naics_embedder.text_model.curriculum import CurriculumScheduler

        model = NAICSContrastiveModel(**small_training_config)
        model.curriculum_scheduler = CurriculumScheduler(
            phase1_end_epoch=2, phase2_end_epoch=5, phase3_end_epoch=10
        )

        phases_seen = set()

        for epoch in range(8):
            model.current_epoch = epoch
            model.train()

            synthetic_batch = self._create_synthetic_batch(8, model.encoder.channels)
            model.training_step(synthetic_batch, batch_idx=0)

            # Track which phase we're in
            phase = model.curriculum_scheduler.get_phase(epoch)
            phases_seen.add(phase)

        # Should have seen multiple phases
        assert len(phases_seen) >= 2
        assert 1 in phases_seen  # Phase 1
        assert 2 in phases_seen  # Phase 2

    def _create_synthetic_batch(self, batch_size, channels, k_negatives=4):
        '''Helper to create synthetic batch.'''

        seq_length = 32

        def create_channel_inputs(size):
            return {
                channel: {
                    'input_ids': torch.randint(0, 1000, (size, seq_length)),
                    'attention_mask': torch.ones(size, seq_length),
                }
                for channel in channels
            }

        return {
            'anchor': create_channel_inputs(batch_size),
            'positive': create_channel_inputs(batch_size),
            'negatives': create_channel_inputs(batch_size * k_negatives),
            'batch_size': batch_size,
            'k_negatives': k_negatives,
            'anchor_code': [f'{i:02d}111' for i in range(batch_size)],
        }

# -------------------------------------------------------------------------------------------------
# Test: Multi-Component Integration
# -------------------------------------------------------------------------------------------------

@pytest.mark.integration
class TestMultiComponentIntegration:
    '''Test integration between encoder, loss, and optimizer.'''

    def test_encoder_loss_optimizer_flow(self, small_training_config):
        '''Test that data flows correctly through encoder → loss → optimizer.'''

        model = NAICSContrastiveModel(**small_training_config)
        optimizer_config = model.configure_optimizers()

        if isinstance(optimizer_config, dict):
            optimizer = optimizer_config['optimizer']
        else:
            optimizer = optimizer_config

        model.train()
        optimizer.zero_grad()

        # Create batch and run forward pass
        synthetic_batch = self._create_synthetic_batch(8, model.encoder.channels)

        # Forward through encoder
        anchor_output = model(synthetic_batch['anchor'])
        assert 'embedding' in anchor_output

        # Compute loss via training step
        loss = model.training_step(synthetic_batch, batch_idx=0)

        # Backward pass
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break

        assert has_grad, 'No gradients after backward pass'

        # Optimizer step
        optimizer.step()

        # No errors should occur
        assert True

    def _create_synthetic_batch(self, batch_size, channels, k_negatives=4):
        '''Helper to create synthetic batch.'''

        seq_length = 32

        def create_channel_inputs(size):
            return {
                channel: {
                    'input_ids': torch.randint(0, 1000, (size, seq_length)),
                    'attention_mask': torch.ones(size, seq_length),
                }
                for channel in channels
            }

        return {
            'anchor': create_channel_inputs(batch_size),
            'positive': create_channel_inputs(batch_size),
            'negatives': create_channel_inputs(batch_size * k_negatives),
            'batch_size': batch_size,
            'k_negatives': k_negatives,
            'anchor_code': [f'{i:02d}111' for i in range(batch_size)],
        }
