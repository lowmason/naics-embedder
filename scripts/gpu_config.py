import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Memory calculation utilities
# -------------------------------------------------------------------------------------------------

@dataclass
class ModelConfig:
    '''Model architecture configuration'''
    base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dim: int = 384  # MiniLM-L6-v2
    num_channels: int = 4  # title, description, excluded, examples
    max_length: int = 512
    lora_r: int = 8
    num_experts: int = 4
    top_k: int = 2
    moe_hidden_dim: int = 1024
    use_fp16: bool = True


@dataclass
class TrainingConfig:
    '''Training configuration'''
    batch_size: int = 4
    n_positives: int = 16
    n_negatives: int = 8
    accumulate_grad_batches: int = 16
    gradient_checkpointing: bool = True


@dataclass
class MemoryEstimate:
    '''Breakdown of memory usage'''
    model_params_mb: float
    optimizer_state_mb: float
    gradients_mb: float
    activations_mb: float
    batch_data_mb: float
    overhead_mb: float
    total_mb: float
    
    def __str__(self) -> str:
        return (
            f"\nMemory Breakdown:\n"
            f"  Model Parameters:    {self.model_params_mb:>8.1f} MB\n"
            f"  Optimizer State:     {self.optimizer_state_mb:>8.1f} MB\n"
            f"  Gradients:           {self.gradients_mb:>8.1f} MB\n"
            f"  Activations:         {self.activations_mb:>8.1f} MB\n"
            f"  Batch Data:          {self.batch_data_mb:>8.1f} MB\n"
            f"  Overhead:            {self.overhead_mb:>8.1f} MB\n"
            f"  {'─' * 35}\n"
            f"  Total Estimated:     {self.total_mb:>8.1f} MB ({self.total_mb/1024:.2f} GB)"
        )


def estimate_model_parameters(model_cfg: ModelConfig) -> int:
    '''
    Estimate total number of trainable parameters.
    
    For LoRA-adapted models, we only train:
    - LoRA parameters in each channel encoder
    - MoE expert networks
    - MoE gating network
    - Projection layers
    
    But we need to load the base model in memory for all channels.
    '''
    # Base model parameters (full model loaded for each channel)
    # MiniLM-L6-v2 has ~22.7M params per instance
    base_model_params = 22_700_000
    total_base_params = base_model_params * model_cfg.num_channels
    
    # LoRA parameters per channel
    # LoRA adds: r * d * 2 per linear layer
    # Approximate: ~50 linear layers in a transformer encoder
    lora_params_per_channel = 2 * model_cfg.lora_r * model_cfg.embedding_dim * 50
    total_lora_params = lora_params_per_channel * model_cfg.num_channels
    
    # MoE parameters
    # Each expert: input_dim -> hidden_dim -> input_dim
    input_dim = model_cfg.embedding_dim * model_cfg.num_channels
    expert_params = (input_dim * model_cfg.moe_hidden_dim + model_cfg.moe_hidden_dim * input_dim)
    total_expert_params = expert_params * model_cfg.num_experts
    
    # Gating network: input_dim -> num_experts
    gating_params = input_dim * model_cfg.num_experts
    
    # Projection layer: input_dim -> embedding_dim
    projection_params = input_dim * model_cfg.embedding_dim
    
    # Total parameters in memory (base models are frozen but still in memory)
    total_params = (
        total_base_params + 
        total_lora_params + 
        total_expert_params + 
        gating_params + 
        projection_params
    )
    
    return total_params


def estimate_activation_memory(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig
) -> float:
    '''
    Estimate activation memory per forward pass.
    
    Activations include intermediate tensors during forward pass:
    - Encoder outputs for each channel
    - Attention maps
    - MoE intermediate activations
    
    With gradient checkpointing, this is significantly reduced.
    '''
    batch_size = train_cfg.batch_size
    n_total_samples = batch_size * (1 + 1 + train_cfg.n_negatives)  # anchor + positive + negatives
    
    # Encoder activations per sample
    # For transformer: multiple layers with attention and FFN
    # Approximate: 12 layers * (4 * hidden_dim + seq_len * hidden_dim) per channel
    num_layers = 6  # MiniLM-L6-v2
    attention_activations = model_cfg.max_length * model_cfg.max_length * 12
    hidden_activations = num_layers * model_cfg.max_length * model_cfg.embedding_dim * 4  
    encoder_activations_per_sample = (
        model_cfg.num_channels * (attention_activations + hidden_activations)
    )
    
    # MoE activations per sample
    moe_input_dim = model_cfg.embedding_dim * model_cfg.num_channels
    moe_activations_per_sample = (
        moe_input_dim * 2 +  # input to MoE (duplicated for routing)
        model_cfg.moe_hidden_dim * model_cfg.top_k * 2 +  # intermediate activations (top_k experts)
        model_cfg.embedding_dim * 2  # output
    )
    
    total_activations_per_sample = encoder_activations_per_sample + moe_activations_per_sample
    
    # Total for all samples in batch
    total_activations = total_activations_per_sample * n_total_samples
    
    # With gradient checkpointing, we only keep ~25% of activations
    if train_cfg.gradient_checkpointing:
        total_activations *= 0.25
    
    # Convert to MB (4 bytes per float32, 2 bytes per float16)
    bytes_per_element = 2 if model_cfg.use_fp16 else 4
    activation_mb = (total_activations * bytes_per_element) / (1024 ** 2)
    
    return activation_mb


def estimate_batch_data_memory(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig
) -> float:
    '''
    Estimate memory for storing batch data (input_ids and attention_mask).
    '''
    batch_size = train_cfg.batch_size
    n_total_samples = batch_size * (1 + 1 + train_cfg.n_negatives)
    
    # Each sample has 4 channels, each with input_ids and attention_mask
    # input_ids: (max_length,) int64 = 8 bytes per element
    # attention_mask: (max_length,) int64 = 8 bytes per element
    elements_per_sample = model_cfg.num_channels * model_cfg.max_length * 2
    bytes_per_sample = elements_per_sample * 8  # int64
    
    total_mb = (bytes_per_sample * n_total_samples) / (1024 ** 2)
    
    return total_mb


def estimate_memory_usage(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig
) -> MemoryEstimate:
    '''
    Estimate total GPU memory usage for training.
    '''
    # Model parameters
    trainable_params = estimate_model_parameters(model_cfg)
    bytes_per_param = 2 if model_cfg.use_fp16 else 4
    model_params_mb = (trainable_params * bytes_per_param) / (1024 ** 2)
    
    # Optimizer state (AdamW stores 2 states per parameter: momentum and variance)
    # Optimizer states are always in float32
    optimizer_state_mb = (trainable_params * 4 * 2) / (1024 ** 2)
    
    # Gradients (same size as parameters)
    gradients_mb = model_params_mb
    
    # Activations
    activations_mb = estimate_activation_memory(model_cfg, train_cfg)
    
    # Batch data
    batch_data_mb = estimate_batch_data_memory(model_cfg, train_cfg)
    
    # Overhead (CUDA context, PyTorch overhead, etc.)
    overhead_mb = 1500  # ~1.5 GB base overhead
    
    # Total
    total_mb = (
        model_params_mb + 
        optimizer_state_mb + 
        gradients_mb + 
        activations_mb + 
        batch_data_mb + 
        overhead_mb
    )
    
    return MemoryEstimate(
        model_params_mb=model_params_mb,
        optimizer_state_mb=optimizer_state_mb,
        gradients_mb=gradients_mb,
        activations_mb=activations_mb,
        batch_data_mb=batch_data_mb,
        overhead_mb=overhead_mb,
        total_mb=total_mb
    )


# -------------------------------------------------------------------------------------------------
# GPU detection and configuration optimization
# -------------------------------------------------------------------------------------------------

def detect_gpu_memory() -> Optional[float]:
    '''Detect available GPU memory in GB.'''
    if not torch.cuda.is_available():
        logger.error('No CUDA-capable GPU detected')
        return None
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    total_gb = total_memory / (1024 ** 3)
    
    gpu_name = torch.cuda.get_device_name(device)
    logger.info(f'Detected GPU: {gpu_name}')
    logger.info(f'Total Memory: {total_gb:.2f} GB')
    
    return total_gb


def find_optimal_batch_size(
    available_memory_gb: float,
    model_cfg: ModelConfig,
    n_positives: int,
    n_negatives: int,
    safety_margin: float = 0.85
) -> Tuple[int, MemoryEstimate]:
    '''
    Binary search to find optimal batch size that fits in memory.
    
    Args:
        available_memory_gb: Available GPU memory in GB
        model_cfg: Model configuration
        n_positives: Number of positives per anchor
        n_negatives: Number of negatives per anchor
        safety_margin: Use only this fraction of memory (default 0.85 = 85%)
    
    Returns:
        Optimal batch size and memory estimate
    '''
    target_memory_mb = available_memory_gb * 1024 * safety_margin
    
    # Binary search for batch size
    min_batch = 1
    max_batch = 128
    best_batch = 1
    best_estimate = None
    
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        
        train_cfg = TrainingConfig(
            batch_size=mid_batch,
            n_positives=n_positives,
            n_negatives=n_negatives
        )
        
        estimate = estimate_memory_usage(model_cfg, train_cfg)
        
        if estimate.total_mb <= target_memory_mb:
            best_batch = mid_batch
            best_estimate = estimate
            min_batch = mid_batch + 1
        else:
            max_batch = mid_batch - 1
    
    if best_estimate is None:
        # Even batch_size=1 doesn't fit, use it anyway and return estimate
        train_cfg = TrainingConfig(
            batch_size=1,
            n_positives=n_positives,
            n_negatives=n_negatives
        )
        best_estimate = estimate_memory_usage(model_cfg, train_cfg)
    
    return best_batch, best_estimate


def suggest_configurations(
    available_memory_gb: float,
    model_cfg: ModelConfig,
    target_effective_batch: int = 256
) -> List[Dict]:
    '''
    Suggest multiple training configurations optimized for the available GPU memory.
    
    Strategies:
    1. Balanced: moderate batch size with moderate accumulation
    2. High throughput: larger batch size, less accumulation
    3. Memory efficient: small batch size, high accumulation
    '''
    suggestions = []
    
    # Define different curriculum stage profiles
    stage_profiles = [
        {'name': '01_stage (Early)', 'n_positives': 32, 'n_negatives': 24},
        {'name': '02-05_stage (Later)', 'n_positives': 16, 'n_negatives': 8},
    ]
    
    for profile in stage_profiles:
        n_pos = profile['n_positives']
        n_neg = profile['n_negatives']
        
        # Find optimal batch size for this configuration
        batch_size, estimate = find_optimal_batch_size(
            available_memory_gb,
            model_cfg,
            n_pos,
            n_neg
        )
        
        # Calculate accumulation steps to reach target effective batch
        accumulate_grad_batches = max(1, target_effective_batch // batch_size)
        effective_batch = batch_size * accumulate_grad_batches
        
        suggestions.append({
            'stage': profile['name'],
            'batch_size': batch_size,
            'n_positives': n_pos,
            'n_negatives': n_neg,
            'accumulate_grad_batches': accumulate_grad_batches,
            'effective_batch_size': effective_batch,
            'memory_estimate': estimate,
            'memory_utilization': f'{(estimate.total_mb / (available_memory_gb * 1024)) * 100:.1f}%'
        })
    
    return suggestions


# -------------------------------------------------------------------------------------------------
# Configuration file generation
# -------------------------------------------------------------------------------------------------

def load_current_config(config_path: str = './conf/config.yaml') -> Dict:
    '''Load current configuration file.'''
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def update_config_file(
    config_path: str,
    updates: Dict,
    backup: bool = True
) -> None:
    '''Update configuration file with new values.'''
    config_path = Path(config_path)
    
    if backup and config_path.exists():
        backup_path = config_path.with_suffix('.yaml.backup')
        import shutil
        shutil.copy(config_path, backup_path)
        logger.info(f'Backup created: {backup_path}')
    
    # Load current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update nested config
    for key_path, value in updates.items():
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f'Configuration updated: {config_path}')


def update_curriculum_file(
    curriculum_path: str,
    n_positives: int,
    n_negatives: int,
    backup: bool = True
) -> None:
    '''Update curriculum stage file with new sample counts.'''
    curriculum_path = Path(curriculum_path)
    
    if backup and curriculum_path.exists():
        backup_path = curriculum_path.with_suffix('.yaml.backup')
        import shutil
        shutil.copy(curriculum_path, backup_path)
    
    with open(curriculum_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['n_positives'] = n_positives
    config['n_negatives'] = n_negatives
    
    with open(curriculum_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f'Curriculum updated: {curriculum_path}')


# -------------------------------------------------------------------------------------------------
# Memory profiling
# -------------------------------------------------------------------------------------------------

def profile_memory_usage(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig
) -> Dict:
    '''
    Profile actual memory usage by running a dummy training step.
    Requires GPU to be available.
    '''
    if not torch.cuda.is_available():
        logger.error('GPU required for memory profiling')
        return {}
    
    logger.info('Running memory profiling (this may take a minute)...')
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    try:
        # Import here to avoid issues if dependencies not available
        from naics_embedder.model.encoder import MultiChannelEncoder
        from naics_embedder.model.loss import HyperbolicInfoNCELoss
        
        device = torch.device('cuda')
        
        # Create model
        model = MultiChannelEncoder(
            base_model_name=model_cfg.base_model_name,
            lora_r=model_cfg.lora_r,
            num_experts=model_cfg.num_experts,
            moe_hidden_dim=model_cfg.moe_hidden_dim,
            use_gradient_checkpointing=train_cfg.gradient_checkpointing
        ).to(device)
        
        if model_cfg.use_fp16:
            model = model.half()
        
        # Create loss function
        loss_fn = HyperbolicInfoNCELoss(
            embedding_dim=model_cfg.embedding_dim
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        batch_size = train_cfg.batch_size
        channels = ['title', 'description', 'excluded', 'examples']
        
        def create_dummy_inputs():
            return {
                channel: {
                    'input_ids': torch.randint(
                        0, 30000, 
                        (batch_size, model_cfg.max_length),
                        device=device
                    ),
                    'attention_mask': torch.ones(
                        (batch_size, model_cfg.max_length),
                        dtype=torch.long,
                        device=device
                    )
                }
                for channel in channels
            }
        
        anchor_inputs = create_dummy_inputs()
        positive_inputs = create_dummy_inputs()
        
        # Negatives batch
        neg_batch_size = batch_size * train_cfg.n_negatives
        negative_inputs = {
            channel: {
                'input_ids': torch.randint(
                    0, 30000,
                    (neg_batch_size, model_cfg.max_length),
                    device=device
                ),
                'attention_mask': torch.ones(
                    (neg_batch_size, model_cfg.max_length),
                    dtype=torch.long,
                    device=device
                )
            }
            for channel in channels
        }
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=model_cfg.use_fp16):
            anchor_out = model(anchor_inputs)
            positive_out = model(positive_inputs)
            negative_out = model(negative_inputs)
            
            loss = loss_fn(
                anchor_out['embedding'],
                positive_out['embedding'],
                negative_out['embedding'],
                batch_size,
                train_cfg.n_negatives
            )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        profile_results = {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_mb': peak,
            'success': True
        }
        
        logger.info(f'  Allocated: {allocated:.1f} MB')
        logger.info(f'  Reserved:  {reserved:.1f} MB')
        logger.info(f'  Peak:      {peak:.1f} MB')
        
        return profile_results
        
    except Exception as e:
        logger.error(f'Memory profiling failed: {e}')
        return {'success': False, 'error': str(e)}
    
    finally:
        torch.cuda.empty_cache()


# -------------------------------------------------------------------------------------------------
# CLI interface
# -------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Optimize training configuration for available GPU memory'
    )
    parser.add_argument(
        '--gpu-memory',
        type=float,
        help='GPU memory in GB (e.g., 24 for RTX 6000, 80 for A100)'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-detect GPU memory'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Run actual memory profiling (requires GPU)'
    )
    parser.add_argument(
        '--target-effective-batch',
        type=int,
        default=256,
        help='Target effective batch size (batch_size * accumulate_grad_batches)'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply suggested configuration to config files'
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default='./conf/config.yaml',
        help='Path to main config file'
    )
    
    args = parser.parse_args()
    
    # Determine available memory
    if args.auto:
        gpu_memory_gb = detect_gpu_memory()
        if gpu_memory_gb is None:
            sys.exit(1)
    elif args.gpu_memory:
        gpu_memory_gb = args.gpu_memory
        logger.info(f'Using specified GPU memory: {gpu_memory_gb:.2f} GB')
    else:
        parser.error('Must specify either --gpu-memory or --auto')
    
    logger.info('')
    logger.info('=' * 80)
    logger.info('GPU Memory Configuration Optimizer')
    logger.info('=' * 80)
    
    # Model configuration
    model_cfg = ModelConfig()
    logger.info('\nModel Configuration:')
    logger.info(f'  Base Model: {model_cfg.base_model_name}')
    logger.info(f'  Embedding Dim: {model_cfg.embedding_dim}')
    logger.info(f'  Channels: {model_cfg.num_channels}')
    logger.info(f'  LoRA Rank: {model_cfg.lora_r}')
    logger.info(f'  MoE Experts: {model_cfg.num_experts}')
    logger.info(f'  FP16: {model_cfg.use_fp16}')
    
    # Generate suggestions
    logger.info(f'\nOptimizing for {gpu_memory_gb:.1f} GB GPU memory...')
    logger.info(f'Target effective batch size: {args.target_effective_batch}')
    
    suggestions = suggest_configurations(
        gpu_memory_gb,
        model_cfg,
        args.target_effective_batch
    )
    
    # Display suggestions
    logger.info('\n' + '=' * 80)
    logger.info('RECOMMENDED CONFIGURATIONS')
    logger.info('=' * 80)
    
    for i, config in enumerate(suggestions, 1):
        logger.info(f"\n{'─' * 80}")
        logger.info(f"Configuration {i}: {config['stage']}")
        logger.info(f"{'─' * 80}")
        logger.info(f"  batch_size:              {config['batch_size']}")
        logger.info(f"  n_positives:             {config['n_positives']}")
        logger.info(f"  n_negatives:             {config['n_negatives']}")
        logger.info(f"  accumulate_grad_batches: {config['accumulate_grad_batches']}")
        logger.info(f"  Effective batch size:    {config['effective_batch_size']}")
        logger.info(f"  Memory utilization:      {config['memory_utilization']}")
        logger.info(config['memory_estimate'])
    
    # Memory profiling
    if args.profile:
        logger.info('\n' + '=' * 80)
        logger.info('MEMORY PROFILING')
        logger.info('=' * 80)
        
        # Use the first suggested config
        best_config = suggestions[0]
        train_cfg = TrainingConfig(
            batch_size=best_config['batch_size'],
            n_positives=best_config['n_positives'],
            n_negatives=best_config['n_negatives'],
            accumulate_grad_batches=best_config['accumulate_grad_batches']
        )
        
        profile_results = profile_memory_usage(model_cfg, train_cfg)
        
        if profile_results.get('success'):
            estimated = best_config['memory_estimate'].total_mb
            actual = profile_results['peak_mb']
            error = abs(actual - estimated) / actual * 100
            
            logger.info('\nEstimate vs Actual:')
            logger.info(f'  Estimated: {estimated:.1f} MB')
            logger.info(f'  Actual:    {actual:.1f} MB')
            logger.info(f'  Error:     {error:.1f}%')
    
    # Apply configuration
    if args.apply:
        logger.info('\n' + '=' * 80)
        logger.info('APPLYING CONFIGURATION')
        logger.info('=' * 80)
        
        # Use the first suggested config as the default
        config = suggestions[0]
        
        # Update main config file
        updates = {
            'data_loader.batch_size': config['batch_size'],
            'training.trainer.accumulate_grad_batches': config['accumulate_grad_batches']
        }
        
        try:
            update_config_file(args.config_path, updates, backup=True)
            
            # Update curriculum files
            conf_dir = Path(args.config_path).parent
            curriculum_dir = conf_dir / 'text_curriculum'
            
            # Update stage 01
            stage_01_path = curriculum_dir / '01_stage.yaml'
            if stage_01_path.exists():
                update_curriculum_file(
                    str(stage_01_path),
                    n_positives=32,
                    n_negatives=24,
                    backup=True
                )
            
            # Update stages 02-05 (if different config recommended)
            if len(suggestions) > 1:
                later_config = suggestions[1]
                for stage_num in ['02', '03', '04', '05']:
                    stage_path = curriculum_dir / f'{stage_num}_stage.yaml'
                    if stage_path.exists():
                        update_curriculum_file(
                            str(stage_path),
                            n_positives=later_config['n_positives'],
                            n_negatives=later_config['n_negatives'],
                            backup=True
                        )
            
            logger.info('\n✓ Configuration files updated successfully!')
            logger.info('  Backup files created with .backup extension')
            
        except Exception as e:
            logger.error(f'\n✗ Failed to update configuration: {e}')
            sys.exit(1)
    
    logger.info('\n' + '=' * 80)
    logger.info('SUMMARY')
    logger.info('=' * 80)
    logger.info(f'GPU Memory: {gpu_memory_gb:.1f} GB')
    logger.info(f"Recommended batch_size: {suggestions[0]['batch_size']}")
    logger.info(f"Recommended accumulate_grad_batches: {suggestions[0]['accumulate_grad_batches']}")
    logger.info(f"Effective batch size: {suggestions[0]['effective_batch_size']}")
    
    if not args.apply:
        logger.info('\nTo apply these settings, run with --apply flag:')
        logger.info(
            '  python scripts/optimize_gpu_config.py'
            f'--gpu-memory {gpu_memory_gb:.0f} --apply'
        )
    
    logger.info('')


if __name__ == '__main__':
    main()
