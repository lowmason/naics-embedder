"""
GPU configuration optimization tools.

Provides functions to detect GPU memory and optimize training configuration
for available GPU resources.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


def detect_gpu_memory() -> Optional[float]:
    """
    Auto-detect GPU memory in GB.
    
    Returns:
        GPU memory in GB, or None if no GPU is available
    """
    if not torch.cuda.is_available():
        return None
    
    device = torch.cuda.current_device()
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    
    return total_memory_gb


def _estimate_memory_per_batch(
    batch_size: int,
    n_positives: int,
    n_negatives: int,
    embedding_dim: int = 768,
    max_length: int = 512
) -> float:
    """
    Estimate GPU memory usage per batch in GB.
    
    This is a rough estimate based on model size, batch size, and sequence length.
    """
    # Base model memory (approximate)
    base_model_memory_gb = 0.5  # ~500MB for base model + LoRA
    
    # Embedding memory: batch_size * (1 + n_positives + n_negatives) * embedding_dim * 4 bytes
    embedding_memory_gb = (
        batch_size * (1 + n_positives + n_negatives) * embedding_dim * 4
    ) / (1024 ** 3)
    
    # Tokenized input memory: batch_size * (1 + n_positives + n_negatives) * max_length * 4 bytes
    input_memory_gb = (
        batch_size * (1 + n_positives + n_negatives) * max_length * 4
    ) / (1024 ** 3)
    
    # Gradient memory (roughly 2x forward pass)
    gradient_memory_gb = (embedding_memory_gb + input_memory_gb) * 2
    
    # Total with overhead
    total_memory_gb = base_model_memory_gb + embedding_memory_gb + input_memory_gb + gradient_memory_gb
    
    # Add 20% overhead for PyTorch and system
    return total_memory_gb * 1.2


def _suggest_batch_config(
    gpu_memory_gb: float,
    n_positives: int,
    n_negatives: int,
    target_effective_batch: int
) -> Dict[str, int]:
    """
    Suggest batch_size and accumulate_grad_batches for given GPU memory.
    
    Returns:
        Dictionary with batch_size and accumulate_grad_batches
    """
    # Start with a conservative batch size
    batch_size = 4
    max_batch_size = 128
    
    # Find maximum batch size that fits in GPU memory
    while batch_size < max_batch_size:
        memory_needed = _estimate_memory_per_batch(
            batch_size + 4,  # Try next size
            n_positives,
            n_negatives
        )
        
        if memory_needed > gpu_memory_gb * 0.8:  # Use 80% of GPU memory max
            break
        
        batch_size += 4
    
    # Calculate accumulate_grad_batches to reach target effective batch
    effective_batch = batch_size
    accumulate_grad_batches = max(1, (target_effective_batch + batch_size - 1) // batch_size)
    
    return {
        'batch_size': batch_size,
        'accumulate_grad_batches': accumulate_grad_batches,
        'effective_batch_size': batch_size * accumulate_grad_batches
    }


def optimize_gpu_config(
    gpu_memory_gb: Optional[float] = None,
    auto_detect: bool = False,
    target_effective_batch: int = 256,
    apply: bool = False,
    config_path: str = 'conf/config.yaml'
) -> Dict:
    """
    Optimize training configuration for available GPU memory.
    
    Args:
        gpu_memory_gb: GPU memory in GB (if None and auto_detect=False, will raise error)
        auto_detect: If True, auto-detect GPU memory
        target_effective_batch: Target effective batch size (batch_size * accumulate_grad_batches)
        apply: If True, apply suggested configuration to config files
        config_path: Path to base config YAML file
    
    Returns:
        Dictionary with:
        - gpu_memory_gb: Detected/specified GPU memory
        - suggestions: List of configuration suggestions per curriculum stage
        - applied: Whether configurations were applied
    """
    # Detect or use provided GPU memory
    if auto_detect:
        detected_memory = detect_gpu_memory()
        if detected_memory is None:
            raise RuntimeError('No GPU detected. Cannot auto-detect GPU memory.')
        gpu_memory_gb = detected_memory
    elif gpu_memory_gb is None:
        raise ValueError('Must provide gpu_memory_gb or set auto_detect=True')
    
    logger.info(f'Optimizing for GPU memory: {gpu_memory_gb:.1f} GB')
    
    # Load base config to get curriculum stages
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    
    with open(config_file, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Find curriculum stages
    curriculum_dir = config_file.parent / 'text_curriculum'
    if not curriculum_dir.exists():
        curriculum_dir = config_file.parent / 'graph_curriculum'
    
    suggestions = []
    applied = False
    
    if curriculum_dir.exists():
        # Get all curriculum stage files
        stage_files = sorted(curriculum_dir.glob('*_text.yaml')) or sorted(curriculum_dir.glob('*_graph.yaml'))
        
        for stage_file in stage_files:
            stage_name = stage_file.stem
            
            # Load curriculum config
            with open(stage_file, 'r') as f:
                stage_config = yaml.safe_load(f) or {}
            
            # Get n_positives and n_negatives from curriculum
            n_positives = stage_config.get('n_positives', 2048)
            n_negatives = stage_config.get('n_negatives', 32)
            
            # Suggest batch configuration
            batch_config = _suggest_batch_config(
                gpu_memory_gb,
                n_positives,
                n_negatives,
                target_effective_batch
            )
            
            # Estimate memory usage
            memory_estimate = _estimate_memory_per_batch(
                batch_config['batch_size'],
                n_positives,
                n_negatives
            )
            memory_utilization = (memory_estimate / gpu_memory_gb) * 100
            
            suggestion = {
                'stage': stage_name,
                'batch_size': batch_config['batch_size'],
                'n_positives': n_positives,
                'n_negatives': n_negatives,
                'accumulate_grad_batches': batch_config['accumulate_grad_batches'],
                'effective_batch_size': batch_config['effective_batch_size'],
                'memory_utilization': memory_utilization,
                'memory_estimate': f'Estimated: {memory_estimate:.2f} GB / {gpu_memory_gb:.1f} GB ({memory_utilization:.1f}%)'
            }
            
            suggestions.append(suggestion)
            
            # Apply if requested
            if apply:
                # Backup original file
                backup_path = stage_file.with_suffix('.yaml.backup')
                if not backup_path.exists():
                    shutil.copy2(stage_file, backup_path)
                    logger.info(f'Backed up {stage_file} to {backup_path}')
                
                # Update config file
                stage_config['batch_size'] = batch_config['batch_size']
                stage_config['accumulate_grad_batches'] = batch_config['accumulate_grad_batches']
                
                with open(stage_file, 'w') as f:
                    yaml.dump(stage_config, f, default_flow_style=False, sort_keys=False)
                
                logger.info(f'Updated {stage_file} with suggested configuration')
                applied = True
    
    # Also suggest for base config if no curriculum stages found
    if not suggestions:
        # Use defaults
        n_positives = 2048
        n_negatives = 32
        
        batch_config = _suggest_batch_config(
            gpu_memory_gb,
            n_positives,
            n_negatives,
            target_effective_batch
        )
        
        memory_estimate = _estimate_memory_per_batch(
            batch_config['batch_size'],
            n_positives,
            n_negatives
        )
        memory_utilization = (memory_estimate / gpu_memory_gb) * 100
        
        suggestion = {
            'stage': 'base_config',
            'batch_size': batch_config['batch_size'],
            'n_positives': n_positives,
            'n_negatives': n_negatives,
            'accumulate_grad_batches': batch_config['accumulate_grad_batches'],
            'effective_batch_size': batch_config['effective_batch_size'],
            'memory_utilization': memory_utilization,
            'memory_estimate': f'Estimated: {memory_estimate:.2f} GB / {gpu_memory_gb:.1f} GB ({memory_utilization:.1f}%)'
        }
        
        suggestions.append(suggestion)
        
        if apply:
            # Backup and update base config
            backup_path = config_file.with_suffix('.yaml.backup')
            if not backup_path.exists():
                shutil.copy2(config_file, backup_path)
                logger.info(f'Backed up {config_file} to {backup_path}')
            
            # Update data_loader section
            if 'data_loader' not in base_config:
                base_config['data_loader'] = {}
            
            base_config['data_loader']['batch_size'] = batch_config['batch_size']
            
            if 'training' not in base_config:
                base_config['training'] = {}
            if 'trainer' not in base_config['training']:
                base_config['training']['trainer'] = {}
            
            base_config['training']['trainer']['accumulate_grad_batches'] = batch_config['accumulate_grad_batches']
            
            with open(config_file, 'w') as f:
                yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f'Updated {config_file} with suggested configuration')
            applied = True
    
    return {
        'gpu_memory_gb': gpu_memory_gb,
        'suggestions': suggestions,
        'applied': applied
    }


