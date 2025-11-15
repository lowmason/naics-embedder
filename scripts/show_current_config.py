#!/usr/bin/env python3
"""
Display current training configuration and estimated memory usage.

Usage:
    python scripts/show_current_config.py
"""

import yaml
from pathlib import Path
import sys

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from optimize_gpu_config import (
    ModelConfig,
    TrainingConfig,
    estimate_memory_usage,
    detect_gpu_memory
)


def load_config(config_path: str = "./conf/config.yaml"):
    """Load main configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_curriculum(curriculum_path: str):
    """Load curriculum stage file."""
    with open(curriculum_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("=" * 80)
    print("Current Training Configuration")
    print("=" * 80)
    
    # Load configurations
    config = load_config()
    
    batch_size = config['data_loader']['batch_size']
    accumulate = config['training']['trainer']['accumulate_grad_batches']
    
    print(f"\nMain Configuration (conf/config.yaml):")
    print(f"  batch_size:              {batch_size}")
    print(f"  accumulate_grad_batches: {accumulate}")
    print(f"  Effective batch size:    {batch_size * accumulate}")
    print(f"  num_workers:             {config['data_loader']['num_workers']}")
    print(f"  precision:               {config['training']['trainer']['precision']}")
    print(f"  max_epochs:              {config['training']['trainer']['max_epochs']}")
    
    # Check curriculum stages
    print("\nCurriculum Stages:")
    conf_dir = Path("./conf")
    curriculum_dir = conf_dir / "curriculum"
    
    stages = []
    for stage_file in sorted(curriculum_dir.glob("*_stage.yaml")):
        curriculum = load_curriculum(stage_file)
        stages.append({
            'name': stage_file.stem,
            'n_positives': curriculum.get('n_positives', 'N/A'),
            'n_negatives': curriculum.get('n_negatives', 'N/A'),
        })
    
    for stage in stages:
        print(f"  {stage['name']:15s}: n_pos={stage['n_positives']:>3}, "
              f"n_neg={stage['n_negatives']:>3}")
    
    # Estimate memory for each stage
    print("\n" + "=" * 80)
    print("Memory Estimates")
    print("=" * 80)
    
    model_cfg = ModelConfig()
    
    # Get unique stage configurations
    unique_configs = {}
    for stage in stages:
        key = (stage['n_positives'], stage['n_negatives'])
        if key not in unique_configs:
            unique_configs[key] = stage['name']
    
    for (n_pos, n_neg), stage_name in unique_configs.items():
        if n_pos == 'N/A' or n_neg == 'N/A':
            continue
            
        train_cfg = TrainingConfig(
            batch_size=batch_size,
            n_positives=n_pos,
            n_negatives=n_neg,
            accumulate_grad_batches=accumulate
        )
        
        memory = estimate_memory_usage(model_cfg, train_cfg)
        
        print(f"\nConfiguration: n_pos={n_pos}, n_neg={n_neg} ({stage_name})")
        print(f"  Batch samples:  {batch_size}")
        print(f"  Total samples:  {batch_size * (1 + 1 + n_neg)} "
              f"(anchor + positive + {n_neg} negatives)")
        print(f"  Estimated GPU:  {memory.total_mb/1024:.2f} GB")
        print(f"    - Model:      {memory.model_params_mb:.1f} MB")
        print(f"    - Optimizer:  {memory.optimizer_state_mb:.1f} MB")
        print(f"    - Gradients:  {memory.gradients_mb:.1f} MB")
        print(f"    - Activations:{memory.activations_mb:>7.1f} MB")
        print(f"    - Batch Data: {memory.batch_data_mb:.1f} MB")
        print(f"    - Overhead:   {memory.overhead_mb:.1f} MB")
    
    # Detect GPU if available
    print("\n" + "=" * 80)
    print("GPU Information")
    print("=" * 80)
    
    gpu_memory = detect_gpu_memory()
    if gpu_memory:
        # Calculate utilization for the most memory-intensive stage
        max_memory = max(
            estimate_memory_usage(
                model_cfg,
                TrainingConfig(
                    batch_size=batch_size,
                    n_positives=stage['n_positives'],
                    n_negatives=stage['n_negatives'],
                    accumulate_grad_batches=accumulate
                )
            ).total_mb / 1024
            for stage in stages
            if stage['n_positives'] != 'N/A'
        )
        
        utilization = (max_memory / gpu_memory) * 100
        
        print(f"\nPeak Memory Usage (worst case):")
        print(f"  Estimated: {max_memory:.2f} GB")
        print(f"  Available: {gpu_memory:.2f} GB")
        print(f"  Utilization: {utilization:.1f}%")
        
        if utilization > 95:
            print("\n⚠️  WARNING: Memory usage is very high!")
            print("   Consider reducing batch_size or n_negatives")
        elif utilization > 85:
            print("\n✓ Good: Memory utilization is in optimal range (85-95%)")
        elif utilization < 60:
            print("\n💡 TIP: GPU is underutilized, consider:")
            print("   - Increasing batch_size")
            print("   - Increasing n_positives or n_negatives")
            print("   - Run: python scripts/optimize_gpu_config.py --auto")
        else:
            print("\n✓ Memory utilization looks reasonable")
    
    print("\n" + "=" * 80)
    print("\nTo optimize configuration for your GPU:")
    print("  python scripts/optimize_gpu_config.py --auto")
    print("\nTo apply suggested changes:")
    print("  python scripts/optimize_gpu_config.py --auto --apply")
    print()


if __name__ == '__main__':
    main()
