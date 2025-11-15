#!/usr/bin/env python3
"""
Quick example script demonstrating how to use the GPU optimizer programmatically.
"""

import sys
sys.path.insert(0, '/erpds/naics-embedder/scripts')

from optimize_gpu_config import (
    ModelConfig,
    TrainingConfig,
    estimate_memory_usage,
    suggest_configurations
)

# Example 1: Estimate memory for a specific configuration
print("Example 1: Memory estimation for custom config")
print("=" * 70)

model_cfg = ModelConfig()
train_cfg = TrainingConfig(
    batch_size=32,
    n_positives=16,
    n_negatives=8,
    accumulate_grad_batches=8
)

memory = estimate_memory_usage(model_cfg, train_cfg)
print(memory)
print(f"\nEffective batch size: {train_cfg.batch_size * train_cfg.accumulate_grad_batches}")
print()

# Example 2: Get suggestions for different GPU sizes
print("\nExample 2: Optimal configs for different GPUs")
print("=" * 70)

gpu_configs = [
    ("RTX 3090", 24),
    ("A100", 40),
    ("A100", 80),
]

for gpu_name, gpu_memory_gb in gpu_configs:
    print(f"\n{gpu_name} ({gpu_memory_gb} GB):")
    print("-" * 70)
    
    suggestions = suggest_configurations(
        gpu_memory_gb,
        ModelConfig(),
        target_effective_batch=256
    )
    
    for config in suggestions:
        print(f"  {config['stage']}:")
        print(f"    batch_size={config['batch_size']}, "
              f"accumulate={config['accumulate_grad_batches']}, "
              f"effective={config['effective_batch_size']}")
        print(f"    n_pos={config['n_positives']}, n_neg={config['n_negatives']}")
        print(f"    Memory: {config['memory_estimate'].total_mb/1024:.1f} GB "
              f"({config['memory_utilization']})")

print("\n" + "=" * 70)
print("Done!")
