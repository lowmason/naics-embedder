# Training Utilities

Helper functions for training orchestration and result collection.

## Overview

The `training` utilities module provides reusable components for the training
workflow, including hardware detection, configuration parsing, checkpoint
management, and summary artifact generation.

## Usage

```python
from pathlib import Path

from naics_embedder.utils.training import (
    detect_hardware,
    parse_config_overrides,
    resolve_checkpoint,
)

hardware = detect_hardware(log_info=True, cuda_precision='bf16-mixed')
overrides, invalid = parse_config_overrides([
    'training.learning_rate=1e-4',
    'data_loader.queries_per_step=128',
])
checkpoint_info = resolve_checkpoint('last', Path('checkpoints'), 'reference')
```

`TrainingResult.best_score` is outcome validation MRR, not an in-sample loss. The four guards
refuse used fresh directories, cross-directory resumes, different settings/seed, and a run
that early stopping ended. A spent epoch budget is a no-op. Exact resume uses `last`, with its
monitor and epoch-summary files. `refuse_other_constructor_settings` separately checks saved
LoRA rank/alpha/dropout and active MoE expert/routing/balancing controls against the current
config, refusing missing required values. Inactive MoE controls are ignored. `run_settings` records 21 settings including effective
accelerator/precision, accumulation, clipping and epoch budget.

## Data Classes

### HardwareInfo

Container for detected hardware configuration.

### CheckpointInfo

Resolved checkpoint path and metadata.

### TrainingResult

Structured result from a completed training run.

## API Reference

::: naics_embedder.utils.training
    options:
      show_source: false
      members:
        - HardwareInfo
        - CheckpointInfo
        - TrainingResult
        - detect_hardware
        - get_gpu_memory_info
        - parse_config_overrides
        - effective_precision
        - run_settings
        - resolve_checkpoint
        - read_checkpoint
        - refuse_a_fresh_start_into_a_used_directory
        - refuse_a_resume_from_another_directory
        - refuse_a_resume_under_other_settings
        - constructor_settings
        - refuse_other_constructor_settings
        - refuse_a_resume_of_a_stopped_run
        - outcome_checkpoint
        - outcome_early_stopping
        - create_trainer
        - save_training_summary
