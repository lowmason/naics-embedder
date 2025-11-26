# Training Guide

This guide explains how to train the NAICS Hyperbolic Embedding System using the **Structure-Aware Dynamic Curriculum (SADC)**. The SADC scheduler automatically advances through curriculum phases—no manual stage lists or chain files are required.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dynamic Curriculum Overview](#dynamic-curriculum-overview)
3. [Running Training](#running-training)
4. [Checkpointing and Resumption](#checkpointing-and-resumption)
5. [Configuration Files](#configuration-files)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Train with default settings

```bash
uv run naics-embedder train
```

### Resume the most recent run

```bash
uv run naics-embedder train --ckpt-path last
```

### Apply overrides without editing config files

```bash
uv run naics-embedder train \
  training.learning_rate=5e-5 \
  training.trainer.max_epochs=20
```

---

## Dynamic Curriculum Overview

The SADC scheduler controls sampling and mining behavior across the full run. It uses percentage-based phase boundaries derived from the configured `training.trainer.max_epochs`:

1. **Phase 1 (0–30%): Structural Initialization**
   - Masks sibling codes and weights negatives by inverse tree distance.
   - Emphasizes structural signals before mining harder negatives.
2. **Phase 2 (30–70%): Geometric Refinement**
   - Enables Lorentzian hard-negative mining.
   - Activates router-guided sampling for the Mixture-of-Experts encoder.
3. **Phase 3 (70–100%): False Negative Mitigation**
   - Turns on clustering-based false-negative elimination (FNE).

Phase transitions are fully automatic; you do not need to enumerate stages or chains. Adjusting `training.trainer.max_epochs` changes when each phase ends, and model hyperparameters such as `tree_distance_alpha` can be overridden to tune structural weighting early in training.

---

## Running Training

```bash
uv run naics-embedder train [OPTIONS] [OVERRIDES...]
```

**Key options**
- `--config PATH` — Path to the base config YAML file (default: `conf/config.yaml`).
- `--ckpt-path PATH` — Checkpoint to resume from, or `"last"` to auto-detect the latest checkpoint.
- `--skip-validation` — Skip pre-flight checks of data files and tokenization caches.
- `OVERRIDES...` — Hydra-style overrides for any config value (e.g., `training.learning_rate=1e-4 data_loader.batch_size=8`).

**Example workflows**
- Run with validation disabled (useful for repeated experiments):
  ```bash
  uv run naics-embedder train --skip-validation
  ```
- Emphasize structural weighting by increasing `tree_distance_alpha` and extend training:
  ```bash
  uv run naics-embedder train \
    training.trainer.max_epochs=24 \
    model.tree_distance_alpha=2.0
  ```
- Resume from an explicit checkpoint path:
  ```bash
  uv run naics-embedder train --ckpt-path checkpoints/latest_run/last.ckpt
  ```

> **Legacy notice:** Static stage lists (`train-seq`, `train-curriculum`, chain configs) are deprecated and hidden. Use the dynamic `train` command instead.

---

## Checkpointing and Resumption

- **Automatic saving:** Best and last checkpoints are written under `checkpoints/<experiment_name>/` based on the experiment name in the config.
- **Resume detection:** `--ckpt-path last` finds the newest `last.ckpt` for the configured experiment. Provide an explicit path to resume from another run.
- **Cross-run initialization:** When resuming from a different experiment, the model weights are loaded but training restarts fresh for the current experiment name.

---

## Configuration Files

### Base configuration (`conf/config.yaml`)

The base config defines data paths, tokenizer settings, model hyperparameters (LoRA, MoE, curvature, adaptive margins), and trainer parameters. Override any field inline using the `OVERRIDES...` syntax shown above.

### Data dependencies

Training expects the parquet artifacts produced by the data pipeline:
- `data/naics_descriptions.parquet`
- `data/naics_training_pairs` (streaming dataset)
- `data/naics_distances.parquet` and `data/naics_distance_matrix.parquet` for structural cues

---

## Troubleshooting

### Checkpoint not found
- Verify the file path or use `--ckpt-path last` to auto-detect the latest checkpoint.
- Confirm the experiment name matches the directory under `checkpoints/`.

### Out of memory
- Lower `data_loader.batch_size` or increase `training.trainer.accumulate_grad_batches`.
- Reduce negatives per anchor via config overrides if memory pressure persists.

### Slow phase transitions
- Shorten `training.trainer.max_epochs` to reach later SADC phases sooner.
- Alternatively, raise `training.trainer.max_epochs` to spend more time in early phases when you need additional structural warmup.
