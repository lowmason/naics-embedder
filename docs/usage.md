# CLI Usage Guide

This guide covers all available CLI commands for the NAICS Embedder system.

## Overview

The NAICS Embedder CLI is organized into three main command groups:

- **`data`** - Data generation and preprocessing commands
- **`tools`** - Utility tools for configuration, GPU optimization, and metrics
- **`train`** - Model training with the dynamic SADC curriculum

## Installation

The CLI is available as the `naics-embedder` command after installation:

```bash
uv run naics-embedder --help
```

---

## Data Commands

### `data preprocess`

Download and preprocess all raw NAICS data files.

**Generates:** `data/naics_descriptions.parquet`

```bash
uv run naics-embedder data preprocess
```

### `data relations`

Compute pairwise graph relationships between all NAICS codes.

**Requires:** `data/naics_descriptions.parquet`  
**Generates:** `data/naics_relations.parquet`

```bash
uv run naics-embedder data relations
```

### `data distances`

Compute pairwise graph distances between all NAICS codes.

**Requires:** `data/naics_descriptions.parquet`  
**Generates:** `data/naics_distances.parquet`

```bash
uv run naics-embedder data distances
```

### `data triplets`

Generate (anchor, positive, negative) training triplets.

**Requires:** 
- `data/naics_descriptions.parquet`
- `data/naics_distances.parquet`

**Generates:** `data/naics_training_pairs.parquet`

```bash
uv run naics-embedder data triplets
```

### `data all`

Run the full data generation pipeline: preprocess, distances, and triplets.

```bash
uv run naics-embedder data all
```

---

## Tools Commands

### `tools config`

Display current training and curriculum configuration.

```bash
uv run naics-embedder tools config
```

**Options:**
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)

```bash
uv run naics-embedder tools config --config conf/config.yaml
```

### `tools gpu`

Optimize training configuration for available GPU memory. Suggests optimal `batch_size` and `accumulate_grad_batches` based on your GPU.

```bash
# Auto-detect GPU memory
uv run naics-embedder tools gpu --auto

# Specify GPU memory manually
uv run naics-embedder tools gpu --gpu-memory 24

# Apply suggested configuration
uv run naics-embedder tools gpu --auto --apply
```

**Options:**
- `--gpu-memory FLOAT` - GPU memory in GB (e.g., 24 for RTX 6000, 80 for A100)
- `--auto` - Auto-detect GPU memory
- `--target-effective-batch INT` - Target effective batch size (default: 256)
- `--apply` - Apply suggested configuration to config files
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)

### `tools visualize`

Visualize training metrics from log files. Creates comprehensive visualizations and analysis of training metrics including:
- Hyperbolic radius over time
- Hierarchy preservation correlations
- Embedding diversity metrics

```bash
uv run naics-embedder tools visualize --stage 02_text
```

**Options:**
- `--stage, -s STR` - Stage name to filter (e.g., `02_text`, default: `02_text`)
- `--log-file PATH` - Path to log file (default: `logs/train_sequential.log`)
- `--output-dir PATH` - Output directory for plots (default: `outputs/visualizations/`)

### `tools investigate`

Investigate why hierarchy preservation correlations might be low. Analyzes ground truth distances, evaluation configuration, and provides recommendations.

```bash
uv run naics-embedder tools investigate
```

**Options:**
- `--distance-matrix PATH` - Path to ground truth distance matrix (default: `data/naics_distance_matrix.parquet`)
- `--config PATH` - Path to config file (default: `conf/config.yaml`)

---

## Training Commands

### `train`

Train the dynamic Structure-Aware Dynamic Curriculum (SADC) workflow. The SADC scheduler automatically progresses through structural initialization, geometric refinement, and false-negative mitigation phases—no manual stage list is required.

```bash
uv run naics-embedder train
```

**Options:**
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)
- `--ckpt-path PATH` - Path to checkpoint file to resume from, or `"last"` to auto-detect the latest checkpoint
- `--skip-validation` - Skip pre-flight data/cache validation
- `OVERRIDES...` - Config overrides (e.g., `training.learning_rate=1e-4 data_loader.batch_size=64`)

**Examples:**

```bash
# Train with default SADC settings
uv run naics-embedder train

# Resume from the latest checkpoint produced by SADC
uv run naics-embedder train --ckpt-path last

# Adjust scheduler behavior via overrides (example: longer Phase 1 and lower learning rate)
uv run naics-embedder train \
  training.trainer.max_epochs=20 \
  training.learning_rate=5e-5
```

> **Note:** The legacy `train-seq`/`train-curriculum` commands for static stage lists are hidden and unsupported. Use the dynamic `train` command instead.

---

## Common Workflows

### Complete Data Pipeline

```bash
# Generate all required data files
uv run naics-embedder data all
```

### Train with Dynamic SADC

```bash
# Launch training with automatic SADC phase scheduling
uv run naics-embedder train
```

### Resume Training

```bash
# Continue from the latest checkpoint in the experiment directory
uv run naics-embedder train --ckpt-path last
```

### View Configuration

```bash
# Display current configuration
uv run naics-embedder tools config
```

### Analyze Training Metrics

```bash
# Visualize training metrics
uv run naics-embedder tools visualize --stage 02_text

# Investigate hierarchy preservation issues
uv run naics-embedder tools investigate
```

---

## Getting Help

For help on any command, use the `--help` flag:

```bash
uv run naics-embedder --help
uv run naics-embedder data --help
uv run naics-embedder tools --help
uv run naics-embedder train --help
```

---

## Configuration Files

The CLI uses configuration files located in the `conf/` directory:

- **Base Config:** `conf/config.yaml` - Main training configuration
- **Text Curricula:** `conf/text_curriculum/*.yaml` - Text training curriculum stages
- **Graph Curricula:** `conf/graph_curriculum/*.yaml` - Graph training curriculum stages
- **Chain Configs:** `conf/text_curriculum/chain_text.yaml` - Sequential training chains

See the [Configuration Documentation](api/config.md) for details on configuration structure.

