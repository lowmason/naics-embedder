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

Download and preprocess all raw NAICS data files. Each code's examples channel holds only its
examples-role index entries, per the committed role table (see `data roles`); the code's other
entries are outcome-panel queries. Preprocessing fails if any validation or test query matches
training text.

**Requires:** `conf/data/index_roles.csv`  
**Generates:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`

```bash
uv run naics-embedder data preprocess
```

**Options:**
- `--source-dir PATH` - Read the four Census files from this directory, by file name, instead of
  downloading them
- `--force` - Overwrite a descriptions file that a configured supervision bundle pins. Refused by
  default: training against that bundle fails closed once the file changes

### `data supervision`

Build one immutable, validated Stage-3 supervision bundle: the codebook, pair facts (structural
distance and relation plus directional explicit exclusions), legacy-compatible distance and
relation views and matrices, training pairs, and curriculum difficulty thresholds. The manifest is
written last and the bundle directory is published atomically.

**Requires:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet` (carried as the
bundle's optional `index_roles` member)  
**Generates:** `data/supervision/stage3-supervision-v1/<bundle-id>/` (prints
`Supervision manifest: <path>`; set `supervision.manifest_path` to it before training)

```bash
uv run naics-embedder data supervision
```

### `data relations`, `data distances`, `data triplets`

Deprecated stage commands. Publishing one partial authority would let artifacts from different
generations mix, so each prints a migration notice and builds the complete supervision bundle
(same as `data supervision`).

### `data all`

Run the full data generation pipeline: preprocess, then build the supervision bundle.

```bash
uv run naics-embedder data all
```

### `data roles`

Draw the frozen index-entry role table, once. Every Census index entry gets exactly one role, per
code and stratified: examples-channel text, or a training, validation or test query for the
outcome panel. No validation or test query matches any training text, exactly or as a
near-duplicate (character-trigram Jaccard of at least 0.9). The table is committed and `data
preprocess` applies it. Redrawing it unseals the validation and test splits, so an existing table
is replaced only with `--force`.

**Generates:** `conf/data/index_roles.csv`, `conf/data/index_roles_provenance.json`

```bash
uv run naics-embedder data roles --source-dir ~/Downloads/Data
```

---

## Tools Commands

### `tools config`

Display current training configuration, including the Structure-Aware Dynamic Curriculum (SADC) schedule.

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

### `tools outcome-baseline`

Score the training-free lexical encoder (hashed character trigrams under cosine distance) on the
outcome panel's validation split: top-1 accuracy, MRR, Hit@1/5/10 and the level of the lowest
common ancestor of the top-1 code and the truth. The read is appended to the selection log; the
test split stays sealed.

**Requires:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`

```bash
uv run naics-embedder tools outcome-baseline
```

**Options:**
- `--purpose TEXT` - Why this read happens; recorded in the selection log
- `--index-roles PATH`, `--descriptions PATH` - Preprocessing outputs (default: the paths in
  `conf/data/download.yaml`)
- `--log PATH` - Selection log (default: `logs/selection_log.jsonl`, from
  `conf/data/outcome_panel.yaml`)
- `--output PATH` - Also write the summary as JSON

---

## Training Commands

### `train`

Train the contrastive encoder with the Structure-Aware Dynamic Curriculum (SADC). The scheduler
drives phase transitions automatically—no curriculum files or chain configs are needed.

```bash
uv run naics-embedder train --config conf/config.yaml
```

**Options:**
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)
- `--ckpt-path PATH` - Path to checkpoint file to resume from, or `"last"` to auto-detect the latest checkpoint in the experiment directory
- `--skip-validation` - Skip pre-flight validation of data files and caches
- `OVERRIDES...` - Config overrides (e.g., `training.learning_rate=1e-4 data_loader.batch_size=64`)

**Examples:**

```bash
# Standard run with SADC
uv run naics-embedder train

# Resume from last checkpoint in the experiment
uv run naics-embedder train --ckpt-path last

# Apply overrides for learning rate and epochs
uv run naics-embedder train --config conf/config.yaml \
  training.learning_rate=1e-4 training.trainer.max_epochs=20
```

### `train-seq`

Deprecated sequential training workflow retained for legacy stage-chain jobs. Use SADC via
`train` for new runs; `train-seq` now requires `--legacy` to acknowledge deprecation.

```bash
uv run naics-embedder train-seq --legacy --num-stages 3
```

**Options:**
- `--num-stages, -n INT` - Number of sequential stages to run (default: 3)
- `--config PATH` - Path to base config YAML file (default: `conf/config.yaml`)
- `--resume` - Resume from last checkpoint if available
- `--legacy` - Required to continue using the deprecated workflow
- `OVERRIDES...` - Config overrides applied to every stage

**Examples:**

```bash
# Reproduce a historical 3-stage run
uv run naics-embedder train-seq --legacy --num-stages 3
```

---

## Common Workflows

### Complete Data Pipeline

```bash
# Generate all required data files
uv run naics-embedder data all
```

### Standard Training

```bash
# Train with the dynamic SADC scheduler
uv run naics-embedder train
```

### Dynamic SADC Training

```bash
# Legacy sequential flow (deprecated)
uv run naics-embedder train-seq --legacy --num-stages 3
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

The CLI reads a single configuration in `conf/config.yaml`:

- **Base Config:** Paths, model hyperparameters, and trainer settings
- **Curriculum:** `curriculum.*` fields configure SADC phase boundaries and false-negative elimination cadence

See the [Configuration Documentation](api/config.md) for details on configuration structure.

