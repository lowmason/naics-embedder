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

### `data regressor-groups`

Draw the regressor panel's held-out four-digit groups, once: a fifth of each sector's groups
(largest remainder, seeded) from the 980 six-digit codes Stage 1's finding names. The QCEW
national slices are read from `qcew_dir` under the sha256 values pinned in
`conf/data/regressor_panel.yaml`, and the codebook under its codes' fingerprint. The table is
committed, and the panel reads it only under the fingerprint pinned there as
`heldout_groups_sha256`. A redraw moves both regressor outer sets, so an existing table is
replaced only with `--force`, and the panel reads a new draw only after a reviewed change of the
pin.

**Generates:** `conf/data/regressor_heldout_groups.csv`,
`conf/data/regressor_heldout_groups_provenance.json`

```bash
uv run naics-embedder data regressor-groups --codebook PATH/naics_codebook.parquet
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

### `tools text-only-table`

Embed every code's text with the arm's backbone, frozen: each of the four channels is mean-pooled
over its tokens, and a code's vector is the mean of its present channels (roadmap D9). The
backbone comes from the local Hugging Face cache (default: `text_only.backbone` in
`conf/data/regressor_panel.yaml`). The regressor panel reduces the table to the arm's dimension
by PCA.

**Generates:** the table and `<stem>_provenance.json` beside it

```bash
uv run naics-embedder tools text-only-table --descriptions data/naics_descriptions.parquet \
  --output /tmp/text_only.parquet
```

**Options:**
- `--descriptions PATH` - The arm's descriptions parquet: the text it reads
- `--output PATH` - Where to write the table
- `--backbone NAME` - The arm's backbone (default: the regressor config's)

### `tools regressor-panel`

Score an arm on the regressor panel (roadmap Stage 3). Every comparator (covariates alone, and
the arm's coordinates, one-hot, ancestor indicators and the text-only table, each alone and with
the covariates) is fitted by ridge on standardized features, with the penalty tuned by nested
grouped folds inside the remainder. The outcome is log employment in year t + 1 from year-t
features. The seen and held-out regimes are separate panels. The validation split reads only the
remainder. The test split opens each regime's sealed outer set first, and both the opening and
the read are logged. The output holds one prediction per row, comparator and repeat (the test
split has one repeat), with the row's outcome and the chosen penalty.

```bash
uv run naics-embedder tools regressor-panel --coordinates arm.parquet \
  --text-only /tmp/text_only.parquet --codebook PATH/naics_codebook.parquet
```

**Options:**
- `--coordinates PATH` - The arm's 2,125-code table in the export form (tangent coordinates at
  the origin for a hyperbolic arm). Lorentz points and constant columns, such as a log map's
  zero time coordinate, are refused
- `--text-only PATH` - The text-only table (`tools text-only-table`)
- `--codebook PATH` - A supervision bundle's `naics_codebook.parquet`
- `--regime seen|heldout` - Regime to score (repeatable; default: both)
- `--level INT` - NAICS level 2-6 (repeatable; default: 6)
- `--split validation|test` - The test split needs `--open-purpose`, which is logged
- `--purpose TEXT` - Why this read happens; recorded in the selection log (default:
  `regressor panel <split> read`)
- `--open-purpose TEXT`, `--reopen-reason TEXT` - Why the outer sets are opened, and why again
- `--log PATH` - Selection log (default: `logs/selection_log.jsonl`)
- `--output PATH` - Write the per-row predictions as parquet

### `tools margins`

Fix each panel's non-inferiority margin δ from a reference arm (Req 5): δ is `--multiple` times
the reference's across-seed standard deviation of the panel's decision statistic (roadmap D10:
per-query MRR on the outcome panel, and the `covariates+embedding` comparator's mean squared
error on each regressor regime at level 6). Fix the margins before any other arm of the decision
reads a panel: a decision refuses every run that read before its margins were fixed.

**Generates:** the margin record (JSON)

```bash
uv run naics-embedder tools margins --reference reference.json --multiple 0.5 \
  --name reference-margins --store ~/naics-artifacts --output margins.json
```

**Options:**
- `--reference PATH` - The reference configuration's arm record, written by the seed-sweep
  driver (`naics_embedder.decision.sweep.run_seed_sweep`)
- `--multiple FLOAT` - Each δ as a multiple of the reference's across-seed standard deviation
- `--name TEXT` - Names the margins in the decision records that use them
- `--store PATH` - The artifact store the arm record references
- `--output PATH` - Where to write the margin record; an existing file is never overwritten

### `tools decide`

Decide among two or more arms under Req 5's rule over the three panels of roadmap D8: the outcome
panel and the regressor panel's seen and held-out regimes at level 6. Each comparison reads the
difference Δ on paired resamples of each panel's units (codes with their queries; four-digit
groups), with each arm's seeds resampled inside every replicate. A is adopted over B when it is
non-inferior on all three panels (the 95 % interval's lower bound above −δ) and superior on at
least one (the 98⅓ % interval above zero). The survivors are the arms no other arm is adopted
over, and the tie order picks among them: fewer components, then lower dimension, then
non-hyperbolic geometry, then the higher held-out gain over ancestors (roadmap D11). The number
of replicates, the bootstrap seed and the seed floor come from `conf/data/decision.yaml`.

**Generates:** the decision record (JSON): the arms with their selection-log records and artifact
references, the margins, every comparison with both intervals, the non-dominated set, the tie
order, the chosen arm, and each arm's reported gains over the sparse comparators

```bash
uv run naics-embedder tools decide --arm candidate.json --arm reference.json \
  --margins margins.json --name dimension-8 --question "Is dimension 8 enough?" \
  --store ~/naics-artifacts --output decision.json
```

**Options:**
- `--arm PATH` - An arm record; repeat for each arm
- `--margins PATH` - The margin record (`tools margins`)
- `--name TEXT`, `--question TEXT` - Name the decision and say what it settles
- `--store PATH` - The artifact store the arm records reference
- `--output PATH` - Where to write the decision record; an existing file is never overwritten

### `tools diagnostics`

Report Req 6's structural diagnostics over every codebook code: sector separation (an AUC),
within-sector rank correlation (over queries and over sectors), MAP over ancestors,
NDCG@5/10/20 with integer lowest-common-ancestor grades, the Pearson correlation of distance
with the tree metric D*, and parent retrieval@1/5 without the 522 unary pairs. The report
describes an arm: nothing selects on it, and no statistic in it has a threshold. To compare two
tables, such as one before and one after graph refinement, report on each.

```bash
uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic \
  --codebook PATH/naics_codebook.parquet
```

**Options:**
- `--table PATH` - The arm's code table in the export form (tangent coordinates at the origin
  for a hyperbolic arm; Lorentz points are refused)
- `--geometry euclidean|spherical|hyperbolic` - The arm's distance: Euclidean, cosine, or the
  geodesic distance after the exponential map at the origin
- `--codebook PATH` - A supervision bundle's `naics_codebook.parquet`; the table must hold
  exactly its codes
- `--curvature FLOAT` - A hyperbolic arm's curvature magnitude (default: 1.0)
- `--output PATH` - Also write the report as JSON

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

