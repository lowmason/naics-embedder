# Quickstart Guide

Install the project with `uv`, then prepare one immutable supervision bundle. The commands
below are operator examples; training and panel reads require the generated artifacts.

## Installation

```bash
git clone https://github.com/lowmason/naics-embedder.git
cd naics-embedder
uv sync
uv run naics-embedder --help
```

`.python-version` pins Python 3.12. Use the locked environment for training and verification.

## Prepare Training Data

```bash
uv run naics-embedder data all
```

This preprocesses NAICS text and builds the `stage3-supervision-v2` bundle. It prints
`Supervision manifest: <path>`. Set that exact immutable path as `supervision.manifest_path`.
The bundle gate runs before model or data-loader construction and cannot be skipped.

## Train the Reference Configuration

```bash
uv run naics-embedder train --config conf/config.yaml   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

The defaults are `experiment_name=reference`, dimension 16, masked-mean fusion, radius bound
8, and 128 task queries per step. CUDA uses `bf16-mixed`; CPU and MPS use `32-true`. The text
Trainer uses one device. Fusion, projection, geometry and losses stay float32 under CUDA.

For a fresh experiment, set a new name before changing settings:

```bash
uv run naics-embedder train experiment_name=reference-small   data_loader.queries_per_step=64 training.learning_rate=1e-5   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

A fresh start refuses a directory that already contains checkpoints. A resume requires the
same bundle, encoder, preprocessing, seed, settings and experiment directory:

```bash
uv run naics-embedder train --ckpt-path last   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

Resume only `last` to continue the latest state. A run that early stopping ended exits 1 with
`early stopping ended the run at epoch k`. A run whose epoch budget is spent trains no further
epochs. Changing the epoch budget is a different run. There is no weights-only migration;
pre-`req11-v1` checkpoints are refused before model loading.

`--skip-validation` skips advisory data/cache checks; the immutable bundle gate still runs.
See [Stage-3 supervision integrity](text_training.md#stage-3-supervision-integrity) and
[exact resume](text_training.md#exact-resume).

## Monitor and Visualize

The outcome validation panel's MRR selects the earliest best epoch. Each epoch writes
`monitor_reads.jsonl` and `epoch_summary.jsonl` in `checkpoints/<experiment_name>/`.
The selected checkpoint is `epoch=NNN.ckpt`; `last.ckpt` is the latest training state.

```bash
uv run naics-embedder tools config
uv run naics-embedder tools visualize   --summary checkpoints/reference/epoch_summary.jsonl   --output-dir outputs/visualizations/reference
```

The figures show MRR, the three losses and their total, logit scales, and per-level radius
means with SD bands. They read the epoch summary, not console logs or structural validation.

## Export and Check Radius

Use the checkpoint named by the monitor's earliest highest MRR, rather than assuming `last`
is the selected one. This example uses `epoch=001.ckpt`:

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/reference/epoch=001.ckpt   --output data/reference/table.parquet   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
uv run naics-embedder tools radius-report --checkpoint checkpoints/reference/epoch=001.ckpt   --table data/reference/table.parquet --output data/reference/radius_report.json   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

Export writes the arm's coordinates, bounded tangents for the default hyperbolic arm, and their
provenance. The radius report checks live radius gradients, per-level spread, sector radii,
manifold validity, distance precision and the three terms' gradients; for a flat arm, only the
terms' and scales' gradients. A failed criterion produces a report and exits 1.

## Compare Configurations

The [reference campaign](text_training.md#reference-campaign) trains seeds 1–10, checks every
run before the first decision read, writes a three-panel arm record with `tools sweep`, and fixes
margins with `tools margins --multiple 3` before candidate reads. Panel test splits stay sealed.
For all options, see [CLI usage](usage.md) or `uv run naics-embedder tools --help`.
