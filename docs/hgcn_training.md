# NAICS Hyperbolic Embedding System — HGCN Refinement Guide

## Overview

This document explains the final stage of the NAICS hyperbolic embedding pipeline: refinement
using a Hyperbolic Graph Convolutional Network (HGCN).

## 1. Purpose of HGCN Refinement

Integrates NAICS taxonomy directly into embedding geometry.

## 2. Input Requirements

- Lorentz hyperbolic embeddings
- NAICS parent–child graph
- Level metadata

Set `supervision_manifest_path` in `conf/graph.yaml` to the manifest printed by
`uv run naics-embedder data supervision`. Relations, training pairs, and the distance matrix are
then read from that one validated bundle, and any explicitly configured path from another source
is rejected. Without a manifest, HGCN falls back to the legacy
`./data/naics_relations.parquet`, `./data/naics_training_pairs.parquet`, and
`./data/naics_distance_matrix.parquet` files, which `data all` no longer writes. HGCN consumes
only its legacy negative fields (`negative_idx`, `negative_code`, `relation_margin`,
`distance_margin`); the repaired Stage-3 semantics and selection policy do not change its training.

## 3. Running the Refinement

```bash
python train_hgcn.py --config configs/hgcn.yaml
```

## 4. HGCN Layer Operation

Each layer performs log-map, graph convolution in tangent space, activation, and exp-map.

## 5. Refinement Loss Functions

- Hyperbolic Triplet Loss
- Per-Level Radial Regularization

## 6. Learnable Curvature

Curvature parameter is optimized jointly.

## 7. Output of HGCN Refinement

Refined Lorentz-model hyperbolic embeddings aligned with taxonomy structure.

## 8. Validation Metrics

Stage 4 now mirrors the text-model evaluation suite so you can verify that graph refinement does not erode global structure:

- **Cophenetic correlation** – correlation between embedding distances and tree distances.
- **Structural Spearman v1** (`structural_spearman_v1`) - average-rank structural agreement,
  with filtered and total unique-pair counts.
- **NDCG@K (default: 5/10/20)** – position-aware ranking quality.
- **Distortion stats** – mean/std/median stretch between embedding and tree distances.

`structural-spearman-v1` uses the strict upper triangle of validated square distance matrices,
averages mirrored values in CPU float64, excludes the diagonal, filters canonical target
distances at `min_distance=0.1`, and assigns average ranks to exact ties. See the
[complete contract](overview.md#structural-spearman-v1).

Lightning logs `val/structural_spearman_v1` only when defined, plus
`val/structural_spearman_v1_n_pairs` and `val/structural_spearman_v1_n_total`.
`training_log.json` uses the existing `val_` prefix for the complete state:

```json
{
  "val_structural_spearman_v1": null,
  "val_structural_spearman_v1_n_pairs": 6,
  "val_structural_spearman_v1_n_total": 6,
  "val_structural_spearman_v1_status": "undefined",
  "val_structural_spearman_v1_reason": "constant_target",
  "val_structural_spearman_v1_definition": "structural-spearman-v1"
}
```

Defined values are numeric with status `defined` and a `null` reason. Undefined results remain
non-fatal, with one of the documented reasons, and do not produce a numeric Lightning scalar.
Malformed matrices, including shape mismatches and non-finite off-diagonal entries loaded from
Parquet, raise `StructuralMetricInputError`; they are not treated as unavailable optional
metrics. A missing distance file remains a separate optional-evaluation condition.

Historical `spearman`, `spearman_correlation`, `val/spearman_correlation`, and
`val_spearman_correlation` fields are `legacy-ordinal-rank-v0`. Their order-sensitive ordinal
ranks are not valid tied-rank Spearman coefficients and are not directly comparable with v1.
No historical artifact is rewritten and no new legacy alias is emitted.

Full HGCN evaluation remains explicitly fixed at curvature `1.0`. This rank repair does not
repair non-unit-curvature distances or change the graph architecture, objectives, or curriculum.

Metrics are logged once per validation run (default: every epoch). They require the precomputed tree distance matrix produced in Stage 2.

### Configuration

Add the following keys to your `GraphConfig` (or `configs/hgcn.yaml`) to customize evaluation:

| Key | Description |
| --- | --- |
| `distance_matrix_parquet` | Path to `naics_distance_matrix.parquet`. Required to unlock hierarchy metrics. |
| `full_eval_frequency` | Run the expensive metrics every _N_ optimizer steps (default `1`, meaning every validation epoch). |
| `ndcg_k_values` | List of K values used for NDCG logging. |

If `distance_matrix_parquet` is missing, HGCN automatically skips the extra metrics and continues with the lightweight batch metrics (triplet accuracy, etc.).

## 9. Pre/Post Verification Workflow

After both Stage 3 and Stage 4 finish, run the automated comparison from [Issue #67](https://github.com/lowmason/naics-embedder/issues/67) to confirm that HGCN preserved the Stage 3 geometry:

```bash
uv run naics-embedder tools verify-stage4 \
  --pre ./output/hyperbolic_projection/encodings.parquet \
  --post ./output/hgcn/encodings.parquet
```

Additional options let you override the distance matrix, relations parquet, or the acceptable degradation thresholds:

| Option | Purpose |
| --- | --- |
| `--max-cophenetic-drop` | Maximum allowable decrease in cophenetic correlation (default `0.02`). |
| `--max-ndcg-drop` | Maximum allowable decrease in NDCG@K (default `0.01`). |
| `--min-local-improvement` | Required increase in parent retrieval accuracy (default `0.05`). |
| `--ndcg-k` | Which `K` to evaluate for NDCG (default `10`). |
| `--parent-top-k` | Size of the neighborhood used for parent retrieval (default `1`). |

The command prints pre/post metrics, deltas, and PASS/FAIL indicators for each threshold. Integrate it into CI to prevent regressions before shipping updated embeddings.
