# System Overview

The project plans three different types of embedding over the bundle's 2,125 NAICS 2022 codes:
relatedness (type 1), text→code (type 2) and economic coordinates (type 3). The NAICS tree is
their shared codebook, not a fourth type. See the [home page](index.md) for each type's use, its
judging panel and its status. This page describes the code that exists and which type it serves.

## What Each Part Serves

| Part | Modules | Serves | State |
|------|---------|--------|-------|
| Supervision bundle and codebook | `data/`, `supervision/` | All three types | Built |
| Text model | `text_model/` | Type 2, and type 3's trained candidates | Built and trained |
| Outcome panel | `panels/outcome.py`, `panels/lexical_encoder.py` | Type 2 (reported only), and checkpoint selection | Built |
| Regressor and growth panels | `panels/regressor.py`, `panels/ridge.py` | Type 3 | Built |
| Decision tooling | `decision/` | Type 3, and the earlier three-panel decisions | Built |
| Deterministic economic candidates | `tree_model/`, `decision/candidates.py` | Type 3 | Built |
| ITA panel | none | Type 2 | Not built |
| Relatedness panel and model | none | Type 1 | Not built |
| HGCN graph refinement | `graph_model/` | No type | Parked |

The economic decision (type 3) has not run, and no text→code (type 2) decision exists under the
new design. No sealed test split has been opened.

## Text Model Architecture

One shared LoRA-adapted backbone reads each present code field and each query. The reference
fusion is the masked mean. One `Linear(384, d)` follows, with d = 8, 16 or 32 (default 16). A
geometry head then maps the projection to a point.

```text
Marked code fields and queries
             |
Shared LoRA backbone, mean pooling per present field
             |
Masked fusion -> one Linear(384, d)
             |
Geometry head: hyperbolic (default), euclidean or spherical
             |
Task + code-to-code (+ radial) terms
             |
Outcome validation MRR -> earliest best checkpoint
             |
2,125-code table export
```

## Shared Text Encoding

The fields are `title`, `description`, `examples`, `excluded` and `query`. Each present text is
marked with its field, such as `title: Computer Systems Design Services`. A null or blank channel
never enters the backbone and does not contribute to fusion. Every field shares the same LoRA
adapters. The reference backbone is `sentence-transformers/all-MiniLM-L6-v2`.

A text longer than the backbone's trained input window uses the committed window-fitting
summaries. Token caches pin the descriptions, tokenizer and window. The checkpoint contract
records the summary hash. Truncation is refused. See the [input-window API](api/input_window.md).

## Fusion and Projection

`model.fusion` chooses masked mean, attention pooling or a mixture of experts (MoE). Attention
starts as the masked mean. MoE routes the masked mean through top-2 experts and adds its
load-balancing term only for that ablation. No fusion choice adds a mining or sampling
curriculum. One affine projection maps the fused backbone vector to `model.dimension`.

## Geometry Heads and Live Radius

`model.geometry` selects the head: `hyperbolic` by default, or the flat `euclidean` and
`spherical` heads (see [geometry arms](text_training.md#geometry-arms)). The default is the
reference configuration, not the winner of the geometry × dimension comparison. That comparison
adopted `euclidean-d16` (`specs/findings/geometry-by-dimension.md`). Under the new design, its
decision stands as history and defines no deliverable.

The rest of this section describes the hyperbolic head. For projection v, let a be its norm and u
its direction. The head computes `r = R * tanh(a / R)`, with `R = model.radius_bound` (default
8). It then maps the tangent r u to `(cosh(r), sinh(r) u)`. The zero vector maps to the origin.
Text curvature is fixed at 1, with no setting or learned parameter, and no fixed radius cap at 2.
The head has no parameters. Gradients reach the projection through r and u.

Training distances use the stable polar form in float32, which gives zero distance for
coincident anchors. Panel reads reconstruct Lorentz points from the exported tangent coordinates
and compute on the CPU in float64. The export columns are `code`, `index`, `level`, then `e0`
through `e{d-1}`, in the bundle's codebook order. Each embedding has d tangent coordinates. The
reconstructed Lorentz point has d + 1 coordinates.

## Three-Term Objective

The objective has three terms. The first two have independent learned positive logit scales:

- `task_loss`: a task query's cross-entropy sums probability over all its named target codes.
- `code_code_loss`: each live code anchor matches a tree-distance soft target over all codes,
  except itself and its unary partner.
- `radial_loss`: squared error from the live anchor radius to `radial_step * (level - 1)`. It
  exists only under the hyperbolic head.

The total is the task loss plus `code_code_weight * code_code_loss` and
`radial_weight * radial_loss`. MoE adds its load-balancing term only under `fusion=moe`. Text
training has no DCL term, false-negative clustering, structural-preference loss or geometric
miner. [The training guide](text_training.md#the-three-terms) defines the candidate sets and
masks.

## Epoch and Candidate Cache

The data module makes two streams: code anchors and eligible task queries. A deterministic
seed/epoch permutation visits each member once. The step count is the query count divided by
`queries_per_step`, rounded up: 11,039 queries at 128 give 87 steps. The code anchors are split
across those steps as evenly as possible.

At fit start and after each epoch, a detached cache encodes all code rows in eval mode without
gradients. At each step, the current live code anchors replace their cached rows. So a task
query's candidates have live gradients where they are current anchors, and detached embeddings
elsewhere. Text training builds no validation loader.

## Monitor, Health and Exact Resume

After the cache refresh, the outcome validation monitor reads MRR (mean reciprocal rank). It
records the selection-log read and writes `monitor_reads.jsonl`. That float64 MRR is
`val/outcome_mrr`. It is the only score for the earliest best checkpoint, the plateau scheduler
and early stopping. `epoch_summary.jsonl` records the same MRR with the loss, scale and per-level
radius health values.

Under the new design, ITA validation reads will select type 2's checkpoints. The outcome panel
keeps this selection role for the trained economic candidates. No ITA monitor exists yet.

Text training uses one device. On CUDA the backbone uses `bf16-mixed`. Fusion, projection, head,
distances and losses stay float32. CPU and MPS use `32-true`. The model uses AdamW, one warmup
epoch and an MRR-driven plateau schedule. The learned scales have no weight decay, and each
optimizer step clamps them.

Checkpoints identify objective `req11-v1`, the bundle, encoder, preprocessing, seed and 21 run
settings. A checkpoint from an earlier objective is refused before model loading. Exact resume
requires the same experiment directory and settings. It rebuilds the code cache, keeps both
JSONL files through the resumed epoch, and continues them. Resume only `--ckpt-path last`. An
early-stopped run exits 1 with `early stopping ended the run at epoch k`. A spent epoch budget
is a no-op. See [exact resume](text_training.md#exact-resume).

## Economic Candidates and Decision

Type 3's test asks whether a candidate's coordinates predict employment growth better than the
tree's ancestor indicators. Both read the same covariates, and the reference is
`covariates+ancestors`. The growth panel has two regimes, `growth_seen` and `growth_heldout`.

[`tools growth-margins`](usage.md#tools-growth-margins) fixed each regime's margin δ once, on
2026-10-17, as 0.05 times the reference's validation MSE. That read scored no candidate
(`specs/findings/growth-margins.md`). `tools decide` compares each candidate with the reference
at 95 % non-inferiority and 97.5 % superiority. Only a candidate adopted over the reference is
eligible. With none, the reference stands.

`conf/data/economic_candidates.yaml` freezes 18 candidates. Nine are deterministic and read text
and the tree only:

- the text-only table reduced by PCA
- the classical scaling of the tree distance
- the text-only table smoothed over each code's lineage

`tools economic-build` builds each of them at d = 8, 16 and 32. The other nine are the trained arms
of the geometry × dimension comparison. `tools economic-read` reads one candidate. The decision
over all 18 has not run. It waits for the project owner's approval.

## Diagnostics and HGCN

`tools diagnostics` reports structural statistics: sector separation, within-sector rank
correlation, ancestor MAP, NDCG graded by the lowest common ancestor, the Pearson correlation of
distance with the tree distance, and parent retrieval without unary pairs. It has no thresholds
and selects nothing.

The hyperbolic graph convolutional network (HGCN) in `graph_model/` is parked. It keeps its
four-phase graph curriculum, triplet objective, graph samplers and curvature utilities, but it
feeds no type. Its old question, whether propagation over the tree helps, went to the smoothed
economic candidates. The relatedness type does not revive the HGCN stage, but it may reuse its
code. See [HGCN training](hgcn_training.md).

### Structural Spearman v1

The definition identifier is `structural-spearman-v1`. External fields use
`structural_spearman_v1`. The Python entry point remains
`HierarchyMetrics.spearman_correlation(predicted_distances, target_distances, min_distance=0.1)`.
The inputs are equal-shaped square tensors with the documented real floating or integer dtypes.
They are detached and moved to the CPU before validation and calculation.

Both orientations of every off-diagonal pair must be finite and symmetric. Each matrix uses its
own source-dtype tolerance: float64 `(rtol=1e-7, atol=1e-9)`, float32 `(1e-5, 1e-7)`,
float16/bfloat16 `(1e-3, 1e-3)`, and exact equality for integers. Mirrored values are promoted to
float64 and averaged. The strict upper triangle contributes exactly `N(N-1)/2` candidates. The
diagonal, including non-finite diagonal sentinels, is ignored.

Canonical target distances below the finite `min_distance` threshold are removed only after
validation. `n_total` counts candidates before this filter, and `n_pairs` counts observations
after it. SciPy uses average ranks for exact ties in the two float64 vectors. Tolerance does not
group near-equal observations. The p-value is discarded. The public result's `correlation` is a
detached float32 scalar on the metric's configured device. CUDA and MPS do not select a different
ranking algorithm. Promotion cannot reverse source quantization.

Malformed inputs raise `StructuralMetricInputError`, a `ValueError` subclass. A statistically
undefined result has a `NaN` tensor and one of these ordered reasons:

1. fewer than two filtered observations (`fewer_than_two_observations`)
2. both vectors constant (`constant_prediction_and_target`)
3. prediction constant (`constant_prediction`)
4. target constant (`constant_target`)

A defined result has `status='defined'` and `reason=None`. An unexpected non-finite SciPy result
for otherwise defined inputs raises `RuntimeError`.

HGCN's training artifacts write an undefined correlation as JSON `null`. Lightning omits that
numeric scalar but logs the pair counts.

All unversioned historical fields (`spearman`, `spearman_correlation`,
`val/spearman_correlation`, `val_spearman_correlation`) identify `legacy-ordinal-rank-v0`. Their
order-sensitive ordinal ranks are not valid tied-rank Spearman, and they are not directly
comparable with v1. Do not rewrite, dual-write or numerically convert old artifacts.

This rank repair does not validate the formula that produced the distance matrices. HGCN's full
evaluation stays fixed at curvature `1.0`. The text objective uses unit curvature without a
curvature setting. Non-unit-curvature HGCN metric corrections are a separate change.

## Reference Campaign (History)

The reference campaign trained ten seeds of the default configuration. It read three panels:
outcome validation MRR, and the regressor panel's seen and held-out regimes at level 6.
`tools sweep` validated every run's checkpoint, settings and complete monitor records before any
export or decision read. `tools margins --multiple 3` fixed the reference margins before any
candidate read. The geometry × dimension comparison then decided under those margins. Both are
complete records. See [the campaign workflow](text_training.md#reference-campaign).

## Implementation Reference

| Area | Modules |
|------|---------|
| Marked text and fusion | `text_model/fields.py`, `shared_encoder.py`, `fusion.py`, `moe.py` |
| Query and target facts | `supervision/activity.py`, `queries.py`, `code_targets.py` |
| Two-stream steps | `text_model/dataloader/datamodule.py` |
| Objective and heads | `text_model/loss.py`, `hyperbolic.py`, `heads.py`, `naics_model.py` |
| Cache and validation monitor | `text_model/monitor.py` |
| Checkpoint campaign | `text_model/checkpoint_runner.py`, `decision/sweep.py` |
| Health artifacts | `text_model/epoch_summary.py`, `radius_report.py` |
| Economic candidates and decision | `tree_model/`, `decision/candidates.py`, `decision/economic.py` |
| Graph refinement (parked) | `graph_model/hgcn.py`, `graph_model/curriculum/` |

The text model has three mixins: `LossMixin` for MoE balancing, `LoggingMixin` for epoch health
and `OptimizerMixin` for AdamW, warmup, plateau and scale clamping.

```bash
uv run naics-embedder tools visualize --summary checkpoints/reference/epoch_summary.jsonl
uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic   --codebook PATH/naics_codebook.parquet
```
