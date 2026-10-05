# NAICS Hyperbolic Embedding System

## Architecture

One shared LoRA-adapted backbone reads each present code field and each query. The reference
fusion is the masked mean, followed by one `Linear(384, d)`, where d is 8, 16 or 32 (default 16).
The text head represents direction and radius on the Lorentz hyperboloid at fixed curvature 1.
The optional HGCN stage refines the explicit parent–child graph using its own curriculum.

```text
Marked code fields and queries
             |
Shared LoRA backbone, mean pooling per present field
             |
Masked fusion -> one Linear(384, d)
             |
Bounded radius and direction -> Lorentz point
             |
Task + code-to-code + radial terms
             |
Outcome validation MRR -> earliest best checkpoint
             |
Tangent code-table export -> optional HGCN -> three-panel decision
```

## Shared Text Encoding

The fields are `title`, `description`, `examples`, `excluded`, and `query`. A present text is
marked with its field, such as `title: Computer Systems Design Services`. Null or blank channels
never enter the backbone and do not contribute to fusion. Every field shares the same LoRA
adapters. The reference backbone is `sentence-transformers/all-MiniLM-L6-v2`.

Texts longer than the backbone's trained input window use the committed window-fitting
summaries. Token caches pin the descriptions, tokenizer and window; the checkpoint contract
records the summary hash. Truncation is refused. See [input-window API](api/input_window.md).

## Fusion and Projection

`model.fusion` chooses masked mean, attention pooling, or MoE. Attention starts as the masked
mean. MoE routes the masked mean through top-2 experts and adds its load-balancing term only for
that ablation. Neither fusion choice introduces a mining or sampling curriculum. One affine
projection maps the fused backbone vector to `model.dimension`.

## Hyperbolic Geometry and Live Radius

For projection v, let a be its norm and u its direction. The head computes
`r = R * tanh(a / R)`, with `R = model.radius_bound` (default 8), then maps tangent r u to
`(cosh(r), sinh(r) u)`. The zero vector maps to the origin. There is no learned or configurable
text curvature and no fixed radius cap at 2. The head has no parameters; gradients reach the
projection through r and u.

Training distances use the stable polar form in float32, including zero distance for coincident
anchors. Panel reads reconstruct Lorentz points from exported tangent coordinates and compute
on the CPU in float64. Export columns are `code`, `index`, `level`, then `e0` through `e{d-1}`
in the bundle's codebook order. Each embedding has d tangent coordinates; the reconstructed
Lorentz point has d + 1 coordinates.

## Three-Term Objective

The objective has three terms with independent learned positive logit scales for the first two:

- `task_loss`: a task query's cross-entropy sums probability over all its named target codes.
- `code_code_loss`: each live code anchor matches a tree-distance soft target over all codes,
  excluding itself and its unary partner.
- `radial_loss`: squared error from the live anchor radius to `radial_step * (level - 1)`.

The total is task loss plus `code_code_weight * code_code_loss` and
`radial_weight * radial_loss`, with the MoE load-balancing term only under `fusion=moe`.
There is no DCL term, false-negative clustering, structural-preference loss or geometric miner
in text training. [The training guide](text_training.md#the-three-terms) defines the candidate
sets and masks.

## Epoch and Candidate Cache

The data module makes two streams: code anchors and eligible task queries. A deterministic
seed/epoch permutation visits each member once. The number of steps is the ceiling of query
count divided by `queries_per_step`: 11,039 queries at 128 give 87 steps. Code anchors are split
across those steps as evenly as possible.

A detached cache encodes all code rows in eval mode without gradients at fit start and after
each epoch. At each step, current live code anchors replace their cached rows. A task query's
candidates therefore include live gradients where it targets current anchors and detached
embeddings elsewhere. No text validation loader is constructed.

## Monitor, Health and Exact Resume

After refreshing the cache, the outcome validation monitor reads MRR, records the selection-log
read, and writes `monitor_reads.jsonl`. That float64 MRR is `val/outcome_mrr`: the sole score for
the earliest best checkpoint, plateau scheduler and early stopping. `epoch_summary.jsonl`
records the same MRR alongside loss, scale and per-level radius health values.

Text training uses one device. CUDA's backbone uses `bf16-mixed`; fusion, projection, head,
distances and losses remain float32. CPU and MPS use `32-true`. The model uses AdamW, one warmup
epoch and an MRR-driven plateau schedule. The learned scales have no weight decay and are
clamped after each optimizer step.

Checkpoints identify objective `req11-v1`, the bundle, encoder, preprocessing, seed and 21 run
settings. A pre-objective checkpoint is refused before model loading. Exact resume requires the
same experiment directory and settings, rebuilds the code cache, and continues both JSONL files
after retaining records through the resumed epoch. Resume only `--ckpt-path last`. A completed
early-stopped run exits 1 with `early stopping ended the run at epoch k`; a spent epoch budget
is a no-op. See [exact resume](text_training.md#exact-resume).

## Diagnostics and HGCN

`tools diagnostics` reports Req 6's structural statistics without thresholds or selection.
HGCN retains its four-phase graph curriculum, triplet objective, graph samplers and curvature
utilities. Its structural validation statistics describe the graph model; they are not text
checkpoint selectors. The HGCN feeder accepts the current text checkpoint contract and exports
code embeddings from the selected checkpoint.

### Structural Spearman v1


The definition identifier is `structural-spearman-v1`; external fields use
`structural_spearman_v1`. The Python entry point remains
`HierarchyMetrics.spearman_correlation(predicted_distances, target_distances, min_distance=0.1)`.
Inputs are equal-shaped square tensors with the documented real floating or integer dtypes.
They are detached and transferred to CPU before validation and calculation.

Both orientations of every off-diagonal pair must be finite and symmetric. Each matrix uses its
own source-dtype tolerance: float64 `(rtol=1e-7, atol=1e-9)`, float32 `(1e-5, 1e-7)`,
float16/bfloat16 `(1e-3, 1e-3)`, and exact equality for integers. Mirrored values are promoted to
float64 and arithmetically averaged; the strict upper triangle contributes exactly
`N(N-1)/2` candidates. The diagonal, including non-finite diagonal sentinels, is ignored.

Canonical target distances below the finite `min_distance` threshold are removed only after
validation. `n_total` counts candidates before filtering; `n_pairs` counts observations after it.
SciPy uses average ranks for exact ties in the two float64 vectors; tolerance does not group
near-equal observations. The p-value is discarded. The public result's `correlation` is a
detached float32 scalar on the metric's configured device; CUDA and MPS do not select a
different ranking algorithm. Source quantization cannot be reversed by promotion.

Malformed inputs raise `StructuralMetricInputError`, a `ValueError` subclass. Statistically
undefined results have a `NaN` tensor and these ordered reasons: fewer than two filtered
observations (`fewer_than_two_observations`), both vectors constant
(`constant_prediction_and_target`), prediction constant (`constant_prediction`), or target
constant (`constant_target`). Defined results have `status='defined'` and `reason=None`.
An unexpected non-finite SciPy result for otherwise defined inputs raises `RuntimeError`.

The general evaluation runner returns the complete result under `structural_spearman_v1`:
`correlation`, `n_pairs`, `n_total`, `definition`, `status`, and `reason`. Training artifacts
serialize undefined correlation as JSON `null`; Lightning omits that numeric scalar but logs
the pair counts.

All unversioned historical fields (`spearman`, `spearman_correlation`,
`val/spearman_correlation`, `val_spearman_correlation`) identify
`legacy-ordinal-rank-v0`. Their order-sensitive ordinal ranks are not valid tied-rank Spearman
and are not directly comparable with v1. Do not rewrite, dual-write, or numerically convert old
artifacts.

This rank repair does not validate the formula that produced the distance matrices. HGCN full
evaluation remains fixed at curvature `1.0`. The text objective uses unit curvature without a
curvature configuration field. Non-unit-curvature HGCN metric corrections are a separate change.

## Reference Campaign

The reference campaign uses ten seeds and three panels: outcome validation MRR, and the
regressor panel's seen and held-out regimes at level 6. `tools sweep` validates every run's
checkpoint, settings and complete monitor records before any export or decision read. Its arm
record carries the monitor reads as well as the three decision reads per seed.

`tools margins --multiple 3` fixes the reference margins before candidate reads. `tools decide`
uses paired two-stage resampling and the required non-inferiority/superiority rule. Structural
diagnostics and training health cannot replace those panel decisions. The panel test splits
remain sealed. See [the campaign workflow](text_training.md#reference-campaign).

## Implementation Reference

| Area | Modules |
|------|---------|
| Marked text and fusion | `text_model/fields.py`, `shared_encoder.py`, `fusion.py`, `moe.py` |
| Query/target facts | `supervision/activity.py`, `queries.py`, `code_targets.py` |
| Two-stream steps | `text_model/dataloader/datamodule.py` |
| Objective and live radius | `text_model/loss.py`, `hyperbolic.py`, `naics_model.py` |
| Cache and validation monitor | `text_model/monitor.py` |
| Checkpoint campaign | `text_model/checkpoint_runner.py`, `decision/sweep.py` |
| Health artifacts | `text_model/epoch_summary.py`, `radius_report.py` |
| Graph refinement | `graph_model/hgcn.py`, `graph_model/curriculum/` |

The text model has three mixins: `LossMixin` for MoE balancing, `LoggingMixin` for epoch health,
and `OptimizerMixin` for AdamW, warmup, plateau and scale clamping. Geometry utility operations
also have a `torch.compile` implementation for the callers that use it.

```bash
uv run naics-embedder tools visualize --summary checkpoints/reference/epoch_summary.jsonl
uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic   --codebook PATH/naics_codebook.parquet
```
