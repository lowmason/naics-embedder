# Text Training Guide

The reference text model learns query retrieval, code geometry and live radii together. It uses
one shared LoRA backbone, one fusion/projection path, two streams per epoch and the three terms
of objective `req11-v1`. The outcome validation panel's MRR selects the earliest best checkpoint.
Graph refinement is a separate stage; see [HGCN training](hgcn_training.md).

## Quick Start

Build the immutable bundle and pass the exact manifest path it prints:

```bash
uv run naics-embedder data all
uv run naics-embedder train --config conf/config.yaml   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

The defaults are experiment `reference`, dimension 16, masked-mean fusion, radius bound 8,
128 queries per step, one warmup epoch and a 40-epoch budget. CUDA uses `bf16-mixed` for the
backbone only; fusion, projection, geometry and losses stay float32. CPU and MPS use `32-true`.
The text Trainer uses one device and refuses `devices > 1`.

## The Shared Encoder

One LoRA-adapted `sentence-transformers/all-MiniLM-L6-v2` reads every present text field,
marked `title:`, `description:`, `examples:`, `excluded:` or `query:`. Blank and absent channels
never enter the backbone. The model mean-pools each channel's tokens, fuses present channels,
and applies exactly one `Linear(384, d)` for d in {8, 16, 32}.

The default fusion is masked mean. Attention pooling is an option. MoE is an ablation: it sends
the masked mean through top-2 experts, with a load-balancing term under `model.fusion=moe` only.
It adds no mining, router-guided sampling or phase transitions.

The parameter-free head splits projection v into radius and direction. With a = norm(v),
`r = R * tanh(a / R)`, where `R = model.radius_bound`; the resulting Lorentz point is
`(cosh(r), sinh(r) u)`. Zero v maps to the origin. Text curvature is fixed at 1, without a
curvature setting. The radius remains live in every term; it is not normalized away or capped
at a fixed norm of 2.

## The Three Terms

`NAICSContrastiveModel.compute_losses` returns `StepLosses`. Distances use the stable polar
form in float32. The task and code-to-code logits have separate learned positive scales;
they start at `loss.logit_scale_init` and are clamped to `loss.logit_scale_range` after each
optimizer step, with no weight decay.

### Task Query Cross-Entropy

`task_loss` learns to decode an activity query to all its named codes. Training-role index
entries contribute their text and code. Explicit redirection phrases contribute activity
queries from the text before the redirection clause; a phrase
that exactly or nearly matches an outcome validation/test query is withheld before training.
Every remaining query is marked `query:` and read by the shared backbone.

For a phrase query, T contains its named codes that are not lineal to the referencing code.
N is the set of referencing codes, and C contains all codes at the query's level
plus N. The loss is `-log(sum(softmax(logits)[T]))` over C. Every member of N remains a candidate,
even when it is related to a target. Explicit exclusions are query facts, not sampled negative
training pairs. The target and candidate identities come from `build_task_queries`.

### Code-to-Code Cross-Entropy

`code_code_loss` uses each live code anchor against all codebook codes. The anchor itself and
its unary partner are removed. The remaining targets are the row-wise softmax of
`-D* / loss.target_temperature`, where D* is the committed taxonomy tree metric. Independent
scaled negative polar distances supply model logits. All target probability stays within the
same keep mask; no negative miner or false-negative clustering participates.

### Radial Error

`radial_loss` is the mean squared error between the live anchor radii and
`loss.radial_step * (level - 1)`. Default step 1 gives target radii 1, 2, 3, 4 and 5 at levels
2 through 6. This is a soft target, not a fixed radius: each anchor's derivative through radius
remains live, and codes at the same level can have different radii.

### Total and Defaults

```text
loss = task_loss
     + code_code_weight * code_code_loss
     + radial_weight * radial_loss
     + moe.load_balancing_coef * load_balancing_loss  (only under fusion=moe)
```

The two term weights and target temperature default to 1. DCL, hierarchy-preservation loss,
structural-preference loss, fixed-cap regularization, text curriculum, negative miners and
false-negative mitigation are retired. Their configuration keys are rejected rather than
reinterpreted as the new objective.

## Two-Stream Epochs and the Code Cache

An epoch visits every code anchor and every eligible query exactly once. Code and query
permutations depend on seed and epoch. The number of steps is
`ceil(n_queries / data_loader.queries_per_step)`; the reference's 11,039 queries at 128 give
87 steps. The 2,125 code anchors are divided across these steps as evenly as possible.
The loader has no text validation split or validation loader.

At fit start, and after each epoch, `refresh_code_cache` encodes all codebook code rows in eval
mode with no gradients. The cache holds detached float32 directions and radii. At each step,
current live code anchors replace their cached rows. The code-to-code and query terms thus use
all candidates while gradients reach the live anchors and queries. Evaluation preserves the
model's existing training flags and does not update dropout or adapters.

The token cache is a separate preprocessing artifact. It records descriptions, tokenizer,
window and summary identities. Long texts resolve through pinned window-fitting summaries;
silent truncation is refused. The per-epoch embedding cache is rebuilt on resume and is not a
checkpoint state that can become stale across processes.

## Stage-3 Supervision Integrity

`data supervision` writes an immutable `stage3-supervision-v2` bundle: codebook, tree distances,
unary pairs, redirections and training pairs, with one manifest and hashes. Loading validates
members, schemas, identifiers and relationships together. A missing manifest or any mismatch
refuses training before model or loader construction. `--skip-validation` only skips advisory
checks and cannot bypass this gate.

The bundle's training-pairs member stays present and validated. Text training never reads it;
it derives task queries and dense code targets from the bundle's other facts. HGCN retains the
training-pairs/sampling path for its separate graph objective.

`CheckpointContract` records contract version, bundle id, codebook fingerprint, objective
`req11-v1`, encoder architecture (layout, fusion, dimension and backbone name), and the
window-summary hash. The radius bound is saved in model hyperparameters and `run_settings`;
the resolved cached backbone revision is recorded in export provenance. Token caches separately
pin descriptions, tokenizer and window identities. Pre-objective and four-copy checkpoints
are refused before `load_from_checkpoint` in training, export, outcome reads and the HGCN feeder.
No checkpoint migrates weights into this objective.

## Outcome Validation Monitor

After each epoch's cache refresh, `OutcomeMonitor` scores the outcome validation split through
`OutcomePanel.score_logged`. Queries use the current shared model and codes use the refreshed cache.
The test split stays sealed. The selection-log read carries training-run id, seed, epoch and
matrix fingerprint; `monitor_reads.jsonl` beside the checkpoints preserves every read.

The float64 MRR is logged as `val/outcome_mrr`. ModelCheckpoint keeps the earliest epoch with
the highest value, plus `last.ckpt`. A tied later value cannot replace the earlier checkpoint.
AdamW has one warmup epoch, followed by an MRR-driven plateau scheduler and early stopping.
Structural statistics neither select a checkpoint nor control either scheduler.

## Epoch Summary and Training Health

`epoch_summary.jsonl` has one increasing, zero-based epoch row. It records MRR and finite health
values: task/code-code/radial/total losses, both logit scales, and radius mean/SD at levels 2–6.
MoE runs also record their load-balancing loss. Rows take `_log_health()`'s Python floats;
Lightning's float32 callback metrics do not round the saved health values. MRR is `null` only
for a run without a monitor.

```bash
uv run naics-embedder tools visualize   --summary checkpoints/reference/epoch_summary.jsonl   --output-dir outputs/visualizations/reference
```

The tool writes one `epoch_metrics.png` figure with MRR, loss, scale and radius panels,
including per-level SD bands. It accepts a
summary path rather than a console-log stage. `read_epoch_summary` refuses malformed, nonfinite
or unordered rows before plotting; the radius panel draws a band wherever both mean and SD
are recorded.

## Exact Resume

Continue only the latest checkpoint of the same run:

```bash
uv run naics-embedder train --ckpt-path last   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

The guards check the experiment directory, bundle, encoder, preprocessing, seed and all
21 settings from `run_settings`. Those include effective accelerator/precision, query batch
size, optimizer and stopping settings, epoch budget, clipping and accumulation. A fresh start
refuses an already used checkpoint directory. A resume from another directory or with other
settings exits 1 before model or data construction. `--checkpoint-load-mode` accepts only
`exact`; there is no weights-only path.

Resume restores the training-run id, optimizer, warmup/plateau and callback state, rebuilds the
code cache, and retains both JSONL files through the resumed epoch before continuing. Kept lines
preserve their bytes. Do not rewind with an older kept epoch checkpoint; use `last`.
A run that early stopping ended exits 1 with `early stopping ended the run at epoch k`, since
Lightning does not restore `trainer.should_stop`. A launcher must treat that message as a
finished run, not retry it. A run whose epoch budget is spent is a harmless no-op. Extending the
budget changes run settings and is refused on exact resume.

`training.trainer.val_check_interval` is retained in the config model but unread by the text
Trainer; there is no validation loader. Monitoring occurs once at each training epoch's end.

## Export, Diagnostics and Radius Verification

Use the earliest highest-MRR epoch from the monitor records. Export encodes every code through
that checkpoint in eval mode and writes bounded tangent coordinates `e0` through `e{d-1}` plus
checkpoint/table provenance. Panel reads reconstruct unit-curvature Lorentz points on the CPU
in float64. A table exported from another checkpoint or preprocessing pin is refused.

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/reference/epoch=001.ckpt   --output data/reference/table.parquet   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
uv run naics-embedder tools radius-report --checkpoint checkpoints/reference/epoch=001.ckpt   --table data/reference/table.parquet --output data/reference/radius_report.json   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

`epoch=001.ckpt` is an example; substitute the selected filename. The radius report is a CPU
read of the checkpoint, table and saved step-zero seed. It does not read an evaluation panel.
It checks nonzero gradients through each anchor radius, per-level SD > 1e-3, positive distinct
sector radii and their least gap, largest-radius manifold error, and float32/float64 distance
agreement over all ordered pairs in row chunks. It also reports nonzero gradient norms of all
three weighted terms and both scales. Failed criteria write the report and exit 1.

The distance comparison requires relative error at most 1e-3 on noncoincident pairs. Exact
coincident tangents must have zero float32 training distance; their float64 read residual is
reported separately. A noncoincident pair whose read distance rounds to zero fails.
Largest-radius manifold error must be at most `1e-9 * x0**2`.

`tools diagnostics` separately reports Req 6's structural measures, with no thresholds and no
checkpoint or arm selection. HGCN keeps its graph curriculum and structural logging; see
[Structural Spearman v1](overview.md#structural-spearman-v1) for that metric's contract.

## Reference Campaign

Train reference seeds 1–10 on CUDA with `bf16-mixed`, then bring the complete run directories
to the Mac for decision reads. Use a distinct directory per seed and keep the config identical
apart from seed and experiment name. Preserve checkpoints, monitor records and epoch summaries.
The frozen text-only comparator must use the same backbone revision and window summaries.

Keep the decision store and records under `~/naics-artifacts`, outside every worktree:

```bash
mkdir -p ~/naics-artifacts/records/stage7
uv run naics-embedder tools sweep --runs 'checkpoints/reference-seed-{seed}' \
  --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 --seed 8 --seed 9 --seed 10 \
  --text-only data/reference/text_only.parquet --store ~/naics-artifacts \
  --output ~/naics-artifacts/records/stage7/reference.json --purpose 'reference campaign validation' \
  --name reference --accelerator cuda \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
uv run naics-embedder tools margins \
  --reference ~/naics-artifacts/records/stage7/reference.json --multiple 3 \
  --name reference-margins --store ~/naics-artifacts \
  --output ~/naics-artifacts/records/stage7/margins.json
```

`--accelerator cuda` names the training settings, even when the read runs on the Mac.
`tools sweep` independently resolves the arm's cached backbone revision; a comparator cannot
supply that revision. Before any export or decision read, its combined preflight checks every
seed's complete epochs through `last.ckpt`, earliest best epoch, selected checkpoint, seed,
training-run id and 21 settings. It also verifies that monitor records identify validation reads
from the correct panel, with valid fingerprints and the correct seed, rather than test reads or
opening events. A selected epoch with a versioned sibling or a saved best score that differs
from the monitor is refused.

Each seed then exports the selected table beside the checkpoint and reads three validation
panels: outcome, regressor seen, and regressor held-out at level 6. The arm record carries all
monitor reads, checkpoint epoch and training-run id as well as decision reads. A decision arm
requires at least five complete seeds; this reference campaign uses ten.

Fix margins before any candidate arm's monitor or decision read. Candidate arm records and
reference records are compared by `tools decide` under Req 5's paired two-stage bootstrap and
tie order. The test splits stay sealed. Radius verification and structural diagnostics describe
the trained model; the three panels decide whether an arm is adopted.

## Troubleshooting

- A missing or inconsistent bundle: regenerate it with `data supervision` and use its printed
  manifest path; do not patch members or weaken hashes.
- A contract or settings refusal: inspect the configured run identity. An old objective or a
  changed seed, budget, radius bound or precision requires a fresh run directory.
- A stopped run: keep its selected checkpoint and monitor evidence; do not relaunch it.
- A nonfinite loss or inert radius term: inspect the epoch summary and `tools radius-report`;
  structural correlation is not the training control score.
- Memory pressure: reduce `data_loader.queries_per_step` for a fresh run, or choose a smaller
  dimension. Changing settings during exact resume is refused.
