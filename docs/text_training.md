# Training Guide

This guide explains how to train the NAICS Hyperbolic Embedding System with the dynamic Structure-Aware Dynamic Curriculum (SADC) scheduler. The current workflow uses a single configuration file (`conf/config.yaml`) to control model, data, trainer, and curriculum settings.

Stage-3 training runs under the **repaired supervision contract** (`stage3-supervision-v1`): one
immutable, validated supervision bundle is the only authority for code identity, structural facts,
explicit exclusions, and training pairs. See
[Stage-3 Supervision Integrity](#stage-3-supervision-integrity) before starting a run.

## Table of Contents

- [Training Guide](#training-guide)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [SADC Scheduler](#sadc-scheduler)
  - [CLI Reference](#cli-reference)
  - [Structural Spearman Validation](#structural-spearman-validation)
  - [Resuming and Overrides](#resuming-and-overrides)
  - [Stage-3 Supervision Integrity](#stage-3-supervision-integrity)
    - [Operator Workflow](#operator-workflow)
    - [The Supervision Bundle](#the-supervision-bundle)
    - [Three Independent Axes](#three-independent-axes)
    - [Candidate Identity](#candidate-identity)
    - [Exactly-One Exclusion Rotation](#exactly-one-exclusion-rotation)
    - [Structural Preference Loss](#structural-preference-loss)
    - [Cache Regeneration](#cache-regeneration)
    - [Exact Resume versus Weights-Only Migration](#exact-resume-versus-weights-only-migration)
    - [Legacy Containment](#legacy-containment)
    - [Rollout Gates](#rollout-gates)
  - [Sampling Architecture: Data Layer vs Model Layer](#sampling-architecture-data-layer-vs-model-layer)
    - [Data Layer (Streaming Dataset)](#data-layer-streaming-dataset)
    - [Model Layer (NAICSContrastiveModel)](#model-layer-naicscontrastivemodel)
    - [Interface Contract](#interface-contract)
  - [Migration for Legacy Chains](#migration-for-legacy-chains)
  - [Troubleshooting](#troubleshooting)

---

## Quick Start

Preprocess data and build the supervision bundle (see `docs/usage.md`), then launch training with
the manifest path that `data supervision` prints:

```bash
uv run naics-embedder data all
uv run naics-embedder train --config conf/config.yaml \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

Apply overrides inline—SADC will stay active and adapt phases automatically:

```bash
uv run naics-embedder train --config conf/config.yaml \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json \
  training.learning_rate=1e-4 training.trainer.max_epochs=20
```

To avoid repeating the override, set `supervision.manifest_path` in `conf/config.yaml`.

---

## SADC Scheduler

The scheduler runs three phases within a single training invocation:

1. **Structural Initialization (0–30%)**
   - Flags: `use_tree_distance`, `mask_siblings`
   - Effect: weights negatives by inverse tree distance and masks siblings.
2. **Geometric Refinement (30–70%)**
   - Flags: `enable_hard_negative_mining`, `enable_router_guided_sampling`
   - Effect: activates Lorentzian hard-negative mining and router-guided MoE sampling.
3. **False Negative Mitigation (70–100%)**
   - Flags: `enable_clustering`
   - Effect: enables clustering-driven false-negative elimination.

Phase boundaries are derived from the trainer's `max_epochs`. Flag transitions are logged to help
verify when each mechanism is active.

Two additional knobs were added for the experimentation tracks in [Issue #44](https://github.com/lowmason/naics-embedder/issues/44):

- `curriculum.phase_mode=two_phase` merges Phase 3 behaviors into Phase 2 for a simpler two-stage schedule.
- `curriculum.anneal.*` enables continuous schedules (e.g., annealing the tree-distance exponent or router mix ratio over `epochs` or when a metric threshold is reached).

---

## CLI Reference

Use the `train` command for all new runs:

```bash
uv run naics-embedder train --config conf/config.yaml
```

Key options:

- `--config PATH` — Base config file (default: `conf/config.yaml`).
- `--ckpt-path PATH` — Use a checkpoint, or `last` to pick the most recent run artifact.
- `--checkpoint-load-mode [exact|weights_only]` — How the checkpoint is used (default: `exact`).
  See [Exact Resume versus Weights-Only Migration](#exact-resume-versus-weights-only-migration).
- `--skip-validation` — Bypass the advisory pre-flight checks of data files and tokenization cache.
  The supervision bundle gate is mandatory and always runs.
- `OVERRIDES...` — Space-separated config overrides (e.g., `training.learning_rate=1e-4`).

---

## Structural Spearman Validation

Text validation reports `structural-spearman-v1` through versioned fields. Each mirrored distance
pair is validated and averaged in CPU float64; only the strict upper triangle (`i < j`) is used,
with the diagonal excluded. Canonical target distances are filtered at `min_distance=0.1`.
SciPy assigns average ranks to exact ties. See the
[complete input and undefined-result contract](overview.md#structural-spearman-v1).

Lightning logs `val/structural_spearman_v1` only when defined and always logs
`val/structural_spearman_v1_n_pairs` and `val/structural_spearman_v1_n_total`.
Undefined results emit one warning with the exact reason and omit the numeric scalar.
Malformed inputs raise `StructuralMetricInputError` and fail validation rather than being
swallowed by the epoch-end evaluation handler.

In distributed text training, structural Spearman and its counts describe **rank 0's sampled
validation population**. All ranks validate their own matrices, but only rank 0 publishes these
fields and undefined warnings and writes `evaluation_metrics.json`. The scalar and counts are
not reduced across ranks: averaging local correlations is not a global Spearman coefficient, and
undefined local populations must not select different collective operations. This rank-zero-only
metric is for reporting, not a distributed early-stopping monitor.

The existing `evaluation_metrics.json` history includes these fields. For example, a valid
four-node evaluation with a constant target produces:

```json
{
  "structural_spearman_v1": null,
  "structural_spearman_v1_n_pairs": 6,
  "structural_spearman_v1_n_total": 6,
  "structural_spearman_v1_status": "undefined",
  "structural_spearman_v1_reason": "constant_target",
  "structural_spearman_v1_definition": "structural-spearman-v1"
}
```

When defined, the value is numeric, status is `defined`, and reason is `null`. The other
undefined reasons are `fewer_than_two_observations`, `constant_prediction_and_target`, and
`constant_prediction`, in that precedence before `constant_target`.

Unversioned historical `spearman`, `spearman_correlation`, `val/spearman_correlation`, and
`val_spearman_correlation` fields are `legacy-ordinal-rank-v0`: defective, order-sensitive
ordinal-rank results, not directly comparable with v1. Existing files are not rewritten, and
new evaluations do not emit legacy aliases.

For comparisons covered by this repair, retain `loss.curvature: 1.0`. After configuring the
supervision manifest as described above, make the comparison setting explicit:

```bash
uv run naics-embedder train --config conf/config.yaml loss.curvature=1.0
```

The rank fix does not correct non-unit-curvature distances. HGCN full evaluation and Stage-4
verification also remain fixed at `1.0`; Stage-4 reports structural Spearman without using it
as an acceptance threshold.

---

## Resuming and Overrides

Resume the latest checkpoint produced under the current experiment name. Exact resume succeeds
only when the checkpoint was trained under the same supervision contract as the configured bundle:

```bash
uv run naics-embedder train --ckpt-path last
```

Override trainer settings without editing YAML:

```bash
uv run naics-embedder train \
  training.trainer.max_epochs=15 training.trainer.accumulate_grad_batches=4
```

---

## Stage-3 Supervision Integrity

The repaired Stage-3 contract keeps every piece of supervision attached to the identity it
describes. Structural facts are never mutated to encode exclusions, every candidate travels under
one occurrence identity from pool construction through every loss, and training fails closed
rather than continuing with partially aligned metadata.

### Operator Workflow

```bash
uv run naics-embedder data supervision
uv run naics-embedder train \
  --config conf/config.yaml \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json

# Exact resume: every identifier must match.
uv run naics-embedder train \
  --ckpt-path checkpoints/run/last.ckpt \
  --checkpoint-load-mode exact \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json

# Explicit legacy initialization: weights only, all training state resets.
uv run naics-embedder train \
  --ckpt-path checkpoints/legacy.ckpt \
  --checkpoint-load-mode weights_only \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

Each step is a gate:

1. **Generate** — `data supervision` builds a complete bundle in a staging directory, validates
   every artifact, writes the manifest last, and atomically publishes
   `data/supervision/stage3-supervision-v1/<bundle-id>/`. It prints
   `Supervision manifest: <path>`; nothing partial is ever visible under that path. The legacy
   stage commands (`data relations`, `data distances`, `data triplets`) print a migration notice
   and build the same complete bundle.
2. **Configure** — `supervision.manifest_path` names exactly one bundle. The shipped
   `conf/config.yaml` leaves it `null`, which parses but cannot train: the mandatory gate stops
   with `Repaired Stage-3 training requires supervision.manifest_path` and prints the command
   above.
3. **Validate** — before any DataModule, checkpoint, or model work, `train` re-validates the whole
   bundle (contract version, member hashes and row counts, Parquet contract metadata, codebook
   order and fingerprint, pair-fact coverage and orientation, matrix reconciliation, training-pair
   joins) and checks that `data_loader.streaming.descriptions_parquet` is the file the bundle was
   generated from. `--skip-validation` does not skip this gate.
4. **Train or resume** — a fresh run, an exact resume (identical contract), or an explicit
   weights-only migration. There is no automatic fallback to legacy files.

### The Supervision Bundle

A bundle is immutable and versioned. Its manifest records the contract and per-artifact schema
versions, NAICS vintage, codebook order and fingerprint, input fingerprints, generator revision,
generation parameters, the structural relation-ID mapping, and every member's path, SHA-256,
row count, and exclusion count. Every Parquet member also carries the contract version, bundle ID,
and schema version in its metadata, so artifacts from different bundles can never be mixed.

| Artifact | Contents |
|---|---|
| `naics_codebook.parquet` | Canonical `code_id` ↔ `code` order |
| `naics_pair_facts.parquet` | One row per unordered code pair: structural distance and relation, both exclusion directions, and their OR |
| `naics_distances.parquet` / `naics_distance_matrix.parquet` | Legacy-compatible long-form and matrix views, reconciled against the pair facts |
| `naics_relations.parquet` / `naics_relation_matrix.parquet` | Legacy-compatible relations with explicit exclusion columns (no exclusion relation) |
| `naics_training_pairs/` | Training pairs with identities, semantic fields, exclusion provenance, and raw structure |
| `curriculum_difficulty_thresholds.json` | Curriculum thresholds derived from the same bundle |

Validation failures name the artifact and bundle ID, for example
`distance_matrix (<bundle-id>): matrix does not reconcile with the long-form pair facts`,
`codebook hash mismatch at <path>: expected <sha>, found <sha>`,
`expected supervision contract stage3-supervision-v1, found <version> in <path>`,
`a direct positive is an explicit exclusion`, or
`structural relation fields contain an exclusion sentinel`. Regenerate the bundle rather than
editing members.

### Three Independent Axes

Each (anchor, candidate) pair carries three independent kinds of supervision:

- **Structure** — the raw NAICS tree distance and relation (`cross_sector` = 99 across sectors).
  Exclusion processing never alters these values.
- **Semantic target and source** — `RELATED` / `UNRELATED` / `UNKNOWN`, sourced from a
  `TRAINING_POSITIVE`, an `EXPLICIT_EXCLUSION`, or `UNLABELED`. Model-derived pseudo-relatedness is
  runtime metadata and is never persisted as ground truth.
- **Exclusion provenance** — which code published the exclusion (`anchor_excludes_candidate`,
  `candidate_excludes_anchor`) and their OR, `is_explicit_exclusion`.

An explicitly excluded pair may be structurally close; the repaired losses read structure and
exclusion separately instead of letting a sentinel distance stand in for both.

### Candidate Identity

A **code ID** says which NAICS code a candidate is; the same code may occur more than once in a
pool. A **candidate UID** `[rank, batch_row, source_slot]` identifies one occurrence. Every
training step encodes one canonical candidate pool, joins pair-dependent supervision (distance,
relation, exclusions, margins) for each local anchor by code ID, and then performs a single checked
gather: `NegativeCandidateBatch.select()` moves embeddings, code IDs, structure, exclusion flags,
margins, router outputs, and validity together and rejects any selection whose UIDs do not match
the pool. Miners only *propose* source indices; they never gather fields. Invalid padding rows keep
source slot `-1`, are never selectable, and never contribute to a loss. In multi-GPU runs only
intrinsic entity fields (UID, code ID, embedding, router output, validity) are gathered; supervision
is recomputed for each local anchor.

### Exactly-One Exclusion Rotation

When an anchor has explicit exclusions in its pool, final selection reserves exactly one slot for
one of them; the other slots come from strategy proposals over non-exclusion candidates, with
duplicates removed by code (keeping the smallest UID), ties broken by code ID then UID, and a
deterministic backfill. Proposals are consulted in order:

1. **Phase 2+ miners.** With hard-negative mining on, the geometric miner proposes its share of the
   `K` slots, `K - int(K * router_mix_ratio)`; with router-guided mining also on, the router fills
   the rest. `router_mix_ratio` comes from `curriculum.anneal` (default 0.5). A reserved exclusion
   slot comes out of the router's share. Miners score one occurrence per code (the smallest
   candidate UID, which the coordinator keeps) and never the anchor or positive code, so on
   multiple GPUs, where a code repeats across rows and ranks of the global pool, the miners still
   fill their slots with distinct codes.
2. **The difficulty proposal** from the data layer, which is the only proposal in Phase 1 and the
   fallback afterwards.
3. **Deterministic backfill** from the remaining eligible codes.

Ordinary candidates are **eligible** only if they are structurally farther from the anchor than
the positive, the same rule every generated training negative satisfies (including its
cross-sector, equal-distance, and lineal special cases). Candidates sourced at runtime, such as
universe backfill and the multi-GPU global pool, therefore never repel a relative the generated
supervision would not treat as a negative. Explicit exclusions are exempt because their exclusion
is authoritative. The reserved exclusion rotates across epochs:

```text
index = (stable_hash(seed, anchor_code_id) + epoch) mod n_exclusions
```

over the anchor's sorted exclusion code IDs, where `stable_hash` is a SHA-256 based,
process-independent hash of the training seed and anchor. This quota replaces the legacy
`phase1_exclusion_weight`, which the repaired configuration rejects. Explicit exclusions always
stay repulsive in the contrastive denominator: a pseudo-related (clustering) signal can never mask
or attract an explicit exclusion.

### Structural Preference Loss

`StructuralPreferenceLoss` replaces LambdaRank. For each anchor it compares every unordered pair of
eligible candidates among the positive and the selected negatives whose structural distances
differ by more than `tie_tolerance`. With `i` the structurally closer candidate and `j` the
farther one, and `d` the learned Lorentz distance to the anchor:

```text
loss_ij = softplus((d_i - d_j + margin) / temperature)
```

The gradient is positive in `d_i` and negative in `d_j`, so every step pulls the structurally
closer candidate in and pushes the farther one out; a correct ordering by at least `margin` costs
little. Explicit exclusions, invalid padding, the anchor's own code, and repeated codes never
participate. Optional pair weights are detached. Comparisons are averaged per anchor, then over
anchors with at least one comparison; with none, the term is a finite differentiable zero. The
term is logged as `train/structural_preference_loss` and configured under
`loss.structural_preference` (`weight`, `margin`, `temperature > 0`, `tie_tolerance >= 0`).

### Cache Regeneration

- **Tokenization cache** — reused only when its JSON sidecar (`<cache>.meta.json`) records the
  bundle's description and codebook fingerprints, tokenizer, and max length; otherwise it is
  rebuilt.
- **Streaming and multi-epoch caches** — stored in a versioned envelope keyed by contract, bundle
  ID, codebook fingerprint, and source-artifact fingerprints; caches from other bundles or legacy
  runs are rejected and regenerated.
- **Curriculum difficulty thresholds** — regenerated from the same bundle during bundle generation.
- **Graph preprocessing** — `uv run python -m
  naics_embedder.graph_model.curriculum.preprocess_curriculum --supervision-manifest <path>` and
  `supervision_manifest_path` in `conf/graph.yaml` read relations, distances, the distance matrix,
  and training pairs from one bundle; HGCN consumes only its legacy negative fields
  (`negative_idx`, `negative_code`, `relation_margin`, `distance_margin`).

### Exact Resume versus Weights-Only Migration

Every new checkpoint records its supervision contract under `stage3_supervision`: supervision
mode, contract version, bundle ID, codebook fingerprint, structural-preference-loss version, and
mining-contract version. Structural matrices are loaded from the validated bundle, not trusted from
checkpoint state.

- **`--checkpoint-load-mode exact`** (default) restores optimizer, scheduler, epoch, global step,
  curriculum, and sampler state. It is allowed only when every contract field matches the runtime
  bundle; otherwise training stops before model construction with
  `exact resume contract mismatch (saved, runtime): {...}`. Checkpoints without a contract fail
  with `legacy checkpoint has no Stage-3 contract and cannot exact resume; use weights_only
  explicitly`.
- **`--checkpoint-load-mode weights_only`** loads only allowlisted encoder parameters (`encoder.*`:
  transformer adapters, projection, MoE, and router). Loss modules and data-derived buffers
  (`loss_fn.`, `hierarchy_loss_fn.`, `lambdarank_loss_fn.`, `structural_preference_loss_fn.`,
  ground-truth distances, `norm_adaptive_margin.`) are skipped. Any other parameter group is fatal.
  Optimizer, scheduler, epoch, global step, curriculum, sampler, and mining state are discarded, and
  the run starts at epoch zero against the validated bundle. The log reports loaded, skipped, and
  freshly initialized parameter groups. This initializes from old weights; it does not undo what an
  old objective learned.

Embedding generation from a checkpoint applies the same contract check.

### Legacy Containment

`supervision.mode: legacy_containment` is an explicit, non-default mode for old configurations. It
is not contract-compliant Stage-3 training:

- LambdaRank, the structural preference loss, and the hierarchy loss are disabled, even when old
  weights are set (`loss.rank_order_weight` and `data_loader.streaming.phase1_exclusion_weight`
  are accepted only in this mode);
- hard-negative and router-based reordering and pseudo-related elimination or attraction are
  disabled; training uses local, unmined negatives in collated order, with the legacy
  repeat-last padding of shorter negative lists;
- MoE routing, load balancing, and radius regularizers still run;
- runs log `LEGACY CONTAINMENT` at startup and `train/integrity/legacy_containment = 1` each epoch;
- checkpoints are tagged with bundle ID `legacy-containment` and codebook fingerprint
  `unversioned`, so they can never exact-resume into repaired training; they may only seed a
  repaired run through `weights_only`.

### Rollout Gates

Four gates must pass before normal repaired Stage-3 training:

```bash
uv run pytest tests/unit/test_candidate_contract.py -q
uv run pytest tests/unit/test_loss.py::test_structural_preference_gradient_corrects_an_inversion -q
uv run pytest tests/unit/test_supervision_artifacts.py -q
uv run pytest tests/integration/test_stage3_training_step.py -q
```

They prove, in order, that a forced reorder keeps every candidate field on one identity, that the
structural preference gradient corrects an inverted pair, that bundle validation fails closed, and
that a full training step feeds every loss the same selected candidates.

During training, epoch-summed integrity counters report selection health:
`train/integrity/anchors_with_exclusions`, the per-reason selections (`quota_selections`,
`geometric_selections`, `router_selections`, `difficulty_selections`, `deterministic_backfills`),
`invalid_candidates_ignored` (padding), `structurally_ineligible_candidates`, and
`duplicate_candidates_removed`.

Validation scores every eligible candidate of each validation pool, including every exclusion,
with no mining or pseudo-labels, so `val/contrastive_loss` depends only on the model and the
epoch-independent validation pools. Its values are not comparable with legacy runs, whose
validation contrasted a fixed negative list.

---

## Sampling Architecture: Data Layer vs Model Layer

This page clarifies the split between the streaming data pipeline and the model during curriculum-driven training.

### Data Layer (Streaming Dataset)

- Build one canonical candidate pool per (anchor, positive) from the bundle's training pairs: all
  of the anchor's explicit exclusions plus unique ordinary codes, never the anchor or positive.
- Phase 1 sampling:
  - Inverse tree-distance weighting (`P(n) ∝ 1 / d_tree(a, n)^α`).
  - Sibling masking (`d_tree <= 2` set to zero).
  - Difficulty proposals over the pool (explicit exclusions are represented by the selection quota,
    not by sampling weight).
- Static baseline (SANS):
  - Set `sampling.strategy=sans_static` to replace the dynamic weighting with fixed near/far buckets.
  - Configure bucket ratios under `sampling.sans_static` (e.g., `near_bucket_weight`, `near_distance_threshold`).
  - The dataloader emits `sampling_metadata` so the model can log near/far percentages per batch, making it easier to benchmark against Issue [#43](https://github.com/lowmason/naics-embedder/issues/43).
- Outputs:
  - Tokenized anchors/positives.
  - One candidate pool per item: code IDs, sampling role/provenance, validity, source slots, and
    difficulty proposals. Pair-dependent supervision is joined later in the model.

### Model Layer (NAICSContrastiveModel)

The model is decomposed into functional **mixins** for maintainability:

| Mixin | Responsibility |
|-------|----------------|
| `DistributedMixin` | Global candidate-pool gathering for multi-GPU training |
| `LossMixin` | Contrastive, hierarchy, structural preference, and radius losses |
| `CurriculumMixin` | Canonical pools, checked negative selection, pseudo-related candidates |
| `LoggingMixin` | Training, selection-health, and validation metric logging |
| `ValidationMixin` | Validation step and evaluation metrics |
| `OptimizerMixin` | Optimizer and scheduler configuration |

**Curriculum-driven behavior:**

- Reads curriculum flags from `CurriculumScheduler`.
- Phase 2+ proposals (source indices into the canonical pool), consulted before the difficulty
  proposal:
  - Embedding-based hard negative proposals (Lorentzian distance), for a `1 - router_mix_ratio`
    share of the slots when router mining is also on.
  - Router-guided proposals (gate confusion) fill the remaining slots.
  - Norm-adaptive margins via `NormAdaptiveMargin` (sech-based decay) are logged for annealing.
- Phase 3:
  - Pseudo-related candidates from clustering, derived only after selection and never including
    an explicit exclusion.
- False-negative strategy (`false_negatives.strategy`):
  - `eliminate` removes effective false negatives from the contrastive denominator (default).
  - `attract` keeps them and applies an auxiliary attraction loss scaled by `attraction_weight`.
  - `hybrid` combines both behaviors for higher precision at the cost of extra compute.
- Logging:
  - Negative relationship distribution and tree-distance bins over the selected negatives.
  - Selection health counters, hard-negative distances, and adaptive margins.

### Interface Contract

- **Inputs expected from data layer:** one candidate pool per item with code IDs, validity, and
  source slots; no pair-dependent supervision is trusted from the batch.
- **Curriculum flags influence:**
  - Phase 1 flags (`use_tree_distance`, `mask_siblings`) act in the data layer.
  - Phase 2/3 flags (`enable_hard_negative_mining`, `enable_router_guided_sampling`, `enable_clustering`) act in the model layer.
- **Selection:** every strategy proposes source indices; the selection coordinator performs the
  only gather, so all losses see the same selected candidates.

## Migration for Legacy Chains

The legacy stage-by-stage curriculum files and chain configs are retired. To reproduce an old
multi-stage job, acknowledge the deprecated workflow explicitly:

```bash
uv run naics-embedder train-seq --legacy --num-stages 3 --config conf/config.yaml
```

New work should rely on `train` plus overrides—the dynamic SADC scheduler replaces manual chains and
static curriculum files.

---

## Performance Optimization

### torch.compile Support

Core Lorentz operations are optimized using PyTorch 2.0+ `torch.compile` for improved throughput:

- **Exponential/Logarithmic maps** — Fused element-wise operations
- **Distance computations** — Compiled Lorentzian distance
- **MoE gating** — Compiled softmax operations
- **Hard negative mining** — Compiled norm and margin computations

Compilation is **enabled by default** when PyTorch 2.0+ is available. Configure via:

```python
from naics_embedder.utils.compile import CompileConfig, set_compile_config

set_compile_config(CompileConfig(
    enabled=True,
    mode='reduce-overhead',  # Best for small tensors / repeated calls
    dynamic=True,            # Support varying batch sizes
))
```

**Disable compilation** via environment variable:

```bash
NAICS_DISABLE_COMPILE=1 uv run naics-embedder train
```

**Benchmark compiled vs eager operations:**

```python
from naics_embedder.utils.compile import benchmark_compile_speedup

results = benchmark_compile_speedup(batch_size=256, embedding_dim=768)
print(f"exp_map speedup: {results['exp_map']['speedup']:.2f}x")
print(f"distance speedup: {results['lorentz_distance']['speedup']:.2f}x")
```

---

## Troubleshooting

- **Dataset checks** — Use `uv run naics-embedder tools config` to confirm paths before training.
- **Supervision gate failures** — The message names the artifact and bundle ID; regenerate the
  bundle with `uv run naics-embedder data supervision` and point `supervision.manifest_path` at the
  printed manifest.
- **Flag visibility** — Curriculum phase transitions and flag values are emitted in training logs.
- **Memory pressure** — Lower `data_loader.batch_size` or increase `accumulate_grad_batches` via
  overrides. Each step encodes the full candidate pool (up to `n_candidates` rows per anchor with
  on-the-fly sampling).
- **Compile issues** — If torch.compile causes problems, disable with `NAICS_DISABLE_COMPILE=1`.
