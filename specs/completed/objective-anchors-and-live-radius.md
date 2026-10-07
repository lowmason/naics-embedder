# Objective, anchors and live radius — Design Spec

**Status:** COMPLETE (2026-10-07) — implemented by
`specs/plans/completed/10-objective-anchors-and-live-radius.md`; deferred items in
`specs/deferred_items.md`

**Roadmap:** `specs/naics-embedding-roadmap.md`, Stage 7 (ROUTING: brainstorming), the reference
configuration. Source spec `specs/naics-embedding.md` at d9126ce. Evidence read at origin/main
9a73637, whose `src/` and `conf/` equal c1f9ee5's. A Python path outside `tests/` is under
`src/naics_embedder/`; every other path starts at the repository root, or at `~`.

**Next skill after approval:** `writing-plans` (plan 10) in a fresh Opus session; execution then
runs on the Sonnet default.

## 1. Purpose

Replace the interim six-term objective and its sampling machinery with Req 11's three terms over
all 2,125 codes on a live radius. Wire every within-run selection to the validation query split
(D6), and delete legacy containment (D2). Then fix δ for each of D8's three panels from a 10-seed
sweep of the reference configuration: the text stage after Reqs 7–11, MiniLM behind the shared
encoder, hyperbolic at dimension 16 (Req 5).

## 2. Scope

### 2.1 In scope

- **Req 11.** The task term, the code–code listwise term and the radial term, with learned logit
  scales; every term Req 11 removes.
- **Req 10.** Every code an anchor in every epoch; a per-epoch cache of code points; no
  eligibility rules, pre-drawn tuples, inverse-distance draws or false-negative clustering.
- **Req 13.** A gradient-passing bound in place of the cap; a virtual root at o with the sectors at
  positive radius; one radial coordinate; curvature fixed at 1 with no parameter; a numerical
  resolution check.
- **Req 8(b) and the training side of 8(c).** Activity phrases as queries, each scoring its
  referencing code; no exclusion as a code–code negative; lineal references text only.
- **Req 9.** Unary pairs out of positive supervision (training side).
- **Req 4.** The in-sample loss goes; checkpointing, early stopping and learning-rate control read
  the validation query split (D6).
- **Req 5.** The reference configuration and δ for each of D8's three panels (D10).
- **Req 6.** The text stage's validation computes no structural statistic; `tools investigate` is
  retired.
- **D2 and D5.** Legacy containment and the weights-only migration are deleted; no training path
  reads the relation margin axis.
- **Verification** "No inert terms", "Coverage", "Radius", "Exclusions" (training half), "Text"
  (unary pairs) and "Selection hygiene" (validation half).
- **Deferred items.** Plan 8's epoch item and plan 7's activity-phrase check are discharged; I4, M9
  and M10 retire as mooted (section 10).

### 2.2 Out of scope

- **The Lambda remote workflow.** `specs/lambda-remote-workflow.md` (APPROVED) is planned and built
  in its own cycle. This stage's campaign waits for it (4.6).
- **Stage 8:** the Euclidean and spherical heads, their maps and the nine-cell decision.
- **Stage 9:** the backbone, MoE, channel-presence and IC ablations, and the term-weight decision
  (R7, section 10).
- **Stages 10–11:** HGCN, its curvature, `losses/level_radius.py`,
  `data/positive_sampling.py` (which HGCN's dataset reads) and the HGCN feeder's fate.
- **Stage 12:** the sealed test splits.
- **The bundle contract.** The training-pairs member stays (R10).
- **Multi-GPU training.**

## 3. Rulings

From the session brief, not re-triaged:

- **R1 (D2).** Legacy containment, its second `training_step` and the checkpoint-contract migration
  are deleted.
- **R2 (D5).** Relation names survive only as arm D's edge types and as diagnostic labels; the
  margin axis leaves the training path.
- **R3 (D6).** The validation query split's MRR selects within a run: checkpoint, early stopping
  and learning-rate control. Both panels are used only between configurations, under Req 5.
- **R4 (D8, D10).** Three panels, each with its decision statistic; 95 % non-inferiority and
  98⅓ % superiority intervals.
- **R5.** Over-window channel texts read as their window-fitting summaries (the ruling of
  2026-10-03, shipped by Stage 6b); the reference trains on them.
- **R6.** Stage 6's floor (top-1 0.1249, MRR 0.2301; level-6 R² 0.2591 and 0.2009) was read on
  truncated text after one interim epoch. It is no baseline, and nothing here compares with it.

From the brainstorm, the user's answers on 2026-10-04:

- **R7. Stated defaults.** The reference's free settings are stated here (4.1–4.4), with no
  selection before δ: the term weights, the listwise target's temperature, the radial step, the
  bound, the learning rate and the epoch budget. A change to a term weight is a later one-factor
  decision under Req 5, and that decision is how Req 11's "term weights are set on validation
  splits" is met (section 10).
- **R8. Seeds and margin.** The reference sweep runs 10 seeds, and each panel's δ is 3 times the
  reference's across-seed SD of the panel's decision statistic (`tools margins --multiple 3`).
  Both are fixed here, before the sweep.
- **R9. Platform.** Training runs on Lambda, under CUDA at `bf16-mixed`, D6's monitor reads
  included (4.4). The export, every decision read, the margins, the artifact store and the
  decision reads' selection log stay on the Mac. The choice binds the training of Stages 8–10.
- **R10. The training-pairs member.** It stays in the bundle contract, unread, and bundle 301cce28
  stays canonical.
- **R11. Training structure.** Two-stream steps over a per-epoch code cache (section 9).

## 4. Design

### 4.1 The objective

Notation: d is the arm's distance (hyperbolic at c = 1, computed as in 4.2), r a point's radius,
λ(i) a code's level (2–6), and D* the bundle's tree metric (`structural_distance` in the pair
facts).

**(i) The task term.** A query is a distinct (text, level) pair with a target set T of codes at
that level and a set N of forced negatives.

- **Index entries.** The role table's 7,200 training entries (988 codes) are queries at level 6,
  with T = {the entry's code} and N empty.
- **Activity phrases.** A phrase row's destinations are its named codes that are not lineal to its
  referencing code. The 8 rows whose named codes are all lineal stay text only (Req 8), and
  withheld rows carry no phrase, which leaves 4,521 of the redirection table's 4,529 phrase rows.
  - Rows are grouped by (phrase, destination level). T is the group's destinations at that level,
    and N is its rows' referencing codes, whatever their level.
  - This gives 3,860 queries: 54 at level 2, 97 at level 3, 214 at level 4, 894 at level 5 and
    2,601 at level 6. 19 of them have two or more targets.
- **Merging.** An index entry and a phrase query with the same text at level 6 are one query, whose
  T and N are the unions. 21 merge, for 11,039 task queries in all.
- **Fail closed.** A query whose T and N overlap is refused at load; bundle 301cce28 has none.
- **Loss.** With C = (all codes at the level) ∪ N:

  L_task = mean over the step's queries of −log Σ_{t ∈ T} softmax_C(−s_q · d(q, c))_t

  A cross-reference query therefore always scores its referencing code (Req 8(b)).

**(ii) The code–code listwise term.** For an anchor a, let J_a be all 2,125 codes except a itself
and, for the 522 unary pairs, a's partner (the pair facts' `unary_pair` flag).

- **Target.** p_a = softmax over J_a of −D*_{aj} / τ_t.
- **Loss.** L_cc = mean over the step's anchors of CE(p_a, softmax over J_a of −s_c · d(a, j)).
- **Unary pairs** are neither positives nor negatives. Masking the partner keeps a unary pair out of
  positive supervision (Req 9) without making it a negative.
- **No exclusion data.** The term never reads the exclusion table, so no exclusion pair can act as
  a code–code negative (Req 8(c)).

**(iii) The radial term, in the hyperbolic arm only.**

  L_rad = mean over the step's anchors of (r_a − ρ · (λ(a) − 1))²

The virtual root sits at o, the sectors at r = ρ and the six-digit codes at r = 5ρ.

**Total.** L = L_task + w_c · L_cc + w_r · L_rad. Under `model.fusion: moe` only, the experts'
load-balancing term is added with `model.moe.load_balancing_coef`, as now.

**Stated defaults (R7):**

| Setting | Default | Config key |
|---|---|---|
| w_c, w_r | 1, 1 | `loss.code_code_weight`, `loss.radial_weight` |
| τ_t | 1 | `loss.target_temperature` |
| ρ | 1, fixed | `loss.radial_step` |
| s_q, s_c | learned: s = exp(θ), starting at 1, clamped to [0.01, 100], no weight decay | `loss.logit_scale_init`, `loss.logit_scale_range` |

### 4.2 Head, radius and precision

**The head.** `HyperbolicHead` (`text_model/hyperbolic.py:84-123`) loses its cap of 2 and its
curvature argument.

- From the projection's output v: ν = ‖v‖, r = R · tanh(ν / R) and û = v / ν, guarded at ν = 0.
  The tangent at o is r · û, which the export writes as now, and the point is exp_o(r · û) at
  c = 1. The export's `coordinates` provenance (`text_model/export.py:45`) names the bounded
  tangent instead of the capped one; nothing checks that field.
- dr/dν = sech²(ν / R) > 0 everywhere, so the bound passes gradient to the radius (Req 13).
  - With R = 8 (`model.radius_bound`), a six-digit code at its target r = 5 keeps dr/dν ≈ 0.61.
  - The cap's saturated points passed about 10⁻⁷ (the roadmap's Req 13 row, by probe).

**One radial coordinate.** r, the geodesic distance from o, is used in every loss, target, log and
diagnostic of the text stage. The text stage stops using x₀ = cosh r (`compute_hyperbolic_radii`,
`text_model/hyperbolic.py:247-259`) and sinh r (`losses/level_radius.py`, which HGCN keeps until
Stage 10, D3).

**Curvature.** Fixed at 1, with no parameter anywhere in the text stage.

- `loss.curvature` goes, as do the `curvature` arguments of the text model, the encoder, the head
  and the loss. Every formula is written for c = 1. The Lorentz ops in `text_model/hyperbolic.py`
  keep theirs, because HGCN calls them.
- `require_unit_curvature` (`text_model/export.py:121`), the export's and the arm encoder's guard
  on c = 1 (plan 8's R8), has nothing left to guard and is deleted. The contract's `objective`
  (4.5) refuses older checkpoints instead.

**Distance.** Training computes every distance from (r, û), never from Lorentz coordinates:

  d(x, y) = 2 · asinh(√(sinh²((r_x − r_y) / 2) + sinh r_x · sinh r_y · ‖û_x − û_y‖² / 4))

- It equals arcosh(−⟨x, y⟩_L) exactly: it is the hyperbolic law of cosines in half-angle form.
  Every term is non-negative, so nothing cancels, and float32 resolves it at every radius the bound
  allows.
- ‖û_x − û_y‖² comes from explicit differences, not from 2 − 2 û_x · û_y, which cancels at small
  angles. At a step's 25 × 2,125 × 16 the differences are cheap.
- The square root is guarded at zero, so the value and its gradient stay finite.
- The reads keep the scorer's float64 `lorentz` distance (`panels/decoding.py`). A test pins the two
  forms' agreement up to r = R.

**Precision.**

- `train` honors `training.trainer.precision` on CUDA.
  - Today `get_device` hardcodes `16-mixed` there (`utils/backend.py:33`), and `train` and
    `create_trainer` pass that value on (`cli/commands/training.py:648`, `utils/training.py:350`).
    So the key is never read.
  - Off CUDA the trainer keeps `32-true`. The shipped default becomes `bf16-mixed` (R9).
  - The validator's accepted values (`utils/config.py:1049`) stay as they are, so no run can
    select a true-half precision such as `bf16-true`.
- Under autocast only the backbone runs in reduced precision. Everything from the projection on
  runs in float32 with autocast disabled: the projection, the head, every distance, every loss term
  and the logit scales. The monitor (4.4) encodes in float32 as well.

**Verification "Radius".** It is checked on each trained seed's selected checkpoint (4.6), through
a reusable `radius_report` (section 6):

- On a real batch, ∂L/∂r_a is nonzero for every anchor.
- At every level, the SD of r exceeds 10⁻³. In the capped run, x₀ had an SD of 4 · 10⁻⁷ (C2 in
  `specs/naics-embedding-review-claude.md`).
- The 20 sector radii are positive and pairwise distinct, and their least pairwise distance is
  reported.
- At the largest radius observed:
  - the float64 Lorentz points the reads use satisfy |⟨x, x⟩_L + 1| ≤ 10⁻⁹ · x₀²;
  - over every pair of the seed's 2,125 codes, the float32 training form agrees with the float64
    form within 10⁻³ relative error.

### 4.3 Data path and the epoch

**What one epoch reads:** every code once as an anchor and every task query once.

- Each epoch draws a permutation of the 2,125 codes and one of the 11,039 queries, both seeded from
  (seed, epoch).
- Both permutations are cut into S = ⌈11,039 / 128⌉ = 87 near-equal chunks, of 126 or 127
  queries and of 24 or 25 codes. A step is one code chunk plus one query chunk.
- No pre-drawn tuple, candidate pool or sampler exists.
- `data_loader.n_epochs` is deleted, so a Lightning epoch and a data epoch are the same thing, and
  `training.trainer.max_epochs` counts these epochs.
  - This discharges plan 8's item. Its Done-when asks the Lambda config to set
    `data_loader.n_epochs` to match what one epoch reads, and no such key remains to set.
- `data_loader.queries_per_step` (default 128) is the one batch knob: the most queries a step
  reads, which sets S = ⌈queries / queries_per_step⌉, and the chunks follow from S.
  `data_loader.batch_size` and `val_split` go, and `training.trainer.accumulate_grad_batches` is 1.

**The cache.** The model holds the 2,125 codes' (r, û) in codebook order.

- It is refreshed at fit start and after each epoch's last step: every code, in eval mode, without
  gradient, in float32, in chunks. The end-of-epoch refresh also feeds that epoch's monitor (4.4)
  and serves the next epoch.
- Within a step, the step's anchors replace their rows with their live points, so gradient flows
  through them both as anchors and as candidates. Every other row is a constant.
- The code–code term and the task term both read it.
- It is not saved in checkpoints: exact resume rebuilds it from the restored weights.

**Inputs.**

- Code chunks come from the tokenization cache through `stack_text_inputs`, with code ids and
  levels.
- Task queries come from two validated bundle members, the role table's training entries and the
  redirection table.
  - They are built once at setup and tokenized once, as `query:` texts at the 128-token window.
  - Each carries its level, its target ids and its forced-negative ids.
- There is no validation dataloader. The in-sample validation rows go (Req 4), and validation is the
  monitor.

**Plan 7's activity-phrase check lands before any phrase is read.**

- `activity_phrase` (`data/redirections.py:62`) moves to a torch-free module that `supervision/`
  can import.
- `validate_redirection_table` (`supervision/artifacts.py:439`) recomputes every non-withheld
  row's phrase by the build's rule (`data/redirections.py:119`: cross-reference rows that name a
  code) and refuses a mismatch. It is verified against bundle 301cce28 (section 7).
- It raises at load and adds no key to `REQUIRED_VALIDATION_RESULTS`, which would make bundle
  301cce28 unloadable.

### 4.4 Selection

**D6's monitor** runs once per epoch, after the end-of-epoch refresh.

- An adapter implements `QueryCodeEncoder` (`panels/outcome.py:50`) for the live model.
  - Queries go through the model in eval mode, in float32 with autocast disabled.
  - Code rows come from the refreshed cache.
  - Both pass through `exp_map_origin` (`text_model/arm_encoder.py:43`) in float64, as
    `ArmEncoder` does.
- It is scored through `OutcomePanel.score` (`panels/outcome.py:196`) on the validation split under
  `lorentz`. That is the path every read takes, so the read is logged. The panel comes from
  `OutcomePanel.from_bundle` and `conf/data/outcome_panel.yaml`'s `selection_log`, which on Lambda
  is the instance's log.
  - The record names the training run's id. A fresh start mints it; every checkpoint saves it,
    and exact resume restores it, so a run that spans instances keeps one id.
  - It also names the seed, the epoch, and the epoch's code table by `matrix_fingerprint`.
- The MRR is logged as `val/outcome_mrr`.

**What the monitor drives (R7's defaults):**

| Control | Setting |
|---|---|
| Checkpoint | `ModelCheckpoint` on `val/outcome_mrr`, mode max, `save_top_k=1`, so the earliest epoch with the highest MRR is kept (a later epoch replaces it only by beating it); plus `last.ckpt` for exact resume |
| Early stopping | `val/outcome_mrr`, mode max, patience 5 |
| Learning rate | AdamW at 1e-4, weight decay 0.01 (none on the logit scales); linear warmup over the first epoch, then `ReduceLROnPlateau` on `val/outcome_mrr` (mode max, factor 0.5, patience 2) |
| Budget | `training.trainer.max_epochs` 40, of 87 steps each |

These replace the monitors on `val/contrastive_loss` (`cli/commands/training.py:604-619`,
`utils/training.py:304-316`, `text_model/mixins/optimizer.py:107-133`).

**A durable carrier for monitor reads.**

- `run_seed_sweep` mints its `run_id` only after the runner returns (`decision/sweep.py:140-143`),
  so today a training run's own reads never reach a record.
- Each run appends its monitor records, as logged, to `monitor_reads.jsonl` in its checkpoint
  directory. `remote sync` pulls that file home with the checkpoints.
- Exact resume continues the file it resumed with (4.6). It keeps the records through the restored
  checkpoint's epoch; a record past it, left by an interrupted segment, stays only in that
  segment's selection log. So the file holds each epoch of the surviving run exactly once.
- `SeedArtifacts` (`decision/sweep.py:52`) and `SeedRun` (`decision/records.py:113`) gain
  `monitor_records`, which defaults to empty.
- `check_arm` (`decision/decide.py:135`) requires a run's monitor records to be outcome-panel
  validation reads that name its training id. Their earliest epoch with the highest MRR must be its
  checkpoint's epoch.
- `check_margins_first` (`decision/decide.py:236`) takes each run's earliest read, monitor reads
  included.
- So every validation read that selected anything reaches Stage 12 inside a decision record.

**Agreement.** In a CPU test, an epoch's monitor MRR equals `read_outcome_validation`
(`text_model/arm_encoder.py:242`) on that epoch's exported checkpoint. On real checkpoints the
two agree within 10⁻³ (section 7), since batch shapes can move float32 near-ties.

**Req 6's leftovers.**

- The text stage's validation computes no structural statistic. `EmbeddingEvaluator`,
  `EmbeddingStatistics`, `HierarchyMetrics` and the clustering hook leave
  `text_model/naics_model.py`, `text_model/mixins/validation.py` and `text_model/mixins/logging.py`.
  HGCN keeps its own until Stage 11.
- `tools investigate` and `tools/_investigate_hierarchy.py` are retired, so Req 6's statistics come
  only from `tools diagnostics`.
- Each epoch logs health values that nothing selects on: each term's value, the two logit scales,
  and r's mean and SD per level. `tools visualize` follows the new epoch summary.

### 4.5 Contracts, config and deletions

**The checkpoint contract** (`supervision/checkpoints.py:99`).

- It drops `supervision_mode`, `mining_contract_version` and `structural_preference_loss_version`,
  with their constants in `supervision/schema.py`. It gains `objective`, `'req11-v1'`, which names
  the three terms, the radial form and the bound.
- A contract saved before Stage 7 parses tolerantly and reads as the old objective. Exact resume,
  the export, the outcome read and the HGCN feeder refuse it, with a message that nothing migrates
  (D2).
- The encoder record and `summaries` stay. `contract_version` is still copied from the manifest,
  so bundle 301cce28 stays valid.

**D2.**

- Legacy containment goes: `_legacy_containment_training_step`
  (`text_model/naics_model.py:649-726`), `containment_contract`, `SupervisionModePolicy`
  (`supervision/mode.py`), `legacy_token_fingerprints`, `_collate_legacy`,
  `announce_legacy_containment` and `supervision.mode`.
- The weights-only migration goes: `load_weights_only`, `MigrationReport`, `log_migration_report`
  and `--checkpoint-load-mode weights_only`. Exact resume stays.

**Deleted with the old objective** (Req 11's removals and Req 10's):

| Area | Deleted |
|---|---|
| Terms | DCL (`HyperbolicInfoNCELoss`), distance matching (`HierarchyPreservationLoss`), pairwise preference (`StructuralPreferenceLoss`), the radius penalty, the logged margin (`NormAdaptiveMargin`) |
| Machinery | `text_model/curriculum.py`, `hard_negative_mining.py`, `hyperbolic_clustering.py` and `false_negative_strategies.py`; `text_model/mixins/curriculum.py` and `distributed.py` |
| Supervision | `supervision/selection.py` (the coordinator), `supervision/candidates.py`, `supervision/margins.py`, `SelectionReason` (`supervision/schema.py`) and its `quota_selections` counter (`text_model/mixins/logging.py`) |
| Data | the pre-sampled rows and their multi-epoch caches; `RepairedMapDataset`, `RepairedPhase1Dataset`, `NAICSMapDataset` and `Phase1MapDataset`; `build_candidate_pool`, `sample_raw_candidates`, the inverse-distance weights and Phase 1's sibling mask; `text_model/dataloader/difficulty_sampler.py`; the collates' candidate fields |

Rewritten: `text_model/loss.py` (the three terms), `text_model/naics_model.py` (one training
step), `text_model/dataloader/datamodule.py` (the two streams), `text_model/mixins/loss.py`,
`logging.py` and `validation.py`, and the scheduler in `optimizer.py`.

**D5.** No training path reads the relation margin axis. The training-pairs member stays (R10), so
the bundle build's `_structural_margins` (`data/create_triplets.py:139-172`) keeps writing its
margin columns. They leave at the next contract bump, together with the member (section 10).

**Stays.**

- `losses/level_radius.py` (HGCN, D3) and `data/positive_sampling.py` (HGCN's dataset).
- The HGCN feeder `generate_embeddings_from_checkpoint`, and `train`'s question about it (Stage 11).
  `train` asks only when stdin is a terminal. Today's `typer.confirm` would abort on an empty stdin
  and wait in a tmux pane, which is a terminal, so the remote launch reads stdin from `/dev/null`
  (4.6).
- Under `moe`, the experts and their load-balancing term, the one extra term. Router-guided mining
  goes with the rest of mining.

Training refuses `devices > 1`, because the cache is per process.

**Config.** Every key is declared in the Pydantic models (`utils/config.py`).

- **Removed:**
  - `supervision.mode`;
  - the `curriculum`, `sampling` and `false_negatives` sections;
  - `data_loader.batch_size`, `n_epochs` and `val_split`;
  - in `data_loader.streaming`, `seed`, the sampling keys from `n_negatives` on, and the legacy
    paths `distances_parquet`, `distance_matrix_parquet`, `relations_parquet` and
    `triplets_parquet`, with their readers (`cli/commands/training.py`, `utils/validation.py`);
  - `model.eval_sample_size`, `eval_every_n_epochs`, `parent_eval_top_k` and `child_eval_top_k`;
  - `loss.temperature`, `curvature`, `base_margin`, `hierarchy_weight`, `structural_preference`,
    `radius_reg_weight` and `level_radius_weight`;
  - `training.use_warmup_cosine` and `warmup_steps`.
- **Added:**
  - `data_loader.queries_per_step: 128`;
  - `model.radius_bound: 8`;
  - `loss.code_code_weight: 1`, `radial_weight: 1`, `target_temperature: 1`, `radial_step: 1`,
    `logit_scale_init: 1` and `logit_scale_range: [0.01, 100]`;
  - `training.warmup_epochs: 1`, `lr_plateau_factor: 0.5`, `lr_plateau_patience: 2` and
    `early_stopping_patience: 5`.
- **Changed:** `training.trainer.max_epochs: 40`, `accumulate_grad_batches: 1` and
  `precision: bf16-mixed`; `experiment_name: reference`, in place of `sadc_default`.
- The `supervision.manifest_path` line is left as it is, so the held `config` commit still replays
  (section 10).

### 4.6 The campaign

**Prerequisite: the remote workflow.** `specs/lambda-remote-workflow.md` (APPROVED) gets its own
writing-plans cycle. It can be built beside this stage's code: its modules are new
(`cli/commands/remote.py`, `remote/`, `conf/remote.yaml`), and where both edit the same docs the
later merge reconciles them. Whichever plan is written second reads the other's merged state,
above all the checkpoint contract (4.5): the remote spec's resume pre-check names the loss and
mining versions this stage deletes. The campaign starts once both have landed, and it needs six
things from that plan:

1. Fresh `naics-embedder train` runs with `key=value` overrides, one per seed, each under its own
   `experiment_name`, launched in tmux with stdin from `/dev/null` (4.5, "Stays").
2. Pulls that bring each run's checkpoint directory home (every checkpoint it kept, `last.ckpt`
   and `monitor_reads.jsonl`), along with the instance's logs.
3. An NTP-synchronized instance clock, checked at bootstrap, because monitor reads' times enter
   `check_margins_first`.
4. Nothing for the decision reads: no QCEW slices, artifact store or decision read on the
   instance. The monitor needs only the bundle, already uploaded, and the pushed
   `conf/data/outcome_panel.yaml`. Its selection log comes home under `logs/remote/<session_id>/`,
   never into the Mac's own log.
5. No weights-only start. The remote spec's three mentions of weights-only loading go stale with
   D2: its out-of-scope note, its note on bundle rebuilds and its bundle gate's message.
6. Exact resume that uploads `monitor_reads.jsonl` with `last.ckpt`. Pulls add or update files,
   so a resumed instance's fresh file would otherwise replace the Mac's complete one.

**Training, on Lambda (R9).**

- Seeds 1–10, each `naics-embedder train seed=<s> experiment_name=stage7-reference-s<s>`.
- The runs use the shipped defaults (4.1–4.5), on CUDA at `bf16-mixed`.
- Each run keeps its monitor-selected best checkpoint and `last.ckpt`.

**Reads and records, on the Mac.**

- **`CheckpointRunner`** loads seeds; it never trains. For each seed it:
  - checks that the run's monitor records cover each epoch through `last.ckpt`'s exactly once;
  - takes the checkpoint of the run's earliest epoch with the highest monitor MRR, by its epoch,
    since pulls never delete and a resumed run can leave an earlier best beside a later one;
  - checks the checkpoint's contract, its seed and the arm's settings;
  - exports the table on the Mac (`text_model/export.py`);
  - returns `SeedArtifacts`, with `ArmEncoder.from_files` and the monitor records.
- **`tools sweep`**, a new command, builds the `ArmSpec` from the config, runs `run_seed_sweep`
  (`decision/sweep.py:111`) and writes the arm record.
  - It runs with the runner, the store, and plan 9's text-only table
    (`checkpoints/plan9_exit/text_only.parquet` in the main checkout). That table's provenance
    names MiniLM at revision 1110a243, descriptions `fe8c54e3…`, summaries `dd425eb5…` and window
    128.
  - The spec is named `reference`, with one component, dimension 16, hyperbolic geometry and
    MiniLM's D9 fields. Its settings are R7's defaults, plus the accelerator and the precision.
- **`tools margins --multiple 3`** fixes each panel's δ from the 10-seed record (R8).
- **Where things live.**
  - The store is `~/naics-artifacts`, outside every worktree.
  - The records are `~/naics-artifacts/records/stage7/reference.json` and `margins.json`.
  - The Mac's selection log gains the 30 decision reads.

**Lock and clock.**

- `uv.lock` is frozen from the first reference training run through the last decision that uses
  these margins (Stages 8–10). Training changes with torch, and `decide` pairs arms only on
  bit-identical panel data. A lock change in that window means a new reference sweep and new
  margins, decided with the user.
- Every decision read runs on the Mac, under one clock. The reference's runs are exempt from the
  margins-first check, and every later arm trains after the margins are fixed.

**The finding.** `specs/findings/reference-configuration.md` is committed and records:

- the runs: instance, GPU, epochs trained and selected, and durations;
- each seed's three statistics, and each panel's SD and δ;
- the record files' sha256;
- copies of the monitor and decision log records;
- each seed's "Radius" and "No inert terms" numbers;
- `tools diagnostics` per seed, for the record only (Req 6).

It draws no comparison with Stage 6's floor (R6).

## 5. Error handling

Named refusals, all ValueError:

- **Queries:** a query whose T and N overlap.
- **Bundle:** a non-withheld redirection row whose phrase differs from its text's (4.3).
- **Config:**
  - a removed key (the models forbid extras);
  - `devices > 1`;
  - `queries_per_step` below 1;
  - a weight below 0;
  - a target temperature, radial step or radius bound at or below 0;
  - a logit-scale range that is empty or not positive.
- **Checkpoints:** a contract whose `objective` is not `req11-v1`, at exact resume, the export, the
  outcome read and the HGCN feeder, citing D2.
- **Records:**
  - monitor records that are not outcome validation reads, that name another training run, or
    whose earliest epoch with the highest MRR is not the selected checkpoint's (`check_arm`);
  - a read before the margins' `fixed_at`, monitor reads included (`check_margins_first`).
- **Runner:** a seed whose checkpoint's seed, settings or contract differ from the arm's; whose
  run has no `monitor_reads.jsonl`, or records that miss or repeat an epoch through `last.ckpt`'s;
  or whose selected epoch has no checkpoint on the Mac.

Guarantees rather than refusals:

- The polar distance and its gradient stay finite at zero separation.
- Exact resume needs no cache state, because the cache is rebuilt at fit start.
- `train` never waits for input when stdin is not a terminal.

## 6. Testing

Tests are written red to green, on the tiny `BertConfig` backbone
(`tests/fixtures/shared_encoder.py`), so CI downloads nothing.

**Exit criteria (CI):**

- **Task term:**
  - the summed-probability loss over T;
  - C is the level's codes plus N, and every cross-reference query scores its referencing code
    (Verification "Exclusions");
  - a T–N overlap fails closed;
  - lineal and withheld rows never become queries;
  - an index entry and a phrase with the same text merge.
- **Code–code term:**
  - changing the exclusion data leaves the loss bit-identical (Verification "Exclusions");
  - the anchor and its unary partner are masked (Verification "Text");
  - the target is softmax(−D* / τ_t).
- **No inert terms.** On a batch, the three terms and both logit scales get nonzero gradient.
  Under `moe`, so does load balancing.
- **Coverage (Verification "Coverage"):**
  - over one epoch, each code is an anchor exactly once and each query is read once;
  - every step scores each anchor against every code except itself and its unary partner;
  - no candidate pool exists.
- **Head and distance:**
  - at ν = 20 the gradient with respect to ν is nonzero, and r ≤ R;
  - the polar form equals the float64 arcosh(−⟨x, y⟩_L) up to r = R;
  - the float32 form stays within tolerance of float64;
  - gradients stay finite at the zero guard.
- **Precision:**
  - under CPU bf16 autocast, the projection, the head, the distances and the losses stay float32;
  - on CUDA, `train` passes the configured precision (a stubbed device check).
- **Cache:**
  - refreshes happen at fit start and at epoch end, in eval mode, without gradient;
  - only in-batch rows carry gradient;
  - exact resume rebuilds the same cache.
- **Monitor and records:**
  - each epoch's read is logged as an outcome validation read with the training id, seed, epoch
    and table;
  - the agreement test (4.4);
  - `monitor_reads.jsonl` is written;
  - exact resume keeps the training id and continues the file, dropping a record past the
    restored epoch, so each epoch appears once;
  - the kept checkpoint is the earliest epoch with the highest MRR, ties included;
  - `check_arm` and `check_margins_first` refuse what section 5 lists.
- **Contracts and CLI:**
  - a pre-Stage-7 checkpoint is refused everywhere, with the objective message;
  - `devices > 1` is refused;
  - the config's removed and added keys validate as 4.5 says;
  - `train` does not prompt without a terminal;
  - `tools investigate` is gone;
  - `tools sweep` runs end to end on fixture checkpoints.
- **The phrase check.** A rehashed fixture table with a wrong phrase is refused. Bundle 301cce28
  is not in CI, so its acceptance is checked in Phase 1 (section 7).
- **`radius_report`.** On fixture tables it reports each quantity Verification "Radius" names, and
  it fails a capped table, where all radii are equal.

**Supporting tests:**

- the epoch permutations depend only on (seed, epoch);
- the warmup and plateau schedule;
- the health logs;
- `tools visualize` on the new summary.

## 7. Exit procedure

**Phase 1: the code, ending at the hard checkpoint.** In a worktree:

1. Build and test the code (section 6).
2. Clone bundle 301cce28 and `data/naics_descriptions.parquet` from the main checkout with
   `cp -cR`, never symlinking or rebuilding them. Pass `supervision.manifest_path` as an override
   on every command. Load the bundle once through the new phrase check, which must accept it
   (plan 7's Done-when).
3. Run a smoke run locally on MPS at `32-true` and the defaults: 3 epochs.
   - Record the epoch time.
   - Check that the monitor reads are logged and that `monitor_reads.jsonl` is written.
   - Export the best checkpoint.
   - Check that its epoch's monitor MRR and `read_outcome_validation` on the export agree within
     10⁻³, and "No inert terms" on a real batch with the real MiniLM.
4. Open the PR with these numbers. Its review and merge are the hard checkpoint: no Lambda time is
   spent before it. Before the worktree goes, append its smoke reads to the main checkout's
   selection log (append-only) and copy them into the PR description.

**Phase 2: the campaign,** from the main checkout, once the remote plan has landed (4.6):

5. Train seeds 1–10 on Lambda, pull every run home, and finish the session. Phase 1 never ran
   `bf16-mixed` on CUDA, so check seed 1 after its first epoch before launching the rest: the
   trainer's precision, a logged monitor read and `monitor_reads.jsonl`.
6. On the Mac, run `tools sweep`, then `tools margins --multiple 3`.
7. Run `radius_report` and the "No inert terms" check on each seed's selected checkpoint.
8. Check that every campaign log record, monitor and decision alike, is a validation read, and
   that none opens a test split.
9. Write the finding. A second PR commits it, and ticks and stamps Stage 7.

**Exit:**

- the margin record holds a δ for each of D8's three panels, at 3 SD over 10 seeds;
- every seed passes "Radius" and "No inert terms";
- the selection-log records show validation reads only;
- the text stage's validation computes no structural statistic.

## 8. Deletions and documentation

**Deleted:** everything 4.5 lists, plus `tools investigate`, the curvature guard (4.2), the
in-sample validation rows and the monitors on `val/contrastive_loss`.

**Documentation:**

- `CLAUDE.md` and `README.md`: the architecture text (the objective, the curriculum, false
  negatives, DCL, the mixin table, the curvature notes, the epoch and the commands).
- `docs/text_training.md`, rewritten around the three terms, the cache, the monitor and the
  campaign.
- `docs/usage.md`: `train`'s options and precision, `tools sweep`, and the removal of
  `tools investigate`.
- `docs/api/`: the pages of deleted and new modules, and `docs/.nav.yml`.

`uv run mkdocs build --strict` must pass locally, since PR CI never builds the docs.

## 9. Chosen approach and rejected alternatives

- **Training structure.**
  - Chosen: two-stream steps over a per-epoch code cache (R11). One epoch is one pass over the
    codes and the queries together, in 87 steps, with every code an anchor once.
  - A probe on MPS on 2026-10-04 measured about a minute per epoch:
    - refresh, 3.6 s;
    - a 32-anchor step, 0.41 s;
    - a 128-query step, 0.26 s;
    - the validation queries, 0.8 s;
    - peak memory, 3.3 GiB.
  - Rejected: alternating code and query steps, because the step ratio becomes a hidden term
    weight.
  - Rejected: query-led code batches. Codes with many queries would be anchors many times per epoch,
    and the 517 codes that no query targets would need padding to be covered.
- **Cache.**
  - Chosen: one refresh per epoch, with in-batch rows live.
  - Rejected: re-encoding all codes every step, at 3.6 s against a 0.4 s step.
  - Rejected: a momentum encoder, a second backbone for a bank that refreshes in seconds.
  - Rejected: writing detached live rows back every step. It mixes rows from different weights
    within an epoch, and Req 10 names a per-epoch cache.
  - Rejected: saving the cache in checkpoints, which would be redundant with the weights.
- **Epoch.** Rejected: keeping `n_epochs` as passes per Lightning epoch. Two notions of an epoch
  are how plan 8's first run came to read 100 epochs in one.
- **Radial term.**
  - Chosen: a squared error to ρ(λ − 1), with ρ fixed.
  - Rejected: a learned ρ. In hyperbolic space the listwise term gains from expanding the
    configuration, so ρ would drift to the bound and recreate the saturated cap.
  - Rejected: a parent-before-child ordering hinge, which places the sectors at positive radius
    only through a margin.
  - Rejected: the virtual root as an extra listwise row. It is shift-invariant in r, so it cannot
    hold the sectors off the origin.
- **Bound.**
  - Chosen: R · tanh(ν / R).
  - Rejected: no bound, since nothing would stop the radii from outgrowing the float64 reads.
  - Rejected: a learned R, which brings back the expansion.
- **Distance.**
  - Chosen: the polar form, in float32.
  - Rejected: Lorentz inner products in float32. At r = 8 the products reach about 2.2 · 10⁶, where
    float32 resolves only 0.25, so small distances blur to about 0.7.
  - Rejected: float64, which MPS does not have.
  - Rejected: a bf16 head.
- **Queries.** Rejected: one query per cross-reference row. Generic phrases that sit on up to seven
  referencing codes would be weighted by their row count.
- **Selection.**
  - Rejected: monitoring a sample of the validation queries. It saves under a second and adds noise
    to D6's statistic.
  - Rejected: keeping monitor reads only in the instance's log. That log is gitignored, and
    Stage 12 needs the reads in the records.
- **Contracts.**
  - Rejected: accepting old contracts once `supervision_mode` is gone. A refusal that names the
    objective is the only honest answer for a checkpoint whose objective no longer exists.
  - Rejected: bumping `contract_version` for the objective. It is copied from the manifest, so the
    bump would invalidate bundle 301cce28.
  - Rejected: a contract bump that removes the training-pairs member (R10).
- **Multi-GPU.** Rejected: keeping the distributed mixin. A per-rank cache needs synchronized
  refreshes, for a speed-up that one GPU does not need.
- **The campaign.**
  - Rejected: a training runner on the Mac (R9).
  - Rejected: decision reads on the instance. They would need the QCEW slices, the store and the
    decision log there, and would tie that platform's numerics to every later decision. The
    monitor's reads select within a run only, so they stay with training.
  - Rejected: a fresh text-only table per campaign. Plan 9's table matches every field D9 checks.
  - Rejected: one PR spanning the campaign, which holds the code review hostage to compute.
  - Rejected: verification by uncommitted scripts alone. "Radius" recurs in every hyperbolic arm
    through Stage 10.
- **Platform.** Chosen by the user: Lambda at `bf16-mixed` (R9). The other options were the Mac at
  `32-true` and Lambda at `32-true`.
- **Reference settings.**
  - Chosen: stated defaults (R7).
  - Rejected: a single-seed pilot on the validation MRR, which would specialize the embedding
    toward retrieval.
  - Rejected: a single-seed pilot on all three panels, which selects on point estimates without
    intervals, the practice Req 5 replaced.
- **Seeds and margin.**
  - Chosen: 10 seeds and δ = 3 SD (R8).
    - With 5 seeds a side, seed noise alone puts the paired interval's half-width near 1.24 SD
      (1.07 SD against a 10-seed reference), before any unit-resampling variance.
    - Smaller multiples leave non-inferiority unattainable, and the tie order then decides nearly
      everything.
    - The SD's 95 % interval is 0.60–2.87 times the estimate at 5 seeds and 0.69–1.83 at 10.
  - Rejected: 10 seeds at 2 SD, and 5 seeds at 2 or 3 SD.

## 10. Rollout note

> Roadmap: specs/naics-embedding-roadmap.md, Stage 7 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**The switch happens at Phase 1's merge.** Main then trains only the three-term objective. Exact
resume, the export, the outcome read and the HGCN feeder refuse every checkpoint saved before it,
plans 8 and 9's included, and nothing migrates (D2). The tokenization cache is unchanged.

**For the roadmap's next resume.** These are routing notes only; the resume edits the roadmap.

- **R7's term-weight decision** belongs with Stage 9's one-factor ablations, made from the selected
  cell under Req 5.
- **R10.** The training-pairs member stays in contract v2, unread, against the Stage 5 entry's
  "until Stage 7 removes it". D5's margin axis stays only in the build's generation of that
  member. Both leave at the next contract bump, which plan 7's tokenizer-revision entry already
  watches for.
- **R9.** Stages 8–10 train on Lambda at `bf16-mixed` and read on the Mac. The remote workflow's
  plan precedes this stage's campaign and supplies 4.6's six items. `uv.lock` is frozen from the
  first reference training run through the last decision that uses these margins.
- **Stage 8's entry** cites the export's and the reads' refusal of c ≠ 1 (plan 8, R8). After this
  stage no curvature exists to refuse.
- **Stage 12** reads the monitor reads from `SeedRun.monitor_records`.

**Held commits.** The held `config` commit's hunk context includes the `supervision.mode` line,
which this stage removes, so its next replay resolves one conflict there. The `graph config`
commit edits only `conf/graph.yaml`, which this stage leaves alone.

**Deferred items** are handled through /deferred at plan completion:

- Plan 8's epoch item is discharged (4.3).
- Plan 7's activity-phrase check is discharged (4.3).
- I4, M9 and M10 retire, mooted by the deletions (4.5).
- Two of plan 8's Review Minor items are partly mooted, and the rest of each stays open:
  - the export's untested branches. Its curvature branches leave with the guard (4.2), and its
    cap check (`tests/unit/test_export.py:174`) becomes a check that the table holds the head's
    bounded tangent.
  - the docs. Its five c = 1 statements become exact, since no curvature is configurable, and its
    weights-only heading goes with D2 (section 8).
- These stay open:
  - plan 8's `_pool_present` item. bf16-mixed keeps float32 master weights, so its true-half
    trigger has not fired.
  - plan 7's tokenizer revision, which waits for a contract bump.
  - the c = 1 watch item. Its text-model half can no longer fire, and its HGCN half stays.

**Model routing.** writing-plans for plan 10 runs in a fresh Opus session from this spec;
execution runs on the Sonnet default.
