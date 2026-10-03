# Shared encoder and low-dimensional projection — Design Spec

**Status:** IN REVIEW (2026-10-03). Each design section was approved in the brainstorm; the written
spec awaits the user's review.

**Roadmap:** `specs/naics-embedding-roadmap.md`, Stage 6 (ROUTING: brainstorming). Source spec
`specs/naics-embedding.md` at d9126ce. Evidence read at origin/main ea2e09f. Paths are under
`src/naics_embedder/` unless they start with `conf/`, `tests/`, `specs/` or `docs/`.

**Next skill after approval:** `writing-plans` (plan 8) in a fresh Opus session; execution then
runs on the Sonnet default.

## 1. Purpose

Replace the four-copy LoRA encoder and its mixture-of-experts fusion (`text_model/encoder.py`)
with one shared encoder: field markers, masked fusion, one affine map to a configurable dimension,
and a query path through the same layers. Add a standalone export of the 2,125-code table in
Req 2's form, and take the first live readings on both panels' validation splits. The current
six-term objective and its dataloader stay as an interim training harness until Stage 7 replaces
them.

## 2. Scope

### 2.1 In scope

- Req 14, encoder half: one shared backbone with one LoRA adapter, field markers, masked fusion,
  and queries through the same encoder as a single field. MoE survives as an ablation option only.
- Req 12, projection and dimension half: exactly one affine map; dimension 8, 16 or 32, default 16.
- Req 9, model side: absent channels are masked out of fusion, with no placeholder.
- Req 16, export half: a standalone export command for the 2,125-code table.
- Verification "Text": no absent channel contributes to fusion.
- A checkpoint-contract record of the encoder architecture.
- Two deferred items: plan 6's Stage 6 item (discharged) and plan 4's scorer item (retired).

### 2.2 Out of scope

- **Stage 6b:** window-fitting summaries of over-long channel texts (section 10).
- **Stage 7:**
  - the objective, anchors and live radius (the norm cap stays here);
  - removing the curvature parameter;
  - selection monitors and δ;
  - the Lambda workflow, building on `specs/lambda-remote-workflow.md`.
- **Stage 8:** the Euclidean and spherical heads.
- **Stage 9:**
  - the backbone choice (MiniLM at revision 1110a243 until then);
  - the MoE and channel-presence ablations;
  - other windows.
- **Stage 10:** HGCN at the text stage's dimension (arm D's input).
- **Unchanged here:** the D9 text-only comparator and the lexical baseline.

## 3. Rulings

From the session brief, not re-triaged:

- **R1.** The shared encoder lands in Stage 6, before the objective.
- **R2.** The backbone stays all-MiniLM-L6-v2 at revision 1110a243 until Stage 9.
- **R3.** Fusion options are masked mean, attention pooling and MoE, MoE as an ablation only. The
  per-channel adapter copies and the load-balancing term go with the default fusion.
- **R4.** Dimension is configurable in {8, 16, 32}, the Exit runs 16, and exactly one affine map
  sits between encoder and point.
- **R5.** The current objective and dataloader serve unchanged as an interim harness until Stage 7,
  apart from the two named deviations R10 and R11.
- **R6.** A hyperbolic export writes Req 2's form: tangent coordinates at the origin, without the
  zero time coordinate.
- **R7.** The channel-presence indicator is an option for Stage 9, not a default.

From the brainstorm, the user's answers on 2026-10-03:

- **R8. Curvature.** A guard on c = 1: the export and the arm encoder refuse a checkpoint whose
  curvature is not 1, and the scorer's `lorentz` distance stays c = 1 only. Plan 4's scorer item
  retires.
- **R9. Compute.** The Exit's training run is a short local run on MPS (1–2 epochs). The Lambda
  workflow is planned in Stage 7's spec.
- **R10. Router-guided mining** runs only under MoE fusion. Otherwise the geometric miner takes
  every mining slot.
- **R11. The load-balancing term** is computed and logged only under MoE (follows from R3).
- **R12. The MoE ablation** fuses by masked mean, then routes the fused vector through the
  experts.
- **R13. Field markers** are text prefixes.
- **R14. Checkpoints.**
  - The contract records the encoder architecture, and an absent record reads as the legacy
    four-copy layout.
  - Exact resume and weights-only both refuse a mismatch, citing D2. Nothing migrates.
- **R15. Summaries.** Over-long channel texts are summarized, not truncated. This becomes roadmap
  Stage 6b, before Stage 7 and buildable beside Stage 6. Stage 6 keeps tail truncation for its
  interim run.

## 4. Design

### 4.1 The encoder

Per code, the path is:

1. field-marked channel texts;
2. one backbone with one LoRA adapter;
3. an attention-masked mean over each channel's tokens;
4. fusion over the present channels;
5. one `Linear(384 → d)`;
6. the geometry head, which gives the point.

- **Backbone.** One `AutoModel` from `model.base_model_name`, with one PEFT LoRA adapter.
  - `model.lora`, `target_modules='all-linear'` and gradient checkpointing stay as they are today.
  - The resolved revision is read as `panels/text_only.py:104-116` reads it.
  - The backbone runs only on present channels. A batch's present (row, channel) pairs are
    gathered into one backbone call, so absent channels and padding rows never enter it. In bundle
    301cce28, 6,428 of the 8,500 channel slots are present.
- **Field markers** (`text_model/fields.py`).
  - One constant lists the five fields: title, description, excluded, examples and query.
  - A field's marker is its name. A marked text is `'<field>: <text>'`.
  - Absent channels get no marker.
- **Fusion** (`text_model/fusion.py`, `model.fusion`):
  - `masked_mean`, the default: the sum over present channels, divided by max(1, number present).
  - `attention`: each present channel's vector is scored by its dot product with a learned
    vector, and a softmax over the present channels weights them.
  - `moe`, ablation only: the masked mean, then `MixtureOfExperts` (`text_model/moe.py`,
    `model.moe`) on that 384-d vector. The experts belong to the fusion step. Only this option
    emits `gate_probs` and `top_k_indices`.
  - Under every option, a row with no present channel fuses to a finite vector with a finite
    gradient. Under `masked_mean` and `attention` that vector is zeros; under `moe` it is the
    experts' output for zeros.
- **Projection.** Exactly one `nn.Linear(384, d)`, with `model.dimension` ∈ {8, 16, 32} and 16 as
  the default. Under every fusion option, the fused vector reaches the point through this one map.
  It replaces two maps:
  - `moe_projection` (`text_model/encoder.py:81`);
  - `HyperbolicProjection`'s `Linear(384, 385)` (`text_model/hyperbolic.py:102`), whose time
    output the exp map discarded.
- **Head.** This is the interim head, hyperbolic only. It replaces `HyperbolicProjection`
  (`text_model/hyperbolic.py:84`) and has no parameters.
  - It rescales the tangent vector to norm at most `max_norm` 2.0, as `hyperbolic.py:136-141`
    does. Stage 7 removes the cap (Req 13).
  - It applies the exp map at the origin, at the harness's curvature.
  - It returns `tangent` (B, d), the capped tangent vector that the export writes, and `embedding`
    (B, d + 1), the Lorentz point that the interim loss reads.
  - Stage 8 adds Euclidean and spherical heads behind the same interface, each naming its decoding
    distance. The encoder does not change.
- **Query path.** A query is a one-field batch, `{'query': …}`. It goes through the same forward:
  backbone, fusion with one present channel, projection and head.
- **Output.** The forward returns `embedding` and `tangent`, plus `gate_probs` and `top_k_indices`
  under `moe`. The `embedding_euc` output, which nothing reads, goes.
- **Model.**
  - `NAICSContrastiveModel` builds `SharedEncoder` (`text_model/shared_encoder.py`) where it built
    `MultiChannelEncoder` (`text_model/naics_model.py:298-307`).
  - Its constructor gains `fusion` and `dimension`, and keeps accepting every pre-Stage-6
    hyperparameter. That way a four-copy checkpoint's saved hyperparameters still construct a
    model and reach the contract check (4.4).
  - `text_model/encoder.py` is deleted.

### 4.2 Data flow and the interim harness

- **Tokenization cache, format v3.** `CACHE_FORMAT`
  (`text_model/dataloader/tokenization_cache.py:25`) becomes `channels-v3`.
  - Each present channel is tokenized as its marked text at the 128-token window. An absent
    channel stays the empty string with `present` False.
  - The sidecar records the marker set and a `summaries` entry. The entry is null in Stage 6, and
    Stage 6b fills it with its artifact's hash. Any mismatch rebuilds the cache.
  - Overflowing texts keep tail truncation until Stage 6b lands.
  - The bundle's window record (`data/supervision_bundle.py:272`) counts unmarked texts, so its
    overflow shares understate by the marker's two tokens. Bundle 301cce28 is not rebuilt for it.
- **One batch format.**
  - `_stack_text_inputs` (`text_model/dataloader/datamodule.py:60-70`) becomes the shared builder
    of every code batch: the collates, the export, and `generate_embeddings_from_checkpoint`.
  - It adds a boolean `present` of shape (B,) per channel. The flag comes from the cache rows,
    which the dataset items already carry.
  - The repaired collate's invalid rows (`datamodule.py:174-180`) carry `present` False.
  - A query batch has the same format under the field `query`, tokenized on the fly by the same
    tokenizer at the same window.
  - The encoder refuses a batch whose channels lack `present`. It never infers presence from the
    attention mask.
- **MoE-only machinery.** These are R10 and R11, the named deviations from R5.
  - The curriculum mixin consults the router-guided miner only under `moe`. Otherwise
    `router_slots` is 0 and the geometric miner proposes all `selection_k`
    (`text_model/mixins/curriculum.py:281-307`).
  - So a default 10-epoch run no longer raises when it reaches phase 2
    (`text_model/curriculum.py:152`, `text_model/mixins/curriculum.py:297-300`).
  - The load-balancing term is computed, added and logged only under `moe`
    (`text_model/naics_model.py:576-590` and `:645-652`; `text_model/mixins/loss.py:461-489`;
    `text_model/mixins/logging.py:398-403`).
  - The same holds for router-diversity logging (`text_model/mixins/logging.py:366-378`).
- **Everything else in the harness stays:**
  - the six-term objective;
  - the curriculum;
  - the candidate pool, whose invalid rows already never reach the encoder
    (`text_model/naics_model.py:464-473`);
  - the validation statistics.

  Nothing outside `text_model/encoder.py` assumes width 384 or 385.
- **HGCN's feeder.** `generate_embeddings_from_checkpoint` (`cli/commands/training.py:184`) keeps
  writing Lorentz `hyp_e*` columns through the shared builder, now d + 1 of them. HGCN finds its
  columns by prefix and stays untouched until Stage 10.
- **Unchanged readers.** The D9 text-only builder and the lexical baseline keep embedding unmarked
  text.
- **Config.**
  - `conf/config.yaml` gains `model.fusion: masked_mean` and `model.dimension: 16`.
  - `ModelConfig` (`utils/config.py:896`) validates both.
  - `model.moe` stays and applies only under `moe`.
  - `build_model_from_config` (`cli/commands/training.py:69`) passes both through.

### 4.3 Export and live reads

- **Export command:** `tools export-table --checkpoint <ckpt> --output <table.parquet>`.
  - **Bundle.** It resolves the bundle as `train` does (`--config` plus `key=value` overrides).
    The checkpoint's supervision contract must match that bundle.
  - **Loading.** The checkpoint loads with `load_from_checkpoint`. Its saved hyperparameters
    rebuild its own fusion and dimension, so the contract's architecture refusal fires only on a
    four-copy checkpoint (4.4).
  - **Encoding.** All 2,125 codes go through the shared builder, in eval mode, without gradient.
  - **The table.** Columns `code`, then `index` and `level` from the descriptions, then
    `e0 … e{d-1}` as float64, in the order of the bundle's codebook.
    - For the hyperbolic head the coordinates are the capped tangent vector at the origin (R6).
      There is no time coordinate, so there is no constant column.
    - The `e` prefix keeps the table distinct from HGCN's `hyp_e` and the text-only table's `t`.
  - **Provenance.** `<stem>_provenance.json`, written beside the table, records:
    - the checkpoint's path and sha256, and its contract (supervision and encoder);
    - the backbone and its resolved revision;
    - `max_length`, the descriptions' sha256 and `summaries: null`;
    - the table's sha256 and its `matrix_fingerprint` (`panels/regressor.py:350`).
  - **Curvature guard.** It refuses a checkpoint whose curvature is not 1 (R8).
- **Arm encoder** (`text_model/arm_encoder.py`). It implements `QueryCodeEncoder`
  (`panels/outcome.py:50`) from a checkpoint and its exported table.
  - **`encode_queries(texts)`.** Marked `query:` texts go through the checkpoint's model in
    batches, in eval mode, without gradient.
    - The tangent vectors are cast with `.cpu().to(torch.float64)`. The form
      `.to(device='cpu', dtype=…)` raises on MPS and is never used.
    - A float64 exp map at the origin then gives (time, space) rows.
  - **`encode_codes(codes)`.** It reads the codes' rows from the table, refusing an unknown code,
    and passes them through the same float64 exp map. The decoded code vectors are therefore a
    fixed function of the table, and the `matrix_fingerprint` a read logs names what was decoded.
  - **Distance.** `'lorentz'` (`panels/decoding.py:46`), named by the head.
  - **Refusals.** Curvature ≠ 1, and a table whose provenance names another checkpoint sha256 or
    another table hash.
  - **Fit with Stage 7.** These are the pieces of Stage 4's `SeedArtifacts`
    (`decision/sweep.py:44`): checkpoint, table, encoder and distance. Stage 7's `ArmRunner`
    returns them.
- **Outcome read:** `tools outcome-panel --checkpoint <ckpt> --table <table> --purpose <why>`.
  - It builds `OutcomePanel.from_bundle` (`panels/outcome.py:108`) on the configured bundle.
  - It scores the validation split with `score` (`:196`) under `lorentz`, with
    `detail={'table': <matrix_fingerprint>, 'checkpoint': <sha256>}`. The `table` key is the one
    the sweep logs (`decision/sweep.py:130`).
  - It prints the summary as `tools outcome-baseline` does (`cli/commands/tools.py:260`).
  - It has no test-split path. Stage 12 opens that split.
- **Regressor read.** No new code. `tools regressor-panel` (`cli/commands/tools.py:417`) runs on
  the validation split with:
  - `--coordinates`: the export;
  - `--text-only`: a table rebuilt by `tools text-only-table` from bundle 301cce28's descriptions;
  - `--codebook`: the bundle's codebook.

  The panel reduces the text-only table to 16 dimensions (D9).
- **Plan 6's deferred fixes.**
  - `regressor_scores` (`decision/scores.py:105`) counts each row's
    `pl.col('repeat').n_unique()` instead of `pl.len()` (`:130`).
  - `coordinate_matrix`'s Lorentz refusal (`panels/regressor.py:309`) is worded around the export
    form alone, since `tools diagnostics` raises it too.

### 4.4 Checkpoints

- **The record.** `CheckpointContract` (`supervision/checkpoints.py:44`) gains
  `encoder: EncoderArchitecture`, a frozen sub-record with four fields:
  - `layout`: `'shared'`, or the legacy `'four-copy'`;
  - `fusion`;
  - `dimension`;
  - `backbone`.
- **Absent reads as legacy.**
  - The field defaults to the legacy record: `layout='four-copy'`, with the other fields empty.
    Every checkpoint saved before Stage 6 therefore validates as four-copy, never as the runtime's
    architecture.
  - A field added later defaults to the value every earlier checkpoint had. Stage 8's `geometry`,
    for example, would default to `'hyperbolic'`.
- **Runtime side.**
  - `contract_for_bundle` (`:65`) and `containment_contract` (`:75`) take the encoder record.
  - The model builds its record from its hyperparameters, and `runtime_contract_for`
    (`cli/commands/training.py:149`) builds one from the config.
  - `contract_version` is still copied from the manifest, so bundle 301cce28 stays valid.
- **No revision field.** The state dict carries the backbone's base weights, so a resumed run
  restores them whatever the local snapshot is. The export's provenance records the revision
  instead.
- **Exact resume.** `validate_checkpoint_contract` (`:98`) compares the whole contract.
  - A four-copy checkpoint is refused with a message that names the `encoder` difference and D2:
    four-copy checkpoints cannot load into the shared encoder.
  - A shared checkpoint of another dimension, fusion or backbone is refused the same way.
  - `load_from_checkpoint` runs `on_load_checkpoint` (`text_model/naics_model.py:440-447`) before
    `load_state_dict`. Export and reads therefore meet the refusal, never a key mismatch.
- **Weights-only.** `load_weights_only` (`:132`) reads the saved contract first.
  - If the saved encoder record (absent counts as legacy) differs from the model's, it refuses
    with the same D2 message, before reading any parameter.
  - It still serves its purpose until Stage 7 deletes it (D2): the same architecture under another
    bundle or supervision contract.

## 5. Error handling

Named refusals, all ValueError:

- **Encoder:** a channel batch without `present`; a field outside the marker set.
- **Config:** `model.fusion` outside {masked_mean, attention, moe}; `model.dimension` outside
  {8, 16, 32}.
- **Export and arm encoder:**
  - curvature ≠ 1;
  - a supervision contract other than the configured bundle's;
  - a table whose provenance names another checkpoint or table;
  - an unknown code.
- **Checkpoints:** another encoder architecture, with the D2 message (4.4).

Guarantees rather than refusals:

- A row with no present channel fuses to a finite vector with a finite gradient (4.1).
- A cache sidecar mismatch (format, markers or summaries) rebuilds the cache.

## 6. Testing

Tests are written red to green.

- Unit tests build the backbone from a tiny `BertConfig`, as `tests/unit/test_text_only.py` does,
  so they download nothing.
- `tests/unit/test_encoder.py` is rewritten; it pins 384 at `:106`.
- The stage-3 integration test's encoder stub
  (`tests/integration/test_stage3_training_step.py:147-150`) moves to the new interface.

**Exit criteria:**

- **Same encoder.**
  - A one-field batch `{F: [T]}` and a one-code batch whose only present channel is F, with text
    T, give identical outputs.
  - `encode_queries([T])` equals the model's forward on `{'query': [T]}`.
  - The model holds exactly one backbone.
- **Masking.** Perturbing an absent channel's `input_ids` and `attention_mask` leaves the output
  bit-identical, under each fusion option.
- **One affine map.** Exactly one `nn.Linear` (384 → d) lies on the path from the fused vector to
  the point, and the head has no parameters.
- **Trains at d = 16.**
  - A training step at d = 16 reaches the LoRA and projection weights with gradient, and logs no
    load-balancing term under `masked_mean`.
  - A phase-2 selection under `masked_mean` neither raises nor fills a router slot.
- **Export.** The table has `code`, `index`, `level` and `e0 … e15` as float64. It passes
  `coordinate_matrix`, and its provenance matches the table and the checkpoint.
- **Live read.**
  - A read on a fixture panel logs `table` equal to the table's `matrix_fingerprint`.
  - Distances from table-decoded code vectors match those from the live forward within float32
    tolerance.
  - Curvature ≠ 1 is refused.

**Supporting tests:**

- masked-mean and attention values; MoE runs after the mean and is the only option emitting gates;
- the collate's `present`, False on invalid rows;
- cache v3:
  - markers on present channels, absent channels unmarked;
  - the sidecar's marker set and `summaries: null`;
  - a v2 cache rebuilds;
- the contract rules of 4.4:
  - absent reads as legacy;
  - the exact-resume and weights-only refusals carry the D2 message;
  - a shared-encoder positive control;
  - the record survives a save round trip;
  - a four-copy checkpoint meets the refusal through `load_from_checkpoint`;
- plan 6's two fixes: a duplicated repeat beside a missing one fails `regressor_scores`, and the
  refusal carries its new wording;
- `generate_embeddings_from_checkpoint` writes d + 1 `hyp_e*` columns;
- MPS: `encode_queries` and `encode_codes` return float64 CPU tensors. The test is skipped unless
  MPS is available, because CI cannot run it.

## 7. Exit procedure

This runs locally on MPS. Off CUDA the trainer picks `32-true` (`utils/backend.py:33`).

1. Clone bundle 301cce28 and `data/naics_descriptions.parquet` (sha256 `fe8c54e3…`) from the main
   checkout into the worktree's `data/` with `cp -cR`. Never symlink or rebuild them.
2. Point `supervision.manifest_path` at the clone with a `key=value` override on every command.
   Never commit it: the pin test fails on a committed path.
3. Train at d = 16 with `masked_mean`:
   `uv run naics-embedder train training.trainer.max_epochs=2 supervision.manifest_path=…`.
4. Export `last.ckpt` with `tools export-table`. No checkpoint selection happens here: the
   harness's monitor reads the in-sample validation loss, which selects nothing (Req 4), and
   Stage 7 wires selection to the validation query split (D6).
5. Build the text-only table with `tools text-only-table`. Then run `tools regressor-panel`
   (validation, both regimes, level 6, with the QCEW slices at `qcew_dir`) and
   `tools outcome-panel`.
6. Write `specs/findings/shared-encoder-first-reading.md`. It records:
   - the run;
   - the export's hashes;
   - both panels' validation numbers, which are a floor, not a target;
   - copies of the selection-log records, since the log is gitignored.

## 8. Deletions and documentation

**Deleted:**

- `text_model/encoder.py`: the four PEFT copies, the concatenation and `moe_projection`;
- `HyperbolicProjection`'s Linear;
- the `embedding_euc` output;
- under the default fusion: gate outputs, the load-balancing term and router mining.

`text_model/moe.py` and `MoEConfig` stay for the `moe` option.

**Documentation:**

- the architecture text in `CLAUDE.md` and `README.md` (the four-copy encoder and MoE fusion);
- `docs/text_training.md`;
- `docs/usage.md`, for the two new commands;
- the API pages in `docs/api/` for the new and deleted modules.

`uv run mkdocs build --strict` must pass locally, since PR CI never builds the docs.

## 9. Chosen approach and rejected alternatives

- **Field markers.**
  - Chosen: text prefixes.
  - Rejected: learned field embeddings, which are random under Stage 9's frozen-encoder control
    and need backbone-specific `inputs_embeds` plumbing.
  - Rejected: new special tokens, which change the tokenizer and are untrained noise under a
    frozen control.
- **Default fusion.**
  - Chosen: masked mean. It has no parameters and makes "no absent channel contributes" exact.
    It is also the D9 comparator's pooling (`panels/text_only.py`, the mean over present
    channels), so an arm and its comparator differ only by training and the projection.
  - Attention pooling stays an option.
- **MoE layout.**
  - Chosen: the masked mean, then the experts.
  - Rejected: per-channel experts, then the mean. It would thread per-channel gate keys through
    harness code that Stage 7 deletes.
  - Rejected: a zero-filled concatenation. Zero slots act as a presence indicator that encodes
    level (Req 9), and Req 14 rejects masked gating.
- **Curvature.**
  - Chosen: the guard.
  - Rejected: a curvature-aware distance, which no run Req 13 allows would call.
  - Rejected: both together (YAGNI).
- **Checkpoints.**
  - Chosen: refusal through an architecture record.
  - Rejected: relying on Lightning's key errors. Their messages don't say why, and the contract
    check passes first.
  - Rejected: migrating the adapters. That is code for a path Stage 7 deletes, with nothing local
    to migrate.
- **Router mining.**
  - Chosen: MoE only.
  - Rejected: leaving it, since the default run would raise at epoch 6.
  - Rejected: pinning phase 1 in config, which silently changes the curriculum.
- **Compute.**
  - Chosen: a local MPS run for the Exit, with Lambda planned in Stage 7.
  - Rejected: specifying the Lambda workflow here, infrastructure this Exit doesn't use.
  - Rejected: a hand-prepared Lambda run, which a run with no quality bar doesn't need.
- **Code side of a read.**
  - Chosen: decode from the exported table, so the logged fingerprint names the decoded vectors.
  - Rejected: decoding from a live forward, which leaves that fingerprint a label only.
- **Summaries.**
  - Chosen: their own Stage 6b.
  - Rejected: building them inside Stage 6. They are a different cycle: generation, review, a
    leakage audit and D9 provenance.
  - Rejected: building them in Stage 9, since Stages 7–8 would then fix δ on truncated text.

## 10. Rollout note

> Roadmap: specs/naics-embedding-roadmap.md, Stage 6 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Stage 6b handoff.** Window-fitting summaries are roadmap Stage 6b, added on 2026-10-03 at the
user's direction.

The overflow was measured at the 128-token window under MiniLM's tokenizer, special tokens
included:

| Channel | Texts over the window | Notes |
|---|---:|---|
| description | 153 of 2,111 | Sectors 16 of 20 (median 246 tokens, maximum 1,131); subsectors 59 of 96 |
| examples | 105 of 1,075 | 19 % of the channel's tokens truncated away |
| excluded | 464 of 1,117 | 26 % of the channel's tokens truncated away |

Three constraints carry over:

1. **Leakage.** Extractive summaries cannot add a leakage match. Abstractive ones need an audit
   against the validation and test queries: a non-selecting read of sealed text, which needs the
   user's approval.
2. **Text identity.** Summaries live in a frozen, pinned artifact, not in the descriptions, whose
   hash bundle 301cce28 records. D9 moves the text-only builder, its provenance, `TextOnlyRef`,
   `ArmSpec` and `decide` onto the summaries' hash.
3. **Window and timing.** Summaries target a window, and they land before Stage 7 fixes δ.

Stage 6 leaves the cache's `summaries` entry null.

**Deferred items,** handled through /deferred at plan completion:

- Plan 4's scorer item retires. Its premise, a learned curvature, never held; Req 13 keeps c = 1,
  and R8's guard refuses anything else.
- Plan 6's Stage 6 item is discharged (4.3).

**Model routing.** writing-plans for plan 8 runs in a fresh Opus session from this spec; execution
runs on the Sonnet default.
