# Shared Encoder and Low-Dimensional Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-embedding-roadmap.md, Stage 6 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Build roadmap Stage 6. Replace the four-copy LoRA encoder and its mixture-of-experts
fusion with one shared encoder: field markers, masked fusion, one affine map to dimension 8, 16
or 32, and a query path through the same layers. Then export the 2,125-code table in Req 2's
form, and take the first live validation-split readings on both panels.

**Architecture:**

- **Fields and the cache.** `text_model/fields.py` names the five fields and marks a present text
  as `'<field>: <text>'`. The tokenization cache moves to format `channels-v3`: it stores marked
  present channels, and its sidecar records the marker set and a null `summaries` entry for
  Stage 6b.
- **One batch format.** One builder, `stack_text_inputs`, gives every channel a boolean
  `present`.
- **The encoder.** `SharedEncoder` (`text_model/shared_encoder.py`) runs one MiniLM with one LoRA
  adapter.
  - It gathers a batch's present (row, field) pairs into one backbone call and mean-pools each
    text.
  - It fuses the present channels (`text_model/fusion.py`): masked mean by default, attention
    pooling, or the MoE ablation.
  - One `Linear(384 → d)` maps the fused vector to a parameter-free hyperbolic head, which
    returns the capped tangent and the Lorentz point.
- **MoE-only machinery.** Router mining and the load-balancing term run only under MoE (R10,
  R11).
- **Checkpoints.** The checkpoint contract records the encoder architecture and refuses any other
  (D2).
- **Export and reads.**
  - `text_model/export.py` writes the table (`code`, `index`, `level`, `e0…e{d-1}`: the tangent
    at the origin) with its provenance.
  - `text_model/arm_encoder.py` implements `QueryCodeEncoder` from a checkpoint and its table.
  - `tools export-table` and `tools outcome-panel` drive them.
- **The Exit** trains one local epoch, exports it and reads both panels' validation splits.

**Tech Stack:**

- Python 3.10 and 3.12 (CI runs both; `.python-version` pins 3.12).
- torch 2.9.1, transformers 4.57.1, peft 0.17.1 (LoRA), and pytorch-lightning 2.5.5, imported as
  `pytorch_lightning`.
- polars 1.35.1 and pydantic 2.12.4 (config, contracts).
- typer and rich (the CLI).
- pytest with xdist; ruff and yapf; mkdocs with mkdocstrings (strict build).

## Global Constraints

Every task's requirements include this section.

### The spec (`specs/shared-encoder-and-projection.md`, APPROVED, last changed 913c3fc), verbatim

§3, Rulings:

> From the session brief, not re-triaged:
>
> - **R1.** The shared encoder lands in Stage 6, before the objective.
> - **R2.** The backbone stays all-MiniLM-L6-v2 at revision 1110a243 until Stage 9.
> - **R3.** Fusion options are masked mean, attention pooling and MoE, MoE as an ablation only. The
>   per-channel adapter copies and the load-balancing term go with the default fusion.
> - **R4.** Dimension is configurable in {8, 16, 32}, the Exit runs 16, and exactly one affine map
>   sits between encoder and point.
> - **R5.** The current objective and dataloader serve unchanged as an interim harness until Stage 7,
>   apart from the two named deviations R10 and R11.
> - **R6.** A hyperbolic export writes Req 2's form: tangent coordinates at the origin, without the
>   zero time coordinate.
> - **R7.** The channel-presence indicator is an option for Stage 9, not a default.
>
> From the brainstorm, the user's answers on 2026-10-03:
>
> - **R8. Curvature.** A guard on c = 1: the export and the arm encoder refuse a checkpoint whose
>   curvature is not 1, and the scorer's `lorentz` distance stays c = 1 only. Plan 4's scorer item
>   retires.
> - **R9. Compute.** The Exit's training run is one full epoch, locally on MPS. Measured on
>   2026-10-03 on an M4 Max, that is about 1.5 hours with validation:
>   - 5,958 rows from 1,273 anchors, so 373 steps at batch 16;
>   - about 9.5 s of backbone work per training step;
>   - about 19 minutes for the validation pass.
>
>   The Lambda workflow is planned in Stage 7's spec.
> - **R10. Router-guided mining** runs only under MoE fusion. Otherwise the geometric miner takes
>   every mining slot.
> - **R11. The load-balancing term** is computed and logged only under MoE (follows from R3).
> - **R12. The MoE ablation** fuses by masked mean, then routes the fused vector through the
>   experts.
> - **R13. Field markers** are text prefixes.
> - **R14. Checkpoints.**
>   - The contract records the encoder architecture, and an absent record reads as the legacy
>     four-copy layout.
>   - Exact resume and weights-only both refuse a mismatch, citing D2. Nothing migrates.
> - **R15. Summaries.** Over-long channel texts are summarized, not truncated. This becomes roadmap
>   Stage 6b, before Stage 7 and buildable beside Stage 6. Stage 6 keeps tail truncation for its
>   interim run.

§5, Error handling:

> Named refusals, all ValueError:
>
> - **Encoder:** a channel batch without `present`; a field outside the marker set.
> - **Config:** `model.fusion` outside {masked_mean, attention, moe}; `model.dimension` outside
>   {8, 16, 32}.
> - **Export and arm encoder:**
>   - curvature ≠ 1;
>   - a supervision contract other than the configured bundle's;
>   - a table whose provenance names another checkpoint or table;
>   - an unknown code.
> - **Checkpoints:** another encoder architecture, with the D2 message (4.4).
>
> Guarantees rather than refusals:
>
> - A row with no present channel fuses to a finite vector with a finite gradient (4.1).
> - A cache sidecar mismatch (format, markers or summaries) rebuilds the cache.

§6, Exit criteria:

> - **Same encoder.**
>   - A one-field batch `{F: [T]}` and a one-code batch whose only present channel is F, with text
>     T, give identical outputs.
>   - `encode_queries([T])` equals the float64 exp map of the `tangent` that the model's forward
>     gives for `{'query': [T]}`.
>   - The model holds exactly one backbone.
> - **Masking.** Perturbing an absent channel's `input_ids` and `attention_mask` leaves the output
>   bit-identical, under each fusion option.
> - **One affine map.** Exactly one `nn.Linear` (384 → d) lies on the path from the fused vector to
>   the point, and the head has no parameters.
> - **Trains at d = 16.**
>   - A training step at d = 16 reaches the LoRA and projection weights with gradient, and logs no
>     load-balancing term under `masked_mean`.
>   - A phase-2 selection under `masked_mean` neither raises nor fills a router slot.
> - **Export.** The table has `code`, `index`, `level` and `e0 … e15` as float64. It passes
>   `coordinate_matrix`, and its provenance matches the table and the checkpoint.
> - **Live read.**
>   - A read on a fixture panel logs `table` equal to the table's `matrix_fingerprint`.
>   - Distances from table-decoded code vectors match those from the live forward within float32
>     tolerance.
>   - Curvature ≠ 1 is refused.

The tasks carry §4 (design), §7 (Exit procedure) and §8 (deletions and documentation). Read the
spec for their full text; a task that cites "4.2" means that section.

### The roadmap (`specs/naics-embedding-roadmap.md` at 52075f9), verbatim

Stage 6:

> Objective: Replace the four-copy LoRA encoder and mixture-of-experts fusion with one shared
> encoder behind field markers, masked fusion, one affine map to a configurable dimension, and a
> query path.
>
> Gap closed: Req 14 (encoder half); Req 12 (projection and dimension half); Req 9 (model-side
> mask); Req 16 (export half).
>
> Produces: One encoder module implementing `QueryCodeEncoder`, with code and query embeddings in
> the same space; fusion options masked mean, attention pooling and MoE (MoE ablation-only); a
> configurable embedding dimension in {8, 16, 32}; a standalone export command writing the
> 2,125-code table in Req 2's form; the per-channel adapter copies and the load-balancing term
> deleted with the default fusion; a `summaries` entry in the tokenization cache's identity, null
> until Stage 6b; an outcome read whose logged `table` names the code vectors the encoder decodes
> against (Stage 4's sweep logs the runner's table as that label, which `decide` checks but cannot
> tie to the decoding).
>
> Exit: A query embeds through the same encoder as a code (test); masking an absent channel's
> input leaves the output unchanged (test); exactly one affine map sits between the encoder and
> the point; the model trains under the interim objective at dimension 16 and the export command
> writes the 2,125-code table, which `tools regressor-panel` reads on its validation split; Stage
> 2's scorer returns live validation-split numbers under the harness's curvature.

The decisions it names:

> - **D2 — Legacy containment mode (Stage 7).** `supervision.mode: legacy_containment`, its second
>   `training_step` (`text_model/naics_model.py:618-693`) and the checkpoint-contract migration
>   appear nowhere in the spec. Decision: Stage 7 deletes them with the old objective; legacy
>   checkpoints cannot load into a 16-dimensional shared encoder anyway.
> - **D6 — The within-run selection statistic (Req 4; Stage 7).** Once the in-sample loss selects
>   nothing, checkpointing, early stopping and learning-rate control need a validation-split
>   statistic the spec does not name. Decision: the validation query split's MRR (Req 3); both
>   panels are used only between configurations, under Req 5.
> - **D9 — Req 2's text-only comparator (Stage 3; Stages 8 and 9 re-run it).** Req 2 names no
>   representation, and review C28, its source, lists two: a frozen encoder and TF-IDF. Decision:
>   the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the arm's
>   dimension. That backbone is the current checkpoint until Stage 9 adopts another, and whichever
>   backbone the arm uses after. The comparator measures what taxonomy training adds over the same
>   encoder reading the same text.

D2's line citation is stale: the legacy step is at `text_model/naics_model.py:616-691`. This plan
keeps it, under the shared encoder, and Stage 7 deletes it.

Stage 6b (window-fitting summaries) is not this plan. It touches
`text_model/dataloader/tokenization_cache.py` too, so whichever of the two lands second
integrates the other.

### Deferred items this plan closes (`specs/deferred_items.md`), verbatim

Plan 4's item (it retires, by the user's disposition under `/deferred`; it is not "done"):

> - [ ] Review Minor: the scorer's `lorentz` distance (`lorentz_distances` in
>       src/naics_embedder/panels/decoding.py) assumes curvature −1, while the text model's
>       curvature is learnable. Roadmap Stage 6 must pass a curvature-aware callable to
>       `OutcomePanel.score(..., distance=...)` or add a curvature parameter. Size: quick-fix.
>       Done when: Stage 6 scores its arm under the trained curvature.

Plan 6's item (Task 1 discharges it):

> - [ ] Review Minor, for Stage 6: (1) `regressor_scores` (src/naics_embedder/decision/scores.py)
>       counts each row's predictions with `pl.len()`, not its distinct `repeat` values, so a
>       duplicated repeat beside a missing one passes; the panel's own `_predict` is the only
>       producer today. (2) `coordinate_matrix`'s refusal of Lorentz points
>       (src/naics_embedder/panels/regressor.py) says "the regressor panel takes the export
>       form", which misleads when `tools diagnostics` raises it. Deferred because Stage 6 owns
>       the export form and next touches both. Fix: count `pl.col('repeat').n_unique()`; word the
>       refusal around the export form alone. Size: quick-fix. Done when: Stage 6's export lands
>       with both changed.

### Decisions already made (do not re-ask)

The spec's §3 is the authority; the brief that ordered this plan restated it:

- one shared MiniLM backbone with one LoRA adapter, run on present channels only;
- text-prefix field markers in tokenization cache v3, whose sidecar carries `summaries: null` for
  Stage 6b;
- fusion `masked_mean` (default), `attention`, or `moe` (masked mean, then the experts; ablation
  only);
- exactly one `Linear(384 → d)`, d in {8, 16, 32}, default 16;
- an interim hyperbolic head that keeps the cap and returns `tangent` and `embedding`;
- router mining and the load-balancing term only under `moe`, the named deviations from "harness
  unchanged";
- a guard on c = 1;
- an encoder record in `CheckpointContract`:
  - an absent record reads as the legacy four-copy layout;
  - exact resume and weights-only refuse a mismatch (D2);
  - nothing migrates;
- export and reads take the encoder record from the checkpoint, and only training's resume
  compares it with the config (but see P17);
- `tools export-table` (tangent `e0…e{d-1}`, with provenance), an arm encoder that decodes codes
  from the exported table, and `tools outcome-panel`;
- the Exit trains one full local epoch on MPS.

Window-fitting summaries are Stage 6b, not this plan.

### This plan's decisions

Each is beyond, or a reading of, a spec line. Do not reopen them during execution.

- **P1. Fields.**
  - `FIELDS = ('title', 'description', 'excluded', 'examples', 'query')`.
  - `CHANNELS` is the first four, in the cache's and the collates' order, and `QUERY = 'query'`.
  - `marker(field)` is `f'{field}: '`. `marked_text(field, text)` refuses a field outside
    `FIELDS`.
- **P2. The sidecar.** It records `field_markers` (`{channel: marker}` for the four cached
  channels) and `summaries: null` beside `cache_format: 'channels-v3'`. `SUMMARIES = None` is a
  module constant in `tokenization_cache.py` that Stage 6b replaces.
- **P3. The batch builder.**
  - `stack_text_inputs(embeddings, fields=CHANNELS)` is public.
  - It refuses an item whose channel lacks `present`, and stacks `present` as a `torch.bool` (B,).
  - One function, `tokenize_field` in `fields.py`, tokenizes a field text for both the cache and
    the query path.
- **P4. Trimmed backbone calls.**
  - Each backbone call is trimmed to the longest present text it holds (right padding,
    measured over present rows only), so padding columns never enter the backbone.
  - This reads 4.1's "padding rows never enter it". It also keeps the masking test bit-identical,
    and it shortens R9's measured step.
- **P5. Fusion.**
  - The interface is `forward(vectors (B, F, H), present (B, F)) -> FusionOutput`.
  - Every option applies the mask itself, whatever the absent slots hold.
  - Attention's learned vector starts at zeros, so attention pooling starts as the masked mean.
  - Absent scores take the dtype's minimum, not −inf, and the softmax weights are multiplied by
    `present`. So an all-absent row has zero weights and a finite gradient.
- **P6. The head.**
  - `HyperbolicHead(curvature, max_norm=2.0)` lives in `text_model/hyperbolic.py`, with the class
    attribute `distance = 'lorentz'`.
  - It caps the norm exactly as `HyperbolicProjection` did and applies the existing
    `_exp_map_zero_compiled` to the (B, d) tangent.
  - `HyperbolicProjection` is deleted whole in Task 7.
- **P7. The backbone loader.** The backbone loads through a module-level `load_base_model(name)`
  in `shared_encoder.py`, which tests replace with a tiny BERT. `SharedEncoder.backbone_revision`
  reads `_commit_hash`, as `panels/text_only.load_backbone` does.
- **P8. The pooler stays (spec 4.1 pins the LoRA config).**
  - `target_modules='all-linear'` also wraps BERT's `pooler.dense`. Mean pooling never reads it,
    so its two LoRA tensors never get a gradient (1.82 % of MiniLM's LoRA parameters), as in the
    four-copy encoder.
  - Tests skip `.pooler.` parameters. Removing the pooler is a later choice, raised at hand-off.
- **P9. Gradient tests read `lora_B`.** PEFT starts `lora_B` at zero, so every `lora_A` gradient
  is exactly zero at the first step. The "trains at d = 16" test asserts nonzero gradients on
  every non-pooler `lora_B` tensor and on `projection.weight`.
- **P10. One switch for R10 and R11: the model attribute `fusion`.**
  - `CurriculumMixin._router_mining_enabled()` is true only under `fusion == 'moe'` with the
    phase flag on. It gates the selection, the hard-negative log's global-batch flag and the
    router-diversity log.
  - Both training steps compute, add and log the load-balancing term only under `moe`.
  - `_combine_loss_terms` and `_log_loss_breakdown` accept `None` for that term.
- **P11. The model's new parameters.** `NAICSContrastiveModel` gains `fusion='masked_mean'`
  (Task 6) and `dimension=16` (Task 7), checked against `FUSIONS` and `DIMENSIONS`. Every
  pre-Stage-6 parameter stays.
- **P12. The config keys.** `ModelConfig.fusion: Literal['masked_mean', 'attention', 'moe']` and
  `ModelConfig.dimension: Literal[8, 16, 32]`, with defaults `masked_mean` and 16.
- **P13. The encoder record.**
  - `EncoderArchitecture(layout, fusion, dimension, backbone)` is frozen and forbids extras;
    `layout` is `Literal['shared', 'four-copy']`.
  - A shared record requires the other three fields, and a four-copy record forbids them.
  - `LEGACY_ENCODER` is `CheckpointContract.encoder`'s default.
  - One helper, `shared_encoder_architecture(fusion=, dimension=, backbone=)`, builds the model's
    record. `encoder_architecture_for(cfg)` in `cli/commands/training.py` calls the same helper
    for the config's.
- **P14. The messages.**
  - `D2_REFUSAL` is one constant.
  - When `encoder` differs, it is appended to the existing
    `exact resume contract mismatch (saved, runtime): {...}`.
  - It also ends the no-contract message, which keeps "cannot exact resume" (existing tests match
    it) and stops suggesting weights-only, which now refuses such a checkpoint too.
- **P15. Weights-only takes the record explicitly.** The signature is
  `load_weights_only(model, path, *, encoder)`, because a bare `nn.Module` has no record. `train`
  passes `runtime_contract.encoder`, which the model's constructor has already matched.
- **P16. Export and reads check the checkpoint before building the model.**
  - The curvature guard reads the saved `hyper_parameters['curvature']`.
  - `validate_supervision_contract(raw, manifest, supervision_mode)` compares every field but
    `encoder`.
  - The encoder refusal stays `load_from_checkpoint`'s `on_load_checkpoint` (4.4).
- **P17. Two config comparators.**
  - Training's exact resume and the HGCN feeder's `validate_exact_resume` both compare the
    encoder record with the config. 4.4's last bullet keeps the feeder's check, and its
    "Only training's exact resume…" is read with it.
  - Export and reads compare none.
- **P18. Where the config is resolved.**
  - Library functions take explicit inputs: a validated bundle, a `TokenizationConfig` (whose
    `descriptions_parquet` names the descriptions) and a device.
  - Only the two CLI commands resolve `--config` plus `key=value` overrides, through the private
    helpers `_run_config` and `_run_bundle` in `tools.py`. Their bundle comes from
    `require_valid_supervision_bundle`, as `train`'s does.
  - `train` warns about an override without `=` and skips it. These two commands refuse one,
    because every read is logged and must run on exactly the config it names.
  - `train()` is not refactored.
- **P19. A stale cache rebuilds.** Export, the arm encoder and the HGCN feeder load the token
  cache with `tokenization_cache(..., use_locking=True)`, so a stale sidecar rebuilds it (§5).
  The feeder changes from fail-fast to rebuild.
- **P20. The export table.**
  - Dtypes: `code` Utf8; `index` and `level` Int64, from the descriptions; `e0…` Float64.
  - Rows are in codebook order.
    - The export sorts the descriptions by `index`, which is the codebook's `code_id`.
    - It refuses them unless their `(index, code)` rows equal the codebook's `(code_id, code)`
      rows.
  - The table's `matrix_fingerprint` is computed before the file is written. A table that
    `coordinate_matrix` refuses is therefore never written.
- **P21. The export provenance.**
  - The path comes from `panels/text_only.provenance_path`: `<stem>_provenance.json`.
  - Keys:
    - `checkpoint` {`path`, `sha256`}, `contract`, `backbone`, `revision`, `max_length`;
    - `descriptions` {`path`, `sha256`}, `summaries`;
    - `codes`, `dimension`, `coordinates`;
    - `table_sha256`, `matrix_fingerprint`, `library_versions`, `generated_at`.
- **P22. The arm encoder.**
  - Its exp map is `exp_map_origin`: float64 at curvature −1, which is c = 1.
  - Queries go through `tokenize_field` with the cache's tokenizer
    (`data_loader.tokenization.tokenizer_name`) at the cache's window.
  - Its `distance` is the head's, `model.encoder.head.distance` (`'lorentz'`). A Stage 8 head
    that names another distance therefore changes the read without touching the arm encoder.
  - It checks the table's provenance before it loads the model.
- **P23. `tools outcome-panel`.** It requires `--purpose`. Its read logs
  `detail={'table': <matrix_fingerprint>, 'checkpoint': <sha256>}` on top of the panel's own
  `encoder` and `distance`.
- **P24. Both repeat counts are checked.**
  - `regressor_scores` requires both `pl.len() == repeats` and
    `pl.col('repeat').n_unique() == repeats` per row. That is stronger than the spec's "instead
    of": a row with repeats [0, 1, 1] would otherwise pass.
  - The message keeps "do not have N predictions".
- **P25. The refusal's wording.** It becomes: "the coordinate table holds Lorentz points on a
  hyperboloid, not the export form (tangent coordinates at the origin for a hyperbolic arm)".
- **P26. Devices.**
  - `tools export-table` and `tools outcome-panel` use `pick_device('auto')` (MPS locally), as
    the HGCN feeder does.
  - Library functions default to `device='cpu'`, and their tests use that default.
  - `encode_token_rows` takes no device: it sends the inputs to the model's.
- **P27. Export and reads need a bundle.**
  - Export reads the bundle's codebook, and reads use its index roles.
  - Under `supervision.mode: legacy_containment` the gate returns no bundle. Both commands then
    refuse with "export and reads need a supervision bundle; legacy containment has none".
  - The feeder keeps its legacy-containment branch, which Stage 7 deletes (D2).

### Recorded deviations

- **R10 and R11** are the spec's two named deviations from R5 ("harness unchanged").
- **§6's "download nothing".**
  - Unit tests use a tiny BERT backbone.
  - Every test that builds a token cache still uses the real MiniLM tokenizer from the Hugging
    Face cache (CI downloads it), as `tests/unit/test_tokenization_cache.py` already does.
    `utils/input_window.check_window` records a window for no other tokenizer.
  - `tests/unit/test_naics_model.py` keeps its real MiniLM backbone.
- **"Trains at d = 16"** asserts gradient on `lora_B` and skips the pooler's LoRA (P8, P9).
- **The stub's location.** §6 cites `tests/integration/test_stage3_training_step.py:147-150`. The
  stub class is at 81-101, and it is installed at 149 and 247.
- **§8's docs list is incomplete.** Task 13 also edits `docs/index.md`, `docs/overview.md`,
  `docs/.nav.yml`, `tests/README.md` and `CLAUDE.md:1204`.

### Project rules

- **Style (CLAUDE.md).**
  - Single quotes, including `'''` docstrings. A string with an apostrophe takes double quotes
    (ruff Q003).
  - YAPF owns layout (100 columns), and ruff lints (E, F, I, Q). **Never run `ruff format`.**
  - One blank line between top-level definitions and after imports.
  - Semantic section dividers; `logging` rather than `print`; type hints on signatures.
  - To keep a vertical Polars chain, fence it with `# yapf: disable` / `# yapf: enable`.
- **Config.** Every config key is declared in a Pydantic model (PRs #104 and #105).
- **Formatting.** Format touched files with `./scripts/format_code.sh <files>`. At the end,
  `./scripts/format_code.sh --check --all` must pass.
- **Git.**
  - Never push to `main`.
  - Never push, cherry-pick or merge the held local commits "config" and "graph config". They
    are named by subject because every sync rewrites their SHAs.
  - Never run bare `git stash`.
  - Commit on this branch only, and end each message with the session's attribution trailer.
  - Add files by explicit path, never `git add -A` or `git add .`. `outputs/` is not gitignored,
    and Task 14's training writes TensorBoard logs there.
- **Data safety.**
  - The main checkout's `data/` is reference only; never write there.
  - Bundle 301cce28 and `data/naics_descriptions.parquet` (sha256 `fe8c54e3…`) are canonical.
    Task 14 copies them into this worktree's `data/` with `cp -cR`. Never symlink or rebuild
    them.
  - Tests write only under `tmp_path`. Every `TokenizationConfig` a test builds names a
    `tmp_path` output. A test that calls code which builds the default
    `./data/token_cache/token_cache.pt` first runs `monkeypatch.chdir(tmp_path)`.
- **Sealed sets.** No `OutcomePanel.open_test`, `RegressorPanel.open_outer` or `--split test`
  (Stage 12 only).
- **Configs.**
  - `conf/config.yaml` keeps `supervision.manifest_path: null`.
  - Task 14 passes `supervision.manifest_path=…` as a `key=value` override on every command, and
    never commits it: the pin test fails on a committed path.
  - Task 7 adds `model.fusion` and `model.dimension` to this worktree's `conf/config.yaml`.
  - `conf/graph.yaml` is not edited.
- **Downloads.** Never download Census or QCEW files. The backbone and its tokenizer come from
  the local Hugging Face cache.
- **Devices.** On MPS, cast with `.cpu().to(torch.float64)`, never
  `.to(device='cpu', dtype=…)`, which raises on MPS.
- **Deferred items.** Do not promote any open item of `specs/deferred_items.md` beyond the two
  this plan closes.
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-embedding-roadmap.md` and `specs/deferred_items.md`, and hand the user the exact
  edit instead.
- **Docs.**
  - `uv run mkdocs build --strict` must pass locally, because PR CI never builds the docs.
  - Griffe is strict: every documented parameter gets its own `Args:` entry.
- **CLI tests.** Rich wraps at 80 columns on CI, so match on `result.output.replace('\n', '')`.
- **Bash tool.** It runs zsh.
  - Quote `=`-leading words (`echo '====='`); an unquoted one aborts the command.
  - If the tool refuses a heredoc or a compound command, run one plain command per call and
    write files with the Write tool.

## Workspace

- **Worktree:** `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-8-shared-encoder`.
  Run every command from its root.
- **Branch:** `claude/plan-8-shared-encoder-075aeb0a`, cut from origin/main `52075f9`. This plan
  is its first commit.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder` stays on local `main` `ec202bf`.
  - That is `09a57bd` plus the held commits "config" (`233c339`) and "graph config" (`ec202bf`),
    six commits behind origin/main.
  - Do not check anything out there.
- **Real inputs, read only (Task 14):**
  - Bundle:
    `/Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/`
    (`codebook_fingerprint` `4662b826…`, `description_fingerprint` `fe8c54e3…`).
  - Descriptions: `/Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`,
    sha256 `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36`.
  - QCEW: `~/Downloads/Data/QCEW` (`qcew_dir` in `conf/data/regressor_panel.yaml`).
  - Backbone and tokenizer:
    `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2` (revision
    1110a243).
- **Outputs (Task 14):** all gitignored except `outputs/`.
  - This worktree's `data/`: the clones, the token cache, and `data/plan8/` for the tables and
    the read summaries.
  - `checkpoints/sadc_default/` and `logs/`.
  - `outputs/sadc_default/` (TensorBoard), which is never added to git.
- **Working directory:** the Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before every commit and before Task 14's commands. If it is not this
  worktree, `cd` back first.

## File structure

| Path | Change | Responsibility |
|---|---|---|
| `src/naics_embedder/decision/scores.py` | Modify (Task 1) | `regressor_scores` checks both repeat counts |
| `src/naics_embedder/panels/regressor.py` | Modify (Task 1) | The Lorentz refusal names the export form alone |
| `src/naics_embedder/text_model/fields.py` | Create (Task 2) | The five fields, their markers, `tokenize_field` |
| `src/naics_embedder/text_model/dataloader/tokenization_cache.py` | Modify (Task 2) | Format `channels-v3`: marked texts; markers and `summaries` in the sidecar |
| `src/naics_embedder/text_model/dataloader/datamodule.py` | Modify (Task 3) | `stack_text_inputs` with `present`; invalid rows absent |
| `src/naics_embedder/text_model/fusion.py` | Create (Task 4) | Masked mean, attention pooling, the MoE ablation |
| `src/naics_embedder/text_model/hyperbolic.py` | Modify (Tasks 5, 7) | `HyperbolicHead` added; `HyperbolicProjection` deleted |
| `src/naics_embedder/text_model/shared_encoder.py` | Create (Task 5) | `SharedEncoder`: one backbone, fusion, one `Linear(384 → d)`, the head |
| `src/naics_embedder/text_model/mixins/curriculum.py` | Modify (Task 6) | Router mining only under `moe` (R10) |
| `src/naics_embedder/text_model/mixins/logging.py` | Modify (Task 6) | Router and load-balancing logs only under `moe` |
| `src/naics_embedder/text_model/mixins/loss.py` | Modify (Task 6) | `_combine_loss_terms` takes no load-balancing term |
| `src/naics_embedder/text_model/naics_model.py` | Modify (Tasks 6, 7, 8) | `fusion`, `dimension`, `SharedEncoder`, the encoder record |
| `src/naics_embedder/text_model/encoder.py` | Delete (Task 7) | The four-copy encoder |
| `src/naics_embedder/utils/config.py` | Modify (Task 7) | `model.fusion`, `model.dimension` |
| `conf/config.yaml` | Modify (Task 7) | `model.fusion: masked_mean`, `model.dimension: 16` |
| `src/naics_embedder/supervision/checkpoints.py` | Modify (Task 8) | `EncoderArchitecture`, D2 refusals, the supervision-only check |
| `src/naics_embedder/cli/commands/training.py` | Modify (Tasks 7, 8, 9) | Builder, summary, encoder record, weights-only, feeder |
| `src/naics_embedder/utils/training.py` | Modify (Task 7) | The training summary records `fusion` and `dimension` |
| `src/naics_embedder/text_model/export.py` | Create (Tasks 9, 10) | Encode token rows; load an arm checkpoint; export the table |
| `src/naics_embedder/text_model/arm_encoder.py` | Create (Tasks 11, 12) | `ArmEncoder` (`QueryCodeEncoder`) and the outcome read |
| `src/naics_embedder/cli/commands/tools.py` | Modify (Tasks 10, 12) | `tools export-table`, `tools outcome-panel` |
| `src/naics_embedder/panels/text_only.py` | Modify (Task 7) | Its docstring stops citing `text_model/encoder.py` |
| `tests/unit/test_decision_scores.py`, `tests/unit/test_regressor_panel.py` | Modify (Task 1) | Plan 6's two fixes |
| `tests/unit/test_fields.py` | Create (Task 2) | Markers and `tokenize_field` |
| `tests/unit/test_tokenization_cache.py` | Modify (Task 2) | Cache v3 |
| `tests/unit/test_datamodule.py`, `tests/fixtures/epoch_datasets.py` | Modify (Task 3) | `present` in hand-built items; the builder's tests |
| `tests/unit/test_naics_model.py` | Modify (Tasks 3, 6, 7, 8) | Fixtures, R10/R11, the shared encoder, the contract |
| `tests/integration/test_stage3_training_step.py` | Modify (Tasks 3, 6, 7) | `present`; R10 cases; `StubSharedEncoder` |
| `tests/unit/test_fusion.py` | Create (Task 4) | Fusion values and masking |
| `tests/unit/test_encoder.py` | Rewrite (Task 5) | `SharedEncoder` on a tiny BERT |
| `tests/unit/test_hyperbolic.py` | Modify (Tasks 5, 7) | `TestHyperbolicHead` replaces `TestHyperbolicProjection` |
| `tests/unit/test_hard_negative_mining.py`, `tests/integration/test_distributed_supervision.py` | Modify (Task 6) | Selection hosts carry `fusion` |
| `tests/unit/test_config.py`, `tests/unit/test_utils_training.py` | Modify (Task 7) | The two config keys; the summary's record of them |
| `tests/unit/test_checkpoint_contract.py` | Rewrite (Task 8) | The encoder record and the D2 refusals |
| `tests/unit/test_cli_training.py` | Modify (Tasks 7, 8) | The keys reach the model; the configured encoder record |
| `tests/fixtures/shared_encoder.py` | Create (Task 5), extend (Task 9) | Tiny backbone (Task 5); five-code descriptions and a shared checkpoint (Task 9) |
| `tests/conftest.py` | Modify (Tasks 5, 7) | Registers `tests.fixtures.shared_encoder`; drops the projection-only fixture |
| `tests/unit/test_export.py` | Create (Tasks 9, 10) | Encoding rows, the feeder, the export |
| `tests/unit/test_arm_encoder.py` | Create (Tasks 11, 12) | The arm encoder and the live read |
| `tests/unit/test_cli_commands.py` | Modify (Tasks 10, 12) | The two commands' wiring |
| `docs/api/encoder.md` | Modify (Tasks 7, 13) | Task 7 points it at `shared_encoder`, so the strict docs build stays importable; Task 13 adds `fields` and `fusion` |
| `CLAUDE.md`, `README.md`, `tests/README.md`, `docs/index.md`, `docs/overview.md`, `docs/text_training.md`, `docs/usage.md`, `docs/api/moe.md`, `docs/.nav.yml` | Modify (Task 13) | The shared encoder replaces the four-copy text; the two new commands |
| `docs/api/export.md` | Create (Task 13) | The export and arm-encoder API page |
| `specs/findings/shared-encoder-first-reading.md` | Create (Task 14) | The Exit's run, export and readings |

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A step would write to the main checkout's `data/`, commit `supervision.manifest_path`, open a
  sealed split, or download a Census or QCEW file.
- `origin/main` gains a commit touching a file in **File structure**, or an open PR does. Stage
  6b's would touch `tokenization_cache.py`.
- Task 14's training fails or runs past 2.5 hours, its export holds other than 2,125 codes, or a
  panel read fails.

## Pre-flight (controller, inline, before Task 1)

- [ ] **Step 1: Confirm the workspace**

Run: `git status --short --branch`
Expected: `## claude/plan-8-shared-encoder-075aeb0a` and nothing else. If the line ends in
`...origin/main`, run `git branch --unset-upstream`.

Run: `git log --oneline origin/main..HEAD`
Expected: only this plan's commit (`docs(plans): add plan 8, the shared encoder and projection`).
If "config" or "graph config" appears, stop.

Run: `git fetch origin`, then
`git log --oneline HEAD..origin/main -- src tests conf docs specs CLAUDE.md README.md`
Expected: no output.
- If anything landed, read it.
- If it touches a file in **File structure**, the roadmap or `specs/deferred_items.md`, stop and
  ask.

Run: `gh pr list --state open`
Expected: no open PR touching a file in **File structure**. If one does, stop and ask.

- [ ] **Step 2: Build the worktree's environment**

Run: `uv sync`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number.

Run: `uv run python -c "import peft, polars, pydantic, pytorch_lightning, torch, transformers; print(peft.__version__, polars.__version__, pydantic.__version__, pytorch_lightning.__version__, torch.__version__, transformers.__version__)"`
Expected: `0.17.1 1.35.1 2.12.4 2.5.5 2.9.1 4.57.1`. If they differ, stop and ask.

- [ ] **Step 3: Run the baseline suite**

Run: `uv run pytest -n auto -q`
Expected: `1697 passed, 1 skipped`, measured at 52075f9 on 2026-10-03 (the skip needs CUDA).
- Each later full-suite run must pass with that one skip.
- Its count is this baseline plus the tests the plan has added by then, minus those it removed.
- The warnings count varies under xdist; ignore it.

- [ ] **Step 4: Check the real inputs, read-only**

Task 14 is the only reader of these files. Check them now so a missing input fails early.

Run: `shasum -a 256 /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`
Expected: `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36`.

Run: `cat ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main`
Expected: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Run: `ls /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json ~/Downloads/Data/QCEW/2022_US000_annual.csv ~/Downloads/Data/QCEW/2023_US000_annual.csv ~/Downloads/Data/QCEW/2024_US000_annual.csv ~/Downloads/Data/QCEW/2025_US000_annual.csv`
Expected: all five paths listed, no error.

- [ ] **Step 5: Route the tasks**

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–13 each get a fresh implementer and a task-reviewer. Give each implementer its task,
  **Global Constraints** and **Workspace**.
- Every code block in a task is exact: an implementer copies it. Each "Replace" text occurs
  exactly once in its file when its edit is made.
- Task 13 (documentation) may go to the docs-writer agent. Its acceptance checks are the gate.
- Task 14, **Final verification** and **Plan completion** run inline in the controller session.
  Task 14 runs a 1.5-hour training job and applies the stop-and-ask conditions.

### Task 1: Plan 6's two deferred fixes

`regressor_scores` counts each row's distinct repeats as well as its predictions (P24).
`coordinate_matrix`'s Lorentz refusal names the export form alone (P25), because
`tools diagnostics` raises it too.

**Files:**
- Modify: `src/naics_embedder/decision/scores.py:105-138`
- Modify: `src/naics_embedder/panels/regressor.py:336-340`
- Test: `tests/unit/test_decision_scores.py`, `tests/unit/test_regressor_panel.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `regressor_scores(predictions, repeats)`, unchanged in signature. It raises `ValueError`
    matching `do not have {repeats} predictions` when a row's prediction count or distinct
    repeat count differs from `repeats`.
  - `coordinate_matrix`'s refusal of Lorentz points reads "the coordinate table holds Lorentz
    points on a hyperboloid, not the export form (tangent coordinates at the origin for a
    hyperbolic arm)".

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_decision_scores.py`, after `test_a_row_without_every_repeat_is_refused`:

```python
def test_a_duplicated_repeat_beside_a_missing_one_is_refused():
    # Repeat 0 twice and repeat 1 never: the two predictions asked for, from one repeat
    predictions = _predictions(
        [
            _prediction('regressor_seen', 'covariates+embedding', 0, '111111', 2023, 1.0),
            _prediction('regressor_seen', 'covariates+embedding', 0, '111111', 2023, 2.0),
        ]
    )

    with pytest.raises(ValueError, match='do not have 2 predictions'):
        regressor_scores(predictions, repeats=2)

def test_a_repeat_counted_twice_beside_every_other_one_is_refused():
    # Repeats 0, 1 and 1: every repeat, but three predictions (both counts are checked)
    predictions = _predictions(
        [
            _prediction('regressor_seen', 'covariates+embedding', repeat, '111111', 2023, 1.0)
            for repeat in (0, 1, 1)
        ]
    )

    with pytest.raises(ValueError, match='do not have 2 predictions'):
        regressor_scores(predictions, repeats=2)
```

Add to `tests/unit/test_regressor_panel.py`, after
`test_lorentz_points_are_refused_and_the_export_form_is_read`:

```python
def test_the_lorentz_refusal_names_the_export_form_alone():
    # tools diagnostics raises it too, so it names no panel
    tangent = np.random.default_rng(1).normal(size=(4, 3))
    time = np.sqrt(1.0 + (tangent**2).sum(axis=1))
    lorentz = pl.DataFrame(
        {
            'code': ['11', '21', '22', '23'],
            'x0': time,
            **{
                f'x{i + 1}': tangent[:, i]
                for i in range(3)
            }
        }
    )

    with pytest.raises(ValueError, match='hyperboloid, not the export form') as excinfo:
        coordinate_matrix(lorentz)

    assert 'regressor panel' not in str(excinfo.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_decision_scores.py tests/unit/test_regressor_panel.py -q -k "duplicated_repeat or counted_twice or export_form_alone"`
Expected:
- `test_a_duplicated_repeat_beside_a_missing_one_is_refused` FAILS with "DID NOT RAISE".
- `test_the_lorentz_refusal_names_the_export_form_alone` FAILS: the message does not match.
- `test_a_repeat_counted_twice_beside_every_other_one_is_refused` already passes. It guards the
  prediction count, which the fix keeps.

- [ ] **Step 3: Count both in `regressor_scores`**

In `src/naics_embedder/decision/scores.py`, in `regressor_scores`'s docstring, replace:

```python
        ValueError: If a row is not a level-6 validation row of a regressor panel, or a row does not
            have exactly ``repeats`` predictions under every comparator of its panel.
```

with:

```python
        ValueError: If a row is not a level-6 validation row of a regressor panel, or a row does not
            have exactly ``repeats`` predictions, one per repeat, under every comparator of its
            panel.
```

Replace:

```python
        .agg(n=pl.len(), value=pl.col('error').mean())
    )
    # yapf: enable
    uneven = rows.filter(pl.col('n') != repeats)
    if uneven.height:
        first = uneven.row(0, named=True)
        raise ValueError(
            f'{uneven.height:,} rows do not have {repeats} predictions each, e.g. '
            f'{first["panel"]} {first["comparator"]} {first["code"]}/{first["feature_year"]}: '
            f'{first["n"]}'
        )
```

with:

```python
        .agg(n=pl.len(), distinct=pl.col('repeat').n_unique(), value=pl.col('error').mean())
    )
    # yapf: enable
    # One prediction per repeat: a duplicated repeat beside a missing one has the right count
    # but too few distinct repeats
    uneven = rows.filter((pl.col('n') != repeats) | (pl.col('distinct') != repeats))
    if uneven.height:
        first = uneven.row(0, named=True)
        raise ValueError(
            f'{uneven.height:,} rows do not have {repeats} predictions each, one per repeat, '
            f'e.g. {first["panel"]} {first["comparator"]} {first["code"]}/'
            f'{first["feature_year"]}: {first["n"]} predictions over {first["distinct"]} repeats'
        )
```

- [ ] **Step 4: Word the refusal around the export form**

In `src/naics_embedder/panels/regressor.py`, replace:

```python
            'the coordinate table holds Lorentz points on a hyperboloid; the regressor panel '
            'takes the export form (tangent coordinates at the origin for a hyperbolic arm)'
```

with:

```python
            'the coordinate table holds Lorentz points on a hyperboloid, not the export form '
            '(tangent coordinates at the origin for a hyperbolic arm)'
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_scores.py tests/unit/test_regressor_panel.py tests/unit/test_diagnostics.py -q`
Expected: all pass. The existing `match='Lorentz points'` and `match='do not have 2 predictions'`
tests still hold.

- [ ] **Step 6: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/decision/scores.py src/naics_embedder/panels/regressor.py tests/unit/test_decision_scores.py tests/unit/test_regressor_panel.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/decision/scores.py src/naics_embedder/panels/regressor.py tests/unit/test_decision_scores.py tests/unit/test_regressor_panel.py
git commit -m "fix(decision): count each row's distinct repeats; word the Lorentz refusal alone"
```

### Task 2: Field markers and tokenization cache format v3

**Files:**
- Create: `src/naics_embedder/text_model/fields.py`
- Modify: `src/naics_embedder/text_model/dataloader/tokenization_cache.py:1-65,208-219,280-292`
- Test: `tests/unit/test_fields.py` (create), `tests/unit/test_tokenization_cache.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - In `naics_embedder.text_model.fields`:
    - `FIELDS = ('title', 'description', 'excluded', 'examples', 'query')`;
    - `CHANNELS = FIELDS[:4]` and `QUERY = 'query'`;
    - `marker(field: str) -> str` and `marked_text(field: str, text: str) -> str`, both raising
      `ValueError` ("unknown field …") outside `FIELDS`;
    - `tokenize_field(tokenizer, field: str, text: Optional[str], max_length: int) -> Dict[str, Any]`,
      which returns `input_ids` and `attention_mask` (shape `(max_length,)`) and `present`
      (bool).
  - In `naics_embedder.text_model.dataloader.tokenization_cache`:
    - `CACHE_FORMAT = 'channels-v3'` and `SUMMARIES: Optional[str] = None`;
    - the sidecar gains `field_markers` and `summaries`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_fields.py`:

```python
'''
Field markers (spec R13) and tokenizing one field text as the cache stores it.
'''

import pytest
import torch
from transformers import AutoTokenizer

from naics_embedder.text_model.fields import (
    CHANNELS,
    FIELDS,
    QUERY,
    marked_text,
    marker,
    tokenize_field,
)

pytestmark = pytest.mark.unit

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'

@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(MINILM)

def test_the_five_fields_and_their_markers():
    assert FIELDS == ('title', 'description', 'excluded', 'examples', 'query')
    assert CHANNELS == FIELDS[:4]
    assert QUERY == 'query'
    assert marker('excluded') == 'excluded: '
    assert marked_text('title', 'Soybean Farming') == 'title: Soybean Farming'

def test_a_field_outside_the_marker_set_is_refused(tokenizer):
    with pytest.raises(ValueError, match='unknown field'):
        marked_text('summary', 'text')
    with pytest.raises(ValueError, match='unknown field'):
        tokenize_field(tokenizer, 'summary', None, 16)

def test_a_present_text_is_tokenized_with_its_marker(tokenizer):
    encoded = tokenize_field(tokenizer, 'title', 'Soybean Farming', 16)

    expected = tokenizer(
        'title: Soybean Farming',
        padding='max_length',
        truncation=True,
        max_length=16,
        return_tensors='pt',
    )
    assert encoded['present'] is True
    assert torch.equal(encoded['input_ids'], expected['input_ids'][0])
    assert torch.equal(encoded['attention_mask'], expected['attention_mask'][0])

@pytest.mark.parametrize('text', [None, '', '   '])
def test_an_absent_text_is_the_unmarked_empty_string(tokenizer, text):
    encoded = tokenize_field(tokenizer, 'examples', text, 16)

    assert encoded['present'] is False
    assert encoded['input_ids'].shape == (16, )
    assert int(encoded['attention_mask'].sum()) == 2  # [CLS] [SEP]
```

In `tests/unit/test_tokenization_cache.py`, add `from transformers import AutoTokenizer` after
`import torch`. Then append at the end of the file:

```python
# -------------------------------------------------------------------------------------------------
# Format v3: field markers and the summaries entry
# -------------------------------------------------------------------------------------------------

def _padded(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(
        text, padding='max_length', truncation=True, max_length=128, return_tensors='pt'
    )['input_ids'][0]

@pytest.mark.unit
def test_present_channels_are_cached_with_their_markers(sample_descriptions_parquet):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    cache = _build_tokenization_cache(
        sample_descriptions_parquet, 'sentence-transformers/all-MiniLM-L6-v2', 128
    )

    assert torch.equal(cache[0]['title']['input_ids'], _padded(tokenizer, 'title: Dog Food Manufacturing'))
    assert torch.equal(
        cache[2]['description']['input_ids'], _padded(tokenizer, 'description: Saw logs into lumber')
    )
    # An absent channel stays the unmarked empty string
    assert cache[0]['excluded']['present'] is False
    assert torch.equal(cache[0]['excluded']['input_ids'], _padded(tokenizer, ''))

@pytest.mark.unit
def test_the_sidecar_records_the_markers_and_null_summaries(tokenization_config, counted_builds):
    tokenization_cache(tokenization_config, **FINGERPRINTS)

    cache_path = Path(tokenization_config.output_path)
    sidecar = json.loads(cache_path.with_name(cache_path.name + '.meta.json').read_text())
    assert sidecar['cache_format'] == 'channels-v3'
    assert sidecar['field_markers'] == {
        'title': 'title: ',
        'description': 'description: ',
        'excluded': 'excluded: ',
        'examples': 'examples: ',
    }
    assert sidecar['summaries'] is None

@pytest.mark.unit
def test_a_cache_in_the_unmarked_v2_format_is_rebuilt(
    tokenization_config, sample_tokenization_cache, counted_builds
):
    cache_path = Path(tokenization_config.output_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample_tokenization_cache, cache_path)
    # The sidecar a Stage 5 cache wrote: the same inputs, unmarked, with no summaries entry
    earlier = {
        **FINGERPRINTS,
        'tokenizer_name': tokenization_config.tokenizer_name,
        'max_length': tokenization_config.max_length,
        'cache_format': 'channels-v2',
    }
    cache_path.with_name(cache_path.name + '.meta.json').write_text(json.dumps(earlier))

    tokenization_cache(tokenization_config, **FINGERPRINTS)

    assert len(counted_builds) == 1

@pytest.mark.unit
def test_a_cache_built_under_other_summaries_is_rebuilt(
    tokenization_config, counted_builds, monkeypatch
):
    tokenization_cache(tokenization_config, **FINGERPRINTS)
    monkeypatch.setattr(
        'naics_embedder.text_model.dataloader.tokenization_cache.SUMMARIES', 'a' * 64
    )

    tokenization_cache(tokenization_config, **FINGERPRINTS)

    assert len(counted_builds) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_fields.py tests/unit/test_tokenization_cache.py -q`
Expected:
- `test_fields.py` errors at collection with `ModuleNotFoundError: No module named 'naics_embedder.text_model.fields'`.
- The four new cache tests FAIL: the cache is unmarked, the sidecar has no `field_markers`, and
  the v2 sidecar still matches.

- [ ] **Step 3: Create `src/naics_embedder/text_model/fields.py`**

```python
'''
The text fields the shared encoder reads, and their markers (Req 14; spec R13).

A code has four channels: title, description, excluded and examples. A query is a fifth field. A
present text is marked with its field's name, ``'<field>: <text>'``, so one backbone can tell the
fields apart. An absent text (null or blank) is the empty string with ``present`` False and no
marker, so fusion can mask it (Req 9).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, Dict, Optional

FIELDS = ('title', 'description', 'excluded', 'examples', 'query')
CHANNELS = FIELDS[:4]
QUERY = 'query'

# -------------------------------------------------------------------------------------------------
# Markers
# -------------------------------------------------------------------------------------------------

def marker(field: str) -> str:
    '''
    The prefix that marks a field's text.

    Raises:
        ValueError: If the field is outside the marker set.
    '''

    if field not in FIELDS:
        raise ValueError(f'unknown field {field!r}; the marker set is {list(FIELDS)}')
    return f'{field}: '

def marked_text(field: str, text: str) -> str:
    '''A present text with its field's marker, ``'<field>: <text>'``.'''

    return marker(field) + text

# -------------------------------------------------------------------------------------------------
# Tokenization
# -------------------------------------------------------------------------------------------------

def tokenize_field(
    tokenizer: Any,
    field: str,
    text: Optional[str],
    max_length: int,
) -> Dict[str, Any]:
    '''
    Tokenize one field text as the tokenization cache stores it.

    A present text is tokenized with its marker. An absent one (null or blank) is the empty
    string, ``[CLS] [SEP]``, with ``present`` False and no marker. Either is truncated and padded
    to ``max_length``.

    Args:
        tokenizer: The backbone's tokenizer.
        field: One of ``FIELDS``.
        text: The field's text; None or blank is absent.
        max_length: Tokens kept, at most the backbone's trained window.

    Returns:
        ``input_ids`` and ``attention_mask`` of shape ``(max_length,)``, and ``present``.

    Raises:
        ValueError: If the field is outside the marker set.
    '''

    prefix = marker(field)
    present = bool((text or '').strip())
    encoded = tokenizer(
        prefix + text if present else '',
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    return {
        'input_ids': encoded['input_ids'].squeeze(0),
        'attention_mask': encoded['attention_mask'].squeeze(0),
        'present': present,
    }
```

- [ ] **Step 4: Move the cache to format v3**

In `src/naics_embedder/text_model/dataloader/tokenization_cache.py`, replace:

```python
from naics_embedder.utils.config import TokenizationConfig
from naics_embedder.utils.input_window import check_window

logger = logging.getLogger(__name__)

# How channels are encoded: every channel at the window, and an absent one as the empty string
# with ``present`` False. A cache in the earlier format (a placeholder text, titles at 24 tokens)
# records no format in its sidecar, so it is rebuilt.
CACHE_FORMAT = 'channels-v2'
```

with:

```python
from naics_embedder.text_model.fields import CHANNELS, marker, tokenize_field
from naics_embedder.utils.config import TokenizationConfig
from naics_embedder.utils.input_window import check_window

logger = logging.getLogger(__name__)

# How channels are encoded: each present channel as its marked text, '<field>: <text>'
# (text_model/fields.py), at the window, and an absent one as the empty string with ``present``
# False and no marker. A cache in an earlier format (unmarked texts, or a placeholder text)
# records another format in its sidecar, or none, so it is rebuilt.
CACHE_FORMAT = 'channels-v3'

# Stage 6b's summaries artifact, by hash: null until it lands. It is part of the cache's identity,
# so a cache built under other summaries is rebuilt.
SUMMARIES: Optional[str] = None
```

Replace the whole body of `_tokenize_text`, from its docstring to its `return`:

```python
    '''
    Tokenize one channel text, truncated and padded to ``max_length``.

    An absent channel (null or blank) is encoded as the empty string, ``[CLS] [SEP]``, never as
    placeholder text, and its ``present`` flag is False so fusion can mask it (Req 9).
    '''

    text = row.get(field) or ''
    present = bool(text.strip())
    if present:
        counter[field] += 1
    else:
        text = ''

    encoded = tokenizer(
        text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt'
    )

    encoding = {
        'input_ids': torch.squeeze(encoded['input_ids']),  # type: ignore
        'attention_mask': torch.squeeze(encoded['attention_mask']),  # type: ignore
        'present': present,
    }

    return encoding, counter
```

with:

```python
    '''
    Tokenize one channel text with its field marker, truncated and padded to ``max_length``.

    An absent channel (null or blank) is encoded as the empty string, ``[CLS] [SEP]``, with no
    marker and never as placeholder text, and its ``present`` flag is False so fusion can mask it
    (Req 9). ``fields.tokenize_field`` does the work, as it does for a query.
    '''

    encoding = tokenize_field(tokenizer, field, row.get(field), max_length)
    if encoding['present']:
        counter[field] += 1
    return encoding, counter
```

In `_cache_identity`, replace:

```python
        'cache_format': CACHE_FORMAT,
    }
```

with:

```python
        'cache_format': CACHE_FORMAT,
        'field_markers': {channel: marker(channel) for channel in CHANNELS},
        'summaries': SUMMARIES,
    }
```

In `tokenization_cache`'s docstring, replace:

```python
    A cache is reused only when its JSON sidecar records exactly the requested description and
    codebook fingerprints, tokenizer, and max length; otherwise it is rebuilt, because its source
    text is independently reproducible.
```

with:

```python
    A cache is reused only when its JSON sidecar records exactly the requested description and
    codebook fingerprints, tokenizer, max length, format, field markers and summaries; otherwise
    it is rebuilt, because its source text is independently reproducible.
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_fields.py tests/unit/test_tokenization_cache.py -q`
Expected: all pass, the existing cache tests included.

- [ ] **Step 6: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/fields.py src/naics_embedder/text_model/dataloader/tokenization_cache.py tests/unit/test_fields.py tests/unit/test_tokenization_cache.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/text_model/fields.py src/naics_embedder/text_model/dataloader/tokenization_cache.py tests/unit/test_fields.py tests/unit/test_tokenization_cache.py
git commit -m "feat(text_model): field markers and tokenization cache format v3"
```

### Task 3: One batch format with `present`

`stack_text_inputs` becomes the one builder of every code and query batch, and gives each field a
boolean `present` (4.2). The repaired collate's invalid rows are absent. Every hand-built test row
gains `present`, because the builder refuses a row without it.

**Files:**
- Modify: `src/naics_embedder/text_model/dataloader/datamodule.py:9,46-70,125-128,174-180,212-214`
- Modify (fixtures): `tests/unit/test_datamodule.py`, `tests/fixtures/epoch_datasets.py:23-30`,
  `tests/unit/test_naics_model.py:79-118`, `tests/integration/test_stage3_training_step.py:103-110`
- Test: `tests/unit/test_datamodule.py`

**Interfaces:**
- Consumes: `CHANNELS` from `naics_embedder.text_model.fields` (Task 2).
- Produces:
  `stack_text_inputs(embeddings: Sequence[Mapping[str, Mapping[str, Any]]], fields: Sequence[str] = CHANNELS) -> Dict[str, Dict[str, torch.Tensor]]`.
  - Per field it returns `input_ids` (B, L), `attention_mask` (B, L) and `present`, a
    `torch.bool` tensor of shape (B,).
  - It raises `ValueError` matching "no present flag" when a row lacks `present`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_datamodule.py`, add `stack_text_inputs` to the import from
`naics_embedder.text_model.dataloader.datamodule` (after `collate_fn`). Then add after
`test_collate_does_not_mutate_input_and_uses_invalid_rows`:

```python
def test_stack_text_inputs_carries_a_boolean_present_per_channel(make_embedding):
    absent = make_embedding()
    absent['excluded']['present'] = False

    batch = stack_text_inputs([make_embedding(), absent])

    assert batch['excluded']['present'].dtype == torch.bool
    assert batch['excluded']['present'].tolist() == [True, False]
    assert batch['title']['present'].tolist() == [True, True]

def test_stack_text_inputs_refuses_a_row_without_present(make_embedding):
    row = make_embedding()
    del row['title']['present']

    with pytest.raises(ValueError, match='no present flag'):
        stack_text_inputs([row])

def test_stack_text_inputs_builds_a_query_batch():
    row = {
        'query': {
            'input_ids': torch.tensor([101, 102]),
            'attention_mask': torch.ones(2, dtype=torch.long),
            'present': True,
        }
    }

    batch = stack_text_inputs([row], fields=('query', ))

    assert list(batch) == ['query']
    assert batch['query']['present'].tolist() == [True]

def test_repaired_collate_marks_invalid_rows_absent(make_repaired_batch_item):
    batch = collate_fn(
        [make_repaired_batch_item([101]),
         make_repaired_batch_item([201, 202, 203])],
        supervision_mode='repaired',
    )

    for channel in ('title', 'description', 'excluded', 'examples'):
        assert batch['candidate_inputs'][channel]['present'].tolist() == [
            True, False, False, True, True, True
        ]
```

- [ ] **Step 2: Give every hand-built row a `present` flag**

The new builder refuses a row without `present`, so each hand-built channel dict below gains one.
Each "Replace" is unique by the line above its dict.

`tests/unit/test_datamodule.py`, fixture `make_embedding`. Replace:

```python
    def _make(seq_len=128):
        return {
            ch: {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
            }
```

with:

```python
    def _make(seq_len=128):
        return {
            ch: {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
                'present': True,
            }
```

`tests/unit/test_datamodule.py`, fixture `make_repaired_batch_item`. Replace:

```python
    def encoded(value: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            channel: {
                'input_ids': torch.tensor([value, value + 1], dtype=torch.long),
                'attention_mask': torch.ones(2, dtype=torch.long),
            }
```

with:

```python
    def encoded(value: int) -> dict[str, dict[str, torch.Tensor]]:
        return {
            channel: {
                'input_ids': torch.tensor([value, value + 1], dtype=torch.long),
                'attention_mask': torch.ones(2, dtype=torch.long),
                'present': True,
            }
```

`tests/unit/test_datamodule.py`, the hierarchy token cache (about line 241). Replace:

```python
                    'input_ids': torch.full((4, ), int(index), dtype=torch.long),
                    'attention_mask': torch.ones(4, dtype=torch.long),
```

with:

```python
                    'input_ids': torch.full((4, ), int(index), dtype=torch.long),
                    'attention_mask': torch.ones(4, dtype=torch.long),
                    'present': True,
```

`tests/unit/test_datamodule.py`, fixture `mock_token_cache`. Replace:

```python
    def make_embedding(idx):
        return {
            ch: {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
            }
```

with:

```python
    def make_embedding(idx):
        return {
            ch: {
                'input_ids': torch.randint(0, 1000, (128, )),
                'attention_mask': torch.ones(128, dtype=torch.long),
                'present': True,
            }
```

`tests/unit/test_datamodule.py`, `test_collate_different_sequence_lengths`. Replace:

```python
    def make_item(seq_len):
        embedding = {
            ch: {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
            }
```

with:

```python
    def make_item(seq_len):
        embedding = {
            ch: {
                'input_ids': torch.randint(0, 1000, (seq_len, )),
                'attention_mask': torch.ones(seq_len, dtype=torch.long),
                'present': True,
            }
```

`tests/unit/test_datamodule.py`, the token cache near line 1152. Replace:

```python
        def make_embedding():
            return {
                ch: {
                    'input_ids': torch.randint(0, 1000, (128, )),
                    'attention_mask': torch.ones(128, dtype=torch.long),
                }
```

with:

```python
        def make_embedding():
            return {
                ch: {
                    'input_ids': torch.randint(0, 1000, (128, )),
                    'attention_mask': torch.ones(128, dtype=torch.long),
                    'present': True,
                }
```

`tests/fixtures/epoch_datasets.py`, `_embedding`. Replace:

```python
            'input_ids': torch.zeros(4, dtype=torch.long),
            'attention_mask': torch.ones(4, dtype=torch.long),
```

with:

```python
            'input_ids': torch.zeros(4, dtype=torch.long),
            'attention_mask': torch.ones(4, dtype=torch.long),
            'present': True,
```

`tests/unit/test_naics_model.py`, fixture `sample_training_batch`, a pre-stacked batch. Replace:

```python
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=test_device),
                'attention_mask': torch.ones(batch_size, seq_length, device=test_device),
```

with:

```python
                'input_ids': torch.randint(0, 1000, (batch_size, seq_length), device=test_device),
                'attention_mask': torch.ones(batch_size, seq_length, device=test_device),
                'present': torch.ones(batch_size, dtype=torch.bool, device=test_device),
```

`tests/unit/test_naics_model.py`, `_tokens`. Replace:

```python
            'input_ids': torch.randint(0, 1000, (seq_length, ), generator=generator),
            'attention_mask': torch.ones(seq_length, dtype=torch.long),
```

with:

```python
            'input_ids': torch.randint(0, 1000, (seq_length, ), generator=generator),
            'attention_mask': torch.ones(seq_length, dtype=torch.long),
            'present': True,
```

`tests/integration/test_stage3_training_step.py`, `_encoded`. Replace:

```python
            'input_ids': torch.tensor([value, value + 1], dtype=torch.long),
            'attention_mask': torch.ones(2, dtype=torch.long),
```

with:

```python
            'input_ids': torch.tensor([value, value + 1], dtype=torch.long),
            'attention_mask': torch.ones(2, dtype=torch.long),
            'present': True,
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_datamodule.py -q`
Expected: the four new tests fail. `stack_text_inputs` cannot be imported, so the module errors
at collection with `ImportError: cannot import name 'stack_text_inputs'`.

- [ ] **Step 4: Build `present` into the one builder**

In `src/naics_embedder/text_model/dataloader/datamodule.py`, replace:

```python
from typing import Any, Dict, List, Optional, Set, Tuple
```

with:

```python
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
```

Replace:

```python
from naics_embedder.utils.config import SamplingConfig, StreamingConfig, TokenizationConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

SUPERVISION_MODES = ('repaired', 'legacy_containment')
CHANNELS = ('title', 'description', 'excluded', 'examples')

# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def _stack_text_inputs(embeddings: List[Dict[str, Dict[str, torch.Tensor]]]
                       ) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        channel: {
            'input_ids': torch.stack([embedding[channel]['input_ids'] for embedding in embeddings]),
            'attention_mask': torch.stack(
                [embedding[channel]['attention_mask'] for embedding in embeddings]
            ),
        }
        for channel in CHANNELS
    }
```

with:

```python
from naics_embedder.text_model.fields import CHANNELS
from naics_embedder.utils.config import SamplingConfig, StreamingConfig, TokenizationConfig
from naics_embedder.utils.utilities import get_indices_codes

logger = logging.getLogger(__name__)

SUPERVISION_MODES = ('repaired', 'legacy_containment')

# -------------------------------------------------------------------------------------------------
# Collate function for DataLoader
# -------------------------------------------------------------------------------------------------

def stack_text_inputs(
    embeddings: Sequence[Mapping[str, Mapping[str, Any]]],
    fields: Sequence[str] = CHANNELS,
) -> Dict[str, Dict[str, torch.Tensor]]:
    '''
    Stack token rows into one batch: per field, ``input_ids``, ``attention_mask`` and a boolean
    ``present`` of shape (B,).

    Every code batch is built here (the collates, the export and the HGCN feeder), and a query
    batch too, under the field ``query``. The encoder reads presence from ``present``, never from
    the attention mask.

    Raises:
        ValueError: If a row's field has no ``present`` flag.
    '''

    for embedding in embeddings:
        for field in fields:
            if 'present' not in embedding[field]:
                raise ValueError(
                    f'a {field!r} token row has no present flag; rebuild the tokenization cache'
                )
    return {
        field: {
            'input_ids': torch.stack([embedding[field]['input_ids'] for embedding in embeddings]),
            'attention_mask': torch.stack(
                [embedding[field]['attention_mask'] for embedding in embeddings]
            ),
            'present': torch.tensor(
                [bool(embedding[field]['present']) for embedding in embeddings],
                dtype=torch.bool,
            ),
        }
        for field in fields
    }
```

Rename the five remaining calls of `_stack_text_inputs(` in this file (in `_collate_legacy` and
`_collate_repaired`) to `stack_text_inputs(`.

In `_collate_repaired`, replace:

```python
            'attention_mask': torch.zeros_like(template[channel]['attention_mask']),
        }
        for channel in CHANNELS
    }
```

with:

```python
            'attention_mask': torch.zeros_like(template[channel]['attention_mask']),
            'present': False,
        }
        for channel in CHANNELS
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_datamodule.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py tests/unit/test_streaming_dataset.py -q`
Expected: all pass.

- [ ] **Step 6: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/dataloader/datamodule.py tests/unit/test_datamodule.py tests/fixtures/epoch_datasets.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/text_model/dataloader/datamodule.py tests/unit/test_datamodule.py tests/fixtures/epoch_datasets.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py
git commit -m "feat(dataloader): one batch builder with a present flag per channel"
```

### Task 4: Fusion over the present channels

**Files:**
- Create: `src/naics_embedder/text_model/fusion.py`
- Test: `tests/unit/test_fusion.py` (create)

**Interfaces:**
- Consumes: `MixtureOfExperts(input_dim, hidden_dim=1024, num_experts=4, top_k=2)` from
  `naics_embedder.text_model.moe` (unchanged).
- Produces, in `naics_embedder.text_model.fusion`:
  - `FUSIONS = ('masked_mean', 'attention', 'moe')`.
  - The dataclass `FusionOutput(vector, gate_probs=None, top_k_indices=None)`.
  - `masked_mean(vectors (B, F, H), present (B, F) bool) -> (B, H)`.
  - Modules `MaskedMeanFusion()`, `AttentionFusion(hidden_size)` (parameter `query`) and
    `MoEFusion(hidden_size, num_experts=4, top_k=2, hidden_dim=1024)` (child `moe`). Each has
    `forward(vectors, present) -> FusionOutput`.
  - `build_fusion(name, hidden_size, *, num_experts=4, top_k=2, moe_hidden_dim=1024) -> nn.Module`,
    raising `ValueError` ("unknown fusion …") outside `FUSIONS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_fusion.py`:

```python
'''
Fusion over present channels (Req 14; Req 9): masked mean, attention pooling and the MoE ablation.
'''

import pytest
import torch

from naics_embedder.text_model.fusion import (
    FUSIONS,
    AttentionFusion,
    MaskedMeanFusion,
    MoEFusion,
    build_fusion,
    masked_mean,
)

pytestmark = pytest.mark.unit

# Row 0: channels 0 and 2 present; row 1: channel 1 only; row 2: no channel present
PRESENT = torch.tensor([[True, False, True], [False, True, False], [False, False, False]])

def _vectors() -> torch.Tensor:
    return torch.tensor(
        [
            [[1.0, 2.0], [100.0, 100.0], [3.0, 4.0]],
            [[50.0, 50.0], [5.0, 6.0], [70.0, 70.0]],
            [[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]],
        ]
    )

def test_masked_mean_averages_the_present_channels_only():
    fused = masked_mean(_vectors(), PRESENT)

    assert fused.tolist() == [[2.0, 3.0], [5.0, 6.0], [0.0, 0.0]]

def test_masked_mean_has_no_parameters():
    assert list(MaskedMeanFusion().parameters()) == []

def test_attention_weights_present_channels_by_a_softmax_of_their_scores():
    fusion = AttentionFusion(hidden_size=2)
    with torch.no_grad():
        fusion.query.copy_(torch.tensor([1.0, 0.0]))

    fused = fusion(_vectors(), PRESENT).vector

    # Row 0's present channels score 1 and 3
    weights = torch.softmax(torch.tensor([1.0, 3.0]), dim=0)
    expected = weights[0] * torch.tensor([1.0, 2.0]) + weights[1] * torch.tensor([3.0, 4.0])
    torch.testing.assert_close(fused[0], expected)
    torch.testing.assert_close(fused[1], torch.tensor([5.0, 6.0]))
    assert fused[2].tolist() == [0.0, 0.0]

def test_attention_starts_as_the_masked_mean():
    fused = AttentionFusion(hidden_size=2)(_vectors(), PRESENT).vector

    torch.testing.assert_close(fused, masked_mean(_vectors(), PRESENT))

@pytest.mark.parametrize('name', FUSIONS)
def test_an_absent_channel_never_contributes(name):
    torch.manual_seed(0)
    fusion = build_fusion(name, hidden_size=2, num_experts=2, top_k=1, moe_hidden_dim=4).eval()
    perturbed = _vectors().clone()
    perturbed[~PRESENT] = -1000.0

    assert torch.equal(fusion(_vectors(), PRESENT).vector, fusion(perturbed, PRESENT).vector)

@pytest.mark.parametrize('name', FUSIONS)
def test_a_row_with_no_present_channel_fuses_finitely_with_a_finite_gradient(name):
    torch.manual_seed(0)
    fusion = build_fusion(name, hidden_size=2, num_experts=2, top_k=1, moe_hidden_dim=4)
    vectors = _vectors().requires_grad_(True)

    fused = fusion(vectors, PRESENT).vector
    fused.sum().backward()

    assert torch.isfinite(fused).all()
    assert torch.isfinite(vectors.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in fusion.parameters()
    )

def test_moe_routes_the_masked_mean_and_is_the_only_option_with_gates():
    torch.manual_seed(0)
    fusion = MoEFusion(hidden_size=2, num_experts=2, top_k=1, hidden_dim=4).eval()

    output = fusion(_vectors(), PRESENT)

    expected, gate_probs, top_k_indices = fusion.moe(masked_mean(_vectors(), PRESENT))
    assert torch.equal(output.vector, expected)
    assert torch.equal(output.gate_probs, gate_probs)
    assert torch.equal(output.top_k_indices, top_k_indices)
    for other in (MaskedMeanFusion(), AttentionFusion(hidden_size=2)):
        result = other(_vectors(), PRESENT)
        assert result.gate_probs is None and result.top_k_indices is None

def test_an_unknown_fusion_is_refused():
    with pytest.raises(ValueError, match='unknown fusion'):
        build_fusion('concatenate', hidden_size=2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_fusion.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'naics_embedder.text_model.fusion'`.

- [ ] **Step 3: Create `src/naics_embedder/text_model/fusion.py`**

```python
'''
Fusion of a code's channel vectors into one vector (Req 14; Req 9; spec 4.1).

Every option reads only the present channels: an absent channel's slot is masked whatever it
holds, and a row with no present channel fuses to a finite vector with a finite gradient.

- ``masked_mean``, the default: the sum over present channels, divided by max(1, number present).
- ``attention``: a learned vector scores each present channel by its dot product, and a softmax
  over the present channels weights them.
- ``moe``, an ablation only (R12): the masked mean, then the mixture of experts on that vector.
  Only this option emits gate probabilities and expert indices.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from naics_embedder.text_model.moe import MixtureOfExperts

FUSIONS = ('masked_mean', 'attention', 'moe')

# -------------------------------------------------------------------------------------------------
# Output and the masked mean
# -------------------------------------------------------------------------------------------------

@dataclass
class FusionOutput:
    '''One fused vector per row, with the experts' gates under ``moe`` only.'''

    vector: torch.Tensor
    gate_probs: Optional[torch.Tensor] = None
    top_k_indices: Optional[torch.Tensor] = None

def masked_mean(vectors: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    '''
    The mean over present channels, (B, F, H) and (B, F) to (B, H).

    A row with no present channel is zeros, divided by one, never by zero.
    '''

    weights = present.to(vectors.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return (vectors * weights).sum(dim=1) / count

# -------------------------------------------------------------------------------------------------
# Fusion options
# -------------------------------------------------------------------------------------------------

class MaskedMeanFusion(nn.Module):
    '''The default fusion. It has no parameters, so no absent channel can contribute.'''

    def forward(self, vectors: torch.Tensor, present: torch.Tensor) -> FusionOutput:
        return FusionOutput(masked_mean(vectors, present))

class AttentionFusion(nn.Module):
    '''
    Attention pooling over the present channels.

    The learned vector starts at zeros, so the pooling starts as the masked mean. An absent
    channel scores the dtype's minimum and its weight is then zeroed, so a row with no present
    channel has zero weights, a zero vector and a finite gradient.

    Args:
        hidden_size: The width of the channel vectors.
    '''

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, vectors: torch.Tensor, present: torch.Tensor) -> FusionOutput:
        scores = vectors @ self.query
        # The score's own dtype bounds the fill: under autocast it can be narrower than the input
        scores = scores.masked_fill(~present, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * present.to(scores.dtype)
        return FusionOutput((weights.unsqueeze(-1) * vectors).sum(dim=1))

class MoEFusion(nn.Module):
    '''
    The MoE ablation (R12): the masked mean, then the mixture of experts on the fused vector.

    The experts belong to the fusion step; the load-balancing term reads their gates (R11).

    Args:
        hidden_size: The width of the channel vectors, the experts' input and output.
        num_experts: The number of experts.
        top_k: The experts each row is routed to.
        hidden_dim: The experts' hidden width.
    '''

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.moe = MixtureOfExperts(
            input_dim=hidden_size,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(self, vectors: torch.Tensor, present: torch.Tensor) -> FusionOutput:
        output, gate_probs, top_k_indices = self.moe(masked_mean(vectors, present))
        return FusionOutput(output, gate_probs, top_k_indices)

def build_fusion(
    name: str,
    hidden_size: int,
    *,
    num_experts: int = 4,
    top_k: int = 2,
    moe_hidden_dim: int = 1024,
) -> nn.Module:
    '''
    The fusion module that ``model.fusion`` names.

    Args:
        name: One of ``FUSIONS``.
        hidden_size: The width of the channel vectors.
        num_experts: The number of experts, under ``moe`` only.
        top_k: The experts each row is routed to, under ``moe`` only.
        moe_hidden_dim: The experts' hidden width, under ``moe`` only.

    Raises:
        ValueError: If the name is outside ``FUSIONS``.
    '''

    if name == 'masked_mean':
        return MaskedMeanFusion()
    if name == 'attention':
        return AttentionFusion(hidden_size)
    if name == 'moe':
        return MoEFusion(
            hidden_size, num_experts=num_experts, top_k=top_k, hidden_dim=moe_hidden_dim
        )
    raise ValueError(f'unknown fusion {name!r}; expected one of {list(FUSIONS)}')
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_fusion.py -q`
Expected: all pass.

- [ ] **Step 5: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/fusion.py tests/unit/test_fusion.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/text_model/fusion.py tests/unit/test_fusion.py
git commit -m "feat(text_model): masked-mean, attention and MoE fusion over present channels"
```

### Task 5: The shared encoder and the interim head

`SharedEncoder` runs one LoRA-adapted backbone over every present text of a batch in one call,
fuses the channels, and maps the fused vector through one `Linear(hidden → d)` to a
parameter-free head (4.1). The model keeps the four-copy encoder until Task 7, so this task only
adds modules and their tests.

The tests build the backbone from a one-layer BERT (§6). `tests/fixtures/shared_encoder.py`
starts here with that backbone, and Task 9 adds the export fixtures to it.

**Files:**
- Modify: `src/naics_embedder/text_model/hyperbolic.py:80-82` (insert `HyperbolicHead` before
  the projection section)
- Create: `src/naics_embedder/text_model/shared_encoder.py`
- Create: `tests/fixtures/shared_encoder.py`
- Modify: `tests/conftest.py:14-18`
- Rewrite: `tests/unit/test_encoder.py`
- Modify: `tests/unit/test_hyperbolic.py:13-20,148-150`

**Interfaces:**
- Consumes: `FIELDS`, `CHANNELS`, `QUERY` (Task 2); `stack_text_inputs` (Task 3, tests only);
  `FUSIONS`, `build_fusion` and `FusionOutput` (Task 4).
- Produces:
  - In `naics_embedder.text_model.hyperbolic`: `HyperbolicHead(curvature: float = 1.0,
    max_norm: float = 2.0)`.
    - It has no parameters, and the class attribute `distance = 'lorentz'`.
    - `forward(tangent (B, d)) -> Tuple[tangent (B, d), embedding (B, d + 1)]`: the capped
      tangent, then its exp map at the origin.
  - In `naics_embedder.text_model.shared_encoder`:
    - `DIMENSIONS = (8, 16, 32)`;
    - `load_base_model(name: str) -> PreTrainedModel`, which tests replace;
    - `SharedEncoder(base_model_name='sentence-transformers/all-MiniLM-L6-v2', lora_r=8,
      lora_alpha=16, lora_dropout=0.1, fusion='masked_mean', dimension=16, num_experts=4, top_k=2,
      moe_hidden_dim=1024, curvature=1.0, use_gradient_checkpointing=True)`.
      - Children, in order: `backbone` (a `PeftModel`), `fusion` (the module), `projection`
        (`nn.Linear(hidden_size, dimension)`) and `head` (`HyperbolicHead`).
      - Attributes: `hidden_size`, `dimension`, `curvature`, `fusion_name` and
        `backbone_revision` (the config's `_commit_hash`, or None).
      - `forward(channel_inputs) -> Dict[str, torch.Tensor]` returns `embedding` (B, d + 1) and
        `tangent` (B, d), plus `gate_probs` and `top_k_indices` under `moe` only.
      - It raises `ValueError` on an unknown fusion or dimension ("unknown fusion",
        "unknown dimension"), a batch with no field ("at least one field"), a field outside
        `FIELDS` ("unknown field") and a field without `present` ("no present flag").
  - In `tests.fixtures.shared_encoder`: `TINY_HIDDEN = 8`, `tiny_bert(name='tiny-bert') ->
    BertModel` and the fixture `tiny_backbone`, which makes every `SharedEncoder` load
    `tiny_bert`.

- [ ] **Step 1: Write the fixture module and register it**

Create `tests/fixtures/shared_encoder.py`:

```python
'''
A tiny backbone for the shared encoder's tests, so they download nothing (spec §6).

``tiny_bert`` builds a one-layer BERT whose vocabulary is MiniLM's, so token rows from the real
tokenizer fit it. ``tiny_backbone`` makes every ``SharedEncoder`` a test builds load it in place of
MiniLM.
'''

import pytest
import torch
from transformers import BertConfig, BertModel

TINY_HIDDEN = 8

def tiny_bert(name: str = 'tiny-bert') -> BertModel:
    '''A seeded one-layer BERT of width 8 over MiniLM's 30,522-token vocabulary.'''

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=30522,
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=512,
    )
    return BertModel(config)

@pytest.fixture
def tiny_backbone(monkeypatch):
    '''Every ``SharedEncoder`` built in the test loads ``tiny_bert`` instead of MiniLM.'''

    monkeypatch.setattr('naics_embedder.text_model.shared_encoder.load_base_model', tiny_bert)
    return tiny_bert
```

The fixture names its target as a string, so registering the module does not import
`shared_encoder` before it exists.

In `tests/conftest.py`, replace:

```python
pytest_plugins = (
    'tests.fixtures.naics_sources',
    'tests.fixtures.regressor_panel',
    'tests.fixtures.supervision',
)
```

with:

```python
pytest_plugins = (
    'tests.fixtures.naics_sources',
    'tests.fixtures.regressor_panel',
    'tests.fixtures.shared_encoder',
    'tests.fixtures.supervision',
)
```

- [ ] **Step 2: Write the failing tests**

Replace the whole of `tests/unit/test_encoder.py` with:

```python
'''
The shared encoder (Req 14; spec 4.1), on a one-layer BERT so these tests download nothing.

One backbone serves codes and queries; an absent channel never reaches the output; exactly one
affine map sits between fusion and the point; and a step reaches every adapter and the projection.
'''

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from peft.tuners.lora import LoraLayer
from transformers import PreTrainedModel

from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.fields import CHANNELS, QUERY
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.hyperbolic import HyperbolicHead, check_lorentz_manifold_validity
from naics_embedder.text_model.shared_encoder import DIMENSIONS, SharedEncoder
from tests.fixtures.shared_encoder import TINY_HIDDEN, tiny_bert

pytestmark = pytest.mark.unit

WIDTH = 8

def _text(ids):
    '''A present text: these token ids, right-padded to WIDTH.'''

    input_ids = torch.zeros(WIDTH, dtype=torch.long)
    input_ids[:len(ids)] = torch.tensor(ids)
    attention_mask = torch.zeros(WIDTH, dtype=torch.long)
    attention_mask[:len(ids)] = 1
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'present': True}

def _absent():
    '''An absent channel as the cache stores it: ``[CLS] [SEP]`` with ``present`` False.'''

    return {**_text([101, 102]), 'present': False}

def _code(**texts):
    '''A code whose named channels hold these token ids; its other channels are absent.'''

    return {
        channel: _text(texts[channel]) if channel in texts else _absent()
        for channel in CHANNELS
    }

# Code 0 has a title and examples; code 1 has a description only
CODES = [
    _code(title=[101, 2001, 2002, 102], examples=[101, 2003, 2004, 2005, 102]),
    _code(description=[101, 2006, 2007, 102]),
]

@pytest.fixture
def make_encoder(tiny_backbone):
    '''Build a ``SharedEncoder`` on the tiny backbone, with these settings overridden.'''

    def make(**overrides):
        settings = {
            'base_model_name': 'tiny-bert',
            'lora_r': 2,
            'lora_alpha': 4,
            'lora_dropout': 0.0,
            'fusion': 'masked_mean',
            'dimension': 8,
            'num_experts': 2,
            'top_k': 1,
            'moe_hidden_dim': 4,
            'curvature': 1.0,
            'use_gradient_checkpointing': False,
        }
        return SharedEncoder(**{**settings, **overrides})

    return make

# -------------------------------------------------------------------------------------------------
# Structure
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fusion', FUSIONS)
def test_one_backbone_then_fusion_one_affine_map_and_a_parameter_free_head(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion, dimension=16)

    # From the fused vector to the point: the projection, then the head
    assert [name for name, _ in encoder.named_children()] == [
        'backbone', 'fusion', 'projection', 'head'
    ]
    assert sum(isinstance(module, PreTrainedModel) for module in encoder.modules()) == 1
    assert isinstance(encoder.projection, nn.Linear)
    assert (encoder.projection.in_features, encoder.projection.out_features) == (TINY_HIDDEN, 16)
    assert isinstance(encoder.head, HyperbolicHead)
    assert list(encoder.head.parameters()) == []
    assert encoder.fusion_name == fusion

def test_one_lora_adapter_wraps_every_linear_layer_of_the_backbone(make_encoder):
    encoder = make_encoder()
    linear_layers = [module for module in tiny_bert().modules() if isinstance(module, nn.Linear)]
    adapted = [module for module in encoder.backbone.modules() if isinstance(module, LoraLayer)]

    assert list(encoder.backbone.peft_config) == ['default']
    # The pooler's dense layer too, which mean pooling never reads (P8)
    assert len(adapted) == len(linear_layers) == 7

def test_an_unknown_fusion_or_dimension_is_refused(make_encoder):
    assert DIMENSIONS == (8, 16, 32)
    with pytest.raises(ValueError, match='unknown fusion'):
        make_encoder(fusion='concatenate')
    with pytest.raises(ValueError, match='unknown dimension'):
        make_encoder(dimension=12)

# -------------------------------------------------------------------------------------------------
# One encoder for codes and queries
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fusion', FUSIONS)
def test_a_one_field_batch_encodes_as_a_code_with_only_that_channel(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion).eval()
    text = [101, 2001, 2002, 2003, 102]
    one_code = stack_text_inputs([_code(excluded=text)])
    one_field = stack_text_inputs([{'excluded': _text(text)}], fields=('excluded', ))
    query = stack_text_inputs([{QUERY: _text(text)}], fields=(QUERY, ))

    with torch.no_grad():
        expected, *others = [encoder(batch) for batch in (one_code, one_field, query)]

    for output in others:
        assert output.keys() == expected.keys()
        for key in output:
            assert torch.equal(output[key], expected[key])

# -------------------------------------------------------------------------------------------------
# Masking
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fusion', FUSIONS)
def test_perturbing_an_absent_channel_leaves_the_output_bit_identical(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion).eval()
    batch = stack_text_inputs(CODES)
    perturbed = stack_text_inputs(CODES)
    for channel in CHANNELS:
        absent = ~perturbed[channel]['present']
        perturbed[channel]['input_ids'][absent] = 2999
        perturbed[channel]['attention_mask'][absent] = 1

    with torch.no_grad():
        clean, noisy = encoder(batch), encoder(perturbed)

    assert clean.keys() == noisy.keys()
    for key in clean:
        assert torch.equal(clean[key], noisy[key])

def test_absent_texts_and_padding_never_enter_the_backbone(make_encoder, monkeypatch):
    encoder = make_encoder().eval()
    calls = []
    forward = encoder.backbone.forward

    def spy(**inputs):
        calls.append(inputs['input_ids'].clone())
        return forward(**inputs)

    monkeypatch.setattr(encoder.backbone, 'forward', spy)
    with torch.no_grad():
        encoder(stack_text_inputs(CODES))

    # The three present texts in one call, field by field, trimmed to the longest (five tokens)
    [input_ids] = calls
    assert input_ids.tolist() == [
        [101, 2001, 2002, 102, 0],
        [101, 2006, 2007, 102, 0],
        [101, 2003, 2004, 2005, 102],
    ]

@pytest.mark.parametrize('fusion', FUSIONS)
def test_a_code_with_no_present_channel_encodes_finitely(make_encoder, fusion):
    encoder = make_encoder(fusion=fusion)

    output = encoder(stack_text_inputs([CODES[0], _code()]))
    output['embedding'].sum().backward()

    assert torch.isfinite(output['embedding']).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )

def test_a_batch_with_no_present_text_never_calls_the_backbone(make_encoder, monkeypatch):
    encoder = make_encoder().eval()
    monkeypatch.setattr(encoder.backbone, 'forward', Mock(side_effect=AssertionError('backbone')))

    with torch.no_grad():
        output = encoder(stack_text_inputs([_code(), _code()]))

    assert output['embedding'].shape == (2, 9)
    assert torch.isfinite(output['embedding']).all()

# -------------------------------------------------------------------------------------------------
# Outputs and refusals
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('fusion', FUSIONS)
def test_only_the_moe_fusion_emits_gates(make_encoder, fusion):
    with torch.no_grad():
        output = make_encoder(fusion=fusion).eval()(stack_text_inputs(CODES))

    gates = {'gate_probs', 'top_k_indices'} if fusion == 'moe' else set()
    assert set(output) == {'embedding', 'tangent'} | gates

@pytest.mark.parametrize('dimension', DIMENSIONS)
def test_the_point_is_the_head_of_the_capped_tangent(make_encoder, dimension):
    with torch.no_grad():
        output = make_encoder(dimension=dimension).eval()(stack_text_inputs(CODES))

    assert output['tangent'].shape == (2, dimension)
    assert output['embedding'].shape == (2, dimension + 1)
    assert (output['tangent'].norm(dim=1) <= 2.0 + 1e-6).all()
    is_valid, _, _ = check_lorentz_manifold_validity(output['embedding'], curvature=1.0)
    assert is_valid
    _, expected = HyperbolicHead(curvature=1.0)(output['tangent'])
    torch.testing.assert_close(output['embedding'], expected)

def test_a_malformed_batch_is_refused(make_encoder):
    encoder = make_encoder()
    batch = stack_text_inputs(CODES)
    del batch['title']['present']

    with pytest.raises(ValueError, match='no present flag'):
        encoder(batch)
    with pytest.raises(ValueError, match='unknown field'):
        encoder({'summary': stack_text_inputs(CODES)['title']})
    with pytest.raises(ValueError, match='at least one field'):
        encoder({})

# -------------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('checkpointing', [False, True])
def test_a_step_reaches_every_adapter_and_the_projection(make_encoder, checkpointing):
    encoder = make_encoder(use_gradient_checkpointing=checkpointing).train()

    encoder(stack_text_inputs(CODES))['embedding'].sum().backward()

    assert encoder.projection.weight.grad.abs().sum() > 0
    adapters = {
        name: parameter
        for name, parameter in encoder.backbone.named_parameters() if 'lora_B' in name
    }
    assert adapters
    for name, parameter in adapters.items():
        if '.pooler.' in name:
            # Mean pooling never reads the pooler, so its adapter gets no gradient (P8)
            assert parameter.grad is None, name
        else:
            # PEFT starts lora_B at zero, so lora_A's first gradient is exactly zero (P9)
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
```

In `tests/unit/test_hyperbolic.py`, replace:

```python
from naics_embedder.text_model.hyperbolic import (
    HyperbolicProjection,
    LorentzDistance,
```

with:

```python
from naics_embedder.text_model.hyperbolic import (
    HyperbolicHead,
    HyperbolicProjection,
    LorentzDistance,
```

Then replace:

```python
# -------------------------------------------------------------------------------------------------
# HyperbolicProjection Tests
# -------------------------------------------------------------------------------------------------
```

with:

```python
# -------------------------------------------------------------------------------------------------
# HyperbolicHead Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHyperbolicHead:
    '''The interim geometry head: no parameters, the cap, then the exp map at the origin.'''

    def test_the_head_has_no_parameters_and_names_its_distance(self):
        head = HyperbolicHead(curvature=1.0)

        assert list(head.parameters()) == []
        assert head.distance == 'lorentz'

    def test_a_tangent_inside_the_cap_passes_unchanged(self):
        tangent = torch.tensor([[0.3, -0.4], [1.0, 1.0]])

        capped, embedding = HyperbolicHead(curvature=1.0)(tangent)

        assert torch.equal(capped, tangent)
        assert embedding.shape == (2, 3)

    def test_a_long_tangent_is_scaled_to_the_cap(self):
        capped, _ = HyperbolicHead(curvature=1.0, max_norm=2.0)(torch.tensor([[3.0, 4.0]]))

        torch.testing.assert_close(capped, torch.tensor([[1.2, 1.6]]))

    @pytest.mark.parametrize('curvature', [0.5, 1.0, 2.0])
    def test_the_point_is_the_exp_map_of_the_capped_tangent(self, curvature):
        tangent = torch.tensor([[0.3, -0.4], [3.0, 4.0]])

        capped, embedding = HyperbolicHead(curvature=curvature)(tangent)

        # LorentzOps takes a (B, d + 1) tangent and ignores its time slot
        padded = torch.cat([torch.zeros(2, 1), capped], dim=1)
        torch.testing.assert_close(embedding, LorentzOps.exp_map_zero(padded, c=curvature))
        is_valid, _, _ = check_lorentz_manifold_validity(embedding, curvature=curvature)
        assert is_valid

# -------------------------------------------------------------------------------------------------
# HyperbolicProjection Tests
# -------------------------------------------------------------------------------------------------
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_encoder.py tests/unit/test_hyperbolic.py -q`
Expected: both modules error at collection:
- `ModuleNotFoundError: No module named 'naics_embedder.text_model.shared_encoder'`;
- `ImportError: cannot import name 'HyperbolicHead'`.

- [ ] **Step 4: Add the head**

In `src/naics_embedder/text_model/hyperbolic.py`, replace:

```python
# -------------------------------------------------------------------------------------------------
# Hyperbolic Projection to Lorentz Model
# -------------------------------------------------------------------------------------------------
```

with:

```python
# -------------------------------------------------------------------------------------------------
# Interim geometry head
# -------------------------------------------------------------------------------------------------

class HyperbolicHead(nn.Module):
    '''
    The interim hyperbolic head (spec 4.1). It has no parameters.

    It rescales each tangent vector to norm at most ``max_norm``, the interim harness's cap
    (Stage 7 removes it, Req 13), then maps it to the Lorentz hyperboloid by the exp map at the
    origin. ``distance`` names the decoding distance for its points.

    Args:
        curvature: The hyperboloid's curvature c.
        max_norm: The cap on a tangent vector's norm.
    '''

    distance = 'lorentz'

    def __init__(self, curvature: float = 1.0, max_norm: float = 2.0):
        super().__init__()
        self.curvature = curvature
        self.max_norm = max_norm

    def forward(self, tangent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Cap the tangent vectors, then map them to the hyperboloid.

        Args:
            tangent: Tangent vectors at the origin, (B, d), without a time coordinate.

        Returns:
            ``(tangent, embedding)``: the capped tangent (B, d), which the export writes, and the
            Lorentz point (B, d + 1), which the interim loss reads.
        '''

        norm = torch.norm(tangent, p=2, dim=1, keepdim=True)
        scale = torch.where(
            norm > self.max_norm, self.max_norm / (norm + 1e-8), torch.ones_like(norm)
        )
        tangent = tangent * scale
        curvature = torch.tensor(self.curvature, device=tangent.device, dtype=tangent.dtype)
        _mark_cudagraph_step()
        return tangent, _exp_map_zero_compiled(tangent, torch.sqrt(curvature))

# -------------------------------------------------------------------------------------------------
# Hyperbolic Projection to Lorentz Model
# -------------------------------------------------------------------------------------------------
```

`Tuple` is already imported there, from `typing`.

- [ ] **Step 5: Create `src/naics_embedder/text_model/shared_encoder.py`**

```python
'''
The shared encoder (Req 14; spec 4.1): one backbone with one LoRA adapter, for every field.

Per code the path is:

1. field-marked channel texts;
2. one backbone with one LoRA adapter;
3. an attention-masked mean over each text's tokens;
4. fusion over the present channels;
5. one ``Linear(hidden → d)``;
6. the geometry head, which gives the point.

A query is a one-field batch, ``{'query': …}``, and takes the same path.

A batch's present (row, field) texts are gathered into one backbone call, trimmed to the longest
of them, so neither an absent text nor a padding column enters the backbone. Presence comes from
each field's ``present`` flag, never from the attention mask (Req 9).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from typing import Dict, List, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, PreTrainedModel

from naics_embedder.text_model.fields import FIELDS
from naics_embedder.text_model.fusion import FUSIONS, build_fusion
from naics_embedder.text_model.hyperbolic import HyperbolicHead

logger = logging.getLogger(__name__)

DIMENSIONS = (8, 16, 32)

# -------------------------------------------------------------------------------------------------
# Backbone
# -------------------------------------------------------------------------------------------------

def load_base_model(name: str) -> PreTrainedModel:
    '''The backbone's pretrained weights. Tests replace this with a one-layer BERT.'''

    return AutoModel.from_pretrained(name)

# -------------------------------------------------------------------------------------------------
# Shared encoder
# -------------------------------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    '''
    One LoRA-adapted backbone for every field, then fusion, one affine map and the head.

    Args:
        base_model_name: The backbone's Hugging Face name.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.
        fusion: One of ``FUSIONS``: ``masked_mean`` (the default), ``attention`` or ``moe``.
        dimension: The embedding dimension, one of ``DIMENSIONS``.
        num_experts: The number of experts, under ``moe`` only.
        top_k: The experts each code is routed to, under ``moe`` only.
        moe_hidden_dim: The experts' hidden width, under ``moe`` only.
        curvature: The head's curvature.
        use_gradient_checkpointing: Recompute the backbone's activations in the backward pass.

    Raises:
        ValueError: If the fusion or the dimension is outside its set.
    '''

    def __init__(
        self,
        base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        fusion: str = 'masked_mean',
        dimension: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
        moe_hidden_dim: int = 1024,
        curvature: float = 1.0,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if fusion not in FUSIONS:
            raise ValueError(f'unknown fusion {fusion!r}; expected one of {list(FUSIONS)}')
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')

        base_model = load_base_model(base_model_name)
        self.hidden_size = int(base_model.config.hidden_size)
        # The resolved snapshot, read as panels/text_only.load_backbone reads it
        self.backbone_revision = getattr(base_model.config, '_commit_hash', None)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules='all-linear',
            lora_dropout=lora_dropout,
            bias='none',
            task_type='FEATURE_EXTRACTION',
        )
        self.backbone = get_peft_model(base_model, lora_config)
        if use_gradient_checkpointing:
            # Both calls are needed: checkpointed blocks reach the adapter only through inputs
            # that require grad
            self.backbone.enable_input_require_grads()
            self.backbone.base_model.gradient_checkpointing_enable()

        self.fusion_name = fusion
        self.fusion = build_fusion(
            fusion,
            self.hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
        )
        self.projection = nn.Linear(self.hidden_size, dimension)
        self.head = HyperbolicHead(curvature=curvature)
        self.dimension = dimension
        self.curvature = curvature

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            'Shared encoder initialized:\n'
            f'  • backbone: {base_model_name} (hidden size {self.hidden_size})\n'
            f'  • fusion: {fusion}; dimension: {dimension}\n'
            f'  • trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n'
        )

    def forward(self, channel_inputs: Mapping[str, Mapping[str, torch.Tensor]]
                ) -> Dict[str, torch.Tensor]:
        '''
        Encode a batch of codes (the four channels) or of queries (the field ``query``).

        Args:
            channel_inputs: Per field, ``input_ids`` and ``attention_mask`` (B, L) and a boolean
                ``present`` (B,), as ``stack_text_inputs`` builds them.

        Returns:
            ``embedding`` (B, d + 1), the Lorentz point; ``tangent`` (B, d), the capped tangent
            vector at the origin; and ``gate_probs`` and ``top_k_indices`` under ``moe`` only.

        Raises:
            ValueError: If the batch has no field, a field outside the marker set, or a field
                without its ``present`` flag.
        '''

        if not channel_inputs:
            raise ValueError('an encoder batch needs at least one field')
        for field, inputs in channel_inputs.items():
            if field not in FIELDS:
                raise ValueError(f'unknown field {field!r}; the marker set is {list(FIELDS)}')
            if 'present' not in inputs:
                raise ValueError(
                    f'the {field!r} batch has no present flag; build it with stack_text_inputs'
                )
        # A fixed field order, so the output never depends on the mapping's key order
        fields = [field for field in FIELDS if field in channel_inputs]
        device = channel_inputs[fields[0]]['input_ids'].device
        present = torch.stack(
            [
                channel_inputs[field]['present'].to(device=device, dtype=torch.bool)
                for field in fields
            ],
            dim=1,
        )

        pooled = self._pool_present(channel_inputs, fields, present)
        fused = self.fusion(pooled, present)
        tangent, embedding = self.head(self.projection(fused.vector))
        output = {'embedding': embedding, 'tangent': tangent}
        if fused.gate_probs is not None:
            output['gate_probs'] = fused.gate_probs
            output['top_k_indices'] = fused.top_k_indices
        return output

    def _pool_present(
        self,
        channel_inputs: Mapping[str, Mapping[str, torch.Tensor]],
        fields: List[str],
        present: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Mean-pool every present (row, field) text in one backbone call, to (B, F, H).

        Present texts are gathered field by field and right-padded to one width, which is then
        trimmed to the longest text, so neither an absent text nor a padding column enters the
        backbone (P4). Absent slots stay zeros, and fusion masks them.
        '''

        batch_size = present.shape[0]
        width = max(int(channel_inputs[field]['input_ids'].shape[1]) for field in fields)
        input_ids: List[torch.Tensor] = []
        attention_mask: List[torch.Tensor] = []
        for column, field in enumerate(fields):
            rows = present[:, column]
            ids = channel_inputs[field]['input_ids'][rows]
            mask = channel_inputs[field]['attention_mask'][rows]
            input_ids.append(F.pad(ids, (0, width - ids.shape[1])))
            attention_mask.append(F.pad(mask, (0, width - mask.shape[1])))
        ids = torch.cat(input_ids)
        mask = torch.cat(attention_mask)
        if ids.shape[0] == 0:
            # No present text in the batch: every slot is absent, and fusion masks them all
            return self.projection.weight.new_zeros((batch_size, len(fields), self.hidden_size))

        used = torch.nonzero(mask.ne(0).any(dim=0))
        length = int(used[-1]) + 1 if used.numel() else 1
        ids, mask = ids[:, :length], mask[:, :length]
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        vectors = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)

        # Field-major slots, in the order the texts were gathered
        flat = vectors.new_zeros((len(fields) * batch_size, vectors.shape[1]))
        flat[present.t().reshape(-1)] = vectors
        return flat.reshape(len(fields), batch_size, -1).transpose(0, 1)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_encoder.py tests/unit/test_hyperbolic.py -q`
Expected: all pass.

- [ ] **Step 7: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/hyperbolic.py src/naics_embedder/text_model/shared_encoder.py tests/fixtures/shared_encoder.py tests/conftest.py tests/unit/test_encoder.py tests/unit/test_hyperbolic.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/text_model/hyperbolic.py src/naics_embedder/text_model/shared_encoder.py tests/fixtures/shared_encoder.py tests/conftest.py tests/unit/test_encoder.py tests/unit/test_hyperbolic.py
git commit -m "feat(text_model): the shared encoder and its parameter-free hyperbolic head"
```

### Task 6: Router mining and the load-balancing term only under MoE (R10, R11)

The model gains `fusion`, the one switch for the MoE-only machinery (P10). Under any other fusion
the router-guided miner is never consulted, so the geometric miner takes every mining slot (R10).
The load-balancing term is computed, added and logged only under `moe` (R11).

This task runs on the four-copy encoder, which still emits gates under every setting. The tests
therefore show the gating follows `fusion`, not the presence of gates. Task 7 then removes the
gates from the default path.

**Files:**
- Modify: `src/naics_embedder/text_model/naics_model.py:46,119,132,172,214-216,576-590,645-652`
- Modify: `src/naics_embedder/text_model/mixins/curriculum.py:92-93,152,193`
- Modify: `src/naics_embedder/text_model/mixins/logging.py:274,366-371,385-403`
- Modify: `src/naics_embedder/text_model/mixins/loss.py:461-489`
- Test: `tests/unit/test_hard_negative_mining.py`, `tests/integration/test_distributed_supervision.py`,
  `tests/integration/test_stage3_training_step.py`, `tests/unit/test_naics_model.py`

**Interfaces:**
- Consumes: `FUSIONS` (Task 4).
- Produces:
  - `NAICSContrastiveModel(..., fusion: str = 'masked_mean', ...)`. The value is stored as
    `model.fusion` (a string) and in `hparams['fusion']`. A value outside `FUSIONS` raises
    `ValueError` ("unknown fusion") before the bundle loads.
  - `CurriculumMixin._router_mining_enabled() -> bool`: true only under `fusion == 'moe'` with
    the phase flag `enable_router_guided_sampling` on. Every host of the mixin sets `fusion`.
  - `LossMixin._combine_loss_terms(contrastive_loss, load_balancing_loss: Optional[Tensor], ...)
    -> Tuple[Tensor, Optional[Tensor]]`.
  - `LoggingMixin._log_loss_breakdown(contrastive_loss, scaled_load_balancing_loss:
    Optional[Tensor], ...)` logs `train/load_balancing_loss` only when that term is not None.

- [ ] **Step 1: Give every selection host a fusion**

In `tests/unit/test_hard_negative_mining.py`, replace:

```python
class _SelectionHost(DistributedMixin, CurriculumMixin):

    def __init__(self, index: SupervisionIndex, flags: dict):
        self.supervision_index = index
        self.current_curriculum_flags = flags
```

with:

```python
class _SelectionHost(DistributedMixin, CurriculumMixin):

    def __init__(self, index: SupervisionIndex, flags: dict, fusion: str = 'moe'):
        self.supervision_index = index
        self.current_curriculum_flags = flags
        self.fusion = fusion
```

The existing router tests pass gates and keep the `moe` default.

In `tests/integration/test_distributed_supervision.py`, `_DistributedSelectionHost.__init__`,
replace:

```python
        self.current_curriculum_flags = {'enable_hard_negative_mining': True}
        self.current_schedule_scalars = {}
```

with:

```python
        self.current_curriculum_flags = {'enable_hard_negative_mining': True}
        self.fusion = 'masked_mean'
        self.current_schedule_scalars = {}
```

- [ ] **Step 2: Write the failing tests**

In `tests/unit/test_hard_negative_mining.py`, add after `test_router_mix_ratio_splits_mined_slots`:

```python
def test_router_mining_runs_only_under_moe_fusion(hierarchy_index):
    # R10: without the MoE fusion there are no gates, and the geometric miner takes every slot
    host = _SelectionHost(
        hierarchy_index,
        {'enable_hard_negative_mining': True, 'enable_router_guided_sampling': True},
        fusion='masked_mean',
    )
    batch = _hierarchy_batch(
        hierarchy_index, [HIERARCHY_EXCLUSION] + CROSS_SECTOR_CODES, HIERARCHY_GRANDPARENT, 4
    )

    selected = host._select_negative_batch(
        batch=batch,
        anchor_output={'embedding': _code_embedding(13).unsqueeze(0)},
        candidate_output={'embedding': _candidate_output(batch)['embedding']},
        candidate_uid=_local_uid(batch),
        batch_idx=0,
    )

    assert _reasons(selected) == [SelectionReason.GEOMETRIC] * 4
```

In `tests/integration/test_stage3_training_step.py`, fixture `tiny_repaired_model`, replace:

```python
    model = model_module.NAICSContrastiveModel(
        base_model_name='test-stub',
        num_experts=2,
```

with:

```python
    model = model_module.NAICSContrastiveModel(
        base_model_name='test-stub',
        # MoE fusion: the selection carries the stub's gates, which the forced-order spy reads
        fusion='moe',
        num_experts=2,
```

Replace the fixture `hierarchy_model` (from `@pytest.fixture` above `def hierarchy_model(` to its
`return model`) with:

```python
@pytest.fixture
def make_hierarchy_model(monkeypatch, hierarchy_manifest):
    monkeypatch.setattr(model_module, 'MultiChannelEncoder', StubMultiChannelEncoder)

    def make(fusion: str):
        model = model_module.NAICSContrastiveModel(
            base_model_name='test-stub',
            fusion=fusion,
            num_experts=2,
            top_k=1,
            moe_hidden_dim=4,
            hierarchy_weight=0.0,
            radius_reg_weight=0.0,
            level_radius_weight=0.0,
            load_balancing_coef=0.0,
            supervision_manifest_path=str(hierarchy_manifest),
        )
        model.current_schedule_scalars = {'router_mix_ratio': 0.5}
        monkeypatch.setattr(model, '_update_curriculum_state', lambda *_args: None)
        monkeypatch.setattr(model, 'log', Mock())
        return model

    return make
```

Replace the parametrized test's head, from `@pytest.mark.parametrize(` through
`    hierarchy_model.current_curriculum_flags = flags`:

```python
@pytest.mark.parametrize(
    ('flags', 'expected'),
    [
        (
            {
                'enable_hard_negative_mining': True,
                'enable_router_guided_sampling': True
            },
            [SelectionReason.GEOMETRIC] * 2 + [SelectionReason.ROUTER] * 2,
        ),
        ({}, [SelectionReason.DIFFICULTY] * 4),
    ],
)
def test_real_coordinator_step_mines_when_enabled_and_respects_eligibility(
    hierarchy_model, monkeypatch, flags, expected
):
    hierarchy_model.current_curriculum_flags = flags
```

with:

```python
MINING = {'enable_hard_negative_mining': True, 'enable_router_guided_sampling': True}

@pytest.mark.parametrize(
    ('fusion', 'flags', 'expected'),
    [
        ('moe', MINING, [SelectionReason.GEOMETRIC] * 2 + [SelectionReason.ROUTER] * 2),
        # R10: without the MoE fusion the router is never consulted, so geometry takes every slot
        ('masked_mean', MINING, [SelectionReason.GEOMETRIC] * 4),
        ('masked_mean', {}, [SelectionReason.DIFFICULTY] * 4),
    ],
)
def test_real_coordinator_step_mines_when_enabled_and_respects_eligibility(
    make_hierarchy_model, monkeypatch, fusion, flags, expected
):
    hierarchy_model = make_hierarchy_model(fusion)
    hierarchy_model.current_curriculum_flags = flags
```

The rest of that test is unchanged.

In `tests/unit/test_naics_model.py`, class `TestModelInitialization`, add after
`test_rejects_unknown_supervision_mode`:

```python
    def test_fusion_defaults_to_masked_mean_and_an_unknown_one_is_refused(
        self, naics_model, model_config
    ):
        assert naics_model.fusion == 'masked_mean'
        assert naics_model.hparams['fusion'] == 'masked_mean'
        with pytest.raises(ValueError, match='unknown fusion'):
            NAICSContrastiveModel(**model_config, fusion='concatenate')
```

In class `TestTrainingStep`, replace `test_training_step_load_balancing_loss` (the whole method)
with:

```python
    @pytest.mark.parametrize(('fusion', 'logged'), [('masked_mean', False), ('moe', True)])
    def test_load_balancing_is_computed_and_logged_only_under_moe(
        self, model_config, repaired_training_batch, monkeypatch, fusion, logged
    ):
        '''R11: only the MoE fusion has experts, so only it has a load-balancing term.'''

        model = NAICSContrastiveModel(**model_config, fusion=fusion)
        log = Mock()
        monkeypatch.setattr(model, 'log', log)
        model.train()

        loss = model.training_step(repaired_training_batch, batch_idx=0)

        keys = {call.args[0] for call in log.call_args_list}
        assert ('train/load_balancing_loss' in keys) is logged
        assert any(key.startswith('train/moe/') for key in keys) is logged
        assert torch.isfinite(loss)
```

Add after `test_combine_loss_terms_scales_load_balancing`:

```python
    def test_combine_loss_terms_without_a_load_balancing_term(self, naics_model):
        '''Outside the MoE fusion there is no term to scale or add (R11).'''

        total_loss, scaled_load_balancing = naics_model._combine_loss_terms(
            torch.tensor(1.0),
            None,
            torch.tensor(0.3),
            torch.tensor(0.2),
            torch.tensor(0.1),
            torch.tensor(0.05),
        )

        assert scaled_load_balancing is None
        assert torch.isclose(total_loss, torch.tensor(1.65))
```

In class `TestCurriculumIntegration`, add after `test_curriculum_phase_transition`:

```python
    def test_phase_two_selection_under_masked_mean_fills_no_router_slot(
        self, naics_model, repaired_training_batch, monkeypatch
    ):
        '''Spec §6: a phase-2 step under the default fusion neither raises nor routes (R10).'''

        log = Mock()
        monkeypatch.setattr(naics_model, 'log', log)
        monkeypatch.setattr(naics_model, '_update_curriculum_state', lambda *_args: None)
        naics_model.current_curriculum_flags = {
            'enable_hard_negative_mining': True,
            'enable_router_guided_sampling': True,
        }
        naics_model.train()

        naics_model.training_step(repaired_training_batch, batch_idx=1)

        counters = {
            call.args[0]: call.args[1].item()
            for call in log.call_args_list if call.args[0].startswith('train/integrity/')
        }
        assert counters['train/integrity/router_selections'] == 0.0
        assert counters['train/integrity/geometric_selections'] > 0.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_hard_negative_mining.py tests/integration/test_stage3_training_step.py tests/unit/test_naics_model.py -q`
Expected failures:
- `test_router_mining_runs_only_under_moe_fusion`: `ValueError: router-guided selection requires
  anchor and candidate gate probabilities`;
- the integration tests and the new model tests: `TypeError` (unexpected keyword argument
  `'fusion'`);
- `test_phase_two_selection_under_masked_mean_fills_no_router_slot`: the router fills two slots,
  so `assert 2.0 == 0.0`.

- [ ] **Step 4: Gate router mining on the fusion**

In `src/naics_embedder/text_model/mixins/curriculum.py`, class docstring, replace:

```python
    - current_curriculum_flags: Dict[str, bool]
    - current_schedule_scalars: Dict[str, float]
```

with:

```python
    - current_curriculum_flags: Dict[str, bool]
    - current_schedule_scalars: Dict[str, float]
    - fusion: str (router mining runs only under ``moe``)
```

Replace:

```python
    def _select_negative_batch(
        self,
        *,
```

with:

```python
    def _router_mining_enabled(self) -> bool:
        '''
        Whether the router-guided miner takes part in this step's selection.

        Only the MoE fusion has experts and gates, so router mining runs only under ``moe`` with
        the phase flag on (R10). Under any other fusion the geometric miner takes every mining
        slot.
        '''

        return self.fusion == 'moe' and bool(
            self.current_curriculum_flags.get('enable_router_guided_sampling', False)
        )

    def _select_negative_batch(
        self,
        *,
```

Replace:

```python
        enable_router = self.current_curriculum_flags.get('enable_router_guided_sampling', False)
        if self._should_use_global_batch(enable_geometric, enable_router):
```

with:

```python
        enable_router = self._router_mining_enabled()
        if self._should_use_global_batch(enable_geometric, enable_router):
```

In `src/naics_embedder/text_model/mixins/logging.py`, `_log_selected_negative_stats`, replace:

```python
        enable_router = self.current_curriculum_flags.get('enable_router_guided_sampling', False)
        with torch.no_grad():
```

with:

```python
        enable_router = self._router_mining_enabled()
        with torch.no_grad():
```

In `_log_router_diversity`, replace:

```python
        if not gate_probs_list or not self.current_curriculum_flags.get(
            'enable_router_guided_sampling', False
        ):
            return
```

with:

```python
        if not gate_probs_list or not self._router_mining_enabled():
            return
```

- [ ] **Step 5: Make the load-balancing term optional**

In `src/naics_embedder/text_model/mixins/loss.py`, replace the whole `_combine_loss_terms` method
with:

```python
    def _combine_loss_terms(
        self,
        contrastive_loss: torch.Tensor,
        load_balancing_loss: Optional[torch.Tensor],
        hierarchy_loss: torch.Tensor,
        structural_preference_loss: torch.Tensor,
        radius_reg_loss: torch.Tensor,
        level_radius_loss_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Combine individual loss components into the final optimization target.

        Args:
            contrastive_loss: Main contrastive loss
            load_balancing_loss: MoE load balancing loss, or None outside the MoE fusion (R11)
            hierarchy_loss: Hierarchy preservation loss
            structural_preference_loss: Structural preference loss over selected candidates
            radius_reg_loss: Radius regularization loss
            level_radius_loss_value: Level-aware radius alignment loss

        Returns:
            Tuple containing the total loss and the scaled load balancing term (None when there
            is no term).
        '''
        scaled_load_balancing_loss = None
        total_loss = contrastive_loss
        if load_balancing_loss is not None:
            scaled_load_balancing_loss = self.load_balancing_coef * load_balancing_loss
            total_loss = total_loss + scaled_load_balancing_loss
        total_loss = (
            total_loss + hierarchy_loss + structural_preference_loss + radius_reg_loss
            + level_radius_loss_value
        )
        return total_loss, scaled_load_balancing_loss
```

In `src/naics_embedder/text_model/mixins/logging.py`, `_log_loss_breakdown`, replace:

```python
        contrastive_loss: torch.Tensor,
        scaled_load_balancing_loss: torch.Tensor,
```

with:

```python
        contrastive_loss: torch.Tensor,
        scaled_load_balancing_loss: Optional[torch.Tensor],
```

Then replace:

```python
        self.log(
            'train/load_balancing_loss',
            scaled_load_balancing_loss,
            prog_bar=True,
            batch_size=batch_size,
        )
```

with:

```python
        # Only the MoE fusion has a load-balancing term (R11)
        if scaled_load_balancing_loss is not None:
            self.log(
                'train/load_balancing_loss',
                scaled_load_balancing_loss,
                prog_bar=True,
                batch_size=batch_size,
            )
```

- [ ] **Step 6: Add the switch to the model**

In `src/naics_embedder/text_model/naics_model.py`, replace:

```python
from naics_embedder.text_model.encoder import MultiChannelEncoder
from naics_embedder.text_model.hard_negative_mining import (
```

with:

```python
from naics_embedder.text_model.encoder import MultiChannelEncoder
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.hard_negative_mining import (
```

In the class docstring's `Args:`, replace:

```python
        lora_dropout: LoRA dropout rate
```

with:

```python
        lora_dropout: LoRA dropout rate
        fusion: Channel fusion: ``masked_mean`` (default), ``attention`` or ``moe``. Router
            mining and the load-balancing term run only under ``moe`` (R10, R11)
```

Replace:

```python
        load_balancing_coef: MoE load balancing coefficient
```

with:

```python
        load_balancing_coef: MoE load balancing coefficient (the term exists only under ``moe``)
```

In the constructor's signature, replace:

```python
        lora_dropout: float = 0.1,
        num_experts: int = 4,
```

with:

```python
        lora_dropout: float = 0.1,
        fusion: str = 'masked_mean',
        num_experts: int = 4,
```

Replace:

```python
        super().__init__()

        self.supervision_policy = SupervisionModePolicy.from_name(supervision_mode)
```

with:

```python
        super().__init__()

        if fusion not in FUSIONS:
            raise ValueError(f'unknown fusion {fusion!r}; expected one of {list(FUSIONS)}')
        # The one switch for the MoE-only machinery: router mining and load balancing (R10, R11)
        self.fusion = fusion

        self.supervision_policy = SupervisionModePolicy.from_name(supervision_mode)
```

In `training_step`, replace:

```python
        # MoE load balancing over anchors, positives, and valid candidate rows only
        valid_candidates = batch['candidate_valid_mask'].reshape(-1)
        valid_candidate_output = {
            name: candidate_output[name][valid_candidates]
            for name in ('gate_probs', 'top_k_indices') if name in candidate_output
        }
        gate_probs_list, topk_indices_list = self._collect_gate_outputs(
            [anchor_output, positive_output, valid_candidate_output]
        )
        self._log_router_diversity(gate_probs_list, batch_size)
        raw_load_balancing_loss = self._compute_load_balancing_loss(
            gate_probs_list,
            topk_indices_list,
            batch_size,
        )
```

with:

```python
        # MoE load balancing over anchors, positives, and valid candidate rows only. Only the MoE
        # fusion has experts, so only it computes, adds and logs the term (R11).
        raw_load_balancing_loss = None
        if self.fusion == 'moe':
            valid_candidates = batch['candidate_valid_mask'].reshape(-1)
            valid_candidate_output = {
                name: candidate_output[name][valid_candidates]
                for name in ('gate_probs', 'top_k_indices') if name in candidate_output
            }
            gate_probs_list, topk_indices_list = self._collect_gate_outputs(
                [anchor_output, positive_output, valid_candidate_output]
            )
            self._log_router_diversity(gate_probs_list, batch_size)
            raw_load_balancing_loss = self._compute_load_balancing_loss(
                gate_probs_list,
                topk_indices_list,
                batch_size,
            )
```

In `_legacy_containment_training_step`, replace:

```python
        gate_probs, topk_indices = self._collect_gate_outputs(
            [anchor_output, positive_output, negative_output]
        )
        load_balancing = self._compute_load_balancing_loss(
            gate_probs,
            topk_indices,
            batch_size,
        )
```

with:

```python
        load_balancing = None
        if self.fusion == 'moe':
            gate_probs, topk_indices = self._collect_gate_outputs(
                [anchor_output, positive_output, negative_output]
            )
            load_balancing = self._compute_load_balancing_loss(
                gate_probs,
                topk_indices,
                batch_size,
            )
```

The calls to `_combine_loss_terms` and `_log_loss_breakdown` in both steps stay as they are; both
now take `None`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_hard_negative_mining.py tests/integration/test_stage3_training_step.py tests/integration/test_distributed_supervision.py tests/unit/test_naics_model.py tests/unit/test_text_validation_metrics.py -q`
Expected: all pass.

- [ ] **Step 8: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/naics_model.py src/naics_embedder/text_model/mixins/curriculum.py src/naics_embedder/text_model/mixins/logging.py src/naics_embedder/text_model/mixins/loss.py tests/unit/test_hard_negative_mining.py tests/integration/test_distributed_supervision.py tests/integration/test_stage3_training_step.py tests/unit/test_naics_model.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/text_model/naics_model.py src/naics_embedder/text_model/mixins/curriculum.py src/naics_embedder/text_model/mixins/logging.py src/naics_embedder/text_model/mixins/loss.py tests/unit/test_hard_negative_mining.py tests/integration/test_distributed_supervision.py tests/integration/test_stage3_training_step.py tests/unit/test_naics_model.py
git commit -m "feat(text_model): router mining and load balancing only under MoE fusion (R10, R11)"
```

### Task 7: Train through the shared encoder

The config gains `model.fusion` and `model.dimension` (4.2). The model builds `SharedEncoder`
where it built `MultiChannelEncoder` and keeps accepting every pre-Stage-6 hyperparameter (4.1).
The four-copy encoder, `HyperbolicProjection` and the `embedding_euc` output are deleted (§8).

The HGCN feeder (`generate_embeddings_from_checkpoint`) still stacks channels by hand, without
`present`, until Task 9 moves it onto the shared builder. The shared encoder refuses those
batches, so the feeder cannot run between this task and Task 9. No test calls it, and `train`
reaches it only when its closing prompt is answered yes.

**Files:**
- Modify: `src/naics_embedder/utils/config.py:896-907`, `conf/config.yaml:116-124`
- Modify: `src/naics_embedder/text_model/naics_model.py:6,46-65,98,120-122,172-174,297-314,422-435`
  (line numbers at 52075f9; Task 6 shifts them, and the Replace blocks are exact)
- Modify: `src/naics_embedder/cli/commands/training.py:98-100,536-539`
- Modify: `src/naics_embedder/utils/training.py:466-469`
- Modify: `src/naics_embedder/panels/text_only.py:8-12`
- Modify: `src/naics_embedder/text_model/hyperbolic.py` (delete `HyperbolicProjection`)
- Delete: `src/naics_embedder/text_model/encoder.py`
- Modify: `docs/api/encoder.md`
- Test: `tests/unit/test_config.py`, `tests/unit/test_cli_training.py`,
  `tests/unit/test_utils_training.py`, `tests/unit/test_naics_model.py`,
  `tests/integration/test_stage3_training_step.py`, `tests/unit/test_hyperbolic.py`,
  `tests/conftest.py:86-93`

**Interfaces:**
- Consumes: `SharedEncoder` and `DIMENSIONS` (Task 5); the model's `fusion` (Task 6).
- Produces:
  - `ModelConfig.fusion: Literal['masked_mean', 'attention', 'moe'] = 'masked_mean'` and
    `ModelConfig.dimension: Literal[8, 16, 32] = 16`.
  - `NAICSContrastiveModel(..., fusion='masked_mean', dimension=16, ...)`. A dimension outside
    `DIMENSIONS` raises `ValueError` ("unknown dimension") before the bundle loads.
  - `model.encoder` is a `SharedEncoder`. `model(batch)` returns `embedding` (B, d + 1) and
    `tangent` (B, d), plus `gate_probs` and `top_k_indices` under `moe`.
  - `build_model_from_config` passes `fusion` and `dimension`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_config.py`, add after `test_base_config_parses_as_repaired_pre_generation`:

```python
def test_the_model_fuses_by_masked_mean_at_dimension_16(valid_config_dict):
    cfg = Config.model_validate(valid_config_dict)

    assert (cfg.model.fusion, cfg.model.dimension) == ('masked_mean', 16)

@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('model.fusion', 'attention'),
        ('model.fusion', 'moe'),
        ('model.dimension', 8),
        ('model.dimension', 32),
    ],
)
def test_every_fusion_and_dimension_in_its_set_is_accepted(key, value):
    cfg = Config().override({key: value})

    assert getattr(cfg.model, key.split('.')[1]) == value

@pytest.mark.parametrize(('key', 'value'), [('model.fusion', 'concat'), ('model.dimension', 12)])
def test_a_fusion_or_dimension_outside_its_set_is_refused(key, value):
    with pytest.raises(ValidationError) as excinfo:
        Config().override({key: value})

    # A Literal refusal: before the keys are declared, the same override fails as extra_forbidden
    assert _error_locs_and_types(excinfo) == [(('model', key.split('.')[1]), 'literal_error')]
```

`_error_locs_and_types` is the module's existing helper. Pinning the error type keeps this test
red until Step 3 declares the keys.

In `tests/unit/test_cli_training.py`, add after
`test_the_train_banner_headlines_no_structural_statistic`:

```python
@pytest.mark.unit
def test_the_train_banner_names_the_fusion_and_dimension(cli_runner, training_env):
    result = cli_runner.invoke(cli_app, ['train', 'model.dimension=8'], catch_exceptions=False)

    assert result.exit_code == 0
    output = result.output.replace('\n', '')
    assert 'Fusion: masked_mean' in output
    assert 'Dimension: 8' in output
```

In `test_repaired_model_and_datamodule_receive_bundle_supervision`, replace:

```python
    assert model_kwargs['structural_preference_weight'] == 0.35
```

with:

```python
    assert model_kwargs['structural_preference_weight'] == 0.35
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('masked_mean', 16)
```

In `tests/unit/test_utils_training.py`, replace:

```python
from pathlib import Path
from types import SimpleNamespace
```

with:

```python
import json
from pathlib import Path
from types import SimpleNamespace
```

At the end of `test_save_training_summary_writes_files`, add:

```python
    snapshot = json.loads(Path(paths['json']).read_text())['config_snapshot']['model']
    assert (snapshot['fusion'], snapshot['dimension']) == ('masked_mean', 16)
```

In `tests/unit/test_naics_model.py`, replace:

```python
import polars as pl
import pytest
import pytorch_lightning as pyl
import torch

from naics_embedder.text_model.dataloader.datamodule import collate_fn
from naics_embedder.text_model.naics_model import (
    NAICSContrastiveModel,
    gather_embeddings_global,
)
```

with:

```python
import polars as pl
import pytest
import pytorch_lightning as pyl
import torch
from transformers import PreTrainedModel

from naics_embedder.text_model.dataloader.datamodule import collate_fn
from naics_embedder.text_model.naics_model import (
    NAICSContrastiveModel,
    gather_embeddings_global,
)
from naics_embedder.text_model.shared_encoder import SharedEncoder
```

Replace `test_encoder_configuration` (the whole method) with:

```python
    def test_encoder_configuration(self, naics_model, model_config):
        '''One MiniLM backbone, masked-mean fusion and one Linear(384 -> 16) to the head.'''

        encoder = naics_model.encoder
        assert isinstance(encoder, SharedEncoder)
        assert encoder.curvature == model_config['curvature']
        assert sum(isinstance(module, PreTrainedModel) for module in naics_model.modules()) == 1
        assert encoder.fusion_name == 'masked_mean'
        assert (encoder.projection.in_features, encoder.projection.out_features) == (384, 16)
        assert list(encoder.head.parameters()) == []

    def test_an_unknown_dimension_is_refused(self, model_config):
        with pytest.raises(ValueError, match='unknown dimension'):
            NAICSContrastiveModel(**model_config, dimension=12)
```

Replace the two methods of `TestForwardPass`, `test_forward_basic` and
`test_forward_output_shapes`, with:

```python
    def test_forward_basic(self, naics_model, sample_training_batch):
        '''The default fusion returns the point and its tangent, and no gates.'''

        with torch.no_grad():
            output = naics_model(sample_training_batch['anchor'])

        assert set(output) == {'embedding', 'tangent'}

    def test_forward_output_shapes(self, naics_model, sample_training_batch):
        '''The Lorentz point is (batch, 17) and its capped tangent (batch, 16).'''

        batch_size = sample_training_batch['batch_size']

        with torch.no_grad():
            output = naics_model(sample_training_batch['anchor'])

        assert naics_model.encoder.dimension == 16
        assert output['embedding'].shape == (batch_size, 17)
        assert output['tangent'].shape == (batch_size, 16)
```

In `TestTrainingStep`, add after `test_training_step_gradient_flow`:

```python
    def test_a_step_at_dimension_16_trains_the_adapter_and_the_projection(
        self, naics_model, repaired_training_batch, monkeypatch
    ):
        '''Spec §6: the step reaches LoRA and the projection, and logs no load-balancing term.'''

        log = Mock()
        monkeypatch.setattr(naics_model, 'log', log)
        naics_model.train()

        naics_model.training_step(repaired_training_batch, batch_idx=0).backward()

        encoder = naics_model.encoder
        assert encoder.dimension == 16
        assert encoder.projection.weight.grad.abs().sum() > 0
        # PEFT starts lora_B at zero, so lora_A's first gradient is exactly zero (P9); the
        # pooler's adapter never gets one, since mean pooling never reads it (P8)
        adapters = {
            name: parameter
            for name, parameter in encoder.backbone.named_parameters()
            if 'lora_B' in name and '.pooler.' not in name
        }
        assert adapters
        for name, parameter in adapters.items():
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
        keys = {call.args[0] for call in log.call_args_list}
        assert 'train/load_balancing_loss' not in keys
        assert not any(key.startswith('train/moe/') for key in keys)
```

In `test_validation_step_embedding_storage`, replace:

```python
            assert embedding.shape[0] == naics_model.encoder.embedding_dim + 1  # Lorentz
```

with:

```python
            assert embedding.shape[0] == naics_model.encoder.dimension + 1  # Lorentz
```

In `tests/integration/test_stage3_training_step.py`, replace the class `StubMultiChannelEncoder`
(the whole class) with:

```python
class StubSharedEncoder(nn.Module):
    dimension = 2

    def __init__(self, *, fusion: str, **_kwargs):
        super().__init__()
        self.emits_gates = fusion == 'moe'
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, channel_inputs):
        raw = channel_inputs['title']['input_ids'][:, 0].to(torch.float32)
        value = raw * self.scale
        tangent = torch.stack([value, value / 2.0], dim=1)
        time = torch.sqrt(1.0 + tangent.square().sum(dim=1, keepdim=True))
        output = {'embedding': torch.cat([time, tangent], dim=1), 'tangent': tangent}
        if self.emits_gates:
            # Like the shared encoder, only the MoE fusion emits gates
            first_gate = ((raw - 1.0) / 10.0).clamp(0.0, 1.0)
            gate_probs = torch.stack([first_gate, 1.0 - first_gate], dim=1)
            output['gate_probs'] = gate_probs
            output['top_k_indices'] = gate_probs.argmax(dim=1, keepdim=True)
        return output
```

Replace both lines that read:

```python
    monkeypatch.setattr(model_module, 'MultiChannelEncoder', StubMultiChannelEncoder)
```

(in `tiny_repaired_model` and `make_hierarchy_model`) with:

```python
    monkeypatch.setattr(model_module, 'SharedEncoder', StubSharedEncoder)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_cli_training.py tests/unit/test_utils_training.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py -q`
Expected failures:
- the config tests: the YAML and `ModelConfig` have no `fusion`;
- the CLI and summary tests: `KeyError: 'fusion'`;
- the model tests: the encoder is a `MultiChannelEncoder`;
- the integration tests: `AttributeError`, since `naics_model` has no `SharedEncoder`.

- [ ] **Step 3: Declare the two config keys**

In `src/naics_embedder/utils/config.py`, class `ModelConfig`, replace:

```python
    base_model_name: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2', description='HuggingFace base model name'
    )
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description='LoRA configuration')
    moe: MoEConfig = Field(
        default_factory=MoEConfig, description='Mixture of Experts configuration'
    )
```

with:

```python
    base_model_name: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2', description='HuggingFace base model name'
    )
    fusion: Literal['masked_mean', 'attention', 'moe'] = Field(
        default='masked_mean',
        description='Channel fusion: masked_mean (the default), attention, or moe (an ablation)',
    )
    dimension: Literal[8, 16, 32] = Field(
        default=16, description='Embedding dimension: the one Linear(hidden -> d) before the head'
    )
    lora: LoRAConfig = Field(default_factory=LoRAConfig, description='LoRA configuration')
    moe: MoEConfig = Field(
        default_factory=MoEConfig,
        description='Mixture of Experts configuration, read only under fusion moe',
    )
```

In `conf/config.yaml` (this worktree's; never copy the main checkout's), replace:

```yaml
model:
  base_model_name: sentence-transformers/all-MiniLM-L6-v2

  lora:
```

with:

```yaml
model:
  base_model_name: sentence-transformers/all-MiniLM-L6-v2
  fusion: masked_mean  # masked_mean, attention, or moe (an ablation only)
  dimension: 16  # 8, 16 or 32: the one Linear(384 -> d) before the geometry head

  lora:
```

Then replace:

```yaml
  moe:
    num_experts: 4
```

with:

```yaml
  moe:  # read only under fusion: moe
    num_experts: 4
```

- [ ] **Step 4: Switch the model to the shared encoder**

In `src/naics_embedder/text_model/naics_model.py`, module docstring, replace:

```python
- MultiChannelEncoder with LoRA fine-tuning and MoE
```

with:

```python
- SharedEncoder: one LoRA-tuned backbone, masked fusion and one affine map to dimension d
```

Replace:

```python
from naics_embedder.text_model.encoder import MultiChannelEncoder
from naics_embedder.text_model.fusion import FUSIONS
```

with:

```python
from naics_embedder.text_model.fusion import FUSIONS
```

Replace:

```python
    ValidationMixin,
    gather_embeddings_global,
)
```

with:

```python
    ValidationMixin,
    gather_embeddings_global,
)
from naics_embedder.text_model.shared_encoder import DIMENSIONS, SharedEncoder
```

In the class docstring, replace:

```python
    - MultiChannelEncoder: LoRA-tuned transformer with Mixture of Experts
```

with:

```python
    - SharedEncoder: one LoRA-tuned backbone over field-marked channels, masked fusion, and one
      affine map to the embedding dimension
```

In its `Args:`, replace:

```python
        num_experts: Number of MoE experts
        top_k: Number of experts to select per token
        moe_hidden_dim: Hidden dimension of MoE layers
```

with:

```python
        dimension: Embedding dimension, one of 8, 16 or 32: the width of the one
            ``Linear(hidden → d)`` before the geometry head
        num_experts: Number of MoE experts (``moe`` only)
        top_k: Number of experts to select per code (``moe`` only)
        moe_hidden_dim: Hidden dimension of MoE layers (``moe`` only)
```

In the signature, replace:

```python
        fusion: str = 'masked_mean',
        num_experts: int = 4,
```

with:

```python
        fusion: str = 'masked_mean',
        dimension: int = 16,
        num_experts: int = 4,
```

Replace:

```python
        if fusion not in FUSIONS:
            raise ValueError(f'unknown fusion {fusion!r}; expected one of {list(FUSIONS)}')
```

with:

```python
        if fusion not in FUSIONS:
            raise ValueError(f'unknown fusion {fusion!r}; expected one of {list(FUSIONS)}')
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')
```

Replace:

```python
        # Initialize encoder
        self.encoder = MultiChannelEncoder(
            base_model_name=base_model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
            curvature=curvature,
        )

        # Initialize loss function
        self.loss_fn = HyperbolicInfoNCELoss(
            embedding_dim=self.encoder.embedding_dim,
```

with:

```python
        # Initialize the shared encoder: one backbone, fusion, one affine map, the head
        self.encoder = SharedEncoder(
            base_model_name=base_model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fusion=fusion,
            dimension=dimension,
            num_experts=num_experts,
            top_k=top_k,
            moe_hidden_dim=moe_hidden_dim,
            curvature=curvature,
        )

        # Initialize loss function
        self.loss_fn = HyperbolicInfoNCELoss(
            embedding_dim=self.encoder.dimension,
```

In `forward`, replace the docstring:

```python
        '''
        Forward pass through the encoder.

        Args:
            channel_inputs: Dictionary of channel inputs with tokenized text

        Returns:
            Dictionary containing:
            - embedding: Hyperbolic embeddings (batch_size, embed_dim + 1)
            - gate_probs: MoE gate probabilities (batch_size, num_experts)
            - top_k_indices: Selected expert indices (batch_size, top_k)
        '''
```

with:

```python
        '''
        Forward pass through the shared encoder.

        Args:
            channel_inputs: Per field, tokenized text and a boolean ``present``, as
                ``stack_text_inputs`` builds them: a code batch's four channels, or ``query``

        Returns:
            Dictionary containing:
            - embedding: Lorentz points (batch_size, dimension + 1)
            - tangent: Capped tangent vectors at the origin (batch_size, dimension)
            - gate_probs, top_k_indices: The experts' gates, under ``moe`` fusion only
        '''
```

- [ ] **Step 5: Pass the keys through the CLI**

In `src/naics_embedder/cli/commands/training.py`, `build_model_from_config`, replace:

```python
        lora_dropout=cfg.model.lora.dropout,
        num_experts=cfg.model.moe.num_experts,
```

with:

```python
        lora_dropout=cfg.model.lora.dropout,
        fusion=cfg.model.fusion,
        dimension=cfg.model.dimension,
        num_experts=cfg.model.moe.num_experts,
```

In `train`, replace:

```python
            f'  • LoRA rank: {cfg.model.lora.r}',
            '  • MoE: ',
            f'    - {cfg.model.moe.num_experts} experts\n',
```

with:

```python
            f'  • LoRA rank: {cfg.model.lora.r}',
            f'  • Fusion: {cfg.model.fusion}',
            f'  • Dimension: {cfg.model.dimension}\n',
```

In `src/naics_embedder/utils/training.py`, `save_training_summary`, replace:

```python
                'base_model': config.model.base_model_name,
                'lora_rank': config.model.lora.r,
```

with:

```python
                'base_model': config.model.base_model_name,
                'fusion': config.model.fusion,
                'dimension': config.model.dimension,
                'lora_rank': config.model.lora.r,
```

- [ ] **Step 6: Delete the four-copy encoder and the projection**

Run: `git rm src/naics_embedder/text_model/encoder.py`

In `src/naics_embedder/text_model/hyperbolic.py`, delete the section from the divider
`# Hyperbolic Projection to Lorentz Model` through the end of `class HyperbolicProjection`
(its `return hyperbolic_embedding`). The `# Lorentz Distance Computation` divider then follows
`HyperbolicHead`.

In `tests/unit/test_hyperbolic.py`, remove the line `    HyperbolicProjection,` from the
import. Then delete the section from the divider `# HyperbolicProjection Tests` through the end
of `test_projection_no_nans_or_infs`, so the `# LorentzDistance Tests` divider follows
`TestHyperbolicHead`.

In `tests/conftest.py`, delete the fixture that only those tests used:

```python
@pytest.fixture
def sample_euclidean_embeddings(test_device, random_seed):
    '''Generate sample Euclidean embeddings.'''

    torch.manual_seed(random_seed)
    batch_size = 16
    dim = 384
    return torch.randn(batch_size, dim, device=test_device)

```

In `src/naics_embedder/panels/text_only.py`, module docstring, replace:

```python
- **Text.** The four channels the arm reads (``title``, ``description``, ``excluded``,
  ``examples``) from the arm's own descriptions file.
- **Pooling.** Each channel is mean-pooled over its tokens under the attention mask, as the arm's
  encoder pools (``text_model/encoder.py``). A code's vector is the mean over its present
```

with:

```python
- **Text.** The four channels the arm reads (``title``, ``description``, ``excluded``,
  ``examples``) from the arm's own descriptions file, without the arm's field markers (spec 4.2).
- **Pooling.** Each channel is mean-pooled over its tokens under the attention mask, as the arm's
  encoder pools (``text_model/shared_encoder.py``). A code's vector is the mean over its present
```

Replace the whole of `docs/api/encoder.md` with:

```markdown
# Shared Encoder API

::: naics_embedder.text_model.shared_encoder
```

That keeps the strict docs build importable; Task 13 writes the docs.

- [ ] **Step 7: Run the tests and the checks**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_cli_training.py tests/unit/test_utils_training.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py tests/unit/test_hyperbolic.py tests/unit/test_encoder.py tests/unit/test_text_only.py -q`
Expected: all pass.

Run: `git grep -n -P 'MultiChannelEncoder|HyperbolicProjection|text_model\.encoder\b|text_model/encoder\.py|embedding_euc' -- src tests conf docs/api ':!tests/README.md'`
Expected: no output.

Run: `uv run mkdocs build --strict`
Expected: the build completes with no warnings.

- [ ] **Step 8: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/utils/config.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/cli/commands/training.py src/naics_embedder/utils/training.py src/naics_embedder/panels/text_only.py src/naics_embedder/text_model/hyperbolic.py tests/unit/test_config.py tests/unit/test_cli_training.py tests/unit/test_utils_training.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py tests/unit/test_hyperbolic.py tests/conftest.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/utils/config.py conf/config.yaml src/naics_embedder/text_model/naics_model.py src/naics_embedder/cli/commands/training.py src/naics_embedder/utils/training.py src/naics_embedder/panels/text_only.py src/naics_embedder/text_model/hyperbolic.py docs/api/encoder.md tests/unit/test_config.py tests/unit/test_cli_training.py tests/unit/test_utils_training.py tests/unit/test_naics_model.py tests/integration/test_stage3_training_step.py tests/unit/test_hyperbolic.py tests/conftest.py
git commit -m "feat(text_model): train through the shared encoder; delete the four-copy encoder"
```

`git rm` in Step 6 already staged `encoder.py`'s removal; naming it in `git add` would fail.

### Task 8: The encoder record in the checkpoint contract (D2)

`CheckpointContract` gains `encoder: EncoderArchitecture` (4.4). An absent record reads as the
legacy four-copy layout, so every checkpoint saved before Stage 6 validates as four-copy, never as
the runtime's architecture. Exact resume and weights-only refuse another architecture with one D2
message, and nothing migrates. The model and the config build their records with one helper, so
the two cannot drift.

Export and reads (Tasks 10 and 11) take the record from the checkpoint. This task adds
`validate_supervision_contract` for them, which compares every field but `encoder` (P16).

**Files:**
- Rewrite: `src/naics_embedder/supervision/checkpoints.py`
- Modify: `src/naics_embedder/text_model/naics_model.py:34-40,232-233,263,272,437-448`
  (line numbers at 52075f9; Tasks 6 and 7 shift them, and the Replace blocks are exact)
- Modify: `src/naics_embedder/cli/commands/training.py:28-35,149-160,410-413,446-448,645`
- Rewrite: `tests/unit/test_checkpoint_contract.py`
- Test: `tests/unit/test_naics_model.py`, `tests/unit/test_cli_training.py`

**Interfaces:**
- Consumes: the model's `fusion`, `dimension` and `base_model_name` (Tasks 6 and 7); the config's
  `model.fusion` and `model.dimension` (Task 7).
- Produces, in `naics_embedder.supervision.checkpoints`:
  - `EncoderArchitecture(layout: Literal['shared', 'four-copy'], fusion: Optional[str] = None,
    dimension: Optional[int] = None, backbone: Optional[str] = None)`. It is frozen and forbids
    extras. A shared record names all three fields, and a four-copy record names none.
  - `LEGACY_ENCODER = EncoderArchitecture(layout='four-copy')`.
  - `shared_encoder_architecture(*, fusion: str, dimension: int, backbone: str) ->
    EncoderArchitecture`.
  - `D2_REFUSAL: str`, which contains "four-copy" and "roadmap D2".
  - `CheckpointContract.encoder: EncoderArchitecture = LEGACY_ENCODER`.
  - The builders take the record as a required keyword:
    - `contract_for_bundle(manifest, supervision_mode='repaired', *, encoder) ->
      CheckpointContract`;
    - `containment_contract(*, encoder) -> CheckpointContract`.
  - `saved_encoder(raw: Optional[dict]) -> EncoderArchitecture`.
  - `validate_supervision_contract(raw: Optional[dict], manifest, supervision_mode='repaired') ->
    CheckpointContract` returns the saved contract. It raises `ValueError` matching
    "supervision contract mismatch", or "no Stage-3 contract".
  - `load_weights_only(model, path, *, encoder: EncoderArchitecture) -> MigrationReport` raises
    "weights-only encoder mismatch … D2" before reading any parameter.
- In `naics_embedder.cli.commands.training`: `encoder_architecture_for(cfg: Config) ->
  EncoderArchitecture`.
- `model.checkpoint_contract.encoder` is the model's own shared record.

- [ ] **Step 1: Write the failing tests**

Replace the whole of `tests/unit/test_checkpoint_contract.py` with:

```python
import pytest
import torch
from torch import nn

from naics_embedder.supervision.checkpoints import (
    D2_REFUSAL,
    LEGACY_ENCODER,
    CheckpointContract,
    EncoderArchitecture,
    contract_for_bundle,
    load_weights_only,
    saved_encoder,
    shared_encoder_architecture,
    validate_checkpoint_contract,
    validate_exact_resume,
    validate_supervision_contract,
)

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
SHARED = shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone=MINILM)

@pytest.fixture
def runtime_contract() -> CheckpointContract:
    return CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
        encoder=SHARED,
    )

@pytest.fixture
def tiny_repaired_model() -> nn.Module:
    model = nn.Module()
    model.encoder = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    model.current_curriculum_flags = {}
    return model

def _save(path, contract=None, **payload):
    '''Save a checkpoint holding ``payload``, under ``contract`` when one is given.'''

    if contract is not None:
        payload['stage3_supervision'] = contract.model_dump()
    torch.save(payload, path)
    return path

# -------------------------------------------------------------------------------------------------
# Exact resume
# -------------------------------------------------------------------------------------------------

def test_matching_new_checkpoint_can_exact_resume(tmp_path, runtime_contract):
    validate_exact_resume(_save(tmp_path / 'new.ckpt', runtime_contract), runtime_contract)

@pytest.mark.parametrize(
    'checkpoint_metadata',
    [
        None,
        {'contract_version': 'legacy'},
        {'bundle_id': 'other-bundle'},
        {'codebook_fingerprint': 'f' * 64},
        {'structural_preference_loss_version': 'other-loss'},
        {'mining_contract_version': 'other-mining'},
    ],
)
def test_legacy_or_mismatched_checkpoint_cannot_exact_resume(
    tmp_path, runtime_contract, checkpoint_metadata
):
    path = tmp_path / 'checkpoint.ckpt'
    payload = {}
    if checkpoint_metadata is not None:
        payload['stage3_supervision'] = {
            **runtime_contract.model_dump(),
            **checkpoint_metadata,
        }
    torch.save(payload, path)

    with pytest.raises(ValueError, match='exact resume'):
        validate_exact_resume(path, runtime_contract)

def test_in_memory_contract_check_names_every_mismatched_field(runtime_contract):
    saved = {**runtime_contract.model_dump(), 'bundle_id': 'bundle-b', 'supervision_mode': 'x'}

    with pytest.raises(ValueError, match='bundle_id') as excinfo:
        validate_checkpoint_contract(saved, runtime_contract)

    assert 'supervision_mode' in str(excinfo.value)
    # A supervision mismatch is not the architecture refusal
    assert D2_REFUSAL not in str(excinfo.value)

def test_a_checkpoint_trained_under_the_exclusion_quota_cannot_exact_resume(
    tmp_path, runtime_contract
):
    # negative-selection-v1 reserved a slot for an explicit exclusion; v2 never selects one
    path = tmp_path / 'quota.ckpt'
    saved = {**runtime_contract.model_dump(), 'mining_contract_version': 'negative-selection-v1'}
    torch.save({'stage3_supervision': saved}, path)

    assert runtime_contract.mining_contract_version == 'negative-selection-v2'
    with pytest.raises(ValueError, match='exact resume'):
        validate_exact_resume(path, runtime_contract)

def test_contract_for_bundle_reads_manifest_identity(validated_bundle):
    contract = contract_for_bundle(validated_bundle.manifest, encoder=SHARED)

    assert contract.supervision_mode == 'repaired'
    assert contract.bundle_id == 'bundle-a'
    assert contract.contract_version == 'stage3-supervision-v2'
    assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
    assert contract.encoder == SHARED

# -------------------------------------------------------------------------------------------------
# The encoder record (spec 4.4)
# -------------------------------------------------------------------------------------------------

def test_an_absent_encoder_record_reads_as_the_legacy_four_copy_layout(runtime_contract):
    saved = runtime_contract.model_dump()
    del saved['encoder']

    assert CheckpointContract.model_validate(saved).encoder == LEGACY_ENCODER
    assert LEGACY_ENCODER == EncoderArchitecture(layout='four-copy')
    assert saved_encoder(saved) == LEGACY_ENCODER
    assert saved_encoder(None) == LEGACY_ENCODER

@pytest.mark.parametrize(
    'record',
    [
        {'layout': 'shared', 'fusion': 'masked_mean', 'dimension': 16},
        {'layout': 'four-copy', 'dimension': 16},
        {'layout': 'concatenated'},
        {'layout': 'shared', 'fusion': 'masked_mean', 'dimension': 16, 'backbone': MINILM, 'x': 1},
    ],
)
def test_a_malformed_encoder_record_is_refused(record):
    with pytest.raises(ValueError):
        EncoderArchitecture(**record)

def test_the_encoder_record_survives_a_save_round_trip(tmp_path, runtime_contract):
    path = _save(tmp_path / 'shared.ckpt', runtime_contract)

    saved = torch.load(path, weights_only=False)['stage3_supervision']

    assert saved['encoder'] == {
        'layout': 'shared',
        'fusion': 'masked_mean',
        'dimension': 16,
        'backbone': MINILM,
    }
    assert CheckpointContract.model_validate(saved) == runtime_contract

@pytest.mark.parametrize(
    'encoder',
    [
        LEGACY_ENCODER,
        shared_encoder_architecture(fusion='masked_mean', dimension=8, backbone=MINILM),
        shared_encoder_architecture(fusion='moe', dimension=16, backbone=MINILM),
        shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone='other/model'),
    ],
)
def test_another_encoder_architecture_cannot_exact_resume(tmp_path, runtime_contract, encoder):
    path = _save(tmp_path / 'other.ckpt', runtime_contract.model_copy(update={'encoder': encoder}))

    with pytest.raises(ValueError, match='exact resume') as excinfo:
        validate_exact_resume(path, runtime_contract)

    assert 'encoder' in str(excinfo.value)
    assert D2_REFUSAL in str(excinfo.value)

def test_a_contract_saved_before_stage_6_meets_the_d2_refusal(tmp_path, runtime_contract):
    saved = runtime_contract.model_dump()
    del saved['encoder']
    path = tmp_path / 'four-copy.ckpt'
    torch.save({'stage3_supervision': saved}, path)

    with pytest.raises(ValueError, match='four-copy') as excinfo:
        validate_exact_resume(path, runtime_contract)

    assert D2_REFUSAL in str(excinfo.value)

def test_a_checkpoint_without_a_contract_cites_d2(tmp_path, runtime_contract):
    with pytest.raises(ValueError, match='cannot exact resume') as excinfo:
        validate_exact_resume(_save(tmp_path / 'legacy.ckpt'), runtime_contract)

    assert D2_REFUSAL in str(excinfo.value)
    assert 'weights_only' not in str(excinfo.value)

# -------------------------------------------------------------------------------------------------
# Export and reads compare the supervision fields only
# -------------------------------------------------------------------------------------------------

def test_the_supervision_check_takes_the_encoder_record_from_the_checkpoint(validated_bundle):
    manifest = validated_bundle.manifest
    other = shared_encoder_architecture(fusion='attention', dimension=8, backbone=MINILM)
    saved = contract_for_bundle(manifest, encoder=other)

    assert validate_supervision_contract(saved.model_dump(), manifest) == saved

@pytest.mark.parametrize(
    'update',
    [
        {'bundle_id': 'other-bundle'},
        {'codebook_fingerprint': 'f' * 64},
        {'supervision_mode': 'legacy_containment'},
    ],
)
def test_the_supervision_check_refuses_another_bundle(validated_bundle, update):
    manifest = validated_bundle.manifest
    saved = contract_for_bundle(manifest, encoder=SHARED).model_copy(update=update)

    with pytest.raises(ValueError, match='supervision contract mismatch'):
        validate_supervision_contract(saved.model_dump(), manifest)

def test_the_supervision_check_refuses_a_checkpoint_without_a_contract(validated_bundle):
    with pytest.raises(ValueError, match='no Stage-3 contract') as excinfo:
        validate_supervision_contract(None, validated_bundle.manifest)

    assert D2_REFUSAL in str(excinfo.value)

# -------------------------------------------------------------------------------------------------
# Weights-only migration
# -------------------------------------------------------------------------------------------------

def test_weights_only_loads_allowlisted_encoder_and_resets_training_state(
    tmp_path, runtime_contract, tiny_repaired_model
):
    encoder_key = next(
        name for name in tiny_repaired_model.state_dict() if name.startswith('encoder.')
    )
    path = _save(
        tmp_path / 'checkpoint.ckpt',
        runtime_contract,
        state_dict={
            encoder_key: torch.ones_like(tiny_repaired_model.state_dict()[encoder_key]),
            'lambdarank_loss_fn.tree_distances': torch.ones((3, 3)),
            'unexpected.weight': torch.ones(1),
        },
        optimizer_states=[{'state': {'x': 1}}],
        epoch=9,
        global_step=123,
    )

    with pytest.raises(ValueError, match='unexpected.weight'):
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

def test_weights_only_reports_loaded_skipped_and_missing_without_restoring_state(
    tmp_path, runtime_contract, tiny_repaired_model
):
    target = tiny_repaired_model.state_dict()
    encoder_keys = sorted(name for name in target if name.startswith('encoder.'))
    loaded_key = encoder_keys[0]
    initial_flags = dict(tiny_repaired_model.current_curriculum_flags)
    # The same architecture under another bundle, which weights-only still serves (D2)
    path = _save(
        tmp_path / 'other-bundle.ckpt',
        runtime_contract.model_copy(update={'bundle_id': 'bundle-b'}),
        state_dict={
            loaded_key: torch.full_like(target[loaded_key], 0.25),
            'loss_fn.legacy_buffer': torch.ones(1),
        },
        optimizer_states=[{'state': {'legacy': 1}}],
        epoch=9,
        global_step=123,
    )

    report = load_weights_only(tiny_repaired_model, path, encoder=SHARED)

    assert report.loaded == (loaded_key, )
    assert report.skipped == ('loss_fn.legacy_buffer', )
    assert report.missing == tuple(encoder_keys[1:])
    assert report.unexpected == ()
    assert tiny_repaired_model.current_curriculum_flags == initial_flags
    assert torch.equal(
        tiny_repaired_model.state_dict()[loaded_key],
        torch.full_like(target[loaded_key], 0.25),
    )

def test_weights_only_rejects_a_checkpoint_with_no_encoder_weights(
    tmp_path, runtime_contract, tiny_repaired_model
):
    path = _save(
        tmp_path / 'loss_only.ckpt',
        runtime_contract,
        state_dict={'loss_fn.legacy_buffer': torch.ones(1)},
    )

    with pytest.raises(ValueError, match='no allowlisted encoder parameters'):
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

def test_weights_only_rejects_shape_mismatched_encoder_weights(
    tmp_path, runtime_contract, tiny_repaired_model
):
    path = _save(
        tmp_path / 'mismatched.ckpt',
        runtime_contract,
        state_dict={'encoder.0.weight': torch.ones((5, 5))},
    )

    with pytest.raises(ValueError, match='encoder.0.weight'):
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

@pytest.mark.parametrize(
    'saved',
    [
        None,
        LEGACY_ENCODER,
        shared_encoder_architecture(fusion='masked_mean', dimension=8, backbone=MINILM),
    ],
)
def test_weights_only_refuses_another_encoder_before_reading_any_parameter(
    tmp_path, runtime_contract, tiny_repaired_model, saved
):
    target = tiny_repaired_model.state_dict()
    key = sorted(name for name in target if name.startswith('encoder.'))[0]
    before = target[key].clone()
    contract = None if saved is None else runtime_contract.model_copy(update={'encoder': saved})
    path = _save(tmp_path / 'other.ckpt', contract, state_dict={key: torch.full_like(before, 0.25)})

    with pytest.raises(ValueError, match='weights-only encoder mismatch') as excinfo:
        load_weights_only(tiny_repaired_model, path, encoder=SHARED)

    assert D2_REFUSAL in str(excinfo.value)
    assert torch.equal(tiny_repaired_model.state_dict()[key], before)
```

In `tests/unit/test_naics_model.py`, replace:

```python
from naics_embedder.text_model.dataloader.datamodule import collate_fn
```

with:

```python
from naics_embedder.supervision.checkpoints import contract_for_bundle, shared_encoder_architecture
from naics_embedder.text_model.dataloader.datamodule import collate_fn
```

In `TestCheckpointContract`, replace:

```python
        assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint

    def test_on_save_checkpoint_writes_contract(self, naics_model):
```

with:

```python
        assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
        assert contract.encoder == shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone='sentence-transformers/all-MiniLM-L6-v2'
        )

    def test_a_runtime_contract_of_another_encoder_is_refused(
        self, model_config, validated_bundle
    ):
        other = contract_for_bundle(
            validated_bundle.manifest,
            encoder=shared_encoder_architecture(
                fusion='masked_mean', dimension=8, backbone=model_config['base_model_name']
            ),
        )

        with pytest.raises(ValueError, match='does not match'):
            NAICSContrastiveModel(**model_config, checkpoint_contract=other)

    def test_on_save_checkpoint_writes_contract(self, naics_model):
```

Replace:

```python
        restored = NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

        assert restored.checkpoint_contract == naics_model.checkpoint_contract
```

with:

```python
        restored = NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

        assert restored.checkpoint_contract == naics_model.checkpoint_contract
        assert restored.checkpoint_contract.encoder.layout == 'shared'
        assert restored.encoder.dimension == 16
```

Add after `test_load_from_checkpoint_rejects_a_legacy_checkpoint`:

```python
    def test_load_from_checkpoint_refuses_a_four_copy_checkpoint_before_its_weights(
        self, naics_model, tmp_path
    ):
        '''Spec 4.4: a pre-Stage-6 checkpoint meets the D2 refusal, never a state-dict key error.'''

        checkpoint = _lightning_checkpoint(naics_model)
        # Contracts saved before Stage 6 carry no encoder record, and their hyperparameters
        # predate fusion and dimension
        del checkpoint['stage3_supervision']['encoder']
        for name in ('fusion', 'dimension'):
            del checkpoint['hyper_parameters'][name]
        # The four-copy layout's keys, which a strict load_state_dict would reject
        checkpoint['state_dict'] = {
            'encoder.encoders.title.base_model.model.embeddings.word_embeddings.weight':
            torch.zeros(1)
        }
        path = tmp_path / 'four-copy.ckpt'
        torch.save(checkpoint, path)

        with pytest.raises(ValueError, match='D2'):
            NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu')

    def test_load_from_checkpoint_refuses_another_dimension(self, naics_model, tmp_path):
        path = tmp_path / 'shared.ckpt'
        torch.save(_lightning_checkpoint(naics_model), path)

        with pytest.raises(ValueError, match='D2'):
            NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu', dimension=8)
```

In `tests/unit/test_cli_training.py`, replace:

```python
from naics_embedder.supervision.checkpoints import CheckpointContract, MigrationReport
```

with:

```python
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    MigrationReport,
    shared_encoder_architecture,
)
```

Replace:

```python
@pytest.fixture
def cli_runner():
```

with:

```python
# The record the default config builds
CONFIGURED_ENCODER = shared_encoder_architecture(
    fusion='masked_mean', dimension=16, backbone='sentence-transformers/all-MiniLM-L6-v2'
)

@pytest.fixture
def cli_runner():
```

In `test_training_checkpoint_resume_passes_ckpt`, replace:

```python
    assert runtime == CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
    )
```

with:

```python
    assert runtime == CheckpointContract(
        supervision_mode='repaired',
        bundle_id='bundle-a',
        codebook_fingerprint='a' * 64,
        encoder=CONFIGURED_ENCODER,
    )
```

In `test_weights_only_never_passes_checkpoint_to_trainer`, replace:

```python
    def fake_load_weights_only(model, path):
        assert path == 'legacy.ckpt'
```

with:

```python
    def fake_load_weights_only(model, path, *, encoder):
        assert path == 'legacy.ckpt'
        assert encoder == CONFIGURED_ENCODER
```

In `test_weights_only_without_an_existing_checkpoint_is_fatal`, replace:

```python
        lambda *_args: pytest.fail('nothing to migrate'),
```

with:

```python
        lambda *_args, **_kwargs: pytest.fail('nothing to migrate'),
```

Add after `test_training_checkpoint_resume_passes_ckpt`:

```python
@pytest.mark.unit
def test_the_runtime_contract_records_the_configured_encoder(training_env):
    training.train(skip_validation=True, overrides=['model.fusion=attention', 'model.dimension=8'])

    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['checkpoint_contract'].encoder == shared_encoder_architecture(
        fusion='attention', dimension=8, backbone='sentence-transformers/all-MiniLM-L6-v2'
    )
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('attention', 8)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_checkpoint_contract.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py -q`
Expected: `test_checkpoint_contract.py` errors at collection with
`ImportError: cannot import name 'D2_REFUSAL'`. The new model and CLI tests fail: the contract has
no `encoder`.

- [ ] **Step 3: Rewrite the contract module**

Replace the whole of `src/naics_embedder/supervision/checkpoints.py` with:

```python
'''
Stage-3 checkpoint contracts: exact resume and explicit weights-only migration.

A repaired checkpoint records the supervision contract it was trained under and the encoder
architecture its weights belong to. Exact resume restores optimizer, epoch, curriculum, and
sampler state, so it requires an identical contract. A checkpoint of the same architecture under
other supervision can only contribute allowlisted encoder weights, through an explicit
weights-only migration that leaves all training state freshly initialized.

Another architecture can do neither. A checkpoint saved before Stage 6 has no encoder record and
reads as the legacy four-copy layout, and nothing migrates it into the shared encoder (roadmap
D2).
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict, model_validator

from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    MINING_CONTRACT_VERSION,
    STRUCTURAL_PREFERENCE_LOSS_VERSION,
)

CHECKPOINT_KEY = 'stage3_supervision'
LEGACY_CONTAINMENT_BUNDLE_ID = 'legacy-containment'
UNVERSIONED_CODEBOOK_FINGERPRINT = 'unversioned'
WEIGHTS_ONLY_ALLOWED_PREFIXES = ('encoder.', )
WEIGHTS_ONLY_EXCLUDED_PREFIXES = (
    'loss_fn.',
    'hierarchy_loss_fn.',
    'lambdarank_loss_fn.',
    'structural_preference_loss_fn.',
    'ground_truth_distances',
    'norm_adaptive_margin.',
)
D2_REFUSAL = (
    'a checkpoint of another encoder architecture cannot load, and nothing migrates it: '
    'four-copy checkpoints cannot load into the shared encoder (roadmap D2)'
)

# -------------------------------------------------------------------------------------------------
# Contract
# -------------------------------------------------------------------------------------------------

class EncoderArchitecture(BaseModel):
    '''
    The encoder architecture a checkpoint's weights belong to (spec 4.4).

    ``shared`` is Stage 6's one backbone, and names its fusion, dimension and backbone.
    ``four-copy`` is the legacy layout of every checkpoint saved before Stage 6, and names nothing
    else. A field added later defaults to the value every earlier checkpoint had.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')

    layout: Literal['shared', 'four-copy']
    fusion: Optional[str] = None
    dimension: Optional[int] = None
    backbone: Optional[str] = None

    @model_validator(mode='after')
    def check_fields_match_the_layout(self) -> 'EncoderArchitecture':
        '''A shared record names its fusion, dimension and backbone; a four-copy one, none.'''

        recorded = (self.fusion, self.dimension, self.backbone)
        if self.layout == 'shared' and None in recorded:
            raise ValueError('a shared encoder record names its fusion, dimension and backbone')
        if self.layout == 'four-copy' and recorded != (None, None, None):
            raise ValueError('a four-copy encoder record names no fusion, dimension or backbone')
        return self

LEGACY_ENCODER = EncoderArchitecture(layout='four-copy')

def shared_encoder_architecture(
    *,
    fusion: str,
    dimension: int,
    backbone: str,
) -> EncoderArchitecture:
    '''
    The record of a Stage-6 shared encoder.

    The model builds its record here from its hyperparameters, and training builds the config's
    here too, so the two cannot drift apart.
    '''

    return EncoderArchitecture(
        layout='shared', fusion=fusion, dimension=dimension, backbone=backbone
    )

class CheckpointContract(BaseModel):
    '''The supervision identity a checkpoint was trained under, and its encoder architecture.'''

    model_config = ConfigDict(frozen=True, extra='forbid')

    supervision_mode: str
    contract_version: str = CONTRACT_VERSION
    bundle_id: str
    codebook_fingerprint: str
    structural_preference_loss_version: str = STRUCTURAL_PREFERENCE_LOSS_VERSION
    mining_contract_version: str = MINING_CONTRACT_VERSION
    # Absent from every contract saved before Stage 6, which therefore reads as four-copy
    encoder: EncoderArchitecture = LEGACY_ENCODER

@dataclass(frozen=True)
class MigrationReport:
    '''Parameter groups a weights-only migration loaded, skipped, or left freshly initialized.'''

    loaded: Tuple[str, ...]
    skipped: Tuple[str, ...]
    missing: Tuple[str, ...]
    unexpected: Tuple[str, ...]

def contract_for_bundle(
    manifest: Any,
    supervision_mode: str = 'repaired',
    *,
    encoder: EncoderArchitecture,
) -> CheckpointContract:
    '''The runtime contract for training against a validated bundle manifest.'''

    return CheckpointContract(
        supervision_mode=supervision_mode,
        contract_version=manifest.contract_version,
        bundle_id=manifest.bundle_id,
        codebook_fingerprint=manifest.codebook_fingerprint,
        encoder=encoder,
    )

def containment_contract(*, encoder: EncoderArchitecture) -> CheckpointContract:
    '''
    The tag every legacy-containment checkpoint carries.

    It can never equal a repaired contract, so containment checkpoints cannot exact-resume into
    repaired training.
    '''

    return CheckpointContract(
        supervision_mode='legacy_containment',
        bundle_id=LEGACY_CONTAINMENT_BUNDLE_ID,
        codebook_fingerprint=UNVERSIONED_CODEBOOK_FINGERPRINT,
        encoder=encoder,
    )

def saved_encoder(raw: Optional[Dict[str, Any]]) -> EncoderArchitecture:
    '''A saved contract's encoder record. No contract, or no record, is the four-copy layout.'''

    if raw is None:
        return LEGACY_ENCODER
    return CheckpointContract.model_validate(raw).encoder

def _differences(saved: CheckpointContract,
                 expected: CheckpointContract) -> Dict[str, Tuple[Any, Any]]:
    return {
        name: (getattr(saved, name), getattr(expected, name))
        for name in CheckpointContract.model_fields
        if getattr(saved, name) != getattr(expected, name)
    }

# -------------------------------------------------------------------------------------------------
# Exact resume
# -------------------------------------------------------------------------------------------------

def _load_checkpoint(path: str | Path) -> Dict[str, Any]:
    # Lightning checkpoints carry pickled hyperparameters and loop state; they are trusted
    # artifacts produced by this project's own training runs.
    return torch.load(Path(path), map_location='cpu', weights_only=False)

def validate_checkpoint_contract(
    raw: Optional[Dict[str, Any]],
    runtime: CheckpointContract,
) -> None:
    '''
    Require a saved checkpoint contract identical to the runtime contract.

    Raises:
        ValueError: If the checkpoint predates the contract (legacy) or any field differs. A
            checkpoint without a contract, or of another encoder architecture, carries the D2
            refusal.
    '''

    if raw is None:
        raise ValueError(
            f'legacy checkpoint has no Stage-3 contract and cannot exact resume; {D2_REFUSAL}'
        )
    saved = CheckpointContract.model_validate(raw)
    if saved != runtime:
        differences = _differences(saved, runtime)
        message = f'exact resume contract mismatch (saved, runtime): {differences}'
        if 'encoder' in differences:
            message = f'{message}; {D2_REFUSAL}'
        raise ValueError(message)

def validate_exact_resume(path: str | Path, runtime: CheckpointContract) -> None:
    '''Require that the checkpoint at ``path`` was trained under the runtime contract.'''

    validate_checkpoint_contract(_load_checkpoint(path).get(CHECKPOINT_KEY), runtime)

def validate_supervision_contract(
    raw: Optional[Dict[str, Any]],
    manifest: Any,
    supervision_mode: str = 'repaired',
) -> CheckpointContract:
    '''
    Require a saved contract whose supervision fields match the configured bundle's.

    Export and reads take the encoder record from the checkpoint (spec 4.4), so it is not compared
    here. ``load_from_checkpoint`` rebuilds the checkpoint's own architecture from its saved
    hyperparameters, and refuses a four-copy one.

    Returns:
        The saved contract.

    Raises:
        ValueError: If the checkpoint has no contract, or a supervision field differs.
    '''

    if raw is None:
        raise ValueError(f'legacy checkpoint has no Stage-3 contract; {D2_REFUSAL}')
    saved = CheckpointContract.model_validate(raw)
    configured = contract_for_bundle(manifest, supervision_mode, encoder=saved.encoder)
    if saved != configured:
        raise ValueError(
            'supervision contract mismatch (saved, configured): '
            f'{_differences(saved, configured)}'
        )
    return saved

# -------------------------------------------------------------------------------------------------
# Weights-only migration
# -------------------------------------------------------------------------------------------------

def load_weights_only(
    model: torch.nn.Module,
    path: str | Path,
    *,
    encoder: EncoderArchitecture,
) -> MigrationReport:
    '''
    Load only allowlisted encoder weights; never optimizer, epoch, curriculum, or sampler state.

    The saved encoder record (an absent one counts as four-copy) must equal ``encoder`` before any
    parameter is read (roadmap D2). Loss buffers and legacy structural matrices are skipped;
    bundle-derived buffers stay as the runtime bundle built them.

    Args:
        model: The freshly built runtime model.
        path: The checkpoint to migrate from.
        encoder: The runtime model's encoder record.

    Raises:
        ValueError: If the checkpoint's encoder record differs from ``encoder``; if it has no
            state dict; if it carries parameters that are neither allowlisted nor known-excluded,
            or allowlisted parameters with mismatched shapes; or if it contributes no allowlisted
            parameter at all.
    '''

    checkpoint = _load_checkpoint(path)
    saved = saved_encoder(checkpoint.get(CHECKPOINT_KEY))
    if saved != encoder:
        raise ValueError(
            f'weights-only encoder mismatch (saved, runtime): {(saved, encoder)}; {D2_REFUSAL}'
        )
    source = checkpoint.get('state_dict')
    if not isinstance(source, dict):
        raise ValueError('weights-only checkpoint has no state_dict')
    target = model.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    skipped = []
    unexpected = []
    for name, value in source.items():
        if name.startswith(WEIGHTS_ONLY_ALLOWED_PREFIXES):
            if name not in target or target[name].shape != value.shape:
                unexpected.append(name)
            else:
                loaded[name] = value
        elif name.startswith(WEIGHTS_ONLY_EXCLUDED_PREFIXES):
            skipped.append(name)
        else:
            unexpected.append(name)
    if unexpected:
        raise ValueError(
            f'weights-only checkpoint has unexpected parameter groups: {sorted(unexpected)}'
        )
    if not loaded:
        raise ValueError(
            f'weights-only checkpoint {path} has no allowlisted encoder parameters to load'
        )
    model.load_state_dict(loaded, strict=False)
    missing = tuple(
        sorted(
            name for name in target
            if name.startswith(WEIGHTS_ONLY_ALLOWED_PREFIXES) and name not in loaded
        )
    )
    return MigrationReport(
        loaded=tuple(sorted(loaded)),
        skipped=tuple(sorted(skipped)),
        missing=missing,
        unexpected=(),
    )
```

The validator's name has no leading underscore: Pydantic treats an underscore-prefixed class
attribute as a private attribute.

- [ ] **Step 4: Give the model its record**

In `src/naics_embedder/text_model/naics_model.py`, replace:

```python
    containment_contract,
    contract_for_bundle,
    validate_checkpoint_contract,
)
```

with:

```python
    containment_contract,
    contract_for_bundle,
    shared_encoder_architecture,
    validate_checkpoint_contract,
)
```

Replace:

```python
        self.save_hyperparameters(ignore=['checkpoint_contract', 'supervision_bundle'])
        self.supervision_mode = supervision_mode
```

with:

```python
        self.save_hyperparameters(ignore=['checkpoint_contract', 'supervision_bundle'])
        self.supervision_mode = supervision_mode
        # The architecture this model's weights belong to; a checkpoint of any other is refused
        # (spec 4.4, roadmap D2)
        encoder_record = shared_encoder_architecture(
            fusion=fusion, dimension=dimension, backbone=base_model_name
        )
```

Replace:

```python
            runtime_contract = contract_for_bundle(bundle.manifest, supervision_mode)
```

with:

```python
            runtime_contract = contract_for_bundle(
                bundle.manifest, supervision_mode, encoder=encoder_record
            )
```

Replace:

```python
            runtime_contract = containment_contract()
```

with:

```python
            runtime_contract = containment_contract(encoder=encoder_record)
```

Replace:

```python
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        '''Record the supervision contract this checkpoint was trained under.'''
        checkpoint[CHECKPOINT_KEY] = self.checkpoint_contract.model_dump()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        '''
        Refuse to restore a checkpoint trained under any other supervision contract.

        Runs for Lightning exact resume and ``load_from_checkpoint``; weights-only migration
        never reaches this hook.
        '''
```

with:

```python
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        '''Record the supervision contract and encoder architecture this checkpoint belongs to.'''
        checkpoint[CHECKPOINT_KEY] = self.checkpoint_contract.model_dump()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        '''
        Refuse to restore a checkpoint of any other supervision contract or encoder architecture.

        Runs for Lightning exact resume and ``load_from_checkpoint`` before the state dict loads,
        so a four-copy checkpoint meets the D2 refusal, never a key mismatch. Weights-only
        migration never reaches this hook; it checks the encoder record itself.
        '''
```

- [ ] **Step 5: Build the config's record in the CLI**

In `src/naics_embedder/cli/commands/training.py`, replace:

```python
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    MigrationReport,
    containment_contract,
    contract_for_bundle,
    load_weights_only,
    validate_exact_resume,
)
```

with:

```python
from naics_embedder.supervision.checkpoints import (
    CheckpointContract,
    EncoderArchitecture,
    MigrationReport,
    containment_contract,
    contract_for_bundle,
    load_weights_only,
    shared_encoder_architecture,
    validate_exact_resume,
)
```

Replace the whole `runtime_contract_for` function with:

```python
def encoder_architecture_for(cfg: Config) -> EncoderArchitecture:
    '''
    The configured run's encoder record.

    It comes from the same helper the model builds its own record with, so the two cannot drift
    (spec 4.4).
    '''

    return shared_encoder_architecture(
        fusion=cfg.model.fusion,
        dimension=cfg.model.dimension,
        backbone=cfg.model.base_model_name,
    )

def runtime_contract_for(
    cfg: Config, bundle: Optional[ValidatedSupervisionBundle]
) -> CheckpointContract:
    '''
    The checkpoint contract of the configured run, its encoder record included.

    The supervision gate returns no bundle only for explicit legacy containment. Training's exact
    resume and the HGCN feeder compare this whole contract with a checkpoint's; export and reads
    take the encoder record from the checkpoint instead (spec 4.4).
    '''

    encoder = encoder_architecture_for(cfg)
    if bundle is None:
        return containment_contract(encoder=encoder)
    return contract_for_bundle(bundle.manifest, cfg.supervision.mode, encoder=encoder)
```

In `train`'s `--checkpoint-load-mode` option, replace:

```python
                'exact: resume optimizer/epoch/curriculum state (requires a matching supervision '
                'contract); weights_only: load allowlisted encoder weights into a fresh run'
```

with:

```python
                'exact: resume optimizer/epoch/curriculum state (requires a matching supervision '
                'contract); weights_only: load allowlisted encoder weights of the same encoder '
                'architecture into a fresh run'
```

In `train`'s docstring, replace:

```python
            checkpoint's supervision contract to match the runtime bundle; ``weights_only``
            loads allowlisted encoder weights into a fresh run starting at epoch zero.
```

with:

```python
            checkpoint's supervision contract to match the runtime bundle; ``weights_only``
            loads allowlisted encoder weights of the same encoder architecture into a fresh run
            starting at epoch zero.
```

Replace:

```python
            report = load_weights_only(model, checkpoint_path)
```

with:

```python
            report = load_weights_only(model, checkpoint_path, encoder=runtime_contract.encoder)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_checkpoint_contract.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py tests/integration/test_stage3_training_step.py -q`
Expected: all pass.

- [ ] **Step 7: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/supervision/checkpoints.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/cli/commands/training.py tests/unit/test_checkpoint_contract.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/supervision/checkpoints.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/cli/commands/training.py tests/unit/test_checkpoint_contract.py tests/unit/test_naics_model.py tests/unit/test_cli_training.py
git commit -m "feat(supervision): record the encoder architecture; refuse any other (D2)"
```

### Task 9: Encoding token rows, and the HGCN feeder through them

Export, the arm encoder and the HGCN feeder all run token rows through an arm's model. This task
builds that one function and the config of the cache it reads. It then moves the feeder onto both,
which closes the feeder gap Task 7 opened: since Task 7 the feeder's hand-built batches carry no
`present` flag, so it cannot run. The fixture module gains a five-code arm, which Tasks 10–12
reuse.

**Files:**
- Create: `src/naics_embedder/text_model/export.py`
- Modify: `src/naics_embedder/cli/commands/training.py`: the imports (`:17-56` at 52075f9) and
  the body of `generate_embeddings_from_checkpoint` (`:262-377` at 52075f9)
- Modify: `tests/fixtures/shared_encoder.py`
- Test: `tests/unit/test_export.py` (create)

**Interfaces:**
- Consumes:
  - `CHANNELS` (Task 2) and `stack_text_inputs(embeddings, fields=CHANNELS)` (Task 3);
  - `tiny_backbone` and `TINY_HIDDEN` (Task 5);
  - `NAICSContrastiveModel(..., fusion, dimension, ...)`, whose forward returns `embedding` and
    `tangent` (Task 7);
  - the encoder record and `runtime_contract_for` (Task 8);
  - from `tests/fixtures/supervision.py`: `text_descriptions_fixture`, `generated_bundle` and
    `validated_bundle`.
- Produces:
  - In `naics_embedder.text_model.export`:
    - `code_token_config(cfg: Config) -> TokenizationConfig`: the cache training reads. Its
      descriptions are `data_loader.streaming.descriptions_parquet`, its tokenizer is
      `data_loader.tokenization.tokenizer_name`, its `max_length` is
      `data_loader.streaming.max_length`, and its `output_path` is the default.
    - `encode_token_rows(model: torch.nn.Module, rows: Sequence[Mapping[str, Mapping[str, Any]]],
      *, fields: Sequence[str] = CHANNELS, batch_size: int = 32) -> Dict[str, torch.Tensor]`.
      - It returns `tangent` (N, d) and `embedding` (N, d + 1), float64 on the CPU, in row order.
      - The model runs in eval mode and without gradient, and stays in eval mode. The inputs go
        to the model's device.
      - It raises `ValueError` ("no token rows") on an empty `rows`.
  - In `tests.fixtures.shared_encoder`:
    - `MINILM`, `FIVE_CODES = ('111111', '111112', '111113', '222222', '333333')`,
      `ARM_DIMENSION = 16` and `TOKEN_WINDOW = 32`;
    - `lightning_checkpoint(model) -> Dict[str, Any]`: the dict Lightning saves, the contract
      included;
    - fixtures:
      - `five_code_descriptions_parquet`: a `Path` under `tmp_path`, with `level` 6;
      - `five_code_token_config`: a `TokenizationConfig` over that file, MiniLM's tokenizer, a
        window of `TOKEN_WINDOW` and the cache under `tmp_path`;
      - `shared_model`: a d = 16 `masked_mean` model of the five-code bundle on the tiny
        backbone, in eval mode;
      - `shared_checkpoint`: that model saved to `tmp_path / 'arm.ckpt'`.
  - `generate_embeddings_from_checkpoint` keeps its signature. It writes `index` and `level`
    (Int64), `code`, then `hyp_e0 … hyp_e{d}` (Float64): d + 1 Lorentz columns. A missing or
    stale cache now rebuilds instead of failing (P19).

- [ ] **Step 1: Extend the fixture module**

In `tests/fixtures/shared_encoder.py`, replace:

```python
'''
A tiny backbone for the shared encoder's tests, so they download nothing (spec §6).

``tiny_bert`` builds a one-layer BERT whose vocabulary is MiniLM's, so token rows from the real
tokenizer fit it. ``tiny_backbone`` makes every ``SharedEncoder`` a test builds load it in place of
MiniLM.
'''

import pytest
import torch
from transformers import BertConfig, BertModel
```

with:

```python
'''
A tiny backbone for the shared encoder's tests, so they download nothing (spec §6), and a
five-code arm built on it.

``tiny_bert`` builds a one-layer BERT whose vocabulary is MiniLM's, so token rows from the real
tokenizer fit it. ``tiny_backbone`` makes every ``SharedEncoder`` a test builds load it in place of
MiniLM.

The arm fixtures train nothing. ``shared_model`` is a d = 16 model of the five-code supervision
bundle (``tests/fixtures/supervision.py``) on the tiny backbone, and ``shared_checkpoint`` saves it
as Lightning would.
'''

from pathlib import Path
from typing import Any, Dict

import polars as pl
import pytest
import pytorch_lightning as pyl
import torch
from transformers import BertConfig, BertModel

from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import TokenizationConfig
```

Then append at the end of the file:

```python
# -------------------------------------------------------------------------------------------------
# A shared-encoder arm of the five-code bundle
# -------------------------------------------------------------------------------------------------

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
# The five-code bundle's codebook, in code_id order
FIVE_CODES = ('111111', '111112', '111113', '222222', '333333')
ARM_DIMENSION = 16
TOKEN_WINDOW = 32

def lightning_checkpoint(model: NAICSContrastiveModel) -> Dict[str, Any]:
    '''The dict Lightning saves for ``model``, its checkpoint contract included.'''

    checkpoint = {
        'state_dict': model.state_dict(),
        'hyper_parameters': dict(model.hparams),
        'pytorch-lightning_version': pyl.__version__,
    }
    model.on_save_checkpoint(checkpoint)
    return checkpoint

@pytest.fixture
def five_code_descriptions_parquet(tmp_path, text_descriptions_fixture) -> Path:
    '''The five-code descriptions with their text channels and a ``level``, under ``tmp_path``.'''

    path = tmp_path / 'naics_descriptions.parquet'
    text_descriptions_fixture.with_columns(level=pl.lit(6)).write_parquet(path)
    return path

@pytest.fixture
def five_code_token_config(tmp_path, five_code_descriptions_parquet) -> TokenizationConfig:
    '''The five codes' token cache: MiniLM's tokenizer, a 32-token window, under ``tmp_path``.'''

    return TokenizationConfig(
        descriptions_parquet=str(five_code_descriptions_parquet),
        tokenizer_name=MINILM,
        max_length=TOKEN_WINDOW,
        output_path=str(tmp_path / 'token_cache' / 'token_cache.pt'),
    )

@pytest.fixture
def shared_model(tiny_backbone, generated_bundle) -> NAICSContrastiveModel:
    '''A d = 16 masked-mean model of the five-code bundle on the tiny backbone, in eval mode.'''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        fusion='masked_mean',
        dimension=ARM_DIMENSION,
        curvature=1.0,
        supervision_manifest_path=str(generated_bundle),
    )
    return model.eval()

@pytest.fixture
def shared_checkpoint(tmp_path, shared_model) -> Path:
    '''``shared_model`` saved as a Lightning checkpoint.'''

    path = tmp_path / 'arm.ckpt'
    torch.save(lightning_checkpoint(shared_model), path)
    return path
```

`shared_model` requests `tiny_backbone`, so any test that loads `shared_checkpoint` back through
`load_from_checkpoint` rebuilds it on the tiny backbone too. The model's own default backbone name
is `all-mpnet-base-v2`, which is why `base_model_name` is passed.

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_export.py`:

```python
'''
Encoding token rows through an arm's model, the HGCN feeder built on it, and the code-table
export (spec 4.3).
'''

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.cli.commands import training as training_cli
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import code_token_config, encode_token_rows
from naics_embedder.utils.config import Config
from tests.fixtures.shared_encoder import ARM_DIMENSION, FIVE_CODES, MINILM, TOKEN_WINDOW

pytestmark = pytest.mark.unit

def _token_rows(token_config, bundle):
    '''The five codes' cached token rows, in codebook order (the descriptions' ``index``).'''

    cache = tokenization_cache(
        token_config,
        description_fingerprint=bundle.manifest.description_fingerprint,
        codebook_fingerprint=bundle.manifest.codebook_fingerprint,
    )
    return [cache[index] for index in range(len(FIVE_CODES))]

# -------------------------------------------------------------------------------------------------
# Encoding token rows
# -------------------------------------------------------------------------------------------------

def test_code_token_config_is_the_cache_training_reads():
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = '/data/descriptions.parquet'
    cfg.data_loader.streaming.max_length = 64
    # The window the feeder used to read, which training never did
    cfg.data_loader.tokenization.max_length = 32

    token_config = code_token_config(cfg)

    assert token_config.descriptions_parquet == '/data/descriptions.parquet'
    assert token_config.tokenizer_name == MINILM
    assert token_config.max_length == 64
    assert token_config.output_path == './data/token_cache/token_cache.pt'

def test_rows_encode_to_float64_cpu_tensors_in_row_order(
    shared_model, five_code_token_config, validated_bundle
):
    rows = _token_rows(five_code_token_config, validated_bundle)

    encoded = encode_token_rows(shared_model, rows, batch_size=2)

    assert encoded['tangent'].shape == (5, ARM_DIMENSION)
    assert encoded['embedding'].shape == (5, ARM_DIMENSION + 1)
    for tensor in encoded.values():
        assert tensor.dtype == torch.float64
        assert tensor.device.type == 'cpu'
    # Three batches give what one forward pass over all five rows gives
    with torch.no_grad():
        whole = shared_model(stack_text_inputs(rows))
    assert torch.allclose(encoded['tangent'], whole['tangent'].to(torch.float64), atol=1e-6)
    assert torch.allclose(encoded['embedding'], whole['embedding'].to(torch.float64), atol=1e-6)

def test_rows_encode_in_eval_mode(shared_model, five_code_token_config, validated_bundle):
    rows = _token_rows(five_code_token_config, validated_bundle)
    # BERT's dropout would make two training-mode passes differ
    shared_model.train()

    first = encode_token_rows(shared_model, rows)['tangent']

    assert not shared_model.training
    assert torch.equal(first, encode_token_rows(shared_model, rows)['tangent'])

def test_no_rows_are_refused(shared_model):
    with pytest.raises(ValueError, match='no token rows'):
        encode_token_rows(shared_model, [])

# -------------------------------------------------------------------------------------------------
# The HGCN feeder
# -------------------------------------------------------------------------------------------------

def test_the_hgcn_feeder_writes_d_plus_one_lorentz_columns(
    monkeypatch, tmp_path, shared_checkpoint, validated_bundle, five_code_descriptions_parquet
):
    # The fixture bundle's description fingerprint hashes the frame, not the file, so the real
    # gate would refuse it
    monkeypatch.setattr(
        training_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle
    )
    monkeypatch.setattr(training_cli, 'pick_device', lambda *_args: torch.device('cpu'))
    # code_token_config keeps the default ./data/token_cache path; the descriptions path is
    # absolute, so it survives the move
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.data_loader.streaming.descriptions_parquet = str(five_code_descriptions_parquet)
    cfg.data_loader.streaming.max_length = TOKEN_WINDOW
    output = tmp_path / 'encodings.parquet'

    # No cache exists yet: the feeder builds it (P19), where it used to fail fast
    training_cli.generate_embeddings_from_checkpoint(
        str(shared_checkpoint), cfg, str(output), batch_size=2
    )

    table = pl.read_parquet(output)
    columns = [f'hyp_e{index}' for index in range(ARM_DIMENSION + 1)]
    assert table.columns == ['index', 'level', 'code', *columns]
    assert table.schema['index'] == pl.Int64
    assert table.schema['level'] == pl.Int64
    assert table.get_column('code').to_list() == list(FIVE_CODES)
    points = table.select(columns).to_numpy()
    # Each row lies on the hyperboloid: -x0^2 + |x|^2 = -1
    assert np.allclose(-points[:, 0]**2 + (points[:, 1:]**2).sum(axis=1), -1.0, atol=1e-4)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_export.py -q`
Expected: a collection error,
`ModuleNotFoundError: No module named 'naics_embedder.text_model.export'`.

- [ ] **Step 4: Create `src/naics_embedder/text_model/export.py`**

```python
'''
Encoding an arm's codes and queries, and exporting its code table (spec 4.3).

``encode_token_rows`` runs token rows through an arm's model: a code's cached channels, or a
marked query. The HGCN feeder, the table export and the arm encoder all encode through it, so a
code embeds the same way wherever it is read.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, Dict, List, Mapping, Sequence

import torch

from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.fields import CHANNELS
from naics_embedder.utils.config import Config, TokenizationConfig

# -------------------------------------------------------------------------------------------------
# Encoding
# -------------------------------------------------------------------------------------------------

def code_token_config(cfg: Config) -> TokenizationConfig:
    '''
    The tokenization cache training reads, as ``NAICSDataModule`` builds it.

    The descriptions and the window are the streaming ones, the tokenizer is the tokenization
    one, and the path is the default. Export and reads therefore load the cache file that
    training built.
    '''

    return TokenizationConfig(
        descriptions_parquet=cfg.data_loader.streaming.descriptions_parquet,
        tokenizer_name=cfg.data_loader.tokenization.tokenizer_name,
        max_length=cfg.data_loader.streaming.max_length,
    )

def encode_token_rows(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    fields: Sequence[str] = CHANNELS,
    batch_size: int = 32,
) -> Dict[str, torch.Tensor]:
    '''
    Encode token rows through the model in batches, in eval mode and without gradient.

    A row maps each field to its tokens (``input_ids``, ``attention_mask`` and ``present``), as
    the tokenization cache stores a code or ``tokenize_field`` returns a query. The batches go to
    the model's device, and the model is left in eval mode.

    Args:
        model: A model whose forward returns ``tangent`` and ``embedding``: the shared encoder,
            or the Lightning module that holds it.
        rows: The token rows, in output order.
        fields: The fields read from each row.
        batch_size: Rows per forward pass.

    Returns:
        ``tangent`` (N, d) and ``embedding`` (N, d + 1), float64 on the CPU, in row order.

    Raises:
        ValueError: If there are no rows, or ``batch_size`` is not positive.
    '''

    if not rows:
        raise ValueError('there are no token rows to encode')
    if batch_size < 1:
        raise ValueError(f'batch_size must be positive, not {batch_size}')
    device = next(model.parameters()).device
    model.eval()
    parts: Dict[str, List[torch.Tensor]] = {'tangent': [], 'embedding': []}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = stack_text_inputs(rows[start:start + batch_size], fields)
            inputs = {
                field: {name: tensor.to(device) for name, tensor in tensors.items()}
                for field, tensors in batch.items()
            }
            output = model(inputs)
            for name, collected in parts.items():
                # .cpu() before the cast: casting an MPS tensor to float64 raises
                collected.append(output[name].cpu().to(torch.float64))
    return {name: torch.cat(collected) for name, collected in parts.items()}
```

- [ ] **Step 5: Move the HGCN feeder onto it**

In `src/naics_embedder/cli/commands/training.py`, delete the line `import torch`, which nothing
else in the module uses.

Replace:

```python
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import (
    CheckpointLoadMode,
    Config,
    TokenizationConfig,
)
```

with:

```python
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import code_token_config, encode_token_rows
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import (
    CheckpointLoadMode,
    Config,
)
```

In `generate_embeddings_from_checkpoint`, replace everything from the line
`    # Load descriptions parquet` through the line `    result_df = base_df.hstack(emb_df)`.
Each of those two lines occurs once in the file. The new text is:

```python
    # Load descriptions parquet
    descriptions_path = config.data_loader.streaming.descriptions_parquet
    logger.info(f'Loading NAICS descriptions from: {descriptions_path}')

    df = pl.read_parquet(descriptions_path).sort('index')
    logger.info(f'Loaded {df.height:,} NAICS codes')

    # The cache training reads; a missing or stale one is rebuilt (spec §5)
    logger.info('Loading tokenization cache...')
    token_cache = tokenization_cache(code_token_config(config), **token_fingerprints)
    logger.info('Tokenization cache loaded')

    # Every code through the shared encoder, in eval mode and without gradient
    logger.info(f'Generating embeddings (batch_size={batch_size})...')
    rows = [token_cache[index] for index in df.get_column('index').to_list()]
    embeddings = encode_token_rows(model, rows, batch_size=batch_size)['embedding']
    embedding_dim = embeddings.shape[1]
    logger.info(f'Generated embeddings: shape={tuple(embeddings.shape)}')

    # d + 1 Lorentz coordinates as hyp_e* columns, which HGCN finds by prefix
    emb_schema = {f'{STAGE3_EMBEDDING_PREFIX}{i}': pl.Float64 for i in range(embedding_dim)}
    emb_df = pl.DataFrame(embeddings.numpy(), schema=emb_schema, orient='row')

    # Combine with metadata, in the Int64 the feeder has always written
    base_df = df.select(
        pl.col('index').cast(pl.Int64),
        pl.col('level').cast(pl.Int64),
        pl.col('code'),
    )

    result_df = base_df.hstack(emb_df)
```

Everything above the replaced block stays as it is:
- the supervision gate and `validate_exact_resume` (P17);
- the legacy-containment branch, which Stage 7 deletes (D2, P27);
- `pick_device('auto')` and `load_from_checkpoint`.

Everything below it stays too: the parquet write and the closing log lines, which still read
`embedding_dim`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_export.py tests/unit/test_cli_training.py -q`
Expected: all pass.

Run: `uv run ruff check src/naics_embedder/cli/commands/training.py`
Expected: `All checks passed!` (no unused `torch` or `TokenizationConfig`).

- [ ] **Step 7: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/export.py src/naics_embedder/cli/commands/training.py tests/fixtures/shared_encoder.py tests/unit/test_export.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/text_model/export.py src/naics_embedder/cli/commands/training.py tests/fixtures/shared_encoder.py tests/unit/test_export.py
git commit -m "feat(text_model): encode token rows once; the HGCN feeder through the shared encoder"
```

### Task 10: The code-table export and `tools export-table`

This task exports every code's capped tangent vector at the origin in Req 2's form, in the
bundle's codebook order, with a provenance file that ties the table to its checkpoint (spec 4.3,
R6).

The checkpoint is checked before any weight loads:
1. the curvature guard (R8);
2. the supervision-only contract check (P16);
3. `load_from_checkpoint`, whose `on_load_checkpoint` refuses another encoder architecture (D2).

**Files:**
- Modify: `src/naics_embedder/text_model/export.py` (Task 9's module)
- Modify: `src/naics_embedder/cli/commands/tools.py`: the module docstring, the imports
  (`:21-60`), `text-only-table`'s provenance line (`:387`) and a new section at the end
- Modify: `tests/fixtures/shared_encoder.py`
- Test: `tests/unit/test_export.py`, `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes:
  - `code_token_config` and `encode_token_rows` (Task 9);
  - `validate_supervision_contract`, `CHECKPOINT_KEY`, `contract_for_bundle` and
    `shared_encoder_architecture` (Task 8);
  - `SUMMARIES` (Task 2);
  - the existing `panels.regressor.coordinate_matrix` and `table_fingerprint`, and
    `panels.text_only.provenance_path`.
- Produces, in `naics_embedder.text_model.export`:
  - `TABLE_PREFIX = 'e'` and `COORDINATES`, the provenance's description of the coordinates;
  - `require_unit_curvature(hyper_parameters: Mapping[str, Any]) -> None`. It raises
    `ValueError` matching `curvature {c:g}` when the saved curvature is not 1. An absent
    curvature reads as 1, the model's default.
  - `load_arm_model(checkpoint_path, bundle: ValidatedSupervisionBundle, *, device='cpu') ->
    Tuple[NAICSContrastiveModel, CheckpointContract]`.
    - It returns the model in eval mode on `device`, and the checkpoint's saved contract.
    - It raises `ValueError`: curvature (R8), "supervision contract mismatch", or the D2
      refusal.
  - `export_code_table(checkpoint_path, bundle, token_config: TokenizationConfig, output_path,
    *, device='cpu', batch_size=32) -> Path`.
    - It writes the table and `<stem>_provenance.json` (P20, P21) and returns the table's path.
    - It raises `ValueError` as `load_arm_model` does. It also raises one matching "codebook"
      when the descriptions' `(index, code)` rows are not the codebook's.
- Produces in `tests.fixtures.shared_encoder`: the fixture `exported_table`, the `Path` of
  `shared_checkpoint`'s table, exported on the CPU to `tmp_path / 'arm_table.parquet'`.
- Produces in `naics_embedder.cli.commands.tools`:
  - `_run_config(config_file: str, overrides: Optional[List[str]]) -> Config` and
    `_run_bundle(cfg: Config) -> ValidatedSupervisionBundle` (P18, P27);
  - `tools export-table --checkpoint <ckpt> --output <table.parquet> [--config <yaml>]
    [key=value ...]`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_export.py`, replace the imports:

```python
import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.cli.commands import training as training_cli
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import code_token_config, encode_token_rows
from naics_embedder.utils.config import Config
from tests.fixtures.shared_encoder import ARM_DIMENSION, FIVE_CODES, MINILM, TOKEN_WINDOW
```

with:

```python
import json

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.cli.commands import training as training_cli
from naics_embedder.panels.regressor import coordinate_matrix, table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.supervision.checkpoints import contract_for_bundle, shared_encoder_architecture
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import (
    code_token_config,
    encode_token_rows,
    export_code_table,
    load_arm_model,
)
from naics_embedder.utils.config import Config
from tests.fixtures.shared_encoder import (
    ARM_DIMENSION,
    FIVE_CODES,
    MINILM,
    TOKEN_WINDOW,
    lightning_checkpoint,
)
```

Then append at the end of the file:

```python
# -------------------------------------------------------------------------------------------------
# The code-table export
# -------------------------------------------------------------------------------------------------

COORDINATE_COLUMNS = [f'e{index}' for index in range(ARM_DIMENSION)]

def test_the_table_is_in_reqs_export_form(exported_table):
    '''Spec §6: code, index, level and e0 … e15 as float64, readable by coordinate_matrix.'''

    table = pl.read_parquet(exported_table)

    assert table.columns == ['code', 'index', 'level', *COORDINATE_COLUMNS]
    assert dict(table.schema) == {
        'code': pl.Utf8,
        'index': pl.Int64,
        'level': pl.Int64,
        **{column: pl.Float64
           for column in COORDINATE_COLUMNS},
    }
    # The bundle's codebook order
    assert table.get_column('code').to_list() == list(FIVE_CODES)
    assert table.get_column('index').to_list() == [0, 1, 2, 3, 4]
    codes, matrix = coordinate_matrix(table)
    assert codes == FIVE_CODES
    assert matrix.shape == (5, ARM_DIMENSION)

def test_the_table_holds_each_codes_capped_tangent(
    exported_table, shared_checkpoint, validated_bundle, five_code_token_config
):
    model, _ = load_arm_model(shared_checkpoint, validated_bundle)
    cache = tokenization_cache(
        five_code_token_config,
        description_fingerprint=validated_bundle.manifest.description_fingerprint,
        codebook_fingerprint=validated_bundle.manifest.codebook_fingerprint,
    )
    rows = [cache[index] for index in range(len(FIVE_CODES))]
    tangent = encode_token_rows(model, rows)['tangent']

    table = pl.read_parquet(exported_table)

    assert np.array_equal(table.select(COORDINATE_COLUMNS).to_numpy(), tangent.numpy())
    # The head caps the tangent at norm 2 before its exp map; the table keeps the capped vector
    assert (np.linalg.norm(tangent.numpy(), axis=1) <= 2.0 + 1e-6).all()

def test_the_provenance_names_the_table_and_the_checkpoint(
    exported_table, shared_checkpoint, validated_bundle, five_code_descriptions_parquet
):
    provenance = json.loads(provenance_path(exported_table).read_text())

    assert provenance['checkpoint'] == {
        'path': str(shared_checkpoint),
        'sha256': sha256_file(shared_checkpoint),
    }
    expected = contract_for_bundle(
        validated_bundle.manifest,
        encoder=shared_encoder_architecture(
            fusion='masked_mean', dimension=ARM_DIMENSION, backbone=MINILM
        ),
    )
    assert provenance['contract'] == expected.model_dump(mode='json')
    assert provenance['backbone'] == MINILM
    # The tiny backbone has no Hugging Face snapshot
    assert provenance['revision'] is None
    assert provenance['max_length'] == TOKEN_WINDOW
    assert provenance['descriptions'] == {
        'path': str(five_code_descriptions_parquet),
        'sha256': sha256_file(five_code_descriptions_parquet),
    }
    assert provenance['summaries'] is None
    assert (provenance['codes'], provenance['dimension']) == (5, ARM_DIMENSION)
    assert provenance['table_sha256'] == sha256_file(exported_table)
    assert provenance['matrix_fingerprint'] == table_fingerprint(pl.read_parquet(exported_table))
    assert set(provenance['library_versions']) == {'peft', 'polars', 'torch', 'transformers'}

def test_a_checkpoint_at_another_curvature_is_refused(
    tmp_path, shared_model, validated_bundle, five_code_token_config
):
    checkpoint = lightning_checkpoint(shared_model)
    checkpoint['hyper_parameters']['curvature'] = 2.0
    path = tmp_path / 'curved.ckpt'
    torch.save(checkpoint, path)
    output = tmp_path / 'table.parquet'

    with pytest.raises(ValueError, match='curvature 2'):
        export_code_table(path, validated_bundle, five_code_token_config, output)
    assert not output.exists()

def test_a_checkpoint_of_another_bundle_is_refused(
    tmp_path, shared_model, validated_bundle, five_code_token_config
):
    checkpoint = lightning_checkpoint(shared_model)
    checkpoint['stage3_supervision']['bundle_id'] = 'bundle-b'
    path = tmp_path / 'other-bundle.ckpt'
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match='supervision contract mismatch'):
        export_code_table(path, validated_bundle, five_code_token_config, tmp_path / 't.parquet')

def test_a_four_copy_checkpoint_is_refused_with_d2(
    tmp_path, shared_model, validated_bundle, five_code_token_config
):
    checkpoint = lightning_checkpoint(shared_model)
    # Contracts saved before Stage 6 carry no encoder record, and their hyperparameters predate
    # fusion and dimension
    del checkpoint['stage3_supervision']['encoder']
    for name in ('fusion', 'dimension'):
        del checkpoint['hyper_parameters'][name]
    path = tmp_path / 'four-copy.ckpt'
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match='D2'):
        export_code_table(path, validated_bundle, five_code_token_config, tmp_path / 't.parquet')

def test_descriptions_that_are_not_the_codebook_are_refused(
    tmp_path, shared_checkpoint, validated_bundle, five_code_token_config, text_descriptions_fixture
):
    other = tmp_path / 'other_descriptions.parquet'
    text_descriptions_fixture.with_columns(
        level=pl.lit(6), code=pl.col('code').str.replace('333333', '333334', literal=True)
    ).write_parquet(other)
    token_config = five_code_token_config.model_copy(update={'descriptions_parquet': str(other)})

    with pytest.raises(ValueError, match='codebook'):
        export_code_table(shared_checkpoint, validated_bundle, token_config, tmp_path / 't.parquet')
```

In `tests/unit/test_cli_commands.py`, add `from naics_embedder.utils.config import Config` after
`from naics_embedder.panels.selection_log import SelectionLog`. Then append at the end of the
file:

```python
# -------------------------------------------------------------------------------------------------
# Shared-encoder arms: export and the outcome read
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def default_config(monkeypatch):
    '''--config resolves to the default Config, whatever file it names.'''

    monkeypatch.setattr(Config, 'from_yaml', classmethod(lambda cls, path: Config()))

@pytest.mark.unit
def test_export_table_exports_under_the_configured_bundle_and_cache(
    monkeypatch, runner, tmp_path, default_config
):
    bundle = object()
    calls = []

    def fake_gate(cfg):
        calls.append(('gate', cfg.data_loader.streaming.max_length))
        return bundle

    def fake_export(checkpoint, chosen, token_config, output, *, device):
        calls.append(('export', checkpoint, chosen, token_config.max_length, output, device))
        return output

    monkeypatch.setattr(tools_cli, 'require_valid_supervision_bundle', fake_gate)
    monkeypatch.setattr(tools_cli, 'export_code_table', fake_export)
    monkeypatch.setattr(tools_cli, 'pick_device', lambda *_args: 'cpu')
    output = tmp_path / 'arm.parquet'

    result = runner.invoke(
        tools_cli.app,
        [
            'export-table', '--checkpoint', 'arm.ckpt', '--output',
            str(output), 'data_loader.streaming.max_length=64'
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [('gate', 64), ('export', 'arm.ckpt', bundle, 64, output, 'cpu')]
    # Rich folds long paths at the terminal's width (80 columns on CI), wherever it falls
    assert 'arm_provenance.json' in result.output.replace('\n', '')

@pytest.mark.unit
def test_export_table_refuses_legacy_containment(monkeypatch, runner, tmp_path, default_config):

    def never(*_args, **_kwargs):
        raise AssertionError('legacy containment reached the export')

    monkeypatch.setattr(tools_cli, 'export_code_table', never)

    result = runner.invoke(
        tools_cli.app,
        [
            'export-table', '--checkpoint', 'arm.ckpt', '--output',
            str(tmp_path / 'arm.parquet'), 'supervision.mode=legacy_containment'
        ],
    )

    assert result.exit_code == 1
    assert 'legacy containment has none' in ' '.join(result.output.split())

@pytest.mark.unit
def test_export_table_refuses_an_override_without_a_value(runner, tmp_path, default_config):
    result = runner.invoke(
        tools_cli.app,
        ['export-table', '--checkpoint', 'arm.ckpt', '--output',
         str(tmp_path / 'arm.parquet'), 'model.dimension'],
    )

    assert result.exit_code == 1
    assert 'key=value' in ' '.join(result.output.split())
```

The legacy test runs the real gate. Under `legacy_containment` it returns no bundle before it
reads a file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_export.py tests/unit/test_cli_commands.py -q`
Expected:
- `test_export.py` errors at collection with `ImportError: cannot import name 'export_code_table'`.
- Two CLI tests FAIL with `AttributeError`, because the `tools` module has no
  `require_valid_supervision_bundle` or `export_code_table` yet:
  - `test_export_table_exports_under_the_configured_bundle_and_cache`;
  - `test_export_table_refuses_legacy_containment`.
- `test_export_table_refuses_an_override_without_a_value` FAILS: the exit code is 2 ("No such
  command 'export-table'"), not 1.

- [ ] **Step 3: Implement the export**

In `src/naics_embedder/text_model/export.py`, replace the module docstring and imports:

```python
'''
Encoding an arm's codes and queries, and exporting its code table (spec 4.3).

``encode_token_rows`` runs token rows through an arm's model: a code's cached channels, or a
marked query. The HGCN feeder, the table export and the arm encoder all encode through it, so a
code embeds the same way wherever it is read.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Any, Dict, List, Mapping, Sequence

import torch

from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.fields import CHANNELS
from naics_embedder.utils.config import Config, TokenizationConfig
```

with:

```python
'''
Encoding an arm's codes and queries, and exporting its code table (spec 4.3).

``encode_token_rows`` runs token rows through an arm's model: a code's cached channels, or a
marked query. The HGCN feeder, the table export and the arm encoder all encode through it, so a
code embeds the same way wherever it is read.

``export_code_table`` writes Req 2's form of an arm: ``code``, ``index``, ``level`` and
``e0 … e{d-1}``, each code's capped tangent vector at the origin (R6), in the bundle's codebook
order. Its provenance ties the table to its checkpoint.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import polars as pl
import torch

from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.supervision.checkpoints import (
    CHECKPOINT_KEY,
    CheckpointContract,
    validate_supervision_contract,
)
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import SUMMARIES, tokenization_cache
from naics_embedder.text_model.fields import CHANNELS
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import Config, TokenizationConfig

logger = logging.getLogger(__name__)

TABLE_PREFIX = 'e'
COORDINATES = 'the capped tangent vector at the origin (spec R6); no time coordinate'
```

Then append at the end of the file:

```python
# -------------------------------------------------------------------------------------------------
# Loading an arm
# -------------------------------------------------------------------------------------------------

def require_unit_curvature(hyper_parameters: Mapping[str, Any]) -> None:
    '''
    Refuse a checkpoint trained at a curvature other than 1 (spec R8).

    The table's tangent coordinates and the scorer's ``lorentz`` distance both assume c = 1. An
    absent curvature is the model's default, 1.

    Raises:
        ValueError: If the saved curvature is not 1.
    '''

    curvature = float(hyper_parameters.get('curvature', 1.0))
    if curvature != 1.0:
        raise ValueError(
            f'the checkpoint was trained at curvature {curvature:g}; export and reads take c = 1 '
            'only (spec R8)'
        )

def load_arm_model(
    checkpoint_path: Union[str, Path],
    bundle: ValidatedSupervisionBundle,
    *,
    device: Union[str, torch.device] = 'cpu',
) -> Tuple[NAICSContrastiveModel, CheckpointContract]:
    '''
    Load an arm's checkpoint for export or a read, refusing it before any weight loads.

    The checkpoint's own hyperparameters rebuild its fusion and dimension, so its encoder record
    is never compared with a config (spec 4.4). Its supervision fields must match ``bundle``.

    Args:
        checkpoint_path: The arm's Lightning checkpoint.
        bundle: The configured supervision bundle.
        device: Where the model runs.

    Returns:
        The model, in eval mode on ``device``, and the checkpoint's saved contract.

    Raises:
        ValueError: If the curvature is not 1 (R8), the supervision contract is not the bundle's,
            or the checkpoint is of another encoder architecture (D2).
    '''

    # Lightning checkpoints carry pickled hyperparameters; they are trusted artifacts of this
    # project's own training runs
    raw = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=False)
    require_unit_curvature(raw.get('hyper_parameters', {}))
    contract = validate_supervision_contract(raw.get(CHECKPOINT_KEY), bundle.manifest)
    # on_load_checkpoint refuses another encoder architecture before the state dict loads (D2)
    model = NAICSContrastiveModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        supervision_manifest_path=str(bundle.manifest_path),
        supervision_bundle=bundle,
    )
    return model.to(device).eval(), contract

# -------------------------------------------------------------------------------------------------
# The code table
# -------------------------------------------------------------------------------------------------

def export_code_table(
    checkpoint_path: Union[str, Path],
    bundle: ValidatedSupervisionBundle,
    token_config: TokenizationConfig,
    output_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = 'cpu',
    batch_size: int = 32,
) -> Path:
    '''
    Export an arm's code table in Req 2's form, with its provenance beside it (spec 4.3).

    Every code goes through the checkpoint's model in eval mode, without gradient. The table
    holds ``code``, then ``index`` and ``level`` from the descriptions (Int64), then ``e0 …
    e{d-1}`` (float64): each code's capped tangent vector at the origin, in the bundle's codebook
    order. The provenance is ``<stem>_provenance.json``.

    Args:
        checkpoint_path: The arm's Lightning checkpoint.
        bundle: The configured supervision bundle.
        token_config: The token cache training read (``code_token_config``). Its
            ``descriptions_parquet`` is the arm's descriptions.
        output_path: The table's parquet path.
        device: Where the model runs.
        batch_size: Codes per forward pass.

    Returns:
        The table's path.

    Raises:
        ValueError: As ``load_arm_model``; if the descriptions' ``(index, code)`` rows are not
            the codebook's ``(code_id, code)`` rows; or if ``coordinate_matrix`` refuses the
            table, which is then not written.
    '''

    model, contract = load_arm_model(checkpoint_path, bundle, device=device)
    descriptions_path = Path(token_config.descriptions_parquet)
    descriptions = pl.read_parquet(descriptions_path).sort('index')
    codebook = pl.read_parquet(bundle.artifact_path('codebook')).sort('code_id')
    described = descriptions.select(pl.col('index').cast(pl.Int64), pl.col('code').cast(pl.Utf8))
    coded = codebook.select(pl.col('code_id').cast(pl.Int64), pl.col('code').cast(pl.Utf8))
    if described.rows() != coded.rows():
        raise ValueError(
            f"the descriptions' (index, code) rows are not bundle "
            f"{bundle.manifest.bundle_id}'s codebook (code_id, code) rows"
        )

    cache = tokenization_cache(
        token_config,
        description_fingerprint=bundle.manifest.description_fingerprint,
        codebook_fingerprint=bundle.manifest.codebook_fingerprint,
    )
    rows = [cache[index] for index in descriptions.get_column('index').to_list()]
    tangent = encode_token_rows(model, rows, batch_size=batch_size)['tangent']
    schema = {f'{TABLE_PREFIX}{index}': pl.Float64 for index in range(tangent.shape[1])}
    table = descriptions.select(
        pl.col('code').cast(pl.Utf8),
        pl.col('index').cast(pl.Int64),
        pl.col('level').cast(pl.Int64),
    ).hstack(pl.DataFrame(tangent.numpy(), schema=schema, orient='row'))
    # Fingerprinted before the write: coordinate_matrix refuses a table no panel could read
    fingerprint = table_fingerprint(table)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_path)
    checkpoint_path = Path(checkpoint_path)
    provenance: Dict[str, Any] = {
        'checkpoint': {
            'path': str(checkpoint_path),
            'sha256': sha256_file(checkpoint_path)
        },
        'contract': contract.model_dump(mode='json'),
        'backbone': contract.encoder.backbone,
        'revision': model.encoder.backbone_revision,
        'max_length': token_config.max_length,
        'descriptions': {
            'path': str(descriptions_path),
            'sha256': sha256_file(descriptions_path)
        },
        'summaries': SUMMARIES,
        'codes': table.height,
        'dimension': tangent.shape[1],
        'coordinates': COORDINATES,
        'table_sha256': sha256_file(output_path),
        'matrix_fingerprint': fingerprint,
        'library_versions': {
            name: version(name)
            for name in ('torch', 'transformers', 'peft', 'polars')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    provenance_path(output_path).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    logger.info(f'Code table ({table.height:,} codes, dimension {tangent.shape[1]}): {output_path}')
    return output_path
```

In `tests/fixtures/shared_encoder.py`, add before
`from naics_embedder.text_model.naics_model import NAICSContrastiveModel`:

```python
from naics_embedder.text_model.export import export_code_table
```

Then append at the end of the file:

```python
@pytest.fixture
def exported_table(tmp_path, shared_checkpoint, validated_bundle, five_code_token_config) -> Path:
    '''``shared_checkpoint``'s code table, exported on the CPU, with its provenance beside it.'''

    return export_code_table(
        shared_checkpoint, validated_bundle, five_code_token_config, tmp_path / 'arm_table.parquet'
    )
```

The fixture lands in this step, not in Step 1, because the fixture module's import of
`export_code_table` would break every test's collection before the function exists.

- [ ] **Step 4: Add `tools export-table`**

In `src/naics_embedder/cli/commands/tools.py`, module docstring, replace:

```python
    diagnostics: Report Req 6's structural diagnostics over every codebook code.
'''
```

with:

```python
    diagnostics: Report Req 6's structural diagnostics over every codebook code.
    export-table: Export an arm's code table in Req 2's form, with its provenance (Stage 6).
'''
```

Replace:

```python
from naics_embedder.panels.text_only import build_text_only_table
from naics_embedder.panels.text_only import provenance_path as text_only_provenance_path
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.metrics_tools import investigate_hierarchy, visualize_metrics
from naics_embedder.utils.config import (
    DecisionConfig,
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
    load_config,
)
from naics_embedder.utils.console import configure_logging
```

with:

```python
from naics_embedder.panels.text_only import build_text_only_table, provenance_path
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.text_model.export import code_token_config, export_code_table
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.metrics_tools import investigate_hierarchy, visualize_metrics
from naics_embedder.utils.config import (
    Config,
    DecisionConfig,
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
    load_config,
)
from naics_embedder.utils.console import configure_logging
from naics_embedder.utils.training import parse_config_overrides
from naics_embedder.utils.utilities import pick_device
from naics_embedder.utils.validation import ValidationError, require_valid_supervision_bundle
```

Replace:

```python
    console.print(f'Provenance: {text_only_provenance_path(path)}')
```

with:

```python
    console.print(f'Provenance: {provenance_path(path)}')
```

Then append at the end of the file:

```python
# -------------------------------------------------------------------------------------------------
# Shared-encoder arms: export and the outcome read (roadmap Stage 6)
# -------------------------------------------------------------------------------------------------

def _run_config(config_file: str, overrides: Optional[List[str]]) -> Config:
    '''
    The run's config as ``train`` resolves it: the YAML file, then ``key=value`` overrides.

    Raises:
        ValueError: If an override has no ``=``. ``train`` skips one with a warning, but a logged
            read must run on exactly the config it names (P18).
    '''

    cfg = Config.from_yaml(config_file)
    override_dict, invalid = parse_config_overrides(overrides)
    if invalid:
        raise ValueError(f'overrides take the form key=value, not {invalid}')
    return cfg.override(override_dict) if override_dict else cfg

def _run_bundle(cfg: Config) -> ValidatedSupervisionBundle:
    '''
    The configured supervision bundle, through ``train``'s gate.

    Raises:
        ValueError: Under legacy containment, which has no bundle (P27).
        ValidationError: As ``require_valid_supervision_bundle``.
    '''

    bundle = require_valid_supervision_bundle(cfg)
    if bundle is None:
        raise ValueError('export and reads need a supervision bundle; legacy containment has none')
    return bundle

@app.command('export-table')
def export_table(
    checkpoint: Annotated[
        str,
        typer.Option('--checkpoint', help="The arm's checkpoint: a training run's last.ckpt, say"),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='The table parquet; its provenance is written beside it'),
    ],
    config_file: Annotated[
        str,
        typer.Option('--config', help='Config YAML naming the bundle and the token cache'),
    ] = 'conf/config.yaml',
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(help="Config overrides, as train takes them (e.g., 'model.dimension=8')"),
    ] = None,
):
    '''
    Export an arm's code table in Req 2's form, with its provenance (roadmap Stage 6).

    Every code goes through the checkpoint's model in eval mode. The table holds ``code``,
    ``index``, ``level`` and ``e0 … e{d-1}``: each code's tangent vector at the origin, in the
    bundle's codebook order. The checkpoint's supervision contract must match the configured
    bundle. Its encoder record is its own, so a d = 8 checkpoint exports under a d = 16 config.

    Example:
        Export a run's last checkpoint::

            $ uv run naics-embedder tools export-table \\
                --checkpoint checkpoints/sadc_default/last.ckpt \\
                --output data/plan8/arm_table.parquet supervision.manifest_path=PATH
    '''

    configure_logging('tools_export_table.log')

    output_path = Path(output)
    try:
        # A bad output path fails before the whole codebook is encoded
        _require_writable(output_path)
        cfg = _run_config(config_file, overrides)
        table = export_code_table(
            checkpoint,
            _run_bundle(cfg),
            code_token_config(cfg),
            output_path,
            device=pick_device('auto'),
        )
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f'[bold red]Export failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(f'Code table: {table}')
    console.print(f'Provenance: {provenance_path(table)}')
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_export.py tests/unit/test_cli_commands.py -q`
Expected: all pass.

Run: `uv run naics-embedder tools export-table --help`
Expected: the help lists `--checkpoint`, `--output` and `--config`, and `OVERRIDES`.

- [ ] **Step 6: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/export.py src/naics_embedder/cli/commands/tools.py tests/fixtures/shared_encoder.py tests/unit/test_export.py tests/unit/test_cli_commands.py`
Run: `uv run pytest -n auto -q`
Expected: all pass, 1 skipped.

```bash
git add src/naics_embedder/text_model/export.py src/naics_embedder/cli/commands/tools.py tests/fixtures/shared_encoder.py tests/unit/test_export.py tests/unit/test_cli_commands.py
git commit -m "feat(text_model): export the code table in Req 2's form; tools export-table"
```

### Task 11: The arm encoder

`ArmEncoder` gives the outcome panel an arm in its `QueryCodeEncoder` form (spec 4.3):
- A query is marked `query:` and goes through the checkpoint's model.
- A code is decoded from the exported table.
- Both pass through one float64 exp map at the origin. The code vectors a read decodes against
  are therefore a fixed function of the table, which its `matrix_fingerprint` names.

The §6 "same encoder" and "live read" criteria are tested here.

**Files:**
- Create: `src/naics_embedder/text_model/arm_encoder.py`
- Test: `tests/unit/test_arm_encoder.py` (create)

**Interfaces:**
- Consumes:
  - `QUERY` and `tokenize_field` (Task 2);
  - `stack_text_inputs` (Task 3, tests only);
  - `HyperbolicHead` and its `distance` (Task 5);
  - `encode_token_rows` (Task 9);
  - `load_arm_model` and the `exported_table` fixture (Task 10);
  - the existing `panels.regressor.coordinate_matrix` and
    `panels.text_only.matrix_fingerprint` / `provenance_path`.
- Produces, in `naics_embedder.text_model.arm_encoder`:
  - `exp_map_origin(tangent: torch.Tensor) -> torch.Tensor`. It maps (N, d) tangents at the
    origin to (time, space) rows (N, d + 1) on the curvature −1 hyperboloid, in float64 on the
    CPU, from any device.
  - `ArmEncoder(model, tokenizer, *, max_length: int, table: pl.DataFrame,
    checkpoint_sha256: str, batch_size: int = 32)`, with these attributes:
    - `model`, `tokenizer` and `max_length`;
    - `distance`, the head's (`'lorentz'`);
    - `table_fingerprint`, the table's `matrix_fingerprint`;
    - `checkpoint_sha256`.
  - `ArmEncoder.from_files(checkpoint_path, table_path, bundle, token_config, *, device='cpu',
    batch_size=32) -> ArmEncoder`.
    - It checks the table's provenance first, then loads through `load_arm_model`.
    - It raises `ValueError` matching "another checkpoint" or "not the table its provenance
      names", or as `load_arm_model` does.
  - `encode_queries(texts: Sequence[str]) -> torch.Tensor` (Q, d + 1) and
    `encode_codes(codes: Sequence[str]) -> torch.Tensor` (C, d + 1), both float64 on the CPU.
    `encode_codes` raises `ValueError` matching "has no row for" on an unknown code.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_arm_encoder.py`:

```python
'''
The arm encoder: queries through the checkpoint's model, codes from its exported table
(spec 4.3).
'''

import json

import polars as pl
import pytest
import torch

from naics_embedder.panels.decoding import lorentz_distances
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.arm_encoder import ArmEncoder, exp_map_origin
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.export import encode_token_rows
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.text_model.hyperbolic import HyperbolicHead
from tests.fixtures.shared_encoder import (
    ARM_DIMENSION,
    FIVE_CODES,
    TOKEN_WINDOW,
    lightning_checkpoint,
)

pytestmark = pytest.mark.unit

QUERIES = ['Edamame farming', 'Lignite mining']

@pytest.fixture
def arm(shared_checkpoint, exported_table, validated_bundle, five_code_token_config) -> ArmEncoder:
    return ArmEncoder.from_files(
        shared_checkpoint, exported_table, validated_bundle, five_code_token_config
    )

def _table_tangent(table_path) -> torch.Tensor:
    table = pl.read_parquet(table_path)
    return torch.tensor(table.select(pl.exclude('code', 'index', 'level')).to_numpy())

# -------------------------------------------------------------------------------------------------
# The exp map at the origin
# -------------------------------------------------------------------------------------------------

def test_the_exp_map_lands_on_the_hyperboloid_as_the_heads_does():
    tangent = torch.randn(6, ARM_DIMENSION) * 0.2
    tangent[0] = 0.0
    # Below the head's cap, so the head maps these tangents unchanged
    assert (torch.linalg.vector_norm(tangent, dim=1) < 2.0).all()

    points = exp_map_origin(tangent)

    assert points.dtype == torch.float64
    assert points.shape == (6, ARM_DIMENSION + 1)
    lorentz_norm = -points[:, 0]**2 + (points[:, 1:]**2).sum(dim=1)
    assert torch.allclose(lorentz_norm, torch.full((6, ), -1.0, dtype=torch.float64), atol=1e-12)
    origin = torch.zeros(ARM_DIMENSION + 1, dtype=torch.float64)
    origin[0] = 1.0
    assert torch.equal(points[0], origin)
    _, head_points = HyperbolicHead()(tangent)
    assert torch.allclose(points, head_points.to(torch.float64), atol=1e-5)

# -------------------------------------------------------------------------------------------------
# Queries and codes
# -------------------------------------------------------------------------------------------------

def test_a_query_embeds_through_the_same_forward_as_a_code(arm):
    '''Spec §6: encode_queries([T]) is the float64 exp map of the forward's {'query': [T]}.'''

    tokens = tokenize_field(arm.tokenizer, QUERY, 'Edamame farming', TOKEN_WINDOW)
    with torch.no_grad():
        output = arm.model(stack_text_inputs([{QUERY: tokens}], fields=(QUERY, )))

    assert torch.equal(arm.encode_queries(['Edamame farming']), exp_map_origin(output['tangent']))

def test_codes_decode_from_the_table_in_the_order_asked(arm, exported_table):
    tangent = _table_tangent(exported_table)

    decoded = arm.encode_codes(['222222', '111111'])

    assert torch.equal(decoded, exp_map_origin(tangent[[3, 0]]))

def test_table_decoded_distances_match_the_live_forward(
    arm, five_code_token_config, validated_bundle
):
    '''Spec §6: decoding from the table matches the live forward within float32 tolerance.'''

    cache = tokenization_cache(
        five_code_token_config,
        description_fingerprint=validated_bundle.manifest.description_fingerprint,
        codebook_fingerprint=validated_bundle.manifest.codebook_fingerprint,
    )
    rows = [cache[index] for index in range(len(FIVE_CODES))]
    # The head's own float32 points, from the forward the export ran
    live = encode_token_rows(arm.model, rows)['embedding']
    queries = arm.encode_queries(QUERIES)

    decoded = arm.encode_codes(list(FIVE_CODES))

    assert torch.allclose(
        lorentz_distances(queries, decoded), lorentz_distances(queries, live), atol=1e-4
    )

def test_an_unknown_code_is_refused(arm):
    with pytest.raises(ValueError, match='has no row for'):
        arm.encode_codes(['111111', '999999'])

def test_the_distance_is_the_heads(arm):
    assert arm.distance == arm.model.encoder.head.distance == 'lorentz'

def test_the_logged_names_are_the_tables_and_the_checkpoints(
    arm, exported_table, shared_checkpoint
):
    provenance = json.loads(provenance_path(exported_table).read_text())

    assert arm.table_fingerprint == provenance['matrix_fingerprint']
    # The name tools regressor-panel logs the same table by
    assert arm.table_fingerprint == table_fingerprint(pl.read_parquet(exported_table))
    assert arm.checkpoint_sha256 == sha256_file(shared_checkpoint)

# -------------------------------------------------------------------------------------------------
# Refusals
# -------------------------------------------------------------------------------------------------

def test_a_table_of_another_checkpoint_is_refused(
    tmp_path, shared_model, exported_table, validated_bundle, five_code_token_config
):
    checkpoint = lightning_checkpoint(shared_model)
    # The same weights in a file with other bytes, as another run's would be
    checkpoint['note'] = 'another run'
    other = tmp_path / 'other.ckpt'
    torch.save(checkpoint, other)

    with pytest.raises(ValueError, match='another checkpoint'):
        ArmEncoder.from_files(other, exported_table, validated_bundle, five_code_token_config)

def test_an_edited_table_is_refused(
    exported_table, shared_checkpoint, validated_bundle, five_code_token_config
):
    pl.read_parquet(exported_table).with_columns(pl.col('e0') * 2).write_parquet(exported_table)

    with pytest.raises(ValueError, match='not the table its provenance names'):
        ArmEncoder.from_files(
            shared_checkpoint, exported_table, validated_bundle, five_code_token_config
        )

def test_a_checkpoint_at_another_curvature_is_refused(
    tmp_path, shared_model, exported_table, validated_bundle, five_code_token_config
):
    '''Spec §6: curvature other than 1 is refused (R8).'''

    checkpoint = lightning_checkpoint(shared_model)
    checkpoint['hyper_parameters']['curvature'] = 2.0
    curved = tmp_path / 'curved.ckpt'
    torch.save(checkpoint, curved)
    # The provenance names the curved checkpoint, so its checks pass and R8's guard is what fires
    path = provenance_path(exported_table)
    provenance = json.loads(path.read_text())
    provenance['checkpoint']['sha256'] = sha256_file(curved)
    path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match='curvature 2'):
        ArmEncoder.from_files(curved, exported_table, validated_bundle, five_code_token_config)

# -------------------------------------------------------------------------------------------------
# Devices
# -------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs an MPS device')
def test_queries_and_codes_on_mps_come_back_float64_on_the_cpu(
    arm, shared_checkpoint, exported_table, validated_bundle, five_code_token_config
):
    '''Spec §6: on MPS, encode_queries and encode_codes return float64 CPU tensors.'''

    on_mps = ArmEncoder.from_files(
        shared_checkpoint, exported_table, validated_bundle, five_code_token_config, device='mps'
    )

    queries = on_mps.encode_queries(QUERIES)
    codes = on_mps.encode_codes(list(FIVE_CODES))

    for vectors in (queries, codes):
        assert vectors.dtype == torch.float64
        assert vectors.device.type == 'cpu'
    assert torch.allclose(queries, arm.encode_queries(QUERIES), atol=1e-4)
    assert torch.equal(codes, arm.encode_codes(list(FIVE_CODES)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_arm_encoder.py -q`
Expected: a collection error,
`ModuleNotFoundError: No module named 'naics_embedder.text_model.arm_encoder'`.

- [ ] **Step 3: Create `src/naics_embedder/text_model/arm_encoder.py`**

```python
'''
An arm as the outcome panel reads it: queries through its checkpoint, codes from its table
(spec 4.3).

``ArmEncoder`` implements ``QueryCodeEncoder`` (``panels/outcome.py``):

- A query is marked ``query:`` and goes through the checkpoint's model.
- A code's vector is decoded from the table ``tools export-table`` wrote from the checkpoint.

Both pass through one float64 exp map at the origin. The code vectors a read decodes against are
therefore a fixed function of the table, and the ``matrix_fingerprint`` the read logs names them.
Checkpoint, table, encoder and distance are the pieces of Stage 4's ``SeedArtifacts``
(``decision/sweep.py``).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
from pathlib import Path
from typing import Any, Sequence, Union

import polars as pl
import torch
from transformers import AutoTokenizer

from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.text_model.export import encode_token_rows, load_arm_model
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.utils.config import TokenizationConfig

# -------------------------------------------------------------------------------------------------
# The exp map at the origin
# -------------------------------------------------------------------------------------------------

def exp_map_origin(tangent: torch.Tensor) -> torch.Tensor:
    '''
    The exponential map at the origin of the curvature -1 hyperboloid (c = 1), in float64.

    It is ``HyperbolicHead``'s map, computed in float64 on the CPU whatever the tangent's device
    and dtype.

    Args:
        tangent: Tangent vectors at the origin (N, d).

    Returns:
        (time, space) rows (N, d + 1), float64 on the CPU.
    '''

    # .cpu() before the cast: casting an MPS tensor to float64 raises
    tangent = tangent.cpu().to(torch.float64)
    norm = torch.linalg.vector_norm(tangent, dim=1, keepdim=True).clamp(min=1e-8)
    return torch.cat([torch.cosh(norm), torch.sinh(norm) / norm * tangent], dim=1)

# -------------------------------------------------------------------------------------------------
# The arm encoder
# -------------------------------------------------------------------------------------------------

class ArmEncoder:
    '''
    ``QueryCodeEncoder`` for one arm: its checkpoint's model and the table exported from it.

    Args:
        model: The arm's model, in eval mode (``load_arm_model``).
        tokenizer: The token cache's tokenizer.
        max_length: The token cache's window.
        table: The arm's exported table.
        checkpoint_sha256: The checkpoint file's SHA-256.
        batch_size: Queries per forward pass.

    Attributes:
        distance: The head's distance, ``'lorentz'`` for the hyperbolic head.
        table_fingerprint: The table's ``matrix_fingerprint``, which a read logs as ``table``.
        checkpoint_sha256: The checkpoint's SHA-256, which a read logs as ``checkpoint``.
    '''

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        max_length: int,
        table: pl.DataFrame,
        checkpoint_sha256: str,
        batch_size: int = 32,
    ):
        codes, matrix = coordinate_matrix(table)
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.distance = model.encoder.head.distance
        self.table_fingerprint = matrix_fingerprint(codes, matrix)
        self.checkpoint_sha256 = checkpoint_sha256
        self._rows = {code: row for row, code in enumerate(codes)}
        self._tangent = torch.tensor(matrix, dtype=torch.float64)

    @classmethod
    def from_files(
        cls,
        checkpoint_path: Union[str, Path],
        table_path: Union[str, Path],
        bundle: ValidatedSupervisionBundle,
        token_config: TokenizationConfig,
        *,
        device: Union[str, torch.device] = 'cpu',
        batch_size: int = 32,
    ) -> 'ArmEncoder':
        '''
        The arm of a checkpoint and the table exported from it.

        The table's provenance is checked before the model loads.

        Args:
            checkpoint_path: The arm's Lightning checkpoint.
            table_path: The table ``tools export-table`` wrote from it.
            bundle: The configured supervision bundle.
            token_config: The token cache training read (``code_token_config``): its tokenizer
                and window tokenize the queries.
            device: Where the model runs.
            batch_size: Queries per forward pass.

        Raises:
            ValueError: If the table's provenance names another checkpoint, or the table file is
                not the one it names; or as ``load_arm_model``: a curvature other than 1 (R8),
                another supervision contract, or another encoder architecture (D2).
            FileNotFoundError: If the table or its provenance is missing.
        '''

        checkpoint_path, table_path = Path(checkpoint_path), Path(table_path)
        provenance = json.loads(provenance_path(table_path).read_text())
        checkpoint_sha256 = sha256_file(checkpoint_path)
        named = provenance['checkpoint']['sha256']
        if named != checkpoint_sha256:
            raise ValueError(
                f'the table was exported from another checkpoint: its provenance names {named}, '
                f'and {checkpoint_path} is {checkpoint_sha256}'
            )
        if provenance['table_sha256'] != sha256_file(table_path):
            raise ValueError(f'{table_path} is not the table its provenance names')
        model, _ = load_arm_model(checkpoint_path, bundle, device=device)
        return cls(
            model,
            AutoTokenizer.from_pretrained(token_config.tokenizer_name),
            max_length=token_config.max_length,
            table=pl.read_parquet(table_path),
            checkpoint_sha256=checkpoint_sha256,
            batch_size=batch_size,
        )

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        '''Marked ``query:`` texts through the model, then the exp map: (Q, d + 1), float64.'''

        tokens = [tokenize_field(self.tokenizer, QUERY, text, self.max_length) for text in texts]
        rows = [{QUERY: row} for row in tokens]
        tangent = encode_token_rows(
            self.model, rows, fields=(QUERY, ), batch_size=self.batch_size
        )['tangent']
        return exp_map_origin(tangent)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        '''
        The codes' table rows through the exp map: (C, d + 1), float64.

        Raises:
            ValueError: If a code has no row in the table.
        '''

        unknown = sorted(set(codes) - set(self._rows))
        if unknown:
            raise ValueError(f'the table has no row for {unknown[:5]} ({len(unknown)} codes)')
        return exp_map_origin(self._tangent[[self._rows[code] for code in codes]])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_arm_encoder.py -v`
Expected:
- Every test passes.
- `test_queries_and_codes_on_mps_come_back_float64_on_the_cpu` runs on an MPS machine, such
  as this M4 Max. It is skipped elsewhere: "needs an MPS device".

- [ ] **Step 5: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/arm_encoder.py tests/unit/test_arm_encoder.py`
Run: `uv run pytest -n auto -q`
Expected: all pass.
- Locally, MPS is available, so the skips are the baseline's one (CUDA).
- On a machine without MPS, this task's MPS test adds one skip, as the existing MPS tests do.

```bash
git add src/naics_embedder/text_model/arm_encoder.py tests/unit/test_arm_encoder.py
git commit -m "feat(text_model): the arm encoder decodes codes from its exported table"
```

### Task 12: The outcome read and `tools outcome-panel`

The first live read scores an arm on the outcome panel's validation split under its own distance.
It logs the table the codes were decoded from and the checkpoint the queries went through
(spec 4.3, P23). This is the read the roadmap's Stage 6 entry asks for. A logged `table` now
names the code vectors the encoder decodes against, which Stage 4's sweep could not tie to the
decoding. There is no test-split path; Stage 12 opens that split.

**Files:**
- Modify: `src/naics_embedder/text_model/arm_encoder.py` (Task 11's module)
- Modify: `src/naics_embedder/cli/commands/tools.py`: the module docstring, the imports and a
  new command at the end
- Test: `tests/unit/test_arm_encoder.py`, `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes:
  - `ArmEncoder` (Task 11);
  - `_run_config`, `_run_bundle`, `code_token_config` and `_require_writable` (Tasks 9, 10, and
    the existing helper);
  - the existing `OutcomePanel.from_bundle(bundle, log_path)` and
    `OutcomePanel.score(encoder, split, purpose, distance, detail)`.
- Produces:
  - `read_outcome_validation(encoder: ArmEncoder, panel: OutcomePanel, purpose: str) ->
    DecodingResult`. It logs one validation read with
    `detail={'encoder': 'ArmEncoder', 'distance': <the head's>, 'table': <matrix_fingerprint>,
    'checkpoint': <sha256>}`.
  - `tools outcome-panel --checkpoint <ckpt> --table <table> --purpose <why> [--config <yaml>]
    [--log <jsonl>] [--output <json>] [key=value ...]`.
    - `--purpose` is required.
    - `--output` writes `{'fingerprint', 'table', 'checkpoint', 'summary'}`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_arm_encoder.py`, replace:

```python
from naics_embedder.panels.decoding import lorentz_distances
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.arm_encoder import ArmEncoder, exp_map_origin
```

with:

```python
from naics_embedder.panels.decoding import lorentz_distances
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.arm_encoder import (
    ArmEncoder,
    exp_map_origin,
    read_outcome_validation,
)
```

Then append at the end of the file:

```python
# -------------------------------------------------------------------------------------------------
# The outcome read
# -------------------------------------------------------------------------------------------------

def test_a_validation_read_logs_the_table_it_decodes_against(
    tmp_path, arm, exported_table, validated_bundle
):
    '''Spec §6: a read on a fixture panel logs table equal to the table's matrix_fingerprint.'''

    log_path = tmp_path / 'selection_log.jsonl'
    panel = OutcomePanel.from_bundle(validated_bundle, log_path)

    result = read_outcome_validation(arm, panel, 'plan 8 fixture read')

    # The five-code bundle's one validation entry: 'Edamame farming', for 111111
    [record] = SelectionLog(log_path).records()
    assert (record['event'], record['split'], record['n_queries']) == ('read', 'validation', 1)
    assert record['purpose'] == 'plan 8 fixture read'
    assert record['detail'] == {
        'encoder': 'ArmEncoder',
        'distance': 'lorentz',
        'table': table_fingerprint(pl.read_parquet(exported_table)),
        'checkpoint': arm.checkpoint_sha256,
    }
    assert (result.summary['n_queries'], result.summary['n_candidates']) == (1, 5)
```

In `tests/unit/test_cli_commands.py`, add `import torch` after `import pytest`. Then append at
the end of the file:

```python
class FakeArm:
    '''Stands in for ArmEncoder: points on the hyperboloid's spatial axes, and the logged names.'''

    distance = 'lorentz'
    table_fingerprint = 't' * 64
    checkpoint_sha256 = 'c' * 64

    def encode_queries(self, texts):
        return torch.zeros(len(texts), 3, dtype=torch.float64)

    def encode_codes(self, codes):
        return torch.tensor([[0.0, float(row), 0.0] for row in range(len(codes))],
                            dtype=torch.float64)

@pytest.mark.unit
def test_outcome_panel_reads_validation_and_logs_the_table(
    monkeypatch, runner, tmp_path, default_config, validated_bundle
):
    calls = []

    def fake_from_files(checkpoint, table, bundle, token_config, *, device):
        calls.append((checkpoint, table, bundle, token_config.max_length, device))
        return FakeArm()

    monkeypatch.setattr(tools_cli, 'require_valid_supervision_bundle', lambda cfg: validated_bundle)
    monkeypatch.setattr(tools_cli.ArmEncoder, 'from_files', staticmethod(fake_from_files))
    monkeypatch.setattr(tools_cli, 'pick_device', lambda *_args: 'cpu')
    log = tmp_path / 'selection_log.jsonl'
    output = tmp_path / 'read.json'

    result = runner.invoke(
        tools_cli.app,
        [
            'outcome-panel', '--checkpoint', 'arm.ckpt', '--table', 'arm.parquet', '--purpose',
            'first live read', '--log',
            str(log), '--output',
            str(output)
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [('arm.ckpt', 'arm.parquet', validated_bundle, 128, 'cpu')]
    [record] = SelectionLog(log).records()
    assert (record['split'], record['purpose']) == ('validation', 'first live read')
    assert record['detail'] == {
        'encoder': 'FakeArm',
        'distance': 'lorentz',
        'table': 't' * 64,
        'checkpoint': 'c' * 64,
    }
    payload = json.loads(output.read_text())
    assert (payload['table'], payload['checkpoint']) == ('t' * 64, 'c' * 64)
    assert payload['summary']['n_queries'] == 1

@pytest.mark.unit
def test_outcome_panel_needs_a_purpose(runner, tmp_path, default_config):
    log = tmp_path / 'selection_log.jsonl'

    result = runner.invoke(
        tools_cli.app,
        ['outcome-panel', '--checkpoint', 'arm.ckpt', '--table', 'arm.parquet', '--log',
         str(log)],
    )

    # Click's usage error: the option is required
    assert result.exit_code == 2
    assert "Missing option '--purpose'" in result.output
    assert SelectionLog(log).records() == []
```

`fake_from_files` is installed as a `staticmethod`, so `ArmEncoder.from_files(...)` calls it with
exactly the command's arguments. The read itself is real: the five-code bundle's panel, its
selection log and its decoding. Only the encoder is a stand-in.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_arm_encoder.py tests/unit/test_cli_commands.py -q`
Expected:
- `test_arm_encoder.py` errors at collection with
  `ImportError: cannot import name 'read_outcome_validation'`.
- `test_outcome_panel_reads_validation_and_logs_the_table` FAILS with `AttributeError`: the
  `tools` module has no `ArmEncoder` yet.
- `test_outcome_panel_needs_a_purpose` FAILS: the output says "No such command
  'outcome-panel'", not "Missing option '--purpose'".

- [ ] **Step 3: Implement the read**

In `src/naics_embedder/text_model/arm_encoder.py`, replace:

```python
from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
```

with:

```python
from naics_embedder.panels.decoding import DecodingResult
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle, sha256_file
from naics_embedder.supervision.schema import IndexRole
```

Then append at the end of the file:

```python
# -------------------------------------------------------------------------------------------------
# The outcome read
# -------------------------------------------------------------------------------------------------

def read_outcome_validation(
    encoder: ArmEncoder,
    panel: OutcomePanel,
    purpose: str,
) -> DecodingResult:
    '''
    Score the arm on the outcome panel's validation split under its own distance.

    The read is logged. Its detail names the table the codes were decoded from (``table``, the
    key Stage 4's sweep logs) and the checkpoint the queries went through (``checkpoint``). There
    is no test-split path here: Stage 12 opens that split.

    Args:
        encoder: The arm.
        panel: The outcome panel of the arm's bundle.
        purpose: Why the read happens; the selection log records it.

    Returns:
        The decoding scores.
    '''

    return panel.score(
        encoder,
        IndexRole.VALIDATION,
        purpose,
        distance=encoder.distance,
        detail={
            'table': encoder.table_fingerprint,
            'checkpoint': encoder.checkpoint_sha256
        },
    )
```

- [ ] **Step 4: Add `tools outcome-panel`**

In `src/naics_embedder/cli/commands/tools.py`, module docstring, replace:

```python
    export-table: Export an arm's code table in Req 2's form, with its provenance (Stage 6).
'''
```

with:

```python
    export-table: Export an arm's code table in Req 2's form, with its provenance (Stage 6).
    outcome-panel: Score an arm on the outcome panel's validation split (Stage 6).
'''
```

Replace:

```python
from naics_embedder.text_model.export import code_token_config, export_code_table
```

with:

```python
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.export import code_token_config, export_code_table
```

Then append at the end of the file:

```python
@app.command('outcome-panel')
def outcome_panel(
    checkpoint: Annotated[
        str,
        typer.Option('--checkpoint', help='The arm checkpoint the table was exported from'),
    ],
    table: Annotated[
        str,
        typer.Option('--table', help='The table tools export-table wrote from the checkpoint'),
    ],
    purpose: Annotated[
        str,
        typer.Option('--purpose', help='Why this read happens; recorded in the selection log'),
    ],
    config_file: Annotated[
        str,
        typer.Option('--config', help='Config YAML naming the bundle and the token cache'),
    ] = 'conf/config.yaml',
    log: Annotated[
        Optional[str],
        typer.Option('--log', help='Selection log (default: the outcome-panel config)'),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option('--output', help='Also write the summary as JSON to this path'),
    ] = None,
    overrides: Annotated[
        Optional[List[str]],
        typer.Argument(help="Config overrides, as train takes them (e.g., 'model.dimension=8')"),
    ] = None,
):
    '''
    Score an arm on the outcome panel's validation split under its own distance (Stage 6).

    Queries go through the checkpoint's model; codes are decoded from the table exported from it.
    The read is logged with the table's ``matrix_fingerprint`` and the checkpoint's SHA-256. The
    test split stays sealed: this command never opens it.

    Example:
        Read the validation split for an exported table::

            $ uv run naics-embedder tools outcome-panel \\
                --checkpoint checkpoints/sadc_default/last.ckpt \\
                --table data/plan8/arm_table.parquet --purpose 'Stage 6 Exit reading' \\
                supervision.manifest_path=PATH
    '''

    configure_logging('tools_outcome_panel.log')

    panel_cfg = load_config(OutcomePanelConfig, 'data/outcome_panel.yaml')
    try:
        # A bad output path fails before the read is logged
        if output:
            _require_writable(Path(output))
        cfg = _run_config(config_file, overrides)
        bundle = _run_bundle(cfg)
        encoder = ArmEncoder.from_files(
            checkpoint, table, bundle, code_token_config(cfg), device=pick_device('auto')
        )
        panel = OutcomePanel.from_bundle(bundle, log or panel_cfg.selection_log)
        result = read_outcome_validation(encoder, panel, purpose)
    except (OSError, ValueError, ValidationError) as exc:
        console.print(f'[bold red]Outcome panel failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print('\n[bold cyan]Outcome panel: arm, validation split[/bold cyan]\n')
    for key, value in result.summary.items():
        formatted = f'{value:.4f}' if isinstance(value, float) else str(value)
        console.print(f'  • {key}: {formatted}')
    console.print(f'\nRead logged to {panel.log.path} (table {encoder.table_fingerprint})\n')

    if output:
        path = Path(output)
        payload = {
            'fingerprint': panel.fingerprint,
            'table': encoder.table_fingerprint,
            'checkpoint': encoder.checkpoint_sha256,
            'summary': result.summary,
        }
        path.write_text(json.dumps(payload, indent=2) + '\n')
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_arm_encoder.py tests/unit/test_cli_commands.py -q`
Expected: all pass.

Run: `uv run naics-embedder tools outcome-panel --help`
Expected: `--checkpoint`, `--table` and `--purpose` are marked required, and there is no
`--split` option.

- [ ] **Step 6: Format, run the suite, commit**

Run: `./scripts/format_code.sh src/naics_embedder/text_model/arm_encoder.py src/naics_embedder/cli/commands/tools.py tests/unit/test_arm_encoder.py tests/unit/test_cli_commands.py`
Run: `uv run pytest -n auto -q`
Expected: all pass; the skips are as in Task 11.

```bash
git add src/naics_embedder/text_model/arm_encoder.py src/naics_embedder/cli/commands/tools.py tests/unit/test_arm_encoder.py tests/unit/test_cli_commands.py
git commit -m "feat(tools): outcome-panel reads an arm's validation split and logs its table"
```

### Task 13: Documentation

The architecture text still describes four encoder copies and MoE fusion. This task rewrites it
for the shared encoder, documents the two new commands, and adds the API pages for the new
modules (spec §8). This task may go to the docs-writer agent. Its acceptance checks (Step 9) are
the gate.

Every "replace" text below occurs exactly once in its file. New prose wraps at 100 columns, except
in `README.md`, which disables the line-length rule and keeps its paragraphs on single lines.

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/index.md`, `docs/overview.md`,
  `docs/text_training.md`, `docs/usage.md`, `tests/README.md`
- Modify: `docs/api/encoder.md` (Task 7's page), `docs/api/moe.md`, `docs/.nav.yml`
- Create: `docs/api/export.md`

**Interfaces:**
- Consumes: the module and command names of Tasks 2–12: `fields`, `fusion`, `shared_encoder`,
  `export`, `arm_encoder`, `tools export-table` and `tools outcome-panel`.
- Produces: documentation only.

- [ ] **Step 1: `CLAUDE.md`**

Replace:

```markdown
combines multi-channel text encoding, Mixture-of-Experts fusion, hyperbolic contrastive learning,
and hyperbolic graph refinement to create geometry-aware embeddings aligned with the hierarchical
NAICS taxonomy.
```

with:

```markdown
combines a shared text encoder over marked text fields, masked fusion, hyperbolic contrastive
learning, and hyperbolic graph refinement to create geometry-aware embeddings aligned with the
hierarchical NAICS taxonomy.
```

Replace:

```markdown
1. **Multi-Channel Text Encoding** (`text_model/encoder.py`)
   - Independent LoRA-adapted transformer encoders for title, description, examples, exclusions
   - Base model: sentence-transformers/all-MiniLM-L6-v2
   - Produces 4 Euclidean embeddings per NAICS code

2. **Mixture-of-Experts Fusion** (`text_model/moe.py`)
   - Top-2 gating with load-balancing loss
   - Adaptively fuses the 4 channel embeddings into a single Euclidean embedding

3. **Hyperbolic Contrastive Learning** (`text_model/naics_model.py`, `text_model/loss.py`)
   - Projects embeddings into Lorentz-model hyperbolic space
```

with:

```markdown
1. **Shared Text Encoding** (`text_model/shared_encoder.py`, `text_model/fields.py`)
   - One LoRA-adapted backbone (sentence-transformers/all-MiniLM-L6-v2) reads every field:
     title, description, excluded, examples, and a query
   - Each present text is marked with its field (`'title: …'`); absent channels never enter
     the backbone
   - Mean-pools one vector per present channel

2. **Fusion and Projection** (`text_model/fusion.py`)
   - `model.fusion`: masked mean (default), attention pooling, or `moe`, an ablation that
     routes the masked mean through `text_model/moe.py`'s experts
   - Exactly one `Linear(384 → d)` maps the fused vector to `model.dimension`, d in {8, 16, 32}

3. **Hyperbolic Contrastive Learning** (`text_model/naics_model.py`, `text_model/loss.py`)
   - A parameter-free head caps the tangent at norm 2 and maps it onto the Lorentz hyperboloid
     (c = 1)
```

Replace:

```markdown
**Final Output:** High-fidelity Lorentz-model hyperbolic embeddings suitable for hierarchical
search, clustering, and downstream ML tasks.
```

with:

```markdown
**Final Output:** High-fidelity Lorentz-model hyperbolic embeddings suitable for hierarchical
search, clustering, and downstream ML tasks. `tools export-table` writes a checkpoint's
2,125-code table in Req 2's form: tangent coordinates at the origin, `e0 … e{d-1}`.
```

Replace:

```markdown
│   │   ├── encoder.py        # Multi-channel LoRA encoder
│   │   ├── moe.py            # Mixture-of-Experts (with torch.compile)
```

with:

```markdown
│   │   ├── fields.py         # The five fields and their markers ('title: …')
│   │   ├── shared_encoder.py # SharedEncoder: one backbone, fusion, one Linear(384 → d) ⭐
│   │   ├── fusion.py         # Masked mean, attention pooling, the MoE ablation
│   │   ├── export.py         # Encode token rows; export the code table (tools export-table)
│   │   ├── arm_encoder.py    # ArmEncoder: queries through the checkpoint, codes from its table
│   │   ├── moe.py            # Mixture-of-Experts, read only under fusion: moe
```

Replace:

```markdown
│   │   │   ├── curriculum.py    # Hard negative mining, router sampling
```

with:

```markdown
│   │   │   ├── curriculum.py    # Hard negative mining; router sampling under moe
```

Replace:

```markdown
### 2. Multi-Channel Architecture
```

with:

```markdown
### 2. Text Fields and the Shared Encoder
```

Replace:

```markdown
Each channel is encoded by a **separate LoRA-adapted transformer** (via PEFT library) to capture
channel-specific semantics.

### 3. Mixture-of-Experts (MoE)

- **Top-k Gating:** Routes each input to the top-k most relevant experts (k=2)
- **Load Balancing:** Auxiliary loss ensures even expert utilization
- **Implementation:** `text_model/moe.py` (with torch.compile for gating ops)
- **Purpose:** Learns adaptive fusion of the 4 channel embeddings
```

with:

```markdown
Every channel goes through **one shared LoRA-adapted backbone** (via the PEFT library), marked with
its field (`'title: Computer Systems Design Services'`), so the backbone can tell the fields apart.
A query is a fifth field, `query`, read by the same backbone. An absent channel (null or blank)
never enters the backbone, and fusion masks it (Req 9).

### 3. Fusion and the Mixture-of-Experts Ablation

`model.fusion` picks how the present channels' vectors become one (`text_model/fusion.py`):

- **`masked_mean`** (default): the mean over present channels, with no parameters
- **`attention`:** attention pooling over present channels; it starts as the masked mean
- **`moe`** (an ablation only): the masked mean, then `text_model/moe.py`'s top-2 experts

Router-guided mining, the load-balancing term and their logs run only under `moe` (spec R10,
R11). Under any other fusion, the geometric miner takes every mining slot.
```

Replace:

```markdown
| `CurriculumMixin` | `mixins/curriculum.py` | Hard negative mining, router-guided sampling |
```

with:

```markdown
| `CurriculumMixin` | `mixins/curriculum.py` | Hard negative mining; router-guided sampling under `moe` |
```

Replace:

```markdown
uv run naics-embedder tools regressor-panel  # Score an arm on the regressor panel
```

with:

```markdown
uv run naics-embedder tools regressor-panel  # Score an arm on the regressor panel
uv run naics-embedder tools export-table  # Export a checkpoint's code table in Req 2's form
uv run naics-embedder tools outcome-panel  # Score an arm on the outcome validation split
```

Replace:

````markdown
```python
# In encoder.py
if use_gradient_checkpointing:
    for channel in self.channels:
        self.encoders[channel].enable_input_require_grads()
        self.encoders[channel].base_model.gradient_checkpointing_enable()
```
````

with:

````markdown
```python
# In shared_encoder.py
if use_gradient_checkpointing:
    self.backbone.enable_input_require_grads()
    self.backbone.base_model.gradient_checkpointing_enable()
```
````

Replace:

```markdown
Both prevent numerical instability in hyperbolic operations.
```

with:

```markdown
Both prevent numerical instability in hyperbolic operations. The text model's curvature is a
fixed 1.0, and export and reads take c = 1 only: `tools export-table` and `tools outcome-panel`
refuse a checkpoint trained at any other curvature (spec R8).
```

Replace:

```markdown
- `test_encoder.py` - Multi-channel encoding
```

with:

```markdown
- `test_encoder.py` - The shared encoder, on a tiny BERT
- `test_fusion.py` - Fusion options and masking
- `test_export.py` - The code-table export and the HGCN feeder
- `test_arm_encoder.py` - The arm encoder and the outcome read
```

Replace:

```markdown
- `text_model/encoder.py` - Multi-channel encoder
- `text_model/moe.py` - Mixture-of-Experts
```

with:

```markdown
- `text_model/shared_encoder.py` - The shared encoder
- `text_model/fusion.py` - Fusion options
- `text_model/moe.py` - Mixture-of-Experts (the `moe` fusion ablation)
```

- [ ] **Step 2: `README.md`**

Replace:

```markdown
The system combines multi-channel text encoding, Mixture-of-Experts fusion, hyperbolic contrastive learning, and a hyperbolic graph refinement stage to produce geometry-aware embeddings aligned with the hierarchical structure of the NAICS taxonomy.
```

with:

```markdown
The system combines a shared text encoder over marked text fields, masked fusion, hyperbolic contrastive learning, and a hyperbolic graph refinement stage to produce geometry-aware embeddings aligned with the hierarchical structure of the NAICS taxonomy.
```

Replace:

```markdown
1. **Multi-channel text encoding** – independent transformer-based encoders for title, description, examples, and exclusions.
2. **Mixture-of-Experts (MoE) fusion** – adaptive fusion of the four embeddings using Top-2 gating.
```

with:

```markdown
1. **Shared text encoding** – one LoRA-adapted transformer reads the title, description, examples and exclusions, each marked with its field, and reads queries through the same layers.
2. **Fusion and projection** – a masked mean over the present channels (attention pooling and a Mixture-of-Experts are options, the MoE an ablation only), then one linear map to dimension d ∈ {8, 16, 32}.
```

Replace:

```markdown
## 2. Stage 1 — Multi-Channel Text Encoding
```

with:

```markdown
## 2. Stage 1 — Shared Text Encoding
```

Replace:

```markdown
Each field is processed independently using a transformer encoder (LoRA-adapted). This produces four Euclidean embeddings:

- Title: (Embedding_title)
- Description: (Embedding_description)
- Examples: Embedding_examples)
- Excluded: (Embedding_excluded)

These embeddings serve as inputs to the fusion stage.
```

with:

```markdown
Every field goes through one shared transformer (LoRA-adapted), marked with its name (`title: …`), so one backbone tells the fields apart. A query is a fifth field, `query`, read by the same layers, so queries and codes share one space. An absent field never enters the backbone. Each present field is mean-pooled into one vector, and these vectors are the fusion stage's input.
```

Replace:

```markdown
## 3. Stage 2 — Mixture-of-Experts Fusion (Top-2 Gating)

The four channel embeddings are concatenated and passed into a **Mixture-of-Experts (MoE)** module. Key components include:

- Top-2 gating to route each input to the two most relevant experts.
- Feed-forward expert networks that learn specialized fusion behaviors.
- Auxiliary load-balancing loss to ensure even expert utilization across batches.

This produces a single fused Euclidean embedding (E_fused) per NAICS code.
```

with:

```markdown
## 3. Stage 2 — Fusion and Projection

The present channels' vectors are fused into one (`model.fusion`):

- **Masked mean** (default): the mean over the present channels, with no parameters.
- **Attention pooling**: a learned weighting over the present channels.
- **Mixture-of-Experts** (an ablation only): the masked mean routed through top-2 experts, with an auxiliary load-balancing loss.

Exactly one linear map then takes the fused vector to dimension d ∈ {8, 16, 32} (`model.dimension`, default 16).
```

Replace:

```markdown
### 4.1 Hyperbolic Projection
```

with:

```markdown
### 4.1 The Hyperbolic Head
```

Replace:

```markdown
The fused Euclidean vector is mapped onto the hyperboloid:

- Uses exponential map at the origin
- Supports learned or fixed curvature
- Ensures numerical stability

The result is a Lorentz embedding (E_hyp).
```

with:

```markdown
The d-dimensional vector is a tangent vector at the origin. A parameter-free head caps its norm at 2 and maps it onto the hyperboloid:

- Uses the exponential map at the origin
- Curvature fixed at c = 1; export and reads refuse any other
- Ensures numerical stability

The result is a Lorentz embedding (E_hyp) with d + 1 coordinates. The export (`tools export-table`) writes the tangent coordinates, `e0 … e{d-1}`: Req 2's form.
```

Replace:

```text
+-------------------------------+
|  Multi-Channel Text Encoder   |
|  (Title / Desc / Examples /   |
|   Excluded via Transformer)   |
+---------------+---------------+
                |
                v
+-------------------------------+
|     Mixture-of-Experts        |
|  Top-2 Gating + Expert MLPs   |
|  Load-Balanced Fusion Layer   |
+---------------+---------------+
                |
                v
+-------------------------------+
|   Hyperbolic Projection       |
|   (Lorentz Exponential Map)   |
+---------------+---------------+
```

with:

```text
+-------------------------------+
|     Shared Text Encoder       |
|  (one LoRA backbone; marked   |
|   fields and queries)         |
+---------------+---------------+
                |
                v
+-------------------------------+
|  Masked Fusion + Linear(→ d)  |
|  (MoE only as an ablation)    |
+---------------+---------------+
                |
                v
+-------------------------------+
|   Hyperbolic Head             |
|   (cap, Lorentz exp map, c=1) |
+---------------+---------------+
```

- [ ] **Step 3: `docs/index.md`**

Replace:

```markdown
- Multi-channel transformer-based text encoding  
- Mixture-of-Experts fusion  
```

with:

```markdown
- A shared transformer-based text encoder over marked fields  
- Masked fusion and one linear map to dimension d  
```

The two trailing spaces on each line are the file's own; keep them.

Replace:

```markdown
- **Router-Guided Sampling**: Prevents expert collapse by selecting negatives that confuse the MoE gating network
```

with:

```markdown
- **Router-Guided Sampling** (under `model.fusion: moe` only): Prevents expert collapse by
  selecting negatives that confuse the MoE gating network
```

- [ ] **Step 4: `docs/overview.md`**

Replace:

```markdown
3. [Multi-Channel Text Encoding](#3-multi-channel-text-encoding)
4. [Mixture-of-Experts Fusion](#4-mixture-of-experts-fusion)
```

with:

```markdown
3. [Shared Text Encoding](#3-shared-text-encoding)
4. [Fusion and Projection](#4-fusion-and-projection)
```

Replace:

```markdown
**2. Mixture-of-Experts Fusion:** Each NAICS code has four text channels (title, description, examples, excluded) with heterogeneous informativeness. MoE with Top-2 gating enables learning multiple specialized fusion strategies, allowing different experts to handle different types of codes.
```

with:

```markdown
**2. One Shared Encoder:** Each NAICS code has four text channels (title, description, examples,
excluded) with heterogeneous informativeness. One LoRA-adapted backbone reads them all, each
marked with its field, and reads queries through the same layers, so codes and queries share one
space. A masked mean fuses the present channels; attention pooling and a Mixture-of-Experts are
options, the MoE an ablation only.
```

Replace:

```markdown
| 1 | Multi-Channel Text Encoding (4 LoRA-adapted transformers) | E_title, E_desc, E_examples, E_excluded (4 × embedding_dim) |
| 2 | Mixture-of-Experts Fusion (Top-2 gating, 4 experts) | E_fused (embedding_dim) |
| 3 | Hyperbolic Projection (Lorentz exponential map) | E_hyp (embedding_dim + 1) |
```

with:

```markdown
| 1 | Shared Text Encoding (one LoRA-adapted backbone, marked fields) | One vector per present channel (384) |
| 2 | Fusion (masked mean by default) and one Linear(384 → d) | Tangent vector v (d, in {8, 16, 32}) |
| 3 | Hyperbolic Head (norm cap, Lorentz exponential map, c = 1) | E_hyp (d + 1) |
```

Replace:

```text
NAICS Code (4 text channels)
        ↓
[Multi-Channel Encoder]
    ├─→ Title Encoder (LoRA) → E_title
    ├─→ Description Encoder (LoRA) → E_desc
    ├─→ Examples Encoder (LoRA) → E_examples
    └─→ Excluded Encoder (LoRA) → E_excluded
        ↓
[Concatenate] → (embedding_dim × 4)
        ↓
[MoE Fusion] → E_fused (embedding_dim)
        ↓
[Hyperbolic Projection] → E_hyp (embedding_dim + 1)
        ↓
[Lorentz Hyperboloid] → Final Embedding
```

with:

```text
NAICS Code (4 text channels) or a query
        ↓
[Field markers] → 'title: …', 'description: …', 'excluded: …', 'examples: …', 'query: …'
        ↓
[Shared Encoder (one LoRA backbone)] → one vector per present channel (384)
        ↓
[Fusion: masked mean | attention | MoE ablation] → fused vector (384)
        ↓
[Linear(384 → d)] → tangent vector v (d)
        ↓
[Hyperbolic Head: cap at norm 2, exp map at the origin] → E_hyp (d + 1)
        ↓
[Lorentz Hyperboloid] → Final Embedding
```

Replace:

```markdown
## 3. Multi-Channel Text Encoding
```

with:

```markdown
## 3. Shared Text Encoding
```

Replace:

```markdown
Each channel uses a separate LoRA-adapted transformer encoder based on sentence-transformers. LoRA (Low-Rank Adaptation) reduces trainable parameters while maintaining expressiveness:
```

with:

```markdown
One LoRA-adapted transformer, based on sentence-transformers, encodes every field. Each present
text is marked with its field (`'title: Software Publishers'`), so the backbone can tell the
fields apart, and an absent text never enters it. LoRA (Low-Rank Adaptation) reduces trainable
parameters while maintaining expressiveness:
```

Replace:

```markdown
| base_model | all-mpnet-base-v2 | Pre-trained sentence transformer |
```

with:

```markdown
| base_model | all-MiniLM-L6-v2 | Pre-trained sentence transformer (revision 1110a243) |
```

Replace everything from the line `## 4. Mixture-of-Experts Fusion` through this line, the last
of the old section 4:

```markdown
This requires synchronizing expert utilization counts (f_i) and router probabilities (P_i) across all distributed workers via AllReduce before computing the loss.
```

Each of those two lines occurs once in the file. The new text is:

```markdown
## 4. Fusion and Projection

The relative importance of text channels varies across NAICS codes, and some codes lack a channel
altogether. Fusion turns the present channels' vectors into one, and `model.fusion` chooses how:

| Option | Parameters | Function |
|--------|------------|----------|
| `masked_mean` (default) | None | Mean over the present channels |
| `attention` | One learned query vector | Softmax-weighted mean over the present channels; starts as the masked mean |
| `moe` (ablation only) | Gating network and experts | The masked mean, routed through top-2 experts (`text_model/moe.py`) |

Every option masks absent channels itself, so perturbing an absent channel's input leaves the
output unchanged (Req 9). The masked mean is also the D9 text-only comparator's pooling, so an
arm and its comparator differ only by training, the projection and the field markers.

Exactly one `Linear(384 → d)` then maps the fused vector to the arm's dimension, d in {8, 16, 32}
(`model.dimension`, default 16). Its output is the tangent vector the hyperbolic head maps onto
the hyperboloid (Section 5).

### The Mixture-of-Experts Ablation

Under `moe`, a gating network routes the fused vector to the top 2 of 4 expert MLPs (hidden size
1024). Two mechanisms exist only for this option (spec R10, R11):

- **Load balancing.** An auxiliary loss, `L_aux = α · N · Σ(f_i · P_i)`, keeps expert use even.
  N is the number of experts, α = 0.01, f_i the share of inputs routed to expert i and P_i its
  mean gate probability. Under distributed training the statistics are synchronized across
  workers before the loss (Section 12).
- **Router-guided mining** (Section 7), which needs the gate outputs.

Under the other options the model has no gates, so neither runs and neither is logged.
```

Replace:

```markdown
### Hyperbolic Projection Implementation

The fused Euclidean embedding is projected onto the hyperboloid via a linear projection followed by the exponential map at the origin. The projection adds the time coordinate dimension (embedding_dim → embedding_dim + 1) and ensures points satisfy the Lorentz constraint through numerically stable clamping.
```

with:

```markdown
### The Hyperbolic Head

The head (`HyperbolicHead`, `text_model/hyperbolic.py`) has no parameters, so the one linear map
of Section 4 is the only affine map between the encoder and the point. It caps the tangent
vector's norm at 2, then applies the exponential map at the origin, which adds the time
coordinate (d → d + 1). It returns both the capped tangent, which the export writes (Req 2's
form), and the point. The curvature is c = 1; export and reads refuse a checkpoint trained at any
other (spec R8).
```

Replace:

```markdown
Router-guided sampling selects negatives that maximize confusion in the MoE gating network. If the router sends anchor and negative to the same experts with similar confidence, they are "computationally indistinguishable." Using these as contrastive negatives forces experts to become more discriminative and combats mode collapse.
```

with:

```markdown
Under `model.fusion: moe` only (spec R10), router-guided sampling selects negatives that maximize
confusion in the MoE gating network. If the router sends anchor and negative to the same experts
with similar confidence, they are "computationally indistinguishable." Using these as contrastive
negatives forces experts to become more discriminative and combats mode collapse. Under any other
fusion the model has no gates, and the geometric miner takes every mining slot.
```

Replace:

```markdown
As the embedding space matures, transition from symbolic tree priors to learned semantics. Sample a candidate pool, then select top-k negatives minimizing Lorentzian distance. Router-guided sampling is also enabled to force expert specialization.

**Curriculum Flags:** `enable_hard_negative_mining=True`, `enable_router_guided_sampling=True`
```

with:

```markdown
As the embedding space matures, transition from symbolic tree priors to learned semantics. Sample
a candidate pool, then select top-k negatives minimizing Lorentzian distance. Under `moe`,
router-guided sampling is also enabled to force expert specialization.

**Curriculum Flags:** `enable_hard_negative_mining=True`, `enable_router_guided_sampling=True`
(the router flag is read only under `moe`)
```

Replace:

```markdown
| 2 | 30-70% | Hard negative mining, router-guided sampling | Refine shape |
```

with:

```markdown
| 2 | 30-70% | Hard negative mining; router-guided sampling under `moe` | Refine shape |
```

Replace:

```markdown
As described in Section 4, ensures even expert utilization:
```

with:

```markdown
Under `model.fusion: moe` only, as described in Section 4, ensures even expert utilization:
```

Replace:

```markdown
| Load Balancing | 0.01 | Expert utilization balance |
```

with:

```markdown
| Load Balancing | 0.01 | Expert utilization balance (`moe` only) |
```

Replace:

```markdown
**Global-Batch Load Balancing:** Expert utilization statistics are synchronized across all workers via AllReduce before computing the auxiliary loss, enabling true domain specialization.
```

with:

```markdown
**Global-Batch Load Balancing:** Under `moe`, expert utilization statistics are synchronized
across all workers via AllReduce before computing the auxiliary loss, enabling true domain
specialization.
```

Replace:

```markdown
| `MultiChannelEncoder` | `text_model/encoder.py` | 4-channel text encoding |
| `MixtureOfExperts` | `text_model/moe.py` | MoE fusion layer |
| `HyperbolicProjection` | `text_model/hyperbolic.py` | Lorentz projection |
```

with:

```markdown
| `SharedEncoder` | `text_model/shared_encoder.py` | One backbone, fusion, one Linear(384 → d), the head |
| `build_fusion` | `text_model/fusion.py` | Masked mean, attention pooling, the MoE ablation |
| `MixtureOfExperts` | `text_model/moe.py` | The experts of the `moe` ablation |
| `HyperbolicHead` | `text_model/hyperbolic.py` | Norm cap and Lorentz exponential map |
| `ArmEncoder` | `text_model/arm_encoder.py` | `QueryCodeEncoder` from a checkpoint and its table |
| `export_code_table` | `text_model/export.py` | The 2,125-code table in Req 2's form |
```

Replace:

```markdown
| `RouterGuidedNegativeMiner` | `text_model/hard_negative_mining.py` | Router-confusion mining |
```

with:

```markdown
| `RouterGuidedNegativeMiner` | `text_model/hard_negative_mining.py` | Router-confusion mining (`moe` only) |
```

Replace:

```markdown
| `CurriculumMixin` | `text_model/mixins/curriculum.py` | Checked negative selection (hard negative and router-guided proposals) |
```

with:

```markdown
| `CurriculumMixin` | `text_model/mixins/curriculum.py` | Checked negative selection (hard negative proposals; router-guided ones under `moe`) |
```

Replace:

```markdown
| Model | base_model_name | all-mpnet-base-v2 |
| LoRA | r / alpha / dropout | 8 / 16 / 0.1 |
| MoE | num_experts / top_k / hidden_dim | 4 / 2 / 1024 |
```

with:

```markdown
| Model | base_model_name | all-MiniLM-L6-v2 |
| Model | fusion / dimension | masked_mean / 16 |
| LoRA | r / alpha / dropout | 8 / 16 / 0.1 |
| MoE (`moe` only) | num_experts / top_k / hidden_dim | 4 / 2 / 1024 |
```

Replace:

```markdown
| MoE | load_balancing_coef | 0.01 |
```

with:

```markdown
| MoE (`moe` only) | load_balancing_coef | 0.01 |
```

Replace:

````markdown
```bash
# Data preprocessing
uv run naics-embedder data all

# Training
uv run naics-embedder train
```
````

with:

````markdown
```bash
# Data preprocessing
uv run naics-embedder data all

# Training
uv run naics-embedder train

# Export a checkpoint's code table, then read the outcome panel's validation split
uv run naics-embedder tools export-table --checkpoint checkpoints/sadc_default/last.ckpt \
  --output arm_table.parquet
uv run naics-embedder tools outcome-panel --checkpoint checkpoints/sadc_default/last.ckpt \
  --table arm_table.parquet --purpose 'why this read happens'
```
````

- [ ] **Step 5: `docs/text_training.md`**

Replace:

```markdown
  - [Quick Start](#quick-start)
  - [SADC Scheduler](#sadc-scheduler)
```

with:

```markdown
  - [Quick Start](#quick-start)
  - [The Shared Encoder](#the-shared-encoder)
  - [SADC Scheduler](#sadc-scheduler)
```

Replace:

```markdown
To avoid repeating the override, set `supervision.manifest_path` in `conf/config.yaml`.

---

## SADC Scheduler
```

with:

````markdown
To avoid repeating the override, set `supervision.manifest_path` in `conf/config.yaml`.

---

## The Shared Encoder

One LoRA-adapted MiniLM backbone (revision 1110a243) reads every field
(`text_model/shared_encoder.py`):

- **Field markers.** A present text is marked with its field: `'title: …'`, `'description: …'`,
  `'excluded: …'`, `'examples: …'` or `'query: …'` (`text_model/fields.py`). An absent text
  (null or blank) is the unmarked empty string with `present` False. The tokenization cache's
  format `channels-v3` stores the marked texts. Its sidecar records the markers and a `summaries`
  entry, null until Stage 6b. A cache built under another format, other markers or other
  summaries is rebuilt.
- **Present channels only.** A batch's present texts go through the backbone in one call, and
  absent texts and padding columns never enter it. Each present text is mean-pooled over its
  tokens.
- **Fusion** (`model.fusion`): `masked_mean` (default), `attention`, or `moe`, the ablation that
  routes the masked mean through the experts of `model.moe`. Router-guided mining and the
  load-balancing term run only under `moe` (spec R10, R11).
- **Projection** (`model.dimension`): exactly one `Linear(384 → d)`, d in {8, 16, 32}, default
  16.
- **Head.** A parameter-free head caps the tangent's norm at 2 and maps it onto the hyperboloid
  at c = 1. The model returns both the capped `tangent` (d) and the point `embedding` (d + 1).

```bash
# An ablation at dimension 8 under the MoE fusion
uv run naics-embedder train supervision.manifest_path=/absolute/path/to/manifest.json \
  model.fusion=moe model.dimension=8
```

A checkpoint records its encoder architecture (layout, fusion, dimension and backbone) in its
contract. A checkpoint of another architecture cannot load, and nothing migrates it: four-copy
checkpoints from before Stage 6 cannot load into the shared encoder (roadmap D2).

After training, `tools export-table` writes the 2,125-code table, and `tools outcome-panel` reads
the outcome panel's validation split (see `docs/usage.md`).

---

## SADC Scheduler
````

Replace:

```markdown
   - Effect: activates Lorentzian hard-negative mining and router-guided MoE sampling.
```

with:

```markdown
   - Effect: activates Lorentzian hard-negative mining, and router-guided sampling under
     `model.fusion: moe` only; under any other fusion the geometric miner fills every slot.
```

Replace:

```markdown
mode, contract version, bundle ID, codebook fingerprint, structural-preference-loss version, and
mining-contract version. Structural matrices are loaded from the validated bundle, not trusted from
checkpoint state.
```

with:

```markdown
mode, contract version, bundle ID, codebook fingerprint, structural-preference-loss version,
mining-contract version, and the encoder architecture (layout, fusion, dimension and backbone).
Structural matrices are loaded from the validated bundle, not trusted from checkpoint state.
```

Replace:

```markdown
  `exact resume contract mismatch (saved, runtime): {...}`. Checkpoints without a contract fail
  with `legacy checkpoint has no Stage-3 contract and cannot exact resume; use weights_only
  explicitly`.
- **`--checkpoint-load-mode weights_only`** loads only allowlisted encoder parameters (`encoder.*`:
  transformer adapters, projection, MoE, and router). Loss modules and data-derived buffers
```

with:

```markdown
  `exact resume contract mismatch (saved, runtime): {...}`. Checkpoints without a contract fail
  with `legacy checkpoint has no Stage-3 contract and cannot exact resume`, followed by the D2
  refusal. A checkpoint of another encoder architecture fails with that refusal too (roadmap D2).
- **`--checkpoint-load-mode weights_only`** loads only allowlisted encoder parameters (`encoder.*`:
  the shared backbone and its adapter, fusion, and the projection). It refuses a checkpoint of
  another encoder architecture before reading any parameter (D2). Loss modules and data-derived
  buffers
```

Replace:

```markdown
- MoE routing, load balancing, and radius regularizers still run;
```

with:

```markdown
- radius regularizers still run, as do MoE routing and load balancing under `model.fusion: moe`;
```

Replace:

```markdown
  - Router-guided proposals (gate confusion) fill the remaining slots.
```

with:

```markdown
  - Router-guided proposals (gate confusion) fill the remaining slots, under `model.fusion: moe`
    only.
```

Replace:

```markdown
  - Phase 2/3 flags (`enable_hard_negative_mining`, `enable_router_guided_sampling`, `enable_clustering`) act in the model layer.
```

with:

```markdown
  - Phase 2/3 flags (`enable_hard_negative_mining`, `enable_router_guided_sampling`,
    `enable_clustering`) act in the model layer; the router flag acts only under `moe`.
```

- [ ] **Step 6: `docs/usage.md`**

Replace:

```markdown
- `--coordinates PATH` - The arm's 2,125-code table in the export form (tangent coordinates at
  the origin for a hyperbolic arm). Lorentz points and constant columns, such as a log map's
  zero time coordinate, are refused
```

with:

```markdown
- `--coordinates PATH` - The arm's 2,125-code table in the export form (`tools export-table`:
  tangent coordinates at the origin for a hyperbolic arm). Lorentz points and constant columns,
  such as a log map's zero time coordinate, are refused
```

Replace:

```markdown
### `tools margins`
```

with:

````markdown
### `tools export-table`

Export an arm's 2,125-code table in Req 2's form (roadmap Stage 6). Every code goes through the
checkpoint's model in eval mode. The table holds `code`, `index` and `level`, then
`e0 … e{d-1}` (float64): each code's tangent vector at the origin, capped at norm 2, in the
bundle's codebook order.

The command resolves the bundle and the token cache as `train` does, from `--config` and
`key=value` overrides. The checkpoint's supervision contract must match the bundle. Its encoder
record is its own, so a d = 8 checkpoint exports under a d = 16 config. A checkpoint trained at a
curvature other than 1, or of the four-copy encoder (roadmap D2), is refused.

**Generates:** the table and `<stem>_provenance.json` beside it. The provenance records the
checkpoint's sha256 and contract, the backbone's revision, the window, the descriptions' sha256,
`summaries`, and the table's sha256 and `matrix_fingerprint`.

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/sadc_default/last.ckpt \
  --output data/arm_table.parquet supervision.manifest_path=/absolute/path/to/manifest.json
```

**Options:**
- `--checkpoint PATH` - The arm's checkpoint
- `--output PATH` - Where to write the table
- `--config PATH` - Config naming the bundle and the token cache (default: `conf/config.yaml`)
- `KEY=VALUE ...` - Config overrides, as `train` takes them; one without `=` is refused

### `tools outcome-panel`

Score an arm on the outcome panel's validation split under its own distance (`lorentz` for the
hyperbolic head). Queries are marked `query:` and go through the checkpoint's model. Codes are
decoded from the table `tools export-table` wrote from that checkpoint, and the table's provenance
must name both. The read is appended to the selection log with the table's `matrix_fingerprint`
(`table`) and the checkpoint's sha256 (`checkpoint`). The test split stays sealed: this command
has no `--split`.

```bash
uv run naics-embedder tools outcome-panel --checkpoint checkpoints/sadc_default/last.ckpt \
  --table data/arm_table.parquet --purpose 'first live reading' \
  supervision.manifest_path=/absolute/path/to/manifest.json
```

**Options:**
- `--checkpoint PATH`, `--table PATH` - The arm's checkpoint and the table exported from it
- `--purpose TEXT` - Why this read happens; recorded in the selection log (required)
- `--config PATH` - Config naming the bundle and the token cache (default: `conf/config.yaml`)
- `--log PATH` - Selection log (default: `logs/selection_log.jsonl`, from
  `conf/data/outcome_panel.yaml`)
- `--output PATH` - Also write the summary as JSON, with the panel's fingerprint, the table's
  `matrix_fingerprint` and the checkpoint's sha256
- `KEY=VALUE ...` - Config overrides, as `train` takes them

### `tools margins`
````

- [ ] **Step 7: `tests/README.md`**

Replace:

```markdown
│   ├── test_moe.py           # Mixture of Experts ✅
│   ├── test_encoder.py       # Multi-channel encoder ✅
```

with:

```markdown
│   ├── test_moe.py           # Mixture of Experts (the moe fusion ablation) ✅
│   ├── test_encoder.py       # The shared encoder (tiny BERT) ✅
│   ├── test_fusion.py        # Fusion options and masking ✅
│   ├── test_export.py        # Code-table export and the HGCN feeder ✅
│   ├── test_arm_encoder.py   # The arm encoder and the outcome read ✅
```

Replace:

```markdown
   - HyperbolicProjection (Euclidean → Lorentz)
```

with:

```markdown
   - HyperbolicHead (norm cap and exp map, no parameters)
```

Replace:

```markdown
3. **text_model/encoder.py** ✅ - `test_encoder.py`
   - Multi-channel transformer encoders (title, description, examples, exclusions)
   - LoRA adaptation layers
   - Channel-specific encoding
```

with:

```markdown
3. **text_model/shared_encoder.py** ✅ - `test_encoder.py`
   - One LoRA-adapted backbone over marked fields and queries
   - Present channels only: masking an absent channel leaves the output bit-identical
   - Fusion options, one affine map, gradient to every adapter
```

- [ ] **Step 8: The API pages and the navigation**

Replace the whole of `docs/api/encoder.md` with:

```markdown
# Shared Encoder API

One LoRA-adapted backbone over marked fields, fusion of the present channels, and one affine map
to dimension d (roadmap Stage 6).

## Fields and markers

::: naics_embedder.text_model.fields

## Fusion

::: naics_embedder.text_model.fusion

## The encoder

::: naics_embedder.text_model.shared_encoder
```

Create `docs/api/export.md`:

```markdown
# Export and Arm Encoder API

The code-table export in Req 2's form, and the arm encoder the outcome panel reads (roadmap
Stage 6).

## Export

::: naics_embedder.text_model.export

## Arm encoder

::: naics_embedder.text_model.arm_encoder
```

Replace the whole of `docs/api/moe.md` with:

```markdown
# Mixture of Experts API

The experts of the `moe` fusion, an ablation only (roadmap Stage 6).

::: naics_embedder.text_model.moe
```

In `docs/.nav.yml`, replace:

```yaml
              - Encoder: api/encoder.md
              - Mixture of Experts: api/moe.md
```

with:

```yaml
              - Shared Encoder: api/encoder.md
              - Export and Arm Encoder: api/export.md
              - Mixture of Experts: api/moe.md
```

- [ ] **Step 9: Acceptance checks**

Run: `git grep -n -i -E 'MultiChannelEncoder|HyperbolicProjection|text_model/encoder\.py|In encoder\.py|embedding_euc|separate LoRA|Independent LoRA|4 Euclidean|four Euclidean|E_fused' -- CLAUDE.md README.md docs tests/README.md ':!docs/report_files'`
Expected: no output.

Run: `git grep -n -E 'tools export-table|tools outcome-panel' -- CLAUDE.md docs/usage.md docs/overview.md docs/text_training.md`
Expected: matches in all four files.

Run: `uv run mkdocs build --strict`
Expected: the build succeeds with no warnings. Griffe is strict, so every documented parameter of
the five new pages' modules needs its own `Args:` entry. Fix any warning in the docstring it names,
then rerun.

Run: `git status --short`
Expected: only the files in **Files** above, and no `site/` (gitignored).

- [ ] **Step 10: Commit**

```bash
git add CLAUDE.md README.md docs/index.md docs/overview.md docs/text_training.md docs/usage.md tests/README.md docs/api/encoder.md docs/api/export.md docs/api/moe.md docs/.nav.yml
git commit -m "docs: the shared encoder, its fusion options, the export and the outcome read"
```

### Task 14: The Exit: one local epoch, the export and both validation reads

This task runs inline in the controller session (spec §7). It trains the default arm (d = 16,
`masked_mean`) for one epoch on MPS, exports `last.ckpt`, reads both panels' validation splits,
and records the run in a finding. Off CUDA the trainer picks `32-true` (`utils/backend.py:33`).

**Rules for every step:**
- Run from the worktree root, and run `pwd` first.
- Write each command out in full. The Bash tool keeps no shell variables between calls.
- Prefix every command that loads the backbone or its tokenizer with `HF_HUB_OFFLINE=1`, so the
  cached revision `1110a243` is the one read (R2).
- Never open a sealed split: no `--split test`, no `OutcomePanel.open_test`, no
  `RegressorPanel.open_outer`.
- Never commit `supervision.manifest_path`, `data/`, `checkpoints/`, `logs/` or `outputs/`.
- Selection happens nowhere. The checkpoint is `last.ckpt`. The harness's monitor reads the
  in-sample validation loss, which selects nothing (Req 4), and Stage 7 wires selection to the
  validation query split (D6).

The bundle clone's manifest, which every command names, is:

```text
/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-8-shared-encoder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

**Files:**
- Create: `specs/findings/shared-encoder-first-reading.md`
- Outputs, never committed:
  - `data/supervision/…` and `data/naics_descriptions.parquet` (the clones);
  - `data/token_cache/`;
  - `data/plan8/`;
  - `checkpoints/sadc_default/`;
  - `logs/plan8_train.out` and `logs/selection_log.jsonl`;
  - `outputs/sadc_default/`.

**Interfaces:**
- Consumes: `naics-embedder train`, plus three commands: `tools export-table` (Task 10),
  `tools outcome-panel` (Task 12), and the existing `tools text-only-table` and
  `tools regressor-panel`.
- Produces: the finding, which Plan completion's roadmap entry cites.

- [ ] **Step 1: Confirm the workspace**

Run: `pwd`
Expected: `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-8-shared-encoder`.

Run: `git status --short`
Expected: no output.

Run: `git log --oneline origin/main..HEAD`
Expected: this plan's commit and the commits of Tasks 1–13; no "config" or "graph config".

Run: `ls logs/selection_log.jsonl checkpoints/sadc_default`
Expected: both "No such file or directory".
- If either exists, stop and ask.
- The selection log is append-only (Req 4), so never delete or move one.

- [ ] **Step 2: Clone the inputs**

Run: `mkdir -p data/supervision/stage3-supervision-v2 data/plan8 logs`

Run: `cp -cR /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c data/supervision/stage3-supervision-v2/`

Run: `cp -c /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet data/naics_descriptions.parquet`

`cp -c` makes APFS clones, so the copies cost no space and the originals stay untouched. Never
symlink, rebuild or edit them.

- [ ] **Step 3: Check the clones**

Run: `shasum -a 256 data/naics_descriptions.parquet`
Expected: `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36`.

Run: `uv run python -c "from naics_embedder.supervision.artifacts import load_validated_bundle; b = load_validated_bundle('/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-8-shared-encoder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json'); print(b.manifest.bundle_id, b.manifest.codebook_fingerprint[:8], b.manifest.description_fingerprint[:8])"`
Expected: `301cce28-539c-42ea-8781-496bbdcf511c 4662b826 fe8c54e3`. The load re-runs every
integrity and relational check.

- [ ] **Step 4: Train one epoch**

Run: `printf 'n\n' > /tmp/plan8-train-answer.txt`

`train` ends by asking whether to generate HGCN embeddings. Answering "n" ends it cleanly. With
no answer, the prompt aborts on end of input and `train` reports "Training failed" after saving
the checkpoint.

Run it detached, so the Bash tool's background limit cannot stop it:

Run: `HF_HUB_OFFLINE=1 nohup uv run naics-embedder train training.trainer.max_epochs=1 supervision.manifest_path=/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-8-shared-encoder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json < /tmp/plan8-train-answer.txt > logs/plan8_train.out 2>&1 & echo $!`
Expected: a process ID. Record it, and the start time, for the finding.

What to expect:
- **Timing.** About 1.5 hours: about 373 training batches at batch 16 (5,958 rows from 1,273
  anchors), about 9.5 s each on the M4 Max, then validation.
- **Cache.** The run first builds the token cache in format `channels-v3`
  (`data/token_cache/token_cache.pt`).
- **Waiting.** Use the Monitor tool with an until-loop on `kill -0 <pid>`, or check
  `tail -n 5 logs/plan8_train.out` every 10–20 minutes. Never run a foreground `sleep`.

Stop and ask if the log shows "Training failed" or a traceback, or if the run passes 2.5 hours.
Stop it with `kill <pid>`.

- [ ] **Step 5: Check the run**

Run: `tail -n 40 logs/plan8_train.out`
Expected:
- the training summary's "Training summary saved" line, then the HGCN prompt answered "n";
- no "Training failed";
- the banner earlier in the log names fusion `masked_mean` and dimension 16 (Task 7).

Run: `ls checkpoints/sadc_default`
Expected: `last.ckpt` and `config.yaml`, among the run's other files.

Write this script to `/tmp/plan8_check_run.py` with the Write tool:

```python
'''Plan 8 Exit: the trained checkpoint's contract, curvature and progress.'''

import torch

checkpoint = torch.load(
    'checkpoints/sadc_default/last.ckpt', map_location='cpu', weights_only=False
)
contract = checkpoint['stage3_supervision']
print('bundle', contract['bundle_id'])
print('encoder', contract['encoder'])
print('curvature', checkpoint['hyper_parameters']['curvature'])
print('epoch', checkpoint['epoch'], 'global_step', checkpoint['global_step'])
```

Run: `uv run python /tmp/plan8_check_run.py`
Expected, with the global step left out:

```text
bundle 301cce28-539c-42ea-8781-496bbdcf511c
encoder {'layout': 'shared', 'fusion': 'masked_mean', 'dimension': 16, 'backbone': 'sentence-transformers/all-MiniLM-L6-v2'}
curvature 1.0
epoch 0 global_step …
```

The global step counts optimizer steps. With `accumulate_grad_batches: 2`, that is about half the
batch count.

Record the epoch, the global step and the wall-clock time for the finding.

- [ ] **Step 6: Export the code table**

Run: `HF_HUB_OFFLINE=1 uv run naics-embedder tools export-table --checkpoint checkpoints/sadc_default/last.ckpt --output data/plan8/arm_table.parquet supervision.manifest_path=/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-8-shared-encoder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json`
Expected: `Code table: data/plan8/arm_table.parquet` and
`Provenance: data/plan8/arm_table_provenance.json`.

- [ ] **Step 7: Check the export**

Write this script to `/tmp/plan8_check_export.py` with the Write tool:

```python
'''Plan 8 Exit: the exported table's form, provenance and tangent norms.'''

import json
from pathlib import Path

import numpy as np
import polars as pl

table = pl.read_parquet('data/plan8/arm_table.parquet')
provenance = json.loads(Path('data/plan8/arm_table_provenance.json').read_text())
columns = [f'e{i}' for i in range(16)]
assert table.columns == ['code', 'index', 'level', *columns], table.columns
assert table.schema['code'] == pl.Utf8
assert table.schema['index'] == table.schema['level'] == pl.Int64
assert all(table.schema[name] == pl.Float64 for name in columns)
assert table.height == provenance['codes'] == 2125
assert provenance['dimension'] == 16
assert provenance['summaries'] is None
assert provenance['revision'] == '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
assert provenance['contract']['encoder']['fusion'] == 'masked_mean'

norms = np.linalg.norm(table.select(columns).to_numpy(), axis=1)
assert norms.max() <= 2.0 + 1e-6, norms.max()
print('levels', dict(table.group_by('level').len().sort('level').iter_rows()))
print(f'norms min {norms.min():.4f} median {np.median(norms):.4f} max {norms.max():.4f}')
print(f'share at the cap {(norms >= 2.0 - 1e-6).mean():.4f}')
for key in ('table_sha256', 'matrix_fingerprint'):
    print(key, provenance[key])
print('checkpoint sha256', provenance['checkpoint']['sha256'])
print('descriptions sha256', provenance['descriptions']['sha256'])
```

Run: `uv run python /tmp/plan8_check_export.py`
Expected:
- no assertion error;
- `levels {2: 20, 3: 96, 4: 308, 5: 689, 6: 1012}`;
- the norms, the share at the cap, and four hashes; the descriptions hash is `fe8c54e3…`.

Stop and ask if the table holds other than 2,125 codes.

Run: `shasum -a 256 data/plan8/arm_table.parquet checkpoints/sadc_default/last.ckpt`
Expected: the first hash equals `table_sha256`, the second the checkpoint `sha256`.

- [ ] **Step 8: Build the text-only table**

Run: `HF_HUB_OFFLINE=1 uv run naics-embedder tools text-only-table --descriptions data/naics_descriptions.parquet --output data/plan8/text_only.parquet`
Expected:
- the table and `data/plan8/text_only_provenance.json`;
- the backbone is the regressor config's MiniLM, read frozen from unmarked text (D9).

- [ ] **Step 9: Read the regressor panel's validation split**

Run: `uv run naics-embedder tools regressor-panel --coordinates data/plan8/arm_table.parquet --text-only data/plan8/text_only.parquet --codebook data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet --purpose 'plan 8 Exit: first reading of the shared encoder (d = 16, masked_mean, one local epoch)' --output data/plan8/regressor_validation.parquet`

The defaults are the validation split, both regimes and level 6. QCEW comes from `qcew_dir`.

Expected:
- one line per regime and comparator: rows, RMSE, R² and median alpha;
- `Reads logged to logs/selection_log.jsonl`.

Copy the printed lines for the finding. Stop and ask if the command fails.

- [ ] **Step 10: Read the outcome panel's validation split**

Run: `HF_HUB_OFFLINE=1 uv run naics-embedder tools outcome-panel --checkpoint checkpoints/sadc_default/last.ckpt --table data/plan8/arm_table.parquet --purpose 'plan 8 Exit: first reading of the shared encoder (d = 16, masked_mean, one local epoch)' --output data/plan8/outcome_validation.json supervision.manifest_path=/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-8-shared-encoder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json`

Expected:
- The printed summary has `distance` `lorentz`, `n_queries` 4,042, `n_codes` 939 and
  `n_candidates` 1,012.
  - These are the counts the lexical stub read (`specs/findings/outcome-panel-splits.md`, 5).
  - Plan 7's finding gives the same 4,042 validation queries under bundle 301cce28.
- It also prints `top1`, `mrr`, `hit_at_1`, `hit_at_5`, `hit_at_10` and `lca_level`.

Stop and ask if the counts differ or the command fails.

- [ ] **Step 11: Check the logged names**

Write this script to `/tmp/plan8_check_log.py` with the Write tool:

```python
'''Plan 8 Exit: the selection log names the exported table and checkpoint, and nothing else.'''

import json
from pathlib import Path

records = [
    json.loads(line) for line in Path('logs/selection_log.jsonl').read_text().splitlines()
]
provenance = json.loads(Path('data/plan8/arm_table_provenance.json').read_text())
table = provenance['matrix_fingerprint']

assert [record['event'] for record in records] == ['read'] * 3, 'an opening was logged'
assert {record['split'] for record in records} == {'validation'}
regressor = [record for record in records if record['panel'].startswith('regressor_')]
outcome = [record for record in records if record['panel'] == 'outcome']
assert sorted(record['panel'] for record in regressor) == [
    'regressor_heldout', 'regressor_seen'
]
assert all(record['detail']['arm'] == table for record in regressor)
assert all(record['detail']['dimension'] == 16 for record in regressor)
assert len(outcome) == 1
assert outcome[0]['detail']['table'] == table
assert outcome[0]['detail']['checkpoint'] == provenance['checkpoint']['sha256']
assert outcome[0]['detail']['distance'] == 'lorentz'
summary = json.loads(Path('data/plan8/outcome_validation.json').read_text())
assert summary['table'] == table
print('one name for the table:', table)
for record in records:
    print(json.dumps(record, sort_keys=True))
```

Run: `uv run python /tmp/plan8_check_log.py`
Expected:
- no assertion error;
- the table's one name, then the three records as JSON lines. Copy these into the finding: the
  log is gitignored.

- [ ] **Step 12: Write the finding**

Create `specs/findings/shared-encoder-first-reading.md` in the format of
`specs/findings/regressor-panel-splits.md`:
- prose wrapped at 100 columns;
- every value copied from Steps 4–11's output, never estimated;
- no value left unfilled.

Its sections:

- **Title and status.** Title: `# Shared encoder first reading: finding`. Then
  `**Status: FINAL (<date>).** Roadmap Stage 6 (`specs/naics-embedding-roadmap.md`).`
  - The finding records the Exit run of plan 8
    (`specs/plans/completed/8-shared-encoder-and-projection.md`), where Plan completion moves
    the plan: one local epoch of the default arm (d = 16, `masked_mean`), its export, and the
    first validation reads of both panels.
  - Say that no sealed split was opened and nothing was selected.
- **`## Sources`.**
  - The inputs:
    - bundle `301cce28-539c-42ea-8781-496bbdcf511c`, cloned with `cp -cR`;
    - the descriptions;
    - the QCEW slices, under the pins in `conf/data/regressor_panel.yaml`;
    - the backbone at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, under
      `HF_HUB_OFFLINE=1`.
  - The device (MPS, `32-true`), Python 3.12, and the library versions in the export's
    `library_versions`.
  - A `| File | sha256 |` table:
    - `data/naics_descriptions.parquet`;
    - `checkpoints/sadc_default/last.ckpt`;
    - `data/plan8/arm_table.parquet`;
    - `data/plan8/text_only.parquet`.
  - Below the table, the arm's `matrix_fingerprint` (from the provenance) and the text-only
    table's (the regressor records' `detail.text_only`).
- **`## 1. The run`.**
  - The command.
  - Start and end times and the wall-clock duration.
  - The batch count, the epoch and the global step.
  - The last logged `val/contrastive_loss`, noted as in-sample, selecting nothing.
  - The token cache's format, `channels-v3`, with `summaries` null.
- **`## 2. The export`.**
  - The table's shape (2,125 rows; `code`, `index`, `level`, `e0 … e15`) and its rows per level.
  - The tangent norms (minimum, median, maximum) and the share of codes at the cap of 2. Stage 7
    removes the cap (Req 13), so this share is what it changes.
  - The provenance's contract: bundle, codebook fingerprint, encoder record.
- **`## 3. The regressor panel`.**
  - Two tables in plan 5's format, `| regressor_seen | Level 6 |` and
    `| regressor_heldout | Level 6 |`, one row per comparator. Each cell is
    `RMSE / R² / median penalty`, pooled over the five repeats, from Step 9's lines.
  - One sentence: the `embedding` comparator is this arm's table, and `text_only` is the frozen
    backbone reduced by PCA to 16 dimensions (D9). The numbers are a floor for Stage 7's arms,
    not a target.
- **`## 4. The outcome panel`.**
  - The `| Metric | Value |` table of `specs/findings/outcome-panel-splits.md` section 5:
    queries / codes / candidates, top-1, MRR, Hit@1 / Hit@5 / Hit@10, mean LCA level.
  - The lexical stub's numbers from that section, as a reference column. Say that no decision
    reads them: Stage 7 compares arms under Req 5.
- **`## 5. The selection log`.**
  - The three records from Step 11, verbatim, in a `json` fence.
  - One sentence: the log was the worktree's gitignored `logs/selection_log.jsonl`, and these
    lines are its copy.
- **`## 6. What later stages read`.**
  - **Stage 6b.** The token cache's `summaries` entry is null, and its format is
    `channels-v3`.
  - **Stage 7.** Four items:
    - `ArmEncoder.from_files` and `read_outcome_validation` (`text_model/arm_encoder.py`);
    - `tools export-table` and `tools outcome-panel`;
    - the cap and its share of codes;
    - selection moves to the validation query split (D6).
  - **Stages 7–11.** An arm's regressor read takes the exported table as `--coordinates`, and
    the three logged names agree: the provenance's `matrix_fingerprint`, the regressor records'
    `detail.arm` and the outcome record's `detail.table`.
- **`## Reproduction`.**
  - Steps 2–10's commands in a `bash` fence, each on one line, with the manifest path written as
    `MANIFEST`.
  - Note that training is not bitwise reproducible on MPS, so a rerun's hashes differ. The panel
    reads reproduce from the committed table only if that table is kept, and it is not committed.

Write this script to `/tmp/plan8_check_lines.py` with the Write tool:

```python
'''Plan 8 Exit: the finding's prose wraps at 100 columns; tables and fences are exempt.'''

from pathlib import Path

path = Path('specs/findings/shared-encoder-first-reading.md')
fenced = False
long_lines = []
for number, line in enumerate(path.read_text().split('\n'), 1):
    if line.startswith('```'):
        fenced = not fenced
    elif not fenced and not line.startswith('|') and len(line) > 100:
        long_lines.append(number)
print(long_lines)
```

Run: `python3 /tmp/plan8_check_lines.py`
Expected: `[]`.

- [ ] **Step 13: Commit the finding**

Run: `git status --short`
Expected: `?? specs/findings/shared-encoder-first-reading.md` and `?? outputs/`. `outputs/` is not
gitignored. It holds the run's TensorBoard logs, so never add it.

```bash
git add specs/findings/shared-encoder-first-reading.md
git commit -m "docs(findings): the shared encoder's first reading (roadmap Stage 6 Exit)"
```

Run: `rm /tmp/plan8-train-answer.txt /tmp/plan8_check_run.py /tmp/plan8_check_export.py /tmp/plan8_check_log.py /tmp/plan8_check_lines.py`

## Final verification (controller, inline)

Run every check before the final review, and paste each output into the ledger.

- [ ] **Step 1: The suite on both CI versions**

Run: `uv run pytest -n auto -q`
Expected:
- all pass, with the baseline's one skip (CUDA);
- the passed count is 1697 plus the tests the plan added, minus those it removed;
- the MPS test runs here rather than skipping.

Run: `UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310 uv run pytest -n auto -q`
Expected:
- the same counts;
- about 300 extra "encountered in matmul" RuntimeWarnings, which come from numpy 2.2 with
  Accelerate on 3.10 and are not a failure.

Run: `rm -rf /tmp/naics-py310`

- [ ] **Step 2: Lint, format and docs**

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0, with no file listed.

Run: `uv run mkdocs build --strict`
Expected: the build succeeds with no warnings.

- [ ] **Step 3: Nothing of the four-copy encoder is left**

Run: `git grep -n -E 'MultiChannelEncoder|HyperbolicProjection|embedding_euc|moe_projection|text_model\.encoder|text_model/encoder\.py' -- src tests conf docs README.md CLAUDE.md ':!docs/report_files'`
Expected: no output.

Run: `git ls-files src/naics_embedder/text_model/encoder.py`
Expected: no output.

- [ ] **Step 4: Spec §6's criteria, test by test**

Each Exit criterion and supporting test of spec §6 maps to tests the plan wrote (the table
below). Step 1 ran them all. This step checks that every one still exists under its name.

Write this script to `/tmp/plan8_check_map.py` with the Write tool:

```python
'''Plan 8: every test that spec §6's criteria map to is collected under its name.'''

import subprocess

NAMES = [
    'test_a_one_field_batch_encodes_as_a_code_with_only_that_channel',
    'test_a_query_embeds_through_the_same_forward_as_a_code',
    'test_one_backbone_then_fusion_one_affine_map_and_a_parameter_free_head',
    'test_perturbing_an_absent_channel_leaves_the_output_bit_identical',
    'TestHyperbolicHead',
    'test_a_step_at_dimension_16_trains_the_adapter_and_the_projection',
    'test_a_step_reaches_every_adapter_and_the_projection',
    'test_load_balancing_is_computed_and_logged_only_under_moe',
    'test_phase_two_selection_under_masked_mean_fills_no_router_slot',
    'test_router_mining_runs_only_under_moe_fusion',
    'test_the_table_is_in_reqs_export_form',
    'test_the_table_holds_each_codes_capped_tangent',
    'test_the_provenance_names_the_table_and_the_checkpoint',
    'test_a_validation_read_logs_the_table_it_decodes_against',
    'test_table_decoded_distances_match_the_live_forward',
    'test_a_checkpoint_at_another_curvature_is_refused',
    'test_masked_mean_averages_the_present_channels_only',
    'test_attention_weights_present_channels_by_a_softmax_of_their_scores',
    'test_moe_routes_the_masked_mean_and_is_the_only_option_with_gates',
    'test_stack_text_inputs_carries_a_boolean_present_per_channel',
    'test_repaired_collate_marks_invalid_rows_absent',
    'test_present_channels_are_cached_with_their_markers',
    'test_the_sidecar_records_the_markers_and_null_summaries',
    'test_a_cache_in_the_unmarked_v2_format_is_rebuilt',
    'test_an_absent_encoder_record_reads_as_the_legacy_four_copy_layout',
    'test_another_encoder_architecture_cannot_exact_resume',
    'test_a_checkpoint_without_a_contract_cites_d2',
    'test_weights_only_refuses_another_encoder_before_reading_any_parameter',
    'test_matching_new_checkpoint_can_exact_resume',
    'test_the_encoder_record_survives_a_save_round_trip',
    'test_load_from_checkpoint_refuses_a_four_copy_checkpoint_before_its_weights',
    'test_a_duplicated_repeat_beside_a_missing_one_is_refused',
    'test_a_repeat_counted_twice_beside_every_other_one_is_refused',
    'test_the_lorentz_refusal_names_the_export_form_alone',
    'test_the_hgcn_feeder_writes_d_plus_one_lorentz_columns',
    'test_queries_and_codes_on_mps_come_back_float64_on_the_cpu',
]

collected = subprocess.run(
    ['uv', 'run', 'pytest', '--collect-only', '-q', 'tests'],
    capture_output=True,
    text=True,
    check=True,
).stdout
print('missing:', [name for name in NAMES if f'::{name}' not in collected])
```

Run: `uv run python /tmp/plan8_check_map.py`
Expected: `missing: []`. A missing name means a test was dropped or renamed during execution:
stop and report it as a deviation.

Run: `rm /tmp/plan8_check_map.py`

| Spec §6 | Tests |
|---|---|
| Same encoder: a one-field batch equals a one-code batch | `test_encoder.py::test_a_one_field_batch_encodes_as_a_code_with_only_that_channel` |
| Same encoder: `encode_queries` is the forward's exp map | `test_arm_encoder.py::test_a_query_embeds_through_the_same_forward_as_a_code` |
| Same encoder: one backbone | `test_encoder.py::test_one_backbone_then_fusion_one_affine_map_and_a_parameter_free_head` |
| Masking, under each fusion | `test_encoder.py::test_perturbing_an_absent_channel_leaves_the_output_bit_identical` (parametrized over `FUSIONS`) |
| One affine map; a parameter-free head | `test_encoder.py::test_one_backbone_then_fusion_one_affine_map_and_a_parameter_free_head`; `test_hyperbolic.py::TestHyperbolicHead` |
| Trains at d = 16: gradient to LoRA and the projection | `test_naics_model.py` `test_a_step_at_dimension_16_trains_the_adapter_and_the_projection`; `test_encoder.py::test_a_step_reaches_every_adapter_and_the_projection` |
| Trains at d = 16: no load-balancing term under `masked_mean` | `test_naics_model.py` `test_load_balancing_is_computed_and_logged_only_under_moe` |
| Trains at d = 16: phase 2 fills no router slot | `test_naics_model.py` `test_phase_two_selection_under_masked_mean_fills_no_router_slot`; `test_hard_negative_mining.py::test_router_mining_runs_only_under_moe_fusion` |
| Export: form, `coordinate_matrix`, provenance | `test_export.py` `test_the_table_is_in_reqs_export_form`, `test_the_table_holds_each_codes_capped_tangent`, `test_the_provenance_names_the_table_and_the_checkpoint` |
| Live read: the logged `table` | `test_arm_encoder.py::test_a_validation_read_logs_the_table_it_decodes_against` |
| Live read: table-decoded distances match the forward | `test_arm_encoder.py::test_table_decoded_distances_match_the_live_forward` |
| Live read: curvature ≠ 1 refused | `test_export.py::test_a_checkpoint_at_another_curvature_is_refused`; `test_arm_encoder.py::test_a_checkpoint_at_another_curvature_is_refused` |
| Fusion values; MoE after the mean, the only one with gates | `test_fusion.py` (all) |
| The collate's `present`, False on invalid rows | `test_datamodule.py` `test_stack_text_inputs_carries_a_boolean_present_per_channel`, `test_repaired_collate_marks_invalid_rows_absent` |
| Cache v3 | `test_tokenization_cache.py` `test_present_channels_are_cached_with_their_markers`, `test_the_sidecar_records_the_markers_and_null_summaries`, `test_a_cache_in_the_unmarked_v2_format_is_rebuilt`; `test_fields.py` (all) |
| Contract: absent reads as legacy | `test_checkpoint_contract.py::test_an_absent_encoder_record_reads_as_the_legacy_four_copy_layout` |
| Contract: exact-resume and weights-only refusals carry D2 | `test_checkpoint_contract.py` `test_another_encoder_architecture_cannot_exact_resume`, `test_a_checkpoint_without_a_contract_cites_d2`, `test_weights_only_refuses_another_encoder_before_reading_any_parameter` |
| Contract: a shared-encoder positive control | `test_checkpoint_contract.py::test_matching_new_checkpoint_can_exact_resume` |
| Contract: the record survives a save round trip | `test_checkpoint_contract.py::test_the_encoder_record_survives_a_save_round_trip` |
| Contract: a four-copy checkpoint meets the refusal through `load_from_checkpoint` | `test_naics_model.py` `test_load_from_checkpoint_refuses_a_four_copy_checkpoint_before_its_weights` |
| Plan 6's two fixes | `test_decision_scores.py` `test_a_duplicated_repeat_beside_a_missing_one_is_refused`, `test_a_repeat_counted_twice_beside_every_other_one_is_refused`; `test_regressor_panel.py::test_the_lorentz_refusal_names_the_export_form_alone` |
| The feeder writes d + 1 `hyp_e*` columns | `test_export.py::test_the_hgcn_feeder_writes_d_plus_one_lorentz_columns` |
| MPS: float64 CPU tensors | `test_arm_encoder.py::test_queries_and_codes_on_mps_come_back_float64_on_the_cpu` |

- [ ] **Step 5: The branch carries only this plan**

Run: `git log --oneline origin/main..HEAD`
Expected:
- this plan's commit, the fourteen task commits and their review fixes;
- no "config" or "graph config".

Run: `git diff origin/main...HEAD -- conf/config.yaml`
Expected: only Task 7's edits:
- the `fusion` and `dimension` keys under `model:`;
- the comment on `moe:`.

Run: `git grep -n 'manifest_path:' -- conf/config.yaml`
Expected: `conf/config.yaml:12:  manifest_path: null  # …`. No path is committed.

- [ ] **Step 6: The final review**

Dispatch the code-reviewer agent, which is pinned to Opus, on the whole branch, with:
- `git diff origin/main...HEAD`;
- this plan;
- spec `specs/shared-encoder-and-projection.md`.

Then, for each finding:
- fix it, with a test where it is behavior, and rerun Steps 1–5; or
- triage it as deferred, which Plan completion's gate handles.

## Plan completion (controller, inline)

Run this after Final verification, once the final review's findings are resolved, and before
finishing-a-development-branch. It is writing-plans' Plan Completion Protocol, with this plan's
edits written out. Every "replace" text below occurs exactly once in its file. Under a step that
changed a name, adjust the text to what shipped.

- [ ] **Step 1: Check for parallel sessions**

`specs/naics-embedding-roadmap.md` and `specs/deferred_items.md` are shared by every session.

Run: `git worktree list`
- Worktrees under `~/Projects/copilot-worktrees/` are review tooling, not sessions.
- If another worktree belongs to a running session, hold the shared edits. Stage 6b's session
  may be running. So may any session the user has mentioned. The shared edits are Step 4, Step
  5, and the roadmap part of Step 7.
- When holding, give the user the exact text of each held edit, and continue with the rest.

- [ ] **Step 2: The resolve-before-defer gate**

Collect the leftovers:
- plan steps skipped or descoped during execution;
- final-review findings that were not fixed.

Partition them:
- **Needs the user's input.** Ask now, as one batched set of questions.
- **Unblocked by an answer.** Implement it now, then restart this protocol.
- **Everything else.** Defer it (Step 5).

Unanswered questions block Steps 3–7.

- [ ] **Step 3: Mark up the plan**

In `specs/plans/8-shared-encoder-and-projection.md`:
- Tick every completed step (`- [x]`).
- Under a step that deviated, add a one-line `> Deviation: …` note.
- Under a skipped step, add `> Skipped: <why> → deferred`.
- After the title line, add the status header below.
  - When the gate deferred nothing, end it with `; nothing deferred` instead.
  - Under inline execution, write `executing-plans`.

```markdown
**Status: COMPLETE (YYYY-MM-DD)** — executed via subagent-driven-development; deferred items in specs/deferred_items.md
```

- [ ] **Step 4: The roadmap: tick Stage 6 and re-validate later stages**

In `specs/naics-embedding-roadmap.md`, replace:

```markdown
- [ ] Stage 6: Shared encoder and low-dimensional projection
```

with:

```markdown
- [x] Stage 6: Shared encoder and low-dimensional projection
```

Replace:

```markdown
      its validation split; Stage 2's scorer returns live validation-split numbers under the
      harness's curvature.
      ROUTING: brainstorming
```

with the text below.
- The `Realized:` values come from the finding, one per slot:
  - `<duration>`: section 1's wall-clock time;
  - `<share>`: section 2's share at the cap;
  - the four R² values: section 3's level-6 cells, with `<emb-…>` from the `embedding` row and
    `<txt-…>` from the `text_only` row of each regime's table;
  - `<top-1>` and `<MRR>`: section 4.
- Wrap at 100 columns with the 6-space indent.

```markdown
      its validation split; Stage 2's scorer returns live validation-split numbers under the
      harness's curvature.
      ROUTING: brainstorming
      Rollout note: the switch happens at merge. Main then trains and loads only the shared
      encoder. A four-copy checkpoint has no encoder record and cannot exact-resume, load
      weights-only, export or be read; nothing migrates it (D2). The tokenization cache rebuilds
      once, as `channels-v3`: each present text is marked by its field, and `summaries` stays
      null until Stage 6b. Router mining, the load-balancing term and their logs run only under
      `model.fusion: moe`. Until Stage 7, the interim head caps the tangent at norm 2, and
      `train`'s checkpoint monitor reads the in-sample validation loss, so an arm's reads take
      `last.ckpt` (D6).
      Realized: one MiniLM backbone (revision 1110a243) with one LoRA adapter reads five fields,
      each marker two tokens. The default arm (masked mean, d = 16, c = 1) trained one local
      epoch on MPS in <duration>. Its export holds 2,125 codes, <share> of them at the cap. On
      the validation splits, the regressor panel's level-6 `embedding` comparator reads R²
      <emb-seen> (seen) and <emb-held-out> (held-out), beside `text_only`'s <txt-seen> and
      <txt-held-out>. The outcome panel reads top-1 <top-1> and MRR <MRR> over 4,042 queries.
      These numbers are a floor for Stage 7 (`specs/findings/shared-encoder-first-reading.md`).
      Stage 6: COMPLETE (YYYY-MM-DD) — implemented by plan 8
      (specs/plans/completed/8-shared-encoder-and-projection.md). Next: resume the roadmap.
```

Then tell the later stages what shipped.

**Stage 6b.** Replace:

```markdown
      null until this stage, and its field markers, about two tokens of each window. Stage 2's
```

with:

```markdown
      null until this stage (format `channels-v3`; a cache built under other summaries is
      rebuilt), and its field markers (`text_model/fields.py`), two tokens of each window under
      MiniLM's tokenizer (the field's name and `:`). Stage 2's
```

**Stage 7.** Replace:

```markdown
      Consumes: Stage 6's encoder and query path; Stage 6b's summaries, which the reference
```

with:

```markdown
      Consumes: Stage 6's encoder and query path (`SharedEncoder` in
      `text_model/shared_encoder.py`; `ArmEncoder.from_files` and `read_outcome_validation` in
      `text_model/arm_encoder.py`; `tools export-table` and `tools outcome-panel`), which supply
      `SeedArtifacts`: the checkpoint, the exported table, the `ArmEncoder` and its `distance`.
      Also Stage 6's interim head (`HyperbolicHead`, `text_model/hyperbolic.py`), whose cap at
      norm 2 this stage replaces, and legacy containment, which export and reads refuse and the
      HGCN feeder still serves. Stage 6b's summaries, which the reference
```

**Stage 8.** Replace:

```markdown
      driver; Stage 6's configurable dimension; Stage 2's scorer, whose registered distances are
```

with:

```markdown
      driver; Stage 6's configurable dimension, and its head's `distance` attribute, which
      `ArmEncoder.distance` reads (`HyperbolicHead.distance` is `lorentz`; export and reads
      refuse c ≠ 1, spec R8); Stage 2's scorer, whose registered distances are
```

**Stage 9.** Replace:

```markdown
      Consumes: Stage 8's selected cell; Stage 6's fusion options; Stage 6b's summaries, keyed by
```

with:

```markdown
      Consumes: Stage 8's selected cell; Stage 6's fusion options (`model.fusion`: `masked_mean`,
      `attention`, `moe`) and its one backbone loader (`load_base_model`,
      `text_model/shared_encoder.py`), whose LoRA adapter (`all-linear`) also wraps the pooler's
      dense layer, which mean pooling never reads; Stage 6b's summaries, keyed by
```

**Stage 10.** Replace:

```markdown
      6's export command; Stage 5's bundle as the graph stage's structural input. Under D* its
```

with:

```markdown
      6's export command, and its HGCN feeder (`generate_embeddings_from_checkpoint`), which
      writes d + 1 `hyp_e*` columns through the shared encoder; Stage 5's bundle as the graph
      stage's structural input. Under D* its
```

Stages 11 and 12 consume Stage 6 only through Stages 7–10's artifacts. Read their entries anyway,
and edit one only if what shipped contradicts it.

Run: `python3 -c "lines = open('specs/naics-embedding-roadmap.md').read().split('\n'); print([i + 1 for i, line in enumerate(lines) if len(line) > 100 and not line.startswith('|')])"`
Expected: `[]`. The gap-analysis table's rows are exempt.

- [ ] **Step 5: Deferred items**

In `specs/deferred_items.md`:

1. **The ticking pass.** Plan 6's Stage 6 item is done (Task 1). Replace:

```markdown
- [ ] Review Minor, for Stage 6: (1) `regressor_scores` (src/naics_embedder/decision/scores.py)
```

with:

```markdown
- [x] Review Minor, for Stage 6: (1) `regressor_scores` (src/naics_embedder/decision/scores.py)
```

Replace:

```markdown
      refusal around the export form alone. Size: quick-fix. Done when: Stage 6's export lands
      with both changed.
```

with:

```markdown
      refusal around the export form alone. Size: quick-fix. Done when: Stage 6's export lands
      with both changed.
      → done in plan 8 (Task 1: `regressor_scores` checks each row's distinct repeats as well as
      its count, and `coordinate_matrix`'s refusal names the export form alone).
```

2. **Plan 4's scorer item is not ticked here.** It is retired under `/deferred`, which only the
   user runs (Step 6).

3. **This plan's own items.** Append the gate's deferred items under
   `## 8-shared-encoder-and-projection — YYYY-MM-DD`, newest last. Each item follows the schema
   in writing-plans' `references/deferred-backlog.md`:
   - self-contained: file paths, why it was deferred, what it would take;
   - a `Size:`;
   - a `Done when:` or a `Revisit if:`.

   Skip the section when nothing was deferred.

- [ ] **Step 6: Backlog triage**

Run: `uv run --no-project --python 3.13 python ~/.claude/skills/writing-plans/scripts/deferred_stats.py`
Expected: its summary line, with the open count, the closure rate and the aged tail. Put it in the
completion report.

If the backlog has 20 or more open items or any aged tail, present the read-only triage proposal:
steps 1–4 of the Triage rubric in `references/deferred-backlog.md`.

Either way, propose plan 4's scorer item as **Retire**, backed by this check:
- Run `git grep -n -E '^  curvature:' conf/config.yaml`. It shows `curvature: 1.0`.
- Run `git grep -n 'def require_unit_curvature' src`. It shows the guard in
  `src/naics_embedder/text_model/export.py`.

Its suggested tick, for `/deferred` to apply:

```markdown
      → retired YYYY-MM-DD: its premise, a learned curvature, never held (`loss.curvature` is a
      fixed 1.0); Req 13 keeps c = 1, and plan 8's guard (`require_unit_curvature`,
      src/naics_embedder/text_model/export.py) refuses any other curvature at export and read.
```

Say that `/deferred` acts on the user's selection. Do not apply a disposition here.

- [ ] **Step 7: Commit the completion markup**

Run: `git status --short`
Expected:
- the plan, the roadmap and `specs/deferred_items.md` modified (fewer if Step 1 held the shared
  edits);
- `?? outputs/`, which is never added.

```bash
git add specs/plans/8-shared-encoder-and-projection.md specs/naics-embedding-roadmap.md specs/deferred_items.md
git commit -m "docs(roadmap): complete Stage 6 and re-validate Stages 6b, 7, 8, 9 and 10"
```

- [ ] **Step 8: Retire the plan and the spec**

No other plan implements `specs/shared-encoder-and-projection.md`, so the spec retires with the
plan. Neither file has relative links to re-point.

Run: `git mv specs/plans/8-shared-encoder-and-projection.md specs/plans/completed/8-shared-encoder-and-projection.md`

Run: `git mv specs/shared-encoder-and-projection.md specs/completed/shared-encoder-and-projection.md`

In `specs/completed/shared-encoder-and-projection.md`, replace:

```markdown
**Status:** APPROVED (2026-10-03). Ready for an implementation plan.
```

with the line below, ending in `nothing deferred` when the gate deferred nothing:

```markdown
**Status:** COMPLETE (YYYY-MM-DD) — implemented by
`specs/plans/completed/8-shared-encoder-and-projection.md`; deferred items in
`specs/deferred_items.md`
```

The roadmap names the spec twice by path, so point both at its new place. Skip this if Step 1
held the roadmap edits, and give the user the text instead.

In `specs/naics-embedding-roadmap.md`, replace:

```markdown
(`specs/shared-encoder-and-projection.md`, Rollout note) records the measurements and the three
constraints the entry carries.
```

with:

```markdown
(`specs/completed/shared-encoder-and-projection.md`, Rollout note) records the measurements and
the three constraints the entry carries.
```

Replace:

```markdown
      measured on 2026-10-03 (`specs/shared-encoder-and-projection.md`, Rollout note):
```

with:

```markdown
      measured on 2026-10-03 (`specs/completed/shared-encoder-and-projection.md`, Rollout note):
```

Run: `git grep -n 'specs/shared-encoder-and-projection' -- ':!specs/plans/completed' ':!specs/completed'`
Expected: no output.

```bash
git add specs/plans/completed/8-shared-encoder-and-projection.md specs/completed/shared-encoder-and-projection.md specs/naics-embedding-roadmap.md
git commit -m "chore(specs): retire plan 8"
```

- [ ] **Step 9: Hand off**

Run: `git log --oneline origin/main..HEAD`
Expected:
- this plan's commits;
- no "config" or "graph config".

Then hand off to finishing-a-development-branch. The user chooses how the branch lands. Never
push without asking.
