# Geometry × Dimension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-embedding-roadmap.md, Stage 8 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Build roadmap Stage 8, geometry × dimension. Add Req 12's Euclidean and spherical arms
beside the hyperbolic one, each training, decoding and exporting under its own geometry, with the
radial term in the hyperbolic arm only. Then train the eight cells Stage 7 did not train, 5 seeds
each, on Lambda, and decide among all nine cells under Req 5 against Stage 7's fixed margins. The
decision record names the non-dominated set and the chosen cell, Stage 9's reference.

**Architecture:**

- **Phase 1, the code (Tasks 1–9).** It ends at the first PR's review and merge, a hard
  checkpoint: no Lambda time is spent before it.
  - **The heads (Task 1).** `text_model/heads.py` adds `EuclideanHead`, whose point is v, and
    `SphericalHead`, whose point is û = v / ‖v‖, beside `HyperbolicHead`. The three share one
    interface: `geometry`, `distance` (the scorer's name for the decoding distance), `radial`, a
    `forward` that returns `HeadPoints`, a training `pair_distance` over polar parts, and a
    `read_points` map from exported coordinates to the float64 points the decoding distance
    reads. `exp_map_origin` moves into `text_model/hyperbolic.py`, and `panels/decoding.py`
    maps each geometry to its distance.
  - **Geometry as a factor (Task 2).** `model.geometry`, `hyperbolic` by default, picks the head.
    The checkpoint's encoder record names it, so exact resume, the remote workflow's resume
    check, `CheckpointRunner`, the HGCN feeder and `load_from_checkpoint` all compare it. A record
    saved before Stage 8 reads as hyperbolic. The 21-key `run_settings` is unchanged.
  - **The objective (Task 3).** Both training distances come from the head, and the radial term
    exists only where the head says it applies. The hyperbolic arm's terms and gradients stay
    Stage 7's, bit for bit.
  - **Export and reads (Task 4).** `ArmEncoder` and `LiveEncoder` read through the head. The
    export writes each arm's own coordinates and names its geometry and coordinates in the
    provenance. The HGCN feeder refuses a flat arm, and `train` asks its HGCN question only of a
    hyperbolic arm.
  - **The decision's guards (Task 5).** `run_seed_sweep` refuses a seed whose encoder decodes by
    another distance than its geometry's. `check_arm` compares each logged read's distance and
    dimension with the arm's. `CheckpointRunner.check` refuses monitor reads under another
    distance.
  - **The radius report (Task 6)** checks a flat arm's terms and scales alone and records its
    radius as null.
  - **The Trainer, the docs and the Exit (Tasks 7–9).** Task 7 trains the flat arms through the
    Trainer, and Task 8 documents the arms. The Exit is read-only on Stage 7's artifacts. Every
    reference seed preflights under the new code, and seed 1's radius report and export come out
    identical but for the added geometry keys. The flat arms' terms get gradient on the real
    bundle.
- **Phase 2, the campaign (Tasks 10–16).** It runs from the main checkout after the first PR has
  merged, on local main with the six private commits replayed onto it. Forty runs train on Lambda
  at `bf16-mixed`. The first seed of each cell is gated after its first epoch, and seeds 2–5
  follow. On the Mac, each cell's `tools sweep` writes its arm record. Each run gets its radius
  report and diagnostics. `tools decide` then decides among the eight new records and Stage 7's
  `reference.json`. The finding and the roadmap stamp land in a second PR.

**Tech Stack:**

- Python 3.10 and 3.12 (CI runs both; `.python-version` pins 3.12).
- torch 2.9.1, pytorch-lightning 2.5.5, transformers 4.57.1 and peft 0.17.1, for MiniLM (revision
  1110a243) behind the shared encoder. CI trains only the tiny `BertConfig` backbone.
- polars 1.35.1 (the bundle and the tables), pydantic 2.12.4 (config, contracts and records), numpy.
- typer and rich (the CLI).
- pytest with xdist; ruff and yapf; mkdocs with mkdocstrings (strict build).
- Phase 2 only: the `naics-embedder remote` commands, on a Lambda CUDA instance.

## Global Constraints

Every task's requirements include this section.

### The spec (`specs/naics-embedding.md` at d9126ce), verbatim

Req 12:

> **Req 12 — Geometry and dimension are experimental factors.** There are three geometry arms,
> Euclidean, spherical (cosine) and hyperbolic, at dimensions 8, 16 and 32. Dimension 16 is the
> reference (user adjudication Q1; ChatGPT C4; Claude C3; Gemini on top-level Q2).
>
> - Geometry and dimension are crossed, because their interaction is the question. Any hyperbolic
>   advantage is expected at low dimension (Claude C3, citing Sala et al. 2018 and Nickel & Kiela
>   2018 with links).
> - Arms share the encoder and the objective. The radial term exists only in the hyperbolic arm
>   (Req 11).
> - Each arm decodes by its own distance and exports the coordinates Req 2 names. One affine map
>   projects to the arm's dimension, since the two stacked maps are equivalent to one (methodology
>   S2 procedure).
> - The winner is chosen by Req 5.
> - Fixing 384 dimensions (methodology S2 procedure) (rejected): the downstream use wants low
>   dimension, and on a fixed radius the realized model is spherical anyway (methodology S2
>   limitation 1).

Req 5:

> **Req 5 — Decision rule.** Every comparison of a configuration A against a configuration B
> follows one rule.
>
> - **Seeds.** Each arm runs at least 5 seeds (Claude C20).
> - **Pairing.** Differences Δ = A − B are paired: both arms are scored on the same resample of the
>   evaluation unit, with seeds nested within the resample (Claude on S4-Q4; ChatGPT on S4-Q4). The
>   unit is codes, with their queries, for the outcome panel, and four-digit-parent groups for the
>   regressor panel.
> - **Reference configuration.** Every comparison starts from one named configuration: the text
>   stage after Reqs 7–11, with the current backbone behind a shared encoder (Req 14), in
>   hyperbolic geometry with Req 13 at dimension 16. That is the current method's geometry at the
>   user's reference dimension (user adjudication Q1).
> - **Margin.** Each panel's non-inferiority margin δ is fixed before any arm runs, as a stated
>   multiple of the reference configuration's across-seed standard deviation on that panel
>   (Claude C22).
> - **Adoption.** A is adopted when it is non-inferior on both panels and superior on at least one
>   (Claude C22; user adjudication Q1).
>   - Non-inferior: the lower bound of the 95% interval on Δ exceeds −δ.
>   - Superior: the 97.5% interval excludes zero. Two panels give two chances to adopt, so each
>     gets half the error rate.
> - **Several arms.** Some decisions have more than two arms: the nine geometry × dimension cells,
>   the backbones, and the graph-stage arms A–D. The survivors are the arms no other arm dominates,
>   and the tie order below picks among them. If dominance cycles and every arm is dominated, the
>   tie order picks among all the arms of the decision.
>   - The extra pairwise comparisons within a decision are not corrected for multiplicity. That is
>     a stated choice, not an oversight: the tie order toward the simpler arm is the guard against
>     a spurious win.
> - **Ties.** When A is not adopted over B, the simpler configuration stands. Simpler means fewer
>   components (stages or post-processing steps), then lower dimension, then non-hyperbolic
>   geometry. Any tie left after that goes to the higher regressor-panel estimate, because that is
>   the designed purpose (user adjudication Q1).
>   - Lower dimension ahead of geometry (chosen): the purpose is low dimension, and a hyperbolic
>     embedding that matches a larger flat one is the geometric result worth keeping (ChatGPT on
>     top-level Q2).
>   - Geometry ahead of dimension (rejected).
> - **Scope of the rule.** It replaces the fixed acceptance thresholds (methodology Composition;
>   methodology S4 procedure; ChatGPT C14; Gemini C14) and gates the graph stage (Req 15).
> - A paired t-test across batches (Gemini remediation 12) (rejected): batches are not experimental
>   units. On the fixed code set the metric has no sampling error, and the variance comes from
>   seeds and query sampling (ChatGPT on S4-Q4).

Req 2's opening and its comparators:

> **Req 2 — Regressor panel.** The embedding's coordinates enter a downstream model as regressors:
> tangent coordinates at $o$ for a hyperbolic arm (methodology S4 procedure), raw coordinates
> otherwise. The panel redesigns the defined economic benchmark (methodology S4 procedure;
> methodology S4 limitation 6).

> - **Comparators** share the downstream model and a tuned penalty (ChatGPT C15; Claude C28):
>   - six-digit one-hot indicators;
>   - ancestor indicators at levels 2–5;
>   - a text-only representation reduced to the same dimension;
>   - covariates only;
>   - covariates plus each representation.

Req 13:

> **Req 13 — A live radius in the hyperbolic arm.**
>
> - **No hard norm cap.** Any bound on the radius must pass gradient to it (ChatGPT C4; Claude C2;
>   Gemini C1). On the cap no gradient reaches the norm, and the retained run saturated at $r = 2$
>   (methodology S2 limitation 1).
> - **A virtual root at $o$,** with the 20 sectors at positive radius, so that no level target asks
>   distinct sectors to coincide (ChatGPT C3; Claude C1; E3).
> - **One radial coordinate, $r$,** used in every loss, target and diagnostic (Claude C5;
>   methodology Cross-component 5).
> - **Curvature fixed at 1,** not a parameter anywhere; the nominal per-layer curvature is removed
>   (ChatGPT on top-level Q5; Claude C5). Distance formulas are written for c = 1 only
>   (methodology Cross-component 4).
>   - A learnable global curvature (Gemini C15) (rejected): against a learned logit scale,
>     curvature only rescales (Claude C5), and today it receives no gradient (methodology S3
>     limitation 3).
> - **Radii stay where the working precision resolves distances.** This is checked numerically,
>   not set from a derived bound (Claude C4; methodology S3 limitation 2).

Verification:

> - [ ] **No inert terms.** On a real batch, every objective term has a nonzero gradient.

> - [ ] **Geometry × dimension.** All nine cells run with at least 5 seeds, and the chosen cell has
>       its Req 5 record.

D8 below supersedes Req 5's two panels and its 97.5 % interval: Req 5 reads three panels, with
the 98⅓ % interval for superiority.

### The roadmap (`specs/naics-embedding-roadmap.md` at 699a5d2), verbatim

Stage 8:

> - [ ] Stage 8: Geometry × dimension
>       Objective: Implement the Euclidean and spherical arms and run the crossed nine-cell
>       comparison under Req 5.
>       Spec: Req 12; Req 5 (several arms, ties); Req 2 (export form per arm); Verification
>       "Geometry × dimension"; D8; D9.
>       Gap closed: Req 12 (geometry arms and the decision).
>       Consumes: Stage 7's reference configuration and δ; Stage 4's tooling and seed-sweep
>       driver; Stage 6's configurable dimension, and its head's `distance` attribute, which
>       `ArmEncoder.distance` reads (`HyperbolicHead.distance` is `lorentz`; Plan 10 removed
>       text curvature and the export/read guard). Only that name follows the head: `ArmEncoder`
>       maps queries and codes through the hyperbolic exp map at the origin whatever the head
>       (`text_model/arm_encoder.py`), so the Euclidean and spherical arms need their own maps
>       there. Stage 2's scorer, whose registered distances are
>       `euclidean`, `cosine` and `lorentz` at curvature −1 (`panels/decoding.py`).
>       Produces: Geometry as a configuration factor with per-arm distance, decoding and export (the
>       radial term only in the hyperbolic arm; every arm's export writes the provenance
>       `tools export-table` writes, tokenizer and summaries included, which `run_seed_sweep` and
>       `ArmEncoder.from_files` check); guards on the two tie-order keys Stage 4
>       leaves open: `run_seed_sweep` refuses an arm whose `SeedArtifacts.distance` does not fit
>       its `ArmSpec.geometry`, and `decide` compares each read's logged `dimension` with the
>       arm's (the decision fixture's reads hard-code 16); the nine-cell decision record; the
>       selected cell as the reference for Stage 9.
>       Exit: All nine cells have at least 5 seeds scored on D8's three panels; the decision
>       record names the non-dominated set and the chosen cell under the tie order; each arm's
>       export uses tangent coordinates for hyperbolic and raw coordinates otherwise.
>       ROUTING: writing-plans

The resume after Stage 7:

> **Resume after Stage 7 (2026-10-07).** Stage 7's completion stamp is authoritative: Plan 10
> Phase 2 is complete, published by PR #128 at public main `7687dfc`. Live GitHub checks now show
> all four jobs passed on both the PR and the public merge: lint, Python 3.10 tests, Python 3.12
> tests and strict docs. The merge's separate docs deployment also passed. This supersedes the
> pending-CI snapshot in `logs/plan10_campaign_evidence/stage7-postmerge-handoff.md` without
> rewriting that receipt or `phase2-completion-handoff.md`.
>
> Local main remains `aa7ac39`, seven commits ahead of public main: all six original private
> commits plus their merge descendant. Its two parents are private tip `2d1f045` and public merge
> `7687dfc`. All six private IDs remain reachable locally and absent from public main; the five
> published documents match the public merge. Source, configuration, the private Lambda spec and
> the lock are unchanged from the private tip. Never push local main, any private commit or its
> descendant. Public native readiness remains a separately authorized follow-up in
> `specs/deferred_items.md`; public CI does not expand the qualification boundary of training
> source `b94ecf3` and Mac-read source `2d1f045`.
>
> The reference and margin files still match the Stage 7 handoff's full hashes (`c885b9c5…` and
> `e619b3b3…`). `uv.lock` still has SHA-256
> `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21`; keep it frozen through
> the last Stages 8–10 decision using these margins. Stages 8–10 train on Lambda at `bf16-mixed`
> and read on the Mac, through `tools sweep` and `CheckpointRunner`. All seeds preflight before
> any export or decision read. Preserve exact settings, seed, contract, LoRA and active-MoE
> controls; continue only from last at the same absolute checkpoint path and user, restoring all
> kept checkpoints and both histories together. Finished runs stay finished. Every hyperbolic arm
> through Stage 10 passes `tools radius-report`; stop on failure without changing tolerances.
>
> Stages 8–12 were re-validated against shipped interfaces and Plan 10 section 10. Stage 8 still
> needs Euclidean and spherical heads, training distances, export and query maps, plus the two
> decision identity guards; text curvature and its guard have already been removed. Stage 9 owns
> R7's term-weight decision from the selected cell. R10 overrides Stage 5's historical removal
> timing: contract v2 retains training pairs, unread by text training, and D5's margin axis remains
> only in their generation until the next contract bump, alongside the tokenizer-revision watch.
> Stages 10 and 11 retain their graph-specific work and decision dependency; no graph repair is
> pulled into Stage 8. Stage 12 consumes `SeedRun.monitor_records`, `training_run` and
> `checkpoint_epoch`, and keeps every sealed set closed until its final evaluation. No campaign,
> export, panel read or sealed opening ran during this resume. Stage 8 routes to writing-plans;
> the next plan number is 12.

D8, D9, D10 and D11:

> - **D8 — Regressor regimes under Req 5 (Stages 4 and 7; every later decision).** Req 2 reports
>   the seen-code and held-out-code regimes separately, but Req 5 counts two panels, each with its
>   own δ and half the error rate, and ends its tie order on "the higher regressor-panel estimate".
>   The finding runs both regimes. Decision: each regime counts as a panel under Req 5, which then
>   has three: the outcome panel and the two regressor regimes. Adoption needs non-inferiority on
>   all three (the 95 % interval, unchanged) and superiority on at least one; each panel gets its
>   own δ and a third of the error rate, so superiority reads the 98⅓ % interval. This supersedes
>   the 97.5 % that Req 5 and Verification "Decision records" name; the Completion audit reads it
>   as this recorded deviation, not an unmet requirement. The final tie-break stayed open; D11
>   settles it.

> - **D9 — Req 2's text-only comparator (Stage 3; Stages 8 and 9 re-run it).** Req 2 names no
>   representation, and review C28, its source, lists two: a frozen encoder and TF-IDF. Decision:
>   the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the arm's
>   dimension. That backbone is the current checkpoint until Stage 9 adopts another, and whichever
>   backbone the arm uses after. The comparator measures what taxonomy training adds over the same
>   encoder reading the same text.

> - **D10 — Each panel's decision statistic (Req 5; Stage 4; Stages 7–12 read it).** Req 5 fixes a
>   δ and an interval per panel but names no statistic. Decision: the outcome panel's is per-query
>   MRR, resampled by code with its queries, the statistic D6 already selects checkpoints on. Each
>   regressor regime's is the out-of-sample mean squared error of the `covariates+embedding`
>   comparator on log employment at level 6, each row's squared error averaged over its repeats
>   (five on a validation read) before resampling by group; its Δ is oriented so that a positive
>   value favours A (B's error minus A's). The sparse comparators never read the arm, and their
>   folds depend only on the regime, level, repeat and group, so the gain over a sparse encoding
>   (over one-hot in the seen regime, over ancestors in the held-out regime) gives the same paired
>   Δ; that gain is reported for Req 1. The text-only comparator is no baseline, since it changes
>   with each arm's dimension and backbone. Every other Req 2 comparator and Req 3 metric is still
>   reported.
> - **D11 — D8's final tie-break (Req 5; Stage 4).** Req 5's tie order ends on "the higher
>   regressor-panel estimate", and D8 gives two. Decision: the held-out regime's, read as its gain
>   over ancestors, so the higher estimate is the lower error. For a held-out code, one-hot
>   predicts only the intercept and ancestors help only through levels 2–3, so that regime is where
>   the embedding's value over sparse encodings is tested.

### Deferred items this plan touches (`specs/deferred_items.md` at 699a5d2), verbatim

None is discharged. Each is quoted with how this plan treats it.

> - [ ] Review Minor (plan-mandated; deferred by the user): `tie_order`'s `TieUnresolvedError`
>       (src/naics_embedder/decision/rule.py) fires on a tie anywhere among the surviving arms,
>       even when first place is clear, so two identical arms entered under different names block
>       a decision whose simpler third arm is obvious. Kept strict so a recorded order is never
>       arbitrary. Fix: raise only when the first two survivors tie. Size: quick-fix. Revisit if:
>       a tie below first place blocks a decision (Stage 8's nine cells are the first multi-arm
>       decision).

Kept strict. Task 15's nine-arm decision is the item's trigger. If `tools decide` fails with
`TieUnresolvedError`, stop and ask (**Stop-and-ask conditions**); never change the rule during the
decision.

> - [ ] HGCN feeder precision (completion gate candidate): `generate_embeddings_from_checkpoint` in
>       src/naics_embedder/cli/commands/training.py still writes float32 head Lorentz points. The text
>       campaign uses bounded tangent export and CPU float64 reconstruction, so this did not affect
>       its panels. Stage 11 owns the feeder. Size: plan. Done when: the feeder writes the float64
>       origin map of the bounded tangent and fixture tests check manifold residuals at large observed
>       radii, without changing the text objective.

Stage 11 still owns it. P10 adds only a refusal of a flat arm, raised before anything is read, and
the hyperbolic feeder path is unchanged.

> - [ ] Large-radius overflow watch (completion gate candidate): `polar_distance` in
>       src/naics_embedder/text_model/hyperbolic.py multiplies float32 sinh terms, while
>       `model.radius_bound` in src/naics_embedder/utils/config.py accepts any finite positive value.
>       Overflow becomes possible around R = 44; this campaign used R = 8 and every all-pairs report
>       passed. Size: quick-fix. Revisit if: an arm proposes a bound near or above that range; then
>       restrict the supported bound or add a numerically stable form and a meaningful boundary test
>       before that arm trains.

Not triggered: every hyperbolic cell keeps R = 8, and no arm proposes another bound.

> - [ ] Older-kept-checkpoint continuation (completion gate candidate): CLI exact resume can accept a
>       kept epoch older than `last.ckpt` through src/naics_embedder/utils/training.py and
>       cli/commands/training.py, rewinding monitor and summary histories. The approved operator path
>       resumes only last, and no campaign run used this rewind. Size: plan. Done when: exact
>       continuation refuses a checkpoint below the experiment last epoch before pruning or model/data
>       construction, with tests for unchanged last continuation and preserved histories on refusal.

Not triggered: Phase 2 continues a run only through `remote train --resume`, which resumes from
`last.ckpt`, as Stage 7 did.

> - [ ] Imported monitor-record strictness (Task 12, completion gate candidate): decision/records.py
>       permits boolean `checkpoint_epoch` and whitespace-only `training_run`; decision/decide.py
>       compares `detail.seed` by value, and the `SeedRun.monitor_records` docstring says oldest first
>       rather than file order. Producer records in all ten audited runs are well formed. Size: plan.
>       Done when: src/naics_embedder/decision/records.py and decide.py reject these malformed
>       imported identities before panel reads, retain valid producer records, and document file
>       order, with focused refusal tests.
> - [ ] Margin-timing error shape (completion gate candidate): `check_margins_first` in
>       src/naics_embedder/decision/decide.py fails closed with KeyError for a missing read time and
>       TypeError for a naive time. No campaign record has either shape. Deferred as imported-record
>       error handling. Size: quick-fix. Done when: it validates presence and timezone awareness and
>       raises a named ValueError before comparison, with tests covering both decision and monitor
>       reads and valid aware times.

Untouched. Task 5 adds a distance check beside these checks and changes none of the behaviours
they name. Stage 8's records are producer records with timezone-aware read times.

> - [ ] Public native-qualification reconciliation (Phase 2 source boundary): the actual campaign
>       source b94ecf3 and Mac-read source 2d1f045 include approved local-only corrections to
>       remote/model_cache.py, canonical.py, loop.py, worker.py, launch.py and workflow.py. Public
>       base a3e82f1 lacks them. The finding explicitly qualifies only the recorded private source.
>       Held configuration, these private commits and their descendants must never be pushed. Size:
>       plan. Revisit if: native readiness is claimed for a public release or later campaign; use a
>       separately authorized fresh public-base change and review/qualification, preserving the
>       private configuration and campaign artifacts.

Not triggered. Phase 2 trains and reads on the private source, as Stage 7 did: merged origin/main
plus the six private commits, never pushed. The finding records that source's SHAs and claims no
public native readiness.

One more item is partly met in passing. Plan completion annotates it and leaves it open:

> - [ ] Review Minor: the export and the arm encoder's reads have untested branches.
>       In src/naics_embedder/text_model/export.py: the `batch_size < 1` refusal (:95), an absent
>       curvature reading as 1 (:131), the claim that a refused table is never written (the
>       curvature test in tests/unit/test_export.py asserts nothing about it), the exact provenance
>       key set (`coordinates` and `generated_at` go unchecked), and a cap check that cannot tell a
>       capped table from an uncapped one. In src/naics_embedder/cli/commands/tools.py:
>       `export-table`'s ValidationError and OSError branches (:903), and `outcome-panel`'s
>       bare-override and legacy refusals, its failure on a bad `--output` before the read is
>       logged, and its payload's `fingerprint` key (:958-984). `exp_map_origin`'s `.cpu()`
>       (src/naics_embedder/text_model/arm_encoder.py:57) never meets an MPS tensor in the tests:
>       removing it fails nothing. Deferred from plan 8's final review as coverage gaps, not
>       defects. Size: plan. Done when: each branch has a test, or a recorded ruling that it needs
>       none.
>       → partly mooted by plan 10: the curvature branches left with `require_unit_curvature`
>       (Task 11), and the cap check is a check that the table holds the head's bounded tangent
>       (Task 4). The other branches stay open.

Task 4's test checks each geometry's `coordinates` in the provenance. `generated_at`, the exact key
set and the other branches stay open.

### Decisions already made (do not re-ask)

The brief that ordered this plan gave these rulings, verbatim:

> Rulings already made, do not re-triage: text curvature and its export/read guard are gone (Plan
> 10). Radial term only in the hyperbolic arm. Export writes tangent coordinates for hyperbolic and
> raw coordinates otherwise, with full export-table provenance. No graph repair in Stage 8. Stage 7
> reference and δ = 3 SD margins are fixed. uv.lock stays frozen at sha256 4167042e8a5a8caa…
> through the last Stage 8–10 decision. Training on Lambda bf16-mixed, reads on the Mac via tools
> sweep / CheckpointRunner, all seeds preflight first, and every hyperbolic arm must pass tools
> radius-report.

The user answered two questions at this plan's checkpoint (2026-10-08):

- **The Euclidean head is the identity.** The point is v, the projection's output, with no bound
  and no normalization. Its export is v (P4).
- **5 seeds per new cell.** That is Req 5's floor. Eight new cells × 5 seeds = 40 Lambda runs. The
  hyperbolic d = 16 cell is Stage 7's reference record, with its 10 seeds. Req 5 allows unequal
  seed counts: `check_pairing` pairs arms on their panels, and the resampling nests each arm's own
  seeds.

### This plan's decisions

Each goes beyond a spec line, or reads one. Do not reopen them during execution. Task numbers name
where each lands.

- **P1. Two phases and a hard checkpoint.** Phase 1 (Tasks 1–9) is the code and a read-only Exit.
  It ends at the first PR's review and merge, and no Lambda time is spent before it. Phase 2
  (Tasks 10–16) is the campaign and the finding. It runs from the main checkout, in a fresh
  session.
- **P2. A heads module (Task 1).** The flat heads, their distances, `GEOMETRIES`, `build_head` and
  `head_of` live in `text_model/heads.py`, which imports `text_model/hyperbolic.py`.
  `exp_map_origin` moves from `arm_encoder.py` to `hyperbolic.py`, so `HyperbolicHead.read_points`
  calls it without an import cycle: `arm_encoder.py` imports the heads. `arm_encoder.py`,
  `monitor.py` and `radius_report.py` import it from `hyperbolic.py`.
- **P3. One head interface (Task 1).** Callers never branch on the geometry; they ask the head.
  - Every head has the class attributes `geometry`, `distance` and `radial`. `distance` is a
    `panels.decoding.DISTANCES` name.
  - `forward(v)` returns `HeadPoints(tangent, embedding, radius, direction)`.
  - The static `pair_distance(radius_a, direction_a, radius_b, direction_b)` is the training
    distance, (A, B) in the inputs' dtype.
  - The static `read_points(tangent)` maps exported coordinates to the float64 CPU points that
    `distance` reads.
- **P4. The Euclidean head is the identity (user).**
  - Its point, tangent and embedding are v. Its radius is ‖v‖ and its direction û.
  - Its training distance, `flat_distance`, is ‖v_a − v_b‖ in the polar form
    √((r_a − r_b)² + r_a · r_b · ‖û_a − û_b‖²). It takes explicit differences and guards its square
    root, so zero separation has a finite gradient and float32 resolves a milliradian.
  - It decodes by `euclidean`. Its read map is the identity, in float64 on the CPU.
- **P5. The spherical arm exports û.**
  - The cosine distance reads û alone, so v's norm gets no training signal and is no part of the
    arm's geometry. The export writes û, on the unit sphere, and the `cosine` decoding distance
    reads it unchanged.
  - Its radius is 1, or 0 at v = 0, with no gradient.
  - Its training distance, `chord_distance` = ‖û_a − û_b‖² / 2, equals the cosine distance
    1 − û_a · û_b, computed without cancellation.
  - This reads Req 2's "raw coordinates otherwise" as the arm's own point. The first PR flags it
    for review.
- **P6. The radial term in the hyperbolic arm only (Req 12; Task 3).**
  - `compute_losses` computes `radial_loss` only when `head.radial`. Otherwise `StepLosses.radial`
    is None, the total leaves it out, and the health logs skip it.
  - In the hyperbolic arm the radial term and the total are computed in Stage 7's order. Its
    autograd graph is built in the same order, so its values and gradients are Stage 7's, bit for
    bit. Task 3's total test uses `torch.equal`, and Task 9 checks the claim on the real seed-1
    checkpoint.
- **P7. The encoder record carries the geometry (Task 2).**
  - `EncoderArchitecture.geometry`, `hyperbolic` by default, joins fusion, dimension and backbone.
    `shared_encoder_architecture` requires it, so no caller leaves it out of the record it
    compares.
  - A checkpoint saved before Stage 8 has no geometry key and reads as hyperbolic. Stage 7's
    checkpoints, records and tables load unchanged.
  - `run_settings` keeps its 21 keys and gains no geometry key. The encoder record already reaches
    every consumer that compares architectures, and a new key would break Stage 7's exact-settings
    identity.
- **P8. The distance and dimension guards (Task 5).** These are the roadmap's two guards, plus one.
  - `run_seed_sweep` refuses a seed whose `SeedArtifacts.distance` is not its geometry's
    (`check_seed_distance`).
  - `check_arm` compares each outcome read's logged `distance`, each regressor read's logged
    `dimension` and each monitor record's `distance` with the arm's.
  - `CheckpointRunner.check` refuses a monitor record under another distance, before any export
    (**Recorded deviations**).
- **P9. The provenance names the geometry (Task 4).** The export provenance adds `geometry` and
  the contract's `encoder.geometry`, and `coordinates` names each geometry's form
  (`COORDINATES`). The hyperbolic arm's `coordinates` string is Stage 7's, byte for byte. So
  Stage 7's tables and a re-export differ only in `geometry`, `contract.encoder.geometry` and
  `generated_at` (Task 9).
- **P10. The HGCN feeder refuses a flat arm (Task 4).** `generate_embeddings_from_checkpoint`
  writes Lorentz points, and a flat arm has none. It raises before it reads the bundle. `train`
  skips its HGCN-embeddings question for a flat arm and logs why. HGCN refines the hyperbolic arm
  only, and the feeder's precision item stays Stage 11's.
- **P11. A flat arm's radius report (Task 6).** `tools radius-report` checks the hyperbolic arm as
  before. A flat arm has no radial term and no live-radius head. Its report records `radius` as
  null and checks the task term, the code-code term and both logit scales for nonzero gradient
  (Verification "No inert terms"). It passes on that alone. Every report names its geometry.
- **P12. The campaign (Phase 2).** Eight new cells × 5 seeds = 40 runs, under Stage 7's settings
  with only `model.geometry` and `model.dimension` changed. The hyperbolic d = 16 cell is
  `reference.json`, not retrained.
- **P13. Names.**
  - An arm is `<geometry>-d<dimension>`: `euclidean-d8` through `spherical-d32`.
  - A run's experiment is `stage8-<geometry>-d<dimension>-s<seed>`. Its checkpoints are in
    `checkpoints/stage8-<geometry>-d<dimension>-s<seed>/`.
  - The records are in `~/naics-artifacts/records/stage8/`.
  - The reference arm keeps its name, `reference`.
- **P14. The first seed of each cell is gated (Task 11).** A cell's seeds 2–5 wait until its seed 1
  is checked.
  - After its first epoch: CUDA `bf16-mixed`, the encoder record's geometry and dimension, and
    every monitor read under the cell's distance and in the pulled selection logs.
  - After it ends: under a flat geometry, its epoch summary has no `loss/radial`.
- **P15. A read-only Phase 1 Exit (Task 9).** No training, no decision read and no sealed opening.
  - Stage 7's ten seeds preflight under the new code (`CheckpointRunner.check` and
    `_check_monitor_records`).
  - The reference record and margins pass `check_arm` and `check_margins_first`.
  - Seed 1's radius report and export come out identical but for the added geometry keys.
  - The flat arms' terms and scales get gradient on the real bundle's first batch, at d = 8.
  - Plan 9's text-only table reduces to d = 8, 16 and 32 under one fingerprint, so the d = 8 and
    d = 32 cells' regressor reads can pair with the reference's (D9).

### Recorded deviations

- **No standalone bitwise unit test (P6).** Two checks carry the hyperbolic arm's bit-for-bit
  claim instead of a golden-value unit test. Task 3's total test uses `torch.equal`. Task 9's
  real-artifact radius report reproduces Stage 7's 25 anchor gradients and five term norms
  exactly.
- **A third guard (P8).** Beyond the roadmap's two, `CheckpointRunner.check` refuses monitor reads
  under another distance, so a mislaunched cell fails at preflight, before any export.
- **Two added keys (P9, P11).** The radius report and the export provenance gain `geometry`.
  Stage 7's saved files lack it, and nothing reads it from them.
- **The spherical export (P5).** Req 2's "raw coordinates otherwise" is read as each flat arm's own
  point: v for Euclidean, û for spherical.
- **Task 10 replays six private commits.** CLAUDE.md's Plan 10 paragraph names "the two held
  private config commits". Local main now holds six: the two configs and four approved local-only
  corrections, on which Stage 7 trained and read. Stage 8's source is merged origin/main plus all
  six, as Stage 7's was.

### Project rules

- **Style (CLAUDE.md).**
  - Single quotes, including `'''` docstrings. A string with an apostrophe takes double quotes
    (ruff Q003).
  - YAPF owns layout (100 columns), and ruff lints (E, F, I, Q). **Never run `ruff format`.**
  - One blank line between top-level definitions and after imports.
  - Semantic section dividers; `logging` rather than `print`; type hints on signatures.
  - Code, comments and docstrings cite the spec (`Req 12`) or a P-number (`P7`), never a task
    number, which means nothing once the plan retires.
- **Formatting.** Format touched files with `./scripts/format_code.sh <files>`. The plan's code is
  already formatted, so the script should change nothing. If it changes a file, keep its layout
  and say so in the ledger. At the end, `./scripts/format_code.sh --check --all` must pass.
- **Git.**
  - Never push without the user's go-ahead, and never push to `main`.
  - Six commits are private: aa8ebd6 `config`, 3fc580a `graph config`, d7927a5, 5ce248f, b94ecf3
    and 2d1f045. Never push them or any descendant, and never cherry-pick or merge them into a
    public branch. Local `main` is a descendant.
  - The main checkout's local pre-push hook refuses the private commits. Never bypass it with
    `--no-verify`. Push a public branch explicitly: `git push -u origin <branch>`, never a bare
    `git push`.
  - Never run bare `git stash`.
  - Commit on this plan's branches only, and end each message with the session's attribution
    trailer.
  - Add files by explicit path, never `git add -A` or `git add .`.
- **Data safety.**
  - Tasks 1–8 write nothing in the main checkout. Task 9 reads its `data/` and `checkpoints/` and
    `~/naics-artifacts` read-only, and writes its outputs under `/tmp`. Its reads may build
    `data/token_cache/` there, the gitignored preprocessing cache built from the canonical inputs,
    and append to `logs/tools_*.log`. Nothing else in the main checkout may change.
  - These are canonical: bundle 301cce28, `data/naics_descriptions.parquet`,
    `checkpoints/plan9_exit/`, `checkpoints/stage7-reference-s*/` and
    `~/naics-artifacts/records/stage7/`. Never edit, move or rebuild them.
  - Tests write only under `tmp_path`.
- **The selection log.** `logs/selection_log.jsonl` is append-only (Req 4): never delete, move or
  edit it, in any checkout. Phase 1 appends nothing to it. Phase 2's sweeps append 120 records
  (Task 13).
- **Sealed splits.**
  - No step opens a split: no `OutcomePanel.open_test`, `RegressorPanel.open_outer` or
    `--split test`.
  - Phase 1 reads no panel at all. Task 9 builds the outcome panel only for its fingerprint.
  - Phase 2's monitor reads score the outcome validation split during training, on Lambda.
    Phase 2's sweeps read the three validation panels once per seed, on the Mac.
- **Configs.** `conf/config.yaml` keeps `supervision.manifest_path: null` on every public branch.
  Phase 2 passes the manifest as a `key=value` override on every command.
- **The lock.** `uv.lock` has sha256
  `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21`. It stays frozen through
  the last Stage 8–10 decision. No task changes it. If a step would, stop and ask.
- **Lambda.** No Lambda time before the first PR has merged (P1). Launching and terminating
  instances is the user's action.
- **Downloads.**
  - Never download Census or QCEW files.
  - The backbone and its tokenizer come from the local Hugging Face cache. Run every command that
    loads them under `HF_HUB_OFFLINE=1`.
- **Devices.** On MPS, cast with `.cpu().to(torch.float64)`, never `.to(device='cpu', dtype=…)`,
  which raises on MPS.
- **Deferred items.** Promote no open item of `specs/deferred_items.md` beyond those named above.
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-embedding-roadmap.md` and `specs/deferred_items.md`, and hand the user the exact
  edit instead.
- **Docs.**
  - `uv run mkdocs build --strict` must pass locally, because PR CI never builds the docs.
  - Griffe is strict. Every documented parameter gets its own `Args:` entry, and a docstring's
    `Returns:` bullets stay one line each: a wrapped bullet warns "Confusing indentation".
- **Tests.**
  - Rich wraps at 80 columns on CI, so CLI tests match on `result.output.replace('\n', '')`, and
    usage errors on `click.unstyle(result.output).replace('\n', '')`.
  - pyproject's `addopts` has `-v`, so a node-ID listing needs `--collect-only -q -q`.
  - New tests use the tiny `BertConfig` backbone (`tests/fixtures/shared_encoder.py`) and the
    fixture bundles (`tests/fixtures/supervision.py`). Nothing downloads a model.
- **Bash tool.** It runs zsh.
  - Brace variables (`${FILE}`): `$FILE:t` is a zsh modifier.
  - Quote `=`-leading words (`echo '====='`) and globs (`--include='*.py'`); an unquoted one
    aborts the command.
  - If the tool refuses a heredoc or a compound command, run one plain command per call and
    write files with the Write tool.

## Workspace

- **Execution ledger:** `logs/plan12_execution.md` in the Phase 1 worktree, ignored by git. Write
  Phase 1's commands, outputs and deviations there. The hard checkpoint's Step 4 copies it to the
  main checkout. Phase 2's ledger is `logs/plan12_campaign.md` in the main checkout.
- **Worktree (Phase 1):**
  `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-12-geometry`, or the app worktree
  this session runs in. Run every command from its root. Task 9 calls its
  absolute path WORKTREE: substitute it, as `git rev-parse --show-toplevel` prints it there.
- **Branch (Phase 1):** `claude/stage-8-geometry-arms`, cut at `origin/main` once this plan has
  merged (**Pre-flight** Step 1).
  - An app worktree is cut from local `main`, which carries the six private commits. So the branch
    is always re-cut from `origin/main`, never from local `main`.
  - Otherwise, from the main checkout, run
    `git worktree add --detach .claude/worktrees/plan-12-geometry origin/main`, then cut the branch
    there.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder`. Phase 1 checks nothing out there and
  writes nothing there but Task 9's permitted cache and tool logs (**Project rules**, Data
  safety).
  - At plan time it is on `claude/plan-12-geometry` at 699a5d2, clean.
  - Local `main` is 22df5e1, the merge of public 699a5d2 into cd0f80b. cd0f80b sits on merge
    aa7ac39 over the six private commits, and is patch-equivalent to PR #129. Task 10 drops all
    three: aa7ac39, cd0f80b and 22df5e1.
  - origin/main is 699a5d2 (PR #129), plus this plan's PR once merged.
- **Real inputs, read only (Task 9):**
  - The bundle:

```text
/Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/
```

  - Stage 7's ten runs, `checkpoints/stage7-reference-s1/` through `-s10/` in the main checkout.
    Seed 1's selected checkpoint is `epoch=003.ckpt`, beside `arm_table_epoch=003.parquet`, its
    provenance and `radius_report.json`.
  - Stage 7's records: `~/naics-artifacts/records/stage7/reference.json` (sha256
    `c885b9c5dc18b6be03670d0cb5a71db3974917da8f719e0dbb1f1ef5eec2d1a2`) and `margins.json` (sha256
    `e619b3b30fcad07ab95b23c4f7cfcba52327db011a02a378874d7017ce10fd2f`), and the store
    `~/naics-artifacts`.
  - Plan 9's text-only table, `checkpoints/plan9_exit/text_only.parquet` (sha256
    `f4fb4f574f880c940ee8cf44c7f86615638e59f113b527d0b86973cd45ba32ff`). D9 reduces it to each
    arm's dimension at read time, so one table serves every cell.
  - The backbone and tokenizer:
    `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2` (revision
    1110a243).
- **Outputs (Task 9):** under `/tmp` only, deleted at the task's end:
  `/tmp/naics_plan12_rr_s1.json`, `/tmp/naics_plan12_s1.parquet` and its provenance, and the
  task's scripts.
- **Phase 2** runs in the main checkout, on local `main` (Phase 2's rules).
- **Working directory.** The Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before every commit. If it is not this worktree, `cd` back first, or
  prefix the command with `cd <worktree> &&`. Task 9 runs from the main checkout on purpose and
  says so.

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A red step passes, or fails other than as its Expected line says, and the cause is not a
  transcription slip.
- A step would do any of these:
  - during Phase 1, write the main checkout's tracked files, or its `data/`, `checkpoints/` or
    `logs/` beyond Task 9's permitted cache and tool logs and the hard checkpoint's ledger copy;
  - commit `supervision.manifest_path`, or change `uv.lock`;
  - push a private commit or a descendant, or push with `--no-verify`;
  - open a sealed split, or read a panel during Phase 1;
  - download a Census or QCEW file, or spend Lambda time before the first PR has merged.
- `origin/main` gains a commit that touches a file in **File structure**, the roadmap or
  `specs/deferred_items.md`, or an open PR does.
- In Task 9, any check fails:
  - a Stage 7 seed's preflight refuses;
  - the reference's spec, panel fingerprint or records fail a check;
  - the radius report or the export differs from Stage 7's beyond the named keys;
  - a flat arm's term or scale has zero or non-finite gradient;
  - the selection log changes.
- In Phase 2, the conditions its tasks name, and `TieUnresolvedError` from `tools decide`.

## File structure

The complete file map for Phase 1. Task 9 produces only uncommitted outputs under `/tmp`. Phase 2
creates `specs/findings/geometry-by-dimension.md` (Task 16), and Plan completion edits the roadmap
and `specs/deferred_items.md`. `AGENTS.md` is a symlink to `CLAUDE.md` and needs no edit of its
own.

| Action | Path | Tasks | Responsibility |
|---|---|---|---|
| Create | `src/naics_embedder/text_model/heads.py` | 1 | The flat heads, `flat_distance`, `chord_distance`, `GEOMETRIES`, `build_head`, `head_of` |
| Modify | `src/naics_embedder/text_model/hyperbolic.py` | 1 | `HyperbolicHead`'s interface; `exp_map_origin` moves here |
| Modify | `src/naics_embedder/panels/decoding.py` | 1 | `GEOMETRY_DISTANCES` |
| Modify | `src/naics_embedder/text_model/arm_encoder.py` | 1, 4 | The moved import (1); reads through the head (4) |
| Modify | `src/naics_embedder/text_model/monitor.py` | 1, 4 | The moved import (1); `LiveEncoder` reads through the head (4) |
| Modify | `src/naics_embedder/text_model/radius_report.py` | 1, 6 | The moved import (1); a flat arm's term gradients (6) |
| Modify | `conf/config.yaml` | 2 | `model.geometry: hyperbolic`, and which keys a flat arm ignores |
| Modify | `src/naics_embedder/utils/config.py` | 2 | `ModelConfig.geometry` |
| Modify | `src/naics_embedder/supervision/checkpoints.py` | 2 | `EncoderArchitecture.geometry`; `shared_encoder_architecture(geometry=)` |
| Modify | `src/naics_embedder/text_model/shared_encoder.py` | 2 | `SharedEncoder(geometry=)` builds the head |
| Modify | `src/naics_embedder/text_model/naics_model.py` | 2, 3 | The geometry hyperparameter and contract (2); the objective under each head (3) |
| Modify | `src/naics_embedder/text_model/mixins/logging.py` | 3 | Health logs skip a term the arm lacks |
| Modify | `src/naics_embedder/cli/commands/training.py` | 2, 4 | Geometry into the model and banner (2); the feeder refusal and the prompt (4) |
| Modify | `src/naics_embedder/cli/commands/tools.py` | 2, 6 | `_sweep_spec` geometry (2); `radius-report` for a flat arm (6) |
| Modify | `src/naics_embedder/text_model/checkpoint_runner.py` | 2, 5 | The expected encoder record (2); the monitor-distance check (5) |
| Modify | `src/naics_embedder/text_model/export.py` | 4 | `COORDINATES`, the provenance's geometry, the feeder's refusal |
| Modify | `src/naics_embedder/decision/decide.py` | 5 | `check_seed_distance`; distance and dimension checks on logged reads |
| Modify | `src/naics_embedder/decision/sweep.py` | 5 | `run_seed_sweep` calls `check_seed_distance` |
| Create | `tests/unit/test_heads.py` | 1, 2 | The heads, their distances and read maps; one list of geometries |
| Modify | `tests/unit/test_arm_encoder.py` | 1, 4 | The moved import (1); a flat arm's read (4) |
| Modify | `tests/unit/test_monitor.py` | 1, 4 | The moved import (1); the live read against the export, per geometry (4) |
| Modify | `tests/fixtures/shared_encoder.py` | 2, 4 | `pre_stage8_checkpoint` (2); `five_code_model`, `geometry_checkpoint` (4) |
| Modify | `tests/unit/test_checkpoint_contract.py` | 2 | The encoder record's geometry; a pre-Stage-8 record |
| Modify | `tests/unit/test_checkpoint_runner.py` | 2, 5 | Another geometry refused (2); monitor reads under another distance (5) |
| Modify | `tests/unit/test_cli_training.py` | 2, 4 | Geometry into the model, banner and sweep spec (2); no HGCN question (4) |
| Modify | `tests/unit/test_config.py` | 2 | `model.geometry`'s default and choices |
| Modify | `tests/unit/test_encoder.py` | 2 | The geometry picks the head |
| Modify | `tests/unit/test_export.py` | 2, 4 | The provenance's encoder record (2); coordinates and the feeder (4) |
| Modify | `tests/unit/test_naics_model.py` | 2, 3 | Geometry and contract (2); the terms under each head (3) |
| Modify | `tests/fixtures/decision.py` | 5 | Logged reads name the arm's distance and dimension |
| Modify | `tests/unit/test_decision.py` | 5 | `check_seed_distance`; a flat arm's logged reads |
| Modify | `tests/unit/test_decision_sweep.py` | 5 | A seed of another distance refused before any read |
| Modify | `tests/unit/test_radius_report.py` | 6 | A flat arm's report |
| Modify | `tests/integration/test_reference_training.py` | 7 | The flat arms through the Trainer |
| Modify | `docs/text_training.md` | 8 | The geometry arms, the head, the terms, the export and the report |
| Modify | `docs/usage.md` | 8 | `export-table`, `tools sweep` and `radius-report` per geometry |
| Modify | `docs/overview.md`, `docs/quickstart.md`, `README.md` | 8 | The three arms |
| Modify | `docs/api/encoder.md`, `docs/api/export.md`, `docs/api/radius_report.md` | 8 | The heads' API page section; the export and report intros |
| Modify | `CLAUDE.md`, `tests/README.md` | 8 | Counts, the tree, the architecture and the test seams |

## Pre-flight (controller, inline, before Task 1)

- [ ] **Step 1: Confirm the workspace**

Run: `git fetch origin`, then:

```bash
git log --oneline --reverse origin/main -- specs/plans/12-geometry-by-dimension.md
```
Expected: this plan's commits, oldest first: `docs(plans): add plan 12, geometry × dimension`,
then its review follow-ups (`docs(plans): …`). If there is no output, this plan has not merged:
stop and ask.

Run: `git checkout --no-track -B claude/stage-8-geometry-arms origin/main`
Expected: `Switched to a new branch 'claude/stage-8-geometry-arms'` (or `Reset branch`).

Run: `git status --short --branch`, then `git log --oneline origin/main..HEAD`
Expected: `## claude/stage-8-geometry-arms` and nothing else, then no output. None of the six
private commits is on this branch.

Run: `git log --oneline 699a5d2..origin/main -- src tests conf docs specs CLAUDE.md README.md`
Expected: this plan's commits, and possibly PR #130's merge commit; no other commit.
- If anything else landed, read it.
- If it touches a file in **File structure**, the roadmap or `specs/deferred_items.md`, stop and
  ask.

Run: `gh pr list --state open`
Expected: no open PR touching a file in **File structure**. If one does, stop and ask.

- [ ] **Step 2: Build the worktree's environment**

Run: `uv sync --locked`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number.

Run:

```bash
uv run python -c "import peft, polars, pydantic, pytorch_lightning, torch, transformers; print(peft.__version__, polars.__version__, pydantic.__version__, pytorch_lightning.__version__, torch.__version__, transformers.__version__)"
```
Expected: `0.17.1 1.35.1 2.12.4 2.5.5 2.9.1 4.57.1`.

Run: `shasum -a 256 uv.lock`
Expected: `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21  uv.lock`. If it
differs, stop and ask: the lock is frozen (**Project rules**).

- [ ] **Step 3: Run the baseline suite**

Run: `uv run pytest -n auto -q`
Expected: `2999 passed, 2 skipped`, measured at 699a5d2 on 2026-10-08 in a worktree without
`data/`. One skip needs CUDA. The other is plan 9's local-only test, which needs
`data/naics_descriptions.parquet`. The main checkout, which has that file, shows `3000 passed,
1 skipped`.
- Each later full-suite run must pass with those skips, and each task gives its count.
- The warnings count varies under xdist; ignore it.

The full suite after each task, measured on this Mac (MPS present, no `data/` in the worktree):

| After | Passed | Skipped |
|---|---:|---:|
| Pre-flight (699a5d2) | 2999 | 2 |
| Task 1 | 3023 | 2 |
| Task 2 | 3053 | 2 |
| Task 3 | 3059 | 2 |
| Task 4 | 3071 | 2 |
| Task 5 | 3082 | 2 |
| Task 6 | 3086 | 2 |
| Task 7 | 3090 | 2 |
| Task 8 | 3090 | 2 |

After Task 8, `uv run pytest --collect-only -q -q | tail -1` reports 3,092 tests. CI has no MPS,
so more tests skip there.

- [ ] **Step 4: Check Task 9's real inputs, read-only**

Task 9 is Phase 1's only reader of these files. Check them now, so a missing input fails early.

Run:

```bash
shasum -a 256 /Users/lowell/naics-artifacts/records/stage7/reference.json /Users/lowell/naics-artifacts/records/stage7/margins.json /Users/lowell/Projects/naics-embedder/checkpoints/plan9_exit/text_only.parquet
```
Expected, in order: `c885b9c5dc18b6be03670d0cb5a71db3974917da8f719e0dbb1f1ef5eec2d1a2`,
`e619b3b30fcad07ab95b23c4f7cfcba52327db011a02a378874d7017ce10fd2f` and
`f4fb4f574f880c940ee8cf44c7f86615638e59f113b527d0b86973cd45ba32ff`.

Run:

```bash
ls /Users/lowell/Projects/naics-embedder/checkpoints/stage7-reference-s1 /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```
Expected: the manifest path, then seed 1's directory listing `arm_table_epoch=003.parquet`,
`arm_table_epoch=003_provenance.json`, `epoch=003.ckpt`, `last.ckpt`, `monitor_reads.jsonl`,
`epoch_summary.jsonl` and `radius_report.json`, among others.

Run: `cat ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main`
Expected: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Run:

```bash
wc -l /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl && shasum -a 256 /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl
```
Record both in the ledger. At plan time they were 37 lines and
`654e2232f8a10eb406cdab580f771575a75a972842f02232d73637a8976a4a66`. If they differ, another session
has read a panel since: record it, and use the new values in Task 9.

- [ ] **Step 5: Route the tasks**

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–7 each get a fresh implementer and a task-reviewer. Give each implementer its task,
  **Global Constraints** and **Workspace**.
- Every code block in a task is exact, so an implementer copies it. Each "Replace" text occurs
  exactly once in its file when its edit is made, and the edits of one file are made in the order
  given. Task 7 makes one file's edits across two steps: keep that order too.
- Task 8 (documentation) may go to the docs-writer agent. Its acceptance checks are the gate.
- Task 9, **Final verification, Phase 1** and the hard checkpoint run inline in the controller
  session.
- Phase 2 runs inline in the controller session, from the main checkout.

## Phase 1: the code (Tasks 1–9)

Phase 1 runs in this plan's worktree (**Workspace**), task by task. It ends at the first PR's
review and merge (**The hard checkpoint**), and no Lambda time is spent before it. Phase 2 starts
later, in a fresh session, from the main checkout.

Each task's red step was measured at plan time on the task's parent commit with the task's test
edits applied, and each green step on the task's own commit. The counts are for the named test
files only. A **Files** list's line ranges locate its Replace texts in the file as the task finds
it, before its edits.

### Task 1: The geometry heads

Req 12's arms share the encoder and the objective, but each has its own head. The head maps the
projection's output v to the arm's point, radius and direction, and owns the arm's training
distance and read map. This task adds the two flat heads beside `HyperbolicHead` and gives all
three one interface (P3), so no later caller branches on the geometry.

- `EuclideanHead`'s point is v (P4), and `SphericalHead`'s is û (P5).
- Each head's `pair_distance` reads the polar parts (r, û) that the code cache already stores.
  Its `read_points` maps exported coordinates to the float64 points the scorer's distance reads.
- `HyperbolicHead` gains the same attributes, with its existing maps: `pair_distance` is
  `polar_distance`, and `read_points` is `exp_map_origin`, which moves here from `arm_encoder.py`
  (P2).
- `panels/decoding.py` gains `GEOMETRY_DISTANCES`, each geometry's decoding distance.

Nothing builds a flat head yet: Task 2 wires the geometry into the encoder.

The tests cover:
- one list of geometries, shared by every layer that names one: the records' `Geometry`, the
  diagnostics' `GEOMETRIES` and the scorer's distances (Task 2 adds the config and the encoder
  record);
- each head's names, the flat heads' points, and the zero vector's finite gradients;
- each training distance against the scorer's distance on the same vectors;
- float32 resolution at a milliradian, where 1 − û_a · û_b would cancel;
- zero separation, refused unpaired shapes, and each read map's float64 CPU points.

**Files:**
- Modify: `src/naics_embedder/panels/decoding.py`, lines 61-66
- Modify: `src/naics_embedder/text_model/arm_encoder.py`, lines 39-66
- Create: `src/naics_embedder/text_model/heads.py` (new)
- Modify: `src/naics_embedder/text_model/hyperbolic.py`, lines 62-67, 72-76, 103-108, 111-119,
  161-164
- Modify: `src/naics_embedder/text_model/monitor.py`, lines 34-43
- Modify: `src/naics_embedder/text_model/radius_report.py`, lines 11-16
- Test: `tests/unit/test_arm_encoder.py`, lines 18-30
- Test: `tests/unit/test_heads.py` (new)
- Test: `tests/unit/test_monitor.py`, lines 26-34, 39-42

**Interfaces:**
- Consumes: nothing from an earlier task. It uses `HeadPoints`, `polar_distance`,
  `_where_positive` and `_refuse_unpaired_shapes` in `text_model/hyperbolic.py`, and `DISTANCES`,
  `euclidean_distances` and `cosine_distances` in `panels/decoding.py`. It also uses
  `decision.records.Geometry` and `metrics.diagnostics.GEOMETRIES`, which already name the three
  geometries.
- Produces:
  - In `naics_embedder.text_model.heads`:
    - `GEOMETRIES = ('euclidean', 'spherical', 'hyperbolic')`.
    - `flat_distance(radius_a, direction_a, radius_b, direction_b) -> torch.Tensor` and
      `chord_distance(...)`, each (A, B) from radii (A,), (B,) and directions (A, d), (B, d). Each
      raises `ValueError('<name> takes radii (A,) and (B,) …')` on unpaired shapes.
    - `EuclideanHead` and `SphericalHead`, `nn.Module`s with no parameters. Their class
      attributes are `geometry`, `distance` (`'euclidean'`, `'cosine'`) and `radial = False`.
      `forward(vectors)` returns `HeadPoints`. The static `pair_distance` is `flat_distance` or
      `chord_distance`, and the static `read_points(tangent)` returns `tangent` as float64 on the
      CPU.
    - `build_head(geometry: str, *, radius_bound: float = 8.0) -> nn.Module`. It raises
      `ValueError("unknown geometry 'poincare'; expected one of ['euclidean', 'spherical',
      'hyperbolic']")` for an unknown name.
    - `head_of(model) -> nn.Module`: a Lightning module's `encoder.head`, or a shared encoder's
      `head`.
  - In `naics_embedder.text_model.hyperbolic`:
    - `HyperbolicHead.geometry = 'hyperbolic'`, `.distance = 'lorentz'` and `.radial = True`.
    - The static `HyperbolicHead.pair_distance` calls `polar_distance`, and the static
      `HyperbolicHead.read_points` calls `exp_map_origin`.
    - `exp_map_origin(tangent) -> torch.Tensor`, (N, d + 1) float64 on the CPU, moved from
      `arm_encoder.py` with its behavior unchanged.
    - `_refuse_unpaired_shapes(*tensors, name='polar_distance')`, which now names its caller.
  - `naics_embedder.panels.decoding.GEOMETRY_DISTANCES = {'euclidean': 'euclidean',
    'spherical': 'cosine', 'hyperbolic': 'lorentz'}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_heads.py` is new. The two other test files import `exp_map_origin` from its new
home.

In `tests/unit/test_arm_encoder.py`, make one edit.

Replace:

```python
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.arm_encoder import (
    ArmEncoder,
    exp_map_origin,
    read_outcome_validation,
)
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.export import encode_token_rows
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.text_model.hyperbolic import HyperbolicHead
from tests.fixtures.shared_encoder import (
    ARM_DIMENSION,
```

with:

```python
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.export import encode_token_rows
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.text_model.hyperbolic import HyperbolicHead, exp_map_origin
from tests.fixtures.shared_encoder import (
    ARM_DIMENSION,
```

Create `tests/unit/test_heads.py`:

```python
'''
The geometry heads of Req 12's three arms (``text_model/heads.py``): their points, training
distances and read maps, and the one list of geometries that every layer names.
'''

from types import SimpleNamespace
from typing import get_args

import pytest
import torch

from naics_embedder.decision import records
from naics_embedder.metrics import diagnostics
from naics_embedder.panels.decoding import (
    DISTANCES,
    GEOMETRY_DISTANCES,
    cosine_distances,
    euclidean_distances,
)
from naics_embedder.text_model.heads import (
    GEOMETRIES,
    EuclideanHead,
    SphericalHead,
    build_head,
    chord_distance,
    flat_distance,
    head_of,
)
from naics_embedder.text_model.hyperbolic import HyperbolicHead, exp_map_origin, polar_distance

pytestmark = pytest.mark.unit

HEADS = {'euclidean': EuclideanHead, 'spherical': SphericalHead, 'hyperbolic': HyperbolicHead}

def _vectors(count: int = 6, *, seed: int = 0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 8, generator=generator, dtype=torch.float64).to(dtype)

# -------------------------------------------------------------------------------------------------
# One list of geometries
# -------------------------------------------------------------------------------------------------

def test_every_layer_names_the_same_three_geometries():
    assert GEOMETRIES == ('euclidean', 'spherical', 'hyperbolic')
    assert tuple(GEOMETRY_DISTANCES) == GEOMETRIES
    assert get_args(records.Geometry) == GEOMETRIES
    assert diagnostics.GEOMETRIES == GEOMETRIES
    assert set(GEOMETRY_DISTANCES.values()) == set(DISTANCES)

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_each_head_names_its_geometry_its_distance_and_whether_the_radial_term_applies(geometry):
    head = build_head(geometry)

    assert isinstance(head, HEADS[geometry])
    assert head.geometry == geometry
    assert head.distance == GEOMETRY_DISTANCES[geometry]
    # The radial term exists only in the hyperbolic arm (Req 12)
    assert head.radial is (geometry == 'hyperbolic')
    assert list(head.parameters()) == []

def test_build_head_gives_the_bound_to_the_hyperbolic_head_and_refuses_an_unknown_geometry():
    assert build_head('hyperbolic', radius_bound=5.0).radius_bound == 5.0
    with pytest.raises(ValueError, match="unknown geometry 'poincare'"):
        build_head('poincare')

def test_head_of_finds_the_head_of_a_module_or_of_its_encoder():
    encoder = SimpleNamespace(head=SphericalHead())

    assert head_of(SimpleNamespace(encoder=encoder)) is encoder.head
    assert head_of(encoder) is encoder.head

# -------------------------------------------------------------------------------------------------
# The flat heads' points
# -------------------------------------------------------------------------------------------------

def test_the_euclidean_point_is_the_vector_itself_with_its_norm_and_direction():
    vectors = _vectors()

    points = EuclideanHead()(vectors)

    assert torch.equal(points.tangent, vectors)
    assert torch.equal(points.embedding, vectors)
    torch.testing.assert_close(points.radius, vectors.norm(dim=1), rtol=1e-15, atol=0.0)
    torch.testing.assert_close(
        points.radius.unsqueeze(1) * points.direction, vectors, rtol=1e-15, atol=1e-15
    )

def test_the_spherical_point_is_the_unit_direction_at_radius_one():
    vectors = _vectors()

    points = SphericalHead()(vectors)

    direction = vectors / vectors.norm(dim=1, keepdim=True)
    torch.testing.assert_close(points.tangent, direction, rtol=1e-15, atol=1e-15)
    assert torch.equal(points.embedding, points.tangent)
    assert torch.equal(points.direction, points.tangent)
    assert torch.equal(points.radius, torch.ones(len(vectors), dtype=torch.float64))
    # No term reads the radius as a quantity to train: it takes no gradient
    assert not SphericalHead()(vectors.clone().requires_grad_()).radius.requires_grad

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_the_zero_vector_has_radius_zero_and_every_gradient_there_is_finite(geometry):
    vectors = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)

    points = build_head(geometry)(vectors)

    assert torch.equal(points.radius.detach(), torch.zeros(2, dtype=torch.float64))
    assert torch.equal(points.direction.detach(), torch.zeros(2, 4, dtype=torch.float64))
    (gradient, ) = torch.autograd.grad(sum(part.sum() for part in points), vectors)
    assert torch.isfinite(gradient).all()

# -------------------------------------------------------------------------------------------------
# Training distances
# -------------------------------------------------------------------------------------------------

def test_the_flat_distance_is_the_euclidean_distance_of_the_vectors():
    head = EuclideanHead()
    vectors_a, vectors_b = _vectors(5, seed=1), _vectors(7, seed=2)
    a, b = head(vectors_a), head(vectors_b)

    distances = head.pair_distance(a.radius, a.direction, b.radius, b.direction)

    expected = euclidean_distances(vectors_a, vectors_b)
    torch.testing.assert_close(distances, expected, rtol=1e-12, atol=1e-12)

def test_the_chord_distance_is_the_cosine_distance_of_the_vectors():
    head = SphericalHead()
    vectors_a, vectors_b = _vectors(5, seed=1), _vectors(7, seed=2)
    a, b = head(vectors_a), head(vectors_b)

    distances = head.pair_distance(a.radius, a.direction, b.radius, b.direction)

    expected = cosine_distances(vectors_a, vectors_b)
    torch.testing.assert_close(distances, expected, rtol=1e-9, atol=1e-12)

def test_the_hyperbolic_heads_distance_and_read_map_are_the_polar_distance_and_the_exp_map():
    head = HyperbolicHead()
    points = head(_vectors(5, dtype=torch.float32))

    distances = head.pair_distance(points.radius, points.direction, points.radius, points.direction)

    expected = polar_distance(points.radius, points.direction, points.radius, points.direction)
    assert torch.equal(distances, expected)
    assert torch.equal(head.read_points(points.tangent), exp_map_origin(points.tangent))

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_float32_resolves_two_points_a_milliradian_apart(geometry):
    '''
    The flat distances take explicit differences, so float32 resolves a small angle that
    1 − û_a · û_b would cancel: at 1e-3 rad its float32 relative error is about 0.3.
    '''

    first = torch.zeros(1, 8, dtype=torch.float64)
    first[0, 0] = 1.0
    second = torch.zeros(1, 8, dtype=torch.float64)
    second[0, 0], second[0, 1] = torch.cos(torch.tensor(1e-3)), torch.sin(torch.tensor(1e-3))
    head = build_head(geometry)
    a, b = head((5.0 * first).float()), head((5.0 * second).float())

    distance = head.pair_distance(a.radius, a.direction, b.radius, b.direction).double()

    exact = DISTANCES[head.distance](5.0 * first, 5.0 * second)
    assert ((distance - exact).abs() / exact).item() <= 1e-3
    if geometry == 'spherical':
        cancelled = (1 - a.direction @ b.direction.T).double()
        assert ((cancelled - exact).abs() / exact).item() > 1e-2

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_zero_separation_has_distance_zero_and_a_finite_gradient(geometry):
    '''Each anchor meets its own live row among the candidates (spec 4.3).'''

    head = build_head(geometry)
    vectors = _vectors(3, dtype=torch.float32).requires_grad_()
    points = head(vectors)

    distances = head.pair_distance(points.radius, points.direction, points.radius, points.direction)

    assert torch.equal(distances.diagonal().detach(), torch.zeros(3))
    assert (distances.detach() + torch.eye(3) > 0).all()
    (gradient, ) = torch.autograd.grad(distances.sum(), vectors)
    assert torch.isfinite(gradient).all()

@pytest.mark.parametrize('distance', [flat_distance, chord_distance])
def test_a_flat_distance_refuses_unpaired_shapes(distance):
    radius, direction = torch.ones(3), torch.ones(3, 4)

    with pytest.raises(ValueError, match=f'{distance.__name__} takes radii'):
        distance(radius.unsqueeze(1), direction, radius, direction)
    with pytest.raises(ValueError, match=f'{distance.__name__} takes radii'):
        distance(radius, direction, radius, direction[:, :2])

# -------------------------------------------------------------------------------------------------
# Read maps
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_the_read_map_takes_an_export_to_the_points_the_heads_distance_reads(geometry):
    head = build_head(geometry)
    points = head(_vectors(4, dtype=torch.float32))

    read = head.read_points(points.tangent)

    assert (read.dtype, read.device.type) == (torch.float64, 'cpu')
    if geometry == 'hyperbolic':
        assert torch.equal(read, exp_map_origin(points.tangent))
    else:
        assert torch.equal(read, points.tangent.double())
    # The read lands where the head put its point
    torch.testing.assert_close(read, points.embedding.double(), rtol=1e-5, atol=1e-6)
```

In `tests/unit/test_monitor.py`, make these 2 edits, in order.

Replace:

```python
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.text_model import monitor
from naics_embedder.text_model.arm_encoder import (
    ArmEncoder,
    exp_map_origin,
    read_outcome_validation,
)
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
```

with:

```python
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
from naics_embedder.text_model import monitor
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
```

Replace:

```python
    export_code_table,
)
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import TokenizationConfig
```

with:

```python
    export_code_table,
)
from naics_embedder.text_model.hyperbolic import exp_map_origin
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import TokenizationConfig
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_heads.py tests/unit/test_arm_encoder.py tests/unit/test_monitor.py -n auto -q
```
Expected: FAIL, `3 errors`, all at collection:
- `tests/unit/test_heads.py`: `ImportError: cannot import name 'GEOMETRY_DISTANCES' from
  'naics_embedder.panels.decoding'`;
- `tests/unit/test_arm_encoder.py` and `tests/unit/test_monitor.py`: `ImportError: cannot import
  name 'exp_map_origin' from 'naics_embedder.text_model.hyperbolic'`.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/panels/decoding.py`, make one edit.

Replace:

```python
    'euclidean': euclidean_distances,
    'cosine': cosine_distances,
    'lorentz': lorentz_distances,
}

def resolve_distance(distance: Union[str, DistanceFn]) -> Tuple[str, DistanceFn]:
```

with:

```python
    'euclidean': euclidean_distances,
    'cosine': cosine_distances,
    'lorentz': lorentz_distances,
}
# Each of Req 12's geometry arms decodes by its own distance: the name its head and its reads use
GEOMETRY_DISTANCES: Dict[str, str] = {
    'euclidean': 'euclidean',
    'spherical': 'cosine',
    'hyperbolic': 'lorentz',
}

def resolve_distance(distance: Union[str, DistanceFn]) -> Tuple[str, DistanceFn]:
```

In `src/naics_embedder/text_model/arm_encoder.py`, make one edit.

Replace:

```python
    load_arm_model,
)
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
```

with:

```python
    load_arm_model,
)
from naics_embedder.text_model.hyperbolic import exp_map_origin
from naics_embedder.utils.config import TokenizationConfig

# -------------------------------------------------------------------------------------------------
```

Create `src/naics_embedder/text_model/heads.py`:

```python
'''
The geometry heads of Req 12's three arms: Euclidean, spherical and hyperbolic.

Every head takes the projection's output v (B, d) and returns ``HeadPoints``:

- ``tangent``: the coordinates the export writes, Req 2's form of the arm. The hyperbolic head
  writes its bounded tangent r · û, the Euclidean head v and the spherical head û;
- ``embedding``: the point the arm's decoding distance reads. It is the Lorentz point (B, d + 1)
  under hyperbolic and ``tangent`` itself otherwise;
- ``radius`` and ``direction``: the polar parts that the code cache keeps and the training
  distance reads.

Each head also names its ``geometry`` and its decoding ``distance`` (a ``panels.decoding``
``DISTANCES`` name). ``radial`` says whether the radial term applies, which it does only in the
hyperbolic arm (Req 12). ``pair_distance`` is its training distance over polar parts, (A, B) in
the inputs' dtype. ``read_points`` maps exported coordinates to the float64 CPU points its
distance reads. The hyperbolic head is ``hyperbolic.HyperbolicHead``; this module adds the two
flat heads and builds a head by name.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
import torch.nn as nn

from naics_embedder.text_model.hyperbolic import (
    HeadPoints,
    HyperbolicHead,
    _refuse_unpaired_shapes,
    _where_positive,
)

# Req 12's geometry arms, in the order of model.geometry's choices
GEOMETRIES = ('euclidean', 'spherical', 'hyperbolic')

# -------------------------------------------------------------------------------------------------
# The flat distances
# -------------------------------------------------------------------------------------------------

def flat_distance(
    radius_a: torch.Tensor,
    direction_a: torch.Tensor,
    radius_b: torch.Tensor,
    direction_b: torch.Tensor,
) -> torch.Tensor:
    '''
    The Euclidean distance ‖v_a − v_b‖ between every point of one set and every point of another,
    from each point's radius r = ‖v‖ and direction û.

    The polar form is exact:

        ‖v_a − v_b‖ = √((r_a − r_b)² + r_a · r_b · ‖û_a − û_b‖²)

    Every term is non-negative, so nothing cancels, and ‖û_a − û_b‖² comes from explicit
    differences, as in ``polar_distance``. The square root is guarded at zero, where an anchor
    meets its own live row: the distance is 0, and the value and its gradient stay finite.

    Args:
        radius_a: The first set's radii, (A,).
        direction_a: Its directions, (A, d): unit vectors, or 0 at the origin.
        radius_b: The second set's radii, (B,).
        direction_b: Its directions, (B, d).

    Returns:
        The distances, (A, B), in the inputs' dtype.

    Raises:
        ValueError: If the shapes are not (A,), (A, d), (B,) and (B, d).
    '''

    _refuse_unpaired_shapes(radius_a, direction_a, radius_b, direction_b, name='flat_distance')
    gap = radius_a.unsqueeze(1) - radius_b.unsqueeze(0)
    chord = (direction_a.unsqueeze(1) - direction_b.unsqueeze(0)).square().sum(dim=2)
    squared = gap.square() + radius_a.unsqueeze(1) * radius_b.unsqueeze(0) * chord
    separated = squared > 0
    return torch.where(
        separated, torch.sqrt(_where_positive(squared, separated)), torch.zeros_like(squared)
    )

def chord_distance(
    radius_a: torch.Tensor,
    direction_a: torch.Tensor,
    radius_b: torch.Tensor,
    direction_b: torch.Tensor,
) -> torch.Tensor:
    '''
    The cosine distance 1 − cos θ between every direction of one set and every direction of
    another, in chord form: ‖û_a − û_b‖² / 2.

    For unit vectors the two are equal, and the chord form takes explicit differences, so it
    resolves small angles in float32, where 1 − û_a · û_b cancels. The radii are not read: on the
    sphere every point is at distance 1 from the origin. The gradient is finite everywhere, zero
    separation included.

    Args:
        radius_a: The first set's radii, (A,); checked for shape only.
        direction_a: Its directions, (A, d).
        radius_b: The second set's radii, (B,); checked for shape only.
        direction_b: Its directions, (B, d).

    Returns:
        The distances, (A, B), in the inputs' dtype.

    Raises:
        ValueError: If the shapes are not (A,), (A, d), (B,) and (B, d).
    '''

    _refuse_unpaired_shapes(radius_a, direction_a, radius_b, direction_b, name='chord_distance')
    return (direction_a.unsqueeze(1) - direction_b.unsqueeze(0)).square().sum(dim=2) / 2

# -------------------------------------------------------------------------------------------------
# The flat heads
# -------------------------------------------------------------------------------------------------

def _polar_parts(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Each vector's norm (B, 1) and unit direction (B, d); the direction is 0 where v is 0.'''

    norm = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    # Guarded by where, not a clamp: at v = 0 the direction and its gradient are 0, not 1/ε
    moving = norm > 0
    direction = torch.where(
        moving, vectors / _where_positive(norm, moving), torch.zeros_like(vectors)
    )
    return norm, direction

class EuclideanHead(nn.Module):
    '''
    The Euclidean head (Req 12): the point is the projection's output v itself.

    Its radius is r = ‖v‖ and its direction û = v / ‖v‖ (0 at v = 0): the polar parts the code
    cache keeps. The training distance is ‖v_a − v_b‖ in polar form (``flat_distance``). The
    export writes v, and a read takes it as it is, in float64. The head has no parameters and no
    bound, so ``model.radius_bound`` is not read, and the radial term does not apply (P4).
    '''

    geometry = 'euclidean'
    distance = 'euclidean'
    radial = False

    def forward(self, vectors: torch.Tensor) -> HeadPoints:
        '''
        The point v, with its radius ‖v‖ and direction û, in the vectors' dtype.

        Args:
            vectors: The projection's outputs, (B, d).

        Returns:
            v as both the tangent and the embedding, the radius and the direction.
        '''

        norm, direction = _polar_parts(vectors)
        return HeadPoints(vectors, vectors, norm.squeeze(1), direction)

    @staticmethod
    def pair_distance(
        radius_a: torch.Tensor,
        direction_a: torch.Tensor,
        radius_b: torch.Tensor,
        direction_b: torch.Tensor,
    ) -> torch.Tensor:
        '''The training distance between two sets of the head's points: ``flat_distance``.'''

        return flat_distance(radius_a, direction_a, radius_b, direction_b)

    @staticmethod
    def read_points(tangent: torch.Tensor) -> torch.Tensor:
        '''Exported points as they are, float64 on the CPU.'''

        # .cpu() before the cast: casting an MPS tensor to float64 raises
        return tangent.cpu().to(torch.float64)

class SphericalHead(nn.Module):
    '''
    The spherical head (Req 12): the point is the direction û = v / ‖v‖ on the unit sphere.

    Its radius is the constant 1, or 0 at v = 0, and takes no gradient: every point on the sphere
    is at distance 1 from the origin, and the radial term does not apply. The training distance is
    the chord form of the cosine distance (``chord_distance``). The export writes û, and a read
    takes it as it is, in float64. It writes û rather than v because, under the cosine distance,
    v's norm receives no training signal, so the regressor panel would read an untrained norm
    (P5).
    '''

    geometry = 'spherical'
    distance = 'cosine'
    radial = False

    def forward(self, vectors: torch.Tensor) -> HeadPoints:
        '''
        The point û, with the constant radius 1 (0 at v = 0), in the vectors' dtype.

        Args:
            vectors: The projection's outputs, (B, d).

        Returns:
            û as the tangent, the embedding and the direction, and the radius.
        '''

        norm, direction = _polar_parts(vectors)
        radius = (norm.squeeze(1) > 0).to(vectors.dtype)
        return HeadPoints(direction, direction, radius, direction)

    @staticmethod
    def pair_distance(
        radius_a: torch.Tensor,
        direction_a: torch.Tensor,
        radius_b: torch.Tensor,
        direction_b: torch.Tensor,
    ) -> torch.Tensor:
        '''The training distance between two sets of the head's points: ``chord_distance``.'''

        return chord_distance(radius_a, direction_a, radius_b, direction_b)

    @staticmethod
    def read_points(tangent: torch.Tensor) -> torch.Tensor:
        '''Exported directions as they are, float64 on the CPU.'''

        # .cpu() before the cast: casting an MPS tensor to float64 raises
        return tangent.cpu().to(torch.float64)

# -------------------------------------------------------------------------------------------------
# A head by name
# -------------------------------------------------------------------------------------------------

def build_head(geometry: str, *, radius_bound: float = 8.0) -> nn.Module:
    '''
    The head of a geometry arm (``model.geometry``).

    Args:
        geometry: One of ``GEOMETRIES``.
        radius_bound: R, read by the hyperbolic head only (``model.radius_bound``).

    Returns:
        The head, with no parameters.

    Raises:
        ValueError: If the geometry is unknown, or as ``HyperbolicHead`` for its bound.
    '''

    if geometry == 'hyperbolic':
        return HyperbolicHead(radius_bound=radius_bound)
    if geometry == 'euclidean':
        return EuclideanHead()
    if geometry == 'spherical':
        return SphericalHead()
    raise ValueError(f'unknown geometry {geometry!r}; expected one of {list(GEOMETRIES)}')

def head_of(model: nn.Module) -> nn.Module:
    '''
    A model's geometry head: the Lightning module's encoder's, or a shared encoder's own.

    Args:
        model: The Lightning module, or its shared encoder.

    Returns:
        The head.
    '''

    return getattr(model, 'encoder', model).head
```

In `src/naics_embedder/text_model/hyperbolic.py`, make these 5 edits, in order.

Replace:

```python
    R = 8. The interim cap at norm 2 passed about 1e-7 at its saturated points. In float32 the
    derivative rounds to 0 only from ν ≈ 8.7R, where tanh rounds to 1. At v = 0 the direction and
    the radius are 0, so the point is the origin, and every gradient there is finite. ``distance``
    names the decoding distance for its points.

    Args:
```

with:

```python
    R = 8. The interim cap at norm 2 passed about 1e-7 at its saturated points. In float32 the
    derivative rounds to 0 only from ν ≈ 8.7R, where tanh rounds to 1. At v = 0 the direction and
    the radius are 0, so the point is the origin, and every gradient there is finite.

    It is Req 12's hyperbolic arm, the one arm with the radial term. ``geometry`` names the arm,
    ``distance`` the decoding distance for its points and ``radial`` whether the radial term
    applies; ``pair_distance`` is its training distance, and ``read_points`` maps exported tangents
    to the points its distance reads. The flat arms' heads are in ``text_model/heads.py``.

    Args:
```

Replace:

```python
    '''

    distance = 'lorentz'

    def __init__(self, radius_bound: float = 8.0):
```

with:

```python
    '''

    geometry = 'hyperbolic'
    distance = 'lorentz'
    # The radial term exists only in the hyperbolic arm (Req 12)
    radial = True

    def __init__(self, radius_bound: float = 8.0):
```

Replace:

```python
        return HeadPoints(tangent, embedding, radius.squeeze(1), direction)

def _refuse_unpaired_shapes(*tensors: torch.Tensor) -> None:
    '''Refuse anything but radii (A,) and (B,) with directions (A, d) and (B, d).'''

    shapes = [tuple(tensor.shape) for tensor in tensors]
```

with:

```python
        return HeadPoints(tangent, embedding, radius.squeeze(1), direction)

    @staticmethod
    def pair_distance(
        radius_a: torch.Tensor,
        direction_a: torch.Tensor,
        radius_b: torch.Tensor,
        direction_b: torch.Tensor,
    ) -> torch.Tensor:
        '''The training distance between two sets of the head's points: ``polar_distance``.'''

        return polar_distance(radius_a, direction_a, radius_b, direction_b)

    @staticmethod
    def read_points(tangent: torch.Tensor) -> torch.Tensor:
        '''Exported bounded tangents as Lorentz points, float64 on the CPU: ``exp_map_origin``.'''

        return exp_map_origin(tangent)

def _refuse_unpaired_shapes(*tensors: torch.Tensor, name: str = 'polar_distance') -> None:
    '''Refuse anything but radii (A,) and (B,) with directions (A, d) and (B, d), for ``name``.'''

    shapes = [tuple(tensor.shape) for tensor in tensors]
```

Replace:

```python
        (count_a, ), (rows_a, width_a), (count_b, ), (rows_b, width_b) = shapes
        paired = (rows_a, rows_b, width_a) == (count_a, count_b, width_b)
    if not paired:
        raise ValueError(
            'polar_distance takes radii (A,) and (B,) with directions (A, d) and (B, d); got '
            + ', '.join(str(shape) for shape in shapes)
        )

def polar_distance(
```

with:

```python
        (count_a, ), (rows_a, width_a), (count_b, ), (rows_b, width_b) = shapes
        paired = (rows_a, rows_b, width_a) == (count_a, count_b, width_b)
    if not paired:
        got = ', '.join(str(shape) for shape in shapes)
        raise ValueError(
            f'{name} takes radii (A,) and (B,) with directions (A, d) and (B, d); got {got}'
        )

def polar_distance(
```

Replace:

```python
    )
    return 2 * torch.asinh(root)

# -------------------------------------------------------------------------------------------------
```

with:

```python
    )
    return 2 * torch.asinh(root)

def exp_map_origin(tangent: torch.Tensor) -> torch.Tensor:
    '''
    The exponential map at the origin of the curvature -1 hyperboloid (c = 1), in float64.

    It is ``HyperbolicHead``'s map, computed in float64 on the CPU whatever the tangent's device
    and dtype: the hyperbolic arm's read map (``HyperbolicHead.read_points``).

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
```

In `src/naics_embedder/text_model/monitor.py`, make one edit.

Replace:

```python
from naics_embedder.panels.text_only import matrix_fingerprint
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.text_model.arm_encoder import exp_map_origin
from naics_embedder.text_model.export import (
    ENCODE_BATCH_SIZE,
    encode_query_texts,
    encode_token_rows,
)

logger = logging.getLogger(__name__)
```

with:

```python
from naics_embedder.panels.text_only import matrix_fingerprint
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.text_model.export import (
    ENCODE_BATCH_SIZE,
    encode_query_texts,
    encode_token_rows,
)
from naics_embedder.text_model.hyperbolic import exp_map_origin

logger = logging.getLogger(__name__)
```

In `src/naics_embedder/text_model/radius_report.py`, make one edit.

Replace:

```python
from naics_embedder.panels.decoding import lorentz_distances
from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.text_model.arm_encoder import exp_map_origin
from naics_embedder.text_model.hyperbolic import polar_distance

# polar_distance materializes (rows, codes, dimension), so bound the first axis (spec 4.2).
```

with:

```python
from naics_embedder.panels.decoding import lorentz_distances
from naics_embedder.panels.regressor import coordinate_matrix
from naics_embedder.text_model.hyperbolic import exp_map_origin, polar_distance

# polar_distance materializes (rows, codes, dimension), so bound the first axis (spec 4.2).
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_heads.py tests/unit/test_arm_encoder.py tests/unit/test_monitor.py -n auto -q
```
Expected: PASS, `102 passed`.

- [ ] **Step 5: Format, lint and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/text_model/heads.py src/naics_embedder/text_model/hyperbolic.py src/naics_embedder/panels/decoding.py src/naics_embedder/text_model/arm_encoder.py src/naics_embedder/text_model/monitor.py src/naics_embedder/text_model/radius_report.py tests/unit/test_heads.py tests/unit/test_arm_encoder.py tests/unit/test_monitor.py
```
Expected: no file changes (`git status --short` lists the same nine files as before).

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run pytest -n auto -q`
Expected: `3023 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/text_model/heads.py src/naics_embedder/text_model/hyperbolic.py src/naics_embedder/panels/decoding.py src/naics_embedder/text_model/arm_encoder.py src/naics_embedder/text_model/monitor.py src/naics_embedder/text_model/radius_report.py tests/unit/test_heads.py tests/unit/test_arm_encoder.py tests/unit/test_monitor.py
git commit -m "feat(text_model): add the Euclidean and spherical geometry heads (Req 12)"
```

### Task 2: Geometry as a configuration factor, in the encoder record

`model.geometry` picks the arm's head (Req 12). The encoder record carries it (P7), so every
consumer that already compares fusion, dimension and backbone now compares the geometry too:
- exact resume and the remote workflow's resume check;
- `CheckpointRunner`;
- the HGCN feeder and `load_from_checkpoint`.

The changes:
- `ModelConfig.geometry` defaults to `hyperbolic`, and `conf/config.yaml` ships it explicitly. Its
  comments say that `model.radius_bound`, `loss.radial_weight` and `loss.radial_step` are read
  only under hyperbolic.
- `EncoderArchitecture.geometry` defaults to `hyperbolic`, so a record saved before Stage 8 reads
  as hyperbolic. A four-copy record (the legacy layout) must be hyperbolic.
- `shared_encoder_architecture` takes `geometry` with no default, so no caller leaves it out.
- `SharedEncoder` and `NAICSContrastiveModel` take `geometry`, check it before the backbone loads,
  and build their head through `build_head`. The model saves it as a hyperparameter and writes it
  into its contract. Its `forward` docstring describes the arm's points.
- `train` passes the geometry and prints it in its banner. `_sweep_spec` names the configured
  geometry rather than a constant, and `CheckpointRunner` expects it in each seed's record.
- The remote workflow's resume pre-check needs no edit. `remote/canonical.py` compares the
  saved contract with `runtime_contract_for(cfg, bundle)`, whose encoder record comes from
  `encoder_architecture_for(cfg)`, which now passes `cfg.model.geometry`.
- `run_settings` is unchanged (P7).

The tests cover:
- the config's default, its three choices and a refused fourth;
- the encoder's head and point width per geometry, an unknown geometry refused before the backbone
  loads, and float32 flat points under bf16 autocast;
- the model's head and contract per geometry, its constructor's signature, `load_from_checkpoint`
  refusing another geometry, and a Stage-7-shaped checkpoint (`pre_stage8_checkpoint`) loading as
  hyperbolic;
- the contract module's pre-Stage-8 record and its required geometry;
- the runner refusing a seed of another geometry, reading a pre-Stage-8 seed as hyperbolic, and
  `tools sweep` refusing another geometry before its first read;
- the CLI's model, contract, banner and sweep spec;
- the export provenance's encoder record, and the one list of geometries, extended to the config
  and the record.

**Files:**
- Modify: `conf/config.yaml`, lines 58-62, 74-80
- Modify: `src/naics_embedder/cli/commands/tools.py`, lines 750-763
- Modify: `src/naics_embedder/cli/commands/training.py`, lines 101-104, 194-199, 507-510
- Modify: `src/naics_embedder/supervision/checkpoints.py`, lines 9-15, 45-53, 56-71, 75-90
- Modify: `src/naics_embedder/text_model/checkpoint_runner.py`, lines 84-88
- Modify: `src/naics_embedder/text_model/naics_model.py`, lines 6-12, 44-47, 144-151, 176-184,
  190-193, 221-224, 246-252, 310-313, 337-342
- Modify: `src/naics_embedder/text_model/shared_encoder.py`, lines 37-41, 108-125, 131-134, 144-147,
  182-186, 191-195, 204-213
- Modify: `src/naics_embedder/utils/config.py`, lines 899-909, 939-943, 951-955
- Modify: `tests/fixtures/shared_encoder.py`, lines 10-14, 191-194
- Test: `tests/unit/test_checkpoint_contract.py`, lines 24-30, 327-330, 343-356, 389-395
- Test: `tests/unit/test_checkpoint_runner.py`, lines 212-215, 313-316
- Test: `tests/unit/test_cli_training.py`, lines 39-43, 341-347
- Test: `tests/unit/test_config.py`, lines 665-668
- Test: `tests/unit/test_encoder.py`, lines 25-28, 258-261, 520-523
- Test: `tests/unit/test_export.py`, lines 344-348
- Test: `tests/unit/test_heads.py`, lines 18-21, 28-31, 45-50
- Test: `tests/unit/test_naics_model.py`, lines 36-39, 271-274, 405-408, 1534-1545, 1671-1674

**Interfaces:**
- Consumes: Task 1's `GEOMETRIES` and `build_head(geometry, *, radius_bound=8.0)`
  (`text_model/heads.py`).
- Produces:
  - `ModelConfig.geometry: Literal['euclidean', 'spherical', 'hyperbolic'] = 'hyperbolic'`
    (`utils/config.py`), read as `cfg.model.geometry`.
  - `EncoderArchitecture.geometry`, the same `Literal`, default `'hyperbolic'`
    (`supervision/checkpoints.py`). Construction raises `ValueError('a four-copy encoder record is
    hyperbolic: it names no other geometry')` for a four-copy record of another geometry.
  - `shared_encoder_architecture(*, fusion: str, dimension: int, backbone: str, geometry: str)
    -> EncoderArchitecture`. `geometry` is keyword-only and required.
  - `SharedEncoder(..., geometry: str = 'hyperbolic', radius_bound: float = 8.0)`, with the head
    at `.head`. An unknown geometry raises `ValueError("unknown geometry 'x'; expected one of
    [...]")` before the backbone loads.
  - `NAICSContrastiveModel(..., geometry: str = 'hyperbolic', ...)`, with the same refusal. It
    saves `hparams.geometry` and writes `contract.encoder.geometry`.
  - In `tests/fixtures/shared_encoder.py`, the fixture `pre_stage8_checkpoint -> Path`:
    `shared_model` saved with no geometry in its encoder record or hyperparameters, as Stage 7
    saved its checkpoints.

- [ ] **Step 1: Write the failing tests**

In `tests/fixtures/shared_encoder.py`, make these 2 edits, in order.

Replace:

```python
bundle (``tests/fixtures/supervision.py``) on the tiny backbone, and ``shared_checkpoint`` saves it
as Lightning would. ``truncated_checkpoint`` and ``pre_stage7_checkpoint`` save it as checkpoints
trained before Stage 6b and before Stage 7, which every load refuses. ``text_only_comparator_table``
is a table a read can be pointed at by mistake: the text-only comparator's, written by its own
builder.
```

with:

```python
bundle (``tests/fixtures/supervision.py``) on the tiny backbone, and ``shared_checkpoint`` saves it
as Lightning would. ``truncated_checkpoint`` and ``pre_stage7_checkpoint`` save it as checkpoints
trained before Stage 6b and before Stage 7, which every load refuses. ``pre_stage8_checkpoint``
saves it as Stage 7 did, naming no geometry, which every load reads as hyperbolic (P7).
``text_only_comparator_table``
is a table a read can be pointed at by mistake: the text-only comparator's, written by its own
builder.
```

Replace:

```python
    return path

def forbid_model_loads(monkeypatch) -> None:
    '''
```

with:

```python
    return path

@pytest.fixture
def pre_stage8_checkpoint(tmp_path, shared_model) -> Path:
    '''
    ``shared_model`` saved as Stage 7 saved its checkpoints (P7): neither its encoder record nor
    its hyperparameters name a geometry, so every load reads it as hyperbolic.
    '''

    checkpoint = lightning_checkpoint(shared_model)
    del checkpoint['stage3_supervision']['encoder']['geometry']
    del checkpoint['hyper_parameters']['geometry']
    path = tmp_path / 'pre_stage8.ckpt'
    torch.save(checkpoint, path)
    return path

def forbid_model_loads(monkeypatch) -> None:
    '''
```

In `tests/unit/test_checkpoint_contract.py`, make these 4 edits, in order.

Replace:

```python
)

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
SHARED = shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone=MINILM)
# Every field of a contract saved since Stage 7 (spec 4.5)
CONTRACT_FIELDS = {
    'contract_version',
```

with:

```python
)

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
SHARED = shared_encoder_architecture(
    fusion='masked_mean', dimension=16, backbone=MINILM, geometry='hyperbolic'
)
# Every field of a contract saved since Stage 7 (spec 4.5)
CONTRACT_FIELDS = {
    'contract_version',
```

Replace:

```python
            'x': 1
        },
    ],
)
```

with:

```python
            'x': 1
        },
        {
            'layout': 'shared',
            'fusion': 'masked_mean',
            'dimension': 16,
            'backbone': MINILM,
            'geometry': 'poincare'
        },
        {
            'layout': 'four-copy',
            'geometry': 'spherical'
        },
    ],
)
```

Replace:

```python
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
```

with:

```python
        'dimension': 16,
        'backbone': MINILM,
        'geometry': 'hyperbolic',
    }
    assert CheckpointContract.model_validate(saved) == runtime_contract

def test_an_encoder_record_saved_before_stage_8_reads_as_hyperbolic(tmp_path, runtime_contract):
    '''P7: no record saved before Stage 8 names a geometry, and each of them is hyperbolic.'''

    saved = runtime_contract.model_dump()
    del saved['encoder']['geometry']
    path = tmp_path / 'pre-stage-8.ckpt'
    torch.save({'stage3_supervision': saved}, path)

    assert CheckpointContract.model_validate(saved).encoder == SHARED
    validate_exact_resume(path, runtime_contract)

def test_a_shared_encoder_record_cannot_leave_out_its_geometry():
    assert SHARED.geometry == 'hyperbolic'
    with pytest.raises(TypeError, match='geometry'):
        shared_encoder_architecture(fusion='masked_mean', dimension=16, backbone=MINILM)

@pytest.mark.parametrize(
    'encoder',
    [
        LEGACY_ENCODER,
        shared_encoder_architecture(
            fusion='masked_mean', dimension=8, backbone=MINILM, geometry='hyperbolic'
        ),
        shared_encoder_architecture(
            fusion='moe', dimension=16, backbone=MINILM, geometry='hyperbolic'
        ),
        shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone='other/model', geometry='hyperbolic'
        ),
        shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone=MINILM, geometry='euclidean'
        ),
        shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone=MINILM, geometry='spherical'
        ),
    ],
)
```

Replace:

```python
    validated_bundle, summaries
):
    manifest = validated_bundle.manifest
    other = shared_encoder_architecture(fusion='attention', dimension=8, backbone=MINILM)
    saved = contract_for_bundle(manifest, encoder=other, summaries=summaries)

    assert validate_supervision_contract(saved.model_dump(), manifest, summaries=summaries) == saved
```

with:

```python
    validated_bundle, summaries
):
    manifest = validated_bundle.manifest
    other = shared_encoder_architecture(
        fusion='attention', dimension=8, backbone=MINILM, geometry='spherical'
    )
    saved = contract_for_bundle(manifest, encoder=other, summaries=summaries)

    assert validate_supervision_contract(saved.model_dump(), manifest, summaries=summaries) == saved
```

In `tests/unit/test_checkpoint_runner.py`, make these 2 edits, in order.

Replace:

```python
        _runner(fixture_run).check(fixture_run.spec, 7)

@pytest.fixture
def sweep_env(tmp_path, monkeypatch, trained_seeds, regressor_rows):
```

with:

```python
        _runner(fixture_run).check(fixture_run.spec, 7)

def test_check_refuses_a_seed_of_another_geometry(fixture_run):
    '''P7: the arm's geometry is its spec's, and a checkpoint of another arm's head is refused.'''

    spec = fixture_run.spec.model_copy(update={'geometry': 'euclidean'})

    with pytest.raises(ValueError, match='seed 7: the checkpoint encoder contract differs'):
        _runner(fixture_run).check(spec, 7)

def test_check_reads_a_seed_saved_before_stage_8_as_hyperbolic(fixture_run):
    '''P7: Stage 7's checkpoints name no geometry, and the reference arm reads them unchanged.'''

    for name in ('last.ckpt', 'epoch=001.ckpt'):
        path = fixture_run.directory / name
        saved = read_checkpoint(path)
        del saved['stage3_supervision']['encoder']['geometry']
        del saved['hyper_parameters']['geometry']
        torch.save(saved, path)

    assert _runner(fixture_run).check(fixture_run.spec, 7).epoch == 1

@pytest.fixture
def sweep_env(tmp_path, monkeypatch, trained_seeds, regressor_rows):
```

Replace:

```python
    assert not list(sweep_env.root.glob('**/*.parquet'))

def test_tools_sweep_refuses_a_text_only_revision_from_another_backbone(sweep_env, tmp_path):
    args = list(sweep_env.args)
```

with:

```python
    assert not list(sweep_env.root.glob('**/*.parquet'))

def test_tools_sweep_refuses_seeds_of_another_geometry_before_the_first_read(sweep_env):
    '''P7: the sweep's config names the arm's geometry, and every seed is checked against it.'''

    result = CliRunner().invoke(tools_cli.app, [*sweep_env.args, 'model.geometry=euclidean'])

    assert result.exit_code == 1, result.output
    output = result.output.replace('\n', '')
    assert 'seed 1' in output and 'encoder contract differs' in output
    assert SelectionLog(sweep_env.log).records() == []
    assert not sweep_env.output.exists()
    assert not list(sweep_env.root.glob('**/*.parquet'))

def test_tools_sweep_refuses_a_text_only_revision_from_another_backbone(sweep_env, tmp_path):
    args = list(sweep_env.args)
```

In `tests/unit/test_cli_training.py`, make these 2 edits, in order.

Replace:

```python
# The record the default config builds
CONFIGURED_ENCODER = shared_encoder_architecture(
    fusion='masked_mean', dimension=16, backbone=MINILM
)
HGCN_QUESTION = 'Generate embeddings parquet file from this checkpoint?'
```

with:

```python
# The record the default config builds
CONFIGURED_ENCODER = shared_encoder_architecture(
    fusion='masked_mean', dimension=16, backbone=MINILM, geometry='hyperbolic'
)
HGCN_QUESTION = 'Generate embeddings parquet file from this checkpoint?'
```

Replace:

```python
    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['checkpoint_contract'].encoder == shared_encoder_architecture(
        fusion='attention', dimension=8, backbone=MINILM
    )
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('attention', 8)

@pytest.mark.unit
```

with:

```python
    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['checkpoint_contract'].encoder == shared_encoder_architecture(
        fusion='attention', dimension=8, backbone=MINILM, geometry='hyperbolic'
    )
    assert (model_kwargs['fusion'], model_kwargs['dimension']) == ('attention', 8)

@pytest.mark.unit
@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_the_model_and_its_contract_follow_the_configured_geometry(training_env, geometry):
    '''Req 12, P7: the geometry reaches the model and the encoder record exact resume compares.'''

    training.train(skip_validation=True, overrides=[f'model.geometry={geometry}'])

    model_kwargs = training_env.trainer.fit_calls[0]['model'].kwargs
    assert model_kwargs['geometry'] == geometry
    assert model_kwargs['checkpoint_contract'].encoder == shared_encoder_architecture(
        fusion='masked_mean', dimension=16, backbone=MINILM, geometry=geometry
    )

@pytest.mark.unit
def test_the_train_banner_names_the_geometry(cli_runner, training_env):
    result = cli_runner.invoke(
        cli_app, ['train', 'model.geometry=spherical'], catch_exceptions=False
    )

    assert result.exit_code == 0
    assert 'Geometry: spherical' in result.output.replace('\n', '')

@pytest.mark.unit
def test_the_sweep_spec_names_the_configured_geometry_outside_the_run_settings(
    training_env, monkeypatch
):
    '''P7: the arm's geometry is its spec's, and the 21 run settings keep their keys.'''

    monkeypatch.setattr(
        tools_cli, 'load_backbone', lambda name: (None, None, 'resolved'), raising=False
    )
    cfg = training.Config.from_yaml('config.yaml').override(
        {
            'model.geometry': 'euclidean',
            'model.dimension': 8
        }
    )

    spec = tools_cli._sweep_spec(cfg, name='euclidean-d8', accelerator='cuda')

    assert (spec.geometry, spec.dimension) == ('euclidean', 8)
    assert len(spec.settings) == 21
    assert 'geometry' not in spec.settings

@pytest.mark.unit
```

In `tests/unit/test_config.py`, make one edit.

Replace:

```python
    assert _error_locs_and_types(excinfo) == [(('model', key.split('.')[1]), 'literal_error')]

def test_the_radius_bound_is_8_by_default_and_as_shipped(valid_config_dict):
    # Spec 4.2 and 4.5: R = 8, so a six-digit code at its target r = 5 keeps dr/dν ≈ 0.61
```

with:

```python
    assert _error_locs_and_types(excinfo) == [(('model', key.split('.')[1]), 'literal_error')]

def test_the_geometry_is_hyperbolic_by_default_and_as_shipped(valid_config_dict):
    '''Req 5's reference configuration is hyperbolic, and the YAML states the key itself.'''

    assert Config().model.geometry == 'hyperbolic'
    assert valid_config_dict['model']['geometry'] == 'hyperbolic'
    assert Config.model_validate(valid_config_dict).model.geometry == 'hyperbolic'

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical', 'hyperbolic'])
def test_every_geometry_arm_is_accepted(geometry):
    assert Config().override({'model.geometry': geometry}).model.geometry == geometry

def test_a_geometry_outside_the_three_arms_is_refused():
    with pytest.raises(ValidationError) as excinfo:
        Config().override({'model.geometry': 'poincare'})

    assert _error_locs_and_types(excinfo) == [(('model', 'geometry'), 'literal_error')]

def test_the_radius_bound_is_8_by_default_and_as_shipped(valid_config_dict):
    # Spec 4.2 and 4.5: R = 8, so a six-digit code at its target r = 5 keeps dr/dν ≈ 0.61
```

In `tests/unit/test_encoder.py`, make these 3 edits, in order.

Replace:

```python
from naics_embedder.text_model.fields import CHANNELS, QUERY
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.hyperbolic import HyperbolicHead, check_lorentz_manifold_validity
from naics_embedder.text_model.shared_encoder import (
```

with:

```python
from naics_embedder.text_model.fields import CHANNELS, QUERY
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.hyperbolic import HyperbolicHead, check_lorentz_manifold_validity
from naics_embedder.text_model.shared_encoder import (
```

Replace:

```python
    assert make_encoder(radius_bound=5.0).head.radius_bound == 5.0
    assert make_encoder().head.radius_bound == 8.0

def test_an_unknown_fusion_or_dimension_is_refused(make_encoder):
```

with:

```python
    assert make_encoder(radius_bound=5.0).head.radius_bound == 5.0
    assert make_encoder().head.radius_bound == 8.0

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_the_geometry_picks_the_head_and_the_width_of_the_point(make_encoder, geometry):
    '''Req 12: every arm shares the encoder; the head and the point's width follow the geometry.'''

    encoder = make_encoder(geometry=geometry).eval()

    with torch.no_grad():
        output = encoder(stack_text_inputs(CODES))

    assert encoder.head.geometry == geometry
    assert [name for name, _ in encoder.named_children()][-1] == 'head'
    assert output['tangent'].shape == (2, 8)
    assert output['embedding'].shape == (2, 9 if geometry == 'hyperbolic' else 8)

def test_an_unknown_geometry_is_refused_before_the_backbone_loads(make_encoder, monkeypatch):

    def never(_name):
        raise AssertionError('the backbone loaded before the geometry was refused')

    monkeypatch.setattr('naics_embedder.text_model.shared_encoder.load_base_model', never)

    with pytest.raises(ValueError, match="unknown geometry 'poincare'"):
        make_encoder(geometry='poincare')

def test_an_unknown_fusion_or_dimension_is_refused(make_encoder):
```

Replace:

```python
    assert {output[name].dtype for name in floating} == {torch.float32}

def test_a_malformed_batch_is_refused(make_encoder):
    encoder = make_encoder()
```

with:

```python
    assert {output[name].dtype for name in floating} == {torch.float32}

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_under_bf16_autocast_a_flat_head_gives_float32_points(make_encoder, geometry):
    encoder = make_encoder(geometry=geometry).eval()

    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        output = encoder(stack_text_inputs(CODES))

    names = ('embedding', 'tangent', 'radius', 'direction')
    assert {output[name].dtype for name in names} == {torch.float32}

def test_a_malformed_batch_is_refused(make_encoder):
    encoder = make_encoder()
```

In `tests/unit/test_export.py`, make one edit.

Replace:

```python
        validated_bundle.manifest,
        encoder=shared_encoder_architecture(
            fusion='masked_mean', dimension=ARM_DIMENSION, backbone=MINILM
        ),
        summaries=summaries_identity(MINILM),
```

with:

```python
        validated_bundle.manifest,
        encoder=shared_encoder_architecture(
            fusion='masked_mean', dimension=ARM_DIMENSION, backbone=MINILM, geometry='hyperbolic'
        ),
        summaries=summaries_identity(MINILM),
```

In `tests/unit/test_heads.py`, make these 3 edits, in order.

Replace:

```python
    euclidean_distances,
)
from naics_embedder.text_model.heads import (
    GEOMETRIES,
```

with:

```python
    euclidean_distances,
)
from naics_embedder.supervision.checkpoints import EncoderArchitecture
from naics_embedder.text_model.heads import (
    GEOMETRIES,
```

Replace:

```python
)
from naics_embedder.text_model.hyperbolic import HyperbolicHead, exp_map_origin, polar_distance

pytestmark = pytest.mark.unit
```

with:

```python
)
from naics_embedder.text_model.hyperbolic import HyperbolicHead, exp_map_origin, polar_distance
from naics_embedder.utils.config import ModelConfig

pytestmark = pytest.mark.unit
```

Replace:

```python
    assert tuple(GEOMETRY_DISTANCES) == GEOMETRIES
    assert get_args(records.Geometry) == GEOMETRIES
    assert diagnostics.GEOMETRIES == GEOMETRIES
    assert set(GEOMETRY_DISTANCES.values()) == set(DISTANCES)

@pytest.mark.parametrize('geometry', GEOMETRIES)
```

with:

```python
    assert tuple(GEOMETRY_DISTANCES) == GEOMETRIES
    assert get_args(records.Geometry) == GEOMETRIES
    assert diagnostics.GEOMETRIES == GEOMETRIES
    assert get_args(ModelConfig.model_fields['geometry'].annotation) == GEOMETRIES
    assert get_args(EncoderArchitecture.model_fields['geometry'].annotation) == GEOMETRIES
    assert set(GEOMETRY_DISTANCES.values()) == set(DISTANCES)

@pytest.mark.parametrize('geometry', GEOMETRIES)
```

In `tests/unit/test_naics_model.py`, make these 5 edits, in order.

Replace:

```python
from naics_embedder.text_model.export import encode_token_rows
from naics_embedder.text_model.fields import CHANNELS, QUERY
from naics_embedder.text_model.loss import LogitScale
from naics_embedder.text_model.monitor import (
```

with:

```python
from naics_embedder.text_model.export import encode_token_rows
from naics_embedder.text_model.fields import CHANNELS, QUERY
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.loss import LogitScale
from naics_embedder.text_model.monitor import (
```

Replace:

```python
            ('fusion', 'masked_mean'),
            ('dimension', 16),
            ('num_experts', 4),
            ('top_k', 2),
```

with:

```python
            ('fusion', 'masked_mean'),
            ('dimension', 16),
            ('geometry', 'hyperbolic'),
            ('num_experts', 4),
            ('top_k', 2),
```

Replace:

```python
        with pytest.raises(ValueError, match='unknown dimension'):
            NAICSContrastiveModel(**model_config, dimension=12)

    def test_the_code_targets_are_buffers_that_checkpoints_leave_out(
```

with:

```python
        with pytest.raises(ValueError, match='unknown dimension'):
            NAICSContrastiveModel(**model_config, dimension=12)

    @pytest.mark.parametrize('geometry', GEOMETRIES)
    def test_the_geometry_picks_the_head_and_enters_the_contract(self, reference_model, geometry):
        '''Req 12: the geometry is a saved hyperparameter, and the encoder record names it (P7).'''

        model = reference_model(geometry=geometry)

        assert model.hparams['geometry'] == geometry
        assert model.encoder.head.geometry == geometry
        assert model.checkpoint_contract.encoder.geometry == geometry

    def test_an_unknown_geometry_is_refused(self, model_config):
        with pytest.raises(ValueError, match="unknown geometry 'poincare'"):
            NAICSContrastiveModel(**model_config, geometry='poincare')

    def test_the_code_targets_are_buffers_that_checkpoints_leave_out(
```

Replace:

```python
        assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
        assert contract.encoder == shared_encoder_architecture(
            fusion='masked_mean', dimension=16, backbone='sentence-transformers/all-MiniLM-L6-v2'
        )

    def test_a_runtime_contract_of_another_encoder_is_refused(self, model_config, validated_bundle):
        other = contract_for_bundle(
            validated_bundle.manifest,
            encoder=shared_encoder_architecture(
                fusion='masked_mean', dimension=8, backbone=model_config['base_model_name']
            ),
            summaries=None,
```

with:

```python
        assert contract.codebook_fingerprint == validated_bundle.manifest.codebook_fingerprint
        assert contract.encoder == shared_encoder_architecture(
            fusion='masked_mean',
            dimension=16,
            backbone='sentence-transformers/all-MiniLM-L6-v2',
            geometry='hyperbolic'
        )

    def test_a_runtime_contract_of_another_encoder_is_refused(self, model_config, validated_bundle):
        other = contract_for_bundle(
            validated_bundle.manifest,
            encoder=shared_encoder_architecture(
                fusion='masked_mean',
                dimension=8,
                backbone=model_config['base_model_name'],
                geometry='hyperbolic'
            ),
            summaries=None,
```

Replace:

```python
            NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu', dimension=8)

    def test_load_from_checkpoint_refuses_a_pre_stage_7_checkpoint(self, pre_stage7_checkpoint):
        '''
```

with:

```python
            NAICSContrastiveModel.load_from_checkpoint(path, map_location='cpu', dimension=8)

    def test_load_from_checkpoint_refuses_another_geometry(self, shared_checkpoint):
        '''P7: the geometry is in the encoder record, so another arm's head never loads.'''

        with pytest.raises(ValueError, match="geometry='euclidean'"):
            NAICSContrastiveModel.load_from_checkpoint(
                shared_checkpoint, map_location='cpu', geometry='euclidean'
            )

    def test_a_checkpoint_saved_before_stage_8_loads_as_hyperbolic(self, pre_stage8_checkpoint):
        restored = NAICSContrastiveModel.load_from_checkpoint(
            pre_stage8_checkpoint, map_location='cpu'
        )

        assert restored.hparams['geometry'] == 'hyperbolic'
        assert restored.encoder.head.geometry == 'hyperbolic'
        assert restored.checkpoint_contract.encoder.geometry == 'hyperbolic'

    def test_load_from_checkpoint_refuses_a_pre_stage_7_checkpoint(self, pre_stage7_checkpoint):
        '''
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_checkpoint_contract.py tests/unit/test_checkpoint_runner.py tests/unit/test_cli_training.py tests/unit/test_config.py tests/unit/test_encoder.py tests/unit/test_export.py tests/unit/test_heads.py tests/unit/test_naics_model.py -n auto -q
```
Expected: FAIL, `24 failed, 431 passed, 29 errors`. The 29 errors come at fixture setup:
- in `test_checkpoint_contract.py` and `test_cli_training.py`, `TypeError:
  shared_encoder_architecture() got an unexpected keyword argument 'geometry'`;
- for `pre_stage8_checkpoint`, `KeyError: 'geometry'`.

The failures are of these kinds:
- `TypeError` for `geometry` in `SharedEncoder` and `NAICSContrastiveModel`;
- pydantic's `model.geometry  Extra inputs are not permitted`;
- `'ModelConfig' object has no attribute 'geometry'` and `KeyError: 'geometry'`;
- `Failed: DID NOT RAISE` for the geometry refusals.

- [ ] **Step 3: Write the implementation**

In `conf/config.yaml`, make these 2 edits, in order.

Replace:

```yaml
  fusion: masked_mean  # masked_mean, attention, or moe (an ablation only)
  dimension: 16  # 8, 16 or 32: the one Linear(384 -> d) before the geometry head
  radius_bound: 8.0  # R: the head gives a vector of norm v the radius R*tanh(v/R) (Req 13)

  lora:
```

with:

```yaml
  fusion: masked_mean  # masked_mean, attention, or moe (an ablation only)
  dimension: 16  # 8, 16 or 32: the one Linear(384 -> d) before the geometry head
  geometry: hyperbolic  # euclidean, spherical or hyperbolic (Req 12): the head and its distance
  radius_bound: 8.0  # R, read only under hyperbolic: the radius R*tanh(v/R) of norm v (Req 13)

  lora:
```

Replace:

```yaml
loss:
  code_code_weight: 1  # w_c
  radial_weight: 1  # w_r
  target_temperature: 1  # τ_t, of the code-code target softmax(-D* / τ_t)
  radial_step: 1  # ρ: level λ's target radius is ρ * (λ - 1)
  logit_scale_init: 1  # Both learned logit scales start here
  logit_scale_range: [0.01, 100]  # And are clamped to this range
```

with:

```yaml
loss:
  code_code_weight: 1  # w_c
  radial_weight: 1  # w_r, read only under hyperbolic (Req 12)
  target_temperature: 1  # τ_t, of the code-code target softmax(-D* / τ_t)
  radial_step: 1  # ρ: level λ's target radius is ρ * (λ - 1), under hyperbolic only
  logit_scale_init: 1  # Both learned logit scales start here
  logit_scale_range: [0.01, 100]  # And are clamped to this range
```

In `src/naics_embedder/cli/commands/tools.py`, make one edit.

Replace:

```python
    return require_valid_supervision_bundle(cfg)

def _sweep_spec(cfg: Config, *, name: str, accelerator: str) -> ArmSpec:
    '''The arm's settings and text identities, from its config and cached backbone (P21, P32).'''

    _, _, revision = load_backbone(cfg.model.base_model_name)
    return ArmSpec(
        name=name,
        components=1,
        dimension=cfg.model.dimension,
        geometry='hyperbolic',
        backbone=cfg.model.base_model_name,
        backbone_revision=revision,
        descriptions_sha256=sha256_file(cfg.data_loader.streaming.descriptions_parquet),
```

with:

```python
    return require_valid_supervision_bundle(cfg)

def _sweep_spec(cfg: Config, *, name: str, accelerator: str) -> ArmSpec:
    '''The arm's geometry, settings and text identities, from its config and cached backbone.'''

    _, _, revision = load_backbone(cfg.model.base_model_name)
    return ArmSpec(
        name=name,
        components=1,
        dimension=cfg.model.dimension,
        geometry=cfg.model.geometry,
        backbone=cfg.model.base_model_name,
        backbone_revision=revision,
        descriptions_sha256=sha256_file(cfg.data_loader.streaming.descriptions_parquet),
```

In `src/naics_embedder/cli/commands/training.py`, make these 3 edits, in order.

Replace:

```python
        fusion=cfg.model.fusion,
        dimension=cfg.model.dimension,
        num_experts=cfg.model.moe.num_experts,
        top_k=cfg.model.moe.top_k,
```

with:

```python
        fusion=cfg.model.fusion,
        dimension=cfg.model.dimension,
        geometry=cfg.model.geometry,
        num_experts=cfg.model.moe.num_experts,
        top_k=cfg.model.moe.top_k,
```

Replace:

```python
        fusion=cfg.model.fusion,
        dimension=cfg.model.dimension,
        backbone=cfg.model.base_model_name,
    )

def runtime_contract_for(cfg: Config, bundle: ValidatedSupervisionBundle) -> CheckpointContract:
```

with:

```python
        fusion=cfg.model.fusion,
        dimension=cfg.model.dimension,
        backbone=cfg.model.base_model_name,
        geometry=cfg.model.geometry,
    )

def runtime_contract_for(cfg: Config, bundle: ValidatedSupervisionBundle) -> CheckpointContract:
```

Replace:

```python
            f'  • LoRA rank: {cfg.model.lora.r}',
            f'  • Fusion: {cfg.model.fusion}',
            f'  • Dimension: {cfg.model.dimension}\n',
            '[cyan]Training:[/cyan]',
```

with:

```python
            f'  • LoRA rank: {cfg.model.lora.r}',
            f'  • Fusion: {cfg.model.fusion}',
            f'  • Geometry: {cfg.model.geometry}',
            f'  • Dimension: {cfg.model.dimension}\n',
            '[cyan]Training:[/cyan]',
```

In `src/naics_embedder/supervision/checkpoints.py`, make these 4 edits, in order.

Replace:

```python
A checkpoint saved before Stage 7 names no objective and reads as ``pre-req11``. Exact resume,
the export, the reads and the HGCN feeder refuse it before any other check, and nothing migrates
it (D2). A checkpoint saved before Stage 6 has no encoder record either, and reads as the legacy
four-copy layout.
'''

# -------------------------------------------------------------------------------------------------
```

with:

```python
A checkpoint saved before Stage 7 names no objective and reads as ``pre-req11``. Exact resume,
the export, the reads and the HGCN feeder refuse it before any other check, and nothing migrates
it (D2). A checkpoint saved before Stage 6 has no encoder record either, and reads as the legacy
four-copy layout; one saved before Stage 8 names no geometry, and reads as hyperbolic (P7).
'''

# -------------------------------------------------------------------------------------------------
```

Replace:

```python
    '''
    The encoder architecture a checkpoint's weights belong to (spec 4.4).

    ``shared`` is Stage 6's one backbone, and names its fusion, dimension and backbone.
    ``four-copy`` is the legacy layout of every checkpoint saved before Stage 6, and names nothing
    else. A field added later defaults to the value every earlier checkpoint had.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')
```

with:

```python
    '''
    The encoder architecture a checkpoint's weights belong to (spec 4.4).

    ``shared`` is Stage 6's one backbone, and names its fusion, dimension and backbone, and its
    geometry head (Req 12). ``four-copy`` is the legacy layout of every checkpoint saved before
    Stage 6, and names nothing else. A field added later defaults to the value every earlier
    checkpoint had: every record saved before Stage 8 is hyperbolic.
    '''

    model_config = ConfigDict(frozen=True, extra='forbid')
```

Replace:

```python
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
```

with:

```python
    fusion: Optional[str] = None
    dimension: Optional[int] = None
    backbone: Optional[str] = None
    # Absent from every record saved before Stage 8, each of which is hyperbolic (P7)
    geometry: Literal['euclidean', 'spherical', 'hyperbolic'] = 'hyperbolic'

    @model_validator(mode='after')
    def check_fields_match_the_layout(self) -> 'EncoderArchitecture':
        '''
        A shared record names its fusion, dimension and backbone; a four-copy one names none of
        them, and is hyperbolic.
        '''

        recorded = (self.fusion, self.dimension, self.backbone)
        if self.layout == 'shared' and None in recorded:
            raise ValueError('a shared encoder record names its fusion, dimension and backbone')
        if self.layout == 'four-copy' and recorded != (None, None, None):
            raise ValueError('a four-copy encoder record names no fusion, dimension or backbone')
        if self.layout == 'four-copy' and self.geometry != 'hyperbolic':
            raise ValueError('a four-copy encoder record is hyperbolic: it names no other geometry')
        return self

LEGACY_ENCODER = EncoderArchitecture(layout='four-copy')
```

Replace:

```python
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
```

with:

```python
    fusion: str,
    dimension: int,
    backbone: str,
    geometry: str,
) -> EncoderArchitecture:
    '''
    The record of a Stage-6 shared encoder and its geometry head (Req 12).

    The model builds its record here from its hyperparameters, and training builds the config's
    here too, so the two cannot drift apart. ``geometry`` has no default, so no caller leaves an
    arm's geometry out of the record it compares (P7).
    '''

    return EncoderArchitecture(
        layout='shared', fusion=fusion, dimension=dimension, backbone=backbone, geometry=geometry
    )

class CheckpointContract(BaseModel):
```

In `src/naics_embedder/text_model/checkpoint_runner.py`, make one edit.

Replace:

```python
            fusion=spec.settings.get('fusion', self.cfg.model.fusion),
            dimension=spec.dimension,
            backbone=spec.backbone
        )
        if contract.encoder != expected:
```

with:

```python
            fusion=spec.settings.get('fusion', self.cfg.model.fusion),
            dimension=spec.dimension,
            backbone=spec.backbone,
            geometry=spec.geometry
        )
        if contract.encoder != expected:
```

In `src/naics_embedder/text_model/naics_model.py`, make these 9 edits, in order.

Replace:

```python
live code cache and selected on the outcome panel's validation MRR (spec 4.1-4.4; D6).

- SharedEncoder: one LoRA-tuned backbone over field-marked channels, masked fusion, one affine map
  to dimension d and the bounded head, which gives each text its radius r and direction û.
- A step reads two streams (spec 4.3): a chunk of the codes, as anchors, and a chunk of the task
  queries. Every candidate comes from the code cache, each code's (r, û) at its last refresh, with
  the step's anchors replaced by their live points.
```

with:

```python
live code cache and selected on the outcome panel's validation MRR (spec 4.1-4.4; D6).

- SharedEncoder: one LoRA-tuned backbone over field-marked channels, masked fusion, one affine map
  to dimension d and the geometry head (Req 12), which gives each text its point, its radius r and
  its direction û.
- A step reads two streams (spec 4.3): a chunk of the codes, as anchors, and a chunk of the task
  queries. Every candidate comes from the code cache, each code's (r, û) at its last refresh, with
  the step's anchors replaced by their live points.
```

Replace:

```python
from naics_embedder.text_model.epoch_summary import EPOCH_SUMMARY, EpochSummary
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.hyperbolic import polar_distance
from naics_embedder.text_model.loss import LogitScale, code_code_loss, radial_loss, task_loss
```

with:

```python
from naics_embedder.text_model.epoch_summary import EPOCH_SUMMARY, EpochSummary
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.hyperbolic import polar_distance
from naics_embedder.text_model.loss import LogitScale, code_code_loss, radial_loss, task_loss
```

Replace:

```python
        dimension: Embedding dimension, one of 8, 16 or 32: the width of the one
            ``Linear(hidden → d)`` before the geometry head
        num_experts: Number of MoE experts (``moe`` only)
        top_k: Number of experts each row is routed to (``moe`` only)
        moe_hidden_dim: Hidden dimension of the experts (``moe`` only)
        radius_bound: R, the head's bound on every radius: r = R · tanh(‖v‖ / R) (spec 4.2)
        code_code_weight: w_c, the code-code term's weight in the total (spec 4.1)
        radial_weight: w_r, the radial term's weight in the total
```

with:

```python
        dimension: Embedding dimension, one of 8, 16 or 32: the width of the one
            ``Linear(hidden → d)`` before the geometry head
        geometry: The geometry arm (Req 12): ``euclidean``, ``spherical`` or ``hyperbolic``
            (the default), which picks the head; the radial term applies under hyperbolic only
        num_experts: Number of MoE experts (``moe`` only)
        top_k: Number of experts each row is routed to (``moe`` only)
        moe_hidden_dim: Hidden dimension of the experts (``moe`` only)
        radius_bound: R, the hyperbolic head's bound on every radius: r = R · tanh(‖v‖ / R)
            (spec 4.2); the flat heads do not read it
        code_code_weight: w_c, the code-code term's weight in the total (spec 4.1)
        radial_weight: w_r, the radial term's weight in the total
```

Replace:

```python
    hyperparameters.

    Raises:
        ValueError: If the fusion or dimension is unknown, a setting is out of its range, the
            manifest is missing, a pre-validated bundle is not the configured one, or the runtime
            contract is not the bundle's.
    '''

    def __init__(
```

with:

```python
    hyperparameters.

    Raises:
        ValueError: If the fusion, dimension or geometry is unknown, a setting is out of its range,
            the manifest is missing, a pre-validated bundle is not the configured one, or the
            runtime contract is not the bundle's.
    '''

    def __init__(
```

Replace:

```python
        fusion: str = 'masked_mean',
        dimension: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
```

with:

```python
        fusion: str = 'masked_mean',
        dimension: int = 16,
        geometry: str = 'hyperbolic',
        num_experts: int = 4,
        top_k: int = 2,
```

Replace:

```python
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')
        _refuse_settings(
            code_code_weight=code_code_weight,
```

with:

```python
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')
        if geometry not in GEOMETRIES:
            raise ValueError(f'unknown geometry {geometry!r}; expected one of {list(GEOMETRIES)}')
        _refuse_settings(
            code_code_weight=code_code_weight,
```

Replace:

```python
        # The architecture this model's weights belong to; a checkpoint of any other is refused
        # (spec 4.4, roadmap D2)
        encoder_record = shared_encoder_architecture(
            fusion=fusion, dimension=dimension, backbone=base_model_name
        )

        # Load the validated supervision bundle before any model construction: the single
```

with:

```python
        # The architecture this model's weights belong to; a checkpoint of any other is refused
        # (spec 4.4, roadmap D2)
        encoder_record = shared_encoder_architecture(
            fusion=fusion, dimension=dimension, backbone=base_model_name, geometry=geometry
        )

        # Load the validated supervision bundle before any model construction: the single
```

Replace:

```python
            fusion=fusion,
            dimension=dimension,
            num_experts=num_experts,
            top_k=top_k,
```

with:

```python
            fusion=fusion,
            dimension=dimension,
            geometry=geometry,
            num_experts=num_experts,
            top_k=top_k,
```

Replace:

```python
        Returns:
            Dictionary containing:
            - embedding: Lorentz points (batch_size, dimension + 1)
            - tangent: Bounded tangent vectors at the origin (batch_size, dimension)
            - radius, direction: Each point's r (batch_size,) and û (batch_size, dimension)
            - gate_probs, top_k_indices: The experts' gates, under ``moe`` fusion only
```

with:

```python
        Returns:
            Dictionary containing:
            - embedding: The arm's points, Lorentz (batch_size, dimension + 1) under hyperbolic
            - tangent: The coordinates the export writes (batch_size, dimension)
            - radius, direction: Each point's r (batch_size,) and û (batch_size, dimension)
            - gate_probs, top_k_indices: The experts' gates, under ``moe`` fusion only
```

In `src/naics_embedder/text_model/shared_encoder.py`, make these 7 edits, in order.

Replace:

```python
from naics_embedder.text_model.fields import FIELDS
from naics_embedder.text_model.fusion import FUSIONS, build_fusion
from naics_embedder.text_model.hyperbolic import HyperbolicHead

logger = logging.getLogger(__name__)
```

with:

```python
from naics_embedder.text_model.fields import FIELDS
from naics_embedder.text_model.fusion import FUSIONS, build_fusion
from naics_embedder.text_model.heads import GEOMETRIES, build_head

logger = logging.getLogger(__name__)
```

Replace:

```python
        lora_dropout: LoRA dropout rate.
        fusion: One of ``FUSIONS``: ``masked_mean`` (the default), ``attention`` or ``moe``.
        dimension: The embedding dimension, one of ``DIMENSIONS``.
        num_experts: The number of experts, under ``moe`` only.
        top_k: The experts each code is routed to, under ``moe`` only.
        moe_hidden_dim: The experts' hidden width, under ``moe`` only.
        radius_bound: R, the head's bound on every radius (``model.radius_bound``).
        use_gradient_checkpointing: Recompute the backbone's activations in the backward pass.
        max_texts_per_call: The most texts one backbone call carries. It bounds a call's memory.
            With dropout off it leaves every output unchanged up to float noise; with dropout on it
            changes which random draws each text gets.

    Raises:
        ValueError: If the fusion or the dimension is outside its set, or ``max_texts_per_call``
            is below 1.
    '''

    def __init__(
```

with:

```python
        lora_dropout: LoRA dropout rate.
        fusion: One of ``FUSIONS``: ``masked_mean`` (the default), ``attention`` or ``moe``.
        dimension: The embedding dimension, one of ``DIMENSIONS``.
        geometry: The geometry arm, one of ``GEOMETRIES`` (``text_model/heads.py``), which picks
            the head (Req 12).
        num_experts: The number of experts, under ``moe`` only.
        top_k: The experts each code is routed to, under ``moe`` only.
        moe_hidden_dim: The experts' hidden width, under ``moe`` only.
        radius_bound: R, the hyperbolic head's bound on every radius (``model.radius_bound``);
            the flat heads do not read it.
        use_gradient_checkpointing: Recompute the backbone's activations in the backward pass.
        max_texts_per_call: The most texts one backbone call carries. It bounds a call's memory.
            With dropout off it leaves every output unchanged up to float noise; with dropout on it
            changes which random draws each text gets.

    Raises:
        ValueError: If the fusion, the dimension or the geometry is outside its set, or
            ``max_texts_per_call`` is below 1.
    '''

    def __init__(
```

Replace:

```python
        fusion: str = 'masked_mean',
        dimension: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
```

with:

```python
        fusion: str = 'masked_mean',
        dimension: int = 16,
        geometry: str = 'hyperbolic',
        num_experts: int = 4,
        top_k: int = 2,
```

Replace:

```python
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')
        if max_texts_per_call < 1:
            raise ValueError(f'max_texts_per_call must be at least 1, got {max_texts_per_call!r}')
```

with:

```python
        if dimension not in DIMENSIONS:
            raise ValueError(f'unknown dimension {dimension!r}; expected one of {list(DIMENSIONS)}')
        if geometry not in GEOMETRIES:
            raise ValueError(f'unknown geometry {geometry!r}; expected one of {list(GEOMETRIES)}')
        if max_texts_per_call < 1:
            raise ValueError(f'max_texts_per_call must be at least 1, got {max_texts_per_call!r}')
```

Replace:

```python
        )
        self.projection = nn.Linear(self.hidden_size, dimension)
        self.head = HyperbolicHead(radius_bound=radius_bound)
        self.dimension = dimension
        self.max_texts_per_call = max_texts_per_call
```

with:

```python
        )
        self.projection = nn.Linear(self.hidden_size, dimension)
        self.head = build_head(geometry, radius_bound=radius_bound)
        self.dimension = dimension
        self.max_texts_per_call = max_texts_per_call
```

Replace:

```python
            'Shared encoder initialized:\n'
            f'  • backbone: {base_model_name} (hidden size {self.hidden_size})\n'
            f'  • fusion: {fusion}; dimension: {dimension}\n'
            f'  • trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n'
        )
```

with:

```python
            'Shared encoder initialized:\n'
            f'  • backbone: {base_model_name} (hidden size {self.hidden_size})\n'
            f'  • fusion: {fusion}; dimension: {dimension}; geometry: {geometry}\n'
            f'  • trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)\n'
        )
```

Replace:

```python
                ``present`` (B,), as ``stack_text_inputs`` builds them.

        Returns:
            ``embedding`` (B, d + 1), the Lorentz point; ``tangent`` (B, d), the bounded tangent
            vector at the origin; ``radius`` (B,) and ``direction`` (B, d), its r and û; and
            ``gate_probs`` and ``top_k_indices`` under ``moe`` only. Every float output is float32
            under autocast too.

        Raises:
            ValueError: If the batch has no field, a field outside the marker set, or a field
```

with:

```python
                ``present`` (B,), as ``stack_text_inputs`` builds them.

        Returns:
            ``embedding``, the point the arm's distance reads: (B, d + 1) on the hyperboloid under
            hyperbolic, (B, d) otherwise; ``tangent`` (B, d), the coordinates the export writes;
            ``radius`` (B,) and ``direction`` (B, d), the point's polar parts; and ``gate_probs``
            and ``top_k_indices`` under ``moe`` only. Every float output is float32 under
            autocast too.

        Raises:
            ValueError: If the batch has no field, a field outside the marker set, or a field
```

In `src/naics_embedder/utils/config.py`, make these 3 edits, in order.

Replace:

```python
        default=16, description='Embedding dimension: the one Linear(hidden -> d) before the head'
    )
    radius_bound: float = Field(
        default=8.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            'R, the bound on every radius: the head gives a vector of norm v the radius '
            'R * tanh(v / R), which passes gradient at any length (Req 13)'
        ),
    )
```

with:

```python
        default=16, description='Embedding dimension: the one Linear(hidden -> d) before the head'
    )
    geometry: Literal['euclidean', 'spherical', 'hyperbolic'] = Field(
        default='hyperbolic',
        description=(
            'The geometry arm (Req 12): the head, its training and decoding distance and the '
            'export form; the radial term applies under hyperbolic only'
        ),
    )
    radius_bound: float = Field(
        default=8.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            'R, the bound on every radius, read only under hyperbolic: the head gives a vector of '
            'norm v the radius R * tanh(v / R), which passes gradient at any length (Req 13)'
        ),
    )
```

Replace:

```python
        ge=0,
        allow_inf_nan=False,
        description="w_r, the radial term's weight in the total",
    )
    target_temperature: float = Field(
```

with:

```python
        ge=0,
        allow_inf_nan=False,
        description="w_r, the radial term's weight in the total, read only under hyperbolic",
    )
    target_temperature: float = Field(
```

Replace:

```python
        gt=0,
        allow_inf_nan=False,
        description="ρ, the radius from one level to the next: level λ's target is ρ · (λ - 1)",
    )
    logit_scale_init: float = Field(
```

with:

```python
        gt=0,
        allow_inf_nan=False,
        description=(
            "ρ, the radius from one level to the next, read only under hyperbolic: level λ's "
            'target is ρ · (λ - 1)'
        ),
    )
    logit_scale_init: float = Field(
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_checkpoint_contract.py tests/unit/test_checkpoint_runner.py tests/unit/test_cli_training.py tests/unit/test_config.py tests/unit/test_encoder.py tests/unit/test_export.py tests/unit/test_heads.py tests/unit/test_naics_model.py -n auto -q
```
Expected: PASS, `591 passed`.

- [ ] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/cli/commands/tools.py src/naics_embedder/cli/commands/training.py src/naics_embedder/supervision/checkpoints.py src/naics_embedder/text_model/checkpoint_runner.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/text_model/shared_encoder.py src/naics_embedder/utils/config.py tests/fixtures/shared_encoder.py tests/unit/test_checkpoint_contract.py tests/unit/test_checkpoint_runner.py tests/unit/test_cli_training.py tests/unit/test_config.py tests/unit/test_encoder.py tests/unit/test_export.py tests/unit/test_heads.py tests/unit/test_naics_model.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict`
Expected: the build succeeds with no warning. `NAICSContrastiveModel.forward`'s `Returns:`
bullets are rendered by mkdocstrings, and a wrapped bullet there warns "Confusing indentation".

Run: `uv run pytest -n auto -q`
Expected: `3053 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add conf/config.yaml src/naics_embedder/cli/commands/tools.py src/naics_embedder/cli/commands/training.py src/naics_embedder/supervision/checkpoints.py src/naics_embedder/text_model/checkpoint_runner.py src/naics_embedder/text_model/naics_model.py src/naics_embedder/text_model/shared_encoder.py src/naics_embedder/utils/config.py tests/fixtures/shared_encoder.py tests/unit/test_checkpoint_contract.py tests/unit/test_checkpoint_runner.py tests/unit/test_cli_training.py tests/unit/test_config.py tests/unit/test_encoder.py tests/unit/test_export.py tests/unit/test_heads.py tests/unit/test_naics_model.py
git commit -m "feat(config): make geometry a configuration factor in the encoder record (Req 12)"
```

### Task 3: The objective under each head

Both training distances now come from the arm's head (P3). The task term reads the queries
against their candidates, and the code-code term reads the anchors against every code, each
through `head.pair_distance`. `compute_losses` no longer imports `polar_distance`.

The radial term exists only in the hyperbolic arm (Req 12; P6):
- `compute_losses` computes `radial_loss` only when `head.radial`. Otherwise `StepLosses.radial`
  is None and the total leaves it out.
- In the hyperbolic arm the radial term and the total are computed in Stage 7's order. The
  autograd graph is built in the same order, so the values and gradients are Stage 7's bit for
  bit.
- The health logs skip a term the arm lacks, so a flat arm logs no `loss/radial`.

The tests cover:
- every term and both scales getting gradient under each geometry; the spherical anchor radius
  takes none;
- a flat arm with no radial term, whose total is exactly task + w_c · code-code;
- float32 distances and terms with autocast off, per fusion and geometry, spied through
  `head.pair_distance`;
- the hyperbolic total, `torch.equal` to Stage 7's formula;
- the candidates, the cache with the anchors live, spied through the head.

**Files:**
- Modify: `src/naics_embedder/text_model/mixins/logging.py`, lines 5-14, 24-32
- Modify: `src/naics_embedder/text_model/naics_model.py`, lines 12-18, 46-50, 70-92, 382-386,
  407-417, 428-440
- Test: `tests/unit/test_naics_model.py`, lines 565-583, 596-602, 625-632, 730-740, 760-765,
  772-780, 782-788, 814-824

**Interfaces:**
- Consumes: Task 1's head interface (`pair_distance`, `radial`); Task 2's `geometry` argument to
  `NAICSContrastiveModel` and `model.encoder.head`.
- Produces: `StepLosses.radial: Optional[torch.Tensor]`, None unless the arm is hyperbolic.
  `StepLosses.total` is task + w_c · code-code, plus w_r · radial in the hyperbolic arm only. The
  health logs record a term only when it is not None.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_naics_model.py`, make these 8 edits, in order.

Replace:

```python
            reference_arm_model.compute_losses(epoch_steps[0])

    def test_every_term_and_both_scales_get_gradient(
        self, reference_arm_model, reference_arm_code_rows, epoch_steps
    ):
        '''Spec 6, No inert terms; and dL/dr_a is nonzero for every anchor (Verification
        "Radius").'''

        model = reference_arm_model
        model.refresh_code_cache(reference_arm_code_rows)

        losses = model.compute_losses(epoch_steps[0])

        assert isinstance(losses, model_module.StepLosses)
        assert losses.load_balancing is None
        weight = model.encoder.projection.weight
        for name in ('task', 'code_code', 'radial'):
            (gradient, ) = torch.autograd.grad(getattr(losses, name), weight, retain_graph=True)
            assert gradient.abs().sum() > 0, name
```

with:

```python
            reference_arm_model.compute_losses(epoch_steps[0])

    @pytest.mark.parametrize('geometry', GEOMETRIES)
    def test_every_term_and_both_scales_get_gradient(
        self, reference_model, reference_arm_code_rows, epoch_steps, geometry
    ):
        '''Spec 6, No inert terms, in every geometry arm; the radial term is the hyperbolic arm's
        alone (Req 12). dL/dr_a is nonzero for every anchor wherever the radius is live
        (Verification "Radius").'''

        model = reference_model(geometry=geometry)
        model.refresh_code_cache(reference_arm_code_rows)

        losses = model.compute_losses(epoch_steps[0])

        assert isinstance(losses, model_module.StepLosses)
        assert losses.load_balancing is None
        terms = ['task', 'code_code']
        if geometry == 'hyperbolic':
            terms.append('radial')
        else:
            assert losses.radial is None
        weight = model.encoder.projection.weight
        for name in terms:
            (gradient, ) = torch.autograd.grad(getattr(losses, name), weight, retain_graph=True)
            assert gradient.abs().sum() > 0, name
```

Replace:

```python
        assert gradients['task', 'code'] is None
        assert gradients['code_code', 'task'] is None
        (radius_gradient, ) = torch.autograd.grad(losses.total, losses.anchor_radius)
        assert losses.anchor_radius.shape == epoch_steps[0]['codes']['ids'].shape
        assert (radius_gradient != 0).all()

    def test_the_total_weights_the_terms_and_the_settings_reach_them(
```

with:

```python
        assert gradients['task', 'code'] is None
        assert gradients['code_code', 'task'] is None
        assert losses.anchor_radius.shape == epoch_steps[0]['codes']['ids'].shape
        if geometry == 'spherical':
            # Every point of the sphere is at radius 1: no term trains it
            assert not losses.anchor_radius.requires_grad
        else:
            (radius_gradient, ) = torch.autograd.grad(losses.total, losses.anchor_radius)
            assert (radius_gradient != 0).all()

    def test_the_total_weights_the_terms_and_the_settings_reach_them(
```

Replace:

```python
        losses = model.compute_losses(epoch_steps[0])

        assert settings == {'target_temperature': 0.5, 'radial_step': 0.75}
        expected = losses.task + 0.25 * losses.code_code + 2.0 * losses.radial
        torch.testing.assert_close(losses.total, expected, rtol=1e-6, atol=0.0)

    def test_load_balancing_reads_both_streams_gets_gradient_and_enters_the_total_under_moe(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch
```

with:

```python
        losses = model.compute_losses(epoch_steps[0])

        assert settings == {'target_temperature': 0.5, 'radial_step': 0.75}
        # Summed in Stage 7's order, so the hyperbolic arm's total is Stage 7's bit for bit (P6)
        assert torch.equal(
            losses.total, losses.task + 0.25 * losses.code_code + 2.0 * losses.radial
        )

    @pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
    def test_a_flat_arm_has_no_radial_term_and_its_total_leaves_it_out(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch, geometry
    ):
        '''The radial term exists only in the hyperbolic arm (Req 12): a flat arm never computes
        it, and its total is L_task + w_c L_cc whatever w_r is.'''

        model = reference_model(geometry=geometry, code_code_weight=0.25, radial_weight=2.0)
        model.refresh_code_cache(reference_arm_code_rows)
        calls = []
        radial = model_module.radial_loss

        def radial_spy(radius, levels, radial_step):
            calls.append(radial_step)
            return radial(radius, levels, radial_step)

        monkeypatch.setattr(model_module, 'radial_loss', radial_spy)

        losses = model.compute_losses(epoch_steps[0])

        assert calls == []
        assert losses.radial is None
        assert torch.equal(losses.total, losses.task + 0.25 * losses.code_code)

    def test_load_balancing_reads_both_streams_gets_gradient_and_enters_the_total_under_moe(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch
```

Replace:

```python
        assert by_query[5, frozenset({'44111'})] == {'31111', '31121', '32111', '44111', '311111'}

    @pytest.mark.parametrize('fusion', ['masked_mean', 'moe'])
    def test_the_distances_and_terms_run_in_float32_with_autocast_off(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch, fusion
    ):
        '''Spec 6, Precision: under CPU bf16 autocast, every distance and term is float32.'''

        model = reference_model(fusion=fusion, moe_hidden_dim=16)
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)
```

with:

```python
        assert by_query[5, frozenset({'44111'})] == {'31111', '31121', '32111', '44111', '311111'}

    @pytest.mark.parametrize(
        ('fusion', 'geometry'),
        [
            ('masked_mean', 'hyperbolic'),
            ('moe', 'hyperbolic'),
            ('masked_mean', 'euclidean'),
            ('masked_mean', 'spherical'),
        ],
    )
    def test_the_distances_and_terms_run_in_float32_with_autocast_off(
        self, reference_model, reference_arm_code_rows, epoch_steps, monkeypatch, fusion, geometry
    ):
        '''Spec 6, Precision: under CPU bf16 autocast, every distance and term is float32, in
        every geometry arm.'''

        model = reference_model(fusion=fusion, moe_hidden_dim=16, geometry=geometry)
        model.log = Mock()
        model.refresh_code_cache(reference_arm_code_rows)
```

Replace:

```python
            return wrapped

        for name in ('polar_distance', 'task_loss', 'code_code_loss', 'radial_loss'):
            monkeypatch.setattr(model_module, name, spy(name, getattr(model_module, name)))
        if fusion == 'moe':
            monkeypatch.setattr(
```

with:

```python
            return wrapped

        for name in ('task_loss', 'code_code_loss', 'radial_loss'):
            monkeypatch.setattr(model_module, name, spy(name, getattr(model_module, name)))
        head = model.encoder.head
        monkeypatch.setattr(head, 'pair_distance', spy('pair_distance', head.pair_distance))
        if fusion == 'moe':
            monkeypatch.setattr(
```

Replace:

```python
            losses = model.compute_losses(epoch_steps[0])

        names = [entry[0] for entry in seen]
        expected = [
            'polar_distance', 'polar_distance', 'task_loss', 'code_code_loss', 'radial_loss'
        ]
        if fusion == 'moe':
            expected.append('load_balancing')
        assert sorted(names) == sorted(expected)
```

with:

```python
            losses = model.compute_losses(epoch_steps[0])

        names = [entry[0] for entry in seen]
        expected = ['pair_distance', 'pair_distance', 'task_loss', 'code_code_loss']
        terms = ['total', 'task', 'code_code', 'anchor_radius']
        if geometry == 'hyperbolic':
            expected.append('radial_loss')
            terms.append('radial')
        if fusion == 'moe':
            expected.append('load_balancing')
        assert sorted(names) == sorted(expected)
```

Replace:

```python
            assert not autocast, name
            assert inputs == {torch.float32}, name
            assert output == torch.float32, name
        for name in ('total', 'task', 'code_code', 'radial', 'anchor_radius'):
            assert getattr(losses, name).dtype == torch.float32, name

    def test_the_candidates_are_the_cache_with_the_anchors_live(
```

with:

```python
            assert not autocast, name
            assert inputs == {torch.float32}, name
            assert output == torch.float32, name
        for name in terms:
            assert getattr(losses, name).dtype == torch.float32, name

    def test_the_candidates_are_the_cache_with_the_anchors_live(
```

Replace:

```python
        monkeypatch.setattr(model.encoder, 'forward', forward_spy)
        candidates = []
        distance = model_module.polar_distance

        def distance_spy(radius_a, direction_a, radius_b, direction_b):
            candidates.append((radius_b, direction_b))
            return distance(radius_a, direction_a, radius_b, direction_b)

        monkeypatch.setattr(model_module, 'polar_distance', distance_spy)

        model.compute_losses(step)
```

with:

```python
        monkeypatch.setattr(model.encoder, 'forward', forward_spy)
        candidates = []
        distance = model.encoder.head.pair_distance

        def distance_spy(radius_a, direction_a, radius_b, direction_b):
            candidates.append((radius_b, direction_b))
            return distance(radius_a, direction_a, radius_b, direction_b)

        monkeypatch.setattr(model.encoder.head, 'pair_distance', distance_spy)

        model.compute_losses(step)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_naics_model.py -n auto -q`
Expected: FAIL, `9 failed, 85 passed`, all in `TestComputeLosses`:
- the flat arms' radial term is not None (`assert tensor(...) is None`);
- the spies on `head.pair_distance` record no call, because `compute_losses` still calls
  `polar_distance` itself: the float32 tests' names differ, and the cache test fails
  `assert 0 == 2`;
- the gradient test fails under `euclidean` and `spherical`.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/text_model/mixins/logging.py`, make these 2 edits, in order.

Replace:

```python
Logging mixin for NAICSContrastiveModel: the epoch's health logs (P20).

Each epoch logs the mean of each term over its steps (``loss/task``, ``loss/code_code``,
``loss/radial``, ``loss/total``, and ``loss/load_balancing`` under ``moe``), the two logit scales
(``logit_scale/task``, ``logit_scale/code_code``), and r's mean and SD at each level from the
refreshed code cache (``radius/mean/level_<k>``, ``radius/sd/level_<k>``). Nothing selects on
them (spec 4.4).

A step has two leading sizes, its anchors and its queries, so Lightning's own epoch means would
weight each step by whichever it took for the batch size. The mixin keeps each step's values and
```

with:

```python
Logging mixin for NAICSContrastiveModel: the epoch's health logs (P20).

Each epoch logs the mean of each term over its steps (``loss/task``, ``loss/code_code``,
``loss/total``, ``loss/radial`` in the hyperbolic arm only, and ``loss/load_balancing`` under
``moe``), the two logit scales (``logit_scale/task``, ``logit_scale/code_code``), and r's mean
and SD at each level from the refreshed code cache (``radius/mean/level_<k>``,
``radius/sd/level_<k>``). Nothing selects on them (spec 4.4).

A step has two leading sizes, its anchors and its queries, so Lightning's own epoch means would
weight each step by whichever it took for the batch size. The mixin keeps each step's values and
```

Replace:

```python
    from naics_embedder.text_model.naics_model import StepLosses

logger = logging.getLogger(__name__)

# The terms whose epoch means are logged; load balancing exists under moe only (R11)
HEALTH_TERMS = ('task', 'code_code', 'radial', 'total', 'load_balancing')

class LoggingMixin:
    '''
```

with:

```python
    from naics_embedder.text_model.naics_model import StepLosses

logger = logging.getLogger(__name__)

# The terms whose epoch means are logged; the radial term exists in the hyperbolic arm only
# (Req 12), and load balancing under moe only (R11)
HEALTH_TERMS = ('task', 'code_code', 'radial', 'total', 'load_balancing')

class LoggingMixin:
    '''
```

In `src/naics_embedder/text_model/naics_model.py`, make these 6 edits, in order.

Replace:

```python
  queries. Every candidate comes from the code cache, each code's (r, û) at its last refresh, with
  the step's anchors replaced by their live points.
- Req 11's terms (spec 4.1): the task term over each query's candidates, the code-code listwise
  term over each anchor's J_a, and the radial term; under ``moe`` only, the experts' load
  balancing.
- The cache is refreshed at fit start and at each epoch's end. The end-of-epoch refresh feeds the
  outcome monitor's read, whose MRR is logged as ``val/outcome_mrr`` and steps the plateau.
```

with:

```python
  queries. Every candidate comes from the code cache, each code's (r, û) at its last refresh, with
  the step's anchors replaced by their live points.
- Req 11's terms (spec 4.1): the task term over each query's candidates and the code-code
  listwise term over each anchor's J_a, both under the arm's own distance, and, in the hyperbolic
  arm only, the radial term (Req 12); under ``moe`` only, the experts' load balancing.
- The cache is refreshed at fit start and at each epoch's end. The end-of-epoch refresh feeds the
  outcome monitor's read, whose MRR is logged as ``val/outcome_mrr`` and steps the plateau.
```

Replace:

```python
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.hyperbolic import polar_distance
from naics_embedder.text_model.loss import LogitScale, code_code_loss, radial_loss, task_loss
from naics_embedder.text_model.mixins import OUTCOME_MRR, LoggingMixin, LossMixin, OptimizerMixin
```

with:

```python
from naics_embedder.text_model.fusion import FUSIONS
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.loss import LogitScale, code_code_loss, radial_loss, task_loss
from naics_embedder.text_model.mixins import OUTCOME_MRR, LoggingMixin, LossMixin, OptimizerMixin
```

Replace:

```python
    '''
    One step's losses (spec 4.1).

    Attributes:
        total: What the step optimizes: L = L_task + w_c · L_cc + w_r · L_rad, plus the
            load-balancing term times its coefficient under ``moe``.
        task: L_task, the task term.
        code_code: L_cc, the code-code listwise term.
        radial: L_rad, the radial term.
        load_balancing: The experts' load-balancing term, before its coefficient; None unless the
            fusion is ``moe``.
        anchor_radius: The anchors' live radii r_a, (A,): every term's gradient reaches the radius
            through them (Verification "Radius").
    '''

    total: torch.Tensor
    task: torch.Tensor
    code_code: torch.Tensor
    radial: torch.Tensor
    load_balancing: Optional[torch.Tensor]
    anchor_radius: torch.Tensor

def _refuse_settings(
```

with:

```python
    '''
    One step's losses (spec 4.1).

    Attributes:
        total: What the step optimizes: L = L_task + w_c · L_cc, plus w_r · L_rad in the
            hyperbolic arm, plus the load-balancing term times its coefficient under ``moe``.
        task: L_task, the task term.
        code_code: L_cc, the code-code listwise term.
        radial: L_rad, the radial term; None unless the arm is hyperbolic (Req 12).
        load_balancing: The experts' load-balancing term, before its coefficient; None unless the
            fusion is ``moe``.
        anchor_radius: The anchors' live radii r_a, (A,): in the hyperbolic arm, every term's
            gradient reaches the radius through them (Verification "Radius").
    '''

    total: torch.Tensor
    task: torch.Tensor
    code_code: torch.Tensor
    radial: Optional[torch.Tensor]
    load_balancing: Optional[torch.Tensor]
    anchor_radius: torch.Tensor

def _refuse_settings(
```

Replace:

```python
        so gradient reaches the codes through the anchors alone, as anchors and as candidates.
        The distances and the terms run in float32 with autocast off: only the backbone runs in
        reduced precision (spec 4.2).

        Args:
```

with:

```python
        so gradient reaches the codes through the anchors alone, as anchors and as candidates.
        The distances and the terms run in float32 with autocast off: only the backbone runs in
        reduced precision (spec 4.2). Every distance is the arm's own (``head.pair_distance``),
        and the radial term exists only in the hyperbolic arm (Req 12).

        Args:
```

Replace:

```python
        ids = codes['ids']
        settings = self.hparams
        with torch.autocast(device_type=ids.device.type, enabled=False):
            anchor_radius = code_output['radius'].float()
            anchor_direction = code_output['direction'].float()
            radius, direction = cache.with_live(ids, anchor_radius, anchor_direction)
            # The task term: each query against the codes at its level and its forced negatives,
            # over all N codes (spec 4.1(i))
            query_distances = polar_distance(
                query_output['radius'].float(),
                query_output['direction'].float(),
```

with:

```python
        ids = codes['ids']
        settings = self.hparams
        head = self.encoder.head
        with torch.autocast(device_type=ids.device.type, enabled=False):
            anchor_radius = code_output['radius'].float()
            anchor_direction = code_output['direction'].float()
            radius, direction = cache.with_live(ids, anchor_radius, anchor_direction)
            # The task term: each query against the codes at its level and its forced negatives,
            # over all N codes (spec 4.1(i))
            query_distances = head.pair_distance(
                query_output['radius'].float(),
                query_output['direction'].float(),
```

Replace:

```python
            # The code-code term: each anchor against J_a (spec 4.1(ii))
            code_code = code_code_loss(
                polar_distance(anchor_radius, anchor_direction, radius, direction),
                self.logit_scale_code(),
                self.structural_distance[ids],
                self._keep(ids),
                settings.target_temperature,
            )
            # The radial term (spec 4.1(iii))
            radial = radial_loss(anchor_radius, codes['levels'], settings.radial_step)
            total = task + settings.code_code_weight * code_code + settings.radial_weight * radial
            load_balancing = None
            if self.fusion == 'moe':
```

with:

```python
            # The code-code term: each anchor against J_a (spec 4.1(ii))
            code_code = code_code_loss(
                head.pair_distance(anchor_radius, anchor_direction, radius, direction),
                self.logit_scale_code(),
                self.structural_distance[ids],
                self._keep(ids),
                settings.target_temperature,
            )
            # The radial term (spec 4.1(iii)), in the hyperbolic arm only (Req 12). It and the
            # total are computed in Stage 7's order, so the hyperbolic arm's values and gradients
            # are Stage 7's bit for bit (P6)
            radial = None
            if head.radial:
                radial = radial_loss(anchor_radius, codes['levels'], settings.radial_step)
            total = task + settings.code_code_weight * code_code
            if radial is not None:
                total = total + settings.radial_weight * radial
            load_balancing = None
            if self.fusion == 'moe':
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_naics_model.py -n auto -q`
Expected: PASS, `94 passed`.

- [ ] **Step 5: Format, lint and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/text_model/mixins/logging.py src/naics_embedder/text_model/naics_model.py tests/unit/test_naics_model.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run pytest -n auto -q`
Expected: `3059 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/text_model/mixins/logging.py src/naics_embedder/text_model/naics_model.py tests/unit/test_naics_model.py
git commit -m "feat(text_model): train under the head's distance, the radial term in hyperbolic only"
```

### Task 4: Export, reads and the HGCN feeder under each geometry

The roadmap's Consumes line names the gap: `ArmEncoder` maps queries and codes through the
hyperbolic exp map whatever the head. Both encoders now read through the head:
- `ArmEncoder` and the monitor's `LiveEncoder` take `head = head_of(model)` and
  `distance = head.distance`. They map the exported or live coordinates through
  `head.read_points`. `LiveEncoder` loses its fixed `distance = 'lorentz'`.
- A flat arm's read is its exported points as they are, under its own distance.

The export writes each head's `tangent` (Req 2's form; P4, P5), as it already did for the
hyperbolic arm. The provenance adds `geometry`, and the contract's encoder record adds its
geometry (Task 2). `COORDINATES` becomes a dict naming each geometry's form, and the hyperbolic
entry keeps Stage 7's string byte for byte (P9).

The HGCN feeder raises for a flat arm before it reads anything, and `train` asks its
HGCN-embeddings question only of a hyperbolic arm (P10).

The fixtures gain:
- `five_code_model(manifest, **overrides)`, which `shared_model` now calls;
- `geometry_checkpoint`, a factory that saves a five-code checkpoint of a given geometry.

The tests cover:
- the feeder's refusal, before the bundle is read;
- Stage 7's hyperbolic coordinates string;
- each geometry's exported coordinates and provenance;
- a flat arm's read against its export;
- the monitor's live read against the read of its own cache's export, per geometry;
- `train` never asking a flat arm the HGCN question.

**Files:**
- Modify: `src/naics_embedder/cli/commands/training.py`, lines 224-231, 241-246, 679-687
- Modify: `src/naics_embedder/text_model/arm_encoder.py`, lines 9-14, 38-44, 81-87, 102-106,
  196-211, 215-219
- Modify: `src/naics_embedder/text_model/export.py`, lines 7-14, 38-51, 93-100, 149-155, 173-178,
  229-234, 287-290, 305-309
- Modify: `src/naics_embedder/text_model/monitor.py`, lines 6-12, 39-43, 91-98, 193-197, 206-216,
  226-243, 248-252
- Modify: `tests/fixtures/shared_encoder.py`, lines 11-18, 21-27, 116-138, 143-146
- Test: `tests/unit/test_arm_encoder.py`, lines 11-17, 20-24, 149-152
- Test: `tests/unit/test_cli_training.py`, lines 713-718
- Test: `tests/unit/test_export.py`, lines 28-31, 190-193, 278-283
- Test: `tests/unit/test_monitor.py`, lines 20-24, 35-38, 580-587, 591-594, 610-622, 635-639

**Interfaces:**
- Consumes: Task 1's `head_of`, `read_points` and `distance`; Task 2's `geometry` in the config,
  the model and the encoder record; Task 3's flat training.
- Produces:
  - `ArmEncoder.head` and `ArmEncoder.distance` (the head's), and the same on `LiveEncoder`.
  - In `text_model/export.py`, `COORDINATES: Dict[str, str]`, keyed by geometry. The provenance
    keys `geometry` and `coordinates` hold the arm's geometry and `COORDINATES[geometry]`.
  - `generate_embeddings_from_checkpoint` raises `ValueError('the HGCN feeder writes Lorentz
    points, and a <geometry> arm has none: HGCN refines the hyperbolic arm only (Req 12)')`
    before it reads anything.
  - `train` logs `a <geometry> arm has no Lorentz points: no HGCN embeddings question` and
    skips the prompt.
  - In `tests/fixtures/shared_encoder.py`, `five_code_model(manifest: Path, **overrides) ->
    NAICSContrastiveModel` and the fixture `geometry_checkpoint -> Callable[[str], Path]`, which
    saves `tmp_path / f'{geometry}_arm.ckpt'`.

- [ ] **Step 1: Write the failing tests**

In `tests/fixtures/shared_encoder.py`, make these 4 edits, in order.

Replace:

```python
as Lightning would. ``truncated_checkpoint`` and ``pre_stage7_checkpoint`` save it as checkpoints
trained before Stage 6b and before Stage 7, which every load refuses. ``pre_stage8_checkpoint``
saves it as Stage 7 did, naming no geometry, which every load reads as hyperbolic (P7).
``text_only_comparator_table``
is a table a read can be pointed at by mistake: the text-only comparator's, written by its own
builder.

``reference_arm_model`` is a d = 16 model of the reference bundle, whose 17 codes span levels 2-6
```

with:

```python
as Lightning would. ``truncated_checkpoint`` and ``pre_stage7_checkpoint`` save it as checkpoints
trained before Stage 6b and before Stage 7, which every load refuses. ``pre_stage8_checkpoint``
saves it as Stage 7 did, naming no geometry, which every load reads as hyperbolic (P7), and
``geometry_checkpoint`` saves its arm in a geometry a test names (Req 12).
``text_only_comparator_table`` is a table a read can be pointed at by mistake: the text-only
comparator's, written by its own builder.

``reference_arm_model`` is a d = 16 model of the reference bundle, whose 17 codes span levels 2-6
```

Replace:

```python
'''

from pathlib import Path
from typing import Any, Dict, List, Mapping

import polars as pl
import pytest
```

with:

```python
'''

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

import polars as pl
import pytest
```

Replace:

```python
    )

@pytest.fixture
def shared_model(tiny_backbone, generated_bundle) -> NAICSContrastiveModel:
    '''
    A d = 16 masked-mean model of the five-code bundle on the tiny backbone, in eval mode.

    It records MiniLM's summaries, as training does: under the test seam, the dummy pin's sha256.
    '''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
        lora_r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        fusion='masked_mean',
        dimension=ARM_DIMENSION,
        supervision_manifest_path=str(generated_bundle),
        summaries=summaries_identity(MINILM),
    )
    return model.eval()

@pytest.fixture
```

with:

```python
    )

def five_code_model(manifest: Path, **overrides: Any) -> NAICSContrastiveModel:
    '''
    A d = 16 masked-mean model of the five-code bundle, in eval mode, with constructor overrides.

    Call it with the tiny backbone in place (``tiny_backbone``). It records MiniLM's summaries, as
    training does: under the test seam, the dummy pin's sha256.
    '''

    arguments: Dict[str, Any] = {
        'base_model_name': MINILM,
        'lora_r': 2,
        'lora_alpha': 4,
        'lora_dropout': 0.0,
        'fusion': 'masked_mean',
        'dimension': ARM_DIMENSION,
        'supervision_manifest_path': str(manifest),
        'summaries': summaries_identity(MINILM),
    }
    arguments.update(overrides)
    return NAICSContrastiveModel(**arguments).eval()

@pytest.fixture
def shared_model(tiny_backbone, generated_bundle) -> NAICSContrastiveModel:
    '''``five_code_model`` with its defaults, on the tiny backbone: the hyperbolic arm.'''

    return five_code_model(generated_bundle)

@pytest.fixture
```

Replace:

```python
    torch.save(lightning_checkpoint(shared_model), path)
    return path

@pytest.fixture
```

with:

```python
    torch.save(lightning_checkpoint(shared_model), path)
    return path

@pytest.fixture
def geometry_checkpoint(tmp_path, tiny_backbone, generated_bundle) -> Callable[[str], Path]:
    '''
    Save ``shared_model``'s arm in a geometry (Req 12) as a Lightning checkpoint: call it with the
    geometry's name.
    '''

    def save(geometry: str) -> Path:
        path = tmp_path / f'{geometry}_arm.ckpt'
        model = five_code_model(generated_bundle, geometry=geometry)
        torch.save(lightning_checkpoint(model), path)
        return path

    return save

@pytest.fixture
```

In `tests/unit/test_arm_encoder.py`, make these 3 edits, in order.

Replace:

```python
import torch

import naics_embedder.text_model.arm_encoder as arm_encoder_module
from naics_embedder.panels.decoding import lorentz_distances
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.selection_log import SelectionLog
```

with:

```python
import torch

import naics_embedder.text_model.arm_encoder as arm_encoder_module
from naics_embedder.panels.decoding import GEOMETRY_DISTANCES, lorentz_distances
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.selection_log import SelectionLog
```

Replace:

```python
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.export import encode_token_rows
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.text_model.hyperbolic import HyperbolicHead, exp_map_origin
```

with:

```python
from naics_embedder.text_model.arm_encoder import ArmEncoder, read_outcome_validation
from naics_embedder.text_model.dataloader.datamodule import stack_text_inputs
from naics_embedder.text_model.export import (
    encode_query_texts,
    encode_token_rows,
    export_code_table,
)
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.text_model.hyperbolic import HyperbolicHead, exp_map_origin
```

Replace:

```python
def test_the_distance_is_the_heads(arm):
    assert arm.distance == arm.model.encoder.head.distance == 'lorentz'

def test_the_logged_names_are_the_tables_and_the_checkpoints(
```

with:

```python
def test_the_distance_is_the_heads(arm):
    assert arm.distance == arm.model.encoder.head.distance == 'lorentz'

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_a_flat_arm_reads_its_exported_points_as_they_are_under_its_own_distance(
    tmp_path, geometry_checkpoint, validated_bundle, five_code_token_config, geometry
):
    '''Req 12: each arm decodes by its own distance, and a flat arm's read map is the identity.'''

    checkpoint = geometry_checkpoint(geometry)
    table = export_code_table(
        checkpoint, validated_bundle, five_code_token_config, tmp_path / f'{geometry}.parquet'
    )
    flat = ArmEncoder.from_files(checkpoint, table, validated_bundle, five_code_token_config)
    log_path = tmp_path / 'selection_log.jsonl'

    read_outcome_validation(
        flat, OutcomePanel.from_bundle(validated_bundle, log_path), 'plan 12 fixture read'
    )

    assert flat.distance == GEOMETRY_DISTANCES[geometry]
    assert torch.equal(flat.encode_codes(['222222', '111111']), _table_tangent(table)[[3, 0]])
    queries = encode_query_texts(flat.model, flat.tokenizer, QUERIES, flat.max_length)
    assert torch.equal(flat.encode_queries(QUERIES), queries)
    [record] = SelectionLog(log_path).records()
    assert record['detail']['distance'] == GEOMETRY_DISTANCES[geometry]

def test_the_logged_names_are_the_tables_and_the_checkpoints(
```

In `tests/unit/test_cli_training.py`, make one edit.

Replace:

```python
    assert questions == [(HGCN_QUESTION, False)]

@pytest.mark.unit
@pytest.mark.parametrize(
    ('make_stdin', 'expected'),
    [
```

with:

```python
    assert questions == [(HGCN_QUESTION, False)]

@pytest.mark.unit
@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_train_never_asks_a_flat_arm_the_hgcn_question(training_env, monkeypatch, geometry):
    '''HGCN refines Lorentz points, and a flat arm has none (Req 12): no question on a terminal.'''

    questions = []
    monkeypatch.setattr(training.typer, 'confirm', lambda *args, **_: questions.append(args))
    monkeypatch.setattr(training, '_stdin_is_terminal', lambda: True)

    training.train(skip_validation=True, overrides=[f'model.geometry={geometry}'])

    assert training_env.trainer.fit_calls
    assert questions == []

@pytest.mark.unit
@pytest.mark.parametrize(
    ('make_stdin', 'expected'),
    [
```

In `tests/unit/test_export.py`, make these 3 edits, in order.

Replace:

```python
)
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.utils.config import Config
from tests.fixtures.shared_encoder import (
```

with:

```python
)
from naics_embedder.text_model.fields import QUERY, tokenize_field
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.utils.config import Config
from tests.fixtures.shared_encoder import (
```

Replace:

```python
    assert np.allclose(-points[:, 0]**2 + (points[:, 1:]**2).sum(axis=1), -1.0, atol=1e-4)

def test_the_hgcn_feeder_refuses_a_checkpoint_trained_on_truncated_text(
    monkeypatch, tmp_path, truncated_checkpoint, validated_bundle, five_code_descriptions_parquet
```

with:

```python
    assert np.allclose(-points[:, 0]**2 + (points[:, 1:]**2).sum(axis=1), -1.0, atol=1e-4)

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_the_hgcn_feeder_refuses_a_flat_arm_before_anything_is_read(
    monkeypatch, tmp_path, geometry_checkpoint, geometry
):
    '''HGCN refines Lorentz points, which only the hyperbolic arm has (Req 12, P10).'''

    checkpoint = geometry_checkpoint(geometry)

    def never(_cfg):
        raise AssertionError('the bundle was read before the flat arm was refused')

    monkeypatch.setattr(training_cli, 'require_valid_supervision_bundle', never)
    forbid_model_loads(monkeypatch)
    cfg = Config()
    cfg.model.geometry = geometry
    output = tmp_path / 'encodings.parquet'

    with pytest.raises(ValueError, match=f'a {geometry} arm has none'):
        training_cli.generate_embeddings_from_checkpoint(str(checkpoint), cfg, str(output))
    assert not output.exists()

def test_the_hgcn_feeder_refuses_a_checkpoint_trained_on_truncated_text(
    monkeypatch, tmp_path, truncated_checkpoint, validated_bundle, five_code_descriptions_parquet
```

Replace:

```python
    assert (np.linalg.norm(tangent.numpy(), axis=1) <= bound).all()
    provenance = json.loads(provenance_path(exported_table).read_text())
    assert provenance['coordinates'] == COORDINATES
    assert COORDINATES.startswith('the bounded tangent vector at the origin')

def test_a_read_rebuilds_the_head_at_the_checkpoints_radius_bound(
```

with:

```python
    assert (np.linalg.norm(tangent.numpy(), axis=1) <= bound).all()
    provenance = json.loads(provenance_path(exported_table).read_text())
    assert provenance['geometry'] == 'hyperbolic'
    assert provenance['coordinates'] == COORDINATES['hyperbolic']

def test_the_hyperbolic_coordinates_are_named_as_stage_7_named_them():
    '''A Stage 7 table's provenance and a Stage 8 hyperbolic export's name one form (Req 12).'''

    assert tuple(COORDINATES) == GEOMETRIES
    assert COORDINATES['hyperbolic'] == (
        'the bounded tangent vector at the origin, r * u with r = R * tanh(|v| / R) (Req 13); '
        'no time coordinate'
    )

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_each_geometry_arm_exports_its_own_coordinates_and_names_them(
    tmp_path, geometry_checkpoint, validated_bundle, five_code_token_config, geometry
):
    '''Req 2's form of each arm (Req 12): the head's tangent, with the geometry it is read in.'''

    checkpoint = geometry_checkpoint(geometry)

    exported = export_code_table(
        checkpoint, validated_bundle, five_code_token_config, tmp_path / f'{geometry}.parquet'
    )

    model, _ = load_arm_model(checkpoint, validated_bundle, summaries=summaries_identity(MINILM))
    assert model.encoder.head.geometry == geometry
    rows = five_code_token_rows(five_code_token_config, validated_bundle)
    tangent = encode_token_rows(model, rows)['tangent'].numpy()
    matrix = pl.read_parquet(exported).select(COORDINATE_COLUMNS).to_numpy()
    assert np.array_equal(matrix, tangent)
    if geometry == 'spherical':
        # û, not v: the sphere's points are unit vectors
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0.0, atol=1e-6)
    provenance = json.loads(provenance_path(exported).read_text())
    assert provenance['geometry'] == geometry
    assert provenance['coordinates'] == COORDINATES[geometry]
    assert provenance['contract']['encoder']['geometry'] == geometry
    assert provenance['dimension'] == ARM_DIMENSION

def test_a_read_rebuilds_the_head_at_the_checkpoints_radius_bound(
```

In `tests/unit/test_monitor.py`, make these 6 edits, in order.

Replace:

```python
from transformers import AutoTokenizer

from naics_embedder.panels.decoding import score_decoding
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
```

with:

```python
from transformers import AutoTokenizer

from naics_embedder.panels.decoding import GEOMETRY_DISTANCES, score_decoding
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.text_only import matrix_fingerprint, provenance_path
```

Replace:

```python
    export_code_table,
)
from naics_embedder.text_model.hyperbolic import exp_map_origin
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
```

with:

```python
    export_code_table,
)
from naics_embedder.text_model.heads import GEOMETRIES
from naics_embedder.text_model.hyperbolic import exp_map_origin
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
```

Replace:

```python
    )

@pytest.fixture
def reference_model(tiny_backbone, reference_manifest, reference_bundle) -> NAICSContrastiveModel:
    '''A d = 16 masked-mean model of the reference bundle on the tiny backbone, in eval mode.'''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
```

with:

```python
    )

@pytest.fixture
def reference_model(
    request, tiny_backbone, reference_manifest, reference_bundle
) -> NAICSContrastiveModel:
    '''
    A d = 16 masked-mean model of the reference bundle on the tiny backbone, in eval mode, in the
    geometry arm an indirect parameter names (Req 12).
    '''

    model = NAICSContrastiveModel(
        base_model_name=MINILM,
```

Replace:

```python
        fusion='masked_mean',
        dimension=ARM_DIMENSION,
        supervision_manifest_path=str(reference_manifest),
        summaries=summaries_identity(MINILM),
```

with:

```python
        fusion='masked_mean',
        dimension=ARM_DIMENSION,
        geometry=request.param,
        supervision_manifest_path=str(reference_manifest),
        summaries=summaries_identity(MINILM),
```

Replace:

```python
    )
    return [cache[code_id] for code_id in range(len(cache))]

def test_the_live_read_is_the_read_of_the_export_of_its_cache(
    tmp_path, tokenizer, reference_model, reference_bundle, reference_token_config
):
    '''
    Spec 4.4's agreement, on the CPU and exactly: the live encoder on the model and its cache, and
    the arm encoder on the model's checkpoint and the table exported from it, decode the
    validation split alike.
    '''

    rows = _code_rows(reference_bundle, reference_token_config)
```

with:

```python
    )
    return [cache[code_id] for code_id in range(len(cache))]

@pytest.mark.parametrize('reference_model', GEOMETRIES, indirect=True)
def test_the_live_read_is_the_read_of_the_export_of_its_cache(
    tmp_path, tokenizer, reference_model, reference_bundle, reference_token_config
):
    '''
    Spec 4.4's agreement, on the CPU and exactly, in every geometry arm: the live encoder on the
    model and its cache, and the arm encoder on the model's checkpoint and the table exported
    from it, decode the validation split alike, under the arm's own distance (Req 12).
    '''

    rows = _code_rows(reference_bundle, reference_token_config)
```

Replace:

```python
    read = outcome_monitor.read(reference_model, cache, training_run='run-a', seed=0, epoch=0)
    live_encoder = monitor.LiveEncoder(reference_model, cache, tokenizer, REFERENCE_WINDOW)
    live = live_panel.score(live_encoder, 'validation', PURPOSE, distance='lorentz')
    exported = read_outcome_validation(
        arm, OutcomePanel.from_bundle(reference_bundle, tmp_path / 'arm_log.jsonl'), PURPOSE
```

with:

```python
    read = outcome_monitor.read(reference_model, cache, training_run='run-a', seed=0, epoch=0)
    live_encoder = monitor.LiveEncoder(reference_model, cache, tokenizer, REFERENCE_WINDOW)
    distance = GEOMETRY_DISTANCES[reference_model.encoder.head.geometry]
    assert live_encoder.distance == arm.distance == distance
    assert read.record['detail']['distance'] == distance
    live = live_panel.score(live_encoder, 'validation', PURPOSE, distance=distance)
    exported = read_outcome_validation(
        arm, OutcomePanel.from_bundle(reference_bundle, tmp_path / 'arm_log.jsonl'), PURPOSE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_arm_encoder.py tests/unit/test_cli_training.py tests/unit/test_export.py tests/unit/test_monitor.py -n auto -q
```
Expected: FAIL, `13 failed, 164 passed`:
- the provenance has no `geometry` (`KeyError: 'geometry'`), and `COORDINATES` is still Stage 7's
  string;
- the flat reads and the monitor's live reads decode by `'lorentz'`, not `'euclidean'` or
  `'cosine'`;
- the feeder reads the bundle before it refuses the flat arm;
- `train` still asks the flat arm `Generate embeddings … checkpoint?`.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/cli/commands/training.py`, make these 3 edits, in order.

Replace:

```python
    Loads a trained model checkpoint, runs inference on all NAICS codes, and
    writes the resulting embeddings to a parquet file compatible with HGCN
    training. The checkpoint must carry the supervision contract of the configured run, the
    contract of its validated bundle; a checkpoint without one, or with another, is refused
    before its model loads. One trained under another objective, as every checkpoint saved
    before Stage 7 was, is refused first, and nothing migrates it (spec 4.5, D2).

    Args:
```

with:

```python
    Loads a trained model checkpoint, runs inference on all NAICS codes, and
    writes the resulting embeddings to a parquet file compatible with HGCN
    training. HGCN refines Lorentz points, which only the hyperbolic arm has, so a configured
    flat arm is refused before anything is read (Req 12). The checkpoint must carry the
    supervision contract of the configured run, the contract of its validated bundle, its
    geometry included; a checkpoint without one, or with another, is refused before its model
    loads. One trained under another objective, as every checkpoint saved before Stage 7 was, is
    refused first, and nothing migrates it (spec 4.5, D2).

    Args:
```

Replace:

```python
    Returns:
        str: Filesystem path to the generated embeddings parquet file.
    '''

    logger.info('=' * 80)
    logger.info('GENERATING EMBEDDINGS FROM CHECKPOINT')
```

with:

```python
    Returns:
        str: Filesystem path to the generated embeddings parquet file.

    Raises:
        ValueError: If the configured arm is not hyperbolic, or the checkpoint's contract is not
            the configured run's.
    '''

    geometry = config.model.geometry
    if geometry != 'hyperbolic':
        raise ValueError(
            f'the HGCN feeder writes Lorentz points, and a {geometry} arm has none: HGCN refines '
            'the hyperbolic arm only (Req 12)'
        )
    logger.info('=' * 80)
    logger.info('GENERATING EMBEDDINGS FROM CHECKPOINT')
```

Replace:

```python
        )

        # Ask about embeddings for HGCN training (the feeder stays until Stage 11), but only on a
        # terminal: a remote launch reads stdin from /dev/null, where typer.confirm would abort the
        # finished run (spec 4.5, "Stays"). Without one, the answer is the question's default, no.
        generate_embeddings = False
        if _stdin_is_terminal():
            console.print('\n[bold cyan]Generate embeddings for HGCN training?[/bold cyan]')
            generate_embeddings = typer.confirm(
```

with:

```python
        )

        # Ask about embeddings for HGCN training (the feeder stays until Stage 11), but only for a
        # hyperbolic arm, the one with Lorentz points (Req 12), and only on a terminal: a remote
        # launch reads stdin from /dev/null, where typer.confirm would abort the finished run
        # (spec 4.5, "Stays"). Otherwise the answer is the question's default, no.
        generate_embeddings = False
        if cfg.model.geometry != 'hyperbolic':
            logger.info(
                f'a {cfg.model.geometry} arm has no Lorentz points: no HGCN embeddings question'
            )
        elif _stdin_is_terminal():
            console.print('\n[bold cyan]Generate embeddings for HGCN training?[/bold cyan]')
            generate_embeddings = typer.confirm(
```

In `src/naics_embedder/text_model/arm_encoder.py`, make these 6 edits, in order.

Replace:

```python
- A code's vector is decoded from the table ``tools export-table`` wrote from the checkpoint.

Both pass through one float64 exp map at the origin. The code vectors a read decodes against are
therefore a fixed function of the table, and the ``matrix_fingerprint`` the read logs names them.
Checkpoint, table, encoder and distance are the pieces of Stage 4's ``SeedArtifacts``
(``decision/sweep.py``).
```

with:

```python
- A code's vector is decoded from the table ``tools export-table`` wrote from the checkpoint.

Both pass through the head's float64 read map (``read_points``): the exp map at the origin in the
hyperbolic arm, and the coordinates as they are in the flat arms (Req 12). The code points a read
decodes against are therefore a fixed function of the table, and the ``matrix_fingerprint`` the
read logs names them.
Checkpoint, table, encoder and distance are the pieces of Stage 4's ``SeedArtifacts``
(``decision/sweep.py``).
```

Replace:

```python
    encode_query_texts,
    load_arm_model,
)
from naics_embedder.text_model.hyperbolic import exp_map_origin
from naics_embedder.utils.config import TokenizationConfig

# -------------------------------------------------------------------------------------------------
```

with:

```python
    encode_query_texts,
    load_arm_model,
)
from naics_embedder.text_model.heads import head_of
from naics_embedder.utils.config import TokenizationConfig

# -------------------------------------------------------------------------------------------------
```

Replace:

```python
        batch_size: Queries per forward pass.

    Attributes:
        distance: The head's distance, ``'lorentz'`` for the hyperbolic head.
        table_fingerprint: The table's ``matrix_fingerprint``, which a read logs as ``table``.
        checkpoint_sha256: The checkpoint's SHA-256, which a read logs as ``checkpoint``.
    '''
```

with:

```python
        batch_size: Queries per forward pass.

    Attributes:
        head: The model's geometry head: its read map takes the queries' and the codes'
            coordinates to the points its distance reads.
        distance: The head's decoding distance: ``'lorentz'``, ``'euclidean'`` or ``'cosine'``
            for the hyperbolic, Euclidean and spherical arms (Req 12).
        table_fingerprint: The table's ``matrix_fingerprint``, which a read logs as ``table``.
        checkpoint_sha256: The checkpoint's SHA-256, which a read logs as ``checkpoint``.
    '''
```

Replace:

```python
        self.max_length = max_length
        self.batch_size = batch_size
        self.distance = model.encoder.head.distance
        self.table_fingerprint = matrix_fingerprint(codes, matrix)
        self.checkpoint_sha256 = checkpoint_sha256
```

with:

```python
        self.max_length = max_length
        self.batch_size = batch_size
        self.head = head_of(model)
        self.distance = self.head.distance
        self.table_fingerprint = matrix_fingerprint(codes, matrix)
        self.checkpoint_sha256 = checkpoint_sha256
```

Replace:

```python
        )

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        '''Marked ``query:`` texts through the model, then the exp map: (Q, d + 1), float64.'''

        tangent = encode_query_texts(
            self.model, self.tokenizer, texts, self.max_length, batch_size=self.batch_size
        )
        return exp_map_origin(tangent)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        '''
        The codes' table rows through the exp map: (C, d + 1), float64.

        Raises:
            ValueError: If a code has no row in the table.
```

with:

```python
        )

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        '''
        Marked ``query:`` texts through the model, then the head's read map: float64, (Q, d + 1)
        in the hyperbolic arm and (Q, d) in the flat arms.
        '''

        tangent = encode_query_texts(
            self.model, self.tokenizer, texts, self.max_length, batch_size=self.batch_size
        )
        return self.head.read_points(tangent)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        '''
        The codes' table rows through the head's read map: float64, (C, d + 1) in the hyperbolic
        arm and (C, d) in the flat arms.

        Raises:
            ValueError: If a code has no row in the table.
```

Replace:

```python
        if unknown:
            raise ValueError(f'the table has no row for {unknown[:5]} ({len(unknown)} codes)')
        return exp_map_origin(self._tangent[[self._rows[code] for code in codes]])

# -------------------------------------------------------------------------------------------------
```

with:

```python
        if unknown:
            raise ValueError(f'the table has no row for {unknown[:5]} ({len(unknown)} codes)')
        return self.head.read_points(self._tangent[[self._rows[code] for code in codes]])

# -------------------------------------------------------------------------------------------------
```

In `src/naics_embedder/text_model/export.py`, make these 8 edits, in order.

Replace:

```python
read. The last three default to batches of ``ENCODE_BATCH_SIZE``.

``export_code_table`` writes Req 2's form of an arm: ``code``, ``index``, ``level`` and
``e0 … e{d-1}``, each code's bounded tangent vector at the origin (R6, Req 13), in the bundle's
codebook order. Its provenance ties the table to its checkpoint.
'''

# -------------------------------------------------------------------------------------------------
```

with:

```python
read. The last three default to batches of ``ENCODE_BATCH_SIZE``.

``export_code_table`` writes Req 2's form of an arm: ``code``, ``index``, ``level`` and
``e0 … e{d-1}``, in the bundle's codebook order. They are each code's coordinates in its geometry
arm's form (``COORDINATES``): the bounded tangent vector at the origin in the hyperbolic arm (R6,
Req 13), v in the Euclidean arm and û in the spherical arm (Req 12). Its provenance ties the table
to its checkpoint and names its geometry.
'''

# -------------------------------------------------------------------------------------------------
```

Replace:

```python
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.fields import CHANNELS, QUERY, tokenize_field
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import Config, TokenizationConfig

logger = logging.getLogger(__name__)

TABLE_PREFIX = 'e'
COORDINATES = (
    'the bounded tangent vector at the origin, r * u with r = R * tanh(|v| / R) (Req 13); '
    'no time coordinate'
)
# Rows per forward pass wherever an arm is encoded: the export, the reads, the training cache and
# the monitor. A chunk is trimmed to its longest text, so the batches set the backbone's shapes,
```

with:

```python
from naics_embedder.text_model.dataloader.tokenization_cache import tokenization_cache
from naics_embedder.text_model.fields import CHANNELS, QUERY, tokenize_field
from naics_embedder.text_model.heads import head_of
from naics_embedder.text_model.naics_model import NAICSContrastiveModel
from naics_embedder.utils.config import Config, TokenizationConfig

logger = logging.getLogger(__name__)

TABLE_PREFIX = 'e'
# What each geometry arm's e0 … e{d-1} are (Req 2, Req 12), as the provenance names them. The
# hyperbolic arm's are named as Stage 7's tables name them
COORDINATES = {
    'euclidean': 'the point v itself, the projection of the fused text (Req 12)',
    'spherical': 'the unit direction u = v / |v| on the sphere (Req 12)',
    'hyperbolic': (
        'the bounded tangent vector at the origin, r * u with r = R * tanh(|v| / R) (Req 13); '
        'no time coordinate'
    ),
}
# Rows per forward pass wherever an arm is encoded: the export, the reads, the training cache and
# the monitor. A chunk is trimmed to its longest text, so the batches set the backbone's shapes,
```

Replace:

```python
        batch_size: Rows per forward pass.

    Returns:
        ``tangent`` (N, d), ``embedding`` (N, d + 1), ``radius`` (N,) and ``direction`` (N, d),
        float64 on the CPU, in row order.

    Raises:
        ValueError: If there are no rows, or ``batch_size`` is not positive.
```

with:

```python
        batch_size: Rows per forward pass.

    Returns:
        ``tangent`` (N, d), ``embedding`` (N, d + 1) in the hyperbolic arm and (N, d) in the flat
        arms, ``radius`` (N,) and ``direction`` (N, d), float64 on the CPU, in row order.

    Raises:
        ValueError: If there are no rows, or ``batch_size`` is not positive.
```

Replace:

```python
        batch_size: Queries per forward pass.

    Returns:
        The queries' bounded tangent vectors at the origin (Q, d), float64 on the CPU.

    Raises:
        ValueError: As ``encode_token_rows``, if there are no texts.
```

with:

```python
        batch_size: Queries per forward pass.

    Returns:
        The queries' coordinates in the arm's export form (the head's ``tangent``), (Q, d),
        float64 on the CPU.

    Raises:
        ValueError: As ``encode_token_rows``, if there are no texts.
```

Replace:

```python
    Load an arm's checkpoint for export or a read, refusing it before any weight loads.

    The checkpoint's own hyperparameters rebuild its fusion and dimension, so its encoder record
    is never compared with a config (spec 4.4). It must have been trained under Req 11's
    objective, its supervision fields must match ``bundle``, and its summaries ``summaries``.
    Curvature is fixed at 1 with no parameter (spec 4.2), so there is none to check.
```

with:

```python
    Load an arm's checkpoint for export or a read, refusing it before any weight loads.

    The checkpoint's own hyperparameters rebuild its fusion, dimension and geometry, so its
    encoder record is never compared with a config (spec 4.4). It must have been trained under Req 11's
    objective, its supervision fields must match ``bundle``, and its summaries ``summaries``.
    Curvature is fixed at 1 with no parameter (spec 4.2), so there is none to check.
```

Replace:

```python
    Every code goes through the checkpoint's model in eval mode, without gradient. The table
    holds ``code``, then ``index`` and ``level`` from the descriptions (Int64), then ``e0 …
    e{d-1}`` (float64): each code's bounded tangent vector at the origin, in the bundle's codebook
    order. The provenance is ``<stem>_provenance.json``.

    Args:
```

with:

```python
    Every code goes through the checkpoint's model in eval mode, without gradient. The table
    holds ``code``, then ``index`` and ``level`` from the descriptions (Int64), then ``e0 …
    e{d-1}`` (float64): each code's coordinates in its arm's form (``COORDINATES``), in the
    bundle's codebook order. The provenance is ``<stem>_provenance.json``; it names the arm's
    geometry and coordinates.

    Args:
```

Replace:

```python
    table.write_parquet(output_path)
    checkpoint_path = Path(checkpoint_path)
    provenance: Dict[str, Any] = {
        'checkpoint': {
```

with:

```python
    table.write_parquet(output_path)
    checkpoint_path = Path(checkpoint_path)
    geometry = head_of(model).geometry
    provenance: Dict[str, Any] = {
        'checkpoint': {
```

Replace:

```python
        'codes': table.height,
        'dimension': tangent.shape[1],
        'coordinates': COORDINATES,
        'table_sha256': sha256_file(output_path),
        'matrix_fingerprint': fingerprint,
```

with:

```python
        'codes': table.height,
        'dimension': tangent.shape[1],
        'geometry': geometry,
        'coordinates': COORDINATES[geometry],
        'table_sha256': sha256_file(output_path),
        'matrix_fingerprint': fingerprint,
```

In `src/naics_embedder/text_model/monitor.py`, make these 7 edits, in order.

Replace:

```python
``OutcomePanel.score_logged``, the path every read takes, so the selection log records the read.
Its ``LiveEncoder`` encodes queries through the live model and decodes codes from the cache, and
both go through the float64 exp map ``ArmEncoder`` uses. The cache is encoded as the export
encodes the code table, in the same batches and order, so on the CPU a read of the live model and
a read of the table exported from its checkpoint agree exactly.

Each read's record, as logged, goes with its MRR to ``monitor_reads.jsonl`` in the run's checkpoint
```

with:

```python
``OutcomePanel.score_logged``, the path every read takes, so the selection log records the read.
Its ``LiveEncoder`` encodes queries through the live model and decodes codes from the cache, and
both go through the head's float64 read map, as ``ArmEncoder``'s do, under the head's distance
(Req 12). The cache is encoded as the export encodes the code table, in the same batches and
order, so on the CPU a read of the live model and a read of the table exported from its
checkpoint agree exactly.

Each read's record, as logged, goes with its MRR to ``monitor_reads.jsonl`` in the run's checkpoint
```

Replace:

```python
    encode_token_rows,
)
from naics_embedder.text_model.hyperbolic import exp_map_origin

logger = logging.getLogger(__name__)
```

with:

```python
    encode_token_rows,
)
from naics_embedder.text_model.heads import head_of

logger = logging.getLogger(__name__)
```

Replace:

```python
        codes: The codes, in codebook order.
        radius: Each code's radius r, (N,) float32 on the model's device.
        direction: Each code's direction û, (N, d) float32 on the model's device.
        tangent: Each code's bounded tangent vector r · û, (N, d) float64 on the CPU: what the
            monitor decodes against, and what the export writes.
    '''

    codes: Tuple[str, ...]
```

with:

```python
        codes: The codes, in codebook order.
        radius: Each code's radius r, (N,) float32 on the model's device.
        direction: Each code's direction û, (N, d) float32 on the model's device.
        tangent: Each code's coordinates in its arm's export form, the head's ``tangent``, (N, d)
            float64 on the CPU: r · û in the hyperbolic arm, v in the Euclidean and û in the
            spherical (Req 12). It is what the monitor decodes against and what the export writes.
    '''

    codes: Tuple[str, ...]
```

Replace:

```python
    '''
    ``QueryCodeEncoder`` for the live model (spec 4.4): queries through the model, codes from the
    code cache, and both through ``exp_map_origin``.

    A query is encoded as ``ArmEncoder`` encodes one (``encode_query_texts``), in eval mode and
```

with:

```python
    '''
    ``QueryCodeEncoder`` for the live model (spec 4.4): queries through the model, codes from the
    code cache, and both through the head's read map (``read_points``).

    A query is encoded as ``ArmEncoder`` encodes one (``encode_query_texts``), in eval mode and
```

Replace:

```python
        batch_size: Queries per forward pass.

    Attributes:
        distance: ``'lorentz'``: every point is the exp map of a tangent at the origin of the
            c = 1 hyperboloid.
    '''

    distance = 'lorentz'

    def __init__(
        self,
```

with:

```python
        batch_size: Queries per forward pass.

    Attributes:
        head: The model's geometry head.
        distance: The head's decoding distance, as ``ArmEncoder``'s: ``'lorentz'`` in the
            hyperbolic arm, ``'euclidean'`` or ``'cosine'`` in the flat arms (Req 12).
    '''

    def __init__(
        self,
```

Replace:

```python
        self.max_length = max_length
        self.batch_size = batch_size
        self._rows = {code: row for row, code in enumerate(cache.codes)}

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        '''Marked ``query:`` texts through the live model, then the exp map: (Q, d + 1), float64.'''

        with _live_encode(self.model):
            tangent = encode_query_texts(
                self.model, self.tokenizer, texts, self.max_length, batch_size=self.batch_size
            )
        return exp_map_origin(tangent)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        '''
        The codes' cached tangents through the exp map: (C, d + 1), float64.

        Raises:
```

with:

```python
        self.max_length = max_length
        self.batch_size = batch_size
        self.head = head_of(model)
        self.distance = self.head.distance
        self._rows = {code: row for row, code in enumerate(cache.codes)}

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        '''Marked ``query:`` texts through the live model, then the head's read map: float64.'''

        with _live_encode(self.model):
            tangent = encode_query_texts(
                self.model, self.tokenizer, texts, self.max_length, batch_size=self.batch_size
            )
        return self.head.read_points(tangent)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        '''
        The codes' cached coordinates through the head's read map: float64.

        Raises:
```

Replace:

```python
        if unknown:
            raise ValueError(f'the code cache has no row for {unknown[:5]} ({len(unknown)} codes)')
        return exp_map_origin(self.cache.tangent[[self._rows[code] for code in codes]])

# -------------------------------------------------------------------------------------------------
```

with:

```python
        if unknown:
            raise ValueError(f'the code cache has no row for {unknown[:5]} ({len(unknown)} codes)')
        return self.head.read_points(self.cache.tangent[[self._rows[code] for code in codes]])

# -------------------------------------------------------------------------------------------------
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_arm_encoder.py tests/unit/test_cli_training.py tests/unit/test_export.py tests/unit/test_monitor.py -n auto -q
```
Expected: PASS, `177 passed`.

- [ ] **Step 5: Format, lint and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/cli/commands/training.py src/naics_embedder/text_model/arm_encoder.py src/naics_embedder/text_model/export.py src/naics_embedder/text_model/monitor.py tests/fixtures/shared_encoder.py tests/unit/test_arm_encoder.py tests/unit/test_cli_training.py tests/unit/test_export.py tests/unit/test_monitor.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run pytest -n auto -q`
Expected: `3071 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/cli/commands/training.py src/naics_embedder/text_model/arm_encoder.py src/naics_embedder/text_model/export.py src/naics_embedder/text_model/monitor.py tests/fixtures/shared_encoder.py tests/unit/test_arm_encoder.py tests/unit/test_cli_training.py tests/unit/test_export.py tests/unit/test_monitor.py
git commit -m "feat(export): export and read each geometry arm's own coordinates (Req 2, Req 12)"
```

### Task 5: The decision's distance and dimension guards

Stage 4 left two tie-order keys unguarded (the roadmap's Produces line). This task guards them
and adds one guard of its own (P8):
- **The distance.** `check_seed_distance` refuses a seed whose encoder decodes by another distance
  than its arm's geometry. `run_seed_sweep` calls it right after `check_seed_table`, before any
  read is logged.
- **The logged reads.** `check_arm` compares each outcome read's logged `distance` and each
  regressor read's logged `dimension` with the arm's. The decision fixture's reads had hard-coded
  16 and `lorentz`. `_check_monitor_records` compares each monitor record's `distance` too.
- **The preflight.** `CheckpointRunner.check` refuses a monitor record read under another distance
  than the arm's geometry, before anything is exported (**Recorded deviations**).

The fixtures derive each logged read's distance and dimension from its arm's spec.
`monitor_records` takes the distance, and the sweep tests' arm is spherical by default, so a
distance check bites there.

The tests cover:
- `check_seed_distance` for each geometry;
- a flat arm's logged reads checked against its own distance and dimension;
- the sweep refusing a seed of another distance before any read;
- the runner refusing monitor reads under another distance, and `tools sweep` checking a later
  seed's distance before the first decision read.

**Files:**
- Modify: `src/naics_embedder/decision/decide.py`, lines 9-16, 76-79, 138-141, 186-198, 218-224,
  243-247, 260-263
- Modify: `src/naics_embedder/decision/sweep.py`, lines 28-32, 135-141, 149-154
- Modify: `src/naics_embedder/text_model/checkpoint_runner.py`, lines 18-21, 96-102, 126-129,
  134-137
- Modify: `tests/fixtures/decision.py`, lines 33-36, 213-225, 240-253, 264-268, 310-314, 330-334
- Test: `tests/unit/test_checkpoint_runner.py`, lines 87-90, 220-223, 313-325
- Test: `tests/unit/test_decision.py`, lines 16-20, 295-298, 312-318, 326-329, 579-582, 600-603,
  612-615
- Test: `tests/unit/test_decision_sweep.py`, lines 37-40, 115-119, 182-186, 226-232, 279-283,
  320-324, 345-349, 360-364

**Interfaces:**
- Consumes: Task 1's `GEOMETRY_DISTANCES`; Task 4's `ArmEncoder.distance`, which
  `SeedArtifacts.distance` carries from the export; Task 2's `ArmSpec.geometry` from
  `_sweep_spec`.
- Produces:
  - `naics_embedder.decision.decide.check_seed_distance(spec: ArmSpec, seed: int, distance: str)
    -> None`. It raises `ValueError("<arm> seed <s>: the encoder decodes by '<distance>', but a
    <geometry> arm decodes by '<expected>' (Req 12)")`.
  - `check_arm` raises `ValueError('<arm> seed <s>: a <panel> read names another [...]')` when a
    logged read's `distance` or `dimension` is not the arm's.
  - `CheckpointRunner.check` raises `ValueError("<arm> seed <s>: a monitor record reads by
    '<distance>', but a <geometry> arm decodes by '<expected>' (Req 12)")`.
  - In `tests/fixtures/decision.py`, `_read(panel, run_id, table, text_only, time, arm_spec)` and
    `monitor_records(..., distance='lorentz')`.

- [ ] **Step 1: Write the failing tests**

In `tests/fixtures/decision.py`, make these 6 edits, in order.

Replace:

```python
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import table_fingerprint
```

with:

```python
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.decoding import GEOMETRY_DISTANCES
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import table_fingerprint
```

Replace:

```python
    return path

def _read(panel: str, run_id: str, table: str, text_only: str, time: str) -> Dict:
    '''A log record shaped as the panel writes it.'''

    detail = {'run': run_id, 'arm_name': run_id.split('/')[0], 'seed': int(run_id.split('-')[-1])}
    if panel == OUTCOME_PANEL:
        detail.update(encoder='SyntheticEncoder', distance='cosine', table=table)
        fingerprint = PANEL_SET.outcome
    else:
        detail.update(level=6, comparators=[], arm=table, text_only=text_only, dimension=16)
        fingerprint = PANEL_SET.regressor
    return {
```

with:

```python
    return path

def _read(
    panel: str, run_id: str, table: str, text_only: str, time: str, arm_spec: ArmSpec
) -> Dict:
    '''A log record shaped as the panel writes it, of a read of ``arm_spec``'s seed.'''

    detail = {'run': run_id, 'arm_name': run_id.split('/')[0], 'seed': int(run_id.split('-')[-1])}
    if panel == OUTCOME_PANEL:
        distance = GEOMETRY_DISTANCES[arm_spec.geometry]
        detail.update(encoder='SyntheticEncoder', distance=distance, table=table)
        fingerprint = PANEL_SET.outcome
    else:
        detail.update(
            level=6, comparators=[], arm=table, text_only=text_only, dimension=arm_spec.dimension
        )
        fingerprint = PANEL_SET.regressor
    return {
```

Replace:

```python
    *,
    time: str,
    fingerprint: str = PANEL_SET.outcome,
) -> List[Dict]:
    '''
    A training run's monitor records, one per epoch from 0, as ``monitor_reads.jsonl`` holds them:
    the epoch's MRR, and its read of the outcome panel's validation split as the selection log
    appended it.

    Each read names the training run, the seed, the epoch and that epoch's code cache, whose
    fingerprint is never a stored table's.
    '''

    return [
```

with:

```python
    *,
    time: str,
    fingerprint: str = PANEL_SET.outcome,
    distance: str = 'lorentz',
) -> List[Dict]:
    '''
    A training run's monitor records, one per epoch from 0, as ``monitor_reads.jsonl`` holds them:
    the epoch's MRR, and its read of the outcome panel's validation split as the selection log
    appended it.

    Each read names its distance (the hyperbolic arm's by default), the training run, the seed,
    the epoch and that epoch's code cache, whose fingerprint is never a stored table's.
    '''

    return [
```

Replace:

```python
                'detail': {
                    'encoder': 'LiveEncoder',
                    'distance': 'lorentz',
                    'training_run': training_run,
                    'seed': seed,
```

with:

```python
                'detail': {
                    'encoder': 'LiveEncoder',
                    'distance': distance,
                    'training_run': training_run,
                    'seed': seed,
```

Replace:

```python
                # argmax returns the first of tied maxima: the earliest best epoch
                'checkpoint_epoch': int(np.argmax(monitor_mrrs)),
                'monitor_records': monitor_records(training_run, seed, monitor_mrrs, time=time),
            }
        runs.append(
```

with:

```python
                # argmax returns the first of tied maxima: the earliest best epoch
                'checkpoint_epoch': int(np.argmax(monitor_mrrs)),
                'monitor_records': monitor_records(
                    training_run,
                    seed,
                    monitor_mrrs,
                    time=time,
                    distance=GEOMETRY_DISTANCES[arm_spec.geometry],
                ),
            }
        runs.append(
```

Replace:

```python
                    _read(
                        panel, run_id, table.matrix_fingerprint, text_only.table.matrix_fingerprint,
                        time
                    ) for panel in PANELS
                ],
```

with:

```python
                    _read(
                        panel, run_id, table.matrix_fingerprint, text_only.table.matrix_fingerprint,
                        time, arm_spec
                    ) for panel in PANELS
                ],
```

In `tests/unit/test_checkpoint_runner.py`, make these 3 edits, in order.

Replace:

```python
                'fingerprint': 'fixture',
                'detail': {
                    'epoch': epoch,
                    'seed': 7,
```

with:

```python
                'fingerprint': 'fixture',
                'detail': {
                    'distance': 'lorentz',
                    'epoch': epoch,
                    'seed': 7,
```

Replace:

```python
        _runner(fixture_run).check(spec, 7)

def test_check_reads_a_seed_saved_before_stage_8_as_hyperbolic(fixture_run):
    '''P7: Stage 7's checkpoints name no geometry, and the reference arm reads them unchanged.'''
```

with:

```python
        _runner(fixture_run).check(spec, 7)

def test_check_refuses_monitor_reads_under_another_distance_than_the_arms(fixture_run):
    '''Req 12: an arm's monitor reads decode by its own distance, or they selected no checkpoint.'''

    records = fixture_run.records
    records[1]['read']['detail']['distance'] = 'cosine'
    _write_records(fixture_run.directory, records)
    message = (
        "seed 7: a monitor record reads by 'cosine', but a hyperbolic arm decodes by 'lorentz' "
        '\\(Req 12\\)'
    )

    with pytest.raises(ValueError, match=message):
        _runner(fixture_run).check(fixture_run.spec, 7)

def test_check_reads_a_seed_saved_before_stage_8_as_hyperbolic(fixture_run):
    '''P7: Stage 7's checkpoints name no geometry, and the reference arm reads them unchanged.'''
```

Replace:

```python
    check_arm(arm, ArtifactStore(sweep_env.output.parent / 'store'), min_seeds=5)

@pytest.mark.parametrize(
    'problem', ['missing-epoch', 'fingerprint', 'seed', 'split', 'training_run']
)
def test_tools_sweep_checks_a_later_seed_before_the_first_decision_read(sweep_env, problem):
    directory = sweep_env.root / 'seed-5'
    records = read_monitor_records(directory / MONITOR_RECORDS)
    if problem == 'missing-epoch':
        records = records[1:]
    elif problem in ('seed', 'training_run'):
        records[0]['read']['detail'][problem] = 999 if problem == 'seed' else 'other-training'
    else:
```

with:

```python
    check_arm(arm, ArtifactStore(sweep_env.output.parent / 'store'), min_seeds=5)

@pytest.mark.parametrize(
    'problem', ['missing-epoch', 'fingerprint', 'seed', 'split', 'training_run', 'distance']
)
def test_tools_sweep_checks_a_later_seed_before_the_first_decision_read(sweep_env, problem):
    directory = sweep_env.root / 'seed-5'
    records = read_monitor_records(directory / MONITOR_RECORDS)
    if problem == 'missing-epoch':
        records = records[1:]
    elif problem == 'distance':
        records[0]['read']['detail']['distance'] = 'cosine'
    elif problem in ('seed', 'training_run'):
        records[0]['read']['detail'][problem] = 999 if problem == 'seed' else 'other-training'
    else:
```

In `tests/unit/test_decision.py`, make these 7 edits, in order.

Replace:

```python
from pydantic import ValidationError

from naics_embedder.decision.decide import check_arm, decide, fix_margins
from naics_embedder.decision.records import (
    ArmRecord,
```

with:

```python
from pydantic import ValidationError

from naics_embedder.decision.decide import check_arm, check_seed_distance, decide, fix_margins
from naics_embedder.decision.records import (
    ArmRecord,
```

Replace:

```python
        (lambda data: _first_read(data)['detail'].update(run='other'), 'another run'),
        (lambda data: _first_read(data)['detail'].update(table='other'), "another \\['table'\\]"),
        (lambda data: _first_read(data).update(fingerprint='other'), 'another'),
        (lambda data: data['runs'][0]['log_records'].pop(), 'not each of'),
```

with:

```python
        (lambda data: _first_read(data)['detail'].update(run='other'), 'another run'),
        (lambda data: _first_read(data)['detail'].update(table='other'), "another \\['table'\\]"),
        (
            lambda data: _first_read(data)['detail'].update(distance='cosine'),
            "another \\['distance'\\]",
        ),
        (lambda data: _first_read(data).update(fingerprint='other'), 'another'),
        (lambda data: data['runs'][0]['log_records'].pop(), 'not each of'),
```

Replace:

```python
        (lambda read: read['detail'].update(text_only='other'), 'text_only'),
        (lambda read: read['detail'].update(arm='other'), 'arm'),
        (lambda read: read.update(fingerprint='other'), 'fingerprint'),
    ],
    ids=['text_only', 'arm', 'fingerprint'],
)
def test_a_regressor_read_must_name_the_runs_tables_and_its_panel(
```

with:

```python
        (lambda read: read['detail'].update(text_only='other'), 'text_only'),
        (lambda read: read['detail'].update(arm='other'), 'arm'),
        (lambda read: read['detail'].update(dimension=16), 'dimension'),
        (lambda read: read.update(fingerprint='other'), 'fingerprint'),
    ],
    ids=['text_only', 'arm', 'dimension', 'fingerprint'],
)
def test_a_regressor_read_must_name_the_runs_tables_and_its_panel(
```

Replace:

```python
    with pytest.raises(ValueError, match=f"another \\['{key}'\\]"):
        _decide([arm, reference], margins, store)

def test_a_decision_compares_at_least_two_arms(store, reference, margins):
```

with:

```python
    with pytest.raises(ValueError, match=f"another \\['{key}'\\]"):
        _decide([arm, reference], margins, store)

@pytest.mark.parametrize(
    'geometry, distance',
    [('euclidean', 'euclidean'), ('spherical', 'cosine'), ('hyperbolic', 'lorentz')],
)
def test_a_seed_decodes_by_the_distance_of_its_arms_geometry(geometry, distance):
    '''Req 12: each arm decodes by its own distance, and no other.'''

    arm_spec = spec('arm', geometry=geometry)

    check_seed_distance(arm_spec, 3, distance)
    for other in sorted({'euclidean', 'cosine', 'lorentz'} - {distance}):
        message = (
            f"^arm seed 3: the encoder decodes by '{other}', but a {geometry} arm decodes by "
            f"'{distance}' \\(Req 12\\)$"
        )
        with pytest.raises(ValueError, match=message):
            check_seed_distance(arm_spec, 3, other)

def test_a_decision_compares_at_least_two_arms(store, reference, margins):
```

Replace:

```python
            "a monitor read names another \\['fingerprint'\\]",
        ),
        (_repeat_an_epoch, 'the monitor records repeat the epochs \\[1\\]'),
    ],
```

with:

```python
            "a monitor read names another \\['fingerprint'\\]",
        ),
        (
            lambda data: _monitor_read(data)['detail'].update(distance='cosine'),
            "a monitor read names another \\['distance'\\]",
        ),
        (_repeat_an_epoch, 'the monitor records repeat the epochs \\[1\\]'),
    ],
```

Replace:

```python
        'another-seed',
        'another-fingerprint',
        'a-repeated-epoch',
    ],
```

with:

```python
        'another-seed',
        'another-fingerprint',
        'another-distance',
        'a-repeated-epoch',
    ],
```

Replace:

```python
    with pytest.raises(ValueError, match=f'^trained seed 3: {message}'):
        check_arm(arm, store, min_seeds=5)

@pytest.mark.parametrize('epoch', [None, 0, 2, 3], ids=['none', 'earlier', 'a-later-tie', 'later'])
```

with:

```python
    with pytest.raises(ValueError, match=f'^trained seed 3: {message}'):
        check_arm(arm, store, min_seeds=5)

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_a_flat_arms_reads_are_checked_against_its_own_distance(store, tmp_path, geometry):
    '''Req 12: a flat arm's outcome and monitor reads decode by its distance, not the Lorentz.'''

    flat = synthetic_arm(
        store, tmp_path, spec('flat', geometry=geometry), {}, monitor_mrrs=MONITOR_MRRS
    )

    check_arm(flat, store, min_seeds=5)
    for edit in (
        lambda data: _first_read(data)['detail'].update(distance='lorentz'),
        lambda data: _monitor_read(data)['detail'].update(distance='lorentz'),
    ):
        with pytest.raises(ValueError, match="another \\['distance'\\]"):
            check_arm(_edited(flat, edit), store, min_seeds=5)

@pytest.mark.parametrize('epoch', [None, 0, 2, 3], ids=['none', 'earlier', 'a-later-tie', 'later'])
```

In `tests/unit/test_decision_sweep.py`, make these 8 edits, in order.

Replace:

```python
SEEDS = (0, 1, 2, 3, 4)
PURPOSE = 'fixture seed sweep'

class SyntheticEncoder:
```

with:

```python
SEEDS = (0, 1, 2, 3, 4)
PURPOSE = 'fixture seed sweep'

def _spec(name, **overrides):
    '''A synthetic arm's spec: its encoder decodes by cosine, the spherical arm's distance.'''

    return spec(name, **{'geometry': 'spherical', **overrides})

class SyntheticEncoder:
```

Replace:

```python
        training_run = f'{arm_spec.name}-training-{seed}'
        records = monitor_records(
            training_run, seed, TRAINED_MRRS, time=TRAINED_AT, fingerprint=self.outcome
        )
        self.returned[seed] = replace(
```

with:

```python
        training_run = f'{arm_spec.name}-training-{seed}'
        records = monitor_records(
            training_run,
            seed,
            TRAINED_MRRS,
            time=TRAINED_AT,
            fingerprint=self.outcome,
            distance='cosine',
        )
        self.returned[seed] = replace(
```

Replace:

```python
    runner = SyntheticRunner(tmp_path / 'runs' / name, informed, _signal(regressor_rows))
    return run_seed_sweep(
        spec(name, dimension=3, **overrides),
        SEEDS,
        runner,
```

with:

```python
    runner = SyntheticRunner(tmp_path / 'runs' / name, informed, _signal(regressor_rows))
    return run_seed_sweep(
        _spec(name, dimension=3, **overrides),
        SEEDS,
        runner,
```

Replace:

```python
    runner = TrainedRunner(tmp_path / 'runs' / 'trained', False, _signal(regressor_rows), outcome)

    arm = run_seed_sweep(
        spec('trained', dimension=3),
        SEEDS,
        runner,
        text_only_table=text_only,
```

with:

```python
    runner = TrainedRunner(tmp_path / 'runs' / 'trained', False, _signal(regressor_rows), outcome)

    arm = run_seed_sweep(
        _spec('trained', dimension=3),
        SEEDS,
        runner,
        text_only_table=text_only,
```

Replace:

```python
    with pytest.raises(ValueError, match='dimension'):
        run_seed_sweep(
            spec('mislabelled', dimension=16),
            SEEDS,
            SyntheticRunner(tmp_path / 'runs' / 'mislabelled', False, _signal(regressor_rows)),
```

with:

```python
    with pytest.raises(ValueError, match='dimension'):
        run_seed_sweep(
            _spec('mislabelled', dimension=16),
            SEEDS,
            SyntheticRunner(tmp_path / 'runs' / 'mislabelled', False, _signal(regressor_rows)),
```

Replace:

```python
    with pytest.raises(ValueError, match='misread seed 0: the table was exported from .*D9'):
        run_seed_sweep(
            spec('misread', dimension=3),
            SEEDS,
            runner,
```

with:

```python
    with pytest.raises(ValueError, match='misread seed 0: the table was exported from .*D9'):
        run_seed_sweep(
            _spec('misread', dimension=3),
            SEEDS,
            runner,
```

Replace:

```python
    with pytest.raises(ValueError, match='has no export provenance'):
        run_seed_sweep(
            spec('unexported', dimension=3),
            SEEDS,
            runner,
```

with:

```python
    with pytest.raises(ValueError, match='has no export provenance'):
        run_seed_sweep(
            _spec('unexported', dimension=3),
            SEEDS,
            runner,
            text_only_table=text_only,
            store=store,
            purpose=PURPOSE,
            **panels,
        )
    assert log.records() == []

def test_a_seed_decoding_by_another_distance_than_its_geometrys_is_refused_before_any_read(
    tmp_path, regressor_rows, panels, store, text_only, log
):
    '''Req 12: a hyperbolic arm decodes by the Lorentz distance, and this encoder by cosine.'''

    runner = SyntheticRunner(tmp_path / 'runs' / 'misdecoded', False, _signal(regressor_rows))
    message = (
        "misdecoded seed 0: the encoder decodes by 'cosine', but a hyperbolic arm decodes by "
        "'lorentz' \\(Req 12\\)"
    )

    with pytest.raises(ValueError, match=message):
        run_seed_sweep(
            _spec('misdecoded', dimension=3, geometry='hyperbolic'),
            SEEDS,
            runner,
```

Replace:

```python
    with pytest.raises(ValueError, match='a seed repeats'):
        run_seed_sweep(
            spec('uninformed', dimension=3),
            (0, 1, 1),
            SyntheticRunner(tmp_path / 'runs' / 'uninformed', False, _signal(regressor_rows)),
```

with:

```python
    with pytest.raises(ValueError, match='a seed repeats'):
        run_seed_sweep(
            _spec('uninformed', dimension=3),
            (0, 1, 1),
            SyntheticRunner(tmp_path / 'runs' / 'uninformed', False, _signal(regressor_rows)),
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_checkpoint_runner.py tests/unit/test_decision.py tests/unit/test_decision_sweep.py -n auto -q
```
Expected: FAIL, `3 failed, 59 passed, 1 error`:
- the error is `tests/unit/test_decision.py` at collection, `ImportError: cannot import name
  'check_seed_distance' from 'naics_embedder.decision.decide'`;
- the failures are the runner's two distance tests and the sweep's distance refusal, none of which
  raises.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/decision/decide.py`, make these 7 edits, in order.

Replace:

```python
  summaries and window (D9);
- each run's log records are validation reads that name the run, its table and its text-only
  table by the fingerprints the store recorded;
- a trained run's monitor records are its training run's reads of the outcome panel's validation
  split, no epoch twice, and its checkpoint is from the earliest epoch with the highest MRR (spec
  4.4); a run with no training run has neither monitor records nor a checkpoint epoch;
- all arms read the same panels, with the same data on them and the same fit settings, so Δ
  pairs item for item.
```

with:

```python
  summaries and window (D9);
- each run's log records are validation reads that name the run, its table and its text-only
  table by the fingerprints the store recorded; its outcome read decodes by the distance of the
  arm's geometry, and its regressor reads name the arm's dimension (Req 12);
- a trained run's monitor records are its training run's reads of the outcome panel's validation
  split, under the distance of the arm's geometry, no epoch twice, and its checkpoint is from the
  earliest epoch with the highest MRR (spec 4.4); a run with no training run has neither monitor
  records nor a checkpoint epoch;
- all arms read the same panels, with the same data on them and the same fit settings, so Δ
  pairs item for item.
```

Replace:

```python
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import VALIDATION
```

with:

```python
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.decoding import GEOMETRY_DISTANCES
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import VALIDATION
```

Replace:

```python
        )

def check_arm(arm: ArmRecord, store: ArtifactStore, min_seeds: int) -> None:
    '''
```

with:

```python
        )

def check_seed_distance(spec: ArmSpec, seed: int, distance: str) -> None:
    '''
    Require a seed's encoder to decode by the distance of the arm's geometry (Req 12).

    Args:
        spec: The arm.
        seed: The seed, which names the refusal.
        distance: The distance the seed's encoder decodes by.

    Raises:
        ValueError: If it is another distance.
    '''

    expected = GEOMETRY_DISTANCES[spec.geometry]
    if distance != expected:
        raise ValueError(
            f'{spec.name} seed {seed}: the encoder decodes by {distance!r}, but a '
            f'{spec.geometry} arm decodes by {expected!r} (Req 12)'
        )

def check_arm(arm: ArmRecord, store: ArtifactStore, min_seeds: int) -> None:
    '''
```

Replace:

```python
            raise ValueError(f'{name}: a log record names another run')
        if record['panel'] == OUTCOME_PANEL:
            named = {'fingerprint': arm.panels.outcome, 'table': run.table.matrix_fingerprint}
            logged = {'fingerprint': record['fingerprint'], 'table': detail.get('table')}
        else:
            named = {
                'fingerprint': arm.panels.regressor,
                'arm': run.table.matrix_fingerprint,
                'text_only': arm.text_only.table.matrix_fingerprint,
            }
            logged = {key: detail.get(key) for key in ('arm', 'text_only')}
            logged['fingerprint'] = record['fingerprint']
        wrong = sorted(key for key in named if logged[key] != named[key])
```

with:

```python
            raise ValueError(f'{name}: a log record names another run')
        if record['panel'] == OUTCOME_PANEL:
            named = {
                'fingerprint': arm.panels.outcome,
                'table': run.table.matrix_fingerprint,
                'distance': GEOMETRY_DISTANCES[arm.spec.geometry],
            }
            logged = {
                'fingerprint': record['fingerprint'],
                'table': detail.get('table'),
                'distance': detail.get('distance'),
            }
        else:
            named = {
                'fingerprint': arm.panels.regressor,
                'arm': run.table.matrix_fingerprint,
                'text_only': arm.text_only.table.matrix_fingerprint,
                'dimension': arm.spec.dimension,
            }
            logged = {key: detail.get(key) for key in ('arm', 'text_only', 'dimension')}
            logged['fingerprint'] = record['fingerprint']
        wrong = sorted(key for key in named if logged[key] != named[key])
```

Replace:

```python
    '''
    Require a trained run's monitor records to be the reads that selected its checkpoint (spec
    4.4): its training run's reads of the outcome panel's validation split, no epoch twice, with
    the checkpoint from the earliest epoch with the highest MRR. A run with no training run has
    neither monitor records nor a checkpoint epoch.

    That the records cover every epoch through the run's last checkpoint is the runner's check
```

with:

```python
    '''
    Require a trained run's monitor records to be the reads that selected its checkpoint (spec
    4.4): its training run's reads of the outcome panel's validation split, under the distance of
    the arm's geometry (Req 12), no epoch twice, with the checkpoint from the earliest epoch with
    the highest MRR. A run with no training run has neither monitor records nor a checkpoint
    epoch.

    That the records cover every epoch through the run's last checkpoint is the runner's check
```

Replace:

```python
    # A read's table is not named: it is its epoch's code cache, which equals the table exported
    # from that epoch's checkpoint only when both were encoded on the CPU (spec 4.4)
    named = {'fingerprint': arm.panels.outcome, 'training_run': run.training_run, 'seed': run.seed}
    epochs: List[int] = []
    for record in run.monitor_records:
```

with:

```python
    # A read's table is not named: it is its epoch's code cache, which equals the table exported
    # from that epoch's checkpoint only when both were encoded on the CPU (spec 4.4)
    named = {
        'fingerprint': arm.panels.outcome,
        'training_run': run.training_run,
        'seed': run.seed,
        'distance': GEOMETRY_DISTANCES[arm.spec.geometry],
    }
    epochs: List[int] = []
    for record in run.monitor_records:
```

Replace:

```python
            'training_run': detail.get('training_run'),
            'seed': detail.get('seed'),
        }
        wrong = sorted(key for key in named if logged[key] != named[key])
```

with:

```python
            'training_run': detail.get('training_run'),
            'seed': detail.get('seed'),
            'distance': detail.get('distance'),
        }
        wrong = sorted(key for key in named if logged[key] != named[key])
```

In `src/naics_embedder/decision/sweep.py`, make these 3 edits, in order.

Replace:

```python
import polars as pl

from naics_embedder.decision.decide import check_seed_table, check_text_only
from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import DECISION_STATISTIC, PANELS, panel_statistic, seed_scores
```

with:

```python
import polars as pl

from naics_embedder.decision.decide import check_seed_distance, check_seed_table, check_text_only
from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import DECISION_STATISTIC, PANELS, panel_statistic, seed_scores
```

Replace:

```python
    Raises:
        ValueError: If a seed repeats; if the text-only table, or a seed's table by its export
            provenance, was not built from the arm's backbone, revision, descriptions, summaries
            and window (D9); or if a seed's table width is not the arm spec's dimension.
    '''

    if len(set(seeds)) != len(seeds):
```

with:

```python
    Raises:
        ValueError: If a seed repeats; if the text-only table, or a seed's table by its export
            provenance, was not built from the arm's backbone, revision, descriptions, summaries
            and window (D9); if a seed's encoder does not decode by the distance of the arm's
            geometry (Req 12); or if a seed's table width is not the arm spec's dimension.
    '''

    if len(set(seeds)) != len(seeds):
```

Replace:

```python
    for seed in seeds:
        artifacts = runner.run(spec, seed)
        # What the seed read, before anything is stored or any panel is read
        check_seed_table(spec, seed, _seed_table_fields(Path(artifacts.table)))
        run_id = f'{spec.name}/seed-{seed}/{uuid.uuid4().hex}'
        checkpoint = store.put(artifacts.checkpoint)
```

with:

```python
    for seed in seeds:
        artifacts = runner.run(spec, seed)
        # What the seed read and how it decodes, before anything is stored or any panel is read
        check_seed_table(spec, seed, _seed_table_fields(Path(artifacts.table)))
        check_seed_distance(spec, seed, artifacts.distance)
        run_id = f'{spec.name}/seed-{seed}/{uuid.uuid4().hex}'
        checkpoint = store.put(artifacts.checkpoint)
```

In `src/naics_embedder/text_model/checkpoint_runner.py`, make these 4 edits, in order.

Replace:

```python
from naics_embedder.decision.records import ArmSpec
from naics_embedder.decision.sweep import SeedArtifacts
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
```

with:

```python
from naics_embedder.decision.records import ArmSpec
from naics_embedder.decision.sweep import SeedArtifacts
from naics_embedder.panels.decoding import GEOMETRY_DISTANCES
from naics_embedder.panels.window_summaries import summaries_identity
from naics_embedder.supervision.artifacts import ValidatedSupervisionBundle
```

Replace:

```python
        Check one seed without exporting or reading a decision panel (P25, spec 5).

        Records must cover every epoch through ``last.ckpt`` exactly once and name its training
        run. The earliest epoch with the highest MRR must have one unambiguous checkpoint,
        whose identity, contract, settings and saved best score agree exactly.

        Raises:
```

with:

```python
        Check one seed without exporting or reading a decision panel (P25, spec 5).

        Records must cover every epoch through ``last.ckpt`` exactly once, name its training run
        and read under the distance of the arm's geometry (Req 12). The earliest epoch with the
        highest MRR must have one unambiguous checkpoint, whose identity, contract, settings and
        saved best score agree exactly.

        Raises:
```

Replace:

```python
        if not isinstance(training_run, str) or not training_run.strip():
            raise ValueError('last.ckpt names no training run')
        epochs = []
        for record in records:
```

with:

```python
        if not isinstance(training_run, str) or not training_run.strip():
            raise ValueError('last.ckpt names no training run')
        distance = GEOMETRY_DISTANCES[spec.geometry]
        epochs = []
        for record in records:
```

Replace:

```python
            if detail.get('training_run') != training_run:
                raise ValueError('a monitor record names another training run than last.ckpt')
            epochs.append(detail['epoch'])
        repeated = sorted(epoch for epoch, count in Counter(epochs).items() if count > 1)
```

with:

```python
            if detail.get('training_run') != training_run:
                raise ValueError('a monitor record names another training run than last.ckpt')
            if detail.get('distance') != distance:
                raise ValueError(
                    f"a monitor record reads by {detail.get('distance')!r}, but a "
                    f'{spec.geometry} arm decodes by {distance!r} (Req 12)'
                )
            epochs.append(detail['epoch'])
        repeated = sorted(epoch for epoch, count in Counter(epochs).items() if count > 1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_checkpoint_runner.py tests/unit/test_decision.py tests/unit/test_decision_sweep.py -n auto -q
```
Expected: PASS, `137 passed`.

- [ ] **Step 5: Format, lint and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/decision/decide.py src/naics_embedder/decision/sweep.py src/naics_embedder/text_model/checkpoint_runner.py tests/fixtures/decision.py tests/unit/test_checkpoint_runner.py tests/unit/test_decision.py tests/unit/test_decision_sweep.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run pytest -n auto -q`
Expected: `3082 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/decision/decide.py src/naics_embedder/decision/sweep.py src/naics_embedder/text_model/checkpoint_runner.py tests/fixtures/decision.py tests/unit/test_checkpoint_runner.py tests/unit/test_decision.py tests/unit/test_decision_sweep.py
git commit -m "feat(decision): guard each seed's and each read's distance and dimension (Req 12)"
```

### Task 6: A flat arm's radius report

Verification "No inert terms" holds for every arm, but the radius checks are the hyperbolic
arm's (Req 13). A flat arm has no radial term and no live-radius head (P11):
- `term_gradients` adds `radial` only when the step has the term, and the anchors' radius
  gradients only when `head.radial`. A flat arm's report therefore checks the task term, the
  code-code term and both logit scales.
- `tools radius-report` runs `radius_report` only when `head.radial`. Otherwise it records
  `radius` as null.
- The report names the arm's `geometry`. It passes when every term and scale has gradient and,
  under hyperbolic, every radius check passes.

The tests cover a flat arm's term gradients, which have no radius entries, and the CLI's flat
report. The existing CLI test also checks the report's `geometry`.

**Files:**
- Modify: `src/naics_embedder/cli/commands/tools.py`, lines 878-885, 907-932
- Modify: `src/naics_embedder/text_model/radius_report.py`, lines 1-3, 220-228, 237-242, 248-259
- Test: `tests/unit/test_radius_report.py`, lines 278-281, 295-304, 340-344, 352-355, 381-384

**Interfaces:**
- Consumes: Task 3's `StepLosses.radial` (None in a flat arm); Task 1's `head.radial` and
  `head.geometry`; Task 2's `geometry` argument, through the test fixture
  `build_reference_model(manifest, bundle, **overrides)`.
- Produces:
  - `term_gradients(model, batch) -> Dict[str, float]`. A flat arm's keys are exactly `task`,
    `code_code`, `logit_scale_task` and `logit_scale_code`.
  - The `tools radius-report` JSON gains `geometry`, and `radius` is null for a flat arm.
    `passed` is `(radius is None or radius.passed) and not inert_terms`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_radius_report.py`, make these 5 edits, in order.

Replace:

```python
            assert torch.equal(parameter.grad, before[name])

def test_disabled_terms_are_reported_as_inert(
    reference_arm_model, reference_arm_code_rows, reference_arm_steps
```

with:

```python
            assert torch.equal(parameter.grad, before[name])

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_a_flat_arms_terms_and_scales_get_gradient_and_it_reports_no_radius_gradient(
    geometry, tiny_backbone, reference_manifest, reference_bundle, reference_arm_code_rows,
    reference_arm_steps
):
    '''Req 12: the radial term and dL/dr_a are the hyperbolic arm's alone.'''

    model = build_reference_model(reference_manifest, reference_bundle, geometry=geometry).eval()
    model.refresh_code_cache(reference_arm_code_rows)

    result = _module().term_gradients(model, reference_arm_steps[0])

    expected = {'task', 'code_code', 'logit_scale_task', 'logit_scale_code'}
    assert set(result) == expected
    assert all(np.isfinite(result[name]) and result[name] > 0 for name in expected)

def test_disabled_terms_are_reported_as_inert(
    reference_arm_model, reference_arm_code_rows, reference_arm_steps
```

Replace:

```python
    reference_arm_token_config, reference_arm_code_rows
):
    model = build_reference_model(
        reference_manifest,
        reference_bundle,
        fusion=getattr(request, 'param', 'masked_mean'),
        moe_hidden_dim=16
    ).eval()
    with torch.no_grad():
        model.encoder.projection.weight.mul_(3)
```

with:

```python
    reference_arm_token_config, reference_arm_code_rows
):
    # An indirect parameter overrides the arm's constructor arguments: its fusion or geometry
    arguments = {'fusion': 'masked_mean', 'moe_hidden_dim': 16, **getattr(request, 'param', {})}
    model = build_reference_model(reference_manifest, reference_bundle, **arguments).eval()
    with torch.no_grad():
        model.encoder.projection.weight.mul_(3)
```

Replace:

```python
    ]

@pytest.mark.parametrize('cli_arm', ['masked_mean', 'moe'], indirect=True)
def test_cli_runs_the_saved_seeds_first_batch_without_reading_a_panel(
    cli_arm, tmp_path, minilm_tokenizer, monkeypatch
```

with:

```python
    ]

@pytest.mark.parametrize(
    'cli_arm',
    [pytest.param({}, id='masked_mean'),
     pytest.param({'fusion': 'moe'}, id='moe')],
    indirect=True,
)
def test_cli_runs_the_saved_seeds_first_batch_without_reading_a_panel(
    cli_arm, tmp_path, minilm_tokenizer, monkeypatch
```

Replace:

```python
    report = json.loads(output.read_text())
    assert report['passed'] is True
    assert report['seed'] == 7
    assert report['epoch'] == report['step'] == 0
```

with:

```python
    report = json.loads(output.read_text())
    assert report['passed'] is True
    assert report['geometry'] == 'hyperbolic'
    assert report['seed'] == 7
    assert report['epoch'] == report['step'] == 0
```

Replace:

```python
    assert not list(tmp_path.rglob('*selection_log*'))

def test_cli_writes_a_failed_capped_report_and_exits_one(cli_arm, tmp_path):
    table = pl.read_parquet(cli_arm.table)
```

with:

```python
    assert not list(tmp_path.rglob('*selection_log*'))

@pytest.mark.parametrize(
    'cli_arm',
    [
        pytest.param({'geometry': 'euclidean'}, id='euclidean'),
        pytest.param({'geometry': 'spherical'}, id='spherical'),
    ],
    indirect=True,
)
def test_cli_checks_a_flat_arms_terms_and_scales_alone(cli_arm, tmp_path, monkeypatch):
    '''Req 12: a flat arm has no radial term or live-radius head, so no radius check runs.'''

    monkeypatch.setattr(
        tools.OutcomePanel, 'score', lambda *args, **kwargs: pytest.fail('panel read')
    )
    monkeypatch.setattr(tools, 'radius_report', lambda *args, **kwargs: pytest.fail('radius'))
    output = tmp_path / 'report.json'

    result = CliRunner().invoke(tools.app, _cli_args(cli_arm, output))

    assert result.exit_code == 0, result.output + str(result.exception)
    report = json.loads(output.read_text())
    assert report['geometry'] == cli_arm.model.encoder.head.geometry
    assert report['radius'] is None
    assert (report['passed'], report['inert_terms']) == (True, [])
    expected = {'task', 'code_code', 'logit_scale_task', 'logit_scale_code'}
    assert set(report['term_gradients']) == expected

    original = tools.term_gradients

    def inert_code_code(*args):
        gradients = original(*args)
        gradients['code_code'] = 0.0
        return gradients

    monkeypatch.setattr(tools, 'term_gradients', inert_code_code)
    failed_output = tmp_path / 'inert.json'
    failed = CliRunner().invoke(tools.app, _cli_args(cli_arm, failed_output))
    assert failed.exit_code == 1, failed.output
    failed_report = json.loads(failed_output.read_text())
    assert (failed_report['passed'], failed_report['inert_terms']) == (False, ['code_code'])
    assert not list(tmp_path.rglob('*selection_log*'))

def test_cli_writes_a_failed_capped_report_and_exits_one(cli_arm, tmp_path):
    table = pl.read_parquet(cli_arm.table)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_radius_report.py -n auto -q`
Expected: FAIL, `6 failed, 21 passed`:
- the CLI report has no geometry (`KeyError: 'geometry'`), in both fusions;
- a flat arm's `term_gradients` multiplies its None radial term (`TypeError: unsupported operand
  type(s) for *: 'float' and 'NoneType'`);
- the flat CLI tests fail, under `euclidean` and `spherical`.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/cli/commands/tools.py`, make these 2 edits, in order.

Replace:

```python
    Check radius variation, geometry and loss gradients for one selected checkpoint.

    Gradients use epoch 0, step 0 of the saved seed and saved query chunk size. The checkpoint
    and its table must share an export provenance. The report is written even when a measured
    criterion fails; a failure exits 1. No evaluation split is scored.
    '''

    configure_logging('tools_radius_report.log')
```

with:

```python
    Check radius variation, geometry and loss gradients for one selected checkpoint.

    Gradients use epoch 0, step 0 of the saved seed and saved query chunk size. The checkpoint
    and its table must share an export provenance. The radius checks are the hyperbolic arm's: a
    flat arm (Req 12) has no radial term or live-radius head, so its report checks its terms and
    scales alone and records ``radius`` as null. The report names the arm's geometry and is
    written even when a measured criterion fails; a failure exits 1. No evaluation split is
    scored.
    '''

    configure_logging('tools_radius_report.log')
```

Replace:

```python
        model.refresh_code_cache(data.code_rows)
        gradients = term_gradients(model, batch)
        anchors = [
            gradients.pop(f'{ANCHOR_GRADIENT_PREFIX}{row}')
            for row in range(len(batch['codes']['ids']))
        ]
        radius = radius_report(pl.read_parquet(table), anchor_radius_gradient=np.array(anchors))
        inert = [
            name for name, value in gradients.items() if not (np.isfinite(value) and value > 0)
        ]
        report = {
            'checkpoint': str(Path(checkpoint).resolve()),
            'table': str(Path(table).resolve()),
            'seed': int(model.hparams.seed),
            'epoch': 0,
            'step': 0,
            'anchor_ids': batch['codes']['ids'].tolist(),
            'radius': asdict(radius),
            'term_gradients': {
                name: value if np.isfinite(value) else None
                for name, value in gradients.items()
            },
            'inert_terms': inert,
            'passed': radius.passed and not inert
        }
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
```

with:

```python
        model.refresh_code_cache(data.code_rows)
        gradients = term_gradients(model, batch)
        head = model.encoder.head
        # Verification "Radius" checks the radial arm, the hyperbolic (Req 12, 13)
        radius = None
        if head.radial:
            anchors = [
                gradients.pop(f'{ANCHOR_GRADIENT_PREFIX}{row}')
                for row in range(len(batch['codes']['ids']))
            ]
            radius = radius_report(pl.read_parquet(table), anchor_radius_gradient=np.array(anchors))
        inert = [
            name for name, value in gradients.items() if not (np.isfinite(value) and value > 0)
        ]
        report = {
            'checkpoint': str(Path(checkpoint).resolve()),
            'table': str(Path(table).resolve()),
            'geometry': head.geometry,
            'seed': int(model.hparams.seed),
            'epoch': 0,
            'step': 0,
            'anchor_ids': batch['codes']['ids'].tolist(),
            'radius': None if radius is None else asdict(radius),
            'term_gradients': {
                name: value if np.isfinite(value) else None
                for name, value in gradients.items()
            },
            'inert_terms': inert,
            'passed': (radius is None or radius.passed) and not inert
        }
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
```

In `src/naics_embedder/text_model/radius_report.py`, make these 4 edits, in order.

Replace:

```python
'''Radius and gradient checks for a selected hyperbolic arm (spec 4.2, Verification Radius).'''

from contextlib import contextmanager
```

with:

```python
'''
Radius and gradient checks for a selected arm (spec 4.2, Verification Radius, No inert terms).

The radius checks are the hyperbolic arm's, the one arm with the radial term and the live-radius
head (Req 12, 13). Every arm's terms and scales are checked for gradient.
'''

from contextlib import contextmanager
```

Replace:

```python
def term_gradients(model: torch.nn.Module, batch: Mapping[str, Any]) -> Dict[str, float]:
    '''
    Measure No inert terms and signed dL/dr_a on one two-stream batch (spec 6, P26).

    Each term's norm is taken over the trainable encoder parameters, with its contribution's
    weight. The two scale gradients come from the total, so a zero-weight code-code term also
    leaves its scale inert. ``anchor_radius/<row>`` names each signed total-loss radius gradient.
    Existing parameter gradients and training flags are preserved. MoE utilization logging is
    suppressed for this computation only, so no Trainer, log warning or histogram is produced.
```

with:

```python
def term_gradients(model: torch.nn.Module, batch: Mapping[str, Any]) -> Dict[str, float]:
    '''
    Measure No inert terms and, in the hyperbolic arm, signed dL/dr_a on one two-stream batch
    (spec 6, P26).

    Each term's norm is taken over the trainable encoder parameters, with its contribution's
    weight. The two scale gradients come from the total, so a zero-weight code-code term also
    leaves its scale inert. The radial term and ``anchor_radius/<row>``, each signed total-loss
    radius gradient, are the hyperbolic arm's alone (Req 12): a flat arm reports neither.
    Existing parameter gradients and training flags are preserved. MoE utilization logging is
    suppressed for this computation only, so no Trainer, log warning or histogram is produced.
```

Replace:

```python
            'task': losses.task,
            'code_code': model.hparams.code_code_weight * losses.code_code,
            'radial': model.hparams.radial_weight * losses.radial
        }
        if losses.load_balancing is not None:
            terms['load_balancing'] = model.hparams.load_balancing_coef * losses.load_balancing
```

with:

```python
            'task': losses.task,
            'code_code': model.hparams.code_code_weight * losses.code_code,
        }
        if losses.radial is not None:
            terms['radial'] = model.hparams.radial_weight * losses.radial
        if losses.load_balancing is not None:
            terms['load_balancing'] = model.hparams.load_balancing_coef * losses.load_balancing
```

Replace:

```python
        ):
            result[name] = _gradient_norm(losses.total, (scale.log_scale, ))
        gradient = torch.autograd.grad(losses.total, losses.anchor_radius, allow_unused=True)[0]
        if gradient is None:
            gradient = torch.zeros_like(losses.anchor_radius)
        result.update(
            {
                f'{ANCHOR_GRADIENT_PREFIX}{row}': float(value)
                for row, value in enumerate(gradient.detach().cpu())
            }
        )
    return result
```

with:

```python
        ):
            result[name] = _gradient_norm(losses.total, (scale.log_scale, ))
        if model.encoder.head.radial:
            gradient = torch.autograd.grad(losses.total, losses.anchor_radius, allow_unused=True)[0]
            if gradient is None:
                gradient = torch.zeros_like(losses.anchor_radius)
            result.update(
                {
                    f'{ANCHOR_GRADIENT_PREFIX}{row}': float(value)
                    for row, value in enumerate(gradient.detach().cpu())
                }
            )
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_radius_report.py -n auto -q`
Expected: PASS, `27 passed`.

- [ ] **Step 5: Format, lint and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/cli/commands/tools.py src/naics_embedder/text_model/radius_report.py tests/unit/test_radius_report.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run pytest -n auto -q`
Expected: `3086 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/cli/commands/tools.py src/naics_embedder/text_model/radius_report.py tests/unit/test_radius_report.py
git commit -m "feat(tools): a flat arm's radius report checks its terms and scales alone"
```

### Task 7: The flat arms through the Trainer

Tasks 1–6 test each piece. This task trains each flat arm through the real Trainer on the
reference fixture bundle, at `32-true` and at `bf16-mixed`, for two epochs. It checks:
- no step computes a radial term, and the epoch summary records no `loss/radial`;
- each epoch's monitor read decodes by the arm's distance;
- each read's MRR equals, exactly and on the CPU, the read of the table exported from that
  epoch's checkpoint, as the hyperbolic arm's test already checks (spec 4.4).

The test's `_ModuleHooks` helper records each step's terms. It must skip a term the arm lacks,
and that one-line fix is this task's implementation. The test's edits are split so the new test
fails first: make Step 1's three edits, then Step 3's two, in that order.

**Files:**
- Test: `tests/integration/test_reference_training.py`, lines 33-38, 42-45, 175-184, 193-197,
  512-515

**Interfaces:**
- Consumes: Tasks 1–6, through `reference_runs` (`tests/integration/test_reference_training.py`),
  `export_code_table`, `ArmEncoder.from_files`, `read_outcome_validation` and
  `read_epoch_summary`.
- Produces: nothing later tasks use.

- [ ] **Step 1: Write the failing test**

In `tests/integration/test_reference_training.py`, make these 3 edits, in order.

Replace:

```python
from pytorch_lightning.loggers import Logger

from naics_embedder.cli.commands import training
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import matrix_fingerprint
```

with:

```python
from pytorch_lightning.loggers import Logger

from naics_embedder.cli.commands import training
from naics_embedder.panels.decoding import GEOMETRY_DISTANCES
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import matrix_fingerprint
```

Replace:

```python
from naics_embedder.text_model.dataloader import datamodule as two_stream
from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule
from naics_embedder.text_model.export import code_token_config, export_code_table
from naics_embedder.text_model.mixins import OUTCOME_MRR
```

with:

```python
from naics_embedder.text_model.dataloader import datamodule as two_stream
from naics_embedder.text_model.dataloader.datamodule import NAICSDataModule
from naics_embedder.text_model.epoch_summary import EPOCH_SUMMARY, read_epoch_summary
from naics_embedder.text_model.export import code_token_config, export_code_table
from naics_embedder.text_model.mixins import OUTCOME_MRR
```

Replace:

```python
        arm = ArmEncoder.from_files(checkpoint, table, reference_bundle, token_config)
        exported = read_outcome_validation(arm, panel, 'the monitor read of the exported epoch')
        assert exported.summary['mrr'] == record['mrr'], epoch
        assert arm.table_fingerprint == record['read']['detail']['table'], epoch
```

with:

```python
        arm = ArmEncoder.from_files(checkpoint, table, reference_bundle, token_config)
        exported = read_outcome_validation(arm, panel, 'the monitor read of the exported epoch')
        assert exported.summary['mrr'] == record['mrr'], epoch
        assert arm.table_fingerprint == record['read']['detail']['table'], epoch

@pytest.mark.parametrize('precision', ['32-true', 'bf16-mixed'])
@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_a_flat_arm_trains_without_the_radial_term_and_reads_by_its_own_distance(
    reference_runs, reference_bundle, tmp_path, geometry, precision
):
    '''
    Req 12 through the Trainer: a flat arm trains without the radial term, records no
    ``loss/radial``, and each epoch's monitor read decodes by its own distance. As in the
    hyperbolic arm, that read's MRR is the read of the table exported from the epoch's checkpoint,
    exactly, on the CPU (spec 4.4).
    '''

    cfg = reference_runs.config(
        'run', {
            'training.trainer.max_epochs': 2,
            'model.geometry': geometry
        }
    )
    monitor = reference_runs.monitor(cfg, tmp_path / 'logs' / 'monitor_log.jsonl')
    every_epoch = tmp_path / 'every_epoch'

    run = reference_runs.build(
        cfg, monitor, callbacks=[_every_epoch(every_epoch)], precision=precision
    ).fit()

    assert run.model.encoder.head.geometry == geometry
    assert len(run.hooks.terms) == 2 * STEPS
    assert all('radial' not in terms for terms in run.hooks.terms)
    summary = read_epoch_summary(run.checkpoint_dir / EPOCH_SUMMARY)
    assert [row['epoch'] for row in summary] == [0, 1]
    assert all('loss/task' in row and 'loss/radial' not in row for row in summary)
    records = read_monitor_records(run.checkpoint_dir / MONITOR_RECORDS)
    distance = GEOMETRY_DISTANCES[geometry]
    assert [record['read']['detail']['distance'] for record in records] == [distance] * 2
    panel = OutcomePanel.from_bundle(reference_bundle, tmp_path / 'logs' / 'arm_log.jsonl')
    token_config = run.datamodule.token_config
    for epoch, record in enumerate(records):
        checkpoint = every_epoch / f'epoch={epoch:03d}.ckpt'
        table = export_code_table(
            checkpoint,
            reference_bundle,
            token_config,
            tmp_path / 'tables' / f'epoch={epoch:03d}.parquet',
        )
        arm = ArmEncoder.from_files(checkpoint, table, reference_bundle, token_config)
        exported = read_outcome_validation(arm, panel, 'the monitor read of the exported epoch')
        assert arm.distance == distance
        assert exported.summary['mrr'] == record['mrr'], epoch
        assert arm.table_fingerprint == record['read']['detail']['table'], epoch
```


- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/integration/test_reference_training.py -n auto -q`
Expected: FAIL, `4 failed, 18 passed`. Each of the four new cases fails with `AttributeError:
'NoneType' object has no attribute 'item'`, in `_ModuleHooks.recorded_losses`.

- [ ] **Step 3: Record only the terms the arm has**

In `tests/integration/test_reference_training.py`, make these 2 edits, in order.

Replace:

```python
        )

class _ModuleHooks:
    '''
    The module's hooks as they ran: each step's terms, each epoch whose end ran with the logit
    scales at that end, and each code-cache refresh with the hook and epoch it ran in.
    '''

    def __init__(self, model: NAICSContrastiveModel):
        self.terms: List[Dict[str, Any]] = []
```

with:

```python
        )

class _ModuleHooks:
    '''
    The module's hooks as they ran: each step's terms (a flat arm's steps have no radial term),
    each epoch whose end ran with the logit scales at that end, and each code-cache refresh with
    the hook and epoch it ran in.
    '''

    def __init__(self, model: NAICSContrastiveModel):
        self.terms: List[Dict[str, Any]] = []
```

Replace:

```python
        def recorded_losses(batch):
            losses = compute_losses(batch)
            terms = {term: getattr(losses, term).item() for term in TERMS}
            self.terms.append({'epoch': model.current_epoch, **terms})
            return losses
```

with:

```python
        def recorded_losses(batch):
            losses = compute_losses(batch)
            terms = {
                term: getattr(losses, term).item()
                for term in TERMS if getattr(losses, term) is not None
            }
            self.terms.append({'epoch': model.current_epoch, **terms})
            return losses
```


- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/integration/test_reference_training.py -n auto -q`
Expected: PASS, `22 passed`.

- [ ] **Step 5: Format, lint and run the full suite**

Run: `./scripts/format_code.sh tests/integration/test_reference_training.py`
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run pytest -n auto -q`
Expected: `3090 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_reference_training.py
git commit -m "test(integration): train the flat arms through the Trainer (Req 12)"
```

### Task 8: Documentation

The guides, the API pages and the two assistant-facing files describe the three arms:
- `docs/text_training.md` gains a "Geometry Arms" section, with one row per arm: its point, radius,
  training distance, export, read map and decoding distance. Its head paragraph, its radial-term
  and total formulas, its export and its radius report say what changes off the hyperbolic arm.
- `docs/usage.md` gives `export-table`, `tools sweep` and `radius-report` per geometry, with the
  monitor's distance check.
- `docs/overview.md`, `docs/quickstart.md` and `README.md` name the three arms and link the new
  section.
- `docs/api/encoder.md` gains a "Geometry Heads" section that renders `text_model.heads`. The
  export and radius-report API intros name the geometry.
- `CLAUDE.md` gives the new counts (118 source files, 100 unit files, 3,092 tests), `heads.py` in
  the tree, the architecture and export notes and the `test_heads.py` seam. `tests/README.md`
  gives the same counts and a "Geometry arms" row.

Source files are not touched.

**Files:**
- Modify: `CLAUDE.md`, lines 7-17, 22-25, 29-35, 54-57, 72-78, 275-279, 448-456
- Modify: `README.md`, lines 31-39, 103-108
- Modify: `docs/api/encoder.md`, lines 1-9, 11-17
- Modify: `docs/api/export.md`, lines 1-6
- Modify: `docs/api/radius_report.md`, lines 1-6
- Modify: `docs/overview.md`, lines 44-48
- Modify: `docs/quickstart.md`, lines 81-87
- Modify: `docs/text_training.md`, lines 31-39, 71-82, 188-196, 203-207
- Modify: `docs/usage.md`, lines 234-247, 276-285, 314-318
- Modify: `tests/README.md`, lines 3-7, 15-19, 30-33

**Interfaces:**
- Consumes: the names Tasks 1–7 shipped: `text_model/heads.py`'s heads and functions,
  `model.geometry`, `COORDINATES`, the report's `geometry` and null `radius`, and the guards'
  messages.
- Produces: the `#geometry-arms` anchor in `docs/text_training.md`, which `docs/overview.md`,
  `docs/usage.md` and `README.md` link.

- [ ] **Step 1: Edit the documentation**

In `CLAUDE.md`, make these 7 edits, in order.

Replace:

```markdown
head. The text objective jointly learns task retrieval, code geometry and radial structure.
An optional HGCN stage refines the parent–child graph with its own objective and curriculum.

The project uses Python 3.10+, uv, PyTorch/Lightning, Transformers/PEFT, Polars/PyArrow, Pydantic,
Typer/Rich, pytest, Ruff/YAPF and MkDocs. `.python-version` pins the local locked environment to
Python 3.12. The source tree has **117 Python files**; tests have **99 unit** files and two
integration files. File counts exclude generated and ignored artifacts.

## Architecture Summary

1. **Shared text encoding** (`text_model/fields.py`, `shared_encoder.py`): one LoRA backbone reads
```

with:

```markdown
head. The text objective jointly learns task retrieval, code geometry and radial structure.
An optional HGCN stage refines the parent–child graph with its own objective and curriculum.

The project uses Python 3.10+, uv, PyTorch/Lightning, Transformers/PEFT, Polars/PyArrow, Pydantic,
Typer/Rich, pytest, Ruff/YAPF and MkDocs. `.python-version` pins the local locked environment to
Python 3.12. The source tree has **118 Python files**; tests have **100 unit** files and two
integration files. File counts exclude generated and ignored artifacts.

## Architecture Summary

1. **Shared text encoding** (`text_model/fields.py`, `shared_encoder.py`): one LoRA backbone reads
```

Replace:

```markdown
   `r = R * tanh(norm(v) / R)` with default R = 8, and the projection's direction, form the
   unit-curvature Lorentz point. Task, code-code and radial terms all retain radius gradients.
4. **Graph refinement** (`graph_model/hgcn.py`): HGCN uses the graph, triplet/radial objectives
   and its four-phase graph curriculum. Its curvature utilities remain separate from text.
```

with:

```markdown
   `r = R * tanh(norm(v) / R)` with default R = 8, and the projection's direction, form the
   unit-curvature Lorentz point. Task, code-code and radial terms all retain radius gradients.
   `model.geometry` (Req 12) can instead select a Euclidean or spherical head
   (`text_model/heads.py`). These share the encoder and the task and code-code terms, decode by
   their own distance and have no radial term. Hyperbolic is the default.
4. **Graph refinement** (`graph_model/hgcn.py`): HGCN uses the graph, triplet/radial objectives
   and its four-phase graph curriculum. Its curvature utilities remain separate from text.
```

Replace:

```markdown
The HGCN stage retains its graph-specific samplers and curriculum.

`tools export-table` writes each code's bounded tangent coordinates at the origin, `e0` through
`e{d-1}`, in Req 2's form. Panel reads reconstruct Lorentz points on the CPU in float64.
The selected text checkpoint is the earliest epoch with the highest outcome validation MRR.

## Directory Structure
```

with:

```markdown
The HGCN stage retains its graph-specific samplers and curriculum.

`tools export-table` writes each code's bounded tangent coordinates at the origin, `e0` through
`e{d-1}`, in Req 2's form; a flat arm writes v (Euclidean) or its direction (spherical), and
the provenance names the geometry. Panel reads reconstruct Lorentz points on the CPU in float64,
or read a flat arm's coordinates as they are.
The selected text checkpoint is the earliest epoch with the highest outcome validation MRR.

## Directory Structure
```

Replace:

```markdown
│   ├── fusion.py              # masked mean, attention, MoE
│   ├── hyperbolic.py           # live-radius head, stable polar distance, Lorentz ops
│   ├── loss.py                # task_loss, code_code_loss, radial_loss, LogitScale
│   ├── naics_model.py         # two-stream training, cache and monitor orchestration
```

with:

```markdown
│   ├── fusion.py              # masked mean, attention, MoE
│   ├── hyperbolic.py           # live-radius head, stable polar distance, Lorentz ops
│   ├── heads.py               # Euclidean and spherical heads, flat distances, build_head
│   ├── loss.py                # task_loss, code_code_loss, radial_loss, LogitScale
│   ├── naics_model.py         # two-stream training, cache and monitor orchestration
```

Replace:

```markdown
└── utils/                     # config, training, input-window and geometry utilities

tests/
├── unit/                      # 99 unit test files
├── integration/               # test_reference_training.py, test_remote_workflow.py
├── fixtures/                  # tiny models, bundles, panels and runs
└── conftest.py
```

with:

```markdown
└── utils/                     # config, training, input-window and geometry utilities

tests/
├── unit/                      # 100 unit test files
├── integration/               # test_reference_training.py, test_remote_workflow.py
├── fixtures/                  # tiny models, bundles, panels and runs
└── conftest.py
```

Replace:

````markdown
```

The current suite collects **3,001 tests**. Actual skip counts depend on local data and hardware
capabilities, including MPS. Tests use fixture data and tiny models. Collection counts are not
coverage percentages. Do not read real sealed splits or run a real campaign to verify
````

with:

````markdown
```

The current suite collects **3,092 tests**. Actual skip counts depend on local data and hardware
capabilities, including MPS. Tests use fixture data and tiny models. Collection counts are not
coverage percentages. Do not read real sealed splits or run a real campaign to verify
````

Replace:

```markdown
## Testing and Validation

There are 99 unit files and two integration files. Important current seams include:

- `test_supervision_queries.py` and `test_supervision_code_targets.py`: query/target identities
  and unary masks. `test_loss.py` and `test_hyperbolic.py`: three terms, live-radius head and
  stable polar distance.
- `test_datamodule.py`, `test_monitor.py`, `test_selection_log_guard.py`: two-stream epochs,
  candidate cache, monitor reads and fail-closed selection-log guards.
```

with:

```markdown
## Testing and Validation

There are 100 unit files and two integration files. Important current seams include:

- `test_supervision_queries.py` and `test_supervision_code_targets.py`: query/target identities
  and unary masks. `test_loss.py` and `test_hyperbolic.py`: three terms, live-radius head and
  stable polar distance. `test_heads.py`: the geometry arms' heads, distances and read maps.
- `test_datamodule.py`, `test_monitor.py`, `test_selection_log_guard.py`: two-stream epochs,
  candidate cache, monitor reads and fail-closed selection-log guards.
```

In `README.md`, make these 2 edits, in order.

Replace:

```markdown
## Live Radius and the Objective

For projection v with norm a and direction u, the parameter-free head computes
`r = R * tanh(a / R)` with `R = model.radius_bound` (default 8), then maps r u to the
unit-curvature Lorentz hyperboloid. Zero v maps to the origin. Text curvature is fixed at 1,
with no curvature setting. Training distances use the stable polar form in float32; panel
reads reconstruct Lorentz points and compute distances on the CPU in float64.

`task_loss` decodes each training query to its named code targets. Its candidates are every
```

with:

```markdown
## Live Radius and the Objective

For projection v with norm a and direction u, the default parameter-free head computes
`r = R * tanh(a / R)` with `R = model.radius_bound` (default 8), then maps r u to the
unit-curvature Lorentz hyperboloid. Zero v maps to the origin. Text curvature is fixed at 1,
with no curvature setting. Training distances use the stable polar form in float32; panel
reads reconstruct Lorentz points and compute distances on the CPU in float64.
`model.geometry` also offers Euclidean and spherical heads, which share the encoder and the
task and code-to-code terms but have no radial term
([geometry arms](docs/text_training.md#geometry-arms)).

`task_loss` decodes each training query to its named code targets. Its candidates are every
```

Replace:

````markdown
```

The table has `code`, `index`, `level`, and bounded tangent coordinates `e0` through `e{d-1}`
in codebook order. Provenance binds it to the checkpoint and preprocessing identities.
The radius report checks live gradients, per-level spread, sector radii, manifold validity,
distance precision and nonzero gradients for the three terms and both scales. A failed
````

with:

````markdown
```

The table has `code`, `index`, `level`, and the arm's coordinates `e0` through `e{d-1}` (bounded
tangents in the hyperbolic arm) in codebook order. Provenance binds it to the checkpoint and
preprocessing identities.
The radius report checks live gradients, per-level spread, sector radii, manifold validity,
distance precision and nonzero gradients for the three terms and both scales. A failed
````

In `docs/api/encoder.md`, make these 2 edits, in order.

Replace:

```markdown
# Shared Encoder API

One LoRA-adapted backbone reads marked fields and queries. Fusion combines the present code
channels, then one affine projection produces a direction and live radius. The text head uses
`r = R * tanh(norm(v) / R)` at unit curvature; it has no curvature parameter.

## Fields and Markers

::: naics_embedder.text_model.fields
```

with:

```markdown
# Shared Encoder API

One LoRA-adapted backbone reads marked fields and queries. Fusion combines the present code
channels, then one affine projection feeds the arm's geometry head. The hyperbolic head uses
`r = R * tanh(norm(v) / R)` at unit curvature; it has no curvature parameter. The Euclidean and
spherical heads take v and its direction as they are (Req 12).

## Fields and Markers

::: naics_embedder.text_model.fields
```

Replace:

```markdown
## Fusion

::: naics_embedder.text_model.fusion

## Encoder

::: naics_embedder.text_model.shared_encoder
```

with:

```markdown
## Fusion

::: naics_embedder.text_model.fusion

## Geometry Heads

::: naics_embedder.text_model.heads

## Encoder

::: naics_embedder.text_model.shared_encoder
```

In `docs/api/export.md`, make one edit.

Replace:

```markdown
# Export and Arm Encoder API

Export writes each code's bounded tangent coordinates in Req 2's form. The arm encoder reads
queries through the checkpoint and codes from its matching table. Checkpoint objective
`req11-v1`, bundle and preprocessing identities are checked before model loading or panel reads.
Unit text curvature is part of the objective, with no curvature setting or legacy migration.
```

with:

```markdown
# Export and Arm Encoder API

Export writes each code's coordinates in Req 2's form of its geometry arm: bounded tangents in
the hyperbolic arm. The arm encoder reads queries through the checkpoint and codes from its
matching table, both through the head's read map. Checkpoint objective
`req11-v1`, bundle and preprocessing identities are checked before model loading or panel reads.
Unit text curvature is part of the objective, with no curvature setting or legacy migration.
```

In `docs/api/radius_report.md`, make one edit.

Replace:

```markdown
# Radius Verification API

CPU verification measures live gradients, radius spread, sector radii, manifold residual and
chunked distance precision. The report does not open or read an evaluation panel.

::: naics_embedder.text_model.radius_report
```

with:

```markdown
# Radius Verification API

CPU verification measures live gradients, radius spread, sector radii, manifold residual and
chunked distance precision for the hyperbolic arm, and term and scale gradients for every arm.
The report does not open or read an evaluation panel.

::: naics_embedder.text_model.radius_report
```

In `docs/overview.md`, make one edit.

Replace:

```markdown
## Hyperbolic Geometry and Live Radius

For projection v, let a be its norm and u its direction. The head computes
`r = R * tanh(a / R)`, with `R = model.radius_bound` (default 8), then maps tangent r u to
`(cosh(r), sinh(r) u)`. The zero vector maps to the origin. There is no learned or configurable
```

with:

```markdown
## Hyperbolic Geometry and Live Radius

`model.geometry` selects the head: `hyperbolic` by default, or the flat `euclidean` and
`spherical` arms that Req 12 compares with it (see
[geometry arms](text_training.md#geometry-arms)). The rest of this section describes the
hyperbolic head. For projection v, let a be its norm and u its direction. The head computes
`r = R * tanh(a / R)`, with `R = model.radius_bound` (default 8), then maps tangent r u to
`(cosh(r), sinh(r) u)`. The zero vector maps to the origin. There is no learned or configurable
```

In `docs/quickstart.md`, make one edit.

Replace:

````markdown
```

Export writes bounded tangent coordinates and their provenance. The radius report checks live
radius gradients, per-level spread, sector radii, manifold validity, distance precision and the
three terms' gradients. A failed criterion produces a report and exits 1.

## Compare Configurations
````

with:

````markdown
```

Export writes the arm's coordinates, bounded tangents for the default hyperbolic arm, and their
provenance. The radius report checks live radius gradients, per-level spread, sector radii,
manifold validity, distance precision and the three terms' gradients; for a flat arm, only the
terms' and scales' gradients. A failed criterion produces a report and exits 1.

## Compare Configurations
````

In `docs/text_training.md`, make these 4 edits, in order.

Replace:

```markdown
It adds no mining, router-guided sampling or phase transitions.

The parameter-free head splits projection v into radius and direction. With a = norm(v),
`r = R * tanh(a / R)`, where `R = model.radius_bound`; the resulting Lorentz point is
`(cosh(r), sinh(r) u)`. Zero v maps to the origin. Text curvature is fixed at 1, without a
curvature setting. The radius remains live in every term; it is not normalized away or capped
at a fixed norm of 2.

## The Three Terms
```

with:

```markdown
It adds no mining, router-guided sampling or phase transitions.

A parameter-free head follows the projection: the arm's geometry head (`model.geometry`, see
[Geometry Arms](#geometry-arms)). The default, the hyperbolic head, splits projection v into
radius and direction. With a = norm(v), `r = R * tanh(a / R)`, where `R = model.radius_bound`;
the resulting Lorentz point is `(cosh(r), sinh(r) u)`. Zero v maps to the origin. Text curvature
is fixed at 1, without a curvature setting. The radius remains live in every term; it is not
normalized away or capped at a fixed norm of 2.

## Geometry Arms

`model.geometry` names one of Req 12's three arms: `hyperbolic` (the default and the Stage 7
reference), `euclidean` or `spherical`. Every arm shares the backbone, fusion, projection, task
term and code-to-code term. Only the head, its distances and its export form differ, and the
radial term exists only in the hyperbolic arm.

| Arm | Point | Training distance | Decoding distance | Export |
| --- | --- | --- | --- | --- |
| `hyperbolic` | `(cosh(r), sinh(r) u)` | stable polar form | `lorentz` | bounded tangent `r u` |
| `euclidean` | v | `norm(v_a - v_b)`, in polar form | `euclidean` | v |
| `spherical` | `u = v / norm(v)` | chord form `norm(u_a - u_b)**2 / 2` | `cosine` | u |

The flat heads have no bound, so `model.radius_bound`, `loss.radial_weight` and
`loss.radial_step` are read under `hyperbolic` only. The spherical arm exports u, not v: the
cosine distance gives v's norm no training signal. Each read maps exported coordinates to
points with its head's own map, the exponential map at the origin in the hyperbolic arm and
the identity in the flat arms, and decodes by its arm's distance.

The geometry is part of the checkpoint's encoder record, not of the 21 run settings. Exact
resume, campaign preflight and every load therefore refuse a checkpoint of another arm. A
checkpoint saved before Stage 8 names no geometry and reads as hyperbolic. The HGCN feeder
refines Lorentz points, so it refuses a flat arm, and `train` asks its HGCN question only of a
hyperbolic run.

## The Three Terms
```

Replace:

````markdown
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
````

with:

````markdown
`loss.radial_step * (level - 1)`. Default step 1 gives target radii 1, 2, 3, 4 and 5 at levels
2 through 6. This is a soft target, not a fixed radius: each anchor's derivative through radius
remains live, and codes at the same level can have different radii. The term exists only in
the hyperbolic arm; a flat arm's `StepLosses.radial` is None.

### Total and Defaults

```text
loss = task_loss
     + code_code_weight * code_code_loss
     + radial_weight * radial_loss                    (only under geometry=hyperbolic)
     + moe.load_balancing_coef * load_balancing_loss  (only under fusion=moe)
```
````

Replace:

````markdown
## Export, Diagnostics and Radius Verification

Use the earliest highest-MRR epoch from the monitor records. Export encodes every code through
that checkpoint in eval mode and writes bounded tangent coordinates `e0` through `e{d-1}` plus
checkpoint/table provenance. Panel reads reconstruct unit-curvature Lorentz points on the CPU
in float64. A table exported from another checkpoint or preprocessing pin is refused.

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/reference/epoch=001.ckpt   --output data/reference/table.parquet   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
````

with:

````markdown
## Export, Diagnostics and Radius Verification

Use the earliest highest-MRR epoch from the monitor records. Export encodes every code through
that checkpoint in eval mode and writes its arm's coordinates `e0` through `e{d-1}` plus
checkpoint/table provenance, which names the geometry. The coordinates are bounded tangents in
the hyperbolic arm, v in the Euclidean arm and u in the spherical arm. Panel reads map them to
points on the CPU in float64, unit-curvature Lorentz points in the hyperbolic arm. A table
exported from another checkpoint or preprocessing pin is refused.

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/reference/epoch=001.ckpt   --output data/reference/table.parquet   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
````

Replace:

```markdown
sector radii and their least gap, largest-radius manifold error, and float32/float64 distance
agreement over all ordered pairs in row chunks. It also reports nonzero gradient norms of all
three weighted terms and both scales. Failed criteria write the report and exit 1.

The distance comparison requires relative error at most 1e-3 on noncoincident pairs. Exact
```

with:

```markdown
sector radii and their least gap, largest-radius manifold error, and float32/float64 distance
agreement over all ordered pairs in row chunks. It also reports nonzero gradient norms of all
three weighted terms and both scales. Failed criteria write the report and exit 1. The radius
checks are the hyperbolic arm's: a flat arm's report names its geometry, records `radius` as
null and checks its two terms' and both scales' gradients.

The distance comparison requires relative error at most 1e-3 on noncoincident pairs. Exact
```

In `docs/usage.md`, make these 3 edits, in order.

Replace:

```markdown
### `tools export-table`

Export a checkpoint's 2,125-code table as `code`, `index`, `level`, then float64 bounded tangent
coordinates `e0` through `e{d-1}`, in codebook order. Each code goes through the checkpoint's
model in eval mode. The current text objective uses unit curvature and
`r = R * tanh(norm(v) / R)`, rather than a cap at 2.

The supervision contract must match the configured bundle. The checkpoint's encoder record is
its own, so a dimension-8 checkpoint can export under a dimension-16 config. Pre-`req11-v1`
checkpoints are refused before model loading. Different window summaries or tokenizer pins are
refused. Export writes `<stem>_provenance.json`, with checkpoint and contract identities,
backbone revision, tokenizer/window/summaries identities, descriptions hash and table fingerprint.

Use the monitor's selected checkpoint; epoch 1 is an example here:
```

with:

```markdown
### `tools export-table`

Export a checkpoint's 2,125-code table as `code`, `index`, `level`, then float64 coordinates
`e0` through `e{d-1}`, in codebook order. Each code goes through the checkpoint's model in eval
mode. A hyperbolic arm writes bounded tangents, with unit curvature and
`r = R * tanh(norm(v) / R)` rather than a cap at 2; a Euclidean arm writes v and a spherical arm
u = v / norm(v) ([geometry arms](text_training.md#geometry-arms)).

The supervision contract must match the configured bundle. The checkpoint's encoder record is
its own, so a dimension-8 checkpoint can export under a dimension-16 config. Pre-`req11-v1`
checkpoints are refused before model loading. Different window summaries or tokenizer pins are
refused. Export writes `<stem>_provenance.json`, with checkpoint and contract identities,
backbone revision, tokenizer/window/summaries identities, descriptions hash, geometry and its
coordinates, and table fingerprint.

Use the monitor's selected checkpoint; epoch 1 is an example here:
```

Replace:

````markdown
Build one arm record from complete trained seed directories. Before the first export or decision
read, the runner checks every seed's epoch coverage, panel fingerprint, training-run id, seed,
21 settings, earliest best checkpoint and exact saved best score. Both last and selected
checkpoints must also match the current config in their saved LoRA rank/alpha/dropout and active
MoE expert count/top-k/hidden dimension/load-balancing coefficient. Missing required values are
refused; the arm retains its 21-key settings identity. A selected epoch with a
versioned sibling is ambiguous and refused. The arm's backbone revision is resolved independently
from its cached backbone; it cannot be copied from the text-only comparator.

```bash
````

with:

````markdown
Build one arm record from complete trained seed directories. Before the first export or decision
read, the runner checks every seed's epoch coverage, panel fingerprint, training-run id, seed,
monitor distance, 21 settings, earliest best checkpoint and exact saved best score. The arm's
geometry comes from `model.geometry`; each checkpoint's encoder record must name it. Both last
and selected checkpoints must also match the current config in their saved LoRA
rank/alpha/dropout and active MoE expert count/top-k/hidden dimension/load-balancing
coefficient. Missing required values are refused; the arm retains its 21-key settings identity.
A selected epoch with a versioned sibling is ambiguous and refused. The arm's backbone revision
is resolved independently from its cached backbone; it cannot be copied from the text-only
comparator.

```bash
````

Replace:

````markdown
and their least gap, manifold error at the largest radius, and chunked all-pairs float32/float64
distance agreement. It reports nonzero gradient norms for all three weighted terms and both
scales on the saved seed's epoch-zero, step-zero batch.

```bash
````

with:

````markdown
and their least gap, manifold error at the largest radius, and chunked all-pairs float32/float64
distance agreement. It reports nonzero gradient norms for all three weighted terms and both
scales on the saved seed's epoch-zero, step-zero batch. The radius checks are the hyperbolic
arm's: a flat arm's report names its geometry, records `radius` as null and checks the task and
code-to-code terms and both scales.

```bash
````

In `tests/README.md`, make these 3 edits, in order.

Replace:

```markdown
## Current Status

The suite contains **99 unit** test files and **2 integration** files, with **3,001 collected
nodes**. Actual skip counts depend on local data and hardware capabilities, including MPS.
Collection counts describe the suite inventory, not measured coverage percentages.
```

with:

```markdown
## Current Status

The suite contains **100 unit** test files and **2 integration** files, with **3,092 collected
nodes**. Actual skip counts depend on local data and hardware capabilities, including MPS.
Collection counts describe the suite inventory, not measured coverage percentages.
```

Replace:

````markdown
```text
tests/
├── unit/                         # 99 unit test files
├── integration/
│   ├── test_reference_training.py # tiny-backbone Trainer workflows
````

with:

````markdown
```text
tests/
├── unit/                         # 100 unit test files
├── integration/
│   ├── test_reference_training.py # tiny-backbone Trainer workflows
````

Replace:

```markdown
| Shared encoder and fusion | `test_encoder.py`, `test_fusion.py`, `test_moe.py`: shared adapters, markers, absent-channel masking and fusion |
| Objective and radius | `test_loss.py`, `test_hyperbolic.py`, `test_naics_model.py`: task/code-code/radial terms, scales, bounded live radius and polar distance |
| Epoch loader | `test_datamodule.py`, `test_tokenization_cache.py`: seed/epoch permutations, exact coverage, code chunks, query tokens and cache identities |
| Cache and outcome monitor | `test_monitor.py`: eval/no-grad refresh, detached candidates, live-anchor replacement, validation reads and durable records |
```

with:

```markdown
| Shared encoder and fusion | `test_encoder.py`, `test_fusion.py`, `test_moe.py`: shared adapters, markers, absent-channel masking and fusion |
| Objective and radius | `test_loss.py`, `test_hyperbolic.py`, `test_naics_model.py`: task/code-code/radial terms, scales, bounded live radius and polar distance |
| Geometry arms | `test_heads.py`: the Euclidean, spherical and hyperbolic heads, their flat or polar training distances, read maps and the one list of geometries (Req 12) |
| Epoch loader | `test_datamodule.py`, `test_tokenization_cache.py`: seed/epoch permutations, exact coverage, code chunks, query tokens and cache identities |
| Cache and outcome monitor | `test_monitor.py`: eval/no-grad refresh, detached candidates, live-anchor replacement, validation reads and durable records |
```

- [ ] **Step 2: Build the docs and check the new anchor and API section**

Run: `uv run mkdocs build --strict -d /tmp/naics_plan12_site`
Expected: the build succeeds with no warning.

Run: `grep -c 'id="geometry-arms"' /tmp/naics_plan12_site/text_training/index.html`
Expected: `1`.

Run:

```bash
grep -c 'href="../text_training/#geometry-arms"' /tmp/naics_plan12_site/overview/index.html /tmp/naics_plan12_site/usage/index.html
```
Expected: `1` for each file. MkDocs does not check anchors under `--strict` here, so this is the
anchor check.

Run:

```bash
grep -o 'id="naics_embedder.text_model.heads.build_head"' /tmp/naics_plan12_site/api/encoder/index.html
```
Expected: one line, `id="naics_embedder.text_model.heads.build_head"`.

Run: `rm -rf /tmp/naics_plan12_site`

- [ ] **Step 3: Check the counts the docs give**

Run: `find src -name '*.py' | wc -l`, then `find tests/unit -name 'test_*.py' | wc -l`
Expected: `118`, then `100`.

Run: `uv run pytest --collect-only -q -q | tail -1`
Expected: `3092 tests collected` followed by a time.

- [ ] **Step 4: Check the line lengths**

Write this script to `/tmp/naics_plan12_check_lines.py` with the Write tool:

````python
'''Plan 12: Markdown lines this change adds stay within 100 columns, outside fences and tables.'''

import re
import subprocess
import sys
from pathlib import Path

base, paths = sys.argv[1], sys.argv[2:]
long_lines = []
for path in paths:
    diff = subprocess.run(['git', 'diff', '-U0', base, '--', path],
                          check=True,
                          capture_output=True,
                          text=True).stdout
    added = set()
    for start, count in re.findall(r'^@@ -\S+ \+(\d+)(?:,(\d+))? @@', diff, flags=re.M):
        first = int(start)
        added.update(range(first, first + (int(count) if count else 1)))
    fenced = False
    for number, line in enumerate(Path(path).read_text().split('\n'), 1):
        if line.lstrip().startswith('```'):
            fenced = not fenced
            continue
        if number in added and not fenced and not line.startswith('|') and len(line) > 100:
            long_lines.append(f'{path}:{number} ({len(line)})')
print(long_lines)
````

Run:

```bash
python3 /tmp/naics_plan12_check_lines.py HEAD CLAUDE.md README.md docs/api/encoder.md docs/api/export.md docs/api/radius_report.md docs/overview.md docs/quickstart.md docs/text_training.md docs/usage.md tests/README.md
```
Expected: `[]`. Only the lines this task adds are checked, and lines in fences and tables are exempt
(**Project rules**). `README.md`'s long fifth line is older than this plan.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -n auto -q`
Expected: `3090 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md README.md docs/api/encoder.md docs/api/export.md docs/api/radius_report.md docs/overview.md docs/quickstart.md docs/text_training.md docs/usage.md tests/README.md
git commit -m "docs: describe the three geometry arms (Req 12)"
```

Keep `/tmp/naics_plan12_check_lines.py` for Task 16; it is deleted there.

### Task 9: The Phase 1 Exit on Stage 7's artifacts (controller, inline)

The Exit runs the worktree's code on Stage 7's real artifacts, read-only (P15). It trains nothing,
reads no panel and opens no split. It checks the claims this phase makes about Stage 7:
- every reference seed still preflights;
- the reference record and margins still pass the decision's checks;
- the hyperbolic arm's report and export are Stage 7's, bit for bit, but for the added geometry
  keys (P6, P9, P11);
- the flat arms' terms and scales get gradient on the real bundle (Verification "No inert terms");
- plan 9's text-only table, 384 wide, reduces to 8, 16 and 32 under the fingerprint
  `reference.json` records, so one table serves every cell (D9).

Every command runs from the main checkout's root, with the worktree's code:
`uv run --project WORKTREE --locked --no-sync` keeps the working directory, so relative paths read
the main checkout's `data/`, `checkpoints/` and `logs/`, while `naics_embedder` imports from
WORKTREE. The exit script asserts it.

**Files:** none committed. Outputs, under `/tmp` only: `/tmp/naics_plan12_rr_s1.json`,
`/tmp/naics_plan12_s1.parquet`, `/tmp/naics_plan12_s1_provenance.json` and two scripts, all
deleted in Step 7. Runs may build `data/token_cache/` in the main checkout and append to its
`logs/tools_radius_report.log` and `logs/tools_export_table.log` (**Project rules**).

**Interfaces:**
- Consumes: Tasks 1–8; the real inputs in **Workspace**.
- Produces: the numbers the first PR carries:
  - the ten seeds' preflight epochs;
  - the radius report's equality with Stage 7's;
  - the export's table and matrix hashes;
  - the two flat arms' four gradient norms;
  - the text-only table's reductions to the three dimensions;
  - the unchanged selection log.

Record each command's output in the ledger.

- [ ] **Step 1: Check both checkouts**

Run: `pwd`, then `git status --short`, then `git log --oneline origin/main..HEAD`
Expected: the worktree's root; no output; then the eight task commits, newest first.

Run: `git -C /Users/lowell/Projects/naics-embedder status --short`
Expected: no output. The exit script checks below that the main checkout's `conf/config.yaml`
parses to the worktree's but for the manifest path. Its branch does not matter otherwise.

Run:

```bash
wc -l /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl && shasum -a 256 /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl
```
Expected: Pre-flight Step 4's values.

- [ ] **Step 2: Seed 1's radius report under the new code**

The paths must be given exactly as below: the report records them resolved, and Step 4 compares
them with Stage 7's report.

Run:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 uv run --project WORKTREE --locked --no-sync naics-embedder tools radius-report --checkpoint 'checkpoints/stage7-reference-s1/epoch=003.ckpt' --table 'checkpoints/stage7-reference-s1/arm_table_epoch=003.parquet' --output /tmp/naics_plan12_rr_s1.json supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```
Expected: exit 0, the report printed. It ends with the term gradients, the last being
`"task": 15.713854933674163`. At plan time it took 28 s.

- [ ] **Step 3: Re-export seed 1**

Run:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 uv run --project WORKTREE --locked --no-sync naics-embedder tools export-table --checkpoint 'checkpoints/stage7-reference-s1/epoch=003.ckpt' --output /tmp/naics_plan12_s1.parquet supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```
Expected: exit 0, ending `Code table: /tmp/naics_plan12_s1.parquet` and
`Provenance: /tmp/naics_plan12_s1_provenance.json`.

- [ ] **Step 4: Compare both with Stage 7's**

Write this script to `/tmp/naics_plan12_compare_seed1.py` with the Write tool:

```python
'''Plan 12, Task 9: seed 1's re-run radius report and re-export against Stage 7's.'''

import hashlib
import json
from pathlib import Path

SEED = Path('checkpoints/stage7-reference-s1')

def flat(record, prefix=''):
    out = {}
    for key, value in record.items():
        if isinstance(value, dict):
            out.update(flat(value, f'{prefix}{key}.'))
        else:
            out[f'{prefix}{key}'] = value
    return out

# The radius report: Stage 7's keys and values exactly, plus the geometry (P11)
saved = json.loads((SEED / 'radius_report.json').read_text())
rerun = json.loads(Path('/tmp/naics_plan12_rr_s1.json').read_text())
assert sorted(set(rerun) - set(saved)) == ['geometry'], sorted(set(rerun) ^ set(saved))
assert not set(saved) - set(rerun)
changed = sorted(key for key in saved if saved[key] != rerun[key])
assert changed == [], changed
assert rerun['geometry'] == 'hyperbolic' and rerun['passed'] is True
print('radius report: Stage 7 values exactly, geometry', rerun['geometry'])

# The export: the same table, its provenance Stage 7's plus the geometry and the new time (P9)
old = flat(json.loads((SEED / 'arm_table_epoch=003_provenance.json').read_text()))
new = flat(json.loads(Path('/tmp/naics_plan12_s1_provenance.json').read_text()))
added = sorted(set(new) - set(old))
assert added == ['contract.encoder.geometry', 'geometry'], added
assert not set(old) - set(new)
changed = sorted(key for key in old if old[key] != new[key])
assert changed == ['generated_at'], changed
assert new['geometry'] == new['contract.encoder.geometry'] == 'hyperbolic'
table = hashlib.sha256(Path('/tmp/naics_plan12_s1.parquet').read_bytes()).hexdigest()
assert table == old['table_sha256'] == new['table_sha256'], table
print('export: table', table[:8], 'matrix', new['matrix_fingerprint'][:8], 'added', added)
```

Run:

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --project WORKTREE --locked --no-sync python /tmp/naics_plan12_compare_seed1.py
```
Expected, exactly:

```text
radius report: Stage 7 values exactly, geometry hyperbolic
export: table a2dee842 matrix 51f17bd3 added ['contract.encoder.geometry', 'geometry']
```
The radius report's 25 anchor gradients, five term norms and every radius check equal Stage 7's,
which is P6's bit-for-bit claim on a real checkpoint. The table's sha256 `a2dee842…` and matrix
fingerprint `51f17bd3…` are Stage 7's seed-1 table's (P9).

- [ ] **Step 5: The exit checks**

Write this script to `/tmp/naics_plan12_exit.py` with the Write tool:

```python
'''Plan 12, Task 9: the Phase 1 Exit's read-only checks on Stage 7's records and the real bundle.'''

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

import naics_embedder
from naics_embedder.cli.commands import training
from naics_embedder.cli.commands.tools import _run_bundle, _run_config, _sweep_spec
from naics_embedder.decision.decide import (
    _check_monitor_records,
    check_arm,
    check_margins,
    check_margins_first,
)
from naics_embedder.decision.records import ArmRecord, MarginRecord, read_record
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import ArmTables
from naics_embedder.text_model.checkpoint_runner import CheckpointRunner
from naics_embedder.text_model.radius_report import term_gradients
from naics_embedder.utils.training import run_settings

WORKTREE = Path(sys.argv[1])
MANIFEST = (
    'data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json'
)
RECORDS = Path.home() / 'naics-artifacts' / 'records' / 'stage7'
SELECTION_LOG = Path('logs/selection_log.jsonl')
SCRATCH_LOG = Path('/tmp/naics_plan12_never_written.jsonl')

def log_identity():
    data = SELECTION_LOG.read_bytes()
    return len(data.splitlines()), hashlib.sha256(data).hexdigest()

code = Path(naics_embedder.__file__).resolve()
print('code under test:', code)
assert code.is_relative_to(WORKTREE.resolve()), 'the code under test is not the worktree'
before = log_identity()
print('selection log before:', before)

# 0. The main checkout's config is the worktree's but for the manifest path
override = [f'supervision.manifest_path={MANIFEST}']
cfg = _run_config('conf/config.yaml', override)
shipped = _run_config(str(WORKTREE / 'conf' / 'config.yaml'), override)
assert cfg.model_dump() == shipped.model_dump(), 'the main checkout config differs'

# 1. Preflight every Stage 7 seed under the Stage 8 code: geometry, distance, settings, contract
bundle = _run_bundle(cfg)
spec = _sweep_spec(cfg, name='reference', accelerator='cuda')
reference = read_record(RECORDS / 'reference.json', ArmRecord)
assert spec.model_dump() == reference.spec.model_dump(), 'the arm spec differs from Stage 7'
panel = OutcomePanel.from_bundle(bundle, SCRATCH_LOG)
assert panel.fingerprint == reference.panels.outcome
runner = CheckpointRunner(
    cfg,
    bundle,
    run_directory=lambda seed: Path(f'checkpoints/stage7-reference-s{seed}'),
    device='cpu'
)
epochs = {run.seed: run.checkpoint_epoch for run in reference.runs}
for seed in range(1, 11):
    selected = runner.check(spec, seed)
    arm = SimpleNamespace(spec=spec, panels=SimpleNamespace(outcome=panel.fingerprint))
    run = SimpleNamespace(
        seed=seed,
        training_run=selected.training_run,
        checkpoint_epoch=selected.epoch,
        monitor_records=selected.monitor_records
    )
    _check_monitor_records(arm, run)
    assert selected.epoch == epochs[seed], (seed, selected.epoch, epochs[seed])
    print(f'seed {seed}: preflight passed, epoch {selected.epoch}')
assert not SCRATCH_LOG.exists()

# 2. The Stage 7 records under the new distance and dimension guards
store = ArtifactStore(Path.home() / 'naics-artifacts')
margins = read_record(RECORDS / 'margins.json', MarginRecord)
check_arm(reference, store, min_seeds=5)
check_margins(margins)
check_margins_first([reference], margins)
print('reference record and margins: checked')

# 3. The flat arms' terms and scales get gradient on the real bundle's first batch, at d = 8
for geometry in ('euclidean', 'spherical'):
    flat = _run_config(
        'conf/config.yaml', [*override, f'model.geometry={geometry}', 'model.dimension=8', 'seed=1']
    )
    contract = training.runtime_contract_for(flat, bundle)
    datamodule = training.build_datamodule_from_config(flat, bundle)
    model = training.build_model_from_config(
        flat,
        contract,
        bundle,
        run_settings=run_settings(flat, accelerator='cpu', precision='32-true'),
        monitor=None
    )
    datamodule.prepare_data()
    datamodule.setup('fit')
    datamodule.set_train_epoch(0)
    model.refresh_code_cache(datamodule.code_rows)
    gradients = term_gradients(model, datamodule.train_dataset[0])
    print(geometry, json.dumps(gradients, sort_keys=True))
    assert set(gradients) == {'task', 'code_code', 'logit_scale_task', 'logit_scale_code'}
    assert all(math.isfinite(value) and value > 0 for value in gradients.values()), gradients

# 4. One text-only table serves every cell (D9): reduced to the arm's width at read time, one
# fingerprint whatever the width
text_only = pl.read_parquet('checkpoints/plan9_exit/text_only.parquet')
codes = pl.read_parquet('checkpoints/stage7-reference-s1/arm_table_epoch=003.parquet')
codes = codes.select('code', 'index', 'level')
generator = np.random.default_rng(0)
for dimension in (8, 16, 32):
    values = [pl.Series(f'e{i}', generator.standard_normal(len(codes))) for i in range(dimension)]
    tables = ArmTables.from_tables(codes.with_columns(values), text_only)
    assert tables.text_only.shape == (len(codes), dimension), tables.text_only.shape
    assert tables.text_only_fingerprint == reference.text_only.table.matrix_fingerprint
print('text-only table: reduced to 8, 16 and 32, under the reference fingerprint')

after = log_identity()
print('selection log after:', after)
assert after == before
print('PHASE 1 EXIT CHECKS PASSED')
```

Run:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 uv run --project WORKTREE --locked --no-sync python /tmp/naics_plan12_exit.py WORKTREE
```
Expected: exit 0 after about 45 s. Among the log output, these lines:
- `code under test:` followed by WORKTREE's `src/naics_embedder/__init__.py`;
- `selection log before:` with Pre-flight Step 4's count and hash;
- `seed 1: preflight passed, epoch 3` through `seed 10: …`, with epochs 3, 2, 22, 2, 14, 24, 28,
  26, 39 and 11 for seeds 1–10;
- `reference record and margins: checked`;
- one `euclidean {…}` and one `spherical {…}` line, each with four positive, finite norms for
  `code_code`, `logit_scale_code`, `logit_scale_task` and `task`;
- `text-only table: reduced to 8, 16 and 32, under the reference fingerprint`;
- `selection log after:`, equal to the line before;
- `PHASE 1 EXIT CHECKS PASSED`.

The models of section 3 are built unseeded, so the flat norms differ run to run. One plan-time
run gave Euclidean task 0.390, code-code 0.338 and scales 0.121 and 0.043. It gave spherical task
1.302, code-code 1.466 and scales 0.385 and 0.107. Record the run's own values.

- [ ] **Step 6: Nothing else changed**

Run: `git -C /Users/lowell/Projects/naics-embedder status --short`
Expected: no output.

Run:

```bash
wc -l /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl && shasum -a 256 /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl
```
Expected: Pre-flight Step 4's values.

Run: `ls /tmp/naics_plan12_never_written.jsonl`
Expected: "No such file or directory": building the outcome panel logged nothing.

- [ ] **Step 7: Clean up**

Run:

```bash
rm /tmp/naics_plan12_rr_s1.json /tmp/naics_plan12_s1.parquet /tmp/naics_plan12_s1_provenance.json /tmp/naics_plan12_compare_seed1.py /tmp/naics_plan12_exit.py
```

## Final verification, Phase 1 (controller, inline)

Run every check before the final review, and paste each output into the ledger.

- [ ] **Step 1: The suite on both CI versions**

Run: `uv run pytest -n auto -q`
Expected: `3090 passed, 2 skipped`.

Run:

```bash
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics_plan12_py310 uv run --locked pytest -n auto -q
```
Expected:
- `3090 passed, 2 skipped`, the same counts as 3.12 (measured at plan time);
- a few hundred extra "encountered in matmul" RuntimeWarnings, which come from numpy 2.2 with
  Accelerate on 3.10 and are not a failure.

Run: `rm -rf /tmp/naics_plan12_py310`

- [ ] **Step 2: Lint, format and docs**

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0, with no file listed.

Run: `uv run mkdocs build --strict -d /tmp/naics_plan12_site`, then `rm -rf /tmp/naics_plan12_site`
Expected: the build succeeds with no warning.

- [ ] **Step 3: The branch**

Run: `git log --oneline origin/main..HEAD`
Expected: the eight task commits, newest first, and none of the six private commits:

```text
docs: describe the three geometry arms (Req 12)
test(integration): train the flat arms through the Trainer (Req 12)
feat(tools): a flat arm's radius report checks its terms and scales alone
feat(decision): guard each seed's and each read's distance and dimension (Req 12)
feat(export): export and read each geometry arm's own coordinates (Req 2, Req 12)
feat(text_model): train under the head's distance, the radial term in hyperbolic only
feat(config): make geometry a configuration factor in the encoder record (Req 12)
feat(text_model): add the Euclidean and spherical geometry heads (Req 12)
```

Run: `git diff --stat origin/main...HEAD | tail -1`
Expected: `44 files changed, 1660 insertions(+), 314 deletions(-)`.

Run: `git diff --name-only origin/main...HEAD -- uv.lock conf/data`
Expected: no output. The lock and the panels' configs are untouched.

- [ ] **Step 4: No caller branches on the geometry (P3)**

Run: `git grep -n -E "[!=]= *'(euclidean|spherical|hyperbolic)'" -- src/naics_embedder/text_model`
Expected: exactly `build_head`'s three lines:

```text
src/naics_embedder/text_model/heads.py:244:    if geometry == 'hyperbolic':
src/naics_embedder/text_model/heads.py:246:    if geometry == 'euclidean':
src/naics_embedder/text_model/heads.py:248:    if geometry == 'spherical':
```
Outside `text_model/`, the HGCN feeder and `train`'s prompt (P10), the tie order (Req 5), the
encoder record's four-copy check (P7) and the diagnostics' geometry dispatch (Stage 4) compare
names, by design.

- [ ] **Step 5: The final review**

Dispatch the code-reviewer agent, which is pinned to Opus, on the whole branch, with:
- `git diff origin/main...HEAD -- src tests conf docs CLAUDE.md README.md`;
- this plan;
- the spec's Req 12, Req 5 and Req 2, and the roadmap's Stage 8 entry.

Optionally run Codex beside it: `codex exec review -m gpt-6-astra` on the same diff (the config's
default model fails for this account; never edit `~/.codex/config.toml`).

Then, for each finding:
- fix it, with a test where it is behavior, and rerun Steps 1–4; or
- triage it as deferred, which Plan completion's gate handles.

## The hard checkpoint: the first PR (controller, inline)

Phase 1 ends here (P1). No Lambda time is spent until this PR has merged.

- [ ] **Step 1: Open the PR, with the user's go-ahead**

Run: `git log --oneline origin/main..HEAD`
Expected: Final verification Step 3's list.

Ask the user before pushing. Then run `git push -u origin claude/stage-8-geometry-arms`, never a
bare `git push`, and open the PR against `main`. The PR description carries:
- what the stage changes (the plan's Goal and Architecture, shortened);
- the recorded deviations (**Global Constraints**), one line each. Flag P5, the spherical arm's
  export of û, as the interpretation the reviewer should confirm;
- Task 9's numbers: the ten preflight epochs, the radius report's equality with Stage 7's, the
  export's hashes, the flat arms' gradient norms, and the unchanged selection log;
- the suite counts, local and CI.

End the description with the session's PR attribution line.

- [ ] **Step 2: Wait for review and merge**

The PR's review and merge are the hard checkpoint. GitHub auto-merge is off, and main has no
required checks.
- Watch `gh pr checks <number>` until the lint and both test jobs pass. CI runs Linux with MKL,
  where this Mac's Accelerate BLAS hides GEMM differences. Task 7's exact equality of the monitor's
  MRR with the exported read is the likeliest to differ there. If only exact-equality assertions
  fail, suspect BLAS before logic. Stop and ask, with the failing assertions and their
  differences, before changing a tolerance.
- Wait for Codex's GitHub review: a 👍 means no findings, and 👀 means it is still reviewing. Fix
  its inline findings first.
- The user merges. Never merge without the user's go-ahead. If the user asks you to merge, run
  `gh pr merge <number> --merge --match-head-commit <sha>`.

Before pushing a follow-up fix, check that the PR is still open (`gh pr view <number> --json
state`). If the user merged meanwhile, land the follow-up as a new PR from `origin/main`.

- [ ] **Step 3: Keep the ledger, then finish the branch**

Finishing removes this worktree, and with it every ignored file. Keep the execution ledger first.
Skip this in the main checkout.

Run: `ls /Users/lowell/Projects/naics-embedder/logs/plan12_worktree`
Expected: "No such file or directory". If it exists, stop and ask.

Run: `mkdir /Users/lowell/Projects/naics-embedder/logs/plan12_worktree`, then
`cp -c logs/plan12_execution.md /Users/lowell/Projects/naics-embedder/logs/plan12_worktree/`

Then hand off to finishing-a-development-branch, which removes the worktree after the merge. Phase
2 starts in a fresh session, from the main checkout. Plan completion's markup waits for Phase 2.

## Phase 2: the campaign

Phase 2 trains the eight new cells and decides among all nine. It starts only after the first PR
has merged (P1). It runs from the main checkout, inline in the controller session, with no
subagents: every step reads or writes the main checkout's gitignored `checkpoints/` and `logs/`,
which are the campaign's home, and the Lambda instance is the user's.

**The cells (P12, P13),** in the order Phase 2 trains them. The two new geometries come first, at
the reference dimension, then the hyperbolic arm at its other dimensions, then the rest:

| Order | `model.geometry` (`<g>`) | `model.dimension` (`<d>`) | Arm | Seed s's run directory |
|---:|---|---:|---|---|
| 1 | euclidean | 16 | `euclidean-d16` | `checkpoints/stage8-euclidean-d16-s<s>/` |
| 2 | spherical | 16 | `spherical-d16` | `checkpoints/stage8-spherical-d16-s<s>/` |
| 3 | hyperbolic | 8 | `hyperbolic-d8` | `checkpoints/stage8-hyperbolic-d8-s<s>/` |
| 4 | hyperbolic | 32 | `hyperbolic-d32` | `checkpoints/stage8-hyperbolic-d32-s<s>/` |
| 5 | euclidean | 8 | `euclidean-d8` | `checkpoints/stage8-euclidean-d8-s<s>/` |
| 6 | euclidean | 32 | `euclidean-d32` | `checkpoints/stage8-euclidean-d32-s<s>/` |
| 7 | spherical | 8 | `spherical-d8` | `checkpoints/stage8-spherical-d8-s<s>/` |
| 8 | spherical | 32 | `spherical-d32` | `checkpoints/stage8-spherical-d32-s<s>/` |
| — | hyperbolic | 16 | `reference` | Stage 7's `checkpoints/stage7-reference-s<s>/`, seeds 1–10 |

**Rules for every Phase 2 step:**

- Run from `/Users/lowell/Projects/naics-embedder`, on local `main` once Task 10 has rebuilt it.
  Run `pwd` and `git branch --show-current` first.
- Write each command out in full, with `<g>`, `<d>`, `<s>`, `<IP>` and `<kkk>` substituted. The
  Bash tool keeps no shell variables between calls.
- Prefix every command that loads the backbone or its tokenizer with `HF_HUB_OFFLINE=1`. If a
  `remote` command cannot reach the instance over SSH, the Bash tool's environment lacks the
  user's SSH agent: prefix the command with `SSH_AUTH_SOCK=<socket>`, where
  `launchctl getenv SSH_AUTH_SOCK` prints the socket, as Stage 7's ledger did.
- A run's training overrides are exactly these four, plus the manifest override below:

```text
seed=<s> experiment_name=stage8-<g>-d<d>-s<s> model.geometry=<g> model.dimension=<d>
```

  They go on `remote up`, on `remote train` and on any resume. Nothing else changes (P12).
- The manifest is passed as the `key=value` override

```text
supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

  on every command that reads the bundle. Local `main`'s private `config` commit sets the same
  path; the override keeps each command independent of it.
- Never open a sealed split: no `--split test`, no `OutcomePanel.open_test`, no
  `RegressorPanel.open_outer`.
- `logs/selection_log.jsonl` is append-only (Req 4). Never delete, move or edit it.
- Continue a run only through `remote train --resume`, which resumes from its `last.ckpt`. Never
  resume from an older kept checkpoint (the older-kept-checkpoint item), never relaunch a failed
  run under the same `experiment_name`, and never resume a finished run to extend it.
- `uv.lock` stays frozen (**Project rules**). Check its hash at the start and end of each task. If
  any step would change it, stop and ask.
- Never push local `main`, a private commit or a descendant.
- Launching and terminating Lambda instances is the user's action. Ask for each instance's
  address, and tell the user when `remote finish` prints "Safe to terminate".
- Record every command and its output in `logs/plan12_campaign.md`.

### Task 10: Phase 2 pre-flight (controller, inline)

**Files:** none committed. Local `main` is rebuilt onto the merged code, and never pushed.

**Interfaces:**
- Consumes: the merged first PR (Tasks 1–9); the six private commits.
- Produces: local `main` = merged origin/main plus the six private commits; the checked lock,
  records, panels and log.

- [ ] **Step 1: Confirm the first PR has merged, and pin it**

Run: `git fetch origin`, then:

```bash
git log --oneline -1 origin/main -- src/naics_embedder/text_model/heads.py
```
Expected: one commit, Task 1's. If there is none, the first PR has not merged: stop.

Run: `git rev-parse origin/main`
Record the 40-character SHA in the ledger as BASE. Every later step of this task names BASE's
value, so a concurrent fetch cannot move it (other sessions share `origin/main`).

Run: `git status --short`, then `git branch --show-current`
Expected: no output, then `main`. If the main checkout is on another branch, ask the user before
switching it: other sessions may be using it.

Run: `git log --oneline BASE..main`
Expected: the six private commits, by subject, and nothing else but merges and cd0f80b's roadmap
commit:
- `config` and `graph config`;
- `fix: enforce verified offline model loading`;
- `fix: defer checkpoint contract import for remote worker`;
- `Fix Mac sync exec ownership and tmux command quoting`;
- `chore(local): point QCEW panel at verified project data`.

At plan time local `main` was 22df5e1, and the list also held 22df5e1, cd0f80b and aa7ac39. If
another commit appears, stop and ask. If `main` already sits on BASE with exactly the six commits
above it, someone has synced it: skip Step 2.

- [ ] **Step 2: Replay the six private commits onto BASE**

The replay drops aa7ac39 and 22df5e1, two merges, and cd0f80b, which PR #129 published. Use the
six commits' SHAs on local `main`, oldest first. At plan time they were aa8ebd6, 3fc580a, d7927a5,
5ce248f, b94ecf3 and 2d1f045. If a sync rewrote them, take them from Step 1's list by subject.

Run: `git worktree add --detach /tmp/naics_plan12_sync BASE`

Run: `git -C /tmp/naics_plan12_sync cherry-pick aa8ebd6 3fc580a d7927a5 5ce248f b94ecf3 2d1f045`
Expected: six commits applied, with no conflict. The plan-time replay onto the dry run's tip was
clean. On a conflict, run `git -C /tmp/naics_plan12_sync cherry-pick --abort` and
`git worktree remove --force /tmp/naics_plan12_sync`, then stop and ask.

Run: `git -C /tmp/naics_plan12_sync diff --name-only BASE HEAD`
Expected: exactly these 17 paths:

```text
conf/config.yaml
conf/data/regressor_panel.yaml
conf/graph.yaml
docs/remote_workflow.md
specs/lambda-remote-workflow.md
src/naics_embedder/remote/canonical.py
src/naics_embedder/remote/launch.py
src/naics_embedder/remote/loop.py
src/naics_embedder/remote/model_cache.py
src/naics_embedder/remote/worker.py
src/naics_embedder/remote/workflow.py
tests/fixtures/remote.py
tests/unit/test_remote_canonical.py
tests/unit/test_remote_launch.py
tests/unit/test_remote_lifecycle_guards.py
tests/unit/test_remote_model_cache.py
tests/unit/test_remote_transport.py
```

Run:

```bash
git -C /tmp/naics_plan12_sync diff BASE HEAD -- conf/config.yaml conf/data/regressor_panel.yaml
```
Expected: two changed lines. `supervision.manifest_path` goes from `null` to the bundle's manifest
path, and `qcew_dir` goes to `/Users/lowell/Projects/naics-embedder/data/QCEW`. The
`geometry: hyperbolic` line stays. If anything else differs, stop and ask.

Run: `git -C /tmp/naics_plan12_sync rev-parse HEAD`
Record it in the ledger as the campaign's source, SOURCE.

Run: `git reset --keep SOURCE`, in the main checkout, on `main`
Expected: `main` moves to SOURCE, and the clean working tree follows it.

Run: `git worktree remove /tmp/naics_plan12_sync`

Run: `git log --oneline origin/main..main`, then `git diff --name-only BASE main | wc -l`
Expected: the six private commits, then `17`. Never push `main`. The pre-push hook refuses it
anyway.

- [ ] **Step 3: The suite on the campaign's source**

Run: `uv sync --locked`, then `uv run pytest -n auto -q`
Expected: exactly two failures, the two pin tests the private config commits fail by design:

```text
tests/unit/test_config.py::test_base_config_parses_as_repaired_pre_generation
tests/unit/test_config.py::TestRegressorPanelConfig::test_yaml_matches_defaults_but_for_the_branch_record
```

Every other test passes or skips. At plan time the replay onto the dry run's tip gave `2 failed,
3152 passed, 2 skipped` in a worktree without `data/`. The main checkout has `data/`, so plan 9's
local-only test runs there too. If any other test fails, stop and ask.

- [ ] **Step 4: The lock, the records and the log**

Run: `shasum -a 256 uv.lock`
Expected: `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21  uv.lock`.

Run:

```bash
uv run python -c "import torch, pytorch_lightning; print(torch.__version__, pytorch_lightning.__version__)"
```
Expected: `2.9.1 2.5.5`.

Run:

```bash
shasum -a 256 /Users/lowell/naics-artifacts/records/stage7/reference.json /Users/lowell/naics-artifacts/records/stage7/margins.json
```
Expected: `c885b9c5dc18b6be03670d0cb5a71db3974917da8f719e0dbb1f1ef5eec2d1a2` and
`e619b3b30fcad07ab95b23c4f7cfcba52327db011a02a378874d7017ce10fd2f`.

Run: `wc -l logs/selection_log.jsonl && shasum -a 256 logs/selection_log.jsonl`
Record both. Phase 1 appended nothing, so they are Pre-flight Step 4's values unless another
session has read a panel since.

Run: `ls -d checkpoints/stage8-* /Users/lowell/naics-artifacts/records/stage8`
Expected: "No such file or directory" for both. If either exists, stop and ask.

Run: `mkdir /Users/lowell/naics-artifacts/records/stage8`

- [ ] **Step 5: The panels Stage 8 reads are Stage 7's**

`check_pairing` requires every arm of the decision to read the reference's panels, with the same
data and fit settings. Rebuild them, without reading them, and compare.

Write this script to `/tmp/naics_plan12_panels.py` with the Write tool:

```python
'''Plan 12, Task 10: rebuild the panels Stage 8's sweeps read, and compare them with Stage 7's.'''

import sys
from dataclasses import asdict
from pathlib import Path

from naics_embedder.cli.commands.tools import REGRESSOR_PANEL_CONFIG, _run_bundle, _run_config
from naics_embedder.decision.records import ArmRecord, read_record
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import DECISION_LEVEL, load_regressor_panel
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.utils.config import RegressorPanelConfig, load_config

MANIFEST = (
    'data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json'
)
REFERENCE = Path.home() / 'naics-artifacts' / 'records' / 'stage7' / 'reference.json'
SCRATCH_LOG = Path('/tmp/naics_plan12_panels_never_written.jsonl')

cfg = _run_config('conf/config.yaml', [f'supervision.manifest_path={MANIFEST}'])
bundle = _run_bundle(cfg)
outcome = OutcomePanel.from_bundle(bundle, SCRATCH_LOG)
regressor_cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
regressor = load_regressor_panel(
    regressor_cfg, bundle.artifact_path('codebook'), log_path=SCRATCH_LOG, levels=[DECISION_LEVEL]
)
settings = asdict(regressor.settings)
rebuilt = {
    'outcome': outcome.fingerprint,
    'outcome_data': outcome.data_fingerprint(IndexRole.VALIDATION),
    'regressor': regressor.fingerprint,
    'regressor_data': regressor.data_fingerprint(DECISION_LEVEL),
    'fit_settings': {
        **settings, 'alphas': list(settings['alphas'])
    },
}
recorded = read_record(REFERENCE, ArmRecord).panels.model_dump()
wrong = sorted(key for key in recorded if recorded[key] != rebuilt[key])
assert not SCRATCH_LOG.exists(), 'building the panels wrote the selection log'
print('panels differ from the reference in', wrong) if wrong else print('PANELS MATCH THE REFERENCE')
sys.exit(1 if wrong else 0)
```

Run: `HF_HUB_OFFLINE=1 uv run --locked --no-sync python /tmp/naics_plan12_panels.py`
Expected: `PANELS MATCH THE REFERENCE`, and exit 0. At plan time it matched on the dry run's code
with the private QCEW directory. If it prints a difference, stop and ask: no sweep may read until
the panels match.

Run: `rm /tmp/naics_plan12_panels.py`

### Task 11: The first seed of each cell (P14)

**Files:** none committed. Outputs, gitignored: `checkpoints/stage8-<g>-d<d>-s1/` for each cell,
`logs/remote/<session_id>/` and `outputs/remote/<session_id>/`.

**Interfaces:**
- Consumes: the `remote` commands; the merged code's `model.geometry`; the cells table.
- Produces: each cell's seed 1, gated after its first epoch and checked once finished.

The remote workflow holds one session at a time, so runs go one after another; the instance can
stay up between them, as Stage 7's did. Stage 7's seeds trained for 399–1,504 s each on one
A100-SXM4-40GB, plus a few minutes of session setup and teardown. A run's checkpoints, monitor
reads and epoch summary come home with every pull.

Write the two check scripts once, with the Write tool. The first is the gate,
`/tmp/naics_plan12_gate.py`:

```python
'''Plan 12, Tasks 11–12: one run's precision, geometry and monitor reads in the pulled files (P14).'''

import json
import sys
from pathlib import Path

from naics_embedder.panels.decoding import GEOMETRY_DISTANCES
from naics_embedder.text_model.monitor import read_monitor_records
from naics_embedder.utils.training import read_checkpoint

RUN = 'checkpoints/stage8-{geometry}-d{dimension}-s{seed}'

geometry, dimension, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
directory = Path(RUN.format(geometry=geometry, dimension=dimension, seed=seed))
last = read_checkpoint(directory / 'last.ckpt')
hparams = last['hyper_parameters']
settings = hparams['run_settings']
assert (settings['accelerator'], settings['precision']) == ('cuda', 'bf16-mixed'), settings
# A record without a geometry reads as hyperbolic (P7)
saved = (hparams['seed'], hparams.get('geometry', 'hyperbolic'), hparams['dimension'])
assert saved == (seed, geometry, dimension), saved
encoder = last['stage3_supervision']['encoder']
assert (encoder.get('geometry', 'hyperbolic'), encoder['dimension']) == (geometry, dimension)
records = read_monitor_records(directory / 'monitor_reads.jsonl')
assert records and records[0]['read']['detail']['epoch'] == 0, 'the run has no epoch 0 read'
distance = GEOMETRY_DISTANCES[geometry]
logged = [
    json.loads(line)
    for path in Path('logs/remote').glob('*/selection_log.jsonl')
    for line in path.read_text().splitlines() if line.strip()
]
for record in records:
    read, detail = record['read'], record['read']['detail']
    assert (read['event'], read['panel'], read['split']) == ('read', 'outcome', 'validation'), read
    assert (detail['seed'], detail['training_run']) == (seed, last['training_run']), detail
    assert detail['distance'] == distance, (detail['distance'], distance)
    assert read in logged, f'epoch {detail["epoch"]} is not in the pulled selection logs'
print(geometry, dimension, seed, 'cuda bf16-mixed', distance, len(records), records[0]['mrr'])
```

The second checks a finished run, `/tmp/naics_plan12_runs.py`:

```python
'''Plan 12, Tasks 11–12: each finished run is complete on the Mac, its summary its own (P14).'''

import sys
from pathlib import Path

from naics_embedder.text_model.epoch_summary import read_epoch_summary
from naics_embedder.text_model.monitor import read_monitor_records
from naics_embedder.utils.training import read_checkpoint

RUN = 'checkpoints/stage8-{geometry}-d{dimension}-s{seed}'
CELLS = [
    ('euclidean', 16), ('spherical', 16), ('hyperbolic', 8), ('hyperbolic', 32), ('euclidean', 8),
    ('euclidean', 32), ('spherical', 8), ('spherical', 32)
]

if sys.argv[1] == 'all':
    runs = [(geometry, dimension, seed) for geometry, dimension in CELLS for seed in range(1, 6)]
else:
    runs = [(sys.argv[1], int(sys.argv[2]), int(seed)) for seed in sys.argv[3:]]
for geometry, dimension, seed in runs:
    directory = Path(RUN.format(geometry=geometry, dimension=dimension, seed=seed))
    records = read_monitor_records(directory / 'monitor_reads.jsonl')
    epochs = [record['read']['detail']['epoch'] for record in records]
    last = read_checkpoint(directory / 'last.ckpt')
    assert epochs == list(range(last['epoch'] + 1)), (directory, epochs, last['epoch'])
    summary = read_epoch_summary(directory / 'epoch_summary.jsonl')
    assert [row['epoch'] for row in summary] == epochs, directory
    assert [row['mrr'] for row in summary] == [record['mrr'] for record in records], directory
    # The radial term, and so its health log, exists in the hyperbolic arm only (Req 12)
    radial = {'loss/radial' in row for row in summary}
    assert radial == {geometry == 'hyperbolic'}, (directory, radial)
    training_runs = {record['read']['detail']['training_run'] for record in records}
    assert training_runs == {last['training_run']}, (directory, training_runs)
    best = max(records, key=lambda record: (record['mrr'], -record['read']['detail']['epoch']))
    selected = best['read']['detail']['epoch']
    assert (directory / f'epoch={selected:03d}.ckpt').is_file(), (directory, selected)
    print(
        f'{geometry}-d{dimension} s{seed}: epochs {len(epochs)}, selected {selected}, '
        f'mrr {best["mrr"]:.6f}'
    )
```

At plan time both passed on Stage 7's ten runs, with their directory pattern pointed there, and the
gate refused a run of another geometry.

For each cell, in the table's order, make Steps 1–6 with s = 1:

- [ ] **Step 1: Bring up the session**

Ask the user for an instance's address, a new one or the one already up. Then:

Run:

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote up --host ubuntu@<IP> seed=1 experiment_name=stage8-<g>-d<d>-s1 model.geometry=<g> model.dimension=<d> supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```
Expected: the tool check, the bundle gate on the Mac, the bootstrap with its clock and native BF16
checks, then `Ready session <session_id> on ubuntu@<IP>`. Stop and ask on any failure.

- [ ] **Step 2: Launch the run**

Run:

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote train seed=1 experiment_name=stage8-<g>-d<d>-s1 model.geometry=<g> model.dimension=<d> supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```
Expected: `Launched segment <segment_id>`.

- [ ] **Step 3: Gate the first epoch**

Poll `HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote sync --once` and
`uv run --locked --no-sync naics-embedder remote status` until both
`checkpoints/stage8-<g>-d<d>-s1/last.ckpt` and `checkpoints/stage8-<g>-d<d>-s1/monitor_reads.jsonl`
have arrived. Wait between polls without blocking (the Monitor tool, or a background command), and
keep reporting progress. More epochs may have finished by the pull; the gate checks them all.

Run: `HF_HUB_OFFLINE=1 uv run --locked --no-sync python /tmp/naics_plan12_gate.py <g> <d> 1`
Expected: one line, `<g> <d> 1 cuda bf16-mixed <distance> <count> <mrr>`. The distance is the
cell's (`euclidean`, `cosine` or `lorentz`), and the count is at least 1.

If the gate fails, run `uv run --locked --no-sync naics-embedder remote finish --stop-training`,
then stop and ask.

- [ ] **Step 4: Let the run finish**

Poll `remote status` until it reports the run's exit code. Expected: 0, after early stopping or the
40-epoch budget. If the run fails, stop and ask; never relaunch it under the same
`experiment_name` (`train` refuses a fresh start into a used directory).

If the instance has to go before the run ends: run `remote finish`, have the user terminate it and
launch another, run Step 1's `remote up` there, then continue with:

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote train --resume seed=1 experiment_name=stage8-<g>-d<d>-s1 model.geometry=<g> model.dimension=<d> supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

- [ ] **Step 5: Finish the session**

Run: `uv run --locked --no-sync naics-embedder remote finish`
Expected: the final pull, no checksum difference and no instance edit, then "Safe to terminate"
with the latest checkpoint's SHA-256 on both sides. Tell the user the instance may be terminated or
kept for the next run.

- [ ] **Step 6: Check the finished run**

Run: `HF_HUB_OFFLINE=1 uv run --locked --no-sync python /tmp/naics_plan12_gate.py <g> <d> 1`
Expected: the gate's line again, now counting every epoch.

Run: `uv run --locked --no-sync python /tmp/naics_plan12_runs.py <g> <d> 1`
Expected: `<g>-d<d> s1: epochs <n>, selected <k>, mrr <mrr>`. Under a flat geometry, the run's
epoch summary has no `loss/radial`; under hyperbolic, every row has it.

Record for the finding, for each run: the instance and GPU (`remote status`), the session and
segment ids, the epochs trained, the selected epoch and its MRR, the start and end times, and the
duration.

Run: `shasum -a 256 uv.lock`
Expected: Task 10's hash.

### Task 12: Seeds 2–5 of each cell

**Files:** none committed. Outputs, gitignored: `checkpoints/stage8-<g>-d<d>-s<s>/` for s = 2–5,
and their sessions' logs and outputs.

**Interfaces:**
- Consumes: Task 11's gated cells and its two scripts.
- Produces: all 40 runs on the Mac, each checked.

For each cell, in the table's order, and for each seed s from 2 to 5, make Steps 1–5:

- [ ] **Step 1: Bring up the session**

Run:

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote up --host ubuntu@<IP> seed=<s> experiment_name=stage8-<g>-d<d>-s<s> model.geometry=<g> model.dimension=<d> supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```
Expected: `Ready session <session_id> on ubuntu@<IP>`.

- [ ] **Step 2: Launch the run**

Run:

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote train seed=<s> experiment_name=stage8-<g>-d<d>-s<s> model.geometry=<g> model.dimension=<d> supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```
Expected: `Launched segment <segment_id>`.

- [ ] **Step 3: Let the run finish**

Poll `remote sync --once` and `remote status` as in Task 11, until `remote status` reports the
exit code. Expected: 0. If the run fails, stop and ask. If the instance has to go first, run
`remote finish`, bring up another instance with Step 1's command, and continue with:

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder remote train --resume seed=<s> experiment_name=stage8-<g>-d<d>-s<s> model.geometry=<g> model.dimension=<d> supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

- [ ] **Step 4: Finish the session**

Run: `uv run --locked --no-sync naics-embedder remote finish`
Expected: "Safe to terminate", with the latest checkpoint's SHA-256 on both sides.

- [ ] **Step 5: Check the finished run**

Run: `HF_HUB_OFFLINE=1 uv run --locked --no-sync python /tmp/naics_plan12_gate.py <g> <d> <s>`
Expected: `<g> <d> <s> cuda bf16-mixed <distance> <count> <mrr>`.

Run: `uv run --locked --no-sync python /tmp/naics_plan12_runs.py <g> <d> <s>`
Expected: `<g>-d<d> s<s>: epochs <n>, selected <k>, mrr <mrr>`.

Record the run's facts for the finding, as in Task 11 Step 6.

After the last run:

- [ ] **Step 6: All 40 runs**

Run: `uv run --locked --no-sync python /tmp/naics_plan12_runs.py all`
Expected: 40 lines, eight cells of five seeds, with no assertion error. `tools sweep` re-checks
all of it but the epoch summaries (Task 13).

Run: `rm /tmp/naics_plan12_gate.py /tmp/naics_plan12_runs.py`

Run: `shasum -a 256 uv.lock`
Expected: Task 10's hash.

### Task 13: The sweeps

**Files:** none committed. Outputs: `~/naics-artifacts/records/stage8/<arm>.json` for the eight
arms, the store's objects, each run's exported table beside its selected checkpoint, and 120
decision reads appended to `logs/selection_log.jsonl`.

**Interfaces:**
- Consumes: `tools sweep` with `CheckpointRunner`, under Task 5's guards; plan 9's text-only
  table, which D9 reduces to each arm's dimension at read time.
- Produces: the eight arm records the decision reads.

Each sweep checks every seed before the first read: its checkpoints, monitor reads, settings,
contract, geometry and distance. It then exports each seed's selected checkpoint and reads it once
on each of the three validation panels. A refusal before the first read logs nothing. A later
failure leaves logged reads with no record, and the log keeps them.

- [ ] **Step 1: Record the log's count**

Run: `wc -l logs/selection_log.jsonl`
Record the count as the sweeps' FIRST. Task 16's receipts start after it.

- [ ] **Step 2: Sweep each cell**

Run each command in turn. After each, check that the record path was printed and that
`wc -l logs/selection_log.jsonl` grew by exactly 15: five seeds, three panels each. Stop and ask if
a sweep refuses a seed or the count is off.

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage8-euclidean-d16-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --name euclidean-d16 --text-only checkpoints/plan9_exit/text_only.parquet --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/euclidean-d16.json --purpose 'Stage 8 geometry-by-dimension sweep: euclidean-d16, 5 seeds (Req 12)' model.geometry=euclidean model.dimension=16 supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage8-spherical-d16-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --name spherical-d16 --text-only checkpoints/plan9_exit/text_only.parquet --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/spherical-d16.json --purpose 'Stage 8 geometry-by-dimension sweep: spherical-d16, 5 seeds (Req 12)' model.geometry=spherical model.dimension=16 supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage8-hyperbolic-d8-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --name hyperbolic-d8 --text-only checkpoints/plan9_exit/text_only.parquet --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/hyperbolic-d8.json --purpose 'Stage 8 geometry-by-dimension sweep: hyperbolic-d8, 5 seeds (Req 12)' model.geometry=hyperbolic model.dimension=8 supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage8-hyperbolic-d32-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --name hyperbolic-d32 --text-only checkpoints/plan9_exit/text_only.parquet --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/hyperbolic-d32.json --purpose 'Stage 8 geometry-by-dimension sweep: hyperbolic-d32, 5 seeds (Req 12)' model.geometry=hyperbolic model.dimension=32 supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage8-euclidean-d8-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --name euclidean-d8 --text-only checkpoints/plan9_exit/text_only.parquet --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/euclidean-d8.json --purpose 'Stage 8 geometry-by-dimension sweep: euclidean-d8, 5 seeds (Req 12)' model.geometry=euclidean model.dimension=8 supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage8-euclidean-d32-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --name euclidean-d32 --text-only checkpoints/plan9_exit/text_only.parquet --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/euclidean-d32.json --purpose 'Stage 8 geometry-by-dimension sweep: euclidean-d32, 5 seeds (Req 12)' model.geometry=euclidean model.dimension=32 supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage8-spherical-d8-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --name spherical-d8 --text-only checkpoints/plan9_exit/text_only.parquet --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/spherical-d8.json --purpose 'Stage 8 geometry-by-dimension sweep: spherical-d8, 5 seeds (Req 12)' model.geometry=spherical model.dimension=8 supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools sweep --runs 'checkpoints/stage8-spherical-d32-s{seed}' --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --name spherical-d32 --text-only checkpoints/plan9_exit/text_only.parquet --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/spherical-d32.json --purpose 'Stage 8 geometry-by-dimension sweep: spherical-d32, 5 seeds (Req 12)' model.geometry=spherical model.dimension=32 supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```

Run: `wc -l logs/selection_log.jsonl`
Expected: FIRST + 120.

- [ ] **Step 3: The records**

Write this script to `/tmp/naics_plan12_records.py` with the Write tool:

```python
'''Plan 12, Task 13: each arm record's seeds, selected epochs and three decision statistics.'''

import sys
from pathlib import Path

from naics_embedder.decision.records import ArmRecord, read_record

for path in sys.argv[1:]:
    arm = read_record(Path(path), ArmRecord)
    print(arm.spec.name, arm.spec.geometry, arm.spec.dimension, len(arm.runs), 'seeds')
    for run in sorted(arm.runs, key=lambda run: run.seed):
        statistics = {key: round(value, 6) for key, value in sorted(run.statistics.items())}
        print(' ', run.seed, run.checkpoint_epoch, statistics)
```

Run:

```bash
uv run --locked --no-sync python /tmp/naics_plan12_records.py /Users/lowell/naics-artifacts/records/stage8/euclidean-d16.json /Users/lowell/naics-artifacts/records/stage8/spherical-d16.json /Users/lowell/naics-artifacts/records/stage8/hyperbolic-d8.json /Users/lowell/naics-artifacts/records/stage8/hyperbolic-d32.json /Users/lowell/naics-artifacts/records/stage8/euclidean-d8.json /Users/lowell/naics-artifacts/records/stage8/euclidean-d32.json /Users/lowell/naics-artifacts/records/stage8/spherical-d8.json /Users/lowell/naics-artifacts/records/stage8/spherical-d32.json /Users/lowell/naics-artifacts/records/stage7/reference.json
```
Expected: nine blocks. Each Stage 8 arm shows `<arm> <g> <d> 5 seeds`, then one line per seed: its
selected epoch and its three statistics, `outcome`, `regressor_heldout` and `regressor_seen`. Each
seed's epoch is the one Task 11 or 12 recorded. The reference block repeats Stage 7's ten seeds.

Run: `shasum -a 256 /Users/lowell/naics-artifacts/records/stage8/*.json`
Record the eight hashes for the finding.

Run: `rm /tmp/naics_plan12_records.py`

### Task 14: Radius, inert terms, the logs and diagnostics

**Files:** none committed. Outputs: each run's `radius_report.json` and `diagnostics.json`, beside
its checkpoints.

**Interfaces:**
- Consumes: `tools radius-report` (Task 6), `tools diagnostics` (Stage 4), and the runs and
  records of Tasks 11–13.
- Produces: every run's "Radius" and "No inert terms" numbers and its diagnostics, for the
  finding.

- [ ] **Step 1: A radius report for every run**

For each of the 40 runs, with `<kkk>` its selected epoch in three digits (`003` for epoch 3, from
Task 13 Step 3), run:

```bash
HF_HUB_OFFLINE=1 uv run --locked --no-sync naics-embedder tools radius-report --checkpoint 'checkpoints/stage8-<g>-d<d>-s<s>/epoch=<kkk>.ckpt' --table 'checkpoints/stage8-<g>-d<d>-s<s>/arm_table_epoch=<kkk>.parquet' --output checkpoints/stage8-<g>-d<d>-s<s>/radius_report.json supervision.manifest_path=data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json
```
Expected: exit 0 for every run. The report reads the model's geometry and dimension from the
checkpoint itself, so it needs no geometry override.
- **A hyperbolic run (d = 8 or 32)** passes every Stage 7 check:
  - ∂L/∂r_a nonzero for every anchor of the real batch, and the SD of r above 10⁻³ at every level;
  - the 20 sector radii positive and pairwise distinct;
  - at the largest radius, the manifold residual and the float32 relative error within tolerance;
  - all five terms and scales with nonzero gradient.
- **A flat run** has `radius` null and passes on its four terms and scales (P11).

**Stop and ask if `tools radius-report` exits 1, raises, or reports a failed check. Never change a
tolerance (the resume ruling).** The report records its failed criterion; the user rules on it.

- [ ] **Step 2: Every report, together**

Write this script to `/tmp/naics_plan12_reports.py` with the Write tool:

```python
'''Plan 12, Task 14: each run's radius report passes, the hyperbolic arm's radius checks with it.'''

import json
import sys
from pathlib import Path

RUN = 'checkpoints/stage8-{geometry}-d{dimension}-s{seed}'
CELLS = [
    ('euclidean', 16), ('spherical', 16), ('hyperbolic', 8), ('hyperbolic', 32), ('euclidean', 8),
    ('euclidean', 32), ('spherical', 8), ('spherical', 32)
]

if sys.argv[1] == 'all':
    runs = [(geometry, dimension, seed) for geometry, dimension in CELLS for seed in range(1, 6)]
else:
    runs = [(sys.argv[1], int(sys.argv[2]), int(seed)) for seed in sys.argv[3:]]
for geometry, dimension, seed in runs:
    directory = Path(RUN.format(geometry=geometry, dimension=dimension, seed=seed))
    report = json.loads((directory / 'radius_report.json').read_text())
    assert report.get('geometry', 'hyperbolic') == geometry, (directory, report.get('geometry'))
    assert report['passed'] and not report['inert_terms'], (directory, report['inert_terms'])
    # Verification "Radius" is the hyperbolic arm's; a flat arm's report has none (P11)
    radius = report['radius']
    assert (radius is None) == (geometry != 'hyperbolic'), directory
    norms = {name: round(value, 6) for name, value in sorted(report['term_gradients'].items())}
    line = f'{geometry}-d{dimension} s{seed}: norms {norms}'
    if radius is not None:
        least = min(abs(value) for value in radius['anchor_radius_gradient'])
        line += (
            f"; least anchor gradient {least:.6g}, least level SD "
            f"{min(radius['level_sd'].values()):.6g}, sector gap {radius['sector_min_gap']:.6g}, "
            f"largest radius {radius['max_radius']:.6g}, manifold residual "
            f"{radius['manifold_error']:.6g}, max relative error {radius['max_relative_error']:.6g}"
        )
    print(line)
```

Run: `uv run --locked --no-sync python /tmp/naics_plan12_reports.py all`
Expected: 40 lines and no assertion error. Each line gives the run's gradient norms. A hyperbolic
run's line adds its least anchor gradient, least level SD, sector gap, largest radius, manifold
residual and largest relative error. At plan time the script reproduced the Stage 7 finding's
radius table from Stage 7's reports.

- [ ] **Step 3: Every campaign record is a validation read**

Write this script to `/tmp/naics_plan12_logs.py` with the Write tool:

```python
'''Plan 12, Task 14: every campaign log record is a validation read, and none opens a split.'''

import json
from pathlib import Path

RUN = 'checkpoints/stage8-{geometry}-d{dimension}-s{seed}'
CELLS = [
    ('euclidean', 16), ('spherical', 16), ('hyperbolic', 8), ('hyperbolic', 32), ('euclidean', 8),
    ('euclidean', 32), ('spherical', 8), ('spherical', 32)
]
SEEDS = range(1, 6)

paths = [Path('logs/selection_log.jsonl'), *Path('logs/remote').glob('*/selection_log.jsonl')]
records = [json.loads(line) for path in paths for line in path.open() if line.strip()]
monitor = [
    json.loads(line)['read']
    for geometry, dimension in CELLS for seed in SEEDS
    for line in (Path(RUN.format(geometry=geometry, dimension=dimension, seed=seed)) /
                 'monitor_reads.jsonl').open() if line.strip()
]
assert all(record['event'] == 'read' for record in records + monitor), 'a split was opened'
assert all(record['split'] == 'validation' for record in records + monitor), 'a test split was read'
print('log records', len(records), 'monitor records', len(monitor))
```

Run: `uv run --locked --no-sync python /tmp/naics_plan12_logs.py`
Expected: no assertion error, then the two counts. The log count is physical: the Mac log's lines
plus every instance log's, and an instance's log carries its earlier sessions' reads too. The
monitor count is the 40 runs' epochs summed.

- [ ] **Step 4: Diagnostics, for the record only (Req 6)**

For each run, with `<kkk>` as in Step 1:

```bash
uv run --locked --no-sync naics-embedder tools diagnostics --table 'checkpoints/stage8-<g>-d<d>-s<s>/arm_table_epoch=<kkk>.parquet' --geometry <g> --codebook data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet --output checkpoints/stage8-<g>-d<d>-s<s>/diagnostics.json
```
Expected: the report written. Nothing selects on it, and it is compared with nothing.

Run: `rm /tmp/naics_plan12_reports.py /tmp/naics_plan12_logs.py`

Run: `shasum -a 256 uv.lock`
Expected: Task 10's hash.

### Task 15: The nine-cell decision

**Files:** none committed. Output: `~/naics-artifacts/records/stage8/decision.json`.

**Interfaces:**
- Consumes: the eight Stage 8 records (Task 13), Stage 7's `reference.json` and `margins.json`,
  and `tools decide` (Stage 4) under Task 5's guards.
- Produces: the decision record. It names every comparison, the non-dominated set, the tie order,
  each arm's held-out gain and the chosen cell, which is Stage 9's reference.

`tools decide` runs `check_arm` on every arm, with each logged read's distance and dimension
(P8), then `check_pairing`, `check_margins` and `check_margins_first`. It resamples the stored
per-unit values; it reads no panel and logs nothing. Each A-against-B comparison adopts A when A
is non-inferior on all three panels and superior on at least one (D8). The tie order then ranks
the arms no other arm is adopted over: fewer components, then lower dimension, then
non-hyperbolic geometry, then the higher held-out gain (D11).

- [ ] **Step 1: Decide**

Run: `wc -l logs/selection_log.jsonl`
Record the count.

Run:

```bash
uv run --locked --no-sync naics-embedder tools decide --arm /Users/lowell/naics-artifacts/records/stage8/euclidean-d16.json --arm /Users/lowell/naics-artifacts/records/stage8/spherical-d16.json --arm /Users/lowell/naics-artifacts/records/stage8/hyperbolic-d8.json --arm /Users/lowell/naics-artifacts/records/stage8/hyperbolic-d32.json --arm /Users/lowell/naics-artifacts/records/stage8/euclidean-d8.json --arm /Users/lowell/naics-artifacts/records/stage8/euclidean-d32.json --arm /Users/lowell/naics-artifacts/records/stage8/spherical-d8.json --arm /Users/lowell/naics-artifacts/records/stage8/spherical-d32.json --arm /Users/lowell/naics-artifacts/records/stage7/reference.json --margins /Users/lowell/naics-artifacts/records/stage7/margins.json --name stage8-geometry-by-dimension --question 'Which geometry and dimension should the text stage use (Req 12)?' --store /Users/lowell/naics-artifacts --output /Users/lowell/naics-artifacts/records/stage8/decision.json
```
Expected:
- `Decision 'stage8-geometry-by-dimension'`, then 72 comparisons, every ordered pair of the nine
  arms, each `adopted` or `not adopted` with its three panel lines;
- then `Non-dominated: …`, `Tie order: …`, `Chosen: …` and `Decision record: …`.

**Stop and ask** if it prints `Decision failed:`. In particular:
- On `TieUnresolvedError`, two survivors tie on every key. That is the `tie_order` item's trigger
  (**Global Constraints**). Never change the rule during the decision.
- A failed guard (a distance, a dimension, the pairing or the margins' time) names the arm and
  seed it refused.

Run: `wc -l logs/selection_log.jsonl`
Expected: Step 1's count; the decision logged nothing.

- [ ] **Step 2: The decision, summarized**

Write this script to `/tmp/naics_plan12_decision.py` with the Write tool:

```python
'''Plan 12, Task 15: the decision's adoptions, non-dominated set, tie order and chosen cell.'''

import sys
from pathlib import Path

from naics_embedder.decision.records import DecisionRecord, read_record

record = read_record(Path(sys.argv[1]), DecisionRecord)
names = [arm.spec.name for arm in record.arms]
print('arms', names)
adopted = {(comparison.a, comparison.b) for comparison in record.comparisons if comparison.adopted}
for name in names:
    print(' ', name, 'adopted over', sorted(other for other in names if (name, other) in adopted))
print('non-dominated', record.non_dominated, 'cycle', record.cycle)
print('tie order', record.tie_order)
print('held-out gain', {name: round(gain, 6) for name, gain in record.heldout_gain.items()})
print('chosen', record.chosen)
```

Run:

```bash
uv run --locked --no-sync python /tmp/naics_plan12_decision.py /Users/lowell/naics-artifacts/records/stage8/decision.json
```
Expected:
- the nine arms, and for each the arms it is adopted over;
- the non-dominated set and whether dominance cycled;
- the tie order over that set and each arm's held-out gain;
- the chosen cell, the tie order's first.

Run: `shasum -a 256 /Users/lowell/naics-artifacts/records/stage8/decision.json`
Record it for the finding.

Run: `rm /tmp/naics_plan12_decision.py`

### Task 16: The finding

**Files:**
- Create: `specs/findings/geometry-by-dimension.md`

**Interfaces:**
- Consumes: the outputs of Tasks 10–15.
- Produces: the finding, which Plan completion's roadmap stamp cites.

- [ ] **Step 1: The receipts, before branching**

The receipts come from the main checkout's gitignored files, so make them before switching
branches.

Write this script to `/tmp/naics_plan12_receipts.py` with the Write tool:

````python
'''Plan 12, Task 16: the finding's selection-log receipts, verbatim, as Markdown sections (Req 4).'''

import sys
from pathlib import Path

RUN = 'checkpoints/stage8-{geometry}-d{dimension}-s{seed}'
CELLS = [
    ('euclidean', 16), ('spherical', 16), ('hyperbolic', 8), ('hyperbolic', 32), ('euclidean', 8),
    ('euclidean', 32), ('spherical', 8), ('spherical', 32)
]
SEEDS = range(1, 6)

# The Mac log's line count before Task 13's first sweep
first = int(sys.argv[1])
monitor = 0
for geometry, dimension in CELLS:
    for seed in SEEDS:
        path = Path(RUN.format(geometry=geometry, dimension=dimension, seed=seed))
        lines = [line for line in (path / 'monitor_reads.jsonl').read_text().splitlines() if line]
        monitor += len(lines)
        print(f'### {geometry}-d{dimension} seed {seed} monitor records\n\n```json')
        print('\n'.join(lines))
        print('```\n')
sweeps = [line for line in Path('logs/selection_log.jsonl').read_text().splitlines()[first:] if line]
print(f'### The {len(sweeps)} Mac sweep reads\n\n```json')
print('\n'.join(sweeps))
print('```')
print(f'monitor records {monitor}, sweep reads {len(sweeps)}', file=sys.stderr)
````

Run, with FIRST Task 13 Step 1's count:

```bash
uv run --locked --no-sync python /tmp/naics_plan12_receipts.py FIRST > /tmp/naics_plan12_receipts.md
```
Expected on stderr: `monitor records <n>, sweep reads 120`, where n is Task 14 Step 3's monitor
count. At plan time the script reproduced the Stage 7 finding's 256 verbatim lines from Stage 7's
files.

- [ ] **Step 2: Branch for the second PR**

Run: `git status --short`
Expected: no output.

Run: `git checkout --no-track -b claude/stage-8-campaign origin/main`
Expected: a clean switch. The private commits stay on `main`; this branch carries only the finding
and the completion markup. Its `conf/` is public, so run no campaign command on it.

- [ ] **Step 3: Write the finding**

Create `specs/findings/geometry-by-dimension.md` in the format of
`specs/findings/reference-configuration.md`. Wrap prose at 100 columns. Copy every value from
Tasks 10–15's output, and never estimate one; leave no value unfilled. Its sections:

- **Title and status.** `# Geometry × dimension: finding`, then
  `**Status: FINAL (<date>).** Roadmap Stage 8 (`specs/naics-embedding-roadmap.md`).` Say:
  - the eight new cells trained under Stage 7's settings, with only `model.geometry` and
    `model.dimension` changed;
  - the hyperbolic d = 16 cell is Stage 7's reference record, with its 10 seeds;
  - no sealed split was opened.
- **`## Sources`.**
  - The bundle `301cce28-539c-42ea-8781-496bbdcf511c` and the descriptions (`fe8c54e3…`).
  - Plan 9's text-only table, reduced to each arm's dimension at read time (D9).
  - The backbone at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
  - `uv.lock`'s sha256, and the torch and Lightning versions (Task 10 Step 4).
  - The source: BASE and SOURCE (Task 10), the six private commits' SHAs on SOURCE, and the
    statement that it is private, as Stage 7's was, with no claim of public native readiness.
  - The platform: Lambda, CUDA, `bf16-mixed` for training; the Mac for every decision read.
  - Stage 7's margins: δ per panel from `margins.json`.
- **`## 1. The runs`.** A table, one row per run, in the cells table's order: the arm, seed,
  instance and GPU, session and segment, epochs trained, selected epoch, start, end and seconds.
  Then a second table: each run's training-run ID, best monitor MRR and selected checkpoint's
  sha256.
- **`## 2. The arms`.** A table, one row per seed of each arm, with its three decision
  statistics: outcome MRR, and each regressor regime's `covariates+embedding` MSE. Include the
  reference's ten seeds from `reference.json`. Then each Stage 8 record's path and sha256.
- **`## 3. The decision`.**
  - The record's path and sha256, its question, and its settings (replicates, bootstrap seed,
    minimum seeds, the two interval levels).
  - A 9 × 9 adoption matrix, rows A and columns B, with `adopted` or a blank.
  - For every adopted comparison, each panel's Δ, its 95 % and 98⅓ % intervals, and −δ.
  - The non-dominated set, whether dominance cycled, the tie order, each arm's held-out gain and
    the chosen cell.
  - One paragraph reading the result against Req 12's question: whether a hyperbolic advantage
    appears at low dimension. Report what the record shows, without a claim beyond it.
- **`## 4. Radius and inert terms`.** Task 14 Step 2's 40 lines as a table: every run's gradient
  norms, and each hyperbolic run's radius columns as Stage 7's finding gives them. Say that the
  flat arms have no radial term or radius check (P11), and that every run passed.
- **`## 5. The selection log`.** Task 14 Step 3's two counts, each log's physical record count and
  sha256, and `/tmp/naics_plan12_receipts.md` pasted whole. The logs are gitignored, so these
  lines are their committed record. Do not sample them.
- **`## 6. Diagnostics, for the record (Req 6)`.** One row per run with the report's summary
  statistics: sector AUC, the two rank correlations, ancestor MAP, NDCG@5/10/20, distance Pearson
  and parent retrieval@1/5. Then each report's sha256. The full reports stay beside each run; say
  that nothing selects on them and that they have no target values.
- **`## 7. What later stages read`.**
  - Stage 9: the chosen cell, its geometry and dimension, as the reference for the backbone and
    one-factor ablations; the decision record's path; Stage 7's margins, unchanged; the lock,
    frozen at Task 10's hash.
  - Stages 10–11, if the chosen cell is flat: the HGCN feeder refuses a flat arm (P10), so the
    graph stage's next resume must say how its arms read a flat text arm.
  - Stage 12: the monitor reads travel in `SeedRun.monitor_records`.
- **`## Reproduction`.** Tasks 11–15's commands in a `bash` fence, one per line. Say that CUDA
  training is not bitwise reproducible, so a rerun's checkpoints, statistics and decision may
  differ.

Run:

```bash
python3 /tmp/naics_plan12_check_lines.py origin/main specs/findings/geometry-by-dimension.md
```
Expected: `[]`. If `/tmp/naics_plan12_check_lines.py` is gone, write it again from Task 8 Step 4.

- [ ] **Step 4: Commit the finding**

Run: `git status --short`
Expected: `?? specs/findings/geometry-by-dimension.md`, and nothing else that is tracked or
unignored. Never add `outputs/`, `checkpoints/` or `logs/`.

```bash
git add specs/findings/geometry-by-dimension.md
git commit -m "docs(findings): geometry × dimension, the nine-cell decision (roadmap Stage 8)"
```

Run:

```bash
rm /tmp/naics_plan12_receipts.py /tmp/naics_plan12_receipts.md /tmp/naics_plan12_check_lines.py
```

## Final verification, Phase 2 (controller, inline)

- [ ] **Step 1: The exit criteria**

Check each against the roadmap's Exit and Verification "Geometry × dimension", citing the output
that shows it:
- all nine cells have at least 5 seeds scored on D8's three panels (Task 13 Step 3);
- the decision record names the non-dominated set and the chosen cell under the tie order (Task 15
  Step 2);
- each arm's export uses tangent coordinates for hyperbolic and raw coordinates otherwise. Each
  table's provenance names its `geometry` and `coordinates` (P9), and Task 13's sweeps passed
  `check_seed_table` and `check_seed_distance`;
- every hyperbolic run passed `tools radius-report`, and every run passed "No inert terms" (Task 14
  Step 2);
- the campaign's log records are validation reads only (Task 14 Step 3).

- [ ] **Step 2: The lock**

Run: `shasum -a 256 uv.lock`
Expected: Task 10's hash. It stays frozen until the last Stage 8–10 decision that uses Stage 7's
margins.

- [ ] **Step 3: Review the finding**

Dispatch the code-reviewer agent on `git diff origin/main...HEAD`, with this plan, Req 12, Req 5
and the roadmap's Stage 8 entry. Fix each finding in the finding, or triage it as deferred for
Plan completion's gate.

## Plan completion (controller, inline)

Run this after Final verification, Phase 2, once the finding's review findings are resolved, and
before finishing-a-development-branch. It is writing-plans' Plan Completion Protocol, with this
plan's edits written out. It runs on `claude/stage-8-campaign` (Task 16 Step 2).

- [ ] **Step 1: Check for parallel sessions**

`specs/naics-embedding-roadmap.md` and `specs/deferred_items.md` are shared by every session.

Run: `git worktree list`
- Worktrees under `~/Projects/copilot-worktrees/` are review tooling, not sessions.
- If another worktree belongs to a running session, or the user has mentioned one, hold the
  shared edits (Steps 4 and 5) and give the user the exact text of each.

- [ ] **Step 2: The resolve-before-defer gate**

Collect the leftovers of both phases: plan steps skipped or descoped, and review findings not
fixed. Partition them:
- those that need the user's input: ask now, as one batched set of questions;
- those an answer unblocks: implement them now, then restart this protocol;
- everything else: defer.

The partition is final only once every question is answered.

- [ ] **Step 3: Mark up this plan**

- Tick every completed step (`- [x]`).
- Under each step that deviated, add a one-line `> Deviation: …` note.
- Annotate each skipped step with `> Skipped: <why> → deferred`.
- Add the status line under the title: `**Status: COMPLETE (<date>)** — executed via <skill>;
  deferred items in specs/deferred_items.md`, or `… ; nothing deferred` when the gate deferred
  nothing.

- [ ] **Step 4: The deferred items**

In `specs/deferred_items.md`:
- Annotate the `tie_order` item with this plan's outcome. Its Revisit condition names Stage 8.
  Write `→ plan 12: the nine-arm decision raised no TieUnresolvedError; the rule stays strict`,
  or, if it did raise, the user's ruling.
- Annotate plan 8's export-branches item: `→ partly done in plan 12: each geometry's coordinates
  are checked in the provenance (Task 4); generated_at, the exact key set and the other branches
  stay open`.
- Append this plan's deferred items, if the gate deferred any, as a `## 12-geometry-by-dimension —
  <date>` section in the file's item schema. Skip the section if nothing was deferred.

- [ ] **Step 5: The roadmap stamp**

In `specs/naics-embedding-roadmap.md`, Stage 8's entry:
- tick it: `- [ ] Stage 8: Geometry × dimension` becomes `- [x] Stage 8: Geometry × dimension`;
- add `Realized:` lines after `ROUTING: writing-plans`, as Stage 7's entry has them. Name the
  nine-cell decision record and its sha256, the non-dominated set and the chosen cell, and the
  finding `specs/findings/geometry-by-dimension.md`;
- add a `Source boundary:` line: training and reads used private local source SOURCE, as Stage 7's
  did, with no claim of public native readiness;
- end with the stamp:

```text
      Stage 8: COMPLETE (<date>) — implemented by plan 12
      (specs/plans/completed/12-geometry-by-dimension.md). Next: resume the roadmap.
```

The stamp asks completion to re-validate later stages. Hand the re-validation notes to the next
roadmap resume rather than editing later stages here, as Stage 7's completion did. Note especially
Stage 9's reference cell, and, if the chosen cell is flat, the HGCN feeder's refusal (P10) for
Stages 10–11.

- [ ] **Step 6: Backlog triage**

Run:

```bash
uv run --no-project --python 3.13 python /Users/lowell/.claude/skills/writing-plans/scripts/deferred_stats.py
```
Report its summary line: the open count, the closure rate and the aged tail. With 20 or more open
items or any aged tail, run steps 1–4 of the skill's Triage rubric and present the proposal. It
is read-only; `/deferred` acts on a selection.

- [ ] **Step 7: Retire the plan**

Run:

```bash
git mv specs/plans/12-geometry-by-dimension.md specs/plans/completed/12-geometry-by-dimension.md
```

Re-point the plan's relative links for its new depth, if it has any. The spec stays: this plan
implements a roadmap stage of `specs/naics-embedding.md`, which later stages still use, and there
is no Stage 8 spec to retire.

```bash
git add specs/plans/completed/12-geometry-by-dimension.md specs/deferred_items.md specs/naics-embedding-roadmap.md
git commit -m "chore(specs): retire plan 12"
```

Then push with the user's go-ahead (`git push -u origin claude/stage-8-campaign`), open the second
PR, and hand off to finishing-a-development-branch. Switch the main checkout back to `main`
afterwards, so the next campaign starts from the private source.
