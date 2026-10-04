# NAICS embedding — Roadmap

> For agentic workers: REQUIRED SKILL: derive-roadmap — resume via its
> reconcile step; route each unticked stage per its ROUTING line; never plan
> this document wholesale.

**Status: APPROVED (2026-09-23); resumed three times on 2026-09-24, once on 2026-10-03 and twice
on 2026-10-04.** Derived in a session that could not ask questions, then the six open questions
were answered interactively and the eleven-stage partition approved at the human checkpoint
(decisions D1–D6 below). Stages 1–6 and 6b are complete. The first resume re-validated Stages 2–11
against Stage 1 and recorded D7 and D8; the second re-validated Stages 3–11 against Stage 2,
recorded D9 and added Stage 12 (below); the third re-validated Stages 4–12 against Stage 3 and
recorded D10 and D11, which settle the former Open questions. Stage 4's completion commit
(9c028ae, 2026-09-25) re-validated the entries of Stages 5–12 against what Stage 4 shipped, and
Stage 5's (768a2b0, 2026-09-26) those of Stages 6, 7, 9 and 10; the fourth resume re-validated
Stages 6–12 against both (below). Stage 6b was added on 2026-10-03 during Stage 6's brainstorm
(below). Stage 6's completion commit (08e0fdc, 2026-10-03) re-validated Stages 6b–10, and the
fifth resume re-validated Stages 11 and 12 and the Gap analysis (below). Stage 6b's completion
commit (1d2240c, 2026-10-04) re-validated Stages 7–12, and the sixth resume re-read the Gap
analysis and routed plan 9's deferred entries (below). Stage 7 is next, per its ROUTING line.

**Basis.** Source spec `specs/naics-embedding.md` at d9126ce, unchanged through origin/main
8057916. Evidence was read at origin/main 620bee2 plus the two held, never-pushed config commits
(`config` and `graph config`) that point `conf/config.yaml` and `conf/graph.yaml` at bundle
18403d29; every sync rewrites their SHAs, so they are named by subject. 620bee2..8057916 changes
only `tests/unit/test_hgcn_metrics.py`. Paths below are
under `src/naics_embedder/` unless they start with `conf/`, `tests/`, `specs/` or `outputs/`.
Spec staleness since d9126ce: PR #102 makes the Staleness line "The graph stage's configuration
still names no bundle" true only of origin's shipped `conf/graph.yaml`; PR #100 (downstream
Lorentz distances in float64) is absent from the Staleness list; PRs #104 and #105 mean every
config key a stage adds must be declared in the Pydantic models. The separate
`specs/lambda-remote-workflow.md` (APPROVED) is not a stage here; Stages 7–10 are multi-seed
campaigns that benefit from it but do not require it.

**Resume after Stage 1 (2026-09-24).** Stage 1's stamp is authoritative: plan 3, merged as PR #108
(fa0cfc1). Stages 2–11 were re-validated at origin/main fa0cfc1. Since the evidence base (620bee2)
nothing under `src/` or `conf/` has changed, so every Gap analysis row stands: PR #106 added two
lines to one test, and PR #108 added `scripts/employment_statistics_coverage.py`, its tests and
the finding. Stage 1 shipped more than its Produces named: the script came with the finding, and
Stage 3 now lists it under Consumes as prior art. Stage 3 takes the finding's grain (years, not
areas) and D7; Stages 4, 7, 8 and 10 take D8. Stages 2, 5, 6, 9 and 11 needed no edit: none
consumes Stage 1, and D8 reaches Stages 9 and 11 only through Stage 4's tooling. D4 now says
where Stage 2, which has no stage spec, records its fractions.

**Resume after Stage 2 (2026-09-24).** Stage 2's stamp is authoritative: plan 4, merged as PR #110
(a688c7a). Stages 3–11 were re-validated at origin/main a688c7a. Unlike Stage 1, Stage 2 changed
`src/`: it added the `panels` package and edited `data/download_data.py`,
`data/supervision_bundle.py`, `supervision/artifacts.py`, `supervision/schema.py`,
`utils/config.py`, `cli/commands/data.py`, `cli/commands/tools.py` and
`conf/data/supervision.yaml`. Gap analysis citations in those files were re-read at a688c7a and
re-pointed in the rows that unticked stages read (Reqs 8, 9, 10 and 13; three in-code rows).
Every verdict stands, and the Req 3 row stays as the entry-time snapshot. Stage 2's finding
(`specs/findings/outcome-panel-splits.md`, section 6) names the interfaces later stages read, and
Stages 3, 4 and 6–8 now cite them. Five gaps surfaced; the review of PR #111 sharpened the third
and added the fifth:

- No stage opened the sealed test splits (Req 4's second bullet). The finding credited Stage 7,
  whose Exit reads validation only. Stage 12 was added with the user's approval, and the
  finding carries an erratum.
- The selection log is gitignored, so Stage 4's decision records now carry its records to
  Stage 12.
- A split's seal holds only while its draw is committed under a fingerprint and it is read only
  through a logged opening; `SelectionLog` records reads but gates none. Stage 3's outer sets
  inherit both rules.
- Stage 5's redirection table must keep Stage 2's leakage guarantee.
- Stage 12 scores every arm in the final configuration's recorded comparisons on sealed data,
  which needs each arm's per-seed encoder checkpoint and 2,125-code table. Stage 4's records now
  reference them, its driver keeps them until Stage 12, and Stage 11's drop path keeps arm D's.

D9 settles Stage 3's text-only comparator, and each panel's decision statistic joins Open
questions for Stage 4. Stage 10 needed no edit.

**Resume after Stage 3 (2026-09-24).** Stage 3's stamp is authoritative: plan 5, merged as PR #114
(2e2e226). Stages 4–12 were re-validated at origin/main 2e2e226, where the spec is still
d9126ce's. Stage 3's completion commit (d05e83f) had already re-validated Stages 4, 7 and 12; the
PR's later fixes pinned the held-out draw's hash in `conf/data/regressor_panel.yaml` (a53d6e4),
which Stage 12's one-opening rule relies on, and fixed a test. Under `src/`, Stage 3 added five
`panels` modules and `data/regressor_group_table.py`, and edited `cli/commands/data.py`,
`cli/commands/tools.py` and `utils/config.py`. Two Gap analysis citations into those files moved
and were re-pointed (the Req 13 row and the `verify-stage4` row); every verdict stands, and the
Req 2 row stays as the entry-time snapshot. Reading the shipped code surfaced five gaps; the
review of PR #115 sharpened the second and third:

- Stage 12 named `tools regressor-panel --split test` as an alternative to
  `RegressorPanel.open_outer`, but the command opens the outer sets on every call and scores one
  table, so a second arm or seed through it would be logged as a reopen, and no command opens the
  outcome test split. Stage 12 now scores every arm and seed through one panel object per sealed
  set.
- The panel reads no arm without its text-only table, which is not pinned across machines, and
  the selection log names both tables by `matrix_fingerprint`, not by file hash. Stage 4's
  artifact references now include each arm's text-only table with its provenance file, which
  records the backbone, the descriptions and the window that D9 and Req 9 constrain, and every
  table reference carries both identifiers.
- Loading the panel re-reads the four QCEW slices from `qcew_dir`, outside the repo, so every
  stage that loads it needs them: Stage 4's driver, Stage 6's check and Stage 12.
- The text-only builder reads at `max_length: 512`, the window Req 9 rejects, while D9 has the
  comparator read the same text as the arm. Stage 5's window policy now covers it (user
  approval), and Stage 9's Exit follows.
- Stage 3 left `metrics/qcew.py`, the definition Req 2 rejects, beside the new panel (plan 5:
  "Prior art left alone"). Stage 4 now removes it with the taxonomy-tasks suite (user approval).

Stage 6's export takes the panel's input contract: no Lorentz points and no constant column.
Stages 7 and 12 cite D10. Stages 8, 10 and 11 needed no edit: every arm exports as many columns
as its dimension, and arm D's references come through Stage 4's schema. D10 and D11 settle the
two Open questions that Stage 4's planning session was to ask; they were asked here instead,
because Stages 7–12 read them too and the roadmap is their only carrier.

**Resume after Stage 5 (2026-10-03).** Stage 5's stamp is authoritative: plan 7, merged as PR #119
(09a57bd). Stages 6–12 were re-validated at origin/main 09a57bd, where the spec is still
d9126ce's. The stage entries stood after the completion commits of Stages 4 and 5, apart from the
curvature below. Stages 8, 11 and 12 needed no edit: Stage 5 left the scorer's registered
distances and the graph stage alone, and bundle 301cce28's codebook keeps the hash Stage 12 pins
(`9b646af1…`; the comment beside it in `conf/data/regressor_panel.yaml` still names bundle
18403d29). Neither commit re-read the Gap analysis beyond one Req 8 citation, so this resume
re-read every citation in the rows unticked stages read and re-pointed each whose mechanism
survives, moved or reworded (Reqs 1, 4, 10, 11, 13, 15, 16 and 17; three in-code rows). Citations
into code that Stage 4 or 5 removed or rewrote stay as entry-time evidence of the gaps those
stages closed: the `verify-stage4` gate in the Req 1 and 5 rows; Req 7's 99 and half-step; Req
8's repeated exclusion text, generated exclusion negatives, quota slot, eligibility exemption and
denominator; Req 9's placeholder, inheritance, unary pairs and 512- and 24-token windows; and Req
10's training-pair count, bundle 18403d29's (301cce28 has 45,163,632). Decisions D2, D3 and D5
keep the citations they were recorded with, and the rows of Reqs 2, 3 and 6 and the five in-code
rows Stages 4 and 5 discharged stay as they were. Every verdict stands. Two gaps surfaced:

- The Stage 6 entry, following plan 4's deferred item, had the interim harness learn its
  curvature. Nothing learns it: `loss.curvature` (`conf/config.yaml:138`) is a fixed float that
  reaches `text_model/hyperbolic.py:98` through `text_model/naics_model.py:177`, as the Req 13
  row says, so at the shipped 1.0 the scorer's `lorentz` distance, which assumes curvature −1, is
  already exact. The entry now says so and leaves the scorer's distance to Stage 6's spec.
- Plans 6 and 7 handed over eleven open entries that no stage routed. The Deferred items
  paragraph now routes them.

**Stage 6b (2026-10-03).** During Stage 6's brainstorm the user ruled that channel texts beyond
the backbone's trained window are summarized, not truncated (Req 9's input windows), as a stage of
their own before Stage 7. Truncation falls hardest on the top levels: 16 of the 20 sector
descriptions and 59 of the 96 subsector ones overflow the 128-token window. The stage is inserted
as 6b, not renumbered, so Stage 6's stamp and every reference to Stages 7–12 stand. Stage 6's spec
(`specs/completed/shared-encoder-and-projection.md`, Rollout note) records the measurements and
the three constraints the entry carries.

**Resume after Stage 6 (2026-10-04).** Stage 6's stamp is authoritative: plan 8, merged as PR #122
(f5c8307). Its completion commit had re-validated Stages 6b–10; this resume re-validated Stages
11 and 12 at origin/main f5c8307, where the spec is still d9126ce's, and re-read every Gap
analysis citation into the 22 files Stage 6 changed under `src/` and `conf/`, in the rows unticked
stages read. Citations whose mechanism moved were re-pointed (Reqs 4, 10, 11, 13 and 16, and the
legacy-containment row), and those Stage 6 rewrote now say what replaced them: Req 11's
load-balancing term exists only under `model.fusion: moe`, Req 13's cap sits in the
parameter-free `HyperbolicHead`, Req 15's text export is the HGCN feeder's d + 1 columns, and Req
16's standalone export exists for the text stage. Citations into code Stage 6 deleted stay as
entry-time evidence: Req 9's `text_model/encoder.py:140`, where fusion now masks absent channels
(`text_model/shared_encoder.py:225-235`, `text_model/fusion.py:40-59`), and the `encoder.py`
citations in the rows of Reqs 3, 12 and 14, the Sequencing paragraph and Stage 2's Consumes. The
Stage 5 resume's curvature path now runs through `text_model/hyperbolic.py:101`. Every verdict
stands. Four gaps surfaced:

- The overflow the Stage 6b entry carried was measured on unmarked texts. The cache tokenizes
  each text with its field marker, so 162 descriptions, 106 examples texts and 485 exclusion
  texts are truncated, not 153, 105 and 464; the sectors (16 of 20) and subsectors (59 of 96) are
  the same either way. The entry now carries both.
- Stage 12 named no query path. Stage 6's is `ArmEncoder.from_files`, which reads only a table
  whose provenance names its checkpoint, its own hash and the window (`text_model/export.py`
  writes one), takes the tokenizer from the arm's config, and maps through the hyperbolic exp
  map only. Stage 10's arm tables now carry that provenance, and Stage 12 inherits Stage 8's
  per-geometry maps.
- Stage 11's drop path did not name the HGCN feeder, `generate_embeddings_from_checkpoint` and
  `train`'s prompt for it (`cli/commands/training.py`), whose `encode_token_rows` the export and
  the arm encoder keep using.
- The bundle build imports `graph_model.curriculum.preprocess_curriculum` for the
  `difficulty_thresholds` member (`data/supervision_bundle.py:694-697`) that the manifest
  requires (`supervision/artifacts.py:177`). Removing the graph stage therefore breaks
  `data supervision`, and dropping the member changes the bundle contract. Stage 11's drop path
  now names it as a contract decision.

**Resume after Stage 6b (2026-10-04).** Stage 6b's stamp is authoritative: plan 9, merged as PR
#123 (c1f9ee5). Its completion commit had re-validated the entries of Stages 7–12; this resume
re-read them at origin/main c1f9ee5, where the spec is still d9126ce's, and found no edit due. It
also re-read every Gap analysis citation into the 19 files Stage 6b changed under `src/` and
`conf/`, in the rows unticked stages read. Only the live citations into
`text_model/naics_model.py` and `cli/commands/training.py` moved, by three and six lines, where
the summaries' sha256 joined the checkpoint contract. They were re-pointed in the rows of Reqs 4,
10, 11, 15 and 16 and the legacy-containment row; the mechanisms they cite are unchanged. The
`tokenization_cache.py` and `supervision/schema.py` citations in the rows of Reqs 7 and 9 and the
title-window row stay as entry-time evidence, and D2 keeps the citation it was recorded with.
Every verdict stands. Req 9's input-window bullet asks only that inputs fit the window, so the
2026-10-03 ruling, which summarizes rather than truncates, meets it and the Completion audit
needs no deviation note. Stage 6b's spec calls Stage 7's panels the real test of its centrality
selection; under that ruling no stage compares summaries with truncation, so Stage 7's reads are
the first panel numbers on the summaries, not a test of them. No gap surfaced.

**Decisions (2026-09-23).** Six ambiguities the spec leaves open, answered by the user at the
checkpoint. Each fixes the named stage; the stage entries cite them.

- **D1 — Regressor panel's inherited definitions (Req 2; Stage 3).** The spec names neither the
  downstream model nor the covariates. Decision: ridge with a nested-cross-validated penalty;
  covariates are log establishment counts and log wages from the same QCEW rows, as in the
  methodology's definition (`metrics/qcew.py`).
- **D2 — Legacy containment mode (Stage 7).** `supervision.mode: legacy_containment`, its second
  `training_step` (`text_model/naics_model.py:618-693`) and the checkpoint-contract migration
  appear nowhere in the spec. Decision: Stage 7 deletes them with the old objective; legacy
  checkpoints cannot load into a 16-dimensional shared encoder anyway.
- **D3 — Arm D's level-radius term (Stage 10).** `losses/level_radius.py:26-29` via
  `graph_model/hgcn.py:744-749` pulls sectors to the origin (E3); Req 17 does not list it and
  Req 15 says arm D is repaired only to run. Decision: where the selected geometry leaves the
  term undefined it is omitted, mirroring Req 11(iii); in the hyperbolic case it is kept as-is,
  origin target included, and the decision record notes the handicap. Arm D gets only the Req 16
  fix.
- **D4 — Index-entry role proportions (Req 3; Stage 2).** The spec fixes one role per entry and
  stratified test entries, not the split among examples-channel text, training, validation and
  test queries, nor a floor for a code's examples channel. Decision: Stage 2's plan sets them, per
  code and stratified, with a floor of one examples-channel entry where a code has enough
  entries; the fractions are recorded in the stage's Rollout note. Stage 2 has no stage spec, so
  that note is a line under its roadmap entry, beside the stamp (resume, 2026-09-24).
- **D5 — Kinship relation taxonomy (Stages 5 and 7).** The bundle's 14 named relations plus
  `cross_sector` and their margin axis (`data/compute_relations.py:107-158`,
  `data/create_triplets.py:142-153`) carry no requirement. Decision: relation names survive only
  as arm D's edge types and as diagnostic labels; the margin axis leaves with the eligibility
  rules in Stage 7.
- **D6 — The within-run selection statistic (Req 4; Stage 7).** Once the in-sample loss selects
  nothing, checkpointing, early stopping and learning-rate control need a validation-split
  statistic the spec does not name. Decision: the validation query split's MRR (Req 3); both
  panels are used only between configurations, under Req 5.

**Decisions (2026-09-24).** Two ambiguities that Stage 1's finding made live, answered by the user
at the resume checkpoint.

- **D7 — The time-respecting outcome (Req 2; Stage 3).** The finding takes branch A, so the panel
  includes a time-respecting outcome; neither the spec nor D1 names its variable or says whether
  the same-year outcome stays. Decision: log annual-average employment in year t+1, from year-t
  features (the representation, plus D1's covariates from the year-t row); the same-year outcome
  is dropped. With 2022–2025 final, feature years run 2022–2024, and splits by time seal
  2024→2025 for the seen regime. That leaves the seen regime's repeated grouped inner folds two
  feature years; Stage 3's plan fits them to that.
- **D8 — Regressor regimes under Req 5 (Stages 4 and 7; every later decision).** Req 2 reports
  the seen-code and held-out-code regimes separately, but Req 5 counts two panels, each with its
  own δ and half the error rate, and ends its tie order on "the higher regressor-panel estimate".
  The finding runs both regimes. Decision: each regime counts as a panel under Req 5, which then
  has three: the outcome panel and the two regressor regimes. Adoption needs non-inferiority on
  all three (the 95 % interval, unchanged) and superiority on at least one; each panel gets its
  own δ and a third of the error rate, so superiority reads the 98⅓ % interval. This supersedes
  the 97.5 % that Req 5 and Verification "Decision records" name; the Completion audit reads it
  as this recorded deviation, not an unmet requirement. The final tie-break stayed open; D11
  settles it.

**Decision (2026-09-24, second resume).** One ambiguity Stage 3 cannot be planned without, answered
by the user at the second resume checkpoint.

- **D9 — Req 2's text-only comparator (Stage 3; Stages 8 and 9 re-run it).** Req 2 names no
  representation, and review C28, its source, lists two: a frozen encoder and TF-IDF. Decision:
  the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the arm's
  dimension. That backbone is the current checkpoint until Stage 9 adopts another, and whichever
  backbone the arm uses after. The comparator measures what taxonomy training adds over the same
  encoder reading the same text.

**Decisions (2026-09-24, third resume).** The two former Open questions, answered by the user at the
third resume checkpoint. Stage 4 builds both into its tooling, through which Stages 7–12 read them.

- **D10 — Each panel's decision statistic (Req 5; Stage 4; Stages 7–12 read it).** Req 5 fixes a
  δ and an interval per panel but names no statistic. Decision: the outcome panel's is per-query
  MRR, resampled by code with its queries, the statistic D6 already selects checkpoints on. Each
  regressor regime's is the out-of-sample mean squared error of the `covariates+embedding`
  comparator on log employment at level 6, each row's squared error averaged over its repeats
  (five on a validation read) before resampling by group; its Δ is oriented so that a positive
  value favours A (B's error minus A's). The sparse comparators never read the arm, and their
  folds depend only on the regime, level, repeat and group, so the gain over a sparse encoding
  (over one-hot in the seen regime, over ancestors in the held-out regime) gives the same paired
  Δ; that gain is reported for Req 1. The text-only comparator is no baseline, since it changes
  with each arm's dimension and backbone. Every other Req 2 comparator and Req 3 metric is still
  reported.
- **D11 — D8's final tie-break (Req 5; Stage 4).** Req 5's tie order ends on "the higher
  regressor-panel estimate", and D8 gives two. Decision: the held-out regime's, read as its gain
  over ancestors, so the higher estimate is the lower error. For a held-out code, one-hot
  predicts only the intercept and ancestors help only through levels 2–3, so that regime is where
  the embedding's value over sparse encodings is tested.

## Gap analysis

| Req | Verdict | Evidence | Note |
|---|---|---|---|
| 1 | missing | `text_model/mixins/validation.py:182` (ground truth is the taxonomy distance matrix), `:167-170` (300-code subsample); `utils/training.py:307` (checkpoint monitors `val/contrastive_loss`); `tools/embeddings_verification.py:33-35` (structural acceptance gate) | No evaluation reads data outside the taxonomy; no sealed split exists; structural statistics still gate (`verify-stage4`). Looked in `metrics/*`, `mixins/validation.py`, `tools/*`, `cli/commands/*`, `graph_model/hgcn.py:863-943`. |
| 2 | implemented-differently | `metrics/qcew.py:79` (one row per code), `:110`, `:136` (`Ridge(alpha=1.0)` on unscaled features), `:131-132` (2022, private only), `:185-187` (one `GroupShuffleSplit`), `:178` (covariates are QCEW's own establishments and wages), `:331` (multi-level loop); no CLI wiring in `cli/commands/*`; only run: `tests/unit/test_graph_downstream_evaluation.py:84` on a synthetic CSV | The module is the definition Req 2 rejects (one row per code, fixed penalty, no ancestor, text-only or covariates-only arms, no regimes) and has never run on real data. The (open) item is unresolved: Stage 1. Outcome data are external BLS QCEW files, not a repo artifact. |
| 3 | missing | `data/download_data.py:86`, `:397` (index file becomes the examples channel, joined with `'; '`); `text_model/encoder.py:122-123` (forward needs all four channels; no single-text path); `metrics/core.py:152` (`RetrievalMetrics` is code–code with binary relevance, called only from the never-instantiated `metrics/runner.py:110`) | No query encoding, decoding, roles or leakage check. Looked in `text_model/*`, `metrics/*`, `tools/*`, `cli/*`, `tests/unit/*`. |
| 4 | missing | `text_model/dataloader/datamodule.py:901` (validation is the same builder with seed + 1000), `:1114`, `:1117` (same `_repaired_dataset` for train and validation), `:865` (`val_split` unused); no `test_step` or `test_dataloader` in `src/`; `cli/commands/training.py:605`, `:739` (checkpoint and export from the best `val/contrastive_loss`); `graph_model/dataloader/hgcn_datamodule.py:212`, `:245-246` (5 % unshuffled tail); `graph_model/hgcn.py:1326`, `:1331` (no checkpointing; last-epoch export) | The in-sample contrastive loss selects everything (`datamodule.py:1157-1158` documents it); no test split; the graph stage has both forbidden behaviours. |
| 5 | missing | `conf/config.yaml:5`, `conf/graph.yaml:100` (single seed); no seeds, pairing, interval, dominance or record code in `src/`, `tests/`, `conf/`, `scripts/`; `tools/embeddings_verification.py:33-35` (the fixed thresholds Req 5 replaces) | Only the mechanism to be replaced exists. The one retained run holds three metric records (`outputs/sadc_default/version_0/evaluation_metrics.json`). |
| 6 | implemented-differently | `metrics/core.py:341` (`cophenetic_correlation`; Pearson at `:389`), `:508` (continuous NDCG grades); `metrics/structural_spearman.py:16` (global, unstratified); `metrics/hierarchy_structure.py:103-106` (parent retrieval over every pair, unary pairs included); `text_model/mixins/validation.py:406-408` (cophenetic on the progress bar); `tools/metrics_tools.py:104-106` (headline table); no sector AUC or within-sector statistic in `src/` | The statistics exist but are global, keep the cophenetic name, grade NDCG continuously, score unary pairs, and headline or gate. |
| 7 | implemented-differently | `data/compute_distances.py:160-164` (−0.5 on lineal pairs), `:244` (`fill_null(99)`); `supervision/schema.py:36`, `:41`; `tests/unit/test_data_distances.py:280`, `:492` pin both; no information-content code anywhere | The collateral term matches D*; the half-step and the 99 constant do not; no virtual root. The prior spec kept 99 deliberately (`specs/completed/stage-3-supervision-integrity.md:117-118`) and scoped out hierarchy changes (`:42-48`). 87.9 % of the bundle's pair facts are 99. |
| 8 | implemented-differently | `data/download_data.py:352`, `:366` (exclusion text exploded per referenced code, so repeated); `data/create_triplets.py:227-229` (exclusion pairs generated as `UNRELATED` negatives); `supervision/selection.py:184-192` (rotating one-slot exclusion quota); `text_model/mixins/curriculum.py:244` (exclusions exempt from eligibility); `text_model/loss.py:78` (always in the denominator); `supervision/index.py:102-104` (symmetric, so the nine lineal references act as negatives); `graph_model/hgcn.py:1157-1162` (edges from structural relations only) | (a) repeated, not once; (b) absent; (c) inverted for negatives (a protected negative was a deliverable of the prior spec, `:270-271`, `:503-505`) and as specified for edges; lineal references untreated. |
| 9 | implemented-differently | `text_model/dataloader/tokenization_cache.py:39-40` (`'[EMPTY]'` placeholder), `text_model/encoder.py:140` (all four channels concatenated, no presence mask); `data/download_data.py:556-560`, `:629` (`.unique` picks an arbitrary child for the 14 four-digit codes), `:567`, `:582-584` (a five-digit code copies its lone child); no provenance column; `data/positive_sampling.py:78-81` (five-digit anchors get their lone child as positive); `metrics/hierarchy_structure.py:103-106`; `conf/config.yaml:95` (`max_length: 512`), `tokenization_cache.py:74` (title fixed at 24) | Placeholder instead of masking; arbitrary inheritance unrecorded; unary pairs in supervision and scoring; the rejected 512 window. The backbone's own files disagree (`sentence_bert_config.json` 256, `tokenizer_config.json` 512), so the (open) item stands. |
| 10 | implemented-differently | `data/compute_distances.py:49-56`, `data/create_triplets.py:104`, `data/supervision_bundle.py:809` (canonical orientation: 35 six-digit codes are never anchors; no ancestor positives); `text_model/naics_model.py:544-547`, `datamodule.py:681-690`, `conf/config.yaml:99` (pools of 24 pre-drawn negatives per pair; manifest `training_pairs` has 45,373,918 rows); `supervision/margins.py:75` (eligibility rule); `text_model/dataloader/streaming_dataset.py:196-198`, `conf/config.yaml:101-102` (inverse-distance draws); `text_model/mixins/curriculum.py:453-464` (hyperbolic k-means pseudo-labels) | Every removed mechanism is present; the 4,675 sibling-positive, grandchild-negative rows reproduce in the live bundle; no per-epoch cache of all code points. |
| 11 | implemented-differently | `text_model/mixins/loss.py:485-494` (five-term sum; six under `model.fusion: moe`); `text_model/loss.py:47-129` (DCL), `:135-228` (distance matching, batch-normalized at `:222-223`), `:330-401` (pairwise preference); `mixins/loss.py:207-209` (radius penalty, zero by construction since r ≤ 2 < 10), `:318` (load balancing, under `moe` only); `losses/level_radius.py:26-29` (level term; gradient ≈ 1e-7 under the cap, by probe); `naics_model.py:588-589` (margin logged only); `loss.py:58` (fixed float temperature); `text_model/mixins/optimizer.py:165-174`, `text_model/curriculum.py:108-116` (scheduler always built; mining active in epochs 6–9 of the shipped 10); no query term in `src/` | No task term; no learned scales; two code–code terms, neither listwise over all codes; every term Req 11 removes is present. Verification "No inert terms" fails today. |
| 12 | missing | `text_model/encoder.py:42` (dimension is the backbone hidden size, 384), `:81`, `:146` (`moe_projection` 1536→384) then `text_model/hyperbolic.py:102` (384→385): two stacked affine maps; no geometry switch in `src/` or `conf/`; `tests/unit/test_encoder.py:106` pins 384 | Dimension not configurable; Lorentz only. |
| 13 | missing | `text_model/hyperbolic.py:99` (`max_norm=2.0` in the parameter-free `HyperbolicHead`), `:116-120` (hard rescale; saturated points pass ≈ 1e-7 gradient, by probe); `losses/level_radius.py:28` (level 2 targets sinh r = 0, the origin); three radial coordinates in use (`level_radius.py:26-27`, `text_model/hyperbolic.py:259`, `text_model/hard_negative_mining.py:60-62`); `utils/config.py:965`, `conf/config.yaml:141` (curvature is a config parameter threaded everywhere); `graph_model/hgcn.py:85-88`, `:106-107` (per-layer parameter detached by `.item()`); `text_model/hyperbolic.py:212-245` (manifold check at tolerance 1e-3; no radius-resolution check) | E1 and E3 premises confirmed by probe. |
| 14 | missing | `text_model/encoder.py:56-61` (four full backbone loads, each with its own LoRA), `:122-123`, `:140` (concatenation into the MoE); `text_model/moe.py:118-125`, `conf/config.yaml:124-128` (MoE is the only fusion); `conf/config.yaml:117` (one `base_model_name`; no candidate list, no freeze flag) | No shared encoder, field markers, masked fusion, query path or backbone selection. |
| 15 | missing | `graph_model/hgcn.py:1300`, `conf/graph.yaml:100` (single seed, one arm); no smoothing, shuffle control or matched-compute tooling in `src/`; `conf/graph.yaml:20` (`tangent_dim: 31`) against the HGCN feeder's d + 1 `hyp_e*` columns, 17 at the default d = 16 (`cli/commands/training.py:302-308`; at entry the export was 385 wide, which by probe `Linear(31, 31)` rejects; no width check at `hgcn.py:1303-1311`) | Arm D cannot run on the text stage's output as configured (the Req 16 fix); arms B, C and E do not exist; the fixed-threshold gate is the only comparison tooling. `conf/graph.yaml:15` names the bundle only in the unpushed local commit. |
| 16 | implemented-differently | `graph_model/hgcn.py:82` (`Linear(dim, dim)`: no learned projection, but width 31 ≠ 385); `:303` (node states are a free `nn.Parameter` seeded from the text points; no retention term); `:1230-1237` (graph export of all rows); `cli/commands/training.py:730-742` (the HGCN feeder only behind an interactive `typer.confirm`); `cli/commands/tools.py:854-908` (`tools export-table`, the text stage's standalone export, Stage 6) | No projection map, as specified, but no shared space either; per-stage tables exist, and the text stage's has a standalone export (Stage 6); no defined final deliverable. |
| 17 | implemented-differently | `conf/graph.yaml:49-53`, `graph_model/hgcn.py:1190`, `:1195-1202` (child, grandchild, great-grandchild and sibling edges, bidirectional, plus self-loops); `hgcn.py:135`, `:143` (edge weight enters twice), `:1182-1183`; `hgcn.py:84`, `:118-120` (tangent LayerNorm; output radius ≈ 5.4 whatever the input, by probe); no distillation or residual to the text points; `hgcn.py:249`, `conf/graph.yaml:38-39` (hinge temperature); `hgcn.py:670-707`, `graph.yaml:73` (adaptive margin on); `hgcn.py:181-182`, `:205-207`, `graph.yaml:24` (uncertainty weights on); `hgcn.py:518-569`, `:592-626`, `graph.yaml:64` (in-file three-phase curriculum filters on); `graph_model/curriculum/*` (four-phase controller package, unwired: `hgcn.py:23` imports only `resolve_graph_config`) | Every mechanism Req 17 would replace is present and on by default. Untouched until Req 15 decides (user adjudication Q2); not applicable before Stage 10. |
| (none) | in-code-but-not-in-spec | `text_model/naics_model.py:649-726`, `supervision/mode.py:35-40` | Legacy containment mode with a second `training_step`. Deleted in Stage 7 (D2). |
| (none) | in-code-but-not-in-spec | `data/supervision_bundle.py`, `supervision/checkpoints.py`, `conf/config.yaml:9-12` | The supervision bundle contract (manifest, contract version, exact-resume checkpoint contract). The spec's Staleness note assumes it without naming it; Stages 2 and 5 version it. Unrecorded decision to fold back: the bundle stays the single authority for structure and text. |
| (none) | in-code-but-not-in-spec | `data/compute_relations.py:109-160`, `conf/data/supervision.yaml:10-25`, `data/create_triplets.py:149-160` | Fourteen named kinship relations plus `cross_sector` as a second structural axis with its own margin. Names survive as edge types and labels only; the margin axis leaves in Stage 7 (D5). |
| (none) | in-code-but-not-in-spec | `losses/level_radius.py:26-29` via `graph_model/hgcn.py:741-746` | The graph stage's level-radius term (sectors at the origin). Req 17 does not list it. Handled in Stage 10 per D3. |
| (none) | in-code-but-not-in-spec | `tools/embeddings_verification.py:33-35`, `cli/commands/tools.py:278` | The `verify-stage4` gate with fixed thresholds; Req 5 replaces them (Stage 4). |
| (none) | in-code-but-not-in-spec | `metrics/graph.py:236-439` | Unwired "taxonomy tasks" suite (parent identification, k-means ARI and NMI, sector logistic regression): the methodology's other never-run benchmark. No requirement keeps it; retire with Stage 4's diagnostics. |
| (none) | in-code-but-not-in-spec | `graph_model/curriculum/*` (about 2,400 lines), `tests/unit/test_graph_curriculum.py` | Unwired four-phase controller, event bus, MACL, samplers and analyzer. Held until Req 15 (user adjudication Q2); leaves in Stage 11 either way. |
| (none) | in-code-but-not-in-spec | `text_model/mixins/validation.py:167-170`, `conf/config.yaml:130` | Structural statistics on a 300-code random subsample; Req 6 names no population. Stage 4 computes over all 2,125 codes (methodology S4 limitation 4). |
| (none) | in-code-but-not-in-spec | `data/download_data.py:291`, `:312-324` | 68 cross-reference rows without a code reference are dropped, and exclusion text is also harvested from "Excluded" description paragraphs; the spec counts 43 non-redirection rows (4,601 − 4,558). Stage 5 reconciles the two counts. |
| (none) | in-code-but-not-in-spec | `text_model/dataloader/tokenization_cache.py:74` | Title channel fixed at 24 tokens; Req 9's window policy covers it in Stage 5. |

## Stages

**Sequencing.** The order follows the spec's Rollout note (items 1–6) at finer grain: item 1 is
Stages 1–4, item 2 is Stages 5–7, item 3 is Stage 8, item 4 is Stage 9, item 5 is Stage 10 and
item 6 is Stage 11. Stage 12, added at the second resume, comes last: Req 4 opens each sealed test
split once, for a final configuration that exists only after Stage 11. Two deliberate departures.
First, Req 14's shared encoder is built in Stage 6, before the objective, rather than with the
Rollout's item 4: Req 5's reference configuration already names it, and nothing can score the
outcome panel until a query can be embedded (`text_model/encoder.py:122-123`). Second, the
graph-stage halves of Reqs 4 and 6 (private tail, last-epoch export, acceptance gate) move to
Stages 4 and 10 instead of item 1, because the stage is untouched until Req 15 runs (user
adjudication Q2); Stage 4 retires the gate's thresholds in the comparison tool, not in the stage.
Stage 1 is the investigation the Rollout note puts first; its finding fixes Stage 3's parameters,
so it stands alone although it produces no software. Stage 6b, added on 2026-10-03, fixes the text
every arm reads, so it lands before Stage 7 trains the reference configuration and fixes δ. Its
only contact with Stage 6 is the tokenization cache, so the two can be built in parallel, and
whichever lands second integrates with the other.

**Deferred items.** `specs/deferred_items.md` was read, not edited, at derivation. I4 (selection
coordinator cost), M9 and M10 (pseudo-label and provenance handling in the curriculum mixin) are
mooted by Stage 7, which deletes that machinery. M6 and M8 (legacy path fields; three commands
each building a bundle) fall to Stage 5, which reversions the bundle contract. The degenerate
graph-curriculum thresholds change with D* in Stage 5 and leave with Stage 11. Each is retired
through /deferred by the stage that discharges it. The resume added plan 3's one entry as that
plan handed it over: the national-total rounding allowance in
`scripts/employment_statistics_coverage.py`. No stage discharges it. It is a standalone
quick-fix, which Stage 3 inherits only if it reuses that script's checks. The second resume adds
plan 4's four entries, as that plan handed them over. Stage 5 discharges two: its contract
requires the three `index_roles_*` validation results, and its member pins the entry text under an
artifact hash. Stage 6 discharges one, a curvature-aware distance for the scorer's live reading.
The fourth is a standalone quick-fix: the near-duplicate threshold and examples floor are
hardcoded outside `data roles`. The third resume adds plan 5's four entries, as that plan handed
them over. One is done: plan 5 pinned the held-out draw's hash (a53d6e4). Stage 4 discharges two:
the note on the decision statistic (D10 averages each row's repeats before resampling, and the
plan also reports the held-out regime by feature year, or records why not), and the mismatch
between the log's `matrix_fingerprint` and the text-only provenance's `table_sha256`, since its
records match logged reads to table files. The fourth, caching the ridge path's SVD per fit row
set, is a standalone quick-fix, revisited if panel reads slow a seed sweep. Stage 3 did not reuse
the coverage script's checks, so plan 3's rounding fix stays standalone. The fourth resume finds
Stage 5's four discharged (M6, M8 and plan 4's two contract items) and adds plan 6's five open
entries and plan 7's six, as those plans handed them over; plan 6's other two closed with PR #118.
Stage 6 discharges one, beside plan 4's scorer item, whose premise the resume note above corrects:
the regressor scores' repeat count and the wording of the panel's refusal of Lorentz points, both
due when its export lands. One lands before Stage 7 trains on the activity phrases as queries: the
loader's check of each phrase's value. Stage 9 decides whether training refuses a backbone other
than the bundle's, and Stage 10 settles the diagnostics report's interfaces before its
keep-or-drop record reads the report. Four wait on a trigger: the tie order's strictness, if a tie
below first place blocks a decision (Stage 8's nine cells are the first with several arms); the
text model's Lorentz distances, exact only at c = 1, if a run leaves c = 1 or HGCN's layer
curvature gains a gradient (Stages 10–11); the manifest's tokenizer revision, at the next bundle
rebuild or contract bump; and the activity phrases a forced role redraw's leakage check skips, if
`data roles --force` runs. Three are standalone quick-fixes: the bundle build's tokenizer order
and its text-channel check, and the `tools` commands' handling of polars errors. The fifth resume
adds plan 8's seven entries, as that plan handed them over. Stage 7 discharges one: what one
training epoch reads, with its Lambda config's `data_loader.n_epochs` set to match. Three hold
parts that touch Stage 6b's edits, and its spec says which it takes: the export provenance's
missing tokenizer name, the cache's load messages, which name only the fingerprints, and the
missing test that changes only a sidecar's markers. One waits on a trigger: `_pool_present`
under true half precision, if a run can select it. The rest are standalone: the export's and the
arm encoder's untested branches and untidy failures, the encoder, fusion and cache test gaps, the
transformers floor and the remaining polish. Stage 6b does not rebuild the bundle, so plan 7's
tokenizer-revision entry keeps its trigger. The sixth resume finds Stage 6b's three parts of
plan 8's entries discharged (the export provenance's tokenizer name, the cache's load messages
and the markers-only sidecar test); the rest of those entries stays open. It adds plan 9's three
entries, as that plan handed them over. Two are standalone: the summaries' build and parser
leave untested and untidy branches, and their readers, tests and docs leave polish and coverage
gaps. One waits on a trigger: an export keys the summaries on its tokenizer's name and a
text-only table on its backbone's, and the sweep's per-seed check compares tokenizers only
through the summaries' sha256, which matters if Stage 9 admits a backbone whose tokenizer name
differs from its own.

- [x] Stage 1: Employment-statistics coverage
      Objective: Verify which public employment series publish NAICS 2022 six-digit cells, for
      which reference years and grains, how much suppression removes at each, and which Req 2
      branch the regressor panel therefore takes.
      Spec: Req 2 (the (open) item and its three pre-specified branches); Verification
      "Employment-statistics coverage"; Rollout note, first paragraph.
      Gap closed: Req 2 (the open item only).
      Consumes: Nothing from a prior stage. The bundle codebook (2,125 codes, 1,012 six-digit)
      as the code universe; `metrics/qcew.py` as prior art only (its 2022, private, one-row-
      per-code slice is the rejected definition).
      Produces: A written finding (`specs/findings/employment-statistics-coverage.md`; the
      default `reports/` path is gitignored) recording years, grains, the suppressed share per
      year, grain and series, the panel population and row grain, whether a time-respecting
      outcome exists, whether the seen-code regime can run, and reasons for each "no". Stage 3
      reads it verbatim.
      Exit: The four items of Verification "Employment-statistics coverage" are recorded with
      the source files and the dates they were read; the chosen Req 2 branch is named.
      ROUTING: writing-plans
      Stage 1: COMPLETE (2026-09-24) — implemented by plan 3
      (specs/plans/completed/3-employment-statistics-coverage.md). Next: resume the roadmap.

- [x] Stage 2: Outcome panel and sealed splits
      Objective: Build the text→code decoding panel and the sealed validation and test splits
      that it and every later selection read.
      Spec: Req 3; Req 4 (text-stage rows: splits, selection log); Req 1 (outcome estimand);
      Verification "Leakage", "Index-entry roles", "Selection hygiene", "Panels" (outcome half);
      D4.
      Gap closed: Req 3; Req 4 (text-stage rows except the monitors, which need Stage 6's query
      path and land in Stage 7); Req 1 (outcome half).
      Consumes: Nothing from Stage 1. The bundle codebook and the index-file ingestion as they
      are today (`data/download_data.py:352-397`). An encoder interface exposing code and query
      embeddings, which the current model cannot satisfy (`text_model/encoder.py:122-123`):
      tests run on stubs, live numbers wait for Stage 6.
      Produces: An index-entry role table (examples-channel text, training query, validation
      query, test query; stratified by code, fractions per D4; 112130 and 541120 candidates
      only) as a bundle member; the examples channel rebuilt from examples-role entries only; a
      leakage checker (exact and near-duplicate at a stated similarity) with its removal count;
      a decoding
      scorer (top-1, MRR, Hit@{1, 5, 10}, lowest-common-ancestor partial credit) over the 1,012
      candidates under a pluggable distance; a selection log recording which split each
      selection read; a sealed test split behind a logged open call.
      Exit: Every index entry holds exactly one role and the two entry-less codes never appear
      as queries (a test asserts both); the leakage check finds no exact match and reports the
      near-duplicates removed; the scorer reports all four metrics for a stub encoder on the
      sealed splits; the selection log exists and the test split cannot be read without a
      logged open.
      ROUTING: writing-plans
      Rollout note (D4): per code, largest-remainder quotas of examples 3/10, training 7/20,
      validation 1/5 and test 3/20; remainder ties broken by a draw seeded with (20260924,
      code); at least one examples-channel entry for every code with entries; held-out quotas
      capped at the code's leak-free entries, the excess to training. Realized: 6,118 / 7,200 /
      4,042 / 3,013 entries (`conf/data/index_roles.csv`, sha256 05099381…).
      Stage 2: COMPLETE (2026-09-24) — implemented by plan 4
      (specs/plans/completed/4-outcome-panel-sealed-splits.md). Next: resume the roadmap.

- [x] Stage 3: Regressor panel
      Objective: Build the regressor panel on the population and grain Stage 1 verified, with
      Req 2's comparators, fitting and two regimes.
      Spec: Req 2 (all bullets except the open item); Req 4 (regressor splits: sealed outer set,
      repeated grouped inner folds); Req 1 (regressor estimand); Verification "Panels"
      (regressor half); D1; D7; D9.
      Gap closed: Req 2 (remainder); Req 1 (regressor half).
      Consumes: Stage 1's finding (`specs/findings/employment-statistics-coverage.md`): its
      decision block, the excluded codes (section 4) and its Sources (the QCEW files and their
      hashes, kept outside the repo). As prior art, Stage 1's
      `scripts/employment_statistics_coverage.py`: a QCEW reader with disclosure classification
      and split-code recovery, whose national-total check awaits plan 3's deferred rounding fix.
      A 2,125-code coordinate table in the arm's export form (tangent coordinates at the origin
      for a hyperbolic arm, raw otherwise), taken as input however it was produced: today only
      the train command's interactive prompt writes one (`cli/commands/training.py:788-800`).
      Stage 2's `SelectionLog` (`panels/selection_log.py`), which takes any panel name, but not
      its seal: `OutcomePanel` enforces the outcome panel's only (finding, section 6).
      Produces: The panel as a command that takes a coordinate table and returns out-of-sample
      predictions per row, keyed by code, year and four-digit-parent group, so that Stage 4 can
      compute whichever statistic Open questions settles. It covers every comparator in each
      regime, the text-only one per D9, on D7's outcome, fitted by ridge on standardized features
      with a nested-cross-validated penalty and with log establishment counts and log wages as the
      covariates (D1). Also produced: the sealed outer sets, readable only through a logged
      opening as Stage 2's test split is (`SelectionLog` records reads but gates none), and every
      outer-set score goes through that opening; the held-out regime's drawn four-digit groups
      committed under a fingerprint as Stage 2's role table is (a redraw gets a
      new fingerprint, which `SelectionLog.openings` would not count as a reopening; the seen
      regime's 2024→2025 set is fixed by D7); the multi-level variant; the branch record Stage 1
      dictated.
      Exit: The panel reports the seen-code and held-out-code regimes separately, with one-hot
      only in the seen regime and every Req 2 comparator scored in each; a test shows the penalty
      is tuned inside the remainder and the outer set is read once; a test shows the panel reads
      the committed outer groups, never a fresh draw; a test shows neither regime's outer set can
      be read without a logged opening; a test shows every row's features are dated
      before its outcome (D7); the branch record matches the finding's decision block.
      ROUTING: writing-plans
      Rollout note: one partition serves both regimes. The held-out outer set holds every row
      of a code containing a held-out four-digit group, the seen outer set the other codes'
      2024→2025 rows, and validation reads the rest. Levels 2–3 are sealed strictly, so the
      held-out regime runs at levels 4–6, and D8's regressor panels are the level-6 cells.
      Realized: 60 of 300 four-digit groups held out, a fifth of each sector's by largest
      remainder with seed 20260924 (`conf/data/regressor_heldout_groups.csv`, sha256
      deddfd4c…, both panels' fingerprint); level-6 rows 1,554 / 777 / 609 (remainder / seen
      outer / held-out outer).
      Stage 3: COMPLETE (2026-09-24) — implemented by plan 5
      (specs/plans/completed/5-regressor-panel.md). Next: resume the roadmap.

- [x] Stage 4: Decision rule and diagnostics
      Objective: Implement Req 5's decision procedure and record, and demote the structural
      statistics to stratified diagnostics that nothing selects on.
      Spec: Req 5; Req 6; Req 1 (taxonomy agreement never a selection criterion); Req 2 (the
      rejected definition); Verification "Decision records", "Diagnostics"; D8; D10; D11.
      Gap closed: Req 5 (except the reference configuration and δ, which Stage 7 supplies);
      Req 6; Req 1 (selection-criterion half); Req 2 (the rejected definition Stage 3 left).
      Consumes: Stage 2's `DecodingResult.per_query`, resampled by code (finding, section 6),
      and Stage 3's per-row predictions, resampled by four-digit group in each regressor regime
      (finding `specs/findings/regressor-panel-splits.md`, section 6): D8's two regressor panels
      are the level-6 cells, a validation read scores each of its rows once per repeat (five),
      and `group` is the code itself at levels 2–3. Plan 5's deferred note on those repeats and
      on the held-out regime's feature years. Stage 3's selection-log reads, which name the arm's
      table and its text-only table by `matrix_fingerprint`, not by file hash. The four QCEW
      slices the panel reads from `qcew_dir` under pinned hashes (`conf/data/regressor_panel.yaml`)
      stay outside the repo, so they must be present wherever the driver scores that panel: a
      Lambda instance has only what is uploaded. No trained arms yet: the tooling is exercised on
      synthetic scores.
      Produces: Decision tooling over D8's three panels: each panel's statistic (D10), paired
      resampling over each panel's unit with seeds nested, 95 % non-inferiority and 98⅓ %
      superiority intervals, δ as a stated multiple of a reference's across-seed standard
      deviation, the non-dominated set, the tie order (its final tie-break: D11), and a
      decision-record schema that carries the selection-log records of the runs it compares (the
      log is gitignored and dies with its worktree or Lambda instance) and immutable references
      (path and content hash) to each arm's text-only table and its provenance file, whose
      backbone, revision, descriptions hash and window the record checks against the arm's (D9),
      and, per arm and seed, to the encoder checkpoint and the 2,125-code table, each table's
      reference also carrying the `matrix_fingerprint` the log names it by; a seed-sweep driver
      that runs a configuration for N seeds, collects every panel's per-unit scores, and keeps
      the referenced artifacts until Stage 12 (a Lambda instance loses them at termination:
      `specs/lambda-remote-workflow.md`); the diagnostics report over all 2,125 codes
      (sector-separation AUC, within-sector rank correlation averaged over sectors and queries,
      MAP over ancestors, NDCG with integer lowest-common-ancestor grades, the Pearson statistic
      without the cophenetic name, unary pairs excluded from parent retrieval);
      `verify-stage4`'s fixed thresholds retired; the
      unwired taxonomy-tasks suite removed, and `metrics/qcew.py` with its re-exports in
      `metrics/__init__.py` and `graph_model/__init__.py`, its API page (`docs/api/qcew_metrics.md`
      and its `docs/.nav.yml` entry: the docs build runs only on main) and its tests, keeping the
      `graph_dataset` case that `tests/unit/test_graph_downstream_evaluation.py:190` parametrizes
      beside it; structural statistics off progress bars and headlines.
      Exit: On synthetic arms with known effects on D8's three panels, the tooling adopts and
      rejects per the rule and writes records with every field Verification "Decision records"
      lists, plus the selection-log records of their runs and the artifact references of every
      arm and seed, text-only tables and their provenance included; the diagnostics report
      contains only Req 6's statistics, stratified as listed, with no threshold and no
      pass/fail; no monitor, gate or headline reads a structural statistic; neither
      `metrics/qcew.py` nor the taxonomy-tasks suite remains.
      ROUTING: writing-plans
      Rollout note: the decision tooling ran on synthetic arms and fixture panels only, so
      Stage 7's reference sweep is its first real use; its artifact store sits outside any
      worktree and is copied off a Lambda instance before termination. `decide` admits a run
      only if its reads are logged after the margins' `fixed_at`, comparing clocks across
      machines, so the machine that fixes the margins and those that train must agree on the
      time. It also pairs arms only on panels holding the same data, hashing the regressor
      rows' log-transformed counts bit for bit, so the reference arm that fixes the margins and
      every arm compared with it must be swept on one platform under one `uv.lock`.
      `tools diagnostics` replaced `verify-stage4`. Text and HGCN validation still log the old
      structural statistics, off every progress bar and headline, for Stages 7 and 11 to
      remove.
      Realized: on the text-only table (all-MiniLM-L6-v2 at revision 1110a243, 384 dimensions,
      spherical), sector-separation AUC 0.8538, within-sector rank correlation 0.2974 over queries
      and 0.3857 over sectors, MAP over ancestors 0.3262, NDCG@10 0.7742, distance Pearson
      0.2378, parent retrieval@1 0.2855 over 1,583 queries; a random table reads at chance
      (AUC 0.4940).
      Stage 4: COMPLETE (2026-09-25) — implemented by plan 6
      (specs/plans/completed/6-decision-rule-and-diagnostics.md). Next: resume the roadmap.

- [x] Stage 5: Supervision target and text
      Objective: Rebuild the supervision bundle around the tree metric D*, directed
      redirections and Req 9's text-construction rules.
      Spec: Req 7 (D*; the IC ablation waits for Stage 9); Req 8(a), 8(c) generation side,
      lineal references, reserved slot removed; Req 9 (masking data side, inheritance, unary
      pairs, input window for the current backbone); Verification "Target", "Exclusions"
      (generation half), "Text" (data half), "Backbone input window" (current checkpoint); D5;
      D9 (the comparator reads the arm's text).
      Gap closed: Req 7 (except the IC ablation); Req 8 (a, lineal, generation side of c);
      Req 9 (except the model-side mask and the channel-presence ablation).
      Consumes: Stage 2's index-entry roles (the examples channel holds examples-role entries
      only). Stage 2 shipped them as an optional `index_roles` member under
      stage3-supervision-v1 and the rebuild in `data preprocess`, but built no bundle: this
      stage's contract version makes the member required, and its rebuild is the first bundle
      carrying both. Stage 2's held-out leakage check (`panels/leakage.py`), which the bundle
      build runs but which reaches activity phrases only by splitting `excluded` at `--`. The
      current bundle contract (`data/supervision_bundle.py`, `supervision/artifacts.py`) as the
      thing to version. The current backbone's own documentation for its trained window. Stage
      3's text-only builder (`panels/text_only.py`), which embeds each code's channels for D9's
      comparator at `text_only.max_length: 512` (`conf/data/regressor_panel.yaml`), the window
      Req 9 rejects. Stage 4's diagnostics report (`metrics/diagnostics.py`), which computes D*
      from the codes' own lineage rather than reading the bundle, so the bundle's new D* must
      equal it on every pair.
      Produces: A new bundle contract version: pair facts carrying D* (no 99, no half-step, a
      virtual root above the sectors); a redirection table (activity phrase, referencing code,
      destination code, lineal flag) with each cross-reference once; exclusion text
      de-duplicated; descriptions with a provenance column and a documented deterministic
      inheritance rule; a unary-pair flag on the 522 five-digit codes; absent channels as nulls
      with the window policy applied; the 43-versus-68 non-redirection count reconciled;
      relation names kept only as edge-type and diagnostic labels (D5). The training-pairs
      member keeps generating, on D* and without exclusion negatives, until Stage
      7 removes it. Consumers (losses, metrics, graph loader, curriculum thresholds) read the
      new values. The two unpushed config commits are superseded by the new manifest path.
      Exit: A bundle validator asserts that D* satisfies the triangle inequality over all
      triples via lowest common ancestors, contains no 99, and gives λ(i) + λ(j) − 2 across
      sectors; each cross-reference appears once in the exclusion text; the build's held-out
      leakage check covers the redirection table's activity phrases, which Stage 7 trains on, and
      finds no match; the nine lineal references are flagged and no generated training row uses
      any exclusion as a negative;
      no placeholder string exists in any channel; every inherited description carries a
      provenance value and the 14 formerly arbitrary choices resolve by the documented rule;
      the 522 unary pairs are flagged and absent from generated positives; the current
      backbone's trained window is recorded with the share of each channel's texts that
      exceeded it, and no input exceeds it, the text-only builder's included.
      ROUTING: writing-plans
      Rollout note: the switch happens at merge. Main then loads only `stage3-supervision-v2`
      bundles, and a checkpoint trained on bundle 18403d29 loads weights-only. Bundle
      `301cce28-539c-42ea-8781-496bbdcf511c`, built from a fresh `data preprocess` (descriptions
      sha256 `fe8c54e3…`), is copied to the main checkout and to every Lambda instance, never
      rebuilt. Held-out queries leak into five cross-references, which are withheld from the
      exclusion text (user decision 2), so "each cross-reference appears once" holds for the
      other 4,596. Until Stage 7, the
      relation margin axis keeps its cross-sector margin, keyed on the `cross_sector` label; the
      `quota_selections` counter stays at zero; and Phase 1's sibling mask masks every candidate
      at D* 2, grandparents and grandchildren included. HGCN's curriculum thresholds, 7, 9 and 10
      under D*, bind until Stage 11 removes them.
      Realized: D* takes integers 1–10 over 2,256,750 pairs, equals `metrics/diagnostics.py`'s on
      every pair and passes the triangle inequality over all 2,125³ ordered triples; 522 unary
      pairs; 4,623 redirection rows (4,601 cross-references and 22 harvested paragraphs; 4,529
      activity phrases; 9 lineal references), with the spec's 43 and the old 68 reconciled; no
      held-out leakage, activity phrases included; 45,163,632 training pairs, none with an
      exclusion or lineal reference as its negative or a unary pair as its positive; a 128-token
      window, beyond which lie 0.0000 of titles, 0.0725 of descriptions, 0.0977 of examples and
      0.4154 of exclusion texts (`specs/findings/supervision-target-and-text.md`).
      Stage 5: COMPLETE (2026-09-26) — implemented by plan 7
      (specs/plans/completed/7-supervision-target-and-text.md). Next: resume the roadmap.

- [x] Stage 6: Shared encoder and low-dimensional projection
      Objective: Replace the four-copy LoRA encoder and mixture-of-experts fusion with one
      shared encoder behind field markers, masked fusion, one affine map to a configurable
      dimension, and a query path.
      Spec: Req 14 (encoder and fusion; the backbone stays current; MoE kept only as an
      ablation option); Req 12 (one affine map; dimensions 8, 16 and 32); Req 9 (masking, model
      side; channel-presence indicator as an option for Stage 9); Req 16 (a standalone export
      command for the 2,125-code table); Verification "Text" (no absent channel contributes to
      fusion).
      Gap closed: Req 14 (encoder half); Req 12 (projection and dimension half); Req 9 (model-
      side mask); Req 16 (export half).
      Consumes: Stage 5's bundle (null channels, window policy) and its tokenization cache,
      which encodes an absent channel as the empty string with a per-channel `present` flag for
      the mask and tokenizes every channel at the 128-token window `utils/input_window.py`
      records. The current objective and
      dataloader, unchanged, as an interim training harness only. Stage 2's `QueryCodeEncoder`
      protocol and `OutcomePanel.score` for a first live validation-split reading (finding,
      section 6). The scorer's `lorentz` distance assumes curvature −1 (c = 1), which is the
      harness's: `loss.curvature` is a fixed 1.0 that nothing learns, though plan 4's deferred
      item assumed a learned one. Whether the scorer takes a curvature-aware distance or a guard
      on c = 1 is for Stage 6's spec to settle.
      Stage 3's panel as the export's reader (finding `specs/findings/regressor-panel-splits.md`,
      section 6): it refuses Lorentz points and constant columns, so a hyperbolic export writes
      the tangent coordinates at the origin without the zero time coordinate a log map keeps. A
      read needs a text-only table rebuilt from Stage 5's descriptions (`tools text-only-table`)
      and the four QCEW slices at `qcew_dir`, under the hashes `conf/data/regressor_panel.yaml`
      pins.
      Produces: One encoder module implementing `QueryCodeEncoder`, with code and query
      embeddings in the same space;
      fusion options masked mean, attention pooling and MoE (MoE ablation-only); a configurable
      embedding dimension in {8, 16, 32}; a standalone export command writing the 2,125-code
      table in Req 2's form; the per-channel adapter copies and the load-balancing term deleted
      with the default fusion; a `summaries` entry in the tokenization cache's identity, null
      until Stage 6b; an outcome read whose logged `table` names the code vectors the
      encoder decodes against (Stage 4's sweep logs the runner's table as that label, which
      `decide` checks but cannot tie to the decoding).
      Exit: A query embeds through the same encoder as a code (test); masking an absent
      channel's input leaves the output unchanged (test); exactly one affine map sits between
      the encoder and the point; the model trains under the interim objective at dimension 16
      and the export command writes the 2,125-code table, which `tools regressor-panel` reads on
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
      epoch (`data_loader.n_epochs=1`) on MPS in about 30 minutes. Its export holds 2,125 codes,
      0.2038 of them at the cap. On the validation splits, the regressor panel's level-6
      `embedding` comparator reads R² 0.2591 (seen) and 0.2009 (held-out), beside `text_only`'s
      0.2975 and 0.2248. The outcome panel reads top-1 0.1249 and MRR 0.2301 over 4,042 queries.
      These numbers are a floor for Stage 7 (`specs/findings/shared-encoder-first-reading.md`).
      Stage 6: COMPLETE (2026-10-03) — implemented by plan 8
      (specs/plans/completed/8-shared-encoder-and-projection.md). Next: resume the roadmap.

- [x] Stage 6b: Window-fitting summaries
      Objective: Replace the tail truncation of channel texts beyond the backbone's trained
      window with frozen summaries that fit it, before Stage 7 trains the reference
      configuration on them.
      Spec: Req 9 (input windows), under the user's ruling of 2026-10-03 that over-long texts are
      summarized, not truncated; Req 3 (leakage); D9 (the text-only comparator reads the arm's
      text).
      Gap closed: none in the Gap analysis; the stage carries out the 2026-10-03 ruling on Req 9's
      input windows, and Stage 9 still records each candidate's window.
      Consumes: Bundle 301cce28's descriptions (sha256 `fe8c54e3…`), unchanged, since summaries
      written into them would change the description fingerprint the bundle records. The
      overflow at the 128-token window under MiniLM's tokenizer, special tokens included,
      measured on 2026-10-03 without the field marker
      (`specs/completed/shared-encoder-and-projection.md`, Rollout note): description 153 of
      2,111 texts (sectors 16 of 20, median 246 tokens, maximum 1,131; subsectors 59 of 96),
      examples 105 of 1,075 (19 % of its tokens), excluded 464 of 1,117 (26 %). With the marker
      the cache adds, 162, 106 and 485 texts are truncated (resume, 2026-10-04). Stage 6's
      tokenization cache, whose sidecar carries a `summaries` entry that stays
      null until this stage (format `channels-v3`; a cache built under other summaries is
      rebuilt), and its field markers (`text_model/fields.py`), two tokens of each window under
      MiniLM's tokenizer (the field's name and `:`). Stage 2's
      leakage matcher (`panels/leakage.py`) and frozen role table: an abstractive summary can
      contain a held-out query, and a fix rewrites the summary, never the table.
      Produces: A frozen, committed summaries artifact with provenance, keyed by code, channel,
      source-text sha256 and target window, and pinned by hash; the tokenization cache and the
      text-only builder (`tools text-only-table`, D9) reading it; its hash in the text-only
      provenance, `TextOnlyRef`, `ArmSpec` and `decide`'s D9 check. The stage spec settles which
      channels it covers (the ruling names descriptions; examples carry the most leakage risk,
      and excluded carries Req 8's redirections) and whether summaries are extractive, which
      cannot add a match, or abstractive, which needs an audit against the validation and test
      queries: a non-selecting read of sealed text that needs the user's explicit approval.
      Exit: No covered channel's text is truncated: each text beyond the window reads as its
      summary, which fits with its field marker (test); the leakage check finds no held-out
      query in any summary; the artifact is committed under a pinned hash, and the cache, the
      text-only table's provenance and the decision records name it.
      ROUTING: brainstorming
      Rollout note: the switch happens at merge. Every `channels-v3` cache rebuilds once. A
      checkpoint made before this stage records no summaries: exact resume, export, the outcome read
      and the HGCN feeder refuse it. `ArmEncoder.from_files` and `run_seed_sweep` refuse a table
      exported before it, and the artifact store refuses a text-only table built before it, so
      `run_seed_sweep` and `decide` do too. Weights-only loading still works, and
      `tools regressor-panel`, which checks no provenance, still pairs pre-6b tables. Plan 8's Exit
      numbers stay the floor its finding records; Stage 7 trains and builds its text-only table on
      the summaries.
      Realized: `conf/data/window_summaries.csv` (sha256 `dd425eb5…`), pinned for MiniLM at 128
      tokens by `WINDOW_SUMMARIES` (`panels/window_summaries.py`), holds 753 extractive summaries,
      picked by backbone centrality (`centrality-v1`) from whole sentences, clauses and examples
      entries: 162 descriptions, 106 examples texts and 485 exclusion texts. They keep a mean 0.654,
      0.616 and 0.622 of their source tokens, and 581 of 1,235, 1,179 of 2,307 and 1,718 of 3,270 of
      their units. No title is over the window. The token cache, the checkpoint contract, the export
      and text-only provenances and the decision records carry the sha256, and the export provenance
      also names the tokenizer. The Exit trained nothing and read no split.
      Stage 6b: COMPLETE (2026-10-04) — implemented by plan 9
      (specs/plans/completed/9-window-fitting-summaries.md). Next: resume the roadmap.

- [ ] Stage 7: Objective, anchors and live radius (the reference configuration)
      Objective: Replace the six-term objective and its sampling machinery with Req 11's three
      terms over all codes on a live radius, wire selection to the validation splits, and fix
      δ per panel from the reference configuration's seeds.
      Spec: Req 11; Req 10; Req 13; Req 8(b), 8(c) training side; Req 4 (the in-sample loss
      selects nothing; monitors read validation splits); Req 5 (reference configuration, δ);
      Req 6 (no structural monitor); Verification "No inert terms", "Coverage", "Radius",
      "Exclusions" (training half), "Text" (unary pairs absent from positive supervision),
      "Selection hygiene" (validation half); D2, D5, D6, D8, D10.
      Gap closed: Req 11; Req 10; Req 13; Req 8 (b, training side of c); Req 4 (monitors);
      Req 5 (reference configuration and δ).
      Consumes: Stage 6's encoder and query path (`SharedEncoder` in
      `text_model/shared_encoder.py`; `ArmEncoder.from_files` and `read_outcome_validation` in
      `text_model/arm_encoder.py`; `tools export-table` and `tools outcome-panel`), which supply
      `SeedArtifacts`: the checkpoint, the exported table, the `ArmEncoder` and its `distance`.
      Also Stage 6's interim head (`HyperbolicHead`, `text_model/hyperbolic.py`), whose cap at
      norm 2 this stage replaces, and legacy containment, which export and reads refuse and the
      HGCN feeder still serves. Stage 6b's summaries (`WINDOW_SUMMARIES` in
      `panels/window_summaries.py`), which the reference configuration and its text-only table read:
      a checkpoint's contract records their sha256, and exact resume, export and reads refuse one
      trained under other summaries, plan 8's Exit checkpoint included, so Stage 6's floor was read
      on truncated text; Stage 5's D*, redirection table and unary
      flags (the table's `activity` phrases with their referencing `code`; its five withheld rows
      carry none); Stage 2's query splits, scorer and selection log (the log's path is
      `OutcomePanelConfig.selection_log`, `logs/selection_log.jsonl`: gitignored, so a
      worktree's log goes with the worktree, and Stage 4's decision records keep its records).
      The test split stays sealed: Stage 12 opens it, as the finding's section 6 erratum says.
      Stage 3's panel, on its validation split only; its text-only table is rebuilt from the
      arm's own descriptions with `tools text-only-table`, because the table Stage 3 built
      embeds bundle 18403d29's text, which Stage 5 replaces (D9), and the store refuses plan 8's
      Exit table (`checkpoints/plan8_exit/`), whose provenance predates the summaries; plan 9's Exit
      built one under them (`checkpoints/plan9_exit/text_only.parquet`). Stage 4's seed-sweep driver
      (`decision.sweep.run_seed_sweep`, whose `ArmRunner` returns each seed's `SeedArtifacts`: the
      checkpoint, the 2,125-code table in the export form with the provenance `tools export-table`
      writes beside it, the `QueryCodeEncoder` and its distance; before any panel read,
      `check_seed_table` refuses a seed whose table's backbone, revision, descriptions, summaries or
      window differ from the arm's `ArmSpec`, whose `summaries_sha256` is `summaries_identity` of
      the arm's backbone), decision tooling (`tools margins`, `tools decide`) and δ procedure. The
      margins
      are fixed from the reference arm's record before any other arm of a decision reads a
      panel, since a decision refuses a run that read first, and the artifact store's root must
      outlive the Lambda instance that trains.
      Produces: The reference configuration (hyperbolic, dimension 16, current backbone, shared
      encoder) with a query→code task term over training queries and activity phrases (the
      referencing code always scored), a listwise code–code term with graded targets from D*
      over all 2,125 codes through a per-epoch code-point cache, a radial term with a virtual
      root and sectors at positive radius, learned logit scales, a gradient-passing bound in
      place of the cap, one radial coordinate, curvature fixed at 1 with no parameter; deleted:
      radius penalty, distance matching, pairwise preference, curriculum and mining rules,
      false-negative clustering, the logged margin, pre-drawn tuples, eligibility rules,
      inverse-distance draws and Phase 1's sibling mask (at D* 2 it also masks grandparents),
      what remains of the exclusion quota (Stage 5 removed its slot; the `quota_selections`
      counter and `SelectionReason.EXCLUSION_QUOTA` remain), the relation margin axis (D5), and
      legacy
      containment (D2); monitors for checkpointing, early stopping and learning rate reading
      the validation query split's MRR (D6); a decision record fixing δ for each of D8's three
      panels from at least 5 seeds; the legacy structural statistics removed from the text
      stage's validation (`text_model/evaluation.py`, `text_model/mixins/validation.py`,
      `text_model/mixins/logging.py`), which Stage 4 took off the progress bar only, and
      `tools investigate` retired, so Req 6's statistics come only from `tools diagnostics`.
      Exit: On a real batch every objective term has a nonzero gradient (test); on a trained
      run the gradient with respect to radius is nonzero, radii vary within each level, the 20
      sectors sit at distinct positive radii, and manifold validity and distance resolution
      hold at the largest observed radius; every code is an anchor and the code–code term
      covers all 2,125 codes at every step with no pre-drawn tuples (test); no exclusion pair
      acts as a code–code negative and each cross-reference query scores its referencing code
      (test); unary pairs are absent from positive supervision; the selection log shows only
      validation splits read; a decision record fixes δ for each of D8's three panels from at
      least 5 seeds; the text stage's validation computes no structural statistic.
      ROUTING: brainstorming

- [ ] Stage 8: Geometry × dimension
      Objective: Implement the Euclidean and spherical arms and run the crossed nine-cell
      comparison under Req 5.
      Spec: Req 12; Req 5 (several arms, ties); Req 2 (export form per arm); Verification
      "Geometry × dimension"; D8; D9.
      Gap closed: Req 12 (geometry arms and the decision).
      Consumes: Stage 7's reference configuration and δ; Stage 4's tooling and seed-sweep
      driver; Stage 6's configurable dimension, and its head's `distance` attribute, which
      `ArmEncoder.distance` reads (`HyperbolicHead.distance` is `lorentz`; export and reads
      refuse c ≠ 1, spec R8). Only that name follows the head: `ArmEncoder` maps queries and
      codes through the hyperbolic exp map at the origin whatever the head
      (`text_model/arm_encoder.py`), so the Euclidean and spherical arms need their own maps
      there. Stage 2's scorer, whose registered distances are
      `euclidean`, `cosine` and `lorentz` at curvature −1 (`panels/decoding.py`).
      Produces: Geometry as a configuration factor with per-arm distance, decoding and export (the
      radial term only in the hyperbolic arm; every arm's export writes the provenance
      `tools export-table` writes, tokenizer and summaries included, which `run_seed_sweep` and
      `ArmEncoder.from_files` check); guards on the two tie-order keys Stage 4
      leaves open: `run_seed_sweep` refuses an arm whose `SeedArtifacts.distance` does not fit
      its `ArmSpec.geometry`, and `decide` compares each read's logged `dimension` with the
      arm's (the decision fixture's reads hard-code 16); the nine-cell decision record; the
      selected cell as the reference for Stage 9.
      Exit: All nine cells have at least 5 seeds scored on D8's three panels; the decision
      record names the non-dominated set and the chosen cell under the tie order; each arm's
      export uses tangent coordinates for hyperbolic and raw coordinates otherwise.
      ROUTING: writing-plans

- [ ] Stage 9: Backbone and one-factor ablations
      Objective: Choose the backbone and settle the one-factor ablations (mixture of experts,
      channel-presence indicator, information-content target) from the selected cell under
      Req 5, and resolve the input-window item for each candidate.
      Spec: Req 14 (backbone: the current checkpoint, at least two current general-purpose
      embedding models, a frozen-encoder control; MoE ablation); Req 9 (channel-presence
      ablation; input window); Req 7 (IC ablation); Req 6 (IC-graded relevance only if IC is
      adopted); Verification "Backbone input window"; D9 (the text-only comparator follows each
      arm's backbone).
      Gap closed: Req 14 (backbone half); Req 9 (channel-presence ablation, window); Req 7 (IC
      ablation).
      Consumes: Stage 8's selected cell; Stage 6's fusion options (`model.fusion`: `masked_mean`,
      `attention`, `moe`) and its one backbone loader (`load_base_model`,
      `text_model/shared_encoder.py`), whose LoRA adapter (`all-linear`) also wraps the pooler's
      dense layer, which mean pooling never reads; Stage 6b's summaries, pinned per backbone, not
      per window (`WINDOW_SUMMARIES` in `panels/window_summaries.py` holds one `SummariesPin`, with
      its window, per tokenizer name). A candidate whose channel texts all fit its window reads them
      as they are; one with a longer text is refused until
      `naics-embedder data summaries --backbone <name> --output <path>` writes its own artifact (the
      default path is MiniLM's) and its pin is committed. Each candidate's summaries are selected
      under its own frozen weights, so two candidates can read different summaries of one text;
      whether one selection backbone serves them all is this stage's brainstorm's call
      (`specs/completed/window-fitting-summaries.md`, section 2.2). Stage 4's tooling.
      Produces: A decision record per factor; the selected text stage (arm A for Stage 10); the
      trained window recorded per candidate with each channel's overflow share (a window enters
      `TRAINED_WINDOWS` in `utils/input_window.py` from the candidate's own documentation, since
      no config accepts a backbone without one, and `input_window_record` computes the shares);
      if IC is
      adopted, the bundle's target and the diagnostics' relevance grades (lowest-common-ancestor
      depths in `metrics/diagnostics.py`, Stage 4) switched to it.
      Exit: Each factor has a Req 5 record with at least 5 seeds per arm, and the frozen-encoder
      control ran; the adopted backbone's window is recorded from its own documentation with
      no input beyond it, the text-only builder's included.
      ROUTING: brainstorming

- [ ] Stage 10: Graph-stage decision experiment
      Objective: Run arms A–E on the selected text stage and decide whether the graph stage
      stays, with arm D repaired only to run in the text stage's space.
      Spec: Req 15; Req 16; Req 4 (graph-stage rows: no private validation tail, no last-epoch
      export, selected like every other arm); Verification "Decision experiment",
      "Deliverable"; D3; D8.
      Gap closed: Req 15; Req 16 (composition and deliverable); Req 4 (graph-stage rows).
      Consumes: Stage 9's selected text stage; Stage 4's tooling and seed-sweep driver; Stage
      6's export command, and its HGCN feeder (`generate_embeddings_from_checkpoint`), which
      writes d + 1 `hyp_e*` columns through the shared encoder; Stage 5's bundle as the graph
      stage's structural input. Under D* its
      curriculum thresholds bind (phase-1 negatives at D* ≤ 7, phase-2 at D* ≤ 9), and arm D runs
      with them, since only Stage 11 removes them.
      Produces: Arm B (matched-compute continuation of the text stage); arm C (parameter-free
      smoothing toward the parent-and-children mean, α tuned on validation); arm D (the graph
      stage at the text stage's dimension, exponential and logarithmic maps per the selected
      geometry, its level-radius term per D3, checkpoint selected on the validation query
      split's MRR (D6), not on the structural statistics HGCN's validation still logs (Stage 4
      took them off the progress bar only), no private tail, no last-epoch export); arm E
      (text-shuffle control, run only if D wins); the keep-or-drop record; the deliverable, a
      2,125-code table from the selected arm. Every arm's per-seed table carries the export's
      provenance (`text_model/export.py`: its checkpoint's hash, its own, the window, the tokenizer
      and the summaries' sha256). Before any panel read, `run_seed_sweep` refuses a seed whose table
      lacks it or whose backbone, revision, descriptions, summaries or window differ from the arm's
      (`check_seed_table`, D9), and `ArmEncoder.from_files` requires it before Stage 12 can embed
      sealed queries against it, so an arm whose table no `tools export-table` call writes (C's
      smoothing, D's graph stage) writes the same provenance.
      Exit: Arms A–D have at least 5 seeds on D8's three panels under the shared selection
      protocol; E ran if and only if D won, and its result is recorded; the decision record
      states keep or drop under Req 5; the 2,125-code table exists for the selected arm.
      ROUTING: writing-plans

- [ ] Stage 11: Graph-stage outcome: repairs or removal
      Objective: If Stage 10 kept the graph stage, apply Req 17's repairs; if it dropped the
      stage, remove the graph stage and its curriculum system from the method.
      Spec: Req 17; Req 15 (decision); Req 16 (one space retained).
      Gap closed: Req 17 (if kept) or its lapse (if dropped).
      Consumes: Stage 10's decision record; arm D as it ran; Stage 4's tooling.
      Produces: If kept: parent–child edges plus self-loops with sibling and multi-generation
      edges only by ablation; an edge weight that enters once; radius-preserving normalization
      or Lorentz-native layers; explicit retention of the text geometry by distillation or a
      residual correction; an objective without hinge temperature, adaptive margin or
      uncertainty weights; no curriculum filters or four-phase controller; each repair adopted
      under Req 5; the deliverable refreshed. If dropped: `graph_model/`, its curriculum
      package, `conf/graph.yaml`, the graph docs and every graph-stage code path removed, the
      HGCN feeder (`generate_embeddings_from_checkpoint` and `train`'s prompt for it) included
      but not the `encode_token_rows` it shares with the export and the arm encoder, with
      the deliverable defined by the text stage alone and arm D's referenced artifacts kept for
      Stage 12 (Req 16 scores its code points against the text stage's queries, so no graph code
      is needed). The bundle build imports the curriculum package for its required
      `difficulty_thresholds` member (`data/supervision_bundle.py:694-697`,
      `supervision/artifacts.py:177`), so the drop path decides how the build computes it or
      whether the contract drops it, and leaves bundle 301cce28 loadable.
      Exit: If kept: a Req 5 record per repair, a final 2,125-code table from the repaired
      stage, and no structural statistic computed at HGCN validation, which Stage 4 took off the
      progress bar only. If dropped: no graph-stage code path remains, the suite passes, the
      deliverable is the text stage's table, and arm D's referenced artifacts still match their
      hashes.
      ROUTING: brainstorming if kept (the retention and normalization choices are open);
      writing-plans if dropped

- [ ] Stage 12: Sealed final evaluation
      Objective: Open each sealed test split once, for the final configuration and the
      comparisons recorded for it, and report Req 1's two estimands on sealed held-out data.
      Spec: Req 4 (the test splits opened once); Req 1 (measured on sealed held-out data); Req 5
      (intervals); Verification "Selection hygiene" (opening half); D8; D10 (the regressor
      estimand is reported as the gain over the regime's sparse encoding).
      Gap closed: Req 4 (the opening); Req 1 (the sealed estimates).
      Consumes: The final configuration and its 2,125-code table (Stage 11's, or Stage 10's if
      the graph stage was dropped); the decision records of Stages 7–11 with the selection-log
      records and artifact references they carry (Stage 4's schema); by those references, the
      per-seed encoder checkpoints and 2,125-code tables of the final configuration and of every
      arm in its recorded comparisons, since sealed queries were never embedded, and each arm's
      text-only table with its provenance, without which the regressor panel reads no arm (it names
      the summaries' sha256, which `decide` compares with the arm's, D9). Stage 6's query path,
      `ArmEncoder.from_files` (`text_model/arm_encoder.py`), which embeds sealed queries from a
      checkpoint and its table: it takes the tokenizer and window from the arm's own config
      (`code_token_config`). Before any model loads, it refuses a table exported before Stage 6b or
      one whose provenance names another window, tokenizer or summaries than that config reads, so
      each recorded arm is read under the summaries pin it was exported under (`WINDOW_SUMMARIES`):
      re-pinning a backbone's artifact refuses its earlier tables and checkpoints. It maps through
      the hyperbolic exp map only, so a Euclidean or spherical arm needs Stage
      8's map; `read_outcome_validation` reads validation only, so the test read goes through
      `OutcomePanel.score` with the `detail` keys (`table`, `checkpoint`) it logs. The
      four QCEW slices under the hashes `conf/data/regressor_panel.yaml` pins, which loading the
      panel re-reads from `qcew_dir` outside the repo, and a bundle codebook with the pinned
      2,125 codes; Stage 2's
      `OutcomePanel.open_test`; Stage 3's sealed outer sets and the logged opening that guards
      them (`RegressorPanel.open_outer`): one opening per regime covers every level the panel
      object has loaded, and the log names the panels `regressor_seen` and `regressor_heldout`,
      both under the fingerprint `deddfd4c…`. Both openings belong to a panel object, so every
      arm and seed is scored through one object per sealed set: one `open_test`, one
      `open_outer` per regime, then a test read per arm and seed. `tools regressor-panel --split
      test` opens the outer sets on every call and scores one table, so a second arm or seed
      through it would be logged as a reopen; no command opens the outcome test split. Stage 4's
      tooling and seed-sweep driver, which score and decide on validation reads only: `decide`
      refuses a run whose records are not validation reads, and `regressor_scores` takes
      level-6 validation rows only. A sealed estimate reuses `decision/resampling.py` and
      `decision/rule.py` on test-split scores, one prediction per outer row, through a path this
      stage adds.
      Produces: One logged opening per sealed set (the outcome test queries and each regressor
      regime's outer set, per D8); sealed estimates with D8's intervals for the final
      configuration and each comparison recorded for it; a written finding.
      Exit: The selection-log records, gathered from the decision records and this stage's own,
      show exactly one opening per sealed set, under the fingerprint Stage 2 or Stage 3
      committed, after every validation read that selected anything; every sealed estimate comes
      from referenced artifacts whose hashes match the records, and each text-only table's
      provenance names its arm's backbone and descriptions; the finding reports each sealed
      estimate with its interval.
      ROUTING: writing-plans

## Stage-spec stamp

Every stage spec's Rollout note carries this line, which writing-plans copies verbatim into the
stage plan's header:

> Roadmap: specs/naics-embedding-roadmap.md, Stage N — on plan completion, tick the stage and
> re-validate later stages against what shipped.

On completion the stamp becomes authoritative:

> Stage N: COMPLETE (YYYY-MM-DD) — implemented by plan <id> (path).
> Next: resume the roadmap.

## Completion

Retirement is gated behind a conformance audit of the accumulated system: re-run the gap rubric
over Reqs 1–17 with evidence per verdict (implementing stage and plan, Deviation notes,
deferred-items entries), and re-check every Verification item in the spec. An unmet requirement
exits exactly two ways: a new stage, with the roadmap staying live, or a conscious deferral with a
written why. This is not a whole-roadmap code review; each stage was reviewed when it merged.
Optional independent check: re-run describe-critique-methodology's Describe mode on the final
system and diff the fresh description against the spec; that re-derivation is the natural input
to the next critique round. When improvement directions are diffuse, route to creative-thinking.
Parking is a first-class exit: remaining stages append to `specs/deferred_items.md` as
self-contained items, and this file moves to `specs/completed/` marked
`PARKED (date) — N of M stages complete`, retiring the methodology and review files with it.
