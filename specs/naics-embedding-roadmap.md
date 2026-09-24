# NAICS embedding — Roadmap

> For agentic workers: REQUIRED SKILL: derive-roadmap — resume via its
> reconcile step; route each unticked stage per its ROUTING line; never plan
> this document wholesale.

**Status: APPROVED (2026-09-23); resumed twice on 2026-09-24.** Derived in a session that could not
ask questions, then the six open questions were answered interactively and the eleven-stage
partition approved at the human checkpoint (decisions D1–D6 below). Stages 1 and 2 are complete.
The first resume re-validated Stages 2–11 against Stage 1 and recorded D7 and D8; the second
re-validated Stages 3–11 against Stage 2, recorded D9 and added Stage 12 (below). Stage 3 is next,
per its ROUTING line, in a fresh session.

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
  as this recorded deviation, not an unmet requirement. The final tie-break stays open (Open
  questions).

**Decision (2026-09-24, second resume).** One ambiguity Stage 3 cannot be planned without, answered
by the user at the second resume checkpoint.

- **D9 — Req 2's text-only comparator (Stage 3; Stages 8 and 9 re-run it).** Req 2 names no
  representation, and review C28, its source, lists two: a frozen encoder and TF-IDF. Decision:
  the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the arm's
  dimension. That backbone is the current checkpoint until Stage 9 adopts another, and whichever
  backbone the arm uses after. The comparator measures what taxonomy training adds over the same
  encoder reading the same text.

## Gap analysis

| Req | Verdict | Evidence | Note |
|---|---|---|---|
| 1 | missing | `text_model/mixins/validation.py:182` (ground truth is the taxonomy distance matrix), `:167-170` (300-code subsample); `utils/training.py:307` (checkpoint monitors `val/contrastive_loss`); `tools/embeddings_verification.py:33-35` (structural acceptance gate) | No evaluation reads data outside the taxonomy; no sealed split exists; structural statistics still gate (`verify-stage4`). Looked in `metrics/*`, `mixins/validation.py`, `tools/*`, `cli/commands/*`, `graph_model/hgcn.py:866-946`. |
| 2 | implemented-differently | `metrics/qcew.py:79` (one row per code), `:110`, `:136` (`Ridge(alpha=1.0)` on unscaled features), `:131-132` (2022, private only), `:185-187` (one `GroupShuffleSplit`), `:178` (covariates are QCEW's own establishments and wages), `:331` (multi-level loop); no CLI wiring in `cli/commands/*`; only run: `tests/unit/test_graph_downstream_evaluation.py:84` on a synthetic CSV | The module is the definition Req 2 rejects (one row per code, fixed penalty, no ancestor, text-only or covariates-only arms, no regimes) and has never run on real data. The (open) item is unresolved: Stage 1. Outcome data are external BLS QCEW files, not a repo artifact. |
| 3 | missing | `data/download_data.py:86`, `:397` (index file becomes the examples channel, joined with `'; '`); `text_model/encoder.py:122-123` (forward needs all four channels; no single-text path); `metrics/core.py:152` (`RetrievalMetrics` is code–code with binary relevance, called only from the never-instantiated `metrics/runner.py:110`) | No query encoding, decoding, roles or leakage check. Looked in `text_model/*`, `metrics/*`, `tools/*`, `cli/*`, `tests/unit/*`. |
| 4 | missing | `text_model/dataloader/datamodule.py:876` (validation is the same builder with seed + 1000), `:1089`, `:1092` (same `_repaired_dataset` for train and validation), `:840` (`val_split` unused); no `test_step` or `test_dataloader` in `src/`; `cli/commands/training.py:662`, `:797` (checkpoint and export from the best `val/contrastive_loss`); `graph_model/dataloader/hgcn_datamodule.py:212`, `:245-246` (5 % unshuffled tail); `graph_model/hgcn.py:1342`, `:1347` (no checkpointing; last-epoch export) | The in-sample contrastive loss selects everything (`datamodule.py:1132-1133` documents it); no test split; the graph stage has both forbidden behaviours. |
| 5 | missing | `conf/config.yaml:5`, `conf/graph.yaml:100` (single seed); no seeds, pairing, interval, dominance or record code in `src/`, `tests/`, `conf/`, `scripts/`; `tools/embeddings_verification.py:33-35` (the fixed thresholds Req 5 replaces) | Only the mechanism to be replaced exists. The one retained run holds three metric records (`outputs/sadc_default/version_0/evaluation_metrics.json`). |
| 6 | implemented-differently | `metrics/core.py:341` (`cophenetic_correlation`; Pearson at `:389`), `:508` (continuous NDCG grades); `metrics/structural_spearman.py:16` (global, unstratified); `metrics/hierarchy_structure.py:103-106` (parent retrieval over every pair, unary pairs included); `text_model/mixins/validation.py:406-408` (cophenetic on the progress bar); `tools/metrics_tools.py:104-106` (headline table); no sector AUC or within-sector statistic in `src/` | The statistics exist but are global, keep the cophenetic name, grade NDCG continuously, score unary pairs, and headline or gate. |
| 7 | implemented-differently | `data/compute_distances.py:160-164` (−0.5 on lineal pairs), `:244` (`fill_null(99)`); `supervision/schema.py:36`, `:41`; `tests/unit/test_data_distances.py:280`, `:492` pin both; no information-content code anywhere | The collateral term matches D*; the half-step and the 99 constant do not; no virtual root. The prior spec kept 99 deliberately (`specs/completed/stage-3-supervision-integrity.md:117-118`) and scoped out hierarchy changes (`:42-48`). 87.9 % of the bundle's pair facts are 99. |
| 8 | implemented-differently | `data/download_data.py:352`, `:366` (exclusion text exploded per referenced code, so repeated); `data/create_triplets.py:227-229` (exclusion pairs generated as `UNRELATED` negatives); `supervision/selection.py:184-192` (rotating one-slot exclusion quota); `text_model/mixins/curriculum.py:244` (exclusions exempt from eligibility); `text_model/loss.py:78` (always in the denominator); `supervision/index.py:102-104` (symmetric, so the nine lineal references act as negatives); `graph_model/hgcn.py:1173-1178` (edges from structural relations only) | (a) repeated, not once; (b) absent; (c) inverted for negatives (a protected negative was a deliverable of the prior spec, `:270-271`, `:503-505`) and as specified for edges; lineal references untreated. |
| 9 | implemented-differently | `text_model/dataloader/tokenization_cache.py:39-40` (`'[EMPTY]'` placeholder), `text_model/encoder.py:140` (all four channels concatenated, no presence mask); `data/download_data.py:556-560`, `:629` (`.unique` picks an arbitrary child for the 14 four-digit codes), `:567`, `:582-584` (a five-digit code copies its lone child); no provenance column; `data/positive_sampling.py:78-81` (five-digit anchors get their lone child as positive); `metrics/hierarchy_structure.py:103-106`; `conf/config.yaml:95` (`max_length: 512`), `tokenization_cache.py:74` (title fixed at 24) | Placeholder instead of masking; arbitrary inheritance unrecorded; unary pairs in supervision and scoring; the rejected 512 window. The backbone's own files disagree (`sentence_bert_config.json` 256, `tokenizer_config.json` 512), so the (open) item stands. |
| 10 | implemented-differently | `data/compute_distances.py:230-231`, `data/create_triplets.py:100`, `data/supervision_bundle.py:676` (canonical orientation: 35 six-digit codes are never anchors; no ancestor positives); `text_model/naics_model.py:516-519`, `datamodule.py:656-665`, `conf/config.yaml:98` (pools of 24 pre-drawn negatives per pair; manifest `training_pairs` has 45,373,918 rows); `supervision/margins.py:84` (eligibility rule); `text_model/dataloader/streaming_dataset.py:197-199`, `conf/config.yaml:100-101` (inverse-distance draws); `text_model/mixins/curriculum.py:445-456` (hyperbolic k-means pseudo-labels) | Every removed mechanism is present; the 4,675 sibling-positive, grandchild-negative rows reproduce in the live bundle; no per-epoch cache of all code points. |
| 11 | implemented-differently | `text_model/mixins/loss.py:485-488` (six-term sum); `text_model/loss.py:47-128` (DCL), `:134-227` (distance matching, batch-normalized at `:221-222`), `:329-400` (pairwise preference); `mixins/loss.py:207-209` (radius penalty, zero by construction since r ≤ 2 < 10), `:318` (load balancing); `losses/level_radius.py:26-29` (level term; gradient ≈ 1e-7 under the cap, by probe); `naics_model.py:560-561` (margin logged only); `loss.py:58` (fixed float temperature); `text_model/mixins/optimizer.py:165-174`, `text_model/curriculum.py:107-115` (scheduler always built; mining active in epochs 6–9 of the shipped 10); no query term in `src/` | No task term; no learned scales; two code–code terms, neither listwise over all codes; every term Req 11 removes is present. Verification "No inert terms" fails today. |
| 12 | missing | `text_model/encoder.py:42` (dimension is the backbone hidden size, 384), `:81`, `:146` (`moe_projection` 1536→384) then `text_model/hyperbolic.py:102` (384→385): two stacked affine maps; no geometry switch in `src/` or `conf/`; `tests/unit/test_encoder.py:106` pins 384 | Dimension not configurable; Lorentz only. |
| 13 | missing | `text_model/hyperbolic.py:94` (`max_norm=2.0`), `:138-140` (hard rescale; saturated points pass ≈ 1e-7 gradient, by probe); `losses/level_radius.py:28` (level 2 targets sinh r = 0, the origin); three radial coordinates in use (`level_radius.py:26-27`, `text_model/hyperbolic.py:281`, `text_model/hard_negative_mining.py:60-62`); `utils/config.py:792`, `conf/config.yaml:138` (curvature is a config parameter threaded everywhere); `graph_model/hgcn.py:85-88`, `:106-107` (per-layer parameter detached by `.item()`); `text_model/hyperbolic.py:234-267` (manifold check at tolerance 1e-3; no radius-resolution check) | E1 and E3 premises confirmed by probe. |
| 14 | missing | `text_model/encoder.py:56-61` (four full backbone loads, each with its own LoRA), `:122-123`, `:140` (concatenation into the MoE); `text_model/moe.py:118-125`, `conf/config.yaml:124-128` (MoE is the only fusion); `conf/config.yaml:117` (one `base_model_name`; no candidate list, no freeze flag) | No shared encoder, field markers, masked fusion, query path or backbone selection. |
| 15 | missing | `graph_model/hgcn.py:1316`, `conf/graph.yaml:100` (single seed, one arm); no smoothing, shuffle control or matched-compute tooling in `src/`; `conf/graph.yaml:20` (`tangent_dim: 31`) against the 385-wide text export (`cli/commands/training.py:362-372`; by probe `Linear(31, 31)` rejects it; no width check at `hgcn.py:1319-1327`) | Arm D cannot run on the text stage's output as configured (the Req 16 fix); arms B, C and E do not exist; the fixed-threshold gate is the only comparison tooling. `conf/graph.yaml:15` names the bundle only in the unpushed local commit. |
| 16 | implemented-differently | `graph_model/hgcn.py:82` (`Linear(dim, dim)`: no learned projection, but width 31 ≠ 385); `:303` (node states are a free `nn.Parameter` seeded from the text points; no retention term); `:1246-1253` (graph export of all rows); `cli/commands/training.py:788-800` (text export only behind an interactive `typer.confirm`); `cli/__init__.py:44-48` (no export command) | No projection map, as specified, but no shared space either; per-stage tables exist, no defined final deliverable, no standalone export. |
| 17 | implemented-differently | `conf/graph.yaml:49-53`, `graph_model/hgcn.py:1206`, `:1211-1218` (child, grandchild, great-grandchild and sibling edges, bidirectional, plus self-loops); `hgcn.py:135`, `:143` (edge weight enters twice), `:1198-1199`; `hgcn.py:84`, `:118-120` (tangent LayerNorm; output radius ≈ 5.4 whatever the input, by probe); no distillation or residual to the text points; `hgcn.py:249`, `conf/graph.yaml:38-39` (hinge temperature); `hgcn.py:673-710`, `graph.yaml:73` (adaptive margin on); `hgcn.py:181-182`, `:205-207`, `graph.yaml:24` (uncertainty weights on); `hgcn.py:521-572`, `:595-629`, `graph.yaml:64` (in-file three-phase curriculum filters on); `graph_model/curriculum/*` (four-phase controller package, unwired: `hgcn.py:23` imports only `resolve_graph_config`) | Every mechanism Req 17 would replace is present and on by default. Untouched until Req 15 decides (user adjudication Q2); not applicable before Stage 10. |
| (none) | in-code-but-not-in-spec | `text_model/naics_model.py:618-693`, `supervision/mode.py:35-40` | Legacy containment mode with a second `training_step`. Deleted in Stage 7 (D2). |
| (none) | in-code-but-not-in-spec | `data/supervision_bundle.py`, `supervision/checkpoints.py`, `conf/config.yaml:9-12` | The supervision bundle contract (manifest, contract version, exact-resume checkpoint contract). The spec's Staleness note assumes it without naming it; Stages 2 and 5 version it. Unrecorded decision to fold back: the bundle stays the single authority for structure and text. |
| (none) | in-code-but-not-in-spec | `data/compute_relations.py:107-158`, `conf/data/supervision.yaml:8-23`, `data/create_triplets.py:142-153` | Fourteen named kinship relations plus `cross_sector` as a second structural axis with its own margin. Names survive as edge types and labels only; the margin axis leaves in Stage 7 (D5). |
| (none) | in-code-but-not-in-spec | `losses/level_radius.py:26-29` via `graph_model/hgcn.py:744-749` | The graph stage's level-radius term (sectors at the origin). Req 17 does not list it. Handled in Stage 10 per D3. |
| (none) | in-code-but-not-in-spec | `tools/embeddings_verification.py:33-35`, `cli/commands/tools.py:257` | The `verify-stage4` gate with fixed thresholds; Req 5 replaces them (Stage 4). |
| (none) | in-code-but-not-in-spec | `metrics/graph.py:236-439` | Unwired "taxonomy tasks" suite (parent identification, k-means ARI and NMI, sector logistic regression): the methodology's other never-run benchmark. No requirement keeps it; retire with Stage 4's diagnostics. |
| (none) | in-code-but-not-in-spec | `graph_model/curriculum/*` (about 2,400 lines), `tests/unit/test_graph_curriculum.py` | Unwired four-phase controller, event bus, MACL, samplers and analyzer. Held until Req 15 (user adjudication Q2); leaves in Stage 11 either way. |
| (none) | in-code-but-not-in-spec | `text_model/mixins/validation.py:167-170`, `conf/config.yaml:130` | Structural statistics on a 300-code random subsample; Req 6 names no population. Stage 4 computes over all 2,125 codes (methodology S4 limitation 4). |
| (none) | in-code-but-not-in-spec | `data/download_data.py:291`, `:312-324` | 68 cross-reference rows without a code reference are dropped, and exclusion text is also harvested from "Excluded" description paragraphs; the spec counts 43 non-redirection rows (4,601 − 4,558). Stage 5 reconciles the two counts. |
| (none) | in-code-but-not-in-spec | `text_model/dataloader/tokenization_cache.py:74` | Title channel fixed at 24 tokens; Req 9's window policy covers it in Stage 5. |

## Open questions

- **D8's final tie-break (Req 5; Stage 4).** Req 5's tie order ends on "the higher
  regressor-panel estimate", and under D8 the regressor panel has two regime estimates. Which one
  breaks the tie (seen, held-out or their mean) is unresolved. Stage 4's planning session asks the
  user before it builds the tie order.
- **Each panel's decision statistic (Req 5; Stage 4).** Req 5 fixes a δ and an interval per
  panel but names no statistic. Req 3 lists four outcome metrics, D6 picks validation MRR only
  for within-run selection, and Req 1's regressor gain names no loss. Stage 4's planning session
  asks the user alongside the tie-break. Stage 3 keeps its output at row grain so that either
  answer can be computed (second resume, 2026-09-24).

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
so it stands alone although it produces no software.

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
hardcoded outside `data roles`.

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

- [ ] Stage 3: Regressor panel
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

- [ ] Stage 4: Decision rule and diagnostics
      Objective: Implement Req 5's decision procedure and record, and demote the structural
      statistics to stratified diagnostics that nothing selects on.
      Spec: Req 5; Req 6; Req 1 (taxonomy agreement never a selection criterion); Verification
      "Decision records", "Diagnostics"; D8.
      Gap closed: Req 5 (except the reference configuration and δ, which Stage 7 supplies);
      Req 6; Req 1 (selection-criterion half).
      Consumes: Stage 2's `DecodingResult.per_query`, resampled by code (finding, section 6),
      and Stage 3's per-row predictions, resampled by four-digit group in each regressor regime.
      No trained arms yet: the tooling is exercised on synthetic scores.
      Produces: Decision tooling over D8's three panels: each panel's statistic (Open
      questions), paired resampling over each panel's unit with seeds nested, 95 %
      non-inferiority and 98⅓ % superiority intervals, δ as a stated multiple of a reference's
      across-seed standard deviation, the non-dominated set, the tie order (its final tie-break:
      Open questions), and a decision-record schema that carries the selection-log records of the
      runs it compares (the log is gitignored and dies with its worktree or Lambda instance) and,
      per arm and seed, immutable references (path and content hash) to the encoder checkpoint
      and the 2,125-code table; a seed-sweep driver that runs a configuration for N seeds,
      collects every panel's per-unit scores, and keeps the referenced artifacts until Stage 12
      (a Lambda instance loses them at termination: `specs/lambda-remote-workflow.md`); the
      diagnostics report over all 2,125 codes
      (sector-separation AUC, within-sector rank correlation averaged over sectors and queries,
      MAP over ancestors, NDCG with integer lowest-common-ancestor grades, the Pearson
      statistic without the cophenetic name, unary pairs excluded from parent retrieval);
      `verify-stage4`'s fixed thresholds retired and the unwired taxonomy-tasks suite removed;
      structural statistics off progress bars and headlines.
      Exit: On synthetic arms with known effects on D8's three panels, the tooling adopts and
      rejects per the rule and writes records with every field Verification "Decision records"
      lists, plus the selection-log records of their runs and the artifact references of every
      arm and seed; the diagnostics report contains only
      Req 6's statistics, stratified as listed, with no threshold and no pass/fail; no monitor,
      gate or headline reads a structural statistic.
      ROUTING: writing-plans

- [ ] Stage 5: Supervision target and text
      Objective: Rebuild the supervision bundle around the tree metric D*, directed
      redirections and Req 9's text-construction rules.
      Spec: Req 7 (D*; the IC ablation waits for Stage 9); Req 8(a), 8(c) generation side,
      lineal references, reserved slot removed; Req 9 (masking data side, inheritance, unary
      pairs, input window for the current backbone); Verification "Target", "Exclusions"
      (generation half), "Text" (data half), "Backbone input window" (current checkpoint); D5.
      Gap closed: Req 7 (except the IC ablation); Req 8 (a, lineal, generation side of c);
      Req 9 (except the model-side mask and the channel-presence ablation).
      Consumes: Stage 2's index-entry roles (the examples channel holds examples-role entries
      only). Stage 2 shipped them as an optional `index_roles` member under
      stage3-supervision-v1 and the rebuild in `data preprocess`, but built no bundle: this
      stage's contract version makes the member required, and its rebuild is the first bundle
      carrying both. Stage 2's held-out leakage check (`panels/leakage.py`), which the bundle
      build runs but which reaches activity phrases only by splitting `excluded` at `--`. The
      current bundle contract (`data/supervision_bundle.py`, `supervision/artifacts.py`) as the
      thing to version. The current backbone's own documentation for its trained window.
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
      exceeded it, and no input exceeds it.
      ROUTING: writing-plans

- [ ] Stage 6: Shared encoder and low-dimensional projection
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
      Consumes: Stage 5's bundle (null channels, window policy). The current objective and
      dataloader, unchanged, as an interim training harness only. Stage 2's `QueryCodeEncoder`
      protocol and `OutcomePanel.score` for a first live validation-split reading (finding,
      section 6). The scorer's `lorentz` distance assumes curvature −1, but the interim harness
      learns its curvature, so Stage 6 passes a curvature-aware distance (plan 4's deferred item).
      Produces: One encoder module implementing `QueryCodeEncoder`, with code and query
      embeddings in the same space;
      fusion options masked mean, attention pooling and MoE (MoE ablation-only); a configurable
      embedding dimension in {8, 16, 32}; a standalone export command writing the 2,125-code
      table in Req 2's form; the per-channel adapter copies and the load-balancing term deleted
      with the default fusion.
      Exit: A query embeds through the same encoder as a code (test); masking an absent
      channel's input leaves the output unchanged (test); exactly one affine map sits between
      the encoder and the point; the model trains under the interim objective at dimension 16
      and the export command writes the 2,125-code table; Stage 2's scorer returns live
      validation-split numbers under the trained curvature.
      ROUTING: brainstorming

- [ ] Stage 7: Objective, anchors and live radius (the reference configuration)
      Objective: Replace the six-term objective and its sampling machinery with Req 11's three
      terms over all codes on a live radius, wire selection to the validation splits, and fix
      δ per panel from the reference configuration's seeds.
      Spec: Req 11; Req 10; Req 13; Req 8(b), 8(c) training side; Req 4 (the in-sample loss
      selects nothing; monitors read validation splits); Req 5 (reference configuration, δ);
      Req 6 (no structural monitor); Verification "No inert terms", "Coverage", "Radius",
      "Exclusions" (training half), "Text" (unary pairs absent from positive supervision),
      "Selection hygiene" (validation half); D2, D5, D6, D8.
      Gap closed: Req 11; Req 10; Req 13; Req 8 (b, training side of c); Req 4 (monitors);
      Req 5 (reference configuration and δ).
      Consumes: Stage 6's encoder and query path; Stage 5's D*, redirection table and unary
      flags; Stage 2's query splits, scorer and selection log (the log's path is
      `OutcomePanelConfig.selection_log`, `logs/selection_log.jsonl`: gitignored, so a
      worktree's log goes with the worktree, and Stage 4's decision records keep its records).
      The test split stays sealed: Stage 12 opens it, as the finding's section 6 erratum says.
      Stage 3's panel; Stage 4's seed-sweep driver, decision tooling and δ procedure.
      Produces: The reference configuration (hyperbolic, dimension 16, current backbone, shared
      encoder) with a query→code task term over training queries and activity phrases (the
      referencing code always scored), a listwise code–code term with graded targets from D*
      over all 2,125 codes through a per-epoch code-point cache, a radial term with a virtual
      root and sectors at positive radius, learned logit scales, a gradient-passing bound in
      place of the cap, one radial coordinate, curvature fixed at 1 with no parameter; deleted:
      radius penalty, distance matching, pairwise preference, curriculum and mining rules,
      false-negative clustering, the logged margin, pre-drawn tuples, eligibility rules,
      inverse-distance draws, the exclusion quota, the relation margin axis (D5), and legacy
      containment (D2); monitors for checkpointing, early stopping and learning rate reading
      the validation query split's MRR (D6); a decision record fixing δ for each of D8's three
      panels from at least 5 seeds.
      Exit: On a real batch every objective term has a nonzero gradient (test); on a trained
      run the gradient with respect to radius is nonzero, radii vary within each level, the 20
      sectors sit at distinct positive radii, and manifold validity and distance resolution
      hold at the largest observed radius; every code is an anchor and the code–code term
      covers all 2,125 codes at every step with no pre-drawn tuples (test); no exclusion pair
      acts as a code–code negative and each cross-reference query scores its referencing code
      (test); unary pairs are absent from positive supervision; the selection log shows only
      validation splits read; a decision record fixes δ for each of D8's three panels from at
      least 5 seeds.
      ROUTING: brainstorming

- [ ] Stage 8: Geometry × dimension
      Objective: Implement the Euclidean and spherical arms and run the crossed nine-cell
      comparison under Req 5.
      Spec: Req 12; Req 5 (several arms, ties); Req 2 (export form per arm); Verification
      "Geometry × dimension"; D8; D9.
      Gap closed: Req 12 (geometry arms and the decision).
      Consumes: Stage 7's reference configuration and δ; Stage 4's tooling and seed-sweep
      driver; Stage 6's configurable dimension; Stage 2's scorer, whose registered distances are
      `euclidean`, `cosine` and `lorentz` at curvature −1 (`panels/decoding.py`).
      Produces: Geometry as a configuration factor with per-arm distance, decoding and export
      (the radial term only in the hyperbolic arm); the nine-cell decision record; the selected
      cell as the reference for Stage 9.
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
      Consumes: Stage 8's selected cell; Stage 6's fusion options; Stage 4's tooling.
      Produces: A decision record per factor; the selected text stage (arm A for Stage 10); the
      trained window recorded per candidate with each channel's overflow share; if IC is
      adopted, the bundle's target and the diagnostics' relevance grades switched to it.
      Exit: Each factor has a Req 5 record with at least 5 seeds per arm, and the frozen-encoder
      control ran; the adopted backbone's window is recorded from its own documentation with
      no input beyond it.
      ROUTING: brainstorming

- [ ] Stage 10: Graph-stage decision experiment
      Objective: Run arms A–E on the selected text stage and decide whether the graph stage
      stays, with arm D repaired only to run in the text stage's space.
      Spec: Req 15; Req 16; Req 4 (graph-stage rows: no private validation tail, no last-epoch
      export, selected like every other arm); Verification "Decision experiment",
      "Deliverable"; D3; D8.
      Gap closed: Req 15; Req 16 (composition and deliverable); Req 4 (graph-stage rows).
      Consumes: Stage 9's selected text stage; Stage 4's tooling and seed-sweep driver; Stage
      6's export command; Stage 5's bundle as the graph stage's structural input.
      Produces: Arm B (matched-compute continuation of the text stage); arm C (parameter-free
      smoothing toward the parent-and-children mean, α tuned on validation); arm D (the graph
      stage at the text stage's dimension, exponential and logarithmic maps per the selected
      geometry, its level-radius term per D3, checkpoint selected on validation, no private
      tail, no last-epoch export); arm E (text-shuffle control, run only if D wins); the
      keep-or-drop record; the deliverable, a 2,125-code table from the selected arm.
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
      package, `conf/graph.yaml`, the graph docs and every graph-stage code path removed, with
      the deliverable defined by the text stage alone and arm D's referenced artifacts kept for
      Stage 12 (Req 16 scores its code points against the text stage's queries, so no graph code
      is needed).
      Exit: If kept: a Req 5 record per repair and a final 2,125-code table from the repaired
      stage. If dropped: no graph-stage code path remains, the suite passes, the deliverable is
      the text stage's table, and arm D's referenced artifacts still match their hashes.
      ROUTING: brainstorming if kept (the retention and normalization choices are open);
      writing-plans if dropped

- [ ] Stage 12: Sealed final evaluation
      Objective: Open each sealed test split once, for the final configuration and the
      comparisons recorded for it, and report Req 1's two estimands on sealed held-out data.
      Spec: Req 4 (the test splits opened once); Req 1 (measured on sealed held-out data); Req 5
      (intervals); Verification "Selection hygiene" (opening half); D8.
      Gap closed: Req 4 (the opening); Req 1 (the sealed estimates).
      Consumes: The final configuration and its 2,125-code table (Stage 11's, or Stage 10's if
      the graph stage was dropped); the decision records of Stages 7–11 with the selection-log
      records and artifact references they carry (Stage 4's schema); by those references, the
      per-seed encoder checkpoints and 2,125-code tables of the final configuration and of every
      arm in its recorded comparisons, since sealed queries were never embedded; Stage 2's
      `OutcomePanel.open_test`; Stage 3's sealed outer sets and the logged opening that guards
      them; Stage 4's tooling and seed-sweep driver.
      Produces: One logged opening per sealed set (the outcome test queries and each regressor
      regime's outer set, per D8); sealed estimates with D8's intervals for the final
      configuration and each comparison recorded for it; a written finding.
      Exit: The selection-log records, gathered from the decision records and this stage's own,
      show exactly one opening per sealed set, under the fingerprint Stage 2 or Stage 3
      committed, after every validation read that selected anything; every sealed estimate comes
      from referenced artifacts whose hashes match the records; the finding reports each sealed
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
