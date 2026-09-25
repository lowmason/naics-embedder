# Decision Rule and Diagnostics Implementation Plan

**Status: COMPLETE (2026-09-25)** — executed via subagent-driven-development; deferred items in specs/deferred_items.md (seven: the final review's coverage and hardening groups, due before Stage 7's first real decision and its Lambda sweep; small items for Stages 6 and 10; a CLI-wide polars catch; the tie rule; and a pre-existing curvature bug in `text_model/hyperbolic.py`)

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-embedding-roadmap.md, Stage 4 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Build roadmap Stage 4. The first part is Req 5's decision procedure over D8's three
panels:

- each panel's statistic (D10);
- paired resampling with seeds nested;
- δ from a reference arm;
- the intervals, the non-dominated set and the tie order, whose final tie-break is D11's;
- a decision record that carries its runs' selection-log records and immutable artifact
  references;
- a seed-sweep driver.

The second part is Req 6's diagnostics report. The third demotes the old structural statistics:
`verify-stage4`, the QCEW benchmark and the unwired taxonomy-tasks suite are removed, and the
statistics come off every progress bar and headline.

**Architecture:** A new package, `naics_embedder.decision`, holds seven modules:

- `scores`: each seed's per-unit scores on the three panels, in one long frame.
- `resampling`: the paired two-stage bootstrap. Units are shared by every arm, and each arm's
  seeds are resampled inside every unit resample.
- `rule`: the two intervals, the verdicts, the non-dominated set and the tie order.
- `records`: Pydantic arm, margin and decision records, written once as JSON.
- `store`: a content-addressed artifact store that the records' references point into.
- `decide`: every guard, `fix_margins` and `decide`.
- `sweep`: `run_seed_sweep`, which runs a configuration for N seeds through the real panel
  interfaces.

`metrics/diagnostics.py` computes Req 6's report from a table in the export form. Three `tools`
commands expose all this: `margins`, `decide` and `diagnostics`. Stage 2's and Stage 3's panels
gain a `detail` argument, so each read names the run it scores. The text-only provenance now
records the `matrix_fingerprint` that the selection log names the table by.

Four things are removed:

- `metrics/qcew.py` and the downstream suite in `metrics/graph.py`;
- `tools verify-stage4` and `tools/embeddings_verification.py`.

Structural statistics come off the text and HGCN progress bars, `tools visualize` and the `train`
banner. Their logged values stay until Stages 7 and 10/11.

**Tech Stack:** Python 3.10 and 3.12 (CI runs both); numpy (`default_rng` multinomial draws);
polars; scipy (`rankdata`); torch (the distances); pydantic (records, config); typer and rich
(the CLI); pytest with xdist; ruff and yapf.

## Global Constraints

Every task's requirements include this section.

### The spec (`specs/naics-embedding.md` at d9126ce), verbatim

- Req 1: "Agreement with the taxonomy becomes an intrinsic diagnostic and never a selection
  criterion (Req 6; ChatGPT C15; Claude C27; Gemini C8; methodology Cross-component 1)."
- Req 2, the rejected definition: "Running the benchmark as defined (Gemini remediation 10)
  (rejected: its comparators cannot isolate the embedding's contribution (methodology S4
  limitation 6))."
- Req 2: "**Comparators** share the downstream model and a tuned penalty (ChatGPT C15; Claude
  C28):"
  - "six-digit one-hot indicators;"
  - "ancestor indicators at levels 2–5;"
  - "a text-only representation reduced to the same dimension;"
  - "covariates only;"
  - "covariates plus each representation."
- Req 5:
  - "**Seeds.** Each arm runs at least 5 seeds (Claude C20)."
  - "**Pairing.** Differences Δ = A − B are paired: both arms are scored on the same resample of
    the evaluation unit, with seeds nested within the resample (Claude on S4-Q4; ChatGPT on
    S4-Q4). The unit is codes, with their queries, for the outcome panel, and four-digit-parent
    groups for the regressor panel."
  - "**Margin.** Each panel's non-inferiority margin δ is fixed before any arm runs, as a stated
    multiple of the reference configuration's across-seed standard deviation on that panel
    (Claude C22)."
  - "**Adoption.** A is adopted when it is non-inferior on both panels and superior on at least
    one (Claude C22; user adjudication Q1)." … "Non-inferior: the lower bound of the 95%
    interval on Δ exceeds −δ." … "Superior: the 97.5% interval excludes zero. Two panels give
    two chances to adopt, so each gets half the error rate."
  - "**Several arms.** Some decisions have more than two arms: the nine geometry × dimension
    cells, the backbones, and the graph-stage arms A–D. The survivors are the arms no other arm
    dominates, and the tie order below picks among them. If dominance cycles and every arm is
    dominated, the tie order picks among all the arms of the decision." … "The extra pairwise
    comparisons within a decision are not corrected for multiplicity. That is a stated choice,
    not an oversight: the tie order toward the simpler arm is the guard against a spurious win."
  - "**Ties.** When A is not adopted over B, the simpler configuration stands. Simpler means
    fewer components (stages or post-processing steps), then lower dimension, then
    non-hyperbolic geometry. Any tie left after that goes to the higher regressor-panel estimate,
    because that is the designed purpose (user adjudication Q1)."
  - "**Scope of the rule.** It replaces the fixed acceptance thresholds (methodology
    Composition; methodology S4 procedure; ChatGPT C14; Gemini C14) and gates the graph stage
    (Req 15)."
- Req 6: "They are reported, but never selected on and never used as a headline (ChatGPT C5;
  Claude C21; Gemini C11):"
  - "sector separation, as the AUC of distance between same-sector and cross-sector pairs;"
  - "within-sector rank correlation, averaged over sectors and over queries;"
  - "mean average precision over ancestors;"
  - "NDCG with integer grades from lowest-common-ancestor depth."

  After the list: "The Pearson statistic drops the "cophenetic" name, since no dendrogram is
  involved (Claude C21; methodology S4 procedure). The 522 unary pairs (Req 9) are excluded from
  parent retrieval (Claude C22). IC-graded relevance (Gemini C11) follows only if Req 7's
  ablation adopts IC."
- Req 7: "The structural target $D^{\ast}$ is tree path length with a virtual root above the 20
  sectors (Claude C7; ChatGPT on S1-Q2):"
  - "$D^{\ast}_{ij} = h_i + h_j$ within a sector;"
  - "$D^{\ast}_{ij} = \lambda(i) + \lambda(j) - 2$ across sectors."
- Req 9: "The 522 unary pairs are five-digit industries whose only child is their six-digit
  code. They leave positive supervision and parent-retrieval scoring, and all 2,125 codes stay in
  the deliverable".
- Verification "Decision records": "Every adopted change has a record with: its arms, at least 5
  seeds each, the δ per panel fixed before the runs, the 95% non-inferiority and 97.5%
  superiority intervals, and, for decisions with more than two arms, the non-dominated set (Req
  5)."
- Verification "Diagnostics": "Structural statistics appear only in the diagnostic report,
  stratified as Req 6 lists."

### The roadmap (`specs/naics-embedding-roadmap.md` at 795080e), verbatim

- Stage 4 Objective: "Implement Req 5's decision procedure and record, and demote the structural
  statistics to stratified diagnostics that nothing selects on."
- Stage 4 Gap closed: "Req 5 (except the reference configuration and δ, which Stage 7 supplies);
  Req 6; Req 1 (selection-criterion half); Req 2 (the rejected definition Stage 3 left)."
- Stage 4 Consumes: "Stage 2's `DecodingResult.per_query`, resampled by code (finding, section
  6), and Stage 3's per-row predictions, resampled by four-digit group in each regressor regime
  (finding `specs/findings/regressor-panel-splits.md`, section 6): D8's two regressor panels are
  the level-6 cells, a validation read scores each of its rows once per repeat (five), and
  `group` is the code itself at levels 2–3. Plan 5's deferred note on those repeats and on the
  held-out regime's feature years. Stage 3's selection-log reads, which name the arm's table and
  its text-only table by `matrix_fingerprint`, not by file hash. The four QCEW slices the panel
  reads from `qcew_dir` under pinned hashes (`conf/data/regressor_panel.yaml`) stay outside the
  repo, so they must be present wherever the driver scores that panel: a Lambda instance has
  only what is uploaded. No trained arms yet: the tooling is exercised on synthetic scores."
- Stage 4 Produces: "Decision tooling over D8's three panels: each panel's statistic (D10),
  paired resampling over each panel's unit with seeds nested, 95 % non-inferiority and 98⅓ %
  superiority intervals, δ as a stated multiple of a reference's across-seed standard deviation,
  the non-dominated set, the tie order (its final tie-break: D11), and a decision-record schema
  that carries the selection-log records of the runs it compares (the log is gitignored and dies
  with its worktree or Lambda instance) and immutable references (path and content hash) to each
  arm's text-only table and its provenance file, whose backbone, revision, descriptions hash and
  window the record checks against the arm's (D9), and, per arm and seed, to the encoder
  checkpoint and the 2,125-code table, each table's reference also carrying the
  `matrix_fingerprint` the log names it by; a seed-sweep driver that runs a configuration for N
  seeds, collects every panel's per-unit scores, and keeps the referenced artifacts until Stage
  12 (a Lambda instance loses them at termination: `specs/lambda-remote-workflow.md`); the
  diagnostics report over all 2,125 codes (sector-separation AUC, within-sector rank correlation
  averaged over sectors and queries, MAP over ancestors, NDCG with integer
  lowest-common-ancestor grades, the Pearson statistic without the cophenetic name, unary pairs
  excluded from parent retrieval); `verify-stage4`'s fixed thresholds retired; the unwired
  taxonomy-tasks suite removed, and `metrics/qcew.py` with its re-exports in
  `metrics/__init__.py` and `graph_model/__init__.py`, its API page (`docs/api/qcew_metrics.md`
  and its `docs/.nav.yml` entry: the docs build runs only on main) and its tests, keeping the
  `graph_dataset` case that `tests/unit/test_graph_downstream_evaluation.py:190` parametrizes
  beside it; structural statistics off progress bars and headlines."
- Stage 4 Exit: "On synthetic arms with known effects on D8's three panels, the tooling adopts
  and rejects per the rule and writes records with every field Verification "Decision records"
  lists, plus the selection-log records of their runs and the artifact references of every arm
  and seed, text-only tables and their provenance included; the diagnostics report contains only
  Req 6's statistics, stratified as listed, with no threshold and no pass/fail; no monitor, gate
  or headline reads a structural statistic; neither `metrics/qcew.py` nor the taxonomy-tasks
  suite remains."
- D6: "Decision: the validation query split's MRR (Req 3); both panels are used only between
  configurations, under Req 5."
- D8: "Decision: each regime counts as a panel under Req 5, which then has three: the outcome
  panel and the two regressor regimes. Adoption needs non-inferiority on all three (the 95 %
  interval, unchanged) and superiority on at least one; each panel gets its own δ and a third of
  the error rate, so superiority reads the 98⅓ % interval. This supersedes the 97.5 % that Req 5
  and Verification "Decision records" name; the Completion audit reads it as this recorded
  deviation, not an unmet requirement."
- D9: "Decision: the arm's own backbone, frozen, embedding each code's text, reduced by PCA to
  the arm's dimension."
- D10: "Decision: the outcome panel's is per-query MRR, resampled by code with its queries, the
  statistic D6 already selects checkpoints on. Each regressor regime's is the out-of-sample mean
  squared error of the `covariates+embedding` comparator on log employment at level 6, each
  row's squared error averaged over its repeats (five on a validation read) before resampling by
  group; its Δ is oriented so that a positive value favours A (B's error minus A's). The sparse
  comparators never read the arm, and their folds depend only on the regime, level, repeat and
  group, so the gain over a sparse encoding (over one-hot in the seen regime, over ancestors in
  the held-out regime) gives the same paired Δ; that gain is reported for Req 1. The text-only
  comparator is no baseline, since it changes with each arm's dimension and backbone. Every
  other Req 2 comparator and Req 3 metric is still reported."
- D11: "Decision: the held-out regime's, read as its gain over ancestors, so the higher estimate
  is the lower error. For a held-out code, one-hot predicts only the intercept and ancestors help
  only through levels 2–3, so that regime is where the embedding's value over sparse encodings
  is tested."
- Section 6 of `specs/findings/outcome-panel-splits.md`: "**Stage 4** resamples by code from
  `DecodingResult.per_query`, one row per query with its code."
- Section 6 of `specs/findings/regressor-panel-splits.md`: "D8's two regressor panels are
  `regressor_seen` and `regressor_heldout` at level 6. It resamples by `group` within a panel and
  level: the four-digit parent at levels 4–6 (Req 5's unit), the code itself at levels 2–3.
  Every arm's validation rows share folds, so paired differences align row for row."

### Decisions already made (do not re-ask)

The user answered three questions at planning (2026-09-24):

1. **Seeds are resampled.** Each bootstrap replicate draws each arm's seeds with replacement,
   independently per arm, inside the unit resample every arm shares. The intervals therefore
   carry training randomness as well as unit sampling.
2. **`verify-stage4` is deleted.** Its command, `tools/embeddings_verification.py`, its tests and
   its doc sections go. The new `tools diagnostics` reports Req 6's statistics for any table, so
   run it on a table before and after refinement. Stage 10 decides the graph stage under Req 5.
3. **Bars and headlines only.** Structural statistics come off both stages' progress bars:
   - text: `val/cophenetic_correlation` and `val/median_distortion`;
   - HGCN: `val/cophenetic_correlation`, `val/ndcg@10` and `val/relation_accuracy`.

   They also come out of `tools visualize`'s plots, tables, grades and advice, and this plan
   adds the `train` banner, whose list announced them as the evaluation. Their logged values
   stay until Stage 7 and Stages 10/11 rework validation. Plan completion writes that leftover
   into those entries.

This plan's own decisions are stated here so that no reviewer needs to re-derive them:

- **Scores (D10).** One long frame per seed, with columns `panel`, `statistic`, `unit`, `item`,
  `feature_year` and `value`.
  - Outcome panel: an item is a validation query id and its unit is its true code. Every Req 3
    metric is a statistic: `top1`, `mrr`, `hit_at_1`, `hit_at_5`, `hit_at_10` and `lca_level`.
    The decision statistic is `mrr`.
  - Regressor regimes: an item is a level-6 validation row, `code/feature_year`, and its unit is
    the row's four-digit group. Every Req 2 comparator is a statistic. Its value is the row's
    squared error averaged over the read's repeats, and a row must have exactly `repeats`
    predictions under each comparator. `repeats` is the panel's `fit_settings`, 5 on a real
    read. The decision statistic is `covariates+embedding`.
- **Paired bootstrap (user decision 1).** Each replicate works in two stages:
  - It draws a panel's units with replacement: one multinomial draw per replicate, shared by
    every arm and seeded `[bootstrap_seed, stream(panel)]`.
  - Inside that draw it draws each arm's seeds, seeded
    `[bootstrap_seed, stream(panel), stream(arm)]`.

  `stream(name)` is the first 16 hex digits of the name's sha256. A replicate's statistic is
  `Σ_s m_s (C·S_s) / (n_seeds (C·N))`: C are the unit counts, m the seed counts, S_s seed s's
  per-unit sums and N the items per unit. The point estimate is the mean over seeds of each
  seed's statistic on every unit. An arm's replicates are the same in every comparison it
  enters. `conf/data/decision.yaml` sets 10,000 replicates, seed 20260924 and a floor of 5 seeds
  that the config cannot lower.
- **Intervals and verdicts.**
  - Intervals are two-sided percentile intervals: 95 % for non-inferiority, and `1 − 0.05/3`
    (98⅓ %) for superiority (D8).
  - Δ is `orientation × (A − B)`, with orientation +1 for MRR and −1 for MSE, so Δ > 0 favours A.
  - Non-inferior: the 95 % lower bound is above −δ. Superior: the 98⅓ % lower bound is above 0.
  - Adopted: non-inferior on all three panels and superior on at least one.
- **Several arms.** Every ordered pair is compared. The survivors are the arms no other arm is
  adopted over. If no arm survives, dominance cycled: the record says so, and the tie order picks
  among all the arms.
- **Tie order.**
  - The key is `(components, dimension, geometry == 'hyperbolic', −held-out gain)`, ascending.
  - The held-out gain is the mean over seeds of MSE(`covariates+ancestors`) −
    MSE(`covariates+embedding`) on `regressor_heldout` (D11).
  - The chosen arm is the first survivor in that order.
  - Two arms equal on every key raise `TieUnresolvedError`. The command reports it and writes no
    record.
- **δ.** `fix_margins` sets each panel's δ to the multiple times the reference arm's across-seed
  standard deviation (ddof 1) of the panel's per-seed statistic. It refuses:
  - a multiple of 0 or less;
  - a standard deviation of 0;
  - a reference that fails the arm checks.

  Choosing the multiple is Stage 7's job. This plan's tests use multiples of 1 to 5 on synthetic
  arms.
- **Guards.** A decision refuses an arm when:
  - it has fewer than 5 distinct seeds, or a repeated seed or run id;
  - its text-only provenance's backbone, revision, descriptions sha256 or window differs from the
    arm spec's (D9);
  - a stored artifact no longer hashes to its reference;
  - a run's log records are not one validation read per panel, each naming the run in
    `detail.run`;
  - a read names another table than the store recorded (`detail.table` on the outcome panel,
    `detail.arm` and `detail.text_only` on the regressor panels), or another panel fingerprint.

  It also refuses a set of arms, the margins' reference included, that read other panels or fit
  settings. And it refuses a non-reference run whose first read precedes the margins' `fixed_at`.
- **Records.** Records are Pydantic models (`extra='forbid'`, frozen) written once as JSON:
  `write_record` refuses to overwrite. A decision record holds:
  - the question and the statistics' definitions;
  - the settings, the margin record and every arm record;
  - every comparison, with Δ, both intervals, δ and the verdicts;
  - the non-dominated set, whether dominance cycled, the tie order, the held-out gains and the
    chosen arm;
  - per arm, every Req 3 metric, every comparator's MSE, the gains over the sparse comparators
    with 95 % intervals, and the held-out regime by feature year.

  An arm record holds its spec, the text-only reference with its provenance, the store root, the
  panel fingerprints, the fit settings and, per seed, references to the checkpoint, the table
  (with its `matrix_fingerprint`), the scores, the decoding and the predictions, plus that
  seed's statistics and log records.
- **Store.** Files live under `objects/<sha[:2]>/<sha>/<name>` below a root outside any worktree.
  References are relative to the root, with the sha256 and size. A file is written once through
  a temporary file and `os.replace`, and verified on every read. `put_text_only` refuses a
  provenance that is missing, describes another file or names another `matrix_fingerprint`.
- **Log names.** The first plan-5 deferred item is resolved: the text-only provenance records
  `matrix_fingerprint` beside `table_sha256`. The panels take a caller's `detail`, merged into
  the read's logged detail, but it can never replace a key the panel logs itself.
- **Seed-sweep driver.** `run_seed_sweep` checks D9 before any read. For each seed it:
  - calls the runner, which returns the checkpoint, the table in the export form, the encoder
    and its distance name;
  - reads each panel's validation split once: the outcome panel with the encoder, and each
    regressor regime at level 6;
  - logs `run`, `arm_name` and `seed` on every read, and `table` on the outcome read;
  - stores every artifact and returns the arm record.

  Run ids are `<arm>/seed-<seed>/<uuid4 hex>`. The driver runs on fixture panels only, until
  Stage 6 adds the export and the query path.
- **Diagnostics (Req 6).** The report covers every code of a table in the export form, which
  must hold exactly the codebook's codes.
  - D* comes from the codes' own lineage: `depth_i + depth_j − 2·depth_LCA`, with sectors at
    depth 1 under the virtual root and combined sectors as one (`panels.decoding.code_lineage`).
  - Distances are Euclidean, cosine for spherical, and for hyperbolic the geodesic distance
    after the exponential map at the origin, at curvature `−c`.
  - Sector separation is the AUC that a cross-sector pair lies farther apart than a same-sector
    pair, with ties counting half.
  - Within-sector rank correlation is computed per query: Spearman over the other codes of its
    sector. It is reported as the mean over queries and as the mean over sectors.
  - MAP over ancestors is computed for each non-sector query, and also by level.
  - NDCG@5/10/20 uses linear lowest-common-ancestor-depth gains (0 to 4) with a
    `1/log2(rank + 1)` discount.
  - The distance Pearson correlation with D* runs over all pairs.
  - Parent retrieval@1/5 is scored for each non-sector query. The 522 unary pairs, six-digit
    codes whose five-digit parent has only them, are not scored.
  - Ties in distance break against relevance, except in the AUC and in Spearman's average ranks.
    There are no thresholds and no pass or fail.
- **Removals.**
  - `metrics/qcew.py` goes with its exports and API page. It is the benchmark Req 2 rejects.
  - The unwired downstream suite (`GraphDownstreamEvaluator`, `run_graph_downstream_suite`) goes
    from `metrics/graph.py`. `GraphEmbeddingDataset` and `compute_validation_metrics` stay.
  - `tools verify-stage4` goes (user decision 2).

### Recorded deviations

- **Superiority reads 98⅓ %, not 97.5 %.** D8 supersedes the level Req 5 and Verification
  "Decision records" name. The records carry both levels in `settings`.
- **Verification "Diagnostics" is met in the report and in every headline, not yet in the logs.**
  Text and HGCN validation still compute and log the old structural statistics, off every
  progress bar (user decision 3). Stage 7 removes them from the text stage and Stages 10/11 from
  the graph stage; Plan completion Step 2 writes both leftovers into the roadmap. Nothing reads
  them: the text stage's checkpoint and early-stopping monitors read `val/contrastive_loss`,
  HGCN runs with `enable_checkpointing=False`, and the graph curriculum controller is not wired
  into training.
- **`tools investigate` stays.** It advises on low hierarchy correlation from the legacy distance
  matrix. It is outside user decision 3, and Plan completion names it in Stage 7's entry.

### Project rules

- **Style (CLAUDE.md).**
  - Single quotes, including `'''` docstrings. A string with an apostrophe takes double quotes
    (ruff Q003).
  - YAPF owns layout (100 columns), and ruff lints (E, F, I, Q). **Never run `ruff format`.**
  - One blank line between top-level definitions and after imports.
  - Semantic section dividers; `logging` rather than `print`; type hints on signatures.
- **Config.** Every config key is declared in a Pydantic model.
- **Formatting.** Format touched files with `./scripts/format_code.sh <files>`. At the end,
  `./scripts/format_code.sh --check --all` must pass.
- **Git.**
  - Never push to `main`.
  - Never push, cherry-pick or merge the held local commits "config" and "graph config". They
    are named by subject because every sync rewrites their SHAs.
  - Never run bare `git stash`.
  - Commit on this branch only, and end each message with the session's attribution trailer.
- **Data safety.**
  - Never write to the main checkout's `data/`. Read bundle 18403d29's codebook and the pinned
    descriptions file (sha256 5107fb83…) by absolute path only.
  - Never build or rebuild a supervision bundle: only Stage 5 rebuilds.
- **Panels.** This plan reads no real panel at all:
  - no `OutcomePanel` or `RegressorPanel` on real data;
  - no `tools outcome-baseline` or `tools regressor-panel`;
  - no `open_test`, `open_outer` or `test`.

  The decision tooling and the seed-sweep driver run on fixture panels and synthetic arms, in
  tests. No `logs/selection_log.jsonl` may appear in this worktree.
- **Downloads.** Never download QCEW or Census files or a Hugging Face model. The backbone loads
  from the local cache (`local_files_only=True`).
- **Deferred items.** Do not promote any open item of `specs/deferred_items.md` beyond the two
  plan-5 items this plan discharges (Plan completion Step 3).
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-embedding-roadmap.md` and `specs/deferred_items.md`, and hand the user the exact
  edit instead.
- **Bash tool.** It runs zsh. Quote `=`-leading words (`echo '====='`); an unquoted one aborts
  the command. If the tool refuses a heredoc or a compound command, run one plain command per call
  and write files with the Write tool.

## Workspace

- **Worktree:**
  `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-6-decision-rule-and-diagnostics`.
  Run every command from its root.
- **Branch:** `claude/plan-6-decision-rule-and-diagnostics-ec267a03`, cut from origin/main
  `33fe689` (PR #115). This plan is its first commit.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder` stays on local `main`: 7ffe551 (29
  commits behind origin/main 33fe689) plus the held commits "config" and "graph config". Do not
  check anything out there.
- **Real inputs, read only (Task 13):**
  - Codebook: `/Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet`
  - Descriptions: `/Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`
  - Backbone: `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`
- **Scratch:** `/tmp/stage4-diagnostics-ec267a03/` (Task 13). Final verification removes it.
- **Working directory:** the Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before Task 13's commands and before every commit, and if it is not
  this worktree, `cd` back first.

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `src/naics_embedder/panels/outcome.py`, `regressor.py`, `text_only.py` | A read's caller detail; tables named by `matrix_fingerprint` | 1 |
| `src/naics_embedder/decision/__init__.py`, `scores.py` | The package; each panel's per-unit scores (D10) | 2 |
| `src/naics_embedder/decision/resampling.py` | The paired two-stage bootstrap | 3 |
| `src/naics_embedder/decision/records.py`, `rule.py` | The record schema; intervals, verdicts, survivors, tie order | 4 |
| `src/naics_embedder/decision/store.py` | The content-addressed artifact store | 5 |
| `tests/fixtures/decision.py` | Synthetic arms with known effects on the three panels | 5 |
| `src/naics_embedder/decision/decide.py` | Guards, `fix_margins`, `decide` | 6 |
| `src/naics_embedder/decision/sweep.py` | `run_seed_sweep` | 7 |
| `src/naics_embedder/utils/config.py`, `conf/data/decision.yaml` | `DecisionConfig` | 8 |
| `src/naics_embedder/cli/commands/tools.py` | `tools margins` and `tools decide` (8), `tools diagnostics` (9), no `verify-stage4` (11) | 8, 9, 11 |
| `src/naics_embedder/metrics/diagnostics.py` | Req 6's report | 9 |
| `src/naics_embedder/metrics/__init__.py` | Export the report (9); drop QCEW and the suite (10) | 9, 10 |
| `src/naics_embedder/metrics/graph.py`, `graph_model/__init__.py` | Drop the suite and the QCEW re-exports | 10 |
| `src/naics_embedder/metrics/qcew.py`, `tools/embeddings_verification.py` | Deleted | 10, 11 |
| `src/naics_embedder/text_model/mixins/validation.py`, `graph_model/hgcn.py` | No structural statistic on a progress bar | 12 |
| `src/naics_embedder/tools/_visualize_metrics.py`, `metrics_tools.py`, `cli/commands/training.py` | No structural statistic in a headline | 12 |
| `docs/api/decision.md`, `docs/api/diagnostics.md`, `docs/.nav.yml`, `docs/usage.md` | The new pages and commands; the removed pages | 8–12 |
| `README.md`, `WARP.md`, `CLAUDE.md`, `docs/hgcn_training.md`, `docs/overview.md` | Documentation of what changed | 8–12 |

Tests:

- New files: `tests/unit/test_decision_scores.py` (2), `test_decision_resampling.py` (3),
  `test_decision_rule.py` (4), `test_decision_store.py` (5), `test_decision.py` (6),
  `test_decision_sweep.py` (7) and `test_diagnostics.py` (9).
- Edited files: `test_outcome_panel.py`, `test_regressor_panel.py` and `test_text_only.py` (1);
  `test_config.py` (8); `test_cli_commands.py` (8, 9 and 11);
  `test_graph_downstream_evaluation.py` and `test_graph_pairwise_distances.py` (10);
  `test_hgcn_metrics.py`, `test_text_validation_metrics.py`, `test_visualize_metrics.py`,
  `test_metrics_tools_api.py` and `test_cli_training.py` (12).
- Deleted files: `test_qcew_multilevel.py` (10) and `test_embeddings_verification.py` (11).

## Expected real-data results

These were verified while writing this plan, with the same code on the same inputs, under Python
3.12 with numpy 2.3.4, polars 1.35.1, scipy 1.16.3, torch 2.9.1 and transformers 4.57.1. Task 13
must reproduce them to 4 decimals.

| Quantity | Value |
|---|---|
| Text-only table | 2,125 codes × 384; backbone revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; about 34 s |
| Text-only table sha256 on this Mac | `6386f912ea4b37984ed9d4e4100f21fbdc29e91c5ee38acb05bd999342f5bbe0` (plan 5 recorded the same) |
| Its `matrix_fingerprint` | `6838c0adf0573821b139183a11db7d232f3965c99d3715202239f9463ba13384` |
| Random table sha256 (Task 13 Step 5) | `e8534cc24d7e931b0ab6c145b307e19c0c0d8a6c0a17a74e74fc828cf221361c` |
| Pairs and queries, every table | 272,103 same-sector and 1,984,647 cross-sector pairs; 2,256,750 in all; within-sector 2,125 queries, 0 undefined; MAP 2,105 queries; NDCG 2,125; parent retrieval 1,583, with 522 unary pairs excluded |
| Text-only, spherical | AUC 0.8538; within-sector 0.2974 (queries) and 0.3857 (sectors); MAP 0.3262 (levels 3–6: 0.1415, 0.1761, 0.2543, 0.4384); NDCG@5/10/20 0.8220, 0.7742, 0.7398; Pearson 0.2378; parent @1 0.2855, @5 0.6639 |
| Random, euclidean | AUC 0.4940; within-sector 0.0038 and 0.0190; MAP 0.0052 (0.0016, 0.0037, 0.0065, 0.0052); NDCG 0.0441, 0.0481, 0.0544; Pearson 0.0030; parent 0.0019, 0.0032 |
| Random, hyperbolic | AUC 0.4928; within-sector 0.0073 and 0.0348; MAP 0.0050 (0.0016, 0.0033, 0.0064, 0.0050); NDCG 0.0384, 0.0427, 0.0494; Pearson 0.0051; parent 0.0006, 0.0038 |
| One report | about 1.3 s |
| Full suite after Task 12 (Python 3.12 and 3.10) | 1590 passed, 1 skipped |

The text-only table reads well above chance on every statistic, and the random table sits at
chance: an AUC near 0.5 and correlations near 0.

---

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- A Task 13 number differs from **Expected real-data results** in its first 4 decimals, or a
  table's sha256 or `matrix_fingerprint` differs.
- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A step would read or open a real panel, write a selection log in this worktree, write to the
  main checkout's `data/`, build a supervision bundle, change `conf/config.yaml` or
  `conf/graph.yaml`, or download anything.
- `origin/main` gains a commit touching a file in **File structure**, or an open PR does.
- The codebook, the descriptions file or a QCEW slice has another sha256 than the pre-flight's,
  or the backbone's cached revision differs. Do not download a replacement.

## Pre-flight (controller, inline, before Task 1)

- [x] **Step 1: Confirm the workspace**

Run: `git status --short --branch`
Expected: `## claude/plan-6-decision-rule-and-diagnostics-ec267a03` and nothing else. The branch
has no upstream yet. If the line ends in `...origin/main`, run `git branch --unset-upstream`, so
that no command treats `main` as this branch's remote branch.

Run: `git log --oneline origin/main..HEAD`
Expected: only this plan's commit (`docs(plans): add plan 6 …`). If "config" or "graph config"
appears, stop.

Run: `git fetch origin`, then
`git log --oneline HEAD..origin/main -- src tests conf docs specs CLAUDE.md README.md WARP.md`
Expected: no output. If anything landed, read it. If it touches a file in **File structure**,
the roadmap or `specs/deferred_items.md`, stop and ask.

Run: `gh pr list --state open`
Expected: no open PR touching a file in **File structure**. If one does, stop and ask.

- [x] **Step 2: Build the worktree's environment**

Run: `uv sync`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number. `.python-version` pins 3.12.

Run: `uv run python -c "import numpy, polars, scipy, torch, transformers; print(numpy.__version__, polars.__version__, scipy.__version__, torch.__version__, transformers.__version__)"`
Expected: `2.3.4 1.35.1 1.16.3 2.9.1 4.57.1`. These are the locked versions for Python 3.12, and
Task 13's expected results were computed with them. If they differ, stop and ask.

- [x] **Step 3: Run the baseline suite**

Run: `uv run pytest -n auto -q`
Expected: `1535 passed, 1 skipped`. Each later full-suite count is this baseline plus the tests
the plan has added by then. The warnings count varies between runs under xdist; ignore it.

- [x] **Step 4: Check the real inputs, read-only**

Run: `shasum -a 256 /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`
Expected:

```text
5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b  …/naics_codebook.parquet
5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d  …/naics_descriptions.parquet
```

Run: `cat ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main`
Expected: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Run: `shasum -a 256 ~/Downloads/Data/QCEW/2022_US000_annual.csv ~/Downloads/Data/QCEW/2023_US000_annual.csv ~/Downloads/Data/QCEW/2024_US000_annual.csv ~/Downloads/Data/QCEW/2025_US000_annual.csv`
Expected:

```text
c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18  …/2022_US000_annual.csv
fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9  …/2023_US000_annual.csv
48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46  …/2024_US000_annual.csv
0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811  …/2025_US000_annual.csv
```

These are the hashes `conf/data/regressor_panel.yaml` pins. This plan does not read the slices,
because the seed-sweep driver scores fixture panels only. The check records that they are in
place for the stages whose sweeps read the real regressor panel. A Lambda instance has them only
if they are uploaded. If a hash differs, stop and ask.

- [x] **Step 5: Route the tasks**

> Deviation: the pre-flight scan raised five findings, which the user ruled on before Task 1: F1 extract `merge_read_detail` (Task 1), F2 reuse the regressor-panel fixture's `text_only_table` (Task 5), F3 enforce the 5-seed floor in `check_arm` (Task 6), F4 keep `provenance.get('matrix_fingerprint', fingerprint)` as written, and F5 `assert metrics` (Task 12). Implementers ran on Sonnet and reviews on Sonnet or Opus; implementer commits carry `Co-Authored-By: Claude Sonnet 5`, the controller's `Claude Opus 5.5`.

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–12: each gets a fresh implementer and a task-reviewer. Give each implementer its task,
  **Global Constraints** and **Workspace**. Every code block in a task is exact: an implementer
  copies it, and each "Replace" text occurs exactly once in its file when its edit is made.
- Task 13, **Final verification** and **Plan completion**: run them inline in the controller
  session. Task 13 reads the real inputs and applies the stop-and-ask conditions.

### Task 1: Reads name their run and their tables

A decision record must tie each run to its selection-log records, and each record to the tables
the run read. This task makes that possible in three ways:

- `OutcomePanel.score` and `RegressorPanel.validation` and `test` take a caller's `detail`, which
  joins the read's logged detail. It cannot replace a key the panel logs itself, so a caller can
  name its run but cannot misreport what the panel read.
- `matrix_fingerprint` moves to `panels/text_only.py`. `text_only_fingerprint` and
  `table_fingerprint` compute the names a regressor read logs the two tables by.
- The text-only provenance records `matrix_fingerprint` beside `table_sha256`. This closes plan
  5's deferred item on the mismatch between the logged name and the file hash.

**Files:**

- Modify: `src/naics_embedder/panels/outcome.py` (`score(..., detail=None)`)
- Modify: `src/naics_embedder/panels/regressor.py` (`table_fingerprint`; `validation` and `test`
  take `detail`)
- Modify: `src/naics_embedder/panels/text_only.py` (`matrix_fingerprint` moves here;
  `text_only_matrix`, `text_only_fingerprint`; the provenance records the table's
  `matrix_fingerprint`)
- Modify: `tests/unit/test_outcome_panel.py`
- Modify: `tests/unit/test_regressor_panel.py`
- Modify: `tests/unit/test_text_only.py`

**Interfaces:**

- Consumes (existing):
  - `OutcomePanel.score(encoder, split, purpose, distance='cosine') -> DecodingResult`;
  - `RegressorPanel.validation(regime, level, arm, purpose) -> pl.DataFrame` and `test` with the
    same arguments;
  - `ArmTables`, with `fingerprint`, `text_only_fingerprint` and `dimension`;
  - `coordinate_matrix(table)` in `panels/regressor.py`.
- Produces, in `naics_embedder.panels.text_only`:
  - `matrix_fingerprint(codes: Sequence[str], matrix: np.ndarray) -> str`, moved from
    `panels/regressor.py`, which now imports it;
  - `text_only_matrix(table: pl.DataFrame) -> Tuple[List[str], np.ndarray]`;
  - `text_only_fingerprint(table: pl.DataFrame) -> str`;
  - the provenance JSON's new key `matrix_fingerprint`.
- Produces, in `naics_embedder.panels.regressor`:
  - `table_fingerprint(table: pl.DataFrame) -> str`, an arm table's logged name;
  - `RegressorPanel.validation(regime, level, arm, purpose, detail: Optional[Mapping[str, Any]] =
    None) -> pl.DataFrame`, and `test` likewise. The panel still logs `level`, `comparators`,
    `arm`, `text_only` and `dimension`. A caller key among them raises
    `ValueError('a read cannot replace the logged [...]')`.
- Produces, in `naics_embedder.panels.outcome`:
  - `OutcomePanel.score(encoder, split, purpose, distance='cosine', detail:
    Optional[Mapping[str, Any]] = None) -> DecodingResult`. The panel still logs `encoder` and
    `distance`, with the same `ValueError` on a clash.

- [x] **Step 1: Write the failing tests**

Modify `tests/unit/test_outcome_panel.py` with one edit. Replace:

```python
    assert log.records()[0]['detail'] == {'encoder': 'OneHotStubEncoder', 'distance': 'cosine'}

```

with:

```python
    assert log.records()[0]['detail'] == {'encoder': 'OneHotStubEncoder', 'distance': 'cosine'}

def test_a_read_carries_the_callers_detail_beside_the_panels_own(panel, log, encoder):
    panel.score(encoder, 'validation', 'seed sweep', detail={'run': 'arm-a/seed-0', 'seed': 0})

    [record] = log.records()
    assert record['detail'] == {
        'encoder': 'OneHotStubEncoder',
        'distance': 'cosine',
        'run': 'arm-a/seed-0',
        'seed': 0,
    }

def test_a_read_cannot_replace_what_the_panel_logs(panel, log, encoder):
    with pytest.raises(ValueError, match='distance'):
        panel.score(encoder, 'validation', 'seed sweep', detail={'distance': 'euclidean'})

    assert log.records() == []

```

Modify `tests/unit/test_regressor_panel.py` with these 3 edits, in order.

**`tests/unit/test_regressor_panel.py`, edit 1 of 3.** Replace:

```python
    summarize,
```

with:

```python
    summarize,
    table_fingerprint,
```

**`tests/unit/test_regressor_panel.py`, edit 2 of 3.** Replace:

```python
from naics_embedder.panels.selection_log import SelectionLog
```

with:

```python
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.panels.text_only import text_only_fingerprint
```

**`tests/unit/test_regressor_panel.py`, edit 3 of 3.** Replace:

```python
def _shift_outcomes(rows, where):
```

with:

```python
def test_a_read_carries_the_callers_detail_beside_the_panels_own(panel, regressor_arm, log):
    run = {'run': 'arm-a/seed-0', 'seed': 0}

    panel.validation(Regime.SEEN, 6, regressor_arm, PURPOSE, detail=run)
    panel.open_outer(Regime.SEEN, PURPOSE)
    panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE, detail=run)

    reads = [r for r in log.records() if r['event'] == 'read']
    assert [r['split'] for r in reads] == ['validation', 'test']
    for record in reads:
        assert record['detail']['run'] == 'arm-a/seed-0'
        assert record['detail']['seed'] == 0
        assert record['detail']['arm'] == regressor_arm.fingerprint
        assert record['detail']['text_only'] == regressor_arm.text_only_fingerprint

@pytest.mark.parametrize('key', ['arm', 'text_only', 'level', 'comparators', 'dimension'])
def test_a_read_cannot_replace_what_the_panel_logs(panel, regressor_arm, log, key):
    with pytest.raises(ValueError, match=key):
        panel.validation(Regime.SEEN, 6, regressor_arm, PURPOSE, detail={key: 'other'})
    panel.open_outer(Regime.SEEN, PURPOSE)
    with pytest.raises(ValueError, match=key):
        panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE, detail={key: 'other'})

    assert [r['event'] for r in log.records()] == ['open']

def test_the_logged_table_names_are_their_matrix_fingerprints(regressor_arm):
    coordinates = coordinate_table(CODEBOOK)
    text = text_only_table(CODEBOOK)

    assert regressor_arm.fingerprint == table_fingerprint(coordinates)
    assert regressor_arm.text_only_fingerprint == text_only_fingerprint(text)
    # Row order does not change a name
    assert table_fingerprint(coordinates.reverse()) == regressor_arm.fingerprint
    assert text_only_fingerprint(text.reverse()) == regressor_arm.text_only_fingerprint

def _shift_outcomes(rows, where):
```

Modify `tests/unit/test_text_only.py` with these 2 edits, in order.

**`tests/unit/test_text_only.py`, edit 1 of 2.** Replace:

```python
    provenance_path,
```

with:

```python
    provenance_path,
    text_only_fingerprint,
```

**`tests/unit/test_text_only.py`, edit 2 of 2.** Replace:

```python
    assert set(provenance['library_versions']) == {'torch', 'transformers', 'polars'}
```

with:

```python
    assert set(provenance['library_versions']) == {'torch', 'transformers', 'polars'}
    # The name a regressor read logs the table by, so a logged read matches this file
    assert provenance['matrix_fingerprint'] == text_only_fingerprint(table)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_outcome_panel.py -q`
Expected: `2 failed, 18 passed`. Both failures,
`test_a_read_carries_the_callers_detail_beside_the_panels_own` and
`test_a_read_cannot_replace_what_the_panel_logs`, raise
`TypeError: OutcomePanel.score() got an unexpected keyword argument 'detail'`.

Run: `uv run pytest tests/unit/test_regressor_panel.py tests/unit/test_text_only.py -q`
Expected: two collection errors:
`ImportError: cannot import name 'table_fingerprint' from 'naics_embedder.panels.regressor'`
and
`ImportError: cannot import name 'text_only_fingerprint' from 'naics_embedder.panels.text_only'`.

- [x] **Step 3: Implement**

> Deviation: F1 (user ruling at pre-flight): after the plan's edits, `merge_read_detail(logged, extra)` was extracted into `panels/selection_log.py`, and `OutcomePanel.score` and `regressor.py`'s `_read_detail` call it, with the same message and no new tests (f2796a2). Final verification Step 5's path list gains `src/naics_embedder/panels/selection_log.py`.

Modify `src/naics_embedder/panels/outcome.py` with these 2 edits, in order.

**`src/naics_embedder/panels/outcome.py`, edit 1 of 2.** Replace:

```python
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, Union
```

with:

```python
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union
```

**`src/naics_embedder/panels/outcome.py`, edit 2 of 2.** Replace:

```python
        distance: Union[str, DistanceFn] = 'cosine',
    ) -> DecodingResult:
        '''Decode one split's queries over every candidate with the encoder, logging the read.'''

        name, _ = resolve_distance(distance)
        detail = {'encoder': type(encoder).__name__, 'distance': name}
        queries = self._read(IndexRole(split), purpose, detail)
```

with:

```python
        distance: Union[str, DistanceFn] = 'cosine',
        detail: Optional[Mapping[str, Any]] = None,
    ) -> DecodingResult:
        '''
        Decode one split's queries over every candidate with the encoder, logging the read.

        ``detail`` joins the read's logged detail, so a caller can name the run it scores; it
        cannot replace the encoder or the distance the panel logs.
        '''

        name, _ = resolve_distance(distance)
        logged = {'encoder': type(encoder).__name__, 'distance': name}
        clash = sorted(set(detail or {}) & set(logged))
        if clash:
            raise ValueError(f'a read cannot replace the logged {clash}')
        queries = self._read(IndexRole(split), purpose, {**logged, **dict(detail or {})})
```

Modify `src/naics_embedder/panels/regressor.py` with these 11 edits, in order.

**`src/naics_embedder/panels/regressor.py`, edit 1 of 11.** Replace:

```python

import hashlib
```

with:

```python

```

**`src/naics_embedder/panels/regressor.py`, edit 2 of 11.** Replace:

```python
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.panels.text_only import TEXT_ONLY_PREFIX, pca_reduce
```

with:

```python
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.panels.text_only import matrix_fingerprint, pca_reduce, text_only_matrix
```

**`src/naics_embedder/panels/regressor.py`, edit 3 of 11.** Replace:

```python

def matrix_fingerprint(codes: Sequence[str], matrix: np.ndarray) -> str:
    '''SHA-256 of the codes and their float64 values, in code order.'''

    order = np.argsort(np.asarray(codes))
    digest = hashlib.sha256('\n'.join(codes[index] for index in order).encode('utf-8'))
    digest.update(np.ascontiguousarray(matrix[order], dtype=np.float64).tobytes())
    return digest.hexdigest()
```

with:

```python

def table_fingerprint(table: pl.DataFrame) -> str:
    '''An arm's coordinate table's ``matrix_fingerprint``: the name a regressor read logs it by.'''

    return matrix_fingerprint(*coordinate_matrix(table))
```

**`src/naics_embedder/panels/regressor.py`, edit 4 of 11.** Replace:

```python
        codes, matrix = coordinate_matrix(coordinates)
        text_columns = [name for name in text_only.columns if name.startswith(TEXT_ONLY_PREFIX)]
        text_codes = text_only.get_column('code').to_list()
        if set(text_codes) != set(codes) or len(text_codes) != len(codes):
            raise ValueError('the coordinate and text-only tables cover different codes')
        text_matrix = np.array(text_only.select(text_columns).to_numpy(), dtype=np.float64)
```

with:

```python
        codes, matrix = coordinate_matrix(coordinates)
        text_codes, text_matrix = text_only_matrix(text_only)
        if set(text_codes) != set(codes) or len(text_codes) != len(codes):
            raise ValueError('the coordinate and text-only tables cover different codes')
```

**`src/naics_embedder/panels/regressor.py`, edit 5 of 11.** Replace:

```python

    def validation(self, regime: Regime, level: int, arm: ArmTables, purpose: str) -> pl.DataFrame:
        '''Out-of-sample predictions for the remainder rows, logging the read.'''

        regime = Regime(regime)
```

with:

```python

    def validation(
        self,
        regime: Regime,
        level: int,
        arm: ArmTables,
        purpose: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> pl.DataFrame:
        '''
        Out-of-sample predictions for the remainder rows, logging the read.

        ``detail`` joins the read's logged detail, so a caller can name the run it scores; it
        cannot replace a key the panel logs itself.
        '''

        regime = Regime(regime)
        logged = self._read_detail(regime, level, arm, detail)
```

**`src/naics_embedder/panels/regressor.py`, edit 6 of 11.** Replace:

```python
        _require_codes(arm, frame)
        self._log_read(regime, VALIDATION, level, arm, purpose, frame.height)
```

with:

```python
        _require_codes(arm, frame)
        self._log_read(regime, VALIDATION, purpose, frame.height, logged)
```

**`src/naics_embedder/panels/regressor.py`, edit 7 of 11.** Replace:

```python

    def test(self, regime: Regime, level: int, arm: ArmTables, purpose: str) -> pl.DataFrame:
        '''
        Predictions for the regime's outer set from a fit on the whole remainder, logging the read.

```

with:

```python

    def test(
        self,
        regime: Regime,
        level: int,
        arm: ArmTables,
        purpose: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> pl.DataFrame:
        '''
        Predictions for the regime's outer set from a fit on the whole remainder, logging the read.

        ``detail`` joins the read's logged detail, as in ``validation``.

```

**`src/naics_embedder/panels/regressor.py`, edit 8 of 11.** Replace:

```python
        regime = Regime(regime)
        self._require_defined(regime, level)
```

with:

```python
        regime = Regime(regime)
        logged = self._read_detail(regime, level, arm, detail)
        self._require_defined(regime, level)
```

**`src/naics_embedder/panels/regressor.py`, edit 9 of 11.** Replace:

```python
        _require_codes(arm, frame)
        self._log_read(regime, TEST, level, arm, purpose, n_outer)
```

with:

```python
        _require_codes(arm, frame)
        self._log_read(regime, TEST, purpose, n_outer, logged)
```

**`src/naics_embedder/panels/regressor.py`, edit 10 of 11.** Replace:

```python

    def _log_read(
        self, regime: Regime, split: str, level: int, arm: ArmTables, purpose: str, n_rows: int
```

with:

```python

    def _read_detail(
        self,
        regime: Regime,
        level: int,
        arm: ArmTables,
        extra: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        detail: Dict[str, Any] = {
            'level': level,
            'comparators': list(comparators(regime, level)),
            'arm': arm.fingerprint,
            'text_only': arm.text_only_fingerprint,
            'dimension': arm.dimension,
        }
        clash = sorted(set(extra or {}) & set(detail))
        if clash:
            raise ValueError(f'a read cannot replace the logged {clash}')
        return {**detail, **dict(extra or {})}

    def _log_read(
        self, regime: Regime, split: str, purpose: str, n_rows: int, detail: Dict[str, Any]
```

**`src/naics_embedder/panels/regressor.py`, edit 11 of 11.** Replace:

```python
            n_queries=n_rows,
            detail={
                'level': level,
                'comparators': list(comparators(regime, level)),
                'arm': arm.fingerprint,
                'text_only': arm.text_only_fingerprint,
                'dimension': arm.dimension,
            },
```

with:

```python
            n_queries=n_rows,
            detail=detail,
```

Modify `src/naics_embedder/panels/text_only.py` with these 4 edits, in order.

**`src/naics_embedder/panels/text_only.py`, edit 1 of 4.** Replace:

```python
import json
```

with:

```python
import hashlib
import json
```

**`src/naics_embedder/panels/text_only.py`, edit 2 of 4.** Replace:

```python
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
```

with:

```python
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
```

**`src/naics_embedder/panels/text_only.py`, edit 3 of 4.** Replace:

```python
def provenance_path(table_path: Path) -> Path:
```

with:

```python
def matrix_fingerprint(codes: Sequence[str], matrix: np.ndarray) -> str:
    '''SHA-256 of the codes and their float64 values, in code order.'''

    order = np.argsort(np.asarray(codes))
    digest = hashlib.sha256('\n'.join(codes[index] for index in order).encode('utf-8'))
    digest.update(np.ascontiguousarray(matrix[order], dtype=np.float64).tobytes())
    return digest.hexdigest()

def text_only_matrix(table: pl.DataFrame) -> Tuple[List[str], np.ndarray]:
    '''The table's codes and its ``t`` columns (float64).'''

    columns = [name for name in table.columns if name.startswith(TEXT_ONLY_PREFIX)]
    codes = table.get_column('code').to_list()
    return codes, np.array(table.select(columns).to_numpy(), dtype=np.float64)

def text_only_fingerprint(table: pl.DataFrame) -> str:
    '''
    The table's ``matrix_fingerprint``: the name a regressor read logs it by.

    The provenance records it beside the file's own hash (``table_sha256``), so a logged read
    matches its table file without re-reading the table.
    '''

    return matrix_fingerprint(*text_only_matrix(table))

def provenance_path(table_path: Path) -> Path:
```

**`src/naics_embedder/panels/text_only.py`, edit 4 of 4.** Replace:

```python
        'table_sha256': sha256_file(output_path),
```

with:

```python
        'table_sha256': sha256_file(output_path),
        'matrix_fingerprint': text_only_fingerprint(table),
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_outcome_panel.py tests/unit/test_regressor_panel.py tests/unit/test_text_only.py -q`
Expected: `105 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1544 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/outcome.py src/naics_embedder/panels/regressor.py src/naics_embedder/panels/text_only.py tests/unit/test_outcome_panel.py tests/unit/test_regressor_panel.py tests/unit/test_text_only.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/outcome.py \
  src/naics_embedder/panels/regressor.py \
  src/naics_embedder/panels/text_only.py \
  tests/unit/test_outcome_panel.py \
  tests/unit/test_regressor_panel.py \
  tests/unit/test_text_only.py
git commit -m "feat(panels): let reads carry a run detail and name tables by matrix_fingerprint"
```

### Task 2: Each panel's per-unit scores (D10)

A seed's scores are one long frame over the three panels, keyed by each panel's resampling unit.

- **Outcome panel.** It contributes every Req 3 metric of each validation query, with the
  query's code as its unit. This is section 6 of the outcome finding: resample by code from
  `DecodingResult.per_query`.
- **Regressor regimes.** Each contributes every comparator's squared error on each level-6
  validation row, with the row's four-digit group as its unit. The error is averaged over the
  row's repeats first. This is plan 5's deferred note: the repeats share one fit, so they are
  not independent draws. A row must have exactly `repeats` predictions under each comparator.
- **Feature year.** Each row keeps its `feature_year`, so the held-out regime can be reported by
  feature year.

**Files:**

- Create: `src/naics_embedder/decision/__init__.py` (the package)
- Create: `src/naics_embedder/decision/scores.py` (each panel's per-unit scores (D10))
- Create: `tests/unit/test_decision_scores.py`

**Interfaces:**

- Consumes:
  - Stage 2's `DecodingResult.per_query`: one row per query, with `query_id`, `code`, `rank`,
    `reciprocal_rank` and `lca_level` (`naics_embedder.panels.decoding`);
  - `METRIC_NAMES`, `HIT_KS` and `score_decoding`, from the same module;
  - `OUTCOME_PANEL` (`panels.outcome`);
  - Stage 3's predictions frame, with columns `PREDICTION_COLUMNS`; and `PANEL_NAMES`, `Regime`,
    `DECISION_LEVEL` (6) and `VALIDATION` (`'validation'`), all existing in `panels.regressor`.
- Produces, in `naics_embedder.decision.scores`:
  - `SEEN_PANEL = 'regressor_seen'`, `HELDOUT_PANEL = 'regressor_heldout'`,
    `PANELS = (OUTCOME_PANEL, SEEN_PANEL, HELDOUT_PANEL)` and `REGRESSOR_PANELS`;
  - `EMBEDDING_COMPARATOR = 'covariates+embedding'`, `DECISION_STATISTIC` (`mrr` for the outcome
    panel, `EMBEDDING_COMPARATOR` for each regime) and `ORIENTATION` (+1 and −1);
  - `SPARSE_COMPARATOR` (`covariates+one_hot` for seen, `covariates+ancestors` for held-out) and
    `STATISTIC_DEFINITIONS`;
  - `SCORE_SCHEMA`, whose columns are `panel`, `statistic`, `unit`, `item`, `feature_year`
    (Int32, null on the outcome panel) and `value`; and `SCORE_COLUMNS`;
  - `outcome_scores(per_query: pl.DataFrame) -> pl.DataFrame`;
  - `regressor_scores(predictions: pl.DataFrame, repeats: int) -> pl.DataFrame`;
  - `seed_scores(per_query: pl.DataFrame, predictions: pl.DataFrame, repeats: int) ->
    pl.DataFrame`;
  - `statistic_values(scores: pl.DataFrame, panel: str, statistic: str) -> pl.DataFrame`;
  - `panel_statistic(scores: pl.DataFrame, panel: str, statistic: str) -> float`;
  - `statistic_means(scores: pl.DataFrame) -> Dict[str, Dict[str, float]]`.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_decision_scores.py` with exactly this content:

```python
'''
Each panel's per-unit scores (D10): per-query metrics by code, and per-row squared errors by
four-digit group with each row's repeats averaged first.
'''

import polars as pl
import pytest
import torch

from naics_embedder.decision.scores import (
    PANELS,
    SCORE_COLUMNS,
    outcome_scores,
    panel_statistic,
    regressor_scores,
    seed_scores,
)
from naics_embedder.panels.decoding import METRIC_NAMES, score_decoding
from naics_embedder.panels.regressor import PREDICTION_COLUMNS

pytestmark = pytest.mark.unit

def _per_query():
    # Query 7 decodes to its code (rank 1); query 8 ties its code with two others (rank 3)
    codes = ['111110', '111120', '211111']
    return score_decoding(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ['111110', '211111'],
        torch.eye(3),
        codes,
        distance='euclidean',
        query_ids=[7, 8],
    ).per_query

def _prediction(panel, comparator, repeat, code, year, prediction, outcome=1.0):
    return {
        'panel': panel,
        'split': 'validation',
        'level': 6,
        'comparator': comparator,
        'repeat': repeat,
        'fold': 0,
        'code': code,
        'group': code[:4],
        'feature_year': year,
        'outcome_year': year + 1,
        'alpha': 1.0,
        'outcome': outcome,
        'prediction': prediction,
    }

def _predictions(rows):
    schema = {
        'panel': pl.Utf8,
        'split': pl.Utf8,
        'level': pl.Int32,
        'comparator': pl.Utf8,
        'repeat': pl.Int32,
        'fold': pl.Int32,
        'code': pl.Utf8,
        'group': pl.Utf8,
        'feature_year': pl.Int32,
        'outcome_year': pl.Int32,
        'alpha': pl.Float64,
        'outcome': pl.Float64,
        'prediction': pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema, orient='row').select(PREDICTION_COLUMNS)

def test_outcome_items_are_queries_resampled_by_their_code():
    scores = outcome_scores(_per_query())

    assert scores.columns == list(SCORE_COLUMNS)
    assert sorted(scores.get_column('statistic').unique().to_list()) == sorted(METRIC_NAMES)
    mrr = scores.filter(pl.col('statistic') == 'mrr').sort('item')
    assert mrr.select('unit', 'item', 'value').rows() == [
        ('111110', '7', 1.0), ('211111', '8', 1 / 3)
    ]
    assert panel_statistic(scores, 'outcome', 'mrr') == pytest.approx((1 + 1 / 3) / 2)

def test_a_rows_repeats_are_averaged_before_it_counts_once():
    predictions = _predictions(
        [
            # 111111 in 2022: errors 1 and 9 over its two repeats, so 5; in 2023: 0 and 4, so 2
            _prediction('regressor_heldout', 'covariates+embedding', 0, '111111', 2022, 2.0),
            _prediction('regressor_heldout', 'covariates+embedding', 1, '111111', 2022, 4.0),
            _prediction('regressor_heldout', 'covariates+embedding', 0, '111111', 2023, 1.0),
            _prediction('regressor_heldout', 'covariates+embedding', 1, '111111', 2023, 3.0),
            _prediction('regressor_heldout', 'covariates+embedding', 0, '222211', 2022, 1.0),
            _prediction('regressor_heldout', 'covariates+embedding', 1, '222211', 2022, 1.0),
        ]
    )

    scores = regressor_scores(predictions, repeats=2)

    assert scores.select('unit', 'item', 'feature_year', 'value').rows() == [
        ('1111', '111111/2022', 2022, 5.0),
        ('1111', '111111/2023', 2023, 2.0),
        ('2222', '222211/2022', 2022, 0.0),
    ]
    assert panel_statistic(scores, 'regressor_heldout',
                           'covariates+embedding') == pytest.approx(7 / 3)

def test_a_row_without_every_repeat_is_refused():
    predictions = _predictions(
        [_prediction('regressor_seen', 'covariates+embedding', 0, '111111', 2023, 1.0)]
    )

    with pytest.raises(ValueError, match='do not have 2 predictions'):
        regressor_scores(predictions, repeats=2)

@pytest.mark.parametrize(
    'column, value', [('split', 'test'), ('level', 5), ('panel', 'regressor_other')]
)
def test_only_level_six_validation_rows_of_the_two_regimes_are_scored(column, value):
    row = _prediction('regressor_seen', 'covariates+embedding', 0, '111111', 2023, 1.0)

    with pytest.raises(ValueError, match='validation rows'):
        regressor_scores(_predictions([{**row, column: value}]), repeats=1)

def test_a_seed_is_scored_on_all_three_panels():
    rows = [
        _prediction(panel, 'covariates+embedding', 0, '111111', 2023, 1.0)
        for panel in ('regressor_seen', 'regressor_heldout')
    ]

    scores = seed_scores(_per_query(), _predictions(rows), repeats=1)

    assert sorted(scores.get_column('panel').unique().to_list()) == sorted(PANELS)
    with pytest.raises(ValueError, match='regressor_heldout'):
        seed_scores(_per_query(), _predictions(rows[:1]), repeats=1)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_decision_scores.py -q`
Expected: one collection error, `ModuleNotFoundError: No module named 'naics_embedder.decision'`.

- [x] **Step 3: Implement**

> Deviation: the task review found that a row missing under one comparator dropped silently; by user ruling, `regressor_scores` also refuses a panel whose rows do not all appear under every comparator it has, with one more `pytest.raises` inside `test_a_row_without_every_repeat_is_refused` (35301c7; counts unchanged).

Create `src/naics_embedder/decision/__init__.py` with exactly this content:

```python
'''
Req 5's decision procedure over D8's three panels (roadmap Stage 4; D8, D10, D11).

- ``scores``: each panel's per-unit scores and its decision statistic (D10).
- ``resampling``: the paired two-stage bootstrap, units shared by every arm, seeds nested.
- ``rule``: the intervals, non-inferiority, superiority, the non-dominated set, the tie order.
- ``records``: the arm, margin and decision records.
- ``store``: the content-addressed store the records' artifact references point into.
- ``decide``: margins from a reference arm, and a decision over arms, with every guard.
- ``sweep``: the seed-sweep driver that runs a configuration for N seeds.
'''
```

Create `src/naics_embedder/decision/scores.py` with exactly this content:

```python
'''
Each panel's per-unit scores and its decision statistic (D10).

Scores are long: one row per panel, statistic and item, with the item's resampling unit.

- **Outcome panel.** An item is a validation query and its unit is its true code. Every Req 3
  metric is a statistic; the decision statistic is ``mrr``, the per-query reciprocal rank that D6
  already selects checkpoints on.
- **Regressor regimes.** An item is a level-6 code in a feature year and its unit is its
  four-digit group. Every Req 2 comparator is a statistic, valued at the row's squared error
  averaged over the read's repeats, so a row counts once however many repeats scored it. The
  decision statistic is the mean squared error of ``covariates+embedding``.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Dict, Tuple

import polars as pl

from naics_embedder.panels.decoding import HIT_KS
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import DECISION_LEVEL, PANEL_NAMES, VALIDATION, Regime

SEEN_PANEL = PANEL_NAMES[Regime.SEEN]
HELDOUT_PANEL = PANEL_NAMES[Regime.HELDOUT]
# D8: the outcome panel and the two regressor regimes
PANELS = (OUTCOME_PANEL, SEEN_PANEL, HELDOUT_PANEL)
REGRESSOR_PANELS = (SEEN_PANEL, HELDOUT_PANEL)
EMBEDDING_COMPARATOR = 'covariates+embedding'
DECISION_STATISTIC = {
    OUTCOME_PANEL: 'mrr',
    SEEN_PANEL: EMBEDDING_COMPARATOR,
    HELDOUT_PANEL: EMBEDDING_COMPARATOR,
}
# Δ = orientation × (A − B) is positive when it favours A: MRR is higher-better, MSE lower-better
ORIENTATION = {OUTCOME_PANEL: 1.0, SEEN_PANEL: -1.0, HELDOUT_PANEL: -1.0}
# Req 1's gain is over the sparse encoding each regime can use (D10)
SPARSE_COMPARATOR = {
    SEEN_PANEL: 'covariates+one_hot',
    HELDOUT_PANEL: 'covariates+ancestors',
}
STATISTIC_DEFINITIONS = {
    OUTCOME_PANEL: 'mean reciprocal rank over validation queries; unit: a code with its queries',
    SEEN_PANEL: (
        'mean squared error of covariates+embedding on level-6 log employment, each row averaged '
        'over its repeats; unit: a four-digit group'
    ),
    HELDOUT_PANEL: (
        'mean squared error of covariates+embedding on level-6 log employment, each row averaged '
        'over its repeats; unit: a four-digit group'
    ),
}
SCORE_SCHEMA = {
    'panel': pl.Utf8,
    'statistic': pl.Utf8,
    'unit': pl.Utf8,
    'item': pl.Utf8,
    'feature_year': pl.Int32,
    'value': pl.Float64,
}
SCORE_COLUMNS: Tuple[str, ...] = tuple(SCORE_SCHEMA)

# -------------------------------------------------------------------------------------------------
# Per-unit scores
# -------------------------------------------------------------------------------------------------

def outcome_scores(per_query: pl.DataFrame) -> pl.DataFrame:
    '''
    The outcome panel's scores from ``DecodingResult.per_query``: every Req 3 metric per query.

    Raises:
        ValueError: If a query id repeats.
    '''

    if per_query.get_column('query_id').is_duplicated().any():
        raise ValueError('the decoding scores repeat a query id')
    metrics = {
        'top1': pl.col('rank') == 1,
        'mrr': pl.col('reciprocal_rank'),
        **{
            f'hit_at_{k}': pl.col(f'hit_at_{k}')
            for k in HIT_KS
        },
        'lca_level': pl.col('lca_level'),
    }
    wide = per_query.select(
        unit=pl.col('code'),
        item=pl.col('query_id').cast(pl.Utf8),
        **{
            name: expr.cast(pl.Float64)
            for name, expr in metrics.items()
        },
    )
    long = wide.unpivot(
        index=['unit', 'item'], on=list(metrics), variable_name='statistic', value_name='value'
    )
    return long.with_columns(
        panel=pl.lit(OUTCOME_PANEL),
        feature_year=pl.lit(None, dtype=pl.Int32),
    ).select(SCORE_COLUMNS).cast(SCORE_SCHEMA)

def regressor_scores(predictions: pl.DataFrame, repeats: int) -> pl.DataFrame:
    '''
    Both regimes' scores from level-6 validation predictions: each comparator's squared error per
    row (a code in a feature year), averaged over the row's repeats.

    Raises:
        ValueError: If a row is not a level-6 validation row of a regressor panel, or a row does not
            have exactly ``repeats`` predictions under a comparator.
    '''

    outside = predictions.filter(
        (pl.col('split') != VALIDATION) | (pl.col('level') != DECISION_LEVEL)
        | ~pl.col('panel').is_in(list(REGRESSOR_PANELS))
    )
    if outside.height:
        raise ValueError(
            f'{outside.height:,} predictions are not level-{DECISION_LEVEL} validation rows of '
            f'{list(REGRESSOR_PANELS)}'
        )
    keys = ['panel', 'comparator', 'group', 'code', 'feature_year']
    # yapf: disable
    rows = (
        predictions
        .with_columns(error=(pl.col('prediction') - pl.col('outcome'))**2)
        .group_by(keys)
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
    return rows.select(
        'panel',
        statistic=pl.col('comparator'),
        unit=pl.col('group'),
        item=pl.concat_str(pl.col('code'), pl.col('feature_year').cast(pl.Utf8), separator='/'),
        feature_year=pl.col('feature_year'),
        value=pl.col('value'),
    ).cast(SCORE_SCHEMA).sort('panel', 'statistic', 'unit', 'item')

def seed_scores(per_query: pl.DataFrame, predictions: pl.DataFrame, repeats: int) -> pl.DataFrame:
    '''
    One seed's scores on all three panels.

    Raises:
        ValueError: If a panel has no scores.
    '''

    scores = pl.concat([outcome_scores(per_query), regressor_scores(predictions, repeats)])
    missing = sorted(set(PANELS) - set(scores.get_column('panel').unique().to_list()))
    if missing:
        raise ValueError(f'no scores for {missing}')
    return scores

def statistic_values(scores: pl.DataFrame, panel: str, statistic: str) -> pl.DataFrame:
    '''
    One statistic's items (``unit``, ``item``, ``feature_year``, ``value``) on one panel.

    Raises:
        ValueError: If the scores hold no such items.
    '''

    rows = scores.filter((pl.col('panel') == panel) & (pl.col('statistic') == statistic))
    if not rows.height:
        raise ValueError(f'no {statistic!r} scores on {panel}')
    return rows.select('unit', 'item', 'feature_year', 'value')

def panel_statistic(scores: pl.DataFrame, panel: str, statistic: str) -> float:
    '''A statistic's mean over its items: one seed's value on every unit.'''

    return float(statistic_values(scores, panel, statistic).get_column('value').mean())

def statistic_means(scores: pl.DataFrame) -> Dict[str, Dict[str, float]]:
    '''Every statistic's mean over its items, by panel.'''

    means: Dict[str, Dict[str, float]] = {}
    for panel, statistic, value in scores.group_by('panel', 'statistic').agg(
        pl.col('value').mean()
    ).sort('panel', 'statistic').iter_rows():
        means.setdefault(panel, {})[statistic] = float(value)
    return means
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_scores.py -q`
Expected: `7 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1551 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/decision/__init__.py src/naics_embedder/decision/scores.py tests/unit/test_decision_scores.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/decision/__init__.py \
  src/naics_embedder/decision/scores.py \
  tests/unit/test_decision_scores.py
git commit -m "feat(decision): add each panel's per-unit scores and decision statistic"
```

### Task 3: The paired two-stage bootstrap

This task implements Req 5's pairing with seeds nested (user decision 1). Each replicate draws
a panel's units with replacement, and every arm shares that draw. Inside it, each arm's seeds
are drawn with replacement, independently per arm. Both draws are multinomial counts from
`default_rng` streams keyed by the panel and the arm, so a decision is reproducible, and an arm's
replicates are the same in every comparison it enters. A replicate's statistic weights each
drawn seed's unit sums by the unit counts.

**Files:**

- Create: `src/naics_embedder/decision/resampling.py` (the paired two-stage bootstrap)
- Create: `tests/unit/test_decision_resampling.py`

**Interfaces:**

- Consumes: `statistic_values(scores, panel, statistic) -> pl.DataFrame` (Task 2), whose columns
  include `unit`, `item` and `value`.
- Produces, in `naics_embedder.decision.resampling`:
  - `stream(name: str) -> int`: the first 16 hex digits of the name's sha256, as an integer;
  - `PanelItems`, a frozen dataclass with `items: Tuple[str, ...]`, `units: Tuple[str, ...]` and
    `unit_index: np.ndarray`. It has `PanelItems.from_values(frame) -> PanelItems`, the property
    `sizes -> np.ndarray` (items per unit), `values(frame) -> np.ndarray` (a seed's values in item
    order, refusing another item set) and `sums(values) -> np.ndarray` (per-unit sums);
  - `unit_draws(panel: str, n_units: int, replicates: int, bootstrap_seed: int) -> np.ndarray`,
    of shape (replicates, n_units);
  - `seed_draws(panel: str, arm: str, n_seeds: int, replicates: int, bootstrap_seed: int) ->
    np.ndarray`, of shape (replicates, n_seeds);
  - `replicate_statistics(sums, sizes, units, seeds) -> np.ndarray`, with `sums` of shape
    (n_seeds, n_units);
  - `point_statistic(sums: np.ndarray, sizes: np.ndarray) -> float`;
  - `percentile_interval(replicates: Sequence[float], level: float) -> Tuple[float, float]`.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_decision_resampling.py` with exactly this content:

```python
'''
The paired two-stage bootstrap (Req 5): units shared by every arm, seeds nested and drawn per arm.
'''

import hashlib

import numpy as np
import polars as pl
import pytest

from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    stream,
    unit_draws,
)

pytestmark = pytest.mark.unit

def _frame(units, items, values):
    return pl.DataFrame({'unit': units, 'item': items, 'value': values})

def test_a_stream_is_the_first_sixteen_hex_digits_of_the_names_sha256():
    assert stream('outcome') == int(hashlib.sha256(b'outcome').hexdigest()[:16], 16)
    assert stream('outcome') != stream('regressor_seen')

def test_unit_draws_depend_on_the_panel_alone_and_seed_draws_on_the_arm_too():
    units = unit_draws('outcome', 7, 50, 20260924)

    assert units.shape == (50, 7)
    assert (units.sum(axis=1) == 7).all()
    np.testing.assert_array_equal(units, unit_draws('outcome', 7, 50, 20260924))
    assert not np.array_equal(units, unit_draws('regressor_seen', 7, 50, 20260924))
    seeds = seed_draws('outcome', 'A', 5, 50, 20260924)
    assert seeds.shape == (50, 5)
    assert (seeds.sum(axis=1) == 5).all()
    np.testing.assert_array_equal(seeds, seed_draws('outcome', 'A', 5, 50, 20260924))
    assert not np.array_equal(seeds, seed_draws('outcome', 'B', 5, 50, 20260924))

def test_a_replicate_is_the_mean_over_drawn_seeds_of_the_item_mean_over_drawn_units():
    # Unit a holds items 1 and 2, unit b item 3; two seeds
    items = PanelItems.from_values(_frame(['b', 'a', 'a'], ['3', '2', '1'], [0.0, 0.0, 0.0]))
    values = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 8.0]])
    sums = items.sums(values)
    units = np.array([[2, 0], [0, 2], [1, 1]])
    seeds = np.array([[2, 0], [1, 1], [0, 2]])

    replicates = replicate_statistics(sums, items.sizes, units, seeds)

    assert items.items == ('1', '2', '3')
    np.testing.assert_array_equal(sums, [[4.0, 5.0], [4.0, 8.0]])
    # Seed 0 on unit a; both seeds on unit b; seed 1 on both units
    np.testing.assert_allclose(replicates, [2.0, 6.5, 4.0])
    assert point_statistic(sums, items.sizes) == pytest.approx((9 / 3 + 12 / 3) / 2)

def test_pairing_cancels_what_the_arms_share_whatever_the_units_drawn():
    rng = np.random.default_rng(0)
    units = [f'u{index // 3:02d}' for index in range(60)]
    base = rng.uniform(0.0, 10.0, size=60)
    items = PanelItems.from_values(_frame(units, [str(index) for index in range(60)], base))
    a = items.sums(np.tile(base, (5, 1)) + 0.1)
    b = items.sums(np.tile(base, (5, 1)))
    draws = unit_draws('outcome', len(items.units), 200, 1)

    delta = replicate_statistics(
        a, items.sizes, draws, seed_draws('outcome', 'A', 5, 200, 1)
    ) - replicate_statistics(b, items.sizes, draws, seed_draws('outcome', 'B', 5, 200, 1))
    unpaired = replicate_statistics(
        a, items.sizes, draws, seed_draws('outcome', 'A', 5, 200, 1)
    ) - replicate_statistics(
        b, items.sizes, unit_draws('other', len(items.units), 200, 1),
        seed_draws('outcome', 'B', 5, 200, 1)
    )

    np.testing.assert_allclose(delta, 0.1)
    assert unpaired.std() > 0.1

def test_paired_arms_must_share_items_and_units():
    items = PanelItems.from_values(_frame(['a', 'b'], ['1', '2'], [0.0, 0.0]))

    with pytest.raises(ValueError, match='same items'):
        items.values(_frame(['a', 'b'], ['1', '3'], [0.0, 0.0]))
    with pytest.raises(ValueError, match='share units'):
        items.values(_frame(['a', 'a'], ['1', '2'], [0.0, 0.0]))
    with pytest.raises(ValueError, match='expected 2'):
        items.values(_frame(['a'], ['1'], [0.0]))
    with pytest.raises(ValueError, match='repeats'):
        PanelItems.from_values(_frame(['a', 'b'], ['1', '1'], [0.0, 0.0]))

def test_the_interval_is_two_sided_percentiles():
    assert percentile_interval(np.arange(101.0), 0.9) == pytest.approx((5.0, 95.0))
    lower, upper = percentile_interval(np.arange(101.0), 1 - 0.05 / 3)
    assert (lower, upper) == pytest.approx((100 * 0.05 / 6, 100 - 100 * 0.05 / 6))
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_decision_resampling.py -q`
Expected: one collection error,
`ModuleNotFoundError: No module named 'naics_embedder.decision.resampling'`.

- [x] **Step 3: Implement**

Create `src/naics_embedder/decision/resampling.py` with exactly this content:

```python
'''
The paired two-stage bootstrap (Req 5: "both arms are scored on the same resample of the
evaluation unit, with seeds nested within the resample").

Each replicate draws the panel's units with replacement, one draw shared by every arm, and then,
inside it, each arm's seeds with replacement, a draw of the arm's own. The replicate's statistic
is the mean over the drawn seeds of the item-weighted mean over the drawn units:

    sum_s m_s (C . S_s) / (n_seeds (C . N))

with C the unit counts, m the seed counts, S_s seed s's per-unit sums and N the items per unit.
Unit counts come from the generator seeded ``[bootstrap_seed, stream(panel)]`` and an arm's seed
counts from ``[bootstrap_seed, stream(panel), stream(arm)]``, so an arm's replicates are the same
in every comparison it enters.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import polars as pl

# -------------------------------------------------------------------------------------------------
# Items and units
# -------------------------------------------------------------------------------------------------

def stream(name: str) -> int:
    '''A generator stream for a name: the first 16 hex digits of its sha256, as an integer.'''

    return int(hashlib.sha256(name.encode('utf-8')).hexdigest()[:16], 16)

@dataclass(frozen=True)
class PanelItems:
    '''One statistic's items in a fixed order (by unit, then item), with each item's unit.'''

    items: Tuple[str, ...]
    units: Tuple[str, ...]
    unit_index: np.ndarray

    @classmethod
    def from_values(cls, frame: pl.DataFrame) -> 'PanelItems':
        '''
        The items of a frame with ``unit`` and ``item`` columns.

        Raises:
            ValueError: If an item repeats.
        '''

        if frame.get_column('item').is_duplicated().any():
            raise ValueError('an item repeats')
        ordered = frame.select('unit', 'item').sort('unit', 'item')
        units = tuple(ordered.get_column('unit').unique(maintain_order=True).to_list())
        position = {unit: index for index, unit in enumerate(units)}
        unit_index = np.array(
            [position[unit] for unit in ordered.get_column('unit').to_list()], dtype=np.int64
        )
        return cls(tuple(ordered.get_column('item').to_list()), units, unit_index)

    @property
    def sizes(self) -> np.ndarray:
        '''Items per unit, as floats.'''

        return np.bincount(self.unit_index, minlength=len(self.units)).astype(np.float64)

    def values(self, frame: pl.DataFrame) -> np.ndarray:
        '''
        The frame's ``value`` column in this order.

        Raises:
            ValueError: If the frame holds other items, or puts an item in another unit: paired
                arms must be scored on the same items (Req 5).
        '''

        if frame.height != len(self.items):
            raise ValueError(f'{frame.height:,} items, expected {len(self.items):,}')
        ordered = frame.sort('unit', 'item')
        if tuple(ordered.get_column('item').to_list()) != self.items:
            raise ValueError('the items differ: paired arms must be scored on the same items')
        expected = [self.units[index] for index in self.unit_index]
        if ordered.get_column('unit').to_list() != expected:
            raise ValueError('an item sits in another unit: paired arms must share units')
        return ordered.get_column('value').to_numpy().astype(np.float64)

    def sums(self, values: np.ndarray) -> np.ndarray:
        '''Per-unit sums of (seeds, items) values: (seeds, units).'''

        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        return np.stack(
            [
                np.bincount(self.unit_index, weights=row, minlength=len(self.units))
                for row in values
            ]
        )

# -------------------------------------------------------------------------------------------------
# Draws and replicates
# -------------------------------------------------------------------------------------------------

def unit_draws(panel: str, n_units: int, replicates: int, bootstrap_seed: int) -> np.ndarray:
    '''Unit counts per replicate, (replicates, n_units), shared by every arm on the panel.'''

    rng = np.random.default_rng([bootstrap_seed, stream(panel)])
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=replicates)

def seed_draws(
    panel: str, arm: str, n_seeds: int, replicates: int, bootstrap_seed: int
) -> np.ndarray:
    '''One arm's seed counts per replicate, (replicates, n_seeds).'''

    rng = np.random.default_rng([bootstrap_seed, stream(panel), stream(arm)])
    return rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=replicates)

def replicate_statistics(
    sums: np.ndarray, sizes: np.ndarray, units: np.ndarray, seeds: np.ndarray
) -> np.ndarray:
    '''
    Each replicate's statistic.

    Args:
        sums: Per-seed unit sums, (n_seeds, n_units).
        sizes: Items per unit, (n_units,).
        units: Unit counts, (replicates, n_units).
        seeds: Seed counts, (replicates, n_seeds).
    '''

    per_seed = units @ sums.T
    items = units @ sizes
    return (seeds * per_seed).sum(axis=1) / (seeds.sum(axis=1) * items)

def point_statistic(sums: np.ndarray, sizes: np.ndarray) -> float:
    '''The mean over seeds of each seed's statistic on every unit.'''

    return float((sums.sum(axis=1) / sizes.sum()).mean())

def percentile_interval(replicates: Sequence[float], level: float) -> Tuple[float, float]:
    '''The two-sided percentile interval at ``level``.'''

    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(np.asarray(replicates, dtype=np.float64), [tail, 1.0 - tail])
    return float(lower), float(upper)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_resampling.py -q`
Expected: `6 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1557 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/decision/resampling.py tests/unit/test_decision_resampling.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/decision/resampling.py \
  tests/unit/test_decision_resampling.py
git commit -m "feat(decision): add the paired two-stage bootstrap over units and seeds"
```

### Task 4: Req 5's rule, the tie order and the record schema

The rule reads Δ's replicates against δ:

- **Non-inferior:** the 95 % interval's lower bound is above −δ.
- **Superior:** the 98⅓ % interval's lower bound is above zero (D8).
- **Adopted:** non-inferior on all three panels and superior on at least one.

Among several arms, the survivors are the arms no other arm is adopted over. If none survives,
dominance cycled, and every arm goes to the tie order. The tie order is:

1. fewer components;
2. then lower dimension;
3. then non-hyperbolic geometry;
4. then the higher held-out gain over ancestors (D11).

Arms equal on every key raise `TieUnresolvedError`.

The records are frozen Pydantic models that forbid unknown keys, and are written once as JSON.

**Files:**

- Create: `src/naics_embedder/decision/records.py` (arm, margin and decision records)
- Create: `src/naics_embedder/decision/rule.py` (intervals, verdicts, the non-dominated set, the tie
  order)
- Create: `tests/unit/test_decision_rule.py`

**Interfaces:**

- Consumes: `percentile_interval(replicates, level)` (Task 3), in `rule.py`. Records name
  panels by string.
- Produces, in `naics_embedder.decision.records`. Every model is frozen with `extra='forbid'`:
  - `ArtifactRef(path: str, sha256: str, bytes: int)`, with the path relative to the store's
    root;
  - `TableRef(ArtifactRef)`, adding `matrix_fingerprint: str`;
  - `TextOnlyRef(table: TableRef, provenance: ArtifactRef, backbone: str, revision:
    Optional[str], descriptions_sha256: str, max_length: int)`;
  - `ArmSpec(name, components: int ≥ 1, dimension: int ≥ 1, geometry: Geometry, backbone,
    backbone_revision: Optional[str], descriptions_sha256, max_length: int ≥ 1, settings:
    Dict[str, Any] = {})`. Here `Geometry` is `Literal['euclidean', 'spherical', 'hyperbolic']`;
  - `PanelSet(outcome: str, regressor: str, fit_settings: Dict[str, Any])`, holding the two
    panels' fingerprints;
  - `SeedRun(seed, run_id, checkpoint: ArtifactRef, table: TableRef, scores: ArtifactRef,
    decoding: ArtifactRef, predictions: ArtifactRef, statistics: Dict[str, float], log_records:
    List[Dict[str, Any]])`;
  - `ArmRecord(kind='arm', spec: ArmSpec, text_only: TextOnlyRef, store: str, panels: PanelSet,
    runs: List[SeedRun], created_at: datetime)`;
  - `PanelMargin(panel, statistic, per_seed: List[float], sd: float, margin: float)`;
  - `MarginRecord(kind='margins', name, multiple: float > 0, reference: ArmRecord, margins:
    List[PanelMargin], fixed_at: datetime)`, with `margin(panel) -> float`;
  - `PanelComparison(panel, delta, noninferiority_interval, superiority_interval, margin,
    non_inferior: bool, superior: bool)`;
  - `Comparison(a, b, panels: List[PanelComparison], adopted: bool)`;
  - `Estimate(point: float, interval: Tuple[float, float])`;
  - `ArmReport(arm, statistics, outcome_metrics, comparator_mse, gain: Dict[str, Estimate],
    heldout_by_feature_year)`;
  - `DecisionSettings(replicates, bootstrap_seed, min_seeds, noninferiority_level,
    superiority_level)`;
  - `DecisionRecord(kind='decision', name, question, created_at, statistics: Dict[str, str],
    settings, margins: MarginRecord, arms: List[ArmRecord], comparisons: List[Comparison],
    non_dominated: List[str], cycle: bool, tie_order: List[str], heldout_gain: Dict[str, float],
    chosen: str, reports: List[ArmReport])`;
  - `write_record(record: BaseModel, path) -> Path`, which raises `FileExistsError` rather than
    overwrite, and `read_record(path, kind: Type[RecordType]) -> RecordType`.
- Produces, in `naics_embedder.decision.rule`:
  - `NONINFERIORITY_LEVEL = 0.95` and `SUPERIORITY_LEVEL = 1.0 - 0.05 / 3`;
  - `TieUnresolvedError(RuntimeError)`;
  - `compare_panel(panel: str, delta: float, replicates: np.ndarray, margin: float) ->
    PanelComparison`;
  - `compare(a: str, b: str, panels: Sequence[PanelComparison]) -> Comparison`;
  - `non_dominated(arms: Sequence[str], comparisons: Sequence[Comparison]) -> Tuple[List[str],
    bool]`, returning the survivors and whether dominance cycled;
  - `tie_order(specs: Sequence[ArmSpec], heldout_gain: Mapping[str, float]) -> List[str]`.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_decision_rule.py` with exactly this content:

```python
'''
Req 5's rule over D8's three panels: 95 % non-inferiority, 98⅓ % superiority, adoption, the
non-dominated set and the tie order (D11 breaks the last tie).
'''

import numpy as np
import pytest

from naics_embedder.decision.records import ArmSpec
from naics_embedder.decision.rule import (
    NONINFERIORITY_LEVEL,
    SUPERIORITY_LEVEL,
    TieUnresolvedError,
    compare,
    compare_panel,
    non_dominated,
    tie_order,
)

pytestmark = pytest.mark.unit

def _spec(name, components=1, dimension=16, geometry='hyperbolic'):
    return ArmSpec(
        name=name,
        components=components,
        dimension=dimension,
        geometry=geometry,
        backbone='b',
        backbone_revision='r',
        descriptions_sha256='d',
        max_length=8,
    )

def _panel(panel, low, high, margin=1.0):
    # Evenly spread replicates: every percentile is known exactly
    replicates = np.linspace(low, high, 12001)
    return compare_panel(panel, (low + high) / 2, replicates, margin)

def test_the_levels_are_d8s():
    assert NONINFERIORITY_LEVEL == 0.95
    assert SUPERIORITY_LEVEL == pytest.approx(0.98333333)

def test_non_inferiority_reads_the_95_interval_and_superiority_the_98_one_third_interval():
    # Replicates spread evenly over [-1, 3]: the 95 % interval is (-0.9, 2.9) and the 98⅓ %
    # interval (-0.9667, 2.9667)
    within = _panel('outcome', -1.0, 3.0, margin=0.95)
    outside = _panel('outcome', -1.0, 3.0, margin=0.85)

    assert within.noninferiority_interval == pytest.approx((-0.9, 2.9))
    assert within.superiority_interval == pytest.approx((-1 + 4 / 120, 3 - 4 / 120))
    assert within.non_inferior and not within.superior
    assert not outside.non_inferior
    assert _panel('outcome', 0.1, 2.0).superior
    # Above zero at 95 % (0.05) but not at 98⅓ % (-0.0167)
    borderline = _panel('outcome', -0.05, 3.95)
    assert borderline.noninferiority_interval[0] > 0 and not borderline.superior

def test_adoption_needs_non_inferiority_everywhere_and_superiority_somewhere():
    superior = _panel('outcome', 0.5, 1.5)
    level = _panel('regressor_seen', -0.5, 0.5)
    inferior = _panel('regressor_heldout', -3.0, -1.5)

    assert compare('A', 'B', [superior, level, level]).adopted
    assert not compare('A', 'B', [level, level, level]).adopted
    assert not compare('A', 'B', [superior, level, inferior]).adopted

def _comparison(a, b, adopted):
    panel = _panel('outcome', 0.5, 1.5) if adopted else _panel('outcome', -0.5, 0.5)
    return compare(a, b, [panel])

def test_the_survivors_are_the_arms_no_other_arm_is_adopted_over():
    comparisons = [_comparison('A', 'B', True), _comparison('B', 'C', False)]

    assert non_dominated(['A', 'B', 'C'], comparisons) == (['A', 'C'], False)

def test_when_dominance_cycles_every_arm_survives():
    comparisons = [
        _comparison('A', 'B', True),
        _comparison('B', 'C', True),
        _comparison('C', 'A', True),
    ]

    assert non_dominated(['A', 'B', 'C'], comparisons) == (['A', 'B', 'C'], True)

def test_the_tie_order_prefers_fewer_components_then_lower_dimension_then_flat_geometry():
    specs = [
        _spec('two-stage', components=2, dimension=8, geometry='euclidean'),
        _spec('wide', dimension=32, geometry='euclidean'),
        _spec('small-hyperbolic', dimension=16, geometry='hyperbolic'),
        _spec('small-flat', dimension=16, geometry='spherical'),
    ]
    gain = {'two-stage': 9.0, 'wide': 9.0, 'small-hyperbolic': 9.0, 'small-flat': 0.0}

    assert tie_order(specs, gain) == ['small-flat', 'small-hyperbolic', 'wide', 'two-stage']

def test_the_last_tie_goes_to_the_higher_held_out_gain():
    specs = [_spec('euclidean', geometry='euclidean'), _spec('spherical', geometry='spherical')]

    assert tie_order(specs, {'euclidean': 0.01, 'spherical': 0.02}) == ['spherical', 'euclidean']
    with pytest.raises(TieUnresolvedError, match='euclidean and spherical'):
        tie_order(specs, {'euclidean': 0.02, 'spherical': 0.02})
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_decision_rule.py -q`
Expected: one collection error,
`ModuleNotFoundError: No module named 'naics_embedder.decision.records'`.

- [x] **Step 3: Implement**

> Deviation: the task review found that a NaN or infinite float was written as null and then failed to read back from a write-once path; by user ruling, `_Record`'s config gained `allow_inf_nan=False`, with one `pytest.raises(ValidationError)` inside `test_non_inferiority_reads_the_95_interval_and_superiority_the_98_one_third_interval` (c1a728d; counts unchanged).

Create `src/naics_embedder/decision/records.py` with exactly this content:

```python
'''
The records of Req 5's decisions (Verification "Decision records"; roadmap Stage 4).

Three kinds, each one JSON file:

- **Arm record.** One configuration's seed sweep: its spec, the text-only table it was paired
  with and that table's provenance (D9), the panels it read, and per seed the encoder checkpoint,
  the 2,125-code table, the stored scores and the selection-log records of the run's reads (the
  log itself is gitignored and dies with its worktree or Lambda instance).
- **Margin record.** Each panel's δ, a stated multiple of a reference arm's across-seed standard
  deviation, fixed before any other arm's first read.
- **Decision record.** Its arms and margins, every paired comparison with its 95 %
  non-inferiority and 98⅓ % superiority intervals (D8), the non-dominated set, the tie order,
  the chosen arm, and the reported statistics.

Artifact references are paths relative to an ``ArtifactStore`` root with their sha256, so a
record names exactly the bytes it was computed from.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

Geometry = Literal['euclidean', 'spherical', 'hyperbolic']
RecordType = TypeVar('RecordType', bound=BaseModel)

class _Record(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

# -------------------------------------------------------------------------------------------------
# Artifacts
# -------------------------------------------------------------------------------------------------

class ArtifactRef(_Record):
    '''A stored file: its path relative to the store root, its sha256 and its size.'''

    path: str
    sha256: str
    bytes: int = Field(ge=0)

class TableRef(ArtifactRef):
    '''A stored code table, with the ``matrix_fingerprint`` the selection log names it by.'''

    matrix_fingerprint: str

class TextOnlyRef(_Record):
    '''An arm's text-only table and its provenance, with the provenance fields D9 checks.'''

    table: TableRef
    provenance: ArtifactRef
    backbone: str
    revision: Optional[str]
    descriptions_sha256: str
    max_length: int

# -------------------------------------------------------------------------------------------------
# Arms
# -------------------------------------------------------------------------------------------------

class ArmSpec(_Record):
    '''
    A configuration, with what the tie order and D9 read.

    Attributes:
        components: Stages or post-processing steps: Req 5's "fewer components".
        backbone, backbone_revision, descriptions_sha256, max_length: What the arm's encoder
            reads, which its text-only table must match (D9).
        settings: The configuration's own settings, recorded as given.
    '''

    name: str = Field(min_length=1)
    components: int = Field(ge=1)
    dimension: int = Field(ge=1)
    geometry: Geometry
    backbone: str
    backbone_revision: Optional[str]
    descriptions_sha256: str
    max_length: int = Field(ge=1)
    settings: Dict[str, Any] = Field(default_factory=dict)

class PanelSet(_Record):
    '''The panels a run read; every arm of a decision must share them (Req 5's pairing).'''

    outcome: str
    regressor: str
    fit_settings: Dict[str, Any]

class SeedRun(_Record):
    '''
    One seed of an arm.

    Attributes:
        scores: The seed's scores on all three panels (``decision.scores.SCORE_COLUMNS``).
        decoding: ``DecodingResult.per_query`` of the outcome read.
        predictions: Both regimes' level-6 validation predictions.
        statistics: Each panel's decision statistic on every unit.
        log_records: The selection-log records of this run's reads.
    '''

    seed: int
    run_id: str
    checkpoint: ArtifactRef
    table: TableRef
    scores: ArtifactRef
    decoding: ArtifactRef
    predictions: ArtifactRef
    statistics: Dict[str, float]
    log_records: List[Dict[str, Any]]

class ArmRecord(_Record):
    '''One configuration's seed sweep (``decision.sweep.run_seed_sweep``).'''

    kind: Literal['arm'] = 'arm'
    spec: ArmSpec
    text_only: TextOnlyRef
    store: str
    panels: PanelSet
    runs: List[SeedRun]
    created_at: datetime

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

class PanelMargin(_Record):
    '''One panel's δ: the multiple times the reference's across-seed standard deviation.'''

    panel: str
    statistic: str
    per_seed: List[float]
    sd: float
    margin: float

class MarginRecord(_Record):
    '''Each panel's δ, fixed from a reference arm before any other arm's first read (Req 5).'''

    kind: Literal['margins'] = 'margins'
    name: str
    multiple: float = Field(gt=0)
    reference: ArmRecord
    margins: List[PanelMargin]
    fixed_at: datetime

    def margin(self, panel: str) -> float:
        '''The panel's δ.'''

        return next(entry.margin for entry in self.margins if entry.panel == panel)

# -------------------------------------------------------------------------------------------------
# Decisions
# -------------------------------------------------------------------------------------------------

class PanelComparison(_Record):
    '''Δ on one panel, oriented so that a positive value favours A, with both intervals.'''

    panel: str
    delta: float
    noninferiority_interval: Tuple[float, float]
    superiority_interval: Tuple[float, float]
    margin: float
    non_inferior: bool
    superior: bool

class Comparison(_Record):
    '''A against B: adopted when non-inferior on every panel and superior on at least one.'''

    a: str
    b: str
    panels: List[PanelComparison]
    adopted: bool

class Estimate(_Record):
    '''A point estimate with its 95 % percentile interval.'''

    point: float
    interval: Tuple[float, float]

class ArmReport(_Record):
    '''
    What a decision reports for an arm besides the rule's statistics (D10).

    Attributes:
        statistics: Each panel's decision statistic.
        outcome_metrics: Every Req 3 metric on the outcome panel.
        comparator_mse: Each regime's mean squared error for every Req 2 comparator.
        gain: Each regime's gain over its sparse encoding (Req 1): the sparse comparator's mean
            squared error minus ``covariates+embedding``'s, on the decision's draws.
        heldout_by_feature_year: The held-out regime by feature year: ``covariates+embedding``,
            ``covariates+ancestors`` and the gain.
    '''

    arm: str
    statistics: Dict[str, float]
    outcome_metrics: Dict[str, float]
    comparator_mse: Dict[str, Dict[str, float]]
    gain: Dict[str, Estimate]
    heldout_by_feature_year: Dict[str, Dict[str, float]]

class DecisionSettings(_Record):
    '''How the decision resampled and which intervals it read.'''

    replicates: int
    bootstrap_seed: int
    min_seeds: int
    noninferiority_level: float
    superiority_level: float

class DecisionRecord(_Record):
    '''A Req 5 decision over two or more arms.'''

    kind: Literal['decision'] = 'decision'
    name: str
    question: str
    created_at: datetime
    statistics: Dict[str, str]
    settings: DecisionSettings
    margins: MarginRecord
    arms: List[ArmRecord]
    comparisons: List[Comparison]
    non_dominated: List[str]
    cycle: bool
    tie_order: List[str]
    heldout_gain: Dict[str, float]
    chosen: str
    reports: List[ArmReport]

# -------------------------------------------------------------------------------------------------
# Files
# -------------------------------------------------------------------------------------------------

def write_record(record: BaseModel, path: Union[str, Path]) -> Path:
    '''Write a record as indented JSON; refuse to overwrite a file.'''

    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; a record is written once')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + '\n', encoding='utf-8')
    return path

def read_record(path: Union[str, Path], kind: Type[RecordType]) -> RecordType:
    '''Read and validate a record of the given kind.'''

    return kind.model_validate_json(Path(path).read_text(encoding='utf-8'))
```

Create `src/naics_embedder/decision/rule.py` with exactly this content:

```python
'''
Req 5's rule on paired resamples, over D8's three panels.

- **Non-inferior** on a panel: the lower bound of the 95 % interval on Δ exceeds −δ.
- **Superior** on a panel: the 98⅓ % interval lies above zero. D8 gives each of the three panels
  a third of the 5 % error rate; this supersedes the 97.5 % that Req 5 names for two panels.
- **Adopted:** non-inferior on every panel and superior on at least one.
- **Several arms:** the survivors are the arms no other arm is adopted over; when every arm is,
  dominance cycles and the tie order picks among all of them. The pairwise comparisons are not
  corrected for multiplicity: the tie order toward the simpler arm is the guard (Req 5).
- **Tie order:** fewer components, then lower dimension, then non-hyperbolic geometry, then the
  higher held-out gain over ancestors (D11).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import List, Mapping, Sequence, Tuple

import numpy as np

from naics_embedder.decision.records import ArmSpec, Comparison, PanelComparison
from naics_embedder.decision.resampling import percentile_interval

NONINFERIORITY_LEVEL = 0.95
SUPERIORITY_LEVEL = 1.0 - 0.05 / 3

class TieUnresolvedError(RuntimeError):
    '''Two candidate arms tie on every key of the tie order.'''

# -------------------------------------------------------------------------------------------------
# Comparisons
# -------------------------------------------------------------------------------------------------

def compare_panel(
    panel: str, delta: float, replicates: np.ndarray, margin: float
) -> PanelComparison:
    '''
    One panel's verdict for A against B.

    Args:
        delta: The point estimate of Δ, positive when it favours A.
        replicates: Δ on each paired resample, oriented the same way.
        margin: The panel's δ.
    '''

    noninferiority = percentile_interval(replicates, NONINFERIORITY_LEVEL)
    superiority = percentile_interval(replicates, SUPERIORITY_LEVEL)
    return PanelComparison(
        panel=panel,
        delta=float(delta),
        noninferiority_interval=noninferiority,
        superiority_interval=superiority,
        margin=float(margin),
        non_inferior=noninferiority[0] > -margin,
        superior=superiority[0] > 0.0,
    )

def compare(a: str, b: str, panels: Sequence[PanelComparison]) -> Comparison:
    '''A against B on every panel.'''

    adopted = all(panel.non_inferior
                  for panel in panels) and any(panel.superior for panel in panels)
    return Comparison(a=a, b=b, panels=list(panels), adopted=adopted)

def non_dominated(arms: Sequence[str], comparisons: Sequence[Comparison]) -> Tuple[List[str], bool]:
    '''
    The arms no other arm is adopted over, and whether dominance cycled.

    Returns:
        ``(survivors, cycle)``: when every arm is dominated, ``cycle`` is true and every arm
        survives, for the tie order to pick among (Req 5).
    '''

    dominated = {comparison.b for comparison in comparisons if comparison.adopted}
    survivors = [arm for arm in arms if arm not in dominated]
    if survivors:
        return survivors, False
    return list(arms), True

def tie_order(specs: Sequence[ArmSpec], heldout_gain: Mapping[str, float]) -> List[str]:
    '''
    The arms from the one that stands first.

    Raises:
        TieUnresolvedError: If two arms tie on every key.
    '''

    def key(spec: ArmSpec) -> Tuple[int, int, bool, float]:
        return (
            spec.components,
            spec.dimension,
            spec.geometry == 'hyperbolic',
            -float(heldout_gain[spec.name]),
        )

    ordered = sorted(specs, key=key)
    for first, second in zip(ordered, ordered[1:]):
        if key(first) == key(second):
            raise TieUnresolvedError(
                f'{first.name} and {second.name} tie on components, dimension, geometry and the '
                'held-out gain'
            )
    return [spec.name for spec in ordered]
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_rule.py -q`
Expected: `7 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1564 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/decision/records.py src/naics_embedder/decision/rule.py tests/unit/test_decision_rule.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/decision/records.py \
  src/naics_embedder/decision/rule.py \
  tests/unit/test_decision_rule.py
git commit -m "feat(decision): add Req 5's rule, tie order and the record schema"
```

### Task 5: The artifact store and the synthetic arms

A record's references point into a content-addressed store:

- **Layout.** Files live at `objects/<sha[:2]>/<sha>/<name>` under a root outside any worktree.
- **Writes.** A file is written once, through a temporary file and `os.replace`.
- **Reads.** A file is verified against its reference on every read.

A text-only table enters the store with its provenance. The store refuses a provenance that is
missing, describes another file, or names another `matrix_fingerprint`. An arm table enters
with the `matrix_fingerprint` a regressor read logs it by.

The fixture module builds synthetic arms with known effects on the three panels. Each arm has
five seeds with fixed offsets, stored artifacts and a validation-read log record per panel. Task
6 decides among these arms.

**Files:**

- Create: `src/naics_embedder/decision/store.py` (the content-addressed artifact store)
- Create: `tests/fixtures/decision.py` (synthetic arms with known effects)
- Create: `tests/unit/test_decision_store.py`

**Interfaces:**

- Consumes:
  - `ArtifactRef`, `TableRef`, `TextOnlyRef`, `ArmSpec`, `PanelSet`, `SeedRun` and `ArmRecord`,
    and in the tests `read_record` and `write_record` (Task 4);
  - `table_fingerprint`, `text_only_fingerprint` and `provenance_path` (Task 1 and existing);
  - `sha256_file` (`naics_embedder.supervision.artifacts`);
  - `SCORE_SCHEMA`, `PANELS`, `SEEN_PANEL`, `HELDOUT_PANEL`, `DECISION_STATISTIC` and
    `panel_statistic` (Task 2);
  - `OUTCOME_PANEL` (`panels.outcome`);
  - in the tests, `coordinate_table` (existing, `tests.fixtures.regressor_panel`).
- Produces, in `naics_embedder.decision.store`:
  - `ArtifactStore(root)`, with `put(path) -> ArtifactRef`,
    `put_frame(frame: pl.DataFrame, name: str) -> ArtifactRef`, `put_table(path) -> TableRef`,
    `put_text_only(table_path) -> TextOnlyRef`, `resolve(reference: ArtifactRef) -> Path` and
    `read_frame(reference: ArtifactRef) -> pl.DataFrame`.
- Produces, in the fixture module `tests.fixtures.decision`. Tests import it directly; it is not
  a plugin:
  - `SIGMA = 0.01`, `SEED_OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)` and `SD`, the across-seed SD
    of a statistic built from those offsets;
  - `BACKBONE`, `REVISION`, `DESCRIPTIONS_SHA256` and `MAX_LENGTH`;
  - `PANEL_SET`, `GROUPS` (30 four-digit groups), `CODES` (60 six-digit codes) and
    `SPARSE_EXTRA`;
  - `spec(name: str, **overrides) -> ArmSpec`;
  - `synthetic_scores(effects: Mapping[str, float], offset: float) -> pl.DataFrame`;
  - `write_text_only(directory: Path, revision: str = REVISION, codes: Sequence[str] = CODES) ->
    Path`;
  - `synthetic_arm(store, directory, arm_spec, effects, *, offsets=SEED_OFFSETS,
    text_only_table=None) -> ArmRecord`.

- [x] **Step 1: Write the fixture module and the failing test**

> Deviation: F2 (user ruling at pre-flight): `write_text_only` builds its table with the regressor-panel fixture's `text_only_table`, imported as `stub_text_only_table` because `synthetic_arm` takes a `text_only_table=` keyword (5b895d4).

Create `tests/fixtures/decision.py` with exactly this content:

```python
'''
Synthetic arms with known effects on D8's three panels (roadmap Stage 4 Exit).

Every arm is scored on one set of items per panel, each with a base value. Seed k of an arm
scores item i at base_i + effect + offset_k on the outcome panel (reciprocal rank: higher is
better) and at base_i − effect + offset_k on each regressor regime (squared error: lower is
better), so a positive effect favours the arm on every panel. Effects are given in units of
``SIGMA``. The five offsets are fixed, ``SIGMA`` × (−2, −1, 0, 1, 2), so every arm's across-seed
standard deviation is ``SIGMA`` × √2.5 on every panel, and δ = multiple × that. Paired Δ cancels
the base values exactly: only the effects and the seed draws remain.

The records carry real artifact references (a store under the test's tmp path) and log records
shaped as the panels write them.
'''

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import polars as pl

from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import (
    DECISION_STATISTIC,
    HELDOUT_PANEL,
    PANELS,
    SCORE_SCHEMA,
    SEEN_PANEL,
    panel_statistic,
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.text_only import provenance_path, text_only_fingerprint
from naics_embedder.supervision.artifacts import sha256_file

SIGMA = 0.01
SEED_OFFSETS = (-2.0, -1.0, 0.0, 1.0, 2.0)
SD = SIGMA * float(np.sqrt(2.5))
BACKBONE = 'tiny-backbone'
REVISION = 'abc123'
DESCRIPTIONS_SHA256 = 'd' * 64
MAX_LENGTH = 16
PANEL_SET = PanelSet(
    outcome='outcome-roles',
    regressor='heldout-draw',
    fit_settings={
        'alphas': [0.1, 1.0],
        'folds': 2,
        'repeats': 2,
        'inner_folds': 2,
        'fold_seed': 20260924,
        'min_groups': 4,
    },
)
GROUPS = tuple(str(1100 + index) for index in range(30))
CODES = tuple(f'{group}{suffix}' for group in GROUPS for suffix in ('11', '12'))
# The sparse comparators never read the arm: their error is the base plus a constant
SPARSE_EXTRA = {SEEN_PANEL: 0.02, HELDOUT_PANEL: 0.03}

def spec(name: str, **overrides) -> ArmSpec:
    '''An arm spec reading the synthetic backbone and text.'''

    fields = {
        'name': name,
        'components': 1,
        'dimension': 16,
        'geometry': 'hyperbolic',
        'backbone': BACKBONE,
        'backbone_revision': REVISION,
        'descriptions_sha256': DESCRIPTIONS_SHA256,
        'max_length': MAX_LENGTH,
        **overrides,
    }
    return ArmSpec(**fields)

def _items(panel: str) -> pl.DataFrame:
    '''The panel's items (unit, item, feature_year) with their base values.'''

    rng = np.random.default_rng(len(panel))
    if panel == OUTCOME_PANEL:
        units = [code for code in CODES for _ in range(2)]
        items = [str(index) for index in range(len(units))]
        years: List[Optional[int]] = [None] * len(units)
        base = rng.uniform(0.2, 0.7, size=len(units))
    else:
        feature_years = (2023, ) if panel == SEEN_PANEL else (2022, 2023)
        rows = [(code, year) for code in CODES for year in feature_years]
        units = [code[:4] for code, _ in rows]
        items = [f'{code}/{year}' for code, year in rows]
        years = [year for _, year in rows]
        base = rng.uniform(0.05, 0.5, size=len(units))
    return pl.DataFrame(
        {
            'unit': units,
            'item': items,
            'feature_year': years,
            'base': base
        },
        schema={
            'unit': pl.Utf8,
            'item': pl.Utf8,
            'feature_year': pl.Int32,
            'base': pl.Float64
        },
    )

def synthetic_scores(effects: Mapping[str, float], offset: float) -> pl.DataFrame:
    '''One seed's scores (``SCORE_COLUMNS``), effects in units of ``SIGMA``.'''

    parts = []
    for panel in PANELS:
        items = _items(panel)
        base = pl.col('base')
        effect = effects.get(panel, 0.0) * SIGMA
        if panel == OUTCOME_PANEL:
            statistics = {
                'mrr': base + effect + offset,
                'top1': (base > 0.5).cast(pl.Float64),
                'hit_at_1': (base > 0.5).cast(pl.Float64),
                'hit_at_5': pl.lit(1.0),
                'hit_at_10': pl.lit(1.0),
                'lca_level': pl.lit(5.0),
            }
        else:
            sparse = 'covariates+one_hot' if panel == SEEN_PANEL else 'covariates+ancestors'
            statistics = {
                'covariates': base + 0.05,
                'covariates+embedding': base - effect + offset,
                sparse: base + SPARSE_EXTRA[panel],
            }
        for statistic, value in statistics.items():
            parts.append(
                items.select(
                    panel=pl.lit(panel),
                    statistic=pl.lit(statistic),
                    unit='unit',
                    item='item',
                    feature_year='feature_year',
                    value=value,
                )
            )
    return pl.concat(parts).cast(SCORE_SCHEMA)

def write_text_only(
    directory: Path, revision: str = REVISION, codes: Sequence[str] = CODES
) -> Path:
    '''A text-only table with its provenance, as ``tools text-only-table`` writes them.'''

    directory.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(11).normal(size=(len(codes), 5))
    schema = {f't{index}': pl.Float64 for index in range(5)}
    table = pl.DataFrame({
        'code': list(codes)
    }).hstack(pl.DataFrame(values, schema=schema, orient='row'))
    path = directory / 'text_only.parquet'
    table.write_parquet(path)
    provenance = {
        'backbone': BACKBONE,
        'revision': revision,
        'descriptions': {
            'path': 'naics_descriptions.parquet',
            'sha256': DESCRIPTIONS_SHA256
        },
        'max_length': MAX_LENGTH,
        'table_sha256': sha256_file(path),
        'matrix_fingerprint': text_only_fingerprint(table),
    }
    provenance_path(path).write_text(json.dumps(provenance, indent=2) + '\n')
    return path

def _table(directory: Path, name: str, seed: int, dimension: int) -> Path:
    '''An arm's code table in the export form.'''

    values = np.random.default_rng([seed, len(name)]).normal(size=(len(CODES), dimension))
    schema = {f'e{index}': pl.Float64 for index in range(dimension)}
    table = pl.DataFrame({
        'code': list(CODES)
    }).hstack(pl.DataFrame(values, schema=schema, orient='row'))
    path = directory / f'{name}-{seed}.parquet'
    table.write_parquet(path)
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
        'time': time,
        'event': 'read',
        'panel': panel,
        'split': 'validation',
        'purpose': 'synthetic arm',
        'fingerprint': fingerprint,
        'n_queries': 0,
        'detail': detail,
    }

def synthetic_arm(
    store: ArtifactStore,
    directory: Path,
    arm_spec: ArmSpec,
    effects: Mapping[str, float],
    *,
    offsets: Sequence[float] = SEED_OFFSETS,
    text_only_table: Optional[Path] = None,
) -> ArmRecord:
    '''An arm record whose seeds score ``synthetic_scores``, read now.'''

    directory = Path(directory) / arm_spec.name
    directory.mkdir(parents=True, exist_ok=True)
    text_only = store.put_text_only(text_only_table or write_text_only(directory / 'text'))
    runs = []
    for seed, offset in enumerate(offsets):
        run_id = f'{arm_spec.name}/seed-{seed}'
        checkpoint = directory / f'checkpoint-{seed}.ckpt'
        checkpoint.write_bytes(f'{arm_spec.name} {seed}'.encode())
        table = store.put_table(_table(directory, arm_spec.name, seed, arm_spec.dimension))
        scores = synthetic_scores(effects, offset * SIGMA)
        time = datetime.now(timezone.utc).isoformat()
        runs.append(
            SeedRun(
                seed=seed,
                run_id=run_id,
                checkpoint=store.put(checkpoint),
                table=table,
                scores=store.put_frame(scores, 'scores.parquet'),
                decoding=store.put_frame(pl.DataFrame({'query_id': [seed]}), 'decoding.parquet'),
                predictions=store.put_frame(
                    pl.DataFrame({'code': [CODES[0]]}), 'predictions.parquet'
                ),
                statistics={
                    panel: panel_statistic(scores, panel, DECISION_STATISTIC[panel])
                    for panel in PANELS
                },
                log_records=[
                    _read(
                        panel, run_id, table.matrix_fingerprint, text_only.table.matrix_fingerprint,
                        time
                    ) for panel in PANELS
                ],
            )
        )
    return ArmRecord(
        spec=arm_spec,
        text_only=text_only,
        store=str(store.root),
        panels=PANEL_SET,
        runs=runs,
        created_at=datetime.now(timezone.utc),
    )
```

Create `tests/unit/test_decision_store.py` with exactly this content:

```python
'''
The artifact store and the record files: immutable references a decision record can name.
'''

import json

import polars as pl
import pytest

from naics_embedder.decision.records import ArmRecord, read_record, write_record
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.decision import CODES, spec, synthetic_arm, write_text_only
from tests.fixtures.regressor_panel import coordinate_table

pytestmark = pytest.mark.unit

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

def test_a_stored_file_outlives_its_source_and_is_verified_on_every_read(store, tmp_path):
    source = tmp_path / 'checkpoint.ckpt'
    source.write_bytes(b'weights')

    reference = store.put(source)
    again = store.put(source)
    source.unlink()

    assert reference == again
    assert reference.sha256 == sha256_file(store.resolve(reference))
    assert reference.path == f'objects/{reference.sha256[:2]}/{reference.sha256}/checkpoint.ckpt'
    assert reference.bytes == len(b'weights')
    store.resolve(reference).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed since it was stored'):
        store.resolve(reference)

def test_a_table_is_stored_with_the_fingerprint_the_log_names_it_by(store, tmp_path):
    table = coordinate_table(CODES, dimension=4)
    path = tmp_path / 'arm.parquet'
    table.write_parquet(path)

    reference = store.put_table(path)

    assert reference.matrix_fingerprint == table_fingerprint(table)

def test_a_text_only_table_is_stored_with_its_provenance(store, tmp_path):
    path = write_text_only(tmp_path / 'text')

    reference = store.put_text_only(path)

    provenance = json.loads(provenance_path(path).read_text())
    assert reference.table.sha256 == provenance['table_sha256']
    assert reference.table.matrix_fingerprint == provenance['matrix_fingerprint']
    assert reference.provenance.sha256 == sha256_file(provenance_path(path))
    assert (reference.backbone, reference.revision, reference.max_length) == (
        provenance['backbone'], provenance['revision'], provenance['max_length']
    )

def test_a_text_only_table_needs_the_provenance_that_describes_it(store, tmp_path):
    path = write_text_only(tmp_path / 'text')
    provenance = json.loads(provenance_path(path).read_text())

    provenance_path(path).write_text(json.dumps({**provenance, 'matrix_fingerprint': 'other'}))
    with pytest.raises(ValueError, match='another matrix_fingerprint'):
        store.put_text_only(path)
    pl.read_parquet(path).head(3).write_parquet(path)
    with pytest.raises(ValueError, match='describes another file'):
        store.put_text_only(path)
    provenance_path(path).unlink()
    with pytest.raises(FileNotFoundError, match='no provenance'):
        store.put_text_only(path)

def test_a_record_is_written_once_and_reads_back_whole(store, tmp_path):
    arm = synthetic_arm(store, tmp_path, spec('A'), {})
    path = tmp_path / 'records' / 'A.json'

    write_record(arm, path)

    assert read_record(path, ArmRecord) == arm
    with pytest.raises(FileExistsError):
        write_record(arm, path)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_decision_store.py -q`
Expected: one collection error,
`ModuleNotFoundError: No module named 'naics_embedder.decision.store'`.

- [x] **Step 3: Implement**

Create `src/naics_embedder/decision/store.py` with exactly this content:

```python
'''
The content-addressed store behind decision records' artifact references (roadmap Stage 4).

A file is copied to ``objects/<sha[:2]>/<sha>/<name>`` under the store root, and its reference is
that relative path with its sha256 and size. The store never deletes or overwrites: an object is
written once and verified again on every read. Records need these files until Stage 12, so the
root belongs outside any worktree (removing a worktree deletes its ignored files), and a Lambda
instance's store must be copied off before the instance terminates
(``specs/lambda-remote-workflow.md``).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Union

import polars as pl

from naics_embedder.decision.records import ArtifactRef, TableRef, TextOnlyRef
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path, text_only_fingerprint
from naics_embedder.supervision.artifacts import sha256_file

# -------------------------------------------------------------------------------------------------
# Store
# -------------------------------------------------------------------------------------------------

class ArtifactStore:
    '''Immutable, content-addressed files under one root.'''

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser()

    def put(self, path: Union[str, Path]) -> ArtifactRef:
        '''
        Copy a file into the store (once per content and name) and return its reference.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the stored copy does not hash to the file's sha256.
        '''

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'{path} is not a file')
        digest = sha256_file(path)
        relative = Path('objects') / digest[:2] / digest / path.name
        target = self.root / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, staging = tempfile.mkstemp(dir=target.parent, prefix='.incoming-')
            os.close(handle)
            shutil.copyfile(path, staging)
            os.replace(staging, target)
        if sha256_file(target) != digest:
            raise ValueError(f'{target} does not hash to {digest}')
        return ArtifactRef(path=relative.as_posix(), sha256=digest, bytes=target.stat().st_size)

    def put_frame(self, frame: pl.DataFrame, name: str) -> ArtifactRef:
        '''Store a frame as a parquet file named ``name``.'''

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            frame.write_parquet(path)
            return self.put(path)

    def put_table(self, path: Union[str, Path]) -> TableRef:
        '''Store an arm's code table (the export form) with its ``matrix_fingerprint``.'''

        reference = self.put(path)
        fingerprint = table_fingerprint(pl.read_parquet(self.resolve(reference)))
        return TableRef(**reference.model_dump(), matrix_fingerprint=fingerprint)

    def put_text_only(self, table_path: Union[str, Path]) -> TextOnlyRef:
        '''
        Store a text-only table and the provenance beside it (``tools text-only-table``).

        Raises:
            FileNotFoundError: If the provenance is missing.
            ValueError: If the provenance describes another file, or names another
                ``matrix_fingerprint`` than the table's.
        '''

        table_path = Path(table_path)
        provenance_file = provenance_path(table_path)
        if not provenance_file.is_file():
            raise FileNotFoundError(f'{provenance_file}: the text-only table has no provenance')
        provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
        table = self.put(table_path)
        if provenance['table_sha256'] != table.sha256:
            raise ValueError(f'{provenance_file} describes another file than {table_path}')
        fingerprint = text_only_fingerprint(pl.read_parquet(self.resolve(table)))
        if provenance.get('matrix_fingerprint', fingerprint) != fingerprint:
            raise ValueError(f'{provenance_file} names another matrix_fingerprint')
        return TextOnlyRef(
            table=TableRef(**table.model_dump(), matrix_fingerprint=fingerprint),
            provenance=self.put(provenance_file),
            backbone=provenance['backbone'],
            revision=provenance['revision'],
            descriptions_sha256=provenance['descriptions']['sha256'],
            max_length=provenance['max_length'],
        )

    def resolve(self, reference: ArtifactRef) -> Path:
        '''
        The stored file, verified.

        Raises:
            FileNotFoundError: If the store has no such file.
            ValueError: If the file no longer hashes to the reference's sha256.
        '''

        path = self.root / reference.path
        if not path.is_file():
            raise FileNotFoundError(f'{path}: not in the artifact store at {self.root}')
        if sha256_file(path) != reference.sha256:
            raise ValueError(f'{path} changed since it was stored')
        return path

    def read_frame(self, reference: ArtifactRef) -> pl.DataFrame:
        '''A stored parquet file, verified.'''

        return pl.read_parquet(self.resolve(reference))
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_store.py -q`
Expected: `5 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1569 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/decision/store.py tests/fixtures/decision.py tests/unit/test_decision_store.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/decision/store.py \
  tests/fixtures/decision.py \
  tests/unit/test_decision_store.py
git commit -m "feat(decision): add the content-addressed artifact store and synthetic arms"
```

### Task 6: Margins and decisions under Req 5, with every guard

`fix_margins` sets each panel's δ from a reference arm. `decide` compares every ordered pair of
arms on paired resamples and writes the record Verification "Decision records" lists. That
record carries the non-dominated set, the tie order, the chosen arm and each arm's reports:

- every Req 3 metric;
- every comparator's MSE;
- the gain over the regime's sparse comparator, with its 95 % interval (D10, reported and not
  decided on);
- the held-out regime by feature year (plan 5's deferred note).

Every guard in **Global Constraints** runs before any resample. The tests are the Exit's
evidence: synthetic arms with known effects on the three panels are adopted and rejected as the
rule says.

**Files:**

- Create: `src/naics_embedder/decision/decide.py` (guards, `fix_margins`, `decide`)
- Create: `tests/unit/test_decision.py`

**Interfaces:**

- Consumes:
  - from `decision.scores` (Task 2): `PANELS`, `HELDOUT_PANEL`, `REGRESSOR_PANELS`,
    `DECISION_STATISTIC`, `ORIENTATION`, `EMBEDDING_COMPARATOR`, `SPARSE_COMPARATOR`,
    `STATISTIC_DEFINITIONS`, `statistic_values`, `panel_statistic` and `statistic_means`;
  - from `decision.resampling` (Task 3): `PanelItems`, `unit_draws`, `seed_draws`,
    `replicate_statistics`, `point_statistic` and `percentile_interval`;
  - from `decision.rule` (Task 4): `NONINFERIORITY_LEVEL`, `SUPERIORITY_LEVEL`, `compare_panel`,
    `compare`, `non_dominated` and `tie_order`;
  - from `decision.records` (Task 4): the record models, and in the tests `read_record` and
    `write_record`;
  - `ArtifactStore` (Task 5), and in the tests `spec`, `synthetic_arm`, `write_text_only`,
    `SIGMA`, `SD` and `SEED_OFFSETS` from `tests.fixtures.decision` (Task 5);
  - `OUTCOME_PANEL` (`panels.outcome`) and `VALIDATION` (`panels.regressor`), both existing, and
    in the tests `METRIC_NAMES` (`panels.decoding`).
- Produces, in `naics_embedder.decision.decide`:
  - `check_text_only(spec: ArmSpec, text_only: TextOnlyRef) -> None` (D9);
  - `check_arm(arm: ArmRecord, store: ArtifactStore, min_seeds: int) -> None`;
  - `check_pairing(arms: Sequence[ArmRecord], margins: MarginRecord) -> None`;
  - `check_margins_first(arms: Sequence[ArmRecord], margins: MarginRecord) -> None`;
  - `fix_margins(reference: ArmRecord, multiple: float, name: str, store: ArtifactStore, *,
    min_seeds: int) -> MarginRecord`;
  - `decide(name: str, question: str, arms: Sequence[ArmRecord], margins: MarginRecord, store:
    ArtifactStore, *, replicates: int, bootstrap_seed: int, min_seeds: int) -> DecisionRecord`.

  Each check raises `ValueError` naming what failed, except that a missing artifact raises the
  store's `FileNotFoundError`. `decide` raises `TieUnresolvedError` when the tie order cannot
  choose.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_decision.py` with exactly this content:

```python
'''
Margins and decisions on synthetic arms with known effects (roadmap Stage 4 Exit).

Effects are in units of ``SIGMA``, and every arm's across-seed standard deviation is ``SD`` =
``SIGMA`` × √2.5, so a margin multiple of 2 gives δ ≈ 3.2 ``SIGMA``. Paired Δ is the effect plus
the difference of two means of five seed offsets drawn with replacement; its 95 % and 98⅓ %
intervals reach about 1.8 and 2.1 ``SIGMA`` either side. An effect of 5 is superior, 0 is not,
and −5 is not non-inferior.
'''

import json

import pytest

from naics_embedder.decision.decide import decide, fix_margins
from naics_embedder.decision.records import ArmRecord, DecisionRecord, read_record, write_record
from naics_embedder.decision.rule import SUPERIORITY_LEVEL
from naics_embedder.decision.scores import PANELS
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.decoding import METRIC_NAMES
from tests.fixtures.decision import (
    SD,
    SEED_OFFSETS,
    SIGMA,
    spec,
    synthetic_arm,
    write_text_only,
)

pytestmark = pytest.mark.unit

REPLICATES = 4000
BOOTSTRAP_SEED = 20260924

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

@pytest.fixture
def reference(store, tmp_path):
    return synthetic_arm(store, tmp_path, spec('reference'), {})

@pytest.fixture
def margins(reference, store):
    return fix_margins(reference, 2.0, 'fixture margins', store, min_seeds=5)

def _decide(arms, margins, store):
    return decide(
        'fixture decision',
        'which arm?',
        arms,
        margins,
        store,
        replicates=REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        min_seeds=5,
    )

def _comparison(record, a, b):
    return next(item for item in record.comparisons if (item.a, item.b) == (a, b))

def _panel(comparison, panel):
    return next(item for item in comparison.panels if item.panel == panel)

def _edited(arm, edit):
    '''The arm record after ``edit`` changes its JSON form.'''

    data = json.loads(arm.model_dump_json())
    edit(data)
    return ArmRecord.model_validate(data)

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

def test_each_margin_is_the_multiple_times_the_references_across_seed_sd(margins):
    assert [entry.panel for entry in margins.margins] == list(PANELS)
    for entry in margins.margins:
        assert len(entry.per_seed) == 5
        assert entry.sd == pytest.approx(SD)
        assert entry.margin == pytest.approx(2 * SD)

def test_a_margin_needs_a_positive_multiple_and_a_reference_that_varies(store, tmp_path, reference):
    with pytest.raises(ValueError, match='positive'):
        fix_margins(reference, 0.0, 'none', store, min_seeds=5)
    flat = synthetic_arm(store, tmp_path, spec('flat'), {}, offsets=(0.0, ) * 5)
    with pytest.raises(ValueError, match='does not vary'):
        fix_margins(flat, 2.0, 'flat', store, min_seeds=5)

# -------------------------------------------------------------------------------------------------
# The rule on known effects
# -------------------------------------------------------------------------------------------------

def test_an_arm_superior_on_one_panel_and_level_on_the_others_is_adopted(
    store, tmp_path, reference, margins
):
    better = synthetic_arm(store, tmp_path, spec('better', dimension=32), {'outcome': 5})

    record = _decide([better, reference], margins, store)

    adopted = _comparison(record, 'better', 'reference')
    assert adopted.adopted
    assert _panel(adopted, 'outcome').delta == pytest.approx(5 * SIGMA)
    assert _panel(adopted, 'outcome').superior
    for panel in ('regressor_seen', 'regressor_heldout'):
        assert _panel(adopted, panel).non_inferior
        assert not _panel(adopted, panel).superior
    assert not _comparison(record, 'reference', 'better').adopted
    # Adopted over the simpler arm: the tie order never runs between them
    assert (record.non_dominated, record.cycle, record.chosen) == (['better'], False, 'better')

def test_superiority_on_one_panel_does_not_rescue_an_inferior_one(
    store, tmp_path, reference, margins
):
    mixed = synthetic_arm(
        store, tmp_path, spec('mixed', dimension=32), {
            'outcome': -5,
            'regressor_seen': 5
        }
    )

    record = _decide([mixed, reference], margins, store)

    rejected = _comparison(record, 'mixed', 'reference')
    assert _panel(rejected, 'regressor_seen').superior
    assert not _panel(rejected, 'outcome').non_inferior
    assert not rejected.adopted
    assert not _comparison(record, 'reference', 'mixed').adopted
    assert record.non_dominated == ['mixed', 'reference']
    assert record.chosen == 'reference'

@pytest.mark.parametrize('dimension, chosen', [(32, 'reference'), (8, 'level')])
def test_without_superiority_the_simpler_arm_stands(
    store, tmp_path, reference, margins, dimension, chosen
):
    level = synthetic_arm(store, tmp_path, spec('level', dimension=dimension), {})

    record = _decide([level, reference], margins, store)

    assert not any(item.adopted for item in record.comparisons)
    assert record.chosen == chosen

def test_a_dominance_cycle_leaves_every_arm_to_the_tie_order(store, tmp_path, reference):
    # δ = 5 SD ≈ 7.9 SIGMA: each arm is non-inferior where it is 5 SIGMA worse
    wide = fix_margins(reference, 5.0, 'wide margins', store, min_seeds=5)
    arms = [
        synthetic_arm(store, tmp_path, spec('outcome-arm', dimension=32), {'outcome': 5}),
        synthetic_arm(store, tmp_path, spec('seen-arm', dimension=16), {'regressor_seen': 5}),
        synthetic_arm(store, tmp_path, spec('heldout-arm', dimension=24), {'regressor_heldout': 5}),
    ]

    record = _decide(arms, wide, store)

    assert all(item.adopted for item in record.comparisons)
    assert record.cycle
    assert record.non_dominated == ['outcome-arm', 'seen-arm', 'heldout-arm']
    assert record.tie_order == ['seen-arm', 'heldout-arm', 'outcome-arm']
    assert record.chosen == 'seen-arm'

def test_the_held_out_gain_over_ancestors_breaks_the_last_tie(store, tmp_path, reference, margins):
    # Half a SIGMA lower held-out error: too small to be superior, enough to break the tie
    arms = [
        synthetic_arm(store, tmp_path, spec('spherical', geometry='spherical'), {}),
        synthetic_arm(
            store, tmp_path, spec('euclidean', geometry='euclidean'), {'regressor_heldout': 0.5}
        ),
    ]

    record = _decide(arms, margins, store)

    assert not any(item.adopted for item in record.comparisons)
    assert record.heldout_gain['euclidean'] == pytest.approx(0.03 + 0.5 * SIGMA)
    assert record.heldout_gain['spherical'] == pytest.approx(0.03)
    assert record.tie_order == ['euclidean', 'spherical']

# -------------------------------------------------------------------------------------------------
# The record (Verification "Decision records")
# -------------------------------------------------------------------------------------------------

def test_the_record_carries_every_field_verification_lists(store, tmp_path, reference, margins):
    better = synthetic_arm(store, tmp_path, spec('better'), {'outcome': 5})
    path = write_record(_decide([better, reference], margins, store), tmp_path / 'decision.json')

    record = read_record(path, DecisionRecord)

    # Its arms, at least 5 seeds each
    assert [arm.spec.name for arm in record.arms] == ['better', 'reference']
    assert all(len(arm.runs) >= 5 for arm in record.arms)
    # The δ per panel, fixed before the runs
    assert [entry.panel for entry in record.margins.margins] == list(PANELS)
    reads = [log['time'] for run in record.arms[0].runs for log in run.log_records]
    assert min(reads) >= record.margins.fixed_at.isoformat()
    # The 95 % non-inferiority and 98⅓ % superiority intervals, on every panel
    assert record.settings.noninferiority_level == 0.95
    assert record.settings.superiority_level == SUPERIORITY_LEVEL
    for comparison in record.comparisons:
        assert [panel.panel for panel in comparison.panels] == list(PANELS)
        for panel in comparison.panels:
            low, high = panel.noninferiority_interval
            wide_low, wide_high = panel.superiority_interval
            assert wide_low <= low <= high <= wide_high
    # The non-dominated set
    assert record.non_dominated == ['better']
    # The selection-log records of the runs and every artifact reference
    for arm in record.arms:
        store.resolve(arm.text_only.table)
        store.resolve(arm.text_only.provenance)
        for run in arm.runs:
            assert sorted(log['panel'] for log in run.log_records) == sorted(PANELS)
            for reference_ in (
                run.checkpoint, run.table, run.scores, run.decoding, run.predictions
            ):
                store.resolve(reference_)
    # What D10 reports besides the rule
    report = record.reports[0]
    assert set(report.outcome_metrics) == set(METRIC_NAMES)
    assert set(report.comparator_mse['regressor_seen']) >= {
        'covariates', 'covariates+embedding', 'covariates+one_hot'
    }
    assert report.gain['regressor_seen'].point == pytest.approx(0.02)
    assert report.gain['regressor_heldout'].point == pytest.approx(0.03)
    assert set(report.heldout_by_feature_year) == {'2022', '2023'}
    assert report.heldout_by_feature_year['2023']['gain'] == pytest.approx(0.03)

# -------------------------------------------------------------------------------------------------
# Guards
# -------------------------------------------------------------------------------------------------

def test_every_arm_needs_five_seeds(store, tmp_path, reference, margins):
    short = synthetic_arm(store, tmp_path, spec('short'), {}, offsets=SEED_OFFSETS[:4])

    with pytest.raises(ValueError, match='fewer than 5'):
        _decide([short, reference], margins, store)

def test_a_run_that_read_before_the_margins_were_fixed_is_refused(store, tmp_path, reference):
    early = synthetic_arm(store, tmp_path, spec('early'), {})
    margins = fix_margins(reference, 2.0, 'late margins', store, min_seeds=5)

    with pytest.raises(ValueError, match='before the margins were fixed'):
        _decide([early, reference], margins, store)

def _first_read(data):
    return data['runs'][0]['log_records'][0]

@pytest.mark.parametrize(
    'edit, message',
    [
        (lambda data: _first_read(data).update(split='test'), 'validation splits only'),
        (lambda data: _first_read(data).update(event='open'), 'validation splits only'),
        (lambda data: _first_read(data)['detail'].update(run='other'), 'another run'),
        (lambda data: _first_read(data)['detail'].update(table='other'), "another \\['table'\\]"),
        (lambda data: _first_read(data).update(fingerprint='other'), 'another'),
        (lambda data: data['runs'][0]['log_records'].pop(), 'not each of'),
    ],
)
def test_a_runs_log_records_must_be_its_own_validation_reads(
    store, tmp_path, reference, margins, edit, message
):
    arm = _edited(synthetic_arm(store, tmp_path, spec('edited'), {}), edit)

    with pytest.raises(ValueError, match=message):
        _decide([arm, reference], margins, store)

def test_a_regressor_read_must_name_the_runs_tables(store, tmp_path, reference, margins):
    arm = _edited(
        synthetic_arm(store, tmp_path, spec('edited'), {}),
        lambda data: data['runs'][0]['log_records'][1]['detail'].update(text_only='other'),
    )

    with pytest.raises(ValueError, match="another \\['text_only'\\]"):
        _decide([arm, reference], margins, store)

def test_paired_arms_must_read_the_same_panels(store, tmp_path, reference, margins):
    arm = _edited(
        synthetic_arm(store, tmp_path, spec('other-folds'), {}),
        lambda data: data['panels']['fit_settings'].update(folds=3),
    )

    with pytest.raises(ValueError, match='other panels or fit settings'):
        _decide([arm, reference], margins, store)

def test_the_text_only_table_must_come_from_the_arms_backbone_and_text(
    store, tmp_path, reference, margins
):
    stale = write_text_only(tmp_path / 'stale', revision='an-older-revision')
    arm = synthetic_arm(store, tmp_path, spec('stale'), {}, text_only_table=stale)

    with pytest.raises(ValueError, match='D9'):
        _decide([arm, reference], margins, store)

def test_a_changed_artifact_is_refused(store, tmp_path, reference, margins):
    arm = synthetic_arm(store, tmp_path, spec('changed'), {})
    store.resolve(arm.runs[2].scores).write_bytes(b'not the scores')

    with pytest.raises(ValueError, match='changed since it was stored'):
        _decide([arm, reference], margins, store)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_decision.py -q`
Expected: one collection error,
`ModuleNotFoundError: No module named 'naics_embedder.decision.decide'`.

- [x] **Step 3: Implement**

> Deviation: F3 (user ruling at pre-flight): `decide.py` gained `MIN_SEEDS = 5`, and `check_arm` refuses `min_seeds < MIN_SEEDS`, with one `pytest.raises` inside `test_a_margin_needs_a_positive_multiple_and_a_reference_that_varies` (8c81ab4; counts unchanged).

Create `src/naics_embedder/decision/decide.py` with exactly this content:

```python
'''
Margins from a reference arm, and a decision over arms (Req 5; D8, D10, D11).

Before any number is computed, every arm is checked:

- it has at least ``min_seeds`` seeds, each read once on each of the three panels;
- every stored artifact still hashes to its reference;
- its text-only table's provenance matches the arm's backbone, revision, descriptions and
  window (D9);
- each run's log records are validation reads that name the run, its table and its text-only
  table by the fingerprints the store recorded;
- all arms read the same panels with the same fit settings, so Δ pairs item for item.

A decision also requires every run other than the margin record's own reference runs to have
read nothing before the margins were fixed.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from naics_embedder.decision.records import (
    ArmRecord,
    ArmReport,
    ArmSpec,
    Comparison,
    DecisionRecord,
    DecisionSettings,
    Estimate,
    MarginRecord,
    PanelMargin,
    SeedRun,
    TextOnlyRef,
)
from naics_embedder.decision.resampling import (
    PanelItems,
    percentile_interval,
    point_statistic,
    replicate_statistics,
    seed_draws,
    unit_draws,
)
from naics_embedder.decision.rule import (
    NONINFERIORITY_LEVEL,
    SUPERIORITY_LEVEL,
    compare,
    compare_panel,
    non_dominated,
    tie_order,
)
from naics_embedder.decision.scores import (
    DECISION_STATISTIC,
    EMBEDDING_COMPARATOR,
    HELDOUT_PANEL,
    ORIENTATION,
    PANELS,
    REGRESSOR_PANELS,
    SPARSE_COMPARATOR,
    STATISTIC_DEFINITIONS,
    panel_statistic,
    statistic_means,
    statistic_values,
)
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OUTCOME_PANEL
from naics_embedder.panels.regressor import VALIDATION

# -------------------------------------------------------------------------------------------------
# Guards
# -------------------------------------------------------------------------------------------------

def check_text_only(spec: ArmSpec, text_only: TextOnlyRef) -> None:
    '''
    Require the text-only table to come from the arm's backbone reading the arm's text (D9).

    Raises:
        ValueError: If the provenance's backbone, revision, descriptions sha256 or window
            differs from the arm's.
    '''

    built = (
        text_only.backbone, text_only.revision, text_only.descriptions_sha256, text_only.max_length
    )
    reads = (spec.backbone, spec.backbone_revision, spec.descriptions_sha256, spec.max_length)
    if built != reads:
        raise ValueError(
            f'{spec.name}: the text-only table was built from {built}, the arm reads {reads} '
            "(D9: the arm's own backbone reading the arm's text)"
        )

def check_arm(arm: ArmRecord, store: ArtifactStore, min_seeds: int) -> None:
    '''
    Require an arm record to be complete, intact and read as the decision reads it.

    Raises:
        ValueError: On the first problem found.
    '''

    name = arm.spec.name
    seeds = [run.seed for run in arm.runs]
    if len(set(seeds)) < min_seeds:
        raise ValueError(f'{name}: {len(set(seeds))} distinct seeds, fewer than {min_seeds}')
    if len(set(seeds)) != len(seeds) or len({run.run_id for run in arm.runs}) != len(arm.runs):
        raise ValueError(f'{name}: a seed or run id repeats')
    check_text_only(arm.spec, arm.text_only)
    store.resolve(arm.text_only.table)
    store.resolve(arm.text_only.provenance)
    for run in arm.runs:
        for reference in (run.checkpoint, run.table, run.scores, run.decoding, run.predictions):
            store.resolve(reference)
        _check_log_records(arm, run)

def _check_log_records(arm: ArmRecord, run: SeedRun) -> None:
    name = f'{arm.spec.name} seed {run.seed}'
    panels = sorted(record['panel'] for record in run.log_records)
    if panels != sorted(PANELS):
        raise ValueError(f'{name}: the run read {panels}, not each of {sorted(PANELS)} once')
    for record in run.log_records:
        detail = record['detail']
        if record['event'] != 'read' or record['split'] != VALIDATION:
            raise ValueError(
                f'{name}: a {record["event"]} of the {record["split"]} split; a decision reads '
                'validation splits only (Req 4)'
            )
        if detail.get('run') != run.run_id:
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
        if wrong:
            raise ValueError(f'{name}: a {record["panel"]} read names another {wrong}')

def check_pairing(arms: Sequence[ArmRecord], margins: MarginRecord) -> None:
    '''
    Require every arm, and the margins' reference, to have read the same panels under the same
    fit settings.

    Raises:
        ValueError: If two arms share a name, or one read other panels.
    '''

    names = [arm.spec.name for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError(f'arm names repeat: {names}')
    others = [(arm.spec.name, arm.panels) for arm in arms[1:]]
    others.append((f'the margin reference {margins.reference.spec.name}', margins.reference.panels))
    for name, panels in others:
        if panels != arms[0].panels:
            raise ValueError(
                f'{name} read other panels or fit settings than {arms[0].spec.name}: paired '
                'arms must be scored on the same resample units (Req 5)'
            )

def check_margins_first(arms: Sequence[ArmRecord], margins: MarginRecord) -> None:
    '''
    Require every run but the margin record's reference runs to read after the margins were fixed.

    Raises:
        ValueError: If a run read before ``margins.fixed_at``.
    '''

    reference = {run.run_id for run in margins.reference.runs}
    for arm in arms:
        for run in arm.runs:
            if run.run_id in reference:
                continue
            first = min(datetime.fromisoformat(record['time']) for record in run.log_records)
            if first < margins.fixed_at:
                raise ValueError(
                    f'{arm.spec.name} seed {run.seed} read at {first.isoformat()}, before the '
                    f'margins were fixed at {margins.fixed_at.isoformat()} (Req 5)'
                )

# -------------------------------------------------------------------------------------------------
# Margins
# -------------------------------------------------------------------------------------------------

def fix_margins(
    reference: ArmRecord,
    multiple: float,
    name: str,
    store: ArtifactStore,
    *,
    min_seeds: int,
) -> MarginRecord:
    '''
    Each panel's δ: ``multiple`` times the reference arm's across-seed standard deviation of the
    panel's decision statistic (Req 5).

    Raises:
        ValueError: If the multiple is not positive, the reference fails ``check_arm``, or a
            panel's statistic does not vary across seeds.
    '''

    if multiple <= 0:
        raise ValueError(f'the margin multiple must be positive, got {multiple}')
    check_arm(reference, store, min_seeds)
    scores = [store.read_frame(run.scores) for run in reference.runs]
    margins: List[PanelMargin] = []
    for panel in PANELS:
        statistic = DECISION_STATISTIC[panel]
        per_seed = [panel_statistic(frame, panel, statistic) for frame in scores]
        sd = float(np.std(per_seed, ddof=1))
        if sd == 0.0:
            raise ValueError(f"{panel}: the reference's {statistic} does not vary across seeds")
        margins.append(
            PanelMargin(
                panel=panel, statistic=statistic, per_seed=per_seed, sd=sd, margin=multiple * sd
            )
        )
    return MarginRecord(
        name=name,
        multiple=multiple,
        reference=reference,
        margins=margins,
        fixed_at=datetime.now(timezone.utc),
    )

# -------------------------------------------------------------------------------------------------
# Resampling every arm
# -------------------------------------------------------------------------------------------------

def _values(
    store: ArtifactStore,
    arms: Sequence[ArmRecord],
    panel: str,
    statistic: str,
    items: Optional[PanelItems] = None,
) -> Tuple[PanelItems, Dict[str, np.ndarray]]:
    '''
    Each arm's (seeds, items) values of one statistic, on items every arm shares: ``items``, or
    else the first seed's.
    '''

    frames = {
        arm.spec.name: [
            statistic_values(store.read_frame(run.scores), panel, statistic) for run in arm.runs
        ]
        for arm in arms
    }
    items = items or PanelItems.from_values(next(iter(frames.values()))[0])
    return items, {
        name: np.stack([items.values(frame) for frame in seeds])
        for name, seeds in frames.items()
    }

def _replicates(
    items: PanelItems,
    values: np.ndarray,
    panel: str,
    arm: str,
    units: np.ndarray,
    settings: DecisionSettings,
) -> Tuple[float, np.ndarray]:
    sums = items.sums(values)
    seeds = seed_draws(panel, arm, values.shape[0], settings.replicates, settings.bootstrap_seed)
    return point_statistic(sums, items.sizes), replicate_statistics(sums, items.sizes, units, seeds)

# -------------------------------------------------------------------------------------------------
# Decision
# -------------------------------------------------------------------------------------------------

def decide(
    name: str,
    question: str,
    arms: Sequence[ArmRecord],
    margins: MarginRecord,
    store: ArtifactStore,
    *,
    replicates: int,
    bootstrap_seed: int,
    min_seeds: int,
) -> DecisionRecord:
    '''
    Req 5's decision over two or more arms.

    Raises:
        ValueError: If fewer than two arms are given, or a guard fails (module docstring).
        TieUnresolvedError: If the tie order cannot separate the surviving arms.
    '''

    if len(arms) < 2:
        raise ValueError('a decision compares at least two arms')
    for arm in arms:
        check_arm(arm, store, min_seeds)
    check_pairing(arms, margins)
    check_margins_first(arms, margins)
    settings = DecisionSettings(
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        min_seeds=min_seeds,
        noninferiority_level=NONINFERIORITY_LEVEL,
        superiority_level=SUPERIORITY_LEVEL,
    )

    names = [arm.spec.name for arm in arms]
    points: Dict[str, Dict[str, float]] = {panel: {} for panel in PANELS}
    draws: Dict[str, Dict[str, np.ndarray]] = {panel: {} for panel in PANELS}
    gains: Dict[str, Dict[str, Estimate]] = {arm: {} for arm in names}
    for panel in PANELS:
        items, values = _values(store, arms, panel, DECISION_STATISTIC[panel])
        units = unit_draws(panel, len(items.units), replicates, bootstrap_seed)
        for arm in names:
            points[panel][arm], draws[panel][arm] = _replicates(
                items, values[arm], panel, arm, units, settings
            )
        if panel in REGRESSOR_PANELS:
            _, sparse = _values(store, arms, panel, SPARSE_COMPARATOR[panel], items)
            for arm in names:
                point, reps = _replicates(
                    items, sparse[arm] - values[arm], panel, arm, units, settings
                )
                gains[arm][panel] = Estimate(
                    point=point, interval=percentile_interval(reps, NONINFERIORITY_LEVEL)
                )

    comparisons: List[Comparison] = []
    for a in names:
        for b in names:
            if a == b:
                continue
            panels = [
                compare_panel(
                    panel,
                    ORIENTATION[panel] * (points[panel][a] - points[panel][b]),
                    ORIENTATION[panel] * (draws[panel][a] - draws[panel][b]),
                    margins.margin(panel),
                ) for panel in PANELS
            ]
            comparisons.append(compare(a, b, panels))

    survivors, cycle = non_dominated(names, comparisons)
    heldout_gain = {arm: gains[arm][HELDOUT_PANEL].point for arm in names}
    order = tie_order([arm.spec for arm in arms if arm.spec.name in survivors], heldout_gain)
    return DecisionRecord(
        name=name,
        question=question,
        created_at=datetime.now(timezone.utc),
        statistics=dict(STATISTIC_DEFINITIONS),
        settings=settings,
        margins=margins,
        arms=list(arms),
        comparisons=comparisons,
        non_dominated=survivors,
        cycle=cycle,
        tie_order=order,
        heldout_gain=heldout_gain,
        chosen=order[0],
        reports=[
            _report(
                arm, store, {panel: points[panel][arm.spec.name]
                             for panel in PANELS}, gains[arm.spec.name]
            ) for arm in arms
        ],
    )

def _report(
    arm: ArmRecord,
    store: ArtifactStore,
    statistics: Mapping[str, float],
    gain: Mapping[str, Estimate],
) -> ArmReport:
    '''Every other statistic D10 reports, each a mean over the arm's seeds.'''

    scores = [store.read_frame(run.scores) for run in arm.runs]
    means = [statistic_means(frame) for frame in scores]
    averaged = {
        panel: {
            statistic: float(np.mean([seed[panel][statistic] for seed in means]))
            for statistic in means[0][panel]
        }
        for panel in means[0]
    }
    by_year: Dict[str, Dict[str, float]] = {}
    heldout = pl.concat([frame.filter(pl.col('panel') == HELDOUT_PANEL)
                         for frame in scores]).group_by('feature_year',
                                                        'statistic').agg(pl.col('value').mean())
    # Each seed scores the same rows, so the pooled mean is the mean over seeds
    for year, statistic, value in heldout.sort('feature_year', 'statistic').iter_rows():
        if statistic in (EMBEDDING_COMPARATOR, SPARSE_COMPARATOR[HELDOUT_PANEL]):
            by_year.setdefault(str(year), {})[statistic] = float(value)
    for values in by_year.values():
        values['gain'] = values[SPARSE_COMPARATOR[HELDOUT_PANEL]] - values[EMBEDDING_COMPARATOR]
    return ArmReport(
        arm=arm.spec.name,
        statistics=dict(statistics),
        outcome_metrics=averaged[OUTCOME_PANEL],
        comparator_mse={panel: averaged[panel]
                        for panel in REGRESSOR_PANELS},
        gain=dict(gain),
        heldout_by_feature_year=by_year,
    )
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision.py -q`
Expected: `21 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1590 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/decision/decide.py tests/unit/test_decision.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/decision/decide.py \
  tests/unit/test_decision.py
git commit -m "feat(decision): add margins and decisions under Req 5, with every guard"
```

### Task 7: The seed-sweep driver

`run_seed_sweep` runs a configuration for N seeds through the real panel interfaces. Before any
read, it checks the text-only table against the arm (D9). Then, for each seed, it:

1. calls the runner;
2. scores the outcome panel's validation split with the seed's encoder, and each regressor
   regime's validation split at level 6, each read logging `run`, `arm_name` and `seed` (plus
   `table` on the outcome read);
3. keeps each read's log records;
4. stores the checkpoint, the table, the scores, the decoding and the predictions.

It returns the arm record that `fix_margins` and `decide` read. The tests run it on the fixture
panels of plans 4 and 5, then decide between two swept arms.

**Files:**

- Create: `src/naics_embedder/decision/sweep.py` (`run_seed_sweep`)
- Create: `tests/unit/test_decision_sweep.py`

**Interfaces:**

- Consumes:
  - `OutcomePanel.score(..., detail=...)` and `RegressorPanel.validation(..., detail=...)`
    (Task 1);
  - `QueryCodeEncoder` (`panels.outcome`);
  - `ArmTables`, `Regime` and `DECISION_LEVEL` (`panels.regressor`);
  - `SelectionLog` (`panels.selection_log`), to read each run's records back;
  - `seed_scores`, `panel_statistic`, `DECISION_STATISTIC` and `PANELS` (Task 2);
  - `ArmSpec`, `ArmRecord`, `PanelSet` and `SeedRun` (Task 4);
  - `ArtifactStore` (Task 5);
  - `check_text_only` (Task 6);
  - `IndexRole` (existing, `naics_embedder.supervision.schema`);
  - in the tests: `fix_margins` and `decide` (Task 6); `DecisionRecord`, `read_record` and
    `write_record` (Task 4); `spec` and `write_text_only` (Task 5's fixture module); and
    `CODEBOOK`, `HELDOUT_GROUPS`, `SETTINGS` and `SIX_DIGIT` (existing,
    `tests.fixtures.regressor_panel`).
- Produces, in `naics_embedder.decision.sweep`:
  - `SeedArtifacts`, a frozen dataclass with `checkpoint: Path`, `table: Path` (the 2,125-code
    table in the export form: `code` plus coordinate columns), `encoder: QueryCodeEncoder` and
    `distance: str`;
  - `ArmRunner(Protocol)`, with `run(spec: ArmSpec, seed: int) -> SeedArtifacts`;
  - `run_seed_sweep(spec: ArmSpec, seeds: Sequence[int], runner: ArmRunner, *, outcome_panel:
    OutcomePanel, regressor_panel: RegressorPanel, text_only_table, store: ArtifactStore,
    purpose: str) -> ArmRecord`. Run ids are `<arm>/seed-<seed>/<uuid4 hex>`.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_decision_sweep.py` with exactly this content:

```python
'''
The seed-sweep driver on the fixture panels (roadmap Stage 4 Exit): every read logged under its
run, every artifact stored, and a decision over two swept arms.

A synthetic runner stands in for training until Stage 6. The informed arm's encoder puts each
query near its code's axis and its table carries each code's employment level and trend; the
uninformed arm's encoder puts each query near a random code's axis and its table is noise.
'''

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.decision.decide import decide, fix_margins
from naics_embedder.decision.records import DecisionRecord, read_record, write_record
from naics_embedder.decision.scores import PANELS
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.sweep import SeedArtifacts, run_seed_sweep
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from tests.fixtures.decision import spec, write_text_only
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, SETTINGS, SIX_DIGIT

pytestmark = pytest.mark.unit

SEEDS = (0, 1, 2, 3, 4)
PURPOSE = 'fixture seed sweep'

class SyntheticEncoder:
    '''Codes on their own axes; each query near its code's axis, or a random code's.'''

    def __init__(self, informed: bool, seed: int):
        self.axis = {code: index for index, code in enumerate(SIX_DIGIT)}
        self.informed = informed
        self.rng = np.random.default_rng([seed, int(informed)])

    def _axes(self, codes):
        indices = torch.tensor([self.axis[code] for code in codes])
        return torch.nn.functional.one_hot(indices, len(self.axis)).to(torch.float64)

    def encode_codes(self, codes):
        return self._axes(codes)

    def encode_queries(self, texts):
        codes = [text.split()[0] for text in texts]
        if not self.informed:
            codes = list(self.rng.choice(SIX_DIGIT, size=len(codes)))
        noise = self.rng.normal(0.0, 0.3, size=(len(codes), len(self.axis)))
        return self._axes(codes) + torch.from_numpy(noise)

class SyntheticRunner:
    '''One seed of the informed or the uninformed arm, written under ``directory``.'''

    def __init__(self, directory, informed, signal):
        self.directory = directory
        self.informed = informed
        self.signal = signal

    def run(self, arm_spec, seed):
        rng = np.random.default_rng([seed, int(self.informed), 7])
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self.directory / f'{arm_spec.name}-{seed}.ckpt'
        checkpoint.write_bytes(f'{arm_spec.name} {seed}'.encode())
        values = rng.normal(size=(len(CODEBOOK), 3))
        if self.informed:
            values[:, :2] = self.signal + rng.normal(0.0, 0.01, size=(len(CODEBOOK), 2))
        schema = {f'e{index}': pl.Float64 for index in range(3)}
        table = pl.DataFrame({
            'code': list(CODEBOOK)
        }).hstack(pl.DataFrame(values, schema=schema, orient='row'))
        path = self.directory / f'{arm_spec.name}-{seed}.parquet'
        table.write_parquet(path)
        return SeedArtifacts(
            checkpoint=checkpoint,
            table=path,
            encoder=SyntheticEncoder(self.informed, seed),
            distance='cosine',
        )

def _signal(regressor_rows):
    '''Each codebook code's mean outcome and its trend over the feature years.'''

    rows = pl.concat([frame for frame in regressor_rows.values()])
    # yapf: disable
    by_code = (
        rows
        .sort('code', 'feature_year')
        .group_by('code', maintain_order=True)
        .agg(
            level=pl.col('outcome').mean(),
            trend=(pl.col('outcome').last() - pl.col('outcome').first()) / 2,
        )
    )
    # yapf: enable
    table = pl.DataFrame({'code': list(CODEBOOK)}).join(by_code, on='code', how='left')
    return table.select('level', 'trend').fill_null(0.0).to_numpy()

def _role_rows():
    rows, entry = [], 0
    for code in SIX_DIGIT:
        for role in ('examples', 'training', 'validation', 'validation', 'test'):
            rows.append((entry, code, f'{code} entry {entry}', role))
            entry += 1
    return pl.DataFrame(
        rows,
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    )

@pytest.fixture
def log(tmp_path):
    return SelectionLog(tmp_path / 'logs' / 'selection_log.jsonl')

@pytest.fixture
def panels(regressor_rows, log):
    return {
        'outcome_panel': OutcomePanel(_role_rows(), SIX_DIGIT, log),
        'regressor_panel': RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS),
    }

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

@pytest.fixture
def text_only(tmp_path):
    return write_text_only(tmp_path / 'text', codes=CODEBOOK)

def _sweep(name, informed, tmp_path, regressor_rows, panels, store, text_only, **overrides):
    runner = SyntheticRunner(tmp_path / 'runs' / name, informed, _signal(regressor_rows))
    return run_seed_sweep(
        spec(name, dimension=3, **overrides),
        SEEDS,
        runner,
        text_only_table=text_only,
        store=store,
        purpose=PURPOSE,
        **panels,
    )

def test_every_seed_is_read_once_per_panel_and_its_records_are_the_logs(
    tmp_path, regressor_rows, panels, store, text_only, log
):
    arm = _sweep('uninformed', False, tmp_path, regressor_rows, panels, store, text_only)

    assert [run.seed for run in arm.runs] == list(SEEDS)
    records = log.records()
    assert len(records) == 3 * len(SEEDS)
    for run in arm.runs:
        assert run.log_records == [r for r in records if r['detail']['run'] == run.run_id]
        assert sorted(r['panel'] for r in run.log_records) == sorted(PANELS)
        outcome = next(r for r in run.log_records if r['panel'] == 'outcome')
        assert outcome['detail']['table'] == run.table.matrix_fingerprint
        for record in run.log_records:
            assert (record['event'], record['split']) == ('read', 'validation')
            assert record['detail']['arm_name'] == 'uninformed'
            if record['panel'] != 'outcome':
                assert record['detail']['arm'] == run.table.matrix_fingerprint
                assert record['detail']['text_only'] == arm.text_only.table.matrix_fingerprint
        assert set(run.statistics) == set(PANELS)
    assert arm.panels.outcome == panels['outcome_panel'].fingerprint
    assert arm.panels.regressor == panels['regressor_panel'].fingerprint

def test_the_artifacts_outlive_the_runners_files(
    tmp_path, regressor_rows, panels, store, text_only
):
    arm = _sweep('uninformed', False, tmp_path, regressor_rows, panels, store, text_only)
    for path in (tmp_path / 'runs' / 'uninformed').iterdir():
        path.unlink()

    for run in arm.runs:
        assert store.resolve(run.checkpoint).read_bytes().startswith(b'uninformed')
        assert pl.read_parquet(store.resolve(run.table)).height == len(CODEBOOK)
        predictions = store.read_frame(run.predictions)
        assert set(predictions.get_column('panel').unique().to_list()) == {
            'regressor_seen', 'regressor_heldout'
        }
        assert store.read_frame(run.decoding).height == 2 * len(SIX_DIGIT)

def test_a_text_only_table_from_another_backbone_is_refused_before_any_read(
    tmp_path, regressor_rows, panels, store, text_only, log
):
    with pytest.raises(ValueError, match='D9'):
        _sweep(
            'uninformed',
            False,
            tmp_path,
            regressor_rows,
            panels,
            store,
            text_only,
            backbone='another/backbone',
        )
    assert log.records() == []

def test_a_decision_over_swept_arms_adopts_the_informed_one(
    tmp_path, regressor_rows, panels, store, text_only
):
    reference = _sweep('uninformed', False, tmp_path, regressor_rows, panels, store, text_only)
    margins = fix_margins(reference, 1.0, 'fixture reference', store, min_seeds=5)
    informed = _sweep('informed', True, tmp_path, regressor_rows, panels, store, text_only)

    record = decide(
        'fixture decision',
        'does the informed arm beat the uninformed one?',
        [informed, reference],
        margins,
        store,
        replicates=2000,
        bootstrap_seed=20260924,
        min_seeds=5,
    )
    path = write_record(record, tmp_path / 'decision.json')

    assert read_record(path, DecisionRecord) == record
    adopted = next(item for item in record.comparisons if item.a == 'informed')
    assert adopted.adopted
    assert all(panel.non_inferior for panel in adopted.panels)
    assert record.chosen == 'informed'
    report = next(item for item in record.reports if item.arm == 'informed')
    assert report.statistics['outcome'] > 0.5
    assert report.gain['regressor_heldout'].point > 0
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_decision_sweep.py -q`
Expected: one collection error,
`ModuleNotFoundError: No module named 'naics_embedder.decision.sweep'`.

- [x] **Step 3: Implement**

> Deviation: the task review found that nothing compared a table's width with the spec's dimension, which the tie order ranks by; by user ruling, `run_seed_sweep` refuses `arm.dimension != spec.dimension` after `ArmTables.from_tables`, with one `pytest.raises` inside `test_a_text_only_table_from_another_backbone_is_refused_before_any_read` (bb970d5; counts unchanged).

Create `src/naics_embedder/decision/sweep.py` with exactly this content:

```python
'''
The seed-sweep driver (roadmap Stage 4): run a configuration for N seeds, read each seed once on
each of D8's three panels, and keep everything a decision record references.

A runner trains (or loads) one seed of a configuration and returns its encoder checkpoint, its
2,125-code table in the export form and a ``QueryCodeEncoder``. Until Stage 6 adds the export and
the query path, only synthetic runners exist, in tests.

Every read goes through the panels, so it is logged, and it carries the run's id, which is how
the arm record picks its runs' records out of the selection log. The checkpoint, the table, the
text-only table with its provenance, the scores, the decoding and the predictions go to the
artifact store.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Union

import polars as pl

from naics_embedder.decision.decide import check_text_only
from naics_embedder.decision.records import ArmRecord, ArmSpec, PanelSet, SeedRun
from naics_embedder.decision.scores import DECISION_STATISTIC, PANELS, panel_statistic, seed_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.outcome import OutcomePanel, QueryCodeEncoder
from naics_embedder.panels.regressor import DECISION_LEVEL, ArmTables, Regime, RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.schema import IndexRole

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Runner interface
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SeedArtifacts:
    '''
    What one seed of a configuration produces.

    Attributes:
        checkpoint: The encoder checkpoint file.
        table: The 2,125-code table in the export form (tangent coordinates if hyperbolic).
        encoder: Queries and codes embedded in one space, for the outcome panel.
        distance: The arm's decoding distance (``panels.decoding.DISTANCES``).
    '''

    checkpoint: Path
    table: Path
    encoder: QueryCodeEncoder
    distance: str

class ArmRunner(Protocol):
    '''Trains or loads one seed of a configuration.'''

    def run(self, spec: ArmSpec, seed: int) -> SeedArtifacts:
        ...

# -------------------------------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------------------------------

def _fit_settings(panel: RegressorPanel) -> Dict[str, Any]:
    settings = asdict(panel.settings)
    return {**settings, 'alphas': list(settings['alphas'])}

def _log_records(logs: Sequence[SelectionLog], run_id: str) -> List[Dict[str, Any]]:
    '''The run's records, from every distinct log file the panels write to.'''

    paths = dict.fromkeys(log.path.resolve() for log in logs)
    return [
        record for path in paths for record in SelectionLog(path).records()
        if record['detail'].get('run') == run_id
    ]

def run_seed_sweep(
    spec: ArmSpec,
    seeds: Sequence[int],
    runner: ArmRunner,
    *,
    outcome_panel: OutcomePanel,
    regressor_panel: RegressorPanel,
    text_only_table: Union[str, Path],
    store: ArtifactStore,
    purpose: str,
) -> ArmRecord:
    '''
    Run a configuration for each seed and read each seed once on each panel's validation split.

    Raises:
        ValueError: If a seed repeats, or the text-only table was not built from the arm's
            backbone, revision, descriptions and window (D9).
    '''

    if len(set(seeds)) != len(seeds):
        raise ValueError(f'a seed repeats: {list(seeds)}')
    text_only = store.put_text_only(text_only_table)
    check_text_only(spec, text_only)
    text_frame = pl.read_parquet(store.resolve(text_only.table))
    logs = [outcome_panel.log, regressor_panel.log]

    runs: List[SeedRun] = []
    for seed in seeds:
        artifacts = runner.run(spec, seed)
        run_id = f'{spec.name}/seed-{seed}/{uuid.uuid4().hex}'
        checkpoint = store.put(artifacts.checkpoint)
        table = store.put_table(artifacts.table)
        detail = {'run': run_id, 'arm_name': spec.name, 'seed': seed}
        arm = ArmTables.from_tables(pl.read_parquet(store.resolve(table)), text_frame)
        regressor_panel.require_arm(arm)
        decoding = outcome_panel.score(
            artifacts.encoder,
            IndexRole.VALIDATION,
            purpose,
            distance=artifacts.distance,
            detail={
                **detail, 'table': table.matrix_fingerprint
            },
        )
        predictions = pl.concat(
            [
                regressor_panel.validation(regime, DECISION_LEVEL, arm, purpose, detail=detail)
                for regime in Regime
            ]
        )
        scores = seed_scores(decoding.per_query, predictions, regressor_panel.settings.repeats)
        runs.append(
            SeedRun(
                seed=seed,
                run_id=run_id,
                checkpoint=checkpoint,
                table=table,
                scores=store.put_frame(scores, 'scores.parquet'),
                decoding=store.put_frame(decoding.per_query, 'decoding.parquet'),
                predictions=store.put_frame(predictions, 'predictions.parquet'),
                statistics={
                    panel: panel_statistic(scores, panel, DECISION_STATISTIC[panel])
                    for panel in PANELS
                },
                log_records=_log_records(logs, run_id),
            )
        )
        logger.info(f'{spec.name} seed {seed}: {runs[-1].statistics}')

    return ArmRecord(
        spec=spec,
        text_only=text_only,
        store=str(store.root),
        panels=PanelSet(
            outcome=outcome_panel.fingerprint,
            regressor=regressor_panel.fingerprint,
            fit_settings=_fit_settings(regressor_panel),
        ),
        runs=runs,
        created_at=datetime.now(timezone.utc),
    )
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision_sweep.py -q`
Expected: `4 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1594 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/decision/sweep.py tests/unit/test_decision_sweep.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/decision/sweep.py \
  tests/unit/test_decision_sweep.py
git commit -m "feat(decision): add the seed-sweep driver"
```

### Task 8: `tools margins` and `tools decide`

This task exposes the decision procedure on the command line:

- `DecisionConfig` holds the replicates, the bootstrap seed and a seed floor that cannot go
  below Req 5's 5.
- `tools margins` writes a margin record from a reference arm record.
- `tools decide` writes a decision record from two or more arm records and a margin record. It
  reports a tie it cannot break and writes nothing.

The docs gain the `decision` API page and both commands.

**Files:**

- Modify: `src/naics_embedder/cli/commands/tools.py` (`tools margins`, `tools decide`)
- Modify: `src/naics_embedder/utils/config.py` (`DecisionConfig`)
- Create: `conf/data/decision.yaml` (replicates, bootstrap seed, seed floor)
- Modify: `tests/unit/test_cli_commands.py`
- Modify: `tests/unit/test_config.py`
- Modify: `docs/.nav.yml` (the API page)
- Create: `docs/api/decision.md` (API page)
- Modify: `docs/usage.md` (the two commands)
- Modify: `CLAUDE.md` (the package, the config file, the two commands)
- Modify: `WARP.md` (the package)

**Interfaces:**

- Consumes: `fix_margins` and `decide` (Task 6); `read_record`, `write_record`, `ArmRecord`,
  `MarginRecord` and, in the tests, `DecisionRecord` (Task 4); `TieUnresolvedError` (Task 4);
  `ArtifactStore` (Task 5); `spec` and `synthetic_arm` from the fixture module
  `tests.fixtures.decision` (Task 5).
- Produces:
  - `DecisionConfig(replicates: int = 10000, ≥ 1000; bootstrap_seed: int = 20260924;
    min_seeds: int = 5, ≥ 5)` in `naics_embedder.utils.config`, with `extra='forbid'`, read from
    `conf/data/decision.yaml`;
  - `naics-embedder tools margins --reference ARM.json --multiple M --name NAME --store ROOT
    --output MARGINS.json`;
  - `naics-embedder tools decide --arm A.json --arm B.json [...] --margins MARGINS.json --name
    NAME --question TEXT --store ROOT --output DECISION.json`.

  Each exits 1 with `Margins failed:` or `Decision failed:` and the reason.

- [x] **Step 1: Write the failing tests**

> Deviation: by user ruling after the task review, `test_decide_reports_a_tie_it_cannot_break` asserts `'tie on components' in ' '.join(result.output.split())`: the plan's `'tie'` also matched pytest's tmp_path name (74348d5).

Modify `tests/unit/test_cli_commands.py` with these 2 edits, in order.

**`tests/unit/test_cli_commands.py`, edit 1 of 2.** Replace:

```python
from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle
```

with:

```python
from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.decision.records import DecisionRecord, MarginRecord, read_record, write_record
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle
from tests.fixtures.decision import spec, synthetic_arm
```

**`tests/unit/test_cli_commands.py`, edit 2 of 2.** Replace:

```python
    assert verify_inputs == {}

```

with:

```python
    assert verify_inputs == {}

# -------------------------------------------------------------------------------------------------
# Decisions (Req 5)
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def decision_inputs(tmp_path):
    '''A synthetic reference arm's record, written, and the store it references.'''

    store = ArtifactStore(tmp_path / 'store')
    reference = write_record(
        synthetic_arm(store, tmp_path, spec('reference'), {}), tmp_path / 'reference.json'
    )
    return store, reference

def _margins(runner, tmp_path, store, reference):
    return runner.invoke(
        tools_cli.app,
        [
            'margins',
            '--reference',
            str(reference),
            '--multiple',
            '2',
            '--name',
            'fixture margins',
            '--store',
            str(store.root),
            '--output',
            str(tmp_path / 'margins.json'),
        ],
    )

def _decide(runner, tmp_path, store, arms):
    arguments = ['decide']
    for path in arms:
        arguments += ['--arm', str(path)]
    arguments += [
        '--margins',
        str(tmp_path / 'margins.json'),
        '--name',
        'fixture decision',
        '--question',
        'which arm?',
        '--store',
        str(store.root),
        '--output',
        str(tmp_path / 'decision.json'),
    ]
    return runner.invoke(tools_cli.app, arguments)

@pytest.mark.unit
def test_margins_writes_each_panels_margin(runner, tmp_path, decision_inputs):
    store, reference = decision_inputs

    result = _margins(runner, tmp_path, store, reference)

    assert result.exit_code == 0, result.output
    record = read_record(tmp_path / 'margins.json', MarginRecord)
    panels = [entry.panel for entry in record.margins]
    assert panels == ['outcome', 'regressor_seen', 'regressor_heldout']
    assert 'regressor_heldout: δ' in result.output.replace('\n', '')

@pytest.mark.unit
def test_margins_reports_a_missing_reference(runner, tmp_path, decision_inputs):
    store, _ = decision_inputs

    result = _margins(runner, tmp_path, store, tmp_path / 'missing.json')

    assert result.exit_code == 1
    assert 'Margins failed' in result.output
    assert not (tmp_path / 'margins.json').exists()

@pytest.mark.unit
def test_decide_adopts_the_better_arm_and_writes_the_record(runner, tmp_path, decision_inputs):
    store, reference = decision_inputs
    assert _margins(runner, tmp_path, store, reference).exit_code == 0
    better = write_record(
        synthetic_arm(store, tmp_path, spec('better', dimension=32), {'outcome': 5}),
        tmp_path / 'better.json',
    )

    result = _decide(runner, tmp_path, store, [better, reference])

    assert result.exit_code == 0, result.output
    output = result.output.replace('\n', '')
    assert 'better over reference: adopted' in output
    assert 'Chosen: better' in output
    assert read_record(tmp_path / 'decision.json', DecisionRecord).chosen == 'better'

@pytest.mark.unit
def test_decide_reports_a_tie_it_cannot_break(runner, tmp_path, decision_inputs):
    store, reference = decision_inputs
    assert _margins(runner, tmp_path, store, reference).exit_code == 0
    twin = write_record(synthetic_arm(store, tmp_path, spec('twin'), {}), tmp_path / 'twin.json')

    result = _decide(runner, tmp_path, store, [twin, reference])

    assert result.exit_code == 1
    assert 'Decision failed' in result.output
    assert 'tie' in result.output.replace('\n', '')
    assert not (tmp_path / 'decision.json').exists()

```

Modify `tests/unit/test_config.py` with these 2 edits, in order.

**`tests/unit/test_config.py`, edit 1 of 2.** Replace:

```python
    Config,
```

with:

```python
    Config,
    DecisionConfig,
```

**`tests/unit/test_config.py`, edit 2 of 2.** Replace:

```python
            RegressorBranchRecord(**{**BRANCH_RECORD, 'branch': 'D'})

```

with:

```python
            RegressorBranchRecord(**{**BRANCH_RECORD, 'branch': 'D'})

@pytest.mark.unit
class TestDecisionConfig:
    '''Req 5's decision procedure (roadmap Stage 4): the paired bootstrap and the seed floor.'''

    def test_yaml_matches_defaults(self):
        cfg = load_config(DecisionConfig, 'data/decision.yaml')

        assert cfg == DecisionConfig()
        assert (cfg.replicates, cfg.bootstrap_seed, cfg.min_seeds) == (10000, 20260924, 5)

    def test_the_seed_floor_is_req_5s(self):
        # Req 5: "Each arm runs at least 5 seeds"
        with pytest.raises(ValidationError):
            DecisionConfig(min_seeds=4)

    def test_rejects_too_few_replicates_and_unknown_keys(self):
        with pytest.raises(ValidationError):
            DecisionConfig(replicates=999)
        with pytest.raises(ValidationError):
            DecisionConfig(seeds=5)

```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py::TestDecisionConfig -q`
Expected: one collection error, `ImportError: cannot import name 'DecisionConfig' from
'naics_embedder.utils.config'`.

Run: `uv run pytest tests/unit/test_cli_commands.py -q`
Expected: `4 failed, 32 passed`. The four failures are `test_margins_writes_each_panels_margin`,
`test_margins_reports_a_missing_reference`,
`test_decide_adopts_the_better_arm_and_writes_the_record` and
`test_decide_reports_a_tie_it_cannot_break`, each with `No such command 'margins'` or
`No such command 'decide'`.

- [x] **Step 3: Implement**

Modify `src/naics_embedder/cli/commands/tools.py` with these 5 edits, in order.

**`src/naics_embedder/cli/commands/tools.py`, edit 1 of 5.** Replace:

```python
    regressor-panel: Score an arm on the regressor panel's validation or sealed test split.
```

with:

```python
    regressor-panel: Score an arm on the regressor panel's validation or sealed test split.
    margins: Fix each panel's non-inferiority margin from a reference arm (Req 5).
    decide: Decide among arms under Req 5's rule over D8's three panels.
```

**`src/naics_embedder/cli/commands/tools.py`, edit 2 of 5.** Replace:

```python
from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
```

with:

```python
from naics_embedder.decision.decide import decide, fix_margins
from naics_embedder.decision.records import ArmRecord, MarginRecord, read_record, write_record
from naics_embedder.decision.rule import TieUnresolvedError
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
```

**`src/naics_embedder/cli/commands/tools.py`, edit 3 of 5.** Replace:

```python
from naics_embedder.utils.config import (
```

with:

```python
from naics_embedder.utils.config import (
    DecisionConfig,
```

**`src/naics_embedder/cli/commands/tools.py`, edit 4 of 5.** Replace:

```python
REGRESSOR_PANEL_CONFIG = 'data/regressor_panel.yaml'
```

with:

```python
REGRESSOR_PANEL_CONFIG = 'data/regressor_panel.yaml'
DECISION_CONFIG = 'data/decision.yaml'
```

**`src/naics_embedder/cli/commands/tools.py`, edit 5 of 5.** Replace:

```python
        console.print(f'Predictions written to {output_path}')
```

with:

```python
        console.print(f'Predictions written to {output_path}')

# -------------------------------------------------------------------------------------------------
# Decisions (Req 5)
# -------------------------------------------------------------------------------------------------

@app.command('margins')
def margins_command(
    reference: Annotated[
        str,
        typer.Option('--reference', help="The reference configuration's arm record (JSON)"),
    ],
    multiple: Annotated[
        float,
        typer.Option('--multiple', help="Each δ as a multiple of the reference's across-seed SD"),
    ],
    name: Annotated[
        str,
        typer.Option('--name', help='Names the margins in the decision records that use them'),
    ],
    store: Annotated[
        str,
        typer.Option('--store', help='The artifact store the arm record references'),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='Where to write the margin record (JSON)'),
    ],
):
    '''
    Fix each panel's non-inferiority margin δ from a reference arm (Req 5).

    δ is the multiple times the reference arm's across-seed standard deviation of the panel's
    decision statistic (D10). Fix the margins before any other arm of a decision reads a panel: a
    decision refuses every run that read before its margins were fixed.

    Example:
        Fix the margins at half a standard deviation::

            $ uv run naics-embedder tools margins --reference reference.json --multiple 0.5 \\
                --name reference-margins --store ~/naics-artifacts --output margins.json
    '''

    configure_logging('tools_margins.log')

    cfg = load_config(DecisionConfig, DECISION_CONFIG)
    try:
        record = fix_margins(
            read_record(reference, ArmRecord),
            multiple,
            name,
            ArtifactStore(store),
            min_seeds=cfg.min_seeds,
        )
        path = write_record(record, output)
    except (OSError, ValueError) as exc:
        console.print(f'[bold red]Margins failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(
        f'\n[bold cyan]Margins {name!r}, fixed {record.fixed_at.isoformat()}[/bold cyan]\n'
    )
    for entry in record.margins:
        console.print(
            f'  • {entry.panel}: δ {entry.margin:.6g} = {multiple:g} × SD {entry.sd:.6g} of '
            f'{entry.statistic} over {len(entry.per_seed)} seeds'
        )
    console.print(f'\nMargin record: {path}\n')

@app.command('decide')
def decide_command(
    arm: Annotated[
        List[str],
        typer.Option('--arm', help='An arm record (JSON); repeat for each arm'),
    ],
    margins: Annotated[
        str,
        typer.Option('--margins', help='The margin record (tools margins)'),
    ],
    name: Annotated[
        str,
        typer.Option('--name', help='Names the decision'),
    ],
    question: Annotated[
        str,
        typer.Option('--question', help='What the decision settles, in a sentence'),
    ],
    store: Annotated[
        str,
        typer.Option('--store', help='The artifact store the arm records reference'),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='Where to write the decision record (JSON)'),
    ],
):
    '''
    Decide among two or more arms under Req 5's rule over D8's three panels.

    Each A-against-B comparison reads Δ on paired resamples of each panel's units, seeds nested.
    A is adopted over B when it is non-inferior on all three panels (the 95 % interval's lower
    bound above −δ) and superior on at least one (the 98⅓ % interval above zero). The survivors
    are the arms no other arm is adopted over, and the tie order picks among them. The record
    carries the arms with their selection-log records and artifact references, the margins, every
    comparison, the non-dominated set, the tie order and the chosen arm.

    Example:
        Decide between a candidate and the reference::

            $ uv run naics-embedder tools decide --arm candidate.json --arm reference.json \\
                --margins margins.json --name dimension-8 --question "Is dimension 8 enough?" \\
                --store ~/naics-artifacts --output decision.json
    '''

    configure_logging('tools_decide.log')

    cfg = load_config(DecisionConfig, DECISION_CONFIG)
    try:
        record = decide(
            name,
            question,
            [read_record(path, ArmRecord) for path in arm],
            read_record(margins, MarginRecord),
            ArtifactStore(store),
            replicates=cfg.replicates,
            bootstrap_seed=cfg.bootstrap_seed,
            min_seeds=cfg.min_seeds,
        )
        path = write_record(record, output)
    except (OSError, ValueError, TieUnresolvedError) as exc:
        console.print(f'[bold red]Decision failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(f'\n[bold cyan]Decision {name!r}[/bold cyan]\n')
    for comparison in record.comparisons:
        verdict = 'adopted' if comparison.adopted else 'not adopted'
        console.print(f'  • {comparison.a} over {comparison.b}: {verdict}')
        for panel in comparison.panels:
            low, high = panel.noninferiority_interval
            upper_low, upper_high = panel.superiority_interval
            console.print(
                f'      {panel.panel}: Δ {panel.delta:+.4g}; 95 % [{low:+.4g}, {high:+.4g}] '
                f'against −δ {-panel.margin:.4g}; 98⅓ % [{upper_low:+.4g}, {upper_high:+.4g}]'
            )
    cycle = ' (dominance cycled)' if record.cycle else ''
    console.print(f'\nNon-dominated: {", ".join(record.non_dominated)}{cycle}')
    console.print(f'Tie order: {", ".join(record.tie_order)}')
    console.print(f'[bold]Chosen: {record.chosen}[/bold]')
    console.print(f'\nDecision record: {path}\n')
```

Modify `src/naics_embedder/utils/config.py` with one edit. Replace:

```python
class SupervisionRuntimeConfig(BaseModel):
```

with:

```python
class DecisionConfig(BaseModel):
    '''Req 5's decision procedure (roadmap Stage 4): the paired bootstrap and the seed floor.'''

    model_config = ConfigDict(extra='forbid')

    replicates: int = Field(
        default=10000, ge=1000, description='Paired two-stage bootstrap replicates per panel'
    )
    bootstrap_seed: int = Field(
        default=20260924, description='Base seed of every unit and seed draw'
    )
    min_seeds: int = Field(default=5, ge=5, description='Req 5: each arm runs at least 5 seeds')

class SupervisionRuntimeConfig(BaseModel):
```

Create `conf/data/decision.yaml` with exactly this content:

```yaml
# Req 5's decision procedure (roadmap Stage 4; D8, D10, D11): the paired bootstrap and the seed floor

# Paired two-stage bootstrap: each replicate draws a panel's units once for every arm (codes for
# the outcome panel, four-digit groups for the regressor regimes), then each arm's seeds inside
# that draw. Unit draws are seeded [bootstrap_seed, stream(panel)] and an arm's seed draws
# [bootstrap_seed, stream(panel), stream(arm)], so a decision is reproducible.
replicates: 10000
bootstrap_seed: 20260924

# Req 5: each arm runs at least 5 seeds; the config cannot go lower
min_seeds: 5
```

Modify `docs/.nav.yml` with one edit. Replace:

```yaml
          - Regressor Panel: api/regressor_panel.md
```

with:

```yaml
          - Regressor Panel: api/regressor_panel.md
          - Decision Rule: api/decision.md
```

Create `docs/api/decision.md` with exactly this content:

```markdown
# Decision Rule API

Req 5's decision procedure over the three panels of roadmap D8, and the records that carry it
(roadmap Stage 4).

## Decision statistics

::: naics_embedder.decision.scores

## Paired resampling

::: naics_embedder.decision.resampling

## The rule

::: naics_embedder.decision.rule

## Records and artifacts

::: naics_embedder.decision.records

::: naics_embedder.decision.store

## Margins and decisions

::: naics_embedder.decision.decide

## Seed sweeps

::: naics_embedder.decision.sweep
```

Modify `docs/usage.md` with one edit. Replace:

```markdown
- `--output PATH` - Write the per-row predictions as parquet

```

with:

````markdown
- `--output PATH` - Write the per-row predictions as parquet

### `tools margins`

Fix each panel's non-inferiority margin δ from a reference arm (Req 5): δ is `--multiple` times
the reference's across-seed standard deviation of the panel's decision statistic (roadmap D10:
per-query MRR on the outcome panel, and the `covariates+embedding` comparator's mean squared
error on each regressor regime at level 6). Fix the margins before any other arm of the decision
reads a panel: a decision refuses every run that read before its margins were fixed.

**Generates:** the margin record (JSON)

```bash
uv run naics-embedder tools margins --reference reference.json --multiple 0.5 \
  --name reference-margins --store ~/naics-artifacts --output margins.json
```

**Options:**
- `--reference PATH` - The reference configuration's arm record, written by the seed-sweep
  driver (`naics_embedder.decision.sweep.run_seed_sweep`)
- `--multiple FLOAT` - Each δ as a multiple of the reference's across-seed standard deviation
- `--name TEXT` - Names the margins in the decision records that use them
- `--store PATH` - The artifact store the arm record references
- `--output PATH` - Where to write the margin record; an existing file is never overwritten

### `tools decide`

Decide among two or more arms under Req 5's rule over the three panels of roadmap D8: the outcome
panel and the regressor panel's seen and held-out regimes at level 6. Each comparison reads the
difference Δ on paired resamples of each panel's units (codes with their queries; four-digit
groups), with each arm's seeds resampled inside every replicate. A is adopted over B when it is
non-inferior on all three panels (the 95 % interval's lower bound above −δ) and superior on at
least one (the 98⅓ % interval above zero). The survivors are the arms no other arm is adopted
over, and the tie order picks among them: fewer components, then lower dimension, then
non-hyperbolic geometry, then the higher held-out gain over ancestors (roadmap D11). The number
of replicates, the bootstrap seed and the seed floor come from `conf/data/decision.yaml`.

**Generates:** the decision record (JSON): the arms with their selection-log records and artifact
references, the margins, every comparison with both intervals, the non-dominated set, the tie
order, the chosen arm, and each arm's reported gains over the sparse comparators

```bash
uv run naics-embedder tools decide --arm candidate.json --arm reference.json \
  --margins margins.json --name dimension-8 --question "Is dimension 8 enough?" \
  --store ~/naics-artifacts --output decision.json
```

**Options:**
- `--arm PATH` - An arm record; repeat for each arm
- `--margins PATH` - The margin record (`tools margins`)
- `--name TEXT`, `--question TEXT` - Name the decision and say what it settles
- `--store PATH` - The artifact store the arm records reference
- `--output PATH` - Where to write the decision record; an existing file is never overwritten

````

Modify `CLAUDE.md` with these 3 edits, in order.

**`CLAUDE.md`, edit 1 of 3.** Replace:

```markdown
│   │   └── regressor.py      # RegressorPanel: two regimes, sealed outer sets, predictions
```

with:

```markdown
│   │   └── regressor.py      # RegressorPanel: two regimes, sealed outer sets, predictions
│   ├── decision/             # Req 5's decision rule over D8's three panels (roadmap Stage 4)
│   │   ├── scores.py         # Each panel's per-unit scores and decision statistic (D10)
│   │   ├── resampling.py     # Paired two-stage bootstrap: units shared, seeds per arm
│   │   ├── rule.py           # Non-inferiority, superiority, non-dominated set, tie order
│   │   ├── records.py        # Arm, margin and decision records (JSON, never overwritten)
│   │   ├── store.py          # Content-addressed artifact store
│   │   ├── decide.py         # Guards, margins and decisions
│   │   └── sweep.py          # Seed-sweep driver: N seeds, every panel read, all stored
```

**`CLAUDE.md`, edit 2 of 3.** Replace:

```markdown
│   │   ├── regressor_heldout_groups.csv  # The regressor panel's held-out groups (committed)
```

with:

```markdown
│   │   ├── regressor_heldout_groups.csv  # The regressor panel's held-out groups (committed)
│   │   ├── decision.yaml          # Decision rule: bootstrap replicates and seed, seed floor
```

**`CLAUDE.md`, edit 3 of 3.** Replace:

```markdown
uv run naics-embedder tools regressor-panel  # Score an arm on the regressor panel
```

with:

```markdown
uv run naics-embedder tools regressor-panel  # Score an arm on the regressor panel
uv run naics-embedder tools margins   # Fix each panel's margin from a reference arm (Req 5)
uv run naics-embedder tools decide    # Decide among arms under Req 5's rule
```

Modify `WARP.md` with one edit. Replace:

```markdown
- `src/naics_embedder/utils/`  
```

with:

```markdown
- `src/naics_embedder/decision/`  
  Req 5's decision rule over the outcome panel and the regressor panel's two regimes: decision statistics, paired resampling, the rule and tie order, decision records, the content-addressed artifact store, and the seed-sweep driver.

- `src/naics_embedder/utils/`  
```

- [x] **Step 4: Run the tests to verify they pass, and build the docs**

Run: `uv run pytest tests/unit/test_config.py::TestDecisionConfig tests/unit/test_cli_commands.py -q`
Expected: `39 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1601 passed, 1 skipped`.

Run: `uv run mkdocs build --strict -q -d /tmp/stage4-docs-ec267a03`, then
`rm -rf /tmp/stage4-docs-ec267a03`
Expected: no output and exit 0. The docs workflow runs only on pushes to `main`, so a PR's CI
never builds the API pages. This build is their only check before merge.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/cli/commands/tools.py src/naics_embedder/utils/config.py tests/unit/test_cli_commands.py tests/unit/test_config.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/cli/commands/tools.py \
  src/naics_embedder/utils/config.py \
  conf/data/decision.yaml \
  tests/unit/test_cli_commands.py \
  tests/unit/test_config.py \
  docs/.nav.yml \
  docs/api/decision.md \
  docs/usage.md \
  CLAUDE.md \
  WARP.md
git commit -m "feat(cli): add tools margins and tools decide"
```

### Task 9: Req 6's diagnostics report and `tools diagnostics`

The report covers every code of a table in the export form, and every statistic on it is
descriptive: no threshold, no pass or fail.

- D* comes from the codes' own lineage, with a virtual root above the sectors (Req 7).
- Distances follow the arm's geometry.
- The statistics are the ones Req 6 lists, stratified as it lists them:
  - sector separation as an AUC;
  - within-sector rank correlation, averaged over queries and over sectors;
  - MAP over ancestors, also by query level;
  - NDCG@5/10/20 with integer lowest-common-ancestor grades, also by level;
  - the Pearson correlation of distance with D*, without the cophenetic name;
  - parent retrieval@1/5 without the 522 unary pairs.

`tools diagnostics` prints the report and can write it as JSON. It replaces `verify-stage4` for
before-and-after comparisons (user decision 2).

**Files:**

- Modify: `src/naics_embedder/cli/commands/tools.py` (`tools diagnostics`)
- Modify: `src/naics_embedder/metrics/__init__.py` (export the report)
- Create: `src/naics_embedder/metrics/diagnostics.py` (Req 6's report)
- Modify: `tests/unit/test_cli_commands.py`
- Create: `tests/unit/test_diagnostics.py`
- Modify: `docs/.nav.yml` (the API page)
- Create: `docs/api/diagnostics.md` (API page)
- Modify: `docs/usage.md` (the command)
- Modify: `CLAUDE.md` (the command)

**Interfaces:**

- Consumes:
  - `code_lineage` (`panels.decoding`), which gives each code's lineage from its sector down to
    itself, with combined sectors as one;
  - `euclidean_distances`, `cosine_distances` and `lorentz_distances` (`panels.decoding`), the
    decoding scorer's own distances;
  - `coordinate_matrix(table)` (`panels.regressor`), which reads the export form's codes and
    coordinates;
  - in the tests, `CODEBOOK` and `coordinate_table` from `tests.fixtures.regressor_panel`.
- Produces, in `naics_embedder.metrics.diagnostics`:
  - `GEOMETRIES = ('euclidean', 'spherical', 'hyperbolic')`, `MAX_DEPTH = 5`,
    `NDCG_KS = (5, 10, 20)` and `PARENT_KS = (1, 5)`;
  - the report models `SectorSeparation(auc, same_sector_pairs, cross_sector_pairs)`,
    `WithinSectorRankCorrelation(mean_over_queries, mean_over_sectors, by_sector, queries,
    undefined_queries)`, `ByQueryLevel(value, queries, by_level)`, `DistancePearson(value, pairs)`,
    `ParentRetrieval(at, queries, unary_pairs_excluded)` and `DiagnosticsReport(codes, geometry,
    curvature, sector_separation, within_sector_rank_correlation, map_over_ancestors, ndcg,
    distance_pearson, parent_retrieval)`;
  - `Tree.from_codes(codes) -> Tree`, with `lineage`, `depth`, `sector`, `parent`,
    `lca_depth()` and `unary_children()`;
  - `pairwise_distances(matrix: np.ndarray, geometry, curvature: float = 1.0) -> np.ndarray`;
  - `sector_separation`, `within_sector_rank_correlation`, `average_precision`, `ndcg_at`,
    `map_over_ancestors`, `ndcg`, `distance_pearson` and `parent_retrieval`, with the signatures
    in Step 3's code;
  - `diagnostics_report(table: pl.DataFrame, geometry, *, codebook_codes: Optional[Sequence[str]]
    = None, curvature: float = 1.0) -> DiagnosticsReport`, exported from
    `naics_embedder.metrics` with `DiagnosticsReport`;
  - `naics-embedder tools diagnostics --table TABLE --geometry G --codebook CODEBOOK
    [--curvature C] [--output REPORT.json]`.

- [x] **Step 1: Write the failing tests**

Modify `tests/unit/test_cli_commands.py` with these 3 edits, in order.

**`tests/unit/test_cli_commands.py`, edit 1 of 3.** Replace:

```python
from naics_embedder.metrics import StructuralMetricInputError
```

with:

```python
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.metrics.diagnostics import DiagnosticsReport
```

**`tests/unit/test_cli_commands.py`, edit 2 of 3.** Replace:

```python
# -------------------------------------------------------------------------------------------------
# Decisions (Req 5)
```

with:

```python
# -------------------------------------------------------------------------------------------------
# Decisions (Req 5) and diagnostics (Req 6)
```

**`tests/unit/test_cli_commands.py`, edit 3 of 3.** Replace:

```python
    assert not (tmp_path / 'decision.json').exists()

```

with:

```python
    assert not (tmp_path / 'decision.json').exists()

@pytest.mark.unit
def test_diagnostics_reports_every_statistic_and_writes_json(runner, tmp_path):
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    table = tmp_path / 'arm.parquet'
    coordinate_table(CODEBOOK, dimension=4).write_parquet(table)
    arguments = ['diagnostics', '--table', str(table), '--codebook', str(codebook)]

    result = runner.invoke(
        tools_cli.app,
        [*arguments, '--geometry', 'hyperbolic', '--output',
         str(tmp_path / 'report.json')],
    )
    unknown = runner.invoke(tools_cli.app, [*arguments, '--geometry', 'poincare'])

    assert result.exit_code == 0, result.output
    output = result.output.replace('\n', '')
    for statistic in (
        'sector separation AUC', 'within-sector rank correlation', 'MAP over ancestors', 'NDCG',
        'distance Pearson', 'parent retrieval'
    ):
        assert statistic in output
    report = json.loads((tmp_path / 'report.json').read_text())
    assert set(report) == set(DiagnosticsReport.model_fields)
    assert unknown.exit_code == 1

@pytest.mark.unit
def test_diagnostics_refuses_a_table_that_misses_a_codebook_code(runner, tmp_path):
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    table = tmp_path / 'arm.parquet'
    coordinate_table(CODEBOOK[1:], dimension=4).write_parquet(table)

    result = runner.invoke(
        tools_cli.app,
        [
            'diagnostics', '--table',
            str(table), '--codebook',
            str(codebook), '--geometry', 'euclidean'
        ],
    )

    assert result.exit_code == 1
    assert 'Diagnostics failed' in result.output

```

Create `tests/unit/test_diagnostics.py` with exactly this content:

```python
'''
Req 6's diagnostics, worked by hand on small trees and distance matrices.

The tree below has two sectors. 11111 has two six-digit children; 11121 has one, 111211, so
(11121, 111211) is its only unary pair (Req 9).
'''

import numpy as np
import polars as pl
import pytest

from naics_embedder.metrics.diagnostics import (
    DiagnosticsReport,
    Tree,
    average_precision,
    diagnostics_report,
    distance_pearson,
    ndcg_at,
    pairwise_distances,
    parent_retrieval,
    sector_separation,
    within_sector_rank_correlation,
)
from tests.fixtures.regressor_panel import CODEBOOK, coordinate_table

pytestmark = pytest.mark.unit

CODES = (
    '11', '111', '1111', '11111', '111111', '111112', '1112', '11121', '111211', '21', '211',
    '2111', '21111', '211111', '211112'
)

@pytest.fixture
def tree():
    return Tree.from_codes(CODES)

def _row(code):
    return CODES.index(code)

def _target(tree):
    return tree.depth[:, None] + tree.depth[None, :] - 2 * tree.lca_depth()

def test_d_star_runs_through_the_lowest_common_ancestor_and_a_virtual_root(tree):
    target = _target(tree)

    assert tree.depth[[_row('11'), _row('111211')]].tolist() == [1, 5]
    assert target[_row('111111'), _row('111112')] == 2  # siblings, through 11111
    assert target[_row('111111'), _row('1111')] == 2  # an ancestor two levels up
    assert target[_row('111111'), _row('111211')] == 6  # through 111
    assert target[_row('111111'), _row('21')] == 6  # through the virtual root: 5 + 1
    assert tree.parent[_row('111')] == _row('11')
    assert tree.unary_children().tolist() == [code == '111211' for code in CODES]

def test_a_code_whose_ancestor_is_missing_is_refused():
    with pytest.raises(ValueError, match='ancestor 1111 is not among the codes'):
        Tree.from_codes(['11', '111', '11111'])

def test_sector_separation_is_the_auc_of_cross_over_same_sector_distances():
    # Same-sector pairs at 1 and 2; cross-sector pairs at 2, 3, 4 and 5: of 8 comparisons, 7
    # are won and one (2 against 2) tied
    distances = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 4.0, 5.0],
            [2.0, 4.0, 0.0, 2.0],
            [3.0, 5.0, 2.0, 0.0],
        ]
    )

    result = sector_separation(distances, np.array([0, 0, 1, 1]))

    assert result.auc == pytest.approx(7.5 / 8)
    assert (result.same_sector_pairs, result.cross_sector_pairs) == (2, 4)

def test_within_sector_rank_correlation_is_one_when_distance_orders_as_d_star(tree):
    target = _target(tree).astype(float)

    exact = within_sector_rank_correlation(target, target, tree.sector, CODES)
    reversed_ = within_sector_rank_correlation(-target, target, tree.sector, CODES)
    flat = within_sector_rank_correlation(np.ones_like(target), target, tree.sector, CODES)

    assert exact.mean_over_queries == pytest.approx(1.0)
    assert exact.by_sector == pytest.approx({'11': 1.0, '21': 1.0})
    assert (exact.queries, exact.undefined_queries) == (15, 0)
    assert reversed_.mean_over_sectors == pytest.approx(-1.0)
    assert (flat.mean_over_queries, flat.undefined_queries) == (None, 15)

def test_average_precision_breaks_ties_against_relevance():
    # Ranked: 0 (relevant), 2, 1 (relevant: tied with 2, so after it), 3 (relevant), 5, 4
    # (relevant: tied with 5)
    distances = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0])
    relevant = np.array([True, True, False, True, True, False])

    assert average_precision(distances, relevant) == pytest.approx(
        (1 / 1 + 2 / 3 + 3 / 4 + 4 / 6) / 4
    )

def test_ndcg_uses_linear_integer_gains_and_breaks_ties_against_them():
    # Ranked gains: 1 then 4 (tied at distance 1), then 3 and 3, then 0
    distances = np.array([1.0, 2.0, 2.0, 3.0, 1.0])
    gains = np.array([4.0, 3.0, 3.0, 0.0, 1.0])

    dcg = 1 + 4 / np.log2(3) + 3 / np.log2(4)
    ideal = 4 + 3 / np.log2(3) + 3 / np.log2(4)
    assert ndcg_at(distances, gains, 3) == pytest.approx(dcg / ideal)
    assert ndcg_at(distances, np.zeros(5), 3) is None

def test_the_pearson_statistic_correlates_distance_with_d_star_over_all_pairs(tree):
    target = _target(tree)

    result = distance_pearson(target.astype(float), target)

    assert result.value == pytest.approx(1.0)
    assert result.pairs == 15 * 14 // 2
    assert distance_pearson(np.ones(target.shape), target).value is None

def test_parent_retrieval_skips_the_unary_pairs_and_breaks_ties_against_the_parent(tree):
    # Under D*, a code's parent and children are all at 1: only a leaf finds its parent first
    result = parent_retrieval(_target(tree).astype(float), tree)

    assert (result.queries, result.unary_pairs_excluded) == (12, 1)
    # Leaves 111111, 111112, 211111 and 211112 rank their parent first
    assert result.at == pytest.approx({'1': 4 / 12, '5': 1.0})

def test_each_geometry_reads_its_own_distance():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0], [6.0, 8.0]])

    euclidean = pairwise_distances(points, 'euclidean')
    spherical = pairwise_distances(points[1:], 'spherical')
    hyperbolic = pairwise_distances(points, 'hyperbolic')
    curved = pairwise_distances(points, 'hyperbolic', curvature=4.0)

    assert euclidean[0, 1] == pytest.approx(5.0)
    assert spherical[0, 2] == pytest.approx(0.0, abs=1e-12)
    # A tangent vector at the origin maps to a point its length away, along its ray
    assert hyperbolic[0, 1] == pytest.approx(5.0)
    assert hyperbolic[1, 3] == pytest.approx(5.0)
    assert curved[0, 1] == pytest.approx(5.0)
    # Off the ray, distances grow faster than Euclidean ones, and faster with curvature
    assert euclidean[1, 2] < hyperbolic[1, 2] < curved[1, 2]

def _table(codes, seed=3, dimension=4):
    return coordinate_table(codes, dimension=dimension, seed=seed)

def test_the_report_holds_only_req_6s_statistics():
    assert set(DiagnosticsReport.model_fields) == {
        'codes',
        'geometry',
        'curvature',
        'sector_separation',
        'within_sector_rank_correlation',
        'map_over_ancestors',
        'ndcg',
        'distance_pearson',
        'parent_retrieval',
    }

def test_the_report_covers_every_code_of_the_tree():
    report = diagnostics_report(_table(CODEBOOK), 'hyperbolic', codebook_codes=CODEBOOK)

    n = len(CODEBOOK)
    assert report.codes == n
    assert report.curvature == 1.0
    pairs = report.sector_separation.same_sector_pairs + report.sector_separation.cross_sector_pairs
    assert pairs == report.distance_pearson.pairs == n * (n - 1) // 2
    assert report.within_sector_rank_correlation.queries == n
    assert report.map_over_ancestors.queries == n - 4  # every code but the four sectors
    assert set(report.ndcg) == {'@5', '@10', '@20'}
    assert report.ndcg['@10'].queries == n
    assert set(report.map_over_ancestors.by_level) == {'3', '4', '5', '6'}
    # 238110 is its five-digit parent's only child
    assert report.parent_retrieval.unary_pairs_excluded == 1
    assert report.parent_retrieval.queries == n - 4 - 1
    assert diagnostics_report(_table(CODEBOOK), 'euclidean').curvature is None

def test_the_report_refuses_a_table_that_is_not_the_codebook():
    with pytest.raises(ValueError, match='covers the codebook'):
        diagnostics_report(_table(CODEBOOK[1:]), 'euclidean', codebook_codes=CODEBOOK)
    lorentz = pl.DataFrame({'code': ['11', '111'], 'x0': [1.0, 2.0], 'x1': [0.0, 3.0**0.5]})
    with pytest.raises(ValueError, match='Lorentz points'):
        diagnostics_report(lorentz, 'hyperbolic')
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_diagnostics.py tests/unit/test_cli_commands.py -q`
Expected: two collection errors, each
`ModuleNotFoundError: No module named 'naics_embedder.metrics.diagnostics'`.

- [x] **Step 3: Implement**

Modify `src/naics_embedder/cli/commands/tools.py` with these 3 edits, in order.

**`src/naics_embedder/cli/commands/tools.py`, edit 1 of 3.** Replace:

```python
    decide: Decide among arms under Req 5's rule over D8's three panels.
```

with:

```python
    decide: Decide among arms under Req 5's rule over D8's three panels.
    diagnostics: Report Req 6's structural diagnostics over every codebook code.
```

**`src/naics_embedder/cli/commands/tools.py`, edit 2 of 3.** Replace:

```python
from naics_embedder.panels.lexical_encoder import (
```

with:

```python
from naics_embedder.metrics.diagnostics import GEOMETRIES, diagnostics_report
from naics_embedder.panels.lexical_encoder import (
```

**`src/naics_embedder/cli/commands/tools.py`, edit 3 of 3.** Replace:

```python
    console.print(f'\nDecision record: {path}\n')
```

with:

```python
    console.print(f'\nDecision record: {path}\n')

# -------------------------------------------------------------------------------------------------
# Diagnostics (Req 6)
# -------------------------------------------------------------------------------------------------

@app.command('diagnostics')
def diagnostics_command(
    table: Annotated[
        str,
        typer.Option(
            '--table',
            help="The arm's code table in the export form (tangent coordinates if hyperbolic)",
        ),
    ],
    geometry: Annotated[
        str,
        typer.Option('--geometry', help='euclidean, spherical or hyperbolic'),
    ],
    codebook: Annotated[
        str,
        typer.Option('--codebook', help="A supervision bundle's naics_codebook.parquet"),
    ],
    curvature: Annotated[
        float,
        typer.Option('--curvature', help="A hyperbolic arm's curvature magnitude"),
    ] = 1.0,
    output: Annotated[
        Optional[str],
        typer.Option('--output', help='Also write the report as JSON to this path'),
    ] = None,
):
    '''
    Report Req 6's structural diagnostics over every codebook code.

    Sector separation (an AUC), within-sector rank correlation (over queries and over sectors),
    MAP over ancestors, NDCG@5/10/20 with integer lowest-common-ancestor grades, the Pearson
    correlation of distance with D*, and parent retrieval@1/5 without the 522 unary pairs. The
    report describes an arm: nothing selects on it, and no statistic in it has a threshold.

    Example:
        Report on a hyperbolic arm's export::

            $ uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic \\
                --codebook PATH/naics_codebook.parquet
    '''

    configure_logging('tools_diagnostics.log')

    if geometry not in GEOMETRIES:
        console.print(f'[bold red]--geometry must be one of {list(GEOMETRIES)}[/bold red]')
        raise typer.Exit(code=1)
    try:
        codes = pl.read_parquet(codebook).get_column('code').to_list()
        report = diagnostics_report(
            pl.read_parquet(table), geometry, codebook_codes=codes, curvature=curvature
        )
    except (OSError, ValueError) as exc:
        console.print(f'[bold red]Diagnostics failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    separation = report.sector_separation
    within = report.within_sector_rank_correlation
    ancestors = report.map_over_ancestors
    parents = report.parent_retrieval

    def formatted(value: Optional[float]) -> str:
        return 'undefined' if value is None else f'{value:.4f}'

    console.print(
        f'\n[bold cyan]Structural diagnostics (Req 6): {report.codes:,} codes, '
        f'{report.geometry}[/bold cyan]\n'
    )
    console.print(
        f'  • sector separation AUC: {separation.auc:.4f} ({separation.same_sector_pairs:,} '
        f'same-sector, {separation.cross_sector_pairs:,} cross-sector pairs)'
    )
    console.print(
        f'  • within-sector rank correlation: {formatted(within.mean_over_queries)} over '
        f'{within.queries - within.undefined_queries:,} queries, '
        f'{formatted(within.mean_over_sectors)} over {len(within.by_sector)} sectors '
        f'({within.undefined_queries:,} undefined)'
    )
    levels = ', '.join(f'level {level} {value:.4f}' for level, value in ancestors.by_level.items())
    console.print(
        f'  • MAP over ancestors: {ancestors.value:.4f} over {ancestors.queries:,} queries '
        f'({levels})'
    )
    ndcg = ', '.join(f'{k} {value.value:.4f}' for k, value in report.ndcg.items())
    console.print(f'  • NDCG: {ndcg}')
    console.print(
        f'  • distance Pearson with D*: {formatted(report.distance_pearson.value)} over '
        f'{report.distance_pearson.pairs:,} pairs'
    )
    at = ', '.join(f'@{k} {value:.4f}' for k, value in parents.at.items())
    console.print(
        f'  • parent retrieval: {at} over {parents.queries:,} queries '
        f'({parents.unary_pairs_excluded} unary pairs excluded)'
    )
    console.print('\nDescriptive only: nothing selects on these, and none has a threshold.\n')

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2) + '\n')
        console.print(f'Report written to {path}')
```

Modify `src/naics_embedder/metrics/__init__.py` with these 3 edits, in order.

**`src/naics_embedder/metrics/__init__.py`, edit 1 of 3.** Replace:

```python
- Hierarchy structure metrics
```

with:

```python
- Hierarchy structure metrics
- Req 6's structural diagnostics report
```

**`src/naics_embedder/metrics/__init__.py`, edit 2 of 3.** Replace:

```python
# Graph-specific metrics
```

with:

```python
# Req 6's diagnostics
from .diagnostics import DiagnosticsReport, diagnostics_report

# Graph-specific metrics
```

**`src/naics_embedder/metrics/__init__.py`, edit 3 of 3.** Replace:

```python
    'StructuralMetricInputError',
```

with:

```python
    'StructuralMetricInputError',
    # Diagnostics
    'DiagnosticsReport',
    'diagnostics_report',
```

Create `src/naics_embedder/metrics/diagnostics.py` with exactly this content:

```python
'''
Req 6's structural diagnostics over every code of a table: reported, never selected on and
never a headline (Req 1; Req 6; Verification "Diagnostics").

- **Sector separation.** The AUC of distance between same-sector and cross-sector pairs: the
  probability that a cross-sector pair lies farther apart than a same-sector pair, ties counting
  half.
- **Within-sector rank correlation.** Per query, Spearman's correlation of distance with D* over
  the other codes of its sector; averaged over queries, and over sectors (each sector's mean over
  its queries).
- **MAP over ancestors.** Per non-sector query, the average precision of its ancestors among
  all other codes ranked by distance.
- **NDCG@k.** Gains are integer depths of the lowest common ancestor (0 across sectors, 1 for a
  shared sector, up to 4), with linear gain and a 1/log2(rank + 1) discount.
- **Distance Pearson.** The Pearson correlation of distance with D* over all pairs: the
  statistic formerly named "cophenetic", which no dendrogram underlies.
- **Parent retrieval@k.** Per non-sector query, whether its parent is among its k nearest codes.
  The 522 unary pairs, five-digit industries whose only child is their six-digit code (Req 9),
  are not scored.

D* is the tree metric through a virtual root (Req 7): depth_i + depth_j − 2 depth_LCA, where a
sector has depth 1. It is computed here from the codes themselves (``panels.decoding``'s
lineage, combined sectors as one). Distances are the arm's own: Euclidean, cosine for a
spherical arm, and for a hyperbolic arm the geodesic distance between the exponential maps of
its tangent coordinates at the origin. A tie in distance is broken against relevance, as the
decoding scorer breaks it against the truth, except in the AUC and Spearman's ranks, which
average ties. No statistic has a threshold, and the report has no pass or fail.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata

from naics_embedder.panels.decoding import (
    code_lineage,
    cosine_distances,
    euclidean_distances,
    lorentz_distances,
)
from naics_embedder.panels.regressor import coordinate_matrix

Geometry = Literal['euclidean', 'spherical', 'hyperbolic']
GEOMETRIES = ('euclidean', 'spherical', 'hyperbolic')
MAX_DEPTH = 5
NDCG_KS = (5, 10, 20)
PARENT_KS = (1, 5)

# -------------------------------------------------------------------------------------------------
# The report
# -------------------------------------------------------------------------------------------------

class _Report(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

class SectorSeparation(_Report):
    auc: float
    same_sector_pairs: int
    cross_sector_pairs: int

class WithinSectorRankCorrelation(_Report):
    '''The means are None when no query's correlation is defined (a collapsed embedding).'''

    mean_over_queries: Optional[float]
    mean_over_sectors: Optional[float]
    by_sector: Dict[str, float]
    queries: int
    undefined_queries: int

class ByQueryLevel(_Report):
    value: float
    queries: int
    by_level: Dict[str, float]

class DistancePearson(_Report):
    '''None when every distance is equal.'''

    value: Optional[float]
    pairs: int

class ParentRetrieval(_Report):
    at: Dict[str, float]
    queries: int
    unary_pairs_excluded: int

class DiagnosticsReport(_Report):
    '''Req 6's statistics, stratified as it lists them; nothing else.'''

    codes: int
    geometry: Geometry
    curvature: Optional[float]
    sector_separation: SectorSeparation
    within_sector_rank_correlation: WithinSectorRankCorrelation
    map_over_ancestors: ByQueryLevel
    ndcg: Dict[str, ByQueryLevel]
    distance_pearson: DistancePearson
    parent_retrieval: ParentRetrieval

# -------------------------------------------------------------------------------------------------
# The tree
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tree:
    '''
    The codes' tree.

    Attributes:
        lineage: (codes, 5) row numbers of each code's ancestors from its sector down to itself,
            −1 below its depth.
        depth: Each code's depth, 1 for a sector.
    '''

    codes: Tuple[str, ...]
    lineage: np.ndarray
    depth: np.ndarray

    @classmethod
    def from_codes(cls, codes: Sequence[str]) -> 'Tree':
        '''
        Raises:
            ValueError: If a code repeats or an ancestor of a code is not among the codes.
        '''

        codes = tuple(codes)
        position = {code: row for row, code in enumerate(codes)}
        if len(position) != len(codes):
            raise ValueError('a code repeats')
        lineage = np.full((len(codes), MAX_DEPTH), -1, dtype=np.int64)
        for row, code in enumerate(codes):
            chain = code_lineage(code)
            missing = [ancestor for ancestor in chain if ancestor not in position]
            if missing:
                raise ValueError(f'{code}: its ancestor {missing[0]} is not among the codes')
            lineage[row, :len(chain)] = [position[ancestor] for ancestor in chain]
        return cls(codes, lineage, (lineage >= 0).sum(axis=1))

    @property
    def sector(self) -> np.ndarray:
        return self.lineage[:, 0]

    @property
    def parent(self) -> np.ndarray:
        '''Each code's parent row, −1 for a sector.'''

        rows = np.arange(len(self.codes))
        return np.where(self.depth > 1, self.lineage[rows, np.maximum(self.depth - 2, 0)], -1)

    def lca_depth(self) -> np.ndarray:
        '''(codes, codes) depth of each pair's lowest common ancestor, 0 across sectors.'''

        depth = np.zeros((len(self.codes), len(self.codes)), dtype=np.int64)
        for column in self.lineage.T:
            depth += (column[:, None] == column[None, :]) & (column[:, None] >= 0)
        return depth

    def unary_children(self) -> np.ndarray:
        '''Whether each code is the six-digit half of a unary pair (Req 9).'''

        parent = self.parent
        counts = np.bincount(parent[parent >= 0], minlength=len(self.codes))
        six_digit = np.array([len(code) == 6 for code in self.codes])
        return six_digit & (parent >= 0) & (counts[np.maximum(parent, 0)] == 1)

# -------------------------------------------------------------------------------------------------
# Distances
# -------------------------------------------------------------------------------------------------

def pairwise_distances(
    matrix: np.ndarray, geometry: Geometry, curvature: float = 1.0
) -> np.ndarray:
    '''
    The arm's own distances between every pair of rows.

    A hyperbolic arm's rows are tangent coordinates at the origin (the export form); each maps to
    the hyperboloid of curvature −``curvature`` by the exponential map at the origin.
    '''

    points = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float64))
    if geometry == 'euclidean':
        return euclidean_distances(points, points).numpy()
    if geometry == 'spherical':
        return cosine_distances(points, points).numpy()
    if geometry != 'hyperbolic':
        raise ValueError(f'geometry must be one of {GEOMETRIES}, got {geometry!r}')
    if curvature <= 0:
        raise ValueError(f'curvature must be positive, got {curvature}')
    root = math.sqrt(curvature)
    scaled = points * root
    norm = scaled.norm(dim=1, keepdim=True)
    space = torch.sinh(norm) * scaled / norm.clamp_min(1e-300)
    lorentz = torch.cat([torch.cosh(norm), space], dim=1)
    return (lorentz_distances(lorentz, lorentz) / root).numpy()

# -------------------------------------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------------------------------------

def sector_separation(distances: np.ndarray, sector: np.ndarray) -> SectorSeparation:
    '''The Mann–Whitney AUC of cross-sector over same-sector distances, over unordered pairs.'''

    upper = np.triu_indices(len(sector), k=1)
    values = distances[upper]
    same = (sector[:, None] == sector[None, :])[upper]
    ranks = rankdata(values)
    n_cross, n_same = int((~same).sum()), int(same.sum())
    auc = (ranks[~same].sum() - n_cross * (n_cross + 1) / 2) / (n_cross * n_same)
    return SectorSeparation(auc=float(auc), same_sector_pairs=n_same, cross_sector_pairs=n_cross)

def within_sector_rank_correlation(
    distances: np.ndarray, target: np.ndarray, sector: np.ndarray, codes: Sequence[str]
) -> WithinSectorRankCorrelation:
    '''Per query, Spearman's correlation of distance with D* over its sector's other codes.'''

    by_query: Dict[int, float] = {}
    undefined = 0
    for query in range(len(sector)):
        candidates = np.flatnonzero((sector == sector[query]) & (np.arange(len(sector)) != query))
        if len(candidates) < 2:
            undefined += 1
            continue
        a = rankdata(distances[query, candidates])
        b = rankdata(target[query, candidates])
        if a.std() == 0 or b.std() == 0:
            undefined += 1
            continue
        by_query[query] = float(np.corrcoef(a, b)[0, 1])
    by_sector: Dict[str, float] = {}
    for row in sorted(set(sector.tolist())):
        values = [value for query, value in by_query.items() if sector[query] == row]
        if values:
            by_sector[codes[row]] = float(np.mean(values))
    return WithinSectorRankCorrelation(
        mean_over_queries=float(np.mean(list(by_query.values()))) if by_query else None,
        mean_over_sectors=float(np.mean(list(by_sector.values()))) if by_sector else None,
        by_sector=by_sector,
        queries=len(by_query) + undefined,
        undefined_queries=undefined,
    )

def _pessimistic_order(distances: np.ndarray, relevance: np.ndarray) -> np.ndarray:
    '''Candidates by distance, a tie broken against relevance (the less relevant first).'''

    return np.lexsort((relevance, distances))

def average_precision(distances: np.ndarray, relevant: np.ndarray) -> float:
    '''One query's average precision of its relevant candidates, ranked by distance.'''

    positions = np.flatnonzero(relevant[_pessimistic_order(distances, relevant)]) + 1
    return float(np.mean(np.arange(1, len(positions) + 1) / positions))

def ndcg_at(distances: np.ndarray, gains: np.ndarray, k: int) -> Optional[float]:
    '''One query's NDCG@k with linear gains; None when no candidate has a gain.'''

    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(gains)[::-1][:k]
    best = (ideal * discount[:len(ideal)]).sum()
    if best == 0:
        return None
    ranked = gains[_pessimistic_order(distances, gains)][:k]
    return float((ranked * discount[:len(ranked)]).sum() / best)

def _by_level(values: Dict[int, float], depth: np.ndarray) -> ByQueryLevel:
    levels: Dict[str, float] = {}
    for level in sorted({int(depth[query]) + 1 for query in values}):
        chosen = [value for query, value in values.items() if depth[query] + 1 == level]
        levels[str(level)] = float(np.mean(chosen))
    return ByQueryLevel(
        value=float(np.mean(list(values.values()))), queries=len(values), by_level=levels
    )

def map_over_ancestors(distances: np.ndarray, tree: Tree) -> ByQueryLevel:
    '''Per non-sector query, the average precision of its ancestors among all other codes.'''

    everything = np.arange(len(tree.codes))
    values: Dict[int, float] = {}
    for query in np.flatnonzero(tree.depth > 1):
        candidates = everything[everything != query]
        relevant = np.isin(candidates, tree.lineage[query, :tree.depth[query] - 1])
        values[int(query)] = average_precision(distances[query, candidates], relevant)
    return _by_level(values, tree.depth)

def ndcg(distances: np.ndarray, grades: np.ndarray, depth: np.ndarray, k: int) -> ByQueryLevel:
    '''
    Per query, NDCG@k over all other codes with integer LCA-depth gains; a query that shares a
    sector with no other code has no gain to rank and is not scored.
    '''

    everything = np.arange(len(depth))
    values: Dict[int, float] = {}
    for query in everything:
        candidates = everything[everything != query]
        value = ndcg_at(distances[query, candidates], grades[query, candidates].astype(float), k)
        if value is not None:
            values[int(query)] = value
    return _by_level(values, depth)

def distance_pearson(distances: np.ndarray, target: np.ndarray) -> DistancePearson:
    '''The Pearson correlation of distance with D* over all unordered pairs.'''

    upper = np.triu_indices(len(target), k=1)
    values = distances[upper]
    if values.std() == 0:
        return DistancePearson(value=None, pairs=len(values))
    value = np.corrcoef(values, target[upper].astype(np.float64))[0, 1]
    return DistancePearson(value=float(value), pairs=len(values))

def parent_retrieval(distances: np.ndarray, tree: Tree) -> ParentRetrieval:
    '''Per non-sector query outside the unary pairs, whether its parent is among its k nearest.'''

    unary = tree.unary_children()
    queries = np.flatnonzero((tree.depth > 1) & ~unary)
    parent = tree.parent[queries]
    others = distances[queries].copy()
    others[np.arange(len(queries)), queries] = np.inf
    to_parent = others[np.arange(len(queries)), parent]
    # A tie is broken against the parent: every code as near as it ranks ahead
    ranks = (others <= to_parent[:, None]).sum(axis=1)
    return ParentRetrieval(
        at={str(k): float((ranks <= k).mean())
            for k in PARENT_KS},
        queries=len(queries),
        unary_pairs_excluded=int(unary.sum()),
    )

# -------------------------------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------------------------------

def diagnostics_report(
    table: pl.DataFrame,
    geometry: Geometry,
    *,
    codebook_codes: Optional[Sequence[str]] = None,
    curvature: float = 1.0,
) -> DiagnosticsReport:
    '''
    Req 6's statistics for an arm's code table in the export form.

    Args:
        table: ``code`` plus coordinates (``panels.regressor.coordinate_matrix``).
        geometry: The arm's geometry, which sets its distance.
        codebook_codes: If given, the table must cover exactly these codes.
        curvature: A hyperbolic arm's curvature magnitude.

    Raises:
        ValueError: If the table's codes differ from ``codebook_codes``, or an ancestor of a code
            is missing.
    '''

    codes, matrix = coordinate_matrix(table)
    if codebook_codes is not None and set(codes) != set(codebook_codes):
        raise ValueError(
            f"the table has {len(codes):,} codes; the report covers the codebook's "
            f'{len(set(codebook_codes)):,}'
        )
    tree = Tree.from_codes(codes)
    distances = pairwise_distances(matrix, geometry, curvature)
    lca = tree.lca_depth()
    target = tree.depth[:, None] + tree.depth[None, :] - 2 * lca
    return DiagnosticsReport(
        codes=len(codes),
        geometry=geometry,
        curvature=curvature if geometry == 'hyperbolic' else None,
        sector_separation=sector_separation(distances, tree.sector),
        within_sector_rank_correlation=within_sector_rank_correlation(
            distances, target, tree.sector, codes
        ),
        map_over_ancestors=map_over_ancestors(distances, tree),
        ndcg={f'@{k}': ndcg(distances, lca, tree.depth, k)
              for k in NDCG_KS},
        distance_pearson=distance_pearson(distances, target),
        parent_retrieval=parent_retrieval(distances, tree),
    )
```

Modify `docs/.nav.yml` with one edit. Replace:

```yaml
          - Decision Rule: api/decision.md
```

with:

```yaml
          - Decision Rule: api/decision.md
          - Diagnostics: api/diagnostics.md
```

Create `docs/api/diagnostics.md` with exactly this content:

```markdown
# Diagnostics API

Req 6's structural diagnostics: reported, never selected on and never a headline (roadmap
Stage 4).

::: naics_embedder.metrics.diagnostics
```

Modify `docs/usage.md` with one edit. Replace:

```markdown
- `--output PATH` - Where to write the decision record; an existing file is never overwritten

```

with:

````markdown
- `--output PATH` - Where to write the decision record; an existing file is never overwritten

### `tools diagnostics`

Report Req 6's structural diagnostics over every codebook code: sector separation (an AUC),
within-sector rank correlation (over queries and over sectors), MAP over ancestors,
NDCG@5/10/20 with integer lowest-common-ancestor grades, the Pearson correlation of distance
with the tree metric D*, and parent retrieval@1/5 without the 522 unary pairs. The report
describes an arm: nothing selects on it, and no statistic in it has a threshold. To compare two
tables, such as one before and one after graph refinement, report on each.

```bash
uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic \
  --codebook PATH/naics_codebook.parquet
```

**Options:**
- `--table PATH` - The arm's code table in the export form (tangent coordinates at the origin
  for a hyperbolic arm; Lorentz points are refused)
- `--geometry euclidean|spherical|hyperbolic` - The arm's distance: Euclidean, cosine, or the
  geodesic distance after the exponential map at the origin
- `--codebook PATH` - A supervision bundle's `naics_codebook.parquet`; the table must hold
  exactly its codes
- `--curvature FLOAT` - A hyperbolic arm's curvature magnitude (default: 1.0)
- `--output PATH` - Also write the report as JSON

````

Modify `CLAUDE.md` with one edit. Replace:

```markdown
uv run naics-embedder tools decide    # Decide among arms under Req 5's rule
```

with:

```markdown
uv run naics-embedder tools decide    # Decide among arms under Req 5's rule
uv run naics-embedder tools diagnostics  # Req 6's structural diagnostics for a table
```

- [x] **Step 4: Run the tests to verify they pass, and build the docs**

Run: `uv run pytest tests/unit/test_diagnostics.py tests/unit/test_cli_commands.py -q`
Expected: `50 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1615 passed, 1 skipped`.

Run: `uv run mkdocs build --strict -q -d /tmp/stage4-docs-ec267a03`, then
`rm -rf /tmp/stage4-docs-ec267a03`
Expected: no output and exit 0.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/cli/commands/tools.py src/naics_embedder/metrics/__init__.py src/naics_embedder/metrics/diagnostics.py tests/unit/test_cli_commands.py tests/unit/test_diagnostics.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/cli/commands/tools.py \
  src/naics_embedder/metrics/__init__.py \
  src/naics_embedder/metrics/diagnostics.py \
  tests/unit/test_cli_commands.py \
  tests/unit/test_diagnostics.py \
  docs/.nav.yml \
  docs/api/diagnostics.md \
  docs/usage.md \
  CLAUDE.md
git commit -m "feat(metrics): add Req 6's diagnostics report and tools diagnostics"
```

### Task 10: Remove the QCEW benchmark and the taxonomy-tasks suite

This task removes two things:

- `metrics/qcew.py`, the benchmark Req 2 rejects, with its re-exports, its API page and its
  tests;
- the unwired downstream suite in `metrics/graph.py` (`GraphDownstreamEvaluator` and
  `run_graph_downstream_suite`) and its tests.

What stays:

- `GraphEmbeddingDataset` and `compute_validation_metrics`;
- the `graph_dataset` conversion case that `tests/unit/test_graph_downstream_evaluation.py:190`
  parametrizes;
- the pairwise-distance test, which now tests `lorentz_distance_matrix` directly instead of
  through the suite.

**Files:**

- Modify: `src/naics_embedder/graph_model/__init__.py` (drop the QCEW and suite re-exports)
- Modify: `src/naics_embedder/metrics/__init__.py` (drop the QCEW and suite exports)
- Modify: `src/naics_embedder/metrics/graph.py` (keeps `GraphEmbeddingDataset` and
  `compute_validation_metrics` only)
- Delete: `src/naics_embedder/metrics/qcew.py`
- Modify: `tests/unit/test_graph_downstream_evaluation.py` (keep the `graph_dataset` conversion
  case; add the removal test)
- Modify: `tests/unit/test_graph_pairwise_distances.py` (test `lorentz_distance_matrix` directly)
- Delete: `tests/unit/test_qcew_multilevel.py`
- Modify: `docs/.nav.yml` (the QCEW page's entry)
- Delete: `docs/api/qcew_metrics.md`
- Modify: `WARP.md` (the metrics package's description)

**Interfaces:**

- Consumes: `lorentz_distance_matrix(embeddings, curvature=1.0)` (existing,
  `naics_embedder.metrics.core`), which the pairwise-distance test now calls directly.
- Produces: `naics_embedder.metrics` and `naics_embedder.graph_model` no longer export
  `QCEWBenchmarkConfig`, `run_qcew_employment_benchmark`, `GraphDownstreamEvaluator` or
  `run_graph_downstream_suite`, and `naics_embedder.metrics.qcew` does not exist.

- [x] **Step 1: Write the failing test and remove the benchmark's tests**

Replace the whole of `tests/unit/test_graph_downstream_evaluation.py` with exactly this content:

```python
'''
``GraphEmbeddingDataset`` reads a Polars embeddings frame into torch without handing torch
read-only memory, whatever the columns' layout.
'''

import importlib
import warnings

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.metrics import GraphEmbeddingDataset

def _lorentz_points(spatial):
    tensor = torch.tensor(spatial, dtype=torch.float32)
    time = torch.sqrt(1.0 + torch.sum(tensor**2, dim=1, keepdim=True))
    return torch.cat([time, tensor], dim=1)

def _graph_fixture():
    codes = ['11111', '111110', '111111', '21111', '211110', '211111']
    levels = [5, 6, 6, 5, 6, 6]
    spatial = [
        [0.0, 0.0],
        [0.08, 0.02],
        [0.09, -0.02],
        [1.2, 0.0],
        [1.28, 0.02],
        [1.30, -0.02],
    ]
    embeddings = _lorentz_points(spatial)
    return GraphEmbeddingDataset(embeddings=embeddings, codes=codes, levels=levels)

# -------------------------------------------------------------------------------------------------
# Polars-to-torch embedding conversion
# -------------------------------------------------------------------------------------------------

EMBED_COLS = ['hgcn_e0', 'hgcn_e1', 'hgcn_e2']
NOT_WRITABLE_WARNING = 'The given NumPy array is not writable'

def _embedding_frame(*, contiguous: bool) -> pl.DataFrame:
    '''Build the fixture's embeddings frame with its hgcn_e* columns in one Fortran buffer.

    Polars' to_numpy() returns a read-only zero-copy view when the selected columns sit
    back-to-back in memory. After a join or parquet read that is up to the allocator; here it is
    fixed. contiguous=False stores the columns in reverse, so selecting them in numeric order
    always takes the copying path instead.
    '''
    dataset = _graph_fixture()
    values = dataset.embeddings.double().numpy()
    columns = EMBED_COLS
    if not contiguous:
        values, columns = values[:, ::-1], EMBED_COLS[::-1]
    embeddings = pl.DataFrame(np.asfortranarray(values), schema=columns, orient='row')
    frame = pl.DataFrame({'code': dataset.codes, 'level': dataset.levels}).hstack(embeddings)

    read_only = not frame.select(EMBED_COLS).to_numpy().flags.writeable
    assert read_only == contiguous, 'fixture no longer controls whether to_numpy() copies'
    return frame

@pytest.fixture
def torch_warns_always():
    '''Re-arm torch warnings that otherwise fire at most once per process.

    The not-writable warning is one, so an earlier test in the same process could trip it and
    leave the test below unable to see it.
    '''
    previous = torch.is_warn_always_enabled()
    torch.set_warn_always(True)
    yield
    torch.set_warn_always(previous)

@pytest.mark.usefixtures('torch_warns_always')
def test_embedding_conversion_never_hands_torch_read_only_memory():
    frame = _embedding_frame(contiguous=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message=NOT_WRITABLE_WARNING, category=UserWarning)
        GraphEmbeddingDataset.from_dataframe(frame)

def test_graph_dataset_values_do_not_depend_on_column_contiguity():
    contiguous = GraphEmbeddingDataset.from_dataframe(_embedding_frame(contiguous=True))
    split = GraphEmbeddingDataset.from_dataframe(_embedding_frame(contiguous=False))

    assert torch.equal(contiguous.embeddings, split.embeddings)

# -------------------------------------------------------------------------------------------------
# The removed benchmark and suite
# -------------------------------------------------------------------------------------------------

def test_the_qcew_benchmark_and_the_downstream_suite_are_gone():
    '''Roadmap Stage 4: neither ``metrics/qcew.py`` nor the taxonomy-tasks suite remains.'''

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('naics_embedder.metrics.qcew')
    modules = [
        importlib.import_module(name) for name in (
            'naics_embedder.metrics', 'naics_embedder.metrics.graph', 'naics_embedder.graph_model'
        )
    ]
    for name in (
        'GraphDownstreamEvaluator',
        'run_graph_downstream_suite',
        'QCEWBenchmarkConfig',
        'run_qcew_employment_benchmark',
    ):
        assert not any(hasattr(module, name) for module in modules), name
```

Modify `tests/unit/test_graph_pairwise_distances.py` with these 4 edits, in order.

**`tests/unit/test_graph_pairwise_distances.py`, edit 1 of 4.** Replace:

```python
'''Precision and memory of GraphDownstreamEvaluator's pairwise Lorentz distances.

```

with:

```python
'''Precision and memory of the pairwise Lorentz distances of ``lorentz_distance_matrix``.

```

**`tests/unit/test_graph_pairwise_distances.py`, edit 2 of 4.** Replace:

```python

from naics_embedder.metrics import GraphDownstreamEvaluator, GraphEmbeddingDataset
```

with:

```python

from naics_embedder.metrics.core import lorentz_distance_matrix
```

**`tests/unit/test_graph_pairwise_distances.py`, edit 3 of 4.** Replace:

```python

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

def _evaluator(embeddings: torch.Tensor) -> GraphDownstreamEvaluator:
    count = embeddings.size(0)
    codes = [str(i) for i in range(count)]
    dataset = GraphEmbeddingDataset(embeddings=embeddings, codes=codes, levels=[6] * count)
    return GraphDownstreamEvaluator(dataset)

def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    return _evaluator(embeddings)._pairwise_distances()
```

with:

```python

def _pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    return lorentz_distance_matrix(embeddings)
```

**`tests/unit/test_graph_pairwise_distances.py`, edit 4 of 4.** Replace:

```python
    count, spatial_dim = 24, 40
    evaluator = _evaluator(hyperboloid_points(count, seed=4, spatial_dim=spatial_dim).float())

    with LargestStorage() as largest:
        evaluator._pairwise_distances()
```

with:

```python
    count, spatial_dim = 24, 40
    points = hyperboloid_points(count, seed=4, spatial_dim=spatial_dim).float()

    with LargestStorage() as largest:
        _pairwise_distances(points)
```

Delete this file with `git rm`:

```bash
git rm tests/unit/test_qcew_multilevel.py
```

- [x] **Step 2: Run the tests to verify the removal test fails**

Run: `uv run pytest tests/unit/test_graph_downstream_evaluation.py tests/unit/test_graph_pairwise_distances.py -q`
Expected: `1 failed, 6 passed`. The failure is
`test_the_qcew_benchmark_and_the_downstream_suite_are_gone`, with
`Failed: DID NOT RAISE <class 'ModuleNotFoundError'>`.

- [x] **Step 3: Remove the benchmark and the suite**

Modify `src/naics_embedder/graph_model/__init__.py` with these 2 edits, in order.

**`src/naics_embedder/graph_model/__init__.py`, edit 1 of 2.** Replace:

```python
from naics_embedder.metrics import (
    GraphDownstreamEvaluator,
    GraphEmbeddingDataset,
    QCEWBenchmarkConfig,
    compute_validation_metrics,
    run_graph_downstream_suite,
    run_qcew_employment_benchmark,
```

with:

```python
from naics_embedder.metrics import (
    GraphEmbeddingDataset,
    compute_validation_metrics,
```

**`src/naics_embedder/graph_model/__init__.py`, edit 2 of 2.** Replace:

```python
    'GraphEmbeddingDataset',
    'GraphDownstreamEvaluator',
    'run_graph_downstream_suite',
    'QCEWBenchmarkConfig',
    'run_qcew_employment_benchmark',
```

with:

```python
    'GraphEmbeddingDataset',
```

Modify `src/naics_embedder/metrics/__init__.py` with these 4 edits, in order.

**`src/naics_embedder/metrics/__init__.py`, edit 1 of 4.** Replace:

```python
- Core metrics classes (EmbeddingEvaluator, HierarchyMetrics, etc.)
- Graph-specific metrics and downstream evaluation
- QCEW benchmark
```

with:

```python
- Core metrics classes (EmbeddingEvaluator, HierarchyMetrics, etc.)
- Graph-specific validation metrics and the graph embedding container
```

**`src/naics_embedder/metrics/__init__.py`, edit 2 of 4.** Replace:

```python
from .graph import (
    GraphDownstreamEvaluator,
    GraphEmbeddingDataset,
    compute_validation_metrics,
    run_graph_downstream_suite,
```

with:

```python
from .graph import (
    GraphEmbeddingDataset,
    compute_validation_metrics,
```

**`src/naics_embedder/metrics/__init__.py`, edit 3 of 4.** Replace:

```python

# QCEW benchmark
from .qcew import (
    QCEWBenchmarkConfig,
    QCEWMultilevelConfig,
    print_multilevel_comparison,
    run_qcew_employment_benchmark,
    run_qcew_multilevel_benchmark,
)

```

with:

```python

```

**`src/naics_embedder/metrics/__init__.py`, edit 4 of 4.** Replace:

```python
    # Graph
    'GraphDownstreamEvaluator',
    'GraphEmbeddingDataset',
    'compute_validation_metrics',
    'run_graph_downstream_suite',
    # Hierarchy structure
    'compute_hierarchy_retrieval_metrics',
    'compute_radius_structure_metrics',
    # QCEW
    'QCEWBenchmarkConfig',
    'QCEWMultilevelConfig',
    'print_multilevel_comparison',
    'run_qcew_employment_benchmark',
    'run_qcew_multilevel_benchmark',
```

with:

```python
    # Graph
    'GraphEmbeddingDataset',
    'compute_validation_metrics',
    # Hierarchy structure
    'compute_hierarchy_retrieval_metrics',
    'compute_radius_structure_metrics',
```

Replace the whole of `src/naics_embedder/metrics/graph.py` with exactly this content:

```python
# -------------------------------------------------------------------------------------------------
# Graph Model Metrics
# -------------------------------------------------------------------------------------------------
'''
Graph-specific metrics.

Contains:
- compute_validation_metrics: Triplet-based validation metrics for hyperbolic embeddings
- GraphEmbeddingDataset: Container for hyperbolic graph embeddings
'''

import logging
from dataclasses import dataclass
from typing import Dict, Sequence, Union

import polars as pl
import torch

from naics_embedder.utils.utilities import (
    STAGE3_EMBEDDING_PREFIX,
    STAGE4_EMBEDDING_PREFIX,
    sorted_embedding_columns,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Validation Metrics
# -------------------------------------------------------------------------------------------------

def _anchor_distances(
    emb: torch.Tensor,
    anchors: torch.Tensor,
    candidates: torch.Tensor,
    c: float,
) -> torch.Tensor:
    '''Float64 CPU Lorentz distances from each anchor to each of its candidates.

    As in lorentz_distance_matrix, x0 is re-derived in float64 from the spatial coordinates:
    float32 rounding of the stored x0 alone puts distances of about 1e-2 on pairs that are close
    far from the origin.

    Args:
        emb: Embeddings of shape ``(N, embedding_dim+1)``.
        anchors: Anchor indices, shape ``(batch_size,)``.
        candidates: Indices to measure from each anchor, shape ``(batch_size, m)``.
        c: Curvature parameter.

    Returns:
        Distances of shape ``(batch_size, m)``.
    '''
    # Move before casting, as MPS has no float64. One fixed layout makes the reductions round
    # identically for C- and Fortran-ordered inputs.
    points = emb.detach().cpu().to(torch.float64).contiguous()
    spatial = points[:, 1:]
    anchors = anchors.cpu()
    candidates = candidates.cpu()
    # Squared norms and dot products share one reduction, so copies of a point at different
    # indices cancel as closely as the point does with itself.
    time = torch.sqrt(1.0 / c + (spatial * spatial).sum(dim=1))
    dot = (spatial[candidates] * spatial[anchors].unsqueeze(1)).sum(dim=-1)
    neg_dot = time[candidates] * time[anchors].unsqueeze(1) - dot
    return c**0.5 * torch.acosh(torch.clamp(neg_dot, min=1.0))

def compute_validation_metrics(
    emb: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    c: float = 1.0,
    top_k: int = 1,
    *,
    as_tensors: bool = False,
) -> Union[Dict[str, float], Dict[str, torch.Tensor]]:
    '''Compute validation metrics for hyperbolic embeddings.

    Distances are computed in float64 on the CPU (see ``_anchor_distances``). Like every distance
    formula here, sqrt(c) * acosh(-<x, y>_L) is only correct for c = 1.

    Args:
        emb: Embeddings tensor of shape ``(N, embedding_dim+1)``.
        anchors: Anchor indices, shape ``(batch_size,)``.
        positives: Positive indices, shape ``(batch_size,)``.
        negatives: Negative indices, shape ``(batch_size, k_negatives)``.
        c: Curvature parameter (default: 1.0).
        top_k: Number of top negatives to consider for auxiliary accuracy.
        as_tensors: Return torch scalars instead of Python floats (for Lightning logging).

    Returns:
        Mapping of metric names to values: float32 tensors on ``emb``'s device, or floats.
    '''
    k_negatives = negatives.size(1)
    effective_top_k = max(1, min(top_k, k_negatives))

    # Column 0 holds each anchor's distance to its positive, the rest to its negatives.
    candidates = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    all_dists_per_anchor = _anchor_distances(emb, anchors, candidates, c)
    positive_dist = all_dists_per_anchor[:, 0]
    negative_dist = all_dists_per_anchor[:, 1:]

    avg_positive_dist = positive_dist.mean()
    avg_negative_dist = negative_dist.mean()

    all_distances = torch.cat([positive_dist, negative_dist.reshape(-1)], dim=0)
    distance_spread = torch.div(all_distances.std(), all_distances.mean().clamp_min(1e-8))

    relation_accuracy = (positive_dist.unsqueeze(1) < negative_dist).all(dim=1).float().mean()

    closest_negatives = torch.topk(negative_dist, k=effective_top_k, dim=1, largest=False).values
    top_k_relation_accuracy = (positive_dist.unsqueeze(1)
                               < closest_negatives).all(dim=1).float().mean()

    order = torch.argsort(all_dists_per_anchor, dim=1)
    positive_rank_tensor = torch.argmax((order == 0).int(), dim=1)
    mean_positive_rank = positive_rank_tensor.float().mean()

    metrics = {
        'avg_positive_dist': avg_positive_dist,
        'avg_negative_dist': avg_negative_dist,
        'distance_spread': distance_spread,
        'relation_accuracy': relation_accuracy,
        'top_k_relation_accuracy': top_k_relation_accuracy,
        'mean_positive_rank': mean_positive_rank,
    }

    if as_tensors:
        # Lightning logs these from the model's device.
        return {k: v.to(device=emb.device, dtype=torch.float32) for k, v in metrics.items()}

    return {k: float(v) for k, v in metrics.items()}

# -------------------------------------------------------------------------------------------------
# Graph embeddings
# -------------------------------------------------------------------------------------------------

@dataclass
class GraphEmbeddingDataset:
    '''Container for a set of hyperbolic graph embeddings.'''

    embeddings: torch.Tensor
    codes: Sequence[str]
    levels: Sequence[int]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError('embeddings tensor must be 2D')

        num_nodes = self.embeddings.size(0)
        if num_nodes != len(self.codes) or num_nodes != len(self.levels):
            raise ValueError(
                'Embeddings, codes, and levels must have the same first dimension '
                f'(got embeddings={num_nodes}, codes={len(self.codes)}, levels={len(self.levels)})'
            )

        self.codes = list(self.codes)
        self.levels = [int(level) for level in self.levels]

    @classmethod
    def from_dataframe(
        cls,
        frame: pl.DataFrame,
        *,
        embedding_prefix: str = STAGE4_EMBEDDING_PREFIX,
        code_column: str = 'code',
        level_column: str = 'level',
    ) -> 'GraphEmbeddingDataset':
        '''Build a dataset from a parquet dataframe.'''

        if code_column not in frame.columns:
            raise ValueError(f'Expected column "{code_column}" in embeddings parquet')
        if level_column not in frame.columns:
            raise ValueError(f'Expected column "{level_column}" in embeddings parquet')

        embed_cols = sorted_embedding_columns(frame.columns, embedding_prefix)
        if not embed_cols:
            raise ValueError(
                f'No embedding columns found with prefix "{embedding_prefix}". '
                f'Set embedding_prefix to match {STAGE4_EMBEDDING_PREFIX}* or '
                f'{STAGE3_EMBEDDING_PREFIX}* columns.'
            )

        # to_numpy() returns a read-only view when the columns happen to sit back-to-back in
        # memory. torch.tensor copies either way and keeps the Fortran layout for both, whereas
        # to_numpy(writable=True) would copy only views to C order.
        tensor = torch.tensor(frame.select(embed_cols).to_numpy(), dtype=torch.float32)
        codes = frame.get_column(code_column).to_list()
        levels = frame.get_column(level_column).to_list()

        return cls(embeddings=tensor, codes=codes, levels=levels)
```

Modify `docs/.nav.yml` with one edit. Replace:

```yaml
          - Diagnostics: api/diagnostics.md
          - QCEW: api/qcew_metrics.md
```

with:

```yaml
          - Diagnostics: api/diagnostics.md
```

Modify `WARP.md` with one edit. Replace:

```markdown
- `src/naics_embedder/metrics/`  
  Shared metric implementations for both text and graph models (hierarchy metrics, downstream graph evaluation, QCEW benchmark, etc.).
```

with:

```markdown
- `src/naics_embedder/metrics/`  
  Shared metric implementations for both text and graph models (hierarchy metrics, graph validation metrics, and `diagnostics.py`: Req 6's structural diagnostics).
```

Delete these files with `git rm`:

```bash
git rm src/naics_embedder/metrics/qcew.py
git rm docs/api/qcew_metrics.md
```

- [x] **Step 4: Run the tests to verify they pass, and build the docs**

Run: `uv run pytest tests/unit/test_graph_downstream_evaluation.py tests/unit/test_graph_pairwise_distances.py -q`
Expected: `7 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1607 passed, 1 skipped`. The count drops by 8: this task adds one test and deletes
nine with the benchmark and the suite.

Run: `git grep -n -E 'qcew_metrics|metrics\.qcew|metrics/qcew|GraphDownstreamEvaluator|run_graph_downstream_suite|QCEWBenchmarkConfig|run_qcew_employment_benchmark' -- src tests docs conf mkdocs.yml README.md WARP.md CLAUDE.md`
Expected: only the removal test's own lines in
`tests/unit/test_graph_downstream_evaluation.py`.

Run: `uv run mkdocs build --strict -q -d /tmp/stage4-docs-ec267a03`, then
`rm -rf /tmp/stage4-docs-ec267a03`
Expected: no output and exit 0. A stale `docs/.nav.yml` entry would fail this build.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/graph_model/__init__.py src/naics_embedder/metrics/__init__.py src/naics_embedder/metrics/graph.py tests/unit/test_graph_downstream_evaluation.py tests/unit/test_graph_pairwise_distances.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

`git rm` in Steps 1 and 3 already staged the deletions.

```bash
git add src/naics_embedder/graph_model/__init__.py \
  src/naics_embedder/metrics/__init__.py \
  src/naics_embedder/metrics/graph.py \
  tests/unit/test_graph_downstream_evaluation.py \
  tests/unit/test_graph_pairwise_distances.py \
  docs/.nav.yml \
  WARP.md
git commit -m "refactor(metrics): remove the QCEW benchmark and the taxonomy-tasks suite"
```

### Task 11: Remove `verify-stage4` and its fixed thresholds

Req 5 replaces fixed acceptance thresholds, and user decision 2 deletes the command that applied
them. Its module, tests, API page and doc sections go with it.

- `README.md` section 5.5 becomes "Diagnostics and Decisions".
- `docs/hgcn_training.md` section 9 becomes "Diagnostics and the Keep-or-Drop Decision".
- `WARP.md` lists `tools diagnostics` in place of `verify-stage4`.

**Files:**

- Modify: `src/naics_embedder/cli/commands/tools.py` (remove `verify-stage4`)
- Delete: `src/naics_embedder/tools/embeddings_verification.py`
- Modify: `tests/unit/test_cli_commands.py`
- Delete: `tests/unit/test_embeddings_verification.py`
- Modify: `docs/.nav.yml` (the verification page's entry)
- Delete: `docs/api/embeddings_verification.md`
- Modify: `docs/hgcn_training.md` (section 9)
- Modify: `docs/overview.md` (two verifier sentences)
- Modify: `README.md` (section 5.5)
- Modify: `WARP.md` (the tools list, the verification commands, the HGCN flow)

**Interfaces:**

- Consumes: `tools diagnostics` (Task 9), `tools margins` and `tools decide` (Task 8), which the
  rewritten doc sections point to.
- Produces: `naics-embedder tools verify-stage4` is gone, and so is
  `naics_embedder.tools.embeddings_verification`.

- [x] **Step 1: Write the failing test and remove the verifier's tests**

Modify `tests/unit/test_cli_commands.py` with these 3 edits, in order.

**`tests/unit/test_cli_commands.py`, edit 1 of 3.** Replace:

```python
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.metrics.diagnostics import DiagnosticsReport
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle
```

with:

```python
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.metrics.diagnostics import DiagnosticsReport
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
```

**`tests/unit/test_cli_commands.py`, edit 2 of 3.** Replace:

```python

def test_verify_stage4_failure_sets_exit_code(monkeypatch, runner):

    def fake_verify(*_args, **_kwargs):
        return {
            'pre': {
                'metric': 0.8
            },
            'post': {
                'metric': 0.7
            },
            'delta': {
                'metric': -0.1
            },
            'checks': {
                'cophenetic': False
            },
            'passed': False,
        }

    monkeypatch.setattr(tools_cli, 'verify_stage4', fake_verify)

    result = runner.invoke(tools_cli.app, ['verify-stage4'])

    assert result.exit_code == 1
    assert 'Verification failed' in result.output or 'failed thresholds' in result.output

@pytest.mark.unit
@pytest.mark.parametrize('undefined', [False, True])
def test_verify_stage4_formats_versioned_spearman(monkeypatch, runner, undefined):
    key = 'structural_spearman_v1'
    value = None if undefined else 0.87831006565368
    delta = None if undefined else 0.125
    payload = {
        'pre': {
            key: value
        },
        'post': {
            key: value
        },
        'delta': {
            key: delta
        },
        'checks': {
            'cophenetic': True,
            'ndcg': True,
            'local_improvement': True
        },
        'passed': True,
    }
    monkeypatch.setattr(tools_cli, 'verify_stage4', lambda *_, **__: payload)
    result = runner.invoke(tools_cli.app, ['verify-stage4'])
    assert result.exit_code == 0, result.output
    if undefined:
        assert result.output.count(f'{key}: N/A') == 3
    else:
        assert result.output.count(f'{key}: 0.8783') == 2
        assert f'{key}: +0.1250' in result.output
    assert 'spearman_correlation' not in result.output

@pytest.mark.unit
def test_verify_stage4_input_error_is_fatal(monkeypatch, runner):

    def invalid(*_args, **_kwargs):
        raise StructuralMetricInputError('structural-spearman-v1: invalid tree_distances')

    monkeypatch.setattr(tools_cli, 'verify_stage4', invalid)
    result = runner.invoke(tools_cli.app, ['verify-stage4'])
    assert result.exit_code == 1
    assert 'Verification failed' in result.output
    assert 'invalid tree_distances' in result.output

@pytest.mark.unit
def test_verify_stage4_has_no_spearman_threshold_option(runner):
    result = runner.invoke(tools_cli.app, ['verify-stage4', '--help'])
    assert result.exit_code == 0
    assert '--max-spearman-drop' not in result.output
    assert '--min-spearman' not in result.output

@pytest.fixture
def verify_inputs(monkeypatch):
    '''The structural input paths verify-stage4 hands to verify_stage4.'''
    seen = {}

    def fake_verify(_stage3, _stage4, distance_matrix, relations, _cfg):
        seen.update(distance_matrix=distance_matrix, relations=relations)
        return {'pre': {}, 'post': {}, 'delta': {}, 'checks': {}, 'passed': True}

    monkeypatch.setattr(tools_cli, 'verify_stage4', fake_verify)
    return seen

@pytest.mark.unit
def test_verify_stage4_defaults_to_the_legacy_structural_files(runner, verify_inputs):
    result = runner.invoke(tools_cli.app, ['verify-stage4'])

    assert result.exit_code == 0, result.output
    assert verify_inputs == {
        'distance_matrix': Path('./data/naics_distance_matrix.parquet'),
        'relations': Path('./data/naics_relations.parquet'),
    }

@pytest.mark.unit
def test_verify_stage4_reads_structure_from_its_supervision_bundle(
    runner, verify_inputs, generated_bundle
):
    result = runner.invoke(
        tools_cli.app, ['verify-stage4', '--supervision-manifest',
                        str(generated_bundle)]
    )

    bundle = load_validated_bundle(generated_bundle)
    assert result.exit_code == 0, result.output
    assert verify_inputs == {
        'distance_matrix': bundle.artifact_path('distance_matrix'),
        'relations': bundle.artifact_path('relations'),
    }

@pytest.mark.unit
def test_verify_stage4_rejects_relations_from_outside_its_bundle(
    runner, verify_inputs, generated_bundle, tmp_path
):
    result = runner.invoke(
        tools_cli.app,
        [
            'verify-stage4',
            '--supervision-manifest',
            str(generated_bundle),
            '--relations',
            str(tmp_path / 'naics_relations.parquet'),
        ],
    )

    assert result.exit_code == 1
    assert 'relations path does not belong' in result.output
    assert verify_inputs == {}

```

with:

```python

```

**`tests/unit/test_cli_commands.py`, edit 3 of 3.** Replace:

```python
    assert 'Diagnostics failed' in result.output

```

with:

```python
    assert 'Diagnostics failed' in result.output

@pytest.mark.unit
def test_verify_stage4_is_gone(runner):
    result = runner.invoke(tools_cli.app, ['verify-stage4'])

    assert result.exit_code != 0
    assert 'No such command' in result.output

```

Delete this file with `git rm`:

```bash
git rm tests/unit/test_embeddings_verification.py
```

- [x] **Step 2: Run the tests to verify the removal test fails**

Run: `uv run pytest tests/unit/test_cli_commands.py -q`
Expected: `1 failed, 30 passed`. The failure is `test_verify_stage4_is_gone`, with
`AssertionError: assert 'No such command' in 'Verification failed: Stage 3 embeddings not found: …'`.

- [x] **Step 3: Remove the command and rewrite its doc sections**

> Deviation: the task review found that `WARP.md:27` still said `tools` included "Stage 4 verification"; it now reads "(including Req 6 diagnostics and Req 5 decisions)" (d31c8e5).

Modify `src/naics_embedder/cli/commands/tools.py` with these 4 edits, in order.

**`src/naics_embedder/cli/commands/tools.py`, edit 1 of 4.** Replace:

```python
from pathlib import Path
from typing import List, Optional, Tuple
```

with:

```python
from pathlib import Path
from typing import List, Optional
```

**`src/naics_embedder/cli/commands/tools.py`, edit 2 of 4.** Replace:

```python
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.graph_model.curriculum.preprocess_curriculum import (
    resolve_graph_supervision_paths,
)
```

with:

```python
from naics_embedder.decision.store import ArtifactStore
```

**`src/naics_embedder/cli/commands/tools.py`, edit 3 of 4.** Replace:

```python
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.embeddings_verification import Stage4VerificationConfig, verify_stage4
```

with:

```python
from naics_embedder.tools.config_tools import show_current_config
```

**`src/naics_embedder/cli/commands/tools.py`, edit 4 of 4.** Replace:

```python

# -------------------------------------------------------------------------------------------------
# Verify Stage 4 against Stage 3
# -------------------------------------------------------------------------------------------------

def _stage4_structural_inputs(
    supervision_manifest: Optional[str],
    distance_matrix: Optional[str],
    relations_parquet: Optional[str],
) -> Tuple[Path, Path]:
    '''
    The distance matrix and relations parquet that verify-stage4 reads.

    With a supervision manifest both come from that one validated bundle (an explicitly supplied
    path must be the bundle's own artifact). Without one, unset paths fall back to the legacy
    ``./data`` files.
    '''
    if supervision_manifest:
        paths = resolve_graph_supervision_paths(
            supervision_manifest,
            distance_matrix_path=distance_matrix,
            relations_path=relations_parquet,
        )
        return paths.distance_matrix, paths.relations
    return (
        Path(distance_matrix or './data/naics_distance_matrix.parquet'),
        Path(relations_parquet or './data/naics_relations.parquet'),
    )

@app.command('verify-stage4')
def verify_stage4_command(
    stage3_parquet: Annotated[
        str,
        typer.Option(
            '--pre',
            help='Path to Stage 3 (pre-HGCN) embeddings parquet',
        ),
    ] = './output/hyperbolic_projection/encodings.parquet',
    stage4_parquet: Annotated[
        str,
        typer.Option(
            '--post',
            help='Path to Stage 4 (HGCN) embeddings parquet',
        ),
    ] = './output/hgcn/encodings.parquet',
    distance_matrix: Annotated[
        Optional[str],
        typer.Option(
            '--distance-matrix',
            help=(
                'Path to ground truth distance matrix parquet (default: the bundle artifact with '
                '--supervision-manifest, else ./data/naics_distance_matrix.parquet)'
            ),
        ),
    ] = None,
    relations_parquet: Annotated[
        Optional[str],
        typer.Option(
            '--relations',
            help=(
                'Path to relations parquet, used for the parent retrieval metric (default: the '
                'bundle artifact with --supervision-manifest, else ./data/naics_relations.parquet)'
            ),
        ),
    ] = None,
    supervision_manifest: Annotated[
        Optional[str],
        typer.Option(
            '--supervision-manifest',
            help='Supervision bundle manifest; the distance matrix and relations come from it',
        ),
    ] = None,
    max_cophenetic_drop: Annotated[
        float,
        typer.Option('--max-cophenetic-drop', help='Allowed drop in cophenetic correlation'),
    ] = 0.02,
    max_ndcg_drop: Annotated[
        float,
        typer.Option('--max-ndcg-drop', help='Allowed drop in NDCG@10'),
    ] = 0.01,
    min_local_improvement: Annotated[
        float,
        typer.Option('--min-local-improvement', help='Required parent retrieval improvement'),
    ] = 0.05,
    ndcg_k: Annotated[
        int,
        typer.Option('--ndcg-k', help='NDCG@K to evaluate'),
    ] = 10,
    parent_top_k: Annotated[
        int,
        typer.Option('--parent-top-k', help='Top-K used for parent retrieval accuracy'),
    ] = 1,
):
    '''
    Compare Stage 3 and Stage 4 embeddings at curvature 1.0.

    Enforce cophenetic, NDCG, and parent-retrieval thresholds. Report structural
    Spearman v1 separately; undefined values and deltas display as N/A.
    '''

    configure_logging('tools_verify_stage4.log')

    cfg = Stage4VerificationConfig(
        max_cophenetic_degradation=max_cophenetic_drop,
        max_ndcg_degradation=max_ndcg_drop,
        min_local_improvement=min_local_improvement,
        ndcg_k=ndcg_k,
        parent_top_k=parent_top_k,
    )

    try:
        distance_matrix_path, relations_path = _stage4_structural_inputs(
            supervision_manifest, distance_matrix, relations_parquet
        )
        result = verify_stage4(
            Path(stage3_parquet),
            Path(stage4_parquet),
            distance_matrix_path,
            relations_path,
            cfg,
        )
    except Exception as exc:
        console.print(f'[bold red]Verification failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print('\n[bold cyan]Stage 4 Verification[/bold cyan]\n')
    console.print('[bold]Pre-HGCN metrics:[/bold]')
    for key, value in result['pre'].items():
        formatted = 'N/A' if value is None else f'{value:.4f}'
        console.print(f'  • {key}: {formatted}')

    console.print('\n[bold]Post-HGCN metrics:[/bold]')
    for key, value in result['post'].items():
        formatted = 'N/A' if value is None else f'{value:.4f}'
        console.print(f'  • {key}: {formatted}')

    console.print('\n[bold]Deltas:[/bold]')
    for key, value in result['delta'].items():
        formatted = 'N/A' if value is None else f'{value:+.4f}'
        console.print(f'  • {key}: {formatted}')

    console.print('\n[bold]Threshold checks:[/bold]')
    for key, passed in result['checks'].items():
        status = '[green]PASS[/green]' if passed else '[red]FAIL[/red]'
        console.print(f'  • {key}: {status}')

    if result['passed']:
        console.print('\n[bold green]✓ Stage 4 verification passed![/bold green]\n')
    else:
        console.print('\n[bold red]✗ Stage 4 verification failed thresholds[/bold red]\n')
        raise typer.Exit(code=1)

```

with:

```python

```

Modify `docs/.nav.yml` with one edit. Replace:

```yaml
          - Training Metrics: api/metrics_tools.md
          - Embeddings Verification: api/embeddings_verification.md
```

with:

```yaml
          - Training Metrics: api/metrics_tools.md
```

Modify `docs/hgcn_training.md` with one edit. Replace:

````markdown

## 9. Pre/Post Verification Workflow

After both Stage 3 and Stage 4 finish, run the automated comparison from [Issue #67](https://github.com/lowmason/naics-embedder/issues/67) to confirm that HGCN preserved the Stage 3 geometry:

```bash
uv run naics-embedder tools verify-stage4 \
  --pre ./output/hyperbolic_projection/encodings.parquet \
  --post ./output/hgcn/encodings.parquet \
  --supervision-manifest data/supervision/stage3-supervision-v1/<bundle-id>/manifest.json
```

Additional options let you override the distance matrix, relations parquet, or the acceptable degradation thresholds:

| Option | Purpose |
| --- | --- |
| `--supervision-manifest` | Read the distance matrix and relations from this validated bundle; an explicit `--distance-matrix` or `--relations` from another source is rejected. |
| `--distance-matrix`, `--relations` | Without a manifest, default to the legacy `./data/naics_distance_matrix.parquet` and `./data/naics_relations.parquet`. |
| `--max-cophenetic-drop` | Maximum allowable decrease in cophenetic correlation (default `0.02`). |
| `--max-ndcg-drop` | Maximum allowable decrease in NDCG@K (default `0.01`). |
| `--min-local-improvement` | Required increase in parent retrieval accuracy (default `0.05`). |
| `--ndcg-k` | Which `K` to evaluate for NDCG (default `10`). |
| `--parent-top-k` | Size of the neighborhood used for parent retrieval (default `1`). |

The command prints pre/post metrics, deltas, and PASS/FAIL indicators for each threshold. Integrate it into CI to prevent regressions before shipping updated embeddings.

### Structural Spearman reporting

The verifier adds `structural_spearman_v1` to `pre`, `post`, and `delta`. A delta is computed
only if both phase values are defined. Otherwise it is `null` in the Python/JSON report and
`N/A` in the CLI. Definition, statuses, reasons, and pair counts are recorded separately:

```json
{
  "pre": {"structural_spearman_v1": 0.8783101},
  "post": {"structural_spearman_v1": null},
  "delta": {"structural_spearman_v1": null},
  "metric_metadata": {
    "structural_spearman_v1": {
      "definition": "structural-spearman-v1",
      "pre": {
        "status": "defined", "reason": null, "n_pairs": 6, "n_total": 6
      },
      "post": {
        "status": "undefined", "reason": "constant_prediction", "n_pairs": 6, "n_total": 6
      }
    }
  }
}
```

This excerpt omits the existing non-Spearman metrics, checks, thresholds, pass/fail flag, and
code list; those remain in the full report. Structural Spearman does not add a fourth check.
Only cophenetic degradation, NDCG degradation, and local parent-retrieval improvement govern
acceptance, using the same options and defaults as before.

Malformed inputs raise `StructuralMetricInputError`; the CLI prints the failure and exits
nonzero instead of printing a partial success report. Undefined correlation alone is
non-fatal. Both phase evaluations stay explicitly fixed at curvature `1.0`, regardless of
training configuration. The rank correction does not fix non-unit-curvature geometry.
````

with:

````markdown

## 9. Diagnostics and the Keep-or-Drop Decision

Report Req 6's structural diagnostics on a table before and after refinement:

```bash
uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic \
  --codebook data/supervision/stage3-supervision-v1/<bundle-id>/naics_codebook.parquet
```

`--table` takes a 2,125-code table in the export form: tangent coordinates at the origin for a
hyperbolic arm. Lorentz points are refused. The report covers sector separation, within-sector
rank correlation, MAP over ancestors, NDCG with integer lowest-common-ancestor grades, the
Pearson correlation of distance with D*, and parent retrieval without the 522 unary pairs. It
has no thresholds and no pass/fail.

Whether the graph stage is kept is decided under Req 5's rule on the outcome and regressor
panels (`tools margins`, `tools decide`; see the [usage guide](usage.md#tools-decide)), never on
structural statistics.
````

Modify `docs/overview.md` with these 2 edits, in order.

**`docs/overview.md`, edit 1 of 2.** Replace:

```markdown
serialize undefined correlation as JSON `null`; Lightning omits that numeric scalar but logs
the pair counts. Stage-4 verification reports pre/post/delta values and metadata but does not
gate acceptance on Spearman.
```

with:

```markdown
serialize undefined correlation as JSON `null`; Lightning omits that numeric scalar but logs
the pair counts.
```

**`docs/overview.md`, edit 2 of 2.** Replace:

```markdown
This rank repair does not validate the formula that produced the distance matrices. HGCN full
evaluation and the Stage-4 verifier remain fixed at curvature `1.0`; text comparison runs retain
```

with:

```markdown
This rank repair does not validate the formula that produced the distance matrices. HGCN full
evaluation remains fixed at curvature `1.0`; text comparison runs retain
```

Modify `README.md` with one edit. Replace:

````markdown

### 5.5 Pre/Post Verification

To ensure the refinement step preserves global structure while improving local parent retrieval, use the following command:

``` bash
uv run naics-embedder tools verify-stage4 \
  --pre ./output/hyperbolic_projection/encodings.parquet \
  --post ./output/hgcn/encodings.parquet \
  --supervision-manifest data/supervision/stage3-supervision-v1/<bundle-id>/manifest.json
```

`--supervision-manifest` reads the distance matrix and relations from the validated bundle.
Without it, `--distance-matrix` and `--relations` default to the legacy
`./data/naics_distance_matrix.parquet` and `./data/naics_relations.parquet`, which
`data all` no longer writes.

The verifier reports cophenetic correlation, NDCG\@K, parent-retrieval accuracy, and
`structural_spearman_v1` pre/post/delta values at fixed curvature `1.0`. Only cophenetic, NDCG,
and local parent-retrieval checks determine pass/fail; structural Spearman is report-only, with
no threshold option. Undefined values or deltas display as `N/A` and serialize as JSON `null`.
Definition, status, reason, and pair counts live under
`metric_metadata['structural_spearman_v1']`; see the
[verification output contract](docs/hgcn_training.md#9-prepost-verification-workflow).
````

with:

````markdown

### 5.5 Diagnostics and Decisions

Report Req 6's structural diagnostics on any 2,125-code table in the export form (tangent
coordinates at the origin for a hyperbolic arm):

``` bash
uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic \
  --codebook data/supervision/stage3-supervision-v1/<bundle-id>/naics_codebook.parquet
```

The report covers sector separation, within-sector rank correlation, MAP over ancestors, NDCG
with integer lowest-common-ancestor grades, the Pearson correlation of distance with D*, and
parent retrieval without the 522 unary pairs. It has no thresholds and no pass/fail. Whether a
change is adopted, graph refinement included, is decided under Req 5's rule on the outcome and
regressor panels: `tools margins` fixes each panel's margin from a reference arm, and
`tools decide` compares the arms (see the [usage guide](docs/usage.md#tools-decide)).
````

Modify `WARP.md` with these 3 edits, in order.

**`WARP.md`, edit 1 of 3.** Replace:

```markdown
  - `metrics_tools.py` – metric visualization and investigation helpers.
  - `embeddings_verification.py` – logic for comparing Stage 3 vs Stage 4 embeddings.
```

with:

```markdown
  - `metrics_tools.py` – metric visualization and investigation helpers.
```

**`WARP.md`, edit 2 of 3.** Replace:

````markdown

Stage 4 (HGCN) verification:

```bash path=null start=null
uv run naics-embedder tools verify-stage4 \
  --pre ./output/hyperbolic_projection/encodings.parquet \
  --post ./output/hgcn/encodings.parquet \
  --supervision-manifest data/supervision/stage3-supervision-v1/<bundle-id>/manifest.json
```

`--supervision-manifest` reads the distance matrix and relations from the bundle; without it they default to the legacy `./data` files.

This command runs hierarchy-aware metrics pre/post HGCN (cophenetic correlation, NDCG@K, parent retrieval) and enforces degradation thresholds (`--max-cophenetic-drop`, `--max-ndcg-drop`, `--min-local-improvement`, `--parent-top-k`). It is the canonical way to gate Stage 4 changes.
````

with:

````markdown

Structural diagnostics and decisions:

```bash path=null start=null
uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic \
  --codebook PATH/naics_codebook.parquet
uv run naics-embedder tools margins --reference reference.json --multiple 0.5 \
  --name reference-margins --store ~/naics-artifacts --output margins.json
uv run naics-embedder tools decide --arm candidate.json --arm reference.json \
  --margins margins.json --name dimension-8 --question "Is dimension 8 enough?" \
  --store ~/naics-artifacts --output decision.json
```

`tools diagnostics` reports Req 6's structural statistics for a table in the export form (tangent coordinates at the origin for a hyperbolic arm), with no thresholds and no pass/fail. `tools margins` fixes each panel's non-inferiority margin from a reference arm, and `tools decide` applies Req 5's rule over the outcome panel and the regressor panel's two regimes. Configurations, the graph stage included, are compared this way, never on structural statistics.
````

**`WARP.md`, edit 3 of 3.** Replace:

```markdown
3. Run HGCN training using `naics_embedder.graph_model.hgcn.main` (e.g., from a script or notebook) configured with a `GraphConfig` YAML.
4. Validate that Stage 4 did not degrade hierarchy metrics using `tools verify-stage4` (above).
```

with:

```markdown
3. Run HGCN training using `naics_embedder.graph_model.hgcn.main` (e.g., from a script or notebook) configured with a `GraphConfig` YAML.
4. Report each table's structural diagnostics with `tools diagnostics` (above). Whether the graph stage is kept is decided under Req 5 with `tools decide`, never on structural statistics.
```

Delete these files with `git rm`:

```bash
git rm src/naics_embedder/tools/embeddings_verification.py
git rm docs/api/embeddings_verification.md
```

- [x] **Step 4: Run the tests to verify they pass, and build the docs**

Run: `uv run pytest tests/unit/test_cli_commands.py -q`
Expected: `31 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1587 passed, 1 skipped`. The count drops by 20: this task adds one test and deletes
21 with the verifier.

Run: `git grep -n -E 'verify-stage4|verify_stage4|embeddings_verification' -- src tests docs conf mkdocs.yml README.md WARP.md CLAUDE.md`
Expected: only `test_verify_stage4_is_gone` in `tests/unit/test_cli_commands.py`.

Run: `uv run mkdocs build --strict -q -d /tmp/stage4-docs-ec267a03`, then
`rm -rf /tmp/stage4-docs-ec267a03`
Expected: no output and exit 0.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/cli/commands/tools.py tests/unit/test_cli_commands.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

`git rm` in Steps 1 and 3 already staged the deletions.

```bash
git add src/naics_embedder/cli/commands/tools.py \
  tests/unit/test_cli_commands.py \
  docs/.nav.yml \
  docs/hgcn_training.md \
  docs/overview.md \
  README.md \
  WARP.md
git commit -m "refactor(cli): remove verify-stage4 and its fixed thresholds"
```

### Task 12: Structural statistics off progress bars and headlines (Req 6)

This task takes the structural statistics off the progress bars and headlines (user decision
3). Their logged values stay for Stages 7 and 10/11.

- **Text stage.** `val/cophenetic_correlation` and `val/median_distortion` leave the progress
  bar.
- **HGCN.** `val/cophenetic_correlation`, `val/ndcg@10` and `val/relation_accuracy` leave the
  progress bar.
- **`tools visualize`.** Its plots, tables, grades and advice no longer read a structural
  statistic, and its summary table in `tools/metrics_tools.py` loses them too.
- **`train` banner.** It no longer announces cophenetic correlation, NDCG or distortion as the
  evaluation.
- **Docs.** `README.md`, `docs/hgcn_training.md`, `docs/overview.md`, `docs/usage.md` and
  `CLAUDE.md` describe the statistics as logged for the record only, with `tools diagnostics` as
  Req 6's report.

Nothing reads the statistics after this task: the text stage's monitors read
`val/contrastive_loss`, HGCN runs with `enable_checkpointing=False`, and the graph curriculum
controller is not wired into training.

**Files:**

- Modify: `src/naics_embedder/cli/commands/training.py` (the `train` banner)
- Modify: `src/naics_embedder/graph_model/hgcn.py` (no structural statistic on the bar)
- Modify: `src/naics_embedder/text_model/mixins/validation.py` (no structural statistic on the bar)
- Modify: `src/naics_embedder/tools/_visualize_metrics.py` (no structural statistic in the plots,
  tables, grades or advice)
- Modify: `src/naics_embedder/tools/metrics_tools.py` (the summary table)
- Modify: `tests/unit/test_cli_training.py`
- Modify: `tests/unit/test_hgcn_metrics.py`
- Modify: `tests/unit/test_metrics_tools_api.py`
- Modify: `tests/unit/test_text_validation_metrics.py`
- Modify: `tests/unit/test_visualize_metrics.py`
- Modify: `docs/hgcn_training.md` (section 8)
- Modify: `docs/overview.md` (section 11)
- Modify: `docs/usage.md` (`tools visualize`)
- Modify: `CLAUDE.md` (validation metrics and pitfall 2)
- Modify: `README.md` (section 5.4)

**Interfaces:**

- Consumes: `tools diagnostics` (Task 9), which the rewritten docs point to.
- Produces: no progress bar, visualize table, grade, advice line or `train` banner line names a
  structural statistic. The logged keys are unchanged.

- [x] **Step 1: Write the failing tests**

> Deviation: F5 (user ruling at pre-flight): `test_parse_leaves_out_the_structural_statistic` asserts `metrics` before its `all(...)`, so an empty parse cannot pass (5db2e49).

Modify `tests/unit/test_cli_training.py` with one edit. Replace:

```python
    assert any(isinstance(cb, TrainDatasetEpochCallback) for cb in callbacks)

```

with:

```python
    assert any(isinstance(cb, TrainDatasetEpochCallback) for cb in callbacks)

@pytest.mark.unit
def test_the_train_banner_headlines_no_structural_statistic(cli_runner, training_env):
    '''Req 6: the structural statistics are logged for the record, never announced as the
    evaluation.'''

    result = cli_runner.invoke(cli_app, ['train'], catch_exceptions=False)

    assert result.exit_code == 0
    output = result.output.replace('\n', '')
    for name in ('Cophenetic', 'NDCG', 'Distortion'):
        assert name not in output
    assert 'Structural statistics, for the record only' in output

```

Modify `tests/unit/test_hgcn_metrics.py` with one edit. Replace:

```python
@pytest.mark.unit
@pytest.mark.parametrize('invalid', ['nan_file', 'asymmetric', 'shape'])
```

with:

```python
@pytest.mark.unit
def test_hgcn_puts_no_structural_statistic_on_the_progress_bar(spearman_hgcn):
    # Req 6: structural statistics are reported, never a headline
    module, _ = spearman_hgcn
    module.on_validation_epoch_start()
    module.validation_step(
        {
            'anchor_idx': torch.tensor([0]),
            'positive_idx': torch.tensor([1]),
            'negative_indices': torch.tensor([[2, 3]])
        },
        batch_idx=0,
    )

    logged = [call.args[0] for call in module.log.call_args_list]
    assert 'val/cophenetic_correlation' in logged
    assert [call.args[0] for call in module.log.call_args_list if call.kwargs.get('prog_bar')] == []

@pytest.mark.unit
@pytest.mark.parametrize('invalid', ['nan_file', 'asymmetric', 'shape'])
```

Modify `tests/unit/test_metrics_tools_api.py` with one edit. Replace:

```python
        assert result['num_epochs'] == 2

```

with:

```python
        assert result['num_epochs'] == 2

    @pytest.mark.skipif(not HAS_VISUALIZE, reason='visualization tools not available')
    def test_visualize_metrics_tables_no_structural_statistic(self, sample_log_file, capsys):
        '''Req 6: the summary table carries no cophenetic or Spearman column.'''
        visualize_metrics(
            stage='test_stage',
            log_file=sample_log_file / 'logs' / 'train_sequential.log',
            output_dir=sample_log_file / 'output',
            project_root=sample_log_file,
        )

        output = capsys.readouterr().out
        assert 'METRICS SUMMARY TABLE' in output
        assert 'Cophenetic' not in output
        assert 'Spearman' not in output

```

Modify `tests/unit/test_text_validation_metrics.py` with one edit. Replace:

```python
@pytest.mark.parametrize('is_global_zero', [False, True])
def test_text_spearman_and_json_are_rank_zero_only(
```

with:

```python
def test_no_structural_statistic_reaches_the_progress_bar(
    tmp_path, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings
):
    # Req 6: structural statistics are reported, never a headline; the collapse checks stay
    prediction, target = structural_distance_matrices
    harness = ValidationHarness(tmp_path, target, structural_lorentz_embeddings)
    monkeypatch.setattr(
        harness.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )

    harness.on_validation_epoch_end()

    logged = [call.args[0] for call in harness.log.call_args_list]
    assert {'val/cophenetic_correlation', 'val/median_distortion'} <= set(logged)
    on_bar = [call.args[0] for call in harness.log.call_args_list if call.kwargs.get('prog_bar')]
    assert on_bar == ['val/norm_cv', 'val/distance_cv']

@pytest.mark.parametrize('is_global_zero', [False, True])
def test_text_spearman_and_json_are_rank_zero_only(
```

Modify `tests/unit/test_visualize_metrics.py` with these 3 edits, in order.

**`tests/unit/test_visualize_metrics.py`, edit 1 of 3.** Replace:

```python

    def test_parse_extracts_cophenetic_correlation(self, sample_log_file):
        '''Test extraction of cophenetic correlation and pair count.'''
        metrics = parse_log_file(sample_log_file)

        assert metrics[0]['cophenetic'] == pytest.approx(0.4521)
        assert metrics[0]['n_pairs'] == 500
        assert metrics[1]['cophenetic'] == pytest.approx(0.5678)
```

with:

```python

    def test_parse_leaves_out_the_structural_statistic(self, sample_log_file):
        '''Req 6: the cophenetic correlation the log still records is not read.'''
        metrics = parse_log_file(sample_log_file)

        assert all('cophenetic' not in m and 'n_pairs' not in m for m in metrics)
```

**`tests/unit/test_visualize_metrics.py`, edit 2 of 3.** Replace:

```python

    def test_print_analysis_shows_hierarchy_preservation(self, sample_log_file, capsys):
        '''Test that hierarchy preservation analysis is shown.'''
        metrics = parse_log_file(sample_log_file)

        print_analysis(metrics, 'test_stage')

        captured = capsys.readouterr()
        assert 'HIERARCHY PRESERVATION' in captured.out
```

with:

```python

    def test_print_analysis_grades_no_structural_statistic(self, sample_log_file, capsys):
        '''Req 6: no grade, trend or recommendation reads a structural statistic.'''
        metrics = parse_log_file(sample_log_file)
        for m in metrics:
            m['cophenetic'] = 0.1

        print_analysis(metrics, 'test_stage')

        captured = capsys.readouterr()
        assert 'HIERARCHY PRESERVATION' not in captured.out
        assert 'ophenetic' not in captured.out
```

**`tests/unit/test_visualize_metrics.py`, edit 3 of 3.** Replace:

```python
        assert metrics[0]['radius_mean'] == pytest.approx(999999.999)

    def test_parse_handles_negative_cophenetic(self, tmp_path):
        '''Test parsing negative cophenetic values.'''
        log_content = """
2024-01-15 10:00:00 - INFO - Running evaluation metrics (epoch 0)
2024-01-15 10:00:01 - INFO - Hyperbolic radius: 2.5 ± 0.1
2024-01-15 10:00:02 - INFO - Hierarchy preservation: cophenetic=-0.1234 (500 pairs)
"""
        log_file = tmp_path / 'negative_cophenetic.log'
        log_file.write_text(log_content)

        metrics = parse_log_file(log_file)

        assert metrics[0]['cophenetic'] == pytest.approx(-0.1234)
```

with:

```python
        assert metrics[0]['radius_mean'] == pytest.approx(999999.999)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_hgcn_metrics.py tests/unit/test_text_validation_metrics.py tests/unit/test_visualize_metrics.py tests/unit/test_metrics_tools_api.py tests/unit/test_cli_training.py -q`
Expected: `6 failed, 71 passed`. The six failures are:

- `test_hgcn_metrics.py::test_hgcn_puts_no_structural_statistic_on_the_progress_bar`
- `test_text_validation_metrics.py::test_no_structural_statistic_reaches_the_progress_bar`
- `test_visualize_metrics.py::TestParseLogFile::test_parse_leaves_out_the_structural_statistic`
- `test_visualize_metrics.py::TestPrintAnalysis::test_print_analysis_grades_no_structural_statistic`
- `test_metrics_tools_api.py::TestVisualizeMetrics::test_visualize_metrics_tables_no_structural_statistic`
- `test_cli_training.py::test_the_train_banner_headlines_no_structural_statistic`

- [x] **Step 3: Implement**

> Deviation: by user ruling after the final whole-branch review, `docs/overview.md`'s structural-statistics table lost its Ideal Value column, whose targets the new paragraph disowns (26cf9a9).

Modify `src/naics_embedder/cli/commands/training.py` with one edit. Replace:

```python
        logger.info('Starting model training with evaluation metrics...\n')
        console.print('[bold cyan]Evaluation metrics enabled:[/bold cyan]')
        console.print('  • Cophenetic correlation (hierarchy preservation)')
        console.print('  • NDCG@k (ranking quality: position-aware metric)')
        console.print('  • Embedding statistics (norms, distances)')
        console.print('  • Collapse detection (variance, norm, distance)')
        console.print('  • Distortion metrics (mean, std)\n')
```

with:

```python
        logger.info('Starting model training with evaluation metrics...\n')
        console.print('[bold cyan]Validation logs:[/bold cyan]')
        console.print('  • Embedding statistics (norms, distances)')
        console.print('  • Collapse detection (variance, norm, distance)')
        # Req 6: a structural statistic is logged for the record, never announced as a headline
        console.print('  • Structural statistics, for the record only (Req 6)\n')
```

Modify `src/naics_embedder/graph_model/hgcn.py` with these 3 edits, in order.

**`src/naics_embedder/graph_model/hgcn.py`, edit 1 of 3.** Replace:

```python
        self._ndcg_k_values = self._normalize_ndcg_k_values(cfg.ndcg_k_values)
        self._primary_ndcg = (
            10 if 10 in self._ndcg_k_values else self._ndcg_k_values[len(self._ndcg_k_values) // 2]
        )
```

with:

```python
        self._ndcg_k_values = self._normalize_ndcg_k_values(cfg.ndcg_k_values)
```

**`src/naics_embedder/graph_model/hgcn.py`, edit 2 of 3.** Replace:

```python

        for name, value in metrics_tensor.items():
            self.log(
                f'val/{name}',
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name == 'relation_accuracy'),
            )
```

with:

```python

        # Req 6: structural statistics are logged for the record, never shown as a headline
        for name, value in metrics_tensor.items():
            self.log(f'val/{name}', value, on_step=False, on_epoch=True)
```

**`src/naics_embedder/graph_model/hgcn.py`, edit 3 of 3.** Replace:

```python
                    )
                    self.log(
                        f'val/{name}',
                        tensor_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=(
                            name == 'cophenetic_correlation' or name == f'ndcg@{self._primary_ndcg}'
                        ),
                    )
```

with:

```python
                    )
                    self.log(f'val/{name}', tensor_value, on_step=False, on_epoch=True)
```

Modify `src/naics_embedder/text_model/mixins/validation.py` with these 2 edits, in order.

**`src/naics_embedder/text_model/mixins/validation.py`, edit 1 of 2.** Replace:

```python
        cophenetic_result = self.hierarchy_metrics.cophenetic_correlation(emb_dists, gt_dists)
        self.log(
            'val/cophenetic_correlation',
            self._to_python_scalar(cophenetic_result['correlation']),
            prog_bar=True,
```

with:

```python
        cophenetic_result = self.hierarchy_metrics.cophenetic_correlation(emb_dists, gt_dists)
        # Req 6: a structural statistic is logged for the record, never shown as a headline
        self.log(
            'val/cophenetic_correlation',
            self._to_python_scalar(cophenetic_result['correlation']),
```

**`src/naics_embedder/text_model/mixins/validation.py`, edit 2 of 2.** Replace:

```python
            self._to_python_scalar(distortion['median_distortion']),
            prog_bar=True,
```

with:

```python
            self._to_python_scalar(distortion['median_distortion']),
```

Modify `src/naics_embedder/tools/_visualize_metrics.py` with these 10 edits, in order.

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 1 of 10.** Replace:

```python
Visualize training metrics from log files.
```

with:

```python
Visualize training metrics from log files.

The structural statistics the training logs still record (the cophenetic correlation among them)
are not read: Req 6 keeps them out of every headline, and ``tools diagnostics`` reports them.
```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 2 of 10.** Replace:

```python

        # Extract cophenetic correlation
        cophenetic_match = re.search(
            r'Hierarchy preservation: cophenetic=([\d.-]+) \((\d+) pairs\)', line
        )
        if cophenetic_match and current_epoch is not None:
            for m in metrics:
                if m.get('epoch') == current_epoch:
                    m.update(
                        {
                            'cophenetic': float(cophenetic_match.group(1)),
                            'n_pairs': int(cophenetic_match.group(2)),
                        }
                    )
                    break

```

with:

```python

```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 3 of 10.** Replace:

```python

    # 4. Hierarchy Preservation (Cophenetic)
    ax4 = plt.subplot(3, 2, 4)
    cophenetic = [m.get('cophenetic', 0) for m in metrics if 'cophenetic' in m]
    epochs_corr = [m['epoch'] for m in metrics if 'cophenetic' in m]

    if epochs_corr and cophenetic:
        ax4.plot(epochs_corr, cophenetic, 'g-o', label='Cophenetic', linewidth=2, markersize=6)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Target (0.7)')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Cophenetic Correlation', fontsize=12)
        ax4.set_title('Hierarchy Preservation', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim((-0.5, 1.0))

    # 5. Radius Standard Deviation
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(epochs, radius_stds, 'orange', marker='o', linewidth=2, markersize=6)
    ax5.set_xlabel('Epoch', fontsize=12)
    ax5.set_ylabel('Radius Std Dev', fontsize=12)
    ax5.set_title('Hyperbolic Radius Spread', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Summary Statistics Table
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
```

with:

```python

    # 4. Radius Standard Deviation
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(epochs, radius_stds, 'orange', marker='o', linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Radius Std Dev', fontsize=12)
    ax4.set_title('Hyperbolic Radius Spread', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Summary Statistics Table, across the bottom row
    ax5 = plt.subplot(3, 1, 3)
    ax5.axis('off')
```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 4 of 10.** Replace:

```python
        
        Hierarchy Preservation:
          Cophenetic: {
            (
                f"{latest.get('cophenetic', 0):.4f}"
                if 'cophenetic' in latest and latest.get('cophenetic') is not None
                else 'N/A'
            )
        }
        
```

with:

```python
        
```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 5 of 10.** Replace:

```python
        Radius:      {first.get('radius_mean', 0):.4f} → {latest.get('radius_mean', 0):.4f}
        Cophenetic:  {
                (
                    f"{first.get('cophenetic', 0):.4f}"
                    if 'cophenetic' in first and first.get('cophenetic') is not None
                    else 'N/A'
                )
            } → {
                (
                    f"{latest.get('cophenetic', 0):.4f}"
                    if 'cophenetic' in latest and latest.get('cophenetic') is not None
                    else 'N/A'
                )
            }
```

with:

```python
        Radius:      {first.get('radius_mean', 0):.4f} → {latest.get('radius_mean', 0):.4f}
```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 6 of 10.** Replace:

```python

        ax6.text(
```

with:

```python

        ax5.text(
```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 7 of 10.** Replace:

```python

    # Hierarchy Preservation Analysis
    cophenetic = [m.get('cophenetic', 0) for m in metrics if 'cophenetic' in m]

    if cophenetic:
        print('\n📈 HIERARCHY PRESERVATION:')
        cophenetic_change = cophenetic[-1] - cophenetic[0]
        print(
            f'   Cophenetic: {cophenetic[0]:.4f} → {cophenetic[-1]:.4f} ({cophenetic_change:+.4f})'
        )

        if cophenetic[-1] > 0.7:
            print('   ✓ Excellent hierarchy preservation!')
        elif cophenetic[-1] > 0.5:
            print('   ℹ️  Good hierarchy preservation, but could improve.')
        elif cophenetic[-1] > 0.3:
            print('   ⚠️  Moderate hierarchy preservation. Model may need more training.')
        else:
            print('   ⚠️  WARNING: Low hierarchy preservation. Consider:')
            print('      - Checking if ground truth distances are correct')
            print('      - Verifying training data quality')
            print('      - Adjusting learning rate or loss function')

```

with:

```python

```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 8 of 10.** Replace:

```python

    if cophenetic and cophenetic[-1] < 0.5:
        print('   1. Hierarchy correlations are low. This could be because:')
        epoch_count = metrics[-1].get('epoch', 0)
        print(f'      - Model is still learning (only {epoch_count} epochs completed)')
        print('      - Hyperbolic space may need more time to organize hierarchy')
        print('      - Consider checking if evaluation sample size is sufficient')

    if radius_means and radius_means[-1] > 15:
        print('   2. Hyperbolic radius is growing rapidly. Monitor for:')
        print('      - Numerical stability issues')
        print('      - Whether this growth correlates with better metrics')

    if cophenetic and len(cophenetic) > 3:
        recent_trend = cophenetic[-3:]
        if all(recent_trend[i] <= recent_trend[i + 1] for i in range(len(recent_trend) - 1)):
            print('   3. Cophenetic correlation is improving! Continue training.')
        elif all(recent_trend[i] >= recent_trend[i + 1] for i in range(len(recent_trend) - 1)):
            print('   3. ⚠️  Cophenetic correlation is declining. Consider:')
            print('      - Early stopping if this continues')
            print('      - Learning rate reduction')
```

with:

```python

    if radius_means and radius_means[-1] > 15:
        print('   1. Hyperbolic radius is growing rapidly. Monitor for:')
        print('      - Numerical stability issues')
        print('      - Whether this growth correlates with better metrics')
    else:
        print('   None.')
```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 9 of 10.** Replace:

```python
        f"{'Epoch':<8} {'Radius':<15} {'Train Loss':<12} "
        f"{'Val Loss':<12} {'Cophenetic':<12} {'Dist CV':<10} {'Collapse':<10}"
```

with:

```python
        f"{'Epoch':<8} {'Radius':<15} {'Train Loss':<12} "
        f"{'Val Loss':<12} {'Dist CV':<10} {'Collapse':<10}"
```

**`src/naics_embedder/tools/_visualize_metrics.py`, edit 10 of 10.** Replace:

```python
        )
        cophenetic = f"{m.get('cophenetic', 0):.4f}" if 'cophenetic' in m else 'N/A'
        dist_cv = f"{m.get('dist_cv', 0):.4f}" if 'dist_cv' in m else 'N/A'
        collapse = 'Yes' if m.get('collapse', False) else 'No'
        print(
            f'{epoch:<8} {radius:<15} {train_loss:<12} {val_loss:<12} '
            f'{cophenetic:<12} {dist_cv:<10} {collapse:<10}'
```

with:

```python
        )
        dist_cv = f"{m.get('dist_cv', 0):.4f}" if 'dist_cv' in m else 'N/A'
        collapse = 'Yes' if m.get('collapse', False) else 'No'
        print(
            f'{epoch:<8} {radius:<15} {train_loss:<12} {val_loss:<12} '
            f'{dist_cv:<10} {collapse:<10}'
```

Modify `src/naics_embedder/tools/metrics_tools.py` with one edit. Replace:

```python
    print('=' * 90)
    print(
        f"{'Epoch':<8} {'Radius':<15} {'Cophenetic':<12} "
        f"{'Spearman':<12} {'Dist CV':<10} {'Collapse':<10}"
    )
    print('-' * 90)
    for m in metrics:
        epoch = m.get('epoch', 'N/A')
        radius = f"{m.get('radius_mean', 0):.2f}±{m.get('radius_std', 0):.2f}"
        cophenetic = f"{m.get('cophenetic', 0):.4f}" if 'cophenetic' in m else 'N/A'
        spearman = f"{m.get('spearman', 0):.4f}" if 'spearman' in m else 'N/A'
        dist_cv = f"{m.get('dist_cv', 0):.4f}" if 'dist_cv' in m else 'N/A'
        collapse = 'Yes' if m.get('collapse', False) else 'No'
        print(
            f'{epoch:<8} {radius:<15} {cophenetic:<12} {spearman:<12} {dist_cv:<10} {collapse:<10}'
        )
```

with:

```python
    print('=' * 90)
    print(f"{'Epoch':<8} {'Radius':<15} {'Dist CV':<10} {'Collapse':<10}")
    print('-' * 90)
    for m in metrics:
        epoch = m.get('epoch', 'N/A')
        radius = f"{m.get('radius_mean', 0):.2f}±{m.get('radius_std', 0):.2f}"
        dist_cv = f"{m.get('dist_cv', 0):.4f}" if 'dist_cv' in m else 'N/A'
        collapse = 'Yes' if m.get('collapse', False) else 'No'
        print(f'{epoch:<8} {radius:<15} {dist_cv:<10} {collapse:<10}')
```

Modify `docs/hgcn_training.md` with one edit. Replace:

```markdown

Stage 4 now mirrors the text-model evaluation suite so you can verify that graph refinement does not erode global structure:
```

with:

```markdown

HGCN validation logs the text model's structural statistics for the record. No progress bar
shows them and nothing selects on them (Req 6):
```

Modify `docs/overview.md` with one edit. Replace:

```markdown
| Metric | Description | Ideal Value |
```

with:

```markdown
These structural statistics are logged for the record only (Req 6): no progress bar shows
them and nothing selects on them. Configurations are compared under Req 5 on the outcome and
regressor panels (`tools margins`, `tools decide`), and Req 6's stratified diagnostics come
from `tools diagnostics` (see the [usage guide](usage.md#tools-diagnostics)).

| Metric | Description | Ideal Value |
```

Modify `docs/usage.md` with one edit. Replace:

```markdown
Visualize training metrics from log files. Creates comprehensive visualizations and analysis of training metrics including:
- Hyperbolic radius over time
- Hierarchy preservation correlations
- Embedding diversity metrics

```

with:

```markdown
Visualize training metrics from log files. Creates comprehensive visualizations and analysis of training metrics including:
- Hyperbolic radius over time, and its spread
- Training and validation loss
- Embedding diversity metrics

The structural statistics the logs still record are not shown: they are diagnostics (Req 6),
reported by `tools diagnostics`.

```

Modify `CLAUDE.md` with these 2 edits, in order.

**`CLAUDE.md`, edit 1 of 2.** Replace:

```markdown

**Expected values (after convergence):**

- Hierarchy correlation: > 0.7
- MAP: > 0.6
```

with:

```markdown

Hierarchy correlation, MAP and the other structural statistics are logged for the record only
(Req 6): no progress bar shows them, nothing selects on them, and they have no target values.
Configurations are compared under Req 5 on the outcome and regressor panels (`tools margins`,
`tools decide`), and Req 6's stratified diagnostics come from `tools diagnostics`.

**Expected values (after convergence):**

```

**`CLAUDE.md`, edit 2 of 2.** Replace:

````markdown

**Symptom:** `hierarchy_corr` metric remains low (<0.3)

**Causes:**

- Insufficient training
- Loss weights not balanced
- Ground truth distances not informative
- Curriculum not adapting properly

**Solutions:**

```bash
# Investigate ground truth distances
uv run naics-embedder tools investigate

# Increase hierarchy loss weight
uv run naics-embedder train loss.hierarchy_weight=0.5

# Train longer
uv run naics-embedder train training.trainer.max_epochs=30
```
````

with:

```markdown

A low `hierarchy_corr` or cophenetic value is not a failure to tune away. Structural statistics
are diagnostics (Req 6), and raising a loss weight or picking a checkpoint because one improves
selects on the taxonomy, which Req 1 rules out. Compare configurations under Req 5
(`tools margins`, `tools decide`), and report Req 6's diagnostics with `tools diagnostics`.
```

Modify `README.md` with these 2 edits, in order.

**`README.md`, edit 1 of 2.** Replace:

```markdown

To ensure graph refinement does not erode the global structure captured by the text model, the same hierarchy-aware metrics introduced earlier in the pipeline are logged:
```

with:

```markdown

HGCN validation logs the same structural statistics as the text model:
```

**`README.md`, edit 2 of 2.** Replace:

```markdown
`structural-spearman-v1` validates square symmetric distance matrices, averages each mirrored
```

with:

```markdown
These are logged for the record only (Req 6): no progress bar shows them and nothing selects on
them. Req 6's stratified diagnostics come from `tools diagnostics` (section 5.5).

`structural-spearman-v1` validates square symmetric distance matrices, averages each mirrored
```

- [x] **Step 4: Run the tests to verify they pass, and build the docs**

Run: `uv run pytest tests/unit/test_hgcn_metrics.py tests/unit/test_text_validation_metrics.py tests/unit/test_visualize_metrics.py tests/unit/test_metrics_tools_api.py tests/unit/test_cli_training.py -q`
Expected: `77 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1590 passed, 1 skipped`.

Run: `uv run mkdocs build --strict -q -d /tmp/stage4-docs-ec267a03`, then
`rm -rf /tmp/stage4-docs-ec267a03`
Expected: no output and exit 0.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/cli/commands/training.py src/naics_embedder/graph_model/hgcn.py src/naics_embedder/text_model/mixins/validation.py src/naics_embedder/tools/_visualize_metrics.py src/naics_embedder/tools/metrics_tools.py tests/unit/test_cli_training.py tests/unit/test_hgcn_metrics.py tests/unit/test_metrics_tools_api.py tests/unit/test_text_validation_metrics.py tests/unit/test_visualize_metrics.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/cli/commands/training.py \
  src/naics_embedder/graph_model/hgcn.py \
  src/naics_embedder/text_model/mixins/validation.py \
  src/naics_embedder/tools/_visualize_metrics.py \
  src/naics_embedder/tools/metrics_tools.py \
  tests/unit/test_cli_training.py \
  tests/unit/test_hgcn_metrics.py \
  tests/unit/test_metrics_tools_api.py \
  tests/unit/test_text_validation_metrics.py \
  tests/unit/test_visualize_metrics.py \
  docs/hgcn_training.md \
  docs/overview.md \
  docs/usage.md \
  CLAUDE.md \
  README.md
git commit -m "feat: take structural statistics off progress bars and headlines (Req 6)"
```

### Task 13: Real-data diagnostics (controller, inline)

This task reads the real codebook and descriptions, read-only, and reads no panel. It checks
four things:

- the text-only table's provenance now records the `matrix_fingerprint` a regressor read logs;
- the store accepts the table and its provenance;
- the diagnostics report covers the real codebook's 2,125 codes with Req 6's statistics only;
- the statistics separate a real representation from chance.

The task commits nothing. If a number differs from **Expected real-data results**, stop and ask.

- [x] **Step 1: Make the scratch directory**

Run: `pwd`
Expected:
`/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-6-decision-rule-and-diagnostics`.
If it prints anything else, `cd` there first.

Run: `mkdir -p /tmp/stage4-diagnostics-ec267a03`

- [x] **Step 2: Build the text-only table**

Run: `COLUMNS=120 uv run naics-embedder tools text-only-table --descriptions /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet --output /tmp/stage4-diagnostics-ec267a03/text_only.parquet`
Expected: about 34 s. The output ends with:

```text
Text-only table (2,125 codes, width 384): /tmp/stage4-diagnostics-ec267a03/text_only.parquet
Text-only table: /tmp/stage4-diagnostics-ec267a03/text_only.parquet
Provenance: /tmp/stage4-diagnostics-ec267a03/text_only_provenance.json
```

Run: `shasum -a 256 /tmp/stage4-diagnostics-ec267a03/text_only.parquet`
Expected: `6386f912ea4b37984ed9d4e4100f21fbdc29e91c5ee38acb05bd999342f5bbe0`.

Run: `uv run python -c "import json; p = json.load(open('/tmp/stage4-diagnostics-ec267a03/text_only_provenance.json')); print(p['table_sha256'], p['matrix_fingerprint'], p['backbone'], p['revision'], p['max_length'], p['codes'], p['hidden_size'])"`
Expected, on one line:

```text
6386f912ea4b37984ed9d4e4100f21fbdc29e91c5ee38acb05bd999342f5bbe0 6838c0adf0573821b139183a11db7d232f3965c99d3715202239f9463ba13384 sentence-transformers/all-MiniLM-L6-v2 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 512 2125 384
```

The provenance carries both names:

- `table_sha256`, the file's hash;
- `matrix_fingerprint`, the name a regressor read logs the table by (Task 1).

- [x] **Step 3: Store the table with its provenance**

Run: `uv run python -c "from naics_embedder.decision.store import ArtifactStore; ref = ArtifactStore('/tmp/stage4-diagnostics-ec267a03/store').put_text_only('/tmp/stage4-diagnostics-ec267a03/text_only.parquet'); print(ref.table.matrix_fingerprint, ref.backbone, ref.revision, ref.max_length, ref.descriptions_sha256[:8])"`
Expected:
`6838c0adf0573821b139183a11db7d232f3965c99d3715202239f9463ba13384 sentence-transformers/all-MiniLM-L6-v2 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 512 5107fb83`.
These are the fields `check_text_only` compares with an arm's spec (D9).

- [x] **Step 4: Report on the text-only table**

Run: `COLUMNS=120 uv run naics-embedder tools diagnostics --table /tmp/stage4-diagnostics-ec267a03/text_only.parquet --geometry spherical --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet --output /tmp/stage4-diagnostics-ec267a03/text_only_diagnostics.json`
Expected, in about 4 s:

```text
Structural diagnostics (Req 6): 2,125 codes, spherical

  • sector separation AUC: 0.8538 (272,103 same-sector, 1,984,647 cross-sector pairs)
  • within-sector rank correlation: 0.2974 over 2,125 queries, 0.3857 over 20 sectors (0 undefined)
  • MAP over ancestors: 0.3262 over 2,105 queries (level 3 0.1415, level 4 0.1761, level 5 0.2543, level 6 0.4384)
  • NDCG: @5 0.8220, @10 0.7742, @20 0.7398
  • distance Pearson with D*: 0.2378 over 2,256,750 pairs
  • parent retrieval: @1 0.2855, @5 0.6639 over 1,583 queries (522 unary pairs excluded)

Descriptive only: nothing selects on these, and none has a threshold.

Report written to /tmp/stage4-diagnostics-ec267a03/text_only_diagnostics.json
```

`COLUMNS=120` keeps Rich from folding the long lines.

Run: `uv run python -c "import json; r = json.load(open('/tmp/stage4-diagnostics-ec267a03/text_only_diagnostics.json')); print(sorted(r)); print(r['geometry'], r['codes'], r['ndcg']['@10']['queries'], {k: round(v, 4) for k, v in r['ndcg']['@10']['by_level'].items()})"`
Expected:

```text
['codes', 'curvature', 'distance_pearson', 'geometry', 'map_over_ancestors', 'ndcg', 'parent_retrieval', 'sector_separation', 'within_sector_rank_correlation']
spherical 2125 2125 {'2': 0.7948, '3': 0.8645, '4': 0.82, '5': 0.7676, '6': 0.7559}
```

The report's keys are Req 6's statistics and nothing else: no threshold, no verdict.

- [x] **Step 5: Report on a random table in two geometries**

Run: `uv run python -c "import numpy as np, polars as pl; codes = pl.read_parquet('/Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet').get_column('code').to_list(); rng = np.random.default_rng(20260924); coords = pl.DataFrame(rng.normal(size=(len(codes), 16)), schema={f'e{i}': pl.Float64 for i in range(16)}, orient='row'); pl.DataFrame({'code': codes}).hstack(coords).write_parquet('/tmp/stage4-diagnostics-ec267a03/random.parquet')"`

Run: `shasum -a 256 /tmp/stage4-diagnostics-ec267a03/random.parquet`
Expected: `e8534cc24d7e931b0ab6c145b307e19c0c0d8a6c0a17a74e74fc828cf221361c`.

Run: `COLUMNS=120 uv run naics-embedder tools diagnostics --table /tmp/stage4-diagnostics-ec267a03/random.parquet --geometry euclidean --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet --output /tmp/stage4-diagnostics-ec267a03/random_euclidean.json`
Expected, after the header line `Structural diagnostics (Req 6): 2,125 codes, euclidean`:

```text
  • sector separation AUC: 0.4940 (272,103 same-sector, 1,984,647 cross-sector pairs)
  • within-sector rank correlation: 0.0038 over 2,125 queries, 0.0190 over 20 sectors (0 undefined)
  • MAP over ancestors: 0.0052 over 2,105 queries (level 3 0.0016, level 4 0.0037, level 5 0.0065, level 6 0.0052)
  • NDCG: @5 0.0441, @10 0.0481, @20 0.0544
  • distance Pearson with D*: 0.0030 over 2,256,750 pairs
  • parent retrieval: @1 0.0019, @5 0.0032 over 1,583 queries (522 unary pairs excluded)
```

Run: `COLUMNS=120 uv run naics-embedder tools diagnostics --table /tmp/stage4-diagnostics-ec267a03/random.parquet --geometry hyperbolic --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet --output /tmp/stage4-diagnostics-ec267a03/random_hyperbolic.json`
Expected, after the header line `Structural diagnostics (Req 6): 2,125 codes, hyperbolic`:

```text
  • sector separation AUC: 0.4928 (272,103 same-sector, 1,984,647 cross-sector pairs)
  • within-sector rank correlation: 0.0073 over 2,125 queries, 0.0348 over 20 sectors (0 undefined)
  • MAP over ancestors: 0.0050 over 2,105 queries (level 3 0.0016, level 4 0.0033, level 5 0.0064, level 6 0.0050)
  • NDCG: @5 0.0384, @10 0.0427, @20 0.0494
  • distance Pearson with D*: 0.0051 over 2,256,750 pairs
  • parent retrieval: @1 0.0006, @5 0.0038 over 1,583 queries (522 unary pairs excluded)
```

The random table sits at chance on every statistic: AUCs near 0.5 and correlations near 0. The
text-only table sits well above it. The hyperbolic run reads the same coordinates as tangent
vectors at the origin, so its distances differ from the Euclidean run's, as its numbers show.

- [x] **Step 6: Compare**

Compare every number in Steps 2–5 with **Expected real-data results**, to 4 decimals. They must
match exactly: the computation is deterministic under the pinned versions. If any differs, stop
and ask. Keep the scratch directory until Final verification Step 8.

## Final verification (controller, inline)

- [x] **Step 1: Full suite on Python 3.12**

> Deviation: run after the final review's fix, at 26cf9a9. 43 warnings against about 41 before: the final review asked for their sources, and every one comes from site-packages or the untouched `tests/unit/test_hgcn.py:108`, through pre-existing tests; the swing is Lightning's warn-once warnings landing in one more xdist worker.

Run: `uv run pytest -n auto -q`
Expected: `1590 passed, 1 skipped`.

- [x] **Step 2: Full suite on Python 3.10, CI's other leg**

> Deviation: the environment was pre-built with `uv sync --locked` while Task 12 ran. The run warned 346 times against 43 on 3.12: numpy 2.2.6, which the lock pins for Python 3.10, raises spurious "divide by zero", "overflow" and "invalid value encountered in matmul" RuntimeWarnings with Accelerate on this Mac (reproduced on plain finite inputs; numpy 2.3.4 raises none), at `decision/resampling.py:131-132` and the older `panels/ridge.py:60-63`. CI's Linux wheels are unaffected.

Run: `UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310-ec267a03 uv run pytest -n auto -q`
Expected: `1590 passed, 1 skipped`, the same as Step 1.

Run: `rm -rf /tmp/naics-py310-ec267a03`

- [x] **Step 3: The CI lint job**

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 4: The docs build**

Run: `uv run mkdocs build --strict -q -d /tmp/stage4-docs-ec267a03`, then
`rm -rf /tmp/stage4-docs-ec267a03`
Expected: no output and exit 0. PR CI never runs the docs workflow, so this build stands in for
it. After the merge, check the first docs run on `main`.

- [x] **Step 5: The branch carries only this plan's commits**

> Deviation: 60 paths: F1 adds `src/naics_embedder/panels/selection_log.py` (checked mechanically against this list). The log held 19 commits: the plan's, the 12 task commits, five task-review fixes and the final review's docs fix.

Run: `git log --oneline origin/main..HEAD`
Expected, read bottom up, because `git log` prints the newest commit first:

- the plan's commit, at the bottom;
- then each task's commit in task order, Task 1's through Task 12's. The pre-flight and Task 13
  commit nothing.
- Review fixes may add commits between them.

Neither "config" nor "graph config" appears.

Run: `git diff --name-only origin/main...HEAD`
Expected: exactly these 59 paths. The three dots diff from the merge base, so a commit that lands
on `origin/main` meanwhile does not show up. Deleted files are listed too.

```text
CLAUDE.md
README.md
WARP.md
conf/data/decision.yaml
docs/.nav.yml
docs/api/decision.md
docs/api/diagnostics.md
docs/api/embeddings_verification.md
docs/api/qcew_metrics.md
docs/hgcn_training.md
docs/overview.md
docs/usage.md
specs/plans/6-decision-rule-and-diagnostics.md
src/naics_embedder/cli/commands/tools.py
src/naics_embedder/cli/commands/training.py
src/naics_embedder/decision/__init__.py
src/naics_embedder/decision/decide.py
src/naics_embedder/decision/records.py
src/naics_embedder/decision/resampling.py
src/naics_embedder/decision/rule.py
src/naics_embedder/decision/scores.py
src/naics_embedder/decision/store.py
src/naics_embedder/decision/sweep.py
src/naics_embedder/graph_model/__init__.py
src/naics_embedder/graph_model/hgcn.py
src/naics_embedder/metrics/__init__.py
src/naics_embedder/metrics/diagnostics.py
src/naics_embedder/metrics/graph.py
src/naics_embedder/metrics/qcew.py
src/naics_embedder/panels/outcome.py
src/naics_embedder/panels/regressor.py
src/naics_embedder/panels/text_only.py
src/naics_embedder/text_model/mixins/validation.py
src/naics_embedder/tools/_visualize_metrics.py
src/naics_embedder/tools/embeddings_verification.py
src/naics_embedder/tools/metrics_tools.py
src/naics_embedder/utils/config.py
tests/fixtures/decision.py
tests/unit/test_cli_commands.py
tests/unit/test_cli_training.py
tests/unit/test_config.py
tests/unit/test_decision.py
tests/unit/test_decision_resampling.py
tests/unit/test_decision_rule.py
tests/unit/test_decision_scores.py
tests/unit/test_decision_store.py
tests/unit/test_decision_sweep.py
tests/unit/test_diagnostics.py
tests/unit/test_embeddings_verification.py
tests/unit/test_graph_downstream_evaluation.py
tests/unit/test_graph_pairwise_distances.py
tests/unit/test_hgcn_metrics.py
tests/unit/test_metrics_tools_api.py
tests/unit/test_outcome_panel.py
tests/unit/test_qcew_multilevel.py
tests/unit/test_regressor_panel.py
tests/unit/test_text_only.py
tests/unit/test_text_validation_metrics.py
tests/unit/test_visualize_metrics.py
```

- [x] **Step 6: The roadmap's Stage 4 Exit, outcome by outcome**

Check each row against the evidence that backs it. Test names are given as `file::test`, and a
`::test` alone continues the file before it.

| Exit outcome | Evidence |
|---|---|
| On synthetic arms with known effects on D8's three panels, the tooling adopts and rejects per the rule | `test_decision.py::test_an_arm_superior_on_one_panel_and_level_on_the_others_is_adopted`, `::test_superiority_on_one_panel_does_not_rescue_an_inferior_one`, `::test_without_superiority_the_simpler_arm_stands`, `::test_a_dominance_cycle_leaves_every_arm_to_the_tie_order` and `::test_the_held_out_gain_over_ancestors_breaks_the_last_tie`; `test_decision_rule.py` (all seven); `test_decision_sweep.py::test_a_decision_over_swept_arms_adopts_the_informed_one`; `test_cli_commands.py::test_decide_adopts_the_better_arm_and_writes_the_record` and `::test_decide_reports_a_tie_it_cannot_break` |
| The records have every field Verification "Decision records" lists | `test_decision.py::test_the_record_carries_every_field_verification_lists`, `::test_each_margin_is_the_multiple_times_the_references_across_seed_sd`, `::test_a_margin_needs_a_positive_multiple_and_a_reference_that_varies` and `::test_every_arm_needs_five_seeds`; `test_decision_store.py::test_a_record_is_written_once_and_reads_back_whole`; `test_cli_commands.py::test_margins_writes_each_panels_margin` |
| The records carry the selection-log records of their runs | `test_decision_sweep.py::test_every_seed_is_read_once_per_panel_and_its_records_are_the_logs`; `test_decision.py::test_a_runs_log_records_must_be_its_own_validation_reads`, `::test_a_regressor_read_must_name_the_runs_tables`, `::test_a_run_that_read_before_the_margins_were_fixed_is_refused` and `::test_paired_arms_must_read_the_same_panels`; `test_outcome_panel.py::test_a_read_carries_the_callers_detail_beside_the_panels_own` and `::test_a_read_cannot_replace_what_the_panel_logs`; `test_regressor_panel.py::test_a_read_carries_the_callers_detail_beside_the_panels_own`, `::test_a_read_cannot_replace_what_the_panel_logs` and `::test_the_logged_table_names_are_their_matrix_fingerprints` |
| The records carry the artifact references of every arm and seed, text-only tables and their provenance included | `test_decision_store.py` (all five); `test_decision_sweep.py::test_the_artifacts_outlive_the_runners_files` and `::test_a_text_only_table_from_another_backbone_is_refused_before_any_read`; `test_decision.py::test_the_text_only_table_must_come_from_the_arms_backbone_and_text` and `::test_a_changed_artifact_is_refused`; `test_text_only.py::test_the_table_and_its_provenance_are_written`; Task 13 Steps 2–3 |
| The diagnostics report contains only Req 6's statistics, stratified as listed, with no threshold and no pass/fail | `test_diagnostics.py` (all twelve), among them `::test_the_report_holds_only_req_6s_statistics`; `test_cli_commands.py::test_diagnostics_reports_every_statistic_and_writes_json` and `::test_diagnostics_refuses_a_table_that_misses_a_codebook_code`; Task 13 Steps 4–5 on the real codebook |
| No monitor, gate or headline reads a structural statistic | Task 12's six tests: `test_hgcn_metrics.py::test_hgcn_puts_no_structural_statistic_on_the_progress_bar`, `test_text_validation_metrics.py::test_no_structural_statistic_reaches_the_progress_bar`, `test_visualize_metrics.py::TestParseLogFile::test_parse_leaves_out_the_structural_statistic` and `::TestPrintAnalysis::test_print_analysis_grades_no_structural_statistic`, `test_metrics_tools_api.py::TestVisualizeMetrics::test_visualize_metrics_tables_no_structural_statistic`, `test_cli_training.py::test_the_train_banner_headlines_no_structural_statistic`; `verify-stage4`'s gates are gone (`test_cli_commands.py::test_verify_stage4_is_gone`); the two `git grep` runs below |
| Neither `metrics/qcew.py` nor the taxonomy-tasks suite remains | `test_graph_downstream_evaluation.py::test_the_qcew_benchmark_and_the_downstream_suite_are_gone`; Task 10 Step 4's `git grep` |

Run: `uv run pytest tests/unit/test_decision.py tests/unit/test_decision_rule.py tests/unit/test_decision_sweep.py tests/unit/test_decision_store.py tests/unit/test_decision_scores.py tests/unit/test_decision_resampling.py tests/unit/test_diagnostics.py tests/unit/test_cli_commands.py tests/unit/test_outcome_panel.py tests/unit/test_regressor_panel.py tests/unit/test_text_only.py tests/unit/test_graph_downstream_evaluation.py tests/unit/test_hgcn_metrics.py tests/unit/test_text_validation_metrics.py tests/unit/test_visualize_metrics.py tests/unit/test_metrics_tools_api.py tests/unit/test_cli_training.py -q`
Expected: `278 passed`.

Run: `git grep -n -E "monitor=" -- src`
Expected: four lines, each `monitor='val/contrastive_loss'`: two in
`src/naics_embedder/cli/commands/training.py` and two in `src/naics_embedder/utils/training.py`.
These are every checkpoint and early-stopping monitor. HGCN runs with
`enable_checkpointing=False` and has no monitor.

Run: `git grep -n -E "CurriculumController\(" -- src`
Expected: no output. The graph curriculum controller, which gates phases on validation metrics,
is never built in training.

- [x] **Step 7: No panel was read and no selection log was written**

Run: `uv run python -c "import glob; print(sorted(glob.glob('logs/*.jsonl') + glob.glob('/tmp/stage4-diagnostics-ec267a03/*.jsonl')))"`
Expected: `[]`. Tests write their selection logs under pytest's temporary directories, Task 13
reads no panel, and `logs/` holds only `.log` files. If a `.jsonl` file appears, stop and ask.

- [x] **Step 8: Remove the scratch directory**

Run: `rm -rf /tmp/stage4-diagnostics-ec267a03`

Nothing else in `/tmp` is this plan's: Step 2 and the docs builds removed their own directories.

## Plan completion

Run the Plan Completion Protocol of writing-plans after the final review. The completion commits
are the branch's last commits. Before editing `specs/naics-embedding-roadmap.md` or
`specs/deferred_items.md`, check whether another Claude session is active in this repository. If
one is, hold both edits and hand your human partner the exact text below.

- [x] **Step 1: Tick the roadmap stage and add the rollout note and the stamp**

> Deviation: by user ruling at the gate, the rollout note also says that `decide` compares log timestamps with the margins' `fixed_at` across machines, so their clocks must agree.

In `specs/naics-embedding-roadmap.md`, make one edit. Replace:

```markdown
- [ ] Stage 4: Decision rule and diagnostics
```

with:

```markdown
- [x] Stage 4: Decision rule and diagnostics
```

Make a second edit. Replace the Stage 4 entry's last lines:

```markdown
      `metrics/qcew.py` nor the taxonomy-tasks suite remains.
      ROUTING: writing-plans

- [ ] Stage 5: Supervision target and text
```

with the lines below, with `YYYY-MM-DD` replaced by the completion date:

```markdown
      `metrics/qcew.py` nor the taxonomy-tasks suite remains.
      ROUTING: writing-plans
      Rollout note: the decision tooling ran on synthetic arms and fixture panels only, so
      Stage 7's reference sweep is its first real use; its artifact store sits outside any
      worktree and is copied off a Lambda instance before termination. `tools diagnostics`
      replaced `verify-stage4`. Text and HGCN validation still log the old structural
      statistics, off every progress bar and headline, for Stages 7 and 11 to remove.
      Realized: on the text-only table (all-MiniLM-L6-v2 at revision 1110a243, 384 dimensions,
      spherical), sector-separation AUC 0.8538, within-sector rank correlation 0.2974 over queries
      and 0.3857 over sectors, MAP over ancestors 0.3262, NDCG@10 0.7742, distance Pearson
      0.2378, parent retrieval@1 0.2855 over 1,583 queries; a random table reads at chance
      (AUC 0.4940).
      Stage 4: COMPLETE (YYYY-MM-DD) — implemented by plan 6
      (specs/plans/completed/6-decision-rule-and-diagnostics.md). Next: resume the roadmap.

- [ ] Stage 5: Supervision target and text
```

- [x] **Step 2: Re-validate the later stages against what shipped**

> Deviation: by user ruling at the gate, Stages 6 and 8 gained edits too: Stage 6's Produces, an outcome read whose logged `table` names the code vectors the encoder decodes against; Stage 8's Produces, guards on the geometry and dimension tie-order keys. The commit says Stages 5–12.

Six later entries consume what Stage 4 shipped in ways their text does not yet say. Each edit's
Replace text occurs exactly once in the roadmap.

In the Stage 5 entry, replace:

```markdown
      comparator at `text_only.max_length: 512` (`conf/data/regressor_panel.yaml`), the window
      Req 9 rejects.
```

with:

```markdown
      comparator at `text_only.max_length: 512` (`conf/data/regressor_panel.yaml`), the window
      Req 9 rejects. Stage 4's diagnostics report (`metrics/diagnostics.py`), which computes D*
      from the codes' own lineage rather than reading the bundle, so the bundle's new D* must
      equal it on every pair.
```

In the Stage 7 entry, make three edits. First, replace:

```markdown
      embeds bundle 18403d29's text, which Stage 5 replaces (D9). Stage 4's seed-sweep driver,
      decision tooling and δ procedure.
```

with:

```markdown
      embeds bundle 18403d29's text, which Stage 5 replaces (D9). Stage 4's seed-sweep driver
      (`decision.sweep.run_seed_sweep`, whose `ArmRunner` returns each seed's `SeedArtifacts`:
      the checkpoint, the 2,125-code table in the export form, the `QueryCodeEncoder` and its
      distance), decision tooling (`tools margins`, `tools decide`) and δ procedure. The margins
      are fixed from the reference arm's record before any other arm of a decision reads a
      panel, since a decision refuses a run that read first, and the artifact store's root must
      outlive the Lambda instance that trains.
```

Second, replace:

```markdown
      the validation query split's MRR (D6); a decision record fixing δ for each of D8's three
      panels from at least 5 seeds.
```

with:

```markdown
      the validation query split's MRR (D6); a decision record fixing δ for each of D8's three
      panels from at least 5 seeds; the legacy structural statistics removed from the text
      stage's validation (`text_model/evaluation.py`, `text_model/mixins/validation.py`,
      `text_model/mixins/logging.py`), which Stage 4 took off the progress bar only, and
      `tools investigate` retired, so Req 6's statistics come only from `tools diagnostics`.
```

Third, replace:

```markdown
      validation splits read; a decision record fixes δ for each of D8's three panels from at
      least 5 seeds.
```

with:

```markdown
      validation splits read; a decision record fixes δ for each of D8's three panels from at
      least 5 seeds; the text stage's validation computes no structural statistic.
```

In the Stage 9 entry, replace:

```markdown
      adopted, the bundle's target and the diagnostics' relevance grades switched to it.
```

with:

```markdown
      adopted, the bundle's target and the diagnostics' relevance grades (lowest-common-ancestor
      depths in `metrics/diagnostics.py`, Stage 4) switched to it.
```

In the Stage 10 entry, replace:

```markdown
      geometry, its level-radius term per D3, checkpoint selected on validation, no private
      tail, no last-epoch export); arm E (text-shuffle control, run only if D wins); the
      keep-or-drop record; the deliverable, a 2,125-code table from the selected arm.
```

with:

```markdown
      geometry, its level-radius term per D3, checkpoint selected on the validation query
      split's MRR (D6), not on the structural statistics HGCN's validation still logs (Stage 4
      took them off the progress bar only), no private tail, no last-epoch export); arm E
      (text-shuffle control, run only if D wins); the keep-or-drop record; the deliverable, a
      2,125-code table from the selected arm.
```

In the Stage 11 entry, replace:

```markdown
      Exit: If kept: a Req 5 record per repair and a final 2,125-code table from the repaired
      stage. If dropped: no graph-stage code path remains, the suite passes, the deliverable is
      the text stage's table, and arm D's referenced artifacts still match their hashes.
```

with:

```markdown
      Exit: If kept: a Req 5 record per repair, a final 2,125-code table from the repaired
      stage, and no structural statistic computed at HGCN validation, which Stage 4 took off the
      progress bar only. If dropped: no graph-stage code path remains, the suite passes, the
      deliverable is the text stage's table, and arm D's referenced artifacts still match their
      hashes.
```

In the Stage 12 entry, replace:

```markdown
      through it would be logged as a reopen; no command opens the outcome test split. Stage 4's
      tooling and seed-sweep driver.
```

with:

```markdown
      through it would be logged as a reopen; no command opens the outcome test split. Stage 4's
      tooling and seed-sweep driver, which score and decide on validation reads only: `decide`
      refuses a run whose records are not validation reads, and `regressor_scores` takes
      level-6 validation rows only. A sealed estimate reuses `decision/resampling.py` and
      `decision/rule.py` on test-split scores, one prediction per outer row, through a path this
      stage adds.
```

Stages 6 and 8 need no edit:

- Stage 6's export and `QueryCodeEncoder` are what the sweep's `ArmRunner` returns, and its
  entry already produces both. `tools diagnostics` reads the same export.
- Stage 8's arms pass their distance through `SeedArtifacts.distance`, and `ArmSpec.geometry`
  carries what the tie order reads. Its entry already consumes Stage 4's tooling and driver.

Commit the roadmap edits with the plan markup in Step 3's commit.

- [x] **Step 3: Mark up this plan and resolve the gate**

> Deviation: the gate asked one batched question: the user chose to drop overview.md's target column (done before Final verification), defer the review's other findings as grouped entries, add the Stage 4, 6 and 8 roadmap text, and defer the tie rule. Seven items were appended; the backlog stands at 19 open, under the triage threshold.

Follow the protocol:

- Run the resolve-before-defer gate.
- Tick every completed step and add `> Deviation:` notes.
- Add the status header.
- Tick the two plan-5 items this plan implemented, in `specs/deferred_items.md`, as below.
- Append this plan's deferred items, if any.
- Run `uv run --no-project --python 3.13 python ~/.claude/skills/writing-plans/scripts/deferred_stats.py`
  and surface its summary line.

The first plan-5 item: replace

```markdown
- [ ] Review Minor: a regressor read's log record names the text-only table by
```

with

```markdown
- [x] Review Minor: a regressor read's log record names the text-only table by
```

and replace

```markdown
      in the provenance. Size: quick-fix. Revisit if: Stage 4's tooling or a later stage has to
      match logged reads to text-only table files.
```

with

```markdown
      in the provenance. Size: quick-fix. Revisit if: Stage 4's tooling or a later stage has to
      match logged reads to text-only table files.
      → done in plan 6 (Task 1 records `matrix_fingerprint` in the provenance beside
      `table_sha256`; Task 5's store refuses a provenance naming another).
```

The second plan-5 item: replace

```markdown
- [ ] Review note, for Stage 4's decision statistic: a validation read scores each row once
```

with

```markdown
- [x] Review note, for Stage 4's decision statistic: a validation read scores each row once
```

and replace

```markdown
      Done when: Stage 4's plan aggregates repeats per row before resampling by group and
      reports the held-out regime by feature year, or records why not.
```

with

```markdown
      Done when: Stage 4's plan aggregates repeats per row before resampling by group and
      reports the held-out regime by feature year, or records why not.
      → done in plan 6 (Task 2 averages each row's repeats, requiring every repeat, before Task
      3's group resampling; Task 6's record reports the held-out regime by feature year).
```

Plan 5's `run_plan` item stays open: its trigger is a Stage 4 seed sweep slowed by panel reads,
and this plan's sweep reads fixture panels only.

Then commit:

```bash
git add specs/naics-embedding-roadmap.md specs/plans/6-decision-rule-and-diagnostics.md specs/deferred_items.md
git commit -m "docs(roadmap): complete Stage 4 and re-validate Stages 5, 7 and 9–12"
```

- [x] **Step 4: Retire the plan**

```bash
git mv specs/plans/6-decision-rule-and-diagnostics.md specs/plans/completed/6-decision-rule-and-diagnostics.md
git commit -m "chore(specs): retire plan 6"
```

This plan has no relative links to re-point, and no spec file retires with it: Stage 4 has no
stage spec.

- [x] **Step 5: Integrate**

Hand over to finishing-a-development-branch. Before opening any PR, check two things:

- `git log --oneline origin/main..HEAD` shows only this branch's commits.
- `git diff --name-only origin/main...HEAD -- conf/config.yaml conf/graph.yaml` prints nothing.

Never push to `main`. A PR merges only after two things:

- CI passes: `lint`, `test (3.10)` and `test (3.12)`;
- Codex's review has given its 👍, or its inline findings are fixed.

After the merge, check the first docs run on `main`: it is the first build of the new API pages
and `docs/.nav.yml` in CI.
