# Regressor Panel Implementation Plan

**Status: COMPLETE (2026-09-24)** — executed via executing-plans; deferred items in specs/deferred_items.md (four from the whole-plan review: one for Stage 4, one due before Stage 12's opening, two standalone)

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-embedding-roadmap.md, Stage 3 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Build roadmap Stage 3's regressor panel: a command that takes an arm's 2,125-code
coordinate table and returns out-of-sample predictions of next-year QCEW employment per row,
keyed by code, year and group, for every Req 2 comparator in two regimes (seen codes and held-out
four-digit groups). Each regime's outer set is drawn once, committed under a fingerprint, and read
only through a logged opening.

**Architecture:** Five modules join the `naics_embedder.panels` package:

- `qcew_rows`: reads the pinned QCEW national slices into cells, the population and the dated
  rows (D7).
- `regressor_splits`: partitions the rows into the remainder, the seen outer set and the held-out
  outer set, and holds the held-out draw's committed table.
- `ridge`: ridge on standardized features along a penalty grid.
- `text_only`: the text-only comparator's table, from the arm's backbone, frozen (D9).
- `regressor`: `RegressorPanel`, its fit plans, the comparators and the logged openings.

`naics-embedder data regressor-groups` draws the held-out groups once, and they are committed as
`conf/data/regressor_heldout_groups.csv`. `naics-embedder tools text-only-table` builds the
text-only table, and `naics-embedder tools regressor-panel` scores an arm. On real data this plan
reads only the validation split, with a stub arm: Stage 12 opens the outer sets.

**Tech Stack:** Python 3.10 and 3.12 (CI runs both); polars; numpy; scikit-learn (`PCA`, and the
reference in the ridge test); torch and transformers (the frozen backbone, on the CPU); pydantic;
typer; pytest with xdist; ruff and yapf.

## Global Constraints

Every task's requirements include this section.

### The spec (`specs/naics-embedding.md` at d9126ce), verbatim

- Req 1: "**As a regressor:** the out-of-sample gain from using the embedding's coordinates in
  place of sparse NAICS encodings in a downstream model (Req 2)."
- Req 2: "The embedding's coordinates enter a downstream model as regressors: tangent coordinates
  at $o$ for a hyperbolic arm (methodology S4 procedure), raw coordinates otherwise."
- Req 2: "**Rows sit below the code.** A row is a code in a given year, or in a given area".
- Req 2: "**Comparators** share the downstream model and a tuned penalty" … "six-digit one-hot
  indicators; ancestor indicators at levels 2–5; a text-only representation reduced to the same
  dimension; covariates only; covariates plus each representation."
- Req 2: "**Fitting.** Features are standardized and the penalty is tuned by nested
  cross-validation".
- Req 2: "Seen codes: rows are split by year or area, so every code appears in training. This is
  where one-hot is a real competitor." … "Held-out codes: every row of a held-out code group
  leaves training, with four-digit parents as groups. Here one-hot can predict only the
  intercept".
- Req 2: "The sealed test (Req 4) is an outer set: held-out four-digit groups for the held-out
  regime, and held-out years or areas for the seen regime. Repeated grouped folds run inside the
  remainder, for penalty tuning and selection."
- Req 2: "The multi-level variant (levels 2–6) is kept".
- Req 4: "Each panel has a validation split and a sealed test split" … "Every selection reads
  validation splits only" … "The test splits are opened once, for the final configuration and
  the comparisons recorded for it".
- Req 5: "Differences Δ = A − B are paired: both arms are scored on the same resample of the
  evaluation unit" … "The unit is codes, with their queries, for the outcome panel, and
  four-digit-parent groups for the regressor panel."
- Verification "Panels", regressor half: "The regressor panel reports the seen-code and
  held-out-code regimes separately, with one-hot only in the seen regime."

### The roadmap (`specs/naics-embedding-roadmap.md`), verbatim

- Stage 3 Produces: "The panel as a command that takes a coordinate table and returns
  out-of-sample predictions per row, keyed by code, year and four-digit-parent group, so that
  Stage 4 can compute whichever statistic Open questions settles. It covers every comparator in
  each regime, the text-only one per D9, on D7's outcome, fitted by ridge on standardized
  features with a nested-cross-validated penalty and with log establishment counts and log wages
  as the covariates (D1). Also produced: the sealed outer sets, readable only through a logged
  opening as Stage 2's test split is (`SelectionLog` records reads but gates none), and every
  outer-set score goes through that opening; the held-out regime's drawn four-digit groups
  committed under a fingerprint as Stage 2's role table is (a redraw gets a new fingerprint,
  which `SelectionLog.openings` would not count as a reopening; the seen regime's 2024→2025 set
  is fixed by D7); the multi-level variant; the branch record Stage 1 dictated."
- Stage 3 Exit: "The panel reports the seen-code and held-out-code regimes separately, with
  one-hot only in the seen regime and every Req 2 comparator scored in each; a test shows the
  penalty is tuned inside the remainder and the outer set is read once; a test shows the panel
  reads the committed outer groups, never a fresh draw; a test shows neither regime's outer set
  can be read without a logged opening; a test shows every row's features are dated before its
  outcome (D7); the branch record matches the finding's decision block."
- D1: "Decision: ridge with a nested-cross-validated penalty; covariates are log establishment
  counts and log wages from the same QCEW rows, as in the methodology's definition
  (`metrics/qcew.py`)."
- D7: "Decision: log annual-average employment in year t+1, from year-t features (the
  representation, plus D1's covariates from the year-t row); the same-year outcome is dropped.
  With 2022–2025 final, feature years run 2022–2024, and splits by time seal 2024→2025 for the
  seen regime. That leaves the seen regime's repeated grouped inner folds two feature years;
  Stage 3's plan fits them to that."
- D8: "Decision: each regime counts as a panel under Req 5, which then has three: the outcome
  panel and the two regressor regimes."
- D9: "Decision: the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the
  arm's dimension."
- Stage 1's decision block (`specs/findings/employment-statistics-coverage.md`), which Stage 3
  reads verbatim: "**Branch:** A." … "**Source:** QCEW annual averages, reference years 2022,
  2023, 2024, 2025, private ownership (own_code 5)." … "**Row grain:** a six-digit code in a
  reference year (national, private ownership)." … "**Population:** 980 codes for the seen-code
  regime and 980 for the held-out-code regime, of the 1,012 six-digit codes in the codebook."

### Decisions already made (do not re-ask)

The rulings this stage inherits (the roadmap's D1, D7, D8 and D9, as quoted above):

- The model is ridge with a nested-cross-validated penalty on standardized features, and the
  covariates are log establishments and log wages.
- The outcome is log annual-average employment in t + 1 from year-t features, for feature years
  2022–2024; the seen regime's 2024→2025 rows are sealed.
- Each regime is its own Req 5 panel.
- The text-only comparator is the arm's own backbone, frozen, reduced by PCA to the arm's
  dimension.
- The outer sets are readable only through a logged opening (`SelectionLog` records reads but
  gates none), and the drawn held-out groups are committed under a fingerprint.
- The output is per-row predictions keyed by code, year and group. Only Stage 12 opens a sealed
  set. Stage 4's open questions (D8's tie-break, each panel's decision statistic) are not this
  plan's.

The user answered two more at planning (2026-09-24):

1. **Disjoint partition.** One partition serves both regimes, so no validation read touches
   either sealed set:
   - **Held-out outer set (H):** every row of a code whose employment includes a held-out
     four-digit group, in all three feature years.
   - **Seen outer set (S):** the other codes' 2024 rows (outcome 2025).
   - **Remainder (R):** the other codes' 2022 and 2023 rows. Validation reads only R, and an
     opening reads R and its own outer set.
2. **Strict seal at levels 2–3.** A level-2 or level-3 code joins H when any held-out group rolls
   into it. The held-out regime therefore runs at levels 4–6, where four-digit parents exist. A
   regime-level cell with fewer than 10 remainder groups (`min_groups`) is undefined: it is
   reported, not scored. D8's two regressor panels are the level-6 cells; levels 2–5 are the
   multi-level variant.

This plan's own decisions are stated here so that no reviewer needs to re-derive them:

- **Rows.** A row is a code in a feature year t ∈ {2022, 2023, 2024}. Its covariates are log
  establishments and log total annual wages from year t (D1); its outcome is log annual-average
  employment in t + 1 (D7).
  - Cells are national (`US000`), private (`own_code` 5), all sizes, annual. Sectors 31-33, 44-45
    and 48-49 are keyed 31, 44 and 48, as in the codebook.
  - BLS publishes 19 six-digit NAICS 238 codes only as residential and nonresidential codes; each
    is its five-digit parent's only child, so its cell is read from the parent's row.
  - A cell is usable when disclosed with positive employment, establishments and wages. A level's
    population is the codes usable in all four window years: 980 at six digits, the branch
    record's population, which every load verifies.
- **Held-out draw.** A fifth of each sector's four-digit groups (300 in the six-digit population)
  is held out, by largest-remainder quotas with the total rounded half up.
  - Remainder ties go to the smaller of per-sector draws from `np.random.default_rng([20260924])`
    taken in sorted sector order.
  - Sector s orders its groups with `np.random.default_rng([20260924, int(s)]).permutation` and
    holds out the first ones. The result is 60 groups.
- **Fingerprint.** The sha256 of the table's canonical CSV (a `group` header, then the sorted
  groups), which equals the committed file's hash. Both regressor panels, `regressor_seen` and
  `regressor_heldout`, log under it, because the seen outer set depends on the draw too.
- **Tuning: nested, grouped, inside the remainder.**
  - Seen validation fits the remainder's 2022 rows and scores its 2023 rows. The 2023 rows fall
    into 5 grouped folds, and each fold's penalty has the least error on the other folds' 2023
    rows under the same 2022 fit. 5 repeats redraw the folds.
  - Held-out validation runs 5 repeats of 5 grouped folds over the remainder. Each fold's penalty
    is tuned by 5 grouped inner folds inside its fit part.
  - The test split, after an opening, fits the whole remainder once and scores the regime's outer
    set. Its penalty comes from the remainder only: the forward split (2022 rows fit, 2023 rows
    scored) in the seen regime, and 5 grouped folds over the remainder on their own seed stream
    (1000) in the held-out regime.
- **"The outer set is read once."** For each regime, level, arm and comparator, one fit on the
  remainder scores the outer set in one pass, and no outer row tunes a penalty. `test()` can run
  again after an opening, and every call is logged, as with Stage 2's `OutcomePanel`. Req 4's
  "opened once" is the opening log's job: a second opening needs a reason and is logged as
  `reopen`.
- **Pairing (Req 5).** Fold assignments are seeded by (fold seed 20260924, regime stream 0 or 1,
  level, repeat[, fold]) and depend on the group ids alone, never on the arm. Every arm is
  therefore scored on the same folds.
- **Ridge.** Features are standardized on the fit rows, and a constant column keeps scale 1. The
  grid has 17 penalties from 0.001 to 100,000 in rounded half-decades, and a tie goes to the
  larger penalty. One singular value decomposition per fit gives the whole path, equal to
  scikit-learn's `make_pipeline(StandardScaler(), Ridge(alpha))`.
- **Comparators.** Covariates alone; then the arm's coordinates (`embedding`), one-hot, ancestor
  indicators and the text-only table, each alone and with the covariates. One-hot (level-L
  indicators) runs in the seen regime only. Ancestor indicators cover levels 2 to L − 1, so they
  are absent at level 2.
- **Text-only table (D9).**
  - The backbone is `sentence-transformers/all-MiniLM-L6-v2`, the arm's `model.base_model_name`.
    It runs frozen, on the CPU, from the local Hugging Face cache.
  - It reads the four channels of the arm's descriptions file. Each channel is mean-pooled under
    the attention mask at the arm's `max_length` of 512, and a code's vector is the mean of its
    present channels: an absent channel is masked out.
  - The arm's 24-token title cap (`tokenization_cache.py:74`, Req 9's item for Stage 5) is not
    replicated.
  - The panel reduces the table by PCA, fitted on all 2,125 codes, to the arm's dimension.
- **Coordinate tables.** Every column but `code` and the export's `index` and `level` is a
  coordinate. A table on a hyperboloid (the train prompt's `hyp_e*` Lorentz points) is refused:
  Req 2 takes tangent coordinates at the origin for a hyperbolic arm.
- **Output.** One row per panel, split, level, comparator, repeat, fold (−1 on the test split),
  code and feature year, with `group`, `outcome_year`, the chosen `alpha`, `outcome` and
  `prediction`. `group` is the four-digit parent at levels 4–6 and the code itself at levels 2–3.
- **Selection log.** `SelectionLog` is unchanged, and its `n_queries` field counts rows for the
  regressor panels.
  - A read records the level, the comparators, the arm's and the text-only table's fingerprints
    and the dimension.
  - An opening's `n_queries` is the level-6 outer rows, and its detail lists every loaded level's
    outer rows.
- **Prior art left alone.** `metrics/qcew.py`, the definition Req 2 rejects, is untouched, and so
  is `scripts/employment_statistics_coverage.py`. This plan does not reuse that script's checks,
  so plan 3's deferred rounding fix does not apply.

### Project rules

- **Style (CLAUDE.md).**
  - Single quotes, including `'''` docstrings.
  - YAPF owns layout (100 columns) and ruff lints (E, F, I, Q). **Never run `ruff format`.**
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
  - Commit on this branch only, ending each message with the session's attribution trailer.
- **Data safety.**
  - Never write to the main checkout's `data/`. Read bundle 18403d29's codebook and the pinned
    descriptions file (sha256 5107fb83…) by absolute path only.
  - Never build or rebuild a supervision bundle: only Stage 5 rebuilds.
  - Never run `data regressor-groups --force` once Task 9 has committed the table.
- **Sealed sets.** Never open a real regressor outer set: never run `tools regressor-panel --split
  test`, `RegressorPanel.open_outer` or `RegressorPanel.test` on the real panel. Only Stage 12
  opens them. Openings and test reads run only on fixture panels, in tests. No `open` or
  `reopen` record for `regressor_seen` or `regressor_heldout` may exist in any log.
- **Downloads.** Never download QCEW or Census files or a Hugging Face model. The slices are read
  from `~/Downloads/Data/QCEW/`, and the backbone from the local cache (`local_files_only=True`).
- **Deferred items.** Do not promote any open item of `specs/deferred_items.md`.
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-embedding-roadmap.md` and `specs/deferred_items.md`, and hand the user the exact
  edit instead.
- **Bash tool.** It runs zsh. If it refuses a heredoc or a compound command, run one plain command
  per call and write files with the Write tool.

## Workspace

- **Worktree:** `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-5-regressor-panel`.
  Run every command from its root.
- **Branch:** `claude/plan-5-regressor-panel-635ffd43`, cut from origin/main `7ffe551` (PR #113).
  This plan is its first commit.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder` stays on local `main`, which is
  origin/main plus the held commits "config" and "graph config". Do not check anything out there.
- **Real inputs, read only:**
  - QCEW national slices: `~/Downloads/Data/QCEW/{2022,2023,2024,2025}_US000_annual.csv`
  - Codebook: `/Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet`
  - Descriptions: `/Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`
  - Backbone: `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`
- **Scratch:** `/tmp/stage3-regressor-panel-635ffd43/` (Task 9). Final verification removes it.
- **Working directory:** the Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before Task 9's commands and before every commit, and if it is not
  this worktree, `cd` back first.

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `src/naics_embedder/panels/qcew_rows.py` | Slices, cells, split-code recovery, population, dated rows | 1 |
| `tests/fixtures/regressor_panel.py` | Synthetic tree, cells and slices (Task 1); stub arm tables (Task 6) | 1, 6 |
| `tests/conftest.py` | Registers the fixture module | 1 |
| `src/naics_embedder/panels/regressor_splits.py` | Hierarchy, the held-out draw, the committed table, the partition | 2 |
| `src/naics_embedder/panels/ridge.py` | Standardized ridge path, errors, penalty choice | 3 |
| `src/naics_embedder/panels/text_only.py` | Frozen-backbone encoding, the table and provenance, PCA | 4 |
| `src/naics_embedder/utils/config.py` | `TextOnlyConfig`, `RegressorBranchRecord`, `RegressorPanelConfig` | 5 |
| `conf/data/regressor_panel.yaml` | Shipped config with the branch record | 5 |
| `src/naics_embedder/panels/regressor.py` | `RegressorPanel`, plans, comparators, arm tables, config helpers | 6 |
| `src/naics_embedder/data/regressor_group_table.py` | `generate_regressor_group_table` (`data regressor-groups`) | 7 |
| `src/naics_embedder/cli/commands/data.py` | `data regressor-groups` | 7 |
| `src/naics_embedder/cli/commands/tools.py` | `tools text-only-table`, `tools regressor-panel` | 8 |
| `conf/data/regressor_heldout_groups.csv`, `…_provenance.json` | The committed draw (generated) | 9 |
| `specs/findings/regressor-panel-splits.md` | The real-data finding | 9 |
| `docs/usage.md`, `docs/api/regressor_panel.md`, `docs/.nav.yml`, `CLAUDE.md` | Documentation | 10 |

Tests: `tests/unit/test_regressor_qcew_rows.py` (1), `test_regressor_splits.py` (2),
`test_regressor_ridge.py` (3), `test_text_only.py` (4), `test_regressor_branch_record.py` and
`test_config.py` (5), `test_regressor_panel.py` (6), `test_regressor_group_table.py` and
`test_cli_commands.py` (7), `test_cli_commands.py` (8), `test_committed_regressor_groups.py` (9).

## Expected real-data results

These were verified while writing this plan, with the same code on the same inputs, under Python
3.12 with numpy 2.3.4, polars 1.35.1, scikit-learn 1.9.1, torch 2.9.1 and transformers 4.57.1.
The draw is deterministic, so Task 9 must reproduce it exactly.

| Quantity | Value |
|---|---|
| Population, levels 2 / 3 / 4 / 5 / 6 | 19 / 88 / 300 / 658 / 980 codes (57 / 264 / 900 / 1,974 / 2,940 rows) |
| Six-digit codes read from their five-digit parent | 19 (238110 to 238990) |
| Four-digit groups / held out | 300 / 60 |
| Held out per sector | 11: 4, 21: 1, 22: 1, 23: 2, 31: 17, 42: 4, 44: 5, 48: 6, 51: 2, 52: 2, 53: 2, 54: 2, 56: 2, 61: 1, 62: 3, 71: 2, 72: 1, 81: 3 (55: 0) |
| Group table sha256 (`conf/data/regressor_heldout_groups.csv`, 306 bytes) | `deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4` |
| Rows per split, remainder / seen outer / held-out outer | L2 2 / 1 / 54; L3 92 / 46 / 126; L4 480 / 240 / 180; L5 1,046 / 523 / 405; L6 1,554 / 777 / 609 |
| Undefined cells | seen at L2 (1 remainder group, fewer than 10); held-out at L2 and L3 (no four-digit parent) |
| Text-only table | 2,125 codes × 384; backbone revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; about 33 s |
| Text-only table sha256 on this Mac | `6386f912ea4b37984ed9d4e4100f21fbdc29e91c5ee38acb05bd999342f5bbe0` (floats: not a stop condition elsewhere) |
| Stub validation reads, `n_queries` | 7 reads: seen L3–L6 92 / 480 / 1,046 / 1,554; held-out L4–L6 480 / 1,046 / 1,554 |
| Stub predictions | 179,170 rows; seen L6 3,885 and held-out L6 7,770 per comparator |
| Seen L6, covariates / one_hot / ancestors (RMSE, R²) | 0.3897, 0.9330 / 0.0601, 0.9984 / 0.6059, 0.8381 |
| Held-out L6, covariates / ancestors (RMSE, R²) | 0.3903, 0.9325 / 1.3602, 0.1804 |
| Full suite after Task 9 (Python 3.12) | 1498 passed, 1 skipped |

---

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- A real-data number in Task 9 differs from **Expected real-data results**, above all the group
  table's sha256. Do not commit a different table: the draw is made once, and a redraw moves both
  outer sets. Rows that do not read the arm (covariates, one-hot, ancestors, each alone or with
  the covariates) must match to 4 decimals. The embedding and text-only rows depend on the
  backbone's floats; on this Mac they match too.
- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A step would open a real regressor outer set (`--split test`, `open_outer` or `test` on the real
  panel), write to the main checkout's `data/`, build a supervision bundle, or change
  `conf/config.yaml` or `conf/graph.yaml`.
- `origin/main` gains a commit touching a file in **File structure**, or an open PR does.
- A QCEW slice, the codebook or the descriptions file has another sha256 than the pre-flight's.
  Do not download a replacement.

## Pre-flight (controller, inline, before Task 1)

- [x] **Step 1: Confirm the workspace**

Run: `git status --short --branch`
Expected: `## claude/plan-5-regressor-panel-635ffd43` and nothing else. The branch has no upstream
yet. If the line ends in `...origin/main`, run `git branch --unset-upstream`, so that no command
treats `main` as this branch's remote branch.

Run: `git log --oneline origin/main..HEAD`
Expected: only this plan's commit (`docs(plans): add plan 5 …`). If "config" or "graph config"
appears, stop.

Run: `git fetch origin`, then
`git log --oneline HEAD..origin/main -- src tests conf docs specs CLAUDE.md`
Expected: no output. If anything landed, read it. If it touches a file in **File structure**,
the roadmap or `specs/findings/`, stop and ask.

Run: `gh pr list --state open`
Expected: no open PR touching a file in **File structure**. If one does, stop and ask.

- [x] **Step 2: Build the worktree's environment**

Run: `uv sync`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number. `.python-version` pins 3.12.

Run: `uv run python -c "import numpy, polars, sklearn, torch, transformers; print(numpy.__version__, polars.__version__, sklearn.__version__, torch.__version__, transformers.__version__)"`
Expected: `2.3.4 1.35.1 1.9.1 2.9.1 4.57.1`. These are the locked versions for Python 3.12, and
Task 9's expected results were computed with them. If they differ, stop and ask.

- [x] **Step 3: Run the baseline suite**

> Deviation: the baseline ran with an extra `-p no:cacheprovider`; the count was the same, 1382 passed, 1 skipped.

Run: `uv run pytest -n auto -q`
Expected: `1382 passed, 1 skipped`. Each later full-suite count is this baseline plus the tests the
plan has added by then. The warnings count varies between runs under xdist; ignore it.

- [x] **Step 4: Check the QCEW slices**

Run: `shasum -a 256 ~/Downloads/Data/QCEW/2022_US000_annual.csv ~/Downloads/Data/QCEW/2023_US000_annual.csv ~/Downloads/Data/QCEW/2024_US000_annual.csv ~/Downloads/Data/QCEW/2025_US000_annual.csv`
Expected:

```text
c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18  …/2022_US000_annual.csv
fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9  …/2023_US000_annual.csv
48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46  …/2024_US000_annual.csv
0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811  …/2025_US000_annual.csv
```

These are Stage 1's finding's hashes, and Task 5 pins them in `conf/data/regressor_panel.yaml`.
If any differs, stop and ask. Never download a replacement.

- [x] **Step 5: Check the bundle's codebook and descriptions, read-only**

Run: `shasum -a 256 /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`
Expected:

```text
5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b  …/naics_codebook.parquet
5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d  …/naics_descriptions.parquet
```

The descriptions hash is bundle 18403d29's `description_fingerprint`, so the text-only table
reads the text the current arm was trained on. Task 9 only reads both files. If a hash differs,
stop and ask.

- [x] **Step 6: Check the backbone is cached**

Run: `cat ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/refs/main`
Expected: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Task 9 loads the backbone with
`local_files_only=True`. If the file is missing, stop and ask; do not download the model.

- [x] **Step 7: Route the tasks**

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–8 and 10: each gets a fresh implementer and a task-reviewer. Give each implementer
  its task, **Global Constraints** and **Workspace**.
- Task 9: run it inline in the controller session. It draws the real groups once, checks them
  against **Expected real-data results**, and applies the stop-and-ask conditions.

### Task 1: QCEW national rows

This task reads the QCEW national slices into cells, the population and the dated panel rows
(D7). It also adds the synthetic fixture that every later regressor test uses.

**Files:**

- Create: `src/naics_embedder/panels/qcew_rows.py`
- Create: `tests/fixtures/regressor_panel.py`
- Modify: `tests/conftest.py` (register the fixture module)
- Test: `tests/unit/test_regressor_qcew_rows.py`

**Interfaces:**

- Consumes: `sha256_file(path) -> str` (existing, `naics_embedder.supervision.artifacts`).
- Produces, in `naics_embedder.panels.qcew_rows`:
  - `WINDOW_YEARS = (2022, 2023, 2024, 2025)`, `FEATURE_YEARS = (2022, 2023, 2024)`,
    `PRIVATE = '5'`, `AGGLVL_BY_LEVEL`, `QCEW_SECTOR_CODES`, `SLICE_COLUMNS`, `CELL_COLUMNS`,
    `ROW_COLUMNS`
  - `slice_name(year: int) -> str`
  - `read_national_slice(path: Path) -> pl.DataFrame`
  - `load_national_cells(qcew_dir: Path, expected_sha256: Mapping[str, str]) -> pl.DataFrame`
  - `find_split_codes(published: Collection[str], codebook_six: Sequence[str]) -> Tuple[str, ...]`
  - `level_cells(cells: pl.DataFrame, codebook_codes: Sequence[str], level: int) -> pl.DataFrame`
  - `usable_cell() -> pl.Expr`
  - `population(cells: pl.DataFrame) -> Tuple[str, ...]`
  - `panel_rows(cells: pl.DataFrame, codes: Collection[str]) -> pl.DataFrame`, with columns
    `code`, `feature_year`, `outcome_year`, `log_estabs`, `log_wages` and `outcome`
- Produces, in the fixture module `tests.fixtures.regressor_panel` (a pytest plugin):
  - `GROUPS`, `SPLIT_CODE`, `SUPPRESSED_CODE`, `HELDOUT_GROUPS`, `CODEBOOK`, `POPULATION`,
    `BRANCH_RECORD`
  - `synthetic_cells() -> pl.DataFrame`
  - `write_qcew_slices(directory: Path, cells: pl.DataFrame) -> Dict[str, str]`
  - `synthetic_rows(cells: pl.DataFrame, levels=(2, 3, 4, 5, 6)) -> Dict[int, pl.DataFrame]`
  - the session fixtures `regressor_cells` and `regressor_rows`

- [x] **Step 1: Write the fixture module and the failing test**

Create `tests/fixtures/regressor_panel.py` with exactly this content:

```python
'''
A miniature regressor panel: a synthetic NAICS tree and its QCEW national slices.

Four sectors, one of them combined (31-33), and seventeen four-digit groups. Every group but 2381
has one five-digit code with two six-digit children. 238110 is its five-digit parent's only child,
and BLS publishes it only as 238111 and 238112, so its cell comes from 23811's row. 523211 is
suppressed in 2023, so it leaves the population, as the finding's excluded codes do. Values
follow a noisy size model, so the covariates carry signal.
'''

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels.qcew_rows import (
    SLICE_COLUMNS,
    WINDOW_YEARS,
    level_cells,
    panel_rows,
    population,
    slice_name,
)

GROUPS = (
    '1111',
    '1112',
    '1113',
    '1114',
    '1121',
    '1122',
    '2381',
    '3111',
    '3112',
    '3211',
    '3212',
    '3321',
    '3322',
    '5221',
    '5222',
    '5231',
    '5232',
)
SPLIT_CODE = '238110'
SUPPRESSED_CODE = '523211'
# One held-out group per sector with more than one group: 111, 321 and 523 are tainted at level 3
HELDOUT_GROUPS = ('1113', '3211', '5231')
QCEW_SECTORS = {'31': '31-33'}
AGGLVL = {2: '14', 3: '15', 4: '16', 5: '17', 6: '18'}

def _six_digit(group: str) -> List[str]:
    return [SPLIT_CODE] if group == '2381' else [f'{group}11', f'{group}12']

SIX_DIGIT = tuple(sorted(code for group in GROUPS for code in _six_digit(group)))
FIVE_DIGIT = tuple(sorted({code[:5] for code in SIX_DIGIT}))
SUBSECTORS = tuple(sorted({group[:3] for group in GROUPS}))
SECTORS = ('11', '23', '31', '52')
CODEBOOK = tuple(sorted((*SECTORS, *SUBSECTORS, *GROUPS, *FIVE_DIGIT, *SIX_DIGIT)))
POPULATION = tuple(code for code in SIX_DIGIT if code != SUPPRESSED_CODE)

BRANCH_RECORD = {
    'branch': 'A',
    'source': 'QCEW annual averages',
    'reference_years': list(WINDOW_YEARS),
    'ownership': '5',
    'grain': 'national',
    'population_seen': len(POPULATION),
    'population_heldout': len(POPULATION),
    'time_respecting_outcome': True,
    'seen_regime': True,
    'excluded_codes': [SUPPRESSED_CODE],
}

# -------------------------------------------------------------------------------------------------
# QCEW cells and slices
# -------------------------------------------------------------------------------------------------

def _published(code: str) -> List[Tuple[str, str]]:
    '''The (industry_code, agglvl_code) rows QCEW publishes for a codebook code.'''

    if code == SPLIT_CODE:
        return [(f'{code[:5]}1', AGGLVL[6]), (f'{code[:5]}2', AGGLVL[6])]
    return [(QCEW_SECTORS.get(code, code), AGGLVL[len(code)])]

def synthetic_cells() -> pl.DataFrame:
    '''The private national annual rows ``read_national_slice`` returns, every window year.'''

    rng = np.random.default_rng(20260924)
    rows = []
    for code in CODEBOOK:
        for industry_code, agglvl_code in _published(code):
            size = rng.normal(7.0, 1.5)
            growth = rng.normal(0.02, 0.05)
            for year in WINDOW_YEARS:
                log_emp = size + growth * (year - WINDOW_YEARS[0]) + rng.normal(0.0, 0.05)
                estabs = int(round(np.exp(size - 2.3 + rng.normal(0.0, 0.1))))
                wages = int(round(np.exp(size + 10.8 + rng.normal(0.0, 0.1))))
                cell = ('', estabs, int(round(np.exp(log_emp))), wages)
                if code == SUPPRESSED_CODE and year == 2023:
                    cell = ('N', 0, 0, 0)
                rows.append((industry_code, agglvl_code, year, *cell))
    return pl.DataFrame(
        rows,
        schema={
            'industry_code': pl.Utf8,
            'agglvl_code': pl.Utf8,
            'year': pl.Int32,
            'disclosure_code': pl.Utf8,
            'estabs': pl.Int64,
            'emp': pl.Int64,
            'wages': pl.Int64,
        },
        orient='row',
    )

def _slice_rows(cells: pl.DataFrame, year: int) -> List[Dict[str, str]]:
    rows = []
    for cell in cells.filter(pl.col('year') == year).iter_rows(named=True):
        base = {
            'area_fips': 'US000',
            'own_code': '5',
            'industry_code': cell['industry_code'],
            'agglvl_code': cell['agglvl_code'],
            'size_code': '0',
            'year': str(year),
            'qtr': 'A',
            'disclosure_code': cell['disclosure_code'],
            'annual_avg_estabs': str(cell['estabs']),
            'annual_avg_emplvl': str(cell['emp']),
            'total_annual_wages': str(cell['wages']),
        }
        rows.append(base)
        # Rows the reader must drop: all ownerships, a state, a quarter
        rows.append({**base, 'own_code': '0', 'annual_avg_emplvl': '1'})
        rows.append({**base, 'area_fips': '01000', 'annual_avg_emplvl': '1'})
        rows.append({**base, 'qtr': '1', 'annual_avg_emplvl': '1'})
    return rows

def write_qcew_slices(directory: Path, cells: pl.DataFrame) -> Dict[str, str]:
    '''Write each window year's national slice; return their sha256 pins by file name.'''

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schema = {name: pl.Utf8 for name in SLICE_COLUMNS}
    pins = {}
    for year in WINDOW_YEARS:
        frame = pl.DataFrame(_slice_rows(cells, year), schema=schema)
        path = directory / slice_name(year)
        frame.write_csv(path)
        pins[slice_name(year)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pins

def synthetic_rows(cells: pl.DataFrame, levels=(2, 3, 4, 5, 6)) -> Dict[int, pl.DataFrame]:
    '''Panel rows per level, as ``RegressorPanel.from_sources`` builds them.'''

    rows = {}
    for level in levels:
        cells_at_level = level_cells(cells, CODEBOOK, level)
        rows[level] = panel_rows(cells_at_level, population(cells_at_level))
    return rows

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture(scope='session')
def regressor_cells() -> pl.DataFrame:
    return synthetic_cells()

@pytest.fixture(scope='session')
def regressor_rows(regressor_cells) -> Dict[int, pl.DataFrame]:
    return synthetic_rows(regressor_cells)
```

Create `tests/unit/test_regressor_qcew_rows.py` with exactly this content:

```python
'''
QCEW national rows for the regressor panel (Req 2; roadmap Stage 3; D1; D7).

Every slice here is synthetic (``tests/fixtures/regressor_panel.py``): the real slices stay outside
the repo, under the sha256 values Stage 1's finding records.
'''

import math

import polars as pl
import pytest

from naics_embedder.panels.qcew_rows import (
    FEATURE_YEARS,
    ROW_COLUMNS,
    WINDOW_YEARS,
    find_split_codes,
    level_cells,
    load_national_cells,
    panel_rows,
    population,
    read_national_slice,
    slice_name,
)
from tests.fixtures.regressor_panel import (
    CODEBOOK,
    POPULATION,
    SPLIT_CODE,
    SUPPRESSED_CODE,
    write_qcew_slices,
)

pytestmark = pytest.mark.unit

def test_the_window_gives_three_feature_years_before_their_outcome_years():
    assert WINDOW_YEARS == (2022, 2023, 2024, 2025)
    assert FEATURE_YEARS == (2022, 2023, 2024)
    assert slice_name(2024) == '2024_US000_annual.csv'

def test_a_slice_keeps_only_national_private_all_size_annual_rows(tmp_path, regressor_cells):
    write_qcew_slices(tmp_path, regressor_cells)

    frame = read_national_slice(tmp_path / slice_name(2022))

    assert frame.columns == [
        'industry_code', 'agglvl_code', 'year', 'disclosure_code', 'estabs', 'emp', 'wages'
    ]
    assert frame.height == regressor_cells.filter(pl.col('year') == 2022).height
    assert frame.get_column('year').dtype == pl.Int32
    assert frame.get_column('emp').dtype == pl.Int64
    # The reader drops every row the fixture adds with one employee
    assert frame.get_column('emp').min() > 1

def test_every_slice_is_read_under_its_pinned_sha256(tmp_path, regressor_cells):
    pins = write_qcew_slices(tmp_path, regressor_cells)

    cells = load_national_cells(tmp_path, pins)

    assert sorted(cells.get_column('year').unique().to_list()) == list(WINDOW_YEARS)
    with pytest.raises(ValueError, match='does not match'):
        load_national_cells(tmp_path, {**pins, slice_name(2023): '0' * 64})
    missing = {name: digest for name, digest in pins.items() if name != slice_name(2025)}
    with pytest.raises(ValueError, match='no pinned sha256'):
        load_national_cells(tmp_path, missing)

def test_combined_sectors_are_keyed_as_in_the_codebook(regressor_cells):
    sectors = level_cells(regressor_cells, CODEBOOK, 2)

    assert sorted(sectors.get_column('code').unique().to_list()) == ['11', '23', '31', '52']

def test_a_split_six_digit_code_is_read_from_its_five_digit_parent(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)
    five = level_cells(regressor_cells, CODEBOOK, 5)

    split = six.filter(pl.col('code') == SPLIT_CODE)
    parent = five.filter(pl.col('code') == SPLIT_CODE[:5])
    assert split.get_column('source').unique().to_list() == ['five_digit_parent']
    assert split.select('year', 'emp', 'estabs',
                        'wages').equals(parent.select('year', 'emp', 'estabs', 'wages'))
    assert set(six.filter(pl.col('code') != SPLIT_CODE).get_column('source')) == {'published'}

def test_a_split_code_must_be_an_only_child():
    published = ['238111', '238112']

    assert find_split_codes(published, ['238110']) == ('238110', )
    with pytest.raises(ValueError, match='not an only child'):
        find_split_codes(published, ['238110', '238115'])

def test_levels_outside_two_to_six_are_refused(regressor_cells):
    with pytest.raises(ValueError, match='level must be one of'):
        level_cells(regressor_cells, CODEBOOK, 7)

def test_the_population_needs_a_usable_cell_in_every_window_year(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    assert population(six) == POPULATION
    assert SUPPRESSED_CODE not in population(six)
    zero_wages = six.with_columns(
        wages=pl.when((pl.col('code') == POPULATION[0])
                      & (pl.col('year') == 2025)).then(0).otherwise(pl.col('wages'))
    )
    assert POPULATION[0] not in population(zero_wages)

def test_every_rows_features_are_dated_before_its_outcome(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    rows = panel_rows(six, POPULATION)

    assert rows.columns == list(ROW_COLUMNS)
    assert rows.height == len(POPULATION) * len(FEATURE_YEARS)
    assert (rows.get_column('outcome_year') == rows.get_column('feature_year') + 1).all()
    assert sorted(rows.get_column('feature_year').unique().to_list()) == list(FEATURE_YEARS)
    row = rows.filter((pl.col('code') == POPULATION[0])
                      & (pl.col('feature_year') == 2023)).row(0, named=True)
    cell = {
        year: six.filter((pl.col('code') == POPULATION[0])
                         & (pl.col('year') == year)).row(0, named=True)
        for year in (2023, 2024)
    }
    assert row['log_estabs'] == pytest.approx(math.log(cell[2023]['estabs']))
    assert row['log_wages'] == pytest.approx(math.log(cell[2023]['wages']))
    assert row['outcome'] == pytest.approx(math.log(cell[2024]['emp']))

def test_panel_rows_refuse_a_code_without_every_window_year(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    with pytest.raises(ValueError, match='panel rows for'):
        panel_rows(six, [*POPULATION, SUPPRESSED_CODE])
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_regressor_qcew_rows.py -q`
Expected: one collection error, `ModuleNotFoundError: No module named
'naics_embedder.panels.qcew_rows'`. The rest of the suite is unaffected, because
`tests/conftest.py` does not load the fixture module until Step 3.

- [x] **Step 3: Write the module and register the fixture module**

Create `src/naics_embedder/panels/qcew_rows.py` with exactly this content:

```python
'''
QCEW national rows for the regressor panel (roadmap Stage 3; Req 2; D1; D7).

Stage 1's finding (``specs/findings/employment-statistics-coverage.md``) fixes the source: QCEW
annual averages for 2022–2025, national grain, private ownership (``own_code`` 5). The panel
reads the Open Data Access national slices (``{year}_US000_annual.csv``) under the sha256 values
the finding records; they agree with the annual single files on every national row.

- **Cells.** One cell per code and year. Sectors 31-33, 44-45 and 48-49 are keyed 31, 44 and 48,
  as in the codebook. Each of the 19 six-digit NAICS 238 codes that BLS publishes only as
  residential (``…1``) and nonresidential (``…2``) codes is its five-digit parent's only child, so
  its cell is read from the parent's row.
- **Usable.** A cell is usable when it is disclosed and its employment, establishments and wages
  are all positive. A suppressed cell is never read as zero.
- **Population.** The codes with a usable cell in every window year.
- **Rows (D7).** One row per population code and feature year t in 2022–2024: the covariates are
  log establishments and log total annual wages from year t (D1), and the outcome is log
  annual-average employment in year t + 1.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from collections import Counter
from pathlib import Path
from typing import Collection, Mapping, Sequence, Tuple

import polars as pl

from naics_embedder.supervision.artifacts import sha256_file

WINDOW_YEARS = (2022, 2023, 2024, 2025)
# D7: features from year t, outcome in t + 1, so the last window year is an outcome year only
FEATURE_YEARS = WINDOW_YEARS[:-1]
PRIVATE = '5'
NATIONAL_AREA = 'US000'
AGGLVL_BY_LEVEL = {2: '14', 3: '15', 4: '16', 5: '17', 6: '18'}
QCEW_SECTOR_CODES = {'31-33': '31', '44-45': '44', '48-49': '48'}

SLICE_COLUMNS = (
    'area_fips',
    'own_code',
    'industry_code',
    'agglvl_code',
    'size_code',
    'year',
    'qtr',
    'disclosure_code',
    'annual_avg_estabs',
    'annual_avg_emplvl',
    'total_annual_wages',
)
CELL_COLUMNS = ('code', 'year', 'disclosure_code', 'estabs', 'emp', 'wages', 'source')
ROW_COLUMNS = ('code', 'feature_year', 'outcome_year', 'log_estabs', 'log_wages', 'outcome')

# -------------------------------------------------------------------------------------------------
# Files
# -------------------------------------------------------------------------------------------------

def slice_name(year: int) -> str:
    '''File name of one year's national slice.'''

    return f'{year}_US000_annual.csv'

def read_national_slice(path: Path) -> pl.DataFrame:
    '''
    The private, all-size annual rows of one national slice.

    Returns:
        ``industry_code``, ``agglvl_code``, ``year`` (Int32), ``disclosure_code`` (blank reads as
        ``''``), and ``estabs``, ``emp``, ``wages`` as Int64.
    '''

    frame = pl.read_csv(Path(path), columns=list(SLICE_COLUMNS), infer_schema=False)
    keys = [name for name in SLICE_COLUMNS if not name.startswith(('annual_', 'total_'))]
    # yapf: disable
    return (
        frame
        .with_columns(pl.col(*keys).fill_null('').str.strip_chars())
        .filter(
            (pl.col('area_fips') == NATIONAL_AREA)
            & (pl.col('own_code') == PRIVATE)
            & (pl.col('size_code') == '0')
            & (pl.col('qtr') == 'A')
        )
        .select(
            'industry_code',
            'agglvl_code',
            pl.col('year').cast(pl.Int32),
            'disclosure_code',
            estabs=pl.col('annual_avg_estabs').str.strip_chars().cast(pl.Int64),
            emp=pl.col('annual_avg_emplvl').str.strip_chars().cast(pl.Int64),
            wages=pl.col('total_annual_wages').str.strip_chars().cast(pl.Int64),
        )
    )
    # yapf: enable

def load_national_cells(qcew_dir: Path, expected_sha256: Mapping[str, str]) -> pl.DataFrame:
    '''
    Read every window year's national slice after checking its sha256.

    Raises:
        ValueError: If a slice has no pinned sha256 or its sha256 differs from the pinned one.
    '''

    frames = []
    for year in WINDOW_YEARS:
        name = slice_name(year)
        if name not in expected_sha256:
            raise ValueError(f'no pinned sha256 for {name}')
        path = Path(qcew_dir).expanduser() / name
        digest = sha256_file(path)
        if digest != expected_sha256[name]:
            raise ValueError(f'{path}: sha256 {digest} does not match {expected_sha256[name]}')
        frames.append(read_national_slice(path))
    return pl.concat(frames)

# -------------------------------------------------------------------------------------------------
# Cells, population and rows
# -------------------------------------------------------------------------------------------------

def find_split_codes(published: Collection[str], codebook_six: Sequence[str]) -> Tuple[str, ...]:
    '''
    Codebook six-digit codes that QCEW publishes only as BLS residential and nonresidential codes.

    Raises:
        ValueError: If such a code is not its five-digit parent's only child, so the parent's row
            would not carry the code's own value.
    '''

    published = set(published)
    codebook = set(codebook_six)
    siblings = Counter(code[:5] for code in codebook)
    split = []
    for code in sorted(codebook):
        bls_codes = {code[:5] + '1', code[:5] + '2'} - codebook
        if code in published or not bls_codes & published:
            continue
        if siblings[code[:5]] != 1:
            raise ValueError(f'{code} is split into BLS codes but is not an only child')
        split.append(code)
    return tuple(split)

def level_cells(cells: pl.DataFrame, codebook_codes: Sequence[str], level: int) -> pl.DataFrame:
    '''
    One cell per codebook code of the given level and year.

    Returns:
        ``code``, ``year``, ``disclosure_code``, ``estabs``, ``emp``, ``wages`` and ``source``
        (``published``, or ``five_digit_parent`` for a split six-digit code).
    '''

    if level not in AGGLVL_BY_LEVEL:
        raise ValueError(f'level must be one of {sorted(AGGLVL_BY_LEVEL)}, got {level}')
    level_codes = sorted(code for code in codebook_codes if len(code) == level)
    rows = cells.filter(pl.col('agglvl_code') == AGGLVL_BY_LEVEL[level]).with_columns(
        code=pl.col('industry_code').replace(QCEW_SECTOR_CODES)
    )
    frames = [
        rows.filter(pl.col('code').is_in(level_codes)).with_columns(source=pl.lit('published'))
    ]
    if level == 6:
        split = find_split_codes(rows.get_column('industry_code').unique(), level_codes)
        parents = pl.DataFrame(
            {
                'industry_code': [code[:5] for code in split],
                'split_code': list(split)
            },
            schema={
                'industry_code': pl.Utf8,
                'split_code': pl.Utf8
            },
        )
        five_digit = cells.filter(pl.col('agglvl_code') == AGGLVL_BY_LEVEL[5])
        frames.append(
            five_digit.join(parents, on='industry_code').with_columns(
                code=pl.col('split_code'), source=pl.lit('five_digit_parent')
            )
        )
    level_frame = pl.concat([frame.select(CELL_COLUMNS) for frame in frames])
    if level_frame.select('code', 'year').is_duplicated().any():
        raise ValueError(f'level {level}: more than one national private cell for a code and year')
    return level_frame.sort('code', 'year')

def usable_cell() -> pl.Expr:
    '''Disclosed, with positive employment, establishments and wages.'''

    return (
        (pl.col('disclosure_code') == '') & (pl.col('emp') > 0) & (pl.col('estabs') > 0)
        & (pl.col('wages') > 0)
    )

def population(cells: pl.DataFrame) -> Tuple[str, ...]:
    '''Codes with a usable cell in every window year, sorted.'''

    years = cells.filter(usable_cell() & pl.col('year').is_in(WINDOW_YEARS))
    complete = years.group_by('code').agg(pl.col('year').n_unique().alias('years'))
    codes = complete.filter(pl.col('years') == len(WINDOW_YEARS)).get_column('code')
    return tuple(sorted(codes.to_list()))

def panel_rows(cells: pl.DataFrame, codes: Collection[str]) -> pl.DataFrame:
    '''
    One row per code and feature year (D7): year-t covariates and the year-(t + 1) outcome.

    Raises:
        ValueError: If a code lacks a usable cell in a window year.
    '''

    usable = cells.filter(usable_cell() & pl.col('code').is_in(list(codes)))
    features = usable.filter(pl.col('year').is_in(FEATURE_YEARS)).select(
        'code',
        feature_year=pl.col('year'),
        outcome_year=pl.col('year') + 1,
        log_estabs=pl.col('estabs').cast(pl.Float64).log(),
        log_wages=pl.col('wages').cast(pl.Float64).log(),
    )
    outcomes = usable.select(
        'code', outcome_year=pl.col('year'), outcome=pl.col('emp').cast(pl.Float64).log()
    )
    rows = features.join(outcomes, on=['code', 'outcome_year'], how='inner')
    expected = len(set(codes)) * len(FEATURE_YEARS)
    if rows.height != expected:
        raise ValueError(f'{rows.height:,} panel rows for {expected:,} code-years')
    return rows.select(ROW_COLUMNS).sort('code', 'feature_year')
```

**`tests/conftest.py`, edit 1 of 1.** Replace:

```python
pytest_plugins = ('tests.fixtures.naics_sources', 'tests.fixtures.supervision')
```

with:

```python
pytest_plugins = (
    'tests.fixtures.naics_sources',
    'tests.fixtures.regressor_panel',
    'tests.fixtures.supervision',
)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_regressor_qcew_rows.py -q`
Expected: `10 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1392 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/qcew_rows.py tests/fixtures/regressor_panel.py tests/conftest.py tests/unit/test_regressor_qcew_rows.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/qcew_rows.py tests/fixtures/regressor_panel.py tests/conftest.py tests/unit/test_regressor_qcew_rows.py
git commit -m "feat(panels): read QCEW national rows for the regressor panel"
```

### Task 2: The partition and the held-out draw

This task partitions the panel rows into the remainder and the two outer sets, draws the held-out
four-digit groups, and reads and writes their committed table.

**Files:**

- Create: `src/naics_embedder/panels/regressor_splits.py`
- Test: `tests/unit/test_regressor_splits.py`

**Interfaces:**

- Consumes: `FEATURE_YEARS` (Task 1); `naics_parent_code(code) -> Optional[str]` (existing,
  `naics_embedder.utils.naics_hierarchy`).
- Produces, in `naics_embedder.panels.regressor_splits`:
  - `SEALED_FEATURE_YEAR = 2024`, `REMAINDER_FEATURE_YEARS = (2022, 2023)`, `GROUP_LEVEL = 4`,
    `SECTOR_LEVEL = 2`
  - `RegressorSplit(str, Enum)`: `REMAINDER = 'remainder'`, `SEEN_OUTER = 'seen_outer'`,
    `HELDOUT_OUTER = 'heldout_outer'`
  - `ancestor_at(code: str, level: int) -> str` and `group_of(code: str) -> str`
  - `codes_fingerprint(codes: Collection[str]) -> str` and
    `read_codebook_codes(path: Path, expected_sha256: str) -> Tuple[str, ...]`
  - `sector_quotas(counts: Dict[str, int], fraction: Fraction, seed: int) -> Dict[str, int]`
  - `draw_heldout_groups(six_digit_codes: Collection[str], fraction: Fraction, seed: int)
    -> Tuple[str, ...]`
  - `group_table_fingerprint(groups: Collection[str]) -> str`,
    `write_group_table(groups: Collection[str], path: Path) -> str` (returns the fingerprint) and
    `read_group_table(path: Path) -> Tuple[str, ...]`
  - `heldout_tainted(code: str, heldout_groups: Collection[str]) -> bool`
  - `assign_splits(rows: pl.DataFrame, heldout_groups: Collection[str]) -> pl.DataFrame`, which
    adds `group` and `split`
  - `split_counts(rows: pl.DataFrame) -> Dict[str, int]` and
    `check_partition(rows: pl.DataFrame, codes: Sequence[str]) -> None`

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_regressor_splits.py` with exactly this content:

```python
'''
The regressor panel's partition and its committed held-out draw (Req 2; Req 4; D7).
'''

import hashlib
from collections import Counter
from fractions import Fraction

import polars as pl
import pytest

from naics_embedder.panels.regressor_splits import (
    REMAINDER_FEATURE_YEARS,
    SEALED_FEATURE_YEAR,
    RegressorSplit,
    ancestor_at,
    assign_splits,
    check_partition,
    codes_fingerprint,
    draw_heldout_groups,
    group_of,
    group_table_fingerprint,
    heldout_tainted,
    read_codebook_codes,
    read_group_table,
    sector_quotas,
    split_counts,
    write_group_table,
)
from tests.fixtures.regressor_panel import CODEBOOK, HELDOUT_GROUPS, POPULATION

pytestmark = pytest.mark.unit

# -------------------------------------------------------------------------------------------------
# Hierarchy
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('code', 'level', 'ancestor'),
    [
        ('332111', 2, '31'),
        ('332111', 4, '3321'),
        ('452210', 2, '44'),
        ('493110', 2, '48'),
        ('493110', 6, '493110'),
        ('321', 2, '31'),
    ],
)
def test_ancestors_follow_the_naics_tree(code, level, ancestor):
    assert ancestor_at(code, level) == ancestor

@pytest.mark.parametrize(('code', 'level'), [('332111', 1), ('3321', 5)])
def test_there_is_no_ancestor_outside_the_codes_levels(code, level):
    with pytest.raises(ValueError, match='no ancestor'):
        ancestor_at(code, level)

def test_a_rows_group_is_its_four_digit_ancestor_or_the_code_itself():
    assert group_of('332111') == '3321'
    assert group_of('33211') == '3321'
    assert group_of('3321') == '3321'
    assert group_of('332') == '332'
    assert group_of('31') == '31'

def test_the_seen_outer_set_is_the_last_feature_year():
    assert SEALED_FEATURE_YEAR == 2024
    assert REMAINDER_FEATURE_YEARS == (2022, 2023)

# -------------------------------------------------------------------------------------------------
# The held-out draw
# -------------------------------------------------------------------------------------------------

def test_quotas_are_largest_remainder_shares_of_each_sector():
    # 3.8, 1.0, 0.6 and 0.2 floor to 4 groups; a fifth of 28, rounded, is 6
    quotas = sector_quotas({'11': 19, '21': 5, '22': 3, '55': 1}, Fraction(1, 5), 20260924)

    assert quotas == {'11': 4, '21': 1, '22': 1, '55': 0}

def test_remainder_ties_are_broken_by_the_seed():
    counts = {sector: 1 for sector in ('11', '21', '22', '23', '31')}

    draws = {seed: sector_quotas(counts, Fraction(1, 5), seed) for seed in range(20)}

    assert all(sum(quotas.values()) == 1 for quotas in draws.values())
    assert draws[3] == sector_quotas(counts, Fraction(1, 5), 3)
    assert len({tuple(sorted(quotas.items())) for quotas in draws.values()}) > 1

@pytest.mark.parametrize('fraction', [Fraction(0), Fraction(1), Fraction(3, 2)])
def test_the_fraction_must_lie_strictly_between_zero_and_one(fraction):
    with pytest.raises(ValueError, match='fraction'):
        sector_quotas({'11': 5}, fraction, 1)

def _codes(groups_by_sector):
    return [f'{group}11' for groups in groups_by_sector.values() for group in groups]

SECTOR_GROUPS = {
    '11': [f'11{number:02d}' for number in range(11, 21)],
    '31': ['3111', '3112', '3211', '3212', '3321'],
    '52': ['5211', '5221', '5222', '5231', '5232'],
}

def test_the_draw_is_seeded_and_stratified_by_sector():
    drawn = draw_heldout_groups(_codes(SECTOR_GROUPS), Fraction(1, 5), 20260924)

    assert drawn == draw_heldout_groups(_codes(SECTOR_GROUPS), Fraction(1, 5), 20260924)
    assert drawn == tuple(sorted(drawn))
    by_sector = Counter(ancestor_at(group, 2) for group in drawn)
    assert by_sector == Counter({'11': 2, '31': 1, '52': 1})
    assert draw_heldout_groups(_codes(SECTOR_GROUPS), Fraction(1, 5), 1) != drawn

def test_a_sectors_draw_does_not_depend_on_the_other_sectors():
    both = draw_heldout_groups(_codes(SECTOR_GROUPS), Fraction(1, 5), 20260924)
    alone = draw_heldout_groups(_codes({'31': SECTOR_GROUPS['31']}), Fraction(1, 5), 20260924)

    assert [group for group in both if group.startswith(('31', '32', '33'))] == list(alone)

# -------------------------------------------------------------------------------------------------
# The committed table and the codebook
# -------------------------------------------------------------------------------------------------

def test_the_fingerprint_is_the_committed_files_sha256(tmp_path):
    path = tmp_path / 'groups.csv'

    fingerprint = write_group_table(['5231', '1113', '3211'], path)

    assert fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
    assert fingerprint == group_table_fingerprint(HELDOUT_GROUPS)
    assert read_group_table(path) == HELDOUT_GROUPS
    assert path.read_text() == 'group\n1113\n3211\n5231\n'

@pytest.mark.parametrize('body', ['group\n1113\n1113\n', 'group\n111\n', 'group\n11131\n'])
def test_a_table_of_anything_but_distinct_four_digit_groups_is_refused(tmp_path, body):
    path = tmp_path / 'groups.csv'
    path.write_text(body)

    with pytest.raises(ValueError, match='distinct four-digit'):
        read_group_table(path)

def test_the_codebook_is_read_under_its_codes_fingerprint(tmp_path):
    path = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code_id': range(len(CODEBOOK)), 'code': list(CODEBOOK)}).write_parquet(path)
    expected = hashlib.sha256(''.join(f'{code}\n' for code in CODEBOOK).encode()).hexdigest()

    assert codes_fingerprint(reversed(CODEBOOK)) == expected
    assert read_codebook_codes(path, expected) == CODEBOOK
    with pytest.raises(ValueError, match='does not match'):
        read_codebook_codes(path, '0' * 64)
    repeated = tmp_path / 'repeated.parquet'
    pl.DataFrame({'code': ['11', '11']}).write_parquet(repeated)
    with pytest.raises(ValueError, match='repeats a code'):
        read_codebook_codes(repeated, expected)

# -------------------------------------------------------------------------------------------------
# The partition
# -------------------------------------------------------------------------------------------------

def test_the_partition_is_disjoint_and_covers_every_row(regressor_rows):
    rows = assign_splits(regressor_rows[6], HELDOUT_GROUPS)

    check_partition(rows, POPULATION)
    held = rows.get_column('group').is_in(list(HELDOUT_GROUPS))
    year = rows.get_column('feature_year')
    split = rows.get_column('split')
    assert (split.filter(held) == RegressorSplit.HELDOUT_OUTER.value).all()
    assert (split.filter(~held & (year == 2024)) == RegressorSplit.SEEN_OUTER.value).all()
    assert (split.filter(~held & (year < 2024)) == RegressorSplit.REMAINDER.value).all()
    assert sum(split_counts(rows).values()) == rows.height

def test_a_held_out_group_seals_every_code_it_rolls_into():
    assert heldout_tainted('311', ['3111'])
    assert heldout_tainted('31', ['3211'])
    assert heldout_tainted('311111', ['3111'])
    assert heldout_tainted('31111', ['3111'])
    assert not heldout_tainted('312', ['3111'])
    assert not heldout_tainted('311211', ['3111'])

def test_aggregate_codes_join_the_held_out_set_in_every_year(regressor_rows):
    rows = assign_splits(regressor_rows[3], HELDOUT_GROUPS)

    tainted = rows.filter(pl.col('code').is_in(['111', '321', '523']))
    assert set(tainted.get_column('split')) == {RegressorSplit.HELDOUT_OUTER.value}
    assert tainted.height == 9
    remainder = rows.filter(pl.col('split') == RegressorSplit.REMAINDER.value)
    assert sorted(remainder.get_column('code').unique()) == ['112', '238', '311', '332', '522']

def test_split_counts_name_every_split():
    rows = pl.DataFrame({'split': [RegressorSplit.REMAINDER.value]})

    assert split_counts(rows) == {'remainder': 1, 'seen_outer': 0, 'heldout_outer': 0}

def test_a_missing_or_repeated_code_year_breaks_the_partition(regressor_rows):
    rows = assign_splits(regressor_rows[6], HELDOUT_GROUPS)

    with pytest.raises(ValueError, match='code-years'):
        check_partition(rows.slice(1), POPULATION)
    # As many rows as code-years, but one code-year twice and another missing
    with pytest.raises(ValueError, match='code-years'):
        check_partition(pl.concat([rows.slice(1), rows.slice(1, 1)]), POPULATION)
    with pytest.raises(ValueError, match='no split'):
        check_partition(rows.with_columns(split=pl.lit(None, dtype=pl.Utf8)), POPULATION)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_regressor_splits.py -q`
Expected: one collection error, `ModuleNotFoundError: No module named
'naics_embedder.panels.regressor_splits'`.

- [x] **Step 3: Write the module**

Create `src/naics_embedder/panels/regressor_splits.py` with exactly this content:

```python
'''
The regressor panel's sealed outer sets (Req 2; Req 4; roadmap Stage 3; D7).

Every panel row is a code in a feature year. One partition serves both regimes, so that no
validation read touches either regime's sealed set:

- **Held-out outer set (H).** Every row of a code whose employment includes a held-out four-digit
  group: at levels 4–6 the code's own four-digit ancestor (or the code itself) is held out; at
  levels 2 and 3 a held-out group rolls into the code.
- **Seen outer set (S).** The remaining rows with feature year 2024, whose outcome is 2025 (D7).
- **Remainder (R).** The remaining rows, feature years 2022 and 2023. Validation reads only these.

The held-out groups are drawn once, stratified by sector, and committed
(``conf/data/regressor_heldout_groups.csv``). The split's fingerprint is the sha256 of that
table's canonical CSV, which equals the committed file's hash: a redraw gets a new fingerprint,
so it would not count as a reopening, and the table is therefore never redrawn.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import math
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Collection, Dict, List, Sequence, Tuple

import numpy as np
import polars as pl

from naics_embedder.panels.qcew_rows import FEATURE_YEARS
from naics_embedder.utils.naics_hierarchy import naics_parent_code

# D7: the seen regime's outer set is the last feature year; the remainder keeps the earlier two
SEALED_FEATURE_YEAR = FEATURE_YEARS[-1]
REMAINDER_FEATURE_YEARS = FEATURE_YEARS[:-1]
GROUP_LEVEL = 4
SECTOR_LEVEL = 2
GROUP_TABLE_SCHEMA = {'group': pl.Utf8}

class RegressorSplit(str, Enum):
    '''Which part of the partition a panel row belongs to.'''

    REMAINDER = 'remainder'
    SEEN_OUTER = 'seen_outer'
    HELDOUT_OUTER = 'heldout_outer'

# -------------------------------------------------------------------------------------------------
# Hierarchy
# -------------------------------------------------------------------------------------------------

def ancestor_at(code: str, level: int) -> str:
    '''The code's ancestor at a level (the code itself at its own level); 31-33 is one sector.'''

    if not SECTOR_LEVEL <= level <= len(code):
        raise ValueError(f'{code} has no ancestor at level {level}')
    while len(code) > level:
        parent = naics_parent_code(code)
        if parent is None:
            raise ValueError(f'{code} has no parent')
        code = parent
    return code

def group_of(code: str) -> str:
    '''
    The row's group: its four-digit ancestor at levels 4–6, the code itself at levels 2 and 3.

    Folds and Stage 4's resampling run over groups (Req 5: four-digit-parent groups).
    '''

    return ancestor_at(code, GROUP_LEVEL) if len(code) >= GROUP_LEVEL else code

def codes_fingerprint(codes: Collection[str]) -> str:
    '''SHA-256 of the codes, sorted, one per line: the codebook's content, whatever its file.'''

    return hashlib.sha256(''.join(f'{code}\n'
                                  for code in sorted(codes)).encode('utf-8')).hexdigest()

def read_codebook_codes(path: Path, expected_sha256: str) -> Tuple[str, ...]:
    '''
    A supervision bundle codebook's codes, sorted, after checking their fingerprint.

    Raises:
        ValueError: If a code repeats or the codes' fingerprint differs from ``expected_sha256``.
    '''

    codes = pl.read_parquet(Path(path), columns=['code']).get_column('code').to_list()
    if len(set(codes)) != len(codes):
        raise ValueError(f'{path}: the codebook repeats a code')
    digest = codes_fingerprint(codes)
    if digest != expected_sha256:
        raise ValueError(f'{path}: codes fingerprint {digest} does not match {expected_sha256}')
    return tuple(sorted(codes))

# -------------------------------------------------------------------------------------------------
# The held-out draw
# -------------------------------------------------------------------------------------------------

def sector_quotas(counts: Dict[str, int], fraction: Fraction, seed: int) -> Dict[str, int]:
    '''
    Largest-remainder quotas of ``fraction`` of each sector's groups.

    The total is ``fraction`` of all groups, rounded half up; remainder ties go to the smaller of
    per-sector draws from ``np.random.default_rng([seed])``, taken in sorted sector order.
    '''

    if not 0 < fraction < 1:
        raise ValueError(f'fraction must lie in (0, 1), got {fraction}')
    sectors = sorted(counts)
    exact = {sector: fraction * counts[sector] for sector in sectors}
    quotas = {sector: math.floor(exact[sector]) for sector in sectors}
    total = math.floor(fraction * sum(counts.values()) + Fraction(1, 2))
    tie_break = dict(zip(sectors, np.random.default_rng([seed]).random(len(sectors))))
    order = sorted(
        sectors, key=lambda sector: (-(exact[sector] - quotas[sector]), tie_break[sector])
    )
    for sector in order[:total - sum(quotas.values())]:
        quotas[sector] += 1
    return quotas

def draw_heldout_groups(six_digit_codes: Collection[str], fraction: Fraction,
                        seed: int) -> Tuple[str, ...]:
    '''
    Draw the held-out four-digit groups, stratified by sector.

    Args:
        six_digit_codes: The six-digit population (Stage 1's 980 codes).
        fraction: Share of each sector's groups to hold out (largest-remainder quotas).
        seed: Base seed; sector s draws its groups with ``np.random.default_rng([seed, int(s)])``.

    Returns:
        The held-out groups, sorted.
    '''

    by_sector: Dict[str, List[str]] = {}
    for group in sorted({ancestor_at(code, GROUP_LEVEL) for code in six_digit_codes}):
        by_sector.setdefault(ancestor_at(group, SECTOR_LEVEL), []).append(group)
    quotas = sector_quotas(
        {
            sector: len(groups)
            for sector, groups in by_sector.items()
        }, fraction, seed
    )
    drawn: List[str] = []
    for sector in sorted(by_sector):
        order = np.random.default_rng([seed, int(sector)]).permutation(len(by_sector[sector]))
        drawn.extend(by_sector[sector][index] for index in order[:quotas[sector]])
    return tuple(sorted(drawn))

# -------------------------------------------------------------------------------------------------
# The committed table
# -------------------------------------------------------------------------------------------------

def _group_table_csv(groups: Collection[str]) -> bytes:
    frame = pl.DataFrame({'group': sorted(groups)}, schema=GROUP_TABLE_SCHEMA)
    return frame.write_csv().encode('utf-8')

def group_table_fingerprint(groups: Collection[str]) -> str:
    '''SHA-256 of the held-out groups' canonical CSV, which equals the committed file's hash.'''

    return hashlib.sha256(_group_table_csv(groups)).hexdigest()

def write_group_table(groups: Collection[str], path: Path) -> str:
    '''Write the canonical CSV and return its sha256 (``group_table_fingerprint``).'''

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_group_table_csv(groups))
    return group_table_fingerprint(groups)

def read_group_table(path: Path) -> Tuple[str, ...]:
    '''
    Read the committed held-out groups.

    Raises:
        ValueError: If a group is not a distinct four-digit code.
    '''

    groups = pl.read_csv(Path(path), schema=GROUP_TABLE_SCHEMA).get_column('group').to_list()
    if len(set(groups)) != len(groups) or any(len(group) != GROUP_LEVEL for group in groups):
        raise ValueError(f'{path}: held-out groups must be distinct four-digit codes')
    return tuple(sorted(groups))

# -------------------------------------------------------------------------------------------------
# The partition
# -------------------------------------------------------------------------------------------------

def heldout_tainted(code: str, heldout_groups: Collection[str]) -> bool:
    '''Whether the code's employment includes a held-out group.'''

    if len(code) >= GROUP_LEVEL:
        return ancestor_at(code, GROUP_LEVEL) in set(heldout_groups)
    return any(ancestor_at(group, len(code)) == code for group in heldout_groups)

def assign_splits(rows: pl.DataFrame, heldout_groups: Collection[str]) -> pl.DataFrame:
    '''
    Add each row's ``group`` and ``split`` (``RegressorSplit``) to panel rows.

    Args:
        rows: Panel rows with ``code`` and ``feature_year``.
        heldout_groups: The committed held-out four-digit groups.
    '''

    codes = sorted(set(rows.get_column('code').to_list()))
    held = set(heldout_groups)
    info = pl.DataFrame(
        {
            'code': codes,
            'group': [group_of(code) for code in codes],
            'tainted': [heldout_tainted(code, held) for code in codes],
        },
        schema={
            'code': pl.Utf8,
            'group': pl.Utf8,
            'tainted': pl.Boolean
        },
    )
    # yapf: disable
    split = (
        pl.when(pl.col('tainted')).then(pl.lit(RegressorSplit.HELDOUT_OUTER.value))
        .when(pl.col('feature_year') == SEALED_FEATURE_YEAR)
        .then(pl.lit(RegressorSplit.SEEN_OUTER.value))
        .otherwise(pl.lit(RegressorSplit.REMAINDER.value))
    )
    # yapf: enable
    return rows.join(info, on='code', how='left').with_columns(split=split).drop('tainted')

def split_counts(rows: pl.DataFrame) -> Dict[str, int]:
    '''Rows per split, every split named (zero when empty).'''

    counts = dict(rows.group_by('split').len().iter_rows())
    return {split.value: int(counts.get(split.value, 0)) for split in RegressorSplit}

def check_partition(rows: pl.DataFrame, codes: Sequence[str]) -> None:
    '''
    Require exactly one row per code and feature year, each in exactly one split.

    Raises:
        ValueError: If a code-year is missing, repeated, or has no split.
    '''

    expected = len(set(codes)) * len(FEATURE_YEARS)
    if rows.height != expected or rows.select('code', 'feature_year').is_duplicated().any():
        raise ValueError(f'{rows.height:,} rows for {expected:,} code-years')
    if rows.get_column('split').is_null().any():
        raise ValueError('a row has no split')
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_regressor_splits.py -q`
Expected: `27 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1419 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/regressor_splits.py tests/unit/test_regressor_splits.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/regressor_splits.py tests/unit/test_regressor_splits.py
git commit -m "feat(panels): partition regressor rows around a committed held-out draw"
```

### Task 3: Ridge on standardized features

This task fits ridge along the whole penalty grid from one singular value decomposition and picks
the penalty.

**Files:**

- Create: `src/naics_embedder/panels/ridge.py`
- Test: `tests/unit/test_regressor_ridge.py`

**Interfaces:**

- Consumes: nothing from earlier tasks.
- Produces, in `naics_embedder.panels.ridge`:
  - `standardized_ridge_path(x_fit, y_fit, x_score, alphas) -> np.ndarray`: predictions of shape
    (score rows, penalties)
  - `squared_errors(y_true, predictions) -> np.ndarray`: summed squared error per penalty
  - `best_alpha_index(errors) -> int`: the least error, a tie going to the larger penalty

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_regressor_ridge.py` with exactly this content:

```python
'''
Ridge on standardized features along a penalty grid (Req 2 "Fitting"; D1).
'''

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from naics_embedder.panels.ridge import (
    best_alpha_index,
    squared_errors,
    standardized_ridge_path,
)

pytestmark = pytest.mark.unit

ALPHAS = [0.001, 0.1, 1.0, 10.0, 1000.0]

def _data(n=40, m=12, p=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n + m, p)) * rng.uniform(0.1, 10.0, size=p) + rng.normal(size=p)
    y = x @ rng.normal(size=p) + rng.normal(scale=0.5, size=n + m)
    return x[:n], y[:n], x[n:]

def _sklearn(x_fit, y_fit, x_score, alpha):
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return model.fit(x_fit, y_fit).predict(x_score)

def test_the_path_matches_scikit_learns_standardized_ridge_at_every_penalty():
    x_fit, y_fit, x_score = _data()

    path = standardized_ridge_path(x_fit, y_fit, x_score, ALPHAS)

    assert path.shape == (len(x_score), len(ALPHAS))
    for column, alpha in enumerate(ALPHAS):
        np.testing.assert_allclose(
            path[:, column], _sklearn(x_fit, y_fit, x_score, alpha), rtol=1e-9, atol=1e-9
        )

def test_more_columns_than_rows_still_matches():
    x_fit, y_fit, x_score = _data(n=8, p=20)

    path = standardized_ridge_path(x_fit, y_fit, x_score, ALPHAS)

    for column, alpha in enumerate(ALPHAS):
        np.testing.assert_allclose(
            path[:, column], _sklearn(x_fit, y_fit, x_score, alpha), rtol=1e-9, atol=1e-9
        )

def test_a_constant_column_keeps_scale_one():
    x_fit, y_fit, x_score = _data()
    x_fit[:, 2] = 3.0

    path = standardized_ridge_path(x_fit, y_fit, x_score, [1.0])

    assert np.isfinite(path).all()
    np.testing.assert_allclose(path[:, 0], _sklearn(x_fit, y_fit, x_score, 1.0), atol=1e-9)

def test_a_huge_penalty_predicts_the_fit_mean():
    x_fit, y_fit, x_score = _data()

    path = standardized_ridge_path(x_fit, y_fit, x_score, [1e12])

    np.testing.assert_allclose(path[:, 0], y_fit.mean(), atol=1e-6)

def test_squared_errors_are_summed_per_penalty():
    predictions = np.array([[1.0, 2.0], [3.0, 3.0]])

    np.testing.assert_allclose(squared_errors(np.array([1.0, 1.0]), predictions), [4.0, 5.0])

def test_a_tie_goes_to_the_larger_penalty():
    assert best_alpha_index(np.array([3.0, 1.0, 1.0, 2.0])) == 2
    assert best_alpha_index(np.array([0.5, 1.0, 2.0])) == 0

@pytest.mark.parametrize(
    ('x_fit', 'y_fit', 'x_score', 'alphas', 'message'),
    [
        (np.ones((4, 2)), np.ones(4), np.ones((2, 3)), [1.0], 'shapes'),
        (np.ones((1, 2)), np.ones(1), np.ones((2, 2)), [1.0], 'fit rows'),
        (np.ones((4, 2)), np.ones(3), np.ones((2, 2)), [1.0], 'fit rows'),
        (np.ones((4, 2)), np.ones(4), np.ones((2, 2)), [], 'positive penalties'),
        (np.ones((4, 2)), np.ones(4), np.ones((2, 2)), [0.0], 'positive penalties'),
    ],
)
def test_bad_inputs_are_refused(x_fit, y_fit, x_score, alphas, message):
    with pytest.raises(ValueError, match=message):
        standardized_ridge_path(x_fit, y_fit, x_score, alphas)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_regressor_ridge.py -q`
Expected: one collection error, `ModuleNotFoundError: No module named
'naics_embedder.panels.ridge'`.

- [x] **Step 3: Write the module**

Create `src/naics_embedder/panels/ridge.py` with exactly this content:

```python
'''
Ridge regression on standardized features along a penalty grid (Req 2 "Fitting"; D1).

``standardized_ridge_path`` matches scikit-learn's ``make_pipeline(StandardScaler(),
Ridge(alpha))`` at every penalty on the grid, from one singular value decomposition of the
standardized fit set. The mean and scale come from the fit rows only, and a constant column
keeps scale 1, as ``StandardScaler`` does.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Sequence

import numpy as np

# -------------------------------------------------------------------------------------------------
# Path and selection
# -------------------------------------------------------------------------------------------------

def standardized_ridge_path(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_score: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    '''
    Predictions for the score rows at every penalty.

    Args:
        x_fit: Fit features (n, p).
        y_fit: Fit outcomes (n,).
        x_score: Features to predict (m, p).
        alphas: Penalties (A,), each positive.

    Returns:
        Predictions (m, A), column a at ``alphas[a]``.
    '''

    x_fit = np.asarray(x_fit, dtype=np.float64)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    x_score = np.asarray(x_score, dtype=np.float64)
    grid = np.asarray(alphas, dtype=np.float64)
    if x_fit.ndim != 2 or x_score.ndim != 2 or x_fit.shape[1] != x_score.shape[1]:
        raise ValueError(f'feature shapes {x_fit.shape} and {x_score.shape} do not match')
    if x_fit.shape[0] != y_fit.shape[0] or x_fit.shape[0] < 2:
        raise ValueError(f'{x_fit.shape[0]} fit rows for {y_fit.shape[0]} outcomes')
    if grid.ndim != 1 or not grid.size or (grid <= 0).any():
        raise ValueError('alphas must be a non-empty list of positive penalties')

    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 10 * np.finfo(np.float64).eps] = 1.0
    z_fit = (x_fit - mean) / scale
    z_score = (x_score - mean) / scale
    y_mean = y_fit.mean()

    u, singular, vt = np.linalg.svd(z_fit, full_matrices=False)
    projected = u.T @ (y_fit - y_mean)
    shrink = singular[:, None] / (singular[:, None]**2 + grid[None, :])
    coefficients = vt.T @ (shrink * projected[:, None])
    return z_score @ coefficients + y_mean

def squared_errors(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    '''Summed squared error per penalty: ``y_true`` (m,) against ``predictions`` (m, A).'''

    residuals = np.asarray(predictions) - np.asarray(y_true, dtype=np.float64)[:, None]
    return (residuals**2).sum(axis=0)

def best_alpha_index(errors: np.ndarray) -> int:
    '''The penalty with the least error; a tie goes to the larger penalty (more shrinkage).'''

    errors = np.asarray(errors, dtype=np.float64)
    return int(len(errors) - 1 - np.argmin(errors[::-1]))
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_regressor_ridge.py -q`
Expected: `11 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1430 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/ridge.py tests/unit/test_regressor_ridge.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/ridge.py tests/unit/test_regressor_ridge.py
git commit -m "feat(panels): add ridge on standardized features along a penalty grid"
```

### Task 4: The text-only comparator's table

This task embeds every code's text with a frozen backbone (D9), writes the table with its
provenance, and reduces tables by PCA. Its tests use a one-layer BERT built from a config, so they
run offline.

**Files:**

- Create: `src/naics_embedder/panels/text_only.py`
- Test: `tests/unit/test_text_only.py`

**Interfaces:**

- Consumes: `sha256_file(path) -> str` (existing, `naics_embedder.supervision.artifacts`).
- Produces, in `naics_embedder.panels.text_only`:
  - `CHANNELS = ('title', 'description', 'excluded', 'examples')`, `POOLING`,
    `TEXT_ONLY_PREFIX = 't'`
  - `encode_code_texts(descriptions, model, tokenizer, *, max_length: int, batch_size: int = 32)
    -> np.ndarray`: one float64 row per code, in `descriptions` order
  - `load_backbone(name: str, *, local_files_only: bool = True) -> Tuple[model, tokenizer,
    revision]`
  - `text_only_frame(codes, vectors) -> pl.DataFrame`: `code` plus `t0` … `t{h-1}`
  - `provenance_path(table_path: Path) -> Path`: `<stem>_provenance.json` beside the table
  - `build_text_only_table(descriptions_path, output_path, *, backbone: str, max_length: int,
    batch_size: int = 32, model=None, tokenizer=None, revision=None) -> Path`
  - `pca_reduce(vectors: np.ndarray, dimension: int) -> np.ndarray`

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_text_only.py` with exactly this content:

```python
'''
The text-only comparator (Req 2; roadmap D9): the arm's backbone, frozen, reading the arm's text.

A one-layer BERT with a seventeen-word vocabulary stands in for the backbone, so these run
offline and fast; the real table uses the backbone from the local Hugging Face cache.
'''

import hashlib
import json

import numpy as np
import polars as pl
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from naics_embedder.panels import text_only
from naics_embedder.panels.text_only import (
    CHANNELS,
    build_text_only_table,
    encode_code_texts,
    pca_reduce,
    provenance_path,
)

pytestmark = pytest.mark.unit

WORDS = [
    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'soybean', 'farming', 'grows', 'soybeans',
    'oilseed', 'canola', 'cattle', 'raises', 'not', 'here', ',', '.'
]

@pytest.fixture(scope='module')
def tokenizer(tmp_path_factory):
    vocab = tmp_path_factory.mktemp('backbone') / 'vocab.txt'
    vocab.write_text('\n'.join(WORDS) + '\n')
    return BertTokenizerFast(vocab_file=str(vocab))

@pytest.fixture(scope='module')
def model():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
    )
    return BertModel(config)

def _descriptions(rows):
    return pl.DataFrame(
        rows,
        schema={
            'code': pl.Utf8,
            **{
                channel: pl.Utf8
                for channel in CHANNELS
            }
        },
        orient='row',
    )

def _encode(frame, model, tokenizer, batch_size=32):
    return encode_code_texts(frame, model, tokenizer, max_length=16, batch_size=batch_size)

def test_a_codes_vector_is_the_mean_of_its_present_channels(model, tokenizer):
    title = _encode(
        _descriptions([('111110', 'soybean farming', None, None, None)]), model, tokenizer
    )
    description = _encode(
        _descriptions([('111110', None, 'grows soybeans', None, None)]), model, tokenizer
    )

    # A blank channel is absent: masked out, never encoded as a placeholder
    both = _encode(
        _descriptions([('111110', 'soybean farming', 'grows soybeans', '   ', None)]), model,
        tokenizer
    )

    np.testing.assert_allclose(both, (title + description) / 2, atol=1e-6)

def test_the_batch_size_does_not_change_the_vectors(model, tokenizer):
    frame = _descriptions(
        [
            ('111110', 'soybean farming', 'grows soybeans here .', None, 'soybean'),
            ('111120', 'oilseed farming', 'grows canola , not soybeans .', 'soybean farming', None),
            ('112111', 'cattle', 'raises cattle', None, None),
        ]
    )

    one = _encode(frame, model, tokenizer, batch_size=1)
    many = _encode(frame, model, tokenizer, batch_size=32)

    assert one.shape == (3, 8)
    assert one.dtype == np.float64
    np.testing.assert_allclose(one, many, atol=1e-5)

def test_a_code_with_no_text_is_refused(model, tokenizer):
    frame = _descriptions(
        [('111110', 'soybean farming', None, None, None), ('111120', None, ' ', None, None)]
    )

    with pytest.raises(ValueError, match='at least one present text channel'):
        _encode(frame, model, tokenizer)

def test_the_backbone_stays_frozen(model, tokenizer):
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.train()

    _encode(_descriptions([('111110', 'soybean farming', None, None, None)]), model, tokenizer)

    assert not model.training
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())

def test_the_table_and_its_provenance_are_written(tmp_path, model, tokenizer):
    descriptions = tmp_path / 'naics_descriptions.parquet'
    _descriptions(
        [
            ('111120', 'oilseed farming', 'grows canola', None, None),
            ('111110', 'soybean farming', 'grows soybeans', None, 'soybean'),
        ]
    ).write_parquet(descriptions)
    output = tmp_path / 'text_only.parquet'

    path = build_text_only_table(
        descriptions,
        output,
        backbone='tiny-bert',
        max_length=16,
        batch_size=1,
        model=model,
        tokenizer=tokenizer,
        revision='abc123',
    )

    table = pl.read_parquet(path)
    assert table.columns == ['code', *[f't{index}' for index in range(8)]]
    assert table.get_column('code').to_list() == ['111110', '111120']
    provenance = json.loads(provenance_path(path).read_text())
    assert provenance_path(path).name == 'text_only_provenance.json'
    assert provenance['backbone'] == 'tiny-bert'
    assert provenance['revision'] == 'abc123'
    assert provenance['channels'] == list(CHANNELS)
    assert provenance['max_length'] == 16
    assert (provenance['codes'], provenance['hidden_size']) == (2, 8)
    assert provenance['table_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert provenance['descriptions']['sha256'] == hashlib.sha256(descriptions.read_bytes()
                                                                  ).hexdigest()
    assert set(provenance['library_versions']) == {'torch', 'transformers', 'polars'}

def test_the_backbone_is_read_from_the_local_cache_only(monkeypatch, model, tokenizer):
    import transformers

    calls = []

    def fake(loaded):

        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return loaded

        return from_pretrained

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', fake(tokenizer))
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', fake(model))
    model.config._commit_hash = 'cafe'

    loaded_model, loaded_tokenizer, revision = text_only.load_backbone('some/backbone')

    assert calls == [('some/backbone', {'local_files_only': True})] * 2
    assert (loaded_model, loaded_tokenizer, revision) == (model, tokenizer, 'cafe')
    assert not loaded_model.training

def test_pca_reduces_to_the_arms_dimension():
    vectors = np.random.default_rng(0).normal(size=(10, 6))

    assert pca_reduce(vectors, 3).shape == (10, 3)
    for dimension in (0, 7, 11):
        with pytest.raises(ValueError, match='cannot reduce'):
            pca_reduce(vectors, dimension)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_text_only.py -q`
Expected: one collection error, `ImportError: cannot import name 'text_only' from
'naics_embedder.panels'`.

- [x] **Step 3: Write the module**

Create `src/naics_embedder/panels/text_only.py` with exactly this content:

```python
'''
The text-only comparator of the regressor panel (Req 2; roadmap D9).

D9: "the arm's own backbone, frozen, embedding each code's text, reduced by PCA to the arm's
dimension". The table holds one vector per code at the backbone's hidden size; the panel reduces
it to the arm's dimension when it scores the arm.

- **Text.** The four channels the arm reads (``title``, ``description``, ``excluded``,
  ``examples``) from the arm's own descriptions file.
- **Pooling.** Each channel is mean-pooled over its tokens under the attention mask, as the arm's
  encoder pools (``text_model/encoder.py``). A code's vector is the mean over its present
  channels: an absent (null or blank) channel is masked out, never encoded as a placeholder.
- **Frozen.** The backbone runs in evaluation mode under ``torch.no_grad``, on the CPU in
  float32, with no adapter.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA

from naics_embedder.supervision.artifacts import sha256_file

logger = logging.getLogger(__name__)

CHANNELS = ('title', 'description', 'excluded', 'examples')
POOLING = 'attention-masked mean over tokens per channel, then the mean over present channels'
TEXT_ONLY_PREFIX = 't'

# -------------------------------------------------------------------------------------------------
# Encoding
# -------------------------------------------------------------------------------------------------

def _present(text: Optional[str]) -> bool:
    return text is not None and bool(text.strip())

def encode_code_texts(
    descriptions: pl.DataFrame,
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    max_length: int,
    batch_size: int = 32,
) -> np.ndarray:
    '''
    Embed every code's text with a frozen backbone.

    Args:
        descriptions: One row per code with the four channel columns.
        model: A Hugging Face encoder returning ``last_hidden_state``.
        tokenizer: Its tokenizer.
        max_length: Tokens kept per channel text.
        batch_size: Texts per forward pass.

    Returns:
        One row per code, in ``descriptions`` order: the mean of its present channels'
        mean-pooled vectors (float64).

    Raises:
        ValueError: If a code has no present channel.
    '''

    model.eval()
    total: Optional[np.ndarray] = None
    counts = np.zeros(descriptions.height, dtype=np.int64)
    for channel in CHANNELS:
        texts = descriptions.get_column(channel).to_list()
        present = [index for index, text in enumerate(texts) if _present(text)]
        for start in range(0, len(present), batch_size):
            batch = present[start:start + batch_size]
            tokens = tokenizer(
                [texts[index] for index in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            )
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors = pooled.to(torch.float64).numpy()
            if total is None:
                total = np.zeros((descriptions.height, vectors.shape[1]), dtype=np.float64)
            total[batch] += vectors
            counts[batch] += 1
    if total is None or (counts == 0).any():
        raise ValueError('every code needs at least one present text channel')
    return total / counts[:, None]

def load_backbone(name: str, *, local_files_only: bool = True) -> Tuple[Any, Any, Optional[str]]:
    '''
    The backbone's model and tokenizer, from the local Hugging Face cache by default.

    Returns:
        ``(model, tokenizer, revision)``; ``revision`` is the resolved snapshot's commit hash.
    '''

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(name, local_files_only=local_files_only)
    return model.to('cpu').eval(), tokenizer, getattr(model.config, '_commit_hash', None)

# -------------------------------------------------------------------------------------------------
# The table
# -------------------------------------------------------------------------------------------------

def text_only_frame(codes: List[str], vectors: np.ndarray) -> pl.DataFrame:
    '''``code`` plus ``t0`` … ``t{h-1}`` (float64).'''

    schema = {f'{TEXT_ONLY_PREFIX}{index}': pl.Float64 for index in range(vectors.shape[1])}
    values = pl.DataFrame(vectors, schema=schema, orient='row')
    return pl.DataFrame({'code': codes}, schema={'code': pl.Utf8}).hstack(values)

def provenance_path(table_path: Path) -> Path:
    '''The provenance JSON written beside a text-only table.'''

    table_path = Path(table_path)
    return table_path.with_name(f'{table_path.stem}_provenance.json')

def build_text_only_table(
    descriptions_path: Path,
    output_path: Path,
    *,
    backbone: str,
    max_length: int,
    batch_size: int = 32,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Any = None,
    revision: Optional[str] = None,
) -> Path:
    '''
    Embed every code's text with the frozen backbone and write the table and its provenance.

    ``model`` and ``tokenizer`` default to ``load_backbone(backbone)``; tests pass small ones.

    Returns:
        The table's path.
    '''

    descriptions_path = Path(descriptions_path)
    descriptions = pl.read_parquet(descriptions_path).sort('code')
    if model is None or tokenizer is None:
        model, tokenizer, revision = load_backbone(backbone)
    vectors = encode_code_texts(
        descriptions, model, tokenizer, max_length=max_length, batch_size=batch_size
    )
    table = text_only_frame(descriptions.get_column('code').to_list(), vectors)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(output_path)

    provenance: Dict[str, Any] = {
        'backbone': backbone,
        'revision': revision,
        'descriptions': {
            'path': str(descriptions_path),
            'sha256': sha256_file(descriptions_path)
        },
        'channels': list(CHANNELS),
        'pooling': POOLING,
        'max_length': max_length,
        'codes': table.height,
        'hidden_size': vectors.shape[1],
        'table_sha256': sha256_file(output_path),
        'library_versions': {
            name: version(name)
            for name in ('torch', 'transformers', 'polars')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    provenance_path(output_path).write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    logger.info(
        f'Text-only table ({table.height:,} codes, width {vectors.shape[1]}): {output_path}'
    )
    return output_path

# -------------------------------------------------------------------------------------------------
# Reduction
# -------------------------------------------------------------------------------------------------

def pca_reduce(vectors: np.ndarray, dimension: int) -> np.ndarray:
    '''
    Reduce the vectors to ``dimension`` principal components (D9: the arm's dimension).

    Raises:
        ValueError: If ``dimension`` exceeds the number of vectors or their width.
    '''

    vectors = np.asarray(vectors, dtype=np.float64)
    if not 1 <= dimension <= min(vectors.shape):
        raise ValueError(
            f'cannot reduce {vectors.shape[0]:,} vectors of width {vectors.shape[1]} '
            f'to {dimension} components'
        )
    return PCA(n_components=dimension, svd_solver='full').fit_transform(vectors)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_text_only.py -q`
Expected: `7 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1437 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/text_only.py tests/unit/test_text_only.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/text_only.py tests/unit/test_text_only.py
git commit -m "feat(panels): build the text-only comparator's table from a frozen backbone"
```

### Task 5: The regressor panel's config and branch record

This task declares the panel's config, ships it with the branch record of Stage 1's finding, and
tests the record against the finding's decision block, read from the committed finding.

**Files:**

- Modify: `src/naics_embedder/utils/config.py` (after `OutcomePanelConfig`)
- Create: `conf/data/regressor_panel.yaml`
- Modify: `tests/unit/test_config.py`
- Test: `tests/unit/test_regressor_branch_record.py`

**Interfaces:**

- Consumes: `BRANCH_RECORD` (Task 1's fixture module); the finding
  `specs/findings/employment-statistics-coverage.md` (existing).
- Produces, in `naics_embedder.utils.config`:
  - `QCEW_SLICE_SHA256` (the four slices' pins) and `RIDGE_ALPHAS` (17 penalties, ascending)
  - `TextOnlyConfig`: `backbone`, `max_length`, `batch_size`
  - `RegressorBranchRecord`: `branch`, `source`, `reference_years`, `ownership`, `grain`,
    `population_seen`, `population_heldout`, `time_respecting_outcome`, `seen_regime`,
    `excluded_codes`, with no defaults
  - `RegressorPanelConfig`: `qcew_dir`, `qcew_sha256`, `codebook_codes_sha256`,
    `heldout_groups_csv`, `provenance_json`, `seed`, `heldout_fraction`, `alphas`, `folds`,
    `repeats`, `inner_folds`, `fold_seed`, `min_groups`, `text_only`, `selection_log`,
    `branch_record` (optional)
- Produces: `conf/data/regressor_panel.yaml`, read with
  `load_config(RegressorPanelConfig, 'data/regressor_panel.yaml')`.

- [x] **Step 1: Write the failing tests**

Modify `tests/unit/test_config.py` with these 2 edits, in order. Each replaced text occurs exactly
once in the file.

**`tests/unit/test_config.py`, edit 1 of 2.** Replace:

```python
    GraphConfig,
    OutcomePanelConfig,
    SamplingConfig,
    SansStaticConfig,
    StructuralPreferenceConfig,
    SupervisionBuildConfig,
    SupervisionRuntimeConfig,
    load_config,
)
```

with:

```python
    GraphConfig,
    OutcomePanelConfig,
    RegressorBranchRecord,
    RegressorPanelConfig,
    SamplingConfig,
    SansStaticConfig,
    StructuralPreferenceConfig,
    SupervisionBuildConfig,
    SupervisionRuntimeConfig,
    load_config,
)
from tests.fixtures.regressor_panel import BRANCH_RECORD
```

**`tests/unit/test_config.py`, edit 2 of 2.** Replace:

```python
            OutcomePanelConfig(test_fraction=0.15)

# -------------------------------------------------------------------------------------------------
```

with:

```python
            OutcomePanelConfig(test_fraction=0.15)

@pytest.mark.unit
class TestRegressorPanelConfig:
    '''The regressor panel (roadmap Stage 3): QCEW pins, the held-out draw, fitting, the record.'''

    def test_yaml_matches_defaults_but_for_the_branch_record(self):
        cfg = load_config(RegressorPanelConfig, 'data/regressor_panel.yaml')

        assert cfg.model_copy(update={'branch_record': None}) == RegressorPanelConfig()
        assert cfg.branch_record is not None
        assert sorted(cfg.qcew_sha256) == [f'{year}_US000_annual.csv' for year in range(2022, 2026)]
        assert cfg.alphas == sorted(set(cfg.alphas))
        assert (cfg.seed, cfg.heldout_fraction, cfg.fold_seed) == (20260924, 0.2, 20260924)
        assert (cfg.folds, cfg.repeats, cfg.inner_folds, cfg.min_groups) == (5, 5, 5, 10)
        assert cfg.heldout_groups_csv == './conf/data/regressor_heldout_groups.csv'
        assert cfg.selection_log == './logs/selection_log.jsonl'

    def test_the_text_only_comparator_reads_like_the_arm(self):
        # D9: the arm's own backbone, at the arm's tokenization length
        arm = yaml.safe_load(Path('conf/config.yaml').read_text())
        text_only = RegressorPanelConfig().text_only

        assert text_only.backbone == arm['model']['base_model_name']
        assert text_only.max_length == arm['data_loader']['tokenization']['max_length']

    @pytest.mark.parametrize('fraction', [0.0, 1.0])
    def test_the_held_out_fraction_lies_strictly_between_zero_and_one(self, fraction):
        with pytest.raises(ValidationError):
            RegressorPanelConfig(heldout_fraction=fraction)

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            RegressorPanelConfig(heldout_share=0.2)
        with pytest.raises(ValidationError):
            RegressorBranchRecord(**BRANCH_RECORD, rule='plan 3')

    def test_the_branch_record_has_no_defaults(self):
        with pytest.raises(ValidationError):
            RegressorBranchRecord(branch='A')
        with pytest.raises(ValidationError):
            RegressorBranchRecord(**{**BRANCH_RECORD, 'branch': 'D'})

# -------------------------------------------------------------------------------------------------
```

Create `tests/unit/test_regressor_branch_record.py` with exactly this content:

```python
'''
The regressor panel's branch record matches Stage 1's finding (roadmap Stage 3 Exit).

The record in ``conf/data/regressor_panel.yaml`` is compared field by field with the finding's
decision block, which Stage 3 reads verbatim, and with section 4's table of excluded codes.
'''

import re
from pathlib import Path

import pytest

from naics_embedder.utils.config import RegressorPanelConfig, load_config

pytestmark = pytest.mark.unit

FINDING = Path('specs/findings/employment-statistics-coverage.md')

@pytest.fixture(scope='module')
def finding() -> str:
    return FINDING.read_text(encoding='utf-8')

@pytest.fixture(scope='module')
def block(finding) -> str:
    match = re.search(r'<!-- decision:begin -->\n(.*?)<!-- decision:end -->', finding, re.DOTALL)
    assert match, 'the finding has no decision block'
    return match.group(1)

@pytest.fixture(scope='module')
def record():
    cfg = load_config(RegressorPanelConfig, 'data/regressor_panel.yaml')
    assert cfg.branch_record is not None
    return cfg.branch_record

def _field(block: str, name: str) -> str:
    match = re.search(rf'^- \*\*{re.escape(name)}:\*\* (.+)$', block, re.MULTILINE)
    assert match, f'the decision block has no {name} line'
    return match.group(1)

def _count(text: str) -> int:
    return int(text.replace(',', ''))

def test_the_branch_and_both_answers_match_the_decision_block(record, block):
    assert _field(block, 'Branch').startswith(f'{record.branch}. ')
    assert _field(block, 'Time-respecting outcome') == (
        'yes.' if record.time_respecting_outcome else 'no.'
    )
    assert _field(block, 'Seen-code regime') == ('yes.' if record.seen_regime else 'no.')

def test_the_source_years_ownership_and_grain_match_the_decision_block(record, block):
    source = _field(block, 'Source')

    assert source.startswith(f'{record.source}, ')
    years = re.search(r'reference years ((?:\d{4}, )*\d{4})', source).group(1)
    assert [int(year) for year in years.split(', ')] == record.reference_years
    assert re.search(r'\(own_code (\d+)\)', source).group(1) == record.ownership
    assert f'({record.grain}, private ownership)' in _field(block, 'Row grain')

def test_the_population_matches_the_decision_block(record, block):
    seen, heldout, codebook = re.search(
        r'([\d,]+) codes for the seen-code regime and ([\d,]+) for the held-out-code regime, '
        r'of the ([\d,]+) six-digit codes',
        _field(block, 'Population'),
    ).groups()

    assert (_count(seen), _count(heldout)) == (record.population_seen, record.population_heldout)
    assert _count(codebook) - len(record.excluded_codes) == record.population_seen

def test_the_excluded_codes_are_section_fours(record, finding):
    section = finding.split('### Codes excluded at the chosen grain', 1)[1].split('\n## ', 1)[0]

    excluded = re.findall(r'^\| (\d{6}) \| no private cell \|$', section, re.MULTILINE)

    assert len(excluded) == 32
    assert record.excluded_codes == excluded
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_regressor_branch_record.py -q`
Expected: two collection errors, `ImportError: cannot import name 'RegressorBranchRecord' from
'naics_embedder.utils.config'` and `ImportError: cannot import name 'RegressorPanelConfig' from
'naics_embedder.utils.config'`.

- [x] **Step 3: Declare the config and ship it**

**`src/naics_embedder/utils/config.py`, edit 1 of 1.** Replace:

```python
class SupervisionRuntimeConfig(BaseModel):
```

with:

```python
# Stage 1's finding (specs/findings/employment-statistics-coverage.md, Sources)
QCEW_SLICE_SHA256 = {
    '2022_US000_annual.csv': 'c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18',
    '2023_US000_annual.csv': 'fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9',
    '2024_US000_annual.csv': '48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46',
    '2025_US000_annual.csv': '0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811',
}
RIDGE_ALPHAS = [
    0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.2, 10.0, 32.0, 100.0, 320.0, 1000.0, 3200.0,
    10000.0, 32000.0, 100000.0
]

class TextOnlyConfig(BaseModel):
    '''The regressor panel's text-only comparator (roadmap D9): the arm's backbone, frozen.'''

    model_config = ConfigDict(extra='forbid')

    backbone: str = Field(
        default='sentence-transformers/all-MiniLM-L6-v2',
        description="The arm's backbone (model.base_model_name), read from the local cache",
    )
    max_length: int = Field(
        default=512,
        ge=1,
        description="Tokens kept per channel text: the arm's data_loader max_length",
    )
    batch_size: int = Field(default=32, ge=1, description='Texts per forward pass')

class RegressorBranchRecord(BaseModel):
    '''
    The Req 2 branch Stage 1's finding dictates: its decision block and section 4's excluded
    codes (specs/findings/employment-statistics-coverage.md).
    '''

    model_config = ConfigDict(extra='forbid')

    branch: Literal['A', 'B', 'C']
    source: str
    reference_years: List[int]
    ownership: str
    grain: str
    population_seen: int
    population_heldout: int
    time_respecting_outcome: bool
    seen_regime: bool
    excluded_codes: List[str]

class RegressorPanelConfig(BaseModel):
    '''The regressor panel (roadmap Stage 3): QCEW inputs, held-out draw, fitting and the log.'''

    model_config = ConfigDict(extra='forbid')

    qcew_dir: str = Field(
        default='~/Downloads/Data/QCEW',
        description='Directory holding the QCEW national slices (outside the repo)',
    )
    qcew_sha256: Dict[str, str] = Field(
        default_factory=lambda: dict(QCEW_SLICE_SHA256),
        description="Each national slice's sha256, from Stage 1's finding",
    )
    codebook_codes_sha256: str = Field(
        default='9b646af189dbc46870861367e7cd399d2328972754dbf1abb30ca5a864f2481a',
        description="SHA-256 of the codebook's codes, sorted, one per line",
    )
    heldout_groups_csv: str = Field(
        default='./conf/data/regressor_heldout_groups.csv',
        description='The committed held-out four-digit groups (data regressor-groups)',
    )
    provenance_json: str = Field(
        default='./conf/data/regressor_heldout_groups_provenance.json',
        description='Where data regressor-groups records how the groups were drawn',
    )
    seed: int = Field(default=20260924, description='Base seed of the held-out draw')
    heldout_fraction: float = Field(
        default=0.2, gt=0.0, lt=1.0, description="Share of each sector's groups held out"
    )
    alphas: List[float] = Field(
        default_factory=lambda: list(RIDGE_ALPHAS), description='Ridge penalties, ascending'
    )
    folds: int = Field(default=5, ge=2, description='Grouped validation folds per repeat')
    repeats: int = Field(default=5, ge=1, description='Validation repeats')
    inner_folds: int = Field(default=5, ge=2, description='Grouped tuning folds (held-out regime)')
    fold_seed: int = Field(default=20260924, description='Base seed of every fold assignment')
    min_groups: int = Field(
        default=10, ge=2, description='Fewest remainder groups a regime needs at a level'
    )
    text_only: TextOnlyConfig = Field(default_factory=TextOnlyConfig)
    selection_log: str = Field(
        default='./logs/selection_log.jsonl',
        description='Append-only log of every panel read and outer-set opening',
    )
    branch_record: Optional[RegressorBranchRecord] = Field(
        default=None, description="The Req 2 branch of Stage 1's finding; required to run"
    )

class SupervisionRuntimeConfig(BaseModel):
```

Create `conf/data/regressor_panel.yaml` with exactly this content:

```yaml
# The regressor panel (roadmap Stage 3): QCEW inputs, the held-out draw, fitting and the log

# QCEW national slices, read only under the sha256 values of Stage 1's finding
# (specs/findings/employment-statistics-coverage.md, Sources); they stay outside the repo
qcew_dir: ~/Downloads/Data/QCEW
qcew_sha256:
  2022_US000_annual.csv: c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18
  2023_US000_annual.csv: fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9
  2024_US000_annual.csv: 48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46
  2025_US000_annual.csv: 0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811

# SHA-256 of the codebook's 2,125 codes, sorted, one per line (bundle 18403d29's codebook)
codebook_codes_sha256: 9b646af189dbc46870861367e7cd399d2328972754dbf1abb30ca5a864f2481a

# data regressor-groups draws the held-out four-digit groups once: a fifth of each sector's
# groups (largest remainder), seeded; the table is committed and never redrawn
heldout_groups_csv: ./conf/data/regressor_heldout_groups.csv
provenance_json: ./conf/data/regressor_heldout_groups_provenance.json
seed: 20260924
heldout_fraction: 0.2

# Ridge on standardized features; the penalty is tuned by nested grouped folds in the remainder
alphas: [0.001, 0.0032, 0.01, 0.032, 0.1, 0.32, 1.0, 3.2, 10.0, 32.0, 100.0, 320.0, 1000.0,
  3200.0, 10000.0, 32000.0, 100000.0]
folds: 5
repeats: 5
inner_folds: 5
fold_seed: 20260924
min_groups: 10

# The text-only comparator (roadmap D9): the arm's backbone, frozen, reading the arm's text
text_only:
  backbone: sentence-transformers/all-MiniLM-L6-v2
  max_length: 512
  batch_size: 32

# Every validation read, outer-set opening and outer read is appended here (Req 4). logs/ is
# gitignored: a log inside a worktree goes when the worktree is removed.
selection_log: ./logs/selection_log.jsonl

# The Req 2 branch Stage 1's finding dictates: its decision block and section 4's excluded codes
branch_record:
  branch: A
  source: QCEW annual averages
  reference_years: [2022, 2023, 2024, 2025]
  ownership: '5'
  grain: national
  population_seen: 980
  population_heldout: 980
  time_respecting_outcome: true
  seen_regime: true
  excluded_codes: ['112130', '517122', '541120', '921110', '921120', '921130', '921140',
    '921150', '921190', '922110', '922120', '922130', '922140', '922150', '922160', '922190',
    '923110', '923120', '923130', '923140', '924110', '924120', '925110', '925120', '926110',
    '926120', '926130', '926140', '926150', '927110', '928110', '928120']
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_regressor_branch_record.py -q`
Expected: `66 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1447 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/utils/config.py tests/unit/test_config.py tests/unit/test_regressor_branch_record.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/utils/config.py conf/data/regressor_panel.yaml tests/unit/test_config.py tests/unit/test_regressor_branch_record.py
git commit -m "feat(config): add the regressor panel config and Stage 1's branch record"
```

### Task 6: The regressor panel

This task builds `RegressorPanel`: the fit plans, the comparators, an arm's tables, the logged
readings and openings, and the config helpers. Its tests carry the roadmap Exit: see **Final
verification**, Step 5, for which test backs which outcome.

**Files:**

- Create: `src/naics_embedder/panels/regressor.py`
- Modify: `tests/fixtures/regressor_panel.py` (add `SETTINGS`, the stub arm tables and the
  `regressor_arm` fixture)
- Test: `tests/unit/test_regressor_panel.py`

**Interfaces:**

- Consumes: Task 1's `qcew_rows` names; Task 2's `regressor_splits` names; Task 3's ridge
  functions; Task 4's `TEXT_ONLY_PREFIX` and `pca_reduce`; Task 5's `RegressorBranchRecord` and
  `RegressorPanelConfig`; `SelectionEvent` and `SelectionLog` (existing,
  `naics_embedder.panels.selection_log`); `SealedSplitError` and `SplitAlreadyOpenedError`
  (existing, `naics_embedder.panels.outcome`).
- Produces, in `naics_embedder.panels.regressor`:
  - `Regime(str, Enum)`: `SEEN = 'seen'`, `HELDOUT = 'heldout'`; `PANEL_NAMES` (`regressor_seen`,
    `regressor_heldout`), `OUTER_SPLITS`, `REGIME_STREAMS`, `TEST_STREAM = 1000`,
    `LEVELS = (2, 3, 4, 5, 6)`, `DECISION_LEVEL = 6`, `VALIDATION = 'validation'`,
    `TEST = 'test'`, `COVARIATES`, `REPRESENTATIONS`, `PREDICTION_COLUMNS`
  - `FitSettings(alphas, folds=5, repeats=5, inner_folds=5, fold_seed=20260924, min_groups=10)`
  - `comparators(regime, level) -> Tuple[str, ...]` and
    `group_folds(groups, n_folds, seed) -> np.ndarray`
  - `FitTask(repeat, fold, fit, score, tuning)`, `Plan = List[FitTask]`,
    `seen_validation_plan(frame, level, settings) -> Plan`,
    `heldout_validation_plan(frame, level, settings) -> Plan` and
    `outer_plan(frame, regime, level, settings) -> Plan`
  - `coordinate_matrix(table) -> (codes, matrix)` and `ArmTables.from_tables(coordinates,
    text_only) -> ArmTables` (`codes`, `coordinates`, `text_only`, `fingerprint`,
    `text_only_fingerprint`, `dimension`, `lookup(codes)`)
  - `feature_matrix(comparator, frame, level, arm) -> np.ndarray`
  - `verify_branch_record(record, codebook_codes, six_digit_population) -> None`
  - `RegressorPanel(rows_by_level, heldout_groups, log, settings)`, with
    `from_sources(*, qcew_dir, qcew_sha256, codebook_codes, heldout_groups_csv, log_path,
    settings, branch_record, levels=LEVELS)`, `levels`, `fingerprint`, `heldout_groups`, `log`,
    `settings`, `split_counts(level)`, `cell_status(regime, level) -> Optional[str]`,
    `validation(regime, level, arm, purpose) -> pl.DataFrame`,
    `open_outer(regime, purpose, *, reopen_reason=None)` and
    `test(regime, level, arm, purpose) -> pl.DataFrame`
  - `require_branch_record(cfg)`, `fit_settings(cfg) -> FitSettings`,
    `load_regressor_panel(cfg, codebook_path, *, log_path=None, levels=LEVELS) -> RegressorPanel`
    and `summarize(predictions) -> pl.DataFrame`
- Produces, in the fixture module: `SETTINGS` (two folds, two repeats, `min_groups` 4),
  `coordinate_table(codes, dimension=3, seed=7)`, `text_only_table(codes, width=5, seed=11)` and
  the session fixture `regressor_arm`.

- [x] **Step 1: Write the failing test**

Create `tests/unit/test_regressor_panel.py` with exactly this content:

```python
'''
The regressor panel (Req 2; Req 4; Req 5; roadmap Stage 3 Exit; D1, D7, D8, D9).

Synthetic QCEW rows (``tests/fixtures/regressor_panel.py``) and a stub arm stand in for the real
data and a trained arm. Every outer set opened here is a fixture's: no test reads a real one.
'''

import hashlib

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels import regressor, regressor_splits
from naics_embedder.panels.outcome import SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.qcew_rows import level_cells, population
from naics_embedder.panels.regressor import (
    OUTER_SPLITS,
    PREDICTION_COLUMNS,
    ArmTables,
    FitSettings,
    Regime,
    RegressorPanel,
    comparators,
    coordinate_matrix,
    feature_matrix,
    group_folds,
    heldout_validation_plan,
    load_regressor_panel,
    outer_plan,
    seen_validation_plan,
    summarize,
    verify_branch_record,
)
from naics_embedder.panels.regressor_splits import (
    RegressorSplit,
    assign_splits,
    codes_fingerprint,
    write_group_table,
)
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.utils.config import RegressorBranchRecord, RegressorPanelConfig
from tests.fixtures.regressor_panel import (
    BRANCH_RECORD,
    CODEBOOK,
    HELDOUT_GROUPS,
    SETTINGS,
    coordinate_table,
    text_only_table,
    write_qcew_slices,
)

pytestmark = pytest.mark.unit

SEEN_COMPARATORS = (
    'covariates',
    'embedding',
    'covariates+embedding',
    'one_hot',
    'covariates+one_hot',
    'ancestors',
    'covariates+ancestors',
    'text_only',
    'covariates+text_only',
)
PURPOSE = 'exit test: fixture panel only'

@pytest.fixture
def log(tmp_path):
    return SelectionLog(tmp_path / 'selection_log.jsonl')

@pytest.fixture
def panel(regressor_rows, log):
    return RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)

def _keys(frame):
    return set(zip(frame.get_column('code').to_list(), frame.get_column('feature_year').to_list()))

def _split(rows, *splits):
    names = [split.value for split in splits]
    return assign_splits(rows, HELDOUT_GROUPS).filter(pl.col('split').is_in(names))

# -------------------------------------------------------------------------------------------------
# Regimes and comparators (Exit: reported separately, one-hot only in the seen regime)
# -------------------------------------------------------------------------------------------------

def test_one_hot_runs_only_in_the_seen_regime_and_ancestors_only_above_the_sector():
    without_one_hot = tuple(name for name in SEEN_COMPARATORS if 'one_hot' not in name)
    without_ancestors = tuple(name for name in SEEN_COMPARATORS if 'ancestors' not in name)

    assert comparators(Regime.SEEN, 6) == SEEN_COMPARATORS
    assert comparators(Regime.HELDOUT, 6) == without_one_hot
    assert comparators(Regime.SEEN, 2) == without_ancestors

def test_each_regime_is_its_own_panel_and_scores_every_comparator(
    panel, regressor_rows, regressor_arm
):
    remainder = _split(regressor_rows[6], RegressorSplit.REMAINDER)
    scored = {
        Regime.SEEN: remainder.filter(pl.col('feature_year') == 2023).height,
        Regime.HELDOUT: remainder.height,
    }

    for regime, name in ((Regime.SEEN, 'regressor_seen'), (Regime.HELDOUT, 'regressor_heldout')):
        predictions = panel.validation(regime, 6, regressor_arm, PURPOSE)

        assert predictions.columns == list(PREDICTION_COLUMNS)
        assert predictions.get_column('panel').unique().to_list() == [name]
        names = predictions.get_column('comparator').unique(maintain_order=True)
        assert tuple(names) == comparators(regime, 6)
        rows = predictions.group_by('comparator').len().get_column('len')
        assert rows.to_list() == [scored[regime] * SETTINGS.repeats] * len(rows)
        assert predictions.get_column('prediction').is_finite().all()

# -------------------------------------------------------------------------------------------------
# Tuning inside the remainder (Exit: the penalty is tuned inside the remainder, the outer set is
# read once)
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('regime', list(Regime))
def test_no_scored_row_tunes_its_own_penalty(regressor_rows, regime):
    frame = _split(regressor_rows[6], RegressorSplit.REMAINDER)
    build = seen_validation_plan if regime is Regime.SEEN else heldout_validation_plan
    groups = frame.get_column('group').to_list()
    codes = frame.get_column('code').to_list()

    plan = build(frame, 6, SETTINGS)

    assert len(plan) == SETTINGS.repeats * SETTINGS.folds
    for task in plan:
        scored = set(task.score)
        assert all(scored.isdisjoint({*fit, *tuned}) for fit, tuned in task.tuning)
        if regime is Regime.SEEN:
            # Seen: every scored code has earlier rows in the fit set
            assert {codes[row] for row in scored} <= {codes[row] for row in task.fit}
        else:
            # Held-out: a scored group leaves the fit set and every tuning split
            assert {groups[row] for row in scored}.isdisjoint(groups[row] for row in task.fit)
            for fit, tuned in task.tuning:
                assert {*fit, *tuned} <= set(task.fit)
                assert {groups[row] for row in tuned}.isdisjoint(groups[row] for row in fit)

@pytest.mark.parametrize('regime', list(Regime))
def test_the_outer_set_is_scored_once_by_a_penalty_tuned_inside_the_remainder(
    regressor_rows, regime
):
    frame = _split(regressor_rows[6], RegressorSplit.REMAINDER, OUTER_SPLITS[regime])
    split = frame.get_column('split').to_list()
    remainder = {row for row, name in enumerate(split) if name == RegressorSplit.REMAINDER.value}

    [task] = outer_plan(frame, regime, 6, SETTINGS)

    assert (task.repeat, task.fold) == (0, -1)
    assert set(task.fit) == remainder
    assert sorted(task.score) == sorted(set(range(frame.height)) - remainder)
    assert task.tuning
    assert all({*fit, *tuned} <= remainder for fit, tuned in task.tuning)

def test_the_test_split_scores_each_outer_row_once_per_comparator(
    panel, regressor_rows, regressor_arm, log
):
    for regime in Regime:
        panel.open_outer(regime, PURPOSE)
        predictions = panel.test(regime, 6, regressor_arm, PURPOSE)

        outer = _keys(_split(regressor_rows[6], OUTER_SPLITS[regime]))
        assert set(predictions.get_column('fold')) == {-1}
        for _, rows in predictions.group_by('comparator'):
            assert rows.height == len(outer)
            assert _keys(rows) == outer

    assert [(r['event'], r['panel'], r['split']) for r in log.records()] == [
        ('open', 'regressor_seen', 'test'),
        ('read', 'regressor_seen', 'test'),
        ('open', 'regressor_heldout', 'test'),
        ('read', 'regressor_heldout', 'test'),
    ]

def test_validation_reads_only_the_remainder_and_logs_each_read(
    panel, regressor_rows, regressor_arm, log
):
    remainder = _keys(_split(regressor_rows[6], RegressorSplit.REMAINDER))

    seen = panel.validation(Regime.SEEN, 6, regressor_arm, PURPOSE)
    heldout = panel.validation(Regime.HELDOUT, 6, regressor_arm, PURPOSE)

    assert _keys(seen) <= remainder
    assert _keys(heldout) == remainder
    records = log.records()
    assert [(r['event'], r['panel'], r['split'], r['n_queries']) for r in records] == [
        ('read', 'regressor_seen', 'validation', len(remainder)),
        ('read', 'regressor_heldout', 'validation', len(remainder)),
    ]
    assert records[0]['fingerprint'] == panel.fingerprint
    assert records[0]['detail']['level'] == 6
    assert records[0]['detail']['arm'] == regressor_arm.fingerprint
    assert records[0]['detail']['text_only'] == regressor_arm.text_only_fingerprint
    assert records[0]['detail']['comparators'] == list(SEEN_COMPARATORS)

# -------------------------------------------------------------------------------------------------
# The committed draw (Exit: the panel reads the committed outer groups, never a fresh draw)
# -------------------------------------------------------------------------------------------------

def _from_sources(tmp_path, cells, groups, **overrides):
    qcew = tmp_path / 'qcew'
    pins = write_qcew_slices(qcew, cells)
    table = tmp_path / 'regressor_heldout_groups.csv'
    write_group_table(groups, table)
    arguments = {
        'qcew_dir': qcew,
        'qcew_sha256': pins,
        'codebook_codes': CODEBOOK,
        'heldout_groups_csv': table,
        'log_path': tmp_path / 'selection_log.jsonl',
        'settings': SETTINGS,
        'branch_record': BRANCH_RECORD,
        'levels': (4, 6),
    }
    return RegressorPanel.from_sources(**{**arguments, **overrides}), table

def test_the_panel_reads_the_committed_groups_and_never_draws(
    tmp_path, monkeypatch, regressor_cells
):

    def no_draw(*args, **kwargs):
        raise AssertionError('the panel drew held-out groups')

    monkeypatch.setattr(regressor_splits, 'draw_heldout_groups', no_draw)

    panel, table = _from_sources(tmp_path, regressor_cells, ['1111'])

    assert not hasattr(regressor, 'draw_heldout_groups')
    assert panel.heldout_groups == ('1111', )
    assert panel.fingerprint == hashlib.sha256(table.read_bytes()).hexdigest()
    assert panel.levels == (4, 6)
    # 111111 and 111112 in three feature years, and 1111 itself in three
    assert panel.split_counts(6)['heldout_outer'] == 6
    assert panel.split_counts(4)['heldout_outer'] == 3

def test_from_sources_refuses_data_the_branch_record_does_not_name(tmp_path, regressor_cells):
    record = {**BRANCH_RECORD, 'excluded_codes': []}

    with pytest.raises(ValueError, match='branch record mismatch'):
        _from_sources(tmp_path, regressor_cells, HELDOUT_GROUPS, branch_record=record)

def test_the_branch_record_must_name_this_panels_data(regressor_cells):
    six = population(level_cells(regressor_cells, CODEBOOK, 6))
    verify_branch_record(BRANCH_RECORD, CODEBOOK, six)

    for change, message in [
        ({
            'branch': 'B'
        }, 'branch A'),
        ({
            'time_respecting_outcome': False
        }, 'time-respecting'),
        ({
            'seen_regime': False
        }, 'seen-code regime'),
        ({
            'reference_years': [2021, 2022, 2023, 2024]
        }, 'reference years'),
        ({
            'ownership': '0'
        }, 'national private'),
        ({
            'excluded_codes': []
        }, 'six-digit population'),
        ({
            'population_heldout': 33
        }, 'populations'),
    ]:
        with pytest.raises(ValueError, match=message):
            verify_branch_record({**BRANCH_RECORD, **change}, CODEBOOK, six)

# -------------------------------------------------------------------------------------------------
# Sealing (Exit: neither regime's outer set can be read without a logged opening)
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('regime', list(Regime))
def test_neither_outer_set_is_read_without_a_logged_opening(panel, regressor_arm, log, regime):
    other = Regime.HELDOUT if regime is Regime.SEEN else Regime.SEEN

    with pytest.raises(SealedSplitError, match='sealed'):
        panel.test(regime, 6, regressor_arm, PURPOSE)
    assert log.records() == []

    panel.open_outer(regime, PURPOSE)

    with pytest.raises(SealedSplitError, match='sealed'):
        panel.test(other, 6, regressor_arm, PURPOSE)
    panel.test(regime, 6, regressor_arm, PURPOSE)
    assert [(r['event'], r['split']) for r in log.records()] == [('open', 'test'), ('read', 'test')]

def test_every_panel_object_opens_for_itself_and_a_reopening_needs_a_reason(
    regressor_rows, regressor_arm, log
):
    first = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)
    first.open_outer(Regime.SEEN, 'first opening')
    second = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)

    with pytest.raises(SealedSplitError):
        second.test(Regime.SEEN, 6, regressor_arm, PURPOSE)
    with pytest.raises(SplitAlreadyOpenedError, match='reopen_reason'):
        second.open_outer(Regime.SEEN, 'second opening')
    second.open_outer(Regime.SEEN, 'second opening', reopen_reason='a fixture rerun')
    second.open_outer(Regime.HELDOUT, 'first held-out opening')

    records = log.records()
    assert [(r['event'], r['panel']) for r in records] == [
        ('open', 'regressor_seen'),
        ('reopen', 'regressor_seen'),
        ('open', 'regressor_heldout'),
    ]
    assert records[1]['detail']['reason'] == 'a fixture rerun'
    # An opening counts the six-digit outer rows, and records every loaded level's
    assert records[0]['n_queries'] == 26
    assert records[0]['detail']['rows_by_level'] == {'2': 1, '3': 5, '4': 14, '5': 14, '6': 26}
    assert records[2]['n_queries'] == 18

def test_openings_are_counted_per_held_out_draw(regressor_rows, log):
    RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS).open_outer(Regime.SEEN, 'first')

    RegressorPanel(regressor_rows, ['1111'], log, SETTINGS).open_outer(Regime.SEEN, 'other draw')

    assert [r['event'] for r in log.records()] == ['open', 'open']

# -------------------------------------------------------------------------------------------------
# Dating (Exit: every row's features are dated before its outcome, D7)
# -------------------------------------------------------------------------------------------------

def test_every_rows_features_are_dated_before_its_outcome(panel, regressor_rows, regressor_arm):
    panel.open_outer(Regime.SEEN, PURPOSE)
    panel.open_outer(Regime.HELDOUT, PURPOSE)

    predictions = pl.concat(
        [
            panel.validation(Regime.SEEN, 6, regressor_arm, PURPOSE),
            panel.validation(Regime.HELDOUT, 6, regressor_arm, PURPOSE),
            panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE),
            panel.test(Regime.HELDOUT, 6, regressor_arm, PURPOSE),
        ]
    )

    for frame in (*regressor_rows.values(), predictions):
        assert (frame.get_column('outcome_year') == frame.get_column('feature_year') + 1).all()

@pytest.mark.parametrize('split', ['validation', 'test'])
def test_the_seen_regime_fits_no_outcome_after_a_scored_rows_feature_year(regressor_rows, split):
    if split == 'validation':
        frame = _split(regressor_rows[6], RegressorSplit.REMAINDER)
        plan = seen_validation_plan(frame, 6, SETTINGS)
    else:
        frame = _split(regressor_rows[6], RegressorSplit.REMAINDER, RegressorSplit.SEEN_OUTER)
        plan = outer_plan(frame, Regime.SEEN, 6, SETTINGS)
    feature = frame.get_column('feature_year').to_list()
    outcome = frame.get_column('outcome_year').to_list()

    for task in plan:
        for fit, scored in ((task.fit, task.score), *task.tuning):
            assert max(outcome[row] for row in fit) <= min(feature[row] for row in scored)

# -------------------------------------------------------------------------------------------------
# Pairing (Req 5) and folds
# -------------------------------------------------------------------------------------------------

def test_folds_do_not_depend_on_the_arm(panel, regressor_arm):
    other = ArmTables.from_tables(
        coordinate_table(CODEBOOK, dimension=4, seed=99), text_only_table(CODEBOOK, seed=5)
    )
    keys = ['comparator', 'repeat', 'fold', 'code', 'feature_year']

    for regime in Regime:
        first = panel.validation(regime, 6, regressor_arm, PURPOSE)
        second = panel.validation(regime, 6, other, PURPOSE)

        assert first.select(keys).equals(second.select(keys))
        covariates = pl.col('comparator') == 'covariates'
        assert first.filter(covariates).equals(second.filter(covariates))

def test_group_folds_keep_a_groups_rows_together_whatever_the_row_order():
    groups = ['a', 'b', 'a', 'c', 'd', 'b', 'e']
    seed = (20260924, 0, 6, 0)

    folds = group_folds(groups, 2, seed)

    assert set(folds.tolist()) == {0, 1}
    assert all(len({f for g, f in zip(groups, folds) if g == group}) == 1 for group in groups)
    assert dict(zip(groups, folds)) == dict(zip(groups[::-1], group_folds(groups[::-1], 2, seed)))
    with pytest.raises(ValueError, match='cannot fill'):
        group_folds(['a', 'b'], 3, seed)

@pytest.mark.parametrize(
    'arguments',
    [
        {
            'alphas': ()
        },
        {
            'alphas': (1.0, 0.1)
        },
        {
            'alphas': (0.0, 1.0)
        },
        {
            'alphas': (1.0, 1.0)
        },
        {
            'alphas': (1.0, ),
            'folds': 1
        },
        {
            'alphas': (1.0, ),
            'repeats': 0
        },
        {
            'alphas': (1.0, ),
            'min_groups': 9
        },
    ],
)
def test_fit_settings_are_checked(arguments):
    with pytest.raises(ValueError):
        FitSettings(**arguments)

# -------------------------------------------------------------------------------------------------
# Levels
# -------------------------------------------------------------------------------------------------

def test_a_cell_without_enough_remainder_groups_is_undefined_not_scored(panel, regressor_arm, log):
    assert panel.cell_status(Regime.SEEN, 2) == '1 remainder groups, fewer than 4'
    assert panel.cell_status(Regime.HELDOUT, 3) == (
        'no four-digit parent: the held-out regime runs at levels 4-6'
    )
    assert panel.cell_status(Regime.SEEN, 3) is None
    assert panel.cell_status(Regime.HELDOUT, 4) is None

    with pytest.raises(ValueError, match='undefined at level 2'):
        panel.validation(Regime.SEEN, 2, regressor_arm, PURPOSE)
    assert log.records() == []

def test_the_multi_level_variant_scores_level_codes_outside_the_held_out_groups(
    panel, regressor_arm
):
    predictions = panel.validation(Regime.SEEN, 3, regressor_arm, PURPOSE)

    assert set(predictions.get_column('code')) <= {'112', '238', '311', '332', '522'}
    assert set(predictions.get_column('level')) == {3}

def test_each_comparator_has_its_own_columns(regressor_rows, regressor_arm):
    frame = regressor_rows[6]
    rows, codes = frame.height, frame.get_column('code').n_unique()

    assert feature_matrix('covariates', frame, 6, regressor_arm).shape == (rows, 2)
    assert feature_matrix('embedding', frame, 6, regressor_arm).shape == (rows, 3)
    assert feature_matrix('text_only', frame, 6, regressor_arm).shape == (rows, 3)
    assert feature_matrix('one_hot', frame, 6, regressor_arm).shape == (rows, codes)
    assert feature_matrix('covariates+one_hot', frame, 6, regressor_arm).shape == (rows, 2 + codes)
    # One indicator per ancestor level, 2 through 5
    assert (feature_matrix('ancestors', frame, 6, regressor_arm).sum(axis=1) == 4).all()
    with pytest.raises(ValueError, match='unknown comparator'):
        feature_matrix('covariates+nothing', frame, 6, regressor_arm)

# -------------------------------------------------------------------------------------------------
# An arm's tables (D9)
# -------------------------------------------------------------------------------------------------

def test_lorentz_points_are_refused_and_the_export_form_is_read():
    codes = ['11', '21', '22', '23']
    tangent = np.random.default_rng(0).normal(size=(4, 3))
    time = np.sqrt(1.0 + (tangent**2).sum(axis=1))
    lorentz = pl.DataFrame(
        {
            'code': codes,
            'x0': time,
            **{
                f'x{i + 1}': tangent[:, i]
                for i in range(3)
            }
        }
    )
    exported = pl.DataFrame(
        {
            'code': codes,
            'index': range(4),
            'level': [2] * 4,
            **{
                f'e{i}': tangent[:, i]
                for i in range(3)
            }
        }
    )

    with pytest.raises(ValueError, match='Lorentz points'):
        coordinate_matrix(lorentz)
    read_codes, matrix = coordinate_matrix(exported)
    assert read_codes == tuple(codes)
    np.testing.assert_array_equal(matrix, tangent)

@pytest.mark.parametrize(
    ('table', 'message'),
    [
        (pl.DataFrame({'e0': [1.0]}), 'no code column'),
        (pl.DataFrame({
            'code': ['11'],
            'index': [0]
        }), 'no coordinate columns'),
        (pl.DataFrame({
            'code': ['11', '11'],
            'e0': [1.0, 2.0]
        }), 'repeats a code'),
        (pl.DataFrame({
            'code': ['11', '21'],
            'e0': [1.0, float('nan')]
        }), 'not finite'),
    ],
)
def test_a_malformed_coordinate_table_is_refused(table, message):
    with pytest.raises(ValueError, match=message):
        coordinate_matrix(table)

def test_the_text_only_table_is_reduced_to_the_arms_dimension_code_by_code(regressor_arm):
    shuffled = text_only_table(CODEBOOK).reverse()

    arm = ArmTables.from_tables(coordinate_table(CODEBOOK), shuffled)

    assert regressor_arm.text_only.shape == (len(CODEBOOK), regressor_arm.dimension)
    np.testing.assert_allclose(arm.text_only, regressor_arm.text_only, atol=1e-10)
    assert arm.text_only_fingerprint == regressor_arm.text_only_fingerprint
    with pytest.raises(ValueError, match='different codes'):
        ArmTables.from_tables(coordinate_table(CODEBOOK), text_only_table(CODEBOOK[1:]))
    with pytest.raises(ValueError, match='no coordinates'):
        regressor_arm.lookup(['999999'])

# -------------------------------------------------------------------------------------------------
# From config, and the summary
# -------------------------------------------------------------------------------------------------

def test_the_config_builds_the_panel_over_a_codebook(tmp_path, regressor_cells):
    qcew = tmp_path / 'qcew'
    pins = write_qcew_slices(qcew, regressor_cells)
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    table = tmp_path / 'groups.csv'
    write_group_table(HELDOUT_GROUPS, table)
    cfg = RegressorPanelConfig(
        qcew_dir=str(qcew),
        qcew_sha256=pins,
        codebook_codes_sha256=codes_fingerprint(CODEBOOK),
        heldout_groups_csv=str(table),
        selection_log=str(tmp_path / 'selection_log.jsonl'),
        alphas=list(SETTINGS.alphas),
        folds=2,
        repeats=2,
        inner_folds=2,
        min_groups=4,
        branch_record=RegressorBranchRecord(**BRANCH_RECORD),
    )

    panel = load_regressor_panel(cfg, codebook, levels=(6, ))

    assert panel.settings == SETTINGS
    assert panel.levels == (6, )
    assert panel.log.path == tmp_path / 'selection_log.jsonl'
    other = load_regressor_panel(cfg, codebook, log_path=tmp_path / 'other.jsonl', levels=(6, ))
    assert other.log.path == tmp_path / 'other.jsonl'
    with pytest.raises(ValueError, match='no branch_record'):
        load_regressor_panel(cfg.model_copy(update={'branch_record': None}), codebook)

def test_the_summary_pools_rows_per_panel_split_level_and_comparator():
    predictions = pl.DataFrame(
        {
            'panel': ['regressor_seen'] * 4,
            'split': ['validation'] * 4,
            'level': [6] * 4,
            'comparator': ['covariates', 'covariates', 'one_hot', 'one_hot'],
            'outcome': [1.0, 3.0, 1.0, 3.0],
            'prediction': [1.0, 3.0, 2.0, 2.0],
            'alpha': [1.0, 1.0, 10.0, 100.0],
        }
    )

    first, second = summarize(predictions).iter_rows(named=True)

    assert (first['comparator'], first['rows'], first['rmse'], first['r2']) == (
        'covariates', 2, 0.0, 1.0
    )
    assert (second['rmse'], second['r2'], second['median_alpha']) == (1.0, 0.0, 55.0)
```

- [x] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_regressor_panel.py -q`
Expected: one collection error, `ImportError: cannot import name 'regressor' from
'naics_embedder.panels'`.

- [x] **Step 3: Write the panel and extend the fixture module**

> Deviation: the whole-plan review changed `panels/regressor.py` after this task. `looks_lorentz` scales each row's tolerance with its x0^2 and reads the level from the row nearest the origin (a50f85b, f640468). A read checks the arm's codes before it is logged, and `require_arm` checks every loaded level (c5ee998). `coordinate_matrix` refuses constant columns (360af12). `require_openable` checks an opening without logging it (b7c21d6). The tests gained behavioural leakage checks and a grouped-fold check for the seen regime (f84e00b).

Create `src/naics_embedder/panels/regressor.py` with exactly this content:

```python
'''
The regressor panel (Req 2; Req 1, regressor estimand; Req 4; roadmap Stage 3; D1, D7, D8, D9).

An arm's coordinates enter a ridge regression as regressors, and the panel returns out-of-sample
predictions per row, keyed by code, year and group, so that Stage 4 can compute whichever
statistic it settles on and resample by four-digit group.

- **Rows (D7).** A code in a feature year t (2022–2024): covariates from year t, outcome log
  employment in t + 1 (``qcew_rows``).
- **Regimes (D8: each is a panel).** ``seen``: every scored code has earlier rows in the fit set;
  one-hot is a real competitor. ``heldout``: every row of a held-out four-digit group leaves the
  fit set; it runs at levels 4–6, where four-digit parents exist.
- **Partition.** One partition serves both regimes (``regressor_splits``): validation reads only
  the remainder, feature years 2022 and 2023 of codes outside the held-out groups.
- **Comparators (Req 2).** Covariates alone (log establishments and log wages, D1), and each
  representation alone and with the covariates: the arm's coordinates, six-digit one-hot (level-L
  one-hot in the multi-level variant; seen regime only), ancestor indicators at levels 2 to L − 1,
  and the text-only table reduced by PCA to the arm's dimension (D9).
- **Fitting.** Ridge on standardized features, the penalty tuned by nested cross-validation
  inside the remainder only (``FitTask``). Fold assignments depend on the regime, level, repeat
  and group ids alone, never on the arm, so every arm is scored on the same folds (Req 5).
- **Sealing.** Each regime's outer set is read only after a logged opening by the same panel
  object; a second opening of the same split (the same held-out draw, by fingerprint) needs a
  stated reason. Every read goes to the selection log, whose ``n_queries`` field counts rows here.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import polars as pl

from naics_embedder.panels.outcome import SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.qcew_rows import (
    PRIVATE,
    WINDOW_YEARS,
    level_cells,
    load_national_cells,
    panel_rows,
    population,
)
from naics_embedder.panels.regressor_splits import (
    GROUP_LEVEL,
    REMAINDER_FEATURE_YEARS,
    SECTOR_LEVEL,
    RegressorSplit,
    ancestor_at,
    assign_splits,
    check_partition,
    group_table_fingerprint,
    read_codebook_codes,
    read_group_table,
    split_counts,
)
from naics_embedder.panels.ridge import (
    best_alpha_index,
    squared_errors,
    standardized_ridge_path,
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.panels.text_only import TEXT_ONLY_PREFIX, pca_reduce
from naics_embedder.utils.config import RegressorBranchRecord, RegressorPanelConfig

class Regime(str, Enum):
    '''The two regressor regimes, each its own panel under Req 5 (D8).'''

    SEEN = 'seen'
    HELDOUT = 'heldout'

PANEL_NAMES = {Regime.SEEN: 'regressor_seen', Regime.HELDOUT: 'regressor_heldout'}
OUTER_SPLITS = {
    Regime.SEEN: RegressorSplit.SEEN_OUTER,
    Regime.HELDOUT: RegressorSplit.HELDOUT_OUTER
}
REGIME_STREAMS = {Regime.SEEN: 0, Regime.HELDOUT: 1}
# The test fit's inner folds draw from their own stream; validation repeats never reach it
TEST_STREAM = 1000
LEVELS = (2, 3, 4, 5, 6)
# D8's two regressor panels are the six-digit regimes; levels 2–5 are the multi-level variant
DECISION_LEVEL = 6
VALIDATION = 'validation'
TEST = 'test'

COVARIATES = 'covariates'
REPRESENTATIONS = ('embedding', 'one_hot', 'ancestors', 'text_only')
COVARIATE_COLUMNS = ('log_estabs', 'log_wages')
METADATA_COLUMNS = ('code', 'index', 'level')
PREDICTION_COLUMNS = (
    'panel',
    'split',
    'level',
    'comparator',
    'repeat',
    'fold',
    'code',
    'group',
    'feature_year',
    'outcome_year',
    'alpha',
    'outcome',
    'prediction',
)

# -------------------------------------------------------------------------------------------------
# Settings, comparators and folds
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FitSettings:
    '''
    How the panel tunes and resamples.

    Args:
        alphas: Ridge penalties, ascending.
        folds: Validation folds per repeat (grouped).
        repeats: Validation repeats, each with its own fold assignment.
        inner_folds: Tuning folds inside a held-out fit set (grouped).
        fold_seed: Base seed of every fold assignment.
        min_groups: Fewest remainder groups a regime needs at a level; fewer is undefined.
    '''

    alphas: Tuple[float, ...]
    folds: int = 5
    repeats: int = 5
    inner_folds: int = 5
    fold_seed: int = 20260924
    min_groups: int = 10

    def __post_init__(self) -> None:
        grid = list(self.alphas)
        if not grid or any(alpha <= 0 for alpha in grid) or grid != sorted(set(grid)):
            raise ValueError('alphas must be distinct positive penalties in ascending order')
        if self.folds < 2 or self.inner_folds < 2 or self.repeats < 1:
            raise ValueError('folds and inner_folds must be at least 2, and repeats at least 1')
        if self.min_groups < 2 * max(self.folds, self.inner_folds):
            raise ValueError('min_groups must be at least twice the larger fold count')

def comparators(regime: Regime, level: int) -> Tuple[str, ...]:
    '''
    Every comparator the regime scores at a level, covariates alone first.

    One-hot runs only in the seen regime, where it can predict more than the intercept; ancestor
    indicators need a level above 2.
    '''

    names = [COVARIATES]
    for representation in REPRESENTATIONS:
        if representation == 'one_hot' and regime is Regime.HELDOUT:
            continue
        if representation == 'ancestors' and level <= SECTOR_LEVEL:
            continue
        names.extend([representation, f'{COVARIATES}+{representation}'])
    return tuple(names)

def group_folds(groups: Sequence[str], n_folds: int, seed: Sequence[int]) -> np.ndarray:
    '''
    A fold per row: the distinct groups, sorted, permuted by ``np.random.default_rng(seed)`` and
    dealt into ``n_folds`` folds in turn, so a group's rows share a fold.

    Raises:
        ValueError: If there are fewer groups than folds.
    '''

    distinct = sorted(set(groups))
    if len(distinct) < n_folds:
        raise ValueError(f'{len(distinct)} groups cannot fill {n_folds} folds')
    order = np.random.default_rng(list(seed)).permutation(len(distinct))
    fold_of = {distinct[index]: position % n_folds for position, index in enumerate(order)}
    return np.array([fold_of[group] for group in groups], dtype=np.int64)

# -------------------------------------------------------------------------------------------------
# Plans: which rows fit, which are scored, which tune the penalty
# -------------------------------------------------------------------------------------------------

Rows = Tuple[int, ...]

@dataclass(frozen=True)
class FitTask:
    '''
    One out-of-sample prediction: fit on ``fit`` and predict ``score`` at the penalty with the
    least summed squared error over the ``tuning`` pairs (fit rows, scored rows).

    Row numbers index the frame the plan was built for; ``fold`` is -1 for the outer set.
    '''

    repeat: int
    fold: int
    fit: Rows
    score: Rows
    tuning: Tuple[Tuple[Rows, Rows], ...]

# One read's fit tasks: every repeat and fold of a validation read, or the outer set's single task
Plan = List[FitTask]

def _rows(mask: np.ndarray) -> Rows:
    return tuple(int(row) for row in np.flatnonzero(mask))

def seen_validation_plan(frame: pl.DataFrame, level: int, settings: FitSettings) -> Plan:
    '''
    Fit on the remainder's 2022 rows and predict its 2023 rows, forward in time (D7).

    The 2023 rows fall into grouped folds; each fold's penalty is chosen on the other folds'
    2023 rows, so no scored row tunes its own penalty. Repeats redraw the folds.
    '''

    years = frame.get_column('feature_year').to_numpy()
    groups = frame.get_column('group').to_list()
    fit = _rows(years == REMAINDER_FEATURE_YEARS[0])
    scored = np.flatnonzero(years == REMAINDER_FEATURE_YEARS[1])
    tasks = []
    for repeat in range(settings.repeats):
        seed = (settings.fold_seed, REGIME_STREAMS[Regime.SEEN], level, repeat)
        folds = group_folds([groups[row] for row in scored], settings.folds, seed)
        for fold in range(settings.folds):
            inside = tuple(int(row) for row in scored[folds == fold])
            outside = tuple(int(row) for row in scored[folds != fold])
            tasks.append(FitTask(repeat, fold, fit, inside, ((fit, outside), )))
    return tasks

def _inner_tuning(
    fit: np.ndarray, groups: Sequence[str], settings: FitSettings, seed: Tuple[int, ...]
) -> Tuple[Tuple[Rows, Rows], ...]:
    inner = group_folds([groups[row] for row in fit], settings.inner_folds, seed)
    return tuple(
        (
            tuple(int(row)
                  for row in fit[inner != fold]), tuple(int(row) for row in fit[inner == fold])
        ) for fold in range(settings.inner_folds)
    )

def heldout_validation_plan(frame: pl.DataFrame, level: int, settings: FitSettings) -> Plan:
    '''
    Repeated grouped folds over the remainder; each fold's penalty is tuned by grouped folds
    inside the rest of the remainder (nested), so a scored group never tunes its own penalty.
    '''

    groups = frame.get_column('group').to_list()
    rows = np.arange(frame.height)
    stream = REGIME_STREAMS[Regime.HELDOUT]
    tasks = []
    for repeat in range(settings.repeats):
        folds = group_folds(groups, settings.folds, (settings.fold_seed, stream, level, repeat))
        for fold in range(settings.folds):
            fit = rows[folds != fold]
            seed = (settings.fold_seed, stream, level, repeat, fold + 1)
            tuning = _inner_tuning(fit, groups, settings, seed)
            tasks.append(FitTask(repeat, fold, _rows(folds != fold), _rows(folds == fold), tuning))
    return tasks

def outer_plan(frame: pl.DataFrame, regime: Regime, level: int, settings: FitSettings) -> Plan:
    '''
    Fit on the whole remainder and predict the regime's outer set once.

    The penalty is tuned inside the remainder only: in the seen regime on the forward split
    (2022 rows fit, 2023 rows scored), in the held-out regime by grouped folds.
    '''

    split = frame.get_column('split').to_numpy()
    remainder = split == RegressorSplit.REMAINDER.value
    outer = split == OUTER_SPLITS[regime].value
    if regime is Regime.SEEN:
        years = frame.get_column('feature_year').to_numpy()
        tuning = (
            (
                _rows(remainder & (years == REMAINDER_FEATURE_YEARS[0])),
                _rows(remainder & (years == REMAINDER_FEATURE_YEARS[1])),
            ),
        )
    else:
        seed = (settings.fold_seed, REGIME_STREAMS[regime], level, TEST_STREAM)
        tuning = _inner_tuning(
            np.flatnonzero(remainder),
            frame.get_column('group').to_list(), settings, seed
        )
    return [FitTask(0, -1, _rows(remainder), _rows(outer), tuning)]

# -------------------------------------------------------------------------------------------------
# An arm's representations and the feature matrices
# -------------------------------------------------------------------------------------------------

def looks_lorentz(matrix: np.ndarray, rtol: float = 1e-3) -> bool:
    '''Whether every row lies on one hyperboloid ``x0^2 - |x|^2 = 1/c`` with ``x0 > 0``.'''

    if matrix.shape[1] < 2:
        return False
    time = matrix[:, 0]
    norm = time**2 - (matrix[:, 1:]**2).sum(axis=1)
    if (time <= 0).any() or (norm <= 0).any():
        return False
    return bool(np.ptp(norm) <= rtol * norm.mean())

def coordinate_matrix(table: pl.DataFrame) -> Tuple[Tuple[str, ...], np.ndarray]:
    '''
    The codes and coordinates of an arm's table in the export form Req 2 names.

    Every column other than ``code`` and the export's ``index`` and ``level`` metadata is a
    coordinate.

    Raises:
        ValueError: If codes repeat, a coordinate is not finite, or the rows lie on a hyperboloid:
            Lorentz points (the train prompt's ``hyp_e*`` export) are not the export form, which
            is tangent coordinates at the origin for a hyperbolic arm (Stage 6's export).
    '''

    if 'code' not in table.columns:
        raise ValueError('the coordinate table has no code column')
    columns = [name for name in table.columns if name not in METADATA_COLUMNS]
    if not columns:
        raise ValueError('the coordinate table has no coordinate columns')
    codes = tuple(table.get_column('code').cast(pl.Utf8).to_list())
    if len(set(codes)) != len(codes):
        raise ValueError('the coordinate table repeats a code')
    matrix = np.array(table.select(columns).to_numpy(), dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('the coordinate table has a coordinate that is not finite')
    if looks_lorentz(matrix):
        raise ValueError(
            'the coordinate table holds Lorentz points on a hyperboloid; the regressor panel '
            'takes the export form (tangent coordinates at the origin for a hyperbolic arm)'
        )
    return codes, matrix

def matrix_fingerprint(codes: Sequence[str], matrix: np.ndarray) -> str:
    '''SHA-256 of the codes and their float64 values, in code order.'''

    order = np.argsort(np.asarray(codes))
    digest = hashlib.sha256('\n'.join(codes[index] for index in order).encode('utf-8'))
    digest.update(np.ascontiguousarray(matrix[order], dtype=np.float64).tobytes())
    return digest.hexdigest()

@dataclass(frozen=True)
class ArmTables:
    '''One arm's coordinates and the text-only table reduced to the arm's dimension (D9).'''

    codes: Tuple[str, ...]
    coordinates: np.ndarray
    text_only: np.ndarray
    fingerprint: str
    text_only_fingerprint: str

    @classmethod
    def from_tables(cls, coordinates: pl.DataFrame, text_only: pl.DataFrame) -> 'ArmTables':
        '''
        Pair an arm's coordinate table with the text-only table of the same codes.

        Raises:
            ValueError: If the two tables cover different codes.
        '''

        codes, matrix = coordinate_matrix(coordinates)
        text_columns = [name for name in text_only.columns if name.startswith(TEXT_ONLY_PREFIX)]
        text_codes = text_only.get_column('code').to_list()
        if set(text_codes) != set(codes) or len(text_codes) != len(codes):
            raise ValueError('the coordinate and text-only tables cover different codes')
        text_matrix = np.array(text_only.select(text_columns).to_numpy(), dtype=np.float64)
        reduced = pca_reduce(text_matrix, matrix.shape[1])
        position = {code: row for row, code in enumerate(text_codes)}
        aligned = reduced[[position[code] for code in codes]]
        return cls(
            codes=codes,
            coordinates=matrix,
            text_only=aligned,
            fingerprint=matrix_fingerprint(codes, matrix),
            text_only_fingerprint=matrix_fingerprint(text_codes, text_matrix),
        )

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[1])

    def lookup(self, codes: Sequence[str]) -> np.ndarray:
        '''Row numbers of the codes in this arm's tables.'''

        position = {code: row for row, code in enumerate(self.codes)}
        missing = sorted(set(codes) - set(position))
        if missing:
            raise ValueError(f'the arm has no coordinates for {len(missing)} codes: {missing[:5]}')
        return np.array([position[code] for code in codes], dtype=np.int64)

def indicators(values: Sequence[str]) -> np.ndarray:
    '''One column per distinct value (sorted), one 1 per row.'''

    categories = sorted(set(values))
    column = {value: index for index, value in enumerate(categories)}
    matrix = np.zeros((len(values), len(categories)), dtype=np.float64)
    matrix[np.arange(len(values)), [column[value] for value in values]] = 1.0
    return matrix

def feature_matrix(comparator: str, frame: pl.DataFrame, level: int, arm: ArmTables) -> np.ndarray:
    '''
    The comparator's features for every row of ``frame``.

    Indicator columns come from the codes in ``frame``, never from outcomes.
    '''

    codes = frame.get_column('code').to_list()
    blocks: List[np.ndarray] = []
    for part in comparator.split('+'):
        if part == COVARIATES:
            blocks.append(np.array(frame.select(COVARIATE_COLUMNS).to_numpy(), dtype=np.float64))
        elif part == 'embedding':
            blocks.append(arm.coordinates[arm.lookup(codes)])
        elif part == 'text_only':
            blocks.append(arm.text_only[arm.lookup(codes)])
        elif part == 'one_hot':
            blocks.append(indicators(codes))
        elif part == 'ancestors':
            blocks.extend(
                indicators([ancestor_at(code, ancestor) for code in codes])
                for ancestor in range(SECTOR_LEVEL, level)
            )
        else:
            raise ValueError(f'unknown comparator part {part!r}')
    return np.hstack(blocks)

def verify_branch_record(
    record: Mapping[str, Any],
    codebook_codes: Collection[str],
    six_digit_population: Collection[str],
) -> None:
    '''
    Require the panel's data to be what Stage 1's finding dictates.

    Args:
        record: The branch record (``RegressorBranchRecord`` fields).
        codebook_codes: The codebook's codes.
        six_digit_population: The six-digit codes with a usable cell in every window year.

    Raises:
        ValueError: If the record names another branch, years, ownership or grain than this
            panel reads, or the population is not the codebook's six-digit codes without the
            record's excluded codes.
    '''

    problems = []
    if record['branch'] != 'A' or not record['time_respecting_outcome']:
        problems.append('the panel implements branch A, with a time-respecting outcome')
    if not record['seen_regime']:
        problems.append('the panel runs the seen-code regime')
    if tuple(record['reference_years']) != WINDOW_YEARS:
        problems.append(f'reference years {record["reference_years"]} are not {WINDOW_YEARS}')
    if record['ownership'] != PRIVATE or record['grain'] != 'national':
        problems.append('the panel reads national private ownership')
    six_digit = set(code for code in codebook_codes if len(code) == 6)
    expected = sorted(six_digit - set(record['excluded_codes']))
    if sorted(six_digit_population) != expected:
        problems.append(
            f'the six-digit population has {len(six_digit_population):,} codes, not the '
            f'{len(expected):,} codebook codes outside the excluded list'
        )
    counts = {record['population_seen'], record['population_heldout']}
    if counts != {len(six_digit_population)}:
        problems.append(f'the record names populations {sorted(counts)}')
    if problems:
        raise ValueError('branch record mismatch: ' + '; '.join(problems))

def run_plan(tasks: Sequence[FitTask], x: np.ndarray, y: np.ndarray,
             alphas: Sequence[float]) -> List[Tuple[FitTask, float, np.ndarray]]:
    '''Each task's chosen penalty and its predictions for the scored rows.'''

    results = []
    for task in tasks:
        errors = np.zeros(len(alphas))
        for fit, scored in task.tuning:
            fit_rows, scored_rows = list(fit), list(scored)
            path = standardized_ridge_path(x[fit_rows], y[fit_rows], x[scored_rows], alphas)
            errors += squared_errors(y[scored_rows], path)
        alpha = float(alphas[best_alpha_index(errors)])
        fit_rows, scored_rows = list(task.fit), list(task.score)
        predictions = standardized_ridge_path(x[fit_rows], y[fit_rows], x[scored_rows], [alpha])
        results.append((task, alpha, predictions[:, 0]))
    return results

# -------------------------------------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------------------------------------

class RegressorPanel:
    '''
    The panel rows at each level, the committed held-out draw, and the log every read goes to.

    Args:
        rows_by_level: Panel rows per level (``qcew_rows.panel_rows``).
        heldout_groups: The committed held-out four-digit groups.
        log: The selection log.
        settings: Tuning and resampling settings.
    '''

    def __init__(
        self,
        rows_by_level: Mapping[int, pl.DataFrame],
        heldout_groups: Collection[str],
        log: SelectionLog,
        settings: FitSettings,
    ):
        self._rows: Dict[int, pl.DataFrame] = {}
        for level, rows in rows_by_level.items():
            if level not in LEVELS:
                raise ValueError(f'level must be one of {LEVELS}, got {level}')
            frame = assign_splits(rows, heldout_groups)
            check_partition(frame, rows.get_column('code').unique().to_list())
            self._rows[level] = frame
        self.heldout_groups: Tuple[str, ...] = tuple(sorted(heldout_groups))
        self.fingerprint = group_table_fingerprint(self.heldout_groups)
        self.log = log
        self.settings = settings
        self._open: Set[Regime] = set()

    @classmethod
    def from_sources(
        cls,
        *,
        qcew_dir: Union[str, Path],
        qcew_sha256: Mapping[str, str],
        codebook_codes: Sequence[str],
        heldout_groups_csv: Union[str, Path],
        log_path: Union[str, Path],
        settings: FitSettings,
        branch_record: Mapping[str, Any],
        levels: Sequence[int] = LEVELS,
    ) -> 'RegressorPanel':
        '''
        The panel from the pinned QCEW slices, a codebook and the committed held-out groups.

        Raises:
            ValueError: If the data are not the population ``branch_record`` names
                (``verify_branch_record``).
        '''

        cells = load_national_cells(Path(qcew_dir), qcew_sha256)
        verify_branch_record(
            branch_record, codebook_codes,
            population(level_cells(cells, codebook_codes, DECISION_LEVEL))
        )
        rows_by_level = {}
        for level in levels:
            cells_at_level = level_cells(cells, codebook_codes, level)
            rows_by_level[level] = panel_rows(cells_at_level, population(cells_at_level))
        return cls(
            rows_by_level,
            read_group_table(Path(heldout_groups_csv)),
            SelectionLog(Path(log_path)),
            settings,
        )

    @property
    def levels(self) -> Tuple[int, ...]:
        return tuple(sorted(self._rows))

    def split_counts(self, level: int) -> Dict[str, int]:
        '''Rows per split at a level (counts only; reading rows goes through the log).'''

        return split_counts(self._frame(level))

    def cell_status(self, regime: Regime, level: int) -> Optional[str]:
        '''None if the regime is defined at the level, otherwise the reason it is not.'''

        regime = Regime(regime)
        if level not in self._rows:
            return f'level {level} is not loaded'
        if regime is Regime.HELDOUT and level < GROUP_LEVEL:
            return 'no four-digit parent: the held-out regime runs at levels 4-6'
        remainder = self._remainder(level)
        if regime is Regime.SEEN:
            remainder = remainder.filter(pl.col('feature_year') == REMAINDER_FEATURE_YEARS[1])
        n_groups = remainder.get_column('group').n_unique()
        if n_groups < self.settings.min_groups:
            return f'{n_groups} remainder groups, fewer than {self.settings.min_groups}'
        return None

    def validation(self, regime: Regime, level: int, arm: ArmTables, purpose: str) -> pl.DataFrame:
        '''Out-of-sample predictions for the remainder rows, logging the read.'''

        regime = Regime(regime)
        self._require_defined(regime, level)
        frame = self._remainder(level)
        if regime is Regime.SEEN:
            plan = seen_validation_plan(frame, level, self.settings)
        else:
            plan = heldout_validation_plan(frame, level, self.settings)
        self._log_read(regime, VALIDATION, level, arm, purpose, frame.height)
        return self._predict(regime, VALIDATION, level, frame, plan, arm)

    def open_outer(
        self, regime: Regime, purpose: str, *, reopen_reason: Optional[str] = None
    ) -> None:
        '''
        Open one regime's sealed outer set for this panel object, logging the opening.

        Raises:
            SplitAlreadyOpenedError: If the log already records an opening of this split and no
                ``reopen_reason`` is given.
        '''

        regime = Regime(regime)
        panel = PANEL_NAMES[regime]
        prior = self.log.openings(panel, self.fingerprint)
        reason = (reopen_reason or '').strip()
        if prior and not reason:
            first = prior[0]
            raise SplitAlreadyOpenedError(
                f'the {panel} outer set was opened at {first["time"]} for {first["purpose"]!r}; '
                'opening it again needs reopen_reason'
            )
        outer = {
            level: frame.filter(pl.col('split') == OUTER_SPLITS[regime].value).height
            for level, frame in sorted(self._rows.items())
        }
        detail: Dict[str, Any] = {'rows_by_level': {str(level): n for level, n in outer.items()}}
        if reason:
            detail['reason'] = reason
        self.log.append(
            SelectionEvent.REOPEN if prior else SelectionEvent.OPEN,
            panel=panel,
            split=TEST,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=outer.get(DECISION_LEVEL, sum(outer.values())),
            detail=detail,
        )
        self._open.add(regime)

    def test(self, regime: Regime, level: int, arm: ArmTables, purpose: str) -> pl.DataFrame:
        '''
        Predictions for the regime's outer set from a fit on the whole remainder, logging the read.

        Raises:
            SealedSplitError: If this panel object has not opened the regime's outer set.
        '''

        regime = Regime(regime)
        self._require_defined(regime, level)
        if regime not in self._open:
            raise SealedSplitError(
                f'the {PANEL_NAMES[regime]} outer set is sealed: call open_outer(regime, purpose) '
                'first, which is logged'
            )
        splits = [RegressorSplit.REMAINDER.value, OUTER_SPLITS[regime].value]
        frame = self._frame(level).filter(pl.col('split').is_in(splits))
        plan = outer_plan(frame, regime, level, self.settings)
        n_outer = len(plan[0].score)
        self._log_read(regime, TEST, level, arm, purpose, n_outer)
        return self._predict(regime, TEST, level, frame, plan, arm)

    def _frame(self, level: int) -> pl.DataFrame:
        if level not in self._rows:
            raise ValueError(f'level {level} is not loaded')
        return self._rows[level]

    def _remainder(self, level: int) -> pl.DataFrame:
        return self._frame(level).filter(pl.col('split') == RegressorSplit.REMAINDER.value)

    def _require_defined(self, regime: Regime, level: int) -> None:
        reason = self.cell_status(regime, level)
        if reason is not None:
            raise ValueError(f'the {regime.value} regime is undefined at level {level}: {reason}')

    def _log_read(
        self, regime: Regime, split: str, level: int, arm: ArmTables, purpose: str, n_rows: int
    ) -> None:
        self.log.append(
            SelectionEvent.READ,
            panel=PANEL_NAMES[regime],
            split=split,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=n_rows,
            detail={
                'level': level,
                'comparators': list(comparators(regime, level)),
                'arm': arm.fingerprint,
                'text_only': arm.text_only_fingerprint,
                'dimension': arm.dimension,
            },
        )

    def _predict(
        self,
        regime: Regime,
        split: str,
        level: int,
        frame: pl.DataFrame,
        plan: Sequence[FitTask],
        arm: ArmTables,
    ) -> pl.DataFrame:
        y = frame.get_column('outcome').to_numpy()
        keys = frame.select('code', 'group', 'feature_year', 'outcome_year', 'outcome')
        parts = []
        for comparator in comparators(regime, level):
            x = feature_matrix(comparator, frame, level, arm)
            for task, alpha, predictions in run_plan(plan, x, y, self.settings.alphas):
                scored = keys[list(task.score)]
                parts.append(
                    scored.with_columns(
                        panel=pl.lit(PANEL_NAMES[regime]),
                        split=pl.lit(split),
                        level=pl.lit(level, dtype=pl.Int32),
                        comparator=pl.lit(comparator),
                        repeat=pl.lit(task.repeat, dtype=pl.Int32),
                        fold=pl.lit(task.fold, dtype=pl.Int32),
                        alpha=pl.lit(alpha, dtype=pl.Float64),
                        prediction=pl.Series(predictions, dtype=pl.Float64),
                    )
                )
        return pl.concat(parts).select(PREDICTION_COLUMNS)

# -------------------------------------------------------------------------------------------------
# From config
# -------------------------------------------------------------------------------------------------

def require_branch_record(cfg: RegressorPanelConfig) -> RegressorBranchRecord:
    '''
    The configured branch record.

    Raises:
        ValueError: If the config has none.
    '''

    if cfg.branch_record is None:
        raise ValueError(
            'the regressor panel config has no branch_record (conf/data/regressor_panel.yaml)'
        )
    return cfg.branch_record

def fit_settings(cfg: RegressorPanelConfig) -> FitSettings:
    '''The config's tuning and resampling settings.'''

    return FitSettings(
        alphas=tuple(cfg.alphas),
        folds=cfg.folds,
        repeats=cfg.repeats,
        inner_folds=cfg.inner_folds,
        fold_seed=cfg.fold_seed,
        min_groups=cfg.min_groups,
    )

def load_regressor_panel(
    cfg: RegressorPanelConfig,
    codebook_path: Union[str, Path],
    *,
    log_path: Optional[Union[str, Path]] = None,
    levels: Sequence[int] = LEVELS,
) -> RegressorPanel:
    '''The panel the config describes, over a codebook, logging to ``log_path`` or the config's.'''

    return RegressorPanel.from_sources(
        qcew_dir=cfg.qcew_dir,
        qcew_sha256=cfg.qcew_sha256,
        codebook_codes=read_codebook_codes(Path(codebook_path), cfg.codebook_codes_sha256),
        heldout_groups_csv=cfg.heldout_groups_csv,
        log_path=log_path or cfg.selection_log,
        settings=fit_settings(cfg),
        branch_record=require_branch_record(cfg).model_dump(),
        levels=levels,
    )

def summarize(predictions: pl.DataFrame) -> pl.DataFrame:
    '''
    Descriptive fit per panel, split, level and comparator: rows, RMSE and R² pooled over repeats.

    A report for reading, not the decision statistic (Stage 4 settles that).
    '''

    keys = ['panel', 'split', 'level', 'comparator']
    # yapf: disable
    return (
        predictions
        .with_columns(
            residual=pl.col('prediction') - pl.col('outcome'),
            centered=pl.col('outcome') - pl.col('outcome').mean().over(keys),
        )
        .group_by(keys, maintain_order=True)
        .agg(
            rows=pl.len(),
            rmse=(pl.col('residual')**2).mean().sqrt(),
            r2=1 - (pl.col('residual')**2).sum() / (pl.col('centered')**2).sum(),
            median_alpha=pl.col('alpha').median(),
        )
    )
    # yapf: enable
```

Replace the whole of `tests/fixtures/regressor_panel.py` with exactly this content. It adds
`SETTINGS`, the stub arm tables and the `regressor_arm` fixture to Task 1's version; the fixture
module now imports `naics_embedder.panels.regressor`, so write the panel first.

```python
'''
A miniature regressor panel: a synthetic NAICS tree, its QCEW national slices and stub arm tables.

Four sectors, one of them combined (31-33), and seventeen four-digit groups. Every group but 2381
has one five-digit code with two six-digit children. 238110 is its five-digit parent's only child,
and BLS publishes it only as 238111 and 238112, so its cell comes from 23811's row. 523211 is
suppressed in 2023, so it leaves the population, as the finding's excluded codes do. Values
follow a noisy size model, so the covariates carry signal.
'''

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels.qcew_rows import (
    SLICE_COLUMNS,
    WINDOW_YEARS,
    level_cells,
    panel_rows,
    population,
    slice_name,
)
from naics_embedder.panels.regressor import ArmTables, FitSettings

GROUPS = (
    '1111',
    '1112',
    '1113',
    '1114',
    '1121',
    '1122',
    '2381',
    '3111',
    '3112',
    '3211',
    '3212',
    '3321',
    '3322',
    '5221',
    '5222',
    '5231',
    '5232',
)
SPLIT_CODE = '238110'
SUPPRESSED_CODE = '523211'
# One held-out group per sector with more than one group: 111, 321 and 523 are tainted at level 3
HELDOUT_GROUPS = ('1113', '3211', '5231')
QCEW_SECTORS = {'31': '31-33'}
AGGLVL = {2: '14', 3: '15', 4: '16', 5: '17', 6: '18'}

def _six_digit(group: str) -> List[str]:
    return [SPLIT_CODE] if group == '2381' else [f'{group}11', f'{group}12']

SIX_DIGIT = tuple(sorted(code for group in GROUPS for code in _six_digit(group)))
FIVE_DIGIT = tuple(sorted({code[:5] for code in SIX_DIGIT}))
SUBSECTORS = tuple(sorted({group[:3] for group in GROUPS}))
SECTORS = ('11', '23', '31', '52')
CODEBOOK = tuple(sorted((*SECTORS, *SUBSECTORS, *GROUPS, *FIVE_DIGIT, *SIX_DIGIT)))
POPULATION = tuple(code for code in SIX_DIGIT if code != SUPPRESSED_CODE)

BRANCH_RECORD = {
    'branch': 'A',
    'source': 'QCEW annual averages',
    'reference_years': list(WINDOW_YEARS),
    'ownership': '5',
    'grain': 'national',
    'population_seen': len(POPULATION),
    'population_heldout': len(POPULATION),
    'time_respecting_outcome': True,
    'seen_regime': True,
    'excluded_codes': [SUPPRESSED_CODE],
}
SETTINGS = FitSettings(
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0), folds=2, repeats=2, inner_folds=2, min_groups=4
)

# -------------------------------------------------------------------------------------------------
# QCEW cells and slices
# -------------------------------------------------------------------------------------------------

def _published(code: str) -> List[Tuple[str, str]]:
    '''The (industry_code, agglvl_code) rows QCEW publishes for a codebook code.'''

    if code == SPLIT_CODE:
        return [(f'{code[:5]}1', AGGLVL[6]), (f'{code[:5]}2', AGGLVL[6])]
    return [(QCEW_SECTORS.get(code, code), AGGLVL[len(code)])]

def synthetic_cells() -> pl.DataFrame:
    '''The private national annual rows ``read_national_slice`` returns, every window year.'''

    rng = np.random.default_rng(20260924)
    rows = []
    for code in CODEBOOK:
        for industry_code, agglvl_code in _published(code):
            size = rng.normal(7.0, 1.5)
            growth = rng.normal(0.02, 0.05)
            for year in WINDOW_YEARS:
                log_emp = size + growth * (year - WINDOW_YEARS[0]) + rng.normal(0.0, 0.05)
                estabs = int(round(np.exp(size - 2.3 + rng.normal(0.0, 0.1))))
                wages = int(round(np.exp(size + 10.8 + rng.normal(0.0, 0.1))))
                cell = ('', estabs, int(round(np.exp(log_emp))), wages)
                if code == SUPPRESSED_CODE and year == 2023:
                    cell = ('N', 0, 0, 0)
                rows.append((industry_code, agglvl_code, year, *cell))
    return pl.DataFrame(
        rows,
        schema={
            'industry_code': pl.Utf8,
            'agglvl_code': pl.Utf8,
            'year': pl.Int32,
            'disclosure_code': pl.Utf8,
            'estabs': pl.Int64,
            'emp': pl.Int64,
            'wages': pl.Int64,
        },
        orient='row',
    )

def _slice_rows(cells: pl.DataFrame, year: int) -> List[Dict[str, str]]:
    rows = []
    for cell in cells.filter(pl.col('year') == year).iter_rows(named=True):
        base = {
            'area_fips': 'US000',
            'own_code': '5',
            'industry_code': cell['industry_code'],
            'agglvl_code': cell['agglvl_code'],
            'size_code': '0',
            'year': str(year),
            'qtr': 'A',
            'disclosure_code': cell['disclosure_code'],
            'annual_avg_estabs': str(cell['estabs']),
            'annual_avg_emplvl': str(cell['emp']),
            'total_annual_wages': str(cell['wages']),
        }
        rows.append(base)
        # Rows the reader must drop: all ownerships, a state, a quarter
        rows.append({**base, 'own_code': '0', 'annual_avg_emplvl': '1'})
        rows.append({**base, 'area_fips': '01000', 'annual_avg_emplvl': '1'})
        rows.append({**base, 'qtr': '1', 'annual_avg_emplvl': '1'})
    return rows

def write_qcew_slices(directory: Path, cells: pl.DataFrame) -> Dict[str, str]:
    '''Write each window year's national slice; return their sha256 pins by file name.'''

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schema = {name: pl.Utf8 for name in SLICE_COLUMNS}
    pins = {}
    for year in WINDOW_YEARS:
        frame = pl.DataFrame(_slice_rows(cells, year), schema=schema)
        path = directory / slice_name(year)
        frame.write_csv(path)
        pins[slice_name(year)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pins

def synthetic_rows(cells: pl.DataFrame, levels=(2, 3, 4, 5, 6)) -> Dict[int, pl.DataFrame]:
    '''Panel rows per level, as ``RegressorPanel.from_sources`` builds them.'''

    rows = {}
    for level in levels:
        cells_at_level = level_cells(cells, CODEBOOK, level)
        rows[level] = panel_rows(cells_at_level, population(cells_at_level))
    return rows

# -------------------------------------------------------------------------------------------------
# Stub arm tables
# -------------------------------------------------------------------------------------------------

def coordinate_table(codes, dimension: int = 3, seed: int = 7) -> pl.DataFrame:
    '''``code`` plus ``e0`` … ``e{dimension-1}``: a stub arm in the export form.'''

    values = np.random.default_rng(seed).normal(size=(len(codes), dimension))
    schema = {f'e{index}': pl.Float64 for index in range(dimension)}
    frame = pl.DataFrame(values, schema=schema, orient='row')
    return pl.DataFrame({'code': list(codes)}, schema={'code': pl.Utf8}).hstack(frame)

def text_only_table(codes, width: int = 5, seed: int = 11) -> pl.DataFrame:
    '''``code`` plus ``t0`` … ``t{width-1}``: a stub text-only table.'''

    values = np.random.default_rng(seed).normal(size=(len(codes), width))
    schema = {f't{index}': pl.Float64 for index in range(width)}
    frame = pl.DataFrame(values, schema=schema, orient='row')
    return pl.DataFrame({'code': list(codes)}, schema={'code': pl.Utf8}).hstack(frame)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture(scope='session')
def regressor_cells() -> pl.DataFrame:
    return synthetic_cells()

@pytest.fixture(scope='session')
def regressor_rows(regressor_cells) -> Dict[int, pl.DataFrame]:
    return synthetic_rows(regressor_cells)

@pytest.fixture(scope='session')
def regressor_arm() -> ArmTables:
    return ArmTables.from_tables(coordinate_table(CODEBOOK), text_only_table(CODEBOOK))
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_regressor_panel.py -q`
Expected: `38 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1485 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/regressor.py tests/fixtures/regressor_panel.py tests/unit/test_regressor_panel.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/regressor.py tests/fixtures/regressor_panel.py tests/unit/test_regressor_panel.py
git commit -m "feat(panels): add RegressorPanel with two regimes and logged outer-set openings"
```

### Task 7: `data regressor-groups` draws the held-out groups once

This task draws the held-out four-digit groups from the six-digit population, writes the table
and its provenance, and refuses to redraw an existing table without `--force`.

**Files:**

- Create: `src/naics_embedder/data/regressor_group_table.py`
- Modify: `src/naics_embedder/cli/commands/data.py`
- Modify: `tests/unit/test_cli_commands.py`
- Test: `tests/unit/test_regressor_group_table.py`

**Interfaces:**

- Consumes: Task 1's `level_cells`, `load_national_cells`, `panel_rows` and `population`; Task 2's
  `SECTOR_LEVEL`, `ancestor_at`, `assign_splits`, `draw_heldout_groups`, `read_codebook_codes`,
  `split_counts` and `write_group_table`; Task 5's `RegressorPanelConfig`; Task 6's
  `DECISION_LEVEL`, `LEVELS`, `require_branch_record` and `verify_branch_record`;
  `generator_revision()` (existing, `naics_embedder.data.supervision_bundle`).
- Produces:
  - `generate_regressor_group_table(cfg: RegressorPanelConfig, codebook_path: Path, *,
    force: bool = False) -> Path`, which writes `cfg.heldout_groups_csv` and
    `cfg.provenance_json`
  - the command `naics-embedder data regressor-groups --codebook PATH [--force]`

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_regressor_group_table.py` with exactly this content:

```python
'''
``data regressor-groups``: the held-out draw, written once with its provenance (roadmap Stage 3).
'''

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from naics_embedder.data.regressor_group_table import generate_regressor_group_table
from naics_embedder.panels.regressor_splits import (
    assign_splits,
    codes_fingerprint,
    read_group_table,
    split_counts,
)
from naics_embedder.utils.config import RegressorBranchRecord, RegressorPanelConfig
from tests.fixtures.regressor_panel import BRANCH_RECORD, CODEBOOK, write_qcew_slices

pytestmark = pytest.mark.unit

@pytest.fixture
def draw_inputs(tmp_path, regressor_cells):
    pins = write_qcew_slices(tmp_path / 'qcew', regressor_cells)
    codebook = tmp_path / 'naics_codebook.parquet'
    pl.DataFrame({'code': list(CODEBOOK)}).write_parquet(codebook)
    cfg = RegressorPanelConfig(
        qcew_dir=str(tmp_path / 'qcew'),
        qcew_sha256=pins,
        codebook_codes_sha256=codes_fingerprint(CODEBOOK),
        heldout_groups_csv=str(tmp_path / 'conf' / 'regressor_heldout_groups.csv'),
        provenance_json=str(tmp_path / 'conf' / 'regressor_heldout_groups_provenance.json'),
        branch_record=RegressorBranchRecord(**BRANCH_RECORD),
    )
    return cfg, codebook

def test_the_draw_writes_the_table_and_its_provenance(draw_inputs, regressor_rows):
    cfg, codebook = draw_inputs

    path = generate_regressor_group_table(cfg, codebook)

    groups = read_group_table(path)
    # A fifth of 6, 1, 6 and 4 groups floors to 2; a fifth of 17, rounded, is 3
    assert len(groups) == 3
    provenance = json.loads(Path(cfg.provenance_json).read_text())
    assert provenance['heldout_groups'] == {
        'path': cfg.heldout_groups_csv,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'groups': 3,
    }
    assert provenance['groups_by_sector'] == {'11': 1, '31': 1, '52': 1}
    assert (provenance['seed'], provenance['fraction']) == (20260924, '1/5')
    assert provenance['six_digit_population'] == 32
    assert provenance['rows_by_level_and_split'] == {
        str(level): split_counts(assign_splits(rows, groups))
        for level, rows in regressor_rows.items()
    }
    assert provenance['qcew_sha256'] == cfg.qcew_sha256
    assert provenance['codebook_codes_sha256'] == cfg.codebook_codes_sha256
    assert {'generator_revision', 'library_versions', 'generated_at'} <= set(provenance)

def test_an_existing_table_is_redrawn_only_with_force(draw_inputs):
    cfg, codebook = draw_inputs
    path = generate_regressor_group_table(cfg, codebook)
    first = path.read_bytes()

    with pytest.raises(FileExistsError, match='drawn once'):
        generate_regressor_group_table(cfg, codebook)
    generate_regressor_group_table(cfg, codebook, force=True)

    assert path.read_bytes() == first

def test_the_draw_needs_the_branch_records_population(draw_inputs):
    cfg, codebook = draw_inputs
    wrong = RegressorBranchRecord(**{**BRANCH_RECORD, 'excluded_codes': []})

    with pytest.raises(ValueError, match='branch record mismatch'):
        generate_regressor_group_table(cfg.model_copy(update={'branch_record': wrong}), codebook)
    with pytest.raises(ValueError, match='no branch_record'):
        generate_regressor_group_table(cfg.model_copy(update={'branch_record': None}), codebook)
    assert not Path(cfg.heldout_groups_csv).exists()
```

**`tests/unit/test_cli_commands.py`, edit 1 of 1.** Replace:

```python
def test_data_roles_refuses_to_redraw_without_force(monkeypatch, runner):

    def refuse(download_cfg, panel_cfg, force):
        raise FileExistsError('the role table exists; pass --force to redraw it')

    monkeypatch.setattr(data_cli, 'generate_index_role_table', refuse)

    result = runner.invoke(data_cli.app, ['roles'])

    assert result.exit_code == 1
    assert '--force' in result.output

```

with:

```python
def test_data_roles_refuses_to_redraw_without_force(monkeypatch, runner):

    def refuse(download_cfg, panel_cfg, force):
        raise FileExistsError('the role table exists; pass --force to redraw it')

    monkeypatch.setattr(data_cli, 'generate_index_role_table', refuse)

    result = runner.invoke(data_cli.app, ['roles'])

    assert result.exit_code == 1
    assert '--force' in result.output

def test_data_regressor_groups_draws_with_the_regressor_config(monkeypatch, runner, tmp_path):
    calls = []

    def fake_generate(cfg, codebook_path, force):
        calls.append((cfg, codebook_path, force))
        return tmp_path / 'regressor_heldout_groups.csv'

    monkeypatch.setattr(data_cli, 'generate_regressor_group_table', fake_generate)

    result = runner.invoke(
        data_cli.app, ['regressor-groups', '--codebook', '/bundle/naics_codebook.parquet']
    )

    assert result.exit_code == 0, result.output
    [(cfg, codebook_path, force)] = calls
    assert cfg.seed == 20260924
    assert cfg.branch_record is not None
    assert codebook_path == Path('/bundle/naics_codebook.parquet')
    assert force is False
    assert 'regressor_heldout_groups.csv' in result.output

def test_data_regressor_groups_refuses_to_redraw_without_force(monkeypatch, runner):

    def refuse(cfg, codebook_path, force):
        raise FileExistsError('the table exists; pass --force only to redraw it deliberately')

    monkeypatch.setattr(data_cli, 'generate_regressor_group_table', refuse)

    result = runner.invoke(data_cli.app, ['regressor-groups', '--codebook', 'codebook.parquet'])

    assert result.exit_code == 1
    assert '--force' in result.output

```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_regressor_group_table.py -q`
Expected: one collection error, `ModuleNotFoundError: No module named
'naics_embedder.data.regressor_group_table'`.

Run: `uv run pytest tests/unit/test_cli_commands.py -q -k regressor_groups`
Expected: `2 failed`, each with `AttributeError: … has no attribute
'generate_regressor_group_table'`.

- [x] **Step 3: Write the generator and the command**

Create `src/naics_embedder/data/regressor_group_table.py` with exactly this content:

```python
'''
Draw the regressor panel's held-out four-digit groups (roadmap Stage 3; Req 2; Req 4).

``naics-embedder data regressor-groups`` runs this once. The table it writes
(``conf/data/regressor_heldout_groups.csv``) is committed, and the panel reads it from then on,
so the held-out regime's outer set never moves. The seen regime's outer set leaves out the
held-out groups too, so a redraw moves both outer sets, and its new fingerprint would not count
as a reopening: an existing table is replaced only with ``force``.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict

from naics_embedder.data.supervision_bundle import generator_revision
from naics_embedder.panels.qcew_rows import (
    level_cells,
    load_national_cells,
    panel_rows,
    population,
)
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    LEVELS,
    require_branch_record,
    verify_branch_record,
)
from naics_embedder.panels.regressor_splits import (
    SECTOR_LEVEL,
    ancestor_at,
    assign_splits,
    draw_heldout_groups,
    read_codebook_codes,
    split_counts,
    write_group_table,
)
from naics_embedder.utils.config import RegressorPanelConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Generate
# -------------------------------------------------------------------------------------------------

def generate_regressor_group_table(
    cfg: RegressorPanelConfig,
    codebook_path: Path,
    *,
    force: bool = False,
) -> Path:
    '''
    Draw the held-out four-digit groups and write the table and its provenance.

    Returns:
        The table's path (``cfg.heldout_groups_csv``).

    Raises:
        FileExistsError: If the table exists and ``force`` is False.
        ValueError: If a pinned hash differs or the data are not the branch record's population.
    '''

    table_path = Path(cfg.heldout_groups_csv)
    if table_path.exists() and not force:
        raise FileExistsError(
            f'{table_path} exists: the held-out groups are drawn once and committed. Redrawing '
            'moves both regressor outer sets; pass --force only to do that deliberately.'
        )
    record = require_branch_record(cfg)
    codes = read_codebook_codes(Path(codebook_path), cfg.codebook_codes_sha256)
    cells = load_national_cells(Path(cfg.qcew_dir), cfg.qcew_sha256)
    six_digit = population(level_cells(cells, codes, DECISION_LEVEL))
    verify_branch_record(record.model_dump(), codes, six_digit)

    fraction = Fraction(str(cfg.heldout_fraction))
    groups = draw_heldout_groups(six_digit, fraction, cfg.seed)
    fingerprint = write_group_table(groups, table_path)

    partition: Dict[str, Dict[str, int]] = {}
    for level in LEVELS:
        cells_at_level = level_cells(cells, codes, level)
        rows = assign_splits(panel_rows(cells_at_level, population(cells_at_level)), groups)
        partition[str(level)] = split_counts(rows)
    by_sector: Dict[str, int] = {}
    for group in groups:
        sector = ancestor_at(group, SECTOR_LEVEL)
        by_sector[sector] = by_sector.get(sector, 0) + 1

    provenance: Dict[str, Any] = {
        'heldout_groups': {
            'path': str(table_path),
            'sha256': fingerprint,
            'groups': len(groups)
        },
        'seed': cfg.seed,
        'fraction': str(fraction),
        'groups_by_sector': dict(sorted(by_sector.items())),
        'six_digit_population': len(six_digit),
        'rows_by_level_and_split': partition,
        'qcew_sha256': dict(sorted(cfg.qcew_sha256.items())),
        'codebook_codes_sha256': cfg.codebook_codes_sha256,
        'generator_revision': generator_revision(),
        'library_versions': {
            name: version(name)
            for name in ('numpy', 'polars')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    provenance_path = Path(cfg.provenance_json)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')

    logger.info(f'Held-out groups: {len(groups)} ({fingerprint}) written to {table_path}')
    logger.info(f'Rows by level and split: {partition}')
    logger.info(f'Provenance written to: {provenance_path}\n')
    return table_path
```

Modify `src/naics_embedder/cli/commands/data.py` with these 4 edits, in order. Each replaced text
occurs exactly once in the file.

**`src/naics_embedder/cli/commands/data.py`, edit 1 of 4.** Replace:

```python
    roles: Draw the frozen index-entry role table, once; it is committed and preprocess applies
        it.
```

with:

```python
    roles: Draw the frozen index-entry role table, once; it is committed and preprocess applies
        it.
    regressor-groups: Draw the regressor panel's held-out four-digit groups, once; they are
        committed and the panel reads them.
```

**`src/naics_embedder/cli/commands/data.py`, edit 2 of 4.** Replace:

```python
from naics_embedder.data.index_role_table import generate_index_role_table
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.utils.config import (
    DownloadConfig,
    OutcomePanelConfig,
    SupervisionBuildConfig,
    load_config,
)
```

with:

```python
from naics_embedder.data.index_role_table import generate_index_role_table
from naics_embedder.data.regressor_group_table import generate_regressor_group_table
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.utils.config import (
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
    SupervisionBuildConfig,
    load_config,
)
```

**`src/naics_embedder/cli/commands/data.py`, edit 3 of 4.** Replace:

```python
OUTCOME_PANEL_CONFIG = 'data/outcome_panel.yaml'
```

with:

```python
OUTCOME_PANEL_CONFIG = 'data/outcome_panel.yaml'
REGRESSOR_PANEL_CONFIG = 'data/regressor_panel.yaml'
```

**`src/naics_embedder/cli/commands/data.py`, edit 4 of 4.** Replace:

```python
    typer.echo(f'Index-entry role table: {table_path}')

# -------------------------------------------------------------------------------------------------
```

with:

```python
    typer.echo(f'Index-entry role table: {table_path}')

# -------------------------------------------------------------------------------------------------
# Draw the regressor panel's held-out groups
# -------------------------------------------------------------------------------------------------

@app.command('regressor-groups')
def regressor_groups(
    codebook: Annotated[
        str,
        typer.Option('--codebook', help="A supervision bundle's naics_codebook.parquet"),
    ],
    force: Annotated[
        bool,
        typer.Option(
            '--force',
            help='Redraw an existing table: moves both regressor outer sets',
        ),
    ] = False,
):
    '''
    Draw the regressor panel's held-out four-digit groups, once.

    Holds out a fifth of each sector's four-digit groups (largest remainder, seeded) from the
    six-digit population Stage 1's finding names. The table is committed, and the regressor
    panel reads it from then on (roadmap Stage 3).

    Output:
        ``conf/data/regressor_heldout_groups.csv`` and
        ``conf/data/regressor_heldout_groups_provenance.json``.

    Example:
        Draw the groups over bundle 18403d29's codebook::

            $ uv run naics-embedder data regressor-groups --codebook PATH/naics_codebook.parquet
    '''

    configure_logging('data_regressor_groups.log')

    console.rule('[bold green]Drawing Regressor Held-Out Groups[/bold green]')

    try:
        table_path = generate_regressor_group_table(
            load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG),
            Path(codebook),
            force=force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        console.print(f'[bold red]{exc}[/bold red]')
        raise typer.Exit(code=1)

    typer.echo(f'Regressor held-out groups: {table_path}')

# -------------------------------------------------------------------------------------------------
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_regressor_group_table.py tests/unit/test_cli_commands.py -q`
Expected: `28 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1490 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/data/regressor_group_table.py src/naics_embedder/cli/commands/data.py tests/unit/test_regressor_group_table.py tests/unit/test_cli_commands.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/data/regressor_group_table.py src/naics_embedder/cli/commands/data.py tests/unit/test_regressor_group_table.py tests/unit/test_cli_commands.py
git commit -m "feat(data): add data regressor-groups to draw the held-out groups once"
```

### Task 8: `tools text-only-table` and `tools regressor-panel`

This task adds the two commands an arm is scored with. The test split needs `--open-purpose`, and
each regime is opened, and logged, before its first test read.

**Files:**

- Modify: `src/naics_embedder/cli/commands/tools.py`
- Modify: `tests/unit/test_cli_commands.py`

**Interfaces:**

- Consumes: Task 4's `build_text_only_table` and `provenance_path`; Task 5's
  `RegressorPanelConfig`; Task 6's `DECISION_LEVEL`, `TEST`, `VALIDATION`, `ArmTables`, `Regime`,
  `load_regressor_panel`, `summarize` and `RegressorPanel`; `SealedSplitError` and
  `SplitAlreadyOpenedError` (existing, `naics_embedder.panels.outcome`); the fixture module's
  `CODEBOOK`, `HELDOUT_GROUPS`, `SETTINGS`, `coordinate_table`, `text_only_table` and
  `regressor_rows`.
- Produces:
  - `naics-embedder tools text-only-table --descriptions PATH --output PATH [--backbone NAME]`
  - `naics-embedder tools regressor-panel --coordinates PATH --text-only PATH --codebook PATH
    [--regime seen|heldout]… [--level N]… [--split validation|test] [--purpose TEXT]
    [--open-purpose TEXT] [--reopen-reason TEXT] [--log PATH] [--output PATH]`

- [x] **Step 1: Write the failing tests**

Modify `tests/unit/test_cli_commands.py` with these 2 edits, in order. Each replaced text occurs
exactly once in the file.

**`tests/unit/test_cli_commands.py`, edit 1 of 2.** Replace:

```python
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle
```

with:

```python
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.panels.regressor import RegressorPanel
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle
from tests.fixtures.regressor_panel import (
    CODEBOOK,
    HELDOUT_GROUPS,
    SETTINGS,
    coordinate_table,
    text_only_table,
)
```

**`tests/unit/test_cli_commands.py`, edit 2 of 2.** Replace:

```python
    assert 'Outcome baseline failed' in result.output
    assert SelectionLog(tmp_path / 'selection_log.jsonl').records() == []
```

with:

```python
    assert 'Outcome baseline failed' in result.output
    assert SelectionLog(tmp_path / 'selection_log.jsonl').records() == []

# -------------------------------------------------------------------------------------------------
# Regressor panel
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_text_only_table_embeds_with_the_regressor_configs_backbone(monkeypatch, runner, tmp_path):
    calls = []

    def fake_build(descriptions, output, *, backbone, max_length, batch_size):
        calls.append((descriptions, output, backbone, max_length, batch_size))
        return output

    monkeypatch.setattr(tools_cli, 'build_text_only_table', fake_build)
    output = tmp_path / 'text_only.parquet'

    result = runner.invoke(
        tools_cli.app,
        ['text-only-table', '--descriptions', 'descriptions.parquet', '--output',
         str(output)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (Path('descriptions.parquet'), output, 'sentence-transformers/all-MiniLM-L6-v2', 512, 32)
    ]
    assert 'text_only_provenance.json' in result.output

def _regressor_arguments(tmp_path):
    coordinates = tmp_path / 'arm.parquet'
    text_only = tmp_path / 'text_only.parquet'
    coordinate_table(CODEBOOK).write_parquet(coordinates)
    text_only_table(CODEBOOK).write_parquet(text_only)
    return [
        'regressor-panel',
        '--coordinates',
        str(coordinates),
        '--text-only',
        str(text_only),
        '--codebook',
        'naics_codebook.parquet',
    ]

@pytest.fixture
def fixture_panel(monkeypatch, tmp_path, regressor_rows):
    '''The fixture panel in place of the real one: every opening below is a fixture's.'''

    log = SelectionLog(tmp_path / 'selection_log.jsonl')
    loaded = []

    def fake_load(cfg, codebook, *, log_path, levels):
        loaded.append((codebook, log_path, tuple(levels)))
        return RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)

    monkeypatch.setattr(tools_cli, 'load_regressor_panel', fake_load)
    return log, loaded

@pytest.mark.unit
def test_regressor_panel_needs_an_open_purpose_for_the_test_split(runner, tmp_path, fixture_panel):
    log, loaded = fixture_panel

    result = runner.invoke(tools_cli.app, [*_regressor_arguments(tmp_path), '--split', 'test'])

    assert result.exit_code == 1
    assert '--open-purpose' in result.output
    assert loaded == []
    assert log.records() == []

@pytest.mark.unit
def test_regressor_panel_scores_validation_and_reports_undefined_cells(
    runner, tmp_path, fixture_panel
):
    log, loaded = fixture_panel
    output = tmp_path / 'predictions.parquet'
    levels = ['--level', '6', '--level', '3', '--level', '2']

    result = runner.invoke(
        tools_cli.app, [*_regressor_arguments(tmp_path), *levels, '--output',
                        str(output)]
    )

    assert result.exit_code == 0, result.output
    assert loaded == [('naics_codebook.parquet', None, (2, 3, 6))]
    assert 'seen, level 2: undefined' in result.output
    assert 'heldout, level 3: undefined' in result.output
    assert [(r['event'], r['panel'], r['split'], r['detail']['level']) for r in log.records()] == [
        ('read', 'regressor_seen', 'validation', 3),
        ('read', 'regressor_seen', 'validation', 6),
        ('read', 'regressor_heldout', 'validation', 6),
    ]
    assert set(pl.read_parquet(output).get_column('split')) == {'validation'}

@pytest.mark.unit
def test_regressor_panel_opens_each_regime_once_before_its_test_read(
    runner, tmp_path, fixture_panel
):
    log, _ = fixture_panel
    arguments = [
        *_regressor_arguments(tmp_path), '--split', 'test', '--open-purpose', 'fixture opening'
    ]

    first = runner.invoke(tools_cli.app, arguments)
    second = runner.invoke(tools_cli.app, arguments)

    assert first.exit_code == 0, first.output
    assert [(r['event'], r['panel'], r['split']) for r in log.records()] == [
        ('open', 'regressor_seen', 'test'),
        ('read', 'regressor_seen', 'test'),
        ('open', 'regressor_heldout', 'test'),
        ('read', 'regressor_heldout', 'test'),
    ]
    assert second.exit_code == 1
    assert 'reopen_reason' in second.output
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_commands.py -q`
Expected: `1 failed, 25 passed, 3 errors`. The text-only test fails, and the three
regressor-panel tests error in their fixture: `tools` has no attribute `build_text_only_table`
or `load_regressor_panel` yet.

- [x] **Step 3: Add the commands**

> Deviation: the whole-plan review changed `tools regressor-panel` after this task. Before opening anything it checks the output path (c5ee998, 099fa12), the arm (c5ee998) and every regime's opening (b7c21d6); it deduplicates `--regime` (b7c21d6) and catches `OSError`; and `--purpose` defaults to `regressor panel <split> read` (1ce40e5).

Modify `src/naics_embedder/cli/commands/tools.py` with these 6 edits, in order. Each replaced
text occurs exactly once in the file.

**`src/naics_embedder/cli/commands/tools.py`, edit 1 of 6.** Replace:

```python
    outcome-baseline: Score the lexical stub encoder on the outcome panel's validation split.
```

with:

```python
    outcome-baseline: Score the lexical stub encoder on the outcome panel's validation split.
    text-only-table: Embed every code's text with the arm's backbone, frozen (roadmap D9).
    regressor-panel: Score an arm on the regressor panel's validation or sealed test split.
```

**`src/naics_embedder/cli/commands/tools.py`, edit 2 of 6.** Replace:

```python
from typing import Optional, Tuple
```

with:

```python
from typing import List, Optional, Tuple
```

**`src/naics_embedder/cli/commands/tools.py`, edit 3 of 6.** Replace:

```python
from naics_embedder.panels.outcome import OutcomePanel
```

with:

```python
from naics_embedder.panels.outcome import OutcomePanel, SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    TEST,
    VALIDATION,
    ArmTables,
    Regime,
    load_regressor_panel,
    summarize,
)
from naics_embedder.panels.text_only import build_text_only_table
from naics_embedder.panels.text_only import provenance_path as text_only_provenance_path
```

**`src/naics_embedder/cli/commands/tools.py`, edit 4 of 6.** Replace:

```python
from naics_embedder.utils.config import DownloadConfig, OutcomePanelConfig, load_config
```

with:

```python
from naics_embedder.utils.config import (
    DownloadConfig,
    OutcomePanelConfig,
    RegressorPanelConfig,
    load_config,
)
```

**`src/naics_embedder/cli/commands/tools.py`, edit 5 of 6.** Replace:

```python
    help='Utility tools for configuration, metrics analysis, and debugging.', no_args_is_help=True
)

# -------------------------------------------------------------------------------------------------
```

with:

```python
    help='Utility tools for configuration, metrics analysis, and debugging.', no_args_is_help=True
)

REGRESSOR_PANEL_CONFIG = 'data/regressor_panel.yaml'

# -------------------------------------------------------------------------------------------------
```

**`src/naics_embedder/cli/commands/tools.py`, edit 6 of 6.** Replace:

```python
        payload = {'fingerprint': panel.fingerprint, 'summary': result.summary}
        path.write_text(json.dumps(payload, indent=2) + '\n')
```

with:

```python
        payload = {'fingerprint': panel.fingerprint, 'summary': result.summary}
        path.write_text(json.dumps(payload, indent=2) + '\n')

# -------------------------------------------------------------------------------------------------
# Regressor panel
# -------------------------------------------------------------------------------------------------

@app.command('text-only-table')
def text_only_table(
    descriptions: Annotated[
        str,
        typer.Option('--descriptions', help="The arm's descriptions parquet: the text it reads"),
    ],
    output: Annotated[
        str,
        typer.Option('--output', help='Where to write the table (parquet)'),
    ],
    backbone: Annotated[
        Optional[str],
        typer.Option('--backbone', help="The arm's backbone (default: the regressor config)"),
    ] = None,
):
    '''
    Embed every code's text with the arm's backbone, frozen (roadmap D9).

    Each of the four channels is mean-pooled over its tokens, and a code's vector is the mean of
    its present channels. The backbone is read from the local Hugging Face cache. The regressor
    panel reduces the table to the arm's dimension by PCA.

    Output:
        The table, and ``<stem>_provenance.json`` beside it.

    Example:
        Embed the text bundle 18403d29 was built from::

            $ uv run naics-embedder tools text-only-table \\
                --descriptions data/naics_descriptions.parquet --output /tmp/text_only.parquet
    '''

    configure_logging('tools_text_only_table.log')

    cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
    try:
        path = build_text_only_table(
            Path(descriptions),
            Path(output),
            backbone=backbone or cfg.text_only.backbone,
            max_length=cfg.text_only.max_length,
            batch_size=cfg.text_only.batch_size,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        console.print(f'[bold red]Text-only table failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print(f'Text-only table: {path}')
    console.print(f'Provenance: {text_only_provenance_path(path)}')

@app.command('regressor-panel')
def regressor_panel(
    coordinates: Annotated[
        str,
        typer.Option(
            '--coordinates',
            help="The arm's code table in the export form (tangent coordinates if hyperbolic)",
        ),
    ],
    text_only: Annotated[
        str,
        typer.Option('--text-only', help='The text-only table (tools text-only-table)'),
    ],
    codebook: Annotated[
        str,
        typer.Option('--codebook', help="A supervision bundle's naics_codebook.parquet"),
    ],
    regime: Annotated[
        Optional[List[Regime]],
        typer.Option('--regime', help='Regime to score (repeatable; default: both)'),
    ] = None,
    level: Annotated[
        Optional[List[int]],
        typer.Option('--level', help='NAICS level 2-6 (repeatable; default: 6)'),
    ] = None,
    split: Annotated[
        str,
        typer.Option('--split', help='validation, or test (sealed: needs --open-purpose)'),
    ] = VALIDATION,
    purpose: Annotated[
        str,
        typer.Option('--purpose', help='Why this read happens; recorded in the selection log'),
    ] = 'regressor panel validation read',
    open_purpose: Annotated[
        Optional[str],
        typer.Option('--open-purpose', help='Why the outer sets are opened (test split only)'),
    ] = None,
    reopen_reason: Annotated[
        Optional[str],
        typer.Option('--reopen-reason', help='Required to open an outer set a second time'),
    ] = None,
    log: Annotated[
        Optional[str],
        typer.Option('--log', help='Selection log (default: the regressor config)'),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option('--output', help='Write the per-row predictions to this parquet'),
    ] = None,
):
    '''
    Score an arm on the regressor panel: out-of-sample predictions per row (roadmap Stage 3).

    Every Req 2 comparator is fitted by ridge on standardized features, the penalty tuned inside
    the remainder. The validation split reads only the remainder; the test split opens each
    regime's sealed outer set first, and both the opening and the read are logged.

    Example:
        Score an arm's table on the validation split of both regimes at six digits::

            $ uv run naics-embedder tools regressor-panel --coordinates arm.parquet \\
                --text-only text_only.parquet --codebook PATH/naics_codebook.parquet
    '''

    configure_logging('tools_regressor_panel.log')

    if split not in (VALIDATION, TEST):
        console.print(f'[bold red]--split must be {VALIDATION} or {TEST}, not {split!r}[/bold red]')
        raise typer.Exit(code=1)
    if split == TEST and not (open_purpose or '').strip():
        console.print(
            '[bold red]The outer sets are sealed: --split test needs --open-purpose, which is '
            'logged.[/bold red]'
        )
        raise typer.Exit(code=1)

    cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
    regimes = regime or list(Regime)
    levels = sorted(set(level or [DECISION_LEVEL]))
    try:
        panel = load_regressor_panel(cfg, codebook, log_path=log, levels=levels)
        arm = ArmTables.from_tables(pl.read_parquet(coordinates), pl.read_parquet(text_only))
        results, undefined = [], []
        for chosen in regimes:
            defined = []
            for number in levels:
                reason = panel.cell_status(chosen, number)
                if reason is None:
                    defined.append(number)
                else:
                    undefined.append((chosen.value, number, reason))
            if split == TEST and defined:
                panel.open_outer(chosen, open_purpose or '', reopen_reason=reopen_reason)
            for number in defined:
                read = panel.validation if split == VALIDATION else panel.test
                results.append(read(chosen, number, arm, purpose))
    except (FileNotFoundError, ValueError, SealedSplitError, SplitAlreadyOpenedError) as exc:
        console.print(f'[bold red]Regressor panel failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    for name, number, reason in undefined:
        console.print(f'  • {name}, level {number}: undefined ({reason})')
    if not results:
        console.print('[bold yellow]No regime was defined at the requested levels.[/bold yellow]')
        raise typer.Exit(code=1)

    predictions = pl.concat(results)
    console.print(f'\n[bold cyan]Regressor panel: {split} split[/bold cyan]\n')
    for row in summarize(predictions).iter_rows(named=True):
        console.print(
            f'  • {row["panel"]}, level {row["level"]}, {row["comparator"]}: rows {row["rows"]:,}, '
            f'RMSE {row["rmse"]:.4f}, R² {row["r2"]:.4f}, median alpha {row["median_alpha"]:g}'
        )
    console.print(f'\nReads logged to {panel.log.path} (fingerprint {panel.fingerprint})\n')

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.write_parquet(path)
        console.print(f'Predictions written to {path}')
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli_commands.py -q`
Expected: `29 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1494 passed, 1 skipped`.

- [x] **Step 5: Check the formatting**

Run: `./scripts/format_code.sh --check src/naics_embedder/cli/commands/tools.py tests/unit/test_cli_commands.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/cli/commands/tools.py tests/unit/test_cli_commands.py
git commit -m "feat(cli): add tools text-only-table and tools regressor-panel"
```

### Task 9: Draw the real groups, run the stub arm, and record the finding (controller, inline)

This is the only task that reads the real QCEW slices, the bundle's codebook and the backbone. It
draws the held-out groups once and commits them, with the provenance, a test that pins the
table's hash, and the finding. Everything else it writes is scratch under
`/tmp/stage3-regressor-panel-635ffd43/`: the text-only table, the stub arm, the selection log,
the predictions and two scripts. Final verification reads that log and then removes the
directory.

It reads the validation split only. Never pass `--split test`, and never call `open_outer` or
`test`, on the real panel: Stage 12 opens the outer sets.

**Files:**

- Create (generated by Step 2): `conf/data/regressor_heldout_groups.csv` and
  `conf/data/regressor_heldout_groups_provenance.json`
- Create: `tests/unit/test_committed_regressor_groups.py`
- Create: `specs/findings/regressor-panel-splits.md`

**Interfaces:**

- Consumes:
  - `naics-embedder data regressor-groups` (Task 7)
  - `naics-embedder tools text-only-table` and `tools regressor-panel` (Task 8)
  - `read_group_table`, `group_table_fingerprint`, `ancestor_at` and `SECTOR_LEVEL` (Task 2)
  - `summarize` (Task 6), `TEXT_ONLY_PREFIX` and `pca_reduce` (Task 4)
  - `sha256_file` (existing, `supervision/artifacts.py`)
- Produces: the committed held-out draw that every later stage reads.
  - The committed-table test runs in CI without the QCEW files.
  - Its pinned hash fails on any redraw.

- [x] **Step 1: Write the committed-table test**

Create `tests/unit/test_committed_regressor_groups.py` with exactly this content:

```python
'''
The committed held-out groups: the regressor panel's real outer sets (Req 2; Req 4; Stage 3).

These read only ``conf/data/regressor_heldout_groups.csv`` and its provenance, so they run in CI
without the QCEW files. The pinned hash fails any accidental redraw: a redraw moves both outer
sets.
'''

import json
from collections import Counter
from pathlib import Path

import pytest

from naics_embedder.panels.regressor_splits import (
    SECTOR_LEVEL,
    ancestor_at,
    group_table_fingerprint,
    read_group_table,
)
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.config import RegressorPanelConfig, load_config

pytestmark = pytest.mark.unit

TABLE = Path('conf/data/regressor_heldout_groups.csv')
PROVENANCE = Path('conf/data/regressor_heldout_groups_provenance.json')
TABLE_SHA256 = 'deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4'

@pytest.fixture(scope='module')
def groups():
    return read_group_table(TABLE)

@pytest.fixture(scope='module')
def provenance():
    return json.loads(PROVENANCE.read_text())

def test_the_committed_table_is_the_recorded_draw(groups, provenance):
    assert sha256_file(TABLE) == TABLE_SHA256
    assert group_table_fingerprint(groups) == TABLE_SHA256
    assert provenance['heldout_groups'] == {
        'path': 'conf/data/regressor_heldout_groups.csv',
        'sha256': TABLE_SHA256,
        'groups': 60,
    }

def test_a_fifth_of_each_sectors_groups_is_held_out(groups, provenance):
    by_sector = Counter(ancestor_at(group, SECTOR_LEVEL) for group in groups)

    assert dict(sorted(by_sector.items())) == provenance['groups_by_sector']
    assert provenance['groups_by_sector'] == {
        '11': 4,
        '21': 1,
        '22': 1,
        '23': 2,
        '31': 17,
        '42': 4,
        '44': 5,
        '48': 6,
        '51': 2,
        '52': 2,
        '53': 2,
        '54': 2,
        '56': 2,
        '61': 1,
        '62': 3,
        '71': 2,
        '72': 1,
        '81': 3,
    }
    assert (provenance['seed'], provenance['fraction']) == (20260924, '1/5')
    assert provenance['six_digit_population'] == 980

def test_the_draw_used_the_configured_inputs(provenance):
    cfg = load_config(RegressorPanelConfig, 'data/regressor_panel.yaml')

    assert provenance['qcew_sha256'] == cfg.qcew_sha256
    assert provenance['codebook_codes_sha256'] == cfg.codebook_codes_sha256
    assert provenance['seed'] == cfg.seed

def test_the_partition_counts_are_recorded(provenance):
    assert provenance['rows_by_level_and_split'] == {
        '2': {
            'remainder': 2,
            'seen_outer': 1,
            'heldout_outer': 54
        },
        '3': {
            'remainder': 92,
            'seen_outer': 46,
            'heldout_outer': 126
        },
        '4': {
            'remainder': 480,
            'seen_outer': 240,
            'heldout_outer': 180
        },
        '5': {
            'remainder': 1046,
            'seen_outer': 523,
            'heldout_outer': 405
        },
        '6': {
            'remainder': 1554,
            'seen_outer': 777,
            'heldout_outer': 609
        },
    }
```

Run: `uv run pytest tests/unit/test_committed_regressor_groups.py -q`
Expected: 4 errors, each `FileNotFoundError` for `conf/data/regressor_heldout_groups.csv` or its
provenance. Step 2 writes both.

- [x] **Step 2: Draw the held-out groups**

Run: `pwd`
Expected: `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-5-regressor-panel`.

Run: `uv run naics-embedder data regressor-groups --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet`
Expected: exit 0 in a few seconds. The last line is
`Regressor held-out groups: conf/data/regressor_heldout_groups.csv`. The log lines above it, which
wrap at the terminal's width, report the 60 groups with their hash and the rows per level and
split.

- [x] **Step 3: Check the draw**

Run: `shasum -a 256 conf/data/regressor_heldout_groups.csv`
Expected: `deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4`. If it differs, stop
and ask. Do not commit a different table.

Run: `uv run pytest tests/unit/test_committed_regressor_groups.py -q`
Expected: `4 passed`. The tests check the hash, the 60 groups by sector, the seed and fraction,
the six-digit population of 980 and the rows per level and split against **Expected real-data
results**.

Run the command of Step 2 again.
Expected: exit 1, with a message that the table exists and is drawn once. The table's hash is
unchanged.

- [x] **Step 4: Build the text-only table**

Run: `uv run naics-embedder tools text-only-table --descriptions /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet --output /tmp/stage3-regressor-panel-635ffd43/text_only.parquet`
Expected: exit 0 in about half a minute, ending with
`Text-only table: /tmp/stage3-regressor-panel-635ffd43/text_only.parquet` and
`Provenance: /tmp/stage3-regressor-panel-635ffd43/text_only_provenance.json`.

Run: `uv run python -c "import json; p = json.load(open('/tmp/stage3-regressor-panel-635ffd43/text_only_provenance.json')); print(p['backbone'], p['revision'], p['codes'], p['hidden_size'], p['max_length'], p['descriptions']['sha256'], p['table_sha256'])"`
Expected, on one line:
`sentence-transformers/all-MiniLM-L6-v2 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 2125 384 512 5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d 6386f912ea4b37984ed9d4e4100f21fbdc29e91c5ee38acb05bd999342f5bbe0`.
The table's own hash (the last field) depends on the backbone's float arithmetic. On this Mac it
reproduces; anywhere else, a different value is not a stop condition.

- [x] **Step 5: Make the stub arm**

With the Write tool, create `/tmp/stage3-regressor-panel-635ffd43/stub_arm.py` with exactly this
content:

```python
'''
Plan 5's stub arm: the text-only table reduced by PCA to 16 dimensions, written in the export form.

It is not an arm. Its coordinates are the text-only comparator's own, so the embedding and
text-only comparators coincide by construction. It only exercises the panel on the real rows.

Usage: uv run python stub_arm.py TEXT_ONLY_PARQUET OUTPUT_PARQUET
'''

import sys

import polars as pl

from naics_embedder.panels.text_only import TEXT_ONLY_PREFIX, pca_reduce

DIMENSION = 16

table = pl.read_parquet(sys.argv[1])
columns = [name for name in table.columns if name.startswith(TEXT_ONLY_PREFIX)]
reduced = pca_reduce(table.select(columns).to_numpy(), DIMENSION)
schema = {f'e{index}': pl.Float64 for index in range(DIMENSION)}
coordinates = pl.DataFrame(reduced, schema=schema, orient='row')
table.select('code').hstack(coordinates).write_parquet(sys.argv[2])
```

Run: `uv run python /tmp/stage3-regressor-panel-635ffd43/stub_arm.py /tmp/stage3-regressor-panel-635ffd43/text_only.parquet /tmp/stage3-regressor-panel-635ffd43/stub_arm.parquet`
Expected: exit 0 with no output.

- [x] **Step 6: Score the stub arm on the validation split**

> Deviation: rerun after the whole-plan review, on 4102514 and on the reviewed tip 099fa12, each into a scratch log and output: both logged the seven reads of Step 7 and no opening, and reproduced Step 8's tables character for character.

Run: `pwd`
Expected: this worktree.

Run: `uv run naics-embedder tools regressor-panel --coordinates /tmp/stage3-regressor-panel-635ffd43/stub_arm.parquet --text-only /tmp/stage3-regressor-panel-635ffd43/text_only.parquet --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet --level 2 --level 3 --level 4 --level 5 --level 6 --purpose 'plan 5 smoke run: stub arm on the validation split' --log /tmp/stage3-regressor-panel-635ffd43/selection_log.jsonl --output /tmp/stage3-regressor-panel-635ffd43/validation.parquet`
Expected: exit 0 in about half a minute. The output wraps at the terminal's width. It lists
three undefined cells, `seen, level 2` (`1 remainder groups, fewer than 10`), `heldout, level 2`
and `heldout, level 3` (`no four-digit parent`). Under `Regressor panel: validation split` come
57 bullets, one per regime, level and comparator, and it ends with
`Predictions written to /tmp/stage3-regressor-panel-635ffd43/validation.parquet`.

- [x] **Step 7: Check the log: seven validation reads and no opening**

Run: `uv run python -c "import json; [print(r['event'], r['panel'], r['split'], r['n_queries'], r['detail']['level'], r['fingerprint'][:8]) for r in map(json.loads, open('/tmp/stage3-regressor-panel-635ffd43/selection_log.jsonl'))]"`
Expected: exactly these seven lines.

```text
read regressor_seen validation 92 3 deddfd4c
read regressor_seen validation 480 4 deddfd4c
read regressor_seen validation 1046 5 deddfd4c
read regressor_seen validation 1554 6 deddfd4c
read regressor_heldout validation 480 4 deddfd4c
read regressor_heldout validation 1046 5 deddfd4c
read regressor_heldout validation 1554 6 deddfd4c
```

- [x] **Step 8: Tabulate the results**

With the Write tool, create `/tmp/stage3-regressor-panel-635ffd43/summary.py` with exactly this
content:

```python
'''
Print the stub run's validation results as the finding's section 5 tables.

Each cell is RMSE / R² / median penalty, pooled over the five repeats.

Usage: uv run python summary.py VALIDATION_PARQUET
'''

import sys

import polars as pl

from naics_embedder.panels.regressor import summarize

summary = summarize(pl.read_parquet(sys.argv[1]))
for panel in ('regressor_seen', 'regressor_heldout'):
    rows = summary.filter(pl.col('panel') == panel)
    levels = sorted(set(rows.get_column('level')))
    print(f'| `{panel}` | ' + ' | '.join(f'Level {level}' for level in levels) + ' |')
    print('|---|' + '---|' * len(levels))
    for comparator in dict.fromkeys(rows.get_column('comparator')):
        cells = []
        for level in levels:
            cell = rows.filter((pl.col('level') == level) & (pl.col('comparator') == comparator))
            row = cell.row(0, named=True)
            cells.append(f"{row['rmse']:.4f} / {row['r2']:.4f} / {row['median_alpha']:g}")
        print(f'| {comparator} | ' + ' | '.join(cells) + ' |')
    print()
```

Run: `uv run python /tmp/stage3-regressor-panel-635ffd43/summary.py /tmp/stage3-regressor-panel-635ffd43/validation.parquet`
Expected: the two tables of the finding's section 5 (Step 9), character for character. Rows that
do not read the arm (covariates, one-hot and ancestors, each alone or with the covariates) must
match exactly. If an embedding or text-only row differs, check Step 4's table hash first.

- [x] **Step 9: Write the finding**

> Deviation: after the whole-plan review, 360af12 extended section 6's Stage 6 bullet: the panel also refuses constant columns, such as the zero time coordinate a log map at the origin keeps. The finding's numbers are unchanged.

Create `specs/findings/regressor-panel-splits.md` with exactly this content, with `<run date>`
replaced by Step 2's date:

````markdown
# Regressor panel and sealed outer sets: finding

**Status: FINAL (<run date>).** Roadmap Stage 3 (`specs/naics-embedding-roadmap.md`). This
finding records the real-data run of plan 5 (`specs/plans/completed/5-regressor-panel.md`):

- the panel's rows, on the population and grain Stage 1's finding verified
- the held-out four-digit groups, drawn once and committed, and the partition they fix
- the text-only comparator's table (D9)
- a stub arm's validation reads, which run the panel end to end

No outer set was opened: every read below is a validation read. Section 6 lists what later stages
read.

## Sources

The QCEW national slices were read from local copies in `~/Downloads/Data/QCEW/` under the
sha256 values of Stage 1's finding (Sources), and nothing was downloaded. The codebook and the
descriptions were read by absolute path from the main checkout, where bundle 18403d29 pins them.
The run used Python 3.12 with numpy 2.3.4, polars 1.35.1, scikit-learn 1.9.1, torch 2.9.1 and
transformers 4.57.1.

| File | sha256 | Used for |
|---|---|---|
| `2022_US000_annual.csv` | `c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18` | Feature year 2022 |
| `2023_US000_annual.csv` | `fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9` | Feature year 2023; outcome year 2023 |
| `2024_US000_annual.csv` | `48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46` | Feature year 2024 (the seen outer set); outcome year 2024 |
| `2025_US000_annual.csv` | `0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811` | Outcome year 2025 |
| Bundle 18403d29's `naics_codebook.parquet` | `5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b` | The 2,125 codes (codes fingerprint `9b646af1…`) |
| `data/naics_descriptions.parquet` | `5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d` | The text-only table's text: the bundle's `description_fingerprint` |

## 1. Rows

A cell is national (`US000`), private (`own_code` 5), all sizes and annual. It is usable when it
is disclosed with positive employment, establishments and wages. A code joins its level's
population when all four window years are usable, and each population code gives three rows,
feature years 2022–2024, each with the outcome dated one year later (D7).

| Level | Codebook codes | Population | Rows | Left out (no private national cell) |
|---|---|---|---|---|
| 2 | 20 | 19 | 57 | 92 |
| 3 | 96 | 88 | 264 | 921 to 928 |
| 4 | 308 | 300 | 900 | the eight NAICS 92 groups |
| 5 | 689 | 658 | 1,974 | 29 NAICS 92 codes, 11213 and 54112 |
| 6 | 1,012 | 980 | 2,940 | section 4's 32 codes of Stage 1's finding |

Every code left out lacks a private national cell; no code with one was lost to suppression or a
non-positive value in the window. At six digits the population is the branch record's 980, and
every load of the panel checks it: the record in `conf/data/regressor_panel.yaml` matches Stage 1's
decision block field by field (`tests/unit/test_regressor_branch_record.py`). BLS publishes 19
six-digit NAICS 238 codes (238110 to 238990) only as residential and nonresidential codes; each is
its five-digit parent's only child, so its cell is read from the parent's row.

## 2. The held-out draw

`data regressor-groups` held out 60 of the population's 300 four-digit groups: a fifth of each
sector's, by largest remainder, with seed 20260924. The floors give 51. The other 9 went to the
seven sectors with remainder 0.8 (11, 42, 44, 48, 54, 71 and 81) and to two of the three with
remainder 0.6 (22 and 53, by the seeded tie-break; 62 stayed at 3).

| Sector | Groups | Held out | Sector | Groups | Held out |
|---|---|---|---|---|---|
| 11 | 19 | 4 | 53 | 8 | 2 |
| 21 | 5 | 1 | 54 | 9 | 2 |
| 22 | 3 | 1 | 55 | 1 | 0 |
| 23 | 10 | 2 | 56 | 11 | 2 |
| 31 | 86 | 17 | 61 | 7 | 1 |
| 42 | 19 | 4 | 62 | 18 | 3 |
| 44 | 24 | 5 | 71 | 9 | 2 |
| 48 | 29 | 6 | 72 | 6 | 1 |
| 51 | 11 | 2 | 81 | 14 | 3 |
| 52 | 11 | 2 | Total | 300 | 60 |

The table, `conf/data/regressor_heldout_groups.csv` (306 bytes), has sha256
`deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4`. That hash is both regressor
panels' fingerprint, and `tests/unit/test_committed_regressor_groups.py` pins it.

## 3. The partition

The held-out outer set (H) holds every row of a code whose employment includes a held-out group.
The seen outer set (S) holds the other codes' 2024 rows, whose outcome is 2025. The remainder (R)
holds the other codes' 2022 and 2023 rows, and validation reads only R (user decision: disjoint).
At levels 2 and 3 a code joins H when any held-out group rolls into it (user decision: strict
seal).

| Level | R rows (codes) | S rows | H rows (codes) | R groups | Seen regime | Held-out regime |
|---|---|---|---|---|---|---|
| 2 | 2 (1) | 1 | 54 (18) | 1 | undefined: 1 group, fewer than 10 | undefined: no four-digit parent |
| 3 | 92 (46) | 46 | 126 (42) | 46 | defined | undefined: no four-digit parent |
| 4 | 480 (240) | 240 | 180 (60) | 240 | defined | defined |
| 5 | 1,046 (523) | 523 | 405 (135) | 240 | defined | defined |
| 6 | 1,554 (777) | 777 | 609 (203) | 240 | defined | defined |

Sector 55, with one group, is the only sector with no held-out group, so it is the level-2
remainder. A group is the four-digit parent at levels 4–6 and the code itself at levels 2 and 3.
D8's two regressor panels are the level-6 cells; levels 3–5 are the multi-level variant.

## 4. Fitting and the text-only table

- **Ridge.** Features are standardized on the fit rows, over 17 penalties from 0.001 to 100,000.
  The whole path comes from one singular value decomposition per fit.
- **Seen regime.** It fits R's 2022 rows and scores R's 2023 rows in 5 grouped folds, each fold's
  penalty chosen on the other folds, over 5 repeats. The test split, after an opening, tunes on
  that forward split and fits all of R.
- **Held-out regime.** It runs 5 repeats of 5 grouped folds over R, each with 5 grouped inner
  folds for the penalty. The test split tunes by 5 grouped folds over R.
- **Pairing.** Fold seeds depend on the regime, level, repeat and group ids alone, so every arm
  meets the same folds.
- **Text-only table (D9).** `sentence-transformers/all-MiniLM-L6-v2` at revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, frozen and on the CPU, embedded 2,125 codes at
  width 384 in about 33 seconds, with `max_length` 512. Each of the four channels is mean-pooled,
  and a code's vector is the mean of its present channels. The table's sha256 on this run was
  `6386f912…`. It is not pinned, because floats can differ across machines. The table stays
  outside the repo; its provenance records the backbone, revision, descriptions hash and library
  versions.

## 5. A stub arm on the validation split

The stub arm is the text-only table itself, reduced by PCA to 16 dimensions, so its `embedding`
comparator coincides with `text_only` by construction. The run checks the panel end to end on the
real rows; it measures no arm. One `tools regressor-panel` call scored levels 2–6 in both regimes
in about half a minute. It logged 7 validation reads: seen at levels 3–6 and held-out at 4–6,
each with `n_queries` equal to its level's remainder rows. It wrote 179,170 predictions.

Each cell is RMSE / R² / median penalty over the five repeats.

| `regressor_seen` | Level 3 | Level 4 | Level 5 | Level 6 |
|---|---|---|---|---|
| covariates | 0.4755 / 0.9354 / 0.001 | 0.4214 / 0.9286 / 0.001 | 0.4142 / 0.9263 / 0.001 | 0.3897 / 0.9330 / 0.001 |
| embedding | 1.2817 / 0.5308 / 0.001 | 1.2853 / 0.3358 / 0.001 | 1.2383 / 0.3413 / 0.001 | 1.2637 / 0.2959 / 0.001 |
| covariates+embedding | 0.2634 / 0.9802 / 0.001 | 0.2812 / 0.9682 / 0.001 | 0.2777 / 0.9669 / 0.001 | 0.2730 / 0.9671 / 0.001 |
| one_hot | 0.0322 / 0.9997 / 0.001 | 0.0623 / 0.9984 / 0.001 | 0.0636 / 0.9983 / 0.001 | 0.0601 / 0.9984 / 0.001 |
| covariates+one_hot | 0.0471 / 0.9994 / 0.001 | 0.0614 / 0.9985 / 0.001 | 0.0633 / 0.9983 / 0.001 | 0.0605 / 0.9984 / 0.001 |
| ancestors | 1.3971 / 0.4425 / 0.001 | 0.9004 / 0.6741 / 0.001 | 0.7805 / 0.7383 / 0.001 | 0.6059 / 0.8381 / 0.001 |
| covariates+ancestors | 0.2519 / 0.9819 / 0.001 | 0.1829 / 0.9866 / 0.32 | 0.1714 / 0.9874 / 0.32 | 0.1303 / 0.9925 / 1 |
| text_only | 1.2817 / 0.5308 / 0.001 | 1.2853 / 0.3358 / 0.001 | 1.2383 / 0.3413 / 0.001 | 1.2637 / 0.2959 / 0.001 |
| covariates+text_only | 0.2634 / 0.9802 / 0.001 | 0.2812 / 0.9682 / 0.001 | 0.2777 / 0.9669 / 0.001 | 0.2730 / 0.9671 / 0.001 |

| `regressor_heldout` | Level 4 | Level 5 | Level 6 |
|---|---|---|---|
| covariates | 0.4229 / 0.9277 / 0.32 | 0.4155 / 0.9255 / 0.32 | 0.3903 / 0.9325 / 0.32 |
| embedding | 1.4010 / 0.2064 / 100 | 1.3200 / 0.2480 / 100 | 1.3206 / 0.2274 / 100 |
| covariates+embedding | 0.3041 / 0.9626 / 1 | 0.2917 / 0.9633 / 0.001 | 0.2871 / 0.9635 / 0.32 |
| ancestors | 1.3702 / 0.2409 / 320 | 1.4033 / 0.1501 / 100 | 1.3602 / 0.1804 / 0.001 |
| covariates+ancestors | 0.2946 / 0.9649 / 1 | 0.3100 / 0.9585 / 0.001 | 0.3163 / 0.9557 / 1 |
| text_only | 1.4010 / 0.2064 / 100 | 1.3200 / 0.2480 / 100 | 1.3206 / 0.2274 / 100 |
| covariates+text_only | 0.3041 / 0.9626 / 1 | 0.2917 / 0.9633 / 0.001 | 0.2871 / 0.9635 / 0.32 |

What the run shows about the panel, not about any arm:

- **Covariates.** Alone they reach R² 0.93 at every level: size carries most of next-year log
  employment. The margin an arm can add lies above that.
- **One-hot in the seen regime.** It reaches R² 0.998 (RMSE 0.06 at six digits). With every
  code's earlier row in the fit set, code identity amounts to last year's level. This is where
  Req 2 expects one-hot to be a real competitor.
- **Seen-regime penalties.** Almost all are the grid's smallest, 0.001. On standardized features
  with hundreds of rows per fit, that is effectively least squares. The held-out regime picks
  larger ones: 0.32 for the covariates and 100 for the 16-dimensional stub.
- **Ancestor indicators.** They help in the seen regime (R² 0.84 alone at six digits) far more
  than in the held-out regime (0.18). A held-out group's own level-4 and level-5 indicators never
  appear in its fit set.

## 6. What later stages read

- **The command.** `naics-embedder tools regressor-panel --coordinates TABLE --text-only TABLE
  --codebook CODEBOOK` writes the per-row predictions with `--output`; in code,
  `RegressorPanel.validation(regime, level, arm, purpose)` returns them. The columns are
  `panel`, `split`, `level`, `comparator`, `repeat`, `fold` (−1 on the test split), `code`,
  `group`, `feature_year`, `outcome_year`, `alpha`, `outcome` and `prediction`.
- **Stage 4** computes each panel's statistic from these rows (Open questions). D8's two
  regressor panels are `regressor_seen` and `regressor_heldout` at level 6. It resamples by
  `group` within a panel and level: the four-digit parent at levels 4–6 (Req 5's unit), the code
  itself at levels 2–3. Every arm's validation rows share folds, so paired differences align row
  for row. An undefined cell (`RegressorPanel.cell_status`) has no rows.
- **Stages 7–11** read the validation split only, with `--log` pointing at the log their decision
  records keep. For these panels `n_queries` counts rows.
- **The text-only table** follows the arm's backbone and text (D9), so it is rebuilt whenever
  either changes: Stage 5 replaces the text this run embedded, and Stage 9 tries other
  backbones. The command is
  `tools text-only-table --descriptions ARM_DESCRIPTIONS --backbone ARM_BACKBONE --output TABLE`.
  The panel reduces the table to each arm's dimension, so Stage 8's dimensions need no rebuild.
- **Stage 6's export** writes tangent coordinates at the origin for a hyperbolic arm. The panel
  refuses Lorentz points.
- **Stage 12** opens each regime's outer set once, with `tools regressor-panel --split test
  --open-purpose …` or `RegressorPanel.open_outer`, under fingerprint `deddfd4c…`. The log's
  panel names are `regressor_seen` and `regressor_heldout`. A second opening needs
  `--reopen-reason` and is logged as `reopen`.

## Reproduction

The draw is deterministic, but it depends on numpy's generator streams, so the table is frozen
and committed rather than redrawn. To check that it reproduces:

1. Run `uv run naics-embedder data regressor-groups --force --codebook CODEBOOK` in a throwaway
   worktree at this commit.
2. Compare `shasum -a 256 conf/data/regressor_heldout_groups.csv` with the hash above.
3. Never commit a redraw.

The text-only table and the stub run reproduce with these commands, where `stub_arm.py` is the
script in plan 5's Task 9:

```bash
uv run naics-embedder tools text-only-table --descriptions DESCRIPTIONS --output /tmp/text_only.parquet
uv run python stub_arm.py /tmp/text_only.parquet /tmp/stub_arm.parquet
uv run naics-embedder tools regressor-panel --coordinates /tmp/stub_arm.parquet --text-only /tmp/text_only.parquet --codebook CODEBOOK --level 2 --level 3 --level 4 --level 5 --level 6 --log /tmp/selection_log.jsonl
```
````

- [x] **Step 10: Run the full suite and check the tree**

Run: `uv run pytest -n auto -q`
Expected: `1498 passed, 1 skipped`.

Run: `./scripts/format_code.sh --check tests/unit/test_committed_regressor_groups.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

Run: `git status --short conf specs/findings tests`
Expected: exactly these four lines. `logs/` is ignored, and nothing was written to `data/`.

```text
?? conf/data/regressor_heldout_groups.csv
?? conf/data/regressor_heldout_groups_provenance.json
?? specs/findings/regressor-panel-splits.md
?? tests/unit/test_committed_regressor_groups.py
```

- [x] **Step 11: Commit**

```bash
git add conf/data/regressor_heldout_groups.csv conf/data/regressor_heldout_groups_provenance.json tests/unit/test_committed_regressor_groups.py specs/findings/regressor-panel-splits.md
git commit -m "feat(data): commit the regressor panel's held-out groups and finding"
```

Keep `/tmp/stage3-regressor-panel-635ffd43/`: Final verification reads its log before removing
it.

### Task 10: Documentation

This task documents the three commands and the new modules: the CLI reference, an API page and its
navigation entry, and CLAUDE.md's tree and command list.

**Files:**

- Modify: `docs/usage.md`
- Create: `docs/api/regressor_panel.md`
- Modify: `docs/.nav.yml`
- Modify: `CLAUDE.md`

**Interfaces:**

- Consumes: the commands `data regressor-groups` (Task 7), `tools text-only-table` and
  `tools regressor-panel` (Task 8); the modules `panels.qcew_rows`, `panels.regressor_splits`,
  `panels.ridge`, `panels.text_only`, `panels.regressor` and `data.regressor_group_table`, whose
  docstrings the API page renders.
- Produces: documentation only.

- [x] **Step 1: Document the commands**

> Deviation: the whole-plan review edited `docs/usage.md` after this task: `--coordinates` names the tables the panel refuses (360af12), `--purpose` gives its default (1ce40e5), and an output row is one per row, comparator and repeat (4102514).

Modify `docs/usage.md` with these 2 edits, in order. Each replaced text occurs exactly once in the
file.

**`docs/usage.md`, edit 1 of 2.** Replace:

````markdown
uv run naics-embedder data roles --source-dir ~/Downloads/Data
```

---
````

with:

````markdown
uv run naics-embedder data roles --source-dir ~/Downloads/Data
```

### `data regressor-groups`

Draw the regressor panel's held-out four-digit groups, once: a fifth of each sector's groups
(largest remainder, seeded) from the 980 six-digit codes Stage 1's finding names. The QCEW
national slices are read from `qcew_dir` under the sha256 values pinned in
`conf/data/regressor_panel.yaml`, and the codebook under its codes' fingerprint. The table is
committed, and the panel reads it. A redraw moves both regressor outer sets, so an existing table
is replaced only with `--force`.

**Generates:** `conf/data/regressor_heldout_groups.csv`,
`conf/data/regressor_heldout_groups_provenance.json`

```bash
uv run naics-embedder data regressor-groups --codebook PATH/naics_codebook.parquet
```

---
````

**`docs/usage.md`, edit 2 of 2.** Replace:

````markdown
- `--output PATH` - Also write the summary as JSON

---
````

with:

````markdown
- `--output PATH` - Also write the summary as JSON

### `tools text-only-table`

Embed every code's text with the arm's backbone, frozen: each of the four channels is mean-pooled
over its tokens, and a code's vector is the mean of its present channels (roadmap D9). The
backbone comes from the local Hugging Face cache (default: `text_only.backbone` in
`conf/data/regressor_panel.yaml`). The regressor panel reduces the table to the arm's dimension
by PCA.

**Generates:** the table and `<stem>_provenance.json` beside it

```bash
uv run naics-embedder tools text-only-table --descriptions data/naics_descriptions.parquet \
  --output /tmp/text_only.parquet
```

**Options:**
- `--descriptions PATH` - The arm's descriptions parquet: the text it reads
- `--output PATH` - Where to write the table
- `--backbone NAME` - The arm's backbone (default: the regressor config's)

### `tools regressor-panel`

Score an arm on the regressor panel (roadmap Stage 3). Every comparator (covariates alone, and
the arm's coordinates, one-hot, ancestor indicators and the text-only table, each alone and with
the covariates) is fitted by ridge on standardized features, with the penalty tuned by nested
grouped folds inside the remainder. The outcome is log employment in year t + 1 from year-t
features. The seen and held-out regimes are separate panels. The validation split reads only the
remainder. The test split opens each regime's sealed outer set first, and both the opening and
the read are logged. The output holds one prediction per row, keyed by code, year and group.

```bash
uv run naics-embedder tools regressor-panel --coordinates arm.parquet \
  --text-only /tmp/text_only.parquet --codebook PATH/naics_codebook.parquet
```

**Options:**
- `--coordinates PATH` - The arm's 2,125-code table in the export form (tangent coordinates at
  the origin for a hyperbolic arm)
- `--text-only PATH` - The text-only table (`tools text-only-table`)
- `--codebook PATH` - A supervision bundle's `naics_codebook.parquet`
- `--regime seen|heldout` - Regime to score (repeatable; default: both)
- `--level INT` - NAICS level 2-6 (repeatable; default: 6)
- `--split validation|test` - The test split needs `--open-purpose`, which is logged
- `--purpose TEXT` - Why this read happens; recorded in the selection log
- `--open-purpose TEXT`, `--reopen-reason TEXT` - Why the outer sets are opened, and why again
- `--log PATH` - Selection log (default: `logs/selection_log.jsonl`)
- `--output PATH` - Write the per-row predictions as parquet

---
````

- [x] **Step 2: Add the API page**

Create `docs/api/regressor_panel.md` with exactly this content:

```markdown
# Regressor Panel API

Arms' coordinates as regressors for next-year QCEW employment, in two sealed regimes (roadmap
Stage 3).

## QCEW rows

::: naics_embedder.panels.qcew_rows

## Partition and held-out draw

::: naics_embedder.panels.regressor_splits

::: naics_embedder.data.regressor_group_table

## Fitting

::: naics_embedder.panels.ridge

## Text-only comparator

::: naics_embedder.panels.text_only

## Panel

::: naics_embedder.panels.regressor
```

Modify `docs/.nav.yml` with one edit. The replaced text occurs exactly once in the file.

**`docs/.nav.yml`, edit 1 of 1.** Replace:

```yaml
          - Outcome Panel: api/outcome_panel.md
          - QCEW: api/qcew_metrics.md
```

with:

```yaml
          - Outcome Panel: api/outcome_panel.md
          - Regressor Panel: api/regressor_panel.md
          - QCEW: api/qcew_metrics.md
```

- [x] **Step 3: Update CLAUDE.md**

Modify `CLAUDE.md` with these 5 edits, in order. Each replaced text occurs exactly once in the
file.

**`CLAUDE.md`, edit 1 of 5.** Replace:

```markdown
│   │   ├── index_role_table.py    # Draw the frozen index-entry role table (data roles)
│   │   ├── compute_relations.py   # Compute relationship measures
```

with:

```markdown
│   │   ├── index_role_table.py    # Draw the frozen index-entry role table (data roles)
│   │   ├── regressor_group_table.py  # Draw the regressor held-out groups (data regressor-groups)
│   │   ├── compute_relations.py   # Compute relationship measures
```

**`CLAUDE.md`, edit 2 of 5.** Replace:

```markdown
│   │   ├── outcome.py        # OutcomePanel: sealed validation and test query splits
│   │   └── lexical_encoder.py  # Training-free trigram stub encoder
│   ├── graph_model/          # Stage 4: HGCN refinement
```

with:

```markdown
│   │   ├── outcome.py        # OutcomePanel: sealed validation and test query splits
│   │   ├── lexical_encoder.py  # Training-free trigram stub encoder
│   │   ├── qcew_rows.py      # QCEW national rows: cells, population, dated rows (D7)
│   │   ├── regressor_splits.py  # The regressor partition and its committed held-out draw
│   │   ├── ridge.py          # Ridge on standardized features along a penalty grid
│   │   ├── text_only.py      # The text-only comparator: a frozen-backbone code table (D9)
│   │   └── regressor.py      # RegressorPanel: two regimes, sealed outer sets, predictions
│   ├── graph_model/          # Stage 4: HGCN refinement
```

**`CLAUDE.md`, edit 3 of 5.** Replace:

```markdown
│   │   ├── index_roles.csv        # The frozen index-entry role table (committed)
│   │   ├── relations.yaml
```

with:

```markdown
│   │   ├── index_roles.csv        # The frozen index-entry role table (committed)
│   │   ├── regressor_panel.yaml   # QCEW pins, held-out draw, ridge grid, folds, branch record
│   │   ├── regressor_heldout_groups.csv  # The regressor panel's held-out groups (committed)
│   │   ├── relations.yaml
```

**`CLAUDE.md`, edit 4 of 5.** Replace:

```markdown
# (data roles drew conf/data/index_roles.csv once; it is committed, and preprocess applies it)
```

with:

```markdown
# (data roles drew conf/data/index_roles.csv once; it is committed, and preprocess applies it)
# (data regressor-groups drew conf/data/regressor_heldout_groups.csv once; it is committed)
```

**`CLAUDE.md`, edit 5 of 5.** Replace:

```markdown
uv run naics-embedder tools outcome-baseline  # Lexical stub on the outcome validation split
```

with:

```markdown
uv run naics-embedder tools outcome-baseline  # Lexical stub on the outcome validation split
uv run naics-embedder tools text-only-table  # Frozen-backbone text table for the regressor panel
uv run naics-embedder tools regressor-panel  # Score an arm on the regressor panel
```

- [x] **Step 4: Build the docs strictly**

Run: `uv run mkdocs build --strict --site-dir /tmp/stage3-regressor-panel-docs-635ffd43`
Expected: exit 0 and `Documentation built in`. The output has no `WARNING` line.

Run: `grep -c generate_regressor_group_table /tmp/stage3-regressor-panel-docs-635ffd43/api/regressor_panel/index.html`
Expected: a count of at least 1. It shows the new page rendered its modules.

Run: `rm -rf /tmp/stage3-regressor-panel-docs-635ffd43`

- [x] **Step 5: Commit**

```bash
git add docs/usage.md docs/api/regressor_panel.md docs/.nav.yml CLAUDE.md
git commit -m "docs: document the regressor panel commands and API"
```

## Final verification (controller, inline)

- [x] **Step 1: Full suite on Python 3.12**

> Deviation: 1533 passed, 1 skipped: the nine review commits, a50f85b through 099fa12, added 35 test cases (1498 before them).

Run: `uv run pytest -n auto -q`
Expected: `1498 passed, 1 skipped`.

- [x] **Step 2: Full suite on Python 3.10, CI's other leg**

> Deviation: likewise 1533 passed, 1 skipped on Python 3.10 after the review commits (1498 before them).

Run: `UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310-635ffd43 uv run pytest -n auto -q`
Expected: `1498 passed, 1 skipped`, the same as Step 1. This leg locks numpy 2.2.6 and
scikit-learn 1.7.2. The committed-table test reads the CSV, so it passes without redrawing.

Run: `rm -rf /tmp/naics-py310-635ffd43`

- [x] **Step 3: The CI lint job**

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 4: The branch carries only this plan's commits**

Run: `git log --oneline origin/main..HEAD`
Expected, read bottom up, because `git log` prints the newest commit first:

- the plan's commit, at the bottom
- then each task's commit in task order, Task 1's through Task 10's (the pre-flight commits
  nothing)
- review fixes may add commits between them

Neither "config" nor "graph config" appears.

Run: `git diff --name-only origin/main...HEAD`
Expected: exactly these 30 paths. The three dots diff from the merge base, so a commit that lands
on `origin/main` meanwhile does not show up.

```text
CLAUDE.md
conf/data/regressor_heldout_groups.csv
conf/data/regressor_heldout_groups_provenance.json
conf/data/regressor_panel.yaml
docs/.nav.yml
docs/api/regressor_panel.md
docs/usage.md
specs/findings/regressor-panel-splits.md
specs/plans/5-regressor-panel.md
src/naics_embedder/cli/commands/data.py
src/naics_embedder/cli/commands/tools.py
src/naics_embedder/data/regressor_group_table.py
src/naics_embedder/panels/qcew_rows.py
src/naics_embedder/panels/regressor.py
src/naics_embedder/panels/regressor_splits.py
src/naics_embedder/panels/ridge.py
src/naics_embedder/panels/text_only.py
src/naics_embedder/utils/config.py
tests/conftest.py
tests/fixtures/regressor_panel.py
tests/unit/test_cli_commands.py
tests/unit/test_committed_regressor_groups.py
tests/unit/test_config.py
tests/unit/test_regressor_branch_record.py
tests/unit/test_regressor_group_table.py
tests/unit/test_regressor_panel.py
tests/unit/test_regressor_qcew_rows.py
tests/unit/test_regressor_ridge.py
tests/unit/test_regressor_splits.py
tests/unit/test_text_only.py
```

- [x] **Step 5: The roadmap's Stage 3 Exit, outcome by outcome**

> Deviation: 150 passed: the review commits added 35 test cases to these files (115 before them). The leakage tests `test_regressor_panel.py::test_no_outcome_outside_the_remainder_reaches_a_test_prediction` and `::test_no_scored_outcome_reaches_its_own_validation_prediction` (f84e00b) also back the second row.

Check each row against the tests that back it. Test names are given as `file::test`, and a
`::test` alone continues the file before it.

| Exit outcome | Tests |
|---|---|
| The panel reports the two regimes separately, with one-hot only in the seen regime and every Req 2 comparator scored in each | `test_regressor_panel.py::test_one_hot_runs_only_in_the_seen_regime_and_ancestors_only_above_the_sector`, `::test_each_regime_is_its_own_panel_and_scores_every_comparator` and `::test_each_comparator_has_its_own_columns`; Task 9 Steps 6 and 8 on the real validation split |
| The penalty is tuned inside the remainder and the outer set is read once | `test_regressor_panel.py::test_no_scored_row_tunes_its_own_penalty`, `::test_the_outer_set_is_scored_once_by_a_penalty_tuned_inside_the_remainder`, `::test_the_test_split_scores_each_outer_row_once_per_comparator` and `::test_validation_reads_only_the_remainder_and_logs_each_read`; `test_regressor_splits.py::test_the_partition_is_disjoint_and_covers_every_row` |
| The panel reads the committed outer groups, never a fresh draw | `test_regressor_panel.py::test_the_panel_reads_the_committed_groups_and_never_draws`; `test_committed_regressor_groups.py::test_the_committed_table_is_the_recorded_draw`; `test_regressor_group_table.py::test_an_existing_table_is_redrawn_only_with_force`; `test_regressor_splits.py::test_the_fingerprint_is_the_committed_files_sha256` |
| Neither regime's outer set can be read without a logged opening | `test_regressor_panel.py::test_neither_outer_set_is_read_without_a_logged_opening`, `::test_every_panel_object_opens_for_itself_and_a_reopening_needs_a_reason` and `::test_openings_are_counted_per_held_out_draw`; `test_cli_commands.py::test_regressor_panel_needs_an_open_purpose_for_the_test_split` and `::test_regressor_panel_opens_each_regime_once_before_its_test_read` |
| Every row's features are dated before its outcome (D7) | `test_regressor_qcew_rows.py::test_the_window_gives_three_feature_years_before_their_outcome_years` and `::test_every_rows_features_are_dated_before_its_outcome`; `test_regressor_panel.py::test_every_rows_features_are_dated_before_its_outcome` and `::test_the_seen_regime_fits_no_outcome_after_a_scored_rows_feature_year` |
| The branch record matches the finding's decision block | `test_regressor_branch_record.py` (all four tests); `test_regressor_panel.py::test_from_sources_refuses_data_the_branch_record_does_not_name` and `::test_the_branch_record_must_name_this_panels_data`; `test_regressor_group_table.py::test_the_draw_needs_the_branch_records_population` |

Two Produces items outside the Exit are backed too:

- Req 5's pairing: `test_regressor_panel.py::test_folds_do_not_depend_on_the_arm`.
- The multi-level variant:
  `test_regressor_panel.py::test_the_multi_level_variant_scores_level_codes_outside_the_held_out_groups`
  and `::test_a_cell_without_enough_remainder_groups_is_undefined_not_scored`.

Run: `uv run pytest tests/unit/test_regressor_panel.py tests/unit/test_regressor_qcew_rows.py tests/unit/test_regressor_splits.py tests/unit/test_regressor_branch_record.py tests/unit/test_committed_regressor_groups.py tests/unit/test_regressor_group_table.py tests/unit/test_cli_commands.py -q`
Expected: `115 passed`.

- [x] **Step 6: No outer set was opened**

Run: `uv run python -c "import glob, json; paths = sorted(glob.glob('/tmp/stage3-regressor-panel-635ffd43/*.jsonl') + glob.glob('logs/*.jsonl')); records = [json.loads(line) for path in paths for line in open(path) if line.strip()]; print(paths, len(records), sum(r['event'] != 'read' for r in records))"`
Expected: `['/tmp/stage3-regressor-panel-635ffd43/selection_log.jsonl'] 7 0`. The only selection log
this plan wrote holds Task 9's seven validation reads and no opening. Tests write their logs
under pytest's temporary directories, and `logs/` holds only `.log` files. If the output differs,
stop and ask: an `open` or `reopen` record for `regressor_seen` or `regressor_heldout` would break
Stage 12's one-opening rule.

- [x] **Step 7: Remove the scratch directory**

Run: `rm -rf /tmp/stage3-regressor-panel-635ffd43`

Nothing else in `/tmp` is this plan's: Task 10 and Step 2 removed their own directories.

## Plan completion

Run the Plan Completion Protocol of writing-plans after the final review. The completion commits
are the branch's last commits. Before editing `specs/naics-embedding-roadmap.md` or
`specs/deferred_items.md`, check whether another Claude session is active in this repository. If
one is, hold both edits and hand your human partner the exact text below.

- [x] **Step 1: Tick the roadmap stage and add the rollout note and the stamp**

In `specs/naics-embedding-roadmap.md`, make one edit. Replace:

```markdown
- [ ] Stage 3: Regressor panel
```

with:

```markdown
- [x] Stage 3: Regressor panel
```

Make a second edit. Replace the Stage 3 entry's last lines:

```markdown
      before its outcome (D7); the branch record matches the finding's decision block.
      ROUTING: writing-plans

- [ ] Stage 4: Decision rule and diagnostics
```

with the lines below, with `YYYY-MM-DD` replaced by the completion date:

```markdown
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
      Stage 3: COMPLETE (YYYY-MM-DD) — implemented by plan 5
      (specs/plans/completed/5-regressor-panel.md). Next: resume the roadmap.

- [ ] Stage 4: Decision rule and diagnostics
```

- [x] **Step 2: Re-validate the later stages against what shipped**

Three later entries consume what Stage 3 shipped in ways their text does not yet say, so each
gets one edit.

In the Stage 4 entry, replace:

```markdown
      and Stage 3's per-row predictions, resampled by four-digit group in each regressor regime.
      No trained arms yet: the tooling is exercised on synthetic scores.
```

with:

```markdown
      and Stage 3's per-row predictions, resampled by four-digit group in each regressor regime
      (finding `specs/findings/regressor-panel-splits.md`, section 6): D8's two regressor panels
      are the level-6 cells, a validation read scores each of its rows once per repeat (five),
      and `group` is the code itself at levels 2–3. No trained arms yet: the tooling is
      exercised on synthetic scores.
```

In the Stage 7 entry, replace:

```markdown
      The test split stays sealed: Stage 12 opens it, as the finding's section 6 erratum says.
      Stage 3's panel; Stage 4's seed-sweep driver, decision tooling and δ procedure.
```

with:

```markdown
      The test split stays sealed: Stage 12 opens it, as the finding's section 6 erratum says.
      Stage 3's panel, on its validation split only; its text-only table is rebuilt from the
      arm's own descriptions with `tools text-only-table`, because the table Stage 3 built
      embeds bundle 18403d29's text, which Stage 5 replaces (D9). Stage 4's seed-sweep driver,
      decision tooling and δ procedure.
```

In the Stage 12 entry, replace:

```markdown
      `OutcomePanel.open_test`; Stage 3's sealed outer sets and the logged opening that guards
      them; Stage 4's tooling and seed-sweep driver.
```

with:

```markdown
      `OutcomePanel.open_test`; Stage 3's sealed outer sets and the logged opening that guards
      them (`RegressorPanel.open_outer`, or `tools regressor-panel --split test` with
      `--open-purpose`): one opening per regime covers every level the panel has loaded, and
      the log names the panels `regressor_seen` and `regressor_heldout`, both under the
      fingerprint `deddfd4c…`; Stage 4's tooling and seed-sweep driver.
```

Stages 6, 8, 9, 10 and 11 need no edit:

- Stage 6's export writes Req 2's form, which is what the panel reads; the panel refuses Lorentz
  points.
- Stage 8's dimensions need no new text-only table: the panel reduces it to each arm's dimension
  by PCA.
- Stage 9 already says the text-only comparator follows each arm's backbone, and
  `tools text-only-table --backbone` builds it.
- Stages 10 and 11 read the panel through Stage 4's tooling.

Commit the roadmap edits with the plan markup in Step 3's commit.

- [x] **Step 3: Mark up this plan and resolve the gate**

Follow the protocol:

- Run the resolve-before-defer gate.
- Tick every completed step and add `> Deviation:` notes.
- Add the status header.
- Tick any earlier deferred item this plan implemented. None is expected: plan 3's rounding fix
  stays open, because this plan does not reuse the coverage script's checks.
- Append this plan's deferred items, if any.
- Run `uv run --no-project --python 3.13 python ~/.claude/skills/writing-plans/scripts/deferred_stats.py`
  and surface its summary line.

Then commit:

```bash
git add specs/naics-embedding-roadmap.md specs/plans/5-regressor-panel.md specs/deferred_items.md
git commit -m "docs(roadmap): complete Stage 3 and re-validate Stages 4, 7 and 12"
```

`git add` of an unchanged `specs/deferred_items.md` is harmless.

- [x] **Step 4: Retire the plan**

```bash
git mv specs/plans/5-regressor-panel.md specs/plans/completed/5-regressor-panel.md
git commit -m "chore(specs): retire plan 5"
```

This plan has no relative links to re-point, and no spec file retires with it: Stage 3 has no
stage spec.

- [x] **Step 5: Integrate**

Hand over to finishing-a-development-branch. Before opening any PR, check two things:

- `git log --oneline origin/main..HEAD` shows only this branch's commits.
- `git diff --name-only origin/main...HEAD -- conf/config.yaml conf/graph.yaml` prints nothing.

Never push to `main`. A PR merges only after CI passes (`lint`, `test (3.10)`, `test (3.12)`) and
Codex's review has given its 👍, or after its inline findings are fixed.
