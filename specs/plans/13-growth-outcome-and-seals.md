# Growth Outcome and One Seal per Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-representation-split-roadmap.md, Stage 1 — on plan completion, tick the
> stage and re-validate later stages against what shipped.

**Goal:** Build Stage 1 of the representation-split roadmap. The regressor panel gains an `outcome`
setting, `level` by default, and a growth outcome, log E(t+1) − log E(t), on its committed rows,
partition and folds, read as `growth_seen` and `growth_heldout` (Req 21). An opening of a regime's
outer set counts for both outcomes (Req 22). Level mode's rows, data fingerprints, folds and scores
are pinned on the fixture panel before anything changes.

**Architecture:**

- **The pins (Task 1).** Before any source changes, a commit of tests alone pins level mode as
  d61f49b computes it on the fixture panel, in three layers (P1):
  - an oracle copy of d61f49b's row builder, exact on every platform;
  - float-free digests and tolerance sums, exact on every platform;
  - literal data fingerprints and prediction digests, exact where they were captured, on macOS
    arm64.
- **Growth rows (Task 2).** `panel_rows` takes a `RegressorOutcome`. The level path is untouched.
  Growth subtracts log E(t), read from the feature year's own cell by the same expression, after
  the rows are built and counted (P2).
- **Growth reads (Task 3).** A `RegressorPanel` carries one outcome, which names its reads through
  `READ_NAMES`. `PANEL_NAMES` stays the levels' names, which D8's decision reads.
  `RegressorPanelConfig.outcome` and `conf/data/regressor_panel.yaml` carry the setting, and
  `load_regressor_panel` passes it on (P3, P4).
- **One seal per split (Task 4).** `seal_names(regime)` names a regime's split under both
  outcomes, and `SelectionLog.openings` takes a collection of names. `require_openable` and
  `open_outer` count openings under all of them, under the unchanged draw fingerprint (P5).
- **Level mode only from the CLI and the sweep (Task 5).** `tools regressor-panel` and
  `run_seed_sweep` refuse a growth panel before any read. Both score an arm on every read, and
  Req 24 forbids a candidate's growth error before the economic margin is recorded (P6).
- **Documentation (Task 6).** The API page, CLAUDE.md, the README and the test README.

Nothing reads real data. Stage 2 builds the reference-only read on this panel, and Stage 3 runs
it on real data.

**Tech Stack:**

- Python 3.10 and 3.12 (CI runs both; `.python-version` pins 3.12).
- polars 1.35.1, numpy (2.3.4 under 3.12, 2.2.6 under 3.10) and pydantic 2.12.4; typer and rich.
- pytest with xdist; ruff and yapf; mkdocs with mkdocstrings (strict build).
- No model, backbone or GPU: the fixture panel is synthetic.

**Spec:** `specs/naics-representation-split.md` (last changed 0a39042), Reqs 20–22. This plan
implements Stage 1 of its roadmap, `specs/naics-representation-split-roadmap.md` (last changed
dc7b9ea). Retirement leaves the spec in `specs/`: Stages 2–3 and the candidates' and text→code
specs cite it.

## Global Constraints

Every task's requirements include this section.

**Locators.** Bare D1–D11 are the embedding roadmap's decisions
(`specs/naics-embedding-roadmap.md`), as the codebase's comments and this plan's code cite them:
D1 the covariates, D7 the next-year outcome, D8 the three-panel decision, D9 the text-only table.
U1–U16 are the split spec's user rulings, and P1–P7 are this plan's decisions.

### The spec (`specs/naics-representation-split.md` at 0a39042), verbatim

Req 20:

> **Req 20 — Uses, and the first panel.** The economic artifact's uses, ranked by importance (U2,
> with use 2 as corrected by U15), are:
>
> 1. employment growth;
> 2. joint non-response + employment growth;
> 3. wages;
> 4. shock exposure;
> 5. nowcasting.
>
> How they are handled for now:
>
> - The first panel, and for now the only one, is employment growth (U6). Each later use joins
>   through its own panel spec, once its data are verified.
> - Joint non-response + employment growth has no panel. Asked for its data source under its earlier
>   name, joint non-response + wage growth, the user answered "No data exists..." (U5).
> - Data stay on the NAICS 2022 basis (U6). Cross-vintage history stays Out of scope, as in
>   naics-embedding.

Req 21:

> **Req 21 — The growth panel retargets the regressor panel (U10, U12).** The regressor panel gains
> one setting, `outcome`, either `level` or `growth`. `level` is the default and is today's panel.
>
> - **Rows.** The committed ones (finding: regressor-panel-splits §1).
>   - These are national, private, annual QCEW cells: 980 six-digit codes, each with feature years
>     2022–2024, giving 2,940 rows. Levels 2–5 are kept for the multi-level variant.
>   - They come from the same sha256-pinned files, and nothing new is downloaded.
> - **Outcome.** log E(t+1) − log E(t), where E is annual-average employment. Every code in the
>   population has positive employment in all four window years, so the difference is always
>   defined.
> - **Covariates.** roadmap D1, unchanged: log establishments and log total annual wages in year t
>   (U12).
> - **Unchanged:**
>   - the partition and the committed held-out draw (60 of 300 four-digit groups);
>   - both regimes and their folds;
>   - the comparators (naics-embedding Req 2, with roadmap D9's text-only table);
>   - ridge fitting with a nested-cross-validated penalty, and decision level 6.
> - **The regimes, as they apply to growth.**
>   - Seen regime: fit on the remainder's 2022→2023 growth and score its 2023→2024 growth. Its outer
>     set is the 2024→2025 rows.
>   - Held-out regime: repeated grouped folds with nested tuning inside the remainder. Its outer set
>     is every row of the held-out groups' codes.
> - **Read names.** Growth reads are logged as `growth_seen` and `growth_heldout`. That keeps them
>   apart from the levels history, `regressor_seen` and `regressor_heldout`.
> - **Levels unchanged.** With `outcome: level`, the rows, data fingerprints, folds and scores are
>   bit-identical to today's, pinned on the fixture panel before any change. Real Stage 7–8 records
>   keep verifying through their stored hashes; nothing re-reads real data to check them.

Req 22:

> **Req 22 — One seal per split.** The outer sets are rows, and 2025 employment underlies both
> outcomes' outer rows. An opening is therefore recorded against the split, not the outcome.
>
> - **The key** is the regime plus `RegressorPanel.fingerprint`. That fingerprint names the held-out
>   draw alone (`src/naics_embedder/panels/regressor.py:531`, `:594`). The row partition follows
>   from the draw and the committed rows, which the data fingerprint pins.
> - **The check.** `require_openable` consults both outcomes' read names under that key.
>   - An opening of `regressor_seen` or `growth_seen` counts as an opening of the seen regime for
>     both outcomes; likewise for the held-out regime.
>   - A second opening needs a stated reason, as today.
> - **Existing records.** Level mode's fingerprint inputs are unchanged, so existing selection-log
>   records still match.
> - **The levels regimes' outer sets** stay unopened (U8).

Supersessions, D7's row:

> | Superseded | Replaced by | Applies to |
> |---|---|---|
> | Roadmap D7, next-year log employment | Log growth from t to t+1 (Req 21) | Economic; `level` stays the default mode |

Verification, this stage's items:

> - [ ] **Levels unchanged.** On the fixture panel, level mode's rows, data fingerprints, folds and
>       scores equal the values pinned before the change.
> - [ ] **Growth rows.** On the fixture panel:
>   - each growth row's outcome equals log E(t+1) − log E(t) from the same cells;
>   - its features are dated year t;
>   - the partition and the held-out draw's fingerprint equal level mode's.
> - [ ] **One seal per split.** After a regime is opened under one outcome, opening it under the
>       other is refused without a stated reason. With a reason, the log records a reopening.
> - [ ] **Checks.** The full suite, Ruff, the read-only YAPF check and the strict docs build pass.

Rollout note, item 2:

> 2. **The growth outcome and the shared seal (Reqs 21–22),** tested on fixtures, with the levels
>    values pinned first.

### The roadmap (`specs/naics-representation-split-roadmap.md` at dc7b9ea), verbatim

Stage 1:

> - [ ] Stage 1: Growth outcome and one seal per split
>       Objective: Give the regressor panel a growth outcome on the committed rows, read under its
>       own names, and make an opening of a regime's split count for both outcomes, with level
>       mode pinned before any change.
>       Spec: Req 21; Req 22; Req 20 (the first panel); Verification "Levels unchanged", "Growth
>       rows", "One seal per split", "Checks"; Rollout note, item 2; Supersessions (D7); D1.
>       Gap closed: Req 21; Req 22; Req 20.
>       Consumes: Nothing from a prior stage. The regressor panel as embedding roadmap Stage 3 built
>       it (`panels/regressor.py`, `panels/qcew_rows.py`, `panels/regressor_splits.py`; finding:
>       regressor-panel-splits §§1, 3 and 6), with its committed draw (`deddfd4c…`) and fixture
>       panel (`tests/fixtures/regressor_panel.py`); `SelectionLog.openings`, which keys an opening
>       on its read name and fingerprint (`panels/selection_log.py:84-91`). A config key is declared
>       in `RegressorPanelConfig` (`utils/config.py`), as every stage's is.
>       Produces: The `outcome` setting, `level` by default; growth reads that return today's
>       per-row prediction columns for every comparator, logged as `growth_seen` and
>       `growth_heldout` under the draw's fingerprint; the growth rows' data fingerprint;
>       `require_openable` counting a regime's openings under both outcomes' read names; level
>       mode's values pinned by test.
>       Exit: On the fixture panel, level mode's rows, data fingerprints, folds and scores equal the
>       values pinned before the change (test); each growth row's outcome is log E(t+1) − log E(t)
>       from the same cells, its features are dated year t, and its partition and draw fingerprint
>       equal level mode's (test); after a regime opens under one outcome, opening it under the
>       other is refused without a reason and logged as a reopening with one (test); Verification
>       "Checks" passes.
>       ROUTING: writing-plans

### The embedding roadmap's D1 and D7 (`specs/naics-embedding-roadmap.md` at d61f49b), verbatim

> - **D1 — Regressor panel's inherited definitions (Req 2; Stage 3).** The spec names neither the
>   downstream model nor the covariates. Decision: ridge with a nested-cross-validated penalty;
>   covariates are log establishment counts and log wages from the same QCEW rows, as in the
>   methodology's definition (`metrics/qcew.py`).

> - **D7 — The time-respecting outcome (Req 2; Stage 3).** The finding takes branch A, so the panel
>   includes a time-respecting outcome; neither the spec nor D1 names its variable or says whether
>   the same-year outcome stays. Decision: log annual-average employment in year t+1, from year-t
>   features (the representation, plus D1's covariates from the year-t row); the same-year outcome
>   is dropped. With 2022–2025 final, feature years run 2022–2024, and splits by time seal
>   2024→2025 for the seen regime. That leaves the seen regime's repeated grouped inner folds two
>   feature years; Stage 3's plan fits them to that.

Req 21 supersedes D7 for the economic artifact, and level mode keeps it as the default.

### Deferred items this plan touches

None. `specs/deferred_items.md` was read at d61f49b and is not edited. Its one open regressor-panel
item, the ridge path's SVD cache, stays standalone (split roadmap, Deferred items). Plan
completion still runs the ticking pass.

### This plan's decisions

Each goes beyond a spec line, or reads one. Do not reopen them during execution. Task numbers name
where each lands.

- **P1. Level mode is pinned in three layers, before any change (Task 1).**
  - **The d61f49b oracle.** `panel_rows_d61f49b`, in the fixture module, is `panel_rows` as
    d61f49b wrote it. Level mode's rows equal its output bit for bit at every level
    (`DataFrame.equals`). Both sides take their logs from the platform's own libm, so the
    equality holds on every platform.
  - **Float-free pins, exact on every platform.** They cover:
    - the draw's fingerprint;
    - each level's split counts and partition digest;
    - the data fingerprint's recipe;
    - the selection-log records of a fixed sequence of reads;
    - each read's digest of its key columns and chosen penalties;
    - level 6's folds.

    Each read's prediction and squared-error sums are also pinned to a relative 1e-9.
  - **Literal pins, exact where captured.** `CAPTURED_HERE` (macOS arm64) marks two tests: each
    level's data fingerprint, and each read's digest over every prediction column. Linux CI
    skips both.
  - **Why the layers.** A literal float pin is a property of a platform's libm and BLAS, not only
    of the code. Seven of level 6's cell values lie within glibc's 0.519-ULP error bound of a
    rounding midpoint, the closest at 0.0019 ULP, so Linux may round a log a last bit
    differently. The oracle carries "Levels unchanged" bit for bit on every platform.
- **P2. `RegressorOutcome`, and growth after the join (Task 2).**
  - `RegressorOutcome(str, Enum)`, with `level` and `growth`, lives in `panels/qcew_rows.py`
    beside `ROW_COLUMNS`, because it names what a row's `outcome` holds.
  - `panel_rows(cells, codes, outcome=RegressorOutcome.LEVEL)` leaves the level path untouched.
    Its growth branch runs after the height check. It joins log E(t), computed by the level
    outcome's own expression from the feature year's cell, and subtracts it.
  - So a growth row differs from its level row in `outcome` alone. From 2023 on, log E(t) is the
    level outcome of the code's row a year earlier, bit for bit.
- **P3. One outcome per panel object, and read names per outcome (Task 3).**
  - `RegressorPanel(..., *, outcome=RegressorOutcome.LEVEL)` is keyword-only.
  - `READ_NAMES[outcome][regime]` names every read.
    `PANEL_NAMES = READ_NAMES[RegressorOutcome.LEVEL]` keeps D8's decision
    (`decision/scores.py`) on the levels history.
  - The partition, the draw fingerprint, the folds, the comparators and the fit settings are
    unchanged. A growth read scores the same rows, keys and folds as the level read. The data
    fingerprint hashes the rows, outcome included, so it names the outcome.
  - Not chosen: a separate growth panel class, or an outcome argument on each read. Req 21 retargets
    the panel through one setting (U10).
- **P4. The config key (Task 3).** `RegressorPanelConfig.outcome: Literal['level', 'growth']`
  defaults to `level`, and `conf/data/regressor_panel.yaml` gains `outcome: level`.
  `TestRegressorPanelConfig` requires that YAML to equal the defaults, but for `branch_record`.
  `load_regressor_panel` passes the setting on.
- **P5. The seal (Task 4).**
  - `seal_names(regime)` returns the regime's read name under each outcome.
  - `SelectionLog.openings(panel, fingerprint)` accepts one name or a collection of names.
  - `require_openable` and `open_outer` count the openings under `seal_names(regime)` and
    `self.fingerprint`. The key stays the regime plus the draw fingerprint, whose inputs are
    unchanged (Req 22), so existing records still match.
  - The refusal names the read name the split was first opened under.
  - `OutcomePanel`'s call, `openings(OUTCOME_PANEL, …)`, is unchanged.
- **P6. Level mode only from `tools regressor-panel` and `tools sweep` (Task 5).** The brief left
  this to the plan.
  - `tools regressor-panel` refuses a config whose `outcome` is `growth`, before it loads
    anything. `run_seed_sweep` refuses a growth `RegressorPanel` before any read, run or store
    write.
  - **Why.** Both score `embedding` and `text_only` on every read. A growth read through either
    would create a candidate's growth error before the economic margin is recorded, which Req 24
    forbids. And D8's decision reads the levels names only. Without the guard, the sweep logs
    the arm's growth reads and only then fails, as measured at plan time: "1,196 predictions are
    not level-6 validation rows of ['regressor_seen', 'regressor_heldout']".
  - **Not chosen: a `--outcome` flag** on `tools regressor-panel`. It would be the first route to
    a candidate's growth error before Stage 3's margin. Stage 2 builds the reference-only read
    and decides how growth reads reach the CLI.
  - **Other readers of the config.** `data regressor-groups` reads its draw settings, and
    `tools text-only-table` its `text_only` settings. Neither builds a panel, so neither needs
    the guard.
  - **The Python API stays open:** `RegressorPanel`, `from_sources` and `load_regressor_panel`.
    Stage 2's reference-only read builds on it.
- **P7. Documentation and counts (Tasks 5 and 6).**
  - Task 5 edits `docs/usage.md` beside the refusals it describes.
  - Task 6 edits `docs/api/regressor_panel.md`, CLAUDE.md, the README and `tests/README.md`.
  - The stated test count becomes 3,119, as collected after Task 5.

### Recorded deviations

- **No red step in Task 1.** The pins describe d61f49b's behavior, so they pass when written.
  Task 1's Step 3 shows they can fail: two mutations of the source each fail named pins, and
  each is reverted.
- **Literal pins on macOS arm64 only (P1).** Two of Task 1's five tests skip off the Mac. The
  spec's bit-for-bit "Levels unchanged" is carried on every platform by the oracle.
- **Refusals beyond the roadmap's Produces (P6).** Stage 1's Produces names no CLI or sweep
  change. The refusals keep Req 24's order until Stage 2's reference-only read exists.
- **U-numbers in `## Decisions`.** writing-plans numbers decisions D1, D2, … . Here the entries
  carry the spec's U-numbers, because bare D-numbers are the embedding roadmap's (Locators).
- **The stated test count.** CLAUDE.md, the README and `tests/README.md` say 3,099 tests;
  d61f49b collects 3,100. Task 6 writes 3,119, the count measured after Task 5.
- **The file name.** This plan is named for its stage, as plan 12 was, not for its spec. The spec
  is shared with Stages 2–3, so retirement leaves it in place (Plan completion).

### Project rules

- **Style (CLAUDE.md).**
  - Single quotes, including `'''` docstrings. A string with an apostrophe takes double quotes
    (ruff Q003).
  - YAPF owns layout (100 columns), and ruff lints (E, F, I, Q). **Never run `ruff format`.**
  - One blank line between top-level definitions and after imports.
  - Semantic section dividers; `logging` rather than `print`; type hints on signatures.
  - Code, comments and docstrings cite the spec (`Req 21`) or the embedding roadmap's decisions
    (`D7`), never a task number or a P-number, which mean nothing once the plan retires.
- **Formatting.** Format touched files with `./scripts/format_code.sh <files>`. The plan's code was
  formatted at plan time, so the script should change nothing. If it changes a file, keep its
  layout and say so in the ledger. At the end, `./scripts/format_code.sh --check --all` must pass.
- **Git.**
  - Never push without the user's go-ahead, and never push to `main`.
  - Local `main` carries private commits that must never be pushed: `config`, `graph config` and
    the others the main checkout's pre-push hook guards. Never push them or a descendant, never
    cherry-pick or merge them into a public branch, and never bypass the hook with `--no-verify`.
  - Push a branch by name: `git push -u origin <branch>`, never a bare `git push`.
  - Never run bare `git stash`.
  - Add files by explicit path, never `git add -A` or `git add .`.
  - End each commit message with the session's attribution trailer.
- **Data safety.** No task reads real data: no QCEW slice, real panel, bundle or `data/` file.
  Tests use the fixture panel (`tests/fixtures/regressor_panel.py`) and write only under
  `tmp_path`. Growth is never read on real data before Stage 3, which needs the user's approval
  at the time.
- **The selection log.** The main checkout's `logs/selection_log.jsonl` is append-only (Req 4).
  Nothing in this plan reads or appends to it.
- **Sealed splits.** No step opens a real split. Fixture tests open fixture outer sets, in
  temporary logs, by design.
- **The lock.** `uv.lock` has sha256
  `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21`. The freeze is released
  (Req 19), but this plan needs no dependency change. If a step would change the lock, stop and
  ask.
- **Configs.** `conf/config.yaml` is unchanged and keeps `supervision.manifest_path: null`.
  `conf/data/regressor_panel.yaml` gains only `outcome: level` and its comment; its pins are
  unchanged.
- **Downloads.** None.
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-representation-split-roadmap.md` and `specs/deferred_items.md`, and hand the user
  the exact edit instead.
- **Docs.**
  - `uv run mkdocs build --strict` must pass locally. PR CI also builds the docs strictly.
  - Griffe is strict: every documented parameter gets its own `Args:` entry.
- **Tests.**
  - Rich wraps at 80 columns on CI, so CLI tests match on whitespace-collapsed output
    (`' '.join(result.output.split())`).
  - pyproject's `addopts` has `-v`, so a count-only listing needs `--collect-only -q -q`.
- **Bash tool.** It runs zsh.
  - Brace variables (`${FILE}`): `$FILE:t` is a zsh modifier.
  - Quote `=`-leading words (`echo '====='`) and globs (`--include='*.py'`); an unquoted one
    aborts the command.
  - If the tool refuses a heredoc or a compound command, run one plain command per call and
    write files with the Write tool.

## Decisions

The brief that ordered this plan (2026-10-12) gave these rulings, which the spec records as user
rulings. Do not re-triage them. Each entry carries its U-number rather than a D-number, because
bare D-numbers are the embedding roadmap's (Locators).

- **U6, U8** (owner, brief, 2026-10-12): growth only, on the NAICS 2022 basis only. The growth
  panel is the economic artifact's only panel for now. Applied in: Tasks 2–3.
- **U10** (owner, brief, 2026-10-12): retarget the regressor panel in place, with an `outcome`
  setting that defaults to `level`. Applied in: Tasks 2–3 (P2–P4).
- **U12** (owner, brief, 2026-10-12): D1's covariates are kept: log establishments and log wages
  in year t. Applied in: Task 2.
- **U8** (owner, brief, 2026-10-12): the levels regimes stay sealed and are not decision panels.
  Applied in: Tasks 4–5 (P5, P6); Project rules (Sealed splits).

---

## Workspace

- **Execution ledger:** `logs/plan13_execution.md` in the execution worktree, ignored by git.
  Write each task's commands, outputs and deviations there. Integration's last step copies it to
  the main checkout before the worktree is removed.
- **Worktree:** the app worktree the execution session runs in, or
  `/Users/lowell/Projects/naics-embedder/.claude/worktrees/plan-13-growth`. Run every command
  from its root.
- **Branch:** `claude/stage-1-growth-outcome`, cut at `origin/main` once this plan has merged
  (**Pre-flight** Step 1).
  - An app worktree is cut from local `main`, which carries the private commits. So the branch
    is always cut from `origin/main`, never from local `main`.
  - Without an app worktree, run, from the main checkout,
    `git worktree add --detach .claude/worktrees/plan-13-growth origin/main`, then cut the branch
    there.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder`. Nothing in this plan writes there
  but Integration's ledger copy. At plan time, local `main` was 6d28d91: origin/main d61f49b plus
  the private commits.
- **Python 3.10 environment:** `/tmp/naics_plan13_py310`. Task 1 builds it, Final verification
  reuses it, and Final verification deletes it.
- **Docs build output:** `/tmp/naics_plan13_site`, deleted after each build.
- **Working directory.** The Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before every commit. If it is not this worktree, `cd` back first, or
  prefix the command with `cd <worktree> &&`.

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- Pre-flight's lock hash, platform or baseline differs from its Expected line.
- Task 1's pins fail on this Mac before any source change: the fixture or the environment
  differs from plan time.
- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A red step passes, or fails other than as its Expected line says, and the cause is not a
  transcription slip.
- On CI, a Task 1 pin not marked `CAPTURED_HERE` fails while it passes here. Suspect libm or BLAS
  before logic, and report the failing assertion. Never loosen a tolerance, and never mark a test
  platform-specific, without the user's ruling.
- A step would do any of these:
  - read real QCEW slices, a real panel, the bundle, `data/` or the main checkout's selection
    log;
  - open a real split, or read growth on real data;
  - change `uv.lock` or `conf/config.yaml`;
  - write the main checkout beyond Integration's ledger copy;
  - push a private commit or a descendant, or push with `--no-verify`.
- `origin/main` gains a commit that touches a file in **File structure**, the split roadmap or
  `specs/deferred_items.md`, or an open PR does.

## File structure

The complete file map. Plan completion also edits `specs/naics-representation-split-roadmap.md`
and moves this plan to `specs/plans/completed/`. `AGENTS.md` is a symlink to `CLAUDE.md` and
needs no edit of its own.

| Action | Path | Tasks | Responsibility |
|---|---|---|---|
| Modify | `tests/fixtures/regressor_panel.py` | 1, 3 | The d61f49b oracle, `panel_rows_d61f49b` (1); `synthetic_rows(outcome=)` and `regressor_growth_rows` (3) |
| Modify | `tests/unit/test_regressor_panel.py` | 1, 3, 4 | Level mode's pins (1); growth reads under their own names (3); one seal per split (4) |
| Modify | `src/naics_embedder/panels/qcew_rows.py` | 2 | `RegressorOutcome`; `panel_rows(outcome=)` |
| Modify | `tests/unit/test_regressor_qcew_rows.py` | 2 | Growth rows from the same cells |
| Modify | `src/naics_embedder/panels/regressor.py` | 3, 4 | `READ_NAMES`, the panel's `outcome` and `read_names`, the outcome through `from_sources` and `load_regressor_panel` (3); `seal_names` and openings under both outcomes (4) |
| Modify | `src/naics_embedder/utils/config.py` | 3 | `RegressorPanelConfig.outcome` |
| Modify | `conf/data/regressor_panel.yaml` | 3, 5 | `outcome: level` (3); its comment names the level-only commands (5) |
| Modify | `tests/unit/test_config.py` | 3 | The outcome's default and choices |
| Modify | `src/naics_embedder/panels/selection_log.py` | 4 | `openings` over a collection of names |
| Modify | `src/naics_embedder/cli/commands/tools.py` | 5 | `tools regressor-panel` refuses growth |
| Modify | `src/naics_embedder/decision/sweep.py` | 5 | `run_seed_sweep` refuses a growth panel |
| Modify | `tests/unit/test_cli_commands.py` | 5 | The CLI's refusal |
| Modify | `tests/unit/test_decision_sweep.py` | 5 | The sweep's refusal |
| Modify | `docs/usage.md` | 5 | The `outcome` setting; level mode only from the CLI and the sweep |
| Modify | `docs/api/regressor_panel.md` | 6 | The intro names both outcomes and the shared seal |
| Modify | `CLAUDE.md`, `README.md`, `tests/README.md` | 6 | The setting, the count and the test seams |

## Pre-flight (controller, inline, before Task 1)

- [ ] **Step 1: Confirm the workspace**

Run: `git fetch origin`, then:

```bash
git log --oneline --reverse origin/main -- specs/plans/13-growth-outcome-and-seals.md
```
Expected: this plan's commits, oldest first: `docs(plans): add plan 13, growth outcome and one
seal per split`, then any review follow-ups (`docs(plans): …`). If there is no output, this plan
has not merged: stop and ask.

Run: `git checkout --no-track -B claude/stage-1-growth-outcome origin/main`
Expected: `Switched to a new branch 'claude/stage-1-growth-outcome'` (or `Reset branch`).

Run: `git status --short --branch`, then `git log --oneline origin/main..HEAD`
Expected: `## claude/stage-1-growth-outcome` and nothing else, then no output. No private commit
is on this branch.

Run: `git log --oneline d61f49b..origin/main -- src tests conf docs specs CLAUDE.md README.md`
Expected: this plan's commits, and possibly their merge commit; no other commit.
- If anything else landed, read it.
- If it touches a file in **File structure**, the split roadmap or `specs/deferred_items.md`,
  stop and ask.

Run: `gh pr list --state open`
Expected: no open PR touching a file in **File structure**. If one does, stop and ask.

- [ ] **Step 2: Build the environment**

Run: `uv sync --locked`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number (3.12.12 at plan time).

Run:

```bash
uv run python -c "import numpy, polars, pydantic; print(numpy.__version__, polars.__version__, pydantic.__version__)"
```
Expected: `2.3.4 1.35.1 2.12.4`.

Run: `shasum -a 256 uv.lock`
Expected: `4167042e8a5a8caa9af62973151f681fffb50afaa1a7f6d1f801bd9e58bdac21  uv.lock`. If it
differs, stop and ask.

Run: `uname -sm`
Expected: `Darwin arm64`. Task 1's literal pins run only there (P1). Elsewhere they skip, and the
plan's counts do not hold: stop and ask.

- [ ] **Step 3: Run the baseline suite**

Run: `uv run pytest -n auto -q`
Expected: `3098 passed, 2 skipped`, measured at d61f49b on 2026-10-12 in a worktree without
`data/`. One skip needs CUDA. The other needs `data/naics_descriptions.parquet`, which only the
main checkout has; there the suite shows `3099 passed, 1 skipped`.
- Each later full-suite run must pass with those skips, and each task gives its count.
- The warnings count varies under xdist; ignore it.

The full suite after each task, measured at plan time on this Mac (no `data/` in the worktree):

| After | Passed | Skipped | Collected |
|---|---:|---:|---:|
| Pre-flight (d61f49b) | 3098 | 2 | 3100 |
| Task 1 | 3103 | 2 | 3105 |
| Task 2 | 3107 | 2 | 3109 |
| Task 3 | 3113 | 2 | 3115 |
| Task 4 | 3115 | 2 | 3117 |
| Task 5 | 3117 | 2 | 3119 |
| Task 6 | 3117 | 2 | 3119 |

CI runs Linux, with no MPS and no `data/`. Each test job runs pytest twice: a 47-test
remote-transport step, then the full suite with coverage. At d61f49b the full suite's summary
line read `3089 passed, 11 skipped` under both Pythons. After Task 5 it should read
`3106 passed, 13 skipped`: the 19 new tests, two of which, Task 1's `CAPTURED_HERE` tests, skip
there.

- [ ] **Step 4: Route the tasks**

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–5 each get a fresh implementer and a task-reviewer. Give each implementer its task,
  **Global Constraints** and **Workspace**.
- Every code block in a task is exact, so an implementer copies it. Each "Replace" text occurs
  exactly once in its file when its edit is made, and one file's edits are made in the order
  given. At plan time, a script applied every block of this plan, in order, to d61f49b's files
  and reproduced each task's dry-run commit exactly.
- Task 6 (documentation) may go to the docs-writer agent. Its acceptance checks are the gate.
- **Final verification**, **Plan completion** and **Integration** run inline in the controller
  session.

## Tasks

Each task's red step was measured at plan time on the task's parent commit with the task's test
edits applied, and each green step on the task's own commit. The counts are for the named test
files only. A **Files** list's line ranges locate its Replace texts in the file as the task finds
it, before its edits.

### Task 1: Pin level mode before any change

Req 21's "Levels unchanged" needs level mode's values pinned before anything changes. Nothing pins
them today: the panel tests check properties, such as no leakage, folds that keep a group
together and sealed outer sets, not values. This task adds the pins in a commit of their own,
before any source change (P1).

- `panel_rows_d61f49b`, in the fixture module, is `panel_rows` as d61f49b wrote it, with the
  usable-cell rule, the feature years and the columns it reads written out. Level mode's rows
  must equal its output bit for bit at every level.
- `level_reads` reads the fixture panel in a fixed order: each defined cell's validation split,
  then each regime's outer set, opened once, at each defined level. The pins cover its log, each
  read's keys and chosen penalties, its sums, and level 6's folds.
- `CAPTURED_HERE` marks the two tests whose literals hold where they were captured: the data
  fingerprints, and the digests over every prediction column.

The values were captured at d61f49b on this Mac, under Python 3.12 (numpy 2.3.4) and Python 3.10
(numpy 2.2.6), which agree bit for bit. The pins pass when written, so this task has no red step.
Step 3 shows that they can fail.

**Files:**
- Test: `tests/fixtures/regressor_panel.py`, lines 169-170
- Test: `tests/unit/test_regressor_panel.py`, lines 8-9, 16-17, 52-53 and the end of the file

**Interfaces:**
- Consumes: nothing from an earlier task. It uses, unchanged:
  - `panel_rows`, `level_cells` and `population` (`naics_embedder.panels.qcew_rows`);
  - `RegressorPanel`, `Regime`, `LEVELS` and `PREDICTION_COLUMNS`
    (`naics_embedder.panels.regressor`), `assign_splits` (`panels.regressor_splits`) and
    `SelectionLog` (`panels.selection_log`);
  - the fixture module's `CODEBOOK`, `HELDOUT_GROUPS` and `SETTINGS`, and its session fixtures
    `regressor_cells`, `regressor_rows` and `regressor_arm`;
  - the test module's `panel` and `log` fixtures and its `PURPOSE`.
- Produces:
  - `panel_rows_d61f49b(cells, codes) -> pl.DataFrame`, in `tests/fixtures/regressor_panel.py`.
  - In `tests/unit/test_regressor_panel.py`:
    - the pins `LEVEL_DRAW`, `LEVEL_PARTITIONS`, `LEVEL_LOG`, `LEVEL_READS`, `LEVEL_SIX_FOLDS`
      and `LEVEL_DATA_FINGERPRINTS`;
    - `PARTITION_COLUMNS`, `KEY_COLUMNS`, the `CAPTURED_HERE` mark and
      `_digest(frame, columns) -> str`;
    - the module fixture `level_reads`, which returns the predictions by `(regime, split, level)`
      and the log's records;
    - five tests, all named `test_level_…`.

    No later task changes a pin.

- [ ] **Step 1: Write the pins**

In `tests/fixtures/regressor_panel.py`, make one edit.

Replace:

```python
# -------------------------------------------------------------------------------------------------
# Stub arm tables
```

with:

```python
# -------------------------------------------------------------------------------------------------
# Level mode as d61f49b built it (Req 21, "Levels unchanged")
# -------------------------------------------------------------------------------------------------

def panel_rows_d61f49b(cells: pl.DataFrame, codes) -> pl.DataFrame:
    '''
    ``qcew_rows.panel_rows`` as d61f49b had it, before the growth outcome, with the usable-cell
    rule, feature years and columns it read written out. Level mode's rows must equal its output
    bit for bit on every platform: both take their logs from the platform's own libm.
    '''

    usable = cells.filter(
        (pl.col('disclosure_code') == '') & (pl.col('emp') > 0) & (pl.col('estabs') > 0)
        & (pl.col('wages') > 0) & pl.col('code').is_in(list(codes))
    )
    features = usable.filter(pl.col('year').is_in([2022, 2023, 2024])).select(
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
    columns = ['code', 'feature_year', 'outcome_year', 'log_estabs', 'log_wages', 'outcome']
    return rows.select(columns).sort('code', 'feature_year')

# -------------------------------------------------------------------------------------------------
# Stub arm tables
```

In `tests/unit/test_regressor_panel.py`, make three edits, in this order, then append to its end.

Replace:

```python
import hashlib

```

with:

```python
import hashlib
import json
import platform
import sys

```

Replace:

```python
from naics_embedder.panels.qcew_rows import level_cells, population
from naics_embedder.panels.regressor import (
```

with:

```python
from naics_embedder.panels.qcew_rows import level_cells, panel_rows, population
from naics_embedder.panels.regressor import (
    LEVELS,
```

Replace:

```python
    coordinate_table,
    text_only_table,
```

with:

```python
    coordinate_table,
    panel_rows_d61f49b,
    text_only_table,
```

Append to the end of the file:

```python

# -------------------------------------------------------------------------------------------------
# Levels unchanged (Req 21): level mode's values, pinned at d61f49b before the growth outcome
# -------------------------------------------------------------------------------------------------

# Captured at d61f49b on macOS arm64, where Python 3.12 (numpy 2.3.4) and Python 3.10 (numpy 2.2.6)
# agree bit for bit under polars 1.35.1
LEVEL_DRAW = 'b47d16873ea7065cdd9a1b9718268d9d5095dfc2b5879c3e966710e3bec9c78d'
# Each level's rows per split (remainder, seen outer set, held-out outer set), and the digest of
# every row's code, feature year, outcome year, group and split, sorted by code and feature year
LEVEL_PARTITIONS = {
    2: ((2, 1, 9), 'ae050a881bfaaeeea3588e545a50f6930afed9f912e808b2ff3868c1bfa41e97'),
    3: ((10, 5, 9), '2cf28b0b3ed39230fa250133b3269d2c081065a4e62b0d72519d973d21d34de7'),
    4: ((28, 14, 9), '40ee441f5f6fc808275c21555925f4247a4cabd90d0b2402b65b2b5e69afaa24'),
    5: ((28, 14, 9), '536780bc9117e2e5b7551d7f8169f655b5223d91fbb7ce5ab584428529e945be'),
    6: ((52, 26, 18), '1f719ca447c6ac66307846f5b869e396ffa04db423c355710288c5fae639d1f7'),
}
# The log of ``level_reads``: each defined cell's validation read, then each regime's opening and
# its outer-set reads
LEVEL_LOG = [
    ('read', 'regressor_seen', 'validation', 10),
    ('read', 'regressor_seen', 'validation', 28),
    ('read', 'regressor_seen', 'validation', 28),
    ('read', 'regressor_seen', 'validation', 52),
    ('read', 'regressor_heldout', 'validation', 28),
    ('read', 'regressor_heldout', 'validation', 28),
    ('read', 'regressor_heldout', 'validation', 52),
    ('open', 'regressor_seen', 'test', 26),
    ('read', 'regressor_seen', 'test', 5),
    ('read', 'regressor_seen', 'test', 14),
    ('read', 'regressor_seen', 'test', 14),
    ('read', 'regressor_seen', 'test', 26),
    ('open', 'regressor_heldout', 'test', 18),
    ('read', 'regressor_heldout', 'test', 9),
    ('read', 'regressor_heldout', 'test', 9),
    ('read', 'regressor_heldout', 'test', 18),
]
# Each read's digest of its key columns and chosen penalties, the sums of its predictions and of
# their squared errors, and the digest of every prediction column
LEVEL_READS = {
    (Regime.SEEN, 'validation', 3): (
        '0d3fe43ef5a84011a6c630421057bacfa43924b7c494e6c5cd6a8a4747467791',
        667.4451283013116,
        5.130531432910643,
        '650a98f3f27e0bd8ba0ab1302ff0dc3ca4b7c904b90a4abfbdb84d78f8ba2761',
    ),
    (Regime.SEEN, 'validation', 4): (
        '2a0db0140205fced1794f7e7d1fba820bc7e90833078443ba2e88a722d889f86',
        1771.3781051987503,
        79.44066289871006,
        '6fe8812179def263a314bc78e935eb9c73103981689d749db49d9cdb20ce42b5',
    ),
    (Regime.SEEN, 'validation', 5): (
        '348a229977df37a13fecec449db1c4fc59a9f3041e01de54395f47233c659705',
        1672.4597586951015,
        162.28358979019288,
        '178f31610ecabfca5a6acdfcb2f2041704d12cb0d2408a1ce9e5005114683b12',
    ),
    (Regime.SEEN, 'validation', 6): (
        'd6a1914d785a674c188d50e7c1eddd738bd079ee50e11ce15a83320dc54e1446',
        3127.5068050704963,
        131.82012311086612,
        'bd48f3a7b2a5e128eeb2cd3f2bc300013455ae238483c804731c532bf051adac',
    ),
    (Regime.HELDOUT, 'validation', 4): (
        'd4ef9be4090fa25a61cc0f756f1adddd2616a22830d483361c43d9e9894bd658',
        2762.234709277887,
        311.1588055545941,
        '3331930a7ef3034fa708b88e148125084527b552ec5a1277e43e8e5cd42f6b95',
    ),
    (Regime.HELDOUT, 'validation', 5): (
        '8a5720352cf749805294ab4dd431122f86c5e01c7fbe117b4d352451aaa2189f',
        2603.8437776456385,
        902.0747609652292,
        '21dd412769a514eda499d2413c0656ec608a3365c4ffe5ba80e409041098c734',
    ),
    (Regime.HELDOUT, 'validation', 6): (
        'fa2b27d4692a5fc28c294a71c58283f73f9eed0a7f626b4fccf14df7f5890d91',
        4890.906428128532,
        438.6849114831849,
        '6173effe96fa20085250cb6f42fb2647e76574cc68cf18401ab7c378d3c287fc',
    ),
    (Regime.SEEN, 'test', 3): (
        '858a694a6dd83a2763f5d1832d9e43d5fdf916a7026c858e1c91714d67bedfb3',
        335.91313347500005,
        1.141657267795871,
        'f1662bb791d10ef9f03d194defe2cc3efcd34e35567ddbda8915e9bdb6ae6871',
    ),
    (Regime.SEEN, 'test', 4): (
        '3aa1a3896e4548f68397f51f5da22fe5e9b913527af8c0fd02f302b7797e25d2',
        889.6516122009634,
        33.50275136228421,
        '5d7dd4b34c774e42b1de017a5135b05dc917e66a7641fd47973c6df0ecb5b3ca',
    ),
    (Regime.SEEN, 'test', 5): (
        'de3c54f5b8e38ec881a3f7f1f0d91434c77980a741cba24535169ab17b875c46',
        839.5593902615201,
        70.59059817108438,
        'cc3f2ed072f133b27fb22a8168d5f3a21bdc25a090c4198067ac6b72f0f66942',
    ),
    (Regime.SEEN, 'test', 6): (
        '4557c61fe0967915cddc93cd3fc2f3735c01d8ebf62b33445815c260ce50ec8b',
        1572.8628775819825,
        69.21606532040232,
        'c4b2af96f52d1192f35dcfa3860455d9be47bfe516c5407baf176560e3bcb8c5',
    ),
    (Regime.HELDOUT, 'test', 4): (
        '96a88a55e35dde451daa931a1c32be5fc46e0dd547fd74bbd5cc34da21cc5f30',
        437.0580054753184,
        116.30107865000055,
        'de2f101d3a16831cf7d3ea39a3ae31752fcd6f3abe85914bf57af52a51d3d0c5',
    ),
    (Regime.HELDOUT, 'test', 5): (
        'ae4b294c299114c6984dc2366ae0174841bf0456068ecb6953c145c190f95c33',
        384.49482348133023,
        165.4899488131396,
        '3b6a5b59e1e81e715ae3dd6db70c15a13a67c9eea98d10956129c44ccc1a4f10',
    ),
    (Regime.HELDOUT, 'test', 6): (
        '8b25d80ee6fe623a979501877b9ce3fa2229ee99baa36d6138141bf9a272a303',
        857.1212167419905,
        98.41731977089233,
        '497f48b7e809d2984b20762684e67b6f4c07a90f2ae3e174f5937ebc42b794e8',
    ),
}
# Level 6's validation folds: each regime's and repeat's groups in fold 0, then in fold 1
LEVEL_SIX_FOLDS = {
    (Regime.SEEN, 0): (
        ('1111', '1121', '1122', '2381', '3111', '3112', '5221'), (
            '1112', '1114', '3212', '3321', '3322', '5222', '5232'
        )
    ),
    (Regime.SEEN, 1): (
        ('1111', '1112', '1114', '2381', '3111', '3322', '5222'), (
            '1121', '1122', '3112', '3212', '3321', '5221', '5232'
        )
    ),
    (Regime.HELDOUT, 0): (
        ('1114', '2381', '3111', '3212', '3321', '3322', '5221'), (
            '1111', '1112', '1121', '1122', '3112', '5222', '5232'
        )
    ),
    (Regime.HELDOUT, 1): (
        ('1111', '1114', '1122', '2381', '3112', '3322', '5221'), (
            '1112', '1121', '3111', '3212', '3321', '5222', '5232'
        )
    ),
}
LEVEL_DATA_FINGERPRINTS = {
    2: 'eca3bfb9319d9c28ba0e48b8cca2279e56aac8711113833bb24d03be96f7881a',
    3: '1b3f9104506d9f6866131731d0f5d7f2c4422538fb0172eae80d12e3b2906e0d',
    4: 'cba83a33960191032af7cff326c02d37c17ec087cd8c206d722bd729a793160c',
    5: '221c729fbd73c45fe07a136cc8c5ca904a022d69390b6bc4dfed5c0a59ce9443',
    6: '9501d616de750922fb899f9310be2ab00587bc9487dc7542b3e1f77ada6821c7',
}
PARTITION_COLUMNS = ['code', 'feature_year', 'outcome_year', 'group', 'split']
KEY_COLUMNS = [name for name in PREDICTION_COLUMNS if name not in ('outcome', 'prediction')]
# Literal float pins hold where they were captured. Another libm can round a log a last bit
# differently: 7 of level 6's cell values lie within glibc's 0.519-ULP error bound of a rounding
# midpoint, the closest at 0.0019 ULP. Another BLAS can round a ridge fit differently. The d61f49b
# oracle and the float-free digests hold on every platform.
CAPTURED_HERE = pytest.mark.skipif(
    (sys.platform, platform.machine()) != ('darwin', 'arm64'),
    reason='pinned where captured, on macOS arm64: another libm or BLAS can differ in a last bit',
)

def _digest(frame, columns):
    '''SHA-256 of the columns' values as JSON, row by row in the frame's order.'''

    return hashlib.sha256(json.dumps(frame.select(columns).rows()).encode('utf-8')).hexdigest()

@pytest.fixture(scope='module')
def level_reads(regressor_rows, regressor_arm, tmp_path_factory):
    '''
    Every read of the fixture panel in level mode: each defined cell's validation split, then each
    regime's outer set, opened once, at each defined level. Returns the predictions by (regime,
    split, level), and the log's records.
    '''

    log = SelectionLog(tmp_path_factory.mktemp('level') / 'selection_log.jsonl')
    panel = RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS)
    cells = [
        (regime, level) for regime in Regime for level in panel.levels
        if panel.cell_status(regime, level) is None
    ]
    reads = {
        (regime, 'validation', level): panel.validation(regime, level, regressor_arm, PURPOSE)
        for regime, level in cells
    }
    for regime in Regime:
        panel.open_outer(regime, PURPOSE)
        for level in [level for other, level in cells if other is regime]:
            reads[regime, 'test', level] = panel.test(regime, level, regressor_arm, PURPOSE)
    return reads, log.records()

def test_level_rows_are_d61f49bs_at_every_level(regressor_cells, regressor_rows):
    for level in LEVELS:
        cells = level_cells(regressor_cells, CODEBOOK, level)
        codes = population(cells)
        oracle = panel_rows_d61f49b(cells, codes)

        assert panel_rows(cells, codes).equals(oracle)
        assert regressor_rows[level].equals(oracle)

def test_level_partitions_and_data_fingerprints_are_pinned(panel, regressor_rows):
    assert panel.fingerprint == LEVEL_DRAW
    for level, (counts, digest) in LEVEL_PARTITIONS.items():
        frame = assign_splits(regressor_rows[level], HELDOUT_GROUPS).sort('code', 'feature_year')
        named = frame.select(sorted(frame.columns))
        payload = json.dumps({'columns': named.columns, 'rows': named.rows()}).encode('utf-8')

        assert tuple(panel.split_counts(level).values()) == counts
        assert _digest(frame, PARTITION_COLUMNS) == digest
        # The data fingerprint hashes every column of those rows, in name order
        assert panel.data_fingerprint(level) == hashlib.sha256(payload).hexdigest()

@CAPTURED_HERE
def test_level_data_fingerprints_are_the_captured_ones(panel):
    assert {level: panel.data_fingerprint(level) for level in LEVELS} == LEVEL_DATA_FINGERPRINTS

def test_level_reads_and_folds_are_pinned(level_reads):
    reads, records = level_reads

    assert [(r['event'], r['panel'], r['split'], r['n_queries']) for r in records] == LEVEL_LOG
    assert list(reads) == list(LEVEL_READS)
    for key, (digest, prediction_sum, error_sum, _) in LEVEL_READS.items():
        sums = reads[key].select(
            predictions=pl.col('prediction').sum(),
            squared_errors=((pl.col('prediction') - pl.col('outcome'))**2).sum(),
        ).row(0)
        assert _digest(reads[key], KEY_COLUMNS) == digest
        assert sums == pytest.approx((prediction_sum, error_sum), rel=1e-9)
    for (regime, repeat), folds in LEVEL_SIX_FOLDS.items():
        rows = reads[regime, 'validation', 6].filter(
            (pl.col('comparator') == 'covariates') & (pl.col('repeat') == repeat)
        )
        assigned = tuple(
            tuple(sorted(rows.filter(pl.col('fold') == fold).get_column('group').unique()))
            for fold in range(SETTINGS.folds)
        )
        assert assigned == folds

@CAPTURED_HERE
def test_level_predictions_are_the_captured_ones_bit_for_bit(level_reads):
    reads, _ = level_reads

    for key, (*_, digest) in LEVEL_READS.items():
        assert _digest(reads[key], list(PREDICTION_COLUMNS)) == digest
```

- [ ] **Step 2: Run the pins**

Run: `uv run pytest tests/unit/test_regressor_panel.py -n auto -q`
Expected: PASS, `85 passed`. The five pins pass at the source as it stands. If any fails, stop
and ask (**Stop-and-ask conditions**).

- [ ] **Step 3: Check that the pins can fail**

Each mutation changes one source line, runs the pins, and is reverted.

Mutation 1, the held-out regime's fold stream:

```bash
perl -pi -e 's/^REGIME_STREAMS = \{Regime\.SEEN: 0, Regime\.HELDOUT: 1\}$/REGIME_STREAMS = {Regime.SEEN: 0, Regime.HELDOUT: 2}/' src/naics_embedder/panels/regressor.py
```

Run: `git diff --stat`
Expected: `src/naics_embedder/panels/regressor.py | 2 +-`, one file changed.

Run: `uv run pytest tests/unit/test_regressor_panel.py -n auto -q -k level_`
Expected: FAIL, `2 failed, 6 passed`: `test_level_reads_and_folds_are_pinned` and
`test_level_predictions_are_the_captured_ones_bit_for_bit`. The `-k` filter also selects three
older tests whose names contain `level_`; they pass.

Run: `git checkout -- src/naics_embedder/panels/regressor.py`

Mutation 2, every log-employment value raised by one or two units in the last place:

```bash
perl -pi -e 's/(outcome=pl\.col\(.emp.\)\.cast\(pl\.Float64\)\.log\(\))$/$1 * (1 + 2**-52)/' src/naics_embedder/panels/qcew_rows.py
```

Run: `git diff --stat`
Expected: `src/naics_embedder/panels/qcew_rows.py | 2 +-`, one file changed.

Run: `uv run pytest tests/unit/test_regressor_panel.py -n auto -q -k level_`
Expected: FAIL, `3 failed, 5 passed`: `test_level_rows_are_d61f49bs_at_every_level`,
`test_level_data_fingerprints_are_the_captured_ones` and
`test_level_predictions_are_the_captured_ones_bit_for_bit`. The float-free pins pass, as they
should: the change is below their resolution, and the oracle is what catches it on every
platform.

Run: `git checkout -- src/naics_embedder/panels/qcew_rows.py`, then `git status --short`
Expected: only the two test files, ` M tests/fixtures/regressor_panel.py` and
` M tests/unit/test_regressor_panel.py`.

- [ ] **Step 4: Run the pins under Python 3.10**

Run:

```bash
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics_plan13_py310 uv run --locked pytest tests/unit/test_regressor_panel.py -n auto -q
```
Expected:
- `85 passed`, the same as under 3.12 (measured at plan time);
- possibly a few "encountered in matmul" RuntimeWarnings, which come from numpy 2.2 with
  Accelerate on 3.10 and are not a failure.

Keep `/tmp/naics_plan13_py310` for **Final verification**.

- [ ] **Step 5: Format, lint and run the full suite**

Run: `./scripts/format_code.sh tests/fixtures/regressor_panel.py tests/unit/test_regressor_panel.py`
Expected: no file changes (`git status --short` lists the same two files as before).

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run pytest -n auto -q`
Expected: `3103 passed, 2 skipped`.

- [ ] **Step 6: Commit the pins alone**

This commit changes no source file, so the pins are on record before any change.

```bash
git add tests/fixtures/regressor_panel.py tests/unit/test_regressor_panel.py
git commit -m "test(panels): pin level mode's regressor values before the growth outcome"
```

### Task 2: Growth rows

Req 21's growth outcome is log E(t+1) − log E(t), where E is annual-average employment.
`panel_rows` gains an `outcome` argument, a `RegressorOutcome` (P2).

- The level path is unchanged, so Task 1's pins keep passing.
- Growth joins log E(t), computed by the level outcome's own expression from the feature year's
  cell, after the rows are built and their count checked. It then subtracts it.
- Every population code has a usable cell in all four window years (`population`), so the
  difference is always defined. The height check still refuses a code without one, under either
  outcome.

The tests cover:
- each growth outcome against `math.log` of the same cells, and its features' dating;
- a growth row's equality with its level row but for the outcome, and, from 2023 on, the bitwise
  equality of log E(t) with the level outcome a year earlier;
- the enum's values, a string outcome, and the refusal of an unknown one;
- the refusal of a code without every window year, under both outcomes.

**Files:**
- Modify: `src/naics_embedder/panels/qcew_rows.py`, lines 2, 18-19, 25-26, 55-56, 202-209 and
  224-225
- Test: `tests/unit/test_regressor_qcew_rows.py`, lines 2, 16-17 and 124-128

**Interfaces:**
- Consumes: Task 1's pins, which must still pass.
- Produces, in `naics_embedder.panels.qcew_rows`:
  - `RegressorOutcome(str, Enum)`, with `LEVEL = 'level'` and `GROWTH = 'growth'`;
  - `panel_rows(cells, codes, outcome: RegressorOutcome = RegressorOutcome.LEVEL) ->
    pl.DataFrame`. It accepts the enum or its string value. Any other value raises `ValueError`
    with "'rate' is not a valid RegressorOutcome".

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_regressor_qcew_rows.py`, make three edits, in this order.

Replace:

```python
QCEW national rows for the regressor panel (Req 2; roadmap Stage 3; D1; D7).
```

with:

```python
QCEW national rows for the regressor panel (Req 2; Req 21; roadmap Stage 3; D1; D7).
```

Replace:

```python
    WINDOW_YEARS,
    find_split_codes,
```

with:

```python
    WINDOW_YEARS,
    RegressorOutcome,
    find_split_codes,
```

Replace:

```python
def test_panel_rows_refuse_a_code_without_every_window_year(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    with pytest.raises(ValueError, match='panel rows for'):
        panel_rows(six, [*POPULATION, SUPPRESSED_CODE])
```

with:

```python
@pytest.mark.parametrize('outcome', list(RegressorOutcome))
def test_panel_rows_refuse_a_code_without_every_window_year(regressor_cells, outcome):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    with pytest.raises(ValueError, match='panel rows for'):
        panel_rows(six, [*POPULATION, SUPPRESSED_CODE], outcome)

# -------------------------------------------------------------------------------------------------
# The growth outcome (Req 21)
# -------------------------------------------------------------------------------------------------

def test_a_growth_rows_outcome_is_the_log_change_in_employment_from_the_same_cells(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)
    cell = {(row['code'], row['year']): row for row in six.iter_rows(named=True)}

    growth = panel_rows(six, POPULATION, RegressorOutcome.GROWTH)

    assert growth.columns == list(ROW_COLUMNS)
    assert growth.height == len(POPULATION) * len(FEATURE_YEARS)
    for row in growth.iter_rows(named=True):
        now, later = cell[row['code'], row['feature_year']], cell[row['code'], row['outcome_year']]
        change = math.log(later['emp']) - math.log(now['emp'])
        # Features from year t; the outcome from t to t + 1
        assert row['outcome_year'] == row['feature_year'] + 1
        assert row['log_estabs'] == pytest.approx(math.log(now['estabs']), abs=1e-12)
        assert row['log_wages'] == pytest.approx(math.log(now['wages']), abs=1e-12)
        assert row['outcome'] == pytest.approx(change, abs=1e-12)

def test_a_growth_row_differs_from_its_level_row_in_the_outcome_alone(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)
    level = panel_rows(six, POPULATION)
    earlier = level.select(
        'code', feature_year=pl.col('feature_year') + 1, earlier=pl.col('outcome')
    )

    growth = panel_rows(six, POPULATION, RegressorOutcome.GROWTH)

    assert growth.drop('outcome').equals(level.drop('outcome'))
    # From 2023 on, log E(t) is the outcome of the code's level row a year earlier, bit for bit
    paired = growth.join(
        level.select('code', 'feature_year', level=pl.col('outcome')), on=['code', 'feature_year']
    ).join(earlier, on=['code', 'feature_year'])
    assert paired.height == 2 * len(POPULATION)
    assert (
        paired.get_column('outcome') == paired.get_column('level') - paired.get_column('earlier')
    ).all()

def test_an_outcome_is_level_or_growth(regressor_cells):
    six = level_cells(regressor_cells, CODEBOOK, 6)

    by_name = panel_rows(six, POPULATION, 'growth')

    assert [outcome.value for outcome in RegressorOutcome] == ['level', 'growth']
    assert by_name.equals(panel_rows(six, POPULATION, RegressorOutcome.GROWTH))
    with pytest.raises(ValueError, match='not a valid RegressorOutcome'):
        panel_rows(six, POPULATION, 'rate')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_regressor_qcew_rows.py tests/unit/test_regressor_panel.py -n auto -q
```
Expected: FAIL, `85 passed, 1 error`. Collecting `tests/unit/test_regressor_qcew_rows.py` raises
`ImportError: cannot import name 'RegressorOutcome' from 'naics_embedder.panels.qcew_rows'`.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/panels/qcew_rows.py`, make six edits, in this order.

Replace:

```python
QCEW national rows for the regressor panel (roadmap Stage 3; Req 2; D1; D7).
```

with:

```python
QCEW national rows for the regressor panel (roadmap Stage 3; Req 2; Req 21; D1; D7).
```

Replace:

```python
  annual-average employment in year t + 1.
'''
```

with:

```python
  annual-average employment in year t + 1.
- **Growth (Req 21).** In growth mode the outcome is log E(t + 1) − log E(t) instead, from the
  same cells, and nothing else changes. Every population code has a usable cell in every window
  year, so the difference is always defined.
'''
```

Replace:

```python
from collections import Counter
from pathlib import Path
```

with:

```python
from collections import Counter
from enum import Enum
from pathlib import Path
```

Replace:

```python
ROW_COLUMNS = ('code', 'feature_year', 'outcome_year', 'log_estabs', 'log_wages', 'outcome')

```

with:

```python
ROW_COLUMNS = ('code', 'feature_year', 'outcome_year', 'log_estabs', 'log_wages', 'outcome')

class RegressorOutcome(str, Enum):
    '''
    A panel row's outcome (Req 21): ``level`` is log employment in t + 1 (D7), today's panel and
    the default; ``growth`` is its log change from t, the economic artifact's first panel (Req 20).
    '''

    LEVEL = 'level'
    GROWTH = 'growth'

```

Replace:

```python
def panel_rows(cells: pl.DataFrame, codes: Collection[str]) -> pl.DataFrame:
    '''
    One row per code and feature year (D7): year-t covariates and the year-(t + 1) outcome.

    Raises:
        ValueError: If a code lacks a usable cell in a window year.
    '''

```

with:

```python
def panel_rows(
    cells: pl.DataFrame,
    codes: Collection[str],
    outcome: RegressorOutcome = RegressorOutcome.LEVEL,
) -> pl.DataFrame:
    '''
    One row per code and feature year (D7): year-t covariates and the outcome ``outcome`` names.

    The level outcome is log employment in t + 1. The growth outcome subtracts log employment in
    t, read from the same cells by the same expression, so a growth row differs from its level row
    in ``outcome`` alone (Req 21).

    Args:
        cells: One level's cells (``level_cells``).
        codes: The codes to build rows for (``population``).
        outcome: ``level`` (the default) or ``growth``.

    Raises:
        ValueError: If ``outcome`` is neither, or a code lacks a usable cell in a window year.
    '''

    outcome = RegressorOutcome(outcome)
```

Replace:

```python
        raise ValueError(f'{rows.height:,} panel rows for {expected:,} code-years')
    return rows.select(ROW_COLUMNS).sort('code', 'feature_year')
```

with:

```python
        raise ValueError(f'{rows.height:,} panel rows for {expected:,} code-years')
    if outcome is RegressorOutcome.GROWTH:
        # log E(t), from the feature year's own cell
        current = outcomes.select(
            'code', feature_year=pl.col('outcome_year'), current=pl.col('outcome')
        )
        rows = rows.join(current, on=['code', 'feature_year'], how='inner').with_columns(
            outcome=pl.col('outcome') - pl.col('current')
        )
    return rows.select(ROW_COLUMNS).sort('code', 'feature_year')
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_regressor_qcew_rows.py tests/unit/test_regressor_panel.py -n auto -q
```
Expected: PASS, `99 passed`. Level mode's pins pass unchanged.

- [ ] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/panels/qcew_rows.py tests/unit/test_regressor_qcew_rows.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan13_site`, then `rm -rf /tmp/naics_plan13_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3107 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/qcew_rows.py tests/unit/test_regressor_qcew_rows.py
git commit -m "feat(panels): add the growth outcome to the regressor panel's rows (Req 21)"
```

### Task 3: Growth reads under their own names

A panel object carries one outcome (U10), and the outcome names every read (Req 21, "Read
names"; P3, P4).

- `READ_NAMES` maps each outcome to its regimes' read names. `PANEL_NAMES` stays the levels'
  names: `decision/scores.py` imports it for D8's decision, which reads the levels history only.
- `RegressorPanel(..., *, outcome=RegressorOutcome.LEVEL)` sets `outcome` and `read_names`. Every
  read logs and labels its predictions under `self.read_names[regime]`.
- The partition, the draw fingerprint, the folds, the comparators and the fitting are unchanged.
  A growth read scores the same rows, keys and folds as the level read. Only the outcome and the
  data fingerprint, which hashes the rows, differ.
- `RegressorPanelConfig.outcome`, `level` by default, and the YAML's `outcome: level` carry the
  setting. `from_sources` and `load_regressor_panel` pass it on.

The fixture module gains the session fixture `regressor_growth_rows`. The tests cover:
- the read names, and the shared partition and draw;
- a growth read of every comparator on level mode's keys and folds;
- a growth opening's records, and its sealed message;
- `from_sources` under growth, and `load_regressor_panel` under either outcome;
- the config's default and choices.

**Files:**
- Modify: `src/naics_embedder/panels/regressor.py`, lines 8-9, 44-45, 78, 513-514, 521-522, 533-534,
  549-551, 577-578, 674, 696, 739-740, 784, 810, 859 and 870-871
- Modify: `src/naics_embedder/utils/config.py`, lines 547-548
- Modify: `conf/data/regressor_panel.yaml`, lines 13-14
- Test: `tests/fixtures/regressor_panel.py`, lines 21-22, 160-166 and 230-231
- Test: `tests/unit/test_regressor_panel.py`, lines 19-23, 798-800 and the end of the file
- Test: `tests/unit/test_config.py`, lines 578-579

**Interfaces:**
- Consumes: `RegressorOutcome` and `panel_rows(cells, codes, outcome)` (Task 2).
- Produces:
  - In `naics_embedder.panels.regressor`:
    - `READ_NAMES: Dict[RegressorOutcome, Dict[Regime, str]]`, with `regressor_seen`,
      `regressor_heldout`, `growth_seen` and `growth_heldout`;
    - `PANEL_NAMES = READ_NAMES[RegressorOutcome.LEVEL]`;
    - `RegressorPanel(rows_by_level, heldout_groups, log, settings, *,
      outcome=RegressorOutcome.LEVEL)`, with the attributes `outcome: RegressorOutcome` and
      `read_names: Mapping[Regime, str]`;
    - `RegressorPanel.from_sources(..., outcome=RegressorOutcome.LEVEL)`;
    - `load_regressor_panel(cfg, codebook, ...)`, which passes `RegressorOutcome(cfg.outcome)`.
  - `RegressorPanelConfig.outcome: Literal['level', 'growth'] = 'level'`
    (`naics_embedder.utils.config`).
  - In `tests/fixtures/regressor_panel.py`: `synthetic_rows(cells, levels=(2, 3, 4, 5, 6),
    outcome=RegressorOutcome.LEVEL)`, and the session fixture `regressor_growth_rows`.

- [ ] **Step 1: Write the failing tests**

In `tests/fixtures/regressor_panel.py`, make three edits, in this order.

Replace:

```python
    WINDOW_YEARS,
    level_cells,
```

with:

```python
    WINDOW_YEARS,
    RegressorOutcome,
    level_cells,
```

Replace:

```python
def synthetic_rows(cells: pl.DataFrame, levels=(2, 3, 4, 5, 6)) -> Dict[int, pl.DataFrame]:
    '''Panel rows per level, as ``RegressorPanel.from_sources`` builds them.'''

    rows = {}
    for level in levels:
        cells_at_level = level_cells(cells, CODEBOOK, level)
        rows[level] = panel_rows(cells_at_level, population(cells_at_level))
```

with:

```python
def synthetic_rows(cells: pl.DataFrame, levels=(2, 3, 4, 5, 6),
                   outcome=RegressorOutcome.LEVEL) -> Dict[int, pl.DataFrame]:
    '''Panel rows per level with ``outcome``, as ``RegressorPanel.from_sources`` builds them.'''

    rows = {}
    for level in levels:
        cells_at_level = level_cells(cells, CODEBOOK, level)
        rows[level] = panel_rows(cells_at_level, population(cells_at_level), outcome)
```

Replace:

```python
@pytest.fixture(scope='session')
def regressor_arm() -> ArmTables:
```

with:

```python
@pytest.fixture(scope='session')
def regressor_growth_rows(regressor_cells) -> Dict[int, pl.DataFrame]:
    return synthetic_rows(regressor_cells, outcome=RegressorOutcome.GROWTH)

@pytest.fixture(scope='session')
def regressor_arm() -> ArmTables:
```

In `tests/unit/test_regressor_panel.py`, make two edits, in this order, then append to its end.

Replace:

```python
from naics_embedder.panels.qcew_rows import level_cells, panel_rows, population
from naics_embedder.panels.regressor import (
    LEVELS,
    OUTER_SPLITS,
    PREDICTION_COLUMNS,
```

with:

```python
from naics_embedder.panels.qcew_rows import (
    RegressorOutcome,
    level_cells,
    panel_rows,
    population,
)
from naics_embedder.panels.regressor import (
    LEVELS,
    OUTER_SPLITS,
    PANEL_NAMES,
    PREDICTION_COLUMNS,
    READ_NAMES,
```

Replace:

```python

    assert panel.settings == SETTINGS
    assert panel.levels == (6, )
```

with:

```python
    growth = load_regressor_panel(
        cfg.model_copy(update={'outcome': 'growth'}), codebook, levels=(6, )
    )

    assert panel.settings == SETTINGS
    assert panel.levels == (6, )
    assert (panel.outcome, growth.outcome) == (RegressorOutcome.LEVEL, RegressorOutcome.GROWTH)
    assert growth.data_fingerprint(6) != panel.data_fingerprint(6)
```

Append to the end of the file:

```python

# -------------------------------------------------------------------------------------------------
# The growth outcome (Req 21): level mode's rows, partition, draw and folds, under its own names
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def growth_panel(regressor_growth_rows, log):
    return RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )

def test_each_outcome_reads_under_its_own_names():
    assert READ_NAMES == {
        RegressorOutcome.LEVEL: {
            Regime.SEEN: 'regressor_seen',
            Regime.HELDOUT: 'regressor_heldout'
        },
        RegressorOutcome.GROWTH: {
            Regime.SEEN: 'growth_seen',
            Regime.HELDOUT: 'growth_heldout'
        },
    }
    # The levels history's names, which D8's decision reads
    assert PANEL_NAMES == READ_NAMES[RegressorOutcome.LEVEL]

def test_growth_keeps_level_modes_partition_and_draw(
    panel, growth_panel, regressor_rows, regressor_growth_rows
):
    assert (panel.outcome, growth_panel.outcome) == (
        RegressorOutcome.LEVEL, RegressorOutcome.GROWTH
    )
    assert growth_panel.fingerprint == panel.fingerprint
    for level in LEVELS:
        level_frame = assign_splits(regressor_rows[level], HELDOUT_GROUPS)
        growth_frame = assign_splits(regressor_growth_rows[level], HELDOUT_GROUPS)

        assert growth_frame.drop('outcome').equals(level_frame.drop('outcome'))
        assert growth_panel.split_counts(level) == panel.split_counts(level)
        # Only the outcome differs, and the data fingerprint names it
        assert growth_panel.data_fingerprint(level) != panel.data_fingerprint(level)

def test_growth_reads_score_every_comparator_on_level_modes_folds_under_their_own_names(
    panel, growth_panel, regressor_arm, log
):
    keys = ['comparator', 'repeat', 'fold', 'code', 'group', 'feature_year', 'outcome_year']

    for regime in Regime:
        level = panel.validation(regime, 6, regressor_arm, PURPOSE)
        growth = growth_panel.validation(regime, 6, regressor_arm, PURPOSE)

        assert growth.columns == list(PREDICTION_COLUMNS)
        name = READ_NAMES[RegressorOutcome.GROWTH][regime]
        assert growth.get_column('panel').unique().to_list() == [name]
        assert growth.select(keys).equals(level.select(keys))
        assert growth.get_column('prediction').is_finite().all()

    records = log.records()
    assert [(r['panel'], r['split'], r['n_queries']) for r in records] == [
        ('regressor_seen', 'validation', 52),
        ('growth_seen', 'validation', 52),
        ('regressor_heldout', 'validation', 52),
        ('growth_heldout', 'validation', 52),
    ]
    # Under the draw's fingerprint, with the level read's detail
    assert {r['fingerprint'] for r in records} == {panel.fingerprint}
    assert records[1]['detail'] == records[0]['detail']

def test_a_growth_opening_and_its_reads_are_logged_under_the_growth_names(
    growth_panel, regressor_arm, log
):
    growth_panel.open_outer(Regime.HELDOUT, PURPOSE)
    predictions = growth_panel.test(Regime.HELDOUT, 6, regressor_arm, PURPOSE)

    assert predictions.get_column('panel').unique().to_list() == ['growth_heldout']
    assert [(r['event'], r['panel'], r['split'], r['n_queries']) for r in log.records()] == [
        ('open', 'growth_heldout', 'test', 18),
        ('read', 'growth_heldout', 'test', 18),
    ]
    with pytest.raises(SealedSplitError, match='growth_seen outer set is sealed'):
        growth_panel.test(Regime.SEEN, 6, regressor_arm, PURPOSE)

def test_from_sources_builds_the_rows_of_its_outcome(
    tmp_path, regressor_cells, regressor_growth_rows
):
    built, _ = _from_sources(
        tmp_path, regressor_cells, HELDOUT_GROUPS, outcome=RegressorOutcome.GROWTH
    )

    expected = RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, built.log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )
    assert built.outcome is RegressorOutcome.GROWTH
    for level in built.levels:
        assert built.data_fingerprint(level) == expected.data_fingerprint(level)
```

In `tests/unit/test_config.py`, make one edit.

Replace:

```python

    def test_the_text_only_comparator_reads_like_the_arm(self):
```

with:

```python

    def test_the_outcome_is_level_by_default_or_growth(self):
        # Req 21: level is today's panel; growth is the economic artifact's first panel
        assert RegressorPanelConfig().outcome == 'level'
        assert load_config(RegressorPanelConfig, 'data/regressor_panel.yaml').outcome == 'level'
        assert RegressorPanelConfig(outcome='growth').outcome == 'growth'
        with pytest.raises(ValidationError):
            RegressorPanelConfig(outcome='rate')

    def test_the_text_only_comparator_reads_like_the_arm(self):
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_regressor_panel.py tests/unit/test_config.py tests/unit/test_regressor_qcew_rows.py -n auto -q
```
Expected: FAIL, `1 failed, 232 passed, 1 error`:
- collecting `tests/unit/test_regressor_panel.py` raises `ImportError: cannot import name
  'READ_NAMES' from 'naics_embedder.panels.regressor'`;
- `TestRegressorPanelConfig::test_the_outcome_is_level_by_default_or_growth` fails with
  `AttributeError: 'RegressorPanelConfig' object has no attribute 'outcome'`.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/utils/config.py`, make one edit.

Replace:

```python
        description="SHA-256 of the codebook's codes, sorted, one per line",
    )
```

with:

```python
        description="SHA-256 of the codebook's codes, sorted, one per line",
    )
    outcome: Literal['level', 'growth'] = Field(
        default='level',
        description="Each row's outcome (Req 21): next-year log employment, or its log growth",
    )
```

In `conf/data/regressor_panel.yaml`, make one edit.

Replace:

```yaml
codebook_codes_sha256: 9b646af189dbc46870861367e7cd399d2328972754dbf1abb30ca5a864f2481a

```

with:

```yaml
codebook_codes_sha256: 9b646af189dbc46870861367e7cd399d2328972754dbf1abb30ca5a864f2481a

# Each row's outcome (Req 21): level, log employment in t + 1 (D7), or growth, its log change from
# t, on the same rows, partition and folds; growth reads are logged as growth_seen and
# growth_heldout
outcome: level

```

In `src/naics_embedder/panels/regressor.py`, make fifteen edits, in this order.

Replace:

```python
- **Rows (D7).** A code in a feature year t (2022–2024): covariates from year t, outcome log
  employment in t + 1 (``qcew_rows``).
```

with:

```python
- **Rows (D7; Req 21).** A code in a feature year t (2022–2024): covariates from year t, and the
  panel's ``outcome`` (``qcew_rows``): log employment in t + 1 (``level``, the default) or its
  log change from t (``growth``). The outcome names the reads: ``regressor_seen`` and
  ``regressor_heldout`` for the levels, ``growth_seen`` and ``growth_heldout`` for growth.
```

Replace:

```python
    WINDOW_YEARS,
    level_cells,
```

with:

```python
    WINDOW_YEARS,
    RegressorOutcome,
    level_cells,
```

Replace:

```python
PANEL_NAMES = {Regime.SEEN: 'regressor_seen', Regime.HELDOUT: 'regressor_heldout'}
```

with:

```python
# Each outcome's read names: growth reads stay apart from the levels history (Req 21)
READ_NAMES = {
    RegressorOutcome.LEVEL: {
        Regime.SEEN: 'regressor_seen',
        Regime.HELDOUT: 'regressor_heldout'
    },
    RegressorOutcome.GROWTH: {
        Regime.SEEN: 'growth_seen',
        Regime.HELDOUT: 'growth_heldout'
    },
}
# The levels history's names, which D8's decision reads
PANEL_NAMES = READ_NAMES[RegressorOutcome.LEVEL]
```

Replace:

```python
        settings: Tuning and resampling settings.
    '''
```

with:

```python
        settings: Tuning and resampling settings.
        outcome: The outcome ``rows_by_level`` were built for, which names every read (Req 21).
    '''
```

Replace:

```python
        settings: FitSettings,
    ):
```

with:

```python
        settings: FitSettings,
        *,
        outcome: RegressorOutcome = RegressorOutcome.LEVEL,
    ):
```

Replace:

```python
        self.settings = settings
        self._open: Set[Regime] = set()
```

with:

```python
        self.settings = settings
        self.outcome = RegressorOutcome(outcome)
        self.read_names: Mapping[Regime, str] = READ_NAMES[self.outcome]
        self._open: Set[Regime] = set()
```

Replace:

```python
    ) -> 'RegressorPanel':
        '''
        The panel from the pinned QCEW slices, a codebook and the committed held-out groups.
```

with:

```python
        outcome: RegressorOutcome = RegressorOutcome.LEVEL,
    ) -> 'RegressorPanel':
        '''
        The panel from the pinned QCEW slices, a codebook and the committed held-out groups, its
        rows built for ``outcome`` (Req 21).
```

Replace:

```python
            rows_by_level[level] = panel_rows(cells_at_level, population(cells_at_level))
        return cls(rows_by_level, heldout_groups, SelectionLog(Path(log_path)), settings)
```

with:

```python
            rows_by_level[level] = panel_rows(cells_at_level, population(cells_at_level), outcome)
        log = SelectionLog(Path(log_path))
        return cls(rows_by_level, heldout_groups, log, settings, outcome=outcome)
```

Replace:

```python
        panel = PANEL_NAMES[Regime(regime)]
```

with:

```python
        panel = self.read_names[Regime(regime)]
```

Replace:

```python
        panel = PANEL_NAMES[regime]
```

with:

```python
        panel = self.read_names[regime]
```

Replace:

```python
                f'the {PANEL_NAMES[regime]} outer set is sealed: call open_outer(regime, purpose) '
                'first, which is logged'
```

with:

```python
                f'the {self.read_names[regime]} outer set is sealed: call '
                'open_outer(regime, purpose) first, which is logged'
```

Replace:

```python
            panel=PANEL_NAMES[regime],
```

with:

```python
            panel=self.read_names[regime],
```

Replace:

```python
                        panel=pl.lit(PANEL_NAMES[regime]),
```

with:

```python
                        panel=pl.lit(self.read_names[regime]),
```

Replace:

```python
    '''The panel the config describes, over a codebook, logging to ``log_path`` or the config's.'''
```

with:

```python
    '''
    The panel the config describes, its ``outcome`` included, over a codebook, logging to
    ``log_path`` or the config's.
    '''
```

Replace:

```python
        levels=levels,
    )
```

with:

```python
        levels=levels,
        outcome=RegressorOutcome(cfg.outcome),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_regressor_panel.py tests/unit/test_config.py tests/unit/test_regressor_qcew_rows.py -n auto -q
```
Expected: PASS, `323 passed`.

- [ ] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/panels/regressor.py src/naics_embedder/utils/config.py tests/fixtures/regressor_panel.py tests/unit/test_regressor_panel.py tests/unit/test_config.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan13_site`, then `rm -rf /tmp/naics_plan13_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3113 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/regressor.py src/naics_embedder/utils/config.py conf/data/regressor_panel.yaml tests/fixtures/regressor_panel.py tests/unit/test_regressor_panel.py tests/unit/test_config.py
git commit -m "feat(panels): read the regressor panel's growth outcome under its own names (Req 21)"
```

### Task 4: One seal per split

The outer sets are rows, and 2025 employment underlies both outcomes' outer rows. So an opening is
recorded against the split, not the outcome (Req 22; P5).

- `seal_names(regime)` lists the regime's read name under each outcome.
- `SelectionLog.openings` accepts one name or a collection of names.
- `require_openable` and `open_outer` count the openings under `seal_names(regime)` and the draw
  fingerprint. The fingerprint's inputs are unchanged, so existing records still match. The
  refusal names the read name the split was first opened under.
- `OutcomePanel`'s call, with its one name, is unchanged.

The tests open a split under one outcome. The other outcome's panel is then refused without a
reason, may still open the other regime, and logs a reopening when given one, in both orders. A
third panel, under growth on another draw, still opens its split for the first time.

**Files:**
- Modify: `src/naics_embedder/panels/selection_log.py`, lines 19 and 84-90
- Modify: `src/naics_embedder/panels/regressor.py`, lines 24-26, 176-177, 693-703, 713-714 and 720
- Test: `tests/unit/test_regressor_panel.py`, lines 450-455

**Interfaces:**
- Consumes: `READ_NAMES`, `RegressorPanel(..., outcome=)`, `read_names` and
  `regressor_growth_rows` (Task 3).
- Produces:
  - `seal_names(regime: Regime) -> Tuple[str, ...]` in `naics_embedder.panels.regressor`:
    `('regressor_seen', 'growth_seen')` for the seen regime, and `('regressor_heldout',
    'growth_heldout')` for the held-out one;
  - `SelectionLog.openings(panel: Union[str, Collection[str]], fingerprint: str) ->
    List[Dict[str, Any]]`;
  - the refusal `SplitAlreadyOpenedError('the <read name> outer set was opened at <time> as
    <first read name> for <purpose>; opening it again needs reopen_reason')`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_regressor_panel.py`, make one edit.

Replace:

```python
def test_openings_are_counted_per_held_out_draw(regressor_rows, log):
    RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS).open_outer(Regime.SEEN, 'first')

    RegressorPanel(regressor_rows, ['1111'], log, SETTINGS).open_outer(Regime.SEEN, 'other draw')

    assert [r['event'] for r in log.records()] == ['open', 'open']
```

with:

```python
def test_openings_are_counted_per_held_out_draw(regressor_rows, regressor_growth_rows, log):
    growth = RegressorOutcome.GROWTH
    RegressorPanel(regressor_rows, HELDOUT_GROUPS, log, SETTINGS).open_outer(Regime.SEEN, 'first')

    RegressorPanel(regressor_rows, ['1111'], log, SETTINGS).open_outer(Regime.SEEN, 'other draw')
    third = RegressorPanel(regressor_growth_rows, ['1112'], log, SETTINGS, outcome=growth)
    third.open_outer(Regime.SEEN, 'a third draw')

    assert [r['event'] for r in log.records()] == ['open', 'open', 'open']

@pytest.mark.parametrize(
    ('first', 'second'),
    [
        (RegressorOutcome.LEVEL, RegressorOutcome.GROWTH),
        (RegressorOutcome.GROWTH, RegressorOutcome.LEVEL),
    ],
)
def test_an_opening_under_either_outcome_seals_the_split_for_both(
    regressor_rows, regressor_growth_rows, log, first, second
):
    # Req 22: 2025 employment underlies both outcomes' outer rows, so the split is the key
    rows = {RegressorOutcome.LEVEL: regressor_rows, RegressorOutcome.GROWTH: regressor_growth_rows}
    opener = RegressorPanel(rows[first], HELDOUT_GROUPS, log, SETTINGS, outcome=first)
    other = RegressorPanel(rows[second], HELDOUT_GROUPS, log, SETTINGS, outcome=second)
    opened = READ_NAMES[first][Regime.SEEN]

    opener.open_outer(Regime.SEEN, 'first opening')

    with pytest.raises(SplitAlreadyOpenedError, match=f'as {opened} for'):
        other.require_openable(Regime.SEEN)
    with pytest.raises(SplitAlreadyOpenedError, match='reopen_reason'):
        other.open_outer(Regime.SEEN, 'second opening')
    # Each regime's outer set is its own split
    other.require_openable(Regime.HELDOUT)
    other.open_outer(Regime.SEEN, 'second opening', reopen_reason='a fixture rerun')

    records = log.records()
    assert [(r['event'], r['panel']) for r in records] == [
        ('open', opened),
        ('reopen', READ_NAMES[second][Regime.SEEN]),
    ]
    assert records[1]['detail']['reason'] == 'a fixture rerun'
    assert {r['fingerprint'] for r in records} == {opener.fingerprint}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_regressor_panel.py tests/unit/test_outcome_panel.py tests/unit/test_selection_log_guard.py -n auto -q
```
Expected: FAIL, `2 failed, 123 passed`. Both cases of
`test_an_opening_under_either_outcome_seals_the_split_for_both` fail with `Failed: DID NOT RAISE
<class 'naics_embedder.panels.outcome.SplitAlreadyOpenedError'>`. The extended
`test_openings_are_counted_per_held_out_draw` already passes: openings of other draws stayed apart
before the change, and must stay apart after it.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/panels/selection_log.py`, make two edits, in this order.

Replace:

```python
from typing import Any, Dict, List, Mapping, Optional
```

with:

```python
from typing import Any, Collection, Dict, List, Mapping, Optional, Union
```

Replace:

```python
    def openings(self, panel: str, fingerprint: str) -> List[Dict[str, Any]]:
        '''The recorded openings (``open`` and ``reopen``) of one panel's test split.'''

        opening = {SelectionEvent.OPEN.value, SelectionEvent.REOPEN.value}
        return [
            record for record in self.records() if record['event'] in opening
            and record['panel'] == panel and record['fingerprint'] == fingerprint
```

with:

```python
    def openings(self, panel: Union[str, Collection[str]], fingerprint: str) -> List[Dict[str, Any]]:
        '''
        The recorded openings (``open`` and ``reopen``) of one test split, oldest first.

        Args:
            panel: The name the split is opened under, or every such name: the regressor panel's
                two outcomes share each regime's outer set (Req 22).
            fingerprint: The split's fingerprint.
        '''

        names = {panel} if isinstance(panel, str) else set(panel)
        opening = {SelectionEvent.OPEN.value, SelectionEvent.REOPEN.value}
        return [
            record for record in self.records() if record['event'] in opening
            and record['panel'] in names and record['fingerprint'] == fingerprint
```

In `src/naics_embedder/panels/regressor.py`, make five edits, in this order.

Replace:

```python
- **Sealing.** Each regime's outer set is read only after a logged opening by the same panel
  object; a second opening of the same split (the same held-out draw, by fingerprint) needs a
  stated reason. Every read goes to the selection log, whose ``n_queries`` field counts rows here.
```

with:

```python
- **Sealing (Req 22).** Each regime's outer set is read only after a logged opening by the same
  panel object. An opening is recorded against the split, not the outcome: one under either
  outcome's read name counts for both, so a second opening of the same split (the same regime and
  held-out draw, by fingerprint) needs a stated reason. Every read goes to the selection log,
  whose ``n_queries`` field counts rows here.
```

Replace:

```python
    return tuple(names)

```

with:

```python
    return tuple(names)

def seal_names(regime: Regime) -> Tuple[str, ...]:
    '''
    Every read name an opening of the regime's outer set is logged under, one per outcome.

    The outer sets are rows, and 2025 employment underlies both outcomes' outer rows, so one
    opening of a regime's split counts for both (Req 22).
    '''

    return tuple(READ_NAMES[outcome][Regime(regime)] for outcome in RegressorOutcome)

```

Replace:

```python
            SplitAlreadyOpenedError: If the log already records an opening of this split and no
                ``reopen_reason`` is given.
        '''

        panel = self.read_names[Regime(regime)]
        prior = self.log.openings(panel, self.fingerprint)
        if prior and not (reopen_reason or '').strip():
            first = prior[0]
            raise SplitAlreadyOpenedError(
                f'the {panel} outer set was opened at {first["time"]} for {first["purpose"]!r}; '
                'opening it again needs reopen_reason'
```

with:

```python
            SplitAlreadyOpenedError: If the log already records an opening of this split, under
                either outcome's read name (``seal_names``), and no ``reopen_reason`` is given.
        '''

        regime = Regime(regime)
        prior = self.log.openings(seal_names(regime), self.fingerprint)
        if prior and not (reopen_reason or '').strip():
            first = prior[0]
            raise SplitAlreadyOpenedError(
                f'the {self.read_names[regime]} outer set was opened at {first["time"]} as '
                f'{first["panel"]} for {first["purpose"]!r}; opening it again needs reopen_reason'
```

Replace:

```python
            SplitAlreadyOpenedError: If the log already records an opening of this split and no
                ``reopen_reason`` is given (``require_openable``).
```

with:

```python
            SplitAlreadyOpenedError: If the log already records an opening of this split, under
                either outcome's read name, and no ``reopen_reason`` is given
                (``require_openable``).
```

Replace:

```python
        prior = self.log.openings(panel, self.fingerprint)
```

with:

```python
        prior = self.log.openings(seal_names(regime), self.fingerprint)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_regressor_panel.py tests/unit/test_outcome_panel.py tests/unit/test_selection_log_guard.py -n auto -q
```
Expected: PASS, `125 passed`.

- [ ] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/panels/selection_log.py src/naics_embedder/panels/regressor.py tests/unit/test_regressor_panel.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan13_site`, then `rm -rf /tmp/naics_plan13_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3115 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/selection_log.py src/naics_embedder/panels/regressor.py tests/unit/test_regressor_panel.py
git commit -m "feat(panels): one seal per regressor split across both outcomes (Req 22)"
```

### Task 5: Level mode only from the CLI and the sweep

`tools regressor-panel` and the sweep read an arm on every read: each read scores `embedding` and
`text_only`. Under growth that is a candidate's growth error, which Req 24 forbids before the
economic margin is recorded from a reference-only read: Stage 2 builds that read, and Stage 3 runs
it on real data. So both read level mode only (P6).

- `tools regressor-panel` refuses a config whose `outcome` is `growth`, with exit code 1, before
  it loads anything.
- `run_seed_sweep` refuses a growth `RegressorPanel` with a `ValueError`, before any read, run or
  store write. Without the guard, the sweep logs the arm's growth reads and only then fails D8's
  level-6 check.
- `docs/usage.md` describes the setting and both refusals, and the YAML's comment names them.

The Python API stays open, because Stage 2's reference-only read builds on `RegressorPanel`.
`data regressor-groups` and `tools text-only-table` read the same config, but neither builds a
panel.

**Files:**
- Modify: `src/naics_embedder/cli/commands/tools.py`, lines 48-49, 386-389 and 407-408
- Modify: `src/naics_embedder/decision/sweep.py`, lines 34-35 and 136-141
- Modify: `conf/data/regressor_panel.yaml`, line 17
- Modify: `docs/usage.md`, lines 212-215 and 287
- Test: `tests/unit/test_cli_commands.py`, lines 701-702
- Test: `tests/unit/test_decision_sweep.py`, lines 22-23 and 404-405

**Interfaces:**
- Consumes: `RegressorOutcome` (Task 2); `RegressorPanel.outcome` and
  `RegressorPanelConfig.outcome` (Task 3). The tests use `test_cli_commands.py`'s
  `fixture_panel` fixture, which records each panel load and returns `(log, loaded)`, and its
  `_regressor_arguments(tmp_path)`; and `test_decision_sweep.py`'s `panels`, `store`,
  `text_only` and `log` fixtures, with `_spec`, `SEEDS`, `SyntheticRunner` and `_signal`.
- Produces:
  - `tools regressor-panel` exits 1, printing "tools regressor-panel reads level mode only, and
    the regressor panel config sets outcome: growth. …", when the config sets `outcome: growth`.
    It loads no panel and logs nothing.
  - `run_seed_sweep` raises `ValueError('the sweep reads level mode only, not growth: …')` before
    any read.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_cli_commands.py`, make one edit.

Replace:

```python
@pytest.mark.unit
def test_regressor_panel_scores_validation_and_reports_undefined_cells(
```

with:

```python
@pytest.mark.unit
def test_regressor_panel_refuses_a_growth_config_before_anything_is_read(
    runner, tmp_path, fixture_panel, monkeypatch
):
    # Every read scores the arm and the text-only table, and on growth no candidate's error may
    # exist before the economic margin is recorded (Req 24)
    log, loaded = fixture_panel
    config = tmp_path / 'regressor_panel.yaml'
    config.write_text('outcome: growth\n')
    monkeypatch.setattr(tools_cli, 'REGRESSOR_PANEL_CONFIG', str(config))

    result = runner.invoke(tools_cli.app, _regressor_arguments(tmp_path))

    assert result.exit_code == 1
    assert 'reads level mode only' in ' '.join(result.output.split())
    assert loaded == []
    assert log.records() == []

@pytest.mark.unit
def test_regressor_panel_scores_validation_and_reports_undefined_cells(
```

In `tests/unit/test_decision_sweep.py`, make two edits, in this order.

Replace:

```python
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.regressor import DECISION_LEVEL, RegressorPanel
```

with:

```python
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import DECISION_LEVEL, RegressorPanel
```

Replace:

```python

def test_a_decision_over_swept_arms_adopts_the_informed_one(
```

with:

```python

def test_a_growth_regressor_panel_is_refused_before_any_read(
    tmp_path, regressor_rows, regressor_growth_rows, panels, store, text_only, log
):
    # D8 decides on the levels regimes, and a growth read of an arm before the economic margin is
    # recorded would break Req 24
    growth = RegressorPanel(
        regressor_growth_rows, HELDOUT_GROUPS, log, SETTINGS, outcome=RegressorOutcome.GROWTH
    )

    with pytest.raises(ValueError, match='reads level mode only'):
        run_seed_sweep(
            _spec('uninformed', dimension=3),
            SEEDS,
            SyntheticRunner(tmp_path / 'runs' / 'uninformed', False, _signal(regressor_rows)),
            text_only_table=text_only,
            store=store,
            purpose=PURPOSE,
            **{
                **panels, 'regressor_panel': growth
            },
        )
    assert log.records() == []
    assert not (tmp_path / 'runs').exists()
    assert not store.root.exists()

def test_a_decision_over_swept_arms_adopts_the_informed_one(
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_cli_commands.py tests/unit/test_decision_sweep.py -n auto -q`
Expected: FAIL, `2 failed, 64 passed`:
- `test_regressor_panel_refuses_a_growth_config_before_anything_is_read` fails with
  `assert 0 == 1`: the command ran;
- `test_a_growth_regressor_panel_is_refused_before_any_read` fails with `AssertionError: Regex
  pattern did not match.` The sweep raised later, `ValueError: 1,196 predictions are not level-6
  validation rows of ['regressor_seen', 'regressor_heldout']`, after it had logged the arm's
  growth reads.

- [ ] **Step 3: Write the implementation**

In `src/naics_embedder/cli/commands/tools.py`, make three edits, in this order.

Replace:

```python
from naics_embedder.panels.outcome import OutcomePanel, SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.regressor import (
```

with:

```python
from naics_embedder.panels.outcome import OutcomePanel, SealedSplitError, SplitAlreadyOpenedError
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import (
```

Replace:

```python
    regime's sealed outer set first, and both the opening and the read are logged.

    Example:
        Score an arm's table on the validation split of both regimes at six digits::
```

with:

```python
    regime's sealed outer set first, and both the opening and the read are logged.

    It reads level mode only. Every read scores the arm and the text-only table, and Req 24
    records the economic margin before any candidate's growth error exists, so a config whose
    ``outcome`` is ``growth`` is refused before anything is read.

    Example:
        Score an arm's table on the validation split of both regimes at six digits::
```

Replace:

```python
    cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
    # A repeated regime is scored once: opening it twice would be refused after the first read
```

with:

```python
    cfg = load_config(RegressorPanelConfig, REGRESSOR_PANEL_CONFIG)
    if RegressorOutcome(cfg.outcome) is not RegressorOutcome.LEVEL:
        console.print(
            '[bold red]tools regressor-panel reads level mode only, and the regressor panel '
            f'config sets outcome: {cfg.outcome}. Every read scores the arm and the text-only '
            "table, and no candidate's growth error may exist before the economic margin is "
            'recorded (Req 24).[/bold red]'
        )
        raise typer.Exit(code=1)
    # A repeated regime is scored once: opening it twice would be refused after the first read
```

In `src/naics_embedder/decision/sweep.py`, make two edits, in this order.

Replace:

```python
from naics_embedder.panels.outcome import OutcomePanel, QueryCodeEncoder
from naics_embedder.panels.regressor import (
```

with:

```python
from naics_embedder.panels.outcome import OutcomePanel, QueryCodeEncoder
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import (
```

Replace:

```python
        ValueError: If a seed repeats; if the text-only table, or a seed's table by its export
            provenance, was not built from the arm's backbone, revision, descriptions, summaries
            and window (D9); if a seed's encoder does not decode by the distance of the arm's
            geometry (Req 12); or if a seed's table width is not the arm spec's dimension.
    '''

```

with:

```python
        ValueError: If the regressor panel is not in level mode (D8; Req 24); if a seed repeats;
            if the text-only table, or a seed's table by its export provenance, was not built
            from the arm's backbone, revision, descriptions, summaries and window (D9); if a
            seed's encoder does not decode by the distance of the arm's geometry (Req 12); or if
            a seed's table width is not the arm spec's dimension.
    '''

    if regressor_panel.outcome is not RegressorOutcome.LEVEL:
        raise ValueError(
            f'the sweep reads level mode only, not {regressor_panel.outcome.value}: D8 decides on '
            'the levels regimes, and an arm read on growth before the economic margin is recorded '
            'would break Req 24'
        )
```

In `conf/data/regressor_panel.yaml`, make one edit.

Replace:

```yaml
# growth_heldout
```

with:

```yaml
# growth_heldout. tools regressor-panel and tools sweep read level mode only (Req 24)
```

In `docs/usage.md`, make two edits, in this order.

Replace:

````markdown
split has one repeat), with the row's outcome and the chosen penalty.

```bash
uv run naics-embedder tools regressor-panel --coordinates arm.parquet \
````

with:

````markdown
split has one repeat), with the row's outcome and the chosen penalty.

The panel's `outcome` setting in `conf/data/regressor_panel.yaml` (Req 21) is `level`, the panel
above, or `growth`: log E(t + 1) − log E(t) on the same rows, read as `growth_seen` and
`growth_heldout`. One opening of a regime's outer set counts for both outcomes (Req 22). This
command reads level mode only. Every read scores the arm and the text-only table, and Req 24
records the economic margin before any candidate's growth error exists, so a config set to
`growth` is refused before anything is read.

```bash
uv run naics-embedder tools regressor-panel --coordinates arm.parquet \
````

Replace:

```markdown
comparator.
```

with:

```markdown
comparator. The sweep reads the regressor panel in level mode only (D8): a regressor config set to
`outcome: growth` is refused before any read.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
uv run pytest tests/unit/test_cli_commands.py tests/unit/test_decision_sweep.py tests/unit/test_config.py -n auto -q
```
Expected: PASS, `285 passed`. `tests/unit/test_config.py` joins the run because the YAML's
comment changed, and its tests compare that YAML with the defaults.

- [ ] **Step 5: Format, lint, build the docs and run the full suite**

Run:

```bash
./scripts/format_code.sh src/naics_embedder/cli/commands/tools.py src/naics_embedder/decision/sweep.py tests/unit/test_cli_commands.py tests/unit/test_decision_sweep.py
```
Expected: no file changes.

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `uv run mkdocs build --strict -d /tmp/naics_plan13_site`, then `rm -rf /tmp/naics_plan13_site`
Expected: the build succeeds with no warning.

Run: `uv run pytest -n auto -q`
Expected: `3117 passed, 2 skipped`.

- [ ] **Step 6: Commit**

```bash
git add src/naics_embedder/cli/commands/tools.py src/naics_embedder/decision/sweep.py conf/data/regressor_panel.yaml docs/usage.md tests/unit/test_cli_commands.py tests/unit/test_decision_sweep.py
git commit -m "feat(tools): read the regressor panel in level mode only from the CLI and the sweep (Req 24)"
```

### Task 6: Documentation

The API page, CLAUDE.md, the README and the test README describe the outcome setting and the
shared seal, and state the new count (P7).

**Files:**
- Modify: `docs/api/regressor_panel.md`, lines 3-4
- Modify: `CLAUDE.md`, lines 229-230, 300 and 483-484
- Modify: `README.md`, line 164
- Modify: `tests/README.md`, lines 5 and 35-36

**Interfaces:**
- Consumes: the names Tasks 1–5 produce.
- Produces: documentation only.

- [ ] **Step 1: Write the documentation**

In `docs/api/regressor_panel.md`, make one edit.

Replace:

```markdown
Arms' coordinates as regressors for next-year QCEW employment, in two sealed regimes (roadmap
Stage 3).
```

with:

```markdown
Arms' coordinates as regressors for QCEW employment, in two sealed regimes (roadmap Stage 3).
The `outcome` setting picks next-year log employment (`level`, the default) or its log change from
year t (`growth`, Req 21), on the same rows, partition and folds. One opening of a regime's outer
set counts for both outcomes (Req 22).
```

In `CLAUDE.md`, make three edits, in this order.

Replace:

```markdown

### Remote Workflow and Phase 2 Boundary
```

with:

```markdown

The regressor panel's `outcome` (`conf/data/regressor_panel.yaml`, Req 21) is `level` by default,
the levels history read as `regressor_seen` and `regressor_heldout`. `growth` reads
log E(t+1) − log E(t) on the same rows, partition and folds, as `growth_seen` and
`growth_heldout`. One opening of a regime's outer set counts for both outcomes (Req 22).
`tools regressor-panel` and `tools sweep` read level mode only, so no candidate's growth error
exists before the economic margin (Req 24). The fixture panel pins level mode's rows, data
fingerprints, folds and scores at d61f49b (`test_regressor_panel.py`).

### Remote Workflow and Phase 2 Boundary
```

Replace:

```markdown
The current suite collects **3,099 tests**. Actual skip counts depend on local data and hardware
```

with:

```markdown
The current suite collects **3,119 tests**. Actual skip counts depend on local data and hardware
```

Replace:

```markdown
  `test_visualize_metrics.py`: campaign preflight, radius checks and durable health artifacts.
- `integration/test_reference_training.py`: tiny-backbone Trainer runs, earliest-best selection,
```

with:

```markdown
  `test_visualize_metrics.py`: campaign preflight, radius checks and durable health artifacts.
- `test_regressor_panel.py`, `test_regressor_qcew_rows.py`: level mode pinned at d61f49b, the
  growth outcome on the same rows and folds, and one seal per regime's split (Reqs 21–22).
- `integration/test_reference_training.py`: tiny-backbone Trainer runs, earliest-best selection,
```

In `README.md`, make one edit.

Replace:

```markdown
The suite collects 3,099 tests in 100 unit files and two integration files. Actual skip counts
```

with:

```markdown
The suite collects 3,119 tests in 100 unit files and two integration files. Actual skip counts
```

In `tests/README.md`, make two edits, in this order.

Replace:

```markdown
The suite contains **100 unit** test files and **2 integration** files, with **3,099 collected
```

with:

```markdown
The suite contains **100 unit** test files and **2 integration** files, with **3,119 collected
```

Replace:

```markdown
| Selection-log guard | `test_selection_log_guard.py`: fail-closed refusals before appending reads or openings |
| Training and contracts | `test_cli_training.py`, `test_utils_training.py`, `test_checkpoint_contract.py`, `test_export.py`, `test_arm_encoder.py`: objective refusal, settings/directory/seed guards, saved constructor controls, CPU/MPS float64 callback transport and export/read provenance |
```

with:

```markdown
| Selection-log guard | `test_selection_log_guard.py`: fail-closed refusals before appending reads or openings |
| Regressor panel | `test_regressor_panel.py`, `test_regressor_qcew_rows.py`: level mode pinned at d61f49b, the growth outcome on the same rows and folds, and one seal per regime's split (Reqs 21–22) |
| Training and contracts | `test_cli_training.py`, `test_utils_training.py`, `test_checkpoint_contract.py`, `test_export.py`, `test_arm_encoder.py`: objective refusal, settings/directory/seed guards, saved constructor controls, CPU/MPS float64 callback transport and export/read provenance |
```

- [ ] **Step 2: Check the stated count and build the docs**

Run: `uv run pytest --collect-only -q -q | tail -1`
Expected: `3119 tests collected`, followed by a time, matching the count the three files now
state.

Run: `uv run mkdocs build --strict -d /tmp/naics_plan13_site`, then `rm -rf /tmp/naics_plan13_site`
Expected: the build succeeds with no warning.

- [ ] **Step 3: Commit**

```bash
git add docs/api/regressor_panel.md CLAUDE.md README.md tests/README.md
git commit -m "docs: record the growth outcome and one seal per split"
```

## Final verification (controller, inline)

Run every check before the final review, and paste each output into the ledger.

- [ ] **Step 1: The suite under both CI versions**

Run: `uv run pytest -n auto -q`
Expected: `3117 passed, 2 skipped`.

Run:

```bash
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics_plan13_py310 uv run --locked pytest -n auto -q
```
Expected:
- `3117 passed, 2 skipped`, the same counts as under 3.12 (measured at plan time);
- a few hundred extra "encountered in matmul" RuntimeWarnings, from numpy 2.2 with Accelerate on
  3.10, which are not a failure.

Run: `rm -rf /tmp/naics_plan13_py310`

- [ ] **Step 2: Lint, format and docs**

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0, with no file listed.

Run: `uv run mkdocs build --strict -d /tmp/naics_plan13_site`, then `rm -rf /tmp/naics_plan13_site`
Expected: the build succeeds with no warning.

- [ ] **Step 3: The branch**

Run: `git log --oneline origin/main..HEAD`
Expected: the six task commits, newest first, and no private commit:

```text
docs: record the growth outcome and one seal per split
feat(tools): read the regressor panel in level mode only from the CLI and the sweep (Req 24)
feat(panels): one seal per regressor split across both outcomes (Req 22)
feat(panels): read the regressor panel's growth outcome under its own names (Req 21)
feat(panels): add the growth outcome to the regressor panel's rows (Req 21)
test(panels): pin level mode's regressor values before the growth outcome
```

Run: `git diff --stat origin/main...HEAD | tail -1`
Expected: `18 files changed, 720 insertions(+), 51 deletions(-)`.

Run: `git diff --name-only origin/main...HEAD -- uv.lock conf/config.yaml specs`
Expected: no output.

Run: `git diff origin/main...HEAD -- conf/data/regressor_panel.yaml`
Expected: five added lines, and no other change: the three-line `outcome` comment,
`outcome: level` and a blank line. No pin changed.

- [ ] **Step 4: The spec's Verification items**

Record in the ledger which test carries each of this stage's items:
- **Levels unchanged:** Task 1's five `test_level_…` tests in `tests/unit/test_regressor_panel.py`.
- **Growth rows:** the three growth tests in `tests/unit/test_regressor_qcew_rows.py`, and
  `test_growth_keeps_level_modes_partition_and_draw`.
- **One seal per split:** `test_an_opening_under_either_outcome_seals_the_split_for_both`, both
  cases.
- **Checks:** Steps 1 and 2.

- [ ] **Step 5: The final review**

Dispatch the code-reviewer agent, which is pinned to Opus, on the whole branch, with:
- `git diff origin/main...HEAD`;
- this plan;
- the spec's Reqs 20–22 and Req 24, and the roadmap's Stage 1 entry.

Also run Codex beside it, from the worktree root:

```bash
codex exec review --base origin/main -m gpt-6-astra
```

- `--base origin/main` makes it review the branch's commits. Without it, Codex reviews only
  uncommitted changes, and there are none at this point.
- `--base` takes no custom prompt: Codex refuses the combination.
- The config's default model fails for this account, so always pass `-m gpt-6-astra`, and never
  edit `~/.codex/config.toml`.

Then, for each finding:
- fix it, with a test where it is behavior, and rerun Steps 1–3; or
- triage it as deferred, which **Plan completion**'s gate handles.

## Plan completion (controller, inline)

Run this after **Final verification**, once the review findings are resolved, and before
**Integration**. It is writing-plans' Plan Completion Protocol, with this plan's edits written
out. Its commits land on `claude/stage-1-growth-outcome`, so the PR carries them.

- [ ] **Step 1: Check for parallel sessions**

`specs/naics-representation-split-roadmap.md` and `specs/deferred_items.md` are shared by every
session.

Run: `git worktree list`
- Worktrees under `~/Projects/copilot-worktrees/` are review tooling, not sessions.
- If another worktree belongs to a running session, or the user has mentioned one, hold the
  shared edits (Steps 4 and 6) and give the user the exact text of each.

- [ ] **Step 2: The resolve-before-defer gate**

Collect the leftovers: plan steps skipped or descoped, and review findings not fixed. Partition
them:
- those that need the user's input: ask now, as one batched set of questions;
- those an answer unblocks: implement them now, then restart this protocol;
- everything else: defer.

The partition is final only once every question is answered.

- [ ] **Step 3: Mark up this plan**

- Tick every completed step (`- [x]`).
- Under each step that deviated, add a one-line `> Deviation: …` note.
- Annotate each skipped step with `> Skipped: <why> → deferred`.
- Add the status line under the title: `**Status: COMPLETE (<date>)** — executed via <skill>;
  deferred items in specs/deferred_items.md`, or `…; nothing deferred` when the gate deferred
  nothing.

- [ ] **Step 4: The deferred items**

In `specs/deferred_items.md`:
- Tick any earlier item this plan implemented, as `- [x] … → done in plan 13`. None was expected
  at plan time.
- Append this plan's deferred items, if the gate deferred any, as a
  `## 13-growth-outcome-and-seals — <date>` section in the file's item schema. Skip the section
  if nothing was deferred.

- [ ] **Step 5: The handoff to later stages**

Add this section at the end of this plan, after **Integration**. It was drafted at plan time.
Check each bullet against what shipped: the deviation notes, the ledger and the final code.
Rewrite any bullet that execution changed, and keep a bullet for any interim name that execution
reverted, saying it is gone.

```markdown
## Handoff to later stages

- **Stage 2 (economic decision tooling)** consumes, as built:
  - **Read names.** `READ_NAMES[RegressorOutcome.GROWTH]` in `panels/regressor.py` gives
    `growth_seen` and `growth_heldout`. `PANEL_NAMES` stays the levels' names, which
    `decision/scores.py` imports for D8's decision; the economic decision's panels are not in it.
  - **The panel.** `RegressorPanel(rows_by_level, heldout_groups, log, settings,
    outcome=RegressorOutcome.GROWTH)`, `RegressorPanel.from_sources(..., outcome=...)`, or
    `load_regressor_panel(cfg, codebook)` with `cfg.outcome == 'growth'`. `RegressorOutcome` is
    in `panels/qcew_rows.py`.
  - **The data fingerprint.** `RegressorPanel.data_fingerprint(level)` hashes the rows, outcome
    included, so a growth panel's differs from level mode's at every level. The draw
    fingerprint, `RegressorPanel.fingerprint`, is the same under both outcomes (`deddfd4c…` on
    the real panel).
  - **Prediction columns.** `PREDICTION_COLUMNS` is unchanged. A growth read's frame carries its
    read name in `panel`.
  - **The seal.** `seal_names(regime)`, and `SelectionLog.openings(names, fingerprint)` over a
    collection of names.
  - **Refusals to lift or route around.** `tools regressor-panel` refuses `outcome: growth`, and
    `run_seed_sweep` refuses a growth panel. Each scores `embedding` and `text_only` on every
    read, so the reference-only read cannot go through either as they stand (Req 24).
  - **Traps.** Two of the level pins (`CAPTURED_HERE`) run only on macOS arm64, and Linux CI
    skips them. `decide` pairs reads only on bit-identical regressor rows, so growth reads pair
    only within one platform and one `uv.lock`.
- **Stage 3 (the growth margin read)** consumes the growth panel through Stage 2's read.
  - The committed `conf/data/regressor_panel.yaml` keeps `outcome: level`.
  - No growth read of real data happened in Stage 1, and no real split was opened.
  - **Existing openings.** Under Req 22, a real `open` or `reopen` of `regressor_seen` or
    `regressor_heldout` under the draw `deddfd4c…` now also seals growth's split for that
    regime. Stage 1 never read the real selection log, and Stage 3 is the first stage to.
    `SelectionLog.openings(seal_names(regime), fingerprint)` lists any such opening.
```

- [ ] **Step 6: The roadmap stamp**

In `specs/naics-representation-split-roadmap.md`, Stage 1's entry:
- tick it: `- [ ] Stage 1: Growth outcome and one seal per split` becomes
  `- [x] Stage 1: Growth outcome and one seal per split`;
- after its `ROUTING: writing-plans` line, add the stamp, in the roadmap's own form (its
  "Stage-spec stamp" section), indented like the entry's other lines:

```text
      > Stage 1: COMPLETE (<date>) — implemented by plan 13
      > (specs/plans/completed/13-growth-outcome-and-seals.md). Its Handoff section names what
      > Stages 2 and 3 consume.
      > Next: resume the roadmap.
```

The stamp asks completion to re-validate later stages. That re-validation is the Handoff section
(Step 5); amending Stages 2–3's entries is the next roadmap resume's job.

- [ ] **Step 7: Backlog triage**

Run:

```bash
uv run --no-project --python 3.13 python /Users/lowell/.claude/skills/writing-plans/scripts/deferred_stats.py
```
Report its summary line: the open count, the closure rate and the aged tail. With 20 or more open
items or any aged tail, run steps 1–4 of the skill's Triage rubric and present the proposal. It
is read-only; `/deferred` acts on a selection.

- [ ] **Step 8: Retire the plan**

Run:

```bash
git mv specs/plans/13-growth-outcome-and-seals.md specs/plans/completed/13-growth-outcome-and-seals.md
```

This plan has no relative links to re-point. The spec stays in `specs/`: Stages 2–3 and the
candidates' and text→code specs cite it, and this plan implements one stage of it.

```bash
git add specs/plans/completed/13-growth-outcome-and-seals.md specs/naics-representation-split-roadmap.md specs/deferred_items.md
git commit -m "chore(specs): retire plan 13"
```

## Integration (controller, inline)

This is finishing-a-development-branch, with this plan's PR written out.

- [ ] **Step 1: Open the PR, with the user's go-ahead**

Run: `git log --oneline origin/main..HEAD`
Expected: **Final verification** Step 3's six commits, any review fixes, and `chore(specs):
retire plan 13`; no private commit.

Ask the user before pushing. Then run `git push -u origin claude/stage-1-growth-outcome`, never a
bare `git push`, and open the PR against `main`. Its description carries:
- what the stage changes: the plan's Goal and Architecture, shortened;
- the recorded deviations, one line each, with P6's refusals flagged for the reviewer;
- the suite counts: `3117 passed, 2 skipped` locally under both Pythons, and the full suite's
  expected summary line in each CI test job, `3106 passed, 13 skipped`.

End the description with the session's PR attribution line.

- [ ] **Step 2: Wait for CI, then merge**

GitHub auto-merge is off, and main has no required checks.
- Watch `gh pr checks <number>` until the lint, docs and both test jobs pass.
- In each test job, the full suite's summary line should read `3106 passed, 13 skipped`. The
  47-test remote-transport step before it is unchanged. If a Task 1 pin not marked
  `CAPTURED_HERE` fails there, stop and ask (**Stop-and-ask conditions**).
- The user merges. If the user asks you to merge, run
  `gh pr merge <number> --merge --match-head-commit <sha>`.

Before pushing a follow-up, check that the PR is still open (`gh pr view <number> --json state`).
If the user merged meanwhile, land the follow-up as a new PR from `origin/main`.

- [ ] **Step 3: Keep the ledger, then remove the worktree**

Removing the worktree deletes every ignored file in it. Keep the execution ledger first.

Run: `ls /Users/lowell/Projects/naics-embedder/logs/plan13_worktree`
Expected: "No such file or directory". If it exists, stop and ask.

Run: `mkdir /Users/lowell/Projects/naics-embedder/logs/plan13_worktree`, then
`cp -c logs/plan13_execution.md /Users/lowell/Projects/naics-embedder/logs/plan13_worktree/`

Then finish the branch: detach onto `origin/main`, delete the local branch, and remove the
worktree, or archive the app session that owns it.
