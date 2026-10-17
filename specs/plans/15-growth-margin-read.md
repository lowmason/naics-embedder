# Growth Margin Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-representation-split-roadmap.md, Stage 3 — on plan completion, tick the
> stage and re-validate later stages against what shipped.

**Goal:** Build Stage 3 of the representation-split roadmap. With the user's approval at the time,
run the reference-only growth read, `tools growth-margins`, once on real QCEW data on the Mac.
It records δ = 0.05 × the reference `covariates+ancestors`'s validation MSE for `growth_seen` and
`growth_heldout`, before any economic candidate exists. Store the margin record outside any
worktree and write the finding that the candidates' spec reads.

**Architecture:** No code changes. The plan runs the command plan 14 shipped, then checks and
records what it did.

- **Pre-flight.** Cut the finding branch from `origin/main` in the execution worktree. Check for
  parallel work on the read's code and configs. Run the level pins that hold only in the
  environment they were captured in.
- **Ready the main checkout (Task 1).** Fast-forward local `main` to `origin/main` with a guard,
  sync its `.venv` to `uv.lock`, record the environment beyond what the record names, and confirm
  the record's path is free and the store is writable. Nothing here reads real data.
- **Gate A: the inputs and the log (Task 2).** With the user's approval to start the stage, hash
  the four QCEW slices and check the codebook's codes against their pin, without loading the
  panel. Then read the main checkout's selection log: its size, a tally, any record that is not
  a validation read, each regime's existing openings, and the growth reads, of which there must
  be none.
- **Gate B: the margin read (Task 3).** Show the user what Tasks 1–2 found and the exact command,
  and ask for a go. Re-check the log right before the read, then run it once from the main
  checkout's root with every path absolute. A failure is followed by a log check before anything
  else, and the read is never re-run on the strength of a failure alone.
- **Verify (Task 4).** Read only: check the record as `tools decide` will, its stored scores
  against a rescore, its environment against the lock, and the log's growth by exactly the read's
  two records after an unchanged prefix.
- **The finding (Task 5).** `specs/findings/growth-margins.md`: the inputs, the commit, the
  environment, δ for each panel, the record and its sha256, and the log's state before and after.

**Tech Stack:** uv 0.13 (`uv run --locked --no-sync`), the `naics-embedder` CLI, Python 3.12
scripts over the package's own APIs (`SelectionLog`, `read_record`, `check_relative_margins`),
`shasum`, git and gh.

**Spec:** `specs/naics-representation-split.md` (Req 24, Margin; Verification "Margin first,
reference only" and "Outer sets closed"; Rollout note, item 4; U13), through the roadmap's
Stage 3 entry. Read both, and plan 14's Handoff
(`specs/plans/completed/14-economic-decision-tooling.md:6483`) and plan 13's
(`specs/plans/completed/13-growth-outcome-and-seals.md:3020`).

## Global Constraints

Every task's requirements include this section.

**Locators.** U1–U16 are the split spec's user rulings. Bare D1–D11 are the embedding roadmap's
decisions (`specs/naics-embedding-roadmap.md`). O1–O5 are the owner's decisions and P1–P8 this
plan's (`## Decisions`). MAIN is the main checkout, `/Users/lowell/Projects/naics-embedder`.

### The stage entry (`specs/naics-representation-split-roadmap.md` at 8aed0a6), verbatim

> - [ ] Stage 3: The growth margin read
> Objective: With the user's approval at the time, run the reference-only growth read on real
> data on the Mac and record δ for both growth panels before any candidate exists.
> Spec: Req 24 (Margin); Verification "Margin first, reference only", "Outer sets closed";
> Rollout note, item 4; U13.
> Gap closed: Req 24 (the recorded margin).
> Consumes: Stage 2's reference-only read and margin record, and Stage 1's growth panel, merged
> to origin/main and synced into local main, where the read runs. `qcew_dir` keeps its public
> default `~/Downloads/Data/QCEW`, where the read finds the QCEW slices and verifies each
> against `qcew_sha256` in `conf/data/regressor_panel.yaml`; no held commit points it there
> (audit row 1, 2026-10-16). That file keeps `outcome: level`, and no real growth read or
> opening has happened (plan 13, Handoff:
> `specs/plans/completed/13-growth-outcome-and-seals.md:3020`; Resolved at resume,
> 2026-10-12). The read is `tools growth-margins`, run from the main checkout's root or with
> absolute `--lock` and `--log`: both resolve against the working directory, and a worktree's
> log goes with the worktree (plan 14, Handoff:
> `specs/plans/completed/14-economic-decision-tooling.md:6483`; Resolved at resume,
> 2026-10-16). Its `--codebook` is bundle 301cce28's, whose codes keep the pinned
> `codebook_codes_sha256` (`specs/naics-embedding-roadmap.md:116-118`), though the pin's
> comment still names bundle 18403d29 (found at resume; Resolved at resume, 2026-10-16). The
> main checkout's selection log (`logs/selection_log.jsonl`, gitignored and append-only), which
> this stage is the first to read: an opening of either level regime under `deddfd4c…` already
> seals that regime's growth split, and `SelectionLog.openings(seal_names(regime), fingerprint)`
> lists any (plan 13, Handoff; Resolved at resume, 2026-10-12); margin first is not checked
> against the log, so it must hold no `growth_seen` or `growth_heldout` read before this one
> (plan 14, Handoff; Resolved at resume, 2026-10-16). An artifact store outside any worktree,
> as embedding roadmap Stages 7–8 kept theirs; the user's approval, asked when the stage starts.
> Produces: The economic margin record, holding δ for `growth_seen` and `growth_heldout` and,
> in its read's environment, the platform, Python and `uv.lock` sha256 it was read under.
> Every candidate read must share them, since `decide` pairs only bit-identical reference
> predictions; bit identity was verified on fixtures only, and BLAS threading is the first
> suspect if real reads fail to pair (plan 14, Handoff; Resolved at resume, 2026-10-16). The
> command writes the record once, to `--output`, and the read's predictions and scores to
> `--store`; it neither stores the record nor prints its sha256, so this stage stores it
> outside any worktree and records its sha256 with `shasum -a 256` (plan 14, Handoff;
> Resolved at resume, 2026-10-16). A written finding that records it with the read's
> selection-log records. The candidates' spec reads both.
> Exit: The margin record exists with r = 0.05 and one δ per growth panel; the selection log
> gained only the two growth validation reads of one run, which log the comparators plan
> 14's Handoff lists per regime and neither `embedding` nor `text_only`; nothing refuses a
> second run, so the margins are fixed once (plan 14, Handoff; Resolved at resume,
> 2026-10-16); no outer set was opened; the finding cites the record's sha256, the platform
> and the lock, and the origin/main commit the read ran at, from which a candidate can
> rebuild that lock after `uv.lock` moves, as it did after Stage 2 (639ce07; found at resume;
> Resolved at resume, 2026-10-16), states that the log held no growth read before the margin
> read (plan 14, Handoff; Resolved at resume, 2026-10-16), and lists the existing openings
> that check found, or says it found none (plan 13, Handoff; Resolved at resume, 2026-10-12).
> ROUTING: writing-plans

### The spec (`specs/naics-representation-split.md` at 0a39042), verbatim

From Req 24:

> - **Margin.** δ = 0.05 × the reference's validation MSE on that panel (U13), where that MSE is
>   the statistic above computed for `covariates+ancestors`. δ is fixed and recorded from a
>   reference-only read before any candidate is read.
>   - The margin read scores only comparators that read no candidate table: `covariates`, `one_hot`,
>     `ancestors` and their covariate pairs. It computes neither `embedding` nor `text_only`, so no
>     candidate's growth error exists before δ is recorded.

From Verification:

> - [ ] **Margin first, reference only.**
>   - The margin read logs only comparators that read no candidate.
>   - The margin record's time precedes every candidate read.
>   - `decide` refuses a margin record that does not.

> - [ ] **Outer sets closed.** The selection log shows validation reads only. No outer set is opened
>       before the economic artifact's own planned Stage 12 opening.

From the Rollout note:

> 4. **The margin read on real data (Req 24).** It runs on the Mac, is logged to the selection log,
>    and needs the user's approval at the time.

U13: "Does the decision section look right: which margin r?" Ruling: "r = 0.05 (Recommended)".

### Project rules

- **Real data only behind the gates.** Nothing reads the QCEW slices, the bundle's codebook or
  the main checkout's selection log before Gate A's approval (Task 2 Step 1). Nothing runs
  `tools growth-margins` before Gate B's go (Task 3 Step 2). Each approval comes from the user in
  chat, at that step, for that step; an earlier approval does not carry over.
- **One read.** `tools growth-margins` runs once. Nothing refuses a second run, and each run
  appends two more growth reads. Never re-run it, whatever happens, without the user's ruling.
- **The selection log is append-only.** Never edit, truncate, move, copy into the repository or
  delete `MAIN/logs/selection_log.jsonl`. Only the read appends to it.
- **The main checkout.** Never check out a branch there and never edit its tracked files. The
  only writes there are Task 1's fast-forward and `uv sync`, the read's own outputs (the log's
  two records and `logs/tools_growth_margins.log`, both gitignored) and Integration's ledger
  copy. Local `main` follows `origin/main` by fast-forward only (Task 1 Step 1's guard).
- **Absolute paths.** Pass every path argument as an absolute path, never with `~`. `write_record`
  and `read_codebook_codes` do not expand `~`, so a `--output=~/…` that the shell leaves alone
  would write a literal `~` directory. `load_config` falls back to the defaults, with only a
  warning, when it cannot find a config file, so every command that loads `conf/` runs from
  MAIN's root, or names the file absolutely.
- **The lock.** Never change `uv.lock`, and never run `uv lock`. Run Python with
  `uv run --locked --no-sync` after Task 1's `uv sync --locked`, as Stages 7–8 did.
- **Git.** Commit only on `claude/stage-3-growth-margins`, cut from `origin/main`. Never push
  without the user's go-ahead, never use `--no-verify`, and never push `main` or a `refs/held/*`
  ref. If a git command stalls, rerun it as `git -c core.fsmonitor=false …`: git's fsmonitor
  daemon has hung `git add` and `git status` here. macOS has no `timeout` command.
- **Commits** end with `Co-Authored-By: <the executing model> <noreply@anthropic.com>`, and the PR
  body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Markdown.** Lines stay within 100 characters outside code blocks and tables. Headings have one
  blank line above and below, lists indent by 2 spaces, and no more than one blank line in a row.
- **Bash tool.** It runs zsh. Brace variables (`${FILE}`); quote globs and `=`-leading words;
  gate a check on `check; echo "exit=$?"`, never on a pipe's tail. Measure line lengths with
  Python, not BSD `awk`. The tool can reset its working directory between calls, so prefix every
  command with `cd <dir> &&`, and run `pwd` before every commit.

## Decisions

The owner's decisions, from the brief this plan was written from and the roadmap resume it cites.
They carry O-numbers, because bare D-numbers are the embedding roadmap's. Do not re-triage them.

- **O1** (owner, brief, 2026-10-16): `--codebook` is bundle 301cce28's `naics_codebook.parquet`,
  whose codes keep the pinned `codebook_codes_sha256`. Applied in: Task 2 Step 3; Task 3 Step 4.
- **O2** (owner, brief, 2026-10-16): the finding records the `origin/main` commit the read ran at,
  because `uv.lock` moved after Stage 2 (639ce07), so a candidate can rebuild that lock. Applied
  in: Task 3 Step 1; Task 5.
- **O3** (owner, brief, 2026-10-16): the real read runs on the Mac, from MAIN's root, with the
  user's approval asked when execution reaches it. Applied in: Task 3 (Gate B).
- **O4** (owner, brief, 2026-10-16; a standing rule): planning ran on Opus; execution runs on
  Sonnet in a fresh session, with this file as the handoff. Applied in: Execution.
- **O5** (owner, brief, 2026-10-16; a standing rule): MAIN's `logs/selection_log.jsonl` is
  append-only, and local `main` follows `origin/main` by guarded fast-forward syncs. Applied in:
  Global Constraints; Task 1 Step 1.

This plan's choices, for the owner's review at the plan PR:

- **P1:** two gates. Gate A, at the stage's start (the roadmap's "asked when the stage starts"),
  covers the read-only input and log checks. Gate B (O3) covers the read, and shows Gate A's
  findings first. Applied in: Tasks 2–3.
- **P2:** the read's arguments. `--name growth-margins`; `--store /Users/lowell/naics-artifacts`,
  Stages 7–8's store; `--output /Users/lowell/naics-artifacts/records/economic/growth_margins.json`,
  as `docs/usage.md` shows; `--purpose 'economic margin read: split roadmap Stage 3, plan 15
  (Req 24)'`; `--log` and `--lock` absolute, MAIN's. Applied in: Task 3 Step 4.
- **P3:** the read runs as `HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync`: nothing
  downloads, the environment is the one Task 1 synced, and Rich does not fold the output that
  `/tmp/naics_plan15/read.txt` captures. Applied in: Task 3 Step 4.
- **P4:** the read runs with the BLAS thread variables unset, as at plan time, and the finding
  says so. The record names the platform, Python and lock only, and threading is the first
  suspect when a real pairing fails. Applied in: Task 1 Step 3; Task 5.
- **P5:** Task 4 also checks the stored scores against a rescore, which `check_relative_margins`
  does not (the "Economic stored-scores check" item under `## 14-economic-decision-tooling` in
  `specs/deferred_items.md`). It is a one-off check of this record, not that item's fix, which
  stays open. Applied in: Task 4 Step 2.
- **P6:** the finding copies the margin record and the read's two log records verbatim, and the
  pre-read openings, if any. Applied in: Task 5.
- **P7:** no edit outside `specs/`. The stale comment beside `codebook_codes_sha256` stays: it is
  not in Stage 3's Produces or Exit, and the read loads that file. Applied in: File structure.
- **P8:** inline execution (executing-plans). The read needs the user in the loop at two gates,
  and a subagent cannot ask them. Applied in: Execution.

---

## Workspace

- **Execution ledger:** `logs/plan15_execution.md` in the execution worktree, ignored by git.
  Write every command, its output and any deviation there. Integration Step 4 copies it to MAIN's
  `logs/`.
- **Worktree:** the app worktree the execution session runs in. Run every command from its root,
  unless the step names MAIN. Call its path WORKTREE, as `git rev-parse --show-toplevel` prints it.
- **Branch:** `claude/stage-3-growth-margins`, cut at `origin/main` once this plan has merged
  (Pre-flight Step 1). An app worktree is cut from local `main`, which can be stale, so the branch
  is always cut from `origin/main`.
- **Main checkout (MAIN):** `/Users/lowell/Projects/naics-embedder`. At plan time it was on `main`
  at d4f8ddc, clean, one merge behind `origin/main` (8aed0a6). Its gitignored `conf/local.yaml`
  names bundle 301cce28's manifest; `tools growth-margins` does not read it.
- **Real inputs (read only, after Gate A):**
  - The QCEW slices, in `qcew_dir`'s default, `/Users/lowell/Downloads/Data/QCEW`:
    `2022_US000_annual.csv` through `2025_US000_annual.csv`, pinned in
    `conf/data/regressor_panel.yaml`.
  - Bundle 301cce28's codebook and manifest, in
    `MAIN/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/`.
    The manifest's sha256 was `68545c669c4fdbc0de9958df6270ec7426994159da0bc119a0f1fd49177ef40f`
    at Stage 7 (`specs/findings/reference-configuration.md`, Sources).
  - The selection log, `MAIN/logs/selection_log.jsonl`.
- **Outputs:**
  - The store, `/Users/lowell/naics-artifacts`, which holds Stages 7–8's records and objects.
    At plan time `records/` held only `stage7/` and `stage8/`.
  - The margin record, `/Users/lowell/naics-artifacts/records/economic/growth_margins.json`,
    which must not exist before the read.
  - Scratch: `/tmp/naics_plan15/`, for this plan's scripts and command outputs. Integration
    Step 4 deletes it.
- **Plan-time values**, for the Expected lines. If one differs at execution, record it and follow
  the step's instruction:
  - `uv.lock` at 8aed0a6:
    `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948`.
  - MAIN's environment: `macOS-26.6.2-arm64-arm-64bit`, Python 3.12.12, numpy 2.3.4,
    polars 1.35.1, BLAS `accelerate`, no thread variable set.

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- The user declines Gate A or Gate B, or answers anything but a clear yes.
- Pre-flight finds an open PR, or a commit on `origin/main` since this plan merged, that touches
  `src/naics_embedder/panels/`, `src/naics_embedder/decision/`, `conf/data/regressor_panel.yaml`,
  `conf/data/decision.yaml`, `uv.lock`, the split roadmap or `specs/findings/`.
- The level pins fail or skip (Pre-flight Step 3).
- Task 1's guard fails: MAIN is not on `main`, not clean, or has commits `origin/main` lacks.
- `uv.lock` differs from the plan-time value, or `uv sync --locked --check` fails after the sync.
- The record's path exists, or the store is not writable.
- A QCEW slice's or the codebook's check fails (Task 2).
- The log check exits non-zero: a growth read already exists, or a record lacks a field (Task 2).
- The log changed between Task 2 and Task 3 Step 3, or MAIN's HEAD is not `origin/main` there.
- `tools growth-margins` exits non-zero (Task 3 Step 5 says what to check first).
- Any check in Task 4 fails.
- A step would do anything the Project rules forbid.

## File structure

| Action | Path | Task | Responsibility |
|---|---|---|---|
| Create | `specs/findings/growth-margins.md` | 5 | The finding: inputs, commit, environment, δ, the record's sha256, the log before and after |
| Modify | `specs/naics-representation-split-roadmap.md` | Plan completion | Tick Stage 3 and add its stamp |
| Modify | `specs/plans/15-growth-margin-read.md` | Plan completion | Markup, status line and Handoff; then `git mv` to `specs/plans/completed/` |
| Read | `specs/deferred_items.md` | Plan completion | The ticking pass: no open item names Stage 3, so nothing is ticked; append only if the gate defers |

No source, test, config, docs or lock file changes. `CLAUDE.md` is not edited (P7).

## Pre-flight (controller, inline, before Task 1)

- [ ] **Step 1: Confirm the workspace**

Run, from WORKTREE:

```bash
git -c core.fsmonitor=false fetch origin
git cat-file -e origin/main:specs/plans/15-growth-margin-read.md; echo "exit=$?"
```

Expected: `exit=0`. This plan is on `origin/main`. If not, stop: its PR has not merged.

```bash
git rev-parse origin/main
git -c core.fsmonitor=false checkout --no-track -B claude/stage-3-growth-margins origin/main
git log --oneline -1
```

Record the `origin/main` SHA as BASE in the ledger.

- [ ] **Step 2: Parallel work**

```bash
git -c core.fsmonitor=false worktree list
gh pr list --repo lowmason/naics-embedder --state open --json number,title,files --jq '.[] | {number, title, files: [.files[].path]}'
git log --oneline 8aed0a6..origin/main -- src/naics_embedder/panels src/naics_embedder/decision conf/data uv.lock specs/naics-representation-split-roadmap.md specs/findings
```

Expected: no open PR's files and no listed commit touch those paths, apart from this plan's own
merge. Otherwise stop and ask (Stop-and-ask). Worktrees under `~/Projects/copilot-worktrees/`
are review tooling, not sessions.

- [ ] **Step 3: The level pins, in this lock's environment**

Two level pins (`CAPTURED_HERE`, `tests/unit/test_regressor_panel.py`) hold only on macOS arm64
with the lock's numpy, polars and Accelerate. Run them, and the growth and economic tests, in a
worktree environment built from the same lock:

```bash
uv sync --locked
uv run --locked --no-sync pytest -q -rs -n auto tests/unit/test_regressor_panel.py tests/unit/test_regressor_qcew_rows.py tests/unit/test_decision_economic.py; echo "exit=$?"
```

Expected: `exit=0`, and no skip reason reading "pinned where captured, on macOS arm64".
`test_level_data_fingerprints_are_the_captured_ones` and
`test_level_predictions_are_the_captured_ones_bit_for_bit` passed. A failure or that skip means
the environment changed since the pins were captured: stop and ask.

- [ ] **Step 4: Start the ledger and the scratch directory**

```bash
mkdir -p logs /tmp/naics_plan15
```

Write `logs/plan15_execution.md` with the Write tool: a title, BASE, the date, and Steps 1–3's
outputs.

## Tasks

### Task 1: Ready the main checkout

Nothing in this task reads real data.

**Files:** none in the repository. Writes MAIN's `.venv` only.

**Interfaces:**
- Consumes: BASE (Pre-flight Step 1).
- Produces: MAIN on `main` at `origin/main`, its `.venv` synced to `uv.lock`; LOCK_SHA; the
  environment receipt (Step 3); the record's path confirmed free.

- [ ] **Step 1: Fast-forward MAIN with the guard**

```bash
cd /Users/lowell/Projects/naics-embedder && git branch --show-current
```

Expected: `main`.

```bash
cd /Users/lowell/Projects/naics-embedder && git -c core.fsmonitor=false status --porcelain; echo "exit=$?"
```

Expected: no file lines, then `exit=0`.

```bash
cd /Users/lowell/Projects/naics-embedder && git log --oneline origin/main..main
```

Expected: nothing. Local `main` has no commit `origin/main` lacks.

```bash
cd /Users/lowell/Projects/naics-embedder && git -c core.fsmonitor=false merge --ff-only origin/main
cd /Users/lowell/Projects/naics-embedder && git rev-parse HEAD origin/main
```

Expected: a fast-forward (or "Already up to date"), then two equal SHAs, BASE or later. If any
check fails, stop and ask: never merge, rebase or reset MAIN to make it pass.

- [ ] **Step 2: Sync MAIN's environment to the lock**

```bash
cd /Users/lowell/Projects/naics-embedder && git diff --quiet HEAD -- uv.lock; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && shasum -a 256 uv.lock
cd /Users/lowell/Projects/naics-embedder && uv sync --locked; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && uv sync --locked --check; echo "exit=$?"
```

Expected: `exit=0`; `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948  uv.lock`;
`exit=0`; `exit=0`. Record the hash as LOCK_SHA. If it differs from the plan-time value,
`uv.lock` moved after this plan was written: stop and tell the user what changed
(`git log --oneline 8aed0a6..origin/main -- uv.lock`) before going on.

- [ ] **Step 3: The environment receipt**

The record names the platform, Python and the lock's sha256. This receipt adds what it does not
(P4). Write `/tmp/naics_plan15/env.py` with the Write tool:

```python
'''Plan 15, Task 1: the read's environment beyond the margin record's platform, Python and lock.'''

import os
import platform

import numpy
import polars

THREAD_VARIABLES = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS'
)

blas = numpy.show_config(mode='dicts')['Build Dependencies']['blas']
print(f'platform {platform.platform()}')
print(f'python {platform.python_version()}')
print(f'numpy {numpy.__version__}')
print(f'polars {polars.__version__}')
print(f"blas {blas['name']} (found {blas['found']})")
for name in THREAD_VARIABLES:
    print(f'{name} {os.environ.get(name, "unset")}')
```

Run:

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan15/env.py
```

Expected, at plan time: `platform macOS-26.6.2-arm64-arm-64bit`, `python 3.12.12`,
`numpy 2.3.4`, `polars 1.35.1`, `blas accelerate (found True)`, and every thread variable
`unset`. A newer macOS point release changes only the platform string: record it and go on.
Pairing compares predictions, not environments (`check_reference_pairing`). If a thread
variable is set, unset it in the shell that runs the read (Task 3), and note it here.

- [ ] **Step 4: The outputs**

```bash
test ! -e /Users/lowell/naics-artifacts/records/economic/growth_margins.json; echo "absent=$?"
test -w /Users/lowell/naics-artifacts; echo "writable=$?"
df -h /Users/lowell/naics-artifacts
ls /Users/lowell/naics-artifacts/records
```

Expected: `absent=0`; `writable=0`; free space in gigabytes (the read stores two small parquet
files); `stage7` and `stage8` and no `economic`. If `economic/` exists, stop and ask: something
may already have read growth.

- [ ] **Step 5: Ledger**

Copy Steps 1–4's outputs into the ledger.

### Task 2: Gate A — the inputs and the log, read only

**Files:** none in the repository. Reads the QCEW slices, the codebook and the selection log.

**Interfaces:**
- Consumes: Task 1's MAIN.
- Produces: the inputs' hashes; LINES_BEFORE and SHA_BEFORE, the log's physical line count and
  sha256; the log's tally, its records other than validation reads, and each regime's existing
  openings, for Gate B and the finding.

- [ ] **Step 1: Ask for Gate A**

Send the user this, and wait for a clear yes:

> Stage 3 is ready to start: MAIN is at `origin/main` <SHA>, its environment is synced to
> `uv.lock` <LOCK_SHA, first 12>, and the record's path is free. May I start Stage 3's real-data
> steps? First, read only: hash the four QCEW slices, check bundle 301cce28's codebook codes
> against their pin, and read MAIN's selection log (its size, a tally, any openings and any growth
> reads). Nothing is written. The margin read itself waits for a second go, after I show you what
> these checks find.

- [ ] **Step 2: The QCEW slices**

```bash
shasum -a 256 /Users/lowell/Downloads/Data/QCEW/2022_US000_annual.csv /Users/lowell/Downloads/Data/QCEW/2023_US000_annual.csv /Users/lowell/Downloads/Data/QCEW/2024_US000_annual.csv /Users/lowell/Downloads/Data/QCEW/2025_US000_annual.csv
```

Expected, the pins in `conf/data/regressor_panel.yaml`:

```text
c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18  …/2022_US000_annual.csv
fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9  …/2023_US000_annual.csv
48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46  …/2024_US000_annual.csv
0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811  …/2025_US000_annual.csv
```

The read verifies them itself; this check finds a mismatch before anything is logged.

- [ ] **Step 3: The codebook (O1)**

Write `/tmp/naics_plan15/codebook.py` with the Write tool:

```python
'''Plan 15, Task 2: the codebook's codes against the panel config's pin, without loading the panel.'''

import sys
from pathlib import Path

from naics_embedder.panels.regressor_splits import read_codebook_codes
from naics_embedder.utils.config import RegressorPanelConfig, load_config

PANEL_CONFIG = Path('/Users/lowell/Projects/naics-embedder/conf/data/regressor_panel.yaml')

if not PANEL_CONFIG.is_file():
    sys.exit(f'{PANEL_CONFIG} is missing')
cfg = load_config(RegressorPanelConfig, PANEL_CONFIG)
codes = read_codebook_codes(Path(sys.argv[1]), cfg.codebook_codes_sha256)
print(f'{len(codes)} codes; their fingerprint is the pin {cfg.codebook_codes_sha256}')
print(f'held-out draw {cfg.heldout_groups_sha256}')
print(f'outcome {cfg.outcome}')
```

Run:

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan15/codebook.py /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && shasum -a 256 data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet
```

Expected: `2125 codes; their fingerprint is the pin
9b646af189dbc46870861367e7cd399d2328972754dbf1abb30ca5a864f2481a`, `held-out draw
deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4`, `outcome level` and
`exit=0`. The manifest hashes to `68545c66…` (Workspace). Record the codebook parquet's sha256
as CODEBOOK_SHA and the manifest's as MANIFEST_SHA. `outcome level` is right: the read sets
growth in memory.

- [ ] **Step 4: The log**

Write `/tmp/naics_plan15/log_state.py` with the Write tool:

```python
'''
Plan 15, Task 2: the main checkout's selection log before the margin read, read only.

Prints the log's physical line count and sha256, a tally of its records, every record that is
not a validation read, each regressor regime's openings under the held-out draw (Req 22) and the
growth reads. Exits 1 if a record lacks a field the margin read's own lookup reads, or if any
growth read exists: the margin read must be the log's first growth read (Req 24).
'''

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from naics_embedder.panels.regressor import Regime, seal_names
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.utils.config import RegressorPanelConfig, load_config

PANEL_CONFIG = Path('/Users/lowell/Projects/naics-embedder/conf/data/regressor_panel.yaml')
FIELDS = {'time', 'event', 'panel', 'split', 'purpose', 'fingerprint', 'n_queries', 'detail'}
GROWTH = ('growth_seen', 'growth_heldout')

def show(record):
    # The log writes each record as json.dumps(record, sort_keys=True): this is its line
    print('    ' + json.dumps(record, sort_keys=True))

path = Path(sys.argv[1])
if not PANEL_CONFIG.is_file():
    sys.exit(f'{PANEL_CONFIG} is missing')
fingerprint = load_config(RegressorPanelConfig, PANEL_CONFIG).heldout_groups_sha256
raw = path.read_bytes()
log = SelectionLog(path)
records = log.records()
newline = raw.endswith(b'\n')
print(f'log {path}')
print(f'lines {len(raw.splitlines())}, records {len(records)}, ends with a newline: {newline}')
print(f'sha256 {hashlib.sha256(raw).hexdigest()}')

malformed = [r for r in records if not FIELDS <= set(r) or not isinstance(r.get('detail'), dict)]
print(f'records missing a field: {len(malformed)}')
for record in malformed:
    show(record)
if malformed:
    sys.exit(1)

tally = Counter((r['event'], r['split'], r['panel']) for r in records)
print('tally (count, event, split, panel):')
for (event, split, panel), count in sorted(tally.items()):
    print(f'  {count:5d}  {event} {split} {panel}')
others = [r for r in records if (r['event'], r['split']) != ('read', 'validation')]
print(f'records other than validation reads: {len(others)}')
for record in others:
    show(record)
for regime in Regime:
    names = seal_names(regime)
    openings = log.openings(names, fingerprint)
    print(f'openings of the {regime.value} split {names} under {fingerprint}: {len(openings)}')
    for record in openings:
        show(record)
growth = [r for r in records if r['panel'] in GROWTH]
print(f'growth records: {len(growth)}')
for record in growth:
    show(record)
sys.exit(1 if growth else 0)
```

Run:

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan15/log_state.py /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl > /tmp/naics_plan15/log_before.txt; echo "exit=$?"
cat /tmp/naics_plan15/log_before.txt
```

Expected: `exit=0`; `ends with a newline: True`; `records missing a field: 0`;
`growth records: 0`. Record `lines` as LINES_BEFORE and `sha256` as SHA_BEFORE. The tally, the
records other than validation reads and the openings are whatever they are: the finding lists
them. An opening of a regime's split already seals its growth split (Req 22), which does not
stop a validation read; Gate B tells the user.

If the exit is 1, stop and ask. A growth read before the margin read breaks margin first
(Req 24), and a record missing `detail` would crash the read's own log lookup after it had
appended its reads.

- [ ] **Step 5: Ledger**

Copy Steps 2–4's outputs into the ledger, with LINES_BEFORE, SHA_BEFORE, CODEBOOK_SHA and
MANIFEST_SHA.

### Task 3: Gate B — the margin read

**Files:** none in the repository. Appends two records to MAIN's selection log and writes the
record and the store's objects.

**Interfaces:**
- Consumes: Tasks 1–2's values.
- Produces: READ_COMMIT, the commit MAIN was at during the read (O2); the margin record at
  `/Users/lowell/naics-artifacts/records/economic/growth_margins.json`; the command's output in
  `/tmp/naics_plan15/read.txt`.

- [ ] **Step 1: Pin the commit (O2)**

```bash
cd /Users/lowell/Projects/naics-embedder && git -c core.fsmonitor=false fetch origin
cd /Users/lowell/Projects/naics-embedder && git rev-parse HEAD origin/main
cd /Users/lowell/Projects/naics-embedder && git -c core.fsmonitor=false status --porcelain; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && shasum -a 256 uv.lock
```

Expected: two equal SHAs, as Task 1 Step 1 left them; no file lines and `exit=0`; LOCK_SHA.
Record the SHA as READ_COMMIT. If the two SHAs differ, `origin/main` moved since Task 1: rerun
Task 1 Steps 1–2, tell the user what the new commits touch (a touch of a Stop-and-ask path stops
here), and come back to this step.

- [ ] **Step 2: Ask for Gate B**

Send the user this, filled from the ledger, and wait for a clear go:

> Gate A's checks are done. MAIN is at `origin/main` <READ_COMMIT>, with `uv.lock` <LOCK_SHA>, on
> <platform>, Python <python>, numpy <numpy> on Accelerate, no BLAS thread variable set. The QCEW
> slices and bundle 301cce28's codes match their pins. The selection log has <LINES_BEFORE>
> records and no growth read. Records other than validation reads: <none, or the list>. Openings
> of the seen split: <none, or the list>; of the held-out split: <none, or the list>.
>
> The margin read runs once and cannot be taken back: it appends two validation reads to the log
> and fixes both growth margins. It scores only `covariates`, `one_hot` (seen regime),
> `ancestors` and their covariate pairs, and opens no outer set. The command, from MAIN's root:
>
> <Step 4's command>
>
> Go?

- [ ] **Step 3: Re-check the log immediately before the read**

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan15/log_state.py /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl > /tmp/naics_plan15/log_gate_b.txt; echo "exit=$?"
diff /tmp/naics_plan15/log_before.txt /tmp/naics_plan15/log_gate_b.txt; echo "diff=$?"
```

Expected: `exit=0` and `diff=0`. The log is as Task 2 found it. Otherwise something wrote to it
in between: stop and ask, and do not run the read.

- [ ] **Step 4: The read**

Run it with the Bash tool's `run_in_background: true`, since it may take minutes, and wait for
its completion notice. Do not poll, and do not start anything else that touches MAIN meanwhile.

```bash
cd /Users/lowell/Projects/naics-embedder && pwd && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools growth-margins \
  --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet \
  --name growth-margins \
  --store /Users/lowell/naics-artifacts \
  --output /Users/lowell/naics-artifacts/records/economic/growth_margins.json \
  --purpose 'economic margin read: split roadmap Stage 3, plan 15 (Req 24)' \
  --log /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl \
  --lock /Users/lowell/Projects/naics-embedder/uv.lock \
  > /tmp/naics_plan15/read.txt 2>&1; echo "exit=$?"
```

Then: `cat /tmp/naics_plan15/read.txt`.

- [ ] **Step 5: If the exit is not 0**

Check the log before anything else:

```bash
cd /Users/lowell/Projects/naics-embedder && wc -l < logs/selection_log.jsonl && shasum -a 256 logs/selection_log.jsonl
```

- The count is LINES_BEFORE and the hash SHA_BEFORE: the read failed before it logged anything.
  This covers an existing record, an unwritable path, a QCEW or codebook mismatch and a missing
  lock, each refused before any read. Stop, show the user the error, and wait. A later run after
  their ruling is still the first.
- The count or the hash differs: one or both growth reads were logged before the failure
  (`fix_relative_margins` logs each regime's read before it writes the store or the record).
  Stop and show the user the appended lines (`tail -n +$((LINES_BEFORE + 1))` of the log).
  Never re-run: a second run would log a second run's reads (Exit).

- [ ] **Step 6: If the exit is 0**

Expected in `/tmp/naics_plan15/read.txt`, after the log lines:

```text
Growth margins 'growth-margins', fixed <FIXED_AT>

  • growth_seen: δ <DELTA_SEEN> = 0.05 × MSE <MSE_SEEN> of covariates+ancestors
  • growth_heldout: δ <DELTA_HELDOUT> = 0.05 × MSE <MSE_HELDOUT> of covariates+ancestors

Read on <platform>, Python <python>, uv.lock <LOCK_SHA>: every candidate read must run there too
Reads logged to /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl (fingerprint deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4)

Margin record: /Users/lowell/naics-artifacts/records/economic/growth_margins.json
```

`COLUMNS=200` keeps Rich from folding these lines in the file. `pwd` printed
`/Users/lowell/Projects/naics-embedder`. If the "Reads logged to" path is not MAIN's log, stop and
ask. Copy the output to the ledger.

### Task 4: Verify the record and the log

Read only: nothing here reads a panel or appends to the log.

**Files:** none in the repository.

**Interfaces:**
- Consumes: the record; LINES_BEFORE, SHA_BEFORE (Task 2); LOCK_SHA (Task 1).
- Produces: RECORD_SHA; the verified values; LINES_AFTER and SHA_AFTER; the read's two log lines,
  for the finding.

- [ ] **Step 1: The record's sha256**

```bash
shasum -a 256 /Users/lowell/naics-artifacts/records/economic/growth_margins.json
```

Record it as RECORD_SHA.

- [ ] **Step 2: Check the record and the log**

Write `/tmp/naics_plan15/verify.py` with the Write tool:

```python
'''
Plan 15, Task 4: the margin record and the selection log after the read, read only.

Checks the record as tools decide will (check_relative_margins, check_economic_margins_first);
its r, reference, panels, environment and draw; its stored scores against a rescore of its
predictions (P5); and that the log gained exactly the read's two validation reads, of the
reference comparators alone, after an unchanged prefix (Req 24; Req 4).
'''

import hashlib
import json
import math
import sys
from pathlib import Path

from naics_embedder.decision.economic import (
    GROWTH_PANELS,
    REFERENCE_COMPARATOR,
    STATISTIC_TOLERANCE,
    check_economic_margins_first,
    check_relative_margins,
)
from naics_embedder.decision.records import RelativeMarginRecord, read_record
from naics_embedder.decision.scores import panel_statistic, regressor_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.qcew_rows import RegressorOutcome
from naics_embedder.panels.regressor import (
    DECISION_LEVEL,
    READ_NAMES,
    Regime,
    reference_comparators,
)
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.config import DecisionConfig, RegressorPanelConfig, load_config

CONF = Path('/Users/lowell/Projects/naics-embedder/conf/data')
# Plan 14's Handoff: the held-out regime has no one-hot comparator at level 6
COMPARATORS = {
    'growth_seen': ['covariates', 'one_hot', 'covariates+one_hot', 'ancestors', 'covariates+ancestors'],
    'growth_heldout': ['covariates', 'ancestors', 'covariates+ancestors'],
}

def require(condition, message):
    if not condition:
        sys.exit(f'FAILED: {message}')

record_path, store_root, log_path, lock_path = (Path(arg) for arg in sys.argv[1:5])
lines_before, sha_before = int(sys.argv[5]), sys.argv[6]
for name in ('decision.yaml', 'regressor_panel.yaml'):
    require((CONF / name).is_file(), f'{CONF / name} is missing')
decision = load_config(DecisionConfig, CONF / 'decision.yaml')
regressor = load_config(RegressorPanelConfig, CONF / 'regressor_panel.yaml')

# The record, as tools decide reads it
record = read_record(record_path, RelativeMarginRecord)
store = ArtifactStore(store_root)
check_relative_margins(record, store)
check_economic_margins_first([], record)
require(record.r == decision.relative_margin == 0.05, f'r is {record.r}')
require(record.reference == REFERENCE_COMPARATOR, f'the reference is {record.reference}')
require(sorted(m.panel for m in record.margins) == sorted(GROWTH_PANELS), 'not one δ per panel')
require(record.read.environment.lock_sha256 == sha256_file(lock_path), 'the lock differs')
require(record.read.panels.regressor == regressor.heldout_groups_sha256, 'another draw')

# The stored scores, which check_relative_margins does not read (P5)
repeats = record.read.panels.fit_settings['repeats']
stored = store.read_frame(record.read.scores)
rescored = regressor_scores(store.read_frame(record.read.predictions), repeats, GROWTH_PANELS)
for entry in record.margins:
    for label, frame in (('stored', stored), ('rescored', rescored)):
        value = panel_statistic(frame, entry.panel, REFERENCE_COMPARATOR)
        require(
            math.isclose(value, entry.reference_mse, rel_tol=STATISTIC_TOLERANCE),
            f'{entry.panel}: {label} MSE {value} is not {entry.reference_mse}',
        )
    require(math.isfinite(entry.margin) and entry.margin > 0, f'{entry.panel}: δ {entry.margin}')
    require(entry.margin == record.r * entry.reference_mse, f'{entry.panel}: δ is not r × MSE')

# The log: an unchanged prefix, then exactly the read's two records
raw = log_path.read_bytes()
lines = raw.splitlines(keepends=True)
require(len(lines) == lines_before + 2, f'the log has {len(lines)} lines, not {lines_before} + 2')
prefix = hashlib.sha256(b''.join(lines[:lines_before])).hexdigest()
require(prefix == sha_before, f'the first {lines_before} lines hash to {prefix}')
new = [json.loads(line) for line in lines[lines_before:]]
require(new == record.read.log_records, "the appended records are not the record's log records")
regimes = {READ_NAMES[RegressorOutcome.GROWTH][regime]: regime for regime in Regime}
require(sorted(r['panel'] for r in new) == sorted(GROWTH_PANELS), 'not one read per growth panel')
for r in new:
    logged = (r['event'], r['split'], r['fingerprint'])
    require(logged == ('read', 'validation', regressor.heldout_groups_sha256), f'{logged}')
    names = list(reference_comparators(regimes[r['panel']], DECISION_LEVEL))
    require(names == COMPARATORS[r['panel']], f'{r["panel"]} comparators {names}')
    detail = {'level': DECISION_LEVEL, 'comparators': names, 'run': record.read.run_id}
    require(r['detail'] == detail, f'{r["panel"]} detail {r["detail"]}')
    parts = {part for name in names for part in name.split('+')}
    require(not parts & {'embedding', 'text_only'}, f'{r["panel"]} read a candidate')

print('all checks passed')
print(f'name {record.name}, r {record.r}, reference {record.reference}')
print(f'fixed_at {record.fixed_at.isoformat()}, run {record.read.run_id}')
for entry in record.margins:
    print(f'{entry.panel}: reference MSE {entry.reference_mse!r}, δ {entry.margin!r}')
environment = record.read.environment
print(f'environment {environment.platform}, Python {environment.python}, '
      f'lock {environment.lock_sha256}')
print(f'draw {record.read.panels.regressor}, level-6 data {record.read.panels.regressor_data}')
for label, ref in (('predictions', record.read.predictions), ('scores', record.read.scores)):
    print(f'{label} {ref.path} sha256 {ref.sha256} ({ref.bytes} bytes)')
print(f'log lines {len(lines)}, sha256 {hashlib.sha256(raw).hexdigest()}')
```

Run:

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan15/verify.py /Users/lowell/naics-artifacts/records/economic/growth_margins.json /Users/lowell/naics-artifacts /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl /Users/lowell/Projects/naics-embedder/uv.lock LINES_BEFORE SHA_BEFORE > /tmp/naics_plan15/verify.txt 2>&1; echo "exit=$?"
cat /tmp/naics_plan15/verify.txt
```

Substitute LINES_BEFORE and SHA_BEFORE from the ledger. Expected: `exit=0`, `all checks passed`,
then the values. Record `log lines` as LINES_AFTER (LINES_BEFORE + 2) and its sha256 as
SHA_AFTER. Any `FAILED:` line, or a traceback, means stop and ask. Never re-run the read.

- [ ] **Step 3: The read's two log lines, verbatim**

```bash
cd /Users/lowell/Projects/naics-embedder && tail -n 2 logs/selection_log.jsonl > /tmp/naics_plan15/margin_reads.jsonl
cat /tmp/naics_plan15/margin_reads.jsonl
cp /Users/lowell/naics-artifacts/records/economic/growth_margins.json /tmp/naics_plan15/growth_margins.json
```

Step 2 has shown these are the read's two records. The copy in `/tmp` is only for Task 5's Write
tool, and is deleted with the scratch directory.

- [ ] **Step 4: MAIN is clean and unmoved**

```bash
cd /Users/lowell/Projects/naics-embedder && git -c core.fsmonitor=false status --porcelain; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && git rev-parse HEAD
```

Expected: no file lines, `exit=0`, READ_COMMIT.

- [ ] **Step 5: Ledger**

Copy Steps 1–4's outputs into the ledger.

### Task 5: The finding

**Files:**
- Create: `specs/findings/growth-margins.md`

**Interfaces:**
- Consumes: the ledger's values from Tasks 1–4 and the files in `/tmp/naics_plan15/`.
- Produces: the finding, which Plan completion's stamp and the candidates' spec cite.

- [ ] **Step 1: Write the finding**

Write `WORKTREE/specs/findings/growth-margins.md` with the Write tool, from this skeleton. Each
`{{…}}` is a value from the ledger or an instruction: replace it, braces included. Take the MSEs
and δ at full precision from Task 4 Step 2 (`!r`), not from the command's six-digit summary.
Copy the record and the log lines byte for byte, and re-wrap prose at 100 characters once the
values are in.

````markdown
# Growth margins: finding

**Status: FINAL ({{DATE}}).** Split roadmap Stage 3 (`specs/naics-representation-split-roadmap.md`),
implemented by plan 15 (`specs/plans/completed/15-growth-margin-read.md`).

The reference `covariates+ancestors` was read alone, on the Mac, on both growth panels' level-6
validation splits, before any economic candidate exists (Req 24). Each panel's δ is r = 0.05
times the reference's validation MSE there (U13): `growth_seen` δ = {{DELTA_SEEN}} and
`growth_heldout` δ = {{DELTA_HELDOUT}}. The selection log held no growth read before this one, and
it gained exactly the read's two validation reads. No outer set was opened. Every economic
candidate is decided against this record.

## Sources

| Input | SHA-256 |
|---|---|
| `~/Downloads/Data/QCEW/2022_US000_annual.csv` | c45cbb64a1b1eef16bfd743510d9d02792ccad82f60e9df202c5daa3e8c5cc18 |
| `~/Downloads/Data/QCEW/2023_US000_annual.csv` | fe9ffe874f6e657f6bb1558971965ce6acc015ace45d831ed32c90d97097aee9 |
| `~/Downloads/Data/QCEW/2024_US000_annual.csv` | 48db086828a01798731242c6d3d4957f80f941afe75463a1ff7d43de774bea46 |
| `~/Downloads/Data/QCEW/2025_US000_annual.csv` | 0b5528f70d66a84ff9729691f365c667a09f854f0af3d841bdd660ef3cb01811 |
| Bundle 301cce28 `manifest.json` | {{MANIFEST_SHA}} |
| Bundle 301cce28 `naics_codebook.parquet` | {{CODEBOOK_SHA}} |
| The codebook's 2,125 codes (`codebook_codes_sha256`) | 9b646af189dbc46870861367e7cd399d2328972754dbf1abb30ca5a864f2481a |
| The held-out draw (`heldout_groups_sha256`) | deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4 |
| `uv.lock` | {{LOCK_SHA}} |

The bundle is `stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c`. Its codes hash to
the panel config's pin (plan 15, Task 2), though the comment beside the pin names bundle
18403d29. The read ran at `origin/main` {{READ_COMMIT}}, from the main checkout's root, on `main`
with a clean tree. A candidate rebuilds its lock in a worktree at {{READ_COMMIT}} with
`uv sync --locked`.

## 1. The read

```bash
{{Task 3 Step 4's command, as run}}
```

| Environment | Value |
|---|---|
| Platform | {{PLATFORM}} |
| Python | {{PYTHON}} |
| numpy | {{NUMPY}} |
| polars | {{POLARS}} |
| BLAS | {{BLAS}} |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `POLARS_MAX_THREADS` | {{THREADS}} (Task 1 Step 3) |

The record names the platform, Python and lock. The rest is recorded here because the record does
not, and a candidate read that fails to pair should check it first (plan 14's Handoff: BLAS
threading is the first suspect). Pairing compares the reference's predictions bit for bit; the
environments only name where each read ran (`check_reference_pairing`).

The command printed:

```text
{{the last lines of /tmp/naics_plan15/read.txt, from "Growth margins" on}}
```

## 2. The margins

| Panel | Statistic | Reference MSE | δ |
|---|---|---|---|
| `growth_seen` | `covariates+ancestors` | {{MSE_SEEN}} | {{DELTA_SEEN}} |
| `growth_heldout` | `covariates+ancestors` | {{MSE_HELDOUT}} | {{DELTA_HELDOUT}} |

r = 0.05 is `relative_margin` in `conf/data/decision.yaml`. The margins were fixed at {{FIXED_AT}},
by the read `{{RUN_ID}}` (Task 4 Step 2). The level-6 data fingerprint is {{DATA_FINGERPRINT}}.

| Record | SHA-256 |
|---|---|
| `~/naics-artifacts/records/economic/growth_margins.json` | {{RECORD_SHA}} |

The read's predictions are `{{PREDICTIONS_PATH}}` (sha256 {{PREDICTIONS_SHA}}) and its scores
`{{SCORES_PATH}}` (sha256 {{SCORES_SHA}}), in the store `~/naics-artifacts`. The record passes
`check_relative_margins` and `check_economic_margins_first`, as `tools decide` will run them. Its
stored scores give each reference MSE within `STATISTIC_TOLERANCE` (1e-12), as a rescore of its
predictions does (plan 15, Task 4). The record, verbatim:

```json
{{the contents of /tmp/naics_plan15/growth_margins.json}}
```

## 3. The selection log

Before the read, `logs/selection_log.jsonl` in the main checkout had {{LINES_BEFORE}} records,
SHA-256 {{SHA_BEFORE}}. Every record had the fields the read's log lookup reads.

| Records | Event | Split | Panel |
|---|---|---|---|
| {{one row per line of Task 2's tally}} |

- **Records other than validation reads:** {{"none", or each line verbatim in a json block}}.
- **Growth reads:** none. The log held no `growth_seen` or `growth_heldout` read before the margin
  read, so the margins are first (Req 24).
- **Existing openings** under the held-out draw `deddfd4c…`, checked with
  `SelectionLog.openings(seal_names(regime), fingerprint)` (Req 22): {{"none for either regime",
  or each regime's openings verbatim. An opening of a regime's level split also seals its growth
  split.}}

After the read the log has {{LINES_AFTER}} records, SHA-256 {{SHA_AFTER}}. Its first {{LINES_BEFORE}} lines
still hash to {{SHA_BEFORE}}. The read appended exactly these two validation reads, the record's own
`log_records`. The seen regime logged `covariates`, `one_hot`, `covariates+one_hot`, `ancestors`
and `covariates+ancestors`, and the held-out regime `covariates`, `ancestors` and
`covariates+ancestors`. Neither logged `embedding` or `text_only`, and no outer set was opened.

```json
{{the two lines of /tmp/naics_plan15/margin_reads.jsonl}}
```

## 4. What the candidates' spec reads

- **The record:** `~/naics-artifacts/records/economic/growth_margins.json`, SHA-256 {{RECORD_SHA}},
  named `growth-margins`, with δ = {{DELTA_SEEN}} on `growth_seen` and {{DELTA_HELDOUT}} on
  `growth_heldout`. `tools decide --margins` takes it.
- **Fixed once.** Nothing refuses a second `tools growth-margins` run. Never run it again: a
  second run would log a second margin read, and a candidate read before a record's `fixed_at`
  is refused under it.
- **Where candidates read.** `decide` pairs a candidate only where the reference's predictions are
  bit-identical to this read's. Read candidates on the Mac, at {{READ_COMMIT}}'s `uv.lock`
  ({{LOCK_SHA}}), with Python {{PYTHON}}, numpy {{NUMPY}} on Accelerate and no BLAS thread variable set.
  Bit identity was verified on fixtures only. If a real candidate read fails to pair, suspect
  BLAS threading first. The check fails closed.
- **Open items.** The deferred items under `## 14-economic-decision-tooling` in
  `specs/deferred_items.md` go to the candidates' spec. Plan 15 checked this record's stored
  scores once; the stored-scores item stays open.

## Reproduction

Do not re-run the read: the margins are fixed once. The commands are recorded so that their
inputs can be checked.

```bash
{{Task 1 Step 2's sync, Task 3 Step 4's command, and the shasum of the record, as run}}
```
````

- [ ] **Step 2: Check the finding's lines**

Write `/tmp/naics_plan15/widths.py` with the Write tool:

```python
'''Plan 15: report Markdown lines over 100 characters outside code blocks and tables.'''

import sys
from pathlib import Path

fenced = False
over = 0
for number, line in enumerate(Path(sys.argv[1]).read_text(encoding='utf-8').splitlines(), 1):
    if line.lstrip().startswith(('```', '~~~')):
        fenced = not fenced
        continue
    if not fenced and not line.lstrip().startswith('|') and len(line) > 100:
        over += 1
        print(f'{number}: {len(line)} characters')
print(f'{over} lines over 100')
sys.exit(1 if over else 0)
```

```bash
python3 /tmp/naics_plan15/widths.py specs/findings/growth-margins.md; echo "exit=$?"
grep -n -F '{{' specs/findings/growth-margins.md; echo "tokens=$?"
```

Expected: `0 lines over 100` and `exit=0`; no line printed and `tokens=1`, so no `{{…}}` is left.
JSON never contains `{{`, so a hit is always an unfilled token.

- [ ] **Step 3: Commit**

```bash
pwd
git add specs/findings/growth-margins.md
git commit -m "docs(findings): record the growth margins from the reference-only read" -m "Stage 3 of the representation-split roadmap (plan 15)." -m "Co-Authored-By: <the executing model> <noreply@anthropic.com>"
```

`pwd` prints WORKTREE.

## Final verification (controller, inline)

- [ ] **Step 1: The Exit**

Check each clause of Stage 3's Exit (Global Constraints), citing the ledger:

- the margin record exists with r = 0.05 and one δ per growth panel (Task 4 Step 2);
- the log gained only the two growth validation reads of one run, logging the comparators plan
  14's Handoff lists per regime and neither `embedding` nor `text_only` (Task 4 Step 2);
- the margins were fixed once: the read ran once (Task 3);
- no outer set was opened (Task 4 Step 2: the only new records are the two reads);
- the finding cites the record's sha256, the platform, the lock and READ_COMMIT; states that the
  log held no growth read before the margin read; and lists the existing openings or says there
  were none (Task 5).

- [ ] **Step 2: The checks**

```bash
uv run --locked --no-sync ruff check src/ tests/; echo "exit=$?"
./scripts/format_code.sh --check --all; echo "exit=$?"
uv run --locked --no-sync pytest -n auto -q; echo "exit=$?"
uv run --locked --no-sync mkdocs build --strict --site-dir /tmp/naics_plan15/site; echo "exit=$?"
rm -rf /tmp/naics_plan15/site
```

Expected: `exit=0` four times. This branch changes no code, so these show the tree it merges into
is green (Verification "Checks"). Record the collected and passed counts; skips depend on the
machine. Run the full suite with `run_in_background: true` if it nears the Bash tool's two-minute
limit.

- [ ] **Step 3: Review**

Dispatch the code-reviewer agent on `git diff BASE...HEAD`, with this plan, the roadmap's Stage 3
entry, Req 24 and the ledger. Ask it to check every value in the finding against the ledger and
`/tmp/naics_plan15/`. Then run the local Codex review:

```bash
codex exec review -m gpt-6-astra --base origin/main
```

Run it in the background, too. Pass `-m gpt-6-astra` and never edit `~/.codex/config.toml`. Fix
each finding in the finding, or triage it for Plan completion's gate.

## Plan completion (controller, inline)

writing-plans' Plan Completion Protocol, with this plan's edits written out. It runs on
`claude/stage-3-growth-margins`. Stage 3 is the roadmap's last stage: do not plan past it, do not
run the roadmap's Completion audit, do not move the roadmap, and do not retire
`specs/naics-representation-split.md`. The roadmap's Completion section keeps the spec in
`specs/` while the candidates' and text→code specs cite it, and the next derive-roadmap resume
runs that audit. The plan's file name follows its stage, not the spec, so the skill's
spec-name match does not apply.

- [ ] **Step 1: Parallel sessions**

`specs/naics-representation-split-roadmap.md` and `specs/deferred_items.md` are shared.

```bash
git -c core.fsmonitor=false fetch origin
git log --oneline BASE..origin/main -- specs/naics-representation-split-roadmap.md specs/deferred_items.md
gh pr list --repo lowmason/naics-embedder --state open --json number,title,files --jq '.[] | {number, title, files: [.files[].path]}'
```

If either shows a change to those files, hold Steps 4–6 and give the user the exact text of each
edit.

- [ ] **Step 2: The resolve-before-defer gate**

Collect the leftovers: plan steps skipped or descoped, and review findings not fixed. Ask the
user, as one batched set of questions, about those that need their input; implement those an
answer unblocks, then restart this protocol; defer the rest. The partition is final only once
every question is answered.

- [ ] **Step 3: Mark up this plan**

- Tick every completed step (`- [x]`).
- Under each step that deviated, add a one-line `> Deviation: …` note.
- Annotate each skipped step with `> Skipped: <why> → deferred`.
- Add the status line under the title: `**Status: COMPLETE (<date>)** — executed via
  executing-plans; nothing deferred`, or `…; deferred items in specs/deferred_items.md` if the
  gate deferred any.

- [ ] **Step 4: The deferred items**

In `specs/deferred_items.md`, tick nothing: no open item names Stage 3, and Task 4's one-off
scores check is not the stored-scores item's fix (P5). If the gate deferred anything, append a
`## 15-growth-margin-read — <date>` section in the file's item schema (self-contained, with
`Size:` and `Done when:` or `Revisit if:`). Otherwise append nothing.

- [ ] **Step 5: The Handoff section**

Append a level-2 section to the end of this plan, titled "Handoff to later stages", whose body is
the single line `None: last stage.` It is the plan's only such section, so a later resume cites
it without a draft to tell it from. The candidates' spec, which is not a stage, reads the finding.

- [ ] **Step 6: The roadmap stamp**

In `specs/naics-representation-split-roadmap.md`, Stage 3's entry:

- tick it: `- [ ] Stage 3: The growth margin read` becomes `- [x] Stage 3: The growth margin read`;
- after its `ROUTING: writing-plans` line, add the stamp, indented as Stages 1 and 2's are:

```text
      > Stage 3: COMPLETE (<date>) — implemented by plan 15
      > (specs/plans/completed/15-growth-margin-read.md).
      > Next: resume the roadmap.
```

Edit nothing else in the roadmap: its status line and the later re-validation are the resume's.

- [ ] **Step 7: Backlog triage**

```bash
uv run --no-project --python 3.13 python /Users/lowell/.claude/skills/writing-plans/scripts/deferred_stats.py
```

Report its summary line: the open count, the closure rate and the aged tail. With 20 or more
open items or any aged tail, run steps 1–4 of the skill's Triage rubric
(`references/deferred-backlog.md`) and present the proposal. It is read-only; `/deferred` acts
on a selection.

- [ ] **Step 8: Retire the plan**

```bash
pwd
git mv specs/plans/15-growth-margin-read.md specs/plans/completed/15-growth-margin-read.md
git add specs/plans/completed/15-growth-margin-read.md specs/naics-representation-split-roadmap.md specs/deferred_items.md
git commit -m "chore(specs): retire plan 15" -m "Co-Authored-By: <the executing model> <noreply@anthropic.com>"
```

This plan has no relative links to re-point. `specs/deferred_items.md` is staged only if Step 4
changed it.

## Integration (controller, inline)

- [ ] **Step 1: Before pushing**

```bash
git -c core.fsmonitor=false fetch origin
git log --oneline BASE..origin/main
git merge-tree --write-tree origin/main HEAD; echo "exit=$?"
```

Expected: `exit=0` (no conflict). If `origin/main` moved, list what it touched. A move into the
roadmap or `specs/deferred_items.md` goes back to Plan completion Step 1.

- [ ] **Step 2: Push and open the PR, with the user's go-ahead**

Ask the user for the go-ahead to push and open the PR. Then:

```bash
git push -u origin claude/stage-3-growth-margins
gh pr create --repo lowmason/naics-embedder --base main --head claude/stage-3-growth-margins --title "Stage 3: the growth margin read" --body-file /tmp/naics_plan15/pr_body.md
```

Write `/tmp/naics_plan15/pr_body.md` first, with the Write tool. It covers: the read (READ_COMMIT,
the two δ, the record's sha256), that it ran once, the log's growth by two validation reads, no
outer set opened, the existing openings or none, and the checks' results. It ends with
`🤖 Generated with [Claude Code](https://claude.com/claude-code)`. The user merges the PR.

- [ ] **Step 3: After the user merges**

```bash
gh pr view claude/stage-3-growth-margins --repo lowmason/naics-embedder --json state,mergeCommit
```

Expected: `MERGED`. Then fast-forward MAIN with Task 1 Step 1's guard, so local `main` has the
finding.

- [ ] **Step 4: Ledger and scratch**

```bash
cp logs/plan15_execution.md /Users/lowell/Projects/naics-embedder/logs/plan15_execution.md
rm -rf /tmp/naics_plan15
```

The worktree's `logs/` is ignored and goes with the worktree, so the ledger's only lasting copy
is MAIN's. The margin record and the store stay where they are.

- [ ] **Step 5: Hand back**

Report the PR, the two δ, RECORD_SHA and READ_COMMIT. Stage 3 is the roadmap's last stage: the
next step is a derive-roadmap resume, which runs the roadmap's Completion. Route to no stage.
Clean up the worktree by detaching it onto `origin/main`, deleting the branch from inside it
(`git branch -d claude/stage-3-growth-margins`), and archiving the session.

## Execution

Run this plan inline with executing-plans (P8), in a fresh session on Sonnet (O4):

```bash
claude --model sonnet
```

Start it on this file, `specs/plans/15-growth-margin-read.md`, once its PR has merged.
