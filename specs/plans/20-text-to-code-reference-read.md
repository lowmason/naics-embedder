# Text→code Reference Read and Gap Implementation Plan

**Status: COMPLETE (2026-11-19)** — executed via executing-plans; deferred items in specs/deferred_items.md

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 4 — on plan completion, tick the stage
> and re-validate later stages against what shipped.

**Goal:** Build Stage 4 of the redesign roadmap. Add the ITA read command and the gap statistic.
Then, with the user's approval at the time, read both text→code reference arms on the real ITA
validation part, and write a finding on the gap between Census phrases and owner text.

**Architecture:** Code first (PR A), then real reads from the main checkout, then the finding
(PR B).

- **The pin (Task 1).** `conf/data/ita_evidence.yaml` pins embedding-data's run and the SHA-256 of
  its two pair tables and `outputs.json` (protocol Section 1). `panels/ita_evidence.py` checks the
  files against it before any read (C1).
- **The reads (Task 2).** `decision/ita_reads.py` reads one reference arm on the ITA validation
  part with role `reference`. On request, it also reads the outcome panel's validation split with
  the same encoder. It stores the per-pair frame and returns an `ItaReferenceRead` record. The
  thin command `tools ita-read` calls it, and its `--dry-run` logs nothing.
- **The gap (Task 3).** `decision/ita_gap.py` computes each arm's Census-phrase, copied and
  headline MRRs, the gaps between them, a code-matched gap, and the D2 verdict. It uses the
  paired two-stage bootstrap. The thin command `tools ita-gap` writes the `ItaGapRecord`. It reads
  no panel and logs nothing.
- **PR A (Task 4)** merges. Then **Gate A (Task 6)** runs the read-only checks and six dry runs,
  **Gate B (Task 7)** runs six logged reads, and **Task 8** writes the gap record and verifies.
- **The finding (Task 9, PR B).** `specs/findings/text-to-code-gap.md` reports aggregates only.

**Tech Stack:** Python 3.12 under uv (`uv run --locked --no-sync`), PyTorch (float64 CPU
distances), Polars, numpy, Pydantic v2, Typer/Rich, pytest, mkdocstrings; the `naics-embedder`
CLI; `shasum`, git and gh. No new dependency.

**Spec:** none. Stage 4 routes to writing-plans and has no stage spec. Its requirements are the
roadmap's Stage 4 entry (`specs/naics-artifact-redesign-roadmap.md`), which cites Req 28 (Judge,
What the ITA text is like, Later), Req 26 (Size gate), U25, U27 and Verification "Gates held" of
`specs/naics-artifact-redesign.md`. Read also:

- text→code's spec, `specs/completed/text-to-code-ita-panel.md`: §2 (Stage 4 adds the read
  command), §4.4 (reads and their detail), §4.5 (the reference arms; Stage 4 chooses nothing),
  §4.6 (the outcome panel is reported beside), §5 (roles) and §7 (Stage 4's handoff);
- plan 17's Handoff (`specs/plans/completed/17-text-to-code-ita-panel.md`, "Handoff to later
  stages");
- embedding-data's `specs/evaluation-protocol.md`, Section 1 (`~/Projects/embedding-data`): the
  Evidence table and conditions C1–C6, which bind every read;
- the findings `outcome-panel-splits.md` §5 (the lexical stub) and `geometry-by-dimension.md`
  (`euclidean-d16`: its selected checkpoints, §2's outcome MRRs and the Stage 8 record).

## Global Constraints

Every task's requirements include this section.

**Locators.** MAIN is the main checkout, `/Users/lowell/Projects/naics-embedder`. STORE is the
artifact store, `/Users/lowell/naics-artifacts`. OUT is
`STORE/records/text_to_code/stage4`. SCRATCH is `/tmp/naics_plan20`. EDATA is
`/Users/lowell/Projects/embedding-data`. ITA_RUN is
`EDATA/data/derived/osha/establishment-targets/stage10_ita_establishment_20261116_r1`. WORKTREE is
the execution session's worktree, as `git rev-parse --show-toplevel` prints it. "Stub" is the arm
`lexical-stub` and "d16" the arm `euclidean-d16`. C, K and H are an arm's MRRs on Census phrases,
copied pairs and headline pairs (D1).

### The Exit (`specs/naics-artifact-redesign-roadmap.md`, Stage 4, at 32e91c5), verbatim

> Exit: The selection log holds the reference's ITA validation reads and no opening of the
> sealed part. No margin record was written, and nothing was selected. The finding reports the
> gap with its intervals, and it names the embedding-data commit and the pair table that it
> read.

### The hold (D3)

Plan 19 (economic Stage 3) is running. Until Stage 3's COMPLETE stamp is on `origin/main`:

- no PR of this plan opens, and no branch of it is pushed;
- no real ITA read, dry run or gap read runs;
- the finding is not written.

Pre-flight checks the stamp first. Only this plan's own planning PR, which touches
`specs/plans/` alone, merged before it.

### Pinned values

| What | Value |
|---|---|
| embedding-data | `origin/main` `de7735c3ff5d3891979d4ce050d1bc6e4e5eca76` at plan time. `specs/evaluation-protocol.md` last changed at `63c06664bdcce3867e2ec76c2607683de1c4320d` (copied-pair counts only). The roadmap read the gates at `fd9e495` |
| Run | ITA_RUN. Not `…_r1_replay`: its values are equal, its bytes are not |
| `outputs.json` (detached pin) | `e19b0d58f61f106c3994aa6fe51f4fb71244188145dd90f8cccaa8dc8a8b09df` |
| `ita_text_pairs.parquet` | 644,265 rows, `60aad7642961e70b987e9d3222d660147861458e12ac52d71d2739d2d21e1e1c` |
| `ita_text_pair_codes.parquet` | 771,971 rows, `c8084e4ad297e3f78391e035ebe51119b07982639ae4daf47f1d47caeb1f8916` |
| Headline pairs (protocol) | 239,364: 221,176 copied (11,920 keys) and 18,188 not copied (7,567 keys) |
| ITA settings | `conf/data/ita_panel.yaml`, unchanged. Split fingerprint `8602ceb243722dc64e0a29a9ba444fe824baf209a848964d7ddf130a9db5cc7c`. r = 0.05 |
| Bundle | `MAIN/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/manifest.json`, named by MAIN's gitignored `conf/local.yaml` |
| Stub code texts | `MAIN/data/naics_descriptions.parquet`, `fe8c54e36efb7470e46122c0071e16c03c3dba1c909073c84c91ec998a0fdc36` (the d16 spec's `descriptions_sha256`) |
| Stub on the outcome panel | MRR 0.6165 on 4,042 validation queries (finding: outcome-panel-splits §5, 2026-09-24). That read came before plan 7 rebuilt the descriptions as `fe8c54e3` (2026-09-26), so this plan reports 0.6165 beside its own read and does not check against it |
| Outcome panel | fingerprint `05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a`, validation data fingerprint `ca91f2e0e361515070fee53a909bbf51ecae6019995b85a8062745185982e81f` (the Stage 8 record's `panels`) |
| Stage 8 record | `STORE/records/stage8/euclidean-d16.json`, `3d5a80d5a3ac5466e10b019773fa09a21e2ad0592af32d7b22273fea3e1bab2c`. Its five `scores` objects hash as recorded (checked at plan time) |
| `uv.lock` | `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948` |
| Bootstrap | 10,000 replicates, seed 20260924 (`conf/data/decision.yaml`); level 0.95 |
| Selection log | `MAIN/logs/selection_log.jsonl`: 159 records, `51aef24099f0303c218271fc9ab110c3f1a8e7a56198d97eab3c84a0731a2a41` at plan time. Plan 19 appends 118 growth reads before this plan runs |

d16's five seeds read from their Stage 8 run directories,
`MAIN/checkpoints/stage8-euclidean-d16-sN/`. Each holds the selected checkpoint, the table and the
table's `_provenance.json`. The store's
copies have no provenance beside them, so `ArmEncoder.from_files` cannot read those. Every SHA-256
below equals the Stage 8 record's (checked at plan time):

| Seed | Checkpoint | SHA-256 | Table | SHA-256 |
|---|---|---|---|---|
| 1 | `epoch=039.ckpt` | `24171c64d390dc59cb8b298b1ee35b3c26cc4b5dd27e46bbf50742efba7efd50` | `arm_table_epoch=039.parquet` | `b67f487ed4e5e337cfb1202a0230aaa62dd36d0ad2649b2d77f8bca3cb035fab` |
| 2 | `epoch=039.ckpt` | `0f592e2ec514e2aad488d7b6a0668dda3a1868d40ef55a04a811ee626a36eb40` | `arm_table_epoch=039.parquet` | `3a88c77e08fe8eaa32959f61a25864e8d4fee57b44b3a795c9fb1ccdcc42345d` |
| 3 | `epoch=039.ckpt` | `09692c6f82cadfe8d17dc8b849cd2479013bb1f74fbfa4d94a06578ac416203a` | `arm_table_epoch=039.parquet` | `3054d864d9cf42562fe9781d8b513dfc81267096399daf4b399c8ae1f9682f48` |
| 4 | `epoch=038.ckpt` | `0d0317e723bc2cd4711abe088f671e97deadae94f5402ed7f10954ffceef1b43` | `arm_table_epoch=038.parquet` | `87b9200c3ef7ba9c697bcd319f37c26d75e063bfac4fe20023fc17ef0d88c6fd` |
| 5 | `epoch=039.ckpt` | `68c07a914b65aa745a9e883062b371a57ac3a6d945993ba638abaab15782dc90` | `arm_table_epoch=039.parquet` | `107c85100fa5ba88bdf0bf7a37d5fe000c2f9a3aba0d57d2c162ffe25e12f75c` |

### Project rules

- **Real data only behind the gates.** Pre-flight hashes files only. Gate A (Task 6) loads the
  real tables but logs nothing. Nothing scores a pair before Gate B's go (Task 7). Each approval
  comes from the user in chat, at that step.
- **One read per arm and seed.** A read appends its record before it encodes anything, so a read
  that fails may already be logged. Never re-run a read without the user's ruling.
- **The sealed part stays sealed.** Nothing calls `ItaPanel.open_sealed`. No code in this plan
  reads the sealed part (C3).
- **Nothing is selected.** Nothing calls `choose_reference`, `fix_ita_margin` or
  `compare_to_reference`. The finding names no reference (spec §4.5).
- **Arm names.** A read's `arm` is exactly `lexical-stub` or `euclidean-d16`, and its `role` is
  `reference`. The seed is a key of its own. `check_ita_margin_first` counts a reference read of
  any other arm name as a candidate read, which would block Stage 5's margin record for good.
- **The selection log is append-only.** Never edit, truncate, move, copy into the repository or
  delete `MAIN/logs/selection_log.jsonl`.
- **MAIN.** Never check out a branch there, and never edit its tracked files. Its only writes are
  Task 5's fast-forward and `uv sync`, the reads' log records, its gitignored `logs/tools_*.log`
  files and Integration's ledger copy. From Task 5 Step 3 until Task 8 passes, nothing merges into
  MAIN: the reads run at one commit, READ_SHA.
- **The repository is public.** No committed file holds a `representative_string`, a
  `description_key`, a pair id or a per-pair row. Those stay in STORE. Every published score
  carries C4's universe statement and C3's label: a development measurement, not evaluation.
- **The lock.** Never change `uv.lock`, and never run `uv lock`.
- **Absolute paths.** Pass every path as an absolute path, never with `~`.
- **Git.** Cut each branch from `origin/main`. Never push `main` or a `refs/held/*` ref, and
  never use `--no-verify`. If a git command stalls, rerun it as `git -c core.fsmonitor=false …`.
  Commits end with `Co-Authored-By: <the executing model> <noreply@anthropic.com>`, and PR
  bodies with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Merging.** Main has no required checks. Merge a PR only once every check that
  `gh pr checks` lists has passed on its head, with
  `gh pr merge <n> --merge --match-head-commit <sha> --delete-branch`.
- **Style.** Python: single quotes, semantic dividers, `./scripts/format_code.sh` on touched files,
  never `ruff format`. Markdown: lines within 100 characters outside code blocks and tables.
- **Bash tool.** It runs zsh. Prefix every command with `cd <dir> &&`, and gate a check on
  `; echo "exit=$?"`, never on a pipe's tail. Run a command that may take over 2 minutes in the
  background, and wait for its completion notice.

## Review Focus

1. **A read logged under another arm name or role.** Stage 5's margin-first check must still
   pass after Stage 4's reads. Test: Task 2,
   `test_stage_4_reads_leave_the_margin_first_check_open`.
2. **The replay run's tables, or a rebuilt table.** Refused before the panel loads, with nothing
   logged. Tests: Task 1, `test_tables_of_another_run_are_refused`; Task 2,
   `test_ita_read_refuses_unpinned_tables_and_logs_nothing`.
3. **A dry run.** It logs nothing, writes no record and stores nothing. Test: Task 2,
   `test_ita_read_dry_run_logs_and_writes_nothing`.
4. **Reads that do not pair:** two runs, two table builds, a d16 seed whose checkpoint is not the
   Stage 8 record's, or a stub outcome read on another outcome split. No gap record is written.
   Test: Task 3, `test_gap_seeds_refuse_reads_that_do_not_pair`.
5. **A code that one panel lacks**, such as an ITA code with no outcome validation query. It is
   left out of the code-matched gap and counted, and nothing fails. Test: Task 3,
   `test_the_matched_gap_skips_codes_one_panel_lacks`.

## Decisions

- **D1** (owner, planning question, 2026-11-18): "Outcome + code check (Recommended)". For each
  reference arm: C = MRR on the outcome panel's held-out Census index entries (validation), H =
  ITA validation headline MRR, K = ITA validation copied MRR. Report C − H and C − K (the gap on
  headline and copied pairs apart), K − H, and a code-matched C − H over the codes both panels
  hold, each with a 95 % bootstrap interval. One logged stub outcome read; d16 reuses Stage 8's
  stored outcome scores. Applied in: Tasks 2, 3, 7, 9.
- **D2** (owner, planning question, 2026-11-18): "Tie to r = 0.05 (Recommended)". The finding
  recommends a later ITA training part when, for both reference arms, the lower bound of the
  95 % interval on the code-matched headline gap exceeds 0.05 × that arm's Census-phrase MRR C.
  That is a gap larger than the margin a candidate must win by (U38). Applied in: Tasks 3, 9.
- **D3** (owner, roadmap session, 2026-11-18): "Yes, same hold". Text→code's code PRs, real ITA
  reads and Stage 4 finding wait for Stage 3's COMPLETE stamp. Applied in: Global Constraints,
  Pre-flight.
- **D4** (owner, plan review, 2026-11-18): "Approve plan (Recommended)": the plan as written,
  P1–P10 included, at PR #169. Applied in: all tasks.
- **D5** (owner, plan review, 2026-11-18): "Inline (Recommended)": executing-plans in a fresh
  Sonnet session. Applied in: Execution.

This plan's choices, approved by D4:

- **P1:** a command, not a scratch script. Spec §2 and §7 give Stage 4 the read command. The logic
  is in `decision/ita_reads.py` and `decision/ita_gap.py`. `tools.py` holds two thin adapters,
  since plans 18 and 19 also edit that file. Applied in: Tasks 2, 3.
- **P2:** the C1 pin is a committed file of its own, `conf/data/ita_evidence.yaml`.
  `conf/data/ita_panel.yaml` and `ItaPanelSettings` stay unchanged, since they feed the split.
  The check also requires the tables' directory to be named for the run, and `outputs.json`
  beside them to match its pin. Under C5, a new run is a new pin, and the owner amends the
  protocol first. Applied in: Task 1.
- **P3:** one run id, RUN = `ita-gap/<uuid4 hex>`, for all six reads. A read's detail is `arm`,
  `role` (`reference`), `run`, and for d16 `seed`, `checkpoint` and `table`. The outcome read
  beside carries the same detail. Applied in: Tasks 2, 7.
- **P4:** d16 reads from its Stage 8 run directories, and every file is pinned to the Stage 8
  record. Its C comes from the record's stored `scores` (panel `outcome`, statistic `mrr`). The
  stub's C comes from one logged outcome read beside its ITA read (`--outcome`). Its outcome
  fingerprints must be the pinned ones, which fix the split and the candidates. The stub is the
  current code over the pinned descriptions, as Stage 5's `fix_ita_margin` will read it, so §5's
  older 0.6165 is reported, not checked. Applied in: Tasks 2, 3, 7, 9.
- **P5:** the bootstrap. Each stratum has its own unit draws: H on `ita_text` (so it equals
  `arm_replicates`), K on `ita_text/copied`, C on `outcome`, and the matched pair on
  `ita_text/matched`. One seed draw per arm, on `(ita_text, arm)`, is shared by every stratum.
  Copied and headline keys are disjoint (protocol counts), so the strata do not overlap. Applied
  in: Task 3.
- **P6:** the code-matched gap gives each code one weight. M holds the codes that report at least
  one headline validation pair and hold at least one outcome validation query. A pair with
  several codes counts under each, as `ItaResult.per_code` counts it. Applied in: Task 3.
- **P7:** two gates, as plan 19 has. Gate A covers the read-only checks and the six dry runs. Gate
  B covers the six reads, with a check after each. Applied in: Tasks 6, 7.
- **P8:** plan 18 is independent. Stage 4 consumes `ArmEncoder.from_files` alone, whose signature
  plan 18 keeps. If plan 18 lands first, rebase PR A on it. The new commands escape Rich markup
  from the start. Applied in: Tasks 2–4.
- **P9:** the outputs go to OUT, outside the repository: six read records, their stored frames
  and `gap.json`. The finding carries aggregates only. Applied in: Tasks 7–9.
- **P10:** inline execution (executing-plans). The gates need the user, and a subagent cannot ask
  them. Applied in: Execution.

---

## Workspace

- **Ledger:** `WORKTREE/logs/plan20_execution.md`, ignored by git. Write every command, its
  output and any deviation there. Integration copies it to `MAIN/logs/`.
- **Branches:** PR A is `claude/stage-4-ita-read-tooling` and PR B is
  `claude/stage-4-text-to-code-gap`. Cut each with
  `git -c core.fsmonitor=false checkout --no-track -B <branch> origin/main`.
- **Scratch:** SCRATCH, for this plan's scripts and command outputs. Integration deletes it.

## Stop-and-ask conditions

Stop, report and wait for your human partner when any of these happens:

- Stage 3's COMPLETE stamp is not on `origin/main` (D3).
- Pre-flight finds an open PR that touches a `src/`, `conf/data/` or `tests/` file that this plan
  creates or modifies.
- embedding-data's Section 1 Evidence values differ from the Pinned values (C5), or its Stage 10
  amendment or V9 decision changed since `fd9e495`.
- A pinned SHA-256 differs from the file on disk, or `uv sync --locked --check` fails.
- PR A's CI fails twice, or a review finding would change `conf/data/ita_panel.yaml`.
- Task 5's guard fails: MAIN is not on `main`, not clean, or has commits that `origin/main` lacks.
- The user declines Gate A or Gate B, or answers anything but a clear yes.
- Another session reads a panel into MAIN's log, or merges, during Tasks 6–8.
- A dry run or a read exits non-zero, or the panel refuses the real tables. A refusal means that
  the inputs disagree with spec §4.1. Report it, and never patch around it.
- A check script exits 1 after a read. Record what it printed, and never re-run the read.
- The stub's outcome fingerprints are not the pinned ones, or its outcome read has other than
  4,042 queries.
- MAIN's HEAD moved during Tasks 6–8 (the drift guard, Task 7 Step 2).
- An output path exists.
- `log_state.py` finds an `ita_text` `open` or `reopen` record, or a run id that starts with
  `ita-margin/`.
- `tools ita-gap` exits non-zero.
- A step would do anything the Project rules forbid.

## File structure

| Action | Path | Task | Responsibility |
|---|---|---|---|
| Create | `conf/data/ita_evidence.yaml` | 1 | The run, the protocol commit and the three pins |
| Create | `src/naics_embedder/panels/ita_evidence.py` | 1 | `ItaEvidence`, its loader and the C1 check |
| Create | `src/naics_embedder/decision/ita_reads.py` | 2 | `ItaReferenceRead`, `OutcomeBeside`, `read_reference` |
| Create | `src/naics_embedder/decision/ita_gap.py` | 3 | `GapSeed`, `gap_seeds`, `arm_gap`, `ArmGap`, `ItaGapRecord`, `gap_record` |
| Modify | `src/naics_embedder/cli/commands/tools.py` | 2, 3 | `tools ita-read` and `tools ita-gap`, after `outcome-panel` |
| Modify | `tests/fixtures/ita.py` | 1–3 | `ita_run_dir`, `outcome_beside_panel`, `IdentifiedStubEncoder`, `gap_fixture` |
| Create | `tests/unit/test_ita_evidence.py`, `test_ita_reads.py`, `test_ita_gap.py` | 1–3 | One file per module |
| Modify | `tests/unit/test_cli_commands.py` | 2, 3 | The two commands |
| Modify | `docs/usage.md`, `docs/api/ita_panel.md`, `docs/api/decision.md`, `CLAUDE.md`, `README.md`, `tests/README.md` | 4 | Commands, API pages, tree and counts |
| Create | `specs/findings/text-to-code-gap.md` | 9 | The finding |
| Modify | `CLAUDE.md` | 9 | One sentence: the gap and the finding |
| Modify | `specs/naics-artifact-redesign-roadmap.md`, `specs/deferred_items.md`, `specs/history.md` | Completion | Stamp, ticks, history entry |
| Move | `specs/plans/20-text-to-code-reference-read.md` | Completion | Markup and Handoff, then `specs/plans/completed/` |

No other source, config or lock file changes. These stay identical to BASE: `panels/ita.py`,
`panels/ita_split.py`, `panels/ita_scoring.py`, `panels/outcome.py`, `panels/decoding.py`,
`panels/selection_log.py`, `decision/text_to_code.py`, `decision/resampling.py`,
`decision/rule.py`, `decision/records.py`, `decision/decide.py`, `decision/economic.py`,
`conf/data/ita_panel.yaml` and `uv.lock`.

## Pre-flight (controller, inline, before Task 1)

- [x] **Step 1: The hold and the workspace**

```bash
git -c core.fsmonitor=false fetch origin
git cat-file -e origin/main:specs/plans/20-text-to-code-reference-read.md; echo "exit=$?"
git show origin/main:specs/naics-artifact-redesign-roadmap.md | grep -n 'Stage 3: COMPLETE'
git rev-parse origin/main
```

Expected: `exit=0`, and one `Stage 3: COMPLETE (…)` stamp line. Without the stamp, stop (D3).
Record the SHA as BASE in the ledger.

- [x] **Step 2: Parallel work**

```bash
git -c core.fsmonitor=false worktree list
gh pr list --repo lowmason/naics-embedder --state open --json number,title,files --jq '.[] | {number, title, files: [.files[].path]}'
git log --oneline 32e91c5..origin/main -- src/naics_embedder/panels src/naics_embedder/decision src/naics_embedder/cli/commands/tools.py conf/data tests/fixtures/ita.py uv.lock
```

Expected: no open PR touches a `src/`, `conf/data/` or `tests/` path in File structure. An open
PR that touches only its docs, count or `specs/` files is no stop: rebase on it if it lands first.
List the commits, which are plan 19's and any others, in the ledger. Note any plan 18 change to
`tools.py` or `text_model/export.py`. Worktrees under `~/Projects/copilot-worktrees/` are review
tooling, not sessions.

- [x] **Step 3: Re-read the gates in embedding-data**

```bash
cd /Users/lowell/Projects/embedding-data && git -c core.fsmonitor=false fetch origin && git rev-parse origin/main
cd /Users/lowell/Projects/embedding-data && git diff --stat fd9e495 origin/main -- specs/evaluation-protocol.md specs/completed/ita-establishment-targets.md specs/findings/v9-ita-naics-minimum-counts.md
cd /Users/lowell/Projects/embedding-data && git show origin/main:specs/evaluation-protocol.md | sed -n '/^## Section 1/,/^### Assignment/p'
```

Expected: Section 1's Run, pin, row counts and two table SHA-256 equal the Pinned values. At plan
time, the only change since `fd9e495` was `63c0666`'s copied-pair counts. Record the SHA as
EDATA_SHA. Read each changed file in full. A change to the Evidence table, to C1–C6, to the Stage
10 amendment's validation-part allowance (G2) or to the V9 minimum (G3) is a stop (C5).

- [x] **Step 4: The files on disk**

```bash
cd /Users/lowell/Projects/embedding-data/data/derived/osha/establishment-targets/stage10_ita_establishment_20261116_r1 && shasum -a 256 outputs.json ita_text_pairs.parquet ita_text_pair_codes.parquet
cd /Users/lowell/Projects/naics-embedder && shasum -a 256 uv.lock data/naics_descriptions.parquet /Users/lowell/naics-artifacts/records/stage8/euclidean-d16.json checkpoints/stage8-euclidean-d16-s*/epoch=0*.ckpt checkpoints/stage8-euclidean-d16-s*/arm_table_epoch=0*.parquet
```

Expected: every SHA-256 is the pinned one. Hashing reads no pair.

- [x] **Step 5: The ledger and SCRATCH**

```bash
mkdir -p logs /tmp/naics_plan20
```

Write `logs/plan20_execution.md` with the Write tool: a title, BASE, EDATA_SHA, the date and the
outputs of Steps 1–4.

## Tasks

### Task 1: The evidence pin and C1

Nothing in this task reads real data.

**Files:**
- Create: `conf/data/ita_evidence.yaml`, `src/naics_embedder/panels/ita_evidence.py`,
  `tests/unit/test_ita_evidence.py`
- Modify: `tests/fixtures/ita.py`

**Interfaces:**
- Consumes: `write_ita_tables`, `ita_tables` (`tests/fixtures/ita.py`); `sha256_file`
  (`supervision/artifacts.py`).
- Produces:
  - `OUTPUTS_FILE = 'outputs.json'`
  - `class ItaEvidence(BaseModel)` (`extra='forbid'`, frozen): `protocol_commit: str` (40
    lowercase hex), `run: str` (no whitespace, non-empty), `outputs_sha256: str`,
    `pairs_sha256: str`, `pair_codes_sha256: str` (each 64 lowercase hex).
  - `load_ita_evidence(path: Union[str, Path]) -> ItaEvidence`
  - `check_ita_run(pairs_path: Union[str, Path], pair_codes_path: Union[str, Path], evidence:
    ItaEvidence) -> None`. It raises `ValueError` when: the two tables are not in one directory;
    that directory's name is not `evidence.run`; the `OUTPUTS_FILE` beside them is missing; or
    the SHA-256 of that file or of either table is not its pin. A hash message names the file,
    its SHA-256, the pin and "C1". It raises `FileNotFoundError` when a table is missing.
  - In `tests/fixtures/ita.py`: `ita_run_dir(directory: Path, pairs: pl.DataFrame, codes:
    pl.DataFrame, run: str = 'stage10_fixture_r1') -> Tuple[Path, Path, ItaEvidence]`. It writes
    both tables and an `outputs.json` under `directory / run`, and returns their paths and the
    evidence that pins them.

`conf/data/ita_evidence.yaml` holds the Pinned values: `protocol_commit`
`63c06664bdcce3867e2ec76c2607683de1c4320d`, `run` `stage10_ita_establishment_20261116_r1`, and
the three SHA-256. Its header comment says that the file copies embedding-data's
`specs/evaluation-protocol.md` Section 1 Evidence table. It also says that C1 checks the files
before every read, and that under C5 a new run waits for the owner's amendment there before this
file changes.

- [x] **Step 1: Write the failing tests** in `tests/unit/test_ita_evidence.py`

```python
def test_the_committed_pin_is_the_protocols():
    evidence = load_ita_evidence('conf/data/ita_evidence.yaml')
    assert evidence.run == 'stage10_ita_establishment_20261116_r1'
    assert evidence.pairs_sha256 == '60aad764…'       # the full Pinned values, all five fields
    ...

def test_matching_tables_pass(tmp_path):
    pairs, codes, evidence = ita_run_dir(tmp_path, *ita_tables())
    check_ita_run(pairs, codes, evidence)   # no error

def test_a_table_of_other_bytes_is_refused(tmp_path):
    # rewrite the pair table with its rows reversed: same values, other bytes
    with pytest.raises(ValueError, match='C1'): ...

def test_tables_of_another_run_are_refused(tmp_path):
    # the same three files copied under 'stage10_fixture_r1_replay', with the original evidence
    with pytest.raises(ValueError, match='run'): ...

def test_a_missing_or_other_outputs_file_is_refused(tmp_path): ...

def test_an_evidence_file_with_an_unknown_key_or_a_short_hash_is_refused(tmp_path): ...
```

- [x] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_ita_evidence.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'naics_embedder.panels.ita_evidence'`.

- [x] **Step 3: Implement the module, the config file and the fixture helper**

- [x] **Step 4: Run them to see them pass**

Run: `uv run pytest tests/unit/test_ita_evidence.py tests/unit/test_ita_panel.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
./scripts/format_code.sh src/naics_embedder/panels/ita_evidence.py tests/unit/test_ita_evidence.py tests/fixtures/ita.py
git add conf/data/ita_evidence.yaml src/naics_embedder/panels/ita_evidence.py tests/unit/test_ita_evidence.py tests/fixtures/ita.py
git commit -m "feat(panels): pin the ITA run and check its tables before a read (C1)"
```

### Task 2: Reference reads and `tools ita-read`

Nothing in this task reads real data.

**Files:**
- Create: `src/naics_embedder/decision/ita_reads.py`, `tests/unit/test_ita_reads.py`
- Modify: `src/naics_embedder/cli/commands/tools.py` (a new command after `outcome-panel`),
  `tests/fixtures/ita.py`, `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes: `ItaEvidence`, `load_ita_evidence`, `check_ita_run` (Task 1); `ItaPanel`
  (`score_logged`, `pairs`, `fingerprint`, `data_fingerprint`), `ItaPart`, `load_ita_settings`;
  `OutcomePanel` (`score_logged`, `fingerprint`, `data_fingerprint`, `from_bundle`);
  `outcome_scores` (`decision/scores.py`); `ArtifactStore.put_frame`, `ArtifactRef`,
  `write_record`; `LEXICAL_STUB`, `EUCLIDEAN_D16`, `REFERENCE_ARMS`, `fix_ita_margin`,
  `check_ita_margin_first` (`decision/text_to_code.py`, the last two in a test only).
- Produces:
  - `READ_ROLE = 'reference'`
  - `class OutcomeBeside(BaseModel)`: `fingerprint: str`, `data_fingerprint: str`, `summary:
    Dict[str, Any]`, `scores: ArtifactRef` (the read's `outcome_scores` frame).
  - `class ItaReferenceRead(BaseModel)` (`extra='forbid'`, frozen, `allow_inf_nan=False`):
    `kind: Literal['ita_reference_read'] = 'ita_reference_read'`, `arm: Literal['lexical-stub',
    'euclidean-d16']`, `seed: Optional[int]` (≥ 1), `run: str`, `purpose: str`, `evidence:
    ItaEvidence`, `split_fingerprint: str`, `data_fingerprint: str`, `headline: Dict[str, Any]`,
    `copied: Dict[str, Any]`, `phrasing: Dict[str, Any]`, `counts: Dict[str, int]`, `per_pair:
    ArtifactRef`, `checkpoint_sha256: Optional[str]`, `table_fingerprint: Optional[str]`,
    `outcome: Optional[OutcomeBeside]`, `log_records: List[Dict[str, Any]]`, `created_at:
    datetime` (UTC).
  - `read_reference(panel: ItaPanel, encoder: QueryCodeEncoder, store: ArtifactStore, evidence:
    ItaEvidence, *, arm: str, seed: Optional[int], run: str, purpose: str, outcome_panel:
    Optional[OutcomePanel] = None) -> ItaReferenceRead`
  - In `tests/fixtures/ita.py`: `outcome_beside_panel(log: SelectionLog) -> OutcomePanel`, an
    `OutcomePanel(roles, ITA_CANDIDATES, log)` whose validation entries' texts are keys of
    `ASSIGNED`, with rows that `validate_index_role_table` accepts (follow
    `tests/unit/test_outcome_panel.py`'s role rows); and `IdentifiedStubEncoder`, an
    `AssignedStubEncoder` with `checkpoint_sha256 = 'c' * 64` and `table_fingerprint = 't' * 64`.

`read_reference` raises `ValueError` before any log record when:

- `arm` is not in `REFERENCE_ARMS`;
- the stub has a seed, or d16 has none or one below 1;
- `run` or `purpose` is blank;
- d16's encoder lacks `checkpoint_sha256` or `table_fingerprint`.

Then it reads the ITA validation part with `panel.score_logged`. The detail is `{'arm', 'role':
READ_ROLE, 'run'}`, plus `'seed'`, `'checkpoint'` and `'table'` for d16. When `outcome_panel` is
given, it then reads the outcome validation split with `outcome_panel.score_logged(encoder,
IndexRole.VALIDATION, purpose, distance=<the encoder's distance, cosine when it has none>,
detail=<the same detail>)`. It stores `result.per_pair` as `ita_per_pair.parquet`, and the
outcome read's `outcome_scores(per_query)` as `outcome_scores.parquet`. It returns the record,
whose `log_records` are the records that its reads appended, in order. It writes no file but the
store's.

`tools ita-read` options: `--arm` (required), `--seed`, `--checkpoint`, `--table` (d16 only, all
three required for it), `--pairs`, `--pair-codes`, `--run`, `--purpose`, `--store` and `--output`
(all required), `--outcome/--no-outcome` (default `--no-outcome`), `--descriptions` (the stub's
code texts, default the download config's `output_parquet`), `--evidence` (default
`conf/data/ita_evidence.yaml`), `--settings` (default `conf/data/ita_panel.yaml`), `--log` (default
the outcome-panel config's `selection_log`), `--config` (default `conf/config.yaml`), `--dry-run`,
and trailing `key=value` overrides. It works in this order, and each refusal exits 1 with an
escaped (`rich.markup.escape`) message and logs nothing:

1. A blank `--purpose` or `--run`; an arm outside `REFERENCE_ARMS`; d16 options on the stub or
   missing on d16. Then `_require_new_record(--output)`.
2. `check_ita_run` (C1), before any table loads.
3. The config, the bundle, then `ItaPanel.from_files`, then `OutcomePanel.from_bundle` under
   `--outcome`. Then the encoder: `LexicalTrigramEncoder(code_texts_from_descriptions(…))` for
   the stub, and `ArmEncoder.from_files(checkpoint, table, bundle, code_token_config(cfg),
   device=pick_device('auto'))` for d16.
4. Under `--dry-run`, it prints `Dry run: every check passed for <arm>[ seed <n>]`. It also prints
   the validation part's pair count (`panel.pairs`' rows whose `part` is `validation`), its data
   fingerprint, the split fingerprint, the encoder's type and its distance. Then it exits 0. It
   logs nothing, stores nothing and writes no record.
5. Otherwise `read_reference`, then `write_record(record, --output)`. It prints the headline,
   copied and phrasing MRR, top-1 and pair counts, the counts, and the outcome summary if one was
   read.

It catches `OSError`, `ValueError`, `ValidationError` and `SealedSplitError`.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_ita_reads.py`, with `ItaPanel(*ita_tables(), ITA_CANDIDATES, ITA_SETTINGS,
SelectionLog(tmp_path / 'log.jsonl'))`, a store under `tmp_path` and Task 1's evidence:

```python
def test_a_stub_read_logs_one_validation_read_with_the_reference_detail(): ...
    # record.arm == 'lexical-stub', record.seed is None, record.outcome is None
    # [entry] = log.records(); entry['panel'] == 'ita_text', entry['split'] == 'validation'
    # entry['detail'] holds {'arm': 'lexical-stub', 'role': 'reference', 'run': RUN}
    # record.log_records == [entry]; record.headline equals a direct panel.score's headline

def test_a_d16_read_logs_its_seed_checkpoint_and_table(): ...
    # IdentifiedStubEncoder; detail['seed'] == 3, detail['checkpoint'] == 'c' * 64,
    # detail['table'] == 't' * 64

def test_an_outcome_read_beside_logs_a_second_record_with_the_same_detail(): ...
    # outcome_beside_panel; two records, ita_text then outcome, equal arm/role/run detail;
    # record.outcome.fingerprint == outcome_panel.fingerprint;
    # store.read_frame(record.outcome.scores) == outcome_scores(direct per_query)

@pytest.mark.parametrize('arm, seed', [('euclidean-d16-s1', 1), ('lexical-stub', 2),
                                       ('euclidean-d16', None), ('euclidean-d16', 0)])
def test_bad_arms_or_seeds_are_refused_before_logging(arm, seed): ...   # log stays empty

def test_a_blank_run_or_purpose_is_refused_before_logging(): ...

def test_the_per_pair_frame_is_stored_and_the_record_round_trips(): ...

def test_stage_4_reads_leave_the_margin_first_check_open(tmp_path):
    # a stub read and a d16 read (seed 1) through read_reference, then
    # fix_ita_margin(panel, {'lexical-stub': [(None, stub)], 'euclidean-d16': [(1, d16)]}, 0.05,
    #                'ita-margins', purpose='fix the margin'),
    # then check_ita_margin_first(record, panel.log) raises nothing
```

In `tests/unit/test_cli_commands.py`, monkeypatch `tools_cli.ItaPanel.from_files` to return the
fixture panel on the given log, `tools_cli.OutcomePanel.from_bundle` to return
`outcome_beside_panel`, `tools_cli.ArmEncoder.from_files` and `tools_cli.LexicalTrigramEncoder` to
return the fixture encoders, `require_valid_supervision_bundle` to return `validated_bundle`, and
`pick_device` to return `'cpu'`. Use Task 1's `ita_run_dir` tables and evidence file.

```python
def test_ita_read_writes_the_record_and_logs_the_reads(): ...
    # stub with --outcome: exit 0, two log records, the record file loads as ItaReferenceRead

def test_ita_read_dry_run_logs_and_writes_nothing(): ...
    # exit 0; 'Dry run: every check passed for lexical-stub' in the output; the log file is
    # absent or empty; --output absent; nothing under the store's objects/

def test_ita_read_refuses_unpinned_tables_and_logs_nothing(): ...
    # evidence with another pairs_sha256: exit 1, 'C1' in output, ItaPanel.from_files never
    # called, no log records

def test_ita_read_refuses_d16_options_on_the_stub_and_their_absence_on_d16(): ...

def test_ita_read_refuses_an_existing_output_before_reading(): ...

def test_ita_read_prints_a_bracketed_error_literally(): ...
    # a ValueError('[/x] bad') from the panel: exit 1, '[/x] bad' printed, no MarkupError
```

- [x] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_ita_reads.py tests/unit/test_cli_commands.py -q -k 'ita'`
Expected: FAIL with `ModuleNotFoundError: No module named 'naics_embedder.decision.ita_reads'`.

- [x] **Step 3: Implement `decision/ita_reads.py`, the fixture helpers and `tools ita-read`**

> Deviation: under `--outcome` the dry run also prints the outcome fingerprints, and a d16 seed below 1 is refused up front (ledgered rulings). Final-review fixes then made a dry run create no directory (e6e9024), check the stub's descriptions against the bundle (57d9669) and test the bundle-failure clause (30f7741).

- [x] **Step 4: Run them to see them pass**

```bash
uv run pytest tests/unit/test_ita_reads.py tests/unit/test_cli_commands.py tests/unit/test_ita_panel.py tests/unit/test_decision_text_to_code.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
./scripts/format_code.sh src/naics_embedder/decision/ita_reads.py src/naics_embedder/cli/commands/tools.py tests/fixtures/ita.py tests/unit/test_ita_reads.py tests/unit/test_cli_commands.py
git add src/naics_embedder/decision/ita_reads.py src/naics_embedder/cli/commands/tools.py tests/fixtures/ita.py tests/unit/test_ita_reads.py tests/unit/test_cli_commands.py
git commit -m "feat(decision): read text→code's reference arms on the ITA validation part"
```

### Task 3: The gap and `tools ita-gap`

Nothing in this task reads real data.

**Files:**
- Create: `src/naics_embedder/decision/ita_gap.py`, `tests/unit/test_ita_gap.py`
- Modify: `src/naics_embedder/cli/commands/tools.py` (a new command after `ita-read`),
  `tests/fixtures/ita.py`, `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes: `ItaReferenceRead`, `read_reference` (Task 2); `ItaEvidence` (Task 1); `ArmRecord`,
  `Estimate`, `read_record`, `write_record` (`decision/records.py`); `ArtifactStore`;
  `PanelItems`, `unit_draws`, `seed_draws`, `replicate_statistics`, `point_statistic`,
  `percentile_interval` (`decision/resampling.py`); `ITA_PANEL`; `OUTCOME_PANEL`
  (`panels/outcome.py`); `REFERENCE_ARMS`, `arm_replicates` (the latter in a test only);
  `synthetic_arm`, `spec` (`tests/fixtures/decision.py`).
- Produces:
  - `GAP_LEVEL = 0.95`; `COPIED_STREAM = 'ita_text/copied'`; `MATCHED_STREAM =
    'ita_text/matched'`.
  - `@dataclass(frozen=True) class GapSeed`: `seed: Optional[int]`, `ita: pl.DataFrame` (a read's
    per-pair frame), `census: pl.DataFrame` (`unit` = code, `item` = query id, `value` =
    reciprocal rank, one row per outcome validation query).
  - `gap_seeds(reads: Sequence[ItaReferenceRead], store: ArtifactStore, stage8: ArmRecord) ->
    Dict[str, List[GapSeed]]`
  - `class ArmGap(BaseModel)` (`extra='forbid'`, frozen, `allow_inf_nan=False`): `arm: str`,
    `n_seeds: int`, `census`, `copied`, `headline`, `census_matched`, `headline_matched`:
    `Estimate`; `n_matched_codes: int`, `n_unmatched_codes: int`; `headline_gap` (C − H),
    `copied_gap` (C − K), `wording_gap` (K − H), `matched_headline_gap`: `Estimate`;
    `threshold: float` (r × C's point); `large: bool`.
  - `arm_gap(arm: str, seeds: Sequence[GapSeed], *, r: float, replicates: int, bootstrap_seed:
    int) -> ArmGap`
  - `class ItaGapRecord(BaseModel)` (same config): `kind: Literal['ita_gap'] = 'ita_gap'`,
    `run: str`, `evidence: ItaEvidence`, `split_fingerprint: str`, `data_fingerprint: str`,
    `outcome_fingerprint: str`, `outcome_data_fingerprint: str`, `stage8_sha256: str`,
    `read_sha256: Dict[str, str]` (each read record's file name to its SHA-256), `replicates:
    int`, `bootstrap_seed: int`, `level: float`, `r: float`, `arms: List[ArmGap]` (the stub
    first), `recommend_training_part: bool`, `created_at: datetime`.
  - `gap_record(read_paths: Sequence[Path], store: ArtifactStore, stage8_path: Path, *, r: float,
    replicates: int, bootstrap_seed: int) -> ItaGapRecord`
  - In `tests/fixtures/ita.py`: `gap_fixture(store, directory) -> Tuple[List[Path], Path]`. It
    writes six read records (the stub with an outcome beside, d16 seeds 1–5) and a Stage 8
    `ArmRecord` built with `synthetic_arm`, its runs renumbered to seeds 1–5 with `model_copy`.
    Its outcome fingerprints and its runs' checkpoint SHA-256 match the d16 reads.

`gap_seeds` raises `ValueError`, and nothing is written, when:

- the arms are not exactly `REFERENCE_ARMS`;
- the reads name more than one `run`, `evidence`, `split_fingerprint` or `data_fingerprint`;
- the stub has other than one read, or its read has no `outcome`;
- a d16 read has an `outcome`;
- d16's seeds repeat or are not the record's seeds;
- a d16 read's `checkpoint_sha256` or `table_fingerprint` is not that seed's in the record
  (`run.checkpoint.sha256`, `run.table.matrix_fingerprint`);
- the stub's `outcome.fingerprint` and `outcome.data_fingerprint` are not the record's
  `panels.outcome` and `panels.outcome_data`.

The stub's census rows come from its stored outcome scores. d16's come from each record run's
`scores` (`store.read_frame` checks its hash), filtered to panel `outcome` and statistic `mrr`.

`arm_gap` is the algorithm below (P5, P6). Means within a code use `math.fsum`, as
`ita_scoring._mean` does.

```text
seeds    = seed_draws(ITA_PANEL, arm, n_seeds, replicates, bootstrap_seed)   # shared below
stratum(frames, stream):                 # frames: one (unit, item, value) frame per seed
    items = PanelItems.from_values(frames[0])
    sums  = items.sums(stack(items.values(f) for f in frames))
    units = unit_draws(stream, len(items.units), replicates, bootstrap_seed)
    return point_statistic(sums, items.sizes), replicate_statistics(sums, items.sizes, units, seeds)
H = stratum(non-copied pairs: unit description_key, item pair_id, value reciprocal_rank, ITA_PANEL)
K = stratum(copied pairs, same columns, COPIED_STREAM)
C = stratum(census rows, OUTCOME_PANEL)
M = sorted(codes of the first seed's non-copied pairs, exploded  ∩  units of its census rows)
n_unmatched_codes = number of those ITA codes outside M
per seed, per c in M: C_c = mean census value of c; H_c = mean reciprocal_rank of the non-copied
    pairs that report c; rows (unit = item = c, value)
Cm = stratum(C_c rows, MATCHED_STREAM); Hm = stratum(H_c rows, MATCHED_STREAM)   # paired by code
gap(a, b) = Estimate(a.point - b.point, percentile_interval(a.reps - b.reps, GAP_LEVEL))
headline_gap = gap(C, H); copied_gap = gap(C, K); wording_gap = gap(K, H)
matched_headline_gap = gap(Cm, Hm)
threshold = r * C.point;  large = matched_headline_gap.interval[0] > threshold          # D2
```

Each `Estimate` of a single stratum is its point and `percentile_interval(reps, GAP_LEVEL)`.
`gap_record` reads the six records with `read_record` and the Stage 8 record, calls `gap_seeds`,
then `arm_gap` for each arm. `recommend_training_part` is true when every arm is `large` (D2).

`tools ita-gap` options: `--read` (repeated, required), `--stage8`, `--store` and `--output`
(required), `--settings` (default `conf/data/ita_panel.yaml`, for r). Replicates and the bootstrap
seed come from `load_config(DecisionConfig, 'data/decision.yaml')`. It refuses an existing
`--output` first. It reads no panel and logs nothing. It prints a table for each arm: each MRR,
each gap with its interval, `n_matched_codes`, `n_unmatched_codes`, the threshold and `large`.
Then it prints `Recommend a later ITA training part (D2): yes|no`. Refusals exit 1 with an
escaped message.

- [x] **Step 1: Write the failing tests** in `tests/unit/test_ita_gap.py`

```python
def test_points_are_the_means_and_their_differences():
    # stub, one seed, hand-made frames: H mean 0.25, K mean 0.75, C mean 0.5
    gap = arm_gap('lexical-stub', [seed], r=0.05, replicates=1000, bootstrap_seed=1)
    assert gap.headline.point == pytest.approx(0.25)
    assert gap.headline_gap.point == pytest.approx(0.25)
    assert gap.copied_gap.point == pytest.approx(-0.25)
    assert gap.wording_gap.point == pytest.approx(0.5)
    assert gap.threshold == pytest.approx(0.025)

def test_the_headline_estimate_is_arm_replicates():
    # three seeds; ItaResult(per_pair=…, others empty) per seed;
    # arm_replicates(items, 'euclidean-d16', results, replicates=2000, bootstrap_seed=7)
    # gives the point and the replicates whose percentile interval equals gap.headline's

def test_the_matched_gap_skips_codes_one_panel_lacks():
    # an ITA code with no census row, and a census code with no ITA pair:
    # n_matched_codes == len(shared), n_unmatched_codes == 1, no error

def test_a_pair_with_two_codes_counts_under_each_in_the_matched_gap(): ...

def test_large_follows_the_registered_rule():
    # values where the matched gap's lower bound clears r * C → large True; where it does not →
    # large False; gap_record's recommend_training_part is True only when both arms are large

def test_the_record_is_reproducible():
    # two gap_record calls on gap_fixture: equal except created_at

@pytest.mark.parametrize('case', ['two runs', 'two data fingerprints', 'stub without outcome',
                                  'd16 with outcome', 'checkpoint not the record\'s',
                                  'seed not the record\'s', 'repeated seed',
                                  'other outcome split', 'arm missing'])
def test_gap_seeds_refuse_reads_that_do_not_pair(case): ...
```

In `tests/unit/test_cli_commands.py`:

```python
def test_ita_gap_writes_the_record_and_logs_nothing(): ...
    # gap_fixture; exit 0; the output loads as ItaGapRecord; 'Recommend a later ITA training
    # part (D2):' in the output; no selection log file was created

def test_ita_gap_refuses_an_existing_output(): ...
```

- [x] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_ita_gap.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'naics_embedder.decision.ita_gap'`.

- [x] **Step 3: Implement `decision/ita_gap.py`, `gap_fixture` and `tools ita-gap`**

> Deviation: `arm_gap` and `gap_record` add refusals the plan does not list (no copied or headline pair, no matched code, a seed without a matched code's values, two reads with one file name), and `gap_fixture` gives the d16 runs real outcome rows; ledgered rulings.

- [x] **Step 4: Run them to see them pass**

```bash
uv run pytest tests/unit/test_ita_gap.py tests/unit/test_ita_reads.py tests/unit/test_cli_commands.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
./scripts/format_code.sh src/naics_embedder/decision/ita_gap.py src/naics_embedder/cli/commands/tools.py tests/fixtures/ita.py tests/unit/test_ita_gap.py tests/unit/test_cli_commands.py
git add src/naics_embedder/decision/ita_gap.py src/naics_embedder/cli/commands/tools.py tests/fixtures/ita.py tests/unit/test_ita_gap.py tests/unit/test_cli_commands.py
git commit -m "feat(decision): measure the gap between Census phrases and owner text"
```

### Task 4: Docs, the final gate and PR A

**Files:**
- Modify: `docs/usage.md`, `docs/api/ita_panel.md`, `docs/api/decision.md`, `CLAUDE.md`,
  `README.md`, `tests/README.md`

- [x] **Step 1: The docs**

- `docs/usage.md`: `### tools ita-read` and `### tools ita-gap` after `### tools outcome-panel`.
  Give one example each, with placeholder paths. Say that `ita-read` checks C1 before any table
  loads, never opens the sealed part, logs a reference read, and has a `--dry-run` that logs
  nothing. Say that `ita-gap` reads no panel.
- `docs/api/ita_panel.md`: a `::: naics_embedder.panels.ita_evidence` block after `ita_split`.
- `docs/api/decision.md`: "ITA reference reads" and "Text→code gap" sections with
  `::: naics_embedder.decision.ita_reads` and `::: naics_embedder.decision.ita_gap`, after the
  text→code margin section.
- `CLAUDE.md`: add the three modules to the directory tree. Under "Monitor, Epoch Summary and
  Selection", after the ITA panel paragraph, add one sentence: `tools ita-read` reads a reference
  arm on the ITA validation part after checking the pinned run
  (`conf/data/ita_evidence.yaml`), and `tools ita-gap` measures the gap from those reads without
  reading a panel. Add the three test files under "Testing and Validation".
- Recount with `find src -name '*.py' | wc -l`, `ls tests/unit/test_*.py | wc -l` and
  `uv run pytest --collect-only -q -q | tail -1`. Write the numbers where `CLAUDE.md`,
  `README.md` and `tests/README.md` state them.

- [x] **Step 2: The untouched seams and the lock**

```bash
git diff --exit-code BASE -- src/naics_embedder/panels/ita.py src/naics_embedder/panels/ita_split.py src/naics_embedder/panels/ita_scoring.py src/naics_embedder/panels/outcome.py src/naics_embedder/panels/decoding.py src/naics_embedder/panels/selection_log.py src/naics_embedder/decision/text_to_code.py src/naics_embedder/decision/resampling.py src/naics_embedder/decision/rule.py src/naics_embedder/decision/records.py src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py conf/data/ita_panel.yaml uv.lock; echo "exit=$?"
```

Expected: `exit=0`.

- [x] **Step 3: The gates**

```bash
uv run pytest -n auto -q
uv run ruff check src/ tests/
./scripts/format_code.sh --check --all; echo "exit=$?"
uv run mkdocs build --strict
```

Expected: no failure; `All checks passed!`; `exit=0`; no MkDocs warning. Check that
`site/api/decision/index.html` has the two new modules' headings.

- [x] **Step 4: No real pair in the tree**

```bash
grep -rn "embedding-data\|stage10_ita_establishment" src/ tests/ docs/ | grep -v "tests/fixtures/ita.py"
```

Expected: docstrings and docs prose only, with no absolute path. `conf/data/ita_evidence.yaml`
names the run, which is a pin, not data.

- [x] **Step 5: Commit**

```bash
git add docs/usage.md docs/api/ita_panel.md docs/api/decision.md CLAUDE.md README.md tests/README.md
git commit -m "docs: add the ITA read and gap commands"
```

- [x] **Step 6: Local review, then PR A**

> Deviation: the executing-plans final review ran here over PR A's range (Codex at e36c38d and an opus code-reviewer); its fixes and the re-run mutant runs (Task 3's under Python 3.10) landed before the push; ledgered rulings.

```bash
mkdir -p /tmp/codex-review && codex exec review -m gpt-6-astra -c sandbox_mode=read-only --base origin/main -o /tmp/codex-review/plan20-pr-a.md; echo "exit=$?"
```

Triage each finding (receiving-code-review). Fix what is real in a new commit, and record the
rest with a reason in the ledger. Re-run Step 3 after any fix. Then push and open PR A against
`main`. Its body says that no real pair is read before it merges, and that `uv.lock` is
unchanged. Bind it in the app (`ccd_pr` `get_status`, then `bind_pr` if needed), and turn on
Auto-fix (`set_monitor`).

### Task 5: Merge PR A and ready MAIN

Nothing in this task reads real data.

**Files:** none in the repository. Writes MAIN's `.venv` only.

**Interfaces:**
- Consumes: PR A (Task 4).
- Produces: READ_SHA, MAIN's HEAD at PR A's merge or later; the environment receipt; a free OUT.

- [x] **Step 1: Merge PR A**

> Deviation: the checks were read once with `gh pr checks` after the owner's "Merge when green", not watched; all four had passed on 261b5c6.

Wait for its checks with `gh pr checks <n> --watch`, run in the background. Once every listed
check has passed on the head SHA, merge with
`gh pr merge <n> --merge --match-head-commit <sha> --delete-branch`. Confirm
`gh pr view <n> --json state,mergeCommit` shows `MERGED`, and that the remote branch is gone.
Report `CI passed, PR merged, remote deleted`.

- [x] **Step 2: Fast-forward MAIN with the guard**

In MAIN: on `main`, `git status --porcelain` empty, `git log --oneline origin/main..main` empty,
then `git -c core.fsmonitor=false merge --ff-only origin/main`.

```bash
cd /Users/lowell/Projects/naics-embedder && git rev-parse HEAD origin/main && git merge-base --is-ancestor <PR A's merge SHA> HEAD; echo "exit=$?"
```

Expected: two equal SHAs, then `exit=0`. Record HEAD as READ_SHA.

- [x] **Step 3: Sync MAIN's environment**

```bash
cd /Users/lowell/Projects/naics-embedder && shasum -a 256 uv.lock conf/data/ita_evidence.yaml conf/data/ita_panel.yaml
cd /Users/lowell/Projects/naics-embedder && uv sync --locked; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && uv sync --locked --check; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools ita-read --help | head -5; echo "exit=$?"
```

Expected: the pinned lock; `exit=0` three times.

- [x] **Step 4: The environment receipt**

Write `SCRATCH/env.py`. It prints the platform, Python, torch, numpy and polars versions, whether
MPS is available, and each BLAS thread variable (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`), or `unset`. Run it from MAIN with
`uv run --locked --no-sync python /tmp/naics_plan20/env.py`, and copy the output into the ledger.

- [x] **Step 5: The outputs**

```bash
ls /Users/lowell/naics-artifacts/records/text_to_code 2>&1
test -w /Users/lowell/naics-artifacts; echo "writable=$?"
df -h /Users/lowell/naics-artifacts
```

Expected: `No such file or directory`, `writable=0`, and free space in gigabytes. If OUT or any
`ita-margin` record exists, stop and ask.

### Task 6: Gate A — the log, C1 and the dry runs

**Files:** none in the repository. Reads the selection log and loads the real tables. Logs
nothing.

**Interfaces:**
- Consumes: READ_SHA (Task 5).
- Produces: LINES_BEFORE and SHA_BEFORE, the log before the reads; six passing dry runs; the
  validation part's pair count and data fingerprint, DATA_FP.

- [x] **Step 1: Ask for Gate A**

Tell the user: PR A merged at READ_SHA, EDATA_SHA, and the environment receipt. Ask for approval
to start the real-data steps: the log check and six dry runs, which load the real tables and the
arms but score nothing and log nothing. Wait for a clear yes.

- [x] **Step 2: The log's state**

Write `SCRATCH/log_state.py`. It takes the log, then optionally `PREFIX_LINES PREFIX_SHA`, then
optionally `RUN`. It prints:

- the line count, record count, final newline and SHA-256;
- the records that lack a field;
- a tally by event, panel and split;
- every `ita_text` record;
- every run id that starts with `ita-margin/`.

It exits 1 on any of these:

- a missing newline or a malformed record;
- an `ita_text` record other than a `validation` `read`;
- any `ita-margin/` run.

Given a prefix, it exits 1 unless the log's first `PREFIX_LINES` lines hash to `PREFIX_SHA`, and
it prints every later record whose `detail.run` is not RUN: another session's reads. Given RUN,
it lists RUN's records. It exits 1 unless they are exactly six `ita_text` validation reads and
one `outcome` validation read. All seven must have role `reference` and purpose PURPOSE. The ITA
reads must be the stub once and d16 seeds 1–5 once each. The outcome read must be the stub's.

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan20/log_state.py /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl > /tmp/naics_plan20/log_before.txt; echo "exit=$?"
cat /tmp/naics_plan20/log_before.txt
```

Expected: `exit=0`, and no `ita_text` record. Plan 19's 118 growth reads and any other sessions'
validation reads are no stop: list them in the ledger. Record `lines` as LINES_BEFORE and
`sha256` as SHA_BEFORE.

- [x] **Step 3: The six dry runs**

> Deviation: EE is the two-digit epoch, so the files are `epoch=039.ckpt` and, for seed 4, `epoch=038.ckpt`, as the Pinned values name them; ledgered ruling.

PURPOSE is `text-to-code reference gap read: redesign roadmap Stage 4, plan 20 (Req 28)`. The
dry runs pass a placeholder run, `ita-gap/dry-run`. The stub's dry run, from MAIN's root:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools ita-read --arm lexical-stub --outcome \
  --pairs /Users/lowell/Projects/embedding-data/data/derived/osha/establishment-targets/stage10_ita_establishment_20261116_r1/ita_text_pairs.parquet \
  --pair-codes /Users/lowell/Projects/embedding-data/data/derived/osha/establishment-targets/stage10_ita_establishment_20261116_r1/ita_text_pair_codes.parquet \
  --descriptions /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet \
  --run ita-gap/dry-run --purpose 'text-to-code reference gap read: redesign roadmap Stage 4, plan 20 (Req 28)' \
  --store /Users/lowell/naics-artifacts \
  --log /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl \
  --output /Users/lowell/naics-artifacts/records/text_to_code/stage4/lexical-stub.json \
  --dry-run > /tmp/naics_plan20/dry-lexical-stub.txt 2>&1; echo "exit=$?"
```

For d16 seed N, drop `--outcome` and `--descriptions`, and pass `--arm euclidean-d16 --seed N
--checkpoint MAIN/checkpoints/stage8-euclidean-d16-sN/epoch=0EE.ckpt --table
MAIN/checkpoints/stage8-euclidean-d16-sN/arm_table_epoch=0EE.parquet --output
OUT/euclidean-d16-sN.json`, with the epoch EE from the Pinned values (038 for seed 4, 039
otherwise), written to `dry-euclidean-d16-sN.txt`. Expected: six `exit=0`, six `Dry run: every
check passed`, one validation pair count and one data fingerprint (DATA_FP) in all six, the
split fingerprint `8602ceb2…`, `LexicalTrigramEncoder`/`cosine` for the stub and
`ArmEncoder`/`euclidean` for d16. Then:

```bash
shasum -a 256 /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl
ls /Users/lowell/naics-artifacts/records/text_to_code 2>&1
```

Expected: SHA_BEFORE, and `No such file or directory`. If the SHA-256 changed, run `log_state.py`
with `LINES_BEFORE SHA_BEFORE`: it must exit 0, and the new records are another session's. List
them in the ledger.

- [x] **Step 4: Ledger**

Copy Steps 2–3's outputs into the ledger. Estimate each read's memory from the pair count: the
stub's query matrix is pairs × 4,096 float32 values before its float64 copy.

### Task 7: Gate B — the six reads

**Files:** none in the repository. Appends seven records to MAIN's log, and writes six records
and their objects in STORE.

**Interfaces:**
- Consumes: Task 6's results.
- Produces: OUT's six read records, each passing `check_read.py`; RUN.

- [x] **Step 1: Ask for Gate B**

First run `mcp__ccd_session_mgmt__list_sessions`, and note each running session whose `cwd` is
under MAIN. Generate RUN once with `python3 -c 'import uuid; print("ita-gap/" + uuid.uuid4().hex)'`
and record it. Show the user Task 6's results, the read command, the order (the stub, then d16
seeds 1–5) and those sessions. Say that the reads append six `ita_text` validation reads and one
outcome validation read to MAIN's log, open nothing and are never re-run. Ask that, until Task 8
passes, no other session merges a PR, syncs MAIN or reads a panel into MAIN's log. Ask for a go,
and wait for a clear yes.

- [x] **Step 2: Read each arm**

Right before the first read, run `log_state.py` with `LINES_BEFORE SHA_BEFORE`. Expected:
`exit=0`. Then, for each read in order, run the drift guard first:

```bash
cd /Users/lowell/Projects/naics-embedder && test "$(git rev-parse HEAD)" = READ_SHA && test -z "$(git status --porcelain)"; echo "exit=$?"
```

Expected: `exit=0`. On `exit=1`, stop before the read. Then run Task 6 Step 3's command for that
arm without `--dry-run`, with `--run RUN`, writing `/tmp/naics_plan20/read-<name>.txt`. Run it
in the background and wait for it to exit. Never start a read before the previous one's Step 3
passes. Expected: `exit=0`. Record the first d16 read's wall time, and tell the user the estimate
for the other four.

- [x] **Step 3: Check each record before the next read**

Write `SCRATCH/check_read.py`. It takes a record path, RUN, the arm and, for d16, the seed. It
loads the record with `read_record(..., ItaReferenceRead)` and `ArtifactStore(STORE)`. Then it
runs these checks in order:

1. `store.resolve` passes for `per_pair` and, for the stub, `outcome.scores`.
2. `run`, `arm` and `seed` are as given, and `evidence` equals MAIN's
   `load_ita_evidence('conf/data/ita_evidence.yaml')`.
3. `split_fingerprint` is `8602ceb2…`, and `data_fingerprint` is DATA_FP.
4. For d16, `checkpoint_sha256` is the seed's pinned checkpoint SHA-256. `table_fingerprint`
   equals the Stage 8 record's `matrix_fingerprint` for that seed.
5. For the stub, `outcome` exists, `summary['n_queries']` is 4042, and its two fingerprints are
   the pinned outcome fingerprints. It prints `summary['mrr']` beside §5's 0.6165, which is
   reported and not checked (P4).
6. `log_records` are exactly the log's records whose `detail.run` is RUN and whose arm and seed
   are this read's. Each has role `reference` and PURPOSE. There is one `ita_text` `validation`
   `read`, and for the stub one `outcome` `validation` `read` after it.

It prints the headline and copied `n_pairs` and MRR, the phrasing MRR and the counts. It exits 1
on the first failure, with its message.

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan20/check_read.py /Users/lowell/naics-artifacts/records/text_to_code/stage4/<name>.json RUN <arm> [<seed>]; echo "exit=$?"
```

Expected: `exit=0`.

- [x] **Step 4: The log after the reads**

> Deviation: the scratch `log_state.py` crashed in its RUN check after listing the records (a loop over (line, record) pairs); fixed and re-run read-only, exit=0.

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan20/log_state.py /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl LINES_BEFORE SHA_BEFORE RUN > /tmp/naics_plan20/log_after.txt; echo "exit=$?"
```

Expected: `exit=0`; RUN's seven records; no `ita_text` record but RUN's six reads; no
`ita-margin/` run. With no other session's records, the log has LINES_BEFORE + 7 lines.

- [x] **Step 5: Ledger**

Copy each read's output, each `check_read.py` output, Step 4's output, and `shasum -a 256` of the
six records into the ledger.

### Task 8: The gap and the verify

**Files:** none in the repository. Writes `OUT/gap.json`.

**Interfaces:**
- Consumes: the six records (Task 7), RUN, READ_SHA, LINES_BEFORE, SHA_BEFORE.
- Produces: GAP_SHA, the SHA-256 of `OUT/gap.json`; the tables that the finding copies.

- [x] **Step 1: The gap**

```bash
test ! -e /Users/lowell/naics-artifacts/records/text_to_code/stage4/gap.json; echo "absent=$?"
```

Expected: `absent=0`. Run Task 7 Step 2's drift guard: `exit=0`. Then, from MAIN's root:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools ita-gap \
  --read /Users/lowell/naics-artifacts/records/text_to_code/stage4/lexical-stub.json \
  --read /Users/lowell/naics-artifacts/records/text_to_code/stage4/euclidean-d16-s1.json \
  --read /Users/lowell/naics-artifacts/records/text_to_code/stage4/euclidean-d16-s2.json \
  --read /Users/lowell/naics-artifacts/records/text_to_code/stage4/euclidean-d16-s3.json \
  --read /Users/lowell/naics-artifacts/records/text_to_code/stage4/euclidean-d16-s4.json \
  --read /Users/lowell/naics-artifacts/records/text_to_code/stage4/euclidean-d16-s5.json \
  --stage8 /Users/lowell/naics-artifacts/records/stage8/euclidean-d16.json \
  --store /Users/lowell/naics-artifacts \
  --output /Users/lowell/naics-artifacts/records/text_to_code/stage4/gap.json \
  > /tmp/naics_plan20/gap.txt 2>&1; echo "exit=$?"
```

Expected: `exit=0`, two arm tables and the D2 line. Either answer is a result, not a failure.

- [x] **Step 2: Verify — read only**

> Deviation: `verify.py` takes RUN as a second argument, which its check 1 needs; ledgered ruling.

Write `SCRATCH/verify.py`. It reads `gap.json` with `read_record(..., ItaGapRecord)` and exits 1
on the first check that fails:

1. `run` is RUN, and `read_sha256` names the six files with their current SHA-256.
2. `stage8_sha256` is `3d5a80d5…`, `replicates` 10,000, `bootstrap_seed` 20260924, `level` 0.95,
   `r` 0.05.
3. `arms` are `lexical-stub` then `euclidean-d16`, with `n_seeds` 1 and 5.
4. `recommend_training_part` equals every arm's `large`, and each `large` follows D2 from its
   estimates.

It prints each arm's estimates as Markdown table rows for the finding. Then:

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan20/verify.py /Users/lowell/naics-artifacts/records/text_to_code/stage4/gap.json; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && git rev-parse HEAD && shasum -a 256 uv.lock /Users/lowell/naics-artifacts/records/text_to_code/stage4/*.json
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan20/log_state.py /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl LINES_BEFORE SHA_BEFORE RUN; echo "exit=$?"
ls /Users/lowell/naics-artifacts/records/text_to_code/stage4
```

Expected: `exit=0`; READ_SHA and the pinned lock; `exit=0` with Task 7 Step 4's counts; only the
six read records and `gap.json`. This shows the Exit: the reference's validation reads are in the
log, there is no `ita_text` opening, no margin record and no choice. Record GAP_SHA. Copy all of
it into the ledger. From here on MAIN may follow `origin/main` again. Tell the user that the read
window is closed.

### Task 9: The finding

**Files:**
- Create: `specs/findings/text-to-code-gap.md`
- Modify: `CLAUDE.md` (the ITA panel paragraph under "Monitor, Epoch Summary and Selection")

- [x] **Step 1: Cut PR B's branch**

```bash
git -c core.fsmonitor=false fetch origin
git -c core.fsmonitor=false checkout --no-track -B claude/stage-4-text-to-code-gap origin/main
```

- [x] **Step 2: Write the finding**

`specs/findings/text-to-code-gap.md`, in `specs/findings/growth-margins.md`'s form. It opens with
`**Status: FINAL (<date>).**`, and names roadmap Stage 4 and plan 20. Its first paragraph gives
each arm's C − H with its interval and the D2 answer. Its second paragraph labels every score as
a development measurement on the ITA validation part, not evaluation (C3). Its sections:

1. **Universe (C4).** The run and its pin. ITA is a program-selected filing population and
   represents no wider set of establishments. Codes are self-reported and unvalidated. A pair's
   headline codes are native NAICS 2022 codes from filing years 2023 to 2025. Its text is the
   most frequent string over all of its records, which can include filing years 2016 to 2022. The
   copied flags catch exact normalized identity only.
2. **Sources.** A table of each input with its SHA-256: EDATA_SHA and the protocol commit, the
   run, `outputs.json`, both tables with their row counts, `conf/data/ita_evidence.yaml`, the
   split fingerprint, DATA_FP, the bundle, the stub's descriptions, d16's five checkpoints and
   tables, the Stage 8 record, the six read records and `gap.json` (GAP_SHA), `uv.lock`, and
   READ_SHA with the environment receipt.
3. **The reads.** The command template, RUN, PURPOSE, the order and wall times, and the
   validation part's counts: pairs, description groups, copied, conflict, mojibake and company
   fallback.
4. **Scores.** For each arm, the headline, copied and phrasing reports: MRR, top-1, hit@5,
   hit@10, mean LCA level and the coarser-level MRRs at levels 2–5. d16's are seed means. The
   stub's outcome summary beside §5's 0.6165, which was read before the descriptions rebuild.
5. **The gap (D1).** For each arm: C, K, H, the matched C and H, the four gaps with their 95 %
   intervals, `n_matched_codes` and `n_unmatched_codes`. Say what each gap isolates. C − H is
   held-out Census phrases against owners' own words. C − K is the same against owners who
   copied a Census phrase. K − H is the wording effect within ITA, which training overlap can
   inflate. The matched gap removes the code mix.
6. **A training part (Req 28, Later).** D2's rule, quoted from this plan; each arm's threshold
   and verdict; the recommendation. Say what follows: a training part needs a later spec and the
   R8.4 amendment (G1). This finding changes nothing in Stage 5.
7. **The selection log.** The records and SHA-256 before and after, RUN's seven records, no
   `ita_text` opening, and no `ita-margin/` run. Nothing was selected, and no margin record
   exists.
8. **What Stage 5 reads.** The six read records, `gap.json` and this finding. Stage 4's reads are
   gap evidence, and the margin record makes its own fresh reads (spec §4.5).

No section quotes a description, a representative string, a pair id or a per-pair row.

- [x] **Step 3: The `CLAUDE.md` sentence**

After the ITA panel paragraph's sentence on the sealed part, add one sentence: "Stage 4 measured
the gap between Census phrases and owner text on the validation part on <date>
(`specs/findings/text-to-code-gap.md`), as a development measurement."

- [x] **Step 4: Check and commit**

Measure the finding's line lengths with Python: at most 100 outside code blocks and tables.
Then:

```bash
grep -c 'representative_string\|description_key' specs/findings/text-to-code-gap.md
git add specs/findings/text-to-code-gap.md CLAUDE.md
git commit -m "docs(findings): record the text→code gap between Census phrases and owner text"
```

Expected for the grep: at most the counts of column names in prose, never a value.

## Plan completion (controller, inline)

Run the writing-plans Plan Completion Protocol on PR B's branch, before it is pushed:

- **Deferred items.** Before editing `specs/deferred_items.md` or the roadmap, check for other
  running sessions that edit them, and ask if one does. Append this plan's deferred items, if
  any.
- **Handoff.** One bullet for Stage 5. It gives the six read records and `gap.json` (paths and
  SHA-256), the finding, RUN, and DATA_FP as the table build that Stage 5's reads must match.
  It names `decision/ita_reads.py`'s `read_reference` and the d16 run directories as a reusable
  read path. It says that Stage 4's reads are reference reads under `REFERENCE_ARMS`, so they
  leave `check_ita_margin_first` open. It records D2's answer. One bullet for embedding roadmap
  Stage 12: the sealed part is unopened, under split fingerprint `8602ceb2…`.
- **Tick and stamp** Stage 4 in `specs/naics-artifact-redesign-roadmap.md`. Run the backlog
  statistics and the deferral scan. Retire this plan to `specs/plans/completed/`, and append its
  `specs/history.md` entry, in the protocol's commits.

## Integration (controller, inline)

1. Push PR B and open it against `main`. Merge it as Task 5 Step 1 merged PR A, and report
   `CI passed, PR merged, remote deleted`.
2. Fast-forward MAIN with the guard, delete the local branches, and `git fetch --prune`.
3. Copy the ledger to `MAIN/logs/plan20_execution.md`. Delete SCRATCH and
   `/tmp/codex-review/plan20-pr-a.md`.
4. Write the resume handoff (prepare-handoff). The next step is a resume of the roadmap, which
   then routes Stage 5 (brainstorming).

## Execution

Planning ran on Opus. Execute in a fresh session on Sonnet (`claude --model sonnet`), inline with
executing-plans (P10, D5), from an app worktree, with this file as the handoff. Start no task before
Pre-flight Step 1 finds Stage 3's COMPLETE stamp.

## Handoff to later stages

- **Stage 5 (text→code candidates and decision).** It consumes the six read records and the gap
  record in `~/naics-artifacts/records/text_to_code/stage4/`: `lexical-stub.json`
  (e200e50fd900a0075a4555f269d778de09d3dfe7e0a7cfe522216b1c0a2f2ee1),
  `euclidean-d16-s1.json` to `-s5.json` (3c6c7d18…45f2, b8ceaabe…8621, 2a546f34…9706,
  017ddb88…5eaa, 00e48327…0523; full values in the finding's Section 2) and `gap.json`
  (f126926250d84178636a6bc087b2f520fbea11de971feec62c22bbd71a256997), and the finding
  `specs/findings/text-to-code-gap.md`. RUN is `ita-gap/d1dd4024708744a9aa33ecf8c346ad36`. DATA_FP,
  8980e3586ec0a18d4623c3a4b3e81527f8c59e34a1f0b4b73d07b046ed86b1a3, under split fingerprint
  8602ceb243722dc64e0a29a9ba444fe824baf209a848964d7ddf130a9db5cc7c, is the table build that
  Stage 5's reads must match. `decision/ita_reads.py`'s `read_reference` (and `tools ita-read`)
  with the d16 run directories `checkpoints/stage8-euclidean-d16-sN/` (`epoch=039`, `038` for
  seed 4) is a reusable read path. Stage 4's reads are reference reads under `REFERENCE_ARMS`, so
  they leave `check_ita_margin_first` open, and the margin record makes its own fresh reads
  (text-to-code spec §4.5). D2's answer is yes: the finding recommends a later ITA training part,
  which needs a later spec and G1, and changes nothing in Stage 5.
- **Differs from the roadmap's Produces.** Nothing: the reference's validation scores and the
  finding exist as the entry states. Beyond the plan, `tools ita-read` refuses stub descriptions
  whose SHA-256 is not the bundle's description fingerprint, and a dry run creates nothing.
- **Traps and deferred items.** `tools ita-read` takes `--outcome` on a d16 read and leaves it off
  by default for the stub, and it does not check a d16 seed's checkpoint and table against the
  Stage 8 record before it logs; both are deferred into Stage 5 (`specs/deferred_items.md`
  § 20-text-to-code-reference-read). Plan 17's items for Stage 5 stay open
  (§ 17-text-to-code-ita-panel). The macOS level pin still fails locally
  (§ 19-economic-decision-read).
- **Embedding roadmap Stage 12.** The ITA sealed part is unopened, under split fingerprint
  8602ceb243722dc64e0a29a9ba444fe824baf209a848964d7ddf130a9db5cc7c. The selection log holds no
  `ita_text` opening (284 lines, SHA-256
  5b98c867ed96e7c05d3d3bca46146b45ada2245d351cfeae58ed6efc4980f6f0, finding Section 7).
