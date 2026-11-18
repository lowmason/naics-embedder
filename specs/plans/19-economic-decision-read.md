# Economic Decision Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 3 — on plan completion, tick the stage
> and re-validate later stages against what shipped.

**Goal:** Build Stage 3 of the redesign roadmap. Pin the shared text-only table's provenance file
in the frozen candidate list. Then, with the user's approval at the time, read the list's 18
economic candidates on the real growth validation splits on the Mac, record the economic decision
against the margins of 2026-10-17, and write the finding.

**Architecture:** One code change, then real reads from the main checkout, then the decision.

- **The pin (Task 1, PR A).** `conf/data/economic_candidates.yaml` gains
  `text_only_provenance_sha256`. The decision's gate, the deterministic read and the trained
  import each refuse a text-only provenance other than the pinned file. This closes the deferred
  item that Stage 3's Exit names. It changes the list's sha256.
- **Merge and ready (Task 2).** PR A merges before any real read. MAIN fast-forwards to that
  merge, its `.venv` syncs to the lock, and the list's new sha256 is recorded.
- **Gate A (Task 3).** With the user's approval to start the reads, check the selection log and
  run the 18 dry runs. Nothing is logged.
- **Gate B (Task 4).** With the user's go, read the 18 candidates one at a time from MAIN's root.
  After each read, a store-only check confirms that its record is intact and pairs with the
  margin read. A pairing failure stops the plan after one candidate, not after 18.
- **The decision (Task 5).** `tools decide` over the 18 records, from MAIN's root. It reads no
  panel and logs nothing.
- **Verify (Task 6).** Read only: the decision record, its list and provenance, and the log's
  growth by exactly the reads' 118 records.
- **The finding (Task 7, PR B).** `specs/findings/economic-decision.md`, with the answer to split
  Req 19's propagation question, and one sentence in `CLAUDE.md`.

**Tech Stack:** Python 3.12 under uv (`uv run --locked --no-sync`), Pydantic, Polars, Typer and
pytest; the `naics-embedder` CLI; Python scripts over the package's own APIs; `shasum`, git and
gh. No new dependency.

**Spec:** none. Stage 3 routes to writing-plans and has no stage spec. Its requirements are the
roadmap's Stage 3 entry (`specs/naics-artifact-redesign-roadmap.md`). It cites Req 29 of
`specs/naics-artifact-redesign.md` (The test, and the margins of 2026-10-17), split Req 24
(`specs/naics-representation-split.md`) and Verification "Economic candidates". Read also:

- the candidates' design, `specs/completed/economic-candidates.md`: §4.4 (the decision), §4.7
  (the propagation rule), §4.8 (the pinned path) and §5 (one role per table or part);
- plan 16's Handoff (`specs/plans/completed/16-economic-candidates.md`, "Handoff to later
  stages");
- the finding `specs/findings/growth-margins.md`, and plan 15
  (`specs/plans/completed/15-growth-margin-read.md`), whose read procedure this plan reuses;
- `specs/deferred_items.md`, § 16-economic-candidates, second item ("Pin the shared text-only
  table's provenance file in the frozen list").

## Global Constraints

Every task's requirements include this section.

**Locators.** MAIN is the main checkout, `/Users/lowell/Projects/naics-embedder`. STORE is the
artifact store, `/Users/lowell/naics-artifacts`. SCRATCH is `/tmp/naics_plan19`. WORKTREE is the
execution session's worktree, as `git rev-parse --show-toplevel` prints it. P1–P9 are this plan's
choices (`## Decisions`).

### The Exit (`specs/naics-artifact-redesign-roadmap.md`, Stage 3, at 8329e55), verbatim

> Exit: Before the first real candidate read, the frozen list pins the text-only table's
> provenance file and the list's sha256 is recorded again (`specs/deferred_items.md`,
> § 16-economic-candidates, second item; plan 16, Handoff, Resolved at resume, 2026-11-17).
> The decision record exists and names each candidate's text-and-tree provenance.
> `tools growth-margins` did not run, and the decision reads the 2026-10-17 record. The
> selection log gained only validation reads, and no outer set opened. The finding cites the
> record's sha256 and the platform, Python and lock that it was read under.

### Pinned values

| What | Value |
|---|---|
| Margin record | `STORE/records/economic/growth_margins.json`, sha256 `4f86a0b2f3bde358bd08a1f3f17f1b4c59ee261ee4e5409ac83615deaa1f37d0` |
| δ | `growth_seen` 0.00015933956325255588, `growth_heldout` 0.0001619577593991326 |
| `uv.lock` | sha256 `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948` |
| Read environment | macOS arm64, Python 3.12.12, numpy 2.3.4 on Accelerate, no BLAS thread variable set (finding: growth-margins §1) |
| Codebook | `MAIN/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet`, sha256 `0f9e61343e673540e1e18ecc5581618be099c5d79531b68153496bf982d6e96d` |
| Text-only table | `MAIN/checkpoints/plan9_exit/text_only.parquet`, sha256 `f4fb4f574f880c940ee8cf44c7f86615638e59f113b527d0b86973cd45ba32ff` |
| Its provenance | `MAIN/checkpoints/plan9_exit/text_only_provenance.json`, sha256 `bd21c628d06b96c8a0ffe1318c18f81199cb00ca8862560dbdb8dc57887ea05e`. All ten Stage 7/8 source records store this file (verified at plan time) |
| Runs root | `MAIN/checkpoints`: `stage7-reference-s1` to `s10`, and `stage8-<arm>-s1` to `s5` |
| Selection log | `MAIN/logs/selection_log.jsonl`: 159 records, sha256 `51aef24099f0303c218271fc9ab110c3f1a8e7a56198d97eab3c84a0731a2a41` at plan time |
| Held-out draw | `deddfd4c395ca2ea8164e4425a4fdfff4e564360d20467d5a76163504cbb8ac4` |
| Margin read's run | `margins/17a7e118f926405ea895bd0f908a549c`, purpose `economic margin read: split roadmap Stage 3, plan 15 (Req 24)`: the log's only two growth reads at plan time |
| The reads | 18 candidates, 59 runs, 118 logged validation reads: nine deterministic candidates of one run, eight trained of five seeds and `hyperbolic-d16` of ten |
| The decision | 10,000 replicates, bootstrap seed 20260924 (`conf/data/decision.yaml`); superiority at 97.5 % over the two growth panels |

The candidates, in the list's order, which is the read order (P5): `text-only-d8`,
`text-only-d16`, `text-only-d32`, `tree-d8`, `tree-d16`, `tree-d32`, `smoothed-d8`,
`smoothed-d16`, `smoothed-d32`, `hyperbolic-d16`, `euclidean-d8`, `euclidean-d16`,
`euclidean-d32`, `hyperbolic-d8`, `hyperbolic-d32`, `spherical-d8`, `spherical-d16` and
`spherical-d32`.

### Project rules

Plan 15's Project rules apply (`specs/plans/completed/15-growth-margin-read.md`, Global
Constraints), with this plan's values:

- **Real data only behind the gates.** The dry runs (Task 3) load no panel, but they read the
  real records and tables, so they wait for Gate A's approval. Nothing loads the growth panel
  before Gate B's go (Task 4). Each approval comes from the user in chat, at that step.
- **One read per candidate.** `tools economic-read` runs once for each candidate. A failed or
  partial read may already have appended log records, and a re-run would append more. Never
  re-run a candidate without the user's ruling.
- **The margins are fixed.** Never run `tools growth-margins`. Never move, edit or delete the
  margin record.
- **The selection log is append-only.** Never edit, truncate, move, copy into the repository or
  delete `MAIN/logs/selection_log.jsonl`. Only the reads append to it.
- **MAIN.** Never check out a branch there, and never edit its tracked files. Its only writes are
  Task 2's fast-forward and `uv sync`, the reads' log records, its gitignored `logs/tools_*.log`
  files and Integration's ledger copy. From Task 2 Step 3 until Task 6 passes, nothing merges
  into MAIN: the reads and the decision run at one commit, READ_SHA.
- **Absolute paths.** Pass every path as an absolute path, never with `~`. `write_record` does
  not expand `~`.
- **The lock.** Never change `uv.lock`, and never run `uv lock`.
- **The reference's code path.** No task edits `panels/regressor.py`, `panels/qcew_rows.py`,
  `panels/regressor_splits.py` or `panels/ridge.py` (spec §4.8).
- **Git.** Cut each branch from `origin/main`. Never push `main` or a `refs/held/*` ref, and
  never use `--no-verify`. If a git command stalls, rerun it as `git -c core.fsmonitor=false …`.
  Commits end with `Co-Authored-By: <the executing model> <noreply@anthropic.com>`, and PR
  bodies with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Merging.** The repository has no auto-merge. Merge a PR only once every check that
  `gh pr checks` lists has passed on its head, with
  `gh pr merge <n> --merge --match-head-commit <sha> --delete-branch`.
- **Style.** Python: single quotes, `./scripts/format_code.sh` on touched files, never
  `ruff format`. Markdown: lines within 100 characters outside code blocks and tables.
- **Bash tool.** It runs zsh. Prefix every command with `cd <dir> &&`, and gate a check on
  `; echo "exit=$?"`, never on a pipe's tail. Run a command that may take over 2 minutes in the
  background, and wait for its completion notice.

## Review Focus

1. **A provenance beside `--text-only` that is not the pinned file, while the table's bytes
   match** (a regenerated provenance). The read and the dry run refuse before the panel loads,
   name the file and the pin, and log nothing. Test: Task 1,
   `test_economic_read_refuses_a_text_only_provenance_other_than_the_lists`.
2. **A trained source record that stores another text-only provenance.** The import refuses
   before the panel loads. Test: Task 1,
   `test_economic_read_refuses_a_trained_source_with_another_text_only_provenance`.
3. **A hand-built economic record that stores another provenance for the same table bytes** (the
   deferred item's case). `tools decide` exits 1, names the candidate and writes no decision
   record. Test: Task 1, `test_decide_refuses_a_record_that_stores_another_text_only_provenance`.
4. **A list file from before the pin**, with no `text_only_provenance_sha256`. Loading it
   refuses and names the field, so `tools decide --candidates OLD` refuses before any comparison.
   Test: Task 1, `test_a_list_without_the_text_only_provenance_pin_is_refused`.
5. **A real read whose reference does not pair with the margin read** (BLAS threading, a changed
   environment). It shows after the first candidate, not after 118 logged reads, and the plan
   stops. Check: Task 4 Step 3, `check_arm.py`.

## Decisions

No owner decision was taken after the roadmap's resume of 2026-11-18.

This plan's choices, for the owner's review at the plan PR:

- **P1:** the pin is a required list field, `text_only_provenance_sha256`, with the sha256
  pattern. Three places refuse another provenance: the decision's gate
  (`_require_shared_text_only`, which runs for both kinds), the deterministic read before the
  panel loads (`_deterministic_candidate`), and the trained import (`_check_source`). The two
  early refusals keep a bad input from logging reads that the decision would then refuse. The
  build (`tools economic-build`) is not changed: its nine tables exist and are pinned. Applied
  in: Task 1.
- **P2:** PR A lands first, alone. The reads and the decision run from MAIN's root at its merge,
  READ_SHA, so the list that the decision stores is the merged one, and the relative defaults
  (`conf/data/regressor_panel.yaml`, `conf/data/decision.yaml`) are MAIN's. Applied in: Tasks
  2–5.
- **P3:** the dry-run text item (§ 16-economic-candidates, first item) stays deferred. Stage 3's
  Exit does not name it, and plan 18 is live in `cli/commands/tools.py`. The ledger says what the
  dry run checks. Applied in: Task 3 Step 2.
- **P4:** two gates, as plan 15 had. Gate A covers the read-only checks and the dry runs. Gate B
  covers the 18 reads and shows Gate A's results first. `tools decide` reads no panel and logs
  nothing, so it needs no gate of its own. Applied in: Tasks 3–5.
- **P5:** the reads run in the list's order, so the first is `text-only-d8`: deterministic, one
  run, two logged reads. Each read is followed by `check_arm.py` before the next starts.
  Applied in: Task 4.
- **P6:** the arguments. `--purpose 'economic candidate read: redesign roadmap Stage 3, plan 19
  (Req 29)'` (PURPOSE); `--output STORE/records/economic/<name>.json`; `--log` and `--lock`
  MAIN's; `--text-only` and `--runs-root` on every read, as plan 16's dry runs passed them. The
  decision: `--name economic`, `--question 'Which candidate, if any, is the economic artifact?'`,
  `--candidates` MAIN's list and `--output STORE/records/economic/decision.json`, as
  `docs/usage.md` shows. Applied in: Tasks 3–5.
- **P7:** every real read and the decision run as
  `HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync`, with the BLAS thread variables
  unset, as the margin read ran (plan 15, P3 and P4). Applied in: Tasks 2–5.
- **P8:** docs. Task 1 changes the economic-read input checks in `docs/usage.md` and the test
  counts. Task 7 adds the finding and one sentence in `CLAUDE.md`. Applied in: Tasks 1 and 7.
- **P9:** inline execution (executing-plans). The gates need the user, and a subagent cannot ask
  them. Applied in: Execution.

---

## Workspace

- **Ledger:** `WORKTREE/logs/plan19_execution.md`, ignored by git. Write every command, its
  output and any deviation there. Integration copies it to `MAIN/logs/`.
- **Branches:** PR A is `claude/economic-text-only-provenance-pin` and PR B is
  `claude/stage-3-economic-decision`. Cut each with
  `git -c core.fsmonitor=false checkout --no-track -B <branch> origin/main`.
- **Outputs:** `STORE/records/economic/<name>.json` for each candidate and `decision.json`. At plan
  time `STORE/records/economic/` held only `growth_margins.json`. The reads' predictions and
  scores go to the store's `objects/`.
- **Scratch:** SCRATCH, for this plan's scripts and command outputs. Integration deletes it.

## Stop-and-ask conditions

Stop, report and wait for your human partner when any of these happens:

- The user declines Gate A or Gate B, or answers anything but a clear yes.
- Pre-flight finds an open PR, or a commit on `origin/main` after 8329e55 other than this plan's
  merge, that touches `src/naics_embedder/decision/`, `src/naics_embedder/panels/regressor.py`,
  `panels/qcew_rows.py`, `panels/regressor_splits.py`, `panels/ridge.py`, `conf/data/`, `uv.lock`
  or `specs/findings/`. A plan 18 change to `cli/commands/tools.py` alone is no stop: PR A is cut
  after it.
- The level pins fail or skip (Pre-flight Step 3).
- PR A's CI fails twice, or a review finding would change `conf/data/economic_candidates.yaml`
  beyond the new key.
- Task 2's guard fails: MAIN is not on `main`, not clean, or has commits `origin/main` lacks.
- `uv.lock` differs from the pinned value, or `uv sync --locked --check` fails.
- An output path exists, or the store is not writable.
- `log_state.py` exits 1 at any step.
- A dry run exits non-zero.
- MAIN's HEAD or its `conf/data/economic_candidates.yaml` moved during Tasks 3–5 (the drift
  guard, Task 4 Step 2). Stop before the next read or the decision.
- A read exits non-zero, or `check_arm.py` fails after it. Record what it printed first, and
  never re-run the read.
- `tools decide` exits non-zero: a pairing refusal, `TieUnresolvedError` or anything else.
- Any check in Task 6 fails.
- A step would do anything the Project rules forbid.

## File structure

| Action | Path | Task | Responsibility |
|---|---|---|---|
| Modify | `src/naics_embedder/decision/candidates.py` | 1 | `CandidateList.text_only_provenance_sha256` |
| Modify | `src/naics_embedder/decision/economic.py` | 1 | The gate refuses another text-only provenance |
| Modify | `src/naics_embedder/decision/trained.py` | 1 | The import refuses it |
| Modify | `src/naics_embedder/cli/commands/tools.py` | 1 | The deterministic read refuses it before the panel loads |
| Modify | `conf/data/economic_candidates.yaml` | 1 | The pin and its comment |
| Modify | `tests/fixtures/economic.py` | 1 | `text_only_pins` and the list helpers' new keyword |
| Modify | `tests/unit/test_economic_candidates.py`, `test_decision_economic.py`, `test_cli_commands.py` | 1 | The pin's tests, and the helpers' callers |
| Modify | `docs/usage.md`, `CLAUDE.md`, `README.md`, `tests/README.md` | 1 | Economic-read's input checks; the test counts |
| Create | `specs/findings/economic-decision.md` | 7 | The finding |
| Modify | `CLAUDE.md` | 7 | One sentence: the decision, its date and the finding |
| Modify | `specs/naics-artifact-redesign-roadmap.md`, `specs/deferred_items.md`, `specs/history.md` | Completion | Stamp, ticks, history entry |
| Move | `specs/plans/19-economic-decision-read.md` | Completion | Markup and Handoff, then `specs/plans/completed/` |

No other source, config or lock file changes.

## Pre-flight (controller, inline, before Task 1)

- [ ] **Step 1: Confirm the workspace**

```bash
git -c core.fsmonitor=false fetch origin
git cat-file -e origin/main:specs/plans/19-economic-decision-read.md; echo "exit=$?"
git rev-parse origin/main
```

Expected: `exit=0`, because this plan has merged. Record the SHA as BASE in the ledger.

- [ ] **Step 2: Parallel work**

```bash
git -c core.fsmonitor=false worktree list
gh pr list --repo lowmason/naics-embedder --state open --json number,title,files --jq '.[] | {number, title, files: [.files[].path]}'
git log --oneline 8329e55..origin/main -- src/naics_embedder/decision src/naics_embedder/panels src/naics_embedder/cli/commands/tools.py conf/data uv.lock specs/findings
```

Expected: no open PR and no commit touches the Stop-and-ask paths. Record any plan 18 change to
`tools.py`. Worktrees under `~/Projects/copilot-worktrees/` are review tooling, not sessions.

- [ ] **Step 3: The level pins, in this lock's environment**

```bash
uv sync --locked
uv run --locked --no-sync pytest -q -rs -n auto tests/unit/test_regressor_panel.py tests/unit/test_regressor_qcew_rows.py tests/unit/test_decision_economic.py tests/unit/test_economic_candidates.py; echo "exit=$?"
```

Expected: `exit=0`, and no skip reason reading "pinned where captured, on macOS arm64". A
failure or that skip means that the environment changed since the pins were captured.

- [ ] **Step 4: The ledger and SCRATCH**

```bash
mkdir -p logs /tmp/naics_plan19
```

Write `logs/plan19_execution.md` with the Write tool: a title, BASE, the date and Steps 1–3's
outputs.

## Tasks

### Task 1: Pin the shared text-only provenance

Nothing in this task reads real data.

**Files:**
- Modify: `src/naics_embedder/decision/candidates.py` (`CandidateList`)
- Modify: `src/naics_embedder/decision/economic.py` (`_require_shared_text_only`, and the
  `check_economic_arm` docstring's step 1)
- Modify: `src/naics_embedder/decision/trained.py` (`_check_source`)
- Modify: `src/naics_embedder/cli/commands/tools.py` (`_deterministic_candidate`, and the
  economic-read docstring's step 4)
- Modify: `conf/data/economic_candidates.yaml`
- Modify: `tests/fixtures/economic.py`
- Test: `tests/unit/test_economic_candidates.py`, `tests/unit/test_decision_economic.py`,
  `tests/unit/test_cli_commands.py`
- Modify: `docs/usage.md` (`tools economic-read`, the inputs of step 4), and the collected-test
  count in `CLAUDE.md`, `README.md` and `tests/README.md`

**Interfaces:**
- Consumes: nothing from an earlier task.
- Produces: `CandidateList.text_only_provenance_sha256: str`, required, with the `_SHA256`
  pattern. The three refusals, with this exact text, where `sha256` is the provenance file's
  and `pin` is the list's:

  ```python
  # The gate, in _require_shared_text_only(name, text_only, candidates), after the table's check
  f"{name}: its text-only provenance {sha256[:12]} is not the shared text-only table's "
  f'provenance {pin[:12]} that the frozen list pins'
  # The import, in trained._check_source, after the table's check
  f"{entry.name}: its source's text-only provenance {sha256[:12]} is not the shared text-only "
  f"table's provenance {pin[:12]} that the frozen list pins"
  # The read, in _deterministic_candidate, right after the table's pin, before the JSON is parsed
  _require_pin(
      str(text_provenance), frozen.text_only_provenance_sha256,
      "the shared text-only table's provenance"
  )
  ```

- Produces (fixtures): `TEXT_ONLY_PROVENANCE_SHA256 = 'd' * 64`, beside `CODEBOOK_SHA256`;
  `text_only_pins(path: Path) -> Dict[str, str]`, which returns `text_only_sha256` and
  `text_only_provenance_sha256`, the sha256 values of the table and of `provenance_path(path)`.
  `fixture_list`, `list_for` and `write_candidate_list` gain the keyword
  `text_only_provenance_sha256: str = TEXT_ONLY_PROVENANCE_SHA256`.

- [ ] **Step 1: Cut PR A's branch**

```bash
git -c core.fsmonitor=false checkout --no-track -B claude/economic-text-only-provenance-pin origin/main
```

- [ ] **Step 2: Write the failing tests**

In `tests/unit/test_economic_candidates.py`:

```python
# Added to test_the_committed_list_is_the_specs_eighteen_candidates
assert committed.text_only_provenance_sha256 == (
    'bd21c628d06b96c8a0ffe1318c18f81199cb00ca8862560dbdb8dc57887ea05e'
)

def test_a_list_without_the_text_only_provenance_pin_is_refused(tmp_path):
    '''Review Focus 4: a list file from before the pin.'''
    content = yaml.safe_load((ROOT / ECONOMIC_CANDIDATES).read_text())
    del content['text_only_provenance_sha256']
    old = tmp_path / 'old.yaml'
    old.write_text(yaml.safe_dump(content, sort_keys=False))
    with pytest.raises(ValidationError, match='text_only_provenance_sha256'):
        load_candidate_list(old)
    content['text_only_provenance_sha256'] = 'not-a-sha256'
    old.write_text(yaml.safe_dump(content, sort_keys=False))
    with pytest.raises(ValidationError, match='text_only_provenance_sha256'):
        load_candidate_list(old)
```

In `tests/unit/test_decision_economic.py`, over the file's `arm` and `frozen` fixtures and its
`trained_read` fixture, beside `test_a_trained_arms_text_only_table_is_the_lists`:

```python
REFUSED = (
    "its text-only provenance .* is not the shared text-only table's provenance "
    'ffffffffffff that the frozen list pins'
)

def test_a_deterministic_arms_text_only_provenance_is_the_lists(arm, frozen, store):
    check_economic_arm(arm, store, frozen)  # the fixture list's two pins pass
    with pytest.raises(ValueError, match=f'^tree: {REFUSED}'):
        check_economic_arm(
            arm, store, frozen.model_copy(update={'text_only_provenance_sha256': 'f' * 64})
        )

def test_a_trained_arms_text_only_provenance_is_the_lists(trained_read, store):
    arm, frozen, _ = trained_read
    with pytest.raises(ValueError, match=f'^euclidean-d3: {REFUSED}'):
        check_economic_arm(
            arm, store, frozen.model_copy(update={'text_only_provenance_sha256': 'f' * 64})
        )
```

In `tests/unit/test_cli_commands.py`, three tests over `_economic_inputs`:

- `test_economic_read_refuses_a_text_only_provenance_other_than_the_lists`, parametrized over
  `DETERMINISTIC` and over `[[], ['--dry-run']]` (ids `read`, `dry-run`). Copy the inputs'
  text-only table to `tmp_path / 'copy'`, and write its provenance beside it as the same JSON
  re-serialized with `indent=4`: the same content in other bytes, so the table's pin and
  `provenance_fields` both pass. Read with `--text-only` set to the copy. Assert
  `_refused_read(result, f"{copy_provenance} is not the shared text-only table's provenance that
  the frozen list pins")`, `_untouched(panel, loaded)` and that `tmp_path / 'read.json'` does not
  exist.
- `test_economic_read_refuses_a_trained_source_with_another_text_only_provenance`, parametrized
  over `[[], ['--dry-run']]`. Write the inputs' list again with
  `text_only_provenance_sha256='f' * 64`: `frozen = inputs.frozen.model_copy(update=...)`, then
  `inputs._replace(candidates=_write_list(tmp_path / 'repinned.yaml', frozen), frozen=frozen)`.
  Read `euclidean-d3`. Assert exit 1,
  `"euclidean-d3: its source's text-only provenance"` and
  `'ffffffffffff that the frozen list pins'` in `_output(result)`, and `_untouched(panel, loaded)`.
- `test_decide_refuses_a_record_that_stores_another_text_only_provenance` (Review Focus 3). Read
  the four fixture candidates as `test_economic_reads_and_decide_record_a_decision_over_both_kinds`
  does. Store a text-only table with the same bytes and another provenance, as
  `test_a_trained_arms_text_only_table_is_its_sources` does:
  `store.put_text_only(write_text_only(tmp_path / 'other', revision='other', codes=CODEBOOK))`,
  and assert that its table's sha256 is the record's. Set `tree-d3`'s record's `text_only` to it,
  and write that record as a new file. Run `tools decide` over the edited record and the other
  three. Assert exit 1, `'Decision failed:'` and `"tree-d3: its text-only provenance"` in
  `_output(result)`, and that the decision's `--output` does not exist.

- [ ] **Step 3: Run the new tests to verify they fail**

```bash
uv run --locked --no-sync pytest -q tests/unit/test_economic_candidates.py tests/unit/test_decision_economic.py tests/unit/test_cli_commands.py -k 'text_only_provenance or committed_list'; echo "exit=$?"
```

Expected: FAIL. The committed list has no `text_only_provenance_sha256`, and the helpers do not
take the keyword.

- [ ] **Step 4: Implement the pin**

- `candidates.py`: add `text_only_provenance_sha256: str = Field(pattern=_SHA256)` after
  `text_only_sha256`, with its Attributes line: "The provenance file beside the shared text-only
  table, which every candidate's text-only reference must store."
- `economic.py`, `trained.py` and `tools.py`: the three refusals (Interfaces). In `tools.py`,
  `_deterministic_candidate`'s docstring and step 4 of `economic_read_command`'s docstring say
  that `--text-only` and the provenance beside it are the ones the list pins.
- `conf/data/economic_candidates.yaml`: add
  `text_only_provenance_sha256: bd21c628d06b96c8a0ffe1318c18f81199cb00ca8862560dbdb8dc57887ea05e`
  after `text_only_sha256`. Extend the header's first bullet to name it: the provenance file
  beside the text-only table, which every record's text-only reference stores.
- `tests/fixtures/economic.py`: the fixtures (Interfaces). Then, in the three test files, replace
  each `text_only_sha256=sha256_file(X)` with `**text_only_pins(X)`
  (`grep -n 'text_only_sha256=sha256_file' tests/`). Where a test takes the table's pin from its
  arms (`{arm.text_only.table.sha256 for arm in ...}`), take the provenance pin from
  `arm.text_only.provenance.sha256` in the same way. A call that passes a wrong
  `text_only_sha256` on purpose keeps it and takes the default provenance pin. `_economic_inputs`
  writes its list with both pins.
- `docs/usage.md`, `tools economic-read`, step 4's inputs: a deterministic candidate's
  "`--text-only` matches the list's pin" becomes "`--text-only` and the provenance beside it
  match the list's two pins". A trained candidate's bullet adds: "the source record's text-only
  table and its provenance are the ones the list pins".

- [ ] **Step 5: Run the touched tests to verify they pass**

```bash
uv run --locked --no-sync pytest -q -n auto tests/unit/test_economic_candidates.py tests/unit/test_decision_economic.py tests/unit/test_cli_commands.py; echo "exit=$?"
```

Expected: `exit=0`.

- [ ] **Step 6: Run the full checks**

```bash
uv run --locked --no-sync pytest -n auto -q; echo "exit=$?"
uv run --locked --no-sync pytest --collect-only -q -q | tail -1
./scripts/format_code.sh src/naics_embedder/decision/candidates.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/trained.py src/naics_embedder/cli/commands/tools.py tests/fixtures/economic.py tests/unit/test_economic_candidates.py tests/unit/test_decision_economic.py tests/unit/test_cli_commands.py
uv run --locked --no-sync ruff check src/ tests/; echo "exit=$?"
./scripts/format_code.sh --check --all; echo "exit=$?"
uv run --locked --no-sync mkdocs build --strict; echo "exit=$?"
```

Expected: every `exit=0`. Find the stated count (the number beside "collect") with
`grep -n 'collect' CLAUDE.md README.md tests/README.md` (4,479 at plan time; plan 18 may change
it first), and put the new collected count in its place in all three files. The unit file count
stays as the files state it.

- [ ] **Step 7: Commit**

```bash
git add src/naics_embedder/decision/candidates.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/trained.py src/naics_embedder/cli/commands/tools.py conf/data/economic_candidates.yaml tests/fixtures/economic.py tests/unit/test_economic_candidates.py tests/unit/test_decision_economic.py tests/unit/test_cli_commands.py docs/usage.md CLAUDE.md README.md tests/README.md
git commit -m "feat(decision): pin the shared text-only provenance in the frozen list"
```

- [ ] **Step 8: Local review, then PR A**

```bash
mkdir -p /tmp/codex-review && codex exec review -m gpt-6-astra -c sandbox_mode=read-only --base origin/main -o /tmp/codex-review/plan19-pr-a.md; echo "exit=$?"
```

Triage each finding (receiving-code-review), fix what is real in a new commit, and record the
rest with a reason in the ledger. Then push and open PR A against `main`. Its body names the
deferred item that it closes, says that the list's sha256 changes, and says that no real read
happens before it merges. Bind it in the app (`ccd_pr` `get_status`, then `bind_pr` if needed)
and turn on Auto-fix (`set_monitor`). Auto-merge is off in this repository, so Task 2 merges by
hand.

### Task 2: Merge PR A and ready MAIN

Nothing in this task reads real data.

**Files:** none in the repository. Writes MAIN's `.venv` only.

**Interfaces:**
- Consumes: PR A (Task 1).
- Produces: READ_SHA, MAIN's HEAD at PR A's merge or later; LIST_SHA, the sha256 of MAIN's
  `conf/data/economic_candidates.yaml` at READ_SHA; the environment receipt; free output paths.

- [ ] **Step 1: Merge PR A**

Wait for its checks with `gh pr checks <n> --watch`, run in the background. Once every listed
check has passed on the head SHA, merge with
`gh pr merge <n> --merge --match-head-commit <sha> --delete-branch`. Confirm
`gh pr view <n> --json state,mergeCommit` shows `MERGED`, and that the remote branch is gone.
Report `CI passed, PR merged, remote deleted`.

- [ ] **Step 2: Fast-forward MAIN with the guard**

Run plan 15's guard (Task 1 Step 1) in MAIN: on `main`, `git status --porcelain` empty,
`git log --oneline origin/main..main` empty, then
`git -c core.fsmonitor=false merge --ff-only origin/main`.

```bash
cd /Users/lowell/Projects/naics-embedder && git rev-parse HEAD origin/main && git merge-base --is-ancestor <PR A's merge SHA> HEAD; echo "exit=$?"
```

Expected: two equal SHAs, then `exit=0`. Record HEAD as READ_SHA.

- [ ] **Step 3: Sync MAIN's environment and record the list**

```bash
cd /Users/lowell/Projects/naics-embedder && shasum -a 256 uv.lock conf/data/economic_candidates.yaml
cd /Users/lowell/Projects/naics-embedder && uv sync --locked; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && uv sync --locked --check; echo "exit=$?"
cd /Users/lowell/Projects/naics-embedder && grep -n 'text_only_provenance_sha256' conf/data/economic_candidates.yaml
```

Expected: `uv.lock`'s pinned sha256; `exit=0` twice; the pin line with `bd21c628…`. Record the
list's sha256 as LIST_SHA: this is the Exit's "recorded again". It differs from plan 16's list
(98c2b5c8…).

- [ ] **Step 4: The environment receipt**

Write `SCRATCH/env.py` as plan 15's `env.py` (Task 1 Step 3), and run it from MAIN with
`uv run --locked --no-sync python /tmp/naics_plan19/env.py`. Expected: the pinned read
environment, and every thread variable `unset`. A newer macOS point release changes only the
platform string: record it and go on. If a thread variable is set, unset it in the shell of every
later read.

- [ ] **Step 5: The outputs**

```bash
ls /Users/lowell/naics-artifacts/records/economic
test -w /Users/lowell/naics-artifacts; echo "writable=$?"
df -h /Users/lowell/naics-artifacts
```

Expected: only `growth_margins.json`; `writable=0`; free space in gigabytes. If any candidate
record or `decision.json` exists, stop and ask: something may already have read a candidate.

- [ ] **Step 6: Ledger**

Copy Steps 1–5's outputs, READ_SHA and LIST_SHA into the ledger.

### Task 3: Gate A — the log and the dry runs

**Files:** none in the repository. Reads the selection log, the store and the real tables.

**Interfaces:**
- Consumes: READ_SHA, LIST_SHA (Task 2).
- Produces: LINES_BEFORE and SHA_BEFORE, the log before the reads; 18 passing dry runs.

- [ ] **Step 1: Ask for Gate A**

Tell the user: PR A merged at READ_SHA; LIST_SHA; the environment receipt. Ask for approval to
start the stage's real-data steps: the log check and 18 dry runs, which load no panel and log
nothing. Wait for a clear yes.

- [ ] **Step 2: The log's state**

Write `SCRATCH/log_state.py` from plan 15's `log_state.py` (Task 2 Step 4), with these changes.
It takes the log, and optionally `PREFIX_LINES PREFIX_SHA`:

- It prints the line count, the record count, the final newline and the sha256, the records that
  lack a field, the tally, the records other than validation reads, and each regime's openings
  under the held-out draw. It exits 1 on a missing newline, a malformed record, any record other
  than a validation read, or any opening.
- It groups the growth records (`growth_seen`, `growth_heldout`) by purpose. It exits 1 on a
  growth record whose purpose is neither the margin read's nor PURPOSE, and it prints how many
  carry each.
- Given a prefix, it exits 1 unless the log's first `PREFIX_LINES` lines hash to `PREFIX_SHA`,
  and it prints every record after them that is not PURPOSE's: another session's reads.

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan19/log_state.py /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl > /tmp/naics_plan19/log_before.txt; echo "exit=$?"
cat /tmp/naics_plan19/log_before.txt
```

Expected, at plan time: `exit=0`, 159 records with the pinned sha256, no openings, the margin
read's two growth records and none with PURPOSE. Records that other sessions added since plan
time are no stop if they are validation reads of other panels: list them in the ledger. Record
`lines` as LINES_BEFORE and `sha256` as SHA_BEFORE.

- [ ] **Step 3: The 18 dry runs**

For each candidate, in the list's order, run from MAIN's root:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools economic-read --candidate NAME \
  --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet \
  --margins /Users/lowell/naics-artifacts/records/economic/growth_margins.json \
  --store /Users/lowell/naics-artifacts \
  --text-only /Users/lowell/Projects/naics-embedder/checkpoints/plan9_exit/text_only.parquet \
  --runs-root /Users/lowell/Projects/naics-embedder/checkpoints \
  --purpose 'economic candidate read: redesign roadmap Stage 3, plan 19 (Req 29)' \
  --log /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl \
  --lock /Users/lowell/Projects/naics-embedder/uv.lock \
  --output /Users/lowell/naics-artifacts/records/economic/NAME.json \
  --dry-run > /tmp/naics_plan19/dry-NAME.txt 2>&1; echo "exit=$?"
```

Expected: 18 times `exit=0` and `Dry run: every check passed for NAME`. These pass the new pin
against the real text-only provenance and all ten source records before anything is logged. The
line says "every check", but the panel's identity, the table's coverage and width, and a
deterministic candidate's D9 comparison wait for the real read (P3). Then:

```bash
shasum -a 256 /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl
ls /Users/lowell/naics-artifacts/records/economic
```

Expected: SHA_BEFORE, and only `growth_margins.json`. If the sha256 changed, run `log_state.py`
with `LINES_BEFORE SHA_BEFORE`: it must exit 0 with no PURPOSE record, and the new records are
another session's. List them in the ledger.

- [ ] **Step 4: Ledger**

Copy Steps 2–3's outputs into the ledger.

### Task 4: Gate B — the 18 reads

**Files:** none in the repository. Appends 118 records to MAIN's log and writes 18 records and
their objects in the store.

**Interfaces:**
- Consumes: Task 3's results.
- Produces: `STORE/records/economic/<name>.json` for the 18 candidates, each passing
  `check_arm.py`.

- [ ] **Step 1: Ask for Gate B**

First run `mcp__ccd_session_mgmt__list_sessions` and note each running session whose `cwd` is
under MAIN. Show the user Task 3's results, the read command, the order and those sessions. Say
that the reads append 118 growth validation reads to MAIN's log, open no outer set and are never
re-run. Ask that, until Task 6 passes, no other session merges a PR or syncs MAIN, and none
reads a panel into MAIN's log. Ask for a go, and wait for a clear yes.

- [ ] **Step 2: Read each candidate**

Right before the first read, run `log_state.py` again with `LINES_BEFORE SHA_BEFORE`. Expected:
`exit=0`, and no PURPOSE record. Then, for each candidate in the list's order, first run the
drift guard:

```bash
cd /Users/lowell/Projects/naics-embedder && test "$(git rev-parse HEAD)" = READ_SHA && test "$(shasum -a 256 conf/data/economic_candidates.yaml | cut -d' ' -f1)" = LIST_SHA; echo "exit=$?"
```

Expected: `exit=0`. Another session's post-merge cleanup fast-forwards MAIN, and a read after
that would run at another commit. On `exit=1`, stop before the read. Then run Task 3 Step 3's
command without `--dry-run`, writing `/tmp/naics_plan19/read-NAME.txt`. Run it in the background
and wait for it to exit. Never start a read before the previous one's Step 3 passes. Expected:
`exit=0`, and `Economic read 'NAME': 1 run` (deterministic), `5 runs` or `10 runs`
(`hyperbolic-d16`). Record the first read's wall time, and tell the user the estimate for the
other 58 runs.

- [ ] **Step 3: Check each record before the next read**

Write `SCRATCH/check_arm.py`. It takes a record path. It loads MAIN's list with
`load_candidate_list`, the margins with `read_record(..., RelativeMarginRecord)`, the store and
the record with `read_record(..., EconomicArmRecord)`. Then it runs, in this order:

1. `check_economic_arm(arm, store, frozen)`;
2. `check_reference_pairing(margins, arm, store)`;
3. `check_economic_margins_first([arm], margins)`;
4. a log check: the log's records whose `detail.run` is one of the arm's run ids number twice
   its runs, and each is a `read` of the `validation` split of `growth_seen` or
   `growth_heldout`, with PURPOSE and the held-out draw's fingerprint.

It prints the name, the run count, each run's id and seed, the mean `covariates+embedding` MSE on
each growth panel and the record's environment. It exits 1 on the first failure, with its message.

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan19/check_arm.py /Users/lowell/naics-artifacts/records/economic/NAME.json; echo "exit=$?"
```

Expected: `exit=0`. If the first read fails the pairing check, stop: suspect BLAS threading
first (finding: growth-margins §4), and do not read another candidate.

- [ ] **Step 4: The log after the reads**

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan19/log_state.py /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl LINES_BEFORE SHA_BEFORE > /tmp/naics_plan19/log_after.txt; echo "exit=$?"
```

Expected: `exit=0`; 118 PURPOSE growth records; the margin read's two; no opening. With no other
session's records, the log has LINES_BEFORE + 118 lines.

- [ ] **Step 5: Ledger**

Copy each read's output, each `check_arm.py` output and Step 4's output into the ledger, with
`shasum -a 256` of the 18 records.

### Task 5: The decision

**Files:** none in the repository. Writes `STORE/records/economic/decision.json` and the stored
list.

**Interfaces:**
- Consumes: the 18 records (Task 4), LIST_SHA (Task 2).
- Produces: DECISION_SHA, the sha256 of the decision record.

- [ ] **Step 1: Run the decision**

```bash
test ! -e /Users/lowell/naics-artifacts/records/economic/decision.json; echo "absent=$?"
```

Expected: `absent=0`. Run Task 4 Step 2's drift guard: `exit=0`. Then run, in the background,
from MAIN's root:

```bash
cd /Users/lowell/Projects/naics-embedder && HF_HUB_OFFLINE=1 COLUMNS=200 uv run --locked --no-sync naics-embedder tools decide \
  --arm /Users/lowell/naics-artifacts/records/economic/text-only-d8.json \
  --arm /Users/lowell/naics-artifacts/records/economic/text-only-d16.json \
  --arm /Users/lowell/naics-artifacts/records/economic/text-only-d32.json \
  --arm /Users/lowell/naics-artifacts/records/economic/tree-d8.json \
  --arm /Users/lowell/naics-artifacts/records/economic/tree-d16.json \
  --arm /Users/lowell/naics-artifacts/records/economic/tree-d32.json \
  --arm /Users/lowell/naics-artifacts/records/economic/smoothed-d8.json \
  --arm /Users/lowell/naics-artifacts/records/economic/smoothed-d16.json \
  --arm /Users/lowell/naics-artifacts/records/economic/smoothed-d32.json \
  --arm /Users/lowell/naics-artifacts/records/economic/hyperbolic-d16.json \
  --arm /Users/lowell/naics-artifacts/records/economic/euclidean-d8.json \
  --arm /Users/lowell/naics-artifacts/records/economic/euclidean-d16.json \
  --arm /Users/lowell/naics-artifacts/records/economic/euclidean-d32.json \
  --arm /Users/lowell/naics-artifacts/records/economic/hyperbolic-d8.json \
  --arm /Users/lowell/naics-artifacts/records/economic/hyperbolic-d32.json \
  --arm /Users/lowell/naics-artifacts/records/economic/spherical-d8.json \
  --arm /Users/lowell/naics-artifacts/records/economic/spherical-d16.json \
  --arm /Users/lowell/naics-artifacts/records/economic/spherical-d32.json \
  --margins /Users/lowell/naics-artifacts/records/economic/growth_margins.json \
  --candidates /Users/lowell/Projects/naics-embedder/conf/data/economic_candidates.yaml \
  --name economic --question 'Which candidate, if any, is the economic artifact?' \
  --store /Users/lowell/naics-artifacts \
  --output /Users/lowell/naics-artifacts/records/economic/decision.json \
  > /tmp/naics_plan19/decide.txt 2>&1; echo "exit=$?"
```

Expected: `exit=0`. The output has each comparison with both intervals, three `Propagation:`
lines, `Adopted over the reference:`, `Non-dominated:`, `Tie order:` and `Chosen:`. A chosen
candidate or `none; the reference covariates+ancestors stands` are both results, not failures.

- [ ] **Step 2: Record it**

```bash
shasum -a 256 /Users/lowell/naics-artifacts/records/economic/decision.json /Users/lowell/naics-artifacts/records/economic/growth_margins.json
```

Expected: a sha256 for the decision, recorded as DECISION_SHA, and the margin record's pinned
sha256. Copy `decide.txt` and both values into the ledger.

### Task 6: Verify — read only

**Files:** none.

**Interfaces:**
- Consumes: DECISION_SHA, LIST_SHA, READ_SHA, LINES_BEFORE, SHA_BEFORE.
- Produces: the provenance table and the decision summary that the finding copies.

- [ ] **Step 1: The decision record**

Write `SCRATCH/verify.py`. It reads the record with `read_record(...,
EconomicDecisionRecord)` and exits 1 on the first check that fails:

1. The stored list (`record.candidates`) has sha256 LIST_SHA, and it loads with
   `text_only_provenance_sha256` `bd21c628…`.
2. `record.margins` equals the margin record that `read_record` loads from the pinned path, and
   that file's sha256 is still the pinned one.
3. The arms are the 18 candidates, in the list's order.
4. Every run of every arm passes `check_candidate_provenance(run.provenance.content, entry,
   frozen)`. The script prints one row per candidate: kind, run count, provenance file path and
   sha256, and the builder (or `export` for a trained run's provenance).
5. `record.propagation` holds three comparisons: `smoothed-dN` over `text-only-dN` for d = 8, 16
   and 32.
6. With `record.eligible` empty, `record.chosen` is None. Otherwise `record.chosen` is the first
   name in `record.tie_order`, and it is in `record.eligible`.

It prints `record.eligible`, `record.non_dominated`, `record.tie_order`, `record.chosen`, each
candidate's `heldout_gain`, and each propagation comparison's verdict and panel intervals.

```bash
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan19/verify.py /Users/lowell/naics-artifacts/records/economic/decision.json; echo "exit=$?"
```

Expected: `exit=0`.

- [ ] **Step 2: Nothing else moved**

```bash
cd /Users/lowell/Projects/naics-embedder && git rev-parse HEAD && shasum -a 256 uv.lock
cd /Users/lowell/Projects/naics-embedder && uv run --locked --no-sync python /tmp/naics_plan19/log_state.py /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl LINES_BEFORE SHA_BEFORE; echo "exit=$?"
```

Expected: READ_SHA, the pinned lock, and `exit=0` with the same counts as Task 4 Step 4. Only two
growth records carry the margin read's purpose, so `tools growth-margins` did not run. Copy
Steps 1–2's outputs into the ledger. From here on MAIN may follow `origin/main` again.

### Task 7: The finding

**Files:**
- Create: `specs/findings/economic-decision.md`
- Modify: `CLAUDE.md` (the economic decision paragraph under "Reference Campaign and
  Verification")

- [ ] **Step 1: Cut PR B's branch**

```bash
git -c core.fsmonitor=false fetch origin
git -c core.fsmonitor=false checkout --no-track -B claude/stage-3-economic-decision origin/main
```

- [ ] **Step 2: Write the finding**

`specs/findings/economic-decision.md`, in `specs/findings/growth-margins.md`'s form. It opens
with `**Status: FINAL (<date>).**`, names roadmap Stage 3 and plan 19, and states the decision in
its first paragraph. Its sections:

1. **Sources.** A table of each input with its sha256: the codebook, the text-only table and its
   provenance, the list (LIST_SHA), the margin record, `uv.lock`, the 18 candidate records and
   the decision record (DECISION_SHA). The commit the reads ran at (READ_SHA), from MAIN's root
   on `main` with a clean tree.
2. **The reads.** The command template, the environment receipt (platform, Python, numpy, polars,
   BLAS, thread variables) and each candidate's run count and mean `covariates+embedding` MSE on
   both panels.
3. **The decision.** Each candidate against `covariates+ancestors`: Δ, both intervals and δ on
   each panel, and adopted or not. Then the eligible, the non-dominated set, the tie order, the
   held-out gains and the chosen candidate, or that the reference stands and there is no economic
   artifact.
4. **Propagation (split Req 19).** At each d, `smoothed-dN` against `text-only-dN`, with both
   intervals. The answer, by spec §4.7's rule: propagation helps at d if the smoothed candidate
   is adopted over its twin. Say what follows: a later spec may revisit the parked graph stage
   for this artifact, or the finding records no evidence that it helps and the graph stage stays
   parked.
5. **Provenance.** Task 6 Step 1's table: each candidate's provenance names text and tree inputs
   only (Req 29).
6. **The selection log.** Records and sha256 before and after, the 118 PURPOSE reads by panel,
   no record other than validation reads, no opening of either regime, and the margin read's two
   growth records only.
7. **What embedding roadmap Stage 12 reads.** The decision record's path and DECISION_SHA, the
   chosen candidate's table (or none), and the environment under which the opening must pair.

- [ ] **Step 3: The `CLAUDE.md` sentence**

After "The growth margins were fixed once, on 2026-10-17 (`specs/findings/growth-margins.md`).",
add one sentence: "The economic decision was recorded on <date>
(`specs/findings/economic-decision.md`): <the chosen candidate, or the reference
`covariates+ancestors` stands>."

- [ ] **Step 4: Check and commit**

Measure the finding's line lengths with Python: at most 100 outside code blocks and tables.
Then:

```bash
git add specs/findings/economic-decision.md CLAUDE.md
git commit -m "docs(findings): record the economic decision"
```

## Plan completion (controller, inline)

Run the writing-plans Plan Completion Protocol on PR B's branch, before it is pushed:

- **Deferred items.** Before editing `specs/deferred_items.md` or the roadmap, check for other
  running sessions that edit them, and ask if one does. Tick § 16-economic-candidates' second
  item `→ done <date> (plan 19)`. The first item (the dry-run text) stays open (P3).
- **Handoff.** No later stage of this roadmap consumes Stage 3. Add one bullet for embedding
  roadmap Stage 12's economic opening: the decision record's path and DECISION_SHA, the finding,
  and the read environment it must pair under. The stamp then names embedding roadmap Stage 12.
- **Tick and stamp** Stage 3 in `specs/naics-artifact-redesign-roadmap.md`, run the backlog
  statistics and the deferral scan, retire this plan to `specs/plans/completed/` and append its
  `specs/history.md` entry, in the protocol's commits.

## Integration (controller, inline)

1. Push PR B and open it against `main`. Merge it as Task 2 Step 1 merged PR A, and report
   `CI passed, PR merged, remote deleted`.
2. Fast-forward MAIN with the guard, delete the local branches, and `git fetch --prune`.
3. Copy the ledger to `MAIN/logs/plan19_execution.md`. Delete SCRATCH and
   `/tmp/codex-review/plan19-pr-a.md`.
4. Write the resume handoff (prepare-handoff): the next step is a resume of the roadmap, which
   routes no stage of its own from here.

## Execution

Planning ran on Opus. Execute in a fresh session on Sonnet (`claude --model sonnet`), inline with
executing-plans (P9), from an app worktree, with this file as the handoff.
