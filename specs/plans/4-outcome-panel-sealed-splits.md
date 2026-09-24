# Outcome Panel and Sealed Splits Implementation Plan

**Status: COMPLETE (2026-09-24)** — executed via executing-plans; deferred items in specs/deferred_items.md (four review Minors, for Stages 5 and 6)

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-embedding-roadmap.md, Stage 2 — on plan completion, tick the stage and
> re-validate later stages against what shipped.

**Goal:** Give every Census index entry exactly one role (examples-channel text, or a training,
validation or test query) and rebuild the examples channel from examples-role entries only. Around
that, build the leakage checker, the text→code decoding scorer, the selection log and the sealed
test split behind a logged opening, each backed by a test, and score a training-free stub on the
real validation split.

**Architecture:** A new package, `naics_embedder.panels`, holds the logic:

- `leakage`: exact and near-duplicate matching.
- `index_roles`: per-code quotas, held-out eligibility, the frozen table and its checks.
- `decoding`: the scorer.
- `selection_log`: the selection log.
- `outcome`: the panel object.
- `lexical_encoder`: a training-free stub arm.

`naics-embedder data roles` draws the role table once, and it is committed as
`conf/data/index_roles.csv`. From then on `data preprocess` applies it: the examples channel holds
examples-role entries only, and every entry is written with its role to
`data/naics_index_roles.parquet`. A supervision bundle can carry that table as an optional
`index_roles` member under the unchanged contract `stage3-supervision-v1`. No bundle is built
here: Stage 5 bumps the contract and rebuilds once.

**Tech Stack:** Python 3.10 and 3.12 (CI runs both); polars; numpy; scikit-learn
(`CountVectorizer`, `HashingVectorizer`); torch (float64 CPU distances); pydantic; typer; pytest
with xdist; ruff and yapf.

## Global Constraints

Every task's requirements include this section.

### The spec (`specs/naics-embedding.md` at d9126ce), verbatim

- Req 1: "**As an outcome representation:** how reliably a point produced from text decodes to
  the right code (Req 3)."
- Req 3: "The outcome use is scored as text→code decoding with the method's own encoder. A
  held-out index entry is encoded as a query and decoded to the nearest six-digit code under the
  arm's own distance".
- Req 3: "**The index file** holds 20,373 entries over 1,010 six-digit codes (verified locally:
  index file). 112130 and 541120 have no entries: they stay decoding candidates but are never
  queries."
- Req 3: "**Every index entry has exactly one role:** examples-channel text, or a training,
  validation or test query, never two (chosen)." … "Test entries are stratified by code."
- Req 3: "**Candidates** are the 1,012 six-digit codes. **Metrics** are exact top-1 accuracy,
  MRR, Hit@k for k ∈ {1, 5, 10}, and hierarchical partial credit. Partial credit is the level of
  the lowest common ancestor of the top-1 code and the truth".
- Req 3: "**Leakage.** No test query may appear, exactly or as a near-duplicate, in any training
  text: titles, descriptions, remaining examples-channel entries, exclusion text, or training
  queries (including the cross-reference activity phrases of Req 8)."
- Req 4: "Each panel has a validation split and a sealed test split" … "Every selection reads
  validation splits only" … "The test splits are opened once, for the final configuration and the
  comparisons recorded for it".
- Verification "Leakage": "Exact and near-duplicate matching of every test query against all
  training text (Req 3's list) finds no exact match. Near-duplicates above a stated similarity are
  removed from the test split and counted."
- Verification "Index-entry roles": "Every index entry holds exactly one role. 112130 and 541120
  are candidates and never queries."
- Verification "Selection hygiene": "A run log shows that every selection read validation splits
  only, and that the test splits were opened once, for the final configuration."
- Verification "Panels", outcome half: "The outcome panel reports top-1, MRR, Hit@k and
  hierarchical partial credit."

### The roadmap (`specs/naics-embedding-roadmap.md`), verbatim

- Stage 2 Produces: "An index-entry role table (examples-channel text, training query, validation
  query, test query; stratified by code, fractions per D4; 112130 and 541120 candidates only) as a
  bundle member; the examples channel rebuilt from examples-role entries only; a leakage checker
  (exact and near-duplicate at a stated similarity) with its removal count; a decoding scorer
  (top-1, MRR, Hit@{1, 5, 10}, lowest-common-ancestor partial credit) over the 1,012 candidates
  under a pluggable distance; a selection log recording which split each selection read; a sealed
  test split behind a logged open call."
- Stage 2 Exit: "Every index entry holds exactly one role and the two entry-less codes never
  appear as queries (a test asserts both); the leakage check finds no exact match and reports the
  near-duplicates removed; the scorer reports all four metrics for a stub encoder on the sealed
  splits; the selection log exists and the test split cannot be read without a logged open."
- D4: "Stage 2's plan sets them, per code and stratified, with a floor of one examples-channel
  entry where a code has enough entries; the fractions are recorded in the stage's Rollout note.
  Stage 2 has no stage spec, so that note is a line under its roadmap entry, beside the stamp".
- D6: "the validation query split's MRR (Req 3); both panels are used only between
  configurations, under Req 5."

### Decisions already made (do not re-ask)

The user answered these four at planning (2026-09-24):

1. **Contract.** `index_roles` is an *optional* member under `stage3-supervision-v1`. Stage 5
   makes the single contract bump (v2: member required, plus D*) and the single rebuild. This plan
   builds no bundle from real data. Bundle 18403d29, the Lambda instances, exact resume and the
   held config commits stay untouched.
2. **Role table.** It is committed as `conf/data/index_roles.csv` (`entry_id`, `code`, `role`;
   about 433 KB), keyed to the pinned index-file sha256. `data preprocess` reads it and never
   regenerates it.
3. **Stub run.** The stub encoder is scored on the real validation split (Task 9). The logged
   test-split opening and read are exercised in a unit test on a fixture panel (Task 4). The real
   test split stays sealed: no `open` record for it exists anywhere.
4. **Overwrite guard.** `data preprocess` refuses to overwrite a descriptions file whose sha256
   equals the `description_fingerprint` of a bundle named by `conf/config.yaml`
   (`supervision.manifest_path`) or `conf/graph.yaml` (`supervision_manifest_path`); `--force`
   overrides. This plan writes real-data output only to the worktree's ignored `data/` and to
   `/tmp`.

This plan's own decisions are stated here so that no reviewer needs to re-derive them:

- **D4 fractions.** Per code, roles get largest-remainder quotas of examples 3/10, training 7/20,
  validation 1/5 and test 3/20, in exact fractions.
  - Remainder ties go to the smallest of four draws from `np.random.default_rng([20260924,
    int(code)])`.
  - Floor: every code with entries keeps at least one examples-channel entry, taken from training
    first, then from the larger of test and validation (test on a tie).
  - Cap: held-out quotas beyond a code's leak-free entries move to training, taken from the larger
    of validation and test (test on ties).
  - Why these shares: most codes are small (median 13 entries), so examples keeps 0.30 and a
    channel always survives. Training gets the largest share because Req 11's task term trains on
    it. Validation (0.20) outweighs test (0.15) because every selection reads validation and D6
    selects on it, while test is read once.
- **Leakage rule.**
  - Normalization: texts are lowercased, keeping ASCII letters and digits, and every other run of
    characters becomes one space.
  - Exact match: the query occurs in a training segment as whole words (equality included).
  - Near-duplicate: the character-trigram Jaccard similarity (scikit-learn `char_wb` trigrams)
    reaches 9/10, compared in integers.
  - Stated limitation: a query whose words appear reordered inside a longer segment is not
    caught.
- **Segments.**
  - Titles are one segment each.
  - Descriptions and exclusion texts split into sentences after `.` or `;`. Each cross-reference
    sentence also yields its activity phrase, the text before `--` (Req 8's phrases).
  - The examples channel splits on `'; '`.
- **Held-out eligibility (conservative).** An entry may become a validation or test query only if
  it matches, exactly or as a near-duplicate, neither static training text (titles, descriptions,
  exclusion text, and the illustrative examples of codes without index entries) nor any other
  index entry.
  - Held-out splits are therefore leak-free whatever the assignment.
  - A final check matches the realized validation and test queries against all realized training
    text and must find 0 exact and 0 near-duplicate matches.
  - The rule covers validation as well as test, because validation MRR is the selection statistic
    (D6).
- **Entry IDs.** `entry_id` is the entry's 0-based row position in the index sheet (20,398 rows),
  stable because the file's sha256 is pinned (`DownloadConfig.index_sha256`).
- **Scorer.**
  - Candidates: all 1,012 six-digit codes.
  - Ties go against the truth: rank = the number of candidates at a distance no greater than the
    truth's, so a constant encoder ranks every truth last.
  - Distances are computed in float64 on the CPU: `euclidean`, `cosine`, `lorentz` (curvature −1,
    time coordinate re-derived), or any callable.
  - LCA levels run from 1 (virtual root) to 6 (same code), and 31-33, 44-45 and 48-49 each count
    as one sector.
- **Selection log.**
  - It is append-only JSON lines at `OutcomePanelConfig.selection_log` (`logs/selection_log.jsonl`,
    gitignored).
  - Events are `read`, `open` and `reopen`. Validation reads are logged; training reads are not.
  - A test read needs a logged opening by the same panel object.
  - A second opening of the same split (the same role-table fingerprint) needs a `reopen_reason`.
- **Split fingerprint.** It is the sha256 of the role assignment's canonical CSV, which equals the
  committed table's file hash. A split is therefore identified the same way whether it is read
  from the CSV, the preprocessing parquet or a bundle member.

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
  - Never push, cherry-pick or merge the held local commits 271f085 "config" and 4ce2834 "graph
    config".
  - Never run bare `git stash`.
  - Commit on this branch only, ending each message with the session's attribution trailer.
- **Data safety.**
  - Never write to the main checkout's `data/`. Its `naics_descriptions.parquet` (sha256
    5107fb83…) is the file bundle 18403d29 pins.
  - Never run `data supervision` or `data all` on real data.
  - Never run `data roles --force` once Task 9 has committed the table.
- **Downloads.** Never download the Census files. Always pass `--source-dir ~/Downloads/Data`
  (the pre-flight checks the cached copies).
- **Deferred items.** Do not promote any open item of `specs/deferred_items.md`.
- **Shared edits.** If another Claude session is active in this repository, hold edits to
  `specs/naics-embedding-roadmap.md` and `specs/deferred_items.md`, and hand the user the exact
  edit instead.
- **Bash tool.** It runs zsh. If it refuses a heredoc or a compound command, run one plain command
  per call and write files with the Write tool.

## Workspace

- **Worktree:** `/Users/lowell/Projects/naics-embedder/.claude/worktrees/outcome-panel-sealed-splits`.
  Run every command from its root.
- **Branch:** `claude/outcome-panel-sealed-splits-522fa329`, cut from origin/main `167d3c9`
  (PR #109). This plan is its first commit.
- **Main checkout:** `/Users/lowell/Projects/naics-embedder` stays on local `main` (4ce2834: plan
  3's e04231e plus the two held commits). Do not check anything out there.
- **Census copies:** `~/Downloads/Data/`. The pre-flight lists their sha256s.
- **Working directory:** the Bash tool can reset its working directory to the main checkout
  between calls. Run `pwd` before Task 9's data commands and before every commit, and if it
  is not this worktree, `cd` back first. A misdirected `data preprocess` would fail in the main
  checkout, because local `main` has no `--source-dir`, and the guard would refuse the pinned
  file anyway. Do not rely on either.

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `src/naics_embedder/panels/__init__.py` | Package docstring | 1 |
| `src/naics_embedder/panels/leakage.py` | Normalization, segments, exact and near-duplicate matching | 1 |
| `src/naics_embedder/supervision/schema.py` | `IndexRole`, `INDEX_ROLES_SCHEMA_VERSION` | 2 |
| `src/naics_embedder/supervision/artifacts.py` | `validate_index_role_table` (Task 2); load-time check of the member (Task 6) | 2, 6 |
| `src/naics_embedder/panels/index_roles.py` | Quotas, eligibility, assignment, the frozen table, consistency checks | 2 |
| `src/naics_embedder/panels/decoding.py` | Distances, LCA level, `score_decoding` | 3 |
| `src/naics_embedder/panels/selection_log.py` | `SelectionLog` | 4 |
| `src/naics_embedder/panels/outcome.py` | `OutcomePanel` (Task 4); `from_bundle` (Task 6) | 4, 6 |
| `src/naics_embedder/data/download_data.py` | Local sources, pinned index, entries, examples channel from roles, overwrite guard | 5 |
| `src/naics_embedder/utils/config.py` | `DownloadConfig` fields (5), `SupervisionBuildConfig.index_roles_parquet` (6), `OutcomePanelConfig` (7) | 5, 6, 7 |
| `conf/data/download.yaml`, `conf/data/supervision.yaml`, `conf/data/outcome_panel.yaml` | Shipped config | 5, 6, 7 |
| `src/naics_embedder/cli/commands/data.py` | `preprocess` options (5), `roles` (7) | 5, 7 |
| `src/naics_embedder/data/supervision_bundle.py` | Optional `index_roles` member; public `generator_revision` | 6 |
| `src/naics_embedder/data/index_role_table.py` | `generate_index_role_table` (`data roles`) | 7 |
| `src/naics_embedder/panels/lexical_encoder.py` | `LexicalTrigramEncoder` | 8 |
| `src/naics_embedder/cli/commands/tools.py` | `tools outcome-baseline` | 8 |
| `conf/data/index_roles.csv`, `conf/data/index_roles_provenance.json` | The frozen table and its provenance (generated) | 9 |
| `specs/findings/outcome-panel-splits.md` | The real-data finding | 9 |
| `docs/usage.md`, `docs/api/outcome_panel.md`, `docs/.nav.yml`, `CLAUDE.md` | Documentation | 10 |

Tests: `tests/unit/test_outcome_leakage.py` (1), `test_index_roles.py` (2),
`test_outcome_decoding.py` (3), `test_outcome_panel.py` (4, 6), `tests/fixtures/naics_sources.py`,
`test_data_download.py` (5), `test_supervision_artifacts.py` and `tests/fixtures/supervision.py`
(6), `test_index_role_table.py` (7), `test_lexical_encoder.py` (8), `test_committed_index_roles.py`
(9), and updates to `test_config.py` and `test_cli_commands.py` (5–8).

## Expected real-data results

These were verified while writing this plan: the same code on the same Census files, with numpy
2.3.4, polars 1.35.1 and scikit-learn 1.9.1 on Python 3.12. The draw is deterministic, so Task 9
must reproduce them exactly.

| Quantity | Value |
|---|---|
| Index sheet rows / "see" rows (`******`) / entries / codes with entries | 20,398 / 25 / 20,373 / 1,010 |
| Role table sha256 (`conf/data/index_roles.csv`, 433,147 bytes) | `05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a` |
| Eligibility: exact_static / near_duplicate_static / exact_entry / near_duplicate_entry | 768 / 813 / 941 / 2,685 (overlapping) |
| Withheld for an exact match / for a near-duplicate only / total / eligible | 1,539 / 2,516 / 4,055 / 16,318 |
| Roles: examples / training / validation / test | 6,118 / 7,200 / 4,042 / 3,013 |
| Codes with each role | 1,010 / 988 / 939 / 893 |
| Realized held-out leakage (validation, test) | exact 0, near-duplicate 0 in both |
| Rebuilt descriptions vs the pinned file (5107fb83…) | only `examples` differs, for 988 codes (all index codes) |
| Lexical stub, validation: queries / codes / candidates | 4,042 / 939 / 1,012 |
| top1 / MRR / Hit@1 / Hit@5 / Hit@10 / mean LCA level | 0.5163 / 0.6165 / 0.5163 / 0.7333 / 0.7971 / 4.3355 |
| Full suite after Task 9 (Python 3.12) | 1380 passed, 1 skipped |

---

## Stop-and-ask conditions

Stop, report, and wait for your human partner when any of these happens:

- A real-data number in Task 9 differs from **Expected real-data results**, above all the role
  table's sha256. Do not commit a different table: the splits are drawn once, and a redraw
  under different code or libraries unseals them.
- A task's tests still fail after its implementation step as written, and the cause is not a
  transcription slip.
- A step would write to the main checkout's `data/`, build a supervision bundle from real data,
  touch bundle 18403d29, or change `conf/config.yaml` or `conf/graph.yaml`.
- `origin/main` gains a commit touching a file in **File structure**, or an open PR does.
- A Census file's sha256 differs from the pre-flight's list. Do not download a replacement.

## Pre-flight (controller, inline, before Task 1)

- [x] **Step 1: Confirm the workspace**

Run: `git status --short --branch`
Expected: `## claude/outcome-panel-sealed-splits-522fa329` and nothing else.

Run: `git log --oneline origin/main..HEAD`
Expected: only this plan's commit (`docs(plans): add plan 4 …`). If "config" or "graph config"
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

Run: `uv run python -c "import numpy, polars, sklearn; print(numpy.__version__, polars.__version__, sklearn.__version__)"`
Expected: `2.3.4 1.35.1 1.9.1`. These are the locked versions for Python 3.12, and Task 9's
expected hash was computed with them. If they differ, stop and ask.

- [x] **Step 3: Run the baseline suite**

Run: `uv run pytest -n auto -q`
Expected: `1248 passed, 1 skipped`. Each later full-suite count is this baseline plus the tests the
plan has added by then.

- [x] **Step 4: Check the cached Census files**

Run: `shasum -a 256 ~/Downloads/Data/2-6\ digit_2022_Codes.xlsx ~/Downloads/Data/2022_NAICS_Descriptions.xlsx ~/Downloads/Data/2022_NAICS_Index_File.xlsx ~/Downloads/Data/2022_NAICS_Cross_References.xlsx`
Expected:

```text
be12ba41002803359f49181c9bf33a03fbd08578f4f4a4c0bbad7aadaaea0316  …/2-6 digit_2022_Codes.xlsx
6222c4d87dcf984970e0ff8a49862ed54b546b089d03956be3f285900cd3d66c  …/2022_NAICS_Descriptions.xlsx
6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63  …/2022_NAICS_Index_File.xlsx
3c50c3bfa9d76862aea471cc8831fd726f0b978619d833e48c54a749d9622144  …/2022_NAICS_Cross_References.xlsx
```

The code enforces only the index file's hash (`DownloadConfig.index_sha256`, Task 5): its row
positions are the role table's `entry_id`s. The other three hashes go into Task 9's finding. If
any hash differs, stop and ask. Never download replacements.

- [x] **Step 5: Check the pinned descriptions file, read-only**

Run: `shasum -a 256 /Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet`
Expected: `5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d`. This is the file
bundle 18403d29 pins. Task 9 only reads it, to compare the rebuilt examples channel against it.
If the hash differs, stop and ask.

- [x] **Step 6: Route the tasks**

Under executing-plans, run every task inline, in order.

Under subagent-driven-development:

- Tasks 1–8 and 10: each gets a fresh implementer and a task-reviewer. Give each implementer
  its task, **Global Constraints** and **Workspace**.
- Task 9: run it inline in the controller session. It draws the real splits once, checks them
  against **Expected real-data results**, and applies the stop-and-ask conditions.

### Task 1: The leakage checker

Req 3 forbids any test query that appears, exactly or as a near-duplicate, in training text.
This task builds the matcher; Task 2 applies it to the index entries. It is pure functions over
strings and one polars frame, with no I/O.

**Files:**

- Create: `src/naics_embedder/panels/__init__.py`
- Create: `src/naics_embedder/panels/leakage.py`
- Test: `tests/unit/test_outcome_leakage.py`

**Interfaces:**

- Consumes: nothing from earlier tasks.
- Produces (`naics_embedder.panels.leakage`):
  - Constants:
    - `NEAR_DUPLICATE_MIN_JACCARD = Fraction(9, 10)`
    - `EXAMPLES_SEPARATOR = '; '`, the examples channel's join string
    - `ACTIVITY_SEPARATOR = '--'`
    - `TEXT_COLUMNS = ('title', 'description', 'examples', 'excluded')`
  - `normalize_text(text: str) -> str`
  - `text_segments(text: Optional[str]) -> List[str]`: normalized sentences, plus each
    cross-reference sentence's activity phrase.
  - `training_text_segments(descriptions: pl.DataFrame, extra_texts: Iterable[str] = ()) -> List[str]`:
    sorted, de-duplicated segments of every code's title, description, examples channel and
    exclusion text, plus `extra_texts`.
  - `LeakageMatches(exact: np.ndarray, near_duplicate: np.ndarray)`: a frozen dataclass whose
    `leaked` property is `exact | near_duplicate`.
  - `find_leakage(queries, corpus, *, min_jaccard=NEAR_DUPLICATE_MIN_JACCARD) -> LeakageMatches`
  - `find_leakage_within(texts, *, min_jaccard=NEAR_DUPLICATE_MIN_JACCARD) -> LeakageMatches`:
    each text against every other text of the same list.
  - Both raise `ValueError` for a query with no letters or digits, or a threshold outside
    (0, 1].

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_outcome_leakage.py` with exactly this content:

```python
'''
Leakage matching between held-out queries and training text (Req 3, "Leakage").

Expected values are worked out by hand from the matching rules, never from production code.
'''

from fractions import Fraction

import polars as pl
import pytest

from naics_embedder.panels.leakage import (
    find_leakage,
    find_leakage_within,
    normalize_text,
    text_segments,
    training_text_segments,
)

pytestmark = pytest.mark.unit

def test_normalize_text_keeps_ascii_letters_and_digits():
    assert normalize_text('Soybean farming, field & seed production') == (
        'soybean farming field seed production'
    )
    assert normalize_text('Nurse practitioners’ offices (e.g., centers)') == (
        'nurse practitioners offices e g centers'
    )
    assert normalize_text('  T-shirts  ') == 't shirts'

def test_text_segments_split_sentences_and_keep_activity_phrases():
    text = (
        'Growing soybeans--are classified in Industry 111110, Soybean Farming; '
        'Growing corn--are classified in Industry 111150. See also.'
    )

    assert text_segments(text) == [
        'growing soybeans are classified in industry 111110 soybean farming',
        'growing soybeans',
        'growing corn are classified in industry 111150',
        'growing corn',
        'see also',
    ]
    assert text_segments(None) == []
    assert text_segments('') == []

def test_training_text_segments_cover_every_channel_and_extra_texts():
    descriptions = pl.DataFrame(
        {
            'title': ['Soybean Farming'],
            'description': ['This industry grows soybeans. It also sells seed.'],
            'examples': ['Soybean farming, field; Soybean seed production'],
            'excluded': [None],
        },
        schema={name: pl.Utf8
                for name in ('title', 'description', 'examples', 'excluded')},
    )

    assert training_text_segments(descriptions, extra_texts=['Dry pea farming']) == [
        'dry pea farming',
        'it also sells seed',
        'soybean farming',
        'soybean farming field',
        'soybean seed production',
        'this industry grows soybeans',
    ]

def test_training_text_segments_require_the_text_columns():
    with pytest.raises(ValueError, match='text columns'):
        training_text_segments(pl.DataFrame({'title': ['Soybean Farming']}))

def test_exact_matches_need_whole_words():
    matches = find_leakage(
        ['art supplies', 'Card shops', 'party'],
        ['Greeting card shops', 'party supplies stores'],
    )

    # 'art supplies' sits inside 'party supplies' but not on a word boundary
    assert matches.exact.tolist() == [False, True, True]
    assert matches.near_duplicate.tolist() == [False, False, False]

def test_reordered_words_are_near_duplicates_but_not_exact():
    matches = find_leakage(['Card shops, greeting'], ['greeting card shops'])

    assert matches.exact.tolist() == [False]
    assert matches.near_duplicate.tolist() == [True]

def test_near_duplicate_threshold_is_inclusive():
    # 'abcdefghi' has 9 distinct trigrams; ' z ' adds 1 (Jaccard 9/10), 'yz' adds 2 (9/11)
    matches = find_leakage(['abcdefghi z', 'abcdefghi yz'], ['abcdefghi'])

    assert matches.exact.tolist() == [False, False]
    assert matches.near_duplicate.tolist() == [True, False]
    assert matches.leaked.tolist() == [True, False]

def test_within_a_list_self_matches_are_skipped_but_duplicates_flag_each_other():
    matches = find_leakage_within(
        ['Soybean farming', 'soybean  farming!', 'Dry pea farming', 'pea farming']
    )

    # 'pea farming' occurs inside 'dry pea farming'; the two soybean texts normalize equal
    assert matches.exact.tolist() == [True, True, False, True]
    # 'dry pea farming' and 'pea farming' share 10 of 13 trigrams: below 9/10
    assert matches.near_duplicate.tolist() == [True, True, False, False]

def test_queries_without_letters_or_digits_are_rejected():
    with pytest.raises(ValueError, match='no letters or digits'):
        find_leakage(['--'], ['soybean farming'])

@pytest.mark.parametrize('threshold', [Fraction(0), Fraction(3, 2)])
def test_threshold_must_lie_in_the_unit_interval(threshold):
    with pytest.raises(ValueError, match='min_jaccard'):
        find_leakage(['soybean farming'], ['soybean farming'], min_jaccard=threshold)

def test_empty_inputs_flag_nothing():
    assert find_leakage([], ['soybean farming']).leaked.tolist() == []
    assert find_leakage(['soybean farming'], []).leaked.tolist() == [False]
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_outcome_leakage.py -q`
Expected: a collection error, `ModuleNotFoundError: No module named 'naics_embedder.panels'`.

- [x] **Step 3: Write the implementation**

Create `src/naics_embedder/panels/__init__.py` with exactly this content:

```python
'''
Sealed evaluation panels (roadmap Stage 2 onward).

The outcome panel scores text-to-code decoding on held-out Census index entries: ``index_roles``
gives every entry exactly one role, ``leakage`` keeps held-out queries out of training text,
``decoding`` scores an encoder over the six-digit candidates, and ``outcome`` exposes the splits
behind ``selection_log``, which records every read and the one logged opening of the test split.
'''
```

Create `src/naics_embedder/panels/leakage.py` with exactly this content:

```python
'''
Leakage between held-out index-entry queries and training text (Req 3, "Leakage").

Texts are compared after ``normalize_text``: lowercase ASCII letters and digits, with every other
run of characters collapsed to one space. A query leaks into a training segment in two ways:

- exact: the query occurs in the segment as whole words (equality included);
- near-duplicate: the character-trigram Jaccard similarity of query and segment reaches the
  threshold, 9/10 by default. Trigrams are taken inside word boundaries, as scikit-learn's
  ``char_wb`` analyzer builds them, so reordered words score as the same text.

Jaccard is compared in integers (``denominator * shared >= numerator * union``), so no pair sits
on a floating-point boundary. The check compares whole queries with whole segments, so a query
whose words appear reordered inside a longer segment is not caught.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

NEAR_DUPLICATE_MIN_JACCARD = Fraction(9, 10)
EXAMPLES_SEPARATOR = '; '
ACTIVITY_SEPARATOR = '--'
TEXT_COLUMNS = ('title', 'description', 'examples', 'excluded')

_NON_ALNUM = re.compile(r'[^a-z0-9]+')
_SENTENCE_BREAK = re.compile(r'(?<=[.;])\s+')
_CHUNK_ROWS = 256

# -------------------------------------------------------------------------------------------------
# Normalization and segmentation
# -------------------------------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    '''Lowercase ASCII letters and digits, every other run of characters one space.'''

    return _NON_ALNUM.sub(' ', text.lower()).strip()

def text_segments(text: Optional[str]) -> List[str]:
    '''
    Normalized sentences of a description or exclusion text.

    A cross-reference sentence ("Growing soybeans--are classified in Industry 111110") also
    yields its activity phrase, the part before ``--``, which later stages train on as a query.
    '''

    if not text:
        return []
    pieces: List[str] = []
    for sentence in _SENTENCE_BREAK.split(text):
        pieces.append(sentence)
        if ACTIVITY_SEPARATOR in sentence:
            pieces.append(sentence.split(ACTIVITY_SEPARATOR, 1)[0])
    return [segment for segment in map(normalize_text, pieces) if segment]

def training_text_segments(
    descriptions: pl.DataFrame,
    extra_texts: Iterable[str] = (),
) -> List[str]:
    '''
    Sorted, de-duplicated normalized segments of every code's training text, plus extra texts.

    Titles are one segment each, descriptions and exclusion texts are split into sentences, and
    the examples channel is split into its ``'; '``-joined entries.
    '''

    missing = [name for name in TEXT_COLUMNS if name not in descriptions.columns]
    if missing:
        raise ValueError(f'descriptions lack text columns: {missing}')
    segments = set()
    for title, description, examples, excluded in descriptions.select(TEXT_COLUMNS).iter_rows():
        title_text = normalize_text(title or '')
        if title_text:
            segments.add(title_text)
        segments.update(text_segments(description))
        segments.update(text_segments(excluded))
        segments.update(
            text for text in map(normalize_text, (examples or '').split(EXAMPLES_SEPARATOR)) if text
        )
    segments.update(text for text in map(normalize_text, extra_texts) if text)
    return sorted(segments)

# -------------------------------------------------------------------------------------------------
# Matching
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LeakageMatches:
    '''Per-query leakage flags, one boolean per query in input order.'''

    exact: np.ndarray
    near_duplicate: np.ndarray

    @property
    def leaked(self) -> np.ndarray:
        return self.exact | self.near_duplicate

def _check_threshold(min_jaccard: Fraction) -> None:
    if not 0 < min_jaccard <= 1:
        raise ValueError(f'min_jaccard must lie in (0, 1], got {min_jaccard}')

def _normalized_queries(queries: Sequence[str]) -> List[str]:
    normalized = [normalize_text(query) for query in queries]
    empty = [query for query, text in zip(queries, normalized) if not text]
    if empty:
        raise ValueError(f'{len(empty):,} queries have no letters or digits, e.g. {empty[0]!r}')
    return normalized

def _containing_row_counts(patterns: Sequence[str], texts: Sequence[str]) -> Dict[str, int]:
    '''For each pattern found, the number of texts that contain it as whole words.'''

    if not patterns or not texts:
        return {}
    padded = [f' {pattern} ' for pattern in patterns]
    # yapf: disable
    found = (
        pl.DataFrame({'text': [f' {text} ' for text in texts]})
        .select(match=pl.col('text').str.extract_many(padded, overlapping=True).list.unique())
        .explode('match')
        .drop_nulls()
        .group_by('match')
        .len()
    )
    # yapf: enable
    return {match[1:-1]: count for match, count in found.iter_rows()}

def _near_duplicate_flags(
    queries: Sequence[str],
    corpus: Sequence[str],
    min_jaccard: Fraction,
    *,
    same_list: bool,
) -> np.ndarray:
    flags = np.zeros(len(queries), dtype=bool)
    if not queries or not corpus:
        return flags
    vectorizer = CountVectorizer(
        analyzer='char_wb', ngram_range=(3, 3), binary=True, lowercase=False, dtype=np.int32
    ).fit(list(queries) + list(corpus))
    query_grams = vectorizer.transform(queries)
    corpus_grams = vectorizer.transform(corpus)
    query_sizes = np.asarray(query_grams.sum(axis=1)).ravel()
    corpus_sizes = np.asarray(corpus_grams.sum(axis=1)).ravel()
    numerator, denominator = min_jaccard.numerator, min_jaccard.denominator
    for start in range(0, len(queries), _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, len(queries))
        shared = (query_grams[start:stop] @ corpus_grams.T).toarray()
        union = query_sizes[start:stop, None] + corpus_sizes[None, :] - shared
        similar = denominator * shared >= numerator * union
        if same_list:
            rows = np.arange(stop - start)
            similar[rows, rows + start] = False
        flags[start:stop] = similar.any(axis=1)
    return flags

def find_leakage(
    queries: Sequence[str],
    corpus: Sequence[str],
    *,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> LeakageMatches:
    '''
    Flag each query that occurs in, or near-duplicates, any corpus text.

    Raises:
        ValueError: If a query has no letters or digits, or the threshold is outside (0, 1].
    '''

    _check_threshold(min_jaccard)
    normalized = _normalized_queries(queries)
    targets = [text for text in map(normalize_text, corpus) if text]
    counts = _containing_row_counts(normalized, targets)
    exact = np.array([counts.get(text, 0) > 0 for text in normalized], dtype=bool)
    near = _near_duplicate_flags(normalized, targets, min_jaccard, same_list=False)
    return LeakageMatches(exact=exact, near_duplicate=near)

def find_leakage_within(
    texts: Sequence[str],
    *,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> LeakageMatches:
    '''
    Flag each text that occurs in, or near-duplicates, another text of the same list.

    A text is never matched against itself, but two identical texts flag each other.
    '''

    _check_threshold(min_jaccard)
    normalized = _normalized_queries(texts)
    counts = _containing_row_counts(normalized, normalized)
    exact = np.array([counts.get(text, 0) > 1 for text in normalized], dtype=bool)
    near = _near_duplicate_flags(normalized, normalized, min_jaccard, same_list=True)
    return LeakageMatches(exact=exact, near_duplicate=near)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_outcome_leakage.py -q`
Expected: `12 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1260 passed, 1 skipped`.

- [x] **Step 5: Lint and format**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels tests/unit/test_outcome_leakage.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` The code above is
already yapf-clean. On a failure, run the same command without `--check`, re-run Step 4, and
record the change as a deviation.

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/__init__.py src/naics_embedder/panels/leakage.py tests/unit/test_outcome_leakage.py
git commit -m "feat(panels): add exact and near-duplicate leakage matching"
```

### Task 2: Index-entry roles

This task gives every index entry exactly one role, per code and stratified (D4). It also adds
the role table's validator, which the supervision layer reuses in Task 6, so the validator
lives in `supervision/artifacts.py` beside the other artifact validators.

**Files:**

- Modify: `src/naics_embedder/supervision/schema.py` (a schema version, and an `IndexRole`
  section before the manifest models)
- Modify: `src/naics_embedder/supervision/artifacts.py` (imports, constants, one validator)
- Create: `src/naics_embedder/panels/index_roles.py`
- Test: `tests/unit/test_index_roles.py`

**Interfaces:**

- Consumes (Task 1): `EXAMPLES_SEPARATOR`, `NEAR_DUPLICATE_MIN_JACCARD`, `find_leakage`,
  `find_leakage_within` and `training_text_segments`.
- Produces in `naics_embedder.supervision.schema`:
  - `INDEX_ROLES_SCHEMA_VERSION = 'index-roles-v1'`
  - `class IndexRole(str, Enum)`: `EXAMPLES = 'examples'`, `TRAINING = 'training'`,
    `VALIDATION = 'validation'` and `TEST = 'test'`.
- Produces in `naics_embedder.supervision.artifacts`:
  - `INDEX_ROLES_ARTIFACT = 'index_roles'`
  - `INDEX_ROLE_COLUMNS = ('entry_id', 'code', 'text', 'role')`
  - `validate_index_role_table(roles: pl.DataFrame, six_digit_codes: Collection[str], *, min_examples_per_code: int = 1) -> None`.
    It raises `ValueError` at the first failing check, in this order. Task 6's loader re-raises
    it as a `ValueError` prefixed with the artifact and the bundle id.
    - `index roles lack required columns`
    - `index roles contain null values`
    - `an index entry holds more than one role`
    - `index roles contain unknown roles`
    - `index roles name codes outside the six-digit codebook`
    - `index roles contain an empty entry text`
    - `… codes have fewer than K examples-role entries`, where a code needs
      `min(min_examples_per_code, its entries)`
- Produces in `naics_embedder.panels.index_roles`:
  - Constants:
    - `ROLE_ORDER = (EXAMPLES, TRAINING, VALIDATION, TEST)`
    - `ROLE_TABLE_SCHEMA = {'entry_id': pl.Int64, 'code': pl.Utf8, 'role': pl.Utf8}`
  - `RoleFractions(examples, training, validation, test)`: a frozen dataclass of `Fraction`s,
    non-negative and summing to 1, else `ValueError` (`… sum to 1 …`).
    - `RoleFractions.from_mapping(mapping: Mapping[str, float])` converts through `str`.
    - `RoleFractions.of(role: IndexRole) -> Fraction`
  - `allocate_role_counts(n_entries: int, n_eligible: int, fractions: RoleFractions, tie_break: Sequence[float], examples_floor: int = 1) -> Dict[IndexRole, int]`
  - `EligibilityReport(eligible: np.ndarray, counts: Dict[str, int])`. The counts keys are
    `entries`, `exact_static`, `near_duplicate_static`, `exact_entry`, `near_duplicate_entry`,
    `withheld_exact`, `withheld_near_duplicate`, `withheld` and `eligible`.
  - `held_out_eligibility(entries: pl.DataFrame, static_segments: Sequence[str], min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD) -> EligibilityReport`
  - `assign_index_roles(entries: pl.DataFrame, eligible: np.ndarray, fractions: RoleFractions, seed: int, examples_floor: int = 1) -> pl.DataFrame`:
    `entry_id`, `code` and `role`, sorted by `entry_id`.
  - The frozen table:
    - `role_table_fingerprint(roles: pl.DataFrame) -> str`
    - `write_role_table(roles: pl.DataFrame, path: Path) -> str`, which returns the fingerprint
    - `read_role_table(path: Path) -> pl.DataFrame`
  - `attach_role_text(roles: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame`: the columns
    are `INDEX_ROLE_COLUMNS`. It raises `ValueError` when the two list different entries or
    give an entry a different code.
  - `examples_channel_by_code(role_rows: pl.DataFrame) -> pl.DataFrame`, with columns `code` and
    `examples`.
  - `verify_examples_channel(descriptions: pl.DataFrame, role_rows: pl.DataFrame) -> None`. It
    raises `ValueError` (`… codes have an examples channel other than their examples-role
    entries …`).
  - `verify_role_leakage(descriptions: pl.DataFrame, role_rows: pl.DataFrame, min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD) -> Dict[str, Dict[str, int]]`.
    It returns `{'validation': {'exact': 0, 'near_duplicate': 0}, 'test': {…}}` and raises
    `ValueError` (`held-out queries match training text: …`) on any match.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_index_roles.py` with exactly this content:

```python
'''
Index-entry roles (Req 3; roadmap D4): quotas, eligibility, assignment and the frozen table.

Expected counts are worked out by hand from the allocation rule, never from production code.
'''

from fractions import Fraction

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels.index_roles import (
    RoleFractions,
    allocate_role_counts,
    assign_index_roles,
    attach_role_text,
    held_out_eligibility,
    read_role_table,
    role_table_fingerprint,
    verify_examples_channel,
    verify_role_leakage,
    write_role_table,
)
from naics_embedder.supervision.artifacts import sha256_file, validate_index_role_table
from naics_embedder.supervision.schema import IndexRole

pytestmark = pytest.mark.unit

E, TR, V, TE = IndexRole.EXAMPLES, IndexRole.TRAINING, IndexRole.VALIDATION, IndexRole.TEST
STAGE2 = RoleFractions.from_mapping(
    {
        'examples': 0.30,
        'training': 0.35,
        'validation': 0.20,
        'test': 0.15
    }
)
QUARTERS = RoleFractions.from_mapping(
    {
        'examples': 0.25,
        'training': 0.25,
        'validation': 0.25,
        'test': 0.25
    }
)
NO_TIES = (0.1, 0.2, 0.3, 0.4)

# -------------------------------------------------------------------------------------------------
# Fractions and quotas
# -------------------------------------------------------------------------------------------------

def test_fractions_are_exact_decimals():
    assert STAGE2.of(E) == Fraction(3, 10)
    assert STAGE2.of(TR) == Fraction(7, 20)
    assert STAGE2.of(V) == Fraction(1, 5)
    assert STAGE2.of(TE) == Fraction(3, 20)

@pytest.mark.parametrize(
    'mapping',
    [
        {
            'examples': 0.30,
            'training': 0.35,
            'validation': 0.20,
            'test': 0.14
        },
        {
            'examples': 0.60,
            'training': 0.45,
            'validation': 0.10,
            'test': -0.15
        },
    ],
)
def test_fractions_must_be_non_negative_and_sum_to_one(mapping):
    with pytest.raises(ValueError, match='sum to 1'):
        RoleFractions.from_mapping(mapping)

@pytest.mark.parametrize(
    ('n_entries', 'n_eligible', 'expected'),
    [
        # 0.30/0.35/0.20/0.15 of 1: training wins the remainder, then the floor moves it
        (1, 1, {
            E: 1,
            TR: 0,
            V: 0,
            TE: 0
        }),
        # of 2: remainders 0.7 (training) and 0.6 (examples) win
        (2, 2, {
            E: 1,
            TR: 1,
            V: 0,
            TE: 0
        }),
        # of 3: training floors to 1; examples (0.9) and validation (0.6) win the remainders
        (3, 3, {
            E: 1,
            TR: 1,
            V: 1,
            TE: 0
        }),
        # of 4: examples and training floor to 1; validation (0.8) and test (0.6) win
        (4, 4, {
            E: 1,
            TR: 1,
            V: 1,
            TE: 1
        }),
        (20, 20, {
            E: 6,
            TR: 7,
            V: 4,
            TE: 3
        }),
        # 7 held out but 5 eligible: validation (larger) gives one, then test (tie) gives one
        (20, 5, {
            E: 6,
            TR: 9,
            V: 3,
            TE: 2
        }),
        # nothing eligible: every held-out quota moves to training
        (4, 0, {
            E: 1,
            TR: 3,
            V: 0,
            TE: 0
        }),
    ],
)
def test_quotas_follow_largest_remainder_floor_and_eligibility(n_entries, n_eligible, expected):
    assert allocate_role_counts(n_entries, n_eligible, STAGE2, NO_TIES) == expected

@pytest.mark.parametrize(
    ('tie_break', 'winner'),
    [((0.5, 0.4, 0.3, 0.1), TE), ((0.5, 0.4, 0.1, 0.3), V)],
)
def test_remainder_ties_go_to_the_smallest_draw(tie_break, winner):
    counts = allocate_role_counts(1, 1, QUARTERS, tie_break, examples_floor=0)

    assert counts == {role: int(role == winner) for role in (E, TR, V, TE)}

def test_floor_takes_from_test_before_validation_when_training_is_empty():
    # Quarters of 2 with validation and test drawing lowest: V1 TE1, then the floor needs one
    counts = allocate_role_counts(2, 2, QUARTERS, (0.9, 0.8, 0.1, 0.2))

    assert counts == {E: 1, TR: 0, V: 1, TE: 0}

def test_quotas_reject_inconsistent_inputs():
    with pytest.raises(ValueError, match='eligible count'):
        allocate_role_counts(2, 3, STAGE2, NO_TIES)
    with pytest.raises(ValueError, match='one draw per role'):
        allocate_role_counts(2, 2, STAGE2, (0.1, 0.2))

# -------------------------------------------------------------------------------------------------
# Eligibility
# -------------------------------------------------------------------------------------------------

def _entries(rows):
    return pl.DataFrame(
        rows,
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8
        },
        orient='row',
    )

def test_eligibility_withholds_exact_and_near_duplicate_matches():
    entries = _entries(
        [
            (0, '111110', 'Soybean farming'),  # equals a static title
            (1, '111110', 'Soybeans, organic'),
            (2, '111120', 'Oilseed farming'),  # reordered twin of entry 3
            (3, '111120', 'Farming, oilseed'),
            (4, '111120', 'Growing corn for grain'),  # contains a static phrase: fine
            (5, '111130', 'Rye farming'),  # occurs inside entry 6
            (6, '111130', 'Rye farming, organic'),
        ]
    )

    report = held_out_eligibility(entries, ['soybean farming', 'growing corn'])

    assert report.eligible.tolist() == [False, True, False, False, True, False, True]
    assert report.counts == {
        'entries': 7,
        'exact_static': 1,
        'near_duplicate_static': 1,
        'exact_entry': 1,
        'near_duplicate_entry': 2,
        'withheld_exact': 2,  # entries 0 and 5
        'withheld_near_duplicate': 2,  # entries 2 and 3
        'withheld': 4,
        'eligible': 3,
    }

# -------------------------------------------------------------------------------------------------
# Assignment
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def assignment_entries():
    rows = [(i, '111110', f'soybean entry {i}') for i in range(20)]
    rows.append((20, '111120', 'oilseed entry'))
    rows += [(i, '111130', f'rye entry {i}') for i in range(21, 25)]
    return _entries(rows)

def _eligible(entries):
    return entries.get_column('code').ne('111130').to_numpy()

def _counts(roles, code):
    grouped = roles.filter(pl.col('code') == code).group_by('role').len()
    return {IndexRole(role): count for role, count in grouped.iter_rows()}

def test_every_entry_gets_exactly_one_role_in_per_code_quotas(assignment_entries):
    roles = assign_index_roles(assignment_entries, _eligible(assignment_entries), STAGE2, seed=7)

    assert roles.columns == ['entry_id', 'code', 'role']
    assert roles.get_column('entry_id').to_list() == list(range(25))
    assert _counts(roles, '111110') == {E: 6, TR: 7, V: 4, TE: 3}
    assert _counts(roles, '111120') == {E: 1}
    assert _counts(roles, '111130') == {E: 1, TR: 3}

def test_ineligible_entries_are_never_held_out(assignment_entries):
    roles = assign_index_roles(assignment_entries, _eligible(assignment_entries), STAGE2, seed=7)

    held_out = roles.filter(pl.col('role').is_in([V.value, TE.value]))
    assert '111130' not in held_out.get_column('code').to_list()

def test_assignment_is_deterministic_per_seed(assignment_entries):
    eligible = _eligible(assignment_entries)
    first = assign_index_roles(assignment_entries, eligible, STAGE2, seed=7)
    again = assign_index_roles(assignment_entries, eligible, STAGE2, seed=7)
    other = assign_index_roles(assignment_entries, eligible, STAGE2, seed=8)

    assert first.equals(again)
    assert not first.equals(other)

def test_assignment_needs_one_flag_per_entry(assignment_entries):
    with pytest.raises(ValueError, match='flags for 25 entries'):
        assign_index_roles(assignment_entries, np.ones(3, dtype=bool), STAGE2, seed=7)

# -------------------------------------------------------------------------------------------------
# The frozen table
# -------------------------------------------------------------------------------------------------

def test_role_table_round_trips_with_string_codes(tmp_path):
    roles = pl.DataFrame(
        {
            'entry_id': [3, 1],
            'code': ['111110', '111120'],
            'role': ['test', 'examples']
        }
    )
    path = tmp_path / 'conf' / 'index_roles.csv'

    digest = write_role_table(roles, path)

    assert digest == sha256_file(path) == role_table_fingerprint(roles)
    assert path.read_text().splitlines() == [
        'entry_id,code,role',
        '1,111120,examples',
        '3,111110,test',
    ]
    assert read_role_table(path).equals(roles.sort('entry_id'))

def test_attach_role_text_joins_on_entry_id():
    entries = _entries([(0, '111110', 'Soybean farming'), (1, '111120', 'Oilseed farming')])
    roles = pl.DataFrame(
        {
            'entry_id': [1, 0],
            'code': ['111120', '111110'],
            'role': ['test', 'examples']
        }
    )

    joined = attach_role_text(roles, entries)

    assert joined.rows() == [
        (0, '111110', 'Soybean farming', 'examples'),
        (1, '111120', 'Oilseed farming', 'test'),
    ]

@pytest.mark.parametrize(
    ('roles', 'message'),
    [
        ({
            'entry_id': [0],
            'code': ['111110'],
            'role': ['examples']
        }, 'different entries'),
        (
            {
                'entry_id': [0, 1],
                'code': ['111110', '111110'],
                'role': ['examples', 'test']
            }, 'different code'
        ),
    ],
)
def test_attach_role_text_rejects_a_table_for_other_entries(roles, message):
    entries = _entries([(0, '111110', 'Soybean farming'), (1, '111120', 'Oilseed farming')])

    with pytest.raises(ValueError, match=message):
        attach_role_text(pl.DataFrame(roles), entries)

# -------------------------------------------------------------------------------------------------
# Table invariants (shared with the bundle loader)
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def role_rows():
    return pl.DataFrame(
        {
            'entry_id': [0, 1, 2, 3],
            'code': ['111110', '111110', '111110', '111120'],
            'text': ['Soybeans, organic', 'Soybean seed', 'Edamame', 'Oilseed farming'],
            'role': ['examples', 'examples', 'validation', 'examples'],
        }
    )

def test_a_valid_table_passes(role_rows):
    validate_index_role_table(role_rows, ['111110', '111120', '112130'])

@pytest.mark.parametrize(
    ('change', 'message'),
    [
        (lambda f: f.with_columns(entry_id=pl.lit(0)), 'more than one role'),
        (lambda f: f.with_columns(role=pl.lit('holdout')), 'unknown roles'),
        (lambda f: f.with_columns(code=pl.lit('11111')), 'outside the six-digit codebook'),
        (lambda f: f.with_columns(role=pl.lit(None, pl.Utf8)), 'null values'),
        (lambda f: f.with_columns(text=pl.lit('  ')), 'empty entry text'),
        (lambda f: f.with_columns(role=pl.lit('training')), 'fewer than 1 examples-role'),
        (lambda f: f.drop('text'), 'lack required columns'),
    ],
)
def test_table_violations_fail_closed(role_rows, change, message):
    with pytest.raises(ValueError, match=message):
        validate_index_role_table(change(role_rows), ['111110', '111120', '112130'])

# -------------------------------------------------------------------------------------------------
# Consistency with descriptions
# -------------------------------------------------------------------------------------------------

def _descriptions(examples_by_code, title='Soybean Farming'):
    codes = sorted(examples_by_code)
    return pl.DataFrame(
        {
            'code': codes,
            'title': [title] * len(codes),
            'description': ['This industry comprises farms.'] * len(codes),
            'examples': [examples_by_code[code] for code in codes],
            'excluded': [None] * len(codes),
        },
        schema={
            'code': pl.Utf8,
            'title': pl.Utf8,
            'description': pl.Utf8,
            'examples': pl.Utf8,
            'excluded': pl.Utf8,
        },
    )

def test_examples_channel_must_hold_exactly_the_examples_role_entries(role_rows):
    good = _descriptions({'111110': 'Soybeans, organic; Soybean seed', '111120': 'Oilseed farming'})
    verify_examples_channel(good, role_rows)

    stale = _descriptions(
        {
            '111110': 'Soybeans, organic; Soybean seed; Edamame',
            '111120': 'Oilseed farming'
        }
    )
    with pytest.raises(ValueError, match='examples channel other than'):
        verify_examples_channel(stale, role_rows)

def test_role_leakage_reports_zero_for_clean_splits(role_rows):
    descriptions = _descriptions(
        {
            '111110': 'Soybeans, organic; Soybean seed',
            '111120': 'Oilseed farming'
        }
    )

    report = verify_role_leakage(descriptions, role_rows)

    assert report == {
        'validation': {
            'exact': 0,
            'near_duplicate': 0
        },
        'test': {
            'exact': 0,
            'near_duplicate': 0
        },
    }

def test_role_leakage_fails_on_a_held_out_query_in_training_text(role_rows):
    leaky = role_rows.with_columns(
        text=pl.when(pl.col('entry_id') == 2).then(pl.lit('Soybean farming')).otherwise('text')
    )
    descriptions = _descriptions(
        {
            '111110': 'Soybeans, organic; Soybean seed',
            '111120': 'Oilseed farming'
        }
    )

    with pytest.raises(ValueError, match='held-out queries match training text'):
        verify_role_leakage(descriptions, leaky)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_index_roles.py -q`
Expected: a collection error,
`ModuleNotFoundError: No module named 'naics_embedder.panels.index_roles'`.

- [x] **Step 3: Add the role enum and the table validator**

Modify `src/naics_embedder/supervision/schema.py` with these 2 edits, in order. Each replaced text
occurs exactly once in the file.

**`src/naics_embedder/supervision/schema.py`, edit 1 of 2.** Replace:

```python
TRAINING_PAIRS_SCHEMA_VERSION = 'training-pairs-v1'
DIFFICULTY_THRESHOLDS_SCHEMA_VERSION = 'difficulty-thresholds-v1'

# -------------------------------------------------------------------------------------------------
```

with:

```python
TRAINING_PAIRS_SCHEMA_VERSION = 'training-pairs-v1'
DIFFICULTY_THRESHOLDS_SCHEMA_VERSION = 'difficulty-thresholds-v1'
INDEX_ROLES_SCHEMA_VERSION = 'index-roles-v1'

# -------------------------------------------------------------------------------------------------
```

**`src/naics_embedder/supervision/schema.py`, edit 2 of 2.** Replace:

```python
    DIFFICULTY = 4
    BACKFILL = 5

# -------------------------------------------------------------------------------------------------
```

with:

```python
    DIFFICULTY = 4
    BACKFILL = 5

# -------------------------------------------------------------------------------------------------
# Index-entry roles (outcome panel)
# -------------------------------------------------------------------------------------------------

class IndexRole(str, Enum):
    '''The one role an index entry holds: examples-channel text, or a query in one split.'''

    EXAMPLES = 'examples'
    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'test'

# -------------------------------------------------------------------------------------------------
```

Modify `src/naics_embedder/supervision/artifacts.py` with these 3 edits, in order. Each replaced
text occurs exactly once in the file.

**`src/naics_embedder/supervision/artifacts.py`, edit 1 of 3.** Replace:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
```

with:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Iterable, List, Optional, Tuple

import numpy as np
```

**`src/naics_embedder/supervision/artifacts.py`, edit 2 of 3.** Replace:

```python
    CONTRACT_VERSION,
    ArtifactFile,
    SemanticTarget,
    SupervisionManifest,
```

with:

```python
    CONTRACT_VERSION,
    ArtifactFile,
    IndexRole,
    SemanticTarget,
    SupervisionManifest,
```

**`src/naics_embedder/supervision/artifacts.py`, edit 3 of 3.** Replace:

```python
            'pair facts exclusion derivation is inconsistent: is_explicit_exclusion must equal '
            'code_i_excludes_code_j OR code_j_excludes_code_i'
        )
```

with:

```python
            'pair facts exclusion derivation is inconsistent: is_explicit_exclusion must equal '
            'code_i_excludes_code_j OR code_j_excludes_code_i'
        )

INDEX_ROLES_ARTIFACT = 'index_roles'
INDEX_ROLE_COLUMNS = ('entry_id', 'code', 'text', 'role')

def validate_index_role_table(
    roles: pl.DataFrame,
    six_digit_codes: Collection[str],
    *,
    min_examples_per_code: int = 1,
) -> None:
    '''
    Fail closed unless every index entry holds exactly one known role for a six-digit code.

    Also requires non-empty entry text, and the examples-channel floor: every code keeps at least
    ``min_examples_per_code`` examples-role entries, or all its entries if it has fewer.
    '''

    missing = [name for name in INDEX_ROLE_COLUMNS if name not in roles.columns]
    if missing:
        raise ValueError(f'index roles lack required columns: {missing}')
    if roles.select(pl.any_horizontal(pl.col(list(INDEX_ROLE_COLUMNS)).is_null()).any()).item():
        raise ValueError('index roles contain null values')
    if roles.get_column('entry_id').is_duplicated().any():
        raise ValueError('an index entry holds more than one role')
    unknown = sorted(
        set(roles.get_column('role').unique().to_list()) - {role.value
                                                            for role in IndexRole}
    )
    if unknown:
        raise ValueError(f'index roles contain unknown roles: {unknown}')
    outside = sorted(set(roles.get_column('code').unique().to_list()) - set(six_digit_codes))
    if outside:
        raise ValueError(f'index roles name codes outside the six-digit codebook: {outside[:5]}')
    if roles.filter(pl.col('text').str.strip_chars().eq('')).height:
        raise ValueError('index roles contain an empty entry text')
    floor = pl.min_horizontal(pl.col('entries'), pl.lit(min_examples_per_code))
    short = roles.group_by('code').agg(
        examples=pl.col('role').eq(IndexRole.EXAMPLES.value).sum(), entries=pl.len()
    ).filter(pl.col('examples') < floor)
    if short.height:
        raise ValueError(
            f'{short.height:,} codes have fewer than {min_examples_per_code} examples-role entries'
        )
```

- [x] **Step 4: Write the role assignment**

Create `src/naics_embedder/panels/index_roles.py` with exactly this content:

```python
'''
Index-entry roles for the outcome panel (Req 3; roadmap D4).

Every entry of the Census NAICS index file holds exactly one role: examples-channel text, or a
training, validation or test query. The assignment is made once, frozen in a committed table
(``conf/data/index_roles.csv``), and consumed by hash afterwards; regeneration is never how the
sealed splits are preserved.

- **Eligibility.** Validation and test queries are drawn only from entries that no training text
  can leak: an entry is withheld from both held-out splits when it matches, exactly or as a
  near-duplicate (``leakage``), any title, description, exclusion text, fallback examples text
  or any other index entry. Checking against every other entry, not only those that end up as
  training text, makes the held-out splits leak-free whatever the assignment.
- **Quotas.** Per code, role counts are largest-remainder quotas of the configured fractions,
  with remainder ties broken by a seeded draw. Every code keeps at least ``examples_floor``
  examples-channel entries, and held-out quotas beyond the code's eligible entries move to
  training, one at a time from the larger of the two held-out quotas (test on a tie).
- **Seeds.** Each code draws from its own generator, seeded by (seed, code), so a code's roles do
  not depend on any other code.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import polars as pl

from naics_embedder.panels.leakage import (
    EXAMPLES_SEPARATOR,
    NEAR_DUPLICATE_MIN_JACCARD,
    find_leakage,
    find_leakage_within,
    training_text_segments,
)
from naics_embedder.supervision.artifacts import INDEX_ROLE_COLUMNS
from naics_embedder.supervision.schema import IndexRole

ROLE_ORDER = (IndexRole.EXAMPLES, IndexRole.TRAINING, IndexRole.VALIDATION, IndexRole.TEST)
ROLE_TABLE_SCHEMA = {'entry_id': pl.Int64, 'code': pl.Utf8, 'role': pl.Utf8}

# -------------------------------------------------------------------------------------------------
# Fractions and per-code quotas
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleFractions:
    '''Exact role fractions; non-negative and summing to one.'''

    examples: Fraction
    training: Fraction
    validation: Fraction
    test: Fraction

    def __post_init__(self) -> None:
        values = [self.of(role) for role in ROLE_ORDER]
        if any(value < 0 for value in values) or sum(values) != 1:
            raise ValueError(
                'role fractions must be non-negative and sum to 1, got '
                f'{[str(value) for value in values]}'
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> 'RoleFractions':
        '''Exact fractions from decimal config values (0.35 becomes 7/20, not a binary float).'''

        return cls(**{role.value: Fraction(str(mapping[role.value])) for role in ROLE_ORDER})

    def of(self, role: IndexRole) -> Fraction:
        return getattr(self, role.value)

def allocate_role_counts(
    n_entries: int,
    n_eligible: int,
    fractions: RoleFractions,
    tie_break: Sequence[float],
    examples_floor: int = 1,
) -> Dict[IndexRole, int]:
    '''
    Role counts for one code with ``n_entries`` entries, ``n_eligible`` of them held-out-eligible.

    Args:
        n_entries: The code's index entries.
        n_eligible: Entries that may become validation or test queries.
        fractions: Target role fractions.
        tie_break: One draw per role, in ``ROLE_ORDER``; the smaller draw wins a remainder tie.
        examples_floor: Minimum examples-role entries (capped at ``n_entries``).

    Returns:
        Counts per role, summing to ``n_entries``.
    '''

    if not 0 <= n_eligible <= n_entries:
        raise ValueError(f'eligible count {n_eligible} must lie in [0, {n_entries}]')
    if len(tie_break) != len(ROLE_ORDER):
        raise ValueError(f'tie_break needs one draw per role, got {len(tie_break)}')

    exact = [fractions.of(role) * n_entries for role in ROLE_ORDER]
    counts = [math.floor(value) for value in exact]
    remainders = [value - count for value, count in zip(exact, counts)]
    order = sorted(range(len(ROLE_ORDER)), key=lambda k: (-remainders[k], tie_break[k]))
    for k in order[:n_entries - sum(counts)]:
        counts[k] += 1
    by_role = dict(zip(ROLE_ORDER, counts))

    while by_role[IndexRole.EXAMPLES] < min(examples_floor, n_entries):
        if by_role[IndexRole.TRAINING]:
            donor = IndexRole.TRAINING
        elif by_role[IndexRole.TEST] >= by_role[IndexRole.VALIDATION]:
            donor = IndexRole.TEST
        else:
            donor = IndexRole.VALIDATION
        by_role[donor] -= 1
        by_role[IndexRole.EXAMPLES] += 1

    while by_role[IndexRole.VALIDATION] + by_role[IndexRole.TEST] > n_eligible:
        if by_role[IndexRole.TEST] >= by_role[IndexRole.VALIDATION]:
            by_role[IndexRole.TEST] -= 1
        else:
            by_role[IndexRole.VALIDATION] -= 1
        by_role[IndexRole.TRAINING] += 1
    return by_role

# -------------------------------------------------------------------------------------------------
# Eligibility and assignment
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EligibilityReport:
    '''
    Which entries may be held out, and why the others are withheld.

    The ``*_static`` and ``*_entry`` counts overlap; ``withheld_exact`` (any exact match) and
    ``withheld_near_duplicate`` (a near-duplicate and no exact match) partition ``withheld``.
    '''

    eligible: np.ndarray
    counts: Dict[str, int]

def held_out_eligibility(
    entries: pl.DataFrame,
    static_segments: Sequence[str],
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> EligibilityReport:
    '''
    Flag the entries that no training text can leak.

    Args:
        entries: Index entries (``entry_id``, ``code``, ``text``).
        static_segments: Every code's title, description, exclusion and fallback examples
            segments (``leakage.training_text_segments`` without index-derived examples).
        min_jaccard: Near-duplicate threshold.
    '''

    texts = entries.get_column('text').to_list()
    static = find_leakage(texts, static_segments, min_jaccard=min_jaccard)
    within = find_leakage_within(texts, min_jaccard=min_jaccard)
    exact = static.exact | within.exact
    near_duplicate_only = (static.near_duplicate | within.near_duplicate) & ~exact
    eligible = ~(exact | near_duplicate_only)
    counts = {
        'entries': len(texts),
        'exact_static': int(static.exact.sum()),
        'near_duplicate_static': int(static.near_duplicate.sum()),
        'exact_entry': int(within.exact.sum()),
        'near_duplicate_entry': int(within.near_duplicate.sum()),
        'withheld_exact': int(exact.sum()),
        'withheld_near_duplicate': int(near_duplicate_only.sum()),
        'withheld': int((~eligible).sum()),
        'eligible': int(eligible.sum()),
    }
    return EligibilityReport(eligible=eligible, counts=counts)

def assign_index_roles(
    entries: pl.DataFrame,
    eligible: np.ndarray,
    fractions: RoleFractions,
    seed: int,
    examples_floor: int = 1,
) -> pl.DataFrame:
    '''
    Give every entry exactly one role, per code and stratified.

    Args:
        entries: Index entries (``entry_id``, ``code``, ...), one row per entry.
        eligible: Held-out eligibility, aligned with ``entries``.
        fractions: Target role fractions.
        seed: Base seed; each code draws from ``np.random.default_rng([seed, int(code)])``.
        examples_floor: Minimum examples-role entries per code.

    Returns:
        ``entry_id``, ``code``, ``role`` sorted by ``entry_id``.
    '''

    if len(eligible) != entries.height:
        raise ValueError(f'eligible has {len(eligible)} flags for {entries.height} entries')
    frame = entries.select('entry_id', 'code').with_columns(
        eligible=pl.Series(np.asarray(eligible, dtype=bool))
    )
    assigned: List[Tuple[int, str, str]] = []
    for (code, ), group in frame.sort('entry_id').group_by('code', maintain_order=True):
        rng = np.random.default_rng([seed, int(code)])
        tie_break = rng.random(len(ROLE_ORDER))
        entry_ids = group.get_column('entry_id').to_numpy()
        flags = group.get_column('eligible').to_numpy()
        counts = allocate_role_counts(
            len(entry_ids), int(flags.sum()), fractions, tie_break, examples_floor
        )
        held_out = rng.permutation(entry_ids[flags])
        n_validation = counts[IndexRole.VALIDATION]
        n_held_out = n_validation + counts[IndexRole.TEST]
        rest = rng.permutation(np.concatenate([held_out[n_held_out:], entry_ids[~flags]]))
        n_examples = counts[IndexRole.EXAMPLES]
        for ids, role in (
            (held_out[:n_validation], IndexRole.VALIDATION),
            (held_out[n_validation:n_held_out], IndexRole.TEST),
            (rest[:n_examples], IndexRole.EXAMPLES),
            (rest[n_examples:], IndexRole.TRAINING),
        ):
            assigned.extend((int(entry_id), str(code), role.value) for entry_id in ids)
    return pl.DataFrame(assigned, schema=ROLE_TABLE_SCHEMA, orient='row').sort('entry_id')

# -------------------------------------------------------------------------------------------------
# The frozen role table
# -------------------------------------------------------------------------------------------------

def _role_table_csv(roles: pl.DataFrame) -> bytes:
    return roles.select(list(ROLE_TABLE_SCHEMA)).sort('entry_id').write_csv().encode('utf-8')

def role_table_fingerprint(roles: pl.DataFrame) -> str:
    '''
    SHA-256 of the role assignment's canonical CSV (``entry_id``, ``code``, ``role``).

    It equals the committed table's file hash, so a split is identified by its assignment
    whichever file it was read from.
    '''

    return hashlib.sha256(_role_table_csv(roles)).hexdigest()

def write_role_table(roles: pl.DataFrame, path: Path) -> str:
    '''Write the canonical CSV and return its sha256 (``role_table_fingerprint``).'''

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_role_table_csv(roles))
    return role_table_fingerprint(roles)

def read_role_table(path: Path) -> pl.DataFrame:
    '''Read the frozen role table, keeping codes as strings.'''

    return pl.read_csv(Path(path), schema=ROLE_TABLE_SCHEMA)

def attach_role_text(roles: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame:
    '''
    Join the frozen role table to the index entries it was built from.

    Returns:
        ``entry_id``, ``code``, ``text``, ``role`` sorted by ``entry_id``.

    Raises:
        ValueError: If the two list different entries, or give an entry different codes.
    '''

    joined = entries.select('entry_id', 'code', 'text').join(
        roles.select('entry_id',
                     pl.col('code').alias('role_code'), 'role'),
        on='entry_id',
        how='full',
        coalesce=True,
    )
    if joined.filter(pl.col('code').is_null() | pl.col('role').is_null()).height:
        raise ValueError('the role table and the index file list different entries')
    if joined.filter(pl.col('code') != pl.col('role_code')).height:
        raise ValueError('the role table gives an entry a different code than the index file')
    return joined.select(INDEX_ROLE_COLUMNS).sort('entry_id')

# -------------------------------------------------------------------------------------------------
# Consistency with the descriptions built from the roles
# -------------------------------------------------------------------------------------------------

def examples_channel_by_code(role_rows: pl.DataFrame) -> pl.DataFrame:
    '''Each code's examples channel: its examples-role entries in index-file order, joined.'''

    # yapf: disable
    return (
        role_rows
        .filter(pl.col('role') == IndexRole.EXAMPLES.value)
        .sort('entry_id')
        .group_by('code', maintain_order=True)
        .agg(examples=pl.col('text').str.join(EXAMPLES_SEPARATOR))
    )
    # yapf: enable

def verify_examples_channel(descriptions: pl.DataFrame, role_rows: pl.DataFrame) -> None:
    '''
    Require every code with index entries to carry exactly its examples-role entries.

    Raises:
        ValueError: If a code's examples channel differs from its examples-role entries.
    '''

    expected = role_rows.select('code').unique().join(
        examples_channel_by_code(role_rows), on='code', how='left'
    )
    checked = expected.join(
        descriptions.select('code', actual=pl.col('examples')), on='code', how='left'
    )
    mismatched = checked.filter(pl.col('examples').ne_missing(pl.col('actual')))
    if mismatched.height:
        raise ValueError(
            f'{mismatched.height:,} codes have an examples channel other than their '
            f'examples-role entries, e.g. {mismatched.get_column("code").sort().to_list()[:5]}'
        )

def verify_role_leakage(
    descriptions: pl.DataFrame,
    role_rows: pl.DataFrame,
    min_jaccard: Fraction = NEAR_DUPLICATE_MIN_JACCARD,
) -> Dict[str, Dict[str, int]]:
    '''
    Match every validation and test query against all training text; fail on any match.

    Training text is every code's title, description, examples channel and exclusion text in
    ``descriptions``, plus every training-role query.

    Returns:
        Exact and near-duplicate match counts per held-out split (all zero on success).
    '''

    training = role_rows.filter(pl.col('role') == IndexRole.TRAINING.value)
    corpus = training_text_segments(descriptions, extra_texts=training.get_column('text'))
    report: Dict[str, Dict[str, int]] = {}
    for role in (IndexRole.VALIDATION, IndexRole.TEST):
        queries = role_rows.filter(pl.col('role') == role.value).get_column('text').to_list()
        matches = find_leakage(queries, corpus, min_jaccard=min_jaccard)
        report[role.value] = {
            'exact': int(matches.exact.sum()),
            'near_duplicate': int(matches.near_duplicate.sum()),
        }
    leaked = {split: found for split, found in report.items() if any(found.values())}
    if leaked:
        raise ValueError(f'held-out queries match training text: {leaked}')
    return report
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_index_roles.py -q`
Expected: `34 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1294 passed, 1 skipped`.

- [x] **Step 6: Lint and format**

Run: `./scripts/format_code.sh --check src/naics_embedder/supervision/schema.py src/naics_embedder/supervision/artifacts.py src/naics_embedder/panels/index_roles.py tests/unit/test_index_roles.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` On a failure, run the
same command without `--check`, re-run Step 5, and record the change as a deviation.

- [x] **Step 7: Commit**

```bash
git add src/naics_embedder/supervision/schema.py src/naics_embedder/supervision/artifacts.py src/naics_embedder/panels/index_roles.py tests/unit/test_index_roles.py
git commit -m "feat(panels): assign index-entry roles per code with leak-free held-out splits"
```

### Task 3: The decoding scorer

The scorer decodes each query point to the nearest of the 1,012 six-digit candidates and
reports Req 3's four metrics. Any encoder or distance plugs in, so later stages score their own
arms with it unchanged.

**Files:**

- Create: `src/naics_embedder/panels/decoding.py`
- Test: `tests/unit/test_outcome_decoding.py`

**Interfaces:**

- Consumes: `naics_embedder.utils.naics_hierarchy.naics_parent_code` (existing), which gives a
  code's parent and handles the combined sectors.
- Produces (`naics_embedder.panels.decoding`):
  - `HIT_KS = (1, 5, 10)`
  - `METRIC_NAMES = ('top1', 'mrr', 'hit_at_1', 'hit_at_5', 'hit_at_10', 'lca_level')`
  - `DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
  - Distances:
    - `euclidean_distances`, `cosine_distances` and `lorentz_distances`, each
      `(queries (Q, D), candidates (C, D)) -> (Q, C)`, float64 CPU inputs
    - `DISTANCES`, which maps each name to its function
  - `resolve_distance(distance: Union[str, DistanceFn]) -> Tuple[str, DistanceFn]`. It raises
    `ValueError` (`unknown distance …`).
  - `code_lineage(code: str) -> Tuple[str, ...]` and `lca_level(code_a: str, code_b: str) -> int`.
    The level runs from 1 (the virtual root) to 6 (the same code).
  - `DecodingResult(per_query: pl.DataFrame, per_code: pl.DataFrame, summary: Dict[str, Any])`
  - `score_decoding(query_points: torch.Tensor, query_codes: Sequence[str], candidate_points: torch.Tensor, candidate_codes: Sequence[str], distance: Union[str, DistanceFn] = 'cosine', query_ids: Optional[Sequence[int]] = None) -> DecodingResult`
    - `per_query` columns: `query_id`, `code`, `top1_code`, `rank`, `reciprocal_rank`,
      `hit_at_1`, `hit_at_5`, `hit_at_10` and `lca_level`.
    - `per_code`: `code`, `n_queries` and the metric means.
    - `summary`: `distance`, `n_queries`, `n_codes` and `n_candidates`, then every name in
      `METRIC_NAMES`.
    - Ranks break ties against the truth.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_outcome_decoding.py` with exactly this content:

```python
'''
Text-to-code decoding scores (Req 3): distances, ranks with pessimistic ties, partial credit.

Expected values are worked out by hand, never from production code.
'''

import math

import pytest
import torch

from naics_embedder.metrics.core import lorentz_distance_matrix
from naics_embedder.panels.decoding import (
    cosine_distances,
    euclidean_distances,
    lca_level,
    lorentz_distances,
    score_decoding,
)

pytestmark = pytest.mark.unit

# -------------------------------------------------------------------------------------------------
# Distances
# -------------------------------------------------------------------------------------------------

def _f64(rows):
    return torch.tensor(rows, dtype=torch.float64)

def test_euclidean_distances():
    distances = euclidean_distances(_f64([[0.0, 0.0]]), _f64([[3.0, 4.0], [0.0, 1.0]]))

    assert distances.tolist() == [[5.0, 1.0]]

def test_cosine_distances():
    distances = cosine_distances(_f64([[1.0, 0.0]]), _f64([[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]]))

    assert distances.tolist() == [[0.0, 1.0, 2.0]]

def test_lorentz_distances_rederive_the_time_coordinate():
    origin = _f64([[1.0, 0.0, 0.0]])
    points = _f64([[math.cosh(1.0), math.sinh(1.0), 0.0], [math.cosh(2.0), 0.0, math.sinh(2.0)]])
    off_manifold = points.clone()
    off_manifold[:, 0] = 0.0

    torch.testing.assert_close(lorentz_distances(origin, points), _f64([[1.0, 2.0]]))
    torch.testing.assert_close(lorentz_distances(origin, off_manifold), _f64([[1.0, 2.0]]))

def test_lorentz_distances_agree_with_the_evaluation_matrix():
    generator = torch.Generator().manual_seed(0)
    space = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    points = torch.cat([torch.sqrt(1.0 + (space * space).sum(dim=1, keepdim=True)), space], dim=1)

    expected = lorentz_distance_matrix(points)[:2, 2:]

    torch.testing.assert_close(lorentz_distances(points[:2], points[2:]), expected)

# -------------------------------------------------------------------------------------------------
# Partial credit
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('code_a', 'code_b', 'level'),
    [
        ('111110', '111110', 6),
        ('111110', '111120', 4),
        ('111110', '111211', 3),
        ('311111', '332111', 2),  # 31-33 is one sector
        ('441110', '452210', 2),  # 44-45
        ('481111', '493110', 2),  # 48-49
        ('111110', '211120', 1),  # only the virtual root
    ],
)
def test_lca_level(code_a, code_b, level):
    assert lca_level(code_a, code_b) == level
    assert lca_level(code_b, code_a) == level

# -------------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------------

CANDIDATES = ['111110', '111120', '111211', '211111']
CANDIDATE_POINTS = [[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [5.0, 5.0]]

@pytest.fixture
def scored():
    return score_decoding(
        torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.0, 2.9], [0.6, 0.0]]),
        ['111110', '111120', '211111', '111110'],
        torch.tensor(CANDIDATE_POINTS),
        CANDIDATES,
        distance='euclidean',
        query_ids=[10, 11, 12, 13],
    )

def test_per_query_ranks_top1_and_partial_credit(scored):
    # 0.1 from its code; 0.8 from its code but 0.2 from 111110; 5.42 from its code and nearer
    # every other candidate; 0.6 from its code but 0.4 from 111120
    assert scored.per_query.select('query_id', 'code', 'top1_code', 'rank', 'lca_level').rows() == [
        (10, '111110', '111110', 1, 6),
        (11, '111120', '111110', 2, 4),
        (12, '211111', '111211', 4, 1),
        (13, '111110', '111120', 2, 4),
    ]
    assert scored.per_query.get_column('hit_at_1').to_list() == [True, False, False, False]
    assert scored.per_query.get_column('hit_at_5').to_list() == [True, True, True, True]

def test_summary_reports_every_metric_query_weighted(scored):
    assert scored.summary == {
        'distance': 'euclidean',
        'n_queries': 4,
        'n_codes': 3,
        'n_candidates': 4,
        'top1': 0.25,
        'mrr': 0.5625,  # (1 + 1/2 + 1/4 + 1/2) / 4
        'hit_at_1': 0.25,
        'hit_at_5': 1.0,
        'hit_at_10': 1.0,
        'lca_level': 3.75,  # (6 + 4 + 1 + 4) / 4
    }

def test_per_code_means(scored):
    assert scored.per_code.select('code', 'n_queries', 'top1', 'mrr', 'lca_level').rows() == [
        ('111110', 2, 0.5, 0.75, 5.0),
        ('111120', 1, 0.0, 0.5, 4.0),
        ('211111', 1, 0.0, 0.25, 1.0),
    ]

def test_a_constant_encoder_ranks_every_truth_last():
    codes = [f'1111{index:02d}' for index in range(12)]

    result = score_decoding(
        torch.ones(3, 2), codes[:3], torch.ones(12, 2), codes, distance='euclidean'
    )

    assert result.per_query.get_column('rank').to_list() == [12, 12, 12]
    assert result.summary['top1'] == 0.0
    assert result.summary['mrr'] == pytest.approx(1 / 12)
    assert result.summary['hit_at_10'] == 0.0

def test_a_tie_with_the_truth_counts_against_it():
    result = score_decoding(
        torch.tensor([[0.5, 0.0]]),
        ['111120'],
        torch.tensor(CANDIDATE_POINTS),
        CANDIDATES,
        distance='euclidean',
    )

    assert result.per_query.select('rank', 'top1_code').rows() == [(2, '111110')]

def test_a_custom_distance_is_pluggable():

    def manhattan(queries, candidates):
        return torch.cdist(queries, candidates, p=1.0)

    result = score_decoding(
        torch.tensor([[0.0, 2.5]]),
        ['111211'],
        torch.tensor(CANDIDATE_POINTS),
        CANDIDATES,
        distance=manhattan,
    )

    assert result.summary['distance'] == 'manhattan'
    assert result.summary['top1'] == 1.0

@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({
            'query_codes': ['112130']
        }, 'not candidates'),
        ({
            'candidate_codes': ['111110', '111110', '111211', '211111']
        }, 'distinct'),
        ({
            'query_points': torch.zeros(1, 3)
        }, 'coordinates'),
        ({
            'query_points': torch.zeros(2, 2)
        }, 'must agree'),
        ({
            'query_points': torch.zeros(0, 2),
            'query_codes': []
        }, 'non-empty 2-D'),
        ({
            'distance': 'manhattan'
        }, 'unknown distance'),
        ({
            'distance': lambda q, c: torch.full((q.shape[0], c.shape[0]), math.nan)
        }, 'non-finite'),
    ],
)
def test_inconsistent_inputs_fail_closed(kwargs, message):
    arguments = {
        'query_points': torch.zeros(1, 2),
        'query_codes': ['111110'],
        'candidate_points': torch.tensor(CANDIDATE_POINTS),
        'candidate_codes': CANDIDATES,
        'distance': 'euclidean',
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        score_decoding(**arguments)

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs an MPS device')
def test_points_on_mps_are_scored_in_float64_on_the_cpu(scored):
    result = score_decoding(
        torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.0, 2.9], [0.6, 0.0]], device='mps'),
        ['111110', '111120', '211111', '111110'],
        torch.tensor(CANDIDATE_POINTS, device='mps'),
        CANDIDATES,
        distance='euclidean',
        query_ids=[10, 11, 12, 13],
    )

    assert result.summary == scored.summary
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_outcome_decoding.py -q`
Expected: a collection error,
`ModuleNotFoundError: No module named 'naics_embedder.panels.decoding'`.

- [x] **Step 3: Write the implementation**

Create `src/naics_embedder/panels/decoding.py` with exactly this content:

```python
'''
Text-to-code decoding scores for the outcome panel (Req 3).

Each query decodes to the nearest candidate code under the arm's own distance. Ties are broken
against the truth: the true code's rank counts every candidate at a distance no greater than its
own, so an encoder that places every point alike ranks the truth last, never first. Distances
are computed in float64 on the CPU whatever the device or dtype of the points.

Metrics per query: exact top-1 accuracy, reciprocal rank, Hit@k for k in ``HIT_KS``, and the
level of the lowest common ancestor of the top-1 code and the truth (6 for the same six-digit
code, 2 for a shared sector, 1 for the virtual root above the sectors).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import polars as pl
import torch
import torch.nn.functional as F

from naics_embedder.utils.naics_hierarchy import naics_parent_code

HIT_KS = (1, 5, 10)
METRIC_NAMES = ('top1', 'mrr', *(f'hit_at_{k}' for k in HIT_KS), 'lca_level')
DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# -------------------------------------------------------------------------------------------------
# Distances (float64 CPU inputs, one row per point)
# -------------------------------------------------------------------------------------------------

def euclidean_distances(queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    '''Euclidean distances, (Q, C), without the matrix-product shortcut that loses precision.'''

    return torch.cdist(queries, candidates, compute_mode='donot_use_mm_for_euclid_dist')

def cosine_distances(queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    '''One minus cosine similarity, (Q, C).'''

    return 1.0 - F.normalize(queries, dim=1) @ F.normalize(candidates, dim=1).T

def lorentz_distances(queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    '''
    Geodesic distances on the curvature -1 hyperboloid, (Q, C), for (time, space) rows.

    The time coordinate is re-derived from the spatial ones, as ``metrics.core``'s
    ``lorentz_distance_matrix`` does, so a point rounded off the hyperboloid is read at its
    spatial position.
    '''

    q_space, c_space = queries[:, 1:], candidates[:, 1:]
    q_time = torch.sqrt(1.0 + (q_space * q_space).sum(dim=1))
    c_time = torch.sqrt(1.0 + (c_space * c_space).sum(dim=1))
    return torch.acosh(torch.clamp(torch.outer(q_time, c_time) - q_space @ c_space.T, min=1.0))

DISTANCES: Dict[str, DistanceFn] = {
    'euclidean': euclidean_distances,
    'cosine': cosine_distances,
    'lorentz': lorentz_distances,
}

def resolve_distance(distance: Union[str, DistanceFn]) -> Tuple[str, DistanceFn]:
    '''A registered distance by name, or a callable with its ``__name__``.'''

    if callable(distance):
        return getattr(distance, '__name__', 'custom'), distance
    if distance not in DISTANCES:
        raise ValueError(f'unknown distance {distance!r}; expected one of {sorted(DISTANCES)}')
    return distance, DISTANCES[distance]

# -------------------------------------------------------------------------------------------------
# Hierarchical partial credit
# -------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def code_lineage(code: str) -> Tuple[str, ...]:
    '''The code's ancestors from its sector down to the code itself.'''

    chain = [code]
    parent = naics_parent_code(code)
    while parent is not None:
        chain.append(parent)
        parent = naics_parent_code(parent)
    return tuple(reversed(chain))

def lca_level(code_a: str, code_b: str) -> int:
    '''
    Level of the lowest common ancestor of two codes, counting the virtual root as level 1.

    A shared sector is level 2 and the same six-digit code level 6; combined sectors (31-33,
    44-45, 48-49) count as one sector.
    '''

    shared = 0
    for ancestor_a, ancestor_b in zip(code_lineage(code_a), code_lineage(code_b)):
        if ancestor_a != ancestor_b:
            break
        shared += 1
    return shared + 1

# -------------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecodingResult:
    '''
    Decoding scores at three grains.

    Attributes:
        per_query: One row per query: ``query_id``, ``code``, ``top1_code``, ``rank``,
            ``reciprocal_rank``, ``hit_at_k`` per k and ``lca_level``. Stage 4 resamples it by
            code.
        per_code: One row per true code: ``code``, ``n_queries`` and each metric's mean.
        summary: Query-weighted means of every metric, with ``n_queries``, ``n_codes``,
            ``n_candidates`` and the distance's name.
    '''

    per_query: pl.DataFrame
    per_code: pl.DataFrame
    summary: Dict[str, Union[int, float, str]]

def _as_float64(points: torch.Tensor, name: str) -> torch.Tensor:
    if points.dim() != 2 or points.shape[0] == 0:
        raise ValueError(f'{name} must be a non-empty 2-D tensor, got shape {tuple(points.shape)}')
    # Move before casting: MPS has no float64
    return points.detach().cpu().to(torch.float64).contiguous()

def _metric_exprs():
    return [
        (pl.col('rank') == 1).alias('top1'),
        pl.col('reciprocal_rank').alias('mrr'),
        *(pl.col(f'hit_at_{k}') for k in HIT_KS),
        pl.col('lca_level'),
    ]

def score_decoding(
    query_points: torch.Tensor,
    query_codes: Sequence[str],
    candidate_points: torch.Tensor,
    candidate_codes: Sequence[str],
    distance: Union[str, DistanceFn] = 'cosine',
    query_ids: Optional[Sequence[int]] = None,
) -> DecodingResult:
    '''
    Decode every query to its nearest candidate and score it against the query's true code.

    Args:
        query_points: Query embeddings, (Q, D).
        query_codes: The true code of each query.
        candidate_points: Candidate embeddings, (C, D), in the same space.
        candidate_codes: Distinct candidate codes, one per row of ``candidate_points``.
        distance: A name in ``DISTANCES`` or a callable taking float64 CPU (Q, D) and (C, D)
            tensors and returning (Q, C) distances.
        query_ids: Identifiers carried into ``per_query`` (index entry ids); 0..Q-1 if omitted.

    Raises:
        ValueError: On mismatched shapes, repeated candidates, a query code that is not a
            candidate, or a distance that is not finite.
    '''

    name, distance_fn = resolve_distance(distance)
    queries = _as_float64(query_points, 'query_points')
    candidates = _as_float64(candidate_points, 'candidate_points')
    query_codes = [str(code) for code in query_codes]
    candidate_codes = [str(code) for code in candidate_codes]
    query_ids = list(range(len(query_codes))) if query_ids is None else list(query_ids)
    if queries.shape[0] != len(query_codes) or len(query_ids) != len(query_codes):
        raise ValueError(
            f'{queries.shape[0]} query points, {len(query_codes)} query codes and '
            f'{len(query_ids)} query ids must agree'
        )
    if candidates.shape[0] != len(candidate_codes):
        raise ValueError(
            f'{candidates.shape[0]} candidate points for {len(candidate_codes)} candidate codes'
        )
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError(
            f'queries have {queries.shape[1]} coordinates, candidates {candidates.shape[1]}'
        )
    position = {code: index for index, code in enumerate(candidate_codes)}
    if len(position) != len(candidate_codes):
        raise ValueError('candidate codes must be distinct')
    unknown = sorted({code for code in query_codes if code not in position})
    if unknown:
        raise ValueError(f'{len(unknown):,} query codes are not candidates, e.g. {unknown[:5]}')

    distances = distance_fn(queries, candidates)
    if distances.shape != (len(query_codes), len(candidate_codes)):
        raise ValueError(f'distance returned shape {tuple(distances.shape)}')
    if not torch.isfinite(distances).all():
        raise ValueError(f'distance {name!r} returned non-finite values')

    rows = torch.arange(len(query_codes))
    true_index = torch.tensor([position[code] for code in query_codes])
    ranks = (distances <= distances[rows, true_index][:, None]).sum(dim=1)
    runner_up = distances.clone()
    runner_up[rows, true_index] = math.inf
    top1_index = torch.where(ranks == 1, true_index, runner_up.argmin(dim=1))
    top1_codes = [candidate_codes[index] for index in top1_index.tolist()]

    per_query = pl.DataFrame(
        {
            'query_id': query_ids,
            'code': query_codes,
            'top1_code': top1_codes,
            'rank': ranks.tolist(),
        },
        schema={
            'query_id': pl.Int64,
            'code': pl.Utf8,
            'top1_code': pl.Utf8,
            'rank': pl.Int64
        },
    ).with_columns(
        reciprocal_rank=1.0 / pl.col('rank'),
        **{f'hit_at_{k}': pl.col('rank') <= k
           for k in HIT_KS},
        lca_level=pl.Series(
            [lca_level(truth, top1) for truth, top1 in zip(query_codes, top1_codes)],
            dtype=pl.Int64,
        ),
    )
    means = per_query.select(expr.mean() for expr in _metric_exprs()).row(0, named=True)
    # yapf: disable
    per_code = (
        per_query
        .group_by('code')
        .agg(pl.len().alias('n_queries'), *(expr.mean() for expr in _metric_exprs()))
        .sort('code')
    )
    # yapf: enable
    summary: Dict[str, Union[int, float, str]] = {
        'distance': name,
        'n_queries': per_query.height,
        'n_codes': per_code.height,
        'n_candidates': len(candidate_codes),
        **{
            metric: float(means[metric])
            for metric in METRIC_NAMES
        },
    }
    return DecodingResult(per_query=per_query, per_code=per_code, summary=summary)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_outcome_decoding.py -q`
Expected: `25 passed`. This Mac has MPS, so the MPS test
runs here. CI skips it.

Run: `uv run pytest -n auto -q`
Expected: `1319 passed, 1 skipped`.

- [x] **Step 5: Lint and format**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/decoding.py tests/unit/test_outcome_decoding.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` On a failure, run the
same command without `--check`, re-run Step 4, and record the change as a deviation.

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/panels/decoding.py tests/unit/test_outcome_decoding.py
git commit -m "feat(panels): add the text-to-code decoding scorer"
```

### Task 4: The selection log and the sealed test split

`OutcomePanel` is the only way later stages read queries. It logs every validation and test
read, and it refuses to read the test split until this panel object has opened it with a logged
call. A second opening of the same split needs a stated reason. The Exit test at the end of the
test file drives the whole path on a one-hot stub encoder.

**Files:**

- Create: `src/naics_embedder/panels/selection_log.py`
- Create: `src/naics_embedder/panels/outcome.py`
- Test: `tests/unit/test_outcome_panel.py`

**Interfaces:**

- Consumes:
  - Task 2: `IndexRole`, `INDEX_ROLE_COLUMNS`, `validate_index_role_table`,
    `role_table_fingerprint` and `verify_examples_channel`.
  - Task 3: `DecodingResult`, `DistanceFn`, `resolve_distance` and `score_decoding`.
- Produces in `naics_embedder.panels.selection_log`:
  - `class SelectionEvent(str, Enum)`: `READ = 'read'`, `OPEN = 'open'` and
    `REOPEN = 'reopen'`.
  - `SelectionLog(path: Path)`:
    - `append(event, *, panel, split, purpose, fingerprint, n_queries, detail=None) -> Dict[str, Any]`
      writes one JSON line with sorted keys: `detail`, `event`, `fingerprint`, `n_queries`,
      `panel`, `purpose`, `split` and `time` (UTC, ISO 8601). It returns the record. A blank
      `purpose` raises `ValueError` (`every selection-log record needs a purpose`).
    - `records() -> List[Dict[str, Any]]`
    - `openings(panel: str, fingerprint: str) -> List[Dict[str, Any]]`
- Produces in `naics_embedder.panels.outcome`:
  - `OUTCOME_PANEL = 'outcome'`
  - `SealedSplitError(RuntimeError)` and `SplitAlreadyOpenedError(RuntimeError)`
  - `QueryCodeEncoder`, a Protocol with
    `encode_queries(texts: Sequence[str]) -> torch.Tensor` and
    `encode_codes(codes: Sequence[str]) -> torch.Tensor`.
  - `OutcomePanel(role_rows: pl.DataFrame, candidates: Sequence[str], log: SelectionLog)`.
    Its attributes are `candidates: Tuple[str, ...]` (sorted), `log` and `fingerprint` (the role
    table's fingerprint).
    - `OutcomePanel.from_files(index_roles_parquet, descriptions_parquet, log_path)`. It refuses
      descriptions whose examples channel is not the roles' (`verify_examples_channel`).
    - `entryless_candidates` (property) `-> Tuple[str, ...]`
    - `training_queries() -> pl.DataFrame`, not logged
    - `validation_queries(purpose: str) -> pl.DataFrame`
    - `open_test(purpose: str, *, reopen_reason: Optional[str] = None) -> None`
    - `test_queries(purpose: str) -> pl.DataFrame`
    - `score(encoder, split, purpose, distance='cosine') -> DecodingResult`. Its read record's
      `detail` is `{'encoder': <class name>, 'distance': <name>}`.
    - Query frames have the columns `entry_id`, `code` and `text`.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_outcome_panel.py` with exactly this content:

```python
'''
The outcome panel and its selection log (Req 3; Req 4; roadmap Stage 2 Exit).

A one-hot stub encoder stands in for a trained arm: each code sits on its own axis, and each query
sits on the axis of the code the stub assigns it, so every rank below is worked out by hand.
'''

import json

import polars as pl
import pytest
import torch

from naics_embedder.panels.decoding import METRIC_NAMES
from naics_embedder.panels.index_roles import role_table_fingerprint
from naics_embedder.panels.outcome import (
    OutcomePanel,
    SealedSplitError,
    SplitAlreadyOpenedError,
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog

pytestmark = pytest.mark.unit

CANDIDATES = ['111110', '111120', '112130', '211111', '311111']
ROWS = [
    (0, '111110', 'Soybean farming', 'examples'),
    (1, '111110', 'Soybean seed production', 'training'),
    (2, '111110', 'Edamame farming', 'validation'),
    (3, '111110', 'Soybeans, organic', 'test'),
    (4, '111120', 'Oilseed farming', 'examples'),
    (5, '111120', 'Canola farming', 'validation'),
    (6, '111120', 'Sunflower farming', 'test'),
    (7, '211111', 'Crude petroleum extraction', 'examples'),
    (8, '211111', 'Oil sands mining', 'test'),
    (9, '311111', 'Dog food manufacturing', 'examples'),
    (10, '311111', 'Cat food manufacturing', 'validation'),
]
# The stub decodes 'Canola farming' and 'Oil sands mining' to the wrong code
ASSIGNED = {
    'Edamame farming': '111110',
    'Canola farming': '111110',
    'Cat food manufacturing': '311111',
    'Soybeans, organic': '111110',
    'Sunflower farming': '111120',
    'Oil sands mining': '111120',
}

class OneHotStubEncoder:
    '''Each code on its own axis; each query on the axis of the code the stub assigns it.'''

    def __init__(self, candidates, assigned):
        self.axis = {code: index for index, code in enumerate(candidates)}
        self.assigned = assigned

    def _points(self, codes):
        indices = torch.tensor([self.axis[code] for code in codes])
        return torch.nn.functional.one_hot(indices, len(self.axis)).to(torch.float32)

    def encode_codes(self, codes):
        return self._points(codes)

    def encode_queries(self, texts):
        return self._points([self.assigned[text] for text in texts])

@pytest.fixture
def role_rows():
    return pl.DataFrame(
        ROWS,
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
def panel(role_rows, log):
    return OutcomePanel(role_rows, CANDIDATES, log)

@pytest.fixture
def encoder():
    return OneHotStubEncoder(CANDIDATES, ASSIGNED)

def _events(log):
    return [(record['event'], record['split']) for record in log.records()]

# -------------------------------------------------------------------------------------------------
# Roles and candidates
# -------------------------------------------------------------------------------------------------

def test_every_entry_holds_one_role_and_entryless_codes_are_never_queries(panel):
    panel.open_test('final configuration')
    splits = {
        'training': panel.training_queries(),
        'validation': panel.validation_queries('check the splits'),
        'test': panel.test_queries('check the splits'),
    }
    query_ids = [entry_id for frame in splits.values() for entry_id in frame['entry_id']]

    assert panel.candidates == tuple(CANDIDATES)
    assert panel.entryless_candidates == ('112130', )
    assert sorted(query_ids) == [1, 2, 3, 5, 6, 8, 10]  # entries 0, 4, 7, 9 are examples text
    assert len(query_ids) == len(set(query_ids))
    for frame in splits.values():
        assert '112130' not in frame['code'].to_list()

def test_the_fingerprint_identifies_the_assignment(panel, role_rows):
    assert panel.fingerprint == role_table_fingerprint(role_rows)

@pytest.mark.parametrize(
    'candidates',
    [CANDIDATES[:1] + CANDIDATES, CANDIDATES + ['11111'], CANDIDATES[1:]],
)
def test_candidates_must_be_distinct_six_digit_codes_covering_the_queries(
    role_rows, log, candidates
):
    with pytest.raises(ValueError):
        OutcomePanel(role_rows, candidates, log)

# -------------------------------------------------------------------------------------------------
# Selection log and sealing
# -------------------------------------------------------------------------------------------------

def test_validation_reads_are_logged_and_training_reads_are_not(panel, log):
    assert panel.training_queries().height == 1
    assert log.records() == []

    validation = panel.validation_queries('tune the learning rate')

    assert validation['entry_id'].to_list() == [2, 5, 10]
    [record] = log.records()
    assert record['event'] == 'read'
    assert record['panel'] == 'outcome'
    assert record['split'] == 'validation'
    assert record['purpose'] == 'tune the learning rate'
    assert record['fingerprint'] == panel.fingerprint
    assert record['n_queries'] == 3

def test_the_test_split_is_sealed_until_a_logged_opening(panel, log, encoder):
    with pytest.raises(SealedSplitError):
        panel.test_queries('peek')
    with pytest.raises(SealedSplitError):
        panel.score(encoder, 'test', 'peek')
    assert log.records() == []

    panel.open_test('final configuration')

    assert panel.test_queries('final configuration')['entry_id'].to_list() == [3, 6, 8]
    assert _events(log) == [('open', 'test'), ('read', 'test')]

def test_every_panel_object_must_open_the_test_split_itself(role_rows, log):
    OutcomePanel(role_rows, CANDIDATES, log).open_test('final configuration')
    later = OutcomePanel(role_rows, CANDIDATES, log)

    with pytest.raises(SealedSplitError):
        later.test_queries('reuse the earlier opening')

def test_a_second_opening_needs_a_reason_and_is_logged_as_a_reopen(role_rows, log):
    OutcomePanel(role_rows, CANDIDATES, log).open_test('final configuration')
    later = OutcomePanel(role_rows, CANDIDATES, log)

    with pytest.raises(SplitAlreadyOpenedError, match='reopen_reason'):
        later.open_test('final configuration')
    later.open_test('final configuration', reopen_reason='the first run crashed before scoring')

    assert _events(log) == [('open', 'test'), ('reopen', 'test')]
    assert log.records()[-1]['detail'] == {'reason': 'the first run crashed before scoring'}

def test_openings_are_counted_per_role_assignment(role_rows, log):
    OutcomePanel(role_rows, CANDIDATES, log).open_test('final configuration')
    reassigned = role_rows.with_columns(
        role=pl.when(pl.col('entry_id') == 1).then(pl.lit('test')).otherwise('role')
    )

    OutcomePanel(reassigned, CANDIDATES, log).open_test('final configuration')

    assert _events(log) == [('open', 'test'), ('open', 'test')]

def test_only_validation_and_test_splits_are_scored(panel, encoder):
    with pytest.raises(ValueError, match='only validation and test'):
        panel.score(encoder, 'training', 'fit check')

def test_log_records_need_a_purpose(log):
    with pytest.raises(ValueError, match='purpose'):
        log.append(
            SelectionEvent.READ,
            panel='outcome',
            split='validation',
            purpose='  ',
            fingerprint='f',
            n_queries=0,
        )

def test_the_log_is_append_only_json_lines(panel, log):
    panel.validation_queries('first')
    panel.validation_queries('second')

    lines = log.path.read_text().splitlines()
    assert [json.loads(line)['purpose'] for line in lines] == ['first', 'second']

# -------------------------------------------------------------------------------------------------
# Exit: a stub encoder scored on the sealed splits
# -------------------------------------------------------------------------------------------------

def test_a_stub_encoder_scores_every_metric_on_both_splits(panel, log, encoder):
    validation = panel.score(encoder, 'validation', 'stub check')
    panel.open_test('stub check on the fixture panel')
    test = panel.score(encoder, 'test', 'stub check on the fixture panel')

    # Validation: entries 2 and 10 decode correctly (rank 1); entry 5 lands on 111110, so its
    # code ties the three other off-axis candidates and ranks last of 5
    assert validation.per_query.select('query_id', 'rank', 'lca_level').rows() == [
        (2, 1, 6),
        (5, 5, 4),
        (10, 1, 6),
    ]
    assert set(METRIC_NAMES) <= set(validation.summary)
    assert validation.summary['top1'] == pytest.approx(2 / 3)
    assert validation.summary['mrr'] == pytest.approx((1 + 1 / 5 + 1) / 3)
    assert validation.summary['hit_at_5'] == 1.0
    assert validation.summary['lca_level'] == pytest.approx(16 / 3)
    # Test: entry 8 (211111) lands on 111120, which shares only the virtual root with it
    assert test.per_query.select('query_id', 'rank', 'lca_level').rows() == [
        (3, 1, 6),
        (6, 1, 6),
        (8, 5, 1),
    ]
    assert test.summary['n_candidates'] == 5
    assert test.summary['lca_level'] == pytest.approx(13 / 3)
    assert _events(log) == [('read', 'validation'), ('open', 'test'), ('read', 'test')]
    assert log.records()[0]['detail'] == {'encoder': 'OneHotStubEncoder', 'distance': 'cosine'}

# -------------------------------------------------------------------------------------------------
# Loading from preprocessing outputs
# -------------------------------------------------------------------------------------------------

def _descriptions(examples):
    return pl.DataFrame(
        {
            'code': CANDIDATES + ['11111'],
            'examples': examples + [None],
        },
        schema={
            'code': pl.Utf8,
            'examples': pl.Utf8
        },
    )

def test_from_files_reads_the_preprocessing_outputs(tmp_path, role_rows):
    roles_path = tmp_path / 'naics_index_roles.parquet'
    descriptions_path = tmp_path / 'naics_descriptions.parquet'
    role_rows.write_parquet(roles_path)
    _descriptions(
        [
            'Soybean farming',
            'Oilseed farming',
            None,
            'Crude petroleum extraction',
            'Dog food manufacturing',
        ]
    ).write_parquet(descriptions_path)

    panel = OutcomePanel.from_files(roles_path, descriptions_path, tmp_path / 'log.jsonl')

    assert panel.candidates == tuple(CANDIDATES)
    assert panel.fingerprint == role_table_fingerprint(role_rows)

def test_from_files_refuses_descriptions_that_hold_every_entry(tmp_path, role_rows):
    roles_path = tmp_path / 'naics_index_roles.parquet'
    descriptions_path = tmp_path / 'naics_descriptions.parquet'
    role_rows.write_parquet(roles_path)
    everything = role_rows.sort('entry_id').group_by('code', maintain_order=True).agg(
        pl.col('text').str.join('; ')
    )
    by_code = dict(everything.iter_rows())
    _descriptions([by_code.get(code) for code in CANDIDATES]).write_parquet(descriptions_path)

    with pytest.raises(ValueError, match='examples channel other than'):
        OutcomePanel.from_files(roles_path, descriptions_path, tmp_path / 'log.jsonl')
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_outcome_panel.py -q`
Expected: a collection error,
`ModuleNotFoundError: No module named 'naics_embedder.panels.outcome'`.

- [x] **Step 3: Write the selection log**

Create `src/naics_embedder/panels/selection_log.py` with exactly this content:

```python
'''
The selection log (Req 4; Verification "Selection hygiene").

An append-only JSON-lines file with one record per event: every read of a panel's validation or
test split, and every opening of a test split. It is the evidence that selections read validation
splits only and that each test split was opened once, for the final configuration. A second
opening needs a stated reason and is recorded as ``reopen``.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Log
# -------------------------------------------------------------------------------------------------

class SelectionEvent(str, Enum):
    '''What a selection-log record reports.'''

    READ = 'read'
    OPEN = 'open'
    REOPEN = 'reopen'

class SelectionLog:
    '''Append-only JSON-lines log of panel reads and test-split openings.'''

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(
        self,
        event: SelectionEvent,
        *,
        panel: str,
        split: str,
        purpose: str,
        fingerprint: str,
        n_queries: int,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        '''
        Append one record and return it.

        Raises:
            ValueError: If ``purpose`` is blank: every read and opening states why it happens.
        '''

        if not purpose.strip():
            raise ValueError('every selection-log record needs a purpose')
        record = {
            'time': datetime.now(timezone.utc).isoformat(),
            'event': SelectionEvent(event).value,
            'panel': panel,
            'split': split,
            'purpose': purpose,
            'fingerprint': fingerprint,
            'n_queries': n_queries,
            'detail': dict(detail or {}),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
        logger.info(f'Selection log: {record["event"]} {panel}/{split} ({purpose}) -> {self.path}')
        return record

    def records(self) -> List[Dict[str, Any]]:
        '''Every record, oldest first; a log that does not exist yet has none.'''

        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding='utf-8').splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def openings(self, panel: str, fingerprint: str) -> List[Dict[str, Any]]:
        '''The recorded openings (``open`` and ``reopen``) of one panel's test split.'''

        opening = {SelectionEvent.OPEN.value, SelectionEvent.REOPEN.value}
        return [
            record for record in self.records() if record['event'] in opening
            and record['panel'] == panel and record['fingerprint'] == fingerprint
        ]
```

- [x] **Step 4: Write the panel**

Create `src/naics_embedder/panels/outcome.py` with exactly this content:

```python
'''
The outcome panel (Req 3; Req 4): text-to-code decoding over sealed query splits.

Validation queries may be read at any time, and every read is logged. The test split is sealed:
reading it needs a logged opening by the same panel object, and opening the same split (the same
role assignment, by fingerprint) a second time needs a stated reason. Training queries are
training data, not a selection, so reading them is not logged.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple, Union

import polars as pl
import torch

from naics_embedder.panels.decoding import (
    DecodingResult,
    DistanceFn,
    resolve_distance,
    score_decoding,
)
from naics_embedder.panels.index_roles import role_table_fingerprint, verify_examples_channel
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.supervision.artifacts import INDEX_ROLE_COLUMNS, validate_index_role_table
from naics_embedder.supervision.schema import IndexRole

OUTCOME_PANEL = 'outcome'

# -------------------------------------------------------------------------------------------------
# Errors and the encoder interface
# -------------------------------------------------------------------------------------------------

class SealedSplitError(RuntimeError):
    '''The test split was read before a logged opening.'''

class SplitAlreadyOpenedError(RuntimeError):
    '''The test split was opened again without a stated reason.'''

class QueryCodeEncoder(Protocol):
    '''What the panel needs from an arm: queries and codes embedded in one space.'''

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        ...

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        ...

# -------------------------------------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------------------------------------

class OutcomePanel:
    '''
    The index-entry query splits, the six-digit candidates, and the log every read goes to.

    Args:
        role_rows: Every index entry with its text and role (``entry_id``, ``code``, ``text``,
            ``role``).
        candidates: The six-digit codes queries decode to, entry-less codes included.
        log: The selection log.
    '''

    def __init__(self, role_rows: pl.DataFrame, candidates: Sequence[str], log: SelectionLog):
        candidates = [str(code) for code in candidates]
        if len(set(candidates)) != len(candidates) or any(len(code) != 6 for code in candidates):
            raise ValueError('candidates must be distinct six-digit codes')
        validate_index_role_table(role_rows, candidates)
        self._rows = role_rows.select(INDEX_ROLE_COLUMNS).sort('entry_id')
        self.candidates: Tuple[str, ...] = tuple(sorted(candidates))
        self.log = log
        self.fingerprint = role_table_fingerprint(self._rows)
        self._test_open = False

    @classmethod
    def from_files(
        cls,
        index_roles_parquet: Union[str, Path],
        descriptions_parquet: Union[str, Path],
        log_path: Union[str, Path],
    ) -> 'OutcomePanel':
        '''
        The panel from ``data preprocess`` outputs: the index roles and the descriptions.

        Raises:
            ValueError: If the descriptions' examples channel is not built from the examples-role
                entries only. A descriptions file from before the roles existed holds every
                entry, held-out queries included, in its examples channel.
        '''

        roles = pl.read_parquet(index_roles_parquet)
        descriptions = pl.read_parquet(descriptions_parquet)
        verify_examples_channel(descriptions, roles)
        candidates = descriptions.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        return cls(roles, candidates.to_list(), SelectionLog(Path(log_path)))

    @property
    def entryless_candidates(self) -> Tuple[str, ...]:
        '''Candidates without index entries: decoded to, never queried (112130 and 541120).'''

        with_entries = set(self._rows.get_column('code').to_list())
        return tuple(code for code in self.candidates if code not in with_entries)

    def training_queries(self) -> pl.DataFrame:
        '''Training queries (``entry_id``, ``code``, ``text``); training data, so not logged.'''

        return self._split(IndexRole.TRAINING)

    def validation_queries(self, purpose: str) -> pl.DataFrame:
        '''Validation queries, logging the read.'''

        return self._read(IndexRole.VALIDATION, purpose)

    def open_test(self, purpose: str, *, reopen_reason: Optional[str] = None) -> None:
        '''
        Open the sealed test split for this panel object, logging the opening.

        Raises:
            SplitAlreadyOpenedError: If the log already records an opening of this split and
                no ``reopen_reason`` is given.
        '''

        prior = self.log.openings(OUTCOME_PANEL, self.fingerprint)
        reason = (reopen_reason or '').strip()
        if prior and not reason:
            first = prior[0]
            raise SplitAlreadyOpenedError(
                f'the outcome test split was opened at {first["time"]} for '
                f'{first["purpose"]!r}; opening it again needs reopen_reason'
            )
        self.log.append(
            SelectionEvent.REOPEN if prior else SelectionEvent.OPEN,
            panel=OUTCOME_PANEL,
            split=IndexRole.TEST.value,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=self._split(IndexRole.TEST).height,
            detail={'reason': reason} if reason else None,
        )
        self._test_open = True

    def test_queries(self, purpose: str) -> pl.DataFrame:
        '''
        Test queries, logging the read.

        Raises:
            SealedSplitError: If this panel object has not opened the test split.
        '''

        return self._read(IndexRole.TEST, purpose)

    def score(
        self,
        encoder: QueryCodeEncoder,
        split: Union[IndexRole, str],
        purpose: str,
        distance: Union[str, DistanceFn] = 'cosine',
    ) -> DecodingResult:
        '''Decode one split's queries over every candidate with the encoder, logging the read.'''

        name, _ = resolve_distance(distance)
        detail = {'encoder': type(encoder).__name__, 'distance': name}
        queries = self._read(IndexRole(split), purpose, detail)
        return score_decoding(
            encoder.encode_queries(queries.get_column('text').to_list()),
            queries.get_column('code').to_list(),
            encoder.encode_codes(list(self.candidates)),
            self.candidates,
            distance=distance,
            query_ids=queries.get_column('entry_id').to_list(),
        )

    def _split(self, role: IndexRole) -> pl.DataFrame:
        return self._rows.filter(pl.col('role') == role.value).select('entry_id', 'code', 'text')

    def _read(
        self,
        role: IndexRole,
        purpose: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> pl.DataFrame:
        if role not in (IndexRole.VALIDATION, IndexRole.TEST):
            raise ValueError(
                f'only validation and test splits are read as selections, not {role.value!r}'
            )
        if role is IndexRole.TEST and not self._test_open:
            raise SealedSplitError(
                'the outcome test split is sealed: call open_test(purpose) first, which is logged'
            )
        queries = self._split(role)
        self.log.append(
            SelectionEvent.READ,
            panel=OUTCOME_PANEL,
            split=role.value,
            purpose=purpose,
            fingerprint=self.fingerprint,
            n_queries=queries.height,
            detail=detail,
        )
        return queries
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_outcome_panel.py -q`
Expected: `16 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1335 passed, 1 skipped`.

- [x] **Step 6: Lint and format**

Run: `./scripts/format_code.sh --check src/naics_embedder/panels/selection_log.py src/naics_embedder/panels/outcome.py tests/unit/test_outcome_panel.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` On a failure, run the
same command without `--check`, re-run Step 5, and record the change as a deviation.

- [x] **Step 7: Commit**

```bash
git add src/naics_embedder/panels/selection_log.py src/naics_embedder/panels/outcome.py tests/unit/test_outcome_panel.py
git commit -m "feat(panels): add the selection log and the outcome panel's sealed test split"
```

### Task 5: Preprocessing applies the role table

`data preprocess` changes in three ways:

- **Examples channel.** It now holds only a code's examples-role entries, in index-file order.
  Codes without index entries keep their "Illustrative Examples:" bullets.
- **Index-roles parquet.** Every entry is written with its text and role to
  `data/naics_index_roles.parquet`.
- **Checks before writing.** One known role per entry, the examples channel built from the
  roles, and no held-out query matching training text.

Three supporting changes:

- The index file must match its pinned sha256, because the table's `entry_id`s are its row
  positions.
- `--source-dir` reads local copies instead of downloading.
- The overwrite guard refuses to rewrite a descriptions file that a configured bundle pins.
  `--force` overrides it.

The builder is split into `load_naics_sources`, `naics_index_entries` and `build_descriptions` so
that Task 7 can build the static training text before any role exists.

A new fixture, `naics_sources`, stands in for the four Census files: nine codes under sector 11
and eleven index-sheet rows. In it:

- one "see" row (`******`)
- one entry padded with whitespace
- one entry equal to a title (`Oilseed (except soybean) farming`), which is therefore never held
  out
- a fallback-examples code (`11119`) and an entry-less six-digit code (`111191`)

**Files:**

- Create: `tests/fixtures/naics_sources.py`
- Modify: `tests/conftest.py` (register the fixture module)
- Modify: `tests/unit/test_data_download.py`
- Modify: `tests/unit/test_config.py`, specifically `TestDownloadConfig`
- Modify: `tests/unit/test_cli_commands.py`, specifically the `data preprocess` tests
- Modify: `src/naics_embedder/utils/config.py`, specifically `DownloadConfig`
- Modify: `conf/data/download.yaml`
- Modify: `src/naics_embedder/data/download_data.py`
- Modify: `src/naics_embedder/cli/commands/data.py`, specifically `preprocess`

**Interfaces:**

- Consumes (Task 2): `IndexRole`, `validate_index_role_table`, `read_role_table`,
  `attach_role_text`, `verify_examples_channel` and `verify_role_leakage`.
- Produces in `DownloadConfig`:
  - `index_roles_parquet: str = './data/naics_index_roles.parquet'`. It must end in `.parquet`,
    like `output_parquet`, and the validator message is now `output paths must point to a
    .parquet file`.
  - `index_roles_csv: str = './conf/data/index_roles.csv'`
  - `index_sha256: str`, which defaults to the pinned
    `6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63` and must match
    `^[0-9a-f]{64}$`.
  - `source_dir: Optional[str] = None`
- Produces in `naics_embedder.data.download_data`:
  - `NaicsSources(titles, descriptions, index, exclusions)`, a frozen dataclass of polars
    frames.
  - `load_naics_sources(cfg: DownloadConfig) -> NaicsSources`
  - `naics_index_entries(sources: NaicsSources) -> pl.DataFrame`, with columns `entry_id`
    (Int64, the sheet row), `code` and `text` (stripped).
  - `build_descriptions(sources: NaicsSources, examples_entries: pl.DataFrame) -> pl.DataFrame`.
    Its columns are `index`, `level`, `code`, `title`, `description`, `examples`, `excluded`
    and `excluded_codes`. Passing `entries.clear()` builds the text without any index-derived
    examples.
  - `PINNING_CONFIGS`, which pins `conf/config.yaml` at `supervision.manifest_path` and
    `conf/graph.yaml` at `supervision_manifest_path`.
  - `pinned_description_fingerprints(pinning_configs=PINNING_CONFIGS) -> Dict[str, str]`
  - `refuse_pinned_overwrite(output: Path, *, force: bool, pinning_configs=PINNING_CONFIGS) -> None`,
    which raises `FileExistsError`.
  - `download_preprocess_data(cfg: Optional[DownloadConfig] = None, *, force: bool = False) -> pl.DataFrame`.
    It raises `FileExistsError` from the guard, `FileNotFoundError` when `cfg.index_roles_csv`
    is missing, and `ValueError` when a check fails. It writes no output file until every
    check passes.
- Produces in `naics_embedder.cli.commands.data`:
  - `DOWNLOAD_CONFIG = 'data/download.yaml'`
  - `SourceDirOption`
  - `_download_config(source_dir: Optional[str]) -> DownloadConfig`
  - `naics-embedder data preprocess [--source-dir DIR] [--force]`, which exits 1 on a refused
    overwrite.

- [x] **Step 1: Add the source fixture**

Create `tests/fixtures/naics_sources.py` with exactly this content:

```python
'''
A miniature set of the four Census NAICS source files, as ``load_naics_sources`` returns them.

Nine codes under one sector. 111110 and 111120 have index entries; 111191 has none, so it is a
decoding candidate only; 11119 has no entries either and falls back to its description's
illustrative examples. One entry of 111120 repeats its code's title, so it can never be held out,
and one sheet row is a "see" cross-reference row, which is not an entry.
'''

import polars as pl
import pytest

from naics_embedder.data.download_data import NaicsSources

TITLES = [
    ('11', 'Agriculture, Forestry, Fishing and Hunting'),
    ('111', 'Crop Production'),
    ('1111', 'Oilseed and Grain Farming'),
    ('11111', 'Soybean Farming'),
    ('111110', 'Soybean Farming'),
    ('11112', 'Oilseed (except Soybean) Farming'),
    ('111120', 'Oilseed (except Soybean) Farming'),
    ('11119', 'Other Grain Farming'),
    ('111191', 'Oilseed and Grain Combination Farming'),
]
DESCRIPTIONS = {
    '11': 'The Sector as a Whole\nThe Agriculture sector comprises farms and ranches.',
    '111': 'This subsector comprises establishments growing crops.',
    '1111': 'This industry group comprises establishments growing oilseeds and grains.',
    '11111': 'This industry comprises establishments growing soybeans.',
    '111110': 'This industry comprises establishments primarily engaged in growing soybeans.',
    '11112': 'This industry comprises establishments growing fibrous oilseed plants.',
    '111120': 'This industry comprises establishments growing oilseed plants except soybeans.',
    '11119': (
        'This industry comprises establishments growing grains not elsewhere classified.\n'
        'Illustrative Examples:\nBarley farming\nRye farming'
    ),
    '111191': 'This industry comprises establishments growing a combination of oilseeds and grains.',
}
# Sheet order is entry order: entry_id is the row position
INDEX_ROWS = [
    ('111110', 'Soybean farming, field and seed production'),
    ('111110', 'Edamame farming'),
    ('111110', 'Soybeans, organic'),
    ('111110', '  Soybean seed production '),
    ('111120', 'Canola farming'),
    ('111120', 'Flaxseed farming'),
    ('111120', 'Sunflower farming'),
    ('111120', 'Safflower farming'),
    ('******', 'Grain farming--see Industry Group 1111'),
    ('111120', 'Rapeseed farming'),
    ('111120', 'Oilseed (except soybean) farming'),
]
EXCLUSIONS = [
    (
        '111110',
        'Growing soybeans for green manure--are classified in Industry 111120, Oilseed (except '
        'Soybean) Farming.',
    ),
]

@pytest.fixture
def naics_sources() -> NaicsSources:
    return NaicsSources(
        titles=pl.DataFrame(
            {
                'index': [number for number, _ in enumerate(TITLES, start=1)],
                'code': [code for code, _ in TITLES],
                'title': [title for _, title in TITLES],
            },
            schema={
                'index': pl.UInt32,
                'code': pl.Utf8,
                'title': pl.Utf8
            },
        ),
        descriptions=pl.DataFrame(
            {
                'code': list(DESCRIPTIONS),
                'description': list(DESCRIPTIONS.values()),
            }
        ),
        index=pl.DataFrame(
            {
                'code': [code for code, _ in INDEX_ROWS],
                'examples': [text for _, text in INDEX_ROWS],
            }
        ),
        exclusions=pl.DataFrame(
            {
                'code': [code for code, _ in EXCLUSIONS],
                'excluded': [text for _, text in EXCLUSIONS],
            }
        ),
    )
```

Modify `tests/conftest.py` with one edit. The replaced text occurs exactly once in the file.

**`tests/conftest.py`, edit 1 of 1.** Replace:

```python
import torch

pytest_plugins = ('tests.fixtures.supervision', )

# -------------------------------------------------------------------------------------------------
```

with:

```python
import torch

pytest_plugins = ('tests.fixtures.naics_sources', 'tests.fixtures.supervision')

# -------------------------------------------------------------------------------------------------
```

- [x] **Step 2: Write the failing tests**

Modify `tests/unit/test_data_download.py` with these 7 edits, in order. Each replaced text occurs
exactly once in the file.

**`tests/unit/test_data_download.py`, edit 1 of 7.** Replace:

```python
from io import BytesIO
from typing import Dict, List, cast

import polars as pl
```

with:

```python
import hashlib
import json
from io import BytesIO
from typing import Dict, List, Set, cast

import polars as pl
```

**`tests/unit/test_data_download.py`, edit 2 of 7.** Replace:

```python
from naics_embedder.data import download_data

@pytest.mark.unit
```

with:

```python
from naics_embedder.data import download_data
from naics_embedder.utils.config import DownloadConfig
from tests.fixtures.naics_sources import TITLES

ENTRY_SCHEMA = {'entry_id': pl.Int64, 'code': pl.Utf8, 'text': pl.Utf8}

@pytest.mark.unit
```

**`tests/unit/test_data_download.py`, edit 3 of 7.** Replace:

```python
@pytest.mark.unit
def test_get_examples_prefers_spreadsheet_entries():
    examples_df = pl.DataFrame({'code': ['111'], 'examples': ['Sheet example']})
    codes = {'111'}
    descriptions_2 = pl.DataFrame(
        {
```

with:

```python
@pytest.mark.unit
def test_get_examples_joins_examples_role_entries_in_index_order():
    examples_entries = pl.DataFrame(
        [(5, '111', 'Second entry'), (2, '111', 'First entry')], schema=ENTRY_SCHEMA, orient='row'
    )
    descriptions_2 = pl.DataFrame(
        {
```

**`tests/unit/test_data_download.py`, edit 4 of 7.** Replace:

```python
    examples, descriptions_examples = download_data._get_examples(
        examples_df, codes, descriptions_2, descriptions_3
    )

    assert examples.height == 1
    row = examples.row(0, named=True)
    assert row['examples'] == 'Sheet example'

    # The examples section, and so the description cutoff, starts at the marker itself
    assert descriptions_examples.height == 1
    assert descriptions_examples.row(0, named=True)['description_id_min'] == 2

@pytest.mark.unit
```

with:

```python
    examples, descriptions_examples = download_data._get_examples(
        {'111'}, examples_entries, descriptions_2, descriptions_3
    )

    assert examples.rows() == [('111', 'First entry; Second entry')]
    # The examples section, and so the description cutoff, starts at the marker itself
    assert descriptions_examples.height == 1
    assert descriptions_examples.row(0, named=True)['description_id_min'] == 2

@pytest.mark.unit
def test_get_examples_rejects_entries_of_codes_without_index_entries():
    examples_entries = pl.DataFrame([(0, '222', 'Stray entry')], schema=ENTRY_SCHEMA, orient='row')
    descriptions = pl.DataFrame({'code': ['111'], 'description_id': [1], 'description': ['Intro']})

    with pytest.raises(ValueError, match='without index entries'):
        download_data._get_examples({'111'}, examples_entries, descriptions, descriptions)

@pytest.mark.unit
```

**`tests/unit/test_data_download.py`, edit 5 of 7.** Replace:

```python
@pytest.mark.unit
@pytest.mark.parametrize(
    'sheet_examples, expected_examples',
    [
        pytest.param(['Grain farming, mixed'], 'Grain farming, mixed', id='index-sheet'),
        pytest.param([], 'Barley farming; Rye farming', id='description-text'),
    ],
)
def test_description_drops_whole_illustrative_examples_section(
    sheet_examples: List[str], expected_examples: str
):
    # A real description split one block per line, the structure _get_examples relies on. The
    # marker and its bullets leave the description whichever source fills the examples column: the
    # Index sheet (preferred) or the bullets themselves.
    descriptions_3 = pl.DataFrame(
        {
```

with:

```python
@pytest.mark.unit
@pytest.mark.parametrize(
    'index_codes, entry_texts, expected_examples',
    [
        pytest.param({'111199'}, ['Grain farming, mixed'], ['Grain farming, mixed'], id='index'),
        pytest.param(set(), [], ['Barley farming; Rye farming'], id='description-text'),
        # Every entry of the code is a query: its examples channel stays empty, never the bullets
        pytest.param({'111199'}, [], [], id='index-without-examples-role'),
    ],
)
def test_description_drops_whole_illustrative_examples_section(
    index_codes: Set[str], entry_texts: List[str], expected_examples: List[str]
):
    # A real description split one block per line, the structure _get_examples relies on. The
    # marker and its bullets leave the description whichever source fills the examples column:
    # the examples-role index entries of a code with entries, or else the bullets themselves.
    descriptions_3 = pl.DataFrame(
        {
```

**`tests/unit/test_data_download.py`, edit 6 of 7.** Replace:

```python
        }
    )
    examples_df = pl.DataFrame(
        {
            'code': ['111199'] * len(sheet_examples),
            'examples': sheet_examples
        },
        schema={
            'code': pl.Utf8,
            'examples': pl.Utf8
        },
    )
    descriptions_exclusions = pl.DataFrame(schema={'code': pl.Utf8, 'description_id': pl.UInt32})

    examples, descriptions_examples = download_data._get_examples(
        examples_df, {'111199'}, descriptions_3, descriptions_3
    )
    descriptions = download_data._get_descriptions_2(
```

with:

```python
        }
    )
    examples_entries = pl.DataFrame(
        [(entry_id, '111199', text) for entry_id, text in enumerate(entry_texts)],
        schema=ENTRY_SCHEMA,
        orient='row',
    )
    descriptions_exclusions = pl.DataFrame(schema={'code': pl.Utf8, 'description_id': pl.UInt32})

    examples, descriptions_examples = download_data._get_examples(
        index_codes, examples_entries, descriptions_3, descriptions_3
    )
    descriptions = download_data._get_descriptions_2(
```

**`tests/unit/test_data_download.py`, edit 7 of 7.** Replace:

```python
        'This industry comprises establishments growing grain.'
    ]
    assert examples.get_column('examples').to_list() == [expected_examples]
```

with:

```python
        'This industry comprises establishments growing grain.'
    ]
    assert examples.get_column('examples').to_list() == expected_examples

# -------------------------------------------------------------------------------------------------
# Local sources and the pinned index file
# -------------------------------------------------------------------------------------------------

CODES_URL = 'https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx'

@pytest.fixture
def captured_bytes(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_read_xlsx_bytes(data, sheet, schema, cols):
        captured['data'] = data
        return pl.DataFrame({'code': ['111110']})

    def no_download(*_args, **_kwargs):
        raise AssertionError('a local source must not be downloaded')

    monkeypatch.setattr(download_data, '_read_xlsx_bytes', fake_read_xlsx_bytes)
    monkeypatch.setattr(download_data, '_download_with_retry', no_download)
    return captured

@pytest.mark.unit
def test_read_xlsx_reads_the_local_copy_named_by_the_url(tmp_path, captured_bytes):
    (tmp_path / '2-6 digit_2022_Codes.xlsx').write_bytes(b'workbook')

    download_data._read_xlsx(
        CODES_URL,
        'Sheet1',
        {},
        {},
        source_dir=str(tmp_path),
        expected_sha256=hashlib.sha256(b'workbook').hexdigest(),
    )

    assert captured_bytes['data'] == b'workbook'

@pytest.mark.unit
def test_read_xlsx_rejects_a_file_other_than_the_pinned_one(tmp_path, captured_bytes):
    (tmp_path / '2-6 digit_2022_Codes.xlsx').write_bytes(b'another workbook')

    with pytest.raises(ValueError, match='pinned'):
        download_data._read_xlsx(
            CODES_URL,
            'Sheet1',
            {},
            {},
            source_dir=str(tmp_path),
            expected_sha256=hashlib.sha256(b'workbook').hexdigest(),
        )
    assert captured_bytes == {}

@pytest.mark.unit
def test_read_xlsx_fails_when_the_local_copy_is_missing(tmp_path, captured_bytes):
    with pytest.raises(FileNotFoundError, match='no local copy'):
        download_data._read_xlsx(CODES_URL, 'Sheet1', {}, {}, source_dir=str(tmp_path))

# -------------------------------------------------------------------------------------------------
# Index entries and the examples channel
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_index_entries_are_sheet_rows_of_six_digit_codes(naics_sources):
    entries = download_data.naics_index_entries(naics_sources)

    # Sheet row 8 is a "see" row (code ******); entry 3 had surrounding whitespace
    assert entries.get_column('entry_id').to_list() == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    assert entries.row(3) == (3, '111110', 'Soybean seed production')
    assert entries.schema == pl.Schema(ENTRY_SCHEMA)

@pytest.mark.unit
def test_build_descriptions_keeps_queries_out_of_the_examples_channel(naics_sources):
    entries = download_data.naics_index_entries(naics_sources)
    examples_entries = entries.filter(pl.col('entry_id').is_in([2, 0, 9]))

    descriptions = download_data.build_descriptions(naics_sources, examples_entries)
    examples = dict(descriptions.select('code', 'examples').iter_rows())

    assert descriptions.get_column('code').to_list() == [code for code, _ in TITLES]
    assert examples['111110'] == 'Soybean farming, field and seed production; Soybeans, organic'
    assert examples['111120'] == 'Rapeseed farming'
    assert examples['11119'] == 'Barley farming; Rye farming'
    assert examples['111191'] is None
    assert 'Illustrative' not in descriptions.filter(pl.col('code') == '11119')['description'][0]

# -------------------------------------------------------------------------------------------------
# The descriptions file a supervision bundle pins
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def pinned_output(tmp_path):
    output = tmp_path / 'naics_descriptions.parquet'
    output.write_bytes(b'descriptions')
    manifest = tmp_path / 'bundle' / 'manifest.json'
    manifest.parent.mkdir()
    fingerprint = hashlib.sha256(b'descriptions').hexdigest()
    manifest.write_text(json.dumps({'description_fingerprint': fingerprint}))
    config = tmp_path / 'config.yaml'
    config.write_text(f'supervision:\n  manifest_path: {manifest}\n')
    graph = tmp_path / 'graph.yaml'
    graph.write_text(f'supervision_manifest_path: {manifest}\n')
    return output, config, graph

@pytest.mark.unit
@pytest.mark.parametrize('pinned_by', ['config', 'graph'])
def test_refuses_to_overwrite_the_pinned_descriptions(pinned_output, pinned_by):
    output, config, graph = pinned_output
    pins = {
        'config': ((config, ('supervision', 'manifest_path')), ),
        'graph': ((graph, ('supervision_manifest_path', )), ),
    }[pinned_by]

    with pytest.raises(FileExistsError, match='pins'):
        download_data.refuse_pinned_overwrite(output, force=False, pinning_configs=pins)
    download_data.refuse_pinned_overwrite(output, force=True, pinning_configs=pins)

@pytest.mark.unit
def test_other_descriptions_files_are_not_pinned(pinned_output, tmp_path):
    output, config, _ = pinned_output
    pins = ((config, ('supervision', 'manifest_path')), )
    output.write_bytes(b'rebuilt descriptions')

    download_data.refuse_pinned_overwrite(output, force=False, pinning_configs=pins)
    download_data.refuse_pinned_overwrite(
        tmp_path / 'absent.parquet', force=False, pinning_configs=pins
    )

@pytest.mark.unit
def test_unset_or_missing_manifests_pin_nothing(tmp_path, pinned_output):
    output, _, _ = pinned_output
    unset = tmp_path / 'unset.yaml'
    unset.write_text('supervision:\n  manifest_path: null\n')
    missing = tmp_path / 'missing.yaml'
    missing.write_text('supervision_manifest_path: nowhere/manifest.json\n')
    pins = (
        (unset, ('supervision', 'manifest_path')),
        (missing, ('supervision_manifest_path', )),
        (tmp_path / 'no_such_config.yaml', ('supervision', 'manifest_path')),
    )

    assert download_data.pinned_description_fingerprints(pins) == {}
    download_data.refuse_pinned_overwrite(output, force=False, pinning_configs=pins)

# -------------------------------------------------------------------------------------------------
# Preprocessing with the frozen role table
# -------------------------------------------------------------------------------------------------

ROLE_TABLE = [
    (0, '111110', 'examples'),
    (1, '111110', 'validation'),
    (2, '111110', 'training'),
    (3, '111110', 'test'),
    (4, '111120', 'examples'),
    (5, '111120', 'examples'),
    (6, '111120', 'validation'),
    (7, '111120', 'training'),
    (9, '111120', 'test'),
    (10, '111120', 'training'),
]

@pytest.fixture
def preprocess_cfg(tmp_path, monkeypatch, naics_sources):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(download_data, 'load_naics_sources', lambda cfg: naics_sources)
    roles_csv = tmp_path / 'conf' / 'index_roles.csv'
    roles_csv.parent.mkdir()
    roles_csv.write_text(
        'entry_id,code,role\n' + ''.join(f'{i},{c},{r}\n' for i, c, r in ROLE_TABLE)
    )
    return DownloadConfig(
        output_parquet=str(tmp_path / 'data' / 'naics_descriptions.parquet'),
        index_roles_parquet=str(tmp_path / 'data' / 'naics_index_roles.parquet'),
        index_roles_csv=str(roles_csv),
    )

@pytest.mark.unit
def test_preprocess_builds_examples_from_the_role_table(preprocess_cfg):
    descriptions = download_data.download_preprocess_data(preprocess_cfg)

    written = pl.read_parquet(preprocess_cfg.output_parquet)
    roles = pl.read_parquet(preprocess_cfg.index_roles_parquet)
    examples = dict(written.select('code', 'examples').iter_rows())
    assert written.equals(descriptions)
    assert examples['111110'] == 'Soybean farming, field and seed production'
    assert examples['111120'] == 'Canola farming; Flaxseed farming'
    assert roles.columns == ['entry_id', 'code', 'text', 'role']
    assert roles.get_column('entry_id').to_list() == [i for i, _, _ in ROLE_TABLE]
    assert roles.row(3) == (3, '111110', 'Soybean seed production', 'test')

@pytest.mark.unit
def test_preprocess_refuses_a_held_out_query_that_matches_training_text(preprocess_cfg):
    # Entry 10 repeats 111120's title; the table makes it a test query
    leaky = [(i, c, 'test' if i == 10 else ('training' if i == 9 else r)) for i, c, r in ROLE_TABLE]
    with open(preprocess_cfg.index_roles_csv, 'w') as handle:
        handle.write('entry_id,code,role\n' + ''.join(f'{i},{c},{r}\n' for i, c, r in leaky))

    with pytest.raises(ValueError, match='held-out queries match training text'):
        download_data.download_preprocess_data(preprocess_cfg)

@pytest.mark.unit
def test_preprocess_needs_the_role_table(preprocess_cfg, tmp_path):
    missing = preprocess_cfg.model_copy(update={'index_roles_csv': str(tmp_path / 'none.csv')})

    with pytest.raises(FileNotFoundError, match='data roles'):
        download_data.download_preprocess_data(missing)
```

Modify `tests/unit/test_config.py` with one edit. The replaced text occurs exactly once in the file.

**`tests/unit/test_config.py`, edit 1 of 1.** Replace:

```python
        with pytest.raises(ValidationError):
            DownloadConfig(output_parquet='./data/output.csv')

# -------------------------------------------------------------------------------------------------
```

with:

```python
        with pytest.raises(ValidationError):
            DownloadConfig(output_parquet='./data/output.csv')
        with pytest.raises(ValidationError):
            DownloadConfig(index_roles_parquet='./data/roles.csv')

    def test_yaml_matches_defaults(self):
        '''The shipped YAML pins the index file and names the committed role table.'''

        cfg = load_config(DownloadConfig, 'data/download.yaml')

        assert cfg == DownloadConfig()
        assert cfg.index_sha256 == (
            '6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63'
        )
        assert cfg.index_roles_csv == './conf/data/index_roles.csv'
        assert cfg.source_dir is None

    def test_index_sha256_must_be_a_hex_digest(self):
        with pytest.raises(ValidationError):
            DownloadConfig(index_sha256='not-a-digest')

# -------------------------------------------------------------------------------------------------
```

Modify `tests/unit/test_cli_commands.py` with these 2 edits, in order. Each replaced text occurs
exactly once in the file.

**`tests/unit/test_cli_commands.py`, edit 1 of 2.** Replace:

```python
def test_data_preprocess_invokes_download(monkeypatch, runner):
    called = {}
    monkeypatch.setattr(
        data_cli, 'download_preprocess_data', lambda: called.setdefault('preprocess', True)
    )
```

with:

```python
def test_data_preprocess_invokes_download(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        data_cli, 'download_preprocess_data', lambda cfg, force: calls.append((cfg, force))
    )
```

**`tests/unit/test_cli_commands.py`, edit 2 of 2.** Replace:

```python
    assert result.exit_code == 0
    assert called['preprocess']

def test_data_all_runs_preprocess_then_one_supervision_build(monkeypatch, runner, tmp_path):
    order = []
    manifest = tmp_path / 'bundle' / 'manifest.json'
    monkeypatch.setattr(data_cli, 'download_preprocess_data', lambda: order.append('preprocess'))

    def fake_generate(cfg):
```

with:

```python
    assert result.exit_code == 0
    [(cfg, force)] = calls
    assert cfg.source_dir is None
    assert cfg.index_roles_csv == './conf/data/index_roles.csv'
    assert force is False

def test_data_preprocess_passes_source_dir_and_force(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(
        data_cli, 'download_preprocess_data', lambda cfg, force: calls.append((cfg, force))
    )

    result = runner.invoke(data_cli.app, ['preprocess', '--source-dir', '/sources', '--force'])

    assert result.exit_code == 0
    assert [(cfg.source_dir, force) for cfg, force in calls] == [('/sources', True)]

def test_data_preprocess_reports_a_refused_overwrite(monkeypatch, runner):

    def refuse(cfg, force):
        raise FileExistsError('pinned; pass --force to overwrite it')

    monkeypatch.setattr(data_cli, 'download_preprocess_data', refuse)

    result = runner.invoke(data_cli.app, ['preprocess'])

    assert result.exit_code == 1
    assert '--force' in result.output

def test_data_all_runs_preprocess_then_one_supervision_build(monkeypatch, runner, tmp_path):
    order = []
    manifest = tmp_path / 'bundle' / 'manifest.json'
    monkeypatch.setattr(
        data_cli, 'download_preprocess_data', lambda cfg, force: order.append('preprocess')
    )

    def fake_generate(cfg):
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_data_download.py tests/unit/test_config.py tests/unit/test_cli_commands.py -q`
Expected: pytest cannot start. It fails with `ImportError: Error importing plugin
"tests.fixtures.naics_sources": cannot import name 'NaicsSources' from
'naics_embedder.data.download_data'`. The conftest registers the fixture module for the whole
suite, so every test waits on Step 5.

- [x] **Step 4: Add the configuration**

Modify `src/naics_embedder/utils/config.py` with these 2 edits, in order. Each replaced text occurs
exactly once in the file.

**`src/naics_embedder/utils/config.py`, edit 1 of 2.** Replace:

```python
        description='Output path for processed descriptions',
    )

    # URLs for data sources
```

with:

```python
        description='Output path for processed descriptions',
    )
    index_roles_parquet: str = Field(
        default='./data/naics_index_roles.parquet',
        description='Output path for every index entry with its text and role',
    )
    index_roles_csv: str = Field(
        default='./conf/data/index_roles.csv',
        description='The frozen index-entry role table (entry_id, code, role) to apply',
    )
    index_sha256: str = Field(
        default='6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63',
        pattern=r'^[0-9a-f]{64}$',
        description='SHA-256 of the index file whose row positions the role table is keyed to',
    )
    source_dir: Optional[str] = Field(
        default=None,
        description='Read the source files from this directory, by URL file name, not the web',
    )

    # URLs for data sources
```

**`src/naics_embedder/utils/config.py`, edit 2 of 2.** Replace:

```python
    )

    @field_validator('output_parquet')
    @classmethod
    def validate_output_parquet(cls, value: str) -> str:
        path = Path(value)
        if path.suffix.lower() != '.parquet':
            raise ValueError('output_parquet must point to a .parquet file')
        return value
```

with:

```python
    )

    @field_validator('output_parquet', 'index_roles_parquet')
    @classmethod
    def validate_output_parquet(cls, value: str) -> str:
        path = Path(value)
        if path.suffix.lower() != '.parquet':
            raise ValueError('output paths must point to a .parquet file')
        return value
```

Modify `conf/data/download.yaml` with one edit. The replaced text occurs exactly once in the file.

**`conf/data/download.yaml`, edit 1 of 1.** Replace:

```yaml
output_parquet: ./data/naics_descriptions.parquet

# URLs for data sources
```

with:

```yaml
output_parquet: ./data/naics_descriptions.parquet
index_roles_parquet: ./data/naics_index_roles.parquet

# The frozen index-entry role table (data roles writes it once) and the index file it is keyed to
index_roles_csv: ./conf/data/index_roles.csv
index_sha256: 6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63

# Read the four files from this directory instead of downloading them (--source-dir)
source_dir: null

# URLs for data sources
```

- [x] **Step 5: Rebuild preprocessing around the role table**

Modify `src/naics_embedder/data/download_data.py` with these 13 edits, in order. Each replaced text
occurs exactly once in the file.

**`src/naics_embedder/data/download_data.py`, edit 1 of 13.** Replace:

```python
# -------------------------------------------------------------------------------------------------

import logging
from io import BytesIO
from typing import Dict, Optional, Set, Tuple

import polars as pl

from naics_embedder.utils.config import DownloadConfig, load_config
from naics_embedder.utils.utilities import download_with_retry as _download_with_retry
```

with:

```python
# -------------------------------------------------------------------------------------------------

import hashlib
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

import polars as pl
import yaml

from naics_embedder.panels.index_roles import (
    attach_role_text,
    read_role_table,
    verify_examples_channel,
    verify_role_leakage,
)
from naics_embedder.supervision.artifacts import sha256_file, validate_index_role_table
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.utils.config import DownloadConfig, load_config
from naics_embedder.utils.utilities import download_with_retry as _download_with_retry
```

**`src/naics_embedder/data/download_data.py`, edit 2 of 13.** Replace:

```python
    ).rename(mapping=cols)

def _read_xlsx(
    url: str,
```

with:

```python
    ).rename(mapping=cols)

def _local_source(url: str, source_dir: str) -> Path:
    '''The local copy of a source file: the URL's file name inside ``source_dir``.'''

    return Path(source_dir).expanduser() / unquote(PurePosixPath(urlparse(url).path).name)

def _read_xlsx(
    url: str,
```

**`src/naics_embedder/data/download_data.py`, edit 3 of 13.** Replace:

```python
    backoff_factor: float = 2.0,
    timeout: float = 30.0,
) -> Optional[pl.DataFrame]:
    '''Download and read Excel file from URL.'''

    data = _download_with_retry(url, max_retries, initial_delay, backoff_factor, timeout)

    if data is None:
        return None

    return _read_xlsx_bytes(data, sheet, schema, cols)
```

with:

```python
    backoff_factor: float = 2.0,
    timeout: float = 30.0,
    source_dir: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> Optional[pl.DataFrame]:
    '''
    Read an Excel file from its URL, or from its local copy in ``source_dir``.

    Raises:
        FileNotFoundError: If ``source_dir`` is set and holds no copy of the file.
        ValueError: If ``expected_sha256`` is set and the file's bytes do not match it.
    '''

    if source_dir is None:
        data = _download_with_retry(url, max_retries, initial_delay, backoff_factor, timeout)
    else:
        path = _local_source(url, source_dir)
        if not path.is_file():
            raise FileNotFoundError(f'no local copy of {url} at {path}')
        data = path.read_bytes()

    if data is None:
        return None

    if expected_sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f'{url} has sha256 {digest}, not the pinned {expected_sha256}: the index-entry '
                'role table is keyed to row positions in the pinned file'
            )

    return _read_xlsx_bytes(data, sheet, schema, cols)
```

**`src/naics_embedder/data/download_data.py`, edit 4 of 13.** Replace:

```python
    # NAICS titles
    titles_df = _read_xlsx(
        url=cfg.url_codes, sheet=cfg.sheet_codes, schema=schema_codes, cols=cfg.rename_codes
    )
```

with:

```python
    # NAICS titles
    titles_df = _read_xlsx(
        url=cfg.url_codes,
        sheet=cfg.sheet_codes,
        schema=schema_codes,
        cols=cfg.rename_codes,
        source_dir=cfg.source_dir,
    )
```

**`src/naics_embedder/data/download_data.py`, edit 5 of 13.** Replace:

```python
        schema=schema_descriptions,
        cols=cfg.rename_descriptions,
    )

    # NAICS index file for examples
    examples_df = _read_xlsx(
        url=cfg.url_index, sheet=cfg.sheet_index, schema=schema_index, cols=cfg.rename_index
    )
```

with:

```python
        schema=schema_descriptions,
        cols=cfg.rename_descriptions,
        source_dir=cfg.source_dir,
    )

    # NAICS index file for examples, pinned: entry IDs are its row positions
    examples_df = _read_xlsx(
        url=cfg.url_index,
        sheet=cfg.sheet_index,
        schema=schema_index,
        cols=cfg.rename_index,
        source_dir=cfg.source_dir,
        expected_sha256=cfg.index_sha256,
    )
```

**`src/naics_embedder/data/download_data.py`, edit 6 of 13.** Replace:

```python
        schema=schema_exclusions,
        cols=cfg.rename_exclusions,
    )
```

with:

```python
        schema=schema_exclusions,
        cols=cfg.rename_exclusions,
        source_dir=cfg.source_dir,
    )
```

**`src/naics_embedder/data/download_data.py`, edit 7 of 13.** Replace:

```python
# -------------------------------------------------------------------------------------------------

def _get_examples(
    examples_df: pl.DataFrame,
    codes: Set[str],
    descriptions_2: pl.DataFrame,
    descriptions_3: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:

    # Example spreadsheet
    # yapf: disable
    examples_1 = (
        examples_df
        .filter(
            pl.col('code').is_in(codes)
        )
        .sort('code')
        .group_by('code', maintain_order=True)
        .agg(
            examples_1=pl.col('examples')
        )
    )
```

with:

```python
# -------------------------------------------------------------------------------------------------

def _get_index_entries(index_df: pl.DataFrame, codes: Set[str]) -> pl.DataFrame:
    '''
    The index file's entries for six-digit codes (``entry_id``, ``code``, ``text``).

    ``entry_id`` is the row's 0-based position in the index sheet, stable for the pinned file.
    Rows naming no six-digit code (the "see" cross-reference rows, coded ``******``) are dropped,
    and entry text is stripped of surrounding whitespace.
    '''

    six_digit = sorted(code for code in codes if len(code) == 6)
    # yapf: disable
    return (
        index_df
        .with_row_index('entry_id')
        .filter(pl.col('code').is_in(six_digit))
        .select(
            entry_id=pl.col('entry_id').cast(pl.Int64),
            code=pl.col('code'),
            text=pl.col('examples').str.strip_chars(),
        )
    )
    # yapf: enable

def _get_examples(
    index_codes: Set[str],
    examples_entries: pl.DataFrame,
    descriptions_2: pl.DataFrame,
    descriptions_3: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    '''
    Each code's examples channel, and where each description's examples section starts.

    A code with index entries takes its examples-role entries only (``examples_entries``, in
    index-file order): its other entries are queries and stay out of the channel (Req 3). A code
    without index entries falls back to the bullets after its description's "Illustrative
    Examples:" marker. Either way the marker and its bullets leave the description.
    '''

    outside = sorted(set(examples_entries.get_column('code').to_list()) - index_codes)
    if outside:
        raise ValueError(f'examples entries name codes without index entries: {outside[:5]}')

    # Examples-role index entries, in index-file order
    # yapf: disable
    examples_1 = (
        examples_entries
        .sort('entry_id')
        .group_by('code', maintain_order=True)
        .agg(
            examples_1=pl.col('text')
        )
    )
```

**`src/naics_embedder/data/download_data.py`, edit 8 of 13.** Replace:

```python
    descriptions_examples = examples_3.select('code', 'description_id_min')

    # Merge examples, preferring spreadsheet example
    examples_4 = examples_1.join(examples_3, how='full', on='code', coalesce=True).select(
        code=pl.col('code'), examples=pl.coalesce('examples_1', 'examples_2')
    )
```

with:

```python
    descriptions_examples = examples_3.select('code', 'description_id_min')

    # Codes without index entries fall back to their description's illustrative examples
    fallback = examples_3.filter(~pl.col('code').is_in(sorted(index_codes)))
    examples_4 = examples_1.join(fallback, how='full', on='code', coalesce=True).select(
        code=pl.col('code'), examples=pl.coalesce('examples_1', 'examples_2')
    )
```

**`src/naics_embedder/data/download_data.py`, edit 9 of 13.** Replace:

```python
    logger.info('Examples:')
    logger.info('  Reference codes:')
    logger.info(f'    Cross-references: {examples_1.height: ,}')
    logger.info(f'    Extracted from descriptions: {examples_3.height: ,}')
    logger.info(f'    Final: {examples.height: ,}')
    logger.info(f'  Number of examples: {examples_cnt: ,}\n')
```

with:

```python
    logger.info('Examples:')
    logger.info('  Reference codes:')
    logger.info(f'    Index entries (examples role): {examples_1.height: ,}')
    logger.info(f'    Extracted from descriptions: {fallback.height: ,}')
    logger.info(f'    Final: {examples.height: ,}')
    logger.info(f'  Number of examples: {examples_cnt: ,}\n')
```

**`src/naics_embedder/data/download_data.py`, edit 10 of 13.** Replace:

```python
# -------------------------------------------------------------------------------------------------
# Combine all and write final output
# -------------------------------------------------------------------------------------------------

def download_preprocess_data() -> pl.DataFrame:
    # Create directories
    make_directories()

    # Load configuration from YAML
    cfg = load_config(DownloadConfig, './data/download.yaml')

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')

    titles_df, descriptions_df, examples_df, exclusions_df = _download_files(cfg)

    titles, codes = _get_titles(titles_df)

    descriptions_2, descriptions_3 = _get_descriptions_1(descriptions_df)

    exclusions, descriptions_exclusions = _get_exclusions(exclusions_df, descriptions_3, codes)

    examples, descriptions_examples = _get_examples(
        examples_df, codes, descriptions_2, descriptions_3
    )
```

with:

```python
# -------------------------------------------------------------------------------------------------
# Sources and the combined descriptions
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NaicsSources:
    '''The four Census NAICS files as read, combined sector codes normalized.'''

    titles: pl.DataFrame
    descriptions: pl.DataFrame
    index: pl.DataFrame
    exclusions: pl.DataFrame

def load_naics_sources(cfg: DownloadConfig) -> NaicsSources:
    '''Read the four files, from ``cfg.source_dir`` when set; the index file must match its pin.'''

    return NaicsSources(*_download_files(cfg))

def naics_index_entries(sources: NaicsSources) -> pl.DataFrame:
    '''The index file's entries for six-digit codes (``entry_id``, ``code``, ``text``).'''

    entries = _get_index_entries(sources.index, set(sources.titles.get_column('code').to_list()))
    logger.info('Index entries:')
    logger.info(f'  Sheet rows: {sources.index.height: ,}')
    logger.info(f'  Entries for six-digit codes: {entries.height: ,}')
    logger.info(f'  Codes with entries: {entries.get_column("code").n_unique(): ,}\n')
    return entries

def build_descriptions(sources: NaicsSources, examples_entries: pl.DataFrame) -> pl.DataFrame:
    '''
    One row per code: title, description, examples channel and exclusions.

    Args:
        sources: The four Census files.
        examples_entries: The index entries that form examples channels (``entry_id``,
            ``code``, ``text``); every other entry of a code with index entries is a query.
    '''

    titles, codes = _get_titles(sources.titles)

    descriptions_2, descriptions_3 = _get_descriptions_1(sources.descriptions)

    exclusions, descriptions_exclusions = _get_exclusions(sources.exclusions, descriptions_3, codes)

    index_codes = set(_get_index_entries(sources.index, codes).get_column('code').to_list())
    examples, descriptions_examples = _get_examples(
        index_codes, examples_entries, descriptions_2, descriptions_3
    )
```

**`src/naics_embedder/data/download_data.py`, edit 11 of 13.** Replace:

```python
    )

    # Join all components and write final output
    # yapf: disable
    naics_final = (
        titles.join(descriptions, how='inner', on='code')
        .join(exclusions, how='left', on='code')
```

with:

```python
    )

    # yapf: disable
    return (
        titles.join(descriptions, how='inner', on='code')
        .join(exclusions, how='left', on='code')
```

**`src/naics_embedder/data/download_data.py`, edit 12 of 13.** Replace:

```python
    # yapf: enable

    (naics_final.write_parquet(cfg.output_parquet))
```

with:

```python
    # yapf: enable

# -------------------------------------------------------------------------------------------------
# Guard the descriptions file a supervision bundle pins
# -------------------------------------------------------------------------------------------------

# Where shipped configs name a supervision bundle: (config file, key path)
PINNING_CONFIGS: Tuple[Tuple[Path, Tuple[str, ...]], ...] = (
    (Path('conf/config.yaml'), ('supervision', 'manifest_path')),
    (Path('conf/graph.yaml'), ('supervision_manifest_path', )),
)

def pinned_description_fingerprints(
    pinning_configs: Sequence[Tuple[Path, Tuple[str, ...]]] = PINNING_CONFIGS,
) -> Dict[str, str]:
    '''
    The ``description_fingerprint`` of each bundle a config names, by manifest path.

    Configs, keys and manifests that do not exist are skipped: they pin nothing.
    '''

    fingerprints: Dict[str, str] = {}
    for config_path, keys in pinning_configs:
        if not Path(config_path).is_file():
            continue
        value = yaml.safe_load(Path(config_path).read_text())
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if not value:
            continue
        manifest_path = Path(value)
        if not manifest_path.is_file():
            logger.warning(f'{config_path} names a missing supervision manifest: {manifest_path}')
            continue
        manifest = json.loads(manifest_path.read_text())
        fingerprints[str(manifest_path)] = manifest['description_fingerprint']
    return fingerprints

def refuse_pinned_overwrite(
    output: Path,
    *,
    force: bool,
    pinning_configs: Sequence[Tuple[Path, Tuple[str, ...]]] = PINNING_CONFIGS,
) -> None:
    '''
    Refuse to overwrite the descriptions file a configured supervision bundle pins.

    Training fails closed once the descriptions file no longer matches its bundle's
    ``description_fingerprint``, so rewriting it would break every run against that bundle.

    Raises:
        FileExistsError: If ``output`` is pinned and ``force`` is False.
    '''

    if force or not Path(output).is_file():
        return
    digest = sha256_file(Path(output))
    pinned = sorted(
        path for path, fingerprint in pinned_description_fingerprints(pinning_configs).items()
        if fingerprint == digest
    )
    if pinned:
        raise FileExistsError(
            f'{output} is the descriptions file supervision bundle {pinned[0]} pins; rebuilding '
            'it would break training against that bundle. Write another output_parquet, or '
            'pass --force to overwrite it.'
        )

# -------------------------------------------------------------------------------------------------
# Combine all and write final output
# -------------------------------------------------------------------------------------------------

def download_preprocess_data(
    cfg: Optional[DownloadConfig] = None,
    *,
    force: bool = False,
) -> pl.DataFrame:
    '''
    Build the descriptions parquet and the index-roles parquet from the Census files.

    Every index entry takes its role from the frozen role table (``cfg.index_roles_csv``). A
    code's examples channel holds its examples-role entries only, and no validation or test query
    may match any training text (Req 3); both are checked before anything is written.

    Args:
        cfg: Download configuration; ``conf/data/download.yaml`` when omitted.
        force: Overwrite a descriptions file that a configured supervision bundle pins.
    '''

    # Create directories
    make_directories()

    # Load configuration from YAML
    if cfg is None:
        cfg = load_config(DownloadConfig, './data/download.yaml')

    logger.info('Configuration:')
    logger.info(cfg.model_dump_json(indent=2))
    logger.info('')

    refuse_pinned_overwrite(Path(cfg.output_parquet), force=force)

    roles_csv = Path(cfg.index_roles_csv)
    if not roles_csv.is_file():
        raise FileNotFoundError(
            f'index-entry role table not found: {roles_csv}; generate it once with '
            '`naics-embedder data roles`'
        )

    sources = load_naics_sources(cfg)

    role_rows = attach_role_text(read_role_table(roles_csv), naics_index_entries(sources))

    naics_final = build_descriptions(
        sources, role_rows.filter(pl.col('role') == IndexRole.EXAMPLES.value)
    )

    six_digit_codes = naics_final.filter(pl.col('level') == 6).get_column('code').to_list()
    validate_index_role_table(role_rows, six_digit_codes)
    verify_examples_channel(naics_final, role_rows)
    leakage = verify_role_leakage(naics_final, role_rows)
    logger.info(f'Held-out queries matching training text: {leakage}\n')

    (naics_final.write_parquet(cfg.output_parquet))
```

**`src/naics_embedder/data/download_data.py`, edit 13 of 13.** Replace:

```python
        message='NAICS codes (text + hierarchy) written to:',
        output_parquet=cfg.output_parquet,
        logger=logger,
    )
```

with:

```python
        message='NAICS codes (text + hierarchy) written to:',
        output_parquet=cfg.output_parquet,
        logger=logger,
    )

    Path(cfg.index_roles_parquet).parent.mkdir(parents=True, exist_ok=True)
    role_rows.write_parquet(cfg.index_roles_parquet)

    _parquet_stats(
        parquet_df=role_rows,
        message='NAICS index entries and their roles written to',
        output_parquet=cfg.index_roles_parquet,
        logger=logger,
    )
```

- [x] **Step 6: Add the preprocess options**

Modify `src/naics_embedder/cli/commands/data.py` with these 5 edits, in order. Each replaced text
occurs exactly once in the file.

**`src/naics_embedder/cli/commands/data.py`, edit 1 of 5.** Replace:

```python
from pathlib import Path

import typer
from rich.console import Console

from naics_embedder.data.download_data import download_preprocess_data
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.utils.config import SupervisionBuildConfig, load_config
from naics_embedder.utils.console import configure_logging
```

with:

```python
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from naics_embedder.data.download_data import download_preprocess_data
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.utils.config import DownloadConfig, SupervisionBuildConfig, load_config
from naics_embedder.utils.console import configure_logging
```

**`src/naics_embedder/cli/commands/data.py`, edit 2 of 5.** Replace:

```python
SUPERVISION_CONFIG = 'data/supervision.yaml'

# -------------------------------------------------------------------------------------------------
```

with:

```python
SUPERVISION_CONFIG = 'data/supervision.yaml'
DOWNLOAD_CONFIG = 'data/download.yaml'

SourceDirOption = Annotated[
    Optional[str],
    typer.Option(
        '--source-dir',
        help='Read the Census files from this directory, by file name, instead of downloading',
    ),
]

def _download_config(source_dir: Optional[str]) -> DownloadConfig:
    cfg = load_config(DownloadConfig, DOWNLOAD_CONFIG)
    if source_dir is not None:
        cfg = cfg.model_copy(update={'source_dir': source_dir})
    return cfg

# -------------------------------------------------------------------------------------------------
```

**`src/naics_embedder/cli/commands/data.py`, edit 3 of 5.** Replace:

```python
@app.command('preprocess')
def preprocess():
    '''
    Download and preprocess all raw NAICS data files.
```

with:

```python
@app.command('preprocess')
def preprocess(
    source_dir: SourceDirOption = None,
    force: Annotated[
        bool,
        typer.Option(
            '--force',
            help='Overwrite a descriptions file that a configured supervision bundle pins',
        ),
    ] = False,
):
    '''
    Download and preprocess all raw NAICS data files.
```

**`src/naics_embedder/cli/commands/data.py`, edit 4 of 5.** Replace:

```python
    The output file contains columns for code, title, description, examples,
    and exclusions for each NAICS code at all hierarchy levels (2-6 digit).

    Output:
        ``data/naics_descriptions.parquet`` - Unified NAICS taxonomy data.

    Example:
```

with:

```python
    The output file contains columns for code, title, description, examples,
    and exclusions for each NAICS code at all hierarchy levels (2-6 digit).
    Each code's examples channel holds its examples-role index entries only,
    per the committed role table (``data roles``).

    Output:
        ``data/naics_descriptions.parquet`` - Unified NAICS taxonomy data.
        ``data/naics_index_roles.parquet`` - Every index entry with its role.

    Example:
```

**`src/naics_embedder/cli/commands/data.py`, edit 5 of 5.** Replace:

```python
    console.rule('[bold green]Stage 1: Preprocessing[/bold green]')

    download_preprocess_data()

    console.print('\n[bold]Preprocessing complete.[/bold]\n')
```

with:

```python
    console.rule('[bold green]Stage 1: Preprocessing[/bold green]')

    try:
        download_preprocess_data(_download_config(source_dir), force=force)
    except FileExistsError as exc:
        console.print(f'[bold red]{exc}[/bold red]')
        raise typer.Exit(code=1)

    console.print('\n[bold]Preprocessing complete.[/bold]\n')
```

- [x] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_data_download.py tests/unit/test_config.py tests/unit/test_cli_commands.py -q`
Expected: `94 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1353 passed, 1 skipped`.

- [x] **Step 8: Lint and format**

Run: `./scripts/format_code.sh --check tests/fixtures/naics_sources.py tests/conftest.py tests/unit/test_data_download.py tests/unit/test_config.py tests/unit/test_cli_commands.py src/naics_embedder/utils/config.py src/naics_embedder/data/download_data.py src/naics_embedder/cli/commands/data.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` On a failure, run the
same command without `--check`, re-run Step 7, and record the change as a deviation.

- [x] **Step 9: Commit**

```bash
git add tests/fixtures/naics_sources.py tests/conftest.py tests/unit/test_data_download.py tests/unit/test_config.py tests/unit/test_cli_commands.py src/naics_embedder/utils/config.py conf/data/download.yaml src/naics_embedder/data/download_data.py src/naics_embedder/cli/commands/data.py
git commit -m "feat(data): build the examples channel from examples-role index entries"
```

### Task 6: The optional `index_roles` bundle member

A supervision bundle can now carry the index roles as an optional member under the unchanged
contract `stage3-supervision-v1`. Decision 1 in **Global Constraints** keeps it optional until
Stage 5.

- **At build time:** the member is written only after three checks against the bundle's own
  descriptions: one role per entry, examples channels from examples-role entries, and no leakage.
- **At load time:** `load_validated_bundle` checks the member's table whenever it is present.
- **Old bundles:** bundles without the member, 18403d29 among them, load unchanged.

`OutcomePanel.from_bundle` reads the member and the codebook's six-digit codes. This task builds
no bundle from real data.

**Files:**

- Modify: `tests/fixtures/supervision.py` (three fixtures)
- Modify: `tests/unit/test_supervision_artifacts.py` (append)
- Modify: `tests/unit/test_outcome_panel.py` (imports, append)
- Modify: `tests/unit/test_config.py`, specifically `TestSupervisionBuildConfig`
- Modify: `src/naics_embedder/utils/config.py`, specifically `SupervisionBuildConfig`
- Modify: `conf/data/supervision.yaml`
- Modify: `src/naics_embedder/supervision/artifacts.py`, specifically `_validate_relations` and
  the `load_validated_bundle` docstring
- Modify: `src/naics_embedder/data/supervision_bundle.py`
- Modify: `src/naics_embedder/panels/outcome.py` (imports, `from_bundle`)

**Interfaces:**

- Consumes:
  - Task 2: `INDEX_ROLES_ARTIFACT`, `INDEX_ROLE_COLUMNS`, `INDEX_ROLES_SCHEMA_VERSION`,
    `validate_index_role_table`, `verify_examples_channel` and `verify_role_leakage`.
  - Task 4: `OutcomePanel` and `SelectionLog`.
- Produces in `SupervisionBuildConfig`: `index_roles_parquet: Optional[str] = None`.
  `conf/data/supervision.yaml` sets it to `./data/naics_index_roles.parquet`.
- Produces in `naics_embedder.data.supervision_bundle`:
  - `ARTIFACT_FILENAMES['index_roles'] = 'naics_index_roles.parquet'`, with schema version
    `index-roles-v1`.
  - `generator_revision() -> str`, renamed from `_generator_revision` so that Task 7 can record
    it.
  - `generate_supervision_bundle_from_frames(..., index_roles: Optional[pl.DataFrame] = None)`.
    When the member is present, `validation_results` gains `index_roles_one_role_per_entry`,
    `index_roles_examples_channel` and `index_roles_no_leakage`.
  - `generate_supervision_bundle(cfg)` reads `cfg.index_roles_parquet` when set and records it
    in `generation_parameters`.
- Produces in `naics_embedder.panels.outcome`:
  `OutcomePanel.from_bundle(bundle: ValidatedSupervisionBundle, log_path) -> OutcomePanel`.
  Without the member it raises `ValueError` (`bundle has no 'index_roles' artifact`).
- Produces test fixtures in `tests/fixtures/supervision.py`:
  - `index_roles_fixture`
  - `text_descriptions_fixture`, the descriptions fixture with text columns
  - `generated_bundle_with_roles`, a bundle with the member

- [x] **Step 1: Write the failing tests**

Modify `tests/fixtures/supervision.py` with one edit. The replaced text occurs exactly once in the
file.

**`tests/fixtures/supervision.py`, edit 1 of 1.** Replace:

```python
# -------------------------------------------------------------------------------------------------
# Candidate batches: every aligned field carries a distinguishable per-slot ordinal (1, 2, 3, ...)
# -------------------------------------------------------------------------------------------------
```

with:

```python
# -------------------------------------------------------------------------------------------------
# Index-entry roles: the optional bundle member
# -------------------------------------------------------------------------------------------------

INDEX_ROLE_ROWS = [
    (0, '111111', 'Soybean farming', 'examples'),
    (1, '111111', 'Edamame farming', 'validation'),
    (2, '111112', 'Canola farming', 'examples'),
    (3, '111112', 'Sunflower farming', 'test'),
    (4, '222222', 'Coal mining', 'examples'),
    (5, '222222', 'Lignite mining', 'training'),
]
INDEX_ROLE_SCHEMA = {'entry_id': pl.Int64, 'code': pl.Utf8, 'text': pl.Utf8, 'role': pl.Utf8}

@pytest.fixture
def index_roles_fixture() -> pl.DataFrame:
    return pl.DataFrame(INDEX_ROLE_ROWS, schema=INDEX_ROLE_SCHEMA, orient='row')

@pytest.fixture
def text_descriptions_fixture(descriptions_fixture) -> pl.DataFrame:
    '''The five-code descriptions with text channels; examples hold examples-role entries only.'''

    examples = {'111111': 'Soybean farming', '111112': 'Canola farming', '222222': 'Coal mining'}
    return descriptions_fixture.with_columns(
        title=pl.concat_str(pl.lit('Industry '), pl.col('code')),
        description=pl.lit('This industry comprises establishments.'),
        examples=pl.col('code').replace_strict(examples, default=None),
        excluded=pl.lit(None, pl.Utf8),
    )

@pytest.fixture
def generated_bundle_with_roles(
    tmp_path, text_descriptions_fixture, pair_facts_fixture, index_roles_fixture
):
    return generate_supervision_bundle_from_frames(
        output_root=tmp_path,
        bundle_id='bundle-roles',
        generator_revision='revision-a',
        naics_vintage=2022,
        descriptions=text_descriptions_fixture,
        pair_facts=pair_facts_fixture,
        index_roles=index_roles_fixture,
    )

# -------------------------------------------------------------------------------------------------
# Candidate batches: every aligned field carries a distinguishable per-slot ordinal (1, 2, 3, ...)
# -------------------------------------------------------------------------------------------------
```

Modify `tests/unit/test_supervision_artifacts.py` with one edit. The replaced text occurs exactly
once in the file.

**`tests/unit/test_supervision_artifacts.py`, edit 1 of 1.** Replace:

```python
    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*pair facts'):
        load_validated_bundle(generated_bundle)
```

with:

```python
    with pytest.raises(ValueError, match='training_pairs.*bundle-a.*pair facts'):
        load_validated_bundle(generated_bundle)

# -------------------------------------------------------------------------------------------------
# The optional index-roles member
# -------------------------------------------------------------------------------------------------

def test_bundle_carries_the_index_roles_after_checking_them(
    generated_bundle_with_roles, index_roles_fixture
):
    manifest = json.loads(generated_bundle_with_roles.read_text())
    record = manifest['artifacts']['index_roles']

    assert record['path'] == 'naics_index_roles.parquet'
    assert record['schema_version'] == 'index-roles-v1'
    assert record['row_count'] == 6
    for check in (
        'index_roles_one_role_per_entry',
        'index_roles_examples_channel',
        'index_roles_no_leakage',
    ):
        assert manifest['validation_results'][check] is True
    bundle = load_validated_bundle(generated_bundle_with_roles)
    assert pl.read_parquet(bundle.artifact_path('index_roles')).equals(index_roles_fixture)

def test_bundle_refuses_an_examples_channel_holding_queries(
    tmp_path, text_descriptions_fixture, pair_facts_fixture, index_roles_fixture
):
    stale = text_descriptions_fixture.with_columns(
        examples=pl.when(pl.col('code') == '111111').then(
            pl.lit('Soybean farming; Edamame farming')
        ).otherwise('examples')
    )

    with pytest.raises(ValueError, match='examples channel other than'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='stale',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=stale,
            pair_facts=pair_facts_fixture,
            index_roles=index_roles_fixture,
        )
    assert list(tmp_path.iterdir()) == []

def test_bundle_refuses_a_held_out_query_matching_training_text(
    tmp_path, text_descriptions_fixture, pair_facts_fixture, index_roles_fixture
):
    # Entry 1 (validation) becomes another code's title
    leaky = index_roles_fixture.with_columns(
        text=pl.when(pl.col('entry_id') == 1).then(pl.lit('Industry 222222')).otherwise('text')
    )

    with pytest.raises(ValueError, match='held-out queries match training text'):
        generate_supervision_bundle_from_frames(
            output_root=tmp_path,
            bundle_id='leaky',
            generator_revision='revision-a',
            naics_vintage=2022,
            descriptions=text_descriptions_fixture,
            pair_facts=pair_facts_fixture,
            index_roles=leaky,
        )

def test_loader_rejects_an_index_entry_with_two_roles(generated_bundle_with_roles):
    _rewrite_member(
        generated_bundle_with_roles,
        'index_roles',
        lambda frame: frame.with_columns(entry_id=pl.lit(0, pl.Int64)),
    )

    with pytest.raises(ValueError, match='index_roles .*more than one role'):
        load_validated_bundle(generated_bundle_with_roles)

def test_production_bundle_takes_the_index_roles_from_its_config(tmp_path, hierarchy_descriptions):
    roles = pl.DataFrame(
        [
            (0, '311111', 'Dog food manufacturing', 'examples'),
            (1, '311111', 'Cat food manufacturing', 'validation'),
            (2, '441111', 'New car dealers', 'examples'),
        ],
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    )
    examples = {'311111': 'Dog food manufacturing', '441111': 'New car dealers'}
    descriptions = hierarchy_descriptions.with_columns(
        description=pl.lit('This industry comprises establishments.'),
        examples=pl.col('code').replace_strict(examples, default=None),
        excluded=pl.lit(None, pl.Utf8),
    )
    descriptions_path = tmp_path / 'naics_descriptions.parquet'
    roles_path = tmp_path / 'naics_index_roles.parquet'
    descriptions.write_parquet(descriptions_path)
    roles.write_parquet(roles_path)
    cfg = SupervisionBuildConfig(
        descriptions_parquet=str(descriptions_path),
        index_roles_parquet=str(roles_path),
        output_root=str(tmp_path / 'bundles'),
    )

    manifest = json.loads(generate_supervision_bundle(cfg).read_text())

    assert manifest['artifacts']['index_roles']['row_count'] == 3
    assert manifest['generation_parameters']['index_roles_parquet'] == str(roles_path.resolve())
```

Modify `tests/unit/test_outcome_panel.py` with these 2 edits, in order. Each replaced text occurs
exactly once in the file.

**`tests/unit/test_outcome_panel.py`, edit 1 of 2.** Replace:

```python
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog

pytestmark = pytest.mark.unit
```

with:

```python
)
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle

pytestmark = pytest.mark.unit
```

**`tests/unit/test_outcome_panel.py`, edit 2 of 2.** Replace:

```python
    with pytest.raises(ValueError, match='examples channel other than'):
        OutcomePanel.from_files(roles_path, descriptions_path, tmp_path / 'log.jsonl')
```

with:

```python
    with pytest.raises(ValueError, match='examples channel other than'):
        OutcomePanel.from_files(roles_path, descriptions_path, tmp_path / 'log.jsonl')

def test_from_bundle_reads_the_member_and_the_codebook(generated_bundle_with_roles, tmp_path):
    bundle = load_validated_bundle(generated_bundle_with_roles)

    panel = OutcomePanel.from_bundle(bundle, tmp_path / 'log.jsonl')

    assert panel.candidates == ('111111', '111112', '111113', '222222', '333333')
    assert panel.entryless_candidates == ('111113', '333333')
    assert panel.validation_queries('check')['entry_id'].to_list() == [1]

def test_from_bundle_needs_the_member(validated_bundle, tmp_path):
    with pytest.raises(ValueError, match='index_roles'):
        OutcomePanel.from_bundle(validated_bundle, tmp_path / 'log.jsonl')
```

Modify `tests/unit/test_config.py` with one edit. The replaced text occurs exactly once in the file.

**`tests/unit/test_config.py`, edit 1 of 1.** Replace:

```python
        cfg = load_config(SupervisionBuildConfig, 'data/supervision.yaml')

        assert cfg == SupervisionBuildConfig()
        assert cfg.contract_version == 'stage3-supervision-v1'
        assert cfg.relation_id['cross_sector'] == 99
```

with:

```python
        cfg = load_config(SupervisionBuildConfig, 'data/supervision.yaml')

        # The shipped build carries the index roles; the default (for fixtures) does not
        assert cfg == SupervisionBuildConfig(index_roles_parquet='./data/naics_index_roles.parquet')
        assert cfg.contract_version == 'stage3-supervision-v1'
        assert cfg.relation_id['cross_sector'] == 99
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py tests/unit/test_outcome_panel.py tests/unit/test_config.py -q`
Expected: `5 failed, 92 passed, 3 errors`. The failures are:

- `TypeError: generate_supervision_bundle_from_frames() got an unexpected keyword argument 'index_roles'`
- `AttributeError: type object 'OutcomePanel' has no attribute 'from_bundle'`
- `TestSupervisionBuildConfig::test_yaml_matches_defaults`

- [x] **Step 3: Add the configuration**

Modify `src/naics_embedder/utils/config.py` with one edit. The replaced text occurs exactly once in
the file.

**`src/naics_embedder/utils/config.py`, edit 1 of 1.** Replace:

```python
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    output_root: str = './data/supervision/stage3-supervision-v1'
    contract_version: Literal['stage3-supervision-v1'] = CONTRACT_VERSION
```

with:

```python
    descriptions_parquet: str = './data/naics_descriptions.parquet'
    index_roles_parquet: Optional[str] = Field(
        default=None,
        description=(
            'Index entries with their roles, from `data preprocess`; when set, the bundle carries '
            'them as its optional index_roles member'
        ),
    )
    output_root: str = './data/supervision/stage3-supervision-v1'
    contract_version: Literal['stage3-supervision-v1'] = CONTRACT_VERSION
```

Modify `conf/data/supervision.yaml` with one edit. The replaced text occurs exactly once in the
file.

**`conf/data/supervision.yaml`, edit 1 of 1.** Replace:

```yaml
descriptions_parquet: ./data/naics_descriptions.parquet
output_root: ./data/supervision/stage3-supervision-v1
contract_version: stage3-supervision-v1
```

with:

```yaml
descriptions_parquet: ./data/naics_descriptions.parquet
index_roles_parquet: ./data/naics_index_roles.parquet
output_root: ./data/supervision/stage3-supervision-v1
contract_version: stage3-supervision-v1
```

- [x] **Step 4: Validate the member at load time**

Modify `src/naics_embedder/supervision/artifacts.py` with these 2 edits, in order. Each replaced
text occurs exactly once in the file.

**`src/naics_embedder/supervision/artifacts.py`, edit 1 of 2.** Replace:

```python
    )

def load_validated_bundle(
    manifest_path: str | Path,
```

with:

```python
    )

    # Optional under stage3-supervision-v1: bundles built before the outcome panel lack it
    if INDEX_ROLES_ARTIFACT in manifest.artifacts:
        roles = read(INDEX_ROLES_ARTIFACT)
        six_digit_codes = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        _in_context(
            INDEX_ROLES_ARTIFACT,
            bundle_id,
            lambda: validate_index_role_table(roles, six_digit_codes.to_list()),
        )

def load_validated_bundle(
    manifest_path: str | Path,
```

**`src/naics_embedder/supervision/artifacts.py`, edit 2 of 2.** Replace:

```python
    metadata, the recorded validation results, and then re-runs the relational checks: codebook
    order and fingerprint, pair-fact identity/orientation/coverage/sentinels/exclusion derivation,
    long-form and matrix reconciliation, and training-pair identity, exclusion, and structure.

    Raises:
```

with:

```python
    metadata, the recorded validation results, and then re-runs the relational checks: codebook
    order and fingerprint, pair-fact identity/orientation/coverage/sentinels/exclusion derivation,
    long-form and matrix reconciliation, training-pair identity, exclusion, and structure, and,
    when the bundle carries one, the index-entry role table (one known role per entry, six-digit
    codes only, the examples-channel floor).

    Raises:
```

- [x] **Step 5: Write the member at build time**

Modify `src/naics_embedder/data/supervision_bundle.py` with these 14 edits, in order. Each replaced
text occurs exactly once in the file.

**`src/naics_embedder/data/supervision_bundle.py`, edit 1 of 14.** Replace:

```python
    iter_training_pair_batches,
)
from naics_embedder.supervision.artifacts import (
    STRUCTURAL_PAIR_COLUMNS,
    codebook_fingerprint,
    sha256_file,
    validate_exclusion_derivation,
    validate_matrix,
    validate_structural_pairs,
```

with:

```python
    iter_training_pair_batches,
)
from naics_embedder.panels.index_roles import verify_examples_channel, verify_role_leakage
from naics_embedder.supervision.artifacts import (
    INDEX_ROLE_COLUMNS,
    INDEX_ROLES_ARTIFACT,
    STRUCTURAL_PAIR_COLUMNS,
    codebook_fingerprint,
    sha256_file,
    validate_exclusion_derivation,
    validate_index_role_table,
    validate_matrix,
    validate_structural_pairs,
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 2 of 14.** Replace:

```python
    DISTANCE_MATRIX_SCHEMA_VERSION,
    DISTANCES_SCHEMA_VERSION,
    PAIR_FACTS_SCHEMA_VERSION,
    RELATION_MATRIX_SCHEMA_VERSION,
```

with:

```python
    DISTANCE_MATRIX_SCHEMA_VERSION,
    DISTANCES_SCHEMA_VERSION,
    INDEX_ROLES_SCHEMA_VERSION,
    PAIR_FACTS_SCHEMA_VERSION,
    RELATION_MATRIX_SCHEMA_VERSION,
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 3 of 14.** Replace:

```python
    'training_pairs': 'naics_training_pairs',
    'difficulty_thresholds': 'curriculum_difficulty_thresholds.json',
}
```

with:

```python
    'training_pairs': 'naics_training_pairs',
    'difficulty_thresholds': 'curriculum_difficulty_thresholds.json',
    INDEX_ROLES_ARTIFACT: 'naics_index_roles.parquet',
}
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 4 of 14.** Replace:

```python
    'training_pairs': TRAINING_PAIRS_SCHEMA_VERSION,
    'difficulty_thresholds': DIFFICULTY_THRESHOLDS_SCHEMA_VERSION,
}
```

with:

```python
    'training_pairs': TRAINING_PAIRS_SCHEMA_VERSION,
    'difficulty_thresholds': DIFFICULTY_THRESHOLDS_SCHEMA_VERSION,
    INDEX_ROLES_ARTIFACT: INDEX_ROLES_SCHEMA_VERSION,
}
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 5 of 14.** Replace:

```python
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def _generator_revision() -> str:
    '''Git revision of the generating checkout, or the installed package version.'''
```

with:

```python
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def generator_revision() -> str:
    '''Git revision of the generating checkout, or the installed package version.'''
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 6 of 14.** Replace:

```python
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from exc

def _validate_training_identity(batch: pl.DataFrame, pair_keys: pl.DataFrame) -> None:
```

with:

```python
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from exc

def _validate_index_roles(
    index_roles: pl.DataFrame,
    descriptions: pl.DataFrame,
    codebook: pl.DataFrame,
) -> Dict[str, bool]:
    '''Check the optional index-roles member against the bundle's own descriptions and codes.'''

    six_digit_codes = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
    validate_index_role_table(index_roles, six_digit_codes.to_list())
    verify_examples_channel(descriptions, index_roles)
    verify_role_leakage(descriptions, index_roles)
    return {
        'index_roles_one_role_per_entry': True,
        'index_roles_examples_channel': True,
        'index_roles_no_leakage': True,
    }

def _validate_training_identity(batch: pl.DataFrame, pair_keys: pl.DataFrame) -> None:
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 7 of 14.** Replace:

```python
    cross_sector_cap: int,
    cap_seed: int,
) -> Dict[str, ArtifactRecord]:
    exclusions = int(pair_facts.get_column('is_explicit_exclusion').sum())
```

with:

```python
    cross_sector_cap: int,
    cap_seed: int,
    index_roles: Optional[pl.DataFrame] = None,
) -> Dict[str, ArtifactRecord]:
    exclusions = int(pair_facts.get_column('is_explicit_exclusion').sum())
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 8 of 14.** Replace:

```python
        'relation_matrix': _record('relation_matrix', parquet('relation_matrix', relation_matrix)),
    }

    counts = {'rows': 0, 'exclusions': 0}
```

with:

```python
        'relation_matrix': _record('relation_matrix', parquet('relation_matrix', relation_matrix)),
    }
    if index_roles is not None:
        records[INDEX_ROLES_ARTIFACT] = _record(
            INDEX_ROLES_ARTIFACT,
            parquet(INDEX_ROLES_ARTIFACT,
                    index_roles.select(INDEX_ROLE_COLUMNS).sort('entry_id')),
        )

    counts = {'rows': 0, 'exclusions': 0}
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 9 of 14.** Replace:

```python
    cross_sector_cap: int = CROSS_SECTOR_NEGATIVE_CAP,
    cap_seed: int = CROSS_SECTOR_CAP_SEED,
) -> Path:
    '''
    Validate canonical frames and publish them as one immutable supervision bundle.

    Artifacts are written to ``<output_root>/.<bundle_id>.staging``; the manifest is written only
```

with:

```python
    cross_sector_cap: int = CROSS_SECTOR_NEGATIVE_CAP,
    cap_seed: int = CROSS_SECTOR_CAP_SEED,
    index_roles: Optional[pl.DataFrame] = None,
) -> Path:
    '''
    Validate canonical frames and publish them as one immutable supervision bundle.

    ``index_roles`` (every index entry with its text and role) becomes the optional
    ``index_roles`` member after three checks against ``descriptions``: one known role per entry,
    examples channels built from examples-role entries only, and no held-out query matching any
    training text.

    Artifacts are written to ``<output_root>/.<bundle_id>.staging``; the manifest is written only
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 10 of 14.** Replace:

```python
    _validate_matrices(pair_facts, codebook, distance_matrix, relation_matrix)
    validation_results['matrix_reconciliation'] = True

    output_root.mkdir(parents=True, exist_ok=True)
```

with:

```python
    _validate_matrices(pair_facts, codebook, distance_matrix, relation_matrix)
    validation_results['matrix_reconciliation'] = True
    if index_roles is not None:
        validation_results.update(_validate_index_roles(index_roles, descriptions, codebook))

    output_root.mkdir(parents=True, exist_ok=True)
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 11 of 14.** Replace:

```python
            cross_sector_cap=cross_sector_cap,
            cap_seed=cap_seed,
        )
        validation_results.update(
```

with:

```python
            cross_sector_cap=cross_sector_cap,
            cap_seed=cap_seed,
            index_roles=index_roles,
        )
        validation_results.update(
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 12 of 14.** Replace:

```python
    descriptions_path = Path(cfg.descriptions_parquet)
    descriptions = pl.read_parquet(descriptions_path)
    codebook = build_codebook(descriptions)
    distances = compute_structural_distances(
```

with:

```python
    descriptions_path = Path(cfg.descriptions_parquet)
    descriptions = pl.read_parquet(descriptions_path)
    parameters = {
        'descriptions_parquet': str(descriptions_path.resolve()),
        'output_root': str(Path(cfg.output_root).resolve()),
    }
    index_roles = None
    if cfg.index_roles_parquet is not None:
        index_roles_path = Path(cfg.index_roles_parquet)
        index_roles = pl.read_parquet(index_roles_path)
        parameters['index_roles_parquet'] = str(index_roles_path.resolve())
    codebook = build_codebook(descriptions)
    distances = compute_structural_distances(
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 13 of 14.** Replace:

```python
        output_root=Path(cfg.output_root),
        bundle_id=str(uuid.uuid4()),
        generator_revision=_generator_revision(),
        naics_vintage=cfg.naics_vintage,
        descriptions=descriptions,
```

with:

```python
        output_root=Path(cfg.output_root),
        bundle_id=str(uuid.uuid4()),
        generator_revision=generator_revision(),
        naics_vintage=cfg.naics_vintage,
        descriptions=descriptions,
```

**`src/naics_embedder/data/supervision_bundle.py`, edit 14 of 14.** Replace:

```python
        description_fingerprint=sha256_file(descriptions_path),
        structural_relation_ids=cfg.relation_id,
        generation_parameters={
            'descriptions_parquet': str(descriptions_path.resolve()),
            'output_root': str(Path(cfg.output_root).resolve()),
        },
    )
```

with:

```python
        description_fingerprint=sha256_file(descriptions_path),
        structural_relation_ids=cfg.relation_id,
        generation_parameters=parameters,
        index_roles=index_roles,
    )
```

- [x] **Step 6: Read the panel from a bundle**

Modify `src/naics_embedder/panels/outcome.py` with these 2 edits, in order. Each replaced text
occurs exactly once in the file.

**`src/naics_embedder/panels/outcome.py`, edit 1 of 2.** Replace:

```python
from naics_embedder.panels.index_roles import role_table_fingerprint, verify_examples_channel
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.supervision.artifacts import INDEX_ROLE_COLUMNS, validate_index_role_table
from naics_embedder.supervision.schema import IndexRole
```

with:

```python
from naics_embedder.panels.index_roles import role_table_fingerprint, verify_examples_channel
from naics_embedder.panels.selection_log import SelectionEvent, SelectionLog
from naics_embedder.supervision.artifacts import (
    INDEX_ROLE_COLUMNS,
    INDEX_ROLES_ARTIFACT,
    ValidatedSupervisionBundle,
    validate_index_role_table,
)
from naics_embedder.supervision.schema import IndexRole
```

**`src/naics_embedder/panels/outcome.py`, edit 2 of 2.** Replace:

```python
        verify_examples_channel(descriptions, roles)
        candidates = descriptions.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        return cls(roles, candidates.to_list(), SelectionLog(Path(log_path)))
```

with:

```python
        verify_examples_channel(descriptions, roles)
        candidates = descriptions.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        return cls(roles, candidates.to_list(), SelectionLog(Path(log_path)))

    @classmethod
    def from_bundle(
        cls,
        bundle: ValidatedSupervisionBundle,
        log_path: Union[str, Path],
    ) -> 'OutcomePanel':
        '''
        The panel from a validated supervision bundle's ``index_roles`` member and codebook.

        The bundle checked the roles against its descriptions when it was built.

        Raises:
            ValueError: If the bundle has no ``index_roles`` member.
        '''

        roles = pl.read_parquet(bundle.artifact_path(INDEX_ROLES_ARTIFACT))
        codebook = pl.read_parquet(bundle.artifact_path('codebook'))
        candidates = codebook.filter(pl.col('code').str.len_chars() == 6).get_column('code')
        return cls(roles, candidates.to_list(), SelectionLog(Path(log_path)))
```

- [x] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_supervision_artifacts.py tests/unit/test_outcome_panel.py tests/unit/test_config.py -q`
Expected: `100 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1360 passed, 1 skipped`.

- [x] **Step 8: Lint and format**

Run: `./scripts/format_code.sh --check tests/fixtures/supervision.py tests/unit/test_supervision_artifacts.py tests/unit/test_outcome_panel.py tests/unit/test_config.py src/naics_embedder/utils/config.py src/naics_embedder/supervision/artifacts.py src/naics_embedder/data/supervision_bundle.py src/naics_embedder/panels/outcome.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` On a failure, run the
same command without `--check`, re-run Step 7, and record the change as a deviation.

- [x] **Step 9: Commit**

```bash
git add tests/fixtures/supervision.py tests/unit/test_supervision_artifacts.py tests/unit/test_outcome_panel.py tests/unit/test_config.py src/naics_embedder/utils/config.py conf/data/supervision.yaml src/naics_embedder/supervision/artifacts.py src/naics_embedder/data/supervision_bundle.py src/naics_embedder/panels/outcome.py
git commit -m "feat(supervision): carry index roles as an optional stage3-supervision-v1 member"
```

### Task 7: `data roles` draws the frozen table

`naics-embedder data roles` runs once and does four things:

1. **Static training text.** It builds every code's titles, descriptions, exclusions and
   fallback examples, without any index entry.
2. **Eligibility.** It withholds from the held-out splits every entry that matches that text or
   another entry.
3. **Assignment.** It assigns roles by the D4 quotas.
4. **Output.** It rebuilds the descriptions from the drawn roles, checks them, and writes the
   CSV and a provenance JSON.

An existing table is replaced only with `--force`, because redrawing unseals both splits.

**Files:**

- Create: `tests/unit/test_index_role_table.py`
- Modify: `tests/unit/test_config.py` (imports, `TestOutcomePanelConfig`)
- Modify: `tests/unit/test_cli_commands.py`, specifically the `data roles` tests
- Modify: `src/naics_embedder/utils/config.py` (import, `OutcomePanelConfig`)
- Create: `conf/data/outcome_panel.yaml`
- Create: `src/naics_embedder/data/index_role_table.py`
- Modify: `src/naics_embedder/cli/commands/data.py` (docstring, imports, `roles`)

**Interfaces:**

- Consumes:
  - Task 2: `ROLE_ORDER`, `RoleFractions`, `held_out_eligibility`, `assign_index_roles`,
    `attach_role_text`, `verify_examples_channel`, `verify_role_leakage`, `write_role_table`,
    `validate_index_role_table`, `INDEX_ROLES_SCHEMA_VERSION` and `IndexRole`.
  - Task 1: `training_text_segments`.
  - Task 5: `load_naics_sources`, `naics_index_entries`, `build_descriptions` and
    `_download_config`.
  - Task 6: `generator_revision`.
- Produces `OutcomePanelConfig` (`extra='forbid'`), which `conf/data/outcome_panel.yaml`
  mirrors:
  - `provenance_json: str = './conf/data/index_roles_provenance.json'`
  - `seed: int = 20260924`
  - `fractions: Dict[str, float]`, which defaults to examples 0.30, training 0.35, validation
    0.20 and test 0.15. It must name exactly those four roles, each non-negative, and sum to 1
    in exact decimal arithmetic.
  - `examples_floor: int = 1` (≥ 0)
  - `near_duplicate_min_jaccard: float = 0.9`, in (0, 1]
  - `selection_log: str = './logs/selection_log.jsonl'`
- Produces in `naics_embedder.data.index_role_table`:
  `generate_index_role_table(download_cfg: DownloadConfig, panel_cfg: OutcomePanelConfig, *, force: bool = False) -> Path`.
  - It returns the CSV path.
  - It raises `FileExistsError` for an existing table without `force`.
  - The provenance keys are `schema_version`, `index_file` (`url`, `sheet`, `sha256`), `seed`,
    `fractions` (as strings), `examples_floor`, `near_duplicate_min_jaccard` (`'9/10'`),
    `eligibility`, `roles`, `codes_with_role`, `held_out_leakage`, `generator_revision`,
    `library_versions`, `generated_at` and `role_table` (`path`, `sha256`).
- Produces in `naics_embedder.cli.commands.data`: `OUTCOME_PANEL_CONFIG =
  'data/outcome_panel.yaml'` and `naics-embedder data roles [--source-dir DIR] [--force]`, which
  exits 1 on a refused redraw.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_index_role_table.py` with exactly this content:

```python
'''
Drawing the frozen index-entry role table on the miniature Census sources (Req 3; roadmap D4).

Per-code quotas are worked out by hand: 111110 has 4 entries, all eligible, so E1 TR1 V1 TE1;
111120 has 6 entries, one of which repeats its title and can never be held out, so E2 TR2 V1 TE1.
'''

import json

import polars as pl
import pytest

from naics_embedder.data import index_role_table
from naics_embedder.panels.index_roles import read_role_table, role_table_fingerprint
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.config import DownloadConfig, OutcomePanelConfig

pytestmark = pytest.mark.unit

@pytest.fixture
def configs(tmp_path, monkeypatch, naics_sources):
    monkeypatch.setattr(index_role_table, 'load_naics_sources', lambda cfg: naics_sources)
    download_cfg = DownloadConfig(index_roles_csv=str(tmp_path / 'conf' / 'index_roles.csv'))
    panel_cfg = OutcomePanelConfig(provenance_json=str(tmp_path / 'conf' / 'provenance.json'))
    return download_cfg, panel_cfg

def test_every_entry_gets_one_role_in_per_code_quotas(configs):
    download_cfg, panel_cfg = configs

    table_path = index_role_table.generate_index_role_table(download_cfg, panel_cfg)
    roles = read_role_table(table_path)

    assert roles.get_column('entry_id').to_list() == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    counts = {
        code: dict(frame.group_by('role').len().sort('role').iter_rows())
        for (code, ), frame in roles.group_by('code')
    }
    assert counts == {
        '111110': {
            'examples': 1,
            'test': 1,
            'training': 1,
            'validation': 1
        },
        '111120': {
            'examples': 2,
            'test': 1,
            'training': 2,
            'validation': 1
        },
    }
    # Entry 10 repeats its code's title, so it is never held out
    assert roles.filter(pl.col('entry_id') == 10)['role'][0] in ('examples', 'training')

def test_provenance_records_the_draw_and_its_checks(configs):
    download_cfg, panel_cfg = configs

    table_path = index_role_table.generate_index_role_table(download_cfg, panel_cfg)
    provenance = json.loads(open(panel_cfg.provenance_json).read())

    assert provenance['role_table'] == {'path': str(table_path), 'sha256': sha256_file(table_path)}
    assert provenance['role_table']['sha256'] == role_table_fingerprint(read_role_table(table_path))
    assert provenance['index_file']['sha256'] == download_cfg.index_sha256
    assert provenance['seed'] == 20260924
    assert provenance['fractions'] == {
        'examples': '3/10',
        'training': '7/20',
        'validation': '1/5',
        'test': '3/20',
    }
    assert provenance['near_duplicate_min_jaccard'] == '9/10'
    assert provenance['eligibility'] == {
        'entries': 10,
        'exact_static': 1,
        'near_duplicate_static': 1,
        'exact_entry': 0,
        'near_duplicate_entry': 0,
        'withheld_exact': 1,
        'withheld_near_duplicate': 0,
        'withheld': 1,
        'eligible': 9,
    }
    assert provenance['roles'] == {'examples': 3, 'training': 3, 'validation': 2, 'test': 2}
    assert provenance['codes_with_role'] == {
        'examples': 2,
        'training': 2,
        'validation': 2,
        'test': 2,
    }
    assert provenance['held_out_leakage'] == {
        'validation': {
            'exact': 0,
            'near_duplicate': 0
        },
        'test': {
            'exact': 0,
            'near_duplicate': 0
        },
    }

def test_the_table_is_drawn_once(configs):
    download_cfg, panel_cfg = configs
    table_path = index_role_table.generate_index_role_table(download_cfg, panel_cfg)
    first = sha256_file(table_path)

    with pytest.raises(FileExistsError, match='--force'):
        index_role_table.generate_index_role_table(download_cfg, panel_cfg)
    index_role_table.generate_index_role_table(download_cfg, panel_cfg, force=True)

    # The same seed redraws the same table
    assert sha256_file(table_path) == first
```

Modify `tests/unit/test_config.py` with these 2 edits, in order. Each replaced text occurs exactly
once in the file.

**`tests/unit/test_config.py`, edit 1 of 2.** Replace:

```python
    DownloadConfig,
    GraphConfig,
    SamplingConfig,
    SansStaticConfig,
```

with:

```python
    DownloadConfig,
    GraphConfig,
    OutcomePanelConfig,
    SamplingConfig,
    SansStaticConfig,
```

**`tests/unit/test_config.py`, edit 2 of 2.** Replace:

```python
            SupervisionBuildConfig(rank_order_weight=0.35)

# -------------------------------------------------------------------------------------------------
# Repaired Stage-3 runtime configuration
```

with:

```python
            SupervisionBuildConfig(rank_order_weight=0.35)

@pytest.mark.unit
class TestOutcomePanelConfig:
    '''How index-entry roles are drawn (roadmap D4), and the selection log.'''

    def test_yaml_matches_defaults(self):
        cfg = load_config(OutcomePanelConfig, 'data/outcome_panel.yaml')

        assert cfg == OutcomePanelConfig()
        assert cfg.fractions == {
            'examples': 0.30,
            'training': 0.35,
            'validation': 0.20,
            'test': 0.15,
        }
        assert cfg.seed == 20260924
        assert cfg.selection_log == './logs/selection_log.jsonl'

    @pytest.mark.parametrize(
        'fractions',
        [
            {
                'examples': 0.30,
                'training': 0.35,
                'validation': 0.35
            },
            {
                'examples': 0.30,
                'training': 0.35,
                'validation': 0.20,
                'test': 0.10
            },
            {
                'examples': 0.60,
                'training': 0.35,
                'validation': 0.20,
                'test': -0.15
            },
        ],
    )
    def test_fractions_name_every_role_and_sum_to_one(self, fractions):
        with pytest.raises(ValidationError):
            OutcomePanelConfig(fractions=fractions)

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValidationError):
            OutcomePanelConfig(test_fraction=0.15)

# -------------------------------------------------------------------------------------------------
# Repaired Stage-3 runtime configuration
```

Modify `tests/unit/test_cli_commands.py` with one edit. The replaced text occurs exactly once in the
file.

**`tests/unit/test_cli_commands.py`, edit 1 of 1.** Replace:

```python
    assert len(calls) == 1
    assert str(manifest) in result.output

def test_tools_config_passes_config_path(monkeypatch, runner, tmp_path):
```

with:

```python
    assert len(calls) == 1
    assert str(manifest) in result.output

def test_data_roles_draws_the_table_with_both_configs(monkeypatch, runner, tmp_path):
    calls = []

    def fake_generate(download_cfg, panel_cfg, force):
        calls.append((download_cfg, panel_cfg, force))
        return tmp_path / 'index_roles.csv'

    monkeypatch.setattr(data_cli, 'generate_index_role_table', fake_generate)

    result = runner.invoke(data_cli.app, ['roles', '--source-dir', '/sources'])

    assert result.exit_code == 0
    [(download_cfg, panel_cfg, force)] = calls
    assert download_cfg.source_dir == '/sources'
    assert panel_cfg.seed == 20260924
    assert force is False
    assert 'index_roles.csv' in result.output

def test_data_roles_refuses_to_redraw_without_force(monkeypatch, runner):

    def refuse(download_cfg, panel_cfg, force):
        raise FileExistsError('the role table exists; pass --force to redraw it')

    monkeypatch.setattr(data_cli, 'generate_index_role_table', refuse)

    result = runner.invoke(data_cli.app, ['roles'])

    assert result.exit_code == 1
    assert '--force' in result.output

def test_tools_config_passes_config_path(monkeypatch, runner, tmp_path):
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_index_role_table.py tests/unit/test_config.py tests/unit/test_cli_commands.py -q`
Expected: two collection errors:

- `ImportError: cannot import name 'OutcomePanelConfig' from 'naics_embedder.utils.config'`
- `ImportError: cannot import name 'index_role_table' from 'naics_embedder.data'`

- [x] **Step 3: Add the configuration**

Modify `src/naics_embedder/utils/config.py` with these 2 edits, in order. Each replaced text occurs
exactly once in the file.

**`src/naics_embedder/utils/config.py`, edit 1 of 2.** Replace:

```python
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union
```

with:

```python
import logging
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union
```

**`src/naics_embedder/utils/config.py`, edit 2 of 2.** Replace:

```python
        }
    )

class SupervisionRuntimeConfig(BaseModel):
```

with:

```python
        }
    )

class OutcomePanelConfig(BaseModel):
    '''How the outcome panel's index-entry roles are drawn (roadmap D4), and its selection log.'''

    model_config = ConfigDict(extra='forbid')

    provenance_json: str = Field(
        default='./conf/data/index_roles_provenance.json',
        description='Where `data roles` records how the role table was drawn',
    )
    seed: int = Field(default=20260924, description='Base seed; each code draws with (seed, code)')
    fractions: Dict[str, float] = Field(
        default_factory=lambda: {
            'examples': 0.30,
            'training': 0.35,
            'validation': 0.20,
            'test': 0.15,
        },
        description="Target share of each code's index entries per role",
    )
    examples_floor: int = Field(
        default=1, ge=0, description='Minimum examples-channel entries for a code with entries'
    )
    near_duplicate_min_jaccard: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description='Character-trigram Jaccard similarity at which two texts are near-duplicates',
    )
    selection_log: str = Field(
        default='./logs/selection_log.jsonl',
        description='Append-only log of every panel read and test-split opening',
    )

    @field_validator('fractions')
    @classmethod
    def validate_fractions(cls, value: Dict[str, float]) -> Dict[str, float]:
        roles = {'examples', 'training', 'validation', 'test'}
        if set(value) != roles:
            raise ValueError(f'fractions must name exactly {sorted(roles)}')
        if any(share < 0 for share in value.values()):
            raise ValueError('fractions must be non-negative')
        if sum(Fraction(str(share)) for share in value.values()) != 1:
            raise ValueError('fractions must sum to 1')
        return value

class SupervisionRuntimeConfig(BaseModel):
```

Create `conf/data/outcome_panel.yaml` with exactly this content:

```yaml
# The outcome panel (roadmap Stage 2): how index-entry roles are drawn, and the selection log

# data roles records the draw here, beside the frozen table (download.yaml: index_roles_csv)
provenance_json: ./conf/data/index_roles_provenance.json

# Per code, stratified: largest-remainder quotas of these shares, remainder ties broken by a draw
# seeded with (seed, code), at least examples_floor examples-channel entries (roadmap D4)
seed: 20260924
fractions:
  examples: 0.30
  training: 0.35
  validation: 0.20
  test: 0.15
examples_floor: 1

# Exact matches and near-duplicates (character-trigram Jaccard at or above this) of any training
# text never become validation or test queries (Req 3, Verification "Leakage")
near_duplicate_min_jaccard: 0.9

# Every validation or test read, and every test-split opening, is appended here (Req 4). logs/ is
# gitignored: a log inside a worktree goes when the worktree is removed.
selection_log: ./logs/selection_log.jsonl
```

- [x] **Step 4: Write the generator**

Create `src/naics_embedder/data/index_role_table.py` with exactly this content:

```python
'''
Draw the frozen index-entry role table (roadmap Stage 2; Req 3; D4).

``naics-embedder data roles`` runs this once. The table it writes (``conf/data/index_roles.csv``)
is committed and ``data preprocess`` applies it from then on, so the sealed validation and test
splits never move. Redrawing reassigns every entry and unseals both splits, so an existing table
is replaced only with ``force``.
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

import polars as pl

from naics_embedder.data.download_data import (
    build_descriptions,
    load_naics_sources,
    naics_index_entries,
)
from naics_embedder.data.supervision_bundle import generator_revision
from naics_embedder.panels.index_roles import (
    ROLE_ORDER,
    RoleFractions,
    assign_index_roles,
    attach_role_text,
    held_out_eligibility,
    verify_examples_channel,
    verify_role_leakage,
    write_role_table,
)
from naics_embedder.panels.leakage import training_text_segments
from naics_embedder.supervision.artifacts import validate_index_role_table
from naics_embedder.supervision.schema import INDEX_ROLES_SCHEMA_VERSION, IndexRole
from naics_embedder.utils.config import DownloadConfig, OutcomePanelConfig

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Generate
# -------------------------------------------------------------------------------------------------

def generate_index_role_table(
    download_cfg: DownloadConfig,
    panel_cfg: OutcomePanelConfig,
    *,
    force: bool = False,
) -> Path:
    '''
    Draw every index entry's role, check the result, and write the table and its provenance.

    Held-out (validation and test) queries are drawn only from entries that match no title,
    description, exclusion text, fallback examples text or other index entry, exactly or as a
    near-duplicate; the provenance counts the entries withheld for each reason.

    Returns:
        The role table's path (``download_cfg.index_roles_csv``).

    Raises:
        FileExistsError: If the table exists and ``force`` is False.
        ValueError: If the drawn roles fail a Req 3 check.
    '''

    table_path = Path(download_cfg.index_roles_csv)
    if table_path.exists() and not force:
        raise FileExistsError(
            f'{table_path} exists: the role table is drawn once and committed. Redrawing it '
            'reassigns every entry and unseals the validation and test splits; pass --force only '
            'to do that deliberately.'
        )
    fractions = RoleFractions.from_mapping(panel_cfg.fractions)
    min_jaccard = Fraction(str(panel_cfg.near_duplicate_min_jaccard))

    sources = load_naics_sources(download_cfg)
    entries = naics_index_entries(sources)

    # Training text whatever the roles: titles, descriptions, exclusion text, and the illustrative
    # examples of codes without index entries
    static = build_descriptions(sources, entries.clear())
    eligibility = held_out_eligibility(entries, training_text_segments(static), min_jaccard)
    roles = assign_index_roles(
        entries, eligibility.eligible, fractions, panel_cfg.seed, panel_cfg.examples_floor
    )

    role_rows = attach_role_text(roles, entries)
    descriptions = build_descriptions(
        sources, role_rows.filter(pl.col('role') == IndexRole.EXAMPLES.value)
    )
    validate_index_role_table(
        role_rows,
        descriptions.filter(pl.col('level') == 6).get_column('code').to_list(),
        min_examples_per_code=panel_cfg.examples_floor,
    )
    verify_examples_channel(descriptions, role_rows)
    leakage = verify_role_leakage(descriptions, role_rows, min_jaccard)

    fingerprint = write_role_table(roles, table_path)
    provenance = _provenance(
        download_cfg, panel_cfg, fractions, min_jaccard, eligibility.counts, roles, leakage
    )
    provenance['role_table'] = {'path': str(table_path), 'sha256': fingerprint}
    provenance_path = Path(panel_cfg.provenance_json)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')

    logger.info(f'Index-entry roles: {provenance["roles"]}')
    logger.info(f'Held-out eligibility: {eligibility.counts}')
    logger.info(f'Role table ({fingerprint}) written to: {table_path}')
    logger.info(f'Provenance written to: {provenance_path}\n')
    return table_path

def _provenance(
    download_cfg: DownloadConfig,
    panel_cfg: OutcomePanelConfig,
    fractions: RoleFractions,
    min_jaccard: Fraction,
    eligibility: Dict[str, int],
    roles: pl.DataFrame,
    leakage: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    by_role = {role.value: roles.filter(pl.col('role') == role.value) for role in ROLE_ORDER}
    return {
        'schema_version': INDEX_ROLES_SCHEMA_VERSION,
        'index_file': {
            'url': download_cfg.url_index,
            'sheet': download_cfg.sheet_index,
            'sha256': download_cfg.index_sha256,
        },
        'seed': panel_cfg.seed,
        'fractions': {
            role.value: str(fractions.of(role))
            for role in ROLE_ORDER
        },
        'examples_floor': panel_cfg.examples_floor,
        'near_duplicate_min_jaccard': str(min_jaccard),
        'eligibility': eligibility,
        'roles': {
            role: frame.height
            for role, frame in by_role.items()
        },
        'codes_with_role': {
            role: frame.get_column('code').n_unique()
            for role, frame in by_role.items()
        },
        'held_out_leakage': leakage,
        'generator_revision': generator_revision(),
        'library_versions': {
            name: version(name)
            for name in ('numpy', 'polars', 'scikit-learn')
        },
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
```

- [x] **Step 5: Add the command**

Modify `src/naics_embedder/cli/commands/data.py` with these 4 edits, in order. Each replaced text
occurs exactly once in the file.

**`src/naics_embedder/cli/commands/data.py`, edit 1 of 4.** Replace:

```python
Commands:
    preprocess: Download raw NAICS files and produce descriptions parquet.
    supervision: Build codebook, pair facts, compatibility distance/relation artifacts,
```

with:

```python
Commands:
    roles: Draw the frozen index-entry role table, once; it is committed and preprocess applies
        it.
    preprocess: Download raw NAICS files and produce descriptions parquet.
    supervision: Build codebook, pair facts, compatibility distance/relation artifacts,
```

**`src/naics_embedder/cli/commands/data.py`, edit 2 of 4.** Replace:

```python
from naics_embedder.data.download_data import download_preprocess_data
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.utils.config import DownloadConfig, SupervisionBuildConfig, load_config
from naics_embedder.utils.console import configure_logging
```

with:

```python
from naics_embedder.data.download_data import download_preprocess_data
from naics_embedder.data.index_role_table import generate_index_role_table
from naics_embedder.data.supervision_bundle import generate_supervision_bundle
from naics_embedder.utils.config import (
    DownloadConfig,
    OutcomePanelConfig,
    SupervisionBuildConfig,
    load_config,
)
from naics_embedder.utils.console import configure_logging
```

**`src/naics_embedder/cli/commands/data.py`, edit 3 of 4.** Replace:

```python
SUPERVISION_CONFIG = 'data/supervision.yaml'
DOWNLOAD_CONFIG = 'data/download.yaml'

SourceDirOption = Annotated[
```

with:

```python
SUPERVISION_CONFIG = 'data/supervision.yaml'
DOWNLOAD_CONFIG = 'data/download.yaml'
OUTCOME_PANEL_CONFIG = 'data/outcome_panel.yaml'

SourceDirOption = Annotated[
```

**`src/naics_embedder/cli/commands/data.py`, edit 4 of 4.** Replace:

```python
    console.print('\n[bold]Preprocessing complete.[/bold]\n')

# -------------------------------------------------------------------------------------------------
```

with:

```python
    console.print('\n[bold]Preprocessing complete.[/bold]\n')

# -------------------------------------------------------------------------------------------------
# Draw the index-entry role table
# -------------------------------------------------------------------------------------------------

@app.command('roles')
def roles(
    source_dir: SourceDirOption = None,
    force: Annotated[
        bool,
        typer.Option(
            '--force',
            help='Redraw an existing table: reassigns every entry and unseals the splits',
        ),
    ] = False,
):
    '''
    Draw the frozen index-entry role table, once.

    Gives every Census index entry exactly one role, per code and stratified: examples-channel
    text, or a training, validation or test query for the outcome panel (roadmap D4). No
    validation or test query matches any training text. The table is committed, and preprocess
    applies it from then on.

    Output:
        ``conf/data/index_roles.csv`` and ``conf/data/index_roles_provenance.json``.

    Example:
        Draw the table from local copies of the Census files::

            $ uv run naics-embedder data roles --source-dir ~/Downloads/Data
    '''

    configure_logging('data_roles.log')

    console.rule('[bold green]Drawing Index-Entry Roles[/bold green]')

    try:
        table_path = generate_index_role_table(
            _download_config(source_dir),
            load_config(OutcomePanelConfig, OUTCOME_PANEL_CONFIG),
            force=force,
        )
    except FileExistsError as exc:
        console.print(f'[bold red]{exc}[/bold red]')
        raise typer.Exit(code=1)

    typer.echo(f'Index-entry role table: {table_path}')

# -------------------------------------------------------------------------------------------------
```

- [x] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_index_role_table.py tests/unit/test_config.py tests/unit/test_cli_commands.py -q`
Expected: `80 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1370 passed, 1 skipped`.

Run: `git status --short conf/data`
Expected: only `?? conf/data/outcome_panel.yaml`. The tests draw their tables under `tmp_path`,
and `conf/data/index_roles.csv` must not exist before Task 9.

- [x] **Step 7: Lint and format**

Run: `./scripts/format_code.sh --check tests/unit/test_index_role_table.py tests/unit/test_config.py tests/unit/test_cli_commands.py src/naics_embedder/utils/config.py src/naics_embedder/data/index_role_table.py src/naics_embedder/cli/commands/data.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` On a failure, run the
same command without `--check`, re-run Step 6, and record the change as a deviation.

- [x] **Step 8: Commit**

```bash
git add tests/unit/test_index_role_table.py tests/unit/test_config.py tests/unit/test_cli_commands.py src/naics_embedder/utils/config.py conf/data/outcome_panel.yaml src/naics_embedder/data/index_role_table.py src/naics_embedder/cli/commands/data.py
git commit -m "feat(data): add data roles to draw the frozen index-entry role table"
```

### Task 8: The lexical stub encoder and `tools outcome-baseline`

The stub encoder needs no training:

- **Queries** are embedded as hashed character trigrams.
- **Codes** are embedded as hashed character trigrams of their title, description and examples
  channel.

It gives the panel a live number before Stage 6's encoder exists. `tools outcome-baseline`
scores it on the validation split, logging one read. It never opens the test split. It refuses
descriptions whose examples channel still holds every entry, because such a file would leak
the queries into the code texts.

**Files:**

- Create: `tests/unit/test_lexical_encoder.py`
- Modify: `tests/unit/test_cli_commands.py` (imports, append)
- Create: `src/naics_embedder/panels/lexical_encoder.py`
- Modify: `src/naics_embedder/cli/commands/tools.py` (docstring, imports, append)

**Interfaces:**

- Consumes:
  - Task 1: `normalize_text`.
  - Task 4: `OutcomePanel.from_files`, `OutcomePanel.score`, `SelectionLog` and the
    `QueryCodeEncoder` protocol.
  - Task 7: `OutcomePanelConfig.selection_log`.
- Produces in `naics_embedder.panels.lexical_encoder`:
  - `CODE_TEXT_COLUMNS = ('title', 'description', 'examples')`
  - `code_texts_from_descriptions(descriptions: pl.DataFrame) -> Dict[str, str]`
  - `LexicalTrigramEncoder(code_texts: Mapping[str, str], n_features: int = 4096)` with:
    - `encode_codes(codes) -> torch.Tensor`, which raises `ValueError` (`… codes have no
      text …`)
    - `encode_queries(texts) -> torch.Tensor`
- Produces in `naics_embedder.cli.commands.tools`: `naics-embedder tools outcome-baseline
  [--purpose TEXT] [--index-roles PATH] [--descriptions PATH] [--log PATH] [--output PATH]`.
  - It prints the summary.
  - It writes `{'fingerprint', 'summary'}` JSON with `--output`.
  - It exits 1 on missing or inconsistent inputs.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_lexical_encoder.py` with exactly this content:

```python
'''The training-free lexical stub encoder for the outcome panel.'''

import polars as pl
import pytest
import torch

from naics_embedder.panels.lexical_encoder import (
    LexicalTrigramEncoder,
    code_texts_from_descriptions,
)

pytestmark = pytest.mark.unit

def test_code_texts_join_title_description_and_examples_but_not_exclusions():
    descriptions = pl.DataFrame(
        {
            'code': ['111110', '112130'],
            'title': ['Soybean Farming', 'Dual-Purpose Cattle Ranching'],
            'description': ['This industry grows soybeans.', 'This industry raises cattle.'],
            'examples': ['Soybean farming, field', None],
            'excluded': ['Growing corn--are classified elsewhere.', None],
        }
    )

    assert code_texts_from_descriptions(descriptions) == {
        '111110': 'Soybean Farming This industry grows soybeans. Soybean farming, field',
        '112130': 'Dual-Purpose Cattle Ranching This industry raises cattle.',
    }

def test_vectors_are_unit_length_and_ignore_case_and_punctuation():
    encoder = LexicalTrigramEncoder({'111110': 'Soybean farming'}, n_features=256)

    codes = encoder.encode_codes(['111110'])
    queries = encoder.encode_queries(['SOYBEAN -- farming!', 'Tobacco'])

    assert codes.shape == (1, 256)
    assert codes.dtype == torch.float32
    torch.testing.assert_close(queries.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(queries[0], codes[0])

def test_codes_without_text_are_rejected():
    encoder = LexicalTrigramEncoder({'111110': 'Soybean farming'})

    with pytest.raises(ValueError, match='no text'):
        encoder.encode_codes(['111110', '112130'])
```

Modify `tests/unit/test_cli_commands.py` with these 3 edits, in order. Each replaced text occurs
exactly once in the file.

**`tests/unit/test_cli_commands.py`, edit 1 of 3.** Replace:

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner
```

with:

```python
import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner
```

**`tests/unit/test_cli_commands.py`, edit 2 of 3.** Replace:

```python
from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.supervision.artifacts import load_validated_bundle
```

with:

```python
from naics_embedder.cli.commands import tools as tools_cli
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.panels.selection_log import SelectionLog
from naics_embedder.supervision.artifacts import load_validated_bundle
```

**`tests/unit/test_cli_commands.py`, edit 3 of 3.** Replace:

```python
    assert 'relations path does not belong' in result.output
    assert verify_inputs == {}
```

with:

```python
    assert 'relations path does not belong' in result.output
    assert verify_inputs == {}

# -------------------------------------------------------------------------------------------------
# Outcome panel: lexical baseline
# -------------------------------------------------------------------------------------------------

BASELINE_ROLES = [
    (0, '111110', 'Soybean farming', 'examples'),
    (1, '111110', 'Edamame farming', 'validation'),
    (2, '111120', 'Canola farming', 'examples'),
    (3, '111120', 'Sunflower farming', 'validation'),
    (4, '111120', 'Rapeseed farming', 'test'),
]

def _baseline_inputs(tmp_path, examples):
    roles = tmp_path / 'naics_index_roles.parquet'
    descriptions = tmp_path / 'naics_descriptions.parquet'
    pl.DataFrame(
        BASELINE_ROLES,
        schema={
            'entry_id': pl.Int64,
            'code': pl.Utf8,
            'text': pl.Utf8,
            'role': pl.Utf8
        },
        orient='row',
    ).write_parquet(roles)
    pl.DataFrame(
        {
            'code': ['111110', '111120', '112130'],
            'title': ['Soybean Farming', 'Oilseed Farming', 'Dual-Purpose Cattle Ranching'],
            'description': ['Grows soybeans.', 'Grows oilseeds.', 'Raises cattle.'],
            'examples': examples,
            'excluded': [None, None, None],
        },
        schema_overrides={
            'excluded': pl.Utf8
        },
    ).write_parquet(descriptions)
    return [
        '--index-roles',
        str(roles),
        '--descriptions',
        str(descriptions),
        '--log',
        str(tmp_path / 'selection_log.jsonl'),
    ]

@pytest.mark.unit
def test_outcome_baseline_scores_validation_and_logs_one_read(runner, tmp_path):
    arguments = _baseline_inputs(tmp_path, ['Soybean farming', 'Canola farming', None])
    output = tmp_path / 'baseline.json'

    result = runner.invoke(tools_cli.app, ['outcome-baseline', *arguments, '--output', str(output)])

    assert result.exit_code == 0, result.output
    assert 'mrr' in result.output
    records = SelectionLog(tmp_path / 'selection_log.jsonl').records()
    assert [(r['event'], r['split'], r['n_queries'])
            for r in records] == [('read', 'validation', 2)]
    summary = json.loads(output.read_text())['summary']
    assert summary['n_queries'] == 2
    assert summary['n_candidates'] == 3

@pytest.mark.unit
def test_outcome_baseline_refuses_descriptions_that_hold_every_entry(runner, tmp_path):
    stale = [
        'Soybean farming; Edamame farming',
        'Canola farming; Sunflower farming; Rapeseed farming',
        None,
    ]

    result = runner.invoke(tools_cli.app, ['outcome-baseline', *_baseline_inputs(tmp_path, stale)])

    assert result.exit_code == 1
    assert 'Outcome baseline failed' in result.output
    assert SelectionLog(tmp_path / 'selection_log.jsonl').records() == []
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_lexical_encoder.py tests/unit/test_cli_commands.py -q`
Expected: a collection error,
`ModuleNotFoundError: No module named 'naics_embedder.panels.lexical_encoder'`.

- [x] **Step 3: Write the encoder**

Create `src/naics_embedder/panels/lexical_encoder.py` with exactly this content:

```python
'''
A training-free lexical encoder for the outcome panel.

Texts are hashed character trigrams of their normalized form (``leakage.normalize_text``),
L2-normalized, so cosine distance decodes a query to the code whose text shares most of its
trigrams. It is the stub arm that exercises the panel end to end on the real validation split
until a trained encoder can embed a query (roadmap Stage 6), and a floor for trained arms. It is
never a candidate configuration.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Dict, Iterable, Mapping, Sequence

import polars as pl
import torch
from sklearn.feature_extraction.text import HashingVectorizer

from naics_embedder.panels.leakage import normalize_text

CODE_TEXT_COLUMNS = ('title', 'description', 'examples')

# -------------------------------------------------------------------------------------------------
# Encoder
# -------------------------------------------------------------------------------------------------

def code_texts_from_descriptions(descriptions: pl.DataFrame) -> Dict[str, str]:
    '''
    Each code's title, description and examples channel, joined by spaces.

    Exclusion text is left out: it names activities the code does not cover.
    '''

    texts = {}
    for row in descriptions.select('code', *CODE_TEXT_COLUMNS).iter_rows(named=True):
        texts[row['code']] = ' '.join(row[name] for name in CODE_TEXT_COLUMNS if row[name])
    return texts

class LexicalTrigramEncoder:
    '''
    Hashed character-trigram vectors for codes (from their texts) and queries.

    Args:
        code_texts: The text each code is embedded from.
        n_features: Hash buckets per vector.
    '''

    def __init__(self, code_texts: Mapping[str, str], n_features: int = 4096):
        self.code_texts = dict(code_texts)
        self.vectorizer = HashingVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 3),
            n_features=n_features,
            alternate_sign=False,
            norm='l2',
            lowercase=False,
            preprocessor=normalize_text,
        )

    def _embed(self, texts: Iterable[str]) -> torch.Tensor:
        return torch.from_numpy(self.vectorizer.transform(list(texts)).toarray()).to(torch.float32)

    def encode_codes(self, codes: Sequence[str]) -> torch.Tensor:
        missing = [code for code in codes if code not in self.code_texts]
        if missing:
            raise ValueError(f'{len(missing):,} codes have no text, e.g. {missing[:5]}')
        return self._embed(self.code_texts[code] for code in codes)

    def encode_queries(self, texts: Sequence[str]) -> torch.Tensor:
        return self._embed(texts)
```

- [x] **Step 4: Add the command**

Modify `src/naics_embedder/cli/commands/tools.py` with these 3 edits, in order. Each replaced text
occurs exactly once in the file.

**`src/naics_embedder/cli/commands/tools.py`, edit 1 of 3.** Replace:

```python
    visualize: Generate visualizations from training log files.
    investigate: Analyze hierarchy preservation metrics.
'''

from pathlib import Path
from typing import Optional, Tuple

import typer
from rich.console import Console
```

with:

```python
    visualize: Generate visualizations from training log files.
    investigate: Analyze hierarchy preservation metrics.
    outcome-baseline: Score the lexical stub encoder on the outcome panel's validation split.
'''

import json
from pathlib import Path
from typing import Optional, Tuple

import polars as pl
import typer
from rich.console import Console
```

**`src/naics_embedder/cli/commands/tools.py`, edit 2 of 3.** Replace:

```python
    resolve_graph_supervision_paths,
)
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.embeddings_verification import Stage4VerificationConfig, verify_stage4
from naics_embedder.tools.metrics_tools import investigate_hierarchy, visualize_metrics
from naics_embedder.utils.console import configure_logging
```

with:

```python
    resolve_graph_supervision_paths,
)
from naics_embedder.panels.lexical_encoder import (
    LexicalTrigramEncoder,
    code_texts_from_descriptions,
)
from naics_embedder.panels.outcome import OutcomePanel
from naics_embedder.supervision.schema import IndexRole
from naics_embedder.tools.config_tools import show_current_config
from naics_embedder.tools.embeddings_verification import Stage4VerificationConfig, verify_stage4
from naics_embedder.tools.metrics_tools import investigate_hierarchy, visualize_metrics
from naics_embedder.utils.config import DownloadConfig, OutcomePanelConfig, load_config
from naics_embedder.utils.console import configure_logging
```

**`src/naics_embedder/cli/commands/tools.py`, edit 3 of 3.** Replace:

```python
        console.print('\n[bold red]✗ Stage 4 verification failed thresholds[/bold red]\n')
        raise typer.Exit(code=1)
```

with:

```python
        console.print('\n[bold red]✗ Stage 4 verification failed thresholds[/bold red]\n')
        raise typer.Exit(code=1)

# -------------------------------------------------------------------------------------------------
# Outcome panel: lexical baseline
# -------------------------------------------------------------------------------------------------

@app.command('outcome-baseline')
def outcome_baseline(
    purpose: Annotated[
        str,
        typer.Option('--purpose', help='Why this read happens; recorded in the selection log'),
    ] = 'lexical baseline on the validation split',
    index_roles: Annotated[
        Optional[str],
        typer.Option(
            '--index-roles',
            help='Index roles parquet from data preprocess (default: the download config)',
        ),
    ] = None,
    descriptions: Annotated[
        Optional[str],
        typer.Option(
            '--descriptions',
            help='Descriptions parquet from data preprocess (default: the download config)',
        ),
    ] = None,
    log: Annotated[
        Optional[str],
        typer.Option('--log', help='Selection log (default: the outcome-panel config)'),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option('--output', help='Also write the summary as JSON to this path'),
    ] = None,
):
    '''
    Score the training-free lexical encoder on the outcome panel's validation split.

    Decodes every validation query to the nearest six-digit code by hashed character trigrams
    under cosine distance, and reports top-1 accuracy, MRR, Hit@1/5/10 and the level of the
    lowest common ancestor. The read is logged in the selection log. The test split stays sealed:
    this command never opens it.

    Example:
        Score the baseline on the preprocessing outputs::

            $ uv run naics-embedder tools outcome-baseline
    '''

    configure_logging('tools_outcome_baseline.log')

    download_cfg = load_config(DownloadConfig, 'data/download.yaml')
    panel_cfg = load_config(OutcomePanelConfig, 'data/outcome_panel.yaml')
    descriptions_path = Path(descriptions or download_cfg.output_parquet)

    try:
        panel = OutcomePanel.from_files(
            index_roles or download_cfg.index_roles_parquet,
            descriptions_path,
            log or panel_cfg.selection_log,
        )
        encoder = LexicalTrigramEncoder(
            code_texts_from_descriptions(pl.read_parquet(descriptions_path))
        )
        result = panel.score(encoder, IndexRole.VALIDATION, purpose)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f'[bold red]Outcome baseline failed:[/bold red] {exc}')
        raise typer.Exit(code=1)

    console.print('\n[bold cyan]Outcome panel: lexical baseline, validation split[/bold cyan]\n')
    for key, value in result.summary.items():
        formatted = f'{value:.4f}' if isinstance(value, float) else str(value)
        console.print(f'  • {key}: {formatted}')
    console.print(f'\nRead logged to {panel.log.path}\n')

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'fingerprint': panel.fingerprint, 'summary': result.summary}
        path.write_text(json.dumps(payload, indent=2) + '\n')
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_lexical_encoder.py tests/unit/test_cli_commands.py -q`
Expected: `26 passed`.

Run: `uv run pytest -n auto -q`
Expected: `1375 passed, 1 skipped`.

- [x] **Step 6: Lint and format**

Run: `./scripts/format_code.sh --check tests/unit/test_lexical_encoder.py tests/unit/test_cli_commands.py src/naics_embedder/panels/lexical_encoder.py src/naics_embedder/cli/commands/tools.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` On a failure, run the
same command without `--check`, re-run Step 5, and record the change as a deviation.

- [x] **Step 7: Commit**

```bash
git add tests/unit/test_lexical_encoder.py tests/unit/test_cli_commands.py src/naics_embedder/panels/lexical_encoder.py src/naics_embedder/cli/commands/tools.py
git commit -m "feat(tools): add the lexical stub encoder and tools outcome-baseline"
```

### Task 9: Draw the real splits, check them, and record the finding (controller, inline)

This is the only task that reads the Census files. It draws the role table once and commits it,
with the provenance, a test that pins the table's hash, and the finding. Everything else it
writes is scratch:

- the rebuilt descriptions and index roles in this worktree's ignored `data/`
- the selection log and the regression script under `/tmp/stage2-outcome-panel-522fa329/`

Nothing touches the main checkout's `data/` except Step 8's read-only comparison. No bundle is
built.

**Files:**

- Create (generated by Step 3): `conf/data/index_roles.csv` and
  `conf/data/index_roles_provenance.json`
- Create: `tests/unit/test_committed_index_roles.py`
- Create: `specs/findings/outcome-panel-splits.md`

**Interfaces:**

- Consumes:
  - `naics-embedder data roles` (Task 7)
  - `naics-embedder data preprocess` (Task 5)
  - `naics-embedder tools outcome-baseline` (Task 8)
  - `read_role_table` and `role_table_fingerprint` (Task 2)
  - `sha256_file` (existing, `supervision/artifacts.py`)
- Produces: the frozen sealed splits that every later stage reads.
  - The committed-table test runs in CI without the Census files.
  - Its pinned hash fails on any redraw.

- [x] **Step 1: Write the committed-table test**

Create `tests/unit/test_committed_index_roles.py` with exactly this content:

```python
'''
The committed index-entry role table: the outcome panel's real sealed splits (Req 3; roadmap D4).

These read only ``conf/data/index_roles.csv`` and its provenance, so they run in CI without the
Census files. The pinned hash fails any accidental redraw: redrawing unseals the splits.
'''

import json
from pathlib import Path

import polars as pl
import pytest

from naics_embedder.panels.index_roles import read_role_table, role_table_fingerprint
from naics_embedder.supervision.artifacts import sha256_file

pytestmark = pytest.mark.unit

TABLE = Path('conf/data/index_roles.csv')
PROVENANCE = Path('conf/data/index_roles_provenance.json')
TABLE_SHA256 = '05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a'

@pytest.fixture(scope='module')
def roles() -> pl.DataFrame:
    return read_role_table(TABLE)

@pytest.fixture(scope='module')
def provenance() -> dict:
    return json.loads(PROVENANCE.read_text())

def test_every_index_entry_holds_exactly_one_role(roles):
    assert roles.height == 20_373
    assert roles.get_column('entry_id').n_unique() == 20_373
    assert roles.get_column('code').n_unique() == 1_010
    assert set(roles.get_column('role').unique()) == {'examples', 'training', 'validation', 'test'}

def test_the_entryless_codes_are_never_queries(roles):
    assert roles.filter(pl.col('code').is_in(['112130', '541120'])).height == 0

def test_every_code_keeps_an_examples_channel_entry(roles):
    with_examples = roles.filter(pl.col('role') == 'examples').get_column('code').n_unique()

    assert with_examples == roles.get_column('code').n_unique()

def test_the_table_is_the_one_its_provenance_describes(roles, provenance):
    assert sha256_file(TABLE) == TABLE_SHA256
    assert role_table_fingerprint(roles) == TABLE_SHA256
    assert provenance['role_table']['sha256'] == TABLE_SHA256
    assert dict(roles.group_by('role').len().iter_rows()) == provenance['roles']

def test_held_out_queries_were_checked_against_training_text(provenance):
    assert provenance['roles'] == {
        'examples': 6_118,
        'training': 7_200,
        'validation': 4_042,
        'test': 3_013,
    }
    assert provenance['eligibility']['withheld_exact'] == 1_539
    assert provenance['eligibility']['withheld_near_duplicate'] == 2_516
    assert provenance['held_out_leakage'] == {
        'validation': {
            'exact': 0,
            'near_duplicate': 0
        },
        'test': {
            'exact': 0,
            'near_duplicate': 0
        },
    }
```

- [x] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/test_committed_index_roles.py -q`
Expected: `5 errors`, starting with
`FileNotFoundError: No such file or directory (os error 2): conf/data/index_roles.csv`.

- [x] **Step 3: Draw the role table, once**

Run: `uv run naics-embedder data roles --source-dir ~/Downloads/Data`
Expected, in about 20 seconds, the log ends with these lines (Rich wraps them at the terminal
width):

```text
Index-entry roles: {'examples': 6118, 'training': 7200, 'validation': 4042, 'test': 3013}
Held-out eligibility: {'entries': 20373, 'exact_static': 768, 'near_duplicate_static': 813, 'exact_entry': 941, 'near_duplicate_entry': 2685, 'withheld_exact': 1539, 'withheld_near_duplicate': 2516, 'withheld': 4055, 'eligible': 16318}
Role table (05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a) written to: conf/data/index_roles.csv
Provenance written to: conf/data/index_roles_provenance.json

Index-entry role table: conf/data/index_roles.csv
```

Do not run `data roles` again on this branch, with or without `--force`. Record today's date
(YYYY-MM-DD) as the run date for Step 10.

- [x] **Step 4: Check the table's hash and size**

Run: `shasum -a 256 conf/data/index_roles.csv`
Expected: `05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a  conf/data/index_roles.csv`

Run: `wc -c conf/data/index_roles.csv`
Expected: `433147 conf/data/index_roles.csv`, with leading spaces.

If the hash differs, stop and ask. Do not commit, and leave both generated files in place for
your human partner.

- [x] **Step 5: Check the provenance**

Run: `uv run python -c "import json; p = json.load(open('conf/data/index_roles_provenance.json')); print(p['roles'], p['codes_with_role'], p['held_out_leakage'], p['eligibility'], p['fractions'], p['library_versions'], sep='\n')"`
Expected:

```text
{'examples': 6118, 'test': 3013, 'training': 7200, 'validation': 4042}
{'examples': 1010, 'test': 893, 'training': 988, 'validation': 939}
{'test': {'exact': 0, 'near_duplicate': 0}, 'validation': {'exact': 0, 'near_duplicate': 0}}
{'eligible': 16318, 'entries': 20373, 'exact_entry': 941, 'exact_static': 768, 'near_duplicate_entry': 2685, 'near_duplicate_static': 813, 'withheld': 4055, 'withheld_exact': 1539, 'withheld_near_duplicate': 2516}
{'examples': '3/10', 'test': '3/20', 'training': '7/20', 'validation': '1/5'}
{'numpy': '2.3.4', 'polars': '1.35.1', 'scikit-learn': '1.9.1'}
```

- [x] **Step 6: Run the committed-table test**

Run: `uv run pytest tests/unit/test_committed_index_roles.py -q`
Expected: `5 passed`.

- [x] **Step 7: Rebuild the descriptions in this worktree**

Run: `uv run naics-embedder data preprocess --source-dir ~/Downloads/Data`
Expected, in about 7 seconds:

- The log contains the line `Held-out queries matching training text: {'validation': {'exact':
  0, 'near_duplicate': 0}, 'test': {'exact': 0, 'near_duplicate': 0}}`.
- It reports 2,125 codes written to `./data/naics_descriptions.parquet` and 20,373 index entries
  written to `./data/naics_index_roles.parquet`.

Both paths are this worktree's ignored `data/`. The guard pins nothing here: both manifest paths
are `null` on this branch.

- [x] **Step 8: Compare the rebuilt descriptions with the pinned file**

Create `/tmp/stage2-outcome-panel-522fa329/regression_check.py` with the Write tool, with exactly
this content:

```python
'''Plan 4, Task 9: compare the rebuilt descriptions with the file bundle 18403d29 pins.'''

import hashlib
import sys
from pathlib import Path

import polars as pl

CANONICAL = Path('/Users/lowell/Projects/naics-embedder/data/naics_descriptions.parquet')
CANONICAL_SHA256 = '5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d'

digest = hashlib.sha256(CANONICAL.read_bytes()).hexdigest()
if digest != CANONICAL_SHA256:
    sys.exit(f'{CANONICAL} is not the pinned descriptions file ({digest})')

old = pl.read_parquet(CANONICAL)
new = pl.read_parquet('data/naics_descriptions.parquet')
roles = pl.read_parquet('data/naics_index_roles.parquet')
others = [name for name in old.columns if name != 'examples']
print('columns equal:', old.columns == new.columns)
print('rows:', old.height, new.height)
print('non-examples columns identical:', old.select(others).equals(new.select(others)))

joined = old.select('code', old_examples='examples').join(
    new.select('code', new_examples='examples'), on='code'
)
changed = joined.filter(~pl.col('old_examples').eq_missing(pl.col('new_examples')))
index_codes = set(roles.get_column('code').to_list())
print('codes with a changed examples channel:', changed.height)
print('all of them index codes:', set(changed.get_column('code').to_list()) <= index_codes)
subset = all(
    set((new_text or '').split('; ')) <= {item.strip() for item in (old_text or '').split('; ')}
    for _, old_text, new_text in changed.iter_rows()
)
print('rebuilt entries all among the old ones:', subset)
print(
    'codes with an examples channel (old, new):',
    old.get_column('examples').is_not_null().sum(),
    new.get_column('examples').is_not_null().sum(),
)
six_digit = new.filter(pl.col('level') == 6).get_column('code').to_list()
print('six-digit candidates:', len(six_digit), '112130' in six_digit, '541120' in six_digit)
```

Run: `uv run python /tmp/stage2-outcome-panel-522fa329/regression_check.py`
Expected:

```text
columns equal: True
rows: 2125 2125
non-examples columns identical: True
codes with a changed examples channel: 988
all of them index codes: True
rebuilt entries all among the old ones: True
codes with an examples channel (old, new): 1075 1075
six-digit candidates: 1012 True True
```

- [x] **Step 9: Score the stub encoder on the validation split**

Run: `uv run naics-embedder tools outcome-baseline --log /tmp/stage2-outcome-panel-522fa329/selection_log.jsonl --output /tmp/stage2-outcome-panel-522fa329/baseline.json`
Expected, in about 3 seconds:

```text
Outcome panel: lexical baseline, validation split

  • distance: cosine
  • n_queries: 4042
  • n_codes: 939
  • n_candidates: 1012
  • top1: 0.5163
  • mrr: 0.6165
  • hit_at_1: 0.5163
  • hit_at_5: 0.7333
  • hit_at_10: 0.7971
  • lca_level: 4.3355

Read logged to /tmp/stage2-outcome-panel-522fa329/selection_log.jsonl
```

Run: `cat /tmp/stage2-outcome-panel-522fa329/selection_log.jsonl`
Expected: exactly one line. Only its `time` differs from this:

```text
{"detail": {"distance": "cosine", "encoder": "LexicalTrigramEncoder"}, "event": "read", "fingerprint": "05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a", "n_queries": 4042, "panel": "outcome", "purpose": "lexical baseline on the validation split", "split": "validation", "time": "2026-09-24T18:24:40.289220+00:00"}
```

The test split stays sealed. Never call `open_test` on the real table in this plan.

- [x] **Step 10: Write the finding**

> Deviation: after the whole-plan review, 7b5e25d amended the finding: section 2 adds the segment-break limitation with an audit against unsplit texts (no test query in any whole title, description or exclusion text; 1 test and 4 validation queries span a sentence end or two examples-channel entries), and section 5 records that `OutcomePanel` alone enforces the seal.

Create `specs/findings/outcome-panel-splits.md` with exactly this content, with `<run date>`
replaced by Step 3's date:

````markdown
# Outcome panel and sealed splits: finding

**Status: FINAL (<run date>).** Roadmap Stage 2 (`specs/naics-embedding-roadmap.md`). This
finding records the real-data run of plan 4
(`specs/plans/completed/4-outcome-panel-sealed-splits.md`):

- the frozen index-entry role table
- the leakage check with its removal count (Verification "Leakage")
- the examples channel rebuilt from examples-role entries
- a training-free stub's scores on the validation split

Section 6 lists what later stages read.

## Sources

The four Census NAICS 2022 files were read from local copies in `~/Downloads/Data/` through
`--source-dir`, with nothing downloaded. The run used Python 3.12 with numpy 2.3.4, polars 1.35.1
and scikit-learn 1.9.1.

| File | sha256 | Used for |
|---|---|---|
| `2022_NAICS_Index_File.xlsx` (sheet `2022NAICS`) | `6506b37b9546dd9cec1f8b79e0b38b68e547a5cce5fd6f8332d35024dbd6cd63` | Index entries; pinned as `index_sha256` in `conf/data/download.yaml` |
| `2-6 digit_2022_Codes.xlsx` | `be12ba41002803359f49181c9bf33a03fbd08578f4f4a4c0bbad7aadaaea0316` | Codes and titles |
| `2022_NAICS_Descriptions.xlsx` | `6222c4d87dcf984970e0ff8a49862ed54b546b089d03956be3f285900cd3d66c` | Descriptions; fallback examples |
| `2022_NAICS_Cross_References.xlsx` | `3c50c3bfa9d76862aea471cc8831fd726f0b978619d833e48c54a749d9622144` | Exclusion text |

## 1. Index entries

- **Rows.** The index sheet has 20,398 rows:
  - 20,373 entries for 1,010 six-digit codes
  - 25 cross-reference ("see") rows, coded `******`, which name no code
- **Entry IDs.** `entry_id` is an entry's 0-based row position in the sheet. The entries are
  rows 0 to 20,372, and the "see" rows are the last 25.
- **Entry-less codes.** 112130 and 541120 have no entries. They are decoding candidates and never
  queries. The candidates are all 1,012 six-digit codes.
- **Entries per code.** The median is 13, the quartiles 7 and 24, and the maximum 320 (315250).
  22 codes have one entry, and 100 have at most three.
- **Cleaning.** 41 entries carried surrounding whitespace, stripped on reading. 4 contain
  non-ASCII characters, which normalization reads as word breaks.

## 2. Leakage: the rule and the removal count

**Normalization.** Texts are lowercased, keeping ASCII letters and digits, and every other run
of characters becomes one space.

**Matches.** A query leaks into a training segment in either of two ways:

- **Exact:** it occurs in the segment as whole words, equality included.
- **Near-duplicate:** its character-trigram Jaccard similarity with the segment is at least
  **9/10**, the stated similarity. The trigrams are scikit-learn's `char_wb`, and the comparison
  is in integers.

**Training text.** Every code's title and description sentences count. So do the exclusion
sentences, together with each cross-reference's activity phrase (Req 8), and the fallback
examples of the 65 codes without index entries. Every other index entry counts too.

**Eligibility.** An entry may become a validation or test query only if it matches none of that
text. Checking against every other entry, not only those that became training text, makes the
held-out splits leak-free whatever the draw.

| Entries that… | Count |
|---|---:|
| occur in static training text | 768 |
| near-duplicate static training text | 813 |
| occur in another index entry | 941 |
| near-duplicate another index entry | 2,685 |
| **are withheld for an exact match** | **1,539** |
| **are withheld as near-duplicates only (the removal count)** | **2,516** |
| are withheld in all | 4,055 |
| are eligible for validation or test | 16,318 |

The first four rows overlap. The two bold rows partition the withheld entries.

**Removal.** Verification "Leakage" asks that near-duplicates above the stated similarity be
removed from the test split and counted. Here they are removed before the draw, from both
held-out splits, because validation MRR selects (D6):

- 2,516 entries are withheld as near-duplicates only.
- 1,539 more are withheld for an exact match.

They become examples-channel text or training queries.

**Realized check.** After the draw, the 4,042 validation queries and the 3,013 test queries were
matched against all realized training text, including the rebuilt examples channels and the
7,200 training queries. Both splits had 0 exact matches and 0 near-duplicates (provenance
`held_out_leakage`).

**Limitation.** A query whose words appear reordered inside a longer segment is not caught.

## 3. Roles (D4)

**Quotas.** Per code, roles get largest-remainder quotas of examples 3/10, training 7/20,
validation 1/5 and test 3/20.

- Remainder ties are broken by draws seeded with (20260924, code).
- Every code with entries keeps at least one examples-channel entry.
- Held-out quotas are capped at the code's eligible entries, and the excess goes to training.

| Role | Entries | Share | Codes |
|---|---:|---:|---:|
| examples | 6,118 | 30.0 % | 1,010 |
| training | 7,200 | 35.3 % | 988 |
| validation | 4,042 | 19.8 % | 939 |
| test | 3,013 | 14.8 % | 893 |

**Coverage.**

- The 22 single-entry codes keep their entry as examples-channel text, so they have no queries.
- 71 codes have no validation query and 117 no test query. Their quotas round to zero, or
  they have too few eligible entries.

**The frozen table.**

- File: `conf/data/index_roles.csv`, with columns `entry_id`, `code` and `role`; 433,147 bytes.
- sha256: `05099381db725244b54ee263222933568f82fd5b212064697e8c7b2cfa6e8a3a`.
- Provenance: `conf/data/index_roles_provenance.json`.
- `data preprocess` applies the table and never redraws it. `data roles --force` would redraw it
  and unseal both splits.

## 4. The rebuilt examples channel

The rebuilt `data/naics_descriptions.parquet` was compared with the file bundle 18403d29 pins
(sha256 `5107fb8349ee8356ffe7670a3cfbbcc49e4b17f4f503bcdf1572c91c5dd39f2d`):

- **Unchanged.** Both have the same 2,125 codes and columns, and every column but `examples` is
  identical.
- **Changed.** `examples` differs for 988 codes, all with index entries. Each now holds only its
  examples-role entries, all of them among its old items. The other 22 codes with index entries
  are the single-entry codes, so their channel is unchanged.
- **Coverage.** 1,075 codes have an examples channel before and after: 1,010 from index entries
  and 65 from fallback examples.
- **No bundle.** No bundle was built. Bundle 18403d29 and the main checkout's `data/` are
  untouched, and the rebuilt files lived only in the plan's worktree. Stage 5 builds the first
  bundle that carries the `index_roles` member.

## 5. The lexical stub on the validation split

`naics-embedder tools outcome-baseline` embeds with hashed character trigrams (4,096 buckets):

- each query's text
- each code's title, description and rebuilt examples channel

It decodes under cosine distance over all 1,012 candidates.

| Metric | Value |
|---|---:|
| Queries / codes / candidates | 4,042 / 939 / 1,012 |
| Top-1 accuracy | 0.5163 |
| MRR | 0.6165 |
| Hit@1 / Hit@5 / Hit@10 | 0.5163 / 0.7333 / 0.7971 |
| Mean lowest-common-ancestor level | 4.3355 |

**The read.** The selection log's only record is this read:

- event `read`, panel `outcome`, split `validation`
- 4,042 queries
- fingerprint `05099381…`, the table's hash
- detail `{"distance": "cosine", "encoder": "LexicalTrigramEncoder"}`

The log was a scratch file, and this record is its copy.

**Sealing.** The test split was not opened: no `open` record exists for it. The logged
open-then-read path runs on fixture data in `tests/unit/test_outcome_panel.py`.

**What the numbers mean.** These are a floor, not memorization. The code texts hold
examples-role entries and never queries, and no held-out query matches any training text.

## 6. What later stages read

- **Stage 3** logs its reads to the same kind of file. `SelectionLog` takes any panel name.
- **Stage 4** resamples by code from `DecodingResult.per_query`, one row per query with its
  code.
- **Stage 5** does two things:
  - makes `index_roles` a required member in its contract version
  - builds the first bundle carrying it, from `data/naics_index_roles.parquet` through
    `SupervisionBuildConfig.index_roles_parquet`
- **Stage 6** implements `QueryCodeEncoder` and takes its first live reading with
  `OutcomePanel.score(encoder, 'validation', purpose)`.
- **Stage 7** does three things:
  - trains the query→code term on `OutcomePanel.training_queries()`
  - selects on validation MRR (D6), logged to `OutcomePanelConfig.selection_log`
  - opens the test split once, with `OutcomePanel.open_test`, for the final configuration

## Reproduction

The draw is deterministic, but it depends on numpy's generator streams and the leakage matcher's
libraries. The table is therefore frozen and committed rather than redrawn. To check it
reproduces:

1. Run `uv run naics-embedder data roles --force --source-dir ~/Downloads/Data` in a throwaway
   worktree at this commit.
2. Compare `shasum -a 256 conf/data/index_roles.csv` with the hash above.
3. Never commit a redraw.

The descriptions and the baseline reproduce with these two commands:

```bash
uv run naics-embedder data preprocess --source-dir ~/Downloads/Data
uv run naics-embedder tools outcome-baseline --log /tmp/selection_log.jsonl
```
````

- [x] **Step 11: Run the full suite and check the tree**

Run: `uv run pytest -n auto -q`
Expected: `1380 passed, 1 skipped`.

Run: `./scripts/format_code.sh --check tests/unit/test_committed_index_roles.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

Run: `git status --short conf specs/findings tests`
Expected: exactly these four lines. `data/` and `logs/` are ignored. The scope leaves out this
plan file, which shows as modified if you tick its boxes as you go.

```text
?? conf/data/index_roles.csv
?? conf/data/index_roles_provenance.json
?? specs/findings/outcome-panel-splits.md
?? tests/unit/test_committed_index_roles.py
```

- [x] **Step 12: Commit**

```bash
git add conf/data/index_roles.csv conf/data/index_roles_provenance.json tests/unit/test_committed_index_roles.py specs/findings/outcome-panel-splits.md
git commit -m "feat(data): commit the outcome panel's frozen index-entry role table and finding"
```

- [x] **Step 13: Remove the scratch files**

Run: `rm -rf /tmp/stage2-outcome-panel-522fa329`

The rebuilt parquets in this worktree's `data/` go when the worktree is removed.

### Task 10: Documentation

This task documents the new commands, adds the API page, and lists the new files in
`CLAUDE.md`'s tree.

**Files:**

- Modify: `docs/usage.md` (`data preprocess`, `data supervision`, a new `data roles` section, a
  new `tools outcome-baseline` section)
- Modify: `docs/.nav.yml` (the API page)
- Modify: `CLAUDE.md` (the directory tree and the command list)
- Create: `docs/api/outcome_panel.md`

**Interfaces:**

- Consumes: the commands and modules of Tasks 1–8.
- Produces: documentation only.

- [x] **Step 1: Update the usage guide**

Modify `docs/usage.md` with these 4 edits, in order. Each replaced text occurs exactly once in the
file.

**`docs/usage.md`, edit 1 of 4.** Replace:

````markdown
### `data preprocess`

Download and preprocess all raw NAICS data files.

**Generates:** `data/naics_descriptions.parquet`

```bash
uv run naics-embedder data preprocess
```

### `data supervision`
````

with:

````markdown
### `data preprocess`

Download and preprocess all raw NAICS data files. Each code's examples channel holds only its
examples-role index entries, per the committed role table (see `data roles`); the code's other
entries are outcome-panel queries. Preprocessing fails if any validation or test query matches
training text.

**Requires:** `conf/data/index_roles.csv`  
**Generates:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`

```bash
uv run naics-embedder data preprocess
```

**Options:**
- `--source-dir PATH` - Read the four Census files from this directory, by file name, instead of
  downloading them
- `--force` - Overwrite a descriptions file that a configured supervision bundle pins. Refused by
  default: training against that bundle fails closed once the file changes

### `data supervision`
````

**`docs/usage.md`, edit 2 of 4.** Replace:

```markdown
written last and the bundle directory is published atomically.

**Requires:** `data/naics_descriptions.parquet`  
**Generates:** `data/supervision/stage3-supervision-v1/<bundle-id>/` (prints
`Supervision manifest: <path>`; set `supervision.manifest_path` to it before training)
```

with:

```markdown
written last and the bundle directory is published atomically.

**Requires:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet` (carried as the
bundle's optional `index_roles` member)  
**Generates:** `data/supervision/stage3-supervision-v1/<bundle-id>/` (prints
`Supervision manifest: <path>`; set `supervision.manifest_path` to it before training)
```

**`docs/usage.md`, edit 3 of 4.** Replace:

````markdown
```bash
uv run naics-embedder data all
```
````

with:

````markdown
```bash
uv run naics-embedder data all
```

### `data roles`

Draw the frozen index-entry role table, once. Every Census index entry gets exactly one role, per
code and stratified: examples-channel text, or a training, validation or test query for the
outcome panel. No validation or test query matches any training text, exactly or as a
near-duplicate (character-trigram Jaccard of at least 0.9). The table is committed and `data
preprocess` applies it. Redrawing it unseals the validation and test splits, so an existing table
is replaced only with `--force`.

**Generates:** `conf/data/index_roles.csv`, `conf/data/index_roles_provenance.json`

```bash
uv run naics-embedder data roles --source-dir ~/Downloads/Data
```
````

**`docs/usage.md`, edit 4 of 4.** Replace:

```markdown
- `--distance-matrix PATH` - Path to ground truth distance matrix (default: `data/naics_distance_matrix.parquet`)
- `--config PATH` - Path to config file (default: `conf/config.yaml`)

---
```

with:

````markdown
- `--distance-matrix PATH` - Path to ground truth distance matrix (default: `data/naics_distance_matrix.parquet`)
- `--config PATH` - Path to config file (default: `conf/config.yaml`)

### `tools outcome-baseline`

Score the training-free lexical encoder (hashed character trigrams under cosine distance) on the
outcome panel's validation split: top-1 accuracy, MRR, Hit@1/5/10 and the level of the lowest
common ancestor of the top-1 code and the truth. The read is appended to the selection log; the
test split stays sealed.

**Requires:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`

```bash
uv run naics-embedder tools outcome-baseline
```

**Options:**
- `--purpose TEXT` - Why this read happens; recorded in the selection log
- `--index-roles PATH`, `--descriptions PATH` - Preprocessing outputs (default: the paths in
  `conf/data/download.yaml`)
- `--log PATH` - Selection log (default: `logs/selection_log.jsonl`, from
  `conf/data/outcome_panel.yaml`)
- `--output PATH` - Also write the summary as JSON

---
````

- [x] **Step 2: Add the API page**

Create `docs/api/outcome_panel.md` with exactly this content:

```markdown
# Outcome Panel API

Sealed text-to-code decoding splits over the Census index entries (roadmap Stage 2).

## Index-entry roles

::: naics_embedder.panels.index_roles

::: naics_embedder.data.index_role_table

## Leakage

::: naics_embedder.panels.leakage

## Decoding

::: naics_embedder.panels.decoding

## Panel and selection log

::: naics_embedder.panels.outcome

::: naics_embedder.panels.selection_log

## Lexical baseline

::: naics_embedder.panels.lexical_encoder
```

Modify `docs/.nav.yml` with one edit. The replaced text occurs exactly once in the file.

**`docs/.nav.yml`, edit 1 of 1.** Replace:

```yaml
          - Graph: api/graph_metrics.md
          - Hierarchy Structure: api/hierarchy_structure_metrics.md
          - QCEW: api/qcew_metrics.md
          - Runner: api/runner.md
```

with:

```yaml
          - Graph: api/graph_metrics.md
          - Hierarchy Structure: api/hierarchy_structure_metrics.md
          - Outcome Panel: api/outcome_panel.md
          - QCEW: api/qcew_metrics.md
          - Runner: api/runner.md
```

- [x] **Step 3: Update CLAUDE.md**

Modify `CLAUDE.md` with these 5 edits, in order. Each replaced text occurs exactly once in the file.

**`CLAUDE.md`, edit 1 of 5.** Replace:

```markdown
│   ├── data/                 # Data preprocessing and generation
│   │   ├── download_data.py  # Download and preprocess NAICS data
│   │   ├── compute_relations.py   # Compute relationship measures
│   │   ├── compute_distances.py   # Compute graph distance measures
```

with:

```markdown
│   ├── data/                 # Data preprocessing and generation
│   │   ├── download_data.py  # Download and preprocess NAICS data
│   │   ├── index_role_table.py    # Draw the frozen index-entry role table (data roles)
│   │   ├── compute_relations.py   # Compute relationship measures
│   │   ├── compute_distances.py   # Compute graph distance measures
```

**`CLAUDE.md`, edit 2 of 5.** Replace:

```markdown
│   │       ├── datamodule.py          # PyTorch Lightning DataModule
│   │       └── tokenization_cache.py  # Disk-based tokenization cache
│   ├── graph_model/          # Stage 4: HGCN refinement
│   │   ├── hgcn.py           # Hyperbolic graph convolutional network
```

with:

```markdown
│   │       ├── datamodule.py          # PyTorch Lightning DataModule
│   │       └── tokenization_cache.py  # Disk-based tokenization cache
│   ├── panels/               # Sealed evaluation panels (roadmap Stage 2 onward)
│   │   ├── leakage.py        # Exact and near-duplicate matching against training text
│   │   ├── index_roles.py    # Index-entry roles: quotas, eligibility, the frozen table
│   │   ├── decoding.py       # Text-to-code decoding scores (top-1, MRR, Hit@k, LCA level)
│   │   ├── selection_log.py  # Append-only log of split reads and test-split openings
│   │   ├── outcome.py        # OutcomePanel: sealed validation and test query splits
│   │   └── lexical_encoder.py  # Training-free trigram stub encoder
│   ├── graph_model/          # Stage 4: HGCN refinement
│   │   ├── hgcn.py           # Hyperbolic graph convolutional network
```

**`CLAUDE.md`, edit 3 of 5.** Replace:

```markdown
│   ├── data/                 # Data generation configs
│   │   ├── download.yaml
│   │   ├── relations.yaml
│   │   ├── distances.yaml
```

with:

```markdown
│   ├── data/                 # Data generation configs
│   │   ├── download.yaml
│   │   ├── outcome_panel.yaml     # Role fractions, seed, near-duplicate threshold, selection log
│   │   ├── index_roles.csv        # The frozen index-entry role table (committed)
│   │   ├── relations.yaml
│   │   ├── distances.yaml
```

**`CLAUDE.md`, edit 4 of 5.** Replace:

```markdown
uv run naics-embedder data all         # Run all data preparation steps
# (data relations / distances / triplets are deprecated and build the same bundle)

# Training commands
```

with:

```markdown
uv run naics-embedder data all         # Run all data preparation steps
# (data relations / distances / triplets are deprecated and build the same bundle)
# (data roles drew conf/data/index_roles.csv once; it is committed, and preprocess applies it)

# Training commands
```

**`CLAUDE.md`, edit 5 of 5.** Replace:

````markdown
uv run naics-embedder tools visualize  # Visualize training metrics
uv run naics-embedder tools investigate  # Investigate hierarchy correlation
```
````

with:

````markdown
uv run naics-embedder tools visualize  # Visualize training metrics
uv run naics-embedder tools investigate  # Investigate hierarchy correlation
uv run naics-embedder tools outcome-baseline  # Lexical stub on the outcome validation split
```
````

- [x] **Step 4: Build the docs strictly**

Run: `uv run mkdocs build --strict --site-dir /tmp/stage2-outcome-panel-docs-522fa329`
Expected: exit 0 and `Documentation built in`. The output has no `WARNING` line.

Run: `grep -c generate_index_role_table /tmp/stage2-outcome-panel-docs-522fa329/api/outcome_panel/index.html`
Expected: a count of at least 1. It shows the new page rendered its modules.

Run: `rm -rf /tmp/stage2-outcome-panel-docs-522fa329`

- [x] **Step 5: Commit**

```bash
git add docs/usage.md docs/api/outcome_panel.md docs/.nav.yml CLAUDE.md
git commit -m "docs: document data roles, tools outcome-baseline and the outcome panel API"
```

## Final verification (controller, inline)

- [x] **Step 1: Full suite on Python 3.12**

> Deviation: 1382 passed, 1 skipped after review commit ecf748d added two tests (1380 before it).

Run: `uv run pytest -n auto -q`
Expected: `1380 passed, 1 skipped`.

- [x] **Step 2: Full suite on Python 3.10, CI's other leg**

> Deviation: likewise 1382 passed, 1 skipped on Python 3.10 after ecf748d (1380 before it).

Run: `UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310-522fa329 uv run pytest -n auto -q`
Expected: `1380 passed, 1 skipped`, the same as Step 1. On a machine without MPS, the MPS test in
`test_outcome_decoding.py` also skips. This leg locks numpy 2.2.6 and scikit-learn 1.7.2. The
committed-table test reads the CSV, so it passes without redrawing.

Run: `rm -rf /tmp/naics-py310-522fa329`

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

Run: `git diff --stat origin/main..HEAD -- conf/config.yaml conf/graph.yaml`
Expected: no output.

- [x] **Step 5: The roadmap's Stage 2 Exit, outcome by outcome**

Check each row against the test that backs it:

| Exit outcome | Tests |
|---|---|
| Every index entry holds exactly one role | `test_committed_index_roles.py::test_every_index_entry_holds_exactly_one_role`; `test_outcome_panel.py::test_every_entry_holds_one_role_and_entryless_codes_are_never_queries`; `test_supervision_artifacts.py::test_loader_rejects_an_index_entry_with_two_roles` |
| The two entry-less codes never appear as queries | `test_committed_index_roles.py::test_the_entryless_codes_are_never_queries`; `test_outcome_panel.py::test_every_entry_holds_one_role_and_entryless_codes_are_never_queries` |
| The leakage check finds no exact match and reports the near-duplicates removed | `test_committed_index_roles.py::test_held_out_queries_were_checked_against_training_text`; `test_index_role_table.py::test_provenance_records_the_draw_and_its_checks`; `test_index_roles.py::test_eligibility_withholds_exact_and_near_duplicate_matches` |
| The scorer reports all four metrics for a stub encoder on the sealed splits | `test_outcome_panel.py::test_a_stub_encoder_scores_every_metric_on_both_splits`; Task 9 Step 9 on the real validation split |
| The selection log exists, and the test split cannot be read without a logged open | `test_outcome_panel.py::test_the_test_split_is_sealed_until_a_logged_opening`, `::test_every_panel_object_must_open_the_test_split_itself` and `::test_a_second_opening_needs_a_reason_and_is_logged_as_a_reopen` |

Run: `uv run pytest tests/unit/test_committed_index_roles.py tests/unit/test_outcome_panel.py tests/unit/test_index_role_table.py tests/unit/test_index_roles.py tests/unit/test_supervision_artifacts.py -q`
Expected: `91 passed`.

## Plan completion

Run the Plan Completion Protocol of writing-plans after the final review. The completion commits
are the branch's last commits. Before editing `specs/naics-embedding-roadmap.md` or
`specs/deferred_items.md`, check whether another Claude session is active in this repository. If
one is, hold both edits and hand your human partner the exact text below.

- [x] **Step 1: Tick the roadmap stage and add the rollout note and the stamp**

In `specs/naics-embedding-roadmap.md`, make one edit. Replace:

```markdown
- [ ] Stage 2: Outcome panel and sealed splits
```

with:

```markdown
- [x] Stage 2: Outcome panel and sealed splits
```

Make a second edit. Replace the Stage 2 entry's last line:

```markdown
      logged open.
      ROUTING: writing-plans

- [ ] Stage 3: Regressor panel
```

with the lines below, with `YYYY-MM-DD` replaced by the completion date:

```markdown
      logged open.
      ROUTING: writing-plans
      Rollout note (D4): per code, largest-remainder quotas of examples 3/10, training 7/20,
      validation 1/5 and test 3/20; remainder ties broken by a draw seeded with (20260924,
      code); at least one examples-channel entry for every code with entries; held-out quotas
      capped at the code's leak-free entries, the excess to training. Realized: 6,118 / 7,200 /
      4,042 / 3,013 entries (`conf/data/index_roles.csv`, sha256 05099381…).
      Stage 2: COMPLETE (YYYY-MM-DD) — implemented by plan 4
      (specs/plans/completed/4-outcome-panel-sealed-splits.md). Next: resume the roadmap.

- [ ] Stage 3: Regressor panel
```

- [x] **Step 2: Re-validate the later stages against what shipped**

Two later entries name what Stage 2 left open, so each gets one line.

In the Stage 5 entry, replace:

```markdown
      Consumes: Stage 2's index-entry roles (the examples channel holds examples-role entries
      only). The current bundle contract (`data/supervision_bundle.py`,
```

with:

```markdown
      Consumes: Stage 2's index-entry roles (the examples channel holds examples-role entries
      only). Stage 2 shipped them as an optional `index_roles` member under
      stage3-supervision-v1 and the rebuild in `data preprocess`, but built no bundle: this
      stage's contract version makes the member required, and its rebuild is the first bundle
      carrying both. The current bundle contract (`data/supervision_bundle.py`,
```

In the Stage 7 entry, replace:

```markdown
      flags; Stage 2's query splits, scorer and selection log; Stage 3's panel; Stage 4's
      seed-sweep driver, decision tooling and δ procedure.
```

with:

```markdown
      flags; Stage 2's query splits, scorer and selection log (the log's path is
      `OutcomePanelConfig.selection_log`, `logs/selection_log.jsonl`: gitignored, so a
      worktree's log goes with the worktree); Stage 3's panel; Stage 4's seed-sweep driver,
      decision tooling and δ procedure.
```

Stages 3, 4 and 6 need no edit:

- Stage 3 consumes the selection log, and `SelectionLog` takes any panel name.
- Stage 4 consumes `DecodingResult.per_query`, one row per query with its code.
- Stage 6 consumes `OutcomePanel.score` with a `QueryCodeEncoder`.

Commit both roadmap edits with the plan markup in Step 3's commit.

- [x] **Step 3: Mark up this plan and resolve the gate**

Follow the protocol:

- Run the resolve-before-defer gate.
- Tick every completed step and add `> Deviation:` notes.
- Add the status header.
- Tick any earlier deferred item this plan implemented. None is expected: the roadmap assigns
  the open items to Stages 5 and 7 or leaves them standalone.
- Append this plan's deferred items, if any.
- Run `deferred_stats.py` and surface its summary line.

Then commit:

```bash
git add specs/naics-embedding-roadmap.md specs/plans/4-outcome-panel-sealed-splits.md specs/deferred_items.md
git commit -m "docs(roadmap): complete Stage 2 and re-validate Stages 5 and 7"
```

`git add` of an unchanged `specs/deferred_items.md` is harmless.

- [x] **Step 4: Retire the plan**

```bash
git mv specs/plans/4-outcome-panel-sealed-splits.md specs/plans/completed/4-outcome-panel-sealed-splits.md
git commit -m "chore(specs): retire plan 4"
```

This plan has no relative links to re-point, and no spec file retires with it: Stage 2 has no
stage spec.

- [x] **Step 5: Integrate**

Hand over to finishing-a-development-branch. Before opening any PR, check two things:

- `git log --oneline origin/main..HEAD` shows only this branch's commits.
- The diff contains no `conf/config.yaml` or `conf/graph.yaml` change.

Never push to `main`.
