# Employment-Statistics Coverage (Roadmap Stage 1) Implementation Plan

**Status: COMPLETE (2026-09-24)** — executed via executing-plans; deferred items in specs/deferred_items.md (one entry, the national-total rounding allowance in `check_invariants`, handed to the project owner to add because parallel sessions may be active)

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-embedding-roadmap.md, Stage 1 — on plan completion, tick the stage and re-validate later stages against what shipped.

**Goal:** Resolve Req 2's (open) item by verification against BLS files. Record which reference
years and grains publish NAICS 2022 six-digit QCEW cells and how much disclosure suppression
removes per year, grain and series, then name the pre-specified Req 2 branch, population and row
grain that Stage 3 takes.

**Architecture:** One analysis script, `scripts/employment_statistics_coverage.py`, reads the QCEW
annual-average single files for 2022–2025, the Open Data Access national slices for 2021–2025,
and the supervision bundle's codebook, whose 1,012 six-digit codes are the universe.

- It classifies every published private cell as disclosed, suppressed or other.
- It recovers the NAICS 238 codes that QCEW splits into BLS residential and nonresidential codes,
  reading each from its five-digit parent.
- It checks reconciliation invariants and applies the decision rule this plan pre-registers.

A fixture test in `tests/unit/` keeps the script honest in CI. The finding,
`specs/findings/employment-statistics-coverage.md`, records the four Verification items with
source files, hashes and read dates. The roadmap then gets Stage 1 ticked and stamped. Nothing
changes in `src/`, the CLI or the configuration.

**Tech Stack:** Python 3.10+ (CI tests 3.10 and 3.12), Polars 1.35 (locked), stdlib `zipfile`,
`hashlib`, `csv` and `argparse`, pytest, ruff, yapf. Downloads use `curl`. BLS web pages refuse
curl with HTTP 403, so they are read in the built-in browser. No new dependencies.

**Workspace (checked in Pre-flight):**

- Worktree: `/Users/lowell/Projects/naics-embedder/.claude/worktrees/employment-statistics-coverage`
- Branch: `claude/employment-statistics-coverage-447c9d1f`, cut from `origin/main` at `6b2d938`
  (PR #107 merged). This plan's commit is on top.
- Local `main` carries two held commits, "config" and "graph config". They must never enter this
  branch.
- The codebook sits in the main checkout's ignored `data/`, not in this worktree. Read it by
  absolute path:
  `/Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet`
  - sha256 `5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b`, the value the
    bundle manifest records under `artifacts.codebook`.
- BLS files go to `~/Downloads/Data/QCEW/` and run outputs to `~/Downloads/Data/QCEW/coverage/`.
  Both sit outside every worktree, so archiving this session cannot delete them.
- Bash in this worktree refuses heredocs, loops and long `&&` chains. Run one plain command per
  call, and create files with the Write and Edit tools.

## Global Constraints

The quoted lines are copied verbatim from `specs/naics-embedding.md` at d9126ce (unchanged
through `origin/main` 6b2d938) and from `specs/naics-embedding-roadmap.md` (APPROVED
2026-09-23). Those files stay authoritative.

- Req 2, the (open) item: "Outcomes come from public employment statistics for NAICS 2022 codes.
  Which series, years and grains are usable is (open — resolved by verification, not argument):
  which reference years are published on NAICS 2022 codes at six digits, at which grains (by
  year, by area), and how much disclosure suppression removes at each. The branches are
  specified in advance:"
  - "If the verified window supports a time-respecting outcome (the outcome dated after the
    features, with splits by time), the panel includes one (ChatGPT C15; Claude C28)."
  - "If it does not, the panel is cross-sectional and records why."
  - "If no grain below the code survives suppression, the panel is held-out-codes only and
    records that the one-hot comparison could not run."
  - "Training on changes coded on an earlier NAICS vintage (Claude C28) is not used without a
    concordance (rejected here; cross-vintage work is Out of scope)."
- Req 2, rows: "Rows sit below the code. A row is a code in a given year, or in a given area
  (ChatGPT C15; Claude C28)."
- Req 2, regimes: "Seen codes: rows are split by year or area, so every code appears in
  training. This is where one-hot is a real competitor." and "Held-out codes: every row of a
  held-out code group leaves training, with four-digit parents as groups."
- Verification "Employment-statistics coverage (discharges Req 2's (open))", which says to record:
  - "the reference years published on NAICS 2022 codes at six digits;"
  - "the grains (by year, by area) at which six-digit series are published;"
  - "for each year, grain and series, the share of six-digit codes suppressed;"
  - "the resulting regressor-panel population and row grain, whether the panel includes a
    time-respecting outcome, and whether the seen-code regime can run, with reasons for any
    "no"."
- Rollout note, first paragraph: "Resolve the employment-statistics (open) item first. It fixes
  the regressor panel's population and whether the panel has a time-respecting outcome, and
  Req 5 needs both panels before anything is adopted. The backbone-window (open) item resolves
  alongside the backbone choice (Req 14)."
- Roadmap D1: "covariates are log establishment counts and log wages from the same QCEW rows, as
  in the methodology's definition (`metrics/qcew.py`)." Every grain the finding reports
  therefore carries the suppression of establishments and wages alongside employment's.
- Roadmap Stage 1, Consumes: "The bundle codebook (2,125 codes, 1,012 six-digit) as the code
  universe; `metrics/qcew.py` as prior art only (its 2022, private, one-row-per-code slice is
  the rejected definition)."
- Roadmap Stage 1, Exit: "The four items of Verification "Employment-statistics coverage" are
  recorded with the source files and the dates they were read; the chosen Req 2 branch is
  named."

Rulings and project rules:

- The roadmap partition and decisions D1–D6 are settled. Do not re-derive or re-ask them.
- Verification, not argument. Every number in the finding comes from the script run on the
  downloaded BLS files. Every documentary claim comes from a BLS page or file read during
  execution, quoted verbatim with its read date.
- A WebFetch summary is a lead, never evidence. During planning one claimed that all 19 NAICS
  238 six-digit codes are split. QCEW's count of 1,030 industries implies 17.
- A suppressed cell is never read as zero, and suppressed detail is never summed.
- `area_fips`, `own_code`, `industry_code` and `agglvl_code` stay strings.
- The decision rule below is fixed before any data is read. So are its parameters:
  `SURVIVAL_FLOOR`, `ASK_BAND`, `MIN_WINDOW_YEARS` and `CANDIDATE_GRAINS`. Never change them
  after a run. When a stop-and-ask condition fires, stop and ask your human partner.
- Downloads need your human partner's explicit permission first, asked in Pre-flight step 4.
  Use the generic User-Agent given in Task 5, and never put the user's email in a request.
- No `src/` changes, no CLI command, no config edits, no edits to `metrics/qcew.py`, and no new
  dependencies.
- Do not edit `specs/deferred_items.md`: parallel sessions may be active. If the Plan Completion
  Protocol would defer anything, ask your human partner and hand over the exact text instead.
- Never push to `main`. The branch integrates by PR through finishing-a-development-branch. Run
  `git log --oneline origin/main..HEAD` before any push; it must list only this branch's
  commits.
- The finding lives at `specs/findings/employment-statistics-coverage.md`, not at the roadmap's
  default `reports/…` path.
  - `/reports/` is gitignored (`.gitignore:78`), so a finding there could never reach a PR, and
    archiving a worktree deletes its ignored files.
  - Task 8 updates the roadmap's Stage 1 **Produces** line to the new path.
- Retirement (Plan Completion Protocol step 5) moves only this plan.
  - `specs/naics-embedding.md` and the roadmap are shared by all eleven stages and stay put.
  - No `specs/employment-statistics-coverage.md` exists, so the protocol's spec-retire clause
    does not fire. The finding stays in `specs/findings/`.
- Python style follows CLAUDE.md:
  - single quotes and one blank line between top-level definitions;
  - section dividers, type hints on the script's functions, and `logging`, never `print`.
- Format with `./scripts/format_code.sh <paths>`; never run `ruff format`. CI lints only `src/`
  and `tests/`, so every code task lints the script by explicit path.
- Every commit ends with the `Co-Authored-By` trailer that the executing session's instructions
  specify.

## Facts verified during planning (2026-09-24)

Tasks 5 and 6 re-verify each fact during execution, and the finding cites the execution-time
reading.

| Fact | Where it was read |
| --- | --- |
| QCEW introduced NAICS 2022 on 2022-09-07, "with the full data release of first quarter 2022" QCEW data. Data for 2017–2021 are coded on NAICS 2017, and data from 2022 on NAICS 2022 | `https://www.bls.gov/cew/classifications/industry/naics-2022.htm`; `https://www.bls.gov/cew/classifications/industry/`; QCEW news-release notes (2022-08-24 entry) |
| QCEW's NAICS 2022 has 21 sectors and 1,030 industries, including BLS-specific residential and nonresidential codes within NAICS 238, and 999999 (unclassified) | `naics-2022.htm` |
| Annual single files exist for 2022–2025, sizes and dates as in Task 5; for 2026 only a first-quarter file exists | HEAD requests to `data.bls.gov` |
| "Final quarterly and annual averages data for each year will be available with the release of first quarter data (preliminary) for the subsequent year." Q1 2026 was released 2026-08-28, so 2022–2025 are final | `https://www.bls.gov/cew/release-calendar.htm` |
| The 2025 files carry Last-Modified 2026-08-21, a week before that release. Record both dates and do not explain the gap away | HEAD requests; release calendar |
| From third-quarter 2025 data (released 2026-03-10), MSA data are published at total covered employment only | `https://www.bls.gov/cew/notices/2025/change-in-the-presentation-of-metropolitan-statistical-area-data-in-qcew.htm` (notice dated 2025-12-30) |
| From 2024 data, Connecticut's planning regions `09110`–`09190` replace its eight legacy counties, and MSAs move from OMB Bulletin 13-01 to 23-01 | bls-data-context `references/qcew.md`; news-release notes (2024-08-21 entry) |
| Aggregation levels: 18/17 national six-/five-digit by ownership; 58/57 statewide; 78/77 county; 48/47 MSA, private only; 11 national by ownership. CSA (30) and MicroSA (80) carry no industry detail | `https://www.bls.gov/cew/classifications/aggregation/agg-level-titles.htm`; Task 5 confirms from `agglevel_titles.csv` |
| On an `N` row, employment and wages are zero-filled while the establishment count stays positive. An undocumented `-` code also occurs | A 2015 Q1 Alabama slice in `~/Downloads/Data/01000.csv` (636 `N` rows, all with positive establishments; 5 `-` rows). The vintage is old, so Task 6 re-checks every 2022–2025 file; the finding never cites this slice |
| The codebook has 1,012 six-digit codes. 19 are in NAICS 238, each its five-digit parent's only child | the bundle codebook |
| `metrics/qcew.py` reads `tot_wages`; the annual files name the column `total_annual_wages` | the prior art against the BLS layout |

## Pre-registered decision rule

This rule is fixed now, before any data is read. `decide()` in Task 3 implements it, and its
tests pin it.

- **Universe.** The 1,012 six-digit codes in the bundle codebook. Every code-level denominator is
  1,012. Absent codes are counted, never dropped.
- **Cell status.** A published cell is one row of an annual file at a grain's six-digit level;
  for a split code, its five-digit parent's row.
  - A blank disclosure code is `disclosed`, `N` is `suppressed`, and any other code (such as `-`)
    is `other`.
  - Employment and wages share one disclosure flag.
  - On an `N` row the establishment count counts as disclosed when it is positive.
  - A code with no cell is `absent`.
- **Split codes.** These are codebook codes QCEW does not publish at six digits, replacing each
  with BLS residential (`xxxxx1`) and nonresidential (`xxxxx2`) codes.
  - Each split code must be its five-digit parent's only child.
  - Its value and status come from the parent's QCEW row, never summed from the children.
- **Ownership.** Private, `own_code` 5.
  - QCEW publishes six-digit cells by ownership only, MSA six-digit cells are private only, and
    summing ownerships is invalid wherever one component is suppressed.
  - Codes with no private cell are listed with the ownerships that have one.
- **Usable row.** A private cell whose employment and wages are disclosed.
- **Grains.**
  - national (18/17);
  - state (58/57, every statewide area published, Puerto Rico and the Virgin Islands included);
  - county (78/77, excluding county part `999`, "unknown or undefined");
  - MSA (48/47).
  - National, state and county are candidates. MSA is tabulated but never a candidate: it has no
    six-digit rows from 2025, and its codes break between 13-01 and 23-01.
- **Window.** 2022–2025, every NAICS 2022 year with an annual file. `--final-years` lists the
  years that BLS's finality rule makes final on the run date, expected to be all four.
- **Survival.** A candidate grain survives when it has six-digit rows in every window year and at
  least `SURVIVAL_FLOOR = 506` codes (50 % of 1,012) can run the seen-code regime.
  - A code can run the regime if it is usable in the last year and in an earlier one (a split by
    year).
  - At an area grain, it can also run the regime if it is usable in two or more areas in the last
    year (a split by area).
- **Choice.** Among surviving candidates, the rule picks the one with the lowest mean share of
  suppressed private cells over the window. Ties break national, state, county. When no grain
  survives, the branch is C.
- **Time-respecting outcome.** The outcome is time-respecting when both hold:
  - the window has at least `MIN_WINDOW_YEARS = 3` consecutive final years;
  - at least 506 codes at the chosen grain are time-eligible. A code is time-eligible when a
    same-area usable pair (t, t+1) exists both for the last pair (2024→2025) and for an earlier
    pair.
  - Yes gives branch A; no gives branch B.
- **Stop and ask.** `needs_user` is set when a deciding count falls in `ASK_BAND = 405–607`
  (40–60 %). The script then exits 2, and the executor stops and asks. The deciding counts are:
  - the chosen grain's seen count;
  - the seen count of any better-ranked candidate;
  - the best seen count when no grain survives;
  - the chosen grain's time-eligible count.
- **Reported but not read by the rule:**
  - every ownership at the national grain;
  - the MSA grain;
  - the establishment counts of suppressed versus disclosed cells, which show whether
    suppression removes small cells (if it does, dropping them truncates the outcome).

## Stop-and-ask conditions

Beyond `needs_user`, stop, report and ask your human partner when any of these holds. Do not
work around it.

1. The script exits non-zero, or `coverage.json` lists any invariant failure.
   - A disagreement between a year's single file and its US000 slice means the independent check
     is working, not that the script is wrong. Report it and never loosen the check.
   - The 2023 pair is the likeliest to differ: its Last-Modified dates are a day apart
     (2024-08-29 and 2024-08-28).
2. The split-code set is not 17 codes, all under NAICS 238, the same in every window year.
3. The vintage check fails either way:
   - a 2022–2025 national file publishes a six-digit code outside the codebook other than the 34
     BLS children of the split codes and 999999;
   - the 2021 slice publishes no such code, which leaves the vintage boundary unproven.
4. The ownership check fails either way:
   - an `own_code` 0 row exists at any six-digit level;
   - more than 60 codebook codes have no private national cell in 2025.
5. `other`-status cells exceed 1 % of published cells at any grain and year.
6. A title in `agglevel_titles.csv` contradicts the grain list above, or the Connecticut or MSA
   row pattern differs from the facts table.
7. A downloaded file's size or Last-Modified differs from Task 5's table, meaning BLS re-issued
   it. Record the new values and ask whether to proceed.

## File Structure

- Create `scripts/employment_statistics_coverage.py`: the whole analysis, standalone and outside
  the package (Stage 3 designs its own loader). It holds:
  - the universe, reading and cells;
  - tables, checks and the pre-registered decision rule;
  - the provenance manifest, report rendering, and a CLI with `manifest` and `run`.
- Create `tests/unit/test_employment_statistics_coverage.py`: fixture tests that load the script
  by path, with no package import.
- Create `specs/findings/employment-statistics-coverage.md`: the finding, written in Tasks 5–7.
- Modify `specs/naics-embedding-roadmap.md` in Task 8: tick Stage 1, correct its Produces path,
  and add the completion stamp.
- Outside the repo: `~/Downloads/Data/QCEW/`, holding the downloads, `headers/`, `MANIFEST.json`
  and `coverage/`.

## Pre-flight (controller, inline, before Task 1)

- [x] **Step 1: Confirm the workspace**

> Deviation: `git log --oneline origin/main..HEAD` printed nothing, because the plan commit (e04231e) was already on origin/main.

Run: `git status --short --branch`
Expected: `## claude/employment-statistics-coverage-447c9d1f` and nothing else.

Run: `git log --oneline origin/main..HEAD`
Expected: only this plan's commit (`docs(plans): add plan 3 …`). If "config" or "graph config"
appears, stop.

Run: `git fetch origin`, then `git log --oneline HEAD..origin/main -- scripts tests/unit specs`.
Expected: no output. If anything landed there, read it. When it touches this plan's files, the
roadmap or `specs/findings/`, stop and ask.

- [x] **Step 2: Build the worktree's environment**

Run: `uv sync`, then `uv run python --version`
Expected: `Python 3.12.` followed by a patch number. `.python-version` pins 3.12.

- [x] **Step 3: Check the codebook**

Run: `shasum -a 256 /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet`
Expected: `5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b`

- [x] **Step 4: Ask for download permission**

Show your human partner Task 5's table: 13 files, 302,173,870 bytes (about 302 MB), from
`data.bls.gov` to `~/Downloads/Data/QCEW/`. Wait for an explicit yes. Without it, stop: the plan
cannot run without the files.

- [x] **Step 5: Route the tasks**

Under executing-plans, run every task inline, in order. Under subagent-driven-development, run
Tasks 1–4 with a fresh implementer and task-reviewer each, and run Tasks 5–8 inline in the
controller session: they need the browser, the files outside the worktree, stop-and-ask
judgment and your human partner.

### Task 1: Script core: universe, reading and cells

**Files:**

- Create: `scripts/employment_statistics_coverage.py`
- Create: `tests/unit/test_employment_statistics_coverage.py`

**Interfaces:**

- Consumes: nothing.
- Produces constants that later tasks use by name:
  - `CODEBOOK_SHA256`, `WINDOW`, `VINTAGE_CHECK_YEAR`, `PRIVATE`, `OWNERSHIPS`
  - `GRAINS`, `CANDIDATE_GRAINS`, `NATIONAL_TOTAL_AGGLVL`, `TOTAL_INDUSTRY`, `NATIONAL_AREA`,
    `UNKNOWN_COUNTY_SUFFIX`, `CONNECTICUT_LEGACY`, `USED_AGGLVLS`
  - `SURVIVAL_FLOOR`, `ASK_BAND`, `MIN_WINDOW_YEARS`
  - `DISCLOSED`, `SUPPRESSED`, `OTHER`, `ABSENT`
  - `ESTABS_COLUMNS`, `KEY_COLUMNS`, `CELL_COLUMNS`, `SERIES`, `BLS`, `SOURCES`
- Produces these functions and types:
  - `Universe(six_digit: tuple[str, ...], only_children: frozenset[str])`
  - `sha256_file(path: Path) -> str`
  - `load_universe(path: Path, expected_sha256: str = CODEBOOK_SHA256) -> Universe`
  - `resolve_estabs_column(header: Sequence[str]) -> str`
  - `read_annual_csv(data: bytes, agglvls: Collection[str] = USED_AGGLVLS) -> pl.DataFrame`,
    returning `KEY_COLUMNS` as strings plus `estabs`, `emp` and `wages` as Int64, with `year` as
    Int32
  - `read_singlefile_zip(path: Path) -> pl.DataFrame`
  - `national_six_digit_codes(frame: pl.DataFrame) -> set[str]`
  - `find_split_codes(published: Collection[str], universe: Universe) -> tuple[str, ...]`
  - `disclosure_status() -> pl.Expr` and `estabs_status() -> pl.Expr`
  - `grain_cells(frame, universe, split, grain) -> pl.DataFrame`, with columns `CELL_COLUMNS`
- Produces test helpers that later tasks' tests use:
  - `_record`, `_annual_rows(year)`, `_csv_bytes(rows)`, `_write_codebook(directory)`,
    `_digest(path)`
  - fixtures `universe` and `frames`
  - `_cells(frames, universe, grain)`

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_employment_statistics_coverage.py` with exactly this content:

```python
import csv
import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import polars as pl
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'employment_statistics_coverage.py'
_SPEC = importlib.util.spec_from_file_location('employment_statistics_coverage', _SCRIPT)
esc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = esc
_SPEC.loader.exec_module(esc)

WINDOW = (2022, 2023, 2024, 2025)
CSV_COLUMNS = (
    *esc.KEY_COLUMNS,
    'annual_avg_estabs',
    'annual_avg_emplvl',
    'total_annual_wages',
    'avg_annual_pay',
)
SIX_DIGIT = ('111110', '112130', '238110', '238120', '541511', '541512', '921110')
HIGHER = (
    '11', '111', '1111', '11111', '112', '1121', '11213', '23', '238', '2381', '23811', '23812',
    '54', '541', '5415', '54151', '92', '921', '9211', '92111'
)

# -------------------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------------------

def _record(year, area, own, industry, agglvl, disclosure, estabs, emp, wages, size='0'):
    return {
        'area_fips': area,
        'own_code': own,
        'industry_code': industry,
        'agglvl_code': agglvl,
        'size_code': size,
        'year': str(year),
        'qtr': 'A',
        'disclosure_code': disclosure,
        'annual_avg_estabs': str(estabs),
        'annual_avg_emplvl': str(emp),
        'total_annual_wages': str(wages),
        'avg_annual_pay': '0',
    }

def _annual_rows(year):
    connecticut = '09001' if year <= 2023 else '09110'
    rows = [
        ('US000', '5', '10', '11', '', 100, 1000, 100000),
        ('US000', '5', '111110', '18', '', 10, 100, 10000),
        ('US000', '5', '541511', '18', 'N', 3, 0, 0),
        ('US000', '5', '541512', '18', '-', 0, 0, 0),
        ('US000', '5', '238111', '18', '', 4, 40, 4000),
        ('US000', '5', '238112', '18', 'N', 1, 0, 0),
        ('US000', '5', '238121', '18', '', 2, 20, 2000),
        ('US000', '5', '238122', '18', 'N', 2, 0, 0),
        ('US000', '5', '999999', '18', '', 1, 5, 500),
        ('US000', '1', '921110', '18', '', 1, 50, 5000),
        ('US000', '5', '11111', '17', '', 10, 100, 10000),
        ('US000', '5', '23811', '17', '', 5, 45, 4500),
        ('US000', '5', '23812', '17', 'N', 4, 0, 0),
        ('01000', '5', '111110', '58', '', 6, 60, 6000),
        ('09000', '5', '111110', '58', '', 3, 30, 3000),
        ('72000', '5', '111110', '58', '', 1, 5, 500),
        ('01000', '5', '541511', '58', 'N', 2, 0, 0),
        ('01000', '5', '23811', '57', '', 3, 25, 2500),
        ('01001', '5', '111110', '78', '', 4, 35, 3500),
        ('01003', '5', '111110', '78', 'N', 1, 0, 0),
        ('01999', '5', '111110', '78', '', 1, 5, 500),
        (connecticut, '5', '111110', '78', '', 2, 20, 2000),
    ]
    if year <= 2024:
        rows.append(('C1010', '5', '111110', '48', '', 3, 30, 3000))
    return [_record(year, *row) for row in rows]

def _csv_bytes(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()

def _write_codebook(directory):
    path = directory / 'naics_codebook.parquet'
    codes = sorted([*HIGHER, *SIX_DIGIT])
    pl.DataFrame(
        {
            'code_id': list(range(len(codes))),
            'code': codes
        },
        schema={
            'code_id': pl.Int32,
            'code': pl.String
        }
    ).write_parquet(path)
    return path

def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def universe(tmp_path):
    path = _write_codebook(tmp_path)
    return esc.load_universe(path, _digest(path))

@pytest.fixture
def frames():
    return {year: esc.read_annual_csv(_csv_bytes(_annual_rows(year))) for year in WINDOW}

def _cells(frames, universe, grain):
    return pl.concat(
        [esc.grain_cells(frames[year], universe, ('238110', '238120'), grain) for year in WINDOW]
    )

# -------------------------------------------------------------------------------------------------
# Universe and reading
# -------------------------------------------------------------------------------------------------

def test_load_universe_checks_the_hash_and_finds_only_children(tmp_path):
    path = _write_codebook(tmp_path)
    universe = esc.load_universe(path, _digest(path))
    assert universe.six_digit == SIX_DIGIT
    assert universe.only_children == {'111110', '112130', '238110', '238120', '921110'}
    with pytest.raises(ValueError, match='sha256'):
        esc.load_universe(path, '0' * 64)

def test_resolve_estabs_column_accepts_one_spelling():
    assert esc.resolve_estabs_column(['annual_avg_estabs']) == 'annual_avg_estabs'
    assert esc.resolve_estabs_column(['annual_avg_estabs_count']) == 'annual_avg_estabs_count'
    with pytest.raises(ValueError, match='exactly one'):
        esc.resolve_estabs_column(['annual_avg_estabs', 'annual_avg_estabs_count'])
    with pytest.raises(ValueError, match='exactly one'):
        esc.resolve_estabs_column(['annual_avg_emplvl'])

def test_read_annual_csv_keeps_codes_as_strings():
    rows = _annual_rows(2024) + [_record(2024, 'US000', '5', '111110', '28', '', 1, 1, 1, '1')]
    frame = esc.read_annual_csv(_csv_bytes(rows))
    assert frame.schema['area_fips'] == pl.String
    assert frame.schema['estabs'] == pl.Int64
    assert frame.schema['year'] == pl.Int32
    assert {'01001', '01999', '09110', 'C1010'} <= set(frame.get_column('area_fips').to_list())
    assert '28' not in frame.get_column('agglvl_code').to_list()
    assert frame.filter(pl.col('industry_code') == '111110').get_column('disclosure_code')[0] == ''

def test_read_annual_csv_rejects_missing_columns():
    data = _csv_bytes(_annual_rows(2024)).replace(b'total_annual_wages', b'tot_wages')
    with pytest.raises(ValueError, match='lacks columns'):
        esc.read_annual_csv(data)

def test_read_singlefile_zip_reads_the_only_csv(tmp_path):
    path = tmp_path / '2024_annual_singlefile.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('2024.annual.singlefile.csv', _csv_bytes(_annual_rows(2024)))
    assert esc.read_singlefile_zip(path).height == len(_annual_rows(2024))

# -------------------------------------------------------------------------------------------------
# Cells
# -------------------------------------------------------------------------------------------------

def test_find_split_codes_detects_bls_children(universe, frames):
    published = esc.national_six_digit_codes(frames[2024])
    assert '921110' in published  # government-only codes count as published
    assert esc.find_split_codes(published, universe) == ('238110', '238120')

def test_find_split_codes_rejects_a_split_code_with_siblings(tmp_path):
    path = tmp_path / 'codebook.parquet'
    pl.DataFrame({'code': ['238110', '238113']}).write_parquet(path)
    universe = esc.load_universe(path, _digest(path))
    with pytest.raises(ValueError, match='only child'):
        esc.find_split_codes(['238111', '238113'], universe)

def test_grain_cells_never_read_suppression_as_zero(universe, frames):
    cells = esc.grain_cells(frames[2024], universe, ('238110', '238120'), 'national')
    private = {row['code']: row for row in cells.filter(pl.col('own_code') == '5').to_dicts()}
    assert set(private) == {'111110', '238110', '238120', '541511', '541512'}
    assert private['541511']['status'] == esc.SUPPRESSED
    assert private['541511']['estabs_status'] == esc.DISCLOSED
    assert private['541512']['status'] == esc.OTHER
    assert private['238110']['source'] == 'five_digit_parent'
    assert private['238110']['status'] == esc.DISCLOSED
    assert private['238120']['status'] == esc.SUPPRESSED
    assert private['111110']['source'] == 'six_digit'

def test_grain_cells_drop_unknown_counties(universe, frames):
    cells = esc.grain_cells(frames[2024], universe, ('238110', '238120'), 'county')
    assert set(cells.get_column('area_fips').to_list()) == {'01001', '01003', '09110'}
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_employment_statistics_coverage.py -q`
Expected: a collection ERROR ending in `FileNotFoundError`, because the script does not exist
yet.

- [x] **Step 3: Write the script's core**

Create `scripts/employment_statistics_coverage.py` with exactly this content:

```python
# -------------------------------------------------------------------------------------------------
# Employment-statistics coverage (roadmap Stage 1)
# -------------------------------------------------------------------------------------------------
'''
Measure how much QCEW disclosure suppression removes from the NAICS 2022 six-digit universe.

Reads the QCEW annual-average single files, the Open Data Access national slices and the
supervision bundle's codebook, then writes the tables and the pre-registered Req 2 decision that
specs/findings/employment-statistics-coverage.md records. A suppressed cell is never read as zero.

    uv run python scripts/employment_statistics_coverage.py manifest --qcew-dir DIR
    uv run python scripts/employment_statistics_coverage.py run --qcew-dir DIR --codebook PATH \
        --final-years 2022 2023 2024 2025 --out-dir DIR
'''

import hashlib
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Sequence

import polars as pl

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------

CODEBOOK_SHA256 = '5c485aa96fc9d016c8aa7f95e269f4222b85e8ee395e529facc7a9f8adcaab7b'
WINDOW = (2022, 2023, 2024, 2025)
VINTAGE_CHECK_YEAR = 2021
PRIVATE = '5'
OWNERSHIPS = ('1', '2', '3', '5')

# Six-digit and five-digit aggregation levels per grain, checked against agglevel_titles.csv.
GRAINS = {
    'national': ('18', '17'),
    'state': ('58', '57'),
    'county': ('78', '77'),
    'msa': ('48', '47'),
}
# MSA is reported but never a candidate: no six-digit rows from 2025, and its codes break
# between 2023 (OMB 13-01) and 2024 (OMB 23-01).
CANDIDATE_GRAINS = ('national', 'state', 'county')
NATIONAL_TOTAL_AGGLVL = '11'
TOTAL_INDUSTRY = '10'
NATIONAL_AREA = 'US000'
UNKNOWN_COUNTY_SUFFIX = '999'
CONNECTICUT_LEGACY = ('09001', '09003', '09005', '09007', '09009', '09011', '09013', '09015')
USED_AGGLVLS = (NATIONAL_TOTAL_AGGLVL, *(level for pair in GRAINS.values() for level in pair))

# Pre-registered in plan 3. Never tune these after reading the data.
SURVIVAL_FLOOR = 506  # 50 % of the codebook's 1,012 six-digit codes
ASK_BAND = (405, 607)  # 40 % to 60 %: a deciding count in here goes to the user
MIN_WINDOW_YEARS = 3

DISCLOSED = 'disclosed'
SUPPRESSED = 'suppressed'
OTHER = 'other'
ABSENT = 'absent'

ESTABS_COLUMNS = ('annual_avg_estabs', 'annual_avg_estabs_count')
KEY_COLUMNS = (
    'area_fips',
    'own_code',
    'industry_code',
    'agglvl_code',
    'size_code',
    'year',
    'qtr',
    'disclosure_code',
)
CELL_COLUMNS = (
    'year',
    'area_fips',
    'own_code',
    'code',
    'source',
    'disclosure_code',
    'status',
    'estabs_status',
    'estabs',
    'emp',
    'wages',
)
SERIES = (('employment', 'status'), ('wages', 'status'), ('establishments', 'estabs_status'))

BLS = 'https://data.bls.gov/cew'
SOURCES = {
    **{
        f'{year}_annual_singlefile.zip': f'{BLS}/data/files/{year}/csv/{year}_annual_singlefile.zip'
        for year in WINDOW
    },
    **{
        f'{year}_US000_annual.csv': f'{BLS}/data/api/{year}/a/area/US000.csv'
        for year in (VINTAGE_CHECK_YEAR, *WINDOW)
    },
    'industry_titles.csv': f'{BLS}/doc/titles/industry/industry_titles.csv',
    'agglevel_titles.csv': f'{BLS}/doc/titles/agglevel/agglevel_titles.csv',
    'area_titles.csv': f'{BLS}/doc/titles/area/area_titles.csv',
    'ownership_titles.csv': f'{BLS}/doc/titles/ownership/ownership_titles.csv',
}

# -------------------------------------------------------------------------------------------------
# Code universe
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Universe:
    '''The codebook's six-digit codes, and those that are their five-digit parent's only child.'''

    six_digit: tuple[str, ...]
    only_children: frozenset[str]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def load_universe(path: Path, expected_sha256: str = CODEBOOK_SHA256) -> Universe:
    '''Load a supervision-bundle codebook's six-digit codes after checking the file's hash.'''
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f'{path}: sha256 {digest} does not match {expected_sha256}')
    codes = pl.read_parquet(path).get_column('code').to_list()
    six_digit = tuple(sorted(code for code in codes if len(code) == 6))
    siblings: dict[str, int] = {}
    for code in six_digit:
        siblings[code[:5]] = siblings.get(code[:5], 0) + 1
    only_children = frozenset(code for code in six_digit if siblings[code[:5]] == 1)
    return Universe(six_digit=six_digit, only_children=only_children)

# -------------------------------------------------------------------------------------------------
# Reading QCEW files
# -------------------------------------------------------------------------------------------------

def resolve_estabs_column(header: Sequence[str]) -> str:
    '''Name of the annual-average establishment column; BLS documents two spellings.'''
    present = [name for name in ESTABS_COLUMNS if name in header]
    if len(present) != 1:
        raise ValueError(f'expected exactly one of {ESTABS_COLUMNS} in the header, found {present}')
    return present[0]

def read_annual_csv(data: bytes, agglvls: Collection[str] = USED_AGGLVLS) -> pl.DataFrame:
    '''Read a QCEW annual-average CSV with every code column kept as a string.

    Returns the key columns plus estabs, emp and wages as Int64, for annual all-size rows at the
    given aggregation levels. A blank disclosure code reads as ''.
    '''
    header = pl.read_csv(data, n_rows=0, infer_schema=False).columns
    estabs = resolve_estabs_column(header)
    needed = [*KEY_COLUMNS, estabs, 'annual_avg_emplvl', 'total_annual_wages']
    missing = [name for name in needed if name not in header]
    if missing:
        raise ValueError(f'QCEW annual CSV lacks columns {missing}')
    renamed = pl.read_csv(data, columns=needed, infer_schema=False).rename(
        {
            estabs: 'estabs',
            'annual_avg_emplvl': 'emp',
            'total_annual_wages': 'wages',
        }
    )
    # yapf: disable
    return (
        renamed
        .with_columns(pl.col(*KEY_COLUMNS).fill_null('').str.strip_chars())
        .with_columns(pl.col('estabs', 'emp', 'wages').str.strip_chars().cast(pl.Int64))
        .filter(
            pl.col('agglvl_code').is_in(list(agglvls))
            & (pl.col('qtr') == 'A')
            & (pl.col('size_code') == '0')
        )
        .with_columns(pl.col('year').cast(pl.Int32))
    )
    # yapf: enable

def read_singlefile_zip(path: Path) -> pl.DataFrame:
    '''Read the one CSV inside a QCEW annual single-file zip.'''
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith('.csv')]
        if len(members) != 1:
            raise ValueError(f'{path.name}: expected one CSV member, found {members}')
        data = archive.read(members[0])
    return read_annual_csv(data)

# -------------------------------------------------------------------------------------------------
# Cells
# -------------------------------------------------------------------------------------------------

def national_six_digit_codes(frame: pl.DataFrame) -> set[str]:
    '''Industry codes on a frame's national six-digit rows, in any ownership.'''
    rows = frame.filter(pl.col('agglvl_code') == GRAINS['national'][0])
    return set(rows.get_column('industry_code').to_list())

def find_split_codes(published: Collection[str], universe: Universe) -> tuple[str, ...]:
    '''Codebook codes QCEW replaces with BLS residential (xxxxx1) and nonresidential (xxxxx2) codes.

    Each such code must be its five-digit parent's only child: the parent's QCEW row then carries
    the code's own value, under the parent's disclosure status.
    '''
    published = set(published)
    codebook = set(universe.six_digit)
    split = []
    for code in universe.six_digit:
        bls_children = {code[:5] + '1', code[:5] + '2'} - codebook
        if code in published or not bls_children & published:
            continue
        if code not in universe.only_children:
            raise ValueError(f'{code} is split into BLS codes but is not an only child')
        split.append(code)
    return tuple(split)

def disclosure_status() -> pl.Expr:
    '''Employment and wage status: one disclosure flag governs both series.'''
    flag = pl.col('disclosure_code')
    # yapf: disable
    return (
        pl.when(flag == '').then(pl.lit(DISCLOSED))
        .when(flag == 'N').then(pl.lit(SUPPRESSED))
        .otherwise(pl.lit(OTHER))
    )
    # yapf: enable

def estabs_status() -> pl.Expr:
    '''Establishment status: on an N row the count stands unless suppression zero-filled it.'''
    flag = pl.col('disclosure_code')
    # yapf: disable
    return (
        pl.when(flag == '').then(pl.lit(DISCLOSED))
        .when((flag == 'N') & (pl.col('estabs') > 0)).then(pl.lit(DISCLOSED))
        .when(flag == 'N').then(pl.lit(SUPPRESSED))
        .otherwise(pl.lit(OTHER))
    )
    # yapf: enable

def grain_cells(
    frame: pl.DataFrame, universe: Universe, split: Collection[str], grain: str
) -> pl.DataFrame:
    '''Every published cell of a codebook six-digit code at one grain, with its status.

    Directly published codes come from the grain's six-digit level; each split code comes from
    its five-digit parent's row. A code with no row is absent: it appears only in code-level
    counts. County rows for unknown or undefined locations (county part 999) are not areas.
    '''
    six_level, five_level = GRAINS[grain]
    direct_codes = sorted(set(universe.six_digit) - set(split))
    direct = frame.filter(
        (pl.col('agglvl_code') == six_level) & pl.col('industry_code').is_in(direct_codes)
    ).with_columns(code=pl.col('industry_code'), source=pl.lit('six_digit'))
    parents = pl.DataFrame(
        {
            'industry_code': [code[:5] for code in split],
            'code': list(split),
        },
        schema={
            'industry_code': pl.String,
            'code': pl.String,
        },
    )
    five_digit = frame.filter(pl.col('agglvl_code') == five_level)
    recovered = five_digit.join(parents, on='industry_code').with_columns(
        source=pl.lit('five_digit_parent')
    )
    cells = pl.concat([direct, recovered], how='diagonal')
    if grain == 'county':
        cells = cells.filter(~pl.col('area_fips').str.ends_with(UNKNOWN_COUNTY_SUFFIX))
    labelled = cells.with_columns(status=disclosure_status(), estabs_status=estabs_status())
    return labelled.select(CELL_COLUMNS)
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_employment_statistics_coverage.py -q`
Expected: `9 passed`.

- [x] **Step 5: Lint and format both files**

Run: `./scripts/format_code.sh --check scripts/employment_statistics_coverage.py tests/unit/test_employment_statistics_coverage.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.` The code above is
already yapf-clean. On a failure, run the same command without `--check`, re-run Step 4, and
record the change as a deviation.

- [x] **Step 6: Commit**

```bash
git add scripts/employment_statistics_coverage.py tests/unit/test_employment_statistics_coverage.py
git commit -m "feat(scripts): add QCEW coverage reader and cell classification"
```

### Task 2: Coverage tables and reconciliation checks

**Files:**

- Modify: `scripts/employment_statistics_coverage.py` (the import block, then append)
- Modify: `tests/unit/test_employment_statistics_coverage.py` (append)

**Interfaces:**

- Consumes: everything Task 1 produces.
- Produces:
  - `code_status_counts(cells, universe, year, own) -> list[dict[str, object]]`. There is one row
    per series in `SERIES`, keyed `year`, `own_code`, `series`, `disclosed`, `suppressed`,
    `other`, `absent`, `recovered_via_parent` and `suppressed_share`.
  - `area_coverage(cells, universe, grain, year) -> dict[str, object]`, keyed:
    - `grain`, `year`, `areas`, `published_cells`, `suppressed_cells`, `other_cells`,
      `suppressed_share`
    - `estabs_suppressed_cells`, `estabs_suppressed_share`
    - `codes_usable`, `codes_usable_2plus_areas`, `codes_published_never_usable`,
      `codes_absent`, `codes_without_usable`, `share_without_usable`, `median_usable_areas`
  - `size_by_status(cells, grain, year) -> list[dict[str, object]]`
  - `vintage_report(published_by_year, universe, split) -> list[dict[str, object]]`
  - `private_gaps(cells, universe, year) -> list[dict[str, object]]`
  - `excluded_codes(cells, universe, window) -> list[dict[str, object]]`
  - `connecticut_areas(county_cells) -> list[dict[str, object]]`
  - `check_invariants(frame, year) -> list[str]`, empty when every invariant holds
  - `singlefile_header(path: Path) -> list[str]`
  - `file_conventions(frame, year) -> dict[str, object]`
  - `compare_national_slices(frame, national_slice, year) -> list[str]`

- [x] **Step 1: Write the failing tests**

Append this block to the end of `tests/unit/test_employment_statistics_coverage.py`, one blank
line after the existing last line:

```python
# -------------------------------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------------------------------

def test_code_status_counts_use_the_whole_universe(universe, frames):
    cells = _cells(frames, universe, 'national')
    rows = {row['series']: row for row in esc.code_status_counts(cells, universe, 2024, '5')}
    employment = rows['employment']
    assert (employment['disclosed'], employment['suppressed'], employment['other']) == (2, 2, 1)
    assert employment['absent'] == 2
    assert employment['recovered_via_parent'] == 2
    establishments = rows['establishments']
    assert (establishments['disclosed'], establishments['suppressed']) == (4, 0)

def test_area_coverage_counts_cells_and_codes(universe, frames):
    row = esc.area_coverage(_cells(frames, universe, 'state'), universe, 'state', 2024)
    assert (row['areas'], row['published_cells'], row['suppressed_cells']) == (3, 5, 1)
    assert (row['codes_usable'], row['codes_usable_2plus_areas']) == (2, 1)
    assert (row['codes_published_never_usable'], row['codes_absent']) == (1, 4)
    assert (row['codes_without_usable'], row['share_without_usable']) == (5, pytest.approx(5 / 7))
    assert (row['estabs_suppressed_cells'], row['estabs_suppressed_share']) == (0, 0.0)
    assert row['median_usable_areas'] == 2.0

def test_size_by_status_compares_establishment_counts(universe, frames):
    rows = esc.size_by_status(_cells(frames, universe, 'state'), 'state', 2024)
    by_status = {row['status']: row for row in rows}
    assert by_status[esc.DISCLOSED]['cells'] == 4
    assert by_status[esc.DISCLOSED]['median_estabs'] == 3.0
    assert by_status[esc.SUPPRESSED]['median_estabs'] == 2.0

def test_vintage_report_flags_codes_outside_the_codebook(universe, frames):
    published = {
        2021: {'111110', '454110', '238111', '238112'},
        2024: set(frames[2024].filter(pl.col('agglvl_code') == '18')['industry_code'].to_list()),
    }
    rows = {row['year']: row for row in esc.vintage_report(published, universe, ('238110', ))}
    assert rows[2021]['outside_codebook'] == 1
    assert rows[2021]['outside_examples'] == ['454110']
    assert rows[2024]['outside_codebook'] == 2  # 238121 and 238122: 238120 not passed as split
    assert rows[2024]['unpublished_examples'] == ['112130', '238120']

def test_private_gaps_and_exclusions(universe, frames):
    national = _cells(frames, universe, 'national')
    gaps = {
        row['code']: row['ownerships_with_cells']
        for row in esc.private_gaps(national, universe, 2025)
    }
    assert gaps == {'112130': [], '921110': ['1']}
    excluded = {
        row['code']: row['reason']
        for row in esc.excluded_codes(national, universe, WINDOW)
    }
    assert excluded == {
        '112130': 'no private cell',
        '238120': 'private cells never usable',
        '541511': 'private cells never usable',
        '541512': 'private cells never usable',
        '921110': 'no private cell',
    }

def test_connecticut_areas_switch_in_2024(universe, frames):
    rows = {row['year']: row for row in esc.connecticut_areas(_cells(frames, universe, 'county'))}
    assert (rows[2023]['legacy_counties'], rows[2023]['planning_regions']) == (1, 0)
    assert (rows[2024]['legacy_counties'], rows[2024]['planning_regions']) == (0, 1)

# -------------------------------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------------------------------

def test_check_invariants_pass_on_consistent_files(frames):
    assert esc.check_invariants(frames[2024], 2024) == []

def test_check_invariants_catch_detail_above_its_total():
    rows = _annual_rows(2024)
    for row in rows:
        if (row['area_fips'], row['industry_code'], row['agglvl_code']) == (
            '01000', '111110', '58'
        ):
            row['annual_avg_emplvl'], row['total_annual_wages'] = '80', '8000'
    failures = esc.check_invariants(esc.read_annual_csv(_csv_bytes(rows)), 2024)
    assert any('state cells exceed their national cell' in failure for failure in failures)
    assert not any('county cells exceed' in failure for failure in failures)

def test_singlefile_header_reads_the_first_line(tmp_path):
    path = tmp_path / '2024_annual_singlefile.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('2024.annual.singlefile.csv', _csv_bytes(_annual_rows(2024)))
    header = esc.singlefile_header(path)
    assert header == list(CSV_COLUMNS)
    assert esc.resolve_estabs_column(header) == 'annual_avg_estabs'

def test_file_conventions_describe_six_digit_rows(frames):
    row = esc.file_conventions(frames[2024], 2024)
    assert row['disclosure_codes'] == {'blank': 12, '-': 1, 'N': 5}
    assert row['own_code_0_rows'] == 0
    assert row['suppressed_rows'] == 5
    assert row['suppressed_rows_with_emp_or_wages'] == 0
    assert row['suppressed_rows_with_estabs'] == 5

def test_compare_national_slices_finds_differences(frames):
    national = frames[2024].filter(pl.col('area_fips') == 'US000')
    assert esc.compare_national_slices(frames[2024], national, 2024) == []
    changed = national.with_columns(
        emp=pl.when(pl.col('industry_code') == '111110').then(99).otherwise(pl.col('emp'))
    )
    assert esc.compare_national_slices(frames[2024], changed, 2024) == [
        '2024: 1 national rows differ between the single file and the slice'
    ]
```

- [x] **Step 2: Run the tests to verify the new ones fail**

Run: `uv run pytest tests/unit/test_employment_statistics_coverage.py -q`
Expected: `11 failed, 9 passed`, with each failure an `AttributeError` naming a function this
task adds, such as `code_status_counts`.

- [x] **Step 3: Implement the tables and checks**

In `scripts/employment_statistics_coverage.py`, replace the line
`from typing import Collection, Sequence` with:

```python
from typing import Collection, Mapping, Sequence
```

Then append this block to the end of the file, one blank line after the existing last line:

```python
# -------------------------------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------------------------------

def code_status_counts(cells: pl.DataFrame, universe: Universe, year: int,
                       own: str) -> list[dict[str, object]]:
    '''National grain: how many codebook codes are disclosed, suppressed, other or absent.'''
    subset = cells.filter((pl.col('year') == year) & (pl.col('own_code') == own))
    if not subset.get_column('code').is_unique().all():
        raise ValueError(f'{year} own {own}: more than one national cell for a code')
    recovered = subset.filter(pl.col('source') == 'five_digit_parent').height
    rows = []
    for series, column in SERIES:
        counts = dict(subset.group_by(column).len().iter_rows())
        rows.append(
            {
                'year': year,
                'own_code': own,
                'series': series,
                DISCLOSED: counts.get(DISCLOSED, 0),
                SUPPRESSED: counts.get(SUPPRESSED, 0),
                OTHER: counts.get(OTHER, 0),
                ABSENT: len(universe.six_digit) - subset.height,
                'recovered_via_parent': recovered,
                'suppressed_share': counts.get(SUPPRESSED, 0) / len(universe.six_digit),
            }
        )
    return rows

def area_coverage(cells: pl.DataFrame, universe: Universe, grain: str,
                  year: int) -> dict[str, object]:
    '''Private cells at one grain and year: suppressed shares and code-level survival.

    Cell shares divide by the published private cells; code shares divide by the codebook's
    six-digit codes, so an absent code counts as having no usable cell.
    '''
    subset = cells.filter((pl.col('year') == year) & (pl.col('own_code') == PRIVATE))
    usable = subset.filter(pl.col('status') == DISCLOSED)
    per_code = usable.group_by('code').agg(pl.col('area_fips').n_unique().alias('areas'))
    published = subset.height
    suppressed = subset.filter(pl.col('status') == SUPPRESSED).height
    estabs_suppressed = subset.filter(pl.col('estabs_status') == SUPPRESSED).height
    codes_published = subset.get_column('code').n_unique()
    without_usable = len(universe.six_digit) - per_code.height
    return {
        'grain': grain,
        'year': year,
        'areas': subset.get_column('area_fips').n_unique(),
        'published_cells': published,
        'suppressed_cells': suppressed,
        'other_cells': subset.filter(pl.col('status') == OTHER).height,
        'suppressed_share': suppressed / published if published else None,
        'estabs_suppressed_cells': estabs_suppressed,
        'estabs_suppressed_share': estabs_suppressed / published if published else None,
        'codes_usable': per_code.height,
        'codes_usable_2plus_areas': per_code.filter(pl.col('areas') >= 2).height,
        'codes_published_never_usable': codes_published - per_code.height,
        'codes_absent': len(universe.six_digit) - codes_published,
        'codes_without_usable': without_usable,
        'share_without_usable': without_usable / len(universe.six_digit),
        'median_usable_areas': per_code.get_column('areas').median() if per_code.height else None,
    }

def size_by_status(cells: pl.DataFrame, grain: str, year: int) -> list[dict[str, object]]:
    '''Establishment counts of private disclosed and suppressed cells: is suppression selective?'''
    subset = cells.filter(
        (pl.col('year') == year) & (pl.col('own_code') == PRIVATE)
        & pl.col('status').is_in([DISCLOSED, SUPPRESSED])
    )
    summary = subset.group_by('status').agg(
        pl.len().alias('cells'),
        pl.col('estabs').median().alias('median_estabs'),
        pl.col('estabs').quantile(0.9).alias('p90_estabs'),
    ).sort('status')
    return [{'grain': grain, 'year': year, **row} for row in summary.to_dicts()]

def vintage_report(
    published_by_year: Mapping[int, Collection[str]], universe: Universe, split: Collection[str]
) -> list[dict[str, object]]:
    '''Per year, national six-digit codes outside the codebook and codebook codes unpublished.

    BLS residential and nonresidential children of the split codes and 999999 (unclassified) are
    expected outside the codebook; anything else there means a different NAICS vintage.
    '''
    codebook = set(universe.six_digit)
    expected_extra = {code[:5] + digit for code in split for digit in '12'} | {'999999'}
    rows = []
    for year in sorted(published_by_year):
        published = set(published_by_year[year])
        outside = sorted(published - codebook - expected_extra)
        unpublished = sorted(codebook - published - set(split))
        rows.append(
            {
                'year': year,
                'published_six_digit': len(published),
                'outside_codebook': len(outside),
                'outside_examples': outside[:12],
                'codebook_unpublished': len(unpublished),
                'unpublished_examples': unpublished[:12],
            }
        )
    return rows

def private_gaps(cells: pl.DataFrame, universe: Universe, year: int) -> list[dict[str, object]]:
    '''National codes with no private cell in a year, and the ownerships that do have one.'''
    subset = cells.filter(pl.col('year') == year)
    private = set(subset.filter(pl.col('own_code') == PRIVATE).get_column('code').to_list())
    owners = dict(
        subset.group_by('code').agg(pl.col('own_code').unique().sort().alias('owners')).iter_rows()
    )
    return [
        {
            'code': code,
            'ownerships_with_cells': owners.get(code, [])
        } for code in universe.six_digit if code not in private
    ]

def excluded_codes(cells: pl.DataFrame, universe: Universe,
                   window: Sequence[int]) -> list[dict[str, object]]:
    '''Codes with no usable private cell anywhere in the window at one grain, with the reason.'''
    private = cells.filter((pl.col('own_code') == PRIVATE) & pl.col('year').is_in(list(window)))
    published = set(private.get_column('code').to_list())
    usable = set(private.filter(pl.col('status') == DISCLOSED).get_column('code').to_list())
    return [
        {
            'code': code,
            'reason': 'private cells never usable' if code in published else 'no private cell',
        } for code in universe.six_digit if code not in usable
    ]

def connecticut_areas(county_cells: pl.DataFrame) -> list[dict[str, object]]:
    '''Connecticut county-equivalents per year: legacy counties or planning regions (from 2024).'''
    rows = []
    for year in sorted(set(county_cells.get_column('year').to_list())):
        areas = set(
            county_cells.filter(
                (pl.col('year') == year)
                & pl.col('area_fips').str.starts_with('09')
            ).get_column('area_fips').to_list()
        )
        regions = {area for area in areas if '09110' <= area <= '09190'}
        rows.append(
            {
                'year': year,
                'legacy_counties': len(areas & set(CONNECTICUT_LEGACY)),
                'planning_regions': len(regions),
            }
        )
    return rows

# -------------------------------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------------------------------

def _disclosed_private(frame: pl.DataFrame, level: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col('agglvl_code') == level) & (pl.col('own_code') == PRIVATE)
        & (pl.col('disclosure_code') == '')
    )

def _nested_excess(detail: pl.DataFrame, parents: pl.DataFrame, year: int, child: str,
                   parent: str) -> list[str]:
    '''Detail cells may not sum past their parent cell (employment allows annual-average rounding).'''
    sums = detail.group_by('parent', 'industry_code').agg(
        pl.col('emp').sum(),
        pl.col('wages').sum(),
        pl.len().alias('cells'),
    )
    joined = sums.join(
        parents.select('parent', 'industry_code', 'emp', 'wages'),
        on=['parent', 'industry_code'],
        suffix='_parent',
    )
    bad = joined.filter(
        (pl.col('emp') > pl.col('emp_parent') + pl.col('cells') / 2 + 1)
        | (pl.col('wages') > pl.col('wages_parent'))
    )
    failures = [
        f'{year}: {child} cells exceed their {parent} cell for {row["parent"]} '
        f'{row["industry_code"]}' for row in bad.head(20).to_dicts()
    ]
    if bad.height > 20:
        failures.append(f'{year}: {bad.height} {child}-over-{parent} excesses in all')
    return failures

def check_invariants(frame: pl.DataFrame, year: int) -> list[str]:
    '''Describe every failed invariant of one year's single file (empty when all hold).

    Disclosed private detail never sums past a published total: six-digit cells against the
    national private total, states (50 plus DC) against the nation per code, and counties
    (unknown locations included) against their state per code.
    '''
    total = frame.filter(
        (pl.col('agglvl_code') == NATIONAL_TOTAL_AGGLVL)
        & (pl.col('own_code') == PRIVATE)
        & (pl.col('industry_code') == TOTAL_INDUSTRY)
        & (pl.col('area_fips') == NATIONAL_AREA)
    )
    if total.height != 1:
        return [f'{year}: expected one national private total row, found {total.height}']
    failures = []
    national = _disclosed_private(frame, GRAINS['national'][0])
    for column in ('emp', 'wages'):
        detail, whole = national.get_column(column).sum(), total.get_column(column).item()
        if detail > whole:
            failures.append(f'{year}: disclosed six-digit {column} {detail} exceeds total {whole}')
    states = _disclosed_private(frame, GRAINS['state'][0])
    failures += _nested_excess(
        states.filter(pl.col('area_fips').str.slice(0, 2).cast(pl.Int32) <= 56).with_columns(
            parent=pl.lit(NATIONAL_AREA)
        ),
        national.rename({'area_fips': 'parent'}),
        year,
        'state',
        'national',
    )
    counties = _disclosed_private(frame, GRAINS['county'][0])
    failures += _nested_excess(
        counties.with_columns(parent=pl.col('area_fips').str.slice(0, 2) + '000'),
        states.rename({'area_fips': 'parent'}),
        year,
        'county',
        'state',
    )
    return failures

def singlefile_header(path: Path) -> list[str]:
    '''Column names on the first line of a QCEW annual single-file zip.'''
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith('.csv'))
        with archive.open(member) as handle:
            first = handle.readline().decode()
    return [name.strip().strip('"') for name in first.strip().split(',')]

def file_conventions(frame: pl.DataFrame, year: int) -> dict[str, object]:
    '''What one annual file's six-digit rows show about the conventions the tables rely on.'''
    six = frame.filter(pl.col('agglvl_code').is_in([six for six, _ in GRAINS.values()]))
    codes = dict(six.group_by('disclosure_code').len().sort('disclosure_code').iter_rows())
    suppressed = six.filter(pl.col('disclosure_code') == 'N')
    shows_values = (pl.col('emp') != 0) | (pl.col('wages') != 0)
    return {
        'year': year,
        'disclosure_codes': {
            code or 'blank': count
            for code, count in codes.items()
        },
        'own_code_0_rows': six.filter(pl.col('own_code') == '0').height,
        'suppressed_rows': suppressed.height,
        'suppressed_rows_with_emp_or_wages': suppressed.filter(shows_values).height,
        'suppressed_rows_with_estabs': suppressed.filter(pl.col('estabs') > 0).height,
    }

def compare_national_slices(frame: pl.DataFrame, national_slice: pl.DataFrame,
                            year: int) -> list[str]:
    '''The single file and the Open Data Access US000 slice must agree on every national row.'''
    keys = ['own_code', 'industry_code', 'agglvl_code']
    values = ['disclosure_code', 'estabs', 'emp', 'wages']
    levels = [NATIONAL_TOTAL_AGGLVL, *GRAINS['national']]
    left = frame.filter(
        (pl.col('area_fips') == NATIONAL_AREA)
        & pl.col('agglvl_code').is_in(levels)
    ).select(*keys, *values)
    right = national_slice.filter(pl.col('agglvl_code').is_in(levels)).select(*keys, *values)
    joined = left.join(right, on=keys, how='full', suffix='_slice', coalesce=True)
    differs = pl.any_horizontal(
        [pl.col(name).ne_missing(pl.col(f'{name}_slice')) for name in values]
    )
    mismatched = joined.filter(differs).height
    if mismatched:
        return [f'{year}: {mismatched} national rows differ between the single file and the slice']
    return []
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_employment_statistics_coverage.py -q`
Expected: `20 passed`.

- [x] **Step 5: Lint and format both files**

Run: `./scripts/format_code.sh --check scripts/employment_statistics_coverage.py tests/unit/test_employment_statistics_coverage.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add scripts/employment_statistics_coverage.py tests/unit/test_employment_statistics_coverage.py
git commit -m "feat(scripts): add QCEW coverage tables and reconciliation checks"
```

### Task 3: The pre-registered decision rule

**Files:**

- Modify: `scripts/employment_statistics_coverage.py` (append)
- Modify: `tests/unit/test_employment_statistics_coverage.py` (append)

**Interfaces:**

- Consumes: `cells` frames from `grain_cells`, plus `CANDIDATE_GRAINS`, `SURVIVAL_FLOOR`,
  `ASK_BAND`, `MIN_WINDOW_YEARS`, `PRIVATE`, `DISCLOSED` and `SUPPRESSED`.
- Produces:
  - `GrainSummary(grain, complete, mean_suppressed_share, seen_by_year, seen_by_area,
    time_eligible, heldout_population)`, with a `.seen` property
  - `Decision(branch, grain, time_respecting, seen_regime, needs_user, reasons)`
  - `summarize_grain(cells, grain, window) -> GrainSummary`
  - `decide(summaries, window, final_years, floor=SURVIVAL_FLOOR, band=ASK_BAND) -> Decision`
  - `BRANCH_TEXT` and `ROW_GRAIN`
  - `render_decision(decision, summaries, window) -> str`, the finding's decision block between
    `<!-- decision:begin -->` and `<!-- decision:end -->`

- [x] **Step 1: Write the failing tests**

Append this block to the end of `tests/unit/test_employment_statistics_coverage.py`, one blank
line after the existing last line:

```python
# -------------------------------------------------------------------------------------------------
# Decision rule
# -------------------------------------------------------------------------------------------------

def _summary(grain, share, seen, time, complete=True, by_area=0):
    return esc.GrainSummary(grain, complete, share, seen, by_area, time, seen)

def test_summarize_grain_counts_what_the_rule_reads(universe, frames):
    national = esc.summarize_grain(_cells(frames, universe, 'national'), 'national', WINDOW)
    assert national.complete
    assert national.mean_suppressed_share == pytest.approx(0.4)
    assert (national.seen_by_year, national.seen_by_area, national.time_eligible) == (2, 0, 2)
    assert national.heldout_population == 2
    county = esc.summarize_grain(_cells(frames, universe, 'county'), 'county', WINDOW)
    assert (county.seen_by_year, county.seen_by_area, county.time_eligible) == (1, 1, 1)
    msa = esc.summarize_grain(_cells(frames, universe, 'msa'), 'msa', WINDOW)
    assert not msa.complete

def test_decide_prefers_the_least_suppressed_surviving_grain():
    summaries = [
        _summary('national', 0.01, 990, 980),
        _summary('state', 0.30, 900, 850),
        _summary('county', 0.55, 700, 600),
        _summary('msa', 0.001, 999, 999, complete=False),
    ]
    decision = esc.decide(summaries, WINDOW, WINDOW)
    assert (decision.branch, decision.grain) == ('A', 'national')
    assert decision.time_respecting and decision.seen_regime and not decision.needs_user

def test_decide_is_cross_sectional_without_three_consecutive_final_years():
    summaries = [_summary('national', 0.01, 990, 980)]
    assert esc.decide(summaries, (2024, 2025), (2024, 2025)).branch == 'B'
    assert esc.decide(summaries, WINDOW, (2022, 2023, 2024)).branch == 'B'
    assert esc.decide(summaries, (2022, 2023, 2025), (2022, 2023, 2025)).branch == 'B'

def test_decide_is_held_out_only_when_no_grain_survives():
    summaries = [_summary('national', 0.2, 300, 300), _summary('state', 0.6, 200, 100)]
    decision = esc.decide(summaries, WINDOW, WINDOW)
    assert (decision.branch, decision.grain, decision.seen_regime) == ('C', None, False)
    assert not decision.needs_user

def test_decide_asks_when_a_deciding_count_is_in_the_band():
    ahead_in_band = [_summary('national', 0.01, 500, 500), _summary('state', 0.3, 800, 700)]
    decision = esc.decide(ahead_in_band, WINDOW, WINDOW)
    assert (decision.branch, decision.grain, decision.needs_user) == ('A', 'state', True)
    time_in_band = [_summary('national', 0.01, 990, 450)]
    decision = esc.decide(time_in_band, WINDOW, WINDOW)
    assert (decision.branch, decision.needs_user) == ('B', True)
    assert esc.decide([_summary('national', 0.2, 450, 0)], WINDOW, WINDOW).needs_user

def test_render_decision_uses_fixed_wording():
    summaries = [_summary('national', 0.01, 990, 980)]
    text = esc.render_decision(esc.decide(summaries, WINDOW, WINDOW), summaries, WINDOW)
    assert text.startswith('<!-- decision:begin -->\n- **Branch:** A. The verified window')
    assert '- **Row grain:** a six-digit code in a reference year (national, private' in text
    assert text.rstrip().endswith('<!-- decision:end -->')
```

- [x] **Step 2: Run the tests to verify the new ones fail**

Run: `uv run pytest tests/unit/test_employment_statistics_coverage.py -q`
Expected: `6 failed, 20 passed`, with each failure an `AttributeError` naming `GrainSummary`
(the tests build summaries through it) or `summarize_grain`.

- [x] **Step 3: Implement the decision rule**

Append this block to the end of `scripts/employment_statistics_coverage.py`, one blank line after
the existing last line:

```python
# -------------------------------------------------------------------------------------------------
# Decision rule (pre-registered in plan 3)
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GrainSummary:
    '''What the decision rule reads about one grain's private cells over the window.'''

    grain: str
    complete: bool  # six-digit rows in every window year
    mean_suppressed_share: float  # suppressed / published cells, averaged over the years
    seen_by_year: int  # codes usable in the last year and in an earlier one
    seen_by_area: int  # codes usable in two or more areas in the last year (0 when national)
    time_eligible: int  # codes with a same-area usable pair in the last pair and an earlier one
    heldout_population: int  # codes usable at least once in the window

    @property
    def seen(self) -> int:
        return max(self.seen_by_year, self.seen_by_area)

@dataclass(frozen=True)
class Decision:
    branch: str  # 'A' time-respecting, 'B' cross-sectional, 'C' held-out codes only
    grain: str | None
    time_respecting: bool
    seen_regime: bool
    needs_user: bool  # a deciding count fell inside the ask band: stop and ask
    reasons: tuple[str, ...]

def summarize_grain(cells: pl.DataFrame, grain: str, window: Sequence[int]) -> GrainSummary:
    '''Reduce one grain's cells to the counts the decision rule reads.'''
    private = cells.filter(pl.col('own_code') == PRIVATE)
    disclosed = private.filter(pl.col('status') == DISCLOSED)
    usable = disclosed.select('year', 'area_fips', 'code').unique()
    years = sorted(window)
    shares = []
    for year in years:
        subset = private.filter(pl.col('year') == year)
        if subset.height:
            shares.append(subset.filter(pl.col('status') == SUPPRESSED).height / subset.height)

    def codes(frame: pl.DataFrame) -> set[str]:
        return set(frame.get_column('code').to_list())

    def paired(start: int) -> set[str]:
        first = usable.filter(pl.col('year') == start).select('area_fips', 'code')
        second = usable.filter(pl.col('year') == start + 1).select('area_fips', 'code')
        return codes(first.join(second, on=['area_fips', 'code']))

    last = years[-1]
    in_last = codes(usable.filter(pl.col('year') == last))
    seen_by_year = len(in_last & codes(usable.filter(pl.col('year').is_in(years[:-1]))))
    seen_by_area = 0
    if grain != 'national':
        areas = usable.filter(pl.col('year') == last).group_by('code').agg(
            pl.col('area_fips').n_unique().alias('areas')
        )
        seen_by_area = areas.filter(pl.col('areas') >= 2).height
    starts = [year for year in years[:-1] if year + 1 in years]
    time_eligible = 0
    if len(starts) >= 2:
        earlier = set().union(*(paired(start) for start in starts[:-1]))
        time_eligible = len(paired(starts[-1]) & earlier)
    return GrainSummary(
        grain=grain,
        complete=len(shares) == len(years),
        mean_suppressed_share=sum(shares) / len(shares) if shares else 1.0,
        seen_by_year=seen_by_year,
        seen_by_area=seen_by_area,
        time_eligible=time_eligible,
        heldout_population=usable.get_column('code').n_unique(),
    )

def decide(
    summaries: Sequence[GrainSummary],
    window: Sequence[int],
    final_years: Collection[int],
    floor: int = SURVIVAL_FLOOR,
    band: tuple[int, int] = ASK_BAND,
) -> Decision:
    '''Apply plan 3's pre-registered rule.

    A candidate is a CANDIDATE_GRAINS grain with six-digit rows in every window year; it
    survives when at least `floor` codes can run the seen-code regime. The chosen grain is the
    surviving candidate with the lowest mean suppressed share (ties: national, state, county).
    The outcome is time-respecting when the window is at least MIN_WINDOW_YEARS consecutive
    final years and at least `floor` codes are time-eligible at the chosen grain. A deciding
    count inside `band` sets needs_user.
    '''

    def in_band(count: int) -> bool:
        return band[0] <= count <= band[1]

    order = {grain: rank for rank, grain in enumerate(CANDIDATE_GRAINS)}
    candidates = sorted(
        (summary for summary in summaries if summary.grain in order and summary.complete),
        key=lambda summary: (summary.mean_suppressed_share, order[summary.grain]),
    )
    reasons = [
        f'{s.grain}: {s.seen} codes can run the seen-code regime (floor {floor}); '
        f'mean suppressed share {s.mean_suppressed_share:.4f}' for s in candidates
    ]
    reasons += [
        f'{s.grain}: not a candidate (not a candidate grain, or no six-digit rows in a window year)'
        for s in summaries if s not in candidates
    ]
    surviving = [summary for summary in candidates if summary.seen >= floor]
    if not surviving:
        best = max((summary.seen for summary in candidates), default=0)
        reasons.append('no grain below the code survives suppression')
        return Decision('C', None, False, False, in_band(best), tuple(reasons))
    chosen = surviving[0]
    needs_user = in_band(chosen.seen)
    for summary in candidates[:candidates.index(chosen)]:
        if in_band(summary.seen):
            needs_user = True
            reasons.append(
                f'{summary.grain} ranks ahead of {chosen.grain} with {summary.seen} '
                f'codes, inside the ask band {band}'
            )
    years = sorted(window)
    consecutive = years == list(range(years[0], years[0] + len(years)))
    final = all(year in final_years for year in years)
    window_ok = len(years) >= MIN_WINDOW_YEARS and consecutive and final
    if not window_ok:
        reasons.append(
            f'window {years} is not {MIN_WINDOW_YEARS} or more consecutive final years '
            f'(final: {sorted(final_years)})'
        )
    elif in_band(chosen.time_eligible):
        needs_user = True
    reasons.append(
        f'{chosen.grain}: {chosen.time_eligible} codes are time-eligible (floor {floor})'
    )
    time_respecting = window_ok and chosen.time_eligible >= floor
    return Decision(
        'A' if time_respecting else 'B', chosen.grain, time_respecting, True, needs_user,
        tuple(reasons)
    )

BRANCH_TEXT = {
    'A': 'The verified window supports a time-respecting outcome (the outcome dated after the '
    'features, with splits by time), so the panel includes one.',
    'B': 'The verified window does not support a time-respecting outcome, so the panel is '
    'cross-sectional.',
    'C': 'No grain below the code survives suppression, so the panel is held-out-codes only and '
    'the one-hot comparison could not run.',
}
ROW_GRAIN = {
    'national': 'a six-digit code in a reference year (national, private ownership)',
    'state': 'a six-digit code in a state in a reference year (private ownership)',
    'county': 'a six-digit code in a county in a reference year (private ownership)',
}

def render_decision(
    decision: Decision, summaries: Sequence[GrainSummary], window: Sequence[int]
) -> str:
    '''The finding's "Decision for Stage 3" block, in fixed wording.'''
    chosen = next((summary for summary in summaries if summary.grain == decision.grain), None)
    lines = [
        '<!-- decision:begin -->',
        f'- **Branch:** {decision.branch}. {BRANCH_TEXT[decision.branch]}',
        f'- **Source:** QCEW annual averages, reference years {", ".join(map(str, window))}, '
        'private ownership (own_code 5).',
    ]
    if chosen is None:
        lines.append('- **Row grain:** one row per code; no grain below the code survives.')
    else:
        lines += [
            f'- **Row grain:** {ROW_GRAIN[chosen.grain]}.',
            f'- **Population:** {chosen.seen} codes for the seen-code regime and '
            f'{chosen.heldout_population} for the held-out-code regime, of the 1,012 six-digit '
            'codes in the codebook.',
        ]
    lines += [
        f'- **Time-respecting outcome:** {"yes" if decision.time_respecting else "no"}.',
        f'- **Seen-code regime:** {"yes" if decision.seen_regime else "no"}.',
        f'- **Rule:** plan 3, survival floor {SURVIVAL_FLOOR} codes, ask band {ASK_BAND[0]} to '
        f'{ASK_BAND[1]}; user review {"required" if decision.needs_user else "not required"}.',
        '- **Reasons:**',
        *[f'  - {reason}' for reason in decision.reasons],
        '<!-- decision:end -->',
    ]
    return '\n'.join(lines) + '\n'
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_employment_statistics_coverage.py -q`
Expected: `26 passed`.

- [x] **Step 5: Lint and format both files**

Run: `./scripts/format_code.sh --check scripts/employment_statistics_coverage.py tests/unit/test_employment_statistics_coverage.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add scripts/employment_statistics_coverage.py tests/unit/test_employment_statistics_coverage.py
git commit -m "feat(scripts): add the pre-registered Req 2 decision rule"
```

### Task 4: Provenance manifest, report and CLI

**Files:**

- Modify: `scripts/employment_statistics_coverage.py` (the import block, then append)
- Modify: `tests/unit/test_employment_statistics_coverage.py` (the import block, then append)

**Interfaces:**

- Consumes: everything Tasks 1–3 produce.
- Produces:
  - `parse_headers(text: str) -> dict[str, str]`
  - `build_manifest(qcew_dir: Path, sources: Mapping[str, str] = SOURCES) -> list[dict]`
  - `markdown_table(rows) -> str` and `render_tables(report) -> str`
  - `run(qcew_dir, codebook, final_years, out_dir, window=WINDOW,
    codebook_sha256=CODEBOOK_SHA256, floor=SURVIVAL_FLOOR, band=ASK_BAND)
    -> tuple[Decision, list[str]]`, which writes `coverage.json`, `tables.md` and `decision.md`
    into `out_dir`
  - `main(argv) -> int` with subcommands `manifest --qcew-dir DIR` and
    `run --qcew-dir DIR --codebook PATH --final-years YEAR... --out-dir DIR`. `main` exits 0, or
    2 when an invariant failed or `needs_user` is set.

- [x] **Step 1: Write the failing tests**

In `tests/unit/test_employment_statistics_coverage.py`, replace the import block (everything
above `_SCRIPT = …`) with:

```python
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import polars as pl
import pytest
```

Then append this block to the end of the file, one blank line after the existing last line:

```python
# -------------------------------------------------------------------------------------------------
# Provenance and run
# -------------------------------------------------------------------------------------------------

def test_parse_headers_keeps_the_last_response():
    text = (
        'HTTP/2 301\r\nlocation: https://x\r\n\r\n'
        'HTTP/2 200\r\ncontent-length: 5\r\nlast-modified: Tue, 02 Sep 2025 11:20:46 GMT\r\n\r\n'
    )
    headers = esc.parse_headers(text)
    assert headers['status'] == '200'
    assert headers['content-length'] == '5'
    assert headers['last-modified'] == 'Tue, 02 Sep 2025 11:20:46 GMT'

def test_build_manifest_records_provenance(tmp_path):
    (tmp_path / 'headers').mkdir()
    (tmp_path / 'a.csv').write_bytes(b'hello')
    (tmp_path / 'headers' / 'a.csv.headers').write_text(
        'HTTP/2 200\r\ncontent-length: 5\r\nlast-modified: Tue, 02 Sep 2025 11:20:46 GMT\r\n\r\n'
    )
    [entry] = esc.build_manifest(tmp_path, {'a.csv': 'https://example.test/a.csv'})
    assert entry['bytes'] == 5
    assert entry['sha256'] == hashlib.sha256(b'hello').hexdigest()
    assert entry['last_modified'] == 'Tue, 02 Sep 2025 11:20:46 GMT'
    (tmp_path / 'a.csv').write_bytes(b'hello!')
    with pytest.raises(ValueError, match='Content-Length'):
        esc.build_manifest(tmp_path, {'a.csv': 'https://example.test/a.csv'})

def _vintage_2017_rows():
    rows = [
        ('US000', '5', '10', '11', '', 100, 1000, 100000),
        ('US000', '5', '111110', '18', '', 10, 100, 10000),
        ('US000', '5', '454110', '18', '', 7, 70, 7000),
        ('US000', '5', '238111', '18', '', 4, 40, 4000),
        ('US000', '5', '238112', '18', 'N', 1, 0, 0),
    ]
    return [_record(2021, *row) for row in rows]

def _write_qcew_dir(directory):
    directory.mkdir()
    for year in WINDOW:
        rows = _annual_rows(year)
        with zipfile.ZipFile(directory / f'{year}_annual_singlefile.zip', 'w') as archive:
            archive.writestr(f'{year}.annual.singlefile.csv', _csv_bytes(rows))
        national = [row for row in rows if row['area_fips'] == 'US000']
        (directory / f'{year}_US000_annual.csv').write_bytes(_csv_bytes(national))
    (directory / '2021_US000_annual.csv').write_bytes(_csv_bytes(_vintage_2017_rows()))
    return directory

def test_run_writes_tables_and_decision(tmp_path):
    qcew_dir = _write_qcew_dir(tmp_path / 'qcew')
    codebook = _write_codebook(tmp_path)
    out_dir = tmp_path / 'out'
    decision, failures = esc.run(
        qcew_dir,
        codebook,
        WINDOW,
        out_dir,
        codebook_sha256=_digest(codebook),
        floor=2,
        band=(0, 0)
    )
    assert failures == []
    assert (decision.branch, decision.grain) == ('A', 'state')
    report = json.loads((out_dir / 'coverage.json').read_text())
    assert report['split_codes'] == ['238110', '238120']
    assert {row['estabs_column'] for row in report['conventions']} == {'annual_avg_estabs'}
    assert report['msa_rows'][-1] == {'year': 2025, 'rows': 0}
    vintage = {row['year']: row['outside_codebook'] for row in report['vintage']}
    assert vintage == {2021: 1, 2022: 0, 2023: 0, 2024: 0, 2025: 0}
    assert '<!-- decision:begin -->' in (out_dir / 'decision.md').read_text()
    assert '### Decision inputs' in (out_dir / 'tables.md').read_text()

def test_main_exit_code_signals_stop_and_ask(tmp_path, monkeypatch):
    argv = [
        'run', '--qcew-dir',
        str(tmp_path), '--codebook',
        str(tmp_path / 'codebook.parquet'), '--final-years', '2022', '--out-dir',
        str(tmp_path / 'out')
    ]
    clean = esc.Decision('A', 'national', True, True, False, ())
    asking = esc.Decision('A', 'national', True, True, True, ())
    monkeypatch.setattr(esc, 'run', lambda *args: (clean, []))
    assert esc.main(argv) == 0
    monkeypatch.setattr(esc, 'run', lambda *args: (clean, ['2024: an invariant failed']))
    assert esc.main(argv) == 2
    monkeypatch.setattr(esc, 'run', lambda *args: (asking, []))
    assert esc.main(argv) == 2
```

- [x] **Step 2: Run the tests to verify the new ones fail**

Run: `uv run pytest tests/unit/test_employment_statistics_coverage.py -q`
Expected: `4 failed, 26 passed`, with each failure an `AttributeError` naming `parse_headers`,
`build_manifest` or `run`.

- [x] **Step 3: Implement provenance, the report and the CLI**

In `scripts/employment_statistics_coverage.py`, replace the import block (everything between
the module docstring and `logger = logging.getLogger(__name__)`) with:

```python
import argparse
import hashlib
import json
import logging
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection, Mapping, Sequence

import polars as pl
```

Then append this block to the end of the file, one blank line after the existing last line:

```python
# -------------------------------------------------------------------------------------------------
# Provenance
# -------------------------------------------------------------------------------------------------

def parse_headers(text: str) -> dict[str, str]:
    '''Parse a `curl -D` header dump; after redirects, the last response wins.'''
    blocks = [block for block in text.replace('\r\n', '\n').split('\n\n') if block.strip()]
    status_line, *field_lines = blocks[-1].splitlines()
    fields = {'status': status_line.split()[1]}
    for line in field_lines:
        name, _, value = line.partition(':')
        fields[name.strip().lower()] = value.strip()
    return fields

def build_manifest(qcew_dir: Path, sources: Mapping[str, str] = SOURCES) -> list[dict[str, object]]:
    '''Provenance of each source file: URL, bytes, sha256, Last-Modified and download time.'''
    entries = []
    for name, url in sources.items():
        path = qcew_dir / name
        headers_path = qcew_dir / 'headers' / f'{name}.headers'
        headers = parse_headers(headers_path.read_text())
        size = path.stat().st_size
        if headers['status'] != '200':
            raise ValueError(f'{name}: HTTP status {headers["status"]}')
        if 'content-length' in headers and int(headers['content-length']) != size:
            raise ValueError(
                f'{name}: {size} bytes on disk, Content-Length '
                f'{headers["content-length"]}'
            )
        downloaded = datetime.fromtimestamp(headers_path.stat().st_mtime, tz=timezone.utc)
        entries.append(
            {
                'file': name,
                'url': url,
                'bytes': size,
                'sha256': sha256_file(path),
                'last_modified': headers.get('last-modified'),
                'downloaded_at': downloaded.isoformat(timespec='seconds'),
            }
        )
    return entries

# -------------------------------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------------------------------

def _cell(value: object) -> str:
    if isinstance(value, float):
        return f'{value:.4f}'
    if isinstance(value, (list, tuple)):
        return ', '.join(map(str, value)) or '-'
    return '-' if value is None else str(value)

def markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return '_none_\n'
    columns = list(rows[0])
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + ' --- |' * len(columns)]
    lines += ['| ' + ' | '.join(_cell(row[column]) for column in columns) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'

def render_tables(report: Mapping[str, object]) -> str:
    failures = [{'failure': failure} for failure in report['failures']]
    split = [{'code': code} for code in report['split_codes']]
    sections = [
        ('Invariant failures', failures),
        ('File conventions (six-digit rows)', report['conventions']),
        ('Vintage check (national six-digit codes)', report['vintage']),
        ('Split codes recovered from their five-digit parent', split),
        ('National grain: codebook codes by status', report['national_status']),
        ('Private cells by grain and year', report['area_coverage']),
        ('Establishments of disclosed and suppressed private cells', report['size_by_status']),
        ('Codes with no private national cell (last year)', report['private_gaps']),
        ('Connecticut county-equivalents', report['connecticut']),
        ('MSA six-digit rows per year', report['msa_rows']),
        ('Decision inputs', report['summaries']),
        ('Codes excluded at the chosen grain', report['excluded']),
    ]
    return '\n'.join(f'### {title}\n\n{markdown_table(rows)}' for title, rows in sections)

# -------------------------------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------------------------------

def run(
    qcew_dir: Path,
    codebook: Path,
    final_years: Collection[int],
    out_dir: Path,
    window: Sequence[int] = WINDOW,
    codebook_sha256: str = CODEBOOK_SHA256,
    floor: int = SURVIVAL_FLOOR,
    band: tuple[int, int] = ASK_BAND,
) -> tuple[Decision, list[str]]:
    '''Compute every table and the decision; write coverage.json, tables.md and decision.md.'''
    universe = load_universe(codebook, codebook_sha256)
    slices = {
        year: read_annual_csv((qcew_dir / f'{year}_US000_annual.csv').read_bytes())
        for year in (VINTAGE_CHECK_YEAR, *window)
    }
    published = {year: national_six_digit_codes(frame) for year, frame in slices.items()}
    split = find_split_codes(published[max(window)], universe)
    failures = [
        f'{year}: split codes {codes} differ from {max(window)}' for year in window
        if (codes := find_split_codes(published[year], universe)) != split
    ]
    parts: dict[str, list[pl.DataFrame]] = {grain: [] for grain in GRAINS}
    conventions = []
    for year in window:
        logger.info('reading %s', year)
        path = qcew_dir / f'{year}_annual_singlefile.zip'
        frame = read_singlefile_zip(path)
        estabs_column = resolve_estabs_column(singlefile_header(path))
        conventions.append({**file_conventions(frame, year), 'estabs_column': estabs_column})
        failures += check_invariants(frame, year)
        failures += compare_national_slices(frame, slices[year], year)
        for grain in GRAINS:
            parts[grain].append(grain_cells(frame, universe, split, grain))
    cells = {grain: pl.concat(frames) for grain, frames in parts.items()}
    summaries = [summarize_grain(cells[grain], grain, window) for grain in GRAINS]
    decision = decide(summaries, window, final_years, floor, band)
    chosen = decision.grain or 'national'
    msa_counts = {year: cells['msa'].filter(pl.col('year') == year).height for year in window}
    msa_rows = [{'year': year, 'rows': count} for year, count in msa_counts.items()]
    summary_rows = [{**asdict(summary), 'seen': summary.seen} for summary in summaries]
    report = {
        'window': list(window),
        'final_years': sorted(final_years),
        'codebook_sha256': codebook_sha256,
        'failures': failures,
        'conventions': conventions,
        'split_codes': list(split),
        'vintage': vintage_report(published, universe, split),
        'national_status': [
            row for year in window for own in OWNERSHIPS
            for row in code_status_counts(cells['national'], universe, year, own)
        ],
        'area_coverage': [
            area_coverage(cells[grain], universe, grain, year) for grain in GRAINS
            for year in window
        ],
        'size_by_status': [
            row for grain in GRAINS for year in window
            for row in size_by_status(cells[grain], grain, year)
        ],
        'private_gaps': private_gaps(cells['national'], universe, max(window)),
        'connecticut': connecticut_areas(cells['county']),
        'msa_rows': msa_rows,
        'summaries': summary_rows,
        'decision': asdict(decision),
        'excluded': excluded_codes(cells[chosen], universe, window),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'coverage.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    (out_dir / 'tables.md').write_text(render_tables(report))
    (out_dir / 'decision.md').write_text(render_decision(decision, summaries, window))
    for failure in failures:
        logger.warning('invariant failed: %s', failure)
    return decision, failures

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='QCEW six-digit coverage for roadmap Stage 1.')
    commands = parser.add_subparsers(dest='command', required=True)
    manifest = commands.add_parser('manifest', help='record provenance of the downloaded files')
    manifest.add_argument('--qcew-dir', type=Path, required=True)
    coverage = commands.add_parser('run', help='compute the tables and the Req 2 decision')
    coverage.add_argument('--qcew-dir', type=Path, required=True)
    coverage.add_argument('--codebook', type=Path, required=True)
    coverage.add_argument('--final-years', type=int, nargs='+', required=True)
    coverage.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if args.command == 'manifest':
        path = args.qcew_dir / 'MANIFEST.json'
        path.write_text(json.dumps(build_manifest(args.qcew_dir), indent=2) + '\n')
        logger.info('wrote %s', path)
        return 0
    decision, failures = run(args.qcew_dir, args.codebook, args.final_years, args.out_dir)
    logger.info('branch %s at grain %s', decision.branch, decision.grain)
    if failures or decision.needs_user:
        review = 'required' if decision.needs_user else 'not required'
        logger.warning('stop and ask: %d invariant failures; user review %s', len(failures), review)
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
```

- [x] **Step 4: Run the tests to verify they pass, on both CI Python versions**

Run: `uv run pytest tests/unit/test_employment_statistics_coverage.py -q`
Expected: `30 passed`.

The test file needs only Polars and pytest, so Python 3.10 can run it without the torch
environment:

Run: `uv run --no-project --python 3.10 --with polars==1.35.1 --with pytest python -m pytest tests/unit/test_employment_statistics_coverage.py --noconftest -q -p no:cacheprovider`
Expected: `30 passed`. `--noconftest` skips `tests/conftest.py`, which imports torch. `uv` may
fetch a 3.10 interpreter first.

- [x] **Step 5: Run the full suite, then lint and format**

> Deviation: the first format check failed on a transcription slip in the test file (`_SCRIPT =Path`); restoring the plan's text made it clean, with the 30 tests still passing.

Run: `uv run pytest -n auto -q`
Expected: every test passes; the count is the pre-existing suite plus 30. This plan touches
nothing under `src/`, so a failure outside `test_employment_statistics_coverage.py` is not its
doing. Record that failure with its output and ask your human partner before continuing.

Run: `./scripts/format_code.sh --check scripts/employment_statistics_coverage.py tests/unit/test_employment_statistics_coverage.py`
Expected: exit 0 with `Clean: no lint issues and no formatting changes.`

- [x] **Step 6: Commit**

```bash
git add scripts/employment_statistics_coverage.py tests/unit/test_employment_statistics_coverage.py
git commit -m "feat(scripts): add provenance manifest, report and CLI for QCEW coverage"
```

### Task 5: Acquire the BLS files and record the sources (controller, inline)

**Files:**

- Create: `specs/findings/employment-statistics-coverage.md` (skeleton plus Sources)
- Outside the repo: `~/Downloads/Data/QCEW/`

**Interfaces:**

- Consumes: permission from Pre-flight step 4, and the `manifest` subcommand from Task 4.
- Produces:
  - the 13 files, their `headers/*.headers` dumps and `MANIFEST.json`
  - the finding skeleton, with its Sources tables filled in

The files, as HEAD requests saw them on 2026-09-24:

| File | URL | Bytes | Last-Modified |
| --- | --- | ---: | --- |
| `2022_annual_singlefile.zip` | `https://data.bls.gov/cew/data/files/2022/csv/2022_annual_singlefile.zip` | 77,024,919 | Thu, 31 Aug 2023 13:49:04 GMT |
| `2023_annual_singlefile.zip` | `https://data.bls.gov/cew/data/files/2023/csv/2023_annual_singlefile.zip` | 82,932,544 | Thu, 29 Aug 2024 14:43:53 GMT |
| `2024_annual_singlefile.zip` | `https://data.bls.gov/cew/data/files/2024/csv/2024_annual_singlefile.zip` | 74,697,761 | Tue, 02 Sep 2025 11:20:46 GMT |
| `2025_annual_singlefile.zip` | `https://data.bls.gov/cew/data/files/2025/csv/2025_annual_singlefile.zip` | 62,799,396 | Fri, 21 Aug 2026 13:12:50 GMT |
| `2021_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2021/a/area/US000.csv` | 846,843 | Wed, 31 Aug 2022 19:18:44 GMT |
| `2022_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2022/a/area/US000.csv` | 823,163 | Thu, 31 Aug 2023 14:13:44 GMT |
| `2023_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2023/a/area/US000.csv` | 849,504 | Wed, 28 Aug 2024 21:18:54 GMT |
| `2024_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2024/a/area/US000.csv` | 847,197 | Tue, 02 Sep 2025 11:07:10 GMT |
| `2025_US000_annual.csv` | `https://data.bls.gov/cew/data/api/2025/a/area/US000.csv` | 841,756 | Fri, 21 Aug 2026 12:51:16 GMT |
| `industry_titles.csv` | `https://data.bls.gov/cew/doc/titles/industry/industry_titles.csv` | 164,248 | Wed, 31 Aug 2022 17:09:54 GMT |
| `agglevel_titles.csv` | `https://data.bls.gov/cew/doc/titles/agglevel/agglevel_titles.csv` | 2,750 | Fri, 15 Oct 2010 05:00:00 GMT |
| `area_titles.csv` | `https://data.bls.gov/cew/doc/titles/area/area_titles.csv` | 343,559 | Fri, 06 Sep 2024 17:10:14 GMT |
| `ownership_titles.csv` | `https://data.bls.gov/cew/doc/titles/ownership/ownership_titles.csv` | 230 | Fri, 15 Oct 2010 05:00:00 GMT |

- [x] **Step 1: Create the directories**

Run: `mkdir -p ~/Downloads/Data/QCEW/headers ~/Downloads/Data/QCEW/coverage`

- [x] **Step 2: Download the 13 files**

Run each line below as its own Bash call. `-f` makes curl fail on an HTTP error, and `-D` keeps
the response headers for the manifest.

```bash
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2022_annual_singlefile.zip.headers -o ~/Downloads/Data/QCEW/2022_annual_singlefile.zip https://data.bls.gov/cew/data/files/2022/csv/2022_annual_singlefile.zip
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2023_annual_singlefile.zip.headers -o ~/Downloads/Data/QCEW/2023_annual_singlefile.zip https://data.bls.gov/cew/data/files/2023/csv/2023_annual_singlefile.zip
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2024_annual_singlefile.zip.headers -o ~/Downloads/Data/QCEW/2024_annual_singlefile.zip https://data.bls.gov/cew/data/files/2024/csv/2024_annual_singlefile.zip
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2025_annual_singlefile.zip.headers -o ~/Downloads/Data/QCEW/2025_annual_singlefile.zip https://data.bls.gov/cew/data/files/2025/csv/2025_annual_singlefile.zip
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2021_US000_annual.csv.headers -o ~/Downloads/Data/QCEW/2021_US000_annual.csv https://data.bls.gov/cew/data/api/2021/a/area/US000.csv
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2022_US000_annual.csv.headers -o ~/Downloads/Data/QCEW/2022_US000_annual.csv https://data.bls.gov/cew/data/api/2022/a/area/US000.csv
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2023_US000_annual.csv.headers -o ~/Downloads/Data/QCEW/2023_US000_annual.csv https://data.bls.gov/cew/data/api/2023/a/area/US000.csv
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2024_US000_annual.csv.headers -o ~/Downloads/Data/QCEW/2024_US000_annual.csv https://data.bls.gov/cew/data/api/2024/a/area/US000.csv
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/2025_US000_annual.csv.headers -o ~/Downloads/Data/QCEW/2025_US000_annual.csv https://data.bls.gov/cew/data/api/2025/a/area/US000.csv
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/industry_titles.csv.headers -o ~/Downloads/Data/QCEW/industry_titles.csv https://data.bls.gov/cew/doc/titles/industry/industry_titles.csv
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/agglevel_titles.csv.headers -o ~/Downloads/Data/QCEW/agglevel_titles.csv https://data.bls.gov/cew/doc/titles/agglevel/agglevel_titles.csv
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/area_titles.csv.headers -o ~/Downloads/Data/QCEW/area_titles.csv https://data.bls.gov/cew/doc/titles/area/area_titles.csv
curl -sS -f -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) naics-embedder-research' -D ~/Downloads/Data/QCEW/headers/ownership_titles.csv.headers -o ~/Downloads/Data/QCEW/ownership_titles.csv https://data.bls.gov/cew/doc/titles/ownership/ownership_titles.csv
```

Expected: every call exits 0, and each file's size matches the table.

- [x] **Step 3: Build the provenance manifest**

Run: `uv run python scripts/employment_statistics_coverage.py manifest --qcew-dir ~/Downloads/Data/QCEW`
Expected: `INFO wrote …/QCEW/MANIFEST.json`. The command raises if a status is not 200 or a size
disagrees with its Content-Length.

Compare each entry's `bytes` and `last_modified` with the table. Any difference is stop-and-ask
condition 7.

- [x] **Step 4: Confirm the lookup files**

> Deviation: `industry_titles.csv` lists 38 BLS 238 codes and no plain 238 code (not 34 and two), so the codebook has 19 split codes, run under the project owner's ruling of 2026-09-24; the 47/48 titles read "by ownership sector", and an added single-file scan shows MSA rows carry only `own_code` 5.

Run each command below as its own call, and keep the output for the finding's section 6:

- `grep -E '^"?(11|17|18|47|48|57|58|77|78)"?,' ~/Downloads/Data/QCEW/agglevel_titles.csv`.
  Expected: the titles name national, MSA (private), statewide and county at the five- and
  six-digit levels, and 11 names the national by-ownership total. A contradiction is
  stop-and-ask condition 6.
- `grep -E '^"?238[0-9]{3}"?,' ~/Downloads/Data/QCEW/industry_titles.csv`. Expected: the BLS
  residential and nonresidential six-digit codes, 34 of them, and exactly two plain NAICS 238
  codes ending in 0. Record those two codes.
- `grep -c -E '^"?999999"?,' ~/Downloads/Data/QCEW/industry_titles.csv`. Expected: `1`.
- `grep -i -E 'unknown|undefined' ~/Downloads/Data/QCEW/area_titles.csv`. Expected: county
  codes ending in `999`, one per state or territory.
- `grep -E '^"?09(001|003|005|007|009|011|013|015|110|120|130|140|150|160|170|180|190)"?,' ~/Downloads/Data/QCEW/area_titles.csv`.
  Expected: the eight legacy Connecticut counties and the nine planning regions.
- `cat ~/Downloads/Data/QCEW/ownership_titles.csv`. Expected: code 5 is private ownership, and
  codes 1–3 are federal, state and local government.

- [x] **Step 5: Read the BLS pages and quote them**

> Deviation: added the "BLS and QCEW NAICS Differences" page (the 238 codes, and the codes not used in the US or by BLS) and extra release-note rows (NAICS 2022, Colorado); OEWS was quoted from its handbook chapter only, and `oes/tables.htm` was not read.

For each page below, open it in the built-in browser (`preview_start` with its URL) and read it
with `get_page_text`. Record the URL, the read date (UTC) and one verbatim quote establishing the
fact.

1. `https://www.bls.gov/cew/classifications/industry/naics-2022.htm`: the introduction sentence,
   dated 2022-09-07, and the "21 sectors and 1,030 industries" sentence.
2. `https://www.bls.gov/cew/classifications/industry/`: the sentence saying which NAICS version
   covers 2017–2021 and which covers 2022 forward.
3. `https://www.bls.gov/cew/release-calendar.htm`: the finality sentence, and the release date of
   first-quarter 2026 data.
4. `https://www.bls.gov/cew/notices/2025/change-in-the-presentation-of-metropolitan-statistical-area-data-in-qcew.htm`:
   the sentence starting "Beginning with third quarter 2025 data".
5. `https://www.bls.gov/cew/about-data/news-release-notes.htm`: the Connecticut planning-region
   entry (2024-08-21) and the MSA entry (2026-03-10).
6. For section 5's screen of other series, find and quote the sentence stating the industry
   detail each program publishes. Where a URL has moved, use the BLS site search in the browser
   and record the page actually read.
   - CES: `https://www.bls.gov/web/empsit/cesseriespub.htm`, or the CES presentation chapter
     `https://www.bls.gov/opub/hom/ces/presentation.htm`.
   - OEWS: `https://www.bls.gov/opub/hom/oews/presentation.htm` and `https://www.bls.gov/oes/tables.htm`.
   - BED: `https://www.bls.gov/opub/hom/bdm/presentation.htm`.

If a page's text contradicts the facts table, the page read during execution wins. Stop and ask
before the run when it changes the window, the finality of any year, or the grain list.

- [x] **Step 6: Write the finding skeleton**

Create `specs/findings/employment-statistics-coverage.md` with the content below. Fill both
Sources tables with exact values: `MANIFEST.json` for the files, and Step 5 for the pages. Leave
every other section's instruction comment in place for Tasks 6 and 7.

```markdown
# Employment-statistics coverage: finding

**Status: DRAFT.** Roadmap Stage 1 (`specs/naics-embedding-roadmap.md`). This finding discharges
Verification "Employment-statistics coverage" and the Req 2 (open) item of
`specs/naics-embedding.md` (d9126ce). It is produced by plan 3 with
`scripts/employment_statistics_coverage.py`. Stage 3 reads the decision block verbatim.

## Decision for Stage 3

<!-- Task 6: paste coverage/decision.md here verbatim. -->

## 1. Reference years published on NAICS 2022 at six digits

<!-- Task 7 -->

## 2. Grains at which six-digit series are published

<!-- Task 7 -->

## 3. Suppressed share per year, grain and series

<!-- Task 7 -->

## 4. Panel population, row grain, time-respecting outcome and seen-code regime

<!-- Task 7 -->

## 5. Other public employment series screened

<!-- Task 7 -->

## 6. File conventions verified

<!-- Task 7 -->

## 7. Consequences for later stages

<!-- Task 7 -->

## Sources

### Files

Read from `~/Downloads/Data/QCEW/`. Download times are the header dumps' modification times
(UTC).

| File | URL | Bytes | sha256 | Last-Modified | Downloaded (UTC) |
| --- | --- | ---: | --- | --- | --- |

### Pages

| Page | URL | Read (UTC) | Quote |
| --- | --- | --- | --- |

## Reproduction

<!-- Task 7 -->

## Appendix: generated tables

<!-- Task 6: paste coverage/tables.md here verbatim. -->
```

- [x] **Step 7: Commit**

```bash
git add specs/findings/employment-statistics-coverage.md
git commit -m "docs(findings): record the QCEW sources for the coverage finding"
```

### Task 6: Run the analysis and verify it (controller, inline)

**Files:**

- Modify: `specs/findings/employment-statistics-coverage.md` (the decision block and the
  appendix)
- Outside the repo: `~/Downloads/Data/QCEW/coverage/`, holding `coverage.json`, `tables.md` and
  `decision.md`
- Scratch: `/tmp/esc-recount.py`, moved to the Trash in Task 8

**Interfaces:**

- Consumes: the files and manifest from Task 5, and the `run` subcommand from Task 4.
- Produces the decision block and tables that Task 7 cites.

- [x] **Step 1: Run the analysis**

> Deviation: the run exited 2 on two national-total employment invariant failures (+46 in 2022, +27 in 2023), which are annual-average rounding; under the project owner's ruling of 2026-09-24 the run stands, with no re-run and no code change.

Run: `uv run python scripts/employment_statistics_coverage.py run --qcew-dir ~/Downloads/Data/QCEW --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet --final-years 2022 2023 2024 2025 --out-dir ~/Downloads/Data/QCEW/coverage`

Pass only the years that Task 5's finality quote makes final. Expected: exit 0, and a last INFO
line reading `branch <A|B|C> at grain <grain>`.

Exit 2 means an invariant failed or `needs_user` is set: stop, show your human partner
`coverage/decision.md` and the failures, and wait. Never re-run with changed parameters.

- [x] **Step 2: Check the stop-and-ask conditions against `coverage/tables.md`**

> Deviation: "Invariant failures" holds those two rounding rows, not `_none_`, and "Split codes" lists 19 codes, not 17; both stand under the rulings of 2026-09-24.

Read `~/Downloads/Data/QCEW/coverage/tables.md` and confirm each item. Any miss is a stop.

- **Invariant failures:** `_none_`.
- **File conventions:** for every year:
  - `own_code_0_rows` is 0 (condition 4);
  - `suppressed_rows_with_emp_or_wages` is 0;
  - `estabs_column` is the same in every year;
  - record which disclosure codes appear and how often.
- **Private cells by grain and year:** `other_cells / published_cells` stays at 1 % or less in
  every row (condition 5).
- **Vintage check:** 2021's `outside_codebook` is greater than 0; 2022–2025 are 0 (condition 3).
- **Split codes:** 17 codes, each starting `238` (condition 2).
- **Codes with no private national cell:** 60 rows or fewer (condition 4).
- **Connecticut county-equivalents:** 2022–2023 show 8 legacy counties and 0 regions; 2024–2025
  show 0 and 9 (condition 6).
- **MSA six-digit rows per year:** more than 0 for 2022–2024, and 0 for 2025 (condition 6).

- [x] **Step 3: Recount the national grain independently**

This recount uses the stdlib `csv` parser and none of the script's logic, so it catches a
reading or status bug in the script. Create `/tmp/esc-recount.py` with this content:

```python
'''Independent recount of national private statuses (plan 3, Task 6 Step 3).'''
import csv
import json
import sys
from pathlib import Path

import polars as pl

STATUS = {'': 'disclosed', 'N': 'suppressed'}

qcew_dir, codebook = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
report = json.loads((qcew_dir / 'coverage' / 'coverage.json').read_text())
split = set(report['split_codes'])
six_digit = [code for code in pl.read_parquet(codebook)['code'].to_list() if len(code) == 6]
for year in report['window']:
    flags = {}
    with (qcew_dir / f'{year}_US000_annual.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['own_code'].strip() == '5' and row['agglvl_code'].strip() in ('17', '18'):
                flags[row['industry_code'].strip()] = row['disclosure_code'].strip()
    counts = {'disclosed': 0, 'suppressed': 0, 'other': 0, 'absent': 0}
    for code in six_digit:
        flag = flags.get(code[:5] if code in split else code)
        counts['absent' if flag is None else STATUS.get(flag, 'other')] += 1
    print(year, counts)
```

Run: `uv run python /tmp/esc-recount.py ~/Downloads/Data/QCEW /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v1/18403d29-3b23-444e-9e81-371d0ca8b7ea/naics_codebook.parquet`

Expected: one line per window year. Each line's `disclosed`, `suppressed`, `other` and `absent`
equal the `employment` row for that year and `own_code` 5 in "National grain: codebook codes by
status". Any difference is a stop.

- [x] **Step 4: Paste the outputs into the finding**

In `specs/findings/employment-statistics-coverage.md`:

- replace `<!-- Task 6: paste coverage/decision.md here verbatim. -->` with the full content of
  `coverage/decision.md`;
- replace `<!-- Task 6: paste coverage/tables.md here verbatim. -->` with the full content of
  `coverage/tables.md`.

Change nothing inside either paste.

- [x] **Step 5: Commit**

```bash
git add specs/findings/employment-statistics-coverage.md
git commit -m "docs(findings): add the coverage run's decision and tables"
```

### Task 7: Write the finding (controller, inline)

**Files:**

- Modify: `specs/findings/employment-statistics-coverage.md`

**Interfaces:**

- Consumes: the decision block, the appendix tables and the Sources from Tasks 5 and 6.
- Produces the finished finding that Stage 3 reads.

Every number in sections 1–4 and 6 must appear in the decision block or in a named appendix
table, and the prose cites the table's title. Markdown rules:

- one blank line around headings, two-space list indents, at most one consecutive blank line;
- prose lines at 100 characters or fewer (tables are exempt).

- [x] **Step 1: Section 1, reference years**

State the NAICS 2022 reference years with six-digit annual files (the window) and their
finality, with the two dates. Then give the evidence for the boundary:

- the BLS quotes from pages 1–2;
- the "Vintage check" row for 2021, whose count and examples show codes outside NAICS 2022;
- the rows for 2022–2025, with 0 codes outside it.

Note that 2026 has only a first-quarter file, so the window ends at 2025. Cite the Req 2 line
that bars earlier-vintage training without a concordance.

- [x] **Step 2: Section 2, grains**

List each grain with its aggregation levels, its area count per year (the `areas` column of
"Private cells by grain and year") and its breaks:

- by year, the national grain;
- by area:
  - state;
  - county, with Connecticut's recode in 2024;
  - MSA, which moves from 13-01 to 23-01 in 2024 and has no six-digit rows in 2025.

Record that CSA and MicroSA carry no industry detail, citing `agglevel_titles.csv`. Also record
that six-digit cells are published by ownership, with no total-covered row.

- [x] **Step 3: Section 3, suppressed shares**

Write one table with these columns, from "Private cells by grain and year":

- `Grain`, `Year`, `Areas`, `Published cells`;
- `Suppressed share of cells (employment and wages)`, from `suppressed_share`;
- `Suppressed share of cells (establishments)`, from `estabs_suppressed_share`;
- `Codes with no usable cell (of 1,012)`, from `codes_without_usable` and
  `share_without_usable`.

Below it, give the national grain by ownership from "National grain: codebook codes by status",
one line per year: disclosed, suppressed, other and absent for `own_code` 5. Explain that
employment and wages share one disclosure flag, so their rows are identical, and that
establishments survive suppression when positive. Close with what "Establishments of disclosed
and suppressed private cells" shows about which cells suppression removes: compare the medians.

- [x] **Step 4: Section 4, population, row grain, time-respecting outcome, seen-code regime**

Restate the decision block's branch, row grain and populations in prose, with the reasons from
its **Reasons** list. Then add:

- "Time-respecting outcome: yes or no", with the reason: window length, finality, and the
  time-eligible count against the floor.
- "Seen-code regime: yes or no", with the reason: the seen count against the floor, and the grain
  chosen by the lowest mean suppressed share.
- The excluded codes: their count, their reasons from "Codes excluded at the chosen grain", and
  the government-only codes from "Codes with no private national cell".
- For any "no", the Req 2 branch sentence the panel therefore follows, quoted from the Global
  Constraints.

- [x] **Step 5: Section 5, other series screened**

Write one short paragraph each for CES, OEWS and BED. Each gives the quote from Task 5 Step 5
item 6, with its URL and read date, and says why the program cannot serve as the panel's source.

- Reasons that apply: industry detail short of all 1,012 six-digit codes; a sample-based
  estimate, not a census count; or no establishment and wage covariates on the same rows.
- Close with D1: the covariates come from the same QCEW rows, so QCEW is the panel's source.
- No downloads and no tables here.

- [x] **Step 6: Section 6, file conventions**

> Deviation: section 6 reports no direct 238 codes and 38 BLS codes (not two and 34), adds the rounding evidence, and cites a new "Appendix: checks outside the script" (lookup files, single-file scan, national reconciliation, recount).

Cover:

- the "File conventions (six-digit rows)" table: disclosure codes with their counts, the
  establishment column's name, no `own_code` 0 rows, and zero-filled employment and wages on `N`
  rows;
- the split codes, recovered from their five-digit unary parents, plus the two direct 238 codes
  and the 34 BLS children from Task 5 Step 4;
- 999999, the unknown-county `999` codes, and the Connecticut and MSA tables;
- the reconciliation checks: detail never exceeds totals, and the single files and the US000
  slices agree;
- the Task 6 Step 3 recount.

- [x] **Step 7: Section 7, consequences for later stages**

> Deviation: the D1 bullet adds that no national private cell is suppressed, the `metrics/qcew.py` bullet adds that it pools every grain and reads zero-filled suppressed cells as zeros, and an extra bullet notes that suppression truncates area-grain outcomes from below.

Write these bullets, filled from the results:

- Stage 3 takes the decision block's branch, population and row grain.
- D1's covariates: a usable row carries both employment and wages under one flag. Establishment
  counts stay published on suppressed rows, but a suppressed row is not usable.
- `metrics/qcew.py` reads `tot_wages` where the files say `total_annual_wages`, and keeps one row
  per code: the rejected definition.
- An area design must handle Connecticut's 2024 recode. An MSA design has only 2022–2024 and a
  2023/2024 break.
- The 2026 annual file does not exist yet, so the window ends at 2025.
- derive-roadmap's resume step re-validates later stages against this finding; no other stage
  entry is edited here.

- [x] **Step 8: Reproduction, and the status line**

> Deviation: Reproduction also gives the commands behind "Appendix: checks outside the script" and notes that the run exits 2.

Under Reproduction:

- give the exact `manifest` and `run` commands from Tasks 5 and 6;
- give the script's commit, from `git log -1 --format=%h -- scripts/employment_statistics_coverage.py`;
- say that the inputs are the files and hashes under Sources.

Change the status line to `**Status: FINAL (YYYY-MM-DD).**`, using today's date.

- [x] **Step 9: Check the finding**

Run: `grep -n -F '<!-- Task' specs/findings/employment-statistics-coverage.md`
Expected: no output, meaning no instruction comment is left.

Read the finding once, top to bottom, against the Verification item's four bullets. Each bullet
must be answered in sections 1–4, with a number traceable to a table or the decision block, and
every "no" must carry its reason.

- [x] **Step 10: Commit**

> Deviation: a review after this commit caught two errors; follow-up commit 77ccf60 qualifies section 1's vintage counts (expected extras excluded) and gives the recount's positional command.

```bash
git add specs/findings/employment-statistics-coverage.md
git commit -m "docs(findings): write the employment-statistics coverage finding"
```

### Task 8: Tick the roadmap, complete the plan, integrate (controller, inline)

**Files:**

- Modify: `specs/naics-embedding-roadmap.md`
- Move: `specs/plans/3-employment-statistics-coverage.md` to
  `specs/plans/completed/3-employment-statistics-coverage.md`

**Interfaces:**

- Consumes: the finished finding.
- Produces the Stage 1 stamp that derive-roadmap's resume step reads.

- [x] **Step 1: Tick Stage 1 and fix its Produces path**

In `specs/naics-embedding-roadmap.md`, replace `- [ ] Stage 1: Employment-statistics coverage`
with `- [x] Stage 1: Employment-statistics coverage`.

Then replace these lines:

```text
      Produces: A written finding (`reports/employment-statistics-coverage.md` unless the stage
      spec says otherwise) recording years, grains, the suppressed share per year, grain and
      series, the panel population and row grain, whether a time-respecting outcome exists,
      whether the seen-code regime can run, and reasons for each "no". Stage 3 reads it
      verbatim.
```

with:

```text
      Produces: A written finding (`specs/findings/employment-statistics-coverage.md`; the
      default `reports/` path is gitignored) recording years, grains, the suppressed share per
      year, grain and series, the panel population and row grain, whether a time-respecting
      outcome exists, whether the seen-code regime can run, and reasons for each "no". Stage 3
      reads it verbatim.
```

- [x] **Step 2: Add the completion stamp under Stage 1's entry**

Replace:

```text
      Exit: The four items of Verification "Employment-statistics coverage" are recorded with
      the source files and the dates they were read; the chosen Req 2 branch is named.
      ROUTING: writing-plans
```

with this, using today's date for `YYYY-MM-DD`:

```text
      Exit: The four items of Verification "Employment-statistics coverage" are recorded with
      the source files and the dates they were read; the chosen Req 2 branch is named.
      ROUTING: writing-plans
      Stage 1: COMPLETE (YYYY-MM-DD) — implemented by plan 3
      (specs/plans/completed/3-employment-statistics-coverage.md). Next: resume the roadmap.
```

Edit no other stage entry: re-validating later stages is derive-roadmap's resume step, which
your human partner starts.

- [x] **Step 3: Commit the roadmap**

```bash
git add specs/naics-embedding-roadmap.md
git commit -m "docs(roadmap): tick Stage 1 and stamp its completion"
```

- [x] **Step 4: Run the Plan Completion Protocol (writing-plans)**

Follow its steps in order, with these specifics:

1. **Gate.** Collect skipped steps and unfixed review findings. Ask your human partner about any
   that stalled on their input, as one batch.
2. **Markup.** Tick every completed step in this file and add `> Deviation: …` notes. Put this
   status header at the top, naming the skill used:
   `**Status: COMPLETE (YYYY-MM-DD)** — executed via <skill>; nothing deferred`. If the gate
   deferred something, end it with the deferral note the gate agreed instead.
3. **Deferred items.** This plan implements no earlier item in `specs/deferred_items.md`, so the
   ticking pass edits nothing. If the gate deferred anything, do not edit the file: hand the
   exact entries to your human partner (parallel sessions may be active).
4. **Backlog triage.** Run
   `uv run --no-project --python 3.13 python ~/.claude/skills/writing-plans/scripts/deferred_stats.py`
   from the worktree root and report its summary line. Present the triage rubric (read-only)
   when it reports 20 or more open items or an aged tail.
5. **Retire.** Run `git mv specs/plans/3-employment-statistics-coverage.md specs/plans/completed/`,
   then commit it with the markup:

   ```bash
   git add specs/plans
   git commit -m "chore(specs): retire plan 3"
   ```

   Plain backtick paths, not relative links, keep this plan correct at its new depth.

- [x] **Step 5: Final checks, clean-up, integration**

> Deviation: these checks ran before Step 4's retirement commit, which moves only this file; `/tmp/esc-area-scan.py`, the single-file scan's scratch copy, was trashed with `/tmp/esc-recount.py`.

Run: `uv run pytest -n auto -q`
Expected: all pass, the same count as Task 4 Step 5.

Run: `./scripts/format_code.sh --check --all`
Expected: exit 0, the same check as CI's `lint` job.

Run: `uv run ruff check scripts/employment_statistics_coverage.py`
Expected: `All checks passed!`

Run: `trash /tmp/esc-recount.py` (macOS moves it to the Trash, which is recoverable)

Run: `git log --oneline origin/main..HEAD`
Expected: only this plan's commits, and neither "config" nor "graph config".

Then use finishing-a-development-branch. A PR is the expected route.

- Never push to `main`, and never enable auto-merge.
- This worktree holds no ignored artifact worth keeping: the data and outputs live in
  `~/Downloads/Data/QCEW/`.
- Worktree clean-up follows that skill and your human partner's practice of archiving the
  session once the PR merges.
