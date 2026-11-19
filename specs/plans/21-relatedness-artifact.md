# Relatedness Judging Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 6

**Goal:** Build relatedness's judging panel on fixtures: the co-ownership split by anchor industry
into a validation and a sealed part, the tie-fair NDCG@10 scorer with its reports, the logged reads
and the one logged sealed opening, and the reference choice, margin record, beat-by-δ test and
provenance check as code.

**Architecture:** Five new source modules, each with its own tests.

- **`panels/relatedness_split.py`.** The settings in `conf/data/relatedness_panel.yaml`, the
  salted hash split of the six-digit code, and the split fingerprint.
- **`panels/relatedness_scoring.py`.** The tie-aware expected DCG, NDCG and AUC, the per-anchor
  scores, and `RelatednessResult` with its headline and reports.
- **`panels/relatedness_arms.py`.** The arm protocol and its two adapters: the tree's −D* and
  cosine over a vector table, plain or centered.
- **`panels/relatedness.py`.** Validation of Stage 10's two co-ownership tables, the gains, and
  `RelatednessPanel`: reads logged to the shared `SelectionLog` under `relatedness_co_ownership`,
  table pins and the sealed opening.
- **`decision/relatedness.py`.** The reference choice, the margin record and its margin-first
  check, the beat-by-δ test at `superiority_level(n)`, and the provenance check.

**Tech Stack:** Python 3.10+ (locked env 3.12), numpy (float64), Polars, Pydantic v2, PyYAML,
pytest, mkdocstrings. No new dependency.

**Spec:** `specs/relatedness-artifact.md` (APPROVED 2026-11-18, U48; merged by PR #170 as
`ec02cf9`). Read it with this plan. §n cites that spec. This plan was cut from origin/main
`c055739`. Stage 10 cites embedding-data's `specs/completed/ita-establishment-targets.md`. The
fixture's columns come from embedding-data at `e6e12b2`: `TARGET_SCHEMA`
(`src/embedding_data/target_contract.py:12`), and `METADATA` and `MARGINAL_SCHEMA`
(`src/embedding_data/ita_co_ownership.py:38-52`). Read them with
`git -C ~/Projects/embedding-data show e6e12b2:<path>`. Never open a table there.

## Global Constraints

Every task's requirements include this section.

- **No real table.** No task reads an embedding-data table, a real bundle, the real selection log
  or any file of the main checkout (`/Users/lowell/Projects/naics-embedder`). Tests use
  `tests/fixtures/relatedness.py` and the existing bundle fixtures, and write under `tmp_path`
  (§1, §6 "Fixtures", §10).
- **Edit nothing outside this worktree's branch.** Read embedding-data code with `git show` only.
- **Untouched seams.** These files stay identical to `c055739`: `panels/outcome.py`,
  `panels/ita.py`, `panels/ita_split.py`, `panels/ita_scoring.py`, `panels/decoding.py`,
  `panels/selection_log.py`, `decision/rule.py`, `decision/resampling.py`,
  `decision/records.py`, `decision/decide.py`, `decision/economic.py`,
  `decision/text_to_code.py` and `decision/provenance.py`. So do their tests:
  `test_outcome_panel.py`, `test_outcome_decoding.py`, `test_ita_split.py`,
  `test_ita_inputs.py`, `test_ita_scoring.py`, `test_ita_panel.py`,
  `test_selection_log_guard.py`, `test_decision_rule.py`, `test_decision_resampling.py`,
  `test_decision.py`, `test_decision_economic.py`, `test_decision_text_to_code.py` and
  `test_economic_candidates.py` (§2 anti-goals, §6). Task 10 checks this with `git diff`. Import
  from these modules freely.
- **No new dependency.** The spec still states a `uv.lock` constraint (§2, §4.7, §6). It lapsed
  with the roadmap's Decisions of 2026-11-19 ("(a) No opening, lock lapses (Recommended)"). This
  plan adds no dependency, so the lock does not move.
- **Names and values.**
  - Panel `relatedness_co_ownership`. Parts `validation` (hash 0) and `sealed` (hash 1).
  - Salt `4bfd5195916df66d0e1c619c6507f2ad`, rule version 1, r = 0.05, k = 10, with NDCG@20 as
    a report.
  - Reference arms `tree`, `text-only` and `text-only-centered`, in that tie order (§4.6).
  - Accepted rows: family `co_ownership`, measure `employer_share`, unit type
    `naics_national_industry`, node ids `naics:2022:national_industry:<code>` (§4.2).
  - Table names `ita_co_ownership` and `co_ownership_marginals`.
- **Refuse, never drop.** An input that disagrees with §4.2 raises `ValueError` at panel
  construction, before any log record is written.
- **Means.** Every mean over anchors uses `math.fsum`, so a result does not depend on row order.
- **Style.** Single quotes, YAPF layout, semantic section dividers, one blank line after imports.
  Format touched Python files with `./scripts/format_code.sh <files>`. Never run `ruff format`.
- **Leave alone.** The roadmap, which the resume after Stage 6 edits (§8). The spec, whose lock
  and code-PR-hold lines are the owner's follow-ups (PR #174 body). `specs/deferred_items.md`,
  until the completion protocol. The methodology branch `claude/naics-artifacts-methodology-985a4c`.

## Review Focus

1. **A repeated ordered pair**, as in a table joined from two builds, is refused. It is not
   summed, and the last row does not win. Pinned by
   `test_an_input_that_disagrees_with_the_spec_is_refused[repeated-pair]` (Task 2).
2. **A row with J = 0, or an industry paired with itself,** is refused. Pinned by
   `test_an_input_that_disagrees_with_the_spec_is_refused[zero-j]` and `[self-pair]` (Task 2).
3. **A non-finite similarity**, such as the NaN cosine of a model table's zero row, refuses the
   read and names the arm and the anchor. A silent NaN would put the code in a tie group of its
   own. Pinned by `test_a_non_finite_similarity_is_refused_naming_the_arm_and_anchor` (Task 6).
4. **An anchor that is co-owned with every candidate** has no AUC. The AUC mean skips it, and the
   headline does not change. Pinned by `test_an_anchor_with_no_unrelated_candidate_has_no_auc`
   (Task 4).
5. **An arm that fails during a read** leaves the read logged, and a constructor refusal logs
   nothing. Pinned by `test_a_read_is_logged_before_the_arm_is_asked` and
   `test_constructor_refusals_log_nothing` (Task 6).

## Decisions

- **D1** (owner, brief, 2026-11-18): hold relatedness code PRs that touch `decision/` or
  `conf/data/` until Stage 3 carries its COMPLETE stamp. Met: the stamp is on origin/main at
  `b4a8823` (plan 19, PR #173), so this plan's execution PR is not held. D-C, whether execution
  could start before the stamp, is moot. Applied in: none (met before planning).
- **D2** (owner, brief, 2026-11-19): Stage 6's planning starts only after PR #174 merges (roadmap
  Decisions, 2026-11-19, "Hold both until merge (Recommended)"). Met: #174 merged as `c055739`,
  where this plan was cut. Applied in: Spec line.
- **D3** (owner, brief, 2026-11-19): the `uv.lock` constraint lapsed (roadmap Decisions,
  2026-11-19). Applied in: Global Constraints.

The plan's own calls. Each one is for the owner's review at plan review:

- **P1** (plan, 2026-11-19): the split hashes `f'{salt}\x00{code}'` in UTF-8 with SHA-256, where
  `code` is the bare six digits, never the node id. It reads the first 8 bytes big-endian, modulo
  2. This is the ITA rule of `panels/ita_split.py`. The rule dict has exactly ITA's seven keys, and
  its fingerprint is `abb6c1e6b7ed16e73e081bfe682dadb3364041b95ee417ecb4de6712e38906b1`. The salt
  was drawn once at planning with `secrets.token_hex(16)` (§4.3, §9). Applied in: Task 1.
- **P2** (plan, 2026-11-19): the settings hold `k: Literal[10]`, because U44 fixes the headline.
  NDCG@20 is the module constant `REPORT_K`. The scorer functions take `k` as an argument, so the
  tests can cut a tie group (§4.3, §4.4). Applied in: Tasks 1, 3 and 4.
- **P3** (plan, 2026-11-19; brief default D-A): the provenance contract that Stage 7's export
  writes:
  - The entries are exactly `PROVENANCE_ENTRIES`, and `writer` is `relatedness_export`.
  - `inputs` maps a name to exactly `{path, sha256}`. It needs `codebook` and `text_only`, may add
    `ita_area_mix` (§4.1, after G1), and refuses any other name.
  - The area-mix run goes in a top-level `runs` entry.
  - The expected SHA-256 of the codebook and of the text-only table are arguments, not a read of
    the economic list.

  Applied in: Task 9.
- **P4** (plan, 2026-11-19; brief default D-B): n = 2 is tested in `decision/relatedness.py`
  only. The second family is a hand-built margin record, with `SelectionLog.append` reads under
  the test-only panel name `relatedness_second_family`. Every margin record carries `panel`. The
  margin-first filter and the bootstrap streams key on it. `RelatednessPanel` has no family
  parameter, and no task parses TRI. Applied in: Tasks 7 and 8.
- **P5** (plan, 2026-11-19): the panel refuses more than §4.2 lists, each for a broken build:
  - a null in a column it reads;
  - a repeated marginal node or ordered pair;
  - an industry paired with itself;
  - a row with J < 1.

  Applied in: Task 2.
- **P6** (plan, 2026-11-19): the data fingerprint's payload is canonical JSON of
  `{'candidates': [...], 'anchors': [[code, [[partner, gain], ...]], ...], 'tables': {...}}`.
  Anchors and partners are sorted by code. Each anchor's candidates are the candidate list less
  the anchor, so this payload names §4.5's "candidates and their gains". Applied in: Task 6.
- **P7** (plan, 2026-11-19): ties are exact float64 equality. The centered text arm subtracts the
  mean of the text table's six-digit rows. In the D9 table these rows are the bundle's six-digit
  codes, so the centered values do not depend on which candidates a read asks for (§4.4). Applied
  in: Tasks 3 and 5.
- **P8** (plan, 2026-11-19): AUC is null for an anchor with no candidate at J = 0. The AUC report
  is the mean over the anchors that have one. A read of a part with no anchor is refused before
  it logs (§4.4). Applied in: Tasks 4 and 6.
- **P9** (plan, 2026-11-19): the margin record's self-check also requires each reference arm to
  be read exactly once. The margin-first check counts a reference-role read of a non-reference arm
  as a candidate read, as ITA does. Applied in: Task 7.
- **P10** (plan, 2026-11-19): before it resamples, the beat-by-δ test refuses:
  - fewer than `min_seeds` candidate seeds;
  - results of mixed parts;
  - a result whose panel is not its family's key;
  - a candidate seed read on another data fingerprint than its reference.

  It does not compare a read's data fingerprint with the margin's. After an embedding-data rebuild,
  the δ fixed on validation still applies, because the margin-first check would refuse a new
  record (§4.6). Applied in: Task 8.

---

## File structure

| File | Responsibility |
|---|---|
| `conf/data/relatedness_panel.yaml` (create) | Panel name, rule version, salt, r, k |
| `src/naics_embedder/panels/relatedness_split.py` (create) | Settings, split rule, split fingerprint |
| `src/naics_embedder/panels/relatedness_scoring.py` (create) | Tie-aware DCG, NDCG and AUC, per-anchor scores, `RelatednessResult` |
| `src/naics_embedder/panels/relatedness_arms.py` (create) | `RelatednessArm`, `TreeArm`, `VectorArm` |
| `src/naics_embedder/panels/relatedness.py` (create) | Table validation, gains, `RelatednessPanel` |
| `src/naics_embedder/decision/relatedness.py` (create) | Reference, margin record, margin-first, test, provenance |
| `tests/fixtures/relatedness.py` (create) | Stage 10's two co-ownership tables with every column and dtype; D* for fixture codes |
| `tests/unit/test_relatedness_split.py`, `test_relatedness_inputs.py`, `test_relatedness_scoring.py`, `test_relatedness_arms.py`, `test_relatedness_panel.py`, `test_decision_relatedness.py` (create) | One file per module seam |
| `docs/api/relatedness_panel.md` (create), `docs/api/decision.md`, `docs/.nav.yml`, `CLAUDE.md`, `README.md`, `tests/README.md` (modify) | API pages, guide and counts |

There is no CLI command and no edit to `cli/commands/tools.py`. Stage 6b adds the read command
(§2 Non-goals).

---

### Task 1: Settings, the split and the fixture tables

**Files:**
- Create: `conf/data/relatedness_panel.yaml`, `src/naics_embedder/panels/relatedness_split.py`,
  `tests/fixtures/relatedness.py`
- Test: `tests/unit/test_relatedness_split.py`

**Interfaces:**
- Produces (`panels/relatedness_split.py`):
  - `RELATEDNESS_PANEL = 'relatedness_co_ownership'`
  - `class RelatednessPart(str, Enum)`: `VALIDATION = 'validation'`, `SEALED = 'sealed'`. It is
    its own enum. The panel shares no class with ITA (§4.5).
  - `class RelatednessPanelSettings(BaseModel)` (frozen, `extra='forbid'`,
    `allow_inf_nan=False`): `panel: Literal['relatedness_co_ownership']`,
    `rule_version: Literal[1]`, `salt: str` (pattern `^[0-9a-f]{32}$`),
    `relative_margin: float` (`gt=0`), `k: Literal[10]`
  - `load_relatedness_settings(path: Union[str, Path]) -> RelatednessPanelSettings`
  - `split_part(code: str, salt: str) -> RelatednessPart` (P1)
  - `split_rule(settings: RelatednessPanelSettings) -> Dict[str, Any]`: exactly
    `{'salt', 'rule_version', 'hash': 'sha256', 'prefix_bytes': 8, 'byteorder': 'big',
    'modulus': 2, 'parts': ['validation', 'sealed']}`
  - `split_fingerprint(settings: RelatednessPanelSettings) -> str`: SHA-256 of
    `json.dumps(split_rule(settings), sort_keys=True, separators=(',', ':'))`
- Produces (`tests/fixtures/relatedness.py`). Its docstring names embedding-data `e6e12b2` and the
  three source lines in the Spec line.
  - `CO_OWNERSHIP_TARGET_SCHEMA`: every `targets-v1` column, in order: `source`, `family`,
    `unit_a_type`, `unit_a`, `unit_b_type`, `unit_b`, `measure` (all `pl.Utf8`), `value`, `se`
    (`pl.Float64`), `se_method` (`pl.Utf8`), `n_unweighted` (`pl.UInt64`), `n_weighted`
    (`pl.Float64`), `sample_set`, `vintage`, `universe` (`pl.Utf8`), `flags`
    (`pl.List(pl.Utf8)`).
  - `CO_OWNERSHIP_MARGINAL_SCHEMA`: `node_id` (`pl.Utf8`), then `n_employers`,
    `n_multi_industry_employers`, `n_records`, `n_employers_including_padded` and
    `n_employers_partial` (all `pl.UInt64`).
  - `TARGETS_FILE = 'ita_co_ownership.parquet'`, `MARGINALS_FILE = 'co_ownership_marginals.parquet'`
  - `RELATEDNESS_CODES`: the 16 codes of the table below, sorted. These are "the bundle's
    six-digit codes".
  - `BASE_MULTI: Dict[str, int]`: m for the 14 codes with a marginal row.
  - `BASE_PAIRS: Tuple[Tuple[str, str, int], ...]`: the 12 unordered pairs (a, b, J), a < b.
  - `RELATEDNESS_SETTINGS = RelatednessPanelSettings(panel='relatedness_co_ownership',
    rule_version=1, salt='4bfd5195916df66d0e1c619c6507f2ad', relative_margin=0.05, k=10)`
  - `SECOND_FAMILY = 'relatedness_second_family'` (P4)
  - `node(code: str) -> str`: `f'naics:2022:national_industry:{code}'`
  - `co_ownership_tables(pairs: Sequence[Tuple[str, str, int]] = BASE_PAIRS, multi:
    Mapping[str, int] = BASE_MULTI) -> Tuple[pl.DataFrame, pl.DataFrame]`. It emits each pair
    in both directions, with Stage 10's `METADATA` values (`source='osha_ita'`,
    `se_method='none'`, `sample_set='ita_300a_2023_2025'`, `vintage='naics=2022'`) and
    `universe='fixture'`, `value` = J / (m(a) + 3), with m(a) = 0 for a code outside `multi`, so
    the `no-marginal` case of Task 2 builds, `se` null, `n_weighted` = J and
    `flags=['size_gate_pending']`. Marginals have `n_employers` = m + 3, `n_records` =
    2 × `n_employers`, `n_employers_including_padded` = `n_employers` and
    `n_employers_partial` = 0. Targets are sorted by every column but `value`, `se`,
    `n_unweighted`, `n_weighted` and `flags`, and marginals by `node_id`, as Stage 10 sorts them.
  - `write_co_ownership_tables(directory: Path, targets: pl.DataFrame, marginals: pl.DataFrame)
    -> Tuple[Path, Path]`
  - `tree_distance(codes: Sequence[str]) -> np.ndarray`: D* = 2 × (6 − `lca_level(a, b)`) off the
    diagonal, 0 on it (`lca_level` from `panels/decoding.py`). This is for fixtures only. The
    panel's tree arm reads the bundle's D*.

The base fixture. Each part comes from P1's salt. Each J/m(a) differs from J/m(b), so a swapped
gain fails a test.

| code | part | m | partners (J) |
|---|---|---|---|
| 111110 | validation | 4 | 311111 (2), 445110 (1), 111120 (2) |
| 111120 | validation | 3 | 311211 (1), 111110 (2) |
| 111150 | sealed | 2 | 111191 (1), 311221 (1) |
| 111191 | validation | 1 | 111150 (1) |
| 311111 | sealed | 5 | 311119 (4), 111110 (2) |
| 311119 | sealed | 4 | 311111 (4) |
| 311211 | validation | 5 | 311212 (3), 111120 (1), 311221 (2) |
| 311212 | sealed | 3 | 311211 (3) |
| 311221 | validation | 3 | 311211 (2), 111150 (1) |
| 312111 | validation | 6 | 312112 (5), 445110 (1) |
| 312112 | sealed | 5 | 312111 (5) |
| 445110 | validation | 2 | 312111 (1), 111110 (1), 621210 (1) |
| 445131 | validation | 0 | none: a candidate, never an anchor |
| 621210 | sealed | 1 | 445110 (1) |
| 211120 | sealed | no marginal row | none: in the codebook, never a candidate |
| 236220 | validation | no marginal row | none: Task 6's rebuild adds it |

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_relatedness_split.py`:

```python
GOLDEN = {
    '111110': 'validation', '111120': 'validation', '111150': 'sealed', '111191': 'validation',
    '211120': 'sealed', '236220': 'validation', '311111': 'sealed', '311119': 'sealed',
    '311211': 'validation', '311212': 'sealed', '311221': 'validation', '312111': 'validation',
    '312112': 'sealed', '445110': 'validation', '445131': 'validation', '621210': 'sealed',
}

def test_split_part_matches_the_golden_parts():
    for code, part in GOLDEN.items():
        assert split_part(code, RELATEDNESS_SETTINGS.salt) is RelatednessPart(part)

def test_the_rule_is_the_ita_rule():
    # for every GOLDEN code: split_part(code, salt).value == ita_split.split_part(code, salt).value
    # and set(split_rule(RELATEDNESS_SETTINGS)) == set(ita_split.split_rule(ITA_SETTINGS))

def test_split_fingerprint_is_pinned():
    assert split_fingerprint(RELATEDNESS_SETTINGS) == (
        'abb6c1e6b7ed16e73e081bfe682dadb3364041b95ee417ecb4de6712e38906b1'
    )

def test_a_changed_salt_changes_the_fingerprint():
    other = RELATEDNESS_SETTINGS.model_copy(update={'salt': '4bfd5195916df66d0e1c619c6507f2ae'})
    assert split_fingerprint(other) == (
        'bc1fb6c5e38bca1479cf5e0464163be95932713a0f2356b2ee95e852ffff3cc1'
    )

def test_committed_settings_hold_the_planned_values():
    # ROOT = Path(__file__).resolve().parents[2]
    assert load_relatedness_settings(ROOT / 'conf/data/relatedness_panel.yaml') == RELATEDNESS_SETTINGS

def test_settings_refuse_bad_values():
    # each raises ValidationError: a 31-character salt, an uppercase salt, relative_margin 0 and
    # inf, k 20, panel 'ita_text', an unknown key

def test_fixture_tables_carry_every_stage_10_column_in_order():
    targets, marginals = co_ownership_tables()
    assert list(targets.schema.items()) == list(CO_OWNERSHIP_TARGET_SCHEMA.items())
    assert list(marginals.schema.items()) == list(CO_OWNERSHIP_MARGINAL_SCHEMA.items())
    assert targets.height == 24 and marginals.height == 14

def test_fixture_gains_are_asymmetric():
    for a, b, j in BASE_PAIRS:
        assert j / BASE_MULTI[a] != j / BASE_MULTI[b]
```

- [ ] **Step 2: Run them to see them fail.** Run:
  `uv run pytest tests/unit/test_relatedness_split.py -q`. Expected: FAIL with
  `ModuleNotFoundError: No module named 'naics_embedder.panels.relatedness_split'`.

- [ ] **Step 3: Write `conf/data/relatedness_panel.yaml`.** It holds the five keys. A comment says
  that the salt was drawn once at plan 21 and never changes, because the sealed part's openings
  are keyed on the split fingerprint (§4.3, §4.5). Another comment says that r is δ as a share of
  the chosen reference's validation headline (§4.6), and that k is the headline's NDCG cut (U44).

```yaml
panel: relatedness_co_ownership
rule_version: 1
salt: '4bfd5195916df66d0e1c619c6507f2ad'
relative_margin: 0.05
k: 10
```

- [ ] **Step 4: Implement `panels/relatedness_split.py` and `tests/fixtures/relatedness.py`**
  with the interfaces above. Import the rule's constants (`HASH`, `PREFIX_BYTES`, `BYTE_ORDER`,
  `MODULUS`) from `panels/ita_split.py`. Do not import `ItaPart` or `ItaPanelSettings`.

- [ ] **Step 5: Run the tests.** Run: `uv run pytest tests/unit/test_relatedness_split.py -q`.
  Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add conf/data/relatedness_panel.yaml src/naics_embedder/panels/relatedness_split.py tests/fixtures/relatedness.py tests/unit/test_relatedness_split.py
git commit -m "feat(panels): add the relatedness panel's salted industry split and fixture tables"
```

---

### Task 2: Co-ownership tables into gains

**Files:**
- Create: `src/naics_embedder/panels/relatedness.py` ("Inputs" section)
- Test: `tests/unit/test_relatedness_inputs.py`

**Interfaces:**
- Consumes: `node_code` and `CODE_PATTERN` from `panels/ita.py` (imported, not changed).
- Produces:
  - `TARGET_COLUMNS: Dict[str, pl.DataType]`: `family`, `measure`, `unit_a_type`, `unit_a`,
    `unit_b_type`, `unit_b` (`pl.Utf8`) and `n_unweighted` (`pl.UInt64`)
  - `MARGINAL_COLUMNS: Dict[str, pl.DataType]`: `node_id` (`pl.Utf8`) and
    `n_multi_industry_employers` (`pl.UInt64`)
  - `CO_OWNERSHIP_FAMILY = 'co_ownership'`, `CO_OWNERSHIP_MEASURE = 'employer_share'`,
    `INDUSTRY_UNIT = 'naics_national_industry'`
  - `TABLES = ('ita_co_ownership', 'co_ownership_marginals')`
  - `@dataclass(frozen=True) class CoOwnership`:
    - `candidates: Tuple[str, ...]`: the codes with a marginal row, sorted.
    - `multi: Mapping[str, int]`: m.
    - `shared: Mapping[Tuple[str, str], int]`: J for each ordered pair with a row.
    - `partners(self, code: str) -> Tuple[str, ...]`: the codes b with a row (code, b), sorted.
    - `gain(self, anchor: str, candidate: str) -> float`: J(anchor, candidate) / m(candidate),
      or 0.0 with no row (§4.4).
  - `co_ownership_inputs(targets: pl.DataFrame, marginals: pl.DataFrame, codebook_codes:
    Sequence[str]) -> CoOwnership`

`co_ownership_inputs` reads only the columns above and raises `ValueError` in this order. Each
message names the table and the codes of the first offending row, and holds the `match` fragment
of the table in Step 1:

1. a missing column, or one of another dtype;
2. a null in a column it reads (P5);
3. a family, measure or unit type other than the accepted ones;
4. a node id that `node_code` refuses;
5. a code outside `codebook_codes`;
6. a repeated marginal node (P5);
7. a target row whose industry has no marginal row;
8. an industry paired with itself (P5);
9. a repeated ordered pair (P5);
10. J < 1 (P5);
11. a pair in one direction only;
12. a pair whose two directions hold different J;
13. a J above m of either industry.

- [ ] **Step 1: Write the failing tests:**

```python
def test_the_base_tables_give_the_fixture_gains():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    assert co.candidates == tuple(sorted(BASE_MULTI))
    assert co.gain('311211', '311212') == 1.0                 # J 3 / m(311212) 3
    assert co.gain('311212', '311211') == pytest.approx(0.6)  # J 3 / m(311211) 5
    assert co.gain('311211', '111120') == pytest.approx(1 / 3)
    assert co.gain('311211', '445131') == 0.0
    assert co.partners('445131') == ()
    assert set(co.shared) == {(a, b) for a, b, _ in BASE_PAIRS} | {(b, a) for a, b, _ in BASE_PAIRS}

def test_codebook_codes_without_a_marginal_row_are_not_candidates():
    co = co_ownership_inputs(*co_ownership_tables(), RELATEDNESS_CODES)
    assert '211120' not in co.candidates and '445131' in co.candidates

def test_row_order_and_extra_columns_do_not_matter():
    # both tables shuffled (seeded) with an added column 'extra' give an equal CoOwnership

@pytest.mark.parametrize('case', CASES, ids=[case.id for case in CASES])
def test_an_input_that_disagrees_with_the_spec_is_refused(case): ...
```

`CASES` builds each one from the base tables:

| id | how | `match` |
|---|---|---|
| `missing-column` | drop `n_unweighted` | `lacks column 'n_unweighted'` |
| `wrong-dtype` | cast `n_unweighted` to `Int64` | `has type` |
| `null` | set `unit_b` null on one row | `null` |
| `family` | `family='co_location'` on one row | `family` |
| `measure` | `measure='employer_count'` on one row | `measure` |
| `unit-type` | `unit_b_type='naics_industry'` on one row | `unit type` |
| `node-level` | marginal `node_id` `naics:2022:industry:31121` | `node id` |
| `node-vintage` | `unit_a` `naics:2017:national_industry:311211` on one row | `node id` |
| `not-in-codebook` | the pair (311211, 999999, 1), with m(999999) = 1 | `not a codebook` |
| `repeated-marginal` | append the marginals' first row again | `repeats` |
| `no-marginal` | the pair (211120, 311211, 1) | `no marginal row` |
| `self-pair` | the row (311211, 311211, 1) | `itself` |
| `repeated-pair` | append the targets' first row again | `repeats` |
| `zero-j` | J = 0 in both directions of (311211, 311212) | `at least 1` |
| `one-way` | drop the row (311212, 311211) | `one direction` |
| `unequal-mirror` | J = 2 on the row (311212, 311211) | `mirror` |
| `j-above-m` | J = 4 in both directions of (311211, 311212) | `exceeds` |

- [ ] **Step 2: Run them to see them fail.** Run:
  `uv run pytest tests/unit/test_relatedness_inputs.py -q`. Expected: FAIL with
  `ModuleNotFoundError` on `naics_embedder.panels.relatedness`.

- [ ] **Step 3: Implement the "Inputs" section** of `panels/relatedness.py`, with a module
  docstring in the form of `panels/ita.py`'s (Inputs, then Panel).

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/relatedness.py tests/unit/test_relatedness_inputs.py
git commit -m "feat(panels): read Stage 10's co-ownership tables into relatedness gains"
```

---

### Task 3: The tie-aware scorer

**Files:**
- Create: `src/naics_embedder/panels/relatedness_scoring.py` ("Scores" section)
- Test: `tests/unit/test_relatedness_scoring.py`

**Interfaces:**
- Produces:
  - `expected_dcg(similarity: np.ndarray, gains: np.ndarray, k: int) -> float`. It is the
    expected DCG@k over every order within tied groups (§4.4). Gains are linear, never
    2^g − 1. A group of exactly equal float64 similarities (P7) that covers the 1-based ranks s to
    e adds `mean(gains of the group) × Σ_{p=s}^{min(e,k)} 1 / log2(p + 1)`. Groups are taken in
    descending similarity.
  - `ideal_dcg(gains: np.ndarray, k: int) -> float`: the DCG@k of the gains sorted in descending
    order.
  - `tie_aware_ndcg(similarity: np.ndarray, gains: np.ndarray, k: int) -> float`: `expected_dcg`
    / `ideal_dcg`. It raises `ValueError` when no gain is positive.
  - `tie_aware_auc(similarity: np.ndarray, relevant: np.ndarray) -> Optional[float]`: the share of
    (relevant, non-relevant) candidate pairs that the similarity orders correctly, with each tie
    counted one half. It is None when either set is empty (P8).

Each function takes one anchor's 1-D arrays over its candidates. It raises `ValueError` when the
arrays differ in length, or when k < 1.

- [ ] **Step 1: Write the failing tests:**

```python
def brute_force_dcg(similarity, gains, k):
    # mean over itertools.product of every permutation of each tied group's members, groups in
    # descending similarity, of the DCG@k of that full order

def test_the_tree_row_of_311211_matches_brute_force():
    # candidates: every BASE_MULTI code but 311211 (13). similarity = -tree_distance row, gains =
    # CoOwnership gains: groups 2 | 4 | 6 (311111, 311119) | 8 (312111, 312112) | 10 (7 codes,
    # with 111120 at gain 1/3), so 2 * 2 * 5040 orders
    for k in (3, 10, 20):   # k = 3 cuts the 6-group; k = 10 cuts the 10-group
        assert expected_dcg(sim, gains, k) == pytest.approx(brute_force_dcg(sim, gains, k), rel=1e-12)

def test_random_tie_structures_match_brute_force():
    # rng = np.random.default_rng(21); 25 trials of 7 candidates, similarities drawn from
    # {0.0, 1.0, 2.0}, gains from {0, 0.25, 0.5, 1}, k drawn from 1..7: equal to brute force

def test_without_ties_it_is_plain_dcg():
    sim = np.array([0.9, 0.5, 0.7]); gains = np.array([1.0, 0.0, 0.5])
    assert expected_dcg(sim, gains, 10) == pytest.approx(1.0 + 0.5 / np.log2(3))

def test_ndcg_does_not_depend_on_candidate_order():
    # the 311211 tree row permuted (seeded): equal tie_aware_ndcg at k 10 and 20

def test_a_perfect_ranking_scores_one_and_no_gain_is_refused(): ...

def test_auc_counts_ties_one_half():
    assert tie_aware_auc(np.array([1.0, 1.0, 0.0]), np.array([True, False, False])) == 0.75

def test_auc_is_none_without_a_relevant_or_an_unrelated_candidate(): ...
```

- [ ] **Step 2: Run them to see them fail.** Run:
  `uv run pytest tests/unit/test_relatedness_scoring.py -q`. Expected: FAIL with
  `ModuleNotFoundError`.

- [ ] **Step 3: Implement the "Scores" section.**

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/relatedness_scoring.py tests/unit/test_relatedness_scoring.py
git commit -m "feat(panels): add relatedness's tie-aware expected NDCG and AUC"
```

---

### Task 4: Per-anchor scores and the result

**Files:**
- Modify: `src/naics_embedder/panels/relatedness_scoring.py` ("Summary" section)
- Test: `tests/unit/test_relatedness_scoring.py`

**Interfaces:**
- Consumes: Task 3; `code_lineage` from `utils/naics_hierarchy.py`.
- Produces:
  - `HEADLINE_K = 10`, `REPORT_K = 20` (P2)
  - `PER_ANCHOR_SCHEMA = {'anchor': pl.Utf8, 'part': pl.Utf8, 'n_relevant': pl.Int64,
    'ndcg_10': pl.Float64, 'ndcg_20': pl.Float64, 'auc': pl.Float64}`
  - `COUNT_KEYS = ('n_industries', 'n_anchors', 'n_no_partner', 'n_candidates', 'n_relevant',
    'n_crossing')`
  - `score_anchors(similarity: np.ndarray, gains: np.ndarray, anchors: Sequence[str],
    candidates: Sequence[str], part: str) -> pl.DataFrame`: one row per anchor, in the given
    order, in `PER_ANCHOR_SCHEMA`. `similarity` and `gains` are (anchors, candidates). An
    anchor's own column, when it is a candidate, is left out (§4.4). `n_relevant` counts the
    positive gains. It raises `ValueError` for a shape mismatch, for a non-finite similarity
    (naming the anchor), and for an anchor with no positive gain.
  - `@dataclass(frozen=True) class RelatednessResult`:
    - `panel: str`, `part: str`, `per_anchor: pl.DataFrame`.
    - `headline`: `{'n_anchors': int, 'ndcg_10': float}`.
    - `reports`: `{'ndcg_20': float, 'auc': Optional[float], 'n_auc_anchors': int}`.
    - `by_sector`: `sector`, `n_anchors` and `ndcg_10`, sorted by `sector`.
    - `counts: Dict[str, int]`, keyed by `COUNT_KEYS`.
    - `data_fingerprint: Optional[str] = None`.
  - `summarize_relatedness(per_anchor: pl.DataFrame, *, panel: str, part: str, counts:
    Mapping[str, int]) -> RelatednessResult`. Every mean uses `math.fsum`. An anchor's sector is
    `code_lineage(anchor)[0]`, so 31-33, 44-45 and 48-49 are one sector each. It raises
    `ValueError` for an empty `per_anchor`, for counts not keyed by `COUNT_KEYS`, and when
    `counts['n_anchors']` differs from the frame's height.

- [ ] **Step 1: Write the failing tests:**

```python
def test_an_anchor_is_left_out_of_its_own_candidates():
    # anchor 'a' among candidates ('a', 'b', 'c'), its own similarity the highest and its own
    # gain 1: ndcg_10 equals tie_aware_ndcg over ('b', 'c') alone

def test_the_headline_is_the_mean_over_anchors_each_once():
    # a result's headline['ndcg_10'] == math.fsum(per_anchor ndcg_10) / height; replacing the
    # per_anchor auc and ndcg_20 columns with other values leaves headline equal

def test_an_anchor_with_no_unrelated_candidate_has_no_auc():
    # an anchor whose every candidate has a positive gain: auc null in per_anchor;
    # reports['auc'] is the mean over the other anchors and n_auc_anchors excludes it;
    # headline['ndcg_10'] counts it

def test_sectors_join_31_to_33():
    # anchors '311211', '321113' and '445110' → by_sector rows ('31', 2, …) and ('44', 1, …)

def test_a_non_finite_similarity_is_refused_naming_the_anchor(): ...

def test_an_anchor_with_no_partner_and_an_empty_result_are_refused(): ...
```

- [ ] **Step 2: Run them to see them fail.** Expected: FAIL with `ImportError` on
  `score_anchors`.

- [ ] **Step 3: Implement the "Summary" section.**

- [ ] **Step 4: Run the tests.** Run: `uv run pytest tests/unit/test_relatedness_scoring.py -q`.
  Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/relatedness_scoring.py tests/unit/test_relatedness_scoring.py
git commit -m "feat(panels): add relatedness's per-anchor scores, headline and reports"
```

---

### Task 5: The arms

**Files:**
- Create: `src/naics_embedder/panels/relatedness_arms.py`
- Test: `tests/unit/test_relatedness_arms.py`

**Interfaces:**
- Consumes: `CodeTargets` (`supervision/code_targets.py`); `text_only_matrix`
  (`panels/text_only.py`).
- Produces:
  - `class RelatednessArm(Protocol)`: `similarity(self, anchors: Sequence[str], candidates:
    Sequence[str]) -> np.ndarray`, which is (len(anchors), len(candidates)) float64 (§4.5).
  - `class TreeArm`. Its constructor is `__init__(self, codes: Sequence[str],
    structural_distance: np.ndarray)`, with `from_targets(cls, targets: CodeTargets) ->
    TreeArm`. Its similarity is −D*, read by code, so the codebook's row order does not matter.
    It raises `ValueError` for repeated codes, a matrix that is not (N, N), and an unknown code
    in a request (naming it).
  - `class VectorArm`. Its constructor is `__init__(self, codes: Sequence[str], vectors:
    np.ndarray, *, center: bool = False)`, with `from_text_only(cls, table: pl.DataFrame, *,
    center: bool = False) -> VectorArm`. Its similarity is the float64 cosine. With `center`, it
    first subtracts the mean of the rows whose code has six digits (P7). It raises `ValueError`
    for repeated codes, a non-finite value, an unknown code in a request (naming the first five
    missing codes and how many there are), and a zero-norm row of a requested code (naming it).

- [ ] **Step 1: Write the failing tests:**

```python
def test_the_tree_arm_is_minus_d_star():
    arm = TreeArm(RELATEDNESS_CODES, tree_distance(RELATEDNESS_CODES))
    assert arm.similarity(['311211'], ['311212', '311221', '311111', '312111', '111110']).tolist() == [[-2, -4, -6, -8, -10]]

def test_the_tree_arm_reads_the_bundles_d_star_by_code(validated_bundle):
    targets = CodeTargets.from_bundle(validated_bundle)
    arm = TreeArm.from_targets(targets)
    # equals -targets.structural_distance at the requested codes' indices; a TreeArm over the same
    # codes and matrix permuted by a seeded permutation gives an identical similarity

def test_plain_cosine():
    arm = VectorArm(['a', 'b', 'c'], np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]]))
    assert arm.similarity(['a'], ['b', 'c']) == pytest.approx(np.array([[2 ** -0.5, 0.0]]))

def test_the_centered_mean_is_over_the_tables_six_digit_codes():
    # a table of the 16 RELATEDNESS_CODES plus '31' and '3112' rows; changing the '31' row leaves
    # the centered similarity alone, and requesting more candidates leaves each value unchanged;
    # the centered values equal the cosine after subtracting the 16 rows' mean

def test_from_text_only_reads_the_t_columns():
    # text_only_frame(codes, vectors) → VectorArm.from_text_only equals VectorArm(codes, vectors)

def test_an_unknown_code_or_a_zero_row_is_refused_naming_it(): ...
```

- [ ] **Step 2: Run them to see them fail.** Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `panels/relatedness_arms.py`.**

- [ ] **Step 4: Run the tests.** Run: `uv run pytest tests/unit/test_relatedness_arms.py -q`.
  Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/relatedness_arms.py tests/unit/test_relatedness_arms.py
git commit -m "feat(panels): add the relatedness tree and vector arms"
```

---

### Task 6: `RelatednessPanel`: reads, pins and the sealed opening

**Files:**
- Modify: `src/naics_embedder/panels/relatedness.py` ("Panel" section)
- Test: `tests/unit/test_relatedness_panel.py`

**Interfaces:**
- Consumes:
  - Tasks 1–5.
  - `SelectionLog`, `SelectionEvent` and `merge_read_detail` (`panels/selection_log.py`).
  - `SealedSplitError` and `SplitAlreadyOpenedError` (`panels/outcome.py`).
  - `ValidatedSupervisionBundle` and `sha256_file` (`supervision/artifacts.py`).
- Produces:
  - `READ_ROLES = ('reference', 'candidate')`
  - `class RelatednessPanel`:
    - `name = RELATEDNESS_PANEL` (a class attribute; P4).
    - `__init__(self, targets: pl.DataFrame, marginals: pl.DataFrame, codebook_codes:
      Sequence[str], settings: RelatednessPanelSettings, log: SelectionLog, *, table_sha256:
      Mapping[str, str])`. It sets `co_ownership` (`co_ownership_inputs`), `candidates`,
      `settings`, `log`, `fingerprint = split_fingerprint(settings)` and `table_sha256`. The
      `table_sha256` keys are exactly `TABLES`, each 64 lowercase hex characters. They enter the
      data fingerprint only.
    - `from_files(cls, targets_parquet, marginals_parquet, bundle: ValidatedSupervisionBundle,
      settings_path, log_path, *, expected_sha256: Mapping[str, str]) -> RelatednessPanel`. It
      takes `sha256_file` of each parquet and refuses a mismatch with its pin before it reads a
      table (§4.2 "Pins"). The `codebook_codes` are the bundle codebook's six-digit codes, read as
      in `ItaPanel.from_files`.
    - `part_of(self, code: str) -> RelatednessPart`
    - `anchors(self, part: Union[RelatednessPart, str]) -> Tuple[str, ...]`: the part's
      candidates with a partner, sorted.
    - `data_fingerprint(self, part: Union[RelatednessPart, str]) -> str` (P6). It is not a read.
    - `open_sealed(self, purpose: str, *, reopen_reason: Optional[str] = None) -> None`
    - `score(self, arm: RelatednessArm, part, purpose: str, detail: Optional[Mapping[str, Any]] =
      None) -> RelatednessResult`
    - `score_logged(self, arm, part, purpose, detail=None) -> Tuple[RelatednessResult,
      Dict[str, Any]]`

`score_logged` runs in this order (§4.5):

1. It refuses, before it logs, a `detail['role']` outside `READ_ROLES`, a sealed read before
   `open_sealed`, and a part with no anchor (P8).
2. It appends a `read` record. The record holds `panel=RELATEDNESS_PANEL`, `split` (the part),
   `fingerprint` (the split fingerprint) and `n_queries` (the number of anchors). Its detail is
   `{'arm_type': type(arm).__name__, 'data_fingerprint': ...}`, merged with the caller's `arm`,
   `seed`, `role` and `run`.
3. Only then does it call `arm.similarity(anchors, candidates)`. It refuses a non-finite
   similarity, naming `arm_type` and the anchor.
4. It scores with `score_anchors` and the panel's gains. It summarizes with the part's counts:
   - `n_industries`: the part's candidates.
   - `n_anchors`.
   - `n_no_partner`: `n_industries` − `n_anchors`.
   - `n_candidates`: the length of `candidates`.
   - `n_relevant`: the sum of the anchors' partners.
   - `n_crossing`: the unordered relevant pairs whose industries fall in different parts.
5. It returns the result with its `data_fingerprint` set.

`open_sealed` follows `ItaPanel.open_sealed`, with this panel's name, split `sealed` and
`n_queries` = the sealed anchors.

- [ ] **Step 1: Write the failing tests.** `panel()` builds the base tables with
  `table_sha256={'ita_co_ownership': 'a' * 64, 'co_ownership_marginals': 'b' * 64}`.

```python
def test_each_industry_falls_in_one_part_and_a_rebuild_keeps_it_there():
    base = panel()
    assert base.anchors('validation') == ('111110', '111120', '111191', '311211', '311221', '312111', '445110')
    assert base.anchors('sealed') == ('111150', '311111', '311119', '311212', '312112', '621210')
    # rebuilt with BASE_MULTI | {'236220': 1} and the pair ('236220', '312111', 1): every base
    # code keeps its part, 236220 is in validation, the fingerprint is equal, and the validation
    # data fingerprint differs

def test_the_counts_of_each_part():
    # validation: n_industries 8, n_anchors 7, n_no_partner 1, n_candidates 14, n_relevant 16,
    # n_crossing 6; sealed (after open_sealed): 6, 6, 0, 14, 8, 6

def test_the_tree_arms_scores_are_the_scorers():
    # TreeArm(RELATEDNESS_CODES, tree_distance(RELATEDNESS_CODES)) on validation: 311211's ndcg_10
    # equals tie_aware_ndcg of its tree row and gains at k 10, and headline['ndcg_10'] is the fsum
    # mean of the seven anchors

def test_a_read_is_logged_before_the_arm_is_asked():
    # an arm whose similarity raises RuntimeError: the log holds one read with panel
    # 'relatedness_co_ownership', split 'validation', fingerprint split_fingerprint(...),
    # n_queries 7, and detail arm_type, data_fingerprint and the caller's arm, role and run

def test_a_non_finite_similarity_is_refused_naming_the_arm_and_anchor():
    # a stub arm with NaN on 311211's row: ValueError naming the stub's type and '311211'; the
    # read record exists

def test_a_role_outside_reference_and_candidate_is_refused_before_logging(): ...

def test_the_sealed_part_is_refused_before_it_is_opened():
    # SealedSplitError, and the log file does not exist

def test_a_second_opening_needs_a_reason_also_after_the_tables_are_rebuilt():
    # panel A opens; panel B over the rebuilt tables (as above) has the same fingerprint;
    # B.open_sealed('again') → SplitAlreadyOpenedError; with reopen_reason='audit' → event 'reopen',
    # detail reason 'audit'

def test_a_part_with_no_anchor_is_refused_before_logging():
    # tables whose pairs are all inside the validation part: open_sealed, then a sealed read raises
    # ValueError 'no anchor'; the log holds the opening only

def test_constructor_refusals_log_nothing():
    # the j-above-m tables of Task 2: RelatednessPanel(...) raises and the log file does not exist

def test_row_order_and_extra_columns_leave_the_data_fingerprint_alone():
    # both tables shuffled with an 'extra' column, same table_sha256: equal data fingerprints for
    # both parts and an equal validation headline

def test_the_data_fingerprint_names_the_table_hashes():
    # another table_sha256 value: another data fingerprint, the same split fingerprint

def test_from_files_checks_the_pins_before_reading(tmp_path, validated_bundle):
    # tables over the bundle's six-digit codes (111111, 111112, 111113, 222222, 333333):
    # multi {'111111': 2, '111112': 2, '222222': 1, '333333': 0}, pairs
    # [('111111', '111112', 2), ('111111', '222222', 1)], written with write_co_ownership_tables.
    # The right pins (sha256_file of each file) set panel.table_sha256 to them. A wrong pin raises
    # ValueError naming the table, the pin and the hash found, and logs nothing

def test_relatedness_reads_stay_out_of_another_runs_log_records():
    # a relatedness validation read and a sealed opening in a log with an outcome read under run
    # 'stage7/x': decision.sweep._log_records(..., 'stage7/x') returns that run's records only
```

- [ ] **Step 2: Run them to see them fail.** Run:
  `uv run pytest tests/unit/test_relatedness_panel.py -q`. Expected: FAIL with `ImportError` on
  `RelatednessPanel`.

- [ ] **Step 3: Implement the "Panel" section** of `panels/relatedness.py`.

- [ ] **Step 4: Run the tests.** Run: `uv run pytest tests/unit/test_relatedness_panel.py
  tests/unit/test_relatedness_inputs.py -q`. Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/relatedness.py tests/unit/test_relatedness_panel.py
git commit -m "feat(panels): add RelatednessPanel with pinned tables, logged reads and one sealed opening"
```

---

### Task 7: The reference choice and the margin record

**Files:**
- Create: `src/naics_embedder/decision/relatedness.py` ("Reference and margin" section)
- Test: `tests/unit/test_decision_relatedness.py`

**Interfaces:**
- Consumes:
  - `RelatednessPanel` and `RelatednessPart` (Tasks 1 and 6), and `RelatednessArm` (Task 5).
  - `SelectionLog` and `SelectionEvent`.
  - `read_time` (`decision/decide.py`) and `_log_records` (`decision/sweep.py`), imported as
    `decision/text_to_code.py` imports them.
- Produces:
  - `TREE = 'tree'`, `TEXT_ONLY = 'text-only'`, `TEXT_ONLY_CENTERED = 'text-only-centered'`
  - `REFERENCE_ARMS = (TREE, TEXT_ONLY, TEXT_ONLY_CENTERED)`, also the tie order (§4.6)
  - `choose_reference(headlines: Mapping[str, float]) -> str`: the highest headline. A tie goes to
    the arm earlier in `REFERENCE_ARMS`. It raises `ValueError` unless the keys are exactly
    `REFERENCE_ARMS` and every value is finite.
  - `class RelatednessMarginRecord(BaseModel)` (frozen, `extra='forbid'`, `allow_inf_nan=False`):
    - `kind: Literal['relatedness_margin'] = 'relatedness_margin'`
    - `panel: str`, `name: str`, `r: float` (`gt=0`), `run_id: str`
    - `reference_headlines: Dict[str, float]`, `reference: str`, `margin: float`
    - `split_fingerprint: str`, `data_fingerprint: str`
    - `embedding_data_commit: str` (pattern `^[0-9a-f]{7,40}$`), `table_sha256: Dict[str, str]`
    - `log_records: List[Dict[str, Any]]`, `fixed_at: datetime`

    These are §4.6's "margin record" fields, with `panel` per P4.
  - `fix_relatedness_margin(panel: RelatednessPanel, reference_arms: Mapping[str,
    RelatednessArm], r: float, name: str, *, embedding_data_commit: str, purpose: str) ->
    RelatednessMarginRecord`
  - `check_relatedness_margin(margin: RelatednessMarginRecord) -> None`
  - `check_relatedness_margin_first(margin: RelatednessMarginRecord, log: SelectionLog) -> None`

`fix_relatedness_margin` behaves as follows:

- Before any read, it refuses an `r` that is not positive and finite, and arms whose keys are not
  exactly `REFERENCE_ARMS`.
- It reads each arm once on validation, under `run_id = f'relatedness-margin/{uuid4().hex}'`,
  with detail `{'arm', 'role': 'reference', 'run'}`.
- Each arm's headline is `result.headline['ndcg_10']`.
- `margin` is `r × reference_headlines[reference]`.
- `panel` is `panel.name`, and `table_sha256` is `dict(panel.table_sha256)`.

`check_relatedness_margin` follows `check_ita_margin`, with this module's names. It requires:

- the headlines' keys to be exactly `REFERENCE_ARMS`;
- the reference to be the one `choose_reference` picks;
- δ to be exactly `r × headline`;
- at least one read;
- each read to be a `read` of `margin.panel`'s validation part, under its run, with role
  `reference`, an arm in `REFERENCE_ARMS`, and its split and data fingerprints;
- each reference arm to be read exactly once (P9).

`check_relatedness_margin_first` runs that check first. It then refuses:

- a `fixed_at` with no timezone;
- a read of the record's own that is later than `fixed_at`;
- any `read` of `margin.panel`'s validation part before `fixed_at` that is not a reference read
  of a reference arm. The message names the read's purpose and time.

- [ ] **Step 1: Write the failing tests.** `arms()` is `{'tree': TreeArm(...), 'text-only':
  VectorArm(codes, vectors), 'text-only-centered': VectorArm(codes, vectors, center=True)}` with
  `vectors = np.random.default_rng(7).normal(size=(16, 8))`. `_hand_record(log, panel_name,
  headlines, data_fingerprint, *, fixed_at=None)` builds a second family's record (P4). It
  appends one validation reference read for each arm under `panel_name` with
  `SelectionLog.append`, all under one run and with `RELATEDNESS_SETTINGS`' split fingerprint and
  `data_fingerprint`. It then returns a record that passes `check_relatedness_margin`, with
  `fixed_at` defaulting to now. Task 8 reuses it.

```python
def test_the_reference_is_the_highest_headline_and_a_tie_goes_to_the_simpler_arm():
    assert choose_reference({'tree': 0.30, 'text-only': 0.42, 'text-only-centered': 0.40}) == 'text-only'
    assert choose_reference({'tree': 0.42, 'text-only': 0.42, 'text-only-centered': 0.42}) == 'tree'
    assert choose_reference({'tree': 0.30, 'text-only': 0.42, 'text-only-centered': 0.42}) == 'text-only'
    assert choose_reference({'tree': 0.30, 'text-only': 0.40, 'text-only-centered': 0.42}) == 'text-only-centered'

def test_choose_reference_refuses_other_arms_or_a_nan(): ...

def test_the_margin_is_r_times_the_chosen_references_headline():
    record = fix_relatedness_margin(panel, arms(), 0.05, 'relatedness-margins',
                                    embedding_data_commit='e6e12b2', purpose='fix the margin')
    assert record.reference == choose_reference(record.reference_headlines)
    assert record.margin == record.r * record.reference_headlines[record.reference]
    assert record.panel == 'relatedness_co_ownership' and record.table_sha256 == panel.table_sha256
    assert sorted(r['detail']['arm'] for r in record.log_records) == sorted(REFERENCE_ARMS)
    check_relatedness_margin_first(record, panel.log)

def test_fix_refuses_a_bad_r_or_other_arms_before_reading(): ...  # the log file does not exist

def test_a_margin_after_a_candidate_read_is_refused(): ...         # names the read's purpose

def test_a_margin_after_a_read_with_no_role_is_refused(): ...

def test_ita_reads_do_not_block_a_relatedness_margin():
    # an ita_text validation read with role candidate, appended first: the check passes

def test_a_second_familys_margin_is_judged_on_its_own_reads():
    # candidate reads on relatedness_co_ownership, then _hand_record(log, SECOND_FAMILY, ...):
    # the check passes. A candidate read on SECOND_FAMILY before a second hand record's fixed_at:
    # that record is refused

def test_a_record_that_disagrees_with_itself_is_refused():
    # model_copy with another margin, another reference, a read of another data fingerprint, and
    # one arm read twice: each raises ValueError

def test_a_naive_fixed_at_or_a_read_after_fixed_at_is_refused(): ...

def test_the_record_round_trips_through_json():
    again = RelatednessMarginRecord.model_validate_json(record.model_dump_json())
    assert again == record
    check_relatedness_margin_first(again, panel.log)
```

- [ ] **Step 2: Run them to see them fail.** Run:
  `uv run pytest tests/unit/test_decision_relatedness.py -q`. Expected: FAIL with
  `ModuleNotFoundError`.

- [ ] **Step 3: Implement the module's "Reference and margin" section**, with a module docstring
  in the form of `decision/text_to_code.py`'s.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/naics_embedder/decision/relatedness.py tests/unit/test_decision_relatedness.py
git commit -m "feat(decision): fix relatedness's margin from reference reads, checked margin-first"
```

---

### Task 8: The beat-by-δ test

**Files:**
- Modify: `src/naics_embedder/decision/relatedness.py` ("Test" section)
- Test: `tests/unit/test_decision_relatedness.py`

**Interfaces:**
- Consumes:
  - Task 7.
  - `RelatednessResult` (Task 4).
  - `PanelItems`, `unit_draws`, `seed_draws`, `replicate_statistics`, `point_statistic` and
    `percentile_interval` (`decision/resampling.py`).
  - `superiority_level` (`decision/rule.py`).
- Produces:
  - `headline_values(result: RelatednessResult) -> pl.DataFrame`: `unit` = `item` = the anchor,
    and `value` = `ndcg_10`. Anchors are the units, with one item each (§4.6).
  - `arm_replicates(items: PanelItems, panel: str, arm: str, seeds: Sequence[RelatednessResult],
    *, replicates: int, bootstrap_seed: int) -> Tuple[float, np.ndarray]`. Unit draws are on the
    stream `panel`, and seed draws on `(panel, arm)`.
  - `@dataclass(frozen=True) class FamilyDelta`: `panel: str`, `reference: str`, `delta: float`,
    `replicates: np.ndarray`, `margin: float`
  - `@dataclass(frozen=True) class FamilyVerdict`: `panel`, `reference`, `delta`,
    `interval: Tuple[float, float]`, `margin`, `beats: bool`
  - `@dataclass(frozen=True) class RelatednessVerdict`: `candidate: str`, `level: float`,
    `families: Tuple[FamilyVerdict, ...]`, `beats: bool`
  - `beat_by_margin(candidate: str, deltas: Sequence[FamilyDelta]) -> RelatednessVerdict`. Here
    n = `len(deltas)`, `level = superiority_level(n)`, and each family's interval is
    `percentile_interval(replicates, level)`. A family beats when the interval's lower bound
    exceeds its margin. The candidate beats when every family beats (§4.6). It raises
    `ValueError` for no family or a repeated panel.
  - `relatedness_test(candidate: str, candidate_seeds: Mapping[str,
    Sequence[RelatednessResult]], references: Mapping[str, RelatednessResult], margins:
    Mapping[str, RelatednessMarginRecord], *, min_seeds: int, replicates: int, bootstrap_seed:
    int) -> RelatednessVerdict`

Before any draw, `relatedness_test` refuses, with `ValueError`:

- mappings that are empty or keyed differently;
- a margin whose `panel` is not its key, or that fails `check_relatedness_margin`;
- fewer than `min_seeds` candidate seeds on a family, naming the family, the count and the floor;
- a result whose `panel` is not its key;
- results of more than one part;
- a candidate seed whose `data_fingerprint` differs from its family's reference.

For each family, it then takes the items from the reference, Δ = the candidate's point minus the
reference's, and the paired replicate differences. Its reference is `margins[key].reference`, and
its margin is `margins[key].margin` (P10).

- [ ] **Step 1: Write the failing tests:**

```python
def test_the_verdict_at_n_1_and_n_2():
    a = FamilyDelta('relatedness_co_ownership', 'tree', 0.02, np.linspace(0.0, 0.04, 1001), 0.0008)
    one = beat_by_margin('c', [a])
    assert one.level == 0.95 and one.families[0].interval[0] == pytest.approx(0.001) and one.beats
    b = FamilyDelta(SECOND_FAMILY, 'tree', 0.04, np.linspace(0.02, 0.06, 1001), 0.01)
    two = beat_by_margin('c', [a, b])
    assert two.level == pytest.approx(0.975)
    assert two.families[0].interval[0] == pytest.approx(0.0005) and not two.families[0].beats
    assert two.families[1].beats and not two.beats

def test_a_superior_candidate_that_does_not_beat_by_delta_does_not_go_forward():
    a = FamilyDelta('relatedness_co_ownership', 'tree', 0.02, np.linspace(0.0, 0.04, 1001), 0.0015)
    verdict = beat_by_margin('c', [a])
    assert verdict.families[0].interval[0] > 0 and not verdict.beats

def test_beat_by_margin_refuses_no_family_or_a_repeated_one(): ...

def test_the_point_statistic_is_the_headline():
    # a panel read's result: point_statistic over every anchor == result.headline['ndcg_10']

def test_relatedness_test_is_reproducible_and_paired():
    # margin from fix_relatedness_margin; reference = the chosen arm's validation result; five
    # candidate seeds = that result with ndcg_10 + 0.1 + 0.001 * i (dataclasses.replace on
    # per_anchor, same data_fingerprint); replicates 2000, bootstrap_seed 20260924: two calls give
    # equal verdicts, families[0].delta == pytest.approx(0.102), level 0.95

def test_relatedness_test_at_two_families():
    # the same inputs plus SECOND_FAMILY results (dataclasses.replace(..., panel=SECOND_FAMILY))
    # and a _hand_record margin: level 0.975 and two FamilyVerdicts

def test_fewer_than_min_seeds_is_refused():
    with pytest.raises(ValueError, match='4 .*5'):
        relatedness_test('c', {'relatedness_co_ownership': seeds[:4]}, ..., min_seeds=5, ...)

def test_seeds_read_on_another_build_or_part_are_refused(): ...
```

- [ ] **Step 2: Run them to see them fail.** Expected: FAIL with `ImportError` on
  `beat_by_margin`.

- [ ] **Step 3: Implement the "Test" section.**

- [ ] **Step 4: Run the tests.** Run: `uv run pytest tests/unit/test_decision_relatedness.py -q`.
  Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/naics_embedder/decision/relatedness.py tests/unit/test_decision_relatedness.py
git commit -m "feat(decision): add relatedness's beat-by-delta test over n judging families"
```

---

### Task 9: The provenance check

**Files:**
- Modify: `src/naics_embedder/decision/relatedness.py` ("Provenance" section)
- Test: `tests/unit/test_decision_relatedness.py`

**Interfaces:**
- Produces (P3):
  - `RELATEDNESS_EXPORT = 'relatedness_export'`
  - `PROVENANCE_ENTRIES = frozenset({'writer', 'codes', 'dimension', 'inputs', 'runs',
    'configuration', 'seed', 'checkpoint', 'table_sha256', 'generated_at',
    'library_versions'})`
  - `CODEBOOK_INPUTS = ('codebook', 'text_only')`, which are required.
  - `TRAINING_FAMILY_INPUTS = ('ita_area_mix',)`, which is optional (§4.1, after G1).
  - `check_relatedness_provenance(content: Mapping[str, Any], *, codebook_sha256: str,
    text_only_sha256: str) -> None`

It checks in this order, and each message names its check and ends with
`(relatedness §4.6: codebook, text-only table and training families only)`:

1. `writer`: it is `RELATEDNESS_EXPORT`.
2. `entries`: they are exactly `PROVENANCE_ENTRIES`. An unknown entry, such as `co_ownership`, is
   refused by name.
3. `inputs`: a mapping that holds `CODEBOOK_INPUTS`, holds nothing outside `CODEBOOK_INPUTS` and
   `TRAINING_FAMILY_INPUTS`, and maps each name to exactly a string `path` and a string `sha256`.
4. `codebook` and `text_only`: each SHA-256 equals its argument.
5. `runs`: a mapping keyed by exactly the training-family inputs present, each value a non-blank
   string.
6. `checkpoint`: exactly a string `path` and a string `sha256`.
7. `seed`: an `int`, not a `bool`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_a_provenance_that_names_co_ownership_is_refused():
    with pytest.raises(ValueError, match="inputs.*ita_co_ownership"):
        check_relatedness_provenance(with_input('ita_co_ownership'), **SHAS)
    with pytest.raises(ValueError, match="entries.*co_ownership"):
        check_relatedness_provenance({**VALID, 'co_ownership': {}}, **SHAS)

def test_a_valid_provenance_passes_with_and_without_the_area_mix(): ...

@pytest.mark.parametrize('case', PROVENANCE_CASES, ids=[case.id for case in PROVENANCE_CASES])
def test_each_rule_refuses(case): ...
# writer 'export'; a missing 'seed'; inputs without 'text_only'; an input with an extra 'run' key;
# another codebook sha256; another text_only sha256; runs {} with ita_area_mix present; runs
# naming 'ita_area_mix' without that input; checkpoint without 'sha256'; seed True
```

- [ ] **Step 2: Run them to see them fail.** Expected: FAIL with `ImportError` on
  `check_relatedness_provenance`.

- [ ] **Step 3: Implement the "Provenance" section.**

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/naics_embedder/decision/relatedness.py tests/unit/test_decision_relatedness.py
git commit -m "feat(decision): refuse a relatedness table whose provenance names a judging input"
```

---

### Task 10: Docs, guide and the final gate

**Files:**
- Create: `docs/api/relatedness_panel.md`
- Modify: `docs/.nav.yml`, `docs/api/decision.md`, `CLAUDE.md`, `README.md`, `tests/README.md`

- [ ] **Step 1: Bring the branch up to date.** Plans 18 and 20 also edit `CLAUDE.md`. Run
  `git fetch origin && git rebase origin/main`, then rerun the six new test files. Expected:
  PASS. If the rebase stops on a conflict, stop and report it.

- [ ] **Step 2: Write `docs/api/relatedness_panel.md`.** It has:
  - A one-paragraph intro. ITA co-ownership judges relatedness and never trains it (U42). The
    validation part selects, and the sealed part opens once, at roadmap Stage 7b (U43, U45).
    Stage 6 reads fixture tables only.
  - §5's role table, copied as written.
  - `:::` blocks for `naics_embedder.panels.relatedness_split`, `naics_embedder.panels.relatedness`,
    `naics_embedder.panels.relatedness_scoring` and `naics_embedder.panels.relatedness_arms`.

  Add `- Relatedness Panel: api/relatedness_panel.md` after the ITA Panel entry in
  `docs/.nav.yml`. Add a "Relatedness margin" section with
  `::: naics_embedder.decision.relatedness` to `docs/api/decision.md`, after "Text→code margin".

- [ ] **Step 3: Update `CLAUDE.md`, `README.md` and `tests/README.md`.**
  - In `CLAUDE.md`, add the five modules to the directory tree's `panels/` and `decision/`
    comments.
  - Add one paragraph after the ITA paragraph under "Monitor, Epoch Summary and Selection". It
    names the panel, its anchor-industry split, NDCG@10, and the sealed part opened once at
    Stage 7b.
  - Add one bullet under "Testing and Validation" that names the six new test files and
    `tests/fixtures/relatedness.py`.
  - Recount with `find src -name '*.py' | wc -l`, `ls tests/unit/test_*.py | wc -l` and
    `uv run pytest --collect-only -q -q | tail -1`. At `c055739` the first two give 128 and 112.
    Write the new numbers wherever the three files state them.

- [ ] **Step 4: Check the untouched seams.** Run:

```bash
git diff --exit-code c055739 -- src/naics_embedder/panels/outcome.py src/naics_embedder/panels/ita.py src/naics_embedder/panels/ita_split.py src/naics_embedder/panels/ita_scoring.py src/naics_embedder/panels/decoding.py src/naics_embedder/panels/selection_log.py src/naics_embedder/decision/rule.py src/naics_embedder/decision/resampling.py src/naics_embedder/decision/records.py src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py src/naics_embedder/decision/text_to_code.py src/naics_embedder/decision/provenance.py tests/unit/test_outcome_panel.py tests/unit/test_outcome_decoding.py tests/unit/test_ita_split.py tests/unit/test_ita_inputs.py tests/unit/test_ita_scoring.py tests/unit/test_ita_panel.py tests/unit/test_selection_log_guard.py tests/unit/test_decision_rule.py tests/unit/test_decision_resampling.py tests/unit/test_decision.py tests/unit/test_decision_economic.py tests/unit/test_decision_text_to_code.py tests/unit/test_economic_candidates.py; echo exit=$?
```

Expected: `exit=0`. If the rebase brought in another PR's change to one of these files, compare
against that PR's merge commit instead, and say so in the ledger.

- [ ] **Step 5: Run the gates.** Each must pass:

```bash
uv run pytest -n auto -q
uv run ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run mkdocs build --strict
```

Expected: no failures, Ruff prints `All checks passed!`, the style gate exits 0, and MkDocs
builds with no warning. Check that the rendered `site/api/relatedness_panel/index.html` has the
four modules' headings, and that `site/api/decision/index.html` has the "Relatedness margin"
anchor.

- [ ] **Step 6: Confirm that no real table was read.** Run:
  `grep -rn "embedding-data\|ita_co_ownership.parquet\|co_ownership_marginals.parquet" src/ tests/ | grep -v "tests/fixtures/relatedness.py"`.
  Expected: only docstrings, comments and file-name constants, with no absolute path.

- [ ] **Step 7: Commit.**

```bash
git add docs/api/relatedness_panel.md docs/.nav.yml docs/api/decision.md CLAUDE.md README.md tests/README.md
git commit -m "docs: add the relatedness panel and margin API pages"
```

---

## Spec coverage

| Spec item | Task |
|---|---|
| §4.1 families and roles, §5 roles | 10 (API page); the spec names every family and part |
| §4.2 inputs, node ids, refusals, zeros | 2 |
| §4.2 pins | 6 (`from_files`) |
| §4.3 unit, rule, config, stability, split fingerprint | 1 and 6 |
| §4.3 mirror pairs and the crossing count | 1 (fixture) and 6 |
| §4.4 arms, candidates, relevance, gain, anchors | 2, 5 and 6 |
| §4.4 ties, NDCG@10, headline, reports, result shape | 3 and 4 |
| §4.5 reads, data fingerprint, sealed opening, arms | 5 and 6 |
| §4.6 reference arms and choice, margin, margin record, margin-first | 7 |
| §4.6 test at 1 − 0.05/n, intervals, seeds, which read decides | 8 |
| §4.6 provenance | 9 |
| §6 constraints and untouched seams | Global Constraints and 10 |
| §10 items 1–2 | 6 and 1 |
| §10 items 3–4 | 2 and 6 |
| §10 items 5–6 | 3 and 4 |
| §10 items 7–8 | 7 and 6 |
| §10 items 9–12 | 8, 8, 7 and 9 |
| §10 items 13–16 | 10 (step 4), §5 and Task 10's page, 10 (step 6), 10 (step 5) |

§4.7 (the method) and §4.8 (the opening) belong to Stages 7 and 7b. Tasks 6, 8 and 9 build what
they consume.
