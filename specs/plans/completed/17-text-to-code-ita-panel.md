# Text→code ITA Panel Implementation Plan

**Status: COMPLETE (2026-11-17)** — executed via subagent-driven-development; deferred items in specs/deferred_items.md

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 2

**Goal:** Build text→code's ITA panel on fixtures: the grouped split into a validation and a
sealed part, the shortlist scorer with its headline, copied, phrasing and conflict reports, the
logged reads and the one logged sealed opening, and the reference choice and beat-by-δ margin
rule as code.

**Architecture:** Four new source modules, each with its own tests.

- **`panels/ita_split.py`.** The settings in `conf/data/ita_panel.yaml`, the salted hash split of
  `description_key`, and the split fingerprint.
- **`panels/ita_scoring.py`.** The shortlist scorer for pairs with one or more reported codes, and
  the summaries (headline, copied, phrasing, counts, per code). It reuses `panels/decoding.py`'s
  helpers and leaves that file unchanged.
- **`panels/ita.py`.** Validation of Stage 10's two tables, node-id parsing, the panel pairs, and
  `ItaPanel`: reads logged to the shared `SelectionLog` under `ita_text`, and the sealed opening.
- **`decision/text_to_code.py`.** The paired bootstrap mapping, the reference choice, the
  beat-by-δ verdict, the margin record and its margin-first check.

**Tech Stack:** Python 3.10+ (locked env 3.12), PyTorch (float64 CPU distances), Polars, numpy,
Pydantic v2, PyYAML, pytest, mkdocstrings. No new dependency.

**Spec:** `specs/completed/text-to-code-ita-panel.md` (APPROVED 2026-11-17, U41; merged by PR #163 as
`4fa2857`). Read it with this plan. §n cites that spec. Stage 10 cites embedding-data's
`specs/ita-establishment-targets.md` at `ec6fb29` ("Tables" section).

## Global Constraints

Every task's requirements include this section.

- **No real ITA pair.** No task reads an embedding-data table, a real bundle's panel or the real
  selection log. Tests use `tests/fixtures/ita.py` and write under `tmp_path` (§1, §6).
- **Untouched seams.** `src/naics_embedder/panels/outcome.py`, `panels/decoding.py`,
  `panels/selection_log.py`, `decision/rule.py`, `decision/resampling.py`, `decision/records.py`,
  `decision/decide.py` and `decision/economic.py` stay identical to `4fa2857`. Their tests
  (`test_outcome_panel.py`, `test_outcome_decoding.py`, `test_selection_log_guard.py`,
  `test_decision_rule.py`, `test_decision_resampling.py`, `test_decision.py`,
  `test_decision_economic.py`) pass with no edit (§2 anti-goals, §6). Task 8 checks it with
  `git diff --exit-code 4fa2857 -- <those files>`.
- **`uv.lock` does not change.** Its sha256 stays
  `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948`. If a step would change it,
  stop and ask the user (§6).
- **Panel name** `ita_text`. **Parts** `validation` (hash 0) and `sealed` (hash 1). **r** = 0.05
  (U38). **Reference arms** `lexical-stub` and `euclidean-d16`; a tie goes to `lexical-stub`
  (§4.5).
- **Refuse, never drop.** An input that disagrees with the spec raises `ValueError` at panel
  construction, before any log record is written (§4.1).
- **Style.** Single quotes, YAPF layout, semantic section dividers, one blank line after imports.
  Format touched Python files with `./scripts/format_code.sh <files>`. Never run `ruff format`.
- **Leave alone.** The roadmap (ticked at completion), `specs/deferred_items.md` until the
  completion protocol, and the methodology branch `claude/naics-artifacts-methodology-985a4c`.

## Review Focus

1. **One 2022 node listed under several filing years for one pair.** It is one reported code, not
   a conflict. Pinned by `test_a_node_repeated_across_years_is_one_reported_code` (Task 2).
2. **Two reported codes at the same distance from the query.** The pair ranks 1, not 2. Pinned by
   `test_tied_reported_codes_do_not_count_against_each_other` (Task 3).
3. **A part whose every panel pair is copied.** The read refuses with a clear message instead of
   a NaN headline that fails later inside a record. Pinned by
   `test_a_part_with_no_headline_pair_is_refused` (Task 4).
4. **The same tables with rows in another order and an extra column.** The panel scores the same
   pairs and logs the same data fingerprint. Pinned by
   `test_row_order_and_extra_columns_leave_the_data_fingerprint_alone` (Task 5).
5. **ITA reads in the shared selection log.** A D8 or economic run's records still hold only its
   own reads. Pinned by `test_ita_reads_stay_out_of_another_runs_log_records` (Task 5).

## Decisions

- **O1** (owner, plan review, 2026-11-17): "Approve the plan": the plan as written, P1–P7
  included. Applied in: the whole plan.

The plan's own calls, approved by O1:

- **P1** (plan, 2026-11-17): the split hashes `f'{salt}\x00{description_key}'` in UTF-8 with
  SHA-256 and reads the first 8 bytes big-endian, modulo 2. The rule fingerprint is the SHA-256 of
  `json.dumps(rule, sort_keys=True, separators=(',', ':'))`. The salt is
  `9986637c5a4468938b7bd5c4f1e8a4cd`, drawn once with `secrets.token_hex(16)` at planning (§4.2,
  §8). Applied in: Task 1.
- **P2** (plan, 2026-11-17): a pair's rank is 1 + the number of *non-reported* candidates at a
  distance ≤ the nearest reported code's distance. This reads §4.3's "other reported codes never
  count against it" literally and keeps ties against the truth. A level-L rank uses the same rule,
  with "reported" replaced by "shares a reported code's level-L ancestor". Applied in: Task 3.
- **P3** (plan, 2026-11-17): reported codes are the distinct codes of the pair's
  `flag_headline_basis` rows. The panel refuses a pair whose distinct-code count disagrees with
  `flag_headline_codes_conflict` (more than one ⇔ true). Applied in: Task 2.
- **P4** (plan, 2026-11-17): the scorer reads the distance from the encoder's `distance`
  attribute (`ArmEncoder` has one), or uses `'cosine'` when it has none (the stub). Applied in:
  Task 5.
- **P5** (plan, 2026-11-17): `score_logged` refuses a caller `role` other than `reference` or
  `candidate`, before it logs. A missing role is allowed and counts as a candidate read in the
  margin-first check (§4.5). Applied in: Task 5.
- **P6** (plan, 2026-11-17): the beat-by-δ comparison does not enforce `min_seeds`. Stage 5's read
  applies `conf/data/decision.yaml`'s floor to candidates. The stub is a one-seed arm (§4.5).
  Applied in: Task 6.
- **P7** (plan, 2026-11-17): the margin record is a Pydantic model in `decision/text_to_code.py`,
  not in `decision/records.py`, so `decide`'s record models stay unchanged. Applied in: Task 7.

---

## File structure

| File | Responsibility |
|---|---|
| `conf/data/ita_panel.yaml` (create) | Panel name, rule version, salt, r |
| `src/naics_embedder/panels/ita_split.py` (create) | Settings, split rule, split fingerprint |
| `src/naics_embedder/panels/ita.py` (create) | Table validation, node ids, panel pairs, `ItaPanel` |
| `src/naics_embedder/panels/ita_scoring.py` (create) | Shortlist ranks, coarser levels, summaries |
| `src/naics_embedder/decision/text_to_code.py` (create) | Bootstrap mapping, reference, verdict, margin |
| `tests/fixtures/ita.py` (create) | Stage 10's two tables with every column and dtype; stub encoder |
| `tests/unit/test_ita_split.py`, `test_ita_inputs.py`, `test_ita_scoring.py`, `test_ita_panel.py`, `test_decision_text_to_code.py` (create) | One file per module seam |
| `docs/api/ita_panel.md` (create), `docs/api/decision.md`, `docs/.nav.yml`, `CLAUDE.md` (modify) | API pages and the guide |

---

### Task 1: Fixture tables, settings and the split

**Files:**
- Create: `tests/fixtures/ita.py`, `conf/data/ita_panel.yaml`,
  `src/naics_embedder/panels/ita_split.py`
- Test: `tests/unit/test_ita_split.py`

**Interfaces:**
- Produces (`panels/ita_split.py`):
  - `ITA_PANEL = 'ita_text'`
  - `class ItaPart(str, Enum)`: `VALIDATION = 'validation'`, `SEALED = 'sealed'`
  - `class ItaPanelSettings(BaseModel)` (frozen, `extra='forbid'`): `panel: Literal['ita_text']`,
    `rule_version: Literal[1]`, `salt: str` (32 lowercase hex characters),
    `relative_margin: float` (> 0, finite)
  - `load_ita_settings(path: Union[str, Path]) -> ItaPanelSettings`
  - `split_rule(settings: ItaPanelSettings) -> Dict[str, Any]`: exactly
    `{'salt', 'rule_version', 'hash': 'sha256', 'prefix_bytes': 8, 'byteorder': 'big',
    'modulus': 2, 'parts': ['validation', 'sealed']}`
  - `split_fingerprint(settings: ItaPanelSettings) -> str`
  - `split_part(description_key: str, salt: str) -> ItaPart`
- Produces (`tests/fixtures/ita.py`):
  - `ITA_PAIRS_SCHEMA`, `ITA_PAIR_CODES_SCHEMA`: `Dict[str, pl.DataType]` with **every** column
    of Stage 10's two tables, in its order, with its dtype (`UInt64` counts, `Int64` years,
    `List(Utf8)` lists, `Boolean` flags).
  - `ITA_CANDIDATES = ('111110', '111120', '211120', '236220', '311111', '321113', '513210',
    '621210')`
  - `ita_tables(extra_pairs: Sequence[Dict] = (), extra_codes: Sequence[Dict] = (),
    replace_codes: Optional[Mapping[str, Sequence[Dict]]] = None) -> Tuple[pl.DataFrame,
    pl.DataFrame]`: the base rows below plus any extras, columns not given filled with neutral
    values (false flags, zero counts, empty lists), cast to the schemas.
  - `write_ita_tables(directory: Path, pairs: pl.DataFrame, codes: pl.DataFrame) ->
    Tuple[Path, Path]` writing `ita_text_pairs.parquet` and `ita_text_pair_codes.parquet`.
  - `ITA_SETTINGS = ItaPanelSettings(panel='ita_text', rule_version=1,
    salt='9986637c5a4468938b7bd5c4f1e8a4cd', relative_margin=0.05)`
  - `class AssignedStubEncoder`: codes one-hot on `ITA_CANDIDATES`' axes, each query on the axis
    of the code in `ASSIGNED[query]`. It has no `distance` attribute.

The base fixture (node id `naics:2022:national_industry:<code>` on every `flag_headline_basis`
row; `description_key` parts are fixed by P1's salt):

| pair_id | description_key | representative_string | part | reported codes | flags | ASSIGNED |
|---|---|---|---|---|---|---|
| p01 | we make dog food | We make dog food | validation | 311111 (rows for 2023 and 2024), plus a 2017 crosswalked row, `flag_headline_basis` false, null node | — | 311111 |
| p02 | we make dog food | we make dog food | validation | 311111 | — | 311111 |
| p03 | oil and gas wells | Oil and gas wells | validation | 211120 | — | 111110 |
| p04 | cat food plant | Cat food plant | validation | 311111, 111110 | conflict | 111110 |
| p05 | software publisher | Software publisher | validation | 513210 | copied_title_any | 513210 |
| p06 | corn farming | Corn farming | validation | 111110 | `flag_headline` false | — |
| p07 | general contractor | General contractor | validation | 236220 | company_fallback | 236220 |
| p08 | soybean farm | Soybean farm | sealed | 111110 | — | 111110 |
| p09 | canola growers | Canola growers | sealed | 111120 | — | 111120 |
| p10 | sawmill | Sawmill | sealed | 321113 | mojibake | 321113 |
| p11 | dental office | Dental office | sealed | 621210 | copied_index_any | 621210 |

- [x] **Step 1: Write the failing tests** in `tests/unit/test_ita_split.py`:

```python
GOLDEN = {
    'we make dog food': 'validation', 'soybean farm': 'sealed', 'oil and gas wells': 'validation',
    'canola growers': 'sealed', 'sawmill': 'sealed', 'dental office': 'sealed',
    'general contractor': 'validation', 'pet food maker': 'validation',
}

def test_split_part_matches_the_golden_parts():
    for key, part in GOLDEN.items():
        assert split_part(key, ITA_SETTINGS.salt) is ItaPart(part)

def test_split_fingerprint_is_pinned():
    assert split_fingerprint(ITA_SETTINGS) == (
        '8602ceb243722dc64e0a29a9ba444fe824baf209a848964d7ddf130a9db5cc7c'
    )

def test_committed_settings_hold_the_planned_values():
    # ROOT = Path(__file__).resolve().parents[2], as test_economic_candidates.py resolves it
    settings = load_ita_settings(ROOT / 'conf/data/ita_panel.yaml')
    assert settings == ITA_SETTINGS

def test_each_description_key_falls_in_one_part():
    pairs, _ = ita_tables()
    parts = pairs.with_columns(
        part=pl.col('description_key').map_elements(
            lambda key: split_part(key, ITA_SETTINGS.salt).value, return_dtype=pl.Utf8))
    assert parts.group_by('description_key').agg(pl.col('part').n_unique()).get_column('part').max() == 1
    assert set(parts.get_column('part')) == {'validation', 'sealed'}

def test_a_rebuilt_table_keeps_every_old_key_in_its_old_part():
    # add p12 'pet food maker'; every base key's part is unchanged and the fingerprint is equal

def test_settings_refuse_a_bad_salt_or_margin():
    # salt of 31 chars, uppercase hex, relative_margin 0 and inf, an unknown key: each ValidationError

def test_fixture_tables_carry_every_stage_10_column():
    pairs, codes = ita_tables()
    assert dict(pairs.schema) == ITA_PAIRS_SCHEMA and dict(codes.schema) == ITA_PAIR_CODES_SCHEMA
```

- [x] **Step 2: Run them to see them fail.** Run: `uv run pytest tests/unit/test_ita_split.py -q`.
  Expected: FAIL with `ModuleNotFoundError: No module named 'naics_embedder.panels.ita_split'`.

- [x] **Step 3: Write `conf/data/ita_panel.yaml`** with the four keys and a comment that the salt
  was drawn once at plan 17 and never changes, because openings of the sealed part are keyed on
  the split fingerprint (§4.2, §4.4):

```yaml
panel: ita_text
rule_version: 1
salt: '9986637c5a4468938b7bd5c4f1e8a4cd'
relative_margin: 0.05
```

- [x] **Step 4: Implement `panels/ita_split.py` and `tests/fixtures/ita.py`** with the interfaces
  above and P1's encoding.
  > Deviation: fixture-only tests are proven by a hand harness (`mutant-run` refuses `tests/` patches), plus process-seed and as-given key tests.

- [x] **Step 5: Run the tests.** Run: `uv run pytest tests/unit/test_ita_split.py -q`.
  Expected: PASS.

- [x] **Step 6: Commit.**

```bash
git add conf/data/ita_panel.yaml src/naics_embedder/panels/ita_split.py tests/fixtures/ita.py tests/unit/test_ita_split.py
git commit -m "feat(panels): add the ITA panel's salted description split and fixture tables"
```

---

### Task 2: Stage 10 tables into panel pairs

**Files:**
- Create: `src/naics_embedder/panels/ita.py` (input section only)
- Test: `tests/unit/test_ita_inputs.py`

**Interfaces:**
- Consumes: `split_part`, `ItaPart` (Task 1); `ita_tables`, `ITA_CANDIDATES`, `ITA_SETTINGS`
  (fixture).
- Produces (`panels/ita.py`):
  - `PAIR_COLUMNS: Dict[str, pl.DataType]`: `pair_id`, `description_key`,
    `representative_string` (Utf8); `flag_headline`, `flag_headline_codes_conflict`,
    `flag_copied_title_any`, `flag_copied_index_any`, `flag_suspected_mojibake`,
    `flag_employer_company_fallback` (Boolean).
  - `PAIR_CODE_COLUMNS: Dict[str, pl.DataType]`: `pair_id`, `naics_2022_node_id` (Utf8),
    `flag_headline_basis` (Boolean).
  - `NODE_PREFIX = ('naics', '2022', 'national_industry')`
  - `node_code(node_id: str) -> str`: the six-digit `<code>`; `ValueError` naming the node id for
    any other system, vintage, level, part count or a code that is not six digits.
  - `PANEL_PAIR_SCHEMA`: `pair_id`, `description_key`, `query` (Utf8), `codes` (List(Utf8),
    sorted, distinct), `copied`, `conflict`, `mojibake`, `company_fallback` (Boolean), `part`
    (Utf8).
  - `panel_pairs(pairs: pl.DataFrame, pair_codes: pl.DataFrame, candidates: Sequence[str],
    salt: str) -> pl.DataFrame`: headline pairs only, sorted by `pair_id`, in
    `PANEL_PAIR_SCHEMA`. `copied = flag_copied_title_any | flag_copied_index_any` (U38).

`panel_pairs` raises `ValueError` for: a missing column or a column of another dtype in either
table (naming it); a repeated `pair_id`; a headline pair with a null `representative_string`; a
headline pair with no `flag_headline_basis` row; a basis row with a null node id; `node_code`'s
refusals; a code not among `candidates`; P3's conflict disagreement. Extra columns are ignored.

- [x] **Step 1: Write the failing tests** in `tests/unit/test_ita_inputs.py`:

```python
def test_panel_pairs_are_the_headline_pairs_with_their_parts():
    frame = panel_pairs(*ita_tables(), ITA_CANDIDATES, ITA_SETTINGS.salt)
    assert frame.get_column('pair_id').to_list() == [f'p{n:02d}' for n in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)]
    assert frame.filter(pl.col('part') == 'sealed').get_column('pair_id').to_list() == ['p08', 'p09', 'p10', 'p11']
    assert frame.filter(pl.col('pair_id') == 'p04').get_column('codes').to_list() == [['111110', '311111']]
    assert frame.filter(pl.col('copied')).get_column('pair_id').to_list() == ['p05', 'p11']

def test_a_node_repeated_across_years_is_one_reported_code():
    frame = panel_pairs(*ita_tables(), ITA_CANDIDATES, ITA_SETTINGS.salt)
    row = frame.filter(pl.col('pair_id') == 'p01').row(0, named=True)
    assert row['codes'] == ['311111'] and row['conflict'] is False

@pytest.mark.parametrize('node_id', [
    'naics:2022:industry:517112', 'naics:2017:national_industry:311111',
    'sic:2022:national_industry:311111', 'naics:2022:national_industry:31111',
])
def test_node_code_refuses_another_system_vintage_level_or_width(node_id):
    with pytest.raises(ValueError, match=re.escape(node_id)):
        node_code(node_id)

def test_a_code_outside_the_candidates_is_refused():
    # p03's basis row replaced by node naics:2022:national_industry:999999 → ValueError naming 999999

def test_a_headline_pair_without_a_reported_code_is_refused(): ...
def test_a_null_representative_string_is_refused(): ...
def test_a_conflict_flag_that_disagrees_with_the_codes_is_refused():
    # p04 with flag_headline_codes_conflict false → ValueError; p03 with it true → ValueError
def test_a_missing_or_retyped_column_is_refused():
    # drop description_key → ValueError naming it; flag_headline cast to Utf8 → ValueError naming it
def test_a_repeated_pair_id_is_refused(): ...
```

- [x] **Step 2: Run them to see them fail.** Run: `uv run pytest tests/unit/test_ita_inputs.py -q`.
  Expected: FAIL with `ModuleNotFoundError: No module named 'naics_embedder.panels.ita'`.

- [x] **Step 3: Implement `node_code` and `panel_pairs`** in `panels/ita.py` under an "Inputs"
  divider.
  > Deviation: `panel_pairs` also refuses nulls in read columns and a repeated `pair_id` anywhere in the pair table.

- [x] **Step 4: Run the tests.** Run: `uv run pytest tests/unit/test_ita_inputs.py -q`.
  Expected: PASS.

- [x] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/ita.py tests/unit/test_ita_inputs.py
git commit -m "feat(panels): read Stage 10's ITA pair tables into panel pairs, refusing disagreements"
```

---

### Task 3: The shortlist scorer

**Files:**
- Create: `src/naics_embedder/panels/ita_scoring.py` (scorer section)
- Test: `tests/unit/test_ita_scoring.py`

**Interfaces:**
- Consumes: `resolve_distance`, `lca_level`, `HIT_KS`, `_as_float64`, `score_decoding` (test
  only) from `panels/decoding.py`; `code_lineage` from `utils/naics_hierarchy.py`.
- Produces:
  - `COARSE_LEVELS = (2, 3, 4, 5)`
  - `score_shortlist(query_points: torch.Tensor, reported_codes: Sequence[Sequence[str]],
    candidate_points: torch.Tensor, candidate_codes: Sequence[str], pair_ids: Sequence[str],
    distance: Union[str, DistanceFn] = 'cosine') -> pl.DataFrame`, one row per pair with:
    `pair_id`, `codes` (List(Utf8), sorted), `top1_code`, `rank` (Int64), `reciprocal_rank`,
    `hit_at_1`, `hit_at_5`, `hit_at_10`, `lca_level` (Int64), and for each L in
    `COARSE_LEVELS`: `rank_l{L}`, `reciprocal_rank_l{L}`, `hit_at_{k}_l{L}`.

Rules (P2): `rank` = 1 + #{non-reported c : d(c) ≤ min over reported d}. `top1_code` is the
nearest reported code when `rank == 1`, else the argmin over non-reported candidates.
`lca_level` is the max of `lca_level(top1_code, code)` over reported codes. Level L's ancestor of
a code is `code_lineage(code)[L - 2]`, so 31–33, 44–45 and 48–49 share one sector. `rank_l{L}` =
1 + #{c whose level-L ancestor is not a reported code's : d(c) ≤ min d over candidates whose
level-L ancestor is}. It raises `ValueError` as `score_decoding` does on shapes, repeated
candidates and non-finite distances, and on an empty `codes` entry or a code that is not a
candidate.

- [x] **Step 1: Write the failing tests:**

```python
def test_single_code_pairs_rank_as_score_decoding_ranks_them():
    # random float32 points (seeded), 6 candidates from ITA_CANDIDATES, 20 queries, one code each,
    # euclidean and cosine: rank and top1_code equal score_decoding's per_query rank and top1_code

def test_ties_count_against_the_truth():
    # every candidate at one identical point: each pair's rank == number of candidates

def test_tied_reported_codes_do_not_count_against_each_other():
    # one query equidistant from reported 311111 and 111110, every other candidate farther
    assert scores.get_column('rank').to_list() == [1]

def test_the_rank_is_the_best_over_reported_codes():
    # reported (211120, 311111); distances put 311111 3rd and 211120 5th; two non-reported in
    # front → rank 3, top1_code the nearest non-reported code, lca_level the max over both

def test_coarser_levels_read_the_lineage_with_combined_sectors():
    # reported 311111; nearest candidate 321113, then 311111 → rank 2, rank_l2 1 (31 and 32 are one
    # sector), rank_l3 2 (311 vs 321); hit_at_1_l2 true, hit_at_1_l3 false

def test_a_code_that_is_not_a_candidate_or_an_empty_code_list_is_refused(): ...
```

- [x] **Step 2: Run them to see them fail.** Run: `uv run pytest tests/unit/test_ita_scoring.py -q`.
  Expected: FAIL with `ModuleNotFoundError`.

- [x] **Step 3: Implement `score_shortlist`.** Compute the (Q, C) distance matrix once, then
  boolean (Q, C) masks for "reported" and for each level's "shares an ancestor"; each rank is
  `1 + (masked_out & (d <= min_in[:, None])).sum(1)`.

- [x] **Step 4: Run the tests.** Run: `uv run pytest tests/unit/test_ita_scoring.py -q`.
  Expected: PASS.

- [x] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/ita_scoring.py tests/unit/test_ita_scoring.py
git commit -m "feat(panels): add the ITA shortlist scorer over every reported code"
```

---

### Task 4: Headline, copied, phrasing and count reports

**Files:**
- Modify: `src/naics_embedder/panels/ita_scoring.py` (summary section)
- Test: `tests/unit/test_ita_scoring.py`

**Interfaces:**
- Consumes: `score_shortlist` (Task 3); `PANEL_PAIR_SCHEMA` frames (Task 2).
- Produces:
  - `SUMMARY_METRICS`: `mrr`, `top1`, `hit_at_1`, `hit_at_5`, `hit_at_10`, `lca_level`, and for
    each L in `COARSE_LEVELS`: `mrr_l{L}`, `hit_at_{k}_l{L}`.
  - `@dataclass(frozen=True) class ItaResult`: `per_pair: pl.DataFrame` (the scorer's columns
    plus `description_key` and `copied`), `per_code: pl.DataFrame` (`code`, `n_pairs`, each
    metric's mean, over non-copied pairs, a pair counted under each of its codes), `headline`,
    `copied`, `phrasing: Dict[str, Union[int, float, str, None]]`, `counts: Dict[str, int]`.
  - `summarize_ita(scores: pl.DataFrame, pairs: pl.DataFrame, distance: str) -> ItaResult`.

`headline`: `n_pairs`, `distance` and each `SUMMARY_METRICS` mean over non-copied pairs.
`copied`: the same over copied pairs; with none, `n_pairs` 0 and each metric `None`. `phrasing`:
`n_groups` and, per metric, the mean over `description_key` groups of the group's non-copied
mean. `counts`: `n_pairs`, `n_groups`, `n_copied`, `n_conflict`, `n_mojibake`,
`n_company_fallback` over the part's panel pairs. A part with no non-copied pair raises
`ValueError('… has no headline pair …')`.

- [x] **Step 1: Write the failing tests.** Build the result without `ItaPanel` (Task 5): take
  `panel_pairs(*ita_tables(), ITA_CANDIDATES, ITA_SETTINGS.salt)` filtered to `part ==
  'validation'`, encode its queries and `ITA_CANDIDATES` with `AssignedStubEncoder`, call
  `score_shortlist(..., distance='cosine')`, then `summarize_ita(scores, pairs, 'cosine')`.
  p03 ranks 8 and the others 1:

```python
def test_the_headline_excludes_copied_pairs_and_counts_each_pair_once():
    assert result.headline['n_pairs'] == 5
    assert result.headline['mrr'] == pytest.approx((1 + 1 + 1 / 8 + 1 + 1) / 5)  # 0.825

def test_the_phrasing_report_counts_each_description_once():
    assert result.phrasing['n_groups'] == 4
    assert result.phrasing['mrr'] == pytest.approx((1 + 1 / 8 + 1 + 1) / 4)  # 0.78125

def test_copied_pairs_are_reported_apart():
    assert result.copied['n_pairs'] == 1 and result.copied['mrr'] == 1.0

def test_counts_report_conflicts_and_flags_without_changing_the_headline():
    assert result.counts == {'n_pairs': 6, 'n_groups': 5, 'n_copied': 1, 'n_conflict': 1,
                             'n_mojibake': 0, 'n_company_fallback': 1}

def test_per_code_counts_a_conflict_pair_under_each_code(): ...
def test_a_part_with_no_headline_pair_is_refused():
    # every pair copied → ValueError matching 'no headline pair'
```

- [x] **Step 2: Run them to see them fail.** Run: `uv run pytest tests/unit/test_ita_scoring.py -q`.
  Expected: FAIL with `ImportError` on `summarize_ita`.

- [x] **Step 3: Implement `ItaResult` and `summarize_ita`.**
  > Deviation: every report mean is `math.fsum(values) / len(values)` (Polars chunked means were not bit-reproducible), and null scores are refused.

- [x] **Step 4: Run the tests.** Expected: PASS.

- [x] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/ita_scoring.py tests/unit/test_ita_scoring.py
git commit -m "feat(panels): add the ITA headline, copied, phrasing and count reports"
```

---

### Task 5: `ItaPanel`: reads, the data fingerprint and the sealed opening

**Files:**
- Modify: `src/naics_embedder/panels/ita.py` (panel section)
- Test: `tests/unit/test_ita_panel.py`

**Interfaces:**
- Consumes: Tasks 1–4; `SelectionLog`, `SelectionEvent`, `merge_read_detail`;
  `SealedSplitError`, `SplitAlreadyOpenedError`, `QueryCodeEncoder` from `panels/outcome.py`
  (imported, not changed); `ValidatedSupervisionBundle`.
- Produces:
  - `READ_ROLES = ('reference', 'candidate')`
  - `class ItaPanel`:
    - `__init__(self, pairs: pl.DataFrame, pair_codes: pl.DataFrame, candidates: Sequence[str],
      settings: ItaPanelSettings, log: SelectionLog)`: candidates must be distinct six-digit
      codes; runs `panel_pairs`; sets `self.candidates` (sorted tuple), `self.settings`,
      `self.log`, `self.fingerprint = split_fingerprint(settings)`.
    - `from_files(cls, pairs_parquet, pair_codes_parquet, bundle: ValidatedSupervisionBundle,
      settings_path, log_path) -> ItaPanel`: candidates are the bundle codebook's six-digit codes,
      as `OutcomePanel.from_bundle` reads them.
    - `data_fingerprint(self, part: Union[ItaPart, str]) -> str`: SHA-256 of
      `json.dumps({'pairs': [[pair_id, query, codes, copied], …], 'candidates': [...]},
      sort_keys=True, separators=(',', ':'))`, pairs in `pair_id` order. Not a read.
    - `open_sealed(self, purpose: str, *, reopen_reason: Optional[str] = None) -> None`
    - `score(self, encoder, part, purpose, detail=None) -> ItaResult`
    - `score_logged(self, encoder: QueryCodeEncoder, part: Union[ItaPart, str], purpose: str,
      detail: Optional[Mapping[str, Any]] = None) -> Tuple[ItaResult, Dict[str, Any]]`

`score_logged` refuses a `detail['role']` outside `READ_ROLES` (P5) and a sealed read before
`open_sealed`, both before it logs. It then appends a `read` record (`panel='ita_text'`,
`split=part`, `fingerprint=self.fingerprint`, `n_queries` = the part's panel pairs, detail
`{'encoder': type name, 'distance': name, 'data_fingerprint': …}` merged with the caller's), and
only then encodes. The distance is `getattr(encoder, 'distance', 'cosine')` (P4). `open_sealed`
follows `OutcomePanel.open_test` with panel `ita_text`, split `sealed`, and the split fingerprint.

- [x] **Step 1: Write the failing tests:**

```python
def test_a_validation_read_is_logged_before_any_query_is_encoded():
    # an encoder whose encode_queries raises: the log still holds one read record whose fields are
    # panel 'ita_text', split 'validation', fingerprint split_fingerprint(ITA_SETTINGS),
    # n_queries 6, detail encoder/distance/data_fingerprint plus the caller's arm/role/run

def test_the_stub_is_scored_by_cosine_and_an_arm_by_its_own_distance():
    # AssignedStubEncoder → detail distance 'cosine'; the same stub with distance='euclidean'
    # set on it → 'euclidean', and the result's headline['distance'] matches

def test_a_role_outside_reference_and_candidate_is_refused_before_logging(): ...

def test_the_sealed_part_is_refused_before_it_is_opened():
    with pytest.raises(SealedSplitError): panel.score(stub, 'sealed', 'peek')
    assert not log.path.exists()

def test_a_second_opening_needs_a_reason_also_after_the_tables_are_rebuilt():
    # panel A opens; panel B over tables with p12 'pet food maker' added: same fingerprint,
    # different validation data_fingerprint; B.open_sealed('again') → SplitAlreadyOpenedError;
    # B.open_sealed('again', reopen_reason='audit') logs event 'reopen' with detail reason 'audit'

def test_refusals_happen_before_anything_is_logged():
    # a node id naics:2022:industry:517112 and a code 999999: ItaPanel(...) raises, log absent

def test_row_order_and_extra_columns_leave_the_data_fingerprint_alone():
    # shuffle both tables' rows (seeded) and add a column 'extra': same panel pairs, same
    # data_fingerprint for both parts, same validation headline

def test_ita_reads_stay_out_of_another_runs_log_records():
    # one ita_text validation read and one sealed opening in a log; decision.sweep._log_records
    # for run id 'stage7/x' returns only that run's records (an outcome read logged with it)

def test_from_files_reads_the_tables_and_the_bundle_codebook():
    # the generated_bundle fixture (tests/fixtures/supervision.py), loaded as
    # test_outcome_panel.py's from_bundle test does. Its six-digit codes are 111111, 111112,
    # 111113, 222222 and 333333. Write ita_tables(replace_codes=...) with p04's basis rows on
    # 111111 and 111112 and every other headline pair's single basis row on 111111.
    assert panel.candidates == ('111111', '111112', '111113', '222222', '333333')
```

- [x] **Step 2: Run them to see them fail.** Run: `uv run pytest tests/unit/test_ita_panel.py -q`.
  Expected: FAIL with `ImportError` on `ItaPanel`.

- [x] **Step 3: Implement `ItaPanel`** under a "Panel" divider in `panels/ita.py`.
  > Deviation: `ItaPanel.pairs` is public, a present `role=None` is refused, and after the final review the result carries `data_fingerprint`.

- [x] **Step 4: Run the tests.** Run: `uv run pytest tests/unit/test_ita_panel.py
  tests/unit/test_ita_inputs.py -q`. Expected: PASS.

- [x] **Step 5: Commit.**

```bash
git add src/naics_embedder/panels/ita.py tests/unit/test_ita_panel.py
git commit -m "feat(panels): add ItaPanel with logged reads and one logged sealed opening"
```

---

### Task 6: Reference choice and the beat-by-δ verdict

**Files:**
- Create: `src/naics_embedder/decision/text_to_code.py`
- Test: `tests/unit/test_decision_text_to_code.py`

**Interfaces:**
- Consumes: `ItaResult` (Task 4); `PanelItems`, `unit_draws`, `seed_draws`,
  `replicate_statistics`, `point_statistic`, `percentile_interval` from
  `decision/resampling.py`; `superiority_level` from `decision/rule.py`.
- Produces:
  - `LEXICAL_STUB = 'lexical-stub'`, `EUCLIDEAN_D16 = 'euclidean-d16'`,
    `REFERENCE_ARMS = (LEXICAL_STUB, EUCLIDEAN_D16)`
  - `headline_values(result: ItaResult) -> pl.DataFrame`: `unit` = `description_key`, `item` =
    `pair_id`, `value` = `reciprocal_rank`, non-copied pairs only.
  - `arm_replicates(items: PanelItems, arm: str, seeds: Sequence[ItaResult], *, replicates: int,
    bootstrap_seed: int) -> Tuple[float, np.ndarray]`: the point MRR and each replicate's, with
    unit draws on stream `'ita_text'` and seed draws on `(ita_text, arm)`.
  - `choose_reference(mrr: Mapping[str, float]) -> str`
  - `@dataclass(frozen=True) class ItaVerdict`: `candidate`, `reference: str`, `delta: float`,
    `interval: Tuple[float, float]`, `level: float`, `margin: float`, `beats: bool`
  - `beat_by_margin(candidate: str, reference: str, delta: float, replicates: np.ndarray,
    margin: float) -> ItaVerdict`: `level = superiority_level(1)` (0.95), `interval =
    percentile_interval(replicates, level)`, `beats = interval[0] > margin` (U40).
  - `compare_to_reference(candidate: str, candidate_seeds: Sequence[ItaResult], reference: str,
    reference_seeds: Sequence[ItaResult], margin: float, *, replicates: int, bootstrap_seed: int)
    -> ItaVerdict`: Δ = candidate − reference on paired draws; items from the reference's first
    seed, every seed of both arms checked by `PanelItems.values`.

`choose_reference` raises `ValueError` unless the keys are exactly `REFERENCE_ARMS` and both
values are finite. The higher MRR wins; a tie goes to `lexical-stub`.

- [x] **Step 1: Write the failing tests:**

```python
def test_the_reference_is_the_higher_mrr_and_a_tie_goes_to_the_stub():
    assert choose_reference({'lexical-stub': 0.30, 'euclidean-d16': 0.42}) == 'euclidean-d16'
    assert choose_reference({'lexical-stub': 0.42, 'euclidean-d16': 0.42}) == 'lexical-stub'

def test_choose_reference_refuses_other_arms_or_a_nan(): ...

def test_a_superior_candidate_that_does_not_beat_by_delta_does_not_go_forward():
    replicates = np.linspace(0.02, 0.04, 1001)
    assert beat_by_margin('c', 'lexical-stub', 0.03, replicates, 0.01).beats is True
    verdict = beat_by_margin('c', 'lexical-stub', 0.03, replicates, 0.025)
    assert verdict.interval[0] > 0 and verdict.beats is False and verdict.level == 0.95

def test_the_point_statistic_is_the_headline_mrr():
    # one fixture validation result: point_statistic over every unit == result.headline['mrr']

def test_compare_to_reference_is_reproducible_and_paired():
    # a stub reference (1 seed) and a 3-seed candidate built from the fixture with other ASSIGNED
    # maps: two calls give equal verdicts; delta == candidate mean MRR − reference MRR

def test_arms_scored_on_other_pairs_are_refused():
    # a candidate seed missing p03 → ValueError from PanelItems.values
```

- [x] **Step 2: Run them to see them fail.** Run:
  `uv run pytest tests/unit/test_decision_text_to_code.py -q`. Expected: FAIL with
  `ModuleNotFoundError`.

- [x] **Step 3: Implement the module's "Reference and verdict" section.**
  > Deviation: an arm with no seed is refused, and after the final review `compare_to_reference` refuses seeds read on another table build.

- [x] **Step 4: Run the tests.** Expected: PASS.

- [x] **Step 5: Commit.**

```bash
git add src/naics_embedder/decision/text_to_code.py tests/unit/test_decision_text_to_code.py
git commit -m "feat(decision): add text→code's reference choice and beat-by-delta verdict"
```

---

### Task 7: The margin record and margin-first check

**Files:**
- Modify: `src/naics_embedder/decision/text_to_code.py` ("Margin" section)
- Test: `tests/unit/test_decision_text_to_code.py`

**Interfaces:**
- Consumes: `ItaPanel`, `ItaPart` (Tasks 1, 5); `choose_reference` (Task 6); `read_time` from
  `decision/decide.py`.
- Produces:
  - `class ItaMarginRecord(BaseModel)` (`extra='forbid'`, frozen, `allow_inf_nan=False`):
    `kind: Literal['ita_margin'] = 'ita_margin'`, `name: str`, `r: float` (> 0), `run_id: str`,
    `reference_mrr: Dict[str, float]`, `reference: str`, `margin: float`,
    `split_fingerprint: str`, `data_fingerprint: str`, `log_records: List[Dict[str, Any]]`,
    `fixed_at: datetime`.
  - `fix_ita_margin(panel: ItaPanel, reference_arms: Mapping[str, Sequence[Tuple[Optional[int],
    QueryCodeEncoder]]], r: float, name: str, *, purpose: str) -> ItaMarginRecord`
  - `check_ita_margin_first(margin: ItaMarginRecord, log: SelectionLog) -> None`

`fix_ita_margin` refuses, before any read, an `r` that is not positive and finite and arms whose
keys are not exactly `REFERENCE_ARMS`. It reads each arm's seeds on validation under a fresh
`run_id = f'ita-margin/{uuid4().hex}'`, detail `{'arm', 'seed' (when not None), 'role':
'reference', 'run'}`. Each arm's MRR is the mean of its seeds' headline MRRs; `margin = r ×
reference_mrr[reference]` (§4.5). `log_records` are the log's records with that run id.
`check_ita_margin_first` raises `ValueError` when `fixed_at` names no timezone; when one of the
record's own reads is later than `fixed_at`; or when the log holds an `ita_text` validation
`read` whose `detail.role` is not `reference` (a missing role included) at a time before
`fixed_at`, naming the read's purpose and time.

- [x] **Step 1: Write the failing tests:**

```python
def test_the_margin_is_r_times_the_chosen_references_validation_mrr():
    # stub as lexical-stub [(None, stub)]; a better stub as euclidean-d16 [(1, a), (2, b)]
    record = fix_ita_margin(panel, arms, 0.05, 'ita-margins', purpose='fix the text→code margin')
    assert record.reference == choose_reference(record.reference_mrr)
    assert record.margin == pytest.approx(0.05 * record.reference_mrr[record.reference])
    assert record.split_fingerprint == panel.fingerprint
    assert {r['detail']['role'] for r in record.log_records} == {'reference'}
    assert len(record.log_records) == 3

def test_fix_ita_margin_refuses_a_bad_r_or_other_arms_before_reading():  # log absent afterwards

def test_a_margin_after_a_candidate_read_is_refused():
    # a candidate-role validation read, then fix_ita_margin → check raises naming the read

def test_a_margin_after_a_read_with_no_role_is_refused(): ...

def test_earlier_reference_reads_and_later_candidate_reads_pass():
    # a Stage-4-style reference read, fix_ita_margin, then a candidate read → check passes

def test_a_naive_fixed_at_or_a_read_after_fixed_at_is_refused():
    # model_copy(update={'fixed_at': naive}) and fixed_at moved before its own reads → ValueError

def test_the_record_round_trips_through_json(): ...
```

- [x] **Step 2: Run them to see them fail.** Expected: FAIL with `ImportError` on
  `fix_ita_margin`.

- [x] **Step 3: Implement the "Margin" section.**
  > Deviation: owner gate added refusals of repeated seeds, `check_ita_margin` (record self-check) and reference reads of a non-reference arm.

- [x] **Step 4: Run the tests.** Run:
  `uv run pytest tests/unit/test_decision_text_to_code.py -q`. Expected: PASS.

- [x] **Step 5: Commit.**

```bash
git add src/naics_embedder/decision/text_to_code.py tests/unit/test_decision_text_to_code.py
git commit -m "feat(decision): fix text→code's margin from reference reads, checked margin-first"
```

---

### Task 8: Docs, guide and the final gate

**Files:**
- Create: `docs/api/ita_panel.md`
- Modify: `docs/.nav.yml`, `docs/api/decision.md`, `CLAUDE.md`

- [x] **Step 1: Write `docs/api/ita_panel.md`.** A one-paragraph intro (judged on ITA
  employer-written descriptions, §1; the outcome panel's validation split is reported beside it
  and decides nothing, and its test split stays sealed, §4.6), §5's role table copied as written,
  then `:::` blocks for `naics_embedder.panels.ita_split`, `naics_embedder.panels.ita` and
  `naics_embedder.panels.ita_scoring`. Add `- ITA Panel: api/ita_panel.md` after the Outcome
  Panel entry in `docs/.nav.yml`. Add a "Text→code margin" section with
  `::: naics_embedder.decision.text_to_code` to `docs/api/decision.md`.

- [x] **Step 2: Update `CLAUDE.md`.** Add the four modules to the directory tree and one
  sentence each under "Monitor, Epoch Summary and Selection" (the ITA panel and its sealed part)
  and "Testing and Validation" (the five test files). Recount: `find src -name '*.py' | wc -l`,
  `ls tests/unit/test_*.py | wc -l`, and `uv run pytest --collect-only -q -q | tail -1`; write
  those numbers where `CLAUDE.md` states them (at `4fa2857` the first two give 124 and 107). Update
  `tests/README.md` too where it states the same counts or lists test files.
  > Deviation: `README.md` and `tests/README.md` counts also updated, and `ItaPanelSettings` documents its fields as `Attributes:` for the strict build.

- [x] **Step 3: Check the untouched seams and the lock.** Run:

```bash
git diff --exit-code 4fa2857 -- src/naics_embedder/panels/outcome.py src/naics_embedder/panels/decoding.py src/naics_embedder/panels/selection_log.py src/naics_embedder/decision/rule.py src/naics_embedder/decision/resampling.py src/naics_embedder/decision/records.py src/naics_embedder/decision/decide.py src/naics_embedder/decision/economic.py uv.lock; echo exit=$?
```

Expected: `exit=0`.

- [x] **Step 4: Run the gates.** Run each, and each must pass:

```bash
uv run pytest -n auto -q
uv run ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run mkdocs build --strict
```

Expected: no failures; Ruff `All checks passed!`; the style gate exits 0; MkDocs builds with no
warning. Check the rendered `site/api/ita_panel/index.html` has the three modules' headings.

- [x] **Step 5: Confirm no real pair was read.** Run:
  `grep -rn "embedding-data\|ita_text_pairs.parquet" src/ tests/ | grep -v "tests/fixtures/ita.py"`.
  Expected: only docstrings and `write_ita_tables`' file names, no absolute path.

- [x] **Step 6: Commit.**

```bash
git add docs/api/ita_panel.md docs/.nav.yml docs/api/decision.md CLAUDE.md
git commit -m "docs: add the ITA panel and text→code margin API pages"
```

---

## Spec coverage

| Spec item | Task |
|---|---|
| §4.1 inputs, node ids, refusals, query text, candidates, lineage | 2, 3, 5 |
| §4.2 split, config, stability, split fingerprint | 1 |
| §4.3 rank, per-pair metrics, headline, copied, phrasing, counts, result shape | 3, 4 |
| §4.4 reads, data fingerprint, sealed opening, encoders | 5 |
| §4.5 reference, choice, margin, margin-first, test, intervals | 6, 7 |
| §4.6 and §5 roles | 8 (API page) |
| §6 constraints | Global Constraints, 8 |
| §9 verification items 1–11 | 1; 4; 3; 2 and 5; 7; 5; 6; 8 step 3; 8 step 1; 8 step 5; 8 step 4 |

## Handoff to later stages

- **Stage 4 (reference read and gap).** It consumes the panel as built:
  - `ItaPanel.from_files(pairs_parquet, pair_codes_parquet, bundle, settings_path, log_path)`
    (`panels/ita.py`), with settings `conf/data/ita_panel.yaml` (salt fixed at plan 17).
  - `score_logged(encoder, 'validation', purpose, detail)`, with detail `arm`, `seed`, `role`
    (`reference`) and a fresh `run`. A `role` of `None` or another word is refused before logging.
  - `ItaResult` (`panels/ita_scoring.py`) holds `headline`, `copied`, `phrasing`, `counts` and
    `per_code`, and its `data_fingerprint` names the table build. All report means use `math.fsum`.
  - Traps: `panel_pairs` refuses any disagreement with spec §4.1, also nulls in the read columns
    and a repeated `pair_id` anywhere in the pair table. A real Stage 10 build that fails these
    refuses at construction and logs nothing.
- **Stage 5 (candidates and decision).** It consumes the margin rule as built in
  `decision/text_to_code.py`:
  - `fix_ita_margin` refuses repeated seeds and seedless arms before any read.
  - `check_ita_margin` checks the record's own consistency, and `check_ita_margin_first` calls it.
    The first-read check also counts a reference-role read of a non-reference arm as a candidate
    read.
  - `compare_to_reference` refuses seeds read on another table build, but takes δ as a float.
  - Deferred into this stage, in `specs/deferred_items.md` § 17-text-to-code-ita-panel:
    - the δ build check;
    - margin-first across several selection logs;
    - the permanent block after an early candidate read;
    - checkpoint fingerprints in the margin reads;
    - two self-check gaps.
