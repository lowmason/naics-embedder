# Economic Candidates Implementation Plan

**Status: COMPLETE (2026-11-15)** — executed via subagent-driven-development; deferred items in specs/deferred_items.md

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 1 — on plan completion, tick the stage
> and re-validate later stages against what shipped.

**Goal:** Build the economic artifact's 18 candidates and their frozen list, a command that reads
one candidate on the growth validation splits (with a dry run), and the economic decision over
deterministic and trained candidates. Test everything on fixtures, then build the nine real
deterministic tables and dry-run all 18 real candidates without reading a panel.

**Architecture:** Four layers, each with its own tests.

- **Builders (`tree_model/`).** Three deterministic builders over the codebook and the shared
  text-only table: the text-only table reduced by PCA, classical scaling of the tree metric D*,
  and lineage smoothing of the text-only vectors. Each writes a table in the export form with a
  fixed-entry provenance beside it.
- **The frozen list and the provenance rule (`decision/candidates.py`, `decision/provenance.py`).**
  `conf/data/economic_candidates.yaml` names the 18 candidates and pins their inputs, the trained
  candidates' Stage 7/8 records and, after Task 10, the built tables. The rule accepts a
  provenance only from the export or one of the three builders, naming only text and tree inputs.
- **Records, checks and the decision (`decision/records.py`, `decision/economic.py`).** An
  economic run may carry a trained seed's selection evidence, copied from its imported Stage 7/8
  seed run. The checks accept one seedless run or at least 5 seeded runs. The decision takes the
  frozen list, refuses any other candidate set, and records the propagation comparisons. Stored
  predictions are bound to their table, and stored scores are checked against a rescore.
- **Commands (`cli/commands/tools.py`).** `tools economic-build` writes the deterministic tables.
  `tools economic-read` runs every check, then reads one candidate, or stops after the checks
  under `--dry-run`. `tools decide` passes the frozen list to the economic decision.

**Tech Stack:** Python 3.10+ (locked env 3.12), numpy `eigh`, scikit-learn PCA through
`panels.text_only.pca_reduce`, Polars, Pydantic v2, PyYAML, Typer, pytest. No new dependency.

**Spec:** `specs/economic-candidates.md` (APPROVED 2026-11-14, U35; content at `068bc39`, merged
by PR #158 as `a1246f8`). Read it with this plan, and plan 14's Handoff
(`specs/plans/completed/14-economic-decision-tooling.md:6483`).

## Global Constraints

Every task's requirements include this section. §n cites `specs/economic-candidates.md`. MAIN is
the main checkout, `/Users/lowell/Projects/naics-embedder`. STORE is `~/naics-artifacts`.

- **No real growth read, and no decision.** No task reads a real growth panel, a QCEW file or a
  real outcome panel. Tests use fixtures under `tmp_path` (§2.2, §6 "Nothing real was read").
- **The margins stand.** Never run `tools growth-margins`. Never write, move or rewrite
  `STORE/records/economic/growth_margins.json` (sha256 `4f86a0b2f3bde358…`). Task 10's dry run
  reads it for its own checks only (§2.2, §5).
- **`uv.lock` does not change.** Its sha256 stays
  `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948`. If a change seems needed,
  stop and ask the user (§4.8).
- **The reference's code path does not change.** `src/naics_embedder/panels/regressor.py`,
  `panels/qcew_rows.py`, `panels/regressor_splits.py` and `panels/ridge.py` stay identical to
  `b4f8c9d` (§4.8). The predictions binding lives in `decision/economic.py`.
- **D8 unchanged.** `tests/unit/test_decision.py`, `test_decision_sweep.py`,
  `test_decision_store.py`, `test_decision_rule.py`, `test_decision_resampling.py` and
  `test_decision_scores.py` pass with no edit. Stage 7/8-shaped `ArmRecord`s load and dump
  unchanged (§4.3, §6 "Stage 8 unchanged").
- **Text and tree only.** A candidate reads only the codebook's text and its tree. No builder or
  test fixture reads a QCEW file, a panel or the selection log (§4.5; split Req 23).
- **No `/Users/` path in a tracked config or source file.** The frozen list names record paths
  relative to the store root and run directories relative to a runs root (§4.1). This plan and
  its Task 10 commands name MAIN's paths, as plan 15 did.
- **The sealed splits stay closed.** No outcome test split or growth outer set opens.
- **Style.** Single quotes, YAPF layout, semantic section dividers, one blank line after imports.
  Format touched Python files with `./scripts/format_code.sh <files>`. Never run `ruff format`.
- **Leave alone.** The roadmap (Stage 1 is ticked at completion), the methodology branch
  `claude/naics-artifacts-methodology-985a4c` and its worktree, and `specs/deferred_items.md`
  until the completion protocol (ask first if other sessions are running).

## Review Focus

The five inputs most likely to bite a person using this software, with the test that pins each:

1. **A dry run given the real `--log`.** It must load no panel and never create or append to the
   log. Pinned by `test_economic_read_dry_run_loads_no_panel_and_leaves_the_log_untouched`
   (Task 8).
2. **A table rebuilt with other bytes, or a stored table edited after its pin.** The read refuses
   it before any panel loads, naming the pin. Pinned by
   `test_economic_read_refuses_a_table_that_differs_from_its_pin_before_loading_the_panel`
   (Task 8).
3. **The wrong shared text-only table** (for example plan 8's, which has a valid provenance).
   Build and read both refuse it before writing or reading anything. Pinned by
   `test_economic_build_refuses_a_text_only_table_other_than_the_lists` and
   `test_economic_read_refuses_a_text_only_table_other_than_the_lists` (Task 8).
4. **A partial store copy that lacks one seed's checkpoint object.** The dry run refuses the
   trained candidate and names the missing file, before any panel loads. Pinned by
   `test_economic_read_refuses_a_trained_candidate_whose_store_lacks_a_checkpoint` (Task 8).
5. **An existing `--output` record.** The read refuses before any check that hashes files, and
   reads nothing. Pinned by `test_economic_read_refuses_an_existing_record_before_any_check`
   (Task 8).

## Decisions

Owner decisions after the spec's approval:

- **O1** (owner, plan review, 2026-11-14): "Approve the plan": the plan as written, P1–P8
  included, with execution by subagent-driven-development in a new session. Applied in: the
  whole plan.

The plan's own choices, approved by O1:

- **P1** (plan, 2026-11-14): the real-data steps run the branch's code from its worktree with
  `uv run --locked`, after a check that `naics_embedder.__file__` is under the worktree's `src/`.
  They pass absolute paths into MAIN for the codebook, the shared text-only table, the runs root
  and the log. The worktree's own `uv.lock` has the pinned sha256 (brief, Defaults 1). Applied
  in: Task 10.
- **P2** (plan, 2026-11-14): `tools economic-build` builds the deterministic tables, so the real
  build is a reproducible command, not an ad-hoc script (§8 leaves the options open). Applied in:
  Task 8, Task 10.
- **P3** (plan, 2026-11-14): `tools economic-read` takes `--runs-root` (the directory holding the
  `stage7-*`/`stage8-*` run directories) and `--text-only` (the shared table, with its provenance
  beside it). In the store, the table and its provenance sit in different object directories, so
  `put_text_only` cannot read them from there. Applied in: Task 8.
- **P4** (plan, 2026-11-14): the binding is a constant `table_fingerprint` column, the run's
  table's `matrix_fingerprint`, added to the predictions in `decision/economic.py` before they
  are stored. The pairing compares only the margin read's own columns (§4.8). Applied in: Task 5.
- **P5** (plan, 2026-11-14): a trained economic arm names its source record (`source`, an
  `ArtifactRef` that the store verifies). Its check runs `check_arm` on the source and requires
  each run's evidence to equal the source seed run's, so `_check_monitor_records` is reused
  unchanged. Verified 2026-11-14, read-only: today's `check_arm` passes on all nine real source
  records, every seed names its training run, and each names text-only table `f4fb4f574f88…`.
  The margin record passes `check_relative_margins`, and its stored scores equal a rescore
  (8,547 rows, maximum relative difference 0.0). Applied in: Tasks 1, 6.
- **P6** (plan, 2026-11-14): the propagation pairs come from the list's kinds: each smoothed
  candidate and the one text-only candidate at its dimension. They are never parsed from names.
  Applied in: Tasks 3, 7.
- **P7** (plan, 2026-11-14): the tree candidate's spec has all five D9 fields null. It is checked
  only against the list's shared text-only pin, and D8's `check_text_only` stays unchanged.
  Applied in: Tasks 1, 3, 5.
- **P8** (plan, 2026-11-14): the classical-scaling builder treats a d-th eigenvalue at most
  `1e-6 × λ1` as non-positive, because a zero eigenvalue rounds to ±1e-16. Applied in: Task 2.

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/naics_embedder/tree_model/__init__.py` | Package docstring: the deterministic economic builders (ADR 0001) |
| `src/naics_embedder/tree_model/scaling.py` | `classical_scaling` of a distance matrix, sign rule, cut refusals |
| `src/naics_embedder/tree_model/smoothing.py` | Lineage weights and lineage smoothing |
| `src/naics_embedder/tree_model/builders.py` | The three builders: tables in the export form and fixed provenance |
| `src/naics_embedder/decision/candidates.py` | The frozen list model and loader, table paths, specs |
| `src/naics_embedder/decision/provenance.py` | §4.5's rule: four builders, fixed entries, pinned inputs |
| `src/naics_embedder/decision/trained.py` | A trained candidate's import from its Stage 7/8 record and run directories |
| `src/naics_embedder/decision/records.py` | Backbone-less `ArmSpec`; trained evidence on `EconomicRun`; record fields |
| `src/naics_embedder/decision/decide.py` | `check_margins_first(..., monitor_reads=)` |
| `src/naics_embedder/decision/economic.py` | Binding, stored scores, both arm kinds, trained read, list-bound decision |
| `src/naics_embedder/cli/commands/tools.py` | `economic-build`, `economic-read`, `decide --candidates` |
| `conf/data/economic_candidates.yaml` | The frozen 18-candidate list |
| `tests/fixtures/economic.py` | Codebook writer, builder-shaped `write_candidate`, list writer, trained source, `scaled_arm` |
| `tests/fixtures/decision.py` | `synthetic_arm(..., codes=CODES)` |
| `tests/unit/test_tree_model.py` | Builders |
| `tests/unit/test_economic_candidates.py` | The list, specs, provenance rule, trained import |
| `tests/unit/test_decision_economic.py` | Records, checks, reads and decision on the fixture growth panel |
| `tests/unit/test_cli_commands.py` | The three commands |
| `docs/api/tree_model.md`, `docs/.nav.yml`, `docs/api/decision.md`, `docs/usage.md`, `CLAUDE.md` | Docs |

---

### Task 1: Records — the backbone-less spec and trained evidence

**Files:**
- Modify: `src/naics_embedder/decision/records.py:81-102` (`ArmSpec`), `:398-445` (`EconomicRun`,
  `EconomicArmRecord`), and the module docstring's economic arm bullet
- Test: `tests/unit/test_decision_economic.py` (new section "Records"; rewrite one test)

**Interfaces:**
- Produces:
  - `ArmSpec.backbone: Optional[str]`, `ArmSpec.descriptions_sha256: Optional[str]`,
    `ArmSpec.max_length: Optional[int] = Field(ge=1)`. All three are still required keys with no
    default, so the JSON shape is unchanged. The `model_validator(mode='after')` accepts either
    all five D9 fields null (a candidate that reads no backbone), or `backbone`,
    `descriptions_sha256` and `max_length` all set. Anything else raises `ValueError`, whose
    message contains `reads its backbone or no backbone`.
  - `ArmSpec.reads_backbone -> bool` (property): `self.backbone is not None`.
  - `EconomicRun` gains, after `log_records`: `checkpoint: Optional[ArtifactRef] = None`,
    `training_run: Optional[str] = Field(default=None, min_length=1, pattern=r'\S')`,
    `checkpoint_epoch: Optional[StrictInt] = Field(default=None, ge=0)`,
    `monitor_records: List[Dict[str, Any]] = Field(default_factory=list)`. Docstring: a trained
    candidate's run carries its seed and these four, copied from its Stage 7/8 seed run (§4.4).
  - `EconomicArmRecord.source: Optional[ArtifactRef] = None`, after `runs`: the imported Stage
    7/8 arm record of a trained candidate, by its path under the store root.

- [x] **Step 1: Write the failing tests**

```python
def test_a_stage_8_record_loads_and_dumps_unchanged(tmp_path):
    '''§4.3: ArmSpec's change keeps Stage 8's records loading and validating unchanged.'''
    store = ArtifactStore(tmp_path / 'store')
    arm = synthetic_arm(store, tmp_path, spec('euclidean-d16', dimension=16,
                        geometry='euclidean'), {}, monitor_mrrs=(0.1, 0.3, 0.2))
    text = arm.model_dump_json(indent=2)
    assert ArmRecord.model_validate_json(text).model_dump_json(indent=2) == text
    check_arm(ArmRecord.model_validate_json(text), store, 5)

NO_BACKBONE = dict.fromkeys(
    ('backbone', 'backbone_revision', 'descriptions_sha256', 'summaries_sha256', 'max_length'))

def test_a_spec_reads_its_backbone_or_no_backbone():
    tree = spec('tree-d3', dimension=3, geometry='euclidean', **NO_BACKBONE)
    assert not tree.reads_backbone
    assert spec('text-only-d3').reads_backbone
    for partial in ({'backbone': None}, {'max_length': None}, {'descriptions_sha256': None}):
        with pytest.raises(ValidationError, match='reads its backbone or no backbone'):
            spec('partial', **partial)

def test_a_trained_run_with_its_selection_evidence_loads(arm, tmp_path):
    '''Replaces test_an_imported_run_with_a_checkpoint_or_monitor_reads_does_not_load: the
    fields exist now, and check_economic_arm refuses them on a deterministic run (Task 5).'''
    data = json.loads(arm.model_dump_json())
    data['runs'][0].update(seed=1, checkpoint={'path': 'objects/a', 'sha256': 'a' * 64,
                           'bytes': 1}, training_run='t-1', checkpoint_epoch=2,
                           monitor_records=[{'mrr': 0.5, 'read': {}}])
    data['source'] = {'path': 'records/x.json', 'sha256': 'b' * 64, 'bytes': 1}
    (tmp_path / 'trained.json').write_text(json.dumps(data))
    loaded = read_record(tmp_path / 'trained.json', EconomicArmRecord)
    assert (loaded.runs[0].training_run, loaded.runs[0].checkpoint_epoch) == ('t-1', 2)
    assert loaded.source.path == 'records/x.json'

@pytest.mark.parametrize('field, value', [('checkpoint_epoch', 1.0), ('checkpoint_epoch', '1'),
                         ('seed', 1.0), ('training_run', '  ')])
def test_imported_evidence_identities_are_strict(arm, tmp_path, field, value):
    data = json.loads(arm.model_dump_json())
    data['runs'][0][field] = value
    (tmp_path / 'x.json').write_text(json.dumps(data))
    with pytest.raises(ValidationError, match=f'runs\\.0\\.{field}'):
        read_record(tmp_path / 'x.json', EconomicArmRecord)
```

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_decision_economic.py -k "stage_8_record or reads_its_backbone or selection_evidence_loads or evidence_identities" -q`
Expected: FAIL. The partial specs load, the evidence fields are "Extra inputs", and the tree spec
fails validation.

- [x] **Step 3: Implement the record changes listed under Interfaces**

- [x] **Step 4: Delete `test_an_imported_run_with_a_checkpoint_or_monitor_reads_does_not_load`**

  Its claim moves to Task 5's `test_a_deterministic_run_with_selection_evidence_is_refused`.
  Then run `uv run pytest tests/unit/test_decision*.py -q`.
  Expected: all PASS, the D8 files with no edit.

- [x] **Step 5: Commit**

```bash
git add src/naics_embedder/decision/records.py tests/unit/test_decision_economic.py
git commit -m "feat(decision): allow backbone-less specs and trained evidence in economic runs"
```

---

### Task 2: The deterministic builders (`tree_model/`)

**Files:**
- Create: `src/naics_embedder/tree_model/__init__.py`, `scaling.py`, `smoothing.py`, `builders.py`
- Modify: `tests/fixtures/economic.py` (add `write_codebook`)
- Test: `tests/unit/test_tree_model.py`

**Interfaces:**
- Consumes: `utils.naics_hierarchy.tree_distance_matrix`, `code_lineage`;
  `panels.text_only.pca_reduce`, `text_only_matrix`, `provenance_path`;
  `panels.regressor.table_fingerprint`; `supervision.artifacts.sha256_file`.
- Produces:
  - `scaling.CUT_GAP = 1e-6`.
  - `scaling.classical_scaling(distances: np.ndarray, dimension: int) -> Tuple[np.ndarray,
    np.ndarray]`. It returns the (n, d) float64 coordinates and the d + 1 leading eigenvalues,
    descending. B = −½ J (D ∘ D) J, `numpy.linalg.eigh`. Coordinates are the d leading
    eigenvectors times √λ. In each column, the entry of largest magnitude (the first, under
    `np.argmax(np.abs(·))`) is positive. `ValueError` if not 1 ≤ d < n; if λ_d ≤ `CUT_GAP` × λ₁
    (message contains `is not positive`); or if (λ_d − λ_{d+1}) / λ_d < `CUT_GAP` (message
    contains `ties`).
  - `smoothing.lineage_weights(depth: int) -> np.ndarray`: the weights from the sector down to
    the code. The code has weight 1, its parent ½, and so on, normalized to sum 1.
  - `smoothing.smooth_over_lineage(codes: Sequence[str], vectors: np.ndarray) -> np.ndarray`:
    each row becomes the weighted mean of its lineage's rows. `ValueError` naming a code whose
    ancestor is not among `codes`.
  - `builders.KINDS = ('text-only', 'tree', 'smoothed')`.
  - `builders.BUILDER_INPUTS: Dict[str, Tuple[str, ...]]`: `{'text-only': ('codebook',
    'text_only'), 'tree': ('codebook',), 'smoothed': ('codebook', 'text_only')}`.
  - `builders.METHODS: Dict[str, str]`: each kind's `method` entry, one sentence naming §4.2's
    method (for `smoothed`, the halving weights).
  - `builders.BUILDER_ENTRIES: Dict[str, FrozenSet[str]]`: every kind writes `builder`,
    `method`, `dimension`, `codes`, `inputs`, `table_sha256`, `matrix_fingerprint` and
    `library_versions` (numpy, scikit-learn, polars). `tree` also writes `eigenvalues` (the
    d + 1 leading). No timestamp, so two builds write the same provenance bytes.
  - `builders.build_candidate(kind: str, dimension: int, codebook: Path, output: Path, *,
    text_only: Optional[Path] = None) -> Path`. It writes the table and its provenance beside it
    (`provenance_path`, `json.dumps(..., indent=2, sort_keys=True)`) and returns the table's
    path. The table has `code` (Utf8), `index` (Int64, the codebook's `code_id`), `level` (Int64,
    `len(code)`) and `e0 … e{d-1}` (Float64), in `code_id` order. Each input is
    `{'path': str(path), 'sha256': ...}`.
    - `text-only`: `text_only_matrix` in the text-only table's row order, then `pca_reduce` to
      d, then reordered to the codebook. This is exactly what `ArmTables.from_tables` computes,
      so a read's two predictions agree bit for bit (§4.2, §6 "D9 consistency").
    - `tree`: `classical_scaling(tree_distance_matrix(codes).astype(np.float64), d)`.
    - `smoothed`: the text-only vectors in codebook order, then `smooth_over_lineage`, then
      `pca_reduce` to d.
    - `ValueError`: unknown kind; `text_only` missing for a text kind or given for `tree`; the
      text-only table's codes differ from the codebook's; or as the steps above.
      `FileExistsError` if `output` exists. Nothing is written on any refusal.
  - `tests/fixtures/economic.py`: `write_codebook(path: Path, codes: Sequence[str] = CODEBOOK) ->
    Path`. It writes `code_id` (Int32, 0…) and `code` in the given order, as bundle 301cce28's
    codebook holds them.

- [x] **Step 1: Write the failing tests**

```python
CHAIN = ('11', '111', '1111')           # D* = [[0,1,2],[1,0,1],[2,1,0]]; B's spectrum 2, 0, 0
STAR = ('11', '111', '112', '113')      # spectrum 2, 2, 0, -0.25: not Euclidean

def test_classical_scaling_recovers_a_chain():
    coordinates, eigenvalues = classical_scaling(tree_distance_matrix(CHAIN).astype(float), 1)
    assert np.abs(coordinates[:, 0]) == pytest.approx([1.0, 0.0, 1.0], abs=1e-12)
    assert eigenvalues == pytest.approx([2.0, 0.0], abs=1e-12)

def test_each_column_s_largest_entry_is_positive():
    coordinates, _ = classical_scaling(tree_distance_matrix(CODEBOOK).astype(float), 3)
    for column in coordinates.T:
        assert column[np.argmax(np.abs(column))] > 0

def test_classical_scaling_is_the_rank_d_part_of_the_double_centred_matrix():
    distances = tree_distance_matrix(CODEBOOK).astype(float)
    coordinates, _ = classical_scaling(distances, 3)
    n = len(CODEBOOK)
    centring = np.eye(n) - 1.0 / n
    values, vectors = np.linalg.eigh(-0.5 * centring @ (distances**2) @ centring)
    top = vectors[:, -3:] * values[-3:]
    assert coordinates @ coordinates.T == pytest.approx(top @ vectors[:, -3:].T, abs=1e-9)

@pytest.mark.parametrize('codes, dimension, message', [
    (CODEBOOK, 4, 'ties'),      # λ4 = λ5 = 135.97…, relative gap 4e-16 (verified 2026-11-14)
    (STAR, 1, 'ties'), (STAR, 3, 'is not positive'), (CHAIN, 2, 'is not positive'),
])
def test_an_arbitrary_or_non_positive_cut_is_refused(codes, dimension, message):
    with pytest.raises(ValueError, match=message):
        classical_scaling(tree_distance_matrix(codes).astype(float), dimension)

def test_lineage_weights_halve_up_the_tree_and_sum_to_one():
    assert lineage_weights(3) == pytest.approx([1 / 7, 2 / 7, 4 / 7])
    assert lineage_weights(1) == pytest.approx([1.0])

def test_smoothing_averages_each_code_over_its_lineage():
    smoothed = smooth_over_lineage(CHAIN, np.array([[7.0, 0.0], [0.0, 7.0], [0.0, 0.0]]))
    assert smoothed == pytest.approx(np.array([[7.0, 0.0], [7 / 3, 14 / 3], [1.0, 2.0]]))

def test_a_code_whose_ancestor_is_missing_is_refused():
    with pytest.raises(ValueError, match='1111'):
        smooth_over_lineage(('11', '1111'), np.zeros((2, 2)))
```

  Add the builder tests, with `codebook = write_codebook(tmp_path / 'naics_codebook.parquet')`,
  `text_only = write_text_only(tmp_path / 'text', codes=CODEBOOK)` and d = 3:
  - `test_each_builder_writes_the_export_form_in_codebook_order`: the columns are `['code',
    'index', 'level', 'e0', 'e1', 'e2']`, `code` and `index` equal the codebook's rows, and
    `level == len(code)`. The provenance's keys equal `BUILDER_ENTRIES[kind]`. Its `inputs`
    keys equal `BUILDER_INPUTS[kind]`, each with the input file's `sha256_file`. Its
    `table_sha256` and `matrix_fingerprint` describe the table.
  - `test_each_builder_writes_the_same_bytes_twice`: two builds into two directories give equal
    `sha256_file` for the tables and for the provenance files.
  - `test_the_text_only_candidate_is_reduced_as_a_read_reduces_it`:
    `np.array_equal(coordinates, ArmTables.from_tables(candidate, text_only_frame).text_only)`.
  - `test_the_smoothed_candidate_reduces_lineage_means`: `np.array_equal` with
    `pca_reduce(smooth_over_lineage(codes, vectors_in_codebook_order), 3)`.
  - `test_the_tree_candidate_is_the_classical_scaling_of_d_star`: `np.array_equal` with
    `classical_scaling(...)[0]`. Its `eigenvalues` entry equals the returned four.
  - `test_a_builder_refuses_text_only_inputs_over_other_codes`:
    `write_text_only(..., codes=CODEBOOK[1:])` raises `ValueError`, and no table is written.
  - `test_a_builder_refuses_text_only_input_for_the_tree_and_none_for_text`.
  - `test_a_builder_writes_no_table_over_an_existing_one`: `FileExistsError`, and the old bytes
    stay.

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_tree_model.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'naics_embedder.tree_model'`.

- [x] **Step 3: Implement `tree_model/` as listed under Interfaces**

  The package docstring cites ADR 0001's rule for a method that fits neither `text_model/` nor
  `graph_model/`. `tree_model` imports nothing from `decision/`.

- [x] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/unit/test_tree_model.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/naics_embedder/tree_model tests/unit/test_tree_model.py tests/fixtures/economic.py
git commit -m "feat(tree_model): build the text-only, tree and smoothed economic candidates"
```

---

### Task 3: The frozen candidate list

**Files:**
- Create: `src/naics_embedder/decision/candidates.py`, `conf/data/economic_candidates.yaml`
- Test: `tests/unit/test_economic_candidates.py`

**Interfaces:**
- Consumes: `records.ArmSpec`, `records.Geometry`; `decide.MIN_SEEDS`.
- Produces (`decision/candidates.py`; models use `records._Record`'s config: extra forbidden,
  frozen):
  - `Kind = Literal['text-only', 'tree', 'smoothed', 'trained']`.
  - `CheckpointContract(bundle_id: str, contract_version: str)`.
  - `CandidateEntry(name: str (min 1), kind: Kind, dimension: int ≥ 1, components: int ≥ 1,
    geometry: Geometry, table_sha256: Optional[str] = None, record: Optional[str] = None,
    record_sha256: Optional[str] = None, arm: Optional[str] = None, seeds: List[StrictInt] = [],
    run_directory: Optional[str] = None)`. The sha fields match `^[0-9a-f]{64}$`.
    - A deterministic entry has no `record`, `record_sha256`, `arm`, `seeds` or
      `run_directory`.
    - A trained entry has no `table_sha256`. Its `record` is relative, with no `..` part. It has
      `record_sha256`, `arm`, at least `MIN_SEEDS` distinct seeds, and a `run_directory` that
      contains `{seed}` and no `/`.
  - `CandidateList(codebook_sha256: str, text_only_sha256: str, checkpoint_contract:
    CheckpointContract, candidates: List[CandidateEntry])`. Names are distinct. Every smoothed
    entry has exactly one text-only entry at its dimension.
    - `.entry(name: str) -> CandidateEntry`: `ValueError` matching `is not on the frozen
      candidate list`.
    - `.names -> List[str]` (property, list order).
    - `.propagation_pairs() -> List[Tuple[str, str]]`: `(smoothed, text-only twin)`, in list
      order (P6).
  - `load_candidate_list(path: Union[str, Path]) -> CandidateList` (`yaml.safe_load`, then
    `model_validate`).
  - `candidate_table_path(store_root: Union[str, Path], name: str) -> Path`:
    `<root>/candidates/economic/<name>.parquet` (§4.2).
  - `economic_spec(entry: CandidateEntry, *, text_only: Optional[Mapping[str, Any]] = None,
    source: Optional[ArmSpec] = None) -> ArmSpec`.
    - `tree`: all five D9 fields None, `settings` empty (P7).
    - `text-only` and `smoothed`: the D9 fields from `text_only`, the mapping that
      `decision.store.provenance_fields` returns (its `revision` is the spec's
      `backbone_revision`).
    - `trained`: `source` under the entry's name. `ValueError` if the source's dimension,
      geometry or components differ from the entry's.
  - `ECONOMIC_CANDIDATES = 'conf/data/economic_candidates.yaml'`.

- [x] **Step 1: Write the failing tests**

```python
SPEC_ROWS = [  # §4.1's table, in its order: (name, kind, dimension, components, geometry)
    *[(f'{kind}-d{d}', kind, d, components, 'euclidean')
      for kind, components in (('text-only', 1), ('tree', 1), ('smoothed', 2))
      for d in (8, 16, 32)],
    ('hyperbolic-d16', 'trained', 16, 1, 'hyperbolic'),
    *[(f'euclidean-d{d}', 'trained', d, 1, 'euclidean') for d in (8, 16, 32)],
    *[(f'hyperbolic-d{d}', 'trained', d, 1, 'hyperbolic') for d in (8, 32)],
    *[(f'spherical-d{d}', 'trained', d, 1, 'spherical') for d in (8, 16, 32)],
]
RECORD_SHA256 = {  # STORE/records/…, hashed 2026-11-14
    'hyperbolic-d16': 'c885b9c5dc18b6be03670d0cb5a71db3974917da8f719e0dbb1f1ef5eec2d1a2',
    'euclidean-d8': '103367f0f8f9270567ef2acfff1bd5a870aece94f4e66d75c51390eeba758dbb',
    'euclidean-d16': '3d5a80d5a3ac5466e10b019773fa09a21e2ad0592af32d7b22273fea3e1bab2c',
    'euclidean-d32': 'cf4edd78d3cd195470fa2d5325e791e1e52209c7dbb0249ee2dc39bc713bf2c0',
    'hyperbolic-d8': '53a7b32959f9b7b281b7127772631d2a57573b10762ac5ba05ae164b0fa08ecd',
    'hyperbolic-d32': '927c3fd96fec78b31d456d0fa88d2c1d8b133cc9da0707cc82f63a92aec36016',
    'spherical-d8': 'ce739fa9b73ad2538fc0c973c2f0e64eec453be3f77ee78534748a6b288a87df',
    'spherical-d16': '28fec47082c3fe82e9d2c23b397c4ccf9a6af7cc2df22d5f1531cfa891c95c01',
    'spherical-d32': '194d2ea6a9420cf6c16574891ff21832b5bbae25dd4cbfe8fa8158fe6eeaa679',
}

ROOT = Path(__file__).resolve().parents[2]      # the repository, whatever the working directory

def test_the_committed_list_is_the_specs_eighteen_candidates():
    committed = load_candidate_list(ROOT / ECONOMIC_CANDIDATES)
    assert [(e.name, e.kind, e.dimension, e.components, e.geometry)
            for e in committed.candidates] == SPEC_ROWS
    assert committed.codebook_sha256 == (
        '0f9e61343e673540e1e18ecc5581618be099c5d79531b68153496bf982d6e96d')
    assert committed.text_only_sha256 == (
        'f4fb4f574f880c940ee8cf44c7f86615638e59f113b527d0b86973cd45ba32ff')
    assert committed.checkpoint_contract == CheckpointContract(
        bundle_id='301cce28-539c-42ea-8781-496bbdcf511c', contract_version='stage3-supervision-v2')
    for entry in committed.candidates:
        if entry.kind != 'trained':
            continue
        stage7 = entry.name == 'hyperbolic-d16'
        assert entry.record == ('records/stage7/reference.json' if stage7
                                else f'records/stage8/{entry.name}.json')
        assert entry.arm == ('reference' if stage7 else entry.name)
        assert entry.seeds == list(range(1, 11 if stage7 else 6))
        assert entry.run_directory == ('stage7-reference-s{seed}' if stage7
                                       else f'stage8-{entry.name}-s{{seed}}')
        assert entry.record_sha256 == RECORD_SHA256[entry.name]

def test_the_committed_list_names_no_user_path():
    assert '/Users/' not in (ROOT / ECONOMIC_CANDIDATES).read_text()

def test_the_propagation_pairs_are_each_smoothed_candidate_and_its_text_only_twin():
    assert load_candidate_list(ROOT / ECONOMIC_CANDIDATES).propagation_pairs() == [
        (f'smoothed-d{d}', f'text-only-d{d}') for d in (8, 16, 32)]
```

  Add these, over a small list built in the test:
  - `test_a_list_that_breaks_an_entry_rule_is_refused`, parametrized: a trained entry with 4
    seeds, with repeated seeds, with an absolute `record`, with `../x.json`, with a
    `run_directory` lacking `{seed}`, or with a `table_sha256`; a deterministic entry with
    seeds; a short sha; an unknown kind. Each raises `ValidationError`.
  - `test_a_list_with_repeated_names_or_an_orphan_smoothed_candidate_is_refused`.
  - `test_a_name_off_the_list_is_refused`.
  - `test_each_kind_has_its_spec`: tree has all five D9 fields None. Text-only and smoothed
    take the text-only fields. Trained is the source spec under the entry's name. A source with
    another dimension raises `ValueError`.

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_economic_candidates.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'naics_embedder.decision.candidates'`.

- [x] **Step 3: Write `conf/data/economic_candidates.yaml`**

  Block style, the 18 entries in `SPEC_ROWS` order. Every deterministic entry has
  `table_sha256: null`, and Task 10 pins them. The header and the first entry of each kind:

```yaml
# The economic candidates (specs/economic-candidates.md §4.1; U33), frozen before Stage 3's first
# read: the decision reads exactly these (decision/candidates.py).
#
# - codebook_sha256, text_only_sha256: what every deterministic builder reads, bundle 301cce28's
#   naics_codebook.parquet and the shared MiniLM text-only table (§4.3).
# - checkpoint_contract: the contract a trained table's export provenance must name (§4.5).
# - A trained candidate's record is its Stage 7/8 arm record under the artifact store's root,
#   pinned by its sha256; run_directory is each seed's run directory under the runs root, which
#   holds that seed's export provenance.
# - A deterministic candidate's table_sha256 pins its built table, once Stage 1 has built it.
codebook_sha256: 0f9e61343e673540e1e18ecc5581618be099c5d79531b68153496bf982d6e96d
text_only_sha256: f4fb4f574f880c940ee8cf44c7f86615638e59f113b527d0b86973cd45ba32ff
checkpoint_contract:
  bundle_id: 301cce28-539c-42ea-8781-496bbdcf511c
  contract_version: stage3-supervision-v2
candidates:
  - name: text-only-d8
    kind: text-only
    dimension: 8
    components: 1
    geometry: euclidean
    table_sha256: null
  - name: hyperbolic-d16
    kind: trained
    dimension: 16
    components: 1
    geometry: hyperbolic
    record: records/stage7/reference.json
    record_sha256: c885b9c5dc18b6be03670d0cb5a71db3974917da8f719e0dbb1f1ef5eec2d1a2
    arm: reference
    seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    run_directory: stage7-reference-s{seed}
```

- [x] **Step 4: Implement `decision/candidates.py` as listed under Interfaces (no trained import yet)**

- [x] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_economic_candidates.py -q`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/decision/candidates.py conf/data/economic_candidates.yaml tests/unit/test_economic_candidates.py
git commit -m "feat(decision): freeze the economic candidate list"
```

---

### Task 4: Text and tree only — the provenance rule

**Files:**
- Create: `src/naics_embedder/decision/provenance.py`
- Modify: `tests/fixtures/economic.py` (`write_candidate`, new constants, `fixture_list` and
  `export_provenance`)
- Test: `tests/unit/test_economic_candidates.py`

**Interfaces:**
- Consumes: `tree_model.builders.KINDS`, `BUILDER_ENTRIES`, `BUILDER_INPUTS`, `METHODS`;
  `candidates.CandidateEntry`, `CandidateList`. `decision/provenance.py` imports `candidates`,
  so `candidates.py` never imports `provenance.py` (Task 6's import goes in its own module).
- Produces:
  - `provenance.EXPORT_ENTRIES: FrozenSet[str]`: `backbone`, `checkpoint`, `codes`, `contract`,
    `coordinates`, `descriptions`, `dimension`, `generated_at`, `library_versions`,
    `matrix_fingerprint`, `max_length`, `revision`, `summaries`, `table_sha256`, `tokenizer`.
    These are Stage 7's ten files, written before 48ff710 (verified 2026-11-14 on MAIN's 50
    files: two key sets).
  - `provenance.EXPORT_ENTRIES_WITH_GEOMETRY = EXPORT_ENTRIES | {'geometry'}` (Stage 8's forty).
  - `provenance.check_candidate_provenance(content: Mapping[str, Any], entry: CandidateEntry,
    candidates: CandidateList) -> None`. It raises `ValueError`, prefixed `{entry.name}: `, and
    each message contains `(Req 29: text and tree only)`. It checks in the order below, and each
    message names what it checked (`builder`, `entries`, `inputs`, `codebook`, `text_only`,
    `contract`).
    - With a `builder` entry: the builder is in `KINDS` and equals `entry.kind`. The keys equal
      `BUILDER_ENTRIES[builder]`. `inputs` is an object whose keys equal
      `BUILDER_INPUTS[builder]`, each `{'path': str, 'sha256': str}`.
      `inputs.codebook.sha256 == candidates.codebook_sha256`, and
      `inputs.text_only.sha256 == candidates.text_only_sha256` where present.
    - With no `builder` entry: the provenance is the export's, and `entry.kind` is `trained`.
      The keys are exactly `EXPORT_ENTRIES` or `EXPORT_ENTRIES_WITH_GEOMETRY`.
      `contract.bundle_id` and `contract.contract_version` equal `candidates.checkpoint_contract`.
      `checkpoint` and `descriptions` are each `{'path', 'sha256'}`.
  - `tests/fixtures/economic.py`:
    - `CODEBOOK_SHA256 = 'c' * 64`.
    - `CONTRACT = CheckpointContract(bundle_id='fixture-bundle', contract_version='fixture-v1')`.
    - `CODEBOOK_INPUT = {'path': 'naics_codebook.parquet', 'sha256': CODEBOOK_SHA256}`.
    - `write_candidate(directory, name, table, **entries) -> Path` now writes a `tree` builder's
      provenance. It has exactly `BUILDER_ENTRIES['tree']`: `method` = `METHODS['tree']`,
      `inputs` = `{'codebook': CODEBOOK_INPUT}`, and `eigenvalues` = d + 1 ones. `entries` join
      or replace entries, as before. The keyword `codebook_sha256: str = CODEBOOK_SHA256`
      sets the codebook input's sha256.
    - `fixture_list(entries: Sequence[CandidateEntry], *, text_only_sha256: str,
      codebook_sha256: str = CODEBOOK_SHA256) -> CandidateList`, with `CONTRACT`. A test
      whose candidates come from `build_candidate` passes the written codebook's
      `sha256_file`, which the builder records.
    - `export_provenance(table_path: Path, arm_spec: ArmSpec, checkpoint: ArtifactRef, *,
      geometry: bool = True) -> Dict[str, Any]`: a full export-shaped provenance with
      `EXPORT_ENTRIES` (plus `geometry` unless `geometry=False`), naming `CONTRACT` and the
      checkpoint's sha256.

- [x] **Step 1: Write the failing tests**

  `test_each_builders_provenance_passes_the_rule`, parametrized over the three kinds: build with
  `build_candidate` over `write_codebook` and `write_text_only(codes=CODEBOOK)`, then call
  `check_candidate_provenance(content, entry, fixture_list([entry], text_only_sha256=…))`. The
  list's `codebook_sha256` here is the written codebook's `sha256_file`, so the real builder's
  inputs match.

  The module's fixtures for the tests below:
  - `tree_entry` is `CandidateEntry(name='tree-d3', kind='tree', dimension=3, components=1,
    geometry='euclidean')`.
  - `tree_content` is the JSON that `write_candidate(tmp_path, 'tree-d3',
    coordinate_table(CODEBOOK))` writes.
  - `trained_entry` is a trained row named `euclidean-d3`.
  - `candidates` is `fixture_list([tree_entry, trained_entry], text_only_sha256='a' * 64)`.
  - `table`, `arm_spec` and `checkpoint` are a written table, `spec('euclidean-d3',
    dimension=3, geometry='euclidean')`, and an `ArtifactRef` with sha256 `'b' * 64`.

```python
@pytest.mark.parametrize('geometry', [False, True], ids=['stage-7', 'stage-8'])
def test_both_export_versions_pass_the_rule(tmp_path, geometry):
    content = export_provenance(table, arm_spec, checkpoint, geometry=geometry)
    check_candidate_provenance(content, trained_entry, candidates)

@pytest.mark.parametrize('edit, message', [
    ({'qcew': {'path': 'qcew/2023.annual.singlefile.csv', 'sha256': 'f' * 64}}, 'entries'),
    ({'panel': 'growth_seen'}, 'entries'),
    ({'selection_log': 'logs/selection_log.jsonl'}, 'entries'),
    ({'inputs': {'codebook': CODEBOOK_INPUT, 'qcew': CODEBOOK_INPUT}}, 'inputs'),
    ({'inputs': {'codebook': {'path': 'naics_codebook.parquet', 'sha256': 'f' * 64}}},
     'codebook'),
    ({'builder': 'hgcn'}, 'builder'),
    ({'builder': 'smoothed'}, 'builder'),        # a tree entry's provenance from another builder
])
def test_a_provenance_outside_text_and_tree_is_refused(tree_content, tree_entry, candidates,
                                                       edit, message):
    with pytest.raises(ValueError, match=f'^tree-d3: .*{message}.*Req 29: text and tree only'):
        check_candidate_provenance({**tree_content, **edit}, tree_entry, candidates)
```

  Add:
  - `test_an_export_naming_another_contract_or_an_unknown_entry_is_refused`: another
    `bundle_id`, another `contract_version`, an extra `qcew` key, or a missing `tokenizer`.
  - `test_an_export_provenance_is_refused_for_a_deterministic_entry`.

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_economic_candidates.py -k provenance -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'naics_embedder.decision.provenance'`.

- [x] **Step 3: Implement `decision/provenance.py` and the fixture changes listed under Interfaces**
  > Deviation: the brief's "change no existing assertion" was wrong for one assertion: `test_decision_economic.py` expected `inputs == {'codebook': 'the fixture codebook'}`, and the new `write_candidate` makes it `{'codebook': CODEBOOK_INPUT}` (ledger ruling, Task 4). The export rule stays shape-only for checkpoint, descriptions and summaries, and Task 6's `check_seed_table` enforces their identity.

- [x] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_economic_candidates.py tests/unit/test_tree_model.py -q`
Expected: PASS.
Run: `uv run pytest tests/unit/test_decision*.py tests/unit/test_cli_commands.py -q`
Expected: PASS. No check applies the rule yet, and `write_candidate`'s new entries change no
existing assertion.

- [x] **Step 5: Commit**

```bash
git add src/naics_embedder/decision/provenance.py tests/fixtures/economic.py tests/unit/test_economic_candidates.py
git commit -m "feat(decision): accept a candidate's provenance only from text and tree builders"
```

---

### Task 5: Deterministic candidates bound to the frozen list

**Files:**
- Modify: `src/naics_embedder/decision/economic.py` (`read_deterministic_arm`,
  `check_economic_arm`, `_check_growth_reads`, `check_reference_pairing`, `_reference_rows`,
  `check_relative_margins`, `decide_economic`, and the module docstring's Candidates and
  Decision bullets)
- Modify: `src/naics_embedder/decision/records.py` (`EconomicDecisionRecord.candidates`)
- Modify: `src/naics_embedder/cli/commands/tools.py` (`decide_command`'s `--candidates`)
- Modify: `tests/fixtures/economic.py` (`candidate_spec`, `scaled_arm`, `list_for`,
  `write_candidate_list`)
- Test: `tests/unit/test_decision_economic.py`, `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes: `CandidateList`, `CandidateEntry`, `load_candidate_list`, `economic_spec`,
  `ECONOMIC_CANDIDATES`; `check_candidate_provenance`; `ArmSpec.reads_backbone`.
- Produces:
  - `economic.BINDING_COLUMN = 'table_fingerprint'` (P4).
  - `read_deterministic_arm(...)`: the signature is unchanged. It calls `check_text_only` only
    when `spec.reads_backbone`. Before storing, it adds `pl.lit(stored.matrix_fingerprint)` as
    `BINDING_COLUMN` to the predictions.
  - `check_economic_arm(arm: EconomicArmRecord, store: ArtifactStore, candidates: CandidateList)
    -> None`. The deterministic branch runs these checks in order (Task 6 adds the trained
    branch):
    1. The arm's list row (`candidates.entry`): the record's dimension, components and
       geometry equal the row's (message contains `its frozen-list row`).
    2. The text-only table: `arm.text_only.table.sha256 == candidates.text_only_sha256`
       (message contains `the shared text-only table`). The recorded fields equal the stored
       provenance's. `check_text_only` runs when `spec.reads_backbone`. A tree row requires
       `not reads_backbone`, and a text row requires `reads_backbone`.
    3. The shape: `arm.source is None`, and exactly one run, with `seed`, `checkpoint`,
       `training_run` and `checkpoint_epoch` None and `monitor_records` empty (message contains
       `one seedless run`). Its table's sha256 equals the row's `table_sha256`. An unpinned row
       raises a message containing `pins no table`, and another sha one containing `differs
       from its pin`.
    4. Per run:
       - The references resolve, and the stored provenance equals the recorded one (as today).
       - `check_candidate_provenance` passes.
       - The `BINDING_COLUMN` values of the stored predictions are exactly
         `[run.table.matrix_fingerprint]` (message contains `predictions belong to another
         table`).
       - The statistics' keys equal `GROWTH_PANELS`, and their values match the rescore, both
         as today.
       - The stored scores equal the rescore (`_check_stored_scores`).
       - `_check_growth_reads` runs as today.
  - `_check_stored_scores(name: str, stored: pl.DataFrame, rescored: pl.DataFrame) -> None`. The
    frames must hold the same `(panel, statistic, unit, item, feature_year)` rows, with every
    `value` within `rel_tol=STATISTIC_TOLERANCE`. Otherwise it raises `ValueError` containing
    `stored scores`.
  - `check_relative_margins(margins, store)`: also requires `margins.read.scores` to equal the
    rescore of its predictions (`_check_stored_scores`).
  - `check_reference_pairing(margins, arm, store)`: zero runs raise
    `f'{name}: an economic arm with no runs pairs nothing (Req 24)'`. It compares every run.
    `_reference_rows(predictions, columns)` first selects `columns`, the margin read's
    predictions' own columns, so the binding column never enters the comparison (§4.8).
  - `decide_economic(name, question, arms, margins, store, *, candidates: Union[str, Path],
    replicates, bootstrap_seed)`. Before today's checks, it loads the list and requires the arm
    names to equal `candidate_list.names` as sets. A name off the list raises `ValueError`
    containing `not on the frozen candidate list`. A list candidate with no arm raises one
    containing `missing`. Duplicate names and the reference name stay refused as today. Each arm
    then passes `check_economic_arm(arm, store, candidate_list)`. The resampling is unchanged.
  - `EconomicDecisionRecord.candidates: ArtifactRef`, after `reports`: the list file as stored
    (`store.put(candidates)`), whose sha256 the record carries (§4.4).
  - `tools decide`: `--candidates PATH` (optional). With relative margins, it passes the path,
    or `ECONOMIC_CANDIDATES` when none is given, to `decide_economic`. With D8 margins, a given
    `--candidates` is refused with `applies only to the economic decision`.
  - `tests/fixtures/economic.py`:
    - `candidate_spec(name, **overrides)` now defaults to the tree's spec: all five D9 fields
      None (P7), to match `write_candidate`'s tree provenance.
    - `scaled_arm(arm: EconomicArmRecord, factors: Mapping[str, float], store: ArtifactStore) ->
      EconomicArmRecord`. This is today's `_scaled` from the test module, moved. It carries the
      binding column through and scales every run.
    - `list_for(arms: Sequence[EconomicArmRecord], store: ArtifactStore, *, text_only_sha256:
      str) -> CandidateList`: one row per arm, kind from the provenance's `builder`, and
      `table_sha256` pinned to the run's table. It takes the keyword `codebook_sha256: str =
      CODEBOOK_SHA256`, as `fixture_list` does, and uses `CONTRACT`.
    - `write_candidate_list(path: Path, arms, store, *, text_only_sha256: str,
      codebook_sha256: str = CODEBOOK_SHA256) -> Path`: the same list, dumped as YAML.
- Rewrites by design. Do not change the code back to make these pass:
  - Every `check_economic_arm(record, store)` call in `test_decision_economic.py` becomes
    `check_economic_arm(record, store, list_for([...], store, text_only_sha256=...))`.
  - `_decide(arms, margins, store)` writes `write_candidate_list(tmp_path / 'candidates.yaml',
    arms, ...)` and passes it. Every existing decision assertion stays.
  - `test_the_pairing_check_alone_refuses_an_arm_without_exactly_one_run` becomes
    `test_an_economic_arm_with_no_runs_pairs_nothing`. Two identical runs now pair.
  - `test_a_candidate_reads_its_own_text_only_table_at_its_dimension` asserts the D9 refusal on
    a text-only candidate (spec from `spec(...)`, table from `build_candidate`). A tree
    candidate is not D9-checked.
  - `test_a_provenance_edited_after_it_was_stored_is_refused` keeps its message. The stored
    comparison runs before the rule.
  - `test_decide_reads_an_economic_decision_from_relative_margins` (`test_cli_commands.py`)
    passes `--candidates` from `write_candidate_list`. Its three output assertions stay.
  - The test module's `candidates` fixture (the factory) keeps its name. Name a
    `CandidateList` local `frozen`.

- [x] **Step 1: Write the failing tests**

```python
def test_a_text_only_candidates_embedding_predicts_as_its_text_only_comparator(
    growth_panel, text_only, margins, store, lock, tmp_path
):
    '''§6 "D9 consistency": text-only-dN's covariates+embedding is its covariates+text_only.'''
    table = build_candidate('text-only', 3, write_codebook(tmp_path / 'cb.parquet'),
                            tmp_path / 'c' / 'text-only-d3.parquet', text_only=text_only)
    arm = _read(table, growth_panel, text_only, margins, store, lock,
                arm_spec=spec('text-only-d3', dimension=3, geometry='euclidean'))
    keys = ['panel', 'repeat', 'code', 'feature_year']
    predictions = store.read_frame(arm.runs[0].predictions)
    embedding = predictions.filter(pl.col('comparator') == 'covariates+embedding').sort(keys)
    text = predictions.filter(pl.col('comparator') == 'covariates+text_only').sort(keys)
    assert embedding.get_column('prediction').equals(text.get_column('prediction'))

def test_a_record_whose_predictions_belong_to_another_table_is_refused(
    candidates, store, text_only
):
    a, b = candidates('a', _both(1.0)), candidates('b', _both(1.0))
    swapped = a.model_copy(update={'runs': [a.runs[0].model_copy(
        update={'predictions': b.runs[0].predictions})]})
    frozen = list_for([a, b], store, text_only_sha256=sha256_file(text_only))
    with pytest.raises(ValueError, match='^a: its predictions belong to another table'):
        check_economic_arm(swapped, store, frozen)

def test_the_decision_reads_exactly_the_frozen_lists_candidates(
    candidates, margins, store, tmp_path, text_only
):
    a, b = candidates('a', _both(0.5)), candidates('b', _both(1.0))
    listed = write_candidate_list(tmp_path / 'list.yaml', [a, b], store,
                                  text_only_sha256=sha256_file(text_only))
    settings = {'candidates': listed, 'replicates': REPLICATES, 'bootstrap_seed': BOOTSTRAP_SEED}
    with pytest.raises(ValueError, match="'b'.*missing"):
        decide_economic('d', 'q', [a], margins, store, **settings)
    c = candidates('c', _both(1.0))
    with pytest.raises(ValueError, match="'c' is not on the frozen candidate list"):
        decide_economic('d', 'q', [a, b, c], margins, store, **settings)
    record = decide_economic('d', 'q', [a, b], margins, store, **settings)
    assert record.candidates.sha256 == sha256_file(listed)
```

  Add:
  - `test_a_record_whose_stored_scores_differ_from_its_predictions_is_refused`: one `value` is
    raised by 1e-6 and the run is re-put.
  - `test_margins_whose_stored_scores_differ_from_their_read_are_refused`.
  - `test_a_record_whose_statistics_name_other_panels_is_refused`: one key is dropped, then an
    extra key is added (deferred `:650`).
  - `test_a_logged_identity_of_another_type_is_refused`, parametrized: `dimension` 3.0, or
    `level` True, in a log record's detail. Both are refused as `names another` (`_same`,
    deferred `:650`).
  - `test_a_deterministic_run_with_selection_evidence_is_refused`, parametrized over a seed, a
    checkpoint, a training run, an epoch, monitor records, and a `source` on the arm. Each
    message contains `one seedless run`.
  - `test_a_candidate_off_its_list_row_is_refused`, parametrized: another dimension or
    geometry, an unpinned row, and a row pinning another sha.
  - `test_a_tree_candidate_reads_no_backbone_and_only_the_shared_text_only_table`: a tree row
    with a backbone spec is refused. A list whose `text_only_sha256` is another sha refuses
    with `the shared text-only table`.
  - `test_a_candidate_whose_provenance_breaks_the_rule_is_refused`: a tree table written by
    `write_candidate(..., qcew={...})` is read, then checked. The message contains `Req 29`.
  - `test_an_economic_arm_with_no_runs_pairs_nothing` (the rewrite above).
  - In `test_cli_commands.py`, `test_decide_refuses_candidates_with_d8_margins`: exit 1 with
    `applies only to the economic decision`, and no record written.

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_decision_economic.py tests/unit/test_cli_commands.py -k "economic or decide" -q`
Expected: the new tests FAIL. `check_economic_arm` takes 2 positional arguments, the predictions
carry no binding column, stored scores are never compared, and `decide_economic` has no
`candidates`.

- [x] **Step 3: Implement the changes listed under Interfaces, then apply the rewrites by design**
  > Deviation: the last assertion of `test_a_single_replicate_record_validates_and_pairs_with_the_reference` selects the margin read's columns first, as spec :296 says (ledger ruling, Task 5). The plan's Rewrites list missed it.

- [x] **Step 4: Run the economic, D8 and CLI tests**

Run: `uv run pytest tests/unit/test_decision*.py tests/unit/test_economic_candidates.py tests/unit/test_cli_commands.py -q`
Expected: PASS, with the D8 files unedited.

- [x] **Step 5: Confirm the reference path is untouched**

Run: `git diff --quiet b4f8c9d -- src/naics_embedder/panels/regressor.py src/naics_embedder/panels/qcew_rows.py src/naics_embedder/panels/regressor_splits.py src/naics_embedder/panels/ridge.py; echo exit=$?`
Expected: `exit=0`.

- [x] **Step 6: Commit**

```bash
git add src/naics_embedder/decision src/naics_embedder/cli/commands/tools.py tests/fixtures/economic.py tests/unit/test_decision_economic.py tests/unit/test_cli_commands.py
git commit -m "feat(decision): bind economic reads to their table and decide over the frozen list"
```

---

### Task 6: Trained candidates — import, read and check

**Files:**
- Create: `src/naics_embedder/decision/trained.py` (the import; `candidates.py` cannot hold it,
  because `provenance.py` imports `candidates.py`)
- Modify: `src/naics_embedder/decision/economic.py` (`read_trained_arm`, the trained branch of
  `check_economic_arm`, `_check_growth_reads`, `check_economic_margins_first`),
  `src/naics_embedder/decision/decide.py:405-443` (`check_margins_first`)
- Modify: `tests/fixtures/decision.py` (`synthetic_arm`, `_table`), `tests/fixtures/economic.py`
  (`trained_source`; trained rows in `list_for`)
- Test: `tests/unit/test_economic_candidates.py`, `tests/unit/test_decision_economic.py`

**Interfaces:**
- Consumes: `decide.check_arm`, `check_seed_table`, `MIN_SEEDS`; `store.check_provenance`,
  `provenance_fields`; `records.read_record`, `ArmRecord`; `check_candidate_provenance`;
  `economic_spec`.
- Produces:
  - `decide.check_margins_first(arms, fixed_at, *, exempt=(), monitor_reads: bool = True)`.
    With `monitor_reads=False`, it times only each run's `log_records`. D8's callers pass
    nothing, so their behavior and messages are unchanged.
    `economic.check_economic_margins_first` passes `monitor_reads=False`. A trained seed's
    monitor reads are outcome reads from training, not growth reads (§4.4).
  - `trained.TrainedImport` (frozen dataclass): `entry: CandidateEntry`, `spec: ArmSpec`,
    `source: ArmRecord`, `source_ref: ArtifactRef` and `provenance: Mapping[int, Path]` (each
    seed's export provenance file).
  - `trained.import_trained_candidate(entry: CandidateEntry, candidates: CandidateList, store:
    ArtifactStore, runs_root: Union[str, Path]) -> TrainedImport`. It reads no panel and writes
    nothing. In order:
    1. The source record. `source_ref = ArtifactRef(path=entry.record,
       sha256=entry.record_sha256, bytes=size)`, verified by `store.resolve` (the message names
       the pin on a mismatch). `read_record(..., ArmRecord)`. `source.spec.name == entry.arm`.
    2. The source's checks. `check_arm(source, store, MIN_SEEDS)`. The sorted source seeds equal
       `entry.seeds`. Every run names a `training_run` (message contains `no selection
       evidence`). `source.text_only.table.sha256 == candidates.text_only_sha256`.
    3. The spec. `spec = economic_spec(entry, source=source.spec)`.
    4. Per seed, in seed order:
       - `path = provenance_path(runs_root / entry.run_directory.format(seed=seed) /
         Path(run.table.path).name)`, which must exist (`FileNotFoundError` naming the path).
       - `check_provenance(content, run.table.sha256, run.table.matrix_fingerprint, str(path))`
         and `check_candidate_provenance(content, entry, candidates)`.
       - `content['checkpoint']['sha256'] == run.checkpoint.sha256` (message contains `another
         checkpoint`).
       - `check_seed_table(spec, seed, provenance_fields(...))`.
  - `economic.read_trained_arm(trained: TrainedImport, *, regressor_panel: RegressorPanel,
    margins: RelativeMarginRecord, store: ArtifactStore, purpose: str, lock: Union[str, Path]) ->
    EconomicArmRecord`.
    - Every check runs before the first read: the growth panel, the same panels as the margins,
      the environment, and `check_recorded_text_only(spec, source.text_only, store)`. Each
      seed's `ArmTables` is built, with its dimension equal to the spec's, and passes
      `require_arm`. Each provenance file is stored (`store.candidate_provenance(run.table,
      store.put(path))`).
    - Then, per seed in seed order, it reads both regimes with the detail `{'run': run_id,
      'arm_name': spec.name, 'seed': seed}` and `run_id = f'{spec.name}/seed-{seed}/{hex}'`.
      It adds the binding column, then stores the scores and predictions.
    - Each `EconomicRun` copies `seed`, `checkpoint`, `training_run`, `checkpoint_epoch` and
      `monitor_records` from the source run. The record has `text_only = source.text_only` and
      `source = trained.source_ref`.
  - `check_economic_arm`'s trained branch (row kind `trained`) runs these checks, then the
    per-run checks of Task 5:
    1. The source. `arm.source` is set and equals the row's pin (path and sha256). The source
       loads as an `ArmRecord` named `entry.arm` and passes `check_arm(source, store,
       MIN_SEEDS)`.
    2. The spec and text-only table. `arm.spec == economic_spec(entry, source=source.spec)`.
       `arm.text_only == source.text_only`, whose sha256 is the list's.
    3. The shape. There are at least `MIN_SEEDS` runs, each with a distinct, non-None seed, and
       the seeds equal the source's. Otherwise the message contains `one seedless run or at
       least 5 runs with distinct seeds`.
    4. The evidence. Each run's `(table, checkpoint, training_run, checkpoint_epoch,
       monitor_records)` equals its source seed run's, and `training_run` is not None (message
       contains `selection evidence`). Its provenance passes the rule as an export, and names
       the run's checkpoint sha256.
  - `_check_growth_reads`: for a run with a seed, the logged `seed` must be the run's (`_same`).
  - `tests/fixtures/decision.py`: `synthetic_arm(..., codes: Sequence[str] = CODES)` and `_table(
    ..., codes)`. The default is unchanged, so D8 tests need no edit.
  - `tests/fixtures/economic.py`:
    - `trained_source(store: ArtifactStore, directory: Path, name: str, *, text_only: Path,
      dimension: int = DIMENSION, geometry: str = 'euclidean') -> Tuple[CandidateEntry, Path]`.
      It runs `synthetic_arm(..., codes=CODEBOOK, monitor_mrrs=(0.1, 0.3, 0.2),
      text_only_table=text_only)`, with five seeds 0–4. It writes the record to
      `<store>/records/<name>.json`. It writes each seed's table and `export_provenance` under
      `<runs_root>/<name>-s<seed>/`. It returns the trained row and the runs root.
    - `list_for` also writes trained rows for arms with a `source`. `record` and
      `record_sha256` are `arm.source`'s path and sha256. `arm` and `seeds` come from the
      loaded source record. `run_directory` is `f'{source.spec.name}-s{{seed}}'`, the layout
      that `trained_source` writes (an `ArmRecord` holds no run-directory pattern).

- [x] **Step 1: Write the failing tests**

  In `test_decision_economic.py`, request `trained` before `margins`, so its monitor reads
  precede the margins as Stage 7/8's do. A test asserts that order:

```python
@pytest.fixture
def trained(store, text_only, tmp_path):
    return trained_source(store, tmp_path / 'trained', 'euclidean-d3', text_only=text_only)

def test_a_trained_candidate_is_read_once_per_seed_on_each_growth_panel(
    trained, margins, growth_panel, store, lock, log, text_only
):
    entry, runs_root = trained
    frozen = fixture_list([entry], text_only_sha256=sha256_file(text_only))
    imported = import_trained_candidate(entry, frozen, store, runs_root)
    arm = read_trained_arm(imported, regressor_panel=growth_panel, margins=margins,
                           store=store, purpose=PURPOSE, lock=lock)
    assert [run.seed for run in arm.runs] == [0, 1, 2, 3, 4]
    assert [(r['panel'], r['detail']['seed']) for r in log.records()[2:]] == [
        (panel, seed) for seed in range(5) for panel in GROWTH_PANELS]
    for run, source_run in zip(arm.runs, imported.source.runs):
        assert (run.table, run.checkpoint, run.training_run, run.checkpoint_epoch,
                run.monitor_records) == (source_run.table, source_run.checkpoint,
                source_run.training_run, source_run.checkpoint_epoch, source_run.monitor_records)
    assert arm.source == imported.source_ref
    check_economic_arm(arm, store, frozen)
    check_reference_pairing(margins, arm, store)
```

  Add (§6 "Trained records", "Margin first"):
  - `test_a_trained_arm_of_the_wrong_shape_is_refused`, parametrized: the trained arm cut to 4
    runs, a seedless run among several, and a single seeded run. Each message contains `one
    seedless run or at least 5 runs with distinct seeds`.
  - `test_a_trained_run_without_selection_evidence_is_refused`: `training_run`,
    `checkpoint_epoch` and `monitor_records` cleared on one run. The message contains
    `selection evidence`.
  - `test_a_trained_run_whose_evidence_differs_from_its_source_is_refused`: `checkpoint_epoch`
    + 1.
  - `test_a_trained_seed_whose_monitor_reads_precede_the_margins_is_accepted`. It asserts
    `min(monitor read times) < margins.fixed_at`, then that `check_economic_margins_first([arm],
    margins)` passes.
  - `test_a_trained_seeds_growth_read_before_the_margins_is_refused`: later margins are fixed
    after the read. The message matches `^euclidean-d3 seed 0 read at `.
  - `test_every_trained_seed_pairs_with_the_margin_read`: one seed's reference prediction is
    moved by 1e-12. The message contains `predictions differ from the margin read's`.
  - `test_d8_still_times_a_trained_runs_monitor_reads`: a `synthetic_arm(...,
    monitor_mrrs=...)` is read before the margins' `fixed_at`. `check_margins_first(...)` with
    its default refuses with `monitor read of epoch`.

  In `test_economic_candidates.py`:
  - `test_import_refuses_a_source_record_that_is_not_its_pin` (the record bytes are edited
    after pinning).
  - `test_import_refuses_a_seed_with_no_export_provenance` (one file is deleted; the message
    names its path).
  - `test_import_refuses_a_provenance_naming_another_checkpoint`.
  - `test_import_refuses_a_source_run_without_selection_evidence` (`synthetic_arm` with no
    `monitor_mrrs`).

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_decision_economic.py tests/unit/test_economic_candidates.py -k "trained or import or d8_still" -q`
Expected: FAIL with `ImportError` for `naics_embedder.decision.trained`, `read_trained_arm` and
`trained_source`.

- [x] **Step 3: Implement the `decide.py` switch, the fixtures, the import and the trained read, as listed under Interfaces**
  > Deviation: `check_economic_arm`'s trained branch also requires the arm's seeds to equal its list row's (ledger ruling, Task 6). The plan's `test_d8_still_times_a_trained_runs_monitor_reads` expected `monitor read of epoch`, but D8's early-read message is `read at ..., before the margins were fixed`, so two tests pin both paths (ledger ruling, Task 6). After the final review, the gate also runs D9 (`provenance_fields`, `check_seed_table`) on each trained run's stored export provenance (commit fa6f2a4; ledger ruling, final review).

- [x] **Step 4: Run the decision and CLI tests**

Run: `uv run pytest tests/unit/test_decision*.py tests/unit/test_economic_candidates.py tests/unit/test_cli_commands.py -q`
Expected: PASS, with the D8 files unedited.

- [x] **Step 5: Commit**

```bash
git add src/naics_embedder/decision tests/fixtures tests/unit/test_decision_economic.py tests/unit/test_economic_candidates.py
git commit -m "feat(decision): import and read trained economic candidates with their selection evidence"
```

---

### Task 7: The propagation report and the decision over both kinds

**Files:**
- Modify: `src/naics_embedder/decision/economic.py` (`decide_economic`),
  `src/naics_embedder/decision/records.py` (`EconomicDecisionRecord.propagation`),
  `src/naics_embedder/cli/commands/tools.py` (`decide_command`'s output)
- Test: `tests/unit/test_decision_economic.py`, `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes: `CandidateList.propagation_pairs`; Task 6's `trained_source`, `scaled_arm`.
- Produces:
  - `decide_economic`: after the comparisons with the reference, `propagation = [against(
    smoothed, twin) for smoothed, twin in candidate_list.propagation_pairs()]` on the
    decision's own draws and δ. They are a report and change no field that chooses (§4.7).
  - `EconomicDecisionRecord.propagation: List[Comparison]`, after `candidates`. The docstring
    says it holds split Req 19's answer at each d: propagation helps at d if `smoothed-dN` is
    adopted over `text-only-dN`.
  - `tools decide`, economic output: after the comparisons, one line per propagation comparison:
    `Propagation: <smoothed> over <twin>: adopted|not adopted`.

- [x] **Step 1: Write the failing tests**

  - `test_the_decision_records_each_propagation_comparison`. A `smoothed-d3` candidate built by
    `build_candidate('smoothed', …)` is scaled 0.5, and its `text-only-d3` twin is scaled 1.0.
    Then `[(c.a, c.b) for c in record.propagation] == [('smoothed-d3', 'text-only-d3')]`, and
    `record.propagation[0].adopted`. `record.chosen` equals the choice made by a decision
    without the report: `smoothed-d3`, adopted over the reference.
  - `test_a_trained_candidates_seeds_are_resampled_inside_each_group_draw`. The inputs are a
    trained arm (5 seeds, `scaled_arm` 0.5) and the reference. For each growth panel, the
    comparison's `noninferiority_interval` equals `percentile_interval` of the reference's
    minus the arm's `replicate_statistics(items.sums(values), items.sizes, unit_draws(panel,
    units, REPLICATES, BOOTSTRAP_SEED), seed_draws(panel, side, n, REPLICATES,
    BOOTSTRAP_SEED))`, with n = 1 for the reference and 5 for the arm (`rel=1e-12`).
  - `test_a_decision_over_a_deterministic_and_a_trained_candidate_records_both`. The record's
    `arms` hold one run and five runs, each comparison has both growth panels, and `reports`
    has both arms (§6 "Decision").
  - In `test_cli_commands.py`:
    - `test_decide_prints_the_chosen_candidate_when_one_is_adopted` (deferred `:650`): the
      record from `scaled_arm(arm, {GROWTH_SEEN: 0.5, GROWTH_HELDOUT: 0.5}, store)` and
      `write_record`. The output contains `Chosen: tree-d3`.
    - `test_decide_prints_each_propagation_comparison`: the output contains `Propagation:
      smoothed-d3 over text-only-d3:`.
    - `test_decide_refuses_a_file_that_is_not_a_margin_record` (deferred `:650`): a JSON with
      `kind: economic_arm` as `--margins`. Exit 1, and the output contains `is not a margin
      record`.
    - Match on `click.unstyle(result.output).replace('\n', '')` (Rich wraps at 80 on CI).

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_decision_economic.py tests/unit/test_cli_commands.py -k "propagation or resampled_inside or deterministic_and_a_trained or chosen_candidate or not_a_margin" -q`
Expected: the propagation tests FAIL with `AttributeError: … 'propagation'`. The resampling,
both-kinds, adopted-path and `_margin_kind` tests may already pass. They pin behavior that
Tasks 5–6 kept and that the deferred item asked to be tested.

- [x] **Step 3: Implement the propagation report, its record field and its output line**

- [x] **Step 4: Run the decision and CLI tests**

Run: `uv run pytest tests/unit/test_decision*.py tests/unit/test_economic_candidates.py tests/unit/test_cli_commands.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/naics_embedder/decision src/naics_embedder/cli/commands/tools.py tests/unit/test_decision_economic.py tests/unit/test_cli_commands.py
git commit -m "feat(decision): report propagation and decide over deterministic and trained candidates"
```

---

### Task 8: The commands — `economic-build` and `economic-read`

**Files:**
- Modify: `src/naics_embedder/cli/commands/tools.py` (the module docstring's command list, two
  new commands)
- Test: `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes: `build_candidate`; `load_candidate_list`, `candidate_table_path`, `economic_spec`,
  `ECONOMIC_CANDIDATES`; `import_trained_candidate`; `read_deterministic_arm`,
  `read_trained_arm`, `check_relative_margins`, `read_environment`; `check_candidate_provenance`;
  `provenance_fields`, `check_provenance`; `_margin_kind`, `_require_new_record`.
- Produces:
  - **`tools economic-build`.** Options: `--codebook PATH` (required), `--text-only PATH`
    (required), `--store PATH` (required), `--candidates PATH` (default `ECONOMIC_CANDIDATES`),
    and `--candidate NAME`, repeatable (default: every deterministic row, in list order).
    - Before writing anything, it requires `sha256_file(codebook) == list.codebook_sha256`,
      `sha256_file(text_only) == list.text_only_sha256`, and every named row deterministic.
    - It writes each table at `candidate_table_path(store, name)` with `build_candidate`, and
      prints the name, path and sha256. For a pinned row, a built table whose sha256 differs is
      refused (message contains `differs from its pin`).
    - Failures print `Economic build failed:` and exit 1.
  - **`tools economic-read`.** Options: `--candidate NAME` (required), `--candidates PATH`
    (default `ECONOMIC_CANDIDATES`), `--codebook PATH` (required), `--margins PATH` (required),
    `--store PATH` (required), `--output PATH` (required), `--text-only PATH` (required for a
    deterministic candidate), `--runs-root PATH` (required for a trained candidate),
    `--purpose` (default `economic candidate read`), `--log`, `--lock` (default `LOCK_FILE`) and
    `--dry-run`. Every step below runs before the panel loads, in this order:
    1. `_require_new_record(output)`.
    2. Load the list and the candidate's row.
    3. Check the margins. `_margin_kind(margins)` must be `relative_margins`, else the command
       refuses with `reads relative margins`. Then `read_record(..., RelativeMarginRecord)` and
       `check_relative_margins`.
    4. The candidate's checks.
       - Deterministic: the table at `candidate_table_path`, with its sha256 equal to the
         row's pin (`pins no table` or `differs from its pin`). `check_provenance` and
         `check_candidate_provenance` on the provenance beside it. `--text-only`'s sha256 equal
         to the list's. The spec from `economic_spec`, with the D9 fields from
         `provenance_fields` of the text-only provenance.
       - Trained: `import_trained_candidate`.
    5. `read_environment(lock)`.
    6. With `--dry-run`, it prints `Dry run: every check passed for <name>; no panel was loaded
       and nothing was logged` and exits 0.
    7. Otherwise it loads the growth panel as `growth-margins` does (`outcome` growth in memory,
       level 6, `log_path=log`). It calls `read_deterministic_arm` or `read_trained_arm`, then
       `write_record`. It prints each growth panel's mean statistic over the runs, the record
       path and the log path.
    - Failures (`OSError`, `ValueError`) print `Economic read failed:` and exit 1.

- [x] **Step 1: Write the failing tests**

  Reuse the `growth_panel` fixture (a fake `load_regressor_panel`) and `_growth_margins`. A
  module-level helper `_economic_inputs(tmp_path, store)` writes these and returns their paths:
  - the fixture codebook and the text-only table;
  - a fixture list with `text-only-d3`, `tree-d3`, `smoothed-d3` and `euclidean-d3` (from
    `trained_source`, over the fixture's store);
  - the built tables, and then the margins.

  `_economic_read(runner, inputs, name, *extra)` invokes `economic-read` with every input
  option and `--output tmp_path / 'read.json'`.

```python
@pytest.mark.unit
def test_economic_read_dry_run_loads_no_panel_and_leaves_the_log_untouched(
    runner, tmp_path, growth_panel
):
    panel, loaded = growth_panel
    inputs = _economic_inputs(tmp_path, ArtifactStore(tmp_path / 'store'))
    log = tmp_path / 'real' / 'selection_log.jsonl'
    for name in ('text-only-d3', 'tree-d3', 'smoothed-d3', 'euclidean-d3'):
        result = _economic_read(runner, inputs, name, '--dry-run', '--log', str(log))
        assert result.exit_code == 0, result.output
        output = click.unstyle(result.output).replace('\n', '')
        assert 'no panel was loaded and nothing was logged' in output
    assert loaded == [('growth', 'naics_codebook.parquet', None, (6, ))]   # the margins' only
    assert not log.exists()
    assert not (tmp_path / 'read.json').exists()
```

  Add:
  - `test_economic_build_writes_each_deterministic_table_with_its_provenance`: three tables
    under `store/candidates/economic/`, each with its sha256 printed.
  - `test_economic_build_refuses_a_text_only_table_other_than_the_lists` (Review Focus 3): exit
    1, and nothing written under `candidates/`.
  - `test_economic_read_refuses_a_text_only_table_other_than_the_lists` (Review Focus 3): exit
    1, and no panel load beyond the margins'.
  - `test_economic_read_refuses_a_table_that_differs_from_its_pin_before_loading_the_panel`
    (Review Focus 2): the stored table is rewritten after pinning. Exit 1, the output contains
    `differs from its pin`, and `loaded` has only the margins' load.
  - `test_economic_read_refuses_a_trained_candidate_whose_store_lacks_a_checkpoint` (Review
    Focus 4): one seed's checkpoint object is deleted from the store. Exit 1, the output names
    the missing object's path, and no panel load.
  - `test_economic_read_refuses_an_existing_record_before_any_check` (Review Focus 5):
    `--output` exists. Exit 1 with `exists; a record is written once`, no panel load, and the
    file unchanged.
  - `test_economic_read_refuses_d8_margins`: the output contains `reads relative margins`.
  - `test_economic_reads_and_decide_record_a_decision_over_both_kinds` (§6 "Decision"):
    - `economic-read` runs for `tree-d3` and `euclidean-d3`, with exit 0 each, and the log
      gains 2 + 10 growth validation reads.
    - Then `tools decide --arm … --arm … --candidates <a list of those two>` exits 0.
    - The record has 1 and 5 runs, and the output contains `Chosen:`.

- [x] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_cli_commands.py -k economic -q`
Expected: FAIL with `No such command 'economic-read'` (and `economic-build`).

- [x] **Step 3: Implement the two commands as listed under Interfaces**
  > Deviation: `load_candidate_list` wraps a YAML parse error as `CandidateListYAMLError`, a `ValueError`, so all three commands that call it refuse a broken list file with one message (ledger ruling, Task 5). `--dry-run` stops after the checks the command lists: the table's width, its panel-code coverage, the panel's identity against the margins and a deterministic candidate's D9 comparison run only in the real read, before its first logged read.

  Each command's docstring has an `Example:` block, as `growth-margins` has.

- [x] **Step 4: Run the CLI tests, then check the Rich wrapping as CI does**

Run: `uv run pytest tests/unit/test_cli_commands.py -q`
Expected: PASS.
Run: `COLUMNS=80 uv run pytest tests/unit/test_cli_commands.py -k "economic or decide" -q --basetemp=/tmp/p16-cli-basetemp-xxxxxxxx`
Expected: PASS. Then remove that `--basetemp` directory.

- [x] **Step 5: Commit**

```bash
git add src/naics_embedder/cli/commands/tools.py tests/unit/test_cli_commands.py
git commit -m "feat(cli): add tools economic-build and economic-read with a dry run"
```

---

### Task 9: Docs and the full gates

**Files:**
- Create: `docs/api/tree_model.md`
- Modify: `docs/.nav.yml` (`Tree Model: api/tree_model.md` under Training, after HGCN Training),
  `docs/api/decision.md` (sections `## The candidate list` →
  `::: naics_embedder.decision.candidates`, `## Text and tree only` →
  `::: naics_embedder.decision.provenance`, and `## Trained candidates` →
  `::: naics_embedder.decision.trained`), `docs/usage.md`
  (`### tools economic-build` and `### tools economic-read` after `### tools growth-margins`, and
  `--candidates` under `### tools decide`), `CLAUDE.md`

**Interfaces:**
- Consumes: the docstrings of Tasks 2–8.

- [x] **Step 1: Write the API page and nav entry**

  `docs/api/tree_model.md`: `# Tree Model API`, one sentence on the three builders (§4.2), and
  `::: naics_embedder.tree_model.scaling`, `.smoothing` and `.builders`.

- [x] **Step 2: Write the usage sections**
  > Deviation: the `economic-read` example in `docs/usage.md` names `/path/to/naics-embedder` for the main checkout, not a home-directory path (final review).

  - Each section gives the command's purpose and an example. The read's example shows the
    worktree form: every path absolute into MAIN, and `--log MAIN/logs/selection_log.jsonl`.
    Each section lists the options.
  - State that the dry run loads no panel and logs nothing.
  - State that a real read appends two growth validation reads per run to `--log`, or to the
    regressor config's log.

- [x] **Step 3: Update `CLAUDE.md`**

  - **The economic decision paragraph.** Replace "read_deterministic_arm (no command yet)" and
    "A trained candidate's seeds have no economic record yet". They become: `tools economic-read`
    reads one frozen-list candidate, deterministic or trained (with `--dry-run`), and
    `tools decide` takes `--candidates`.
  - **The other sections.** Add `tree_model/` to the directory tree and the Architecture
    Summary. Add `test_tree_model.py` and `test_economic_candidates.py` to Testing and
    Validation.
  - **The counts.** Recount them:
    `find src -name '*.py' | wc -l`, `ls tests/unit/test_*.py | wc -l`, and
    `uv run pytest --collect-only -q -q | tail -1`. Update the source, unit and test counts to
    the printed numbers.

- [x] **Step 4: Run the gates**

Run: `uv run ruff check src/ tests/`
Expected: `All checks passed!`
Run: `./scripts/format_code.sh --check --all; echo exit=$?`
Expected: `exit=0`.
Run: `uv run mkdocs build --strict 2>&1 | tail -3`
Expected: the build succeeds with no warning. Open
`site/api/tree_model/index.html` and the new `docs/usage.md` anchors, and check that they render.
Run: `uv run pytest -n auto -q 2>&1 | tail -3`
Expected: all pass, with skips only for local data or hardware. No failure.
Run: `shasum -a 256 uv.lock`
Expected: `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948  uv.lock`.

- [x] **Step 5: Commit**

```bash
git add docs CLAUDE.md
git commit -m "docs: document the economic candidates, their builders and commands"
```

---

### Task 10: The nine real tables and the 18-candidate dry run (real data, no panel)

This task runs only after Tasks 1–9 pass. It reads the real codebook, the shared text-only table,
the Stage 7/8 records, checkpoints and export provenance, and the margin record. It reads no
panel and no QCEW file, and it logs nothing (§4.6). WT is this worktree's absolute path.

**Files:**
- Modify: `conf/data/economic_candidates.yaml` (nine `table_sha256` pins),
  `tests/unit/test_economic_candidates.py` (require every pin)

- [x] **Step 1: Pin the environment and the log (P1)**

Run, from WT:
- `uv sync --locked`
- `uv run --locked python -c "import naics_embedder; print(naics_embedder.__file__)"`.
  Expected: `WT/src/naics_embedder/__init__.py`. Otherwise stop, because the run would use
  MAIN's code.
- `shasum -a 256 uv.lock /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl
  ~/naics-artifacts/records/economic/growth_margins.json`, recorded as BEFORE. Expected:
  `uv.lock` is `6de4fbdc…` and the margins are `4f86a0b2…`.
- `ls ~/naics-artifacts/candidates 2>&1`. Expected: `No such file or directory`. If the
  directory exists, stop and show the user what it holds. Never overwrite it.

- [x] **Step 2: Build the nine tables**

Run:

```bash
uv run --locked naics-embedder tools economic-build --codebook /Users/lowell/Projects/naics-embedder/data/supervision/stage3-supervision-v2/301cce28-539c-42ea-8781-496bbdcf511c/naics_codebook.parquet --text-only /Users/lowell/Projects/naics-embedder/checkpoints/plan9_exit/text_only.parquet --store ~/naics-artifacts
```

Expected: exit 0, and nine lines, each with a name, a path under
`~/naics-artifacts/candidates/economic/` and a sha256. The three `tree-dN` cuts pass, at gaps
1,779.1/1,631.5, 970.08/961.94 and 398.63/396.16 (§4.2).

- [x] **Step 3: Rebuild into scratch and compare**

  Run the same command with `--store /tmp/plan16-rebuild`. Expected: nine sha256 values
  identical to Step 2's. Then `rm -rf /tmp/plan16-rebuild`.

- [x] **Step 4: Pin the tables and require every pin**

  Write Step 2's nine sha256 values into the list's `table_sha256` fields. In
  `test_the_committed_list_is_the_specs_eighteen_candidates`, add: every deterministic entry's
  `table_sha256` matches `^[0-9a-f]{64}$`.
  Run: `uv run pytest tests/unit/test_economic_candidates.py -q`. Expected: PASS.
  Commit:

```bash
git add conf/data/economic_candidates.yaml tests/unit/test_economic_candidates.py
git commit -m "chore(conf): pin the nine built economic candidate tables"
```

- [x] **Step 5: Dry-run all 18 candidates**

  The trained checks hash about 5 GB of checkpoints. The pre-check of the nine records took
  seconds on this Mac. Run one `tools economic-read --dry-run` per list name, with:
  - `--codebook <the Step 2 codebook>`
  - `--margins ~/naics-artifacts/records/economic/growth_margins.json`
  - `--store ~/naics-artifacts` and `--text-only <the Step 2 table>`
  - `--runs-root /Users/lowell/Projects/naics-embedder/checkpoints`
  - `--log /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl`
  - `--lock WT/uv.lock` and `--output /tmp/plan16-dry/<name>.json`

  The loop prints `<name> exit=<code>` after each run.
  Expected: 18 lines with `exit=0`, and each run prints `Dry run: every check passed`.

- [x] **Step 6: Verify nothing real was read or written**

Run:
- `shasum -a 256 uv.lock /Users/lowell/Projects/naics-embedder/logs/selection_log.jsonl ~/naics-artifacts/records/economic/growth_margins.json`.
  Expected: identical to BEFORE.
- `ls /tmp/plan16-dry 2>&1`. Expected: no record files. Then remove the directory.
- `git diff --quiet b4f8c9d -- src/naics_embedder/panels/regressor.py src/naics_embedder/panels/qcew_rows.py src/naics_embedder/panels/regressor_splits.py src/naics_embedder/panels/ridge.py; echo exit=$?`.
  Expected: `exit=0`.
- `ls ~/naics-artifacts/records/economic`. Expected: only `growth_margins.json`.

  Record Step 2's nine sha256 values, the 18 exit codes and the BEFORE/after sha256 values for
  the plan's Handoff section (completion protocol, step 3a).

---

## Spec coverage

| Spec | Task |
|---|---|
| §4.1 list, names, pins, record paths | 3, 10 |
| §4.2 builders, determinism, store location | 2, 8, 10 |
| §4.3 shared comparator, tree record, ArmSpec compatibility | 1, 3, 5 |
| §4.4 records, checks, margin first, decision over the list, list sha | 1, 5, 6, 7 |
| §4.5 provenance rule, both export versions | 4, 5, 6 |
| §4.6 command, import, dry run, where it runs | 6, 8, 10 |
| §4.7 propagation comparisons | 3, 7 |
| §4.8 lock, reference path, binding | Global Constraints, 5, 9, 10 |
| §5 roles | Global Constraints, 10 |
| §6 verification items | 1–10 (each test names its item) |
| §7 deferred `:622` (trained), `:633` (stored scores), `:640` (binding), `:650` (subset) | 6–8; 5; 5; 5 and 7 |
| §8 docs | 9 |

---

## Handoff to later stages

- **Stage 3 (Economic — decision read).** It consumes the following, as built.
  - **The frozen list.** `conf/data/economic_candidates.yaml`, loaded by `load_candidate_list`
    (`decision/candidates.py`): 18 candidates, nine deterministic and nine trained. `decide_economic`
    takes it, records its sha256 and refuses any other candidate set. `tools decide --candidates`
    defaults to this file (spec §4.1, §4.4).
  - **The nine real tables.** Built on 2026-11-15 by `tools economic-build` under
    `~/naics-artifacts/candidates/economic/<name>.parquet`, each with `<name>_provenance.json`. A
    second build on this Mac wrote the same bytes. The list pins each table (spec §4.2):

    | Candidate | sha256 |
    |---|---|
    | `text-only-d8` | `474393b2ad3aee41540491c37f285fd37176b311176d386ac0b06848a5d6f0cb` |
    | `text-only-d16` | `8ed9ea4f396f93c6425bfbb591c85b8b96573cd8e443e32cfba8be2e2f8c7fdb` |
    | `text-only-d32` | `b15508a7ade357620d0cfcca944e291b32fb873fee37af197792afecc8937570` |
    | `tree-d8` | `b2da0e6656b4ffd6f5f22c0382084d876b6965752109a1c5caed922786326115` |
    | `tree-d16` | `b54b8d3b725ed3195c9be77b78e9c8b6c08768c2a5bdbd0e6525b8d6c800e4d7` |
    | `tree-d32` | `e8e4d011f4d6a19d5a1554211c712c271162c454cedfa576335a094280d713d2` |
    | `smoothed-d8` | `c73b2c405e99a0e613048dbe76ddb5ed38a118ce784aa96405f3e3e12e778e90` |
    | `smoothed-d16` | `064ee668ec55344f1eef143abf0ea67194cef1279a710f32ce2ebc06d78e2f67` |
    | `smoothed-d32` | `e1c7341488ba3efc6752a198b7517b91fc66f2fde39637d7a1ac6ed2118bb6ea` |

  - **The read command.** `tools economic-read --candidate NAME` (options and an example in
    `docs/usage.md`). It reads one candidate on `growth_seen` and `growth_heldout` and writes its
    economic arm record. A trained candidate is imported from its Stage 7/8 arm record in the store
    (pinned by sha256) and its seeds' run directories under `--runs-root`, then read once per seed:
    5 seeds, and 10 for `hyperbolic-d16` (spec §4.6).
  - **The decision output.** `tools decide` prints one `Propagation:` line for each smoothed
    candidate and its text-only twin, and chooses nothing by it (spec §4.7). The answer to split
    Req 19 is Stage 3's finding.
  - **Differs from the roadmap's `Produces`.** A trained candidate has no new record before Stage 3
    reads it: the Stage 7/8 arm records are imported at the read. The roadmap's `Consumes` cites
    `read_deterministic_arm` (`decision/economic.py:250`), `decide_economic` (`:537`) and an
    `EconomicRun` with no seed field. All three changed, so use the names, not the lines.
  - **Traps and refusals found in execution.**
    - A real read must run under `uv.lock` 6de4fbdc on macOS arm64 with Python 3.12.12, from the
      main checkout or with absolute `--log` and `--lock` (spec §4.8). The worktree's `uv.lock`
      and the main checkout's have the same sha256.
    - A real read appends two growth validation reads for each run to the log: 10 for a 5-seed
      candidate and 20 for `hyperbolic-d16`. It needs the owner's approval when Stage 3 starts.
    - A dry run is not a full check. It stops before the panel loads. The panel's identity against
      the margins, the table's panel-code coverage and width, and a deterministic candidate's D9
      comparison run only in the real read, and each refuses before its first logged read. The
      `--dry-run` help text, the command docstring and the printed line still say "every check"
      (deferred, below).
    - A deterministic row must pin its table before `economic-read` accepts it. All nine do. A
      failed or partial `economic-build` leaves its tables on disk, and a rebuild refuses with
      `exists` until they are deleted by hand.
    - The decision gate runs D9 (`check_seed_table`) on each trained run's stored export
      provenance. It does not pin a deterministic candidate's text-only provenance file, so a
      hand-built record could store another provenance for the same table bytes (deferred, below).
    - `tools growth-margins` must not run again. The decision reads the 2026-10-17 margin record,
      sha256 `4f86a0b2f3bde358bd08a1f3f17f1b4c59ee261ee4e5409ac83615deaa1f37d0`.
  - **Verified without a panel read (Task 10).** All 18 `tools economic-read --dry-run` runs
    exited 0 and printed `Dry run: every check passed`. The sha256 values before and after the
    dry runs are equal: `uv.lock` `6de4fbdc…f948`, the main checkout's
    `logs/selection_log.jsonl` `51aef240…2a41` and `growth_margins.json` `4f86a0b2…37d0`.
    `~/naics-artifacts/records/economic` holds only `growth_margins.json`.
  - **Deferred into Stage 3.** Pin the text-only table's provenance file in the list before the
    first real read: `specs/deferred_items.md`, § 16-economic-candidates. It changes the list's
    sha256, so the list is re-recorded.
