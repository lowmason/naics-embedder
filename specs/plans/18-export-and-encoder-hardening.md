# Export and Encoder Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via
> subagent-driven-development (the default) — or executing-plans when your human partner chose
> inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close plan 8's three open review items on the arm export, the two arm commands and the
encoder tests. The export writes a whole table and provenance pair or nothing. The two commands
refuse bad input with one handled message. Each named coverage gap gets a test that fails when
the line it names breaks.

**Architecture:** Two code tasks and two test-only tasks, then one verification task.

- **Task 1** reorders `export_code_table` so that every input hash comes before the table write,
  and removes the table when its provenance cannot be written.
- **Task 2** makes `load_arm_model` refuse an unreadable checkpoint as a `ValueError`. It escapes
  Rich markup in `tools export-table` and `tools outcome-panel`, and refuses a blank `--purpose`
  before the model loads.
- **Tasks 3 and 4** add or repair tests only. No source line changes.

**Tech Stack:** Python 3.10+ (locked env 3.12), PyTorch 2.9.1, Polars, Typer, Rich, pytest. No new
dependency.

**Spec:** none. The requirement list is the text of three items in `specs/deferred_items.md`,
section `## 8-shared-encoder-and-projection — 2026-10-03`, including their `→ partly` notes:

1. "Review Minor: the export and the arm encoder's reads have untested branches" (item 1).
2. "Review Minor: the export and the outcome read handle a few failures untidily" (item 2).
3. "Review Minor: the encoder, fusion and cache tests leave gaps" (item 3).

Line numbers below are at `ca85e81` (PR #164's merge). The items' own line numbers are older.

## Coverage of the three items

Each open sub-case and its task:

- Item 1, `batch_size < 1` refusal, now `export.py:114-115` in `encode_token_rows`: Task 1.
- Item 1, a table that `coordinate_matrix` refuses is never written: Task 1.
- Item 1, the exact provenance key set and `generated_at`: Task 1.
- Item 1, `export-table`'s OSError branch (`tools.py:1593`): Task 2.
- Item 1, `outcome-panel`'s bare-override and legacy refusals, its bad `--output` refusal before
  the read is logged, and its payload's `fingerprint` key (`tools.py:1650-1676`): Task 2.
- Item 1, the read maps' `.cpu()` never meets an MPS tensor: Task 3. `exp_map_origin` moved from
  `arm_encoder.py:57` to `hyperbolic.py:174`, and its `.cpu()` is at `hyperbolic.py:188`. The flat
  heads' are at `heads.py:177` and `:226`.
- Item 2, the table written (`export.py:298`) before the input hashes (`:304`, `:313`) and the
  provenance (`:329`), and the unresolved paths (`:303`, `:311`): Task 1.
- Item 2, unescaped Rich markup (`tools.py:1594`, `:1597-1598`, `:1662`, `:1669`): Task 2.
- Item 2, corrupt-checkpoint tracebacks (`export.py:209`): Task 2.
- Item 2, the late blank `--purpose` refusal: Task 2.
- Item 2, `docs/usage.md`'s outcome-panel paragraph (`:268-282`) and the window: Task 2.
- Item 3, `AttentionFusion`'s autocast case (`fusion.py:92-93`): Task 3.
- Item 3, the eval-only finite-gradient test (`test_fusion.py:84-104`): Task 3.
- Item 3, the shared width 8 (`test_encoder.py:43`, `:108`; `tests/fixtures/shared_encoder.py:43`):
  Task 4.
- Item 3, the contract test that matches only `'bundle'` (`test_naics_model.py:1663-1672`): Task 4.
- Item 3, the HGCN feeder's encoder-mismatch refusal (`training.py:273`): Task 4.
- Item 3, two backward passes through one graph (`_MpsStateReplay`, `shared_encoder.py:56-73`):
  Task 4.

Dropped sub-cases, each already settled:

- Item 1, an absent curvature read as 1: mooted by plan 10 (Task 11). No curvature is read.
- Item 1, the cap check: mooted by plan 10 (Task 4).
  `test_the_table_holds_each_codes_bounded_tangent` checks the bounded tangent.
- Item 1, the `coordinates` key: done in plan 12 (Task 4).
- Item 1, `export-table`'s ValidationError branch: covered since `b4f10c9` by
  `test_export_table_refuses_the_removed_supervision_mode`. Task 2 only adds its message check.
- Item 2, the tokenizer in the provenance: done in plan 9 (Task 7).
- Item 3, a cache sidecar that changes only the markers: done in plan 9 (Task 1).

## Global Constraints

- **`pyproject.toml` dependencies and `uv.lock` do not change.** The lock's sha256 stays
  `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948`. Redesign Stage 3 pins
  every economic read to it. If a change seems needed, stop and ask.
- **Fixtures only.** Use fixture bundles, tiny models and `tmp_path`. Read no real panel, export
  no real checkpoint and run nothing on Lambda.
- **The export's content stays.** The table's bytes and every provenance value except the two
  `path` entries stay as they are. The key set stays `EXPORT_ENTRIES_WITH_GEOMETRY`
  (`decision/provenance.py:49`), which the Req 29 rule reads.
- **Shared fixtures stay.** Do not change `TINY_HIDDEN` or any other value in
  `tests/fixtures/shared_encoder.py`: every fixture checkpoint is built at them.
- **Style.** Single quotes, semantic dividers, `./scripts/format_code.sh` on touched files only.
  Never run `ruff format`.
- **Pinning tests.** Many tests here pin behavior that already works, so they pass at once. Their
  red step is a sabotage: make the named change, see the named failure, then restore the line.
  Run `git diff` after each restore to confirm that the source is back.
- **MPS.** CI has no MPS device, so MPS-only tests skip there. Run them on the Mac.
- **Neighbors.** Plan 17 (text→code ITA panel) is live on another branch. At `ca85e81` its file
  list shares only `CLAUDE.md` (the test count) with this plan. If it lands first, rebase on it
  and re-pin any line number here that moved.

## Review Focus

- A re-export to an existing table path that fails after the table write. A person expects no
  table at the path that a read could take as good. Task 1 pins it.
- A path or an error message that holds Rich markup, such as `[/x]` or pydantic's
  `[type=…, input_value=…]`. A person expects it printed literally, with exit 1 and no
  `MarkupError` traceback. Task 2 pins it.
- A relative `--checkpoint`, later read from another directory. A person expects the provenance to
  name the file absolutely. Task 1 pins it.
- An empty, truncated or non-checkpoint file as `--checkpoint`. A person expects one line that
  names the file, and exit 1. Task 2 pins it.
- A whitespace-only `--purpose`. A person expects a refusal before any model loads, with nothing
  logged. Task 2 pins it.

## Decisions

- **D1** (owner, brief, 2026-11-18): plan the three items as one themed plan, with no spec, as
  the owner's "Plan" pick in the /deferred pass of 2026-11-18 (PR #164). Applied in: all tasks.
- **D2** (owner, brief, 2026-11-18): never touch `pyproject.toml` dependencies or `uv.lock`.
  Applied in: Global Constraints.
- **D3** (owner, brief, 2026-11-18): edits to `specs/deferred_items.md` touch only these three
  items, since other sessions edit the file. Applied in: Task 5 and the plan's own commit.

---

### Task 1: The export writes a whole pair or nothing

**Files:**
- Modify: `src/naics_embedder/text_model/export.py:227-331` (`export_code_table`)
- Test: `tests/unit/test_export.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `export_code_table`'s signature is unchanged. Its provenance's `checkpoint.path` and
  `descriptions.path` are `str(Path(...).resolve())`. Task 2 edits `load_arm_model` in the same
  file.

- [ ] **Step 1: Write the tests** in `tests/unit/test_export.py`

  - `test_a_batch_size_below_one_is_refused`, parametrized over `0` and `-1`:
    `encode_token_rows(shared_model, rows, batch_size=size)` raises
    `ValueError, match='batch_size must be positive'`. Build `rows` as
    `test_rows_encode_to_float64_cpu_tensors_in_row_order` does.
  - `test_a_table_coordinate_matrix_refuses_is_never_written`: monkeypatch
    `naics_embedder.text_model.export.encode_token_rows` with a wrapper that calls the real one
    and sets `tangent[0, 0] = float('nan')`. `export_code_table` raises
    `ValueError, match='not finite'`, and neither the table nor `provenance_path(output)` exists.
  - `test_an_input_hash_failure_leaves_no_table`: monkeypatch the module's `sha256_file` to raise
    `OSError('unreadable')` for the checkpoint path and call the real one otherwise. The export
    raises `OSError`, and the table does not exist.
  - `test_a_failed_provenance_write_leaves_no_table`: export once to `output`. Then monkeypatch
    the module's `provenance_path` to return `tmp_path / 'missing' / 'p.json'` and export to
    `output` again. It raises `FileNotFoundError`, and `output` does not exist.
  - `test_the_provenance_has_exactly_the_entries_the_economic_rule_reads`: for `exported_table`,
    `set(provenance) == EXPORT_ENTRIES_WITH_GEOMETRY`, and
    `datetime.fromisoformat(provenance['generated_at']).utcoffset() == timedelta(0)`.
  - `test_the_provenance_records_absolute_input_paths`: `monkeypatch.chdir(tmp_path)`, then
    export with the checkpoint and the token config's `descriptions_parquet` each given as
    `os.path.relpath(...)`. Both `path` entries equal `str(<file>.resolve())`.

- [ ] **Step 2: Run them**

  Run: `uv run pytest tests/unit/test_export.py -q -q`
  Expected: the input-hash, provenance-write and absolute-path tests FAIL, on `not output.exists()`
  or on a relative path. The other three pass: they pin behavior that already works.

- [ ] **Step 3: Reorder `export_code_table`**

  Resolve both input paths. Build every provenance entry except `table_sha256` before
  `table.write_parquet`: the two hashes, `library_versions` and `generated_at`. Then write the
  table, add `table_sha256` and write the provenance in one `try`. On any exception there, unlink
  the table and `provenance_path(output_path)` with `missing_ok=True` and re-raise. Update the
  Raises section: a refused table is not written, and a failed provenance write leaves no table.
  `load_arm_model` still receives the path as given.

- [ ] **Step 4: Run them again**

  Run: `uv run pytest tests/unit/test_export.py -q -q`
  Expected: all PASS.

- [ ] **Step 5: Sabotage the three pinning tests, one at a time**

  - Delete `export.py:114-115`: `[0]` fails with `range() arg 3 must not be zero`, and `[-1]`
    fails with a `RuntimeError` from `torch.cat` on an empty list.
  - Move `fingerprint = table_fingerprint(table)` just below `table.write_parquet`, and delete the
    cleanup's table unlink: the NaN test fails on `not output.exists()`. Each of the two defenses
    alone keeps it green.
  - Delete the `'generated_at'` entry: the key-set test fails.

  Restore each line, and confirm with `git diff`.

- [ ] **Step 6: Run the provenance readers**

  ```bash
  uv run pytest tests/unit/test_arm_encoder.py tests/unit/test_economic_candidates.py tests/unit/test_checkpoint_runner.py -q -q
  ```

  Expected: PASS. They read the export's provenance by its sha256 entries, not its paths.

- [ ] **Step 7: Format and commit**

  ```bash
  ./scripts/format_code.sh src/naics_embedder/text_model/export.py tests/unit/test_export.py
  git add src/naics_embedder/text_model/export.py tests/unit/test_export.py
  git commit -m "fix(export): write a whole table and provenance pair or nothing"
  ```

### Task 2: The arm commands refuse cleanly

**Files:**
- Modify: `src/naics_embedder/text_model/export.py:175-221` (`load_arm_model`)
- Modify: `src/naics_embedder/panels/selection_log.py:40-60`
- Modify: `src/naics_embedder/cli/commands/tools.py:1544-1676`
- Modify: `docs/usage.md:243-282`
- Test: `tests/unit/test_export.py`, `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes: Task 1's `export.py`.
- Produces: `require_purpose(purpose: str) -> None` in `naics_embedder.panels.selection_log`. It
  raises `ValueError('every selection-log record needs a purpose')` when `purpose.strip()` is
  empty. `SelectionLog.append` calls it. `load_arm_model` raises
  `ValueError(f'{checkpoint_path} is not a readable checkpoint: {exc}')` when `torch.load` fails.

- [ ] **Step 1: Write the tests**

  In `tests/unit/test_export.py`:
  - `test_an_unreadable_checkpoint_is_refused_by_name`, parametrized over three files: the bytes
    `b'not a checkpoint'` (`UnpicklingError`), `b''` (`EOFError`), and the first half of
    `shared_checkpoint`'s bytes (`RuntimeError`). `load_arm_model(path, validated_bundle,
    summaries=summaries_identity(MINILM))` raises `ValueError, match='is not a readable
    checkpoint'`, and the message holds `path.name`.

  In `tests/unit/test_cli_commands.py`, Rich wraps output at 80 columns on CI. It wraps a phrase
  at a space and folds a long path inside the path. So assert phrases on
  `' '.join(click.unstyle(result.output).split())`, and assert path fragments (`[/x]arm.parquet`,
  `[red]/arm_provenance.json`) on `click.unstyle(result.output).replace('\n', '')`. A test that
  checks both kinds reads the output both ways. Every refusal asserts
  `isinstance(result.exception, SystemExit)` and `exit_code == 1`.
  - Extend `test_export_table_refuses_the_removed_supervision_mode` (`:1270`): assert
    `'Export failed'` and `"input_value='legacy_containment'"` in the message. Rich swallows the
    second today.
  - Extend `test_export_table_exports_under_the_configured_bundle_and_cache` (`:1237`): put the
    output under `tmp_path / '[red]'`, and assert `'[red]/arm_provenance.json'` in the output.
  - `test_export_table_prints_an_unwritable_output_literally`: make the directory
    `tmp_path / '[' / 'x]arm.parquet'` and pass it as `--output`. `export_code_table` raises
    `AssertionError` if called. The message holds `'Export failed'` and `'[/x]arm.parquet'`.
    Today Rich raises `MarkupError` on `[/x]`.
  - `test_export_table_refuses_an_unreadable_checkpoint_without_a_traceback`: patch
    `require_valid_supervision_bundle` to return `validated_bundle` and `pick_device` to return
    `'cpu'`. Pass a checkpoint file holding `b'not a checkpoint'`. The message holds
    `'is not a readable checkpoint'`.
  - `test_outcome_panel_refuses_bad_overrides_before_the_model_loads`, parametrized over
    `('model.dimension', 'key=value')` and
    `('supervision.mode=legacy_containment', "input_value='legacy_containment'")`.
    `ArmEncoder.from_files` raises `AssertionError` if called. The message holds
    `'Outcome panel failed'` and the expected text, and the log has no records.
  - `test_outcome_panel_refuses_a_blank_purpose_before_the_model_loads`: `--purpose '   '`, with
    `require_valid_supervision_bundle` and `ArmEncoder.from_files` both raising `AssertionError`.
    The message holds `'needs a purpose'`, and the log has no records.
  - `test_outcome_panel_refuses_an_unwritable_output_before_the_read_is_logged`: set up as
    `test_outcome_panel_reads_validation_and_logs_the_table` (`:1321`), with `FakeArm`, but make
    `--output` an existing directory. The message holds `'Outcome panel failed'`, and the log has
    no records.
  - `test_outcome_panel_refuses_an_unreadable_checkpoint_without_a_traceback`: copy
    `exported_table` to `tmp_path / 'arm.parquet'`. Write its provenance beside it, with
    `checkpoint.sha256` set to the sha256 of a checkpoint file holding `b'not a checkpoint'`.
    Patch the gate and `pick_device` as above, and pass `data_loader.streaming.max_length=32`
    (`TOKEN_WINDOW`). The message holds `'is not a readable checkpoint'`, and the log has no
    records.
  - Extend `test_outcome_panel_reads_validation_and_logs_the_table`: assert
    `set(payload) == {'fingerprint', 'table', 'checkpoint', 'summary'}` and
    `payload['fingerprint'] == record['fingerprint']`.

- [ ] **Step 2: Run them**

  ```bash
  uv run pytest tests/unit/test_export.py tests/unit/test_cli_commands.py -q -q -k "unreadable or supervision_mode or configured_bundle or unwritable or bad_overrides or blank_purpose or reads_validation"
  ```

  Expected: FAIL for the unreadable-checkpoint tests (an uncaught `UnpicklingError`), the markup
  tests (missing text, or `MarkupError`), the legacy-override case and the blank purpose. The
  unwritable outcome output, the bare override and the payload keys pass: they pin behavior that
  already works.

- [ ] **Step 3: Implement**

  - `selection_log.py`: add `require_purpose` and call it from `SelectionLog.append` in place of
    `:58-59`.
  - `export.py`: wrap only the `torch.load` call (`:209`) in `try`. Catch
    `(pickle.UnpicklingError, EOFError, RuntimeError)` and raise the `ValueError` from it. Add it
    to the docstring's Raises. Do not widen the commands' `except` tuples: a `RuntimeError` from
    a forward pass must still show its traceback.
  - `tools.py`: import `escape` from `rich.markup`. Call `require_purpose(purpose)` as the first
    statement in `outcome_panel`'s `try`. Escape `str(exc)` in both refusal lines (`:1594`,
    `:1662`) and the paths in `:1597`, `:1598` and `:1669`.
  - `docs/usage.md`, export-table paragraph: say that the provenance records the checkpoint and
    descriptions as absolute paths, and that a failed export leaves no table. Outcome-panel
    paragraph: replace "Provenance must match the checkpoint and preprocessing pins." with
    "The table's provenance must name this checkpoint. A read refuses a table exported under
    another `data_loader.streaming.max_length`, tokenizer or window summaries." Add to the
    `--purpose` bullet that a blank purpose is refused before the model loads.

- [ ] **Step 4: Run them again**

  Run the Step 2 command, then:

  ```bash
  uv run pytest tests/unit/test_monitor.py tests/unit/test_selection_log_guard.py -q -q
  ```

  Expected: all PASS. `test_monitor.py:395` still matches `'needs a purpose'`.

  Then reproduce CI's wrapping on the Mac, where `COLUMNS` is unset:

  ```bash
  COLUMNS=80 uv run pytest tests/unit/test_cli_commands.py -q -q --basetemp=/tmp/p18-cli -k "unreadable or supervision_mode or configured_bundle or unwritable or bad_overrides or blank_purpose or reads_validation"
  ```

  Expected: PASS. Then delete `/tmp/p18-cli`.

- [ ] **Step 5: Sabotage the three pinning tests, one at a time**

  - Move `outcome_panel`'s `_require_writable(Path(output))` below `read_outcome_validation`: the
    unwritable-output test finds a logged record.
  - Delete the bare-override refusal in `_run_config` (`tools.py:1321-1322`): the
    `model.dimension` case loses its `key=value` message.
  - Delete the payload's `'fingerprint'` entry: the payload test fails on its key set.

  Restore each line, and confirm with `git diff`.

- [ ] **Step 6: Build the docs**

  Run: `uv run mkdocs build --strict`
  Expected: exit 0, with no warnings.

- [ ] **Step 7: Format and commit**

  ```bash
  ./scripts/format_code.sh src/naics_embedder/text_model/export.py src/naics_embedder/panels/selection_log.py src/naics_embedder/cli/commands/tools.py tests/unit/test_export.py tests/unit/test_cli_commands.py
  git add src/naics_embedder/text_model/export.py src/naics_embedder/panels/selection_log.py src/naics_embedder/cli/commands/tools.py docs/usage.md tests/unit/test_export.py tests/unit/test_cli_commands.py
  git commit -m "fix(cli): refuse unreadable checkpoints, blank purposes and markup cleanly"
  ```

### Task 3: Read maps and fusion meet other devices, dtypes and modes

**Files:**
- Test: `tests/unit/test_heads.py`, `tests/unit/test_fusion.py`

**Interfaces:**
- Consumes: nothing. This task changes no source line.
- Produces: nothing other tasks use.

- [ ] **Step 1: Write the tests**

  - `test_heads.py`: `test_the_read_map_brings_an_mps_export_to_float64_on_the_cpu`, parametrized
    over `GEOMETRIES` and skipped without MPS. With
    `points = head(_vectors(4, dtype=torch.float32))`, the value
    `read = head.read_points(points.tangent.to('mps'))` has dtype float64 and device `cpu`, and
    `torch.equal(read, head.read_points(points.tangent))`.
  - `test_fusion.py`: `test_attention_fuses_finitely_under_autocast_where_its_scores_are_narrower`.
    Under `torch.autocast('cpu', dtype=torch.bfloat16)`, with `_fusion('attention')`, float32
    `_vectors()` whose absent slots hold `-1000.0`, and `requires_grad`:
    `(vectors @ fusion.query).dtype == torch.bfloat16` first. Then the fused vector and
    `vectors.grad` are finite, `fusion.query.grad` is finite and not `None`, and row 2 fuses to
    zeros.
  - `test_fusion.py`: add `@pytest.mark.parametrize('mode', ['eval', 'train'])` to
    `test_a_row_with_no_present_channel_fuses_finitely_with_a_finite_gradient` (`:84-104`), with
    `fusion = _fusion(name).train(mode == 'train')`. Train mode turns on MoE's dropout
    (`moe.py:68`).

- [ ] **Step 2: Run them**

  Run: `uv run pytest tests/unit/test_heads.py tests/unit/test_fusion.py -q -q`
  Expected: PASS. On the Mac the MPS cases run, and on CI they skip.

- [ ] **Step 3: Sabotage each new test, one at a time**

  - Drop `.cpu()` from `hyperbolic.py:188`, then from `heads.py:177`, then from `:226`: the
    matching MPS case raises `TypeError` (`Cannot convert a MPS Tensor to float64 dtype`).
  - Change `fusion.py:93` to `torch.finfo(vectors.dtype).min`: the autocast test raises
    `RuntimeError: value cannot be converted to type at::BFloat16 without overflow`.

  Restore each line, and confirm with `git diff`. The train-mode case extends a test that already
  pins the finite gradient. It needs no sabotage of its own.

- [ ] **Step 4: Format and commit**

  ```bash
  ./scripts/format_code.sh tests/unit/test_heads.py tests/unit/test_fusion.py
  git add tests/unit/test_heads.py tests/unit/test_fusion.py
  git commit -m "test(heads, fusion): pin the MPS read map, autocast scores and train mode"
  ```

### Task 4: Encoder and contract tests test what they name

**Files:**
- Test: `tests/unit/test_encoder.py`, `tests/unit/test_naics_model.py`, `tests/unit/test_export.py`

**Interfaces:**
- Consumes: nothing. This task changes no source line.
- Produces: nothing other tasks use.

- [ ] **Step 1: Separate the widths in `test_encoder.py`**

  Set `WIDTH` (`:43`) to `12`, since the longest text in `MIXED_PRESENCE` has 7 tokens. Set
  `make_encoder`'s default `'dimension'` (`:108`) to `16`. Change `:277-278` to `(2, 16)` and
  `(2, 17 if geometry == 'hyperbolic' else 16)`. The window, the backbone width (`TINY_HIDDEN`, 8)
  and the dimension now differ. The radius bound R = 8.0 (`:260`, `:503`) is a setting, not a
  width, and stays.

- [ ] **Step 2: Write the tests**

  - `test_naics_model.py`: rewrite `test_runtime_contract_must_match_the_loaded_bundle`
    (`:1663-1672`). Build `NAICSContrastiveModel(**model_config)` and take its
    `checkpoint_contract` as `same`. Passing `checkpoint_contract=same` constructs: that is the
    control. `same.model_copy(update={'bundle_id': 'other-bundle'})` raises
    `ValueError, match=r"runtime checkpoint contract .*'bundle_id': 'other-bundle'"`. Only the
    bundle differs, so only the bundle can refuse it.
  - `test_export.py`:
    `test_the_hgcn_feeder_refuses_a_checkpoint_of_another_encoder_before_its_model_loads`,
    parametrized over `('dimension', 8)` and `('fusion', 'attention')`. Set up as
    `test_the_hgcn_feeder_refuses_a_checkpoint_trained_on_truncated_text` (`:214-232`), plus
    `forbid_model_loads(monkeypatch)`, then `setattr(cfg.model, field, value)`. The feeder raises
    `ValueError, match=r"exact resume contract mismatch .*'encoder'.*nothing migrates it"`, and
    no output exists. Do not use geometry: the flat-arm refusal runs first.
  - `test_encoder.py`, after `:644`:
    `test_two_backward_passes_through_one_graph_reenter_the_mps_replay`. Patch
    `torch.backends.mps.is_available` to return `True`. Make `torch.mps.get_rng_state` a `Mock`
    that returns a new one-element tensor on each call, and `torch.mps.set_rng_state` a `Mock`.
    With `make_encoder(use_gradient_checkpointing=True, lora_dropout=0.1).train()`, take
    `loss = encoder(stack_text_inputs(CODES))['tangent'].sum()`. Run
    `loss.backward(retain_graph=True)` and record `first = set_state.call_count`, then run
    `loss.backward()`. Assert `set_state.call_count == 2 * first > 0`. A probe at `ca85e81` gave
    3 recorded states and 6 then 12 calls. It is meant to run on CI, since it needs no MPS device,
    but the probe ran only on the Mac. If CI fails it, replace it with a direct test that enters
    one `_MpsStateReplay` twice with mocked `get_rng_state` and `set_rng_state`.

- [ ] **Step 3: Run them**

  ```bash
  uv run pytest tests/unit/test_encoder.py tests/unit/test_naics_model.py tests/unit/test_export.py -q -q
  ```

  Expected: PASS.

- [ ] **Step 4: Sabotage the new tests, one at a time**

  - In `naics_model.py:284`, compare only `encoder` and `summaries`, not the whole contract: the
    contract test's `pytest.raises` fails.
  - Delete the `validate_exact_resume` call at `training.py:273`: the feeder test reaches a model
    load, and `forbid_model_loads` fails it.
  - Replace `_MpsStateReplay(...)` at `shared_encoder.py:94` with an equivalent
    `contextlib.contextmanager` generator: the second backward raises `AttributeError`
    (`'_GeneratorContextManager' object has no attribute 'args'`), as a probe at `ca85e81` showed.

  Restore each line, and confirm with `git diff`.

- [ ] **Step 5: Format and commit**

  ```bash
  ./scripts/format_code.sh tests/unit/test_encoder.py tests/unit/test_naics_model.py tests/unit/test_export.py
  git add tests/unit/test_encoder.py tests/unit/test_naics_model.py tests/unit/test_export.py
  git commit -m "test(encoder): separate fixture widths and pin the contract, feeder and replay"
  ```

### Task 5: Verify and record

**Files:**
- Modify: `CLAUDE.md:368`, `README.md:261` (the collected test count)

- [ ] **Step 1: Run the full gates**

  ```bash
  uv run pytest -n auto -q
  uv run ruff check src/ tests/
  ./scripts/format_code.sh --check --all
  uv run mkdocs build --strict
  ```

  Expected: no failures, each check exits 0, and only MPS or data-dependent tests skip.

- [ ] **Step 2: Update the test count**

  Run: `uv run pytest --collect-only -q -q | tail -1`. Write that count into `CLAUDE.md:368` and
  `README.md:261`. No unit test file was added, so the file counts stay. If a rebase conflicts
  here, take the newly collected count.

- [ ] **Step 3: Confirm the lock**

  Run: `shasum -a 256 uv.lock` and `git diff ca85e81 -- pyproject.toml uv.lock`
  Expected: `6de4fbdcde8a2e944c7a9daed21f09f8a01ed479052179994ba4b0d488fef948`, and an empty diff.

- [ ] **Step 4: Commit**

  ```bash
  git add CLAUDE.md README.md
  git commit -m "docs: update the collected test count for plan 18"
  ```

  The Plan Completion Protocol then ticks the three items `→ done` in `specs/deferred_items.md`
  and touches no other item there.
