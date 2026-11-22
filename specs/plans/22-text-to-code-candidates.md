# Text→code Candidates, Plan A (Code on Fixtures) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: implement this plan task-by-task via subagent-driven-development (the default) — or executing-plans when your human partner chose inline execution at the handoff. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build every piece of code that Stage 5's real campaign needs, verified on fixtures only:
checkpoint retention, the frozen control, the two new backbones' pins and checks, the margin, the
candidate selection reads and the decision.

**Architecture:** Training keeps a grid of checkpoints through a `ModelCheckpoint` subclass that
keeps the parent's state key. Two new run settings, the retention schedule and the frozen switch,
appear in `run_settings` only when they are not at their defaults, so Stage 8's 21-key records
still compare equal. Three new `tools` commands run the margin record, the per-seed selection
reads and the decision. One committed arms file names the four arms and their roles.

**Tech Stack:** Python 3.12 (CI also 3.10), PyTorch Lightning 2.5.5, PEFT, Polars, Pydantic,
Typer/Rich, pytest.

**Spec:** `specs/text-to-code-candidates.md` (approved, U54). This plan is the spec's §7 Plan A.
Plan B (the real stage) is planned after this plan merges, from `## Handoff to Plan B` below.
This plan's header deliberately carries no roadmap line, so its completion does not stamp Stage 5
(D2).

## Global Constraints

- No real read and no real run. Score no real panel, run no `tools` command that logs a read of a
  real panel, open no sealed part, and scan no embedding-data table. Tests read fixture tables
  only (`tests/fixtures/ita.py` and the reference fixture bundle).
- Gates re-read at embedding-data `46a4d94` (2026-11-22). Since the spec's `3382744`, only
  USAspending files changed: `specs/evaluation-protocol.md`,
  `specs/completed/ita-establishment-targets.md` and `specs/embedding-evaluation-data-spec.md`
  did not change.
- Arm names, exactly: `euclidean-d16-ita`, `gte-small-d16`, `bge-small-d16` (role `candidate`),
  `frozen-minilm-d16` (role `control`). The reference arms stay `lexical-stub` and
  `euclidean-d16` (`REFERENCE_ARMS`). No candidate read uses a reference arm's name.
- Backbones: `sentence-transformers/all-MiniLM-L6-v2` (revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`), `thenlper/gte-small` (revision
  `17e1f347d17fe144873b1201da91788898c639cd`), `BAAI/bge-small-en-v1.5` (revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`). The two new revisions are each repository's `main`
  on 2026-11-22, read from the Hub API with no download.
- Input length 128 for every arm (U53): `data_loader.tokenization.max_length` and
  `data_loader.streaming.max_length` are 128. New summaries artifacts fit window 128.
- Retention for Stage 5's arms: every 4th epoch (0-based epochs 3, 7, …, 39), the last epoch and
  the outcome-best epoch, each once. Budget 40 epochs. Seeds 1 to 5.
- δ = 0.05 × the chosen reference's validation headline MRR (Stage 2's spec §4.5). The decision
  test is Stage 2's: a candidate beats the reference when the lower bound of the 95 % interval on
  Δ exceeds +δ.
- Stage 4's table build: DATA_FP
  `8980e3586ec0a18d4623c3a4b3e81527f8c59e34a1f0b4b73d07b046ed86b1a3`, split fingerprint
  `8602ceb243722dc64e0a29a9ba444fe824baf209a848964d7ddf130a9db5cc7c`. The new read commands
  require the expected data fingerprint as an argument and refuse another build before they log.
- Candidate and control reads log role `candidate`. A control read also logs `control: true` in
  its detail. No new role.
- The selection log is append-only. The new read commands take `--log` as a required absolute
  path, so a worktree never writes its own relative `logs/selection_log.jsonl`.
- A default config gives `run_settings` exactly the 21 keys printed below, in this order. A
  Stage 8 checkpoint saved under them loads, exports, reads and compares equal:
  `fusion, dimension, radius_bound, code_code_weight, radial_weight, target_temperature,
  radial_step, logit_scale_init, logit_scale_range, learning_rate, weight_decay, warmup_epochs,
  lr_plateau_factor, lr_plateau_patience, early_stopping_patience, max_epochs, queries_per_step,
  accumulate_grad_batches, gradient_clip_val, accelerator, precision`.
- The outcome callback's state key stays this literal string, under retention too:
  `"ModelCheckpoint{'monitor': 'val/outcome_mrr', 'mode': 'max', 'every_n_train_steps': 0,
  'every_n_epochs': 1, 'train_time_interval': None}"`.
- `uv.lock` does not change. No new package.
- Style: single quotes, YAPF via `scripts/format_code.sh`, semantic dividers, never `ruff format`.
  Count tests with `uv run pytest --collect-only -q -q`.
- The known local failure: `test_level_predictions_are_the_captured_ones_bit_for_bit` fails on
  this Mac (`specs/deferred_items.md` § 19). Report it, do not fix it.

## Review Focus

1. **A grid epoch that was once outcome-best and was then beaten keeps its file.** Lightning's
   top-1 rotation deletes its previous best. A person who later selects on ITA expects every grid
   epoch to be there. Test: Task 1, `test_a_beaten_best_on_the_grid_keeps_its_file`.
2. **A Stage 8 run still pulls, preflights, resumes and exports after this change.** A person
   reading the margin expects the five Stage 8 seeds to load unchanged. Tests: Task 1
   (`test_the_state_key_is_the_stage8_literal`, `test_default_settings_are_the_21_stage8_keys`),
   Task 3 (`test_a_checkpoint_without_the_frozen_hparam_loads_and_exports`).
3. **`data summaries --backbone thenlper/gte-small` without `--output` must not touch MiniLM's
   artifact.** Test: Task 5, `test_a_new_backbone_writes_its_own_path`.
4. **A selection run on a pull that lost one kept file refuses before it logs any read.** A
   person expects no wasted candidate read in the append-only log. Test: Task 8,
   `test_select_refuses_a_missing_kept_file_before_any_read`.
5. **A selection batch that died part-way and was run again must not mix reads.** The first
   batch's reads stay in the log. The person expects the rerun's choice to use only its own reads.
   Test: Task 8, `test_choose_refuses_reads_from_two_runs`.

## Decisions

- **D1** (owner, plan review, 2026-11-22): Plan B is planned after Plan A merges, from this
  plan's Handoff ("After Plan A"). Applied in: `## Handoff to Plan B`.
- **D2** (owner, plan review, 2026-11-22): only Plan B's header carries the roadmap line, so Plan
  A's completion stamps nothing ("Only Plan B has it"). Applied in: the header.
- **D3** (owner, plan review, 2026-11-22): every test stays on fixtures, and the two downloads are
  one final task gated on the owner's approval at the time ("Gated Task 11"). Applied in:
  Task 11.
- **D4** (owner, plan review, 2026-11-22): plan 22 approved as committed ("Approve, open PR"),
  to run subagent-driven in a fresh session ("Subagent-Driven"). Applied in: the header.
- **D5** (owner, plan review, 2026-11-22): the local Codex review's four fixes are accepted ("Accept,
  open PR"): one staged model cache for every pinned backbone, the margin reads every Stage 8
  seed, Task 11 downloads `main` and checks it against the pin, and Task 11's bundle check runs
  the worktree's code. Applied in: Task 4, Task 7, Task 11 and `## Handoff to Plan B`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/naics_embedder/utils/config.py` | `training.checkpoint_every_n_epochs`, `model.freeze_backbone`, the backbone tie | 1, 3, 4 |
| `src/naics_embedder/utils/training.py` | `RetainingCheckpoint`, `retained_epochs`, `refuse_another_kept_set`, `run_settings` | 1, 2, 3 |
| `src/naics_embedder/remote/canonical.py` | the pull's kept-set check | 2 |
| `src/naics_embedder/text_model/checkpoint_runner.py` | the Mac preflight's kept-set check, `SeedSelection.kept_epochs` | 2 |
| `src/naics_embedder/text_model/shared_encoder.py`, `naics_model.py`, `cli/commands/training.py` | the frozen switch | 3 |
| `src/naics_embedder/utils/input_window.py`, `remote/model_cache.py` | windows and revision pins | 4 |
| `src/naics_embedder/data/window_summaries.py`, `panels/window_summaries.py`, `cli/commands/data.py` | summaries at a given window, per-backbone paths | 5 |
| `src/naics_embedder/decision/ita_reads.py`, `cli/commands/tools.py` (`ita-read`) | the Stage 8 identity check, `--outcome` | 6 |
| `src/naics_embedder/decision/text_to_code.py`, `tools ita-margin` | the margin record | 7 |
| `conf/data/text_to_code_arms.yaml`, `src/naics_embedder/decision/ita_selection.py`, `tools ita-select` | arms, candidate reads, per-seed choice | 8 |
| `src/naics_embedder/decision/ita_decision.py`, `tools ita-decide` | the decision | 9 |
| `.claude/hooks/evidence_guard.py` | the three new commands need approval | 7, 8, 9 |

`cli/commands/tools.py` is 2,009 lines. The three new commands are thin adapters over the two new
decision modules, so they stay in `tools.py` with the other `ita-*` commands, after `ita-gap`.

**Parallel plan.** Plan 18 (`specs/plans/18-export-and-encoder-hardening.md`) is approved and not
yet executed. It edits `text_model/export.py`, `panels/selection_log.py`, `tools.py:1544-1676`
(`export-table`, `outcome-panel`) and `docs/usage.md:243-282`. This plan edits none of those
lines. Whichever plan executes second rebases on the first and re-runs its tests. Stage 7a works
under `graph_model/`, `decision/` and `conf/`. Before Task 1, run
`gh pr list --state open --json number,headRefName,files` and stop if an open PR touches a file
in the table above.

---

### Task 1: Checkpoint retention

**Files:**
- Modify: `src/naics_embedder/utils/config.py:1042-1085` (`TrainingConfig`)
- Modify: `src/naics_embedder/utils/training.py:252-296` (`run_settings`), `:585-607`
  (`outcome_checkpoint`), `:629-700` (`create_trainer`)
- Create: `tests/unit/test_checkpoint_retention.py`
- Modify: `tests/unit/test_utils_training.py`

**Interfaces:**
- Produces:
  - `TrainingConfig.checkpoint_every_n_epochs: Optional[int] = None` (`ge=1`). None keeps
    today's behaviour.
  - `retained_epochs(through: int, *, every: Optional[int], final: bool, best: int) -> List[int]`:
    sorted, unique. With `every` None: `[best]`. Otherwise every `e <= through` with
    `(e + 1) % every == 0`, plus `through` when `final`, plus `best`.
  - `class RetainingCheckpoint(ModelCheckpoint)`, built with `every: int`.
  - `outcome_checkpoint(checkpoint_dir: Path, *, every: Optional[int] = None) -> ModelCheckpoint`:
    the plain callback when `every` is None, else `RetainingCheckpoint`.
  - `run_settings` adds `'checkpoint_every_n_epochs': k` as its last key only when `k` is not
    None.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_checkpoint_retention.py`.

  Use a scripted LightningModule: one linear layer, a 2-step train loader, and
  `on_train_epoch_end` that logs `val/outcome_mrr` from a list (`self.log(OUTCOME_MRR,
  scores[self.current_epoch])`). Train it on CPU with `outcome_checkpoint(tmp_path, every=4)` and
  `outcome_early_stopping(patience)` only, `limit_val_batches=0`, `num_sanity_val_steps=0`,
  `logger=False`. `files(tmp_path)` returns the sorted epoch numbers of `epoch=NNN.ckpt` files and
  fails on any `-v` name.

```python
def test_retained_epochs_for_the_stage5_grid():
    assert retained_epochs(39, every=4, final=True, best=39) == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    assert retained_epochs(39, every=4, final=True, best=20) == [3, 7, 11, 15, 19, 20, 23, 27, 31, 35, 39]
    assert retained_epochs(17, every=4, final=True, best=12) == [3, 7, 11, 12, 15, 17]
    assert retained_epochs(17, every=4, final=False, best=12) == [3, 7, 11, 12, 15]
    assert retained_epochs(39, every=None, final=True, best=20) == [20]

def test_a_full_run_keeps_the_grid_the_last_and_the_best(tmp_path):
    # 12 epochs, rising to epoch 5, then flat: best is 5, grid 3, 7, 11, last 11
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] + [0.6] * 6
    fit(tmp_path, scores, max_epochs=12, patience=20)
    assert files(tmp_path) == [3, 5, 7, 11]
    assert (tmp_path / 'last.ckpt').is_file()

def test_a_beaten_best_on_the_grid_keeps_its_file(tmp_path):
    # Epoch 3 is best when saved and on the grid, then epoch 4 beats it: 3 stays, once
    scores = [0.1, 0.2, 0.3, 0.9, 0.95, 0.5, 0.5, 0.5]
    fit(tmp_path, scores, max_epochs=8, patience=20)
    assert files(tmp_path) == [3, 4, 7]

def test_a_beaten_best_off_the_grid_is_deleted(tmp_path):
    scores = [0.1, 0.9, 0.2, 0.2, 0.95, 0.2, 0.2, 0.2]
    fit(tmp_path, scores, max_epochs=8, patience=20)
    assert files(tmp_path) == [3, 4, 7]  # epoch 1, once best, is gone

def test_a_best_on_the_grid_is_one_file(tmp_path):
    scores = [0.1, 0.2, 0.3, 0.9] + [0.5] * 4
    fit(tmp_path, scores, max_epochs=8, patience=20)
    assert files(tmp_path) == [3, 7]

def test_an_early_stopped_run_keeps_its_stop_epoch(tmp_path):
    # Best at 2, patience 3: stops at epoch 5; kept 2 (best), 3 (grid), 5 (last)
    scores = [0.1, 0.2, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    fit(tmp_path, scores, max_epochs=10, patience=3)
    assert files(tmp_path) == [2, 3, 5]

def test_the_state_key_is_the_stage8_literal(tmp_path):
    literal = ("ModelCheckpoint{'monitor': 'val/outcome_mrr', 'mode': 'max', "
               "'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}")
    assert outcome_checkpoint(tmp_path).state_key == literal
    assert outcome_checkpoint(tmp_path, every=4).state_key == literal
```

  In `tests/unit/test_utils_training.py`:

```python
def test_default_settings_are_the_21_stage8_keys():
    assert list(run_settings(Config(), accelerator='cuda', precision='bf16-mixed')) == STAGE8_KEYS

def test_retention_enters_the_settings_only_when_set():
    cfg = Config().override({'training.checkpoint_every_n_epochs': 4})
    settings = run_settings(cfg, accelerator='cuda', precision='bf16-mixed')
    assert list(settings) == [*STAGE8_KEYS, 'checkpoint_every_n_epochs']
    assert settings['checkpoint_every_n_epochs'] == 4

def test_exact_resume_refuses_a_changed_retention():
    # Saved with 4, now default; and saved default, now 4: both refused, naming the key
    ...  # two pytest.raises(ValueError, match='checkpoint_every_n_epochs') around
         # refuse_a_resume_under_other_settings
```

  `STAGE8_KEYS` is the literal 21-name list in Global Constraints. Also extend the existing
  `create_trainer` test to assert that `cfg.training.checkpoint_every_n_epochs=4` gives a
  `RetainingCheckpoint` and the default gives exactly `ModelCheckpoint`.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_checkpoint_retention.py tests/unit/test_utils_training.py -q`
  Expected: FAIL with `ImportError` on `retained_epochs` and `RetainingCheckpoint`.

- [ ] **Step 3: Implement the setting, `retained_epochs`, `RetainingCheckpoint` and the wiring**

  `RetainingCheckpoint` overrides three members. The order inside the epoch-end save is the point:
  the grid file is written before `last.ckpt`, so a crash between them never leaves a `last.ckpt`
  whose epoch lacks its kept file.

```python
class RetainingCheckpoint(ModelCheckpoint):
    @property
    def state_key(self) -> str:
        # The parent's literal key: every guard reads Stage 8 state under it
        ...  # ModelCheckpoint's _generate_state_key output with the class name 'ModelCheckpoint'

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Parent body, then the grid save between the top-1 save and the last save:
        # if not skip and save-on-epoch-end: candidates; _save_topk_checkpoint;
        # if epoch is on the grid, or trainer.should_stop, or epoch + 1 == trainer.max_epochs,
        # and epoch=NNN.ckpt does not exist: self._save_checkpoint(trainer, that path);
        # _save_last_checkpoint
        ...

    def _remove_checkpoint(self, trainer, filepath: str) -> None:
        # Skip a file named epoch=NNN.ckpt whose NNN is on the grid; remove anything else
        ...
```

  `create_trainer` passes `every=cfg.training.checkpoint_every_n_epochs`. Every other caller of
  `outcome_checkpoint` (the three resume and kept-set guards) keeps calling it with no `every`:
  the state key is the same either way. Update `outcome_checkpoint`'s docstring and
  `run_settings`'s ("21 keys, and the retention key when it is set").

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest tests/unit/test_checkpoint_retention.py tests/unit/test_utils_training.py tests/unit/test_cli_training.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/naics_embedder/utils/config.py src/naics_embedder/utils/training.py tests/unit/test_checkpoint_retention.py tests/unit/test_utils_training.py
git commit -m 'feat(training): keep every 4th epoch, the last and the outcome-best under one callback'
```

---

### Task 2: The kept set is authoritative on pull and preflight

**Files:**
- Modify: `src/naics_embedder/utils/training.py` (new `refuse_another_kept_set`, after
  `refuse_a_resume_of_a_stopped_run`)
- Modify: `src/naics_embedder/remote/canonical.py:218-313` (`validate_continuation_set`)
- Modify: `src/naics_embedder/text_model/checkpoint_runner.py:44-52` (`SeedSelection`),
  `:111-184` (`_selection`)
- Modify: `tests/unit/test_remote_canonical.py`, `tests/unit/test_checkpoint_runner.py`

**Interfaces:**
- Consumes: `retained_epochs` (Task 1).
- Produces:
  - `refuse_another_kept_set(directory: Path, last: Mapping[str, Any], best_epoch: int) -> List[int]`.
    It reads `every` from `last['hyper_parameters']['run_settings'].get('checkpoint_every_n_epochs')`.
    With None it returns `[best_epoch]` and checks nothing new (today's checks stay where they
    are). Otherwise `final` is true when the saved EarlyStopping state's `stopped_epoch` is above
    0 or `last['epoch'] + 1 == run_settings['max_epochs']`. It requires the set of epochs named
    by `epoch=*.ckpt` files (any `-v` name counts as extra) to equal
    `retained_epochs(last['epoch'], every=every, final=final, best=best_epoch)`, and returns it.
  - Its refusal: `ValueError('the kept checkpoints are not the retained set: missing [..],
    extra [..]')`, with file names in the extra list.
  - `SeedSelection.kept_epochs: Tuple[int, ...] = ()`, the returned list.

- [ ] **Step 1: Write the failing tests**

  Build a retained run directory with Task 1's scripted module and `fit` helper (move both to
  `tests/fixtures/retention.py` so both test files import them), or with
  `tests/fixtures/remote.py`'s builder where the canonical tests need its state shape.

```python
def test_preflight_returns_the_retained_epochs(...):
    assert runner.check(spec, seed).kept_epochs == (3, 5, 7, 11)

def test_preflight_refuses_a_missing_kept_file(...):
    (directory / 'epoch=007.ckpt').unlink()
    with pytest.raises(ValueError, match=r'missing \[7\]'):
        runner.check(spec, seed)

def test_preflight_refuses_an_extra_kept_file(...):
    shutil.copy(directory / 'epoch=003.ckpt', directory / 'epoch=003-v1.ckpt')
    with pytest.raises(ValueError, match='extra'):
        runner.check(spec, seed)

def test_continuation_refuses_a_missing_or_extra_kept_file(...):
    # The same two cases through validate_continuation_set
    ...

def test_a_run_without_retention_keeps_todays_checks(...):
    # The existing one-file fixtures (tests/fixtures/remote.py:288-308,
    # test_checkpoint_runner.py:79, :151) pass unchanged, and kept_epochs == (selected,)
    ...

def test_an_early_stopped_retained_run_passes(...):
    assert runner.check(spec, seed).kept_epochs == (2, 3, 5)
```

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_checkpoint_runner.py tests/unit/test_remote_canonical.py -q`
  Expected: FAIL on the new tests (`kept_epochs` missing, no refusal).

- [ ] **Step 3: Implement `refuse_another_kept_set` and call it** in `_selection` after the
  selected checkpoint's score check, and in `validate_continuation_set` after `selected_epoch`.

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest tests/unit/test_checkpoint_runner.py tests/unit/test_remote_canonical.py tests/unit/test_remote_sync.py tests/integration/test_remote_workflow.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/naics_embedder/utils/training.py src/naics_embedder/remote/canonical.py src/naics_embedder/text_model/checkpoint_runner.py tests/
git commit -m 'feat(remote): refuse a pull or preflight whose kept checkpoints are not the retained set'
```

---

### Task 3: The frozen control and Stage 8 compatibility

**Files:**
- Modify: `src/naics_embedder/utils/config.py:843-875` (`ModelConfig`)
- Modify: `src/naics_embedder/text_model/shared_encoder.py:128-201`
- Modify: `src/naics_embedder/text_model/naics_model.py:185-320`
- Modify: `src/naics_embedder/cli/commands/training.py:85-130` (`build_model_from_config`)
- Modify: `src/naics_embedder/utils/training.py` (`run_settings`)
- Modify: `tests/unit/test_encoder.py`, `tests/unit/test_naics_model.py`,
  `tests/unit/test_utils_training.py`, `tests/unit/test_checkpoint_runner.py`

**Interfaces:**
- Produces:
  - `ModelConfig.freeze_backbone: bool = False`.
  - `SharedEncoder(..., freeze_backbone: bool = False)` and
    `NAICSContrastiveModel(..., freeze_backbone: bool = False)`, saved by
    `save_hyperparameters` like every other constructor argument.
  - `run_settings` adds `'freeze_backbone': True`, after the retention key, only when it is True.

  The switch keeps the LoRA adapters and sets `requires_grad_(False)` on every parameter of
  `self.backbone`, adapters included. LoRA's B starts at zero, so a frozen adapter adds nothing,
  and the state dict, `shared_encoder_architecture` and the checkpoint contract stay d16's. The
  backbone still follows `.train()` and `.eval()`, so its dropout acts as in d16: the only factor
  is which weights train. `constructor_settings` does not change: `load_from_checkpoint` rebuilds
  the switch from the saved hyperparameters, and `run_settings` refuses a changed switch on
  resume.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_frozen_backbone_does_not_change_over_a_step(...):
    # tiny-bert SharedEncoder(freeze_backbone=True); one AdamW step on a loss of its output
    assert all(torch.equal(before[name], after[name]) for name in backbone_names)
    assert not torch.equal(before_projection, after_projection)

def test_an_unfrozen_backbone_trains_its_adapters(...):
    # The same step with the default: some lora_ parameter changes

def test_freeze_enters_the_settings_only_when_set():
    cfg = Config().override({'model.freeze_backbone': True})
    assert list(run_settings(cfg, accelerator='cuda', precision='bf16-mixed'))[-1] == 'freeze_backbone'

def test_exact_resume_refuses_a_changed_freeze(): ...

def test_a_frozen_checkpoint_rebuilds_frozen(...):
    # Save a frozen tiny model's checkpoint, load_arm_model it: every backbone parameter has
    # requires_grad False, and the exported table equals a direct encode

def test_a_checkpoint_without_the_frozen_hparam_loads_and_exports(trained_seeds, ...):
    # Copy a trained_seeds checkpoint, delete hyper_parameters['freeze_backbone'], save it.
    # load_arm_model and export_code_table succeed, ArmEncoder.from_files reads the table, and
    # refuse_a_resume_under_other_settings against run_settings(default cfg) and
    # refuse_other_constructor_settings pass
```

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_encoder.py tests/unit/test_naics_model.py tests/unit/test_utils_training.py -q`
  Expected: FAIL with `TypeError: ... unexpected keyword argument 'freeze_backbone'`.

- [ ] **Step 3: Implement the switch** through `ModelConfig`, `build_model_from_config`,
  `NAICSContrastiveModel` and `SharedEncoder`, and the `run_settings` key. The encoder's startup
  log line names the switch.

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest tests/unit/test_encoder.py tests/unit/test_naics_model.py tests/unit/test_utils_training.py tests/unit/test_checkpoint_runner.py tests/unit/test_cli_training.py tests/unit/test_arm_encoder.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/naics_embedder tests
git commit -m 'feat(text_model): add a frozen-backbone control that keeps the d16 contract'
```

---

### Task 4: The two backbones' windows, revisions and the backbone tie

**Files:**
- Modify: `src/naics_embedder/utils/input_window.py:1-12, :19` (docstring and `TRAINED_WINDOWS`)
- Modify: `src/naics_embedder/remote/model_cache.py:14-16, :35-38, :82-85`
- Modify: `src/naics_embedder/utils/config.py` (a `model_validator` on `Config`)
- Modify: `src/naics_embedder/remote/launch.py:240-258` (`_session_model_cache`: subset check)
- Modify: `tests/unit/test_input_window.py`, `tests/unit/test_remote_model_cache.py`,
  `tests/unit/test_config.py`, `tests/unit/test_cli_training.py:405-416`,
  `tests/unit/test_tokenization_cache.py`, `tests/unit/test_remote_launch.py`

**Interfaces:**
- Produces:
  - `TRAINED_WINDOWS['thenlper/gte-small'] = 512`, `TRAINED_WINDOWS['BAAI/bge-small-en-v1.5'] = 512`.
  - `PINNED_REVISIONS: Dict[str, str]` in `model_cache.py`, holding the three revisions of Global
    Constraints. `MINILM` and `MINILM_REVISION` stay as names. `verify_model_cache` and
    `stage_model_cache` refuse a name that is not in `PINNED_REVISIONS` ("no revision is pinned
    for {name}") and a revision that differs from its pin.
  - **One cache for every pinned backbone.** Today `remote up` re-verifies an existing
    `.remote/model-cache/hub` against the new config's model names, so a MiniLM cache refuses a
    gte config (`offline model cache inventory or SHA256 differs`), on the Mac and on the
    instance. Instead, `stage_model_cache(root, cfg, *, hub=None)` stages every name in
    `PINNED_REVISIONS`, whatever the config, and requires the config's three names to be among
    them. Launch's `_session_model_cache` (`remote/launch.py:240-258`) requires the config's names
    to be a subset of `cache['revisions']`, not equal to it. Each arm then reuses one verified
    cache. Nothing is repaired: an existing staged cache with another name set is refused with
    "the staged offline model cache holds {names}, not every pinned backbone: remove
    .remote/model-cache and run remote up again".
  - `Config` validator `the_backbone_is_the_tokenizer`: refuses unless `model.base_model_name`,
    `data_loader.tokenization.tokenizer_name` and `data_loader.streaming.tokenizer_name` are one
    name. The message names all three. The tokenization validators already refuse a name with no
    trained window, so the tie also binds the training backbone to `TRAINED_WINDOWS`.

  Record in `input_window.py`'s module docstring, beside MiniLM's paragraph, what the cards give.
  For gte-small at `17e1f347…`, the card's limitation says texts "will be truncated to a maximum
  of 512 tokens". `config.json` gives `max_position_embeddings` 512 and
  `sentence_bert_config.json` gives `max_seq_length` 512. `tokenizer_config.json`'s
  `model_max_length` is the unset sentinel 1e30, so it is not evidence. For bge-small-en-v1.5 at
  `5c38ec7c…`, the card's table gives sequence length 512, its example tokenizes at
  `max_length=512`, and `max_position_embeddings`, `max_seq_length` and `model_max_length` are
  512. Both have vocabulary size 30,522, as MiniLM has. The pooling configs differ: gte mean,
  bge CLS (`1_Pooling/config.json`). The encoder mean-pools every arm (spec §4.3).

  **The bundle's backbone does not bind training** (spec §4.3, my call there):
  `manifest.input_window.backbone` names the window the bundle was built under, and the bundle's
  targets and texts do not depend on it. The tie above and the token cache's summaries check are
  the runtime checks. This closes deferred § 7's design question. Its "verified against bundle
  301cce28" part is Task 11's local check.

  **The summaries-key watch** (deferred § 9) closes as done: the tie makes the tokenizer's name
  the backbone's for every training config, so `export.py`'s and `text_only.py`'s summaries keys
  cannot split for a trained arm.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize('backbone', [MINILM, GTE_SMALL, BGE_SMALL])
def test_each_backbone_accepts_length_128(backbone):
    cfg = Config().override(tie(backbone) | {'data_loader.tokenization.max_length': 128,
                                             'data_loader.streaming.max_length': 128})
    assert cfg.data_loader.tokenization.max_length == 128

def test_a_backbone_without_a_window_is_refused():
    with pytest.raises(ValidationError, match='no trained input window is recorded'):
        Config().override(tie('bert-base-uncased'))

def test_a_tokenizer_other_than_the_backbone_is_refused():
    with pytest.raises(ValidationError, match='one name'):
        Config().override({'model.base_model_name': GTE_SMALL})

def test_an_over_window_text_without_a_pin_is_refused_for_a_new_backbone(...):
    # tie(GTE_SMALL) at 128 on the fixture descriptions, WINDOW_SUMMARIES without GTE_SMALL,
    # a tokenizer stub that counts one long text over 128: the token cache build raises
    # "no window summaries are pinned for thenlper/gte-small"

def test_the_cache_refuses_an_unpinned_name(...): ...
def test_the_cache_refuses_a_new_backbone_at_another_revision(...): ...
def test_the_cache_accepts_each_pinned_backbone(...): ...
def test_one_staged_cache_launches_each_backbone(...):
    # fake hub with the three pinned snapshots: stage once, then _session_model_cache accepts a
    # MiniLM, a gte and a bge config from the same record
def test_a_staged_cache_with_another_name_set_is_refused(...):
    # an existing MiniLM-only .remote/model-cache: stage_model_cache raises 'remove .remote/model-cache'
```

  `tie(name)` returns the three overrides. Update `test_cli_training.py:405-416`, which overrides
  only `model.base_model_name=other/backbone`: it now passes the tie, and monkeypatches
  `TRAINED_WINDOWS` for `other/backbone`.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_input_window.py tests/unit/test_config.py tests/unit/test_remote_model_cache.py -q`
  Expected: FAIL (no window for gte-small, no tie refusal, unpinned names pass).

- [ ] **Step 3: Implement the windows, the pins, the one cache and the tie.** Launch's effective
  config is a `Config`, so the tie runs on the Mac before launch with no launch change. Add one
  test to `test_remote_launch.py` that renders an untied override and expects the refusal.
  Update `.claude/rules/remote.md`'s model-cache sentence: every pinned backbone is staged.

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest -n auto -q`
  Expected: PASS except the § 19 failure. Fix any other test that built an untied config.

- [ ] **Step 5: Commit**

```bash
git add src/naics_embedder tests
git commit -m 'feat(text_model): record gte-small and bge-small windows and pins, and tie the tokenizer to the backbone'
```

---

### Task 5: Summaries at a given window, at a path of their own

**Files:**
- Modify: `src/naics_embedder/data/window_summaries.py:255-333` (`generate_window_summaries`)
- Modify: `src/naics_embedder/panels/window_summaries.py:48` (paths)
- Modify: `src/naics_embedder/cli/commands/data.py:235-284` (`summaries`)
- Modify: `tests/unit/test_window_summaries_build.py`, `tests/unit/test_committed_window_summaries.py`

**Interfaces:**
- Produces:
  - `summaries_path(backbone: str) -> str` in `panels/window_summaries.py`: MiniLM's is
    `WINDOW_SUMMARIES_PATH` (`conf/data/window_summaries.csv`). Another backbone's is
    `conf/data/window_summaries_{slug}.csv`, with `slug = backbone.replace('/', '__')`, for example
    `conf/data/window_summaries_thenlper__gte-small.csv`. The provenance stays `provenance_path`
    of the CSV, so it moves with it.
  - `generate_window_summaries(descriptions_path, output_path, *, backbone, window: Optional[int] = None, force=False, model=None, tokenizer=None, revision=None) -> SummariesPin`:
    `window = check_window(backbone, window)`, so None is the trained window and a window above
    it is refused. The pin records `window`. It refuses an `output_path` that a `WINDOW_SUMMARIES`
    pin of another backbone names ("{path} is the pinned artifact of {other}"), with or without
    `force`.
  - `data summaries --window N` (default None), and `--output` defaulting to
    `summaries_path(backbone)` after the backbone resolves.

- [ ] **Step 1: Write the failing tests** (the stub backbone and `WINDOW` of
  `test_window_summaries_build.py:168` apply)

```python
def test_a_window_below_the_trained_window_is_written_and_pinned(...):
    pin = generate_window_summaries(..., backbone=STUB, window=WINDOW - 8)
    assert pin.window == WINDOW - 8
    assert all(row['window'] == WINDOW - 8 for row in read_window_summaries(Path(pin.path)).iter_rows(named=True))

def test_a_window_above_the_trained_window_is_refused(...):
    with pytest.raises(ValueError, match='exceeds the trained input window'):
        generate_window_summaries(..., backbone=STUB, window=WINDOW + 1)

def test_no_window_is_the_trained_window(...):
    assert generate_window_summaries(..., backbone=STUB).window == WINDOW

def test_a_new_backbone_writes_its_own_path(...):
    # CLI with --backbone thenlper/gte-small and no --output, in a tmp checkout holding a copy of
    # conf/data/window_summaries.csv: the new file is conf/data/window_summaries_thenlper__gte-small.csv,
    # and MiniLM's bytes are unchanged
    ...

def test_another_backbones_pinned_path_is_refused_even_with_force(...): ...

def test_minilm_default_artifact_is_unchanged():
    # test_committed_window_summaries.py: MiniLM's pin path, sha256 and window 128 stay
    ...
```

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_window_summaries_build.py tests/unit/test_committed_window_summaries.py -q`
  Expected: FAIL (`window` is not a parameter).

- [ ] **Step 3: Implement the parameter, the path function and the CLI options.** Update the
  command's docstring Output and Example (`--backbone thenlper/gte-small --window 128`).

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest tests/unit/test_window_summaries_build.py tests/unit/test_committed_window_summaries.py tests/unit/test_window_summaries.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/naics_embedder tests
git commit -m 'feat(data): write window summaries at a given window, at a path per backbone'
```

---

### Task 6: The d16 identity check and `tools ita-read`'s `--outcome`

**Files:**
- Modify: `src/naics_embedder/decision/ita_reads.py` (new `check_d16_seed`)
- Modify: `src/naics_embedder/cli/commands/tools.py:1705-1920` (`_check_ita_arm`, `ita_read_command`)
- Modify: `tests/unit/test_ita_reads.py`, `tests/unit/test_cli_commands.py` (the `ita-read` tests)

**Interfaces:**
- Produces:
  - `check_d16_seed(stage8: ArmRecord, seed: int, checkpoint_sha256: str, table_fingerprint: str) -> None`:
    raises `ValueError` when the record has no run of `seed`, or its
    `run.checkpoint.sha256` or `run.table.matrix_fingerprint` differs, naming the seed and both
    values. The comparisons are `ita_gap.gap_seeds`' (`decision/ita_gap.py:170-181`).
  - `tools ita-read --stage8 PATH`: required on `euclidean-d16`, refused on the stub. The d16
    encoder is built, then `check_d16_seed` runs on its `checkpoint_sha256` and
    `table_fingerprint`, before any log record and also under `--dry-run`.
  - `--outcome/--no-outcome` becomes `Optional[bool] = None`. On the stub, None means `--outcome`,
    so its record carries C unless `--no-outcome` is passed. On `euclidean-d16`, `--outcome` is
    refused before any check: "a euclidean-d16 read takes no --outcome: its C is the Stage 8
    record's stored scores (P4)".

- [ ] **Step 1: Write the failing tests**

```python
def test_check_d16_seed_refuses_another_checkpoint(stage8_record): ...
def test_check_d16_seed_refuses_another_table(stage8_record): ...
def test_check_d16_seed_refuses_a_seed_the_record_lacks(stage8_record): ...
def test_ita_read_refuses_outcome_on_d16_before_any_log(...):
    # exit 1, the log file is unchanged (byte-equal)
def test_ita_read_refuses_a_d16_checkpoint_other_than_stage8_before_any_log(...):
    # exit 1, the log is unchanged, the output is not written
def test_ita_read_stub_reads_the_outcome_by_default(...):
    # exit 0, two log records, record.outcome is not None
```

  Build `stage8_record` as a small `ArmRecord` fixture whose seed-1 run names the fixture
  checkpoint's SHA-256 and table fingerprint (reuse the record builder that `test_ita_gap.py`
  uses).

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_ita_reads.py tests/unit/test_cli_commands.py -q -k 'ita'`
  Expected: FAIL (`check_d16_seed` missing, `--stage8` unknown).

- [ ] **Step 3: Implement `check_d16_seed` and the two CLI changes.** Update the command's
  docstring, its Example and plan 20's d16 command shape in `docs/usage.md` (`--stage8`).

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest tests/unit/test_ita_reads.py tests/unit/test_ita_gap.py tests/unit/test_cli_commands.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/naics_embedder tests docs/usage.md
git commit -m 'fix(cli): check a d16 read against the Stage 8 record and read the stub with its outcome'
```

---

### Task 7: The margin record

**Files:**
- Modify: `src/naics_embedder/decision/text_to_code.py:260-432` (`ItaMarginRecord`,
  `fix_ita_margin`, `check_ita_margin`)
- Modify: `src/naics_embedder/cli/commands/tools.py` (new `ita-margin`, after `ita-gap`)
- Modify: `.claude/hooks/evidence_guard.py:42-56`, `tests/unit/test_claude_hooks.py:86-110`
- Modify: `tests/unit/test_decision_text_to_code.py`, `tests/unit/test_cli_commands.py`

**Interfaces:**
- Consumes: `check_d16_seed` (Task 6).
- Produces:
  - `class MarginRead(BaseModel)` (frozen, `extra='forbid'`): `arm: str`, `seed: Optional[int]`,
    `mrr: float`, `per_pair: ArtifactRef`, `checkpoint_sha256: Optional[str]`,
    `table_fingerprint: Optional[str]`.
  - `ItaMarginRecord` gains `reads: List[MarginRead]`, after `log_records`. No margin record
    exists yet, so the schema change breaks nothing stored.
  - `fix_ita_margin(panel, reference_arms, r, name, *, purpose, store: ArtifactStore, stage8: ArmRecord, data_fingerprint: str) -> ItaMarginRecord`.
    Before any read, it refuses: `panel.data_fingerprint(ItaPart.VALIDATION) != data_fingerprint`;
    a `euclidean-d16` encoder without `checkpoint_sha256` or `table_fingerprint`; d16 seeds that
    are not exactly the seeds of `stage8.runs` ("the margin reads every Stage 8 seed: missing
    [..], extra [..]"); and any d16 seed that fails `check_d16_seed`. Each d16 read logs `checkpoint` and `table` in its detail, as
    `read_reference` does. Each read's `per_pair` goes to `store` as `ita_per_pair.parquet`.
    It uses `panel.score_logged`.
  - `check_ita_margin` also refuses: a record whose `log_records` lack a read of either reference
    arm; a repeated `(arm, seed)` among `log_records`; a d16 read with no `checkpoint` or `table`;
    and `reads` that do not match `log_records` one for one in arm, seed, checkpoint and table.
  - `tools ita-margin`: `--pairs`, `--pair-codes`, `--descriptions` (the stub's, checked against
    the bundle as `ita-read` checks them), repeatable `--seed`/`--checkpoint`/`--table` (equal
    counts, aligned), `--stage8`, `--data-fingerprint`, `--r` (default 0.05), `--name`,
    `--purpose`, `--store`, `--output`, `--log` (required, absolute), `--evidence`, `--settings`,
    `--config`, `--dry-run`, overrides. The C1 check (`check_ita_run`) and every refusal above
    come before any log record. `--dry-run` runs every check and prints the pair count and both
    fingerprints, and logs nothing.
  - `PANEL_READS` gains `ita-margin`, `ita-select` and `ita-decide`. `DRY_RUN_COMMANDS` gains
    `ita-margin` and `ita-select`.

- [ ] **Step 1: Write the failing tests**

```python
def test_fix_refuses_a_d16_seed_not_in_stage8_before_any_read(...):
    # the log is byte-equal after the ValueError
def test_fix_refuses_another_table_build_before_any_read(...): ...
def test_fix_refuses_a_missing_stage8_seed_before_any_read(...):
    # four of the record's five seeds: ValueError naming missing [5], the log is byte-equal
def test_fix_logs_each_d16_checkpoint_and_table(...):
    assert [r['detail'].get('checkpoint') for r in margin.log_records][1:] == [...]
def test_fix_stores_each_reads_per_pair_frame(...):
    assert store.read_frame(margin.reads[0].per_pair).height == validation_pairs
def test_check_refuses_a_record_without_both_reference_arms(...):
    # model_copy with the stub's log_records and reads removed
def test_check_refuses_a_repeated_seed(...): ...
def test_check_refuses_reads_that_differ_from_log_records(...): ...
def test_margin_cli_requires_an_absolute_log(...): ...
def test_margin_cli_dry_run_logs_nothing(...): ...
```

  In `test_claude_hooks.py`, add the three commands to `PANEL_READS` and the two dry runs to the
  pass list.

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_decision_text_to_code.py tests/unit/test_claude_hooks.py -q`
  Expected: FAIL.

- [ ] **Step 3: Implement the record fields, the two functions, the command and the hook lists.**
  Update the module docstring's Margin paragraph and the hook's docstring.

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest tests/unit/test_decision_text_to_code.py tests/unit/test_claude_hooks.py tests/unit/test_cli_commands.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/naics_embedder .claude/hooks tests
git commit -m 'feat(decision): fix the ITA margin from Stage 8-checked reads, with their frames stored'
```

---

### Task 8: The arms, the candidate reads and each seed's choice

**Files:**
- Create: `conf/data/text_to_code_arms.yaml`
- Create: `src/naics_embedder/decision/ita_selection.py`
- Modify: `src/naics_embedder/cli/commands/tools.py` (new `ita-select`)
- Create: `tests/unit/test_ita_selection.py`

**Interfaces:**
- Consumes: `SeedSelection.kept_epochs` (Task 2), `check_ita_margin_first`, `export_code_table`,
  `ArmEncoder.from_files`, `ItaPanel.score_logged`.
- Produces:
  - `conf/data/text_to_code_arms.yaml`: a list of four arms in tie order. Each has `name`, `role`
    (`candidate` or `control`) and `overrides` (the `key=value` strings that turn
    `conf/config.yaml` into the arm). Every arm sets `model.geometry=euclidean`,
    `model.dimension=16`, `training.checkpoint_every_n_epochs=4`, both `max_length=128`, and its
    backbone's three tied names. `frozen-minilm-d16` adds `model.freeze_backbone=true`. The arm's
    `experiment_name` is Plan B's.
  - `class StageArm(BaseModel)`: `name`, `role: Literal['candidate', 'control']`,
    `overrides: List[str]`. `load_stage5_arms(path: str = 'conf/data/text_to_code_arms.yaml') -> List[StageArm]`
    refuses a name in `REFERENCE_ARMS`, a repeated name, and other than exactly one control.
  - `CANDIDATE_ROLE = 'candidate'`, `STAGE5_SEEDS = (1, 2, 3, 4, 5)`.
  - `class ItaCandidateRead(BaseModel)` (frozen, `extra='forbid'`): `kind='ita_candidate_read'`,
    `arm`, `seed`, `epoch`, `control: bool`, `run`, `purpose`, `evidence`, `split_fingerprint`,
    `data_fingerprint`, `headline`, `copied`, `phrasing`, `counts`, `per_pair: ArtifactRef`,
    `checkpoint_sha256`, `table_fingerprint`, `log_records`, `created_at`.
  - `read_candidate(panel, encoder, store, evidence, *, arm: StageArm, seed: int, epoch: int, run: str, purpose: str) -> ItaCandidateRead`:
    refuses, before any log record, a seed outside `STAGE5_SEEDS`, a negative epoch, a blank run
    or purpose, and an encoder without its two identities. Detail: `arm`, `role: 'candidate'`,
    `run`, `seed`, `epoch`, `checkpoint`, `table`, and `control: true` for the control only.
  - `class SeedChoice(BaseModel)`: `kind='ita_seed_choice'`, `arm`, `seed`, `run`,
    `kept_epochs: List[int]`, `reads: List[ItaCandidateRead]`, `epoch`, `mrr`,
    `data_fingerprint`.
  - `choose_seed_epoch(reads: Sequence[ItaCandidateRead], kept_epochs: Sequence[int]) -> SeedChoice`:
    refuses reads that cover other than `kept_epochs` once each, or name more than one arm, seed,
    run or data fingerprint. The chosen epoch is the earliest with the highest `headline['mrr']`.
  - `refuse_ita_monitor_reads(records: Sequence[Mapping[str, Any]]) -> None`: refuses any record
    of a run's `monitor_reads.jsonl` whose `read['panel']` is `ita_text`, naming its epoch.
    Under U49 no ITA read is logged off the Mac, so this is the "across logs" check of deferred
    § 17.
  - `tools ita-select`: `--arm NAME` (from the arms file), repeatable `--seed`, `--runs` (a
    template with `{seed}`, as `tools sweep` takes it), `--margin`, `--pairs`, `--pair-codes`,
    `--data-fingerprint`, `--purpose`, `--store`, `--output-dir`, `--log` (required, absolute),
    `--accelerator` (default `cuda`), `--evidence`, `--settings`, `--arms`, `--dry-run`. Order:
    1. Before any read: C1; `check_ita_margin_first(margin, log)`; the margin's
       `data_fingerprint` equals `--data-fingerprint` and the panel's; each output
       `<output-dir>/<arm>-s<seed>.json` is new; for each seed, `CheckpointRunner.check` (the
       kept set, Task 2) and `refuse_ita_monitor_reads`. A dry run stops here and prints each
       seed's kept epochs.
    2. One run id for the command, `ita-select/<arm>/<uuid4 hex>`.
    3. Per seed, per kept epoch in order: `export_code_table` to
       `<run>/arm_table_epoch=NNN.parquet` (the same function `tools export-table` calls), then
       `ArmEncoder.from_files`, then `read_candidate`.
    4. Per seed, after all its reads: `choose_seed_epoch`, then `write_record`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_ita_selection.py` (fixture panel of
  `tests/fixtures/ita.py`, fixture encoders carrying `checkpoint_sha256` and
  `table_fingerprint`)

```python
def test_the_committed_arms_file_names_the_four_arms_in_tie_order():
    assert [a.name for a in load_stage5_arms()] == ['euclidean-d16-ita', 'gte-small-d16', 'bge-small-d16', 'frozen-minilm-d16']
    assert [a.role for a in load_stage5_arms()].count('control') == 1
    # each arm's overrides validate as a Config with max_length 128 and retention 4

def test_arms_refuse_a_reference_name(tmp_path): ...
def test_a_candidate_read_logs_role_candidate(...):
    assert record.log_records[0]['detail']['role'] == 'candidate'
    assert 'control' not in record.log_records[0]['detail']
def test_a_control_read_logs_control_true(...): ...
def test_a_candidate_read_counts_against_a_later_margin(...):
    # check_ita_margin_first refuses a margin fixed after this read
def test_choose_takes_the_earliest_epoch_with_the_highest_mrr(...):
    # epochs [3, 7, 11, 12] with MRRs [0.2, 0.4, 0.4, 0.3] -> 7
def test_choose_refuses_a_missing_or_repeated_epoch(...): ...
def test_choose_refuses_reads_from_two_runs(...): ...
def test_monitor_reads_with_an_ita_record_are_refused(...): ...
def test_select_refuses_a_missing_kept_file_before_any_read(...):
    # a retained fixture run with one kept file removed: exit 1, the log is byte-equal
def test_select_refuses_without_a_margin_first(...):
    # a candidate read already logged before the margin: exit 1 before any new read
def test_select_reads_every_kept_epoch_then_writes_each_choice(...):
    # one retained tiny run (Task 2's fixture), seed 1: reads == kept epochs, one record written
```

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_ita_selection.py -q`
  Expected: FAIL with `ModuleNotFoundError: naics_embedder.decision.ita_selection`.

- [ ] **Step 3: Implement the arms file, the module and the command.**

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest tests/unit/test_ita_selection.py tests/unit/test_decision_text_to_code.py tests/unit/test_claude_hooks.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add conf/data/text_to_code_arms.yaml src/naics_embedder tests
git commit -m 'feat(decision): read each kept epoch on ITA validation and choose each seed earliest-best'
```

---

### Task 9: The decision

**Files:**
- Create: `src/naics_embedder/decision/ita_decision.py`
- Modify: `src/naics_embedder/decision/text_to_code.py:83-258` (type hints only:
  `ScoredSeed` protocol)
- Modify: `src/naics_embedder/cli/commands/tools.py` (new `ita-decide`)
- Create: `tests/unit/test_ita_decision.py`

**Interfaces:**
- Consumes: `ItaMarginRecord.reads` (Task 7), `SeedChoice` and `load_stage5_arms` (Task 8),
  `compare_to_reference`.
- Produces:
  - `class ScoredSeed(Protocol)`: `per_pair: pl.DataFrame`, `data_fingerprint: Optional[str]`.
    `headline_values`, `arm_replicates` and `compare_to_reference` take it; `ItaResult` satisfies
    it. `StoredSeed` (frozen dataclass with those two fields) is built from a stored frame.
  - `class ArmScore(BaseModel)`: `arm`, `role`, `mrr`, `seeds: List[int]`, `epochs: List[int]`.
  - `class CandidateVerdict(BaseModel)`: `candidate`, `delta`, `interval: Tuple[float, float]`,
    `level`, `beats: bool`.
  - `class ItaDecisionRecord(BaseModel)`: `kind='ita_decision'`, `margin_name`, `margin_run_id`,
    `reference`, `margin`, `split_fingerprint`, `data_fingerprint`, `scores: List[ArmScore]`,
    `verdicts: List[CandidateVerdict]`, `forward: Optional[str]`, `replicates`,
    `bootstrap_seed`, `decided_at`.
  - `decide_text_to_code(margin, log, store, choices: Mapping[str, Sequence[SeedChoice]], arms: Sequence[StageArm], *, replicates: int, bootstrap_seed: int) -> ItaDecisionRecord`.
    Steps:
    1. `check_ita_margin_first(margin, log)`.
    2. `choices` holds every arm of `arms`, each with exactly the seeds `STAGE5_SEEDS`.
    3. Every chosen read's `data_fingerprint` equals `margin.data_fingerprint`: "the margin was
       fixed on data fingerprint X, but arm A seed S was read on Y" (deferred § 17, item 1).
    4. The reference's seeds are `margin.reads` of `margin.reference`, read back with
       `store.read_frame`. Each candidate's seeds are its chosen epoch's stored frames.
    5. Each candidate gets `compare_to_reference`. The control gets a score and no verdict.
    6. `forward` is the beating candidate with the highest point MRR. A tie goes to the earlier
       arm in `arms` (the file's tie order). None when no candidate beats the reference.
  - `tools ita-decide`: `--margin`, repeatable `--choice` (record paths), `--store`, `--log`
    (required, absolute), `--arms`, `--output`, `--replicates` (default 10,000),
    `--bootstrap-seed` (default 0). It reads no panel.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_ita_decision.py`. Build frames with
  known reciprocal ranks so Δ's interval sits clearly above or below δ.

```python
def test_the_decision_refuses_a_margin_on_another_build(...):
    with pytest.raises(ValueError, match='margin was fixed on data fingerprint'):
        decide_text_to_code(...)
def test_the_decision_refuses_an_arm_without_five_seeds(...): ...
def test_the_reference_stands_when_none_beats_it(...):
    assert record.forward is None
def test_the_highest_beating_candidate_goes_forward(...):
    assert record.forward == 'gte-small-d16'
def test_a_tie_goes_to_the_earlier_arm(...):
    assert record.forward == 'euclidean-d16-ita'
def test_the_control_never_goes_forward(...):
    # the control scores far above everyone: forward is still a candidate or None, and no verdict
    # names the control
def test_the_decision_refuses_without_a_margin_first(...): ...
```

- [ ] **Step 2: Run them to verify they fail**

  Run: `uv run pytest tests/unit/test_ita_decision.py -q`
  Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the protocol, the module and the command.**

- [ ] **Step 4: Run them to verify they pass**

  Run: `uv run pytest tests/unit/test_ita_decision.py tests/unit/test_decision_text_to_code.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/naics_embedder tests
git commit -m 'feat(decision): decide text-to-code from the margin and the chosen epochs'
```

---

### Task 10: Documents and the whole-suite gate

**Files:**
- Modify: `CLAUDE.md` (Architecture 1 and the run-settings paragraph: 21 keys plus up to two
  when set; Checkpoint Contract; the hook list; file and test counts; Directory Structure for
  `ita_selection.py`, `ita_decision.py` and `conf/data/text_to_code_arms.yaml`)
- Modify: `.claude/rules/text-to-code.md` (Reads and Tools: the three commands; Traps: drop the
  two fixed `ita-read` traps and the `check_ita_margin` trap; Tests: the two new files)
- Modify: `docs/api/decision.md` (sections for `ita_selection` and `ita_decision`, after
  `ita_gap`), `docs/usage.md` (the three commands, `data summaries --window`), `docs/text_training.md`
  (retention and the frozen switch), `README.md` (test count)

- [ ] **Step 1: Update the documents.** Every changed docstring that mkdocstrings renders
  documents its arguments.

- [ ] **Step 2: Run the gates**

```bash
uv run pytest -n auto -q
uv run ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run mkdocs build --strict
uv run pytest --collect-only -q -q | tail -1
```

  Expected: the suite passes except the § 19 failure; Ruff, the style gate and the strict build
  pass. Put the collected count into `CLAUDE.md` and `README.md`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md .claude/rules/text-to-code.md docs
git commit -m 'docs: describe retention, the frozen control and the Stage 5 ITA commands'
```

---

### Task 11: The real backbones' summaries and the bundle check (gated)

**This task waits for your human partner's approval, given at the time, of two downloads:**
`thenlper/gte-small` at `17e1f347d17fe144873b1201da91788898c639cd` (about 67 MB of
`model.safetensors`) and `BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
(about 133 MB), into `~/.cache/huggingface/hub`. Without the approval, skip the task, mark it
`> Skipped: not approved → deferred`, and Plan B starts with it.

**Files:**
- Create: `conf/data/window_summaries_thenlper__gte-small.csv` and its provenance JSON
- Create: `conf/data/window_summaries_BAAI__bge-small-en-v1.5.csv` and its provenance JSON
- Modify: `src/naics_embedder/panels/window_summaries.py:67-73` (two pins)
- Modify: `tests/unit/test_committed_window_summaries.py` (the two pins' sha256 and window)
- Modify: this plan (the recorded local check)

- [ ] **Step 1: Download each model's `main`, then check it is the pin.** Use
  `huggingface_hub.snapshot_download(repo_id)` with no `revision`. A commit-hash `revision` writes
  no `refs/main`, and `load_backbone` and `stage_model_cache` read `refs/main` offline. Then check
  that `refs/main` and the snapshot directory name equal the pinned revision. If `main` has moved
  since 2026-11-22, stop and ask your human partner before you change a pin.

- [ ] **Step 2: Check the shared vocabulary.** Load the three tokenizers. Record whether their
  vocabularies are equal and whether each counts the bundle's marked channel texts the same.

- [ ] **Step 3: Write each artifact.** Run the worktree's code, read the main checkout's
  descriptions by absolute path, and write into the worktree by absolute `--output`, so nothing
  lands in the main checkout's `conf/`. Take the descriptions path from bundle 301cce28's
  `generation_parameters['descriptions_parquet']`. Repeat for `BAAI/bge-small-en-v1.5`.

```bash
cd <worktree> && HF_HUB_OFFLINE=1 uv run naics-embedder data summaries --backbone thenlper/gte-small --window 128 --descriptions /Users/lowell/Projects/naics-embedder/<descriptions_parquet> --output <worktree>/conf/data/window_summaries_thenlper__gte-small.csv
```

  Commit the pins, and record each over-window count and whether each artifact's rows equal
  MiniLM's.

- [ ] **Step 4: The bundle-301cce28 check.** Run the worktree's code and pins (`cd <worktree> &&
  HF_HUB_OFFLINE=1 uv run python <script>`, with the script kept outside the repository), never
  the main checkout's package, which lacks Task 4's windows and this task's pins. Read the main
  checkout's inputs by absolute path: the manifest that its gitignored `conf/local.yaml` names,
  joined to `/Users/lowell/Projects/naics-embedder/`, loaded with `load_validated_bundle`, and
  the descriptions that the bundle's `generation_parameters` name. For each of the three
  backbones, build the tied config at length 128 and resolve the descriptions' channel texts
  through `resolve_channel_texts` with that backbone's tokenizer (the token cache's path), and
  run `stage_model_cache` into a temporary root. Record the backbone, the over-window count,
  that resolution succeeded and that the cache staged. Write the record under this step as a
  `> Check:` line.

- [ ] **Step 5: Run the committed-artifact tests and commit**

  Run: `uv run pytest tests/unit/test_committed_window_summaries.py -q`
  Expected: PASS.

```bash
git add conf/data src/naics_embedder/panels/window_summaries.py tests specs/plans/22-text-to-code-candidates.md
git commit -m 'feat(data): pin 128-token window summaries for gte-small and bge-small'
```

---

## Deferred entries this plan settles (spec §4.9)

| Entry | Settled by |
|---|---|
| § 7 backbone tie | Task 4 (the tie and the bundle-not-binding decision), Task 11 (bundle 301cce28) |
| § 9 summaries key | Task 4: done by the tie |
| § 17 margin δ against the build | Task 9, step 3 |
| § 17 margin-first across logs | Task 8, `refuse_ita_monitor_reads`, with the written why |
| § 17 a candidate read blocks later margins | Stays open as a design item. Written why: Plan B fixes the margin first, and `ita-select` refuses to read before it |
| § 17 margin reads name no checkpoint | Task 7 |
| § 17 margin self-check gaps | Task 7 |
| § 20 `--outcome` | Task 6 |
| § 20 d16 identity | Task 6 (`ita-read`), Task 7 (`ita-margin`) |
| § 10 native qualification | Plan B (the first campaign run) |

The completion ticks these with `→ done` or the written why. It leaves the five held marks in
`specs/deferred_items.md` (`:12`, `:132`, `:335`, `:363`, `:455`) unchanged.

## Handoff to Plan B

Written at planning time from the planned interfaces. Execution updates it where a name changes.

- **Commands, in order.** `tools ita-margin` (the margin record), the campaign through the
  `remote` adapters with each arm's overrides from `conf/data/text_to_code_arms.yaml` and an
  `experiment_name` per arm and seed, `tools ita-select` per arm (one batch per arm), then
  `tools ita-decide`. All three ask for approval through the evidence guard, and their dry runs
  pass.
- **The margin's inputs.** The stub's descriptions, d16 seeds 1–5 from
  `checkpoints/stage8-euclidean-d16-sN/` (`epoch=039`, `038` for seed 4) with their exported
  tables, `~/naics-artifacts/records/stage8/euclidean-d16.json`, DATA_FP, and the main checkout's
  `logs/selection_log.jsonl` by absolute path.
- **Left to Plan B by the spec (§8).** The qualification run's tolerance, stated before the run;
  how the reads are batched and the drift check before each (`ita-select` already runs C1 and the
  margin-first check before its first read); whether `remote finish` pulls one run at a time.
- **Token cache.** It is rebuilt on the instance from its identity, which names the tokenizer, so
  arms that share `data_loader.tokenization.output_path` rebuild it at each backbone change.
  Plan B may give each backbone its own path to skip the rebuilds.
- **The staged model cache.** The Mac's `.remote/model-cache` from Stage 8 holds MiniLM only,
  so the first `remote up` after this plan refuses it (Task 4). Plan B removes it once, with the
  owner's approval, before that `up`. Every instance starts fresh.
- **Pins.** If Task 11 was skipped, Plan B starts with it: the summaries pins and the
  bundle-301cce28 check come before any gte or bge run.
- **Deferred.** Plan B files the checkpoint-cleanup entry of spec §4.4 and carries the
  `Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 5` line in its header.
