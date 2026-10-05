# Approved whole-branch review fixes I1/I2

Base: `1faca08078bcb399a2b9f36035bd210a136552d7`. The controller relayed the user's explicit
approval of both plan conflicts before implementation. `final-review.md` was read first.

## Changes and approved interpretation

- I1: the retained HGCN feeder deserializes the complete checkpoint on CPU, then moves only the
  constructed model to the selected device and switches to eval mode. The existing contract
  refusal remains before model loading. The real fixture feeder runs on both CPU and actual
  MPS with a CPU float64 callback score; it still writes valid d+1 Lorentz columns.
- I2: `constructor_settings` maps current config values to the already saved constructor
  hyperparameter names. `refuse_other_constructor_settings` refuses missing or differing
  LoRA rank/alpha/dropout, plus expert count/top-k/hidden dimension/load-balancing coefficient
  under active MoE fusion. Messages name each differing saved/current value. CLI exact resume
  checks these before data/monitor/model construction. The runner applies the same check to
  both last and selected checkpoints during every seed's preflight, before exports or decision
  panel reads. `tools sweep` already preflights all seeds before regressor panel construction;
  its production ordering required no change.
- MoE controls are deliberately inactive under masked mean and attention. `build_fusion`
  constructs experts only for MoE; the model applies load balancing only for MoE (R11). Thus
  inactive expert settings neither need comparison nor need to be saved. Tests cover both
  inactive fusions with changed config values and absent expert hyperparameters.
- The literal 21-key `run_settings`/`ArmSpec.settings` dictionary is unchanged. Current
  checkpoints already save these constructor arguments through `save_hyperparameters`.
  Nothing migrates weights or rewrites checkpoints or finished artifacts. The original spec
  and approved plan were not edited. This supplementary guard and CPU-first feeder boundary
  are the two expressly approved departures from the incomplete literal plan instructions.
- Operator guides and the future Lambda resume pre-check describe the supplementary guard.
  Existing last-only, both-JSONL, absolute-path, finished-run and later-Lambda rules are kept.
  README/CLAUDE/tests README now report collection inventory instead of static pass/skip
  counts; skips depend on local data/hardware. Inventory is 2,315 nodes in the unchanged 83
  unit files and one integration file.

## Regression evidence

Meaningful regressions were written before production changes. Initial RED reproduced the
CLI's nine independent changed/missing-constructor cases, both-checkpoint alpha/dropout cases,
all-seed rejection gaps, and both feeder device boundaries. One new all-seed case initially
assumed a fixture selected epoch 1. The covering GREEN exposed a missing fixture file instead
of testing that case. It was corrected to derive the earliest best epoch from the fixture's
monitor records. All 17 corrected regressions were then rerun against the reviewed baseline
and failed for the intended missing behavior; owned production edits were restored afterward.
No speculative production patch was needed for the fixture correction.

RED command (baseline production):

```bash
uv run pytest tests/unit/test_cli_training.py tests/unit/test_checkpoint_runner.py \
  tests/unit/test_export.py \
  -k 'constructor_changes or other_constructor or hgcn_feeder_writes' -q
```

Result: **17 failed, 108 deselected, 2 warnings** in 3.55 seconds, pytest exit 1.
CLI tests did not raise; runner cases reached blocked export/panel seams; feeder cases
passed device rather than CPU to the real loader. Both CPU and actual MPS cases ran.
Corrected output: `/tmp/plan10_finalfix_red_verified.out`.

Additional boundary coverage tests all seven required active controls missing individually,
unchanged settings for all three fusions, and unchanged 21-key identity.

Final covering GREEN command:

```bash
uv run pytest tests/unit/test_cli_training.py tests/unit/test_checkpoint_runner.py \
  tests/unit/test_export.py tests/unit/test_utils_training.py \
  tests/integration/test_reference_training.py -q
```

Result: **205 passed, 52 warnings** in 24.15 seconds, exit 0. Includes actual CPU/MPS feeder,
unchanged CLI resume, both saved-checkpoint identities, all-seed no-early-export/panel refusal,
and real Trainer fixture exact-resume/monitor/cache integration.
Output: `/tmp/plan10_finalfix_green.out`.

Warnings are existing fixture/framework warnings: intentional CPU Trainers on a GPU host,
small main-process loaders, bf16 model-summary estimation, expected nonempty resumed fixture
directories, and the deliberately refused mid-epoch continuation. No warnings were suppressed.

## Formatting and documentation gates

- `./scripts/format_code.sh` with the seven owned Python paths: passed Ruff fix, YAPF and Ruff.
- `uv run ruff check src/ tests/`: **All checks passed**, exit 0.
- `./scripts/format_code.sh --check --all`: **Clean: no lint issues and no formatting changes**,
  exit 0. No mass formatting was performed.
- `uv run pytest --collect-only -q`: **2315 tests collected** in 0.47 seconds, exit 0.
- `uv run mkdocs build --strict`: completed successfully in 1.55 seconds, exit 0 before the
  final doc edits; final rerun below confirms them.

## Scope, skill audit and limits

Used clean-code, clean-coder, test-driven-development and verification-before-completion.
No adjacent refactoring was performed. Applied in-scope test standards T1/T6 to the uncovered
readers/identity controls, and N1/G2 to the named constructor-setting mapping and typed helper
boundary. No unrelated tidy edits share the behavior commit.

All writes, uv and Git mutations used native-host escalation for this managed worktree outside
writable roots. No approval rejection occurred. Existing untracked `outputs/plan10_smoke`
remains untouched. No real training, panel read, embedding export, checkpoint mutation,
download, Lambda operation or push was performed. All executed training/exports/panels are
existing test fixtures in pytest temporary roots, with tiny backbones and fixture bundles.

The completed smoke was never relaunched or resumed. Its evidence and append-only history are
preserved. The controller will run both full Python-version suites and final independent review;
this report claims only the freshly executed covering gates. CUDA/Lambda and Linux/MKL remain
later verification boundaries. No known production blocker remains in the approved scope.

Final strict docs rerun after usage guide and line wraps: successful in 1.37 seconds, exit 0.
