# NAICS Embedder Test Suite

## Current Status

The suite contains **100 unit** test files and **2 integration** files, with **3,099 collected
nodes**. Actual skip counts depend on local data and hardware capabilities, including MPS.
Collection counts describe the suite inventory, not measured coverage percentages.

Tests use immutable fixture bundles and tiny backbones. Training/panel outputs belong under
`tmp_path`; verification does not read real sealed splits, train a real NAICS model or run a
real selection campaign. Hugging Face fixture dependencies are resolved locally.

## Test Structure

```text
tests/
├── unit/                         # 100 unit test files
├── integration/
│   ├── test_reference_training.py # tiny-backbone Trainer workflows
│   └── test_remote_workflow.py    # GNU-rsync workflow qualification
├── fixtures/                     # shared encoder, supervision, panels and run records
└── conftest.py                   # common fixtures
```

## Current Training Coverage

| Interface | Tests and behavior |
|-----------|--------------------|
| Query facts and dense targets | `test_supervision_queries.py`, `test_supervision_code_targets.py`: eligible role/phrase queries, target and referring identities, levels, lineal/unary masks |
| Shared encoder and fusion | `test_encoder.py`, `test_fusion.py`, `test_moe.py`: shared adapters, markers, absent-channel masking and fusion |
| Objective and radius | `test_loss.py`, `test_hyperbolic.py`, `test_naics_model.py`: task/code-code/radial terms, scales, bounded live radius and polar distance |
| Geometry arms | `test_heads.py`: the Euclidean, spherical and hyperbolic heads, their flat or polar training distances, read maps and the one list of geometries (Req 12) |
| Epoch loader | `test_datamodule.py`, `test_tokenization_cache.py`: seed/epoch permutations, exact coverage, code chunks, query tokens and cache identities |
| Cache and outcome monitor | `test_monitor.py`: eval/no-grad refresh, detached candidates, live-anchor replacement, validation reads and durable records |
| Selection-log guard | `test_selection_log_guard.py`: fail-closed refusals before appending reads or openings |
| Training and contracts | `test_cli_training.py`, `test_utils_training.py`, `test_checkpoint_contract.py`, `test_export.py`, `test_arm_encoder.py`: objective refusal, settings/directory/seed guards, saved constructor controls, CPU/MPS float64 callback transport and export/read provenance |
| Campaign runner | `test_checkpoint_runner.py`: all-seed preflight, earliest best, epoch coverage, saved-score/version-sibling refusals and monitor-record pass-through |
| Verification and summary | `test_radius_report.py`, `test_epoch_summary.py`, `test_visualize_metrics.py`: live gradients, radius/distance checks, durable finite health, resume and SD bands |
| Trainer integration | `integration/test_reference_training.py`: cache/monitor order, earliest tied best, warmup/plateau, early stopping, exact resume and health |

Other unit files cover graph refinement, bundle construction, data preprocessing, panel
scoring, paired resampling, decision records, configuration and geometry/console utilities.
The text model has no streaming-negative dataset, curriculum/miner, DCL or structural-preference
loss; their obsolete tests were removed with those modules. HGCN retains its separate sampler
and graph curriculum tests.

## Running Tests

```bash
uv run pytest -n auto -q
uv run pytest tests/unit/test_monitor.py -q
uv run pytest tests/integration/test_reference_training.py -q
uv run pytest tests/ --cov=naics_embedder --cov-report=html
uv run pytest --collect-only -q
```

`.python-version` selects Python 3.12. CI tests both 3.10 and 3.12; use a separate environment
when checking 3.10 locally:

```bash
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310 uv run pytest -n auto -q
```

## Coverage and Limits

Measure coverage with pytest-cov rather than inferring it from inventory or pass counts.
`integration/test_reference_training.py` trains the reference fixture through the real Trainer,
but no test runs the whole production pipeline from raw-data generation through text, HGCN and
a real decision. HGCN's one-batch fit tests its structural logging; it does not establish a
complete graph training campaign. Historical analysis and model runs are separate evidence.

When a test asserts a refusal, ensure its fixture violates only the intended condition and
assert that no export/read occurred first. A broad `raises(ValueError)` can pass after the
wrong branch or after a side effect. Exercise later seeds as well as the first in all-seed
preflight tests. Test both missing and invalid durable records and radius mean/SD bands.

## Known Issues

Python 3.14 is outside the local pin: the locked torch 2.9.x can fail while importing
`torch._inductor`. See [initial setup](../CLAUDE.md#development-setup). MPS-dependent tests skip
when MPS is unavailable. Do not add broad warning suppression to hide contract or numerical
failures.

## Test Markers and CI

The project registers unit, integration, slow and gpu markers. Check
`pyproject.toml` for their descriptions. `.github/workflows/tests.yml` runs pytest/coverage on
Python 3.10 and 3.12 and a separate Ruff/YAPF lint job. PR CI also builds MkDocs with strict
checks and locked dependencies; check rendered docs locally:

```bash
uv run ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run mkdocs build --strict
```

## Adding and Reviewing Tests

Write a failing behavior test before library/pipeline implementation. Use small deterministic
fixtures, exact identities and boundary cases. Preserve unrelated edits when working in
parallel. For model/data work, validate the scientific property: gradients through the intended
term, hold-out role, paired units, exact once-per-epoch coverage or fail-closed reads.
Do not merely restate an implementation formula or count a mocked call.

For geometry, check symmetry, coincidence, manifold constraints and precision at both small
and large radii. Curvature-parameter tests belong to Lorentz/HGCN utility callers; the text
head's curvature is fixed. Build valid Lorentz points through an exponential map instead of
feeding random unconstrained coordinates to a distance test.

For files and caches, use `tmp_path`, assert hashes/contracts and verify malformed inputs fail
before writes. For resume, compare continued records and training state with an uninterrupted
fixture run. For plotting, inspect the plotted values and SD bands as well as the output path.

## Debugging Failed Tests

```bash
uv run pytest tests/unit/test_loss.py -v --tb=long
uv run pytest tests/unit/test_loss.py --pdb
uv run pytest --lf
uv run pytest -s
```

Reproduce and isolate the cause before changing code. A failed guard may signal a broken
fixture contract rather than a reason to relax production validation. Target tests and the
full suite must pass before a completion claim; report actual skips and warnings.

## Resources

- [Project conventions](../CLAUDE.md)
- [Text objective and operator workflow](../docs/text_training.md)
- [System overview](../docs/overview.md)
- [CLI usage](../docs/usage.md)
- CI configuration: `.github/workflows/tests.yml`
