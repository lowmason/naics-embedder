# CLAUDE.md - AI Assistant Guide for NAICS Embedder

## Project Overview

NAICS Embedder learns a shared hyperbolic space for classification codes and activity queries.
One LoRA-adapted text backbone, masked fusion and one projection feed a live radius/direction
head. The text objective jointly learns task retrieval, code geometry and radial structure.
An optional HGCN stage refines the parent–child graph with its own objective and curriculum.

The project uses Python 3.10+, uv, PyTorch/Lightning, Transformers/PEFT, Polars/PyArrow, Pydantic,
Typer/Rich, pytest, Ruff/YAPF and MkDocs. `.python-version` pins the local locked environment to
Python 3.12. The source tree has **103 Python files**; tests have **83 unit** files and one
integration file. File counts exclude generated and ignored artifacts.

## Architecture Summary

1. **Shared text encoding** (`text_model/fields.py`, `shared_encoder.py`): one LoRA backbone reads
   marked title, description, examples, exclusions and queries. Blank code fields never enter it.
2. **Fusion and projection** (`text_model/fusion.py`): masked mean by default, attention or MoE
   as options, then one `Linear(384, d)` for d in {8, 16, 32}, default 16. MoE is an ablation.
3. **Live radius and objective** (`text_model/hyperbolic.py`, `loss.py`, `naics_model.py`):
   `r = R * tanh(norm(v) / R)` with default R = 8, and the projection's direction, form the
   unit-curvature Lorentz point. Task, code-code and radial terms all retain radius gradients.
4. **Graph refinement** (`graph_model/hgcn.py`): HGCN uses the graph, triplet/radial objectives
   and its four-phase graph curriculum. Its curvature utilities remain separate from text.

The text model has no curvature parameter, DCL term, false-negative clustering, negative miner,
router-guided sampler, text phase curriculum, distributed cache or in-sample validation loader.
The HGCN stage retains its graph-specific samplers and curriculum.

`tools export-table` writes each code's bounded tangent coordinates at the origin, `e0` through
`e{d-1}`, in Req 2's form. Panel reads reconstruct Lorentz points on the CPU in float64.
The selected text checkpoint is the earliest epoch with the highest outcome validation MRR.

## Directory Structure

Paths below are relative to the repository; the tree shows the principal modules.

```text
src/naics_embedder/
├── cli/commands/              # data, tools and training CLI commands
├── data/                      # preprocessing, redirections, bundle generation
├── supervision/
│   ├── activity.py            # canonical activity-phrase parser
│   ├── queries.py             # eligible task queries and target/referring codes
│   ├── code_targets.py        # dense tree targets and anchor/unary keep mask
│   ├── artifacts.py           # immutable bundle loading and validation
│   ├── checkpoints.py         # objective/encoder/preprocessing checkpoint contracts
│   └── schema.py              # bundle schemas and roles
├── text_model/
│   ├── fields.py              # field markers
│   ├── shared_encoder.py      # one backbone, fusion and projection
│   ├── fusion.py              # masked mean, attention, MoE
│   ├── hyperbolic.py           # live-radius head, stable polar distance, Lorentz ops
│   ├── loss.py                # task_loss, code_code_loss, radial_loss, LogitScale
│   ├── naics_model.py         # two-stream training, cache and monitor orchestration
│   ├── monitor.py             # CodeCache, LiveEncoder, OutcomeMonitor
│   ├── epoch_summary.py       # durable epoch MRR and health values
│   ├── checkpoint_runner.py   # all-seed preflight and selected-checkpoint export
│   ├── radius_report.py       # radius and no-inert-terms verification
│   ├── export.py              # tangent code-table export and provenance
│   ├── arm_encoder.py         # checkpoint queries and exported code coordinates
│   ├── moe.py                 # optional experts and load balancing
│   ├── dataloader/            # two-stream steps, tokenization cache
│   └── mixins/                # loss, logging and optimizer responsibilities
├── panels/                    # outcome/regressor panels and selection-log guards
├── decision/                  # paired resampling, margins, arm records and decisions
├── graph_model/               # HGCN, graph data loading and four-phase curriculum
├── metrics/                   # structural and retrieval diagnostics
├── tools/                     # config display, epoch-summary plotting and metrics API
└── utils/                     # config, training, input-window and geometry utilities

tests/
├── unit/                      # 83 unit test files
├── integration/               # test_reference_training.py
├── fixtures/                  # tiny models, bundles, panels and runs
└── conftest.py

conf/config.yaml               # reference text configuration
conf/graph.yaml                # separate HGCN configuration
conf/data/                     # frozen panel pins, roles, summaries and held-out groups
docs/                          # guides and MkDocs API pages
scripts/format_code.sh         # Ruff fixes/imports, then YAPF layout
```

Generated data, checkpoints, logs and reports are separate artifacts. Do not confuse historical
training logs or a real panel campaign with fixture verification.

## Key Concepts and Patterns

### Shared Fields, Window and Fusion

The fields are `title`, `description`, `examples`, `excluded` and `query`. Mark each present
text before tokenization; every field shares the LoRA adapters. Resolve over-window channel
text through committed window-fitting summaries; never silently truncate it. Token caches and
checkpoint contracts record the summary hash. Descriptions and tokenizer/window identities
belong to the token cache.

Masked mean has no parameters. Attention starts as masked mean. Under `fusion=moe`, only the
experts' load-balancing term is added; other fusions do not call it. Exactly one affine map
produces the d-dimensional projection.

### Three-Term Objective

`task_loss` sums softmax probability over each query's named targets. Candidates are every code
at that query's level plus its referencing codes. Phrase targets omit named codes lineal to the
referring code; a target/referring-code overlap is refused. Training-role index entries and
non-withheld phrases supply queries; held-out query text does not enter training.

`code_code_loss` matches the softmax of `-D* / target_temperature` over all codebook codes except
the anchor and its unary partner. It never reads exclusion data. `radial_loss` is mean squared
error from live radius to `radial_step * (level - 1)`. Both term weights and radial step default
to 1. Task and code-code logits use independent learned positive scales, clamped to [0.01, 100]
without weight decay. The total adds MoE balancing only under that fusion.

### Radius and Precision

The head computes `r = R * tanh(a / R)`, with a = norm(v), then maps the bounded tangent r u to
`(cosh(r), sinh(r) u)`. Its origin guard handles zero v. Text curvature is fixed at 1 with no
setting or learned parameter. Radius r is the radial quantity in text losses, health and reports.

Float32 stable polar distances avoid cancellation when two large-radius points are nearby.
CUDA uses `bf16-mixed` only in the backbone. Fusion, projection, head, distances and losses
remain float32. CPU and MPS use `32-true`. The text Trainer has one device and refuses more.
Lorentz operations with curvature arguments and the higher-level manifold utilities remain for
HGCN/general callers; do not infer a text curvature option from them.

### Two-Stream Epoch and Code Cache

An epoch visits every code anchor and eligible task query once, in seed/epoch permutations.
The query count and `data_loader.queries_per_step` determine steps; the reference's 11,039
queries at 128 give 87 steps, over which the 2,125 code anchors are distributed evenly.
The training-pairs member remains validated in the bundle but is never read by text training.
HGCN retains its separate streaming/sampling path.

`refresh_code_cache` runs in eval/no-grad mode at fit start and each epoch end. Its detached
candidate rows are replaced by the current step's live anchors. The embedding cache is rebuilt
on resume; the token cache is a separate preprocessing artifact. No text validation loader is
built.

### Model Mixins

| Mixin | Responsibility |
|-------|----------------|
| `LossMixin` | MoE load balancing under `fusion=moe` |
| `LoggingMixin` | Epoch loss means, scales and per-level radius mean/SD |
| `OptimizerMixin` | AdamW, warmup/plateau and post-step scale clamping |

The model itself coordinates losses, cache refresh, monitor reads and checkpoint metadata.

### Monitor, Epoch Summary and Selection

After cache refresh, `OutcomeMonitor` reads outcome validation through `OutcomePanel.score_logged`.
Its float64 MRR, `val/outcome_mrr`, selects the earliest highest-MRR epoch, controls the plateau
scheduler and drives early stopping. It records training-run id, seed, epoch and matrix
fingerprint in the selection log and durable `monitor_reads.jsonl`. Test splits stay sealed.

`epoch_summary.jsonl` records the same MRR plus loss/task, code_code, radial and total, both
logit scales, and radius mean/SD at levels 2–6. MoE includes load-balancing loss. Health values
come from `_log_health()`'s Python floats, not rounded Lightning callback metrics.
`tools visualize --summary PATH` writes one `epoch_metrics.png` figure with four panels.

Structural metrics are no longer text validation or scheduler controls. `tools diagnostics`
reports Req 6 without thresholds or adoption. HGCN keeps structural logging and its graph
curriculum. The outcome and two regressor panels decide adoption under Req 5.

### Checkpoint Contract and Exact Resume

The contract identifies objective `req11-v1`, bundle, encoder and preprocessing. Old objective
checkpoints are refused before model loading in training, export, outcome reads and the HGCN
feeder. Nothing migrates weights into this objective.

Exact resume requires the same experiment directory, seed and all 21 `run_settings`, including
budget, precision, clipping and accumulation, as well as the contract. Fresh starts refuse
used checkpoint directories. Resume only `--ckpt-path last`; an older kept checkpoint can rewind
records and is not the operator continuation path.

The cache is rebuilt. Monitor and summary files retain their original lines through the
restored epoch, prune later interrupted lines and continue. An early-stopped run exits 1 with
`early stopping ended the run at epoch k`; a launcher must regard it as finished. A spent epoch
budget is a harmless no-op. Changing max_epochs is a settings mismatch, not an extension.
`--checkpoint-load-mode` accepts only `exact`.

`training.trainer.val_check_interval` remains a parsed field but is unread by the text Trainer.
The monitor runs once at epoch end, without in-sample validation.

### Reference Campaign and Verification

Train seeds 1–10 with the same settings on Lambda CUDA, then perform decision reads on the Mac.
`tools sweep --runs '...{seed}' --seed ...` checks every seed before any export or decision read.
It requires complete monitor epochs, consistent panel/run/seed/settings identities, an
unambiguous earliest best checkpoint and exact saved best score. A selected epoch with a
versioned sibling is refused. Each arm record carries all monitor reads and three panel reads
per seed: outcome, regressor seen and regressor held-out at level 6.

Fix the reference margins with `tools margins --multiple 3` before candidate monitor or decision
reads, then use `tools decide`. Use the cached arm backbone revision for comparison with the
frozen text-only table; the comparator cannot supply the arm's revision.

`tools radius-report` reads the saved seed's first batch and exported table on the CPU without
reading a panel. It verifies anchor-radius gradients, per-level SD > 1e-3, positive distinct
sector radii, manifold error and chunked distance precision, plus gradients for the three terms
and both scales. A failed criterion writes the report and exits 1.

## Development Setup

```bash
git clone https://github.com/lowmason/naics-embedder.git
cd naics-embedder
uv sync
uv run naics-embedder --help
```

Use `.python-version`'s Python 3.12 with the locked dependencies. CI also tests Python 3.10;
use a separate environment to check that version rather than replacing `.venv`.

### Running Commands

```bash
uv run naics-embedder data all
uv run naics-embedder train   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
uv run naics-embedder train --ckpt-path last   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
uv run naics-embedder tools config
uv run naics-embedder tools visualize --summary checkpoints/reference/epoch_summary.jsonl
uv run naics-embedder tools sweep --help
uv run naics-embedder tools radius-report --help
uv run naics-embedder tools diagnostics --help
```

`data supervision` prints the exact manifest path. Retired relations/distances/triplets commands
build nothing and exit 1. The manifest gate cannot be skipped. HGCN reads `conf/graph.yaml` and
runs with `uv run python -m naics_embedder.graph_model.hgcn`; set its manifest separately.

### Running Tests

```bash
uv run pytest -n auto -q
uv run pytest tests/unit/test_loss.py
uv run pytest --cov=naics_embedder
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310 uv run pytest -n auto
```

The current suite collects 2,282 tests: **2280 passed, 2 skipped** on a host with MPS;
**2273 passed, 9 skipped** without MPS. Tests use fixture data and tiny models. These counts
are not coverage percentages. Do not read real sealed splits or run a real campaign to verify
an ordinary code/doc change.

## Code Style and Conventions

### Python Formatting

**Tools** (both configured in `pyproject.toml`, both run by `scripts/format_code.sh`):

- **YAPF:** The formatter. It owns layout: line wrapping, blank lines, bracket placement.
- **Ruff:** The linter (E, F, I, Q rules), plus import sorting via `ruff check --fix`.
  **Never run `ruff format`**: it forces double quotes (which the Q rules reject) and 2 blank
  lines between top-level definitions, which would rewrite nearly every file.

**Key Rules (from `pyproject.toml`):**

- **Line length:** YAPF wraps at 100 characters; ruff's E501 allows up to 105 for lines YAPF
  cannot split
- **Quotes:** Single quotes (`'` not `"`), including `'''` docstrings
- **Blank lines:** 1 between top-level definitions, and 1 after the import block
- **Indentation:** 4 spaces; closing brackets dedented onto their own line
- **Imports:** Sorted by ruff's isort rules (I)
- **Method chaining:** YAPF packs a chain onto as few lines as fit and splits before a `.` only
  when a line overflows. To keep a vertical Polars chain, fence it, as `data/download_data.py`
  does:

```python
# yapf: disable
result = (
    df
    .filter(pl.col('code').is_not_null())
    .select(['code', 'title', 'description'])
    .collect()
)
# yapf: enable
```

**Format Code:**

```bash
# Files changed on your branch vs origin/main, including uncommitted and untracked ones
./scripts/format_code.sh

# Specific files or directories
./scripts/format_code.sh src/naics_embedder/text_model/loss.py

# Check only: exits non-zero on lint issues or files YAPF would reformat
./scripts/format_code.sh --check src/naics_embedder/text_model/loss.py
```

**Keep the tree clean:** `ruff check src/ tests/` and `./scripts/format_code.sh --check --all` both
pass, and CI's `lint` job fails if either stops passing. Format the files your change touches,
don't mass-fix unrelated code, and keep `./scripts/format_code.sh --all` out of feature PRs.

### Markdown Formatting

**Rules (from `.markdownlint.jsonc`, which is gitignored, so fresh clones don't have it):**

- **Headings:** 1 blank line above and below
- **Lists:** Indent by 2 spaces
- **Max consecutive blank lines:** 1

**Convention (not linted, since that file disables the MD013 line-length rule):**

- **Line length:** 100 characters (code blocks and tables exempt)

### Section Dividers

Python files use **semantic section dividers** with consistent formatting:

```python
# -------------------------------------------------------------------------------------------------
# Section Title
# -------------------------------------------------------------------------------------------------

# Code here...
```

**Common sections:**

- Imports and settings
- Config / Constants
- Utilities
- Main logic
- Entry point

### Logging

Use Python's `logging` module (not `print`):

```python
import logging

logger = logging.getLogger(__name__)

logger.info('Starting training...')
logger.warning('Curvature out of bounds, clamping to safe range')
logger.error('Failed to load checkpoint')
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Optional, Dict, List, Tuple

def compute_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float = 1.0
) -> torch.Tensor:
    ...
```

### Docstrings

Use **single-quote triple-quoted docstrings**:

```python
def exp_map_zero(x_tan: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    '''
    Exponential map from tangent space at origin to Lorentz hyperboloid.

    Args:
        x_tan: Tangent vector (N, D)
        c: Curvature (positive scalar)

    Returns:
        Hyperbolic point on hyperboloid (N, D+1)
    '''
    ...
```

## Common Development Tasks

### Change the Text Objective or Loader

Edit `text_model/loss.py`, `naics_model.py`, `supervision/queries.py`, `code_targets.py`, or
`text_model/dataloader/datamodule.py` as appropriate. Write a failing behavior test first. Check
query candidate/target identities, unary masks, live-anchor replacement and radius/scale
gradients, rather than merely mirroring the formula. Update the config contract deliberately;
settings are compared on exact resume and campaign preflight.

### Change Configuration

Use Pydantic models in `utils/config.py` and keys that the current code reads. For a fresh run:

```bash
uv run naics-embedder train experiment_name=reference-small   data_loader.queries_per_step=64 training.learning_rate=1e-5   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

Do not reintroduce removed objective/mining/curvature settings or bypass bundle validation.
`tools config` validates the YAML over defaults and displays the run inputs and settings.

### Debug Training or Add Metrics

Reproduce a failure with a fixture first. Use the epoch summary for term means, scales and
radius spread, and the radius report for live gradients and precision. MRR is the text control
score. Keep structural diagnostics separate from selection; adding a statistic must not create
a second checkpoint selector. HGCN metrics belong in its graph/metrics modules.

### Work on HGCN

Keep the graph curriculum and sampler changes in `graph_model/`; the text model has no phase
curriculum. Validate graph bundle paths with `GraphConfig`. The graph stage retains triplet loss,
per-level radial regularization, curvature utilities and structural statistics. Its full
evaluation currently uses curvature 1; a non-unit-curvature correction is a separate concern.

## Testing and Validation

There are 83 unit files and one integration file. Important current seams include:

- `test_supervision_queries.py` and `test_supervision_code_targets.py`: query/target identities
  and unary masks. `test_loss.py` and `test_hyperbolic.py`: three terms, live-radius head and
  stable polar distance.
- `test_datamodule.py`, `test_monitor.py`, `test_selection_log_guard.py`: two-stream epochs,
  candidate cache, monitor reads and fail-closed selection-log guards.
- `test_cli_training.py`, `test_utils_training.py`, `test_checkpoint_contract.py`:
  contracts, exact resume, stopped runs and settings guards.
- `test_checkpoint_runner.py`, `test_radius_report.py`, `test_epoch_summary.py`,
  `test_visualize_metrics.py`: campaign preflight, radius checks and durable health artifacts.
- `integration/test_reference_training.py`: tiny-backbone Trainer runs, earliest-best selection,
  schedule/early stopping, health records and exact resume.
- Graph, panel, resampling and decision tests cover their separate interfaces.

Keep generated fixture outputs under `tmp_path`. Use reference fixture bundles instead of real
data. Run meaningful target tests and the full suite before claiming completion. Measure coverage
with pytest-cov when needed; test counts alone do not establish it.

## CI/CD and Documentation

`.github/workflows/tests.yml` runs Ruff and YAPF in a Python-3.12 lint job, plus pytest/coverage
on Python 3.10 and 3.12 with locked dependencies. The docs workflow deploys on main/master;
PR CI does not build documentation, so run strict MkDocs locally for rendered documentation.

```bash
uv run ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run mkdocs build --strict
```

Format only touched Python files with `scripts/format_code.sh`; the full `--check --all` gate is
read-only. Never use `ruff format`. New API pages use mkdocstrings and must appear in
`docs/.nav.yml`. Re-point links when removing a heading, and check rendered anchors as well as
the strict build. Public rendered docstrings must document their arguments accurately.

## Git Workflow

### Branch Naming

Follow the pattern: `claude/<descriptive-name>-<session-id>`

**Examples:**

- `claude/update-claude-md-01FgsKX3pMhy1GMWM6ivoh4U`
- `claude/add-graph-curriculum-ABC123XYZ`

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add hierarchy preservation loss to training loop"
git commit -m "Fix gradient flow issue in hyperbolic convolution"
git commit -m "Implement 4-phase graph curriculum system"
git commit -m "Update CLAUDE.md to reflect current codebase structure"

# Bad
git commit -m "fix bug"
git commit -m "updates"
git commit -m "yo brah!"
```

### Push Workflow

Always push to the designated Claude branch:

```bash
# Push with upstream tracking
git push -u origin claude/update-claude-md-01FgsKX3pMhy1GMWM6ivoh4U

# If push fails due to network, retry with exponential backoff
# (2s, 4s, 8s, 16s - up to 4 retries)
```

## File Modification Guidelines

Prefer editing an existing module when it owns the behavior. Create a module for a distinct
responsibility, with tests at its interface. Configuration belongs in `conf/config.yaml` and
`utils/config.py`; panels use `conf/data/`, and HGCN uses `conf/graph.yaml`.

Keep model changes in the shared encoder/head/objective, epoch orchestration in `naics_model.py`,
monitor/cache in `monitor.py`, and campaign preflight in `checkpoint_runner.py`. Health artifacts
belong in `epoch_summary.py`; radius verification belongs in `radius_report.py`. CLI commands
call those interfaces rather than duplicating their contracts.

## Checklist for AI Assistants

- Explore requirements and constraints before nontrivial changes.
- Follow the written plan in order and surface deviations.
- Write failing behavior tests before library/pipeline changes; reproduce before debugging.
- Use fixture bundles and `tmp_path`, with no real panel reads for code verification.
- Follow Python single-quote, YAPF and semantic-divider conventions.
- Run relevant tests, full suite, Ruff and the read-only full style gate.
- Build rendered docs with strict MkDocs and verify changed links/anchors.
- Report actual output and material limitations, not inferred success.
- Update API pages and operator examples when interfaces change.

## Architecture Decisions

Functional mixins keep loss balancing, logging and optimization distinct; epoch orchestration
stays in the model. The shared backbone and projection make code fields and queries one space.
The two-stream epoch covers every query and code once while a detached code cache bounds work.
The monitor separates held-out task quality from training health and structural diagnostics.
HGCN's graph curriculum remains separate from the text objective.

The polar distance serves stable float32 text training, while Lorentz/manifold utilities serve
export, panel reads and graph callers. Pydantic validates readable YAML configs and rejects
retired settings. The checkpoint objective contract prevents old states from entering a new
objective through accidental keyword or weights migration.

## Additional Resources

- [README](README.md): system architecture and onboarding.
- [Quickstart](docs/quickstart.md) and [CLI usage](docs/usage.md): operator commands.
- [Text training](docs/text_training.md): objective, epoch, monitor and campaign.
- [HGCN training](docs/hgcn_training.md): graph refinement.
- [Test suite](tests/README.md): fixtures, coverage and test contracts.
- [Documentation](https://lowmason.github.io/naics-embedder/): rendered guides and API references.
- Repository and issues: <https://github.com/lowmason/naics-embedder>.
