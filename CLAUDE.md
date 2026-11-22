# CLAUDE.md - AI Assistant Guide for NAICS Embedder

## Project Overview

NAICS Embedder serves three artifacts over the bundle's 2,125 NAICS 2022 codes. Each is judged
only on evidence for its own use (user adjudication Q4, `specs/naics-artifact-redesign.md`, which
redefines the artifacts of Q3's `specs/naics-representation-split.md`):

- **Relatedness:** one embedding of how industries relate in practice, trained on the tree's
  edges and on embedding-data's observed-relation families, and judged on families it never
  trained on.
- **Text→code:** retrieval of a ranked shortlist of six-digit codes for an owner's own
  description of an establishment, judged on ITA employer-written descriptions.
- **Economic:** low-dimensional coordinates for downstream models, judged on economic panels, the
  first being national employment growth.

The NAICS tree is not an artifact. It is the codebook that all three use. The trained text model
places classification codes and activity queries in one space. One LoRA-adapted text backbone,
masked fusion and one projection feed a live radius/direction head. The text objective jointly
learns task retrieval, code geometry and radial structure. An optional HGCN stage refines the
parent–child graph with its own objective and curriculum.

The redesign's stages are in `specs/naics-artifact-redesign-roadmap.md`, one chain per artifact
(`specs/adr/0002-one-roadmap-with-a-chain-per-artifact.md`). Stages that read embedding-data
tables are held until that repository's gates open. In `specs/naics-embedding-roadmap.md`, Stage 9
is held, the graph stage (Stages 10–11) is parked and Stage 12 is one sealed opening per judged
artifact. The economic artifact's first roadmap is complete
(`specs/completed/naics-representation-split-roadmap.md`).

The project uses Python 3.10+, uv, PyTorch/Lightning, Transformers/PEFT, Polars/PyArrow, Pydantic,
Typer/Rich, pytest, Ruff/YAPF and MkDocs. `.python-version` pins the local locked environment to
Python 3.12. The source tree has **136 Python files**; tests have **121 unit** files and two
integration files. File counts exclude generated and ignored artifacts.

## Architecture Summary

1. **Shared text encoding** (`text_model/fields.py`, `shared_encoder.py`): one LoRA backbone reads
   marked title, description, examples, exclusions and queries. Blank code fields never enter it.
2. **Fusion and projection** (`text_model/fusion.py`): masked mean by default, attention or MoE
   as options, then one `Linear(384, d)` for d in {8, 16, 32}, default 16. MoE is an ablation.
3. **Live radius and objective** (`text_model/hyperbolic.py`, `loss.py`, `naics_model.py`):
   `r = R * tanh(norm(v) / R)` with default R = 8, and the projection's direction, form the
   unit-curvature Lorentz point. Task, code-code and radial terms all retain radius gradients.
   `model.geometry` (Req 12) can instead select a Euclidean or spherical head
   (`text_model/heads.py`). These share the encoder and the task and code-code terms, decode by
   their own distance and have no radial term. Hyperbolic is the default.
4. **Graph refinement** (`graph_model/hgcn.py`): HGCN uses the graph, triplet/radial objectives
   and its four-phase graph curriculum. Its curvature utilities remain separate from text.
5. **Economic candidates** (`tree_model/`, `decision/candidates.py`): three deterministic builders,
   none trained, each at d in {8, 16, 32}: the text-only table reduced by PCA, the classical
   scaling of the tree metric D*, and the text-only vectors smoothed over each code's lineage.
   `conf/data/economic_candidates.yaml` freezes the 18 candidates the economic decision reads,
   these nine and nine trained Stage 7/8 arms. A candidate reads the codebook's text and tree only
   (`specs/completed/economic-candidates.md`).

The text model has no curvature parameter, DCL term, false-negative clustering, negative miner,
router-guided sampler, text phase curriculum, distributed cache or in-sample validation loader.
The HGCN stage retains its graph-specific samplers and curriculum.

`tools export-table` writes each code's bounded tangent coordinates at the origin, `e0` through
`e{d-1}`, in Req 2's form; a flat arm writes v (Euclidean) or its direction (spherical), and
the provenance names the geometry. Panel reads reconstruct Lorentz points on the CPU in float64,
or read a flat arm's coordinates as they are.
The selected text checkpoint is the earliest epoch with the highest outcome validation MRR.

## Directory Structure

Paths below are relative to the repository; the tree shows the principal modules.

```text
src/naics_embedder/
├── cli/commands/              # data, tools, training and remote CLI adapters
├── remote/                    # verified code/input transport, exact launch and coherent pulls
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
│   ├── heads.py               # Euclidean and spherical heads, flat distances, build_head
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
├── panels/                    # outcome/regressor panels and selection-log guards;
│                              # ita_split.py: the ITA panel's settings and salted split;
│                              # ita_evidence.py: the pinned ITA run and its C1 check;
│                              # ita.py: the ITA panel, its pair tables and sealed opening;
│                              # ita_scoring.py: the ITA shortlist ranks and summaries;
│                              # relatedness_split.py: the relatedness panel's settings and split;
│                              # relatedness.py: the relatedness panel and its sealed opening;
│                              # relatedness_scoring.py: tie-aware NDCG and AUC for relatedness;
│                              # relatedness_arms.py: the tree and text-only arms' similarities
├── decision/                  # paired resampling, margins, arm records and decisions;
│                              # economic.py: the growth margin read and economic decision;
│                              # candidates.py: the frozen candidate list;
│                              # provenance.py: the text-and-tree-only rule;
│                              # trained.py: a trained candidate's import;
│                              # text_to_code.py: text→code's margin and decision on the ITA panel;
│                              # ita_reads.py: Stage 4's reference reads on the ITA panel;
│                              # ita_gap.py: the gap between Census phrases and owner text;
│                              # relatedness.py: relatedness's margin, test and provenance rule
├── tree_model/                # deterministic economic candidates: scaling, smoothing, builders
├── graph_model/               # HGCN, graph data loading and four-phase curriculum
├── losses/                    # level_radius.py: HGCN's per-level radial loss
├── metrics/                   # structural and retrieval diagnostics
├── tools/                     # config display, epoch-summary plotting and metrics API
└── utils/                     # config, training, input-window and geometry utilities

tests/
├── unit/                      # 121 unit test files
├── integration/               # test_reference_training.py, test_remote_workflow.py
├── fixtures/                  # tiny models, bundles, panels and runs
└── conftest.py

conf/config.yaml               # reference text configuration
conf/graph.yaml                # separate HGCN configuration
conf/remote.yaml               # remote transport settings
conf/local.example.yaml        # form of the gitignored conf/local.yaml: this machine's manifests
conf/data/                     # frozen panel pins, roles, summaries and held-out groups
docs/                          # guides and MkDocs API pages
scripts/format_code.sh         # Ruff fixes/imports, then YAPF layout
.claude/rules/                 # path-scoped rules: economic, text-to-code, relatedness, remote
```

Generated data, checkpoints, logs and reports are separate artifacts. Do not confuse historical
training logs or a real panel campaign with fixture verification.

## Scoped Rules

Each artifact's mechanics and the remote workflow's contracts live in `.claude/rules/`. Claude
Code loads a rule when it reads or edits a file that the rule's `paths` match. A rule does not
load for a command run or a diff review, so this file keeps every guardrail. Other agents, Codex
included, read the rule before they change its area:

- `.claude/rules/economic.md`: regressor outcome modes, economic reads, `decide` and pairing.
- `.claude/rules/text-to-code.md`: the ITA panel's reads and tools, and their traps.
- `.claude/rules/relatedness.md`: the relatedness panel's scores and status, and its traps.
- `.claude/rules/remote.md`: the remote adapters, launch requirements, model cache and Plan 10.

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
error from live radius to `radial_step * (level - 1)`; it exists only in the hyperbolic arm, and
a flat arm trains on the other two terms (Req 12). Both term weights and radial step default
to 1. Task and code-code logits use independent learned positive scales, clamped to [0.01, 100]
without weight decay. The total adds MoE balancing only under that fusion.

### Radius and Precision

The head computes `r = R * tanh(a / R)`, with a = norm(v), then maps the bounded tangent r u to
`(cosh(r), sinh(r) u)`. Its origin guard handles zero v. Text curvature is fixed at 1 with no
setting or learned parameter. In the hyperbolic arm, radius r is the radial quantity in text
losses, health and reports.

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
logit scales, and radius mean/SD at levels 2–6. `loss/radial` appears only for the hyperbolic
arm. MoE includes load-balancing loss. Health values come from `_log_health()`'s Python floats,
not rounded Lightning callback metrics.
`tools visualize --summary PATH` writes one `epoch_metrics.png` figure with four panels.

Structural metrics are no longer text validation or scheduler controls. `tools diagnostics`
reports Req 6 without thresholds or adoption. HGCN keeps structural logging and its graph
curriculum. The outcome and two regressor panels decide adoption under Req 5.

Text→code's own judge is the ITA panel (`ita_text`, `panels/ita.py`), which scores ITA
employer-written descriptions. Its pairs split by a salted hash of `description_key` into a
validation part, which selects, and a sealed part. The sealed part opens only through
`ItaPanel.open_sealed` and is read once, at embedding roadmap Stage 12. The outcome panel's
validation split is reported beside it and decides nothing for text→code, and its test split
stays sealed. No ITA pair enters training, and no code or test reads a real pair. The reads, the
tools and the traps are in `.claude/rules/text-to-code.md`.

Relatedness's own judge is the relatedness panel (`relatedness_co_ownership`,
`panels/relatedness.py`), which scores ITA co-ownership. Its anchor industries split by a salted
hash of the six-digit code into a validation part, which selects, and a sealed part. The sealed
part opens only through `RelatednessPanel.open_sealed` and is read once, at redesign roadmap
Stage 7b. No co-ownership row enters training, and no code or test reads a real table. Nothing
logs a read of either panel's raw sealed rows (`ItaPanel`'s pairs, `panel.co_ownership`), so a
review must catch one. The scores, the status and the traps are in `.claude/rules/relatedness.md`.

### Checkpoint Contract and Exact Resume

The contract identifies objective `req11-v1`, bundle, encoder and preprocessing. Old objective
checkpoints are refused before model loading in training, export, outcome reads and the HGCN
feeder. Nothing migrates weights into this objective.

Exact resume also compares saved constructor hyperparameters against the current config: LoRA
rank/alpha/dropout always, and MoE expert count/top-k/hidden dimension/load-balancing coefficient
under active MoE fusion. Missing required values fail closed; inactive MoE controls are ignored.
Campaign preflight checks both last and selected checkpoints before exports or decision reads.
The literal 21-key `run_settings` identity and finished artifacts remain unchanged.

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

The regressor panel reads `level` by default and `growth` on the same rows and folds. One opening
of a regime's outer set counts for both outcomes (Req 22). `tools regressor-panel` and
`tools sweep` read level mode only, so no candidate's growth error exists before the economic
margin (Req 24). `tools decide` refuses a candidate that read before the margins, and paired reads
must share one `uv.lock` on macOS arm64. The growth margins were fixed once, on 2026-10-17
(`specs/findings/growth-margins.md`). Never run `tools growth-margins` again: nothing refuses a
second run, and every economic candidate is decided against that record. The economic decision
was recorded on 2026-11-18 (`specs/findings/economic-decision.md`): no candidate is adopted, and
the reference `covariates+ancestors` stands. The mechanics are in `.claude/rules/economic.md`.

### Remote Workflow and Phase 2 Boundary

Five `remote` CLI adapters call `RemoteWorkflow`: up/train/sync/finish/status. Canonical inputs
travel explicitly and are never rebuilt on an instance. Only safe=true finish prints
`Safe to terminate`; abandon records possible loss and stays unsafe. The adapters' contracts,
launch requirements, model cache and Plan 10 history are in `.claude/rules/remote.md`. See
[remote workflow](docs/remote_workflow.md) and [remote API](docs/api/remote.md).

Plan 10's qualification covers the recorded private source only: native readiness for a public
release or a later campaign needs its own qualification on a new experiment, never
`plan10_smoke`, and stays an open item in `specs/deferred_items.md`. These fixtures execute no
real Lambda campaign. A session on the primary checkout sets the manifests in its gitignored
`conf/local.yaml`; no private config commit is replayed or held. Held config SHAs that completed
plans and findings cite stay reachable locally under `refs/held/*` in the primary clone; never
push them. Exports, QCEW, stores and decisions stay on the Mac; remote selection logs stay
separate and sealed test/outer splits stay closed.

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
uv run naics-embedder train
uv run naics-embedder train --ckpt-path last
uv run naics-embedder tools config
uv run naics-embedder tools visualize --summary checkpoints/reference/epoch_summary.jsonl
uv run naics-embedder tools sweep --help
uv run naics-embedder tools radius-report --help
uv run naics-embedder tools diagnostics --help
```

`data supervision` prints the exact manifest path. Put it in the gitignored `conf/local.yaml`
(copy `conf/local.example.yaml`): `config.supervision.manifest_path` for training, tools and
remote, and `graph.supervision_manifest_path` for HGCN. It may set only those two keys, each a
repo-relative path under `data/`. Commands read it after the tracked YAML and before `key=value`
overrides; `tools config` marks a manifest it set. `NAICS_EMBEDDER_LOCAL_CONFIG=off` ignores the
file and any other value names a file to read instead; the test suite sets `off`. Retired
relations/distances/triplets commands build nothing and exit 1. The manifest gate cannot be
skipped. HGCN reads `conf/graph.yaml`, then the override's graph manifest, and runs with
`uv run python -m naics_embedder.graph_model.hgcn`.

### Running Tests

```bash
uv run pytest -n auto -q
uv run pytest tests/unit/test_loss.py
uv run pytest --cov=naics_embedder
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/naics-py310 uv run pytest -n auto
```

The current suite collects **5,636 tests**. Actual skip counts depend on local data and hardware
capabilities, including MPS. Tests use fixture data and tiny models. Collection counts are not
coverage percentages. Do not read real sealed splits or run a real campaign to verify
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
logger.warning('Skipping hierarchy metrics: node codes unavailable')
logger.error('Failed to load checkpoint')
```

### Type Hints

Use type hints for function signatures:

```python
def radial_loss(radius: torch.Tensor, levels: torch.Tensor, radial_step: float) -> torch.Tensor:
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
uv run naics-embedder train experiment_name=reference-small \
  data_loader.queries_per_step=64 training.learning_rate=1e-5
```

Do not reintroduce removed objective/mining/curvature settings or bypass bundle validation.
`tools config` validates the YAML over defaults and displays the run inputs and settings.
`conf/local.yaml` is not a settings channel: it sets only the two supervision manifests, and
refuses any other key, because a remote instance never sees it.

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

There are 121 unit files and two integration files. Important current seams include:

- `test_supervision_queries.py` and `test_supervision_code_targets.py`: query/target identities
  and unary masks. `test_loss.py` and `test_hyperbolic.py`: three terms, live-radius head and
  stable polar distance. `test_heads.py`: the geometry arms' heads, distances and read maps.
- `test_datamodule.py`, `test_monitor.py`, `test_selection_log_guard.py`: two-stream epochs,
  candidate cache, monitor reads and fail-closed selection-log guards.
- `test_cli_training.py`, `test_utils_training.py`, `test_checkpoint_contract.py`:
  contracts, exact resume, stopped runs and settings guards.
- `test_checkpoint_runner.py`, `test_radius_report.py`, `test_epoch_summary.py`,
  `test_visualize_metrics.py`: campaign preflight, radius checks and durable health artifacts.
- `test_local_config.py`: `conf/local.yaml`'s two keys and `data/` paths, its place between the
  tracked YAML and overrides, its root and env var, and the graph config's explicit fields.
- Each artifact's tests: the economic, text→code and relatedness rules in `.claude/rules/` list
  them. All of them read fixture tables only.
- `integration/test_reference_training.py`: tiny-backbone Trainer runs, earliest-best selection,
  schedule/early stopping, health records and exact resume.
- Graph, panel, resampling and decision tests cover their separate interfaces.

Keep generated fixture outputs under `tmp_path`. Use reference fixture bundles instead of real data.
`tests/conftest.py` sets `NAICS_EMBEDDER_LOCAL_CONFIG=off` before any fixture runs, so no test reads
a developer's `conf/local.yaml`; `test_local_config.py` turns it on in a tmp checkout. Run
meaningful target tests and the full suite before claiming completion. Measure coverage with
pytest-cov when needed; test counts alone do not establish it.

### Platform Gotchas

- Rich output wraps at 80 columns on CI but not locally. Match CLI output on
  `output.replace('\n', '')`, and reproduce a CI failure with `COLUMNS=80`.
- On MPS, `.to(device='cpu', dtype=torch.float64)` raises. Use `.cpu().to(torch.float64)`.
  CI has no MPS, so guard such code with a test that skips unless MPS is present.
- `pl.DataFrame(array, schema=...)` transposes a square array. Pass `orient='row'`, and avoid
  fixtures where the code count equals the dimension.
- `addopts` sets `-v`, so count tests with `uv run pytest --collect-only -q -q`.

## CI/CD and Documentation

`.github/workflows/tests.yml` runs on pull requests and pushes to main/master. With locked
dependencies, it runs Ruff and YAPF in a Python-3.12 `lint` job, pytest/coverage in a `test` job
on Python 3.10 and 3.12, and `mkdocs build --strict` in a Python-3.12 `docs` job, so PR CI builds
the docs strictly. The GitHub Pages deploy, `.github/workflows/docs.yml`, runs only on pushes to
main/master or a manual dispatch, so PR CI never exercises a `docs.yml` edit until it merges. Run
these checks, including the strict docs build, locally before pushing:

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

Name a branch `claude/<descriptive-name>`. Desktop-app worktrees add a six-character hex suffix,
for example `claude/naics-embedder-execution-f9a8b6`. Cut every branch from `origin/main` and
land it by PR.

### Commit Messages

Use Conventional Commits, `type(scope): subject`. The types are `feat`, `fix`, `test`, `docs`,
`chore`, `refactor` and `style`. The scope names the package or area:

```bash
git commit -m 'feat(decision): refuse a relatedness table whose provenance names a judging input'
git commit -m 'test(decision): pin that each judging family takes its own anchors'
git commit -m 'docs(specs): complete plan 21, stamp Stage 6 and file three deferred items'
git commit -m 'chore(specs): retire plan 21'
```

### Push Workflow

Push only the named branch, with `git push -u origin <branch>`. Never push `main` or `refs/held/*`.
The primary clone's local pre-push hook refuses held commits, a pinned manifest or `/Users/` path
in the config files, and `conf/local.yaml`. Never bypass it with `--no-verify`.

## File Modification Guidelines

Prefer editing an existing module when it owns the behavior. Create a module for a distinct
responsibility, with tests at its interface. Configuration belongs in `conf/config.yaml` and
`utils/config.py`; panels use `conf/data/`, and HGCN uses `conf/graph.yaml`. This machine's two
supervision manifests go in the gitignored `conf/local.yaml`, never in tracked YAML.

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
