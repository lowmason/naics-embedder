# NAICS Hyperbolic Embedding System

<!-- markdownlint-disable MD013 -->

[![PyPI Version](https://img.shields.io/pypi/v/naics-embedder)](https://pypi.org/project/naics-embedder/) [![GitHub Release](https://img.shields.io/github/v/release/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/releases) [![PyPI Downloads](https://img.shields.io/pypi/dm/naics-embedder)](https://pypi.org/project/naics-embedder/) [![License](https://img.shields.io/github/license/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/blob/main/LICENSE) [![Documentation](https://github.com/lowmason/naics-embedder/actions/workflows/docs.yml/badge.svg)](https://github.com/lowmason/naics-embedder/actions/workflows/docs.yml) [![Tests](https://github.com/lowmason/naics-embedder/actions/workflows/tests.yml/badge.svg)](https://github.com/lowmason/naics-embedder/actions/workflows/tests.yml) [![Coverage](https://codecov.io/gh/lowmason/naics-embedder/branch/main/graph/badge.svg)](https://github.com/lowmason/naics-embedder/main/graph) [![Issues](https://img.shields.io/github/issues/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/issues) [![Last Commit](https://img.shields.io/github/last-commit/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/commits/main) [![Contributors](https://img.shields.io/github/contributors/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/graphs/contributors) [![Repo size](https://img.shields.io/github/repo-size/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder) [![Top language](https://img.shields.io/github/languages/top/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder)


The system learns a shared hyperbolic space for NAICS codes and activity queries. One
LoRA-adapted transformer reads marked fields; masked fusion and one linear projection produce
live direction and radius. Text training optimizes query decoding, code geometry and radius
jointly. Optional HGCN refinement then uses the explicit parent–child graph.

## Architecture

1. **Shared text encoding:** the same backbone reads title, description, examples, exclusions
   and queries. Absent code fields never enter the backbone.
2. **Fusion and projection:** masked mean by default, attention or MoE as options, then one
   `Linear(384, d)`, where d is 8, 16 or 32 (default 16).
3. **Text objective:** task cross-entropy, code-to-code soft-target cross-entropy and radial
   error, with an auxiliary load-balancing term only under MoE.
4. **HGCN refinement:** the separate graph curriculum and triplet objective refine the exported
   code geometry. Adoption is decided on the three evaluation panels.

```text
Marked fields/queries -> shared LoRA backbone -> masked fusion -> Linear(384, d)
    -> live radius and direction -> task + code-to-code + radial terms
    -> earliest best outcome-MRR checkpoint -> tangent table -> optional HGCN
    -> outcome, regressor-seen and regressor-heldout decision
```

## Live Radius and the Objective

For projection v with norm a and direction u, the parameter-free head computes
`r = R * tanh(a / R)` with `R = model.radius_bound` (default 8), then maps r u to the
unit-curvature Lorentz hyperboloid. Zero v maps to the origin. Text curvature is fixed at 1,
with no curvature setting. Training distances use the stable polar form in float32; panel
reads reconstruct Lorentz points and compute distances on the CPU in float64.

`task_loss` decodes each training query to its named code targets. Its candidates are every
code at the query's level plus every explicit referring code. `code_code_loss` matches a
softmax of negative taxonomy distance over all codes except the anchor and its unary partner.
`radial_loss` softly targets `radial_step * (level - 1)` while keeping gradients through each
radius. Both code-code and radial weights default to 1. Task and code-code logit scales are
learned independently, start at 1 and are clamped to [0.01, 100] without weight decay.

There is no text DCL loss, negative miner, false-negative clustering or phase curriculum.
The reference uses two streams: every code anchor and every eligible query once per epoch.
With 11,039 queries and 128 queries per step, the reference epoch has 87 steps. A detached code
cache is refreshed at fit start and epoch end; each step replaces its anchor rows with live
embeddings. The training-pairs bundle member stays validated but is unread by text training.

## Monitor and Artifacts

After each cache refresh, the outcome validation panel's float64 MRR becomes `val/outcome_mrr`.
It selects the earliest best epoch and controls plateau scheduling and early stopping. The
model writes `monitor_reads.jsonl` and `epoch_summary.jsonl` beside the checkpoints. The summary
contains MRR, three term losses, total loss, both scales and per-level radius mean/SD.

The selected checkpoint is `epoch=NNN.ckpt`; `last.ckpt` holds the latest training state.
Checkpoints identify objective `req11-v1`, bundle, encoder, preprocessing, seed and 21 run
settings. Old objectives are refused before model loading; there is no weights-only migration.
See [the training guide](docs/text_training.md) for the exact contracts and resume rules.

## Quick Start

```bash
git clone https://github.com/lowmason/naics-embedder.git
cd naics-embedder
uv sync
uv run naics-embedder data all
uv run naics-embedder train --config conf/config.yaml   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

`data all` prints the immutable manifest path. The mandatory gate validates the bundle before
model or data-loader construction. `--skip-validation` only skips advisory checks. Retired
`data relations`, `data distances` and `data triplets` commands build nothing and exit 1.

CUDA uses `bf16-mixed` for the backbone; fusion, projection, geometry and losses remain float32.
CPU and MPS use `32-true`. The text Trainer uses one device.

```bash
uv run naics-embedder train --ckpt-path last   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
uv run naics-embedder tools visualize   --summary checkpoints/reference/epoch_summary.jsonl
```

Resume only `last`, with the same experiment directory, bundle, encoder, preprocessing, seed
and settings. An early-stopped run exits 1 with `early stopping ended the run at epoch k`;
a spent epoch budget trains no further epochs. Changing the budget or other settings requires
a fresh run directory. The code cache is rebuilt, and both JSONL files continue from the
restored epoch.

## Export and Radius Verification

Export the earliest highest-MRR checkpoint identified by the monitor. This example uses epoch 1:

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/reference/epoch=001.ckpt   --output data/reference/table.parquet   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
uv run naics-embedder tools radius-report --checkpoint checkpoints/reference/epoch=001.ckpt   --table data/reference/table.parquet --output data/reference/radius_report.json   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

The table has `code`, `index`, `level`, and bounded tangent coordinates `e0` through `e{d-1}`
in codebook order. Provenance binds it to the checkpoint and preprocessing identities.
The radius report checks live gradients, per-level spread, sector radii, manifold validity,
distance precision and nonzero gradients for the three terms and both scales. A failed
criterion writes the report and exits 1; the report does not read an evaluation panel.

## Three-Panel Campaign

Train ten seeds with the same reference settings, then run `tools sweep` over their complete
run directories. It checks every seed before its first export or decision read. Each seed
contributes outcome validation and the regressor panel's seen and held-out regimes at level 6;
the arm record carries the monitor reads that selected its checkpoint.

Fix reference margins with `tools margins --multiple 3` before candidate monitor or decision
reads. `tools decide` applies Req 5's paired resampling, non-inferiority, superiority and tie
order. The panel test splits stay sealed. See
[the reference campaign](docs/text_training.md#reference-campaign) for complete commands.

## Graph Refinement and Structural Diagnostics

HGCN retains its four-phase graph curriculum, triplet objective and curvature utilities. Set
its bundle manifest in `conf/graph.yaml`, then run:

```bash
uv run python -m naics_embedder.graph_model.hgcn
```

See [HGCN training](docs/hgcn_training.md) for input artifacts and graph configuration.
HGCN structural statistics remain descriptive. `tools diagnostics` reports Req 6's sector
separation, within-sector rank correlation, ancestor MAP, LCA-graded NDCG, distance/D* Pearson
correlation and parent retrieval without unary pairs. They neither select the text checkpoint
nor replace a three-panel decision.

`structural-spearman-v1` validates symmetric distance matrices, averages mirrored values in
CPU float64 and uses strict upper-triangle pairs with average ranks for exact ties. Undefined
results have an explicit status and JSON `null`. Historical unversioned Spearman fields are
`legacy-ordinal-rank-v0` and are not directly comparable. See the
[metric contract](docs/overview.md#structural-spearman-v1).

## Development

The project pins Python 3.12 for its locked environment. Tests use fixture bundles and tiny
models; they do not perform the real training or selection campaign.

```bash
uv run pytest -n auto -q
uv run ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run mkdocs build --strict
```

The suite contains 2,285 tests in 83 unit files and one integration file. On a host with MPS,
2,283 pass and two skip; without MPS, 2,276 pass and nine skip. Coverage is measured separately,
not inferred from these counts. See [tests/README.md](tests/README.md) for test contracts and
[CLAUDE.md](CLAUDE.md) for project conventions.
