# CLI Usage Guide

The CLI has `data` and `tools` groups plus the top-level `train` command. Use `--help` on any
command for its exact options. Operator examples below require the named generated artifacts.

## Installation

```bash
uv sync
uv run naics-embedder --help
```

## Data Commands

### `data preprocess`

Download and preprocess all raw NAICS data files. Each code's examples channel holds only its
examples-role index entries, per the committed role table (see `data roles`); the code's other
entries are outcome-panel queries. It also writes the redirection table: every cross-reference
row and harvested "Excluded" paragraph once, with the codes it names. Each code's exclusion
channel is built from that table, and a row that a held-out query leaks into is withheld from
the channel. A code without official description text inherits its only child's, and
`description_source` records whose text it is. An absent channel is null. Preprocessing fails if
any validation or test query matches training text or an activity phrase.

**Requires:** `conf/data/index_roles.csv`  
**Generates:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`,
`data/naics_redirections.parquet`

```bash
uv run naics-embedder data preprocess
```

**Options:**
- `--source-dir PATH` - Read the four Census files from this directory, by file name, instead of
  downloading them
- `--force` - Overwrite a descriptions file that a configured supervision bundle pins. Refused by
  default: training against that bundle fails closed once the file changes

### `data supervision`

Build one immutable, validated Stage-3 supervision bundle. It holds the codebook, the pair facts
(D*, the structural relation, directional explicit exclusions and the unary-pair flag),
legacy-compatible distance and relation views and matrices, and training pairs. It also holds
the curriculum difficulty thresholds, the index roles and the redirection table. The manifest
records the backbone's input window and each channel's texts beyond it. The manifest is written
last and the bundle directory is published atomically.

**Requires:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`,
`data/naics_redirections.parquet` (the last two become required bundle members), and the
backbone's tokenizer in the local Hugging Face cache  
**Generates:** `data/supervision/stage3-supervision-v2/<bundle-id>/` (prints
`Supervision manifest: <path>`; set `supervision.manifest_path` to it before training)

```bash
uv run naics-embedder data supervision
```

### `data relations`, `data distances`, `data triplets`

Deprecated stage commands. Publishing one partial authority would let artifacts from different
generations mix, so each prints a migration notice and exits with status 1 without building
anything; `data supervision` builds the complete bundle.

### `data all`

Run the full data generation pipeline: preprocess, then build the supervision bundle.

```bash
uv run naics-embedder data all
```

### `data roles`

Draw the frozen index-entry role table, once. Every Census index entry gets exactly one role, per
code and stratified: examples-channel text, or a training, validation or test query for the
outcome panel. No validation or test query matches any training text, exactly or as a
near-duplicate (character-trigram Jaccard of at least 0.9). The table is committed and `data
preprocess` applies it. Redrawing it unseals the validation and test splits, so an existing table
is replaced only with `--force`.

**Generates:** `conf/data/index_roles.csv`, `conf/data/index_roles_provenance.json`

```bash
uv run naics-embedder data roles --source-dir ~/Downloads/Data
```

### `data summaries`

Build the window-fitting summaries of over-long channel texts, once per backbone (roadmap
Stage 6b). Every description, examples or excluded text whose marked form (`'description: …'`,
special tokens included) is over the backbone's trained window is summarized by whole units of
the text: its sentences, or clauses and segmenter pieces of an over-long sentence, or its
examples entries. The frozen backbone (from the local Hugging Face cache) keeps, greedily, the
units whose pooled vectors best approximate the whole text's, in source order, while the marked
summary fits. Every unit boundary is a boundary of the leakage segmenter, so a summary's segments
are a subset of its text's and the leakage checks need no sealed read. The artifact is checked as
every reader checks it before it is moved into place; commit it with the pin the command prints,
in `WINDOW_SUMMARIES` (`panels/window_summaries.py`). An existing artifact is replaced only with
`--force`, and a new artifact needs a new pin.

**Generates:** `conf/data/window_summaries.csv`, `conf/data/window_summaries_provenance.json`

```bash
HF_HUB_OFFLINE=1 uv run naics-embedder data summaries
```

**Options:**
- `--descriptions PATH` - The descriptions parquet (default: `./data/naics_descriptions.parquet`)
- `--backbone NAME` - The backbone (default: `data_loader.tokenization.tokenizer_name` in
  `conf/config.yaml`)
- `--output PATH` - The artifact (default: `conf/data/window_summaries.csv`)
- `--force` - Replace an existing artifact

### `data regressor-groups`

Draw the regressor panel's held-out four-digit groups, once: a fifth of each sector's groups
(largest remainder, seeded) from the 980 six-digit codes Stage 1's finding names. The QCEW
national slices are read from `qcew_dir` under the sha256 values pinned in
`conf/data/regressor_panel.yaml`, and the codebook under its codes' fingerprint. The table is
committed, and the panel reads it only under the fingerprint pinned there as
`heldout_groups_sha256`. A redraw moves both regressor outer sets, so an existing table is
replaced only with `--force`, and the panel reads a new draw only after a reviewed change of the
pin.

**Generates:** `conf/data/regressor_heldout_groups.csv`,
`conf/data/regressor_heldout_groups_provenance.json`

```bash
uv run naics-embedder data regressor-groups --codebook PATH/naics_codebook.parquet
```

---

## Tools Commands

### `tools config`

Display the configuration validated over defaults, including run inputs and settings. The
command exits 1 on a missing or invalid file.

```bash
uv run naics-embedder tools config --config conf/config.yaml
```

### `tools visualize`

Plot the durable epoch summary. One `epoch_metrics.png` has four panels: monitor MRR, term and
total losses, both learned scales, and per-level radius means with SD bands. Structural metrics
are separate diagnostics. The tool reads no console logs or evaluation panels.

```bash
uv run naics-embedder tools visualize   --summary checkpoints/reference/epoch_summary.jsonl   --output-dir outputs/visualizations/reference
```

- `--summary PATH` is required: the run's `epoch_summary.jsonl`.
- `--output-dir PATH` defaults to `visualizations` beside the summary.

### `tools outcome-baseline`

Score the training-free lexical encoder (hashed character trigrams under cosine distance) on the
outcome panel's validation split: top-1 accuracy, MRR, Hit@1/5/10 and the level of the lowest
common ancestor of the top-1 code and the truth. The read is appended to the selection log; the
test split stays sealed.

**Requires:** `data/naics_descriptions.parquet`, `data/naics_index_roles.parquet`

```bash
uv run naics-embedder tools outcome-baseline
```

**Options:**
- `--purpose TEXT` - Why this read happens; recorded in the selection log
- `--index-roles PATH`, `--descriptions PATH` - Preprocessing outputs (default: the paths in
  `conf/data/download.yaml`)
- `--log PATH` - Selection log (default: `logs/selection_log.jsonl`, from
  `conf/data/outcome_panel.yaml`)
- `--output PATH` - Also write the summary as JSON

### `tools text-only-table`

Embed every code's text with the arm's backbone, frozen: each of the four channels is mean-pooled
over its tokens, and a code's vector is the mean of its present channels (roadmap D9). The
backbone comes from the local Hugging Face cache (default: `text_only.backbone` in
`conf/data/regressor_panel.yaml`). A channel text over the backbone's trained input window is
read as its window-fitting summary, as the arm reads it, so no channel text is truncated, and
`text_only.max_length` may not exceed the window (Req 9). The provenance records the summaries'
sha256 (`summaries`). The decision store records it, and `check_text_only` compares it with the
arm's (D9). The regressor panel reduces the table to the arm's dimension by PCA.

**Generates:** the table and `<stem>_provenance.json` beside it

```bash
uv run naics-embedder tools text-only-table --descriptions data/naics_descriptions.parquet \
  --output /tmp/text_only.parquet
```

**Options:**
- `--descriptions PATH` - The arm's descriptions parquet: the text it reads
- `--output PATH` - Where to write the table
- `--backbone NAME` - The arm's backbone (default: the regressor config's)

### `tools regressor-panel`

Score an arm on the regressor panel (roadmap Stage 3). Every comparator (covariates alone, and
the arm's coordinates, one-hot, ancestor indicators and the text-only table, each alone and with
the covariates) is fitted by ridge on standardized features, with the penalty tuned by nested
grouped folds inside the remainder. The outcome is log employment in year t + 1 from year-t
features. The seen and held-out regimes are separate panels. The validation split reads only the
remainder. The test split opens each regime's sealed outer set first, and both the opening and
the read are logged. The output holds one prediction per row, comparator and repeat (the test
split has one repeat), with the row's outcome and the chosen penalty.

```bash
uv run naics-embedder tools regressor-panel --coordinates arm.parquet \
  --text-only /tmp/text_only.parquet --codebook PATH/naics_codebook.parquet
```

**Options:**
- `--coordinates PATH` - The arm's 2,125-code table in the export form (`tools export-table`:
  tangent coordinates at the origin for a hyperbolic arm). Lorentz points and constant columns,
  such as a log map's zero time coordinate, are refused
- `--text-only PATH` - The text-only table (`tools text-only-table`)
- `--codebook PATH` - A supervision bundle's `naics_codebook.parquet`
- `--regime seen|heldout` - Regime to score (repeatable; default: both)
- `--level INT` - NAICS level 2-6 (repeatable; default: 6)
- `--split validation|test` - The test split needs `--open-purpose`, which is logged
- `--purpose TEXT` - Why this read happens; recorded in the selection log (default:
  `regressor panel <split> read`)
- `--open-purpose TEXT`, `--reopen-reason TEXT` - Why the outer sets are opened, and why again
- `--log PATH` - Selection log (default: `logs/selection_log.jsonl`)
- `--output PATH` - Write the per-row predictions as parquet

### `tools export-table`

Export a checkpoint's 2,125-code table as `code`, `index`, `level`, then float64 bounded tangent
coordinates `e0` through `e{d-1}`, in codebook order. Each code goes through the checkpoint's
model in eval mode. The current text objective uses unit curvature and
`r = R * tanh(norm(v) / R)`, rather than a cap at 2.

The supervision contract must match the configured bundle. The checkpoint's encoder record is
its own, so a dimension-8 checkpoint can export under a dimension-16 config. Pre-`req11-v1`
checkpoints are refused before model loading. Different window summaries or tokenizer pins are
refused. Export writes `<stem>_provenance.json`, with checkpoint and contract identities,
backbone revision, tokenizer/window/summaries identities, descriptions hash and table fingerprint.

Use the monitor's selected checkpoint; epoch 1 is an example here:

```bash
uv run naics-embedder tools export-table --checkpoint checkpoints/reference/epoch=001.ckpt   --output data/reference/table.parquet   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

- `--checkpoint PATH`, `--output PATH` are required.
- `--config PATH` defaults to `conf/config.yaml`.
- `KEY=VALUE ...` overrides bundle/token-cache configuration. An argument without `=` is refused.

### `tools outcome-panel`

Score the outcome validation split through the checkpoint's query encoder and the table exported
from that checkpoint. The read logs table fingerprint and checkpoint hash. Provenance must match
the checkpoint and preprocessing pins. The test split stays sealed; there is no `--split`.

```bash
uv run naics-embedder tools outcome-panel --checkpoint checkpoints/reference/epoch=001.ckpt   --table data/reference/table.parquet --purpose 'selected checkpoint validation'   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

- `--checkpoint PATH`, `--table PATH`, `--purpose TEXT` are required.
- `--config PATH` defaults to `conf/config.yaml`.
- `--log PATH` defaults to the outcome-panel configuration's selection log.
- `--output PATH` also writes JSON with panel/table/checkpoint identities and scores.
- `KEY=VALUE ...` takes the same config overrides as export.

### `tools sweep`

Keep the store and JSON records under `~/naics-artifacts`, outside every worktree.
Build one arm record from complete trained seed directories. Before the first export or decision
read, the runner checks every seed's epoch coverage, panel fingerprint, training-run id, seed,
21 settings, earliest best checkpoint and exact saved best score. A selected epoch with a
versioned sibling is ambiguous and refused. The arm's backbone revision is resolved independently
from its cached backbone; it cannot be copied from the text-only comparator.

```bash
mkdir -p ~/naics-artifacts/records/stage7
uv run naics-embedder tools sweep --runs 'checkpoints/reference-seed-{seed}' \
  --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 --seed 8 --seed 9 --seed 10 \
  --text-only data/reference/text_only.parquet --store ~/naics-artifacts \
  --output ~/naics-artifacts/records/stage7/reference.json --purpose 'reference campaign validation' \
  --name reference --accelerator cuda \
  supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

- `--runs TEXT` is required and must contain `{seed}`.
- `--seed INT` is required and repeatable; seed ids must be distinct. Decisions need at least five.
- `--text-only PATH`, `--store PATH`, `--output PATH`, `--purpose TEXT` are required.
- `--name TEXT` defaults to `reference`.
- `--accelerator TEXT` defaults to `cuda`: the training accelerator for run-settings comparison,
  even when exports and reads run on the Mac.
- `--log PATH` overrides the selection log; `--config PATH` defaults to `conf/config.yaml`.
- `KEY=VALUE ...` configures the bundle, encoder and training settings of the arm.

Each seed exports its selected checkpoint and reads outcome validation plus the seen and held-out
regressor regimes at level 6. The record stores all monitor reads, selected epoch and training-run
id, as well as decision reads and content-addressed artifacts. Test splits stay sealed. Fix
reference margins before candidate monitor or decision reads; see
[the campaign workflow](text_training.md#reference-campaign).

### `tools radius-report`

Read a checkpoint and its exported table on the CPU, without reading an evaluation panel. The
report checks live anchor-radius gradients, per-level SD > 1e-3, positive distinct sector radii
and their least gap, manifold error at the largest radius, and chunked all-pairs float32/float64
distance agreement. It reports nonzero gradient norms for all three weighted terms and both
scales on the saved seed's epoch-zero, step-zero batch.

```bash
uv run naics-embedder tools radius-report --checkpoint checkpoints/reference/epoch=001.ckpt   --table data/reference/table.parquet --output data/reference/radius_report.json   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

- `--checkpoint PATH`, `--table PATH` are required and must have matching provenance.
- `--output PATH` also writes the JSON report.
- `--config PATH` defaults to `conf/config.yaml`; `KEY=VALUE ...` supplies config overrides.

A failed criterion writes the report and exits 1. Distance relative error must be at most 1e-3
on noncoincident pairs. Coincident tangents must have zero float32 training distance; their
float64 residual is reported separately. Largest-radius manifold error must be at most
`1e-9 * x0**2`. This is a verification report, not a panel decision.

### `tools margins`

Fix each panel's non-inferiority margin δ from a reference arm (Req 5): δ is `--multiple` times
the reference's across-seed standard deviation of the panel's decision statistic (roadmap D10:
per-query MRR on the outcome panel, and the `covariates+embedding` comparator's mean squared
error on each regressor regime at level 6). Fix the margins before any other arm of the decision
reads a panel, including its training monitor reads: a decision refuses every run that read
before its margins were fixed.

**Generates:** the margin record (JSON)

```bash
uv run naics-embedder tools margins \
  --reference ~/naics-artifacts/records/stage7/reference.json --multiple 3 \
  --name reference-margins --store ~/naics-artifacts \
  --output ~/naics-artifacts/records/stage7/margins.json
```

**Options:**
- `--reference PATH` - The reference configuration's arm record, written by the seed-sweep
  driver (`naics_embedder.decision.sweep.run_seed_sweep`)
- `--multiple FLOAT` - Each δ as a multiple of the reference's across-seed standard deviation
- `--name TEXT` - Names the margins in the decision records that use them
- `--store PATH` - The artifact store the arm record references
- `--output PATH` - Where to write the margin record; an existing file is refused before any
  work, never overwritten

### `tools decide`

Decide among two or more arms under Req 5's rule over the three panels of roadmap D8: the outcome
panel and the regressor panel's seen and held-out regimes at level 6. Each comparison reads the
difference Δ on paired resamples of each panel's units (codes with their queries; four-digit
groups), with each arm's seeds resampled inside every replicate. A is adopted over B when it is
non-inferior on all three panels (the 95 % interval's lower bound above −δ) and superior on at
least one (the 98⅓ % interval above zero). The survivors are the arms no other arm is adopted
over, and the tie order picks among them: fewer components, then lower dimension, then
non-hyperbolic geometry, then the higher held-out gain over ancestors (roadmap D11). The number
of replicates, the bootstrap seed and the seed floor come from `conf/data/decision.yaml`.

**Generates:** the decision record (JSON): the arms with their selection-log records and artifact
references, the margins, every comparison with both intervals, the non-dominated set, the tie
order, the chosen arm, and each arm's reported gains over the sparse comparators

```bash
uv run naics-embedder tools decide \
  --arm ~/naics-artifacts/records/stage7/candidate.json \
  --arm ~/naics-artifacts/records/stage7/reference.json \
  --margins ~/naics-artifacts/records/stage7/margins.json \
  --name dimension-8 --question "Is dimension 8 enough?" \
  --store ~/naics-artifacts --output ~/naics-artifacts/records/stage7/decision.json
```

**Options:**
- `--arm PATH` - An arm record; repeat for each arm
- `--margins PATH` - The margin record (`tools margins`)
- `--name TEXT`, `--question TEXT` - Name the decision and say what it settles
- `--store PATH` - The artifact store the arm records reference
- `--output PATH` - Where to write the decision record; an existing file is refused before any
  work, never overwritten

### `tools diagnostics`

Report Req 6's structural diagnostics over every codebook code: sector separation (an AUC),
within-sector rank correlation (over queries and over sectors), MAP over ancestors,
NDCG@5/10/20 with integer lowest-common-ancestor grades, the Pearson correlation of distance
with the tree metric D*, and parent retrieval@1/5 without the 522 unary pairs. The report
describes an arm: nothing selects on it, and no statistic in it has a threshold. To compare two
tables, such as one before and one after graph refinement, report on each.

```bash
uv run naics-embedder tools diagnostics --table arm.parquet --geometry hyperbolic \
  --codebook PATH/naics_codebook.parquet
```

**Options:**
- `--table PATH` - The arm's code table in the export form (tangent coordinates at the origin
  for a hyperbolic arm; Lorentz points are refused)
- `--geometry euclidean|spherical|hyperbolic` - The arm's distance: Euclidean, cosine, or the
  geodesic distance after the exponential map at the origin
- `--codebook PATH` - A supervision bundle's `naics_codebook.parquet`; the table must hold
  exactly its codes
- `--curvature FLOAT` - A hyperbolic arm's curvature magnitude (default: 1.0)
- `--output PATH` - Also write the report as JSON

---

## Training Commands

### `train`

Train the shared reference encoder with task, code-to-code and radial terms. The epoch has two
streams and a per-epoch code cache; it has no text curriculum or validation loader. The outcome
validation monitor's `val/outcome_mrr` selects the earliest best checkpoint and drives plateau
scheduling and early stopping.

```bash
uv run naics-embedder train --config conf/config.yaml   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

- `--config PATH` defaults to `conf/config.yaml`.
- `--ckpt-path PATH` supplies an exact checkpoint; `last` resolves the current experiment's
  latest checkpoint. Continue only `last`, rather than rewinding to an older kept epoch.
- `--checkpoint-load-mode TEXT` accepts only `exact`; pre-objective checkpoints and weights-only
  migration are refused.
- `--skip-validation` skips advisory data/cache checks. The immutable supervision gate always runs.
- `KEY=VALUE ...` overrides configuration, such as `data_loader.queries_per_step=64` or
  `training.learning_rate=1e-5`. A fresh run needs a distinct experiment name.

Defaults are experiment `reference`, dimension 16, masked mean, radius bound 8 and 128 queries
per step. CUDA uses `bf16-mixed` for the backbone while fusion, projection, head, distances and
losses stay float32. CPU and MPS use `32-true`. Text training uses one device; `devices > 1` is
refused. `training.trainer.val_check_interval` is retained but unread.

Fresh starts refuse used checkpoint directories. Exact resume requires the same experiment
directory, bundle, encoder, preprocessing, seed and all 21 run settings, including the epoch
budget. The code cache is rebuilt; `monitor_reads.jsonl` and `epoch_summary.jsonl` retain lines
through the resumed epoch and then continue. A run that early stopping ended exits 1 with
`early stopping ended the run at epoch k`. A spent epoch budget is a harmless no-op. Launchers
must treat the early-stopping message as finished rather than retrying it.

```bash
uv run naics-embedder train --ckpt-path last   supervision.manifest_path=/absolute/path/to/<bundle-id>/manifest.json
```

See [exact resume](text_training.md#exact-resume) for artifacts and guards. The former sequential
stage-chain workflow is removed. Run HGCN separately with
`uv run python -m naics_embedder.graph_model.hgcn`; see [HGCN training](hgcn_training.md).

## Getting Help

```bash
uv run naics-embedder --help
uv run naics-embedder data --help
uv run naics-embedder tools --help
uv run naics-embedder train --help
uv run naics-embedder tools sweep --help
uv run naics-embedder tools radius-report --help
uv run naics-embedder tools visualize --help
```

## Configuration Files

`conf/config.yaml` defines the text encoder, objective, two-stream loader and training settings.
Panel pins and the selection log are configured under `conf/data/`; HGCN has `conf/graph.yaml`.
See [the configuration API](api/config.md) and [text training](text_training.md).
