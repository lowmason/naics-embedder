# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

**NAICS Hyperbolic Embedding System** learns Lorentz-model hyperbolic embeddings for NAICS codes using a four-stage pipeline:

1. **Multi-channel text encoding** (`src/naics_embedder/text_model/encoder.py`)
   - Separate LoRA-adapted transformer encoders for title, description, examples, and exclusions.
2. **Mixture-of-Experts fusion** (`text_model/moe.py`)
   - Top‑2 gated experts fuse the four channel embeddings into one Euclidean embedding.
3. **Hyperbolic contrastive learning** (`text_model/naics_model.py`, `text_model/loss.py`)
   - Projects into Lorentz space and optimizes a decoupled contrastive loss with hierarchy-aware regularizers and a dynamic curriculum (SADC).
4. **Hyperbolic graph refinement (HGCN)** (`graph_model/hgcn.py`)
   - Hyperbolic GCN over the explicit NAICS parent–child graph with its own curriculum and hierarchy-aware metrics.

The codebase is organized around this pipeline plus a rich config/validation layer and a CLI.

## Code architecture (big picture)

### Top-level layout

- `src/naics_embedder/cli.py`, `src/naics_embedder/cli/`  
  Typer-based CLI. The top-level command is `naics-embedder`, with three groups:
  - `data` – data generation / preprocessing
  - `tools` – config/metrics utilities (including Stage 4 verification)
  - `train` – text-model training entrypoint

- `conf/config.yaml` + `src/naics_embedder/utils/config.py`  
  Single source of truth for configuration via Pydantic models (`Config`, `GraphConfig`, etc.).  
  All training/data/graph/curriculum settings flow through this file and these models rather than being hard-coded.

- `src/naics_embedder/text_model/` (Stages 1–3)
  - `encoder.py` – multi-channel encoder wrapper around the base transformer with LoRA.
  - `moe.py` – Top‑2 gated Mixture-of-Experts for channel fusion.
  - `hyperbolic.py` – low-level Lorentz operations (exp/log maps, distance, projection).
  - `loss.py` – decoupled contrastive loss plus hierarchy/rank/radius regularizers.
  - `curriculum.py`, `hard_negative_mining.py`, `hyperbolic_clustering.py`, `false_negative_strategies.py` – SADC curriculum, hard-negative mining, and false-negative mitigation.
  - `evaluation.py` – hierarchy- and geometry-aware metrics for the text model.
  - `dataloader/` – Polars/Parquet streaming datasets, tokenization cache, and Lightning `DataModule`.
  - `naics_model.py` – PyTorch Lightning module for Stages 1–3; built from mixins:
    - `mixins/distributed.py`, `loss.py`, `curriculum.py`, `logging.py`, `validation.py`, `optimizer.py` implement orthogonal concerns.

- `src/naics_embedder/graph_model/` (Stage 4 HGCN)
  - `hgcn.py` – hyperbolic graph convolution layers, Lightning module, training loop, and save/export helpers.
  - `dataloader/` – graph `DataModule` and streaming dataset over NAICS relations.
  - `curriculum/` – event-driven 4‑phase curriculum (controller, event bus, adaptive loss, sampling, monitoring, preprocessing).

- `src/naics_embedder/metrics/`  
  Shared metric implementations for both text and graph models (hierarchy metrics, downstream graph evaluation, QCEW benchmark, etc.).

- `src/naics_embedder/utils/`  
  Cross-cutting infrastructure:
  - `config.py` – config models and YAML I/O.
  - `training.py` – hardware detection, trainer construction helpers, checkpoint resolution.
  - `hyperbolic.py` – manifold abstractions (`LorentzManifold`, `CurvatureManager`, `ManifoldAdapter`).
  - `backend.py`, `console.py`, `validation.py`, `warnings.py`, `utilities.py`, `distance_matrix.py`, `naics_hierarchy.py` – device/console utilities, pre-flight validation, warning management, hierarchy utilities.

- `src/naics_embedder/tools/`  
  Non-training utilities surfaced via `naics-embedder tools`:
  - `config_tools.py` – config inspection.
  - `metrics_tools.py` – metric visualization and investigation helpers.
  - `embeddings_verification.py` – logic for comparing Stage 3 vs Stage 4 embeddings.

- `data/` (generated), `checkpoints/`, `logs/`, `outputs/`, `reports/`  
  Runtime artifacts (gitignored) for data, training outputs, and analysis.

- `tests/`  
  Pytest suite (unit + some integration) covering geometry, curriculum, data pipeline, CLI, utils, and graph model.

For deeper architectural detail (hyperbolic math, curricula, distributed training, etc.), refer to `CLAUDE.md` and the MkDocs site referenced in `README.md`.

## Environment & installation

- Python: 3.10+  
- Package manager: [`uv`](https://github.com/astral-sh/uv) – all commands below assume uv.

Install dependencies:

```bash path=null start=null
uv sync
```

(If uv is not installed yet, install it once with `pip install uv`.)

## Core CLI workflows

All CLI entrypoints are invoked via `uv run naics-embedder ...` from the repo root.

### Data generation (Stage 0)

Full pipeline (recommended first run):

```bash path=null start=null
uv run naics-embedder data all
```

Individual steps if you need finer control:

```bash path=null start=null
uv run naics-embedder data preprocess   # build naics_descriptions.parquet
uv run naics-embedder data relations    # build naics_relations.parquet
uv run naics-embedder data distances    # build naics_distances.parquet
uv run naics-embedder data triplets     # build naics_training_pairs.parquet
```

### Text model training (Stages 1–3)

Standard training run with the SADC curriculum:

```bash path=null start=null
uv run naics-embedder train
```

Common variations:

```bash path=null start=null
# Use an explicit config file
uv run naics-embedder train --config conf/config.yaml

# Resume from latest checkpoint for the current experiment
uv run naics-embedder train --ckpt-path last

# Override config values inline (dot-notation)
uv run naics-embedder train \
  training.learning_rate=1e-4 \
  training.trainer.max_epochs=20 \
  data_loader.batch_size=32

# Skip pre-flight validation once data/config are known-good
uv run naics-embedder train --skip-validation
```

The `train` command wires together:

- `Config.from_yaml(...)` in `utils.config`  
- hardware detection and overrides via `utils.training.detect_hardware` / `parse_config_overrides`  
- text `NAICSDataModule` and `NAICSContrastiveModel`  
- a Lightning `Trainer` configured from `cfg.training.trainer`.

### Tools

Configuration and diagnostics:

```bash path=null start=null
# Show the effective training/curriculum configuration
uv run naics-embedder tools config

# Visualize training metrics from logs (plots into outputs/visualizations/)
uv run naics-embedder tools visualize --stage 02_text

# Analyze why hierarchy preservation metrics are low
uv run naics-embedder tools investigate
```

Stage 4 (HGCN) verification:

```bash path=null start=null
uv run naics-embedder tools verify-stage4 \
  --pre ./output/hyperbolic_projection/encodings.parquet \
  --post ./output/hgcn/encodings.parquet \
  --distance-matrix ./data/naics_distance_matrix.parquet \
  --relations ./data/naics_relations.parquet
```

This command runs hierarchy-aware metrics pre/post HGCN (cophenetic correlation, NDCG@K, parent retrieval) and enforces degradation thresholds (`--max-cophenetic-drop`, `--max-ndcg-drop`, `--min-local-improvement`, `--parent-top-k`). It is the canonical way to gate Stage 4 changes.

### HGCN refinement (Stage 4)

The HGCN training logic lives in `src/naics_embedder/graph_model/hgcn.py` as a PyTorch Lightning module (`HGCNLightningModule`) plus a `main(config_file: str = 'conf/config.yaml')` entrypoint.  
Configuration is provided by `GraphConfig` (also backed by `conf/config.yaml`).

Typical flow:

1. Train or load a Stage 3 checkpoint via `uv run naics-embedder train`.
2. Use the confirmation prompt at the end of `train` to generate `output/hyperbolic_projection/encodings.parquet` (Stage 3 embeddings) via `generate_embeddings_from_checkpoint(...)` in `cli/commands/training.py`.
3. Run HGCN training using `naics_embedder.graph_model.hgcn.main` (e.g., from a script or notebook) configured with a `GraphConfig` YAML.
4. Validate that Stage 4 did not degrade hierarchy metrics using `tools verify-stage4` (above).

Refer to `docs/hgcn_training.md` and `src/naics_embedder/graph_model/hgcn.py` for the up-to-date graph config fields and metrics.

## Testing

Tests are written with pytest and are expected to be run via uv.

Run the full suite:

```bash path=null start=null
uv run pytest
```

Useful variants:

```bash path=null start=null
# Run with coverage
uv run pytest --cov=naics_embedder --cov-report=term-missing

# Run a specific test file
uv run pytest tests/unit/test_encoder.py

# Run a single test by name
uv run pytest tests/unit/test_encoder.py -k "test_multi_channel_shapes"

# Use markers (see pyproject.toml and tests/README.md)
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m gpu

# Parallel test execution
uv run pytest -n auto
```

The unit tests are heavily used as executable documentation for geometry, curriculum behavior, data loading, graph refinement, and CLI wiring; when changing those areas, start by reading the corresponding `tests/unit/test_*.py` files.

## Linting & formatting

Python style is enforced via Ruff and YAPF (see `[tool.ruff]` and dev dependencies in `pyproject.toml`). Key points:

- Line length: **105** characters.
- Indentation: 4 spaces.
- Quotes: prefer single quotes for strings and docstrings.
- Method chaining: split across lines with dots aligned vertically.
- Markdown linting is configured via `.markdownlint.jsonc` (line length 100, spacing rules, etc.).

Typical formatting workflow:

```bash path=null start=null
# Lint and format Python (source + tests)
uv run ruff check src tests
uv run ruff format src tests
uv run yapf -i -r src tests
```

If CLAUDE.md mentions helper scripts (e.g., a `scripts/format_code.sh` wrapper) but they are not present in the current tree, prefer the explicit uv-based commands above.

## Working with configuration

Configuration is central to this project and is shared between text and graph stages.

- `conf/config.yaml` is the primary configuration file.  
- `src/naics_embedder/utils/config.py` defines strongly-typed Pydantic models (`Config`, `GraphConfig`, and nested sections like `data_loader`, `curriculum`, `false_negatives`, etc.).
- CLI commands load config exclusively through these models (`Config.from_yaml`, `GraphConfig.from_yaml`) and apply overrides via `parse_config_overrides()`.

Guidelines for Warp agents:

- Prefer changing behavior by editing `conf/config.yaml` and/or the Pydantic config models rather than hard-coding new constants inside training or model code.
- Keep the text and graph curricula consistent with their config sections; if you introduce new curriculum behavior, extend the relevant config model and document it in docs + CLAUDE.md.

## Key implementation patterns

- **Mixin-based text model:** `NAICSContrastiveModel` in `text_model/naics_model.py` is composed from functional mixins in `text_model/mixins/`. When changing training behavior (loss, logging, curriculum, optimizer), find the appropriate mixin rather than adding everything to a single class.

- **Hyperbolic abstractions:** Low-level Lorentz math lives in `text_model/hyperbolic.py`; higher-level manifold utilities and compile-time optimization live in `utils/hyperbolic.py` and `utils/compile.py`. When touching hyperbolic operations, be mindful of both layers and the associated tests.

- **Curriculum & sampling:** SADC for the text model spans `text_model/curriculum.py` and the streaming dataset/dataloader. The HGCN curriculum is separate and implemented under `graph_model/curriculum/`. Make sure any curriculum changes remain compatible with logging and monitoring code so downstream analyses in `reports/` keep working.

- **Validation:** Before training, configuration and data are validated via `utils/validation.py` and invoked from `cli/commands/training.py`. If you add new required files or config invariants, extend this validation layer so failures surface early with clear messages, and keep tests in `tests/unit/test_utils_validation.py` up to date.

## Project-specific guidelines for AI assistants

These are distilled from `CLAUDE.md` and are particularly relevant for automated tools like Warp:

- **Use uv everywhere** for Python execution in this repo (`uv run ...`, `uv sync`, `uv run pytest`, etc.).
- **Favor editing existing modules** over creating new ones when extending functionality (e.g., extend `text_model/loss.py` for new losses, `cli/commands/*.py` for new CLI commands, `conf/config.yaml` for new hyperparameters).
- **Align with existing abstractions:**
  - Text model: use the mixin structure and existing evaluation/metric helpers.
  - Graph model: use `GraphConfig`, `HGCNLightningModule`, and the curriculum utilities rather than re-implementing training loops.
- **Respect the logging and warning infrastructure:** use `naics_embedder.utils.console.configure_logging` and centralized warning configuration (`utils/warnings.py`) when adding new CLIs or scripts.
- **Treat tests as a contract:** when modifying core geometry, curriculum, data loading, or CLI behavior, update or extend the relevant tests under `tests/unit/` and ensure `uv run pytest` stays green.
- **Git workflow expectations:** if you are making branches/commits on behalf of an AI assistant, mirror the patterns in `CLAUDE.md` (descriptive branch names and commit messages) and avoid force-pushing over human work.

For deeper, highly detailed guidance (hyperbolic math, curricula, optimization tricks, known pitfalls), read `CLAUDE.md` and the MkDocs docs under `docs/` (especially `quickstart.md`, `usage.md`, `text_training.md`, and `hgcn_training.md`).
