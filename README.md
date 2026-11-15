# NAICS Embedder

<!-- markdownlint-disable MD013 -->

[![GitHub Release](https://img.shields.io/github/v/release/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/releases) [![License](https://img.shields.io/github/license/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/blob/main/LICENSE) [![Documentation](https://github.com/lowmason/naics-embedder/actions/workflows/docs.yml/badge.svg)](https://github.com/lowmason/naics-embedder/actions/workflows/docs.yml) [![Tests](https://github.com/lowmason/naics-embedder/actions/workflows/tests.yml/badge.svg)](https://github.com/lowmason/naics-embedder/actions/workflows/tests.yml) [![Coverage](https://codecov.io/gh/lowmason/naics-embedder/branch/main/graph/badge.svg)](https://codecov.io/gh/lowmason/naics-embedder) [![Issues](https://img.shields.io/github/issues/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/issues) [![Last Commit](https://img.shields.io/github/last-commit/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/commits/main)

<!-- markdownlint-enable MD013 -->

NAICS Embedder is a research project. It plans three different types of embedding for the 2,125
codes of NAICS 2022, the North American Industry Classification System. Each type has its own
use, and each is judged only on evidence for that use. **No embedding of any type is adopted
yet.** The package is not published on PyPI.

## Three Types of Embedding

Design ruling Q4 (2026-11-14) defines the three types
([design spec](specs/naics-artifact-redesign.md)). The NAICS tree is not one of them. It is the
codebook that all three use: the codes, their parents and ancestors, and the tree distance.

| Type | Embedding | Its use | Judged on |
|---|---|---|---|
| 1 | Relatedness | One embedding of how industries relate in practice: peers and competitors, co-ownership and co-location | Whole families of observed relations that it never trained on. It must beat the better of the tree distance and a frozen text-only table |
| 2 | Text→code | A retrieval system that is to take an owner's description of an establishment and return a ranked shortlist of six-digit codes | Descriptions that employers wrote for the OSHA Injury Tracking Application (ITA), split into a validation part and a sealed part |
| 3 | Economic | Low-dimensional coordinates that replace one-hot and other sparse encodings in downstream models | Economic panels, against the tree's ancestor indicators. The first panel is national employment growth |

The design spec calls each type an artifact. Type 1 is to train on the tree's edges and on
families of observed relations from a companion repository, embedding-data. Types 2 and 3 are
built only from the codebook's own text and the tree. A six-digit code's ancestors follow from
the codebook, so type 2 is to return six-digit codes. Today's text model decodes each query among
the codes at that query's own level, plus any code whose cross-reference produced the query.

## Status

This is the status on 2026-11-15. The
[redesign roadmap](specs/naics-artifact-redesign-roadmap.md) holds one chain of stages for each
type. No sealed test split has been opened for any type.

| Type | Built | Not built | Next |
|---|---|---|---|
| 1 Relatedness | Nothing. No module reads an embedding-data table | The judging panel, its margins and the model | Roadmap Stages 6 and 7, both held on gates in embedding-data |
| 2 Text→code | The trained text model, with its training, checkpoint selection and export. The outcome panel, which no longer decides type 2 | The ITA panel, its scoring and its split. embedding-data has not built the ITA pairs yet | Roadmap Stage 2, the ITA scoring and split on fixtures, needs no gate. Stages 4 and 5 are held |
| 3 Economic | The growth panel, its margins (fixed on 2026-10-17), the decision tooling and 18 frozen candidates | The decision. No candidate has been read on the growth panels | Roadmap Stage 3, the decision read, waits for the project owner's approval |

Each held stage waits on some of these gates in embedding-data:

- Its owner changes the rule that bars its tables from training. Stage 7 waits on this gate.
- Its owner changes the rule that bars selection on the ITA pairs. Stage 5 waits on this gate.
- Its owner changes the rule that sets one count gate after its full build. Stages 4–7 wait.
- A successor evaluation-protocol spec exists. Stages 4–7 wait.
- Each family that a stage reads is built and passes its own minimum-count gate. Stages 4–7 wait.

No trained embedding or checkpoint ships with the repository. The trained tables, checkpoints and
decision records stay on the project owner's machine.

## Results So Far

An earlier design judged one embedding on three panels at once. Its recorded results stand as
history. They show why one embedding no longer serves every use. MRR is mean reciprocal rank on
the outcome panel's validation split, which decodes Census index entries held out from training.

- The reference configuration (hyperbolic, dimension 16, ten seeds) has a mean MRR of 0.413
  ([finding](specs/findings/geometry-by-dimension.md)).
- A comparison of three geometries at three dimensions adopted `euclidean-d16`, with a mean MRR
  of 0.592 (same finding). Under the new design, this winner defines no deliverable.
- A lexical stub with no training scores an MRR of 0.6165
  ([finding](specs/findings/outcome-panel-splits.md)). It embeds hashed character trigrams.

## The Text Model

The trained text model places NAICS codes and activity queries in one space. It is type 2's
current model. Its exported tables are also nine of type 3's candidates.

1. One LoRA-adapted backbone, `sentence-transformers/all-MiniLM-L6-v2`, reads each marked field:
   title, description, examples, exclusions and query. A blank field never enters it.
2. Masked-mean fusion combines a code's present fields. Attention and a mixture of experts (MoE)
   are options. One `Linear(384, d)` follows, with d = 8, 16 or 32.
3. A geometry head maps the projection to a point. `model.geometry` selects `hyperbolic`,
   `euclidean` or `spherical`.
4. Three loss terms train together: task-query cross-entropy, code-to-code soft targets from the
   tree distance, and a radial term in the hyperbolic head only.
5. The outcome panel's validation MRR selects the earliest best epoch. `tools export-table` writes
   that checkpoint's 2,125-code table.

```text
Marked fields and queries -> shared LoRA backbone -> masked fusion -> Linear(384, d)
    -> geometry head -> task + code-to-code (+ radial) terms
    -> earliest best outcome-MRR checkpoint -> 2,125-code table
```

`conf/config.yaml` keeps the hyperbolic reference at dimension 16 as its default. It is not the
`euclidean-d16` arm that the comparison adopted. Under the new design, ITA validation reads will
select and decide for type 2, and the outcome panel will only report. The outcome panel keeps
its selection role for the trained economic candidates.

## Other Built Parts

- **Supervision bundle.** `data all` builds one immutable bundle: the codebook, tree distances,
  unary pairs, redirections and training pairs, with hashes. Training refuses a bundle that fails
  validation.
- **Panels.** The outcome panel scores text→code decoding. The regressor panel reads Quarterly
  Census of Employment and Wages (QCEW) data. It predicts next-year employment levels, or
  employment growth for type 3. Each test or outer split stays sealed behind one logged opening.
- **Decision tooling.** `tools sweep`, `tools margins`, `tools growth-margins` and `tools decide`
  compare arms by paired resampling, with non-inferiority and superiority margins.
- **Economic candidates.** `conf/data/economic_candidates.yaml` freezes 18 candidates. Nine are
  deterministic: a text-only table reduced by PCA, a classical scaling of the tree distance, and
  the text-only table smoothed over each code's lineage, each at d = 8, 16 and 32. Nine are the
  trained arms of the earlier comparison. `tools economic-build` builds the deterministic tables,
  and `tools economic-read` reads one candidate.
- **Remote workflow.** Five `remote` commands prepare a Lambda CUDA instance, train there and
  bring each run back to the Mac.
- **HGCN, parked.** `graph_model/` holds a hyperbolic graph convolutional network (HGCN) with its
  own curriculum. It is parked and feeds no type. Type 1 does not revive it, but it may reuse its
  code.

## Quick Start

```bash
git clone https://github.com/lowmason/naics-embedder.git
cd naics-embedder
uv sync
uv run naics-embedder data all
uv run naics-embedder train
```

`data all` prints the path of the bundle's manifest. Copy `conf/local.example.yaml` to the
gitignored `conf/local.yaml`, and put that path there as `config.supervision.manifest_path`.
`train` then trains the default configuration. CUDA uses `bf16-mixed` for the backbone, and CPU
and MPS use `32-true`. The text Trainer uses one device.

```bash
uv run naics-embedder train --ckpt-path last
uv run naics-embedder tools visualize --summary checkpoints/reference/epoch_summary.jsonl
uv run naics-embedder tools export-table --checkpoint checkpoints/reference/epoch=001.ckpt \
  --output data/reference/table.parquet
```

Resume only from `last`, with the same run directory and settings. Export the epoch that the
monitor selected. `epoch=001.ckpt` is an example. See the
[quickstart](docs/quickstart.md), the [text training guide](docs/text_training.md) and the
[CLI guide](docs/usage.md).

## Development

The project pins Python 3.12 for its locked environment. The tests use fixture bundles and tiny
models. They run no real training campaign and read no real panel.

```bash
uv run pytest -n auto -q
uv run ruff check src/ tests/
./scripts/format_code.sh --check --all
uv run mkdocs build --strict
```

The suite collects 4,086 tests in 107 unit files and two integration files. Skip counts depend on
local data and hardware, including MPS. See [tests/README.md](tests/README.md) for the test
contracts, [CLAUDE.md](CLAUDE.md) for the project conventions and the
[documentation site](https://lowmason.github.io/naics-embedder/) for the guides and API pages.
