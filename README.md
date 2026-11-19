# NAICS Embedder

<!-- markdownlint-disable MD013 -->

[![GitHub Release](https://img.shields.io/github/v/release/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/releases) [![License](https://img.shields.io/github/license/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/blob/main/LICENSE) [![Documentation](https://github.com/lowmason/naics-embedder/actions/workflows/docs.yml/badge.svg)](https://github.com/lowmason/naics-embedder/actions/workflows/docs.yml) [![Tests](https://github.com/lowmason/naics-embedder/actions/workflows/tests.yml/badge.svg)](https://github.com/lowmason/naics-embedder/actions/workflows/tests.yml) [![Coverage](https://codecov.io/gh/lowmason/naics-embedder/branch/main/graph/badge.svg)](https://codecov.io/gh/lowmason/naics-embedder) [![Issues](https://img.shields.io/github/issues/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/issues) [![Last Commit](https://img.shields.io/github/last-commit/lowmason/naics-embedder)](https://github.com/lowmason/naics-embedder/commits/main)

<!-- markdownlint-enable MD013 -->

This repository is a refactor. It began as one embedding of the NAICS codes with two uses: to
rank codes for a text description, and to replace sparse NAICS encodings in economic models
([original design](specs/naics-embedding.md), Req 1). In October 2026 the project owner asked
whether one embedding was trying to do too much. The owner then split the work into separate
embeddings ([split spec](specs/naics-representation-split.md)). One reason that spec records is
that a lexical baseline with no training ranks codes better than the adopted trained
configuration ([Results So Far](#results-so-far)). A short history of the first model will be
added later.

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

All nine trained configurations of the October 2026 comparison have a mean score below that of a
lexical stub with no training. The task is text→code (type 2): given a short phrase, rank all
1,012 six-digit codes. The test phrases are 4,042 entries of the Census Bureau's NAICS 2022
index, for 939 codes, held out from all training text
([finding](specs/findings/outcome-panel-splits.md#1-index-entries), §§1–3). They form the
validation split of the outcome panel, the project's fixed test set for this task. The score is
mean reciprocal rank (MRR): the mean over queries of 1 / (rank of the correct code). An MRR of 1.0
means that the correct code always ranks first.

These scores come from the earlier one-embedding design, and they stand as history. The trained
scores, in date order, and the stub's score:

| Finding dated | Model | Training seeds | Distance | Validation MRR | Finding |
|---|---|---:|---|---:|---|
| 2026-10-03 | First reading: an earlier hyperbolic version, dimension 16, after one training epoch on the Mac | 1 | Lorentz | 0.2301 | [shared encoder §4](specs/findings/shared-encoder-first-reading.md#4-the-outcome-panel) |
| 2026-10-07 | Reference: hyperbolic, dimension 16 | 10 | Lorentz | 0.413 (mean) | [reference](specs/findings/reference-configuration.md), [geometry §2](specs/findings/geometry-by-dimension.md#2-the-arms) |
| 2026-10-11 | Adopted: `euclidean-d16`, Euclidean, dimension 16 | 5 | Euclidean | 0.592 (mean) | [geometry §§2–3](specs/findings/geometry-by-dimension.md#3-the-decision) |
| 2026-09-24 | Lexical stub: hashed character trigrams, no training | none | cosine | 0.6165 | [outcome panel §5](specs/findings/outcome-panel-splits.md#5-the-lexical-stub-on-the-validation-split) |

The October comparison covered three geometries, hyperbolic, Euclidean and spherical, each at
dimension 8, 16 and 32. Its hyperbolic configuration at dimension 16 is the reference, which
trained earlier and was not retrained. The spread of the reference's ten seeds set the
comparison's margins: how far one configuration could fall below another and still count as no
worse. Each of the nine means is below 0.6165. So is each seed score behind them, from the 40 new
runs and the reference's ten seeds. The best seed scored 0.5995
([finding](specs/findings/geometry-by-dimension.md#2-the-arms), §2).

The scores share one test, but they differ in other ways. All of them rank the same 1,012
candidate codes for the same 4,042 queries, under one frozen assignment of index entries to
splits ([finding](specs/findings/outcome-panel-splits.md#3-roles-d4), §3). The log record of
each trained score names that assignment by its hash: see the first reading
([finding](specs/findings/shared-encoder-first-reading.md#5-the-selection-log), §5), the
reference ([finding](specs/findings/reference-configuration.md)) and the October runs
([finding](specs/findings/geometry-by-dimension.md#5-the-selection-log), §5). The differences:

- **Distance.** Each model ranks codes by its own distance: cosine for the stub, Lorentz
  (hyperbolic) distance for the first reading and the reference, and Euclidean distance for
  `euclidean-d16`.
- **Seeds.** The stub's 0.6165 is one score, with no training and no seed. The reference and
  `euclidean-d16` figures are means over 10 and 5 training seeds. The first reading is one run.
- **Selection.** Each seed of the nine configurations kept the training epoch that scored best on
  this same validation split. The stub and the first reading selected nothing.
- **Training length.** All 15 runs of three configurations, `euclidean-d16` among them, used the
  full 40-epoch budget and kept an epoch from 37 to 39. One reference seed did the same. The
  finding says that the budget bound those runs, and that their scores "are not converged
  values". The other five configurations stopped early, after 7 to 11 epochs, and the comparison
  did not examine why ([finding](specs/findings/geometry-by-dimension.md#1-the-runs), §1).

No decision has compared the stub with a trained model. The October decision compared the nine
trained configurations only ([finding](specs/findings/geometry-by-dimension.md#3-the-decision),
§3). The first-reading finding says that no decision uses the stub's numbers
([finding](specs/findings/shared-encoder-first-reading.md#4-the-outcome-panel), §4).

The stub is the command `tools outcome-baseline`
([finding](specs/findings/outcome-panel-splits.md#5-the-lexical-stub-on-the-validation-split),
§5). It turns each text into hashed character trigrams, in 4,096 buckets, and ranks codes by
cosine similarity. It compares the query text with each code's title, description and examples
text. For a code with index entries, the examples text is a set of that code's other index
entries. A frozen assignment split the index entries of each code into 6,118 examples entries,
7,200 training queries, 4,042 validation queries and 3,013 test queries (§3). The 65 codes that
have examples text but no index entries take it from the Census descriptions file (§§1, 4).

An index entry that matched another index entry or any code text never became a validation or
test query. Such entries became examples text or training queries. A match is an occurrence as
whole words, or a character-trigram Jaccard similarity of at least 0.9 (§2). After the
assignment, no validation query matched any code text or training query. The finding calls the
stub's scores "a floor, not memorization" (§5). Its check compares whole texts. So it misses a
query whose words appear in another order inside a longer text, or that spans two sentences or
two examples entries. An audit of unsplit texts found 2 validation queries inside whole
descriptions and 2 inside joined examples text (§2).

The project changed how text→code is judged, in two design rulings. A design ruling is the
project owner's recorded decision on the design.

- **Ruling Q3 (2026-10-12)** split the one embedding into separate embeddings
  ([split spec](specs/naics-representation-split.md#motivation--one-space-three-purposes)). Its
  motivation lists the stub's 0.6165 above `euclidean-d16`'s 0.592 as a sign that each use has a
  stronger specialist. Q3 kept `euclidean-d16`'s adoption as history, with no deliverable. It
  kept the outcome panel as text→code's only judge.
- **Ruling Q4 (2026-11-14)** moved the judging of text→code to descriptions that employers wrote
  for the OSHA Injury Tracking Application (ITA)
  ([design spec](specs/naics-artifact-redesign.md#the-textcode-artifact), Req 28). Its spec
  gives the use as the reason: an owner who describes an establishment in their own words. The
  owner named ITA descriptions as the closest text to that use. The spec does not cite the stub.

The approved Q4 spec sets these rules for text→code:

- The ITA descriptions are split into a validation part and a sealed part. Each normalized
  description falls in one part only. The validation part is to select checkpoints and choose
  among candidates. The sealed part is to be scored once, for the final decision.
- The headline score leaves out each ITA description that, after normalization, equals an
  official NAICS title or index entry. These copied pairs are reported separately. Each distinct
  employer-and-description pair counts once.
- The outcome panel stays. For text→code, it is reported beside the ITA scores and decides
  nothing. It still selects the training epoch of each trained economic (type 3) candidate.

None of these rules has run on real ITA data. embedding-data, the companion repository, has not
built the ITA pairs. Selection on the ITA validation part also waits for a rule change there
([roadmap](specs/naics-artifact-redesign-roadmap.md#stages), Stages 4 and 5).

From here, a text→code model must beat the better of the lexical stub and `euclidean-d16` on the
ITA validation part, by a margin. The project owner ruled this on 2026-11-16
([roadmap](specs/naics-artifact-redesign-roadmap.md), the ruling after Decisions). The better of
the two is the comparator, which the ruling calls text→code's reference. The score is the
headline MRR. The margin is relative: r times the comparator's MRR, fixed from scores of the
comparator alone before any candidate is scored. The stub has no training seeds, so no spread
across seeds can set the margin. Text→code's own spec, roadmap Stage 2, sets r.

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

The suite collects 4,508 tests in 112 unit files and two integration files. Skip counts depend on
local data and hardware, including MPS. See [tests/README.md](tests/README.md) for the test
contracts, [CLAUDE.md](CLAUDE.md) for the project conventions and the
[documentation site](https://lowmason.github.io/naics-embedder/) for the guides and API pages.
