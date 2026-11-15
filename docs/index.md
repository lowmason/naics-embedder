# NAICS Embedder

NAICS Embedder is a research project. It plans three different types of embedding for the 2,125
codes of NAICS 2022, the North American Industry Classification System. Each type has its own
use, and each is judged only on evidence for that use. **No embedding of any type is adopted
yet.**

## Three Types of Embedding

Design ruling Q4 (2026-11-14) defines the three types. The repository records it in
`specs/naics-artifact-redesign.md`. The NAICS tree is not one of the types. It is the codebook
that all three use: the codes, their parents and ancestors, and the tree distance.

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

This is the status on 2026-11-15. The roadmap, `specs/naics-artifact-redesign-roadmap.md`, holds
one chain of stages for each type. No sealed test split has been opened for any type.

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

The hyperbolic graph convolutional network (HGCN) in `graph_model/` is parked. It feeds no type.
See [HGCN training](hgcn_training.md).

## Results So Far

An earlier design judged one embedding on three panels at once. Its recorded results stand as
history. MRR is mean reciprocal rank on the outcome panel's validation split, which decodes
Census index entries held out from training.

| Arm | Seeds | Validation MRR |
|---|---:|---:|
| Reference: hyperbolic, dimension 16 | 10 | 0.413 (mean) |
| `euclidean-d16`, adopted over eight other geometry × dimension cells | 5 | 0.592 (mean) |
| Lexical stub: hashed character trigrams, no training | none | 0.6165 (one read) |

Under the new design, the adopted `euclidean-d16` defines no deliverable. The default
configuration is still the hyperbolic reference. The findings are
`specs/findings/geometry-by-dimension.md` and `specs/findings/outcome-panel-splits.md`.

## Guides

- [Quickstart](quickstart.md): prepare the bundle and start one run.
- [System overview](overview.md): the text model's architecture and what each part serves.
- [Text training](text_training.md): the objective, cache, monitor, exact resume and campaign.
- [CLI usage](usage.md): commands, options and artifacts.
- [Remote workflow](remote_workflow.md): training on a Lambda CUDA instance.
- [HGCN training](hgcn_training.md): the parked graph refinement stage.
