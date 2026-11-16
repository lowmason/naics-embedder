# NAICS Embedder

This repository is a refactor. It began as one embedding of the NAICS codes with two uses: to rank
codes for a text description, and to replace sparse NAICS encodings in economic models ([original
design][original-design], Req 1). In October 2026 the project owner asked whether one embedding was
trying to do too much. The owner then split the work into separate embeddings ([split
spec][split-spec]). One reason that spec records is that a lexical baseline with no training ranks
codes better than the adopted trained configuration ([Results So Far](#results-so-far)). A short
history of the first model will be added later.

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

All nine trained configurations of the October 2026 comparison have a mean score below that of a
lexical stub with no training. The task is text→code (type 2): rank all 1,012 six-digit codes for a
short phrase that describes an activity. The test phrases are 4,042 entries of the Census Bureau's
NAICS 2022 index, for 939 codes, held out from all training text
([finding][outcome-panel-1-index-entries], §§1–3). They form the validation split of the outcome
panel, the project's fixed test set for this task. The score is mean reciprocal rank (MRR): the mean
over queries of 1 / (rank of the correct code). An MRR of 1.0 means that the correct code always
ranks first.

These scores come from the earlier one-embedding design, and they stand as history. The trained
scores, in date order, and the stub's score:

| Finding dated | Model | Training seeds | Distance | Validation MRR | Finding |
|---|---|---:|---|---:|---|
| 2026-10-03 | First reading: an earlier hyperbolic version, dimension 16, after one training epoch on the Mac | 1 | Lorentz | 0.2301 | [shared encoder §4][first-reading-4-the-outcome-panel] |
| 2026-10-07 | Reference: hyperbolic, dimension 16 | 10 | Lorentz | 0.413 (mean) | [reference][reference], [geometry §2][geometry-2-the-arms] |
| 2026-10-11 | Adopted: `euclidean-d16`, Euclidean, dimension 16 | 5 | Euclidean | 0.592 (mean) | [geometry §§2–3][geometry-3-the-decision] |
| 2026-09-24 | Lexical stub: hashed character trigrams, no training | none | cosine | 0.6165 | [outcome panel §5][outcome-panel-5-the-lexical-stub-on-the-validation-split] |

The October comparison trained three geometries, hyperbolic, Euclidean and spherical, each at
dimension 8, 16 and 32. The reference is its hyperbolic configuration at dimension 16. The spread of
the reference's ten seeds set the comparison's margins: how far one configuration could fall below
another and still count as no worse. Each of the nine means is below 0.6165. So is each seed score
behind them, from the 40 new runs and the reference's ten seeds. The best seed scored 0.5995
([finding][geometry-2-the-arms], §2).

The scores share one test, but they differ in other ways. All of them rank the same 1,012 candidate
codes for the same 4,042 queries, under one frozen assignment of index entries to splits
([finding][outcome-panel-3-roles-d4], §3). The log record of each trained score names that
assignment by its hash: see the first reading ([finding][first-reading-5-the-selection-log], §5),
the reference ([finding][reference]) and the October runs ([finding][geometry-5-the-selection-log],
§5). The differences:

- **Distance.** Each model ranks codes by its own distance: cosine for the stub, Lorentz
  (hyperbolic) distance for the first reading and the reference, and Euclidean distance for
  `euclidean-d16`.
- **Seeds.** The stub's 0.6165 is one score, with no training and no seed. The reference and
  `euclidean-d16` figures are means over 10 and 5 training seeds. The first reading is one run.
- **Selection.** Each seed of the nine configurations kept the training epoch that scored best on
  this same validation split. The stub and the first reading selected nothing.

No decision has compared the stub with a trained model. The October decision compared the nine
trained configurations only ([finding][geometry-3-the-decision], §3). The first-reading finding says
that no decision uses the stub's numbers ([finding][first-reading-4-the-outcome-panel], §4).

The stub is the command `tools outcome-baseline`
([finding][outcome-panel-5-the-lexical-stub-on-the-validation-split], §5). It turns each text into
hashed character trigrams, in 4,096 buckets, and ranks codes by cosine similarity. It compares the
query text with each code's title, description and examples text. For a code with index entries, the
examples text is a set of that code's other index entries. A frozen assignment split the index
entries of each code into 6,118 examples entries, 7,200 training queries, 4,042 validation queries
and 3,013 test queries (§3). The 65 codes that have examples text but no index entries take it from
the Census descriptions file (§§1, 4).

An index entry that matched another index entry or any code text never became a validation or test
query. Such entries became examples text or training queries. A match is an occurrence as whole
words, or a character-trigram Jaccard similarity of at least 0.9 (§2). After the assignment, no
validation query matched any code text or training query. The finding calls the stub's scores "a
floor, not memorization" (§5). Its check compares whole texts. So it misses a query whose words
appear in another order inside a longer text, or that spans two sentences or two examples entries.
An audit of unsplit texts found 2 validation queries inside whole descriptions and 2 inside joined
examples text (§2).

The project changed how text→code is judged, in two design rulings. A design ruling is the project
owner's recorded decision on the design.

- **Ruling Q3 (2026-10-12)** split the one embedding into separate embeddings ([split
  spec][split-spec-motivation--one-space-three-purposes]). Its motivation lists the stub's 0.6165
  above `euclidean-d16`'s 0.592 as a sign that each use has a stronger specialist. Q3 kept
  `euclidean-d16`'s adoption as history, with no deliverable. It kept the outcome panel as
  text→code's only judge.
- **Ruling Q4 (2026-11-14)** moved the judging of text→code to descriptions that employers wrote for
  the OSHA Injury Tracking Application (ITA) ([design spec][design-spec-the-textcode-artifact], Req
  28). Its spec gives the use as the reason: an owner who describes an establishment in their own
  words. The owner named ITA descriptions as the closest text to that use. The spec does not cite
  the stub.

The approved Q4 spec sets these rules for text→code:

- The ITA descriptions are split into a validation part and a sealed part. Each normalized
  description falls in one part only. The validation part is to select checkpoints and choose among
  candidates. The sealed part is to be scored once, for the final decision.
- The headline score leaves out each ITA description that, after normalization, equals an official
  NAICS title or index entry. These copied pairs are reported separately. Each distinct
  employer-and-description pair counts once.
- The outcome panel stays. For text→code, it is reported beside the ITA scores and decides nothing.
  It still selects the training epoch of each trained economic (type 3) candidate.

None of these rules has run on real ITA data. embedding-data, the companion repository, has not
built the ITA pairs. Selection on the ITA validation part also waits for a rule change there
([roadmap][roadmap-stages], Stages 4 and 5).

From here, a text→code model must beat the better of the lexical stub and `euclidean-d16` on the ITA
validation part, by a margin. The project owner ruled this on 2026-11-16 ([roadmap][roadmap], the
ruling after Decisions). The better of the two is the comparator, which the ruling calls text→code's
reference. The score is the headline MRR. The margin is relative: r times the comparator's MRR,
fixed from scores of the comparator alone before any candidate is scored. The stub has no training
seeds, so no spread across seeds can set the margin. Text→code's own spec, roadmap Stage 2, sets r.

[design-spec-the-textcode-artifact]: https://github.com/lowmason/naics-embedder/blob/main/specs/naics-artifact-redesign.md#the-textcode-artifact
[first-reading-4-the-outcome-panel]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/shared-encoder-first-reading.md#4-the-outcome-panel
[first-reading-5-the-selection-log]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/shared-encoder-first-reading.md#5-the-selection-log
[geometry-2-the-arms]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/geometry-by-dimension.md#2-the-arms
[geometry-3-the-decision]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/geometry-by-dimension.md#3-the-decision
[geometry-5-the-selection-log]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/geometry-by-dimension.md#5-the-selection-log
[original-design]: https://github.com/lowmason/naics-embedder/blob/main/specs/naics-embedding.md
[outcome-panel-1-index-entries]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/outcome-panel-splits.md#1-index-entries
[outcome-panel-3-roles-d4]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/outcome-panel-splits.md#3-roles-d4
[outcome-panel-5-the-lexical-stub-on-the-validation-split]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/outcome-panel-splits.md#5-the-lexical-stub-on-the-validation-split
[reference]: https://github.com/lowmason/naics-embedder/blob/main/specs/findings/reference-configuration.md
[roadmap]: https://github.com/lowmason/naics-embedder/blob/main/specs/naics-artifact-redesign-roadmap.md
[roadmap-stages]: https://github.com/lowmason/naics-embedder/blob/main/specs/naics-artifact-redesign-roadmap.md#stages
[split-spec]: https://github.com/lowmason/naics-embedder/blob/main/specs/naics-representation-split.md
[split-spec-motivation--one-space-three-purposes]: https://github.com/lowmason/naics-embedder/blob/main/specs/naics-representation-split.md#motivation--one-space-three-purposes

## Guides

- [Quickstart](quickstart.md): prepare the bundle and start one run.
- [System overview](overview.md): the text model's architecture and what each part serves.
- [Text training](text_training.md): the objective, cache, monitor, exact resume and campaign.
- [CLI usage](usage.md): commands, options and artifacts.
- [Remote workflow](remote_workflow.md): training on a Lambda CUDA instance.
- [HGCN training](hgcn_training.md): the parked graph refinement stage.
