# NAICS artifact redesign — Design Spec

> For agentic workers: REQUIRED NEXT SKILL: derive-roadmap — do not plan
> this spec directly and do not split it into per-subsystem plans.

**Status (2026-11-14).** Draft for the user's review. It records user adjudication Q4, which
redefines the three artifacts of `naics-representation-split.md` (Q3). That spec is not edited.
Supersessions lists each clause this spec replaces.

Three artifacts serve the bundle's 2,125 NAICS 2022 codes. Each is judged only on evidence for its
own use:

| Artifact | User's type | What it is | Judged on |
|---|---|---|---|
| Relatedness | 1 | One embedding of how industries relate in practice, trained on the tree's edges and on observed-relation families from embedding-data | Whole families it never trained on, against the better of the tree's distance and the text-only table (Req 27) |
| Text→code | 2 | A retrieval system: an owner's description of an establishment in, a ranked shortlist of six-digit codes out | ITA employer-written descriptions, split into a validation part and a sealed part (Req 28) |
| Economic | 3 | Low-dimensional coordinates that replace one-hot and other sparse encodings in downstream models | Economic panels, against the tree's ancestor indicators (split Reqs 20–24) |

The NAICS tree is no longer an artifact. It is the shared codebook that all three use.

**Design provenance.** A brainstorming session with the user on 2026-11-14. The inputs read were:

- `naics-representation-split.md`, `naics-embedding.md`, the embedding roadmap and the finding
  `growth-margins.md`, at origin/main 7fdc170;
- embedding-data's `specs/embedding-evaluation-data-spec.md` and its roadmap, at that repository's
  main 147802f;
- embedding-data's Stage 10 spec, `specs/ita-establishment-targets.md`, which its owner approved on
  2026-11-13. It was read on its unmerged branch at 49b394c.

- **Locators.**
  - (split Req n) cites `naics-representation-split.md`, and (naics-embedding Req n) the original
    spec.
  - (Un) cites a user ruling, and (Pn) a proposal the user did not correct, in the tables below.
  - (embedding-data Rn) cites embedding-data's spec at 147802f, and (Stage 10) its approved Stage
    10 spec at 49b394c.
  - Checks run while writing are marked (verified locally: …).
- **Numbering.** Requirements continue from Req 25, and user rulings from U17, so each number is
  unique across the three specs.

**User rulings (2026-11-14).** Free-text rulings are quoted as given. For a multiple-choice
question, the ruling is the chosen option's label.

| Id | Question | Ruling |
|---|---|---|
| U17 | (the user's own proposal) | "For representational split, I see three types of embeddings: 1. GraphSAGE or hyperbolic embeddings trained on taxonomic edges and possibly behavioral or economic edges depending on the richness of the embedding data (~/Projects/embedding-data). 2. text→code embeddings to be used as a retrieval system. Here the code is at the 6-digit level, as once that is known, so are the ancestor codes. 3. Embeddings to be used as an economic artifact: low-dimensional coordinates that replace one-hot and other sparse encodings in downstream models, judged on economic panels (created by ~/Projects/embedding-data)." |
| U18 | What would use the graph embedding, and what would a good one let you do? | A list of uses, abridged here to its headings and items: "1. Financial Risk, Underwriting & Fraud Detection": small business loan default prediction; know your business (KYB) and anomaly detection. "2. Corporate Intelligence & Peer Discovery": automated peer and competitor detection; holding company and multi-industry classification. "3. Supply Chain & B2B Market Modeling": supply chain dependency and disruption analysis; look-alike prospecting and lead scoring. "4. Urban Planning & Retail Site Selection": co-visitation and foot-traffic forecasting. |
| U19 | One blended space, or one view per relation? | "go with one blended space; also, for now, disregard graph embedding uses for which ~/Projects/embedding-data have no coverage--but for those uses (firm-level data, input-output tables), provide a blurb I can use with brainstorming in that repo." |
| U20 | (the user's question: train on one source and judge on another, or split one source at random; then this ruling on judging on held-out families) | "Agreed with different families, falling back to one split." |
| U21 | Which uses should the embedding-data brief cover? | "All six" |
| U22 | Does the type 1 playback look right? What should an economic panel from embedding-data measure? | "Type 1: looks good; Type 3: Does the embedding, when compared to one-hot encoding, give a better prediction on economic panels. Does this make sense?" |
| U23 | May type 1 compete for type 3? | "A: text and tree now" |
| U24 | What text will people feed type 2, and what do they do with the code it returns? | "ITA's employer-written descriptions are the closest to what I am imaging. Such as, what is my NAICS code for the establishment (later, subsidiary or firm) that I own?" |
| U25 | Should ITA text train type 2, or only judge it? | "Judge only first" |
| U26 | How should the redesign be recorded? | "New spec + roadmap" |
| U27 | Where should text→code's development decisions (checkpoint selection, choosing among candidates) read? | "Split ITA (Recommended)" |
| U28 | Who writes the scoring of ITA text–code pairs (metrics, shortlist ranks, and any split)? | "Text→code spec (Recommended)" |
| U29 | Must naics-embedder wait for embedding-data's full build and its V9 cell-count gate before scoring on any embedding-data table? | "Per family (Recommended)" |
| U30 | What must the relatedness embedding beat on the families it never trained on? | "Tree and text (Recommended)" |

U27–U30 answered the questions a builder check raised after the draft was written. Reqs 26–28
state what each chosen option means.

**Proposals the user did not correct.** Each was stated in the session as a proposal, and the user
went on without correcting it.

| Id | Proposal |
|---|---|
| P1 | The tree is the shared codebook, not an artifact. |
| P2 | "Panels created by embedding-data" means later economic panels, each through its own panel spec. |
| P3 | Type 2's headline score uses ITA pairs not flagged as copied taxonomy text. Copied pairs are reported separately. |
| P4 | Each distinct ITA pair counts once. |
| P5 | Type 2 returns a shortlist. The score reads the rank of the reported code, and the match at coarser levels is reported as well. |
| P6 | Subsidiaries and firms come later, in their own spec. |
| P7 | The methodology critique round waits until this spec merges. The description is then refreshed. |

The type 1 playback that U22 approved also stated these, as proposals: type 1 works at the industry
level; credit scoring is the economic artifact's use; the output is flat; and the R8.4 amendment
(Req 26) is the owner's decision.

## Motivation

Q3 made the hierarchy an artifact that needed no work: the tree itself, judged on nothing. The
user's uses for a graph embedding (U18) ask for something the tree cannot give. Two industries can
sit far apart in the tree and close together in practice: they share owners, locate together and
hire alike. embedding-data was built to measure exactly that (embedding-data, Problem). So the
relatedness artifact replaces the hierarchy, and the tree becomes the codebook every artifact uses.

The other two artifacts keep their purpose and gain a clearer use. Text→code now serves an owner who
describes an establishment in their own words (U24). The economic artifact's test is the one its
panel already runs (U22).

## Core principle

The split's Core principle stands. Each artifact is judged only on evidence for its own use, and
nothing is built from an outcome it is judged on. Measurement comes before models.

Q4 adds one rule. For each artifact, every table, or every part of a split table, has one role:
training, selection or judging (Req 26).

## Requirements

### The artifacts

**Req 25 — Three artifacts and a shared codebook (U17, U26).** Q4 supersedes split Req 18's
artifact table with the table above.

- **The codebook** is the bundle's codes, parents, ancestors and tree metric (naics-embedding
  Req 7). Nothing judges it (P1). It is:
  - training input to the relatedness artifact;
  - the source of a six-digit code's ancestors for text→code;
  - the source of the economic artifact's bar, the ancestor indicators.
- Split Req 18's rules carry over. Each artifact has its own reference, margins, deliverable and
  sealed opening. No artifact's adoption decision reads another artifact's panel.
- Each artifact's design belongs to its own stage spec. This spec fixes only what the user decided.

**Req 26 — One role per table or part (U19, U20, U27, U29).**

- For each artifact, every table, or every part of a split table, has one role: training,
  selection or judging. Training covers training terms, curriculum and hard-negative mining.
  Selection covers checkpoint and candidate choice. A part that judges an artifact is never read for
  its training or selection. embedding-data R8.4 likewise lets a family used for tuning be split,
  and reports only the held-out part.
- Judge on whole families held out from training, from a different source where possible. Where no
  other family can judge, split one family. The split keeps each repeated unit on one side, such as
  a normalized description, a household or a row of shares. The stage spec names the unit.
- **Employment data.** Split Req 23's time-window rule applies to every artifact that an economic
  panel reads.
- **Size gate (U29).** Each embedding-data family passes its own minimum-count gate (embedding-data
  V9) when it is built, before naics-embedder reads it. No artifact waits for embedding-data's full
  build.
- **Owner gates in embedding-data.** Each is the owner's change in that repository:
  - **R8.4.** It says "Benchmark tables are never training data", and the repository restates the
    rule in several places (verified locally: its spec, `CLAUDE.md`, `README.md` and methodology
    at 147802f, and the Stage 10 spec at 49b394c). No embedding-data table trains an artifact until
    the owner amends R8.4. The amendment gates training, not the writing of a stage spec (my call).
  - **Stage 10's anti-goal** against any table that feeds "embedder training or model selection".
    It must allow selection on the validation part of the ITA pairs (U27, Req 28).
  - **The Rollout's V9 sentence,** which sets the gate "after the first full build and before any
    embedder is scored". It must allow the per-family gate (U29).

### The relatedness artifact

**Req 27 — One blended embedding of industry relatedness (U17–U22, U30).**

- **What.** One embedding of the 2,125 codes, at the industry level (U19, U22). Firm graphs stay
  with the users who hold them.
- **Uses now.** Peer and competitor discovery, multi-industry profiles from co-ownership, and
  co-location for site selection (U18, U22). Credit scoring is a feature use and belongs to the
  economic artifact (U22).
- **Uses deferred (U19, U21).** These six uses need data that embedding-data does not have:
  - KYB and anomaly detection: firm-to-firm transactions or vendor links;
  - peer detection and holding-company profiles: parent-subsidiary ownership across EINs;
  - supply-chain dependency: input-output or B2B flows;
  - look-alike prospecting: links from firms to their customers' industries;
  - co-visitation: visit sequences between kinds of establishment.

  A separate brainstorm in embedding-data looks for sources for all six. Until a source exists, the
  relatedness artifact ignores these uses.
- **Training input.** The tree's edges plus embedding-data families that the stage spec chooses
  (U17), under Req 26 and its owner gates.
- **Judging.** Whole held-out families (Req 26). The reference is the better of the tree's own
  distance and the frozen text-only table (roadmap D9), so the embedding must add something that
  neither gives (U30). "The reference stands, and no relatedness artifact ships" is an allowed
  outcome. The stage spec proposes which families train and which judge, the scores, the margins
  and the sealed opening.
- **When it can start** (my calls):
  - The stage spec can be written before the R8.4 amendment. The amendment gates training only.
  - The stage waits only for the families its spec names to be built and gated, not for all of
    embedding-data's roadmap.
  - The embedding-data brainstorm on the deferred uses does not gate it.
  - If R8.4 is never amended, the stage stays parked. Its purpose is to add observed relations.
    Without them it would repeat the economic candidates' work on text and the tree.
- **Output.** Flat (U22): cosine similarity and averaging over subsidiaries both assume a flat
  space. The stage spec chooses the method. GraphSAGE is the user's named example (U17).
- **Not the parked graph stage.** The relatedness artifact does not revive the embedding roadmap's
  Stages 10–11, in which HGCN refines the text embedding. Those stay parked with naics-embedding
  Reqs 15 and 17. Their question, whether propagating features over the tree helps the economic
  artifact, goes to the economic candidates' stage (split Req 19). The relatedness stage may reuse
  code from `graph_model/`.
- **Not an economic candidate now** (U23; Req 29).
- A check that holds out whole industries, to test that the method places an unseen code, was
  proposed and not taken up. The stage spec may take it up.

### The text→code artifact

**Req 28 — Retrieval for an owner's own description (U17, U24, U25, U27, U28).**

- **Use.** An owner describes an establishment in their own words and asks for its NAICS code
  (U24). Subsidiaries and firms come later, in their own spec (P6). A firm can span industries, so
  it needs its own rule.
- **Output.** A ranked shortlist of six-digit codes (P5). Ancestors follow from the codebook (U17).
  Owner descriptions are often coarser than codes, so one answer is unreliable.
- **Training.** The codebook's own text only: titles, descriptions, examples, exclusions, index
  entries and cross-reference phrases, as today (U25).
  The stage spec decides what happens to training targets above six digits.
- **Judge.** ITA employer-written text–code pairs, built by embedding-data Stage 10 (U24, U25).
  - The pairs are split, grouped by normalized description, into a validation part and a sealed
    part (U27). The validation part makes the development decisions: checkpoint selection, the
    choice among candidates and early measurement of the gap. The sealed part is read once, at
    text→code's opening.
  - Text→code's stage spec writes the scoring: the metrics, the shortlist ranks and the split
    (U28). embedding-data's successor evaluation-protocol spec points to it for these pairs.
  - The headline score uses pairs not flagged as copied titles or index entries. Copied pairs are
    reported separately (P3).
  - Each distinct employer-and-description pair counts once (P4).
  - The score reads the rank of the reported code, and the match at coarser levels (P5).
- **What the ITA text is like** (Stage 10; its counts are read-only recon, not accepted values):
  - OSHA says the ITA description is "often one of the NAICS index titles". Stage 10 flags each
    description whose normalized text equals an official title or index entry.
  - It repeats heavily: 3.2 million records hold 29,067 distinct normalized descriptions.
  - Descriptions are often coarser than codes.
  - Codes are self-reported. Only native 2022 codes, in 2023–2025, are headline evidence.
  - Coverage follows OSHA's filing rules, so some industries are thin or missing.
- **The outcome panel stays** (my call). It still selects trained economic candidates' checkpoints
  (split Req 23). For text→code it is reported beside the ITA scores and no longer decides.
  At text→code's sealed opening (embedding roadmap Stage 12), the read that decides is the ITA
  sealed part. Text→code's stage spec decides whether the outcome test split is also opened, for
  the report.
- **Owner gate.** Selection on the ITA validation part waits for the owner's change to Stage 10's
  anti-goal (Req 26).
- **Later.** If the measured gap between Census phrases and owner text is large, a later spec may
  add a training part to the ITA split, grouped the same way (Req 26), after the R8.4 amendment.
- Stage 9's backbone and fusion ablations stay with this artifact (split Req 19).

### The economic artifact

**Req 29 — Unchanged test; candidates read text and the tree (U22, U23).**

- **The test.** The user's test is "Does the embedding, when compared to one-hot encoding, give a
  better prediction on economic panels" (U22). Split Reqs 20–24 implement it:
  - the candidate's coordinates against the ancestor indicators, both with the covariates, under
    the margin δ;
  - six-digit one-hot is still reported;
  - two regimes, seen and held-out.

  The margins fixed on 2026-10-17 stand (finding: growth-margins).
- **Candidates read text and the tree only** (U23; split Req 23, U7). The relatedness artifact is
  not a candidate now. Its embedding-data inputs come from ITA 2016–2025, ACS 2023–2024 and CPS
  from 2025 on (embedding-data R1.1). The growth panel scores feature years 2022–2024, with
  outcomes through 2025 (verified locally: `panels/regressor.py`). Split Req 23's employment-data
  rule applies.
- **Later.** The relatedness artifact may become a candidate under split Req 23's rule, once a
  reference year outside its training window is final. That year needs its own panel spec and its
  own margin record. The margins of 2026-10-17 are never read again.
- **Later panels** built from embedding-data tables each join through their own panel spec (P2;
  split Req 20). Under Req 26, a table that judges an economic candidate never trains it.
- The candidates' stage carries the four items in the split spec's Status, including split Req 19's
  question about propagation over the tree.

## Supersessions

| Superseded | Replaced by | Applies to |
|---|---|---|
| Split Req 18's hierarchy row: the tree as an artifact, "Exact, with nothing trained", judged on "Nothing" | The tree as the codebook (Req 25), and the relatedness artifact (Req 27) | All |
| Split Req 18's text→code row: judged on "The outcome panel only" | ITA text–code pairs, split into validation and sealed parts (Req 28). The outcome panel is reported and keeps selecting economic checkpoints | Text→code |
| Embedding roadmap Stage 12: text→code's decision on the outcome panel's sealed test split (`OutcomePanel.open_test`) | A decision on the ITA sealed part (Req 28) | Text→code |
| Split Planned sources: the path `~/Projects/estab-data`, and "This spec reads none of them" | `~/Projects/embedding-data`, read by the relatedness artifact (Req 27) and text→code (Req 28) | Relatedness, text→code |
| Split Out of scope: "The planned sources above" | The same | Relatedness, text→code |

Unchanged:

- split Reqs 19–24, as they apply, and the split's Core principle;
- the growth panel, its margins of 2026-10-17 and the economic decision rule;
- Stage 12 as one sealed opening per artifact;
- embedding roadmap Stage 9 held, and Stages 10–11 parked.

### Deferred items this touches

`specs/deferred_items.md` was read, not edited.

- **With the economic candidates' stage:** `14-economic-decision-tooling`. A trained candidate that
  reads only text and the tree is still allowed (U23), so its trained-candidate item stays live.
  Its predictions-to-table binding item says "Revisit if: the candidates' spec builds candidates
  from tables it does not control".
- **With the parked Stages 10–11:** the graph-stage items the split spec lists, and the tech-debt
  audit's HGCN legacy-input fallback. The relatedness stage reads them if it reuses `graph_model/`.
- **With text→code:** the Stage 9 items the split spec moved there.

## Verification — observable outcomes

- [ ] **Q4 recorded.** CLAUDE.md's overview names the three artifacts of Req 25 and the tree as
      their codebook. The embedding roadmap points to this spec. `naics-representation-split.md` is
      unchanged.
- [ ] **Roadmap derived.** Its stages follow the Rollout note's dependencies, and each is a
      spec→plan cycle.
- [ ] **One role per table or part.** Each artifact's stage spec lists every table or part it
      reads, with its one role (Req 26).
- [ ] **Gates held.**
  - No embedding-data table trains an artifact before the R8.4 amendment.
  - No embedding-data family is read before it passes its own count gate.
  - No ITA pair is read for training. Only the validation part is read for selection, and only
    after Stage 10's anti-goal allows it.
  - The ITA sealed part is read once, at text→code's opening.
- [ ] **Economic candidates.** Every candidate's provenance names only text and tree inputs. The
      growth margins are not read again.

## Out of scope

- Each artifact's design: its stage spec.
- Sources for the deferred uses: the embedding-data brainstorm (U19, U21).
- The changes in embedding-data that Req 26's owner gates name: R8.4, Stage 10's anti-goal and
  the V9 sentence. Its evaluation protocol for tables other than the ITA pairs.
- Text→code for subsidiaries and firms (P6).
- Training text→code on ITA text, until the gap is measured (U25).
- The relatedness artifact as an economic candidate, until Req 29's later year.

## Rollout note

Dependencies run in this order. The staging itself belongs to derive-roadmap.

1. **Record Q4.** Update CLAUDE.md's overview, the embedding roadmap's pointer, and Stage 12's
   text→code opening (Supersessions). Docs only.
2. **Economic candidates.** Ready now: the growth panel, its margins and the decision tooling exist.
3. **Text→code.** Its stage spec, which writes the ITA scoring and split (U28), can start now. Its
   reads wait for Stage 10's pairs to be built and gated in embedding-data, and selection on the
   validation part waits for the change to Stage 10's anti-goal.
4. **Relatedness.** Its stage spec can start once the families it names are built and gated.
   Training waits for the R8.4 amendment. In embedding-data at 147802f, Stage 6's ITA area mix is
   built, Stage 10 (co-ownership) is approved but not built, the ACS targets (Stage 9) are on hold,
   and CPS job-change flows are planned.

The methodology description on its unmerged branch, `specs/naics-artifacts-methodology.md`,
describes Q3. Its critique round waits until this spec merges, and the description is then
refreshed to Q4 (P7).
