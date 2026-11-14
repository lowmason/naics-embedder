# 0002. Stage the redesign in one roadmap, with one independent chain per artifact

- **Status:** Accepted (2026-11-14)
- **Date:** 2026-11-14
- **Deciders:** the project owner ("Accept both", 2026-11-14)
- **Blast radius:** naics-embedder `specs/`: the redesign's roadmap and the embedding roadmap's
  pointers. embedding-data supplies gates only.

## Context

`specs/naics-artifact-redesign.md` (Q4, approved as U31) routes to derive-roadmap, and the user
chose "New spec + roadmap" (U26). derive-roadmap names a roadmap after its spec's filename, so this
one will be `specs/naics-artifact-redesign-roadmap.md`. It sequences stages by dependency, and the
next stage it hands off is the first unticked one whose `Consumes` are all met (its §4). A stage
entry has fixed fields, with no track field. This record decides how the redesign's work is
grouped, before that derivation. Evidence is read at origin/main 7fdc170.

These facts drive the choice:

1. **The artifacts are ready at different times** (the spec's Rollout note). Economic candidates
   are ready now. Text→code's stage spec can start now, but its reads wait for embedding-data
   Stage 10's pairs and for the change to Stage 10's anti-goal. Relatedness waits for the
   families it names and for the R8.4 amendment, and it parks if R8.4 is never amended (Req 27).
   Items 2–4 of the Rollout note name no dependency on each other.
2. **Each artifact is decided alone** (Req 25). Each has its own reference, margins, deliverable
   and sealed opening.
3. **One step comes first.** Recording Q4 (Rollout note, item 1) precedes the others. For Q3,
   the derivation recorded the split as a resume of the embedding roadmap, merged with the new
   roadmap (`specs/completed/naics-representation-split-roadmap.md`, Decisions).
4. **One constraint crosses artifacts.** Economic candidate reads pair only where the reference
   predictions match the 2026-10-17 margin read bit for bit, under `uv.lock` 6de4fbdc (ADR 0001,
   Context). A stage in another chain that changes the lock, for example to add a graph library
   for relatedness, can stop that pairing.
5. **The embedding roadmap still holds related stages.** Stage 9 is held, Stages 10–11 are
   parked, and Stage 12 holds one sealed opening per judged artifact. Req 28 places text→code's
   sealed opening in that Stage 12, and it gives Stage 9's ablations to text→code. Stage 12's
   amendment at the eighth resume places the economic opening there too. Req 27 leaves
   relatedness's opening to its own stage spec.

## Decision

We will stage the redesign in one roadmap whose stages form one chain per artifact. A chain runs
from its artifact's stage spec to its adoption decision, and each sealed opening stays where the
spec puts it. A stage's `Consumes` names only the codebook, tooling already on main and earlier
stages of its own artifact's chain. A blocked chain therefore never holds back another chain.

## Consequences

- **Positive:**
  - Economic candidates and text→code's stage spec can start at once, while relatedness waits for
    embedding-data.
  - Each chain can be read, held or parked alone. Parking relatedness parks its own stages only.
  - The facts that cross artifacts have one home: the Q4 record, the lock constraint and the
    owner gates in embedding-data (Req 26).
- **Negative:**
  - The lock constraint is a dependency between chains that `Consumes` cannot show. Before the
    economic decision, each stage spec that changes `uv.lock` must say how the economic reads
    still pair, for example by running them from a checkout at lock 6de4fbdc.
  - The text→code and economic chains both build on `text_model/` (ADR 0001). A change that one
    chain makes there reaches the other chain's next training run.
  - The text→code and economic openings sit in another roadmap, the embedding roadmap's Stage 12.
    If the relatedness stage spec puts its opening there too, that stage cannot tick while
    relatedness is parked.
  - derive-roadmap counts an input as met unless a Handoff or a stamp shows it missing. A gate in
    embedding-data has neither, so each resume checks those gates by hand. Otherwise it can hand
    off a stage that cannot run yet.
  - The roadmap carries three chains that move at different speeds, so each resume re-checks
    every chain.
- **Neutral / follow-on:**
  - The derivation decides whether Q4's record is a stage or, as for Q3, a resume of the
    embedding roadmap.
  - That record decides how each chain points to embedding roadmap Stages 9 and 12.
  - The check that this holds: in the derived roadmap, no stage's `Consumes` names a stage of
    another artifact's chain.

## Alternatives considered

- **Three roadmaps, one per artifact.** Rejected: derive-roadmap writes one roadmap per spec, so
  three roadmaps need three specs. That goes against U26 ("New spec + roadmap") and against the
  spec's header, which forbids splitting it into per-subsystem plans. The facts that cross
  artifacts would also have no single home.
- **Resume the embedding roadmap and add the stages there.** Rejected: that roadmap's gap analysis
  scores `naics-embedding.md`'s requirements, so Reqs 25–29 would need a second gap analysis in
  the same file. For Q3, the split spec got a roadmap of its own.
- **One roadmap with one linear sequence in Rollout order.** Each stage would consume the stage
  before it. Rejected: text→code and relatedness wait for gates in embedding-data, so one unmet
  external gate would hold back an artifact that does not need it.

## Trade-offs & reversibility

This trades one ordered list, which is easy to follow, for chains that move at their own speed. It
is a two-way door while the roadmap is new: a resume can add a `Consumes` between chains, or merge
them. Revisit it with a superseding ADR if:

- one chain comes to need another chain's output, for example if relatedness becomes an economic
  candidate (Req 29, Later);
- the embedding roadmap's Stage 12 holds back text→code's or the economic sealed opening, for
  example while it waits for another artifact;
- embedding-data's owner does not amend R8.4, so the relatedness chain parks.
