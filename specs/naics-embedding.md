# NAICS embedding — Design Spec

> For agentic workers: REQUIRED NEXT SKILL: derive-roadmap — do not plan
> this spec directly and do not split it into per-subsystem plans.

The NAICS embedding method is re-specified around how its output is used. Each of the 2,125
NAICS 2022 codes gets a low-dimensional point that serves two uses: as regressors that replace
one-hot and other sparse NAICS encodings in downstream models, and as a representation into which
text decodes to the right code. One shared text encoder is trained with a query→code term and one
hierarchy regularizer on a proper tree metric. Its geometry (Euclidean, spherical or hyperbolic),
dimension (8, 16 or 32), backbone and fusion are chosen by paired, multi-seed comparisons on two
sealed panels, one per use, that no training term reads. The graph stage stays only if it wins a
decision experiment on the same panels. Every mechanism that cannot act is removed, and three
errors in the reviewed description are recorded as errata.

**Design provenance.** Five input files in `specs/`: the reviewed description
`naics-embedding-methodology.md` (committed e96eb3f on 2026-09-23, components relabeled S1–S4 in
1a31f7b; not edited by this spec); the prompt all three reviewers received,
`naics-embedding-prompt.md`; and three reviews, `naics-embedding-review-chatgpt.md` (C1–C16),
`naics-embedding-review-claude.md` (C1–C30) and `naics-embedding-review-gemini.md` (C1–C15; only
its text, lines 1–160, was read, the rest being embedded image data). The description's S2 is the
project's "Stage 3" (the text stage) and its S3 is the project's "Stage 4" (the graph stage); this
spec uses the description's labels.

- **Staleness.** Since e96eb3f, only implementation fixes touched the method's code: 3497231,
  5c2c994, d2bd22a, PR #98 (merged on origin/main as 8a3fb86), and PR #99 (0339775, merged
  during synthesis; bitwise-identical results). 33145c2 points the text
  stage's configuration at the supervision bundle the description was measured on. The graph
  stage's configuration still names no bundle. The description is therefore current.
- **Adjudication.** All three reviews are first passes. ChatGPT says so, and Claude records no
  push-back. Gemini's ledger narrates designer push-back that never happened, so its
  Accepted/Rejected labels carry no weight. None opens with the prompt's required routing header.
  - ChatGPT's citations are unresolved tokens, so its state-of-the-art claims count only where
    Claude's linked citations or a local check corroborate them.
  - Several of Gemini's in-text citations resolve to unrelated references, and its C9 means one
    thing in its body and another in its ledger.
  - The adjudication is this synthesis's triage (2026-09-23) plus the user's two rulings. Q1: the
    embedding exists to replace one-hot and other sparse encodings (16 coordinates, for example,
    instead of 1,000+ indicators), and it must serve both as a regressor and as an outcome so it
    is not specialized. Q2: run the graph-stage decision experiment before deleting or repairing
    the stage.
- **Locators for the description.** (methodology X), where X is Component inventory,
  Composition, Cross-component n, Notation, S1–S4, or Top-level Qn. A component locator may be
  narrowed to Notation, formulation, procedure, limitation n, evaluation, or Qn (its numbered
  open question).
- **Locators for the reviews.** (ChatGPT Cn), (Claude Cn) and (Gemini Cn) cite numbered points.
  (ChatGPT on S2-Q3), (Claude on top-level Q6) and the like cite a review's answer to one of the
  description's open questions. (Gemini remediation n) cites a row of Gemini's remediation table,
  and (Gemini ledger Cn) cites its ledger where that differs from its body.
- **Other locators.** The user's rulings are (user adjudication Q1, 2026-09-23) and (user
  adjudication Q2, 2026-09-23). Checks run during synthesis are (verified locally: arithmetic),
  (verified locally: index file) and (verified locally: cross-reference file). The last two ran
  against the Census 2022 NAICS source files the data pipeline downloads.

## Errata to the reviewed description

The description stays as committed. These errors are recorded here instead.

- **E1: an incorrect statement.** (methodology S2 procedure) says the decoupled contrastive term
  is "unbounded below". Under the norm cap every pairwise distance lies in [0, 4]. With K = 24
  negatives and τ = 0.07, the term is therefore bounded below by ln 24 − 4/0.07 ≈ −53.96. The
  bound is reached when the positive coincides with the anchor and every negative sits at the
  anchor's antipode on the cap. The term is unbounded only without the cap (ChatGPT C4; Claude C2;
  verified locally: arithmetic). (Gemini C12) repeats the incorrect claim as a defect, and its
  ledger's "rejection" of it records push-back that never took place (rejected).
- **E2: D is not a metric.** $D$ is presented as a structural distance (methodology Notation;
  methodology S1 procedure) and as a "target metric" (methodology S1 Q1). The text stage's
  estimand asks for geodesic distances proportional to $D$ (methodology S2 formulation). $D$
  violates the triangle inequality for every within-sector pair other than parent–child: 269,998
  of the 272,103 within-sector pairs.
  - Collateral pairs: routed through the lowest common ancestor, the two legs sum to
    $h_i + h_j - 1$, less than $D_{ij} = h_i + h_j$.
  - Lineal pairs s ≥ 2 steps apart: routed through any intermediate ancestor, the legs sum to
    $s - 1$, less than $D_{ij} = s - \tfrac12$.
  - (ChatGPT C2) gives the grandparent chain 0.5 + 0.5 < 1.5 as its example (verified locally:
    arithmetic).
  - No metric realizes $D$ even up to scale, so the proportionality estimand is unattainable. The
    cross-sector constant adds no violation.
- **E3: an omitted loss conflict.** (methodology S2 limitation 2) lists the conflicts among the
  loss terms but omits one.
  - The level term targets $\sinh r = 0$, and hence the origin $o$, for every level-2 code
    (methodology S2 procedure). The graph stage's level term has the same target (methodology S3
    procedure).
  - Meanwhile $D$ places every pair of the 20 sectors 99 apart, so no configuration zeroes both
    terms.
  - While the cap holds every point at $r = 2$, the level term has no gradient (methodology S2
    limitation 1). The conflict is hidden, not absent (ChatGPT C3; Claude C1; verified locally:
    arithmetic).

## Motivation — the real gaps

The reviewed method cannot show that it does what it was built for:

- **Nothing measures the designed purpose.** Both stages optimize agreement with the taxonomy they
  train on, and the evaluation measures only that agreement. No statistic uses information
  outside the taxonomy, and the two downstream benchmarks were never run (methodology
  Cross-component 1; methodology S4 limitation 1; methodology S4 limitation 6; ChatGPT C15;
  Claude C27; Gemini C8).
- **Selection is in-sample.** Validation reuses the training anchors and pairs, and shares 89.1%
  of its triples with training. A single seed and run supply every number (methodology S2
  limitation 4; methodology S2 limitation 8; ChatGPT C6; Claude C19, C20; Gemini C13).
- **The target is not a metric, and one constant dominates it.** $D$ breaks the triangle inequality
  (E2). The cross-sector constant 99 covers 87.9% of pairs and turns distance matching and the
  global statistics into a same-sector test (methodology S1 limitation 2; methodology S4
  limitation 2; ChatGPT C5; Claude C7, C21; Gemini C4, C11).
- **The geometry does no hyperbolic work.** The hard norm cap saturated in the one retained run,
  leaving only angular information. The level target contradicts the target for sectors (E3)
  (methodology S2 limitation 1; ChatGPT C4; Claude C2, C3; Gemini C1).
- **Much of the machinery cannot act, and the rest conflicts.** Two loss terms, the curriculum,
  false-negative clustering and the graph stage's curvature are inert or nominal. Negative
  eligibility contradicts the target (methodology S2 limitation 2; methodology S2 limitation 3;
  methodology S3 limitation 3; ChatGPT C7, C10; Claude C12, C14; Gemini C6).
- **Text construction injects nuisance signal.** Placeholders leak level, repeated exclusion text
  is truncated, and inherited descriptions duplicate 522 parent–child pairs (methodology S1
  limitation 6; ChatGPT C9; Claude C10, C15; Gemini C5).
- **The graph stage cannot run and has not shown value.** It cannot compose with the text stage as
  configured. Its free node states can overwrite the text geometry, and it brings no information
  the text stage lacks (methodology Composition; methodology S3 limitation 1; ChatGPT C1, C12;
  Claude C23, C24; Gemini C3).

## Core principle

Errors are fixed outright (Reqs 7–11). Every choice among working alternatives earns its place on
held-out evidence for both uses: geometry, dimension, backbone, fusion and the graph stage are
settled by paired, multi-seed comparisons on two sealed panels that no training term reads
(Reqs 1–5), not by argument or by the literature's priors. A mechanism that cannot act in the
configuration that ships is deleted, not repaired: "inert machinery should default to deletion,
not rehabilitation" (ChatGPT on top-level Q5).

## Requirements

### Estimand, panels and decisions (S4)

**Req 1 — Two co-primary estimands.** The method's quality criterion is its value in its two
uses, measured on sealed held-out data (user adjudication Q1, 2026-09-23):

- **As a regressor:** the out-of-sample gain from using the embedding's coordinates in place of
  sparse NAICS encodings in a downstream model (Req 2).
- **As an outcome representation:** how reliably a point produced from text decodes to the right
  code (Req 3).

Agreement with the taxonomy becomes an intrinsic diagnostic and never a selection criterion
(Req 6; ChatGPT C15; Claude C27; Gemini C8; methodology Cross-component 1).

- Both panels as co-primary (chosen).
- Retrieval alone as primary (ChatGPT C15; Claude on top-level Q7) and an economic KPI alone
  (Gemini ledger C8) (rejected: each specializes the embedding to one use).
- Structural reconstruction as the quality criterion (rejected: the target is optimized directly,
  so the evaluation is circular (methodology S4 limitation 1)).
- Gemini's compromise of keeping reconstruction as a gate (Gemini ledger C8) survives only as the
  diagnostics of Req 6.

**Req 2 — Regressor panel.** The embedding's coordinates enter a downstream model as regressors:
tangent coordinates at $o$ for a hyperbolic arm (methodology S4 procedure), raw coordinates
otherwise. The panel redesigns the defined economic benchmark (methodology S4 procedure;
methodology S4 limitation 6).

- **Rows sit below the code.** A row is a code in a given year, or in a given area (ChatGPT C15;
  Claude C28). The description's one row per code (methodology S4 procedure) cannot hold a code
  out of some rows but not others, so it cannot support the seen-code regime below.
- **Outcomes** come from public employment statistics for NAICS 2022 codes. Which series, years
  and grains are usable is (open — resolved by verification, not argument): which reference
  years are published on NAICS 2022 codes at six digits, at which grains (by year, by area), and
  how much disclosure suppression removes at each. The branches are specified in advance:
  - If the verified window supports a time-respecting outcome (the outcome dated after the
    features, with splits by time), the panel includes one (ChatGPT C15; Claude C28).
  - If it does not, the panel is cross-sectional and records why.
  - If no grain below the code survives suppression, the panel is held-out-codes only and
    records that the one-hot comparison could not run.
  - Training on changes coded on an earlier NAICS vintage (Claude C28) is not used without a
    concordance (rejected here; cross-vintage work is Out of scope).
- **Comparators** share the downstream model and a tuned penalty (ChatGPT C15; Claude C28):
  - six-digit one-hot indicators;
  - ancestor indicators at levels 2–5;
  - a text-only representation reduced to the same dimension;
  - covariates only;
  - covariates plus each representation.
- **Fitting.** Features are standardized and the penalty is tuned by nested cross-validation
  (Claude C28). A fixed penalty of 1 on unscaled features (methodology S4 procedure) (rejected).
- **Two regimes, reported separately.**
  - Seen codes: rows are split by year or area, so every code appears in training. This is
    where one-hot is a real competitor.
  - Held-out codes: every row of a held-out code group leaves training, with four-digit parents
    as groups. Here one-hot can predict only the intercept (Claude C28; methodology S4
    limitation 6).
  - The sealed test (Req 4) is an outer set: held-out four-digit groups for the held-out regime,
    and held-out years or areas for the seen regime. Repeated grouped folds run inside the
    remainder, for penalty tuning and selection.
- The multi-level variant (levels 2–6) is kept (methodology S4 procedure).
- Running the benchmark as defined (Gemini remediation 10) (rejected: its comparators cannot
  isolate the embedding's contribution (methodology S4 limitation 6)).

**Req 3 — Outcome panel.** The outcome use is scored as text→code decoding with the method's own
encoder. A held-out index entry is encoded as a query and decoded to the nearest six-digit code
under the arm's own distance (Claude C19; ChatGPT C15; ChatGPT on top-level Q7).

- **The index file** holds 20,373 entries over 1,010 six-digit codes (verified locally: index
  file). 112130 and 541120 have no entries: they stay decoding candidates but are never queries.
- **Every index entry has exactly one role:** examples-channel text, or a training, validation or
  test query, never two (chosen). This rule extends Claude C19, which removes test entries from
  the channel but leaves training queries unaddressed. Without it, a training query would sit
  inside its own code's examples channel and reward string matching. Test entries are stratified
  by code.
- **Candidates** are the 1,012 six-digit codes. **Metrics** are exact top-1 accuracy, MRR, Hit@k
  for k ∈ {1, 5, 10}, and hierarchical partial credit. Partial credit is the level of the lowest
  common ancestor of the top-1 code and the truth (Claude C19; ChatGPT on top-level Q7).
- **Leakage.** No test query may appear, exactly or as a near-duplicate, in any training text:
  titles, descriptions, remaining examples-channel entries, exclusion text, or training queries
  (including the cross-reference activity phrases of Req 8).
- **Scope limit.** This panel measures decodability for text queries. The rest of the
  NAICS-as-outcome use, decoding a point predicted from non-text predictors, needs labeled
  records that are not publicly available (Claude on S4-Q1) and is Out of scope.

**Req 4 — Holdout, sealing and selection.** Each panel has a validation split and a sealed test
split (ChatGPT C6; Claude C19; Gemini C13).

- Every selection reads validation splits only: checkpoint, early stopping, learning-rate
  control, loss weights, backbone, dimension, geometry and graph-stage arm.
- The test splits are opened once, for the final configuration and the comparisons recorded for
  it (Claude on S2-Q6).
- The in-sample validation contrastive loss selects nothing (methodology S2 limitation 4;
  methodology S2 evaluation).
- The graph stage gets no private validation tail and no last-epoch export (methodology S3
  procedure; Claude C25).

**Req 5 — Decision rule.** Every comparison of a configuration A against a configuration B
follows one rule.

- **Seeds.** Each arm runs at least 5 seeds (Claude C20).
- **Pairing.** Differences Δ = A − B are paired: both arms are scored on the same resample of the
  evaluation unit, with seeds nested within the resample (Claude on S4-Q4; ChatGPT on S4-Q4). The
  unit is codes, with their queries, for the outcome panel, and four-digit-parent groups for the
  regressor panel.
- **Reference configuration.** Every comparison starts from one named configuration: the text
  stage after Reqs 7–11, with the current backbone behind a shared encoder (Req 14), in
  hyperbolic geometry with Req 13 at dimension 16. That is the current method's geometry at the
  user's reference dimension (user adjudication Q1).
- **Margin.** Each panel's non-inferiority margin δ is fixed before any arm runs, as a stated
  multiple of the reference configuration's across-seed standard deviation on that panel
  (Claude C22).
- **Adoption.** A is adopted when it is non-inferior on both panels and superior on at least one
  (Claude C22; user adjudication Q1).
  - Non-inferior: the lower bound of the 95% interval on Δ exceeds −δ.
  - Superior: the 97.5% interval excludes zero. Two panels give two chances to adopt, so each
    gets half the error rate.
- **Several arms.** Some decisions have more than two arms: the nine geometry × dimension cells,
  the backbones, and the graph-stage arms A–D. The survivors are the arms no other arm dominates,
  and the tie order below picks among them. If dominance cycles and every arm is dominated, the
  tie order picks among all the arms of the decision.
  - The extra pairwise comparisons within a decision are not corrected for multiplicity. That is
    a stated choice, not an oversight: the tie order toward the simpler arm is the guard against
    a spurious win.
- **Ties.** When A is not adopted over B, the simpler configuration stands. Simpler means fewer
  components (stages or post-processing steps), then lower dimension, then non-hyperbolic
  geometry. Any tie left after that goes to the higher regressor-panel estimate, because that is
  the designed purpose (user adjudication Q1).
  - Lower dimension ahead of geometry (chosen): the purpose is low dimension, and a hyperbolic
    embedding that matches a larger flat one is the geometric result worth keeping (ChatGPT on
    top-level Q2).
  - Geometry ahead of dimension (rejected).
- **Scope of the rule.** It replaces the fixed acceptance thresholds (methodology Composition;
  methodology S4 procedure; ChatGPT C14; Gemini C14) and gates the graph stage (Req 15).
- A paired t-test across batches (Gemini remediation 12) (rejected): batches are not experimental
  units. On the fixed code set the metric has no sampling error, and the variance comes from
  seeds and query sampling (ChatGPT on S4-Q4).

**Req 6 — Structural statistics become stratified diagnostics.** They are reported, but never
selected on and never used as a headline (ChatGPT C5; Claude C21; Gemini C11):

- sector separation, as the AUC of distance between same-sector and cross-sector pairs;
- within-sector rank correlation, averaged over sectors and over queries;
- mean average precision over ancestors;
- NDCG with integer grades from lowest-common-ancestor depth.

The Pearson statistic drops the "cophenetic" name, since no dendrogram is involved (Claude C21;
methodology S4 procedure). The 522 unary pairs (Req 9) are excluded from parent retrieval
(Claude C22). IC-graded relevance (Gemini C11) follows only if Req 7's ablation adopts IC.

### Supervision (S1)

**Req 7 — A tree metric through a virtual root.** The structural target $D^{\ast}$ is tree path
length with a virtual root above the 20 sectors (Claude C7; ChatGPT on S1-Q2):

- $D^{\ast}_{ij} = h_i + h_j$ within a sector;
- $D^{\ast}_{ij} = \lambda(i) + \lambda(j) - 2$ across sectors.

There is no half-step for lineal pairs (ChatGPT C2; Claude C6; E2) and no cross-sector constant
(ChatGPT C5; Claude C7; Gemini C4). $D^{\ast}$ is a tree metric and so satisfies the triangle
inequality.

- **IC ablation.** Depth-aware similarity from intrinsic information content (Lin or
  Jiang–Conrath) is an ablation of the target, adopted only under Req 5 (Claude C6; ChatGPT C2)
  (chosen).
- Adopting IC outright (Claude C6; Gemini C4) (rejected): $D^{\ast}$ already removes the errors,
  and the evidence for IC comes from lexical taxonomies (ChatGPT C2).
- Employment-based information content (Claude on S1-Q1) (rejected): it puts downstream economic
  outcomes into the training target and contaminates the regressor panel.
- A data-driven cross-sector value (Gemini remediation 4) (rejected): it has the same
  contamination problem (Claude on S1-Q2).

**Req 8 — Exclusions are directed redirections.** A cross-reference reroutes an activity rather
than asserting that two codes are unrelated. 4,558 of the 4,601 cross-reference rows read
"…are classified in Industry j"; for example, 111120 has "Growing soybeans--are classified in
Industry 111110" (verified locally: cross-reference file). The method uses them in three ways:

- **(a) As de-duplicated exclusion-channel text,** each cross-reference appearing once (Claude
  C10; ChatGPT C9).
- **(b) As training queries for the query→code term (Req 11).** The activity phrase q has its
  destination j as the target, and the referencing code i is always among its negatives, so that
  s(q, j) > s(q, i) (ChatGPT C8; Claude C8).
- **(c) Never as code–code repulsion,** never as a negative in the code–code term, and never as
  a graph edge by virtue of exclusion status (ChatGPT C8; Claude C8, C25; methodology
  Cross-component 3).

Lineal references stay text only and never act as negatives: eight codes name an ancestor and one
names its child (methodology S1 procedure; Claude C9; ChatGPT on S1-Q3). The reserved exclusion
slot is removed (methodology S2 procedure).

- Keeping exclusions as repulsive negatives and adding signed repulsive graph edges (Gemini C7;
  Gemini on top-level Q4) (rejected): boundaries lie between near neighbors, and 474 of the 2,394
  sibling pairs are exclusions (methodology S1 limitation 4), so repulsion fights the hierarchy.

**Req 9 — Text construction.**

- **Masking.** Absent channels are masked out of fusion; there is no placeholder text (ChatGPT C9;
  Claude C15; Gemini C5). The examples channel is empty for every level-2–4 code, so a
  placeholder encodes level (methodology S1 limitation 6). A channel-presence indicator is an
  ablation, adopted only under Req 5 (Claude C15).
- **Inheritance.** Every inherited description carries its provenance, and the 14 arbitrary
  inheritance choices become a deterministic, documented rule (ChatGPT C9; Claude on S1-Q5).
- **Unary pairs.** The 522 unary pairs are five-digit industries whose only child is their
  six-digit code. They leave positive supervision and parent-retrieval scoring, and all 2,125
  codes stay in the deliverable (Claude C15, C22) (chosen).
  - Collapsing each chain into one code (Claude C15) (rejected): excluding the pairs removes both
    defects without changing the code universe that every stage shares (methodology
    Cross-component 2).
- **Input windows.** Inputs fit the backbone's trained input window, which is
  (open — resolved by verification, not argument). For the current checkpoint, the description
  states 256 tokens
  (methodology S2 procedure), while Claude C16 quotes the model card's 128-token fine-tuning
  length. Windows beyond it, such as today's 512 (rejected) (Claude C16).

**Req 10 — Anchors and candidates.**

- **Anchors.** Every code is an anchor at every level, including all 1,012 six-digit codes.
  Nothing depends on code numbering or stored orientation (ChatGPT C7; Claude C11).
- **Candidates.** The code–code term scores each anchor against all 2,125 codes through a
  per-epoch cache of code points, with graded targets from $D^{\ast}$ (Claude C13; ChatGPT on
  S2-Q3).
- **Removed.** Negative-eligibility rules, fixed pre-drawn tuples and inverse-distance draws are
  removed, and with them the conflict that made 4,675 grandchildren negatives for sibling
  positives (ChatGPT C7; Claude C12; Gemini C6; Gemini C9).
- **False negatives.** They are known exactly from the taxonomy, so no clustering estimates them
  (ChatGPT on top-level Q5; Claude on top-level Q5).

### Text stage (S2)

**Req 11 — One task term, one hierarchy regularizer.** The text stage's objective has three
terms:

- **(i) Query→code retrieval, the task term.** Training-split index entries and cross-reference
  activity phrases (Req 8b) are scored against the codes at the target's level. A
  cross-reference query also always scores its referencing code (Claude on top-level Q7;
  ChatGPT on top-level Q1).
- **(ii) A code–code listwise term, the hierarchy regularizer.** For each anchor, it is the
  cross-entropy between the softmax of −d over all 2,125 codes and a target distribution from
  $D^{\ast}$ (Claude C13, C14; ChatGPT C10).
- **(iii) A live radial term, in the hyperbolic arm only (Req 13).** It is geometry-specific: it
  fixes where depth lives, not what is related. The objective therefore still has one task term
  and one hierarchy regularizer (Claude C14; ChatGPT on S2-Q2).

Temperatures are learned logit scales (Claude C2), and term weights are set on validation splits
(Req 4). No term reads downstream outcomes, so the panels stay independent of training (Claude
C27).

The following are removed (methodology S2 procedure; ChatGPT C10; Claude C14; Gemini ledger C9;
ChatGPT on top-level Q5; Claude on top-level Q5):
- the radius penalty, which is inert;
- the load-balancing term, which goes with the mixture of experts (Req 14);
- distance matching, with its batch-dependent normalization;
- the pairwise preference term, which the listwise term subsumes;
- the three-phase curriculum and its mining rules;
- false-negative clustering;
- the logged-only margin.

Rejected alternatives:
- The six-term objective (rejected).
- Repairing the radius penalty once the cap is gone (Gemini on top-level Q5) (rejected): the
  radial term replaces it.
- Entailment cones as the primary loss (Gemini remediation 8) (rejected): neither use needs
  entailment queries (Out of scope).

**Req 12 — Geometry and dimension are experimental factors.** There are three geometry arms,
Euclidean, spherical (cosine) and hyperbolic, at dimensions 8, 16 and 32. Dimension 16 is the
reference (user adjudication Q1; ChatGPT C4; Claude C3; Gemini on top-level Q2).

- Geometry and dimension are crossed, because their interaction is the question. Any hyperbolic
  advantage is expected at low dimension (Claude C3, citing Sala et al. 2018 and Nickel & Kiela
  2018 with links).
- Arms share the encoder and the objective. The radial term exists only in the hyperbolic arm
  (Req 11).
- Each arm decodes by its own distance and exports the coordinates Req 2 names. One affine map
  projects to the arm's dimension, since the two stacked maps are equivalent to one (methodology
  S2 procedure).
- The winner is chosen by Req 5.
- Fixing 384 dimensions (methodology S2 procedure) (rejected): the downstream use wants low
  dimension, and on a fixed radius the realized model is spherical anyway (methodology S2
  limitation 1).

**Req 13 — A live radius in the hyperbolic arm.**

- **No hard norm cap.** Any bound on the radius must pass gradient to it (ChatGPT C4; Claude C2;
  Gemini C1). On the cap no gradient reaches the norm, and the retained run saturated at $r = 2$
  (methodology S2 limitation 1).
- **A virtual root at $o$,** with the 20 sectors at positive radius, so that no level target asks
  distinct sectors to coincide (ChatGPT C3; Claude C1; E3).
- **One radial coordinate, $r$,** used in every loss, target and diagnostic (Claude C5;
  methodology Cross-component 5).
- **Curvature fixed at 1,** not a parameter anywhere; the nominal per-layer curvature is removed
  (ChatGPT on top-level Q5; Claude C5). Distance formulas are written for c = 1 only
  (methodology Cross-component 4).
  - A learnable global curvature (Gemini C15) (rejected): against a learned logit scale,
    curvature only rescales (Claude C5), and today it receives no gradient (methodology S3
    limitation 3).
- **Radii stay where the working precision resolves distances.** This is checked numerically,
  not set from a derived bound (Claude C4; methodology S3 limitation 2).

**Req 14 — One shared encoder; the backbone is chosen, not inherited.**

- **Encoder.** One shared encoder reads every channel behind a field marker. Masked fusion
  (attention pooling or a masked mean) combines the channels that are present, and queries pass
  through the same encoder as a single field (ChatGPT C11; Claude C17, C18; ChatGPT on S2-Q4).
  The per-channel adapter copies are removed (Claude C17).
- **Mixture of experts.** The sparse mixture of experts survives only as an ablation arm under
  Req 5 (Claude on S2-Q4) (chosen).
  - Keeping it with masked gating (Gemini C5; Gemini remediation 7) (rejected): sparse routing
    targets conditional compute at scale (Claude C18), and half the codes would feed it two
    placeholder channels (methodology S1 procedure).
- **Backbone.** It is chosen under Req 5 from the current checkpoint and at least two current
  general-purpose embedding models, with a frozen-encoder control (ChatGPT C11; Claude C17;
  Claude on S2-Q5). No model-ranking claim is relied on: the candidates the reviews name are
  unverified (ChatGPT on S2-Q5; Claude C17).

### Graph stage (S3)

**Req 15 — The graph stage earns its place in a decision experiment.** Per user adjudication Q2
(2026-09-23), the experiment runs before any deletion or repair (ChatGPT C12; ChatGPT on
top-level Q6; Claude C23; Claude on top-level Q6).

Starting from the text stage selected under Reqs 11–14, five arms run at the selected geometry
and dimension, each for at least 5 seeds, scored on both panels:

- **(A)** the text stage alone;
- **(B)** plus extra text-stage training at matched compute;
- **(C)** plus parameter-free smoothing: each code moves toward the mean of its parent and
  children in the arm's coordinates, by a factor α tuned on validation (Claude on S3-Q1);
- **(D)** plus the graph stage as designed, repaired only to run: it works in the text stage's
  space and dimension (Req 16) and is selected like every other arm (Req 4);
- **(E)** a text-shuffle control, with node inputs permuted across codes before the graph stage,
  run only if D wins.

**Decision.** The graph stage is kept only if D is adopted over both B and C under Req 5. If E
matches D, the stage is learning the tree rather than refining text, and that is recorded
(ChatGPT on top-level Q6). Arms B and C can be adopted like any other change, and whichever arm
Req 5 selects supplies the deliverable (Req 16).

**Geometry of arm D.** The graph arm runs in the selected geometry. Its layer body is kept, and its
exponential and logarithmic maps become the identity for Euclidean and normalization for
spherical (chosen).
- Forcing a hyperbolic text stage for the experiment (rejected): it would test the graph stage
  against a baseline that already lost under Req 12.

Repairing the stage first into an inductive network that back-propagates into the text encoder
(Gemini C3; Gemini remediation 6) (rejected): the stage adds no observational information
(ChatGPT C12; Claude C23), so its value is an empirical question, and repairs follow only a win
(Req 17).

**Req 16 — Composition and deliverable.**

- **One space.** Any graph-stage output lives in the text stage's space and dimension, so its code
  points can be scored against queries the text stage encodes. No reduction or projection map is
  introduced (Claude C24; ChatGPT C1; methodology Composition).
  - A learned projection between the stages (Gemini C10; Gemini remediation 3) (rejected): it
    creates a third, unaligned space.
- **The deliverable is always defined.** It is a table of all 2,125 codes and their coordinates,
  taken from the arm Req 15 selects. That arm is the text stage, possibly with longer training or
  smoothing, unless the graph stage is kept (Claude C22, C29). This closes
  the gap where the method did not define a deliverable when A = 0 (methodology Composition).

**Req 17 — Repairs that apply only if the graph stage is kept.** If Req 15 keeps the graph stage,
it also gets:

- parent–child edges plus self-loops, with sibling and multi-generation edges added only by
  ablation (ChatGPT on S3-Q4; Claude on S3-Q4);
- an edge weight that enters only once (Claude C25; methodology S3 procedure);
- normalization that preserves radius, or Lorentz-native layers, in place of tangent-space
  LayerNorm (ChatGPT C13; Claude C24; Claude on S3-Q3);
- explicit retention of the text geometry, by distance distillation to the text-stage points or
  by a residual correction (ChatGPT on S3-Q2; Claude C23; Claude on S3-Q2);
- an objective without the hinge temperature, the adaptive margin, or the uncertainty weights
  (Claude C26; ChatGPT on S3-Q5);
- no curriculum filters and no four-phase controller (ChatGPT on top-level Q5; Claude on
  top-level Q5).

Riemannian batch normalization as the presumed replacement (Gemini C2; Gemini remediation 2)
(rejected): its only support is a Gemini citation this synthesis could not verify (Gemini on
top-level Q5). If Req 15 drops the graph stage, this requirement lapses, and the graph stage and
its curriculum system leave the method.

## Verification — observable outcomes

- [ ] **Employment-statistics coverage (discharges Req 2's (open)).** Record:
  - the reference years published on NAICS 2022 codes at six digits;
  - the grains (by year, by area) at which six-digit series are published;
  - for each year, grain and series, the share of six-digit codes suppressed;
  - the resulting regressor-panel population and row grain, whether the panel includes a
    time-respecting outcome, and whether the seen-code regime can run, with reasons for any
    "no".
- [ ] **Backbone input window (discharges Req 9's (open)).** Record the chosen backbone's trained
      input window from its own documentation. After de-duplication, report the share of each
      channel's texts that exceed it; the channel policy leaves no input beyond it.
- [ ] **Leakage.** Exact and near-duplicate matching of every test query against all training
      text (Req 3's list) finds no exact match. Near-duplicates above a stated similarity are
      removed from the test split and counted.
- [ ] **Index-entry roles.** Every index entry holds exactly one role. 112130 and 541120 are
      candidates and never queries.
- [ ] **Selection hygiene.** A run log shows that every selection read validation splits only,
      and that the test splits were opened once, for the final configuration.
- [ ] **Decision records.** Every adopted change has a record with: its arms, at least 5 seeds
      each, the δ per panel fixed before the runs, the 95% non-inferiority and 97.5%
      superiority intervals, and, for decisions with more than two arms, the non-dominated set
      (Req 5).
- [ ] **Target.** $D^{\ast}$ satisfies the triangle inequality over all triples (checked through
      lowest common ancestors) and contains no 99. Cross-sector values equal
      $\lambda(i) + \lambda(j) - 2$.
- [ ] **Exclusions.** No exclusion pair acts as a code–code negative. No graph edge exists
      because of exclusion status. Lineal references never act as negatives, and each
      cross-reference appears once in the exclusion text.
- [ ] **Text.** No absent channel contributes to fusion. Unary pairs are absent from positive
      supervision and from parent retrieval. All 2,125 codes are in the deliverable.
- [ ] **Radius (hyperbolic arm).** On a trained run:
  - the gradient with respect to radius is nonzero;
  - radii vary within each level;
  - the 20 sectors sit at distinct, positive radii;
  - manifold validity and distance resolution hold at the working precision for the largest
    radius observed.
- [ ] **No inert terms.** On a real batch, every objective term has a nonzero gradient.
- [ ] **Coverage.** Every code is an anchor, and the code–code term covers all 2,125 codes at
      every step, with no pre-drawn tuples.
- [ ] **Panels.** The regressor panel reports the seen-code and held-out-code regimes separately,
      with one-hot only in the seen regime. The outcome panel reports top-1, MRR, Hit@k and
      hierarchical partial credit.
- [ ] **Diagnostics.** Structural statistics appear only in the diagnostic report, stratified as
      Req 6 lists.
- [ ] **Geometry × dimension.** All nine cells run with at least 5 seeds, and the chosen cell has
      its Req 5 record.
- [ ] **Decision experiment.** Arms A–D run with at least 5 seeds under the shared selection
      protocol. The keep-or-drop decision is recorded under Req 5, and E runs if D wins.
- [ ] **Deliverable.** A 2,125-code table exists for the final configuration, whatever the
      graph-stage decision.

## Out of scope

- **External relatedness validation.** Industry similarity from company filings, input–output
  linkages, labor flows, coagglomeration and product-classification overlap (Claude C27; ChatGPT
  C15; ChatGPT on S1-Q2). Entry criterion: a use that needs relatedness beyond the taxonomy, with
  each source's availability verified first.
- **Cross-vintage placement and concordance-based tests** (ChatGPT C16; methodology
  Cross-component 9).
- **The outcome use with non-text predictors,** and independently labeled establishment
  descriptions (Claude on S4-Q1).
- **Entailment, cone or box embeddings of ancestor direction** (Claude C30; ChatGPT on S2-Q1;
  Gemini remediation 8). Revisit only if the hyperbolic arm wins and a use needs entailment
  queries.
- **Nested representations that yield several dimensions from one model** (Claude C17). The sweep
  trains each dimension separately.
- **Exclusions as evidence of relatedness,** for example the 1,204 cross-sector exclusion pairs
  (methodology S1 limitation 2). No review proposes this as a training signal; route it to
  brainstorming if wanted.
- **Editing the reviewed description.** Its errata live in this spec.

## Rollout note

Resolve the employment-statistics (open) item first. It fixes the regressor panel's population
and whether the panel has a time-respecting outcome, and Req 5 needs both panels before anything
is adopted. The backbone-window (open) item resolves alongside the backbone choice (Req 14).

Dependencies run in this order. The staging itself belongs to derive-roadmap.

1. Panels, splits and the decision rule (Reqs 1–6), before any model change.
2. The supervision and text fixes (Reqs 7–10) and the objective (Req 11), assembled into the
   reference configuration that Req 5 names. Its seeds fix δ for each panel.
3. Geometry × dimension (Req 12), crossed.
4. The backbone and fusion ablations (Req 14), each varied one factor at a time from the
   reference, which keeps compute bounded.
5. The graph-stage decision (Req 15).
6. Req 17, only if Req 15 keeps the stage.

This spec is not a stage of an existing roadmap, so it carries no roadmap stamp line.
