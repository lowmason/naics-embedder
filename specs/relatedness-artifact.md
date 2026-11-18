# Relatedness artifact — Design Spec

**Status:** DRAFT (2026-11-18), for the user's review. Written in a brainstorming session. A
builder check raised five points. One was the user's to rule (U47, the relevance floor). The other
four are folded in as my calls: the gain's base (§4.4), five seeds (§4.6, §4.7), when TRI may join
(§4.1, §4.6) and the centered text reference (§4.4).

**Roadmap:** `specs/naics-artifact-redesign-roadmap.md`, Stage 6 (ROUTING: brainstorming). This spec
also fixes what Stages 6b and 7 and the relatedness opening read (§8). Source spec:
`specs/naics-artifact-redesign.md` (Q4) Reqs 25–27, U17–U22, U29 and U30. Evidence was read at
origin/main 8329e55, and in embedding-data (`~/Projects/embedding-data`) at 114f33a, which is past
the roadmap's gate read at fd9e495. embedding-data's main moved to e6e12b2 during the session, where
the owner approved the TRI parent-links spec (Stage 5). Paths are under `src/naics_embedder/` unless
they start with `conf/`, `tests/`, `specs/` or `docs/`, or name a top-level file.

**Next skill after approval:** `writing-plans` (plan 20) in a fresh Opus session, with this file as
the handoff. Execution then runs on Sonnet in its own fresh session.

**Design provenance.** A brainstorming session with the user on 2026-11-18. The inputs read were:

- Q4's spec, ADRs 0001 and 0002, and the roadmap's Stage 6, 6b and 7 entries and Gates paragraph.
- The completed Stage 2 spec `specs/completed/text-to-code-ita-panel.md`, `panels/selection_log.py`,
  `panels/text_only.py`, `decision/resampling.py`, `decision/text_to_code.py`,
  `supervision/code_targets.py`, and the text-only table's provenance file in the main checkout's
  `checkpoints/plan9_exit/`.
- In embedding-data at 114f33a: `specs/evaluation-protocol.md`,
  `specs/naics-evaluation-data-roadmap.md`, `specs/deliverable-map.md`, the source spec's R7.2,
  R7.4 and R8.4, methodology K5 and K6, `specs/completed/ita-establishment-targets.md` (its
  co-ownership target and its amendment of 2026-11-18),
  `specs/completed/target-contract-ita-area-mix.md`, `specs/uncovered-uses-evidence-sources.md`,
  the V9 decision `specs/findings/v9-ita-naics-minimum-counts.md`, and the report keys in
  `src/embedding_data/ita_co_ownership.py`.

No embedding-data table was read. Every count below comes from a spec or a finding.

- **Locators.** (Req n), (Un) and (Pn) cite Q4's spec or a ruling below. (Stage 10) cites
  embedding-data's `specs/completed/ita-establishment-targets.md`, (protocol) its
  `specs/evaluation-protocol.md`, (V9) its V9 decision and (K5) or (K6) its methodology, all at
  114f33a. "Stage N" alone is the roadmap's.
- **My calls.** Choices I made while writing are marked (my call). Choices that I played back in the
  session, and that the user went on without correcting, are marked (my call, played back).

**User rulings (2026-11-18).** Rulings continue from U41, so each number is unique across the specs.
For a multiple-choice question, the ruling is the chosen option's label.

| Id | Question | Ruling |
|---|---|---|
| U42 | Only ITA co-ownership and the 2023 ITA area mix are built, and the family that judges can never train. Which use must the embedding prove on held-out evidence: ownership (co-ownership judges, the area mix and the tree's edges train), co-location (the reverse), or a labor family that is not built yet? | "Ownership judges (Recommended)" |
| U43 | Where do the development reads go (the better reference, the margin, checkpoints and model variants): co-ownership split by anchor industry into a validation part that selects and a sealed part read once, co-ownership whole with tuning on held-out area-mix counties, or a split for the margin only with tuning on area-mix counties? | "Split co-ownership (Recommended)" |
| U44 | Which headline score decides: NDCG@10 on each anchor's shortlist, AUC over each anchor's whole ranking, or both as two decision panels? | "Shortlist: NDCG@10 (Recommended)" |
| U45 | Where does relatedness's one sealed opening live: in its own chain, after Stage 7, or in embedding roadmap Stage 12 with the other two artifacts? | "Own chain (Recommended)" |
| U46 | How does Stage 7 build the embedding: GraphSAGE, a blended factorization, or both as candidates? | "A. GraphSAGE (Recommended)" |
| U47 | Which co-owned pairs count as related: every published pair (J ≥ 1, as the V9 decision set the minimum), or only pairs with two or more shared employers (J ≥ 2, the draft)? The draft's J ≥ 2 had cited V9's unfiltered design-time curve. The decision's testable curve shows 94.84 % of single-employer cells recurring from 2023 to 2024 | "J ≥ 1, as V9 (Recommended)" |

## 1. Intent

Two industries can sit far apart in the NAICS tree and still go together in practice. One owner
runs both, or they locate in the same places. The relatedness artifact is one flat embedding of the
2,125 codes that captures this (Req 27). Its uses now are peer and competitor discovery,
multi-industry profiles from co-ownership, and co-location for site selection (U18, U22).

This spec decides how the artifact is judged and how it is built. The judge is ownership: ITA
co-ownership, held out from training (U42). The embedding trains on the tree's edges and the 2023
ITA area mix, through GraphSAGE (U46). It must beat the better of the tree's distance and the frozen
text-only table by a fixed margin on a shortlist score, NDCG@10 (U30, U44). If it does not, the
reference stands and no relatedness artifact ships (Req 27).

Stage 6 builds the judging panel and the margin rule on fixtures. It reads no real embedding-data
table, so no gate holds it.

**Success.** On fixtures with embedding-data's co-ownership schema, the panel splits, scores, logs
and seals as this spec says. The margin rule is code that Stage 6b runs without a new decision. The
family that trains is named, so the owner can make the R8.4 decision (G1).

## 2. Goals, non-goals and anti-goals

**Goals.**

1. One role for every family and part that the relatedness stages read (Req 26).
2. A deterministic split of co-ownership into a validation part and a sealed part, by anchor
   industry (U43).
3. A tie-fair NDCG@10 score of any arm on a part, with its reports (U44).
4. Logged reads, and one logged opening of the sealed part. A second opening needs a reason.
5. The reference choice, the margin and the beat-by-δ test, as code (U30, U40's pattern).
6. The method that Stage 7 builds, so that Stage 7 routes to writing-plans (U46).
7. The place of the sealed opening (U45).

**Non-goals.**

- A command that reads real tables. Stage 6b adds it, from this API (my call, as Stage 2 did).
- The GraphSAGE model, its training input and its export. Stage 7 builds them (§4.7).
- Roles for families that are not built: ACS, CPS, BEA and USAspending (§4.1).
- The deferred uses of Req 27, and firm-level graphs.
- Any change in embedding-data. §7 states what this spec asks of its owner.

**Anti-goals.** The design fails, even if it runs, when it:

- lets a co-ownership row, or a quantity derived from one, enter training (Req 26).
- lets the area mix, or any other training family, judge relatedness.
- lets one anchor industry's row fall in both parts, or lets a rebuilt table count as a new sealed
  part, so that a second opening passes without a reason.
- scores against raw shares, which rewards placing large industries near everything.
- breaks ties against the truth, which cripples the tree reference and makes the bar easy.
- changes `uv.lock`, `OutcomePanel`, `ItaPanel`, `compare_panel`, `decision/text_to_code.py`, the
  economic decision, or any logged record shape that `decide` checks.

## 3. Approaches

**Family roles (U42).**

| Option | Trade-off | Ruling |
|---|---|---|
| Ownership judges: co-ownership judges, the area mix and the tree's edges train | Proves peer discovery and multi-industry profiles on a direct industry-pair relation, with marginals for a null model. TRI parent links can join later as a judge from another source. Co-location is not judged, and the training signal is noisy: a median of 1 record per area-mix cell (K5) | Chosen |
| Co-location judges: the area mix judges, co-ownership trains | Proves site selection, but needs a derived industry-by-industry co-location score from noisy cells. The ACS area mix, a possible second judge, is on hold after three failed builds | Rejected |
| A labor family judges (CPS job-change flows or ACS staffing), and both ITA families train | Judges from another source, as Req 26 prefers. ACS has three failed builds and CPS has not started, so Stage 6b could wait months | Rejected |

**Where the development reads go (U43).**

| Option | Trade-off | Ruling |
|---|---|---|
| Split co-ownership by anchor industry: the validation part selects, the sealed part judges once | The text→code pattern (U27). Tuning aims at the claimed use, and a candidate that fails on validation never spends the sealed read. Costs about half the judging data | Chosen |
| Keep co-ownership whole, and tune on held-out area-mix counties | The purest transfer claim and full judging power. The model is tuned for co-location, not ownership, and nothing checks it before the one opening | Rejected |
| Split co-ownership for the reference, the margin and a go/no-go check, and tune on area-mix counties | Keeps tuning off ownership. Two selection reads to build and log | Rejected |

**Score (U44).** NDCG@10 was chosen: peer discovery shows a short list, and text→code's headline is
a shortlist score too. AUC is stable and uses every pair, but it ignores whether the top of the list
is right. It stays as a report. Both scores as two decision panels was the strictest option. It was
rejected, and with TRI it would make four panels.

**Opening (U45).** Its own chain was chosen. Relatedness can then park, if R8.4 is never amended,
without holding back embedding roadmap Stage 12 (ADR 0002, Negative). One place for all three
openings was rejected for that reason.

**Method (U46).**

| Option | Trade-off | Ruling |
|---|---|---|
| A. GraphSAGE, through PyG's `SAGEConv` | The user's named example (U17). Inductive: the text-only vectors as node features place a code that has no area-mix data. It needs a training loop, seeds and hyperparameters, so more selection reads | Chosen |
| B. A blended factorization: above-chance association matrices from the tree and county co-presence, reduced by truncated SVD | Deterministic, one replicate and few knobs. Linear, and not the named method | Rejected |
| C. Both, as candidates | Shows whether the GNN beats a simple baseline. About twice the model work | Rejected |

Two methods were dropped before the choice:

- Fine-tuning the shared text encoder with a co-location term. Every change in `text_model/` reaches
  the next training run of the text→code and economic chains (ADR 0002).
- node2vec. PyG's random walks need `torch-cluster` or `pyg-lib`, and neither is installed. Adding
  one would change `uv.lock`.

## 4. Design

### 4.1 Families and roles

- **Judging: ITA co-ownership** (Stage 10's `ita_co_ownership.parquet` and
  `co_ownership_marginals.parquet`). The validation part selects. The sealed part judges and is read
  once (U42, U43).
- **Training: the tree's parent–child edges** (the codebook) **and the 2023 ITA area mix**
  (`ita_area_mix.parquet`, run `stage6_ita_area_mix_20261014_1000_csv`). The area mix trains only
  after the owner's R8.4 decision allows it (G1). It never judges. No part of it selects, because
  selection reads co-ownership's validation part only (my call, from U43).
- **A second judge, later: TRI parent links** (embedding-data Stage 5, whose spec the owner approved
  at e6e12b2). TRI may join as a judging family only through an amendment of this spec, made
  before the sealed opening (§4.8). TRI must then be built, count-gated and have a protocol section.
  Its own margin record must come before any candidate read on TRI (§4.6). A late join is safe,
  because a candidate must beat the reference on every judging family, so a new family can only
  make the test harder (my call, from the builder check: Stage 6b need not wait for TRI). Its split
  uses this spec's rule and salt, so each industry sits in the same part in both families (my
  call). TRI never trains relatedness, because it measures the relation that co-ownership judges
  (embedding-data's `uncovered-uses-evidence-sources.md`, Rules for every new family).
- **Training, later: the ACS workplace area mix.** It may join training through an amendment (my
  call, played back). It measures the relation that the ITA area mix measures, so it never judges.
- **Unassigned:** CPS job-change flows, ACS staffing patterns and worker profiles, BEA input–output
  tables and USAspending subawards. An amendment assigns each one's role before any read (my call,
  played back).
- **Off-limits: the ITA text–code pairs.** They belong to text→code (protocol C3).
- **Not read:** `co_ownership_support.parquet` and the cross-year link table. The panel needs J,
  m(a) and the codes only (my call).

### 4.2 Inputs and code mapping

The panel reads two co-ownership tables and the bundle's codebook. It validates the columns that it
uses and refuses a table that lacks one.

- **Target rows.** `ita_co_ownership.parquet`, in the `targets-v1` form (R7.4). The panel reads
  `family`, `measure`, `unit_a_type`, `unit_a`, `unit_b_type`, `unit_b` and `n_unweighted`.
  `n_unweighted` is J(a,b): the number of employers (EINs) with establishments in both industries
  in the same year (Stage 10). The panel refuses a row whose family is not `co_ownership`, whose
  measure is not `employer_share`, or whose unit types are not `naics_national_industry`.
- **Marginals.** `co_ownership_marginals.parquet`. The panel reads `node_id` and
  `n_multi_industry_employers`, which is m(a): the employers in industry a that also operate in
  another industry in the same year (Stage 10).
- **Node ids.** A node id has the form `naics:2022:national_industry:<code>`, as in the text pairs.
  The panel takes `<code>` and refuses, before any read:
  - a node id of another system, vintage or level.
  - a code that is not one of the bundle's six-digit codes.
  - a target row whose industry has no marginal row.
  - a pair emitted in one direction only, or with two different J values. Stage 10 emits both
    directions with one J (my call: the check costs nothing and catches a broken build).
  - a pair whose J exceeds m of either industry. Every employer counted in J(a,b) is counted in
    m(a) and m(b) (my call, the same reason).
- **Pins.** The read command (Stage 6b) takes each table's expected SHA-256 from the protocol
  section that embedding-data's Stage 16 writes (§7). The panel refuses a mismatch before it scores
  (protocol C1's form, my call).
- **Zeros.** Stage 10 emits only pairs with J ≥ 1. Two industries with no row have J = 0.

### 4.3 The split

- **Unit.** The anchor industry (U43). Each six-digit industry's whole row of shares falls in one
  part. Req 26 names "a row of shares" as such a unit.
- **Rule.** A part comes from the SHA-256 of a fixed salt and the six-digit code. The first 8 bytes
  of the hash, as an unsigned integer modulo 2, give the part: 0 is validation, 1 is sealed. This is
  the ITA rule of the Stage 2 spec, §4.2 (my call, played back).
- **Config.** `conf/data/relatedness_panel.yaml` holds the panel name (`relatedness_co_ownership`),
  the rule version, the salt, r and k. The salt is chosen once, when the file
  is first committed, and never changes (my call).
- **Stability.** The rule depends on the code alone. A rebuilt table keeps each industry in its
  part.
- **Split fingerprint.** The SHA-256 of the canonical JSON of the rule: the salt, the version, the
  hash algorithm, the modulus and the part names. It identifies the split, not the rows (§4.5).
- **Mirror pairs** (my call, played back). J(a,b) appears in a's row and in b's row. When a and b
  fall in different parts, each part sees the pair once, from its own anchor. This is accepted:
  co-ownership never trains, the scored rows never overlap, and a strict split by pair would halve
  the data again. The panel reports how many relevant pairs cross the parts (my call).

### 4.4 Scoring

An **arm** gives a similarity between any two six-digit codes. There are three kinds:

- **Tree:** −D*, the tree metric (naics-embedding Req 7, `supervision/code_targets.py`). Between
  six-digit codes, D* takes 5 values: 2, 4, 6, 8 and 10.
- **Text-only:** the frozen D9 table's 384-dimension vectors, unreduced (table SHA-256 f4fb4f57…,
  the economic list's `text_only_sha256`). It gives two reference arms: plain cosine, and cosine
  after the mean vector of the bundle's six-digit codes is subtracted. My calls: D9 reduces the
  table by PCA to an arm's dimension, but this reference is fixed before any model exists, so no
  model's dimension can set the cut. Mean-pooled sentence vectors share a common direction, which
  can weaken plain cosine, so the centered arm keeps a scoring choice from making the bar easy
  (builder check).
- **Model:** cosine over a relatedness table's d-dimension vectors (§4.7).

For each anchor in a part:

- **Candidates.** The six-digit codes that have a marginal row, other than the anchor itself (my
  call, played back). An industry that ITA never sees cannot count against an arm.
- **Relevant partner.** Industry b is a relevant partner of anchor a when J(a,b) ≥ 1, that is,
  when a target row joins them (U47). This is the V9 decision's minimum count for co-ownership.
- **Gain.** The lift among multi-industry employers over the expectation given each industry's
  count of them, J(a,b) · M / (m(a) · m(b)), where M is the number of multi-industry employers
  (K6-Q1). Within one anchor, m(a) and M are constant, and NDCG divides by the anchor's ideal DCG.
  So the gain reduces to J(a,b) / m(b): of b's multi-industry employers, the share that also
  operate in a. It lies in (0, 1]. Candidates that are not relevant have gain 0. My call, from the
  builder check: the played-back base, multi-industry employers, stays, and only the log goes. The
  bundle ships m(a) for each industry but not the total M, which a log gain would need. Without
  M, no co-owned pair can be marked as below chance, so every relevant partner has a positive
  gain.
- **Anchors.** An industry of the part with at least one relevant partner. The part's other
  industries are counted, not scored.
- **Ties** (my call, played back). Each arm's DCG@10 is its expected value over all orders within
  tied groups, computed exactly (McSherry and Najork, 2008). A group that covers ranks s to e adds
  the mean gain of its members times the sum of 1/log2(p + 1) over the positions p from s to
  min(e, 10). Every arm uses this rule. Text→code's rule counts ties against the truth. Under that
  rule, the tree's 5 values would rank most partners behind every tied non-partner.
- **NDCG@10.** The expected DCG@10 divided by the ideal DCG@10.

Over a part:

- **Headline (U44).** The mean NDCG@10 over the part's anchors, each anchor once.
- **Reports.** These never decide, and never add to n:
  - AUC per anchor: relevant partners against candidates with J = 0, ties counted one half.
  - NDCG@20.
  - the headline by the anchor's sector.
  - counts: anchors, candidates, relevant partners, industries with no relevant partner, and
    crossing pairs (§4.3).
- **Result shape.** A per-anchor frame (anchor, part, number of relevant partners, NDCG@10, NDCG@20
  and AUC) and a summary of the headline and reports.

### 4.5 The panel, its reads and the sealed opening

`RelatednessPanel` in `panels/relatedness.py` mirrors `ItaPanel`'s behavior and shares no class with
it (my call, as U39 ruled for text→code). It writes to the same `SelectionLog`, under the panel name
`relatedness_co_ownership`.

- **Reads.** `score_logged(arm, part, purpose, detail)` appends a `read` record before it computes
  any similarity. The record names the panel, the part, the split fingerprint and the number of
  anchors. Its detail names the arm, its seed where it has one, its role (`reference` or
  `candidate`), a fresh run id (`run`) and the part's data fingerprint (my call, after the ITA
  reads). The margin-first check (§4.6) reads these fields.
- **Data fingerprint.** The SHA-256 of what a read scores: for each anchor in order, its code, its
  candidates and their gains, and the two tables' SHA-256 values. It is logged with each read and is
  not a read itself.
- **Sealed part.** Reads of the sealed part fail until `open_sealed(purpose, reopen_reason=None)` is
  called on the panel object. That call checks
  `log.openings('relatedness_co_ownership', split_fingerprint)`. A prior opening without a non-empty
  `reopen_reason` is refused. Otherwise it appends `open`, or `reopen` with the reason.
- **Why the split fingerprint keys openings.** A rebuild in embedding-data changes the rows, so a
  row fingerprint would let a second opening pass as a first. The split fingerprint does not change
  on a rebuild, so the refusal holds.
- **Arms.** An arm returns the similarity matrix of anchors by candidates. One adapter wraps a
  vector table with cosine, and one wraps the tree's D*. The builder names them.

### 4.6 Reference and margin rule

- **Reference arms (U30).** The tree, and the text-only table under plain and centered cosine
  (§4.4). None has seeds, so each is one replicate.
- **Choosing the reference.** For each judging family, the arm with the highest validation
  headline, as a point estimate. A tie goes to the tree, then to plain cosine, the simpler arms
  (my call). The choice is a selection read, so it waits for G6 (§6).
- **The margin.** δ = r × the chosen reference's validation headline, with r = 0.05 (my call, played
  back, as U38 set for text→code). The margin is relative because no reference arm has seeds. It is
  fixed from reference reads only, before any candidate read.
- **The margin record.** One for each judging family. It holds r, every reference arm's validation
  headline, the chosen arm, δ, the split fingerprint, the family's data fingerprint, the
  embedding-data commit, the tables' SHA-256 values and `fixed_at`.
- **Margin-first, in code** (my call, after the ITA pattern). The margin record makes its own fresh
  reference reads under a new run id. A check refuses a margin record when the log holds any
  validation read of that family's panel with role `candidate`, or with no role, made before
  `fixed_at`. Training-monitor reads count as candidate reads. A family that joins later (§4.1)
  gets its own record under the same check.
- **The test (U40's pattern).** For each judging family, Δ = candidate headline − reference
  headline. A candidate beats the reference when the lower bound of the interval on Δ exceeds +δ on
  every judging family. The level is `superiority_level(n)`, 1 − 0.05/n, where n is the number of
  judging families when the test runs: 95 % with co-ownership alone, 97.5 % if TRI joins.
- **Intervals.** The paired bootstrap of `decision/resampling.py`. Anchors are the units, with one
  item each. A candidate's seeds are nested within each resample. The replicates and the bootstrap
  seed come from `conf/data/decision.yaml`, as the existing decisions use them.
- **Seeds.** A candidate has at least `min_seeds` seeds, 5 in `conf/data/decision.yaml` (Req 5). The
  test refuses fewer. The reference arms are deterministic (builder check, answered by Req 5).
- **Which read decides.** Stage 7 chooses one forward candidate on validation. If it does not beat
  the reference by δ on validation, it does not go forward, the sealed part stays sealed, and the
  reference stands (my call: an unspent sealed part can still judge a later candidate). If a
  family joins after Stage 7, the forward candidate must also pass the validation test on it, at the
  new n, before the opening. At the opening (§4.8), the same test runs on every family's sealed part
  with the δ fixed on validation. If it fails there, the reference stands.
- **Provenance** (my call). The decision refuses a candidate table whose provenance names any input
  other than the codebook, the text-only table and the training families of §4.1. This enforces the
  first anti-goal in code, as `decision/provenance.py` does for economic candidates.
- **Code.** A module in `decision/` (the plan names it) holds the reference choice, the margin
  record, the margin-first check, the test and the provenance check, with fixture tests. It does not
  change `compare_panel`, `decision/text_to_code.py` or the economic rule. Its test takes n, because
  `beat_by_margin` in `decision/text_to_code.py` fixes n = 1, and that module stays unchanged (my
  call).

### 4.7 The method (Stage 7)

Stage 7 builds the model. It trains on real data only after G1 opens. Whether it builds and tests
the model on fixtures earlier is the roadmap resume's call.

- **Library.** PyTorch Geometric's `SAGEConv`. `torch-geometric` 2.7.0 is already a dependency,
  because HGCN uses it, so `uv.lock` stays at 6de4fbdc and the economic reads still pair (ADR 0002,
  Negative).
- **Graph.** One node for each codebook code, 2,125 in all.
  - Tree edges join each code to its parent, in both directions, from the codebook.
  - Co-location edges join six-digit industries, from the 2023 ITA area mix. Let k(a) be the
    counties where industry a has a cell, K(a,b) the counties where both have cells, and C the
    counties with eligible records. An edge joins a and b when K(a,b) ≥ 2 and the lift
    K(a,b) · C / (k(a) · k(b)) exceeds 1. Its weight is the log of that lift. My call: presence in a
    county, not its share, because most cells hold one record (K5).
- **Node features.** The frozen text-only vectors, 384 dimensions (the D9 table, unreduced).
- **Model.** `SAGEConv` layers with mean aggregation over both edge types, and an output of d
  dimensions, L2-normalized. The similarity is cosine, and the output is flat (U22).
- **Loss.** Link prediction with negative sampling. Positives are co-location edges, drawn in
  proportion to their weight. Negatives are random pairs of six-digit industries. Tree edges carry
  messages. Whether they also count as positives, and with what weight, is a hyperparameter (my
  call).
- **Hyperparameters and seeds.** Stage 7's plan fixes the grid (d, layers, hidden width, tree-edge
  weight, edge threshold and learning rate) and the seeds before the first candidate read (my
  call). Each configuration runs at least `min_seeds` seeds, 5 (Req 5).
- **Selection on the validation part (U43).** For each seed, the checkpoint is the earliest epoch
  with the highest validation headline, as the text monitor selects. The forward candidate is the
  configuration with the highest mean over its seeds. Each epoch's read is logged with role
  `candidate`.
- **Unseen codes.** Stage 7 reports the validation headline for anchors whose industry has no
  co-location edge, apart from the others. Text and the tree alone place those codes. This takes up
  Req 27's proposed check on unseen industries as a report, with no extra training run (my call).
- **Export.** Each seed's table of the 2,125 codes by d, with a provenance that names the codebook,
  the text-only table, the area-mix run and table SHA-256, the configuration, the seed and the
  checkpoint. Only six-digit codes are judged. Codes at levels 2–5 are placed, not judged.
- **Compute.** The graph is small, so it trains on the Mac (CPU or MPS), with no Lambda campaign (my
  call).
- **Place.** A module of its own inside `graph_model/`, apart from HGCN (ADR 0001). The plan names
  the files.

### 4.8 The sealed opening (U45)

Relatedness opens its sealed part in its own chain, at a stage after Stage 7. The roadmap resume
after Stage 6 adds that stage, for example as Stage 7b. With the user's approval at the time, that
stage opens the sealed part once, scores the forward candidate's seeds and the chosen reference,
applies §4.6's test with the validation δ, and records the decision. Embedding roadmap Stage 12
keeps the text→code and economic openings only, so a parked relatedness chain does not hold it back
(ADR 0002, Negative).

## 5. One role per table or part (Req 26)

| Table or part | Read by | Role for relatedness |
|---|---|---|
| Codebook codes and lineage | Panel, model | The codebook (P1), no role: candidates, D* and tree edges |
| The tree's parent–child edges | Stage 7 | Training |
| The text-only table (D9), unreduced | Stage 7, panel | Training input (node features), and the two text reference arms. It is codebook text, not an embedding-data table |
| `ita_area_mix.parquet`, 2023 | Stage 7 | Training, after G1 |
| `ita_co_ownership.parquet` and `co_ownership_marginals.parquet`, validation part | Stages 6b and 7 | Selection: the reference, the margin, checkpoints and the forward candidate, after G6 |
| The same tables, sealed part | The opening stage (§4.8) | Judging, read once |
| `co_ownership_support.parquet`, the cross-year link table | Nothing | Not read |
| ITA text–code pairs | Nothing | Off-limits: text→code's (protocol C3) |
| TRI parent links | Nothing yet | Judging, only through an amendment before the sealed opening, with its own margin record first (§4.1) |
| ACS, CPS, BEA and USAspending families | Nothing | Unassigned (§4.1) |

Stage 6 reads fixture tables only. Split Req 23's time-window rule for employment data does not
apply, because no economic panel reads the relatedness artifact (Req 29).

## 6. Gates and constraints

**Gates** (roadmap, Gates under Stages). Stage 6 reads no real table, so none of them holds it.

- **G1, R8.4.** It holds Stage 7's training on the area mix. This spec names the one embedding-data
  family that trains: the 2023 ITA area mix. The tree's edges and the text-only table come from the
  codebook and need no amendment.
- **G3 and G5.** The V9 decision sets a minimum count of 1 for co-ownership and the 2023 area mix,
  and both are built. They are open for both families.
- **G4.** No protocol section exists for co-ownership or the area mix. embedding-data's Stage 16
  writes them (§7). It holds Stages 6b and 7.
- **G6, a new gate** (my call: the roadmap resume records it). Stage 10's amendment of 2026-11-18
  allows selection on the text–code pairs only. It states: "Co-ownership and the link table keep
  the full exclusion." Selection on co-ownership's validation part (U43) therefore needs the owner
  to extend that exception to co-ownership, as G2 did for the text pairs. G6 holds Stage 6b's
  reference choice and margin record, and Stage 7's selection.

**Constraints.**

- **`uv.lock`.** It does not change. If a plan step would change it, the step stops and says how the
  economic reads still pair (roadmap, Across chains).
- **Code PRs held** (user, 2026-11-18). A relatedness PR that touches `decision/`, `conf/data/`,
  `uv.lock`, `specs/findings/`, `panels/regressor.py`, `panels/qcew_rows.py`,
  `panels/regressor_splits.py` or `panels/ridge.py` stays unopened until Stage 3 carries its
  COMPLETE stamp. Stage 6's code touches `decision/` and `conf/data/`, so its code PR waits. This
  spec touches `specs/` only.
- **Untouched seams.** `OutcomePanel`, `ItaPanel`, `score_decoding`, `compare_panel`,
  `decision/text_to_code.py`, the economic decision, and every logged record shape that `decide`
  checks.
- **Fixtures.** A fixture builds the two co-ownership tables with Stage 10's exact columns and
  types. It holds a mirror pair split across the parts, ties in the tree arm, a target row whose
  industry has no marginal row, a node id of another level, an industry with no relevant partner,
  and a pair whose J exceeds a marginal count. Its outputs go under `tmp_path`.
- **Style and docs.** The repository's conventions apply. The new modules get API pages in `docs/`
  and entries in `docs/.nav.yml`.

## 7. What this spec asks of embedding-data's owner

The user owns embedding-data. These are requests for that repository's own design sessions. This
spec edits nothing there.

1. **Stage 16's sections** for co-ownership and the 2023 area mix, framed like Section 1:
   - co-ownership: assign its split and its scoring to this spec. The validation part selects, the
     sealed part judges and is opened once, and nothing trains. Name the run, the pin and the
     SHA-256 of `ita_co_ownership.parquet` and `co_ownership_marginals.parquet`.
   - area mix: assign it to relatedness training only, if R8.4 allows it, and state that it never
     judges relatedness. Name the run, the pin and the table's SHA-256.
2. **G6.** Extend Stage 10's selection exception to co-ownership's validation part, under the same
   limits as for the text pairs.
3. **G1, the R8.4 decision.** The 2023 ITA area mix trains the relatedness embedding. No other
   embedding-data family trains it under this spec.

## 8. Handoff to later stages

- **Stage 6 (plan 20)** builds the panel (§4.2–§4.5), its config and the margin-rule module (§4.6),
  on fixtures only.
- **Stage 6b** adds the read command, pins the embedding-data commit and the tables' SHA-256 values
  in its finding, reads both reference arms on the validation part and records the margin. It waits
  for G3, G4, G5 and G6.
- **Stage 7** builds the model (§4.7), trains after G1, selects on validation, chooses the forward
  candidate and applies the validation test.
- **The opening stage** (§4.8) opens the sealed part once and records the decision.
- **The roadmap resume after Stage 6** adds G6 to the Gates paragraph and to Stage 6b's and Stage
  7's HELD stamps, adds the opening stage, and records that Stage 7's method is chosen, so that
  Stage 7 routes to writing-plans.

## 9. Left to the builder

- Module, class and function names, the result types and the fixture builder.
- How the panel and the margin module share helpers with `panels/ita.py` and
  `decision/text_to_code.py` without changing them.
- The salt's value.

## 10. Verification — observable outcomes (roadmap Stage 6, Exit)

- [ ] Each anchor industry falls in one part only. A rebuilt fixture with an added industry keeps
      every old industry in its old part (test).
- [ ] The split fingerprint comes from the committed `conf/data/relatedness_panel.yaml`, and a
      changed salt changes it (test).
- [ ] The panel reads only the co-ownership target and marginal tables. It refuses another family,
      measure, unit type or node-id level, a one-way or unequal mirror pair, a J above either
      industry's m, and a table whose SHA-256 differs from its pin (test).
- [ ] Every pair with a target row is relevant (J ≥ 1), each gain equals J(a,b) / m(b), and the
      candidates are the industries with a marginal row (test).
- [ ] For the tree arm on a small fixture, the expected NDCG@10 equals the mean over every order
      within tied groups, found by brute force (test).
- [ ] The headline counts each anchor once, and no report changes it (test).
- [ ] A margin record is refused when the log holds an earlier validation read with role
      `candidate` or no role (test).
- [ ] A second opening of the sealed part is refused without a reason, also after the fixture tables
      are rebuilt. With a reason, it logs `reopen` (test).
- [ ] The reference choice among the tree and the plain and centered text arms, δ = 0.05 × the
      headline and the beat-by-δ test at 1 − 0.05/n give the expected verdicts on fixture
      replicates, for n = 1 and n = 2. This includes a candidate that is superior to the reference
      but does not beat it by δ (test).
- [ ] The test refuses a candidate with fewer than `min_seeds` seeds (test).
- [ ] A second family's margin record is accepted after candidate reads on the first family, and
      refused when the log holds a candidate read on the second family before its `fixed_at`
      (test).
- [ ] A candidate table whose provenance names co-ownership is refused (test).
- [ ] `OutcomePanel`, `ItaPanel`, `score_decoding`, `compare_panel` and
      `decision/text_to_code.py` are unchanged, and their tests pass unchanged.
- [ ] §5 names every family and part, with its one role.
- [ ] No real embedding-data table was read.
- [ ] The suite, Ruff, the style gate and the strict docs build pass.

## Rollout note

> Roadmap: specs/naics-artifact-redesign-roadmap.md, Stage 6 — on plan completion, tick the stage
> and re-validate later stages against what shipped.
