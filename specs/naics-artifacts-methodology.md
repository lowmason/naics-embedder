# NAICS code representations methodology

> For agentic workers: when specs/naics-artifacts-critique.md exists beside this
> file, REQUIRED SKILL: describe-critique-methodology (synthesize mode) —
> do not draft a spec directly.

This document describes how the 2,125 industries of the 2022 North American Industry
Classification System (NAICS) are represented for three uses. Each use has its own artifact, and
each artifact is judged only on evidence for its own use:

- **Relatedness:** one embedding of how industries go together in practice, for peer and
  competitor discovery, multi-industry profiles and co-location. It is to be trained on the
  classification's tree and on observed co-location, and judged on observed co-ownership, which it
  never trains on.
- **Text-to-code retrieval:** a ranked shortlist of six-digit codes for an owner's own description
  of an establishment. It is trained on the classification's own text and judged on descriptions
  that employers wrote when they filed workplace-injury summaries.
- **Economic coordinates:** a low-dimensional vector per code, for example 16 numbers in place of
  more than 1,000 indicator columns. They are meant to replace sparse industry indicators as
  regressors in downstream models, and are judged on national employment growth.

The classification itself, with its codes, its tree and its text, is not an artifact. It is the
codebook that all three artifacts use.

The document is written for an external methodological review and deliberately omits
implementation detail. It describes the method as it stands on 2026-11-22: what has run on real
data, what is built and tested only on small synthetic tables, and what is specified but not yet
built. It replaces a description of 2026-10-18 that was never reviewed. That version treated the
tree as a third artifact and judged retrieval on held-out index phrases only. An earlier version of
the system, one hyperbolic embedding meant to serve every use, was reviewed externally on
2026-09-23. Neither earlier description need be read.

Numbers come from the system's recorded evaluations on validation splits or validation parts. No
sealed split, sealed part or outer set has been opened. A figure for a trained configuration is a
mean over its training seeds unless stated otherwise. Counts marked "reported by the data builder"
come from that builder's own read-only reconnaissance and are not accepted values.

**Status legend.** Every component carries one of five statuses:

- **Run on real data:** the procedure has run on the real inputs, and its results are on record.
- **Built, fixtures only:** the procedure exists and is tested on small synthetic tables, but has
  read no real data.
- **Specified, not built:** an approved design fixes the procedure, and nothing implements it yet.
- **Not yet specified:** only an outline of the work exists. Its design is still to be written.
- **Parked:** the procedure exists, nothing reads its output, and no planned work revives it.

## Component inventory

Components are labeled S1–S11, which leaves C1, C2, … free for numbering critique points.

| Component (by methodological role) | Status | Estimand / output | Consumes |
|---|---|---|---|
| **S1. Codebook and taxonomy supervision** | Run on real data | The codebook: codes, tree, tree metric $D^{\ast}$, unary pairs and each code's text fields. Training queries with targets, referring codes and candidates. One role for every index entry. Window-fitting summaries | The official NAICS 2022 structure, descriptions, index file and cross-reference file |
| **S2. Shared text encoder** (the *text stage*) | Run on real data: 50 trained code tables over nine geometry × dimension cells | An inductive map from text to a point in a $d$-dimensional geometry, the code table $Z$ and the query map | S1. S3's validation split, which selects checkpoints today |
| **S3. Census-phrase retrieval panel** | Run on real data | Per-query reciprocal rank of the true code among the 1,012 six-digit codes, for held-out index phrases, and its mean (MRR). Reported for text-to-code, decides nothing for it, and selects S2's checkpoints | S1's validation and sealed test queries. A query map and a code table |
| **S4. Owner-description retrieval panel** | Run on real data: its validation part, for two reference arms. Its margin rule and sealed opening are built, fixtures only | Per-pair rank of an employer's reported code in the shortlist for its own description, the headline MRR on owners' own words, and text-to-code's adoption test | Employer–description pairs from workplace-injury filings. The codebook. A query map and a code table |
| **S5. Text-to-code candidate search** | Not yet specified | Candidate encoders selected on S4's validation part, text-to-code's margin record, and at most one forward candidate for S4's sealed part | S2's encoder family, S4 |
| **S6. Growth regressor panel** | Run on real data | Per-row out-of-sample squared error of a ridge model of next-year employment growth, in a seen and a held-out regime | A coordinate table or a reference encoding. Public employment statistics |
| **S7. Economic candidates** | Run on real data. The economic decision is recorded | Eighteen candidate coordinate tables: nine deterministic, built from frozen text and the tree, and nine trained, from S2 | S1. The frozen backbone. S2's exported tables |
| **S8. Co-ownership relatedness panel** | Built, fixtures only | Per-anchor NDCG@10 of an arm's ranking of industries against observed co-ownership, and relatedness's adoption test | Co-ownership counts and marginals from workplace-injury filings. The codebook. An arm's similarities |
| **S9. Relatedness embedding** | Specified, not built | A flat embedding of the 2,125 codes from a graph neural network over tree and co-location edges | S1. The frozen backbone's code vectors. Co-location counts from workplace-injury filings. S8's validation part |
| **S10. Paired decision protocol** | Run on real data for the economic decision. The text-to-code and relatedness rules are built, fixtures only | Adoption decisions between arms or candidates under margins fixed before any candidate read | Per-item values from S3, S4, S6 and S8 for every arm and seed. Margin records |
| **S11. Graph refinement over the tree** | Parked | Refined hyperbolic points for the fixed code set | A hyperbolic code table from S2. The tree as a graph |

## Composition

The components form three chains over one shared codebook. Each chain fixes its reference and its
margin before it reads any candidate, chooses on validation data only, and opens its sealed data at
most once.

1. **Codebook and supervision.** S1 is a deterministic function of the official files and fixed
   seeds. Every other component reads it.
2. **Text-to-code.** S2 fits $\theta$ by stochastic gradient descent on its three-term objective
   $\mathcal{L}$ and keeps
   $$\hat\theta = \text{the earliest } \theta_{e} \text{ maximizing } \operatorname{MRR}^{\mathrm{sel}}(\theta_{e}), \quad e = 0, 1, \dots,$$
   where $\operatorname{MRR}^{\mathrm{sel}}$ is today S3's validation MRR and, once S5 runs, S4's
   validation headline MRR. It exports the query map $q \mapsto \psi_{g}(\phi_{\hat\theta}(q))$
   and the code table $Z$. The artifact ranks the six-digit codes $\mathcal{C}_{6}$ by
   $\operatorname{dist}_{g}(p_{q}, p_{c})$. S10's text-to-code rule judges it on S4: a candidate
   must beat the better of two references by a margin $\delta^{\mathrm{t2c}}$ on S4's validation
   part, and then again on S4's sealed part. The two references are a lexical baseline and S2's
   Euclidean arm at $d = 16$. S3 is reported beside S4 and decides nothing for this artifact.
3. **Economic.** A candidate table $Z'$ from S7 enters S6's ridge model beside two covariates.
   S10's economic rule judges it against the ancestor-indicator reference on both growth regimes.
   The deliverable is the chosen candidate's 2,125-row table, or none.
4. **Relatedness.** S9 will train a graph neural network on tree edges and co-location edges, with
   frozen text vectors as node features, and export a table $Z^{\mathrm{rel}}$. S8 scores
   $Z^{\mathrm{rel}}$ by cosine similarity against observed co-ownership. S10's relatedness rule
   requires it to beat the best of three reference arms by a margin $\delta^{\mathrm{rel}}$, on
   S8's validation part and then once on S8's sealed part. The reference arms are the tree's own
   distance and two cosine similarities of the frozen text vectors.
5. **Graph refinement.** S11 is parked, and nothing reads it.

**Inputs from outside the classification.** Two families of external data enter:

- **Employment statistics.** S6 reads public annual averages of the Quarterly Census of Employment
  and Wages (QCEW) for 2022 through 2025, at national grain.
- **Workplace-injury filings.** S4, S8 and S9 read tables that a separate data-building project
  derives from the Injury Tracking Application (ITA). Through it, establishments that the
  Occupational Safety and Health Administration's (OSHA) rules require to report file their annual
  summary of work injuries (Form 300A). Each record carries the employer's tax identifier, a
  self-reported NAICS code and a free-text industry description. ITA's filers are a
  program-selected population that represents no wider set of establishments, and their codes are
  not validated. Only records with a code native to NAICS 2022, filed for 2023 to 2025, are
  headline evidence: 823,481 of the 3,206,055 records (reported by the data builder). That
  project owns its tables and its rules, and this system reads them read-only.

**Where things stand.**

| Artifact | Reference | State on 2026-11-22 |
|---|---|---|
| Text-to-code | The better of the lexical baseline and the Euclidean $d = 16$ arm, by S4's validation headline MRR | Both references read on S4's validation part: MRR 0.432 for the lexical baseline and 0.450 for the Euclidean arm. No margin record, no candidate and no opening exist |
| Economic | Ridge on the covariates plus ancestor indicators | Decided on 2026-11-18. None of the 18 candidates is adopted, so the reference stands and there is no economic artifact. The economic chain has ended, and its outer sets stay sealed for a later round |
| Relatedness | The best of the tree's distance and two frozen-text cosines, by S8's validation NDCG@10 | Panel and rule exist on fixtures only. No real table has been read, and no model exists |

Before the artifacts were split, one shared space was chosen on three panels at once: S3's, and
S6's seen and held-out regimes in level mode. That decision adopted the Euclidean arm at $d = 16$
(S10). It remains a record. Its winner now serves as one of text-to-code's two references and as
one of the 18 economic candidates.

**Permissions held by the data builder.** The builder's owner must grant three things before the
relatedness chain can read real data, and has granted none of them:

- permission to train any artifact on a table that the builder publishes, which S9 needs for
  co-location
- permission to select on co-ownership's validation part, which S8's reference choice and margin
  and S9's checkpoint selection need
- a protocol section that pins the exact co-ownership and co-location tables a read may use.

The same owner has permitted selection on S4's validation part and has pinned S4's tables. If the
first permission is never granted, the relatedness chain parks.

**Couplings that survive the split.**

- **Retrieval selects every trained table.** All 50 of S2's tables were selected on S3's
  Census-phrase MRR. So were the nine trained economic candidates and text-to-code's Euclidean
  reference.
- **One part serves several selections.** Within text-to-code and within relatedness, one
  validation part is to choose the reference, fix the margin, select checkpoints and choose the
  forward candidate. Only the sealed part judges.
- **The frozen text vectors are everywhere.** They are S6's frozen-text comparator, three economic
  candidates directly and three more through smoothing, two of S8's reference arms, and S9's node
  features.
- **The tree is everywhere.** It supplies S2's code–code target, S6's reference, three economic
  candidates, one of S8's reference arms and S9's tree edges.
- **One filing population feeds two artifacts' evidence.** S4's pairs, S8's co-ownership and S9's
  co-location all come from the same workplace-injury filings.

## Cross-component assumptions

1. **One code universe and one vintage.** Every component indexes the same 2,125 codes of NAICS
   2022. The employment statistics are published on that basis, and the injury filings are read
   only where the filer reported a code native to it. *Breaks if violated:* a new vintage needs
   new supervision, new panels and a concordance. Nothing here maps between vintages, and about
   40% of the 2024–2025 filings, which carry an older vintage's code, supply no evidence.
2. **One role per table or part.** For each artifact, every table, or every part of a split table,
   has exactly one role: training, selection or judging. Training covers training terms,
   curriculum and negative mining. Selection covers checkpoint, reference and candidate choice.
   *Breaks:* a part read for both selection and judging turns its judging estimate into another
   validation read.
3. **Judges never train.** No held-out index phrase, no owner-description pair and no co-ownership
   row enters any training text or training graph. Economic candidates read the classification's
   text and tree only, never employment data or any panel. *Breaks:* each panel would score a
   representation fitted to its own outcome.
4. **Validation-only selection, single openings.** Every selection reads validation splits or
   validation parts only. Each artifact's sealed data is opened once, after its validation test,
   and a second opening needs a stated reason that is logged. *Breaks:* repeated openings turn the
   sealed estimate into another validation read.
5. **Training and judging families share a population.** Relatedness is to be trained on
   co-location and judged on co-ownership. Both are built from the same filers' records. *Breaks:*
   an employer that runs establishments of two industries in one county adds to both families, so
   the held-out family is not independent of the training family, and a model can score on
   co-ownership through co-location it was trained on.
6. **Self-reported codes are truth.** Both new judges, S4 and S8, take the filer's own code as the
   label. *Breaks:* misreported codes put noise into the label, and a filer who copies a
   classification title reports what the classification's text suggests, which favors a model
   trained on that text.
7. **Pairing within a tolerance.** Paired comparisons require identical rows. In the economic
   decision, they also require the reference's predictions to equal the margin read's to within
   $10^{-13}$, absolute, per prediction. *Breaks:* compared reads must run under one locked set of
   library versions, and a platform whose arithmetic differs by more than the tolerance cannot
   pair.
8. **Seeds carry training randomness.** A trained arm's variability is represented by five to ten
   seeds. A deterministic arm or candidate has none. *Breaks:* comparisons between trained and
   deterministic arms treat the deterministic side as exact.
9. **The tree is the only structure in two artifacts.** Text-to-code and the economic candidates
   learn structure from the tree alone. Only relatedness reads observed relations. *Breaks:*
   relatedness that the tree does not encode is neither trained nor credited in the other two
   artifacts, and an economic candidate that re-encodes the tree competes with a reference that is
   the tree.

## Notation

Shared symbols. Each component section extends this table.

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\mathcal{C}$ | The codebook: NAICS 2022 codes at all five levels | $N = 2125$ codes |
| $\mathcal{C}_{l}$ | Codes at level $l$ | 20, 96, 308, 689 and 1,012 codes at $l = 2, \dots, 6$ |
| $c$, $c'$, $j$ | Generic codes | $\mathcal{C}$ |
| $a$, $a'$ | An anchor code and a code compared with it (S2, S8, S9) | $\mathcal{C}$ |
| $\lambda(c)$ | Level of $c$: its number of digits. A sector written as a range, such as 31–33, is level 2 | $\{2, \dots, 6\}$ |
| $\varnothing$ | Virtual root above the 20 sectors | level 1 |
| $D^{\ast}(c, c')$ | Tree metric: the number of edges on the path from $c$ to $c'$, through $\varnothing$ across sectors | $\{0, 1, \dots, 10\}$ |
| $\mathcal{U}$ | Unary pairs: a five-digit code with exactly one child, and that child | 522 pairs |
| $\mu(c)$ | Unary partner of $c$: its pair's other member, or empty when $c$ is in no pair | |
| $F$ | Code text fields: title, description, examples, exclusions | four fields |
| $x^{f}_{c}$, $m^{f}_{c}$ | Text of field $f$ of code $c$, and its presence indicator | token sequences, $\{0, 1\}$ |
| $q$ | A query: a short activity phrase or an owner's description | text |
| $\theta$ | Trainable parameters of the text encoder (S2) | |
| $\phi_{\theta}$ | Text map to the projection space (S2) | text $\to \mathbb{R}^{d}$ |
| $g$, $\psi_{g}$ | Geometry arm, and its parameter-free head | {Euclidean, spherical, hyperbolic}, $\mathbb{R}^{d} \to$ points |
| $d$ | Embedding dimension | $\{8, 16, 32\}$ |
| $p$, $p_{c}$, $p_{q}$ | A point, the point of code $c$, the point of query $q$ | |
| $\operatorname{dist}_{g}$ | Distance of arm $g$ (S2) | $[0, \infty)$ |
| $r$ | Radius of a point (S2, S11) | $[0, \infty)$ |
| $\mathbb{H}^{d}$ | Unit-curvature hyperboloid: $p \in \mathbb{R}^{d+1}$ with $\langle p, p\rangle_{\mathcal{L}} = -1$ and $p_{0} > 0$ | |
| $\langle p, p'\rangle_{\mathcal{L}}$ | Lorentz inner product, $-p_{0}p'_{0} + \sum_{l=1}^{d} p_{l}p'_{l}$ | $\mathbb{R}$ |
| $z_{c}$, $Z$ | Exported coordinates of code $c$, and the code table of all codes | $\mathbb{R}^{d}$, $2125 \times d$ |
| $\bar h^{\circ}_{c}$ | Frozen text vector of code $c$: the backbone without adapters reads each present field's text without a field marker (with over-window texts replaced by their summaries), mean-pools its token states, and averages over the present fields | $\mathbb{R}^{384}$ |
| $\operatorname{PCA}_{d}$ | Projection onto the first $d$ principal components, fitted on all 2,125 codes | $\mathbb{R}^{384} \to \mathbb{R}^{d}$ |
| $Z'$ | A candidate coordinate table for the economic artifact (S7) | $2125 \times d$ |
| $\operatorname{RR}$, $\operatorname{MRR}$ | Reciprocal rank of a query's truth, and its mean over a set of queries | $(0, 1]$ |
| $\operatorname{MRR}^{\mathrm{sel}}$ | The MRR that selects S2's checkpoints: S3's validation MRR today, S4's validation headline once S5 runs | $(0, 1]$ |
| $\operatorname{part}(\cdot)$ | A salted-hash split of a unit into a validation part (0) and a sealed part (1) (S4, S8) | $\{0, 1\}$ |
| $k$ | Training seed | |
| $P$ | A panel or part read by a decision | |
| $\delta$ | A decision margin, in the units of its panel's statistic (S10) | |

## Per-component descriptions

### S1. Codebook and taxonomy supervision

**Status: run on real data.** Unchanged since the previous description.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\xi$ | A cross-reference row: a sentence in a code's text that sends an activity elsewhere, for example "Growing soybeans--are classified in Industry 111110" | 4,623 rows |
| $o(\xi)$ | The referring code, whose text contains $\xi$ | $\mathcal{C}$ |
| $\operatorname{nm}(\xi)$ | Codes the row names | subset of $\mathcal{C}$ |
| $\operatorname{lin}(\xi)$ | Named codes that are ancestors or descendants of $o(\xi)$ | subset of $\operatorname{nm}(\xi)$ |
| $\operatorname{act}(\xi)$ | Activity phrase: the row's text before "is" or "are" followed by "classified" or "included" | text, or none |
| $\mathcal{E}$ | Index entries: short activity phrases the classification lists under six-digit codes | 20,373 entries over 1,010 codes |
| $\mathcal{E}^{\mathrm{ex}}$, $\mathcal{E}^{\mathrm{tr}}$, $\mathcal{E}^{\mathrm{val}}$, $\mathcal{E}^{\mathrm{te}}$ | Entry roles: examples-field text, training query, validation query, sealed test query | 6,118, 7,200, 4,042 and 3,013 entries |
| $\mathcal{Q}$ | Training queries | 11,039 |
| $\lambda(q)$ | Level of a training query's target codes | $\{2, \dots, 6\}$ |
| $T(q)$ | Target codes of $q$ | non-empty subset of $\mathcal{C}_{\lambda(q)}$ |
| $N(q)$ | Referring codes of $q$: codes whose text redirected the activity elsewhere | subset of $\mathcal{C}$ |
| $C(q)$ | Candidates of $q$: $\mathcal{C}_{\lambda(q)} \cup N(q)$ | subset of $\mathcal{C}$ |
| $L$ | Input window: the backbone's trained sequence length, special tokens included | 128 tokens |
| $\operatorname{depth}(c)$ | Depth of $c$ below the virtual root, $\lambda(c) - 1$. The root's is 0 | $\{0, \dots, 5\}$ |
| $c \wedge c'$ | Lowest common ancestor of $c$ and $c'$: the virtual root when they lie in different sectors | |

#### Problem formulation and data-generating story

S1 turns four official files into the supervision that every later component reads: the list of
codes with titles, the descriptions, the index file and the cross-reference file. It is
deterministic apart from one frozen, seeded draw (the entry roles, S3), and it posits no sampling
model. Its outputs are the estimand:

- the codebook and tree, which every artifact uses
- the tree metric $D^{\ast}$ and the unary pairs
- each code's text fields
- the training queries, with targets, referring codes and candidates
- one role for every index entry
- extractive summaries for field texts too long for the window.

They are written once as an immutable bundle with a manifest of hashes, which every later stage
validates before use.

#### Estimation / inference procedure

**Codebook and tree.**

- 2,125 codes: 20, 96, 308, 689 and 1,012 at levels 2 to 6. Three sectors published as ranges
  (31–33, 44–45, 48–49) are one code each.
- A code's parent is the code with its last digit removed. A three-digit code's parent is its
  sector, with the range sectors unified. A virtual root sits above the 20 sector trees.
- The tree metric is the edge count through the lowest common ancestor:
  $$D^{\ast}(c, c') = \operatorname{depth}(c) + \operatorname{depth}(c') - 2\operatorname{depth}(c \wedge c').$$
  Within a sector this is the path length. Across sectors it is $\lambda(c) + \lambda(c') - 2$.
  Values are the integers 1 to 10, and between two six-digit codes only 2, 4, 6, 8 and 10 occur.
  88% of the 2,256,750 pairs are cross-sector. The triangle inequality was checked over every
  ordered triple.
- **Unary pairs.** 522 five-digit codes have exactly one six-digit child, and each forms a pair
  with that child. Single-child parents at levels 2 to 4 are not paired. For example, 140
  four-digit codes have one child.

**Text fields.** Each field is either present or absent. There is no placeholder.

- **Title:** as published, present for every code. None exceeds the window.
- **Description:** the published text, cleaned of markup, with its cross-reference paragraphs and
  illustrative-example blocks removed.
  - 1,449 codes have their own text.
  - 662 codes with no text and a single child inherit that child's description, reworded to
    their level: the 522 five-digit unary parents and 140 four-digit codes.
  - 14 codes with several children and no text have none.
- **Examples:** for the 1,010 codes with index entries, exactly their examples-role entries (S3),
  joined. For 65 codes without entries, the illustrative examples from their description. Present
  for 1,075 codes. Absent for every code at levels 2 to 4, for 624 of the 689 five-digit codes,
  and for 2 six-digit codes.
- **Exclusions:** the text of the code's cross-reference rows, each row once, present for 1,117
  codes.

**Redirections.**

- **Rows.** A cross-reference reroutes an activity. It does not assert that two codes are
  unrelated. Of 4,623 rows, 4,533 name at least one code.
- **Phrases.** The activity phrase $\operatorname{act}(\xi)$ is the text before the redirection
  verb. 4,529 rows carry one, 3,326 of them distinct.
- **Lineal rows.** Nine rows name an ancestor or a descendant of their own code. Those stay text
  only.
- **Withheld rows.** Five rows matched by a held-out query (S3) are withheld: their phrase leaves
  training and their text leaves the exclusions field.
- **Rows that name no code.** 68 rows name their destination only in words and yield no query.

**Training queries.** A query is a distinct pair of text and level, with targets $T$ and referring
codes $N$:

- each training-role entry contributes (its text, 6), with its code in $T$
- each non-withheld row $\xi$ and each named code $j \notin \operatorname{lin}(\xi)$ contribute
  $(\operatorname{act}(\xi), \lambda(j))$, with $j \in T$ and $o(\xi) \in N$
- rows that share a phrase and a level merge, taking unions, and an entry and a phrase with the
  same text at level 6 merge the same way
- a query whose $T$ and $N$ overlap is refused, and none occurs.

The result is 7,200 entry queries and 3,860 phrase queries (54, 97, 214, 894 and 2,601 at levels 2
to 6), with 21 merges: 11,039 queries, 9,780 of them at level 6. Nineteen queries have two or more
targets, and 517 codes are no query's target.

**Window-fitting summaries.** Each present field text is read with a two-token marker naming its
field. A marked text longer than $L$ is replaced by an extractive summary:

1. The text is cut into units: sentences, re-split into clauses and then pieces when one is too
   long. The examples field is cut into its entries.
2. The frozen backbone, without adapters, embeds each unit.
3. Units are added greedily. Each step adds the unit that most raises the cosine between the kept
   units' token-weighted sum and the whole text's, while the marked, source-ordered join fits in
   $L$. Selection stops when nothing fits or nothing raises the cosine.
4. The kept units are emitted in source order.

There are 753 summaries: 162 descriptions (7.7% of those present), 106 examples fields (9.9%) and
485 exclusions fields (43%). On average they keep 62% to 65% of a text's units. Summaries are
committed with a hash, and each is checked to be an in-order extract that fits. No field text is
truncated. A query longer than the window would be truncated. How many are, if any, is not
recorded.

**The bundle.** Every member is hashed. At each load, the codebook order, every property of
$D^{\ast}$, the unary flags, the consistency of redirections and exclusions, and the shape of the
role table are checked again. Leakage and the examples field's composition are checked when the
bundle is built, not when it is loaded.

**Graph-stage material.** The bundle also holds an older universe of training pairs for S11:
within-sector related pairs that are neither exclusions nor unary pairs, with capped negatives. It
is validated at every load, and the text stage never reads it.

#### Assumptions and limitations

1. **Official text describes codes comparably.** *Breaks if violated:* text length and style vary
   by sector. 662 codes carry a child's text and 14 have no description. An encoder can confuse
   style with content.
2. **Field presence carries level.** The examples field is absent for every code at levels 2 to 4
   and for most five-digit codes. *Breaks:* although the masked mean hides absence, which fields
   are present still reveals a code's level.
3. **Near-identical codes.** A unary parent carries its child's description, so the pair is close
   to identical in text. Only these 522 pairs are masked from each other in S2. *Breaks:* the 140
   four-digit single-child codes share text with their child but remain separate codes at
   $D^{\ast} = 1$.
4. **Every edge counts the same.** *Breaks:* two six-digit siblings and two sibling subsectors
   both sit at $D^{\ast} = 2$, whatever their economic distance.
5. **Redirection wording.** The phrase pattern keys on one verb construction. *Breaks:* rows
   worded otherwise contribute no query, and 68 name no code at all.
6. **Centrality summaries drop the atypical.** *Breaks:* the units least like the whole are the
   first to go, and in exclusion texts those are often the boundary-defining sentences. 3,186 of
   the 4,623 redirection rows sit in summarized exclusion texts. Their destinations still reach
   training through their phrase queries.
7. **The leakage rule is one-sided.** Only held-out queries are checked. *Breaks:* a training query
   may string-match an examples-field entry of its own code, so the task term partly rewards string
   matching.
8. **Untargeted codes.** 517 codes are no query's target. *Breaks:* their points come from their
   text and the code–code term alone.
9. **Owners copy this text.** The classification's titles and index entries are also what many
   owners type (S4). *Breaks:* the more an encoder learns the classification's own phrasing, the
   better it scores on copied owner text, which S4 therefore reports apart from its headline.

#### Evaluation criteria

- $D^{\ast}$ is an integer tree metric, satisfies the triangle inequality and has the cross-sector
  identity, checked at build and at every load.
- No held-out index phrase matches any training text, phrases included, exactly or as a
  near-duplicate.
- Every summary is an in-order extract of its source and fits the window. Identical sources get
  identical summaries.
- Every bundle member is re-hashed and re-validated before any training, export or read.

#### Open questions for the reviewer

1. Is an unweighted edge count the right code–code target, or should similarity be weighted by
   depth or by information content computed from the classification alone? An
   information-content target is one of S5's planned ablations.
2. Are cross-reference phrases good training queries? Owner-written descriptions now exist (S4),
   and a later training part drawn from them is recommended but not designed. Which other public
   activity-to-code sources could supervise retrieval: survey write-ins, crosswalks from other
   classifications, business registers?
3. For texts longer than the window, are extractive centrality summaries preferable to a
   longer-window backbone or to chunked encoding?
4. Should near-identical codes beyond five-to-six-digit chains be masked from each other too?

### S2. Shared text encoder (the text stage)

**Status: run on real data.** Fifty trained code tables exist: ten seeds of the original reference
configuration and five seeds of each of eight other geometry × dimension cells. The method is
unchanged since the previous description. What changed is how its outputs are used.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\mathcal{M}$ | The pretrained backbone: a six-layer transformer sentence encoder (MiniLM-L6-v2, about 22 million weights), frozen except for its adapters | width $H = 384$, input window 128 tokens |
| $\omega$ | Low-rank adapters on every linear map of $\mathcal{M}$: rank 8, scaling factor 16, dropout 0.1 | trainable |
| $h^{f}_{c}$ | Channel vector of field $f$ of code $c$: the mean of $\mathcal{M}$'s final token states over the field's tokens, padding excluded | $\mathbb{R}^{H}$ |
| $\bar h_{c}$ | Fused vector of code $c$ | $\mathbb{R}^{H}$ |
| $W$, $b$ | The projection's weight and bias | $\mathbb{R}^{d \times H}$, $\mathbb{R}^{d}$ |
| $v$ | Projection output, $W\bar h + b$ | $\mathbb{R}^{d}$ |
| $\nu$, $u$ | Norm $\Vert v\Vert$ and direction $v/\nu$ of $v$, with $u = 0$ when $v = 0$ | $[0, \infty)$, unit vectors or 0 |
| $R$ | Radius bound of the hyperbolic arm | 8 |
| $s_{\mathrm{task}}$, $s_{\mathrm{code}}$ | Learned logit scales of the task term and the code–code term | $[0.01, 100]$, both starting at 1 |
| $\tau$ | Temperature of the code–code target | 1 |
| $\rho$ | Radial step: the target radius gained per level | 1 |
| $w_{\mathrm{code}}$, $w_{\mathrm{rad}}$ | Weights of the code–code and radial terms | 1, 1 |
| $J_{a}$ | Comparison set of anchor $a$: every code except $a$ and its unary partner, $\mathcal{C} \setminus (\{a\} \cup \mu(a))$ | $\subset \mathcal{C}$ |
| $\pi_{a}$, $\sigma_{a}$ | Target and model distributions of anchor $a$ over $J_{a}$ | probability vectors |
| $\mathcal{A}$, $\mathcal{B}$ | Code anchors and task queries of one step | 24 or 25 anchors, 126 or 127 queries |
| $\tilde p_{c}$ | Cached point of code $c$: its radius and direction at the last refresh, held without gradient | |
| $e$ | Epoch index | $0, \dots, 39$ |
| $\eta$ | Learning rate | peak $10^{-4}$ |
| $\hat\theta$ | Selected parameters: the earliest epoch with the highest selection MRR | |

#### Problem formulation and data-generating story

The text stage fits one inductive map from text to a point, shared by code texts and queries:

$$\phi_{\theta}:\ \text{marked text fields} \;\longmapsto\; v \in \mathbb{R}^{d} \;\longmapsto\; p = \psi_{g}(v),$$

where $\theta = (\omega, W, b, \log s_{\mathrm{task}}, \log s_{\mathrm{code}})$ and $\psi_{g}$ is
the parameter-free head of geometry arm $g$. A code is read through its present fields, a query
through its single query field. The map is inductive: any text, including text that never appeared
in training, has a point. An owner's description is read the same way as an index phrase.

The observed data are fixed: the codes' fields $x^{f}_{c}$ with presence $m^{f}_{c}$, the training
queries $q \in \mathcal{Q}$ with their levels $\lambda(q)$, targets $T(q)$ and referring codes
$N(q)$, the tree metric $D^{\ast}$, the unary partners $\mu$ and the levels $\lambda$ (all from
S1). No generative model of the text is posited. The stage is discriminative metric learning. Its
randomness comes from the training seed $k$: the adapters' and projection's initialization, adapter
dropout, and the per-epoch orders of codes and queries.

The estimand is $\hat\theta$, the parameters that the selection rule picks, and through it two
outputs: the **code table** $Z = (z_{c})_{c \in \mathcal{C}}$, every code's exported coordinates,
and the **query map** $q \mapsto \psi_{g}(\phi_{\theta}(q))$ used for retrieval. Three
requirements are written into the objective:

1. a query should rank its target codes first among its candidates
2. distances between codes should reproduce the neighborhoods of the tree metric
3. in the hyperbolic arm only, a code's radius should track its level.

#### Estimation / inference procedure

**Architecture.** Each present field of a code is prefixed with a marker naming the field (title,
description, examples, exclusions, or query for a query) and read by the one adapted backbone. A
field longer than the input window is replaced by its committed window-fitting summary (S1). Text
is never silently truncated. Absent fields never enter the backbone.

- Channel vectors: $h^{f}_{c}$ is the mean of the final token states over the field's tokens.
- Fusion: the default is the masked mean over present fields,
  $$\bar h_{c} = \frac{\sum_{f \in F} m^{f}_{c}\, h^{f}_{c}}{\max\!\bigl(1, \sum_{f \in F} m^{f}_{c}\bigr)},$$
  so a missing field neither contributes nor signals its absence. Two alternatives exist as
  ablations, and no recorded run uses them: attention pooling (a learned vector scores the present
  fields, softmax-weighted, starting at the masked mean) and a top-2 mixture of four experts
  applied to the masked mean, with a load-balancing penalty of weight 0.01.
- Projection: one affine map, $v = W\bar h_{c} + b$, to $d \in \{8, 16, 32\}$.
- Head: $p = \psi_{g}(v)$, by arm.

The frozen text vector $\bar h^{\circ}_{c}$ of the shared notation differs from this map's fused
vector in two ways: it has no adapters, and its field texts carry no field marker. It reads the
same summaries for over-window texts.

**Geometry arms.** The three heads share everything before them and differ in the point, the
distance and the exported coordinates.

| Arm $g$ | Point $p$ | Distance $\operatorname{dist}_{g}(p_{1}, p_{2})$ | Exported $z$ |
|---|---|---|---|
| Euclidean | $v$, with radius $r = \nu$ | $\Vert v_{1} - v_{2}\Vert = \sqrt{(r_{1} - r_{2})^{2} + r_{1} r_{2} \Vert u_{1} - u_{2}\Vert^{2}}$ | $v$ |
| Spherical | $u$, with radius 1 | $\tfrac12 \Vert u_{1} - u_{2}\Vert^{2} = 1 - \cos\angle(u_{1}, u_{2})$ | $u$ |
| Hyperbolic | $(\cosh r, \sinh r\, u) \in \mathbb{H}^{d}$, with $r = R \tanh(\nu / R)$ | $2 \operatorname{arsinh}\sqrt{\sinh^{2}\tfrac{r_{1} - r_{2}}{2} + \sinh r_{1} \sinh r_{2}\, \tfrac{\Vert u_{1} - u_{2}\Vert^{2}}{4}}$ | $r u$, the tangent vector at the origin |

- The hyperbolic distance equals the geodesic distance $\operatorname{arcosh}(-\langle p_{1},
  p_{2}\rangle_{\mathcal{L}})$ at unit curvature. It is computed in the half-angle form above,
  whose terms are all non-negative, so nearby points at large radius do not cancel in single
  precision. Curvature is fixed at 1, and there is no curvature parameter.
- The bound $r = R\tanh(\nu/R)$ keeps every radius below $R$ while passing gradient at any length:
  $dr/d\nu = \operatorname{sech}^{2}(\nu/R)$, about 0.61 at radius 5 when $R = 8$.
- The spherical arm exports $u$ rather than $v$, because its distance gives $\nu$ no training
  signal.
- A table is read back by the arm's own map: the exponential map at the origin for the hyperbolic
  arm, the identity for the flat arms.

**Objective.** One step reads two streams, the anchors $\mathcal{A}$ and the queries
$\mathcal{B}$. Every candidate point is a cached point $\tilde p_{c}$, except that the step's
anchors are replaced by their live points. Gradient therefore reaches a code only through the steps
in which it is an anchor, and reaches every query directly. Three terms:

$$\mathcal{L}_{\mathrm{task}} = \frac{1}{|\mathcal{B}|} \sum_{q \in \mathcal{B}} -\log \frac{\sum_{c \in T(q)} \exp\!\bigl(-s_{\mathrm{task}} \operatorname{dist}_{g}(p_{q}, p_{c})\bigr)}{\sum_{c \in C(q)} \exp\!\bigl(-s_{\mathrm{task}} \operatorname{dist}_{g}(p_{q}, p_{c})\bigr)}, \qquad C(q) = \mathcal{C}_{\lambda(q)} \cup N(q),$$

$$\mathcal{L}_{\mathrm{code}} = \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} -\sum_{j \in J_{a}} \pi_{a}(j) \log \sigma_{a}(j), \qquad \pi_{a}(j) = \frac{\exp\bigl(-D^{\ast}(a, j)/\tau\bigr)}{\sum_{j' \in J_{a}} \exp\bigl(-D^{\ast}(a, j')/\tau\bigr)}, \quad \sigma_{a}(j) = \frac{\exp\bigl(-s_{\mathrm{code}} \operatorname{dist}_{g}(p_{a}, p_{j})\bigr)}{\sum_{j' \in J_{a}} \exp\bigl(-s_{\mathrm{code}} \operatorname{dist}_{g}(p_{a}, p_{j'})\bigr)},$$

$$\mathcal{L}_{\mathrm{rad}} = \frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \bigl(r_{a} - \rho\,(\lambda(a) - 1)\bigr)^{2} \quad \text{(hyperbolic arm only)},$$

$$\mathcal{L} = \mathcal{L}_{\mathrm{task}} + w_{\mathrm{code}}\, \mathcal{L}_{\mathrm{code}} + w_{\mathrm{rad}}\, \mathcal{L}_{\mathrm{rad}}.$$

- The task term sums probability over all of a query's targets, so a query with two targets pays
  nothing for how it splits mass between them. Codes outside $C(q)$ carry no probability and take
  no gradient. Every referring code stays a candidate, so a redirected activity is pushed toward
  its destination and away from the code whose text redirected it.
- The code–code term is a listwise cross-entropy against a fixed graded target. It reads no
  exclusion data, so an exclusion never acts as code–code repulsion. Removing the unary partner
  from $J_{a}$ makes a five-digit industry and its only six-digit child neither positive nor
  negative for each other.
- The radial term is a soft target: radii 1, 2, 3, 4 and 5 at levels 2 to 6 under $\rho = 1$, with
  codes of one level free to differ. Queries have no radial target.
- Each scale is the exponential of a trainable logarithm, clamped to $[\log 0.01, \log 100]$.
- Under the mixture-of-experts ablation only, the load-balancing penalty is added. No negative
  miner, false-negative estimate, curriculum or other term exists.

**Epochs and the cache.** An epoch visits every training query and every code anchor exactly
once, in orders permuted by seed and epoch. With $|\mathcal{Q}| = 11{,}039$ queries at most 128 per
step, an epoch has 87 steps of 126 or 127 queries, and the 2,125 anchors are spread over them as
evenly as possible. At the start of training and after each epoch, every code is encoded in
evaluation mode (adapter dropout off, no gradient) to refresh the cache. Within an epoch,
candidates therefore lag the live parameters by up to 87 steps.

**Optimization.**

- AdamW with decoupled weight decay 0.01, none on the two scales.
- The learning rate rises linearly over the first epoch to $\eta = 10^{-4}$.
- After each epoch, the end-of-epoch cache is used to score the selection panel. Today that is
  S3's validation split. The learning rate is halved after more than two consecutive epochs
  without a new best selection MRR. Training stops after five such epochs, or at 40 epochs.
- Gradient norm clipped at 1, one device. On a GPU the backbone runs in bfloat16 mixed precision,
  and pooling onward (fusion, projection, head, distances, losses) in single precision.

**Selection and export.** $\hat\theta$ is the earliest epoch with the highest selection MRR. A
later tie never replaces it. Every code is then encoded under $\hat\theta$ in evaluation mode, and
its coordinates $z_{c}$ are written with a provenance record naming the arm, backbone revision,
supervision and summaries. The structural diagnostics (Evaluation criteria) play no part in
selection, the learning-rate schedule or stopping. S5 plans to replace S3 by S4's validation part
as the selection panel for text-to-code candidates.

**Configurations on record.** The original reference configuration is the hyperbolic arm at
$d = 16$ with every value above. A geometry-by-dimension experiment then trained the other eight
cells of {Euclidean, spherical, hyperbolic} × {8, 16, 32}, changing nothing else, on five seeds
each. The original reference has ten seeds. Mean validation statistics, from S3 and S6 in level
mode:

| Arm | Seeds | Census-phrase MRR (SD) | Selected epochs | Level regressor MSE, seen (SD) | Held-out (SD) |
|---|---|---|---|---|---|
| Euclidean, 16 | 5 | 0.592 (0.005) | 38–39 | 0.0793 (0.0054) | 0.0895 (0.0070) |
| Euclidean, 8 | 5 | 0.521 (0.006) | 37–39 | 0.1037 (0.0147) | 0.1112 (0.0166) |
| Spherical, 32 | 5 | 0.520 (0.004) | 1 | 0.0780 (0.0037) | 0.0952 (0.0057) |
| Euclidean, 32 | 5 | 0.519 (0.005) | 1 | 0.0756 (0.0043) | 0.0921 (0.0063) |
| Hyperbolic, 32 | 5 | 0.453 (0.014) | 1–5 | 0.0812 (0.0054) | 0.0982 (0.0078) |
| Spherical, 16 | 5 | 0.440 (0.004) | 1–2 | 0.0932 (0.0086) | 0.1051 (0.0109) |
| Hyperbolic, 8 | 5 | 0.422 (0.041) | 38–39 | 0.1186 (0.0121) | 0.1273 (0.0147) |
| Hyperbolic, 16 (original reference) | 10 | 0.413 (0.040) | 2–39 | 0.0848 (0.0111) | 0.0950 (0.0123) |
| Spherical, 8 | 5 | 0.329 (0.016) | 2–5 | 0.1122 (0.0158) | 0.1191 (0.0168) |

S10's rule, applied to all nine cells, adopted Euclidean $d = 16$ over each of the others. That
decision stands as a record and defines no deliverable. Since then the nine cells' 50 tables have
served in two new roles:

- the Euclidean $d = 16$ arm, with its five recorded checkpoints, is one of text-to-code's two
  references on S4, and it is not selected again there
- all nine cells are economic candidates (S7), read on S6's growth regimes with their recorded
  checkpoints.

#### Assumptions and limitations

1. **The official text suffices to place a code.** *Breaks if violated:* codes with thin text (S1:
   blank descriptions or examples) are placed mostly by their title and the tree terms, and their
   points carry little of the text's information.
2. **Index entries and redirection phrases stand in for real queries.** Every training query is
   the classification's own index or cross-reference phrase. *Breaks:* owners' own words are
   differently worded. S4 measures the shift: on owner descriptions that copy no classification
   text, the Euclidean $d = 16$ arm's MRR is 0.14 to 0.16 below its MRR on held-out index phrases.
3. **The tree metric is the right code–code target.** *Breaks:* industries that are economically
   or semantically close in different branches, such as making a good and wholesaling it, are
   pushed apart by a target that only knows the tree. $\tau = 1$ fixes how sharply the target
   decays, with no ablation on record.
4. **Candidates are restricted to one level.** The task term ranks a query only against codes at
   its targets' level plus its referring codes. *Breaks:* nothing in training teaches the model to
   separate a query from codes at other levels, which matters for any use that searches the whole
   codebook.
5. **Stale candidates.** Cached candidates lag the live parameters by up to one epoch, and a code
   takes gradient only as one of the 24 or 25 anchors of a step. *Breaks:* when parameters move
   fast, the task term's softmax is computed against out-of-date candidates.
6. **Low dimension.** The projection to $d \le 32$ was chosen for the economic use. *Breaks:* for
   retrieval, the bottleneck discards most of the backbone's 384 dimensions. Since the split,
   retrieval carries no low-dimension requirement, and the economic decision adopted no trained
   table.
7. **A bi-encoder.** Queries and codes are encoded independently and compared by one distance.
   *Breaks:* distinctions that depend on reading the query and the code text together are out of
   reach. No reranking stage exists.
8. **Training length interacts with the arm.** Euclidean $d \in \{8, 16\}$ and hyperbolic $d = 8$
   selected epochs 37–39 of a 40-epoch budget, at or near its end. Every $d = 32$ arm and the
   spherical arms at $d \le 16$ peaked by epoch 5. *Breaks:* the geometry-by-dimension comparison
   compares arms at different points of their training curves, so it confounds geometry and
   dimension with budget and stopping.
9. **Seed sensitivity in the original reference.** Its ten seeds span validation MRR 0.369 to
   0.470. Three seeds stopped at epoch 7 or 8 with their best at epoch 2 or 3, and they are among
   the four weakest on all three level-mode panels. *Breaks:* five consecutive non-improving epochs
   may end runs that would improve later, and the across-seed SD that set the shared-space margins
   includes this stopping noise.
10. **A fixed radial schedule.** The radial target spaces levels equally, whatever a subtree's
    branching. *Breaks:* a schedule that fights the distance terms in wide subtrees. The
    hyperbolic arm at $d = 16$ lost to the Euclidean arm, and no experiment separates the radial
    term's contribution from the geometry's.
11. **One selection criterion for several uses.** Every table was selected on Census-phrase
    retrieval. *Breaks:* when a table serves as economic coordinates or as a text-to-code
    reference on owner text, the epoch read is the Census-phrase-best one, not the best for that
    use.

#### Evaluation criteria

- **Selection:** the selection panel's validation MRR, read once per epoch. Today that is S3's.
- **Radius verification** (hyperbolic arm), on the first training batch and the exported table,
  without reading any panel:
  - every anchor's radius takes nonzero gradient
  - the per-level radius SD exceeds $10^{-3}$ at every level
  - the 20 sector radii are positive and distinct
  - the largest-radius point satisfies $|\langle p, p\rangle_{\mathcal{L}} + 1| \le 10^{-9}
    p_{0}^{2}$
  - the single-precision distance agrees with a double-precision evaluation to relative error
    $10^{-3}$ over all ordered pairs
  - all three terms and both scales take finite, positive gradient norms.

  All ten seeds of the original reference passed.
- **Training health,** recorded per epoch: each term's mean, both scales, and the radius mean and
  SD at each level.
- **Structural diagnostics,** reported and never selected on:
  - sector separation, the AUC of distances between same-sector and cross-sector pairs
  - within-sector rank correlation between distance and $D^{\ast}$
  - mean average precision of a code's ancestors
  - NDCG with grades from the depth of the lowest common ancestor.

  Unary pairs are excluded from parent retrieval.
- **Adoption:** only S10, on the panels of the artifact that reads the table.

#### Open questions for the reviewer

1. Retrieval no longer needs a low-dimensional space, and owner descriptions are now the judge.
   Which state-of-the-art design fits short owner descriptions against about 1,000 six-digit codes
   with official descriptions: a full-width bi-encoder, a cross-encoder or language-model reranker
   over bi-encoder candidates, hierarchical decoding, or classification with a large language
   model?
2. Is a multi-target softmax over one level's codes a sound retrieval objective, compared with a
   softmax over the whole codebook or a hierarchical softmax?
3. Is a once-per-epoch cache of candidate points with live anchors an accepted approximation, or
   should it be replaced by a momentum encoder, more frequent refresh, or in-batch negatives?
4. Arms that peak at epoch 1 and arms still improving at epoch 39 share one schedule and budget.
   How should a comparison of architectures control training length?
5. Is there published evidence on when hyperbolic heads help text-conditioned retrieval over a
   shallow tree of about 2,000 nodes, at dimension 8 to 32?

### S3. Census-phrase retrieval panel

**Status: run on real data.** The method is unchanged since the previous description. Its role
changed: it no longer decides anything for text-to-code. It is reported beside S4, and it still
selects S2's checkpoints until S5 moves that selection to S4.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $c^{\ast}_{i}$ | True code of entry $i$ | $\mathcal{C}_{6}$ |
| $\operatorname{rank}_{i}$ | Number of candidates at a distance no greater than the true code's | $1, \dots, 1012$ |
| $\operatorname{lca}(c, c')$ | Level of the lowest common ancestor. 1 is the virtual root | $\{1, \dots, 6\}$ |
| $\operatorname{Jac}$ | Jaccard similarity of two texts' sets of character trigrams | $[0, 1]$ |
| $\operatorname{MRR}^{\mathrm{cen}}$ | This panel's MRR on its validation split | $(0, 1]$ |

#### Problem formulation and data-generating story

The panel takes a short description of an activity and asks for the six-digit code that
classifies it. It measures this on index entries that the trained model never saw: each entry is a
query whose true code is known. The estimand is the expected reciprocal rank of the true code
among all 1,012 six-digit codes, for a new phrase drawn like the held-out entries.

- **Roles.** Every index entry has exactly one role. Within each code, entries are divided among
  examples, training, validation and test in shares 30%, 35%, 20% and 15%, by largest-remainder
  rounding, with at least one examples entry per code. Each code's draw uses its own seeded random
  stream, so a code's roles do not depend on any other code's. The examples role supplies the
  code's examples field (S1), which is how 30% of each code's own index phrases become part of its
  code text.
- **Stratified by code, not grouped.** A validation or test query's code also has training queries
  and an examples field. The panel therefore measures decoding of unseen phrasings to codes seen in
  training. 22 codes with a single entry keep it as examples text and have no queries. 71 codes
  have no validation query and 117 no test query. Two six-digit codes have no index entries. They
  remain candidates and are never queries.
- **Leakage rule.** An entry may become a validation or test query only if it matches nothing in
  the training text, where a match is either:
  - an exact match: the normalized entry occurs as whole words in a segment
  - a near-duplicate: $\operatorname{Jac} \ge 0.9$.

  The segments searched are every code's title, its description and exclusion sentences, the
  activity phrases of cross-references, fallback examples for codes without entries, and every
  other index entry. 4,055 entries were ineligible (1,539 exact, 2,516 near-duplicate only), and
  16,318 eligible. After the draw, no held-out query matched any realized training text, training
  queries included. Five redirection phrases that a held-out query matched were withheld from
  training (S1). Reordered words inside a longer segment, and spans across a sentence break, are
  not caught. An audit found two validation queries inside descriptions, and two validation and
  one test query inside joined examples fields.
- **Frozen.** The role assignment is committed with a fingerprint. Openings of the test split are
  counted against that fingerprint, so a redraw would read as a new, never-opened split. Redrawing
  is refused.

#### Estimation / inference procedure

A read encodes the split's queries through the query map and decodes against the code table
exported from the same checkpoint. Distances are computed in double precision by the arm's own
decoding distance (S2): Euclidean, cosine, or the Lorentz geodesic distance.

$$\operatorname{rank}_{i} = \sum_{c \in \mathcal{C}_{6}} \mathbb{1}\!\left[\operatorname{dist}_{g}(p_{i}, p_{c}) \le \operatorname{dist}_{g}(p_{i}, p_{c^{\ast}_{i}})\right], \qquad \operatorname{RR}_{i} = 1/\operatorname{rank}_{i}, \qquad \operatorname{MRR}^{\mathrm{cen}} = \frac{1}{|\mathcal{E}^{\mathrm{val}}|} \sum_{i \in \mathcal{E}^{\mathrm{val}}} \operatorname{RR}_{i}.$$

- Ties count against the truth: an encoder that places every code alike ranks every truth last.
- Each query has exactly one true code.
- Per query, the read also records top-1 accuracy, Hit@$k$ for $k \in \{1, 5, 10\}$, and
  hierarchical partial credit, $\operatorname{lca}$ of the top-ranked code and the truth. MRR is
  the only statistic that selects.
- Every metric is a query-weighted mean. Entries per code range from 1 to 320, median 13.

**Who reads the validation split.**

- S2's training monitor reads it after every epoch to select the checkpoint. The original
  reference's ten seeds alone made 226 monitor reads.
- S10's shared-space decision read each selected checkpoint once more, before the artifacts were
  split.
- S4's gap reads report it beside the owner-description scores.

**The seal.** The test split may be read only after an explicit, logged opening, and a second
opening requires a stated reason. Every read and opening is appended to an append-only selection
log with the split's fingerprint, the purpose and the reading arm. Each training host keeps its
own log, which holds the training monitor's reads. The analysis machine's log holds the decision
reads. Enforcement is procedural: roles are visible in the committed files, and the panel is the
only path through which later stages read queries. On 2026-11-19 the analysis machine's log held
284 validation reads over all panels, 56 of them of this panel, and no opening of any sealed
split. No opening of this test split is planned: it would serve only a report, and a sealed split
spent on a report can judge nothing later.

**Comparator.** The only non-trained comparator is a lexical baseline. A code's text (title,
description and examples field) and the query are hashed into 4,096 character-trigram features,
normalized to unit length, and compared by cosine distance. On the validation split it scores MRR
0.6165, top-1 0.516, Hit@5 0.733, Hit@10 0.797 and mean partial-credit level 4.34. It gave the
same values on the current code texts in 2026-11 as on the code texts of 2026-09, before the
supervision was regenerated.

**Results on record.** Mean validation MRR is 0.413 for the original reference over ten seeds and
0.592 for the Euclidean $d = 16$ arm (S2's table), against 0.6165 for the lexical baseline.

#### Assumptions and limitations

1. **Census phrasing is easier than owners' phrasing.** *Breaks if violated:* this is now
   measured, not assumed. Both reference arms score 0.14 to 0.18 MRR lower on owners' own words
   (S4), so this panel overstates field performance.
2. **Seen codes only.** The stratified split measures new phrasings of codes that have training
   queries. *Breaks:* nothing measures a code with no training phrases, such as a code new in a
   later vintage.
3. **The eligibility filter selects distinctive phrases.** An entry that occurs inside any other
   text cannot be held out. *Breaks:* generic or ambiguous phrasings, the hard cases in practice,
   are under-represented in the held-out splits, and the direction of the bias in MRR is unknown.
4. **One true code per phrase.** *Breaks:* an activity that legitimately fits several codes is
   scored as a miss whenever the listed code is not ranked first.
5. **Code text contains same-code phrases.** The examples field holds 30% of each code's own index
   entries. *Breaks:* the panel partly measures phrase-to-phrase similarity within a code. That
   likely explains why the trigram baseline beats every trained arm here, although the Euclidean
   $d = 16$ arm beats it on S4's headline.
6. **Query weighting.** Codes with many entries dominate the MRR. *Breaks:* rare codes, which are
   most of the codebook by count, carry little weight.
7. **It still selects.** The panel decides nothing for text-to-code but still selects S2's
   checkpoints. *Breaks:* every trained table on record is the Census-phrase-best epoch, which is
   not the epoch that S4 or S6 would choose.

#### Evaluation criteria

The panel's own adequacy is judged by:

- leakage audits of held-out text against all training text, by exact match and by trigram
  near-duplication
- the frozen, fingerprinted split, with append-only read logging
- a non-trained floor, the lexical baseline, scored on the same split.

#### Open questions for the reviewer

1. Now that it decides nothing, should this panel keep any role: a cheap training monitor, a
   diagnostic of the gap to owner text, or none?
2. Should the leakage rule also exclude semantic paraphrases, not only exact and trigram near
   duplicates?
3. Should a grouped split hold out whole codes to measure zero-shot decoding?

### S4. Owner-description retrieval panel

**Status: run on real data for its validation part.** The two reference arms were read on it on
2026-11-19, to measure the gap between Census phrases and owners' words. The margin record, the
adoption test and the sealed opening are built and tested on fixtures only. No candidate has been
read, nothing has been selected, and the sealed part is closed.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\mathcal{O}$ | Panel pairs: distinct pairs of an employer and a normalized description, kept when the employer is known, the description is neither empty nor a placeholder, and at least one record carries a code native to NAICS 2022, filed for 2023–2025 | |
| $i$ | A panel pair | $\mathcal{O}$ |
| $\operatorname{nd}(i)$ | Normalized description of $i$: Unicode-normalized, case-folded, whitespace collapsed and trimmed, punctuation kept | text |
| $q_{i}$ | Query of $i$: the most frequent original string among its records | text |
| $T_{i}$ | Reported codes of $i$: the distinct six-digit codes on its eligible records | non-empty subset of $\mathcal{C}_{6}$ |
| $\operatorname{cp}(i)$ | Copied indicator: 1 when $\operatorname{nd}(i)$ equals the normalized form of an official title at levels 2–6 or an index entry, of the 2012, 2017 or 2022 classification | $\{0, 1\}$ |
| $\mathfrak{h}$ | A fixed salted cryptographic hash of a normalized description, read as an unsigned 64-bit integer | |
| $\mathcal{O}^{\mathrm{val}}$, $\mathcal{O}^{\mathrm{sea}}$ | Validation part and sealed part | partition of $\mathcal{O}$ |
| $\operatorname{anc}_{l}(c)$ | Level-$l$ ancestor of $c$, which is $c$ itself at $l = 6$ | $\mathcal{C}_{l}$ |
| $\mathcal{T}^{(l)}_{i}$ | Matching codes of $i$ at level $l$: the six-digit codes $c'$ with $\operatorname{anc}_{l}(c') \in \operatorname{anc}_{l}(T_{i})$, so $\mathcal{T}^{(6)}_{i} = T_{i}$ | subset of $\mathcal{C}_{6}$ |
| $\operatorname{rank}^{(l)}_{i}$, $\operatorname{rank}_{i}$ | Shortlist rank of $i$'s nearest matching code at level $l$, and its value at level 6 | $1, \dots, 1012$ |
| $\operatorname{MRR}^{\mathrm{own}}$ | Headline: mean reciprocal rank over a part's non-copied pairs, each pair once | $(0, 1]$ |
| $\operatorname{MRR}^{\mathrm{cp}}$ | Mean reciprocal rank over a part's copied pairs | $(0, 1]$ |
| $\operatorname{MRR}^{\mathrm{phr}}$ | Phrasing report: the mean over normalized descriptions of the mean reciprocal rank of their non-copied pairs | $(0, 1]$ |
| $A^{\mathrm{lex}}$, $A^{\mathrm{E16}}$ | The two reference arms: S3's lexical baseline and S2's Euclidean $d = 16$ arm | |
| $r^{\mathrm{t2c}}$ | Text-to-code's relative margin | 0.05 |
| $\delta^{\mathrm{t2c}}$ | Text-to-code's margin: $r^{\mathrm{t2c}}$ times the chosen reference's validation headline MRR | MRR units |

#### Problem formulation and data-generating story

An owner describes an establishment in their own words and asks for its code. Text-to-code answers
with a ranked shortlist of six-digit codes, and the code's ancestors follow from the codebook. Owner
descriptions are often coarser than codes, so one answer is unreliable and a shortlist is the
output. Subsidiaries and whole firms, which can span industries, are out of scope.

The panel measures this on workplace-injury filings. An employer files one summary per
establishment and year, with a free-text industry description and a self-reported code. The data
builder groups records into pairs of employer and normalized description, pooled over all ten
filing years, 2016–2025. A pair's query is its most frequent original string. Its reported codes
come only from its records with a code native to NAICS 2022, filed for 2023–2025. When those
records carry several codes, every one is kept and the pair is flagged as a conflict.

The estimand is the expected reciprocal rank of an owner's reported code, for an owner drawn like
the non-copied panel pairs, each pair counted once. The owner chose that unit: owners, not distinct
phrasings, are the headline population. Copied pairs are scored and reported apart, because they
reward recall of the classification's own text.

The text repeats heavily. The 3.2 million records hold 49,575 distinct original strings and 29,067
distinct normalized descriptions (reported by the data builder). OSHA's own guide states that the
form collects a text description that is often one of the classification's index titles.

#### Estimation / inference procedure

**Inputs and refusals.** The panel reads the builder's pair table and pair-code table, and the
codebook. Before any read, it refuses a code of another classification, vintage or level, a code
that is not one of the 1,012 six-digit candidates, and a panel pair with no reported code. A
refusal, not a silent drop, signals that the inputs disagree.

**The split.** The unit is the normalized description:
$$\operatorname{part}(i) = \mathfrak{h}\bigl(\operatorname{nd}(i)\bigr) \bmod 2, \qquad 0 \mapsto \mathcal{O}^{\mathrm{val}}, \quad 1 \mapsto \mathcal{O}^{\mathrm{sea}}.$$

- Every pair with one normalized description falls in one part. One employer can still appear in
  both parts under different descriptions.
- The salt is fixed once and never changes. The rule depends on the description alone, so a rebuilt
  table puts each old description in its old part and each new one in a part of its own, with no
  stored assignment.
- The split's fingerprint is a hash of the rule itself: salt, rule version, hash, modulus and part
  names. It identifies the split, not the rows. Openings of the sealed part are counted against it,
  so a rebuilt table cannot pass as a fresh sealed part.

**Scoring.** The candidates are the 1,012 six-digit codes, ranked for $q_{i}$ by the encoder's own
distance. Any matching code counts, and ties with non-matching codes count against the truth, as
in S3. At level $l$, the matching codes are those that share a reported code's level-$l$
ancestor, and the rank is one plus the non-matching candidates at a distance no greater than the
nearest matching one:
$$\operatorname{rank}^{(l)}_{i} = 1 + \sum_{c' \in \mathcal{C}_{6} \setminus \mathcal{T}^{(l)}_{i}} \mathbb{1}\!\left[\operatorname{dist}(p_{q_{i}}, p_{c'}) \le \min_{c \in \mathcal{T}^{(l)}_{i}} \operatorname{dist}(p_{q_{i}}, p_{c})\right], \qquad \operatorname{rank}_{i} = \operatorname{rank}^{(6)}_{i}, \qquad \operatorname{RR}_{i} = 1/\operatorname{rank}_{i}.$$
A pair's other reported codes never count against it, even when they tie with its nearest one.

- **Per pair:** reciprocal rank, Hit@1, Hit@5 and Hit@10. For the coarser levels, the highest level
  of the lowest common ancestor between the top-ranked code and any reported code, and
  $\operatorname{rank}^{(l)}_{i}$ for each $l \in \{2, \dots, 5\}$, with its reciprocal rank and
  hits. This reads the shortlist that an owner who only knows their sector would see.
- **Headline:** the mean reciprocal rank over the part's non-copied pairs,
  $$\operatorname{MRR}^{\mathrm{own}} = \operatorname{mean}\bigl\{\operatorname{RR}_{i} : i \in \text{the part},\ \operatorname{cp}(i) = 0\bigr\}.$$
- **Reports, which never decide:** $\operatorname{MRR}^{\mathrm{cp}}$ over copied pairs, the
  phrasing report $\operatorname{MRR}^{\mathrm{phr}}$, and counts of pairs, descriptions, copied
  pairs, conflict pairs, and pairs whose employer is known only by company name or whose text shows
  encoding damage. The flags in the counts change no score.

**Reads and the seal.** Every read appends a record to the shared selection log before any query
is encoded. The record names the panel, the part, the split fingerprint, the arm, its seed, its
role (reference or candidate), a run identifier, and a fingerprint of the scored rows. Reads of the
sealed part fail until an explicit, logged opening. A second opening needs a stated, logged reason.

**References.** Two arms, chosen by the owner before any ITA read:

- $A^{\mathrm{lex}}$, S3's lexical baseline, with no seeds
- $A^{\mathrm{E16}}$, S2's Euclidean $d = 16$ arm, with its five recorded checkpoints, which S3
  selected. It is not selected again on this panel: a reference re-selected on the judge would be a
  candidate.

**Margin and test** (built on fixtures, not yet run). The margin read makes fresh reads of both
references on $\mathcal{O}^{\mathrm{val}}$. The reference is the arm with the higher headline MRR
as a point estimate, with $A^{\mathrm{E16}}$'s MRR the mean over its seeds, and a tie going to
$A^{\mathrm{lex}}$. Then $\delta^{\mathrm{t2c}} = r^{\mathrm{t2c}} \cdot
\operatorname{MRR}^{\mathrm{own}}(\text{reference})$. A margin record is refused when the log holds
any earlier read of this panel with the role candidate, or with no role. Training-monitor reads
count as candidate reads, so the margin must exist before any candidate trains against this panel.
S10 gives the test, which requires a candidate to beat the reference by $\delta^{\mathrm{t2c}}$.

**Which read decides.** The validation part chooses at most one forward candidate. A candidate
that does not beat the reference by $\delta^{\mathrm{t2c}}$ on validation does not go forward, and
the reference stands. The sealed part is opened once, after that choice, and the same test runs on
it with the validation $\delta^{\mathrm{t2c}}$. If the candidate fails there, the reference stands.
S3's test split is not opened.

**Results on record: the gap reads** (2026-11-19). The validation part holds 125,616 panel pairs in
9,657 normalized descriptions. 117,108 pairs are copied. 8,508 pairs, in 3,768 descriptions, are
headline pairs. 227 are conflict pairs, and 16,506 have an employer known only by company name.

| Arm | Report | Pairs or descriptions | MRR | Top-1 | Hit@5 | Hit@10 | MRR at level 2 | MRR at level 4 |
|---|---|---|---|---|---|---|---|---|
| Lexical baseline | headline | 8,508 | 0.4322 | 0.3606 | 0.4999 | 0.5549 | 0.6957 | 0.4915 |
| Euclidean $d = 16$ | headline | 8,508 | 0.4498 | 0.3377 | 0.5751 | 0.6660 | 0.7958 | 0.5336 |
| Lexical baseline | copied | 117,108 | 0.7448 | 0.6552 | 0.8490 | 0.8911 | 0.9258 | 0.8053 |
| Euclidean $d = 16$ | copied | 117,108 | 0.7223 | 0.5887 | 0.8950 | 0.9388 | 0.9558 | 0.8417 |
| Lexical baseline | phrasing | 3,768 | 0.4041 | 0.3240 | 0.4900 | 0.5448 | 0.7059 | 0.4762 |
| Euclidean $d = 16$ | phrasing | 3,768 | 0.3862 | 0.2813 | 0.5037 | 0.5837 | 0.7324 | 0.4900 |

The Euclidean arm's seeds span headline MRR 0.4422 to 0.4578.

The gap compares $\operatorname{MRR}^{\mathrm{cen}}$ (S3's validation split), the headline
$\operatorname{MRR}^{\mathrm{own}}$ and the copied $\operatorname{MRR}^{\mathrm{cp}}$. Each
interval is a 95% percentile interval from 10,000 replicates of a two-stage bootstrap. Each panel
resamples its own units, description groups here and codes in S3, and one draw of seeds per arm
serves both. The code-matched gap restricts both panels to the 689 codes present in both, and gives
each code one weight. 41 codes have headline pairs but no S3 validation query.

| Arm | $\operatorname{MRR}^{\mathrm{cen}} - \operatorname{MRR}^{\mathrm{own}}$ | Code-matched | $\operatorname{MRR}^{\mathrm{cen}} - \operatorname{MRR}^{\mathrm{cp}}$ | $\operatorname{MRR}^{\mathrm{cp}} - \operatorname{MRR}^{\mathrm{own}}$ |
|---|---|---|---|---|
| Lexical baseline | 0.184 [0.113, 0.253] | 0.151 [0.118, 0.185] | −0.128 [−0.181, −0.076] | 0.313 [0.231, 0.392] |
| Euclidean $d = 16$ | 0.143 [0.081, 0.202] | 0.163 [0.136, 0.191] | −0.130 [−0.180, −0.079] | 0.273 [0.201, 0.344] |

- Owners' own words are harder than held-out Census phrases for both arms, also within the same
  codes. A copied phrase is easier than a held-out one, so the copied pairs would flatter any arm.
- A pre-registered rule asked whether to recommend a later training part drawn from these pairs:
  yes when, for both arms, the lower bound of the code-matched gap exceeds 0.05 times that arm's
  $\operatorname{MRR}^{\mathrm{cen}}$. Both lower bounds (0.118 and 0.136) exceed their thresholds
  (0.031 and 0.030), so the finding recommends one. That part needs its own design and the data
  owner's permission, and nothing about it is built.
- *Projection, not a recorded value:* if the margin read reproduces these headline values, the
  Euclidean arm becomes the reference, and $\delta^{\mathrm{t2c}} \approx 0.05 \times 0.450 =
  0.022$.

#### Assumptions and limitations

1. **Self-reported codes are truth** (cross-component assumption 6). Conflict pairs count as a hit
   when any reported code ranks well. *Breaks if violated:* misreported codes cap every arm's
   score, and the any-code rule is lenient for the 227 conflict pairs.
2. **Exact copies only.** The copied indicator catches exact normalized identity with a title or an
   index entry. *Breaks:* a lightly edited index title counts as the owner's own words, which
   inflates the headline toward the copied score.
3. **A small, selected headline population.** Only 6.8% of the validation part's pairs are
   headline pairs. *Breaks:* owners who did not copy may differ systematically from those who did,
   for example in industry or in wording, so the headline describes a minority of filers, and its
   precision rests on 3,768 descriptions.
4. **Owners as the unit.** Each pair counts once. *Breaks:* a description used by many employers
   counts many times, so common descriptions dominate the headline. The phrasing report, which
   gives each description one weight, ranks the two references the other way round.
5. **Filing rules shape coverage.** OSHA's filing rules decide which establishments file. *Breaks:*
   some industries are thin or missing, and the panel's code mix differs from S3's. The
   code-matched gap corrects the comparison with S3, not the coverage.
6. **Text and code from different years.** A pair's text is the most frequent string over
   2016–2025, while its headline codes come from 2023–2025. *Breaks:* an establishment whose
   activity or wording changed contributes a query and a code from different times.
7. **The reference keeps its Census-selected checkpoints.** Candidates will be selected on this
   panel's validation part, and the Euclidean reference was not. *Breaks:* a candidate's
   validation gain over the reference includes selection optimism. Only the sealed part, read once,
   removes it.
8. **One part for every development choice** (S5). *Breaks:* the reference choice, the margin,
   every checkpoint and the forward candidate are all chosen on $\mathcal{O}^{\mathrm{val}}$.

#### Evaluation criteria

- Each normalized description falls in exactly one part, and a rebuilt table keeps every old
  description in its old part.
- Codes of another vintage or level, and codes outside the candidates, are refused before any read.
- The headline excludes copied pairs and counts each pair once.
- Every read is logged before it scores, and the sealed part opens once, keyed to the split rule.
- A non-trained floor, the lexical baseline, is scored on the same part.

#### Open questions for the reviewer

1. Is MRR over six-digit codes the right headline for self-reported owner text that is often
   coarser than a code? Should the reported coarser-level matches carry weight in the decision?
2. Should near-copies of titles and index entries, found by fuzzy or semantic matching, also leave
   the headline? Would enough pairs remain?
3. Owners or phrasings: which population should the headline stand for, given that the two
   weightings rank the references differently?
4. What do published industry autocoders report on employer- or respondent-written text, and is
   there a public benchmark this panel could be compared with?
5. Is a margin of about 0.022 MRR on 8,508 pairs in 3,768 descriptions a test with adequate power
   for the candidates S5 will bring?

### S5. Text-to-code candidate search

**Status: not yet specified.** Only an outline exists. Its design session has not run, so every
choice below is a planned direction, not a fixed procedure.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\mathcal{X}$ | The set of text-to-code candidates: encoder configurations, each trained on several seeds | |
| $X$ | A candidate | $\mathcal{X}$ |
| $\hat\theta_{X,k}$ | Checkpoint of seed $k$ of candidate $X$, selected on S4's validation headline | |
| $X^{\star}$ | The forward candidate: the one candidate taken to S4's sealed part, if any | $\mathcal{X}$ |

#### Problem formulation and data-generating story

The question is whether any text encoder beats text-to-code's reference on owners' own words by
the margin $\delta^{\mathrm{t2c}}$. Training uses the classification's own text only: titles,
descriptions, examples, exclusions, index entries and cross-reference phrases, as in S1 and S2. No
owner-description pair enters training. The estimand is the forward candidate's headline gain over
the reference on S4's sealed part.

#### Estimation / inference procedure

The outline fixes this order:

1. **Margin first.** Record $\delta^{\mathrm{t2c}}$ from fresh reference reads (S4) before any
   candidate trains or is read.
2. **Candidates.** The outline names these factors, each to be varied from one base configuration:
   - the backbone: the current one, at least two current general-purpose embedding models, and a
     frozen-encoder control with no adapters
   - fusion by the mixture of experts, against the masked mean
   - an indicator of which fields are present, fed to the encoder
   - an information-content code–code target in place of the edge-count target
   - the weights of the code–code and radial terms.

   Each backbone reads its own input window and, where a text exceeds it, its own window-fitting
   summaries. What happens to training targets above six digits is still to be decided.
3. **Selection.** Each candidate trains on at least five seeds. Each seed keeps the earliest epoch
   with the highest S4 validation headline MRR, which replaces S3 as S2's selection panel for
   these runs. How the forward candidate is chosen among candidates is still to be decided.
4. **Validation test.** $X^{\star}$ goes forward only if it beats the reference by
   $\delta^{\mathrm{t2c}}$ on the validation part (S10).
5. **Sealed opening.** S4's sealed part is opened once, and the same test decides.

A later training part drawn from owner descriptions is recommended (S4) but belongs to a later
design, after the data owner's permission.

#### Assumptions and limitations

1. **One factor at a time.** *Breaks if violated:* factors that interact, such as a backbone and
   its window or fusion, are compared only at the base configuration's other values.
2. **One validation part for every choice.** *Breaks:* many candidates, seeds and epochs selected
   on 3,768 descriptions inflate the forward candidate's validation gain. The sealed read removes
   the bias for the one candidate it tests, at the cost of spending the sealed part.
3. **No owner text in training.** *Breaks:* the gap of S4 is attacked only through the encoder's
   generalization, not through training on the target distribution.
4. **The family is S2's.** The listed factors all vary S2's bi-encoder. *Breaks:* designs outside
   that family, such as rerankers or language-model classifiers, are not on the outline.

#### Evaluation criteria

- The margin record precedes every candidate read, enforced in the log.
- Every selection read is a read of S4's validation part.
- Each candidate has at least five seeds, and the forward candidate's test uses S10's rule.

#### Open questions for the reviewer

1. Which candidate families should the search cover first, given a reference near MRR 0.45 on
   owner text: full-width bi-encoders, rerankers over a shortlist, hierarchical decoding, or
   language-model classification?
2. Is a one-factor-at-a-time search defensible here, against a small factorial design or a
   sequential search with a held-back confirmation?
3. How should selection optimism from many choices on one validation part be bounded before the
   single sealed read?

### S6. Growth regressor panel

**Status: run on real data.** The method is unchanged since the previous description. Since then,
all 18 economic candidates have been read on its growth regimes (S7), and the economic decision is
recorded (S10).

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $E_{c,t}$ | Annual-average private employment of code $c$ in year $t$, national | persons |
| $t$ | Feature year. The outcome year is $t + 1$ | 2022, 2023, 2024 |
| $w_{c,t}$ | Covariates: log establishments and log total annual wages, both of year $t$ | $\mathbb{R}^{2}$ |
| $y_{c,t}$ | Outcome: $\ln E_{c,t+1} - \ln E_{c,t}$ in growth mode, $\ln E_{c,t+1}$ in level mode | log ratio, log persons |
| $f(c)$ | Representation block of code $c$: a candidate's coordinates $z'_{c}$, one-hot indicators, ancestor indicators, or frozen-text coordinates | |
| $\operatorname{grp}(c)$ | Four-digit group of code $c$, the unit of folds and resampling | 300 groups |
| $\mathcal{H}$ | Held-out groups | 60 of 300 |
| $\kappa$ | Ridge penalty | 17 values, $10^{-3}$ to $10^{5}$ |
| $\beta_{0}$, $\beta_{w}$, $\beta_{f}$ | Intercept and coefficients of the downstream model | |
| $\varepsilon_{c,t}$ | Error of row $(c, t)$ | log units |
| $\chi_{i}$ | Standardized feature row $i$: covariates and representation block, centered and scaled on the fit rows | |

#### Problem formulation and data-generating story

The economic artifact's purpose is to replace sparse industry indicators as regressors. The panel
asks how much out-of-sample error a downstream model makes when a code's representation joins two
covariates:

$$y_{c,t} = \beta_{0} + w_{c,t}^{\top}\beta_{w} + f(c)^{\top}\beta_{f} + \varepsilon_{c,t},$$

fitted by ridge regression. Representations are compared within one regime at a time:

- **Seen regime.** Every code appears in the fit, and prediction moves forward in time: fit on one
  year's transitions, predict the next. One-hot indicators are a real competitor here.
- **Held-out regime.** Whole four-digit groups are absent from the fit, so a representation must
  carry information across codes. One-hot indicators can predict only the intercept here.

Growth mode is the economic outcome. Level mode, $\ln E_{c,t+1}$, is history: it served the
shared-space decision and is dominated by persistence.

**Data.** Public QCEW annual averages for 2022 through 2025: national grain, private ownership,
all establishment sizes, on the 2022 classification. The files are pinned by hash.

- A code enters the population when it has a disclosed cell with positive employment,
  establishments and wages in all four years: 980 of 1,012 six-digit codes. The 32 excluded are 29
  public-administration codes with no private cell and 3 codes the publisher does not use.
- 19 construction codes, published only as residential and nonresidential splits, take their
  five-digit parent's cell. Each is its parent's only child.
- Rows are code-years: $(c, t)$ for $t \in \{2022, 2023, 2024\}$, giving 2,940 six-digit rows.
  Levels 2 to 5 are built the same way for a multi-level variant, and decisions read level 6.

**Partition.** A fifth of each sector's four-digit groups, 60 of 300, is held out by a seeded,
committed draw.

- Held-out outer set: every row of the 203 codes $c$ with $\operatorname{grp}(c) \in \mathcal{H}$,
  609 rows.
- Seen outer set: the remaining 777 codes' 2024→2025 rows.
- Remainder: those codes' 2022 and 2023 rows, 1,554 rows in 240 groups. Validation reads only the
  remainder.

The outer sets are sealed. 2025 employment enters only outer rows, so one opening of a regime
counts for both outcomes. The economic decision adopted no candidate, so no opening is planned, and
both outer sets stay sealed for a later economic round.

#### Estimation / inference procedure

**Ridge.** Features are standardized on the fit rows (population SD, and a constant column gets
scale 1). The intercept is the fit rows' mean outcome and is not penalized, and every standardized
column takes the same penalty $\kappa$, covariates included. Over the fit rows $i$, with $\bar y$
the fit rows' mean outcome, the fit minimizes
$$\sum_{i} \bigl(y_{i} - \bar y - \chi_{i}^{\top}\beta\bigr)^{2} + \kappa\Vert\beta\Vert^{2}.$$
$\kappa$ minimizes the squared error summed over the task's tuning pairs, with ties going to the
larger penalty.

**Validation plans.** Folds are grouped by $\operatorname{grp}(c)$: five folds, five repeats. They
depend only on regime, level, repeat and group, never on the representation.

- **Seen regime.** Fit on all remainder 2022 rows and score the remainder 2023 rows. The 2023 rows
  are dealt into five grouped folds, and each fold's penalty is tuned on the other folds' 2023
  rows. The fit set is the same for every fold, so repeats only redraw which rows choose the
  penalty. 777 rows scored.
- **Held-out regime.** All remainder rows, both years, are dealt into five grouped folds. Fit on
  four folds and score the fifth, tuning the penalty by five inner grouped folds inside the fit
  rows. 1,554 rows scored. Time order across rows is not enforced: fit rows mix 2022 and 2023 rows
  of other groups.
- **Outer reads** (sealed, never run): fit on the whole remainder, then score the outer set.

**Comparators.** The covariates alone, and each representation alone and with the covariates:

- **candidate coordinates:** $z'_{c}$, the same $d$ values in every year of a code
- **one-hot indicators:** seen regime only
- **ancestor indicators:** one indicator block per ancestor level, 2 to 5
- **frozen-text coordinates:** $\operatorname{PCA}_{d}(\bar h^{\circ}_{c})$ at the candidate's
  $d$. One frozen-text table serves every candidate, so this comparator is the same in every read
  and decides nothing.

Indicator columns are built from the fit frame. A code's own level-4 and level-5 ancestor columns
are therefore all zero on the fit rows of the held-out regime. There, ancestors help only through
levels 2 and 3.

**Statistic.** Each scored row's squared error is averaged over its five repeats. The panel
statistic is the mean over rows. A candidate's statistic is that of the covariates plus its
coordinates. The economic reference's is that of the covariates plus ancestor indicators.

**Results on record.**

- **Level mode** (the shared-space decision). The original reference arm's covariates-plus-embedding
  MSE was 0.0848 in the seen regime and 0.0950 in the held-out regime. S2's table gives every
  arm's. Descriptive fits recorded when the panel was built show what this outcome rewards. With
  covariates alone, $R^{2}$ is 0.933 in both regimes. One-hot indicators reach 0.998 in the seen
  regime. Covariates plus ancestors reach 0.993 seen and 0.956 held out.
- **Growth mode.** The reference, covariates plus ancestor indicators, has validation MSE 0.003187
  in the seen regime and 0.003239 in the held-out regime, a root mean squared error of about 5.6
  log points. These two numbers fixed the economic margins (S10). The 18 candidates' errors are in
  S7.

#### Assumptions and limitations

1. **National six-digit employment growth represents the downstream uses.** The ranked uses are
   employment growth, joint non-response with employment growth, wages, shock exposure and
   nowcasting. Only the first has a panel. *Breaks if violated:* a representation that wins or
   loses here may behave differently in models of other outcomes or at a sub-national grain.
2. **A short window.** Validation scores two transitions, 2022→2023 and 2023→2024, and the seen
   outer set one more. *Breaks:* post-pandemic dynamics specific to those years can decide which
   representation wins.
3. **No time order in the held-out regime.** *Breaks:* common year shocks are learned from other
   groups' same-year rows, so held-out errors are optimistic for a forecasting use.
4. **One penalty for all columns.** Covariates, dense coordinates and indicator blocks share
   $\kappa$ after standardization. *Breaks:* a representation's apparent value depends on how its
   columns' scale interacts with that shared penalty. Sixteen dense columns and several hundred
   sparse indicator columns are shrunk alike.
5. **Unweighted rows.** Each code counts equally whatever its size. *Breaks:* uses that aggregate
   employment weight large industries, which this statistic does not.
6. **Linear use of coordinates.** *Breaks:* a representation useful through interactions or
   nonlinear terms is undervalued.
7. **Current size is not a covariate.** $\ln E_{c,t}$ is not a regressor in growth mode. Size
   enters only through establishments and wages. *Breaks:* size-dependent growth is credited to
   whichever representation best proxies size.
8. **A complete population.** No cell is suppressed at the national grain. *Breaks:* nothing here
   speaks to the suppressed cells that dominate sub-national use.
9. **Replacement, not complement.** A candidate's coordinates replace the indicators. Coordinates
   added to the indicators are never scored. *Breaks:* coordinates that help only beside the
   indicators are judged useless.

#### Evaluation criteria

- Hash-pinned inputs and a committed, fingerprinted held-out draw.
- A fixed comparator set sharing one estimator and penalty grid.
- Rows identical across reads, and reference predictions equal to within the pairing tolerance,
  which S10's pairing requires.
- Sealed outer sets with append-only opening records.

#### Open questions for the reviewer

1. Is ridge regression of next-year national employment growth on two covariates plus a
   $d$-dimensional code vector a credible test of economic coordinates? How do published studies
   evaluate industry embeddings or text-based industry measures as regressors, and on which
   outcomes?
2. Should the panel weight rows by employment, add year effects, add current size, or move to a
   sub-national grain?
3. Do two validation transitions carry enough signal to separate candidates whose errors differ by
   a few percent?
4. Is ancestor indicators under one shared ridge penalty the right bar? The classical alternative
   to sparse indicators is partial pooling over the tree.

### S7. Economic candidates

**Status: run on real data.** All 18 candidates were built or imported, read on both growth
regimes, and decided on 2026-11-18. None is adopted, so there is no economic artifact. The economic
chain trains nothing more.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $Z^{\mathrm{txt}}_{d}$ | Frozen-text candidate: $z'_{c} = \operatorname{PCA}_{d}(\bar h^{\circ}_{c})$ | $2125 \times d$ |
| $Z^{\mathrm{tree}}_{d}$ | Tree candidate: classical scaling of $D^{\ast}$ to $d$ dimensions | $2125 \times d$ |
| $Z^{\mathrm{sm}}_{d}$ | Smoothed candidate: frozen text vectors averaged over each code's lineage, then reduced to $d$ | $2125 \times d$ |
| $Z^{g,d}$ | Trained candidate: S2's arm $g$ at dimension $d$, one table per seed | $2125 \times d$ per seed |
| $\operatorname{anc}_{l}(c)$ | Level-$l$ ancestor of $c$ (as in S4) | $\mathcal{C}_{l}$ |
| $\Omega$ | Doubly centered matrix of squared tree distances | $2125 \times 2125$ |
| $\Lambda_{m}$, $\mathbf{e}_{m}$ | The $m$-th largest eigenvalue of $\Omega$, and its unit eigenvector | $\mathbb{R}$, $\mathbb{R}^{2125}$ |
| $\tilde h_{c}$ | Lineage-smoothed frozen text vector of $c$ | $\mathbb{R}^{384}$ |

#### Problem formulation and data-generating story

Does any table of $d$ coordinates per code, built from the classification's text and tree alone,
predict employment growth better than the tree's ancestor indicators? The candidates are fixed in
advance, and no candidate reads employment data or any panel. The estimand is each candidate's
oriented difference from the reference on S6's two growth regimes, and the decision of S10's
economic rule over all of them.

A second question rides on the same reads: does propagating features over the tree help? It is
answered by comparing each smoothed candidate with its unsmoothed twin, as a report that chooses
nothing.

#### Estimation / inference procedure

**Deterministic builders.** Each reads the codebook and, for text, the one shared frozen-text
table. Each writes a 2,125-code table. Two builds write the same bytes.

- **Frozen text.** $Z^{\mathrm{txt}}_{d} = \operatorname{PCA}_{d}(\bar h^{\circ})$, with the
  principal components fitted on all 2,125 codes. This is the frozen-text comparator of S6 promoted
  to candidate.
- **Tree.** Classical scaling of the tree metric, in double precision:
  $$\Omega = -\tfrac12 \Bigl(I - \tfrac{1}{N}\mathbf{1}\mathbf{1}^{\top}\Bigr)\bigl(D^{\ast} \circ D^{\ast}\bigr)\Bigl(I - \tfrac{1}{N}\mathbf{1}\mathbf{1}^{\top}\Bigr), \qquad z'_{c} = \bigl(\sqrt{\Lambda_{1}}\, \mathbf{e}_{1,c}, \dots, \sqrt{\Lambda_{d}}\, \mathbf{e}_{d,c}\bigr),$$
  where $\circ$ is the elementwise product. Each column's sign is fixed so that its entry of
  largest magnitude is positive. The builder refuses a non-positive $\Lambda_{d}$, and a
  relative gap below $10^{-6}$ between $\Lambda_{d}$ and $\Lambda_{d+1}$. $D^{\ast}$ is not
  Euclidean: 424 eigenvalues of $\Omega$ are negative. The eigenvalues at the cuts are 1,779.1
  against 1,631.5 ($d = 8$), 970.08 against 961.94 ($d = 16$) and 398.63 against 396.16
  ($d = 32$).
- **Smoothed.** Each code's frozen text vector is replaced by a weighted mean over its lineage,
  from its sector down to itself, with weight halving at each step up:
  $$\tilde h_{c} = \frac{\sum_{l=2}^{\lambda(c)} 2^{-(\lambda(c) - l)}\, \bar h^{\circ}_{\operatorname{anc}_{l}(c)}}{\sum_{l=2}^{\lambda(c)} 2^{-(\lambda(c) - l)}}, \qquad Z^{\mathrm{sm}}_{d} = \operatorname{PCA}_{d}(\tilde h).$$
  The weights are fixed in the design and never tuned.

**Trained candidates.** The nine geometry × dimension cells of S2, read as they are. Each seed's
table and its checkpoint's selection evidence are imported from the records of S2's campaigns:
ten seeds for the original reference cell, five for each other cell. Every checkpoint was selected
on S3. Each seed is read once on each growth regime, and seeds are resampled within the decision
(S10).

**The frozen list.** The 18 candidates were fixed before the first read, and the decision requires
exactly them. Their component counts, which the tie order reads, are 1 for every candidate except
the smoothed ones, which count 2.

**Text and tree only, enforced.** A candidate's provenance may name only the codebook, its
descriptions and summaries, the backbone and its revision, a text checkpoint trained on this
supervision, and the shared frozen-text table. A provenance that names employment statistics, a
panel, the selection log or an unknown input is refused.

**Results on record** (mean squared error × $10^{3}$, validation, mean over a trained candidate's
seeds). The reference's are 3.187 (seen) and 3.239 (held out).

| Candidate | $d = 8$ seen | $d = 8$ held out | $d = 16$ seen | $d = 16$ held out | $d = 32$ seen | $d = 32$ held out |
|---|---|---|---|---|---|---|
| Frozen text | 3.675 | 3.199 | 3.664 | 3.216 | 3.552 | 3.122 |
| Tree | 3.696 | 3.320 | 3.640 | 3.310 | 3.609 | 3.339 |
| Smoothed | 3.635 | 3.181 | 3.656 | 3.235 | 3.515 | 3.143 |
| Trained, Euclidean | 3.690 | 3.295 | 3.643 | 3.226 | 3.603 | 3.163 |
| Trained, spherical | 3.666 | 3.246 | 3.651 | 3.239 | 3.620 | 3.180 |
| Trained, hyperbolic | 3.660 | 3.201 | 3.627 | 3.181 | 3.596 | 3.163 |

- **Seen regime.** Every candidate is worse than the reference, by 10.3% to 16.0% of its error.
- **Held-out regime.** Candidates range from 3.6% better to 3.1% worse than the reference.
- **Decision** (S10). None of the 18 is non-inferior on both regimes, so none is eligible and the
  reference stands. The largest held-out gain is the frozen-text candidate's at $d = 32$, and it
  fails non-inferiority in the seen regime.
- **Propagation.** At each $d$, the smoothed candidate is non-inferior to its frozen-text twin on
  both regimes and superior on neither. There is no evidence that smoothing over the lineage helps.

The reads appended 118 validation reads to the selection log, and no outer set opened.

#### Assumptions and limitations

1. **Components fitted on every code.** The principal components use all 2,125 codes, including
   the held-out groups. They read no outcome. *Breaks if violated:* the held-out codes shape the
   axes that their own coordinates lie on, which is a mild form of transduction.
2. **Classical scaling of a non-Euclidean metric.** Negative eigenvalues are dropped. *Breaks:* the
   coordinates distort $D^{\ast}$. At $d = 16$ and $d = 32$ the kept and the first dropped
   eigenvalues are within 1% of each other, so which directions survive the cut is close to
   arbitrary.
3. **The tree candidate re-encodes the reference.** The ancestor indicators already hold the
   tree. *Breaks:* in the seen regime the tree candidate can at best compress what the reference
   says exactly. Only in the held-out regime, where the reference's deeper indicator columns are
   empty, could it add anything, and it did worst there.
4. **One smoothing scheme.** *Breaks:* the propagation answer holds for halving weights over the
   lineage only, not for propagation in general.
5. **Retrieval-selected trained tables** (S2, limitation 11). *Breaks:* a regression-selected epoch
   of the same runs might score differently.
6. **One backbone's frozen text.** *Breaks:* every text-based candidate inherits one sentence
   encoder's view of the descriptions.
7. **Eighteen chances, no correction.** *Breaks:* had any candidate passed, chance alone could
   explain one pass among 18. The sealed opening was the planned guard, and it is not needed.

#### Evaluation criteria

- Each builder is deterministic, and each table's hash is its identity.
- Every candidate's provenance names text and tree inputs only.
- The list is frozen before the first read, and the decision requires exactly its members.
- The reference's predictions in every read pair with the margin read's (S10).

#### Open questions for the reviewer

1. Every candidate loses to ancestor indicators in the seen regime and roughly ties them held out.
   Is that a substantive negative, or does S6's design (two transitions, one shared penalty,
   unweighted rows, replacement only) bias the test toward indicators?
2. Which economic candidates deserve a later round: partial pooling of indicator coefficients over
   the tree, coordinates beside the indicators rather than in place of them, supervised reductions,
   or the relatedness embedding once a year outside its training window exists?
3. Is lineage smoothing with fixed halving weights a fair test of "propagation over the tree", or
   should the propagation question be closed only after a tuned or learned smoother?

### S8. Co-ownership relatedness panel

**Status: built, fixtures only.** The panel, its split, its scores, its logged reads and sealed
opening, and its margin rule and test exist and are tested on synthetic tables. No real
co-ownership table has been read. The first real read waits on two of the data owner's
permissions: selection on co-ownership's validation part, and a protocol section that pins the
tables.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $J(a, a')$ | Co-owners of $a$ and $a'$: employers, by tax identifier, with eligible records in both industries in the same filing year | count, $J \ge 1$ for every published pair |
| $J_{\bullet}(a)$ | Multi-industry employers of $a$: employers with eligible records in $a$ and in at least one other industry in the same year | count |
| $J_{\bullet\bullet}$ | All multi-industry employers, over every industry | count, not published |
| $\mathcal{C}^{\mathrm{co}}_{6}$ | Candidate industries: the six-digit codes with a multi-industry count | subset of $\mathcal{C}_{6}$ |
| $\mathfrak{h}'$ | A fixed salted cryptographic hash of a six-digit code, read as an unsigned 64-bit integer | |
| $\mathcal{C}^{\mathrm{val}}_{6}$, $\mathcal{C}^{\mathrm{sea}}_{6}$ | Validation and sealed parts of the anchor industries | partition of $\mathcal{C}^{\mathrm{co}}_{6}$ |
| $\operatorname{rel}_{a}(a')$ | Gain of $a'$ for anchor $a$ | $[0, 1]$ |
| $\operatorname{sim}_{A}(a, a')$ | Similarity of $a$ and $a'$ under arm $A$ | $\mathbb{R}$ |
| $\operatorname{NDCG}_{10}(a)$ | Expected normalized discounted cumulative gain of anchor $a$'s ranking at depth 10 | $[0, 1]$ |
| $A^{\mathrm{tree}}$, $A^{\mathrm{cos}}$, $A^{\mathrm{ccos}}$ | The three reference arms: $-D^{\ast}$, the cosine of frozen text vectors, and the cosine after the mean frozen text vector of the six-digit codes is subtracted | |
| $\mathcal{F}$, $n_{\mathcal{F}}$ | The judging families and their number | {co-ownership}, 1 |
| $r^{\mathrm{rel}}$, $\delta^{\mathrm{rel}}$ | Relatedness's relative margin, and its margin: $r^{\mathrm{rel}}$ times the chosen reference's validation headline | 0.05, NDCG units |

#### Problem formulation and data-generating story

Two industries can sit far apart in the tree and still go together in practice, because one owner
runs both. The relatedness artifact is meant for peer and competitor discovery and for
multi-industry profiles, and its judge is observed co-ownership.

The data builder counts co-ownership from the workplace-injury filings. A record is eligible when
it was filed for 2023–2025, carries a code native to NAICS 2022, and has a usable employer tax
identifier. For an employer, the industries on its eligible records in one year form a set.
$J(a, a')$ counts the employers whose set contains both $a$ and $a'$ in the same year, each
employer once however many records it has. Only pairs with $J \ge 1$ are published, in both
directions. The builder reports that 8,110 employers span two or more industries in the same year,
giving 19,576 unordered industry pairs, 6,516 of them with $J \ge 2$ (reported by the data
builder).

The estimand is the expected $\operatorname{NDCG}_{10}$ of an arm's ranking of candidate industries
by similarity, against co-ownership lift, for an anchor industry drawn like the sealed part's
anchors.

#### Estimation / inference procedure

**Inputs and refusals.** The panel reads the co-ownership counts, the multi-industry marginals and
the codebook. Before any read it refuses a code of another classification, vintage or level, a
code outside the six-digit codes, a pair whose industry has no marginal, a pair published in one
direction only or with two different counts, and a pair whose $J$ exceeds either industry's
$J_{\bullet}$. The tables' hashes must match the pins in the data owner's protocol.

**The split.** The unit is the anchor industry, so each industry's whole row of co-ownership falls
in one part:
$$\operatorname{part}(a) = \mathfrak{h}'(a) \bmod 2, \qquad 0 \mapsto \mathcal{C}^{\mathrm{val}}_{6}, \quad 1 \mapsto \mathcal{C}^{\mathrm{sea}}_{6}.$$
The salt is fixed once, the rule depends on the code alone, and the rule's fingerprint keys the
sealed opening, as in S4. A pair whose two industries fall in different parts appears in each
part once, from that part's own anchor. This is accepted, and the panel counts these crossing
pairs.

**Scoring an anchor.**

- **Candidates:** $\mathcal{C}^{\mathrm{co}}_{6} \setminus \{a\}$. An industry that the filings
  never show cannot count against an arm.
- **Relevance:** $a'$ is a relevant partner of $a$ when $J(a, a') \ge 1$.
- **Gain:** the lift among multi-industry employers over its expectation from the two industries'
  counts, $J(a, a')\, J_{\bullet\bullet} / \bigl(J_{\bullet}(a) J_{\bullet}(a')\bigr)$. Within
  one anchor, $J_{\bullet}(a)$ and $J_{\bullet\bullet}$ are constant, and NDCG divides by the
  anchor's ideal value, so the gain reduces to
  $$\operatorname{rel}_{a}(a') = \frac{J(a, a')}{J_{\bullet}(a')},$$
  the share of $a'$'s multi-industry employers that also operate in $a$. Candidates that are not
  relevant have gain 0.
- **Anchors:** the part's industries with at least one relevant partner. The others are counted,
  not scored.
- **Ties.** Each arm's DCG@10 is its expected value over all orders within tied groups, computed
  exactly (McSherry and Najork, 2008). A tied group covering ranks $s$ to $e$ adds its members'
  mean gain times $\sum_{p = s}^{\min(e, 10)} 1/\log_{2}(p + 1)$. Without this rule, the tree arm,
  whose similarity takes only five values between six-digit codes, would rank most partners behind
  every tied non-partner. S3 and S4 count ties against the truth instead.
- $\operatorname{NDCG}_{10}(a)$ is the expected DCG@10 divided by the ideal DCG@10.

**Over a part.** The headline is the mean of $\operatorname{NDCG}_{10}(a)$ over the part's
anchors, each anchor once. Reports, which never decide: the AUC per anchor of relevant partners
against candidates with $J = 0$ (ties count one half), NDCG@20, the headline by the anchor's
sector, and counts of anchors, candidates, relevant partners, unscored industries and crossing
pairs.

**Arms.** An arm supplies $\operatorname{sim}_{A}(a, a')$ for every pair of six-digit codes:

- $A^{\mathrm{tree}}$: $-D^{\ast}(a, a')$
- $A^{\mathrm{cos}}$: the cosine of $\bar h^{\circ}_{a}$ and $\bar h^{\circ}_{a'}$, unreduced, at
  384 dimensions, because no model's dimension exists yet to set a cut
- $A^{\mathrm{ccos}}$: the same after the mean of $\bar h^{\circ}$ over the six-digit codes is
  subtracted. Mean-pooled sentence vectors share a common direction, which can weaken the plain
  cosine and make the bar too easy
- a model arm: the cosine of two rows of a relatedness table (S9).

**Reads and the seal.** As in S4: every read is logged before it scores, with the panel, part,
split fingerprint, arm, seed, role, run identifier and a fingerprint of the scored rows. Reads of
the sealed part fail until one logged opening, and a second opening needs a logged reason.

**Margin and test** (built on fixtures). For each judging family, the reference is the arm with the
highest validation headline as a point estimate, with ties going to $A^{\mathrm{tree}}$ and then to
$A^{\mathrm{cos}}$. Then $\delta^{\mathrm{rel}} = r^{\mathrm{rel}} \times$ that headline. Each arm
is one replicate, because none has seeds, which is why the margin is relative. The margin record
comes from fresh reference reads, before any candidate read on that family. Training-monitor reads
count as candidate reads. A candidate with fewer than five seeds is refused. S10 gives the test: a
candidate must beat the reference by $\delta^{\mathrm{rel}}$ on every judging family. A candidate
table whose provenance names any input other than the codebook, the frozen-text table and the
training families is refused.

**Which read decides.** The validation part chooses at most one forward candidate. If it does not
beat the reference on validation, the sealed part stays sealed for a later candidate, and the
reference stands. Otherwise the sealed part is opened once, and the same test runs on it with the
validation margin.

**A second judging family, later.** Parent–subsidiary links from a second public source, the
Toxics Release Inventory's reported parent companies, may join as a judge only through a design
amendment made before the sealed opening, with its own margin record first and the same split
rule and salt. A candidate must then beat the reference on both families, at the level S10 sets
for two. Co-location from the American Community Survey's place-of-work data may
join training, never judging. Labor-flow, staffing, input–output and contract-award families have
no role yet.

#### Assumptions and limitations

1. **Co-ownership measures the claimed relatedness.** *Breaks if violated:* co-ownership within a
   tax identifier reflects how firms are organized, for example franchising, vertical integration
   or holding structures, and one firm can hold several identifiers, so co-ownership across a
   firm's identifiers is missed.
2. **One shared employer makes a partner.** *Breaks:* a single co-owner can be idiosyncratic. The
   builder reports that 94.84% of single-employer pairs recur from 2023 to 2024, which speaks to
   stability, not to relevance.
3. **No pair is below chance.** The gain is positive for every co-owned pair. *Breaks:* a pair that
   co-occurs less often than its industries' counts predict still counts as relevant. A log-lift
   gain would need the total $J_{\bullet\bullet}$, which the builder does not publish.
4. **Mirror pairs cross the parts.** *Breaks:* a pair seen from a validation anchor is also scored
   from its sealed anchor. Selection on validation thus sees part of the sealed part's evidence,
   from the other side.
5. **The training family shares the judges' filers** (cross-component assumption 5). *Breaks:* a
   model trained on co-location can score on co-ownership through the same employers'
   establishments.
6. **Filing rules shape coverage.** *Breaks:* industries that OSHA's rules designate are
   over-represented, and industries that never file are not judged.
7. **Each anchor counts once.** *Breaks:* an anchor with one relevant partner weighs as much as an
   anchor with hundreds, so the headline is noisy where partners are few.
8. **Two tie rules in one system.** *Breaks:* relatedness credits ties at their expected value,
   while text-to-code counts them against the truth, so the two artifacts' scores are not on one
   footing for coarse arms.

#### Evaluation criteria

- Each anchor industry falls in one part only, and a rebuilt table keeps each industry in its part.
- The tables are refused before any read when their hashes, codes or mirror counts disagree.
- The tie rule equals the brute-force mean over every order within tied groups, on fixtures.
- Every read is logged before it scores, and the sealed part opens once, keyed to the split rule.
- Three non-trained reference arms, including a centered text cosine, set the bar.

#### Open questions for the reviewer

1. Is same-year co-ownership within an employer tax identifier a credible judge of industry
   relatedness for peer discovery and multi-industry profiles? How does it compare with published
   relatedness measures: co-agglomeration, labor flows between industries, or input–output links?
2. Is relevance at one co-owner, with gain $J(a, a') / J_{\bullet}(a')$, the right graded
   relevance? Or should relevance require co-ownership above a null model's expectation?
3. Is NDCG@10 per anchor the right headline, against AUC, a rank correlation with lift, or a
   calibrated association measure?
4. Are mirror pairs that cross the parts an acceptable leak, or should the split be by unordered
   pair, at the cost of half the data?

### S9. Relatedness embedding

**Status: specified, not built.** The method was chosen in an approved design. Its exact
configuration grid and seeds are to be fixed in a plan before the first candidate read. Training
on co-location waits for the data owner's permission. If that permission never comes, this
component parks.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\Gamma(a)$ | Counties where industry $a$ has a cell in the 2023 area mix | set of counties |
| $n_{\Gamma}$ | Counties with eligible 2023 records | count |
| $\operatorname{lift}(a, a')$ | County co-presence lift, $\lvert\Gamma(a) \cap \Gamma(a')\rvert\, n_{\Gamma} / \bigl(\lvert\Gamma(a)\rvert \lvert\Gamma(a')\rvert\bigr)$ | $[0, \infty)$ |
| $\mathcal{N}(c)$ | Neighbors of $c$ over tree edges and co-location edges | subset of $\mathcal{C}$ |
| $h^{(l)}_{c}$ | State of node $c$ after layer $l$, with $h^{(0)}_{c} = \bar h^{\circ}_{c}$ | $\mathbb{R}^{384}$ at $l = 0$ |
| $\Theta^{(l)}_{1}$, $\Theta^{(l)}_{2}$ | Weights of layer $l$ for the node's own state and for its neighbors' mean | |
| $z^{\mathrm{rel}}_{c}$, $Z^{\mathrm{rel}}$ | Output of code $c$, of unit length, and the table of all codes | $\mathbb{R}^{d}$, $2125 \times d$ |
| $w_{\mathrm{tree}}$ | Weight of tree edges as training positives | hyperparameter, 0 allowed |

#### Problem formulation and data-generating story

The relatedness embedding is to place codes so that industries an owner runs together are close.
It cannot train on co-ownership, which judges it. It trains instead on two other signals: the
tree's parent–child edges, and co-location, the counties in which two industries are both present.
The estimand is a flat table $Z^{\mathrm{rel}}$ of the 2,125 codes whose cosine similarity ranks
co-owned industries high on S8's sealed part.

The co-location data is the builder's 2023 area mix: for each county, the injury filings' records
of each industry, from records with a native NAICS 2022 code and an assigned county, each record
with weight one. Most cells hold a single record, so the design uses presence in a county, not
shares.

#### Estimation / inference procedure

**Graph.** One node for each of the 2,125 codes.

- **Tree edges** join each code to its parent, in both directions.
- **Co-location edges** join six-digit industries $a$ and $a'$ when
  $\lvert\Gamma(a) \cap \Gamma(a')\rvert \ge 2$ and $\operatorname{lift}(a, a') > 1$. The edge's
  weight is $\ln \operatorname{lift}(a, a')$.

**Model.** GraphSAGE (Hamilton, Ying and Leskovec, 2017) with mean aggregation over a node's
neighbors of both edge types, starting from the frozen text vectors:
$$h^{(l+1)}_{c} = \sigma\Bigl(\Theta^{(l)}_{1} h^{(l)}_{c} + \Theta^{(l)}_{2}\, \frac{1}{\lvert\mathcal{N}(c)\rvert} \sum_{c' \in \mathcal{N}(c)} h^{(l)}_{c'}\Bigr), \qquad z^{\mathrm{rel}}_{c} = \frac{h^{(L')}_{c}}{\Vert h^{(L')}_{c}\Vert},$$
where $L'$ is the number of layers, the last layer has $d$ outputs, and $\sigma$ is a nonlinearity
applied between layers and not after the last. The similarity is cosine, and the output is flat.
The design names the aggregation and the output. The number of layers, the widths and the
nonlinearity are hyperparameters. It does not say whether edge weights or edge types enter the
aggregation.

**Loss.** Link prediction with negative sampling. Positives are co-location edges, drawn in
proportion to their weight. Negatives are random pairs of six-digit industries. Tree edges carry
messages, and whether they also count as positives is the hyperparameter $w_{\mathrm{tree}}$. The
design does not fix the loss's exact form.

**Selection.**

- A grid over $d$, the number of layers, the hidden width, $w_{\mathrm{tree}}$, the co-location
  threshold and the learning rate is fixed before the first candidate read, with the seeds.
- Each configuration trains on at least five seeds. Each seed keeps the earliest epoch with the
  highest S8 validation headline, and every epoch's read is logged as a candidate read.
- The forward candidate is the configuration with the highest mean validation headline over its
  seeds. S10's test then decides on validation, and once on S8's sealed part.

**Placement of unseen codes.** The features are text, so a code with no co-location edge is still
placed, by its text and its tree neighbors. The validation headline is reported apart for anchors
with no co-location edge.

**Export.** Each seed's table of all 2,125 codes, with a provenance naming the codebook, the
frozen-text table, the area-mix table, the configuration, the seed and the checkpoint. Only
six-digit codes are judged. Codes at levels 2 to 5 are placed, not judged. The graph is small, and
training runs on one workstation.

**Alternatives weighed in the design.** A blended factorization, a truncated singular value
decomposition of above-chance association matrices from the tree and county co-presence, was
rejected: it is deterministic and has few settings, but it is linear and not the method the owner
named. Running both as candidates was rejected as twice the model work. Fine-tuning S2's encoder
with a co-location term was rejected, because a change to S2 reaches every later text-to-code
run.

#### Assumptions and limitations

1. **Co-location transfers to co-ownership.** *Breaks if violated:* the model learns where
   industries locate together, and is judged on who owns them together. Industries that locate
   together for reasons unrelated to ownership, such as shared infrastructure, would rank high and
   be scored as misses.
2. **The same filers feed training and judging** (cross-component assumption 5). *Breaks:* one
   employer's establishments in one county add to both, so the held-out score can rise through
   the training signal itself.
3. **Presence from single records.** *Breaks:* one filing makes an industry present in a county,
   so co-location edges carry the noise of single records, and the threshold of two shared
   counties is a guess, partly searched by the grid.
4. **Mixed edges, one mean.** *Breaks:* where co-location is sparse, a node's tree neighbors
   dominate its mean, and the model reverts toward the tree, which is one of its references.
5. **Many choices on one validation part.** *Breaks:* a grid times five seeds times every epoch,
   all selected on co-ownership's validation part, inflates the forward candidate's validation
   gain. Mirror pairs (S8, limitation 4) carry part of it into the sealed part.
6. **No linear candidate.** *Breaks:* a graph network that beats the references could still lose
   to a factorization of the same data, and nothing would show it.

#### Evaluation criteria

- No co-ownership row enters training, and the provenance rule enforces it.
- Every selection read is a read of S8's validation part, logged after the margin record.
- Each configuration has at least five seeds, and the grid is fixed before the first read.
- Adoption only through S10's test on S8, once on the sealed part.

#### Open questions for the reviewer

1. Is GraphSAGE over tree and co-location edges, with frozen text features, a sound method for
   this target? Should a linear baseline, such as a spectral embedding of the co-location lift
   matrix, run as a second candidate after all?
2. Is link prediction on co-location edges the right objective when the judge is a ranking of
   co-owned industries? Would a ranking loss, or a loss on lift values, transfer better?
3. How should co-location edges be defined when most county cells hold one record?
4. Given that training and judging share filers, which check would bound the leak, for example
   scoring co-ownership without employers that also contribute the co-location edges of the same
   pair?

### S10. Paired decision protocol

**Status: run on real data for the shared-space decision and the economic decision.** The
text-to-code and relatedness rules, which require a candidate to beat its reference by a margin,
are built and tested on fixtures only.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $X$, $Y$ | Two arms compared: the challenger and the incumbent or reference | configurations, candidate tables or reference arms |
| $\mathcal{P}$, $n$ | The panels or parts of one decision, and their number | |
| $\zeta^{X,k}_{i}$ | Value of item $i$ under arm $X$ and seed $k$ | see the item table below |
| $\mathcal{G}_{P}$ | Resampling units of panel $P$ | see the item table below |
| $\mathcal{I}_{G}$ | Items of unit $G$ | |
| $K_{X}$ | Seeds of arm $X$ | at least 5 for a trained arm, 1 for a deterministic arm |
| $B$ | Bootstrap replicates | 10,000 |
| $U^{(b)}_{G}$ | Multiplicity of unit $G$ in replicate $b$ | multinomial, $\lvert\mathcal{G}_{P}\rvert$ trials |
| $M^{(b)}_{X,k}$ | Multiplicity of seed $k$ of arm $X$ in replicate $b$ | multinomial, $K_{X}$ trials |
| $\hat S_{X,P}$ | Panel statistic of arm $X$: MRR, NDCG@10 or mean squared error | |
| $o_{P}$ | Orientation: $+1$ when higher is better, $-1$ when lower is better | $\{+1, -1\}$ |
| $\Delta_{P}(X, Y)$ | Oriented difference, positive when it favors $X$ | |
| $q_{\alpha}$ | The $\alpha$-quantile of the $B$ replicate differences | $\alpha \in (0, 1)$ |
| $\delta_{P}$ | Margin of panel $P$ | units of $\hat S$ |
| $\gamma_{X}$ | Held-out gain of $X$: the ancestor-indicator reference's held-out error minus $X$'s | squared log units |

#### Problem formulation and data-generating story

A decision asks whether arm $X$ should replace arm $Y$ on a fixed set of validation panels or
parts. Two sources of variation are treated as random: training randomness, indexed by the seed,
and the sample of evaluation units, viewed as drawn from a population like those in the panel. The
estimand on panel $P$ is the expected oriented difference $\mathbb{E}[\Delta_{P}(X, Y)]$ over both
sources. It is answered with a margin and a lower confidence bound rather than a point comparison.

One resampling core serves four decisions with two rules:

| Decision | Panels | Rule | State |
|---|---|---|---|
| Shared-space (history) | S3, and S6's seen and held-out regimes in level mode | Non-inferior on every panel and superior on one | Ran once, before the split |
| Economic | S6's seen and held-out growth regimes | Non-inferior on every panel and superior on one, against the ancestor-indicator reference | Ran once, 2026-11-18: none adopted |
| Text-to-code | S4's headline, on its validation part and then its sealed part | Beat the reference by $\delta$ | Built, fixtures only |
| Relatedness | S8's headline for each judging family, on its validation part and then its sealed part | Beat the reference by $\delta$ on every family | Built, fixtures only |

#### Estimation / inference procedure

**Item values and units.** Each seed of each arm is read once on each panel. The item values
$\zeta$ are stored per item with their unit.

| Panel | Item and value | Unit | Statistic |
|---|---|---|---|
| S3 | Validation query, its reciprocal rank (4,042 queries) | True code (939) | MRR |
| S4 | Headline pair, its reciprocal rank (8,508 pairs in the validation part) | Normalized description (3,768) | MRR |
| S6 | Scored row, its squared error averaged over five fold repeats (777 seen, 1,554 held-out) | Four-digit group (240) | MSE |
| S8 | Anchor industry, its $\operatorname{NDCG}_{10}$ (one item per unit) | Anchor industry | Mean NDCG@10 |

The point estimate averages over seeds the statistic computed on every unit:

$$\hat S_{X,P} = \frac{1}{K_{X}} \sum_{k=1}^{K_{X}} \frac{\sum_{G \in \mathcal{G}_{P}} \sum_{i \in \mathcal{I}_{G}} \zeta^{X,k}_{i}}{\sum_{G \in \mathcal{G}_{P}} \lvert\mathcal{I}_{G}\rvert}.$$

**The paired two-stage bootstrap.** For each replicate $b = 1, \dots, B$:

1. **Units, shared.** Unit multiplicities $U^{(b)}$ are drawn by sampling $\lvert\mathcal{G}_{P}\rvert$
   units with replacement. The draw is seeded by the panel alone, so every arm on the panel sees
   the same units: this is the pairing.
2. **Seeds, per arm.** Each arm's seed multiplicities $M^{(b)}_{X}$ are drawn with replacement from
   its own seeds, under a stream seeded by the panel and the arm. Seeds are not paired across arms,
   and an arm's draws are identical in every comparison it enters. A deterministic arm has one
   seed, so its seed stage is degenerate.
3. **Statistic.** The mean over drawn seeds of the item-weighted mean over drawn units:
   $$\hat S^{(b)}_{X,P} = \frac{1}{K_{X}} \sum_{k} M^{(b)}_{X,k}\, \frac{\sum_{G} U^{(b)}_{G} \sum_{i \in \mathcal{I}_{G}} \zeta^{X,k}_{i}}{\sum_{G} U^{(b)}_{G}\, \lvert\mathcal{I}_{G}\rvert}.$$
4. **Difference.**
   $\Delta^{(b)}_{P}(X, Y) = o_{P}\bigl(\hat S^{(b)}_{X,P} - \hat S^{(b)}_{Y,P}\bigr)$.

Panels are resampled independently of one another. Intervals are two-sided percentile intervals of
the $B$ replicate differences, with linear interpolation between order statistics.

**Rule 1: non-inferior everywhere, superior somewhere.** With $n = |\mathcal{P}|$ panels:

- **Non-inferior on $P$:** $q_{0.025} > -\delta_{P}$, the lower bound of the 95% interval.
- **Superior on $P$:** $q_{0.025/n} > 0$, the lower bound of the two-sided $1 - 0.05/n$ interval:
  98⅓% for three panels, 97.5% for two. Only the lower bound is tested.
- **Adopted:** $X$ is adopted over $Y$ when it is non-inferior on every panel and superior on at
  least one. Each additional panel is thus an additional chance to adopt, and the superiority level
  divides the 5% error rate among them.

**Rule 2: beat by the margin.** $X$ beats the reference $Y$ when
$$q_{0.025/n}\bigl(\Delta_{P}(X, Y)\bigr) > +\delta_{P} \quad \text{for every } P \in \mathcal{P},$$
the lower bound of the two-sided $1 - 0.05/n$ interval. Text-to-code has one panel, so $n = 1$ and
the level is 95%. Relatedness has $n = n_{\mathcal{F}}$ judging families, so 95% with co-ownership
alone and 97.5% if a second family joins. The rule exists because Rule 1's margin never binds on a
single panel: with one panel, Rule 1 reduces to plain superiority. Rule 2 turns the margin into a
required minimum gain. A reference arm without seeds is one replicate. The relatedness test
refuses a candidate with fewer than five seeds. The text-to-code test has no such check, and S5's
outline plans at least five seeds per candidate.

**Several arms** (Rule 1). Every ordered pair of arms is compared. An arm is dominated when some
other arm is adopted over it, and the survivors are the non-dominated arms. If dominance cycles so
that every arm is dominated, all arms survive. A fixed order then picks one survivor:

1. fewer components (stages or post-processing steps)
2. lower dimension
3. non-hyperbolic over hyperbolic geometry
4. the higher held-out gain $\gamma$.

An arm that ties on every key leaves the decision unresolved. Lower dimension precedes geometry by
design: low dimension is the economic purpose, and a hyperbolic space that only matches a larger
flat one is the geometric result worth keeping. The pairwise comparisons within a decision are not
corrected for multiplicity. The order toward the simpler arm is the stated guard against a
spurious win. In the economic decision, a candidate is eligible only if it is adopted over the
reference, and dominance and the order pick among eligible candidates.

**Margins.**

| Decision | Margin $\delta_{P}$ | Values |
|---|---|---|
| Shared-space | Three times the across-seed SD of the original reference's per-seed statistic | 0.1187 (S3 MRR), 0.0333 (seen level MSE), 0.0368 (held-out level MSE) |
| Economic | 0.05 times the reference's validation MSE, fixed on 2026-10-17 | 0.0001593 (seen growth), 0.0001620 (held-out growth) |
| Text-to-code | 0.05 times the chosen reference's validation headline MRR | not yet fixed (about 0.022 by projection, S4) |
| Relatedness | 0.05 times the chosen reference's validation headline NDCG@10, per judging family | not yet fixed |

The three later margins are relative because a seed-spread margin is undefined for references
without seeds. For the economic margins, two alternatives were rejected:

- a multiple of the reference's SD across fold repeats, which is near zero in the seen regime,
  where repeats only redraw the tuning folds, so non-inferiority would collapse into superiority
- a multiple of the bootstrap SD of the reference's MSE, which double-counts resampling that the
  interval already covers.

The read that fixed the economic margins scored only comparators that use no candidate table: the
covariates, one-hot indicators (seen regime only), ancestor indicators, and each with the
covariates. So no candidate's growth error existed before the margins were recorded.

**Ordering and integrity rules.**

- **Margins first.** Every read of a candidate must postdate its decision's margin record. For
  text-to-code and relatedness, every logged read carries a role, and a margin record is refused
  when the log holds an earlier candidate read, or a read with no role, on its panel.
  Training-monitor reads count as candidate reads there. For the economic decision the rule covers
  growth reads only, because the trained candidates' monitor reads were S3 reads made before the
  margins existed.
- **Identical panels.** Every arm in a decision is read on the same split draws, the same data,
  hashed exactly, and the same fit settings.
- **Pairing within a tolerance** (economic decision). Each candidate's read refits the reference
  on its own run. The reference's rows must equal the margin read's exactly, and each reference
  prediction must equal the margin read's to within $10^{-13}$, absolute. The first real read ran
  under a newer operating-system release than the margin read, with the same locked library
  versions. 11,406 of the 11,655 reference predictions differed in their last bits, by at most
  $6.9 \times 10^{-16}$. The reference's validation MSE moved by $2.7 \times 10^{-16}$, relative.
  A difference of $10^{-13}$ per prediction moves the reference's MSE by about $7 \times 10^{-11}$
  of $\delta$.
- **Write once.** Records are written once and refuse overwriting.

**Results on record.**

- **The shared-space decision ran once,** before the split, over the nine geometry × dimension
  cells (S2's table). Of 72 ordered comparisons, 33 adopted. The Euclidean arm at $d = 16$ was the
  only non-dominated arm and was adopted over each of the other eight. Against the original
  reference:

  | Panel | $\Delta$ | 95% interval | 98⅓% interval | $-\delta_{P}$ | Non-inferior | Superior |
  |---|---|---|---|---|---|---|
  | S3 MRR | 0.179 | [0.152, 0.206] | [0.145, 0.212] | −0.119 | yes | yes |
  | Seen level MSE | 0.0056 | [−0.0047, 0.0158] | [−0.0070, 0.0182] | −0.0333 | yes | no |
  | Held-out level MSE | 0.0055 | [−0.0069, 0.0176] | [−0.0099, 0.0206] | −0.0368 | yes | no |

  The winner was superior on all three panels over the three $d = 8$ arms, and on S3 alone over
  the original reference and every $d = 32$ arm. Across matched dimensions, five comparisons
  adopted a hyperbolic arm over a flat one and eleven the reverse.
- **The economic decision ran once,** over the 18 candidates of S7, with $B = 10{,}000$.
  - Seen regime: every $\Delta$ is negative, from $-0.00033$ to $-0.00051$, and no candidate is
    non-inferior. The 95% intervals are six to eight times as wide as $\delta$, for example
    $[-0.00121, +0.00001]$ for the frozen-text candidate at $d = 8$.
  - Held-out regime: 16 of 18 candidates are non-inferior and none is superior. The intervals are
    one to two times $\delta$ wide. The frozen-text candidate at $d = 32$ has the largest gain,
    $\Delta = +0.000117$, with 97.5% interval $[-0.000044, +0.000295]$.
  - No candidate is adopted, the eligible and non-dominated sets are empty, and the reference
    stands.

#### Assumptions and limitations

1. **Units are exchangeable and independent.** *Breaks if violated:* dependence between units,
   within a subsector or a sector, makes the intervals too narrow. In S4, one employer can own
   pairs in several description units.
2. **Few seeds represent training randomness.** A trained arm's seed stage resamples five values,
   and seeds are unpaired across arms. *Breaks:* a percentile bootstrap over five seeds describes
   their spread poorly, and the seed stage cannot remove variation that two arms share.
3. **Margins are judgments.** Three across-seed SDs was 29% of the original reference's mean MRR.
   Five percent of a reference's statistic is a tolerance, not an estimate of anything. *Breaks:*
   the same 5% means very different things across MSE, MRR and NDCG, and against the panels' own
   noise.
4. **Unequal power across regimes.** *Breaks:* in the economic decision, the seen regime's
   intervals are six to eight times $\delta$, so non-inferiority there demands predictions that
   track the reference's closely. The held-out regime's intervals are one to two times $\delta$.
   Rule 1 then rests on whichever regime is noisiest.
5. **Validation reuse.** In text-to-code and relatedness, one validation part chooses the
   reference, fixes the margin, selects checkpoints and picks the forward candidate. *Breaks:* the
   forward candidate's validation gain is optimistically biased by an unmeasured amount. The sealed
   read removes the bias once.
6. **No multiplicity correction across candidates.** 72 comparisons ran in one decision, and the
   economic decision had 18 candidates. *Breaks:* the familywise error rate is not controlled. The
   order toward the simpler arm and the single sealed read are the only guards.
7. **Two rules in one system.** Rule 1 adopts on non-inferiority everywhere plus superiority
   somewhere, and Rule 2 requires a gain of $\delta$ everywhere. *Breaks:* the economic artifact
   could be adopted for a gain that text-to-code or relatedness would reject.
8. **Pooled item weighting.** Each unit's weight is its item count. *Breaks:* codes with many
   validation queries, descriptions with many pairs and groups with many codes dominate the
   statistics.
9. **Environment-bound pairing.** *Breaks:* decisions reproduce only under the locked library
   versions of the reference read, and on platforms whose arithmetic stays within the tolerance.

#### Evaluation criteria

The protocol is judged by properties of its decisions, not by a panel of its own:

- **Reproducibility:** fixed replicate seeds by panel and arm, write-once records, and exact
  re-hashing of every stored artifact.
- **Pre-registration:** margins and references fixed before any read of a compared arm, enforced
  by timestamps and read roles in the log.
- **Error control:** 5% per superiority family by panel count, and 2.5% one-sided per lower bound
  at $n = 1$.
- **Seal discipline:** decisions read validation data only, and each sealed set is opened once,
  after its artifact's validation test, for the forward candidate and its reference.

#### Open questions for the reviewer

1. Is a two-stage percentile bootstrap, with units paired across arms and seeds resampled
   independently per arm, a sound basis for comparing trained models with five to ten seeds? Would
   a hierarchical model, a paired-seed design or a permutation test serve better?
2. How should margins be set for representation choices: from the reference's noise, as a fraction
   of its statistic, or from the downstream use's own tolerance?
3. One validation part both selects and decides within each new artifact. What protocol removes or
   bounds the resulting selection bias, short of a third split?
4. Should all three artifacts use one adoption rule? Rule 1 lets the economic artifact win on one
   regime while merely matching on the other. Rule 2 asks text-to-code and relatedness for a
   margin of gain everywhere.
5. The economic decision compared deterministic tables with a fixed reference, so its interval
   reflects only the resampling of four-digit groups. Is that the right uncertainty for choosing
   regressors that downstream models will use on all codes?

### S11. Graph refinement over the tree

**Status: parked.** The method is unchanged since the previous description, and nothing reads its
output. Its question for the economic artifact, whether propagating information over the tree
helps, was answered with lineage smoothing (S7): there is no evidence that it does. The relatedness
embedding (S9) may reuse its graph machinery, but not its method. It is described briefly here.

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $p^{G}_{c}$ | Node state of code $c$: a free point, initialized at S2's hyperbolic point $p_{c}$ | $\mathbb{H}^{d}$, held in $\mathbb{R}^{d+1}$ |
| $\bar p^{G}_{c}$ | Refined point of code $c$: the second layer's output | $\mathbb{H}^{d}$ |
| $\operatorname{dist}^{G}$ | Hyperbolic distance between refined points | $[0, \infty)$ |

#### Problem formulation and data-generating story

The stage asks whether message passing along the tree refines S2's hyperbolic code points. It is
transductive: each of the 2,125 codes has a free node state, initialized at its text-stage point.
It has no path for new text, so it cannot serve retrieval. Its estimand is the refined points after
a fixed number of epochs.

#### Estimation / inference procedure

- **Graph.** Child, sibling, grandchild and great-grandchild edges from S1, both directions, with
  fixed weights 1, 0.708, 0.5 and 0.3125, and one self-loop per node.
- **Layers.** Two layers of attention-weighted message passing in the tangent space at the origin:
  map the states to the tangent space, apply a square linear map, weight each edge's message by a
  learned score plus the edge's log-weight, normalize across coordinates, and map back.
- **Objective.** A triplet term on refined distances, with positives drawn from descendants and
  siblings and negatives from S1's older pair universe, and a level term pulling
  $\sinh r$ toward $(\lambda - 2)/2$. The two are weighted by learned log-variances.
- **Curriculum.** A warm-up of two epochs with a halved margin, then hard-negative weighting from
  epoch 4, over eight epochs. No checkpoint is selected: the last epoch is exported.

**It cannot compose as configured.** The layers' width does not match the text stage's
hyperbolic points at any allowed $d$, so the first step would fail. The stage refuses flat arms,
and every adopted or reference arm since the shared-space decision is flat.

#### Assumptions and limitations

1. **A fixed code set.** *Breaks if violated:* no refined point exists for new text or for codes
   of a later vintage.
2. **The tree is the propagation graph.** *Breaks:* for economic coordinates, propagation over the
   tree mostly re-expresses what the ancestor-indicator reference already holds.
3. **Normalization overrides radius.** Normalizing across all coordinates sets each output's
   radius near the square root of the width, about 4.1 at $d = 16$. *Breaks:* the text stage's
   radial structure does not carry through, and the level term's targets of 0 to 1.44 conflict
   with the text stage's targets of 1 to 5.
4. **No selection.** *Breaks:* the refined points are an unselected endpoint, and its logged
   statistics do not measure generalization.
5. **Mechanisms that do not act.** A declared learnable curvature receives no update, and several
   configured filters and weights cannot change the objective. *Breaks:* the configuration
   describes a richer method than the one that would run.

#### Evaluation criteria

Whether the stage returns would be decided only by S10's rule on the panels of the artifact it
would serve, never by its structural statistics.

#### Open questions for the reviewer

1. Given the smoothing result and that no artifact now reads hyperbolic points, should graph
   refinement over the tree be retired outright?

## Open questions for the reviewer

1. **The split and its judges.** Is serving relatedness, text-to-code retrieval and economic
   coordinates with three artifacts, each judged only on its own evidence, sound? Is each judge
   the right one: owner-written injury-filing descriptions for retrieval, national employment
   growth for coordinates, and same-year co-ownership for relatedness? What is lost by no longer
   asking one space to serve all three?
2. **One filing population.** Text-to-code's judge, relatedness's judge and relatedness's training
   signal all come from the same workplace-injury filings. Which independence conditions matter
   for each held-out claim, and which checks would bound the dependence?
3. **Text-to-code.** Owners' own words score 0.14 to 0.18 MRR below Census phrases, and a trigram
   baseline nearly ties the trained encoder on them. What should the text-to-code artifact become
   under the current state of the art in industry autocoding? Should it train on owner text, and
   if so, how should a training part coexist with the validation and sealed parts?
4. **Economic coordinates.** No candidate built from text or the tree beats ancestor indicators.
   Is the conclusion robust to the panel's design? Which candidate families, outcomes or panels
   deserve a later round, and should the relatedness embedding compete once a year outside its
   training window exists?
5. **Relatedness.** Is a graph network over tree and co-location edges, with frozen text features,
   judged against the tree's distance and frozen-text cosine on co-ownership NDCG@10, a sound
   design? Which published relatedness measures or benchmarks should it be compared with?
6. **Margins and rules.** Three relative margins of 5% and two adoption rules govern the three
   artifacts. Is that coherent, and what would a principled margin be for each use?
7. **Missing evidence.** Which additional panel or check would most change these conclusions?
   Candidates include other economic outcomes, a sub-national grain, employment-weighted errors,
   codes held out of retrieval training, near-copy filtering of owner text, and a relatedness judge
   from an independent source.
