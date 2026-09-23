# NAICS hyperbolic embedding methodology

> For agentic workers: when specs/naics-embedding-critique.md exists beside this
> file, REQUIRED SKILL: describe-critique-methodology (synthesize mode) —
> do not draft a spec directly.

This document describes a method that represents every industry in the 2022 North American
Industry Classification System (NAICS) as a point in hyperbolic space. The point is learned from
the official text that defines each industry and is supervised by the classification's own
hierarchy. The document is written for an external methodological review and deliberately omits
implementation detail.

It describes the method **as it executes under its reference configuration** on 2026-09-23. The
hyperparameter values quoted throughout are that configuration's. Some designed mechanisms exist
but cannot act under that configuration. The text says so rather than describing design intent
as behavior, using two labels:

- **inert**: the mechanism runs but provably cannot change the objective;
- **nominal**: the quantity is declared learnable or adaptive but receives no update.

There is one exception. The graph stage cannot run under its reference configuration (see
Composition), so C3 describes how it would run against the current supervision, using its
reference hyperparameters.

The document reports counts measured on the current supervision data. Apart from one geometric
observation from a historical run, it reports no model-performance results. The project states
three purposes for the representation: hierarchical search over codes, clustering, and features
for downstream economic modeling. The method itself optimizes
and evaluates only agreement with the taxonomy it is trained on.

## Component inventory

| Component (by methodological role) | Estimand / output | Consumes |
|---|---|---|
| **C1. Taxonomy supervision construction** | Structural distance $D$, structural relation $R$, directional exclusion indicators $X$; a positive-pair universe and, per pair, a candidate list of negatives | The official NAICS 2022 code list and, per code, its title, description, illustrative examples, and cross-reference ("excluded activities") text |
| **C2. Text-conditioned hyperbolic contrastive encoder** (the *text stage*; the project's "Stage 3") | An inductive map $F_{\Theta}$ from a code's four text fields to $\mathbb{H}^{n}$; the text-stage embedding $Z^{(3)}$ of every code | The four text fields; $D$, $R$, $X$ and sampled training tuples from C1 |
| **C3. Graph-convolutional hyperbolic refinement** (the *graph stage*; the project's "Stage 4") | Refined points $Z^{(4)}$ for the fixed code set (transductive) | $Z^{(3)}$ as initial values; a relation-typed taxonomy graph built from $R$; positive pairs and candidate negatives from C1 |
| **C4. Evaluation protocol** | Structural-agreement statistics $M(Z^{(3)})$, $M(Z^{(4)})$; an accept/reject decision $A$ on $Z^{(4)}$ relative to $Z^{(3)}$; two defined but never-executed downstream benchmarks | $Z^{(3)}$, $Z^{(4)}$, $D$, the taxonomy, and (for one benchmark) public employment statistics |

## Composition

Symbols are defined in the Notation section below. The components compose sequentially, with
point estimates only and no feedback between stages.

1. **Supervision.** C1 is a deterministic function of the official documents plus seeded
   sampling. It yields $D$, $R$, $X$, the positive universe $\mathcal{P}$, and candidate lists
   $\mathcal{U}_{ap}$.
2. **Text stage.** C2 fits $\Theta$ by stochastic gradient descent on its objective
   $\mathcal{L}^{(3)}$. It keeps $\hat\Theta$, the epoch with the lowest validation contrastive
   loss, and then encodes every code:
   $$Z^{(3)}_{i} = F_{\hat\Theta}(t_{i}) \in \mathbb{H}^{n}, \qquad n = 384, \qquad i \in \mathcal{C}.$$
3. **Graph stage.** C3 treats per-code node states $H$ as free parameters, initialized at
   $H = Z^{(3)}$. It fits them jointly with layer parameters $\Xi$ and exports the last epoch:
   $$Z^{(4)} = G_{\hat\Xi}(\hat H;\ \mathcal{G}).$$
   The text encoder is frozen by then and receives no gradient from this stage.
4. **Acceptance.** C4 computes $M(Z)$ for $Z \in \{Z^{(3)}, Z^{(4)}\}$ over all codes and accepts
   the refinement when
   $$A = \mathbb{1}\left[\Delta\rho_{\mathrm{P}} \ge -0.02\right] \cdot \mathbb{1}\left[\Delta\mathrm{NDCG}@10 \ge -0.01\right] \cdot \mathbb{1}\left[\Delta\mathrm{PR}@1 \ge 0.05\right] = 1.$$
   The method does not define the deliverable when $A = 0$.

Three properties of the composition matter for review:

- **The stage spaces are not aligned.** $F_{\hat\Theta}$ is inductive: any text maps into
  $\mathbb{H}^{n}$. $Z^{(4)}$ exists only for $\mathcal{C}$. A new description can be encoded, but
  its point lies outside the refined geometry, and no alignment map between the two is defined.
- **The dimensions do not match.** The text stage emits points in $\mathbb{R}^{n+1} =
  \mathbb{R}^{385}$. The graph stage's layers are square maps on their own ambient dimension,
  $n'+1$, which the reference configuration sets to 31. No reduction or lifting map is specified,
  so the composition is undefined as configured. No text-stage versus graph-stage comparison has
  been produced.
- **There is one supervisory source.** Every training target in both stages, and every C4
  statistic, derives from the same hierarchy and the same positive-pair universe.

## Cross-component assumptions

1. **Structural agreement stands in for quality.** Both stages optimize, and C4 measures,
   agreement with the taxonomy's own metric. *Breaks if violated:* the intended uses (search from
   business descriptions, economic features) may depend on relatedness the taxonomy does not
   encode. No stage optimizes for it, and no statistic would detect its absence.
2. **The code universe is shared.** Both stages and C4 index the same 2,125 codes in the same
   order. *Breaks:* any mismatch silently misaligns targets and points.
3. **An exclusion means the same thing in both stages.** The text stage treats every explicit
   exclusion as a mandatory repulsive negative. The graph stage ignores exclusion status, so
   structurally adjacent excluded pairs become message-passing neighbors. Under the current
   supervision there are 481 such pairs: 474 siblings, 5 grandparent–grandchild pairs, 1
   parent–child pair, and 1 great-grandparent pair. *Breaks:* the graph stage can pull together
   exactly the pairs the text stage pushed apart.
4. **One curvature everywhere.** Both stages and every evaluation use $c = 1$.
   - Every distance formula in the system is correct only at $c = 1$. For $c \ne 1$ they compute
     $\sqrt{c}\,\operatorname{arcosh}(-\langle x,y\rangle_{\mathcal{L}})$ instead of
     $c^{-1/2}\operatorname{arcosh}(-c\langle x,y\rangle_{\mathcal{L}})$.
   - The graph stage's per-layer curvature is nominal.

   *Breaks:* any non-unit-curvature run produces invalid distances in both training and
   evaluation.
5. **One radial coordinate.** Losses and diagnostics actually use three radial quantities:
   - $x_{0} = \cosh r$, in the diagnostics;
   - $\Vert x_{s}\Vert = \sinh r$, in the level-alignment losses of both stages, the text stage's
     radius penalty, and the graph stage's margin;
   - $r$ itself, in a logged-only margin.

   *Breaks:* level targets and level diagnostics measure different, nonlinearly related
   quantities, and cross-stage radius comparisons mix them.
6. **Compatible dimensions** (see Composition). *Breaks:* the stages cannot compose as
   configured.
7. **A transductive refinement is acceptable for the intended uses.** *Breaks:* for any query
   outside $\mathcal{C}$ (new business text, a future NAICS vintage), only the unrefined
   text-stage map applies.
8. **Stage outputs are selected comparably.** The text stage is selected by an in-sample
   contrastive loss; the graph stage has no selection and exports its last epoch. *Breaks:* the
   acceptance gate compares a selected checkpoint with an unselected endpoint.
9. **A single vintage.** Everything is NAICS 2022, with no concordance to other vintages.
   *Breaks:* codes whose content changes across vintages.

## Notation

Shared symbols. Each component section extends this table.

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\mathcal{C}$ | Set of NAICS 2022 codes, all five levels | $N = 2125$ codes |
| $i$, $j$ | Generic codes | $\mathcal{C}$ |
| $a$, $p$ | Anchor and positive code of a training tuple | $\mathcal{C}$ |
| $\lambda(i)$ | Level of code $i$, its number of digits | $\{2,3,4,5,6\}$ |
| $\mathcal{T}$ | Taxonomy forest, one tree per sector, edges parent to child | 20 trees |
| $\operatorname{pa}(i)$ | Parent of $i$ | $\mathcal{C}$; sectors have none |
| $D_{ij}$ | Structural distance (C1) | $\{0.5, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 99\}$ for $i \ne j$; 0 for $i = j$ |
| $D^{\times}$ | Structural distance assigned to every cross-sector pair | 99 |
| $R_{ij}$ | Structural relation category (C1) | 14 within-sector categories plus *cross-sector* |
| $X_{i \to j}$ | The exclusion text of $i$ names code $j$ | $\{0,1\}$ |
| $X_{ij}$ | Explicit exclusion, $X_{i \to j} \lor X_{j \to i}$ | $\{0,1\}$ |
| $t^{(k)}_{i}$ | Text channel $k$ of code $i$: 1 title, 2 description, 3 illustrative examples, 4 exclusion text | token sequences |
| $t_{i}$ | The four channels of code $i$ together | |
| $n$ | Spatial dimension of the text-stage hyperbolic space | 384 |
| $\mathbb{H}^{n}$ | Hyperboloid of curvature $-1$: points $x \in \mathbb{R}^{n+1}$ with $\langle x,x\rangle_{\mathcal{L}} = -1$ and $x_{0} > 0$ | |
| $\langle x,y\rangle_{\mathcal{L}}$ | Lorentz inner product, $-x_{0}y_{0} + \sum_{k=1}^{n} x_{k}y_{k}$ | $\mathbb{R}$ |
| $x_{0}$, $x_{s}$ | Time coordinate and spatial part $(x_{1},\dots,x_{n})$ of a point $x$ | $\mathbb{R}$, $\mathbb{R}^{n}$ |
| $d(x,y)$ | Geodesic distance, $\operatorname{arcosh}\max(1, -\langle x,y\rangle_{\mathcal{L}})$ | $[0,\infty)$ |
| $o$ | Origin $(1,0,\dots,0)$ | $\mathbb{H}^{n}$ |
| $\exp_{o}$, $\log_{o}$ | Exponential and logarithmic maps at $o$; for a tangent vector $v \in \mathbb{R}^{n}$, $\exp_{o}(v) = (\cosh\Vert v\Vert,\ \sinh\Vert v\Vert\,v/\Vert v\Vert)$ | |
| $r(x)$ | Geodesic radius $d(o,x)$; $x_{0} = \cosh r$ and $\Vert x_{s}\Vert = \sinh r$ | $[0,\infty)$ |
| $c$ | Curvature magnitude | fixed at 1 |
| $F_{\Theta}$ | Text-stage map with parameters $\Theta$ (C2) | texts to $\mathbb{H}^{n}$ |
| $G_{\Xi}$ | Graph-stage network with parameters $\Xi$ (C3) | |
| $Z^{(3)}$, $Z^{(4)}$ | Text-stage and graph-stage embeddings of all codes | $N$ points |
| $e$ | Training epoch index | $0, 1, \dots$ |
| $\hat\Theta$, $\hat\Xi$, $\hat H$ | A hat marks the fitted value of a parameter, as selected or exported | |
| $M(Z)$ | Vector of structural-agreement statistics of an embedding (C4) | |
| $\Delta$ | Graph-stage value minus text-stage value of a statistic | |
| $A$ | Acceptance indicator of the graph stage (C4) | $\{0,1\}$ |

## Per-component descriptions

### C1. Taxonomy supervision construction

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $h_{i}$, $h_{j}$ | Steps from $i$ and from $j$ up to their lowest common ancestor | nonnegative integers |
| $\iota(R)$ | Integer index of relation category $R$ (table below) | $\{1,\dots,14\} \cup \{99\}$ |
| $\mathcal{X}_{a}$ | Exclusion set of $a$, the codes $j$ with $X_{aj} = 1$ | subset of $\mathcal{C}$ |
| $V(i)$ | Orientation-admissible set of code $i$ (defined below) | subset of $\mathcal{C}$ |
| $\mathcal{U}_{ap}$ | Generated candidate negatives of the pair $(a,p)$ | subset of $\mathcal{C}$ |
| $\mu^{R}_{apj}$, $\mu^{D}_{apj}$ | Relation margin and distance margin of candidate $j$ for $(a,p)$ | $\mathbb{R}$ |
| $\mathcal{P}$ | Runtime positive-pair universe | 4,220 pairs |
| $\alpha_{\mathrm{D}}$ | Exponent of the inverse-distance negative weights | 1.5 |
| $K$ | Number of negatives per training tuple | 24 |
| $\mathcal{Q}_{ap}$ | Candidate pool of a training tuple | at least $K$ codes |
| $O_{ap}$ | Ordinary (non-exclusion) members of the pool | $K$ or $K-1$ codes |
| $\chi$ | Placeholder text substituted for an empty channel | a fixed string |

#### Problem formulation and data-generating story

**Observed.** The official NAICS 2022 publications:

- the list of 2,125 codes with their titles;
- a description of each code;
- an index of illustrative business activities keyed to six-digit codes;
- cross-reference text, in which a code names activities excluded from it and the code where
  each is classified instead.

**Latent.** Nothing. C1 is a deterministic transform plus seeded sampling.

**How the data arise.** The process is editorial. The classification's authors group
establishments by similarity of production process into a five-level hierarchy (Office of
Management and Budget, NAICS 2022). They write a description for each class and use
cross-references to mark boundaries between classes. Those boundaries are very often between
neighboring classes.

**Estimands.** All are deterministic given the documents:

- a structural distance $D$ and a relation $R$ on every unordered pair, derived from tree
  positions alone;
- directional exclusion indicators $X_{i \to j}$;
- four text channels per code;
- a universe of positive pairs and, for each, a candidate list of negatives.

The sampling estimand is a distribution over training tuples $(a, p, \mathcal{Q}_{ap})$, from
which the text stage draws.

#### Estimation / inference procedure

**Code universe and trees.**

- $\mathcal{C}$ contains every code of levels 2 through 6. By level the counts are 20, 96, 308,
  689 and 1,012.
- The three combined sectors (31–33, 44–45, 48–49) are each represented by a single sector code.
  Every subsector whose first two digits fall in the combined range is its child.
- Each sector roots its own tree. There is no super-root.

**Text channels.**

- **Title**, $t^{(1)}$: the official title, verbatim.
- **Description**, $t^{(2)}$: the official description with its cross-reference paragraphs and
  illustrative-example list removed, then lightly normalized.
  - For 522 five-digit industries, the official description is only a pointer to a single
    six-digit child. Each inherits that child's description, with "industry" wording adjusted.
  - 154 four-digit codes likewise inherit a five-digit child's description. For 14 of them, one
    of several differing candidates is chosen arbitrarily.
  - As a result, each of those 522 five-digit industries shares its child's title and has a
    near-identical description.
- **Illustrative examples**, $t^{(3)}$: the code's index entries joined into one list. Index
  entries exist only for six-digit codes. A code without index entries uses its published
  illustrative-example paragraph if it has one; such paragraphs exist for 381 codes.
  - This channel is empty for every code at levels 2–4, for 624 of 689 five-digit codes, and for
    2 six-digit codes: 49.4% of codes overall.
- **Exclusion text**, $t^{(4)}$: the raw cross-reference prose (1,091 five- and six-digit codes),
  or the exclusion paragraph of the description (22 two- and three-digit codes).
  - The text is concatenated $k$ times, where $k$ is the number of codes it names.
  - It is empty for 47.6% of codes.
  - Because of the repetition, 457 of the 1,113 non-empty exclusion texts exceed the 512-token
    encoder window. Without the repetition, only 5 would.
- **Empty channels.** An empty channel is replaced by a fixed placeholder string $\chi$, which is
  then encoded like any other text.

**Exclusions.**

- $X_{i \to j} = 1$ when $i$'s exclusion text contains a whitespace-preceded run of 2–6 digits
  equal to a code $j \ne i$.
- Counts: 4,586 directed references, forming 3,954 unordered pairs (632 of them mutual) and
  involving 1,643 codes. 1,204 of the pairs cross sectors.
- A check of the whole cross-reference file found every matched reference preceded by a level
  word such as "Industry" or "Sector".
- Nine references are lineal. Eight codes name one of their own ancestors, and one names its own
  child.

**Structural distance.** Here $h_{i}$ and $h_{j}$ are the steps from $i$ and $j$ up to their
lowest common ancestor.
$$D_{ij} = \begin{cases} 0 & i = j, \\ h_{i} + h_{j} - \tfrac{1}{2} & \text{same tree, one an ancestor of the other,} \\ h_{i} + h_{j} & \text{same tree, otherwise,} \\ D^{\times} = 99 & \text{different trees.} \end{cases}$$

- The cross-sector value is a fixed constant, not a path length through a root. No derivation of
  its magnitude is recorded.
- Twelve values occur (table below). 1,984,647 of the 2,256,750 unordered pairs (87.9%) take the
  cross-sector value.
- Relation categories name the kinship of the deeper code relative to the shallower one; equal
  depths are ordered by code number. The index $\iota$ increases with $D$ with one exception:
  *sibling* ($\iota = 2$, $D = 2$) precedes *grandchild* ($\iota = 3$, $D = 1.5$).
- Exclusion processing never alters $D$ or $R$.

| $D$ | Relation categories (index $\iota$) | Unordered pairs |
|---|---|---|
| 0.5 | child (1) | 2,105 |
| 1.5 | grandchild (3) | 2,009 |
| 2.0 | sibling (2) | 2,394 |
| 2.5 | great-grandchild (4) | 1,701 |
| 3.0 | nephew or niece (5) | 7,413 |
| 3.5 | great-great-grandchild (6) | 1,012 |
| 4.0 | cousin (7); grand-nephew or grand-niece (8) | 19,144 |
| 5.0 | great-grand-nephew or great-grand-niece (9); first cousin once removed (10) | 38,841 |
| 6.0 | second cousin (11); first cousin twice removed (12) | 62,241 |
| 7.0 | second cousin once removed (13) | 70,835 |
| 8.0 | third cousin (14) | 64,408 |
| 99 | cross-sector (99) | 1,984,647 |

**Generated (stored) positives and negatives.**

- Every same-tree, non-excluded pair, in the stored orientation (shallower code first; code order
  at equal depth), is a generated positive labeled *related*. That is 269,353 pairs over 2,090
  anchors.
- For a generated positive $(a,p)$, the orientation-admissible set is
  $$V(i) = \{\, j : \lambda(j) > \lambda(i) \,\} \ \cup\ \{\, j : \lambda(j) = \lambda(i) \text{ and } (j \text{ is numerically larger than } i \text{ or has a different two-digit prefix}) \,\}.$$
  This set is an artifact of the stored orientation.
- A candidate $j \notin \{a, p\}$ is a generated negative when $j \in V(a) \cap V(p)$ and
  $$D_{aj} = D^{\times} \quad \text{or} \quad \Big(\iota(R_{aj}) > \iota(R_{ap}) \ \text{ and } \ D_{aj} \ge D_{ap} - \tfrac{1}{2}\Big).$$
- The same rule is recorded as two margins, and a candidate is kept when both are positive:
  - $\mu^{R} = \iota(R_{aj}) - \iota(R_{ap})$, or 15 for a cross-sector candidate;
  - $\mu^{D} = D_{aj} - D_{ap}$, with special values: 10 for a cross-sector candidate, and, when
    $\mu^{R} > 0$, $1/3$ where the distance difference is 0 and $2/3$ where it is $-1/2$.
- Cross-sector non-exclusion negatives are capped at 100 per pair by a deterministic hash
  ranking. Exclusions are exempt from the cap.
- Explicit exclusions are labeled *unrelated*; all other negatives are labeled *unknown*.

Consequences:

- Negatives are never shallower than the anchor or the positive.
- For sibling positives, the anchor's grandchildren ($D = 1.5 < 2$) are admissible negatives
  (4,675 stored rows).
- 75,460 rows pair a negative with a positive at the same structural distance.
- Every pair has exactly 100 cross-sector candidates, and 68.3% of pairs have no within-sector
  candidate at all.

**Runtime positives.** A stratified sampler derives the positives that training actually uses.
The graph stage uses the same sampler. It has three strata:

- **Descendants.** For an anchor at levels 2–5: its descendants at the first deeper level where
  it has more than one descendant. For five-digit anchors this is always their children. Weights
  are uniform.
- **Siblings.** For each sibling pair, only the numerically smaller code is an anchor, and the
  larger one is its positive.
- **Ancestors.** These are enumerated, but none survives. A pair is kept only if it has generated
  negatives, and generation uses the shallower-first orientation.

Sampling draws up to 4 positives per stratum per anchor per sampling round, weighted, without
replacement. The resulting universe $\mathcal{P}$ is 4,220 pairs over 1,273 anchors: all 1,113
codes at levels 2–5, but only 160 of the 1,012 six-digit codes. By relation: child 1,936,
grandchild 344, great-grandchild 17, great-great-grandchild 3, sibling 1,920.

**Runtime negatives and the candidate pool.**

- **Raw draw.** For each sampled $(a,p)$, $K = 24$ raw negatives are drawn without replacement
  from $\mathcal{U}_{ap}$, with probability proportional to $D_{aj}^{-\alpha_{\mathrm{D}}}$.
  - A sibling mask at $D = 2$ exists but never binds, because generation never admits the
    anchor's siblings.
  - The realized mix, shown below, is dominated by distant within-sector codes.
- **Pool.**
  $$\mathcal{Q}_{ap} = \big(\mathcal{X}_{a} \setminus \{a,p\}\big) \cup O_{ap}, \qquad \lvert O_{ap}\rvert = \begin{cases} K & \text{if } \mathcal{X}_{a} \setminus \{a,p\} = \varnothing, \\ K - 1 & \text{otherwise.} \end{cases}$$
  - $O_{ap}$ holds raw draws. When there are too few, it is backfilled uniformly from codes that
    satisfy the margin rule (70 of about 3,704 tuples per round).
  - The pool contains *all* of the anchor's exclusions, in either direction and regardless of
    structure. In one sampling round, 564 of 8,994 pooled exclusion entries are shallower than
    the anchor, and 50 are the anchor's own ancestor or descendant.
  - For one three-digit anchor, both of its exclusions are its own descendants.

| $D$ of raw draw | 7 | 5 | 6 | 4 | 99 | 3 | 1.5 | 8 | 2.5 | 3.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Share of draws | 21.9% | 17.7% | 17.0% | 12.5% | 10.1% | 8.0% | 5.1% | 4.8% | 2.4% | 0.6% |

**Sampling schedule and partition.**

- One sampling round yields about 3,704 tuples.
- The text stage pre-draws 100 rounds, with seeds $42 + \text{round}$, and concatenates them. That
  gives 370,320 tuples, with each pair appearing about 88 times with different negatives. The
  same tuples are reused in every training epoch.
- A validation set is built identically, with seeds $1042 + \text{round}$.
- No partition of codes, subtrees, or pairs exists anywhere in the method.

**Integrity.** All derived supervision forms one immutable, versioned bundle carrying
fingerprints of its inputs. Consumers refuse mixed or inconsistent bundles.

#### Assumptions and limitations

1. **$D$ measures relatedness.** Path length in the authors' production-process tree stands in
   for relatedness. *Breaks:* relatedness through products, markets, inputs, or co-movement is
   invisible, and the target inherits every editorial partition choice.
2. **All cross-sector pairs are equally unrelated, at 99.** *Breaks:* related industries in
   different sectors (manufacturing and wholesaling the same goods, for instance) get the same
   target as unrelated ones. The authors' own 1,204 cross-sector exclusion pairs mark such
   boundaries. The magnitude, about twelve times the largest within-sector value, dominates every
   magnitude-sensitive consumer: the text stage's distance-matching loss, and the Pearson and
   NDCG statistics of C4.
3. **Lineal pairs are half a step closer than their path length.** *Breaks:* it is an unexplained
   convention that orders lineal pairs ahead of collateral pairs of equal path length.
4. **An exclusion reference means "unrelated".** *Breaks:* cross-references mark boundaries, and
   boundaries usually lie between near neighbors: 474 of the 2,394 sibling pairs are exclusions.
   The nine lineal exclusions tell a code to repel its own ancestor or descendant.
5. **Pattern-matched references are complete and correct.** Any whitespace-preceded 2–6-digit run
   equal to a code counts. *Breaks:* missed or spurious references go undetected. The check
   above found no spurious matches.
6. **Channel texts are comparable across codes.** *Breaks:*
   - Channel availability is set by level: examples are absent for every code at levels 2–4.
     Channel presence therefore leaks level.
   - Inherited descriptions make 522 parent–child pairs near-duplicates.
   - Repetition makes exclusion-text length scale with the number of referenced codes and
     truncates 41% of it.
7. **Positive supervision covers the hierarchy evenly.** *Breaks:*
   - Siblings supervise in one direction only, and ancestors never do.
   - 852 of the 1,012 six-digit codes are never anchors. Most leaves are therefore supervised only
     as someone else's positive or negative.
   - Which codes act as anchors depends on code numbering.
8. **Negative eligibility is consistent with $D$.** *Breaks:*
   - Eligibility is keyed on the relation index, which disagrees with $D$ for sibling versus
     grandchild. 4,675 stored rows pair a sibling positive with a structurally *closer*
     grandchild negative. The inverse-distance weights then favor such negatives: at
     $1.5^{-1.5} \approx 0.54$, grandchildren carry the largest weight of any eligible
     distance.
   - The orientation artifact $V$ means shallower codes, and numerically smaller same-level codes
     with the same prefix, are never negatives.
9. **Cross-sector negatives are adequately represented.** There are 100 per pair, each with
   weight $99^{-1.5} \approx 0.001$, together about 10% of draws. *Breaks:* cross-sector
   confusions are rarely trained against, yet C4's statistics are dominated by cross-sector
   pairs.
10. **No holdout is needed.** *Breaks:* nothing downstream can measure generalization to unseen
    codes, subtrees, or texts.

#### Evaluation criteria

Construction is judged only by internal-consistency invariants, each checked before any training:

- exclusion processing never alters structure;
- directional exclusion provenance survives, and the symmetric indicator equals the disjunction;
- the pairwise and matrix views of $D$ and $R$ reconcile;
- every training record joins to known codes and pair facts;
- no exclusion occupies a positive slot;
- regeneration from the same inputs is deterministic, up to bundle identifiers.

No criterion assesses whether $D$, or the positives and negatives derived from it, agrees with
any external notion of industry relatedness.

#### Open questions for the reviewer

1. Is tree path length, with the half-step lineal adjustment, a defensible target metric for
   industry relatedness? What does the literature on taxonomy-based similarity (information
   content, depth-scaled measures, learned metrics) suggest instead?
2. How should cross-sector pairs be encoded? Options include a constant sentinel, a finite path
   through a virtual root, a data-driven value, or exclusion from magnitude-sensitive losses and
   statistics.
3. What is the right role for "excluded activities" cross-references? Should they be hard
   negatives, a distinct boundary relation, pair-level constraints, or text-only evidence? How
   should lineal exclusions be treated?
4. Is this positive set a sound sampling of the hierarchy for contrastive learning? It consists of
   descendants at the first branching level plus forward siblings, with no ancestors, and most
   leaves are never anchors. What sampling designs do hierarchical contrastive methods use?
5. Should empty channels and inherited descriptions be handled differently, for example by
   masking, channel-presence indicators, or deduplication?

### C2. Text-conditioned hyperbolic contrastive encoder (text stage)

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\theta_{0}$ | Frozen pretrained encoder weights; each channel has its own copy | |
| $\phi_{k}$ | Low-rank adapter parameters of channel $k$ | rank 8 |
| $\Psi_{k}$ | Final-layer token states of the adapted encoder of channel $k$ | $\mathbb{R}^{n}$ per token |
| $u^{(k)}_{i}$ | Pooled embedding of $t^{(k)}_{i}$ | $\mathbb{R}^{n}$ |
| $q_{i}$ | Concatenation of the four pooled embeddings | $\mathbb{R}^{4n}$ |
| $W_{g}$ | Gate weight matrix | 4 rows |
| $\gamma_{i}$ | Gate probabilities over the four experts | probability vector of length 4 |
| $\mathcal{S}_{i}$ | Indices of the two largest gate logits | 2 experts |
| $\Phi_{b}$ | Expert network $b$, with weight matrices $W_{1,b}$, $W_{2,b}$ | $\mathbb{R}^{4n}$ to $\mathbb{R}^{4n}$ |
| $y_{i}$ | Fused vector | $\mathbb{R}^{4n}$ |
| $v_{i}$, $\bar v_{i}$ | Tangent vector at $o$ before and after the norm cap | $\mathbb{R}^{n}$ |
| $\bar\nu$ | Tangent-norm cap | 2 |
| $z_{i}$ | Text-stage point of code $i$, $F_{\Theta}(t_{i})$ | $\mathbb{H}^{n}$ |
| $\eta(a)$ | Fixed hash of the training seed and the anchor | integers |
| $\mathcal{N}_{ap}$ | Selected negatives of a tuple | $K$ codes |
| $\tilde{\mathcal{N}}_{ap}$ | Selected negatives eligible for the contrastive term | subset of $\mathcal{N}_{ap}$ |
| $B$ | Micro-batch size | 16 tuples |
| $B'$ | Tuples in the micro-batch with at least one eligible negative | at most $B$ |
| $\tau$ | Contrastive temperature | 0.07 |
| $\mathcal{L}_{\mathrm{C}}$ | Decoupled contrastive loss | |
| $\mathcal{L}_{\mathrm{H}}$ | Hierarchy (distance-matching) loss | |
| $\mathcal{L}_{\mathrm{P}}$ | Pairwise structural-preference loss | |
| $\mathcal{L}_{\mathrm{R}}$ | Radius penalty | |
| $\mathcal{L}_{\mathrm{V}}$ | Level-radius alignment loss | |
| $\mathcal{L}_{\mathrm{B}}$ | Expert load-balancing loss | |
| $\mathcal{L}^{(3)}$ | Total text-stage objective | |
| $\bar d$, $\bar D$ | Mean learned and mean structural distance over the pairs of one micro-batch | |
| $f_{b}$, $P_{b}$ | Routed fraction and mean gate probability of expert $b$ | $[0,1]$ |
| $\psi(u, t)$ | Huber function with threshold 1: $\tfrac12(u-t)^{2}$ if $\lvert u-t\rvert < 1$, else $\lvert u-t\rvert - \tfrac12$ | $[0,\infty)$ |
| $\bar{\mathcal{L}}^{\mathrm{val}}$ | Validation contrastive loss | |
| $\vartheta_{ij}$ | Angle between the spatial directions of $z_{i}$ and $z_{j}$ | $[0,\pi]$ |

#### Problem formulation and data-generating story

**Observed.** Per code, the four token sequences $t^{(1)}_{i},\dots,t^{(4)}_{i}$. **Supervision.**
The tuples, $D$, and $X$ from C1.

There is no probabilistic model of the data. The estimand is defined implicitly as the minimizer
of an empirical objective: a map $F_{\Theta}$ under which, for sampled tuples,

- the anchor lies closer to its positive than to its negatives (contrastive term);
- pairwise geodesic distances are proportional to $D$ (hierarchy term);
- candidates are ordered by their structural distances (preference term);
- radius grows with level (level term).

Because $F_{\Theta}$ is a function of text, it is inductive: any new text maps into
$\mathbb{H}^{n}$.

#### Estimation / inference procedure

**Channel encoding.**

- **Backbone.** Each channel $k$ has its own copy of a frozen 6-layer, 384-dimensional sentence
  encoder: the published all-MiniLM-L6-v2 checkpoint (Wang et al. 2020; Reimers & Gurevych
  2019).
- **Adapters.** Each copy has its own low-rank adapters $\phi_{k}$ on every linear layer of its
  attention and feed-forward blocks (Hu et al. 2022): rank 8, scale 2, dropout 0.1.
- **Token windows.** 24 tokens for titles and 512 for the other channels. The checkpoint's native
  window is 256.
- **Pooling.** The channel embedding is the attention-masked mean of final-layer token states,
  special tokens included, with no normalization:
  $$u^{(k)}_{i} = \operatorname{mean}_{\text{tokens}}\ \Psi_{k}\big(t^{(k)}_{i}\big).$$

**Fusion (sparse mixture of experts; Shazeer et al. 2017).** $q_{i}$ is the concatenation of
the four channel embeddings.
$$\gamma_{i} = \operatorname{softmax}(W_{g} q_{i}), \qquad y_{i} = \sum_{b \in \mathcal{S}_{i}} \frac{\gamma_{i,b}}{\sum_{b' \in \mathcal{S}_{i}} \gamma_{i,b'}}\ \Phi_{b}(q_{i}), \qquad \Phi_{b}(q) = W_{2,b}\,\mathrm{ReLU}(W_{1,b}\,q).$$

- The expert hidden width is 768, with dropout 0.1.
- There is no gating noise and no capacity limit.

**Projection onto the hyperboloid.**

- Two successive affine maps take $y_{i}$ to $v_{i} \in \mathbb{R}^{n}$. With nothing nonlinear
  between them, they are equivalent to one.
- The vector is capped and mapped to the hyperboloid:
  $$\bar v_{i} = v_{i} \min\!\left(1, \frac{\bar\nu}{\Vert v_{i}\Vert}\right), \qquad z_{i} = F_{\Theta}(t_{i}) = \exp_{o}(\bar v_{i}), \qquad r(z_{i}) = \Vert \bar v_{i}\Vert \le \bar\nu = 2.$$
- Every text-stage point therefore lies in the closed geodesic ball of radius 2, and every
  pairwise distance lies in $[0,4]$.
- Trainable parameters are the adapters, gate, experts, and projections: 11.5 million of 102.4
  million.

**Negative selection.**

- **Reserved exclusion.** For each tuple $(a, p, \mathcal{Q}_{ap})$ with
  $\mathcal{X}_{a} \cap \mathcal{Q}_{ap} \ne \varnothing$, exactly one exclusion is reserved. With
  the pooled exclusions sorted by code, it is the one at index
  $(\eta(a) + e) \bmod \lvert \mathcal{X}_{a} \cap \mathcal{Q}_{ap}\rvert$, so the choice rotates
  across epochs.
- **Remaining slots.** The other $K - 1$ slots (all $K$ when no exclusion is pooled) are filled
  from $O_{ap}$.
- **The set is fixed.** $\lvert O_{ap}\rvert$ equals the number of remaining slots, so **every
  ordinary pool member is selected**. The curriculum's mining rules (below) can reorder
  $\mathcal{N}_{ap}$ but never change it.
- **No sharing.** Negatives belong to their tuple; there is no in-batch sharing.

**Objective.** Terms are computed over a micro-batch of $B = 16$ tuples.

*Contrastive, decoupled* (Yeh et al. 2022). The eligible set $\tilde{\mathcal{N}}_{ap}$ is the
selected negatives minus pseudo-related non-exclusions. No negative is pseudo-related under the
reference (see the curriculum below), so $\tilde{\mathcal{N}}_{ap} = \mathcal{N}_{ap}$. Over the
$B'$ tuples with at least one eligible negative:
$$\mathcal{L}_{\mathrm{C}} = \frac{1}{B'} \sum_{(a,p)} \left[ \frac{d(z_{a}, z_{p})}{\tau} + \log \sum_{j \in \tilde{\mathcal{N}}_{ap}} \exp\!\left(-\frac{d(z_{a}, z_{j})}{\tau}\right) \right].$$
The positive is excluded from the log-sum-exp, so the term is unbounded below.

*Hierarchy, or distance matching.* Over all pairs $(i,j)$ among the $2B$ anchors and positives
of the micro-batch with $D_{ij} \ge 0.1$:
$$\mathcal{L}_{\mathrm{H}} = \operatorname{mean}_{(i,j)} \left( \frac{d(z_{i}, z_{j})}{\bar d} - \frac{D_{ij}}{\bar D} \right)^{2}.$$
- Only distance ratios are constrained.
- Most pairs in a micro-batch cross sectors; by estimate, roughly 85% for a randomly composed
  batch. Normalized within-sector targets then fall in roughly $[0.006, 0.09]$, against roughly
  1.2 for cross-sector pairs, so the term is close to a two-level target: same sector or not.

*Structural preference.* This replaced an earlier listwise ranking loss. It runs over the
positive and the non-exclusion selected negatives of each tuple. Take every unordered pair
$\{i,j\}$ of distinct codes with $\lvert D_{ai} - D_{aj}\rvert > 10^{-6}$, oriented so that
$D_{ai} < D_{aj}$, with margin 0.1 and temperature 1:
$$\operatorname{softplus}\big( d(z_{a}, z_{i}) - d(z_{a}, z_{j}) + 0.1 \big),$$
averaged per anchor, then over anchors with at least one pair. Its gradient pulls the
structurally closer member in and pushes the farther member out.

*Radius penalty.* $\mathcal{L}_{\mathrm{R}}$ is the mean of $\max(0, \Vert z_{s}\Vert - 10)^{2}$
over anchors, positives, and selected negatives. **Inert:** under the cap,
$\Vert z_{s}\Vert = \sinh r \le \sinh 2 \approx 3.63$, so the term is identically zero.

*Level alignment.* Over anchors and positives:
$$\mathcal{L}_{\mathrm{V}} = \operatorname{mean}_{i}\ \psi\!\left( \Vert z_{i,s}\Vert,\ \frac{\lambda(i) - 2}{2} \right).$$
This targets $\sinh r \in \{0, 0.5, 1, 1.5, 2\}$ for levels 2–6, which are geodesic radii of 0,
0.48, 0.88, 1.20, and 1.44.

*Load balancing* (Fedus et al. 2022). Over every row encoded in the micro-batch (anchors,
positives, and all valid pool members):
$$\mathcal{L}_{\mathrm{B}} = 4 \sum_{b=1}^{4} f_{b} P_{b}, \qquad f_{b} = \text{share of rows with } b \in \mathcal{S}_{i}, \qquad P_{b} = \operatorname{mean}_{i} \gamma_{i,b}.$$
$f_{b}$ is a count and passes no gradient. Under perfect balance the term equals 2.

*Total.*
$$\mathcal{L}^{(3)} = \mathcal{L}_{\mathrm{C}} + 0.01\,\mathcal{L}_{\mathrm{B}} + 0.45\,\mathcal{L}_{\mathrm{H}} + 0.35\,\mathcal{L}_{\mathrm{P}} + 0.10\,\mathcal{L}_{\mathrm{R}} + 0.15\,\mathcal{L}_{\mathrm{V}}.$$
No weight is learned or phase-dependent.

**Optimization.**

- **Optimizer.** AdamW with learning rate $10^{-4}$, weight decay 0.01, and a gradient-norm clip
  of 1.
- **Updates.** Micro-batches of 16 with 2-step accumulation give 32 tuples per update. One epoch
  is one pass over the 370,320 pre-drawn tuples, about 11,600 updates.
- **Length.** At most 10 epochs, seed 42.
- **Learning rate in practice.** The rate halves after *more than* 3 epochs without improvement
  in $\bar{\mathcal{L}}^{\mathrm{val}}$, but early stopping ends training after 3 such epochs. The
  rate is therefore effectively constant.
- **Export.** The model exported is the epoch with the lowest $\bar{\mathcal{L}}^{\mathrm{val}}$.
  Export is deterministic, with dropout off.

**Validation loss.** $\bar{\mathcal{L}}^{\mathrm{val}}$ is $\mathcal{L}_{\mathrm{C}}$ on the
validation tuples, with every eligible pool member as a negative:

- *all* pooled exclusions (mean about 2.4, maximum 27) rather than one;
- no rotation;
- no pseudo-labels.

**Curriculum (designed as three phases; inert under the reference).** Phases are set by epoch:
phase 1 is epochs 0–5, phase 2 is epochs 6–7, and phase 3 is epochs 8–9. The boundaries are at
fractions 0.5 and 0.7 of 10 epochs.

- **Phase 1.** Selection is the reserved exclusion plus pool order. The inverse-distance
  weighting of C1 is applied when tuples are pre-drawn, so it holds in every phase, not only
  this one.
- **Phases 2–3.** Two proposal rules fill the non-exclusion slots, in order:
  - a geometric rule takes half the slots: the eligible candidates with the smallest
    $d(z_{a}, z_{j})$;
  - a routing rule takes the rest: the candidates whose gate distribution is closest to the
    anchor's, scored by $-\mathrm{KL}(\gamma_{a} \,\Vert\, \gamma_{j})$ (Kullback–Leibler
    divergence).

  **Inert:** the pool equals the slots, so neither rule can change the selected set.
- **Phase 3, false-negative mitigation.** Anchors are periodically clustered by hyperbolic
  k-means:
  - k-means++ initialization, with assignment by $d$;
  - an origin-anchored approximate centroid that is biased toward $o$;
  - at most 80 clusters, over at most 1,600 clustered anchors.

  A non-exclusion negative in the anchor's cluster is removed from the contrastive log-sum-exp.
  An alternative setting instead adds an attraction term. **Inert:** re-clustering happens at
  multiples of 5 epochs after phase 3 begins. Within 10 epochs that never occurs, so no
  pseudo-labels exist.
- **A margin that enters no loss.** A per-anchor margin $0.5\,\operatorname{sech}(r(z_{a}))$ is
  computed and logged only.

Two settings outside the reference make the pool larger than $K$. One draws 48 raw candidates
per tuple, with a linear easy-to-hard schedule over distance buckets during phase 1. The other
is multi-device training, where the pool is the union across devices. Only in these settings do
the mining rules change the selected set.

#### Assumptions and limitations

1. **The norm cap is a numerical safeguard that seldom binds.** *Breaks, and has been observed to
   break.*
   - **Observation.** The only retained historical run was trained under an earlier,
     since-repaired supervision contract with the same geometry. By its second epoch every point
     sat on the cap: mean $x_{0} = 3.7621958 = \cosh 2$, standard deviation $4 \times 10^{-7}$.
   - **Why it persists.** On the cap $r \equiv 2$. The output depends on $v$ only through
     $v / \Vert v\Vert$, so no gradient reaches the norm, and nothing in the current objective
     pulls points back inside.
   - **Consequence.** Distances satisfy
     $\cosh d(z_{i}, z_{j}) = \cosh^{2} 2 - \sinh^{2} 2 \,\cos\vartheta_{ij}$, a monotone function
     of angle alone. The representation carries only angular (spherical) information,
     $\mathcal{L}_{\mathrm{V}}$ has zero gradient, and the radial encoding of depth that
     motivates hyperbolic geometry is unavailable.
2. **The loss terms are mutually consistent.** *Breaks:*
   - For sibling positives, the anchor's grandchildren are negatives. $\mathcal{L}_{\mathrm{C}}$
     repels them, while $\mathcal{L}_{\mathrm{P}}$, ordering by $D$ ($1.5 < 2$), pulls them inside
     the positive.
   - $\mathcal{L}_{\mathrm{H}}$ is dominated by the cross-sector constant.
   - Two of the six terms are inert.
3. **Negative selection adapts to the model.** *Breaks:* under the reference, $\mathcal{N}_{ap}$
   is fixed at pre-draw time, apart from the rotating exclusion. The three-phase curriculum does
   not change the objective.
4. **Validation measures generalization.** *Breaks:*
   - Validation tuples use the same 1,273 anchors and the same 4,220 pairs.
   - 89.1% of validation (anchor, positive, negative) triples also occur in training.
   - Model selection, early stopping, and learning-rate control all read this in-sample loss,
     which also weights exclusions differently from training.
5. **Channels contribute independent information.** *Breaks:*
   - The placeholder $\chi$ for empty channels itself signals level, because examples are empty
     for every code at levels 2–4.
   - Each channel adapts separately, so nothing learned from one channel's text transfers to
     another.
6. **The input windows suffice.** *Breaks:* 512-token windows exceed the checkpoint's native
   256-token regime. Even so, the repetition of exclusion text pushes 41% of it past 512 tokens,
   where it is truncated.
7. **Load balancing reflects routing over codes.** *Breaks:* it is computed over every encoded row
   of the micro-batch, a mixture of anchors, positives, and negatives, not over codes.
8. **One run is representative.** *Breaks:* there is one seed and one run, with no uncertainty
   quantification.

#### Evaluation criteria

- **Model selection:** the lowest $\bar{\mathcal{L}}^{\mathrm{val}}$, an in-sample statistic (see
  limitation 4).
- **Reported but not used for selection:**
  - the C4 structural statistics, on 300 codes drawn uniformly at random each epoch from the 1,273
    validation anchors (only 160 of the 1,012 six-digit codes can ever be drawn);
  - manifold validity, $\lvert\langle z,z\rangle_{\mathcal{L}} + 1\rvert < 10^{-3}$;
  - per-level mean $x_{0}$;
  - Euclidean collapse diagnostics on spatial coordinates.
- **Contract properties verified independently of training:**
  - the preference term's gradient moves an inverted pair toward the correct order;
  - each selected negative's embedding, code, structural facts, exclusion flags, and routing
    output stay aligned through selection;
  - supervision artifacts fail closed.
- **Per-epoch selection-health counts:** exclusions reserved, backfills, duplicates removed, and
  ineligible candidates.

#### Open questions for the reviewer

1. Given the observed saturation at the norm cap, is hyperbolic geometry doing any work here?
   - Which parameterizations or objectives keep depth expressible along the radius? Candidates
     include hyperbolic entailment cones, radius-targeted or Busemann-type objectives, and
     uncapped exponential maps with learned scaling.
   - Is 384 spatial dimensions a regime where hyperbolic geometry helps at all?
2. Is a six-term objective, two of whose terms are inert, justified? Or should the method
   converge on one principled hierarchy-aware contrastive loss for multi-level positives, and if
   so, which does the literature recommend?
3. How should negatives be chosen for a 2,125-code taxonomy? Options include fixed
   inverse-distance draws, real hard-negative mining over a larger pool, or every code as a
   negative, since the code set is small enough to keep all of its embeddings in a memory
   bank.
4. Is a sparse mixture of experts over four channel embeddings (top 2 of 4 experts) justified
   against simpler fusion? Alternatives include concatenation, attention pooling, or one encoder
   over the concatenated fields, especially given that two channels are placeholders for half the
   codes.
5. Is a 6-layer, 384-dimensional encoder with per-channel adapters appropriate? Alternatives
   include current general-purpose embedding models, a shared encoder with channel markers, or a
   frozen encoder with a light trained head.
6. What validation design would make model selection meaningful here?

### C3. Graph-convolutional hyperbolic refinement (graph stage)

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\mathcal{G} = (\mathcal{C}, \mathcal{E})$ | Refinement graph with edge set $\mathcal{E}$ | $N$ nodes |
| $\kappa(i,j)$ | Edge type: child, grandchild, great-grandchild, sibling, or self | 5 types |
| $\omega_{ij}$ | Edge weight | $(0, 1]$ |
| $\mathcal{B}(i)$ | Neighbors of $i$ in $\mathcal{G}$, including $i$ | |
| $n'$ | Graph-stage spatial dimension; the ambient dimension is $n' + 1$ | 30 in the reference |
| $H$ | Free node states, one row per code | $N$ rows of length $n'+1$ |
| $\ell$ | Layer index | $\{1, 2\}$ |
| $x_{i}$, $x'_{i}$ | Point of code $i$ entering and leaving a layer; after the last layer, $x_{i}$ denotes the output | $\mathbb{R}^{n'+1}$ |
| $[w]_{s}$ | Coordinates $1,\dots,n'$ of a vector $w$ (coordinate 0 dropped) | $\mathbb{R}^{n'}$ |
| $\mathrm{MLP}_{\ell}$ | Attention score network of layer $\ell$ | |
| $W_{\ell}$, $b_{\ell}$ | Square linear map and bias of layer $\ell$ | |
| $\epsilon_{\kappa}$ | Learned embedding of edge type $\kappa$ | $\mathbb{R}^{n'+1}$ |
| $\beta_{\ell}$ | Learned sibling score bonus of layer $\ell$ | initially 0.15 |
| $\upsilon_{i}$, $\hat\upsilon_{i}$, $\bar\upsilon_{i}$ | Tangent vector, its linear transform, and the neighborhood aggregate | $\mathbb{R}^{n'+1}$ |
| $\sigma_{ij}$ | Attention score of edge $(i,j)$ | $\mathbb{R}$ |
| $\alpha_{ij}$ | Attention coefficient of edge $(i,j)$ | $(0,1)$ |
| $\mathrm{LN}_{\ell}$ | Layer normalization with learned gain and shift | |
| $\mathcal{N}^{\mathrm{g}}_{ap}$ | Graph-stage negatives of a pair | 48 codes |
| $m_{a}$ | Adaptive triplet margin | $[0.4, 2]$ |
| $\zeta_{e}$ | Margin scale in epoch $e$ | 0.5 or 1 |
| $T_{e}$ | Temperature in epoch $e$ | about 0.07 to 0.10 |
| $\pi_{aj}$ | Weight of negative $j$ for anchor $a$ | sums to 1 over $j$ |
| $\mathcal{L}_{\mathrm{T}}$, $\mathcal{L}^{\mathrm{g}}_{\mathrm{V}}$ | Graph-stage triplet loss and level loss (batch means) | |
| $s_{1}$, $s_{2}$ | Learned log-variance loss weights | $\mathbb{R}$ |
| $\mathcal{L}^{(4)}$ | Total graph-stage objective | |

#### Problem formulation and data-generating story

**Observed.** $Z^{(3)}$ and the taxonomy.

**Estimand.** Node positions $Z^{(4)}$ that minimize a triplet-plus-level objective. They are
produced by two message-passing layers (in the lineage of hyperbolic graph convolution; Chami et
al. 2019) from free node states initialized at $Z^{(3)}$.

- Because the node states are free parameters, the network is a reparameterization of a free
  embedding with a smoothing inductive bias.
- It has no text input and cannot place a new code.
- Nothing ties $Z^{(4)}$ to $Z^{(3)}$ beyond initialization.

#### Estimation / inference procedure

**Graph.** Edges are undirected, with these weights:

| Edge type | Weight |
|---|---|
| parent–child | 1 |
| sibling | 0.708 |
| grandparent–grandchild | 0.5 |
| great-grandparent–great-grandchild | 0.3125 |
| self-loop | 1 |

- Each weight is a per-type base scaled by $1/(1 + 0.2(\iota - 1))$.
- There are about 18,500 directed edges, self-loops included.
- Under the current supervision, 481 explicitly excluded pairs are among these edges
  (cross-component assumption 3).
- The graph stage's reference configuration still reads an earlier supervision version. That
  version encoded exclusions as a separate structural relation, so excluded pairs were not edges.
  This section describes the graph stage under the current supervision.

**Layer $\ell = 1, 2$.** The curvature is nominally learnable per layer but effectively 1, because
no gradient reaches it.
$$\begin{aligned}
\upsilon_{i} &= \log_{o}(x_{i}) && \text{(time coordinate 0)} \\
\hat\upsilon_{i} &= W_{\ell}\,\upsilon_{i} + b_{\ell} \\
\sigma_{ij} &= \mathrm{MLP}_{\ell}\big([\hat\upsilon_{i} \,\Vert\, \hat\upsilon_{j} \,\Vert\, \epsilon_{\kappa(i,j)}]\big) + \log \omega_{ij} + \beta_{\ell}\,\mathbb{1}[\kappa(i,j) = \text{sibling}] \\
\alpha_{ij} &= \operatorname{softmax}_{j \in \mathcal{B}(i)}\ \sigma_{ij} \\
\bar\upsilon_{i} &= \sum_{j \in \mathcal{B}(i)} \omega_{ij}\,\alpha_{ij}\,\big(\hat\upsilon_{j} + \epsilon_{\kappa(i,j)}\big) \\
x'_{i} &= \exp_{o}\Big(\big[\mathrm{LN}_{\ell}\big(\upsilon_{i} + \operatorname{dropout}(\bar\upsilon_{i})\big)\big]_{s}\Big)
\end{aligned}$$

- The score network has one hidden layer of 64 units, with a SiLU activation.
- There is no node-wise nonlinearity.
- The edge weight enters twice: in the score, as $\log \omega$, and in the message, as a
  multiplier. The aggregate is therefore a sub-convex combination whose total mass depends on a
  node's mix of edge types.
- The residual adds the pre-linear tangent vector.
- The time coordinate of the normalized vector is discarded before the exponential map.
- Dropout is 0.1.
- The node states $H$ are unconstrained vectors in $\mathbb{R}^{n'+1}$, initialized on the
  hyperboloid. Each layer reads them only through $\log_{o}$.

**Training examples.**

- Positives come from the C1 sampler: one sampling round of descendant and sibling pairs, with no
  ancestors, as in C1.
- Each positive gets 48 negatives, drawn uniformly without replacement from $\mathcal{U}_{ap}$.
  About 54% of them are cross-sector.
- The examples are drawn once and fixed for all epochs.
- Exclusion status and margins are not used.

**Objective.** Distances $d$ are at $c = 1$ on the current forward output.
$$\mathcal{L}_{\mathrm{T}} = \operatorname{mean}_{(a,p)} \sum_{j \in \mathcal{N}^{\mathrm{g}}_{ap}} \pi_{aj}\, \frac{\max\big(0,\ d(x_{a}, x_{p}) - d(x_{a}, x_{j}) + m_{a}\big)}{T_{e}}.$$

- **Negative weights.** $\pi_{aj} = 1/48$ before epoch 4. From epoch 4,
  $\pi_{aj} \propto \exp(-d(x_{a}, x_{j}) / 0.5)$, restricted to the 24 closest negatives.
- **Margin.**
  $$m_{a} = \zeta_{e} \cdot \operatorname{clip}_{[0.4,\,2]}\Big( 1 + 0.5\big(1 - \tanh\tfrac{d(x_{a}, x_{p})}{4}\big) + 0.2\big(1 - \tanh\tfrac{\sinh r(x_{a})}{6}\big) \Big).$$
  - Before scaling it lies in $(1, 1.7]$, so the clip never binds.
  - Gradients flow through it.
  - $\zeta_{e} = 0.5$ in epochs 0–1 and 1 afterwards.
- **Temperature.** $T_{e}$ falls linearly from 0.10 to 0.08 over the 8 epochs and is multiplied by
  0.9 from epoch 4. It only rescales the hinge.
- **Level term.** $\mathcal{L}^{\mathrm{g}}_{\mathrm{V}}$ is the mean of
  $\psi(\sinh r(x_{i}), (\lambda(i) - 2)/2)$ over the distinct nodes of the batch.
- **Combination.** The two terms are combined with homoscedastic-uncertainty weights (Kendall et
  al. 2018):
  $$\mathcal{L}^{(4)} = \tfrac12 e^{-s_{1}}\,\mathcal{L}_{\mathrm{T}} + \tfrac12 e^{-s_{2}}\,\mathcal{L}^{\mathrm{g}}_{\mathrm{V}} + \tfrac12 (s_{1} + s_{2}).$$

**Curriculum as executed.** There are three epoch-indexed states:

- warm-up, epochs 0–1: $\zeta = 0.5$;
- expansion, epochs 2–3: no change;
- discrimination, epochs 4–7: hard-negative weighting, and temperature multiplied by 0.9.

Two designed elements do not run:

- The relation- and distance-based sample filters are no-ops, because the examples carry no
  relation or distance information.
- A separately designed four-phase controller, with adaptive loss, specialized samplers, and
  teacher distillation, is not part of the executed method.

**Optimization.**

- AdamW, with learning rate $7.5 \times 10^{-4}$ for the layers and $7.5 \times 10^{-5}$ for the
  node states, and weight decay $10^{-5}$.
- Per-epoch learning-rate multipliers: 1, 1, 0.95, 0.81, 0.61, 0.39, 0.19, 0.05.
- Gradient clip 1, batch 64, 8 epochs, about 55 updates per epoch. Every update runs the full
  graph.
- The last epoch's output is exported. There is no checkpoint selection.

**Validation.**

- **Examples.** The last 5% of examples in anchor-code order. That tail holds anchors from only
  two sectors, "Other Services" and "Public Administration".
- **Setup.** They are evaluated with the full graph, and every node state remains trainable.
- **Statistics.**
  - per batch: triplet accuracy, the share with
    $d(x_{a}, x_{p}) < \min_{j \in \mathcal{N}^{\mathrm{g}}_{ap}} d(x_{a}, x_{j})$, and
    the mean rank of the positive;
  - once per epoch: the C4 statistics over all codes.

#### Assumptions and limitations

1. **Initialization carries the text information forward.** *Breaks:* no term retains the
   $Z^{(3)}$ distances. The objective rewards only taxonomy triplets and radius targets, so
   refinement can overwrite distinctions learned from text at no cost.
2. **Normalization preserves geometry.** *Breaks:*
   - $\mathrm{LN}_{\ell}$ fixes the norm of each tangent output near its gain, about
     $\sqrt{n'+1}$ at initialization, whatever the input radius. Radius is therefore not
     inherited from $Z^{(3)}$.
   - At the text stage's dimension ($n'+1 = 385$), initial radii are about 19.6 and $x_{0}$ reaches
     about $10^{8}$ after two layers. That is beyond single-precision resolution for distances.
   - At 31 dimensions, radii are about 5.5.
3. **Curvature is learned.** It is nominal: $c \equiv 1$.
4. **The curriculum targets difficulty.** It is mostly inert (see the procedure).
5. **Excluded pairs are not neighbors.** *Breaks:* when structurally adjacent, they are edges
   (cross-component assumption 3).
6. **Validation is informative.** *Breaks:* the validation tail shares the full graph and the
   trainable node states with training, and it covers two sectors. No model selection happens.
7. **The examples are representative.** *Breaks:* a single fixed draw of positives and negatives
   serves every epoch, and uniform negatives are 54% cross-sector.
8. **Temperature and loss weights act distinctly.** *Breaks:* $T_{e}$ is a pure scale on a hinge.
   The learned $s_{1}$ can absorb any scale, and the optimizer normalizes step sizes. The
   schedule's only lasting effect is on the relative weight of the triplet and level terms while
   $s_{1}$ adapts.
9. **The refinement is transductive.** *Breaks:* there is no refined point for new text or new
   codes.
10. **The stage composition is defined.** *Breaks:* see Composition (dimension).

#### Evaluation criteria

- Per-batch triplet accuracy and the mean rank of the positive, on the validation tail.
- Per-epoch C4 statistics over all codes, at $c = 1$.
- The C4 acceptance gate, relative to the text stage.

#### Open questions for the reviewer

1. The text-conditioned encoder is already trained on the same hierarchy. What can a transductive
   graph stage add? What minimal controls would show that it adds anything? Candidates: simple
   neighbor smoothing, a matched run of extra text-stage optimization, and text shuffled among
   nodes.
2. Should refinement retain the text geometry explicitly, through distillation or a retention
   penalty? How does the hyperbolic graph-network literature do this?
3. Is message passing in the tangent space at the origin, followed by layer normalization,
   appropriate? The alternatives are fully hyperbolic (Lorentz-native) layers, or hyperbolic
   attention that preserves radius.
4. Should the graph contain only parent–child edges, the classical setting, or the added sibling,
   grandparent, and great-grandparent edges as well? How should exclusions enter?
5. Is a triplet hinge with an adaptive margin and uncertainty weighting sensible, compared with
   the distortion-minimizing or ranking objectives standard in hierarchy embedding?

### C4. Evaluation protocol

#### Notation (extension)

| Symbol | Meaning | Domain / units |
|---|---|---|
| $\mathcal{C}_{\mathrm{ev}}$ | Codes under evaluation | all of $\mathcal{C}$, or a 300-code sample |
| $\Pi$ | Unordered distinct pairs of $\mathcal{C}_{\mathrm{ev}}$ with $D_{ij} \ge 0.1$ | |
| $\rho_{\mathrm{S}}$ | Spearman rank correlation of $d$ and $D$ over $\Pi$ | $[-1, 1]$ |
| $\rho_{\mathrm{P}}$ | Pearson correlation of $d$ and $D$ over $\Pi$ | $[-1, 1]$ |
| $g_{ij}$ | Graded relevance of code $j$ to query $i$ | $[0, 1]$ |
| $D^{\max}_{i}$ | Largest structural distance among query $i$'s candidates | 99 in practice |
| $\mathrm{NDCG}@k$ | Normalized discounted cumulative gain at cutoff $k$ | $[0, 1]$ |
| $\mathrm{PR}@k$ | Parent retrieval rate at cutoff $k$ | $[0, 1]$ |
| $\mathrm{CR}@k$ | Child recall at cutoff $k$ | $[0, 1]$ |
| $\xi$ | Sibling-confusion rate | $[0, 1]$ |

#### Problem formulation and data-generating story

**Observed.** A finite set of points and the finite target $D$.

- The statistics are computed exactly over the chosen code population. There is no sampling model,
  apart from the text stage's per-epoch 300-code draw.
- The implicit estimand is how well the embedding's metric reconstructs the taxonomy's metric, in
  four senses: rank (Spearman), linear (Pearson), top of list (NDCG), and local (parent, child,
  and sibling retrieval).
- No estimand is defined for any intended use, such as matching descriptions to codes or economic
  prediction. Two downstream benchmarks are defined but have never been run.

#### Estimation / inference procedure

All distances are $d$ at $c = 1$.

- **$\rho_{\mathrm{S}}$.**
  - Each unordered pair counts once: mirrored entries are averaged and the diagonal is excluded.
  - Ties get average ranks. The filter $D \ge 0.1$ drops no distinct pair.
  - The statistic is reported as undefined, with a reason, when there are fewer than two pairs or
    either vector is constant.
  - The definition is versioned. Earlier values used ordinal ranks, depend on code order, and are
    not comparable.
- **$\rho_{\mathrm{P}}$.** Pearson correlation over the same pairs. The protocol calls it
  "cophenetic correlation", but no dendrogram is involved.
- **$\mathrm{NDCG}@k$**, for $k \in \{5, 10, 20\}$ (Järvelin & Kekäläinen 2002).
  - Queries: codes $i$ with at least 20 candidates.
  - Candidates: codes $j \ne i$ with $D_{ij} \ge 0.1$, ranked by $d(z_{i}, z_{j})$.
  - Graded relevance: $g_{ij} = (D^{\max}_{i} - D_{ij}) / D^{\max}_{i}$, where $D^{\max}_{i} = 99$
    for every query. Within-sector relevance therefore lies in $[0.919, 0.995]$, and cross-sector
    relevance is about 0.
  - Discount $1/\log_{2}(1 + \text{rank})$; the ideal ordering is by relevance; the result is the
    mean over queries.
- **Distortion.** $d(z_{i}, z_{j}) / D_{ij}$ over all pairs, summarized by its mean, standard
  deviation, median, and extremes.
- **$\mathrm{PR}@k$.** The fraction of non-sector codes whose parent is among their $k$ nearest
  codes, with every code under evaluation a candidate.
- **$\mathrm{CR}@5$.** The mean, over parents, of the share of their children among their 5
  nearest codes.
- **$\xi$.** The fraction of codes whose nearest neighbor is a sibling.
- **Radius by level.** The mean $x_{0}$ per level, the separation between adjacent levels, and the
  share of parent–child pairs in which the child has the larger $x_{0}$.
- **Populations.** The text stage computes these on 300 codes drawn anew each epoch from the
  validation anchors. The graph stage and the acceptance gate use all 2,125 codes.

**Acceptance gate.** $A$ is as defined in Composition, computed on all codes, with thresholds
fixed in advance:

- the cophenetic correlation may drop by at most 0.02;
- NDCG@10 may drop by at most 0.01;
- parent retrieval must rise by at least 0.05.

$\rho_{\mathrm{S}}$ and its change are reported but do not gate.

**Defined but never executed.** No results exist for either benchmark.

- *Taxonomy tasks:*
  - parent identification among candidates at the parent's level (top 1, 3, and 5);
  - sibling precision at 5 and at 10;
  - k-means clusters compared with the two- and three-digit groupings, by adjusted Rand index and
    normalized mutual information;
  - logistic regression predicting the sector from tangent coordinates, with a stratified 80/20
    split.
- *Economic benchmark:*
  - One row per six-digit code; the target is log 2022 annual average private-sector employment.
  - Ridge regression (penalty 1, unscaled features) on the tangent-space coordinates of the
    embedding.
  - A single 80/20 split that holds out codes; reported as $R^{2}$ and RMSE.
  - Comparators: code one-hot indicators, which can only predict the intercept for held-out codes;
    and the embedding plus log establishment counts and log wages.
  - A multi-level variant extends it to levels 2–6.

#### Assumptions and limitations

1. **Agreement with $D$ is the quality criterion.** *Breaks:* this is circular with training,
   since the same target is optimized. No statistic uses information outside the taxonomy, and
   none evaluates held-out codes, new text, or economic outcomes.
2. **The statistics resolve the hierarchy.** *Breaks:* 87.9% of pairs share $D = 99$.
   - $\rho_{\mathrm{P}}$ then comes close to a point-biserial correlation with same-sector
     membership.
   - $\rho_{\mathrm{S}}$ is dominated by one large tied block.
   - NDCG relevance is nearly binary: same sector or not.
   - So within-sector ordering barely moves any global statistic.
3. **The gate tests refinement meaningfully.** *Breaks:* it requires a gain in parent retrieval,
   which the graph stage trains on directly (parent–child pairs are both edges and positives),
   while tolerating global losses. The thresholds are not derived from any noise estimate.
4. **Populations match across uses.** *Breaks:* text-stage statistics during training use a random
   300-code sample that can include only 160 of the 1,012 six-digit codes, while the gate uses
   all codes. Samples also differ from epoch to epoch, which adds noise.
5. **Uncertainty is negligible.** *Breaks:* there is a single run with a single seed, and no
   confidence intervals or paired resampling for the deltas.
6. **Downstream relevance has been established.** *Breaks:* neither downstream benchmark has run,
   and the economic benchmark's comparators cannot isolate the value of the embedding:
   - one-hot indicators carry no information for held-out codes;
   - the model with covariates has no covariates-only comparator;
   - there is no paired text-stage versus graph-stage comparison;
   - it is a cross-sectional level regression, not a forecast.
7. **Distances are correct.** They are correct only at $c = 1$.

#### Evaluation criteria

The protocol's own required properties:

- the rank statistic is invariant to simultaneous permutation of codes and is correct under ties;
- malformed inputs raise an error instead of returning a value;
- undefined statistics are reported explicitly;
- definitions are versioned, so historical values are never silently compared.

There is no criterion for the protocol's external validity.

#### Open questions for the reviewer

1. What evaluation would establish usefulness beyond reproducing the taxonomy? Candidates: held-out
   codes or subtrees; matching independently labeled establishment descriptions to codes; economic
   outcomes with time-respecting splits. Which baselines are essential: frozen pretrained text,
   taxonomy-only representations such as ancestor indicators, exact tree operations?
2. For hierarchy reconstruction, which metrics are standard (mean average precision of ancestors,
   distortion, rank of the true ancestor), and how should the cross-sector tie block be handled?
3. How should the refinement gate be designed so that it does not reward what the graph stage
   trains on directly?
4. What uncertainty quantification is appropriate when comparing two embeddings of the same finite
   code set?

## Open questions for the reviewer

1. Where does the overall design sit relative to the state of the art in taxonomy embedding,
   hierarchical text classification, and taxonomy expansion? The design is a text-conditioned
   hyperbolic contrastive encoder, followed by transductive hyperbolic graph refinement over the
   same taxonomy. Is there a simpler design that dominates it?
2. Is hyperbolic geometry justified for a 2,125-node, five-level taxonomy embedded in 384
   dimensions, given that the realized representation is confined to a sphere of fixed radius?
   Would a Euclidean or spherical model with a hierarchy-aware objective be equivalent in
   practice?
3. Is it defensible to supervise with, and evaluate against, the same structural target? If not,
   what independent signal should drive each?
4. How should explicit exclusions be treated consistently across the two stages?
5. Which inert or nominal mechanisms should be repaired, and which removed, judged by the
   literature's evidence for each? The candidates are the radius penalty, curriculum mining,
   false-negative clustering, adaptive margins, learnable curvature, and the graph-curriculum
   filters.
6. What is the minimal experiment that would decide whether the graph stage should exist at all?
7. For the stated uses (search over codes by description, clustering, economic features), what
   estimand should replace structural reconstruction, and what does that imply for the objective?
