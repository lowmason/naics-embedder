# Methodological review of the NAICS hyperbolic embedding system

## Overall assessment

My first-pass conclusion is that this is an **interesting but methodologically over-engineered hierarchy-reconstruction system whose strongest ideas are not yet connected to a valid evaluation of its stated purposes**. The text-conditioned use of official NAICS definitions is promising; using hierarchy as auxiliary supervision is well motivated; and hyperbolic geometry is a reasonable hypothesis for a tree-like label space. But the reference methodology has several correctness problems before questions of state of the art even arise: the S2→S3 composition is undefined at the configured dimensions, the proposed “structural distance” is not actually a metric, some objectives are mutually contradictory, validation does not constitute a holdout, and the only acceptance criteria are derived from the same taxonomy used as training supervision. fileciteturn0file0

More importantly, the research literature has moved away from the premise that a good hierarchical representation should be judged primarily by whether pairwise distances reconstruct an editorial tree. Classic hyperbolic work focused on parsimoniously representing hierarchy, often in very low dimensions; directional methods such as entailment cones explicitly model the partial order; taxonomy-expansion work evaluates whether unseen concepts can be attached to the right place; hierarchical text-classification work increasingly injects hierarchical supervision directly into text encoders; and recent results show that simple hierarchy-aware encoder architectures can match considerably more elaborate graph-based systems. citeturn11search13turn11search9turn13search1turn16search0turn12search3turn12search0

So I would locate the design as follows:

> **Conceptually:** a thoughtful combination of approximately 2017–2022 ideas—hyperbolic embedding, hierarchy-aware contrastive learning, sparse experts, hyperbolic GNNs—with some original NAICS-specific supervision engineering.
>
> **Relative to the 2026 methodological frontier:** behind the state of the art in experimental design, inductive evaluation, directional hierarchy modeling, taxonomy expansion, text-embedding baselines, and complexity control.
>
> **Most likely dominant simpler alternative:** a shared modern text encoder over cleanly marked NAICS fields, trained with a task-aligned retrieval/ranking objective plus at most one hierarchy-aware regularizer, with no graph stage unless a controlled experiment establishes incremental value.

This is a **provisional first-pass review, not an adjudicated critique**. I will retain the C-numbers below through our discussion so that points can be accepted, modified, or rejected explicitly. Nothing below should later be represented as “adjudicated” unless we actually adjudicate it.

## Where the methodology sits relative to published work

The strongest published precedent for the central geometric hypothesis is Nickel and Kiela's Poincaré embedding work. They showed that hyperbolic spaces can represent hierarchical data parsimoniously and outperform Euclidean counterparts on hierarchy reconstruction; subsequent Lorentz-model work improved optimization. Sala et al. showed just how strongly the argument is tied to **low-dimensional efficiency**: their constructive method reached mean average precision 0.989 on WordNet in only two hyperbolic dimensions and explicitly studied the dimension/precision tradeoff. citeturn11search13turn11search9

That matters here because S2 uses **384 spatial dimensions**. The published motivation for hyperbolic embeddings is not “hierarchy exists, therefore hyperbolic”; it is substantially “tree-like growth is expensive to represent in low-dimensional Euclidean space but natural in negatively curved space.” A 384-dimensional representation substantially weakens that argument and makes matched Euclidean and spherical controls indispensable. The literature does not prove that hyperbolic space cannot help in 384 dimensions; it means the methodology cannot simply import the low-dimensional hyperbolic argument without testing whether it survives in this regime. citeturn11search9turn13search0

The system also ignores a major branch of hierarchy-embedding work: **directional order geometry**. Its learned distance is symmetric even though “ancestor of” is directed. Hyperbolic entailment cones were designed specifically to represent DAG partial orders via nested geodesically convex regions and reported better representation capacity and generalization than then-strong baselines. The current methodology instead tries to recover direction indirectly through radius, while its contrastive distance itself cannot distinguish \(a\to b\) from \(b\to a\). citeturn13search1

There is an even closer literature: **hyperbolic taxonomy expansion**. HyperExpan explicitly combined hyperbolic representation learning with the problem of placing new concepts in an existing taxonomy and reported improvements over Euclidean representation-learning baselines. TEMP reframed expansion as ranking candidate taxonomy paths for a new concept. More recent LORex combines candidate ranking with lineage-oriented reasoning. These approaches are methodologically closer to the project's inductive aspiration than reconstructing all pairwise NAICS distances: they ask whether a representation can correctly place something not already embedded as a free node. citeturn16search7turn16search0turn12search5

The hierarchical text-classification literature likewise undermines the assumption that a separate graph-refinement stage is naturally necessary. HGCLR showed that hierarchical information can be injected into the text encoder through contrastive training so that the hierarchy need not remain present at inference. HILL likewise combines text representations and structural information through a hierarchy-aware contrastive scheme. Most strikingly, HYDRA's 2025 experiments found a comparatively simple multi-head encoder-only approach matched or exceeded more complex hierarchical text-classification systems using components such as graph encoders and label semantics on four benchmarks. These are not NAICS experiments, so they do not establish what will win here; they establish that **architectural complexity has to earn its keep empirically**. citeturn12search3turn12search4turn12search0

S3 is also not a particularly current realization of hyperbolic graph learning. HGCN was already an **inductive** hyperbolic GCN in 2019: it maps node features into hyperbolic representations and supports different trainable curvature by layer. Later work has explicitly criticized repeated tangent-space processing as a limitation and developed Lorentz-native operations; more recent graph work likewise uses manifold-valued graph convolution. That does not make tangent-space GNNs invalid, but it makes the particular combination here—free per-node states, origin tangent maps, Euclidean LayerNorm, radius destruction, and nominal curvature—hard to characterize as state of the art. citeturn13search0turn17search0turn17search6

Finally, the text backbone is not contemporary enough to serve as the sole embedding baseline in 2026. Current embedding research includes much larger and substantially newer models and much broader benchmarks; Qwen3-Embedding, for example, introduced 0.6B/4B/8B variants and a multi-stage embedding/reranking pipeline, while MMTEB evaluates hundreds of tasks and also illustrates that bigger is not automatically better—the best public model in that study was substantially smaller than several billion-parameter alternatives. The lesson is not “replace MiniLM with Qwen3-8B”; it is **benchmark a credible set of current text encoders at matched costs rather than letting all-MiniLM-L6-v2 define the semantic ceiling**. citeturn14academia49turn14academia50

The project's use of official label descriptions, however, is well supported. Work on label-description training has shown sizable gains in zero-shot text classification, and other work directly formulates classification with complex label definitions as semantic matching. Thus I would preserve the basic idea “NAICS definitions should participate in representation learning”; I would question almost everything layered around that idea before questioning the idea itself. citeturn16search5turn16search10

## Provisional critique points

The following are the points I think are substantive enough to carry forward to adjudication.

| ID | Type | Severity | Provisional finding |
|---|---|---:|---|
| **C1** | **Fix an error** | Blocker | The S2→S3 composition is undefined in the reference configuration: S2 exports ambient dimension 385 and S3 expects 31, with no specified projection. If curvature is ever changed from 1, the documented distance formulas also become wrong. |
| **C2** | **Fix an error** | Critical | \(D\) is called a structural distance/metric but the half-step lineal rule violates the triangle inequality. It therefore cannot be exactly realized by *any* metric embedding, hyperbolic or otherwise. |
| **C3** | **Fix an error** | Critical | The level-radius objective says every sector root belongs at the unique hyperbolic origin, while the structural objective says every pair of distinct sector roots has distance 99. The objectives are mutually impossible. |
| **C4** | **Fix an error / better method** | Critical | The radius cap can eliminate the very radial degree of freedom that is supposed to encode hierarchy. The retained historical run saturated it; when saturation occurs, ranking becomes purely angular. The method therefore has not established that it is meaningfully hyperbolic. |
| **C5** | **Adopt a better method** | Critical | The arbitrary cross-sector sentinel \(99\) turns several losses and evaluation statistics primarily into same-sector versus different-sector objectives. It should not be used as a quantitative semantic distance. |
| **C6** | **Fix an error** | Critical | Validation is not a generalization test: training and validation use the same codes and positive pairs, with 89.1% triple overlap. S3 validation is weaker still because the “held-out” examples use the full graph and trainable node states. |
| **C7** | **Fix an error** | High | S1 sampling depends on arbitrary storage/code orientation and sometimes labels a structurally *closer* code as the negative. No ancestors survive; most leaves are never anchors. |
| **C8** | **Adopt a better method** | High | Exclusions are semantically misrepresented as symmetric “unrelated” code-code relations. They are more naturally directed boundary/redirection evidence, and S2/S3 currently give them contradictory treatment. |
| **C9** | **Fix an error** | High | Several text preprocessing choices manufacture nuisance signal: repeated exclusion prose causes truncation, fixed placeholders reveal channel absence/level, inherited descriptions create near-duplicate parent-child text, and 14 inheritance choices are arbitrary. |
| **C10** | **Adopt a better method** | High | S2 has too many simultaneously weakly justified objectives and curricula. Two losses are inert, mining is inert, clustering never executes, and the remaining objectives are partly incompatible. |
| **C11** | **Adopt a better method** | Medium–high | Four separate encoder adaptations plus sparse MoE fusion are not justified against simple shared-encoder fusion; MiniLM is no longer a sufficient contemporary semantic baseline. |
| **C12** | **Adopt a better method** | Critical decision | S3 brings no independent information source: it trains on the same taxonomy as S2, has free node states, and has no text-retention term. It is capable of becoming a new transductive taxonomy embedding rather than refining the text geometry. Its existence should be experimentally earned. |
| **C13** | **Adopt a better method** | High if S3 survives | The graph layer's origin-tangent processing plus LayerNorm has no demonstrated geometric rationale here and actively resets radius. Parent/child-only and simple smoothing controls are missing. |
| **C14** | **Fix an error** | High | The S4 gate rewards a property—parent retrieval—that the graph stage directly trains on, uses arbitrary fixed tolerances, and provides no stochastic uncertainty from multiple runs. |
| **C15** | **Extend/realign scope** | Highest scientific value | The stated purposes—description search, clustering, economic features—have no corresponding training or executed evaluation estimand. Structural reconstruction is at best an intrinsic diagnostic, not evidence of usefulness for those purposes. |
| **C16** | **Extend scope** | Medium | Single-vintage evaluation is too narrow for an economic representation intended for continued use; NAICS is revised periodically, and a 2027 revision is already in process. Cross-vintage concordance is a valuable natural generalization test. citeturn15search0turn15search2 |

Several deserve fuller treatment.

**C2 is stronger than the description currently recognizes.** Let \(A\to B\to C\) be a two-edge ancestor chain. Under S1,

\[
D(A,B)=0.5,\qquad D(B,C)=0.5,\qquad D(A,C)=1.5.
\]

Therefore

\[
D(A,C)=1.5>D(A,B)+D(B,C)=1.
\]

That violates the triangle inequality. fileciteturn0file0

So the half-step construction is not merely an unconventional weighting. It is mathematically incompatible with any metric space. A hyperbolic model cannot solve this by being expressive enough; **no metric geometry can simultaneously realize those target distances**. If those numbers are intended only as ordinal relevance labels, call them scores and use ranking supervision. If actual distance matching is intended, replace them with a metric.

This also puts the taxonomy-similarity literature in the right perspective. Simple edge counting has long been criticized because taxonomy edges need not represent uniform semantic increments. Resnik's information-content measure outperformed edge-count similarity on its classic human-similarity benchmark, and Jiang–Conrath combined taxonomy structure with corpus statistics. Those results are from lexical taxonomies rather than industrial classifications, so they do not directly validate an information-content metric for NAICS; they establish that **unweighted editorial path length is not a uniquely privileged notion of semantic similarity**. citeturn18academia24turn14search7

For NAICS, this point is particularly important because NAICS itself is **production-oriented**. Census describes it as grouping economic activities based on production processes. NAPCS, by contrast, is market/demand based and permits the same product to span NAICS industries or sectors. Thus “NAICS tree distance” and “economic relatedness” are different constructs by design, and NAPCS offers one plausible independent signal for testing that distinction. citeturn15search0turn15search5turn15search7

**C3 creates a direct objective contradiction.** S2's level target at level 2 is

\[
\lVert z_s\rVert = 0,
\]

which in the hyperboloid means \(r=0\), hence the single point \(o\). Thus the level term wants all 20 sector roots to coincide. But \(D\) assigns every different-sector pair—including sector roots—99, and the hierarchy loss wants them much farther apart than any within-sector pair. fileciteturn0file0

No weight tuning resolves the underlying incompatibility; it only decides which requirement loses. A more defensible radial model would encode **specificity/order without demanding one unique point per level**. Hyperbolic entailment cones are directly relevant because they represent hierarchy as a partial-order relation rather than forcing depth into a hand-set scalar radius. citeturn13search1

**C4 makes the question “is hyperbolic geometry doing anything?” answerable mathematically.** If every point is on \(r=2\), then

\[
\cosh d(i,j)
=\cosh^2 2-\sinh^2 2\cos\vartheta_{ij}.
\]

Because \(\operatorname{arcosh}\) is monotone, ranking neighbors by hyperbolic distance is exactly the same as ranking them by angle/cosine on that fixed-radius sphere. The model still applies a hyperbolic nonlinear transformation to angular separation, but the characteristic radial representation of depth has disappeared. fileciteturn0file0

I would therefore not say “hyperbolic geometry definitely fails,” because the saturation evidence comes from the retained historical run rather than a completed current-supervision run. I would say: **the current parameterization contains a demonstrated failure mode under which its hyperbolic motivation disappears, and it lacks the Euclidean/spherical controls needed to establish that the failure is absent or harmless.**

There is also a smaller factual error in the methodology text: it calls the decoupled contrastive term “unbounded below.” With \(r\le2\), all pairwise distances satisfy \(0\le d\le4\), so that loss is bounded below. With \(K=24,\tau=.07\), its theoretical lower bound under the cap is

\[
\log 24-\frac4{0.07}\approx -53.96.
\]

The substantive issue remains: the objective has a strong incentive to put negatives at the maximum available distance. But “unbounded below” is not true under the stated parameterization. fileciteturn0file0

**C5 is not just a scale-choice concern.** Because 87.9% of pairs have \(D=99\), the hierarchy loss's batch normalizer, Pearson correlation, Spearman correlation, distortion summaries and NDCG relevance are all heavily shaped by one enormous cross-sector tie block. fileciteturn0file0 The consequence is that a representation can improve “taxonomy reconstruction” largely by learning sector boundaries without becoming materially better at the difficult within-sector distinctions relevant to code search.

The most defensible structural alternative is a virtual super-root if one truly wants a tree metric. But even that should not be confused with semantic industry relatedness: every different-sector path would still meet at the same artificial root. For semantic relatedness I would instead separate within-taxonomy structure from independent economic signals such as product overlap; Census explicitly notes that the same NAPCS product may span multiple NAICS industries and sectors. citeturn15search7

**C6 and C15 are, in my view, the project's central scientific problem.** Optimizing NAICS hierarchy and then measuring agreement with that same hierarchy tells you whether the optimization worked. It does not establish that the learned representation is useful for retrieving a code from a business description, discovering useful clusters, or improving an economic model. Modern taxonomy-expansion work instead constructs unseen-concept placement problems; label-description research directly tests matching/classification; hierarchical classification papers evaluate actual classification tasks. citeturn16search0turn12search5turn16search5turn12search3

That distinction should govern the whole redesign: **taxonomy reconstruction should become an intrinsic diagnostic, not the primary estimand.**

**C8 changes how I would use the exclusions.** An official exclusion of the form “activities X are classified in code \(j\), not code \(i\)” is highly valuable supervisory information, but its natural interpretation is not “the embeddings for \(i\) and \(j\) must be far apart.” It is closer to:

\[
\operatorname{score}(X,j)>\operatorname{score}(X,i).
\]

That is a directional, activity-conditioned boundary constraint. Symmetrizing it to \(X_{ij}\) destroys that semantics. The fact that hundreds of exclusions are siblings and several are lineal is evidence against absolute geometric repulsion, not evidence that those official links are bad data. fileciteturn0file0

**C12 is the point on which I would be most willing to delete a component rather than repair it.** HGCN's influential formulation is inductive and transforms input features rather than learning unconstrained free embeddings for each node. Your S3 initializes free states from text, then has no term requiring the final states to retain the text geometry. Because its graph is itself derived from the taxonomy already supervising S2, it adds inductive bias and optimization capacity but essentially no new observational information. citeturn13search0

That does not mean graph refinement cannot help. It means “graph stage improves PR@1” is not enough to justify it: parent-child links are both graph edges and training positives. A much simpler neighbor-smoothing operation could produce the same gain.

## Improvements ranked by expected value

I would separate **decision value** from the order in which engineering must physically occur. For example, C1 is a hard blocker if S3 is executed, but deciding whether S3 should exist at all has higher scientific value than carefully repairing it.

| Rank | Action | Classification | Why I expect high value |
|---:|---|---|---|
| **1** | Define and run a genuinely independent benchmark for **description → NAICS retrieval** before more architecture work. | **Fix evaluation / realign estimand** | It answers whether any learned representation serves the project's clearest stated use and tells you whether hierarchy reconstruction correlates with what matters. |
| **2** | Replace the current \(D\) supervision with either a valid metric or explicitly ordinal/directional supervision; remove the half-step construction. | **Fix an error** | Current targets are geometrically impossible to satisfy exactly and internally disagree with negative generation. |
| **3** | Run matched **Euclidean, unit-sphere/cosine, and hyperbolic** versions of the same encoder/objective across dimensions such as 16/32/64/384. | **Adopt a better experimental method** | This determines whether hyperbolicity itself contributes anything rather than assuming it. Classic work's strongest advantage is low-dimensional hierarchy representation. citeturn11search9turn11search13 |
| **4** | Decide whether S3 exists with a minimal controlled experiment, rather than repairing it first. | **Adopt a better method** | It may be redundant because S2 already consumes the same hierarchy. Recent hierarchical text work provides strong precedent for hierarchy-aware encoders without graph inference. citeturn12search3turn12search0 |
| **5** | Remove \(D^\times=99\) from magnitude-sensitive objectives; evaluate within-sector structure separately from cross-sector discrimination. | **Fix an error / better target** | It currently dominates the numerical training and evaluation problem. |
| **6** | Redesign positive/negative sampling symmetrically around the actual task: all nodes can be anchors; eliminate code-number orientation; use hierarchy-aware hard negatives. | **Fix an error** | It removes arbitrary supervision and focuses learning on the distinctions the model must actually make. |
| **7** | Represent exclusions as directed typed boundary/redirection supervision rather than mandatory symmetric repulsion. | **Adopt a better method** | It aligns the supervision with what the text actually says and eliminates the S2/S3 contradiction. |
| **8** | Introduce real code/subtree holdouts for model selection and a sealed independent test. | **Fix an error** | Current validation estimates another negative draw over training relationships, not generalization. Taxonomy-expansion work provides natural held-out-placement formulations. citeturn16search0turn12search5 |
| **9** | Simplify S2 to one shared field-aware encoder, simple fusion, one principal task loss, and at most one hierarchy regularizer. | **Adopt a better method** | The current MoE, six losses and inert curricula make causal attribution almost impossible; recent HTC results show simpler models can be highly competitive. citeturn12search0 |
| **10** | Clean the text construction: stop exclusion repetition, mask rather than text-encode missing fields, preserve inheritance provenance, eliminate arbitrary inheritance. | **Fix an error** | These are avoidable nuisance signals and truncation mechanisms before representation learning even begins. |
| **11** | Benchmark contemporary text embedding backbones against MiniLM under a common head/objective. | **Adopt a better method** | Modern embedding models and benchmarks have advanced substantially, while MMTEB also cautions that size alone does not determine quality. citeturn14academia49turn14academia50 |
| **12** | Replace the refinement gate with independent-task selection and repeated-seed paired comparison. | **Fix an error** | The current gate can be optimized almost directly by S3 and has no empirically calibrated uncertainty. Statistical testing needs to match the experimental unit and evaluation setup. citeturn11search0 |
| **13** | Only if hyperbolic models win, repair/modernize the geometry: remove the hard cap, use a stable radial/order formulation, and consider entailment/order-aware or Lorentz-native layers. | **Adopt a better method** | Otherwise this is complexity spent optimizing a geometry that has not earned inclusion. citeturn13search1turn17search0 |
| **14** | Add cross-vintage NAICS evaluation and economic time-respecting benchmarks. | **Extend scope** | NAICS is revised every five years, and 2027 updates are already under review, making vintage shift an unusually natural external test. citeturn15search0turn15search2 |

I would **not** prioritize repairing all nominal/inert machinery. In particular, I would not spend methodological budget making the false-negative clustering, adaptive margins, graph curriculum filters, or learnable curvature “work as designed” until much simpler baselines establish that there is a residual problem for those mechanisms to solve.

## Direct answers to the component open questions

**S1 — Taxonomy supervision construction**

| Question | My answer |
|---|---|
| **Is tree path length with the half-step defensible?** | **Plain tree path length is defensible as a topology-reconstruction target, but not as a general measure of industry relatedness. The half-step modification is not defensible as a distance because it breaks the triangle inequality.** If the purpose is similarity rather than topology, investigate depth/information-content methods or learn relevance externally. Resnik and Jiang–Conrath are classic evidence that flat edge counting need not track semantic similarity well. citeturn18academia24turn14search7 |
| **What should replace cross-sector 99?** | For strict structural reconstruction, a virtual super-root produces a legitimate finite tree path. For semantic relatedness, I would **not assign one quantitative cross-sector distance at all**: use a separate cross-sector discrimination term and evaluate within-sector ordering separately, or derive relatedness from independent economic/product evidence. NAPCS is particularly attractive because it is explicitly demand/product based and spans NAICS boundaries. citeturn15search5turn15search7 |
| **Role of exclusions?** | Treat them as **directed typed boundary/redirection evidence**. They are hard negatives for an *activity/query → code* decision when the excluded activity should map to the referenced destination; they are not automatically repulsive code-code pairs. Lineal exclusions should not override the ancestor relation with a generic “unrelated” constraint. |
| **Is the positive set sound?** | **No.** Code-number-dependent sibling orientation has no semantic justification; eliminating ancestors and leaving most leaves out as anchors creates avoidable asymmetry. Sample query roles symmetrically, include all levels, and use graded hierarchy relations or directional ancestor supervision. Taxonomy-expansion methods instead train on attachment/path decisions that directly imitate unseen-concept placement. citeturn16search0turn12search5 |
| **Empty channels and inherited descriptions?** | Mask missing fields rather than representing “missing” as semantic text. Use explicit field markers with a shared encoder. Do not repeat exclusion prose. Preserve an inheritance indicator if inherited text must be used; ideally separate “own description” from inherited context. The 14 arbitrary candidate selections should be replaced by a deterministic semantically justified rule or all candidates. |

A subtle point on information content: I would **not** simply use employment frequency to instantiate Resnik's \(IC=-\log p\) and call the result semantic specificity. Large industries are economically frequent, which is not necessarily the inverse of conceptual specificity. Intrinsic descendant counts or an independently justified establishment distribution could be tested, but this is an empirical alternative, not an automatic correction. Resnik's result establishes that nonuniform edge information can matter, not that his WordNet estimator transfers unchanged to NAICS. citeturn18academia24

**S2 — Text-conditioned hyperbolic encoder**

| Question | My answer |
|---|---|
| **Given cap saturation, is hyperbolic geometry doing work?** | **Not in the intended hierarchical sense if saturation occurs.** At constant radius, nearest-neighbor order is an angular/spherical order. Run matched spherical and Euclidean controls first. Hyperbolic geometry historically has its clearest representation-capacity case at low dimension. citeturn11search9turn11search13 |
| **Which parameterization preserves radial depth?** | First remove the hard clipping mechanism that creates a zero radial gradient outside the ball. If hierarchy direction is genuinely wanted, entailment/order objectives are more principled than manually assigning one radius per digit level. An uncapped but numerically stable Lorentz representation with learned scale is also reasonable. citeturn13search1 |
| **Is 384-D hyperbolic space justified?** | **Not yet.** It may work, but 384 dimensions make the classic capacity argument much less compelling. Dimension must be an experimental factor, not inherited from MiniLM. Sala et al.'s results illustrate the low-dimensional premise particularly clearly. citeturn11search9 |
| **Six-term objective?** | **No, not in its present form.** Choose one primary estimand-driven loss and perhaps one hierarchy regularizer. For search, use retrieval/classification contrastive supervision with hierarchy-aware negatives; for pure hierarchical representation, use directional or graded hierarchy supervision. Remove inert terms rather than counting them as model features. |
| **Negative choice?** | With only 2,125 codes, make every code available in a cached code bank and mine globally, or even compute all code scores where practical. Concentrate learning on plausible confusions—siblings, cousins, semantically similar cross-sector codes—while explicitly handling false negatives. Fixed pre-drawn negatives sacrifice most of the point of representation learning. Contemporary embedding systems routinely emphasize high-quality/hard negatives. citeturn14academia49turn14academia50 |
| **Sparse MoE justified?** | **No evidence yet.** Four experts receiving the same concatenated four-field vector are substantially more complicated than a shared encoder with field markers plus masked mean/attention fusion. Require an ablation showing the MoE wins. HYDRA is relevant evidence that hierarchical NLP complexity often fails to justify itself. citeturn12search0 |
| **Is MiniLM appropriate?** | It is a reasonable efficiency baseline, not an adequate 2026 state-of-the-art baseline. Compare a small contemporary embedding model, a stronger current embedding model, and MiniLM under the same downstream objective. Qwen3-Embedding and MMTEB illustrate how much the embedding landscape has changed and why task-specific comparison matters. citeturn14academia49turn14academia50 |
| **What validation design?** | Use a fixed **grouped holdout**. For inductive taxonomy behavior, remove complete leaf nodes or subtrees from supervision and test attachment/retrieval. For the actual search purpose, hold out independently labeled business descriptions. Neither checkpoint selection nor LR scheduling should inspect relationships appearing in training. |

I would also change the hierarchy loss independently of all that. Normalizing \(d\) and \(D\) by their **micro-batch means** makes the target for one pair depend on which unrelated pairs happen to share its batch, while gradients through the learned \(\bar d\) couple all pairs. With the cross-sector mass, this effectively normalizes against a batch-specific same-sector/cross-sector mixture. fileciteturn0file0 If quantitative distance matching survives, use a globally defined scale or fit a global monotonic/calibration transform. More likely, I would drop quantitative distance matching for an ordinal relevance/order objective.

**S3 — Graph refinement**

| Question | My answer |
|---|---|
| **What can the graph stage add after hierarchy-supervised S2?** | Potentially useful local smoothing and a topology-specific inductive bias—but **no new factual source**. It therefore needs a strong incremental-value test. HGCN's original contribution was inductive feature-based graph representation, unlike unconstrained per-node re-embedding. citeturn13search0 |
| **Minimal controls?** | Compare S2 alone; S2 plus simple one-step neighbor averaging with a tuned residual coefficient; S2 plus the proposed S3; and S2 plus a matched amount of extra S2 optimization. If S3 wins, add a **text-shuffle control**: permute S2 embeddings across graph nodes before S3. If performance barely changes, S3 is using taxonomy topology rather than refining text. |
| **Should it retain text geometry?** | **Yes, if “refinement” is the intended claim.** Either make \(Z^{(3)}\) the actual node features rather than trainable free node identities, use a residual correction around \(Z^{(3)}\), or add point/pairwise distillation. Otherwise call S3 what it is: a transductive taxonomy embedding initialized by S2. |
| **Tangent origin + LayerNorm appropriate?** | Tangent-space processing itself has substantial precedent, including early hyperbolic GNNs, so I would not reject it categorically. But **this LayerNorm construction is poorly aligned with the goal of preserving radius**. Later fully hyperbolic work explicitly argues that repeated tangent-space operations limit hyperbolic modeling, giving a credible alternative if hyperbolicity ultimately matters. citeturn13search0turn17search0 |
| **Which graph edges?** | Start with **parent–child only**. Siblings, grandparents and great-grandparents are deterministic consequences of the tree, so they should enter only if ablation demonstrates benefit. Exclusions should be omitted from positive propagation or modeled as a separate signed/typed boundary relation. |
| **Triplet hinge/adaptive margin/uncertainty weighting?** | A triplet objective is not inherently wrong, but the current adaptive margin, hinge “temperature,” and learned uncertainty weight constitute needless coupled machinery. Ranking/softmax attachment objectives are closer to taxonomy-expansion practice; if metric reconstruction is truly the objective, directly optimize a principled distortion criterion. TEMP is one example of taxonomy-path ranking, though its particular dynamic margin is not automatically the right answer here. citeturn16search0 |

The most decisive S3 experiment does **not** require perfecting S3 first. My minimal decision matrix would be:

\[
\text{S2}
\quad\text{vs}\quad
\text{S2 + simple smoothing}
\quad\text{vs}\quad
\text{S2 + S3}
\quad\text{vs}\quad
\text{S2 + matched extra training}.
\]

Run all four on the *same genuinely held-out retrieval/attachment task*, with multiple seeds. Only if S3 materially and reproducibly wins should the project spend effort on hyperbolic graph-layer sophistication.

**S4 — Evaluation**

| Question | My answer |
|---|---|
| **What establishes utility beyond reproducing NAICS?** | For search: independently labeled **business/establishment description → NAICS code** retrieval. For inductive taxonomy use: held-out node/subtree attachment and, eventually, cross-vintage transfer. For economic features: out-of-time prediction or explanatory incremental value over strong economic and taxonomy baselines. |
| **Essential baselines?** | Frozen modern text embeddings; fine-tuned text encoder without hierarchy; exact tree operations/tree distance; ancestor-indicator or one-hot hierarchical structural features; Euclidean/spherical/hyperbolic versions under a matched loss and dimension; simple neighbor smoothing; for economic tasks, covariates-only and taxonomy-only baselines. |
| **Hierarchy reconstruction metrics?** | Prefer per-query parent/ancestor rank, MRR, mean average precision/recall over ancestors, held-out edge attachment accuracy, and within-sector distortion/order measures. Hyperbolic hierarchy work commonly emphasizes reconstruction/ranking quality such as MAP rather than a global correlation dominated by one tie block. Sala et al., for example, report MAP for WordNet embedding quality. citeturn11search9 |
| **How handle cross-sector ties?** | Never let the 87.9% tie block dominate a single headline statistic. Report **within-sector macro averages**, cross-sector discrimination separately, and a global metric only as secondary context. If a virtual root is adopted, report its reconstruction separately from fine-grained within-sector structure. |
| **How redesign the gate?** | Gate primarily on an **independent held-out task** that S3 does not directly optimize. Then impose noninferiority bounds on secondary metrics. If only taxonomy reconstruction is available, hide edges/subtrees from the graph and gate on them—not on parent pairs used as messages and positives. |
| **Uncertainty?** | Separate two sources. On all 2,125 fixed codes, the exact metric itself has no finite-population sampling error. But training is stochastic, so report across seeds. For query-level external evaluation, use paired resampling or randomization appropriate to the evaluation unit; where subtree/sector dependencies matter, resample at that grouped level rather than pretending codes are independent. Statistical tests should be chosen around the actual dependence structure and metric. citeturn11search0 |

The current economic benchmark needs substantial redesign before it can establish usefulness. A random 80/20 regression of contemporaneous 2022 employment on embeddings tests whether embedding coordinates are associated with cross-sectional industrial scale. It is not a forecast. Its one-hot comparator cannot generalize information about a held-out code, and “embedding + wages + establishments” lacks the necessary “wages + establishments only” baseline. fileciteturn0file0

A credible economic experiment would look more like

\[
Y_{i,t+1}
=
f(\text{economic covariates}_{i,\le t},
  \text{representation}_{i,t})
\]

with a chronological evaluation period and incremental comparison among: economic covariates only; taxonomy features only; text representation only; text + hierarchy representation; and covariates + each representation. If forecasting is not actually the purpose, define a different economic estimand explicitly rather than calling a cross-sectional fit predictive evidence.

## Direct answers to the overall open questions

**Overall Q1: Where does the design sit relative to SOTA, and is there a simpler design that likely dominates it?**

It is **not state of the art as an integrated methodology**, although several individual ingredients were state of the art in their respective eras. The canonical hyperbolic hierarchy work is from 2017–2018; inductive HGCN appeared in 2019; hyperbolic taxonomy expansion appeared by 2021; hierarchy-aware text contrastive methods were established by 2022; and recent taxonomy-expansion and hierarchical-classification literature places much greater emphasis on unseen-concept placement, direct semantic matching, and controlling unnecessary architecture. citeturn11search13turn13search1turn13search0turn16search7turn12search3turn12search5turn12search0

The simpler candidate I would put first is:

\[
\boxed{
\text{shared modern text encoder}
+
\text{field markers/masks}
+
\text{task-aligned contrastive or ranking loss}
+
\text{one hierarchy-aware regularizer}
}
\]

For search, a business text is the query and the 2,125 code descriptions are the candidate corpus. Because the candidate set is tiny by modern retrieval standards, this is unusually favorable to full-candidate or aggressively hard-negative training. The graph stage enters only after proving incremental value.

**Overall Q2: Is hyperbolic geometry justified for 2,125 nodes, five levels and 384 dimensions under fixed-radius realization?**

**Not on the existing evidence.** Hyperbolic geometry is plausible a priori because the target is hierarchical, and classic work supports the geometry for hierarchy representation. But the strongest published argument is low-dimensional parsimony, while the current model uses 384 dimensions and contains a saturation mechanism that can collapse all radial information. citeturn11search13turn11search9

Moreover, conditional on equal radii, the proposed hyperbolic neighbor ordering is mathematically equivalent to spherical/angular neighbor ordering. So a spherical cosine model is not merely a generic baseline; it is a **diagnostic control for the exact failure mode already observed historically**.

I would test dimensions such as 16, 32, 64, and 384 under matched Euclidean, spherical, and hyperbolic objectives. A finding that hyperbolic-32 matches Euclidean-384 would be a real geometric result. A finding that spherical-384 matches hyperbolic-384 would tell you the current system's gains are angular rather than hyperbolic.

**Overall Q3: Is supervising with and evaluating against the same structural target defensible?**

Yes **only as an intrinsic training diagnostic**. No as evidence of representation quality for the stated uses.

The taxonomy can supervise an inductive bias while an independent signal evaluates usefulness. For search, independently classified establishment/business descriptions are the cleanest signal. For taxonomy generalization, hide concepts/subtrees and ask the model to recover their attachment, as taxonomy-expansion methods do. For economic use, use external economic outcomes and strong non-embedding baselines. citeturn16search0turn12search5turn16search5

**Overall Q4: How should explicit exclusions be treated consistently?**

As a **directed relation with textual scope**.

The cleanest supervision is not

\[
X_{ij}=1\Rightarrow d(i,j)\text{ large},
\]

but something like

\[
(i,\text{excluded activity }q,j)
\Rightarrow
s(q,j)>s(q,i).
\]

If the graph stage survives, either leave these pairs out of positive message passing or assign them a separate boundary-relation channel. Do not repel them in S2 and then smooth them together in S3.

**Overall Q5: Which inert/nominal mechanisms should be repaired versus removed?**

My recommendation is deliberately conservative:

| Mechanism | Recommendation |
|---|---|
| **Radius penalty** | **Remove now.** If the hard cap is redesigned, introduce a new geometrically coherent radial/order regularizer rather than repairing this inert threshold. |
| **Curriculum mining** | **Remove from the reference methodology now.** Reintroduce only after candidate pool \(>\!K\) and show that adaptive mining beats simple global hard-negative mining. |
| **False-negative clustering** | **Remove initially.** It risks using the model's own hierarchy-shaped geometry to certify its own negatives as false; there is no evidence here that this complexity is needed. |
| **Adaptive margins** | **Remove unless ablation earns them.** A fixed ranking/contrastive objective is a stronger baseline. Dynamic-margin taxonomy methods exist, but their success does not validate this particular margin formula. citeturn16search0 |
| **Learnable curvature** | **Do not repair yet. Fix \(c=1\) honestly.** If a hyperbolic baseline first beats matched alternatives, then correct all \(c\)-dependent formulas and test learned curvature. HGCN provides precedent for genuinely trainable per-layer curvature. citeturn13search0 |
| **Graph curriculum filters** | **Remove until there is an empirical need.** Supplying missing metadata just to activate a mechanism is not methodological justification. |

That is a general recommendation for this system: **inert machinery should default to deletion, not rehabilitation**.

**Overall Q6: Minimal experiment deciding whether the graph stage should exist?**

Use one fixed independent holdout and compare:

\[
\begin{array}{ll}
A:&\text{S2 only}\\
B:&\text{S2 + simple neighbor smoothing}\\
C:&\text{S2 + proposed S3}\\
D:&\text{S2 + matched extra S2 training}.
\end{array}
\]

Keep representation dimension, validation data, downstream scoring, and computational budget as matched as practical. Repeat across seeds.

The primary criterion should be the external task—preferably held-out description→code retrieval or held-out-node attachment—not PR@1 over the graph that trained S3.

If \(C\) does not reliably beat both \(B\) and \(D\), **delete S3**.

If \(C\) wins, run the text-shuffle control. If shuffling text embeddings across graph nodes barely hurts \(C\), then the stage is not really refining a text representation; it is learning a transductive taxonomy representation. That might still be useful, but it is a different methodological claim.

**Overall Q7: What estimand should replace structural reconstruction for each stated use?**

For **hierarchical search from descriptions**, the estimand should be expected ranking quality of the correct NAICS code(s) for previously unseen business descriptions:

\[
\mathbb E_{(q,Y)\sim\mathcal D_{\text{business}}}
[
\operatorname{RetrievalMetric}(q,Y)
].
\]

Useful primary metrics are MRR, Recall@\(k\), exact six-digit accuracy where labels are unambiguous, and hierarchical error such as distance between predicted and true code. Label-description and semantic-matching literature directly supports treating rich category definitions as the candidate side of such a matching problem. citeturn16search5turn16search10

For **clustering**, first define what a “good” cluster means independently of NAICS. Reproducing two- or three-digit sectors merely tests taxonomy reproduction again. A more meaningful target could be coherence in NAPCS product overlap, supply-chain/input-output relationships, establishment/customer similarity, or another domain-specific external relation. NAPCS is especially interesting because it deliberately supplies a market/demand perspective different from NAICS's production-oriented hierarchy. citeturn15search7

For **economic features**, the estimand should be the **incremental out-of-sample value of the representation over information already available from conventional taxonomy and economic covariates**. That can be measured through time-respecting prediction, treatment/economic models where representation is a nuisance/control feature, or another explicitly specified econometric task. The essential comparison is

\[
\text{strong conventional baseline}
\quad\text{vs}\quad
\text{strong baseline + learned representation},
\]

not learned representation versus an unusable one-hot code.

Finally, **cross-vintage placement deserves to become a fourth benchmark even if it is not a production purpose**. Census states that NAICS is reviewed every five years, and as of September 2026 proposed 2027 updates are under active review. The 2017→2022 transition included code combinations, separations and conceptual changes. A model that genuinely understands industry descriptions should have some ability to relate changed/new codes across vintages; a pure transductive S3 cannot do that without retraining node identities. citeturn15search0turn15search2turn15search4

## Positions I would treat as most important to adjudicate first

The review is not yet adjudicated. For the interactive phase, these are the propositions on which the rest of the critique depends most strongly:

**C2:** the half-step \(D\) is not a metric and should not remain the quantitative target of a metric embedding.

**C15/C6:** structural reconstruction against the training taxonomy is an intrinsic diagnostic, not the primary quality criterion; at least one independent task must determine model choice.

**C4:** until a matched spherical/Euclidean experiment says otherwise, the fixed-radius failure mode means hyperbolicity is unproven rather than a settled design choice.

**C12:** the graph stage should be presumed unnecessary until it beats both simple smoothing and matched additional text-stage training on independent held-out data.

**C8:** exclusions should encode directional boundary/redirection semantics rather than unconditional symmetric code-code repulsion.

**C5:** \(D^\times=99\) should not participate as an ordinary quantitative distance in the current losses and global evaluation statistics.

Those six positions would substantially simplify the system if upheld. Conversely, a convincing rejection of any of them would materially change my recommendation for the subsequent methodology.