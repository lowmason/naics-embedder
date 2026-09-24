# Methodological Review: Hyperbolic NAICS 2022 Embedding (S1–S4)

**Bottom line:** As configured, this is not a working hyperbolic taxonomy embedding. It is a spherical embedding, and its objective, targets and evaluation cannot tell whether it generalizes. Three things undercut the method: (i) a hard tangent-norm cap combined with a low-temperature contrastive loss that rewards radius without limit, which forces every point onto one sphere; (ii) a level-radius target that puts all 20 sectors at the origin; (iii) supervision and evaluation that both come from the same in-sample hierarchy, with no held-out codes, subtrees or text anywhere. A single-stage design would very likely dominate it: a modern encoder, a depth-aware tree similarity with a virtual root, a full-softmax listwise objective over all 2,125 codes, low dimension with a live radial term, and selection on held-out NAICS index-entry retrieval. That last claim is a prediction that has to be tested, not a result.

## TL;DR

- **State of the art.** The design mixes the Poincaré/Lorentz embedding line (Nickel & Kiela 2017, 2018) with hierarchical contrastive text encoding (HGCLR, Wang et al. 2022; HiMulConE, Zhang et al. 2022). It ignores the three results that matter most here: hyperbolic advantages show up at low dimension (Sala et al. 2018: MAP 0.989 in 2 dimensions vs. 0.87 at 200 dimensions for optimized Poincaré on WordNet); symmetric distances cannot encode ancestor direction (entailment cones, Ganea et al. 2018); and the standard protocol for claiming generalization is held-out edges or nodes (Ganea et al. 2018 withhold non-basic transitive-closure edges).
- **Errors.** There are outright errors: sectors coincide at r = 0; negative eligibility contradicts D; exclusion text is repeated k times; 512-token windows are used on a checkpoint whose card truncates at 256; validation is in-sample with 89.1% triple overlap; distance formulas are wrong for c ≠ 1; and the graph stage cannot compose (385 vs. 31 dims) and would overflow float32 at the radii its LayerNorm implies. Several other choices are defensible but stale: the MoE fusion, per-channel adapter copies, and uncertainty weighting on a hinge.
- **Highest-value actions, in order:** (1) build a held-out benchmark from the NAICS index entries (CorpFacts, a third-party tool that loads the Census 2022 index file, reports "2,125 codes and 20,373 indexed business activities") plus held-out subtrees; (2) replace the D-with-99 target with a depth-aware, virtual-root similarity; (3) remove the hard cap or make it soft, and run a Euclidean/cosine control at equal dimension; (4) collapse the six-term loss into one full-softmax listwise loss over all codes; (5) reclassify exclusions as confusable-neighbor hard negatives for text-to-code retrieval, not geometric repulsion; (6) delete the graph stage unless it beats a parameter-free smoothing baseline on held-out edges across seeds.

---

## 1. Where the methodology sits relative to the state of the art

**Lineage.** S2 is a hybrid-architecture hyperbolic network: a Euclidean encoder, then exp_o onto the hyperboloid, with feature clipping. That is exactly the setup of Guo, Wang, Chen & Yu (CVPR 2022, <https://arxiv.org/abs/2107.11472>). The brief's author list for this paper is wrong; the actual authors are Yunhui Guo, Xudong Wang, Yubei Chen and Stella X. Yu. S2's objective descends from contrastive text encoding with hierarchy guidance (HGCLR, Wang et al., ACL 2022, <https://aclanthology.org/2022.acl-long.491/>). Its distance-matching term L_H is a batch-ratio variant of the cophenetic-correlation regularizers ℓ2-CPCC (Zeng et al., ICLR 2023) and HypStructure (Sinha et al., NeurIPS 2024, <https://proceedings.neurips.cc/paper_files/paper/2024/hash/a5d2da376bab7624b3caeb9f78fcaa2f-Abstract-Conference.html>). S3 descends from HGCN-style tangent-space message passing (Chami et al. 2019).

**What the literature established that the design ignores:**

1. **Hyperbolic gains are a low-dimension phenomenon.** Sala, De Sa, Gu & Ré (ICML 2018, <https://proceedings.mlr.press/v80/sala18a.html>) embed WordNet combinatorially with MAP 0.989 in two dimensions; optimized Poincaré embeddings reach 0.87 at 200 dimensions. Ganea et al. (2018, <https://arxiv.org/abs/1804.01882>) report every WordNet result at 5 and 10 dimensions. Nickel & Kiela (2018, <https://arxiv.org/abs/1806.03417>) say the Lorentz model gives "better embeddings, especially in low dimensions." None of these papers motivates 384 dimensions for a 2,125-node tree.
2. **The radius must stay live to encode depth.** Nickel & Kiela (2017, <https://arxiv.org/abs/1705.08039>) note that "the hierarchical organization of the space is solely determined by the distance of points to the origin". HIE (Yang et al., ICML 2023, <https://arxiv.org/abs/2306.09118>) found that standard hyperbolic models do not reliably learn this on their own; it adds a root-anchored radial regularizer and reports gains "up to 21.4%". HypStructure adds a centering loss that puts the root near the origin. The methodology has a radial regularizer (L_V), but the cap switches it off (C2) and its target is mis-specified for a forest (C1).
3. **Symmetric distances cannot encode ancestor direction.** On WordNet nouns with 50% of non-basic edges in training, Ganea et al. report test F1 of 92.8% (5D) and 94.4% (10D) for hyperbolic cones, against 83.6%/85.3% for Poincaré distance embeddings and 72.8%/78.1% for Euclidean. Suzuki et al. (2019, <https://arxiv.org/abs/1902.04335>) report spherical disk embeddings at 93.4% (5D, 50%). Shadow cones (Yu, Liu, Tseng & De Sa, ICLR 2024, <https://arxiv.org/abs/2305.15215>), which "consistently and significantly outperform existing entailment cone constructions," generalize these. The methodology measures parent retrieval with a symmetric distance, so parent and child differ only through radius, which the cap has removed.
4. **Held-out evaluation is standard.** Ganea et al. remove the WordNet root, keep the transitive reduction ("basic" edges) in training, and withhold "non-basic" edges (578,477 of 661,127 over 82,114 nodes), training with 0/10/25/50% of them and testing with 10 negatives per positive. The methodology has no partition of any kind (S1 assumption 10).
5. **Text-conditioned hierarchy injection does not need a transductive stage.** HGCLR's stated point is that after training "the HGCLR enhanced text encoder can dispense with the redundant hierarchy." That argues directly against S3.
6. **Production NAICS coders exist and set the practical bar.** The Census Bureau's BEACON (Dumbacher, Whitehead, Jeong & Pfeiff, Journal of Data Science 2025, <https://jds-online.org/journal/JDS/article/1423/file/pdf>) trains on over 4.3 million labeled descriptions, including NAICS manual descriptions via the Classification Assistance Tool, predicts hierarchically (2-digit, then 6-digit), and was used "over half a million times" in the 2022 Economic Census. The SS-4 autocoder uses logistic regression on word/bigram dictionaries; in 2015 it autocoded 79% of 3.6 million new business records, about 69% of those to a full 6-digit code. These are the baselines an index-entry benchmark should be compared against.
7. **Independent relatedness signals exist and are public:** Hoberg–Phillips TNIC from 10-K text (<http://hobergphillips.tuck.dartmouth.edu/>), vertical TNIC from BEA I-O commodity text, Neffke & Henning (2013) skill relatedness from labor flows, and Ellison, Glaeser & Kerr (2010) coagglomeration (<https://doi.org/10.1257/aer.100.3.1195>). The method uses none of them, even for validation.

---

## 2. What is wrong, stale, or unsupported — numbered critique points

### Geometry and radius

**C1. All 20 sectors are pulled to the origin, so they coincide.**

- _Concerns:_ S2 L_V; S3 L^g_V; S1 "No super-root". The target sinh r = (λ−2)/2 gives r = 0 for every level-2 code, so all 20 sectors are asked to sit at o (pairwise distance 0) while D^× = 99 asks them to be maximally apart. This is an **error**: the terms contradict each other exactly. Under the cap L_V has zero gradient, so the error is masked by C2, not absent.
- _Category:_ fix an error. _EV:_ high; it decides whether radial depth coding can work once the cap is fixed.
- _Literature:_ Nickel & Kiela 2017 embed the WordNet noun hierarchy under one root at the origin. For a forest, Ganea et al. 2018 remove the root and "co-embed the resulting subgraphs together to prevent overlapping embeddings". HypStructure's centering anchors one root. Standard fix: a virtual root at r = 0 with sectors at r_1 > 0, e.g. target sinh r = (λ−1)/2, leaving sector directions free to spread.

**C2. The hard cap plus DCL at τ = 0.07 actively drives every point onto the cap.**

- _Concerns:_ S2 projection (ν̄ = 2); S2 assumption 1; L_C.
- _Mechanism._ At equal radius r, cosh d = cosh²r − sinh²r·cos θ (hyperbolic law of cosines with r_1 = r_2; confirmed), so the maximum distance is 2r. L_C = d_ap/τ + log Σ exp(−d_aj/τ) keeps falling as negatives separate, and the largest achievable separation grows with r, so the gradient points outward until the cap binds. On the cap, ∂r/∂v = 0 and nothing pulls points back. The historical run (x_0 = 3.7621958 = cosh 2, sd 4e−7) is the expected outcome.
- _Correction to the document._ With the cap, distances lie in [0, 4], so with K = 24 negatives L_C is **bounded**, roughly [ln 24 − 4/0.07, 4/0.07 + ln 24] ≈ [−54.0, 60.3]. "Unbounded below" holds only without the cap. The logit range 4/0.07 ≈ 57 is still extreme, which is why the loss saturates quickly.
- _Category:_ fix an error. _EV:_ high; this is the mechanism that makes the model spherical.
- _Literature:_ Guo et al. 2022 introduce exactly this clip, CLIP(x;r) = min{1, r/‖x‖}·x, to fix vanishing gradients from Poincaré embeddings drifting to the boundary under a _classifier_ loss. They note clipping "shrinks the effective radius". They "fix r to be 1.0 in all the experiments" (baseline HNNs "use a clipping value of 15"), and their supplement reports "a sweet spot in terms of choosing r which is neither too large (causing vanishing gradient problem) nor too small (not enough capacity)." Classification needs no radial depth code, so their "no harm to accuracy" result does not transfer. MERU (Desai et al., ICML 2023, <https://arxiv.org/abs/2304.09172>) uses negative Lorentzian distance with a learned temperature, learnable curvature and an entailment loss it calls "crucial for better structure and interpretability." Remedies ranked: (a) remove the hard cap and add a radius-targeted term with a live gradient (HIE/HypStructure centering plus the corrected target from C1); (b) if a bound is needed, a soft cap v ↦ R·tanh(‖v‖/R)·v/‖v‖ with R ≈ 6–8 (float32 limits in C4); (c) learn the logit scale instead of fixing τ; (d) add an entailment-cone term, or a Busemann/ideal-prototype term (Ghadimi Atigh, Keller-Ressel & Mettes, NeurIPS 2021, <https://arxiv.org/abs/2106.14472>, which places "prototypes on the ideal boundary of the Poincaré ball" with a "penalised Busemann loss").

**C3. Hyperbolic geometry at 384 dimensions with a fixed radius is spherical geometry, and no control could show otherwise.**

- _Concerns:_ S2 (n = 384); top-level Q2.
- On a fixed-radius sphere d is monotone in angle, so every rank statistic equals its cosine counterpart: the model is a cosine model with an unusual temperature. At 384 dimensions angular capacity does not bind for 2,125 points, so even with a live radius the volume-growth argument buys little. The residual case for hyperbolic space is inductive bias (radius encodes depth; geodesics route through ancestors), which operates only with a live radius at low dimension.
- _Category:_ adopt a better method (dimension plus a mandatory control). _EV:_ high; it decides whether "hyperbolic" belongs in the title.
- _Literature:_ Sala et al. 2018; Nickel & Kiela 2018. HypStructure reports gains "especially under low-dimensional scenarios". In Sinha et al.'s Table 1 (CIFAR-10, ResNet-18, SupCon, mean over 3 seeds), distortion δ_rel is 0.232 for flat, 0.174 for ℓ2-CPCC and 0.094 for HypStructure, with test CPCC 0.573 / 0.966 / 0.992. Wang & Isola (ICML 2020, alignment and uniformity on the hypersphere) is the right lens for the realized model (not link-verified).

**C4. The radius budget and numerical regime are mismatched to the claimed geometry.**

- _Concerns:_ S2 cap ν̄ = 2 (Poincaré norm tanh(1) ≈ 0.76); S3 assumption 2.
- Sala et al. show distortion (1+ε) for a tree of diameter l and maximal degree D needs Ω(l·log D/ε) bits, and h-MDS reached perfect MAP only at 512 bits of precision. At 384 dimensions that scale is unnecessary, but then hyperbolic volume growth is unused (C3). Radius 2 has no derivation in either regime.
- _Numerical limits._ Mishne, Wan, Wang & Yang (ICML 2023, <https://arxiv.org/abs/2211.00181>) show that in float64 the Poincaré ball represents points correctly only within radius ≈ 38 and the Lorentz model within ≈ 19 (x_0 = 10^8 makes ⟨x,x⟩\_L round to 0). Applying their propositions to float32 (my derivation, epsilon 2^−24): Lorentz needs x_0 < 4096, i.e. r ≲ 9.0; Poincaré r ≲ 17. So S3's radii ≈ 19.6 at 385 dims (x_0 ≈ 1.6×10^8) are unrepresentable in float32 and at the float64 edge; ≈ 5.5 at 31 dims is safe.
- _Category:_ adopt a better method. _EV:_ medium.

**C5. Radial coordinates are inconsistent, and curvature is wrong for c ≠ 1.**

- _Concerns:_ cross-component assumptions 4–5; S3 assumption 3.
- The correct distance at curvature −c is d_c(x,y) = (1/√c)·arcosh(−c⟨x,y⟩\_L) with ⟨x,x⟩\_L = −1/c (confirmed). The implemented √c·arcosh(−⟨x,y⟩\_L) is an **error** for c ≠ 1, latent because c ≡ 1. With a learnable output scale, curvature is a pure rescaling; it matters only against fixed-scale targets (cap, level targets, fixed τ). Using three radial coordinates (cosh r, sinh r, r) makes level diagnostics non-comparable with level targets.
- _Category:_ fix an error. _EV:_ low now, medium if curvature is freed.
- _Literature:_ MERU reports learned curvature "crucial when scaling model size" (fixed c = 1 ViT-L "is difficult to optimize"), with negligible effect at ViT-B: an optimization aid at scale, not a representational need for a 2,125-node tree. Standardize on r.

### Supervision target (S1)

**C6. Path length with a −½ lineal adjustment ignores depth.**

- _Concerns:_ S1 D definition; S1 assumption 3.
- Under D, level-6 siblings are exactly as similar as level-3 siblings: the textbook Rada path-length defect. Depth-scaled (Wu & Palmer 1994; Leacock & Chodorow 1998) and information-content measures (Resnik 1995; Lin 1998; Jiang & Conrath 1997; intrinsic IC per Seco, Veale & Hayes 2004) correct it. The −½ convention has no source; under Lin/JC lineal pairs are handled naturally because the lowest common ancestor is the ancestor.
- _Category:_ adopt a better method. _EV:_ high; every loss and statistic inherits D.
- _Literature:_ surveyed in Budanitsky & Hirst (2006) and Harispe et al. (2015) (not link-verified here). For NAICS, IC can be economic: IC(c) = −log(share of U.S. employment or establishments in the subtree of c, from QCEW/CBP). I found no published use of economic mass as IC frequency for an industry taxonomy; treat it as a novel, defensible choice.

**C7. The cross-sector sentinel D^× = 99 dominates everything magnitude-sensitive.**

- _Concerns:_ S1 assumption 2; S2 L_H; S4 ρ_P and NDCG.
- 87.9% of pairs sit at 99, about 12× the within-sector maximum of 8. Under L_H's batch-ratio normalization, within-sector targets shrink to about [0.006, 0.09] against about 1.2 cross-sector, so the term becomes a same-sector classifier. This is an **error of specification**: no derivation, and it contradicts the author's own 1,204 cross-sector exclusion pairs.
- _Category:_ fix an error. _EV:_ high.
- _Literature:_ a virtual root is standard (single-rooted WordNet experiments; Ganea et al.'s co-embedded forest). With it, cross-sector leaves are 10 edges apart vs. a within-sector maximum of 8; under Lin/Resnik with root IC 0, cross-sector similarity is exactly 0 with no sentinel. Alternatively keep cross-sector pairs only in rank/contrastive terms.

**C8. Exclusions are treated as "unrelated" repulsive negatives, contradicting NAICS semantics.**

- _Concerns:_ S1 assumption 4; S2 reserved-exclusion slot; top-level Q4.
- The 2022 NAICS Manual (<https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf>) states cross-references as "Establishments primarily engaged in -- … are classified in Industry …", and its FAQ tells coders to "read the full description of the industry (including the narrative, cross-references, and illustrative examples), and determine if that description fits". Cross-references are **disambiguation pointers to where a confusable activity goes**, not assertions of unrelatedness; e.g. 111120 cross-references its sibling 111110 (soybean farming). The author's counts agree: 474 of 2,394 sibling pairs are exclusions. The Manual contains no explicit definitional sentence; this reading rests on the form and the FAQ.
- _Category:_ fix an error. _EV:_ high, for search (hardest confusions) and geometry (it repels near-neighbors).
- _Literature:_ hard negatives (Robinson et al., ICLR 2021) and debiased contrastive learning (Chuang et al., NeurIPS 2020) show the most informative negatives are near the positive, provided false negatives are controlled (not link-verified here).

**C9. Lineal and shallower exclusions are admitted as negatives.**

- _Concerns:_ S1 exclusions (9 lineal references); S1 pool (564 shallower and 50 ancestor/descendant entries per round).
- Repelling one's own ancestor contradicts the hierarchy target; one three-digit anchor's only exclusions are its own descendants. **Error.** Drop lineal pairs from every negative role; keep them as text.
- _Category:_ fix an error. _EV:_ medium (few cases, each a direct contradiction).

**C10. Exclusion text is repeated k times.**

- _Concerns:_ S1 exclusion channel; S2 assumption 6. Repetition pushes 457 of 1,113 texts (41%) past 512 tokens vs. 5 without it, destroys content, and makes mean-pooled embeddings depend on k. **Error (bug).**
- _Category:_ fix an error. _EV:_ medium; trivial fix.

**C11. Positive sampling is uneven and orientation-dependent.**

- _Concerns:_ S1 runtime positives; S1 assumption 7. Ancestors are never positives, siblings one direction only, 852 of 1,012 six-digit codes (84%) never anchors, anchor status depends on numbering. The leaves, which matter most for text-to-code search, get the least supervision.
- _Category:_ adopt a better method. _EV:_ high.
- _Literature:_ Nickel & Kiela 2017 and Ganea et al. 2018 train on the transitive closure (10 corrupted negatives per positive). HiMulConE (Zhang et al., CVPR 2022, <https://arxiv.org/abs/2204.13207>) samples "at least one positive pair from each level in the hierarchy"; its HiConE term ensures pairs farther apart in label space never get a smaller loss than closer pairs, which L_P approximates. With 2,125 codes there is no reason to sample (C13).

**C12. Negative eligibility is keyed on a relation index that disagrees with D.**

- _Concerns:_ S1 assumption 8; S2 assumption 2. 4,675 rows make a grandchild (D = 1.5) a negative for a sibling positive (D = 2), and the weight 1.5^−1.5 ≈ 0.54 makes them among the heaviest negatives; L_C repels them while L_P pulls them inside the positive. Internal inconsistency: **error**. Orientation rules also bar shallower and numerically smaller same-prefix codes from being negatives.
- _Category:_ fix an error. _EV:_ medium; disappears under C13.

### Objective (S2)

**C13. With a fully known 2,125 × 2,125 target, sampled tuples are the wrong estimator; use a full-softmax listwise loss.**

- _Concerns:_ S1 pre-drawn tuples (370,320, each pair about 88 times, fixed negatives); S2 negative selection (fixed set, curriculum inert); S2 Q3.
- A full similarity matrix costs 2,125² ≈ 4.5M distances per step, trivial with a per-epoch cached code bank and gradients through in-batch rows. A ListNet-style cross-entropy between softmax(−d(z_a,·)/τ_d) and softmax(s(a,·)/τ_s), with s the depth-aware similarity of C6, uses every code as a negative with correct graded weight and removes eligibility logic (C12), pre-draw staleness, curriculum, false-negative clustering and the reserved-exclusion slot. Sampled negatives are justified only for large or unlabeled candidate sets.
- _Category:_ adopt a better method. _EV:_ high.
- _Literature:_ MoCo/memory banks (He et al. 2020) and negative-sampling work (Robinson 2021; Chuang 2020; Kalantidis 2020) address large or unlabeled candidate sets (not link-verified). h-MDS (Sala et al. 2018) is the extreme case: with a known metric, fit it directly.

**C14. A six-term objective with two inert terms and inconsistent live terms is not justified.**

- _Concerns:_ S2 total loss; S2 Q2. L_R is identically zero under the cap; L_V has zero gradient under saturation; L_H is a same-sector classifier (C7); L_P conflicts with L_C (C12); L_B only regularizes routing (C18); weights (0.45, 0.35, 0.15, …) have no provenance.
- _Category:_ adopt a better method. _EV:_ medium-high.
- _Literature:_ converge on one listwise term (C13) plus at most a live radial term (HIE; HypStructure centering) and optionally an entailment term (Ganea et al. 2018; MERU).

### Text inputs and encoder

**C15. The 522 inherited descriptions make parent–child pairs text-identical, and χ leaks level.**

- _Concerns:_ S1 text channels; S1 assumption 6; S2 assumption 5.
- The 2022 NAICS structure table confirms 522 six-digit industries "Same as 5-digit" out of 1,012 (490 U.S. detail): unary chains denoting the same industry. Training them as D = 0.5 pairs with near-identical text teaches nothing and inflates parent retrieval (C22). The placeholder χ in the examples channel appears for every level 2–4 code, so channel presence is a level feature.
- _Category:_ fix an error (leakage) and adopt a better method (collapse). _EV:_ medium.
- _Remedy:_ collapse unary chains (evaluate on the six-digit code) or exclude those pairs from training and evaluation; replace χ with masking plus a channel-presence bit; give level as an explicit, ablatable input if wanted.

**C16. The 512-token window on all-MiniLM-L6-v2 is outside the checkpoint's regime.**

- _Concerns:_ S2 token windows; S2 assumption 6. The model card (<https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>) states "input text longer than 256 word pieces is truncated" and that fine-tuning "sequence length was limited to 128 tokens". Position embeddings reach 512, so 512 runs without error but untrained. A maintainer's TRECCovid test in the HF discussion (<https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/discussions/54>; not peer-reviewed) found NDCG@10 0.472 at 256 tokens, rising to 0.513 at 128, 0.555 at 64 and 0.597 at 32, and states 512 "performed worse than truncating those same sequences to 256." **Error** relative to documented behavior.
- _Category:_ fix an error. _EV:_ medium.

**C17. Per-channel frozen copies with separate LoRA adapters are parameter-heavy and unsupported; the encoder is stale.**

- _Concerns:_ S2 channel encoding (4 × 22.7M parameters, 11.5M trainable); S2 Q5. With about 8,500 short texts, full fine-tuning of one 22M-parameter encoder is cheap and removes the duplication; a shared encoder with field-marker prefixes is the standard multi-field retrieval design. Newer encoders (E5, BGE, GTE, Nomic, Arctic, Jina v3, ModernBERT-based, NV-Embed, Qwen-based) offer longer context and higher MTEB retrieval scores; I could not verify 2026 rankings, so choose by the index-entry benchmark (C19). Matryoshka representation learning (Kusupati et al., NeurIPS 2022; not link-verified) gives low-dimensional features for regression.
- _Category:_ adopt a better method. _EV:_ medium.

**C18. The sparse top-2-of-4 MoE and stacked affine projections are unsupported.**

- _Concerns:_ S2 fusion; S2 assumption 7; S2 Q4. Sparse MoE was built for conditional compute at scale (Shazeer et al. 2017; Fedus et al. 2022), not small-data fusion of four fixed embeddings, two of which are placeholders for about half the codes, with L_B computed over a mixed row population. Two affine maps with no nonlinearity collapse to one. I found no evidence for MoE fusion at this scale; it should lose to attention pooling or one encoder over concatenated fields unless shown otherwise.
- _Category:_ adopt a better method (simplify). _EV:_ low-medium.

### Validation, uncertainty and statistics

**C19. Validation is in-sample, so model selection is uninformative.**

- _Concerns:_ S2 assumption 4; S1 assumption 10; Composition step 2; S2 Q6. Same 1,273 anchors and 4,220 pairs; 89.1% of validation triples occur in training; selection, early stopping and LR control read this loss. **Error** for any generalization claim, and every stated use is out-of-sample.
- _Category:_ fix an error. _EV:_ high; nothing else here can be assessed without it.
- _Remedy:_ (a) **Index-entry retrieval.** The 2022 index file keys illustrative activities to six-digit codes (CorpFacts, a third-party tool that loads the Census index "in full," reports "2,125 codes and 20,373 indexed business activities"; I could not open the Census xlsx to confirm). Hold out 20% of entries per code, stratified, remove them from the examples channel, encode as queries, and report MRR and Hit@1/5/10 over the 1,012 six-digit codes plus hierarchical partial credit (shared 2–5-digit prefix of top-1, or its Wu–Palmer similarity). (b) **Subtree holdout** (taxonomy-expansion style, TaxoExpan/HyperExpan; not link-verified): remove whole five-digit subtrees and score parent rank (MRR, Hit@k). (c) **Edge holdout** per Ganea et al. 2018 for reconstruction. Select only on (a).

**C20. There is no uncertainty quantification.**

- _Concerns:_ S2 assumption 8; S4 assumption 5; S4 Q4. One seed, one run, gate thresholds 0.02/0.01/0.05 without a noise estimate: an **error** in inference.
- _Category:_ fix an error. _EV:_ high relative to low cost.
- _Literature:_ Reimers & Gurevych 2017; Bouthillier et al. 2021 (variance from both data sampling and seeds); Sakai 2006; Smucker, Allan & Carterette 2007 (not link-verified). Use ≥5 seeds per arm and a paired bootstrap over queries with seeds nested; report the 95% CI of Δ.

**C21. The global statistics are dominated by the tie block, and several are mislabeled.**

- _Concerns:_ S4 procedure; S4 assumption 2; S4 Q2. With 87.9% of pairs tied at 99, ρ_P is essentially the point-biserial correlation with same-sector membership and ρ_S is dominated by one tied block. "Cophenetic correlation" is a misnomer (no dendrogram). g = (99 − D)/99 puts within-sector grades in [0.919, 0.995] vs. about 0 cross-sector, so NDCG@10 approximates binary same-sector precision.
- _Category:_ fix an error. _EV:_ high relative to low cost.
- _Remedy:_ report (i) AUC of d for same- vs cross-sector pairs; (ii) within-sector Spearman averaged over sectors and queries; (iii) MAP over ancestors and mean rank (Nickel & Kiela 2017); (iv) within-sector distortion (Sala-style); (v) NDCG with integer grades from lowest-common-ancestor depth (Järvelin & Kekäläinen 2002).

**C22. Parent retrieval is confounded by text-identical unary children, and the gate rewards what S3 trains on.**

- _Concerns:_ S4 PR@k; acceptance gate; S4 assumption 3; S4 Q3. For the 522 unary chains the parent's nearest neighbor is trivially its identical child. The gate requires ΔPR@1 ≥ +0.05 while tolerating global losses, and S3 trains directly on parent–child edges. **Error of design.**
- _Category:_ fix an error. _EV:_ medium.
- _Remedy:_ gate on held-out metrics S3 does not train on (C19(a)/(c)); exclude unary pairs; set thresholds from seed/bootstrap noise (superiority CI excludes 0 on the primary metric; non-inferiority CI above −δ on guards); define A = 0 ⇒ deliver Z^(3).

### Graph stage (S3)

**C23. The graph stage is a transductive free embedding with no tie to the text geometry.**

- _Concerns:_ S3 formulation; S3 assumption 1; Composition; S3 Q1–Q2. Free node states initialized at Z^(3) with no retention term can overwrite text information at no cost, cannot embed new descriptions, and create a second, unaligned space. For search from business text it can only move codes relative to S2-encoded queries. For a pure tree it repeats supervision the text stage already had.
- _Category:_ adopt a better method (remove, or retain explicitly). _EV:_ medium.
- _Literature:_ HGCLR dispenses with the hierarchy at inference. Correct & Smooth (Huang et al., ICLR 2021, <https://arxiv.org/abs/2010.13993>) shows shallow models plus label-propagation post-processing "exceed or match" state-of-the-art GNNs on standard transductive benchmarks, making parameter-free smoothing the necessary baseline. If kept, add a retention term such as Σ_ij (d(z^4_i,z^4_j) − d(z^3_i,z^3_j))² and train on held-out-masked edges.

**C24. The stages cannot compose, and the graph stage would overflow.**

- _Concerns:_ Composition (385 vs. 31 dims); S3 assumption 2; cross-component assumption 6. Tangent-space LayerNorm followed by exp_o sets every radius near √(n′+1) (≈ 5.6 at 31 dims, ≈ 19.6 at 385), discarding radial depth and, at 385 dims, exceeding float32 (C4). **Errors** that make S3 undefined or numerically invalid as configured.
- _Category:_ fix an error if S3 is kept. _EV:_ high as a blocker, low if S3 is removed.
- _Literature:_ Lorentz-native layers (HyboNet, Chen et al., ACL 2022; Lorentzian GCN, Zhang et al., WWW 2021) and hyperbolic normalization (Hypformer, Yang et al., KDD 2024) avoid the round trip (not link-verified).

**C25. Excluded pairs are edges, edge weight enters twice, and validation/selection are broken.**

- _Concerns:_ S3 graph (481 excluded pairs are edges); S3 layer; S3 validation. log ω in the score plus ω as multiplier gives a sub-convex aggregate whose mass depends on a node's edge-type mix: hidden type-dependent shrinkage. The validation tail covers only sectors 81 and 92 and shares trainable node states; the last epoch is exported. **Errors.**
- _Category:_ fix an error. _EV:_ low-medium; moot if S3 is removed.
- _Literature:_ GNN evaluation pitfalls (Shchur et al. 2018; Errica et al. 2020; not link-verified) show split choice and missing baselines flip rankings.

**C26. The triplet hinge with adaptive margin, temperature and uncertainty weights is over-parameterized.**

- _Concerns:_ S3 objective; S3 assumption 8; S3 Q5. Dividing a hinge by T_e is a pure rescaling absorbed by e^(−s_1). Kendall, Gal & Cipolla (CVPR 2018) derived their weighting from Gaussian/Boltzmann likelihoods; on a non-likelihood hinge the ½e^(−s)L + ½s form has no noise-model meaning and just drives s toward log L. The adaptive margin never hits its clip.
- _Category:_ adopt a better method (remove). _EV:_ low.
- _Literature:_ Kendall et al. 2018; GradNorm (Chen et al. 2018) (not link-verified).

### Scope and external validity

**C27. Supervision and evaluation come from one source, so evaluation is circular.**

- _Concerns:_ cross-component assumption 1; S1 assumption 1; S4 assumption 1; top-level Q3/Q7. A perfect score reproduces what exact tree operations (ancestor indicators, lowest-common-ancestor depth) give for free.
- _Category:_ extend scope. _EV:_ high for economic and clustering uses.
- _Literature and data:_ Hoberg–Phillips TNIC (public pairwise 10-K similarity; TNIC-3 at roughly 3-digit SIC granularity; firm-level, so aggregate via a firm-to-NAICS map); vertical TNIC (Frésard, Hoberg & Phillips); I-O vertical relatedness (Fan & Lang 2000; public BEA I-O tables at detailed levels mapping to roughly 4–6-digit NAICS); skill relatedness (Neffke & Henning 2013, SMJ 34(3):297–316); coagglomeration (Ellison, Glaeser & Kerr 2010, AER 100:1195–1213: input–output linkages the most important driver, "closely followed" by labor sharing; Diodato, Neffke & O'Clery 2018: drivers vary by industry and over time). Use these only as held-out validation targets to keep them independent.

**C28. The economic benchmark is weakly designed.**

- _Concerns:_ S4 economic benchmark; S4 assumption 6. Ridge (penalty 1, unscaled) on 384 coordinates with a single 80/20 split, a one-hot comparator that cannot predict held-out codes, and no covariates-only comparator. A cross-sectional level regression mostly measures whether sector and size are encoded.
- _Category:_ fix an error, then extend scope. _EV:_ medium.
- _Remedy:_ (i) comparators: 2/3/4/5-digit ancestor indicators (which can predict held-out six-digit codes), frozen encoder, TF-IDF, covariates-only; (ii) standardize and tune the penalty by nested CV; (iii) repeated grouped CV with groups = 4-digit parents; (iv) prefer time-dimension outcomes (employment growth, co-movement) with time-respecting splits (train on pre-2022 QCEW changes, test on 2022–2025) and pairwise targets (I-O, labor flows, coagglomeration) scored by rank correlation on held-out pairs.

**C29. Stage spaces are not aligned, and the deliverable is undefined.** _Concerns:_ Composition properties 1–2; assumption 7. Z^(4) exists only for C, new text maps into Z^(3), no alignment exists, and A = 0 has no deliverable. **Error.** _Category:_ fix an error. _EV:_ medium; resolved by removing S3.

**C30. The asymmetric ancestor relation is not represented.** _Concerns:_ S2 formulation; S4 PR@k. For level-flexible search a query should retrieve its six-digit code and "entail up" to its sector; a symmetric distance with a dead radius cannot express this. _Category:_ adopt a better method (optional). _EV:_ medium. _Literature:_ entailment cones (Ganea et al. 2018), disk embeddings (Suzuki et al. 2019), shadow cones, box embeddings (Vilnis et al. 2018; Dasgupta et al. 2020; Boratko et al. 2021; not link-verified), MERU's entailment loss.

---

## 3. Improvements ranked by expected value

| Rank | Action                                                                                                           | Category              | C-points     |
| ---- | ---------------------------------------------------------------------------------------------------------------- | --------------------- | ------------ |
| 1    | Held-out benchmarks: index-entry retrieval, subtree holdout, edge holdout; select only on index-entry retrieval  | fix an error          | C19, C22     |
| 2    | ≥5 seeds plus paired bootstrap CIs; stratified statistics (AUC, within-sector Spearman, MAP over ancestors)      | fix an error          | C20, C21     |
| 3    | Virtual root plus depth-aware similarity (Lin/JC with intrinsic or employment-based IC)                          | adopt a better method | C6, C7, C1   |
| 4    | Fix the radius: no hard cap (or soft cap), corrected level targets, learned logit scale                          | fix an error          | C1, C2       |
| 5    | Euclidean/cosine twin and dimension sweep (8, 16, 32, 64, 384)                                                   | adopt a better method | C3, C4       |
| 6    | One full-softmax listwise loss over all 2,125 codes, optional live radial term                                   | adopt a better method | C11–C14      |
| 7    | Exclusions as confusable hard negatives only in text→code retrieval; drop lineal pairs; remove k-fold repetition | fix an error          | C8–C10       |
| 8    | Collapse unary chains; mask empty channels with presence bits; stay within 256 tokens                            | fix an error          | C15, C16     |
| 9    | One shared, fully fine-tuned encoder with field markers; benchmark 2–3 modern encoders                           | adopt a better method | C17, C18     |
| 10   | Delete S3, or run the decision experiment (top-level Q6)                                                         | adopt a better method | C23–C26, C29 |
| 11   | External validation (TNIC, I-O, labor flows, coagglomeration); redesigned economic benchmark                     | extend scope          | C27, C28     |
| 12   | Standardize radial coordinates; fix curvature formulas                                                           | fix an error          | C5           |
| 13   | Optional entailment/cone term                                                                                    | adopt a better method | C30          |

---

## 4. Answers to every open question

### S1

**S1 Q1 — Is path length with the half-step lineal adjustment defensible?** No, not as the primary target: it ignores depth (the Rada defect) and the −½ convention has no source. Use Lin, s = 2·IC(lca)/(IC(i)+IC(j)), or Jiang–Conrath, with a virtual root, taking IC intrinsically (Seco et al. 2004) or economically (−log employment share of the subtree, from QCEW). Economic IC suits the economic-feature use but has no NAICS precedent I could find, so test its sensitivity against intrinsic IC. Wu–Palmer is an IC-free fallback. Learned metrics belong only in validation against external signals (C27).

**S1 Q2 — How to encode cross-sector pairs?** A finite path through a virtual root: standard (single-rooted WordNet experiments; Ganea et al.'s rooted co-embedding), and under Lin/Resnik it gives similarity exactly 0. Keep cross-sector pairs in rank/contrastive terms; exclude or down-weight them in magnitude-matching terms; never use a constant 12× the within-sector maximum. A data-driven value (I-O, labor flows) is legitimate only as a validation target, or validation stops being independent.

**S1 Q3 — Role of cross-references; lineal exclusions?** The Manual's form ("Establishments primarily engaged in -- … are classified in …") and its instruction to read cross-references when deciding fit mark them as **disambiguation pointers to confusable neighbors**. Use them (a) as de-duplicated text in the exclusion channel; (b) as a distinct "boundary" relation supplying hard negatives only in the text-query→code retrieval objective; (c) never as geometric repulsion between code embeddings. Drop lineal exclusions from every negative role; they are scope notes, not dissimilarity.

**S1 Q4 — Is the positive set sound?** No. Positives should cover the transitive closure in both directions (Nickel & Kiela 2017; Ganea et al. 2018), or at least one positive per level per anchor (HiMulConE), with every code, especially the 1,012 leaves, as an anchor. With 2,125 codes, a full listwise target (C13) makes sampling moot.

**S1 Q5 — Empty channels and inherited descriptions?** Mask, don't placeholder: drop absent channels from fusion, add presence indicators, and make level an explicit, ablatable input if wanted. Collapse the 522 unary chains (confirmed "Same as 5-digit" in the NAICS structure table). For the 154 inherited four-digit descriptions, prefer the official text even when short rather than choosing arbitrarily. Deduplicate near-identical texts before splitting.

### S2

**S2 Q1 — Is hyperbolic geometry doing work; how to keep depth on the radius; does 384 help?** As realized, no: every point sits at r = 2, so the representation is spherical (C2, C3). Keep depth expressible with the C2 remedies: no hard cap, a live radial target with a virtual root at r = 0, optionally entailment cones or a Busemann objective, and a learned logit scale; a soft cap at R ≈ 6–8 stays below the float32 Lorentz limit of about 9. At 384 dimensions hyperbolic is unlikely to beat cosine; the evidence (Sala et al.; Ganea et al.; Nickel & Kiela 2018; HypStructure) places its advantage at low dimension. Run the dimension sweep with a Euclidean twin.

**S2 Q2 — Six terms or one principled loss?** One: a listwise cross-entropy between softmax(−d/τ_d) over all codes and a target distribution from the depth-aware similarity, which subsumes L_C, L_H and L_P and inherits HiConE's ordering property. Add at most one radial term and, if direction matters, one entailment term. Remove L_R, L_B (with the MoE), the curriculum and the clustering.

**S2 Q3 — How to choose negatives for 2,125 codes?** Every code is a negative, with graded weights from the target distribution, scored against a per-epoch cached code bank. Hard negatives matter only in the query→code loss, where cross-referenced codes and same-parent siblings are the natural hard set. Fixed inverse-distance pre-draws are strictly worse: stale, weighted the wrong way, and inconsistent with D (C12).

**S2 Q4 — Is sparse MoE justified?** No evidence supports it at this scale. Preferred alternatives: (1) one encoder over concatenated fields with markers; (2) shared-encoder per-field encodings fused by attention pooling with presence masking; (3) concatenation plus a linear layer. Keep MoE only if it beats (1)–(2) on held-out retrieval across seeds.

**S2 Q5 — Is the 6-layer 384-dim encoder with per-channel adapters appropriate?** Defensible as a cheap baseline, not as the final design. Use one shared, fully fine-tuned encoder; benchmark 2–3 current embedding models on the index-entry task; keep inputs within the trained length (C16). Also run a frozen encoder with a light trained head as the "how much does fine-tuning add" control.

**S2 Q6 — What validation makes selection meaningful?** The C19 benchmark: stratified held-out index entries as queries (primary metric MRR@10 on six-digit codes with hierarchical partial credit), plus held-out five-digit subtrees and held-out edges. Freeze the test split, tune on a separate validation split, report test once, with seeds and bootstrap CIs.

### S3

**S3 Q1 — What can a transductive graph stage add; minimal controls?** For a pure tree already used as supervision: very little, and nothing for new-text queries. Controls: (a) extra text-stage optimization at matched compute; (b) parameter-free smoothing z*i ← exp_o((1−α)·log_o z_i + α·mean*{j∈pa/children(i)} log_o z_j), α tuned on validation (Correct & Smooth logic); (c) node texts shuffled before initialization; (d) random initialization. Decision rule in top-level Q6.

**S3 Q2 — Retain the text geometry explicitly?** Yes, if S3 survives: a distance-distillation term to Z^(3), or restrict S3 to a low-rank correction of Z^(3). I found no hyperbolic-GNN paper addressing retention for text-initialized free node states specifically; teacher distillation is the general pattern.

**S3 Q3 — Tangent-space message passing plus LayerNorm?** Inappropriate as configured: Euclidean LayerNorm fixes the tangent norm and hence the radius near √(n′+1), deleting radial depth and overflowing at 385 dims (C24). Use Lorentz-native layers (HyboNet), or aggregate by Lorentz centroid/Einstein midpoint with radius-preserving hyperbolic normalization. Better, see top-level Q6.

**S3 Q4 — Which edges; how do exclusions enter?** If kept: parent–child edges plus self-loops only. Sibling and multi-generation edges add shortcuts that 2 layers already reach and speed over-smoothing (Li, Han & Wu 2018; Oono & Suzuki 2020; not link-verified). Exclusions must not be message-passing edges; if used at all, as a typed "boundary" edge with a learned (possibly negative) gate, validated on held-out data.

**S3 Q5 — Triplet hinge with adaptive margin and uncertainty weighting?** No: the margin clip never binds, T_e is redundant with e^(−s_1), and homoscedastic weighting has no likelihood meaning on a hinge (C26). Standard hierarchy objectives are the negative-sampling softmax over distances (Nickel & Kiela), distortion minimization (Sala et al. h-MDS) and cone/margin entailment losses (Ganea et al.). With a known target, use the listwise loss from S2 Q2.

### S4

**S4 Q1 — What establishes usefulness; essential baselines?** Held-out index-entry retrieval (primary), held-out subtree placement, agreement with external relatedness (TNIC, I-O, labor flows, coagglomeration) on held-out pairs, and time-respecting prediction for the economic use. An independently labeled establishment-description set would be the true external test; BEACON's >4.3M write-ins are not public, and public candidates (SAM.gov, OSHA records) are untested here. Essential baselines: BM25/TF-IDF over the same NAICS text (BEACON-style dictionaries are the production reference), a frozen pretrained encoder, taxonomy-only ancestor indicators, exact tree operations for reconstruction metrics, and the Euclidean/cosine twin.

**S4 Q2 — Standard reconstruction metrics; the tie block?** MAP over ancestors and mean rank (Nickel & Kiela 2017); distortion and MAP (Sala et al. 2018); F1 on withheld non-basic edges (Ganea et al. 2018); MRR and Hit@k of the true parent for inserted nodes (taxonomy expansion). For the tie block: report sector-separation AUC separately, compute rank statistics within sector or per query over within-sector candidates, use integer graded relevance from lowest-common-ancestor depth, and never headline a global correlation over the 2.26M pairs.

**S4 Q3 — A gate that doesn't reward what S3 trains on?** Gate on metrics disjoint from S3's training signal: held-out index-entry retrieval and held-out edge F1 on edges masked from S3's graph and loss. Require superiority on the primary metric (CI excludes 0) and non-inferiority on guards (CI lower bound > −δ, δ from seed variance). Exclude unary chains. A = 0 ⇒ deliver Z^(3).

**S4 Q4 — Uncertainty when comparing two embeddings of the same code set?** Two sources: training randomness and the finite query set. Use ≥5 seeds per arm (Bouthillier et al. 2021 argue for accounting for both) and a paired bootstrap over queries, pairing arms on the same resample, with seeds nested (Sakai 2006; Smucker et al. 2007). For pairwise statistics, bootstrap _codes_, not pairs, because pairs sharing a code are dependent. Report Δ with a 95% CI.

### Top-level

**Q1 — Position and a dominating simpler design.** See Section 1: the design hybridizes 2017–2019 hyperbolic taxonomy embedding with 2022 hierarchical contrastive text encoding and lacks both lines' evaluation standards. The likely dominating design is the single-stage one in the bottom line, with a Euclidean twin as control; an HGCLR-style hierarchy-in-encoder design should also be tried if search is primary.

**Q2 — Is hyperbolic justified; would Euclidean/spherical be equivalent?** As realized the model _is_ spherical, so a cosine model with the same objective is equivalent by construction. With a live radius, hyperbolic is justified only if it beats the Euclidean twin at matched dimension on held-out metrics; the literature predicts any gap appears at 8–32 dimensions and vanishes by a few hundred.

**Q3 — Supervising and evaluating on the same target?** Only for a reconstruction claim, and only with held-out edges or nodes. For the stated uses, no. Train on the taxonomy (it is authoritative about membership) but evaluate on held-out text, independent labeled descriptions if obtainable, and external relatedness that never enters training.

**Q4 — Consistent exclusion treatment?** One rule: exclusions are boundary/confusable pairs, used (a) as de-duplicated text; (b) as hard negatives only in query→code retrieval; (c) never as repulsion between code embeddings or as message-passing edges. Lineal exclusions are text-only.

**Q5 — Repair or remove each inert/nominal mechanism?**

- _Radius penalty L_R:_ remove; inert under the cap and redundant with a correct level term.
- _Curriculum mining:_ remove for the code–code loss (full softmax). For query→code retrieval, hard negatives have support (Robinson et al. 2021) if false negatives are controlled, which here are known exactly from the tree.
- _False-negative clustering:_ remove; false negatives are known from the taxonomy, so k-means estimation (PCL/SwAV-style prototypes) is unnecessary.
- _Adaptive margins:_ remove; the clip never binds and this form has no literature support.
- _Learnable curvature:_ repair only if moving to large models (MERU: helps optimization at ViT-L scale); otherwise fix c = 1 and learn a logit scale. Fix the formulas either way.
- _Graph-curriculum filters:_ remove with the graph stage.
- _Temperature schedule with learned log-variance weights:_ remove; redundant on a hinge (C26).

**Q6 — Minimal experiment deciding whether S3 should exist.** From one fixed Z^(3), repaired to compose (same dimension), mask 20% of parent–child edges, stratified by sector, from S3's graph and all training signals. Run five arms at matched compute, 5 seeds each: (A) Z^(3); (B) Z^(3) plus extra text-stage epochs; (C) parameter-free tree smoothing with α tuned on a validation edge fold; (D) S3; (E) S3 with node texts shuffled before encoding. Evaluate masked-edge parent MRR and Hit@1 (excluding unary chains) and held-out index-entry MRR (queries encoded by S2, scored against each arm's code points). Keep S3 only if D beats both B and C on masked-edge MRR with a paired-bootstrap 95% CI excluding 0 **and** is non-inferior on index-entry MRR. If E ≈ D, S3 learns the tree independently of text. I predict S3 fails; that is a prediction, not a finding.

**Q7 — Estimand for search, clustering and economic features.**

- _Search:_ P(code | description) for new descriptions, measured by held-out index-entry MRR/Hit@k with hierarchical partial credit. Implies a query→code retrieval loss (index entries as queries, all codes as candidates, cross-references as hard negatives) with the code–code taxonomy loss as a regularizer.
- _Clustering:_ agreement with _external_ relatedness, not 2-/3-digit groupings (circular). Implies validation against TNIC, I-O, labor flows and coagglomeration.
- _Economic features:_ out-of-sample predictive gain over taxonomy-only features under time-respecting, group-aware splits. Implies a low-dimensional output (Matryoshka-style or small n) and comparators that can predict held-out codes.

---

## Caveats

- **Citation verification.** Links opened or abstract-confirmed in this review are the ones given inline with URLs above (plus Pal et al. HyCoCLIP, arXiv 2410.06912, and Yeh et al. DCL, DOI 10.1007/978-3-031-19809-0_38). Shadow cones is confirmed as Yu, Liu, Tseng & De Sa, ICLR 2024, and HypStructure's distortion figures are confirmed from its Table 1 (arXiv 2412.01023). Every reference marked "not link-verified" is a standard citation to check before publication. I did not verify 2026 MTEB standings.
- **NAICS index file size.** CorpFacts reports "2,125 codes and 20,373 indexed business activities" from the Census 2022 index, and other third-party sources agree (one notes 112130 and 541120 are missing from the Census file). I could not confirm this from the Census xlsx. The 2022 Manual has no explicit definitional sentence for "cross-references"; my reading rests on their form and the FAQ.
- **Derived numbers.** The float32 radius limits (≈ 9 Lorentz, ≈ 17 Poincaré) and DCL bounds (≈ [−54, 60]) are my derivations from cited formulas, not published figures.
- **Predictions vs. findings.** Claims that a simpler design "dominates" or that S3 will fail the decision experiment are literature-based predictions to be tested with the C19 benchmark.
- **Guo et al. scope.** Their "no harm to accuracy" finding concerns classification, not hierarchy reconstruction, so it does not license the cap here.
