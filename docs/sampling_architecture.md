---
title: Sampling Architecture
---

# Sampling Architecture: Data Layer vs Model Layer

This page clarifies the split between the streaming data pipeline and the model during curriculum-driven training.

## Data Layer (Streaming Dataset)
- Build candidate pools from precomputed triplets and taxonomy indices.
- Phase 1 sampling:
  - Inverse tree-distance weighting (`P(n) ∝ 1 / d_tree(a, n)^α`).
  - Sibling masking (`d_tree <= 2` set to zero).
  - Explicit exclusion mining (`excluded_codes` map) with high-priority weights and an `explicit_exclusion` flag on sampled negatives.
- Outputs:
  - Tokenized anchors/positives.
  - Negatives annotated with `relation_margin`, `distance_margin`, and `explicit_exclusion`.
  - Shared negatives reused for multi-level positives (ancestor supervision).

## Model Layer (NAICSContrastiveModel)
- Reads curriculum flags from `CurriculumScheduler`.
- Phase 2+ sampling:
  - Embedding-based hard negative mining (Lorentzian distance).
  - Router-guided negative mining (gate confusion).
  - Norm-adaptive margins.
- Phase 3 sampling:
  - False-negative masking via clustering/pseudo-labels.
- Logging:
  - Negative relationship distribution.
  - Tree-distance bins.
  - Router confusion and adaptive margins.

## Interface Contract
- **Inputs expected from data layer:** negative embeddings and optional `explicit_exclusion` flag; negatives per anchor already filtered/weighted for Phase 1.
- **Curriculum flags influence:**
  - Phase 1 flags (`use_tree_distance`, `mask_siblings`) act in the data layer.
  - Phase 2/3 flags (`enable_hard_negative_mining`, `enable_router_guided_sampling`, `enable_clustering`) act in the model layer.
- **Re-sampling:** Phase 1 weighting occurs in streaming_dataset; later phases reuse provided negatives but reorder/mix based on mining strategies.
