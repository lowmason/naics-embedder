# Advanced Features

This document describes the advanced features and enhancements implemented in the NAICS Hyperbolic Embedding System.

## Table of Contents

1. [Hard Negative Mining](#hard-negative-mining)
2. [Router-Guided Negative Mining](#router-guided-negative-mining)
3. [Global Batch Sampling](#global-batch-sampling)
4. [Structure-Aware Dynamic Curriculum](#structure-aware-dynamic-curriculum)
5. [Multi-Level Supervision](#multi-level-supervision)
6. [Hyperbolic K-Means Clustering](#hyperbolic-k-means-clustering)
7. [Norm-Adaptive Margins](#norm-adaptive-margins)

---

## Hard Negative Mining

**Issue #15**: Lorentzian Hard Negative Mining

### Overview

Hard negative mining selects the most challenging negatives for training by choosing negatives that are geometrically close to anchors in hyperbolic space. This provides a stronger learning signal than random negative sampling.

### Implementation

```python
LorentzianHardNegativeMiner(
    curvature=1.0,
    safety_epsilon=1e-5
)
```

### Process

1. For each anchor, compute Lorentzian distances to all candidate negatives
2. Select top-k negatives with smallest distances (hardest negatives)
3. Use these hard negatives in the contrastive loss

### Benefits

- **Better Learning Signal**: Hard negatives provide more informative gradients
- **Faster Convergence**: Model learns to distinguish between similar codes more effectively
- **Improved Representations**: Embeddings develop finer-grained distinctions

### Configuration

Enabled automatically when `enable_hard_negative_mining` is set in the curriculum scheduler.

---

## Router-Guided Negative Mining

**Issue #16**: Router-Guided Negative Mining for MoE

### Overview

Router-guided negative mining prevents "Expert Collapse" in the Mixture-of-Experts layer by selecting negatives that confuse the gating network. These negatives are identified by having similar expert probability distributions to anchors.

### Implementation

```python
RouterGuidedNegativeMiner(
    metric='kl_divergence',  # or 'cosine_similarity'
    temperature=1.0
)
```

### Process

1. Compute gate probabilities for anchors and negatives
2. Measure confusion using KL-divergence or cosine similarity
3. Select negatives with highest confusion (most similar gate distributions)
4. Mix router-hard negatives with embedding-hard negatives (default: 50/50)

### Metrics

- **KL-Divergence**: Measures how different two probability distributions are
  - Lower KL-divergence = more confusion (similar distributions)
- **Cosine Similarity**: Measures the angle between two probability vectors
  - Higher cosine similarity = more confusion (similar directions)

### Benefits

- **Prevents Expert Collapse**: Ensures all experts are utilized effectively
- **Diverse Negative Sampling**: Captures negatives that confuse the routing mechanism
- **Better MoE Training**: Improves expert specialization and load balancing

### Configuration

Enabled automatically when `enable_router_guided_sampling` is set in the curriculum scheduler.

---

## Global Batch Sampling

**Issue #19**: Global Batch Sampling for Distributed Training

### Overview

Global batch sampling enables hard negative mining across all GPUs in distributed training. This is crucial for finding meaningful "Cousin" negatives that may not appear in small local batches (e.g., size 32).

### Implementation

Automatically enabled when:
- Distributed training is active (`torch.distributed.is_initialized()`)
- Multiple GPUs available (`world_size > 1`)
- Hard negative mining or router-guided sampling is enabled

### Process

1. **Gather Phase**: Collect negative embeddings from all GPUs using `torch.distributed.all_gather`
2. **Distance Computation**: Compute distances from local anchors to all global negatives
3. **Selection**: Select top-k hardest negatives from the global pool
4. **Gradient Flow**: Gradients flow back through the all_gather operation to all GPUs

### Memory Management

**Example Configuration:**
- `batch_size=32`, `world_size=4`, `k_negatives=24`
- Global negatives: ~9MB per GPU
- Similarity matrix: ~393KB per batch

**Monitoring:**
- `train/global_batch/global_negatives_memory_mb`: Memory usage for global negatives
- `train/global_batch/similarity_matrix_memory_mb`: Memory usage for similarity matrix
- `train/global_batch/global_batch_size`: Effective global batch size
- `train/global_batch/global_k_negatives`: Number of negatives per anchor globally

### Benefits

- **Larger Negative Pool**: Access to negatives from all GPUs, not just local batch
- **Better Hard Negatives**: More likely to find meaningful "Cousin" relationships
- **Improved Training**: Higher quality negative samples lead to better representations

### Gradient Flow

The implementation uses `torch.distributed.all_gather` which preserves gradients:
- If input embeddings require gradients, gathered tensors also have gradients
- During backpropagation, gradients are scattered back to each rank
- All GPUs receive gradient updates for their embeddings

---

## Structure-Aware Dynamic Curriculum

**Issue #12**: Structure-Aware Dynamic Curriculum (SADC)

### Overview

The Structure-Aware Dynamic Curriculum progressively enables advanced training features based on training progress. This allows the model to start with simple training and gradually introduce complexity.

### Curriculum Phases

#### Phase 0: Early Training
- Basic contrastive learning
- Standard negative sampling
- No hard negative mining
- No false negative masking

#### Phase 1: Mid Training
- Enable hard negative mining
- Enable false negative clustering
- Track negative sample type distribution

#### Phase 2: Advanced Training
- Enable router-guided sampling
- Mix embedding-hard and router-hard negatives
- Full curriculum features active

### Features

- **Automatic Phase Transitions**: Phases activate based on epoch thresholds
- **Negative Sample Tracking**: Logs distribution of negative types (child/sibling/cousin/distant)
- **Smooth Progression**: Gradually introduces complexity as model improves

### Configuration

Managed automatically by the `CurriculumScheduler` class. Phase transitions are based on:
- Current epoch
- Training progress
- Curriculum configuration

---

## Multi-Level Supervision

**Issue #18**: Multi-Level Supervision

### Overview

Multi-level supervision allows each anchor to have multiple positive examples at different hierarchy levels. This provides richer supervision signals and explicitly models relationships at different levels.

### Implementation

- Batch is expanded so each positive level is a separate training item
- Loss naturally sums over all positive levels
- Provides gradient accumulation across hierarchy levels

### Benefits

- **Rich Supervision**: Model learns from multiple positive relationships simultaneously
- **Hierarchy Awareness**: Explicitly models relationships at different levels
- **Better Representations**: Captures hierarchical structure more effectively

### Usage

Enabled automatically when the dataset provides `positive_levels` in the batch.

---

## Hyperbolic K-Means Clustering

**Issue #17**: Hyperbolic K-Means for False-Negative Detection

### Overview

Unlike standard Euclidean K-Means, the system uses Hyperbolic K-Means that operates directly in Lorentz space. This is more appropriate for hyperbolic embeddings and preserves geometric structure during clustering.

### Implementation

```python
HyperbolicKMeans(
    n_clusters=500,
    curvature=1.0,
    max_iter=100,
    tol=1e-4
)
```

### Process

1. Initialize cluster centroids in Lorentz space
2. Assign embeddings to nearest centroid using Lorentzian distances
3. Update centroids in hyperbolic space
4. Repeat until convergence

### Benefits

- **Geometric Consistency**: Clusters respect hyperbolic geometry
- **Better False-Negative Detection**: More accurate cluster assignments
- **Preserves Structure**: Maintains hierarchical relationships during clustering

### Usage

Used for false-negative mitigation:
1. Periodically cluster embeddings (default: every 5 epochs after epoch 10)
2. Identify negatives sharing cluster label with anchor
3. Mask these false negatives in the contrastive loss

---

## Norm-Adaptive Margins

### Overview

Norm-adaptive margins adapt to the hyperbolic radius of anchors, providing more appropriate margins for different regions of hyperbolic space.

### Formula

```
m(a) = m₀ * sech(||a||_L)
```

where:
- `m₀` is the base margin (default: 0.5)
- `||a||_L` is the Lorentz norm (hyperbolic radius) of anchor `a`
- `sech` is the hyperbolic secant function: `sech(x) = 1 / cosh(x)`

### Behavior

- **Small Norm (Near Origin)**: Margin is close to base margin `m₀`
- **Large Norm (Far from Origin)**: Margin decreases as `sech(||a||_L)` approaches 0
- **Adaptive Difficulty**: Anchors near the leaf boundary (large norm) have smaller margins

### Benefits

- **Adaptive Difficulty**: Margins adapt to the hyperbolic geometry
- **Geometric Awareness**: More appropriate margins for different regions of hyperbolic space
- **Better Training**: Prevents over-penalization of anchors far from origin

### Configuration

Computed automatically when hard negative mining is enabled. Logged metrics:
- `train/curriculum/adaptive_margin_mean`: Mean adaptive margin
- `train/curriculum/adaptive_margin_min`: Minimum adaptive margin
- `train/curriculum/adaptive_margin_max`: Maximum adaptive margin

---

## Summary

These advanced features work together to improve training:

1. **Hard Negative Mining** provides challenging negatives for better learning
2. **Router-Guided Sampling** prevents expert collapse in MoE
3. **Global Batch Sampling** enables access to negatives from all GPUs
4. **Structure-Aware Curriculum** gradually introduces complexity
5. **Multi-Level Supervision** provides richer training signals
6. **Hyperbolic K-Means** improves false-negative detection
7. **Norm-Adaptive Margins** adapt to hyperbolic geometry

All features are automatically enabled based on training configuration and progress, requiring no manual intervention.

