# System Architecture

This document provides a comprehensive overview of the NAICS Hyperbolic Embedding System architecture, detailing each component, data flow, and design decisions.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Data Flow](#data-flow)
4. [Training Pipeline](#training-pipeline)
5. [Mathematical Foundations](#mathematical-foundations)
6. [Design Decisions](#design-decisions)

---

## System Overview

The NAICS Hyperbolic Embedding System is a unified framework for learning hierarchical representations of the North American Industry Classification System (NAICS) taxonomy. The system combines:

- **Multi-channel text encoding** using transformer-based encoders
- **Mixture-of-Experts (MoE) fusion** for adaptive feature combination
- **Hyperbolic contrastive learning** in Lorentz space
- **False-negative mitigation** via curriculum clustering
- **Hierarchy preservation** through specialized loss functions
- **Optional HGCN refinement** for graph-based structure integration

The final output is a set of **Lorentz-model hyperbolic embeddings** that preserve the hierarchical structure of the NAICS taxonomy while capturing semantic relationships from text descriptions.

---

## Architecture Components

### 1. Multi-Channel Text Encoder

The system processes each NAICS code through four independent text channels:

- **Title**: Short code name (e.g., "Software Publishers")
- **Description**: Detailed explanation of the code
- **Examples**: Representative examples of businesses in this category
- **Excluded**: Codes explicitly excluded from this category

#### Implementation

```python
MultiChannelEncoder(
    base_model_name='sentence-transformers/all-mpnet-base-v2',
    lora_r=8,
    lora_alpha=16,
    num_experts=4,
    curvature=1.0
)
```

**Key Features:**

- **LoRA Adaptation**: Each channel uses a separate LoRA-adapted transformer encoder
  - Reduces trainable parameters while maintaining expressiveness
  - Universal LoRA targeting (`target_modules='all-linear'`) works with any transformer architecture
  - Default: `r=8`, `alpha=16`, `dropout=0.1`

- **Gradient Checkpointing**: Enabled by default to reduce memory usage
  - Trades computation for memory during backpropagation
  - Critical for large batch sizes or limited GPU memory

- **Channel Independence**: Each channel learns specialized representations
  - Title encoder focuses on concise category names
  - Description encoder captures detailed semantics
  - Examples encoder learns from representative instances
  - Excluded encoder learns negative semantics

**Output**: Four Euclidean embeddings `(E_title, E_desc, E_examples, E_excluded)`, each of dimension `embedding_dim` (typically 768 for MPNet).

---

### 2. Mixture-of-Experts (MoE) Fusion

The four channel embeddings are concatenated and passed through a Mixture-of-Experts layer for adaptive fusion.

#### Architecture

```python
MixtureOfExperts(
    input_dim=embedding_dim * 4,  # 4 channels concatenated
    hidden_dim=1024,
    num_experts=4,
    top_k=2
)
```

**Components:**

1. **Gating Network**: Linear layer that computes expert selection scores
   - Input: Concatenated channel embeddings `(batch_size, embedding_dim * 4)`
   - Output: Expert scores `(batch_size, num_experts)`

2. **Top-K Selection**: Selects the `top_k=2` most relevant experts per input
   - Reduces computation while maintaining expressiveness
   - Softmax normalization over selected experts

3. **Expert Networks**: Each expert is a 2-layer MLP:

   ```
   Linear(input_dim → hidden_dim) → ReLU → Dropout(0.1) → Linear(hidden_dim → input_dim)
   ```

4. **Load Balancing**: Auxiliary loss ensures even expert utilization
   - Prevents expert collapse (all inputs routed to same expert)
   - Coefficient: `load_balancing_coef=0.01`

**Output**: Fused Euclidean embedding `E_fused` of dimension `embedding_dim`, projected back from `embedding_dim * 4` via a linear projection layer.

---

### 3. Hyperbolic Projection

The fused Euclidean embedding is projected into **Lorentz-model hyperbolic space** to align with hierarchical structure.

#### Lorentz Model

The Lorentz model represents hyperbolic space as points on a hyperboloid:

- **Coordinates**: `(x₀, x₁, ..., xₙ)` where:
  - `x₀` is the time coordinate (hyperbolic radius)
  - `x₁...xₙ` are spatial coordinates
- **Constraint**: `-x₀² + x₁² + ... + xₙ² = -1/c` (Lorentz inner product)
- **Curvature**: `c` controls the curvature of the space (default: `c=1.0`)

#### Implementation

```python
HyperbolicProjection(
    input_dim=embedding_dim,
    curvature=1.0
)
```

**Projection Process:**

1. **Linear Projection**: Maps Euclidean embedding to tangent space
   - `Linear(embedding_dim → embedding_dim + 1)`
   - Adds the time coordinate dimension

2. **Exponential Map**: Maps from tangent space to hyperboloid

   ```
   x₀ = cosh(||v|| / √c)
   x_rest = (sinh(||v|| / √c) * v) / ||v||
   ```

   - Ensures points satisfy the Lorentz constraint
   - Numerically stable with clamping

**Output**: Hyperbolic embedding `E_hyp` of shape `(batch_size, embedding_dim + 1)` on the Lorentz hyperboloid.

---

### 4. Hyperbolic Contrastive Learning

Contrastive learning is performed directly in hyperbolic space using **Decoupled Contrastive Learning (DCL)** with **Lorentzian geodesic distances**.

#### Decoupled Contrastive Learning (DCL) Loss

```python
HyperbolicInfoNCELoss(
    embedding_dim=embedding_dim,
    temperature=0.07,
    curvature=1.0
)
```

**Note**: Despite the class name, this loss function implements DCL rather than standard InfoNCE.

**Distance Computation:**

Lorentzian distance between two points `u, v` on the hyperboloid:

```
d(u, v) = √c * arccosh(-⟨u, v⟩_L)
```

where the Lorentz inner product is:

```
⟨u, v⟩_L = u₁v₁ + ... + uₙvₙ - u₀v₀
```

**Loss Function:**

Decoupled Contrastive Learning (DCL) loss with hyperbolic distances:

```
pos_sim = -d(anchor, positive) / τ
neg_sims = -d(anchor, negativeᵢ) / τ  for all i
L = (-pos_sim + logsumexp(neg_sims)).mean()
```

where `τ` is the temperature parameter (default: `0.07`).

**Key Advantages of DCL:**

- **Decoupled gradients**: Positive and negative terms are computed separately, improving gradient flow
- **Numerical stability**: Uses `logsumexp` for stable computation of the negative term
- **Flexibility**: Can yield negative loss values (unlike InfoNCE), which is expected behavior

#### Hard Negative Mining (Issue #15)

The system implements **Lorentzian Hard Negative Mining** to select the most challenging negatives for training. Instead of using randomly sampled negatives, the model selects negatives that are geometrically close to anchors in hyperbolic space.

**Implementation:**

```python
LorentzianHardNegativeMiner(
    curvature=1.0,
    safety_epsilon=1e-5
)
```

**Process:**

1. For each anchor, compute Lorentzian distances to all candidate negatives
2. Select top-k negatives with smallest distances (hardest negatives)
3. Use these hard negatives in the contrastive loss

**Benefits:**

- **Better Learning Signal**: Hard negatives provide more informative gradients
- **Faster Convergence**: Model learns to distinguish between similar codes more effectively
- **Improved Representations**: Embeddings develop finer-grained distinctions

#### Router-Guided Negative Mining (Issue #16)

To prevent "Expert Collapse" in the MoE layer, the system implements **Router-Guided Negative Mining**. This selects negatives that confuse the gating network by assigning similar expert probabilities to anchors and negatives.

**Implementation:**

```python
RouterGuidedNegativeMiner(
    metric='kl_divergence',  # or 'cosine_similarity'
    temperature=1.0
)
```

**Process:**

1. Compute gate probabilities for anchors and negatives
2. Measure confusion using KL-divergence or cosine similarity
3. Select negatives with highest confusion (most similar gate distributions)
4. Mix router-hard negatives with embedding-hard negatives (default: 50/50)

**Benefits:**

- **Prevents Expert Collapse**: Ensures all experts are utilized effectively
- **Diverse Negative Sampling**: Captures negatives that confuse the routing mechanism
- **Better MoE Training**: Improves expert specialization and load balancing

#### Global Batch Sampling (Issue #19)

For distributed training, the system implements **Global Batch Sampling** to enable hard negative mining across all GPUs. This is crucial for finding meaningful "Cousin" negatives that may not appear in small local batches.

**Implementation:**

```python
# Automatically enabled when:
# - Distributed training is active (torch.distributed.is_initialized())
# - Multiple GPUs available (world_size > 1)
# - Hard negative mining or router-guided sampling is enabled
```

**Process:**

1. **Gather Phase**: Collect negative embeddings from all GPUs using `torch.distributed.all_gather`
2. **Distance Computation**: Compute distances from local anchors to all global negatives
3. **Selection**: Select top-k hardest negatives from the global pool
4. **Gradient Flow**: Gradients flow back through the all_gather operation to all GPUs

**Memory Management:**

- Global negatives: ~9MB per GPU (for batch_size=32, world_size=4, k_negatives=24)
- Similarity matrix: ~393KB per batch
- Automatically logged for monitoring: `train/global_batch/global_negatives_memory_mb`

**Benefits:**

- **Larger Negative Pool**: Access to negatives from all GPUs, not just local batch
- **Better Hard Negatives**: More likely to find meaningful "Cousin" relationships
- **Improved Training**: Higher quality negative samples lead to better representations

#### False-Negative Mitigation

A curriculum-based procedure removes semantically similar negatives:

1. **Clustering**: Periodically cluster embeddings using **Hyperbolic K-Means** (Issue #17)
   - Clusters directly in Lorentz space using Lorentzian distances
   - Default: every 5 epochs after epoch 10
   - Default: 500 clusters (adaptive based on dataset size)
   - Safeguard ensures `n_clusters >= 1` to prevent errors
2. **Masking**: Identify negatives sharing cluster label with anchor
3. **Exclusion**: Mask these false negatives with `-inf` in the negative similarities before `logsumexp`

**Hyperbolic K-Means (Issue #17):**

Unlike standard Euclidean K-Means, the system uses **Hyperbolic K-Means** that operates directly in Lorentz space:

- Uses Lorentzian distances for cluster assignment
- Updates cluster centroids in hyperbolic space
- Preserves the geometric structure of embeddings

This prevents the model from incorrectly separating close hierarchical neighbors. The masking is retained in the DCL formulation, ensuring false negatives do not contribute to the loss.

#### Norm-Adaptive Margin

The system implements **Norm-Adaptive Margins** that adapt to the hyperbolic radius of anchors:

```
m(a) = m₀ * sech(||a||_L)
```

where:
- `m₀` is the base margin (default: 0.5)
- `||a||_L` is the Lorentz norm (hyperbolic radius) of anchor `a`
- `sech` is the hyperbolic secant function

**Benefits:**

- **Adaptive Difficulty**: Anchors near the leaf boundary (large norm) have smaller margins
- **Geometric Awareness**: Margins adapt to the hyperbolic geometry
- **Better Training**: More appropriate margins for different regions of hyperbolic space

---

### 5. Hierarchy Preservation Loss

Additional loss component that directly optimizes hierarchy preservation by matching embedding distances to tree distances.

#### Implementation

```python
HierarchyPreservationLoss(
    tree_distances=ground_truth_distances,
    code_to_idx=code_to_idx,
    weight=0.325,  # Default weight
    min_distance=0.1
)
```

**Loss Computation:**

For each pair of codes in the batch:

```
L_hierarchy = weight * MSE(embedding_distance, tree_distance)
```

- **Embedding Distance**: Lorentzian geodesic distance between hyperbolic embeddings
- **Tree Distance**: Ground truth distance in the NAICS taxonomy tree
- **Weight**: Controls the importance of hierarchy preservation (default: `0.325`)

This loss encourages the embedding space to directly reflect the hierarchical structure of NAICS.

---

### 6. Rank Order Preservation Loss (LambdaRank)

Global ranking optimization using LambdaRank to preserve rank order relationships.

#### Implementation

```python
LambdaRankLoss(
    tree_distances=ground_truth_distances,
    code_to_idx=code_to_idx,
    weight=0.275,  # Default weight
    sigma=1.0,
    ndcg_k=10
)
```

**Key Features:**

- **Position-Aware**: Optimizes NDCG@k (Normalized Discounted Cumulative Gain)
- **Gradient Weighting**: Uses LambdaRank gradients that weight pairs by their impact on ranking
- **Global Optimization**: Considers all pairs, not just anchor-positive-negative triplets

This loss ensures that the relative ordering of codes in the embedding space matches their hierarchical relationships.

---

### 7. Radius Regularization

Prevents hyperbolic embeddings from collapsing to the origin or expanding too far.

#### Implementation

Regularization term that encourages embeddings to maintain reasonable hyperbolic radii:

```
L_radius = radius_reg_weight * ||r - target_radius||²
```

where `r` is the hyperbolic radius (time coordinate `x₀`) and `target_radius` is a learned or fixed value.

**Default Weight**: `0.01`

---

### 8. Evaluation Metrics

The system computes comprehensive evaluation metrics during training:

#### Hierarchy Metrics

- **Cophenetic Correlation**: Measures how well embedding distances preserve tree structure
- **Spearman Correlation**: Rank-order correlation between embedding and tree distances
- **NDCG@k**: Position-aware ranking quality metric (k ∈ {5, 10, 20})
- **Distortion**: Mean, median, and std deviation of distance distortions

#### Embedding Statistics

- **Lorentz Norm**: Mean and violations of the Lorentz constraint
- **Hyperbolic Radius**: Mean and std of hyperbolic radii
- **Pairwise Distances**: Mean and std of embedding distances

#### Collapse Detection

- **Norm CV**: Coefficient of variation of embedding norms
- **Distance CV**: Coefficient of variation of pairwise distances
- **Variance Collapse**: Detects if embeddings collapse to a single point

---

## Data Flow

### Forward Pass

```
NAICS Code (4 text channels)
    ↓
[Multi-Channel Encoder]
    ├─→ Title Encoder (LoRA) → E_title
    ├─→ Description Encoder (LoRA) → E_desc
    ├─→ Examples Encoder (LoRA) → E_examples
    └─→ Excluded Encoder (LoRA) → E_excluded
    ↓
[Concatenate] → (embedding_dim * 4)
    ↓
[MoE Fusion] → E_fused (embedding_dim)
    ↓
[Hyperbolic Projection] → E_hyp (embedding_dim + 1)
    ↓
[Lorentz Hyperboloid]
```

### Training Step

```
Batch: (anchors, positives, negatives)
    ↓
[Forward Pass] → Hyperbolic embeddings
    ↓
[Global Batch Sampling] (if distributed + hard negative mining enabled)
    ├─→ Gather negatives from all GPUs
    └─→ Create global negative pool
    ↓
[Hard Negative Mining] (if enabled)
    ├─→ Compute Lorentzian distances to all negatives
    ├─→ Select top-k hardest negatives
    └─→ (Optionally) Router-guided negative selection
    ↓
[Compute Distances]
    ├─→ Anchor-Positive distances
    └─→ Anchor-Negative distances (hard negatives)
    ↓
[Apply False-Negative Mask] (if available)
    ↓
[Decoupled Contrastive Learning (DCL) Loss]
    ↓
[Additional Losses]
    ├─→ Hierarchy Preservation Loss
    ├─→ LambdaRank Loss
    ├─→ Radius Regularization
    └─→ MoE Load Balancing Loss
    ↓
[Total Loss] → Backpropagation
    ↓
[Gradient Flow] → Updates embeddings on all GPUs (if distributed)
```

---

## Training Pipeline

### Single-Stage Training

1. **Data Loading**: Stream triplets (anchor, positive, negatives) based on curriculum config
2. **Forward Pass**: Encode and project to hyperbolic space
3. **Loss Computation**: Combine contrastive, hierarchy, and ranking losses
4. **Backpropagation**: Update encoder, MoE, and projection parameters
5. **Evaluation**: Compute metrics every N epochs
6. **Early Stopping**: Monitor validation loss with patience

### Sequential Curriculum Training

Multiple stages with progressive difficulty:

- **Stage 1**: Coarse-grained relationships (e.g., level 2-3 codes)
- **Stage 2**: Finer-grained relationships (e.g., level 3-4 codes)
- **Stage 3+**: Specialized relationships or edge cases

Each stage:

1. Loads checkpoint from previous stage
2. Trains with stage-specific curriculum
3. Saves best checkpoint for next stage

### Structure-Aware Dynamic Curriculum (SADC) (Issue #12)

The system implements a **Structure-Aware Dynamic Curriculum** that progressively enables advanced training features based on training progress:

**Curriculum Phases:**

1. **Phase 0 (Early Training)**: Basic contrastive learning
   - Standard negative sampling
   - No hard negative mining
   - No false negative masking

2. **Phase 1 (Mid Training)**: Enhanced negative sampling
   - Enable hard negative mining
   - Enable false negative clustering
   - Track negative sample type distribution

3. **Phase 2 (Advanced Training)**: Advanced techniques
   - Enable router-guided sampling
   - Mix embedding-hard and router-hard negatives
   - Full curriculum features active

**Features:**

- **Automatic Phase Transitions**: Phases activate based on epoch thresholds
- **Negative Sample Tracking**: Logs distribution of negative types (child/sibling/cousin/distant)
- **Smooth Progression**: Gradually introduces complexity as model improves

### Multi-Level Supervision (Issue #18)

The system supports **Multi-Level Supervision** where each anchor can have multiple positive examples at different hierarchy levels:

**Implementation:**

- Batch is expanded so each positive level is a separate training item
- Loss naturally sums over all positive levels
- Provides gradient accumulation across hierarchy levels

**Benefits:**

- **Rich Supervision**: Model learns from multiple positive relationships simultaneously
- **Hierarchy Awareness**: Explicitly models relationships at different levels
- **Better Representations**: Captures hierarchical structure more effectively

### False-Negative Curriculum

After initial training (default: epoch 10), periodically:

1. Generate embeddings for all codes
2. Cluster embeddings using **Hyperbolic K-Means** (Issue #17) in Lorentz space
3. Update pseudo-labels based on cluster assignments
4. Mask false negatives in subsequent training

**Hyperbolic K-Means Clustering:**

- Operates directly in Lorentz space using Lorentzian distances
- More appropriate for hyperbolic embeddings than Euclidean K-Means
- Preserves geometric structure during clustering

---

## Mathematical Foundations

### Hyperbolic Geometry

The system uses the **Lorentz model** of hyperbolic space, which has several advantages:

1. **Differentiable**: Smooth operations suitable for gradient-based optimization
2. **Numerically Stable**: Well-conditioned distance computations
3. **Hierarchical Structure**: Natural representation for tree-like data

#### Lorentz Inner Product

For two points `u = (u₀, u₁, ..., uₙ)` and `v = (v₀, v₁, ..., vₙ)`:

```
⟨u, v⟩_L = Σᵢ₌₁ⁿ uᵢvᵢ - u₀v₀
```

#### Lorentz Distance

Geodesic distance on the hyperboloid:

```
d(u, v) = √c * arccosh(-⟨u, v⟩_L)
```

#### Exponential Map

Maps from tangent space to hyperboloid:

```
exp₀(v) = (cosh(||v||/√c), sinh(||v||/√c) * v/||v||)
```

### Contrastive Learning

The system uses **Decoupled Contrastive Learning (DCL)** loss, which decouples the positive and negative terms for improved gradient flow:

```
pos_sim = -d(anchor, positive) / τ
neg_sims = [-d(anchor, negativeᵢ) / τ for all i]
L = (-pos_sim + logsumexp(neg_sims)).mean()
```

In hyperbolic space, similarity is defined as negative distance:

```
sim(u, v) = -d(u, v)
```

**Key Differences from InfoNCE:**

- DCL computes `logsumexp(neg_sims)` directly rather than using the softmax normalization of InfoNCE
- The positive term is simply `-pos_sim` rather than being part of a log-softmax
- This decoupling provides better gradient flow and numerical stability
- DCL loss can be negative (unlike InfoNCE), which is expected behavior

---

## Design Decisions

### Why Hyperbolic Space?

1. **Hierarchical Structure**: Hyperbolic space naturally represents tree-like hierarchies
2. **Distance Properties**: Geodesic distances capture hierarchical relationships
3. **Capacity**: More capacity than Euclidean space for hierarchical data

### Why Lorentz Model?

1. **Differentiability**: Smooth operations for gradient-based learning
2. **Numerical Stability**: Well-conditioned distance computations
3. **Standard Form**: Widely used in machine learning literature

### Why Multi-Channel Encoding?

1. **Rich Semantics**: Different text fields capture different aspects
2. **Specialization**: Each channel can learn field-specific patterns
3. **Robustness**: Reduces reliance on any single text field

### Why MoE Fusion?

1. **Adaptive Combination**: Learns how to combine channels based on context
2. **Efficiency**: Top-k routing reduces computation
3. **Expressiveness**: Multiple experts capture diverse fusion patterns

### Why LoRA?

1. **Parameter Efficiency**: Reduces trainable parameters by ~90%
2. **Flexibility**: Can adapt any transformer architecture
3. **Memory Efficiency**: Enables larger batch sizes

### Why False-Negative Mitigation?

1. **Hierarchical Ambiguity**: Close codes in hierarchy may be sampled as negatives
2. **Curriculum Learning**: Gradually refines negative sampling as embeddings improve
3. **Better Representations**: Prevents model from incorrectly separating similar codes

---

## Component Dependencies

```
NAICSContrastiveModel
    ├─→ MultiChannelEncoder
    │   ├─→ 4x LoRA-adapted Transformers
    │   ├─→ MixtureOfExperts
    │   └─→ HyperbolicProjection
    ├─→ HyperbolicInfoNCELoss (implements DCL)
    ├─→ HierarchyPreservationLoss (optional)
    ├─→ LambdaRankLoss (optional)
    └─→ Evaluation Components
        ├─→ EmbeddingEvaluator
        ├─→ EmbeddingStatistics
        └─→ HierarchyMetrics
```

---

## Configuration

Key hyperparameters (see `conf/config.yaml`):

- **Model**: `base_model_name`, `lora_r`, `lora_alpha`, `num_experts`, `top_k`
- **Hyperbolic**: `curvature`, `temperature`
- **Loss Weights**: `hierarchy_weight`, `rank_order_weight`, `radius_reg_weight`
- **Training**: `learning_rate`, `weight_decay`, `warmup_steps`, `use_warmup_cosine`
- **False Negatives**: `fn_curriculum_start_epoch`, `fn_cluster_every_n_epochs`, `fn_num_clusters`
- **Distributed Training**: `training.trainer.devices` (number of GPUs)

## Distributed Training

The system supports multi-GPU distributed training with automatic global batch sampling:

### Setup

Configure the number of devices in `conf/config.yaml`:

```yaml
training:
  trainer:
    devices: 4  # Number of GPUs
    accelerator: 'gpu'
```

### Global Batch Sampling

When distributed training is enabled with hard negative mining:

- **Automatic Activation**: Global batch sampling activates automatically
- **Memory Efficient**: Monitors and logs VRAM usage
- **Gradient Flow**: Gradients flow back through all_gather to all GPUs
- **Better Negatives**: Access to negatives from all GPUs, not just local batch

### Monitoring

TensorBoard logs include:

- `train/global_batch/global_negatives_memory_mb`: Memory usage for global negatives
- `train/global_batch/similarity_matrix_memory_mb`: Memory usage for similarity matrix
- `train/global_batch/global_batch_size`: Effective global batch size
- `train/global_batch/global_k_negatives`: Number of negatives per anchor globally

---
