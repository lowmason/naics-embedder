# NAICS Graph Model Analysis Report

## Executive Summary

The `graph_model` module implements Stage 4 of the NAICS Hyperbolic Embedding System:
**Hyperbolic Graph Convolutional Network (HGCN) Refinement**. This stage takes
Lorentz-space embeddings produced by the text model and refines them by propagating
information along the NAICS parent-child taxonomy graph.

The implementation demonstrates several sophisticated design patterns, including learnable
uncertainty-weighted loss balancing and tangent-space message passing. However, there are
notable opportunities for improvement, particularly around curriculum learning and
evaluation metrics.

---

## 1. Architecture Overview

### 1.1 Module Structure

```text
graph_model/
├── __init__.py                 # Module exports
├── hgcn.py                     # Core HGCN implementation (~500+ lines)
├── evaluation.py               # Validation metrics
└── dataloader/
    ├── __init__.py
    ├── hgcn_datamodule.py      # PyTorch Lightning DataModule
    └── hgcn_streaming_dataset.py  # Triplet streaming
```

### 1.2 Core Components

The graph model consists of three primary components:

**HyperbolicConvolution** — A PyTorch Geometric `MessagePassing` layer that performs graph
convolution in hyperbolic space via tangent-space operations.

**HGCN** — A stack of `HyperbolicConvolution` layers with learnable loss weights.

**HGCNLightningModule** — The PyTorch Lightning training wrapper that orchestrates training,
validation, and logging.

---

## 2. Detailed Code Analysis

### 2.1 HyperbolicConvolution Layer

```python
class HyperbolicConvolution(MessagePassing):
    def __init__(self, dim: int, dropout: float = 0.1, learnable_curvature: bool = True):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)
        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('curvature', torch.tensor(1.0))
```

**Forward Pass Flow:**

1. **Log map** — Project from hyperboloid to tangent space at origin
2. **Linear transform** — Apply learned linear transformation
3. **Message passing** — Aggregate neighbor features (mean aggregation)
4. **Residual + LayerNorm** — Add residual connection in tangent space
5. **Exp map** — Project back to hyperboloid

**Optimal Aspects:**

- **Curvature clamping** to `[0.1, 10.0]` prevents numerical instability
- **Residual connections** in tangent space preserve input information
- **LayerNorm** before exp_map stabilizes training
- **Mean aggregation** is appropriate for tree structures

**Sub-optimal Aspects:**

- **Single linear transform per layer** — More expressive architectures (e.g.,
  attention-weighted aggregation) could better capture hierarchical relationships
- **No edge features** — Parent-child vs sibling relationships are treated identically

### 2.2 HGCN Model with Uncertainty Weighting

```python
class HGCN(nn.Module):
    def __init__(self, ...):
        if learnable_loss_weights:
            self.log_var_triplet = nn.Parameter(torch.zeros(1))
            self.log_var_level = nn.Parameter(torch.zeros(1))

    def get_loss_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        precision_triplet = torch.exp(-self.log_var_triplet)
        precision_level = torch.exp(-self.log_var_level)
        reg = 0.5 * (self.log_var_triplet + self.log_var_level)
        return precision_triplet, precision_level, reg
```

#### Optimal: Uncertainty Weighting

This implements *homoscedastic uncertainty* from Kendall et al. (2018). Instead of manually
tuning loss weights, the model learns task-specific uncertainties. The precision (inverse
variance) automatically balances the triplet and level losses. The regularization term
`0.5 * sum(log_var)` prevents trivial solutions where all weights go to zero.

### 2.3 Loss Functions

#### Hyperbolic Triplet Loss

```python
def triplet_loss_hyp(emb, anchors, positives, negatives, margin, temperature, c):
    d_pos = LorentzOps.distance(emb[anchors], emb[positives], c=c)
    d_neg = LorentzOps.distance(
        emb[anchors].unsqueeze(1),
        emb[negatives],
        c=c
    )
    loss = F.relu(d_pos.unsqueeze(-1) - d_neg + margin)
    scaled = loss / temperature
    return scaled.mean()
```

**Optimal Aspects:**

- Uses true Lorentzian geodesic distance
- Temperature scaling provides gradient control
- Standard margin-based formulation

**Sub-optimal Aspects:**

- **Fixed margin** — Unlike the text model's norm-adaptive margins, HGCN uses a static margin
- **No hard negative mining** — All negatives are weighted equally

#### Per-Level Radial Regularization

```python
def level_radius_loss(emb, indices, levels):
    unique_levels = levels[indices].unique()
    total_loss = 0.0
    for lv in unique_levels:
        mask = levels[indices] == lv
        radii = emb[indices[mask], 0]  # x₀ = hyperbolic radius
        target = radii.mean().detach()
        total_loss += ((radii - target) ** 2).mean()
    return total_loss / len(unique_levels)
```

#### Optimal: Level-Aware Regularization

This enforces that codes at the same hierarchical level (2-digit, 4-digit, 6-digit) have
similar hyperbolic radii. This is geometrically meaningful — in hyperbolic space, radius
corresponds to hierarchy depth.

### 2.4 Data Loading

The `HGCNDataModule` materializes triplets using the same streaming infrastructure as the text model:

```python
class HGCNDataModule(pl.LightningDataModule):
    def prepare_data(self) -> None:
        self._materialized_triplets = load_streaming_triplets(self._streaming_cfg)

    def setup(self, stage: Optional[str] = None) -> None:
        # Split into train/val (5% validation by default)
        split_idx = int(len(data) * (1 - self.val_split))
        self._train_dataset = TripletDataset(data[:split_idx])
        self._val_dataset = TripletDataset(data[split_idx:])
```

**Optimal Aspects:**

- Reuses the sophisticated Phase 1 sampling with exclusion-weighted negatives
- Efficient Polars-based loading
- Proper train/val splitting

**Sub-optimal Aspects:**

- **No curriculum filtering** — Unlike text training, HGCN doesn't use the curriculum
  parameters (anchor_level, relation_margin, etc.)
- **Static triplet pool** — Triplets are materialized once; no epoch-to-epoch resampling

---

## 3. Evaluation Metrics Analysis

### 3.1 Current Validation Metrics

The HGCN module uses `compute_validation_metrics()`:

```python
metrics = compute_validation_metrics(
    emb_upd, anchors, positives, negatives,
    c=1.0, top_k=min(5, negatives.size(1)), as_tensors=True
)
```

Based on the codebase, this computes:

- **relation_accuracy** — Whether positive is closer than negatives
- Basic distance-based metrics

### 3.2 Missing Metrics for HGCN

The text model's `evaluation.py` computes comprehensive metrics that are **not used** by HGCN:

| Metric | Text Model | HGCN | Impact |
|--------|-----------|------|--------|
| Cophenetic Correlation | ✅ | ❌ | Critical for hierarchy |
| Spearman Correlation | ✅ | ❌ | Rank preservation |
| NDCG@k | ✅ | ❌ | Retrieval quality |
| Lorentz Norm Violations | ✅ | ❌ | Manifold validity |
| Collapse Detection | ✅ | ❌ | Training health |
| Mean Distortion | ✅ | ❌ | Distance fidelity |

**Recommendation:** Port the comprehensive evaluation suite from `text_model/evaluation.py`
to HGCN's validation step.

---

## 4. Curriculum Learning Analysis

### 4.1 Current State: No Dynamic Curriculum

The text model implements a sophisticated 3-phase **Structure-Aware Dynamic Curriculum (SADC)**:

1. **Phase 1 (0-30%):** Tree-distance weighted sampling with exclusion mining
2. **Phase 2 (30-70%):** Embedding-based hard negative mining
3. **Phase 3 (70-100%):** False negative mitigation via clustering

**HGCN uses none of this.** It applies Phase 1 sampling statically throughout training.

### 4.2 Would Dynamic Curriculum Improve HGCN?

**Yes, likely.** Here's why:

**Case for Curriculum in HGCN:**

1. **Embedding Quality Varies** — Early HGCN epochs have poorly-refined embeddings. Hard
   negatives selected on random embeddings are spurious.

2. **Graph Structure is Gradual** — Message passing needs multiple iterations to propagate
   information. Early stopping with hard negatives could lock in bad structure.

3. **Curvature is Learning** — The learnable curvature parameter changes throughout
   training. Distance-based sampling should adapt.

**Proposed HGCN Curriculum:**

| Phase | Epochs | Sampling Strategy | Rationale |
|-------|--------|-------------------|-----------|
| Warm-up | 1-2 | Easy negatives (far in tree) | Initialize structure |
| Refinement | 3-6 | Tree-distance weighted | Standard Phase 1 |
| Hardening | 7-8 | Embedding-based hard negatives | Polish fine distinctions |

### 4.3 Implementation Sketch

```python
# In HGCNLightningModule
def on_train_epoch_start(self):
    progress = self.current_epoch / self.cfg.num_epochs
    
    if progress < 0.25:
        # Warm-up: easy negatives only
        self.datamodule.update_sampling(min_tree_distance=6)
    elif progress < 0.75:
        # Standard Phase 1
        self.datamodule.update_sampling(use_phase1_weights=True)
    else:
        # Hard negative mining from current embeddings
        self.datamodule.update_sampling(
            use_embedding_mining=True,
            current_embeddings=self.get_final_embeddings()
        )
```

---

## 5. Additional Recommendations

### 5.1 Critical: More Evaluation Metrics

**Current gap:** HGCN logs minimal metrics during validation. Add:

```python
# In validation_step
from naics_embedder.text_model.evaluation import (
    EmbeddingEvaluator,
    HierarchyMetrics,
    EmbeddingStatistics
)

evaluator = EmbeddingEvaluator(emb_upd, tree_distances)
hierarchy = evaluator.compute_hierarchy_metrics()
stats = evaluator.compute_embedding_statistics()

self.log('val/cophenetic', hierarchy.cophenetic_correlation)
self.log('val/spearman', hierarchy.spearman_correlation)
self.log('val/ndcg_5', hierarchy.ndcg_5)
self.log('val/lorentz_violations', stats.lorentz_violations)
self.log('val/collapse_detected', float(stats.variance_collapse))
```

### 5.2 Important: Edge-Type Aware Aggregation

The current implementation treats all edges equally. NAICS has meaningful edge types:

- **Parent-child** (vertical hierarchy)
- **Sibling** (horizontal relationships via shared parent)

```python
# Proposed enhancement
class EdgeTypeAwareConvolution(MessagePassing):
    def __init__(self, dim: int, edge_types: int = 2):
        self.edge_transforms = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(edge_types)
        ])
    
    def message(self, x_j, edge_type):
        return self.edge_transforms[edge_type](x_j)
```

### 5.3 Nice-to-Have: Attention-Weighted Aggregation

Instead of mean aggregation:

```python
# Current: mean aggregation
x_agg = self.propagate(edge_index, x=x_lin)

# Proposed: attention aggregation
class HyperbolicGAT(MessagePassing):
    def __init__(self, dim):
        self.att = nn.Linear(2 * dim, 1)
    
    def message(self, x_i, x_j):
        alpha = torch.sigmoid(self.att(torch.cat([x_i, x_j], dim=-1)))
        return alpha * x_j
```

### 5.4 Testing Gap: Zero Coverage

Per the test README, `graph_model/` has **zero test coverage**. Priority tests:

1. `test_manifold_preservation` — Verify outputs stay on Lorentz hyperboloid
2. `test_gradient_flow` — Ensure gradients flow through exp/log maps (Issue #10)
3. `test_curvature_clamping` — Verify extreme curvatures are handled
4. `test_loss_weight_learning` — Verify uncertainty weighting converges

---

## 6. Summary: Optimal vs Sub-Optimal

### ✅ Optimal Design Decisions

| Aspect | Implementation | Why It's Good |
|--------|----------------|---------------|
| Uncertainty Weighting | Learned `log_var` parameters | Auto-balances losses without manual tuning |
| Tangent Space Ops | Log → Linear → Exp | Numerically stable hyperbolic operations |
| Curvature Clamping | `[0.1, 10.0]` | Prevents numerical overflow/underflow |
| Level Regularization | Per-level radius targets | Enforces meaningful geometric structure |
| Residual Connections | In tangent space before exp_map | Preserves gradient flow and input signal |
| PyG Integration | Uses MessagePassing | Efficient, GPU-friendly graph ops |

### ⚠️ Sub-Optimal / Missing Features

| Aspect | Current State | Recommendation | Priority |
|--------|--------------|----------------|----------|
| Curriculum Learning | None | Implement 3-phase curriculum | High |
| Evaluation Metrics | Minimal | Port text_model evaluator | High |
| Hard Negative Mining | None in HGCN | Add embedding-based mining | Medium |
| Edge Types | Not differentiated | Edge-type aware convolution | Medium |
| Adaptive Margins | Fixed margin | Norm-adaptive margins | Medium |
| Test Coverage | 0% | Add comprehensive tests | High |
| Attention Aggregation | Mean only | Optional attention layer | Low |

---

## 7. Conclusion

The `graph_model` implementation is architecturally sound, leveraging state-of-the-art
techniques like uncertainty-weighted multi-task learning and Lorentz-space message passing.
However, it significantly lags behind the text model in training sophistication (no
curriculum) and evaluation comprehensiveness (minimal metrics).

**Top 3 Actionable Items:**

1. **Add comprehensive evaluation metrics** — Cophenetic correlation, NDCG@k, collapse detection
2. **Implement dynamic curriculum** — Warm-up → standard → hard negative phases
3. **Write unit tests** — Target >70% coverage, especially for manifold validity

These improvements would bring HGCN to parity with the text model's training infrastructure
and provide much better visibility into training dynamics.
