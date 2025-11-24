# NAICS Hyperbolic Embedding System

This documentation describes a unified hyperbolic representation learning framework for the
**North American Industry Classification System (NAICS)**.  

The system consists of four sequential stages:

- Multi-channel transformer-based text encoding  
- Mixture-of-Experts fusion  
- Lorentz-model hyperbolic contrastive learning  
- Hyperbolic Graph Convolutional refinement (HGCN)

The final output are geometry-aware embeddings aligned with the hierarchical structure of
the NAICS taxonomy. These **Lorentz-model hyperbolic embeddings** are suitable for similarity
search, hierarchical modeling, graph-based reasoning, and downstream machine learning applications.

## Key Features

### Advanced Training Techniques

- **Hard Negative Mining**: Selects geometrically challenging negatives using Lorentzian distances
- **Router-Guided Sampling**: Prevents expert collapse by selecting negatives that confuse the MoE gating network
- **Global Batch Sampling**: Enables hard negative mining across all GPUs in distributed training
- **Structure-Aware Dynamic Curriculum**: Progressively enables advanced features based on training progress
- **Multi-Level Supervision**: Supports multiple positive examples at different hierarchy levels
- **Hyperbolic K-Means Clustering**: Clusters embeddings directly in Lorentz space for false-negative mitigation

### Loss Functions

- **Decoupled Contrastive Learning (DCL)**: Improved gradient flow and numerical stability
- **Hierarchy Preservation Loss**: Directly optimizes embedding distances to match tree structure
- **LambdaRank Loss**: Position-aware ranking optimization using NDCG
- **Radius Regularization**: Prevents hyperbolic embeddings from collapsing or expanding too far

### Distributed Training

- **Multi-GPU Support**: Automatic global batch sampling for better hard negative mining
- **Memory Efficient**: Monitors and logs VRAM usage for distributed operations
- **Gradient Flow**: Proper gradient propagation through all_gather operations

Use the navigation menu to explore system architecture, training procedures,
and API references for each module.
