'''Metrics utilities for analyzing embedding structure and evaluation.

This module consolidates all metrics-related functionality:
- Core metrics classes (EmbeddingEvaluator, HierarchyMetrics, etc.)
- Graph-specific validation metrics and the graph embedding container
- Hierarchy structure metrics
- Req 6's structural diagnostics report
- Evaluation runner
'''

# Core metrics classes
from .core import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
    RetrievalMetrics,
)

# Req 6's diagnostics
from .diagnostics import DiagnosticsReport, diagnostics_report

# Graph-specific metrics
from .graph import (
    GraphEmbeddingDataset,
    compute_validation_metrics,
)

# Hierarchy structure metrics
from .hierarchy_structure import (
    compute_hierarchy_retrieval_metrics,
    compute_radius_structure_metrics,
)

# Evaluation runner
from .runner import NAICSEvaluationRunner
from .structural_spearman import (
    STRUCTURAL_SPEARMAN_DEFINITION,
    STRUCTURAL_SPEARMAN_KEY,
    StructuralMetricInputError,
)

__all__ = [
    # Core
    'EmbeddingEvaluator',
    'EmbeddingStatistics',
    'HierarchyMetrics',
    'RetrievalMetrics',
    'STRUCTURAL_SPEARMAN_DEFINITION',
    'STRUCTURAL_SPEARMAN_KEY',
    'StructuralMetricInputError',
    # Diagnostics
    'DiagnosticsReport',
    'diagnostics_report',
    # Graph
    'GraphEmbeddingDataset',
    'compute_validation_metrics',
    # Hierarchy structure
    'compute_hierarchy_retrieval_metrics',
    'compute_radius_structure_metrics',
    # Runner
    'NAICSEvaluationRunner',
]
