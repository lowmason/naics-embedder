'''Metrics utilities for analyzing embedding structure and evaluation.

This module consolidates all metrics-related functionality:
- Core metrics classes (EmbeddingEvaluator, HierarchyMetrics, etc.)
- Graph-specific metrics and downstream evaluation
- QCEW benchmark
- Hierarchy structure metrics
- Evaluation runner
'''

# Core metrics classes
from .core import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
    RetrievalMetrics,
)

# Graph-specific metrics
from .graph import (
    GraphDownstreamEvaluator,
    GraphEmbeddingDataset,
    compute_validation_metrics,
    run_graph_downstream_suite,
)

# Hierarchy structure metrics
from .hierarchy_structure import (
    compute_hierarchy_retrieval_metrics,
    compute_radius_structure_metrics,
)

# QCEW benchmark
from .qcew import (
    QCEWBenchmarkConfig,
    QCEWMultilevelConfig,
    print_multilevel_comparison,
    run_qcew_employment_benchmark,
    run_qcew_multilevel_benchmark,
)

# Evaluation runner
from .runner import NAICSEvaluationRunner

__all__ = [
    # Core
    'EmbeddingEvaluator',
    'EmbeddingStatistics',
    'HierarchyMetrics',
    'RetrievalMetrics',
    # Graph
    'GraphDownstreamEvaluator',
    'GraphEmbeddingDataset',
    'compute_validation_metrics',
    'run_graph_downstream_suite',
    # Hierarchy structure
    'compute_hierarchy_retrieval_metrics',
    'compute_radius_structure_metrics',
    # QCEW
    'QCEWBenchmarkConfig',
    'QCEWMultilevelConfig',
    'print_multilevel_comparison',
    'run_qcew_employment_benchmark',
    'run_qcew_multilevel_benchmark',
    # Runner
    'NAICSEvaluationRunner',
]
