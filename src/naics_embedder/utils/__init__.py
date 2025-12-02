'''
Utility modules for NAICS Embedder.

This package provides shared utilities for configuration, logging, device
management, warning suppression, hyperbolic geometry, and torch.compile support.

Modules:
    backend: Device detection and GPU memory utilities.
    compile: torch.compile configuration and compiled Lorentz operations.
    config: Pydantic configuration models for training and data.
    console: Rich console logging and table formatting.
    hyperbolic: Lorentz model operations and manifold utilities.
    utilities: General helper functions for data and file operations.
    warnings: Centralized warning suppression configuration.
'''

from naics_embedder.utils.backend import get_device
from naics_embedder.utils.compile import (
    CompileConfig,
    CompiledLorentzOps,
    benchmark_compile_speedup,
    compiled_exp_map_zero,
    compiled_log_map_zero,
    compiled_lorentz_distance,
    compiled_minkowski_dot,
    compiled_project_to_hyperboloid,
    get_compile_config,
    maybe_compile,
    set_compile_config,
)
from naics_embedder.utils.config import Config, load_config
from naics_embedder.utils.console import configure_logging
from naics_embedder.utils.hyperbolic import (
    CurvatureConfig,
    CurvatureManager,
    LorentzManifold,
    ManifoldAdapter,
    validate_hyperbolic_embeddings,
)
from naics_embedder.utils.warnings import configure_warnings

__all__ = [
    'get_device',
    'Config',
    'load_config',
    'configure_logging',
    'configure_warnings',
    # Hyperbolic utilities
    'CurvatureConfig',
    'CurvatureManager',
    'LorentzManifold',
    'ManifoldAdapter',
    'validate_hyperbolic_embeddings',
    # Compile utilities
    'CompileConfig',
    'CompiledLorentzOps',
    'get_compile_config',
    'set_compile_config',
    'maybe_compile',
    'compiled_exp_map_zero',
    'compiled_log_map_zero',
    'compiled_lorentz_distance',
    'compiled_minkowski_dot',
    'compiled_project_to_hyperboloid',
    'benchmark_compile_speedup',
]
