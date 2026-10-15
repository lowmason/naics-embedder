'''
Dead code the 2026-10-15 technical-debt audit deleted stays deleted.

Each entry had no caller: the torch.compile utilities (compilation was never enabled, so every
decorator returned its function unchanged), and the general evaluation runner with the two metric
classes only it used (it measured Euclidean distance whatever the geometry), and the HGCN
curriculum analyzer (it read a history file nothing writes). Git history keeps them.
'''

import importlib
import importlib.util

import pytest

# A bare module path must not be importable; 'module:attribute' must not be an attribute of it
DELETIONS = [
    # torch.compile utilities: never enabled, so maybe_compile was an identity
    'naics_embedder.utils.compile',
    'naics_embedder.utils:maybe_compile',
    'naics_embedder.utils:get_compile_config',
    'naics_embedder.utils:set_compile_config',
    'naics_embedder.utils:CompileConfig',
    'naics_embedder.utils:CompiledLorentzOps',
    'naics_embedder.utils:benchmark_compile_speedup',
    'naics_embedder.utils:compiled_exp_map_zero',
    'naics_embedder.utils:compiled_log_map_zero',
    'naics_embedder.utils:compiled_lorentz_distance',
    'naics_embedder.utils:compiled_minkowski_dot',
    'naics_embedder.utils:compiled_project_to_hyperboloid',
    'naics_embedder.text_model.hyperbolic:maybe_compile',
    'naics_embedder.text_model.hyperbolic:_COMPILE_AVAILABLE',
    'naics_embedder.text_model.moe:maybe_compile',
    'naics_embedder.text_model.moe:_COMPILE_AVAILABLE',
    'naics_embedder.utils.hyperbolic:maybe_compile',
    'naics_embedder.utils.hyperbolic:_COMPILE_AVAILABLE',
    # The general evaluation runner and the metric classes only it used
    'naics_embedder.metrics.runner',
    'naics_embedder.metrics:NAICSEvaluationRunner',
    'naics_embedder.metrics:RetrievalMetrics',
    'naics_embedder.metrics:EmbeddingStatistics',
    'naics_embedder.metrics.core:RetrievalMetrics',
    'naics_embedder.metrics.core:EmbeddingStatistics',
    # The HGCN curriculum analyzer: nothing writes the history it reads
    'naics_embedder.graph_model.curriculum.monitoring',
    'naics_embedder.graph_model.curriculum:CurriculumAnalyzer',
    'naics_embedder.graph_model.curriculum:PhaseMetrics',
    'naics_embedder.graph_model.curriculum:generate_report',
    'naics_embedder.graph_model.curriculum:plot_curriculum_training',
    'naics_embedder.graph_model:CurriculumAnalyzer',
    'naics_embedder.graph_model:generate_report',
]

@pytest.mark.unit
@pytest.mark.parametrize('path', DELETIONS)
def test_the_audited_dead_code_is_deleted(path):
    '''Nothing the audit found dead comes back.'''

    module, _, attribute = path.partition(':')
    if not attribute:
        assert importlib.util.find_spec(module) is None
        return
    assert not hasattr(importlib.import_module(module), attribute)
