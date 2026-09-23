import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from naics_embedder.metrics import (
    EmbeddingEvaluator,
    EmbeddingStatistics,
    HierarchyMetrics,
    StructuralMetricInputError,
)
from naics_embedder.text_model.mixins.logging import LoggingMixin
from naics_embedder.text_model.mixins.validation import ValidationMixin

pytestmark = pytest.mark.unit

class ValidationHarness(ValidationMixin, LoggingMixin):

    def __init__(self, directory: Path, target: torch.Tensor, embeddings: torch.Tensor) -> None:
        self.device = torch.device('cpu')
        self.current_epoch = 0
        self.hparams = SimpleNamespace(curvature=1.0, eval_sample_size=4, eval_every_n_epochs=1)
        self.trainer = SimpleNamespace(callback_metrics={})
        self.logger = SimpleNamespace(log_dir=str(directory))
        self.log = Mock()
        self.embedding_eval = EmbeddingEvaluator()
        self.embedding_stats = EmbeddingStatistics()
        self.hierarchy_metrics = HierarchyMetrics()
        for component in (self.embedding_eval, self.embedding_stats, self.hierarchy_metrics):
            component.device = 'cpu'
        self.naics_hierarchy = None
        self.supervision_policy = SimpleNamespace(enable_pseudo_related=False)
        self.ground_truth_distances = target
        self.code_to_idx = {f'n{i}': i for i in range(4)}
        self.validation_embeddings = {f'n{i}': embeddings[i] for i in range(4)}
        self.validation_codes = list(self.validation_embeddings)
        self.evaluation_metrics_history = []

@pytest.mark.parametrize('undefined', [False, True])
def test_text_validation_versioned_logs_and_strict_json(
    tmp_path, caplog, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings,
    undefined
):
    prediction, target = structural_distance_matrices
    if undefined:
        target.fill_(1.0)
        target.fill_diagonal_(0.0)
    harness = ValidationHarness(tmp_path, target, structural_lorentz_embeddings)
    monkeypatch.setattr(
        harness.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    harness.on_validation_epoch_end()

    history = json.loads((tmp_path / 'evaluation_metrics.json').read_text())
    json.dumps(history, allow_nan=False)
    record = history[0]
    key = 'structural_spearman_v1'
    assert record[f'{key}_definition'] == 'structural-spearman-v1'
    assert record[f'{key}_n_pairs'] == record[f'{key}_n_total'] == 6
    assert isinstance(record[f'{key}_n_pairs'], int)
    assert {'spearman', 'spearman_correlation', 'spearman_n_pairs'}.isdisjoint(record)
    logged = {call.args[0]: call.args[1] for call in harness.log.call_args_list}
    assert logged[f'val/{key}_n_pairs'] == logged[f'val/{key}_n_total'] == 6
    assert 'val/spearman_correlation' not in logged
    assert 'val/spearman_n_pairs' not in logged
    assert f'val/{key}_status' not in logged
    warnings = [
        item.getMessage() for item in caplog.records
        if 'structural-spearman-v1 undefined:' in item.getMessage()
    ]
    if undefined:
        assert record[key] is None
        assert record[f'{key}_status'] == 'undefined'
        assert record[f'{key}_reason'] == 'constant_target'
        assert f'val/{key}' not in logged
        assert len(warnings) == 1
        assert 'constant_target' in warnings[0]
    else:
        assert record[key] == pytest.approx(0.87831006565368, abs=1e-7)
        assert record[f'{key}_status'] == 'defined'
        assert record[f'{key}_reason'] is None
        assert logged[f'val/{key}'] == pytest.approx(record[key])
        assert warnings == []
    assert harness.validation_embeddings == {}
    assert harness.validation_codes == []

@pytest.mark.parametrize('invalid', ['nan', 'asymmetric', 'unaligned_codes'])
def test_text_validation_propagates_input_errors_before_hierarchy_logging(
    tmp_path, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings, invalid
):
    prediction, target = structural_distance_matrices
    harness = ValidationHarness(tmp_path, target, structural_lorentz_embeddings)
    if invalid == 'nan':
        target[1, 0] = float('nan')
    elif invalid == 'asymmetric':
        target[1, 0] = 10.0
    else:
        del harness.code_to_idx['n3']
    monkeypatch.setattr(
        harness.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    with pytest.raises(StructuralMetricInputError):
        harness.on_validation_epoch_end()
    logged_names = [call.args[0] for call in harness.log.call_args_list]
    assert not any(
        'cophenetic' in name or 'ndcg@' in name or 'distortion' in name for name in logged_names
    )
    assert harness.evaluation_metrics_history == []
    assert not (tmp_path / 'evaluation_metrics.json').exists()
    assert harness.validation_embeddings == {}
    assert harness.validation_codes == []
