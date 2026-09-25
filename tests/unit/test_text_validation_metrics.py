import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection

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
        self.trainer = SimpleNamespace(callback_metrics={}, is_global_zero=True)
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
@pytest.mark.parametrize('is_global_zero', [False, True])
def test_text_validation_propagates_input_errors_before_hierarchy_logging(
    tmp_path, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings, invalid,
    is_global_zero
):
    prediction, target = structural_distance_matrices
    harness = ValidationHarness(tmp_path, target, structural_lorentz_embeddings)
    harness.trainer.is_global_zero = is_global_zero
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

def test_no_structural_statistic_reaches_the_progress_bar(
    tmp_path, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings
):
    # Req 6: structural statistics are reported, never a headline; the collapse checks stay
    prediction, target = structural_distance_matrices
    harness = ValidationHarness(tmp_path, target, structural_lorentz_embeddings)
    monkeypatch.setattr(
        harness.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )

    harness.on_validation_epoch_end()

    logged = [call.args[0] for call in harness.log.call_args_list]
    assert {'val/cophenetic_correlation', 'val/median_distortion'} <= set(logged)
    on_bar = [call.args[0] for call in harness.log.call_args_list if call.kwargs.get('prog_bar')]
    assert on_bar == ['val/norm_cv', 'val/distance_cv']

@pytest.mark.parametrize('is_global_zero', [False, True])
def test_text_spearman_and_json_are_rank_zero_only(
    tmp_path, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings,
    is_global_zero
):
    prediction, target = structural_distance_matrices
    harness = ValidationHarness(tmp_path, target, structural_lorentz_embeddings)
    harness.trainer.is_global_zero = is_global_zero
    monkeypatch.setattr(
        harness.embedding_eval, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    harness.on_validation_epoch_end()
    calls = [
        call for call in harness.log.call_args_list if 'structural_spearman_v1' in call.args[0]
    ]
    assert len(calls) == (3 if is_global_zero else 0)
    for call in calls:
        assert call.kwargs['rank_zero_only'] is True
        assert call.kwargs['sync_dist'] is False
    assert (tmp_path / 'evaluation_metrics.json').exists() == is_global_zero
    assert len(harness.evaluation_metrics_history) == int(is_global_zero)

def _distributed_validation_worker(
    rank, init_file, directory, matrices, embeddings, undefined_rank, queue
):
    dist.init_process_group(
        backend='gloo',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        prediction, target = [value.clone() for value in matrices]
        if rank == undefined_rank:
            target.fill_(1.0)
            target.fill_diagonal_(0.0)
        harness = ValidationHarness(directory, target, embeddings)
        harness.trainer.is_global_zero = rank == 0
        collection = _ResultCollection(training=False)
        strategy = DDPStrategy()

        def log(name, value, **kwargs):
            collection.log(
                'on_validation_epoch_end',
                name,
                torch.as_tensor(value).float(),
                on_step=False,
                on_epoch=True,
                sync_dist_fn=strategy.reduce,
                **kwargs,
            )

        harness.log = log
        harness.embedding_eval.compute_pairwise_distances = Mock(return_value=prediction)
        harness.on_validation_epoch_end()
        logged = collection.metrics(on_step=False)['log']
        queue.put(
            (
                rank,
                {
                    name: value.item()
                    for name, value in logged.items() if 'structural_spearman' in name
                },
            )
        )
    finally:
        dist.destroy_process_group()

@pytest.mark.integration
@pytest.mark.parametrize('undefined_rank', [0, 1])
def test_distributed_mixed_spearman_status_preserves_rank_zero_population(
    tmp_path, structural_distance_matrices, structural_lorentz_embeddings, undefined_rank
):
    queue = mp.get_context('spawn').SimpleQueue()
    mp.spawn(
        _distributed_validation_worker,
        args=(
            tmp_path / 'gloo-init',
            tmp_path,
            structural_distance_matrices,
            structural_lorentz_embeddings,
            undefined_rank,
            queue,
        ),
        nprocs=2,
        join=True,
    )
    results = dict(queue.get() for _ in range(2))
    key = 'structural_spearman_v1'
    assert results[1] == {}
    assert results[0][f'val/{key}_n_pairs'] == results[0][f'val/{key}_n_total'] == 6
    history = json.loads((tmp_path / 'evaluation_metrics.json').read_text())
    json.dumps(history, allow_nan=False)
    record = history[0]
    if undefined_rank == 0:
        assert f'val/{key}' not in results[0]
        assert record[key] is None
        assert record[f'{key}_reason'] == 'constant_target'
    else:
        assert results[0][f'val/{key}'] == pytest.approx(0.87831006565368, abs=1e-7)
        assert record[key] == pytest.approx(results[0][f'val/{key}'])
        assert record[f'{key}_reason'] is None
