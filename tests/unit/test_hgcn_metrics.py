import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest
import torch

from naics_embedder.graph_model.hgcn import HGCNLightningModule, save_outputs
from naics_embedder.metrics import StructuralMetricInputError
from naics_embedder.utils.config import GraphConfig

def _lorentz_points(spatial):
    spatial_tensor = torch.tensor(spatial, dtype=torch.float32)
    time = torch.sqrt(1.0 + torch.sum(spatial_tensor**2, dim=1, keepdim=True))
    return torch.cat([time, spatial_tensor], dim=1)

def test_hgcn_full_eval_metrics(tmp_path):
    codes = ['11', '111', '112']
    matrix = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    ).numpy()
    columns = {}
    for idx, code in enumerate(codes):
        columns[f'idx_{idx}-code_{code}'] = matrix[:, idx]
    distance_df = pl.DataFrame(columns)
    distance_path = tmp_path / 'distance.parquet'
    distance_df.write_parquet(distance_path)

    cfg = GraphConfig(
        distance_matrix_parquet=str(distance_path),
        ndcg_k_values=[2],
        full_eval_frequency=1,
        tangent_dim=3,
        n_hgcn_layers=1,
        dropout=0.0,
        learnable_curvature=False,
        learnable_loss_weights=False,
        k_total=1,
        n_positive_samples=1,
        batch_size=1,
    )

    embeddings = _lorentz_points([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    levels = torch.tensor([2, 3, 3], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_types = torch.zeros(edge_index.size(1), dtype=torch.long)
    edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32)
    edge_meta = {'edge_type_count': 1, 'sibling_type_id': None}
    node_metadata = pl.DataFrame({'code': codes})

    module = HGCNLightningModule(
        cfg, embeddings.clone(), levels, edge_index, edge_types, edge_weights, edge_meta,
        node_metadata
    )

    assert module.tree_distances is not None
    assert module._should_run_full_eval(batch_idx=0) is True

    metrics = module._compute_full_validation_metrics(module.forward())
    assert metrics is not None
    assert 'cophenetic_correlation' in metrics
    assert 'ndcg@2' in metrics

@pytest.fixture
def spearman_hgcn(
    tmp_path, monkeypatch, structural_distance_matrices, structural_lorentz_embeddings
):
    prediction, target = structural_distance_matrices
    codes = ['n0', 'n1', 'n2', 'n3']
    distance_path = tmp_path / 'distances.parquet'
    pl.DataFrame({
        f'idx_{i}-code_{code}': target[:, i].numpy()
        for i, code in enumerate(codes)
    }).write_parquet(distance_path)
    cfg = GraphConfig(
        distance_matrix_parquet=str(distance_path),
        relations_parquet=str(tmp_path / 'absent-relations.parquet'),
        curriculum_cache_dir=str(tmp_path),
        curriculum_enabled=False,
        output_parquet=str(tmp_path / 'encodings.parquet'),
        ndcg_k_values=[2],
        full_eval_frequency=1,
        tangent_dim=3,
        n_hgcn_layers=1,
        dropout=0.0,
        learnable_curvature=False,
        learnable_loss_weights=False,
        k_total=2,
        n_positive_samples=1,
        batch_size=1,
    )
    levels = torch.tensor([2, 3, 3, 4])
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    metadata = pl.DataFrame({'index': range(4), 'code': codes, 'level': levels.tolist()})
    module = HGCNLightningModule(
        cfg,
        structural_lorentz_embeddings,
        levels,
        edges,
        torch.zeros(edges.shape[1], dtype=torch.long),
        torch.ones(edges.shape[1]),
        {
            'edge_type_count': 1,
            'sibling_type_id': None
        },
        metadata,
    )
    monkeypatch.setattr(module, 'forward', Mock(return_value=structural_lorentz_embeddings))
    monkeypatch.setattr(module, 'log', Mock())
    monkeypatch.setattr(
        module.embedding_evaluator, 'compute_pairwise_distances', lambda *_, **__: prediction
    )
    return module, metadata

@pytest.mark.unit
@pytest.mark.parametrize('undefined', [False, True])
def test_hgcn_spearman_full_metrics_logs_and_history(tmp_path, spearman_hgcn, undefined):
    module, metadata = spearman_hgcn
    if undefined:
        module.tree_distances.fill_(1.0)
        module.tree_distances.fill_diagonal_(0.0)
    full = module._compute_full_validation_metrics(module.forward())
    key = 'structural_spearman_v1'
    assert full is not None
    assert full[f'{key}_n_pairs'] == full[f'{key}_n_total'] == 6
    assert full[f'{key}_definition'] == 'structural-spearman-v1'
    assert {'spearman_correlation', 'spearman_n_pairs'}.isdisjoint(full)

    module.history.append({'epoch': 1, 'loss': 1.0})
    module.on_validation_epoch_start()
    module.validation_step(
        {
            'anchor_idx': torch.tensor([0]),
            'positive_idx': torch.tensor([1]),
            'negative_indices': torch.tensor([[2, 3]])
        },
        batch_idx=0,
    )
    module.on_validation_epoch_end()
    save_outputs(
        str(tmp_path),
        module.embeddings.detach(),
        metadata,
        module.cfg,
        module.model,
        module.export_history(),
    )
    history = json.loads((tmp_path / 'training_log.json').read_text())
    json.dumps(history, allow_nan=False)
    record = history[0]
    assert record[f'val_{key}_definition'] == 'structural-spearman-v1'
    assert record[f'val_{key}_n_pairs'] == record[f'val_{key}_n_total'] == 6
    assert isinstance(record[f'val_{key}_n_pairs'], int)
    assert {'val_spearman_correlation', 'val_spearman_n_pairs'}.isdisjoint(record)
    logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
    assert f'val/{key}_n_pairs' in logged
    assert f'val/{key}_n_total' in logged
    assert f'val/{key}_status' not in logged
    assert f'val/{key}_reason' not in logged
    assert f'val/{key}_definition' not in logged
    assert 'val/spearman_correlation' not in logged
    assert 'val/spearman_n_pairs' not in logged
    if undefined:
        assert full[key] is None
        assert record[f'val_{key}'] is None
        assert record[f'val_{key}_status'] == 'undefined'
        assert record[f'val_{key}_reason'] == 'constant_target'
        assert f'val/{key}' not in logged
    else:
        assert full[key].item() == pytest.approx(0.87831006565368, abs=1e-7)
        assert record[f'val_{key}'] == pytest.approx(0.87831006565368, abs=1e-7)
        assert record[f'val_{key}_status'] == 'defined'
        assert record[f'val_{key}_reason'] is None
        assert logged[f'val/{key}'].item() == pytest.approx(record[f'val_{key}'])
    module.on_validation_epoch_start()
    assert module._full_val_metrics == {}

@pytest.mark.unit
@pytest.mark.parametrize('invalid', ['nan_file', 'asymmetric', 'shape'])
def test_hgcn_malformed_distances_fail_validation(spearman_hgcn, invalid):
    module, _ = spearman_hgcn
    if invalid == 'nan_file':
        path = Path(module.cfg.distance_matrix_parquet)
        frame = pl.read_parquet(path)
        columns = frame.to_dict(as_series=False)
        columns['idx_0-code_n0'][1] = float('nan')
        pl.DataFrame(columns).write_parquet(path)
        module.tree_distances = module._load_tree_distance_tensor(module.node_codes)
        assert torch.isnan(module.tree_distances[1, 0])
    elif invalid == 'asymmetric':
        module.tree_distances[1, 0] = 10.0
    else:
        module.tree_distances = module.tree_distances[:3, :3]
    with pytest.raises(StructuralMetricInputError):
        module.validation_step(
            {
                'anchor_idx': torch.tensor([0]),
                'positive_idx': torch.tensor([1]),
                'negative_indices': torch.tensor([[2, 3]])
            },
            batch_idx=0,
        )
    logged_names = [call.args[0] for call in module.log.call_args_list]
    assert 'val/cophenetic_correlation' not in logged_names
    assert 'val/ndcg@2' not in logged_names
    assert module._full_val_metrics == {}

@pytest.mark.unit
def test_hgcn_does_not_skip_unexpected_spearman_failure(monkeypatch, spearman_hgcn):
    module, _ = spearman_hgcn
    monkeypatch.setattr(
        'naics_embedder.metrics.structural_spearman.spearmanr',
        lambda *_: SimpleNamespace(statistic=float('nan')),
    )
    with pytest.raises(RuntimeError, match='structural-spearman-v1.*n_pairs=6'):
        module._compute_full_validation_metrics(module.forward())
