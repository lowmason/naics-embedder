import json
from dataclasses import asdict

import numpy as np
import polars as pl
import pytest
import torch
from scipy.stats import spearmanr

from naics_embedder.graph_model.hgcn import load_embeddings
from naics_embedder.metrics import EmbeddingEvaluator, StructuralMetricInputError
from naics_embedder.tools.embeddings_verification import (
    Stage4VerificationConfig,
    verify_stage4,
)

def _write_embeddings(path, codes, spatial_vectors):
    rows = []
    for idx, (code, vec) in enumerate(zip(codes, spatial_vectors)):
        vec = torch.tensor(vec, dtype=torch.float32)
        time = torch.sqrt(1 + torch.sum(vec**2))
        rows.append(
            {
                'index': idx,
                'level': len(code),
                'code': code,
                'hyp_e0': float(time),
                'hyp_e1': float(vec[0]),
                'hyp_e2': float(vec[1]),
            }
        )
    pl.DataFrame(rows).write_parquet(path)

def _write_distance_matrix(path, codes):
    data = {}
    base = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    ).numpy()
    for idx, code in enumerate(codes):
        data[f'idx_{idx}-code_{code}'] = base[:, idx]
    pl.DataFrame(data).write_parquet(path)

def _write_relations(path):
    pl.DataFrame(
        {
            'idx_i': [0, 0],
            'idx_j': [1, 2],
            'code_i': ['11', '11'],
            'code_j': ['111', '112'],
            'relation_id': [1, 1],
            'relation': ['child', 'child'],
        }
    ).write_parquet(path)

def test_verify_stage4_pass(tmp_path):
    codes = ['11', '111', '112']
    stage3_path = tmp_path / 'stage3.parquet'
    stage4_path = tmp_path / 'stage4.parquet'
    distance_path = tmp_path / 'distance.parquet'
    relations_path = tmp_path / 'relations.parquet'

    _write_embeddings(stage3_path, codes, [(0.0, 0.0), (0.2, 0.0), (0.0, 0.2)])
    _write_embeddings(stage4_path, codes, [(0.0, 0.0), (0.15, 0.0), (0.0, 0.15)])
    _write_distance_matrix(distance_path, codes)
    _write_relations(relations_path)

    cfg = Stage4VerificationConfig(
        max_cophenetic_degradation=0.5,
        max_ndcg_degradation=0.5,
        min_local_improvement=0.0,
        ndcg_k=1,
    )

    result = verify_stage4(
        stage3_path,
        stage4_path,
        distance_path,
        relations_path,
        cfg,
    )

    assert result['passed'] is True
    assert 'cophenetic_correlation' in result['pre']
    assert f'parent_retrieval@{cfg.parent_top_k}' in result['post']

def test_verify_stage4_threshold_failure(tmp_path):
    codes = ['11', '111', '112']
    stage3_path = tmp_path / 'stage3.parquet'
    stage4_path = tmp_path / 'stage4.parquet'
    distance_path = tmp_path / 'distance.parquet'
    relations_path = tmp_path / 'relations.parquet'

    vectors = [(0.0, 0.0), (0.2, 0.0), (0.0, 0.2)]
    _write_embeddings(stage3_path, codes, vectors)
    _write_embeddings(stage4_path, codes, vectors)
    _write_distance_matrix(distance_path, codes)
    _write_relations(relations_path)

    cfg = Stage4VerificationConfig(
        max_cophenetic_degradation=0.0,
        max_ndcg_degradation=0.0,
        min_local_improvement=0.1,
        ndcg_k=1,
    )

    result = verify_stage4(
        stage3_path,
        stage4_path,
        distance_path,
        relations_path,
        cfg,
    )

    assert result['passed'] is False
    assert result['checks']['local_improvement'] is False

@pytest.fixture
def structural_verification_files(tmp_path, monkeypatch, structural_distance_matrices):
    _, target = structural_distance_matrices
    codes = ['n0', 'n1', 'n2', 'n3']
    paths = (
        tmp_path / 'pre.parquet',
        tmp_path / 'post.parquet',
        tmp_path / 'distances.parquet',
        tmp_path / 'relations.parquet',
    )
    vectors = [(0.0, 0.0), (0.2, 0.0), (0.0, 0.3), (0.2, 0.4)]
    _write_embeddings(paths[0], codes, vectors)
    _write_embeddings(paths[1], codes, vectors)
    pl.DataFrame({
        f'idx_{i}-code_{code}': target[:, i].numpy()
        for i, code in enumerate(codes)
    }).write_parquet(paths[2])
    pl.DataFrame(
        {
            'code_i': ['n0', 'n0', 'n0'],
            'code_j': ['n1', 'n2', 'n3'],
            'relation': ['child', 'child', 'child'],
        }
    ).write_parquet(paths[3])
    monkeypatch.setattr('naics_embedder.metrics.core.get_device', lambda: ('cpu', '32-true', 0))
    return paths

def _report_config() -> Stage4VerificationConfig:
    return Stage4VerificationConfig(
        max_cophenetic_degradation=2.0,
        max_ndcg_degradation=1.0,
        min_local_improvement=-1.0,
        ndcg_k=1,
    )

@pytest.mark.unit
def test_verifier_real_distances_match_scipy(
    structural_verification_files, structural_distance_matrices
):
    paths = structural_verification_files
    result = verify_stage4(*paths, _report_config())
    embeddings, _, _ = load_embeddings(str(paths[0]), torch.device('cpu'))
    distances = EmbeddingEvaluator().compute_pairwise_distances(
        embeddings, metric='lorentz', curvature=1.0
    ).double().numpy()
    _, target = structural_distance_matrices
    rows, columns = np.triu_indices(4, k=1)
    observations = (distances[rows, columns] + distances[columns, rows]) / 2.0
    expected = float(spearmanr(observations, target.numpy()[rows, columns]).statistic)
    assert result['pre']['structural_spearman_v1'] == pytest.approx(expected, abs=1e-7)
    assert result['post']['structural_spearman_v1'] == pytest.approx(expected, abs=1e-7)
    assert result['delta']['structural_spearman_v1'] == 0.0

@pytest.mark.unit
def test_negative_spearman_delta_does_not_gate_verification(
    monkeypatch, structural_verification_files, structural_distance_matrices
):
    prediction, _ = structural_distance_matrices
    reversed_prediction = 7.0 - prediction
    reversed_prediction.fill_diagonal_(0.0)
    matrices = iter([prediction, reversed_prediction])
    monkeypatch.setattr(
        EmbeddingEvaluator, 'compute_pairwise_distances', lambda *_, **__: next(matrices)
    )
    result = verify_stage4(*structural_verification_files, _report_config())
    key = 'structural_spearman_v1'
    assert result['pre'][key] == pytest.approx(0.87831006565368, abs=1e-7)
    assert result['post'][key] == pytest.approx(-0.87831006565368, abs=1e-7)
    assert result['delta'][key] == pytest.approx(-1.75662013130736, abs=2e-7)
    assert set(result['checks']) == {'cophenetic', 'ndcg', 'local_improvement'}
    assert result['passed'] is True
    metadata = result['metric_metadata'][key]
    assert metadata['definition'] == 'structural-spearman-v1'
    for phase in ('pre', 'post'):
        assert metadata[phase] == {
            'status': 'defined',
            'reason': None,
            'n_pairs': 6,
            'n_total': 6,
        }
    for phase in ('pre', 'post', 'delta'):
        assert {'spearman', 'spearman_correlation'}.isdisjoint(result[phase])
    assert set(asdict(Stage4VerificationConfig())) == {
        'max_cophenetic_degradation',
        'max_ndcg_degradation',
        'min_local_improvement',
        'ndcg_k',
        'parent_top_k',
    }
    json.dumps(result, allow_nan=False)

@pytest.mark.unit
@pytest.mark.parametrize('undefined_phases', [('pre', ), ('post', ), ('pre', 'post')])
def test_verifier_undefined_values_and_delta_are_null(
    monkeypatch, structural_verification_files, structural_distance_matrices, undefined_phases
):
    prediction, _ = structural_distance_matrices
    constant = torch.ones_like(prediction)
    constant.fill_diagonal_(0.0)
    matrices = iter(
        [constant if phase in undefined_phases else prediction for phase in ('pre', 'post')]
    )
    monkeypatch.setattr(
        EmbeddingEvaluator, 'compute_pairwise_distances', lambda *_, **__: next(matrices)
    )
    result = verify_stage4(*structural_verification_files, _report_config())
    key = 'structural_spearman_v1'
    for phase in ('pre', 'post'):
        metadata = result['metric_metadata'][key][phase]
        assert metadata['n_pairs'] == metadata['n_total'] == 6
        if phase in undefined_phases:
            assert result[phase][key] is None
            assert metadata['status'] == 'undefined'
            assert metadata['reason'] == 'constant_prediction'
        else:
            assert result[phase][key] == pytest.approx(0.87831006565368, abs=1e-7)
            assert metadata['status'] == 'defined'
            assert metadata['reason'] is None
        assert {'spearman', 'spearman_correlation'}.isdisjoint(result[phase])
    assert result['delta'][key] is None
    assert set(result['checks']) == {'cophenetic', 'ndcg', 'local_improvement'}
    assert result['passed'] is True
    encoded = json.dumps(result, allow_nan=False)
    assert json.loads(encoded)['delta'][key] is None

@pytest.mark.unit
@pytest.mark.parametrize('invalid', ['nan_lower', 'asymmetric', 'prediction_shape'])
def test_verifier_rejects_malformed_distances(monkeypatch, structural_verification_files, invalid):
    paths = structural_verification_files
    if invalid == 'prediction_shape':
        monkeypatch.setattr(
            EmbeddingEvaluator,
            'compute_pairwise_distances',
            lambda *_, **__: torch.ones(3, 3),
        )
    else:
        frame = pl.read_parquet(paths[2])
        columns = frame.to_dict(as_series=False)
        columns['idx_0-code_n0'][1] = float('nan') if invalid == 'nan_lower' else 10.0
        pl.DataFrame(columns).write_parquet(paths[2])
    with pytest.raises(StructuralMetricInputError):
        verify_stage4(*paths, _report_config())
