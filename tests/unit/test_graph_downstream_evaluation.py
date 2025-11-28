import csv

import polars as pl
import pytest
import torch

from naics_embedder.graph_model.evaluation import (
    GraphDownstreamEvaluator,
    GraphEmbeddingDataset,
    QCEWBenchmarkConfig,
    run_qcew_employment_benchmark,
)
from naics_embedder.utils.naics_hierarchy import NaicsHierarchy

def _lorentz_points(spatial):
    tensor = torch.tensor(spatial, dtype=torch.float32)
    time = torch.sqrt(1.0 + torch.sum(tensor**2, dim=1, keepdim=True))
    return torch.cat([time, tensor], dim=1)

def _graph_fixture():
    codes = ['11111', '111110', '111111', '21111', '211110', '211111']
    levels = [5, 6, 6, 5, 6, 6]
    spatial = [
        [0.0, 0.0],
        [0.08, 0.02],
        [0.09, -0.02],
        [1.2, 0.0],
        [1.28, 0.02],
        [1.30, -0.02],
    ]
    embeddings = _lorentz_points(spatial)
    dataset = GraphEmbeddingDataset(embeddings=embeddings, codes=codes, levels=levels)
    hierarchy = NaicsHierarchy(
        [
            ('11111', '111110'),
            ('11111', '111111'),
            ('21111', '211110'),
            ('21111', '211111'),
        ]
    )
    return dataset, hierarchy

def test_taxonomy_and_similarity_metrics():
    dataset, hierarchy = _graph_fixture()
    evaluator = GraphDownstreamEvaluator(dataset)

    taxonomy = evaluator.taxonomy_reconstruction(hierarchy, k_values=(1, 2))
    assert taxonomy['top_1_parent_accuracy'] == pytest.approx(1.0)

    similarity = evaluator.industry_similarity(hierarchy, k_values=(1, 2))
    assert similarity['precision@1'] >= 0.5
    assert similarity['mean_first_sibling_rank'] <= 2.0

def test_clustering_and_classification_metrics():
    dataset, _ = _graph_fixture()
    evaluator = GraphDownstreamEvaluator(dataset)

    clustering = evaluator.clustering_quality(digits=(2, ), random_state=0)
    assert clustering['ari_2digit'] > 0.8
    assert clustering['nmi_2digit'] > 0.8

    classification = evaluator.classification_benchmark(digits=2, test_size=0.34, random_state=0)
    assert 0.0 <= classification['accuracy'] <= 1.0
    assert 0.0 <= classification['macro_f1'] <= 1.0
    assert classification['n_train'] > 0
    assert classification['n_test'] > 0

def test_qcew_benchmark_runs(tmp_path):
    dataset, _ = _graph_fixture()
    data = {
        'index': list(range(len(dataset.codes))),
        'code': dataset.codes,
        'level': dataset.levels,
    }
    for dim in range(dataset.embeddings.size(1)):
        data[f'hgcn_e{dim}'] = dataset.embeddings[:, dim].tolist()
    embeddings_df = pl.DataFrame(data)
    embed_path = tmp_path / 'embeddings.parquet'
    embeddings_df.write_parquet(embed_path)

    csv_path = tmp_path / 'qcew.csv'
    rows = [
        {
            'year': 2022,
            'own_code': 5,
            'industry_code': '111110',
            'annual_avg_emplvl': 100,
            'annual_avg_estabs': 10,
            'tot_wages': 1000,
        },
        {
            'year': 2022,
            'own_code': 5,
            'industry_code': '111111',
            'annual_avg_emplvl': 90,
            'annual_avg_estabs': 8,
            'tot_wages': 900,
        },
        {
            'year': 2022,
            'own_code': 5,
            'industry_code': '211110',
            'annual_avg_emplvl': 150,
            'annual_avg_estabs': 12,
            'tot_wages': 2000,
        },
        {
            'year': 2022,
            'own_code': 5,
            'industry_code': '211111',
            'annual_avg_emplvl': 160,
            'annual_avg_estabs': 14,
            'tot_wages': 2300,
        },
    ]
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'year',
                'own_code',
                'industry_code',
                'annual_avg_emplvl',
                'annual_avg_estabs',
                'tot_wages',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    config = QCEWBenchmarkConfig(
        qcew_csv_path=csv_path,
        embedding_parquet=embed_path,
        test_size=0.5,
        random_state=0,
    )
    results = run_qcew_employment_benchmark(config)

    assert set(results.keys()) == {'embedding', 'one_hot', 'hybrid', 'metadata'}
    assert results['embedding']['rmse'] >= 0.0
    assert results['one_hot']['r2'] <= 1.0
