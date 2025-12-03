# -------------------------------------------------------------------------------------------------
# QCEW Benchmark
# -------------------------------------------------------------------------------------------------
'''
QCEW (Quarterly Census of Employment and Wages) downstream regression benchmark.

Contains:
- QCEWBenchmarkConfig: Configuration dataclass for the benchmark
- run_qcew_employment_benchmark: Compare embedding, one-hot, and hybrid regressors
'''

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import polars as pl
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from naics_embedder.text_model.hyperbolic import LorentzOps

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def _sorted_embedding_columns(columns: Sequence[str], prefix: str) -> list[str]:
    '''Return embedding columns sorted numerically by suffix.'''

    relevant = [col for col in columns if col.startswith(prefix)]
    if not relevant:
        return []

    def _sort_key(name: str) -> tuple[int, int | str]:
        suffix = name[len(prefix):]
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    return sorted(relevant, key=_sort_key)

def _load_qcew_slice(config: 'QCEWBenchmarkConfig') -> pl.DataFrame:
    required_cols = [
        'year',
        'own_code',
        'industry_code',
        'annual_avg_emplvl',
        'annual_avg_estabs',
        'tot_wages',
    ]

    lazy = pl.scan_csv(
        str(config.qcew_csv_path),
        null_values=['', 'NA', 'N/A'],
        infer_schema_length=100,
    )

    filtered = (
        lazy.select(required_cols).with_columns(
            pl.col('industry_code').cast(pl.Utf8).str.strip_chars(),
            pl.col('year').cast(pl.Int32),
            pl.col('own_code').cast(pl.Int32),
            pl.col('annual_avg_emplvl').cast(pl.Float64),
            pl.col('annual_avg_estabs').cast(pl.Float64),
            pl.col('tot_wages').cast(pl.Float64),
        ).filter(pl.col('year') == config.year).filter(
            pl.col('own_code') == config.ownership_code
        ).filter(pl.col('industry_code').str.len_chars() == config.min_code_length)
    )

    aggregated = (
        filtered.group_by('industry_code').agg(
            pl.col('annual_avg_emplvl').mean().alias('avg_emp'),
            pl.col('annual_avg_estabs').mean().alias('avg_estabs'),
            pl.col('tot_wages').mean().alias('avg_wages'),
        ).with_columns(
            (pl.col('avg_emp') + 1e-6).alias('avg_emp'),
            (pl.col('avg_estabs') + 1e-6).alias('avg_estabs'),
            (pl.col('avg_wages') + 1e-6).alias('avg_wages'),
        ).with_columns(pl.col('avg_emp').log().alias('log_avg_emp'))
    )

    return aggregated.collect()

def _tangent_from_frame(
    frame: pl.DataFrame, embed_cols: Sequence[str], curvature: float
) -> np.ndarray:
    tensor = torch.from_numpy(frame.select(embed_cols).to_numpy()).float()
    with torch.no_grad():
        tangent = LorentzOps.log_map_zero(tensor, c=curvature)
    return tangent[:, 1:].detach().cpu().numpy()

def _fit_ridge_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    model = Ridge(alpha=alpha)
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    return {
        'r2': float(r2_score(y[test_idx], preds)),
        'rmse': float(np.sqrt(mean_squared_error(y[test_idx], preds))),
    }

# -------------------------------------------------------------------------------------------------
# Configuration and benchmark
# -------------------------------------------------------------------------------------------------

@dataclass
class QCEWBenchmarkConfig:
    '''Configuration for the QCEW downstream regression benchmark.'''

    qcew_csv_path: Path
    embedding_parquet: Path
    embedding_prefix: str = 'hgcn_e'
    code_column: str = 'code'
    curvature: float = 1.0
    year: int = 2022
    ownership_code: int = 5
    min_code_length: int = 6
    test_size: float = 0.2
    random_state: int = 42
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.qcew_csv_path = Path(self.qcew_csv_path)
        self.embedding_parquet = Path(self.embedding_parquet)

def run_qcew_employment_benchmark(config: QCEWBenchmarkConfig) -> Dict[str, Dict[str, float]]:
    '''Compare embedding, one-hot, and hybrid regressors on QCEW employment prediction.'''

    if not config.qcew_csv_path.exists():
        raise FileNotFoundError(f'QCEW CSV not found: {config.qcew_csv_path}')
    if not config.embedding_parquet.exists():
        raise FileNotFoundError(f'Embeddings parquet not found: {config.embedding_parquet}')

    qcew_df = _load_qcew_slice(config)
    if qcew_df.is_empty():
        raise ValueError('Filtered QCEW dataframe is empty; check year/ownership filters.')

    embed_df = pl.read_parquet(str(config.embedding_parquet))
    embed_cols = _sorted_embedding_columns(embed_df.columns, config.embedding_prefix)
    if not embed_cols:
        raise ValueError(
            f'No embedding columns with prefix "{config.embedding_prefix}" found in '
            f'{config.embedding_parquet}'
        )
    if config.code_column not in embed_df.columns:
        raise ValueError(f'Column "{config.code_column}" missing from embeddings parquet.')

    joined = qcew_df.join(
        embed_df.select([config.code_column, *embed_cols]),
        left_on='industry_code',
        right_on=config.code_column,
        how='inner',
    )
    if joined.is_empty():
        raise ValueError('No overlapping NAICS codes between QCEW slice and embeddings.')

    y = joined.get_column('log_avg_emp').to_numpy()
    codes = joined.get_column('industry_code').to_list()
    groups = np.array(codes)

    X_embed = _tangent_from_frame(joined, embed_cols, config.curvature)
    scalars = joined.select(['avg_estabs', 'avg_wages']).to_numpy()
    log_scalars = np.log1p(scalars).astype(np.float64)
    X_hybrid = np.concatenate([X_embed, log_scalars], axis=1)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
    X_one_hot = encoder.fit_transform(np.array(codes).reshape(-1, 1))

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.test_size, random_state=config.random_state
    )
    try:
        train_idx, test_idx = next(splitter.split(X_embed, y, groups=groups))
    except ValueError as exc:
        raise ValueError(
            'Unable to create a cold-start split. Provide more NAICS codes or adjust test_size.'
        ) from exc

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError('Cold-start split produced an empty train/test partition.')

    results = {
        'embedding':
        _fit_ridge_model(
            X_embed, y, train_idx=train_idx, test_idx=test_idx, alpha=config.ridge_alpha
        ),
        'one_hot':
        _fit_ridge_model(
            X_one_hot, y, train_idx=train_idx, test_idx=test_idx, alpha=config.ridge_alpha
        ),
        'hybrid':
        _fit_ridge_model(
            X_hybrid, y, train_idx=train_idx, test_idx=test_idx, alpha=config.ridge_alpha
        ),
        'metadata': {
            'n_samples': float(len(y)),
            'n_train': float(len(train_idx)),
            'n_test': float(len(test_idx)),
            'n_joined_codes': float(joined.height),
        },
    }

    return results
