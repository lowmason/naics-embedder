# -------------------------------------------------------------------------------------------------
# QCEW Benchmark
# -------------------------------------------------------------------------------------------------
'''
QCEW (Quarterly Census of Employment and Wages) downstream regression benchmark.

Contains:
- QCEWBenchmarkConfig: Configuration dataclass for the benchmark
- run_qcew_employment_benchmark: Compare embedding, one-hot, and hybrid regressors
- run_qcew_multilevel_benchmark: Compare across all NAICS levels (2-6 digits)
'''

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

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

def _load_qcew_slice(
    config: 'QCEWBenchmarkConfig',
    code_length: int | None = None,
) -> pl.DataFrame:
    '''Load and aggregate QCEW data for a specific code length.

    Args:
        config: Benchmark configuration.
        code_length: NAICS code length to filter (overrides config.min_code_length if provided).

    Returns:
        Aggregated QCEW dataframe with avg_emp, avg_estabs, avg_wages, log_avg_emp.
    '''
    required_cols = [
        'year',
        'own_code',
        'industry_code',
        'annual_avg_emplvl',
        'annual_avg_estabs',
        'tot_wages',
    ]

    target_length = code_length if code_length is not None else config.min_code_length

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
        ).filter(pl.col('industry_code').str.len_chars() == target_length)
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


# -------------------------------------------------------------------------------------------------
# Multi-level benchmark
# -------------------------------------------------------------------------------------------------

NAICS_LEVEL_NAMES = {
    2: 'sector',
    3: 'subsector',
    4: 'industry_group',
    5: 'naics_industry',
    6: 'national_industry',
}


@dataclass
class QCEWMultilevelConfig:
    '''Configuration for multi-level QCEW benchmark.'''

    qcew_csv_path: Path
    embedding_parquet: Path
    embedding_prefix: str = 'hgcn_e'
    code_column: str = 'code'
    curvature: float = 1.0
    year: int = 2022
    ownership_code: int = 5
    levels: Sequence[int] = (2, 3, 4, 5, 6)
    test_size: float = 0.2
    random_state: int = 42
    ridge_alpha: float = 1.0
    min_samples: int = 10

    def __post_init__(self) -> None:
        self.qcew_csv_path = Path(self.qcew_csv_path)
        self.embedding_parquet = Path(self.embedding_parquet)
        self.levels = tuple(sorted(set(self.levels)))


def _run_single_level_benchmark(
    qcew_df: pl.DataFrame,
    embed_df: pl.DataFrame,
    embed_cols: Sequence[str],
    config: QCEWMultilevelConfig,
    level: int,
) -> Dict[str, Any] | None:
    '''Run benchmark for a single NAICS level.

    Returns None if insufficient data for the level.
    '''
    # Filter embeddings to codes of the target length
    level_embed_df = embed_df.filter(
        pl.col(config.code_column).str.len_chars() == level
    )

    if level_embed_df.is_empty():
        logger.warning(f'No embeddings found for {level}-digit codes')
        return None

    # Join with QCEW data
    joined = qcew_df.join(
        level_embed_df.select([config.code_column, *embed_cols]),
        left_on='industry_code',
        right_on=config.code_column,
        how='inner',
    )

    if joined.height < config.min_samples:
        logger.warning(
            f'Insufficient samples for {level}-digit level: {joined.height} < {config.min_samples}'
        )
        return None

    y = joined.get_column('log_avg_emp').to_numpy()
    codes = joined.get_column('industry_code').to_list()
    groups = np.array(codes)

    # Prepare feature matrices
    X_embed = _tangent_from_frame(joined, embed_cols, config.curvature)
    scalars = joined.select(['avg_estabs', 'avg_wages']).to_numpy()
    log_scalars = np.log1p(scalars).astype(np.float64)
    X_hybrid = np.concatenate([X_embed, log_scalars], axis=1)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
    X_one_hot = encoder.fit_transform(np.array(codes).reshape(-1, 1))

    # Create train/test split
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.test_size, random_state=config.random_state
    )
    try:
        train_idx, test_idx = next(splitter.split(X_embed, y, groups=groups))
    except ValueError:
        logger.warning(f'Unable to create split for {level}-digit level')
        return None

    if len(train_idx) < 2 or len(test_idx) < 2:
        logger.warning(f'Split too small for {level}-digit level')
        return None

    return {
        'embedding': _fit_ridge_model(
            X_embed, y, train_idx=train_idx, test_idx=test_idx, alpha=config.ridge_alpha
        ),
        'one_hot': _fit_ridge_model(
            X_one_hot, y, train_idx=train_idx, test_idx=test_idx, alpha=config.ridge_alpha
        ),
        'hybrid': _fit_ridge_model(
            X_hybrid, y, train_idx=train_idx, test_idx=test_idx, alpha=config.ridge_alpha
        ),
        'metadata': {
            'n_samples': float(len(y)),
            'n_train': float(len(train_idx)),
            'n_test': float(len(test_idx)),
            'n_unique_codes': float(len(set(codes))),
            'embedding_dim': float(X_embed.shape[1]),
            'one_hot_dim': float(X_one_hot.shape[1]),
        },
    }


def run_qcew_multilevel_benchmark(
    config: QCEWMultilevelConfig,
) -> Dict[str, Any]:
    '''Compare embedding vs one-hot encoding across all NAICS levels.

    This benchmark evaluates how well learned embeddings generalize across
    different levels of the NAICS hierarchy compared to simple one-hot encoding.

    Args:
        config: Multi-level benchmark configuration.

    Returns:
        Nested dictionary: {level_name: {model_type: {metric: value}}}

        Example structure:
        {
            'level_2_sector': {
                'embedding': {'r2': 0.85, 'rmse': 0.12},
                'one_hot': {'r2': 0.72, 'rmse': 0.18},
                'hybrid': {'r2': 0.87, 'rmse': 0.11},
                'metadata': {...}
            },
            'level_3_subsector': {...},
            ...
            'summary': {
                'embedding_avg_r2': 0.82,
                'one_hot_avg_r2': 0.65,
                'embedding_advantage': 0.17,
                ...
            }
        }
    '''
    if not config.qcew_csv_path.exists():
        raise FileNotFoundError(f'QCEW CSV not found: {config.qcew_csv_path}')
    if not config.embedding_parquet.exists():
        raise FileNotFoundError(f'Embeddings parquet not found: {config.embedding_parquet}')

    # Load embeddings once
    embed_df = pl.read_parquet(str(config.embedding_parquet))
    embed_cols = _sorted_embedding_columns(embed_df.columns, config.embedding_prefix)
    if not embed_cols:
        raise ValueError(
            f'No embedding columns with prefix "{config.embedding_prefix}" found in '
            f'{config.embedding_parquet}'
        )
    if config.code_column not in embed_df.columns:
        raise ValueError(f'Column "{config.code_column}" missing from embeddings parquet.')

    results: Dict[str, Any] = {}
    successful_levels: list[int] = []

    # Run benchmark for each level
    for level in config.levels:
        level_name = f'level_{level}_{NAICS_LEVEL_NAMES.get(level, "unknown")}'
        logger.info(f'Running benchmark for {level}-digit NAICS codes ({level_name})...')

        # Load QCEW data for this level
        # Create a temporary single-level config
        single_config = QCEWBenchmarkConfig(
            qcew_csv_path=config.qcew_csv_path,
            embedding_parquet=config.embedding_parquet,
            embedding_prefix=config.embedding_prefix,
            code_column=config.code_column,
            curvature=config.curvature,
            year=config.year,
            ownership_code=config.ownership_code,
            min_code_length=level,
            test_size=config.test_size,
            random_state=config.random_state,
            ridge_alpha=config.ridge_alpha,
        )

        qcew_df = _load_qcew_slice(single_config, code_length=level)
        if qcew_df.is_empty():
            logger.warning(f'No QCEW data found for {level}-digit codes')
            continue

        level_results = _run_single_level_benchmark(
            qcew_df, embed_df, embed_cols, config, level
        )

        if level_results is not None:
            results[level_name] = level_results
            successful_levels.append(level)
            logger.info(
                f'  {level_name}: embedding R²={level_results["embedding"]["r2"]:.3f}, '
                f'one_hot R²={level_results["one_hot"]["r2"]:.3f}, '
                f'n={level_results["metadata"]["n_samples"]:.0f}'
            )

    # Compute summary statistics
    if successful_levels:
        embed_r2_values = [results[f'level_{l}_{NAICS_LEVEL_NAMES.get(l, "unknown")}']['embedding']['r2']
                          for l in successful_levels]
        onehot_r2_values = [results[f'level_{l}_{NAICS_LEVEL_NAMES.get(l, "unknown")}']['one_hot']['r2']
                           for l in successful_levels]
        hybrid_r2_values = [results[f'level_{l}_{NAICS_LEVEL_NAMES.get(l, "unknown")}']['hybrid']['r2']
                           for l in successful_levels]

        embed_rmse_values = [results[f'level_{l}_{NAICS_LEVEL_NAMES.get(l, "unknown")}']['embedding']['rmse']
                            for l in successful_levels]
        onehot_rmse_values = [results[f'level_{l}_{NAICS_LEVEL_NAMES.get(l, "unknown")}']['one_hot']['rmse']
                             for l in successful_levels]

        results['summary'] = {
            'levels_evaluated': {str(l): NAICS_LEVEL_NAMES.get(l, 'unknown') for l in successful_levels},
            'embedding': {
                'avg_r2': float(np.mean(embed_r2_values)),
                'std_r2': float(np.std(embed_r2_values)),
                'avg_rmse': float(np.mean(embed_rmse_values)),
            },
            'one_hot': {
                'avg_r2': float(np.mean(onehot_r2_values)),
                'std_r2': float(np.std(onehot_r2_values)),
                'avg_rmse': float(np.mean(onehot_rmse_values)),
            },
            'hybrid': {
                'avg_r2': float(np.mean(hybrid_r2_values)),
                'std_r2': float(np.std(hybrid_r2_values)),
            },
            'comparison': {
                'embedding_vs_onehot_r2_diff': float(np.mean(embed_r2_values) - np.mean(onehot_r2_values)),
                'embedding_wins': sum(1 for e, o in zip(embed_r2_values, onehot_r2_values) if e > o),
                'onehot_wins': sum(1 for e, o in zip(embed_r2_values, onehot_r2_values) if o > e),
                'per_level_r2_diff': {
                    f'level_{l}': float(e - o)
                    for l, e, o in zip(successful_levels, embed_r2_values, onehot_r2_values)
                },
            },
        }

    return results


def print_multilevel_comparison(results: Dict[str, Any]) -> None:
    '''Print a formatted comparison table of multi-level benchmark results.'''
    print('\n' + '=' * 80)
    print('QCEW MULTI-LEVEL BENCHMARK RESULTS')
    print('=' * 80)

    # Print per-level results
    print(f'\n{"Level":<25} {"Embedding R²":>12} {"One-Hot R²":>12} {"Hybrid R²":>12} {"Δ (Emb-OH)":>12} {"N":>8}')
    print('-' * 80)

    for key, level_results in results.items():
        if key == 'summary':
            continue

        embed_r2 = level_results['embedding']['r2']
        onehot_r2 = level_results['one_hot']['r2']
        hybrid_r2 = level_results['hybrid']['r2']
        diff = embed_r2 - onehot_r2
        n_samples = level_results['metadata']['n_samples']

        # Format level name nicely
        level_name = key.replace('level_', '').replace('_', ' ').title()

        print(f'{level_name:<25} {embed_r2:>12.3f} {onehot_r2:>12.3f} {hybrid_r2:>12.3f} {diff:>+12.3f} {n_samples:>8.0f}')

    # Print summary
    if 'summary' in results:
        summary = results['summary']
        print('-' * 80)
        print(f'{"AVERAGE":<25} {summary["embedding"]["avg_r2"]:>12.3f} {summary["one_hot"]["avg_r2"]:>12.3f} '
              f'{summary["hybrid"]["avg_r2"]:>12.3f} {summary["comparison"]["embedding_vs_onehot_r2_diff"]:>+12.3f}')

        print('\n' + '-' * 40)
        print('SUMMARY')
        print('-' * 40)
        print(f'Embedding wins: {summary["comparison"]["embedding_wins"]} levels')
        print(f'One-hot wins:   {summary["comparison"]["onehot_wins"]} levels')
        print(f'Avg R² improvement: {summary["comparison"]["embedding_vs_onehot_r2_diff"]:+.3f}')

    print('=' * 80 + '\n')
