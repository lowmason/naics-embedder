# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    mean_squared_error,
    normalized_mutual_info_score,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder

from naics_embedder.text_model.hyperbolic import LorentzOps
from naics_embedder.utils.naics_hierarchy import NaicsHierarchy

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Validation Metrics
# -------------------------------------------------------------------------------------------------

def compute_validation_metrics(
    emb: torch.Tensor,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    c: float = 1.0,
    top_k: int = 1,
    *,
    as_tensors: bool = False,
) -> Union[Dict[str, float], Dict[str, torch.Tensor]]:
    '''Compute validation metrics for hyperbolic embeddings.

    Args:
        emb: Embeddings tensor of shape ``(N, embedding_dim+1)``.
        anchors: Anchor indices, shape ``(batch_size,)``.
        positives: Positive indices, shape ``(batch_size,)``.
        negatives: Negative indices, shape ``(batch_size, k_negatives)``.
        c: Curvature parameter (default: 1.0).
        top_k: Number of top negatives to consider for auxiliary accuracy.
        as_tensors: Return torch scalars instead of Python floats (for Lightning logging).

    Returns:
        Mapping of metric names to values (either tensors or floats).
    '''
    batch_size, k_negatives = negatives.shape
    effective_top_k = max(1, min(top_k, k_negatives))

    anchor_emb = emb[anchors]
    positive_emb = emb[positives]

    positive_dist = LorentzOps.lorentz_distance(anchor_emb, positive_emb, c=c)

    negative_emb = emb[negatives.reshape(-1)].view(batch_size, k_negatives, -1)
    anchor_expanded = anchor_emb.unsqueeze(1).expand(-1, k_negatives, -1)
    negative_dist = LorentzOps.lorentz_distance(
        anchor_expanded.reshape(-1, anchor_expanded.size(-1)),
        negative_emb.reshape(-1, negative_emb.size(-1)),
        c=c,
    ).view(batch_size, k_negatives)

    avg_positive_dist = positive_dist.mean()
    avg_negative_dist = negative_dist.mean()

    all_distances = torch.cat([positive_dist, negative_dist.reshape(-1)], dim=0)
    distance_spread = torch.div(all_distances.std(), all_distances.mean().clamp_min(1e-8))

    relation_accuracy = (positive_dist.unsqueeze(1) < negative_dist).all(dim=1).float().mean()

    closest_negatives = torch.topk(negative_dist, k=effective_top_k, dim=1, largest=False).values
    top_k_relation_accuracy = (positive_dist.unsqueeze(1)
                               < closest_negatives).all(dim=1).float().mean()

    all_dists_per_anchor = torch.cat([positive_dist.unsqueeze(1), negative_dist], dim=1)
    order = torch.argsort(all_dists_per_anchor, dim=1)
    positive_rank_tensor = torch.argmax((order == 0).int(), dim=1)
    mean_positive_rank = positive_rank_tensor.float().mean()

    metrics = {
        'avg_positive_dist': avg_positive_dist,
        'avg_negative_dist': avg_negative_dist,
        'distance_spread': distance_spread,
        'relation_accuracy': relation_accuracy,
        'top_k_relation_accuracy': top_k_relation_accuracy,
        'mean_positive_rank': mean_positive_rank,
    }

    if as_tensors:
        return metrics

    return {k: float(v.detach().cpu()) for k, v in metrics.items()}

# -------------------------------------------------------------------------------------------------
# Downstream evaluation data structures
# -------------------------------------------------------------------------------------------------

def _sorted_embedding_columns(columns: Sequence[str], prefix: str) -> List[str]:
    '''Return embedding columns sorted numerically by suffix.'''

    relevant = [col for col in columns if col.startswith(prefix)]
    if not relevant:
        return []

    def _sort_key(name: str) -> Tuple[int, Union[int, str]]:
        suffix = name[len(prefix):]
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    return sorted(relevant, key=_sort_key)

@dataclass
class GraphEmbeddingDataset:
    '''Container for a set of hyperbolic graph embeddings.'''

    embeddings: torch.Tensor
    codes: Sequence[str]
    levels: Sequence[int]

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError('embeddings tensor must be 2D')

        num_nodes = self.embeddings.size(0)
        if num_nodes != len(self.codes) or num_nodes != len(self.levels):
            raise ValueError(
                'Embeddings, codes, and levels must have the same first dimension '
                f'(got embeddings={num_nodes}, codes={len(self.codes)}, levels={len(self.levels)})'
            )

        self.codes = list(self.codes)
        self.levels = [int(level) for level in self.levels]

    @classmethod
    def from_dataframe(
        cls,
        frame: pl.DataFrame,
        *,
        embedding_prefix: str = 'hgcn_e',
        code_column: str = 'code',
        level_column: str = 'level',
    ) -> 'GraphEmbeddingDataset':
        '''Build a dataset from a parquet dataframe.'''

        if code_column not in frame.columns:
            raise ValueError(f'Expected column "{code_column}" in embeddings parquet')
        if level_column not in frame.columns:
            raise ValueError(f'Expected column "{level_column}" in embeddings parquet')

        embed_cols = _sorted_embedding_columns(frame.columns, embedding_prefix)
        if not embed_cols:
            raise ValueError(
                f'No embedding columns found with prefix "{embedding_prefix}". '
                'Set embedding_prefix to match hgcn_e* or hyp_e* columns.'
            )

        tensor = torch.from_numpy(frame.select(embed_cols).to_numpy()).float()
        codes = frame.get_column(code_column).to_list()
        levels = frame.get_column(level_column).to_list()

        return cls(embeddings=tensor, codes=codes, levels=levels)

class GraphDownstreamEvaluator:
    '''Evaluate downstream metrics for graph-refined NAICS embeddings.'''

    def __init__(self, dataset: GraphEmbeddingDataset, *, curvature: float = 1.0):
        self.dataset = dataset
        self.curvature = float(curvature)
        self._distance_cache: Optional[torch.Tensor] = None
        self._tangent_cache: Optional[np.ndarray] = None
        self._code_to_idx = {code: idx for idx, code in enumerate(dataset.codes)}
        self._level_to_indices: Dict[int, List[int]] = {}
        for idx, level in enumerate(dataset.levels):
            self._level_to_indices.setdefault(level, []).append(idx)

    def _pairwise_distances(self) -> torch.Tensor:
        if self._distance_cache is not None:
            return self._distance_cache

        emb = self.dataset.embeddings.detach()
        with torch.no_grad():
            u = emb.unsqueeze(1)
            v = emb.unsqueeze(0)
            spatial = torch.sum(u[:, :, 1:] * v[:, :, 1:], dim=-1)
            time = u[:, :, 0] * v[:, :, 0]
            dot = spatial - time
            arccosh_arg = torch.clamp(-dot, min=1.0 + 1e-6)
            sqrt_c = torch.sqrt(torch.tensor(self.curvature, device=emb.device, dtype=emb.dtype))
            distances = sqrt_c * torch.acosh(arccosh_arg)

        self._distance_cache = distances.detach().cpu()
        return self._distance_cache

    def _tangent_features(self) -> np.ndarray:
        if self._tangent_cache is not None:
            return self._tangent_cache

        with torch.no_grad():
            tangent = LorentzOps.log_map_zero(self.dataset.embeddings, c=self.curvature)

        # Drop the time coordinate (always zero in tangent space)
        self._tangent_cache = tangent[:, 1:].detach().cpu().numpy()
        return self._tangent_cache

    def taxonomy_reconstruction(
        self,
        hierarchy: NaicsHierarchy,
        *,
        k_values: Sequence[int] = (1, 3, 5),
    ) -> Dict[str, float]:
        '''Evaluate parent retrieval accuracy from embeddings alone.'''

        k_values = sorted({int(k) for k in k_values if int(k) > 0})
        if not k_values:
            raise ValueError('k_values must contain at least one positive integer')

        distances = self._pairwise_distances().numpy()
        hits = np.zeros(len(k_values), dtype=np.float64)
        evaluated = 0

        for child_idx, code in enumerate(self.dataset.codes):
            parent = hierarchy.get_parent(code)
            if parent is None:
                continue

            parent_idx = self._code_to_idx.get(parent)
            if parent_idx is None:
                continue

            parent_level = self.dataset.levels[child_idx] - 1
            candidate_indices = [
                idx for idx in self._level_to_indices.get(parent_level, []) if idx != child_idx
            ]
            if not candidate_indices:
                continue

            ordered = np.argsort(distances[child_idx, candidate_indices])
            candidates_ranked = [candidate_indices[i] for i in ordered]
            if not candidates_ranked:
                continue

            evaluated += 1
            for idx, k in enumerate(k_values):
                top = candidates_ranked[:min(k, len(candidates_ranked))]
                if parent_idx in top:
                    hits[idx] += 1

        if evaluated == 0:
            raise ValueError('No eligible nodes found for taxonomy reconstruction.')

        results = {'evaluated_nodes': float(evaluated)}
        for k, hit in zip(k_values, hits):
            results[f'top_{k}_parent_accuracy'] = float(hit / evaluated)
        return results

    def industry_similarity(
        self,
        hierarchy: NaicsHierarchy,
        *,
        k_values: Sequence[int] = (5, 10),
    ) -> Dict[str, float]:
        '''Measure whether nearest neighbours correspond to semantic siblings.'''

        k_values = sorted({int(k) for k in k_values if int(k) > 0})
        if not k_values:
            raise ValueError('k_values must contain at least one positive integer')

        distances = self._pairwise_distances().numpy()
        precision_lists: Dict[int, List[float]] = {k: [] for k in k_values}
        first_sibling_ranks: List[int] = []

        for code, idx in self._code_to_idx.items():
            siblings = [
                sib for sib in hierarchy.get_siblings(code)
                if sib in self._code_to_idx and sib != code
            ]
            if not siblings:
                continue

            ordered = np.argsort(distances[idx])
            ordered = [int(o) for o in ordered if o != idx]
            if not ordered:
                continue

            sibling_indices = {self._code_to_idx[s] for s in siblings}
            for k in k_values:
                top = ordered[:min(k, len(ordered))]
                denom = max(1, min(k, len(ordered)))
                hits = len(set(top) & sibling_indices)
                precision_lists[k].append(hits / denom)

            sibling_positions = [
                ordered.index(sib_idx) + 1 for sib_idx in sibling_indices if sib_idx in ordered
            ]
            if sibling_positions:
                first_sibling_ranks.append(min(sibling_positions))

        evaluated = first_sibling_ranks.__len__()
        if evaluated == 0:
            raise ValueError('No nodes with sibling matches were found for similarity metrics.')

        results = {'evaluated_nodes': float(evaluated)}
        for k, values in precision_lists.items():
            if values:
                results[f'precision@{k}'] = float(np.mean(values))
        results['mean_first_sibling_rank'] = float(np.mean(first_sibling_ranks))
        return results

    def clustering_quality(
        self,
        *,
        digits: Sequence[int] = (2, 3),
        random_state: int = 42,
    ) -> Dict[str, float]:
        '''Compute ARI/NMI by clustering embeddings and comparing to NAICS prefixes.'''

        features = self._tangent_features()
        results: Dict[str, float] = {}

        for length in digits:
            mask = np.array([len(code) >= length for code in self.dataset.codes], dtype=bool)
            if mask.sum() < 2:
                continue

            labels = [code[:length] for code, keep in zip(self.dataset.codes, mask) if keep]
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                continue

            data = features[mask]
            n_clusters = min(len(unique_labels), len(data))
            if n_clusters < 2:
                continue

            model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state)
            preds = model.fit_predict(data)
            results[f'ari_{length}digit'] = float(adjusted_rand_score(labels, preds))
            results[f'nmi_{length}digit'] = float(normalized_mutual_info_score(labels, preds))

        if not results:
            raise ValueError('Not enough samples to compute clustering metrics.')
        return results

    def classification_benchmark(
        self,
        *,
        digits: int = 2,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        '''Train a linear classifier to predict NAICS prefixes.'''

        features = self._tangent_features()
        mask = np.array([len(code) >= digits for code in self.dataset.codes], dtype=bool)
        data = features[mask]
        labels = np.array([code[:digits] for code, keep in zip(self.dataset.codes, mask) if keep])

        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            raise ValueError('Classification benchmark requires at least two label classes.')
        if np.any(counts < 2):
            raise ValueError('Each class must appear at least twice for a stratified split.')

        X_train, X_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        model = LogisticRegression(max_iter=500, multi_class='auto')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            'accuracy': float(accuracy_score(y_test, preds)),
            'macro_f1': float(f1_score(y_test, preds, average='macro')),
            'n_train': float(len(y_train)),
            'n_test': float(len(y_test)),
        }

def run_graph_downstream_suite(
    dataset: GraphEmbeddingDataset,
    hierarchy: NaicsHierarchy,
    *,
    curvature: float = 1.0,
    taxonomy_k: Sequence[int] = (1, 3, 5),
    sibling_k: Sequence[int] = (5, 10),
    clustering_digits: Sequence[int] = (2, 3),
    classification_digits: int = 2,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    '''Convenience helper to run all downstream evaluations sequentially.'''

    evaluator = GraphDownstreamEvaluator(dataset, curvature=curvature)
    taxonomy = evaluator.taxonomy_reconstruction(hierarchy, k_values=taxonomy_k)
    similarity = evaluator.industry_similarity(hierarchy, k_values=sibling_k)
    clustering = evaluator.clustering_quality(digits=clustering_digits, random_state=random_state)
    classification = evaluator.classification_benchmark(
        digits=classification_digits, random_state=random_state
    )

    return {
        'taxonomy': taxonomy,
        'similarity': similarity,
        'clustering': clustering,
        'classification': classification,
    }

# -------------------------------------------------------------------------------------------------
# QCEW benchmark
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

def _load_qcew_slice(config: QCEWBenchmarkConfig) -> pl.DataFrame:
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
