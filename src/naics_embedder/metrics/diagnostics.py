'''
Req 6's structural diagnostics over every code of a table: reported, never selected on and
never a headline (Req 1; Req 6; Verification "Diagnostics").

- **Sector separation.** The AUC of distance between same-sector and cross-sector pairs: the
  probability that a cross-sector pair lies farther apart than a same-sector pair, ties counting
  half.
- **Within-sector rank correlation.** Per query, Spearman's correlation of distance with D* over
  the other codes of its sector; averaged over queries, and over sectors (each sector's mean over
  its queries).
- **MAP over ancestors.** Per non-sector query, the average precision of its ancestors among
  all other codes ranked by distance.
- **NDCG@k.** Gains are integer depths of the lowest common ancestor (0 across sectors, 1 for a
  shared sector, up to 4), with linear gain and a 1/log2(rank + 1) discount.
- **Distance Pearson.** The Pearson correlation of distance with D* over all pairs: the
  statistic formerly named "cophenetic", which no dendrogram underlies.
- **Parent retrieval@k.** Per non-sector query, whether its parent is among its k nearest codes.
  The 522 unary pairs, five-digit industries whose only child is their six-digit code (Req 9),
  are not scored.

D* is the tree metric through a virtual root (Req 7): depth_i + depth_j − 2 depth_LCA, where a
sector has depth 1. It is computed here from the codes themselves (``panels.decoding``'s
lineage, combined sectors as one). Distances are the arm's own: Euclidean, cosine for a
spherical arm, and for a hyperbolic arm the geodesic distance between the exponential maps of
its tangent coordinates at the origin. A tie in distance is broken against relevance, as the
decoding scorer breaks it against the truth, except in the AUC and Spearman's ranks, which
average ties. No statistic has a threshold, and the report has no pass or fail.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata

from naics_embedder.panels.decoding import (
    code_lineage,
    cosine_distances,
    euclidean_distances,
    lorentz_distances,
)
from naics_embedder.panels.regressor import coordinate_matrix

Geometry = Literal['euclidean', 'spherical', 'hyperbolic']
GEOMETRIES = ('euclidean', 'spherical', 'hyperbolic')
MAX_DEPTH = 5
NDCG_KS = (5, 10, 20)
PARENT_KS = (1, 5)

# -------------------------------------------------------------------------------------------------
# The report
# -------------------------------------------------------------------------------------------------

class _Report(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

class SectorSeparation(_Report):
    auc: float
    same_sector_pairs: int
    cross_sector_pairs: int

class WithinSectorRankCorrelation(_Report):
    '''The means are None when no query's correlation is defined (a collapsed embedding).'''

    mean_over_queries: Optional[float]
    mean_over_sectors: Optional[float]
    by_sector: Dict[str, float]
    queries: int
    undefined_queries: int

class ByQueryLevel(_Report):
    value: float
    queries: int
    by_level: Dict[str, float]

class DistancePearson(_Report):
    '''None when every distance is equal.'''

    value: Optional[float]
    pairs: int

class ParentRetrieval(_Report):
    at: Dict[str, float]
    queries: int
    unary_pairs_excluded: int

class DiagnosticsReport(_Report):
    '''Req 6's statistics, stratified as it lists them; nothing else.'''

    codes: int
    geometry: Geometry
    curvature: Optional[float]
    sector_separation: SectorSeparation
    within_sector_rank_correlation: WithinSectorRankCorrelation
    map_over_ancestors: ByQueryLevel
    ndcg: Dict[str, ByQueryLevel]
    distance_pearson: DistancePearson
    parent_retrieval: ParentRetrieval

# -------------------------------------------------------------------------------------------------
# The tree
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Tree:
    '''
    The codes' tree.

    Attributes:
        lineage: (codes, 5) row numbers of each code's ancestors from its sector down to itself,
            −1 below its depth.
        depth: Each code's depth, 1 for a sector.
    '''

    codes: Tuple[str, ...]
    lineage: np.ndarray
    depth: np.ndarray

    @classmethod
    def from_codes(cls, codes: Sequence[str]) -> 'Tree':
        '''
        Raises:
            ValueError: If a code repeats or an ancestor of a code is not among the codes.
        '''

        codes = tuple(codes)
        position = {code: row for row, code in enumerate(codes)}
        if len(position) != len(codes):
            raise ValueError('a code repeats')
        lineage = np.full((len(codes), MAX_DEPTH), -1, dtype=np.int64)
        for row, code in enumerate(codes):
            chain = code_lineage(code)
            missing = [ancestor for ancestor in chain if ancestor not in position]
            if missing:
                raise ValueError(f'{code}: its ancestor {missing[0]} is not among the codes')
            lineage[row, :len(chain)] = [position[ancestor] for ancestor in chain]
        return cls(codes, lineage, (lineage >= 0).sum(axis=1))

    @property
    def sector(self) -> np.ndarray:
        return self.lineage[:, 0]

    @property
    def parent(self) -> np.ndarray:
        '''Each code's parent row, −1 for a sector.'''

        rows = np.arange(len(self.codes))
        return np.where(self.depth > 1, self.lineage[rows, np.maximum(self.depth - 2, 0)], -1)

    def lca_depth(self) -> np.ndarray:
        '''(codes, codes) depth of each pair's lowest common ancestor, 0 across sectors.'''

        depth = np.zeros((len(self.codes), len(self.codes)), dtype=np.int64)
        for column in self.lineage.T:
            depth += (column[:, None] == column[None, :]) & (column[:, None] >= 0)
        return depth

    def unary_children(self) -> np.ndarray:
        '''Whether each code is the six-digit half of a unary pair (Req 9).'''

        parent = self.parent
        counts = np.bincount(parent[parent >= 0], minlength=len(self.codes))
        six_digit = np.array([len(code) == 6 for code in self.codes])
        return six_digit & (parent >= 0) & (counts[np.maximum(parent, 0)] == 1)

# -------------------------------------------------------------------------------------------------
# Distances
# -------------------------------------------------------------------------------------------------

def pairwise_distances(
    matrix: np.ndarray, geometry: Geometry, curvature: float = 1.0
) -> np.ndarray:
    '''
    The arm's own distances between every pair of rows.

    A hyperbolic arm's rows are tangent coordinates at the origin (the export form); each maps to
    the hyperboloid of curvature −``curvature`` by the exponential map at the origin.
    '''

    points = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float64))
    if geometry == 'euclidean':
        return euclidean_distances(points, points).numpy()
    if geometry == 'spherical':
        return cosine_distances(points, points).numpy()
    if geometry != 'hyperbolic':
        raise ValueError(f'geometry must be one of {GEOMETRIES}, got {geometry!r}')
    if curvature <= 0:
        raise ValueError(f'curvature must be positive, got {curvature}')
    root = math.sqrt(curvature)
    scaled = points * root
    norm = scaled.norm(dim=1, keepdim=True)
    space = torch.sinh(norm) * scaled / norm.clamp_min(1e-300)
    lorentz = torch.cat([torch.cosh(norm), space], dim=1)
    return (lorentz_distances(lorentz, lorentz) / root).numpy()

# -------------------------------------------------------------------------------------------------
# Statistics
# -------------------------------------------------------------------------------------------------

def sector_separation(distances: np.ndarray, sector: np.ndarray) -> SectorSeparation:
    '''The Mann–Whitney AUC of cross-sector over same-sector distances, over unordered pairs.'''

    upper = np.triu_indices(len(sector), k=1)
    values = distances[upper]
    same = (sector[:, None] == sector[None, :])[upper]
    ranks = rankdata(values)
    n_cross, n_same = int((~same).sum()), int(same.sum())
    auc = (ranks[~same].sum() - n_cross * (n_cross + 1) / 2) / (n_cross * n_same)
    return SectorSeparation(auc=float(auc), same_sector_pairs=n_same, cross_sector_pairs=n_cross)

def within_sector_rank_correlation(
    distances: np.ndarray, target: np.ndarray, sector: np.ndarray, codes: Sequence[str]
) -> WithinSectorRankCorrelation:
    '''Per query, Spearman's correlation of distance with D* over its sector's other codes.'''

    by_query: Dict[int, float] = {}
    undefined = 0
    for query in range(len(sector)):
        candidates = np.flatnonzero((sector == sector[query]) & (np.arange(len(sector)) != query))
        if len(candidates) < 2:
            undefined += 1
            continue
        a = rankdata(distances[query, candidates])
        b = rankdata(target[query, candidates])
        if a.std() == 0 or b.std() == 0:
            undefined += 1
            continue
        by_query[query] = float(np.corrcoef(a, b)[0, 1])
    by_sector: Dict[str, float] = {}
    for row in sorted(set(sector.tolist())):
        values = [value for query, value in by_query.items() if sector[query] == row]
        if values:
            by_sector[codes[row]] = float(np.mean(values))
    return WithinSectorRankCorrelation(
        mean_over_queries=float(np.mean(list(by_query.values()))) if by_query else None,
        mean_over_sectors=float(np.mean(list(by_sector.values()))) if by_sector else None,
        by_sector=by_sector,
        queries=len(by_query) + undefined,
        undefined_queries=undefined,
    )

def _pessimistic_order(distances: np.ndarray, relevance: np.ndarray) -> np.ndarray:
    '''Candidates by distance, a tie broken against relevance (the less relevant first).'''

    return np.lexsort((relevance, distances))

def average_precision(distances: np.ndarray, relevant: np.ndarray) -> float:
    '''One query's average precision of its relevant candidates, ranked by distance.'''

    positions = np.flatnonzero(relevant[_pessimistic_order(distances, relevant)]) + 1
    return float(np.mean(np.arange(1, len(positions) + 1) / positions))

def ndcg_at(distances: np.ndarray, gains: np.ndarray, k: int) -> Optional[float]:
    '''One query's NDCG@k with linear gains; None when no candidate has a gain.'''

    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(gains)[::-1][:k]
    best = (ideal * discount[:len(ideal)]).sum()
    if best == 0:
        return None
    ranked = gains[_pessimistic_order(distances, gains)][:k]
    return float((ranked * discount[:len(ranked)]).sum() / best)

def _by_level(values: Dict[int, float], depth: np.ndarray) -> ByQueryLevel:
    levels: Dict[str, float] = {}
    for level in sorted({int(depth[query]) + 1 for query in values}):
        chosen = [value for query, value in values.items() if depth[query] + 1 == level]
        levels[str(level)] = float(np.mean(chosen))
    return ByQueryLevel(
        value=float(np.mean(list(values.values()))), queries=len(values), by_level=levels
    )

def map_over_ancestors(distances: np.ndarray, tree: Tree) -> ByQueryLevel:
    '''Per non-sector query, the average precision of its ancestors among all other codes.'''

    everything = np.arange(len(tree.codes))
    values: Dict[int, float] = {}
    for query in np.flatnonzero(tree.depth > 1):
        candidates = everything[everything != query]
        relevant = np.isin(candidates, tree.lineage[query, :tree.depth[query] - 1])
        values[int(query)] = average_precision(distances[query, candidates], relevant)
    return _by_level(values, tree.depth)

def ndcg(distances: np.ndarray, grades: np.ndarray, depth: np.ndarray, k: int) -> ByQueryLevel:
    '''
    Per query, NDCG@k over all other codes with integer LCA-depth gains; a query that shares a
    sector with no other code has no gain to rank and is not scored.
    '''

    everything = np.arange(len(depth))
    values: Dict[int, float] = {}
    for query in everything:
        candidates = everything[everything != query]
        value = ndcg_at(distances[query, candidates], grades[query, candidates].astype(float), k)
        if value is not None:
            values[int(query)] = value
    return _by_level(values, depth)

def distance_pearson(distances: np.ndarray, target: np.ndarray) -> DistancePearson:
    '''The Pearson correlation of distance with D* over all unordered pairs.'''

    upper = np.triu_indices(len(target), k=1)
    values = distances[upper]
    if values.std() == 0:
        return DistancePearson(value=None, pairs=len(values))
    value = np.corrcoef(values, target[upper].astype(np.float64))[0, 1]
    return DistancePearson(value=float(value), pairs=len(values))

def parent_retrieval(distances: np.ndarray, tree: Tree) -> ParentRetrieval:
    '''Per non-sector query outside the unary pairs, whether its parent is among its k nearest.'''

    unary = tree.unary_children()
    queries = np.flatnonzero((tree.depth > 1) & ~unary)
    parent = tree.parent[queries]
    others = distances[queries].copy()
    others[np.arange(len(queries)), queries] = np.inf
    to_parent = others[np.arange(len(queries)), parent]
    # A tie is broken against the parent: every code as near as it ranks ahead
    ranks = (others <= to_parent[:, None]).sum(axis=1)
    return ParentRetrieval(
        at={str(k): float((ranks <= k).mean())
            for k in PARENT_KS},
        queries=len(queries),
        unary_pairs_excluded=int(unary.sum()),
    )

# -------------------------------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------------------------------

def diagnostics_report(
    table: pl.DataFrame,
    geometry: Geometry,
    *,
    codebook_codes: Optional[Sequence[str]] = None,
    curvature: float = 1.0,
) -> DiagnosticsReport:
    '''
    Req 6's statistics for an arm's code table in the export form.

    Args:
        table: ``code`` plus coordinates (``panels.regressor.coordinate_matrix``).
        geometry: The arm's geometry, which sets its distance.
        codebook_codes: If given, the table must cover exactly these codes.
        curvature: A hyperbolic arm's curvature magnitude.

    Raises:
        ValueError: If the table's codes differ from ``codebook_codes``, or an ancestor of a code
            is missing.
    '''

    codes, matrix = coordinate_matrix(table)
    if codebook_codes is not None and set(codes) != set(codebook_codes):
        raise ValueError(
            f"the table has {len(codes):,} codes; the report covers the codebook's "
            f'{len(set(codebook_codes)):,}'
        )
    tree = Tree.from_codes(codes)
    distances = pairwise_distances(matrix, geometry, curvature)
    lca = tree.lca_depth()
    target = tree.depth[:, None] + tree.depth[None, :] - 2 * lca
    return DiagnosticsReport(
        codes=len(codes),
        geometry=geometry,
        curvature=curvature if geometry == 'hyperbolic' else None,
        sector_separation=sector_separation(distances, tree.sector),
        within_sector_rank_correlation=within_sector_rank_correlation(
            distances, target, tree.sector, codes
        ),
        map_over_ancestors=map_over_ancestors(distances, tree),
        ndcg={f'@{k}': ndcg(distances, lca, tree.depth, k)
              for k in NDCG_KS},
        distance_pearson=distance_pearson(distances, target),
        parent_retrieval=parent_retrieval(distances, tree),
    )
