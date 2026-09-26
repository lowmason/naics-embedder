from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

SECTOR_CODE_LENGTH = 2
UNARY_PARENT_LENGTH = 5
# Combined sectors 31-33, 44-45, and 48-49 are keyed by their first code, as in compute_relations
COMBINED_SECTOR_KEYS = {'32': '31', '33': '31', '45': '44', '49': '48'}

def naics_parent_code(code: str) -> Optional[str]:
    '''The code's parent in the NAICS tree, or None for a sector.'''

    if len(code) <= SECTOR_CODE_LENGTH:
        return None
    if len(code) == SECTOR_CODE_LENGTH + 1:
        sector = code[:SECTOR_CODE_LENGTH]
        return COMBINED_SECTOR_KEYS.get(sector, sector)
    return code[:-1]

@lru_cache(maxsize=None)
def code_lineage(code: str) -> Tuple[str, ...]:
    '''The code's ancestors from its sector down to the code itself.'''

    chain = [code]
    parent = naics_parent_code(code)
    while parent is not None:
        chain.append(parent)
        parent = naics_parent_code(parent)
    return tuple(reversed(chain))

def tree_distance_matrix(codes: Sequence[str]) -> np.ndarray:
    '''
    D* between every two codes (Req 7): the tree path length through a virtual root.

    A code's depth is the length of its lineage, 1 for a sector, and D* is
    ``depth_i + depth_j - 2 depth_LCA`` with the virtual root at depth 0. Pairs across sectors
    therefore get λ(i) + λ(j) − 2, where λ is the number of digits. Combined sectors (31-33,
    44-45, 48-49) count as one. Only the code strings are read, so ancestors need not be among
    ``codes``.

    Returns:
        ``(len(codes), len(codes))`` int64 matrix, zero on the diagonal.
    '''

    lineages = [code_lineage(code) for code in codes]
    if not lineages:
        return np.zeros((0, 0), dtype=np.int64)
    ids: Dict[str, int] = {}
    lineage = np.full((len(lineages), max(map(len, lineages))), -1, dtype=np.int64)
    for row, chain in enumerate(lineages):
        for depth, ancestor in enumerate(chain):
            lineage[row, depth] = ids.setdefault(ancestor, len(ids))
    depth = (lineage >= 0).sum(axis=1)
    shared = np.zeros((len(lineages), len(lineages)), dtype=np.int64)
    for column in lineage.T:
        shared += (column[:, None] == column[None, :]) & (column[:, None] >= 0)
    return depth[:, None] + depth[None, :] - 2 * shared

def unary_pairs(codes: Iterable[str]) -> List[Tuple[str, str]]:
    '''
    The unary pairs among ``codes`` (Req 9): a five-digit code and its only six-digit child.

    Returns:
        ``(parent, child)`` pairs sorted by parent.
    '''

    present = set(codes)
    children: Dict[str, List[str]] = defaultdict(list)
    for code in present:
        parent = naics_parent_code(code)
        if parent is not None and len(parent) == UNARY_PARENT_LENGTH and parent in present:
            children[parent].append(code)
    return sorted((parent, kids[0]) for parent, kids in children.items() if len(kids) == 1)

class HierarchyIntegrityError(ValueError):
    '''A relations file lacks a NAICS parent link between two codes it contains.'''

class NaicsHierarchy:
    '''In-memory representation of the NAICS hierarchy derived from relations parquet data.'''

    def __init__(self, parent_child_pairs: Sequence[Tuple[str, str]]):
        self.parent_by_child: Dict[str, str] = {}
        self.children_by_parent = defaultdict(list)
        self._parent_child_pairs: List[Tuple[str, str]] = []

        seen_pairs = set()
        for parent, child in parent_child_pairs:
            if not parent or not child:
                continue
            key = (parent, child)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            # Keep the first observed parent for a child to avoid conflicting mappings.
            if child not in self.parent_by_child:
                self.parent_by_child[child] = parent
                self.children_by_parent[parent].append(child)
                self._parent_child_pairs.append(key)

    @classmethod
    def from_relations_parquet(cls, relations_path: Path) -> 'NaicsHierarchy':
        '''
        Build a hierarchy object from the relations parquet.

        Expects columns `code_i`, `code_j`, and either `relation_id` or `relation`/`relationship`.

        Raises:
            HierarchyIntegrityError: If the file lacks a parent link between two codes it contains.
        '''
        if not relations_path.exists():
            raise FileNotFoundError(f'NAICS relations parquet not found: {relations_path}')

        df = pl.read_parquet(relations_path)
        if 'code_i' not in df.columns or 'code_j' not in df.columns:
            raise ValueError('relations parquet must contain code_i and code_j columns')

        relation_expr = None
        if 'relation_id' in df.columns:
            relation_expr = pl.col('relation_id') == 1
        elif 'relation' in df.columns:
            relation_expr = pl.col('relation') == 'child'
        elif 'relationship' in df.columns:
            relation_expr = pl.col('relationship') == 'child'
        else:
            raise ValueError(
                'relations parquet must contain either relation_id or relation/relationship columns'
            )

        parent_child_pairs: List[Tuple[str, str]] = []
        for row in df.filter(relation_expr).select('code_i', 'code_j').iter_rows(named=True):
            parent = row['code_i']
            child = row['code_j']
            parent_child_pairs.append((parent, child))

        hierarchy = cls(parent_child_pairs)
        codes = set(df.get_column('code_i').unique()) | set(df.get_column('code_j').unique())
        missing = hierarchy.missing_parent_links(codes)
        if missing:
            shown = ', '.join(f'{parent}->{child}' for parent, child in missing[:5])
            raise HierarchyIntegrityError(
                f'{relations_path} lacks NAICS parent links between codes it contains '
                f'({len(missing):,} missing, e.g. {shown}). Legacy relations files label some '
                "parent/child pairs 'excluded'; read relations from a supervision bundle instead."
            )
        return hierarchy

    def get_parent(self, code: str) -> Optional[str]:
        return self.parent_by_child.get(code)

    def get_children(self, code: str) -> List[str]:
        return list(self.children_by_parent.get(code, []))

    def get_siblings(self, code: str) -> List[str]:
        parent = self.get_parent(code)
        if parent is None:
            return []
        return [sibling for sibling in self.children_by_parent.get(parent, []) if sibling != code]

    def missing_parent_links(self, codes: Iterable[str]) -> List[Tuple[str, str]]:
        '''
        NAICS (parent, child) links between two of ``codes`` that this hierarchy lacks, sorted.

        A code whose parent is absent from ``codes`` is a root of a partial tree, not a gap.
        '''
        present = set(codes)
        missing = set()
        for code in present:
            parent = naics_parent_code(code)
            if parent in present and self.parent_by_child.get(code) != parent:
                missing.add((parent, code))
        return sorted(missing)

    @property
    def parent_child_pairs(self) -> List[Tuple[str, str]]:
        return list(self._parent_child_pairs)

@lru_cache(maxsize=4)
def load_naics_hierarchy(relations_path: str) -> NaicsHierarchy:
    '''Load (and cache) a NAICS hierarchy from the relations parquet file.'''

    path = Path(relations_path).expanduser().resolve()
    return NaicsHierarchy.from_relations_parquet(path)
