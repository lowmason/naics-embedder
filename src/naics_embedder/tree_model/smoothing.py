'''
Lineage smoothing of code vectors (spec economic-candidates, 4.2: ``smoothed-dN``).

A code's vector is replaced by a weighted mean over its lineage, from its sector down to the code
(``utils.naics_hierarchy.code_lineage``). The weights halve at each step up the tree and are
fixed: the code weighs 1, its parent ½, its grandparent ¼, and so on, normalized to sum 1.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Dict, Sequence

import numpy as np

from naics_embedder.utils.naics_hierarchy import code_lineage

# -------------------------------------------------------------------------------------------------
# Weights
# -------------------------------------------------------------------------------------------------

def lineage_weights(depth: int) -> np.ndarray:
    '''
    The weights of a lineage of ``depth`` codes, from the sector down to the code.

    The code weighs 1, its parent ½, its grandparent ¼, and so on, normalized to sum 1: for a
    depth of 3, (1/7, 2/7, 4/7).

    Raises:
        ValueError: If ``depth`` is below 1.
    '''

    if depth < 1:
        raise ValueError(f'a lineage has a depth of at least 1, not {depth}')
    weights = 0.5**np.arange(depth - 1, -1, -1, dtype=np.float64)
    return weights / weights.sum()

# -------------------------------------------------------------------------------------------------
# Smoothing
# -------------------------------------------------------------------------------------------------

def smooth_over_lineage(codes: Sequence[str], vectors: np.ndarray) -> np.ndarray:
    '''
    Replace each code's vector by the weighted mean over its lineage's vectors.

    Args:
        codes: The codes, one per row of ``vectors``, in any order.
        vectors: One row per code.

    Returns:
        The smoothed (len(codes), width) float64 vectors, in the order of ``codes`` and in C
        order whatever the layout of ``vectors``: a PCA of them then does not depend on it. A
        sector, whose lineage is itself, keeps its vector.

    Raises:
        ValueError: If ``codes`` and ``vectors`` differ in length, a code repeats, or a code's
            ancestor is not among ``codes``.
    '''

    vectors = np.ascontiguousarray(vectors, dtype=np.float64)
    if len(codes) != vectors.shape[0]:
        raise ValueError(f'{len(codes):,} codes but {vectors.shape[0]:,} rows of vectors')
    position: Dict[str, int] = {code: row for row, code in enumerate(codes)}
    if len(position) != len(codes):
        raise ValueError('the codes repeat a code')
    smoothed = np.empty(vectors.shape, dtype=np.float64)
    for row, code in enumerate(codes):
        lineage = code_lineage(code)
        missing = [ancestor for ancestor in lineage if ancestor not in position]
        if missing:
            raise ValueError(f'the code {code} has an ancestor not among the codes: {missing[0]}')
        smoothed[row] = lineage_weights(len(lineage)) @ vectors[[position[a] for a in lineage]]
    return smoothed
