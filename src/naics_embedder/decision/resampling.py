'''
The paired two-stage bootstrap (Req 5: "both arms are scored on the same resample of the
evaluation unit, with seeds nested within the resample").

Each replicate draws the panel's units with replacement, one draw shared by every arm, and then,
inside it, each arm's seeds with replacement, a draw of the arm's own. The replicate's statistic
is the mean over the drawn seeds of the item-weighted mean over the drawn units:

    sum_s m_s (C . S_s) / (n_seeds (C . N))

with C the unit counts, m the seed counts, S_s seed s's per-unit sums and N the items per unit.
Unit counts come from the generator seeded ``[bootstrap_seed, stream(panel)]`` and an arm's seed
counts from ``[bootstrap_seed, stream(panel), stream(arm)]``, so an arm's replicates are the same
in every comparison it enters.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

import hashlib
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import polars as pl

# -------------------------------------------------------------------------------------------------
# Items and units
# -------------------------------------------------------------------------------------------------

def stream(name: str) -> int:
    '''A generator stream for a name: the first 16 hex digits of its sha256, as an integer.'''

    return int(hashlib.sha256(name.encode('utf-8')).hexdigest()[:16], 16)

@dataclass(frozen=True)
class PanelItems:
    '''One statistic's items in a fixed order (by unit, then item), with each item's unit.'''

    items: Tuple[str, ...]
    units: Tuple[str, ...]
    unit_index: np.ndarray

    @classmethod
    def from_values(cls, frame: pl.DataFrame) -> 'PanelItems':
        '''
        The items of a frame with ``unit`` and ``item`` columns.

        Raises:
            ValueError: If an item repeats.
        '''

        if frame.get_column('item').is_duplicated().any():
            raise ValueError('an item repeats')
        ordered = frame.select('unit', 'item').sort('unit', 'item')
        units = tuple(ordered.get_column('unit').unique(maintain_order=True).to_list())
        position = {unit: index for index, unit in enumerate(units)}
        unit_index = np.array(
            [position[unit] for unit in ordered.get_column('unit').to_list()], dtype=np.int64
        )
        return cls(tuple(ordered.get_column('item').to_list()), units, unit_index)

    @property
    def sizes(self) -> np.ndarray:
        '''Items per unit, as floats.'''

        return np.bincount(self.unit_index, minlength=len(self.units)).astype(np.float64)

    def values(self, frame: pl.DataFrame) -> np.ndarray:
        '''
        The frame's ``value`` column in this order.

        Raises:
            ValueError: If the frame holds other items, or puts an item in another unit: paired
                arms must be scored on the same items (Req 5).
        '''

        if frame.height != len(self.items):
            raise ValueError(f'{frame.height:,} items, expected {len(self.items):,}')
        ordered = frame.sort('unit', 'item')
        if tuple(ordered.get_column('item').to_list()) != self.items:
            raise ValueError('the items differ: paired arms must be scored on the same items')
        expected = [self.units[index] for index in self.unit_index]
        if ordered.get_column('unit').to_list() != expected:
            raise ValueError('an item sits in another unit: paired arms must share units')
        return ordered.get_column('value').to_numpy().astype(np.float64)

    def sums(self, values: np.ndarray) -> np.ndarray:
        '''Per-unit sums of (seeds, items) values: (seeds, units).'''

        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        return np.stack(
            [
                np.bincount(self.unit_index, weights=row, minlength=len(self.units))
                for row in values
            ]
        )

# -------------------------------------------------------------------------------------------------
# Draws and replicates
# -------------------------------------------------------------------------------------------------

def unit_draws(panel: str, n_units: int, replicates: int, bootstrap_seed: int) -> np.ndarray:
    '''Unit counts per replicate, (replicates, n_units), shared by every arm on the panel.'''

    rng = np.random.default_rng([bootstrap_seed, stream(panel)])
    return rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=replicates)

def seed_draws(
    panel: str, arm: str, n_seeds: int, replicates: int, bootstrap_seed: int
) -> np.ndarray:
    '''One arm's seed counts per replicate, (replicates, n_seeds).'''

    rng = np.random.default_rng([bootstrap_seed, stream(panel), stream(arm)])
    return rng.multinomial(n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=replicates)

def replicate_statistics(
    sums: np.ndarray, sizes: np.ndarray, units: np.ndarray, seeds: np.ndarray
) -> np.ndarray:
    '''
    Each replicate's statistic.

    Args:
        sums: Per-seed unit sums, (n_seeds, n_units).
        sizes: Items per unit, (n_units,).
        units: Unit counts, (replicates, n_units).
        seeds: Seed counts, (replicates, n_seeds).
    '''

    per_seed = units @ sums.T
    items = units @ sizes
    return (seeds * per_seed).sum(axis=1) / (seeds.sum(axis=1) * items)

def point_statistic(sums: np.ndarray, sizes: np.ndarray) -> float:
    '''The mean over seeds of each seed's statistic on every unit.'''

    return float((sums.sum(axis=1) / sizes.sum()).mean())

def percentile_interval(replicates: Sequence[float], level: float) -> Tuple[float, float]:
    '''The two-sided percentile interval at ``level``.'''

    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(np.asarray(replicates, dtype=np.float64), [tail, 1.0 - tail])
    return float(lower), float(upper)
