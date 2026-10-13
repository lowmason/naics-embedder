'''
Req 5's rule on paired resamples, over a decision's panels.

- **Non-inferior** on a panel: the lower bound of the 95 % interval on Δ exceeds −δ.
- **Superior** on a panel: the interval at 1 − 0.05/n lies above zero, for a decision over n
  panels, each of which gets an even share of the 5 % error rate (Req 24). D8's three panels read
  98⅓ %, which supersedes the 97.5 % that Req 5 names for two; the economic decision's two growth
  panels read 97.5 %.
- **Adopted:** non-inferior on every panel and superior on at least one.
- **Several arms:** the survivors are the arms no other arm is adopted over; when every arm is,
  dominance cycles and the tie order picks among all of them. The pairwise comparisons are not
  corrected for multiplicity: the tie order toward the simpler arm is the guard (Req 5).
- **Tie order:** fewer components, then lower dimension, then non-hyperbolic geometry, then the
  higher held-out gain over ancestors (D11).
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import List, Mapping, Sequence, Tuple

import numpy as np

from naics_embedder.decision.records import ArmSpec, Comparison, PanelComparison
from naics_embedder.decision.resampling import percentile_interval

NONINFERIORITY_LEVEL = 0.95
# The error rate superiority splits evenly over a decision's panels (Req 24)
SUPERIORITY_ERROR = 0.05

class TieUnresolvedError(RuntimeError):
    '''Two candidate arms tie on every key of the tie order.'''

# -------------------------------------------------------------------------------------------------
# Comparisons
# -------------------------------------------------------------------------------------------------

def superiority_level(panels: int) -> float:
    '''
    The superiority interval's level for a decision over ``panels`` panels: 1 − 0.05/n (Req 24).

    Raises:
        ValueError: If ``panels`` is below 1.
    '''

    if panels < 1:
        raise ValueError(f'a decision reads at least one panel, not {panels}')
    return 1.0 - SUPERIORITY_ERROR / panels

def compare_panel(
    panel: str,
    delta: float,
    replicates: np.ndarray,
    margin: float,
    *,
    superiority: float,
) -> PanelComparison:
    '''
    One panel's verdict for A against B.

    Args:
        panel: The panel's name.
        delta: The point estimate of Δ, positive when it favours A.
        replicates: Δ on each paired resample, oriented the same way.
        margin: The panel's δ.
        superiority: The superiority interval's level (``superiority_level``).
    '''

    noninferiority = percentile_interval(replicates, NONINFERIORITY_LEVEL)
    upper = percentile_interval(replicates, superiority)
    return PanelComparison(
        panel=panel,
        delta=float(delta),
        noninferiority_interval=noninferiority,
        superiority_interval=upper,
        margin=float(margin),
        non_inferior=noninferiority[0] > -margin,
        superior=upper[0] > 0.0,
    )

def compare(a: str, b: str, panels: Sequence[PanelComparison]) -> Comparison:
    '''A against B on every panel.'''

    adopted = all(panel.non_inferior
                  for panel in panels) and any(panel.superior for panel in panels)
    return Comparison(a=a, b=b, panels=list(panels), adopted=adopted)

def non_dominated(arms: Sequence[str], comparisons: Sequence[Comparison]) -> Tuple[List[str], bool]:
    '''
    The arms no other arm is adopted over, and whether dominance cycled.

    Returns:
        ``(survivors, cycle)``: when every arm is dominated, ``cycle`` is true and every arm
        survives, for the tie order to pick among (Req 5).
    '''

    dominated = {comparison.b for comparison in comparisons if comparison.adopted}
    survivors = [arm for arm in arms if arm not in dominated]
    if survivors:
        return survivors, False
    return list(arms), True

def tie_order(specs: Sequence[ArmSpec], heldout_gain: Mapping[str, float]) -> List[str]:
    '''
    The arms from the one that stands first.

    Raises:
        TieUnresolvedError: If two arms tie on every key.
    '''

    def key(spec: ArmSpec) -> Tuple[int, int, bool, float]:
        return (
            spec.components,
            spec.dimension,
            spec.geometry == 'hyperbolic',
            -float(heldout_gain[spec.name]),
        )

    ordered = sorted(specs, key=key)
    for first, second in zip(ordered, ordered[1:]):
        if key(first) == key(second):
            raise TieUnresolvedError(
                f'{first.name} and {second.name} tie on components, dimension, geometry and the '
                'held-out gain'
            )
    return [spec.name for spec in ordered]
