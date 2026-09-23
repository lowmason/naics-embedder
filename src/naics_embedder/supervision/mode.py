'''
Explicit supervision-mode policy.

Repaired Stage-3 training is the default and requires a validated bundle. Legacy containment is an
explicit, tagged mode that may read old artifacts but permits only local, unmined contrastive
learning plus supervision-independent regularizers; it is not contract-compliant training.
'''

from dataclasses import dataclass

@dataclass(frozen=True)
class SupervisionModePolicy:
    '''What a supervision mode permits; consulted through explicit predicates, never inferred.'''

    name: str
    require_bundle: bool
    enable_structural_losses: bool
    enable_candidate_reordering: bool
    enable_pseudo_related: bool
    checkpoint_tag: str

    @classmethod
    def from_name(cls, name: str) -> 'SupervisionModePolicy':
        if name == 'repaired':
            return cls(
                name='repaired',
                require_bundle=True,
                enable_structural_losses=True,
                enable_candidate_reordering=True,
                enable_pseudo_related=True,
                checkpoint_tag='stage3-supervision-v1',
            )
        if name == 'legacy_containment':
            return cls(
                name='legacy_containment',
                require_bundle=False,
                enable_structural_losses=False,
                enable_candidate_reordering=False,
                enable_pseudo_related=False,
                checkpoint_tag='LEGACY-CONTAINMENT',
            )
        raise ValueError(f'unknown supervision mode {name!r}')
