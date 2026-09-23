# Stable, dependency-light exports only. Runtime types live in submodules (artifacts, index,
# candidates, selection, checkpoints) so importing configuration never pulls in torch.
from naics_embedder.supervision.schema import (
    CONTRACT_VERSION,
    MINING_CONTRACT_VERSION,
    STRUCTURAL_PREFERENCE_LOSS_VERSION,
    SelectionReason,
    SemanticSource,
    SemanticTarget,
)

__all__ = [
    'CONTRACT_VERSION',
    'MINING_CONTRACT_VERSION',
    'STRUCTURAL_PREFERENCE_LOSS_VERSION',
    'SemanticSource',
    'SemanticTarget',
    'SelectionReason',
]
