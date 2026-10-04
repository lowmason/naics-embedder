'''
Window-fitting summaries of over-long channel texts (Req 9, "Input windows"; roadmap Stage 6b).

``WINDOW_SUMMARIES`` pins, per backbone, the committed artifact of extractive summaries, and
``summaries_identity`` is the sha256 every identity site records: the token cache's sidecar, the
checkpoint contract, the export and text-only provenances, and the decision store.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, Optional

# -------------------------------------------------------------------------------------------------
# The pin
# -------------------------------------------------------------------------------------------------

WINDOW_SUMMARIES_PATH = 'conf/data/window_summaries.csv'

@dataclass(frozen=True)
class SummariesPin:
    '''
    A committed summaries artifact.

    Attributes:
        path: The artifact, relative to the repository root.
        sha256: The artifact's sha256, which every identity site records.
        window: The trained window the summaries fit.
    '''

    path: str
    sha256: str
    window: int

# Keyed by backbone. Plan 9's Exit adds MiniLM's entry together with the artifact it pins.
WINDOW_SUMMARIES: Dict[str, SummariesPin] = {}

def summaries_identity(backbone: str) -> Optional[str]:
    '''
    The sha256 of the backbone's pinned summaries, or None when it has no pin.

    Args:
        backbone: A Hugging Face model or tokenizer name.

    Returns:
        The pin's sha256, or None.
    '''

    pin = WINDOW_SUMMARIES.get(backbone)
    return None if pin is None else pin.sha256
