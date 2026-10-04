'''
Window-fitting summaries (roadmap Stage 6b): the pin and its identity (spec 4.6), the units
(4.2), the artifact (4.5) and the resolver (4.7).
'''

import subprocess
import sys
from pathlib import Path

import pytest

from naics_embedder.panels import window_summaries
from naics_embedder.panels.window_summaries import SummariesPin, summaries_identity

pytestmark = pytest.mark.unit

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'

# -------------------------------------------------------------------------------------------------
# The pin and its identity
# -------------------------------------------------------------------------------------------------

def test_the_identity_is_the_pins_sha256_and_none_without_a_pin(monkeypatch):
    pin = SummariesPin(path='summaries.csv', sha256='a' * 64, window=16)
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, 'tiny-backbone', pin)

    assert summaries_identity('tiny-backbone') == 'a' * 64
    assert summaries_identity('unpinned/backbone') is None

def test_the_seam_pins_minilm_alone_to_a_pin_no_test_can_read():
    '''tests/conftest.py's autouse seam (spec section 6).'''

    assert list(window_summaries.WINDOW_SUMMARIES) == [MINILM]
    pin = window_summaries.WINDOW_SUMMARIES[MINILM]
    assert summaries_identity(MINILM) == pin.sha256
    assert pin.sha256 is not None
    assert pin.window == 128
    assert not Path(pin.path).exists()

@pytest.mark.real_window_summaries
def test_the_marker_leaves_the_committed_pins_alone():
    # Every committed pin names its artifact; the seam's dummy names a file that does not exist
    for pin in window_summaries.WINDOW_SUMMARIES.values():
        assert Path(pin.path).is_file()

def test_the_module_imports_no_torch():
    '''The resolver runs in the token cache and the text-only builder; it loads no model.'''

    imported = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import naics_embedder.panels.window_summaries; '
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert imported.stdout.strip() == 'False'
