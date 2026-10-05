'''
Shared pytest fixtures for NAICS Embedder test suite.

This module provides reusable fixtures for testing hyperbolic embeddings,
data processing, and model components.
'''

import logging
from pathlib import Path
from typing import Iterator, Optional

import polars as pl
import pytest
import torch

from naics_embedder.panels import window_summaries

pytest_plugins = (
    'tests.fixtures.naics_sources',
    'tests.fixtures.regressor_panel',
    'tests.fixtures.shared_encoder',
    'tests.fixtures.supervision',
    'tests.fixtures.checkpoint_runs',
)

# -------------------------------------------------------------------------------------------------
# Test Configuration
# -------------------------------------------------------------------------------------------------

@pytest.fixture(scope='session')
def test_device():
    '''Get device for testing (CPU for CI compatibility).'''

    return torch.device('cpu')

@pytest.fixture(scope='session')
def random_seed():
    '''Fixed random seed for reproducibility.'''

    return 42

@pytest.fixture(autouse=True)
def set_random_seeds(random_seed):
    '''Automatically set random seeds before each test.'''

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

# -------------------------------------------------------------------------------------------------
# Window summaries: the dummy pin (Stage 6b spec, section 6)
# -------------------------------------------------------------------------------------------------

MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
# A pin no test can read: its file does not exist. Fixture texts fit their windows, so the resolver
# never looks for it, while every identity site records its sha256.
DUMMY_SUMMARIES_PIN = window_summaries.SummariesPin(
    path='/nonexistent/window_summaries.csv', sha256='5' * 64, window=128
)

def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'real_window_summaries: reads the committed window-summaries pins; the dummy-pin seam '
        'stays off',
    )

@pytest.fixture(autouse=True)
def dummy_window_summaries(request, monkeypatch):
    '''
    ``WINDOW_SUMMARIES`` holds one entry, MiniLM's dummy pin, unless the test is marked
    ``real_window_summaries``.

    The dict is changed in place, and every identity site reads it at call time, so each site
    records the dummy's sha256. A test that needs no pin deletes the entry with
    ``monkeypatch.delitem`` or names an unpinned backbone.
    '''

    if request.node.get_closest_marker('real_window_summaries') is not None:
        return
    for backbone in list(window_summaries.WINDOW_SUMMARIES):
        monkeypatch.delitem(window_summaries.WINDOW_SUMMARIES, backbone)
    monkeypatch.setitem(window_summaries.WINDOW_SUMMARIES, MINILM, DUMMY_SUMMARIES_PIN)

# -------------------------------------------------------------------------------------------------
# The repository's selection log: no test writes it
# -------------------------------------------------------------------------------------------------

# The shipped configs log every read to ./logs/selection_log.jsonl, relative to the working
# directory, which is the repository root when the suite runs. A test that reads a panel, builds a
# training monitor, trains or sweeps points the log at tmp_path.
REPOSITORY_SELECTION_LOG = Path(__file__).resolve().parents[1] / 'logs' / 'selection_log.jsonl'

def selection_log_size(path: Path) -> Optional[int]:
    '''The log's size in bytes, or None if it does not exist.'''

    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None

def _described_size(size: Optional[int]) -> str:
    return 'absent' if size is None else f'{size} bytes'

def refuse_a_changed_selection_log(path: Path, size: Optional[int]) -> None:
    '''
    Fail if the log is no longer the size it was (``size``, None if it did not exist).

    The log is append-only, so any write changes its size.
    '''

    now = selection_log_size(path)
    if now != size:
        pytest.fail(
            f'the suite wrote {path} (its size went from {_described_size(size)} to '
            f'{_described_size(now)}): a test that logs a read points its log at tmp_path',
            pytrace=False,
        )

@pytest.fixture(scope='session', autouse=True)
def guard_the_repository_selection_log() -> Iterator[Path]:
    '''
    Record the repository's selection log at the session's start, and fail the session at its end
    if the log changed. It only reads the log's size: it never creates or writes the file.
    '''

    size = selection_log_size(REPOSITORY_SELECTION_LOG)
    yield REPOSITORY_SELECTION_LOG
    refuse_a_changed_selection_log(REPOSITORY_SELECTION_LOG, size)

# -------------------------------------------------------------------------------------------------
# Hyperbolic Geometry Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_tangent_vectors(test_device, random_seed):
    '''Generate sample tangent vectors for testing exponential map.

    Vectors are normalized to have reasonable norms (~1-2) to avoid numerical
    overflow in sinh/cosh computations during exp_map.
    '''

    torch.manual_seed(random_seed)
    batch_size = 16
    dim = 384
    # Create random tangent vectors with time component = 0 (proper tangent at origin)
    tangent = torch.randn(batch_size, dim + 1, device=test_device)
    tangent[:, 0] = 0.0  # Time component should be 0 for tangent at origin
    # Scale to have reasonable norms for numerical stability
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0
    return tangent

@pytest.fixture
def sample_lorentz_embeddings(test_device, random_seed):
    '''Generate valid Lorentz embeddings for testing.

    Uses scaled tangent vectors to ensure numerical stability in exp_map.
    '''

    from naics_embedder.text_model.hyperbolic import LorentzOps

    torch.manual_seed(random_seed)
    batch_size = 16
    dim = 384
    # Create tangent vectors with reasonable norms for numerical stability
    tangent = torch.randn(batch_size, dim + 1, device=test_device)
    tangent[:, 0] = 0.0  # Time component should be 0 for tangent at origin
    # Scale to have norm ~2 (avoids sinh/cosh overflow)
    tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0
    return LorentzOps.exp_map_zero(tangent, c=1.0)

@pytest.fixture(params=[0.1, 0.5, 1.0, 5.0, 10.0])
def curvature_values(request):
    '''Parametrize tests across different curvature values.'''

    return request.param

# -------------------------------------------------------------------------------------------------
# Data Processing Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def sample_naics_data(tmp_path):
    '''Create minimal NAICS data for testing.'''

    data = {
        'index': list(range(15)),
        'code': [
            '31',
            '311',
            '3111',
            '31111',
            '311111',  # Manufacturing - Food
            '32',
            '321',
            '3211',
            '32111',
            '321111',  # Manufacturing - Wood
            '44',
            '441',
            '4411',
            '44111',
            '441111',  # Retail - Motor vehicles
        ],
        'level': [2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6],
        'title': [
            'Manufacturing',
            'Food Manufacturing',
            'Animal Food Manufacturing',
            'Animal Food Manufacturing',
            'Dog and Cat Food Manufacturing',
            'Manufacturing',
            'Wood Product Manufacturing',
            'Sawmills and Wood Preservation',
            'Sawmills and Wood Preservation',
            'Sawmills',
            'Retail Trade',
            'Motor Vehicle and Parts Dealers',
            'Automobile Dealers',
            'New Car Dealers',
            'New Car Dealers',
        ],
    }

    df = pl.DataFrame(data)
    parquet_path = tmp_path / 'naics_test.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

@pytest.fixture
def sample_naics_relations(tmp_path, sample_naics_data):
    '''Create sample NAICS relationship data.'''

    relations_data = {
        'idx_i': [0, 1, 2, 3, 5, 6, 7, 10, 11, 12],
        'idx_j': [1, 2, 3, 4, 6, 7, 8, 11, 12, 13],
        'code_i': ['31', '311', '3111', '31111', '32', '321', '3211', '44', '441', '4411'],
        'code_j': [
            '311',
            '3111',
            '31111',
            '311111',
            '321',
            '3211',
            '32111',
            '441',
            '4411',
            '44111',
        ],
        'relation': ['child'] * 10,
        'relation_id': [1] * 10,
    }

    df = pl.DataFrame(relations_data)
    parquet_path = tmp_path / 'naics_relations_test.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

@pytest.fixture
def sample_naics_distances(tmp_path):
    '''Create sample NAICS distance data.'''

    distances_data = {
        'idx_i': [0, 0, 0, 1, 1, 2],
        'idx_j': [1, 2, 3, 2, 3, 3],
        'code_i': ['31', '31', '31', '311', '311', '3111'],
        'code_j': ['311', '3111', '31111', '3111', '31111', '31111'],
        'distance': [0.5, 1.5, 2.5, 0.5, 1.5, 0.5],
    }

    df = pl.DataFrame(distances_data)
    parquet_path = tmp_path / 'naics_distances_test.parquet'
    df.write_parquet(parquet_path)

    return str(parquet_path)

# -------------------------------------------------------------------------------------------------
# Model Component Fixtures
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def batch_size():
    '''Standard batch size for testing.'''

    return 16

@pytest.fixture
def num_channels():
    '''Number of text channels (title, description, examples, exclusions).'''

    return 4

@pytest.fixture
def sample_batch(batch_size, num_channels, test_device, random_seed):
    '''Generate sample batch of multi-channel embeddings.'''

    torch.manual_seed(random_seed)
    return {
        'title': torch.randn(batch_size, 384, device=test_device),
        'description': torch.randn(batch_size, 384, device=test_device),
        'examples': torch.randn(batch_size, 384, device=test_device),
        'exclusions': torch.randn(batch_size, 384, device=test_device),
    }

# -------------------------------------------------------------------------------------------------
# Logging Configuration for Tests
# -------------------------------------------------------------------------------------------------

@pytest.fixture(scope='session', autouse=True)
def configure_logging():
    '''Configure logging for test runs.'''

    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(levelname)s - %(name)s - %(message)s',
    )

    # Suppress specific noisy loggers
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)

# -------------------------------------------------------------------------------------------------
# Temporary Directory Helpers
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    '''Create temporary checkpoint directory.'''

    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    return checkpoint_dir

@pytest.fixture
def temp_data_dir(tmp_path):
    '''Create temporary data directory.'''

    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def temp_config_dir(tmp_path):
    '''Create temporary config directory.'''

    config_dir = tmp_path / 'conf'
    config_dir.mkdir()
    return config_dir

@pytest.fixture
def structural_distance_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]],
        dtype=torch.float64,
    )
    return prediction, target

@pytest.fixture
def structural_lorentz_embeddings() -> torch.Tensor:
    spatial = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.0, 0.3], [0.2, 0.4]])
    time = torch.sqrt(1.0 + spatial.square().sum(dim=1, keepdim=True))
    return torch.cat([time, spatial], dim=1)
