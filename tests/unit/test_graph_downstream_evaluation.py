'''
``GraphEmbeddingDataset`` reads a Polars embeddings frame into torch without handing torch
read-only memory, whatever the columns' layout.
'''

import importlib
import warnings

import numpy as np
import polars as pl
import pytest
import torch

from naics_embedder.metrics import GraphEmbeddingDataset

def _lorentz_points(spatial):
    tensor = torch.tensor(spatial, dtype=torch.float32)
    time = torch.sqrt(1.0 + torch.sum(tensor**2, dim=1, keepdim=True))
    return torch.cat([time, tensor], dim=1)

def _graph_fixture():
    codes = ['11111', '111110', '111111', '21111', '211110', '211111']
    levels = [5, 6, 6, 5, 6, 6]
    spatial = [
        [0.0, 0.0],
        [0.08, 0.02],
        [0.09, -0.02],
        [1.2, 0.0],
        [1.28, 0.02],
        [1.30, -0.02],
    ]
    embeddings = _lorentz_points(spatial)
    return GraphEmbeddingDataset(embeddings=embeddings, codes=codes, levels=levels)

# -------------------------------------------------------------------------------------------------
# Polars-to-torch embedding conversion
# -------------------------------------------------------------------------------------------------

EMBED_COLS = ['hgcn_e0', 'hgcn_e1', 'hgcn_e2']
NOT_WRITABLE_WARNING = 'The given NumPy array is not writable'

def _embedding_frame(*, contiguous: bool) -> pl.DataFrame:
    '''Build the fixture's embeddings frame with its hgcn_e* columns in one Fortran buffer.

    Polars' to_numpy() returns a read-only zero-copy view when the selected columns sit
    back-to-back in memory. After a join or parquet read that is up to the allocator; here it is
    fixed. contiguous=False stores the columns in reverse, so selecting them in numeric order
    always takes the copying path instead.
    '''
    dataset = _graph_fixture()
    values = dataset.embeddings.double().numpy()
    columns = EMBED_COLS
    if not contiguous:
        values, columns = values[:, ::-1], EMBED_COLS[::-1]
    embeddings = pl.DataFrame(np.asfortranarray(values), schema=columns, orient='row')
    frame = pl.DataFrame({'code': dataset.codes, 'level': dataset.levels}).hstack(embeddings)

    read_only = not frame.select(EMBED_COLS).to_numpy().flags.writeable
    assert read_only == contiguous, 'fixture no longer controls whether to_numpy() copies'
    return frame

@pytest.fixture
def torch_warns_always():
    '''Re-arm torch warnings that otherwise fire at most once per process.

    The not-writable warning is one, so an earlier test in the same process could trip it and
    leave the test below unable to see it.
    '''
    previous = torch.is_warn_always_enabled()
    torch.set_warn_always(True)
    yield
    torch.set_warn_always(previous)

@pytest.mark.usefixtures('torch_warns_always')
def test_embedding_conversion_never_hands_torch_read_only_memory():
    frame = _embedding_frame(contiguous=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message=NOT_WRITABLE_WARNING, category=UserWarning)
        GraphEmbeddingDataset.from_dataframe(frame)

def test_graph_dataset_values_do_not_depend_on_column_contiguity():
    contiguous = GraphEmbeddingDataset.from_dataframe(_embedding_frame(contiguous=True))
    split = GraphEmbeddingDataset.from_dataframe(_embedding_frame(contiguous=False))

    assert torch.equal(contiguous.embeddings, split.embeddings)

# -------------------------------------------------------------------------------------------------
# The removed benchmark and suite
# -------------------------------------------------------------------------------------------------

def test_the_qcew_benchmark_and_the_downstream_suite_are_gone():
    '''Roadmap Stage 4: neither ``metrics/qcew.py`` nor the taxonomy-tasks suite remains.'''

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('naics_embedder.metrics.qcew')
    modules = [
        importlib.import_module(name) for name in (
            'naics_embedder.metrics', 'naics_embedder.metrics.graph', 'naics_embedder.graph_model'
        )
    ]
    for name in (
        'GraphDownstreamEvaluator',
        'run_graph_downstream_suite',
        'QCEWBenchmarkConfig',
        'run_qcew_employment_benchmark',
    ):
        assert not any(hasattr(module, name) for module in modules), name
