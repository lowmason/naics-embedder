from pathlib import Path

import httpx
import pytest

from naics_embedder.utils.config import DirConfig
from naics_embedder.utils.utilities import (
    download_with_retry,
    make_directories,
    setup_directory,
    sorted_embedding_columns,
)

@pytest.mark.unit
def test_make_directories_creates_all(tmp_path):
    cfg = DirConfig(
        checkpoint_dir=str(tmp_path / 'ckpts'),
        conf_dir=str(tmp_path / 'conf'),
        data_dir=str(tmp_path / 'data'),
        docs_dir=str(tmp_path / 'docs'),
        log_dir=str(tmp_path / 'logs'),
        output_dir=str(tmp_path / 'outputs'),
    )

    make_directories(cfg)

    for path in cfg.model_dump().values():
        assert Path(path).exists()

@pytest.mark.unit
def test_download_with_retry_succeeds_after_retry(monkeypatch):
    attempts = {'count': 0}

    class DummyResponse:

        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    def fake_get(url, timeout):
        attempts['count'] += 1
        if attempts['count'] == 1:
            raise httpx.HTTPError('boom')
        return DummyResponse(b'data')

    monkeypatch.setattr('naics_embedder.utils.utilities.httpx.get', fake_get)
    monkeypatch.setattr('naics_embedder.utils.utilities.time.sleep', lambda *_: None)

    data = download_with_retry('https://example.com', max_retries=1)

    assert data == b'data'
    assert attempts['count'] == 2

@pytest.mark.unit
def test_download_with_retry_raises_after_exhaustion(monkeypatch):

    def boom(*_args, **_kwargs):
        raise httpx.HTTPError('boom')

    monkeypatch.setattr('naics_embedder.utils.utilities.httpx.get', boom)
    monkeypatch.setattr('naics_embedder.utils.utilities.time.sleep', lambda *_: None)

    with pytest.raises(httpx.HTTPError):
        download_with_retry('https://example.com', max_retries=0)

@pytest.mark.unit
def test_setup_directory_creates_path(tmp_path):
    target = tmp_path / 'new_dir'

    path = setup_directory(str(target))

    assert path.exists()

@pytest.mark.unit
def test_sorted_embedding_columns_orders_numeric_suffixes_as_integers():
    numeric_order = [f'hyp_e{i}' for i in range(12)]
    string_order = sorted(numeric_order)  # hyp_e0, hyp_e1, hyp_e10, hyp_e11, hyp_e2, ...
    assert string_order != numeric_order

    assert sorted_embedding_columns(string_order, 'hyp_e') == numeric_order

@pytest.mark.unit
def test_sorted_embedding_columns_puts_non_numeric_suffixes_last():
    # The bare prefix has an empty suffix, which sorts before any digit as a plain string
    columns = ['hyp_e_norm', 'hyp_e10', 'hyp_e', 'hyp_e2']
    expected = ['hyp_e2', 'hyp_e10', 'hyp_e', 'hyp_e_norm']

    assert sorted_embedding_columns(columns, 'hyp_e') == expected

@pytest.mark.unit
def test_sorted_embedding_columns_keeps_only_prefixed_columns():
    columns = ['index', 'level', 'code', 'hyp_e1', 'hgcn_e0', 'hyp_e0']

    assert sorted_embedding_columns(columns, 'hyp_e') == ['hyp_e0', 'hyp_e1']
    assert sorted_embedding_columns(columns, 'hgcn_e') == ['hgcn_e0']
    assert sorted_embedding_columns(['index', 'level', 'code'], 'hyp_e') == []
