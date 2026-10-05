import types

import pytest
import torch

from naics_embedder.utils import backend

def _stub_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2, raising=False)
    monkeypatch.setattr(torch.backends, 'mps', types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(torch.version, 'cuda', '12.1')

@pytest.mark.unit
def test_get_device_prefers_cuda(monkeypatch):
    _stub_cuda(monkeypatch)

    device, precision, num = backend.get_device()

    assert device == 'cuda'
    # The shipped training precision (spec 4.2, R9)
    assert precision == 'bf16-mixed'
    assert num == 2

@pytest.mark.unit
@pytest.mark.parametrize('cuda_precision', ['32', '16-mixed', 'bf16-mixed'])
def test_get_device_returns_the_cuda_precision_it_is_given(monkeypatch, cuda_precision):
    '''On CUDA the precision is the caller's (train passes training.trainer.precision, spec 4.2).'''

    _stub_cuda(monkeypatch)

    device, precision, _ = backend.get_device(cuda_precision=cuda_precision)

    assert (device, precision) == ('cuda', cuda_precision)

@pytest.mark.unit
@pytest.mark.parametrize('mps', [True, False], ids=['mps', 'cpu'])
def test_get_device_runs_32_true_off_cuda_whatever_the_cuda_precision(monkeypatch, mps):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)
    monkeypatch.setattr(torch.backends, 'mps', types.SimpleNamespace(is_available=lambda: mps))

    device, precision, _ = backend.get_device(cuda_precision='16-mixed')

    assert (device, precision) == ('mps' if mps else 'cpu', '32-true')

@pytest.mark.unit
def test_get_device_uses_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)
    monkeypatch.setattr(torch.backends, 'mps', types.SimpleNamespace(is_available=lambda: True))

    device, precision, num = backend.get_device()

    assert device == 'mps'
    assert precision == '32-true'
    assert num == 1

@pytest.mark.unit
def test_get_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False, raising=False)
    monkeypatch.setattr(torch.backends, 'mps', types.SimpleNamespace(is_available=lambda: False))

    device, precision, num = backend.get_device()

    assert device == 'cpu'
    assert precision == '32-true'
    assert num == 0
