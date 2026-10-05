'''Native BF16 and fake executable bootstrap qualification.'''

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from naics_embedder.remote.worker import gpu_evidence

BOOTSTRAP = Path(__file__).parents[2] / 'src/naics_embedder/remote/bootstrap.sh'

class FakeCuda:

    def __init__(self, available=True, native=True):
        self.available = available
        self.native = native
        self.selected = []
        self.calls = []

    def is_available(self):
        return self.available

    @contextmanager
    def device(self, index):
        self.selected.append(index)
        yield

    def is_bf16_supported(self, **kwargs):
        self.calls.append(kwargs)
        return self.native

    def get_device_properties(self, index):
        assert index == 0
        return SimpleNamespace(name='device zero', major=8, minor=0, total_memory=123456)

def test_native_bf16_selects_zero_and_captures_evidence(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7,3')
    cuda = FakeCuda()
    evidence = gpu_evidence(SimpleNamespace(cuda=cuda))
    assert cuda.selected == [0] and cuda.calls == [{'including_emulation': False}]
    assert evidence.logical_index == 0 and evidence.cuda_visible_devices == '7,3'
    assert evidence.name == 'device zero' and evidence.total_memory_bytes == 123456
    assert evidence.compute_capability == (8, 0) and evidence.native_bf16

@pytest.mark.parametrize('available,native', [(False, True), (True, False)])
def test_unqualified_gpu_refuses(available, native):
    with pytest.raises(RuntimeError, match='native BF16'):
        gpu_evidence(SimpleNamespace(cuda=FakeCuda(available, native)))

def test_missing_api_and_property_exception_refuse():
    cuda = FakeCuda()
    cuda.is_bf16_supported = None
    with pytest.raises(RuntimeError, match='native BF16'):
        gpu_evidence(SimpleNamespace(cuda=cuda))
    cuda = FakeCuda()
    cuda.get_device_properties = lambda index: (_ for _ in ()).throw(ValueError('driver'))
    with pytest.raises(RuntimeError, match='native BF16'):
        gpu_evidence(SimpleNamespace(cuda=cuda))

@pytest.fixture
def fake_bootstrap(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'uv.lock').write_text('locked')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    log = tmp_path / 'calls'

    def executable(name, body):
        path = bin_dir / name
        path.write_text('#!/bin/bash\n' + body + '\n')
        path.chmod(0o755)
        return path

    executable('sha256sum', 'echo fixed-lock-hash uv.lock')
    executable('tmux', 'exit 0')
    executable('rsync', 'echo "rsync  version 3.5.1  protocol version 32"')
    executable('timedatectl', 'echo "${FAKE_NTP:-yes}"')
    executable('sudo', 'echo "$*" >> "$FAKE_LOG"; exit "${FAKE_SUDO_EXIT:-0}"')
    executable(
        'uv', 'echo "$*" >> "$FAKE_LOG"; if [[ "$1" == sync ]]; then '
        'exit "${FAKE_SYNC_EXIT:-0}"; fi; echo "{\\"repo\\":\\"fake\\",'
        '\\"accelerator\\":\\"cuda\\",\\"gpu_evidence\\":{\\"native_bf16\\":true}}"'
    )
    executable('curl', 'echo download >> "$FAKE_LOG"; exit 9')
    env = dict(os.environ, PATH=f'{bin_dir}:/usr/bin:/bin', HOME=str(tmp_path), FAKE_LOG=str(log))
    return repo, bin_dir, env, log

def test_prepared_bootstrap_json_and_locked_sync(fake_bootstrap):
    repo, _, env, log = fake_bootstrap
    result = subprocess.run(['bash', str(BOOTSTRAP), str(repo)], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout)['accelerator'] == 'cuda'
    assert 'sync --locked' in log.read_text() and 'apt-get' not in log.read_text()
    assert b'sync' in result.stderr

@pytest.mark.parametrize('setting', [{'FAKE_SYNC_EXIT': '7'}, {'FAKE_NTP': 'no'}])
def test_bootstrap_refuses_failed_sync_or_ntp(fake_bootstrap, setting):
    repo, _, env, _ = fake_bootstrap
    result = subprocess.run(
        ['bash', str(BOOTSTRAP), str(repo)], env=dict(env, **setting), capture_output=True
    )
    assert result.returncode != 0 and result.stderr

@pytest.mark.parametrize('sudo_exit', ['1', '100'])
def test_missing_tmux_reports_apt_failure(fake_bootstrap, sudo_exit):
    repo, bin_dir, env, log = fake_bootstrap
    (bin_dir / 'tmux').unlink()
    result = subprocess.run(
        ['bash', str(BOOTSTRAP), str(repo)],
        env=dict(env, FAKE_SUDO_EXIT=sudo_exit),
        capture_output=True
    )
    assert result.returncode != 0 and 'apt-get' in log.read_text()

def test_missing_uv_installer_failure_retains_step_output(fake_bootstrap):
    repo, bin_dir, env, log = fake_bootstrap
    (bin_dir / 'uv').unlink()
    result = subprocess.run(['bash', str(BOOTSTRAP), str(repo)], env=env, capture_output=True)
    assert result.returncode == 9
    assert 'download' in log.read_text() and b'official standalone uv' in result.stderr

def test_missing_uv_installs_to_absolute_home_path(fake_bootstrap):
    repo, bin_dir, env, log = fake_bootstrap
    uv_content = (bin_dir / 'uv').read_text()
    (bin_dir / 'uv').unlink()
    installer = repo / 'fake-installer'
    installer.write_text(
        '#!/bin/sh\n[ "$UV_NO_MODIFY_PATH" = 1 ] || exit 20\n'
        'mkdir -p "$UV_INSTALL_DIR"\ncat > "$UV_INSTALL_DIR/uv" <<\'UV\'\n' + uv_content
        + '\nUV\nchmod +x "$UV_INSTALL_DIR/uv"\n'
    )
    (bin_dir / 'curl').write_text(
        '#!/bin/bash\necho download >> "$FAKE_LOG"\n'
        'cp "$FAKE_INSTALLER" "${@: -1}"\n'
    )
    env['FAKE_INSTALLER'] = str(installer)
    result = subprocess.run(['bash', str(BOOTSTRAP), str(repo)], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    assert b'/.local/bin/uv' in result.stderr
    assert 'download' in log.read_text()

@pytest.mark.parametrize('failure', ['gpu', 'ntp-unavailable', 'lock-change'])
def test_bootstrap_late_refusals(fake_bootstrap, failure):
    repo, bin_dir, env, _ = fake_bootstrap
    if failure == 'gpu':
        uv = bin_dir / 'uv'
        uv.write_text(
            '#!/bin/bash\nif [[ "$1" == sync ]]; then exit 0; fi\n'
            'echo "CUDA native BF16 qualification failed" >&2\nexit 12\n'
        )
    elif failure == 'ntp-unavailable':
        (bin_dir / 'timedatectl').unlink()
    else:
        uv = bin_dir / 'uv'
        uv.write_text('#!/bin/bash\necho changed >> uv.lock\n')
        checksum = bin_dir / 'sha256sum'
        checksum.write_text('#!/bin/bash\n/usr/bin/shasum -a 256 "$1"\n')
    result = subprocess.run(['bash', str(BOOTSTRAP), str(repo)], env=env, capture_output=True)
    assert result.returncode != 0 and result.stderr

class TwoDeviceCuda(FakeCuda):

    def __init__(self, native_by_device):
        super().__init__()
        self.native_by_device = native_by_device
        self.current = None

    @contextmanager
    def device(self, index):
        self.selected.append(index)
        self.current = index
        try:
            yield
        finally:
            self.current = None

    def is_bf16_supported(self, **kwargs):
        self.calls.append((self.current, kwargs))
        return self.native_by_device[self.current]

@pytest.mark.parametrize('native_by_device,passes', [((False, True), False), ((True, False), True)])
def test_two_device_capability_always_qualifies_logical_zero(native_by_device, passes):
    cuda = TwoDeviceCuda(native_by_device)
    if passes:
        evidence = gpu_evidence(SimpleNamespace(cuda=cuda))
        assert evidence.logical_index == 0 and evidence.native_bf16
    else:
        with pytest.raises(RuntimeError, match='native BF16'):
            gpu_evidence(SimpleNamespace(cuda=cuda))
    assert cuda.selected == [0]
    assert cuda.calls == [(0, {'including_emulation': False})]
