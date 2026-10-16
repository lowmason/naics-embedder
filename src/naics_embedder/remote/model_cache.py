'''Transfer trusted Mac snapshots into a verified, repository-contained offline Hub cache.'''

import re
import shutil
import tempfile
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from naics_embedder.remote.config import relative_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.utils.config import Config

CACHE_PATH = '.remote/model-cache/hub'
MINILM = 'sentence-transformers/all-MiniLM-L6-v2'
MINILM_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'

def _cache_hashes(root: Path) -> dict[str, str]:
    '''Use the transport's anchored inventory; links and extra files cannot become model inputs.'''
    from naics_embedder.remote.worker import _inventory
    inventory = _inventory(root, CACHE_PATH)['files']
    if not inventory or any(item['kind'] != 'file' for item in inventory):
        raise ValueError('offline model cache must contain regular files only')
    return {CACHE_PATH + '/' + item['path']: item['sha256'] for item in inventory}

def verify_model_cache(root: Path, record: dict) -> dict:
    '''Refuse changed or incomplete cache bytes, including ref changes, before launch.'''
    if not isinstance(record, dict) or set(record) != {'revisions', 'hashes'}:
        raise ValueError('missing or malformed offline model cache record')
    revisions, hashes = record['revisions'], record['hashes']
    if not isinstance(revisions, dict) or not revisions or not isinstance(hashes, dict):
        raise ValueError('malformed offline model cache revisions or hashes')
    for name, revision in revisions.items():
        if not isinstance(name, str) or not re.fullmatch(r'[\w.-]+/[\w.-]+', name):
            raise ValueError('invalid offline model cache name')
        if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('invalid offline model cache revision')
        if name == MINILM and revision != MINILM_REVISION:
            raise ValueError('offline MiniLM cache revision differs from the reference pin')
        ref = CACHE_PATH + '/models--' + name.replace('/', '--') + '/refs/main'
        if ref not in hashes:
            raise ValueError('offline model cache needs its recorded main ref')
    for path, digest in hashes.items():
        if relative_path(root, path) != path or not path.startswith(CACHE_PATH + '/'):
            raise ValueError('offline model cache path escapes its fixed root')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid offline model cache hash')
    try:
        actual = _cache_hashes(root)
    except (OSError, ValueError) as error:
        raise ValueError(f'offline model cache inventory failed: {error}') from error
    if actual != hashes:
        raise ValueError('offline model cache inventory or SHA256 differs')
    for name, revision in revisions.items():
        ref = root / CACHE_PATH / ('models--' + name.replace('/', '--')) / 'refs/main'
        if ref.read_text().strip() != revision:
            raise ValueError('offline model cache main ref differs from recorded revision')
    if _cache_hashes(root) != actual:
        raise ValueError('offline model cache changed during verification')
    return record

def stage_model_cache(root: Path, cfg: Config, *, hub: Path | None = None) -> dict:
    '''Materialize cached snapshot blobs without downloading or repairing an existing cache.'''
    names = {
        cfg.model.base_model_name, cfg.data_loader.tokenization.tokenizer_name,
        cfg.data_loader.streaming.tokenizer_name
    }
    hub = Path(HF_HUB_CACHE if hub is None else hub).resolve()
    sources = {}
    revisions = {}
    for name in sorted(names):
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', name):
            raise ValueError('offline cache needs a Hub model name')
        model = hub / ('models--' + name.replace('/', '--'))
        ref = model / 'refs/main'
        if any(path.is_symlink() for path in (model, model / 'refs', model / 'snapshots')):
            raise ValueError('offline model cache roots cannot be links')
        if not ref.is_file() or ref.is_symlink():
            raise ValueError(f'offline model cache missing main ref: {name}')
        revision = ref.read_text().strip()
        if not re.fullmatch(r'[0-9a-f]{40}', revision) or (
            name == MINILM and revision != MINILM_REVISION
        ):
            raise ValueError(f'offline model cache revision differs from the pin: {name}')
        snapshot = model / 'snapshots' / revision
        if not snapshot.is_dir() or snapshot.is_symlink():
            raise ValueError(f'offline model cache missing snapshot: {name}')
        required = {'config.json'}
        if name == cfg.model.base_model_name:
            if any(
                (snapshot / file).exists() or (snapshot / file).is_symlink()
                for file in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
            ):
                raise ValueError(f'offline model cache indexed weights are unsupported: {name}')
            if not any(
                (snapshot / file).is_file() for file in ('model.safetensors', 'pytorch_model.bin')
            ):
                raise ValueError(f'offline model cache missing weights: {name}')
        if name in {
            cfg.data_loader.tokenization.tokenizer_name, cfg.data_loader.streaming.tokenizer_name
        }:
            required.add('tokenizer_config.json')
            if not any((snapshot / file).is_file() for file in ('tokenizer.json', 'vocab.txt')):
                raise ValueError(f'offline model cache missing tokenizer: {name}')
        if not all((snapshot / file).is_file() for file in required):
            raise ValueError(f'offline model cache snapshot is incomplete: {name}')
        relative_model = 'models--' + name.replace('/', '--')
        sources[relative_model + '/refs/main'] = ref
        from naics_embedder.remote.worker import _credential_name
        for path in snapshot.rglob('*'):
            if _credential_name(path.relative_to(snapshot).as_posix()):
                raise ValueError('credential path forbidden in offline model cache snapshot')
            if path.is_dir():
                if path.is_symlink():
                    raise ValueError('offline model cache snapshot directory is a link')
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(model.resolve()) or not resolved.is_file():
                raise ValueError('offline model cache snapshot link escapes its model')
            sources[relative_model + '/snapshots/' + revision + '/'
                    + path.relative_to(snapshot).as_posix()] = resolved
        revisions[name] = revision
    record = {
        'revisions': revisions,
        'hashes': {
            CACHE_PATH + '/' + name: sha256_file(path)
            for name, path in sources.items()
        }
    }
    destination = root / CACHE_PATH
    if destination.exists() or destination.is_symlink():
        verify_model_cache(root, record)
        return record
    relative_path(root, CACHE_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / 'hub'
        for name, source in sources.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256_file(target) != record['hashes'][CACHE_PATH + '/' + name]:
                raise ValueError('offline model cache source changed during staging')
        staging.rename(destination)
    verify_model_cache(root, record)
    return record

def verify_remote_model_cache(transport: object, record: dict) -> None:
    '''Compare the complete anchored remote inventory with the Mac's verified cache hashes.'''
    inventory = transport.probe('inventory', {'path': CACHE_PATH})['files']
    if any(item['kind'] != 'file' for item in inventory):
        raise ValueError('remote offline model cache contains links or nonregular files')
    actual = {CACHE_PATH + '/' + item['path']: item['sha256'] for item in inventory}
    if actual != record['hashes']:
        raise ValueError('remote offline model cache inventory or SHA256 differs')
