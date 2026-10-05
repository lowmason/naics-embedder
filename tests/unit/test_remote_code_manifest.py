'''Code manifests preserve Git's complete working tree and classify remote edits.'''

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from naics_embedder.remote.code_manifest import code_entries, deletion_set, instance_edits
from naics_embedder.remote.session import FileEntry

def entry(path: str, kind: str = 'file', mode: int = 0o644) -> FileEntry:
    return FileEntry(path, '1' * 64, 1, mode, kind, None)

def test_deleted_code_only_is_removed_from_a_push():
    assert deletion_set((entry('src/old.py'), ), (entry('src/new.py'), )) == ('src/old.py', )

@pytest.mark.parametrize(
    'name', [
        '.env', '.env.local', '.ssh/id_rsa', '.aws/config', 'id_ed25519', 'key.pem', 'secret.key',
        'key.p12'
    ]
)
def test_credential_file_refuses_the_whole_push(remote_repo, name):
    path = remote_repo.root / name
    path.parent.mkdir(exist_ok=True)
    path.write_text('fixture=value\n')
    with pytest.raises(ValueError, match=name.replace('.', r'\.')):
        code_entries(remote_repo.root)

def test_credentials_are_refused_before_any_file_contents_are_read(remote_repo, monkeypatch):
    (remote_repo.root / '.env').write_text('fixture=value\n')

    def refuse_read(*args, **kwargs):
        raise AssertionError('contents read before credential preflight')

    monkeypatch.setattr(Path, 'open', refuse_read)
    with pytest.raises(ValueError, match=r'\.env'):
        code_entries(remote_repo.root)

@pytest.mark.parametrize('target', ['../outside', 'missing', '/tmp/outside'])
def test_external_and_broken_links_refuse(remote_repo, target):
    (remote_repo.root / 'link').symlink_to(target)
    with pytest.raises(ValueError, match='link'):
        code_entries(remote_repo.root)

def test_hash_modes_links_deleted_and_staged_paths(remote_repo):
    root = remote_repo.root
    (root / 'src/tiny.py').unlink()
    (root / 'new name\nline').write_bytes(b'\0\xff')
    (root / 'exec').write_bytes(b'run')
    (root / 'exec').chmod(0o755)
    subprocess.run(['git', '-C', str(root), 'add', 'exec'], check=True)
    entries = {item.path: item for item in code_entries(root)}
    assert 'src/tiny.py' not in entries
    assert entries['exec'].mode == 0o755
    assert entries['new name\nline'].sha256 == hashlib.sha256(b'\0\xff').hexdigest()
    assert entries['AGENTS.md'].target == 'CLAUDE.md'
    assert entries['AGENTS.md'].sha256 == hashlib.sha256(b'CLAUDE.md').hexdigest()

def test_special_files_refuse(remote_repo):
    os.mkfifo(remote_repo.root / 'pipe')
    with pytest.raises(ValueError, match='pipe'):
        code_entries(remote_repo.root)

def test_submodules_refuse(remote_repo):
    root = remote_repo.root
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip()
    subprocess.run(
        ['git', '-C',
         str(root), 'update-index', '--add', '--cacheinfo', f'160000,{sha},module'],
        check=True
    )
    with pytest.raises(ValueError, match='module'):
        code_entries(root)

def test_instance_edits_all_categories_and_previously_pushed_runtime_paths():
    expected = (entry('src/change.py'), entry('src/delete.py'), entry('outputs/tracked'))
    actual = (
        entry('src/change.py', 'symlink'), entry('src/unexpected.py'),
        entry('outputs/tracked', 'directory'), entry('data/new'), entry('src/a.pyc'),
        entry('nested/__pycache__/x'), entry('.aws/new')
    )
    assert instance_edits(expected, actual, ('*.pyc', '__pycache__/')) == {
        'modified': ('outputs/tracked', 'src/change.py'),
        'deleted': ('src/delete.py', ),
        'new': ('src/unexpected.py', )
    }

def test_expected_paths_are_checked_even_if_ignored():
    assert instance_edits((entry('src/x.pyc'), ), (), ('*.pyc', ))['deleted'] == ('src/x.pyc', )

def test_deleted_tracked_credentials_are_refused_before_patch_capture(remote_repo):
    root = remote_repo.root
    (root / '.env').write_text('fixture=value\n')
    subprocess.run(['git', '-C', str(root), 'add', '.env'], check=True)
    subprocess.run(
        [
            'git', '-C',
            str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', 'commit',
            '-qm', 'credential'
        ],
        check=True
    )
    subprocess.run(['git', '-C', str(root), 'rm', '-q', '.env'], check=True)
    with pytest.raises(ValueError, match=r'\.env'):
        code_entries(root)

def test_symlink_to_ignored_credential_refuses_before_hashing(remote_repo, monkeypatch):
    root = remote_repo.root
    (root / '.env').write_text('fixture=value\n')
    with (root / '.gitignore').open('a') as stream:
        stream.write('.env\n')
    (root / 'zlink').symlink_to('.env')

    def refuse_read(*args, **kwargs):
        raise AssertionError('contents read before link credential preflight')

    monkeypatch.setattr(Path, 'open', refuse_read)
    with pytest.raises(ValueError, match='zlink'):
        code_entries(root)

def test_symlink_cycle_refuses_with_named_value_error(remote_repo):
    (remote_repo.root / 'cycle').symlink_to('cycle')
    with pytest.raises(ValueError, match='cycle'):
        code_entries(remote_repo.root)

def test_ignored_special_file_is_runtime(remote_repo):
    os.mkfifo(remote_repo.root / 'data/pipe')
    assert 'data/pipe' not in {item.path for item in code_entries(remote_repo.root)}

@pytest.mark.parametrize('path', ['../source.py', '/source.py', 'src//file', 'src/./file'])
def test_instance_scan_refuses_unvalidated_paths(path):
    with pytest.raises(ValueError):
        instance_edits((), (entry(path), ), ())

def test_link_target_must_exist_in_the_reconstructible_file_set(remote_repo):
    (remote_repo.root / 'link').symlink_to('data/naics_descriptions.parquet')
    with pytest.raises(ValueError, match='link'):
        code_entries(remote_repo.root)

def test_safe_directory_link_is_preserved(remote_repo):
    (remote_repo.root / 'source-link').symlink_to('src')
    assert next(
        item for item in code_entries(remote_repo.root) if item.path == 'source-link'
    ).target == 'src'
