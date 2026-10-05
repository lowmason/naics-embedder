'''Immutable push records reconstruct tracked and untracked working-tree bytes.'''

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from naics_embedder.remote.code_manifest import code_entries
from naics_embedder.remote.provenance import write_push_record

def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(root), *args])

def test_full_working_tree_round_trip(remote_repo, tmp_path):
    root = remote_repo.root
    (root / 'binary').write_bytes(bytes(range(256)))
    (root / 'exec').write_text('old\n')
    (root / 'exec').chmod(0o755)
    git(root, 'add', '.')
    git(
        root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', 'commit', '-qm',
        'base'
    )
    (root / 'binary').write_bytes(b'\xff\0changed')
    (root / 'exec').write_text('staged\n')
    git(root, 'add', 'exec')
    (root / 'exec').write_text('unstaged\n')
    (root / 'src/tiny.py').unlink()
    (root / 'new name\nline').write_bytes(b'\0untracked')
    (root / 'new executable').write_text('run\n')
    (root / 'new executable').chmod(0o755)
    (root / 'new link').symlink_to('new executable')
    record = write_push_record(root, 'fixture@host', 'push-one', 10000000)
    clone = tmp_path / 'clone'
    subprocess.run(['git', 'clone', '-q', str(root), str(clone)], check=True)
    git(clone, 'checkout', '-q', record.head_sha)
    git(clone, 'apply', str(record.directory / 'uncommitted.patch'))
    with tarfile.open(record.directory / 'untracked.tar') as archive:
        for member in archive.getmembers():
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            if member.issym():
                assert (clone / member.name).parent.joinpath(member.linkname
                                                             ).resolve().is_relative_to(clone)
        archive.extractall(clone)
    assert code_entries(clone) == record.entries
    assert json.loads((record.directory / 'hashes.json').read_text()) == {
        item.path: item.sha256
        for item in record.entries
    }
    assert json.loads((record.directory / 'files.json').read_text()) == [
        {
            **item.__dict__
        } for item in record.entries
    ]
    provenance = json.loads((record.directory / 'provenance.json').read_text())
    assert provenance['host'] == 'fixture@host' and provenance['dirty'] is True
    assert provenance['file_count'] == len(record.entries)
    assert {item['path']
            for item in provenance['untracked']} == {
                'new name\nline', 'new executable', 'new link'
            }

def test_untracked_total_cap_and_exact_boundary(remote_repo):
    root = remote_repo.root
    (root / 'first').write_bytes(b'a' * 5000000)
    (root / 'second').write_bytes(b'b' * 5000000)
    record = write_push_record(root, 'host', 'at-cap', 10000000)
    assert record.directory.exists()
    (root / 'second').write_bytes(b'b' * 5000001)
    with pytest.raises(ValueError, match='first.*second'):
        write_push_record(root, 'host', 'too-large', 10000000)
    assert not (root / '.remote/pushes/too-large').exists()

def test_id_is_immutable(remote_repo):
    record = write_push_record(remote_repo.root, 'host', 'one', 10000000)
    before = (record.directory / 'provenance.json').read_bytes()
    with pytest.raises(ValueError, match='exists'):
        write_push_record(remote_repo.root, 'different', 'one', 10000000)
    assert (record.directory / 'provenance.json').read_bytes() == before

@pytest.mark.parametrize('identifier', ['', '..', '../escape', '/absolute', 'bad/part'])
def test_unsafe_record_id_refuses(remote_repo, identifier):
    with pytest.raises(ValueError):
        write_push_record(remote_repo.root, 'host', identifier, 10000000)

def test_failed_record_write_does_not_publish(remote_repo, monkeypatch):

    def fail(*args, **kwargs):
        raise OSError('fixture archive failure')

    monkeypatch.setattr(tarfile, 'open', fail)
    with pytest.raises(OSError, match='fixture'):
        write_push_record(remote_repo.root, 'host', 'failed', 10000000)
    assert not (remote_repo.root / '.remote/pushes/failed').exists()

def test_changing_source_during_record_does_not_publish(remote_repo, monkeypatch):
    original = tarfile.open

    def mutate(*args, **kwargs):
        (remote_repo.root / 'src/tiny.py').write_text('changed\n')
        return original(*args, **kwargs)

    monkeypatch.setattr(tarfile, 'open', mutate)
    with pytest.raises(ValueError, match='changed'):
        write_push_record(remote_repo.root, 'host', 'unstable', 10000000)
    assert not (remote_repo.root / '.remote/pushes/unstable').exists()

@pytest.mark.parametrize('directory_component', [False, True])
def test_ignored_intermediate_link_never_publishes(remote_repo, directory_component):
    root = remote_repo.root
    with (root / '.gitignore').open('a') as stream:
        stream.write('ignored-link\n')
    (root / 'ignored-link').symlink_to('src' if directory_component else 'src/tiny.py')
    (root / 'visible-link').symlink_to(
        'ignored-link/tiny.py' if directory_component else 'ignored-link'
    )
    with pytest.raises(ValueError, match='visible-link.*ignored-link'):
        write_push_record(root, 'host', 'omitted-chain', 10000000)
    assert not (root / '.remote/pushes/omitted-chain').exists()

@pytest.mark.parametrize('directory_component', [False, True])
def test_fully_included_link_chain_round_trip(remote_repo, tmp_path, directory_component):
    root = remote_repo.root
    (root / 'included-link').symlink_to('src' if directory_component else 'src/tiny.py')
    (root / 'visible-link').symlink_to(
        'included-link/tiny.py' if directory_component else 'included-link'
    )
    record = write_push_record(root, 'host', 'included-chain', 10000000)
    clone = tmp_path / 'clone'
    subprocess.run(['git', 'clone', '-q', str(root), str(clone)], check=True)
    git(clone, 'checkout', '-q', record.head_sha)
    with tarfile.open(record.directory / 'untracked.tar') as archive:
        for member in archive.getmembers():
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert (clone / member.name).parent.joinpath(member.linkname).resolve().is_relative_to(
                clone
            )
        archive.extractall(clone)
    assert (clone / 'visible-link').read_bytes() == b'VALUE = 1\n'
    assert code_entries(clone) == record.entries
