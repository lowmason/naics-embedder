'''
The artifact store and the record files: immutable references a decision record can name.
'''

import json
import shutil

import polars as pl
import pytest
from pydantic import BaseModel

from naics_embedder.decision.records import ArmRecord, ArtifactRef, read_record, write_record
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.panels.regressor import table_fingerprint
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.decision import CODES, spec, synthetic_arm, write_text_only
from tests.fixtures.regressor_panel import coordinate_table

pytestmark = pytest.mark.unit

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

def test_a_stored_file_outlives_its_source_and_is_verified_on_every_read(store, tmp_path):
    source = tmp_path / 'checkpoint.ckpt'
    source.write_bytes(b'weights')

    reference = store.put(source)
    again = store.put(source)
    source.unlink()

    assert reference == again
    assert reference.sha256 == sha256_file(store.resolve(reference))
    assert reference.path == f'objects/{reference.sha256[:2]}/{reference.sha256}/checkpoint.ckpt'
    assert reference.bytes == len(b'weights')
    store.resolve(reference).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed since it was stored'):
        store.resolve(reference)

def test_the_root_is_absolute_whatever_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert ArtifactStore('store').root == (tmp_path / 'store').resolve()

@pytest.mark.parametrize('leaving', ['../outside.bin', 'absolute'])
def test_a_reference_that_leaves_the_store_is_refused(store, tmp_path, leaving):
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'not in the store')
    store.root.mkdir(parents=True)
    path = str(outside) if leaving == 'absolute' else leaving
    reference = ArtifactRef(path=path, sha256=sha256_file(outside), bytes=outside.stat().st_size)

    with pytest.raises(ValueError, match='leaves the artifact store'):
        store.resolve(reference)

@pytest.mark.parametrize('name', ['nested/scores.parquet', '..', ''])
def test_a_frame_is_stored_under_a_plain_file_name(store, name):
    with pytest.raises(ValueError, match='not a file name'):
        store.put_frame(pl.DataFrame({'value': [1.0]}), name)

def _staged(store):
    '''Staging files left anywhere under the store root.'''

    return list(store.root.rglob('.incoming-*'))

def test_a_source_that_changes_mid_copy_is_never_stored(store, tmp_path, monkeypatch):
    source = tmp_path / 'checkpoint.ckpt'
    source.write_bytes(b'weights')
    hashed = sha256_file(source)
    copy = shutil.copyfile

    def copy_after_a_rewrite(src, dst):
        # Another process rewrites the source after the store hashed it
        source.write_bytes(b'new weights')
        return copy(src, dst)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, 'copyfile', copy_after_a_rewrite)
        with pytest.raises(ValueError, match='changed while it was copied'):
            store.put(source)

    assert not (store.root / 'objects' / hashed[:2] / hashed / source.name).exists()
    assert _staged(store) == []
    # The source, stable again, is stored
    assert store.put(source).sha256 == sha256_file(source)

def test_a_failed_copy_leaves_no_staging_file(store, tmp_path, monkeypatch):
    source = tmp_path / 'checkpoint.ckpt'
    source.write_bytes(b'weights')

    def fail(src, dst):
        raise OSError('the disk is full')

    monkeypatch.setattr(shutil, 'copyfile', fail)
    with pytest.raises(OSError, match='the disk is full'):
        store.put(source)

    assert _staged(store) == []

def test_a_table_is_stored_with_the_fingerprint_the_log_names_it_by(store, tmp_path):
    table = coordinate_table(CODES, dimension=4)
    path = tmp_path / 'arm.parquet'
    table.write_parquet(path)

    reference = store.put_table(path)

    assert reference.matrix_fingerprint == table_fingerprint(table)

def test_a_text_only_table_is_stored_with_its_provenance(store, tmp_path):
    path = write_text_only(tmp_path / 'text')

    reference = store.put_text_only(path)

    provenance = json.loads(provenance_path(path).read_text())
    assert reference.table.sha256 == provenance['table_sha256']
    assert reference.table.matrix_fingerprint == provenance['matrix_fingerprint']
    assert reference.provenance.sha256 == sha256_file(provenance_path(path))
    assert (reference.backbone, reference.revision, reference.max_length) == (
        provenance['backbone'], provenance['revision'], provenance['max_length']
    )

def test_a_text_only_table_needs_the_provenance_that_describes_it(store, tmp_path):
    path = write_text_only(tmp_path / 'text')
    provenance = json.loads(provenance_path(path).read_text())

    provenance_path(path).write_text(json.dumps({**provenance, 'matrix_fingerprint': 'other'}))
    with pytest.raises(ValueError, match='another matrix_fingerprint'):
        store.put_text_only(path)
    pl.read_parquet(path).head(3).write_parquet(path)
    with pytest.raises(ValueError, match='describes another file'):
        store.put_text_only(path)
    provenance_path(path).unlink()
    with pytest.raises(FileNotFoundError, match='no provenance'):
        store.put_text_only(path)

def _shortened(path, provenance):
    '''The table rewritten, so the provenance describes another file.'''

    pl.read_parquet(path).head(3).write_parquet(path)

def _without_max_length(path, provenance):
    provenance.pop('max_length')
    provenance_path(path).write_text(json.dumps(provenance))

@pytest.mark.parametrize(
    'edit, message',
    [(_shortened, 'describes another file'), (_without_max_length, "lacks the field 'max_length'")],
)
def test_nothing_is_stored_until_the_provenance_describes_the_table(store, tmp_path, edit, message):
    path = write_text_only(tmp_path / 'text')
    edit(path, json.loads(provenance_path(path).read_text()))

    with pytest.raises(ValueError, match=message):
        store.put_text_only(path)
    assert not (store.root / 'objects').exists()

def test_a_record_is_written_once_and_reads_back_whole(store, tmp_path):
    arm = synthetic_arm(store, tmp_path, spec('A'), {})
    path = tmp_path / 'records' / 'A.json'

    write_record(arm, path)

    assert read_record(path, ArmRecord) == arm
    with pytest.raises(FileExistsError):
        write_record(arm, path)

class _Note(BaseModel):
    '''A stand-in record.'''

    text: str

def test_a_writer_racing_for_the_path_is_never_overwritten(tmp_path):
    path = tmp_path / 'records' / 'note.json'

    class Racing(_Note):

        def model_dump_json(self, **kwargs):
            # Another writer creates the record while this one serializes
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('the other writer')
            return super().model_dump_json(**kwargs)

    with pytest.raises(FileExistsError, match='written once'):
        write_record(Racing(text='mine'), path)
    assert path.read_text() == 'the other writer'

def test_a_write_that_fails_leaves_the_path_free(tmp_path):
    path = tmp_path / 'records' / 'note.json'

    class Unwritable(_Note):

        def model_dump_json(self, **kwargs):
            # A lone surrogate serializes but cannot be encoded as UTF-8, so the write fails
            return '{"text": "\udc80"}'

    with pytest.raises(UnicodeEncodeError):
        write_record(Unwritable(text='mine'), path)
    assert not path.exists()
    write_record(_Note(text='retried'), path)
    assert read_record(path, _Note).text == 'retried'
