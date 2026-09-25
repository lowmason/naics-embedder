'''
The artifact store and the record files: immutable references a decision record can name.
'''

import json

import polars as pl
import pytest

from naics_embedder.decision.records import ArmRecord, read_record, write_record
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

def test_a_record_is_written_once_and_reads_back_whole(store, tmp_path):
    arm = synthetic_arm(store, tmp_path, spec('A'), {})
    path = tmp_path / 'records' / 'A.json'

    write_record(arm, path)

    assert read_record(path, ArmRecord) == arm
    with pytest.raises(FileExistsError):
        write_record(arm, path)
