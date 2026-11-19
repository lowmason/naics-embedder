'''
The ITA evidence pin and its C1 check (plan 20, Task 1).

``conf/data/ita_evidence.yaml`` copies embedding-data's protocol Section 1 Evidence table: the run,
the protocol commit and three SHA-256 pins. ``check_ita_run`` refuses tables that are not the
pinned run's bytes before any read. The fixture's run directory is written under ``tmp_path``.
'''

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from naics_embedder.panels.ita_evidence import (
    OUTPUTS_FILE,
    ItaEvidence,
    check_ita_run,
    load_ita_evidence,
)
from naics_embedder.supervision.artifacts import sha256_file
from tests.fixtures.ita import (
    PAIR_CODES_FILE,
    PAIRS_FILE,
    ita_run_dir,
    ita_tables,
    write_ita_tables,
)

ROOT = Path(__file__).resolve().parents[2]  # the repository, whatever the working directory
EVIDENCE_FILE = ROOT / 'conf/data/ita_evidence.yaml'

# -------------------------------------------------------------------------------------------------
# The committed pin
# -------------------------------------------------------------------------------------------------

def test_the_committed_pin_is_the_protocols():
    evidence = load_ita_evidence(EVIDENCE_FILE)
    assert evidence.protocol_commit == '63c06664bdcce3867e2ec76c2607683de1c4320d'
    assert evidence.run == 'stage10_ita_establishment_20261116_r1'
    assert evidence.outputs_sha256 == (
        'e19b0d58f61f106c3994aa6fe51f4fb71244188145dd90f8cccaa8dc8a8b09df'
    )
    assert evidence.pairs_sha256 == (
        '60aad7642961e70b987e9d3222d660147861458e12ac52d71d2739d2d21e1e1c'
    )
    assert evidence.pair_codes_sha256 == (
        'c8084e4ad297e3f78391e035ebe51119b07982639ae4daf47f1d47caeb1f8916'
    )

def test_the_committed_pin_holds_no_local_path():
    assert '/Users/' not in EVIDENCE_FILE.read_text(encoding='utf-8')

def test_the_evidence_is_frozen():
    evidence = load_ita_evidence(EVIDENCE_FILE)
    with pytest.raises(ValidationError):
        evidence.run = 'stage10_ita_establishment_20261116_r1_replay'

# -------------------------------------------------------------------------------------------------
# The check
# -------------------------------------------------------------------------------------------------

def test_matching_tables_pass(tmp_path):
    pairs, codes, evidence = ita_run_dir(tmp_path, *ita_tables())
    check_ita_run(pairs, codes, evidence)  # no error
    check_ita_run(str(pairs), str(codes), evidence)  # string paths too

def test_the_fixture_run_directory_is_named_for_the_run(tmp_path):
    pairs, codes, evidence = ita_run_dir(tmp_path, *ita_tables(), run='stage10_other_r2')
    assert pairs == tmp_path / 'stage10_other_r2' / PAIRS_FILE
    assert codes == tmp_path / 'stage10_other_r2' / PAIR_CODES_FILE
    assert evidence.run == 'stage10_other_r2'
    assert evidence.outputs_sha256 == sha256_file(tmp_path / 'stage10_other_r2' / OUTPUTS_FILE)
    assert evidence.pairs_sha256 == sha256_file(pairs)
    assert evidence.pair_codes_sha256 == sha256_file(codes)

@pytest.mark.parametrize('table', ['pairs', 'codes'])
def test_a_table_of_other_bytes_is_refused(tmp_path, table):
    pairs_frame, codes_frame = ita_tables()
    pairs, codes, evidence = ita_run_dir(tmp_path, pairs_frame, codes_frame)
    # The same rows in reverse order: the same values, other bytes
    if table == 'pairs':
        pairs_frame.reverse().write_parquet(pairs)
        changed, pin = pairs, evidence.pairs_sha256
    else:
        codes_frame.reverse().write_parquet(codes)
        changed, pin = codes, evidence.pair_codes_sha256
    with pytest.raises(ValueError, match='C1') as refusal:
        check_ita_run(pairs, codes, evidence)
    message = str(refusal.value)
    assert changed.name in message
    assert sha256_file(changed) in message
    assert pin in message

def test_tables_of_another_run_are_refused(tmp_path):
    pairs, codes, evidence = ita_run_dir(tmp_path, *ita_tables())
    replay = tmp_path / 'stage10_fixture_r1_replay'
    shutil.copytree(pairs.parent, replay)
    with pytest.raises(ValueError, match='run'):
        check_ita_run(replay / PAIRS_FILE, replay / PAIR_CODES_FILE, evidence)

def test_tables_in_two_directories_are_refused(tmp_path):
    pairs, codes, evidence = ita_run_dir(tmp_path, *ita_tables())
    elsewhere = tmp_path / 'elsewhere' / evidence.run
    elsewhere.mkdir(parents=True)
    moved = Path(shutil.copy2(codes, elsewhere / PAIR_CODES_FILE))
    shutil.copy2(pairs.parent / OUTPUTS_FILE, elsewhere / OUTPUTS_FILE)
    with pytest.raises(ValueError, match='one directory'):
        check_ita_run(pairs, moved, evidence)

def test_a_missing_outputs_file_is_refused(tmp_path):
    pairs, codes, evidence = ita_run_dir(tmp_path, *ita_tables())
    (pairs.parent / OUTPUTS_FILE).unlink()
    with pytest.raises(ValueError, match=OUTPUTS_FILE):
        check_ita_run(pairs, codes, evidence)

def test_an_other_outputs_file_is_refused(tmp_path):
    pairs, codes, evidence = ita_run_dir(tmp_path, *ita_tables())
    outputs = pairs.parent / OUTPUTS_FILE
    outputs.write_text('{"run": "another"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='C1') as refusal:
        check_ita_run(pairs, codes, evidence)
    message = str(refusal.value)
    assert OUTPUTS_FILE in message
    assert sha256_file(outputs) in message
    assert evidence.outputs_sha256 in message

@pytest.mark.parametrize('missing', [PAIRS_FILE, PAIR_CODES_FILE])
def test_a_missing_table_is_refused(tmp_path, missing):
    pairs, codes, evidence = ita_run_dir(tmp_path, *ita_tables())
    (pairs.parent / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        check_ita_run(pairs, codes, evidence)

def test_tables_written_without_a_run_directory_are_refused(tmp_path):
    pairs, codes = write_ita_tables(tmp_path, *ita_tables())
    (tmp_path / OUTPUTS_FILE).write_text('{}\n', encoding='utf-8')
    evidence = ItaEvidence(
        protocol_commit='0' * 40,
        run='stage10_fixture_r1',
        outputs_sha256=sha256_file(tmp_path / OUTPUTS_FILE),
        pairs_sha256=sha256_file(pairs),
        pair_codes_sha256=sha256_file(codes),
    )
    with pytest.raises(ValueError, match='run'):
        check_ita_run(pairs, codes, evidence)

# -------------------------------------------------------------------------------------------------
# The evidence file
# -------------------------------------------------------------------------------------------------

def _write_evidence(tmp_path: Path, **changes) -> Path:
    content = yaml.safe_load(EVIDENCE_FILE.read_text(encoding='utf-8'))
    content.update(changes)
    path = tmp_path / 'ita_evidence.yaml'
    path.write_text(yaml.safe_dump(content), encoding='utf-8')
    return path

def test_an_evidence_file_with_an_unknown_key_or_a_short_hash_is_refused(tmp_path):
    with pytest.raises(ValidationError):
        load_ita_evidence(_write_evidence(tmp_path, panel='ita_text'))
    with pytest.raises(ValidationError):
        load_ita_evidence(_write_evidence(tmp_path, pairs_sha256='60aad764'))

@pytest.mark.parametrize(
    'changes',
    [
        {
            'protocol_commit': '63c0666'
        },
        {
            'protocol_commit': '63C06664BDCCE3867E2EC76C2607683DE1C4320D'
        },
        {
            'outputs_sha256': 'E19B0D58F61F106C3994AA6FE51F4FB71244188145DD90F8CCCAA8DC8A8B09DF'
        },
        {
            'pair_codes_sha256': 'c8084e4ad297e3f78391e035ebe51119b07982639ae4daf47f1d47caeb1f8916a'
        },
        {
            'run': ''
        },
        {
            'run': 'stage10 ita r1'
        },
        {
            'run': 'stage10_ita_r1\n'
        },
    ],
    ids=[
        'short-commit',
        'upper-commit',
        'upper-hash',
        'long-hash',
        'empty-run',
        'spaced-run',
        'newline-run',
    ],
)
def test_an_evidence_value_out_of_form_is_refused(tmp_path, changes):
    with pytest.raises(ValidationError):
        load_ita_evidence(_write_evidence(tmp_path, **changes))

@pytest.mark.parametrize(
    'field', ['protocol_commit', 'run', 'outputs_sha256', 'pairs_sha256', 'pair_codes_sha256']
)
def test_an_evidence_file_without_a_key_is_refused(tmp_path, field):
    content = yaml.safe_load(EVIDENCE_FILE.read_text(encoding='utf-8'))
    del content[field]
    path = tmp_path / 'ita_evidence.yaml'
    path.write_text(yaml.safe_dump(content), encoding='utf-8')
    with pytest.raises(ValidationError):
        load_ita_evidence(path)
