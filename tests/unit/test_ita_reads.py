'''
Text→code's reference reads on the ITA validation part (plan 20, Task 2).

``read_reference`` reads one reference arm on the fixture panel with role ``reference``, and on
request the outcome panel's validation split beside it, under the same detail. Its refusals come
before any log record. The stored frames and the record are checked against a direct read of the
same panel, and Stage 5's margin-first check is run after two such reads.
'''

from datetime import timedelta

import pytest
from naics_embedder.decision.ita_reads import (
    ITA_PER_PAIR_FILE,
    OUTCOME_SCORES_FILE,
    READ_ROLE,
    ItaReferenceRead,
    read_reference,
)
from pydantic import ValidationError

from naics_embedder.decision.records import read_record, write_record
from naics_embedder.decision.scores import outcome_scores
from naics_embedder.decision.store import ArtifactStore
from naics_embedder.decision.text_to_code import check_ita_margin_first, fix_ita_margin
from naics_embedder.panels.ita import ItaPanel
from naics_embedder.panels.selection_log import SelectionLog
from tests.fixtures.ita import (
    ITA_CANDIDATES,
    ITA_SETTINGS,
    AssignedStubEncoder,
    IdentifiedStubEncoder,
    ita_run_dir,
    ita_tables,
    outcome_beside_panel,
)

pytestmark = pytest.mark.unit

RUN = 'ita-gap/0123456789abcdef0123456789abcdef'
PURPOSE = 'text-to-code reference gap read: fixture'

@pytest.fixture
def log(tmp_path):
    return SelectionLog(tmp_path / 'logs' / 'selection_log.jsonl')

@pytest.fixture
def panel(log):
    return ItaPanel(*ita_tables(), ITA_CANDIDATES, ITA_SETTINGS, log)

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

@pytest.fixture
def evidence(tmp_path):
    _, _, pinned = ita_run_dir(tmp_path / 'edata', *ita_tables())
    return pinned

def _direct(tmp_path, encoder):
    '''The same validation read on a panel of its own log: what the record must report.'''

    direct = ItaPanel(
        *ita_tables(), ITA_CANDIDATES, ITA_SETTINGS, SelectionLog(tmp_path / 'direct.jsonl')
    )
    return direct.score(encoder, 'validation', 'direct read')

def _stub(panel, store, evidence, **changes):
    arguments = {'arm': 'lexical-stub', 'seed': None, 'run': RUN, 'purpose': PURPOSE}
    arguments.update(changes)
    return read_reference(panel, AssignedStubEncoder(), store, evidence, **arguments)

def _d16(panel, store, evidence, encoder=None, **changes):
    arguments = {'arm': 'euclidean-d16', 'seed': 3, 'run': RUN, 'purpose': PURPOSE}
    arguments.update(changes)
    return read_reference(panel, encoder or IdentifiedStubEncoder(), store, evidence, **arguments)

# -------------------------------------------------------------------------------------------------
# One read
# -------------------------------------------------------------------------------------------------

def test_a_stub_read_logs_one_validation_read_with_the_reference_detail(
    tmp_path, panel, log, store, evidence
):
    record = _stub(panel, store, evidence)

    assert (record.kind, record.arm, record.seed, record.outcome) == (
        'ita_reference_read', 'lexical-stub', None, None
    )
    assert (record.checkpoint_sha256, record.table_fingerprint) == (None, None)
    assert (record.run, record.purpose, record.evidence) == (RUN, PURPOSE, evidence)
    [entry] = log.records()
    assert (entry['event'], entry['panel'], entry['split'], entry['purpose']) == (
        'read', 'ita_text', 'validation', PURPOSE
    )
    assert {
        key: entry['detail'][key]
        for key in ('arm', 'role', 'run')
    } == {
        'arm': 'lexical-stub',
        'role': READ_ROLE,
        'run': RUN,
    }
    assert READ_ROLE == 'reference'
    assert not {'seed', 'checkpoint', 'table'} & set(entry['detail'])
    assert record.log_records == [entry]
    direct = _direct(tmp_path, AssignedStubEncoder())
    assert record.headline == direct.headline
    assert record.copied == direct.copied
    assert record.phrasing == direct.phrasing
    assert record.counts == direct.counts
    assert record.split_fingerprint == panel.fingerprint
    assert record.data_fingerprint == panel.data_fingerprint('validation')
    assert record.created_at.utcoffset() == timedelta(0)

def test_a_d16_read_logs_its_seed_checkpoint_and_table(panel, log, store, evidence):
    record = _d16(panel, store, evidence)

    [entry] = log.records()
    assert entry['detail']['arm'] == 'euclidean-d16'
    assert entry['detail']['role'] == 'reference'
    assert entry['detail']['run'] == RUN
    assert entry['detail']['seed'] == 3
    assert entry['detail']['checkpoint'] == 'c' * 64
    assert entry['detail']['table'] == 't' * 64
    assert (record.arm, record.seed) == ('euclidean-d16', 3)
    assert (record.checkpoint_sha256, record.table_fingerprint) == ('c' * 64, 't' * 64)
    assert record.log_records == [entry]

def test_an_outcome_read_beside_logs_a_second_record_with_the_same_detail(
    tmp_path, panel, log, store, evidence
):
    outcome_panel = outcome_beside_panel(log)

    record = _stub(panel, store, evidence, outcome_panel=outcome_panel)

    ita_entry, outcome_entry = log.records()
    assert (ita_entry['panel'], outcome_entry['panel']) == ('ita_text', 'outcome')
    assert (outcome_entry['event'], outcome_entry['split'], outcome_entry['purpose']) == (
        'read', 'validation', PURPOSE
    )
    for key in ('arm', 'role', 'run'):
        assert outcome_entry['detail'][key] == ita_entry['detail'][key]
    assert outcome_entry['detail']['distance'] == 'cosine'
    assert record.log_records == [ita_entry, outcome_entry]
    assert record.outcome.fingerprint == outcome_panel.fingerprint
    assert record.outcome.data_fingerprint == outcome_panel.data_fingerprint('validation')
    direct = outcome_beside_panel(SelectionLog(tmp_path / 'direct.jsonl')
                                  ).score(AssignedStubEncoder(), 'validation', 'direct read')
    assert record.outcome.summary == direct.summary
    assert store.read_frame(record.outcome.scores).equals(outcome_scores(direct.per_query))
    assert record.outcome.scores.path.endswith('/' + OUTCOME_SCORES_FILE)

def test_a_d16_outcome_read_carries_its_identities_and_its_distance(panel, log, store, evidence):
    encoder = IdentifiedStubEncoder()
    encoder.distance = 'euclidean'

    _d16(panel, store, evidence, encoder=encoder, outcome_panel=outcome_beside_panel(log))

    ita_entry, outcome_entry = log.records()
    for entry in (ita_entry, outcome_entry):
        assert entry['detail']['distance'] == 'euclidean'
        assert (entry['detail']['seed'], entry['detail']['checkpoint'],
                entry['detail']['table']) == (3, 'c' * 64, 't' * 64)

# -------------------------------------------------------------------------------------------------
# Refusals before any record
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    'arm, seed',
    [
        ('euclidean-d16-s1', 1), ('lexical-stub', 2), ('euclidean-d16', None), ('euclidean-d16', 0),
        ('euclidean-d16', -1), ('candidate', None)
    ],
)
def test_bad_arms_or_seeds_are_refused_before_logging(
    tmp_path, panel, log, store, evidence, arm, seed
):
    with pytest.raises(ValueError):
        read_reference(
            panel,
            IdentifiedStubEncoder(),
            store,
            evidence,
            arm=arm,
            seed=seed,
            run=RUN,
            purpose=PURPOSE,
            outcome_panel=outcome_beside_panel(log)
        )
    assert log.records() == []
    assert not (tmp_path / 'store').exists()

@pytest.mark.parametrize('field', ['run', 'purpose'])
@pytest.mark.parametrize('blank', ['', '   ', '\n'])
def test_a_blank_run_or_purpose_is_refused_before_logging(
    panel, log, store, evidence, field, blank
):
    with pytest.raises(ValueError, match=field):
        _stub(panel, store, evidence, **{field: blank})
    assert log.records() == []

@pytest.mark.parametrize('missing', ['checkpoint_sha256', 'table_fingerprint'])
def test_a_d16_encoder_without_its_identities_is_refused_before_logging(
    panel, log, store, evidence, missing
):

    class Unidentified(AssignedStubEncoder):
        pass

    encoder = Unidentified()
    if missing != 'checkpoint_sha256':
        encoder.checkpoint_sha256 = 'c' * 64
    if missing != 'table_fingerprint':
        encoder.table_fingerprint = 't' * 64
    with pytest.raises(ValueError, match=missing):
        _d16(panel, store, evidence, encoder=encoder)
    assert log.records() == []

# -------------------------------------------------------------------------------------------------
# The stored frame and the record
# -------------------------------------------------------------------------------------------------

def test_the_per_pair_frame_is_stored_and_the_record_round_trips(tmp_path, panel, store, evidence):
    record = _stub(panel, store, evidence)

    direct = _direct(tmp_path, AssignedStubEncoder())
    assert store.read_frame(record.per_pair).equals(direct.per_pair)
    assert record.per_pair.path.endswith('/' + ITA_PER_PAIR_FILE)
    path = write_record(record, tmp_path / 'records' / 'lexical-stub.json')
    assert read_record(path, ItaReferenceRead) == record

def test_the_record_refuses_an_unknown_key_a_bad_seed_or_another_arm(panel, store, evidence):
    record = _d16(panel, store, evidence)
    fields = record.model_dump()

    with pytest.raises(ValidationError):
        ItaReferenceRead.model_validate({**fields, 'reference': 'lexical-stub'})
    with pytest.raises(ValidationError):
        ItaReferenceRead.model_validate({**fields, 'seed': 0})
    with pytest.raises(ValidationError):
        ItaReferenceRead.model_validate({**fields, 'arm': 'euclidean-d16-s3'})
    with pytest.raises(ValidationError):
        record.run = 'ita-gap/other'

# -------------------------------------------------------------------------------------------------
# Stage 5's margin-first check
# -------------------------------------------------------------------------------------------------

def test_stage_4_reads_leave_the_margin_first_check_open(panel, store, evidence):
    _stub(panel, store, evidence)
    _d16(panel, store, evidence, seed=1)

    margin = fix_ita_margin(
        panel,
        {
            'lexical-stub': [(None, AssignedStubEncoder())],
            'euclidean-d16': [(1, IdentifiedStubEncoder())],
        },
        0.05,
        'ita-margins',
        purpose='fix the margin',
    )

    check_ita_margin_first(margin, panel.log)  # raises nothing
