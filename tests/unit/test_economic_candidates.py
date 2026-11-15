'''
The frozen economic candidate list (spec economic-candidates, 4.1; plan 16, Task 3) and the
text-and-tree-only provenance rule (4.5; Task 4).

The committed list is read as it stands. The entry and list rules are exercised over small lists
built here, and each kind's ArmSpec is checked against the text-only table's provenance fields.
The provenance rule reads each builder's real provenance and a fabricated export's.
'''

import json
import re
from pathlib import Path
from typing import Any, Dict, NamedTuple

import pytest
import yaml
from pydantic import ValidationError

from naics_embedder.decision.candidates import (
    ECONOMIC_CANDIDATES,
    CandidateEntry,
    CandidateList,
    CheckpointContract,
    candidate_table_path,
    economic_spec,
    load_candidate_list,
)
from naics_embedder.decision.provenance import (
    EXPORT_ENTRIES,
    EXPORT_ENTRIES_WITH_GEOMETRY,
    check_candidate_provenance,
)
from naics_embedder.decision.records import ArmRecord, ArmSpec, ArtifactRef, read_record, write_record
from naics_embedder.decision.store import ArtifactStore, provenance_fields
from naics_embedder.decision.trained import import_trained_candidate
from naics_embedder.panels.text_only import provenance_path
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.tree_model.builders import BUILDER_ENTRIES, BUILDER_INPUTS, KINDS, build_candidate
from tests.fixtures.decision import SEED_OFFSETS, spec, synthetic_arm, write_text_only
from tests.fixtures.economic import (
    CODEBOOK_INPUT,
    CODEBOOK_SHA256,
    CONTRACT,
    export_provenance,
    fixture_list,
    trained_source,
    write_candidate,
    write_codebook,
)
from tests.fixtures.regressor_panel import CODEBOOK, coordinate_table

pytestmark = pytest.mark.unit

SPEC_ROWS = [  # §4.1's table, in its order: (name, kind, dimension, components, geometry)
    *[(f'{kind}-d{d}', kind, d, components, 'euclidean')
      for kind, components in (('text-only', 1), ('tree', 1), ('smoothed', 2))
      for d in (8, 16, 32)],
    ('hyperbolic-d16', 'trained', 16, 1, 'hyperbolic'),
    *[(f'euclidean-d{d}', 'trained', d, 1, 'euclidean') for d in (8, 16, 32)],
    *[(f'hyperbolic-d{d}', 'trained', d, 1, 'hyperbolic') for d in (8, 32)],
    *[(f'spherical-d{d}', 'trained', d, 1, 'spherical') for d in (8, 16, 32)],
]
RECORD_SHA256 = {  # STORE/records/…, hashed 2026-11-14
    'hyperbolic-d16': 'c885b9c5dc18b6be03670d0cb5a71db3974917da8f719e0dbb1f1ef5eec2d1a2',
    'euclidean-d8': '103367f0f8f9270567ef2acfff1bd5a870aece94f4e66d75c51390eeba758dbb',
    'euclidean-d16': '3d5a80d5a3ac5466e10b019773fa09a21e2ad0592af32d7b22273fea3e1bab2c',
    'euclidean-d32': 'cf4edd78d3cd195470fa2d5325e791e1e52209c7dbb0249ee2dc39bc713bf2c0',
    'hyperbolic-d8': '53a7b32959f9b7b281b7127772631d2a57573b10762ac5ba05ae164b0fa08ecd',
    'hyperbolic-d32': '927c3fd96fec78b31d456d0fa88d2c1d8b133cc9da0707cc82f63a92aec36016',
    'spherical-d8': 'ce739fa9b73ad2538fc0c973c2f0e64eec453be3f77ee78534748a6b288a87df',
    'spherical-d16': '28fec47082c3fe82e9d2c23b397c4ccf9a6af7cc2df22d5f1531cfa891c95c01',
    'spherical-d32': '194d2ea6a9420cf6c16574891ff21832b5bbae25dd4cbfe8fa8158fe6eeaa679',
}

ROOT = Path(__file__).resolve().parents[2]  # the repository, whatever the working directory

# -------------------------------------------------------------------------------------------------
# The committed list
# -------------------------------------------------------------------------------------------------

def test_the_committed_list_is_the_specs_eighteen_candidates():
    committed = load_candidate_list(ROOT / ECONOMIC_CANDIDATES)
    assert [
        (e.name, e.kind, e.dimension, e.components, e.geometry) for e in committed.candidates
    ] == SPEC_ROWS
    assert committed.codebook_sha256 == (
        '0f9e61343e673540e1e18ecc5581618be099c5d79531b68153496bf982d6e96d'
    )
    assert committed.text_only_sha256 == (
        'f4fb4f574f880c940ee8cf44c7f86615638e59f113b527d0b86973cd45ba32ff'
    )
    assert committed.checkpoint_contract == CheckpointContract(
        bundle_id='301cce28-539c-42ea-8781-496bbdcf511c', contract_version='stage3-supervision-v2'
    )
    for entry in committed.candidates:
        if entry.kind != 'trained':
            continue
        stage7 = entry.name == 'hyperbolic-d16'
        assert entry.record == (
            'records/stage7/reference.json' if stage7 else f'records/stage8/{entry.name}.json'
        )
        assert entry.arm == ('reference' if stage7 else entry.name)
        assert entry.seeds == list(range(1, 11 if stage7 else 6))
        assert entry.run_directory == (
            'stage7-reference-s{seed}' if stage7 else f'stage8-{entry.name}-s{{seed}}'
        )
        assert entry.record_sha256 == RECORD_SHA256[entry.name]

def test_the_committed_list_names_no_user_path():
    assert '/Users/' not in (ROOT / ECONOMIC_CANDIDATES).read_text()

def test_the_propagation_pairs_are_each_smoothed_candidate_and_its_text_only_twin():
    assert load_candidate_list(ROOT / ECONOMIC_CANDIDATES).propagation_pairs() == [
        (f'smoothed-d{d}', f'text-only-d{d}') for d in (8, 16, 32)
    ]

# -------------------------------------------------------------------------------------------------
# Small lists
# -------------------------------------------------------------------------------------------------

def _trained(name: str = 'trained-d8', **edits) -> dict:
    '''A valid trained entry, as the YAML gives it; ``edits`` replace or add keys.'''

    return {
        'name': name,
        'kind': 'trained',
        'dimension': 8,
        'components': 1,
        'geometry': 'euclidean',
        'record': f'records/stage8/{name}.json',
        'record_sha256': 'a' * 64,
        'arm': name,
        'seeds': [1, 2, 3, 4, 5],
        'run_directory': f'stage8-{name}-s{{seed}}',
        **edits,
    }

def _deterministic(kind: str = 'tree', dimension: int = 8, **edits) -> dict:
    '''A valid deterministic entry named ``<kind>-d<dimension>``.'''

    return {
        'name': f'{kind}-d{dimension}',
        'kind': kind,
        'dimension': dimension,
        'components': 2 if kind == 'smoothed' else 1,
        'geometry': 'euclidean',
        'table_sha256': None,
        **edits,
    }

def _list(*entries: dict, **edits) -> dict:
    '''A list as the YAML gives it, over ``entries``; ``edits`` replace or add keys.'''

    return {
        'codebook_sha256': '1' * 64,
        'text_only_sha256': '2' * 64,
        'checkpoint_contract': {
            'bundle_id': 'bundle',
            'contract_version': 'contract'
        },
        'candidates': list(entries),
        **edits,
    }

def _with_twin(entry: dict) -> dict:
    '''A list of the entry, with the text-only entry a smoothed one needs.'''

    twin = [_deterministic('text-only', entry['dimension'])] if entry['kind'] == 'smoothed' else []
    return _list(*twin, entry)

# -------------------------------------------------------------------------------------------------
# Entry rules
# -------------------------------------------------------------------------------------------------

BAD_SHAS = {
    'a 63-character sha': 'a' * 63,
    'a 65-character sha': 'a' * 65,
    'an upper-case sha': 'A' * 64,
    'a non-hex sha': 'g' * 64,
    'a sha with a trailing newline': 'a' * 64 + '\n',
    'an empty sha': '',
}

@pytest.mark.parametrize(
    'entry',
    [
        pytest.param(_trained(seeds=[1, 2, 3, 4]), id='trained: 4 seeds'),
        pytest.param(_trained(seeds=[]), id='trained: no seeds'),
        pytest.param(_trained(seeds=[1, 1, 1, 1, 1]), id='trained: five entries, one seed'),
        pytest.param(_trained(seeds=[1, 2, 3, 4, 5, 5]), id='trained: a repeat past the floor'),
        pytest.param(_trained(seeds=[1, 1, 2, 3, 4]), id='trained: a repeat, four distinct seeds'),
        pytest.param(_trained(seeds=['1', '2', '3', '4', '5']), id='trained: seeds as strings'),
        pytest.param(_trained(seeds=[True, 2, 3, 4, 5]), id='trained: a bool seed'),
        pytest.param(_trained(seeds=[1.0, 2, 3, 4, 5]), id='trained: a float seed'),
        pytest.param(_trained(record='/records/x.json'), id='trained: an absolute record'),
        pytest.param(_trained(record='../x.json'), id='trained: a record that starts with ..'),
        pytest.param(_trained(record='records/../x.json'), id='trained: a record with .. inside'),
        pytest.param(_trained(record='records/stage8/..'), id='trained: a record ending in ..'),
        pytest.param(_trained(record=''), id='trained: a blank record'),
        pytest.param(_trained(record=None), id='trained: no record'),
        pytest.param(_trained(record_sha256=None), id='trained: no record_sha256'),
        pytest.param(_trained(arm=None), id='trained: no arm'),
        pytest.param(_trained(arm=''), id='trained: a blank arm'),
        pytest.param(_trained(run_directory=None), id='trained: no run_directory'),
        pytest.param(
            _trained(run_directory='stage8-s'), id='trained: a run_directory lacking {seed}'
        ),
        pytest.param(_trained(run_directory='stage8-s{Seed}'), id='trained: {Seed} is not {seed}'),
        pytest.param(
            _trained(run_directory='stage8/s{seed}'), id='trained: a run_directory with /'
        ),
        pytest.param(
            _trained(run_directory='/s{seed}'), id='trained: a run_directory with a lead /'
        ),
        pytest.param(
            _trained(run_directory='s{seed}/'), id='trained: a run_directory with a last /'
        ),
        pytest.param(_trained(table_sha256='b' * 64), id='trained: a table_sha256'),
        pytest.param(_deterministic('tree', seeds=[1, 2, 3, 4, 5]), id='deterministic: seeds'),
        pytest.param(_deterministic('tree', dimension=0), id='dimension 0'),
        pytest.param(_deterministic('tree', components=0), id='components 0'),
        pytest.param(_deterministic('tree', name=''), id='a blank name'),
        pytest.param(_deterministic('tree', geometry='poincare'), id='an unknown geometry'),
        pytest.param(_deterministic('untrained'), id='an unknown kind'),
        pytest.param(_deterministic('tree', extra=1), id='an unknown key'),
        pytest.param(_trained(extra=1), id='trained: an unknown key'),
        *[
            pytest.param(_trained(record_sha256=bad), id=f'trained: record_sha256, {label}')
            for label, bad in BAD_SHAS.items()
        ],
        *[
            pytest.param(
                _deterministic('tree', table_sha256=bad), id=f'tree: table_sha256, {label}'
            ) for label, bad in BAD_SHAS.items()
        ],
    ],
)
def test_a_list_that_breaks_an_entry_rule_is_refused(entry):
    with pytest.raises(ValidationError):
        CandidateList.model_validate(_with_twin(entry))

@pytest.mark.parametrize('kind', ['text-only', 'tree', 'smoothed'])
@pytest.mark.parametrize(
    'field, value',
    [
        ('record', 'records/stage8/x.json'),
        ('record', ''),
        ('record_sha256', 'a' * 64),
        ('arm', 'reference'),
        ('arm', ''),
        ('seeds', [1, 2, 3, 4, 5]),
        ('seeds', [1]),
        ('run_directory', 'stage8-x-s{seed}'),
        ('run_directory', ''),
    ],
)
def test_a_deterministic_entry_names_no_record_arm_seeds_or_run_directory(kind, field, value):
    entry = _deterministic(kind, **{field: value})
    with pytest.raises(ValidationError):
        CandidateList.model_validate(_with_twin(entry))

@pytest.mark.parametrize(
    'entry',
    [
        pytest.param(_trained(), id='trained: exactly the 5 seeds'),
        pytest.param(_trained(seeds=list(range(1, 11))), id='trained: 10 seeds'),
        pytest.param(_trained(seeds=[3, 5, 7, 9, 11]), id='trained: seeds that are not 1 to n'),
        pytest.param(_trained(seeds=[5, 4, 3, 2, 1]), id='trained: seeds out of order'),
        pytest.param(_trained(record='records/stage8/a..b.json'), id='trained: .. inside a name'),
        pytest.param(_trained(record='x.json'), id='trained: a record in the root'),
        pytest.param(_trained(run_directory='{seed}'), id='trained: a run_directory of {seed}'),
        pytest.param(_trained(run_directory='s{seed}-x'), id='trained: {seed} inside the name'),
        pytest.param(_deterministic('tree', table_sha256='b' * 64), id='tree: a pinned table'),
        pytest.param(_deterministic('tree'), id='tree: an unpinned table'),
        pytest.param(_deterministic('text-only'), id='text-only'),
        pytest.param(_deterministic('smoothed'), id='smoothed'),
    ],
)
def test_an_entry_that_keeps_each_rule_loads(entry):
    candidates = CandidateList.model_validate(_with_twin(entry))
    loaded = candidates.entry(entry['name'])
    assert loaded.kind == entry['kind']
    assert loaded.seeds == entry.get('seeds', [])

def test_the_models_are_frozen_and_the_sections_take_no_unknown_key():
    candidates = CandidateList.model_validate(_list(_deterministic('tree')))
    for model, field in (
        (candidates, 'codebook_sha256'), (candidates.candidates[0], 'name'), (
            candidates.checkpoint_contract, 'bundle_id'
        )
    ):
        with pytest.raises(ValidationError):
            setattr(model, field, 'other')
    with pytest.raises(ValidationError):
        CandidateList.model_validate(_list(_deterministic('tree'), extra=1))
    with pytest.raises(ValidationError):
        CandidateList.model_validate(
            _list(
                _deterministic('tree'),
                checkpoint_contract={
                    'bundle_id': 'b',
                    'contract_version': 'c',
                    'extra': 1
                }
            )
        )
    with pytest.raises(ValidationError):
        CandidateList.model_validate(
            _list(_deterministic('tree'), checkpoint_contract={'bundle_id': 'b'})
        )

# -------------------------------------------------------------------------------------------------
# List rules
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    'entries',
    [
        pytest.param([_deterministic('tree'), _deterministic('tree')], id='one name twice'),
        pytest.param([_deterministic('tree'), _trained('tree-d8')], id='a name across kinds'),
        pytest.param([_deterministic('smoothed')], id='a smoothed entry with no text-only entry'),
        pytest.param(
            [_deterministic('smoothed', 8),
             _deterministic('text-only', 16)],
            id='a smoothed entry with a text-only entry at another dimension'
        ),
        pytest.param(
            [_deterministic('smoothed', 8),
             _deterministic('tree', 8)],
            id='a tree entry at its dimension is no twin'
        ),
        pytest.param(
            [_deterministic('smoothed', 8), _trained('euclidean-d8')],
            id='a trained entry at its dimension is no twin'
        ),
        pytest.param(
            [
                _deterministic('text-only', 8),
                _deterministic('text-only', 8, name='text-only-b'),
                _deterministic('smoothed', 8),
            ],
            id='two text-only entries at its dimension'
        ),
        pytest.param(
            [
                _deterministic('text-only', 8),
                _deterministic('smoothed', 8),
                _deterministic('smoothed', 16),
            ],
            id='one smoothed entry has its twin and the next has none'
        ),
    ],
)
def test_a_list_with_repeated_names_or_an_orphan_smoothed_candidate_is_refused(entries):
    with pytest.raises(ValidationError):
        CandidateList.model_validate(_list(*entries))

def test_two_smoothed_entries_at_a_dimension_share_its_one_text_only_twin():
    candidates = CandidateList.model_validate(
        _list(
            _deterministic('text-only', 8),
            _deterministic('smoothed', 8),
            _deterministic('smoothed', 8, name='smoothed-b'),
        )
    )
    assert candidates.propagation_pairs() == [
        ('smoothed-d8', 'text-only-d8'), ('smoothed-b', 'text-only-d8')
    ]

def test_the_names_are_the_lists_in_its_order():
    candidates = CandidateList.model_validate(
        _list(
            _deterministic('tree', 32), _trained('b-d8'), _deterministic('tree', 8), _trained(
                'a-d8'
            )
        )
    )
    assert candidates.names == ['tree-d32', 'b-d8', 'tree-d8', 'a-d8']

def test_an_entry_is_found_by_its_name():
    candidates = CandidateList.model_validate(_list(_deterministic('tree', 8), _trained('b-d8')))
    assert candidates.entry('b-d8') == CandidateEntry.model_validate(_trained('b-d8'))
    assert candidates.entry('tree-d8').kind == 'tree'

@pytest.mark.parametrize('name', ['nope', 'tree-d', 'TREE-D8', 'tree-d8 ', ''])
def test_a_name_off_the_list_is_refused(name):
    candidates = CandidateList.model_validate(_list(_deterministic('tree', 8)))
    with pytest.raises(ValueError, match='is not on the frozen candidate list'):
        candidates.entry(name)

def test_the_propagation_pairs_follow_the_kinds_and_dimensions_not_the_names_or_order():
    # The list's order, the names' order and the dimensions' order all differ
    candidates = CandidateList.model_validate(
        _list(
            _deterministic('smoothed', 32, name='b'),
            _deterministic('text-only', 8, name='twin-8'),
            _deterministic('smoothed', 8, name='c'),
            _deterministic('text-only', 32, name='twin-32'),
            _deterministic('smoothed', 16, name='a'),
            _deterministic('text-only', 16, name='twin-16'),
            _deterministic('tree', 8),
        )
    )
    assert candidates.propagation_pairs() == [('b', 'twin-32'), ('c', 'twin-8'), ('a', 'twin-16')]

def test_a_list_without_a_smoothed_candidate_has_no_propagation_pair():
    candidates = CandidateList.model_validate(
        _list(_deterministic('text-only', 8), _deterministic('tree', 8))
    )
    assert candidates.propagation_pairs() == []

# -------------------------------------------------------------------------------------------------
# Loading and table paths
# -------------------------------------------------------------------------------------------------

def test_a_list_loads_from_a_yaml_file_given_as_a_path_or_a_string(tmp_path):
    entries = [_deterministic('text-only', 8), _deterministic('smoothed', 8), _trained('t-d8')]
    path = tmp_path / 'candidates.yaml'
    path.write_text(yaml.safe_dump(_list(*entries)))
    expected = CandidateList.model_validate(_list(*entries))
    assert load_candidate_list(path) == expected
    assert load_candidate_list(str(path)) == expected

@pytest.mark.parametrize(
    'text, error',
    [
        ('!!python/name:os.getcwd', yaml.YAMLError),
        ('- just\n- a list\n', ValidationError),
        ('', ValidationError),
        ('codebook_sha256: x: y\n', yaml.YAMLError),
    ],
    ids=['a python tag', 'a list', 'an empty file', 'broken yaml'],
)
def test_a_file_that_is_not_a_candidate_list_is_refused(tmp_path, text, error):
    path = tmp_path / 'candidates.yaml'
    path.write_text(text)
    with pytest.raises(error):
        load_candidate_list(path)

@pytest.mark.parametrize(
    'text', ['!!python/name:os.getcwd', 'codebook_sha256: x: y\n'],
    ids=['a python tag', 'broken yaml']
)
def test_a_file_that_is_not_safe_yaml_is_refused_as_a_value_error_naming_it(tmp_path, text):
    '''The commands' handlers catch ValueError: a hand-broken list prints their refusal.'''

    path = tmp_path / 'candidates.yaml'
    path.write_text(text)
    with pytest.raises(ValueError, match=f'^{re.escape(str(path))} is not safe YAML: '):
        load_candidate_list(path)

def test_a_table_lives_under_the_stores_candidates_economic_directory(tmp_path):
    expected = tmp_path / 'candidates' / 'economic' / 'tree-d8.parquet'
    assert candidate_table_path(tmp_path, 'tree-d8') == expected
    assert candidate_table_path(str(tmp_path), 'tree-d8') == expected
    assert candidate_table_path('store', 'smoothed-d16') == Path(
        'store/candidates/economic/smoothed-d16.parquet'
    )

# -------------------------------------------------------------------------------------------------
# Each kind's spec
# -------------------------------------------------------------------------------------------------

TEXT_ONLY_PROVENANCE = {
    'table_sha256': '3' * 64,
    'matrix_fingerprint': 'fingerprint',
    'backbone': 'minilm',
    'revision': 'rev-1',
    'descriptions': {
        'sha256': '4' * 64
    },
    'summaries': '5' * 64,
    'max_length': 512,
}

def _text_only(**edits) -> dict:
    '''The text-only fields ``provenance_fields`` reads from the provenance above.'''

    return provenance_fields(
        {
            **TEXT_ONLY_PROVENANCE,
            **edits
        }, '3' * 64, 'fingerprint', 'the text-only table'
    )

def _entry(row: dict) -> CandidateEntry:
    return CandidateEntry.model_validate(row)

NO_BACKBONE = {
    'backbone': None,
    'backbone_revision': None,
    'descriptions_sha256': None,
    'summaries_sha256': None,
    'max_length': None,
}

@pytest.mark.parametrize('given', [None, 'the text-only fields'])
def test_a_tree_candidate_reads_no_backbone(given):
    text_only = _text_only() if given else None
    built = economic_spec(_entry(_deterministic('tree', 16)), text_only=text_only)
    assert built == ArmSpec(
        name='tree-d16', components=1, dimension=16, geometry='euclidean', **NO_BACKBONE
    )
    assert built.settings == {}
    assert not built.reads_backbone

@pytest.mark.parametrize('kind, components', [('text-only', 1), ('smoothed', 2)])
@pytest.mark.parametrize(
    'pins, revision, summaries',
    [({}, 'rev-1', '5' * 64), ({
        'revision': None,
        'summaries': None
    }, None, None)],
    ids=['a pinned backbone', 'an unpinned backbone'],
)
def test_a_text_only_or_smoothed_candidate_takes_the_text_only_fields(
    kind, components, pins, revision, summaries
):
    entry = _entry(_deterministic(kind, 32))
    built = economic_spec(entry, text_only=_text_only(**pins))
    assert built == ArmSpec(
        name=f'{kind}-d32',
        components=components,
        dimension=32,
        geometry='euclidean',
        backbone='minilm',
        backbone_revision=revision,
        descriptions_sha256='4' * 64,
        summaries_sha256=summaries,
        max_length=512,
    )
    assert built.settings == {}
    assert built.reads_backbone

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_candidate_that_reads_the_backbone_needs_the_text_only_fields(kind):
    with pytest.raises(ValueError, match='text-only'):
        economic_spec(_entry(_deterministic(kind, 8)))

def test_the_spec_takes_its_shape_from_the_entry_not_from_its_kind():
    row = _deterministic('smoothed', 24, name='odd', components=3, geometry='spherical')
    built = economic_spec(_entry(row), text_only=_text_only())
    assert (built.name, built.components, built.dimension, built.geometry) == (
        'odd', 3, 24, 'spherical'
    )

def test_a_trained_candidate_is_its_source_spec_under_the_entrys_name():
    source = spec('reference', dimension=16, geometry='hyperbolic', settings={'warmup': 3})
    entry = _entry(_trained('hyperbolic-d16', dimension=16, geometry='hyperbolic'))
    built = economic_spec(entry, source=source)
    assert built.name == 'hyperbolic-d16'
    assert built == source.model_copy(update={'name': 'hyperbolic-d16'})
    assert built.settings == {'warmup': 3}
    assert built.backbone == source.backbone
    assert source.name == 'reference'

def test_a_trained_candidate_ignores_the_text_only_fields():
    source = spec('reference', dimension=8, geometry='euclidean')
    entry = _entry(_trained('t-d8'))
    assert economic_spec(entry, source=source, text_only=_text_only(backbone='other')) == (
        source.model_copy(update={'name': 't-d8'})
    )

@pytest.mark.parametrize(
    'edits, word',
    [
        ({
            'dimension': 16
        }, 'dimension'),
        ({
            'geometry': 'spherical'
        }, 'geometry'),
        ({
            'components': 2
        }, 'components'),
        ({
            'dimension': 16,
            'geometry': 'spherical',
            'components': 2
        }, 'dimension.*geometry.*comp'),
    ],
    ids=['dimension', 'geometry', 'components', 'all three'],
)
def test_a_source_that_differs_from_the_entry_is_refused(edits, word):
    source = spec('t-d8', **{'dimension': 8, 'geometry': 'euclidean', **edits})
    with pytest.raises(ValueError, match=word):
        economic_spec(_entry(_trained('t-d8')), source=source)

def test_a_trained_candidate_needs_its_source_spec():
    with pytest.raises(ValueError, match='source'):
        economic_spec(_entry(_trained('t-d8')))

# -------------------------------------------------------------------------------------------------
# Text and tree only: the provenance rule (4.5)
# -------------------------------------------------------------------------------------------------

RULE = '(Req 29: text and tree only)'
DETERMINISTIC = {
    kind: CandidateEntry(
        name=f'{kind}-d3',
        kind=kind,
        dimension=3,
        components=2 if kind == 'smoothed' else 1,
        geometry='euclidean',
    )
    for kind in KINDS
}

class Built(NamedTuple):
    '''Each builder's provenance, as ``build_candidate`` wrote it, and the list that matches it.'''

    contents: Dict[str, Dict[str, Any]]
    frozen: CandidateList

def _build_every_kind(directory: Path) -> Built:
    codebook = write_codebook(directory / 'naics_codebook.parquet')
    text_only = write_text_only(directory / 'text', codes=CODEBOOK)
    contents = {}
    for kind in KINDS:
        reads_text = 'text_only' in BUILDER_INPUTS[kind]
        table = build_candidate(
            kind,
            3,
            codebook,
            directory / f'{kind}-d3.parquet',
            text_only=text_only if reads_text else None,
        )
        contents[kind] = json.loads(provenance_path(table).read_text())
    frozen = fixture_list(
        list(DETERMINISTIC.values()),
        text_only_sha256=sha256_file(text_only),
        codebook_sha256=sha256_file(codebook),
    )
    return Built(contents, frozen)

@pytest.fixture(scope='module')
def built(tmp_path_factory) -> Built:
    return _build_every_kind(tmp_path_factory.mktemp('built'))

@pytest.fixture
def tree_entry():
    return CandidateEntry(
        name='tree-d3', kind='tree', dimension=3, components=1, geometry='euclidean'
    )

@pytest.fixture
def tree_content(tmp_path):
    path = write_candidate(tmp_path, 'tree-d3', coordinate_table(CODEBOOK))
    return json.loads(provenance_path(path).read_text())

@pytest.fixture
def trained_entry():
    return _entry(_trained('euclidean-d3', dimension=3))

@pytest.fixture
def candidates(tree_entry, trained_entry):
    return fixture_list([tree_entry, trained_entry], text_only_sha256='a' * 64)

@pytest.fixture
def table(tmp_path):
    path = tmp_path / 'table.parquet'
    coordinate_table(CODEBOOK).write_parquet(path)
    return path

@pytest.fixture
def arm_spec():
    return spec('euclidean-d3', dimension=3, geometry='euclidean')

@pytest.fixture
def checkpoint():
    return ArtifactRef(path='checkpoints/euclidean-d3.ckpt', sha256='b' * 64, bytes=1)

@pytest.fixture
def export_content(table, arm_spec, checkpoint):
    return export_provenance(table, arm_spec, checkpoint)

def _refused(content, entry, frozen, check):
    '''
    The rule refuses the content with a message that starts with the entry's name, then names
    the check that refused it, and ends with the rule's citation.
    '''

    pattern = f'^{re.escape(entry.name)}: {re.escape(check)}: .*{re.escape(RULE)}$'
    with pytest.raises(ValueError, match=pattern):
        check_candidate_provenance(content, entry, frozen)

def _without(content, key):
    return {name: value for name, value in content.items() if name != key}

def _with_input(content, name, **edits):
    '''The builder provenance with one input's entries replaced.'''

    inputs = {**content['inputs'], name: {**content['inputs'][name], **edits}}
    return {**content, 'inputs': inputs}

QCEW = {'path': 'qcew/2023.annual.singlefile.csv', 'sha256': 'f' * 64}

# -------------------------------------------------------------------------------------------------
# What passes
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('kind', KINDS)
def test_each_builders_provenance_passes_the_rule(tmp_path, kind):
    built_here = _build_every_kind(tmp_path)
    entry = DETERMINISTIC[kind]
    content = built_here.contents[kind]
    assert content['builder'] == kind
    check_candidate_provenance(content, entry, built_here.frozen)

@pytest.mark.parametrize('geometry', [False, True], ids=['stage-7', 'stage-8'])
def test_both_export_versions_pass_the_rule(
    table, arm_spec, checkpoint, trained_entry, candidates, geometry
):
    content = export_provenance(table, arm_spec, checkpoint, geometry=geometry)
    check_candidate_provenance(content, trained_entry, candidates)

@pytest.mark.parametrize(
    'edit, message',
    [
        ({
            'qcew': {
                'path': 'qcew/2023.annual.singlefile.csv',
                'sha256': 'f' * 64
            }
        }, 'entries'),
        ({
            'panel': 'growth_seen'
        }, 'entries'),
        ({
            'selection_log': 'logs/selection_log.jsonl'
        }, 'entries'),
        ({
            'inputs': {
                'codebook': CODEBOOK_INPUT,
                'qcew': CODEBOOK_INPUT
            }
        }, 'inputs'),
        (
            {
                'inputs': {
                    'codebook': {
                        'path': 'naics_codebook.parquet',
                        'sha256': 'f' * 64
                    }
                }
            }, 'codebook'
        ),
        ({
            'builder': 'hgcn'
        }, 'builder'),
        ({
            'builder': 'smoothed'
        }, 'builder'),  # a tree entry's provenance from another builder
    ],
)
def test_a_provenance_outside_text_and_tree_is_refused(
    tree_content, tree_entry, candidates, edit, message
):
    with pytest.raises(ValueError, match=f'^tree-d3: .*{message}.*Req 29: text and tree only'):
        check_candidate_provenance({**tree_content, **edit}, tree_entry, candidates)

@pytest.mark.parametrize(
    'contract',
    [
        {
            **CONTRACT.model_dump(), 'codebook_fingerprint': 'x',
            'objective': 'req11-v1'
        },
        dict(CONTRACT.model_dump()),
    ],
    ids=['the full contract an export writes', 'the two named fields'],
)
def test_an_exports_contract_may_carry_more_than_the_two_named_fields(
    export_content, trained_entry, candidates, contract
):
    check_candidate_provenance({**export_content, 'contract': contract}, trained_entry, candidates)

def test_the_fixture_tree_provenance_is_the_tree_builders_entries(tree_content, tree_entry):
    assert set(tree_content) == BUILDER_ENTRIES['tree']
    assert tree_content['builder'] == 'tree'
    assert tree_content['inputs'] == {'codebook': CODEBOOK_INPUT}
    assert tree_content['dimension'] == 3
    assert tree_content['codes'] == len(CODEBOOK)
    assert tree_content['eigenvalues'] == [1.0] * 4

def test_the_export_entries_are_the_fifteen_that_stage_7_and_8_write():
    assert EXPORT_ENTRIES == frozenset(
        {
            'backbone',
            'checkpoint',
            'codes',
            'contract',
            'coordinates',
            'descriptions',
            'dimension',
            'generated_at',
            'library_versions',
            'matrix_fingerprint',
            'max_length',
            'revision',
            'summaries',
            'table_sha256',
            'tokenizer',
        }
    )
    assert EXPORT_ENTRIES_WITH_GEOMETRY == EXPORT_ENTRIES | {'geometry'}

def test_the_fixture_exports_hold_the_two_known_versions_entries(
    table, arm_spec, checkpoint, export_content
):
    assert set(export_content) == EXPORT_ENTRIES_WITH_GEOMETRY
    assert set(export_provenance(table, arm_spec, checkpoint, geometry=False)) == EXPORT_ENTRIES
    assert EXPORT_ENTRIES_WITH_GEOMETRY == EXPORT_ENTRIES | {'geometry'}
    assert export_content['contract'] == CONTRACT.model_dump()
    assert export_content['checkpoint']['sha256'] == checkpoint.sha256

def test_a_tree_provenance_names_no_text_only_input_so_the_lists_text_only_sha_is_not_read(
    tree_content, tree_entry
):
    other = fixture_list([tree_entry], text_only_sha256='9' * 64)
    check_candidate_provenance(tree_content, tree_entry, other)

def test_the_fixture_codebook_sha_is_settable(tmp_path):
    path = write_candidate(
        tmp_path, 'tree-d3', coordinate_table(CODEBOOK), codebook_sha256='7' * 64
    )
    content = json.loads(provenance_path(path).read_text())
    assert content['inputs'] == {'codebook': {**CODEBOOK_INPUT, 'sha256': '7' * 64}}
    assert CODEBOOK_INPUT['sha256'] == CODEBOOK_SHA256

def test_write_candidate_entries_join_or_replace_the_provenance_entries(tmp_path):
    path = write_candidate(
        tmp_path, 'tree-d3', coordinate_table(CODEBOOK), qcew=QCEW, method='another method'
    )
    content = json.loads(provenance_path(path).read_text())
    assert content['qcew'] == QCEW
    assert content['method'] == 'another method'
    assert content['table_sha256'] == sha256_file(path)

# -------------------------------------------------------------------------------------------------
# A builder's provenance
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize(
    'edit',
    [
        {
            'qcew': QCEW
        },
        {
            'panel': 'growth_seen'
        },
        {
            'selection_log': 'logs/selection_log.jsonl'
        },
        {
            'generated_at': '2026-11-14T00:00:00+00:00'
        },
    ],
    ids=['qcew', 'panel', 'selection log', 'a timestamp'],
)
def test_a_builder_provenance_with_an_entry_its_builder_does_not_write_is_refused(
    built, kind, edit
):
    _refused({**built.contents[kind], **edit}, DETERMINISTIC[kind], built.frozen, 'entries')

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_only_the_tree_builder_writes_eigenvalues(built, kind):
    content = {**built.contents[kind], 'eigenvalues': [1.0]}
    _refused(content, DETERMINISTIC[kind], built.frozen, 'entries')

@pytest.mark.parametrize(
    'kind, key',
    [
        pytest.param(kind, key, id=f'{kind}: {key}') for kind in KINDS
        for key in sorted(BUILDER_ENTRIES[kind] - {'builder'})
    ],
)
def test_a_builder_provenance_missing_an_entry_its_builder_writes_is_refused(built, kind, key):
    content = _without(built.contents[kind], key)
    _refused(content, DETERMINISTIC[kind], built.frozen, 'entries')

def test_the_builders_entries_are_what_the_real_builders_wrote(built):
    for kind in KINDS:
        assert set(built.contents[kind]) == BUILDER_ENTRIES[kind]

@pytest.mark.parametrize(
    'name, builder',
    [
        pytest.param(name, builder, id=f'{name} entry, {builder} builder') for name in KINDS
        for builder in KINDS if name != builder
    ],
)
def test_a_provenance_from_another_builder_is_refused(built, name, builder):
    _refused(built.contents[builder], DETERMINISTIC[name], built.frozen, 'builder')

@pytest.mark.parametrize(
    'builder', ['hgcn', 'trained', 'Tree', 'tree ', '', None, ['tree'], 3, {
        'tree': 1
    }]
)
def test_a_builder_outside_the_three_is_refused(built, builder):
    content = {**built.contents['tree'], 'builder': builder}
    _refused(content, DETERMINISTIC['tree'], built.frozen, 'builder')

@pytest.mark.parametrize('kind', KINDS)
def test_a_trained_entry_is_refused_a_builders_provenance(built, trained_entry, candidates, kind):
    _refused(built.contents[kind], trained_entry, candidates, 'builder')

@pytest.mark.parametrize('builder', ['trained', 'euclidean-d3', None, ''])
def test_a_trained_entry_is_refused_a_provenance_that_names_any_builder(
    export_content, trained_entry, candidates, builder
):
    # A builder entry, whatever it holds, means the provenance is a builder's, never an export's
    _refused({**export_content, 'builder': builder}, trained_entry, candidates, 'builder')

@pytest.mark.parametrize(
    'inputs',
    [None, [], ['codebook'], 'codebook', 3, {}, {
        'qcew': CODEBOOK_INPUT
    }, True],
    ids=[
        'null', 'a list', 'a list of the names', 'a string', 'an int', 'empty', 'one wrong name',
        'a bool'
    ],
)
def test_inputs_that_are_not_the_builders_object_are_refused(
    tree_content, tree_entry, candidates, inputs
):
    _refused({**tree_content, 'inputs': inputs}, tree_entry, candidates, 'inputs')

@pytest.mark.parametrize(
    'kind, inputs_edit',
    [
        ('tree', lambda inputs: {
            **inputs, 'text_only': inputs['codebook']
        }),
        ('tree', lambda inputs: {
            **inputs, 'qcew': inputs['codebook']
        }),
        ('tree', lambda inputs: {
            'codebook': inputs['codebook'],
            'panel': inputs['codebook']
        }),
        ('tree', lambda inputs: {
            'Codebook': inputs['codebook']
        }),
        ('text-only', lambda inputs: {
            'codebook': inputs['codebook']
        }),
        ('text-only', lambda inputs: {
            'text_only': inputs['text_only']
        }),
        ('text-only', lambda inputs: {
            **inputs, 'qcew': inputs['codebook']
        }),
        ('smoothed', lambda inputs: {
            'codebook': inputs['codebook']
        }),
        ('smoothed', lambda inputs: {
            'text_only': inputs['text_only']
        }),
        ('smoothed', lambda inputs: {
            **inputs, 'selection_log': inputs['codebook']
        }),
        ('smoothed', lambda inputs: {
            'codebook': inputs['codebook'],
            'text': inputs['text_only']
        }),
    ],
    ids=[
        'tree: a text-only input', 'tree: a qcew input', 'tree: a panel input instead of text',
        'tree: a renamed codebook', 'text-only: no text-only input', 'text-only: no codebook',
        'text-only: a qcew input', 'smoothed: no text-only input', 'smoothed: no codebook',
        'smoothed: a selection log input', 'smoothed: a renamed text-only input'
    ],
)
def test_inputs_other_than_the_builders_are_refused(built, kind, inputs_edit):
    content = built.contents[kind]
    edited = {**content, 'inputs': inputs_edit(content['inputs'])}
    _refused(edited, DETERMINISTIC[kind], built.frozen, 'inputs')

INPUT_SHAPES = {
    'a bare string': 'the fixture codebook',
    'null': None,
    'a list': ['path', 'sha256'],
    'no sha256': {
        'path': 'x'
    },
    'no path': {
        'sha256': 'f' * 64
    },
    'empty': {},
    'an extra key': {
        'path': 'x',
        'sha256': 'f' * 64,
        'bytes': 1
    },
    'a non-string path': {
        'path': 1,
        'sha256': 'f' * 64
    },
    'a null path': {
        'path': None,
        'sha256': 'f' * 64
    },
    'a non-string sha256': {
        'path': 'x',
        'sha256': 7
    },
    'a null sha256': {
        'path': 'x',
        'sha256': None
    },
}

@pytest.mark.parametrize(
    'kind, name',
    [
        pytest.param(kind, name, id=f'{kind}: {name}') for kind in KINDS
        for name in BUILDER_INPUTS[kind]
    ],
)
@pytest.mark.parametrize('shape', INPUT_SHAPES.values(), ids=INPUT_SHAPES.keys())
def test_an_input_that_is_not_a_path_and_a_sha256_is_refused(built, kind, name, shape):
    content = built.contents[kind]
    edited = {**content, 'inputs': {**content['inputs'], name: shape}}
    _refused(edited, DETERMINISTIC[kind], built.frozen, 'inputs')

@pytest.mark.parametrize('kind', KINDS)
def test_a_codebook_that_is_not_the_lists_is_refused(built, kind):
    content = _with_input(built.contents[kind], 'codebook', sha256='f' * 64)
    _refused(content, DETERMINISTIC[kind], built.frozen, 'codebook')

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_text_only_table_that_is_not_the_lists_is_refused(built, kind):
    content = _with_input(built.contents[kind], 'text_only', sha256='f' * 64)
    _refused(content, DETERMINISTIC[kind], built.frozen, 'text_only')

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_text_only_sha_is_checked_against_the_lists_not_the_codebooks(built, kind):
    # The text-only input names the codebook's sha, which the list holds for the codebook alone
    content = _with_input(
        built.contents[kind],
        'text_only',
        sha256=built.contents[kind]['inputs']['codebook']['sha256']
    )
    _refused(content, DETERMINISTIC[kind], built.frozen, 'text_only')

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_codebook_sha_is_checked_against_the_lists_not_the_text_only_tables(built, kind):
    text_only_sha = built.contents[kind]['inputs']['text_only']['sha256']
    content = _with_input(built.contents[kind], 'codebook', sha256=text_only_sha)
    _refused(content, DETERMINISTIC[kind], built.frozen, 'codebook')

# -------------------------------------------------------------------------------------------------
# The checks' order
# -------------------------------------------------------------------------------------------------

def test_the_builder_is_checked_before_the_entries(tree_content, tree_entry, candidates):
    content = {**tree_content, 'builder': 'hgcn', 'qcew': QCEW}
    _refused(content, tree_entry, candidates, 'builder')

def test_the_entries_are_checked_before_the_inputs(tree_content, tree_entry, candidates):
    content = {**tree_content, 'qcew': QCEW, 'inputs': None}
    _refused(content, tree_entry, candidates, 'entries')

def test_the_inputs_are_checked_before_the_codebook(tree_content, tree_entry, candidates):
    inputs = {'codebook': {**CODEBOOK_INPUT, 'sha256': 'f' * 64}, 'qcew': QCEW}
    _refused({**tree_content, 'inputs': inputs}, tree_entry, candidates, 'inputs')

@pytest.mark.parametrize('shape', INPUT_SHAPES.values(), ids=INPUT_SHAPES.keys())
def test_an_inputs_shape_is_checked_before_the_codebooks_sha(built, shape):
    # The text-only input is malformed and the codebook is another: the shape is refused first
    content = _with_input(built.contents['smoothed'], 'codebook', sha256='f' * 64)
    content = {**content, 'inputs': {**content['inputs'], 'text_only': shape}}
    _refused(content, DETERMINISTIC['smoothed'], built.frozen, 'inputs')

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_the_codebook_is_checked_before_the_text_only_table(built, kind):
    content = _with_input(built.contents[kind], 'codebook', sha256='f' * 64)
    content = _with_input(content, 'text_only', sha256='e' * 64)
    _refused(content, DETERMINISTIC[kind], built.frozen, 'codebook')

# -------------------------------------------------------------------------------------------------
# An export's provenance
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('geometry', [False, True], ids=['stage-7', 'stage-8'])
@pytest.mark.parametrize(
    'edit',
    [
        {
            'qcew': QCEW
        }, {
            'panel': 'growth_seen'
        }, {
            'selection_log': 'logs/selection_log.jsonl'
        }, {
            'inputs': {
                'codebook': CODEBOOK_INPUT
            }
        }, {
            'method': 'a builder entry'
        }
    ],
    ids=['qcew', 'panel', 'selection log', 'inputs', 'a builder entry'],
)
def test_an_export_with_an_unknown_entry_is_refused(
    table, arm_spec, checkpoint, trained_entry, candidates, geometry, edit
):
    content = {**export_provenance(table, arm_spec, checkpoint, geometry=geometry), **edit}
    _refused(content, trained_entry, candidates, 'entries')

@pytest.mark.parametrize('geometry', [False, True], ids=['stage-7', 'stage-8'])
@pytest.mark.parametrize('key', sorted(EXPORT_ENTRIES))
def test_an_export_missing_an_entry_is_refused(
    table, arm_spec, checkpoint, trained_entry, candidates, geometry, key
):
    content = export_provenance(table, arm_spec, checkpoint, geometry=geometry)
    _refused(_without(content, key), trained_entry, candidates, 'entries')

@pytest.mark.parametrize(
    'edit, check',
    [
        (lambda c: {
            **c, 'contract': {
                **CONTRACT.model_dump(), 'bundle_id': 'other'
            }
        }, 'contract'),
        (
            lambda c: {
                **c, 'contract': {
                    **CONTRACT.model_dump(), 'contract_version': 'v0'
                }
            }, 'contract'
        ),
        (lambda c: {
            **c, 'qcew': QCEW
        }, 'entries'),
        (lambda c: _without(c, 'tokenizer'), 'entries'),
    ],
    ids=['another bundle_id', 'another contract_version', 'an extra qcew key', 'no tokenizer'],
)
def test_an_export_naming_another_contract_or_an_unknown_entry_is_refused(
    export_content, trained_entry, candidates, edit, check
):
    _refused(edit(export_content), trained_entry, candidates, check)

@pytest.mark.parametrize(
    'contract',
    [
        pytest.param({'contract_version': 'fixture-v1'}, id='no bundle_id'),
        pytest.param({'bundle_id': 'fixture-bundle'}, id='no contract_version'),
        pytest.param(
            {
                'bundle_id': 'fixture-v1',
                'contract_version': 'fixture-bundle'
            }, id='the two swapped'
        ),
        pytest.param(
            {
                'bundle_id': 'Fixture-bundle',
                'contract_version': 'fixture-v1'
            },
            id='bundle_id in another case'
        ),
        pytest.param(
            {
                'bundle_id': 'fixture-bundle',
                'contract_version': 'fixture-v1 '
            },
            id='contract_version with a space'
        ),
        pytest.param({
            'bundle_id': None,
            'contract_version': None
        }, id='both null'),
        pytest.param({}, id='empty'),
        pytest.param(None, id='null'),
        pytest.param('fixture-bundle', id='a string'),
        pytest.param(['fixture-bundle', 'fixture-v1'], id='a list'),
    ],
)
def test_an_export_whose_contract_is_not_the_lists_is_refused(
    export_content, trained_entry, candidates, contract
):
    _refused({**export_content, 'contract': contract}, trained_entry, candidates, 'contract')

@pytest.mark.parametrize('field', ['checkpoint', 'descriptions'])
@pytest.mark.parametrize('shape', INPUT_SHAPES.values(), ids=INPUT_SHAPES.keys())
def test_an_export_whose_input_file_is_not_a_path_and_a_sha256_is_refused(
    export_content, trained_entry, candidates, field, shape
):
    _refused({**export_content, field: shape}, trained_entry, candidates, 'inputs')

def test_the_contract_is_checked_before_the_exports_input_files(
    export_content, trained_entry, candidates
):
    content = {
        **export_content,
        'contract': {
            **CONTRACT.model_dump(), 'bundle_id': 'other'
        },
        'checkpoint': None,
    }
    _refused(content, trained_entry, candidates, 'contract')

def test_the_entries_are_checked_before_the_exports_contract(
    export_content, trained_entry, candidates
):
    content = {**export_content, 'qcew': QCEW, 'contract': None}
    _refused(content, trained_entry, candidates, 'entries')

@pytest.mark.parametrize('kind', KINDS)
def test_an_export_provenance_is_refused_for_a_deterministic_entry(
    built, export_content, candidates, kind
):
    _refused(export_content, DETERMINISTIC[kind], built.frozen, 'builder')

def test_a_builders_provenance_that_lost_its_builder_entry_reads_as_an_export(
    tree_content, tree_entry, candidates
):
    _refused(_without(tree_content, 'builder'), tree_entry, candidates, 'builder')

def test_the_entry_kind_is_checked_before_an_exports_entries(
    tree_entry, candidates, export_content
):
    _refused({**export_content, 'qcew': QCEW}, tree_entry, candidates, 'builder')

def test_every_refusal_names_the_entry_and_cites_the_rule(tree_content, tree_entry, candidates):
    with pytest.raises(ValueError) as info:
        check_candidate_provenance({**tree_content, 'qcew': QCEW}, tree_entry, candidates)
    message = str(info.value)
    assert message.startswith('tree-d3: ')
    assert message.endswith(RULE)
    assert 'qcew' in message

# -------------------------------------------------------------------------------------------------
# A trained candidate's import (4.4, 4.6; Task 6)
# -------------------------------------------------------------------------------------------------

# Not seed 0 alone: a check of the first seed only must not pass
LATER_SEED = 3

class Source(NamedTuple):
    '''A trained candidate's Stage 7/8 inputs (``trained_source``) and the list of its row.'''

    entry: CandidateEntry
    runs_root: Path
    frozen: CandidateList

@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / 'store')

@pytest.fixture
def shared_text_only(tmp_path):
    return write_text_only(tmp_path / 'text', codes=CODEBOOK)

@pytest.fixture
def source(store, shared_text_only, tmp_path) -> Source:
    entry, runs_root = trained_source(
        store, tmp_path / 'trained', 'euclidean-d3', text_only=shared_text_only
    )
    return Source(
        entry, runs_root, fixture_list([entry], text_only_sha256=sha256_file(shared_text_only))
    )

def _record(store, entry) -> ArmRecord:
    return read_record(store.root / entry.record, ArmRecord)

def _repinned(store, entry, record, name='edited') -> CandidateEntry:
    '''The entry pinned to ``record``, written into the store as ``records/<name>.json``.'''

    path = write_record(record, store.root / 'records' / f'{name}.json')
    return entry.model_copy(
        update={
            'record': f'records/{name}.json',
            'record_sha256': sha256_file(path)
        }
    )

def _with_runs(record, index, **update):
    runs = list(record.runs)
    runs[index] = runs[index].model_copy(update=update)
    return record.model_copy(update={'runs': runs})

BARE = {'training_run': None, 'checkpoint_epoch': None, 'monitor_records': []}

def _bare(record):
    '''The record with no run carrying selection evidence: D8's checks pass it.'''

    return record.model_copy(update={'runs': [run.model_copy(update=BARE) for run in record.runs]})

def _provenance(source: Source, seed: int) -> Path:
    '''Seed ``seed``'s export provenance, beside its table in its run directory.'''

    return source.runs_root / f'euclidean-d3-s{seed}' / f'euclidean-d3-{seed}_provenance.json'

def _edit_provenance(source: Source, seed: int, **entries) -> Path:
    path = _provenance(source, seed)
    path.write_text(json.dumps({**json.loads(path.read_text()), **entries}))
    return path

def _files(root: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob('*')) if path.is_file()
    }

def test_import_returns_the_spec_the_source_and_each_seeds_provenance(source, store, tmp_path):
    '''It reads no panel and writes nothing.'''

    before = _files(tmp_path)

    imported = import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

    record = store.root / 'records' / 'euclidean-d3.json'
    assert imported.entry == source.entry
    assert imported.source == read_record(record, ArmRecord)
    assert imported.source_ref == ArtifactRef(
        path='records/euclidean-d3.json', sha256=sha256_file(record), bytes=record.stat().st_size
    )
    assert imported.spec == economic_spec(source.entry, source=imported.source.spec)
    assert imported.provenance == {seed: _provenance(source, seed) for seed in range(5)}
    assert _files(tmp_path) == before

def test_import_refuses_a_deterministic_entry(source, store):
    entry = CandidateEntry(
        name='tree-d3', kind='tree', dimension=3, components=1, geometry='euclidean'
    )

    with pytest.raises(ValueError, match='^tree-d3: a tree candidate is read from its table'):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_source_record_missing_from_the_store(source, store):
    (store.root / source.entry.record).unlink()

    with pytest.raises(
        FileNotFoundError, match='^euclidean-d3: its source record records/euclidean-d3.json is'
    ):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

@pytest.mark.parametrize('edit', ['bytes', 'pin'], ids=['edited-after-pinning', 'another-pin'])
def test_import_refuses_a_source_record_that_is_not_its_pin(source, store, edit):
    '''The record still loads as the same arm: only its pin tells it from the pinned bytes.'''

    entry = source.entry
    if edit == 'bytes':
        path = store.root / entry.record
        path.write_text(read_record(path, ArmRecord).model_dump_json())
    else:
        entry = entry.model_copy(update={'record_sha256': 'f' * 64})

    with pytest.raises(ValueError, match=f'^euclidean-d3: .* is not its pin {entry.record_sha256}'):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_source_record_of_another_arm(source, store):
    entry = source.entry.model_copy(update={'arm': 'reference'})

    with pytest.raises(
        ValueError,
        match="^euclidean-d3: its source record holds the arm 'euclidean-d3', not "
        "'reference'"
    ):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_runs_d8s_checks_on_the_source(source, store):
    '''§4.4: the checkpoint is the earliest highest-MRR epoch (``check_arm``).'''

    record = _record(store, source.entry)
    epoch = record.runs[LATER_SEED].checkpoint_epoch + 1
    entry = _repinned(store, source.entry, _with_runs(record, LATER_SEED, checkpoint_epoch=epoch))

    with pytest.raises(
        ValueError,
        match=f'^euclidean-d3 seed {LATER_SEED}: the earliest epoch with the highest monitor MRR',
    ):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_source_whose_store_lacks_a_checkpoint(source, store):
    '''A partial store copy: the seed's checkpoint object is gone, and the refusal names it.'''

    checkpoint = store.resolve(_record(store, source.entry).runs[LATER_SEED].checkpoint)
    checkpoint.unlink()

    with pytest.raises(FileNotFoundError, match=re.escape(str(checkpoint))):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_source_whose_seeds_are_not_its_rows(source, store):
    entry = source.entry.model_copy(update={'seeds': [0, 1, 2, 3, 5]})

    with pytest.raises(
        ValueError,
        match='^euclidean-d3: its source holds the seeds \\[0, 1, 2, 3, 4\\], and its row names',
    ):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

@pytest.fixture
def six_seeds(store, shared_text_only, tmp_path) -> Source:
    '''A source with a sixth seed: five of its seeds are at least 5, and still not all of them.'''

    entry, runs_root = trained_source(
        store,
        tmp_path / 'trained',
        'euclidean-d3',
        text_only=shared_text_only,
        offsets=(*SEED_OFFSETS, 0.5),
    )
    return Source(
        entry, runs_root, fixture_list([entry], text_only_sha256=sha256_file(shared_text_only))
    )

@pytest.mark.parametrize(
    'seeds', [[0, 1, 2, 3, 4], [0, 1, 2, 4, 5]], ids=['the-last-seed', 'a-middle-seed']
)
def test_import_refuses_a_row_that_names_only_some_of_its_sources_seeds(six_seeds, store, seeds):
    '''The import would read all six of the source's seeds, and the frozen list names five.'''

    import_trained_candidate(six_seeds.entry, six_seeds.frozen, store, six_seeds.runs_root)
    entry = six_seeds.entry.model_copy(update={'seeds': seeds})

    with pytest.raises(
        ValueError,
        match='^euclidean-d3: its source holds the seeds \\[0, 1, 2, 3, 4, 5\\], and its row '
        f'names {re.escape(str(seeds))}',
    ):
        import_trained_candidate(entry, six_seeds.frozen, store, six_seeds.runs_root)

def test_import_refuses_a_row_that_names_more_seeds_than_its_source_holds(source, store):
    '''The frozen list would name seed 5, which the decision never reads.'''

    entry = source.entry.model_copy(update={'seeds': [0, 1, 2, 3, 4, 5]})

    with pytest.raises(
        ValueError,
        match='^euclidean-d3: its source holds the seeds \\[0, 1, 2, 3, 4\\], and its row names '
        '\\[0, 1, 2, 3, 4, 5\\]',
    ):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_reads_its_seeds_in_order_whatever_the_records_order(source, store):
    record = _record(store, source.entry)
    entry = _repinned(store, source.entry, record.model_copy(update={'runs': record.runs[::-1]}))

    imported = import_trained_candidate(entry, source.frozen, store, source.runs_root)

    assert list(imported.provenance) == [0, 1, 2, 3, 4]
    _provenance(source, 1).unlink()
    _provenance(source, LATER_SEED).unlink()
    with pytest.raises(FileNotFoundError, match=re.escape(str(_provenance(source, 1)))):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_source_run_without_selection_evidence(
    source, store, shared_text_only, tmp_path
):
    untrained = synthetic_arm(
        store,
        tmp_path / 'untrained',
        spec('euclidean-d3', dimension=3, geometry='euclidean'),
        {},
        codes=CODEBOOK,
        text_only_table=shared_text_only,
    )
    entry = _repinned(store, source.entry, untrained, 'untrained')

    with pytest.raises(ValueError, match='no selection evidence'):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_source_with_one_run_without_selection_evidence(source, store):
    record = _with_runs(_record(store, source.entry), LATER_SEED, **BARE)
    entry = _repinned(store, source.entry, record)

    with pytest.raises(
        ValueError, match=f'^euclidean-d3: the source seeds \\[{LATER_SEED}\\] name no training'
    ):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_source_whose_text_only_table_is_not_the_lists(source, store):
    frozen = source.frozen.model_copy(update={'text_only_sha256': 'f' * 64})

    with pytest.raises(ValueError, match="^euclidean-d3: its source's text-only table .* is not"):
        import_trained_candidate(source.entry, frozen, store, source.runs_root)

def test_import_refuses_a_source_whose_spec_is_not_its_rows(source, store):
    entry = source.entry.model_copy(update={'dimension': 4})

    with pytest.raises(ValueError, match="^euclidean-d3: its source spec 'euclidean-d3' has dim"):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_seed_with_no_export_provenance(source, store):
    path = _provenance(source, LATER_SEED)
    path.unlink()

    with pytest.raises(FileNotFoundError, match=re.escape(str(path))):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

def test_import_names_the_seed_whose_export_provenance_is_missing(source, store):
    '''The refusal names the candidate and the seed before the path, not the file system alone.'''

    path = _provenance(source, LATER_SEED)
    path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match=f'^euclidean-d3 seed {LATER_SEED}: {re.escape(str(path))}: no export provenance',
    ):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

def test_import_refuses_an_export_provenance_that_describes_another_table(source, store):
    path = _edit_provenance(source, LATER_SEED, table_sha256='f' * 64)

    with pytest.raises(
        ValueError, match=f'^{re.escape(str(path))} describes another file than the table'
    ):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

def test_import_refuses_an_export_provenance_outside_text_and_tree(source, store):
    _edit_provenance(source, LATER_SEED, qcew=QCEW)

    with pytest.raises(ValueError, match='^euclidean-d3: entries: .*qcew.*\\(Req 29'):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

def test_import_refuses_a_provenance_naming_another_checkpoint(source, store):
    checkpoint = {'path': 'checkpoints/euclidean-d3-s1/best.ckpt', 'sha256': 'f' * 64}
    _edit_provenance(source, 1, checkpoint=checkpoint)

    with pytest.raises(ValueError, match='^euclidean-d3 seed 1: .*another checkpoint'):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

D9_REFUSAL = f'^euclidean-d3 seed {LATER_SEED}: the table was exported from .*\\(D9\\)'
CHECKPOINT_REFUSAL = f'^euclidean-d3 seed {LATER_SEED}: .*another checkpoint'

@pytest.mark.parametrize(
    'entries, message',
    [
        pytest.param({'backbone': 'other-backbone'}, D9_REFUSAL, id='another backbone'),
        pytest.param({'revision': 'other'}, D9_REFUSAL, id='another revision'),
        pytest.param(
            {'descriptions': {
                'path': 'naics_descriptions.parquet',
                'sha256': 'f' * 64
            }},
            D9_REFUSAL,
            id='another descriptions sha256',
        ),
        pytest.param({'descriptions': QCEW}, D9_REFUSAL, id='a QCEW file as its descriptions'),
        pytest.param({'summaries': 'f' * 64}, D9_REFUSAL, id='another summaries sha256'),
        pytest.param({'summaries': QCEW['path']}, D9_REFUSAL, id='a QCEW path as its summaries'),
        pytest.param({'summaries': None}, D9_REFUSAL, id='no summaries'),
        pytest.param({'max_length': 32}, D9_REFUSAL, id='another window'),
        pytest.param(
            {'checkpoint': {
                'path': 'checkpoints/best.ckpt',
                'sha256': 'f' * 64
            }},
            CHECKPOINT_REFUSAL,
            id='another checkpoint sha256',
        ),
        pytest.param({'checkpoint': QCEW}, CHECKPOINT_REFUSAL, id='a QCEW file as its checkpoint'),
    ],
)
def test_import_refuses_an_export_provenance_of_other_inputs(source, store, entries, message):
    '''
    Spec :226-229, :235 (ledger ruling, Task 4): the rule reads these slots' shape, and the import
    their identity: the backbone, revision, descriptions, summaries and window are what the arm
    reads (D9), and the checkpoint is the seed's.
    '''

    _edit_provenance(source, LATER_SEED, **entries)

    with pytest.raises(ValueError, match=message):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

@pytest.mark.parametrize(
    'entries, message',
    [
        pytest.param(
            {
                'table_sha256': 'f' * 64,
                'qcew': QCEW
            },
            'describes another file than the table',
            id='the table before the rule',
        ),
        pytest.param(
            {
                'qcew': QCEW,
                'checkpoint': {
                    'path': 'best.ckpt',
                    'sha256': 'f' * 64
                }
            },
            '^euclidean-d3: entries: ',
            id='the rule before the checkpoint',
        ),
        pytest.param(
            {
                'checkpoint': {
                    'path': 'best.ckpt',
                    'sha256': 'f' * 64
                },
                'backbone': 'other'
            },
            CHECKPOINT_REFUSAL,
            id='the checkpoint before d9',
        ),
    ],
)
def test_a_seeds_export_provenance_is_checked_in_order(source, store, entries, message):
    _edit_provenance(source, LATER_SEED, **entries)

    with pytest.raises(ValueError, match=message):
        import_trained_candidate(source.entry, source.frozen, store, source.runs_root)

def _broken(
    source, store, *, record=None, seeds=None, text_only=None, dimension=None, arm=None, pin=None
):
    '''The import's inputs, each given edit applied: a record, a row or a list.'''

    entry = source.entry
    if record is not None:
        entry = _repinned(store, entry, record(_record(store, entry)))
    update = {
        field: value
        for field, value in (
            ('seeds', seeds), ('dimension', dimension), ('arm', arm), ('record_sha256', pin)
        ) if value is not None
    }
    frozen = source.frozen
    if text_only is not None:
        frozen = frozen.model_copy(update={'text_only_sha256': text_only})
    return entry.model_copy(update=update), frozen

@pytest.mark.parametrize(
    'edits, message',
    [
        pytest.param(
            {
                'pin': 'f' * 64,
                'arm': 'reference'
            }, 'is not its pin', id='the pin before the arm'
        ),
        pytest.param(
            {
                'record': lambda record: _with_runs(record, 2, checkpoint_epoch=2),
                'seeds': [0, 1, 2, 3, 5],
            },
            'the earliest epoch with the highest monitor MRR',
            id='d8 before the seeds',
        ),
        pytest.param(
            {
                'record': _bare,
                'seeds': [0, 1, 2, 3, 5]
            },
            'its source holds the seeds',
            id='the seeds before the evidence',
        ),
        pytest.param(
            {
                'record': _bare,
                'text_only': 'f' * 64
            },
            'no selection evidence',
            id='the evidence before the text-only table',
        ),
        pytest.param(
            {
                'text_only': 'f' * 64,
                'dimension': 4
            },
            'the shared text-only table',
            id='the text-only table before the spec',
        ),
    ],
)
def test_the_import_checks_its_source_in_order(source, store, edits, message):
    entry, frozen = _broken(source, store, **edits)

    with pytest.raises(ValueError, match=message):
        import_trained_candidate(entry, frozen, store, source.runs_root)

def test_the_import_checks_its_source_and_spec_before_any_seed(source, store):
    _provenance(source, 0).unlink()
    entry = source.entry.model_copy(update={'dimension': 4})

    with pytest.raises(ValueError, match="its source spec 'euclidean-d3' has dimension"):
        import_trained_candidate(entry, source.frozen, store, source.runs_root)
