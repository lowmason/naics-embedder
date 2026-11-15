'''
The frozen economic candidate list (spec economic-candidates, 4.1; plan 16, Task 3).

The committed list is read as it stands. The entry and list rules are exercised over small lists
built here, and each kind's ArmSpec is checked against the text-only table's provenance fields.
'''

from pathlib import Path

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
from naics_embedder.decision.records import ArmSpec
from naics_embedder.decision.store import provenance_fields
from tests.fixtures.decision import spec

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
    candidates = CandidateList.model_validate(
        _list(
            _deterministic('smoothed', 32, name='x'),
            _deterministic('text-only', 8, name='twin-8'),
            _deterministic('smoothed', 8, name='y'),
            _deterministic('text-only', 32, name='twin-32'),
            _deterministic('tree', 8),
        )
    )
    assert candidates.propagation_pairs() == [('x', 'twin-32'), ('y', 'twin-8')]

def test_a_list_without_a_smoothed_candidate_has_no_propagation_pair():
    candidates = CandidateList.model_validate(
        _list(_deterministic('text-only', 8), _deterministic('tree', 8))
    )
    assert candidates.propagation_pairs() == []

# -------------------------------------------------------------------------------------------------
# Loading and table paths
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('as_text', [False, True])
def test_a_list_loads_from_a_yaml_file(tmp_path, as_text):
    entries = [_deterministic('text-only', 8), _deterministic('smoothed', 8), _trained('t-d8')]
    path = tmp_path / 'candidates.yaml'
    path.write_text(yaml.safe_dump(_list(*entries)))
    loaded = load_candidate_list(str(path) if as_text else path)
    assert loaded == CandidateList.model_validate(_list(*entries))

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
    row = _deterministic('smoothed', 24, name='odd', components=3)
    built = economic_spec(_entry(row), text_only=_text_only())
    assert (built.name, built.components, built.dimension, built.geometry) == (
        'odd', 3, 24, 'euclidean'
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
