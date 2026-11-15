'''
The deterministic economic builders (``tree_model``; spec 4.2, economic-candidates).

The text-only, tree and smoothed candidates are written from the fixture codebook and its
text-only table. Classical scaling is checked against hand-worked spectra and the fixture tree.
'''

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from naics_embedder.panels.regressor import ArmTables, table_fingerprint
from naics_embedder.panels.text_only import pca_reduce, provenance_path, text_only_matrix
from naics_embedder.supervision.artifacts import sha256_file
from naics_embedder.tree_model import builders, scaling
from naics_embedder.tree_model.builders import (
    BUILDER_ENTRIES,
    BUILDER_INPUTS,
    KINDS,
    METHODS,
    build_candidate,
)
from naics_embedder.tree_model.scaling import CUT_GAP, classical_scaling
from naics_embedder.tree_model.smoothing import lineage_weights, smooth_over_lineage
from naics_embedder.utils.naics_hierarchy import code_lineage, tree_distance_matrix
from tests.fixtures.decision import write_text_only
from tests.fixtures.economic import write_codebook
from tests.fixtures.regressor_panel import CODEBOOK

CHAIN = ('11', '111', '1111')  # D* = [[0,1,2],[1,0,1],[2,1,0]]; B's spectrum 2, 0, 0
STAR = ('11', '111', '112', '113')  # spectrum 2, 2, 0, -0.25: not Euclidean
DIMENSION = 3

# -------------------------------------------------------------------------------------------------
# Classical scaling
# -------------------------------------------------------------------------------------------------

def test_classical_scaling_recovers_a_chain():
    coordinates, eigenvalues = classical_scaling(tree_distance_matrix(CHAIN).astype(float), 1)
    assert np.abs(coordinates[:, 0]) == pytest.approx([1.0, 0.0, 1.0], abs=1e-12)
    assert eigenvalues == pytest.approx([2.0, 0.0], abs=1e-12)

def test_each_column_s_largest_entry_is_positive():
    coordinates, _ = classical_scaling(tree_distance_matrix(CODEBOOK).astype(float), 3)
    for column in coordinates.T:
        assert column[np.argmax(np.abs(column))] > 0

def test_classical_scaling_is_the_rank_d_part_of_the_double_centred_matrix():
    distances = tree_distance_matrix(CODEBOOK).astype(float)
    coordinates, _ = classical_scaling(distances, 3)
    n = len(CODEBOOK)
    centring = np.eye(n) - 1.0 / n
    values, vectors = np.linalg.eigh(-0.5 * centring @ (distances**2) @ centring)
    top = vectors[:, -3:] * values[-3:]
    assert coordinates @ coordinates.T == pytest.approx(top @ vectors[:, -3:].T, abs=1e-9)

@pytest.mark.parametrize(
    'codes, dimension, message',
    [
        (CODEBOOK, 4, 'ties'),  # λ4 = λ5 = 135.97…, relative gap 4e-16 (verified 2026-11-14)
        (STAR, 1, 'ties'),
        (STAR, 3, 'is not positive'),
        (CHAIN, 2, 'is not positive'),
    ]
)
def test_an_arbitrary_or_non_positive_cut_is_refused(codes, dimension, message):
    with pytest.raises(ValueError, match=message):
        classical_scaling(tree_distance_matrix(codes).astype(float), dimension)

def _points(seed: int, rows: int = 9, width: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, width))

def _distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

def test_classical_scaling_reproduces_the_distances_of_a_euclidean_configuration():
    points = _points(1)
    coordinates, eigenvalues = classical_scaling(_distances(points), 3)
    assert _distances(coordinates) == pytest.approx(_distances(points), abs=1e-9)
    centred = points - points.mean(axis=0)
    assert eigenvalues[:3] == pytest.approx(np.linalg.svd(centred, compute_uv=False)**2, abs=1e-9)

def test_a_cut_after_every_positive_eigenvalue_but_one_is_kept():
    # an equilateral triangle: B = J / 2, so the spectrum is 1/2, 1/2, 0 and d = n - 1 = 2
    triangle = 1.0 - np.eye(3)
    coordinates, eigenvalues = classical_scaling(triangle, 2)
    assert coordinates.shape == (3, 2)
    assert eigenvalues == pytest.approx([0.5, 0.5, 0.0], abs=1e-12)
    assert _distances(coordinates) == pytest.approx(triangle, abs=1e-12)

def test_two_codes_give_one_coordinate_each():
    coordinates, eigenvalues = classical_scaling(
        tree_distance_matrix(('11', '23')).astype(float), 1
    )
    assert np.abs(coordinates[:, 0]) == pytest.approx([1.0, 1.0], abs=1e-12)
    assert eigenvalues == pytest.approx([2.0, 0.0], abs=1e-12)

@pytest.mark.parametrize('dimension', [0, -1, 3, 4])
def test_a_dimension_outside_one_to_n_minus_one_is_refused(dimension):
    with pytest.raises(ValueError, match='dimension'):
        classical_scaling(_distances(_points(2, rows=3)), dimension)

def test_the_eigenvalues_are_the_d_plus_one_leading_in_descending_order():
    _, eigenvalues = classical_scaling(tree_distance_matrix(CODEBOOK).astype(float), 3)
    assert eigenvalues.shape == (4, )
    assert eigenvalues == pytest.approx([554.89480068, 434.44998578, 164.35243481, 135.97399396])

def test_a_cut_at_a_negative_eigenvalue_is_not_positive():
    # D* is not Euclidean: the last of B's n - 1 possible cuts lies in its negative eigenvalues
    distances = tree_distance_matrix(CODEBOOK).astype(float)
    with pytest.raises(ValueError, match='is not positive'):
        classical_scaling(distances, len(CODEBOOK) - 1)

def test_the_cut_gap_is_a_millionth():
    assert CUT_GAP == 1e-6

@pytest.mark.parametrize('seed', range(12))
def test_the_sign_rule_holds_on_any_configuration(seed):
    coordinates, _ = classical_scaling(_distances(_points(seed)), 2)
    for column in coordinates.T:
        assert column[np.argmax(np.abs(column))] > 0

def _with_spectrum(monkeypatch, values, vectors=None):
    '''
    ``numpy.linalg.eigh`` reporting these eigenvalues (ascending) and, by default, identity
    eigenvectors: the cuts' exact boundaries cannot be reached through a real decomposition.
    '''

    ascending = np.array(sorted(values), dtype=np.float64)
    reported = np.eye(len(ascending)) if vectors is None else np.asarray(vectors, dtype=np.float64)
    monkeypatch.setattr(scaling.np.linalg, 'eigh', lambda matrix: (ascending, reported))

def test_the_sign_rule_takes_the_first_of_tied_extremes(monkeypatch):
    # the leading eigenvector has four entries of equal magnitude: the first must be positive
    vectors = np.eye(4)
    vectors[:, 3] = [-0.5, 0.5, -0.5, 0.5]
    _with_spectrum(monkeypatch, [0.0, 0.0, 1.0, 4.0], vectors)
    coordinates, _ = classical_scaling(np.zeros((4, 4)), 1)
    assert coordinates[:, 0] == pytest.approx([1.0, -1.0, 1.0, -1.0])

def test_a_d_th_eigenvalue_at_the_cut_gap_of_the_first_is_not_positive(monkeypatch):
    _with_spectrum(monkeypatch, [0.0, CUT_GAP * 1.0, 1.0])
    with pytest.raises(ValueError, match='is not positive'):
        classical_scaling(np.zeros((3, 3)), 2)

def test_a_d_th_eigenvalue_just_above_the_cut_gap_of_the_first_is_positive(monkeypatch):
    _with_spectrum(monkeypatch, [0.0, np.nextafter(CUT_GAP * 1.0, 1.0), 1.0])
    coordinates, _ = classical_scaling(np.zeros((3, 3)), 2)
    assert coordinates.shape == (3, 2)

def test_the_positivity_cut_is_relative_to_the_first_eigenvalue(monkeypatch):
    _with_spectrum(monkeypatch, [0.0, 1.0, 1e6])
    with pytest.raises(ValueError, match='is not positive'):
        classical_scaling(np.zeros((3, 3)), 2)

def test_a_relative_gap_of_exactly_the_cut_gap_is_not_a_tie(monkeypatch):
    # 1e6 and 999999.0 differ by exactly 1e6 * 1e-6 in floating point
    assert (1e6 - 999999.0) / 1e6 == CUT_GAP
    _with_spectrum(monkeypatch, [0.0, 999999.0, 1e6])
    coordinates, eigenvalues = classical_scaling(np.zeros((3, 3)), 1)
    assert coordinates.shape == (3, 1)
    assert eigenvalues == pytest.approx([1e6, 999999.0])

def test_a_relative_gap_below_the_cut_gap_is_a_tie(monkeypatch):
    _with_spectrum(monkeypatch, [0.0, 999999.5, 1e6])
    with pytest.raises(ValueError, match='ties'):
        classical_scaling(np.zeros((3, 3)), 1)

def test_the_tie_gap_is_divided_by_the_d_th_eigenvalue(monkeypatch):
    # 1e6 and 999999.0000005 differ by 5e-7 of the d-th eigenvalue but 1.0000005e-6 of the next
    _with_spectrum(monkeypatch, [0.0, 999999.0000005, 1e6])
    with pytest.raises(ValueError, match='ties'):
        classical_scaling(np.zeros((3, 3)), 1)

@pytest.mark.parametrize('scale', [1e-5, 1e5])
def test_the_cuts_do_not_depend_on_the_scale_of_the_distances(scale):
    distances = scale * tree_distance_matrix(CHAIN).astype(float)
    coordinates, _ = classical_scaling(distances, 1)
    assert np.abs(coordinates[:, 0]) / scale == pytest.approx([1.0, 0.0, 1.0], abs=1e-9)

def test_a_small_d_th_eigenvalue_is_no_tie_when_the_next_is_smaller_still(monkeypatch):
    _with_spectrum(monkeypatch, [0.0, 0.0, 1e-9])
    assert classical_scaling(np.zeros((3, 3)), 1)[0].shape == (3, 1)

def test_a_negative_next_eigenvalue_is_no_tie(monkeypatch):
    _with_spectrum(monkeypatch, [-3.0, 0.0, 2.0])
    assert classical_scaling(np.zeros((3, 3)), 1)[0].shape == (3, 1)

# -------------------------------------------------------------------------------------------------
# Lineage smoothing
# -------------------------------------------------------------------------------------------------

def test_lineage_weights_halve_up_the_tree_and_sum_to_one():
    assert lineage_weights(3) == pytest.approx([1 / 7, 2 / 7, 4 / 7])
    assert lineage_weights(1) == pytest.approx([1.0])

def test_lineage_weights_of_depth_two_and_five():
    assert lineage_weights(2) == pytest.approx([1 / 3, 2 / 3])
    assert lineage_weights(5) == pytest.approx(np.array([1, 2, 4, 8, 16]) / 31)

@pytest.mark.parametrize('depth', [0, -1])
def test_lineage_weights_refuse_a_depth_below_one(depth):
    with pytest.raises(ValueError, match='depth'):
        lineage_weights(depth)

def test_smoothing_averages_each_code_over_its_lineage():
    smoothed = smooth_over_lineage(CHAIN, np.array([[7.0, 0.0], [0.0, 7.0], [0.0, 0.0]]))
    assert smoothed == pytest.approx(np.array([[7.0, 0.0], [7 / 3, 14 / 3], [1.0, 2.0]]))

def test_a_code_whose_ancestor_is_missing_is_refused():
    with pytest.raises(ValueError, match='1111'):
        smooth_over_lineage(('11', '1111'), np.zeros((2, 2)))

def test_the_refusal_names_the_missing_ancestor_too():
    with pytest.raises(ValueError, match=r'code 1111 has an ancestor .*: 111$'):
        smooth_over_lineage(('11', '112', '1111'), np.zeros((3, 2)))

def test_a_missing_sector_is_refused():
    with pytest.raises(ValueError, match=r'code 111 has an ancestor .*: 11$'):
        smooth_over_lineage(('111', '1111'), np.zeros((2, 2)))

def test_smoothing_leaves_a_sector_as_it_is_and_weights_a_six_digit_code_by_its_lineage():
    vectors = np.random.default_rng(3).normal(size=(len(CODEBOOK), 4))
    smoothed = smooth_over_lineage(CODEBOOK, vectors)
    row = {code: vectors[index] for index, code in enumerate(CODEBOOK)}
    position = {code: index for index, code in enumerate(CODEBOOK)}
    assert smoothed[position['11']] == pytest.approx(row['11'])
    expected = sum(
        weight * row[code]
        for weight, code in zip([1 / 31, 2 / 31, 4 / 31, 8 / 31, 16 / 31], code_lineage('238110'))
    )
    assert smoothed[position['238110']] == pytest.approx(expected)

def test_smoothing_follows_a_combined_sector_to_its_first_code():
    # 321's parent is the combined sector 31-33, keyed by 31
    codes = ('31', '321', '3211')
    vectors = np.array([[1.0], [10.0], [100.0]])
    assert smooth_over_lineage(codes, vectors)[2] == pytest.approx([(1 + 20 + 400) / 7])

def test_smoothing_does_not_depend_on_the_codes_order():
    vectors = np.random.default_rng(4).normal(size=(len(CODEBOOK), 3))
    order = np.random.default_rng(5).permutation(len(CODEBOOK))
    shuffled = smooth_over_lineage([CODEBOOK[i] for i in order], vectors[order])
    assert shuffled == pytest.approx(smooth_over_lineage(CODEBOOK, vectors)[order])

@pytest.mark.parametrize('order', ['C', 'F'])
def test_smoothing_returns_the_same_bytes_in_c_order_whatever_the_input_layout(order):
    vectors = np.random.default_rng(7).normal(size=(len(CODEBOOK), 6))
    smoothed = smooth_over_lineage(CODEBOOK, np.array(vectors, order=order))
    assert smoothed.flags['C_CONTIGUOUS']
    assert np.array_equal(smoothed, smooth_over_lineage(CODEBOOK, np.array(vectors, order='C')))

def test_smoothing_refuses_vectors_of_other_codes():
    with pytest.raises(ValueError, match='rows'):
        smooth_over_lineage(CHAIN, np.zeros((2, 2)))

def test_smoothing_refuses_a_repeated_code():
    with pytest.raises(ValueError, match='repeat'):
        smooth_over_lineage(('11', '11'), np.zeros((2, 2)))

# -------------------------------------------------------------------------------------------------
# Builders
# -------------------------------------------------------------------------------------------------

@pytest.fixture
def codebook(tmp_path):
    return write_codebook(tmp_path / 'naics_codebook.parquet')

@pytest.fixture
def text_only(tmp_path):
    return write_text_only(tmp_path / 'text', codes=CODEBOOK)

def _read_provenance(table_path: Path) -> dict:
    return json.loads(provenance_path(table_path).read_text())

def _built(kind, tmp_path, codebook, text_only, name='candidate', dimension=DIMENSION):
    '''Build the kind into ``tmp_path / name``, reading the text-only table when it reads one.'''

    inputs = text_only if 'text_only' in BUILDER_INPUTS[kind] else None
    return build_candidate(
        kind,
        dimension,
        codebook,
        tmp_path / name / f'{kind}-d{dimension}.parquet',
        text_only=inputs
    )

def _coordinates(path: Path) -> np.ndarray:
    return np.array(pl.read_parquet(path).select(['e0', 'e1', 'e2']).to_numpy(), dtype=np.float64)

def test_the_builder_constants_are_what_the_provenance_rule_reads():
    common = {
        'builder', 'method', 'dimension', 'codes', 'inputs', 'table_sha256', 'matrix_fingerprint',
        'library_versions'
    }
    assert KINDS == ('text-only', 'tree', 'smoothed')
    assert BUILDER_INPUTS == {
        'text-only': ('codebook', 'text_only'),
        'tree': ('codebook', ),
        'smoothed': ('codebook', 'text_only'),
    }
    assert BUILDER_ENTRIES == {
        'text-only': frozenset(common),
        'tree': frozenset(common | {'eigenvalues'}),
        'smoothed': frozenset(common),
    }
    assert set(METHODS) == set(KINDS)
    assert all(isinstance(text, str) and text.endswith('.') for text in METHODS.values())
    assert 'halving' in METHODS['smoothed']

@pytest.mark.parametrize('kind', KINDS)
def test_each_builder_writes_the_export_form_in_codebook_order(kind, tmp_path, codebook, text_only):
    path = _built(kind, tmp_path, codebook, text_only)
    table = pl.read_parquet(path)
    assert table.columns == ['code', 'index', 'level', 'e0', 'e1', 'e2']
    assert table.schema == {
        'code': pl.Utf8,
        'index': pl.Int64,
        'level': pl.Int64,
        'e0': pl.Float64,
        'e1': pl.Float64,
        'e2': pl.Float64,
    }
    frame = pl.read_parquet(codebook)
    assert table.get_column('code').to_list() == frame.get_column('code').to_list()
    assert table.get_column('index').to_list() == frame.get_column('code_id').to_list()
    assert table.get_column('level').to_list() == [len(code) for code in CODEBOOK]
    provenance = _read_provenance(path)
    assert set(provenance) == BUILDER_ENTRIES[kind]
    assert provenance['builder'] == kind
    assert provenance['method'] == METHODS[kind]
    assert provenance['dimension'] == DIMENSION
    assert provenance['codes'] == len(CODEBOOK)
    assert set(provenance['inputs']) == set(BUILDER_INPUTS[kind])
    assert provenance['inputs']['codebook'] == {
        'path': str(codebook),
        'sha256': sha256_file(codebook)
    }
    if kind != 'tree':
        assert provenance['inputs']['text_only'] == {
            'path': str(text_only),
            'sha256': sha256_file(text_only)
        }
    assert provenance['table_sha256'] == sha256_file(path)
    assert provenance['matrix_fingerprint'] == table_fingerprint(table)
    assert set(provenance['library_versions']) == {'numpy', 'scikit-learn', 'polars'}
    assert provenance_path(path).read_text() == json.dumps(
        provenance, indent=2, sort_keys=True
    ) + '\n'

@pytest.mark.parametrize('kind', KINDS)
def test_each_builder_writes_the_same_bytes_twice(kind, tmp_path, codebook, text_only):
    first = _built(kind, tmp_path, codebook, text_only, name='first')
    second = _built(kind, tmp_path, codebook, text_only, name='second')
    assert first != second
    assert sha256_file(first) == sha256_file(second)
    assert sha256_file(provenance_path(first)) == sha256_file(provenance_path(second))

def test_the_text_only_candidate_is_reduced_as_a_read_reduces_it(tmp_path, codebook, text_only):
    path = _built('text-only', tmp_path, codebook, text_only)
    arm = ArmTables.from_tables(pl.read_parquet(path), pl.read_parquet(text_only))
    assert np.array_equal(_coordinates(path), arm.text_only)

def test_the_smoothed_candidate_reduces_lineage_means(tmp_path, codebook, text_only):
    path = _built('smoothed', tmp_path, codebook, text_only)
    codes, vectors = text_only_matrix(pl.read_parquet(text_only))
    assert tuple(codes) == CODEBOOK
    expected = pca_reduce(smooth_over_lineage(codes, vectors), 3)
    assert np.array_equal(_coordinates(path), expected)

def test_the_tree_candidate_is_the_classical_scaling_of_d_star(tmp_path, codebook, text_only):
    path = _built('tree', tmp_path, codebook, text_only)
    coordinates, eigenvalues = classical_scaling(tree_distance_matrix(CODEBOOK).astype(float), 3)
    assert np.array_equal(_coordinates(path), coordinates)
    assert _read_provenance(path)['eigenvalues'] == eigenvalues.tolist()
    assert len(_read_provenance(path)['eigenvalues']) == DIMENSION + 1

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_builder_refuses_text_only_inputs_over_other_codes(kind, tmp_path, codebook):
    other = write_text_only(tmp_path / 'other', codes=CODEBOOK[1:])
    output = tmp_path / 'out' / 'candidate.parquet'
    with pytest.raises(ValueError, match='codes'):
        build_candidate(kind, 3, codebook, output, text_only=other)
    assert not output.parent.exists()

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_builder_refuses_text_only_inputs_over_more_codes(kind, tmp_path, codebook):
    other = write_text_only(tmp_path / 'other', codes=(*CODEBOOK, '999999'))
    output = tmp_path / 'out' / 'candidate.parquet'
    with pytest.raises(ValueError, match='codes'):
        build_candidate(kind, 3, codebook, output, text_only=other)
    assert not output.parent.exists()

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_builder_refuses_text_only_inputs_that_repeat_a_code(kind, tmp_path, codebook):
    # the same codes as the codebook's, one of them twice
    other = write_text_only(tmp_path / 'other', codes=(*CODEBOOK, CODEBOOK[0]))
    output = tmp_path / 'out' / 'candidate.parquet'
    with pytest.raises(ValueError, match='codes'):
        build_candidate(kind, 3, codebook, output, text_only=other)
    assert not output.parent.exists()

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_builder_refuses_text_only_inputs_with_another_code_of_the_same_count(
    kind, tmp_path, codebook
):
    other = write_text_only(tmp_path / 'other', codes=(*CODEBOOK[1:], '999999'))
    output = tmp_path / 'out' / 'candidate.parquet'
    with pytest.raises(ValueError, match='codes'):
        build_candidate(kind, 3, codebook, output, text_only=other)
    assert not output.parent.exists()

def test_a_builder_refuses_text_only_input_for_the_tree_and_none_for_text(
    tmp_path, codebook, text_only
):
    output = tmp_path / 'out' / 'candidate.parquet'
    with pytest.raises(ValueError, match='tree'):
        build_candidate('tree', 3, codebook, output, text_only=text_only)
    for kind in ('text-only', 'smoothed'):
        with pytest.raises(ValueError, match=kind):
            build_candidate(kind, 3, codebook, output)
    assert not output.parent.exists()

def test_a_builder_refuses_an_unknown_kind(tmp_path, codebook, text_only):
    output = tmp_path / 'out' / 'candidate.parquet'
    with pytest.raises(ValueError, match='unknown'):
        build_candidate('graph', 3, codebook, output, text_only=text_only)
    assert not output.parent.exists()

@pytest.mark.parametrize('kind', KINDS)
def test_a_builder_writes_no_table_over_an_existing_one(kind, tmp_path, codebook, text_only):
    output = tmp_path / f'{kind}.parquet'
    output.write_bytes(b'an earlier table')
    inputs = text_only if 'text_only' in BUILDER_INPUTS[kind] else None
    with pytest.raises(FileExistsError):
        build_candidate(kind, 3, codebook, output, text_only=inputs)
    assert output.read_bytes() == b'an earlier table'
    assert not provenance_path(output).exists()

def test_a_tied_cut_writes_nothing(tmp_path, codebook):
    output = tmp_path / 'out' / 'tree-d4.parquet'
    with pytest.raises(ValueError, match='ties'):
        build_candidate('tree', 4, codebook, output)
    assert not output.parent.exists()

@pytest.mark.parametrize('kind', ['text-only', 'smoothed'])
def test_a_dimension_wider_than_the_text_writes_nothing(kind, tmp_path, codebook, text_only):
    output = tmp_path / 'out' / f'{kind}-d6.parquet'
    with pytest.raises(ValueError, match='components'):
        build_candidate(kind, 6, codebook, output, text_only=text_only)
    assert not output.parent.exists()

@pytest.mark.parametrize('kind', KINDS)
def test_a_table_no_panel_could_read_writes_nothing(
    kind, tmp_path, codebook, text_only, monkeypatch
):

    def refuse(table):
        raise ValueError('the coordinate table has constant columns')

    monkeypatch.setattr(builders, 'table_fingerprint', refuse)
    output = tmp_path / 'out' / f'{kind}.parquet'
    inputs = text_only if 'text_only' in BUILDER_INPUTS[kind] else None
    with pytest.raises(ValueError, match='constant'):
        build_candidate(kind, 3, codebook, output, text_only=inputs)
    assert not output.parent.exists()

def test_a_builder_makes_the_output_directory(tmp_path, codebook):
    output = tmp_path / 'a' / 'b' / 'tree-d3.parquet'
    assert build_candidate('tree', 3, codebook, output) == output
    assert output.exists() and provenance_path(output).exists()

@pytest.mark.parametrize('kind', KINDS)
def test_each_builder_writes_one_coordinate_at_dimension_one(kind, tmp_path, codebook, text_only):
    path = _built(kind, tmp_path, codebook, text_only, dimension=1)
    table = pl.read_parquet(path)
    assert table.columns == ['code', 'index', 'level', 'e0']
    provenance = _read_provenance(path)
    assert provenance['dimension'] == 1
    if kind == 'tree':
        assert len(provenance['eigenvalues']) == 2

def test_write_codebook_writes_code_id_and_code_in_the_given_order(tmp_path):
    path = write_codebook(tmp_path / 'sub' / 'codebook.parquet', ('23', '11', '111'))
    frame = pl.read_parquet(path)
    assert frame.schema == {'code_id': pl.Int32, 'code': pl.Utf8}
    assert frame.rows() == [(0, '23'), (1, '11'), (2, '111')]

# -------------------------------------------------------------------------------------------------
# Row order
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('kind', KINDS)
def test_the_index_is_the_codebook_s_code_id(kind, tmp_path, text_only):
    path = tmp_path / 'offset_codebook.parquet'
    ids = [10 + 3 * row for row in range(len(CODEBOOK))]
    pl.DataFrame(
        {
            'code_id': ids,
            'code': list(CODEBOOK)
        }, schema={
            'code_id': pl.Int32,
            'code': pl.Utf8
        }
    ).write_parquet(path)
    table = pl.read_parquet(_built(kind, tmp_path, path, text_only))
    assert table.get_column('index').to_list() == ids

def _shuffled_inputs(tmp_path):
    '''
    A codebook whose ``code_id`` order is neither its row order nor sorted, and a text-only table
    in a third order. Returns the codebook path, the codes in ``code_id`` order and the table path.
    '''

    rng = np.random.default_rng(6)
    by_id = [CODEBOOK[i] for i in rng.permutation(len(CODEBOOK))]
    rows = rng.permutation(len(CODEBOOK))
    path = tmp_path / 'shuffled_codebook.parquet'
    pl.DataFrame(
        {
            'code_id': [int(i) for i in rows],
            'code': [by_id[i] for i in rows]
        },
        schema={
            'code_id': pl.Int32,
            'code': pl.Utf8
        },
    ).write_parquet(path)
    table_order = tuple(CODEBOOK[i] for i in rng.permutation(len(CODEBOOK)))
    return path, tuple(by_id), write_text_only(tmp_path / 'text', codes=table_order)

@pytest.mark.parametrize('kind', KINDS)
def test_each_builder_follows_the_codebook_s_code_id_order(kind, tmp_path):
    codebook_path, by_id, text_path = _shuffled_inputs(tmp_path)
    path = _built(kind, tmp_path, codebook_path, text_path)
    table = pl.read_parquet(path)
    assert table.get_column('code').to_list() == list(by_id)
    assert table.get_column('index').to_list() == list(range(len(by_id)))
    assert table.get_column('level').to_list() == [len(code) for code in by_id]

def test_the_text_only_candidate_is_reduced_in_the_text_only_table_s_order(tmp_path):
    codebook_path, _, text_path = _shuffled_inputs(tmp_path)
    path = _built('text-only', tmp_path, codebook_path, text_path)
    arm = ArmTables.from_tables(pl.read_parquet(path), pl.read_parquet(text_path))
    assert np.array_equal(_coordinates(path), arm.text_only)

def test_the_smoothed_candidate_smooths_over_the_codebook_s_order(tmp_path):
    codebook_path, by_id, text_path = _shuffled_inputs(tmp_path)
    path = _built('smoothed', tmp_path, codebook_path, text_path)
    codes, vectors = text_only_matrix(pl.read_parquet(text_path))
    position = {code: row for row, code in enumerate(codes)}
    ordered = vectors[[position[code] for code in by_id]]
    expected = pca_reduce(smooth_over_lineage(by_id, ordered), 3)
    assert np.array_equal(_coordinates(path), expected)

def test_the_tree_candidate_scales_the_codebook_s_order(tmp_path):
    codebook_path, by_id, _ = _shuffled_inputs(tmp_path)
    path = _built('tree', tmp_path, codebook_path, None)
    expected, _ = classical_scaling(tree_distance_matrix(by_id).astype(float), 3)
    assert np.array_equal(_coordinates(path), expected)
