'''
The relatedness panel's arms (plan 21, Task 5; spec relatedness-artifact, 4.4 and 4.5).

An arm returns the float64 similarity matrix of anchors by candidates, read by code. The tree arm is
-D* and the vector arm is cosine, plain or centered on the mean of the table's six-digit rows. The
oracles are written here from the definitions: D* from ``tests/fixtures/relatedness.py``'s
``tree_distance`` and the bundle's own matrix, and cosine from a dot product over two norms. The
codes that no table holds are never real ones. No test reads a real table.
'''

from typing import Callable, List, Sequence, Tuple

import numpy as np
import pytest

from naics_embedder.panels.relatedness_arms import TreeArm, VectorArm
from naics_embedder.panels.text_only import text_only_frame
from naics_embedder.supervision.code_targets import CodeTargets
from tests.fixtures.relatedness import RELATEDNESS_CODES, tree_distance

# Rows of other lengths than six, added to a table of the 16 six-digit codes
OTHER_CODES = ('31', '3112')

# Seven codes that no table here holds, in neither sorted nor table order
MISSING = ['999999', '111111', '555555', '222222', '888888', '333333', '777777']

# -------------------------------------------------------------------------------------------------
# Tables and an oracle
# -------------------------------------------------------------------------------------------------

def seeded_vectors(rows: int, width: int, seed: int = 0) -> np.ndarray:
    '''Seeded vectors with a common offset, so plain and centered cosine differ.'''

    return np.random.default_rng(seed).normal(size=(rows, width)) + np.linspace(2.0, 6.0, width)

def reference_cosine(
    codes: Sequence[str],
    vectors: np.ndarray,
    anchors: Sequence[str],
    candidates: Sequence[str],
    *,
    center: bool,
) -> np.ndarray:
    '''Cosine from the definition, one pair at a time, after the mean of the six-digit rows.'''

    rows = {code: np.array(vector, dtype=np.float64) for code, vector in zip(codes, vectors)}
    if center:
        six_digit = [row for code, row in rows.items() if len(code) == 6]
        mean = np.sum(six_digit, axis=0) / len(six_digit)
        rows = {code: row - mean for code, row in rows.items()}
    return np.array(
        [
            [
                float(rows[a] @ rows[b]) / (np.linalg.norm(rows[a]) * np.linalg.norm(rows[b]))
                for b in candidates
            ] for a in anchors
        ]
    ).reshape(len(anchors), len(candidates))

def tree_arm() -> TreeArm:
    return TreeArm(RELATEDNESS_CODES, tree_distance(RELATEDNESS_CODES))

def vector_arm() -> VectorArm:
    return VectorArm(RELATEDNESS_CODES, seeded_vectors(16, 5))

def centered_arm() -> VectorArm:
    return VectorArm(RELATEDNESS_CODES, seeded_vectors(16, 5), center=True)

# The three arms the panel reads, each over the 16 codes
ARMS = [
    pytest.param(tree_arm, id='tree'),
    pytest.param(vector_arm, id='plain'),
    pytest.param(centered_arm, id='centered'),
]

# The two constructors that take a table, each over the same 16 codes
TABLE_ARMS = [
    pytest.param(tree_arm, id='tree'),
    pytest.param(vector_arm, id='vector'),
]

def refusal(call: Callable[[], object]) -> str:
    '''The message of the ``ValueError`` that the call raises.'''

    with pytest.raises(ValueError) as excinfo:
        call()
    return str(excinfo.value)

# -------------------------------------------------------------------------------------------------
# The tree arm: minus D*, by code
# -------------------------------------------------------------------------------------------------

def test_the_tree_arm_is_minus_d_star():
    arm = TreeArm(RELATEDNESS_CODES, tree_distance(RELATEDNESS_CODES))
    assert arm.similarity(['311211'], ['311212', '311221', '311111', '312111', '111110'
                                       ]).tolist() == [[-2, -4, -6, -8, -10]]

def check_the_tree_arm_reads_d_star_by_code(targets: CodeTargets) -> None:
    '''Requests out of codebook order, with repeats, and a table in a seeded permuted order.'''

    codes = list(targets.codes)
    position = {code: index for index, code in enumerate(codes)}
    assert targets.structural_distance.dtype == np.float32
    arm = TreeArm.from_targets(targets)

    anchors = [codes[3], codes[0], codes[3]]
    candidates = [codes[4], codes[0], codes[2], codes[2]]
    expected = -targets.structural_distance[np.ix_(
        [position[code] for code in anchors], [position[code] for code in candidates]
    )].astype(np.float64)
    similarity = arm.similarity(anchors, candidates)
    assert similarity.dtype == np.float64
    assert similarity.shape == (3, 4)
    assert np.array_equal(similarity, expected)
    assert np.array_equal(arm.similarity(codes, codes), -targets.structural_distance)

    permutation = np.random.default_rng(7).permutation(len(codes))
    assert permutation.tolist() != list(range(len(codes)))
    permuted = TreeArm(
        [codes[index] for index in permutation],
        targets.structural_distance[np.ix_(permutation, permutation)],
    )
    assert np.array_equal(permuted.similarity(anchors, candidates), similarity)
    assert np.array_equal(permuted.similarity(codes, codes), arm.similarity(codes, codes))

def test_the_tree_arm_reads_the_bundles_d_star_by_code(validated_bundle):
    targets = CodeTargets.from_bundle(validated_bundle)
    check_the_tree_arm_reads_d_star_by_code(targets)

def test_the_tree_arm_reads_a_bundle_of_mixed_levels_by_code(reference_bundle):
    targets = CodeTargets.from_bundle(reference_bundle)
    assert len(targets.codes) == 17
    assert len(set(targets.structural_distance.ravel().tolist())) > 5
    check_the_tree_arm_reads_d_star_by_code(targets)

def test_the_tree_arm_gives_zero_for_a_code_against_itself():
    assert tree_arm().similarity(['311211'], ['311211', '311212']).tolist() == [[0, -2]]

# -------------------------------------------------------------------------------------------------
# The tree arm's refusals
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(5, 6), (6, 5), (6, 6), (4, 4), (25, ), (5, 5, 1)])
def test_a_distance_matrix_that_is_not_n_by_n_is_refused(shape):
    message = refusal(lambda: TreeArm(RELATEDNESS_CODES[:5], np.zeros(shape)))
    assert '5 by 5' in message
    assert str(shape) in message

def test_repeated_codes_are_refused_naming_them():
    codes = ['311211', '311212', '311211', '111110', '111110', '311221']
    message = refusal(lambda: TreeArm(codes, np.zeros((6, 6))))
    assert message.endswith('repeated (2): 311211, 111110')

# -------------------------------------------------------------------------------------------------
# Cosine, plain and centered
# -------------------------------------------------------------------------------------------------

def test_plain_cosine():
    arm = VectorArm(['a', 'b', 'c'], np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]]))
    assert arm.similarity(['a'], ['b', 'c']) == pytest.approx(np.array([[2**-0.5, 0.0]]))

def test_cosine_is_the_definition_over_the_whole_table():
    vectors = seeded_vectors(16, 5)
    arm = VectorArm(RELATEDNESS_CODES, vectors)
    anchors = ['445131', '111110', '311211']
    candidates = list(reversed(RELATEDNESS_CODES))
    similarity = arm.similarity(anchors, candidates)
    assert similarity.shape == (3, 16)
    assert np.allclose(
        similarity,
        reference_cosine(RELATEDNESS_CODES, vectors, anchors, candidates, center=False),
        rtol=0,
        atol=1e-12,
    )

def test_cosine_gives_minus_one_zero_and_one():
    vectors = np.array([[2.0, 0.0], [-3.0, 0.0], [0.0, 5.0], [4.0, 0.0]])
    arm = VectorArm(['a', 'b', 'c', 'd'], vectors)
    assert arm.similarity(['a'], ['b', 'c', 'd']) == pytest.approx(np.array([[-1.0, 0.0, 1.0]]))

def test_cosine_ignores_the_length_of_either_vector():
    vectors = seeded_vectors(16, 5)
    anchors = ['111110', '311211']
    candidates = list(RELATEDNESS_CODES)
    scale = np.random.default_rng(3).uniform(0.1, 10.0, size=(16, 1))
    plain = VectorArm(RELATEDNESS_CODES, vectors).similarity(anchors, candidates)
    scaled = VectorArm(RELATEDNESS_CODES, vectors * scale).similarity(anchors, candidates)
    assert np.allclose(scaled, plain, rtol=0, atol=1e-12)

def test_the_similarity_is_float64_whatever_the_tables_type():
    anchors, candidates = ['311211'], ['311212', '311221']
    for vectors in (
        seeded_vectors(16, 5).astype(np.float32),
        np.arange(1, 81).reshape(16, 5),
    ):
        for center in (False, True):
            arm = VectorArm(RELATEDNESS_CODES, vectors, center=center)
            assert arm.similarity(anchors, candidates).dtype == np.float64

def test_a_request_in_any_order_gives_the_same_pairs():
    arm = vector_arm()
    anchors = list(RELATEDNESS_CODES)
    candidates = list(RELATEDNESS_CODES)
    similarity = arm.similarity(anchors, candidates)
    order = np.random.default_rng(5).permutation(16)
    assert order.tolist() != list(range(16))
    shuffled = [RELATEDNESS_CODES[index] for index in order]
    assert np.array_equal(arm.similarity(shuffled, shuffled), similarity[np.ix_(order, order)])

def test_the_centered_mean_is_over_the_tables_six_digit_codes():
    codes = RELATEDNESS_CODES + OTHER_CODES
    vectors = seeded_vectors(18, 5)
    arm = VectorArm(codes, vectors, center=True)
    anchors = ['311211', '111110', '621210']
    everyone = list(RELATEDNESS_CODES)
    wide = arm.similarity(anchors, everyone)

    # the centered values are the cosine after the mean of the 16 rows is subtracted
    expected = reference_cosine(codes, vectors, anchors, everyone, center=True)
    assert np.allclose(wide, expected, rtol=0, atol=1e-12)
    plain = VectorArm(codes, vectors).similarity(anchors, everyone)
    assert not np.allclose(wide, plain, rtol=0, atol=1e-3)

    # a request that leaves out some six-digit codes leaves each value unchanged
    chosen = [everyone[index] for index in (14, 2, 9)]
    narrow = arm.similarity(anchors, chosen)
    assert np.array_equal(narrow, wide[:, [14, 2, 9]])
    assert np.array_equal(arm.similarity(anchors[:1], everyone[:1]), wide[:1, :1])

    # the rows of the sector and the four-digit code are not in the mean
    other = vectors.copy()
    other[16] += 50.0
    other[17] -= 20.0
    assert np.array_equal(VectorArm(codes, other, center=True).similarity(anchors, everyone), wide)

    # each of the 16 rows is
    for row in (0, 7, 15):
        other = vectors.copy()
        other[row] += 5.0
        moved = VectorArm(codes, other, center=True).similarity(anchors, everyone)
        assert not np.allclose(moved, wide, rtol=0, atol=1e-3)

def test_a_code_that_is_not_six_digit_is_centered_on_the_six_digit_mean():
    codes = RELATEDNESS_CODES + OTHER_CODES
    vectors = seeded_vectors(18, 5)
    arm = VectorArm(codes, vectors, center=True)
    anchors = ['31', '3112', '311211']
    candidates = ['311211', '31', '3112', '111110']
    expected = reference_cosine(codes, vectors, anchors, candidates, center=True)
    assert np.allclose(arm.similarity(anchors, candidates), expected, rtol=0, atol=1e-12)

def test_the_mean_takes_exactly_the_six_digit_rows():
    # lengths 2, 5, 6, 6 and 7: only the two six-digit rows make the mean
    codes = ['22', '11111', '111111', '222222', '1111111']
    vectors = np.array(
        [[5.0, 1.0, 2.0], [9.0, 7.0, 1.0], [2.0, 6.0, 3.0], [8.0, 1.0, 4.0], [3.0, 3.0, 9.0]]
    )
    arm = VectorArm(codes, vectors, center=True)
    similarity = arm.similarity(codes, codes)
    assert np.allclose(
        similarity, reference_cosine(codes, vectors, codes, codes, center=True), rtol=0, atol=1e-12
    )
    # about their own mean, two rows point opposite ways
    assert similarity[2, 3] == pytest.approx(-1.0)

def test_centering_needs_a_six_digit_row():
    vectors = np.array([[1.0, 2.0], [3.0, 5.0], [2.0, 9.0]])
    message = refusal(lambda: VectorArm(['31', '3112', '31121'], vectors, center=True))
    assert 'six-digit' in message
    # plain cosine has no mean to take
    assert VectorArm(['31', '3112', '31121'], vectors).similarity(['31'], ['3112']).shape == (1, 1)

# -------------------------------------------------------------------------------------------------
# A text-only table is read by its code and t columns
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('center', [False, True])
def test_from_text_only_reads_the_t_columns(center):
    codes = list(RELATEDNESS_CODES) + list(OTHER_CODES)
    vectors = seeded_vectors(18, 5)
    table = text_only_frame(codes, vectors)
    assert table.columns == ['code', 't0', 't1', 't2', 't3', 't4']
    from_table = VectorArm.from_text_only(table, center=center)
    direct = VectorArm(codes, vectors, center=center)
    anchors = ['311211', '31', '621210', '3112']
    candidates = list(reversed(codes))
    assert np.array_equal(
        from_table.similarity(anchors, candidates), direct.similarity(anchors, candidates)
    )
    assert np.allclose(
        from_table.similarity(anchors, candidates),
        reference_cosine(codes, vectors, anchors, candidates, center=center),
        rtol=0,
        atol=1e-12,
    )

def test_from_text_only_is_plain_unless_told_to_center():
    codes = list(RELATEDNESS_CODES)
    vectors = seeded_vectors(16, 5)
    table = text_only_frame(codes, vectors)
    plain = VectorArm.from_text_only(table).similarity(['311211'], codes)
    assert np.array_equal(plain, VectorArm(codes, vectors).similarity(['311211'], codes))
    centered = VectorArm.from_text_only(table, center=True).similarity(['311211'], codes)
    assert not np.allclose(plain, centered, rtol=0, atol=1e-3)

# -------------------------------------------------------------------------------------------------
# An empty request, a request of one
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('make_arm', ARMS)
def test_an_empty_request_gives_an_empty_matrix_of_the_right_shape(make_arm):
    arm = make_arm()
    for anchors, candidates, shape in (
        ([], ['311211', '311212', '111110'], (0, 3)),
        (['311211', '111110'], [], (2, 0)),
        ([], [], (0, 0)),
        (['311211'], ['311212'], (1, 1)),
    ):
        similarity = arm.similarity(anchors, candidates)
        assert similarity.shape == shape
        assert similarity.dtype == np.float64

@pytest.mark.parametrize('make_arm', ARMS)
def test_a_request_of_one_pair_is_the_matching_cell_of_the_whole_table(make_arm):
    arm = make_arm()
    whole = arm.similarity(list(RELATEDNESS_CODES), list(RELATEDNESS_CODES))
    assert whole.shape == (16, 16)
    row, column = RELATEDNESS_CODES.index('445131'), RELATEDNESS_CODES.index('111120')
    assert arm.similarity(['445131'], ['111120'])[0, 0] == whole[row, column]

# -------------------------------------------------------------------------------------------------
# An unknown code in a request: the first five are named, with how many there are
# -------------------------------------------------------------------------------------------------

def test_an_unknown_code_or_a_zero_row_is_refused_naming_it():
    assert '999999' in refusal(lambda: tree_arm().similarity(['311211'], ['999999']))
    assert '999999' in refusal(lambda: vector_arm().similarity(['999999'], ['311211']))
    vectors = seeded_vectors(16, 5)
    vectors[3] = 0.0
    arm = VectorArm(RELATEDNESS_CODES, vectors)
    assert RELATEDNESS_CODES[3] in refusal(
        lambda: arm.similarity([RELATEDNESS_CODES[3]], ['311211'])
    )
    assert RELATEDNESS_CODES[3] in refusal(
        lambda: arm.similarity(['311211'], [RELATEDNESS_CODES[3]])
    )

@pytest.mark.parametrize('make_arm', TABLE_ARMS)
def test_the_first_five_unknown_codes_are_named_with_how_many_there_are(make_arm):
    arm = make_arm()
    anchors = ['311211', MISSING[0], MISSING[1], '111110', MISSING[2]]
    candidates = [MISSING[3], '311212', MISSING[4], MISSING[5], MISSING[6]]
    message = refusal(lambda: arm.similarity(anchors, candidates))
    assert message.endswith('(7): 999999, 111111, 555555, 222222, 888888 and 2 more')
    for code in MISSING[5:]:
        assert code not in message

@pytest.mark.parametrize('make_arm', TABLE_ARMS)
@pytest.mark.parametrize('count', [1, 3, 5])
def test_every_unknown_code_is_named_when_there_are_five_or_fewer(make_arm, count):
    arm = make_arm()
    missing = MISSING[:count]
    message = refusal(lambda: arm.similarity(['311211', *missing[:2]], [*missing[2:], '111110']))
    assert message.endswith(f'({count}): ' + ', '.join(missing))

@pytest.mark.parametrize('make_arm', TABLE_ARMS)
def test_an_unknown_code_is_refused_whichever_side_it_is_on(make_arm):
    arm = make_arm()
    for anchors, candidates in (
        (['999999', '311211'], ['311212']),
        (['311211'], ['311212', '999999']),
        ([], ['999999']),
        (['999999'], []),
    ):
        message = refusal(lambda: arm.similarity(anchors, candidates))
        assert message.endswith('(1): 999999')

@pytest.mark.parametrize('make_arm', TABLE_ARMS)
def test_an_unknown_code_named_twice_is_counted_once(make_arm):
    arm = make_arm()
    message = refusal(lambda: arm.similarity(['999999', '999999'], ['999999', '311211']))
    assert message.endswith('(1): 999999')

# -------------------------------------------------------------------------------------------------
# Repeated codes and a table that does not fit its codes
# -------------------------------------------------------------------------------------------------

def test_repeated_codes_are_refused_by_the_vector_arm_naming_them():
    codes = ['311211', '311212', '311211', '111110', '111110', '311221']
    message = refusal(lambda: VectorArm(codes, seeded_vectors(6, 3)))
    assert message.endswith('repeated (2): 311211, 111110')

@pytest.mark.parametrize('shape', [(5, ), (6, 3), (4, 3), (5, 3, 1)])
def test_a_vector_table_that_is_not_one_row_per_code_is_refused(shape):
    message = refusal(lambda: VectorArm(RELATEDNESS_CODES[:5], np.ones(shape)))
    assert '5' in message
    assert str(shape) in message

# -------------------------------------------------------------------------------------------------
# A non-finite value
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('center', [False, True])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('code', ['311211', '31'])
def test_a_non_finite_value_anywhere_in_the_table_is_refused_naming_the_code(center, bad, code):
    codes = RELATEDNESS_CODES + OTHER_CODES
    vectors = seeded_vectors(18, 5)
    # one entry of a row: its other entries are finite, and no request has named it
    vectors[codes.index(code), 2] = bad
    message = refusal(lambda: VectorArm(codes, vectors, center=center))
    assert message.endswith(f'(1): {code}')

def test_every_row_with_a_non_finite_value_is_named_in_table_order():
    codes = RELATEDNESS_CODES + OTHER_CODES
    vectors = seeded_vectors(18, 5)
    vectors[17, 0] = np.inf
    vectors[3, 4] = np.nan
    message = refusal(lambda: VectorArm(codes, vectors))
    assert message.endswith('(2): 111191, 3112')

# -------------------------------------------------------------------------------------------------
# A zero-norm row of a requested code
# -------------------------------------------------------------------------------------------------

def zero_row_table() -> Tuple[List[str], np.ndarray]:
    '''Six codes of three columns each: z and y have a zero row, a, b, c and d do not.'''

    codes = ['a', 'b', 'c', 'z', 'y', 'd']
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    return codes, vectors

def test_a_zero_row_is_refused_naming_it_whichever_side_it_is_on():
    codes, vectors = zero_row_table()
    arm = VectorArm(codes, vectors)
    for anchors, candidates in (
        (['z'], ['a', 'b']),
        (['a', 'b'], ['c', 'z']),
        (['z', 'a'], ['z']),
        (['a'], ['z', 'z']),
        ([], ['z']),
        (['z'], []),
    ):
        message = refusal(lambda: arm.similarity(anchors, candidates))
        assert message.endswith('(1): z')

def test_every_requested_zero_row_is_named_in_request_order():
    codes, vectors = zero_row_table()
    arm = VectorArm(codes, vectors)
    message = refusal(lambda: arm.similarity(['a', 'y'], ['z', 'b', 'y']))
    assert message.endswith('(2): y, z')

def test_a_zero_row_that_no_request_names_never_refuses_a_read():
    codes, vectors = zero_row_table()
    arm = VectorArm(codes, vectors)
    similarity = arm.similarity(['a', 'b'], ['c', 'd'])
    assert np.isfinite(similarity).all()
    assert similarity == pytest.approx(
        reference_cosine(codes, vectors, ['a', 'b'], ['c', 'd'], center=False)
    )

def test_the_zero_check_follows_the_centering():
    codes = ['111110', '111120', '111150', '31']
    # the third row is the mean of the first three, so it is zero once centered
    vectors = np.array([[1.0, 2.0, 4.0], [3.0, 4.0, 6.0], [2.0, 3.0, 5.0], [9.0, 8.0, 7.0]])
    plain = VectorArm(codes, vectors)
    assert np.isfinite(plain.similarity(['111150'], ['111110', '111120'])).all()
    centered = VectorArm(codes, vectors, center=True)
    assert refusal(lambda: centered.similarity(['111150'], ['111110'])).endswith('(1): 111150')
    assert refusal(lambda: centered.similarity(['111110'], ['111150'])).endswith('(1): 111150')
    # a row that is zero once centered, and not requested, never refuses
    assert np.isfinite(centered.similarity(['111110', '31'], ['111120', '31'])).all()

def test_a_zero_row_is_not_zero_once_centered_on_a_mean_that_is_not_zero():
    codes = ['111110', '111120', '111150', '111191']
    vectors = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    centered = VectorArm(codes, vectors, center=True)
    similarity = centered.similarity(['111110'], ['111120', '111150', '111191'])
    assert np.isfinite(similarity).all()
    assert similarity == pytest.approx(
        reference_cosine(codes, vectors, ['111110'], ['111120', '111150', '111191'], center=True)
    )
    plain = VectorArm(codes, vectors)
    assert refusal(lambda: plain.similarity(['111110'], ['111120'])).endswith('(1): 111110')
