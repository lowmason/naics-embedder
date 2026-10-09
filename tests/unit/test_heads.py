'''
The geometry heads of Req 12's three arms (``text_model/heads.py``): their points, training
distances and read maps, and the one list of geometries that every layer names.
'''

from types import SimpleNamespace
from typing import get_args

import pytest
import torch

from naics_embedder.decision import records
from naics_embedder.metrics import diagnostics
from naics_embedder.panels.decoding import (
    DISTANCES,
    GEOMETRY_DISTANCES,
    cosine_distances,
    euclidean_distances,
)
from naics_embedder.text_model.heads import (
    GEOMETRIES,
    EuclideanHead,
    SphericalHead,
    build_head,
    chord_distance,
    flat_distance,
    head_of,
)
from naics_embedder.text_model.hyperbolic import HyperbolicHead, exp_map_origin, polar_distance

pytestmark = pytest.mark.unit

HEADS = {'euclidean': EuclideanHead, 'spherical': SphericalHead, 'hyperbolic': HyperbolicHead}

def _vectors(count: int = 6, *, seed: int = 0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 8, generator=generator, dtype=torch.float64).to(dtype)

# -------------------------------------------------------------------------------------------------
# One list of geometries
# -------------------------------------------------------------------------------------------------

def test_every_layer_names_the_same_three_geometries():
    assert GEOMETRIES == ('euclidean', 'spherical', 'hyperbolic')
    assert tuple(GEOMETRY_DISTANCES) == GEOMETRIES
    assert get_args(records.Geometry) == GEOMETRIES
    assert diagnostics.GEOMETRIES == GEOMETRIES
    assert set(GEOMETRY_DISTANCES.values()) == set(DISTANCES)

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_each_head_names_its_geometry_its_distance_and_whether_the_radial_term_applies(geometry):
    head = build_head(geometry)

    assert isinstance(head, HEADS[geometry])
    assert head.geometry == geometry
    assert head.distance == GEOMETRY_DISTANCES[geometry]
    # The radial term exists only in the hyperbolic arm (Req 12)
    assert head.radial is (geometry == 'hyperbolic')
    assert list(head.parameters()) == []

def test_build_head_gives_the_bound_to_the_hyperbolic_head_and_refuses_an_unknown_geometry():
    assert build_head('hyperbolic', radius_bound=5.0).radius_bound == 5.0
    with pytest.raises(ValueError, match="unknown geometry 'poincare'"):
        build_head('poincare')

def test_head_of_finds_the_head_of_a_module_or_of_its_encoder():
    encoder = SimpleNamespace(head=SphericalHead())

    assert head_of(SimpleNamespace(encoder=encoder)) is encoder.head
    assert head_of(encoder) is encoder.head

# -------------------------------------------------------------------------------------------------
# The flat heads' points
# -------------------------------------------------------------------------------------------------

def test_the_euclidean_point_is_the_vector_itself_with_its_norm_and_direction():
    vectors = _vectors()

    points = EuclideanHead()(vectors)

    assert torch.equal(points.tangent, vectors)
    assert torch.equal(points.embedding, vectors)
    torch.testing.assert_close(points.radius, vectors.norm(dim=1), rtol=1e-15, atol=0.0)
    torch.testing.assert_close(
        points.radius.unsqueeze(1) * points.direction, vectors, rtol=1e-15, atol=1e-15
    )

def test_the_spherical_point_is_the_unit_direction_at_radius_one():
    vectors = _vectors()

    points = SphericalHead()(vectors)

    direction = vectors / vectors.norm(dim=1, keepdim=True)
    torch.testing.assert_close(points.tangent, direction, rtol=1e-15, atol=1e-15)
    assert torch.equal(points.embedding, points.tangent)
    assert torch.equal(points.direction, points.tangent)
    assert torch.equal(points.radius, torch.ones(len(vectors), dtype=torch.float64))
    # No term reads the radius as a quantity to train: it takes no gradient
    assert not SphericalHead()(vectors.clone().requires_grad_()).radius.requires_grad

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_the_zero_vector_has_radius_zero_and_every_gradient_there_is_finite(geometry):
    vectors = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)

    points = build_head(geometry)(vectors)

    assert torch.equal(points.radius.detach(), torch.zeros(2, dtype=torch.float64))
    assert torch.equal(points.direction.detach(), torch.zeros(2, 4, dtype=torch.float64))
    (gradient, ) = torch.autograd.grad(sum(part.sum() for part in points), vectors)
    assert torch.isfinite(gradient).all()

# -------------------------------------------------------------------------------------------------
# Training distances
# -------------------------------------------------------------------------------------------------

def test_the_flat_distance_is_the_euclidean_distance_of_the_vectors():
    head = EuclideanHead()
    vectors_a, vectors_b = _vectors(5, seed=1), _vectors(7, seed=2)
    a, b = head(vectors_a), head(vectors_b)

    distances = head.pair_distance(a.radius, a.direction, b.radius, b.direction)

    expected = euclidean_distances(vectors_a, vectors_b)
    torch.testing.assert_close(distances, expected, rtol=1e-12, atol=1e-12)

def test_the_chord_distance_is_the_cosine_distance_of_the_vectors():
    head = SphericalHead()
    vectors_a, vectors_b = _vectors(5, seed=1), _vectors(7, seed=2)
    a, b = head(vectors_a), head(vectors_b)

    distances = head.pair_distance(a.radius, a.direction, b.radius, b.direction)

    expected = cosine_distances(vectors_a, vectors_b)
    torch.testing.assert_close(distances, expected, rtol=1e-9, atol=1e-12)

def test_the_hyperbolic_heads_distance_and_read_map_are_the_polar_distance_and_the_exp_map():
    head = HyperbolicHead()
    points = head(_vectors(5, dtype=torch.float32))

    distances = head.pair_distance(points.radius, points.direction, points.radius, points.direction)

    expected = polar_distance(points.radius, points.direction, points.radius, points.direction)
    assert torch.equal(distances, expected)
    assert torch.equal(head.read_points(points.tangent), exp_map_origin(points.tangent))

@pytest.mark.parametrize('geometry', ['euclidean', 'spherical'])
def test_float32_resolves_two_points_a_milliradian_apart(geometry):
    '''
    The flat distances take explicit differences, so float32 resolves a small angle that
    1 − û_a · û_b would cancel: at 1e-3 rad its float32 relative error is about 0.3.
    '''

    first = torch.zeros(1, 8, dtype=torch.float64)
    first[0, 0] = 1.0
    second = torch.zeros(1, 8, dtype=torch.float64)
    second[0, 0], second[0, 1] = torch.cos(torch.tensor(1e-3)), torch.sin(torch.tensor(1e-3))
    head = build_head(geometry)
    a, b = head((5.0 * first).float()), head((5.0 * second).float())

    distance = head.pair_distance(a.radius, a.direction, b.radius, b.direction).double()

    exact = DISTANCES[head.distance](5.0 * first, 5.0 * second)
    assert ((distance - exact).abs() / exact).item() <= 1e-3
    if geometry == 'spherical':
        cancelled = (1 - a.direction @ b.direction.T).double()
        assert ((cancelled - exact).abs() / exact).item() > 1e-2

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_zero_separation_has_distance_zero_and_a_finite_gradient(geometry):
    '''Each anchor meets its own live row among the candidates (spec 4.3).'''

    head = build_head(geometry)
    vectors = _vectors(3, dtype=torch.float32).requires_grad_()
    points = head(vectors)

    distances = head.pair_distance(points.radius, points.direction, points.radius, points.direction)

    assert torch.equal(distances.diagonal().detach(), torch.zeros(3))
    assert (distances.detach() + torch.eye(3) > 0).all()
    (gradient, ) = torch.autograd.grad(distances.sum(), vectors)
    assert torch.isfinite(gradient).all()

@pytest.mark.parametrize('distance', [flat_distance, chord_distance])
def test_a_flat_distance_refuses_unpaired_shapes(distance):
    radius, direction = torch.ones(3), torch.ones(3, 4)

    with pytest.raises(ValueError, match=f'{distance.__name__} takes radii'):
        distance(radius.unsqueeze(1), direction, radius, direction)
    with pytest.raises(ValueError, match=f'{distance.__name__} takes radii'):
        distance(radius, direction, radius, direction[:, :2])

# -------------------------------------------------------------------------------------------------
# Read maps
# -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_the_read_map_takes_an_export_to_the_points_the_heads_distance_reads(geometry):
    head = build_head(geometry)
    points = head(_vectors(4, dtype=torch.float32))

    read = head.read_points(points.tangent)

    assert (read.dtype, read.device.type) == (torch.float64, 'cpu')
    if geometry == 'hyperbolic':
        assert torch.equal(read, exp_map_origin(points.tangent))
    else:
        assert torch.equal(read, points.tangent.double())
    # The read lands where the head put its point
    torch.testing.assert_close(read, points.embedding.double(), rtol=1e-5, atol=1e-6)
