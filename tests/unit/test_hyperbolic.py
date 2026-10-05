'''
Unit tests for hyperbolic geometry operations.

Tests the core Lorentz model operations including exponential/logarithmic maps,
distance computations, manifold validity checks, and numerical stability.

The head and the polar distance follow spec 4.2 and §6 "Head and distance": a gradient-passing
radius bound, and the training distance from (r, û), checked against the reads' float64 Lorentz
distance up to r = R.
'''

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from naics_embedder.panels.decoding import lorentz_distances
from naics_embedder.text_model import hyperbolic
from naics_embedder.text_model.hyperbolic import (
    HyperbolicHead,
    LorentzOps,
    check_lorentz_manifold_validity,
    compute_hyperbolic_radii,
)
from tests.fixtures.hyperboloid import hyperboloid_points

# The shipped bound, model.radius_bound
BOUND = 8.0

# -------------------------------------------------------------------------------------------------
# LorentzOps Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestLorentzOps:
    '''Test suite for LorentzOps utility class.'''

    def test_exp_log_roundtrip(self, sample_lorentz_embeddings):
        '''Test that exp(log(x)) ≈ x for valid Lorentz embeddings.'''

        c = 1.0
        log_emb = LorentzOps.log_map_zero(sample_lorentz_embeddings, c=c)
        reconstructed = LorentzOps.exp_map_zero(log_emb, c=c)

        assert torch.allclose(reconstructed, sample_lorentz_embeddings, atol=1e-5)

    def test_log_exp_roundtrip(self, sample_tangent_vectors):
        '''Test that log(exp(v)) ≈ v for tangent vectors.'''

        c = 1.0
        # Project tangent to hyperboloid
        hyp_emb = LorentzOps.exp_map_zero(sample_tangent_vectors, c=c)
        # Map back to tangent space
        reconstructed = LorentzOps.log_map_zero(hyp_emb, c=c)

        # Note: Only spatial components should match (time component should be 0)
        assert torch.allclose(reconstructed[:, 1:], sample_tangent_vectors[:, 1:], atol=1e-4)

    def test_exp_map_produces_valid_embeddings(self, sample_tangent_vectors):
        '''Test that exp_map produces embeddings on the Lorentz manifold.'''

        c = 1.0
        hyp_emb = LorentzOps.exp_map_zero(sample_tangent_vectors, c=c)

        is_valid, lorentz_norms, violations = check_lorentz_manifold_validity(
            hyp_emb, curvature=c, tolerance=1e-3
        )

        assert is_valid, f'Max violation: {violations.max().item()}'
        assert torch.allclose(lorentz_norms, torch.tensor(-1.0 / c), atol=1e-3)

    @pytest.mark.parametrize('curvature', [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_exp_map_with_different_curvatures(self, sample_tangent_vectors, curvature):
        '''Test exp_map works correctly for different curvature values.'''

        hyp_emb = LorentzOps.exp_map_zero(sample_tangent_vectors, c=curvature)

        is_valid, lorentz_norms, _ = check_lorentz_manifold_validity(
            hyp_emb, curvature=curvature, tolerance=1e-2
        )

        assert is_valid
        expected_norm = -1.0 / curvature
        assert torch.allclose(lorentz_norms, torch.tensor(expected_norm), atol=1e-2)

    def test_distance_is_positive(self, sample_lorentz_embeddings):
        '''Test that Lorentzian distances are always non-negative.'''

        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        distances = LorentzOps.lorentz_distance(x, y, c=1.0)

        assert torch.all(distances >= 0), 'Distances must be non-negative'

    def test_distance_symmetry(self, sample_lorentz_embeddings):
        '''Test that d(x, y) = d(y, x) (symmetry).'''

        x = sample_lorentz_embeddings[:8]
        y = sample_lorentz_embeddings[8:]

        d_xy = LorentzOps.lorentz_distance(x, y, c=1.0)
        d_yx = LorentzOps.lorentz_distance(y, x, c=1.0)

        assert torch.allclose(d_xy, d_yx, atol=1e-6)

    def test_distance_to_self_is_zero(self, sample_lorentz_embeddings):
        '''Test that d(x, x) = 0 (identity of indiscernibles).'''

        distances = LorentzOps.lorentz_distance(
            sample_lorentz_embeddings, sample_lorentz_embeddings, c=1.0
        )

        # Use 5e-3 tolerance due to floating point precision in acosh near 1.0
        assert torch.allclose(distances, torch.zeros_like(distances), atol=5e-3)

    def test_triangle_inequality(self, sample_lorentz_embeddings):
        '''Test triangle inequality: d(x, z) ≤ d(x, y) + d(y, z).'''

        batch_size = sample_lorentz_embeddings.shape[0]
        third = batch_size // 3

        x = sample_lorentz_embeddings[:third]
        y = sample_lorentz_embeddings[third:2 * third]
        z = sample_lorentz_embeddings[2 * third:3 * third]

        d_xz = LorentzOps.lorentz_distance(x, z, c=1.0)
        d_xy = LorentzOps.lorentz_distance(x, y, c=1.0)
        d_yz = LorentzOps.lorentz_distance(y, z, c=1.0)

        # Allow small numerical tolerance
        assert torch.all(d_xz <= d_xy + d_yz + 1e-4), 'Triangle inequality violated'

    def test_numerical_stability_small_norms(self, test_device):
        '''Test numerical stability for very small norm tangent vectors.'''

        small_tangent = torch.randn(10, 385, device=test_device) * 1e-8

        hyp_emb = LorentzOps.exp_map_zero(small_tangent, c=1.0)

        is_valid, _, _ = check_lorentz_manifold_validity(hyp_emb, curvature=1.0, tolerance=1e-3)
        assert is_valid
        assert not torch.any(torch.isnan(hyp_emb))
        assert not torch.any(torch.isinf(hyp_emb))

    def test_numerical_stability_large_norms(self, test_device):
        '''Test numerical stability for very large norm tangent vectors.'''

        large_tangent = torch.randn(10, 385, device=test_device) * 100

        hyp_emb = LorentzOps.exp_map_zero(large_tangent, c=1.0)

        assert not torch.any(torch.isnan(hyp_emb))
        assert not torch.any(torch.isinf(hyp_emb))

# -------------------------------------------------------------------------------------------------
# HyperbolicHead Tests
# -------------------------------------------------------------------------------------------------

def _vectors_of_norm(norms, *, dimension: int = 16, seed: int = 0) -> torch.Tensor:
    '''Float64 vectors in random directions with these norms, one row per norm.'''

    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(len(norms), dimension, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return torch.tensor(norms, dtype=torch.float64).unsqueeze(1) * directions

@pytest.mark.unit
class TestHyperbolicHead:
    '''The head: no parameters and no curvature, a gradient-passing bound, then the exp map.'''

    def test_the_head_has_no_parameters_and_names_its_distance(self):
        head = HyperbolicHead()

        assert list(head.parameters()) == []
        assert head.distance == 'lorentz'
        assert head.radius_bound == BOUND

    def test_the_head_takes_no_curvature_and_no_cap(self):
        with pytest.raises(TypeError):
            HyperbolicHead(curvature=1.0)
        with pytest.raises(TypeError):
            HyperbolicHead(max_norm=2.0)

    @pytest.mark.parametrize('bound', [0.0, -1.0, math.inf, math.nan])
    def test_a_bound_that_is_not_positive_and_finite_is_refused(self, bound):
        with pytest.raises(ValueError, match='radius_bound must be a positive finite number'):
            HyperbolicHead(radius_bound=bound)

    def test_the_head_returns_the_tangent_the_point_the_radius_and_the_direction(self):
        points = HyperbolicHead()(_vectors_of_norm([0.3, 5.0, 40.0], dimension=4))

        assert isinstance(points, hyperbolic.HeadPoints)
        assert points._fields == ('tangent', 'embedding', 'radius', 'direction')
        assert points.tangent.shape == (3, 4)
        assert points.embedding.shape == (3, 5)
        assert points.radius.shape == (3, )
        assert points.direction.shape == (3, 4)

    @pytest.mark.parametrize('bound', [BOUND, 5.0])
    def test_the_radius_is_the_bounded_norm_and_the_tangent_is_r_times_the_direction(self, bound):
        '''Spec 4.2: r = R · tanh(ν / R), û = v / ν, and the tangent at o is r · û, at any R.'''

        vectors = _vectors_of_norm([0.3, 5.0, 40.0, 1e3])
        norm = vectors.norm(dim=1)

        points = HyperbolicHead(radius_bound=bound)(vectors)

        torch.testing.assert_close(points.radius, bound * torch.tanh(norm / bound))
        torch.testing.assert_close(points.direction, vectors / norm.unsqueeze(1))
        torch.testing.assert_close(points.tangent, points.radius.unsqueeze(1) * points.direction)
        torch.testing.assert_close(points.tangent.norm(dim=1), points.radius)
        assert (points.radius <= bound).all()

    def test_the_point_is_the_exp_map_of_the_bounded_tangent(self):
        vectors = _vectors_of_norm([0.3, 2.0, 5.0, 20.0], dimension=2)

        points = HyperbolicHead()(vectors)

        # (cosh r, sinh r · û), which is LorentzOps' exp map at c = 1 of the bounded tangent; it
        # takes a (B, d + 1) tangent and ignores its time slot
        expected = torch.cat(
            [
                torch.cosh(points.radius).unsqueeze(1),
                torch.sinh(points.radius).unsqueeze(1) * points.direction,
            ],
            dim=1,
        )
        torch.testing.assert_close(points.embedding, expected)
        padded = torch.cat([torch.zeros(4, 1, dtype=torch.float64), points.tangent], dim=1)
        torch.testing.assert_close(points.embedding, LorentzOps.exp_map_zero(padded, c=1.0))
        is_valid, _, _ = check_lorentz_manifold_validity(points.embedding, tolerance=1e-6)
        assert is_valid

    def test_at_nu_20_the_radius_still_takes_gradient_and_stays_below_the_bound(self):
        '''Spec §6: at ν = 20 the gradient with respect to ν is nonzero, and r ≤ R (Req 13).'''

        nu = torch.tensor(20.0, requires_grad=True)
        direction = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)

        points = HyperbolicHead(radius_bound=BOUND)(nu * direction)
        points.radius.sum().backward()

        assert points.radius.item() <= BOUND
        # dr/dν = sech²(ν / R), about 0.027 here; the interim cap passed about 1e-7
        assert nu.grad.item() > 0.02
        assert nu.grad.item() == pytest.approx(1.0 / math.cosh(20.0 / BOUND)**2, rel=1e-4)

    @pytest.mark.parametrize('output', ['tangent', 'embedding', 'radius', 'direction'])
    def test_the_zero_vector_is_the_origin_and_every_gradient_there_is_finite(self, output):
        vectors = torch.zeros(2, 16, requires_grad=True)

        points = HyperbolicHead()(vectors)
        getattr(points, output).sum().backward()

        assert torch.equal(points.tangent, torch.zeros(2, 16))
        assert torch.equal(points.radius, torch.zeros(2))
        assert torch.equal(points.direction, torch.zeros(2, 16))
        origin = torch.zeros(2, 17)
        origin[:, 0] = 1.0
        assert torch.equal(points.embedding, origin)
        assert torch.isfinite(vectors.grad).all()

# -------------------------------------------------------------------------------------------------
# The polar distance
# -------------------------------------------------------------------------------------------------

def _polar_parts(points: torch.Tensor):
    '''(r, û) of float64 hyperboloid points: r = asinh ‖x_s‖, exact near the origin too.'''

    space = points[:, 1:]
    norm = space.norm(dim=1)
    return torch.asinh(norm), space / norm.unsqueeze(1)

def _points_up_to_the_bound() -> torch.Tensor:
    '''Float64 c = 1 points in dimension 16 at radii in [0, R], four of them at exactly R.'''

    inside = hyperboloid_points(64, seed=0, min_radius=0.0, max_radius=BOUND, spatial_dim=16)
    at_bound = hyperboloid_points(4, seed=1, min_radius=BOUND, max_radius=BOUND, spatial_dim=16)
    return torch.cat([inside, at_bound])

def _off_diagonal(size: int) -> torch.Tensor:
    return ~torch.eye(size, dtype=torch.bool)

@pytest.mark.unit
class TestPolarDistance:
    '''Spec 4.2: d(x, y) from (r, û), the hyperbolic law of cosines in half-angle form.'''

    def test_the_distance_is_pairwise_and_refuses_other_shapes(self):
        radius_a, radius_b = torch.rand(3), torch.rand(5)
        direction_a = torch.nn.functional.normalize(torch.randn(3, 16), dim=1)
        direction_b = torch.nn.functional.normalize(torch.randn(5, 16), dim=1)

        distances = hyperbolic.polar_distance(radius_a, direction_a, radius_b, direction_b)

        assert distances.shape == (3, 5)
        with pytest.raises(ValueError, match='polar_distance takes'):
            hyperbolic.polar_distance(radius_a.unsqueeze(1), direction_a, radius_b, direction_b)
        with pytest.raises(ValueError, match='polar_distance takes'):
            hyperbolic.polar_distance(radius_a, direction_a, radius_b, direction_b[:, :8])

    def test_the_polar_form_equals_the_float64_lorentz_distance_up_to_the_bound(self):
        '''Spec §6: the reads' float64 arcosh(−⟨x, y⟩_L) (panels/decoding.py), up to r = R.'''

        points = _points_up_to_the_bound()
        radius, direction = _polar_parts(points)
        assert radius.max().item() == pytest.approx(BOUND)

        polar = hyperbolic.polar_distance(radius, direction, radius, direction)

        # Off the diagonal: there the Lorentz form reads arcosh(1 + rounding), about 3e-5
        off = _off_diagonal(len(points))
        reference = lorentz_distances(points, points)
        torch.testing.assert_close(polar[off], reference[off], rtol=1e-9, atol=0.0)

    def test_float32_under_bf16_autocast_stays_within_tolerance_of_float64(self):
        '''Spec §6: the float32 form, under CPU bf16 autocast too, on well-separated pairs.'''

        points = _points_up_to_the_bound()
        radius, direction = _polar_parts(points)
        off = _off_diagonal(len(points))
        reference = lorentz_distances(points, points)
        assert reference[off].min().item() > 0.1

        with torch.autocast('cpu', dtype=torch.bfloat16):
            polar = hyperbolic.polar_distance(
                radius.float(), direction.float(), radius.float(), direction.float()
            )

        assert polar.dtype == torch.float32
        torch.testing.assert_close(polar.double()[off], reference[off], rtol=1e-5, atol=0.0)

    def test_a_small_angle_is_resolved_from_explicit_differences(self):
        '''‖û_x − û_y‖² by differences: 2 − 2 û_x · û_y would cancel at this angle in float32.'''

        angle = 1e-3
        generator = torch.Generator().manual_seed(3)
        base, other = torch.randn(2, 16, generator=generator, dtype=torch.float64)
        base = base / base.norm()
        other = other - (other @ base) * base
        other = other / other.norm()
        turned = math.cos(angle) * base + math.sin(angle) * other
        radius = torch.tensor([1.0])
        first, second = base.float().unsqueeze(0), turned.float().unsqueeze(0)

        polar = hyperbolic.polar_distance(radius, first, radius, second)

        # Two points at radius r an angle θ apart: sinh(d / 2) = sinh r · sin(θ / 2)
        exact = 2 * math.asinh(math.sinh(1.0) * math.sin(angle / 2))
        assert polar.item() == pytest.approx(exact, rel=1e-3)

    @pytest.mark.parametrize('radius', [0.0, 3.0, BOUND])
    def test_zero_separation_has_a_finite_value_and_gradient(self, radius):
        '''Spec §5: the polar distance and its gradient stay finite at zero separation.'''

        radii = torch.tensor([radius], requires_grad=True)
        unit = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
        # The origin has no direction: the head gives û = 0 there
        directions = (unit if radius > 0 else torch.zeros(1, 16)).requires_grad_(True)

        distance = hyperbolic.polar_distance(radii, directions, radii, directions)
        distance.sum().backward()

        assert 0.0 <= distance.item() <= 1e-5
        assert torch.isfinite(radii.grad).all()
        assert torch.isfinite(directions.grad).all()

# -------------------------------------------------------------------------------------------------
# Manifold Validity Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestManifoldValidity:
    '''Test suite for manifold validity checking functions.'''

    def test_valid_embeddings_pass_check(self, sample_lorentz_embeddings):
        '''Test that valid embeddings pass the manifold check.'''

        is_valid, lorentz_norms, violations = check_lorentz_manifold_validity(
            sample_lorentz_embeddings, curvature=1.0, tolerance=1e-3
        )

        assert is_valid
        assert torch.allclose(lorentz_norms, torch.tensor(-1.0), atol=1e-3)
        assert torch.all(violations < 1e-3)

    def test_invalid_embeddings_fail_check(self, test_device):
        '''Test that invalid embeddings fail the manifold check.'''

        # Create deliberately invalid embeddings (random points)
        invalid_embeddings = torch.randn(10, 385, device=test_device)

        is_valid, _, violations = check_lorentz_manifold_validity(
            invalid_embeddings, curvature=1.0, tolerance=1e-3
        )

        assert not is_valid
        assert torch.any(violations > 1e-3)

    @pytest.mark.parametrize('curvature', [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_validity_check_with_different_curvatures(self, sample_tangent_vectors, curvature):
        '''Test validity checking works for different curvatures.'''

        hyp_emb = LorentzOps.exp_map_zero(sample_tangent_vectors, c=curvature)

        is_valid, lorentz_norms, _ = check_lorentz_manifold_validity(
            hyp_emb, curvature=curvature, tolerance=1e-2
        )

        assert is_valid
        expected_norm = -1.0 / curvature
        assert torch.allclose(lorentz_norms, torch.tensor(expected_norm), atol=1e-2)

    def test_tolerance_parameter(self, sample_lorentz_embeddings):
        '''Test that tolerance parameter affects validity check.'''

        # Should pass with loose tolerance
        is_valid_loose, _, _ = check_lorentz_manifold_validity(
            sample_lorentz_embeddings, curvature=1.0, tolerance=1e-2
        )

        # Should pass with moderate tolerance for well-formed embeddings
        # Note: 1e-6 is too strict for float32 computations with cosh/sinh
        is_valid_moderate, _, _ = check_lorentz_manifold_validity(
            sample_lorentz_embeddings, curvature=1.0, tolerance=1e-4
        )

        assert is_valid_loose
        assert is_valid_moderate

# -------------------------------------------------------------------------------------------------
# Hyperbolic Radii Tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHyperbolicRadii:
    '''Test suite for hyperbolic radius computation.'''

    def test_compute_radii_shape(self, sample_lorentz_embeddings):
        '''Test that radii computation produces correct shape.'''

        radii = compute_hyperbolic_radii(sample_lorentz_embeddings)

        assert radii.shape == (sample_lorentz_embeddings.shape[0], )

    def test_radii_are_positive(self, sample_lorentz_embeddings):
        '''Test that hyperbolic radii are always positive.'''

        radii = compute_hyperbolic_radii(sample_lorentz_embeddings)

        assert torch.all(radii > 0), 'Hyperbolic radii must be positive'

    def test_radii_equal_time_coordinate(self, sample_lorentz_embeddings):
        '''Test that radii equal the time coordinate (x₀).'''

        radii = compute_hyperbolic_radii(sample_lorentz_embeddings)
        time_coords = sample_lorentz_embeddings[:, 0]

        assert torch.allclose(radii, time_coords, atol=1e-8)

    def test_origin_has_unit_radius(self, test_device):
        '''Test that the origin on hyperboloid has radius 1.'''

        # Origin in Lorentz model: (1, 0, 0, ..., 0)
        dim = 385
        origin = torch.zeros(1, dim, device=test_device)
        origin[0, 0] = 1.0

        radius = compute_hyperbolic_radii(origin)

        assert torch.allclose(radius, torch.tensor([1.0], device=test_device), atol=1e-6)

# -------------------------------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestHyperbolicProperties:
    '''Property-based tests using Hypothesis for robustness.'''

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        dim=st.integers(min_value=64, max_value=512),
    )
    @settings(max_examples=10, deadline=None)
    def test_exp_preserves_batch_size_property(self, batch_size, dim):
        '''Property test: exp_map preserves batch size for any valid input.'''

        tangent = torch.randn(batch_size, dim + 1)
        hyp = LorentzOps.exp_map_zero(tangent, c=1.0)

        assert hyp.shape[0] == batch_size
        assert hyp.shape[1] == dim + 1

    @given(curvature=st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=10, deadline=None)
    def test_exp_produces_valid_manifold_property(self, curvature):
        '''Property test: exp_map always produces valid manifold points.'''

        # Create tangent vectors with controlled norms for numerical stability
        tangent = torch.randn(8, 385)
        tangent[:, 0] = 0.0  # Time component should be 0 for tangent at origin
        # Scale to reasonable norm (around 2) to avoid sinh/cosh overflow
        tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8) * 2.0

        hyp = LorentzOps.exp_map_zero(tangent, c=curvature)

        is_valid, _, _ = check_lorentz_manifold_validity(hyp, curvature=curvature, tolerance=1e-2)

        assert is_valid
