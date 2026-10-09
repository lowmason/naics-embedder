'''
The geometry heads of Req 12's three arms: Euclidean, spherical and hyperbolic.

Every head takes the projection's output v (B, d) and returns ``HeadPoints``:

- ``tangent``: the coordinates the export writes, Req 2's form of the arm. The hyperbolic head
  writes its bounded tangent r · û, the Euclidean head v and the spherical head û;
- ``embedding``: the point the arm's decoding distance reads. It is the Lorentz point (B, d + 1)
  under hyperbolic and ``tangent`` itself otherwise;
- ``radius`` and ``direction``: the polar parts that the code cache keeps and the training
  distance reads.

Each head also names its ``geometry`` and its decoding ``distance`` (a ``panels.decoding``
``DISTANCES`` name). ``radial`` says whether the radial term applies, which it does only in the
hyperbolic arm (Req 12). ``pair_distance`` is its training distance over polar parts, (A, B) in
the inputs' dtype. ``read_points`` maps exported coordinates to the float64 CPU points its
distance reads. The hyperbolic head is ``hyperbolic.HyperbolicHead``; this module adds the two
flat heads and builds a head by name.
'''

# -------------------------------------------------------------------------------------------------
# Imports and settings
# -------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
import torch.nn as nn

from naics_embedder.text_model.hyperbolic import (
    HeadPoints,
    HyperbolicHead,
    _refuse_unpaired_shapes,
    _where_positive,
)

# Req 12's geometry arms, in the order of model.geometry's choices
GEOMETRIES = ('euclidean', 'spherical', 'hyperbolic')

# -------------------------------------------------------------------------------------------------
# The flat distances
# -------------------------------------------------------------------------------------------------

def flat_distance(
    radius_a: torch.Tensor,
    direction_a: torch.Tensor,
    radius_b: torch.Tensor,
    direction_b: torch.Tensor,
) -> torch.Tensor:
    '''
    The Euclidean distance ‖v_a − v_b‖ between every point of one set and every point of another,
    from each point's radius r = ‖v‖ and direction û.

    The polar form is exact:

        ‖v_a − v_b‖ = √((r_a − r_b)² + r_a · r_b · ‖û_a − û_b‖²)

    Every term is non-negative, so nothing cancels, and ‖û_a − û_b‖² comes from explicit
    differences, as in ``polar_distance``. The square root is guarded at zero, where an anchor
    meets its own live row: the distance is 0, and the value and its gradient stay finite.

    At v = 0 exactly, or where the float32 norm underflows, the polar form gives the point no
    gradient, as ``polar_distance`` does for the hyperbolic head (P4).

    Args:
        radius_a: The first set's radii, (A,).
        direction_a: Its directions, (A, d): unit vectors, or 0 at the origin.
        radius_b: The second set's radii, (B,).
        direction_b: Its directions, (B, d).

    Returns:
        The distances, (A, B), in the inputs' dtype.

    Raises:
        ValueError: If the shapes are not (A,), (A, d), (B,) and (B, d).
    '''

    _refuse_unpaired_shapes(radius_a, direction_a, radius_b, direction_b, name='flat_distance')
    gap = radius_a.unsqueeze(1) - radius_b.unsqueeze(0)
    chord = (direction_a.unsqueeze(1) - direction_b.unsqueeze(0)).square().sum(dim=2)
    squared = gap.square() + radius_a.unsqueeze(1) * radius_b.unsqueeze(0) * chord
    separated = squared > 0
    return torch.where(
        separated, torch.sqrt(_where_positive(squared, separated)), torch.zeros_like(squared)
    )

def chord_distance(
    radius_a: torch.Tensor,
    direction_a: torch.Tensor,
    radius_b: torch.Tensor,
    direction_b: torch.Tensor,
) -> torch.Tensor:
    '''
    The cosine distance 1 − cos θ between every direction of one set and every direction of
    another, in chord form: ‖û_a − û_b‖² / 2.

    For unit vectors the two are equal, and the chord form takes explicit differences, so it
    resolves small angles in float32, where 1 − û_a · û_b cancels. The radii are not read: on the
    sphere every point is at distance 1 from the origin. The gradient is finite everywhere, zero
    separation included.

    Args:
        radius_a: The first set's radii, (A,); checked for shape only.
        direction_a: Its directions, (A, d).
        radius_b: The second set's radii, (B,); checked for shape only.
        direction_b: Its directions, (B, d).

    Returns:
        The distances, (A, B), in the inputs' dtype.

    Raises:
        ValueError: If the shapes are not (A,), (A, d), (B,) and (B, d).
    '''

    _refuse_unpaired_shapes(radius_a, direction_a, radius_b, direction_b, name='chord_distance')
    return (direction_a.unsqueeze(1) - direction_b.unsqueeze(0)).square().sum(dim=2) / 2

# -------------------------------------------------------------------------------------------------
# The flat heads
# -------------------------------------------------------------------------------------------------

def _polar_parts(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Each vector's norm (B, 1) and unit direction (B, d); the direction is 0 where v is 0.'''

    norm = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    # Guarded by where, not a clamp: at v = 0 the direction and its gradient are 0, not 1/ε
    moving = norm > 0
    direction = torch.where(
        moving, vectors / _where_positive(norm, moving), torch.zeros_like(vectors)
    )
    return norm, direction

class EuclideanHead(nn.Module):
    '''
    The Euclidean head (Req 12): the point is the projection's output v itself.

    Its radius is r = ‖v‖ and its direction û = v / ‖v‖ (0 at v = 0): the polar parts the code
    cache keeps. The training distance is ‖v_a − v_b‖ in polar form (``flat_distance``). The
    export writes v, and a read takes it as it is, in float64. The head has no parameters and no
    bound, so ``model.radius_bound`` is not read, and the radial term does not apply (P4).
    '''

    geometry = 'euclidean'
    distance = 'euclidean'
    radial = False

    def forward(self, vectors: torch.Tensor) -> HeadPoints:
        '''
        The point v, with its radius ‖v‖ and direction û, in the vectors' dtype.

        Args:
            vectors: The projection's outputs, (B, d).

        Returns:
            v as both the tangent and the embedding, the radius and the direction.
        '''

        norm, direction = _polar_parts(vectors)
        return HeadPoints(vectors, vectors, norm.squeeze(1), direction)

    @staticmethod
    def pair_distance(
        radius_a: torch.Tensor,
        direction_a: torch.Tensor,
        radius_b: torch.Tensor,
        direction_b: torch.Tensor,
    ) -> torch.Tensor:
        '''The training distance between two sets of the head's points: ``flat_distance``.'''

        return flat_distance(radius_a, direction_a, radius_b, direction_b)

    @staticmethod
    def read_points(tangent: torch.Tensor) -> torch.Tensor:
        '''Exported points as they are, float64 on the CPU.'''

        # .cpu() before the cast: casting an MPS tensor to float64 raises
        return tangent.cpu().to(torch.float64)

class SphericalHead(nn.Module):
    '''
    The spherical head (Req 12): the point is the direction û = v / ‖v‖ on the unit sphere.

    Its radius is the constant 1, or 0 at v = 0, and takes no gradient: every point on the sphere
    is at distance 1 from the origin, and the radial term does not apply. The training distance is
    the chord form of the cosine distance (``chord_distance``). The export writes û, and a read
    takes it as it is, in float64. It writes û rather than v because, under the cosine distance,
    v's norm receives no training signal, so the regressor panel would read an untrained norm
    (P5).
    '''

    geometry = 'spherical'
    distance = 'cosine'
    radial = False

    def forward(self, vectors: torch.Tensor) -> HeadPoints:
        '''
        The point û, with the constant radius 1 (0 at v = 0), in the vectors' dtype.

        Args:
            vectors: The projection's outputs, (B, d).

        Returns:
            û as the tangent, the embedding and the direction, and the radius.
        '''

        norm, direction = _polar_parts(vectors)
        radius = (norm.squeeze(1) > 0).to(vectors.dtype)
        return HeadPoints(direction, direction, radius, direction)

    @staticmethod
    def pair_distance(
        radius_a: torch.Tensor,
        direction_a: torch.Tensor,
        radius_b: torch.Tensor,
        direction_b: torch.Tensor,
    ) -> torch.Tensor:
        '''The training distance between two sets of the head's points: ``chord_distance``.'''

        return chord_distance(radius_a, direction_a, radius_b, direction_b)

    @staticmethod
    def read_points(tangent: torch.Tensor) -> torch.Tensor:
        '''Exported directions as they are, float64 on the CPU.'''

        # .cpu() before the cast: casting an MPS tensor to float64 raises
        return tangent.cpu().to(torch.float64)

# -------------------------------------------------------------------------------------------------
# A head by name
# -------------------------------------------------------------------------------------------------

def build_head(geometry: str, *, radius_bound: float = 8.0) -> nn.Module:
    '''
    The head of a geometry arm (``model.geometry``).

    Args:
        geometry: One of ``GEOMETRIES``.
        radius_bound: R, read by the hyperbolic head only (``model.radius_bound``).

    Returns:
        The head, with no parameters.

    Raises:
        ValueError: If the geometry is unknown, or as ``HyperbolicHead`` for its bound.
    '''

    if geometry == 'hyperbolic':
        return HyperbolicHead(radius_bound=radius_bound)
    if geometry == 'euclidean':
        return EuclideanHead()
    if geometry == 'spherical':
        return SphericalHead()
    raise ValueError(f'unknown geometry {geometry!r}; expected one of {list(GEOMETRIES)}')

def head_of(model: nn.Module) -> nn.Module:
    '''
    A model's geometry head: the Lightning module's encoder's, or a shared encoder's own.

    Args:
        model: The Lightning module, or its shared encoder.

    Returns:
        The head.
    '''

    return getattr(model, 'encoder', model).head
