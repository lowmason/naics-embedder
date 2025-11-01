# src/naics_gemini/utils/hyperbolic.py
import torch

def lorentz_dot(u, v):
    """Lorentzian inner product."""
    uv = u * v
    # The first coordinate has a negative sign
    return torch.sum(uv[:, 1:], dim=1) - uv[:, 0]

def lorentz_distance(u, v, c=1.0):
    """Calculate the distance between two points in the Lorentz model."""
    dot_product = lorentz_dot(u, v)
    # Clamp to avoid numerical errors leading to values slightly less than -1
    clamped_dot = torch.clamp(dot_product, min=-1.0)
    dist = torch.sqrt(torch.tensor(c)) * torch.acosh(-clamped_dot)
    return dist

def exp_map_zero(v, c=1.0):
    """Exponential map from the tangent space at the origin to the hyperboloid."""
    sqrt_c = torch.sqrt(torch.tensor(c))
    norm_v = torch.norm(v, p=2, dim=1, keepdim=True)
    # Avoid division by zero for zero vectors
    norm_v = torch.clamp(norm_v, min=1e-8)
    
    sinh_term = torch.sinh(norm_v / sqrt_c)
    
    # Project to hyperboloid
    x0 = torch.cosh(norm_v / sqrt_c)
    x_rest = (sinh_term * v) / norm_v
    
    return torch.cat([x0, x_rest], dim=1)
