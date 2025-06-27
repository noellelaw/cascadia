import torch
import torch.nn as nn

class GridDistances(nn.Module):
    """Compute Euclidean distances between grid cells."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None):
        # X: (B, H, W) = coordinates or centroids
        B, H, W = X.shape
        X_flat = X.view(B, H * W, 1)
        dist = torch.cdist(X_flat, X_flat, p=2).view(B, H, W, H, W)
        if mask is not None:
            dist = dist * mask.unsqueeze(1).unsqueeze(1)
        return dist


class FlowAngles(nn.Module):
    """Compute angles between flow gradients."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, grad_x: torch.Tensor, grad_y: torch.Tensor):
        # Compute angles (in radians) between horizontal and vertical slopes
        dot = (grad_x * grad_y).sum(dim=-1)
        norm_x = grad_x.norm(dim=-1)
        norm_y = grad_y.norm(dim=-1)
        cos_angle = dot / (norm_x * norm_y + self.eps)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        return angle

class VirtualNodes(nn.Module):
    # Something done in chroma that could be used for inferred locations
    # for cascading infrastructure 
    def __init__(self, offset=(1.0, 0.0)):
        super().__init__()
        self.offset = torch.tensor(offset).view(1, 1, 1, 2)

    def forward(self, X: torch.Tensor):
        # X: (B, H, W, 2)=position of each grid cell
        return X + self.offset.to(X.device)

def normed_vec(V: torch.Tensor, distance_eps: float = 1e-3) -> torch.Tensor:
    """Normalized vectors with distance smoothing.

    This normalization is computed as `U = V / sqrt(|V|^2 + eps)` to avoid cusps
    and gradient discontinuities.

    Args:
        V (Tensor): Batch of vectors with shape `(..., num_dims)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        U (Tensor): Batch of normalized vectors with shape `(..., num_dims)`.
    """
    # Unit vector from i to j
    mag_sq = (V ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + distance_eps)
    U = V / mag
    return U

def normed_cross(
    V1: torch.Tensor, V2: torch.Tensor, distance_eps: float = 1e-3
) -> torch.Tensor:
    """Normalized cross product between vectors.

    This normalization is computed as `U = V / sqrt(|V|^2 + eps)` to avoid cusps
    and gradient discontinuities.

    Args:
        V1 (Tensor): Batch of vectors with shape `(..., 3)`.
        V2 (Tensor): Batch of vectors with shape `(..., 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        C (Tensor): Batch of cross products `v_1 x v_2` with shape `(..., 3)`.
    """
    C = normed_vec(torch.cross(V1, V2, dim=-1), distance_eps=distance_eps)
    return C


def frames_from_backbone(X: torch.Tensor, distance_eps: float = 1e-3):
    """Convert a backbone into local reference frames.

    Args:
        X (Tensor): Backbone coordinates with shape `(..., 4, 3)`.
        distance_eps (float, optional): Distance smoothing parameter for
            for computing distances as `sqrt(sum_sq) -> sqrt(sum_sq + eps)`.
            Default: 1E-3.

    Returns:
        R (Tensor): Reference frames with shape `(..., 3, 3)`.
        X_CA (Tensor): C-alpha coordinates with shape `(..., 3)`
    """
    X_N, X_CA, X_C, X_O = X.unbind(-2)
    u_CA_N = normed_vec(X_N - X_CA, distance_eps)
    u_CA_C = normed_vec(X_C - X_CA, distance_eps)
    n_1 = u_CA_N
    n_2 = normed_cross(n_1, u_CA_C, distance_eps)
    n_3 = normed_cross(n_1, n_2, distance_eps)
    R = torch.stack([n_1, n_2, n_3], -1)
    return R, X_CA