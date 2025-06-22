"""Layers for generating flood extents

This module contains pytorch layers for parametrically generating and
manipulating flood backbones. These can be used in tandem with loss functions
to generate and optimize flood structures (e.g. spatial extent / depth from predictions) or
used as an intermediate layer in a learned structure generation model.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cascadia.layers.structure import geometry, transforms


class FloodBackbone(nn.Module):
    """
    Flood backbone layer with optimizable spatial extent or depth.
    Produces a 2D flood grid with spatial perturbations and optional transformation.
    
    Args:
        grid_height (int): Number of rows in the flood map.
        grid_width (int): Number of columns in the flood map.
        num_batch (int): Batch size.
        use_terrain_slope (bool): Whether to optimize flow slope vectors.
        X_init (torch.Tensor, optional): Predefined flood extent (B, H, W).
    Outputs:
        X (torch.Tensor): Flood extent/depth grid of shape (B, H, W).
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_batch: int = 1,
        use_terrain_slope: bool = False,
        X_init: Optional[torch.Tensor] = None,
    ):
        super(FloodBackbone, self).__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_batch = num_batch
        self.use_terrain_slope = use_terrain_slope

        # Initialize flood surface as a learnable parameter
        if X_init is not None:
            self.X = nn.Parameter(X_init)
        else:
            self.X = nn.Parameter(
                torch.randn(num_batch, grid_height, grid_width) * 0.01  # start small
            )

        # Optional: terrain slope as a proxy for internal flow structure
        if self.use_terrain_slope:
            self.slope_x = nn.Parameter(torch.zeros(num_batch, grid_height, grid_width))
            self.slope_y = nn.Parameter(torch.zeros(num_batch, grid_height, grid_width))

        # Optional global transformation (e.g., shifting flood centroid)
        self.transform = RigidTransform2D(num_batch=num_batch)

    def forward(self) -> torch.Tensor:
        X_transformed = self.transform(self.X)
        if self.use_terrain_slope:
            # Optionally return slope fields for advanced GNN or physics constraints
            return X_transformed, self.slope_x, self.slope_y
        return X_transformed
    
class RigidTransform2D(nn.Module):
    """2D shift and rotation for flood maps."""
    def __init__(self, num_batch: int = 1, scale_dx: float = 1.0, scale_angle: float = 1.0):
        super().__init__()
        self.dx = nn.Parameter(torch.zeros(num_batch, 2))  # x, y shift
        self.angle = nn.Parameter(torch.zeros(num_batch))  # rotation angle
        self.scale_dx = scale_dx
        self.scale_angle = scale_angle

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Apply shift and rotation to 2D map
        # X shape: (B, H, W)
        B, H, W = X.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=X.device),
            torch.arange(W, device=X.device),
            indexing='ij'
        )
        coords = torch.stack([grid_x, grid_y], dim=0).float()  # (2, H, W)

        # Center coords
        coords -= torch.tensor([[W / 2], [H / 2]], device=X.device).unsqueeze(-1)

        transformed = []
        for i in range(B):
            theta = self.angle[i] * self.scale_angle
            R = torch.tensor([
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta),  torch.cos(theta)]
            ], device=X.device)
            shifted = R @ coords.view(2, -1)
            shifted = shifted + self.dx[i][:, None] * self.scale_dx
            shifted = shifted.view(2, H, W)

            # Bilinear sample
            grid = torch.stack([shifted[1] / H * 2, shifted[0] / W * 2], dim=-1)  # normalize
            grid = grid.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)
            X_i = X[i].unsqueeze(0).unsqueeze(0)
            X_out = torch.nn.functional.grid_sample(X_i, grid, mode='bilinear', align_corners=True)
            transformed.append(X_out.squeeze())

        return torch.stack(transformed, dim=0)
    
class FloodBackbone(nn.Module):
    """
    Backbone builder for flood extent geometry.

    This module initializes and optionally optimizes flood extent maps
    from a latent parameterization.

    Args:
        grid_shape (Tuple[int, int]): Height and width of the flood extent grid.
        num_batch (int): Number of samples per batch.
        init_mode (str): Mode for initialization ('zeros', 'noise', or 'gaussian').
        use_latent_params (bool): Whether to initialize a learnable latent code.

    Output:
        X (Tensor): Flood extent tensor of shape (batch_size, H, W, 1)
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        num_batch: int = 1,
        init_mode: str = "zeros",  # 'zeros', 'noise', or 'gaussian'
        use_latent_params: bool = False,
    ):
        super(FloodBackbone, self).__init__()
        self.H, self.W = grid_shape
        self.num_batch = num_batch
        self.use_latent_params = use_latent_params

        if self.use_latent_params:
            self.latent = nn.Parameter(torch.randn(num_batch, 32))  # Learnable seed
            self.decoder = nn.Sequential(
                nn.Linear(32, self.H * self.W),
                nn.ReLU(),
                nn.Unflatten(1, (self.H, self.W)),
            )
        else:
            if init_mode == "zeros":
                X_init = torch.zeros(num_batch, self.H, self.W)
            elif init_mode == "noise":
                X_init = torch.rand(num_batch, self.H, self.W)
            elif init_mode == "gaussian":
                X_init = torch.randn(num_batch, self.H, self.W)
            else:
                raise ValueError("Unsupported init_mode")

            self.X = nn.Parameter(X_init)

    def forward(self) -> torch.Tensor:
        if self.use_latent_params:
            X = self.decoder(self.latent)
        else:
            X = self.X
        X = X.unsqueeze(-1)  # Add channel dim (e.g., for flood depth)
        X = X - X.mean(dim=(1, 2, 3), keepdim=True)  # Optional centering
        return X
    
class FloodFrameBuilder(nn.Module):
    """
    Build local flood patches from transformation parameters.

    Args:
        patch_template (torch.Tensor): Template local patch shape,
            shape `(1, 1, P, 2)` for a 2D flood patch with P points.
        distance_eps (float): Small epsilon for normalization stability.

    Inputs:
        R (torch.Tensor): Local rotation matrices (B, N, 2, 2) or angles (B, N, 1)
        t (torch.Tensor): Local translation vectors (B, N, 2) - e.g., flood cell centers.
        C (torch.Tensor): Connectivity or activation mask (B, N).
        q (optional): If using 2D rotations via quaternions or angles.

    Outputs:
        X (torch.Tensor): Reconstructed flood patch geometry (B, N, P, 2)
    """

    def __init__(self, patch_template: torch.Tensor, distance_eps: float = 1e-5):
        super().__init__()
        self.register_buffer("patch_template", patch_template)  # shape: (1, 1, P, 2)
        self.distance_eps = distance_eps

    def forward(
        self,
        R: Optional[torch.Tensor],  # (B, N, 2, 2)
        t: torch.Tensor,            # (B, N, 2)
        C: torch.Tensor,            # (B, N)
        q: Optional[torch.Tensor] = None,  # Optional: angles or quaternions
    ):
        B, N, _ = t.shape
        P = self.patch_template.shape[2]

        # If R is not given, convert angle or q to rotation matrix
        if R is None:
            angle = q.squeeze(-1)  # (B, N)
            cos, sin = torch.cos(angle), torch.sin(angle)
            R = torch.stack([
                torch.stack([cos, -sin], dim=-1),
                torch.stack([sin,  cos], dim=-1),
            ], dim=-2)  # shape: (B, N, 2, 2)

        # Broadcast patch template to (B, N, P, 2)
        patch = self.patch_template.expand(B, N, P, 2)

        # Apply rotation
        patch_rotated = torch.einsum("bnij,bnpj->bnpi", R, patch)

        # Apply translation
        patch_translated = patch_rotated + t.unsqueeze(2)

        # Apply mask
        mask = C.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        patch_masked = patch_translated * mask

        return patch_masked
    
class GraphBackboneUpdate(nn.Module):
    """Graph-based flood extent updater using node and edge embeddings.

    Args:
        dim_nodes (int): Node embedding size.
        dim_edges (int): Edge embedding size.
        distance_scale (float): Multiplier for spatial delta predictions.
        method (str): Update method: 'local', 'neighbor', 'neighbor_global'.
        iterations (int): Number of iterations for equilibrium.
        unconstrained (bool): If True, allows local refinement beyond initial guess.
    """

    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        distance_scale: float = 1.0,
        method: str = "neighbor",
        iterations: int = 1,
        unconstrained: bool = True,
    ):
        super(GraphBackboneUpdate, self).__init__()
        self.distance_scale = distance_scale
        self.iterations = iterations
        self.method = method
        self.unconstrained = unconstrained

        if method == "local":
            self.W_update = nn.Linear(dim_nodes, 1)
        elif method == "neighbor":
            self.W_update = nn.Linear(dim_edges, 1)
        elif method == "neighbor_global":
            self.W_update = nn.Linear(dim_edges + dim_nodes, 1)

        if unconstrained:
            self.W_refine = nn.Linear(dim_nodes, 1)

    def forward(
        self,
        X: torch.Tensor,            # (B, N, 1) flood extent
        node_h: torch.Tensor,       # (B, N, Dn)
        edge_h: torch.Tensor,       # (B, N, K, De)
        edge_idx: torch.LongTensor, # (B, N, K)
        mask_i: torch.Tensor,       # (B, N)
        mask_ij: torch.Tensor       # (B, N, K)
    ) -> torch.Tensor:

        B, N, _ = X.shape

        if self.method == "local":
            dX = self.W_update(node_h) * self.distance_scale
            X_update = X + dX

        elif self.method in ["neighbor", "neighbor_global"]:
            dX = self.W_update(edge_h).squeeze(-1)  # (B, N, K)
            weights = torch.sigmoid(dX) * mask_ij  # (B, N, K)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)

            # Gather neighbor values
            X_neighbors = torch.gather(
                X.squeeze(-1), 1, edge_idx
            )  # (B, N, K)

            X_pred = (weights * X_neighbors).sum(-1, keepdim=True)  # (B, N, 1)
            X_update = X_pred

            if self.method == "neighbor_global":
                global_context = node_h.mean(1, keepdim=True).expand(-1, N, -1)
                dX_global = self.W_update(torch.cat([edge_h.mean(2), global_context], dim=-1))
                X_update += self.distance_scale * dX_global

        if self.unconstrained:
            d_refine = self.W_refine(node_h)
            X_update += d_refine

        return X_update