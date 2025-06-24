# Very very experimental code for flood field reconstruction and analysis.
# This is not yet implemented and is a placeholder for future work. 
# When I am a bit smarter and more competent.

import torch
import torch.nn as nn
from cascadia import constants

class FloodFieldBuilder(nn.Module):
    """Reconstructs full flood fields from backbone depth and categorical attributes."""
    def forward(self, X_backbone, D):
        # Placeholder: Add transformation logic to reconstruct flood field
        return X_backbone + D.unsqueeze(-1).float() * 0.01  # simple synthetic logic


class FieldAngles(nn.Module):
    """Converts flood fields into feature angles or other structural encodings."""
    def forward(self, X_full, mask=None):
        # Placeholder: Derive pseudo-angle features from X_full 
        # This could be based on spatial gradients, curvature, or other properties
        # I will not be implementing this soon 
        dx = torch.diff(X_full, dim=-2, prepend=X_full[:, :, :1])
        angles = torch.atan2(dx[..., 1], dx[..., 0])
        return angles


class LossFloodfieldRMSD(nn.Module):
    """Computes RMSD between predicted and ground truth flood fields."""
    def forward(self, X_pred, X_true, mask=None):
        if mask is not None:
            diff = (X_pred - X_true) * mask
            n = mask.sum()
        else:
            diff = X_pred - X_true
            n = X_pred.numel()
        return torch.sqrt((diff ** 2).sum() / n)


class LossFloodfieldClashes(nn.Module):
    """Penalizes overlapping flood regions or physically implausible configurations."""
    def forward(self, X_field):
        # e.g. penalize high curvature or proximity 
        diff = torch.diff(X_field, dim=-2)
        norm = torch.norm(diff, dim=-1)
        penalty = torch.relu(1.0 - norm)  # encourage spacing
        return penalty.mean()


class FloodField(nn.Module):
    def __init__(self):
        super().__init__()
        self.field_to_X = FloodFieldBuilder()
        self.X_to_field = FieldAngles()
        self.loss_rmsd = LossFloodfieldRMSD()
        self.loss_clash = LossFloodfieldClashes()

    def forward(self, X_backbone, D):
        X_full = self.field_to_X(X_backbone, D)
        angles = self.X_to_field(X_full)
        return X_full, angles

    def compute_losses(self, X_pred, X_true, mask=None):
        rmsd = self.loss_rmsd(X_pred, X_true, mask)
        clash = self.loss_clash(X_pred)
        return {"rmsd": rmsd, "clash": clash}

def _gather_grid_square_mask(C, S, atoms_per_aa, num_atoms):
    device = S.device
    atoms_per_aa = torch.tensor(atoms_per_aa, dtype=torch.long)
    atoms_per_aa = atoms_per_aa.to(device).unsqueeze(0).expand(S.shape[0], -1)

    # (B,A) @ (B,L)  => (B,L)
    atoms_per_residue = torch.gather(atoms_per_aa, -1, S)
    atoms_per_residue = (C > 0).float() * atoms_per_residue

    ix_expand = torch.arange(num_atoms, device=device).reshape([1, 1, -1])
    mask_atoms = ix_expand < atoms_per_residue.unsqueeze(-1)
    mask_atoms = mask_atoms.float()
    return mask_atoms

def grid_square_mask(C, D):
    """Constructs a all-atom coordinate mask from a sequence and chain map.

    Inputs:
        C (tensor): Chain map with shape `(batch_size, HxW)`.
        D (tensor): Descriptor tokens with shape `(batch_size, HxW)`.

    Outputs:
        mask_atoms (tensor): Atomic mask with shape
            `(batch_size, num_residues, 16)`.
    """
    return _gather_grid_square_mask(C, D, constants.NLCD, 16)