"""Layers for building graph representations of flood structure.

This module contains pytorch layers for representing flood structure as a
graph with node and edge features based on geometric information. The graph
features are differentiable with respect to input coordinates and can be used
for building flood scoring functions and optimizing flood geometries
natively in pytorch.
"""

import json
import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cascadia.data.flood import Flood
from cascadia.layers import graph
from cascadia.layers.basic import FourierFeaturization, PositionalEncoding
from cascadia.layers.structure import backbone, geometry, transforms


class FloodFeatureGraph(nn.Module):
    """Graph featurizer for flood fields.

    This module builds graph representations of flood structures that are
    differentiable with respect to input coordinates and invariant with respect
    to global rotations and translations. It takes as input a batch of
    flood backbones, constructs a sparse graph
    with grid squares as nodes, and featurizes the backbones in terms of node and
    edge feature tensors.

    The graph representation has 5 components:
        1. Node features `node_h` representing grid squares in the flood field.
        2. Edge features `edge_h` representing relationships between grid squares.
        3. Index map `edge_idx` representing graph topology.
        4. Node mask `mask_i` that specifies which nodes are present.
        5. Edge mask `mask_ij` that specifies which edges are present.

    Criteria for constructing the graph currently include k-Nearest Neighbors or
    distance-weighted edge sampling.

    Node and edge features are specified as tuples to make it simpler to add
    additional features and options while retaining backwards compatibility.
    Specifically, each node or edge feature type can be added to the list either
    in default configuration by a `'feature_name'` keyword, or in modified form
    with a `('feature_name', feature_kwargs)` tuple.

    Example usage:
        graph = FloodFeatureGraph(
            graph_type='knn',
            node_features=('dihedrals',),
            edge_features=[
                'field_distance',
                ('dmat_6grid', {'D_function': 'log'})
            ]
        )
        node_h, edge_h, edge_idx, mask_i, mask_ij = graph(X, C)

        This builds a kNN graph with flood-relevant node features (e.g., elevation, depth, landcover) 
        as node features and 6-grid cell neighborhoods to encode local hydrolic / spatial context, 
        where the options for post-processing are passed as a kwargs dict.

    Args:
        dim_nodes (int): Hidden dimension of node features.
        dim_edges (int): Hidden dimension of edge features.
        num_neighbors (int): Maximum degree of the graph.
        graph_kwargs (dict): Arguments for graph construction. Default is None.
        node_features (list): List of node feature strings and optional args.
            Valid feature strings are `{internal_coords}`.
        edge_features (list): List of node feature strings and optional args.
            Valid feature strings are `{'distances_6grid','distances_field'}`.
        centered (boolean): Flag for enabling feature centering. If `True`,
            the features will be will centered by subtracting an empirical mean
            that was computed on the reference PDB `centered_pdb`. The statistics
            are per-dimension of every node and edge feature. If they have not
            previously been computed, the PDB will be downloaded, featurized,
            and aggregated into local statistics that are cached in the repo.
        centered_pdb (str): PDB code for the reference PDB to compute some
            empirical feature statistics from.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, HxW, 1)`. 
        C (LongTensor, optional): field map with shape
            `(num_batch, HxW)`. The field map codes positions as `0`
            when masked, positive integers for field indices, and negative
            integers to represent missing residues of the corresponding
            positive integers.
        custom_D (Tensor, optional): Pre-computed custom distance map
            for graph construction `(numb_batch,num_residues,num_residues)`.
            If present, this will override the behavior of `graph_type` and used
            as the distances for k-nearest neighbor graph construction.
        custom_mask_2D (Tensor, optional): Custom 2D mask to apply to `custom_D`
            with shape `(numb_batch,num_residues,num_residues)`.

    Outputs:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_residues, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_residues, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_residues, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_residues)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
    """

    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        num_neighbors: int = 30,
        graph_kwargs: dict = None,
        node_features: tuple = ("internal_coords",),
        edge_features: tuple = ("distances_6grid", "distances_field"),
        centered: bool = True,
        centered_pdb: str = "2g3n",
    ):
        super(FloodFeatureGraph, self).__init__()

        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.num_neighbors = num_neighbors
        graph_kwargs = graph_kwargs if graph_kwargs is not None else {}
        self.graph_builder = FloodGraph(num_neighbors, **graph_kwargs)
        self.node_features = node_features
        self.edge_features = edge_features

        def _init_layer(layer_dict, features):
            # Parse option string
            custom_args = not isinstance(features, str)
            key = features[0] if custom_args else features
            kwargs = features[1] if custom_args else {}
            return layer_dict[key](**kwargs)

        # Node feature compilation
        node_dict = {
            "internal_coords": FloodNodeInternalCoords,
            "cartesian_coords": NodeCartesianCoords,
            "radii": NodeRadii,
        }
        self.node_layers = nn.ModuleList(
            [_init_layer(node_dict, option) for option in self.node_features]
        )
        # Edge feature compilation
        edge_dict = {
            "distances_6grid": EdgeDistance6grid,
            "distances_2grid": EdgeDistance2grid,
            "orientations_2grid": EdgeOrientationField,
            "position_2grid": EdgePositionalEncodings,
            "distances_field": EdgeDistanceField,
            "orientations_field": EdgeOrientationField,
            "cartesian_coords": EdgeCartesianCoords,
            "random_fourier_2grid": EdgeRandomFourierFeatures2grid,
        }
        self.edge_layers = nn.ModuleList(
            [_init_layer(edge_dict, option) for option in self.edge_features]
        )

        # Load feature centering params as buffers
        self.centered = centered
        self.centered_pdb = centered_pdb.lower()
        if self.centered:
            self._load_centering_params(self.centered_pdb)

        """
            Storing separate linear transformations for each layer, rather than concat + one
            large linear, provides a more even weighting of the different input
            features when used with standard weight initialization. It has the
            specific effect actually re-weighting the weight variance based on
            the number of input features for each feature type. Otherwise, the
            relative importance of each feature goes with the number of feature
            dimensions.
        """
        self.node_linears = nn.ModuleList(
            [nn.Linear(l.dim_out, self.dim_nodes) for l in self.node_layers]
        )
        self.edge_linears = nn.ModuleList(
            [nn.Linear(l.dim_out, self.dim_edges) for l in self.edge_layers]
        )
        return
    
    def forward(
        self,
        X: torch.Tensor,
        C: torch.Tensor,
        edge_idx: Optional[torch.LongTensor] = None,
        mask_ij: torch.Tensor = None,
        custom_D: Optional[torch.Tensor] = None,
        custom_mask_2D: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor
    ]:
        mask_i = field_map_to_mask(C)
        if mask_ij is None or edge_idx is None:
            edge_idx, mask_ij = self.graph_builder(
                X, C, custom_D=custom_D, custom_mask_2D=custom_mask_2D
            )

        # Aggregate node layers
        node_h = None
        for i, layer in enumerate(self.node_layers):
            node_h_l = layer(X, edge_idx, C)
            if self.centered:
                node_h_l = node_h_l - self.__getattr__(f"node_means_{i}")
            node_h_l = self.node_linears[i](node_h_l)
            node_h = node_h_l if node_h is None else node_h + node_h_l
        if node_h is None:
            node_h = torch.zeros(list(X.shape[:2]) + [self.dim_nodes], device=X.device)

        # Aggregate edge layers
        edge_h = None
        for i, layer in enumerate(self.edge_layers):
            edge_h_l = layer(X, edge_idx, C)
            if self.centered:
                edge_h_l = edge_h_l - self.__getattr__(f"edge_means_{i}")
            edge_h_l = self.edge_linears[i](edge_h_l)
            edge_h = edge_h_l if edge_h is None else edge_h + edge_h_l
        if edge_h is None:
            edge_h = torch.zeros(list(X.shape[:2]) + [self.dim_nodes], device=X.device)

        # Apply masks
        node_h = mask_i.unsqueeze(-1) * node_h
        edge_h = mask_ij.unsqueeze(-1) * edge_h

        return node_h, edge_h, edge_idx, mask_i, mask_ij

    def _load_centering_params(self, reference_pdb: str):
        basepath = os.path.join(tempfile.gettempdir(), "generate", "params")
        if not os.path.exists(basepath):
            os.makedirs(basepath)

        filename = f"centering_{reference_pdb}.params"
        self.centering_file = os.path.join(basepath, filename)
        key = (
            reference_pdb
            + ";"
            + json.dumps(self.node_features)
            + ";"
            + json.dumps(self.edge_features)
        )

        # Attempt to load saved centering params, otherwise compute and cache
        json_line = None
        with open(self.centering_file, "a+") as f:
            prefix = key + "\t"
            f.seek(0)
            for line in f:
                if line.startswith(prefix):
                    json_line = line.split(prefix)[1]
                    break

            if json_line is not None:
                print("Loaded from cache")
                param_dictionary = json.loads(json_line)
            else:
                print(f"Computing reference stats for {reference_pdb}")
                param_dictionary = self._reference_stats(reference_pdb)
                json_line = json.dumps(param_dictionary)
                f.write(prefix + "\t" + json_line + "\n")

        for i, layer in enumerate(self.node_layers):
            key = json.dumps(self.node_features[i])
            tensor = torch.tensor(param_dictionary[key], dtype=torch.float32)
            tensor = tensor.view(1, 1, -1)
            self.register_buffer(f"node_means_{i}", tensor)

        for i, layer in enumerate(self.edge_layers):
            key = json.dumps(self.edge_features[i])
            tensor = torch.tensor(param_dictionary[key], dtype=torch.float32)
            tensor = tensor.view(1, 1, -1)
            self.register_buffer(f"edge_means_{i}", tensor)
        return

    def _reference_stats(self, reference_pdb):
        X, C, _ = Flood.from_PDBID(reference_pdb).to_XCD()
        stats_dict = self._feature_stats(X, C)
        return stats_dict
    
    def _feature_stats(self, X, C, verbose=False, center=False):
        mask_i = field_map_to_mask(C)
        edge_idx, mask_ij = self.graph_builder(X, C)

        def _masked_stats(feature, mask, dims, verbose=False):
            mask = mask.unsqueeze(-1)
            feature = mask * feature
            sum_mask = mask.sum()
            mean = feature.sum(dims, keepdim=True) / sum_mask
            var = torch.sum(mask * (feature - mean) ** 2, dims) / sum_mask
            std = torch.sqrt(var)
            mean = mean.view(-1)
            std = std.view(-1)

            if verbose:
                frac = (100.0 * std**2 / (mean**2 + std**2)).type(torch.int32)
                print(f"Fraction of raw variance: {frac}")
            return mean, std

        # Collect statistics
        stats_dict = {}

        # Aggregate node layers
        for i, layer in enumerate(self.node_layers):
            node_h = layer(X, edge_idx, C)
            if center:
                node_h = node_h - self.__getattr__(f"node_means_{i}")
            mean, std = _masked_stats(node_h, mask_i, dims=[0, 1])

            # Store in dictionary
            key = json.dumps(self.node_features[i])
            stats_dict[key] = mean.tolist()

        # Aggregate node layers
        for i, layer in enumerate(self.edge_layers):
            edge_h = layer(X, edge_idx, C)
            if center:
                edge_h = edge_h - self.__getattr__(f"edge_means_{i}")
            mean, std = _masked_stats(edge_h, mask_ij, dims=[0, 1, 2])

            # Store in dictionary
            key = json.dumps(self.edge_features[i])
            stats_dict[key] = mean.tolist()

        # Round to small number of decimal places
        stats_dict = {k: [round(f, 3) for f in v] for k, v in stats_dict.items()}
        return stats_dict
    
class FloodGraph(nn.Module):
    """Build a graph topology given a flood backbone.

    Args:
        num_neighbors (int): Maximum number of neighbors in the graph.
        distance_grid_square_type (int): grid_square type for computing gridcell - gridcell
            distances for graph construction. Negative values will specify
            centroid across grid_square types. Default is `-1` (centroid).
        cutoff (float): Cutoff distance for graph construction. If not None,
            mask any edges further than this cutoff. Default is `None`.
        mask_interfaces (Boolean): Restrict connections only to within fields,
            excluding-between field interactions. Default is `False`.
        criterion (string, optional): Method used for building graph from distances.
            Currently supported methods are `{knn, random_log, random_linear}`.
            Default is `knn`.
        random_alpha (float, optional): Length scale parameter for random graph
            generation. Default is 3.
        random_temperature (float, optional): Temperature parameter for
            random graph sampling. Between 0 and 1 this value will interpolate
            between a normal k-NN graph and sampling from the graph generation
            process. Default is 1.0.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, HxW, 1)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, HxW, 1)`.
        custom_D (torch.Tensor, optional): Optional external distance map, for example
            based on other distance metrics, with shape
            `(num_batch, num_grid_cells, num_grid_cells)`.
        custom_mask_2D (torch.Tensor, optional): Optional mask to apply to distances
            before computing dissimilarities with shape
            `(num_batch, num_grid_cells, num_grid_cells)`.

    Outputs:
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_grid_cells, num_neighbors)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
    """

    def __init__(
        self,
        num_neighbors: int = 30,
        distance_grid_square_type: int = -1,
        cutoff: Optional[float] = None,
        mask_interfaces: bool = False,
        criterion: str = "knn",
        random_alpha: float = 3.0,
        random_temperature: float = 1.0,
        random_min_local: float = 20,
        deterministic: bool = False,
        deterministic_seed: int = 10,
    ):
        super(FloodGraph, self).__init__()
        self.num_neighbors = num_neighbors
        self.distance_grid_square_type = distance_grid_square_type
        self.cutoff = cutoff
        self.mask_interfaces = mask_interfaces
        self.distances = geometry.GridDistances()
        self.knn = kNN(k_neighbors=num_neighbors)

        self.criterion = criterion
        self.random_alpha = random_alpha
        self.random_temperature = random_temperature
        self.random_min_local = random_min_local
        self.deterministic = deterministic
        self.deterministic_seed = deterministic_seed

    def _mask_distances(self, X, C, custom_D=None, custom_mask_2D=None):
        mask_1D = field_map_to_mask(C)
        mask_2D = mask_1D.unsqueeze(2) * mask_1D.unsqueeze(1)
        if self.distance_grid_square_type > 0:
            X_grid_square = X[:, :, self.distance_grid_square_type, :]
        else:
            X_grid_square = X.mean(dim=2)
        if custom_D is None:
            D = self.distances(X_grid_square, dim=1)
        else:
            D = custom_D

        if custom_mask_2D is None:
            if self.mask_interfaces:
                mask_2D = torch.eq(C.unsqueeze(1), C.unsqueeze(2))
                mask_2D = mask_2D * mask_2D.type(torch.float32)
            if self.cutoff is not None:
                mask_cutoff = (D <= self.cutoff).type(torch.float32)
                mask_2D = mask_cutoff * mask_2D
        else:
            mask_2D = custom_mask_2D
        return D, mask_1D, mask_2D

    def _perturb_distances(self, D):
        # Replace distance by log-propensity
        if self.criterion == "random_log":
            logp_edge = -3 * torch.log(D)
        elif self.criterion == "random_linear":
            logp_edge = -D / self.random_alpha
        elif self.criterion == "random_uniform":
            logp_edge = D * 0
        else:
            return D

        if not self.deterministic:
            Z = torch.rand_like(D)
        else:
            with torch.random.fork_rng():
                torch.random.manual_seed(self.deterministic_seed)
                Z_shape = [1] + list(D.shape)[1:]
                Z = torch.rand(Z_shape, device=D.device)

        # Sample Gumbel noise
        G = -torch.log(-torch.log(Z))

        # Negate because are doing argmin instead of argmax
        D_key = -(logp_edge / self.random_temperature + G)

        return D_key
    
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        custom_D: Optional[torch.Tensor] = None,
        custom_mask_2D: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        D, mask_1D, mask_2D = self._mask_distances(X, C, custom_D, custom_mask_2D)

        if self.criterion != "knn":
            if self.random_min_local > 0:
                # Build first k-NN graph (local)
                self.knn.k_neighbors = self.random_min_local
                edge_idx_local, _, mask_ij_local = self.knn(D, mask_1D, mask_2D)

                # Build mask exluding these first ones
                mask_ij_remaining = 1.0 - mask_ij_local
                mask_2D_remaining = torch.ones_like(mask_2D).scatter(
                    2, edge_idx_local, mask_ij_remaining
                )
                mask_2D = mask_2D * mask_2D_remaining

                # Build second k-NN graph (random)
                self.knn.k_neighbors = self.num_neighbors - self.random_min_local
                D = self._perturb_distances(D)
                edge_idx_random, _, mask_ij_random = self.knn(D, mask_1D, mask_2D)
                edge_idx = torch.cat([edge_idx_local, edge_idx_random], 2)
                mask_ij = torch.cat([mask_ij_local, mask_ij_random], 2)

                # Handle small proteins
                k = min(self.num_neighbors, D.shape[-1])
                edge_idx = edge_idx[:, :, :k]
                mask_ij = mask_ij[:, :, :k]

                self.knn.k_neighbors = self.num_neighbors
                return edge_idx.contiguous(), mask_ij.contiguous()
            else:
                D = self._perturb_distances(D)

        edge_idx, edge_D, mask_ij = self.knn(D, mask_1D, mask_2D)
        return edge_idx, mask_ij
    
class kNN(nn.Module):
    """Build a k-nearest neighbors graph given a dissimilarity matrix.

    Args:
        k_neighbors (int): Number of nearest neighbors to include as edges of
            each node in the graph.

    Inputs:
        D (torch.Tensor): Dissimilarity matrix with shape
            `(num_batch, num_nodes, num_nodes)`.
        mask (torch.Tensor, optional): Node mask with shape `(num_batch, num_nodes)`.
        mask_2D (torch.Tensor, optional): Edge mask with shape
            `(num_batch, num_nodes, num_nodes)`.

    Outputs:
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, k)`. The slice `edge_idx[b,i,:]` contains
            the indices `{j in N(i)}` of the  k nearest neighbors of node `i`
            in object `b`.
        edge_D (torch.Tensor): Distances to each neighbor with shape
            `(num_batch, num_nodes, k)`.
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
    """

    def __init__(self, k_neighbors: int):
        super(kNN, self).__init__()
        self.k_neighbors = k_neighbors

    def forward(
        self,
        D: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_2D: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        mask_full = None
        if mask is not None:
            mask_full = mask.unsqueeze(2) * mask.unsqueeze(1)
        if mask_2D is not None:
            mask_full = mask_2D if mask_full is None else mask_full * mask_2D
        if mask_full is not None:
            max_float = np.finfo(np.float32).max
            D = mask_full * D + (1.0 - mask_full) * max_float

        k = min(self.k_neighbors, D.shape[-1])
        edge_D, edge_idx = torch.topk(D, int(k), dim=-1, largest=False)

        mask_ij = None
        if mask_full is not None:
            mask_ij = graph.collect_edges(mask_full.unsqueeze(-1), edge_idx)
            mask_ij = mask_ij.squeeze(-1)
        return edge_idx, edge_D, mask_ij

def manning_ideal_depth(inflow, slope, roughness_ref, eps=1e-6):
    """
    Computes ideal flood depth using Manning's formula 

    h ≈ (q * n / sqrt(S))^(3/5)

    Args:
        inflow (Tensor): (B, N, 1) inflow per node (e.g., m³/s).
        slope (Tensor): (B, N, 1) local terrain slope.
        roughness_ref (float): Reference roughness (e.g., 0.035).
        eps (float): Small value to avoid division by zero.

    Returns:
        Tensor: Ideal depth estimate (B, N, 1)
    """
    return ((inflow * roughness_ref) / (torch.sqrt(slope + eps) + eps)) ** (3.0 / 5.0)

class FloodNodeInternalCoords(nn.Module):
    """
    Node-level flood features based on the Diffusion Wave Approximation (DWA).
    
    Args:
        include_ideality (bool): Whether to add Manning-based ideality terms.
        distance_eps (float): Small constant for numerical stability.
        log_depth (bool): Apply log to depths for scale smoothing.
    """

    def __init__(self, include_ideality=False, distance_eps=0.01, log_depth=False):
        super(FloodNodeInternalCoords, self).__init__()
        self.include_ideality = include_ideality
        self.distance_eps = distance_eps
        self.log_depth = log_depth
        self.dim_out = 6 if include_ideality else 4

        # Reference roughness for ideal depth calculation
        roughness_ref = 0.035
        self.register_buffer("roughness_ref", torch.tensor(roughness_ref))

    def forward(self, elevation, inflow, slope, roughness, mask=None):
        """
        Args:
            elevation: (B, N, 1)
            inflow: (B, N, 1)
            slope: (B, N, 1)
            roughness: (B, N, 1)
            mask: Optional (B, N, 1) – binary mask for valid nodes
        
        Returns:
            node_h: (B, N, D) feature tensor
        """

        # Diffusion Wave Approximation: h ≈ q * n / sqrt(S)
        denom = torch.sqrt(slope + self.distance_eps) + self.distance_eps
        depth = inflow * roughness / denom

        if self.log_depth:
            depth = torch.log(depth + self.distance_eps)

        feature_list = [elevation, inflow, slope, depth]

        if self.include_ideality:
            ideal_depth = manning_ideal_depth(
                inflow=inflow,
                slope=slope,
                roughness_ref=self.roughness_ref,
                eps=self.distance_eps,
            )

            if self.log_depth:
                ideal_depth = torch.log(ideal_depth + self.distance_eps)

            depth_error = (depth - ideal_depth) ** 2
            feature_list.extend([ideal_depth, depth_error])

        node_h = torch.cat(feature_list, dim=-1)

        if mask is not None:
            node_h = node_h * mask

        return node_h

class NodeRadii(nn.Module):
    """
    Node features representing spatial variability.

    Args:
        length_scale (float): Normalizing factor for distances (e.g., in meters).
    
    Inputs:
        X (Tensor): Coordinates of each node (e.g., lat/lon/elevation), shape (B, N, D).
        mask (Tensor): Optional mask (B, N, 1) to ignore padded or invalid nodes.
    
    Outputs:
        node_h (Tensor): (B, N, D) spread or "radius" of flood-relevant position features.
    """

    def __init__(self, length_scale: float = 100.0):
        super(NodeRadii, self).__init__()
        self.length_scale = length_scale
        self.dim_out = 3  # if X has 3D coords (e.g., lat/lon/elev or x/y/depth)

    def forward(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # X: (B, N, D)
        B, N, D = X.shape

        if mask is None:
            mask = torch.ones((B, N, 1), device=X.device)

        # Compute batch-wise weighted centroid
        X_center = (mask * X).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + 1e-6)

        # Normalize and square distance from centroid
        radii = (mask * ((X - X_center) / self.length_scale) ** 2)

        return radii  # (B, N, D)

class Edge6grids(nn.Module):
    """Build concatenation of 3mer coordinates on graph edges.

    This layer assembles the pairwise concatenations of the coordinates
    `{X_a for a in {i-1,i,i+1,j-1,j,j+1}}` along every edge in a graph. This can
    be used for stitching of '6grid PairTERMs'.

    Args:
        require_contiguous (boolean, optional): Whether to enforce that
            `{i-1,i,i+1}` and`{j-1,j,j+1}` are each made up of contiguous
            residues from the same protein field. Default is `True`.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_grid_square_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        mask (Tensor, optional): Mask tensor with shape
            `(num_batch, num_residues)`.

    Outputs:
        X_ij (torch.Tensor): Pairwise-concatenated 3mers with shape
            `(num_batch, num_residues, num_neighbors, 2*num_grid_square_types, 3)`.
        mask_ij (Tensor, if mask): Propagated mask tensor for edges with shape
            `(num_batch, num_residues, num_neighbors)`.
    """

    def __init__(self, require_contiguous: bool = True):
        super(Edge6grids, self).__init__()
        self.require_contiguous = require_contiguous

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _pair_expand(h, collate_fun):
            # Build local neighborhoods [i-1, i, i+1]
            h_left = F.pad(h[:, :-1, :], (0, 0, 1, 0), "constant", 0)
            h_middle = h[:, :, :]
            h_right = F.pad(h[:, 1:, :], (0, 0, 0, 1), "constant", 0)
            h_i = collate_fun((h_left, h_middle, h_right))

            # Concatenate [j-1, j, j+1] of neighbors
            h_j = graph.collect_neighbors(h_i, edge_idx)
            h_i_tile = h_i.unsqueeze(-2).expand(h_j.size())
            h_ij = collate_fun((h_i_tile, h_j))
            return h_ij

        # Concatenation collation function for stitching
        _cat = lambda hs: torch.cat(hs, dim=-1)

        # Cumulative product collation function for mask propagation
        def _mul(hs):
            result = hs[0]
            for h_i in hs[1:]:
                result = result * h_i
            return result

        # Element-wise enforce values are greater than 0 and equal
        def _nonzero_and_equal(hs):
            entry_0 = hs[0]
            result = (hs[0] > 0.0).type(torch.float32)
            for h_i in hs[1:]:
                result = result * (entry_0 == h_i).type(torch.float32)
            return result

        # Build local neighborhoods [i-1, i, i+1]
        # X [batch, position, grid_square, xyz]
        X_flat = X.reshape(X.size(0), X.size(1), -1)
        X_ij = _pair_expand(X_flat, collate_fun=_cat)
        X_ij = X_ij.view(list(X_ij.size())[:-1] + [-1, 3])

        if C is not None:
            if self.require_contiguous:
                mask_ij = _pair_expand(C.unsqueeze(-1), collate_fun=_nonzero_and_equal)
            else:
                mask = field_map_to_mask(C)
                mask_ij = _pair_expand(mask.unsqueeze(-1), collate_fun=_mul)

            return X_ij, mask_ij
        else:
            return X_ij
         
class Edge2grids(nn.Module):
    """Build concatenation of 1mer coordinates on graph edges.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_grid_square_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Outputs:
        X_ij (torch.Tensor): Pairwise-concatenated 3mers with shape
            `(num_batch, num_residues, num_neighbors, 2*num_grid_square_types, 3)`.
        mask_ij (Tensor, if mask): Propagated mask tensor for edges with shape
            `(num_batch, num_residues, num_neighbors)`.
    """

    def __init__(self):
        super(Edge2grids, self).__init__()

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_batch = edge_idx.shape[0]
        num_residues = edge_idx.shape[1]
        num_neighbors = edge_idx.shape[2]
        num_grid_square_types = X.shape[2]
        shape_X = [num_batch, num_residues, num_neighbors, num_grid_square_types * 3]
        X_flat = X.reshape(num_batch, num_residues, -1)
        X_i = X_flat.unsqueeze(2).expand(shape_X)
        X_j = graph.collect_neighbors(X_flat, edge_idx).expand(shape_X)
        X_ij = torch.cat([X_i, X_j], -1)
        X_ij = X_ij.reshape(
            num_batch, num_residues, num_neighbors, 2 * num_grid_square_types, 3
        )
        if C is not None:
            mask_i = field_map_to_mask(C).unsqueeze(-1)
            mask_j = graph.collect_neighbors(mask_i, edge_idx)
            mask_ij = mask_i.unsqueeze(2) * mask_j
            return X_ij, mask_ij
        else:
            return X_ij

        
class EdgeDistance6grid(nn.Module):
    """Edge features based on field distance matrices along each i,j 6grid.

    Args:
        feature (str, optional): Option string in {'log', 'inverse', 'raw'}
            specifying how to process the raw distance features.
            Defaults to 'log'.
        distance_eps (float, optional): Smoothing parameter to prevent feature
            explosion at small distances. Can be thought of as a 'minimum length
            scale'. Defaults to 0.01.
        require_contiguous (boolean, optional): Whether to enforce that each
            3mer, `{i-1,i,i+1}` and`{j-1,j,j+1}`, is made up of contiguous
            residues from the same protein field. Default is `False` for
            backwards compatibility, but `True` is recommended as best practice.

    Attributes:
        dim_out (int): Number of dimensions of the output features.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_grid_square_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Outputs:
        edge_h (torch.Tensor): Edge distance matrix features with shape
            `(num_batch, num_residues, num_neighbors, (6 * num_grid_square_types)**2)`
    """

    def __init__(
        self,
        feature: str = "log",
        distance_eps: float = 0.01,
        num_grid_square_types: int = 4,
        require_contiguous: bool = False,
    ):
        super(EdgeDistance6grid, self).__init__()
        self.feature = feature
        self.distance_eps = distance_eps
        self.num_grid_square_types = num_grid_square_types
        self.layer_6grids = Edge6grids(require_contiguous=require_contiguous)
        self.layer_distance = geometry.Distances()

        # Public attribute
        self.dim_out = (6 * num_grid_square_types) ** 2

        self.feature = feature
        feature_functions = {
            "log": self.log_func,
            "inverse": self.inverse_func,
            "raw": self.raw_func,
        }
        self.feature_function = feature_functions[feature]

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        X_ij, mask_ij = self.layer_6grids(X, edge_idx, C=C)
        D_ij = self.layer_distance(X_ij, dim=-2)
        feature_ij = self.feature_function(D_ij)
        feature_ij_flat = feature_ij.reshape(list(D_ij.shape[:3]) + [-1])
        edge_h = mask_ij * feature_ij_flat
        # debug_plot_edge6griddist(edge_h, feature=self.feature)
        return edge_h

    def log_func(self, D):
        return torch.log(D + self.distance_eps)

    def inverse_func(self, D):
        return 1.0 / (D + self.distance_eps)

    def raw_func(self, D):
        return D
    
class EdgeDistance2grid(nn.Module):
    """Edge features based on field distance matrices along each i,j 2grid.

    Args:
        feature (str, optional): Option string in {'log', 'inverse', 'raw'}
            specifying how to process the raw distance features.
            Defaults to 'log'.
        distance_eps (float, optional): Smoothing parameter to prevent feature
            explosion at small distances. Can be thought of as a 'minimum length
            scale'. Defaults to 0.01.

    Attributes:
        dim_out (int): Number of dimensions of the output features.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_atom_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Outputs:
        edge_h (torch.Tensor): Edge distance matrix features with shape
            `(num_batch, num_residues, num_neighbors, (6 * num_atom_types)**2)`
    """

    def __init__(
        self,
        features: str = "rbf+log",
        distance_eps: float = 0.01,
        num_atom_types: int = 4,
        rbf_min: float = 0.0,
        rbf_max: float = 20.0,
        rbf_count: int = 20,
    ):
        super(EdgeDistance2grid, self).__init__()
        self.distance_eps = distance_eps
        self.num_atom_types = num_atom_types
        self.layer_2grids = Edge2grids()
        self.layer_distance = geometry.GridDistances()

        features = features.split("+")
        if not isinstance(features, list):
            features = [features]
        self.features = features
        if "rbf" in self.features:
            self.rbf_function = RBFExpansion(rbf_min, rbf_max, rbf_count)
        dim_base = (2 * num_atom_types) ** 2
        feature_dims = {
            "log": dim_base,
            "inverse": dim_base,
            "raw": dim_base,
            "rbf": dim_base * rbf_count,
        }

        # Public attribute
        self.dim_out = sum([feature_dims[d] for d in features])

        self.feature_funcs = {
            "log": lambda D: torch.log(D + self.distance_eps),
            "inverse": lambda D: 1.0 / (D + self.distance_eps),
            "raw": lambda D: D,
            "rbf": lambda D: self.rbf_function(D),
        }

    def featurize(self, D):
        h_list = []
        for feature in self.features:
            h = self.feature_funcs[feature](D)
            h_list.append(h)
        h = torch.cat(h_list, -1)
        return h

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        X_ij, mask_ij = self.layer_2grids(X, edge_idx, C=C)
        D_ij = self.layer_distance(X_ij, dim=-2)
        shape_flat = list(D_ij.shape[:3]) + [-1]
        D_ij = D_ij.reshape(shape_flat)
        feature_ij = self.featurize(D_ij)

        # DEBGUG
        # _debug_plot_edges(edge_idx, feature_ij, unravel=True)
        # exit(0)
        edge_h = mask_ij * feature_ij
        return edge_h


class EdgeOrientationField(nn.Module):
    """Edge features encoding the relative orientations of fields and field grid_squares.

    Args:
        feature (str, optional): Option string in {'log', 'inverse', 'raw'}
            specifying how to process the raw distance features.
            Defaults to 'log'.
        distance_eps (float, optional): Smoothing parameter to prevent feature
            explosion at small distances. Can be thought of as a 'minimum length
            scale'. Defaults to 0.1.
        distance_eps (float, optional): Like `distance_eps`, but for orientation
            calculations. Can be thought of as a 'minimum length scale'
            Defaults to 1E-5.

    Attributes:
        dim_out (int): Number of dimensions of the output features.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_grid_square_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Outputs:
        edge_h (torch.Tensor): Edge distance matrix features with shape
            `(num_batch, num_residues, num_neighbors, 24)`
    """

    def __init__(
        self, feature: str = "log", distance_eps: float = 0.1, norm_eps: float = 1e-1
    ):
        super(EdgeOrientationField, self).__init__()
        self.distance_eps = distance_eps
        self.norm_eps = norm_eps

        self.feature = feature
        feature_functions = {
            "log": lambda D: torch.log(D + self.distance_eps),
            "inverse": lambda D: 1.0 / (D + self.distance_eps),
            "raw": lambda D: D,
        }
        self.feature_function = feature_functions[feature]

        # Public attribute
        self.dim_out = 24

    def _normed_vec(self, V):
        # Unit vector from i to j
        mag_sq = (V**2).sum(dim=-1, keepdim=True)
        mag = torch.sqrt(mag_sq + self.norm_eps)
        V_norm = V / mag
        return V_norm

    def _reference_frames(self, X):
        # Build reference frames at each i
        X_N, X_CA, X_C, X_O = X.unbind(2)
        _normed_cross = lambda U_a, U_b: self._normed_vec(torch.cross(U_a, U_b, dim=-1))
        u_CA_N = self._normed_vec(X_N - X_CA)
        u_CA_C = self._normed_vec(X_C - X_CA)
        n_1 = u_CA_N
        n_2 = _normed_cross(n_1, u_CA_C)
        n_3 = _normed_cross(n_1, n_2)
        R = torch.stack([n_1, n_2, n_3], -1)
        return R

    def _reference_frames_field(self, X, C):
        # Build reference frames at each i
        X_N, X_CA, X_C, X_O = X.unbind(2)
        _normed_cross = lambda U_a, U_b: self._normed_vec(torch.cross(U_a, U_b, dim=-1))
        u_CA_N = self._normed_vec(X_N - X_CA)
        u_CA_C = self._normed_vec(X_C - X_CA)

        u_CA_N_avg = self._field_average(u_CA_N, C)
        u_CA_C_avg = self._field_average(u_CA_C, C)

        n_1 = self._normed_vec(u_CA_N_avg)
        n_2 = _normed_cross(n_1, self._normed_vec(u_CA_C_avg))
        n_3 = _normed_cross(n_1, n_2)
        R = torch.stack([n_1, n_2, n_3], -1)
        return R

    def _field_average(self, node_h, C, eps=1e-5):
        # Compute the per-field averages of each feature within a field, in place
        num_batch, num_residues = list(C.shape)
        num_fields = int(torch.max(C).item())

        # Build a position == field expanded mask (B,L,C)
        C_expand = C.unsqueeze(-1).expand(-1, -1, num_fields)
        idx = torch.arange(num_fields, device=C.device) + 1
        idx_expand = idx.view(1, 1, -1)
        mask_expand = (idx_expand == C_expand).type(torch.float32)
        mask_expand = mask_expand.unsqueeze(-1)

        # Masked reduction
        node_h_expand = node_h.unsqueeze(2).expand(-1, -1, num_fields, -1)
        node_h_field_average = (mask_expand * node_h_expand).sum(1, keepdim=True) / (
            (mask_expand).sum(1, keepdim=True) + eps
        )

        # Back-expand (B,C,K) => (B,L,3)
        node_h_field_average = (mask_expand * node_h_field_average).sum(2)
        return node_h_field_average

    def _R_neighbors(self, R_i, edge_idx):
        num_batch, num_residues, num_k = list(edge_idx.shape)
        R_flat_i = R_i.reshape(num_batch, num_residues, 9)
        R_flat_j = graph.collect_neighbors(R_flat_i, edge_idx)
        R_j = R_flat_j.reshape(num_batch, num_residues, num_k, 3, 3)
        return R_j

    def _transformation_features(self, X_i, X_j, R_i, R_j, edge_idx, edges=True):
        # Distance and direction
        dX = X_j - X_i.unsqueeze(2).contiguous()
        L = torch.sqrt((dX**2).sum(-1, keepdim=True) + self.distance_eps)
        u_ij = torch.einsum("niab,nija->nijb", R_i, dX / L)

        # Relative orientation
        R_relative_ij = torch.einsum("niab,nijac->nijbc", R_i, R_j)
        q_ij = geometry.quaternions_from_rotations(R_relative_ij)

        h = torch.cat((self.feature_function(L), u_ij, q_ij), dim=-1)
        return h

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        num_batch, num_residues, num_k = list(edge_idx.shape)

        # Compute local positions (C-alpha) and frames (B, L, 4)
        R_i = self._reference_frames(X)
        R_field_i = self._reference_frames_field(X, C)

        # X field
        X_i = X[:, :, 1, :]
        X_j = graph.collect_neighbors(X_i, edge_idx)
        X_field_i = self._field_average(X_i, C)
        X_field_j = graph.collect_neighbors(X_field_i, edge_idx)

        # Relative field features
        R_field_j = self._R_neighbors(R_field_i, edge_idx)
        R_j = self._R_neighbors(R_i, edge_idx)

        h_field_to_field = self._transformation_features(
            X_field_i, X_field_j, R_field_i, R_field_j, edge_idx
        )
        h_field_to_node = self._transformation_features(
            X_field_i, X_j, R_field_i, R_j, edge_idx
        )
        h_node_to_node = self._transformation_features(X_i, X_j, R_i, R_j, edge_idx)
        edge_h = torch.cat((h_field_to_field, h_field_to_node, h_node_to_node), -1)

        # DEBUG:
        # h = h_node_to_node
        # _debug_plot_edges(edge_idx, h[:,:,:,0].unsqueeze(-1))
        # _debug_plot_edges(edge_idx, h[:,:,:,1:4])
        # _debug_plot_edges(edge_idx, h[:,:,:,5:9])

        mask_i = field_map_to_mask(C).unsqueeze(-1)
        mask_j = graph.collect_neighbors(mask_i, edge_idx)
        mask_ij = mask_i.unsqueeze(2) * mask_j
        edge_h = mask_ij * edge_h
        return edge_h


class EdgeDistanceField(nn.Module):
    """Edge features based on distance matrices along each i,j 6grid.

    These feature capture (signed) intra-field distances as well as distinguish
    between same vs. different field.

    Args:

    Attributes:
        dim_out (int): Number of dimensions of the output features.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_grid_square_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Outputs:
        edge_h (torch.Tensor): Edge field distance features with shape
            `(num_batch, num_residues, num_neighbors, 2)`
    """

    def __init__(self):
        super(EdgeDistanceField, self).__init__()

        # Public attribute
        self.dim_out = 3

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Is the edge intra-field or inter-field?
        field_i = C.unsqueeze(-1)
        field_j = graph.collect_neighbors(field_i, edge_idx).squeeze(-1)
        is_interface = torch.ne(field_i, field_j).type(torch.float32)

        # If it is intra-field, what is the field distance?
        residue_i = torch.arange(edge_idx.shape[1], device=X.device).view((1, -1, 1))
        residue_j = edge_idx
        D_signed = (residue_j - residue_i).type(torch.float32)
        D_residue = torch.abs(D_signed)
        D_intra = (1.0 - is_interface) * torch.log(D_residue + 1.0)
        D_intra_sign = (1.0 - is_interface) * torch.sign(D_signed)

        edge_h = torch.stack([is_interface, D_intra, D_intra_sign], dim=-1)
        return edge_h


class EdgePositionalEncodings(nn.Module):
    """Edge features based on positional encodings of field distance |i-j|.

    Args:
        dim_embeddings (int): Embedding dimension.

    Attributes:
        dim_out (int): Number of dimensions of the output features.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_grid_square_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Outputs:
        edge_h (torch.Tensor): Edge field distance features with shape
            `(num_batch, num_residues, num_neighbors, 2)`
    """

    def __init__(self, dim_embedding: int = 128, period_range: tuple = (1.0, 1000.0)):
        super(EdgePositionalEncodings, self).__init__()

        # Public attribute
        self.dim_out = dim_embedding
        self.encoding = PositionalEncoding(
            d_model=dim_embedding, period_range=period_range
        )

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Is the edge intra-field or inter-field?
        field_i = C.unsqueeze(-1)
        field_j = graph.collect_neighbors(field_i, edge_idx).squeeze(-1)
        mask_intrafield = torch.eq(field_i, field_j).float()

        # If it is intra-field, what is the field distance?
        residue_i = torch.arange(edge_idx.shape[1], device=X.device).view((1, -1, 1))
        residue_j = edge_idx
        D_signed = (residue_j - residue_i).float()
        edge_h = mask_intrafield[..., None] * self.encoding(D_signed[..., None])
        return edge_h

class EdgeRandomFourierFeatures2grid(nn.Module):
    # okay but you have to laugh bc in the protein domain this is a 2grid and not a 2grid
    # and so I made the joke of well hopefully this is benign 
    # (I am my biggest fan)(likely my only)
    """For edge-ij computes a random fourier projection of the SE3-invariant feature t_ji
    pointing from i to j in the local frame of residue i, optionally including the projection
    of the associated quaternion representation of R_ji the rotation from taking you from frame i to frame j
    Features are decayed exponentially at rate alpha.
    Args:
        dim_embedding (int): dimension of embedding
        trainable (bool): Whether to train the weight matrix of the fourier features
        scale (float): The scale (standard deviation) to sample random weights from
        use_quaternion (bool): Whether to embed the quaternion representation as well

    Inputs:
        X (torch.tensor): of size (batch, length, (4 or 14), 3)
        edge_idx (torch.LongTensor): of size (batch, length, num_neighbors)
        C (torch.tensor): of size (batch, length)

    Outputs:
        edge_h (torch.tensor): of size (batch, length, num_neighbors, dim_embedding)
    """

    def __init__(
        self,
        dim_embedding: int = 128,
        trainable: bool = False,
        scale: float = 1.0,
        use_quaternion: bool = False,
        seed: int = 10,
    ):
        super().__init__()

        self._seed = seed
        with torch.random.fork_rng():
            torch.random.manual_seed(self._seed)

            self.vector_f = FourierFeaturization(
                3, dim_embedding, trainable=trainable, scale=scale
            )
            self.distance_f = FourierFeaturization(
                64, dim_embedding, trainable=trainable, scale=scale
            )

            self.use_quaternion = use_quaternion
            if self.use_quaternion:
                self.quat_f = FourierFeaturization(
                    4, dim_embedding, trainable=trainable, scale=scale
                )

        self.layer_2grids = Edge2grids()
        self.layer_distance = geometry.Distances()
        self.frame_builder = backbone.FrameBuilder()
        self.dim_out = dim_embedding

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        X_ij, mask_ij = self.layer_2grids(X, edge_idx, C=C)
        D_ij = self.layer_distance(X_ij, dim=-2)
        D_ij = D_ij.reshape(*D_ij.size()[:-2], -1)

        R_i, t_i, _ = self.frame_builder.inverse(X, C)
        R_j, t_j = transforms.collect_neighbor_transforms(R_i, t_i, edge_idx)
        R_ji, t_ji = transforms.compose_inner_transforms(
            R_j, t_j, R_i.unsqueeze(-3), t_i.unsqueeze(-2)
        )

        edge_h = self.vector_f(t_ji) + self.distance_f(D_ij)

        if self.use_quaternion:
            Q_ji = geometry.quaternions_from_rotations(R_ji)
            edge_h = edge_h + self.quat_f(Q_ji)

        return edge_h
    
class RBFExpansion(nn.Module):
    def __init__(
        self,
        value_min: float,
        value_max: float,
        num_rbf: int,
        std: Optional[float] = None,
    ):
        super(RBFExpansion, self).__init__()
        rbf_centers = torch.linspace(value_min, value_max, num_rbf)
        self.register_buffer("rbf_centers", rbf_centers)
        if std is None:
            std = (rbf_centers[1] - rbf_centers[0]).item()
        self.std = std

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shape = list(h.shape)
        shape_ones = [1 for _ in range(len(shape))] + [-1]
        rbf_centers = self.rbf_centers.view(shape_ones)
        h = torch.exp(-(((h.unsqueeze(-1) - rbf_centers) / self.std) ** 2))
        h = h.view(shape[:-1] + [-1])
        return h


class NodeCartesianCoords(nn.Module):
    """Node features containing raw relative coordinates.

    Warning: these features are not rotationally invariant.

    Args:
        scale_factor (float, optional): Scale factor to rescale raw coordinates
            for neural network processing. Default is 0.3.
        num_grid_square_types (int, optional): Number of grid_square types. Default is 4.

    Attributes:
        dim_out (int): Number of dimensions of the output features.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_grid_square_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Outputs:
        edge_h (torch.Tensor): Node relative coordinates features with shape
            `(num_batch, num_residues, 3 * (num_grid_square_types)**2)`
    """

    def __init__(self, scale_factor: float = 0.3, num_grid_square_types: int = 4):
        super(NodeCartesianCoords, self).__init__()
        self.scale_factor = scale_factor
        self.num_grid_square_types = num_grid_square_types

        # Public attribute
        self.dim_out = 3 * (num_grid_square_types**2)

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        num_batch, num_residues, num_neighbors = list(edge_idx.shape)

        dX = X.unsqueeze(-2) - X.unsqueeze(-3)
        node_h = self.scale_factor * dX.reshape([num_batch, num_residues, -1])

        if C is not None:
            mask_i = field_map_to_mask(C)
            node_h = mask_i.unsqueeze(-1) * node_h
        return node_h


class EdgeCartesianCoords(nn.Module):
    """Edge features containing raw relative coordinates.

    Warning: these features are not rotationally invariant.

    Args:
        scale_factor (float, optional): Scale factor to rescale raw coordinates
            for neural network processing. Default is 0.1.
        num_grid_square_types (int, optional): Number of grid_square types. Default is 4.

    Attributes:
        dim_out (int): Number of dimensions of the output features.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
            `(num_batch, num_residues, num_grid_square_types, 3)`.
        edge_idx (torch.LongTensor): Graph indices for expansion with shape
            `(num_batch, num_residues, num_neighbors)`.
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Outputs:
        edge_h (torch.Tensor): Edge relative coordinates features with shape
            `(num_batch, num_residues, num_neighbors, 3 * (num_grid_square_types)**2)`
    """

    def __init__(self, scale_factor: float = 0.1, num_grid_square_types: int = 4):
        super(EdgeCartesianCoords, self).__init__()
        self.scale_factor = scale_factor
        self.num_grid_square_types = num_grid_square_types

        # Public attribute
        self.dim_out = 3 * (num_grid_square_types**2)

    def forward(
        self,
        X: torch.Tensor,
        edge_idx: torch.LongTensor,
        C: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        num_batch, num_residues, num_neighbors = list(edge_idx.shape)

        # Collect coordiates and j
        X_flat = X.reshape([num_batch, num_residues, -1])
        X_j_flat = graph.collect_neighbors(X_flat, edge_idx)
        X_j = X_j_flat.reshape(
            [num_batch, num_residues, num_neighbors, 1, self.num_grid_square_types, 3]
        )

        X_i = X.reshape([num_batch, num_residues, 1, self.num_grid_square_types, 1, 3])
        dX = X_j - X_i
        edge_h = self.scale_factor * dX.reshape(
            [num_batch, num_residues, num_neighbors, -1]
        )
        if C is not None:
            mask_i = field_map_to_mask(C)
            mask_i_expand = mask_i.unsqueeze(-1)
            mask_j = graph.collect_neighbors(mask_i_expand, edge_idx)
            mask_ij = mask_j * mask_i_expand.unsqueeze(-1)
            edge_h = mask_ij * edge_h
        return edge_h


def field_map_to_mask(C: torch.LongTensor) -> torch.Tensor:
    """Convert field map into a mask.

    Args:
        C (torch.LongTensor): field map with shape
            `(num_batch, num_residues)`.

    Returns:
        mask (Tensor, optional): Mask tensor with shape
            `(num_batch, num_residues)`.
    """
    return (C > 0).type(torch.float32)


def _cgo_cylinder(X1, X2, radius=0.5, rgb=(0.0, 0.0, 1.0)):
    x1, y1, z1 = X1.data.numpy().flatten().tolist()
    x2, y2, z2 = X2.data.numpy().flatten().tolist()
    r1, g1, b1 = rgb
    r2, g2, b2 = rgb
    cgo_str = (
        f"[ 9.0, {x1}, {y1}, {z1}, {x2}, {y2}, {z2}, {radius}, {r1}, {g1}, {b1}, {r2},"
        f" {g2}, {b2} ]"
    )
    return cgo_str


def _cgo_sphere(X1, radius=1.0):
    x1, y1, z1 = X1.data.numpy().flatten().tolist()
    cgo_str = f"[ 7.0, {x1}, {y1}, {z1}, {radius}]"
    return cgo_str


def _cgo_color(rgb=(0.0, 0.0, 1.0)):
    r, g, b = rgb
    cgo_str = f"[ 6.0, {r}, {g}, {b}]"
    return cgo_str
