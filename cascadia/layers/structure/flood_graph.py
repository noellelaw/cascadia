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
            "orientations_2grid": EdgeOrientation2grid,
            "position_2grid": EdgePositionalEncodings,
            "distances_field": EdgeDistancefield,
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
        X, C, _ = Flood.from_PDBID(reference_pdb).to_XCS()
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
        mask_interfaces (Boolean): Restrict connections only to within chains,
            excluding-between chain interactions. Default is `False`.
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
        C (torch.LongTensor): Chain map with shape
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
