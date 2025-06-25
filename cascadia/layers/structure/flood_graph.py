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
            "internal_coords": NodeInternalCoords,
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