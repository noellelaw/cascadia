import torch
import torch.nn as nn

from types import SimpleNamespace
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cascadia import constants
from cascadia.data.xcs import validate_XC
from cascadia.layers import complexity, graph
from cascadia.layers.structure import diffusion, potts, flood_graph, floodfield
from cascadia.layers.structure.flood_graph_alldata import (
    EdgeSidechainsDirect,
    NodefieldRBF,
)
from cascadia.utility.model import load_model as utility_load_model

# Placeholder: need to replace with full FloodFeatureGraph
class FloodGraphDesign(nn.Module):
    """
    Constructs a graph from coastal configuration and elevation metadata.
    Each node represents a spatial patch or pixel, and edges encode local connectivity.
    It encodes backbones with a `BackboneEncoderGNN`
    and then autoregressively factorizes the joint distribution of
    flood field conformations given these graph embeddings.
    Optional first order marginal and Potts sequence decoders are also available

    Args:
        dim_nodes (int): Hidden dimension of node tensors of underlying GNNs.
        dim_edges (int): Hidden dimension of edge tensors of underlying GNNs.
        num_neighbors (int): Number of neighbors per node for underlying GNNs.
        node_features (tuple): List of node feature specifications for
            structure encoder. Features can be given as strings or as
            dictionaries.
        edge_features (tuple): List of edge feature specifications for
            structure encoder. Features can be given as strings or as
            dictionaries.
        sequence_embedding (str): How to represent sequence when decoding.
            Currently the only option is `linear`.
        flood_field_embedding (str): How to represent flood field when decoding. 
            Options include: 
            - "field_linear" → simple MLP on flood velocity/pressure
            - "field_rbf" → encode continuous fields via radial basis bins (e.g., flow bins)
            - "direct_fourier" → apply RFF to spatial or meteorological inputs
            - "mixed_field" → combine RBF and direct encodings
        floodfields (bool): Whether to decode flood depths while jointly 
            predicting local flow fields. Default is false bc Im a lil too dumb to implement it atm.
            Will be a better software engineer another day maybe.
        num_layers (int): Number of layers of underlying GNNs. Can be overridden
            for the structure encoder by `num_layers_encoder`.
        num_layers_encoder (int, optional): Number of layers for structure
            encoder GNN.
        dropout (float): Dropout fraction used for all encoders and decoders
            except for the marginal sequence likelihood decoder in
            `decoder_S_marginals`.
        node_mlp_layers (int): Number of hidden layers for node update function
            of underlying GNNs.
        node_mlp_dim (int, optional): Dimension of hidden layers for node update
            function of underlying GNNs, defaults to match output dimension.
        edge_update (bool): Whether to include an edge update step in the GNNs.
        edge_mlp_layers (int): Number of hidden layers for edge update function
            of underlying GNNs.
        edge_mlp_dim (int, optional): Dimension of hidden layers for edge update
            function of underlying GNNs, defaults to match output dimension.
        skip_connect_input (bool): Whether to include skip connections between
            layers of underlying GNNs.
        mlp_activation (str): MLP nonlinearity function, `relu` or `softplus`
            accepted.
        num_landcover_classes (int): Number of discrete land cover or infrastructure classes 
            used in the flood decoder 
        num_field_bins (int): Number of bins for flood field encoding for "field_rbf" 
            or "mixed_field". (self.rbf = RBFEmbed(input_dim=1, num_centers=num_field_bins))
        decoder_num_hidden (int): Dimension of hidden decoder layers.
        label_smoothing (float): Level of smoothing to apply to flood field labels.
        separate_packing (bool): If True, then autoregressively factorize
            flood flow fields in two stages where the full sequence is predicted
            before all of the infrastructures????. Otherwise an interleaved factorization
            will be used that autoregressively predicts both the flood depth
            and flow or failure in an alternating manner. Default is True.
            Will come back to this for IIC because oof, I don't know where to start.  
        graph_criterion (str): Graph criterion for structure encoder, defines
            how neighbors are chosen. See
            `cascadia.models.graph_design.BackboneEncoderGNN` for
            allowed values.
        graph_random_min_local (int): Minimum number of neighbors in GNN that
            come from local neighborhood, before random neighbors are chosen.
        graph_attentional (bool): Currently unused, previously used for
            experimental GNN attention mechanism.
        graph_num_attention_heads (int): Currently unused, previously used for
            experimental GNN attention mechanism.
        predict_D_marginals (bool): Whether to train marginal description decoder.
        predict_D_potts (bool): Whether to train Potts description decoder.
        potts_parameterization (str): How to parametrize Potts description decoder,
            see `chroma.layer.structure.potts` for allowed values.
        potts_num_factors (int, optional): Number of factors to use for Potts
            sequence decoder.
        potts_symmetric_J (bool): Whether to force J tensor of Potts model to be
            symmetric.
        noise_schedule (str, optional): Noise schedule for mapping between
            diffusion time and noise level, see
            chroma.layers.structure.diffusion.DiffusionChainCov for allowed
            values. If not set, model should only be provided with denoised
            backbones.
        noise_covariance_model (str): Covariance mode for mapping between
            diffusion time and noise level, see
            chroma.layers.structure.diffusion.DiffusionChainCov for allowed
            values.
        noise_complex_scaling (bool): Whether to scale noise for complexes.
        noise_beta_range (Tuple[float, float]): Minimum and maximum noise levels
            for noise schedule.
        noise_log_snr_range (Tuple[float, float]): Range of log signal-to-noise
            ratio for noising.


    Inputs:
        X (torch.Tensor): Backbone coordinates for flood field (depth / extent) with shape
            `(num_batch, H, W) or (B, H, W, 1)`.
        C (torch.LongTensor): Conditioning map (SLR / meteorological) with shape `(num_batch, H, W, 1)`.
        D (torch.LongTensor): Description tensor (Land cover or infrastructure class labels at each grid cell) with shape
            `(num_batch, H, W)`.
        t (torch.Tensor, optional): Diffusion timesteps corresponding to noisy
            input backbones, of shape `(num_batch)`. Use zeros when passing
            structures without noise.
        sample_noise (bool, optional): Whether to apply noise to input
            backbones.
        permute_idx (torch.LongTensor, optional): Permutation tensor for fixing
            the autoregressive decoding order `(B, H*W)`. If
            `None` (default), a random decoding order will be generated.
        priority (torch.Tensor, optional): Priority values for constraining
            residue orderings with shape `(num_batch, H, W)`.
            If residues are assigned to integer-valued groups, the sampled
            permutation will be ordered such that all residues within a lower-
            valued priority group will occur before residues with higher-valued
            priority assignments.

    Output:
        List[torch_geometric.data.Data]: Batch of graphs, one per input image.
    """

    def __init__(
        self,
        dim_nodes: int = 128,
        dim_edges: int = 128,
        num_neighbors: int = 30,
        node_features: tuple = (("internal_coords", {"log_lengths": True}),),
        edge_features: tuple = (
            "distances_2mer",
            "orientations_2mer",
            "distances_chain",
        ),
        sequence_embedding: str = "linear",
        floodfield_embedding: str = "field_rbf",
        floodfields: bool = False,
        num_layers: int = 3,
        num_layers_encoder: Optional[int] = None,
        dropout: float = 0.1,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        skip_connect_input: bool = False,
        mlp_activation: str = "softplus",
        num_landcover_classes: int = 20,
        num_field_bins: int = 20,
        decoder_num_hidden: int = 512,
        label_smoothing: float = 0.1,
        separate_packing: bool = True,
        graph_criterion: str = "knn",
        graph_random_min_local: int = 20,
        graph_attentional: bool = False,
        graph_num_attention_heads: int = 4,
        predict_D_marginals: bool = False,
        predict_D_potts: bool = False,
        potts_parameterization: str = "factor",
        potts_num_factors: Optional[int] = None,
        potts_symmetric_J: bool = True,
        noise_schedule: Optional[str] = None,
        noise_covariance_model: str = "brownian",
        noise_complex_scaling: bool = False,
        noise_beta_range: Tuple[float, float] = (0.2, 70.0),
        noise_log_snr_range: Tuple[float, float] = (-7.0, 13.5),
        checkpoint_gradients: bool = False,
        **kwargs
    ) -> None:
        """Initialize GraphDesign network."""
        super(FloodGraphDesign, self).__init__()

        # Save configuration in kwargs
        self.kwargs = locals()
        self.kwargs.pop("self")
        for key in list(self.kwargs.keys()):
            if key.startswith("__") and key.endswith("__"):
                self.kwargs.pop(key)
        args = SimpleNamespace(**self.kwargs)

        # Important global options
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.num_landcover_classes = num_landcover_classes
        self.num_field_bins = num_field_bins
        self.separate_packing = separate_packing
        self.floodfields = floodfields
        self.predict_S_potts = predict_S_potts
        self.traversal = FloodTraversalSpatial()

        # Encoder GNN process backbone
        self.encoder = BackboneEncoderGNN(
            dim_nodes=args.dim_nodes,
            dim_edges=args.dim_edges,
            num_neighbors=args.num_neighbors,
            node_features=args.node_features,
            edge_features=args.edge_features,
            num_layers=(
                args.num_layers
                if args.num_layers_encoder is None
                else args.num_layers_encoder
            ),
            node_mlp_layers=args.node_mlp_layers,
            node_mlp_dim=args.node_mlp_dim,
            edge_update=args.edge_update,
            edge_mlp_layers=args.edge_mlp_layers,
            edge_mlp_dim=args.edge_mlp_dim,
            mlp_activation=args.mlp_activation,
            dropout=args.dropout,
            skip_connect_input=args.skip_connect_input,
            graph_criterion=args.graph_criterion,
            graph_random_min_local=args.graph_random_min_local,
            checkpoint_gradients=checkpoint_gradients,
        )

        # Time features for diffusion
        if args.noise_schedule is not None:
            self.noise_perturb = diffusion.DiffusionChainCov(
                noise_schedule=args.noise_schedule,
                beta_min=args.noise_beta_range[0],
                beta_max=args.noise_beta_range[1],
                log_snr_range=args.noise_log_snr_range,
                covariance_model=args.noise_covariance_model,
                complex_scaling=args.noise_complex_scaling,
            )
            self.time_features = diffusion.NoiseTimeEmbedding(
                dim_embedding=args.dim_nodes,
                noise_schedule=self.noise_perturb.noise_schedule,
            )

        # Decoder GNN process backbone
        if self.floodfields:
            self.decoder = FloodfieldDecoderGNN(
                dim_nodes=args.dim_nodes,
                dim_edges=args.dim_edges,
                num_neighbors=args.num_neighbors,
                predict_D=False, # for now, we only predict flood fields
                predict_field=(not args.separate_packing),
                sequence_embedding=args.sequence_embedding,
                sidechain_embedding=args.sidechain_embedding,
                num_layers=args.num_layers,
                node_mlp_layers=args.node_mlp_layers,
                node_mlp_dim=args.node_mlp_dim,
                edge_update=args.edge_update,
                edge_mlp_layers=args.edge_mlp_layers,
                edge_mlp_dim=args.edge_mlp_dim,
                mlp_activation=args.mlp_activation,
                dropout=args.dropout,
                skip_connect_input=args.skip_connect_input,
                num_landcover_classes=args.num_landcover_classes,
                num_field_bins=args.num_field_bins,
                decoder_num_hidden=args.decoder_num_hidden,
                label_smoothing=args.label_smoothing,
                checkpoint_gradients=checkpoint_gradients,
            )

        if args.predict_D_marginals:
            self.decoder_D_marginals = NodePredictorD(
                num_landcover_classes=args.num_landcover_classes,
                dim_nodes=args.dim_nodes,
                dim_hidden=args.decoder_num_hidden,
                label_smoothing=args.label_smoothing,
            )

        if args.predict_D_potts:
            self.decoder_D_potts = potts.GraphPotts(
                dim_nodes=args.dim_nodes,
                dim_edges=args.dim_edges,
                num_states=args.num_landcover_classes,
                parameterization=args.potts_parameterization,
                num_factors=args.potts_num_factors,
                symmetric_J=args.potts_symmetric_J,
                dropout=args.dropout,
                label_smoothing=args.label_smoothing,
            )

        if args.separate_packing:
            # Optionally do a two-stage autoregressive prediction
            self.embed_S = nn.Embedding(args.num_landcover_classes, args.dim_nodes)
            self.encoder_S_gnn = graph.GraphNN(
                dim_nodes=args.dim_nodes,
                dim_edges=args.dim_edges,
                num_layers=args.num_layers,
                node_mlp_layers=args.node_mlp_layers,
                node_mlp_dim=args.node_mlp_dim,
                edge_update=args.edge_update,
                edge_mlp_layers=args.edge_mlp_layers,
                edge_mlp_dim=args.edge_mlp_dim,
                mlp_activation=args.mlp_activation,
                dropout=args.dropout,
                norm="transformer",
                scale=args.num_neighbors,
                skip_connect_input=args.skip_connect_input,
                checkpoint_gradients=checkpoint_gradients,
            )
            self.decoder_field = FloodfieldDecoderGNN(
                dim_nodes=args.dim_nodes,
                dim_edges=args.dim_edges,
                num_neighbors=args.num_neighbors,
                predict_S=False,
                predict_field=True,
                sequence_embedding=args.sequence_embedding,
                sidechain_embedding=args.sidechain_embedding,
                num_layers=args.num_layers,
                node_mlp_layers=args.node_mlp_layers,
                node_mlp_dim=args.node_mlp_dim,
                edge_update=args.edge_update,
                edge_mlp_layers=args.edge_mlp_layers,
                edge_mlp_dim=args.edge_mlp_dim,
                mlp_activation=args.mlp_activation,
                dropout=args.dropout,
                skip_connect_input=args.skip_connect_input,
                num_landcover_classes=args.num_landcover_classes,
                num_field_bins=args.num_field_bins,
                decoder_num_hidden=args.decoder_num_hidden,
                label_smoothing=args.label_smoothing,
                checkpoint_gradients=checkpoint_gradients,
            )

        if floodfields:
            self.field_to_X = floodfield.SideChainBuilder()
            self.X_to_field = floodfield.FieldAngles()
            self.loss_rmsd = floodfield.LossFloodfieldRMSD()
            self.loss_clash = floodfield.LossFloodfieldClashes()

        self.loss_eps = 1e-5

    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        D: torch.LongTensor,
        t: Optional[torch.Tensor] = None,
        sample_noise: bool = False,
        permute_idx: Optional[torch.LongTensor] = None,
        priority: Optional[torch.LongTensor] = None,
    ) -> dict:
        # Sample noisy backbones
        X_noise = X
        if sample_noise and hasattr(self, "noise_perturb"):
            X_bb = X[:, :, :4, :]
            _schedule = self.noise_perturb.noise_schedule
            t = self.noise_perturb.sample_t(C, t)
            X_noise_bb = self.noise_perturb(X_bb, C, t=t)
            if self.sidechains:
                # Rebuild sidechains on noised backbone from native field angles
                field, mask_field = self.X_to_field(X, C, D)
                X_noise, mask_X = self.field_to_X(X_noise_bb, C, D, field)
            else:
                pass
                # TODO IDK what to return here

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X_noise, C, t=t)

        logp_D_marginals = None
        if self.kwargs["predict_D_marginals"]:
            logp_D_marginals, _ = self.decoder_D_marginals(D, node_h, mask_i)

        logp_D_potts = None
        if self.kwargs["predict_D_potts"]:
            logp_D_potts = self.decoder_D_potts.loss(
                D, node_h, edge_h, edge_idx, mask_i, mask_ij
            )

        # Sample random permutations and build autoregressive mask
        if permute_idx is None:
            permute_idx = self.traversal(X, C, priority=priority)

        if self.floodfields:
            # In one-stage packing, predict S and field angles in an interleaved manner
            (
                logp_D,
                logp_field,
                field,
                mask_field,
                node_h_field,
                _,
                _,
                _,
                mask_ij_causal,
            ) = self.decoder(
                X_noise, C, D, node_h, edge_h, edge_idx, mask_i, mask_ij, permute_idx
            )
        else:
            logp_D = (None,)
            logp_field = None
            field = None
            mask_field = None
            node_h_field = None
            mask_ij_causal = None

        if self.separate_packing:
            # In two-stage packing, re-process embeddings with sequence
            node_h = node_h + mask_i.unsqueeze(-1) * self.embed_D(D)
            node_h, edge_h = self.encoder_D_gnn(
                node_h, edge_h, edge_idx, mask_i, mask_ij
            )
            _, logp_field, field, mask_field, node_h_field, _, _, _, _ = self.decoder_field(
                X_noise, C, D, node_h, edge_h, edge_idx, mask_i, mask_ij, permute_idx
            )
        if t is None:
            t = torch.zeros(C.size(0), device=C.device)
        outputs = {
            "logp_D": logp_D,
            "logp_field": logp_field,
            "logp_D_marginals": logp_D_marginals,
            "logp_D_potts": logp_D_potts,
            "field": field,
            "mask_field": mask_field,
            "node_h_field": node_h_field,
            "mask_i": mask_i,
            "mask_ij": mask_ij,
            "mask_ij_causal": mask_ij_causal,
            "edge_idx": edge_idx,
            "permute_idx": permute_idx,
            "X_noise": X_noise,
            "t": t,
        }
        return outputs
    

    def set_gradient_checkpointing(self, flag: bool):
        """Sets gradient checkpointing to `flag` on all relevant modules"""
        self.encoder.checkpoint_gradients = flag
        self.encoder.gnn.checkpoint_gradients = flag
        if self.floodfields:
            self.decoder.checkpoint_gradients = flag
            self.decoder.gnn.checkpoint_gradients = flag
        if self.separate_packing:
            self.encoder_D_gnn.checkpoint_gradients = flag
            self.decoder_field.checkpoint_gradients = flag
            self.decoder_field.gnn.checkpoint_gradients = flag

    @validate_XC()
    def encode(
        self, X: torch.Tensor, C: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the backbone and (optionally) the noise level.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, H, W, 3)`.
            C (torch.LongTensor): Chain map with shape
                `(num_batch, num_residues)`.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.

        Returns:
            node_h (torch.Tensor): Node features with shape
                `(num_batch, H, dim_nodes)`.
            edge_h (torch.Tensor): Edge features with shape
                `(num_batch, H, num_neighbors, dim_edges)`.
            edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, H, num_neighbors)`.
            mask_i (torch.Tensor): Node mask with shape
                `(num_batch, H)`.
            mask_ij (torch.Tensor): Edge mask with shape
                 `(num_batch, num_nodes, num_neighbors)`.
        """

        node_h_aux = None
        if hasattr(self, "time_features"):
            t = 0.0 if t is None else t
            node_h_aux = self.time_features(t)

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encoder(
            X, C, node_h_aux=node_h_aux
        )
        return node_h, edge_h, edge_idx, mask_i, mask_ij
    
    @validate_XC()
    def predict_marginals(
        self, X: torch.Tensor, C: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict sequence marginal likelihoods.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, H, W, 1)`.
            C (torch.LongTensor): Field map with shape
                `(num_batch, H, W, 1)`.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.

        Returns:
            log_probs_D (torch.Tensor): Node-wise sequence log probabilities
                with shape `(num_batch, H, 20)`. lol why 20. 
            mask_i (torch.Tensor): Node mask with shape
                `(num_batch, H)`.
        """

        if not self.kwargs["predict_D_marginals"]:
            raise Exception(
                "This version of GraphDesign was not trained with marginal prediction"
            )
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C, t)
        log_probs_D = self.decoder_D_marginals.log_probs_D(node_h, mask_i)
        return log_probs_D, mask_i
    
    @validate_XC()
    def predict_potts(
        self, X: torch.Tensor, C: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        """Predict sequence Potts model.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, H, W, 1)`.
            C (torch.LongTensor): Field map with shape
                `(num_batch, H, W, 1)`.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.

        Returns:
            h (torch.Tensor): The h tensor of a Potts model with dimensions
                `(seq_length, n_tokens)`.
            J (torch.Tensor): The J tensor of a Potts model with dimensions
                `(seq_length, seq_length, n_tokens, n_tokens)`.
            edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_residues, num_neighbors)` from GNN encoding.
        """
        if not self.kwargs["predict_S_potts"]:
            raise Exception(
                "This version of GraphDesign was not trained with Potts prediction"
            )
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C, t)
        h, J = self.decoder_S_potts(node_h, edge_h, edge_idx, mask_i, mask_ij)
        return h, J, edge_idx
    
    @validate_XC()
    def loss(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        D: torch.LongTensor,
        t: Optional[torch.Tensor] = None,
        permute_idx: Optional[torch.LongTensor] = None,
        sample_noise: bool = False,
        batched: bool = True,
        **kwargs
    ) -> dict:
        """Compute losses used for training.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, H, W, 1)`.
            C (torch.LongTensor): Chain map with shape
                `(num_batch, H, W, 1)`.
            D (torch.LongTensor): Description tensor with shape
                `(num_batch, H)`.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.
            permute_idx (torch.LongTensor, optional): Permutation tensor for
                fixing the autoregressive decoding order
                `(num_batch, num_residues)`. If `None` (default), a random
                decoding order will be generated.
            sample_noise (bool): Whether to apply noise to input backbones.
            batched (bool): Whether to batch average losses.

        Returns (dict):
            neglogp (torch.Tensor): Sum of `neglogp_D` and `neglogp_field` with
                shape `(num_batch, H, W, 1)`.
            neglogp_D (torch.Tensor): Average negative log probability per
                height, width identity with shape `(num_batch, H, W, 1)`.
            neglogp_D_marginals (torch.Tensor): Average negative log probability
                per height, width identity from marginal decoder with shape
                `(num_batch, H, W)`.
            neglogp_D_potts (torch.Tensor): Average negative log probability per
                residue identity from Potts decoder with shape
                `(num_batch, num_residues)`.
            neglogp_field (torch.Tensor): Average negative log probability per field
                angle with shape `(num_batch, num_residues)`.
            mask_field (torch.Tensor): Field angle mask with shape
                `(batch_size, H, W, 2)`.
            rmsd (torch.Tensor): Average RMSD per field after sampling.
            clash (torch.Tensor): Average number of clashes per field after
                sampling.
            permute_idx (LongTensor, optional): Permutation tensor that was
                used for the autoregressive decoding order with shape
                `(num_batch, H, W)`.
        """

        o = self.forward(
            X, C, D, t=t, permute_idx=permute_idx, sample_noise=sample_noise
        )

        # Aggregate into per-residue scores for the batch
        if batched:
            _avg = lambda m, l: (m * l).sum() / (m.sum() + self.loss_eps)
        else:
            _avg = lambda m, l: (m * l).sum(dim=tuple(range(1, l.dim()))) / (
                m.sum(dim=tuple(range(1, l.dim()))) + self.loss_eps
            )
        mask_D = o["mask_i"]
        neglogp_D = -_avg(mask_S, o["logp_D"])
        neglogp_field = -_avg(o["mask_field"], o["logp_field"])
        neglogp = neglogp_D + neglogp_field
        if o["logp_D_marginals"] is not None:
            neglogp_D_marginals = -_avg(mask_D, o["logp_D_marginals"])
            neglogp = neglogp + neglogp_D_marginals
        else:
            neglogp_D_marginals = None
        if o["logp_D_potts"] is not None:
            neglogp_D_potts = -_avg(mask_D, o["logp_D_potts"])
            neglogp = neglogp + neglogp_D_potts
        else:
            neglogp_D_potts = None

        # Evaluate sampled side chains
        decoder = self.decoder_field if self.separate_packing else self.decoder
        field_sample = decoder.decoder_field.sample(
            D, o["mask_field"], o["node_h_field"], o["mask_i"], temperature=0.01
        )
        X_sample, mask_X = self.field_to_X(o["X_noise"][:, :, :4, :], C, D, field_sample)

        # RMSD loss
        rmsd_i = self.loss_rmsd(o["X_noise"], X_sample, C, D)
        rmsd = _avg(mask_D, rmsd_i)

        # Clash loss measures clashes generated to the past
        clashes = self.loss_clash(
            X_sample, C, D, edge_idx=o["edge_idx"], mask_ij=o["mask_ij_causal"]
        )
        clash = _avg(mask_D, clashes)

        losses = {
            "neglogp": neglogp,
            "neglogp_D": neglogp_D,
            "neglogp_D_marginals": neglogp_D_marginals,
            "neglogp_D_potts": neglogp_D_potts,
            "neglogp_field": neglogp_field,
            "mask_field": o["mask_field"],
            "rmsd": rmsd,
            "clash": clash,
            "permute_idx": o["permute_idx"],
            "t": o["t"],
        }
        return losses
    

class BackboneEncoderGNN(nn.Module):
    """Graph Neural Network for processing protein structure into graph embeddings.

    Args:
        See documention of `structure.protein_graph.ProteinFeatureGraph`,
        and `graph.GraphNN` for more details.

        dim_nodes (int): Hidden dimension of node tensors.
        dim_edges (int): Hidden dimension of edge tensors.
        num_neighbors (int): Number of neighbors per nodes.
        node_features (tuple): List of node feature specifications. Features
            can be given as strings or as dictionaries.
        edge_features (tuple): List of edge feature specifications. Features
            can be given as strings or as dictionaries.
        num_layers (int): Number of layers.
        node_mlp_layers (int): Number of hidden layers for node update
            function.
        node_mlp_dim (int, optional): Dimension of hidden layers for node update
            function, defaults to match output dimension.
        edge_update (bool): Whether to include an edge update step.
        edge_mlp_layers (int): Number of hidden layers for edge update
            function.
        edge_mlp_dim (int, optional): Dimension of hidden layers for edge update
            function, defaults to match output dimension.
        skip_connect_input (bool): Whether to include skip connections between
            layers.
        mlp_activation (str): MLP nonlinearity function, `relu` or `softplus`
            accepted.
        dropout (float): Dropout fraction.
        graph_distance_feature_type (int):  Feature index or name to use for computing
            spatial distances between grid cells or infrastructure nodes for graph construction.
            Negative values will specify centroid across all feature channels (e.g., mean over
            depth, elevation, slope, etc.). Default is `-1` (centroid).
        graph_cutoff (float, optional): Cutoff distance for graph construction:
            mask any edges further than this cutoff. Default is `None`.
        graph_mask_interfaces (bool): Restrict connections only to within
            chains, excluding-between chain interactions. Default is `False`.
        graph_criterion (str): Method used for building graph from distances.
            Currently supported methods are `{knn, random_log, random_linear}`.
            Default is `knn`.
        graph_random_min_local (int): Minimum number of neighbors in GNN that
            come from local neighborhood, before random neighbors are chosen.
        checkpoint_gradients (bool): Switch to implement gradient checkpointing
            during training.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        node_h_aux (torch.LongTensor, optional): Auxiliary node features with
            shape `(num_batch, num_residues, dim_nodes)`.
        edge_h_aux (torch.LongTensor, optional): Auxiliary edge features with
            shape `(num_batch, num_residues, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor, optional): Input edge indices for neighbors
            with shape `(num_batch, num_residues, num_neighbors)`.
        mask_ij (torch.Tensor, optional): Input edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.

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
        dim_nodes: int = 128,
        dim_edges: int = 128,
        num_neighbors: int = 30,
        node_features: tuple = (("internal_coords", {"log_lengths": True}),),
        edge_features: tuple = (
            "distances_2mer",
            "orientations_2mer",
            "distances_chain",
        ),
        num_layers: int = 3,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        skip_connect_input: bool = False,
        mlp_activation: str = "softplus",
        dropout: float = 0.1,
        graph_distance_atom_type: int = -1,
        graph_cutoff: Optional[float] = None,
        graph_mask_interfaces: bool = False,
        graph_criterion: str = "knn",
        graph_random_min_local: int = 20,
        checkpoint_gradients: bool = False,
        **kwargs
    ) -> None:
        """Initialize BackboneEncoderGNN."""
        super(BackboneEncoderGNN, self).__init__()

        # Save configuration in kwargs
        self.kwargs = locals()
        self.kwargs.pop("self")
        for key in list(self.kwargs.keys()):
            if key.startswith("__") and key.endswith("__"):
                self.kwargs.pop(key)
        args = SimpleNamespace(**self.kwargs)

        # Important global options
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.checkpoint_gradients = checkpoint_gradients

        graph_kwargs = {
            "distance_atom_type": args.graph_distance_atom_type,
            "cutoff": args.graph_cutoff,
            "mask_interfaces": args.graph_mask_interfaces,
            "criterion": args.graph_criterion,
            "random_min_local": args.graph_random_min_local,
        }

        self.feature_graph = flood_graph.FloodFeatureGraph(
            dim_nodes=args.dim_nodes,
            dim_edges=args.dim_edges,
            num_neighbors=args.num_neighbors,
            graph_kwargs=graph_kwargs,
            node_features=args.node_features,
            edge_features=args.edge_features,
        )

        self.gnn = graph.GraphNN(
            dim_nodes=args.dim_nodes,
            dim_edges=args.dim_edges,
            num_layers=args.num_layers,
            node_mlp_layers=args.node_mlp_layers,
            node_mlp_dim=args.node_mlp_dim,
            edge_update=args.edge_update,
            edge_mlp_layers=args.edge_mlp_layers,
            edge_mlp_dim=args.edge_mlp_dim,
            mlp_activation=args.mlp_activation,
            dropout=args.dropout,
            norm="transformer",
            scale=args.num_neighbors,
            skip_connect_input=args.skip_connect_input,
            checkpoint_gradients=checkpoint_gradients,
        )

    @validate_XC(all_atom=False)
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        node_h_aux: Optional[torch.Tensor] = None,
        edge_h_aux: Optional[torch.Tensor] = None,
        edge_idx: Optional[torch.Tensor] = None,
        mask_ij: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor
    ]:
        """Encode XC backbone structure into node and edge features."""
        num_batch, num_residues = C.shape

        # Hack to enable checkpointing
        if self.checkpoint_gradients and (not X.requires_grad):
            X.requires_grad = True

        node_h, edge_h, edge_idx, mask_i, mask_ij = self._checkpoint(
            self.feature_graph, X, C, edge_idx, mask_ij
        )

        if node_h_aux is not None:
            node_h = node_h + mask_i.unsqueeze(-1) * node_h_aux
        if edge_h_aux is not None:
            edge_h = edge_h + mask_ij.unsqueeze(-1) * edge_h_aux

        node_h, edge_h = self.gnn(node_h, edge_h, edge_idx, mask_i, mask_ij)
        return node_h, edge_h, edge_idx, mask_i, mask_ij

    def _checkpoint(self, module: nn.Module, *args) -> nn.Module:
        if self.checkpoint_gradients:
            return checkpoint(module, *args)
        else:
            return module(*args)


class FloodTraversalSpatial(nn.Module):
    """Samples spatially correlated permutations over a flood-prone area.

    Args:
        smooth_alpha (float): Smoothing factor for spatial correlation.
        smooth_steps (int): Number of smoothing iterations.
        smooth_randomize (bool): Whether to randomly sample smoothing strength.
        graph_num_neighbors (int): Number of spatial neighbors.
        deterministic (bool): Use deterministic permutation order.
    """

    def __init__(
        self,
        smooth_alpha: float = 1.0,
        smooth_steps: int = 5,
        smooth_randomize: bool = True,
        graph_num_neighbors: int = 30,
        deterministic: bool = False,
    ) -> None:
        super(FloodTraversalSpatial, self).__init__()

        self.smooth_alpha = smooth_alpha
        self.smooth_steps = smooth_steps
        self.smooth_randomize = smooth_randomize
        self.deterministic = deterministic
        self._determistic_seed = 42
        self.norm_eps = 1e-5

        self.flood_graph = flood_graph.FloodGraph(num_neighbors=graph_num_neighbors)

    def forward(
        self,
        X: torch.Tensor,  # shape (B, N, F), e.g., (batch, locations, features like depth/elevation)
        C: torch.LongTensor,  # shape (B, N), e.g., SLR or meteorology map ID
        priority: Optional[torch.Tensor] = None,  # (B, N) criticality score
    ):
        B, N, _ = X.shape

        if not self.deterministic:
            z = torch.rand(B, N, device=X.device)
        else:
            with torch.random.fork_rng():
                torch.manual_seed(self._determistic_seed)
                z = torch.rand(B, N, device=X.device)

        alpha = self.smooth_alpha
        if self.smooth_randomize and not self.deterministic:
            alpha = torch.rand((), device=X.device)

        if alpha > 0:
            edge_idx, mask_ij = self.flood_graph(X, C)  # spatial adjacency
            for _ in range(self.smooth_steps):
                z_neighbors = graph.collect_neighbors(z.unsqueeze(-1), edge_idx).squeeze(-1)
                z_average = (mask_ij * z_neighbors).sum(2) / (
                    mask_ij.sum(2) + self.norm_eps
                )
                z = alpha * z_average + (1.0 - alpha) * z

        if priority is not None:
            z = z + priority

        permute_idx = torch.argsort(z, dim=-1)
        return permute_idx
    
def load_model(
    weight_file: str,
    device: str = "cpu",
    strict: bool = False,
    strict_unexpected: bool = True,
    verbose: bool = True,
) -> FloodGraphDesign:
    """Load model `FloodGraphDesign`

    Args:
        weight_file (str): The destination path of the model weights to load.
            Compatible with files saved by `save_model`.
        device (str, optional): Pytorch device specification, e.g. `'cuda'` for
        GPU. Default is `'cpu'`.
        strict (bool): Whether to require that the keys match between the
            input file weights and the model created from the parameters stored
            in the model kwargs.
        strict_unexpected (bool): Whether to require that there are no
            unexpected keys when loading model weights, as distinct from the
            strict option which doesn't allow for missing keys either. By
            default, we use this option rather than strict for ease of
            development when adding model features.

    Returns:
        model (GraphDesign): Instance of `GraphDesign` with loaded weights.
    """
    return utility_load_model(
        weight_file,
        FloodGraphDesign,
        device=device,
        strict=strict,
        strict_unexpected=strict_unexpected,
        verbose=verbose,
    )