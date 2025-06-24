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
from cascadia.data.xcd import validate_XC
from cascadia.layers import complexity, graph
from cascadia.layers.structure import diffusion, potts, flood_graph, floodfield
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
        self.predict_S_potts = predict_D_potts
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
                field_embedding=args.field_embedding,
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
                field_embedding=args.field_embedding,
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
            self.field_to_X = floodfield.FieldBuilder()
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
            if self.fields:
                # Rebuild flood fields on noised backbone from native field angles
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
        neglogp_D = -_avg(mask_D, o["logp_D"])
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
    

    @torch.no_grad()
    @validate_XC()
    def sample(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        D: Optional[torch.LongTensor] = None,
        t: Optional[Union[float, torch.Tensor]] = None,
        t_packing: Optional[Union[float, torch.Tensor]] = None,
        mask_sample: Optional[torch.Tensor] = None,
        permute_idx: Optional[torch.LongTensor] = None,
        temperature_D: float = 0.1,
        temperature_field: float = 1e-3,
        clamped: bool = False,
        resample_field: bool = True,
        return_scores: bool = False,
        top_p_D: Optional[float] = None,
        ban_D: Optional[tuple] = None,
        sampling_method: Literal["potts", "autoregressive"] = "autoregressive",
        regularization: Optional[str] = "LCP",
        potts_sweeps: int = 500,
        potts_proposal: Literal["dlmc", "chromatic"] = "dlmc",
        verbose: bool = False,
        symmetry_order: Optional[int] = None,
    ) -> tuple:
        """Sample sequence and side chain conformations given an input structure.

        Args:
            X (torch.Tensor): All atom coordinates with shape
                `(num_batch, H, W, 1)`.
            C (torch.LongTensor): SLR / meteorological maps with shape
                `(num_batch, H, W, K)`.
            D (torch.LongTensor): Sequence tensor with shape
                `(num_batch, H, W, 1)`.
            t (float or torch.Tensor, optional): Diffusion time for models trained with
                diffusion augmentation of input structures. Setting `t=0` or
                `t=None` will condition the model to treat the structure as
                exact coordinates, while values of `t > 0` will condition
                the model to treat structures as though they were drawn from
                noise-augmented ensembles with that noise level. Default is `None`,
                while for robust design we recommend `t=0.5`. May be a float or
                a tensor of shape `(num_batch)`.
            t_packing (float or torch.Tensor, optional): Potentially separate diffusion
                time for packing.
            mask_sample (torch.Tensor, optional): Binary tensor mask indicating
                positions to be sampled with shape `(num_batch, num_residues)` or
                position-specific valid amino acid choices with shape
                `(num_batch, num_residues, num_alphabet)`. If `None` (default), all
                positions will be sampled.
            permute_idx (LongTensor, optional): Permutation tensor for fixing
                the autoregressive decoding order `(num_batch, num_residues)`.
                If `None` (default), a random decoding order will be generated.
            temperature_D (float): Temperature parameter for sampling description
                tokens. A value of `temperature_D=1.0` corresponds to the
                model's unadjusted positions, though because of training such as
                label smoothing values less than 1.0 are recommended. Default is
                `0.1`.
            temperature_field (float): Temperature parameter for sampling field
                angles. Even if a high temperature sequence is sampled, this is
                recommended to always be low. Default is `1E-3`.
            clamped (bool): If `True`, no sampling is done and the likelihood
                values will be calculated for the input sequence and structure.
                Used for validating the sequential versus parallel decoding
                modes. Default is `False`.
            resample_field (bool): If `True`, all field angles will be resampled,
                even for sequence positions that were not sampled (i.e. the model
                will perform global repacking). Default is `True`.
            return_scores (bool): If `True`, return dictionary containing
                likelihood scores similar to those produced by `forward`.
            top_p_D (float, optional): Option to perform top-p sampling for
                autoregressive sequence decoding. If not `None` it will be the
                top-p value [1].
                [1] Holtzman et al. The Curious Case of Neural Text Degeneration. (2020)
            ban_D (tuple, optional): An optional set of token indices from
                `cascadia.constants.NCLD` to ban during sampling.
            sampling_method (str): Sampling method for decoding sequence from structure.
                If `autoregressive`, sequences will be designed by ancestral sampling with
                the autoregessive decoder head. If `potts`, sequences will be designed
                via MCMC with the potts decoder head.
            regularization (str, optional): Optional sequence regularization to use
                during decoding. Can be `LCP` for Local Composition Perplexity regularization
                which penalizes local sequence windows from having unnaturally low
                compositional entropies. (Implemented for both `potts` and `autoregressive`)
            potts_sweeps (int): Number of sweeps to perform for MCMC sampling of `potts`
                decoder. A sweep corresponds to a sufficient number of Monte Carlo steps
                such that every position could have changed.
            potts_proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `chromatic`
                for Gibbs sampling with graph coloring.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
            symmetry_order (int, optional): Optional integer argument to enable
                symmetric sequence decoding under `symmetry_order`-order symmetry.
                The first `(num_nodes // symmetry_order)` states will be free to
                move, and all consecutively tiled sets of states will be locked
                to these during decoding. Internally this is accomplished by
                summing the parameters Potts model under a symmetry constraint
                into this reduced sized system and then back imputing at the end.
                Currently only implemented for Potts models.

        Returns:
            X_sample (torch.Tensor): Sampled all atom coordinates with shape
                `(num_batch, H, W, 1)`.
            D_sample (torch.LongTensor): Sampled description tensor with shape
                `(num_batch, H, W)`.
            permute_idx (torch.LongTensor): Permutation tensor that was used
                for the autoregressive decoding order with shape
                `(num_batch, H, W)`.
            scores (dict, optional): Dictionary containing likelihood scores
                similar to those produced by `forward`.
        """
        if X.shape[2] == 4:
            X = F.pad(X, [0, 0, 0, 10])
        landcover_classes = constants.NLCD
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C, t=t)

        # Process sampling mask
        logits_init = torch.zeros(
            list(C.shape) + [len(landcover_classes)], device=C.device
        ).float()
        if ban_D is not None:
            ban_D = [landcover_classes.index(c) for c in ban_D]
        mask_sample, mask_sample_1D, D_init = potts.init_sampling_masks(
            logits_init, mask_sample, D=D, ban_D=ban_D
        )
        if not clamped:
            D = D_init

        # Sample random permutations and build autoregressive mask
        if permute_idx is None:
            permute_idx = self.traversal(X, C, priority=mask_sample_1D)

        if symmetry_order is not None and not (sampling_method == "potts"):
            raise NotImplementedError(
                "Symmetric decoding is currently only supported for Potts models"
            )

        if sampling_method == "potts":
            if not self.kwargs["predict_D_potts"]:
                raise Exception(
                    "This FloodGraphDesign model was not trained with Potts prediction"
                )

            # Complexity regularization
            penalty_func = None
            mask_ij_coloring = None
            edge_idx_coloring = None
            if regularization == "LCP":
                C_complexity = (
                    C
                    if symmetry_order is None
                    else C[:, : C.shape[1] // symmetry_order]
                )
                penalty_func = lambda _D: complexity.complexity_lcp(_D, C_complexity)
                # edge_idx_coloring, mask_ij_coloring = complexity.graph_lcp(C, edge_idx, mask_ij)

            D_sample, _ = self.decoder_S_potts.sample(
                node_h,
                edge_h,
                edge_idx,
                mask_i,
                mask_ij,
                D=D,
                mask_sample=mask_sample,
                temperature=temperature_D,
                num_sweeps=potts_sweeps,
                penalty_func=penalty_func,
                proposal=potts_proposal,
                rejection_step=(potts_proposal == "chromatic"),
                verbose=verbose,
                edge_idx_coloring=edge_idx_coloring,
                mask_ij_coloring=mask_ij_coloring,
                symmetry_order=symmetry_order,
            )
            field_sample, logp_D, logp_field = None, None, None
        else:
            # Sample sequence (and field angles if one-stage)

            # Complexity regularization
            bias_D_func = None
            if regularization == "LCP":
                bias_D_func = complexity.complexity_scores_lcp_t

            D_sample, field_sample, logp_D, logp_field, _ = self.decoder.decode(
                X,
                C,
                D,
                node_h,
                edge_h,
                edge_idx,
                mask_i,
                mask_ij,
                permute_idx,
                temperature_D=temperature_D,
                temperature_field=temperature_field,
                sample=not clamped,
                mask_sample=mask_sample,
                resample_field=resample_field,
                top_p_D=top_p_D,
                ban_D=ban_D,
                bias_S_func=bias_D_func,
            )

        if self.separate_packing:
            if t != t_packing:
                node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(
                    X, C, t=t_packing
                )

            # In two-stage packing, re-process embeddings with sequence
            node_h = node_h + mask_i.unsqueeze(-1) * self.embed_D(D_sample)
            node_h, edge_h = self.encoder_D_gnn(
                node_h, edge_h, edge_idx, mask_i, mask_ij
            )
            _, field_sample, _, logp_field, _ = self.decoder_field.decode(
                X,
                C,
                D_sample,
                node_h,
                edge_h,
                edge_idx,
                mask_i,
                mask_ij,
                permute_idx,
                temperature_field=temperature_field,
                sample=not clamped,
                mask_sample=mask_sample_1D,
                resample_field=resample_field,
            )

        # Rebuild side chains
        X_sample, mask_X = self.field_to_X(X[:, :, :4, :], C, D_sample, field_sample)

        if return_scores:
            if sampling_method == "potts":
                raise NotImplementedError

            # Summarize
            mask_field = floodfield.field_mask(C, D_sample)
            neglogp_D = -(mask_i * logp_D).sum([1]) / (
                (mask_i).sum([1]) + self.loss_eps
            )
            neglogp_field = -(mask_field * logp_field).sum([1, 2]) / (
                mask_field.sum([1, 2]) + self.loss_eps
            )

            scores = {
                "neglogp_D": neglogp_D,
                "neglogp_field": neglogp_field,
                "logp_D": logp_D,
                "logp_field": logp_field,
                "mask_i": mask_i,
                "mask_field": mask_field,
            }
            return X_sample, D_sample, permute_idx, scores
        else:
            return X_sample, D_sample, permute_idx

    @validate_XC()
    def pack(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        D: torch.LongTensor,
        permute_idx: Optional[torch.LongTensor] = None,
        temperature_field: float = 1e-3,
        clamped: bool = False,
        resample_field: bool = True,
        return_scores: bool = False,
    ) -> tuple:
        """Sample side chain conformations given an input structure.

        Args:
            X (torch.Tensor): All flood depth coordinates with shape
                `(num_batch, H, W, 1)`.
            C (torch.LongTensor): Field map with shape
                `(num_batch, H, W, K)`.
            D (torch.LongTensor): Descriptor tensor with shape
                `(num_batch, num_residues)`.
            permute_idx (LongTensor, optional): Permutation tensor for fixing
                the autoregressive decoding order `(num_batch, num_residues)`.
                If `None` (default), a random decoding order will be generated.
            temperature_field (float): Temperature parameter for sampling field
                angles. Even if a high temperature sequence is sampled, this is
                recommended to always be low. Default is `1E-3`.
            clamped (bool): If `True`, no sampling is done and the likelihood
                values will be calculated for the input sequence and structure.
                Used for validating the sequential versus parallel decoding
                modes. Default is `False`
            resample_field (bool): If `True`, all field angles will be resampled,
                even for sequence positions that were not sampled (i.e. global
                repacking). Default is `True`.
            return_scores (bool): If `True`, return dictionary containing
                likelihood scores similar to those produced by `forward`.

        Returns:
            X_sample (torch.Tensor): Sampled all atom coordinates with shape
                `(num_batch, num_residues, 14, 3)`.
            neglogp_field (torch.Tensor, optional): Average negative log
                probability per field angle.
            permute_idx (torch.LongTensor): Permutation tensor that was used
                for the autoregressive decoding order with shape
                `(num_batch, num_residues)`.
            scores (dict, optional): Dictionary containing likelihood scores
                similar to those produced by `forward`.
        """
        assert self.separate_packing

        with torch.no_grad():
            if X.shape[2] == 4:
                X = F.pad(X, [0, 0, 0, 10])

            node_h, edge_h, edge_idx, mask_i, mask_ij = self.encode(X, C)

            # Sample random permutations and build autoregressive mask
            if permute_idx is None:
                permute_idx = self.traversal(X, C)

            # In two-stage packing, re-process embeddings with sequence
            node_h = node_h + mask_i.unsqueeze(-1) * self.embed_D(D)
            node_h, edge_h = self.encoder_D_gnn(
                node_h, edge_h, edge_idx, mask_i, mask_ij
            )
            _, field_sample, _, logp_field, _ = self.decoder_field.decode(
                X,
                C,
                D,
                node_h,
                edge_h,
                edge_idx,
                mask_i,
                mask_ij,
                permute_idx,
                temperature_field=temperature_field,
                sample=not clamped,
                resample_field=resample_field,
            )

            X_sample, mask_X = self.field_to_X(X[:, :, :4, :], C, D, field_sample)

            # Summarize
            mask_field = floodfield.field_mask(C, D)
            neglogp_field = -(mask_field * logp_field).sum([1, 2]) / (
                mask_field.sum([1, 2]) + self.loss_eps
            )
        if return_scores:
            scores = {
                "neglogp_field": neglogp_field,
                "logp_field": logp_field,
                "mask_i": mask_i,
                "mask_field": mask_field,
            }
            return X_sample, permute_idx, scores
        else:
            return X_sample, permute_idx

        return X_sample, neglogp_field, permute_idx # unsure why this boi here (linter?)

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

class FloodfieldDecoderGNN(nn.Module):
    """Autoregressively generate floodfields given backbone graph embeddings.

    Args:
        See documention of `structure.protein_graph.ProteinFeatureGraph`,
        and `graph.GraphNN` for more details.

        dim_nodes (int): Hidden dimension of node tensors.
        dim_edges (int): Hidden dimension of edge tensors.
        num_neighbors (int): Number of neighbors per nodes.
        predict_D (bool): Whether to predict land descriptors.
        predict_field (bool): Whether to predict field angles. I still dont really understand this
            but I dont have to rn. 
        sequence_embedding (str): How to represent sequence when decoding.
            Currently the only option is `linear`.
        floodfield_embedding (str): How to represent field angles when decoding.
            Options include `field_linear` for a simple linear layer, `field_rbf`
            for a featurization based on smooth binning of field angles,
            `X_direct` which directly encodes the high-res flood coordinates using
            random Fourier features, and `mixed_field_X` which uses both the
            featurizations of `field_rbf` and of `X_direct`.
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
        num_landcover_classes (int): Number of possible residues.
        num_field_bins (int): Number of field bins for smooth binning of field angles
            used when `floodfield_embedding` is `field_rbf` or `mixed_field_X`.
        decoder_num_hidden (int): Dimension of hidden layers.
        label_smoothing (float): Level of smoothing to apply to sequence and
            floodfield labels.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, H, W, 1)`.
        C (torch.LongTensor): SLR / Meteorological map with shape `(num_batch, H, W, K)`.
        D (torch.LongTensor): Land descriptor tensor with shape
            `(num_batch, H, W, 1)`.
        node_h (torch.Tensor): Node features with shape
            `(num_batch, HxW, dim_nodes)`. # wondering if H should really be HxW???
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, HxW, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_residues, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape
            `(num_batch, num_residues)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
        permute_idx (torch.LongTensor): Permutation tensor for fixing the
            autoregressive decoding order `(num_batch, num_residues)`. If
            `None` (default), a random decoding order will be generated.

    Outputs:
        logp_D (torch.Tensor): Sequence log likelihoods per gridcell with shape
            `(num_batch, HxW)`.
        logp_field (torch.Tensor): Field angle Log likelihoods per residue with
            shape `(num_batch, num_residues, 4)`.
        field (torch.Tensor): floodfield field angles in radians with shape
            `(num_batch, num_residues, 4)`.
        mask_field (torch.Tensor): Mask for field angles with shape
            `(num_batch, num_residues, 4)`.
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
        predict_D: bool = True,
        predict_field: bool = True,
        sequence_embedding: str = "linear",
        floodfield_embedding: str = "mixed_field_X",
        num_layers: int = 3,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        skip_connect_input: bool = False,
        mlp_activation: str = "softplus",
        dropout: float = 0.1,
        num_landcover_classes: int = 20,
        num_field_bins: int = 20,
        decoder_num_hidden: int = 512,
        label_smoothing: float = 0.1,
        checkpoint_gradients: bool = False,
        **kwargs
    ):
        super(FloodfieldDecoderGNN, self).__init__()

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

        # Predict D, field or both?
        assert predict_D or predict_field
        self.predict_D = predict_D
        self.predict_field = predict_field

        self.sequence_embedding = sequence_embedding
        self.floodfield_embedding = floodfield_embedding
        if self.sequence_embedding == "linear":
            self.W_D = nn.Embedding(num_landcover_classes, dim_edges)

        # If we are predicting field angles, then embed them
        if self.predict_field:
            if self.floodfield_embedding == "field_linear":
                self.W_field = nn.Linear(8, dim_edges)
            elif self.floodfield_embedding == "field_rbf":
                self.embed_field = NodeFieldRBF(
                    dim_out=args.dim_edges, num_field=4, num_field_bins=args.num_field_bins
                )
            elif self.floodfield_embedding == "X_direct":
                self.embed_X = EdgeFlooodfieldsDirect(dim_out=dim_edges)
            elif self.floodfield_embedding == "mixed_field_X":
                self.embed_field = NodefieldRBF(
                    dim_out=args.dim_edges, num_field=4, num_field_bins=args.num_field_bins
                )
                self.embed_X = EdgeFloodfieldsDirect(dim_out=dim_edges, basis_type="rff")

        # Decoder GNN process backbone
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

        if self.predict_D:
            self.decoder_D = NodePredictorD(
                num_landcover_classes=args.num_landcover_classes,
                dim_nodes=args.dim_nodes,
                dim_hidden=args.decoder_num_hidden,
                label_smoothing=args.label_smoothing,
            )

        if self.predict_field:
            self.decoder_field = NodePredictorField(
                num_landcover_classes=args.num_landcover_classes,
                num_field_bins=args.num_field_bins,
                dim_nodes=args.dim_nodes,
                dim_hidden=args.decoder_num_hidden,
                label_smoothing=args.label_smoothing,
            )

        self.loss_eps = 1e-5
        self.field_to_X = floodfield.FloodFieldBuilder()
        self.X_to_field = floodfield.FieldAngles()
    
    @validate_XC()
    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        D: torch.LongTensor,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
        permute_idx: torch.LongTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.LongTensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Predict descriptor and field angles autoregressively given graph features."""

        # Permute graph representation
        (
            node_h_p,
            edge_h_p,
            edge_idx_p,
            mask_i_p,
            mask_ij_p,
        ) = graph.permute_graph_embeddings(
            node_h, edge_h, edge_idx, mask_i, mask_ij, permute_idx
        )

        # Permute sequence and side chain field angles
        X_p = graph.permute_tensor(X, 1, permute_idx)
        C_p = graph.permute_tensor(C, 1, permute_idx)
        D_p = graph.permute_tensor(D, 1, permute_idx) # Hmmm
        field, mask_field = self.X_to_field(X, C, D)
        field_p = graph.permute_tensor(field, -2, permute_idx)

        # Decode system autoregressively in the permuted coordinates
        node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p = self._decode_inner(
            X_p, C_p, D_p, field_p, node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p
        )

        # Unpermute graph representation
        permute_idx_inverse = torch.argsort(permute_idx, dim=-1)
        node_h, edge_h, edge_idx, mask_i, mask_ij = graph.permute_graph_embeddings(
            node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p, permute_idx_inverse
        )

        # Predict per-position joint probabilities of each side-chain's sequence and structure
        logp_D, log_probs_D, logp_field, log_probs_field = None, None, None, None
        if self.predict_D:
            (logp_D, log_probs_d,) = self.decoder_D(D, node_h, mask_i)
        if self.predict_field:
            (logp_field, log_probs_field,) = self.decoder_field(
                D, field, mask_field, node_h, mask_i
            )
        return (
            logp_D,
            logp_field,
            field,
            mask_field,
            node_h,
            edge_h,
            edge_idx,
            mask_i,
            mask_ij,
        )
    
    def _decode_inner(
        self, X_p, C_p, D_p, field_p, node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p
    ):
        # Build autoregressive mask
        mask_ij_p = graph.edge_mask_causal(edge_idx_p, mask_ij_p)

        # Add sequence context
        h_D_p = self.W_D(D_p)
        h_D_p_ij = graph.collect_neighbors(h_D_p, edge_idx_p)
        edge_h_p = edge_h_p + mask_ij_p.unsqueeze(-1) * h_D_p_ij

        # Add side chain context
        if self.predict_field:
            if self.floodfield_embedding in ["field_rbf", "mixed_field_X"]:
                h_field_p = self.embed_field(field_p)
                h_field_p_ij = graph.collect_neighbors(h_field_p, edge_idx_p)
                edge_h_p = edge_h_p + mask_ij_p.unsqueeze(-1) * h_field_p_ij

            if self.floodfield_embedding == "mixed_field_X":
                edge_feature = self.embed_X(X_p, C_p, D_p, edge_idx_p)
                edge_h_p = edge_h_p + mask_ij_p.unsqueeze(-1) * edge_feature

        # Run decoder GNN in parallel (permuted)
        node_h_p, edge_h_p = self.gnn(
            node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p
        )
        return node_h_p, edge_h_p, edge_idx_p, mask_i_p, mask_ij_p
    
    def _decode_scatter(self, tensor, src, t):
        """Decoding utility function: Scatter."""
        idx = (t * torch.ones_like(src)).long()
        tensor.scatter_(1, idx, src)

    def _decode_pre_func(self, t, tensors_t):
        """Decoding pre-step function adds features based on current D and field."""
        _scatter_t = lambda tensor, src: self._decode_scatter(tensor, src, t)

        # Gather relevant tensors at step t
        edge_h_p_t = tensors_t["edge_h_cache"][0][:, t, :, :].unsqueeze(1)
        edge_idx_p_t = tensors_t["edge_idx"][:, t, :].unsqueeze(1)
        mask_ij_p_t = tensors_t["mask_ij"][:, t, :].unsqueeze(1)

        # Update the edge embeddings at t with the relevant context
        mask_ij_p_t = mask_ij_p_t.unsqueeze(-1)

        # Add sequence context
        h_D_p_ij_t = graph.collect_neighbors(tensors_t["h_D_p"], edge_idx_p_t)
        edge_h_p_t = edge_h_p_t + mask_ij_p_t * h_D_p_ij_t

        # Add field context
        if self.predict_field:
            if self.floodfield_embedding in ["field_rbf", "mixed_field_X"]:
                h_field_p_ij_t = graph.collect_neighbors(
                    tensors_t["h_field_p"], edge_idx_p_t
                )
                edge_h_p_t = edge_h_p_t + mask_ij_p_t * h_field_p_ij_t
            if self.floodfield_embedding == "mixed_field_X":
                h_field_p_ij_t = self.embed_X.step(
                    t,
                    tensors_t["X_p"],
                    tensors_t["C_p"],
                    tensors_t["D_p"],
                    edge_idx_p_t,
                )
                edge_h_p_t = edge_h_p_t + mask_ij_p_t * h_field_p_ij_t

        _scatter_t(tensors_t["edge_h_cache"][0], edge_h_p_t)
        return tensors_t
    
    def _decode_post_func(
        self,
        t,
        tensors_t,
        D_p_input,
        field_p_input,
        temperature_D,
        temperature_field,
        sample,
        resample_field,
        mask_sample,
        mask_sample_p=None,
        top_p_D=None,
        ban_D=None,
        bias_D_func=None,
    ):
        """Decoding post-step function updates D and field."""
        _scatter_t = lambda tensor, src: self._decode_scatter(tensor, src, t)

        # Gather relevant tensors at step t
        C_p_t = tensors_t["C_p"][:, t].unsqueeze(1)
        edge_h_p_t = tensors_t["edge_h_cache"][0][:, t, :, :].unsqueeze(1)
        edge_idx_p_t = tensors_t["edge_idx"][:, t, :].unsqueeze(1)
        mask_i_p_t = tensors_t["mask_i"][:, t].unsqueeze(1)
        mask_ij_p_t = tensors_t["mask_ij"][:, t, :].unsqueeze(1)
        node_h_p_t = tensors_t["node_h_cache"][-1][:, t, :].unsqueeze(1)
        idx_p_t = tensors_t["idx_p"][:, t].unsqueeze(1)

        # Sample updated sequence
        D_p_t = D_p_input[:, t].unsqueeze(1).clone()
        if self.predict_D and sample:
            bias_D = None
            if bias_D_func is not None:
                bias_D = bias_D_func(
                    t,
                    tensors_t["D_p"],
                    tensors_t["C_p"],
                    tensors_t["idx_p"],
                    edge_idx_p_t,
                    mask_ij_p_t,
                )
            mask_D_t = None
            if mask_sample_p is not None:
                mask_D_t = mask_sample_p[:, t]
            D_p_t = self.decoder_D.sample(
                node_h_p_t,
                mask_i_p_t,
                temperature=temperature_D,
                top_p=top_p_D,
                bias=bias_D,
                mask_S=mask_D_t,
            )

        _scatter_t(tensors_t["D_p"], D_p_t)

        # Sample updated side chain conformations
        mask_field_p_t = floodfield.field_mask(C_p_t, D_p_t)
        field_p_t = field_p_input[:, t].unsqueeze(1).clone()
        if self.predict_field and sample:
            # Sample field angles
            field_p_t_sample = self.decoder_field.sample(
                D_p_t, mask_field_p_t, node_h_p_t, mask_i_p_t, temperature=temperature_field
            )

            if mask_sample_p is not None and not resample_field:
                m = mask_sample_p[:, t].unsqueeze(-1).expand([-1, 4])
                field_p_t = torch.where(m > 0, field_p_t_sample, field_p_t)
            else:
                field_p_t = field_p_t_sample

            # Rebuild side chain
            X_p_t_bb = tensors_t["X_p"][:, t, :4, :].unsqueeze(1)
            X_p_t, _ = self.field_to_X(X_p_t_bb, C_p_t, D_p_t, field_p_t)
            _scatter_t(tensors_t["X_p"], X_p_t)
        _scatter_t(tensors_t["field_p"], field_p_t)

        # Score the updated sequence and field angles
        if self.predict_D:
            logp_D_p_t, _ = self.decoder_D(D_p_t, node_h_p_t, mask_i_p_t)
            _scatter_t(tensors_t["logp_D_p"], logp_D_p_t)
        if self.predict_field:
            logp_field_p_t, _ = self.decoder_field(
                D_p_t, field_p_t, mask_field_p_t, node_h_p_t, mask_i_p_t
            )
            _scatter_t(tensors_t["logp_field_p"], logp_field_p_t)

        # Update sequence and field features (permuted)
        h_D_p_t = self.W_D(D_p_t)
        _scatter_t(tensors_t["h_D_p"], h_D_p_t)

        # Cache field embeddings
        if self.predict_field and self.floodfield_embedding in ["field_rbf", "mixed_field_X"]:
            h_field_p_t = self.embed_field(field_p_t)
            _scatter_t(tensors_t["h_field_p"], h_field_p_t)
        return tensors_t
    
    @validate_XC()
    def decode(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        D: torch.LongTensor,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
        permute_idx: torch.LongTensor,
        temperature_D: float = 0.1,
        temperature_field: float = 1e-3,
        sample: bool = True,
        mask_sample: Optional[torch.Tensor] = None,
        resample_field: bool = True,
        top_p_D: Optional[float] = None,
        ban_D: Optional[tuple] = None,
        bias_D_func: Optional[Callable] = None,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Autoregressively decode sequence and field angles from graph features.

        Args:
            X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
            C (torch.LongTensor): Field map with shape
                `(num_batch, num_residues)`.
            D (torch.LongTensor): Descriptor tensor with shape
                `(num_batch, num_residues)`.
            node_h (torch.Tensor): Node features with shape
                `(num_batch, num_residues, dim_nodes)`.
            edge_h (torch.Tensor): Edge features with shape
                `(num_batch, num_residues, num_neighbors, dim_edges)`.
            edge_idx (torch.LongTensor): Edge indices for neighbors with shape
                `(num_batch, num_residues, num_neighbors)`.
            mask_i (torch.Tensor): Node mask with shape
                `(num_batch, num_residues)`.
            mask_ij (torch.Tensor): Edge mask with shape
                 `(num_batch, num_nodes, num_neighbors)`.
            temperature_field (float): Temperature parameter for sampling field
                angles. Even if a high temperature sequence is sampled, this is
                recommended to always be low. Default is `1E-3`.
            sample (bool): Whether to sample sequence and field angles.
            mask_sample (torch.Tensor, optional): Binary tensor mask indicating
                positions to be sampled with shape `(num_batch, num_residues)`.
                If `None` (default), all positions will be sampled.
            resample_field (bool): If `True`, all field angles will be resampled,
                even for sequence positions that were not sampled (i.e. global
                repacking). Default is `True`.
            top_p_D (float, optional): Top-p cutoff for Nucleus Sampling, see
                Holtzman et al ICLR 2020.
            ban_D (tuple, optional): An optional set of token indices from
                `chroma.constants.AA20` to ban during sampling.

        Returns:
            D (torch.LongTensor): Descriptor tensor with shape
                `(num_batch, num_residues)`.
            field (torch.Tensor): field angles with shape
                `(num_batch, num_residues, 4)`.
            logp_D (torch.Tensor): Sequence log likelihoods per residue with
                shape `(num_batch, num_residues)`.
            logp_field (torch.Tensor): field angle Log likelihoods per residue with
                shape `(num_batch, num_residues, 4)`.
            tensors (dict): Processed tensors from GNN decoding.
        """

        # Permute graph representation
        (
            node_h_p,
            edge_h_p,
            edge_idx_p,
            mask_i_p,
            mask_ij_p,
        ) = graph.permute_graph_embeddings(
            node_h, edge_h, edge_idx, mask_i, mask_ij, permute_idx
        )
        field, mask_field = self.X_to_field(X, C, S)

        # Build autoregressive mask
        mask_ij_p = graph.edge_mask_causal(edge_idx_p, mask_ij_p)

        # Initialize tensors
        B, N, K = list(edge_idx.shape)
        device = node_h.device
        idx = torch.arange(end=N, device=device)[None, :].expand(C.shape)
        tensors_init = {
            "X_p": graph.permute_tensor(X, 1, permute_idx),
            "C_p": graph.permute_tensor(C, 1, permute_idx),
            "idx_p": graph.permute_tensor(idx, 1, permute_idx),
            "D_p": torch.zeros_like(S),
            "field_p": torch.zeros([B, N, 4], device=device),
            "h_D_p": torch.zeros([B, N, self.dim_edges], device=device),
            "h_field_p": torch.zeros([B, N, self.dim_edges], device=device),
            "node_h": node_h_p,
            "edge_h": edge_h_p,
            "edge_idx": edge_idx_p,
            "mask_i": mask_i_p,
            "mask_ij": mask_ij_p,
            "logp_D_p": torch.zeros([B, N], device=device),
            "logp_field_p": torch.zeros([B, N, 4], device=device),
        }

        # As a sanity check against future state leakage,
        # we initialize D and field and zero and write in the true value
        # during sequential decoding
        D_p_input = graph.permute_tensor(D, 1, permute_idx)
        field_p_input = graph.permute_tensor(field, 1, permute_idx)
        mask_sample_p = None
        if mask_sample is not None:
            mask_sample_p = graph.permute_tensor(mask_sample, 1, permute_idx)

        # Pre-step function features current sequence and field angles
        pre_step_func = self._decode_pre_func

        # Post-step function samples sequence and/or field angles at step t
        post_step_func = lambda t, tensors_t: self._decode_post_func(
            t,
            tensors_t,
            D_p_input,
            field_p_input,
            temperature_D,
            temperature_field,
            sample,
            resample_field,
            mask_sample,
            mask_sample_p,
            top_p_D=top_p_D,
            ban_D=ban_D,
            bias_D_func=bias_D_func,
        )

        # Sequentially step through a forwards pass of the GNN at each
        # position along the node dimension (1), running _pre_func
        # and each iteration and _post_func after each iteration
        tensors = self.gnn.sequential(
            tensors_init,
            pre_step_function=pre_step_func,
            post_step_function=post_step_func,
        )

        # Unpermute sampled sequence and field angles
        permute_idx_inverse = torch.argsort(permute_idx, dim=-1)
        D = graph.permute_tensor(tensors["D_p"], 1, permute_idx_inverse)
        field = graph.permute_tensor(tensors["field_p"], 1, permute_idx_inverse)
        logp_D = graph.permute_tensor(tensors["logp_D_p"], 1, permute_idx_inverse)
        logp_field = graph.permute_tensor(tensors["logp_field_p"], 1, permute_idx_inverse)

        return D, field, logp_D, logp_field, tensors
    
def _filter_logits_top_p(logits, p=0.9):
    """Filter logits by top-p (Nucleus sampling).

    See Holtzman et al, ICLR 2020.

    Args:
        logits (Tensor): Logits with shape `(..., num_classes)`.
        p (float): Cutoff probability.

    Returns:
        logits_filters (Tensor): Filtered logits
            with shape `(..., num_classes)`.
    """
    logits_sort, indices_sort = torch.sort(logits, dim=-1, descending=True)
    probs_sort = F.softmax(logits_sort, dim=-1)
    probs_cumulative = torch.cumsum(probs_sort, dim=-1)

    # Remove tokens outside nucleus (aside from top token)
    logits_sort_filtered = logits_sort.clone()
    logits_sort_filtered[probs_cumulative > p] = -float("Inf")
    logits_sort_filtered[..., 0] = logits_sort[..., 0]

    # Unsort
    logits_filtered = logits_sort_filtered.gather(-1, indices_sort.argsort(-1))
    return logits_filtered

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