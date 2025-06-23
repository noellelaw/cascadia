"""Joint generative model for flood scenario synthesis and infrastructure impact propagation.

This framework supports both unconditional and conditional flood generation using 
diffusion-based backbones and graph-based refinement networks. It enables physically 
plausible, programmable flood scenario design under varying sea level rise (SLR), 
storm intensities, and topographic constraints.

Applications include:
    - Generating high-resolution synthetic flood extents across diverse geographic regions
    - Conditioning flood generation on climate drivers (e.g., SLR, wind fields, rainfall)
    - Propagating cascading infrastructure failures via graph neural networks
    - Supporting scenario planning for resilient infrastructure and emergency response
"""
import copy
import inspect
from collections import defaultdict, namedtuple
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from cascadia.data.flood import Flood
from cascadia.layers.structure.backbone import FloodBackbone
from cascadia.models import graph_backbone, graph_design_for_flood_impact


class Cascadia(nn.Module):
    """Cascadia: A generative model for flood risk assessment.

    Cascadia is a generative model for generating floods in high and low resource zones. 
    It combines a diffusion model for generating flood backbones together with discrete
    generative models for sequence and sidechain conformations given SLR and meteorological 
    data. It enables programmatic design of floods through a conditioning
    framework. This class provides an interface to:
        * Load model weights
        * Sample flood extents, both unconditionally and conditionally
        * Perform sequence design of sampled backbones

    Args:
        weights_backbone (str, optional): The name of the pre-trained weights
            to use for the backbone network.

        weights_design (str, optional): The name of the pre-trained weights
            to use for the autoregressive design network.

        device (Optional[str]): The device on which to load the networks. If
            not specified, will automatically use a CUDA device if available,
            otherwise CPU.

        strict (bool): Whether to strictly enforce that all keys in `weights`
            match the keys in the model's state_dict.

        verbose (bool, optional): Show outputs from download and loading.
            Default False.
    """

    def __init__(
        self,
        weights_backbone: str = "none",
        weights_design: str = "none",
        device: Optional[str] = None,
        strict: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        import warnings

        warnings.filterwarnings("ignore")

        # If no device is explicity specified automatically set device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.backbone_network = graph_backbone.load_model(
            weights_backbone, device=device, strict=strict, verbose=verbose
        ).eval()

        self.design_network = graph_design_for_flood_impact.load_model(
            weights_design,
            device=device,
            strict=strict,
            verbose=False,
        ).eval()

    def sample(
        self,
        # Backbone Args
        samples: int = 1,
        steps: int = 500,
        grid_size: List[int] = [100],
        tspan: List[float] = (1.0, 0.001),
        flood_init: Flood = None,
        conditioner: Optional[nn.Module] = None,
        langevin_factor: float = 2,
        langevin_isothermal: bool = False,
        inverse_temperature: float = 10,
        initialize_noise: bool = True,
        integrate_func: Literal["euler_maruyama", "heun"] = "euler_maruyama", # check this
        sde_func: Literal["langevin", "reverse_sde", "ode"] = "reverse_sde", # check this
        trajectory_length: int = 200,
        full_output: bool = False,
        # Sidechain Args
        design_ban_S: Optional[List[str]] = None,
        design_method: Literal["potts", "autoregressive"] = "potts",
        design_selection: Optional[Union[str, torch.Tensor]] = None,
        design_t: Optional[float] = 0.5,
        temperature_S: float = 0.01,
        temperature_chi: float = 1e-3,
        top_p_S: Optional[float] = None,
        verbose: bool = False,
    ) -> Union[
        Union[Flood, List[Flood]], Tuple[Union[Flood, List[Flood]], dict]
    ]:
        """
        Performs Backbone Sampling and Sequence Design and returns a Flood or list
        of Floods. Optionally this method can return additional arguments to show
        details of the sampling procedure.

        Args:
            Backbone sampling:
                samples (int, optional): The number of flood scenarios to generate.
                    Default is 1.
                steps (int, optional): The number of diffusion timesteps used during
                    the reverse SDE integration process. More steps yield higher quality
                    but increase computation time. Default is 500.
                grid_size (List[int], optional): The spatial dimensions of the flood grid
                    (e.g., [128, 128] for a 128x128 map). Default is [128,128].
                conditioner (Conditioner, optional): A conditioning module that injects 
                    auxiliary data such as sea level rise (SLR), meteorological drivers 
                    (wind, rainfall, pressure), and bathymetry. Used to guide flood generation.
                    Default is None.
                langevin_isothermal (bool, optional): Whether to use the isothermal
                    version of the Langevin SDE. May improve numerical stability. 
                    Default is False.
                integrate_func (str, optional): The integration scheme for solving the 
                    stochastic differential equation. Options include 'euler_maruyama' or 'heun'.
                    Default is 'euler_maruyama'.
                sde_func (str, optional): The name of the SDE function to use. Defaults
                    to “reverse_sde”.
                langevin_factor (float, optional): The factor that controls the strength
                    of the Langevin noise. Default is 2.
                inverse_temperature (float, optional): The inverse temperature parameter
                    for the SDE. Default is 10.
                flood_init (Flood, optional): The initial flood state. Defaults
                    to None.
                full_output (bool, optional): Whether to return the complete denoising
                    trajectory, including intermediate flood maps (X_trajectory), the
                    model's internal denoising targets (Xhat_trajectory), and the
                    unconstrained diffusion process (Xunc_trajectory). Default is False.
                initialize_noise (bool, optional): Whether to initialize the noise for
                    the SDE integration. Default is True.
                tspan (List[float], optional): The time span for the SDE integration.
                    Default is (1.0, 0.001).
                trajectory_length (int, optional): The number of sampled steps in the
                    trajectory output.  Maximum is `steps`. Default 200.
                **kwargs: Additional keyword arguments for the integration function.

            Sequence and flood grid sampling:
                design_ban_S (list of str, optional): Optional constraints on land cover or infrastructure types 
                    to exclude from refinement (e.g., banning roads from being reassigned to green space).
                design_method (str, optional): Specifies the refinement or synthesis method. 
                    Can be `gnn` (graph-based impact propagation) or `autoregressive` (pixel-wise flood refinement). 
                    Default is `gnn`.
                design_selection (str or Tensor, optional): Optional spatial mask to restrict refinement or 
                    synthesis. Can be:
                    1) A semantic selection string (e.g., "urban & coastal")
                    2) A binary mask of shape `(num_batch, H, W)` where 1 indicates refinable regions
                    3) A tensor of shape `(num_batch, H, W, num_classes)` specifying allowable class changes
                design_t (float or Tensor, optional): Diffusion time for models trained with noise-augmented 
                    flood inputs. Setting `t=0` treats the flood state as exact; higher values simulate 
                    uncertainty or degraded input conditions. Default is 0.5.
                temperature_S (float, optional): Temperature for flood class sampling (e.g., land use or damage class).
                    Default 0.01.
                temperature_chi (float, optional): Temperature for auxiliary dynamic fields (e.g., flow angle or velocity).
                    Default 1e-3.
                top_p_S (float, optional): Top-p sampling cutoff if using autoregressive flood class generation.

        Returns:
            floods: Sampled `Flood` object or list of  sampled `Flood` objects in
                the case of multiple outputs.
            full_output_dictionary (dict, optional): Additional outputs if
                `full_output=True`.
        """

        # Get KWARGS
        input_args = locals()

        # Dynamically get acceptable kwargs for each method
        backbone_keys = set(inspect.signature(self._sample).parameters)
        design_keys = set(inspect.signature(self.design).parameters)

        # Filter kwargs for each method using dictionary comprehension
        backbone_kwargs = {k: input_args[k] for k in input_args if k in backbone_keys}
        design_kwargs = {k: input_args[k] for k in input_args if k in design_keys}

        # Perform Sampling
        sample_output = self._sample(**backbone_kwargs)

        if full_output:
            flood_sample, output_dictionary = sample_output
        else:
            flood_sample = sample_output
            output_dictionary = None

        # Perform Design
        if design_method is None:
            floods = flood_sample
        else:
            if isinstance(flood_sample, list):
                floods = [
                    self.design(flood, **design_kwargs) for flood in flood_sample
                ]
            else:
                floods = self.design(flood_sample, **design_kwargs)

        # Perform conditioner postprocessing
        if (conditioner is not None) and hasattr(conditioner, "_postprocessing_"):
            floods, output_dictionary = self._postprocess(
                conditioner, floods, output_dictionary
            )

        if full_output:
            return floods, output_dictionary
        else:
            return floods

    def _postprocess(self, conditioner, floods, output_dictionary):
        if output_dictionary is None:
            if isinstance(floods, list):
                floods = [
                    conditioner._postprocessing_(flood) for flood in floods
                ]
            else:
                floods = conditioner._postprocessing_(floods)
        else:
            if isinstance(floods, list):
                p_dicts = []
                floods = []
                for i, flood in enumerate(floods):
                    p_dict = {}
                    for key, value in output_dictionary.items():
                        p_dict[key] = value[i]

                    flood, p_dict = conditioner._postprocessing_(flood, p_dict)
                    p_dicts.append(p_dict)

                # Merge Output Dictionaries
                output_dictionary = defaultdict(list)
                for p_dict in p_dicts:
                    for k, v in p_dict.items():
                        output_dictionary[k].append(v)
            else:
                floods, output_dictionary = conditioner._postprocessing_(
                    floods, output_dictionary
                )
        return floods, output_dictionary

    def _sample(
        self,
        samples: int = 1,
        steps: int = 500,
        grid_shapes: List[int] = [100],
        tspan: List[float] = (1.0, 0.001),
        flood_init: Flood = None,
        conditioner: Optional[nn.Module] = None,
        langevin_factor: float = 2,
        langevin_isothermal: bool = False,
        inverse_temperature: float = 10,
        initialize_noise: bool = True,
        integrate_func: Literal["euler_maruyama", "heun"] = "euler_maruyama",
        sde_func: Literal["langevin", "reverse_sde", "ode"] = "reverse_sde",
        trajectory_length: int = 200,
        full_output: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[List[Flood], List[Flood]],
        Tuple[List[Flood], List[Flood], List[Flood], List[Flood]],
    ]:
        """Samples backbones given chain lengths by integrating SDEs.

        Args:
            samples (int, optional): The number of floods to sample. Default is 1.
            steps (int, optional): The number of integration steps for the SDE.
                Default is 500.
            grid_shapes (List[int], optional): The shape of the flood grid.
                Default is [100].
            conditioner (Conditioner, optional): The conditioner object that provides
                the conditioning information. Default is None.
            langevin_isothermal (bool, optional): Whether to use the isothermal version
                of the Langevin SDE. Default is False.
            integrate_func (str, optional): The name of the integration function to use.
                Default is `euler_maruyama`.
            sde_func (str, optional): The name of the SDE function to use. Default is
                “reverse_sde”.
            langevin_factor (float, optional): The factor that controls the strength of
                the Langevin noise. Default is 2.
            inverse_temperature (float, optional): The inverse temperature parameter
                for the SDE. Default is 10.
            flood_init (Flood, optional): The initial flood state. Default is
                None.
            full_output (bool, optional): Whether to return the full outputs of the SDE
                integration, including Xhat and Xunc. Default is False.
            initialize_noise (bool, optional): Whether to initialize the noise for the
                SDE integration. Default is True.
            tspan (List[float], optional): The time span for the SDE integration.
                Default is (1.0, 0.001).
            trajectory_length (int, optional): The number of sampled steps in the
                trajectory output.  Maximum is `steps`. Default 200.
            **kwargs: Additional keyword arguments for the integration function.

        Returns:
            floods: Sampled `Flood` object or list of  sampled `Flood` objects in
                the case of multiple outputs.
            full_output_dictionary (dict, optional): Additional outputs if
                `full_output=True`.
        """

        if flood_init is not None:
            X_unc, C_unc, D_unc = flood_init.to_XCD()
        else:
            X_unc, C_unc, D_unc = self._init_floods(samples, grid_shapes)

        outs = self.backbone_network.sample_sde(
            C_unc,
            X_init=X_unc,
            conditioner=conditioner,
            tspan=tspan,
            langevin_isothermal=langevin_isothermal,
            integrate_func=integrate_func,
            sde_func=sde_func,
            langevin_factor=langevin_factor,
            inverse_temperature=inverse_temperature,
            N=steps,
            initialize_noise=initialize_noise,
            **kwargs,
        )

        if D_unc.shape != outs["C"].shape:
            D = torch.zeros_like(outs["C"]).long()
        else:
            D = D_unc

        assert D.shape == outs["C"].shape

        floods = [
            Flood.from_XCD(outs_X[None, ...], outs_C[None, ...], outs_D[None, ...])
            for outs_X, outs_C, outs_D in zip(outs["X_sample"], outs["C"], D)
        ]
        if samples == 1:
            floods = floods[0]

        if not full_output:
            return floods
        else:
            outs["S"] = S
            trajectories = self._format_trajectory(
                outs, "X_trajectory", trajectory_length
            )

            trajectories_Xhat = self._format_trajectory(
                outs, "Xhat_trajectory", trajectory_length
            )

            # use unconstrained C and D for Xunc_trajectory
            outs["D"] = D_unc
            outs["C"] = C_unc
            trajectories_Xunc = self._format_trajectory(
                outs, "Xunc_trajectory", trajectory_length
            )

            if samples == 1:
                full_output_dictionary = {
                    "trajectory": trajectories[0],
                    "Xhat_trajectory": trajectories_Xhat[0],
                    "Xunc_trajectory": trajectories_Xunc[0],
                }
            else:
                full_output_dictionary = {
                    "trajectory": trajectories,
                    "Xhat_trajectory": trajectories_Xhat,
                    "Xunc_trajectory": trajectories_Xunc,
                }

            return floods, full_output_dictionary

    def _format_trajectory(self, outs, key, trajectory_length):
        trajectories = [
            Flood.from_XCD_trajectory(
                [
                    outs_X[i][None, ...]
                    for outs_X in self._resample_trajectory(
                        trajectory_length, outs[key]
                    )
                ],
                outs_C[None, ...],
                outs_D[None, ...],
            )
            for i, (outs_C, outs_D) in enumerate(zip(outs["C"], outs["D"]))
        ]
        return trajectories

    def _resample_trajectory(self, trajectory_length, trajectory):
        if trajectory_length < 0:
            raise ValueError(
                "The trajectory length must fall on the interval [0, sample_steps]."
            )
        n = len(trajectory)
        trajectory_length = min(n, trajectory_length)
        idx = torch.linspace(0, n - 1, trajectory_length).long()
        return [trajectory[i] for i in idx]

    def design(
        self,
        flood: Flood,
        design_method: Literal["gnn", "autoregressive"] = "gnn",
        design_selection: Optional[Union[str, torch.Tensor]] = None,
        design_t: Optional[float] = 0.5,
        temperature_extent: float = 0.01,
        temperature_dynamics: float = 1e-3,
        top_p_extent: Optional[float] = None,
        verbose: bool = False,
    ) -> Flood:
        """
        Refines flood extent or propagates cascading infrastructure impacts using a learned model.

        Depending on `design_method`, this can:
            * Refine flood extents with autoregressive or diffusion-based spatial smoothing
            * Generate infrastructure interdependency graphs (e.g., cascading failures)

        Args:
            flood (Flood): The Flood object to refine or augment.
            design_method (str): Refinement method. 
                'gnn' = infrastructure impact propagation (default).
                'autoregressive' = pixel-wise flood class refinement.
            design_selection (str or Tensor, optional): Optional spatial mask or selector 
                that restricts the region to be refined or updated.
            design_t (float, optional): Diffusion timestep used to condition the design model. 
                Set to 0 for deterministic refinement; use >0 for uncertainty-aware generation.
            temperature_extent (float): Sampling temperature for class predictions (e.g., land/water/urban).
            temperature_dynamics (float): Sampling temperature for auxiliary fields (e.g., flow velocity).
            top_p_extent (float, optional): Optional top-p sampling cutoff for class predictions.
            verbose (bool): If True, prints debug or model status info.

        Returns:
            Flood: A refined Flood object with updated spatial fields or infrastructure impact states.
        """
        flood = copy.deepcopy(flood)
        flood.canonicalize()

        X, C, D = flood.to_XCD()

        # Optional refinement mask
        mask_sample = None
        if design_selection is not None:
            if isinstance(design_selection, str):
                design_selection = flood.get_mask(design_selection)
            mask_sample = design_selection

        # Run the refinement or propagation model
        X_sample, D_sample, _ = self.design_network.sample(
            X,
            C,
            D,
            t=design_t,
            mask_sample=mask_sample,
            temperature_extent=temperature_extent,
            temperature_dynamics=temperature_dynamics,
            top_p=top_p_extent,
            method=design_method,
            verbose=verbose,
        )

        flood.sys.update_with_XCD(X_sample, C=None, D=D_sample)
        return flood

    def pack(
        self, flood: Flood, temperature_dynamics: float = 1e-3, clamped: bool = False
    ) -> Flood:
        """
        Refines internal flood dynamics (e.g., velocity, flow direction) using the design network.

        This step performs constrained sampling or scoring of auxiliary dynamic fields
        associated with the flood extent, such as surface flow vectors, using a
        learned model.

        Args:
            flood (Flood): The Flood object to refine.
            temperature_dynamics (float): Sampling temperature for continuous dynamic fields.
                Lower values yield more deterministic refinement. Default is 1e-3.
            clamped (bool): If True, performs deterministic evaluation (e.g., log-likelihood)
                of the current flood dynamics without sampling. Used for diagnostics or scoring.

        Returns:
            Flood: A new Flood object with updated dynamic fields.
        """
        X, C, D = flood.to_XCD()

        X_repack, D_repack, _ = self.design_network.pack(
            X,
            C,
            D,
            temperature_dynamics=temperature_dynamics,
            clamped=clamped,
            return_scores=True,
        )

        flood.sys.update_with_XCD(X_repack, C=None, D=D_repack)

        return flood

    def score_backbone(
        self,
        floods: Union[List[Flood], Flood],
        num_samples: int = 50,
        tspan: List[float] = [1e-4, 1.0],
    ) -> Union[List[dict], dict]:
        """
        Scores generated flood scenarios using metrics that assess physical structure, 
        denoising consistency, and sample quality under the reverse diffusion process.

        Metrics may include:
            - elbo: Evidence lower bound estimate on log-likelihood
            - elbo_X: Component-wise elbo for the flood field
            - rmsd_ratio: Root-mean-squared deviation ratio over time
            - spatial_mse: Mean squared error in key spatial regions (e.g., urban, coastal)
            - smoothness_penalty: Difference-of-Gaussians penalty on sharp spatial transitions
            - coverage_entropy: Entropy of flooded vs. non-flooded area
            - structure_mismatch: Difference from physically plausible topographic constraints

        Args:
            floods (Flood or list of Flood): One or more Flood objects to score.
            num_samples (int): Number of intermediate timepoints to sample from the diffusion 
                trajectory to compute stochastic metrics. Default is 50.
            tspan (List[float]): The start and end time of the diffusion integration used for 
                scoring. Default is [1e-4, 1.0] (from near-deterministic to high-noise).

        Returns:
            dict or List[dict]: Dictionary of metrics for each flood object.
                Each metric is returned as a namedtuple with value and subcomponents.
        """
        device = next(self.parameters()).device
        if isinstance(floods, list):
            X, C, D = self._flood_list_to_XCD(floods, device=device)
        else:
            X, C, D = floods.to_XCD(device=device)

        # Estimate metrics using the flood backbone DDPM
        metrics, metrics_samples = self.backbone_network.estimate_metrics(
            X, C, return_samples=True, num_samples=num_samples, tspan=tspan
        )

        if isinstance(floods, list):
            metric_dictionary = [
                self._make_metric_dictionary(metrics, metrics_samples, idx=i)
                for i in range(len(floods))
            ]
        else:
            metric_dictionary = self._make_metric_dictionary(metrics, metrics_samples)

        return metric_dictionary

    def score_sequence(
        self,
        floods: Union[List[Flood], Flood],
        t: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Scores refined flood outputs using learned likelihoods and physically-inspired metrics 
        from the design network.

        Metrics may include:
            - negative log-likelihood (-log p) of flood class assignments (e.g., flooded/non-flooded/urban)
            - smoothness or entropy penalties for irregular transitions
            - mismatch in critical infrastructure coverage or logical consistency
            - deviation from expected hydrodynamic fields (e.g., pressure, velocity)

        Args:
            floods (Flood or list of Flood): One or more Flood objects to evaluate.
            t (torch.Tensor, optional): Diffusion timestep(s) corresponding to the input flood fields,
                used if the model was trained with diffusion noise. Shape = (batch_size,).
                Default is None (treat input as deterministic).

        Returns:
            dict or List[dict]: Dictionary of per-flood scores, returned as named tuples
                or flat key-value pairs depending on the scoring backend.
        """
        device = next(self.parameters()).device

        if isinstance(floods, list):
            X, C, D = self._flood_list_to_XCD(floods, high_res=True, device=device)
            output_scores = [{} for _ in range(len(floods))]
        else:
            X, C, D = floods.to_XCD(high_res=True, device=device)
            output_scores = {}

        # Compute model-based losses
        losses = self.design_network.loss(X, C, D, t=t, batched=False)

        for name, loss_tensor in losses.items():
            loss_list = [_t.squeeze() for _t in loss_tensor.split(1)]
            if isinstance(floods, list):
                for i, loss in enumerate(loss_list):
                    output_scores[i][name] = loss
            else:
                output_scores[name] = loss_list[0]

        return output_scores

    def _flood_list_to_XCD(self, list_of_floods, high_res=False, device=None):
        """
        Pads and batches a list of Flood objects with varying spatial dimensions.
        """

        # Extract tensors
        Xs, Cs, Ds = zip(*[flood.to_XCD(high_res=high_res) for flood in list_of_floods])

        # Determine max H, W for padding
        H_max = max(X.shape[-2] for X in Xs)
        W_max = max(X.shape[-1] for X in Xs)

        if device is None:
            device = Xs[0].device

        # Pad each tensor to (H_max, W_max)
        with torch.no_grad():
            X = torch.cat([
                nn.functional.pad(x, (0, W_max - x.shape[-1], 0, H_max - x.shape[-2])) 
                for x in Xs
            ])
            C = torch.cat([
                nn.functional.pad(c, (0, W_max - c.shape[-1], 0, H_max - c.shape[-2])) 
                for c in Cs
            ])
            D = torch.cat([
                nn.functional.pad(d, (0, 0, 0, W_max - d.shape[-2], 0, H_max - d.shape[-3]))
                if d.dim() == 4 else d  # safeguard
                for d in Ds
            ])

        return X.to(device), C.to(device), D.to(device)

    def score(
        self,
        floods: Union[List[Flood], Flood],
        num_samples: int = 50,
        tspan: List[float] = [1e-4, 1.0],
    ) -> Tuple[Union[List[dict], dict], dict]:
        """
        Computes unified score dictionary for each Flood object, combining:
            - backbone-based diffusion sampling metrics
            - design network likelihood or quality metrics

        Args:
            floods (Flood or List[Flood]): One or more Flood objects to evaluate.
            num_samples (int): Number of diffusion timepoints to sample for scoring. Default is 50.
            tspan (List[float]): Time range [t0, t1] for reverse diffusion scoring.

        Returns:
            Tuple:
                - Union[dict, List[dict]]: Combined per-flood score dictionary.
                - dict: Raw or additional output from the scoring backbones (if needed).
        """
        backbone_scores = self.score_backbone(floods, num_samples, tspan)
        sequence_scores = self.score_sequence(floods)
        if isinstance(floods, list):
            for ss in sequence_scores:
                ss["t_seq"] = ss.pop("t")
            return [bs | ss for bs, ss in zip(backbone_scores, sequence_scores)]
        else:
            sequence_scores["t_seq"] = sequence_scores.pop("t")
            return backbone_scores | sequence_scores

    def _make_metric_dictionary(self, metrics, metrics_samples, idx=None):
        """
        Process Metrics into a Single Dictionary

        Args:
            metrics (dict): Aggregated statistics from the DDPM backbone (mean, std, etc.).
            metrics_samples (dict): Per-sample metric time series from intermediate diffusion steps.
            idx (int, optional): If scoring a list of floods, this selects the index.

        Returns:
            dict: A dictionary of namedtuples with human-readable metrics.
        """
        
        metric_dictionary = {}
        for k, vs in metrics_samples.items():
            if k == "t":
                metric_dictionary["t"] = vs
            elif k in ["X", "X0_pred"]:
                if idx is None:
                    v = metrics[k]
                else:
                    vs = vs[idx]
                    v = metrics[k][idx]
                score = namedtuple(k, ["value", "samples"])
                metric_dictionary[k] = score(value=v, samples=vs)
            else:
                if idx is None:
                    v = metrics[k].item()
                else:
                    vs = vs[idx]
                    v = metrics[k][idx].item()
                vs = [i.item() for i in vs]
                score = namedtuple(k, ["score", "subcomponents"])
                metric_dictionary[k] = score(score=v, subcomponents=vs)

        return metric_dictionary

    def _init_backbones(self, num_fields: int, grid_shape: List[Tuple[int, int]]):
        """
        Initialize flood depth maps, land use classes, and metadata as tensors.

        Args:
            num_samples (int): Number of flood fields to generate.
            grid_shape (List[Tuple[int, int]]): Shape (H, W) of each flood grid.

        Returns:
            Tuple of tensors (X, C, D): depth, land class, metadata.
        """
        X = FloodBackbone(
            num_batch=num_fields,
            num_residues=sum(grid_shape),
            init_state="alpha",
        )()
        C = torch.cat(
            [torch.full([rep], i + 1) for i, rep in enumerate(grid_shape)]
        ).expand(X.shape[0], -1)
        D = torch.zeros_like(C)

        device = next(self.parameters()).device
        return [i.to(device) for i in [X, C, D]]