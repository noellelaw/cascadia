"""XCD represents flood structure as a tuple of PyTorch tensors.

The tensors in an XCD representation are:

    `X` (FloatTensor), the Cartesian coordinates or raster grid values
        representing the flood surface with shape `(num_batch, H, W, num_channels)`.

    `C` (LongTensor), the region mask or scenario map encoding per-grid cell
        chain assignments or flood scenario regions with shape `(num_batch, H, W)`.

    `D` (LongTensor), optional land cover classes (e.g., NLCD indices) with
        shape `(num_batch, H, W)`, used for conditioning or auxiliary supervision.
"""

from functools import partial, wraps
from inspect import getfullargspec

import torch
from torch.nn import functional as F

def validate_XCD(high_res=None, sequence=True):
    """Decorator factory that adds XCD validation to any function.

    Args:
        high_res (int, optional): If set, checks that the X tensor has this many channels.
        sequence (bool, optional): If True, ensures S and O match or are inferred from each other.
    """

    def decorator(func):
        @wraps(func)
        def new_func(*args, **kwargs):
            args = list(args)
            arg_list = getfullargspec(func)[0]
            tensors = {}
            for var in ["X", "C", "D", "O"]:
                try:
                    if var in kwargs:
                        tensors[var] = kwargs[var]
                    else:
                        tensors[var] = args[arg_list.index(var)]
                except IndexError:
                    tensors[var] = None
                except ValueError:
                    if not sequence and var in ["D", "O"]:
                        pass
                    else:
                        raise Exception(
                            f"Variable {var} is required by validation but not defined!"
                        )

            if tensors["X"] is not None and tensors["C"] is not None:
                if tensors["X"].shape[:3] != tensors["C"].shape:
                    raise ValueError(
                        f"X shape {tensors['X'].shape} does not match C shape {tensors['C'].shape}"
                    )
            if high_res is not None and tensors["X"] is not None:
                if tensors["X"].shape[-1] != high_res:
                    raise ValueError(f"Expected {high_res} channels but got {tensors['X'].shape[-1]}")

            if sequence and (tensors["D"] is not None or tensors["O"] is not None):
                if tensors["O"] is None:
                    if "O" in kwargs:
                        kwargs["O"] = F.one_hot(tensors["D"], num_classes=16).float()
                    else:
                        args[arg_list.index("O")] = F.one_hot(tensors["D"], num_classes=16).float()
                elif tensors["D"] is None:
                    if "D" in kwargs:
                        kwargs["D"] = tensors["O"].argmax(dim=-1)
                    else:
                        args[arg_list.index("D")] = tensors["O"].argmax(dim=-1)
                else:
                    if not torch.allclose(tensors["O"].argmax(dim=-1), tensors["D"]):
                        raise ValueError("D and O are both provided but don't match!")

            return func(*args, **kwargs)

        return new_func

    return decorator

validate_XC = partial(validate_XCD, sequence=False)
