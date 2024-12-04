"""
AIRI proximity operator
"""

import csv
import os
import torch
from onnx2torch import convert

from .prox_op import ProxOp

# TODO: Add faceting functionality


class ProxOpAIRI(ProxOp):
    """
    AIRI proximity operator

    This class implements the AIRI proximity operator which uses AIRI denoisers for regularisations.
    It uses a shelf of pre-trained AIRI denoisers and selects the appropriate one based on
    the estimated maximum intensitie of target image and the heuristic noise level.
    """

    def __init__(
        self,
        shelf_path: str,
        rand_trans: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the AIRI proximity operator with the given parameters.

        Args:
            shelf_path (str): Path to the CSV file containing the denoiser shelf.
            rand_trans (bool, optional): If True, apply random transformations to the input
                before applying denoisers. Defaults to True.
            device (torch.device, optional): The device on which the computations are
                performed. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input. Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages. Defaults to True.
        """
        super().__init__(device=device, dtype=dtype)

        self._rand_trans = rand_trans
        self._net_scaling = 1.0
        self._shelf = {}
        self._network = None
        self._verbose = verbose

        # load paths of networks
        if not os.path.isfile(shelf_path):
            raise FileNotFoundError("Shelf file not found: " + shelf_path)
        with open(shelf_path, newline="", encoding="utf-8") as shelf_file:
            shelf_reader = csv.reader(shelf_file, delimiter=",")
            for row in shelf_reader:
                self._shelf[float(row[0])] = row[1]
                if not os.path.isfile(row[1]):
                    raise FileNotFoundError("Denoiser file not found: " + row[1])
        if not self._shelf:
            raise RuntimeError("Shelf is empty: " + shelf_path)
        self._shelf = dict(sorted(self._shelf.items()))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the AIRI prox operator to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The denoised tensor.
        """
        if self._rand_trans:
            flip_dim = ([], (2,), (3,), (2, 3))[torch.randint(4, (1,)).item()]
            rot_times = torch.randint(4, (1,)).item()
            x = torch.flip(x, dims=flip_dim)
            x = torch.rot90(x, k=rot_times, dims=[2, 3])
            x = x.contiguous()
        result = self._network(x / self._net_scaling) * self._net_scaling
        if self._rand_trans:
            result = torch.rot90(result, k=-rot_times, dims=[2, 3])
            result = torch.flip(result, dims=flip_dim)
            result = result.contiguous()

        return result

    def update(self, heuristic: float, peak_est: float) -> None:
        """
        Updates the denoiser selection and scaling factor based on the given
        heuristic noise level and maximum intensity.

        Args:
            heuristic (float): The heuristic noise level.
            peak_est (float): The estimated maximum intensity.
        """
        peak_min = 0
        peak_max = 0

        sigma_s = max(
            filter(lambda i: i <= heuristic / peak_est, self._shelf.keys()),
            default=None,
        )
        if sigma_s:
            peak_max = heuristic / sigma_s
            sigma_s1 = min(
                filter(lambda i: i > sigma_s, self._shelf.keys()), default=None
            )
            if sigma_s1:
                peak_min = heuristic / sigma_s1
        else:
            sigma_s = min(self._shelf.keys())
            peak_min = heuristic / sigma_s
            peak_max = float("inf")

        self._network = convert(self._shelf[sigma_s]).to(self._device).eval()
        self._net_scaling = heuristic / sigma_s

        if self._verbose:
            print(
                f"\nSHELF *** Inverse of the estimated target dynamic range: {heuristic/peak_est}",
                flush=True,
            )
            print(f"SHELF *** Using network: {self._shelf[sigma_s]}", flush=True)
            print(
                f"SHELF *** Peak value is expected in range: [{peak_min}, {peak_max}]",
                flush=True,
            )
            print(
                f"SHELF *** scaling factor applied to the image: {self._net_scaling}",
                flush=True,
            )

        return (peak_min, peak_max)
