"""
Proximity operator for positivity
"""

import torch

from .prox_op import ProxOp


class ProxOpPositivity(ProxOp):
    """
    Proximity operator for positivity.

    This class implements a proximity operator for positivity. It provides a method
    for projecting onto the positivity domain, which is equivalent to applying the
    ReLU function.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: bool = True,
    ) -> None:
        """
        Initializes the Positivity proximity operator with the given parameters.

        Args:
            device (torch.device, optional): The device on which the computations are
                performed. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type of the input.
                Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages.
                Defaults to True.
        """
        super().__init__(device=device, dtype=dtype)
        self._verbose = verbose

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input onto the positivity domain, same as ReLU.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The projected tensor.
        """
        return torch.nn.functional.relu(x)

    def update(self, *args, **kwargs) -> None:
        """
        No updates need for this prox operator.
        """
