"""
Proximity operator for projecting vectors to elipse
"""

import torch

from .prox_op import ProxOp


class ProxOpElipse(ProxOp):
    """
    Proximity operator for projecting vectors to elipse (weighted l2 ball
    with preconditioning matrix).

    It uses the forward-backward algorithm to solve the projection.
    """

    def __init__(
        self,
        center: torch.Tensor,
        precond_weight: torch.Tensor,
        radius: float,
        itr_min: int = 1,
        itr_max: int = 10,
        rel_var_tol: float = 1e-6,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
        verbose: bool = False,
    ):
        """
        Initializes the Elipse proximity operator with the given parameters.

        Args:
            center (torch.Tensor): The center of the ellipse.
            precond_weight (torch.Tensor): The preconditioning weight.
            radius (float): The estimated l2 boundary between the target and center.
            itr_min (int, optional): The minimum number of iterations. Defaults to 1.
            itr_max (int, optional): The maximum number of iterations. Defaults to 10.
            rel_var_tol (float, optional): The relative variance tolerance. Defaults to 1e-6.
            device (torch.device, optional): The device on which the computations are
                performed. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The input and output data type.
                Defaults to torch.float.
            verbose (bool, optional): If True, print progress messages.
                Defaults to False.
        """
        super().__init__(device=device, dtype=dtype)

        self._center = center.to(device=self._device, dtype=self._dtype)
        self._precond_weight = precond_weight.to(device=self._device, dtype=self._dtype)
        self._radius = radius
        self._itr_min = itr_min
        self._itr_max = itr_max
        self._rel_var_tol = rel_var_tol
        self._verbose = verbose

        # initialisation
        self._prev = self._center
        self._gd_step_size = (
            1.0 / torch.max(torch.abs(self._precond_weight)).item() ** 2
        )

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input onto elipse based on forward-backward algorithm.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The projected tensor.
        """
        result = self._prev
        for i in range(self._itr_max):
            result = self._proj_l2_ball(
                result - self._gd_step_size * self._precond_weight * (result - x)
            )
            rel_var = torch.linalg.vector_norm(
                x - self._prev
            ) / torch.linalg.vector_norm(x)
            self._prev = result
            if i > self._itr_min and rel_var.item() < self._rel_var_tol:
                break
        return result

    @torch.no_grad()
    def _proj_l2_ball(self, x):
        """
        Project input onto l2 ball.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The projected tensor.
        """
        x -= self._center
        return self._center + x * torch.minimum(
            self._radius / torch.linalg.vector_norm(x),
            torch.ones(1, 1, device=self._device),
        )

    def update(self, *args, **kwargs):
        """
        No updates need for this prox operator.
        """

    def get_radius(self) -> float:
        """
        Return the radius of the l2 ball when preconditioning weight is not applied.

        Returns:
            float: The radius of the l2 ball.
        """
        return self._radius

    def get_precond_weight(self) -> torch.Tensor:
        """
        Return the preconditioning weight used for the proximity operator.

        Returns:
            torch.Tensor: The preconditioning weight.
        """
        return self._precond_weight
