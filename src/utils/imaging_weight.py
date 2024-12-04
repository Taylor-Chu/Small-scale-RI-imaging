"""
Functions for generating imaging weights
"""

from typing import Literal, Tuple, Optional
import numpy as np


def gen_imaging_weight(
    u: np.ndarray,
    v: np.ndarray,
    img_size: Tuple[int, int],
    weight_type: Literal["uniform", "briggs", "robust"] = "uniform",
    natural_weight: Optional[np.ndarray] = None,
    grid_size: int = 1,
    robustness: float = 0.0,
):
    """
    Generate uniform weights or Briggs (robust) weights for RI measurements based on
    the uv sampling pattern.

    Args:
        u (np.ndarray): The u coordinates of the sampling pattern.
        v (np.ndarray): The v coordinates of the sampling pattern.
        img_size (Tuple[int, int]): The size of the image to be reconstructed.
        weight_type (Literal["uniform", "briggs", "robust"], optional):
            The type of weights to be generated. Can be "uniform" or "briggs".
            Defaults to "uniform".
        natural_weight (np.ndarray, optional): The natural weights for the uv sampling pattern.
            Defaults to None.
        grid_size (int, optional): The size of the grid for the uv sampling pattern.
            Defaults to 1.
        robustness (float, optional): The robustness factor for the Briggs weighting.
            Defaults to 0.0.

    Returns:
        np.ndarray: The generated weights for the uv sampling pattern.

    Raises:
        NotImplementedError: If the weight_type is not "uniform" or "briggs".
    """
    # flatting u & v vector
    u = u.reshape((-1, 1)).astype(np.double)
    v = v.reshape((-1, 1)).astype(np.double)

    # consider only half of the plane
    u[v < 0] = -u[v < 0]
    v[v < 0] = -v[v < 0]

    # Initialize
    nmeas = u.size
    weight_grid_size = np.floor(
        np.array((img_size[0] * grid_size, img_size[1] * grid_size))
    )
    gridded_weight = np.zeros(
        weight_grid_size.astype(int), dtype=np.double
    )  # Initialize gridded weights matrix with zeros
    image_weight = np.ones((nmeas, 1))

    # grid uv points
    q = np.floor((u + np.pi) * weight_grid_size[1] / 2.0 / np.pi).astype(int) - 1
    p = (
        np.floor((v + np.pi) * weight_grid_size[0] / 2.0 / np.pi).astype(int) - 1
    )  # matching index in matlab

    if weight_type == "uniform":
        for idx in range(nmeas):
            gridded_weight[p[idx], q[idx]] += 1.0
        # Apply weighting
        image_weight = 1.0 / np.sqrt(gridded_weight[p, q])
    elif weight_type in ["robust", "briggs"]:
        # inverse of the noise variance
        natural_weight = natural_weight.reshape((-1, 1)).astype(np.double)
        natural_weight2 = natural_weight**2
        if natural_weight2.size == 1:
            natural_weight2 = natural_weight2[0] * np.ones((nmeas, 1))
        for idx in range(nmeas):
            gridded_weight[p[idx], q[idx]] += natural_weight2[idx]
        # Compute robust scale factor
        robust_scale = (np.sum(gridded_weight) / np.sum(gridded_weight**2)) * (
            5 * 10 ** (-robustness)
        ) ** 2
        # Apply weighting
        image_weight = 1.0 / np.sqrt(1.0 + robust_scale * gridded_weight[p, q])
    else:
        raise NotImplementedError("Image weighting type: " + weight_type)

    return image_weight
