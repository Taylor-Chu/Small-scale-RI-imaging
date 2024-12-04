"""
Functions for loading measurement data from data file
"""

from typing import Optional, Tuple, Literal
import numpy as np
import torch
import scipy.io as sio
import h5py

from .imaging_weight import gen_imaging_weight


def load_mat_data_file_2_tensor_ri(
    file_path: str,
    sr_factor: Optional[float] = None,
    im_pixel_size: Optional[float] = None,
    data_weighting_flag: bool = False,
    load_weight_flag: bool = True,
    img_size: Tuple[int, int] = (512, 512),
    weight_type: Literal["uniform", "briggs", "robust"] = "uniform",
    grid_size: int = 1,
    robustness: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> dict:
    """
    Load RI measurement data from .mat file to torch tensors.

    Args:
        file_path (str): The path to the .mat file.
        sr_factor (float, optional): Super-resolution factor. Defaults to None.
        im_pixel_size (float, optional): Image pixel size. Defaults to None.
        data_weighting_flag (bool, optional): Flag to apply data weighting. Defaults to False.
        load_weight_flag (bool, optional): Flag to load weight. Defaults to True.
        img_size (Tuple[int, int], optional): Image size. Defaults to (512, 512).
        weight_type (Literal["uniform", "briggs", "robust"], optional): Type of weight.
            Defaults to "uniform".
        grid_size (int, optional): Grid size. Defaults to 1.
        robustness (float, optional): Robustness parameter. Defaults to 0.0.
        dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.float64.
        device (torch.device, optional): Device to store the tensors.
            Defaults to torch.device("cpu").
        verbose (bool, optional): Flag to print verbose messages. Defaults to True.

    Returns:
        dict: A dictionary containing the loaded data.
    """
    mat_file_ver, _ = sio.matlab.matfile_version(file_path)
    if mat_file_ver == 2:
        data = {}
        with h5py.File(file_path, "r") as h5_file:
            for key, h5obj in h5_file.items():
                if isinstance(h5obj, h5py.Dataset):
                    data[key] = np.array(h5obj)
                    if data[key].dtype.names and "imag" in data[key].dtype.names:
                        data[key] = data[key]["real"] + 1j * data[key]["imag"]
                    if not data[key].any():
                        data[key] = np.array([])
    else:
        data = sio.loadmat(file_path)

    # set defaut values for missing fields
    if "maxProjBaseline" not in data:
        data["maxProjBaseline"] = np.sqrt(
            np.max(data["u"].flatten() ** 2 + data["v"].flatten() ** 2)
        ).item()
    if "w" not in data:
        data["w"] = np.zeros(data["u"].shape)

    # calculate pixel size and half spatial bandwidth
    if not im_pixel_size:
        spatial_bandwidth = 2 * data["maxProjBaseline"].item()
        im_pixel_size = (180.0 / np.pi) * 3600.0 / (sr_factor * spatial_bandwidth)
        if verbose:
            print(
                f"INFO: default pixelsize: {im_pixel_size} arcsec, "
                f"that is {sr_factor} x nominal resolution."
            )
    elif verbose:
        print(f"INFO: user specified pixelsize: {im_pixel_size} arcsec.")
    half_spatial_bandwidth = (180.0 / np.pi) * 3600.0 / (im_pixel_size) / 2.0

    # u v will be normalised between [-pi,pi] for the NUFFT
    data["u"] = data["u"].astype(np.double) * np.pi / half_spatial_bandwidth
    data["v"] = -data["v"].astype(np.double) * np.pi / half_spatial_bandwidth
    data["w"] = -data["w"]

    # weighting
    if data_weighting_flag and not load_weight_flag:
        data["nWimag"] = gen_imaging_weight(
            data["u"],
            data["v"],
            img_size,
            weight_type=weight_type,
            natural_weight=data["nW"],
            grid_size=grid_size,
            robustness=robustness,
        )
        if verbose:
            print("INFO: calculate image weights")
    elif not data_weighting_flag or data["nWimag"].size == 0:
        data["nWimag"] = np.ones((1, 1), dtype=np.double)
        if verbose:
            print("INFO: imaging weight will not be applied.")
    elif verbose:
        print("INFO: load imaging weight from file.")

    # move data to tensor
    data["u"] = torch.tensor(data["u"], device=device, dtype=dtype).view(1, 1, -1)
    data["v"] = torch.tensor(data["v"], device=device, dtype=dtype).view(1, 1, -1)
    data["w"] = torch.tensor(data["w"], device=device, dtype=dtype).view(1, 1, -1)
    data["nW"] = torch.tensor(data["nW"], device=device, dtype=dtype).view(1, 1, -1)
    if dtype == torch.float32:
        data["nWimag"] = torch.tensor(
            data["nWimag"], device=device, dtype=torch.complex64
        ).view(1, 1, -1)
        data["y"] = torch.tensor(data["y"], device=device, dtype=torch.complex64).view(
            1, 1, -1
        )
    else:
        data["nWimag"] = torch.tensor(
            data["nWimag"], device=device, dtype=torch.complex128
        ).view(1, 1, -1)
        data["y"] = torch.tensor(data["y"], device=device, dtype=torch.complex128).view(
            1, 1, -1
        )

    # apply weighting to measurements
    data["y"] *= data["nW"] * data["nWimag"]

    return data
