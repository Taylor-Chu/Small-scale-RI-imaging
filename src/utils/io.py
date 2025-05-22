import gc
import glob
import os
import pathlib
from pathlib import Path

import h5py
import numpy as np
import torch
from astropy.io import fits
from scipy.constants import speed_of_light
from scipy.io import loadmat
from scipy.io.matlab import matfile_version
from torch.utils.data import Dataset


def load_data_to_tensor_holo(
    main_data_file: str,
    dirty_file_path: str,
    data_size: int,
    img_size: tuple,
    super_resolution: float = 1.5,
    data_path: str = None,
    dirac_peak: float = None,
    use_ROP: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu"),
    data: dict = None,
):
    """Read u, v and imweight from specified path.

    Parameters
    ----------
    uv_file_path : str
        Path to the file containing sampling pattern, natural weights and (optional) imaging weights.
    super_resolution : float
        Super resolution factor.
    image_pixel_size : float, optional
        Image pixel size in arcsec, by default None
    data_weighting : bool, optional
        Flag to apply imaging weights, by default True
    load_weight : bool, optional
        Flag to load imaging weights from the file, by default False. If set to False and data_weighting is True, the imaging weights will be generated.
    load_die : bool, optional
        Flag to load DIEs from the file, by default False
    weight_name : str, optional
        Name of the imaging weights in the data file, by default 'nWimag'
    dtype : torch.dtype, optional
        Data type to be used, by default torch.float64
    device : torch.device, optional
        Device to be used, by default torch.device('cpu')
    verbose : bool, optional
        Flag to print information, by default True

    Returns
    -------
    data: dict
        Dictionary containing u, v, w, (optional) y, nW, (optional) nWimag and other information.
    """

    if data is None:
        data = {}
    data_holo = {}
    mat_version, _ = matfile_version(main_data_file)
    if mat_version == 2:
        with h5py.File(main_data_file, "r") as h5File:
            for key, h5obj in h5File.items():
                if isinstance(h5obj, h5py.Dataset):
                    data_holo[key] = np.array(h5obj)
                    if data_holo[key].dtype.names and "imag" in data_holo[key].dtype.names:
                        data_holo[key] = data_holo[key]["real"] + 1j * data_holo[key]["imag"]
                elif isinstance(h5obj, h5py.Group):
                    data_holo[key] = {}
                    for key2, h5obj2 in h5obj.items():
                        data_holo[key][key2] = np.array(h5obj2)
                        if data_holo[key][key2].dtype.names and "imag" in data_holo[key][key2].dtype.names:
                            data_holo[key][key2] = (
                                data_holo[key][key2]["real"] + 1j * data_holo[key][key2]["imag"]
                            )
                else:
                    print("Type not implemented to be read here", h5obj)
    else:
        loadmat(main_data_file, mdict=data_holo)

    if not use_ROP:
        num_data = 0
        for i_f, f in enumerate(data_holo["freqs"].squeeze()):
            data_tmp = loadmat(
                os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"), variable_names=["data_I"]
            )
            num_data += data_tmp["data_I"].size

        data["u"] = np.zeros((1, 1, num_data), dtype=np.float64)
        data["v"] = np.zeros((1, 1, num_data), dtype=np.float64)
        data["y"] = np.zeros((1, 1, num_data), dtype=np.complex128)
        data["nW"] = np.zeros((1, 1, num_data), dtype=np.float64)
        counter = 0
        for i_f, f in enumerate(data_holo["freqs"].squeeze()):
            data_tmp = loadmat(os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"))
            new_counter = counter + data_tmp["data_I"].size
            data["u"][0, 0, counter:new_counter] = data_holo["uvw"][:, 0][data_tmp["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["v"][0, 0, counter:new_counter] = data_holo["uvw"][:, 1][data_tmp["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["y"][0, 0, counter:new_counter] = data_tmp["data_I"].squeeze()
            data["nW"][0, 0, counter:new_counter] = data_tmp["weightsNat"].squeeze()
            counter = new_counter

        max_proj_baseline = np.max(np.sqrt(data["u"] ** 2 + data["v"] ** 2))
        data["max_proj_baseline"] = max_proj_baseline
        spatial_bandwidth = 2 * max_proj_baseline
        image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
        print(
            f"INFO: default pixelsize: {image_pixel_size:.4e} arcsec, that is {super_resolution:.4f} x nominal resolution.",
            flush=True,
        )
        data["super_resolution"] = super_resolution

        data["u"] = torch.tensor(data["u"], dtype=dtype, device=device).view(1, 1, -1)
        data["v"] = -torch.tensor(data["v"], dtype=dtype, device=device).view(1, 1, -1)
        data["y"] = torch.tensor(data["y"], dtype=torch.complex128, device=device).view(1, 1, -1)
        data["nW"] = torch.tensor(data["nW"], dtype=torch.complex128, device=device).view(1, 1, -1)
        halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

        data["u"] = data["u"] * np.pi / halfSpatialBandwidth
        data["v"] = data["v"] * np.pi / halfSpatialBandwidth
        
    else:
        # prepare data for MROP
        num_data = 0
        for i_f, f in enumerate(data_holo["freqs"].squeeze()):
            data_tmp = loadmat(
                os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"), variable_names=["data_I"]
            )
            num_data += data_tmp["data_I"].size

        data["batches"] = data_holo["batches_flagged"]
        data["ant1"] = data_holo["ant1_flagged"]
        data["ant2"] = data_holo["ant2_flagged"]

        data["u"] = np.zeros((1, 1, num_data), dtype=np.float64)
        data["v"] = np.zeros((1, 1, num_data), dtype=np.float64)
        data["y"] = np.zeros((1, 1, num_data), dtype=np.complex128)
        data["nW"] = np.zeros((1, 1, num_data), dtype=np.float64)
        counter = 0
        for i_f, f in enumerate(data_holo["freqs"].squeeze()):
            data_ch_i_f = loadmat(os.path.join(data_path, f"273-X08_data_ch_{i_f+1}.mat"))
            new_counter = counter + data_ch_i_f["data_I"].size
            data["u"][0, 0, counter:new_counter] = data_holo["uvw"][:, 0][data_ch_i_f["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["v"][0, 0, counter:new_counter] = data_holo["uvw"][:, 1][data_ch_i_f["flag"].squeeze() == 1] / (
                speed_of_light / f
            )
            data["y"][0, 0, counter:new_counter] = data_ch_i_f["data_I"].squeeze()
            data["nW"][0, 0, counter:new_counter] = data_ch_i_f["weightsNat"].squeeze()
            counter = new_counter

        max_proj_baseline = np.max(np.sqrt(data["u"] ** 2 + data["v"] ** 2))
        data["max_proj_baseline"] = max_proj_baseline
        spatial_bandwidth = 2 * max_proj_baseline
        image_pixel_size = (180.0 / np.pi) * 3600.0 / (super_resolution * spatial_bandwidth)
        print(
            f"INFO: default pixelsize: {image_pixel_size:.4e} arcsec, that is {super_resolution:.4f} x nominal resolution.",
            flush=True,
        )
        data["super_resolution"] = super_resolution

        data["u"] = torch.tensor(data["u"], dtype=dtype, device=device).view(1, 1, -1)
        data["v"] = -torch.tensor(data["v"], dtype=dtype, device=device).view(1, 1, -1)
        data["y"] = torch.tensor(data["y"], dtype=torch.complex128, device=device).view(1, 1, -1)
        data["nW"] = torch.tensor(data["nW"], dtype=torch.complex128, device=device).view(1, 1, -1)
        halfSpatialBandwidth = (180.0 / np.pi) * 3600.0 / (image_pixel_size) / 2.0

        data["u"] = data["u"] * np.pi / halfSpatialBandwidth
        data["v"] = data["v"] * np.pi / halfSpatialBandwidth

    del data_holo  # , tmp, uniques, counts
    gc.collect()

    return data
