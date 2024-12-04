# Small scale RI imaging
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![license](https://img.shields.io/badge/license-GPL--3.0-brightgreen.svg)](LICENSE)

- [Small scale RI imaging](#small-scale-ri-imaging)
  - [Description](#description)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
    - [Cloning the project](#cloning-the-project)
    - [Updating submodules (optional)](#updating-submodules-optional)
    - [Pretrained AIRI denoisers](#pretrained-airi-denoisers)
    - [Python environment](#python-environment)
  - [Input Files](#input-files)
    - [Measurement file](#measurement-file)
    - [Configuration file](#configuration-parameter-file)
  - [Usage and Examples](#usage-and-examples)

## Description

This repository provides pure Python implementation of ``uSARA``, ``AIRI`` and ``cAIRI`` for small scale RI imaging problem which focuses on monochromatic intensity and a narrow filed of view.

``uSARA``[1,2] is the unconstrained counterpart of the ``SARA`` algorithm. It is underpinned by the forward-backward algorithmic structure for solving inverse imaging problem and uses a handcrafted sparsity-based image model.

``AIRI``[2] and its constraint variant ``cAIRI``[3] are Plug-and-Play (PnP) algorithms used to solve the inverse imaging problem. By inserting carefully trained AIRI denoisers into the proximal splitting algorithms, one waives the computational complexity of optimisation algorithms induced by sophisticated image priors, and the sub-optimality of handcrafted priors compared to Deep Neural Networks.

The details of these algorithms are discussed in the following papers.

>[1] Repetti, A., & Wiaux, Y., [A forward-backward algorithm for reweighted procedures: Application to radio-astronomical imaging](https://doi.org/10.1109/ICASSP40776.2020.9053284). *IEEE ICASSP 2020*, 1434-1438, 2020.
>
>[2] Terris, M., Dabbech, A., Tang, C., & Wiaux, Y., [Image reconstruction algorithms in radio interferometry: From handcrafted to learned regularization denoisers](https://doi.org/10.1093/mnras/stac2672). *MNRAS, 518*(1), 604-622, 2023.
>
>[3] Terris, M., Tang, C., Jackson, A., & Wiaux, Y., [The AIRI plug-and-play algorithm for image reconstruction in radio-interferometry: variations and robustness](https://arxiv.org/abs/2312.07137v3), 2024, *preprint arXiv:2312.07137v3.* 

We also point the readers to [AIRI](https://github.com/basp-group/AIRI) and [uSARA](https://github.com/basp-group/uSARA) repositories for their MATLAB implementations.

## Dependencies 

This repository relies on two packages:

1. [`RI Measurement Operator`](https://github.com/basp-group/RI-measurement-operator/tree/python) for the python implementation of radio-interferometric measurement operator;
2. [`PyTorch Wavelet Toolbox`](https://github.com/v0lta/PyTorch-Wavelet-Toolbox) for SARA wavelet dictionary.

These packages associate with the following publications


>[4] Wolter, M., Blanke, F., Garcke, J., & Hoyt, C. T., [ptwt-The PyTorch Wavelet Toolbox](https://www.jmlr.org/papers/v25/23-0636.html). *JLMR*, 25(80), 1-7. 2024.

## Installation

### Cloning the project

To clone the project with the required submodules, you may consider one of the following set of instructions.

- Cloning the project using `https`: you should run the following command
```bash
git clone --recurse-submodules https://github.com/basp-group/Small-scale-RI-imaging.git
```
- Cloning the project using SSH key for GitHub: you should run the following command
```bash
git clone git@github.com:basp-group/Small-scale-RI-imaging.git
```

Next, edit the `.gitmodules` file, replacing the `https` addresses with the `git@github.com` counterpart as follows: 

```bash
[submodule "src/ri_measurement_operator"]
	path = src/ri_measurement_operator
	url = git@github.com/basp-group/RI-measurement-operator.git
	branch = python
```

Finally, follow the instructions in the next session [Updating submodules (optional)](#updating-submodules-optional) to clone the submodule into the repository's path.

The full path to the Small Scale RI Imaging repository is referred to as `$SSRI` in the rest of the documentation.

### Updating submodules (optional)

To update the submodules from your local `$SSRI` repository, run the following commands: 

```bash
git pull
git submodule sync --recursive          # update submodule address, in case the url has changed
git submodule update --init --recursive # update the content of the submodules
git submodule update --remote --merge   # fetch and merge latest state of the submodule
```

###  Pretrained AIRI denoisers
If you'd like to use our trained AIRI denoisers, you can find the ONNX files on [Heriot-Watt Research Portal](https://doi.org/10.17861/aa1f43ee-2950-4fce-9140-5ace995893b0). You should download `v1_airi_astro-based_oaid_shelf.zip` and `v1_airi_mri-based_mrid_shelf.zip`, then copy the unzipped folders to ``$SSRI/airi_denoisers/`` folder of this repository. Alternatively, make sure to update the full paths to the DNNs in the `.csv` file of the denoiser shelf.

### Python environment
We recommend to start by creating virtual environment with python version higher than 3.11 using management tools such as ``conda`` or ``venv``. Users should then refer to [PyTorch official website](https://pytorch.org) for the proper command to install the PyTorch version that fits your system. Finally, install other required packages using the following command:

```bash
pip install -r requirements.txt
```

## Input Files
### Measurement file
The current code takes as input data a measurement file in ``.mat`` format containing the following fields:

```python 
  "y"               # vector; data (Stokes I)
  "u"               # vector; u coordinate (in units of the wavelength)
  "v"               # vector; v coordinate (in units of the wavelength)
  "w"               # vector; w coordinate (in units of the wavelength)
  "nW"              # vector; inverse of the noise standard deviation 
  "nWimag"          # vector; square root of the imaging weights if available (Briggs or uniform), empty otherwise
  "frequency"       # scalar; observation frequency
  "maxProjBaseline" # scalar; maximum projected baseline (in units of the wavelength; formally  max(sqrt(u.^2+v.^2)))
```

An example measurement file ``3c353_meas_dt_1_seed_0.mat`` is provided in the folder ``$SSRI/data``. The full synthetic test set used in [1] can be found in this (temporary) [Dropbox link](https://www.dropbox.com/scl/fo/et0o4jl0d9twskrshdd7j/h?rlkey=gyl3fj3y7ca1tmoa1gav71kgg&dl=0).

To extract the measurement file from Measurement Set Tables (MS), you can use the utility Python script `$SSRI/ms2mat/ms2mat.py`. Instructions are provided in the [Readme File](https://github.com/basp-group/Small-scale-RI-imaging/blob/main/ms2mat/README.md).


### Configuration (parameter) file
The configuration file is a ``.json`` format file comprising all parameters to run the different algorithms. A template file is provided in `$SSRI/config/`. An example `example.json` is provided in `$SSRI/config/`. A detailed description about the fields in the configuration file is provided [here](https://github.com/basp-group/Small-scale-RI-imaging/blob/main/config/README.md).

## Usage and Examples
The algorithms can be launched through script `run_imager.py`. It uses the configuration file ``$SSRI/config/example.json``. 

```BASH
python run_imager.py
```

It also accepts 12 optional name-argument pairs which will overwrite corresponding fields in the configuration file.

```bash
python run_imager.py \
    -c $PTH_CONFIG \                    # path of the configuration file
    --src_name $SRC_NAME \               # name of the target src used for output filenames
    --data_file $DATA_FILE \             # path of the measurement file
    --result_path  $RESULT_PATH \        # path where the result folder will be created
    --algorithm  $ALGORITHM \            # algorithm that will be used for imaging
    --im_dim_x  $IM_DIM_X \              # horizontal number of pixels in the final reconstructed image
    --im_dim_y  $IM_DIM_Y \              # vertical number of pixels in the final reconstructed image
    --dnn_shelf_path  $DNN_SHELF_PATH \  # path of the denoiser shelf configuration file
    --im_pixel_size  $IM_PIXEL_SIZE \    # pixel size of the reconstructed image in the unit of arcsec
    --superresolution  $SUPERRE \        # used if pixel size not provided 
    --groundtruth  $GROUNDTRUTH \        # path of the groundtruth image when available
    --run_id  $RUN_ID                    # identification number of the imaging run used for output filenames
```

Detailed instructions for how to launch `uSARA` and `AIRI` are provided in [tutorial_airi_python.ipynb](./tutorial_airi_python.ipynb) and [tutorial_usara_python.ipynb](./tutorial_usara_python.ipynb). The scripts in these two Jupyter notebooks will reconstruct the groundtruth image `$SSRI/data/3c353_gdth.fits` from the measurement file `$SSRI/data/3c353_meas_dt_1_seed_0.mat`. The results will be saved in the folder `$SSRI/results/`.
