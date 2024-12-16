# Configuration (parameter) file

The algorithms implemented in this repository are launched through the script ``run_imager.py``. This function accepts a ``.json`` file where all the parameters required by the algorithms are defined. A sample configuration file ``example.json`` is given in the folder ``$SSRI/config``. In this document, we'll provide explanations for all the fields in this file.

The configuration file is composed by three parts, i.e. Main, General and Denoiser. 

1. Main
    - ``src_name``(optional): Experiment/target source name tag, used in the output filenames. If this field is empty, the script will take the filename given in the ``data_file``.
    - ``data_file``: Path to the measurement (data) file. The measurement file must be in ``.mat`` format containing fields discussed [here](https://github.com/basp-group-private/Small-scale-RI-imaging?tab=readme-ov-file#measurement-file).
    - ``result_path``(optional): Path to the output files. The script will create a folder in ``$result_path`` with name ``${src_name}_${algorithm}_ID_${run_id}_heuScale_${heu_noise_scale}_maxItr_${im_max_itr}``. The results will then be saved in this folder. Default: ``$SSRI/results``.
    - ``algorithm``: Imaging algorithm, must be set to ``usara``, ``airi`` or ``cairi``.
    - ``im_dim_x``: Horizontal dimension of the estimated image.
    - ``im_dim_y``: Vertical dimension of the estimated image.
    - ``im_pixel_size``(optional): Pixel size of the estimated image in the unit of arcsec. If empty, its value is inferred from ``superresolution`` such that ``imPixelSize = (180 / pi) * 3600 / (superresolution * 2 * maxProjBaseline)``.
    - ``superresolution``(optional): Imaging super-resolution factor, used when the pixel size is not provided (recommended to be in ``[1.5, 2.5]``). Default: ``1.0``.
    - ``groundtruth``(optional): Path of the groundtruth image. The file must be in ``.fits `` format, and is used to compute reconstruction metrics if a valid path is provided.
    - ``run_id``(optional): Identification number of the current task.

    The values of the entries in Main will be overwritten if corresponding name-value arguments are fed into ``run_imager.py``.

2. General
    - ``flag``
        - ``flag_imaging``(optional): Enable imaging. If ``false``, the back-projected data (dirty image) and corresponding PSF are generated. Default: ``true``.
        - ``flag_data_weighting``(optional): Enable data-weighting scheme. Default: ``true``.
        - ``verbose``(optional): Enable verbose mode. Default: ``true``.

    - ``weighting``
        - ``weight_load``(optional): Flag to indicate whether reads imaging weights from the data file if ``flag_data_weighting`` is ``true``. If the field ``nWimag`` in ``data_file`` is empty, then the image weight will be set to ``1.0``. Default: ``true``.
        - ``weight_type``(optional): The data-weighting scheme ``["briggs" | "robust", "uniform"]``, if ``flag_data_weighting`` is ``true`` and ``weight_load`` is ``false``. Default: ``briggs``.
        - ``weight_robustness``(optional): Briggs (robust) parameter to be set in ``[-2, 2]``. Default: ``0``.
        - ``weight_gridsize``(optional): Padding factor involved in the density of the sampling. Default: ``2``.

    - ``computing``
        - ``ncpus``(optional): Number of CPUs used for imaging task. If empty, the script will make use of the available CPUs.
        - ``meas_device``(optional): The device that the measurement operators run on. The supported list of devices is ``["cuda", "cpu"]``. If this filed is empty, then the first available device in this list will be chosen.
        - ``meas_dtype``(optional): The data type of the target image that the measurement operators will accept and give. The list of possible options is ``["float" | "float32" | "single", "float64", "double"]``. Please note that some ``cuda`` devices do not support double precision and the scripts may yield runtime error in this case. Default:``single``.
        - ``prox_device``(optional): The device that the proximity operators run on. The supported list of devices is ``["cuda", "mps", "cpu"]``. If this filed is empty, then the first available device in this list will be chosen.
        - ``prox_dtype``(optional): The data type that the proximity operators accept and give. The list of possible options is ``["float" | "float32" | "single", "float64", "double"]``. Please note that ``mps`` and some ``cuda`` devices do not support double precision and the scripts may yield runtime error in this case. Default:``single``.

    - ``nufft``
        - ``nufft_package``: The NUFFT package used in the imaging task. The supported list of packages is ``["finufft","pynufft","tkbnufft"]``. Default:``finufft``. Please check [FINUFFT](https://flatironinstitute.github.io/pytorch-finufft/), [PyNUFFT](https://pynufft.readthedocs.io/en/latest/), and [TorchKbNUFFT](https://torchkbnufft.readthedocs.io/en/stable/) for the details of the packages.
        - ``nufft_mode``: TorchKbNUFFT supports different NUFFT [interpolation mode](https://github.com/mmuckley/torchkbnufft?tab=readme-ov-file#operation-modes-and-stages). It can be chosen from ``table`` and ``matrix``. The ``table`` mode is faster but claimed to be slightly less accurate. Default:``table``.  

3. Denoiser
    - ``usara`` and ``usara_default``

        If the imaging ``algorithm``is specified as ``usara``, then the fields in the section will be loaded.

        - ``heu_reg_param_scale``(optional): Adjusting factor applied to the regularisation parameter calculated based on the heuristic noise levels. Default: ``1.0``.
        - ``reweighting``: Enable reweighting algorithm.  Default: ``true``.
        - ``approx_meas_op``(optional): Use approximate measurement operator during reconstruction. Default: ``false``.
        - ``im_min_itr``(optional): Minimum number of iterations in the forward-backward algorithm (inner loop). Default: ``10``.
        - ``im_max_itr``(optional): Maximum number of iterations in the forward-backward algorithm (inner loop). Default: ``2000``.
        - ``im_var_tol``(optional): Tolerance on the relative variation of the estimation in the forward-backward algorithm (inner loop) to indicate convergence. Default: ``1e-4``.
        - ``im_max_outer_itr``(optional): Maximum number of iterations in the reweighting algorithm (outer loop).  Default: ``10``.
        - ``im_var_outer_tol``(optional): Tolerance on the relative variation of the estimation in the reweighting algorithm (outer loop) to indicate convergence. Default: ``1e-4``.
        - ``itr_save``(optional): Interval of iterations for saving intermediate results. Default: ``500``.

    - ``airi`` and ``airi_default``

        If the imaging ``algorithm``is specified as ``airi``, then the fields in the section will be loaded.

        - ``heu_noise_scale``(optional): Adjusting factor applied to the heuristic noise level. Default: ``1.0``.
        - ``dnn_shelf_path``: Path of the ``.csv`` file that defines a shelf of denoisers. The ``.csv`` file has two columns. The first column is the training noise level of a denoiser and the second column is the path to the denoiser. The denoiser must be in ``.onnx`` format. Two sample ``.csv`` files are provided in ``$SSRI/airi_denoisers``.
        - ``im_peak_est``(optional): Estimated maximum intensity of the true image. If this field is empty, the default value is the maximum intensity of the back-projected (dirty) image normalised by the peak value of the PSF.
        - ``dnn_adaptive_peak``(optional): Enable the adaptive denoiser selection scheme. The details of this scheme can be found in [[1]](https://arxiv.org/abs/2312.07137v2). Default: ``true``.
        - ``dnn_apply_transform``(optional): Apply random rotation and flipping before denoising then undo the transform after denoising. Default: ``true``.
        - ``approx_meas_op``(optional): Use approximate measurement operator during reconstruction. Default: ``false``.
        - ``im_min_itr``(optional): Minimum number of iterations in the forward-backward algorithm. Default: ``200``.
        - ``im_max_itr``(optional): Maximum number of iterations in the forward-backward algorithm. Default: ``2000``.
        - ``im_var_tol``(optional): Tolerance on the relative variation of the estimation in the forward-backward algorithm to indicate convergence. Default: ``1e-4``.
        - ``itr_save``(optional): Interval of iterations for saving intermediate results. Default: ``500``.
        - ``dnn_adaptive_peak_tol_max``(optional): Initial relative peak value variation tolerance for the adaptive denoiser selection scheme. Default: ``0.1``.
        - ``dnn_adaptive_peak_tol_min``(optional): Minimum relative peak value variation tolerance for the adaptive denoiser selection scheme. Default: ``1e-3``.
        - ``dnn_adaptive_peak_tol_step``(optional): Decaying factor for the relative peak value variation tolerance in the adaptive denoiser selection scheme. It will be applied to the current tolerance after one time of denoiser reselection. Default: ``0.1``.

    - ``cairi`` and ``cairi_default``

        If the imaging ``algorithm``is specified as ``cairi``, then the fields in the section will be loaded.

        - ``heu_noise_scale``(optional): Adjusting factor applied to the heuristic noise level. Default: ``1.0``.
        - ``dnn_shelf_path``: Path of the ``.csv`` file that defines a shelf of denoisers. The ``.csv`` file has two columns. The first column is the training noise level of a denoiser and the second column is the path to the denoiser. The denoiser must be in ``.onnx`` format. Two sample ``.csv`` files are provided in ``$SSRI/airi_denoisers``.
        - ``im_peak_est``(optional): Estimated maximum intensity of the true image. If this field is empty, the default value is the maximum intensity of the back-projected (dirty) image normalised by the peak value of the PSF.
        - ``dnn_adaptive_peak``(optional): Enable the adaptive denoiser selection scheme. The details of this scheme can be found in [[1]](https://arxiv.org/abs/2312.07137v2). Default: ``true``.
        - ``dnn_apply_transform``(optional): Apply random rotation and flipping before denoising then undo the transform after denoising. Default: ``true``.
        - ``im_min_itr``(optional): Minimum number of iterations in the primal-dual algorithm. Default: ``200``.
        - ``im_max_itr``(optional): Maximum number of iterations in the primal-dual algorithm. Default: ``2000``.
        - ``im_var_tol``(optional): Tolerance on the relative variation of the estimation in the primal-dual algorithm to indicate convergence. Default: ``1e-4``.
        - ``itr_save``(optional): Interval of iterations for saving intermediate results. Default: ``500``.
        - ``dnn_adaptive_peak_tol_max``(optional): Initial relative peak value variation tolerance for the adaptive denoiser selection scheme. Default: ``0.1``.
        - ``dnn_adaptive_peak_tol_min``(optional): Minimum relative peak value variation tolerance for the adaptive denoiser selection scheme. Default: ``1e-3``.
        - ``dnn_adaptive_peak_tol_step``(optional): Decaying factor for the relative peak value variation tolerance in the adaptive denoiser selection scheme. It will be applied to the current tolerance after one time of denoiser reselection. Default: ``0.1``.
