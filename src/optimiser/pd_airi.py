"""
Constrained AIRI built on primal-dual algorithm
"""

import os
from typing import Union
import torch
import numpy as np
from astropy.io import fits

from .primal_dual import PrimalDualPrecond
from ..prox_operator import ProxOpAIRI, ProxOpElipse
from ..ri_measurement_operator.pysrc.measOperator import MeasOpNUFFT


class PDAIRI(PrimalDualPrecond):
    """
    This class implements constrained AIRI built on the primal-dual algorithm.

    It handles data fidelity in the dual step and uses AIRI denoisers in the
    primal step for regularization.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        meas_op: MeasOpNUFFT,
        prox_op_prime: ProxOpAIRI,
        prox_op_dual_data: ProxOpElipse,
        im_min_itr: int = 100,
        im_max_itr: int = 2000,
        im_var_tol: float = 1e-4,
        im_peak_est: Union[float, None] = None,
        heu_noise_scale: float = 1.0,
        adapt_net_select: bool = True,
        peak_tol_min: float = 1e-3,
        peak_tol_max: float = 0.1,
        peak_tol_step: float = 0.1,
        verbose: bool = True,
        save_pth: str = "results",
        file_prefix: str = "",
        iter_save: int = 1000,
    ) -> None:
        """
        Initialises the PDAIRI class.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOpTkbnRI): The measurement operator.
            prox_op_prime (ProxOpAIRI): The proximal operator for the prime variable.
            prox_op_dual_data (ProxOpElipse): The proximal operator for the dual data.
            im_min_itr (int, optional): The minimum number of iterations. Defaults to 100.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            im_var_tol (float, optional): The image variation tolerance. Defaults to 1e-4.
            im_peak_est (Union[float, None], optional): The estimated peak of the image.
                Defaults to None.
            heu_noise_scale (float, optional): The heuristic noise scale. Defaults to 1.0.
            adapt_net_select (bool, optional): Whether to adaptively select the network.
                Defaults to True.
            peak_tol_min (float, optional): The minimum peak tolerance. Defaults to 1e-3.
            peak_tol_max (float, optional): The maximum peak tolerance. Defaults to 0.1.
            peak_tol_step (float, optional): The step size for peak tolerance. Defaults to 0.1.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            save_pth (str, optional): The path to save the results. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
            iter_save (int, optional): The number of iterations after which to save the results.
                Defaults to 1000.
        """
        super().__init__(
            meas,
            meas_op,
            prox_op_prime,
            prox_op_dual_data,
            im_max_itr=im_max_itr,
            save_pth=save_pth,
            file_prefix=file_prefix,
        )

        self._im_min_itr = im_min_itr
        self._im_var_tol = im_var_tol
        self._im_peak_est = im_peak_est
        self._heu_noise_scale = heu_noise_scale
        self._adapt_net_select = adapt_net_select
        self._peak_tol_min = peak_tol_min
        self._peak_tol = peak_tol_max
        self._peak_tol_step = peak_tol_step
        self._peak_val_range = []
        self._verbose = verbose
        self._iter_save = iter_save

        self._prev_peak_val = 1.0
        self._heuristic = 1.0
        self._im_rel_var = 1.0

    def initialisation(self) -> None:
        """
        Initialises specific parameters of the algorithm including step size,
        the maximum intensity of the target image, and heuristic noise level.
        The AIRI proximal operator is also initialised.
        """
        # calculate step size
        self._meas_op.set_real_flag(False)
        self._precond_weight = self._prox_op_dual_data.get_precond_weight()
        self._meas_op.set_precond_weight(torch.sqrt(self._precond_weight))
        self._pr_step_size = 0.5 / self._meas_op.get_op_norm()
        if self._verbose:
            print(
                f"INFO: measurement operator norm for primal-dual {self._meas_op.get_op_norm()}",
                flush=True,
            )
        self._meas_op.set_precond_weight(
            torch.ones(1, 1, device=self._meas_op.get_device())
        )
        self._meas_op.set_real_flag(True)

        # estimate peak value
        if self._im_peak_est is None:
            self._prev_peak_val = self._meas_bp.max().item() / self._psf_peak
            if self._verbose:
                print(
                    "INFO: use normalised dirty peak as",
                    f"estimated image peak value: {self._prev_peak_val}",
                    flush=True,
                )
        else:
            self._prev_peak_val = self._im_peak_est
            if self._verbose:
                print(
                    f"\nINFO: user specified the estimated image peak value: {self._prev_peak_val}",
                    flush=True,
                )

        # heuristic noise level
        self._heuristic = 1 / np.sqrt(2 * self._meas_op.get_op_norm())
        if self._verbose:
            print(
                f"INFO: measurement operator norm {self._meas_op.get_op_norm()}",
                flush=True,
            )
            print(f"INFO: heuristic noise level: {self._heuristic}", flush=True)
        heu_corr_factor = np.sqrt(
            self._meas_op.get_op_norm_prime() / self._meas_op.get_op_norm()
        )
        if not np.isclose(heu_corr_factor, 1.0):
            self._heuristic *= heu_corr_factor
            if self._verbose:
                print(
                    f"INFO: heuristic noise level after correction: {self._heuristic},",
                    f"corection factor {heu_corr_factor}",
                )
        if not np.isclose(self._heu_noise_scale, 1.0):
            self._heuristic *= self._heu_noise_scale
            if self._verbose:
                print(
                    f"INFO: heuristic noise level after scaling: {self._heuristic},",
                    f"scaling factor {self._heu_noise_scale}",
                )

        # initialise AIRI prox
        self._peak_val_range = self._prox_op_prime.update(
            self._heuristic, self._prev_peak_val
        )

        if self._verbose:
            print(f"INFO: primal step size: {self._pr_step_size}")
            print("\n*************************************************", flush=True)
            print("********* STARTING ALGORITHM:   cAIRI   *********", flush=True)
            print("*************************************************", flush=True)

    def _each_iter_begin(self):
        """
        Constrained AIRI does not require any specific operation at
        the beginning of each iteration.
        """

    @torch.no_grad()
    def _stop_criteria(self):
        """
        Determines the stop criteria for the algorithm based on image relative variation
        and data fidelity.

        Returns:
            bool: Whether the stop criteria has been met.
        """
        # img relative variation
        self._im_rel_var = torch.linalg.vector_norm(self._model - self._model_prev) / (
            torch.linalg.vector_norm(self._model) + 1e-10
        )
        self._model_prev = self._model

        # stop criteria
        if self._iter + 1 >= self._im_min_itr and self._im_rel_var < self._im_var_tol:
            curr_l2_error = torch.linalg.vector_norm(
                self._meas - self._meas_op.forward_op(self._model)
            )
            # log
            if self._verbose:
                print(
                    f"\nIter {self._iter+1}: relative variation {self._im_rel_var},",
                    f"data fidelity {curr_l2_error}\ntimings: primal step {self._t_primal} sec,",
                    f"dual step {self._t_dual} sec, iteration {self._t_iter} sec.",
                    flush=True,
                )
            if curr_l2_error < self._prox_op_dual_data.get_radius():
                return True
        elif self._verbose:
            print(
                f"\nIter {self._iter+1}: relative variation {self._im_rel_var}",
                f"\ntimings: primal step {self._t_primal} sec, dual step {self._t_dual} sec,",
                f"iteration {self._t_iter} sec.",
                flush=True,
            )

        return False

    @torch.no_grad()
    def _each_iter_end(self):
        """
        Saves intermediate results for every `iter_save` iteration and updates the
        AIRI proximal operator with the latest image peak value if the relative
        variation of the peak value is smaller than certain tolerence.
        """
        # save intermediate results
        if (self._iter + 1) % self._iter_save == 0:
            fits.writeto(
                os.path.join(
                    self._save_pth,
                    self._file_prefix
                    + "tmp_model_itr_"
                    + str(self._iter + 1)
                    + ".fits",
                ),
                self.get_model_image(dtype=torch.float32),
                overwrite=True,
            )
            fits.writeto(
                os.path.join(
                    self._save_pth,
                    self._file_prefix
                    + "tmp_residual_itr_"
                    + str(self._iter + 1)
                    + ".fits",
                ),
                self.get_residual_image(dtype=torch.float32) / self._psf_peak,
                overwrite=True,
            )

        # AIRI denoiser selection
        if self._adapt_net_select:
            curr_peak_val = self._model.max().item()
            peak_var = abs(curr_peak_val - self._prev_peak_val) / self._prev_peak_val
            if self._verbose:
                print(
                    f"  Model image peak value {curr_peak_val}, relative variation = {peak_var}",
                    flush=True,
                )

            if peak_var < self._peak_tol and (
                curr_peak_val < self._peak_val_range[0]
                or curr_peak_val > self._peak_val_range[1]
            ):
                self._peak_val_range = self._prox_op_prime.update(
                    self._heuristic, self._prev_peak_val
                )
            self._prev_peak_val = curr_peak_val

    @torch.no_grad()
    def finalisation(self):
        """
        Finalises the algorithm by printing the total time and number of iterations.
        The final model and residual images will be save to `save_pth`.
        """
        if self._verbose:
            print("\n**************************************", flush=True)
            print("********** END OF ALGORITHM **********", flush=True)
            print("**************************************\n", flush=True)
            print(
                f"Imaging finished in {self._t_total} sec, ",
                f"total number of iterations {self._iter+1}",
                flush=True,
            )

        # save final results
        fits.writeto(
            os.path.join(self._save_pth, self._file_prefix + "model_image.fits"),
            self.get_model_image(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(
                self._save_pth, self._file_prefix + "residual_dirty_image.fits"
            ),
            self.get_residual_image(),
            overwrite=True,
        )
        fits.writeto(
            os.path.join(
                self._save_pth,
                self._file_prefix + "normalised_residual_dirty_image.fits",
            ),
            self.get_residual_image() / self._psf_peak,
            overwrite=True,
        )
