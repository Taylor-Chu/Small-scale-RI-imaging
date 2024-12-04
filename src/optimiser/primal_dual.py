"""
Primal-dual algorithm
"""

import os
from timeit import default_timer as timer
import torch
from astropy.io import fits

from .optimiser import Optimiser
from ..prox_operator import ProxOp
from ..ri_measurement_operator.pysrc.measOperator import MeasOp
from ..prox_operator import ProxOpElipse


class PrimalDualPrecond(Optimiser):
    """
    Primal-dual algorithm with preconditioned dual data update.

    This implementation support solving objective function with two
    non-differentiable terms. The data-fidelity term with pre-conditioning is
    handeled in the dual update step. The regularization term is handeled in the
    primal update step.
    """

    def __init__(
        self,
        meas: torch.Tensor,
        meas_op: MeasOp,
        prox_op_prime: ProxOp,
        prox_op_dual_data: ProxOpElipse,
        im_max_itr: int = 2000,
        save_pth: str = "results",
        file_prefix: str = "",
    ) -> None:
        """
        Constructs all the necessary attributes for the PrimalDualPrecond object.

        Args:
            meas (torch.Tensor): The measurement tensor.
            meas_op (MeasOp): The measurement operator.
            prox_op_prime (ProxOp): The proximal operator for the prime variable.
            prox_op_dual_data (ProxOpElipse): The proximal operator for the dual data.
            im_max_itr (int, optional): The maximum number of iterations. Defaults to 2000.
            save_pth (str, optional): The path to save the results. Defaults to "results".
            file_prefix (str, optional): The prefix of the saving files. Defaults to "".
        """
        super().__init__(meas, meas_op, save_pth=save_pth, file_prefix=file_prefix)
        self._prox_op_prime = prox_op_prime
        self._prox_op_dual_data = prox_op_dual_data
        self._start_iter = 0
        self._im_max_itr = im_max_itr
        self._pr_step_size = 1.0
        self._precond_weight = torch.ones(1, 1, device=meas_op.get_device())
        self._dual = torch.zeros_like(self._meas)

        # timing
        self._iter = 1
        self._t_total = 0.0
        self._t_iter = 0.0
        self._t_primal = 0.0
        self._t_dual = 0.0

        # cuda event
        self._cuda_timing = False
        if self._meas_op.get_device() == torch.device("cuda"):
            self._cuda_timing = True

        # save dirty image and psf
        self._meas_bp = self._meas_op.adjoint_op(self._meas).to(
            self._meas_op.get_device()
        )
        self._psf = self._meas_op.get_psf()
        self._psf_peak = self._psf.max().item()
        fits.writeto(
            os.path.join(self._save_pth, "dirty.fits"),
            self.get_dirty_image() / self._psf_peak,
            overwrite=True,
        )
        fits.writeto(
            os.path.join(self._save_pth, "psf.fits"),
            self.get_psf(),
            overwrite=True,
        )

    @torch.no_grad()
    def run(self) -> None:
        """
        Runs the main loop of primal-dual algorithm.

        This method iteratively updates the prime and dual variables until the
        stop criteria is met or the maximum number of iterations is reached.
        """
        if self._cuda_timing:
            primal_start_event = torch.cuda.Event(enable_timing=True)
            primal_end_event = torch.cuda.Event(enable_timing=True)
            dual_start_event = torch.cuda.Event(enable_timing=True)
            dual_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()

        self._t_total = timer()
        for self._iter in range(self._start_iter, self._im_max_itr):
            self._t_iter = timer()
            self._each_iter_begin()

            # primal step
            if self._cuda_timing:
                primal_start_event.record()
            else:
                self._t_primal = timer()
            self._model = self._model - self._pr_step_size * self._meas_op.adjoint_op(
                self._dual
            )
            self._model = self._model.to(
                device=self._prox_op_prime.get_device(),
                dtype=self._prox_op_prime.get_data_type(),
            )
            self._model = self._prox_op_prime(self._model)
            if self._cuda_timing:
                primal_end_event.record()
                torch.cuda.synchronize()
                self._t_primal = primal_start_event.elapsed_time(primal_end_event) / 1e3
            else:
                self._t_primal = timer() - self._t_primal

            # dual step
            if self._cuda_timing:
                dual_start_event.record()
            else:
                self._t_dual = timer()
            self._model = self._model.to(device=self._meas_op.get_device()).to(
                dtype=self._meas_op.get_data_type()
            )
            # TODO: better way to write the dual update?
            self._dual = self._dual / self._precond_weight + self._meas_op.forward_op(
                2 * self._model - self._model_prev
            )
            self._dual = (
                self._dual - self._prox_op_dual_data(self._dual)
            ) * self._precond_weight
            if self._cuda_timing:
                dual_end_event.record()
                torch.cuda.synchronize()
                self._t_dual = dual_start_event.elapsed_time(dual_end_event) / 1e3
            else:
                self._t_dual = timer() - self._t_dual
            self._t_iter = timer() - self._t_iter

            if self._stop_criteria():
                break

            self._each_iter_end()

            self._model_prev = self._model

        self._t_total = timer() - self._t_total
