"""
Base class for proximity operator
"""

from abc import ABC, abstractmethod
from typing import Any
import torch


class ProxOp(ABC):
    """
    Base class for proximity operator.

    This class provides the base functionality for a proximity operator.
    It defines common methods for various proximity operators, such as
    applying the operator to an input and updating the operator.
    The actual implementation of these methods are left to the subclasses.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float,
    ) -> None:
        """
        Initialize the ProxOp class.

        Args:
            device (torch.device, optional): The device for the tensors.
                Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): The data type for the tensors.
                Defaults to torch.float.
        """
        self._dtype = dtype
        self._device = device

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> Any:
        """
        Apply proximity operator to input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            NotImplemented: This method should be implemented in a subclass.
        """
        return NotImplemented

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """
        Update proximity operator.

        This method should be implemented in a subclass.
        The input arguments can be arbitrary and should be specified in the subclass.
        """
        return NotImplemented

    def get_device(self) -> torch.device:
        """
        Return the device that the proximity operator is running on.

        Returns:
            torch.device: The device that the proximity operator is running on.
        """
        return self._device

    def get_data_type(self) -> torch.dtype:
        """
        Return the data type of the data that the proximity operator will accept and return.

        Returns:
            torch.dtype: The data type of the data that the proximity operator will accept
                and return.
        """
        return self._dtype
