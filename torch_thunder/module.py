from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Optional

from torch import nn, Tensor

type Loss = Tensor


class ThunderModule(ABC, nn.Module):
    def __init__(
        self,
        args: Optional[Namespace] = None,
    ):
        super(ThunderModule, self).__init__()

        if args is not None:
            self.args = args

    def hyperparameters(self) -> Optional[dict]:
        """
        Return hyperparameters of the model as a dictionary

        Returns:
            dict: hyperparameters of the model
        """
        if hasattr(self, "args"):
            return vars(self.args)
        return None

    def loss_fn(
        self,
        pred: Tensor,
        target: Tensor,
        **kwargs,
    ) -> Any:
        """
        User-defined loss function that can be used in training and validation steps

        Args:
            pred (Tensor): model prediction
            target (Tensor): target value
            **kwargs: additional arguments

        Returns:
            Any
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Method `forward` must be implemented")

    @abstractmethod
    def training_step(
        self, batch: tuple[Tensor, ...], batch_idx: int, epoch: int, **kwargs
    ) -> Loss:
        raise NotImplementedError("Method `training_step` must be implemented")

    @abstractmethod
    def validation_step(
        self, batch: tuple[Tensor, ...], batch_idx: int, epoch: int, **kwargs
    ) -> Loss:
        raise NotImplementedError("Method `validation_step` must be implemented")

    @abstractmethod
    def prediction_step(self, *args, **kwargs) -> Any | Tensor:
        raise NotImplementedError
