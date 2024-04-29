from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Optional

from torch import nn, Tensor

Loss = Tensor


class ThunderModule(ABC, nn.Module):
    def __init__(
        self,
        args: Optional[Namespace] = None,
    ):
        super(ThunderModule, self).__init__()

        if args is not None:
            self.args = args

    def hyperparameters(self) -> Optional[dict]:
        if hasattr(self, "args"):
            return vars(self.args)
        return None

    def loss_fn(
        self,
        pred: Tensor,
        target: Tensor,
        **kwargs,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def training_step(
        self, batch: tuple[Tensor, ...], batch_idx: int, epoch: int, **kwargs
    ) -> Loss:
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self, batch: tuple[Tensor, ...], batch_idx: int, epoch: int, **kwargs
    ) -> Loss:
        raise NotImplementedError

    @abstractmethod
    def prediction_step(
        self, batch: tuple[Tensor, ...], batch_idx: int, epoch: int, **kwargs
    ) -> Any | Tensor:
        raise NotImplementedError
