"""Define types useful for the project."""
from typing import TypeVar, Union

from torch import Tensor
from torch.utils.data import TensorDataset

from epiclass.core.data.eager import KnownData, UnknownData

TensorData = TypeVar("TensorData", TensorDataset, Tensor)
SomeData = Union[KnownData, UnknownData]
