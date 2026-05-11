"""Define types useful for the project."""
from typing import TypeVar, Union

from torch import Tensor
from torch.utils.data import TensorDataset

from epiclass.core.lazy.lazy_data_classes import LazyKnownData, LazyUnknownData

TensorData = TypeVar("TensorData", TensorDataset, Tensor)
SomeData = Union[LazyKnownData, LazyUnknownData]
