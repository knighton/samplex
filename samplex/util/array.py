from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import Union

import numpy as np
from numpy.typing import NDArray

from samplex.util.base import SxUtilError


class SxUtilArrayError(SxUtilError):
    """Errors of `samplex.util.array`."""


class SxIndexWrapError(SxUtilArrayError, TypeError):
    """Index wraparound error."""

    en = (
        'Array index integer is out of range. Supported is the range '
        '[-size, size).'
    )


class SxPointDtypeError(SxUtilArrayError, TypeError):
    """Point dtype error."""

    en = '0-dimensional scalar arrays for indexing must have integer dtpye.'


class SxBoolMaskNdimError(SxUtilArrayError, IndexError):
    """Boolean mask ndim error."""

    en = 'Boolean masks must be 1-dimensional.'


class SxBoolMaskLenError(SxUtilArrayError, IndexError):
    """Boolean mask length error."""

    en = 'Boolean masks must match the underlying array length.'


class SxVecDtypeError(SxUtilArrayError, TypeError):
    """Vector dtype error."""

    en = '1-dimensional arrays for indexing must have integer or boolean dtype.'


class SxSliceStepError(SxUtilArrayError, ValueError):
    """Slice step error."""

    en = 'The slice step cannot be zero, or else it would not go anywhere.'


class SxIndexTypeError(SxUtilArrayError, TypeError):
    """Index type error."""

    en = 'Unsupported type used for indexing an array.'


T = TypeVar('T')


SimpleIndex = Union[
    int,
    np.integer,
    type(Ellipsis),
    slice,
    list[int],
    tuple[int, ...],
    NDArray[np.integer],
    NDArray[np.bool_],
]

RecursiveIndex = Union[
    SimpleIndex,
    list[Any],
    tuple[Any, ...],
    NDArray[np.integer | np.bool_],
]


def _wrap(idx: int, num: int) -> int:
    """Wrap an index to handle negative indices and bounds checking."""
    if not (-num <= idx < num):
        raise SxIndexWrapError(index=idx, length=num)

    if idx < 0:
        idx += num

    return idx


class Array(ABC, Sequence, Generic[T]):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getoneitem__(self, idx: int) -> T:
        ...

    def _handle_array_index(self, arr: NDArray[Any], num: int) -> Any:
        """Handle numpy array indexing cases."""
        if not arr.ndim:
            if not np.issubdtype(arr.dtype, np.integer):
                raise SxPointDtypeError(dtype=arr.dtype)

            return self.__getoneitem__(_wrap(int(arr), num))

        if not arr.shape[0]:
            return [[] for _ in range(arr.shape[0])]

        if arr.dtype == np.bool_:
            if arr.ndim != 1:
                raise SxBoolMaskNdimError(shape=arr.shape)

            if arr.shape[0] != num:
                raise SxBoolMaskLenError(mask_len=len(arr), array_len=num)

            indices = np.where(arr)[0]
            return [self.__getoneitem__(int(i)) for i in indices]

        if np.issubdtype(arr.dtype, np.integer):
            if arr.ndim == 1:
                return [self.__getoneitem__(_wrap(int(i), num)) for i in arr]
            return [self[i] for i in arr]

        raise SxVecDtypeError(dtype=arr.dtype)

    def __getitem__(self, arg: RecursiveIndex) -> Any:
        num = len(self)

        if isinstance(arg, int | np.integer):
            return self.__getoneitem__(_wrap(int(arg), num))

        if isinstance(arg, type(Ellipsis)):
            return [self.__getoneitem__(i) for i in range(num)]

        if isinstance(arg, slice):
            if arg.step == 0:
                raise SxSliceStepError(slice=arg)

            return [self.__getoneitem__(i) for i in range(*arg.indices(num))]

        if isinstance(arg, list):
            return [self[i] for i in arg]

        if isinstance(arg, tuple):
            return tuple(self[i] for i in arg)

        if isinstance(arg, np.ndarray):
            return self._handle_array_index(arg, num)

        raise SxIndexTypeError(type=type(arg))
