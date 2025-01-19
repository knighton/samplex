import numpy as np
from numpy.typing import NDArray

from samplex.util.array import Array
from samplex.util.base import SxUtilError


class SxUtilRangeArrError(SxUtilError):
    """Errors of `samplex.util.uniq_list`."""


class SxIndexOutOfRangeError(SxUtilRangeArrError, IndexError):
    """Range array Index out of range error."""

    en = 'The index provided to RangeArray was out of range.'


class RangeArray(Array[tuple[int, int]]):
    def __init__(self, tails: NDArray[np.int64]):
        self.tails = tails

    def __len__(self) -> int:
        return len(self.tails)

    def __getoneitem__(self, range_id: int) -> tuple[int, int]:
        if not (0 <= range_id < len(self.tails)):
            raise SxIndexOutOfRangeError(
                range_id=range_id, num_ranges=len(self.tails)
            )

        if not range_id:
            head = 0
            tail = int(self.tails[0])
        else:
            head = int(self.tails[range_id - 1])
            tail = int(self.tails[range_id])

        return head, tail
