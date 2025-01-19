from collections.abc import Iterator
from typing import Self

import numpy as np
from numpy.typing import NDArray

from samplex.util.array import Array
from samplex.util.base import SxUtilError
from samplex.util.multi import get_same_attr
from samplex.util.range_arr import RangeArray


class SxUtilRangeIdxError(SxUtilError):
    """Errors of `samplex.util.range_idx`."""


class SxRangeIndexInitNoRangesError(SxUtilRangeIdxError, IndexError):
    """RangeIndex initialized with no ranges error."""

    en = 'Attempted to initialize a RangeIndex with no ranges to index. There must be at least one range.'


class SxRangeIndexInitBadRangesError(SxUtilRangeIdxError, ValueError):
    """RangeIndex initialized with bad ranges error."""

    en = 'Attempted to initialize a RangeIndex with invalid sized range(s). Each range must be of positive size.'


class SxIndexOutOfRangeError(SxUtilRangeIdxError, IndexError):
    """RangeIndex item ID out of range error."""

    en = 'The item ID provided to RangeIndex was out of range.'


class SxCatNoRangeIndexesError(SxUtilRangeIdxError, IndexError):
    """Cat no RangeIndexes error."""

    en = 'Cannot concatenate empty list of RangeIndexes.'


class RangeIndex(Array[tuple[int, int]]):
    def __init__(
        self,
        tails: NDArray[np.int64],
        check: bool = True,
    ):
        if not len(tails):
            raise SxRangeIndexInitNoRangesError(tails=tails)

        tails = np.asarray(tails, dtype=np.int64)

        if check:
            diffs = np.diff(tails)
            if not (0 < diffs).all():
                neg_len_cnt = (diffs < 0).sum()
                zero_len_cnt = (diffs == 0).sum()
                raise SxRangeIndexInitBadRangesError(
                    num_ranges=len(tails),
                    num_zero_len_ranges=zero_len_cnt,
                    num_neg_len_ranges=neg_len_cnt,
                )

        self.num_ranges = len(tails)
        self.num_items = int(tails[-1])
        self.tails = tails
        self.bounds = RangeArray(self.tails)

    @classmethod
    def from_lens(cls, lens: NDArray[np.int64]) -> Self:
        lens = np.asarray(lens, dtype=np.int64)
        return cls(lens.cumsum())

    def __len__(self) -> int:
        return self.num_items

    def __getoneitem__(self, item_id: int) -> tuple[int, int]:
        if not (0 <= item_id < self.num_items):
            raise SxIndexOutOfRangeError(
                item_id=item_id, num_ranges=len(self.tails)
            )

        range_id = int(np.searchsorted(self.tails, item_id, side='right'))
        range_item_id = item_id - int(
            self.tails[range_id - 1] if range_id else 0
        )
        return range_id, range_item_id

    @staticmethod
    def cat(objs: list['RangeIndex']) -> 'RangeIndex':
        if not objs:
            raise SxCatNoRangeIndexesError(
                objs=objs,
            )

        tails = []
        offset = 0
        for obj in objs:
            tails.append(offset + obj.tails)
            offset = tails[-1][-1]
        tails = np.concatenate(tails)
        return RangeIndex(tails)

    @staticmethod
    def each_range_begin(objs: list['RangeIndex']) -> Iterator[int]:
        """Yields indices where any range_id changes."""
        if not objs:
            return

        end_item_id = get_same_attr(objs, 'num_items')

        if not end_item_id:
            return

        item_id = 0
        yield item_id

        while True:
            next_item_id = end_item_id
            for obj in objs:
                range_id, _ = obj[item_id]
                if range_id + 1 < obj.num_ranges:
                    range_start, range_end = obj.bounds[range_id]
                    if range_end < next_item_id:
                        next_item_id = range_end
            item_id = next_item_id
            if not (item_id < end_item_id):
                break
            yield item_id
