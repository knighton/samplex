from typing import Generic
from typing import Protocol
from typing import Self
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from samplex.util.base import SxUtilError


class SxUtilUniqListError(SxUtilError):
    """Errors of `samplex.util.uniq_list`."""


class SxUniqValMismatchError(SxUtilUniqListError, ValueError):
    """Unique value mismatch error."""

    en = 'Mismatch detected between the list of unique values and the keys of the mapping of unique values to their corresponding indices.'


class SxUniqValIdxMismatchError(SxUtilUniqListError, ValueError):
    """Unique value index mismatch error."""

    en = 'Mismatch detected between the list of unique value indices and the values of the mapping of unique values to their corresponding indices.'


class SxUniqValUnusedError(SxUtilUniqListError, ValueError):
    """Unique value not used error."""

    en = 'One or more unique values in the list vocabulary are not actually used in the list itself.'


class Comparable(Protocol):
    def __lt__(self, other: Self) -> bool:
        ...


T = TypeVar('T', bound=Comparable)


class UniqueList(Generic[T]):
    """A list of often repeated items, which is stored deduped to save space."""

    def __init__(
        self,
        uniq_vals: list[T],
        val_ids: NDArray[np.int64],
        val2val_id: dict[T, int],
        check: bool = False,
    ):
        if check:
            # First check: Values match between list and dict
            if len(uniq_vals) != len(val2val_id):
                left = set(uniq_vals)
                right = set(val2val_id)
                uniq_vals_only = sorted(left.difference(right))
                dict_keys_only = sorted(right.difference(left))
                raise SxUniqValMismatchError(
                    uniq_vals_only=uniq_vals_only,
                    dict_keys_only=dict_keys_only,
                )

            # Second check: Indices are sequential
            got = sorted(val2val_id.values())
            req = list(range(len(got)))
            if got != req:
                got_set = set(got)
                req_set = set(req)
                got_only = sorted(got_set.difference(req_set))
                req_only = sorted(req_set.difference(got_set))
                raise SxUniqValIdxMismatchError(
                    uniq_vals_only=got_only,
                    dict_vals_only=req_only,
                )

            # Final check: All values are used
            has = np.zeros(len(uniq_vals), np.bool_)
            has[val_ids] = True
            unused = (has == 0).nonzero()[0].tolist()
            if not has.all():
                raise SxUniqValUnusedError(unused_vals=unused)

        self.uniq_vals = uniq_vals
        self.val_ids = val_ids
        self.val2val_id = val2val_id

    @classmethod
    def from_vals(cls, vals: list[T]) -> Self:
        """Initialize from a full list of values, dupes included."""
        vals_set = set(vals)
        uniq_vals = sorted(vals_set)

        val2val_id = {}
        for val_id, val in enumerate(uniq_vals):
            val2val_id[val] = val_id

        val_ids = np.empty(len(vals), np.int64)
        for idx, val in enumerate(vals):
            val_ids[idx] = val2val_id[val]

        return cls(uniq_vals, val_ids, val2val_id)

    @staticmethod
    def cat(objs: list['UniqueList[T]']) -> 'UniqueList[T]':
        """Concatenate multiple UniqueLists into one big UniqueList."""
        # Get unique sorted values from all lists
        uniq_vals = []
        for obj in objs:
            uniq_vals.extend(obj.uniq_vals)
        uniq_vals = sorted(set(uniq_vals))

        # Create new mapping
        val2val_id = {val: val_id for val_id, val in enumerate(uniq_vals)}

        # Calculate total length needed for val_ids
        num_val_ids = sum(len(obj.val_ids) for obj in objs)

        # Create and fill val_ids array
        val_ids = np.empty(num_val_ids, np.int64)
        idx = 0
        for obj in objs:
            # Create conversion array for this object's val_ids
            conv = np.empty(len(obj.uniq_vals), np.int64)
            for from_val_id, val in enumerate(obj.uniq_vals):
                conv[from_val_id] = val2val_id[val]

            # Map this object's val_ids to new indices
            count = len(obj.val_ids)
            val_ids[idx : idx + count] = conv[obj.val_ids]
            idx += count

        return UniqueList(uniq_vals, val_ids, val2val_id)
