from collections.abc import Sequence
from typing import Any
from typing import TypeVar

from samplex.util.base import SxUtilError


class SxUtilMultiError(SxUtilError):
    """Errors of `samplex.util.multi`."""


class SxEnsureOneOrMultiError(SxUtilMultiError, ValueError):
    """Ensure one or multi error."""

    en = 'Input must consist of either one or the given count of elements.'


class SxGetSameAttrEmptyError(SxUtilMultiError, ValueError):
    """Get same attribute empty error."""

    en = 'At least one object must be provided to get the given attribute of.'


class SxGetSameAttrInequalError(SxUtilMultiError, ValueError):
    """Get same attribute inequal error."""

    en = 'The corresponding attributes of the given objects must all compare as equal.'


T = TypeVar('T')


def ensure_count(val: T | Sequence[T], cnt: int) -> list[T]:
    if isinstance(val, Sequence):
        ret = list(val)
        if len(ret) == 1:
            ret *= cnt
        elif len(ret) == cnt:
            pass
        else:
            raise SxEnsureOneOrMultiError(value=val, count=cnt)
    else:
        ret = [val] * cnt
    return ret


def get_same_attr(objs: list, key: str) -> Any:
    if not objs:
        raise SxGetSameAttrEmptyError(objs=objs)

    vals = [getattr(obj, key) for obj in objs]
    vals_set = set(vals)

    if len(vals_set) != 1:
        raise SxGetSameAttrInequalError(vals=vals, vals_set=vals_set)

    return vals[0]
