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

    de = 'Array-Index-Ganzzahl ist außerhalb des gültigen Bereichs. Unterstützt wird der Bereich [-size, size).'
    en = 'Array index integer is out of range. Supported is the range [-size, size).'
    es = 'El índice del array está fuera de rango. El rango soportado es [-size, size).'
    fr = "L'indice du tableau est hors limites. La plage supportée est [-size, size)."
    ja = '配列のインデックスが範囲外です。サポートされている範囲は [-size, size) です。'
    ko = '배열 인덱스가 범위를 벗어났습니다. 지원되는 범위는 [-size, size) 입니다.'
    ru = 'Индекс массива вне допустимого диапазона. Поддерживается диапазон [-size, size).'
    zh = '数组索引整数超出范围。支持的范围是 [-size, size)。'


class SxPointDtypeError(SxUtilArrayError, TypeError):
    """Point dtype error."""

    de = '0-dimensionale Skalar-Arrays für die Indizierung müssen den Datentyp Integer haben.'
    en = '0-dimensional scalar arrays for indexing must have integer dtype.'
    es = 'Los arrays escalares de 0 dimensiones para indexación deben tener dtype entero.'
    fr = "Les tableaux scalaires de dimension 0 pour l'indexation doivent avoir un type de données entier."
    ja = 'インデックス付けに使用する0次元スカラー配列は整数型である必要があります。'
    ko = '인덱싱을 위한 0차원 스칼라 배열은 정수 dtype을 가져야 합니다.'
    ru = 'Скалярные массивы нулевой размерности для индексации должны иметь целочисленный тип данных.'
    zh = '用于索引的0维标量数组必须具有整数数据类型。'


class SxBoolMaskNdimError(SxUtilArrayError, IndexError):
    """Boolean mask ndim error."""

    de = 'Boolesche Masken müssen eindimensional sein.'
    en = 'Boolean masks must be 1-dimensional.'
    es = 'Las máscaras booleanas deben ser unidimensionales.'
    fr = 'Les masques booléens doivent être unidimensionnels.'
    ja = 'ブール型マスクは1次元である必要があります。'
    ko = '부울 마스크는 1차원이어야 합니다.'
    ru = 'Булевы маски должны быть одномерными.'
    zh = '布尔掩码必须是一维的。'


class SxBoolMaskLenError(SxUtilArrayError, IndexError):
    """Boolean mask length error."""

    de = 'Boolesche Masken müssen die gleiche Länge wie das zugrunde liegende Array haben.'
    en = 'Boolean masks must match the underlying array length.'
    es = 'Las máscaras booleanas deben coincidir con la longitud del array subyacente.'
    fr = 'Les masques booléens doivent correspondre à la longueur du tableau sous-jacent.'
    ja = 'ブール型マスクは基礎となる配列の長さと一致する必要があります。'
    ko = '부울 마스크는 기본 배열의 길이와 일치해야 합니다.'
    ru = 'Длина булевой маски должна совпадать с длиной базового массива.'
    zh = '布尔掩码的长度必须与底层数组的长度相匹配。'


class SxVecDtypeError(SxUtilArrayError, TypeError):
    """Vector dtype error."""

    de = 'Eindimensionale Arrays für die Indizierung müssen einen ganzzahligen oder booleschen Datentyp haben.'
    en = '1-dimensional arrays for indexing must have integer or boolean dtype.'
    es = 'Los arrays unidimensionales para indexación deben tener dtype entero o booleano.'
    fr = "Les tableaux unidimensionnels pour l'indexation doivent avoir un type de données entier ou booléen."
    ja = 'インデックス付けに使用する1次元配列は整数型またはブール型である必要があります。'
    ko = '인덱싱을 위한 1차원 배열은 정수 또는 부울 dtype을 가져야 합니다.'
    ru = 'Одномерные массивы для индексации должны иметь целочисленный или булев тип данных.'
    zh = '用于索引的一维数组必须具有整数或布尔数据类型。'


class SxSliceStepError(SxUtilArrayError, ValueError):
    """Slice step error."""

    de = 'Die Slice-Schrittweite darf nicht Null sein, da sonst keine Iteration möglich ist.'
    en = 'The slice step cannot be zero, or else it would not go anywhere.'
    es = 'El paso del slice no puede ser cero, ya que no iría a ninguna parte.'
    fr = "Le pas de la tranche ne peut pas être zéro, sinon elle n'irait nulle part."
    ja = 'スライスのステップは0にできません。0の場合は進むことができません。'
    ko = '슬라이스 단계는 0이 될 수 없습니다. 그렇지 않으면 어디로도 갈 수 없습니다.'
    ru = 'Шаг среза не может быть равен нулю, иначе он никуда не пойдет.'
    zh = '切片步长不能为零，否则无法进行迭代。'


class SxIndexTypeError(SxUtilArrayError, TypeError):
    """Index type error."""

    de = 'Nicht unterstützter Typ für die Array-Indizierung verwendet.'
    en = 'Unsupported type used for indexing an array.'
    es = 'Tipo no soportado utilizado para indexar un array.'
    fr = "Type non pris en charge utilisé pour l'indexation d'un tableau."
    ja = '配列のインデックス付けに未サポートの型が使用されました。'
    ko = '배열 인덱싱에 지원되지 않는 유형이 사용되었습니다.'
    ru = 'Использован неподдерживаемый тип для индексации массива.'
    zh = '使用了不支持的类型来索引数组。'


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
