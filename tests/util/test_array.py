from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, Union
import numpy as np
from numpy.typing import NDArray
import pytest

from samplex.util.array import Array


class MyArray(Array[int]):
    """Implementation class for testing Array."""

    def __init__(self, data: list[int]):
        self._data = data.copy()  # Defensive copy

    def __len__(self) -> int:
        return len(self._data)

    def __getoneitem__(self, idx: int) -> int:
        return self._data[idx]


class TestArray:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures before each test method"""
        self.empty_arr = MyArray([])
        self.single_arr = MyArray([42])
        self.arr = MyArray([0, 1, 2, 3, 4])
        self.negative_arr = MyArray([-2, -1, 0, 1, 2])
        self.repeated_arr = MyArray([1, 1, 2, 2, 3])

    # Basic Integer Indexing
    def test_basic_positive_index(self):
        assert self.arr[0] == 0
        assert self.arr[4] == 4
        assert self.negative_arr[0] == -2
        assert self.repeated_arr[0] == 1

    def test_basic_negative_index(self):
        assert self.arr[-1] == 4
        assert self.arr[-5] == 0
        assert self.negative_arr[-1] == 2
        assert self.repeated_arr[-1] == 3

    @pytest.mark.parametrize('idx', [5, 6, 100, -6, -7, -100])
    def test_index_out_of_bounds(self, idx):
        with pytest.raises(IndexError, match='Index out of range'):
            self.arr[idx]

    # Slice Tests
    @pytest.mark.parametrize(
        'arr',
        [
            MyArray([0, 1, 2, 3, 4]),
            MyArray([-2, -1, 0, 1, 2]),
            MyArray([1, 1, 2, 2, 3]),
        ],
    )
    def test_slice_full(self, arr):
        assert arr[:] == list(arr._data)
        assert arr[::] == list(arr._data)
        assert arr[::1] == list(arr._data)

    @pytest.mark.parametrize(
        'start,stop,step,expected',
        [
            (1, 4, None, [1, 2, 3]),
            (None, None, 2, [0, 2, 4]),
            (1, None, 2, [1, 3]),
            (None, 4, 2, [0, 2]),
            (1, -1, None, [1, 2, 3]),
            (-4, -1, None, [1, 2, 3]),
        ],
    )
    def test_slice_positive_step(self, start, stop, step, expected):
        result = self.arr[slice(start, stop, step)]
        assert result == expected

    @pytest.mark.parametrize(
        'start,stop,step,expected',
        [
            (None, None, -1, [4, 3, 2, 1, 0]),
            (3, 0, -1, [3, 2, 1]),
            (None, None, -2, [4, 2, 0]),
            (-1, -6, -1, [4, 3, 2, 1, 0]),
            (4, None, -2, [4, 2, 0]),
        ],
    )
    def test_slice_negative_step(self, start, stop, step, expected):
        result = self.arr[slice(start, stop, step)]
        assert result == expected

    @pytest.mark.parametrize(
        'slice_obj',
        [slice(1, 1), slice(4, 0), slice(0, 4, -1), slice(-1, -6, 1)],
    )
    def test_slice_empty(self, slice_obj):
        assert self.arr[slice_obj] == []

    @pytest.mark.parametrize(
        'slice_obj,expected',
        [
            (slice(-10, 10), [0, 1, 2, 3, 4]),
            (slice(10, -10, -1), [4, 3, 2, 1, 0]),
            (slice(-100, 100, 2), [0, 2, 4]),
        ],
    )
    def test_slice_oversized_bounds(self, slice_obj, expected):
        assert self.arr[slice_obj] == expected

    def test_slice_zero_step(self):
        with pytest.raises(ValueError, match='Slice step cannot be zero: '):
            self.arr[::0]

    def test_slice_none_values(self):
        assert self.arr[None:None:None] == [0, 1, 2, 3, 4]
        assert self.arr[None:None:-1] == [4, 3, 2, 1, 0]
        assert self.arr[None:None:2] == [0, 2, 4]

    # Sequence Indexing Tests
    @pytest.mark.parametrize(
        'sequence,expected',
        [
            ([1, 3], [1, 3]),
            ((1, 3), (1, 3)),
            ([], []),
            ((), ()),
            ([1, [2, 3], (4,)], [1, [2, 3], (4,)]),
            ([1, [2, [3]], ((4,),)], [1, [2, [3]], ((4,),)]),
        ],
    )
    def test_sequence_indexing(self, sequence, expected):
        assert self.arr[sequence] == expected

    # Numpy Array Tests
    def test_numpy_scalar(self):
        for dtype in [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]:
            assert self.arr[np.array(1, dtype=dtype)] == 1

    @pytest.mark.parametrize(
        'indices,expected',
        [([1, 3], [1, 3]), ([0, 4], [0, 4]), ([4, 0], [4, 0]), ([], [])],
    )
    def test_numpy_1d(self, indices, expected):
        assert self.arr[np.array(indices)] == expected

    def test_numpy_2d(self):
        indices = np.array([[1], [3]])
        assert self.arr[indices] == [[1], [3]]

        indices = np.array([[1, 2], [3, 4]])
        assert self.arr[indices] == [[1, 2], [3, 4]]

    @pytest.mark.parametrize(
        'mask,expected',
        [
            ([True, False, True, False, True], [0, 2, 4]),
            ([True] * 5, [0, 1, 2, 3, 4]),
            ([False] * 5, []),
        ],
    )
    def test_numpy_bool_mask(self, mask, expected):
        assert self.arr[np.array(mask)] == expected

    def test_numpy_bool_mask_errors(self):
        # Wrong length
        with pytest.raises(
            IndexError, match='Boolean mask length must match array length'
        ):
            self.arr[np.array([True, False])]

        # Wrong shape
        with pytest.raises(
            IndexError, match='Boolean mask must be 1-dimensional'
        ):
            self.arr[np.array([[True], [False]])]

    def test_numpy_empty_arrays(self):
        # 1D empty array
        assert self.arr[np.array([])] == []
        # 2D empty array
        assert self.arr[np.array([[]], dtype=int)] == [[]]
        # Multiple empty rows
        assert self.arr[np.array([[], []], dtype=int)] == [[], []]
        # 3D empty array
        assert self.arr[np.array([[[]]], dtype=int)] == [[[]]]

    def test_numpy_wrong_dtype(self):
        invalid_dtypes = [
            np.array([1.0, 2.0]),  # float
            np.array(['1', '2']),  # string
            np.array([1 + 2j, 3 + 4j]),  # complex
        ]
        for arr in invalid_dtypes:
            with pytest.raises(
                TypeError, match='must have integer or boolean dtype'
            ):
                self.arr[arr]

    # Special Cases
    def test_ellipsis(self):
        assert self.arr[...] == [0, 1, 2, 3, 4]
        assert self.empty_arr[...] == []
        assert self.single_arr[...] == [42]

    def test_empty_array_behavior(self):
        assert self.empty_arr[:] == []
        assert self.empty_arr[[]] == []
        assert self.empty_arr[...] == []
        with pytest.raises(IndexError):
            self.empty_arr[0]
        with pytest.raises(IndexError):
            self.empty_arr[-1]

    def test_single_element_array(self):
        assert self.single_arr[:] == [42]
        assert self.single_arr[0] == 42
        assert self.single_arr[-1] == 42
        assert self.single_arr[...] == [42]
        with pytest.raises(IndexError):
            self.single_arr[1]
        with pytest.raises(IndexError):
            self.single_arr[-2]

    @pytest.mark.parametrize(
        'invalid_index',
        ['invalid', 1.5, {1, 2, 3}, complex(1, 2), object(), lambda x: x],
    )
    def test_invalid_index_types(self, invalid_index):
        with pytest.raises(TypeError):
            self.arr[invalid_index]

    # Additional Edge Cases
    def test_nested_slice_behavior(self):
        nested = [slice(1, 3), slice(None, None, -1)]
        assert self.arr[nested] == [[1, 2], [4, 3, 2, 1, 0]]

    def test_mixed_indexing(self):
        mixed = [1, slice(2, 4), np.array([4]), ...]
        assert self.arr[mixed] == [1, [2, 3], [4], [0, 1, 2, 3, 4]]


if __name__ == '__main__':
    pytest.main([__file__])
