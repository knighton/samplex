from abc import ABC
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, Union
import numpy as np
from numpy.typing import NDArray
import pytest

from samplex.util.array import Array
from samplex.util.array import SxIndexOutOfRangeError
from samplex.util.array import SxBoolMaskLenError
from samplex.util.array import SxBoolMaskNdimError
from samplex.util.array import SxSliceStepError
from samplex.util.array import SxVecDtypeError
from samplex.util.array import SxPointDtypeError
from samplex.util.array import SxIndexTypeError


class MyArray(Array[int]):
    """Implementation class for testing Array."""

    def __init__(self, data: list[int]):
        self._data = data.copy()

    def __len__(self) -> int:
        return len(self._data)

    def __getoneitem__(self, idx: int) -> int:
        return self._data[idx]


class TestArrayBasicIndexing:
    """Test basic integer indexing behaviors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.empty_arr = MyArray([])
        self.single_arr = MyArray([42])
        self.arr = MyArray([0, 1, 2, 3, 4])
        self.negative_arr = MyArray([-2, -1, 0, 1, 2])
        self.repeated_arr = MyArray([1, 1, 2, 2, 3])

    def test_positive_indexing(self):
        """Test positive index access."""
        test_cases = [
            (self.arr, 0, 0),
            (self.arr, 4, 4),
            (self.negative_arr, 0, -2),
            (self.negative_arr, 4, 2),
            (self.repeated_arr, 0, 1),
            (self.repeated_arr, 4, 3),
        ]
        for arr, idx, expected in test_cases:
            assert arr[idx] == expected

    def test_negative_indexing(self):
        """Test negative index access."""
        test_cases = [
            (self.arr, -1, 4),
            (self.arr, -5, 0),
            (self.negative_arr, -1, 2),
            (self.negative_arr, -5, -2),
            (self.repeated_arr, -1, 3),
            (self.repeated_arr, -5, 1),
        ]
        for arr, idx, expected in test_cases:
            assert arr[idx] == expected

    @pytest.mark.parametrize('arr,idx,length', [
        (MyArray([]), 0, 0),
        (MyArray([]), -1, 0),
        (MyArray([42]), 1, 1),
        (MyArray([42]), -2, 1),
        (MyArray([0, 1, 2]), 3, 3),
        (MyArray([0, 1, 2]), -4, 3),
        (MyArray(list(range(5))), 5, 5),
        (MyArray(list(range(5))), -6, 5),
    ])
    def test_index_out_of_bounds(self, arr, idx, length):
        """Test out of bounds index access."""
        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            arr[idx]
        assert excinfo.value.kwargs['index'] == idx
        assert excinfo.value.kwargs['length'] == length


class TestArraySlicing:
    """Test array slicing behaviors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.arr = MyArray(list(range(5)))

    @pytest.mark.parametrize('slice_obj,expected', [
        (slice(None), [0, 1, 2, 3, 4]),
        (slice(2), [0, 1]),
        (slice(1, 3), [1, 2]),
        (slice(None, None, 2), [0, 2, 4]),
        (slice(-3, None), [2, 3, 4]),
        (slice(None, -2), [0, 1, 2]),
        (slice(-4, -1), [1, 2, 3]),
        (slice(None, None, -1), [4, 3, 2, 1, 0]),
        (slice(3, 1, -1), [3, 2]),
        (slice(-2, -4, -1), [3, 2]),
    ])
    def test_valid_slices(self, slice_obj, expected):
        """Test various valid slice operations."""
        assert self.arr[slice_obj] == expected

    def test_empty_slices(self):
        """Test slices that result in empty lists."""
        empty_cases = [
            # Empty slices with positive step:
            slice(0, 0),  # Empty at start
            slice(1, 1),  # Empty in middle
            slice(5, 5),  # Empty at end
            slice(5, 1),  # Start > stop
            slice(-1, -2),  # Negative indices, start > stop

            # Empty slices with negative step:
            slice(0, 1, -1),  # Start < stop
            slice(2, 3, -1),  # Start < stop in middle
            slice(-2, -1, -1),  # Negative indices, start < stop
        ]
        for s in empty_cases:
            assert self.arr[s] == [], f"Slice {s} should produce empty list"

    def test_slice_with_zero_step(self):
        """Test that slice with step=0 raises appropriate error."""
        with pytest.raises(SxSliceStepError) as excinfo:
            self.arr[::0]
        assert excinfo.value.kwargs['slice'] == slice(None, None, 0)

    def test_oversized_slice_indices(self):
        """Test slices with out-of-bounds indices."""
        test_cases = [
            (slice(-10, 10), [0, 1, 2, 3, 4]),
            (slice(-10, 10, 2), [0, 2, 4]),
            (slice(10, -10, -1), [4, 3, 2, 1, 0]),
            (slice(100, None, -2), [4, 2, 0]),
        ]
        for s, expected in test_cases:
            assert self.arr[s] == expected


class TestArrayNumpyIndexing:
    """Test numpy array indexing behaviors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.arr = MyArray(list(range(5)))

    def test_scalar_integer_dtypes(self):
        """Test numpy scalar integers of various dtypes."""
        dtypes = [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
        ]
        for dtype in dtypes:
            assert self.arr[np.array(2, dtype=dtype)] == 2

    def test_scalar_non_integer_dtypes(self):
        """Test numpy scalar arrays with invalid dtypes."""
        invalid_scalars = [
            np.array(1.0),
            np.array('1'),
            np.array(True),
            np.array(1 + 2j),
        ]
        for scalar in invalid_scalars:
            with pytest.raises(SxPointDtypeError) as excinfo:
                self.arr[scalar]
            assert excinfo.value.kwargs['dtype'] == scalar.dtype

    def test_1d_integer_arrays(self):
        """Test 1D numpy integer arrays."""
        test_cases = [
            (np.array([1, 3], dtype=np.int32), [1, 3]),
            (np.array([4, 0], dtype=np.uint8), [4, 0]),
            (np.array([], dtype=np.int64), []),
        ]
        for arr, expected in test_cases:
            assert self.arr[arr] == expected

    def test_2d_integer_arrays(self):
        """Test 2D numpy integer arrays."""
        test_cases = [
            (np.array([[1], [3]]), [[1], [3]]),
            (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
            (np.array([[]], dtype=int), [[]]),
        ]
        for arr, expected in test_cases:
            assert self.arr[arr] == expected

    def test_boolean_masks(self):
        """Test boolean mask indexing."""
        test_cases = [
            ([True, False, True, False, True], [0, 2, 4]),
            ([True] * 5, [0, 1, 2, 3, 4]),
            ([False] * 5, []),
        ]
        for mask, expected in test_cases:
            assert self.arr[np.array(mask)] == expected

    def test_boolean_mask_errors(self):
        """Test error cases for boolean masks."""
        # Wrong length
        with pytest.raises(SxBoolMaskLenError) as excinfo:
            self.arr[np.array([True, False])]
        assert excinfo.value.kwargs['mask_len'] == 2
        assert excinfo.value.kwargs['array_len'] == 5

        # Wrong dimensionality
        shapes = [(2, 1), (2, 2), (1, 2, 1)]
        for shape in shapes:
            with pytest.raises(SxBoolMaskNdimError) as excinfo:
                self.arr[np.zeros(shape, dtype=bool)]
            assert excinfo.value.kwargs['shape'] == shape

    def test_invalid_dtypes(self):
        """Test numpy arrays with invalid dtypes."""
        invalid_arrays = [
            np.array([1.0, 2.0]),
            np.array(['1', '2']),
            np.array([1 + 2j, 3 + 4j]),
            np.array([[1.0], [2.0]]),
        ]
        for arr in invalid_arrays:
            with pytest.raises(SxVecDtypeError) as excinfo:
                self.arr[arr]
            assert excinfo.value.kwargs['dtype'] == arr.dtype


class TestArrayMiscIndexing:
    """Test miscellaneous indexing behaviors."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.arr = MyArray(list(range(5)))

    def test_ellipsis(self):
        """Test Ellipsis indexing."""
        assert self.arr[...] == list(range(5))
        assert MyArray([])[...] == []
        assert MyArray([42])[...] == [42]

    def test_sequences(self):
        """Test sequence indexing."""
        test_cases = [
            ([1, 3], [1, 3]),
            ((1, 3), (1, 3)),
            ([], []),
            ((), ()),
            ([1, [2, 3], (4,)], [1, [2, 3], (4,)]),
        ]
        for sequence, expected in test_cases:
            assert self.arr[sequence] == expected

    @pytest.mark.parametrize('invalid_index', [
        'str',
        1.5,
        {1, 2, 3},
        complex(1, 2),
        object(),
        lambda x: x,
    ])
    def test_invalid_index_types(self, invalid_index):
        """Test invalid index types."""
        with pytest.raises(SxIndexTypeError) as excinfo:
            self.arr[invalid_index]
        assert excinfo.value.kwargs['type'] == type(invalid_index)


class TestArrayEdgeCases:
    """Test array edge cases and corner behaviors."""

    def test_empty_array(self):
        """Test empty array behaviors."""
        arr = MyArray([])
        assert len(arr) == 0
        assert arr[:] == []
        assert arr[...] == []
        assert arr[[]] == []
        assert arr[()] == ()
        assert arr[np.array([], dtype=int)] == []

        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            arr[0]
        assert excinfo.value.kwargs == {'index': 0, 'length': 0}

    def test_single_element_array(self):
        """Test single-element array behaviors."""
        arr = MyArray([42])
        assert len(arr) == 1
        assert arr[:] == [42]
        assert arr[0] == 42
        assert arr[-1] == 42
        assert arr[...] == [42]
        assert arr[[0]] == [42]
        assert arr[(0,)] == (42,)
        assert arr[np.array([0])] == [42]

        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            arr[1]
        assert excinfo.value.kwargs == {'index': 1, 'length': 1}

    def test_nested_indices(self):
        """Test nested and complex indexing patterns."""
        arr = MyArray(list(range(5)))
        test_cases = [
            ([slice(1, 3), slice(None, None, -1)], [[1, 2], [4, 3, 2, 1, 0]]),
            ([np.array([1, 2]), [3, 4]], [[1, 2], [3, 4]]),
            ([[1], np.array([[2]]), (3,)], [[1], [[2]], (3,)]),
        ]
        for index, expected in test_cases:
            assert arr[index] == expected


if __name__ == '__main__':
    pytest.main([__file__])
