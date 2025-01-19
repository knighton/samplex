import numpy as np
import pytest

from samplex.util.range_arr import (
    RangeArray,
    SxIndexOutOfRangeError as MySxIndexOutOfRangeError,
)
from samplex.util.array import (
    SxIndexOutOfRangeError,
    SxBoolMaskLenError,
    SxBoolMaskNdimError,
    SxSliceStepError,
    SxVecDtypeError,
    SxPointDtypeError,
    SxIndexTypeError,
)


class TestRangeArrayBasics:
    """Test basic RangeArray functionality."""

    @pytest.fixture
    def empty_array(self):
        """Create an empty RangeArray."""
        return RangeArray(np.array([], dtype=np.int64))

    @pytest.fixture
    def single_array(self):
        """Create a RangeArray with one range."""
        return RangeArray(np.array([5], dtype=np.int64))

    @pytest.fixture
    def multi_array(self):
        """Create a RangeArray with multiple ranges."""
        return RangeArray(np.array([3, 7, 10], dtype=np.int64))

    def test_empty_array(self, empty_array):
        """Test empty RangeArray properties."""
        assert len(empty_array) == 0
        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            empty_array[0]
        assert excinfo.value.kwargs == {'index': 0, 'length': 0}

    def test_single_array(self, single_array):
        """Test RangeArray with single range."""
        assert len(single_array) == 1
        assert single_array[0] == (0, 5)
        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            single_array[1]
        assert excinfo.value.kwargs == {'index': 1, 'length': 1}

    def test_getoneitem_direct(self, multi_array):
        """Test __getoneitem__ method directly."""
        assert multi_array.__getoneitem__(0) == (0, 3)
        assert multi_array.__getoneitem__(1) == (3, 7)
        assert multi_array.__getoneitem__(2) == (7, 10)

        # Test that out of range indices raise SxIndexOutOfRangeError
        with pytest.raises(MySxIndexOutOfRangeError) as excinfo:
            multi_array.__getoneitem__(3)
        assert excinfo.value.kwargs == {'range_id': 3, 'num_ranges': 3}

        with pytest.raises(MySxIndexOutOfRangeError) as excinfo:
            multi_array.__getoneitem__(-1)
        assert excinfo.value.kwargs == {'range_id': -1, 'num_ranges': 3}

    def test_getitem_vs_getoneitem(self, multi_array):
        """Compare behavior of __getitem__ vs __getoneitem__."""
        # __getitem__ should handle negative indices
        assert multi_array[-1] == multi_array.__getoneitem__(2)
        assert multi_array[-2] == multi_array.__getoneitem__(1)
        assert multi_array[-3] == multi_array.__getoneitem__(0)

        # __getitem__ should raise IndexOutOfRange
        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            multi_array[3]
        assert excinfo.value.kwargs == {'index': 3, 'length': 3}

        # While __getoneitem__ raises SxIndexOutOfRangeError
        with pytest.raises(MySxIndexOutOfRangeError) as excinfo:
            multi_array.__getoneitem__(3)
        assert excinfo.value.kwargs == {'range_id': 3, 'num_ranges': 3}


class TestRangeArrayIndexing:
    """Test RangeArray indexing behaviors."""

    @pytest.fixture
    def array(self):
        """Create a test RangeArray."""
        return RangeArray(np.array([5, 8, 12, 15, 20], dtype=np.int64))

    def test_positive_indexing(self, array):
        """Test positive index access."""
        expected = [(0, 5), (5, 8), (8, 12), (12, 15), (15, 20)]
        for i, exp in enumerate(expected):
            assert array[i] == exp

    def test_negative_indexing(self, array):
        """Test negative index access."""
        expected = [(0, 5), (5, 8), (8, 12), (12, 15), (15, 20)]
        for i in range(-5, 0):
            assert array[i] == expected[i]

    def test_out_of_bounds(self, array):
        """Test out of bounds index access."""
        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            array[5]
        assert excinfo.value.kwargs == {'index': 5, 'length': 5}

        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            array[-6]
        assert excinfo.value.kwargs == {'index': -6, 'length': 5}

    def test_slice_indexing(self, array):
        """Test slice indexing."""
        assert array[:2] == [(0, 5), (5, 8)]
        assert array[2:4] == [(8, 12), (12, 15)]
        assert array[::2] == [(0, 5), (8, 12), (15, 20)]
        assert array[::-1] == [(15, 20), (12, 15), (8, 12), (5, 8), (0, 5)]

    def test_empty_slices(self, array):
        """Test empty slice results."""
        assert array[2:1] == []
        assert array[-1:-2] == []
        assert array[5:] == []
        assert array[:0] == []

    def test_slice_with_zero_step(self, array):
        """Test slice with zero step."""
        with pytest.raises(SxSliceStepError) as excinfo:
            array[::0]
        assert excinfo.value.kwargs == {'slice': slice(None, None, 0)}


class TestRangeArrayNumpyIndexing:
    """Test numpy array indexing behaviors."""

    @pytest.fixture
    def array(self):
        """Create a test RangeArray."""
        return RangeArray(np.array([3, 7, 10, 15], dtype=np.int64))

    def test_scalar_integer_array(self, array):
        """Test numpy scalar integer indexing."""
        assert array[np.array(2)] == (7, 10)

    def test_scalar_non_integer_array(self, array):
        """Test numpy scalar non-integer indexing."""
        with pytest.raises(SxPointDtypeError) as excinfo:
            array[np.array(1.0)]
        assert excinfo.value.kwargs['dtype'] == np.dtype('float64')

    def test_1d_integer_array(self, array):
        """Test 1D numpy integer array indexing."""
        indices = np.array([1, 3])
        expected = [(3, 7), (10, 15)]
        assert array[indices] == expected

    def test_boolean_mask(self, array):
        """Test boolean mask indexing."""
        mask = np.array([True, False, True, False])
        expected = [(0, 3), (7, 10)]
        assert array[mask] == expected

    def test_boolean_mask_wrong_length(self, array):
        """Test boolean mask with wrong length."""
        mask = np.array([True, False])
        with pytest.raises(SxBoolMaskLenError) as excinfo:
            array[mask]
        assert excinfo.value.kwargs == {'mask_len': 2, 'array_len': 4}

    def test_boolean_mask_wrong_ndim(self, array):
        """Test boolean mask with wrong dimensions."""
        mask = np.array([[True, False], [False, True]])
        with pytest.raises(SxBoolMaskNdimError) as excinfo:
            array[mask]
        assert excinfo.value.kwargs == {'shape': (2, 2)}


class TestRangeArrayEdgeCases:
    """Test RangeArray edge cases."""

    def test_zero_length_ranges(self):
        """Test RangeArray with zero-length ranges."""
        array = RangeArray(np.array([0, 0, 5, 5], dtype=np.int64))
        assert array[0] == (0, 0)
        assert array[1] == (0, 0)
        assert array[2] == (0, 5)
        assert array[3] == (5, 5)

    def test_large_indices(self):
        """Test RangeArray with large indices."""
        large_nums = np.array([1000000, 2000000, 3000000], dtype=np.int64)
        array = RangeArray(large_nums)
        assert array[0] == (0, 1000000)
        assert array[1] == (1000000, 2000000)
        assert array[2] == (2000000, 3000000)

    def test_ellipsis_indexing(self):
        """Test ellipsis indexing."""
        array = RangeArray(np.array([3, 7], dtype=np.int64))
        assert array[...] == [(0, 3), (3, 7)]

    @pytest.mark.parametrize('invalid_index', [
        1.5,
        'string',
        complex(1, 2),
        {1, 2, 3},
        lambda x: x,
    ])
    def test_invalid_index_types(self, invalid_index):
        """Test invalid index types."""
        array = RangeArray(np.array([5], dtype=np.int64))
        with pytest.raises(SxIndexTypeError) as excinfo:
            array[invalid_index]
        assert excinfo.value.kwargs['type'] == type(invalid_index)


if __name__ == '__main__':
    pytest.main([__file__])
