import numpy as np
import pytest

from samplex.util.range_idx import (
    RangeIndex,
    SxRangeIndexInitNoRangesError,
    SxRangeIndexInitBadRangesError,
    SxIndexOutOfRangeError,
    SxCatNoRangeIndexesError,
)


class TestRangeIndexInitialization:
    """Test RangeIndex initialization and validation."""

    def test_valid_initialization(self):
        """Test valid RangeIndex initialization cases."""
        # Single range
        ri = RangeIndex(np.array([5], dtype=np.int64))
        assert ri.num_ranges == 1
        assert ri.num_items == 5

        # Multiple ranges
        ri = RangeIndex(np.array([3, 7, 10], dtype=np.int64))
        assert ri.num_ranges == 3
        assert ri.num_items == 10

        # Initialize with check=False
        ri = RangeIndex(np.array([3, 7, 10], dtype=np.int64), check=False)
        assert ri.num_ranges == 3
        assert ri.num_items == 10

    def test_empty_initialization(self):
        """Test initialization with empty array."""
        with pytest.raises(SxRangeIndexInitNoRangesError) as excinfo:
            RangeIndex(np.array([], dtype=np.int64))
        assert 'tails' in excinfo.value.kwargs

    def test_invalid_ranges(self):
        """Test initialization with invalid range sizes."""
        # Zero-length range
        with pytest.raises(SxRangeIndexInitBadRangesError) as excinfo:
            RangeIndex(np.array([5, 5, 10], dtype=np.int64))
        assert excinfo.value.kwargs['num_zero_len_ranges'] == 1

        # Negative length range
        with pytest.raises(SxRangeIndexInitBadRangesError) as excinfo:
            RangeIndex(np.array([5, 3, 10], dtype=np.int64))
        assert excinfo.value.kwargs['num_neg_len_ranges'] == 1

        # Multiple invalid ranges
        with pytest.raises(SxRangeIndexInitBadRangesError) as excinfo:
            RangeIndex(np.array([5, 5, 4, 10], dtype=np.int64))
        assert excinfo.value.kwargs['num_zero_len_ranges'] == 1
        assert excinfo.value.kwargs['num_neg_len_ranges'] == 1


class TestRangeIndexFromLens:
    """Test RangeIndex.from_lens constructor."""

    def test_valid_lens(self):
        """Test from_lens with valid lengths."""
        ri = RangeIndex.from_lens(np.array([2, 4, 3], dtype=np.int64))
        assert ri.num_ranges == 3
        assert ri.num_items == 9
        assert np.array_equal(ri.tails, np.array([2, 6, 9], dtype=np.int64))

    def test_single_len(self):
        """Test from_lens with single length."""
        ri = RangeIndex.from_lens(np.array([5], dtype=np.int64))
        assert ri.num_ranges == 1
        assert ri.num_items == 5
        assert np.array_equal(ri.tails, np.array([5], dtype=np.int64))

    def test_zero_lens(self):
        """Test from_lens with zero lengths."""
        with pytest.raises(SxRangeIndexInitBadRangesError):
            RangeIndex.from_lens(np.array([2, 0, 3], dtype=np.int64))


class TestRangeIndexAccess:
    """Test RangeIndex item access."""

    @pytest.fixture
    def range_index(self):
        return RangeIndex(np.array([3, 7, 10], dtype=np.int64))

    def test_valid_access(self, range_index):
        """Test valid item access."""
        # First range
        assert range_index[0] == (0, 0)
        assert range_index[1] == (0, 1)
        assert range_index[2] == (0, 2)

        # Second range
        assert range_index[3] == (1, 0)
        assert range_index[6] == (1, 3)

        # Third range
        assert range_index[7] == (2, 0)
        assert range_index[9] == (2, 2)

    def test_out_of_bounds(self, range_index):
        """Test out of bounds access."""
        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            range_index.__getoneitem__(10)
        assert excinfo.value.kwargs['item_id'] == 10
        assert excinfo.value.kwargs['num_ranges'] == 3

        with pytest.raises(SxIndexOutOfRangeError) as excinfo:
            range_index.__getoneitem__(-1)
        assert excinfo.value.kwargs['item_id'] == -1
        assert excinfo.value.kwargs['num_ranges'] == 3


class TestRangeIndexConcatenation:
    """Test RangeIndex concatenation."""

    def test_valid_concatenation(self):
        """Test valid concatenation cases."""
        ri1 = RangeIndex(np.array([3, 5], dtype=np.int64))
        ri2 = RangeIndex(np.array([2, 4], dtype=np.int64))
        ri3 = RangeIndex(np.array([3], dtype=np.int64))

        result = RangeIndex.cat([ri1, ri2, ri3])
        assert result.num_ranges == 5
        assert result.num_items == 12
        assert np.array_equal(
            result.tails,
            np.array([3, 5, 7, 9, 12], dtype=np.int64)
        )

    def test_empty_concatenation(self):
        """Test concatenation with empty list."""
        with pytest.raises(SxCatNoRangeIndexesError) as excinfo:
            RangeIndex.cat([])
        assert 'objs' in excinfo.value.kwargs

    def test_single_concatenation(self):
        """Test concatenation with single RangeIndex."""
        ri = RangeIndex(np.array([3, 5], dtype=np.int64))
        result = RangeIndex.cat([ri])
        assert result.num_ranges == 2
        assert result.num_items == 5
        assert np.array_equal(result.tails, ri.tails)


class TestRangeIndexEachRangeBegin:
    """Test RangeIndex.each_range_begin method."""

    def test_valid_each_range_begin(self):
        """Test valid each_range_begin cases."""
        ri1 = RangeIndex(np.array([2, 4], dtype=np.int64))
        ri2 = RangeIndex(np.array([2, 4], dtype=np.int64))

        # Should yield indices where any range_id changes
        result = list(RangeIndex.each_range_begin([ri1, ri2]))
        assert result == [0, 2]

    def test_empty_each_range_begin(self):
        """Test each_range_begin with empty list."""
        assert list(RangeIndex.each_range_begin([])) == []

    def test_single_each_range_begin(self):
        """Test each_range_begin with single RangeIndex."""
        ri = RangeIndex(np.array([2, 4, 6], dtype=np.int64))
        result = list(RangeIndex.each_range_begin([ri]))
        assert result == [0, 2, 4]

    def test_mismatched_lengths(self):
        """Test each_range_begin with mismatched lengths."""
        ri1 = RangeIndex(np.array([2, 4], dtype=np.int64))
        ri2 = RangeIndex(np.array([3, 5], dtype=np.int64))

        with pytest.raises(Exception):  # Should raise some kind of error
            list(RangeIndex.each_range_begin([ri1, ri2]))


class TestRangeIndexEdgeCases:
    """Test RangeIndex edge cases."""

    def test_large_ranges(self):
        """Test with large range values."""
        large_vals = np.array([1000000, 2000000, 3000000], dtype=np.int64)
        ri = RangeIndex(large_vals)
        assert ri.num_ranges == 3
        assert ri.num_items == 3000000

        # Test access within large ranges
        assert ri[1000000] == (1, 0)
        assert ri[2999999] == (2, 999999)

    def test_many_ranges(self):
        """Test with many small ranges."""
        # Create 1000 ranges of size 1
        lens = np.ones(1000, dtype=np.int64)
        tails = np.cumsum(lens)
        ri = RangeIndex(tails)

        assert ri.num_ranges == 1000
        assert ri.num_items == 1000

        # Test access across many ranges
        assert ri[0] == (0, 0)
        assert ri[500] == (500, 0)
        assert ri[999] == (999, 0)


if __name__ == '__main__':
    pytest.main([__file__])
