import pytest
import numpy as np
from typing import Any, TypeVar, Generic, Protocol
from samplex.util.uniq_list import (
    UniqueList,
    SxUniqValMismatchError,
    SxUniqValIdxMismatchError,
    SxUniqValUnusedError,
)

T = TypeVar('T')

class TestUniqueListBasics:
    """Test basic UniqueList functionality."""

    def test_empty_list(self):
        """Test creating an empty UniqueList."""
        ul = UniqueList([], np.array([], dtype=np.int64), {})
        assert isinstance(ul.uniq_vals, list)
        assert isinstance(ul.val_ids, np.ndarray)
        assert isinstance(ul.val2val_id, dict)
        assert len(ul.uniq_vals) == 0
        assert len(ul.val_ids) == 0
        assert len(ul.val2val_id) == 0

    def test_single_element(self):
        """Test UniqueList with a single element."""
        ul = UniqueList(
            uniq_vals=['a'],
            val_ids=np.array([0], dtype=np.int64),
            val2val_id={'a': 0},
            check=True
        )
        assert ul.uniq_vals == ['a']
        assert np.array_equal(ul.val_ids, np.array([0]))
        assert ul.val2val_id == {'a': 0}

    def test_repeated_elements(self):
        """Test UniqueList with repeated elements."""
        # Create from ['a', 'a', 'b', 'a', 'b', 'c']
        ul = UniqueList(
            uniq_vals=['a', 'b', 'c'],
            val_ids=np.array([0, 0, 1, 0, 1, 2], dtype=np.int64),
            val2val_id={'a': 0, 'b': 1, 'c': 2},
            check=True
        )
        assert ul.uniq_vals == ['a', 'b', 'c']
        assert np.array_equal(ul.val_ids, np.array([0, 0, 1, 0, 1, 2]))
        assert ul.val2val_id == {'a': 0, 'b': 1, 'c': 2}


class TestUniqueListFromVals:
    """Test UniqueList.from_vals constructor."""

    def test_empty_from_vals(self):
        """Test from_vals with empty list."""
        ul = UniqueList.from_vals([])
        assert ul.uniq_vals == []
        assert len(ul.val_ids) == 0
        assert ul.val2val_id == {}

    def test_single_from_vals(self):
        """Test from_vals with single element."""
        ul = UniqueList.from_vals(['x'])
        assert ul.uniq_vals == ['x']
        assert np.array_equal(ul.val_ids, np.array([0]))
        assert ul.val2val_id == {'x': 0}

    def test_duplicates_from_vals(self):
        """Test from_vals with duplicate elements."""
        input_vals = ['b', 'a', 'b', 'c', 'a']
        ul = UniqueList.from_vals(input_vals)
        assert ul.uniq_vals == ['a', 'b', 'c']  # Should be sorted
        assert len(ul.val_ids) == len(input_vals)
        # Verify each value maps to correct index
        expected_ids = np.array([1, 0, 1, 2, 0], dtype=np.int64)
        assert np.array_equal(ul.val_ids, expected_ids)

    def test_mixed_types_from_vals(self):
        """Test from_vals with comparable mixed type elements."""
        with pytest.raises(TypeError):
            UniqueList.from_vals([1, 'a'])  # Int and str aren't comparable

    @pytest.mark.parametrize("input_vals,expected_uniq", [
        ([], []),  # Empty list
        (['a'], ['a']),  # Single element
        (['a', 'a'], ['a']),  # Duplicates
        (['b', 'a'], ['a', 'b']),  # Sorting
        (['c', 'a', 'b', 'a', 'c'], ['a', 'b', 'c']),  # Complex case
    ])
    def test_from_vals_parametrized(self, input_vals, expected_uniq):
        """Parametrized tests for from_vals."""
        ul = UniqueList.from_vals(input_vals)
        assert ul.uniq_vals == expected_uniq
        assert len(ul.val_ids) == len(input_vals)
        assert len(ul.val2val_id) == len(expected_uniq)

        # Verify reconstruction
        reconstructed = [ul.uniq_vals[i] for i in ul.val_ids]
        assert reconstructed == input_vals


class TestUniqueListConcatenation:
    """Test UniqueList concatenation functionality."""

    def test_empty_concat(self):
        """Test concatenating empty UniqueLists."""
        ul1 = UniqueList.from_vals([])
        ul2 = UniqueList.from_vals([])
        result = UniqueList.cat([ul1, ul2])
        assert result.uniq_vals == []
        assert len(result.val_ids) == 0
        assert result.val2val_id == {}

    def test_single_concat(self):
        """Test concatenating with a single UniqueList."""
        ul = UniqueList.from_vals(['a', 'b'])
        result = UniqueList.cat([ul])
        assert result.uniq_vals == ul.uniq_vals
        assert np.array_equal(result.val_ids, ul.val_ids)
        assert result.val2val_id == ul.val2val_id

    def test_disjoint_concat(self):
        """Test concatenating UniqueLists with disjoint values."""
        ul1 = UniqueList.from_vals(['a', 'b'])
        ul2 = UniqueList.from_vals(['c', 'd'])
        result = UniqueList.cat([ul1, ul2])
        assert result.uniq_vals == ['a', 'b', 'c', 'd']
        assert len(result.val_ids) == 4

    def test_overlapping_concat(self):
        """Test concatenating UniqueLists with overlapping values."""
        ul1 = UniqueList.from_vals(['a', 'b', 'a'])
        ul2 = UniqueList.from_vals(['b', 'c', 'b'])
        result = UniqueList.cat([ul1, ul2])
        assert result.uniq_vals == ['a', 'b', 'c']
        assert len(result.val_ids) == 6

    def test_multiple_concat(self):
        """Test concatenating multiple UniqueLists."""
        ul1 = UniqueList.from_vals(['a'])
        ul2 = UniqueList.from_vals(['b'])
        ul3 = UniqueList.from_vals(['c'])
        result = UniqueList.cat([ul1, ul2, ul3])
        assert result.uniq_vals == ['a', 'b', 'c']
        assert len(result.val_ids) == 3

    def test_complex_concat(self):
        """Test complex concatenation scenario."""
        ul1 = UniqueList.from_vals(['b', 'a', 'b'])
        ul2 = UniqueList.from_vals(['c', 'b', 'a'])
        ul3 = UniqueList.from_vals(['a', 'd', 'd'])
        result = UniqueList.cat([ul1, ul2, ul3])
        assert result.uniq_vals == ['a', 'b', 'c', 'd']
        assert len(result.val_ids) == 9

        # Verify concatenation preserves order
        reconstructed = [result.uniq_vals[i] for i in result.val_ids]
        expected = ['b', 'a', 'b', 'c', 'b', 'a', 'a', 'd', 'd']
        assert reconstructed == expected


class TestUniqueListValidation:
    """Test UniqueList validation and error cases."""

    def test_uniq_val_mismatch(self):
        """Test detection of mismatched unique values."""
        with pytest.raises(SxUniqValMismatchError) as excinfo:
            UniqueList(
                uniq_vals=['a', 'b'],  # Contains 'b'
                val_ids=np.array([0, 1], dtype=np.int64),  # Valid indices for both values
                val2val_id={'c': 1},  # Has 'c' instead of 'b'
                check=True
            )

        # Verify the error contains the correct values
        assert sorted(excinfo.value.kwargs['uniq_vals_only']) == ['a', 'b']
        assert sorted(excinfo.value.kwargs['dict_keys_only']) == ['c']

    def test_uniq_val_idx_mismatch(self):
        """Test detection of mismatched value indices."""
        with pytest.raises(SxUniqValIdxMismatchError):
            UniqueList(
                uniq_vals=['a', 'b'],
                val_ids=np.array([0, 1], dtype=np.int64),
                val2val_id={'a': 0, 'b': 2},  # Index 2 is out of range
                check=True
            )

    def test_unused_values(self):
        """Test detection of unused unique values."""
        with pytest.raises(SxUniqValUnusedError):
            UniqueList(
                uniq_vals=['a', 'b', 'c'],
                val_ids=np.array([0, 0], dtype=np.int64),  # 'b' and 'c' never used
                val2val_id={'a': 0, 'b': 1, 'c': 2},
                check=True
            )

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        # This would raise errors with check=True
        ul = UniqueList(
            uniq_vals=['a', 'b'],
            val_ids=np.array([0], dtype=np.int64),
            val2val_id={'a': 0, 'c': 1},
            check=False
        )
        assert ul.uniq_vals == ['a', 'b']


class TestUniqueListEdgeCases:
    """Test edge cases and corner conditions."""

    def test_large_lists(self):
        """Test with large lists to check performance and memory."""
        size = 10000
        input_vals = ['val' + str(i % 100) for i in range(size)]
        ul = UniqueList.from_vals(input_vals)
        assert len(ul.uniq_vals) == 100
        assert len(ul.val_ids) == size

    def test_numeric_values(self):
        """Test with numeric values."""
        ul = UniqueList.from_vals([1, 2, 1, 3, 2])
        assert ul.uniq_vals == [1, 2, 3]
        reconstructed = [ul.uniq_vals[i] for i in ul.val_ids]
        assert reconstructed == [1, 2, 1, 3, 2]

    @pytest.mark.parametrize("input_vals", [
        [],  # Empty
        ['a'],  # Single
        ['a'] * 1000,  # Many duplicates
        list('abcdefghij'),  # No duplicates
        ['z', 'y', 'x'],  # Reverse order
    ])
    def test_various_inputs(self, input_vals):
        """Test various input patterns."""
        ul = UniqueList.from_vals(input_vals)
        reconstructed = [ul.uniq_vals[i] for i in ul.val_ids]
        assert reconstructed == input_vals

    def test_custom_comparable_type(self):
        """Test with a custom comparable type."""
        class ComparableObject:
            def __init__(self, value):
                self.value = value

            def __lt__(self, other):
                return self.value < other.value

            def __eq__(self, other):
                return self.value == other.value

            def __hash__(self):
                return hash(self.value)

        objects = [ComparableObject(1), ComparableObject(2), ComparableObject(1)]
        ul = UniqueList.from_vals(objects)
        assert len(ul.uniq_vals) == 2
        assert len(ul.val_ids) == 3


if __name__ == '__main__':
    pytest.main([__file__])
