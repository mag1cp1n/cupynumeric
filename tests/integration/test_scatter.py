# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Tests targeting the advanced-indexing scatter pipeline."""

import numpy as np
import pytest

import cupynumeric as num
from cupynumeric.settings import settings


@pytest.mark.parametrize("n", [5, 500])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
class TestGeneralPathScatter:
    """Parametric coverage of the scatter fast path."""

    def test_two_1d_indices(self, n: int, dtype: np.dtype) -> None:
        rng = np.random.default_rng(42)
        # Use permutations so all (row, col) pairs are unique and
        # last-writer-wins ambiguity does not creep in.
        rows_np = rng.permutation(n)
        cols_np = rng.permutation(n)
        vals_np = rng.standard_normal(n).astype(dtype) * 100

        data_np = np.zeros((n, n), dtype=dtype)
        data_num = num.array(data_np)

        data_np[rows_np, cols_np] = vals_np
        data_num[num.array(rows_np), num.array(cols_np)] = num.array(vals_np)
        assert np.array_equal(data_np, data_num)

    def test_broadcast_indices(self, n: int, dtype: np.dtype) -> None:
        rng = np.random.default_rng(43)
        m = max(n // 5, 2)
        rows_np = rng.permutation(n)[:m].reshape(m, 1)
        cols_np = rng.permutation(n)[:m].reshape(1, m)
        vals_np = rng.standard_normal((m, m)).astype(dtype) * 100

        data_np = np.zeros((n, n), dtype=dtype)
        data_num = num.array(data_np)

        data_np[rows_np, cols_np] = vals_np
        data_num[num.array(rows_np), num.array(cols_np)] = num.array(vals_np)
        assert np.array_equal(data_np, data_num)

    def test_mixed_slice_and_arrays(self, n: int, dtype: np.dtype) -> None:
        rng = np.random.default_rng(44)
        k = max(n // 10, 2)
        idx0_np = rng.permutation(n)[:k]
        idx1_np = rng.permutation(k)
        vals_np = rng.standard_normal((k, n)).astype(dtype) * 100

        data_np = np.zeros((n, n, k), dtype=dtype)
        data_num = num.array(data_np)

        data_np[idx0_np, :, idx1_np] = vals_np
        data_num[num.array(idx0_np), :, num.array(idx1_np)] = num.array(
            vals_np
        )
        assert np.array_equal(data_np, data_num)

    def test_noncontiguous_sparse_indices(
        self, n: int, dtype: np.dtype
    ) -> None:
        rng = np.random.default_rng(45)
        k = max(n // 10, 2)
        idx0_np = rng.permutation(k)
        idx1_np = rng.permutation(k)
        vals_np = rng.standard_normal((k, n)).astype(dtype) * 100

        data_np = np.zeros((k, n, k), dtype=dtype)
        data_num = num.array(data_np)

        data_np[idx0_np, :, idx1_np] = vals_np
        data_num[num.array(idx0_np), :, num.array(idx1_np)] = num.array(
            vals_np
        )
        assert np.array_equal(data_np, data_num)

    def test_negative_indices(self, n: int, dtype: np.dtype) -> None:
        rng = np.random.default_rng(46)
        rows_np = rng.permutation(n).astype(np.int64)
        rows_np[: n // 2] -= n
        cols_np = rng.permutation(n).astype(np.int64)
        cols_np[n // 2 :] -= n
        vals_np = rng.standard_normal(n).astype(dtype) * 100

        data_np = np.zeros((n, n), dtype=dtype)
        data_num = num.array(data_np)

        data_np[rows_np, cols_np] = vals_np
        data_num[num.array(rows_np), num.array(cols_np)] = num.array(vals_np)
        assert np.array_equal(data_np, data_num)


def test_setitem_bool_mask_array_value() -> None:
    """Bool single-array key + non-scalar value: this combination routes
    through the scatter fast path on single-GPU."""
    rng = np.random.default_rng(47)
    arr_np = rng.standard_normal(20)
    arr_num = num.array(arr_np)
    mask_np = arr_np > 0
    mask_num = num.array(mask_np)
    n_true = int(mask_np.sum())
    vals_np = rng.standard_normal(n_true)
    vals_num = num.array(vals_np)

    arr_np[mask_np] = vals_np
    arr_num[mask_num] = vals_num
    assert np.array_equal(arr_np, arr_num)


def test_setitem_bool_mask_scalar_value() -> None:
    """Bool single-array + scalar value goes through putmask via the bool
    shortcut."""
    arr_np = np.arange(20, dtype=np.float64)
    arr_num = num.array(arr_np)
    mask_np = arr_np > 10
    mask_num = num.array(mask_np)

    arr_np[mask_np] = -1.0
    arr_num[mask_num] = -1.0
    assert np.array_equal(arr_np, arr_num)


def test_setitem_transformed_view_scalar() -> None:
    """``arr[1, cols] = scalar`` exercises the transformed-store + 0-D value path."""
    arr_np = np.arange(20, dtype=np.float64).reshape(4, 5)
    arr_num = num.array(arr_np)
    cols_np = np.array([0, 2, 4])
    cols_num = num.array(cols_np)

    arr_np[1, cols_np] = -1.0
    arr_num[1, cols_num] = -1.0
    assert np.array_equal(arr_np, arr_num)


def test_setitem_transformed_view_array() -> None:
    """``arr[:, cols] = array``: full-column update via scatter."""
    arr_np = np.arange(20, dtype=np.float64).reshape(4, 5)
    arr_num = num.array(arr_np)
    cols_np = np.array([0, 2, 4])
    vals_np = np.full((4, 3), -1.0)

    arr_np[:, cols_np] = vals_np
    arr_num[:, num.array(cols_np)] = num.array(vals_np)
    assert np.array_equal(arr_np, arr_num)


def test_setitem_empty_selection_nonempty_array() -> None:
    """All-False bool mask on a non-empty array."""
    arr_np = np.arange(10, dtype=np.float32)
    arr_num = num.array(arr_np)
    mask_np = np.zeros(10, dtype=bool)
    mask_num = num.array(mask_np)
    vals_np = np.array([], dtype=np.float32)
    vals_num = num.array(vals_np)

    arr_np[mask_np] = vals_np
    arr_num[mask_num] = vals_num
    assert np.array_equal(arr_np, arr_num)


@pytest.mark.skipif(
    not settings.bounds_check_enabled("indexing"),
    reason="indexing bounds checking is disabled in this environment",
)
def test_setitem_out_of_bounds_raises() -> None:
    """Bounds-check parity with the get side."""
    arr = num.zeros((3, 4), dtype=np.float32)
    idx = num.array([0, 999], dtype=np.int64)
    vals = num.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    with pytest.raises(IndexError):
        arr[idx, :] = vals


def test_setitem_zero_d_index_into_1d() -> None:
    """Edge case 0-D index + 0-D value -> empty ``out_shape == ()``."""
    arr_np = np.array([4, 3, 2, 1, 0])
    arr_num = num.array(arr_np)
    offset_num = num.arange(5)[3]
    arr_num[offset_num] = -1
    arr_np[3] = -1
    assert np.array_equal(arr_np, arr_num)


def test_setitem_negative_zero_d_index() -> None:
    """0-D negative index into a 1-D array; exercises index normalization."""
    arr_np = np.arange(5)
    arr_num = num.array(arr_np)
    offset_num = num.array(-2, dtype=np.int64)
    arr_num[offset_num] = 99
    arr_np[-2] = 99
    assert np.array_equal(arr_np, arr_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
