/* Copyright 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <algorithm>
#include <vector>

// Useful for IDEs
#include "cupynumeric/matrix/solve.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, Type::Code CODE>
struct SolveImplBody;

template <Type::Code CODE>
struct support_solve : std::false_type {};
template <>
struct support_solve<Type::Code::FLOAT64> : std::true_type {};
template <>
struct support_solve<Type::Code::FLOAT32> : std::true_type {};
template <>
struct support_solve<Type::Code::COMPLEX64> : std::true_type {};
template <>
struct support_solve<Type::Code::COMPLEX128> : std::true_type {};

template <VariantKind KIND>
struct SolveImpl {
  TaskContext context;
  explicit SolveImpl(TaskContext context) : context(context) {}

  template <Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<support_solve<CODE>::value && DIM >= 2>* = nullptr>
  void operator()(legate::PhysicalStore a_array,
                  legate::PhysicalStore b_array,
                  legate::PhysicalStore x_array) const
  {
    using VAL = type_of<CODE>;

#ifdef DEBUG_CUPYNUMERIC
    assert(a_array.dim() >= 2);
    assert(a_array.dim() == DIM);
    assert(b_array.dim() == DIM);
    assert(x_array.dim() == DIM);
#endif
    const auto a_shape = a_array.shape<DIM>();
    const auto b_shape = b_array.shape<DIM>();
    const auto x_shape = x_array.shape<DIM>();

    // Note that a, b and x may report different extents on the batch
    // dimensions: when one of them was broadcast/promoted along a batch dim,
    // the store cannot observe the partitioning along that fake dim and
    // reports the full unpartitioned extent instead. Recover the real
    // per-task working set on the batch dims by intersecting along those
    // dims only (the trailing two dims differ between the stores: a is
    // (m, n) while b and x are (m, nrhs)).
    Rect<DIM> a_active = a_shape;
    Rect<DIM> b_active = b_shape;
    Rect<DIM> x_active = x_shape;
    for (int32_t i = 0; i < DIM - 2; ++i) {
      const auto lo  = std::max({a_shape.lo[i], b_shape.lo[i], x_shape.lo[i]});
      const auto hi  = std::min({a_shape.hi[i], b_shape.hi[i], x_shape.hi[i]});
      a_active.lo[i] = lo;
      a_active.hi[i] = hi;
      b_active.lo[i] = lo;
      b_active.hi[i] = hi;
      x_active.lo[i] = lo;
      x_active.hi[i] = hi;
    }

    if (a_active.empty()) {
      return;
    }

    const int64_t m    = a_active.hi[DIM - 2] - a_active.lo[DIM - 2] + 1;
    const int64_t n    = a_active.hi[DIM - 1] - a_active.lo[DIM - 1] + 1;
    const int64_t nrhs = b_active.hi[DIM - 1] - b_active.lo[DIM - 1] + 1;

    int64_t batchsize_total = 1;
    std::vector<int64_t> batchdims;
    for (auto i = 0; i < DIM - 2; ++i) {
      batchdims.push_back(a_active.hi[i] - a_active.lo[i] + 1);
      batchsize_total *= batchdims.back();
    }

#ifdef DEBUG_CUPYNUMERIC
    assert(m > 0);
    assert(nrhs > 0);
    assert(m == n);
    assert(batchsize_total > 0);
    assert(b_active.hi[DIM - 2] - b_active.lo[DIM - 2] + 1 == m);
    assert(x_active.hi[DIM - 2] - x_active.lo[DIM - 2] + 1 == m);
    assert(x_active.hi[DIM - 1] - x_active.lo[DIM - 1] + 1 == nrhs);
    for (auto i = 0; i < batchdims.size(); ++i) {
      assert(b_active.hi[i] - b_active.lo[i] + 1 == batchdims[i]);
      assert(x_active.hi[i] - x_active.lo[i] + 1 == batchdims[i]);
    }
#endif

    size_t a_strides[DIM];
    size_t b_strides[DIM];
    size_t x_strides[DIM];

    auto* a = a_array.read_accessor<VAL, DIM>(a_active).ptr(a_active, a_strides);
    auto* b = b_array.read_accessor<VAL, DIM>(b_active).ptr(b_active, b_strides);
    auto* x = x_array.write_accessor<VAL, DIM>(x_active).ptr(x_active, x_strides);

#ifdef DEBUG_CUPYNUMERIC
    // per matrix col-major
    assert(a_array.is_future() ||
           (a_strides[DIM - 2] == 1 && static_cast<int64_t>(a_strides[DIM - 1]) == m));
    assert(b_array.is_future() || (b_strides[DIM - 2] == 1 &&
                                   (nrhs == 1 || static_cast<int64_t>(b_strides[DIM - 1]) == m)));
    assert(x_array.is_future() || (x_strides[DIM - 2] == 1 &&
                                   (nrhs == 1 || static_cast<int64_t>(x_strides[DIM - 1]) == m)));
#endif

#ifdef DEBUG_CUPYNUMERIC
    int64_t full_stride = m * m;
    for (int i = batchdims.size() - 1; i >= 0; --i) {
      if (batchdims[i] > 1) {
        assert(a_strides[i] == full_stride);
        full_stride *= batchdims[i];
      }
    }
#endif

    SolveImplBody<KIND, CODE>{context}(batchsize_total, m, n, nrhs, a, b, x);
  }

  template <Type::Code CODE,
            int32_t DIM,
            std::enable_if_t<!support_solve<CODE>::value || DIM<2>* = nullptr> void
            operator()(legate::PhysicalStore a_array,
                       legate::PhysicalStore b_array,
                       legate::PhysicalStore x_array) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void solve_template(TaskContext& context)
{
  auto a_array = context.input(0);
  auto b_array = context.input(1);
  auto x_array = context.output(0);
  double_dispatch(std::max(1, a_array.dim()),
                  a_array.type().code(),
                  SolveImpl<KIND>{context},
                  a_array,
                  b_array,
                  x_array);
}

}  // namespace cupynumeric
