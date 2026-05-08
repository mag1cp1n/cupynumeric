/* Copyright 2026 NVIDIA Corporation
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

#include "cupynumeric/index/zipgather.h"
#include "cupynumeric/index/zip.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <int32_t DIM>
inline bool is_dense_row_major(const size_t strides[DIM], const Rect<DIM>& rect, size_t elem_size)
{
  size_t expected = elem_size;
  for (int d = DIM - 1; d >= 0; --d) {
    if (strides[d] != expected) {
      return false;
    }
    expected *= (rect.hi[d] - rect.lo[d] + 1);
  }
  return true;
}

// Loads index values from a per-dim accessor at a Point<DIM>. Storage is
// templated so the same struct works for std::vector<AccessorRO<...>> on the
// CPU/OMP path and Buffer<AccessorRO<...>, 1> on the GPU path.
template <int DIM, typename Storage>
struct SparseIndexLoader {
  Storage index_arrays;

  LEGATE_HOST_DEVICE int64_t load(size_t dim, const Point<DIM>& p, size_t /*idx*/) const
  {
    return index_arrays[dim][p];
  }
};

// Loads index values from per-dim raw int64_t pointers using a flat index.
// Storage may be std::vector<const int64_t*> or Buffer<const int64_t*, 1>.
template <int DIM, typename Storage>
struct DenseIndexLoader {
  Storage index_ptrs;

  LEGATE_HOST_DEVICE int64_t load(size_t dim, const Point<DIM>& /*p*/, size_t idx) const
  {
    return index_ptrs[dim][idx];
  }
};

// Builds the source-side Point<N> consumed by zip / zipgather kernels.
//
// When narrays == N every output dimension is provided by an index array.
// Otherwise the leading [0, start_index) dimensions are copied from p, the
// next narrays come from the index arrays, and the trailing dimensions come
// from broadcast positions in p (offset by key_dim - narrays).
//
// compute_index normalizes a raw index against an extent. CPU/OMP variants
// pass a lambda that may signal out-of-bounds; the GPU variant passes a
// device functor wrapping compute_idx_cuda.
template <int DIM, int N, typename Loader, typename ComputeIndexFn>
LEGATE_HOST_DEVICE inline Point<N> build_source_point(const Point<DIM>& p,
                                                      const Loader& loader,
                                                      size_t flat_idx,
                                                      int64_t narrays,
                                                      int64_t key_dim,
                                                      int64_t start_index,
                                                      const DomainPoint& shape,
                                                      const ComputeIndexFn& compute_index)
{
  Point<N> new_point;

  if (narrays == N) {
    for (size_t i = 0; i < N; ++i) {
      new_point[i] = compute_index(loader.load(i, p, flat_idx), shape[i]);
    }
  } else {
    for (int64_t i = 0; i < start_index; ++i) {
      new_point[i] = p[i];
    }
    for (size_t i = 0; i < static_cast<size_t>(narrays); ++i) {
      const auto dim = start_index + i;
      new_point[dim] = compute_index(loader.load(i, p, flat_idx), shape[dim]);
    }
    for (size_t i = start_index + narrays; i < N; ++i) {
      const int64_t j = key_dim + i - narrays;
      new_point[i]    = p[j];
    }
  }

  return new_point;
}

template <VariantKind KIND, int DIM, int N>
struct ZipGatherImplBody;

template <VariantKind KIND>
struct ZipGatherDimDispatch {
  TaskContext context;
  explicit ZipGatherDimDispatch(TaskContext context) : context(context) {}

  template <int DIM, int N>
  void operator()(ZipGatherArgs& args) const
  {
    ZipGatherImplBody<KIND, DIM, N>{context}(args);
  }
};

template <VariantKind KIND>
static void zipgather_template(TaskContext& context)
{
  int64_t key_dim     = context.scalar(0).value<int64_t>();
  int64_t start_index = context.scalar(1).value<int64_t>();
  auto shape          = context.scalar(2).value<DomainPoint>();
  bool check_bounds   = context.scalar(3).value<bool>();

  std::vector<PhysicalStore> inputs;
  auto all_inputs = context.inputs();
  for (uint32_t i = 1; i < all_inputs.size(); ++i) {
    inputs.emplace_back(all_inputs[i]);
  }

  ZipGatherArgs args{context.output(0),
                     context.input(0),
                     std::move(inputs),
                     key_dim,
                     start_index,
                     shape,
                     check_bounds};
  double_dispatch(std::max(1, args.out.dim()),
                  std::max(1, args.source.dim()),
                  ZipGatherDimDispatch<KIND>{context},
                  args);
}

}  // namespace cupynumeric
