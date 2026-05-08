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

#include "cupynumeric/index/zipgather.h"
#include "cupynumeric/index/zipgather_template.inl"

#include <cstring>

namespace cupynumeric {

using namespace legate;

template <int DIM, int N>
struct ZipGatherImplBody<VariantKind::CPU, DIM, N> {
  TaskContext context;
  explicit ZipGatherImplBody(TaskContext context) : context(context) {}

  using VAL = int64_t;

  void operator()(ZipGatherArgs& args) const
  {
    auto out_rect = args.out.shape<DIM>();
    auto src_rect = args.source.shape<N>();
    auto out_acc  = args.out.write_accessor<char, DIM>(out_rect);
    auto src_acc  = args.source.read_accessor<char, N>(src_rect);

    size_t out_bstrides[DIM];
    char* out_bytes = reinterpret_cast<char*>(out_acc.ptr(out_rect, out_bstrides));

    size_t src_bstrides[N];
    const char* src_bytes = reinterpret_cast<const char*>(src_acc.ptr(src_rect, src_bstrides));

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(out_rect);
    if (volume == 0) {
      return;
    }

    const size_t elem_size = args.source.type().size();
    LEGATE_ASSERT(elem_size == args.out.type().size());

#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
    bool out_dense     = is_dense_row_major<DIM>(out_bstrides, out_rect, elem_size);
    bool indices_dense = true;
    bool source_dense  = is_dense_row_major<N>(src_bstrides, src_rect, elem_size);
#else
    bool out_dense     = false;
    bool indices_dense = false;
    bool source_dense  = false;
#endif

    std::vector<AccessorRO<VAL, DIM>> index_arrays;
    index_arrays.reserve(args.inputs.size());
    for (auto& input : args.inputs) {
      auto input_rect = input.shape<DIM>();
      LEGATE_ASSERT(input_rect == out_rect);
      index_arrays.push_back(input.read_accessor<VAL, DIM>(input_rect));
#if !LEGATE_DEFINED(LEGATE_BOUNDS_CHECKS)
      indices_dense = indices_dense && index_arrays.back().accessor.is_dense_row_major(input_rect);
#endif
    }

    const auto compute_index = [&](coord_t index, coord_t extent) {
      return args.check_bounds ? compute_idx(index, extent) : compute_idx_unchecked(index, extent);
    };
    const auto out_stride_point = Point<DIM>{out_bstrides};
    const auto src_stride_point = Point<N>{src_bstrides};
    const int64_t narrays       = index_arrays.size();

    if (out_dense && indices_dense && source_dense) {
      std::vector<const int64_t*> index_ptrs;
      index_ptrs.reserve(index_arrays.size());
      for (auto& index_array : index_arrays) {
        index_ptrs.push_back(index_array.ptr(out_rect));
      }

      using DenseLoader = DenseIndexLoader<DIM, const std::vector<const int64_t*>&>;
      const auto loader = DenseLoader{index_ptrs};
      for (size_t idx = 0; idx < volume; ++idx) {
        Point<DIM> p{};
        if (narrays != N) {
          p = pitches.unflatten(idx, out_rect.lo);
        }
        const auto new_point = build_source_point<DIM, N>(
          p, loader, idx, narrays, args.key_dim, args.start_index, args.shape, compute_index);
        memcpy(out_bytes + idx * elem_size, src_bytes + new_point.dot(src_stride_point), elem_size);
      }
    } else {
      using SparseLoader = SparseIndexLoader<DIM, const std::vector<AccessorRO<VAL, DIM>>&>;
      const auto loader  = SparseLoader{index_arrays};
      for (size_t idx = 0; idx < volume; ++idx) {
        const auto p         = pitches.unflatten(idx, out_rect.lo);
        const auto new_point = build_source_point<DIM, N>(
          p, loader, idx, narrays, args.key_dim, args.start_index, args.shape, compute_index);
        memcpy(out_bytes + p.dot(out_stride_point),
               src_bytes + new_point.dot(src_stride_point),
               elem_size);
      }
    }
  }
};

/*static*/ void ZipGatherTask::cpu_variant(TaskContext context)
{
  zipgather_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{

static const auto cupynumeric_reg_task_ = []() -> char {
  ZipGatherTask::register_variants();
  return 0;
}();

}  // namespace

}  // namespace cupynumeric
