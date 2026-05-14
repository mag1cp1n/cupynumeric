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

#include "cupynumeric/index/zipscatter.h"
// `zip_common.inl` provides `is_dense_row_major`, `Sparse/DenseIndexLoader`,
// and `build_source_point`, all shared with the gather path because scatter
// computes the same Point<N> from index arrays — only the read/write
// direction differs.
#include "cupynumeric/index/zip_common.inl"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

template <VariantKind KIND, int DIM, int N>
struct ZipScatterImplBody;

template <VariantKind KIND>
struct ZipScatterDimDispatch {
  TaskContext context;
  explicit ZipScatterDimDispatch(TaskContext context) : context(context) {}

  template <int DIM, int N>
  void operator()(ZipScatterArgs& args) const
  {
    ZipScatterImplBody<KIND, DIM, N>{context}(args);
  }
};

template <VariantKind KIND>
static void zipscatter_template(TaskContext& context)
{
  int64_t key_dim     = context.scalar(0).value<int64_t>();
  int64_t start_index = context.scalar(1).value<int64_t>();
  auto shape          = context.scalar(2).value<DomainPoint>();
  bool check_bounds   = context.scalar(3).value<bool>();

  // Inputs layout (matches `_issue_zipscatter_task` in deferred.py):
  //   - input(0)              : source values (DIM-dim, isomorphic to indices)
  //   - input(1)..input(K)    : K per-dim int64 index arrays (DIM-dim)
  //   - input(K+1)            : `out` re-added so Legate preserves cells we
  //                             don't overwrite; ignored by the kernel body
  //                             (we already get it via output(0))
  std::vector<PhysicalStore> inputs;
  auto all_inputs = context.inputs();
  // size() >= 3 always: source + at least one index array + result-preserve.
  for (uint32_t i = 1; i + 1 < all_inputs.size(); ++i) {
    inputs.emplace_back(all_inputs[i]);
  }

  ZipScatterArgs args{context.output(0),
                      context.input(0),
                      std::move(inputs),
                      key_dim,
                      start_index,
                      shape,
                      check_bounds};
  // `out` is N-dim and `source` is DIM-dim — the template parameter order
  // matches ZipGatherImplBody (DIM = "iteration dim", N = "indexed-array dim")
  // so we can reuse the shared `build_source_point` helper unchanged.
  double_dispatch(std::max(1, args.source.dim()),
                  std::max(1, args.out.dim()),
                  ZipScatterDimDispatch<KIND>{context},
                  args);
}

}  // namespace cupynumeric
