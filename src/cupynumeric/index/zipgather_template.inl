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
#include "cupynumeric/index/zip_common.inl"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

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
