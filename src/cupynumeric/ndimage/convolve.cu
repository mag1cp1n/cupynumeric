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

#include "cupynumeric/ndimage/convolve.h"
#include "cupynumeric/ndimage/convolve_template.inl"

#include "cupynumeric/cuda_help.h"
#include "cupynumeric/pitches.h"

namespace cupynumeric {

using namespace legate;

__device__ int32_t get_boundary_index(const int32_t idx,
                                      const int32_t size,
                                      const CuPyNumericNdimageConvolveMode mode)
{
  switch (mode) {
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_REFLECT: {
      int32_t out = idx;

      out = max(out, -1 - out);
      out %= size * 2;
      out = min(out, 2 * size - 1 - out);
      return out;
    }
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_MIRROR: {
      if (size == 1) {
        return 0;
      }
      int32_t out = idx;

      out = max(out, -out);
      out %= 2 * size - 2;
      out = min(out, 2 * size - 2 - out);
      return out;
    }
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_NEAREST: return min(max(idx, int32_t{0}), size - 1);
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_WRAP: {
      const int32_t out = idx % size;
      return out < 0 ? out + size : out;
    }
    case CUPYNUMERIC_NDIMAGE_CONVOLVE_CONSTANT: return (idx < 0 || idx >= size) ? -1 : idx;
    default: return idx;
  }
}

template <int DIM_IDX, int DIM, typename VAL>
struct FilterLoop {
  __device__ static void run(const Point<DIM> output_point,
                             const VAL* input,
                             const Point<DIM> input_strides,
                             const Rect<DIM> input_rect,
                             const VAL* weights_base,
                             const Point<DIM> weights_strides,
                             const Rect<DIM> weights_rect,
                             const CuPyNumericNdimageConvolveMode mode,
                             const VAL cval,
                             const Point<DIM> origins,
                             const bool use_cval,
                             const bool correlate,
                             VAL& acc)
  {
    const int32_t w_size_dim     = weights_rect.hi[DIM_IDX] - weights_rect.lo[DIM_IDX] + 1;
    const int32_t input_size_dim = input_rect.hi[DIM_IDX] - input_rect.lo[DIM_IDX] + 1;

    const VAL* weights_ptr =
      correlate ? weights_base : weights_base + (w_size_dim - 1) * weights_strides[DIM_IDX];
    int32_t traversal_dir = correlate ? 1 : -1;

    const int32_t center_dim = w_size_dim / 2 + origins[DIM_IDX];
    for (int32_t iw = 0, curr_idx_dim = output_point[DIM_IDX] + center_dim - w_size_dim + 1;
         iw < w_size_dim;
         iw++, curr_idx_dim++, weights_ptr += traversal_dir * weights_strides[DIM_IDX]) {
      const int32_t actual_idx_dim = get_boundary_index(curr_idx_dim, input_size_dim, mode);

      const bool next_use_cval = use_cval || (actual_idx_dim < 0);
      const VAL* next_input =
        next_use_cval ? input : (input + input_strides[DIM_IDX] * actual_idx_dim);

      FilterLoop<DIM_IDX + 1, DIM, VAL>::run(output_point,
                                             next_input,
                                             input_strides,
                                             input_rect,
                                             weights_ptr,
                                             weights_strides,
                                             weights_rect,
                                             mode,
                                             cval,
                                             origins,
                                             next_use_cval,
                                             correlate,
                                             acc);
    }
  }
};

template <int DIM, typename VAL>
struct FilterLoop<DIM, DIM, VAL> {
  __device__ static void run(const Point<DIM>,
                             const VAL* input,
                             const Point<DIM>,
                             const Rect<DIM>,
                             const VAL* weights,
                             const Point<DIM>,
                             const Rect<DIM>,
                             const CuPyNumericNdimageConvolveMode,
                             const VAL cval,
                             const Point<DIM>,
                             const bool use_cval,
                             const bool,
                             VAL& acc)
  {
    const VAL input_value = use_cval ? cval : (*input);
    acc += input_value * (*weights);
  }
};

template <typename VAL, int DIM>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  convolve_kernel(AccessorWO<VAL, DIM> out,
                  const Pitches<DIM - 1> out_pitches,
                  const size_t output_volume,
                  const Point<DIM> output_lo,
                  const VAL* input,
                  const Point<DIM> input_strides,
                  const Rect<DIM> input_rect,
                  const VAL* weights,
                  const Point<DIM> weights_strides,
                  const Rect<DIM> weights_rect,
                  const CuPyNumericNdimageConvolveMode mode,
                  const VAL cval,
                  const Point<DIM> origins,
                  const bool use_cval,
                  const bool correlate)
{
  const size_t idx = global_tid_1d();

  if (idx >= output_volume) {
    return;
  }

  // determine the output point that this thread is responsible for
  auto output_point = out_pitches.unflatten(idx, output_lo);

  VAL acc = 0;

  FilterLoop<0, DIM, VAL>::run(output_point,
                               input,
                               input_strides,
                               input_rect,
                               weights,
                               weights_strides,
                               weights_rect,
                               mode,
                               cval,
                               origins,
                               false,
                               correlate,
                               acc);

  out[output_point] = acc;
}

template <typename VAL, int DIM>
struct NdimageConvolveImplBody<VariantKind::GPU, VAL, DIM> {
  TaskContext context;
  explicit NdimageConvolveImplBody(TaskContext context) : context(context) {}

  void operator()(AccessorWO<VAL, DIM> out,
                  AccessorRO<VAL, DIM> input,
                  AccessorRO<VAL, DIM> weights,
                  const Rect<DIM>& io_rect,
                  const Rect<DIM>& weights_rect,
                  CuPyNumericNdimageConvolveMode mode,
                  VAL cval,
                  Point<DIM> origins) const
  {
    // get pointers and strides to input and weights
    size_t input_strides[DIM];
    const auto input_ptr = input.ptr(io_rect, input_strides);

    size_t weights_strides[DIM];
    const auto weights_ptr = weights.ptr(weights_rect, weights_strides);

    // get output pitches
    Pitches<DIM - 1> output_pitches;
    const size_t output_volume = output_pitches.flatten(io_rect);

    const size_t num_blocks = (output_volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const auto stream       = context.get_task_stream();

    convolve_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(out,
                                                                  output_pitches,
                                                                  output_volume,
                                                                  io_rect.lo,
                                                                  input_ptr,
                                                                  Point<DIM>(input_strides),
                                                                  io_rect,
                                                                  weights_ptr,
                                                                  Point<DIM>(weights_strides),
                                                                  weights_rect,
                                                                  mode,
                                                                  cval,
                                                                  origins,
                                                                  false,
                                                                  false);
  }
};

/*static*/ void NdimageConvolveTask::gpu_variant(TaskContext context)
{
  ndimage_convolve_template<VariantKind::GPU>(context);
}

}  // namespace cupynumeric
