// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "multidim_image_augmentation/cc/platform/types.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace tabulated_functions_op_internal {
// Allows to globally disable AVX optimizations. Used only for benchmarking.
bool use_avx_optimizations = true;
}  // namespace tabulated_functions_op_internal
namespace {

// Optimized implementation of ApplyTabulatedFunctions. In the default case this
// is unimplemented. Returns the number of pixels processed.
template <typename Tin, typename Tout>
int64 ComputeOptimized(int num_channels, int num_pixels, const Tin* input_p,
                       Tout* output_p, const Tout* table_p, int64 table_size,
                       float offset, float scale) {
  return 0;
}

#ifdef __AVX2__
// Optimized implementation for AVX2 on floats. Returns the number of pixels
// processed, which may be zero or (num_pixels // 8) * 8.
template <>
int64 ComputeOptimized<float, float>(int num_channels, int num_pixels,
                                     const float* input_p, float* output_p,
                                     const float* table_p, int64 table_size,
                                     float offset, float scale) {
  if (!tensorflow::port::TestCPUFeature(tensorflow::port::CPUFeature::AVX2) ||
      num_channels != 1) {
    return 0;
  }

  // Pre-broadcast vectors of zeroes and the maximum allowed index.
  __m256 zero_m256 = _mm256_set1_ps(0.0f);
  __m256 limit_m256 = _mm256_set1_ps(static_cast<float>(table_size - 2));

  // Pre-broadcast vectors of scale and offset for addition later.
  __m256 scale_m256 = _mm256_set1_ps(scale);
  __m256 offset_m256 = _mm256_set1_ps(offset);

  const int64 kVectorWidth = 8;
  int64 num_vector_pixels = num_pixels / kVectorWidth;
  for (int64 i = 0; i < num_vector_pixels; ++i) {
    // Load 8 consecutive floats.
    __m256 input_float = _mm256_load_ps(&input_p[i * kVectorWidth]);
    // Apply the broadcasted offset and scaling.
    __m256 index_float = (input_float + offset_m256) * scale_m256;

    // Clamp to [0, limit] using a min(max()) pair.
    __m256 index_clamped_float =
        _mm256_min_ps(_mm256_max_ps(zero_m256, index_float), limit_m256);

    // Perform std::floor() by purely converting float to int. We know all
    // values are positive so the round-down behaviour is correct here.
    __m256i index_int = _mm256_cvttps_epi32(index_clamped_float);
    // Perform the index lookups for f0. Note scale=4 because floats are
    // 32-bit.
    __m256 f0 = _mm256_i32gather_ps(table_p, index_int, /*scale=*/4);
    // Perform the index lookups for f1.
    __m256 f1 = _mm256_i32gather_ps(table_p, index_int + 1,
                                    /*scale=*/4);
    __m256 m = f1 - f0;

    // Regenerate the std::floor() value by rounding index_clamped_float. We
    // could perform a float-to-int here instead but that would be more
    // expensive.
    __m256 index_rel = index_float - _mm256_floor_ps(index_clamped_float);
    _mm256_store_ps(&output_p[i * kVectorWidth], f0 + m * index_rel);
  }

  return num_vector_pixels * 8;
}
#endif

template <typename INTYPE, typename OUTTYPE>
class ApplyTabulatedFunctionsOp : public tensorflow::OpKernel {
 public:
  explicit ApplyTabulatedFunctionsOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("offset", &offset_));
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& input = context->input(0);
    int num_channels = input.dim_size(input.dims() - 1);
    auto tabulated_function = context->input(1).tensor<OUTTYPE, 2>();
    OP_REQUIRES(context, tabulated_function.dimension(0) == num_channels,
                tensorflow::errors::InvalidArgument(
                    "incompatible number of channels. The input tensor has ",
                    num_channels, " channels, and there are ",
                    tabulated_function.dimension(0), " tabulated functions"));
    tensorflow::Tensor* output;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, context->input(0).shape(), &output));
    const INTYPE* input_p = input.flat<INTYPE>().data();
    OUTTYPE* output_p = output->flat<OUTTYPE>().data();

    int64 num_pixels = input.NumElements() / num_channels;

    // Optimized implementation, if available.
    int64 n = 0;
    if (tabulated_functions_op_internal::use_avx_optimizations) {
      n = ComputeOptimized(num_channels, num_pixels, input_p, output_p,
                           &tabulated_function(0, 0),
                           tabulated_function.dimension(1), offset_, scale_);
    }
    int64 i = n * num_channels;
    for (; n < num_pixels; ++n) {
      for (int channel = 0; channel < num_channels; ++channel) {
        float index_float = (input_p[i] + offset_) * scale_;

        // find the two closest control points
        int index = std::floor(index_float);
        if (index < 0) index = 0;
        if (index > tabulated_function.dimension(1) - 2)
          index = tabulated_function.dimension(1) - 2;

        // compute the linear function between the control points
        OUTTYPE f0 = tabulated_function(channel, index);
        OUTTYPE f1 = tabulated_function(channel, index + 1);
        OUTTYPE m = f1 - f0;

        // apply it to the input value
        float index_rel = index_float - index;
        output_p[i] = static_cast<OUTTYPE>(f0 + m * index_rel);
        ++i;
      }
    }
  }

 protected:
  float offset_;
  float scale_;
};

#define REGISTER_KERNEL(INTYPE, OUTTYPE)                               \
  REGISTER_KERNEL_BUILDER(Name("ApplyTabulatedFunctions")              \
                              .Device(tensorflow::DEVICE_CPU)          \
                              .TypeConstraint<INTYPE>("input_type")    \
                              .TypeConstraint<OUTTYPE>("output_type"), \
                          ApplyTabulatedFunctionsOp<INTYPE, OUTTYPE>)

REGISTER_KERNEL(float, float)
REGISTER_KERNEL(float, int64)
REGISTER_KERNEL(float, int32)
REGISTER_KERNEL(float, uint8)
REGISTER_KERNEL(int64, float)
REGISTER_KERNEL(int64, int64)
REGISTER_KERNEL(int64, int32)
REGISTER_KERNEL(int64, uint8)
REGISTER_KERNEL(int32, float)
REGISTER_KERNEL(int32, int64)
REGISTER_KERNEL(int32, int32)
REGISTER_KERNEL(int32, uint8)
REGISTER_KERNEL(uint8, float)
REGISTER_KERNEL(uint8, int64)
REGISTER_KERNEL(uint8, int32)
REGISTER_KERNEL(uint8, uint8)
#undef REGISTER_KERNEL

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
