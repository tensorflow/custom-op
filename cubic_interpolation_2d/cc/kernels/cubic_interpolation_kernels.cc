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

#include "multidim_image_augmentation/kernels/cubic_interpolation.h"
#include "multidim_image_augmentation/platform/types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::errors::InvalidArgument;

// Validates the size of `input` in dimension `dim` against the preconditions
// described in the Op API docs.
bool ValidateInputForDim(OpKernelContext* context, int dim, const Tensor& input,
                         int32 factor, int32 output_size) {
  if (input.dim_size(dim) % 2 != output_size % 2) {
    context->CtxFailure(InvalidArgument(
        "output size and input size must both be odd or both be even. dim=",
        dim, " input size=", input.dim_size(dim),
        " output size=", output_size, input.shape().DebugString()));
    return false;
  }
  if (input.dim_size(dim) % 2 == 0 && factor % 2 != 1) {
    context->CtxFailure(InvalidArgument(
        "factor must be odd as input and output size is even. dim=", dim,
        " input size=", input.dim_size(dim), " factor=", factor,
        input.shape().DebugString()));
    return false;
  }
  return true;
}

bool ValidateInput(OpKernelContext* context, const Tensor& input,
                   const std::vector<int32>& factors,
                   const std::vector<int32>& output_spatial_shape) {
  for (int d = 0; d < factors.size(); ++d) {
    if (!ValidateInputForDim(context, d, input, factors[d],
                             output_spatial_shape[d])) {
      return false;
    }
  }
  return true;
}

// CPU kernel implementation for the TensorFlow Op 'cubic_interpolation1d'.
class CubicInterpolation1DOp : public OpKernel {
 public:
  explicit CubicInterpolation1DOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("factor", &factor_));
    OP_REQUIRES_OK(context, context->GetAttr("output_length", &output_length_));
  }
  ~CubicInterpolation1DOp() override {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 2,
                InvalidArgument("input must be rank 2",
                                input.shape().DebugString()));
    int out_shape_0 = output_length_;
    if (output_length_ == 0) {
      out_shape_0 = (input.dim_size(0) - 1) * factor_ + 1;
    }
    if (!ValidateInputForDim(context, 0, input, factor_, out_shape_0)) return;

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, {out_shape_0, input.dim_size(1)}, &output));

    auto in = input.tensor<float, 2>();
    auto out = output->tensor<float, 2>();
    if (out_shape_0 <= (input.dim_size(0) - 3) * factor_ + 1) {
      // If we can do full interpolation everything is fine.
      CubicInterpolation1d(in, factor_, &out);
    } else {
      // We need to add a border pixel a each end to ensure always two
      // neighbours in each direction.
      Eigen::Tensor<float, 2, Eigen::RowMajor> in_padded(in.dimension(0) + 2,
                                                         in.dimension(1));
      Eigen::array<int64, 2> offsets = {1, 0};
      Eigen::array<int64, 2> extents = {in.dimension(0), in.dimension(1)};
      in_padded.slice(offsets, extents) = in;

      // Set padding values by linear extrapolation.
      for (int channel = 0; channel < in_padded.dimension(1); ++channel) {
        in_padded(0, channel) =
            2 * in_padded(1, channel) - in_padded(2, channel);
        in_padded(in_padded.dimension(0) - 1, channel) =
            2 * in_padded(in_padded.dimension(0) - 2, channel) -
            in_padded(in_padded.dimension(0) - 3, channel);
      }
      // Use the padded input for interpolation.
      CubicInterpolation1d(in_padded, factor_, &out);
    }
  }

 private:
  int factor_;
  int output_length_;
};

// Register the CPU kernel for the TensorFlow Op 'cubic_interpolation1d'.
REGISTER_KERNEL_BUILDER(Name("CubicInterpolation1D").Device(DEVICE_CPU),
                        CubicInterpolation1DOp);

// CPU kernel implementation for the TensorFlow Op 'cubic_interpolation2d'.
class CubicInterpolation2DOp : public OpKernel {
 public:
  explicit CubicInterpolation2DOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("factors", &factors_));
    OP_REQUIRES(
        context, factors_.size() == 2,
        InvalidArgument("factors must be rank 2, got ", factors_.size()));
    OP_REQUIRES_OK(context, context->GetAttr("output_spatial_shape",
                                             &output_spatial_shape_));
    OP_REQUIRES(context, output_spatial_shape_.size() == 2,
                InvalidArgument("output_spatial_shape must be rank 2, got ",
                                output_spatial_shape_.size()));
  }
  ~CubicInterpolation2DOp() override = default;

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 3,
                InvalidArgument("input must be rank 3",
                                input.shape().DebugString()));
    if (!ValidateInput(context, input, factors_, output_spatial_shape_)) return;
    int64 num_channels = input.dim_size(2);
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0,
                                {output_spatial_shape_[0],
                                 output_spatial_shape_[1], num_channels},
                                &output));

    auto in = input.tensor<float, 3>();
    auto out = output->tensor<float, 3>();
    CubicInterpolation2d(in, {factors_[0], factors_[1]}, &out);
  }

 private:
  std::vector<int32> factors_;
  std::vector<int32> output_spatial_shape_;
};

// Register the CPU kernel for the TensorFlow Op 'cubic_interpolation2d'.
REGISTER_KERNEL_BUILDER(Name("CubicInterpolation2D").Device(DEVICE_CPU),
                        CubicInterpolation2DOp);

// CPU kernel implementation for the TensorFlow Op 'cubic_interpolation3d'.
class CubicInterpolation3DOp : public OpKernel {
 public:
  explicit CubicInterpolation3DOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("factors", &factors_));
    OP_REQUIRES(
        context, factors_.size() == 3,
        InvalidArgument("factors must be rank 3, got ", factors_.size()));
    OP_REQUIRES_OK(context, context->GetAttr("output_spatial_shape",
                                             &output_spatial_shape_));
    OP_REQUIRES(context, output_spatial_shape_.size() == 3,
                InvalidArgument("output_spatial_shape must be rank 3, got ",
                                output_spatial_shape_.size()));
  }
  ~CubicInterpolation3DOp() override = default;

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                InvalidArgument("input must be rank 4",
                                input.shape().DebugString()));
    if (!ValidateInput(context, input, factors_, output_spatial_shape_)) return;
    int64 num_channels = input.dim_size(3);
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0,
                       {output_spatial_shape_[0], output_spatial_shape_[1],
                        output_spatial_shape_[2], num_channels},
                       &output));

    auto in = input.tensor<float, 4>();
    auto out = output->tensor<float, 4>();
    CubicInterpolation3d(in, {factors_[0], factors_[1], factors_[2]}, &out);
  }

 private:
  std::vector<int32> factors_;
  std::vector<int32> output_spatial_shape_;
};

// Register the CPU kernel for the TensorFlow Op 'cubic_interpolation3d'.
REGISTER_KERNEL_BUILDER(Name("CubicInterpolation3D").Device(DEVICE_CPU),
                        CubicInterpolation3DOp);

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind