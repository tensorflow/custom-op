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

#include "multidim_image_augmentation/cc/kernels/apply_deformation.h"

#include <array>
#include <string>

#include "multidim_image_augmentation/cc/ops/apply_deformation_ops.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

using tensorflow::DEVICE_CPU;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;

enum SpatialDims {
  k2SpatialDims = 2,
  k3SpatialDims = 3,
};

//
// Wrappers for ApplyDeformation with a template specialization that allows
// the correct Deform method (2D vs 3D) to be selected based on the
// spatial_dims template parameters, or error if the style is unsupported.
//
template <SpatialDims spatial_dims, InterpolationStyle interpolation_style,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
class Applier {};

// 2D Deform.
template <InterpolationStyle interpolation_style,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
class Applier<k2SpatialDims, interpolation_style, extrapolation_style,
              conversion_style> {
 public:
  template <typename InTensor, typename DeformTensor, typename OutTensor>
  static void Apply(OpKernelContext* context, const InTensor& in,
                    const DeformTensor& deform, OutTensor* out,
                    const typename InTensor::Scalar* padding_constant) {
    ApplyDeformation<interpolation_style, extrapolation_style,
                     conversion_style>::Deform2D(in, deform, out,
                                                 padding_constant);
  }
};

// Error handler for unsupported 2D Deform configurations.
template <ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
class Applier<k2SpatialDims, kMixedNearestLinear, extrapolation_style,
              conversion_style> {
 public:
  template <typename InTensor, typename DeformTensor, typename OutTensor>
  static void Apply(OpKernelContext* context, const InTensor& in,
                    const DeformTensor& deform, OutTensor* out,
                    const typename InTensor::Scalar* padding_constant) {
    context->CtxFailure(
        InvalidArgument("Deform 2D does not support `mixed_nearest_linear`."));
  }
};

// 3D Deform.
template <InterpolationStyle interpolation_style,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
class Applier<k3SpatialDims, interpolation_style, extrapolation_style,
              conversion_style> {
 public:
  template <typename InTensor, typename DeformTensor, typename OutTensor>
  static void Apply(OpKernelContext* context, const InTensor& in,
                    const DeformTensor& deform, OutTensor* out,
                    const typename InTensor::Scalar* padding_constant) {
    ApplyDeformation<interpolation_style, extrapolation_style,
                     conversion_style>::Deform3D(in, deform, out,
                                                 padding_constant);
  }
};

//
// ApplyDeformation Op Kernel implementations.
//

template <SpatialDims spatial_dims, typename InType, typename OutType>
class ApplyDeformationOp : public OpKernel {
 public:
  explicit ApplyDeformationOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, GetAttributes<Status>(context, &attrs_));
    OP_REQUIRES(
        context,
        attrs_.requested_output_spatial_shape.empty() ||
            attrs_.requested_output_spatial_shape.size() == spatial_dims,
        InvalidArgument(
            "If specified, output_spatial_shape must have one element per "
            "input dimension"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(kInputIndex);
    const Tensor& deformation = context->input(kDeformIndex);
    const Tensor& padding_constant = context->input(kPaddingConstIndex);

    TensorShape output_shape(deformation.shape());

    for (int i = 0; i < attrs_.requested_output_spatial_shape.size(); ++i) {
      DCHECK(i < spatial_dims);
      auto size = attrs_.requested_output_spatial_shape[i];
      if (size >= 0) {
        OP_REQUIRES(
            context, size <= output_shape.dim_size(i),
            InvalidArgument(
                "output_spatial_shape must not exceed the deformation field "
                "size in any dimension"));
        output_shape.set_dim(i, size);
      }
    }
    auto num_channels = attrs_.output_num_channels >= 0
                            ? attrs_.output_num_channels
                            : input.dim_size(spatial_dims);
    const InType* eigen_padding_p = nullptr;
    if (attrs_.extrapolation_style == "const_padding") {
      OP_REQUIRES(
          context, padding_constant.NumElements() == num_channels,
          InvalidArgument(
              "padding constant must be a vector with num_channels elements."));
      eigen_padding_p = padding_constant.flat<InType>().data();
    }
    output_shape.set_dim(spatial_dims, num_channels);

    // NOTE: If making any change to the computation of `output_shape` above,
    // a corresponding change is also needed in ApplyDeformationShapeFunction.

    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    EigenTensorOut eigen_output = output->tensor<OutType, kTensorRank>();

    ApplyDeform(context, input.tensor<InType, kTensorRank>(),
                deformation.tensor<float, kTensorRank>(), &eigen_output,
                eigen_padding_p);
  }

 private:
  // One additional dimension for the channels.
  static constexpr int kTensorRank = spatial_dims + 1;

  typedef Eigen::Tensor<InType, kTensorRank, Eigen::RowMajor> EigenTensorIn;
  typedef Eigen::Tensor<float, kTensorRank, Eigen::RowMajor> EigenTensorDeform;
  typedef Eigen::TensorMap<Eigen::Tensor<OutType, kTensorRank, Eigen::RowMajor>,
                           Eigen::Aligned>
      EigenTensorOut;

  // Helper method to actually apply the deform on the image provided. This uses
  // a chain of internal methods to map out the runtime-provided style
  // parameters (in the `attrs_` member) to the appropriate compile-time
  // function instances via template parameters.
  void ApplyDeform(OpKernelContext* context, const EigenTensorIn& in,
                   const EigenTensorDeform& deform, EigenTensorOut* out,
                   const InType* padding_constant) {
    ExpandInterpolations(context, in, deform, out, padding_constant);
  }

  void ExpandInterpolations(OpKernelContext* context, const EigenTensorIn& in,
                            const EigenTensorDeform& deform,
                            EigenTensorOut* out,
                            const InType* padding_constant) {
    if (attrs_.interpolation_style == "nearest") {
      ExpandExtrapolations<kNearest>(context, in, deform, out,
                                     padding_constant);
    } else if (attrs_.interpolation_style == "linear") {
      ExpandExtrapolations<kLinear>(context, in, deform, out, padding_constant);
    } else if (attrs_.interpolation_style == "mixed_nearest_linear") {
      ExpandExtrapolations<kMixedNearestLinear>(context, in, deform, out,
                                                padding_constant);
    } else {
      LOG(FATAL) << "Bad interpolation style " << attrs_.interpolation_style;
    }
  }

  template <InterpolationStyle interpolation_style>
  void ExpandExtrapolations(OpKernelContext* context, const EigenTensorIn& in,
                            const EigenTensorDeform& deform,
                            EigenTensorOut* out,
                            const InType* padding_constant) {
    if (attrs_.extrapolation_style == "mirror") {
      ExpandConversions<interpolation_style, kMirror>(context, in, deform, out,
                                                      padding_constant);
    } else if (attrs_.extrapolation_style == "zero_padding") {
      ExpandConversions<interpolation_style, kZeroPadding>(
          context, in, deform, out, padding_constant);
    } else if (attrs_.extrapolation_style == "const_padding") {
      ExpandConversions<interpolation_style, kConstPadding>(
          context, in, deform, out, padding_constant);
    } else {
      LOG(FATAL) << "Bad extrapolation style " << attrs_.extrapolation_style;
    }
  }

  template <InterpolationStyle interpolation_style,
            ExtrapolationStyle extrapolation_style>
  void ExpandConversions(OpKernelContext* context, const EigenTensorIn& in,
                         const EigenTensorDeform& deform, EigenTensorOut* out,
                         const InType* padding_constant) {
    if (attrs_.conversion_style == "no_conversion") {
      Apply<interpolation_style, extrapolation_style, kNoConversion>(
          context, in, deform, out, padding_constant);
    } else if (attrs_.conversion_style == "indexed_to_one_hot") {
      Apply<interpolation_style, extrapolation_style, kIndexedToOneHot>(
          context, in, deform, out, padding_constant);
    } else {
      LOG(FATAL) << "Bad conversion style " << attrs_.conversion_style;
    }
  }

  template <InterpolationStyle interpolation_style,
            ExtrapolationStyle extrapolation_style,
            ConversionStyle conversion_style>
  void Apply(OpKernelContext* context, const EigenTensorIn& in,
             const EigenTensorDeform& deform, EigenTensorOut* out,
             const InType* padding_constant) {
    Applier<spatial_dims, interpolation_style, extrapolation_style,
            conversion_style>::Apply(context, in, deform, out,
                                     padding_constant);
  }

  // Parameters supplied from client.
  DeformationAttributes attrs_;
};

#define REGISTER_KERNEL(INTYPE, OUTTYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("ApplyDeformation2D")                           \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<INTYPE>("input_type")            \
                              .TypeConstraint<OUTTYPE>("output_type"),         \
                          ApplyDeformationOp<k2SpatialDims, INTYPE, OUTTYPE>); \
  REGISTER_KERNEL_BUILDER(Name("ApplyDeformation3D")                           \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<INTYPE>("input_type")            \
                              .TypeConstraint<OUTTYPE>("output_type"),         \
                          ApplyDeformationOp<k3SpatialDims, INTYPE, OUTTYPE>);

// The set of registrations shall satisfy all the {input_type x output_type}
// combinations declared for ApplyDeformation2D and ApplyDeformation3D ops.
// Note that each instantiation of the kernel instantiates 2x the number of
// style combinations (36 at time of writing) of the underlying ApplyDeformation
// template.

REGISTER_KERNEL(float, float)
REGISTER_KERNEL(uint8, float)
REGISTER_KERNEL(int32, float)

REGISTER_KERNEL(float, uint8)
REGISTER_KERNEL(uint8, uint8)
REGISTER_KERNEL(int32, uint8)

REGISTER_KERNEL(float, int32)
REGISTER_KERNEL(uint8, int32)
REGISTER_KERNEL(int32, int32)

#undef REGISTER_KERNEL

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
