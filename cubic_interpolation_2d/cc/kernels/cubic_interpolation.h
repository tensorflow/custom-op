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

// Provides a cubic bspline interpolation (upsampling by an integer factor) for
// 1D, 2D, or 3D multi-channel arrays. The arrays must be provided as rank-2,
// rank-3, or rank-4 Eigen::Tensor with a RowMajor layout or compatible types.

#ifndef MULTIDIM_IMAGE_AUGMENTATION_KERNELS_CUBIC_INTERPOLATION_H_
#define MULTIDIM_IMAGE_AUGMENTATION_KERNELS_CUBIC_INTERPOLATION_H_

#include "cc/kernels/bspline.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace deepmind {
namespace multidim_image_augmentation {

// Provides a cubic bspline interpolation for a 1D multi-channel array (smooth
// "upsampling" by an integer factor). 'input_array' and 'output_array' must be
// rank-2 Eigen::Tensor with a RowMajor layout or compatible classes like
// Eigen::TensorMap.  Example usage (data has 3 channels here):
//
//     Eigen::Tensor<float, 2, Eigen::RowMajor> control_points(5, 3);
//     Eigen::Tensor<float, 2, Eigen::RowMajor> dense(21, 3);
//     int scale_factor = 10;
//     ...   // Initialize control points.
//     CubicInterpolation1d(control_points, scale_factor, &dense);
//
// For a detailed description of the restrictions, see
// CubicBSplineInterpolationCentered() in bspline.h.
template <typename InTensor, typename OutTensor>
void CubicInterpolation1d(const InTensor& input_array, int factor,
                          OutTensor* output_array_p) {
  static_assert(
      static_cast<int>(InTensor::Layout) == static_cast<int>(Eigen::RowMajor),
      "Input Tensor must have row major layout");
  static_assert(InTensor::NumIndices == 2, "Input Tensor must be rank 2");
  static_assert(
      static_cast<int>(OutTensor::Layout) == static_cast<int>(Eigen::RowMajor),
      "Output Tensor must have row major layout");
  static_assert(OutTensor::NumIndices == 2, "Output Tensor must be rank 2");
  OutTensor& output_array = *output_array_p;
  CHECK_EQ(input_array.dimension(1), output_array.dimension(1))
      << "Input and output must have the same number of channels";
  // Setup bspline basis function.
  std::vector<BSplineKernel> kernels_x0 = CreateBSplineKernels(factor);

  for (int comp = 0; comp < input_array.dimension(1); ++comp) {
    // Scale up in x0 direction with bspline kernels.
    SimpleConstArrayView in;
    in.data = &input_array(0, comp);
    in.num_lines = 1;
    in.num_elem = input_array.dimension(0);
    in.stride_lines = input_array.dimension(0) * input_array.dimension(1);
    in.stride_elem = input_array.dimension(1);
    SimpleArrayView out;
    out.data = &output_array(0, comp);
    out.num_lines = in.num_lines;
    out.num_elem = output_array.dimension(0);
    out.stride_lines = output_array.dimension(0) * output_array.dimension(1);
    out.stride_elem = output_array.dimension(1);
    CubicBSplineInterpolationCentered(in, out, kernels_x0);
  }
}

// Provides a cubic bspline interpolation for a 2D multi-channel array,
// analogous to CubicInterpolation1d (see above) but expects rank-3 Tensor.
template <typename InTensor, typename OutTensor>
void CubicInterpolation2d(const InTensor& input_array, Eigen::Vector2i factors,
                          OutTensor* output_array_p) {
  static_assert(
      static_cast<int>(InTensor::Layout) == static_cast<int>(Eigen::RowMajor),
      "Input Tensor must have row major layout");
  static_assert(InTensor::NumIndices == 3, "Input Tensor must be rank 3");
  static_assert(
      static_cast<int>(OutTensor::Layout) == static_cast<int>(Eigen::RowMajor),
      "Output Tensor must have row major layout");
  static_assert(OutTensor::NumIndices == 3, "Output Tensor must be rank 3");
  OutTensor& output_array = *output_array_p;
  CHECK_EQ(input_array.dimension(2), output_array.dimension(2))
      << "Input and output must have the same number of channels";

  // Setup bspline basis functions.
  std::vector<BSplineKernel> kernels_x0 = CreateBSplineKernels(factors(0));
  std::vector<BSplineKernel> kernels_x1 = CreateBSplineKernels(factors(1));

  Eigen::Tensor<float, 2, Eigen::RowMajor> tmp1(output_array.dimension(0),
                                                input_array.dimension(1));

  for (int comp = 0; comp < input_array.dimension(2); ++comp) {
    // Scale up in x0 direction with bspline kernel.
    SimpleConstArrayView in;
    in.data = &input_array(0, 0, comp);
    in.num_lines = input_array.dimension(1);
    in.num_elem = input_array.dimension(0);
    in.stride_lines = input_array.dimension(2);
    in.stride_elem = input_array.dimension(1) * input_array.dimension(2);
    SimpleArrayView out;
    out.data = tmp1.data();
    out.num_lines = in.num_lines;
    out.num_elem = tmp1.dimension(0);
    out.stride_lines = 1;
    out.stride_elem = tmp1.dimension(1);
    CubicBSplineInterpolationCentered(in, out, kernels_x0);

    // Scale up in x1 direction with bspline kernel.
    in.data = tmp1.data();
    in.num_lines = tmp1.dimension(0);
    in.num_elem = tmp1.dimension(1);
    in.stride_lines = tmp1.dimension(1);
    in.stride_elem = 1;
    out.data = &output_array(0, 0, comp);
    out.num_lines = in.num_lines;
    out.num_elem = output_array.dimension(1);
    out.stride_lines = output_array.dimension(1) * output_array.dimension(2);
    out.stride_elem = output_array.dimension(2);
    CubicBSplineInterpolationCentered(in, out, kernels_x1);
  }
}

// Provides a cubic bspline interpolation for a 3D multi-channel array,
// analogous to CubicInterpolation1d (see above) but expects rank-4 Tensor.
template <typename InTensor, typename OutTensor>
void CubicInterpolation3d(const InTensor& input_array, Eigen::Vector3i factors,
                          OutTensor* output_array_p) {
  static_assert(static_cast<int>(InTensor::Layout) == Eigen::RowMajor,
                "Input Tensor must have row major layout");
  static_assert(InTensor::NumIndices == 4, "Input Tensor must be rank 4");
  static_assert(static_cast<int>(OutTensor::Layout) == Eigen::RowMajor,
                "Output Tensor must have row major layout");
  static_assert(OutTensor::NumIndices == 4, "Output Tensor must be rank 4");
  OutTensor& output_array = *output_array_p;
  CHECK_EQ(input_array.dimension(3), output_array.dimension(3))
      << "Input and output must have the same number of channels";

  // Setup bspline basis functions.
  std::vector<BSplineKernel> kernels_x0 = CreateBSplineKernels(factors(0));
  std::vector<BSplineKernel> kernels_x1 = CreateBSplineKernels(factors(1));
  std::vector<BSplineKernel> kernels_x2 = CreateBSplineKernels(factors(2));

  Eigen::Tensor<float, 3, Eigen::RowMajor> tmp1(output_array.dimension(0),
                                                input_array.dimension(1),
                                                input_array.dimension(2));
  Eigen::Tensor<float, 3, Eigen::RowMajor> tmp2(output_array.dimension(0),
                                                output_array.dimension(1),
                                                input_array.dimension(2));

  for (int comp = 0; comp < input_array.dimension(3); ++comp) {
    // Scale up in x0 direction with bspline kernel.
    SimpleConstArrayView in;
    in.data = &input_array(0, 0, 0, comp);
    in.num_lines = input_array.dimension(1) * input_array.dimension(2);
    in.num_elem = input_array.dimension(0);
    in.stride_lines = input_array.dimension(3);
    in.stride_elem = input_array.dimension(1) * input_array.dimension(2) *
                     input_array.dimension(3);
    SimpleArrayView out;
    out.data = tmp1.data();
    out.num_lines = in.num_lines;
    out.num_elem = tmp1.dimension(0);
    out.stride_lines = 1;
    out.stride_elem = tmp1.dimension(1) * tmp1.dimension(2);
    CubicBSplineInterpolationCentered(in, out, kernels_x0);

    // Scale up in x1 direction with bspline kernel.
    for (int x0 = 0; x0 < output_array.dimension(0); ++x0) {
      in.data = &tmp1(x0, 0, 0);
      in.num_lines = tmp1.dimension(2);
      in.num_elem = tmp1.dimension(1);
      in.stride_lines = 1;
      in.stride_elem = tmp1.dimension(2);
      out.data = &tmp2(x0, 0, 0);
      out.num_lines = in.num_lines;
      out.num_elem = tmp2.dimension(1);
      out.stride_lines = 1;
      out.stride_elem = tmp1.dimension(2);
      CubicBSplineInterpolationCentered(in, out, kernels_x1);
    }

    // Scale up in x2 direction with bspline kernel.
    in.data = tmp2.data();
    in.num_lines = tmp2.dimension(0) * tmp2.dimension(1);
    in.num_elem = tmp2.dimension(2);
    in.stride_lines = tmp2.dimension(2);
    in.stride_elem = 1;
    out.data = &output_array(0, 0, 0, comp);
    out.num_lines = in.num_lines;
    out.num_elem = output_array.dimension(2);
    out.stride_lines = output_array.dimension(2) * output_array.dimension(3);
    out.stride_elem = output_array.dimension(3);
    CubicBSplineInterpolationCentered(in, out, kernels_x2);
  }
}
}  // namespace multidim_image_augmentation
}  // namespace deepmind

#endif  // MULTIDIM_IMAGE_AUGMENTATION_KERNELS_CUBIC_INTERPOLATION_H_