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

// Provides a fast cubic bspline interpolation for (N - 1) equidistantly spaced
// sampling postions between the control points. This can be interpreted as a
// smooth upsampling by an integer factor N. Be aware that the resulting
// function usually does _not_ pass through the control points.

#ifndef MULTIDIM_IMAGE_AUGMENTATION_KERNELS_BSPLINE_H_
#define MULTIDIM_IMAGE_AUGMENTATION_KERNELS_BSPLINE_H_

#include <vector>

#include "tensorflow/core/platform/logging.h"

namespace deepmind {
namespace multidim_image_augmentation {

struct BSplineKernel {
  float b0;
  float b1;
  float b2;
  float b3;
};

// Computes the bspline kernel functions for a given scale factor.
inline std::vector<BSplineKernel> CreateBSplineKernels(int scale_factor) {
  std::vector<BSplineKernel> kernels(scale_factor);
  for (int i = 0; i < scale_factor; ++i) {
    float x = static_cast<float>(i) / scale_factor;
    kernels[i].b0 = 1. / 6 * (-x * x * x + 3 * x * x - 3 * x + 1);
    kernels[i].b1 = 1. / 6 * (3 * x * x * x + -6 * x * x + 4);
    kernels[i].b2 = 1. / 6 * (-3 * x * x * x + 3 * x * x + 3 * x + 1);
    kernels[i].b3 = 1. / 6 * x * x * x;
  }
  return kernels;
}

// SimpleArrayView provides a 2D view of an n-dimensional float array. Only
// used to aggregate the parameters for CubicBSplineInterpolationCentered()
// below.
//
// - 'data' points to the first element of the array.
// - 'num_elem' and 'stride_elem' define the iteration in the inner loop.
// - 'num_lines', 'stride_lines' define the iteration in the outer loop.
struct SimpleArrayView {
  float* data = nullptr;
  int num_lines = 0;
  int num_elem = 0;
  int stride_lines = 1;
  int stride_elem = 1;
};

// const version of SimpleArrayView
struct SimpleConstArrayView {
  const float* data = nullptr;
  int num_lines = 0;
  int num_elem = 0;
  int stride_lines = 1;
  int stride_elem = 1;
};

// CubicBSplineInterpolationCentered performs a 1D fast cubic b-spline
// interpolation (can be interpreted as smooth upsmapling with an integer
// factor) where the centers of the control point array and the dense output
// array are aligned. Be aware that the resulting function usually does _not_
// pass through the control points. Due to the centering certain restrictions
// apply on the number of control points and the scaling factor (see below).
//
// This implementation does no extrapolation. Due to the kernel size of 4 in
// cubic b-spline interpolation, the interpolated array can only extend from
// the second control point to the second last controlpoint (see below for
// an illustration of "max_out_length").
//
// The input and output array are provided as SimpleArrayView's (see
// definition above). The strided element access allows easy application to
// multichannel-data and n-dimensional arrays. The specification of multiple
// strided "lines" (implemented as outer loop) allows to process a
// multichannel signal or slices of a n-dimensional array in a single run.
//
// Example code (for more examples see bspline_test.cc).
//
//    float control_points[7 * 3];
//    float dense_array[13 * 3];
//    for (int i = 0; i < 7 * 3; ++i) {
//      control_points[i] = i;
//    }
//    SimpleConstArrayView in;
//    in.data = control_points;
//    in.num_lines = 3;
//    in.num_elem = 7;
//    in.stride_lines = 7;
//    in.stride_elem = 1;
//    SimpleArrayView out;
//    out.data = dense_array;
//    out.num_lines = in.num_lines;
//    out.num_elem = 13;
//    out.stride_lines = 13;
//    out.stride_elem = 1;
//    int scale_factor = 4;
//    std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);
//    CubicBSplineInterpolationCentered(in, out, kernels);
//
//
// Restrictions due to alignment of centers:
//
// Case 1: Number of control points and number of output elements are
// both odd. Then the center is located _on_ a control point. This works
// for even or odd scale factor (illustration shows scale factor 4).
//
//                    max_out_length
//          |<----------------------------->|
//          |                               |
//  # - - - # - - - # - - - # - - - # - - - # - - - #  control points
//              |           |           |
//              |        center         |
//              |           |           |
//              V           V           V
//              # # # # # # # # # # # # #   dense output
//              |                       |
//              |<--------------------->|
//                    out_num_elem
//
//
// Case 2: Number of control points and number of output elements are
// both even. Then the center is located between control points and
// between output elements. This only works for odd scale factor
// (illustration shows scale factor 5).
//
//                    max_out_length
//            |<--------------------------->|
//            |                             |
//  # - - - - # - - - - # - - - - # - - - - # - - - - #  control points
//                |          |          |
//                |        center       |
//                |          |          |
//                V          V          V
//                # # # # # # # # # # # #  dense output
//                |                     |
//                |<------------------->|
//                      out_num_elem
//
inline void CubicBSplineInterpolationCentered(
    SimpleConstArrayView in, SimpleArrayView out,
    const std::vector<BSplineKernel>& kernels) {
  CHECK_GE(in.num_elem, 3)
      << "A cubic b-spline interpolation needs at least 3 control points";
  CHECK_GE(in.num_lines, 1);
  CHECK_NE(in.stride_lines, 0);
  CHECK_NE(in.stride_elem, 0);
  CHECK_GE(out.num_elem, 1);
  CHECK_GE(out.num_lines, 1);
  CHECK_NE(out.stride_lines, 0);
  CHECK_NE(out.stride_elem, 0);

  int max_out_length = (in.num_elem - 3) * kernels.size() + 1;
  CHECK_GE(max_out_length, out.num_elem)
      << "given parameters were:"
      << "\n  in.num_lines:     " << in.num_lines
      << "\n  in.num_elem:      " << in.num_elem
      << "\n  in.stride_lines:  " << in.stride_lines
      << "\n  in.stride_elem:   " << in.stride_elem
      << "\n  out.num_elem:     " << out.num_elem
      << "\n  out.stride_lines: " << out.stride_lines
      << "\n  out.stride_elem:  " << out.stride_elem
      << "\n  kernels.size():   " << kernels.size()
      << "\n  max_out_length:   " << max_out_length;
  int double_out_offset = max_out_length - out.num_elem;
  CHECK_EQ(0, double_out_offset % 2)
      << "given parameters were:"
      << "\n  in.num_lines:     " << in.num_lines
      << "\n  in.num_elem:      " << in.num_elem
      << "\n  in.stride_lines:  " << in.stride_lines
      << "\n  in.stride_elem:   " << in.stride_elem
      << "\n  out.num_elem:     " << out.num_elem
      << "\n  out.stride_lines: " << out.stride_lines
      << "\n  out.stride_elem:  " << out.stride_elem
      << "\n  kernels.size():   " << kernels.size()
      << "\n  max_out_length:   " << max_out_length;
  int out_offset = double_out_offset / 2;
  int in_first_elem = out_offset / kernels.size();
  for (int line_i = 0; line_i < std::min(in.num_lines, out.num_lines);
       ++line_i) {
    int out_elem_i = 0;
    int start_kernel_index = out_offset % kernels.size();
    float* out_p = out.data + line_i * out.stride_lines;
    for (int elem_i = in_first_elem; elem_i < in.num_elem - 3; ++elem_i) {
      int in_offs = line_i * in.stride_lines + elem_i * in.stride_elem;
      float w0 = in.data[in_offs];
      float w1 = in.data[in_offs + in.stride_elem];
      float w2 = in.data[in_offs + 2 * in.stride_elem];
      float w3 = in.data[in_offs + 3 * in.stride_elem];
      for (int i = start_kernel_index; i < kernels.size(); ++i) {
        const BSplineKernel& k = kernels[i];
        *out_p = w0 * k.b0 + w1 * k.b1 + w2 * k.b2 + w3 * k.b3;
        out_p += out.stride_elem;
        ++out_elem_i;
        if (out_elem_i == out.num_elem) break;
      }
      if (out_elem_i == out.num_elem) break;
      start_kernel_index = 0;
    }
    // Special case for max_out_length elements, i.e. the last output
    // element lies on the second last control point. In this case,
    // only three neighbouring control points (instead of four) are
    // needed, because the kernel value of b3 is zero.
    if (max_out_length == out.num_elem) {
      int in_offs =
          line_i * in.stride_lines + (in.num_elem - 3) * in.stride_elem;
      float w0 = in.data[in_offs];
      float w1 = in.data[in_offs + in.stride_elem];
      float w2 = in.data[in_offs + 2 * in.stride_elem];
      const BSplineKernel& k = kernels[0];
      *out_p = w0 * k.b0 + w1 * k.b1 + w2 * k.b2;
    }
  }
}
}  // namespace multidim_image_augmentation
}  // namespace deepmind

#endif  // MULTIDIM_IMAGE_AUGMENTATION_KERNELS_BSPLINE_H_
