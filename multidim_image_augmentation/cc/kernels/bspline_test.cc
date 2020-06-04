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

#include "multidim_image_augmentation/cc/kernels/bspline.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

using ::testing::FloatEq;
using ::testing::Pointwise;

TEST(BsplineTest, TestCenteredInterpolation) {
  // Case 1: Both odd.
  //
  //  # - - - # - - - # - - - # - - - # - - - # - - - #
  //              |           |           |
  //              |           |           |
  //              V           V           V
  //              # # # # # # # # # # # # #
  //
  float control_points[7];
  float dense_array[13];
  for (int i = 0; i < 7; ++i) {
    control_points[i] = i;
  }
  SimpleConstArrayView in;
  in.data = control_points;
  in.num_lines = 1;
  in.num_elem = 7;
  in.stride_lines = 7;
  in.stride_elem = 1;
  SimpleArrayView out;
  out.data = dense_array;
  out.num_lines = in.num_lines;
  out.num_elem = 13;
  out.stride_lines = 13;
  out.stride_elem = 1;
  int scale_factor = 4;
  std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);

  CubicBSplineInterpolationCentered(in, out, kernels);

  EXPECT_THAT(dense_array,
              Pointwise(FloatEq(), {1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25,
                                    3.5, 3.75, 4.0, 4.25, 4.5}));
}

TEST(BsplineTest, TestCenteredInterpolation2) {
  // Case 1: Both odd.
  //
  //  # - - - # - - - # - - - # - - - # - - - # - - - #
  //                    |     |     |
  //                    |     |     |
  //                    V     V     V
  //                    # # # # # # #
  //
  float control_points[7];
  float dense_array[7];
  for (int i = 0; i < 7; ++i) {
    control_points[i] = i;
  }
  SimpleConstArrayView in;
  in.data = control_points;
  in.num_lines = 1;
  in.num_elem = 7;
  in.stride_lines = 7;
  in.stride_elem = 1;
  SimpleArrayView out;
  out.data = dense_array;
  out.num_lines = in.num_lines;
  out.num_elem = 7;
  out.stride_lines = 7;
  out.stride_elem = 1;
  int scale_factor = 4;
  std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);

  CubicBSplineInterpolationCentered(in, out, kernels);

  EXPECT_THAT(dense_array,
              Pointwise(FloatEq(), {2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75}));
}

TEST(BsplineTest, TestCenteredInterpolation3) {
  // Case 1: Both odd.
  //
  //  # - - - # - - - # - - - # - - - # - - - # - - - #
  //                  |       |       |
  //                  |       |       |
  //                  V       V       V
  //                  # # # # # # # # #
  //
  float control_points[7];
  float dense_array[9];
  for (int i = 0; i < 7; ++i) {
    control_points[i] = i;
  }
  SimpleConstArrayView in;
  in.data = control_points;
  in.num_lines = 1;
  in.num_elem = 7;
  in.stride_lines = 7;
  in.stride_elem = 1;
  SimpleArrayView out;
  out.data = dense_array;
  out.num_lines = in.num_lines;
  out.num_elem = 9;
  out.stride_lines = 9;
  out.stride_elem = 1;
  int scale_factor = 4;
  std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);

  CubicBSplineInterpolationCentered(in, out, kernels);

  EXPECT_THAT(dense_array, Pointwise(FloatEq(), {2.0, 2.25, 2.5, 2.75, 3.0,
                                                 3.25, 3.5, 3.75, 4.0}));
}

TEST(BsplineTest, TestCenteredInterpolationMaximalOutput) {
  // Case 1: Both odd.
  //
  //  # - - - # - - - # - - - # - - - #
  //          |       |       |
  //          |       |       |
  //          V       V       V
  //          # # # # # # # # #
  //
  float control_points[5];
  float dense_array[9];
  for (int i = 0; i < 5; ++i) {
    control_points[i] = i;
  }
  SimpleConstArrayView in;
  in.data = control_points;
  in.num_lines = 1;
  in.num_elem = 5;
  in.stride_lines = 5;
  in.stride_elem = 1;
  SimpleArrayView out;
  out.data = dense_array;
  out.num_lines = in.num_lines;
  out.num_elem = 9;
  out.stride_lines = 9;
  out.stride_elem = 1;
  int scale_factor = 4;
  std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);

  CubicBSplineInterpolationCentered(in, out, kernels);

  EXPECT_THAT(dense_array, Pointwise(FloatEq(), {1.0, 1.25, 1.5, 1.75, 2.0,
                                                 2.25, 2.5, 2.75, 3.0}));
}

TEST(BsplineTest, TestCenteredInterpolation4) {
  // Case 1: Both odd.
  //
  //  # - - - # - - - # - - - # - - - # - - - # - - - #
  //              |           |           |
  //              |           |           |
  //              V           V           V
  //              # # # # # # # # # # # # #
  //
  float control_points[7 * 3];
  float dense_array[13 * 3];
  for (int i = 0; i < 7 * 3; ++i) {
    control_points[i] = i;
  }
  SimpleConstArrayView in;
  in.data = control_points;
  in.num_lines = 3;
  in.num_elem = 7;
  in.stride_lines = 7;
  in.stride_elem = 1;
  SimpleArrayView out;
  out.data = dense_array;
  out.num_lines = in.num_lines;
  out.num_elem = 13;
  out.stride_lines = 13;
  out.stride_elem = 1;
  int scale_factor = 4;
  std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);

  CubicBSplineInterpolationCentered(in, out, kernels);

  // clang-format off
  EXPECT_THAT( dense_array,
               Pointwise(FloatEq(),
                         {1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
                               4.0, 4.25, 4.5,
                          8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0, 10.25, 10.5,
                               10.75, 11.0, 11.25, 11.5,
                          15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25,
                               17.5, 17.75, 18.0, 18.25, 18.5}));
  // clang-format on
}

TEST(BsplineTest, TestMinimalSize) {
  // Case 1: Both odd -- only three control points.
  //
  //  # - - - # - - - #
  //          |
  //          |
  //          V
  //          #
  //
  float control_points[3] = {1, 3, 2};
  float dense_array[1];
  SimpleConstArrayView in;
  in.data = control_points;
  in.num_lines = 1;
  in.num_elem = 3;
  in.stride_lines = 3;
  in.stride_elem = 1;
  SimpleArrayView out;
  out.data = dense_array;
  out.num_lines = in.num_lines;
  out.num_elem = 1;
  out.stride_lines = 1;
  out.stride_elem = 1;
  int scale_factor = 4;
  std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);

  CubicBSplineInterpolationCentered(in, out, kernels);

  EXPECT_THAT(dense_array[0],
              FloatEq(1.0 / 6 *
                      (control_points[0] + 4.0 * control_points[1] +
                       control_points[2])));
}

TEST(BsplineTest, TestCenteredInterpolationEven) {
  // Case 2: Both even, only works for odd CP spacing.
  //
  //                    max_out_length
  //            |<--------------------------->|
  //            |                             |
  //  # - - - - # - - - - # - - - - # - - - - # - - - - # control points
  //                |          |          |
  //                |          |          |
  //                V          V          V
  //                # # # # # # # # # # # #  dense output
  //                |                     |
  //                |<------------------->|
  //                        out_num_elem
  //
  float control_points[6];
  float dense_array[12];
  int scale_factor = 5;
  for (int i = 0; i < 6; ++i) {
    control_points[i] = i;
  }
  SimpleConstArrayView in;
  in.data = control_points;
  in.num_lines = 1;
  in.num_elem = 6;
  in.stride_lines = 6;
  in.stride_elem = 1;
  SimpleArrayView out;
  out.data = dense_array;
  out.num_lines = in.num_lines;
  out.num_elem = 12;
  out.stride_lines = 12;
  out.stride_elem = 1;
  std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);

  CubicBSplineInterpolationCentered(in, out, kernels);

  EXPECT_THAT(dense_array,
              Pointwise(FloatEq(), {1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
                                    3.2, 3.4, 3.6}));
}

TEST(BsplineTest, TestOutNumLines) {
  // Fewer output lines than input
  //
  //  # - - - # - - - #
  //          |
  //          |
  //          V
  //          #
  //
  float control_points[6] = {1, 3, 2, 6, 7, 8};
  float dense_array[2] = {200, 300};
  SimpleConstArrayView in;
  in.data = control_points;
  in.num_lines = 2;
  in.num_elem = 3;
  in.stride_lines = 3;
  in.stride_elem = 1;
  SimpleArrayView out;
  out.data = dense_array;
  out.num_lines = 1;
  out.num_elem = 1;
  out.stride_lines = 1;
  out.stride_elem = 1;
  int scale_factor = 4;
  std::vector<BSplineKernel> kernels = CreateBSplineKernels(scale_factor);

  CubicBSplineInterpolationCentered(in, out, kernels);

  EXPECT_THAT(dense_array[0],
              FloatEq(1.0 / 6 *
                      (control_points[0] + 4.0 * control_points[1] +
                       control_points[2])));
  EXPECT_EQ(dense_array[1], 300);  // Unmodified
}


}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
