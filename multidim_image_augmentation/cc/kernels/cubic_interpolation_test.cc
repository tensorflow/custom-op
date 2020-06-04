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

#include "multidim_image_augmentation/cc/kernels/cubic_interpolation.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

using ::testing::FloatEq;
using ::testing::Pointwise;

template <typename T, int N>
std::vector<T> EigenTensorToStdVector(
    Eigen::Tensor<T, N, Eigen::RowMajor> tensor) {
  return std::vector<T>(tensor.data(), tensor.data() + tensor.size());
}

TEST(CubicInterpolationTest, TestEigenTensorMemoryLayout) {
  Eigen::Tensor<int, 2, Eigen::RowMajor> A(3, 5);
  int c = 0;
  for (int i0 = 0; i0 < 3; ++i0) {
    for (int i1 = 0; i1 < 5; ++i1) {
      A(i0, i1) = c++;
    }
  }
  for (int i = 0; i < 15; ++i) {
    EXPECT_EQ(i, A.data()[i]);
  }
}

TEST(CubicInterpolationTest, TestEigenTensorMapMemoryLayout) {
  int data[3 * 5];
  Eigen::TensorMap<Eigen::Tensor<int, 2, Eigen::RowMajor> > A(data, 3, 5);
  int c = 0;
  for (int i0 = 0; i0 < 3; ++i0) {
    for (int i1 = 0; i1 < 5; ++i1) {
      A(i0, i1) = c++;
    }
  }
  for (int i = 0; i < 15; ++i) {
    EXPECT_EQ(i, A.data()[i]);
  }
}

TEST(CubicInterpolationTest, TestEigenTensorMapMemoryLayout4D) {
  int data[3 * 5 * 2 * 2];
  Eigen::TensorMap<Eigen::Tensor<int, 4, Eigen::RowMajor> > A(data, 3, 5, 2, 2);
  int c = 0;
  for (int i0 = 0; i0 < 3; ++i0) {
    for (int i1 = 0; i1 < 5; ++i1) {
      for (int i2 = 0; i2 < 2; ++i2) {
        for (int i3 = 0; i3 < 2; ++i3) {
          A(i0, i1, i2, i3) = c++;
        }
      }
    }
  }
  for (int i = 0; i < 3 * 5 * 2 * 2; ++i) {
    EXPECT_EQ(i, A.data()[i]);
  }
}

TEST(CubicInterpolationTest, Test1DInterpolation) {
  Eigen::Tensor<float, 2, Eigen::RowMajor> control_points(5, 3);  // 3 channels.
  float c = 0;
  for (int x0 = 0; x0 < 5; ++x0) {
    control_points(x0, 0) = c++;
    control_points(x0, 1) = c++;
    control_points(x0, 2) = c++;
  }
  Eigen::Tensor<float, 2, Eigen::RowMajor> dense(21, 3);
  CubicInterpolation1d(control_points, 10, &dense);
  // clang-format off
  const float expected[] = { 3.0,  4.0,  5.0,
                             3.3,  4.3,  5.3,
                             3.6,  4.6,  5.6,
                             3.9,  4.9,  5.9,
                             4.2,  5.2,  6.2,
                             4.5,  5.5,  6.5,
                             4.8,  5.8,  6.8,
                             5.1,  6.1,  7.1,
                             5.4,  6.4,  7.4,
                             5.7,  6.7,  7.7,
                             6.0,  7.0,  8.0,
                             6.3,  7.3,  8.3,
                             6.6,  7.6,  8.6,
                             6.9,  7.9,  8.9,
                             7.2,  8.2,  9.2,
                             7.5,  8.5,  9.5,
                             7.8,  8.8,  9.8,
                             8.1,  9.1, 10.1,
                             8.4,  9.4, 10.4,
                             8.7,  9.7, 10.7,
                             9.0, 10.0, 11.0};
  // clang-format on
  EXPECT_THAT(EigenTensorToStdVector(dense), Pointwise(FloatEq(), expected));

  float epsilon = 1e-5;
  EXPECT_NEAR(control_points(1, 0), dense(0, 0), epsilon);
  EXPECT_NEAR(control_points(1, 1), dense(0, 1), epsilon);
  EXPECT_NEAR(control_points(2, 0), dense(10, 0), epsilon);
  EXPECT_NEAR(control_points(3, 0), dense(20, 0), epsilon);
  EXPECT_NEAR(control_points(3, 2), dense(20, 2), epsilon);
}

TEST(CubicInterpolationTest, Test1DInterpolation_Factor1) {
  Eigen::Tensor<float, 2, Eigen::RowMajor> control_points(3, 2);  // 2 channels.
  float c = 0;
  for (int x0 = 0; x0 < 3; ++x0) {
    control_points(x0, 0) = c++;
    control_points(x0, 1) = c++;
  }
  Eigen::Tensor<float, 2, Eigen::RowMajor> dense(1, 2);
  CubicInterpolation1d(control_points, 1, &dense);
  float epsilon = 1e-5;
  EXPECT_NEAR(control_points(1, 0), dense(0, 0), epsilon);
  EXPECT_NEAR(control_points(1, 1), dense(0, 1), epsilon);
}

TEST(CubicInterpolationTest, Test2DInterpolation) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> grid(5, 5, 1);
  float c = 0;
  for (int x0 = 0; x0 < 5; ++x0) {
    for (int x1 = 0; x1 < 5; ++x1) {
      grid(x0, x1, 0) = c++;
    }
  }
  Eigen::Tensor<float, 3, Eigen::RowMajor> dense(21, 21, 1);
  CubicInterpolation2d(grid, {10, 10}, &dense);
  float epsilon = 1e-5;
  EXPECT_NEAR(grid(1, 1, 0), dense(0, 0, 0), epsilon);
  EXPECT_NEAR(grid(2, 2, 0), dense(10, 10, 0), epsilon);
  EXPECT_NEAR(grid(3, 3, 0), dense(20, 20, 0), epsilon);
}

TEST(CubicInterpolationTest, Test3DInterpolation) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> grid(5, 5, 5, 1);
  float c = 0;
  for (int x0 = 0; x0 < 5; ++x0) {
    for (int x1 = 0; x1 < 5; ++x1) {
      for (int x2 = 0; x2 < 5; ++x2) {
        grid(x0, x1, x2, 0) = c++;
      }
    }
  }
  Eigen::Tensor<float, 4, Eigen::RowMajor> dense(21, 21, 21, 1);
  dense.setConstant(-37);
  CubicInterpolation3d(grid, {10, 10, 10}, &dense);
  float epsilon =
      2e-5;  // Need a larger epsilon here, because of many interpolations.
  EXPECT_NEAR(grid(1, 1, 1, 0), dense(0, 0, 0, 0), epsilon);
  EXPECT_NEAR(grid(1, 1, 3, 0), dense(0, 0, 20, 0), epsilon);
  EXPECT_NEAR(grid(1, 3, 1, 0), dense(0, 20, 0, 0), epsilon);
  EXPECT_NEAR(grid(2, 2, 2, 0), dense(10, 10, 10, 0), epsilon);
  EXPECT_NEAR(grid(3, 1, 1, 0), dense(20, 0, 0, 0), epsilon);
  EXPECT_NEAR(grid(3, 3, 3, 0), dense(20, 20, 20, 0), epsilon);
}

TEST(CubicInterpolationTest, Test3DInterpolationSingleSlice) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> grid(3, 5, 5, 1);
  float c = 0;
  for (int x0 = 0; x0 < 3; ++x0) {
    for (int x1 = 0; x1 < 5; ++x1) {
      for (int x2 = 0; x2 < 5; ++x2) {
        grid(x0, x1, x2, 0) = c++;
      }
    }
  }
  Eigen::Tensor<float, 4, Eigen::RowMajor> dense(1, 21, 21, 1);
  dense.setConstant(-37);
  CubicInterpolation3d(grid, {1, 10, 10}, &dense);
  float epsilon = 1e-5;
  EXPECT_NEAR(grid(1, 1, 1, 0), dense(0, 0, 0, 0), epsilon);
  EXPECT_NEAR(grid(1, 1, 3, 0), dense(0, 0, 20, 0), epsilon);
  EXPECT_NEAR(grid(1, 3, 1, 0), dense(0, 20, 0, 0), epsilon);
}

TEST(CubicInterpolationTest, Test3DInterpolationWithTensorMap) {
  float raw_grid[5 * 5 * 5 * 1];
  Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor> > grid(raw_grid, 5,
                                                                   5, 5, 1);
  float c = 0;
  for (int x0 = 0; x0 < 5; ++x0) {
    for (int x1 = 0; x1 < 5; ++x1) {
      for (int x2 = 0; x2 < 5; ++x2) {
        grid(x0, x1, x2, 0) = c++;
      }
    }
  }
  Eigen::Tensor<float, 4, Eigen::RowMajor> dense(21, 21, 21, 1);
  dense.setConstant(-37);
  CubicInterpolation3d(grid, {10, 10, 10}, &dense);
  float epsilon =
      2e-5;  // Need a larger epsilon here, because of many interpolations.
  EXPECT_NEAR(grid(1, 1, 1, 0), dense(0, 0, 0, 0), epsilon);
  EXPECT_NEAR(grid(1, 1, 3, 0), dense(0, 0, 20, 0), epsilon);
  EXPECT_NEAR(grid(1, 3, 1, 0), dense(0, 20, 0, 0), epsilon);
  EXPECT_NEAR(grid(2, 2, 2, 0), dense(10, 10, 10, 0), epsilon);
  EXPECT_NEAR(grid(3, 1, 1, 0), dense(20, 0, 0, 0), epsilon);
  EXPECT_NEAR(grid(3, 3, 3, 0), dense(20, 20, 20, 0), epsilon);
}

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
