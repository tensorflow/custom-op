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

#include <iterator>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/test_benchmark.h"

namespace deepmind {
namespace multidim_image_augmentation {

namespace {

using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Pointwise;

template <typename T, int N>
std::vector<T> EigenTensorToStdVector(
    const Eigen::Tensor<T, N, Eigen::RowMajor>& tensor) {
  return std::vector<T>(tensor.data(), tensor.data() + tensor.size());
}

template <typename T>
void SetEigenTensor2DToIdentity(Eigen::Tensor<T, 3, Eigen::RowMajor>* tensor) {
  for (int x0 = 0; x0 < tensor->dimension(0); ++x0) {
    for (int x1 = 0; x1 < tensor->dimension(1); ++x1) {
      (*tensor)(x0, x1, 0) = x0;
      (*tensor)(x0, x1, 1) = x1;
    }
  }
}

template <typename T>
void SetEigenTensor3DToIdentity(Eigen::Tensor<T, 4, Eigen::RowMajor>* tensor) {
  for (int x0 = 0; x0 < tensor->dimension(0); ++x0) {
    for (int x1 = 0; x1 < tensor->dimension(1); ++x1) {
      for (int x2 = 0; x2 < tensor->dimension(2); ++x2) {
        (*tensor)(x0, x1, x2, 0) = x0;
        (*tensor)(x0, x1, x2, 1) = x1;
        (*tensor)(x0, x1, x2, 2) = x2;
      }
    }
  }
}

TEST(MirrorAtBoundaryTest, Basics) {
  // clang-format off
  int mapped_x[20];
  int x[20] =
      {-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  for (int i = 0; i < 20; ++i) {
    mapped_x[i] = MirrorAtBoundary(x[i], 5);
  }

  EXPECT_THAT(mapped_x, Pointwise(Eq(),
      { 1,  2,  3,  4,  3,  2,  1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1,  2,  3,  4}));
  // clang-format on
}

//
// Tests for Interpolate2DNearest
//
TEST(Interpolate2DNearestTest, ExtrapolationMirror) {
  const int extent_x0 = 2;
  const int extent_x1 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  Interpolate2DNearest<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0., 0., nullptr, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate2DNearest<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0.9, 0.1, nullptr, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {12., 13., 14.}));
  Interpolate2DNearest<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0., -1.4, nullptr, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {3., 4., 5.}));
}

TEST(Interpolate2DNearestTest, ExtrapolationConstPad) {
  const int extent_x0 = 2;
  const int extent_x1 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  float pad_element[num_channels] = {42., 37., 13};
  Interpolate2DNearest<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0., 0., pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate2DNearest<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0.9, 0.1, pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {12., 13., 14.}));
  Interpolate2DNearest<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0., -1.4, pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), pad_element));
}

TEST(Interpolate2DNearestTest, IndexedToOneHot) {
  const int extent_x0 = 1;
  const int extent_x1 = 1;
  const int num_channels = 1;
  const int out_num_channels = 10;
  float in[extent_x0 * extent_x1 * num_channels];
  float out[1 * out_num_channels] = {0.};
  in[0] = 3;
  Interpolate2DNearest<float, float, kMirror, kIndexedToOneHot>(
      in, extent_x0, extent_x1, num_channels, 0., 0., nullptr, out);
  EXPECT_THAT(out,
              Pointwise(FloatEq(), {0., 0., 0., 1., 0., 0., 0., 0., 0., 0.}));
}

//
// Tests for Interpolate2DLinear
//
TEST(Interpolate2DLinearTest, ExtrapolationMirror) {
  const int extent_x0 = 2;
  const int extent_x1 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  Interpolate2DLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0., 0., nullptr, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate2DLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0.9, 0., nullptr, out);
  // clang-format off
  EXPECT_THAT(out, Pointwise(FloatEq(), {0. * 0.1 + 12. * 0.9,
                                         1. * 0.1 + 13. * 0.9,
                                         2. * 0.1 + 14. * 0.9}));
  Interpolate2DLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0., -2.4, nullptr,
      out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {6. * 0.6 +  9. * 0.4,
                                         7. * 0.6 + 10. * 0.4,
                                         8. * 0.6 + 11. * 0.4}));
  // clang-format on
}

TEST(Interpolate2DLinearTest, ExtrapolationConstPad) {
  const int extent_x0 = 2;
  const int extent_x1 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  float pad_element[num_channels] = {42., 37., 13};
  Interpolate2DLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0., 0., pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate2DLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0.9, 0., pad_element, out);
  // clang-format off
  EXPECT_THAT(out, Pointwise(FloatEq(), {0. * 0.1 + 12. * 0.9,
                                         1. * 0.1 + 13. * 0.9,
                                         2. * 0.1 + 14. * 0.9}));
  // clang-format on
  Interpolate2DLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, 0., -1.4, pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), pad_element));

  Interpolate2DLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, num_channels, -0.7, 2., pad_element, out);
  // clang-format off
  EXPECT_THAT(out, Pointwise(FloatEq(), {6. * 0.3 + 42. * 0.7,
                                         7. * 0.3 + 37. * 0.7,
                                         8. * 0.3 + 13. * 0.7}));
  // clang-format on
}

TEST(Interpolate2DLinearTest, IndexedToOneHot) {
  const int extent_x0 = 1;
  const int extent_x1 = 2;
  const int num_channels = 1;
  const int out_num_channels = 10;
  float in[extent_x0 * extent_x1 * num_channels];
  float out[1 * out_num_channels] = {0.};
  in[0] = 3;
  in[1] = 7;
  Interpolate2DLinear<float, float, kMirror, kIndexedToOneHot>(
      in, extent_x0, extent_x1, num_channels, 0., 0.42, nullptr, out);
  EXPECT_THAT(
      out, Pointwise(FloatEq(), {0., 0., 0., 0.58, 0., 0., 0., 0.42, 0., 0.}));
}

//
// Tests for Interpolate3DNearest
//
TEST(Interpolate3DNearestTest, ExtrapolationMirror) {
  const int extent_x0 = 2;
  const int extent_x1 = 3;
  const int extent_x2 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  Interpolate3DNearest<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0., nullptr,
      out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate3DNearest<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0.9, 0.1, 0.2, nullptr,
      out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {36., 37., 38.}));
  Interpolate3DNearest<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., -1.4, nullptr,
      out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {3., 4., 5.}));
}

TEST(Interpolate3DNearestTest, ExtrapolationConstPad) {
  const int extent_x0 = 2;
  const int extent_x1 = 3;
  const int extent_x2 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  float pad_element[num_channels] = {42., 37., 13};
  Interpolate3DNearest<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0.,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate3DNearest<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0.9, 0.1, 0.2,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {36., 37., 38.}));
  Interpolate3DNearest<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., -1.4,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), pad_element));
}

TEST(Interpolate3DNearestTest, IndexedToOneHot) {
  const int extent_x0 = 1;
  const int extent_x1 = 1;
  const int extent_x2 = 1;
  const int num_channels = 1;
  const int out_num_channels = 10;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  float out[1 * out_num_channels] = {0.};
  in[0] = 3;
  Interpolate3DNearest<float, float, kMirror, kIndexedToOneHot>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0., nullptr,
      out);
  EXPECT_THAT(out,
              Pointwise(FloatEq(), {0., 0., 0., 1., 0., 0., 0., 0., 0., 0.}));
}

//
// Tests for Interpolate3DLinear
//
TEST(Interpolate3DLinearTest, ExtrapolationMirror) {
  const int extent_x0 = 2;
  const int extent_x1 = 3;
  const int extent_x2 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  Interpolate3DLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0., nullptr,
      out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate3DLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0.9, 0., 0., nullptr,
      out);
  // clang-format off
  EXPECT_THAT(out, Pointwise(FloatEq(), {0. * 0.1 + 36. * 0.9,
                                         1. * 0.1 + 37. * 0.9,
                                         2. * 0.1 + 38. * 0.9}));
  Interpolate3DLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., -2.4, nullptr,
      out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {6. * 0.6 +  9. * 0.4,
                                         7. * 0.6 + 10. * 0.4,
                                         8. * 0.6 + 11. * 0.4}));
  // clang-format on
}

TEST(Interpolate3DLinearTest, ExtrapolationConstPad) {
  const int extent_x0 = 2;
  const int extent_x1 = 3;
  const int extent_x2 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  float pad_element[num_channels] = {42., 37., 13};
  Interpolate3DLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0.,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate3DLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0.9, 0., 0.,
      pad_element, out);
  // clang-format off
  EXPECT_THAT(out, Pointwise(FloatEq(), {0. * 0.1 + 36. * 0.9,
                                         1. * 0.1 + 37. * 0.9,
                                         2. * 0.1 + 38. * 0.9}));
  // clang-format on
  Interpolate3DLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., -1.4,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), pad_element));

  Interpolate3DLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., -0.7, 2.,
      pad_element, out);
  // clang-format off
  EXPECT_THAT(out, Pointwise(FloatEq(), {6. * 0.3 + 42. * 0.7,
                                         7. * 0.3 + 37. * 0.7,
                                         8. * 0.3 + 13. * 0.7}));
  // clang-format on
}

TEST(Interpolate3DLinearTest, IndexedToOneHot) {
  const int extent_x0 = 1;
  const int extent_x1 = 1;
  const int extent_x2 = 2;
  const int num_channels = 1;
  const int out_num_channels = 10;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  float out[1 * out_num_channels] = {0.};
  in[0] = 3;
  in[1] = 7;
  Interpolate3DLinear<float, float, kMirror, kIndexedToOneHot>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0.42, nullptr,
      out);
  EXPECT_THAT(
      out, Pointwise(FloatEq(), {0., 0., 0., 0.58, 0., 0., 0., 0.42, 0., 0.}));
}

//
// Tests for Interpolate3DMixedNearestLinear
//
TEST(Interpolate3DMixedNearestLinearTest, ExtrapolationMirror) {
  const int extent_x0 = 2;
  const int extent_x1 = 3;
  const int extent_x2 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  Interpolate3DMixedNearestLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0., nullptr,
      out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate3DMixedNearestLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0.9, 0., 0., nullptr,
      out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {36., 37., 38.}));
  Interpolate3DMixedNearestLinear<float, float, kMirror, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0.4, 0., -2.4, nullptr,
      out);
  // clang-format off
  EXPECT_THAT(out, Pointwise(FloatEq(), {6. * 0.6 +  9. * 0.4,
                                         7. * 0.6 + 10. * 0.4,
                                         8. * 0.6 + 11. * 0.4}));
  // clang-format on
}

TEST(Interpolate3DMixedNearestLinearTest, ExtrapolationConstPad) {
  const int extent_x0 = 2;
  const int extent_x1 = 3;
  const int extent_x2 = 4;
  const int num_channels = 3;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  std::iota(std::begin(in), std::end(in), 0.0f);

  float out[num_channels];
  float pad_element[num_channels] = {42., 37., 13};
  Interpolate3DMixedNearestLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0.,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {0., 1., 2.}));
  Interpolate3DMixedNearestLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0.9, 0., 0.,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), {36., 37., 38.}));
  Interpolate3DMixedNearestLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., -1.4,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), pad_element));
  Interpolate3DMixedNearestLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 4., 0., -2,
      pad_element, out);
  EXPECT_THAT(out, Pointwise(FloatEq(), pad_element));

  Interpolate3DMixedNearestLinear<float, float, kConstPadding, kNoConversion>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., -0.7, 2.,
      pad_element, out);
  // clang-format off
  EXPECT_THAT(out, Pointwise(FloatEq(), {6. * 0.3 + 42. * 0.7,
                                         7. * 0.3 + 37. * 0.7,
                                         8. * 0.3 + 13. * 0.7}));
  // clang-format on
}

TEST(Interpolate3DMixedNearestLinearTest, IndexedToOneHot) {
  const int extent_x0 = 1;
  const int extent_x1 = 1;
  const int extent_x2 = 2;
  const int num_channels = 1;
  const int out_num_channels = 10;
  float in[extent_x0 * extent_x1 * extent_x2 * num_channels];
  float out[1 * out_num_channels] = {0.};
  in[0] = 3;
  in[1] = 7;
  Interpolate3DMixedNearestLinear<float, float, kMirror, kIndexedToOneHot>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 0., 0., 0.42, nullptr,
      out);
  EXPECT_THAT(
      out, Pointwise(FloatEq(), {0., 0., 0., 0.58, 0., 0., 0., 0.42, 0., 0.}));

  float pad_element = 5;
  float out2[1 * out_num_channels] = {0.};
  Interpolate3DMixedNearestLinear<float, float, kConstPadding,
                                  kIndexedToOneHot>(
      in, extent_x0, extent_x1, extent_x2, num_channels, 3., 0., 0.42,
      &pad_element, out2);
  EXPECT_THAT(out2,
              Pointwise(FloatEq(), {0., 0., 0., 0., 0., 1., 0., 0., 0., 0.}));
}

//
// Tests for ApplyDeformation::Deform2D
//
TEST(Deform2DTest, IdentityTransform) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(10, 7, 3);   // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(10, 7, 3);  // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(10, 7, 2);

  in.setRandom();
  out.setZero();
  SetEigenTensor2DToIdentity(&deform);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(in)));
}

TEST(Deform2DTest, IdentityTransformOneChannel) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(1, 8, 1);   // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 8, 1);  // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 8, 2);

  in.setRandom();
  out.setZero();
  SetEigenTensor2DToIdentity(&deform);

  ApplyDeformation<kNearest, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                    &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(in)));
}

TEST(Deform2DTest, IdentityTransformOneChannelLinear) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(1, 8, 1);   // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 8, 1);  // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 8, 2);

  in.setRandom();
  out.setZero();
  SetEigenTensor2DToIdentity(&deform);

  ApplyDeformation<kNearest, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                    &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(in)));
}

TEST(Deform2DTest, Uint8ToFloat) {
  Eigen::Tensor<uint8, 3, Eigen::RowMajor> in(4, 7, 3);   // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(4, 7, 3);  // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(4, 7, 2);

  std::iota(in.data(), in.data() + in.size(), 0);
  out.setZero();
  SetEigenTensor2DToIdentity(&deform);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(in)));
}  // namespace

TEST(Deform2DTest, Uint8ToUint8) {
  Eigen::Tensor<uint8, 3, Eigen::RowMajor> in(4, 10, 3);   // e.g. RGB image
  Eigen::Tensor<uint8, 3, Eigen::RowMajor> out(4, 10, 3);  // e.g. RGB image
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(4, 10, 2);

  std::iota(in.data(), in.data() + in.size(), 0);
  out.setZero();
  SetEigenTensor2DToIdentity(&deform);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(in)));
}

TEST(Deform2DTest, ExtrapolationMirror) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(1, 5, 1);
  in.setValues({{{0}, {1}, {2}, {3}, {4}}});

  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 21, 2);
  deform.setValues(
      {{{0, -10}, {0, -9}, {0, -8}, {0, -7}, {0, -6}, {0, -5}, {0, -4},
        {0, -3},  {0, -2}, {0, -1}, {0, 0},  {0, 1},  {0, 2},  {0, 3},
        {0, 4},   {0, 5},  {0, 6},  {0, 7},  {0, 8},  {0, 9},  {0, 10}}});

  Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 21, 1);

  ApplyDeformation<kLinear, kMirror, kNoConversion>::Deform2D(in, deform, &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {2., 1., 0., 1., 2., 3., 4., 3., 2., 1., 0.,
                                    1., 2., 3., 4., 3., 2., 1., 0., 1., 2.}));
}

TEST(Deform2DTest, ExtrapolationZero) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(1, 5, 1);
  in.setValues({{{10}, {11}, {12}, {13}, {14}}});

  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 21, 2);
  deform.setValues(
      {{{0, -10}, {0, -9}, {0, -8}, {0, -7}, {0, -6}, {0, -5}, {0, -4},
        {0, -3},  {0, -2}, {0, -1}, {0, 0},  {0, 1},  {0, 2},  {0, 3},
        {0, 4},   {0, 5},  {0, 6},  {0, 7},  {0, 8},  {0, 9},  {0, 10}}});

  Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 21, 1);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                   &out);

  EXPECT_THAT(
      EigenTensorToStdVector(out),
      Pointwise(FloatEq(), {0.,  0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 10.,
                            11., 12., 13., 14., 0., 0., 0., 0., 0., 0.}));
}

TEST(Deform2DTest, ExtrapolationConst) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(1, 5, 1);
  in.setValues({{{10}, {11}, {12}, {13}, {14}}});

  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 21, 2);
  deform.setValues(
      {{{0, -10}, {0, -9}, {0, -8}, {0, -7}, {0, -6}, {0, -5}, {0, -4},
        {0, -3},  {0, -2}, {0, -1}, {0, 0},  {0, 1},  {0, 2},  {0, 3},
        {0, 4},   {0, 5},  {0, 6},  {0, 7},  {0, 8},  {0, 9},  {0, 10}}});

  Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 21, 1);

  float padding_constant = 42;
  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(
      in, deform, &out, &padding_constant);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(),
                        {42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 10.,
                         11., 12., 13., 14., 42., 42., 42., 42., 42., 42.}));
}

TEST(Deform2DTest, LinearInterpolation) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(2, 2, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(2, 2, 2);
  // clang-format off
  in.setValues({{{0}, {1}},
                {{2}, {3}}});
  deform.setValues(
      {{{0,   0.5}, {1, 1.25} },
       {{0.5, 0.5}, {1, 0.25} }});
  // clang-format on

  Eigen::Tensor<float, 3, Eigen::RowMajor> out(2, 2, 1);

  ApplyDeformation<kLinear, kMirror, kNoConversion>::Deform2D(in, deform, &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {0.5, 2.75, 1.5, 2.25}));

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {0.5, 2.25, 1.5, 2.25}));
}

TEST(Deform2DTest, LinearInterpolationUint8) {
  Eigen::Tensor<uint8, 3, Eigen::RowMajor> in(2, 2, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(2, 2, 2);
  // clang-format off
  in.setValues({{{0}, {10}},
                {{20}, {30}}});
  deform.setValues(
      {{{0,   0.5}, {1, 1.25}},
       {{0.5, 0.5}, {1, 0.25}}});
  // clang-format on

  Eigen::Tensor<uint8, 3, Eigen::RowMajor> out(2, 2, 1);

  ApplyDeformation<kLinear, kMirror, kNoConversion>::Deform2D(in, deform, &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {5, 27, 15, 22}));

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {5, 22, 15, 22}));
}

TEST(Deform2DTest, NearestInterpolation) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(2, 2, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(2, 2, 2);
  // clang-format off
  in.setValues({{{0}, {1}},
                {{2}, {3}}});
  deform.setValues(
      {{{0,   0.5}, {1, 1.75}},
       {{0.5, 0.5}, {1, 0.25}}});
  // clang-format on

  Eigen::Tensor<float, 3, Eigen::RowMajor> out(2, 2, 1);

  ApplyDeformation<kNearest, kMirror, kNoConversion>::Deform2D(in, deform,
                                                               &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {1., 2., 3., 2.}));

  ApplyDeformation<kNearest, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                    &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {1., 0., 3., 2.}));
}

//
// Tests for AVX2 implementations of ApplyDeformation::Deform2D
//

TEST(Deform2DTest, LinearInterpolation_AVX2) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(2, 9, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 8, 2);
  // clang-format off
  in.setValues({{{0}, {1}, {4}, {5}, {8},  {9},  {12}, {13}, {14}},
                {{2}, {3}, {6}, {7}, {10}, {11}, {14}, {15}, {16}}});
  deform.setValues(
      {{{1, 0}, {0, 1}, {0.3, 1.2}, {0.25, 2.1}, {0.5, 0.5}, {0.8, 4.6},
           {0.5, 5.6}, {0.25, 5.7} }});
  // clang-format on

  Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 8, 1);

  // Run without AVX.
  ApplyDeformation<kLinear, kZeroPadding, kNoConversion,
                   /*use_avx_optimizations=*/false>::Deform2D(in, deform, &out);

  EXPECT_THAT(
      EigenTensorToStdVector(out),
      Pointwise(FloatEq(), {2.0, 1.0, 2.2, 4.6, 1.5, 10.2, 11.8, 11.6}));

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                   &out);

  EXPECT_THAT(
      EigenTensorToStdVector(out),
      Pointwise(FloatEq(), {2.0, 1.0, 2.2, 4.6, 1.5, 10.2, 11.8, 11.6}));
}

TEST(Deform2DTest, NearestInterpolation_AVX2) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(2, 9, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 8, 2);
  // clang-format off
  in.setValues({{{0}, {1}, {4}, {5}, {8},  {9},  {12}, {13}, {14}},
                {{2}, {3}, {6}, {7}, {10}, {11}, {14}, {15}, {16}}});
  deform.setValues(
      {{{1, 0}, {0, 1}, {0.3, 1.2}, {0.25, 2.1}, {0.5, 0.5}, {0.8, 4.6},
            {0.5, 5.6}, {0.25, 5.7} }});
  // clang-format on

  Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 8, 1);

  // Run without AVX.
  ApplyDeformation<kNearest, kZeroPadding, kNoConversion,
                   /*use_avx_optimizations=*/false>::Deform2D(in, deform, &out);

  EXPECT_THAT(
      EigenTensorToStdVector(out),
      Pointwise(FloatEq(), {2.0, 1.0, 1.0, 4.0, 3.0, 11.0, 14.0, 12.0}));

  ApplyDeformation<kNearest, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                    &out);

  EXPECT_THAT(
      EigenTensorToStdVector(out),
      Pointwise(FloatEq(), {2.0, 1.0, 1.0, 4.0, 3.0, 11.0, 14.0, 12.0}));
}

TEST(Deform2DTest, NearestInterpolation_AVX2_Random) {
  const int64 kNumTests = 1000;
  for (int i = 0; i < kNumTests; ++i) {
    Eigen::Tensor<float, 3, Eigen::RowMajor> in(2, 9, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 8, 2);
    // clang-format off
    in.setValues({{{0}, {1}, {4}, {5}, {8},  {9},  {12}, {13}, {14}},
                  {{2}, {3}, {6}, {7}, {10}, {11}, {14}, {15}, {16}}});
    // clang-format on
    deform.setRandom();
    deform = deform * 5.0f;

    Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 8, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> out_opt(1, 8, 1);

    ApplyDeformation<kNearest, kZeroPadding, kNoConversion, false>::Deform2D(
        in, deform, &out);

    ApplyDeformation<kNearest, kZeroPadding, kNoConversion, true>::Deform2D(
        in, deform, &out_opt);

    EXPECT_THAT(EigenTensorToStdVector(out),
                Pointwise(FloatEq(), EigenTensorToStdVector(out_opt)));
  }
}

TEST(Deform2DTest, LinearInterpolation_AVX2_Random) {
  const int64 kNumTests = 1000;
  for (int i = 0; i < kNumTests; ++i) {
    Eigen::Tensor<float, 3, Eigen::RowMajor> in(2, 9, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> deform(1, 8, 2);
    // clang-format off
    in.setValues({{{0}, {1}, {4}, {5}, {8},  {9},  {12}, {13}, {14}},
                  {{2}, {3}, {6}, {7}, {10}, {11}, {14}, {15}, {16}}});
    // clang-format on
    deform.setRandom();
    deform = deform * 5.0f;

    Eigen::Tensor<float, 3, Eigen::RowMajor> out(1, 8, 1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> out_opt(1, 8, 1);

    ApplyDeformation<kLinear, kZeroPadding, kNoConversion, false>::Deform2D(
        in, deform, &out);

    ApplyDeformation<kLinear, kZeroPadding, kNoConversion, true>::Deform2D(
        in, deform, &out_opt);

    ASSERT_THAT(EigenTensorToStdVector(out_opt),
                Pointwise(FloatNear(1e-5), EigenTensorToStdVector(out)));
  }
}

//
// Tests for ApplyDeformation::Deform3D
//
TEST(Deform3DTest, IdentityTransform) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(4, 10, 7, 3);   // e.g. RGB image
  Eigen::Tensor<float, 4, Eigen::RowMajor> out(4, 10, 7, 3);  // e.g. RGB image
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(4, 10, 7, 3);

  in.setRandom();
  out.setZero();
  SetEigenTensor3DToIdentity(&deform);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(in)));
}

TEST(Deform3DTest, Uint8ToFloat) {
  Eigen::Tensor<uint8, 4, Eigen::RowMajor> in(4, 10, 7, 3);   // e.g. RGB image
  Eigen::Tensor<float, 4, Eigen::RowMajor> out(4, 10, 7, 3);  // e.g. RGB image
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(4, 10, 7, 3);

  std::iota(in.data(), in.data() + in.size(), 0);
  out.setZero();
  SetEigenTensor3DToIdentity(&deform);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(in)));
}

TEST(Deform3DTest, Uint8ToUint8) {
  Eigen::Tensor<uint8, 4, Eigen::RowMajor> in(4, 10, 7, 3);   // e.g. RGB image
  Eigen::Tensor<uint8, 4, Eigen::RowMajor> out(4, 10, 7, 3);  // e.g. RGB image
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(4, 10, 7, 3);

  std::iota(in.data(), in.data() + in.size(), 0);
  out.setZero();
  SetEigenTensor3DToIdentity(&deform);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(in)));
}

TEST(Deform3DTest, ExtrapolationMirror) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(1, 1, 5, 1);
  in.setValues({{{{0}, {1}, {2}, {3}, {4}}}});

  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(1, 1, 21, 3);
  deform.setValues(
      {{{{0, 0, -10}, {0, 0, -9}, {0, 0, -8}, {0, 0, -7}, {0, 0, -6},
         {0, 0, -5},  {0, 0, -4}, {0, 0, -3}, {0, 0, -2}, {0, 0, -1},
         {0, 0, 0},   {0, 0, 1},  {0, 0, 2},  {0, 0, 3},  {0, 0, 4},
         {0, 0, 5},   {0, 0, 6},  {0, 0, 7},  {0, 0, 8},  {0, 0, 9},
         {0, 0, 10}}}});

  Eigen::Tensor<float, 4, Eigen::RowMajor> out(1, 1, 21, 1);

  ApplyDeformation<kLinear, kMirror, kNoConversion>::Deform3D(in, deform, &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {2., 1., 0., 1., 2., 3., 4., 3., 2., 1., 0.,
                                    1., 2., 3., 4., 3., 2., 1., 0., 1., 2.}));
}

TEST(Deform3DTest, ExtrapolationZero) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(1, 1, 5, 1);
  in.setValues({{{{10}, {11}, {12}, {13}, {14}}}});

  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(1, 1, 21, 3);
  deform.setValues(
      {{{{0, 0, -10}, {0, 0, -9}, {0, 0, -8}, {0, 0, -7}, {0, 0, -6},
         {0, 0, -5},  {0, 0, -4}, {0, 0, -3}, {0, 0, -2}, {0, 0, -1},
         {0, 0, 0},   {0, 0, 1},  {0, 0, 2},  {0, 0, 3},  {0, 0, 4},
         {0, 0, 5},   {0, 0, 6},  {0, 0, 7},  {0, 0, 8},  {0, 0, 9},
         {0, 0, 10}}}});

  Eigen::Tensor<float, 4, Eigen::RowMajor> out(1, 1, 21, 1);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(in, deform,
                                                                   &out);

  EXPECT_THAT(
      EigenTensorToStdVector(out),
      Pointwise(FloatEq(), {0.,  0.,  0.,  0.,  0., 0., 0., 0., 0., 0., 10.,
                            11., 12., 13., 14., 0., 0., 0., 0., 0., 0.}));
}

TEST(Deform3DTest, ExtrapolationZeroManyChannels) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(1, 1, 1, 1000);
  std::iota(in.data(), in.data() + in.size(), 0);

  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(1, 1, 1, 3);
  deform.setValues({{{{0, 0, -10}}}});

  Eigen::Tensor<float, 4, Eigen::RowMajor> out(1, 1, 1, 1000);

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(in, deform,
                                                                   &out);

  Eigen::Tensor<float, 4, Eigen::RowMajor> zeros(1, 1, 1, 1000);
  zeros.setZero();
  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), EigenTensorToStdVector(zeros)));
}

TEST(Deform3DTest, ExtrapolationConst) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(1, 1, 5, 1);
  in.setValues({{{{10}, {11}, {12}, {13}, {14}}}});

  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(1, 1, 21, 3);
  deform.setValues(
      {{{{0, 0, -10}, {0, 0, -9}, {0, 0, -8}, {0, 0, -7}, {0, 0, -6},
         {0, 0, -5},  {0, 0, -4}, {0, 0, -3}, {0, 0, -2}, {0, 0, -1},
         {0, 0, 0},   {0, 0, 1},  {0, 0, 2},  {0, 0, 3},  {0, 0, 4},
         {0, 0, 5},   {0, 0, 6},  {0, 0, 7},  {0, 0, 8},  {0, 0, 9},
         {0, 0, 10}}}});

  Eigen::Tensor<float, 4, Eigen::RowMajor> out(1, 1, 21, 1);

  float padding_constant = 42;
  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(
      in, deform, &out, &padding_constant);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(),
                        {42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 10.,
                         11., 12., 13., 14., 42., 42., 42., 42., 42., 42.}));
}

TEST(Deform3DTest, LinearInterpolation) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(2, 2, 2, 1);
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(1, 2, 2, 3);
  // clang-format off
  in.setValues({{{{0}, {1}},
                 {{2}, {3}}},
                {{{0}, {0}},
                 {{0}, {3}}}});
  deform.setValues(
      {{{{0,   0,   0.5}, {0, 1, 1.25}},
        {{0.5, 0.5, 0.5}, {1, 1, 0.25}}}});
  // clang-format on

  Eigen::Tensor<float, 4, Eigen::RowMajor> out(1, 2, 2, 1);

  ApplyDeformation<kLinear, kMirror, kNoConversion>::Deform3D(in, deform, &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {0.5, 2.75, 1.125, 0.75}));

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {0.5, 2.25, 1.125, 0.75}));
}

TEST(Deform3DTest, LinearInterpolationUint8) {
  Eigen::Tensor<uint8, 4, Eigen::RowMajor> in(2, 2, 2, 1);
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(1, 2, 2, 3);
  // clang-format off
  in.setValues({{{{0},  {10}},
                 {{20}, {30}}},
                {{{0},  {0}},
                 {{0},  {30}}}});
  deform.setValues(
      {{{{0,   0,   0.5}, {0, 1, 1.25}},
        {{0.5, 0.5, 0.5}, {1, 1, 0.25}}}});
  // clang-format on

  Eigen::Tensor<uint8, 4, Eigen::RowMajor> out(1, 2, 2, 1);

  ApplyDeformation<kLinear, kMirror, kNoConversion>::Deform3D(in, deform, &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {5, 27, 11, 7}));

  ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(in, deform,
                                                                   &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {5, 22, 11, 7}));
}

TEST(Deform3DTest, NearestInterpolation) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(2, 2, 2, 1);
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(1, 2, 2, 3);
  // clang-format off
  in.setValues({{{{0}, {1}},
                 {{2}, {3}}},
                {{{0}, {0}},
                 {{0}, {3}}}});
  deform.setValues(
      {{{{0,   0,   0.5}, {0, 1, 1.75}},
        {{0.5, 0.5, 0.5}, {1, 1, 0.25}}}});
  // clang-format on

  Eigen::Tensor<float, 4, Eigen::RowMajor> out(1, 2, 2, 1);

  ApplyDeformation<kNearest, kMirror, kNoConversion>::Deform3D(in, deform,
                                                               &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {1., 2., 3., 0.}));

  ApplyDeformation<kNearest, kZeroPadding, kNoConversion>::Deform3D(in, deform,
                                                                    &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {1., 0., 3., 0.}));
}

TEST(Deform3DTest, MixedNearestLinearInterpolation) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(2, 2, 2, 1);
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(1, 2, 2, 3);
  // clang-format off
  in.setValues({{{{0}, {1}},
                 {{2}, {3}}},
                {{{0}, {0}},
                 {{0}, {3}}}});
  deform.setValues(
      {{{{0,   0,   0.5}, {0, 1, 1.25}},
        {{0.5, 0.5, 0.5}, {1, 1, 0.25}}}});
  // clang-format on

  Eigen::Tensor<float, 4, Eigen::RowMajor> out(1, 2, 2, 1);

  ApplyDeformation<kMixedNearestLinear, kMirror, kNoConversion>::Deform3D(
      in, deform, &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {0.5, 2.75, 0.75, 0.75}));

  ApplyDeformation<kMixedNearestLinear, kZeroPadding, kNoConversion>::Deform3D(
      in, deform, &out);

  EXPECT_THAT(EigenTensorToStdVector(out),
              Pointwise(FloatEq(), {0.5, 2.25, 0.75, 0.75}));
}

//
// Benchmarks for ApplyDeformation
//
// Run these with:
// bazel run -c opt --dynamic_mode=off --copt=-gmlt <path> -- --benchmarks=all
//

// TODO: Get these working in OSS build.
#if defined(PLATFORM_GOOGLE)

// Hits the AVX2 optimized fast-path for 1-channel nearest-neighbor, but
// disables the fast-path.
static void BM_Deform2D3968To3072_NoAvx(benchmark::State& state) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(3968, 3072, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(3072, 2304, 2);
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(3072, 2304, 1);

  in.setRandom();
  SetEigenTensor2DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kNearest, kZeroPadding, kNoConversion,
                     /*use_avx_optimizations=*/false>::Deform2D(in, deform,
                                                                &out);
  }
}

BENCHMARK(BM_Deform2D3968To3072_NoAvx);

// Hits the AVX2 optimized fast-path for 1-channel nearest-neighbor.
static void BM_Deform2D3968To3072_Avx(benchmark::State& state) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(3968, 3072, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(3072, 2304, 2);
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(3072, 2304, 1);

  in.setRandom();
  SetEigenTensor2DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kNearest, kZeroPadding, kNoConversion>::Deform2D(
        in, deform, &out);
  }
}

BENCHMARK(BM_Deform2D3968To3072_Avx);

static void BM_Deform2D3968To3072_Linear_NoAvx(benchmark::State& state) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(3968, 3072, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(3072, 2304, 2);
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(3072, 2304, 1);

  in.setRandom();
  SetEigenTensor2DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kLinear, kZeroPadding, kNoConversion,
                     /*use_avx_optimizations=*/false>::Deform2D(in, deform,
                                                                &out);
  }
}

BENCHMARK(BM_Deform2D3968To3072_Linear_NoAvx);

// Hits the AVX2 optimized fast-path for 1-channel nearest-neighbor.
static void BM_Deform2D3968To3072_Linear_Avx(benchmark::State& state) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(3968, 3072, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(3072, 2304, 2);
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(3072, 2304, 1);

  in.setRandom();
  SetEigenTensor2DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                     &out);
  }
}

BENCHMARK(BM_Deform2D3968To3072_Linear_Avx);

static void BM_Deform2D512To400(benchmark::State& state) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(512, 512, 3);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(400, 400, 2);
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(400, 400, 3);

  in.setRandom();
  SetEigenTensor2DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform2D(in, deform,
                                                                     &out);
  }
}

BENCHMARK(BM_Deform2D512To400);

static void BM_Deform2D512To400OneHot(benchmark::State& state) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> in(512, 512, 1);
  Eigen::Tensor<float, 3, Eigen::RowMajor> deform(400, 400, 2);
  Eigen::Tensor<float, 3, Eigen::RowMajor> out(400, 400, 1);

  in.setRandom();
  SetEigenTensor2DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kLinear, kZeroPadding, kIndexedToOneHot>::Deform2D(
        in, deform, &out);
  }
}

BENCHMARK(BM_Deform2D512To400OneHot);

static void BM_Deform3D256To128(benchmark::State& state) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(256, 256, 256, 3);
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(128, 128, 128, 3);
  Eigen::Tensor<float, 4, Eigen::RowMajor> out(128, 128, 128, 3);

  in.setRandom();
  SetEigenTensor3DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kLinear, kMirror, kNoConversion>::Deform3D(in, deform,
                                                                &out);
  }
}

BENCHMARK(BM_Deform3D256To128);

static void BM_Deform3D256To128Nearest(benchmark::State& state) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(256, 256, 256, 3);
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(128, 128, 128, 3);
  Eigen::Tensor<float, 4, Eigen::RowMajor> out(128, 128, 128, 3);

  in.setRandom();
  SetEigenTensor3DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kNearest, kMirror, kNoConversion>::Deform3D(in, deform,
                                                                 &out);
  }
}

BENCHMARK(BM_Deform3D256To128Nearest);

static void BM_Deform3D256To128Mixed(benchmark::State& state) {
  Eigen::Tensor<float, 4, Eigen::RowMajor> in(256, 256, 256, 3);
  Eigen::Tensor<float, 4, Eigen::RowMajor> deform(128, 128, 128, 3);
  Eigen::Tensor<float, 4, Eigen::RowMajor> out(128, 128, 128, 3);

  in.setRandom();
  SetEigenTensor3DToIdentity(&deform);

  for (auto _ : state) {
    ApplyDeformation<kMixedNearestLinear, kMirror, kNoConversion>::Deform3D(
        in, deform, &out);
  }
}

BENCHMARK(BM_Deform3D256To128Mixed);

#endif  // #if defined(PLATFORM_GOOGLE)

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
