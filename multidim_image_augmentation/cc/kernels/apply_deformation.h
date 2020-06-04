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

// Applies a deformation field (vector field) to a given image with different
// interpolation, extrapolation and conversion options.

#ifndef MULTIDIM_IMAGE_AUGMENTATION_KERNELS_APPLY_DEFORMATION_H_
#define MULTIDIM_IMAGE_AUGMENTATION_KERNELS_APPLY_DEFORMATION_H_

#include <cmath>
#include <vector>

#include "multidim_image_augmentation/cc/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/logging.h"

namespace deepmind {
namespace multidim_image_augmentation {

enum InterpolationStyle {
  // Nearest neighbour interpolation.
  kNearest,

  // Linear interpolation.
  kLinear,

  // Nearest neighbour interpolation in x0-direction, and linear interpolation
  // in (x1, x2)-direction. This is useful, if there is a jitter between the
  // slices, and you apply an non-integer scaling in the x0-direction.
  kMixedNearestLinear,

  kNumInterpolationStyles  // Add new styles above here.
};

enum ExtrapolationStyle {
  // Extrapolation by mirroring.
  kMirror,

  // Extrapolation by zero padding.
  kZeroPadding,

  // Extrapolation by padding with a given constant value.
  kConstPadding,

  kNumExtrapolatoinStyles  // Add new styles above here.
};

enum ConversionStyle {
  // No conversion of values (e.g. 5 channel input --> 5 channel output).
  kNoConversion,

  // Convert the indexed input segmentation map (1 channel with values like 3
  // for class 3) to a one-hot-encoded output segmentation map (e.g. 8 channels
  // with values like (0, 0, 0, 1, 0, 0, 0, 0) for class 3). The one-hot values
  // will be collected from the neighbouring pixels. I.e. I.e. the result would
  // be identical when first applying the one-hot mapping to the input image
  // and then applying a deformation with linear interpolation to the resulting
  // multi-channel image.
  kIndexedToOneHot,

  kNumConversionStyles  // Add new styles above here.
};

// Helper function for extrapolation by mirroring. Maps a position outside of
// the valid interval to the corresponding position within the valid
// interval. The interval is [0, width). This function requires width >= 1.
//
// Example for N = 5              |<----valid---->|
// x:        -7 -6 -5 -4 -3 -2 -1 | 0  1  2  3  4 | 5  6  7  8  9 10 11 12
// mapped_x:  1  2  3  4  3  2  1 | 0  1  2  3  4 | 3  2  1  0  1  2  3  4
inline int MirrorAtBoundary(int64 x, int64 width) {
  // If x is within the interval, everything is fine.
  if (0 <= x && x < width) return x;

  // If the interval has only one element, all mapped positions point to that
  // element.
  if (width == 1) return 0;

  // Map positions outside the boundaries to the corresponding position within
  // the valid interval.
  int64 mapped_x = std::abs(x) % (width * 2 - 2);
  if (mapped_x >= width) {
    mapped_x = (width * 2 - 2) - mapped_x;
  }
  return mapped_x;
}

// Interpolates a value in a 2D multi-channel array using nearest neighbor
// interpolation. The input array has the order (x0, x1, channel).
//
// Parameters:
//   in            Pointer to the input array.
//   extent_x0     Extents of the array.
//   extent_x1     -"-
//   num_channels  -"-
//   x0, x1        Position for interpolation.
//   pad_element   Pointer to padding element (num_channels components). For
//                 zero padding the caller is responsible to provide a vector
//                 with zeros here.
//   out           Pointer to output element (num_channels components).
//
// The meaning of the different options for ExtrapolationStyle and
// ConversionStyle are described above at their definitions.
template <typename InType, typename OutType,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
void Interpolate2DNearest(const InType* in, int64 extent_x0, int64 extent_x1,
                          int64 num_channels, float x0, float x1,
                          const InType* pad_element, OutType* out) {
  // Round coordinates.
  int64 int_x0 = std::floor(x0 + 0.5f);
  int64 int_x1 = std::floor(x1 + 0.5f);

  // Pointer to source pixel.
  const InType* p;
  switch (extrapolation_style) {
    case kMirror: {
      // Mirror at boundaries.
      int_x0 = MirrorAtBoundary(int_x0, extent_x0);
      int_x1 = MirrorAtBoundary(int_x1, extent_x1);
      const int64 stride0 = extent_x1 * num_channels;
      const int64 stride1 = num_channels;
      p = in + int_x0 * stride0 + int_x1 * stride1;
      break;
    }
    case kZeroPadding:
    case kConstPadding: {
      if (int_x0 >= 0 && int_x0 < extent_x0 && int_x1 >= 0 &&
          int_x1 < extent_x1) {
        const int64 stride0 = extent_x1 * num_channels;
        const int64 stride1 = num_channels;
        p = in + int_x0 * stride0 + int_x1 * stride1;
      } else {
        p = pad_element;
      }
    }
  }

  // Iterate over all channels and copy the values. If requested, apply
  // on-the-fly conversion from indexed to one-hot-encoding.
  switch (conversion_style) {
    case kNoConversion: {
      std::copy_n(p, num_channels, out);
      break;
    }
    case kIndexedToOneHot: {
      out[static_cast<int64>(*p)] = 1;
    }
  }
}

// Interpolates a value in a 2D multi-channel array using linear interpolation.
// For documentation of the parameters, see Interpolate2DNearest above.
template <typename InType, typename OutType,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
void Interpolate2DLinear(const InType* in, int64 extent_x0, int64 extent_x1,
                         int64 num_channels, float x0, float x1,
                         const InType* pad_element, OutType* out) {
  // Compute the floor and the residual part of each coordinate.
  const int64 int_x0 = std::floor(x0);
  const int64 int_x1 = std::floor(x1);
  const float res_x0 = x0 - int_x0;
  const float res_x1 = x1 - int_x1;

  // Compute weights for the 8 neighbour elements.
  const float w00 = (1.f - res_x0) * (1.f - res_x1);
  const float w01 = (1.f - res_x0) * (res_x1);
  const float w10 = (res_x0) * (1.f - res_x1);
  const float w11 = (res_x0) * (res_x1);

  // Setup pointers to the 8 neighbour elements,
  // according to the extrapolation style.
  const InType* p00;
  const InType* p01;
  const InType* p10;
  const InType* p11;
  const int64 stride0 = extent_x1 * num_channels;
  const int64 stride1 = num_channels;

  switch (extrapolation_style) {
    case kMirror: {
      // Compute valid positions for all neighbour 6 directions.
      const int64 x0_0 = MirrorAtBoundary(int_x0, extent_x0);
      const int64 x1_0 = MirrorAtBoundary(int_x1, extent_x1);
      const int64 x0_1 = MirrorAtBoundary(int_x0 + 1, extent_x0);
      const int64 x1_1 = MirrorAtBoundary(int_x1 + 1, extent_x1);

      // Pointers to the 8 neighbour elements in the first channel.
      p00 = in + x0_0 * stride0 + x1_0 * stride1;
      p01 = in + x0_0 * stride0 + x1_1 * stride1;
      p10 = in + x0_1 * stride0 + x1_0 * stride1;
      p11 = in + x0_1 * stride0 + x1_1 * stride1;
      break;
    }
    case kZeroPadding:
    case kConstPadding: {
      // Check which of the 6 neighbour directions are within bounds.
      const bool valid_x0_0 = (0 <= int_x0 && int_x0 < extent_x0);
      const bool valid_x1_0 = (0 <= int_x1 && int_x1 < extent_x1);
      const bool valid_x0_1 = (0 <= int_x0 + 1 && int_x0 + 1 < extent_x0);
      const bool valid_x1_1 = (0 <= int_x1 + 1 && int_x1 + 1 < extent_x1);

      // Pointers to 8 neighbour elements, or to pad_element if out of bounds.
      const InType* p = in + int_x0 * stride0 + int_x1 * stride1;
      p00 = (valid_x0_0 && valid_x1_0) ? p + 0 * stride0 + 0 * stride1
                                       : pad_element;
      p01 = (valid_x0_0 && valid_x1_1) ? p + 0 * stride0 + 1 * stride1
                                       : pad_element;
      p10 = (valid_x0_1 && valid_x1_0) ? p + 1 * stride0 + 0 * stride1
                                       : pad_element;
      p11 = (valid_x0_1 && valid_x1_1) ? p + 1 * stride0 + 1 * stride1
                                       : pad_element;
    }
  }

  // Iterate over all channels and do the interpolation. If requested, apply
  // on-the-fly conversion from indexed to one-hot-encoding.
  switch (conversion_style) {
    case kNoConversion: {
      for (int64 i = 0; i < num_channels; ++i) {
        out[i] = w00 * p00[i] + w01 * p01[i] + w10 * p10[i] + w11 * p11[i];
      }
      break;
    }
    case kIndexedToOneHot: {
      // Distribute the contributions of all neighbouring pixels to
      // the respective channel, i.e. do one-hot encoding on-the-fly.
      out[static_cast<int64>(*p00)] += w00;
      out[static_cast<int64>(*p01)] += w01;
      out[static_cast<int64>(*p10)] += w10;
      out[static_cast<int64>(*p11)] += w11;
    }
  }
}

// Interpolates a value in a 3D multi-channel array using nearest neighbour
// interpolation. The input array has the order (x0, x1, x2, channel).
//
// Parameters:
//   in            Pointer to the input array.
//   extent_x0     Extents of the array.
//   extent_x1     -"-
//   extent_x2     -"-
//   num_channels  -"-
//   x0, x1, x2    Position for interpolation.
//   pad_element   Pointer to padding element (num_channels components). For
//                 zero padding the caller is responsible to provide a vector
//                 with zeros here.
//   out           Pointer to output element (num_channels components).
//
// The meaning of the different options for ExtrapolationStyle and
// ConversionStyle are described above at their definitions.
//
template <typename InType, typename OutType,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
void Interpolate3DNearest(const InType* in, int64 extent_x0, int64 extent_x1,
                          int64 extent_x2, int64 num_channels, float x0,
                          float x1, float x2, const InType* pad_element,
                          OutType* out) {
  // Round coordinates.
  int64 int_x0 = std::floor(x0 + 0.5f);
  int64 int_x1 = std::floor(x1 + 0.5f);
  int64 int_x2 = std::floor(x2 + 0.5f);

  // Pointer to source pixel.
  const InType* p;
  switch (extrapolation_style) {
    case kMirror: {
      // Mirror at boundaries.
      int_x0 = MirrorAtBoundary(int_x0, extent_x0);
      int_x1 = MirrorAtBoundary(int_x1, extent_x1);
      int_x2 = MirrorAtBoundary(int_x2, extent_x2);
      const int64 stride0 = extent_x1 * extent_x2 * num_channels;
      const int64 stride1 = extent_x2 * num_channels;
      const int64 stride2 = num_channels;
      p = in + int_x0 * stride0 + int_x1 * stride1 + int_x2 * stride2;
      break;
    }
    case kZeroPadding:
    case kConstPadding: {
      if (int_x0 >= 0 && int_x0 < extent_x0 && int_x1 >= 0 &&
          int_x1 < extent_x1 && int_x2 >= 0 && int_x2 < extent_x2) {
        const int64 stride0 = extent_x1 * extent_x2 * num_channels;
        const int64 stride1 = extent_x2 * num_channels;
        const int64 stride2 = num_channels;
        p = in + int_x0 * stride0 + int_x1 * stride1 + int_x2 * stride2;
      } else {
        p = pad_element;
      }
    }
  }

  // Iterate over all channels and copy the values. If requested, apply
  // on-the-fly conversion from indexed to one-hot-encoding.
  switch (conversion_style) {
    case kNoConversion: {
      std::copy_n(p, num_channels, out);
      break;
    }
    case kIndexedToOneHot: {
      out[static_cast<int64>(*p)] = 1;
    }
  }
}

// Interpolates a value in a 3D multi-channel array using linear interpolation.
// For documentation of the parameters, see Interpolate3DNearest above.
template <typename InType, typename OutType,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
void Interpolate3DLinear(const InType* in, int64 extent_x0, int64 extent_x1,
                         int64 extent_x2, int64 num_channels, float x0,
                         float x1, float x2, const InType* pad_element,
                         OutType* out) {
  // Compute the floor and the residual part of each coordinate.
  const int64 int_x0 = std::floor(x0);
  const int64 int_x1 = std::floor(x1);
  const int64 int_x2 = std::floor(x2);
  const float res_x0 = x0 - int_x0;
  const float res_x1 = x1 - int_x1;
  const float res_x2 = x2 - int_x2;

  // Compute weights for the 8 neighbour elements.
  const float w000 = (1.f - res_x0) * (1.f - res_x1) * (1.f - res_x2);
  const float w001 = (1.f - res_x0) * (1.f - res_x1) * (res_x2);
  const float w010 = (1.f - res_x0) * (res_x1) * (1.f - res_x2);
  const float w011 = (1.f - res_x0) * (res_x1) * (res_x2);
  const float w100 = (res_x0) * (1.f - res_x1) * (1.f - res_x2);
  const float w101 = (res_x0) * (1.f - res_x1) * (res_x2);
  const float w110 = (res_x0) * (res_x1) * (1.f - res_x2);
  const float w111 = (res_x0) * (res_x1) * (res_x2);

  // Setup pointers to the 8 neighbour elements,
  // according to the extrapolation style.
  const InType* p000;
  const InType* p001;
  const InType* p010;
  const InType* p011;
  const InType* p100;
  const InType* p101;
  const InType* p110;
  const InType* p111;
  const int64 stride0 = extent_x1 * extent_x2 * num_channels;
  const int64 stride1 = extent_x2 * num_channels;
  const int64 stride2 = num_channels;

  switch (extrapolation_style) {
    case kMirror: {
      // Compute valid positions for all neighbour 6 directions.
      const int64 x0_0 = MirrorAtBoundary(int_x0, extent_x0);
      const int64 x1_0 = MirrorAtBoundary(int_x1, extent_x1);
      const int64 x2_0 = MirrorAtBoundary(int_x2, extent_x2);
      const int64 x0_1 = MirrorAtBoundary(int_x0 + 1, extent_x0);
      const int64 x1_1 = MirrorAtBoundary(int_x1 + 1, extent_x1);
      const int64 x2_1 = MirrorAtBoundary(int_x2 + 1, extent_x2);

      // Pointers to the 8 neighbour elements in the first channel.
      p000 = in + x0_0 * stride0 + x1_0 * stride1 + x2_0 * stride2;
      p001 = in + x0_0 * stride0 + x1_0 * stride1 + x2_1 * stride2;
      p010 = in + x0_0 * stride0 + x1_1 * stride1 + x2_0 * stride2;
      p011 = in + x0_0 * stride0 + x1_1 * stride1 + x2_1 * stride2;
      p100 = in + x0_1 * stride0 + x1_0 * stride1 + x2_0 * stride2;
      p101 = in + x0_1 * stride0 + x1_0 * stride1 + x2_1 * stride2;
      p110 = in + x0_1 * stride0 + x1_1 * stride1 + x2_0 * stride2;
      p111 = in + x0_1 * stride0 + x1_1 * stride1 + x2_1 * stride2;
      break;
    }
    case kZeroPadding:
    case kConstPadding: {
      // Check which of the 6 neighbour directions are within bounds.
      const bool valid_x0_0 = (0 <= int_x0 && int_x0 < extent_x0);
      const bool valid_x1_0 = (0 <= int_x1 && int_x1 < extent_x1);
      const bool valid_x2_0 = (0 <= int_x2 && int_x2 < extent_x2);
      const bool valid_x0_1 = (0 <= int_x0 + 1 && int_x0 + 1 < extent_x0);
      const bool valid_x1_1 = (0 <= int_x1 + 1 && int_x1 + 1 < extent_x1);
      const bool valid_x2_1 = (0 <= int_x2 + 1 && int_x2 + 1 < extent_x2);

      // Pointers to 8 neighbour elements, or to pad_element if out of bounds.
      const InType* p =
          in + int_x0 * stride0 + int_x1 * stride1 + int_x2 * stride2;
      p000 = (valid_x0_0 && valid_x1_0 && valid_x2_0)
                 ? p + 0 * stride0 + 0 * stride1 + 0 * stride2
                 : pad_element;
      p001 = (valid_x0_0 && valid_x1_0 && valid_x2_1)
                 ? p + 0 * stride0 + 0 * stride1 + 1 * stride2
                 : pad_element;
      p010 = (valid_x0_0 && valid_x1_1 && valid_x2_0)
                 ? p + 0 * stride0 + 1 * stride1 + 0 * stride2
                 : pad_element;
      p011 = (valid_x0_0 && valid_x1_1 && valid_x2_1)
                 ? p + 0 * stride0 + 1 * stride1 + 1 * stride2
                 : pad_element;
      p100 = (valid_x0_1 && valid_x1_0 && valid_x2_0)
                 ? p + 1 * stride0 + 0 * stride1 + 0 * stride2
                 : pad_element;
      p101 = (valid_x0_1 && valid_x1_0 && valid_x2_1)
                 ? p + 1 * stride0 + 0 * stride1 + 1 * stride2
                 : pad_element;
      p110 = (valid_x0_1 && valid_x1_1 && valid_x2_0)
                 ? p + 1 * stride0 + 1 * stride1 + 0 * stride2
                 : pad_element;
      p111 = (valid_x0_1 && valid_x1_1 && valid_x2_1)
                 ? p + 1 * stride0 + 1 * stride1 + 1 * stride2
                 : pad_element;
    }
  }

  // Iterate over all channels and do the interpolation. If requested, apply
  // on-the-fly conversion from indexed to one-hot-encoding.
  switch (conversion_style) {
    case kNoConversion: {
      for (int64 i = 0; i < num_channels; ++i) {
        out[i] = w000 * p000[i] + w001 * p001[i] + w010 * p010[i] +
                 w011 * p011[i] + w100 * p100[i] + w101 * p101[i] +
                 w110 * p110[i] + w111 * p111[i];
      }
      break;
    }
    case kIndexedToOneHot: {
      // Distribute the contributions of all neighbouring pixels to
      // the respective channel, i.e. do one-hot encoding on-the-fly.
      out[static_cast<int64>(*p000)] += w000;
      out[static_cast<int64>(*p001)] += w001;
      out[static_cast<int64>(*p010)] += w010;
      out[static_cast<int64>(*p011)] += w011;
      out[static_cast<int64>(*p100)] += w100;
      out[static_cast<int64>(*p101)] += w101;
      out[static_cast<int64>(*p110)] += w110;
      out[static_cast<int64>(*p111)] += w111;
    }
  }
}

// Interpolates a value in a 3D multi-channel array using mixed interpolation:
// Nearest neighbor in x0-direction and linear interpolation in x1,x2-direction.
template <typename InType, typename OutType,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
void Interpolate3DMixedNearestLinear(const InType* in, int64 extent_x0,
                                     int64 extent_x1, int64 extent_x2,
                                     int64 num_channels, float x0, float x1,
                                     float x2, const InType* pad_element,
                                     OutType* out) {
  // Round coordinate in x0 direction.
  int64 int_x0 = std::floor(x0 + 0.5f);

  // Pointer to source slice.
  const InType* slice;
  switch (extrapolation_style) {
    case kMirror: {
      // Mirror at boundaries.
      int_x0 = MirrorAtBoundary(int_x0, extent_x0);
      const int64 stride0 = extent_x1 * extent_x2 * num_channels;
      slice = in + int_x0 * stride0;
      break;
    }
    case kZeroPadding:
    case kConstPadding: {
      if (int_x0 >= 0 && int_x0 < extent_x0) {
        const int64 stride0 = extent_x1 * extent_x2 * num_channels;
        slice = in + int_x0 * stride0;
      } else {
        slice = pad_element;
      }
    }
  }

  // If we are on a valid slice, do 2D linear interpolation there
  if (slice != pad_element) {
    Interpolate2DLinear<InType, OutType, extrapolation_style, conversion_style>(
        slice, extent_x1, extent_x2, num_channels, x1, x2, pad_element, out);
  } else {
    // copy pad_element to output. If requested, apply
    // on-the-fly conversion from indexed to one-hot-encoding.
    switch (conversion_style) {
      case kNoConversion: {
        std::copy_n(pad_element, num_channels, out);
        break;
      }
      case kIndexedToOneHot: {
        out[static_cast<int64>(*pad_element)] = 1;
      }
    }
  }
}

// Perform an optimized Transform2D. The default implementation returns false to
// indicate it did not run.
template <typename InTensor, typename DeformTensor, typename OutTensor,
          InterpolationStyle interpolation_style,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style>
class OptimizedTransform2D {
 public:
  static bool Run(const InTensor& in, const DeformTensor& deform,
                  const typename InTensor::Scalar* padding_constant,
                  OutTensor* out) {
    return false;
  }
};

// Performs the 2D deformation. Helper function for ApplyDeformation::Deform2D.
template <typename InTensor, typename DeformTensor, typename OutTensor,
          typename Functor>
static void Transform2D(const InTensor& in, const DeformTensor& deform,
                        Functor Interpolator,
                        const typename InTensor::Scalar* padding_constant,
                        OutTensor* out_p) {
  using InType = typename InTensor::Scalar;
  using DeformType = typename DeformTensor::Scalar;
  using OutType = typename OutTensor::Scalar;

  OutTensor& out = *out_p;

  const int64 in_extent_x0 = in.dimension(0);
  const int64 in_extent_x1 = in.dimension(1);
  const int64 num_channels = in.dimension(2);
  const int64 out_extent_x0 = out.dimension(0);
  const int64 out_extent_x1 = out.dimension(1);

  // Use central part of deformation map if target image is smaller.
  const int64 offset0 = (deform.dimension(0) - out_extent_x0) / 2;
  const int64 offset1 = (deform.dimension(1) - out_extent_x1) / 2;

  // create a zero-padding vector, if necessary
  const InType* padding_constant_p;
  std::vector<InType> zero_padding;
  if (padding_constant != nullptr) {
    padding_constant_p = padding_constant;
  } else {
    zero_padding.resize(num_channels, 0);
    padding_constant_p = zero_padding.data();
  }

  const InType* in_p = &in(0, 0, 0);
  for (int64 x0 = 0; x0 < out_extent_x0; ++x0) {
    const DeformType* deform_iter = &deform(offset0 + x0, offset1, 0);
    OutType* out_iter = &out(x0, 0, 0);
    for (int64 x1 = 0; x1 < out_extent_x1; ++x1) {
      Interpolator(in_p, in_extent_x0, in_extent_x1, num_channels,
                   deform_iter[0], deform_iter[1], padding_constant_p,
                   out_iter);
      deform_iter += 2;
      out_iter += out.dimension(2);
    }
  }
}

// Performs the 3D deformation. Helper function for ApplyDeformation::Deform3D.
template <typename InTensor, typename DeformTensor, typename OutTensor,
          typename Functor>
static void Transform3D(const InTensor& in, const DeformTensor& deform,
                        Functor Interpolator,
                        const typename InTensor::Scalar* padding_constant,
                        OutTensor* out_p) {
  using InType = typename InTensor::Scalar;
  using DeformType = typename DeformTensor::Scalar;
  using OutType = typename OutTensor::Scalar;

  OutTensor& out = *out_p;

  const int64 in_extent_x0 = in.dimension(0);
  const int64 in_extent_x1 = in.dimension(1);
  const int64 in_extent_x2 = in.dimension(2);
  const int64 num_channels = in.dimension(3);
  const int64 out_extent_x0 = out.dimension(0);
  const int64 out_extent_x1 = out.dimension(1);
  const int64 out_extent_x2 = out.dimension(2);

  // Use central part of deformation map if target image is smaller.
  const int64 offset0 = (deform.dimension(0) - out_extent_x0) / 2;
  const int64 offset1 = (deform.dimension(1) - out_extent_x1) / 2;
  const int64 offset2 = (deform.dimension(2) - out_extent_x2) / 2;

  // create a zero-padding vector, if necessary
  const InType* padding_constant_p;
  std::vector<InType> zero_padding;
  if (padding_constant != nullptr) {
    padding_constant_p = padding_constant;
  } else {
    zero_padding.resize(num_channels, 0);
    padding_constant_p = zero_padding.data();
  }

  const InType* in_p = &in(0, 0, 0, 0);
  for (int64 x0 = 0; x0 < out_extent_x0; ++x0) {
    for (int64 x1 = 0; x1 < out_extent_x1; ++x1) {
      const DeformType* deform_iter =
          &deform(offset0 + x0, offset1 + x1, offset2, 0);
      OutType* out_iter = &out(x0, x1, 0, 0);
      for (int64 x2 = 0; x2 < out_extent_x2; ++x2) {
        Interpolator(in_p, in_extent_x0, in_extent_x1, in_extent_x2,
                     num_channels, deform_iter[0], deform_iter[1],
                     deform_iter[2], padding_constant_p, out_iter);
        deform_iter += 3;
        out_iter += out.dimension(3);
      }
    }
  }
}

// Applies a deformation field (vector field) to a given image. The deformation
// field describes the backward transformation, i.e. for each position in the
// _output_ image it specifies the corresponding position in the _input_ image:
//
//   O(x) = I(D(x))
//
// where (in the case of 3D single-channel images):
//
//   x in R^3
//   I: R^3 --> R    (input image)
//   O: R^3 --> R    (output image)
//   D: R^3 --> R^3  (deformation field)
//
// The implementation iterates over all positions in the output image. For each
// output position it fetches the corresponding input position from the
// deformation field, interpolates (or extrapolates) the value at this position
// in the input image and stores the resulting value in the output image. The
// vectors in the deformation field must be provided as raw pixel coordinates,
// i.e. (x0, x1, x2) relative to the upper-left-front corner of the array.
// Example usage:
//
//   // 3D images with 2 channels.
//   Eigen::Tensor<float, 4, Eigen::RowMajor> in(4, 10, 7, 2);
//   Eigen::Tensor<float, 4, Eigen::RowMajor> out(4, 10, 7, 2);
//   Eigen::Tensor<float, 4, Eigen::RowMajor> deform(4, 10, 7, 3);
//
//   // Initialize images.
//   in.setRandom();
//   out.setZero();
//
//   // Initialize deformation field as identity transformation.
//   for (int x0 = 0; x0 < deform.dimension(0); ++x0) {
//     for (int x1 = 0; x1 < deform.dimension(1); ++x1) {
//       for (int x2 = 0; x2 < deform.dimension(2); ++x2) {
//         deform(x0, x1, x2, 0) = x0;
//         deform(x0, x1, x2, 1) = x1;
//         deform(x0, x1, x2, 2) = x2;
//       }
//     }
//   }
//
//   // Apply deformation with linear interpolation, zero padding extrapolation
//   // and no conversion of the intensities.
//   ApplyDeformation<kLinear, kZeroPadding, kNoConversion>::Deform3D(in,
//                                                                    deform,
//                                                                    &out);
//
// The meaning of the different options for InterpolationStyle,
// ExtrapolationStyle and ConversionStyle are described above at their
// definitions.
//
// All tensors (in, out, deform) must be 3-D or 4-D (for Deform2D and Deform3D
// respectively) and have a RowMajor layout with shape `[extent_x0, extent_x1,
// (extent_x2,) num_channels]` . `in` and `out` can be single-channel
// (num_channels = 1) or multi-channel images. In case of `kNoConversion` the
// number of channels must be identical. For `kIndexedToOneHot` the input image
// (usually a segmentation map) must be single-channel, and the output image
// must have enough channels to store the one-hot-encoding.
//
// ATTENTION (for `kIndexedToOneHot`): If the input segmentation map contains a
// value outside the interval [0, number of output channels), this function
// will die with an error.
//
// The input image and the output image can have arbitrary spatial extents. The
// deformation field must be as large as the output image or larger. If it is
// larger, the central part of the deformation field is used. This is
// especially useful when a segmentation network takes a larger input image
// than the output segmentation map (e.g. a u-net that uses valid convolutions
// only), but both need to be deformed with the same deformation field.
//
template <InterpolationStyle interpolation_style,
          ExtrapolationStyle extrapolation_style,
          ConversionStyle conversion_style,
          bool use_avx_optimizations = true>
class ApplyDeformation {
 public:
  // Deforms a 2-D multi-channel array (3-D Tensor). See class documentation for
  // details.
  template <typename InTensor, typename DeformTensor, typename OutTensor>
  static void Deform2D(
      const InTensor& in, const DeformTensor& deform, OutTensor* out_p,
      const typename InTensor::Scalar* padding_constant = nullptr) {
    OutTensor& out = *out_p;
    static_assert(interpolation_style != kMixedNearestLinear,
                  "`kMixedNearestLinear` can not be used for 2D deformation.");
    static_assert(static_cast<int>(InTensor::Layout) == Eigen::RowMajor,
                  "Input Tensor must have row major layout.");
    static_assert(InTensor::NumIndices == 3, "Input Tensor must be 3-D.");
    static_assert(static_cast<int>(DeformTensor::Layout) == Eigen::RowMajor,
                  "Deform Tensor must have row major layout.");
    static_assert(DeformTensor::NumIndices == 3, "Deform Tensor must be 3-D.");
    static_assert(static_cast<int>(OutTensor::Layout) == Eigen::RowMajor,
                  "Output Tensor must have row major layout.");
    static_assert(OutTensor::NumIndices == 3, "Output Tensor must be 3-D.");
    if (conversion_style == kIndexedToOneHot) {
      DCHECK_EQ(in.dimension(2), 1) << "Input image must have 1 channel for "
                                       "indexed-to-one-hot conversion.";
      // Check if all values in the input segmentation map are in the allowed
      // interval [0, number of output channels)
      for (int64 i = 0; i < in.size(); ++i) {
        DCHECK_GE(in.data()[i], 0)
            << "Input image (segmentation map) must only contain "
               "positive values. Value at index "
            << i << " failed.";
        DCHECK_LE(in.data()[i], out.dimension(2))
            << "Value " << in.data()[i]
            << " in input segmentation map at position " << i
            << " cannot be represented as one-hot-encoding in a vector with "
               "only "
            << out.dimension(2) << " elements.";
      }
    } else {
      DCHECK_EQ(in.dimension(2), out.dimension(2))
          << "`in` and `out` must have same number of channels, if no "
             "conversion is selected.";
    }
    DCHECK_EQ(deform.dimension(2), 2)
        << "Deformation field must have 2 channels.";
    DCHECK_GE(deform.dimension(0), out.dimension(0))
        << "Deformation field size in x0 direction must be greater or equal "
           "than output image size.";
    DCHECK_GE(deform.dimension(1), out.dimension(1))
        << "Deformation field size in x1 direction must be greater or equal "
           "than output image size.";
    DCHECK_EQ((deform.dimension(0) - out.dimension(0)) % 2, 0)
        << "Difference bewteen deformation field size and output image size "
           "in x0 direction must be even.";
    DCHECK_EQ((deform.dimension(1) - out.dimension(1)) % 2, 0)
        << "Difference bewteen deformation field size and output image size "
           "in x1 direction must be even.";

    using InType = typename InTensor::Scalar;
    using OutType = typename OutTensor::Scalar;

    // For one-hot-encoding, initialise the output tensor to zero.
    if (conversion_style == kIndexedToOneHot) {
      out_p->setZero();
    }

    if (use_avx_optimizations &&
        OptimizedTransform2D<InTensor, DeformTensor, OutTensor,
                             interpolation_style, extrapolation_style,
                             conversion_style>::Run(in, deform,
                                                    padding_constant, out_p)) {
      return;
    }
    switch (interpolation_style) {
      case kNearest: {
        Transform2D(in, deform,
                    Interpolate2DNearest<InType, OutType, extrapolation_style,
                                         conversion_style>,
                    padding_constant, out_p);
        break;
      }
      case kLinear: {
        Transform2D(in, deform,
                    Interpolate2DLinear<InType, OutType, extrapolation_style,
                                        conversion_style>,
                    padding_constant, out_p);
        break;
      }
      default: {
        LOG(ERROR) << "Unsupported interpolation style.";
        break;
      }
    }
  }

  // Deforms a 3-D multi-channel array (4-D Tensor). See class documentation for
  // details.
  template <typename InTensor, typename DeformTensor, typename OutTensor>
  static void Deform3D(
      const InTensor& in, const DeformTensor& deform, OutTensor* out_p,
      const typename InTensor::Scalar* padding_constant = nullptr) {
    OutTensor& out = *out_p;
    static_assert(static_cast<int>(InTensor::Layout) == Eigen::RowMajor,
                  "Input Tensor must have row major layout.");
    static_assert(InTensor::NumIndices == 4, "Input Tensor must be 4-D.");
    static_assert(static_cast<int>(DeformTensor::Layout) == Eigen::RowMajor,
                  "Deform Tensor must have row major layout.");
    static_assert(DeformTensor::NumIndices == 4, "Deform Tensor must be 4-D.");
    static_assert(static_cast<int>(OutTensor::Layout) == Eigen::RowMajor,
                  "Output Tensor must have row major layout.");
    static_assert(OutTensor::NumIndices == 4, "Output Tensor must be 4-D.");
    if (conversion_style == kIndexedToOneHot) {
      DCHECK_EQ(in.dimension(3), 1) << "Input image must have 1 channel for "
                                       "indexed-to-one-hot conversion.";
      // Check if all values in the input segmentation map are in the allowed
      // interval [0, number of output channels)
      for (int64 i = 0; i < in.size(); ++i) {
        DCHECK_GE(in.data()[i], 0)
            << "Input image (segmentation map) must only contain "
               "positive values. Value at index "
            << i << " failed.";
        DCHECK_LE(in.data()[i], out.dimension(3))
            << "Value " << in.data()[i]
            << " in input segmentation map at position " << i
            << " cannot be represented as one-hot-encoding in a vector with "
               "only "
            << out.dimension(3) << " elements.";
      }
    } else {
      DCHECK_EQ(in.dimension(3), out.dimension(3))
          << "`in` and `out` must have same number of channels, if no "
             "conversion is selected.";
    }
    DCHECK_EQ(deform.dimension(3), 3)
        << "Deformation field must have 3 channels.";
    DCHECK_GE(deform.dimension(0), out.dimension(0))
        << "Deformation field size in x0 direction must be greater or equal "
           "than output image size.";
    DCHECK_GE(deform.dimension(1), out.dimension(1))
        << "Deformation field size in x1 direction must be greater or equal "
           "than output image size.";
    DCHECK_GE(deform.dimension(2), out.dimension(2))
        << "Deformation field size in x2 direction must be greater or equal "
           "than output image size.";
    DCHECK_EQ((deform.dimension(0) - out.dimension(0)) % 2, 0)
        << "Difference bewteen deformation field size and output image size "
           "in x0 direction must be even.";
    DCHECK_EQ((deform.dimension(1) - out.dimension(1)) % 2, 0)
        << "Difference bewteen deformation field size and output image size "
           "in x1 direction must be even.";
    DCHECK_EQ((deform.dimension(2) - out.dimension(2)) % 2, 0)
        << "Difference bewteen deformation field size and output image size "
           "in x2 direction must be even.";

    using InType = typename InTensor::Scalar;
    using OutType = typename OutTensor::Scalar;

    // For one-hot-encoding, initialise the output tensor to zero.
    if (conversion_style == kIndexedToOneHot) {
      out_p->setZero();
    }

    switch (interpolation_style) {
      case kNearest: {
        Transform3D(in, deform,
                    Interpolate3DNearest<InType, OutType, extrapolation_style,
                                         conversion_style>,
                    padding_constant, out_p);
        break;
      }
      case kLinear: {
        Transform3D(in, deform,
                    Interpolate3DLinear<InType, OutType, extrapolation_style,
                                        conversion_style>,
                    padding_constant, out_p);
        break;
      }
      case kMixedNearestLinear: {
        Transform3D(in, deform,
                    Interpolate3DMixedNearestLinear<
                        InType, OutType, extrapolation_style, conversion_style>,
                    padding_constant, out_p);
        break;
      }
    }
  }
};

}  // namespace multidim_image_augmentation
}  // namespace deepmind

#ifdef __AVX2__
#endif

#endif  // MULTIDIM_IMAGE_AUGMENTATION_KERNELS_APPLY_DEFORMATION_H_
