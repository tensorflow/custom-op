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

// Common utility functions shared between the appy_deformation op and kernel.

#ifndef MULTIDIM_IMAGE_AUGMENTATION_OPS_APPLY_DEFORMATION_OPS_H_
#define MULTIDIM_IMAGE_AUGMENTATION_OPS_APPLY_DEFORMATION_OPS_H_

#include <string>
#include <vector>

namespace deepmind {
namespace multidim_image_augmentation {

// The ApplyDeformation2D/3D operator inputs are referred to by their
// zero-based index within that declaration throughout the rest of the code,
// thus the values of this enum must match the order of Input() calls in their
// corresponding REGISTER_OP calls.
enum OperationInputIndices {
  kInputIndex = 0,
  kDeformIndex = 1,
  kPaddingConstIndex = 2,
};

struct DeformationAttributes {
  std::string interpolation_style;
  std::string extrapolation_style;
  std::string conversion_style;
  std::vector<int> requested_output_spatial_shape;
  int output_num_channels;
};

// Helper to populate the operation attributes from either an InferenceContext
// or an OpKernelConstruction instance (or any other object supporting a
// compatible GetAttr() method).
template <typename Status, typename Context>
Status GetAttributes(Context* context, DeformationAttributes* attrs_out) {
  Status s = context->GetAttr("interpolation", &attrs_out->interpolation_style);
  if (!s.ok()) return s;

  s = context->GetAttr("extrapolation", &attrs_out->extrapolation_style);
  if (!s.ok()) return s;

  s = context->GetAttr("conversion", &attrs_out->conversion_style);
  if (!s.ok()) return s;

  s = context->GetAttr("output_spatial_shape",
                       &attrs_out->requested_output_spatial_shape);
  if (!s.ok()) return s;

  s = context->GetAttr("output_num_channels", &attrs_out->output_num_channels);
  if (!s.ok()) return s;

  return s;
}

}  // namespace multidim_image_augmentation
}  // namespace deepmind

#endif  // MULTIDIM_IMAGE_AUGMENTATION_OPS_APPLY_DEFORMATION_OPS_H_
