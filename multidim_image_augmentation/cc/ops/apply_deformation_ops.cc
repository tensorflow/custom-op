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

#include "multidim_image_augmentation/cc/ops/apply_deformation_ops.h"

#include <array>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

using tensorflow::Status;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

enum SpatialDims {
  k2SpatialDims = 2,
  k3SpatialDims = 3,
};

template <SpatialDims>
Status ApplyDeformationShapeFunction(InferenceContext* context);

REGISTER_OP("ApplyDeformation2D")
    .Input("input: input_type")             // The order of these Input() calls
    .Input("deformation: float")            //  must match their order in
    .Input("padding_constant: input_type")  // `OperationInputIndices`.
    .Output("output: output_type")
    .Attr("interpolation: {'linear', 'nearest'} = 'linear'")
    .Attr(
        "extrapolation: {'mirror', 'zero_padding', 'const_padding'} = 'mirror'")
    .Attr(
        "conversion: {'no_conversion', 'indexed_to_one_hot'} = 'no_conversion'")
    .Attr("output_spatial_shape: list(int) = []")
    .Attr("output_num_channels: int = -1")
    .Attr("input_type: {float, uint8, int32} = DT_FLOAT")
    .Attr("output_type: {float, uint8, int32} = DT_FLOAT")
    .SetShapeFn(ApplyDeformationShapeFunction<k2SpatialDims>)
    .Doc(R"doc(
Applies a deformation field (vector field) to a given 2D image.

Applies a deformation field (vector field) to a given 2D image with
different interpolation, extrapolation and conversion options.
The deformation field describes the backward transformation, i.e. for each
position in the _output_ image it specifies the corresponding position in the
_input_ image:
```
   O(x) = I(D(x))

where (in the case of 2D single-channel images):

   x in R^2
   I: R^2 --> R    (input image)
   O: R^2 --> R    (output image)
   D: R^2 --> R^2  (deformation field)
```
The implementation iterates over all positions in the output image. For each
output position it fetches the corresponding input position from the
deformation field, interpolates (or extrapolates) the value at this position
in the input image and stores the resulting value in the output image. The
vectors in the deformation field must be provided as raw pixel coordinates,
i.e. (x0, x1) relative to the upper-left-front corner of the array.

Usage example for a 2D image with 3 channels:

```python
from multidim_image_augmentation import augmentation_ops
with tf.Session():
  input = np.random.random([10, 7, 3]).astype(np.float32)
  deformation = np.ndarray([10, 7, 2], dtype=np.float32)

  for x0 in range(deformation.shape[0]):
    for x1 in range(deformation.shape[1]):
      deformation[x0, x1, 0] = x0
      deformation[x0, x1, 1] = x1

  result = augmentation_ops.apply_deformation2d(
      input, deformation, [])
  self.assertEqual(result.get_shape(), input.shape)
  output = result.eval()

  self.assertAllEqual(output, input)
```

input: The multi-channel input image is a 3-D Tensor of shape
  `[s0, s1, num_channels]` where `s` stands for `spatial shape`. If
  `indexed_to_one_hot` conversion is requested the number of channels must be 1.
deformation: The deformation vector field is a 3-D Tensor of shape
  `[s0, s1, num_components]` where `s` stands for `spatial shape`.
  `num_components` must be 2. The spatial shape must be equal or larger than the
  required output image shape. If it is larger, the central part of the
  deformation field is used. This is especially useful when a segmentation
  network takes a larger source image than the target segmentation map (e.g. a
  u-net that uses valid convolutions only), but both need to be deformed with
  the same deformation field.
padding_constant:  A 1-D Tensor with shape `[num_channels]`.
  The padding value. Type must match the input image and size must match the
  number of output channels. For `zero_padding` or `mirror` extrapolations this
  tensor can be empty.
output: The multi-channel output image is a 3-D Tensor of shape
  `[s0, s1, num_channels]` where `s` stands for `spatial shape`. The spatial
  shape is usually equal to that of the deformation field. A smaller shape can
  be specified by the `output_spatial_shape` attribute. In case of
  `no_conversion` the number of channels will be identical to the input image.
  For `indexed_to_one_hot` a sufficient number of channels to store the one-hot
  encoded vectors must be specified by the `output_num_channels` attribute.
interpolation: The interpolation style determines how output
  values are computed. Select from `nearest` neighbor interpolation or
  `linear` interpolation. (Note 2D deform does not support
  `mixed_nearest_linear`.)
extrapolation: The extrapolation style determines how the output is filled where
  the deformation field calls out elements beyond the bounds of the input image.
  Select from `mirror` for extrapolation by mirroring, `zero_padding` to pad
  with zeros, or `const_padding` to pad with a constant value: see the
  `padding_constant` attribute.
conversion: On-the-fly value conversion. `no_conversion` passes
  input to output channels (e.g. 5 channel input yeilds a 5 channel output).
  `indexed_to_one_hot` converts the indexed input segmentation map (1 channel
  with values like 3 for class 3) to a one-hot-encoded output segmentation map
  (e.g. 8 channels with values like (0, 0, 0, 1, 0, 0, 0, 0) for class 3). The
  one-hot values will be collected from the neighbouring pixels. I.e. the result
  would be identical when first applying the one-hot mapping to the input image
  and then applying a deformation with linear interpolation to the resulting
  multi-channel image.
output_spatial_shape: spatial shape of output image [x0, x1]. If smaller than
  the full deformation field, the applied deformation field will be centrally
  cropped from the full deformation field. Default: use shape of the full
  deformation field.
output_num_channels: Specifies the number of output channels. If no output
  conversion is in use the output channels will match the number of input
  channels -- passing a negative (or ommitting) will do this for you. For
  `indexed_to_one_hot` conversion this must be equal or greater than the largest
  integral value + 1 in the input value space. See also the `conversion`
  attribute. Default: `output` number channels matches that of `intput`.
)doc");

REGISTER_OP("ApplyDeformation3D")
    .Input("input: input_type")             // The order of these Input() calls
    .Input("deformation: float")            //  must match their order in
    .Input("padding_constant: input_type")  // `OperationInputIndices`.
    .Output("output: output_type")
    .Attr(
        "interpolation: {'linear', 'nearest', 'mixed_nearest_linear'} = "
        "'linear'")
    .Attr(
        "extrapolation: {'mirror', 'zero_padding', 'const_padding'} = 'mirror'")
    .Attr(
        "conversion: {'no_conversion', 'indexed_to_one_hot'} = 'no_conversion'")
    .Attr("output_spatial_shape: list(int) = []")
    .Attr("output_num_channels: int = -1")
    .Attr("input_type: {float, uint8, int32} = DT_FLOAT")
    .Attr("output_type: {float, uint8, int32} = DT_FLOAT")
    .SetShapeFn(ApplyDeformationShapeFunction<k3SpatialDims>)
    .Doc(R"doc(
Applies a deformation field (vector field) to a given 3D image.

Applies a deformation field (vector field) to a given 3D image with
different interpolation, extrapolation and conversion options.
The deformation field describes the backward transformation, i.e. for each
position in the _output_ image it specifies the corresponding position in the
_input_ image:
```
   O(x) = I(D(x))

where (in the case of 3D single-channel images):

   x in R^3
   I: R^3 --> R    (input image)
   O: R^3 --> R    (output image)
   D: R^3 --> R^3  (deformation field)
```
The implementation iterates over all positions in the output image. For each
output position it fetches the corresponding input position from the
deformation field, interpolates (or extrapolates) the value at this position
in the input image and stores the resulting value in the output image. The
vectors in the deformation field must be provided as raw pixel coordinates,
i.e. (x0, x1, x2) relative to the upper-left-front corner of the array.

Usage example for a 3D image with 2 channels:

```python
from multidim_image_augmentation import augmentation_ops
with tf.Session():
  input = np.random.random([10, 7, 5, 2]).astype(np.float32)
  deformation = np.ndarray([10, 7, 5, 3], dtype=np.float32)

  for x0 in range(deformation.shape[0]):
    for x1 in range(deformation.shape[1]):
      for x2 in range(deformation.shape[2]):
        deformation[x0, x1, 0] = x0
        deformation[x0, x1, 1] = x1
        deformation[x0, x1, 2] = x2

  result = augmentation_ops.apply_deformation3D(
      input, deformation, [])
  self.assertEqual(result.get_shape(), input.shape)
  output = result.eval()

  self.assertAllEqual(output, input)
```

input: The multi-channel input image is a 4-D Tensor of shape
  `[s0, s1, s2, num_channels]` where `s` stands for `spatial shape`. If
  `indexed_to_one_hot` conversion is requested the number of channels must be 1.
deformation: The deformation vector field is a 4-D Tensor of shape
  `[s0, s1, s2, num_components]` where `s` stands for `spatial shape`.
  `num_components` must be 3.  The spatial shape must be equal or larger than
  the output image. If it is larger, the central part of the deformation field
  is used. This is especially useful when a segmentation network takes a larger
  source image than the target segmentation map (e.g. a u-net that uses valid
  convolutions only), but both need to be deformed with the same deformation
  field.
padding_constant:  A 1-D Tensor with shape `[num_channels]`.
  The padding value. Type must match the iput image and size must match the
  number of output channels. For `zero_padding` or `mirror` extrapolations this
  tensor can be empty.
output: The multi-channel output image is a 4-D tensor of shape
  `[s0, s1, s2, num_channels]` where `s` stands for `spatial shape`. The spatial
  shape is usually equal to that of the deformation field. A smaller shape can
  be specified by the `output_spatial_shape` attribute. In case of
  `no_conversion` the number of channels will be identical to the input
  image. For `indexed_to_one_hot` a sufficient number of channels to store the
  one-hot encoded vectors must be specified by the `output_num_channels`
  attribute.
interpolation: The interpolation style determines how output value are computed.
  Select from `nearest` neighbor interpolation, `linear` interpolation, or
  `mixed_nearest_linear` which uses nearest neighbour interpolation in
  x0-direction, and linear interpolation in (x1, x2)-direction. This is useful
  if there is a jitter between the slices, and you apply an non-integer scaling
  in the x0-direction.
extrapolation: The extrapolation style determines how the output is filled where
  the deformation field calls out elements beyond the bounds of the input image.
  Select from `mirror` for extrapolation by mirroring, `zero_padding` to pad
  with zeros, or `const_padding` to pad with a constant value: see the
  `padding_constant` attribute.
conversion: On-the-fly conversation. `no_conversion` provides passes
  input to output channels (e.g. 5 channel input yields a 5 channel output).
  `indexed_to_one_hot` converts the indexed input segmentation map (1 channel
  with values like 3 for class 3) to a one-hot-encoded output segmentation map
  (e.g. 8 channels with values like (0, 0, 0, 1, 0, 0, 0, 0) for class 3). The
  one-hot values will be collected from the neighbouring pixels. I.e. the result
  would be identical when first applying the one-hot mapping to the input image
  and then applying a deformation with linear interpolation to the resulting
  multi-channel image.
output_spatial_shape: spatial shape of output image [x0, x1, x2]. If smaller
  than the full  deformation field, the applied deformation field will be
  centrally cropped from the full deformation field. Default: use shape of the
  full deformation field.
output_num_channels: Specifies the number of output channels. If no output
  conversion is in use the output channels will match the number of input
  channels -- passing a negative (or ommitting) will do this for you. For
  `indexed_to_one_hot` conversion this must be equal or greater than the largest
  integral value + 1 in the input value space. See also the `conversion`
  attribute. Default: `output` number channels matches that of `intput`.
)doc");

template <SpatialDims spatial_dims>
Status ApplyDeformationShapeFunction(InferenceContext* context) {
  DeformationAttributes attrs;
  TF_RETURN_IF_ERROR(GetAttributes<Status>(context, &attrs));

  // One additional dimension for the channels.
  static const int kTensorRank = spatial_dims + 1;

  ShapeHandle input_shape, deform_shape;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(kInputIndex), kTensorRank,
                                       &input_shape));
  TF_RETURN_IF_ERROR(context->WithRank(context->input(kDeformIndex),
                                       kTensorRank, &deform_shape));

  ShapeHandle output_shape(deform_shape);

  for (int i = 0; i < attrs.requested_output_spatial_shape.size(); ++i) {
    if (attrs.requested_output_spatial_shape[i] >= 0) {
      auto dim = context->MakeDim(attrs.requested_output_spatial_shape[i]);
      TF_RETURN_IF_ERROR(
          context->ReplaceDim(output_shape, i, dim, &output_shape));
    }
  }

  DimensionHandle num_channels =
      (attrs.output_num_channels >= 0)
          ? context->MakeDim(attrs.output_num_channels)
          : context->Dim(input_shape, spatial_dims);
  // Assert that `padding_size` matches the number of channels if const padding
  // is being used.
  if (attrs.extrapolation_style == "const_padding") {
    ShapeHandle padding_shape;
    TF_RETURN_IF_ERROR(context->WithRank(context->input(kPaddingConstIndex), 1,
                                         &padding_shape));
    DimensionHandle padding_size = context->Dim(padding_shape, 0);
    TF_RETURN_IF_ERROR(
        context->Merge(padding_size, num_channels, &num_channels));
  }
  TF_RETURN_IF_ERROR(context->ReplaceDim(output_shape, spatial_dims,
                                         num_channels, &output_shape));
  context->set_output(0, output_shape);

  return Status::OK();
}

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
