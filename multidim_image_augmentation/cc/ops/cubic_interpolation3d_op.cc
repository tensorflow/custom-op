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

#include <vector>

#include "multidim_image_augmentation/cc/platform/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

// Creates the interface for the TensorFlow Op 'cubic_interpolation3d'.
REGISTER_OP("CubicInterpolation3D")
    .Input("input: float")
    .Output("output: float")
    .Attr("factors: list(int) >= 3")
    .Attr("output_spatial_shape: list(int) >= 3")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
      DimensionHandle channels = c->Dim(input_shape, -1);
      std::vector<int32> factors;
      TF_RETURN_IF_ERROR(c->GetAttr("factors", &factors));
      std::vector<int32> output_spatial_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("output_spatial_shape",
                                    &output_spatial_shape));

      // Check attributes.
      if (factors.size() != 3) {
        return ::tensorflow::errors::InvalidArgument(
            "factors must be rank 3, got ", factors.size());
      }
      if (factors[0] <= 0 || factors[1] <= 0 || factors[2] <= 0) {
        return ::tensorflow::errors::InvalidArgument(
            "Each factor must be greater than 0, got (", factors[0], ", ",
            factors[1], ", ", factors[2], ")");
      }
      if (output_spatial_shape.size() != 3) {
        return ::tensorflow::errors::InvalidArgument(
            "output_spatial_shape must be rank 3, got ",
            output_spatial_shape.size());
      }
      if (output_spatial_shape[0] <= 0 || output_spatial_shape[1] <= 0 ||
          output_spatial_shape[2] <= 0) {
        return ::tensorflow::errors::InvalidArgument(
            "`output_spatial_shape` must be greater than 0, got (",
            output_spatial_shape[0], ", ", output_spatial_shape[1], ", ",
            output_spatial_shape[2], ")");
      }

      DimensionHandle out_shape_0 = c->MakeDim(output_spatial_shape[0]);
      DimensionHandle out_shape_1 = c->MakeDim(output_spatial_shape[1]);
      DimensionHandle out_shape_2 = c->MakeDim(output_spatial_shape[2]);
      c->set_output(
          0, c->MakeShape({out_shape_0, out_shape_1, out_shape_2, channels}));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Performs a 3D fast cubic b-spline interpolation (upscaling).

Performs a 3D fast cubic b-spline interpolation (can be interpreted as smooth
upsampling with the integer factors given in `factors`) where the centers of
the control point array and the dense output array are aligned. Be aware that
the resulting function usually does _not_ pass through the control points. Due
to the centering, certain restrictions apply on the number of control points and
the scaling factors. See `cubic_interpolation1d` for details.

```
Usage example:

```python
from multidim_image_augmentation import augmentation_ops
with tf.Session():
  grid = np.ndarray([5, 5, 5, 2], dtype=np.float32)
  # Fill in some values.
  # ...

  # Do the bspline interpolation.
  dense = augmentation_ops.cubic_interpolation_3d(
      input=grid, factors=[10, 10, 10], output_spatial_shape=[21, 21, 21]).eval()
```

input:= A 4-D float Tensor with shape `[spatial_shape_0, spatial_shape_1,
spatial_shape_2, num_channels]`.
factors: Scaling factors.
output_spatial_shape: The spatial shape of the output tensor.

output: 3-D with shape `[output_spatial_shape_0, output_spatial_shape_1,
output_spatial_shape_2, channels]`
)doc");

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
