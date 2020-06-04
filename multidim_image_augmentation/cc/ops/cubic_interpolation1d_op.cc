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

#include "multidim_image_augmentation/cc/platform/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

// Creates the interface for the TensorFlow Op 'cubic_interpolation1d'.
REGISTER_OP("CubicInterpolation1D")
    .Input("input: float")
    .Output("output: float")
    .Attr("factor: int >= 1")
    .Attr("output_length: int >= 0 = 0")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
      DimensionHandle num_channels = c->Dim(input_shape, -1);
      int32 factor;
      TF_RETURN_IF_ERROR(c->GetAttr("factor", &factor));
      int32 output_length;
      TF_RETURN_IF_ERROR(c->GetAttr("output_length", &output_length));

      DimensionHandle out_shape_0;
      if (output_length > 0) {
        out_shape_0 = c->MakeDim(output_length);
      } else {
        DimensionHandle input_length = c->Dim(input_shape, 0);
        if (c->ValueKnown(input_length)) {
          out_shape_0 = c->MakeDim((c->Value(input_length) - 1) * factor + 1);
        } else {
          out_shape_0 = c->UnknownDim();
        }
      }

      c->set_output(0, c->MakeShape({out_shape_0, num_channels}));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Performs a 1D fast cubic b-spline interpolation (upscaling).

Performs a 1D fast cubic b-spline interpolation (can be interpreted as smooth
upsampling with an integer factor) where the centers of the control point array
and the dense output array are aligned. Be aware that the resulting function
usually does _not_ pass through the control points. Due to the centering
certain restrictions apply on the number of control points and the scaling
factor:

Case 1: Number of control points and number of output elements are both
odd. Then the center is located _on_ a control point. This works for even or
odd scale factor (illustration shows scale factor 4).

```
 # - - - # - - - # - - - # - - - # - - - # - - - #  control points
             |           |           |
             |        center         |
             |           |           |
             V           V           V
             # # # # # # # # # # # # #   dense output
             |                       |
             |<--------------------->|
                   output_length
```

Case 2: Number of control points and number of output elements are both
even. Then the center is located between control points and between output
elements. This only works for odd scale factor (illustration shows scale factor
5).

```
 # - - - - # - - - - # - - - - # - - - - # - - - - #  control points
               |          |          |
               |        center       |
               |          |          |
               V          V          V
               # # # # # # # # # # # #  dense output
               |                     |
               |<------------------->|
                    output_length
```
Usage example:

```python
from multidim_image_augmentation import augmentation_ops
with tf.Session():
  grid = np.ndarray([5, 2], dtype=np.float32)
  # Fill in some values.
  # ...

  # Do the bspline interpolation.
  dense = augmentation_ops.cubic_interpolation_1d(
      input=grid, factor=10, output_length=21).eval()
```

input:= A 2-D float Tensor with shape `[length, num_channels]`.
factor: Scaling factor.
output_length: The spatial length the output tensor, or 0 for maximum length, i.e. (length - 1) * factor + 1.

output: 2-D with shape `[output_length, num_channels]`
)doc");

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
