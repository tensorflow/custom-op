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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

REGISTER_OP("ApplyTabulatedFunctions")
    .Input("input: input_type")
    .Input("tabulated_functions: output_type")
    .Output("output: output_type")
    .Attr("offset: float = 0.0")
    .Attr("scale: float = 1.0")
    .Attr("input_type: {float, int64, int32, uint8} = DT_FLOAT")
    .Attr("output_type: {float, int64, int32, uint8} = DT_FLOAT")
    .SetShapeFn(tensorflow::shape_inference::UnchangedShape)
    .Doc(R"doc(
Applies tabulated piecewise linear functions.

Every element of the input tensor is used as index for the
tabulated_function to produce the corresponding output. The values for
non-integer indices are linearly inerpolated bewteen the two neighbours. Values
for indices outside the table are linearly extrapolated. Each channel has
its own lookup-table. In pseudo code (ignoring the boundary cases):

    x = scale * (offset + input[...,channel])
    i = floor(x)
    w = x - i
    output[..., channel] = (1 - w) * tabulated_functions[channel, i]
                               + w * tabulated_functions[channel, i+1]

input: n-D float Tensor: multi-channel source tensor of any dimension (... ,
  channel)
tabulated_functions: 2-D float Tensor containing the tabulated functions
  (channel, index)
offset: offest to compute the index in the tabulated function
scale: scale factor to compute the index in the tabulated function

output: multi-channel target tensor with same shape as the input tensor
)doc");

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
