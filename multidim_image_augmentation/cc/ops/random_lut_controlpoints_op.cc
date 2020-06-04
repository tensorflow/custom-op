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

#include <random>

#include "multidim_image_augmentation/cc/platform/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;

uint64 RandomSeed() {
  std::random_device device("/dev/urandom");
  std::mt19937_64 seed_rng(device());
  return seed_rng();
}

REGISTER_OP("RandomLUTControlPoints")
    .SetIsStateful()
    .Output("output: float")
    .Attr("new_black_range: list(float)")
    .Attr("new_white_range: list(float)")
    .Attr("slope_min: float = 0.8")
    .Attr("slope_max: float = 1.2")
    .Attr("num_control_point_insertions: int = 2")
    .SetShapeFn([](InferenceContext* c) {
      int32 num_control_points;
      TF_RETURN_IF_ERROR(
          c->GetAttr("num_control_point_insertions", &num_control_points));

      DimensionHandle out_shape = c->MakeDim((1 << num_control_points) + 1);
      c->set_output(0, c->MakeShape({out_shape}));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Creates controlpoints for a random monotonic increasing tabulated function.

Iteratively creates controlpoints for a random monotonic increasing
function.  It starts with a uniform random value for black and a
uniform random value for white. Then iteratively inserts random
controlpoints between the existing ones, conforming to the slope_min
and slope_max constraints.

new_black_range: 2-element float list specifying the range for the
  new "black" value, i.e. the start value of the
  tabulated function. Default: [-0.1, 0.1]
new_white_range: 2-element float list specifying the range for the
  new "white" value, i.e. the end value of the
  tabulated function. Default: [0.9, 1.1]
slope_min: minimum slope for the resulting function
slope_max: maximum slope for the resulting function
num_control_point_insertions: number of splits. i.e.
    no split: 2 points
     1 split: 3 points
    2 splits: 5 points
    3 splits: 9 points,

output: 1-D Tensor containing the control points
)doc");

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
