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
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace {

uint64 RandomSeed() {
  std::random_device device("/dev/urandom");
  std::mt19937_64 seed_rng(device());
  return seed_rng();
}

class RandomLUTControlPointsOp : public tensorflow::OpKernel {
 public:
  explicit RandomLUTControlPointsOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("new_black_range", &new_black_range_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("new_white_range", &new_white_range_));
    OP_REQUIRES_OK(context, context->GetAttr("slope_min", &slope_min_));
    OP_REQUIRES_OK(context, context->GetAttr("slope_max", &slope_max_));
    OP_REQUIRES_OK(context, context->GetAttr("num_control_point_insertions",
                                             &num_control_point_insertions_));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    int num_elements =
        tensorflow::MathUtil::IPow(2, num_control_point_insertions_) + 1;
    tensorflow::Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {num_elements}, &output));
    // initial lut has only start and end point
    //
    std::vector<float> lut(2);
    lut[0] = uniform_random_value(new_black_range_[0], new_black_range_[1]);
    lut[1] = uniform_random_value(new_white_range_[0], new_white_range_[1]);
    float dx = 1;

    for (int iter = 0; iter < num_control_point_insertions_; ++iter) {
      // insert intermediate points
      std::vector<float> newlut;
      newlut.push_back(lut[0]);
      dx /= 2;
      for (int i = 0; i < lut.size() - 1; ++i) {
        float left_constraint_from = lut[i] + slope_min_ * dx;
        float left_constraint_to = lut[i] + slope_max_ * dx;
        float right_constraint_from = lut[i + 1] - slope_max_ * dx;
        float right_constraint_to = lut[i + 1] - slope_min_ * dx;
        float ymin = std::max(left_constraint_from, right_constraint_from);
        float ymax = std::min(left_constraint_to, right_constraint_to);
        newlut.push_back(uniform_random_value(ymin, ymax));
        newlut.push_back(lut[i + 1]);
      }
      lut = newlut;
    }
    std::copy(lut.begin(), lut.end(), output->flat<float>().data());
  }

 private:
  float uniform_random_value(float lo, float hi) {
    auto randoms = uniform_random_generator_(&gen_int_);
    return lo + (hi - lo) * randoms[0];
  }
  tensorflow::random::PhiloxRandom gen_int_{static_cast<uint64>(RandomSeed())};
  tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom,
                                          float>
      uniform_random_generator_;
  std::vector<float> new_black_range_;
  std::vector<float> new_white_range_;
  float slope_min_;
  float slope_max_;
  int num_control_point_insertions_;
};

REGISTER_KERNEL_BUILDER(
    Name("RandomLUTControlPoints").Device(tensorflow::DEVICE_CPU),
    RandomLUTControlPointsOp);

}  // namespace
}  // namespace multidim_image_augmentation
}  // namespace deepmind
