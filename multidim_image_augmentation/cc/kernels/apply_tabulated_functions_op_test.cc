// Copyright 2019 Google LLC
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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace deepmind {
namespace multidim_image_augmentation {
namespace tabulated_functions_op_internal {
// Allows to globally disable AVX optimizations. Used only for benchmarking.
extern bool use_avx_optimizations;
}  // namespace tabulated_functions_op_internal
}  // namespace multidim_image_augmentation
}  // namespace deepmind

namespace tensorflow {
static Tensor MakeRandomTensor(const TensorShape& shape, int range) {
  Tensor tensor(DT_FLOAT, TensorShape(shape));
  tensor.flat<float>() = tensor.flat<float>().setRandom();
  // Domain change from [-1..1] -> [0..range]
  tensor.flat<float>() = (tensor.flat<float>() + 1.0f) * (float)(range / 2);
  return tensor;
}

static Tensor MakeZeroTensor(const TensorShape& shape) {
  Tensor tensor(DT_FLOAT, TensorShape(shape));
  tensor.flat<float>() = tensor.flat<float>().setZero();
  return tensor;
}

static Graph* ApplyTabulatedFunctions(const char* op, const char* type,
                                      int channels, int pixels,
                                      int table_size) {
  Graph* graph = new Graph(OpRegistry::Global());

  Tensor input_t;
  if (std::string(type) == "random") {
    // Random inputs stress the cache and should make the benchmark be dominated
    // by memory accesses.
    input_t = MakeRandomTensor({pixels, channels}, table_size);
  } else if (std::string(type) == "zeros") {
    // Zero inputs always access the same table element and give a best-case
    // performance by eliminating cache overheads.
    input_t = MakeZeroTensor({pixels, channels});
  } else {
    LOG(FATAL) << "Unknown input type";
  }
  Tensor table_t = MakeRandomTensor({channels, table_size}, /*range=*/1024);

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* table = test::graph::Constant(graph, table_t, "table");

  Node* n;
  TF_CHECK_OK(NodeBuilder(graph->NewName("apply_tabulated_functions"), op)
                  .Input(input)
                  .Input(table)
                  .Finalize(graph, &n));
  return graph;
}

// This formulation of the benchmark function is a little verbose and ugly but
// the style is cribbed from the microbenchmarks in
// //tensorflow/framework/kernels.
#define BM_NAME(name, type, C, P, T, O) name##_##type##_##C##x##P##x##T##__##O

// Instantiate the benchmark with:
//    type: "random" or "zeros"; determine the input tensor contents. Random
//    gives an upper-case estimate (with cache misses), zeros gives a lower-case
//    estimate (no cache misses).
//    C: Number of channels
//    P: Number of pixels
//    T: Lookup table size
//    O: Use optimized implementation true/false.
#define BM_ApplyTabulatedFunctionsHelper(type, C, P, T, LABEL, O)             \
  static void BM_NAME(BM_ApplyTabulatedFunctions, type, C, P, T,              \
                      O)(int iters) {                                         \
    testing::SetLabel(LABEL);                                                 \
    deepmind::multidim_image_augmentation::tabulated_functions_op_internal::  \
        use_avx_optimizations = O;                                            \
    test::Benchmark("cpu", ApplyTabulatedFunctions("ApplyTabulatedFunctions", \
                                                   #type, C, P, T))           \
        .Run(iters);                                                          \
  }                                                                           \
  BENCHMARK(BM_NAME(BM_ApplyTabulatedFunctions, type, C, P, T, O));

#define BM_ApplyTabulatedFunctions(type, C, P, T, LABEL)                  \
  BM_ApplyTabulatedFunctionsHelper(type, C, P, T, "Naive " LABEL, false); \
  BM_ApplyTabulatedFunctionsHelper(type, C, P, T, "Optimized " LABEL, true);

BM_ApplyTabulatedFunctions(random, 1, 2048, 1024, "2048 pixel, 1 channel");
BM_ApplyTabulatedFunctions(zeros, 1, 2048, 1024,
                           "2048 pixel, 1 channel (zero)");
BM_ApplyTabulatedFunctions(random, 1, 2097152, 1024,
                           "2048*1024 pixel, 1 channel");
BM_ApplyTabulatedFunctions(zeros, 1, 2097152, 1024,
                           "2048*1024 pixel, 1 channel (zero)");

}  // namespace tensorflow
