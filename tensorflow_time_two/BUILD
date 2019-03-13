licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")



cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "@local_config_cuda//cuda:darwin": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
        ],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "time_two_ops_gpu",
    srcs = ["cc/kernels/time_two.h", "cc/kernels/time_two_kernels.cu.cc"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":cuda",  "@local_config_cuda//cuda:cuda_headers"]),
    alwayslink = 1,
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
)

cc_binary(
    name = 'python/ops/_time_two_ops.so',
    srcs = [
        "cc/kernels/time_two.h",
        "cc/kernels/time_two_kernels.cc",
        "cc/ops/time_two_ops.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":time_two_ops_gpu"]),
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"] + if_cuda(["-DGOOGLE_CUDA=1"])
)

py_library(
    name = "time_two_ops_py",
    srcs = ([
        "python/ops/time_two_ops.py",
    ]),
    data = [
        ":python/ops/_time_two_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "time_two_ops_py_test",
    srcs = [
        "python/ops/time_two_ops_test.py"
    ],
    main = "python/ops/time_two_ops_test.py",
    deps = [
        ":time_two_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "time_two_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":time_two_ops_py"
    ],
    srcs_version = "PY2AND3",
)
