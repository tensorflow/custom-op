licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_library(
    name = "types",
    hdrs = ["cc/types.h"],
)


cc_library(
    name = "bspline",
    hdrs = ["cc/kernels/bspline.h"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "cubic_interpolation",
    hdrs = ["cc/kernels/cubic_interpolation.h"],
    copts = ["-Iexternal/eigen"],
    deps = [
        ":types",
        ":bspline",
        "@eigen",
    ],
)


cc_binary(
    name = 'python/ops/_cubic_interpolation2d_ops.so',
    srcs = [
        "cc/ops/cubic_interpolation_2d_op.cc",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        ":types",
        ":bspline",
        "@eigen",
    ],
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    copts = select({
        ":windows": ["/DEIGEN_STRONG_INLINE=inline", "-DTENSORFLOW_MONOLITHIC_BUILD", "/DPLATFORM_WINDOWS", "/DEIGEN_HAS_C99_MATH", "/DTENSORFLOW_USE_EIGEN_THREADPOOL", "/DEIGEN_AVOID_STL_ARRAY", "/Iexternal/gemmlowp", "/wd4018", "/wd4577", "/DNOGDI", "/UTF_COMPILE_LIBRARY"],
        "//conditions:default": ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0", "-Iexternal/eigen"],
    }),
)


py_library(
    name = "cubic_interpolation_2d_ops_py",
    srcs = ([
        "python/ops/cubic_interpolation_2d_ops.py",
    ]),
    data = [
        ":python/ops/_cubic_interpolation2d_ops.so"
    ],
    srcs_version = "PY2AND3",
)


py_library(
    name = "cubic_interpolation_2d_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":cubic_interpolation_2d_ops_py"
    ],
    srcs_version = "PY2AND3",
)
