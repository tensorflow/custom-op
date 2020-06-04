licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_library(
    name = "types",
    hdrs = ["cc/platform/types.h"],
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
    deps = [
        ":bspline",
        "@eigen",
    ],
)

cc_library(
    name = "ops_hdrs",
    hdrs = ["cc/ops/apply_deformation_ops.h"],
)

cc_library(
    name = "apply_deformation",
    hdrs = ["cc/kernels/apply_deformation.h"],
    deps = [
        ":types",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@eigen",
    ],
)


cc_library(
    name = "augmentation_ops_kernels",
    srcs = [
        "cc/kernels/apply_deformation_ops.cc",
        "cc/kernels/apply_tabulated_functions_op.cc",
        "cc/kernels/cubic_interpolation_ops.cc",
        "cc/kernels/random_lut_controlpoints_op.cc",
    ],
    hdrs = ["cc/ops/apply_deformation_ops.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":apply_deformation",
        ":cubic_interpolation",
        ":ops_hdrs",
        ":types",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@eigen",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "python/ops/_augmentation_ops.so",
    srcs = [
        "cc/ops/apply_deformation_ops.cc",
        "cc/ops/apply_tabulated_functions_op.cc",
        "cc/ops/cubic_interpolation1d_op.cc",
        "cc/ops/cubic_interpolation2d_op.cc",
        "cc/ops/cubic_interpolation3d_op.cc",
        "cc/ops/random_lut_controlpoints_op.cc",
    ],
    copts = select({
        ":windows": [
            "/DEIGEN_STRONG_INLINE=inline",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++11",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "-Iexternal/eigen",
        ],
    }),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":augmentation_ops_kernels",
        ":types",
    ],
)


py_library(
    name = "augmentation_ops_py",
    srcs = ([
        "python/ops/augmentation_ops.py",
    ]),
    data = [
        ":python/ops/_augmentation_ops.so",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "augmentation_ops_py_package",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    srcs_version = "PY2AND3",
    deps = [
        ":augmentation_ops_py",
    ],
)
