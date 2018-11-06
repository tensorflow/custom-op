# TensorFlow Custom Op
This is a guide for users who want to write custom c++ op for TensorFlow and distribute the op as a pip package. This repository serves as both a working example of the op building and packaging process, as well as a template/starting point for writing your own ops.

## Build Example zero_out Op
If you want to try out the process of building a pip package for custom op, you can use the source code from this repository following the instructions below.

### Setup Docker Container
You are going to build the op inside a Docker container. Pull the provided Docker image from TensorFlow's Docker hub and start a container.

```bash
  docker pull tensorflow/tensorflow:custom-op
  docker run -it tensorflow/tensorflow:custom-op /bin/bash
```

Inside the Docker container, clone this repository. The code in this repository came from the [Adding an op](https://www.tensorflow.org/extend/adding_an_op) guide.
```bash
git clone https://github.com/tensorflow/custom-op.git
cd zero_out
```

### Build PIP Package
You can build the pip package with either Bazel or make.

With bazel:
```bash
  ./configure.sh
  bazel build build_pip_pkg
  bazel-bin/build_pip_pkg artifacts
```

With Makefile:
```bash
  make pip_pkg
```

### Install and Test PIP Package
Once the pip package has been built, you can install it with,
```bash
pip install artifacts/*.whl
```
Then test out the pip package
```bash
cd ..
python -c "import tensorflow as tf;import tensorflow_zero_out as zero_out_module;print(zero_out_module.zero_out([[1,2], [3,4]]).eval(session=tf.Session()))"
```
And you should see the op zeroed out all input elements except the first one:
```bash
[[1 0]
 [0 0]]
```

## Create and Distribute Custom Op
Now you are ready to write and distribute your own ops. The example in this repository has done the boiling plate work for setting up build systems and package files needed for creating a pip package. We recommend using this repository as a template. 


### Template Overview
First let's go through a quick overview of the folder structure of this template repository.
```
├── tensorflow_zero_out
│   ├── cc
│   │   ├── kernels  # op kernel implementation
│   │   │   └── zero_out_kernels.cc
│   │   └── ops  # op interface definition
│   │       └── zero_out_ops.cc
│   ├── python
│   │   ├── ops
│   │   │   ├── __init__.py
│   │   │   ├── zero_out_ops.py   # Load and extend the ops in python
│   │   │   └── zero_out_ops_test.py  # tests for ops
│   │   └── __init__.py
|   |
│   ├── BUILD  # BUILD file for all op targets
│   └── __init__.py  # top level __init__ file that imports the custom op
│
├── tf  # Set up TensorFlow pip package as external dependency for Bazel
│   ├── BUILD
│   ├── BUILD.tpl
│   └── tf_configure.bzl
|
├── BUILD  # top level Bazel BUILD file that contains pip package build target
├── build_pip_pkg.sh  # script to build pip package for Bazel and Makefile
├── configure.sh  # script to install TensorFlow and setup action_env for Bazel
├── LICENSE
├── Makefile  # Makefile for building shared library and pip package
├── setup.py  # file for creating pip package
├── MANIFEST.in  # files for creating pip package
├── README.md
└── WORKSPACE  # Used by Bazel to specify tensorflow pip package as an external dependency

```
The op implementation, including both c++ and python code, goes under `tensorflow_zero_out` dir. You will want to replace this directory with the corresponding content of your own ops. `tf` folder contains the code for setting up TensorFlow pip package as an external dependency for Bazel only. You shouldn't need to change the content of this folder. You also don't need this folder if you are using other build systems, such as Makefile. To build a pip package for your op, you will also need to update a few files at the top level of the template, for example, `setup.py`, `MANIFEST.in` and `build_pip_pkg.sh`.

### Setup
First, clone this template repo.
```bash
git clone -b test https://github.com/tensorflow/custom-op.git my_op
cd my_op
```

#### Docker
Next, set up a Docker container using the provided Docker image for building and testing the ops. The provided Docker image `tensorflow/tensorflow:custom-op` is based on Ubuntu 14.04, and it contains the same versions of tools and libraries used for building the official TensorFlow pip packages. It also comes with Bazel pre-installed. We have seen many cases where dependency version differences and ABI incompatibilities cause the custom op extension users build to not work properly with TensorFlow's released pip packages. Therefore, it is *highly recommended* to use the provided Docker image to build your custom op. To get the Docker image, run
```bash
docker pull tensorflow/tensorflow:custom-op
```

You might want to use Docker volumes to map a `work_dir` from host to the container, so that you can edit files on the host, and build with the latest changes in the Docker container. To do so, run
```bash
docker run -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op
```

#### Run configure.sh
Last step before starting implementing the ops, you want to set up the build environment. The custom ops will need to depend on TensorFlow headers and shared library libtensorflow_framework.so, which are distributed with TensorFlow official pip package. If you would like to use Bazel to build your ops, you might also want to set a few action_envs so that Bazel can find the installed TensorFlow. We provide a `configure` script that does these for you. Simply run `./configure.sh` in the docker container and you are good to go.


### Add Op Implementation
Now you are ready to implement your op. Following the instructions at [Adding a New Op](https://www.tensorflow.org/extend/adding_an_op), add definition of your op interface under `<your_op>/cc/ops/` and kernel implementation under `<your_op>/cc/kernels/`.


### Build and Test Op

#### Bazel
To build the custom op shared library with Bazel, follow the cc_binary example in [`tensorflow_zero_out/BUILD`](https://github.com/tensorflow/custom-op/blob/master/tensorflow_zero_out/BUILD#L5). You will need to depend on the header files and libtensorflow_framework.so from TensorFlow pip package to build your op. Earlier we mentioned that the template has already setup TensorFlow pip package as an external dependency in `tf` directory, and the pip package is listed as `local_config_tf` in [`WORKSPACE`](https://github.com/tensorflow/custom-op/blob/master/WORKSPACE) file. Your op can depend directly on TensorFlow header files and 'libtensorflow_framework.so' with the following:
```python
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
```

You will need to keep both above dependencies for your op. To build the shared library with Bazel, run the following command in your Docker container
```bash
bazel build tensorflow_zero_out:python/ops/_zero_out_ops.so
```

#### Makefile
To build the custom op shared library with make, follow the example in [`Makefile`](https://github.com/tensorflow/custom-op/blob/master/Makefile) for `_zero_out_ops.so` and run the following command in your Docker container:
```bash
make op
```

#### Extend and Test the Op in Python
Once you have built your custom op shared library, you can follow the example in [`tensorflow_zero_out/python/ops`](https://github.com/tensorflow/custom-op/tree/master/tensorflow_zero_out/python/ops), and instructions [here](https://www.tensorflow.org/extend/adding_an_op#use_the_op_in_python) to create a module in Python for your op. Both guides use TensorFlow API `tf.load_op_library`, which loads the shared library and registers the ops with the TensorFlow framework.
```python
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_zero_out_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_zero_out_ops.so'))
zero_out = _zero_out_ops.zero_out

```

You can also add Python tests like what we have done in `tensorflow_zero_out/python/ops/zero_out_ops_test.py` to check that your op is working as intended.


##### Run Tests with Bazel
To add the python library and tests targets to Bazel, please follow the examples for `py_library` target `tensorflow_zero_out:zero_out_ops_py` and `py_test` target `tensorflow_zero_out:zero_out_ops_py_test` in `tensorflow_zero_out/BUILD` file. To run your test with bazel, do the following in Docker container,

```bash
bazel test tensorflow_zero_out:zero_out_ops_py_test
```
Or run all tests with

```bash
bazel test tensorflow_zero_out:all
```

##### Run Tests with Make
To add the test target to make, please follow the example in `Makefile`. To run your python test, simply run the following in Docker container,
```bash
make test
```


### Build PIP Package
Now your op works, you might want to build a pip package for it so the community can also benefit from your work. This template provides the basic setup needed to build your pip package. First, you will need to update the following top level files based on your op.

- `setup.py` contains information about your package (such as the name and version) as well as which code files to include. 
- `MANIFEST.in` contains the list of additional files you want to include in the source distribution. Here you want to make sure the shared library for your custom op is included in the pip package.
- `build_pip_pkg.sh` creates the package hierarchy, and calls `bdist_wheel` to assemble your pip package.

You can use either Bazel or Makefile to build the pip package.


#### Build with Bazel
You can find the target for pip package in the top level `BUILD` file. Inside the data list of this `build_pip_pkg` target, you want to include the python library target ` //tensorflow_zero_out:zero_out_py` in addition to the top level files. To build the pip package builder, run the following command in Docker container,
```bash
bazel build :build_pip_pkg
```

The bazel build command creates a binary named build_pip_package, which you can use to build the pip package. For example, the following builds your .whl package in the `artifacts` directory:
```bash
bazel-bin/build_pip_pkg artifacts
```

#### Build with make
Building with make also invoke the same `build_pip_pkg.sh` script. You can run,
```bash
make pip_pkg
```

### Test PIP Package
Before publishing your pip package, test your pip package.
```bash
pip install artifacts/*.whl
python -c "import tensorflow as tf;import tensorflow_zero_out as zero_out_module;print(zero_out_module.zero_out([[1,2], [3,4]]).eval(session=tf.Session()))"
```


### Publish PIP Package
Once your pip package has been thoroughly tested, you can distribute your package by uploading your package to the Python Package Index. Please follow the [official instruction](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives) from Pypi.


