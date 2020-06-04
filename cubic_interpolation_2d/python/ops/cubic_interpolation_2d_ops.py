from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

print(os.path.abspath(os.path.dirname(__file__)))
print(resource_loader.get_path_to_datafile('_cubic_interpolation2d_ops.so'))
cubic_interpolation2d_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_cubic_interpolation2d_ops.so'))

cubic_interpolation2d = cubic_interpolation2d_ops.cubic_interpolation_2d