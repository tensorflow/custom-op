#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

# Check if we are building GPU or CPU ops, default CPU
while [[ "$INSTALL_CPU" == "" ]]; do
  read -p "Do you want to build ops again TensorFlow CPU pip package?"\
" Y or enter for CPU (tensorflow), N for GPU (tensorflow-gpu). [Y/n] " INPUT
  case $INPUT in
    [Yy]* ) echo "Build with CPU pip package."; INSTALL_CPU=1;;
    [Nn]* ) echo "Build with GPU pip package."; INSTALL_CPU=0;;
    "" ) echo "Build with CPU pip package."; INSTALL_CPU=1;;
    * ) echo "Invalid selection: " $INPUT;;
  esac
done



# CPU
if [[ "$INSTALL_CPU" == "1" ]]; then

  # Check if it's installed
  if [[ $(pip show tensorflow) == *tensorflow* ]]; then
    echo 'Using installed tensorflow'
  else
    # Uninstall GPU version if it is installed.
    if [[ $(pip show tensorflow-gpu) == *tensorflow-gpu* ]]; then
      echo 'Already have tensorflow-gpu installed. Uninstalling......\n'
      pip uninstall tensorflow-gpu
    fi
    # Install CPU version
    echo 'Installing tensorflow......\n'
    pip install tensorflow
  fi

else

  # Check if it's installed
  if [[ $(pip show tensorflow-gpu) == *tensorflow-gpu* ]]; then
    echo 'Using installed tensorflow-gpu'
  else
    # Uninstall CPU version if it is installed.
    if [[ $(pip show tensorflow) == *tensorflow* ]]; then
      echo 'Already have tensorflow non-gpu installed. Uninstalling......\n'
      pip uninstall tensorflow
    fi
    # Install CPU version
    echo 'Installing tensorflow-gpu .....\n'
    pip install tensorflow-gpu
  fi
fi


TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_LFLAGS:2}
