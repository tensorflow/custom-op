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

// This header brings standard integral types into scope.

#ifndef CC_TYPES_H_
#define CC_TYPES_H_

namespace deepmind {
namespace multidim_image_augmentation {

// Adapted from tensorflow/core/platform/default/integral_types.h
// TODO: replace this with inclusion of integral_types.h from absl,
// when it arrives.

typedef signed char int8;  // NOLINT(runtime/int)
typedef short int16;  // NOLINT(runtime/int)
typedef int int32;  // NOLINT(runtime/int)
typedef long long int64;  // NOLINT(runtime/int)

typedef unsigned char uint8;  // NOLINT(runtime/int)
typedef unsigned short uint16;  // NOLINT(runtime/int)
typedef unsigned int uint32;  // NOLINT(runtime/int)
typedef unsigned long long uint64;  // NOLINT(runtime/int)

}  // namespace multidim_image_augmentation
}  // namespace deepmind


#endif  // CC_TYPES_H_