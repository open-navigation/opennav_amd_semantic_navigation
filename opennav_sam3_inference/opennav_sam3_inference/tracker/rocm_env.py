# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""
ROCm environment defaults for gfx1151 (Strix Halo / Ryzen AI Max+ 395).

Call apply() at the top of any module that touches MIGraphX or the ROCm GPU
stack. Uses os.environ.setdefault, so explicit user exports take precedence.

Script-specific flags such as MIGRAPHX_MLIR_USE_SPECIFIC_OPS and
MIGRAPHX_SKIP_BENCHMARKING intentionally remain with their callers.
"""

import os


HSA_OVERRIDE_GFX_VERSION = '11.5.1'
MIGRAPHX_GPU_HIP_FLAGS = '-Wno-error -Wno-lifetime-safety-intra-tu-suggestions'
TRANSFORMERS_OFFLINE = '1'

_DEFAULTS = {
    'HSA_OVERRIDE_GFX_VERSION': HSA_OVERRIDE_GFX_VERSION,
    'MIGRAPHX_GPU_HIP_FLAGS': MIGRAPHX_GPU_HIP_FLAGS,
    'TRANSFORMERS_OFFLINE': TRANSFORMERS_OFFLINE,
}


def apply() -> None:
    """Apply ROCm defaults without overriding explicitly configured values."""
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, value)
