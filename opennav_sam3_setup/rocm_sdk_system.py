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

"""Resolve the gfx1151 PyTorch wheel against the system ROCm installation.

AMD's gfx1151 PyTorch wheel is built against ROCm 7.13 and normally imports
ROCm libraries from Python wheels. OpenNav uses the ABI-compatible system
ROCm 7.14 libraries instead, selected through ``ROCM_PATH``. The 7.14.1-0
Debian package set exposes the 7.14.0 API version below; both values match the
validated rc4 container.
"""

import glob
import os
from pathlib import Path


__version__ = "7.14.0"


def initialize_process(**_kwargs):
    """Let the dynamic linker resolve ROCm libraries from LD_LIBRARY_PATH."""


def find_libraries(*shortnames: str) -> list[Path]:
    """Resolve requested ROCm shared libraries below ``ROCM_PATH``."""
    rocm_path = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    paths: list[Path] = []
    for shortname in shortnames:
        matches = sorted(glob.glob(str(rocm_path / "lib" / f"lib{shortname}.so*")))
        if not matches:
            matches = sorted(
                glob.glob(
                    str(rocm_path / "**" / f"lib{shortname}.so*"),
                    recursive=True,
                )
            )
        if not matches:
            raise FileNotFoundError(f"ROCm library not found: {shortname}")
        paths.append(Path(matches[0]))
    return paths
