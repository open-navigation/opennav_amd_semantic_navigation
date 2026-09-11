# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is installed into the dedicated Conda environment by setup.sh.
# It is sourced by Conda, not executed as a standalone program.

if [[ "${_OPENNAV_SAM3_RUNTIME_ACTIVE:-0}" != "1" ]]; then
    export _OPENNAV_SAM3_SAVED_PYTHONPATH="${PYTHONPATH-}"
    export _OPENNAV_SAM3_SAVED_PYTHONPATH_SET="${PYTHONPATH+x}"
    export _OPENNAV_SAM3_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
    export _OPENNAV_SAM3_SAVED_LD_LIBRARY_PATH_SET="${LD_LIBRARY_PATH+x}"
    export _OPENNAV_SAM3_SAVED_ROCM_PATH="${ROCM_PATH-}"
    export _OPENNAV_SAM3_SAVED_ROCM_PATH_SET="${ROCM_PATH+x}"
    export _OPENNAV_SAM3_SAVED_SAM3_ROCM_PATH="${SAM3_ROCM_PATH-}"
    export _OPENNAV_SAM3_SAVED_SAM3_ROCM_PATH_SET="${SAM3_ROCM_PATH+x}"
    export _OPENNAV_SAM3_SAVED_MIGRAPHX_ROOT="${SAM3_MIGRAPHX_ROOT-}"
    export _OPENNAV_SAM3_SAVED_MIGRAPHX_ROOT_SET="${SAM3_MIGRAPHX_ROOT+x}"
    export _OPENNAV_SAM3_SAVED_HSA_OVERRIDE="${HSA_OVERRIDE_GFX_VERSION-}"
    export _OPENNAV_SAM3_SAVED_HSA_OVERRIDE_SET="${HSA_OVERRIDE_GFX_VERSION+x}"
    export _OPENNAV_SAM3_SAVED_MIGRAPHX_FLAGS="${MIGRAPHX_GPU_HIP_FLAGS-}"
    export _OPENNAV_SAM3_SAVED_MIGRAPHX_FLAGS_SET="${MIGRAPHX_GPU_HIP_FLAGS+x}"
    export _OPENNAV_SAM3_SAVED_TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE-}"
    export _OPENNAV_SAM3_SAVED_TRANSFORMERS_OFFLINE_SET="${TRANSFORMERS_OFFLINE+x}"

    _OPENNAV_SAM3_CONFIG="${CONDA_PREFIX}/etc/opennav_sam3_runtime.conf"
    if [[ -r "${_OPENNAV_SAM3_CONFIG}" ]]; then
        source "${_OPENNAV_SAM3_CONFIG}"
    fi
    export SAM3_MIGRAPHX_ROOT="${SAM3_MIGRAPHX_ROOT:-/opt/opennav-sam3/runtime/0.2.0-rc4/migraphx}"
    export SAM3_ROCM_PATH="${SAM3_ROCM_PATH:-/opt/rocm}"
    export ROCM_PATH="${SAM3_ROCM_PATH}"
    export PATH="${SAM3_MIGRAPHX_ROOT}/bin:${ROCM_PATH}/bin${PATH:+:${PATH}}"
    export PYTHONPATH="${SAM3_MIGRAPHX_ROOT}/lib${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${SAM3_MIGRAPHX_ROOT}/lib:${SAM3_MIGRAPHX_ROOT}/lib/migraphx/lib:${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${ROCM_PATH}/core-7.14/lib:${ROCM_PATH}/core-7.14/lib/host-math/lib:${ROCM_PATH}/core-7.14/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"
    export MIGRAPHX_GPU_HIP_FLAGS="${MIGRAPHX_GPU_HIP_FLAGS:--Wno-error -Wno-lifetime-safety-intra-tu-suggestions}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export _OPENNAV_SAM3_RUNTIME_ACTIVE=1
    unset _OPENNAV_SAM3_CONFIG
fi
