# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Restore the environment captured by the matching activation hook.
if [[ "${_OPENNAV_SAM3_RUNTIME_ACTIVE:-0}" == "1" ]]; then
    _opennav_sam3_remove_path_entry() {
        local target="$1"
        local entry
        local result=""
        local separator=""
        local removed=0
        local -a entries=()

        IFS=: read -r -a entries <<< "${PATH-}"
        for entry in "${entries[@]}"; do
            if [[ "${removed}" == "0" && "${entry}" == "${target}" ]]; then
                removed=1
                continue
            fi
            result+="${separator}${entry}"
            separator=:
        done
        export PATH="${result}"
    }

    _opennav_sam3_remove_path_entry "${SAM3_MIGRAPHX_ROOT}/bin"
    _opennav_sam3_remove_path_entry "${ROCM_PATH}/bin"
    unset -f _opennav_sam3_remove_path_entry

    if [[ "${_OPENNAV_SAM3_SAVED_PYTHONPATH_SET}" == "x" ]]; then
        export PYTHONPATH="${_OPENNAV_SAM3_SAVED_PYTHONPATH}"
    else
        unset PYTHONPATH
    fi
    if [[ "${_OPENNAV_SAM3_SAVED_LD_LIBRARY_PATH_SET}" == "x" ]]; then
        export LD_LIBRARY_PATH="${_OPENNAV_SAM3_SAVED_LD_LIBRARY_PATH}"
    else
        unset LD_LIBRARY_PATH
    fi
    if [[ "${_OPENNAV_SAM3_SAVED_ROCM_PATH_SET}" == "x" ]]; then
        export ROCM_PATH="${_OPENNAV_SAM3_SAVED_ROCM_PATH}"
    else
        unset ROCM_PATH
    fi
    if [[ "${_OPENNAV_SAM3_SAVED_SAM3_ROCM_PATH_SET}" == "x" ]]; then
        export SAM3_ROCM_PATH="${_OPENNAV_SAM3_SAVED_SAM3_ROCM_PATH}"
    else
        unset SAM3_ROCM_PATH
    fi
    if [[ "${_OPENNAV_SAM3_SAVED_MIGRAPHX_ROOT_SET}" == "x" ]]; then
        export SAM3_MIGRAPHX_ROOT="${_OPENNAV_SAM3_SAVED_MIGRAPHX_ROOT}"
    else
        unset SAM3_MIGRAPHX_ROOT
    fi
    if [[ "${_OPENNAV_SAM3_SAVED_HSA_OVERRIDE_SET}" == "x" ]]; then
        export HSA_OVERRIDE_GFX_VERSION="${_OPENNAV_SAM3_SAVED_HSA_OVERRIDE}"
    else
        unset HSA_OVERRIDE_GFX_VERSION
    fi
    if [[ "${_OPENNAV_SAM3_SAVED_MIGRAPHX_FLAGS_SET}" == "x" ]]; then
        export MIGRAPHX_GPU_HIP_FLAGS="${_OPENNAV_SAM3_SAVED_MIGRAPHX_FLAGS}"
    else
        unset MIGRAPHX_GPU_HIP_FLAGS
    fi
    if [[ "${_OPENNAV_SAM3_SAVED_TRANSFORMERS_OFFLINE_SET}" == "x" ]]; then
        export TRANSFORMERS_OFFLINE="${_OPENNAV_SAM3_SAVED_TRANSFORMERS_OFFLINE}"
    else
        unset TRANSFORMERS_OFFLINE
    fi

    unset _OPENNAV_SAM3_SAVED_PYTHONPATH
    unset _OPENNAV_SAM3_SAVED_PYTHONPATH_SET
    unset _OPENNAV_SAM3_SAVED_LD_LIBRARY_PATH
    unset _OPENNAV_SAM3_SAVED_LD_LIBRARY_PATH_SET
    unset _OPENNAV_SAM3_SAVED_ROCM_PATH
    unset _OPENNAV_SAM3_SAVED_ROCM_PATH_SET
    unset _OPENNAV_SAM3_SAVED_SAM3_ROCM_PATH
    unset _OPENNAV_SAM3_SAVED_SAM3_ROCM_PATH_SET
    unset _OPENNAV_SAM3_SAVED_MIGRAPHX_ROOT
    unset _OPENNAV_SAM3_SAVED_MIGRAPHX_ROOT_SET
    unset _OPENNAV_SAM3_SAVED_HSA_OVERRIDE
    unset _OPENNAV_SAM3_SAVED_HSA_OVERRIDE_SET
    unset _OPENNAV_SAM3_SAVED_MIGRAPHX_FLAGS
    unset _OPENNAV_SAM3_SAVED_MIGRAPHX_FLAGS_SET
    unset _OPENNAV_SAM3_SAVED_TRANSFORMERS_OFFLINE
    unset _OPENNAV_SAM3_SAVED_TRANSFORMERS_OFFLINE_SET
    unset _OPENNAV_SAM3_RUNTIME_ACTIVE
fi
