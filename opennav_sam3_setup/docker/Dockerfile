# syntax=docker/dockerfile:1
# Build context: opennav_sam3_setup/ (see docker/build.sh).

FROM ubuntu:24.04@sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG ROCM_VERSION=7.14
ARG ROCM_PACKAGE_VERSION=7.14.1-0
ARG TORCH_INDEX=https://repo.amd.com/rocm/whl/gfx1151/
ARG TORCH_VERSION=2.11.0+rocm7.13.0
ARG TORCHVISION_VERSION=0.26.0+rocm7.13.0
ARG TRITON_VERSION=3.6.0+rocm7.13.0
ARG PYPI_INDEX=https://pypi.org/simple

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg libgl1 libglib2.0-0 libnuma1 libssl3 \
        python3 python3-dev python3-pip python3-venv \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://repo.amd.com/rocm/packages/gpg/rocm.gpg \
        | gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg \
    && printf '%s\n' \
        'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main' \
        > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        "amdrocm-runtime-dev${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-blas-dev${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-blas-host${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-dnn-dev${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-dnn-host${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-hipblas-common-dev${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-blas${ROCM_VERSION}-gfx1151=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-dnn${ROCM_VERSION}-gfx1151=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-fft-host${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-fft${ROCM_VERSION}-gfx1151=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-rand-host${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-rand${ROCM_VERSION}-gfx1151=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-rccl-host${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-rccl${ROCM_VERSION}-gfx1151=${ROCM_PACKAGE_VERSION}" \
    && mkdir -p /opt/rocm \
    && for directory in bin lib libexec include share llvm amdgcn; do \
        ln -snf "core-${ROCM_VERSION}/${directory}" "/opt/rocm/${directory}"; \
    done \
    && rm -rf /var/lib/apt/lists/*

# These named contexts contain downloaded, checksum-verified binaries. No
# compiler source or compiler output directory is part of this Docker build.
COPY --from=migraphx /migraphx/ /opt/migraphx-develop/
COPY --from=ort /onnxruntime_migraphx-1.24.2-cp312-cp312-linux_x86_64.whl \
    /tmp/onnxruntime_migraphx-1.24.2-cp312-cp312-linux_x86_64.whl
COPY requirements.txt /tmp/sam3-requirements.txt
COPY runtime-constraints.txt /tmp/sam3-runtime-constraints.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv --system-site-packages /opt/sam3-runtime \
    && /opt/sam3-runtime/bin/python -m pip install --only-binary=:all: \
        --index-url "${PYPI_INDEX}" --constraint /tmp/sam3-runtime-constraints.txt \
        /tmp/onnxruntime_migraphx-1.24.2-cp312-cp312-linux_x86_64.whl \
    && /opt/sam3-runtime/bin/python -m pip install --only-binary=:all: --no-deps \
        --index-url "${TORCH_INDEX}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "triton==${TRITON_VERSION}" \
    && /opt/sam3-runtime/bin/python -m pip install --only-binary=:all: \
        --index-url "${PYPI_INDEX}" --constraint /tmp/sam3-runtime-constraints.txt \
        transformers==5.8.1 safetensors==0.7.0 opencv-python==4.11.0.86 \
        networkx jinja2 filelock fsspec sympy -r /tmp/sam3-requirements.txt \
    && rm /tmp/onnxruntime_migraphx-1.24.2-cp312-cp312-linux_x86_64.whl \
        /tmp/sam3-requirements.txt /tmp/sam3-runtime-constraints.txt

RUN mkdir -p /opt/sam3-runtime/lib/python3.12/site-packages/rocm_sdk
COPY rocm_sdk_system.py /opt/sam3-runtime/lib/python3.12/site-packages/rocm_sdk/__init__.py

# ROS 2 Jazzy (Ubuntu 24.04 / Python 3.12 matches the venv above). Only what
# the SAM3 node, its launch files and colcon need; Nav2 stays outside.
RUN apt-get update \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base ros-dev-tools ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-cv-bridge ros-jazzy-vision-msgs \
        ros-jazzy-image-proc ros-jazzy-depth-image-proc \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /opt/opennav-entrypoint.sh

ENV PATH=/opt/sam3-runtime/bin:/opt/migraphx-develop/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/migraphx-develop/lib
ENV LD_LIBRARY_PATH=/opt/migraphx-develop/lib:/opt/migraphx-develop/lib/migraphx/lib:/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-7.14/lib:/opt/rocm/core-7.14/lib/host-math/lib:/opt/rocm/core-7.14/lib/rocm_sysdeps/lib:/usr/local/lib
ENV ROCM_PATH=/opt/rocm
ENV SAM3_ROCM_PATH=/opt/rocm
ENV SAM3_MIGRAPHX_ROOT=/opt/migraphx-develop
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV TRANSFORMERS_OFFLINE=1
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# expandable_segments hangs the first GPU op in this container; sam3_node.py only
# setdefault()s it to True, so this value takes precedence.
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
ENV MIGRAPHX_GPU_HIP_FLAGS="-Wno-error -Wno-lifetime-safety-intra-tu-suggestions"

LABEL org.opencontainers.image.title="OpenNav SAM3 binary-assembled ROCm 7.14 runtime" \
      org.opencontainers.image.description="Precompiled MIGraphX and ORT; no runtime source compilation"

ENTRYPOINT ["/opt/opennav-entrypoint.sh"]
CMD ["bash"]
