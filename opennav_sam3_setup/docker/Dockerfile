# syntax=docker/dockerfile:1
# Build context: opennav_sam3_setup/ (see docker/build.sh).

# Official ROS 2 Jazzy image (Ubuntu 24.04 / Python 3.12): ROS, colcon, the
# ament linters and the ROS apt source are already installed.
FROM ros:jazzy-ros-base@sha256:103b6c417973394659af6804d4917cd489b91e5ed22eb7bef263025a3bed2410
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ARG ROCM_VERSION=10.1
ARG ROCM_PACKAGE_VERSION=10.1.0-3
ARG PYPI_INDEX=https://pypi.org/simple

# System ROCm 10.1 from AMD's official apt repository, plus the ROS packages
# the SAM3 node and its launch files need beyond ros-base. Nav2 stays outside.
RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg \
        | gpg --dearmor -o /etc/apt/keyrings/amdrocm.gpg \
    && printf '%s\n' \
        'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://stable.repo.amd.com/rocm/core/packages/ubuntu2404 stable main' \
        > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libnuma1 python3-venv \
        "ros-${ROS_DISTRO}-rmw-cyclonedds-cpp" \
        "ros-${ROS_DISTRO}-cv-bridge" "ros-${ROS_DISTRO}-vision-msgs" \
        "ros-${ROS_DISTRO}-image-proc" "ros-${ROS_DISTRO}-depth-image-proc" \
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
        "amdrocm-hipfile${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
        "amdrocm-hipfile-dev${ROCM_VERSION}=${ROCM_PACKAGE_VERSION}" \
    && mkdir -p /opt/rocm \
    && for directory in bin lib libexec include share llvm amdgcn; do \
        ln -snf "core-${ROCM_VERSION}/${directory}" "/opt/rocm/${directory}"; \
    done \
    && rm -rf /var/lib/apt/lists/*

# The `wheels` named context holds the checksum-verified official AMD wheels
# (Torch 2.14 + gfx1151/gfx115x device images, MIGraphX 2.18, the MIGraphX
# plugin EP) and ONNX Runtime 1.29.0 from PyPI. build.sh verifies every SHA256
# before the build starts. No compiler source or compiler output is involved.
COPY --from=wheels / /tmp/wheels/
COPY requirements.txt /tmp/sam3-requirements.txt
COPY runtime-constraints.txt /tmp/sam3-runtime-constraints.txt

# The Torch wheels declare ROCm wheel dependencies, while this image supplies
# the same-version ROCm libraries via the pinned APT packages. They are
# therefore installed with --no-deps, and the rocm_sdk shim below stops a second
# wheel-supplied ROCm stack from overriding them. amd-torch-device-gfx115x is
# required in addition to the gfx1151 wheel; without it the default SDPA fails
# with hipErrorInvalidImage. The venv sees the ROS Python packages, colcon and
# the system NumPy through --system-site-packages.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv --system-site-packages /opt/sam3-runtime \
    && /opt/sam3-runtime/bin/python -m pip install --only-binary=:all: --no-deps /tmp/wheels/*.whl \
    && /opt/sam3-runtime/bin/python -m pip install --only-binary=:all: \
        --index-url "${PYPI_INDEX}" --constraint /tmp/sam3-runtime-constraints.txt \
        transformers==5.8.1 safetensors==0.7.0 opencv-python==4.11.0.86 \
        networkx jinja2 filelock fsspec sympy -r /tmp/sam3-requirements.txt \
    && rm -rf /tmp/wheels /tmp/sam3-requirements.txt /tmp/sam3-runtime-constraints.txt

RUN mkdir -p /opt/sam3-runtime/lib/python3.12/site-packages/rocm_sdk
COPY rocm_sdk_system.py /opt/sam3-runtime/lib/python3.12/site-packages/rocm_sdk/__init__.py

COPY docker/entrypoint.sh /opt/opennav-entrypoint.sh

ENV PATH=/opt/sam3-runtime/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/core-10.1/lib:/opt/rocm/core-10.1/lib/host-math/lib:/opt/rocm/core-10.1/lib/rocm_sysdeps/lib:/usr/local/lib
ENV ROCM_PATH=/opt/rocm
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV TRANSFORMERS_OFFLINE=1
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# expandable_segments hangs the first GPU model upload in this container;
# sam3_node.py only setdefault()s it to True, so this value takes precedence.
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
ENV MIGRAPHX_GPU_HIP_FLAGS="-Wno-error -Wno-lifetime-safety-intra-tu-suggestions"

LABEL org.opencontainers.image.title="OpenNav SAM3 ROCm 10.1 runtime (official packages)" \
      org.opencontainers.image.description="ROS 2 Jazzy + ROCm 10.1 + official MIGraphX 2.18 wheel + ONNX Runtime 1.29 plugin EP + Torch 2.14; no runtime source compilation"

ENTRYPOINT ["/opt/opennav-entrypoint.sh"]
CMD ["bash"]
