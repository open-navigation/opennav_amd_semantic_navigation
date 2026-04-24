# Copyright (C) 2026 Open Navigation LLC. All rights reserved.

ARG BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.8.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8

# Base tooling + locale + universe repo for ROS apt sources
RUN apt-get update && apt-get install -y \
    git curl wget gnupg2 lsb-release locales software-properties-common vim \
    && locale-gen en_US.UTF-8 \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Jazzy
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-base \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U \
    empy==3.3.4 lark catkin_pkg \
    'numpy<2' \
    huggingface_hub matplotlib Pillow

# transformers from git (SAM3 support not yet in a stable release).
# Installed non-editable so it lands in site-packages and is importable via
# PYTHONPATH from the system Python that ROS 2 uses to launch the node.
RUN git clone --depth 1 https://github.com/huggingface/transformers.git /transformers \
    && pip install --no-cache-dir '/transformers[torch]'

RUN git config --global credential.helper store

# Copy only the opennav_* ROS packages from the workspace src/ tree.
WORKDIR /ros_ws
COPY --parents opennav_*/ src/
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

# Create an entrypoint to automatically login to Hugging Face
# You need an account that has permissions to use the SAM3 models
RUN printf '#!/bin/bash\nset -e\n\
if [ -n "$HF_TOKEN" ]; then\n\
  hf auth login --token "$HF_TOKEN" --add-to-git-credential\n\
fi\n\
# Make PyTorch (installed in the ROCm base image'"'"'s conda Python) visible\n\
# to the system Python used by ROS 2 Jazzy node launchers.\n\
TORCH_SITE=$(python3 -c "import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))" 2>/dev/null || true)\n\
if [ -n "$TORCH_SITE" ]; then\n\
  export PYTHONPATH="$TORCH_SITE:${PYTHONPATH:-}"\n\
fi\n\
. /opt/ros/jazzy/setup.bash\n\
. /ros_ws/install/setup.bash\n\
exec "$@"\n' > /ros_ws/entrypoint.sh && chmod +x /ros_ws/entrypoint.sh

ENTRYPOINT ["/ros_ws/entrypoint.sh"]
CMD ["ros2", "run", "opennav_sam3_inference", "sam3_node"]
