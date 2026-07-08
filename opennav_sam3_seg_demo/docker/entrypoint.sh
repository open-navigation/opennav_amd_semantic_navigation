#!/bin/bash
set -e

source /opt/ros/${ROS_DISTRO}/setup.bash
source ~/miniforge3/etc/profile.d/conda.sh
source ~/orbbec-driver/setup.bash
source ~/opennav_sam3_msgs/setup.bash
source ~/opennav_sam3_seg_demo/setup.bash
source ~/opennav_sam3_inference/setup.bash

export PYTHONPATH=/opt/rocm-7.2.0/lib:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib/migraphx/lib:/opt/rocm-7.2.0/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:512
export MIGRAPHX_GPU_HIP_FLAGS="-Wno-error -Wno-lifetime-safety-intra-tu-suggestions"

if [ -f ~/configs/cyclonedds.xml ]; then
    export CYCLONEDDS_URI=~/configs/cyclonedds.xml
fi

SETUP_DIR=~/opennav_sam3_setup

conda activate opennav-sam3-inference

if ! [ -d ${SAM3_MODEL_DIR}/sam3 ]; then
    (cd ${SETUP_DIR} && ./download-weights.sh --yes)
else
    printf "Model weights folder found!\n"
fi

if ! [ -d ${SAM3_MODEL_DIR}/onnx_files* ]; then
    printf "\n\nBuilding runtime artifacts...\n\n"
    (cd ${SETUP_DIR} && python export/build.py --pipeline text --imgsz 504)
else
    printf "Build artifacts folder found!\n\n"
fi

conda deactivate

ros2 launch opennav_sam3_seg_demo demo.launch.py