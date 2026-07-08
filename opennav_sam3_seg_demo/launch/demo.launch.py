# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

configs_path = Path("/home/sam3-demo-usr/configs")  # Inside the Docker container
camera_topic = '/camera/color/image_raw' 
segmentation_topic = '/sam3_inference/segmentation_mask' 

def generate_launch_description():
    sam3_node_pkg = FindPackageShare('opennav_sam3_inference')
    orbbec_node_pkg = FindPackageShare('orbbec_camera')

    sam3_node_launch_args = [
        ('image_topic', camera_topic), 
        ('sensor_processing_pipeline', 'false'),
        ('log_level', 'info')
    ]

    orbbec_node_launch_args = []

    if os.path.isdir(configs_path):
        sam3_node_params_file = configs_path / "sam3-inference.yaml"
        
        if os.path.isfile(sam3_node_params_file):
            sam3_node_launch_args.append(('params_file', str(sam3_node_params_file)))
            print ("Using custom parameters for SAM3 node")

        orbbec_params_file = configs_path / "orbbec-camera.yaml"
        
        if os.path.isfile(orbbec_params_file):
            orbbec_node_launch_args.append(('config_file_path', str(orbbec_params_file)))
            print ("Using custom parameters for camera driver")

    else:
        print ("\n\nUsing default parameters\n\n")

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    orbbec_node_pkg, 'launch', 'gemini_330_series_low_cpu.launch.py'])]),
                launch_arguments=orbbec_node_launch_args
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    sam3_node_pkg, 'launch', 'sam3_inference.launch.py'])]),
                launch_arguments=sam3_node_launch_args
        ),

        Node(
            package='opennav_sam3_seg_demo',
            executable='image_display',
            name='image_display',
            remappings=[
                ('~/image', segmentation_topic),
            ],
        ),

        Node(
            package='opennav_sam3_seg_demo',
            executable='change_prompt_panel',
            name='change_prompt_panel',
            output='screen',
        ),

    ])
