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

def generate_launch_description():
    sam3_node_pkg = FindPackageShare('opennav_sam3_inference')
    orbbec_node_pkg = FindPackageShare('orbbec_camera')

    sam3_node_launch_args = [
        ('image_topic', '/camera/color/image_raw'), 
        ('sensor_processing_pipeline', 'false'),
        ('log_level', 'info')
    ]
    
    orbbec_node_launch_args = []

    image_display_args = [
        '--lock-perspective',
        '--hide-title', 
        '--freeze-layout'
    ]

    if os.path.isdir(configs_path):
        sam3_node_params_file = configs_path / "sam3-inference.yaml"
        
        if os.path.isfile(sam3_node_params_file):
            sam3_node_launch_args.append(('params_file', str(sam3_node_params_file)))
            print ("Using custom parameters for SAM3 node")

        orbbec_params_file = configs_path / "orbbec-camera.yaml"
        
        if os.path.isfile(orbbec_params_file):
            orbbec_node_launch_args.append(('config_file_path', str(orbbec_params_file)))
            print ("Using custom parameters for camera driver")
        
        image_viewer_perspective_file = configs_path / "sam3-demo.perspective"

        if os.path.isfile(image_viewer_perspective_file):
            image_display_args[:0] = ['--perspective-file', str(image_viewer_perspective_file)]
        else:
            image_display_args[:0] = ['--standalone', 'rqt_image_view/ImageView']
            image_display_args.extend(['--args', '/sam3_inference/segmentation_mask'])

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

        # Alternative way for image_view
        # Node(
        #     package='rqt_image_view',
        #     executable='rqt_image_view',
        #     name='sam3_image_view',
        #     arguments=[
        #         '/sam3_inference/segmentation_mask',
        #         '--hide-title',
        #     ],
        #     output='screen',
        # ),

        Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='image_display',
            arguments=image_display_args,
            output='screen',
        ),

        Node(
            package='opennav_sam3_seg_demo',
            executable='change_prompt_panel',
            name='change_prompt_panel',
            output='screen',
        ),

    ])
