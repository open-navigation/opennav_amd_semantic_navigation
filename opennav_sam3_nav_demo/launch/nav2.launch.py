# Copyright (c) 2026 Open Navigation LLC
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

"""This is all-in-one launch script intended for use by nav2 developers."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    SetEnvironmentVariable,
)

from launch.substitutions import LaunchConfiguration

from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode, ParameterFile
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    # Get the launch directory
    opennav_sam3_nav_dir = get_package_share_directory('opennav_sam3_nav_demo')
    bt_navigator_dir = get_package_share_directory('nav2_bt_navigator')

    # Create the launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = os.path.join(opennav_sam3_nav_dir, 'config', 'nav2_semantic_params.yaml')
    nav2pose_bt_xml = LaunchConfiguration('nav2pose_bt_xml')

    lifecycle_nodes = [
        'controller_server',
        'smoother_server',
        'planner_server',
        'behavior_server',
        'bt_navigator',
        'waypoint_follower',
        'velocity_smoother',
        'collision_monitor']

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'default_nav_to_pose_bt_xml': nav2pose_bt_xml}

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key='',
            param_rewrites=param_substitutions,
            convert_types=True),
        allow_substs=True)

    stdout_linebuf_envvar = SetEnvironmentVariable(
        'RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    # Declare the launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true',
    )

    declare_bt_xml_cmd = DeclareLaunchArgument(
        'nav2pose_bt_xml',
        default_value=os.path.join(
            bt_navigator_dir, 'behavior_trees', 'navigate_to_pose_w_replanning_and_recovery.xml'),
        description='Which navigate to pose BT to use',
    )

    # Specify the actions
    bringup_cmd_group = GroupAction([
        Node(
            name='nav2_container',
            package='rclcpp_components',
            executable='component_container_isolated',
            parameters=[configured_params, {'autostart': True}],
            arguments=['--ros-args', '--log-level', 'info'],
            output='screen'),
    
        # Local odometry-only navigation for demonstration purposes in a variety of environments
        # without needing to setup tenuous localization solutions outdoors, open fields, indoors, etc.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'map', 'odom'],
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        LoadComposableNodes(
            target_container='nav2_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='nav2_controller',
                    plugin='nav2_controller::ControllerServer',
                    name='controller_server',
                    parameters=[configured_params, {'use_sim_time': use_sim_time}],
                    remappings=[('cmd_vel', 'cmd_vel_nav')]),
                ComposableNode(
                    package='nav2_smoother',
                    plugin='nav2_smoother::SmootherServer',
                    name='smoother_server',
                    parameters=[configured_params, {'use_sim_time': use_sim_time}]),
                ComposableNode(
                    package='nav2_planner',
                    plugin='nav2_planner::PlannerServer',
                    name='planner_server',
                    parameters=[configured_params, {'use_sim_time': use_sim_time}]),
                ComposableNode(
                    package='nav2_behaviors',
                    plugin='behavior_server::BehaviorServer',
                    name='behavior_server',
                    parameters=[configured_params, {'use_sim_time': use_sim_time}]),
                ComposableNode(
                    package='nav2_bt_navigator',
                    plugin='nav2_bt_navigator::BtNavigator',
                    name='bt_navigator',
                    parameters=[configured_params, {'use_sim_time': use_sim_time}]),
                ComposableNode(
                    package='nav2_waypoint_follower',
                    plugin='nav2_waypoint_follower::WaypointFollower',
                    name='waypoint_follower',
                    parameters=[configured_params, {'use_sim_time': use_sim_time}]),
                ComposableNode(
                    package='nav2_velocity_smoother',
                    plugin='nav2_velocity_smoother::VelocitySmoother',
                    name='velocity_smoother',
                    parameters=[configured_params, {'use_sim_time': use_sim_time}],
                    remappings=[('cmd_vel', 'cmd_vel_nav')]),
                ComposableNode(
                    package='nav2_collision_monitor',
                    plugin='nav2_collision_monitor::CollisionMonitor',
                    name='collision_monitor',
                    parameters=[configured_params, {'use_sim_time': use_sim_time}]),
                ComposableNode(
                    package='nav2_lifecycle_manager',
                    plugin='nav2_lifecycle_manager::LifecycleManager',
                    name='lifecycle_manager_navigation',
                    parameters=[{'use_sim_time': use_sim_time,
                                'autostart': True,
                                'node_names': lifecycle_nodes}]),
            ],
        )
    ])

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(declare_bt_xml_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(bringup_cmd_group)
    return ld
