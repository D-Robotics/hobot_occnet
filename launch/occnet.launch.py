# Copyright (c) 2022，Horizon Robotics.
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

import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default_value'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def generate_launch_description():
    occ_model_file_path =  os.path.join(
        get_package_share_directory('hobot_occnet'),
        'config',
        'drobotics_occ_network_v0_0_1.bin'
    )

    local_image_dir =  os.path.join(
        get_package_share_directory('hobot_occnet'),
        'config'
    )

    node_params = [
        {'name':'occ_model_file_path', 'default_value': occ_model_file_path, 'description': 'occ_model_file_path'},
        {'name':'local_image_dir', 'default_value': local_image_dir, 'description': 'local_image_path'},
        {'name':'log_level', 'default_value':'info', 'description': 'log_level'},
    ]

    launch = declare_configurable_parameters(node_params)
    launch.append(Node(
        package='hobot_occnet',
        executable='hobot_occnet_node',
        output='screen',
        parameters=[set_configurable_parameters(node_params)],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    ))

    return LaunchDescription(launch)
