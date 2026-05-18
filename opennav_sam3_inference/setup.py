# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from glob import glob

from setuptools import setup

package_name = 'opennav_sam3_inference'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, package_name + '.tracker'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='ROS 2 node for SAM3 text-prompted image segmentation using PyTorch.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sam3_node = opennav_sam3_inference.sam3_node:main',
        ],
    },
)
