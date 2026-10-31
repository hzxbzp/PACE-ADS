from setuptools import setup
from glob import glob
import os

package_name = 'car_following'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'carla_navigation'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zhipeng Bao',
    maintainer_email='zb28097@uga.edu',
    description='PACE-ADS: CARLA/ROS 2 simulation of the personalization tests',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'scenario_node = car_following.scenario_node:main',
            'psychological_monitor_node = car_following.psychological_monitor_node:main',
            'control_node = car_following.control_node:main',
            'decision_node = car_following.decision_node:main',
            'record_node = car_following.record_node:main',
        ],
    },
)
