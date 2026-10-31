"""Starts carla_ros_bridge (loads Town06), spawns the ego vehicle with its sensors and rviz2.

    ros2 launch car_following carla_ros_bridge.launch.py scenario:=<scenario>

scenario: traffic_light, pedestrian, work_zone or car_following. It selects
config/objects_<scenario>.json, which only differ in the spawn point of the ego vehicle.
"""

import os

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


config_dir = os.path.join(get_package_share_directory('car_following'), 'config')


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='scenario',
            default_value='traffic_light',
            description='Personalization scenario: traffic_light, pedestrian, work_zone or car_following'
        ),
        launch.actions.DeclareLaunchArgument(
            name='host',
            default_value='localhost'
        ),
        launch.actions.DeclareLaunchArgument(
            name='port',
            default_value='2000'
        ),
        launch.actions.DeclareLaunchArgument(
            name='timeout',
            default_value='60'
        ),
        launch.actions.DeclareLaunchArgument(
            name='town',
            default_value='Town06'
        ),
        launch.actions.DeclareLaunchArgument(
            name='passive',
            default_value='False',
        ),
        launch.actions.DeclareLaunchArgument(
            name='synchronous_mode',
            default_value='True',
            description='Enable/disable synchronous mode. If enabled, the ROS bridge waits until the expected data is received for all sensors'
        ),
        launch.actions.DeclareLaunchArgument(
            name='synchronous_mode_wait_for_vehicle_control_command',
            default_value='False',
        ),
        launch.actions.DeclareLaunchArgument(
            name='fixed_delta_seconds',
            default_value='0.05'
        ),
        launch.actions.DeclareLaunchArgument(
            name='register_all_sensors',
            default_value='True',
            description='Enable/disable the registration of all sensors. If disabled, only sensors spawned by the bridge are registered'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ego_vehicle_role_name', # role_name of the ego vehicle, must be the id used in objects_<scenario>.json
            default_value='hero',
            description='Role names to identify ego vehicles. '
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'carla_ros_bridge'), 'carla_ros_bridge.launch.py')
            ),
            launch_arguments={
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'town': launch.substitutions.LaunchConfiguration('town'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'passive': launch.substitutions.LaunchConfiguration('passive'),
                'synchronous_mode': launch.substitutions.LaunchConfiguration('synchronous_mode'),
                'synchronous_mode_wait_for_vehicle_control_command': launch.substitutions.LaunchConfiguration('synchronous_mode_wait_for_vehicle_control_command'),
                'fixed_delta_seconds': launch.substitutions.LaunchConfiguration('fixed_delta_seconds'),
                'register_all_sensors': launch.substitutions.LaunchConfiguration('register_all_sensors'),
                'ego_vehicle_role_name': launch.substitutions.LaunchConfiguration('ego_vehicle_role_name')
            }.items()
        ),
        ######## arguments for spawning the ego vehicle and its sensors ########
        launch.actions.DeclareLaunchArgument(
            name='objects_definition_file',
            default_value=[os.path.join(config_dir, 'objects_'), LaunchConfiguration('scenario'), '.json'],
            description='Ego vehicle and sensor definition (the ego spawn point depends on the scenario)'
        ),
        launch.actions.DeclareLaunchArgument(
            name='spawn_sensors_only',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='role_name',
            default_value='hero'
        ),
        launch.actions.DeclareLaunchArgument(
            name='control_id',
            default_value='control'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'carla_spawn_objects'), 'carla_spawn_objects.launch.py')
            ),
            launch_arguments={
                'objects_definition_file': launch.substitutions.LaunchConfiguration('objects_definition_file'),
                'spawn_sensors_only': launch.substitutions.LaunchConfiguration('spawn_sensors_only')
            }.items()
        ),
        # set the initial pose via rviz for the vehicle with role_name
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'carla_spawn_objects'), 'set_initial_pose.launch.py')
            ),
            launch_arguments={
                'role_name': launch.substitutions.LaunchConfiguration('role_name'),
                'control_id': launch.substitutions.LaunchConfiguration('control_id')
            }.items()
        ),
        launch.actions.DeclareLaunchArgument(
            name='rviz',
            default_value='True',
            description='Start rviz2 (closing rviz2 stops the launch)'
        ),
        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            remappings=[
                (
                    "carla/hero/spectator_pose",
                    "/carla/hero/rgb_view/control/set_transform"
                )
            ],
            arguments=[
                '-d', os.path.join(config_dir, 'carla_ad_demo_ros2.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz')),
            on_exit=launch.actions.Shutdown()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
