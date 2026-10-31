"""Scenario settings shared by the PACE-ADS nodes.

The four personalization tests of the paper (Section 3.3) all run on CARLA Town06:

    traffic_light  - traffic signal interaction
    pedestrian     - pedestrian interaction
    work_zone      - work zone navigation
    car_following  - car following

Select one with the ROS parameter ``scenario``, e.g.
``ros2 run car_following control_node --ros-args -p scenario:=work_zone``.
"""

TRAFFIC_LIGHT = 'traffic_light'
PEDESTRIAN = 'pedestrian'
WORK_ZONE = 'work_zone'
CAR_FOLLOWING = 'car_following'

SCENARIOS = (TRAFFIC_LIGHT, PEDESTRIAN, WORK_ZONE, CAR_FOLLOWING)
DEFAULT_SCENARIO = TRAFFIC_LIGHT

TOWN = 'Town06'

# Legal speed limit of the road in km/h (DrivingPreference.speed_limit)
SPEED_LIMIT = 60

# Route destination of the ego vehicle and of the surrounding vehicles
# (index into world.get_map().get_spawn_points()).
DESTINATION_SPAWN_INDEX = 183

# Surrounding vehicles are spawned in the lane of spawn point 433. Each entry is the
# longitudinal offset in m from the previous vehicle (the first entry is measured from
# spawn point 433). It can be changed with the scenario_node parameter ``traffic_gaps``.
TRAFFIC_GAPS = {
    WORK_ZONE: [-15.5, 5.0, 15.0, 25.0],
    CAR_FOLLOWING: [-15.5],
}

# Order of the values of the std_msgs/Float32MultiArray message on /algorithm_params_topic.
# A NaN entry, or a missing trailing entry, leaves the corresponding parameter unchanged.
PARAMETER_LAYOUT = {
    TRAFFIC_LIGHT: ['max_speed', 'tf', 'T', 'a', 'b', 's_0'],
    PEDESTRIAN: ['max_speed', 'ped', 'T', 'a', 'b', 's_0', 'angle_th'],
    WORK_ZONE: ['max_speed', 'wz', 'T', 'a', 'b', 's_0', 'safety_distance'],
    CAR_FOLLOWING: ['max_speed', 'T', 'a', 'b', 's_0', 'lc_flag', 'cruise_flag'],
}

# Meaning of the behavior parameters (see carla_navigation/behavior_agent.py).
PARAMETER_DESCRIPTIONS = {
    'max_speed': 'target cruise speed in km/h (limited by the 60 km/h speed limit)',
    'tf': 'reaction time to a red traffic light in s; the vehicle starts to brake when the '
          'traffic light is closer than (current speed x tf)',
    'ped': 'reaction time to a pedestrian on the path in s; the vehicle starts to brake when '
           'the pedestrian is closer than (current speed x ped)',
    'wz': 'reaction time to the work zone in s; the lane change is prepared when the work '
          'zone is closer than (current speed x wz)',
    'angle_th': 'half angle in deg of the detection cone used to decide whether the '
                'pedestrian still blocks the path (smaller = the vehicle restarts earlier)',
    'safety_distance': 'minimum gap in m to the front and to the rear vehicle in the target '
                       'lane for accepting a lane change',
    'T': 'IDM desired time headway in s',
    'a': 'IDM maximum acceleration in m/s^2',
    'b': 'IDM comfortable deceleration in m/s^2',
    's_0': 'IDM minimum gap in m (in the pedestrian scenario also the stopping distance to '
           'the pedestrian)',
    'lc_flag': '1 = change to the adjacent left lane, 0 = stay in the current lane',
    'cruise_flag': '1 = cruise at max_speed without following the lead vehicle, 0 = follow '
                   'the lead vehicle',
}

# Default driving preference of the ego vehicle in each scenario (benchmark configuration,
# used while the occupant is relaxed). Parameters that are not listed keep the defaults of
# carla_navigation.behavior_agent.DrivingPreference.
DRIVING_PREFERENCE = {
    TRAFFIC_LIGHT: {
        'max_speed': 40, 'ttc_tf': 4.0, 'ttc_ped': 1.5, 'angle_th': 20, 'ttc_wz': 5.0,
        'safety_distance': 10.0, 'ttc_car': 2.0, 'T': 2.5, 'a': 0.55, 'b': 1.67, 's_0': 3.0,
    },
    PEDESTRIAN: {
        'max_speed': 40, 'ttc_tf': 3.0, 'ttc_ped': 3.5, 'angle_th': 25, 'ttc_wz': 5.0,
        'safety_distance': 10.0, 'ttc_car': 2.0, 'T': 2.5, 'a': 0.35, 'b': 1.67, 's_0': 2.0,
    },
    WORK_ZONE: {
        'max_speed': 40, 'ttc_tf': 3.0, 'ttc_ped': 4.5, 'angle_th': 30, 'ttc_wz': 4.0,
        'safety_distance': 5.0, 'ttc_car': 2.0, 'T': 1.5, 'a': 0.75, 'b': 1.67, 's_0': 4.0,
    },
    CAR_FOLLOWING: {
        'max_speed': 40, 'ttc_tf': 3.0, 'ttc_ped': 4.5, 'angle_th': 30, 'ttc_wz': 6.0,
        'safety_distance': 20.0, 'ttc_car': 2.0, 'T': 1.5, 'a': 0.75, 'b': 1.67, 's_0': 3.0,
    },
}


def get_scenario(node):
    """Return the validated value of the ``scenario`` ROS parameter of ``node``."""
    if node.has_parameter('scenario'):
        scenario = node.get_parameter('scenario').value
    else:
        scenario = node.declare_parameter('scenario', DEFAULT_SCENARIO).value
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Choose one of: {', '.join(SCENARIOS)}")
    return scenario
