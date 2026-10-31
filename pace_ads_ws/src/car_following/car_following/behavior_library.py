"""Behavior library (Alib) of the motion planning module, as documented to the coordinator.

For each scenario the library lists the driving behaviors that the host AV stack
(carla_navigation/behavior_agent.py) can execute, with a functional description, and the
configurable parameters with their meaning, default value and permitted range. The permitted
ranges are the pre-defined safety envelope: the parameters decided by the coordinator are
clamped to them before they are sent to the control node.
"""

import math

from car_following.scenario_config import (CAR_FOLLOWING, DRIVING_PREFERENCE, PARAMETER_DESCRIPTIONS,
                                           PARAMETER_LAYOUT, PEDESTRIAN, TRAFFIC_LIGHT, WORK_ZONE)

# Parameter name on /algorithm_params_topic -> attribute of DrivingPreference
PREFERENCE_KEY = {'tf': 'ttc_tf', 'ped': 'ttc_ped', 'wz': 'ttc_wz'}

UNITS = {
    'max_speed': 'km/h', 'tf': 's', 'ped': 's', 'wz': 's', 'angle_th': 'deg',
    'safety_distance': 'm', 'T': 's', 'a': 'm/s^2', 'b': 'm/s^2', 's_0': 'm',
}

CAR_FOLLOW = 'car following'
LANE_CHANGE_LEFT = 'lane change left'

BEHAVIORS = {
    TRAFFIC_LIGHT: {
        'stop and go at traffic light': (
            'Cruise in the current lane at max_speed. When the red traffic light is closer than '
            '(current speed x tf), decelerate and stop at the stop line. After the light turns '
            'green, accelerate back to max_speed with the Intelligent Driver Model (maximum '
            'acceleration a, comfortable deceleration b, time headway T, minimum gap s_0).'),
    },
    PEDESTRIAN: {
        'yield to pedestrian': (
            'Cruise in the current lane at max_speed. When a pedestrian on or next to the path is '
            'closer than (current speed x ped), decelerate and stop about (s_0 + 0.8) m before '
            'the pedestrian. Wait until the pedestrian has left the detection cone of half angle '
            'angle_th in front of the vehicle, then accelerate back to max_speed with the '
            'Intelligent Driver Model (a, b, T, s_0).'),
    },
    WORK_ZONE: {
        'work zone lane change': (
            'Cruise in the current lane at max_speed. When the work zone that closes the ego lane '
            'is closer than (current speed x wz), change to the adjacent left lane as soon as the '
            'gap is acceptable: the gaps to the front and to the rear vehicle in the target lane '
            'must be at least safety_distance, and the gap between these two vehicles at least '
            'the ego length plus 2 x safety_distance. While waiting for a gap, approach the work '
            'zone with the Intelligent Driver Model (a, b, T, s_0), stopping before it if '
            'necessary. After the lane change, cruise at max_speed.'),
    },
    CAR_FOLLOWING: {
        CAR_FOLLOW: (
            'Follow the lead vehicle in the current lane with the Intelligent Driver Model: desired '
            'speed max_speed, time headway T, maximum acceleration a, comfortable deceleration b '
            'and minimum gap s_0. A max_speed below the speed of the lead vehicle makes the ego '
            'vehicle fall back smoothly; the model still keeps a safe gap when the lead vehicle '
            'brakes.'),
        LANE_CHANGE_LEFT: (
            'Change to the adjacent left lane and then cruise there at max_speed. Choose it only '
            'if the Driver Agent reports that the adjacent left lane exists and is free of nearby '
            'vehicles. The lane change cannot be undone: afterwards the vehicle keeps cruising in '
            'the new lane and max_speed remains adjustable.'),
    },
}

# Permitted range of every parameter set by the coordinator (safety envelope). The upper
# speed bound is the speed limit of the test (60 km/h; cruising limited to 40 km/h in the
# pedestrian and work zone tests).
PARAMETER_RANGES = {
    TRAFFIC_LIGHT: {
        'max_speed': (15.0, 60.0), 'tf': (1.0, 8.0), 'T': (0.5, 5.0), 'a': (0.1, 3.0),
        'b': (1.0, 4.0), 's_0': (1.0, 8.0),
    },
    PEDESTRIAN: {
        'max_speed': (15.0, 40.0), 'ped': (1.0, 8.0), 'T': (0.5, 5.0), 'a': (0.1, 3.0),
        'b': (1.0, 4.0), 's_0': (1.0, 8.0), 'angle_th': (15.0, 45.0),
    },
    WORK_ZONE: {
        'max_speed': (15.0, 40.0), 'wz': (2.0, 10.0), 'T': (0.5, 5.0), 'a': (0.1, 3.0),
        'b': (1.0, 4.0), 's_0': (2.0, 8.0), 'safety_distance': (1.0, 25.0),
    },
    CAR_FOLLOWING: {
        'max_speed': (30.0, 60.0), 'T': (0.5, 5.0), 'a': (0.1, 3.0), 'b': (1.0, 4.0),
        's_0': (1.5, 8.0),
    },
}


def coordinator_parameters(scenario):
    """Parameters set by the coordinator (the car-following flags follow from the behavior)."""
    return list(PARAMETER_RANGES[scenario])


def default_value(scenario, name):
    """Default value of a parameter in a scenario (scenario_config.DRIVING_PREFERENCE)."""
    return float(DRIVING_PREFERENCE[scenario][PREFERENCE_KEY.get(name, name)])


def _fmt(value):
    return f'{value:g}'


class BehaviorLibrary:
    """Behavior library of one scenario, with the parameter values currently in use."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.behaviors = BEHAVIORS[scenario]
        self.ranges = PARAMETER_RANGES[scenario]
        self.defaults = {name: default_value(scenario, name) for name in self.ranges}
        self.current = dict(self.defaults)
        self.behavior = next(iter(self.behaviors))
        self.lane_changed = False

    def describe(self):
        """Text description of the library for the coordinator."""
        lines = ['Behaviors:']
        for name, description in self.behaviors.items():
            lines.append(f'- {name}: {description}')
        lines.append(f'Current behavior: {self.behavior}')
        if self.lane_changed:
            lines.append('The lane change has been executed; the vehicle cruises in the new lane.')
        lines.append('Parameters (meaning; default; permitted range; current value):')
        for name, (low, high) in self.ranges.items():
            unit = UNITS[name]
            lines.append(f'- {name}: {PARAMETER_DESCRIPTIONS[name]}; default {_fmt(self.defaults[name])} {unit}; '
                         f'range [{_fmt(low)}, {_fmt(high)}] {unit}; current {_fmt(self.current[name])} {unit}')
        return '\n'.join(lines)

    def apply(self, behavior, parameters):
        """
        Clamp the coordinator's decision to the safety envelope and remember it.

            :param behavior: behavior selected by the coordinator
            :param parameters: dictionary {parameter name: value} decided by the coordinator
            :return: (values in the order of PARAMETER_LAYOUT[scenario], list of notes)
        """
        notes = []
        if behavior not in self.behaviors:
            notes.append(f"Unknown behavior '{behavior}'; the behavior '{self.behavior}' is kept.")
            behavior = self.behavior
        for name, (low, high) in self.ranges.items():
            value = parameters.get(name)
            if not isinstance(value, (int, float)) or math.isnan(value):
                notes.append(f'No valid value for {name}; the current value {_fmt(self.current[name])} is kept.')
                continue
            clamped = min(max(float(value), low), high)
            if clamped != value:
                notes.append(f'{name} = {_fmt(value)} is outside [{_fmt(low)}, {_fmt(high)}] and was set to {_fmt(clamped)}.')
            self.current[name] = clamped

        if self.scenario == CAR_FOLLOWING:
            if behavior == LANE_CHANGE_LEFT:
                self.lane_changed = True
            elif self.lane_changed:
                notes.append('The lane change cannot be undone; the vehicle keeps cruising in the new lane.')
        self.behavior = behavior

        values = []
        for name in PARAMETER_LAYOUT[self.scenario]:
            if name == 'lc_flag':
                values.append(1.0 if self.lane_changed else 0.0)
            elif name == 'cruise_flag':
                values.append(0.0)
            else:
                values.append(self.current.get(name, float('nan')))
        return values, notes
