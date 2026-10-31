# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations.

PACE-ADS: this agent drives the surrounding vehicles created by scenario_node
(the vehicles in the adjacent lane of the work zone scenario and the lead vehicle
of the car-following scenario). """

import math
import numpy as np
import carla
from car_following.lat_controller import VehiclePIDController_lat
from carla_navigation.basic_agent import BasicAgent
from carla_navigation.local_planner import RoadOption
from carla_navigation.misc import get_speed, positive, get_trafficlight_trigger_location


class DrivingPreference(object):
    """Class for changing driving preference."""
    max_speed = 40
    speed_limit = 60
    ttc_tf = 3.0
    ttc_ped = 4.5
    react_dist_tf = max_speed/3.6 * ttc_tf
    react_dist_ped = 30
    angle_th = 30

    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 6
    tailgate_counter = 0
    T = 1.5
    a = 0.75
    b = 1.67
    s_0 = 4.0

    speed_lim_dist = 3

class TrafficAgent(BasicAgent):
    """
    TrafficAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, front_veh, s_new, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param front_veh: preceding vehicle to follow (None for the first vehicle)
            :param s_new: desired minimum gap to the preceding vehicle (m)
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._behavior = DrivingPreference()
        self._speed = 0
        self._speed_limit = self._behavior.speed_limit
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._sampling_resolution = 4.5
        self.destination = None
        self.replan_flag = 0
        self.walker_laneid = None
        self.ego_laneid = None
        self.sign_lane_id = -6
        self.lc_direct = -1
        self.lc_destin_flag = 0
        self.front_veh = front_veh
        self.s_new = s_new
        self.args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}
        self.lat_controller = VehiclePIDController_lat(self._vehicle, args_lateral=self.args_lateral_dict)

    def update_driving_preference(self, param_dict):
        self._behavior.max_speed = param_dict['max_speed']
        self._behavior.speed_limit = param_dict['speed_limit']

        if self._behavior.ttc_tf != param_dict['tf']:
            self._behavior.ttc_tf = param_dict['tf']
            self._behavior.react_dist_tf = get_speed(self._vehicle)/3.6 * self._behavior.ttc_tf

        if self._behavior.ttc_ped != param_dict['ped']:
            self._behavior.ttc_ped = param_dict['ped']
            self._behavior.react_dist_ped = get_speed(self._vehicle)/3.6 * self._behavior.ttc_ped

        self._behavior.angle_th = param_dict['angle_th']
        self._behavior.speed_decrease = param_dict['speed_decrease']
        self._behavior.safety_time = param_dict['safety_time']
        self._behavior.min_proximity_threshold = param_dict['min_proximity_threshold']
        self._behavior.braking_distance = param_dict['braking_distance']
        self._behavior.tailgate_counter = param_dict['tailgate_counter']
        self._behavior.T = param_dict['T']
        self._behavior.a = param_dict['a']
        self._behavior.b = param_dict['b']
        self._behavior.s_0 = param_dict['s_0']

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._behavior.speed_limit
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

        # get ego lane id
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)
        self.ego_laneid = ego_wpt.lane_id

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, light = self._affected_by_traffic_light(lights_list, self._behavior.react_dist_tf)

        return affected, light

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state: # if there is no vehicle in the right lane
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance, _ = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance, _ = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance, _ = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 200]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance, _ = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance, _ = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            if self.walker_laneid is not None and self.walker_laneid <= self.ego_laneid:
                walker_state, walker, distance, laneid = self._vehicle_obstacle_detected(walker_list,200, up_angle_th=self._behavior.angle_th)
            else:
                walker_state, walker, distance, laneid = self._vehicle_obstacle_detected(walker_list,200, up_angle_th=80)

        if walker_state:
            self.replan_flag = 1
            if laneid is not None:
                self.walker_laneid = laneid
            else:
                print("No lane id for walker!")

        if self.replan_flag == 1 and not walker_state:
            if self.destination is not None:
                self.set_destination(self.destination)
                self.replan_flag = 0
            else:
                print("No destination to replan!")

        return walker_state, walker, distance

    def workzone_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any workzone.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return workzone_state: True if the work zone closes the lane, False if not
            :return sign: construction sign of the work zone
            :return distance: distance to the construction sign
        """

        sign_list = self._world.get_actors().filter("static.prop.warningconstruction")
        def dist(w): return w.get_location().distance(waypoint.transform.location)

        if not sign_list:
            return False, None, None

        sign = [w for w in sign_list if dist(w) < 200]
        if not sign:
            return False, None, None
        sign = sign[0]

        if self.sign_lane_id == self.ego_laneid:
            # calculate distance between sign and ego vehicle
            sign_location = sign.get_location()
            ego_location = self._vehicle.get_location()
            distance = abs(sign_location.x - ego_location.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            return True, sign, distance
        else:
            return False, None, None

    def car_following_manager(self, distance, debug=False):
        """
        Car-following control of a surrounding vehicle: IDM with respect to the
        preceding vehicle (front_veh) and lateral PID control.

            :param distance: longitudinal distance to the preceding vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        vehicle_speed = get_speed(self._vehicle) / 3.6  # m/s
        front_velocity = get_speed(self.front_veh) / 3.6  # m/s
        relative_velocity = vehicle_speed - front_velocity
        acc = self.IDM(vehicle_speed, distance, relative_velocity,
                            v_0=self._behavior.max_speed/3.6, T=0.1,
                            a=5.0, b=2.0, s_0=self.s_new)

        if acc >= 0:
            throttle = acc*2.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(acc)/8.0 # tesla model 3 max deceleration is 8 m/s^2

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        front_waypoint = ego_vehicle_wp.next(20.0)[0]
        steering = self.lat_controller.run_step(waypoint=front_waypoint)

        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.steer = steering
        control.hand_brake = False

        return control

    def run_step(self, target_speed=None, debug=False):
        """
        Execute one step of navigation.

            :param target_speed: prescribed speed in km/h (lead vehicle of the car-following
                scenario); None for the surrounding vehicles of the work zone scenario
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information() # Update next waypoint, direction (RoadOption of waypoint), and speed limit/target.

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        if target_speed is not None:
            # Lead vehicle of the car-following scenario: track the prescribed speed
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

            ego_vehicle_loc = self._vehicle.get_location()
            ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
            front_waypoint = ego_vehicle_wp.next(20.0)[0]
            steering = self.lat_controller.run_step(waypoint=front_waypoint)
            control.steer = steering

        elif self.front_veh is not None:
            # Traffic flow: follow the preceding vehicle
            ego_vehicle_location = self._vehicle.get_location()
            front_vehicle_location = self.front_veh.get_location()
            distance = abs(front_vehicle_location.x - ego_vehicle_location.x)

            control = self.car_following_manager(distance)

        else:
            # Traffic flow: the first vehicle cruises at max_speed
            target_speed = min([self._behavior.max_speed, self._speed_limit])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

            ego_vehicle_loc = self._vehicle.get_location()
            ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
            front_waypoint = ego_vehicle_wp.next(20.0)[0]
            steering = self.lat_controller.run_step(waypoint=front_waypoint)
            control.steer = steering

        return control

    def lane_change_control(self, waypoint, direction):
        """
        This method implements a lane change behavior.

            :param waypoint: current waypoint of the agent
            :param direction: direction to change the lane. 1 for right, -1 for left
            :return control: carla.VehicleControl
        """
        if self.lc_destin_flag == 0:
            front_wp = waypoint.next(15.0)[0]

            if direction == -1:
                left_wpt = front_wp.get_left_lane()
                # draw the target waypoint in the left lane
                self._world.debug.draw_point(left_wpt.transform.location, size=0.5, color=carla.Color(0, 255, 0), life_time=1.0)
            elif direction == 1:
                right_wpt = front_wp.get_right_lane()
            else:
                print("Invalid direction for lane change!")

            if direction == 1 and right_wpt.lane_type == carla.LaneType.Driving:
                print("Changing lane to the right!")
                self.set_destination(self.destination, right_wpt.transform.location)
                self.lc_destin_flag = 1
            elif direction == -1 and left_wpt.lane_type == carla.LaneType.Driving:
                print("Changing lane to the left!")
                self.set_destination(self.destination, left_wpt.transform.location)
                self.lc_destin_flag = 1

        target_speed = min([self._behavior.max_speed, self._speed_limit])
        self._local_planner.set_speed(target_speed)
        control = self._local_planner.run_step()

        return control

    def cruising(self):
        ego_speed = get_speed(self._vehicle)/3.6    # m/s
        distance = 100000
        acc = self.IDM(ego_speed, distance, ego_speed,
                        v_0=self._behavior.max_speed/3.6, T=1.5,
                        a=0.75, b=1.67, s_0=1.0)

        if acc >= 0:
            throttle = acc*2.0
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(acc)/8.0 # tesla model 3 max deceleration is 8 m/s^2

        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False

        return control


    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def IDM(self, vel_car, rel_dis, rel_vel, v_0, T, a, b, s_0):
        if v_0 == 0.0:
            v_0 = 1e-3
        Term_int = s_0 + vel_car * T + (vel_car * rel_vel / (2 * math.sqrt(a * b)))
        acc = a * (1 - ((vel_car / v_0) ** 4) - (Term_int / rel_dis) ** 2)
        return acc

    def decel_stop_tf(self, traffic_light):
        """
        This method implements a deceleration behavior to stop the vehicle
        at traffic lights.

            :param traffic_light: carla.Actor object
            :return control: carla.VehicleControl
        """
        ego_vehicle_location = self._vehicle.get_location()
        trigger_location = get_trafficlight_trigger_location(traffic_light)
        trigger_wp = self._map.get_waypoint(trigger_location)
        print(f"tf location: {trigger_location}")
        distance = abs(trigger_wp.transform.location.x - ego_vehicle_location.x)
        if distance > 1.8:
            ego_speed = get_speed(self._vehicle)/3.6    # m/s
            acc = self.IDM(ego_speed, distance, ego_speed,
                            v_0=self._behavior.max_speed/3.6, T=1.5,
                            a=0.75, b=1.67, s_0=1.0)

            if acc >= 0:
                throttle = acc*2.0
                brake = 0.0
            else:
                throttle = 0.0
                brake = abs(acc)/8.0 # tesla model 3 max deceleration is 8 m/s^2
        else:
            throttle = 0.0
            brake = 1.0

        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False

        return control

    def decel_stop_ped(self, distance, final_dist):
        """
        This method implements a deceleration behavior to stop the vehicle
        in front of a pedestrian.

            :param distance: distance to the pedestrian
            :param final_dist: desired stopping distance to the pedestrian
            :return control: carla.VehicleControl
        """
        if distance <= final_dist+0.8:
            throttle = 0.0
            brake = 1.0
        else:
            ego_speed = get_speed(self._vehicle)/3.6    # m/s
            acc = self.IDM(ego_speed, distance, ego_speed,
                            v_0=self._behavior.max_speed/3.6, T=1.5,
                            a=0.75, b=1.67, s_0=final_dist)

            if acc >= 0:
                throttle = acc*2.0
                brake = 0.0
            else:
                throttle = 0.0
                brake = abs(acc)/8.0 # tesla model 3 max deceleration is 8 m/s^2

        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False

        return control

    def car_following_IDM(self):
        """
        This method implements a car-following behavior based on the IDM model.

            :return control: carla.VehicleControl
        """
        if self.front_veh is not None:
            ego_speed = get_speed(self._vehicle)/3.6    # m/s
            vel_car = get_speed(self.front_veh)/3.6    # m/s
            rel_vel = ego_speed - vel_car

            ego_vehicle_location = self._vehicle.get_location()
            front_vehicle_location = self.front_veh.get_location()
            distance = abs(front_vehicle_location.x - ego_vehicle_location.x)
        else:
            ego_speed = get_speed(self._vehicle)/3.6
            rel_vel = ego_speed
            distance = 100000

        target_speed = min([self._behavior.max_speed, self._speed_limit])

        acc = self.IDM(ego_speed, distance, rel_vel,
                        v_0=target_speed, T=self._behavior.T,
                        a=self._behavior.a, b=self._behavior.b, s_0=self.s_new)

        if acc >= 0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(acc)/8.0

        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False

        return control
