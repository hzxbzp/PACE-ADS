# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations.

PACE-ADS: the ego vehicle is driven by this agent (host AV stack). Its driving
preference (cruise speed, reaction times, gap acceptance and IDM parameters) is
updated at run time with the parameters decided by the coordinator
(update_driving_preference). The behavior executed in run_step depends on the
personalization scenario. """

import math
import numpy as np
import carla
from carla_navigation.basic_agent import BasicAgent
from carla_navigation.local_planner import RoadOption
from carla_navigation.misc import get_speed, positive, get_trafficlight_trigger_location
from car_following.scenario_config import CAR_FOLLOWING, PEDESTRIAN, TRAFFIC_LIGHT, WORK_ZONE


class DrivingPreference(object):
    """Class for changing driving preference."""
    max_speed = 40
    speed_limit = 60
    ttc_tf = 4.0
    react_dist_tf = max_speed/3.6 * ttc_tf

    ttc_ped = 1.5
    react_dist_ped = max_speed/3.6 * ttc_ped
    angle_th = 20

    ttc_wz = 5.0
    react_dist_wz = max_speed/3.6 * ttc_wz
    safety_distance = 10.0
    ttc_car = 2.0

    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 6
    tailgate_counter = 0

    T = 2.5
    a = 0.55
    b = 1.67
    s_0 = 3.0
    lc_flag = 0.0
    cruise_flag = 0.0

    speed_lim_dist = 3

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, front_vehicle=None, opt_dict={}, map_inst=None, grp_inst=None,
                 scenario=None, preference=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param front_vehicle: lead vehicle of the car-following scenario (None otherwise)
            :param scenario: personalization scenario (see car_following.scenario_config)
            :param preference: dictionary that overrides the DrivingPreference defaults
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._behavior = DrivingPreference()
        if preference:
            for key, value in preference.items():
                setattr(self._behavior, key, value)
            # reaction distances are defined as in DrivingPreference
            self._behavior.react_dist_tf = self._behavior.max_speed/3.6 * self._behavior.ttc_tf
            self._behavior.react_dist_ped = self._behavior.max_speed/3.6 * self._behavior.ttc_ped
            self._behavior.react_dist_wz = self._behavior.max_speed/3.6 * self._behavior.ttc_wz
        self._scenario = scenario
        self._speed = 0
        self._speed_limit = self._behavior.speed_limit
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._sampling_resolution = 4.5
        self.destination = None
        self.ped_flag = 0
        self.tf_flag = 0
        self.replan_flag = 0
        self.walker_laneid = None
        self.ego_laneid = None
        self.sign_lane_id = -6  # lane id of the lane closed by the work zone
        self.lc_direct = -1     # lane change direction: -1 = left, 1 = right
        self.lc_destin_flag = 0

        self.front_vehicle = front_vehicle

    def update_driving_preference(self, param_dict):
        """
        Update the driving preference with the parameters decided by the coordinator.
        Only the parameters contained in param_dict are changed.

            :param param_dict: dictionary {parameter name: value}, see
                car_following.scenario_config.PARAMETER_LAYOUT
        """
        if 'max_speed' in param_dict:
            self._behavior.max_speed = param_dict['max_speed']

        if 'tf' in param_dict and self._behavior.ttc_tf != param_dict['tf']:
            self._behavior.ttc_tf = param_dict['tf']
            self._behavior.react_dist_tf = get_speed(self._vehicle)/3.6 * self._behavior.ttc_tf

        if 'ped' in param_dict and self._behavior.ttc_ped != param_dict['ped']:
            self._behavior.ttc_ped = param_dict['ped']
            self._behavior.react_dist_ped = get_speed(self._vehicle)/3.6 * self._behavior.ttc_ped

        if 'wz' in param_dict and self._behavior.ttc_wz != param_dict['wz']:
            self._behavior.ttc_wz = param_dict['wz']
            self._behavior.react_dist_wz = get_speed(self._vehicle)/3.6 * self._behavior.ttc_wz

        for key in ('angle_th', 'safety_distance', 'T', 'a', 'b', 's_0', 'lc_flag', 'cruise_flag'):
            if key in param_dict:
                setattr(self._behavior, key, param_dict[key])

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
        vehicle_list = [v for v in vehicle_list if dist(v) < 500 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance, _ = self._vehicle_obstacle_detected(vehicle_list, 500, up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance, _ = self._vehicle_obstacle_detected(vehicle_list, 500, up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance, _ = self._vehicle_obstacle_detected(vehicle_list, 500, up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def lc_object_avoid_manager(self):
        """
        Gap acceptance for the lane change of the work zone scenario.

            :return: True if the gap in the target lane is acceptable, False otherwise
        """
        if self.lc_direct == -1:
            target_lane_id = self.ego_laneid + 1
        elif self.lc_direct == 1:
            target_lane_id = self.ego_laneid - 1
        else:
            print("Invalid lane change direction!")
            return

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        ego_location = self._vehicle.get_location()
        ego_velocity = abs(self._vehicle.get_velocity().x) # ego speed
        ego_s = ego_location.x
        ego_half_length = self._vehicle.bounding_box.extent.x  # half length of the ego vehicle

        filtered_vehicles = []
        for v in vehicle_list:
            if v.id == self._vehicle.id:
                continue  # skip the ego vehicle

            v_loc = v.get_location()
            v_wpt = self._map.get_waypoint(v_loc)
            if v_wpt.lane_id == target_lane_id or v_wpt.lane_id == target_lane_id+1:
                v_half_length = v.bounding_box.extent.x  # half length of the vehicle
                filtered_vehicles.append((v, v_loc.x, abs(v.get_velocity().x), v_half_length))

        # sort by the longitudinal position s (x coordinate)
        filtered_vehicles.sort(key=lambda x: x[1])

        front_vehicle = None
        rear_vehicle = None
        min_front_distance = float('inf')
        min_rear_distance = float('inf')

        for v, s, speed, half_length in filtered_vehicles:
            distance = abs(s - ego_s)
            if s < ego_s and distance < min_front_distance:
                front_vehicle = v  # closest vehicle ahead of the ego vehicle (driving towards -x)
                front_vehicle_speed = speed
                front_half_length = half_length
                min_front_distance = distance
            elif s > ego_s and distance < min_rear_distance:
                rear_vehicle = v  # closest vehicle behind the ego vehicle
                rear_vehicle_speed = speed
                rear_half_length = half_length
                min_rear_distance = distance

        # gap between the front and the rear vehicle
        if front_vehicle and rear_vehicle:
            d_gap = front_vehicle.get_location().distance(rear_vehicle.get_location()) - (rear_half_length + front_half_length)
        else:
            d_gap = float("inf")  # no front or no rear vehicle: the gap is large enough

        min_safe_gap = ego_half_length * 2 + self._behavior.safety_distance*2  # ego length + safety distance to both vehicles

        if d_gap < min_safe_gap:
            return False

        def compute_ttc(d_rel, v_rel):
            """Time to collision; infinite if the relative speed does not close the gap."""
            if v_rel > 0:  # the gap is closing
                return d_rel / v_rel
            else:
                return float("inf")  # v_rel <= 0: no collision

        d_front = float("inf")
        if front_vehicle:
            d_front = abs(front_vehicle.get_location().x - ego_location.x) - front_half_length - ego_half_length
            v_rel_front = ego_velocity - front_vehicle_speed
            ttc_front = compute_ttc(d_front, v_rel_front)
            print(f"Front vehicle d_front: {d_front:.2f}m")
        else:
            ttc_front = float("inf")

        if rear_vehicle:
            d_rear = abs(ego_location.x - rear_vehicle.get_location().x) - rear_half_length - ego_half_length
            v_rel_rear = rear_vehicle_speed - ego_velocity
            ttc_rear = compute_ttc(d_rear, v_rel_rear)
        else:
            ttc_rear = float("inf")

        if d_front < self._behavior.safety_distance:
            return False  # risk of hitting the front vehicle

        if rear_vehicle:
            if d_rear < self._behavior.safety_distance:
                return False  # risk of being hit by the rear vehicle

        print("Lane change is safe, changing lanes")
        return True

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
        walker_list = [w for w in walker_list if dist(w) < 500]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance, _ = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance, _ = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            if self.walker_laneid is not None and self.walker_laneid <= self.ego_laneid:
                walker_state, walker, distance, laneid = self._vehicle_obstacle_detected(walker_list,500, up_angle_th=self._behavior.angle_th)
            else:
                walker_state, walker, distance, laneid = self._vehicle_obstacle_detected(walker_list,500, up_angle_th=80)

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
            :return workzone_state: True if the work zone closes the ego lane, False if not
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

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information() # Update next waypoint, direction (RoadOption of waypoint), and speed limit/target.

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior (traffic signal interaction)
        if self._scenario == TRAFFIC_LIGHT:
            affected, traffic_light = self.traffic_light_manager()
            if affected:
                self.tf_flag = 1
                return self.decel_stop_tf(traffic_light)
            elif self.tf_flag == 0:
                # approach: cruise at the target speed
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
            elif self.tf_flag == 1:
                # restart after the light turned green: IDM acceleration
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control1 = self._local_planner.run_step(debug=debug)
                control = self.car_following_IDM(None, 10000)
                control.steer = control1.steer

        # 2.1: Pedestrian avoidance behaviors (pedestrian interaction)
        elif self._scenario == PEDESTRIAN:
            walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

            if walker_state:
                self.ped_flag = 1
                distance = abs(walker.get_location().x - ego_vehicle_loc.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

                if distance < self._behavior.react_dist_ped:
                    print("Distance: ", distance)
                    return self.decel_stop_ped(distance, self._behavior.s_0)
                else:
                    target_speed = min([self._behavior.max_speed, self._speed_limit])
                    self._local_planner.set_speed(target_speed)
                    control = self._local_planner.run_step(debug=debug)

            elif self.ped_flag == 1:
                # restart after the pedestrian has cleared the path: IDM acceleration
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control1 = self._local_planner.run_step(debug=debug)
                control = self.car_following_IDM(None, 10000)
                control.steer = control1.steer

            elif self.ped_flag == 0:
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)

        # 2.2: Work zone avoidance behaviors (work zone navigation)
        elif self._scenario == WORK_ZONE:
            workzone_state, workzone, w_distance = self.workzone_avoid_manager(ego_vehicle_wp)

            if workzone_state:
                if w_distance < self._behavior.react_dist_wz:
                    lc_safe = self.lc_object_avoid_manager()
                    if lc_safe:
                        return self.lane_change_control(ego_vehicle_wp, self.lc_direct)
                    else:
                        # wait for an acceptable gap: approach the work zone with the IDM
                        control = self.car_following_IDM(None,  w_distance)
                        control.steer = 0.0
                        return control
                else:
                    target_speed = min([self._behavior.max_speed, self._speed_limit])
                    self._local_planner.set_speed(target_speed)
                    control = self._local_planner.run_step(debug=debug)

            else:
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)

        # 2.3: Car following behaviors (car following)
        elif self._scenario == CAR_FOLLOWING:
            if self._behavior.cruise_flag == 0.0 and self._behavior.lc_flag == 0.0:
                if self.front_vehicle:
                    # longitudinal (x) distance between the centers of the two vehicles
                    distance = abs(self.front_vehicle.get_location().x - ego_vehicle_loc.x)
                    print(f"Distance to front vehicle: {distance}")
                    control1 = self.car_following_manager(self.front_vehicle, distance)
                    control = self.car_following_IDM(self.front_vehicle, distance)
                    control.steer = control1.steer
                else:
                    target_speed = min([self._behavior.max_speed, self._speed_limit])
                    self._local_planner.set_speed(target_speed)
                    control = self._local_planner.run_step(debug=debug)

            elif self._behavior.lc_flag == 1.0 and self.lc_destin_flag == 0:
                control = self.lane_change_control(ego_vehicle_wp, self.lc_direct)
            elif self._behavior.lc_flag == 1.0 and self.lc_destin_flag == 1:
                # lane change completed: cruise in the new lane
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)

            elif self._behavior.cruise_flag == 1.0:
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)

        # Normal behavior (also used if the flags hold values other than 0 or 1)
        if control is None:
            target_speed = min([self._behavior.max_speed, self._speed_limit])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def lane_change_control(self, waypoint, direction):
        """
        This method implements a lane change behavior.

            :param waypoint: current waypoint of the agent
            :param direction: direction to change the lane. 1 for right, -1 for left
            :return control: carla.VehicleControl
        """
        if self.lc_destin_flag == 0:
            front_wp = waypoint.next(12.0)[0]

            if direction == -1:
                left_wpt = front_wp.get_left_lane()
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
        control.brake = 0.01
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def IDM(self, vel_car, rel_dis, rel_vel, v_0, T, a, b, s_0):
        if v_0 == 0.0:
            v_0 = 1e-3

        Term_int = s_0 + max(0, vel_car * T + (vel_car * rel_vel / (2 * math.sqrt(a * b))))

        acc = a * (1 - ((vel_car / v_0) ** 4) - (Term_int / max(rel_dis, s_0)) ** 2)

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
        distance = abs(trigger_wp.transform.location.x - ego_vehicle_location.x)
        print(f"Distance to traffic light: {distance}")
        if distance > 1.8:
            ego_speed = get_speed(self._vehicle)/3.6    # m/s
            acc = self.IDM(ego_speed, distance, ego_speed,
                            v_0=self._behavior.max_speed/3.6, T=1.5,
                            a=0.75, b=1.67, s_0=1.0)

            if acc >= 0:
                throttle = acc
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
                throttle = acc
                brake = 0.0
            else:
                throttle = 0.0
                brake = abs(acc)/8.0 # tesla model 3 max deceleration is 8 m/s^2

        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False

        return control

    def car_following_IDM(self, vehicle, distance):
        """
        This method implements a car-following behavior based on the IDM model.

            :param vehicle: carla.Actor object
            :param distance: distance to the vehicle
            :return control: carla.VehicleControl
        """
        ego_speed = get_speed(self._vehicle)/3.6    # m/s

        if vehicle is not None:
            vel_car = get_speed(vehicle)/3.6    # m/s
            rel_vel = ego_speed - vel_car
        else:
            rel_vel = ego_speed

        target_speed = self._behavior.max_speed

        acc = self.IDM(ego_speed, distance, rel_vel,
                        v_0=target_speed/3.6, T=self._behavior.T,
                        a=self._behavior.a, b=self._behavior.b, s_0=self._behavior.s_0)

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
