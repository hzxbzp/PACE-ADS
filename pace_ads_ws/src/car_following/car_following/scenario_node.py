#!/usr/bin/env python3
"""Scenario node: builds the personalization test scenarios of PACE-ADS in CARLA Town06.

Select the scenario with the ROS parameter ``scenario``:

    traffic_light  All traffic lights are red and frozen at the start. After the ego vehicle
                   has stopped at the stop line, switch them to green with
                   ros2 topic pub --once /traffic_light_control std_msgs/msg/String "data: 'green'"
    pedestrian     A pedestrian stands at the side of the road ahead of the ego vehicle.
                   After the ego vehicle has stopped, start the crossing with
                   ros2 topic pub --once /ped_control std_msgs/msg/String "data: 'go'"
    work_zone      A work zone closes the ego lane; vehicles drive at 40 km/h in the adjacent
                   left lane (gaps set by the parameter ``traffic_gaps``).
    car_following  A lead vehicle drives ahead of the ego vehicle with the speed profile
                   40 km/h (40 s) -> 60 km/h (30 s) -> 30 km/h.

The ego vehicle (role name 'hero') and its sensors are spawned by carla_ros_bridge
(launch/carla_ros_bridge.launch.py with config/objects_<scenario>.json).
"""

import math
import time

import carla
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile
from ros_compatibility.node import CompatibleNode
from std_msgs.msg import String

from carla_navigation.misc import get_speed
from carla_navigation.traffic_agent import TrafficAgent
from car_following.scenario_config import (CAR_FOLLOWING, DESTINATION_SPAWN_INDEX, PEDESTRIAN,
                                           TOWN, TRAFFIC_GAPS, TRAFFIC_LIGHT, WORK_ZONE,
                                           get_scenario)


class ScenarioNode(CompatibleNode):
    def __init__(self):
        super(ScenarioNode, self).__init__('scenario_node')
        self.scenario = get_scenario(self)
        self.get_logger().info(f"Scenario: {self.scenario}")

        # Carla connection. Town06 is loaded by carla_ros_bridge, which also spawns the ego
        # vehicle. The world is not reloaded here, because that would remove the ego vehicle.
        self.carla_client = carla.Client('localhost', 2000)
        self.carla_client.set_timeout(60.0)
        self.role_name = 'hero'
        self.wait_for_ego_vehicle()
        self.world = self.carla_client.get_world()
        map_name = self.world.get_map().name
        if TOWN not in map_name:
            self.get_logger().warn(
                f"The current map is '{map_name}', but the scenarios are defined for {TOWN}.")
        self.spawn_points = self.world.get_map().get_spawn_points()
        # self.draw_spawning_points_with_index()

        # parameters
        self.sim_dt = 0.05
        self.spawned_actors = []

        # set traffic scenario
        self.traffics = []
        self.traffic_agents = []
        self.distances = []
        self.pedestrian = None
        self.phase = "accelerate_to_40"
        self.phase_start_time = None

        if self.scenario == TRAFFIC_LIGHT:
            # red light ahead of the ego vehicle
            self.set_all_traffic_lights(carla.TrafficLightState.Red)
        else:
            self.set_traffic_light()

        if self.scenario == PEDESTRIAN:
            self.set_pedestrian()
        elif self.scenario in (WORK_ZONE, CAR_FOLLOWING):
            self.distances = [float(d) for d in
                              self.get_param('traffic_gaps', TRAFFIC_GAPS[self.scenario])]
            if self.scenario == WORK_ZONE:
                self.generate_construction_zone()
            self.set_traffic_flow()
            self.set_traffic_agent()

        self.get_logger().info(f"Scenario '{self.scenario}' is ready.")

        # ROS2 subscription for traffic light state
        self.subscription = self.create_subscription(
            String,
            'traffic_light_control',  # Topic to receive commands
            self.traffic_light_callback,
            10
        )

        # ROS2 subscription for starting the pedestrian crossing
        self.subscription2 = self.create_subscription(
            String,
            'ped_control',  # Topic to receive commands
            self.pedestrain_callback,
            10
        )

        # Latched message telling that the scenario is ready (used by run_scenario.sh)
        self.ready_publisher = self.create_publisher(
            String, 'scenario_ready', QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.ready_publisher.publish(String(data=self.scenario))

    def wait_for_ego_vehicle(self):
        """Wait until carla_ros_bridge has spawned the ego vehicle."""
        while True:
            world = self.carla_client.get_world()
            for vehicle in world.get_actors().filter('vehicle.*'):
                if vehicle.attributes.get('role_name') == self.role_name:
                    self.get_logger().info("Found the ego vehicle.")
                    return
            self.get_logger().info("Waiting for the ego vehicle spawned by carla_ros_bridge ...",
                                   throttle_duration_sec=5.0)
            time.sleep(0.5)

    def pedestrain_callback(self, msg):
        state = msg.data.strip().lower()
        if state == "go":
            self.pedestrain_crossing()

    def traffic_light_callback(self, msg):
        state = msg.data.strip().lower()
        if state not in ["green", "yellow", "red"]:
            self.get_logger().warn(f"Invalid traffic light state: {state}")
            return
        self.get_logger().info(f"Received traffic light state: {state}")

        # Update traffic lights in CARLA
        state_mapping = {
            "green": carla.TrafficLightState.Green,
            "yellow": carla.TrafficLightState.Yellow,
            "red": carla.TrafficLightState.Red
        }
        self.set_all_traffic_lights(state_mapping[state])
        self.get_logger().info(f"All traffic lights updated to {state.upper()}.")

    def set_all_traffic_lights(self, state):
        """Set all traffic lights of the map to ``state`` and freeze them."""
        self.world.wait_for_tick()
        self.all_traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for traffic_light in self.all_traffic_lights:
            traffic_light.set_state(state)
            traffic_light.freeze(True)

    def set_traffic_light(self):
        # Set traffic lights to green (scenarios without traffic light interaction)
        self.set_all_traffic_lights(carla.TrafficLightState.Green)
        self.get_logger().info("All traffic lights have been set to Green permanently.")

    def show_waypoints(self):
        """Debug helper: draw the indices of the 100 waypoints ahead of the ego vehicle."""
        ego_vehicle = None
        while not ego_vehicle:
            for vehicle in self.world.get_actors().filter('vehicle.*'):
                if vehicle.attributes['role_name'] == self.role_name:
                    ego_vehicle = vehicle
            time.sleep(0.1)
        start_location = ego_vehicle.get_location()
        self.world.debug.draw_string(start_location, "0", draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=1000.0,
                                    persistent_lines=True)
        for i in range(0, 100):
            front_waypoint = self.world.get_map().get_waypoint(start_location).next(2.0)[0]
            self.world.debug.draw_string(front_waypoint.transform.location, str(i), draw_shadow=False,
                                    color=carla.Color(r=255, g=0, b=0), life_time=1000.0,
                                    persistent_lines=True)
            start_location = front_waypoint.transform.location

    def draw_spawning_points_with_index(self):
        """Debug helper: draw the index of every spawn point of the map."""
        for i, spawn_point in enumerate(self.spawn_points):
            self.world.debug.draw_string(
                spawn_point.location, str(i), draw_shadow=False,
                color=carla.Color(r=255, g=0, b=0), life_time=1000.0,
                persistent_lines=True
            )

    def set_pedestrian(self):
        # Pedestrian next to the road at spawn point 48
        spawn_point = self.spawn_points[48]
        self.ped_start_location = spawn_point.location
        self.ped_start_location.y -= 7  # keep the pedestrian off the lane
        # the pedestrian crosses 15 m in -y
        self.ped_end_location = carla.Location(x=self.ped_start_location.x, y=self.ped_start_location.y - 15, z=self.ped_start_location.z)
        self.ped_target_location = self.ped_end_location

        # Spawn the pedestrian
        pedestrian_blueprint = self.world.get_blueprint_library().find('walker.pedestrian.0001')
        pedestrian_blueprint.set_attribute('role_name', 'ped')

        pedestrian_transform = carla.Transform(self.ped_start_location, carla.Rotation(yaw=-90))
        self.pedestrian = self.world.try_spawn_actor(pedestrian_blueprint, pedestrian_transform)

        self.get_logger().info("Pedestrian spawned at spawn point 48, the coordinate is: ({:.2f}, {:.2f}, {:.2f}).".format(
            self.ped_start_location.x, self.ped_start_location.y, self.ped_start_location.z
        ))
        if not self.pedestrian:
            self.get_logger().error("Failed to spawn pedestrian at spawn point 48.")
            return
        self.spawned_actors.append(self.pedestrian)

        # Walker control used for the crossing
        self.pedestrian_control = carla.WalkerControl()

        # Move the spectator camera to the pedestrian
        spectator = self.world.get_spectator()
        spectator_transform = carla.Transform(
            carla.Location(self.ped_start_location.x-30,
                        self.ped_start_location.y,
                        self.ped_start_location.z + 20.0),  # raise the viewpoint
            carla.Rotation(pitch=-15, yaw=0, roll=0)  # look slightly down
        )
        spectator.set_transform(spectator_transform)

    def pedestrain_crossing(self):
        if self.pedestrian is None:
            self.get_logger().error("There is no pedestrian in this scenario.")
            return
        self.ped_location = self.pedestrian.get_location()

        # Distance between the pedestrian and its start point
        distance_to_start = math.sqrt(
            (self.ped_location.x - self.ped_start_location.x) ** 2 +
            (self.ped_location.y - self.ped_start_location.y) ** 2 +
            (self.ped_location.z - self.ped_start_location.z) ** 2
        )

        if distance_to_start > 30:
            # world didn't tick
            self.get_logger().error("wait for world tick")
            return

        distance_to_end = math.sqrt(
            (self.ped_location.x - self.ped_end_location.x) ** 2 +
            (self.ped_location.y - self.ped_end_location.y) ** 2 +
            (self.ped_location.z - self.ped_end_location.z) ** 2
        )

        # Close to the end point: walk back to the start point
        if distance_to_end < 0.7:
            self.ped_target_location = self.ped_start_location
        # Close to the start point: walk to the end point
        elif distance_to_start < 0.7:
            self.ped_target_location = self.ped_end_location

        # Walking direction
        walk_direction = carla.Vector3D(
            x=self.ped_target_location.x - self.ped_location.x,
            y=self.ped_target_location.y - self.ped_location.y,
            z=self.ped_target_location.z - self.ped_location.z
        )
        norm = math.sqrt(walk_direction.x**2 + walk_direction.y**2 + walk_direction.z**2)
        if norm != 0:
            walk_direction.x /= norm
            walk_direction.y /= norm
            walk_direction.z /= norm

        # Walker control
        self.pedestrian_control.speed = 3.0  # walking speed (m/s)
        self.pedestrian_control.direction = walk_direction
        self.pedestrian_control.jump = False

        self.pedestrian.apply_control(self.pedestrian_control)

    def _track(self, actor):
        """Remember an actor spawned by this node so that it is removed at shutdown."""
        if actor is not None:
            self.spawned_actors.append(actor)
        return actor

    def generate_construction_zone(self):
        # Step 1: Define the blueprint for traffic cones, barriers, and construction signs
        cone_blueprint = self.world.get_blueprint_library().find('static.prop.trafficcone01')
        barrier_blueprint = self.world.get_blueprint_library().find('static.prop.streetbarrier')
        sign_blueprint = self.world.get_blueprint_library().find('static.prop.warningconstruction')

        # Step 2: Choose a central location for the construction zone
        center_point = self.spawn_points[297].location

        # Step 3: Define dimensions and spacing for the construction zone
        zone_length = 15.0  # Length of the construction zone in meters
        zone_width = 6.0    # Width of the construction zone in meters
        cone_spacing = 2.0  # Spacing between cones

        # Step 4: Create traffic cones around the construction zone perimeter
        for x_offset in range(-int(zone_length/2), int(zone_length/2) + 1, int(cone_spacing)):
            # Modify the y_offset to create an asymmetric placement
            near_side_offset = zone_width - 5  # Closer to the center on one side
            far_side_offset = 5  # Further away from the center on the other side

            for y_offset in [-near_side_offset, far_side_offset]:  # One side closer, one side further
                cone_location = carla.Location(
                    x=center_point.x + x_offset,
                    y=center_point.y + y_offset,
                    z=center_point.z
                )
                self.cone_actor = self._track(self.world.try_spawn_actor(cone_blueprint, carla.Transform(cone_location)))

        # Step 5: Place barriers at both ends of the construction zone
        barrier_rotation = carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
        barrier1_location = carla.Location(x=center_point.x+zone_length/2, y=center_point.y+zone_width-5, z=center_point.z )
        self.barrier_actor1 = self._track(self.world.try_spawn_actor(barrier_blueprint, carla.Transform(barrier1_location, barrier_rotation)))
        barrier2_location = carla.Location(x=center_point.x+zone_length/2, y=center_point.y+zone_width-3, z=center_point.z )
        self.barrier_actor2 = self._track(self.world.try_spawn_actor(barrier_blueprint, carla.Transform(barrier2_location, barrier_rotation)))
        barrier3_location = carla.Location(x=center_point.x-zone_length/2, y=center_point.y+zone_width-5, z=center_point.z )
        self.barrier_actor3 = self._track(self.world.try_spawn_actor(barrier_blueprint, carla.Transform(barrier3_location, barrier_rotation)))
        barrier4_location = carla.Location(x=center_point.x-zone_length/2, y=center_point.y+zone_width-3, z=center_point.z )
        self.barrier_actor4 = self._track(self.world.try_spawn_actor(barrier_blueprint, carla.Transform(barrier4_location, barrier_rotation)))

        # Step 6: Add a construction sign at the entrance of the zone
        sign_rotation = carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)
        sign1_location = carla.Location(
            x=center_point.x + zone_length/2 + 8.0,  # Place the sign a bit before the zone starts
            y=center_point.y + zone_width - 5,
            z=center_point.z
        )
        self.sign1_actor = self._track(self.world.try_spawn_actor(sign_blueprint, carla.Transform(sign1_location, sign_rotation)))
        sign2_location = carla.Location(
            x=center_point.x + zone_length/2 + 8.0,  # Place the sign a bit before the zone starts
            y=center_point.y + zone_width - 3,
            z=center_point.z
        )
        self.sign2_actor = self._track(self.world.try_spawn_actor(sign_blueprint, carla.Transform(sign2_location, sign_rotation)))
        self.get_logger().info("Work zone spawned around spawn point 297.")

    def set_traffic_flow(self):
        # Blueprint library
        blueprint_library = self.world.get_blueprint_library()

        # Vehicle models of the surrounding vehicles (in spawn order)
        sedan_list = [
            "vehicle.audi.a2",
            "vehicle.audi.tt",
            "vehicle.chevrolet.impala",
            "vehicle.nissan.micra",
            "vehicle.mini.cooper_s",
            "vehicle.bmw.grandtourer",
            "vehicle.mercedes.coupe"
        ]
        vehicle_bp_all = [blueprint_library.find(car) for car in sedan_list]

        base_location = self.spawn_points[433].location
        base_rotation = self.spawn_points[433].rotation

        previous_distance = 0  # longitudinal position of the previous vehicle
        for i, distance_offset in enumerate(self.distances):
            previous_distance += distance_offset

            # Spawn location of this vehicle (offset along x from spawn point 433)
            spawn_location = carla.Location(
                x=base_location.x + previous_distance,
                y=base_location.y,
                z=base_location.z
            )

            transform = carla.Transform(spawn_location, base_rotation)

            vehicle_bp = vehicle_bp_all[i % len(vehicle_bp_all)]

            # Try to spawn the vehicle
            vehicle = self.world.try_spawn_actor(vehicle_bp, transform)

            if vehicle:
                self.traffics.append(vehicle)
                self.spawned_actors.append(vehicle)
                print(f"Spawned vehicle {i+1}: gap to the previous vehicle {distance_offset} m")

        print("All vehicles have been spawned")

    def set_traffic_agent(self):
        self.destination = self.spawn_points[DESTINATION_SPAWN_INDEX].location
        self.world.wait_for_tick()  # the new vehicles have valid locations after a tick

        for i in range(0, len(self.traffics)):
            if i == 0:
                traffic_agent = TrafficAgent(self.traffics[i], None, self.distances[i])
            else:
                traffic_agent = TrafficAgent(self.traffics[i], self.traffics[i-1], self.distances[i])
            traffic_agent.destination = self.destination
            traffic_agent.set_destination(self.destination)
            self.traffic_agents.append(traffic_agent)

        if self.traffic_agents:
            self.leading_veh = self.traffic_agents[0]._vehicle

    def control_traffic(self):
        """Apply one control step to the surrounding vehicles."""
        if self.scenario == CAR_FOLLOWING:
            self.control_lead_vehicle()
            return

        # Work zone: the first vehicle cruises at 40 km/h, the others follow their predecessor
        for i in range(0, len(self.traffic_agents)):
            control = self.traffic_agents[i].run_step(debug=False)
            control.manual_gear_shift = False
            self.traffics[i].apply_control(control)

    def control_lead_vehicle(self):
        """Speed profile of the lead vehicle: 40 km/h (40 s) -> 60 km/h (30 s) -> 30 km/h."""
        speed = get_speed(self.leading_veh)

        if self.phase == "accelerate_to_40":
            target_speed = 40
            self.phase_start_time = time.time()
            if speed >= target_speed-1:
                self.phase = "hold_40"
                self.phase_start_time = time.time()

        elif self.phase == "hold_40":
            target_speed = 40
            if time.time() - self.phase_start_time >= 40:
                self.phase = "accelerate_to_60"

        elif self.phase == "accelerate_to_60":
            target_speed = 60
            if speed >= target_speed-1:
                self.phase = "hold_60"
                self.phase_start_time = time.time()

        elif self.phase == "hold_60":
            target_speed = 60
            if time.time() - self.phase_start_time >= 30:
                self.phase = "decelerate_to_30"

        elif self.phase == "decelerate_to_30":
            target_speed = 30

        control = self.traffic_agents[0].run_step(target_speed=target_speed, debug=False)
        control.manual_gear_shift = False
        self.traffics[0].apply_control(control)

    def destroy_actors(self):
        """Remove the actors spawned by this node."""
        for actor in self.spawned_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
            except RuntimeError:
                pass
        self.spawned_actors = []


def main(args=None):
    rclpy.init(args=args)
    scenario_node = ScenarioNode()
    try:
        if scenario_node.traffic_agents:
            # Control the surrounding vehicles once per simulation step
            while rclpy.ok():
                rclpy.spin_once(scenario_node, timeout_sec=0.0)
                scenario_node.world.wait_for_tick()
                scenario_node.control_traffic()
        else:
            rclpy.spin(scenario_node)
    except KeyboardInterrupt:
        pass
    finally:
        scenario_node.destroy_actors()
        scenario_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
