#!/usr/bin/env python3
"""Control node: drives the ego vehicle with the host AV stack (CARLA BehaviorAgent).

The BehaviorAgent (carla_navigation/behavior_agent.py) plans the route to the destination
and executes the driving behavior of the scenario (stop at a red light, stop for a
pedestrian, work zone lane change, car following) with PID and IDM controllers.
The behavior parameters decided by the coordinator (decision_node) are received on
/algorithm_params_topic (std_msgs/Float32MultiArray); they can also be published manually.
The order of the values for each scenario is PARAMETER_LAYOUT in
car_following/scenario_config.py; NaN entries leave a parameter unchanged.
"""

import math
import time

import carla
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from carla_navigation.behavior_agent import BehaviorAgent
from car_following.scenario_config import (CAR_FOLLOWING, DESTINATION_SPAWN_INDEX,
                                           DRIVING_PREFERENCE, PARAMETER_LAYOUT, get_scenario)


class ControlNode(Node):
    def __init__(self):
        super(ControlNode, self).__init__('control_node')
        self.scenario = get_scenario(self)
        self.get_logger().info(f"Scenario: {self.scenario}")

        # Carla connection
        self.carla_client = carla.Client('localhost', 2000)
        self.carla_client.set_timeout(60.0)
        self.world = self.carla_client.get_world()
        self.spawn_points = self.world.get_map().get_spawn_points()

        # set traffic manager
        self.traffic_manager = self.carla_client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)

        # parameters
        self.sim_dt = 0.05
        self.role_name = 'hero'
        self.front_vehicle_type = 'vehicle.audi.a2'  # lead vehicle of the car-following scenario

        # Get vehicle information (the car-following scenario also waits for the lead vehicle)
        self.ego_vehicle = None
        self.front_vehicle = None
        self.get_vehicles_info()

        # assign the planner and controller of the host AV stack to the ego vehicle
        self.set_agent()

        # Create subscriber for the behavior parameters
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'algorithm_params_topic',  # Topic name
            self.decision_result_callback,
            10)

    def decision_result_callback(self, msg):
        """
        Callback function to handle the received behavior parameters.
        """
        names = PARAMETER_LAYOUT[self.scenario]
        if len(msg.data) > len(names):
            self.get_logger().warn(
                f"Received {len(msg.data)} values, but the {self.scenario} layout has "
                f"{len(names)} ({', '.join(names)}); the extra values are ignored.")
        param_dict = {name: float(value) for name, value in zip(names, msg.data)
                      if not math.isnan(value)}
        if not param_dict:
            return

        self.agent.update_driving_preference(param_dict)
        self.get_logger().info(f"Received parameters: {param_dict}")

    def get_vehicles_info(self):
        # Ego vehicle and front vehicle
        wait_for_front_vehicle = self.scenario == CAR_FOLLOWING
        while self.ego_vehicle is None or (wait_for_front_vehicle and self.front_vehicle is None):
            self.vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in self.vehicles:
                if self.ego_vehicle is None and vehicle.attributes.get('role_name') == self.role_name:
                    self.ego_vehicle = vehicle
                    self.get_logger().info(f"Get the ego vehicle {self.ego_vehicle.attributes['role_name']}")

                if self.front_vehicle is None and vehicle.type_id == self.front_vehicle_type:
                    self.front_vehicle = vehicle
                    self.get_logger().info(f"Get the front vehicle {self.front_vehicle.type_id}")
            time.sleep(0.1)

    def set_agent(self):
        self.agent = BehaviorAgent(self.ego_vehicle, self.front_vehicle, scenario=self.scenario,
                                   preference=DRIVING_PREFERENCE[self.scenario])
        self.get_logger().info("speed limit: {}".format(self.agent._speed_limit))
        self.destination = self.spawn_points[DESTINATION_SPAWN_INDEX].location
        self.agent.destination = self.destination
        self.agent.set_destination(self.destination)

    def control_loop(self):
        control = self.agent.run_step(debug=False)
        control.manual_gear_shift = False
        self.ego_vehicle.apply_control(control)


def main(args=None):
    rclpy.init(args=args)
    control_node = ControlNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(control_node, timeout_sec=0.03)  # process the ROS 2 callbacks
            control_node.control_loop()  # one control step

            if control_node.agent.done():
                control_node.get_logger().info("The target has been reached, stopping the simulation")
                break

    except KeyboardInterrupt:
        pass
    finally:
        control_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
