#!/usr/bin/env python3
"""Record node: logs the trajectories of all vehicles and pedestrians.

One row is written per simulation step (carla_ros_bridge publishes the ego speedometer
once per tick). One CSV file per actor type is written to the directory given by the
parameter ``output_dir`` (default: ~/pace_ads_output/<scenario>_<date>_<time>), e.g.
vehicle_tesla_model3.csv for the ego vehicle.

Columns: sim_time [s], type_id, role_name, x [m], y [m], vx [m/s], vy [m/s], speed_kmh
(CARLA world coordinates).
"""

import csv
import math
import os
import time
from datetime import datetime

import carla
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from car_following.scenario_config import get_scenario

HEADER = ['sim_time', 'type_id', 'role_name', 'x', 'y', 'vx', 'vy', 'speed_kmh']


class RecordNode(Node):
    def __init__(self):
        super(RecordNode, self).__init__('record_node')
        self.scenario = get_scenario(self)
        default_dir = os.path.join(os.path.expanduser('~'), 'pace_ads_output',
                                   f"{self.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir = os.path.expanduser(self.declare_parameter('output_dir', default_dir).value)
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(f"Writing trajectories to {self.output_dir}")

        # Carla connection
        self.carla_client = carla.Client('localhost', 2000)
        self.carla_client.set_timeout(60.0)
        self.world = self.carla_client.get_world()

        # Get vehicle information
        self.role_name = 'hero'
        self.ego_vehicle = None
        self.get_vehicles_info()

        self.subscription = self.create_subscription(
            Float32,
            '/carla/hero/speedometer',  # Topic name
            self.result_callback,
            10)

    def result_callback(self, msg):
        """
        Write one row per vehicle and pedestrian for the current simulation step.
        """
        sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
        actors = self.world.get_actors()
        for actor in list(actors.filter('vehicle.*')) + list(actors.filter('walker.pedestrian.*')):
            location = actor.get_location()
            velocity = actor.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

            csv_file = os.path.join(self.output_dir, actor.type_id.replace('.', '_') + '.csv')
            new_file = not os.path.exists(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(HEADER)
                writer.writerow([sim_time, actor.type_id, actor.attributes.get('role_name', ''),
                                 location.x, location.y, velocity.x, velocity.y, speed])

    def get_vehicles_info(self):
        # Ego vehicle
        while not self.ego_vehicle:
            self.vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in self.vehicles:
                if vehicle.attributes.get('role_name') == self.role_name:
                    self.ego_vehicle = vehicle
                    self.get_logger().info(f"Get the ego vehicle {self.ego_vehicle.attributes['role_name']}")
                    break

            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    record_node = RecordNode()

    try:
        rclpy.spin(record_node)
    except KeyboardInterrupt:
        pass
    finally:
        record_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
