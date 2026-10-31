#!/usr/bin/env python3
"""Decision node: PACE-ADS driver agent and coordinator.

Driver agent: runs continuously. Every inference uses the latest image of the front camera and
the measurements of the perception module (simulator ground truth: ego state, nearby vehicles
and pedestrians, the traffic light of the ego lane and work-zone elements) and returns the
scene understanding Y.

Coordinator: invoked for every new occupant report (/occupant_report from
psychological_monitor_node; a state code can also be published manually on
/psychological_output). It receives Y, the occupant report R and the behavior library of the
scenario, and returns the behavior, the parameter values and an explanation N. The parameters
are clamped to the permitted ranges of the behavior library (safety envelope) and published on
/algorithm_params_topic in the order of scenario_config.PARAMETER_LAYOUT.

With the parameter output_dir, the inputs and outputs of the agents are appended to
<output_dir>/agent_log.jsonl.
"""

import json
import math
import os
import threading
import time
from datetime import datetime

import carla
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32, String

from carla_navigation.misc import get_speed
from car_following.agents_threads_runs import (coordinator_submit, create_threads, driver_submit,
                                               get_coordinator_response, get_driver_response,
                                               wait_on_run)
from car_following.behavior_library import BehaviorLibrary
from car_following.occupant_state import STATE_LABELS
from car_following.scenario_config import SPEED_LIMIT, get_scenario

WORK_ZONE_PROPS = ('static.prop.warningconstruction', 'static.prop.trafficcone01',
                   'static.prop.streetbarrier')

COLOR_NAMES = {
    'black': (20, 20, 20), 'white': (235, 235, 235), 'gray': (120, 120, 120),
    'silver': (185, 185, 185), 'red': (190, 30, 30), 'orange': (230, 120, 20),
    'yellow': (225, 200, 40), 'green': (40, 140, 60), 'blue': (30, 60, 180),
    'brown': (110, 70, 40),
}


def color_name(rgb_text):
    """Closest color name of a CARLA color attribute ('R,G,B'); empty if unknown."""
    try:
        rgb = [int(v) for v in rgb_text.split(',')]
    except ValueError:
        return ''
    if len(rgb) != 3:
        return ''
    return min(COLOR_NAMES, key=lambda name: sum((c - r) ** 2 for c, r in zip(COLOR_NAMES[name], rgb)))


def vehicle_model(type_id):
    """Readable model of a CARLA blueprint id, e.g. vehicle.audi.a2 -> audi a2."""
    return ' '.join(type_id.split('.')[1:]).replace('_', ' ')


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')
        self.scenario = get_scenario(self)
        self.output_dir = os.path.expanduser(self.declare_parameter('output_dir', '').value)
        self.perception_range = float(self.declare_parameter('perception_range', 120.0).value)
        self.get_logger().info(f'decision_node is running (scenario: {self.scenario})')

        # Carla connection
        self.carla_client = carla.Client('localhost', 2000)
        self.carla_client.set_timeout(60.0)
        self.world = self.carla_client.get_world()
        self.map = self.world.get_map()
        self.role_name = 'hero'
        self.ego_vehicle = None
        self.get_vehicles_info()

        # Agents (each agent keeps its own conversation) and behavior library of the scenario
        self.driver_thread, _, self.coordinator_thread = create_threads()
        self.library = BehaviorLibrary(self.scenario)

        # Data shared by the driver-agent thread and the coordinator thread
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.driver_output = None
        self.occupant_state = None
        self.previous_state = None
        self.occupant_instruction = 'None'
        self.pending_report = False
        self.bridge = CvBridge()
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # The driver agent and the coordinator run in separate callback groups, i.e. in parallel
        driver_group = MutuallyExclusiveCallbackGroup()
        coordinator_group = MutuallyExclusiveCallbackGroup()

        self.publisher_ = self.create_publisher(Float32MultiArray, 'algorithm_params_topic', 10)
        # queue depth 1: every driver-agent inference uses the latest camera image
        self.image_subscription = self.create_subscription(
            Image, '/carla/hero/rgb_front/image', self.image_callback, 1, callback_group=driver_group)
        self.report_subscription = self.create_subscription(
            String, 'occupant_report', self.occupant_report_callback, 10, callback_group=coordinator_group)
        self.state_subscription = self.create_subscription(
            Int32, 'psychological_output', self.psychological_callback, 10, callback_group=coordinator_group)
        self.coordinator_timer = self.create_timer(0.1, self.run_coordinator, callback_group=coordinator_group)

    def get_vehicles_info(self):
        # Ego vehicle
        while self.ego_vehicle is None:
            for vehicle in self.world.get_actors().filter('vehicle.*'):
                if vehicle.attributes.get('role_name') == self.role_name:
                    self.ego_vehicle = vehicle
                    self.get_logger().info(f"Get the ego vehicle {self.ego_vehicle.attributes['role_name']}")
                    break
            time.sleep(0.1)

    # -------------------------------------------------------------------------------------
    # Occupant reports (psychologist agent)
    # -------------------------------------------------------------------------------------

    def occupant_report_callback(self, msg):
        try:
            report = json.loads(msg.data)
            state = report['state']
        except (ValueError, KeyError, TypeError):
            self.get_logger().error(f"Invalid occupant report: {msg.data}")
            return
        if state not in STATE_LABELS.values():
            self.get_logger().error(f"Unknown occupant state in the report: {state}")
            return
        self.set_occupant_report(state, str(report.get('instruction') or 'None'))

    def psychological_callback(self, msg):
        state = STATE_LABELS.get(msg.data)
        if state is None:
            self.get_logger().error(f"Received unknown psychological state: {msg.data}")
            return
        self.set_occupant_report(state, 'None')

    def set_occupant_report(self, state, instruction):
        with self.lock:
            self.previous_state = self.occupant_state
            self.occupant_state = state
            if instruction.strip() and instruction.strip().lower() != 'none':
                self.occupant_instruction = instruction.strip()
            self.pending_report = True
        self.get_logger().info(f"Occupant report: state = {state}, instruction = {instruction}")

    # -------------------------------------------------------------------------------------
    # Driver agent
    # -------------------------------------------------------------------------------------

    def image_callback(self, front_img):
        """One driver-agent inference with the latest camera image and measurements."""
        try:
            cv_img_front = self.bridge.imgmsg_to_cv2(front_img, desired_encoding='bgr8')
            success, encoded_image = cv2.imencode('.png', cv_img_front)
            if not success:
                self.get_logger().error("Failed to encode front image")
                return
            perception = self.get_perception()
            start = time.time()
            run = driver_submit(self.driver_thread, encoded_image.tobytes(), json.dumps(perception, indent=1))
            run = wait_on_run(run, self.driver_thread)
            driver_output = get_driver_response(self.driver_thread)
        except Exception as e:
            self.get_logger().error(f"Driver agent failed: {e}")
            return

        if driver_output is None:
            self.get_logger().warn("Driver output is None. Skip this cycle.")
            return
        latency = time.time() - start
        with self.lock:
            self.driver_output = driver_output
        self.get_logger().info(f"Driver agent ({latency:.1f} s): {driver_output.get('scene_description', '')}")
        self.log_event('driver_agent', {'latency_s': latency, 'perception': perception, 'output': driver_output})

    def get_perception(self):
        """Measurements of the perception module (simulator ground truth) in the ego frame."""
        ego = self.ego_vehicle
        ego_transform = ego.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        ego_wp = self.map.get_waypoint(ego_location)
        ego_half_length = ego.bounding_box.extent.x

        def ego_frame(location):
            """(dx, dy): dx > 0 ahead of the ego vehicle, dy > 0 to its right."""
            wx = location.x - ego_location.x
            wy = location.y - ego_location.y
            return wx * forward.x + wy * forward.y, wx * right.x + wy * right.y

        def relative_lane(location):
            """0 = ego lane, -1 = left lane, +1 = right lane; None = other road or direction."""
            wp = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is None or wp.road_id != ego_wp.road_id or wp.lane_id * ego_wp.lane_id <= 0:
                return None
            return ego_wp.lane_id - wp.lane_id if ego_wp.lane_id < 0 else wp.lane_id - ego_wp.lane_id

        def position_text(dx, dy, lane):
            longitudinal = 'front' if dx >= 0 else 'rear'
            if lane is None:
                side = 'right' if dy > 1.75 else ('left' if dy < -1.75 else '')
            else:
                side = 'left' if lane < 0 else ('right' if lane > 0 else '')
            return f'{side} {longitudinal}'.strip()

        def is_driving_lane(wp):
            return wp is not None and wp.lane_type == carla.LaneType.Driving and wp.lane_id * ego_wp.lane_id > 0

        perception = {
            'ego': {
                'speed_kmh': round(get_speed(ego), 1),
                'lane': 0,
                'adjacent_left_lane_exists': is_driving_lane(ego_wp.get_left_lane()),
                'adjacent_right_lane_exists': is_driving_lane(ego_wp.get_right_lane()),
            },
            'traffic_light': None,
            'vehicles': [],
            'pedestrians': [],
            'work_zone': None,
        }
        actors = self.world.get_actors()

        for vehicle in actors.filter('vehicle.*'):
            location = vehicle.get_location()
            if vehicle.id == ego.id or location.distance(ego_location) > self.perception_range:
                continue
            dx, dy = ego_frame(location)
            lane = relative_lane(location)
            gap = max(0.0, abs(dx) - ego_half_length - vehicle.bounding_box.extent.x)
            description = f"{color_name(vehicle.attributes.get('color', ''))} {vehicle_model(vehicle.type_id)}".strip()
            perception['vehicles'].append({
                'id': f'vehicle_{vehicle.id}',
                'type': description,
                'relative_position': position_text(dx, dy, lane),
                'lane': lane,
                'gap_m': round(gap, 1),
                'dx_m': round(dx, 1),
                'dy_m': round(dy, 1),
                'speed_kmh': round(get_speed(vehicle), 1),
            })

        for walker in actors.filter('walker.pedestrian.*'):
            location = walker.get_location()
            if location.distance(ego_location) > self.perception_range:
                continue
            dx, dy = ego_frame(location)
            velocity = walker.get_velocity()
            perception['pedestrians'].append({
                'id': f'pedestrian_{walker.id}',
                'relative_position': position_text(dx, dy, None),
                'distance_m': round(math.hypot(dx, dy), 1),
                'dx_m': round(dx, 1),
                'dy_m': round(dy, 1),
                'speed_kmh': round(get_speed(walker), 1),
                'lateral_speed_mps': round(velocity.x * right.x + velocity.y * right.y, 2),  # > 0: moving to the right
            })

        # Traffic light that applies to the ego lane: nearest stop line ahead in the ego lane
        nearest = None
        for light in actors.filter('traffic.traffic_light'):
            for stop_wp in light.get_stop_waypoints():
                dx, dy = ego_frame(stop_wp.transform.location)
                heading = stop_wp.transform.get_forward_vector()
                aligned = heading.x * forward.x + heading.y * forward.y > 0.8
                if 0.0 < dx < 200.0 and abs(dy) < 2.0 and aligned and (nearest is None or dx < nearest[0]):
                    nearest = (dx, light)
        if nearest is not None:
            perception['traffic_light'] = {
                'state': str(nearest[1].state).lower(),
                'distance_to_stop_line_m': round(max(0.0, nearest[0] - ego_half_length), 1),
            }

        # Work zone: cones, barriers and construction signs ahead
        elements = []
        for prop in actors.filter('static.prop.*'):
            if prop.type_id not in WORK_ZONE_PROPS:
                continue
            location = prop.get_location()
            if location.distance(ego_location) > self.perception_range:
                continue
            dx, dy = ego_frame(location)
            if dx > 0.0:
                elements.append((dx, dy, relative_lane(location)))
        if elements:
            in_ego_lane = [e for e in elements if e[2] == 0]
            first = min(in_ego_lane or elements, key=lambda e: e[0])
            perception['work_zone'] = {
                'number_of_elements': len(elements),
                'lanes_occupied': sorted({e[2] for e in elements if e[2] is not None}),
                'closes_ego_lane': bool(in_ego_lane),
                'distance_m': round(max(0.0, first[0] - ego_half_length), 1),
            }
        return perception

    # -------------------------------------------------------------------------------------
    # Coordinator
    # -------------------------------------------------------------------------------------

    def run_coordinator(self):
        """Coordinator: called for every new occupant report (checked every 0.1 s)."""
        with self.lock:
            if not self.pending_report or self.driver_output is None:
                return
            self.pending_report = False
            driver_output = {key: value for key, value in self.driver_output.items() if key != 'reasoning'}
            occupant_report = {
                'state': self.occupant_state,
                'previous_state': self.previous_state or 'none',
                'instruction': self.occupant_instruction,
            }
            self.occupant_instruction = 'None'  # an instruction is handled once
        ego_info = {
            'speed_kmh': round(get_speed(self.ego_vehicle), 1),
            'speed_limit_kmh': SPEED_LIMIT,
            'current_behavior': self.library.behavior,
        }
        library_text = self.library.describe()

        start = time.time()
        try:
            run = coordinator_submit(self.coordinator_thread, self.scenario, driver_output,
                                     occupant_report, ego_info, library_text)
            run = wait_on_run(run, self.coordinator_thread)
            coordinator_output = get_coordinator_response(self.coordinator_thread)
        except Exception as e:
            self.get_logger().error(f"Coordinator failed: {e}")
            return
        if coordinator_output is None:
            return
        latency = time.time() - start

        values, notes = self.library.apply(coordinator_output.get('behavior'),
                                           coordinator_output.get('parameters') or {})
        msg = Float32MultiArray()
        msg.data = [float(v) for v in values]
        self.publisher_.publish(msg)

        self.get_logger().info(
            f"Coordinator ({latency:.1f} s): behavior = {coordinator_output.get('behavior')}, "
            f"parameters = {coordinator_output.get('parameters')}")
        for note in notes:
            self.get_logger().warn(note)
        self.get_logger().info(f"Explanation to the occupant: {coordinator_output.get('explanation')}")
        self.log_event('coordinator', {
            'latency_s': latency,
            'occupant_report': occupant_report,
            'ego': ego_info,
            'output': coordinator_output,
            'published': [None if math.isnan(v) else v for v in msg.data],
            'notes': notes,
        })

    def log_event(self, kind, data):
        """Append one record to <output_dir>/agent_log.jsonl."""
        if not self.output_dir:
            return
        record = {
            'kind': kind,
            'wall_time': datetime.now().isoformat(timespec='milliseconds'),
            'sim_time': self.world.get_snapshot().timestamp.elapsed_seconds,
        }
        record.update(data)
        with self.log_lock:
            with open(os.path.join(self.output_dir, 'agent_log.jsonl'), 'a') as f:
                f.write(json.dumps(record) + '\n')


def main(args=None):
    rclpy.init(args=args)
    decision_node = DecisionNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(decision_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        decision_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
