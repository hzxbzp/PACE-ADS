#!/usr/bin/env python3
"""Occupant monitor node (psychologist agent).

Type the name of an occupant state to inject the occupant signals of that state, which
reproduces the prescribed occupant-state sequence of an experiment run. A verbal command can
be added after '|', e.g.

    impatient
    very anxious | Please slow down.

Signals mode (parameter samples_dir set): the EEG, heart-rate and face-image sample of the typed
state (samples_dir/<state>/, see occupant_signals.py) and the verbal command are sent to the
psychologist agent; the state it infers and the instruction it extracts are published.
Direct mode (samples_dir not set): the typed state and the verbal command are published as they
are (useful for tests without the Emognition data or an OpenAI API key).

Output: /occupant_report (std_msgs/String with the JSON object
{"state", "instruction", "injected_state", "source"}). With the parameter output_dir, every
report is also appended to <output_dir>/psychologist_log.jsonl.
"""

import json
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from car_following.agents_threads_runs import (create_threads, get_psychologist_response,
                                               psychologist_submit, wait_on_run)
from car_following.occupant_signals import format_signals, load_sample, sample_folder
from car_following.occupant_state import STATE_INPUTS, STATE_LABELS, parse_state


class PsychologicalMonitorNode(Node):
    def __init__(self):
        super().__init__('psychological_monitor_node')
        self.samples_dir = os.path.expanduser(self.declare_parameter('samples_dir', '').value)
        self.output_dir = os.path.expanduser(self.declare_parameter('output_dir', '').value)
        self.publisher_ = self.create_publisher(String, 'occupant_report', 10)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        if self.samples_dir:
            _, self.psychologist_thread, _ = create_threads()
            self.get_logger().info(f'Signals mode: samples are read from {self.samples_dir}')
        else:
            self.psychologist_thread = None
            self.get_logger().info('Direct mode: the typed state is published without the psychologist agent')
        self.get_logger().info('Valid states: ' + ', '.join(STATE_INPUTS))

        self.timer = self.create_timer(1.0, self.listen_for_input)  # check the keyboard input every second

    def listen_for_input(self):
        user_input = input("Occupant state [| verbal command]: ").strip()
        if not user_input:
            return
        state_text, _, command = user_input.partition('|')
        code = parse_state(state_text)
        if code is None:
            self.get_logger().warn(
                f"Unknown state '{state_text.strip()}'. Valid states: {', '.join(STATE_INPUTS)}")
            return
        injected_state = STATE_LABELS[code]
        command = command.strip() or 'None'

        reasoning, sample_meta = None, None
        if self.psychologist_thread is None:
            report = {'state': injected_state, 'instruction': command,
                      'injected_state': injected_state, 'source': 'typed'}
        else:
            try:
                sample = load_sample(sample_folder(self.samples_dir, injected_state))
                run = psychologist_submit(self.psychologist_thread, format_signals(sample),
                                          sample['face'], sample['face_mime'], command)
                run = wait_on_run(run, self.psychologist_thread)
                output = get_psychologist_response(self.psychologist_thread)
            except Exception as e:
                self.get_logger().error(f"Psychologist agent failed: {e}")
                return
            if output is None:
                return
            report = {'state': output['state'], 'instruction': output['instruction'],
                      'injected_state': injected_state, 'source': 'psychologist agent'}
            reasoning, sample_meta = output['reasoning'], sample['meta']
            self.get_logger().info(f"Psychologist reasoning: {reasoning}")

        msg = String()
        msg.data = json.dumps(report)
        self.publisher_.publish(msg)
        self.get_logger().info(
            f"Injected: {injected_state}; reported state: {report['state']}; instruction: {report['instruction']}")

        if self.output_dir:
            record = {'wall_time': datetime.now().isoformat(timespec='milliseconds'),
                      'verbal_command': command, 'report': report, 'reasoning': reasoning,
                      'sample': sample_meta}
            with open(os.path.join(self.output_dir, 'psychologist_log.jsonl'), 'a') as f:
                f.write(json.dumps(record) + '\n')


def main(args=None):
    rclpy.init(args=args)
    psychological_monitor_node = PsychologicalMonitorNode()
    try:
        rclpy.spin(psychological_monitor_node)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        psychological_monitor_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
