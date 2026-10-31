#!/usr/bin/env bash
# Starts carla_ros_bridge and the PACE-ADS nodes of one personalization test, each node in its
# own xterm window (sudo apt install xterm).
#
# Prerequisites (see README.md): the CARLA 0.9.15 server is running, carla-ros-bridge and
# pace_ads_ws are built, and OPENAI_API_KEY is exported in this shell.
#
# Usage:  ./run_scenario.sh <traffic_light|pedestrian|work_zone|car_following> [samples_dir]
#   samples_dir  folder of the occupant signal samples (psychologist agent in the loop);
#                without it, the typed occupant state is published directly.
# Environment:
#   ROS_BRIDGE_WS  workspace of carla-ros-bridge (default: ~/carla-ros-bridge)
#   OUTPUT_ROOT    folder of the output files (default: ~/pace_ads_output)

set -e

SCENARIO=${1:-}
SAMPLES_DIR=${2:-}
case "$SCENARIO" in
    traffic_light|pedestrian|work_zone|car_following) ;;
    *) echo "Usage: $0 <traffic_light|pedestrian|work_zone|car_following> [samples_dir]"; exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_BRIDGE_WS=${ROS_BRIDGE_WS:-$HOME/carla-ros-bridge}
OUT_DIR="${OUTPUT_ROOT:-$HOME/pace_ads_output}/${SCENARIO}_$(date +%Y%m%d_%H%M%S)"
SETUP="source /opt/ros/humble/setup.bash && source $ROS_BRIDGE_WS/install/setup.bash && source $HERE/pace_ads_ws/install/setup.bash"
ARGS="--ros-args -p scenario:=$SCENARIO"

open_terminal() {
    xterm -hold -fa 'Monospace' -fs 11 -T "$1" -e bash -c "$SETUP && $2" &
}

open_terminal "carla_ros_bridge" "ros2 launch car_following carla_ros_bridge.launch.py scenario:=$SCENARIO"
echo "Waiting for the ego vehicle ..."
bash -c "$SETUP && until ros2 topic list | grep -q /carla/hero/speedometer; do sleep 1; done"

open_terminal "scenario_node" "ros2 run car_following scenario_node $ARGS"
echo "Waiting for the scenario ..."
bash -c "$SETUP && until ros2 topic list | grep -q /scenario_ready; do sleep 1; done"

open_terminal "decision_node" "ros2 run car_following decision_node $ARGS -p output_dir:=$OUT_DIR"
if [ -n "$SAMPLES_DIR" ]; then
    open_terminal "psychological_monitor_node" \
        "ros2 run car_following psychological_monitor_node --ros-args -p samples_dir:=$SAMPLES_DIR -p output_dir:=$OUT_DIR"
else
    open_terminal "psychological_monitor_node" \
        "ros2 run car_following psychological_monitor_node --ros-args -p output_dir:=$OUT_DIR"
fi
open_terminal "record_node" "ros2 run car_following record_node $ARGS -p output_dir:=$OUT_DIR"
sleep 3
open_terminal "control_node" "ros2 run car_following control_node $ARGS"

echo "Scenario $SCENARIO started. Output files: $OUT_DIR"
