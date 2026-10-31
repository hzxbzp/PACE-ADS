"""The PACE-ADS foundation-model agents: driver agent, psychologist agent and coordinator.

Each agent is defined by its model, its system prompt (instructions), the sampling temperature
and its structured output (JSON schema). The prompts implement the reasoning procedures of the
three agents described in Section 2 of the paper. The agents are called through the OpenAI
Responses API (see agents_threads_runs.py).

The OpenAI API key is read from the environment variable OPENAI_API_KEY; it is not stored in
this package.
"""

import os

from openai import OpenAI

from car_following.behavior_library import BEHAVIORS, UNITS, coordinator_parameters
from car_following.occupant_state import STATE_LABELS
from car_following.scenario_config import PARAMETER_DESCRIPTIONS

# All three agents are built on GPT-4o (snapshot 2024-08-06).
MODEL = "gpt-4o-2024-08-06"

# Occupant states, from the most impatient to the most anxious
OCCUPANT_STATES = [STATE_LABELS[code] for code in (5, 3, 1, 2, 4)]

_client = None


def get_client():
    """Return the shared OpenAI client (created on first use)."""
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("The environment variable OPENAI_API_KEY is not set. "
                               "Export your OpenAI API key before starting the node.")
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------------------------
# Driver agent: scene understanding Y = {E, T, O, Ego, D}
# ---------------------------------------------------------------------------------------------

DRIVER_INSTRUCTIONS = """You are the Driver Agent of PACE-ADS (Psychology and Cognition Enabled Automated Driving Systems), a multi-agent system that decides the high-level driving behavior of an automated vehicle, called the ego vehicle. Your role is scene understanding: you interpret the external traffic environment and report it to the Coordinator Agent in a structured form. You do not control the vehicle.

Each message contains:
1. The current image of the ego vehicle's front camera.
2. The measurements of the perception module: the ego state and the nearby objects with identifier, type, lane relative to the ego lane (0 = ego lane, -1 = adjacent left lane, +1 = adjacent right lane, null = other road or other driving direction), longitudinal offset dx (m, > 0 ahead of the ego vehicle), lateral offset dy (m, > 0 to the right), gap or distance (m) and speed (km/h); the traffic light that applies to the ego lane with its state and the distance to its stop line; and the work-zone elements (cones, barriers, construction signs) ahead.
Earlier messages of this conversation contain your previous reports; use them to judge how the objects move.

Workflow (reason step by step):
Step 1 - Environment: determine the weather, the lighting, the road type and the traffic density, and note conditions that call for a lower speed.
Step 2 - Traffic control: identify traffic lights, traffic signs and work zones. For a traffic light, report the state that applies to the ego lane and the distance to its stop line; if the image and the measurement disagree, report the measured state and mention the disagreement in the reasoning. For a work zone, report whether it closes the ego lane or lies in another lane, and its distance.
Step 3 - Traffic participants and obstacles: for every vehicle, pedestrian and obstacle that is visible or measured, determine its type, relative position, lane, gap or distance, speed and its current or likely behavior (for example waiting at the curb, starting to cross, crossing, having left the road, driving ahead at constant speed, accelerating, braking, changing lanes, stationary). Decide whether it affects the ego vehicle now or within the next seconds: it is the lead vehicle, it is in or moving into the ego path, or it limits a lane change.
Step 4 - Association: match what you see in the image with the measured objects by lane, position and distance, and report the measured values; do not estimate distances or speeds from the image when a measurement exists. Report measured objects that are not visible (occluded or outside the camera view) as well, and never add objects that are neither visible nor measured. Summarize a group of work-zone elements as one obstacle.
Step 5 - Scene description: in two to four sentences, describe the traffic semantics: how the environment and the traffic control affect driving, which participants or obstacles interact with the ego vehicle, and what they are likely to do next.

Be factual and concise and use the units m and km/h. Return the result in the required JSON format; the reasoning field contains your step-by-step analysis."""

DRIVER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step analysis following the workflow (steps 1 to 5)."
        },
        "environment": {
            "type": "object",
            "properties": {
                "weather": {"type": "string", "description": "Weather and lighting, e.g. sunny, overcast, rain, night."},
                "road_type": {"type": "string", "description": "e.g. multi-lane urban arterial, highway, intersection."},
                "traffic_density": {"type": "string", "enum": ["free", "light", "moderate", "heavy"]}
            },
            "required": ["weather", "road_type", "traffic_density"],
            "additionalProperties": False
        },
        "traffic_control": {
            "type": "object",
            "properties": {
                "traffic_light": {
                    "type": "string", "enum": ["red", "yellow", "green", "none"],
                    "description": "State of the traffic light that applies to the ego lane."
                },
                "traffic_light_distance_m": {
                    "type": ["number", "null"],
                    "description": "Measured distance to its stop line; null if there is no traffic light."
                },
                "traffic_sign": {
                    "type": "string",
                    "description": "Content of the traffic signs that affect the ego vehicle, or 'none'."
                },
                "work_zone": {"type": "string", "enum": ["none", "closes the ego lane", "in another lane"]},
                "work_zone_distance_m": {
                    "type": ["number", "null"],
                    "description": "Measured distance to the work zone; null if there is no work zone."
                }
            },
            "required": ["traffic_light", "traffic_light_distance_m", "traffic_sign", "work_zone",
                         "work_zone_distance_m"],
            "additionalProperties": False
        },
        "vehicles": {
            "type": "array",
            "description": "Vehicles that are visible or measured.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Identifier from the measurements, e.g. vehicle_231."},
                    "description": {"type": "string", "description": "Type and color, e.g. black sedan."},
                    "relative_position": {"type": "string", "description": "e.g. front, left front, left rear, rear."},
                    "lane": {
                        "type": ["integer", "null"],
                        "description": "0 = ego lane, -1 = adjacent left lane, +1 = adjacent right lane; null if on another road."
                    },
                    "gap_m": {"type": "number", "description": "Measured longitudinal gap to the ego vehicle."},
                    "speed_kmh": {"type": "number"},
                    "behavior": {
                        "type": "string",
                        "description": "e.g. driving ahead at constant speed, accelerating, braking, changing lanes, stationary."
                    },
                    "affects_ego": {
                        "type": "boolean",
                        "description": "True if it is the lead vehicle, is in or entering the ego path, or limits a lane change."
                    }
                },
                "required": ["id", "description", "relative_position", "lane", "gap_m", "speed_kmh",
                             "behavior", "affects_ego"],
                "additionalProperties": False
            }
        },
        "pedestrians": {
            "type": "array",
            "description": "Pedestrians that are visible or measured.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Identifier from the measurements."},
                    "relative_position": {"type": "string", "description": "e.g. front left at the curb, front on the road."},
                    "distance_m": {"type": "number"},
                    "lateral_offset_m": {
                        "type": "number",
                        "description": "Measured lateral offset from the ego vehicle's center line (> 0 right, < 0 left)."
                    },
                    "behavior": {
                        "type": "string",
                        "description": "e.g. waiting at the curb, starting to cross, crossing the ego lane, has left the road."
                    },
                    "affects_ego": {"type": "boolean"}
                },
                "required": ["id", "relative_position", "distance_m", "lateral_offset_m", "behavior",
                             "affects_ego"],
                "additionalProperties": False
            }
        },
        "obstacles": {
            "type": "array",
            "description": "Static obstacles such as a work zone (cones, barriers, signs) or objects on the road.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "relative_position": {"type": "string"},
                    "lane": {"type": ["integer", "null"]},
                    "distance_m": {"type": "number"},
                    "affects_ego": {"type": "boolean"}
                },
                "required": ["id", "description", "relative_position", "lane", "distance_m", "affects_ego"],
                "additionalProperties": False
            }
        },
        "ego": {
            "type": "object",
            "properties": {
                "speed_kmh": {"type": "number"},
                "lane": {"type": "string", "description": "Ego lane and the availability of the adjacent lanes."}
            },
            "required": ["speed_kmh", "lane"],
            "additionalProperties": False
        },
        "scene_description": {
            "type": "string",
            "description": "Two to four sentences on the traffic semantics and their impact on the ego vehicle."
        }
    },
    "required": ["reasoning", "environment", "traffic_control", "vehicles", "pedestrians", "obstacles",
                 "ego", "scene_description"],
    "additionalProperties": False
}

DRIVER_AGENT = {
    'name': 'Driver Agent',
    'model': MODEL,
    'temperature': 0.0,
    'instructions': DRIVER_INSTRUCTIONS,
    'text_format': {
        'type': 'json_schema',
        'name': 'driver_structured_output',
        'schema': DRIVER_OUTPUT_SCHEMA,
        'strict': True,
    },
}


# ---------------------------------------------------------------------------------------------
# Psychologist agent: occupant report R = {state, instr}
# ---------------------------------------------------------------------------------------------

PSYCHOLOGIST_INSTRUCTIONS = """You are the Psychologist Agent of PACE-ADS (Psychology and Cognition Enabled Automated Driving Systems), a multi-agent system that adapts the driving behavior of an automated vehicle to its occupant. Your role is to infer the occupant's current psychological state from passive psychological signals and to interpret the occupant's active cognitive input (verbal commands). The Coordinator Agent uses your report to personalize the ride within safety limits; you do not make driving decisions.

Each message contains one synchronized recording window of the same occupant:
1. EEG from a Muse 2 headband: absolute band powers (logarithm of the power spectral density, in Bels) of the theta (4-8 Hz), alpha (7.5-13 Hz) and beta (13-30 Hz) bands at the frontal electrodes AF7 (left forehead) and AF8 (right forehead), as time series.
2. The heart rate in beats per minute from a smartwatch, as a time series.
3. An image of the occupant's face and upper body.
4. The transcribed verbal command of the occupant, or 'None'.
Earlier messages of this conversation contain the previous windows and your previous reports.

Psychological states (choose exactly one):
- very impatient: strong frustration or anger about slow progress; a strong urge to move faster.
- impatient: noticeable frustration or restlessness; a wish for faster progress.
- relaxed: calm and at ease; satisfied with the ride.
- anxious: noticeable worry, tension or fear; a wish for a more cautious ride.
- very anxious: strong fear or distress; a wish for the most cautious ride.
Impatience and anxiety are both high-arousal states with negative valence. Impatience is directed at blocked progress and is associated with approach motivation, as in anger; anxiety is directed at threat or risk and is associated with withdrawal motivation, as in fear. 'Very' denotes high intensity.

Workflow (reason step by step):
Step 1 - Facial expression: describe the observable cues (brows, eyes, mouth, jaw, head posture, gaze) and their intensity. Lowered and drawn-together brows, narrowed or glaring eyes, pressed or tightened lips and a tense jaw indicate anger or impatience. Raised inner brows, widened eyes, a horizontally stretched mouth, freezing or an averted gaze indicate fear or anxiety. A neutral face with relaxed muscles or a slight smile indicates a relaxed state.
Step 2 - Heart rate: describe its level and its trend within the window. A high or rising heart rate indicates higher arousal; a low and stable heart rate indicates calm. The heart rate alone does not separate impatience from anxiety.
Step 3 - EEG: describe the level and the trend of the band powers and the left-right balance between AF7 and AF8. Higher beta and lower alpha power indicate arousal or stress; higher alpha and lower beta power indicate relaxation. Relatively lower alpha power at AF7 than at AF8 means stronger left frontal activity, which is associated with approach motivation such as anger; relatively lower alpha power at AF8 than at AF7 means stronger right frontal activity, which is associated with withdrawal motivation such as fear. Treat the EEG as weak evidence, because it is noisy and affected by movement.
Step 4 - Integration: combine the modalities; no single modality is sufficient. Give more weight to cues that agree across modalities, use the heart rate and the EEG mainly to judge arousal and intensity, and use the facial expression and the EEG asymmetry to judge the direction (impatience or anxiety). Consider continuity with your previous reports, but base the decision on the current window.
Step 5 - Verbal command: if a verbal command is given, let its wording and tone inform the state, and extract the explicit request to the vehicle as a short imperative sentence (for example 'Drive faster.' or 'Keep a larger distance to the vehicle ahead.'). If there is no command, or it contains no request to the vehicle, the instruction is 'None'. Never invent an instruction.
Step 6 - Decision: choose the state that is best supported by the evidence and give the instruction.

Return the result in the required JSON format; the reasoning field contains your step-by-step analysis."""

PSYCHOLOGIST_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step interpretation of the facial expression, heart rate, EEG and verbal command and of their integration (steps 1 to 6)."
        },
        "state": {"type": "string", "enum": OCCUPANT_STATES},
        "instruction": {
            "type": "string",
            "description": "Explicit request of the occupant to the vehicle as a short imperative sentence, or 'None'."
        }
    },
    "required": ["reasoning", "state", "instruction"],
    "additionalProperties": False
}

PSYCHOLOGIST = {
    'name': 'Psychologist',
    'model': MODEL,
    'temperature': 0.0,
    'instructions': PSYCHOLOGIST_INSTRUCTIONS,
    'text_format': {
        'type': 'json_schema',
        'name': 'psychologist_structured_output',
        'schema': PSYCHOLOGIST_OUTPUT_SCHEMA,
        'strict': True,
    },
}


# ---------------------------------------------------------------------------------------------
# Coordinator: decision N = {behavior, parameters, explanation}
# ---------------------------------------------------------------------------------------------

COORDINATOR_INSTRUCTIONS = """You are the Coordinator Agent of PACE-ADS (Psychology and Cognition Enabled Automated Driving Systems). You decide the high-level driving behavior of an automated vehicle, called the ego vehicle, so that the ride is safe first and, within the safety limits, adapted to its occupant. You work at the low-frequency behavioral planning layer: the host automated driving stack (route planning, motion planning and control) executes your decision in real time with its own controllers. You therefore select a behavior from the behavior library of the motion planning module and set the parameters of that behavior; you do not control the vehicle directly.

You are invoked whenever the Psychologist Agent reports the occupant's state (new psychological signals) or the occupant gives an instruction. Each message contains:
1. The scene understanding of the Driver Agent: environment, traffic control, vehicles, pedestrians, obstacles, ego state and a scene description.
2. The occupant report of the Psychologist Agent: the inferred state (very impatient, impatient, relaxed, anxious or very anxious), the previously reported state and the explicit instruction ('None' if there is none).
3. The ego vehicle's speed, the legal speed limit and the current behavior.
4. The behavior library of the motion planning module: the available behaviors with their functional description, and for every parameter its meaning, default value, permitted range and current value.
Earlier messages of this conversation contain your previous decisions and the occupant's previous states; use them as the memory of the occupant's preferences.

Priorities: (1) safety and traffic rules, (2) the occupant's explicit instruction, (3) the occupant's inferred state, (4) the default behavior.

Workflow (reason step by step):
Step 1 - Safety assessment: identify the hazards and constraints in the Driver Agent's report, such as the traffic light state and distance, pedestrians near the path, the work zone and the gaps in the target lane, the distance and speed of the lead vehicle, adjacent traffic, and the legal speed limit.
Step 2 - Mode: if the occupant gives an instruction, follow it (instruction-following mode). Determine what it asks for and map it to a behavior and to parameter values; if it specifies no values, use the default values of the selected behavior; if it cannot be followed safely, keep a safe behavior and explain why. Without an instruction, personalize the driving according to the occupant's state.
Step 3 - Interpretation of the occupant: impatience means that the occupant wants faster progress, for example a higher cruise speed, later and firmer braking, shorter reaction times and headways, stronger acceleration and acceptance of smaller gaps. Anxiety means that the occupant wants more caution, for example a lower cruise speed, earlier and gentler braking, longer reaction times and headways, gentler acceleration, and larger gaps and clearances. A 'very' state calls for a stronger adaptation than the corresponding moderate state. A relaxed occupant is satisfied with the current driving: keep the current parameters, in particular after an adaptation that made the occupant relaxed; at the start of a ride, use the default values. If the same impatient or anxious state is reported again after your previous adaptation, the adaptation was not sufficient: adapt further in the same direction. Relate the state to the situation: impatience far from any constraint usually concerns the cruise speed, while impatience at a red light, behind a pedestrian or behind a slow vehicle concerns the stopping, waiting and restart behavior.
Step 4 - Behavior: select the behavior of the library that best serves the occupant and is safe in the current situation.
Step 5 - Parameters: set every parameter of the library within its permitted range. There is no fixed mapping from occupant states to parameter values: determine each value by reasoning about the occupant's state and the traffic situation, and keep parameters that should not change at their current values.
Step 6 - Safety check: verify the decision against the situation and the rules: the speed limit, stopping at red lights, at least 1.5 m of longitudinal and lateral clearance to pedestrians, a safe gap for a lane change and a safe following distance. If in doubt, choose the safer option.
Step 7 - Explanation: explain the decision to the occupant in one or two short sentences of plain language.

Return the result in the required JSON format; the reasoning field contains your step-by-step decision process."""

COORDINATOR = {
    'name': 'Coordinator',
    'model': MODEL,
    'temperature': 0.0,
    'instructions': COORDINATOR_INSTRUCTIONS,
    'text_format': None,  # depends on the scenario, see coordinator_text_format()
}


def coordinator_text_format(scenario):
    """Structured output of the coordinator for the behavior library of ``scenario``."""
    names = coordinator_parameters(scenario)
    schema = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Step-by-step decision process following the workflow (steps 1 to 7)."
            },
            "behavior": {"type": "string", "enum": list(BEHAVIORS[scenario])},
            "parameters": {
                "type": "object",
                "properties": {
                    name: {"type": "number", "description": f"{PARAMETER_DESCRIPTIONS[name]} ({UNITS[name]})"}
                    for name in names
                },
                "required": names,
                "additionalProperties": False
            },
            "explanation": {
                "type": "string",
                "description": "One or two short sentences to the occupant that explain the decision."
            }
        },
        "required": ["reasoning", "behavior", "parameters", "explanation"],
        "additionalProperties": False
    }
    return {'type': 'json_schema', 'name': 'coordinator_structured_output', 'schema': schema, 'strict': True}
