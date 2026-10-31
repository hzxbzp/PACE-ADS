"""Helpers that send messages to the PACE-ADS agents and read their replies.

The agents are called through the OpenAI Responses API. The function names follow the
original implementation, which used OpenAI Assistants threads and runs (an API that OpenAI
retired on 26 August 2026):

* a "thread" is a Conversation object: the agent's responses are chained with
  previous_response_id, so the agent sees its own earlier inputs and outputs (server-side
  conversation state, as with an Assistants thread);
* a "run" is one responses.create call with the agent's instructions, model, temperature and
  JSON output schema; truncation="auto" drops the oldest turns if the context window of the
  model is exceeded.
"""

import base64
import json

from car_following.multi_agents import (COORDINATOR, DRIVER_AGENT, PSYCHOLOGIST,
                                        coordinator_text_format, get_client)


class Conversation:
    """Conversation state of one agent (plays the role of an Assistants thread)."""

    def __init__(self, agent):
        self.agent = agent
        self.previous_response_id = None
        self.last_output_text = None


class Run:
    """Result of one agent call (plays the role of an Assistants run)."""

    def __init__(self, response):
        self.response = response
        self.id = response.id
        self.status = response.status


def create_threads():
    """Create the conversations of the driver agent, the psychologist agent and the coordinator."""
    return Conversation(DRIVER_AGENT), Conversation(PSYCHOLOGIST), Conversation(COORDINATOR)


def wait_on_run(run, thread):
    """Agent calls are synchronous, so the run is complete when this is called."""
    return run


def text_content(text):
    """Input item with text."""
    return {'type': 'input_text', 'text': text}


def image_content(image_bytes, mime_type='image/png'):
    """Input item with an image, sent as a base64 data URL."""
    data = base64.b64encode(image_bytes).decode('utf-8')
    return {'type': 'input_image', 'image_url': f'data:{mime_type};base64,{data}', 'detail': 'high'}


def submit_message(thread, content, text_format=None):
    """Send one user message to the agent of ``thread`` and store its reply."""
    agent = thread.agent
    request = {
        'model': agent['model'],
        # instructions are not carried over from previous responses, so they are sent every time
        'instructions': agent['instructions'],
        'input': [{'role': 'user', 'content': content}],
        'temperature': agent['temperature'],
        'truncation': 'auto',
        'store': True,
    }
    text_format = text_format or agent.get('text_format')
    if text_format:
        request['text'] = {'format': text_format}
    if thread.previous_response_id:
        request['previous_response_id'] = thread.previous_response_id

    response = get_client().responses.create(**request)
    thread.previous_response_id = response.id
    thread.last_output_text = response.output_text
    return Run(response)


def _parse_json(thread, agent_name):
    """Return the JSON reply of the last call, or None if the agent returned no valid JSON."""
    text = thread.last_output_text
    if not text:
        print(f"The {agent_name} returned no text output.")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"The {agent_name} returned invalid JSON: {text[:300]}")
        return None


def driver_submit(thread, image_bytes, perception_text):
    """Driver agent input: front camera image and measurements of the perception module."""
    content = [
        image_content(image_bytes, 'image/png'),
        text_content("Measurements of the perception module:\n" + perception_text),
        text_content("Analyze the current traffic scene following your workflow."),
    ]
    return submit_message(thread, content)


def get_driver_response(thread):
    return _parse_json(thread, 'driver agent')


def psychologist_submit(thread, signals_text, face_image_bytes=None, face_mime_type='image/jpeg',
                        verbal_command='None'):
    """Psychologist agent input: EEG and heart rate (text), face image and verbal command."""
    content = [text_content(signals_text)]
    if face_image_bytes:
        content.append(text_content("Image of the occupant:"))
        content.append(image_content(face_image_bytes, face_mime_type))
    content.append(text_content(f"Verbal command of the occupant (transcribed): {verbal_command or 'None'}"))
    content.append(text_content("Infer the occupant's current psychological state and instruction "
                                "following your workflow."))
    return submit_message(thread, content)


def get_psychologist_response(thread):
    return _parse_json(thread, 'psychologist agent')


def coordinator_submit(thread, scenario, driver_output, occupant_report, ego_info, library_text):
    """Coordinator input: scene understanding, occupant report, ego state and behavior library."""
    content = [
        text_content("Scene understanding of the Driver Agent:\n" + json.dumps(driver_output, indent=1)),
        text_content("Occupant report of the Psychologist Agent: "
                     f"state = {occupant_report['state']}; "
                     f"previous state = {occupant_report['previous_state']}; "
                     f"instruction = {occupant_report['instruction']}"),
        text_content(f"Ego vehicle: speed = {ego_info['speed_kmh']} km/h; "
                     f"legal speed limit = {ego_info['speed_limit_kmh']} km/h; "
                     f"current behavior = {ego_info['current_behavior']}"),
        text_content("Behavior library of the motion planning module:\n" + library_text),
        text_content("Decide the behavior and the value of every parameter following your workflow."),
    ]
    return submit_message(thread, content, coordinator_text_format(scenario))


def get_coordinator_response(thread):
    return _parse_json(thread, 'coordinator')
