"""Occupant psychological states exchanged by psychological_monitor_node and decision_node.

The state inferred by the psychologist agent is published as an integer code on
/psychological_output (std_msgs/Int32) and passed to the coordinator as text.
Codes 1-3 keep the numbering of the original implementation, where they were called
comfortable, nervous and impatient; codes 4 and 5 add the two intensity levels.
"""

STATE_LABELS = {
    1: 'relaxed',
    2: 'anxious',
    3: 'impatient',
    4: 'very anxious',
    5: 'very impatient',
}

LABEL_TO_CODE = {label: code for code, label in STATE_LABELS.items()}

# Accepted keyboard inputs (the names of the original implementation are accepted as aliases).
STATE_INPUTS = {
    'relaxed': 1,
    'comfortable': 1,
    'anxious': 2,
    'nervous': 2,
    'impatient': 3,
    'very anxious': 4,
    'very nervous': 4,
    'very impatient': 5,
}


def parse_state(text):
    """Return the state code of a keyboard input, or None if the input is not a state."""
    return STATE_INPUTS.get(' '.join(text.strip().lower().split()))
