"""Occupant signal samples for the psychologist agent.

One sample folder per occupant state, named after the state with underscores
(e.g. <samples_dir>/very_impatient/), contains
    eeg.csv    time_s, Alpha_AF7, Alpha_AF8, Beta_AF7, Beta_AF8, Theta_AF7, Theta_AF8
               (Muse 2 absolute band powers in Bels)
    hr.csv     time_s, heart_rate_bpm
    face.jpg   (or face.png) image of the occupant
    meta.json  optional: source of the sample
The samples are prepared from the Emognition dataset with tools/prepare_emognition_sample.py;
the dataset itself is not redistributed with this package.
"""

import csv
import json
import os

EEG_CHANNELS = ['Alpha_AF7', 'Alpha_AF8', 'Beta_AF7', 'Beta_AF8', 'Theta_AF7', 'Theta_AF8']


def sample_folder(samples_dir, state_label):
    """Folder of the sample of an occupant state."""
    return os.path.join(samples_dir, state_label.replace(' ', '_'))


def _read_columns(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'{path} contains no data')
    return {key: [float(row[key]) for row in rows] for key in rows[0]}


def _interval(times):
    return (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 0.0


def load_sample(folder):
    """Load the EEG, heart rate, face image and metadata of one sample folder."""
    eeg = _read_columns(os.path.join(folder, 'eeg.csv'))
    missing = [c for c in EEG_CHANNELS if c not in eeg]
    if missing:
        raise ValueError(f"eeg.csv in {folder} lacks the columns {', '.join(missing)}")
    hr = _read_columns(os.path.join(folder, 'hr.csv'))

    face, face_mime = None, None
    for name, mime in (('face.jpg', 'image/jpeg'), ('face.jpeg', 'image/jpeg'), ('face.png', 'image/png')):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                face, face_mime = f.read(), mime
            break
    if face is None:
        raise ValueError(f'{folder} contains no face image (face.jpg or face.png)')

    meta = {}
    meta_path = os.path.join(folder, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return {
        'eeg': {c: eeg[c] for c in EEG_CHANNELS},
        'eeg_interval_s': _interval(eeg['time_s']),
        'hr': hr['heart_rate_bpm'],
        'hr_interval_s': _interval(hr['time_s']),
        'duration_s': len(eeg['time_s']) * _interval(eeg['time_s']),
        'face': face,
        'face_mime': face_mime,
        'meta': meta,
    }


def format_signals(sample):
    """Text with the EEG and heart-rate time series of a sample, for the psychologist agent."""
    n_eeg = len(sample['eeg'][EEG_CHANNELS[0]])
    lines = [
        f"EEG (Muse 2, absolute band power in Bels; {n_eeg} values per channel, one value every "
        f"{sample['eeg_interval_s']:.2f} s, window of {sample['duration_s']:.1f} s):"
    ]
    for channel in EEG_CHANNELS:
        values = ', '.join(f'{v:.3f}' for v in sample['eeg'][channel])
        lines.append(f'{channel}: [{values}]')
    hr_values = ', '.join(f'{v:.0f}' for v in sample['hr'])
    lines.append(f"Heart rate (beats per minute, one value every {sample['hr_interval_s']:.1f} s): [{hr_values}]")
    return '\n'.join(lines)
