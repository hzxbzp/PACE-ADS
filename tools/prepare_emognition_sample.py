#!/usr/bin/env python3
"""Prepare an occupant signal sample for the PACE-ADS psychologist agent from the Emognition dataset.

The Emognition Wearable Dataset 2020 (Saganowski et al., Scientific Data 9, 158, 2022;
https://doi.org/10.7910/DVN/R9WAF4) is not redistributed with this package. Obtain it from its
authors, unpack study_data.zip (and the upper-body videos), and run, for example:

    python3 tools/prepare_emognition_sample.py \
        --study-data /path/to/study_data --participant 22 --clip ANGER \
        --state "very impatient" --start 30 --duration 20 \
        --video /path/to/upper_body_video_of_the_clip.mp4 --out-dir ~/occupant_samples

The sample is written to <out-dir>/<state>/ (spaces replaced by underscores):
    eeg.csv    time_s, Alpha_AF7, Alpha_AF8, Beta_AF7, Beta_AF8, Theta_AF7, Theta_AF8
    hr.csv     time_s, heart_rate_bpm
    face.jpg   frame of the upper-body video (or a copy of --face-image)
    meta.json  source of the sample and the participant's self-report for the clip
The signals of the stimulus (film) phase are cut to [start, start + duration) seconds and
averaged in bins of --eeg-interval and --hr-interval seconds; no filtering or feature
extraction is applied. EEG samples recorded while the headband was not on the head
(HeadBandOn != 1) and heart-rate readings of 0 are skipped.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime

EEG_CHANNELS = ['Alpha_AF7', 'Alpha_AF8', 'Beta_AF7', 'Beta_AF8', 'Theta_AF7', 'Theta_AF8']
STATES = ['very impatient', 'impatient', 'relaxed', 'anxious', 'very anxious']
CLIPS = ['BASELINE', 'NEUTRAL', 'AWE', 'DISGUST', 'SURPRISE', 'ANGER', 'ENTHUSIASM', 'LIKING',
         'FEAR', 'AMUSEMENT', 'SADNESS']


def parse_time(text):
    """Parse an Emognition timestamp.

    Muse 2: 2020-07-16T09:39:03.001000; Samsung watch: 2020-07-16T09:39:03:032406.
    """
    date, _, clock = text.partition('T')
    parts = clock.split(':')
    if len(parts) == 4:
        clock = ':'.join(parts[:3]) + '.' + parts[3]
    return datetime.fromisoformat(f'{date}T{clock}')


def is_valid(value):
    return isinstance(value, (int, float)) and not math.isnan(value)


def bin_means(times, values, start, duration, interval):
    """Mean of the valid values in consecutive bins of ``interval`` s in [start, start + duration)."""
    sums, counts = {}, {}
    for t, v in zip(times, values):
        if start <= t < start + duration and is_valid(v):
            k = int((t - start) // interval)
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def remove_face_images(folder):
    for name in ('face.jpg', 'face.jpeg', 'face.png'):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            os.remove(path)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare an occupant signal sample for the psychologist agent from the Emognition dataset.')
    parser.add_argument('--study-data', required=True,
                        help='folder of the unpacked study_data.zip (one folder per participant)')
    parser.add_argument('--participant', required=True, type=int, help='participant id (22 to 64)')
    parser.add_argument('--clip', required=True, choices=CLIPS, help='film clip of the stimulus phase')
    parser.add_argument('--state', required=True, choices=STATES,
                        help='occupant state that the sample represents')
    parser.add_argument('--out-dir', required=True,
                        help='folder of the occupant samples (samples_dir of psychological_monitor_node)')
    parser.add_argument('--start', type=float, default=0.0,
                        help='start of the window in s from the beginning of the clip (default: 0)')
    parser.add_argument('--duration', type=float, default=20.0, help='length of the window in s (default: 20)')
    parser.add_argument('--eeg-interval', type=float, default=0.25, help='EEG bin length in s (default: 0.25)')
    parser.add_argument('--hr-interval', type=float, default=1.0, help='heart-rate bin length in s (default: 1)')
    face = parser.add_mutually_exclusive_group(required=True)
    face.add_argument('--video', help='upper-body video of the clip; one frame is saved as face.jpg')
    face.add_argument('--face-image', help='image of the occupant (.jpg or .png) used as face image')
    parser.add_argument('--video-time', type=float, default=None,
                        help='time of the video frame in s from the start of the video (default: middle of the window)')
    args = parser.parse_args()

    folder = os.path.join(os.path.expanduser(args.study_data), str(args.participant))
    prefix = os.path.join(folder, f'{args.participant}_{args.clip}_STIMULUS')

    # EEG: Muse 2 absolute band powers (rows at 256 Hz)
    muse = load_json(prefix + '_MUSE.json')
    t0 = parse_time(muse['TimeStamp'][0])
    eeg_times = [(parse_time(t) - t0).total_seconds() for t in muse['TimeStamp']]
    headband_on = muse.get('HeadBandOn')
    eeg_bins = {}
    for channel in EEG_CHANNELS:
        values = muse[channel]
        if headband_on is not None:
            values = [v if (is_valid(on) and on == 1) else None for v, on in zip(values, headband_on)]
        eeg_bins[channel] = bin_means(eeg_times, values, args.start, args.duration, args.eeg_interval)
    eeg_keys = sorted(set.intersection(*(set(bins) for bins in eeg_bins.values())))
    if not eeg_keys:
        sys.exit('No valid EEG data in the selected window.')

    # Heart rate: Samsung Galaxy Watch (10 Hz); same clock as the Muse timestamps
    watch = load_json(prefix + '_SAMSUNG_WATCH.json')
    hr_times = [(parse_time(t) - t0).total_seconds() for t, _ in watch['heartRate']]
    hr_values = [v if (is_valid(v) and v > 0) else None for _, v in watch['heartRate']]
    hr_bins = bin_means(hr_times, hr_values, args.start, args.duration, args.hr_interval)
    if not hr_bins:
        sys.exit('No valid heart-rate data in the selected window.')

    out = os.path.join(os.path.expanduser(args.out_dir), args.state.replace(' ', '_'))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'eeg.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s'] + EEG_CHANNELS)
        for k in eeg_keys:
            writer.writerow([round(k * args.eeg_interval, 3)] + [round(eeg_bins[c][k], 5) for c in EEG_CHANNELS])
    with open(os.path.join(out, 'hr.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'heart_rate_bpm'])
        for k in sorted(hr_bins):
            writer.writerow([round(k * args.hr_interval, 3), round(hr_bins[k], 2)])

    # Face image
    remove_face_images(out)
    if args.face_image:
        extension = os.path.splitext(args.face_image)[1].lower()
        if extension not in ('.jpg', '.jpeg', '.png'):
            sys.exit('--face-image must be a .jpg or .png file')
        shutil.copyfile(args.face_image, os.path.join(out, 'face.png' if extension == '.png' else 'face.jpg'))
        face_source = os.path.basename(args.face_image)
    else:
        import cv2
        video_time = args.video_time if args.video_time is not None else args.start + args.duration / 2
        capture = cv2.VideoCapture(args.video)
        capture.set(cv2.CAP_PROP_POS_MSEC, video_time * 1000.0)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            sys.exit(f'Could not read a frame at {video_time:.2f} s from {args.video}')
        if frame.shape[1] > 1280:
            height = int(round(frame.shape[0] * 1280 / frame.shape[1]))
            frame = cv2.resize(frame, (1280, height))
        cv2.imwrite(os.path.join(out, 'face.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        face_source = f'{os.path.basename(args.video)} at {video_time:.2f} s'

    # Self-report of the participant after the clip (1 = not at all, 5 = extremely)
    self_report = None
    questionnaire_path = os.path.join(folder, f'{args.participant}_QUESTIONNAIRES.json')
    if os.path.exists(questionnaire_path):
        for entry in load_json(questionnaire_path).get('questionnaires', []):
            if entry.get('movie') == args.clip:
                self_report = {'emotions': entry.get('emotions'), 'sam': entry.get('sam')}
                break

    meta = {
        'state': args.state,
        'dataset': 'Emognition Wearable Dataset 2020 (https://doi.org/10.7910/DVN/R9WAF4)',
        'participant': args.participant,
        'clip': args.clip,
        'stage': 'STIMULUS',
        'window_start_s': args.start,
        'window_duration_s': args.duration,
        'eeg_interval_s': args.eeg_interval,
        'hr_interval_s': args.hr_interval,
        'eeg_rows': len(eeg_keys),
        'hr_rows': len(hr_bins),
        'face_image': face_source,
        'self_report': self_report,
    }
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Sample written to {out}: {len(eeg_keys)} EEG rows, {len(hr_bins)} heart-rate rows, face image from {face_source}')
    if self_report:
        print(f"Self-report for {args.clip}: emotions {self_report['emotions']}, SAM {self_report['sam']}")
        rating = (self_report['emotions'] or {}).get(args.clip)
        if rating is not None and args.state != 'relaxed':
            very = args.state.startswith('very')
            if very and rating < 4:
                print(f'Warning: {args.clip} was rated {rating} (< 4), which does not indicate a very intense state.')
            elif not very and rating >= 4:
                print(f"Warning: {args.clip} was rated {rating} (>= 4), which indicates the 'very' intensity level.")


if __name__ == '__main__':
    main()
