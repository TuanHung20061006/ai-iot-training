#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import base64
import json
import mimetypes
import threading
import time
from collections import Counter
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
from urllib.parse import urlparse

import cv2 as cv
import mediapipe as mp
import numpy as np

from app import HISTORY_POINTS
from app import calc_history_point
from app import calc_landmark_list
from app import get_label
from app import pre_process_landmark
from app import pre_process_point_history
from app import read_label_csv
from model import KeyPointClassifier
from model import PointHistoryClassifier


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / 'web'
KEYPOINT_LABEL_PATH = BASE_DIR / 'model/keypoint_classifier/keypoint_classifier_label.csv'
POINT_HISTORY_LABEL_PATH = (
    BASE_DIR / 'model/point_history_classifier/point_history_classifier_label.csv'
)
KEYPOINT_MODEL_PATH = BASE_DIR / 'model/keypoint_classifier/keypoint_classifier.tflite'
POINT_HISTORY_MODEL_PATH = (
    BASE_DIR / 'model/point_history_classifier/point_history_classifier.tflite'
)
MAX_REQUEST_BYTES = 6 * 1024 * 1024
TEXT_MAX_CHARS = 2000
ACTION_LABELS = {
    'space': 'Space',
    'backspace': 'Backspace',
    'delete': 'Backspace',
    'clear': 'Clear',
    'enter': 'Enter',
}


class GestureRecognizer:
    def __init__(
        self,
        history_point='hand',
        hold_seconds=2.0,
        cooldown_seconds=0.8,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ):
        self.lock = threading.Lock()
        self.history_point = history_point
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds

        self.keypoint_classifier = KeyPointClassifier(
            model_path=str(KEYPOINT_MODEL_PATH))
        self.point_history_classifier = PointHistoryClassifier(
            model_path=str(POINT_HISTORY_MODEL_PATH))
        self.keypoint_labels = read_label_csv(KEYPOINT_LABEL_PATH)
        self.point_history_labels = read_label_csv(POINT_HISTORY_LABEL_PATH)

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)

        self.text = ''
        self.candidate_label = ''
        self.candidate_source = ''
        self.candidate_started_at = None
        self.cooldown_until = 0.0
        self.last_committed_label = ''
        self.last_commit = ''
        self.last_keypoint_label = ''
        self.last_motion_label = ''
        self.hand_present = False

    def process_data_url(self, image_data):
        image = self._decode_image(image_data)
        with self.lock:
            return self._process_image(image)

    def apply_control(self, action):
        now = time.monotonic()
        with self.lock:
            if action == 'clear':
                self.text = ''
                self._reset_candidate(release_latch=True)
            elif action == 'backspace':
                self.text = self.text[:-1]
                self._reset_candidate(release_latch=True)
            elif action == 'reset':
                self._reset_candidate(release_latch=True)
            return self._snapshot(now)

    def _decode_image(self, image_data):
        if not image_data:
            raise ValueError('Missing image data')
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        raw = base64.b64decode(image_data)
        image_array = np.frombuffer(raw, dtype=np.uint8)
        image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode image')
        return image

    def _process_image(self, image):
        now = time.monotonic()

        image = cv.flip(image, 1)
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.hands.process(rgb_image)
        rgb_image.flags.writeable = True

        if not results.multi_hand_landmarks:
            self.hand_present = False
            self.last_keypoint_label = ''
            self.last_motion_label = ''
            self.point_history.append([0, 0])
            committed = self._advance_candidate('', '', now)
            return self._snapshot(now, committed=committed)

        hand_landmarks = results.multi_hand_landmarks[0]
        landmark_list = calc_landmark_list(image, hand_landmarks)
        pre_processed_landmarks = pre_process_landmark(landmark_list)

        keypoint_id = self.keypoint_classifier(pre_processed_landmarks)
        keypoint_label = get_label(self.keypoint_labels, keypoint_id)

        self.point_history.append(
            calc_history_point(landmark_list, self.history_point))
        pre_processed_history = pre_process_point_history(
            image, self.point_history)

        motion_label = ''
        if len(pre_processed_history) == (self.history_length * 2):
            motion_id = self.point_history_classifier(pre_processed_history)
            self.finger_gesture_history.append(motion_id)
            most_common_motion = Counter(
                self.finger_gesture_history).most_common(1)
            if most_common_motion:
                motion_label = get_label(self.point_history_labels,
                                         most_common_motion[0][0])

        candidate_label, candidate_source = self._choose_candidate(
            keypoint_label, motion_label)
        committed = self._advance_candidate(candidate_label, candidate_source,
                                            now)

        self.hand_present = True
        self.last_keypoint_label = keypoint_label
        self.last_motion_label = motion_label
        return self._snapshot(now, committed=committed)

    def _choose_candidate(self, keypoint_label, motion_label):
        motion_candidate = self._motion_candidate(motion_label)
        if motion_candidate:
            return motion_candidate, 'motion'

        keypoint_candidate = self._text_candidate(keypoint_label)
        if keypoint_candidate:
            return keypoint_candidate, 'keypoint'

        return '', ''

    def _motion_candidate(self, label):
        label = self._clean_label(label)
        if not label:
            return ''

        action = ACTION_LABELS.get(label.lower())
        if action:
            return action
        if len(label) == 1:
            return label
        return ''

    def _text_candidate(self, label):
        label = self._clean_label(label)
        if not label:
            return ''
        return label

    def _clean_label(self, label):
        label = (label or '').strip()
        if not label:
            return ''
        lowered = label.lower()
        if lowered.startswith('class'):
            return ''
        if lowered in ('none', 'unknown', 'invalid'):
            return ''
        return label

    def _advance_candidate(self, label, source, now):
        if not label:
            self._reset_candidate(release_latch=True)
            return ''

        if label != self.candidate_label:
            self.candidate_label = label
            self.candidate_source = source
            self.candidate_started_at = now
            if label != self.last_committed_label:
                self.last_committed_label = ''

        elapsed = now - self.candidate_started_at
        in_cooldown = now < self.cooldown_until
        already_committed = label == self.last_committed_label
        if elapsed >= self.hold_seconds and not in_cooldown and not already_committed:
            self._commit_label(label)
            self.cooldown_until = now + self.cooldown_seconds
            self.last_committed_label = label
            self.last_commit = label
            return label

        return ''

    def _commit_label(self, label):
        if label == 'Space':
            self.text += ' '
        elif label == 'Backspace':
            self.text = self.text[:-1]
        elif label == 'Clear':
            self.text = ''
        elif label == 'Enter':
            self.text += '\n'
        else:
            self.text += label
        self.text = self.text[-TEXT_MAX_CHARS:]

    def _reset_candidate(self, release_latch=False):
        self.candidate_label = ''
        self.candidate_source = ''
        self.candidate_started_at = None
        if release_latch:
            self.last_committed_label = ''

    def _snapshot(self, now, committed=''):
        elapsed = 0.0
        if self.candidate_started_at is not None:
            elapsed = max(0.0, now - self.candidate_started_at)

        return {
            'text': self.text,
            'hand_present': self.hand_present,
            'keypoint_label': self.last_keypoint_label,
            'motion_label': self.last_motion_label,
            'candidate_label': self.candidate_label,
            'candidate_source': self.candidate_source,
            'candidate_elapsed': elapsed,
            'hold_seconds': self.hold_seconds,
            'cooldown_remaining': max(0.0, self.cooldown_until - now),
            'committed': committed,
            'last_commit': self.last_commit,
        }


class GestureRequestHandler(BaseHTTPRequestHandler):
    server_version = 'GestureWeb/1.0'

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self._send_file(WEB_DIR / 'index.html')
            return
        if parsed.path.startswith('/web/'):
            relative = unquote(parsed.path[len('/web/'):])
            requested_path = (WEB_DIR / relative).resolve()
            if not self._is_inside(requested_path, WEB_DIR.resolve()):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._send_file(requested_path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/predict':
            self._handle_predict()
            return
        if parsed.path == '/api/control':
            self._handle_control()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):
        return

    def _handle_predict(self):
        try:
            payload = self._read_json()
            state = self.server.recognizer.process_data_url(
                payload.get('image', ''))
            self._send_json(state)
        except Exception as exc:
            self._send_json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_control(self):
        try:
            payload = self._read_json()
            action = payload.get('action', '')
            state = self.server.recognizer.apply_control(action)
            self._send_json(state)
        except Exception as exc:
            self._send_json({'error': str(exc)}, HTTPStatus.BAD_REQUEST)

    def _read_json(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > MAX_REQUEST_BYTES:
            raise ValueError('Request body is too large')
        body = self.rfile.read(content_length)
        if not body:
            return {}
        return json.loads(body.decode('utf-8'))

    def _send_file(self, path):
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(path.name)[0]
        if content_type is None:
            content_type = 'application/octet-stream'

        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _is_inside(path, directory):
        try:
            path.relative_to(directory)
            return True
        except ValueError:
            return False


class GestureHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--history_point', choices=HISTORY_POINTS, default='hand')
    parser.add_argument('--hold_seconds', type=float, default=2.0)
    parser.add_argument('--cooldown_seconds', type=float, default=0.8)
    parser.add_argument('--min_detection_confidence', type=float, default=0.7)
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5)
    return parser.parse_args()


def main():
    args = get_args()
    recognizer = GestureRecognizer(
        history_point=args.history_point,
        hold_seconds=args.hold_seconds,
        cooldown_seconds=args.cooldown_seconds,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    server = GestureHTTPServer((args.host, args.port), GestureRequestHandler)
    server.recognizer = recognizer
    print(f'Gesture web app running at http://{args.host}:{args.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
