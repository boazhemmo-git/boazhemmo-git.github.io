"""Logic tests for the pipeline with a scripted detector (no model needed).

    python -m unittest test_pipeline -v
"""
import unittest

import numpy as np

from detector import Detection
from pipeline import Pipeline, Settings

W, H = 640, 360
rng = np.random.default_rng(0)
BACKGROUND = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)


class ScriptedDetector:
    """Returns whatever boxes the test says are in the current frame."""

    def __init__(self):
        self.boxes, self.calls = [], 0

    def detect(self, frame):
        self.calls += 1
        return [Detection(b, label, 0.9) for b, label in self.boxes]


def scene(sway=None, obj=None):
    """Static background, optional flickering 'tree' region, optional solid moving object."""
    frame = BACKGROUND.copy()
    if sway is not None:
        x1, y1, x2, y2 = sway
        frame[y1:y2, x1:x2] = rng.integers(0, 255, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
    if obj is not None:
        x1, y1, x2, y2 = (int(v) for v in obj)
        frame[y1:y2, x1:x2] = (235, 235, 235)
    return frame


def run(frames_and_boxes, settings=None):
    det = ScriptedDetector()
    pipe = Pipeline(det, settings or Settings())
    events = []
    for i, (frame, boxes) in enumerate(frames_and_boxes):
        det.boxes = boxes
        events += pipe.process(frame, i * 0.25)
    return events, det


WARMUP = [(scene(), [])] * 15
TREE = (400, 20, 620, 200)


class PipelineTest(unittest.TestCase):
    def test_static_scene_never_runs_detector(self):
        events, det = run(WARMUP + [(scene(), [])] * 40)
        self.assertEqual(events, [])
        self.assertEqual(det.calls, 0)

    def test_wind_alone_does_not_alert(self):
        events, det = run(WARMUP + [(scene(sway=TREE), []) for _ in range(60)])
        self.assertEqual(events, [])
        self.assertGreater(det.calls, 0)  # gate opened, detector found nothing alive

    def test_walking_person_alerts_once(self):
        seq = []
        for i in range(30):
            box = (20 + i * 15, 150, 80 + i * 15, 330)
            seq.append((scene(sway=TREE, obj=box), [(box, "person")]))
        events, _ = run(WARMUP + seq)
        self.assertEqual([e.label for e in events], ["person"])

    def test_statue_in_front_of_waving_bush_does_not_alert(self):
        statue = (430, 40, 500, 190)  # inside the swaying region, never changes position
        seq = [(scene(sway=TREE), [(statue, "person")]) for _ in range(60)]
        events, _ = run(WARMUP + seq)
        self.assertEqual(events, [])

    def test_single_frame_misdetection_does_not_alert(self):
        ghost = (430, 40, 500, 190)
        seq = [(scene(sway=TREE), [(ghost, "dog")] if i == 5 else []) for i in range(40)]
        events, _ = run(WARMUP + seq)
        self.assertEqual(events, [])

    def test_cooldown_suppresses_repeat_of_same_label(self):
        def walk(y):
            return [(scene(obj=b), [(b, "dog")])
                    for b in ((20 + i * 20, y, 100 + i * 20, y + 60) for i in range(12))]
        gap = [(scene(), [])] * 12
        seq = WARMUP + walk(100) + gap + walk(250)
        self.assertEqual(len(run(seq, Settings(cooldown_s=3600))[0]), 1)
        self.assertEqual(len(run(seq, Settings(cooldown_s=0))[0]), 2)


if __name__ == "__main__":
    unittest.main()
