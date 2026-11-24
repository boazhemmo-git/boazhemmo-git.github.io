"""Decides which frames contain a *moving living thing*.

Three stages, each of which removes a different kind of false alarm:

1. Motion gate (background subtraction). Nothing moves -> the detector is not
   run at all. This only saves CPU; on a windy day the gate is open constantly.
2. Object detector. Only people/animals produce boxes, so swaying trees,
   shadows, rain and headlights never get past this stage, however much they move.
3. Track confirmation. A box only alerts once it has been seen several times,
   has changing pixels inside it, and has actually changed position. That drops
   one-frame misdetections and statues/posters/sleeping pets sitting in front of
   a waving bush.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from detector import Detection


@dataclass
class Settings:
    motion_width: int = 320        # frames are downscaled to this width for motion analysis
    min_motion: float = 0.002      # fraction of the frame that must change to wake the detector
    box_motion: float = 0.03       # fraction of a box that must be changing to count as "moving"
    min_hits: int = 3              # moving detections needed before a track can alert
    still_iou: float = 0.85        # box must drift below this IoU vs. where it started
    match_iou: float = 0.25        # detection<->track association threshold
    max_misses: int = 8            # processed frames a track survives without a detection
    cooldown_s: float = 30.0       # min seconds between alerts for the same label
    warmup_frames: int = 10        # frames used to learn the background before alerting


@dataclass
class Event:
    t: float
    label: str
    score: float
    box: tuple[float, float, float, float]
    track_id: int


@dataclass
class _Track:
    id: int
    first: tuple
    box: tuple
    hits: int = 1
    moving_hits: int = 0
    misses: int = 0
    displaced: bool = False
    alerted: bool = False
    scores: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return max(self.scores, key=self.scores.get)


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class Pipeline:
    def __init__(self, detector, settings: Settings | None = None):
        self.detector = detector
        self.s = settings or Settings()
        self._bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25,
                                                      detectShadows=False)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._frames = 0
        self._last_alert: dict[str, float] = {}
        # Diagnostics since the last reset_stats(): what each stage saw.
        self.stats = {"frames": 0, "motion": 0, "detector": 0, "best": {}}
        self.tracks_view: list[tuple[tuple, str, bool]] = []  # for drawing: box, label, alerted

    def _motion_mask(self, frame: np.ndarray) -> np.ndarray:
        scale = self.s.motion_width / frame.shape[1]
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = self._bg.apply(gray)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

    def process(self, frame: np.ndarray, t: float) -> list[Event]:
        """Feed one frame (BGR) with its timestamp in seconds. Returns new alerts."""
        mask = self._motion_mask(frame)
        self._frames += 1
        self.stats["frames"] += 1
        if self._frames <= self.s.warmup_frames:
            return []

        moving = np.count_nonzero(mask) / mask.size >= self.s.min_motion
        if not moving and not self._tracks:
            self.tracks_view = []
            return []
        # While tracks exist keep detecting, so a pause mid-walk does not split the track.
        dets = self.detector.detect(frame, self.stats["best"])
        self.stats["motion"] += moving
        self.stats["detector"] += 1
        self._associate(dets, mask, frame.shape[1])

        events = []
        for tr in self._tracks:
            if (tr.alerted or tr.misses or tr.moving_hits < self.s.min_hits
                    or not tr.displaced):
                continue
            tr.alerted = True  # one alert per track, even if suppressed by cooldown
            label = tr.label
            if t - self._last_alert.get(label, -1e18) < self.s.cooldown_s:
                continue
            self._last_alert[label] = t
            events.append(Event(t, label, tr.scores[label] / tr.hits, tr.box, tr.id))
        self.tracks_view = [(tr.box, tr.label, tr.alerted) for tr in self._tracks
                            if not tr.misses]
        return events

    def _associate(self, dets: list[Detection], mask: np.ndarray, frame_w: int) -> None:
        pairs = sorted(((iou(tr.box, d.box), ti, di)
                        for ti, tr in enumerate(self._tracks)
                        for di, d in enumerate(dets)), reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < self.s.match_iou:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr, d = self._tracks[ti], dets[di]
            tr.box, tr.misses = d.box, 0
            tr.hits += 1
            tr.scores[d.label] = tr.scores.get(d.label, 0.0) + d.score
            if self._box_moving(d.box, mask, frame_w):
                tr.moving_hits += 1
            if iou(tr.first, d.box) < self.s.still_iou:
                tr.displaced = True
        for ti, tr in enumerate(self._tracks):
            if ti not in used_t:
                tr.misses += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                tr = _Track(self._next_id, d.box, d.box, scores={d.label: d.score})
                tr.moving_hits = int(self._box_moving(d.box, mask, frame_w))
                self._next_id += 1
                self._tracks.append(tr)
        self._tracks = [tr for tr in self._tracks if tr.misses <= self.s.max_misses]

    def _box_moving(self, box, mask: np.ndarray, frame_w: int) -> bool:
        k = mask.shape[1] / frame_w
        x1, y1, x2, y2 = (int(round(v * k)) for v in box)
        region = mask[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)]
        return region.size > 0 and np.count_nonzero(region) / region.size >= self.s.box_motion
