#!/usr/bin/env python3
"""
Mall Shoplifting-Detection (Behaviour-Risk Flagging)
====================================================

A retail loss-prevention assistant that watches a store / mall camera feed
and flags **behaviour risk** for staff review.  It never decides that a
theft happened -- it raises timestamped flags when a tracked person shows
one or more of four heuristic risk signals:

1. **LONG DWELL**       -- the person stays inside one floor zone longer
                           than a configurable limit (default 10 s).
2. **REPEATED REACH**   -- while standing in one zone, the person's box
                           repeatedly grows taller and settles back (the
                           box-geometry signature of arm raises toward a
                           shelf), 3+ times in the same zone stay.
3. **ITEM VANISHED**    -- a carried item (handbag, backpack, suitcase,
                           sports ball, bottle, cup, umbrella) that was
                           stably associated with the person disappears
                           while the person is still in frame -- a
                           concealment-style signal.
4. **RAPID EXIT**       -- a person already at WATCH level or above leaves
                           the scene at high speed.

Each signal adds points to a per-person risk score:

* score < 30   -> LOW   (green box)
* 30 - 59      -> WATCH (amber box)
* score >= 60  -> RISK FLAG (red box + alert banner + event-log entry)

Every signal, and every RISK FLAG, is printed to the console and appended
to a timestamped event log (``events.log`` by default).

One shared risk pipeline serves both run modes:

* **Real mode** (``--source``) -- YOLOv8 detects people and carried items;
  ByteTrack (via Ultralytics) supplies the person track IDs.
* **Simulation mode** (``--simulate``) -- a store scene is rendered with
  OpenCV (shelves, shoppers, one carried handbag).  The scene publishes
  person / item boxes derived from the rendered figures through a small
  centroid tracker, producing exactly the same detection stream the real
  mode produces.  The risk engine only ever sees that stream -- it never
  peeks at the scene's scripted ground truth.

Honesty note
------------
These are heuristics for human review, NOT proof of theft.  Innocent
shoppers dwell at shelves, reach for products, put bags down and hurry
out of stores every day.  Real-mode accuracy depends on camera angle,
lighting and how cleanly YOLOv8 separates people in a crowd.

Examples
--------
    python mall_shoplifting_detection.py                  (file chooser)
    python mall_shoplifting_detection.py --source pick
    python mall_shoplifting_detection.py --source mall.mp4
    python mall_shoplifting_detection.py --source 0       (webcam)
    python mall_shoplifting_detection.py --simulate

Press Q to quit.  The annotated video is written to ``output.mp4``
(change with ``--output``).

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_WIDTH, SIM_HEIGHT = 1280, 720   # simulation canvas
SIM_FPS = 30                        # simulation frame rate

# COCO classes treated as "carried items" (class id -> label)
ITEM_CLASSES = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    32: "sports ball",
    39: "bottle",
    41: "cup",
}
PERSON_CLASS = 0

ZONE_COLS, ZONE_ROWS = 6, 4         # floor-zone grid over the frame

DWELL_LIMIT_S = 10.0                # seconds in one zone before LONG DWELL
DWELL_POINTS = 25                   # points for crossing the dwell limit
DWELL_EXTRA_PER_S = 1.0             # extra points per second beyond limit
DWELL_EXTRA_CAP = 15

REACH_RISE = 0.12                   # box must grow >=12% taller than baseline
REACH_MIN_S = 0.15                  # ... held at least this long
REACH_SETTLE_S = 1.5                # ... and settle back within this window
REACH_STILL_SPEED = 30.0            # px/s; reaches only count while ~stationary
REACH_TRIGGER = 3                   # reaches in one zone stay -> event
REACH_POINTS = 20                   # points at the trigger
REACH_EXTRA_POINTS = 5              # per further reach (capped)
REACH_EXTRA_CAP = 20

ITEM_STABLE_S = 1.0                 # association time before "carrying"
ITEM_GONE_S = 2.0                   # absence time before ITEM VANISHED
ITEM_POINTS = 35

EXIT_MIN_SCORE = 30                 # must be WATCH+ for RAPID EXIT to count
EXIT_SPEED = 260.0                  # px/s centroid speed at track loss
EXIT_POINTS = 20

WATCH_SCORE = 30
FLAG_SCORE = 60

TRACK_LOST_S = 0.6                  # unseen this long -> track finalised

COLOUR_LOW = (0, 190, 70)           # green (BGR)
COLOUR_WATCH = (0, 190, 240)        # amber
COLOUR_FLAG = (0, 55, 230)          # red
COLOUR_TEXT = (255, 255, 255)
COLOUR_ITEM = (230, 190, 0)         # cyan-ish for item boxes


# ---------------------------------------------------------------------------
# Detection stream (shared by real mode and simulation mode)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: str                 # "person" or an item label
    conf: float
    track_id: int            # -1 when the source gives no ID

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Simulation scene
# ---------------------------------------------------------------------------
# A rendered store interior.  Each shopper follows a keyframed path; the
# publisher computes every published box from the figure's drawn geometry
# (including a raised arm, which makes the person box taller) and runs a
# centroid tracker over the published boxes, so the risk engine receives
# the same kind of stream YOLO + ByteTrack produce in real mode.

@dataclass
class SimShopper:
    name: str
    colour: tuple
    # path segments: (t0, t1, x0, y0, x1, y1) -- feet position lerped over time
    segments: list
    reaches: list = field(default_factory=list)   # reach start times (s)
    bag_span: tuple | None = None                 # (t0, t1) bag visible
    height: float = 112.0
    width: float = 46.0

    def pos(self, t: float):
        for (t0, t1, x0, y0, x1, y1) in self.segments:
            if t0 <= t <= t1:
                if t1 == t0:
                    return x1, y1, True
                f = (t - t0) / (t1 - t0)
                return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, True
        return 0.0, 0.0, False

    def arm_raise(self, t: float) -> float:
        """0..1 arm raise amount from the scripted reach times."""
        best = 0.0
        for rt in self.reaches:
            if rt <= t <= rt + 1.1:
                best = max(best, math.sin(math.pi * (t - rt) / 1.1))
        return best

    def bag_visible(self, t: float) -> bool:
        return self.bag_span is not None and self.bag_span[0] <= t <= self.bag_span[1]


def build_sim_shoppers():
    return [
        # Browses an aisle, pauses briefly, leaves at walking speed.
        SimShopper(
            name="browser", colour=(200, 160, 60),
            segments=[(1.0, 10.0, 1200, 205, 300, 205),
                      (10.0, 13.0, 300, 205, 300, 205),
                      (13.0, 20.0, 300, 205, 80, 640)],
        ),
        # Stands at a shelf ~15 s doing five shelf reaches, then leaves.
        SimShopper(
            name="dweller", colour=(90, 90, 220),
            segments=[(2.0, 6.0, 60, 600, 430, 305),
                      (6.0, 21.5, 430, 305, 430, 305),
                      (21.5, 27.5, 430, 305, 1240, 620)],
            reaches=[8.0, 11.0, 14.0, 17.0, 20.0],
        ),
        # Stands at a shelf the whole time (~11.5 s) with no reaches.
        SimShopper(
            name="stander", colour=(70, 180, 120),
            segments=[(3.0, 5.0, 1240, 120, 905, 305),
                      (5.0, 16.5, 905, 305, 905, 305),
                      (16.5, 22.0, 905, 305, 1240, 640)],
        ),
        # Carries a handbag; the bag disappears mid-stay (concealment-style),
        # then the shopper hurries out.
        SimShopper(
            name="carrier", colour=(220, 120, 200),
            segments=[(2.0, 6.0, 60, 640, 700, 470),
                      (6.0, 15.5, 700, 470, 700, 470),
                      (15.5, 17.3, 700, 470, 1250, 640)],
            bag_span=(2.0, 13.0),
        ),
    ]


class CentroidTracker:
    """Minimal centroid tracker used by the simulation publisher."""

    def __init__(self, max_dist=90.0, max_age_s=1.0):
        self.max_dist = max_dist
        self.max_age_s = max_age_s
        self.next_id = 1
        self.tracks = {}   # id -> (cx, cy, last_t)

    def update(self, boxes, t):
        """boxes: list of (x1, y1, x2, y2). Returns list of track ids."""
        cents = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        used = set()
        out = []
        for (cx, cy) in cents:
            best_id, best_d = None, self.max_dist
            for tid, (tx, ty, tt) in self.tracks.items():
                if tid in used or t - tt > self.max_age_s:
                    continue
                d = math.hypot(cx - tx, cy - ty)
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = (cx, cy, t)
            used.add(best_id)
            out.append(best_id)
        self.tracks = {tid: v for tid, v in self.tracks.items()
                       if t - v[2] <= self.max_age_s}
        return out


class SimScene:
    """Renders the store scene and publishes its detection stream."""

    def __init__(self, seconds=30.0):
        self.seconds = seconds
        self.shoppers = build_sim_shoppers()
        self.person_tracker = CentroidTracker()
        self.item_tracker = CentroidTracker()

    def _draw_store(self, img):
        img[:] = (52, 48, 44)                       # floor
        # aisle floor shading
        cv2.rectangle(img, (0, 150), (SIM_WIDTH, SIM_HEIGHT), (60, 56, 52), -1)
        # header band + sign
        cv2.rectangle(img, (0, 0), (SIM_WIDTH, 64), (30, 30, 34), -1)
        cv2.putText(img, "MEGA MART  -  AISLE VIEW", (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        # shelf units (top, middle, lower)
        for (sx1, sy1, sx2, sy2) in [(140, 128, 1140, 168),
                                    (140, 258, 1140, 298),
                                    (140, 418, 1140, 458)]:
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (96, 74, 52), -1)
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (150, 120, 84), 2)
            for px in range(sx1 + 18, sx2 - 10, 44):   # product blocks
                cv2.rectangle(img, (px, sy1 + 8), (px + 26, sy2 - 8),
                              (int(90 + (px * 7) % 120), int(120 + (px * 3) % 90),
                               int(70 + (px * 11) % 110)), -1)
        cv2.putText(img, "ENTRANCE", (36, 706), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2)
        cv2.putText(img, "EXIT", (1180, 706), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2)

    def _figure_geometry(self, shopper: SimShopper, t: float):
        """Drawn extent of a shopper: returns (person_box, bag_box|None)."""
        x, y, active = shopper.pos(t)
        if not active:
            return None, None
        # gentle walking bob
        moving = any(t0 <= t <= t1 and (x0, y0) != (x1, y1)
                     for (t0, t1, x0, y0, x1, y1) in shopper.segments)
        bob = math.sin(t * 9.0) * 2.0 if moving else 0.0
        raise_h = shopper.arm_raise(t) * 26.0        # raised arm extends the box
        x1 = x - shopper.width / 2.0
        x2 = x + shopper.width / 2.0
        y2 = y + bob
        y1 = y2 - shopper.height - raise_h
        bag = None
        if shopper.bag_visible(t):
            bx1 = x2 + 5
            bag = (bx1, y2 - 62.0, bx1 + 27.0, y2 - 30.0)
        return (x1, y1, x2, y2), bag

    def _draw_shopper(self, img, shopper: SimShopper, t: float):
        box, bag = self._figure_geometry(shopper, t)
        if box is None:
            return
        x1, y1, x2, y2 = box
        x = (x1 + x2) / 2.0
        head_r = 13
        head_cy = y1 + head_r + shopper.arm_raise(t) * 0  # head under raised arm
        # body
        cv2.rectangle(img, (int(x1 + 6), int(y1 + 2 * head_r)),
                      (int(x2 - 6), int(y2 - 26)), shopper.colour, -1)
        # legs
        step = int(math.sin(t * 9.0) * 7) if any(
            t0 <= t <= t1 and (a, b) != (c, d)
            for (t0, t1, a, b, c, d) in shopper.segments) else 0
        cv2.line(img, (int(x - 8), int(y2 - 26)), (int(x - 8 + step), int(y2)),
                 shopper.colour, 6)
        cv2.line(img, (int(x + 8), int(y2 - 26)), (int(x + 8 - step), int(y2)),
                 shopper.colour, 6)
        # head
        cv2.circle(img, (int(x), int(head_cy + 2 * head_r - head_r)),
                   head_r, (210, 180, 160), -1)
        # arm: raised toward the shelf when reaching
        shoulder = (int(x + shopper.width / 2 - 6), int(y1 + 2 * head_r + 8))
        lift = shopper.arm_raise(t)
        hand = (int(shoulder[0] + 10 + 14 * lift),
                int(shoulder[1] - 44 * lift + 22 * (1 - lift)))
        cv2.line(img, shoulder, hand, shopper.colour, 6)
        # carried bag
        if bag is not None:
            bx1, by1, bx2, by2 = bag
            cv2.rectangle(img, (int(bx1), int(by1)), (int(bx2), int(by2)),
                          (40, 40, 160), -1)
            cv2.rectangle(img, (int(bx1), int(by1)), (int(bx2), int(by2)),
                          (20, 20, 90), 2)

    def frame(self, t: float):
        """Returns (image, [Detection, ...]) for sim time t."""
        img = np.zeros((SIM_HEIGHT, SIM_WIDTH, 3), dtype=np.uint8)
        self._draw_store(img)
        person_boxes, item_boxes, item_labels = [], [], []
        for shopper in self.shoppers:
            box, bag = self._figure_geometry(shopper, t)
            if box is None:
                continue
            person_boxes.append(box)
            if bag is not None:
                item_boxes.append(bag)
                item_labels.append("handbag")
        p_ids = self.person_tracker.update(person_boxes, t)
        i_ids = self.item_tracker.update(item_boxes, t)
        for shopper in self.shoppers:
            self._draw_shopper(img, shopper, t)
        dets = []
        for box, tid in zip(person_boxes, p_ids):
            dets.append(Detection(box[0], box[1], box[2], box[3],
                                  "person", 0.99, tid))
        for box, tid, lab in zip(item_boxes, i_ids, item_labels):
            dets.append(Detection(box[0], box[1], box[2], box[3],
                                  lab, 0.95, tid))
        return img, dets



# ---------------------------------------------------------------------------
# Risk engine
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    tid: int
    first_seen: float
    last_seen: float
    box: tuple = (0, 0, 0, 0)
    score: float = 0.0
    reasons: list = field(default_factory=list)      # ordered, de-duplicated
    # zone / dwell
    zone: tuple | None = None
    zone_since: float = 0.0
    dwell_now: float = 0.0                            # current zone-stay length
    max_dwell: float = 0.0
    dwell_fired_zone: tuple | None = None
    dwell_extra_counted: int = 0
    # centroid history for speed: deque of (t, cx, cy)
    hist: deque = field(default_factory=lambda: deque(maxlen=90))
    # reach detection
    heights: deque = field(default_factory=lambda: deque(maxlen=90))  # (t, h)
    reach_phase: str = "idle"                         # idle | raised
    raise_t: float = 0.0
    reach_cooldown_until: float = 0.0
    reaches_in_zone: int = 0
    reach_fired: bool = False
    reach_extra_counted: int = 0
    # carried item
    carrying: bool = False
    item_label: str = ""
    item_first_t: float | None = None   # first association in this episode
    item_last_t: float | None = None    # most recent association
    # flags
    flag_fired: bool = False
    finalised: bool = False

    def level(self) -> str:
        if self.score >= FLAG_SCORE:
            return "RISK"
        if self.score >= WATCH_SCORE:
            return "WATCH"
        return "LOW"

    def add_reason(self, label: str):
        if label not in self.reasons:
            self.reasons.append(label)


class EventLog:
    def __init__(self, path: str | None):
        self.path = path
        self.events = []            # (t, tid, type, detail)
        self.counts = {}
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("Mall Shoplifting-Detection -- behaviour-risk event log\n")
                fh.write("Flags are heuristic risk signals for staff review, "
                         "NOT proof of theft.\n\n")

    def add(self, t: float, tid: int, etype: str, detail: str):
        line = f"[{t:7.1f}s] ID {tid:>3}  {etype:<14} {detail}"
        print(line, flush=True)
        self.events.append((t, tid, etype, detail))
        self.counts[etype] = self.counts.get(etype, 0) + 1
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class RiskEngine:
    def __init__(self, frame_w: int, frame_h: int, log: EventLog,
                 dwell_limit: float = DWELL_LIMIT_S):
        self.fw, self.fh = frame_w, frame_h
        self.log = log
        self.dwell_limit = dwell_limit
        self.tracks: dict[int, TrackState] = {}
        self.all_ids: set[int] = set()
        self.max_dwell_overall = 0.0
        self.max_score_overall = 0.0

    # -- helpers -----------------------------------------------------------
    def zone_of(self, cx: float, cy: float) -> tuple:
        col = min(ZONE_COLS - 1, max(0, int(cx / (self.fw / ZONE_COLS))))
        row = min(ZONE_ROWS - 1, max(0, int(cy / (self.fh / ZONE_ROWS))))
        return (col, row)

    @staticmethod
    def zone_name(zone: tuple) -> str:
        return f"{chr(65 + zone[1])}{zone[0] + 1}"

    def speed_at(self, st: TrackState, t_now: float, back: float = 0.4) -> float:
        """Centroid speed (px/s) between ~`back` seconds ago and the latest point."""
        if len(st.hist) < 2:
            return 0.0
        t1, x1, y1 = st.hist[-1]
        past = st.hist[0]
        for item in st.hist:
            if item[0] <= t1 - back:
                past = item
            else:
                break
        dt = t1 - past[0]
        if dt <= 1e-6:
            return 0.0
        return math.hypot(x1 - past[1], y1 - past[2]) / dt

    def _check_flag(self, st: TrackState, t: float):
        if st.score >= FLAG_SCORE and not st.flag_fired:
            st.flag_fired = True
            self.log.add(t, st.tid, "RISK FLAG",
                         "score %.0f -- reasons: %s"
                         % (st.score, ", ".join(st.reasons) or "n/a"))
        self.max_score_overall = max(self.max_score_overall, st.score)

    # -- per-signal updates --------------------------------------------------
    def _update_dwell(self, st: TrackState, t: float, zone: tuple):
        if st.zone != zone:
            st.zone = zone
            st.zone_since = t
            st.dwell_fired_zone = None
            st.dwell_extra_counted = 0
            st.reaches_in_zone = 0
            st.reach_fired = False
            st.reach_extra_counted = 0
        st.dwell_now = t - st.zone_since
        st.max_dwell = max(st.max_dwell, st.dwell_now)
        self.max_dwell_overall = max(self.max_dwell_overall, st.dwell_now)
        if st.dwell_now >= self.dwell_limit:
            if st.dwell_fired_zone != zone:
                st.dwell_fired_zone = zone
                st.score += DWELL_POINTS
                st.add_reason("LONG DWELL")
                self.log.add(t, st.tid, "LONG DWELL",
                             "zone %s, dwell %.1fs (+%d)"
                             % (self.zone_name(zone), st.dwell_now, DWELL_POINTS))
            extra = int(st.dwell_now - self.dwell_limit)
            add = min(extra, DWELL_EXTRA_CAP) - st.dwell_extra_counted
            if add > 0:
                st.score += add * DWELL_EXTRA_PER_S
                st.dwell_extra_counted += add

    def _update_reach(self, st: TrackState, t: float):
        if not st.heights:
            return
        # baseline: median height over the window ending 0.3 s ago
        past = [h for (tt, h) in st.heights if tt <= t - 0.3]
        if len(past) < 5:
            return
        baseline = float(np.median(past))
        h_now = st.heights[-1][1]
        speed = self.speed_at(st, t, back=0.5)
        if speed > REACH_STILL_SPEED:
            st.reach_phase = "idle"
            return
        if st.reach_phase == "idle":
            if t >= st.reach_cooldown_until and h_now >= baseline * (1 + REACH_RISE):
                st.reach_phase = "raised"
                st.raise_t = t
        elif st.reach_phase == "raised":
            held = t - st.raise_t
            if h_now <= baseline * 1.05 and held >= REACH_MIN_S:
                st.reach_phase = "idle"
                st.reach_cooldown_until = t + 0.8
                st.reaches_in_zone += 1
                if st.reaches_in_zone == REACH_TRIGGER:
                    st.reach_fired = True
                    st.score += REACH_POINTS
                    st.add_reason("REPEATED REACH")
                    self.log.add(t, st.tid, "REPEATED REACH",
                                 "x%d in zone %s (+%d)"
                                 % (st.reaches_in_zone,
                                    self.zone_name(st.zone), REACH_POINTS))
                elif st.reaches_in_zone > REACH_TRIGGER:
                    add = min(REACH_EXTRA_POINTS, REACH_EXTRA_CAP
                              - st.reach_extra_counted * REACH_EXTRA_POINTS)
                    if add > 0:
                        st.score += add
                        st.reach_extra_counted += 1
            elif held > REACH_SETTLE_S:
                st.reach_phase = "idle"     # never settled -- not a reach

    def _update_item(self, st: TrackState, t: float, item_label: str | None):
        # Span-based: an item counts as "carried" once it has been associated
        # with the person over a span of >= ITEM_STABLE_S (brief detector
        # flicker inside the span is tolerated).  It counts as vanished when
        # the person is still being tracked but no association has occurred
        # for ITEM_GONE_S.
        if item_label is not None:
            if st.item_first_t is None or (
                    st.item_last_t is not None
                    and t - st.item_last_t > ITEM_GONE_S):
                st.item_first_t = t            # new carry episode
            st.item_last_t = t
            st.item_label = item_label
            if t - st.item_first_t >= ITEM_STABLE_S:
                st.carrying = True
        else:
            if st.item_last_t is not None and t - st.item_last_t >= ITEM_GONE_S:
                if st.carrying:
                    st.carrying = False
                    st.score += ITEM_POINTS
                    st.add_reason("ITEM VANISHED")
                    self.log.add(t, st.tid, "ITEM VANISHED",
                                 "%s no longer visible while person stays "
                                 "in frame (+%d)" % (st.item_label, ITEM_POINTS))
                st.item_first_t = None
                st.item_last_t = None

    # -- track lifecycle -----------------------------------------------------
    def _finalise(self, st: TrackState, t: float, stream_end: bool):
        st.finalised = True
        if stream_end:
            return
        speed = self.speed_at(st, st.last_seen, back=0.4)
        if st.score >= EXIT_MIN_SCORE and speed >= EXIT_SPEED:
            st.score += EXIT_POINTS
            st.add_reason("RAPID EXIT")
            self.log.add(t, st.tid, "RAPID EXIT",
                         "left scene at %.0f px/s with score %.0f (+%d)"
                         % (speed, st.score - EXIT_POINTS, EXIT_POINTS))
            self._check_flag(st, t)

    def update(self, dets: list, t: float):
        persons = [d for d in dets if d.cls == "person" and d.track_id >= 0]
        items = [d for d in dets if d.cls != "person"]
        seen = set()
        for d in persons:
            seen.add(d.track_id)
            st = self.tracks.get(d.track_id)
            if st is None:
                st = TrackState(tid=d.track_id, first_seen=t, last_seen=t)
                self.tracks[d.track_id] = st
                self.all_ids.add(d.track_id)
            st.last_seen = t
            st.box = (d.x1, d.y1, d.x2, d.y2)
            st.hist.append((t, d.cx, d.cy))
            st.heights.append((t, d.h))
            zone = self.zone_of(d.cx, d.cy)
            self._update_dwell(st, t, zone)
            self._update_reach(st, t)
            # associate nearest carried item whose centre sits in/near the box
            best, best_dist = None, None
            for it in items:
                ex1, ey1 = d.x1 - 0.5 * d.w, d.y1 - 0.15 * d.h
                ex2, ey2 = d.x2 + 0.5 * d.w, d.y2 + 0.15 * d.h
                if ex1 <= it.cx <= ex2 and ey1 <= it.cy <= ey2 or iou(d, it) > 0.03:
                    dist = math.hypot(it.cx - d.cx, it.cy - d.cy)
                    if best_dist is None or dist < best_dist:
                        best, best_dist = it, dist
            self._update_item(st, t, best.cls if best else None)
            self._check_flag(st, t)
        # finalise tracks that have gone missing
        for tid, st in self.tracks.items():
            if tid not in seen and not st.finalised and t - st.last_seen > TRACK_LOST_S:
                self._finalise(st, t, stream_end=False)

    def finish(self, t: float):
        for st in self.tracks.values():
            if not st.finalised:
                self._finalise(st, t, stream_end=True)

    def active(self):
        return [st for st in self.tracks.values() if not st.finalised]

    def summary(self) -> dict:
        return {
            "people": len(self.all_ids),
            "events": dict(self.log.counts),
            "risk_flags": self.log.counts.get("RISK FLAG", 0),
            "max_dwell": self.max_dwell_overall,
            "max_score": self.max_score_overall,
        }



# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

LEVEL_COLOURS = {"LOW": COLOUR_LOW, "WATCH": COLOUR_WATCH, "RISK": COLOUR_FLAG}


def draw_overlay(img, engine: RiskEngine, dets: list, t: float,
                 source_label: str, frames_done: int):
    h, w = img.shape[:2]
    # faint zone grid
    for c in range(1, ZONE_COLS):
        x = int(c * w / ZONE_COLS)
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
    for r in range(1, ZONE_ROWS):
        y = int(r * h / ZONE_ROWS)
        cv2.line(img, (0, y), (w, y), (255, 255, 255), 1)
    grid = np.zeros_like(img)
    cv2.addWeighted(img, 0.97, grid, 0.03, 0, img)

    # item boxes (thin, cyan)
    person_boxes = {d.track_id: d for d in dets if d.cls == "person"}
    for d in dets:
        if d.cls != "person":
            cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          COLOUR_ITEM, 1)
            cv2.putText(img, d.cls, (int(d.x1), max(12, int(d.y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_ITEM, 1)

    # person boxes coloured by risk level
    flagged = []
    for tid, d in person_boxes.items():
        st = engine.tracks.get(tid)
        if st is None:
            continue
        colour = LEVEL_COLOURS[st.level()]
        thick = 3 if st.level() == "RISK" else 2
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                      colour, thick)
        label = "ID %d  RISK %.0f  %.1fs" % (tid, st.score, st.dwell_now)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ly = max(th + 6, int(d.y1) - 6)
        cv2.rectangle(img, (int(d.x1), ly - th - 6),
                      (int(d.x1) + tw + 8, ly + 2), colour, -1)
        cv2.putText(img, label, (int(d.x1) + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 2)
        if st.carrying:
            cv2.putText(img, "carrying: %s" % st.item_label,
                        (int(d.x1), int(d.y2) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_ITEM, 1)
        if st.level() == "RISK":
            flagged.append(st)

    # header bar
    cv2.rectangle(img, (0, 0), (w, 44), (15, 15, 15), -1)
    active = engine.active()
    header = ("Mall Risk Monitor  |  %s  |  t=%.1fs  |  people now: %d  |  "
              "tracked: %d  |  events: %d"
              % (source_label, t, len(active), len(engine.all_ids),
                 len(engine.log.events)))
    cv2.putText(img, header, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                COLOUR_TEXT, 2)

    # alert banner
    if flagged:
        worst = max(flagged, key=lambda s: s.score)
        cv2.rectangle(img, (0, 44), (w, 84), (0, 40, 200), -1)
        banner = ("!! RISK FLAG -- ID %d  score %.0f -- %s  "
                  "(heuristic flag for staff review, not proof of theft)"
                  % (worst.tid, worst.score, ", ".join(worst.reasons)))
        cv2.putText(img, banner, (12, 71), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    COLOUR_TEXT, 2)

    # legend
    legend = "GREEN low   AMBER watch   RED risk flag   CYAN carried item"
    cv2.putText(img, legend, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                COLOUR_TEXT, 1)
    return img


# ---------------------------------------------------------------------------
# Real-mode detection source (YOLOv8 + ByteTrack)
# ---------------------------------------------------------------------------

class YoloSource:
    def __init__(self, weights: str, conf: float, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.tracker = tracker
        self.classes = [PERSON_CLASS] + sorted(ITEM_CLASSES)

    def detect(self, frame) -> list:
        results = self.model.track(frame, persist=True, tracker=self.tracker,
                                   classes=self.classes, conf=self.conf,
                                   verbose=False)
        dets = []
        if not results:
            return dets
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return dets
        names = results[0].names
        ids = boxes.id
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            label = "person" if cls_id == PERSON_CLASS else names[cls_id]
            tid = int(ids[i]) if ids is not None else -1
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            dets.append(Detection(x1, y1, x2, y2, label,
                                  float(boxes.conf[i]), tid))
        return dets


# ---------------------------------------------------------------------------
# File chooser (his standing delivery rule)
# ---------------------------------------------------------------------------

def pick_video_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                   ("All files", "*.*")])
    root.destroy()
    return path or None


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(args) -> dict:
    log = EventLog(args.log)
    writer = None
    frames_done = 0
    engine = None
    t = 0.0

    if args.simulate:
        scene = SimScene(seconds=args.seconds)
        fps = float(SIM_FPS)
        engine = RiskEngine(SIM_WIDTH, SIM_HEIGHT, log,
                            dwell_limit=args.dwell)
        total = int(args.seconds * SIM_FPS)
        source_label = "SIMULATION"
        cap = None
        yolo = None
    else:
        source = args.source
        if source is None or (isinstance(source, str) and source.lower() == "pick"):
            source = pick_video_file()
            if not source:
                print("No video selected -- exiting.")
                return {}
        try:
            source = int(source)
            source_label = f"WEBCAM {source}"
        except (TypeError, ValueError):
            source_label = os.path.basename(str(source))
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Could not open source: {source}")
            return {}
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        engine = RiskEngine(fw, fh, log, dwell_limit=args.dwell)
        yolo = YoloSource(args.weights, args.conf, args.tracker)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Mall Shoplifting-Detection -- behaviour-risk flagging")
    print("Flags are heuristics for staff review, NOT proof of theft.")
    print(f"Source: {source_label} | dwell limit {args.dwell:.0f}s | "
          f"output {args.output} | log {args.log}")

    start = time.time()
    try:
        while True:
            if args.simulate:
                if frames_done >= total:
                    break
                t = frames_done / fps
                frame, dets = scene.frame(t)
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                t = frames_done / fps
                if args.max_seconds and t >= args.max_seconds:
                    break
                dets = yolo.detect(frame)

            engine.update(dets, t)
            draw_overlay(frame, engine, dets, t, source_label, frames_done)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            frames_done += 1

            if not args.no_display:
                cv2.imshow("Mall Shoplifting-Detection (Q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    print("Quit by user.")
                    break
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    engine.finish(t)
    s = engine.summary()
    elapsed = time.time() - start
    print("\n--- END-OF-RUN SUMMARY ---")
    print(f"Frames processed : {frames_done} ({elapsed:.1f}s wall clock)")
    print(f"People tracked   : {s['people']}")
    print(f"RISK FLAGs       : {s['risk_flags']}")
    for etype in ("LONG DWELL", "REPEATED REACH", "ITEM VANISHED", "RAPID EXIT"):
        print(f"  {etype:<15}: {s['events'].get(etype, 0)}")
    print(f"Longest dwell    : {s['max_dwell']:.1f}s")
    print(f"Highest score    : {s['max_score']:.0f}")
    print(f"Annotated video  : {args.output}")
    print(f"Event log        : {args.log}")
    if log.path:
        with open(log.path, "a", encoding="utf-8") as fh:
            fh.write("\nSUMMARY: frames=%d people=%d risk_flags=%d events=%s "
                     "max_dwell=%.1fs max_score=%.0f\n"
                     % (frames_done, s["people"], s["risk_flags"], s["events"],
                        s["max_dwell"], s["max_score"]))
    return {"frames": frames_done, **s}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Mall behaviour-risk flagging (loss-prevention heuristics)")
    ap.add_argument("--source", default=None,
                    help="video path, webcam index, or 'pick' for the file "
                         "chooser (default: file chooser)")
    ap.add_argument("--simulate", action="store_true",
                    help="run the rendered simulation scene instead")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="simulation length in seconds (default 30)")
    ap.add_argument("--output", default="output.mp4",
                    help="annotated output video (default output.mp4)")
    ap.add_argument("--log", default="events.log",
                    help="event log path (default events.log)")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="YOLOv8 weights (default yolov8n.pt, auto-downloaded "
                         "by Ultralytics on first use)")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="detection confidence threshold (default 0.30)")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="ByteTrack config (default bytetrack.yaml; "
                         "bytetrack_mall.yaml keeps IDs through crowd "
                         "occlusions longer)")
    ap.add_argument("--dwell", type=float, default=DWELL_LIMIT_S,
                    help="dwell limit in seconds (default 10)")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop real mode after this many seconds (0 = full)")
    ap.add_argument("--no-display", action="store_true",
                    help="do not open a display window (headless runs)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
