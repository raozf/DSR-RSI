"""Head-RGB feature motion checked against observed robot vertical lift.

This is a conservative acquisition guard, not a native task-success predicate.
The fixed head-camera projection is supplied by the existing calibrated worker.
"""
from __future__ import annotations

import numpy as np


class VisualLiftGuard:
    def __init__(self, *, min_points=6, pixel_tolerance=7.0, max_missing=3):
        self.min_points = min_points
        self.pixel_tolerance = pixel_tolerance
        self.max_missing = max_missing
        self.reset()

    def reset(self):
        self.gray = self.points = self.origin_points = None
        self.initial_ee_z = None
        self.initial_world = None
        self.initial_pixel = None
        self.missing = 0
        self.hold_offsets = []
        self.last_sequence = -1
        self.status = "uninitialized"

    def initialize(self, rgb, box, *, ee_z, world_xyz, project, sequence):
        import cv2
        gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        x0, y0, x1, y1 = np.asarray(box, float)
        height, width = gray.shape
        # Leave a margin to avoid tracking the tabletop at the box boundary.
        dx, dy = (x1-x0)*0.15, (y1-y0)*0.15
        mask = np.zeros_like(gray)
        mask[max(0,int(y0+dy)):min(height,int(y1-dy)), max(0,int(x0+dx)):min(width,int(x1-dx))] = 255
        points = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.04,
                                       minDistance=3, mask=mask)
        self.gray = gray
        self.points = points
        self.origin_points = None if points is None else points.copy()
        self.initial_ee_z = float(ee_z)
        self.initial_world = np.asarray(world_xyz, float)
        self.initial_pixel = np.asarray(project(world_xyz[:2], world_z=world_xyz[2]), float)
        self.last_sequence = sequence
        self.status = "tracking" if points is not None and len(points) >= self.min_points else "uncertain"
        return self.status

    def observe(self, rgb, *, ee_z, project, sequence, holding=False):
        import cv2
        if sequence <= self.last_sequence:
            raise ValueError("Lift guard requires a fresh observation")
        self.last_sequence = sequence
        if self.status in {"failed", "uncertain", "uninitialized"}:
            return self.status
        gray = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        nxt, good, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None)
        if nxt is None:
            self.status = "uncertain"
            return self.status
        previous, back_good, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None)
        if previous is None:
            self.status = "uncertain"
            return self.status
        keep = good.reshape(-1).astype(bool) & back_good.reshape(-1).astype(bool)
        keep &= np.linalg.norm(previous.reshape(-1,2)-self.points.reshape(-1,2), axis=1) < 1.5
        if keep.sum() < self.min_points:
            self.missing += 1
            self.status = "uncertain"
            return self.status
        self.points = nxt[keep].reshape(-1,1,2)
        self.origin_points = self.origin_points[keep].reshape(-1,1,2)
        self.gray = gray
        offsets = self.points.reshape(-1,2) - self.origin_points.reshape(-1,2)
        measured = np.median(offsets, axis=0)
        coherent = np.linalg.norm(offsets-measured, axis=1) <= self.pixel_tolerance
        if coherent.sum() < self.min_points:
            self.status = "uncertain"
            return self.status
        delta_z = float(ee_z) - self.initial_ee_z
        target = self.initial_world.copy()
        target[2] += delta_z
        expected = np.asarray(project(target[:2], world_z=target[2])) - self.initial_pixel
        # A closed gripper with a stationary tabletop target must never pass.
        if delta_z >= 0.035 and (np.linalg.norm(measured-expected) > self.pixel_tolerance
                               or np.dot(measured, expected) <= 0):
            self.status = "failed"
            return self.status
        if holding:
            self.hold_offsets.append(measured)
            self.hold_offsets = self.hold_offsets[-3:]
            if len(self.hold_offsets) == 3:
                spread = np.max(np.ptp(np.asarray(self.hold_offsets), axis=0))
                self.status = "stable" if spread <= self.pixel_tolerance/2 else "failed"
        return self.status
