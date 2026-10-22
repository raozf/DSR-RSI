"""Cartesian skills compile into bounded linear moves, never simulator joint replay."""
import math
import numpy as np
from .geometry import rotation_distance, vector
from .model import PreparationError, Step


class Skills:
    def __init__(self, config):
        self.config = config
        self.poses = {side: config.park(side).copy() for side in config.arms}
        self.steps = []

    def move(self, side, pose, stage):
        goal = vector(pose, 6, stage)
        start = self.poses[side].copy()
        # Task fixtures use one calibrated grasp orientation per arm. Rotation with a
        # held cup/ruler needs additional swept-volume qualification and is rejected.
        if rotation_distance(start[3:], goal[3:]) > 0.01:
            raise PreparationError(stage+": use the same calibrated TCP orientation as park; in-air rotation is not qualified")
        stride = self.config.safety["max_step_mm"]-2*self.config.safety["position_tolerance_mm"]
        count = max(1, math.ceil(float(np.linalg.norm(goal[:3]-start[:3])) / stride))
        for i in range(1, count+1):
            p = start.copy()
            p[:3] = start[:3]+(goal[:3]-start[:3])*i/count
            p[3:] = goal[3:]
            self.steps.append(Step("move", stage, side, p.tolist()))
        self.poses[side] = goal.copy()

    def grip(self, side, opening, stage):
        self.steps.append(Step("gripper", stage, side, opening=opening))

    def checkpoint(self, stage, expected, instruction):
        self.steps.append(Step("checkpoint", stage, expected=expected, instruction=instruction))

    def above(self, pose):
        p = vector(pose, 6, "grasp/contact pose").copy()
        if p[2] >= self.config.safe_z:
            raise PreparationError("Task contact/grasp must be below safe_transit_z_mm")
        p[2] = self.config.safe_z
        return p

    def park(self, side, stage):
        p = self.poses[side].copy()
        p[2] = self.config.safe_z
        self.move(side, p, stage+".lift")
        target = self.config.park(side).copy()
        at_height = target.copy()
        at_height[2] = self.config.safe_z
        self.move(side, at_height, stage+".transit")
        self.move(side, target, stage+".park")

    def transfer(self, side, source_tcp, target_tcp, object_label, stage):
        self.grip(side, 1.0, stage+".open")
        self.move(side, self.above(source_tcp), stage+".approach")
        self.move(side, source_tcp, stage+".descend")
        self.grip(side, 0.0, stage+".grasp")
        self.checkpoint(stage+".grasp_check", {"kind": "grasp", "object": object_label},
                        f"Confirm {side} gripper holds {object_label}; blocks remain in place.")
        self.move(side, self.above(source_tcp), stage+".lift")
        self.move(side, self.above(target_tcp), stage+".carry")
        self.move(side, target_tcp, stage+".place")
        self.grip(side, 1.0, stage+".release")
        self.move(side, self.above(target_tcp), stage+".retract")
        self.park(side, stage+".return")
