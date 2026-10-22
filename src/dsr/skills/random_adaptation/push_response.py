"""Contact response in target-local metres (yaw scaled by object radius)."""
from dataclasses import dataclass
import numpy as np


@dataclass
class PushResponseModel:
    gain: float = 1.
    edge: str | None = None
    updates: int = 0
    no_response: int = 0
    valid: bool = False

    def observe(self, *, edge, command_delta, object_delta,
                support_tilt_rad=0., same_support=True):
        command, actual = np.asarray(command_delta, float), np.asarray(object_delta, float)
        self.valid = False
        if (command.shape != (3,) or actual.shape != (3,) or
                not np.isfinite([*command, *actual, support_tilt_rad]).all()):
            return 'invalid_observation'
        if not edge or not same_support or abs(support_tilt_rad) > np.deg2rad(8):
            return 'reobserve'
        if self.edge not in (None, edge):
            self.edge, self.updates, self.no_response = edge, 0, 0
            return 'reobserve'
        norm = float(np.linalg.norm(command))
        if not .002 <= norm <= .12:
            return 'reobserve'
        gain = float(np.dot(actual, command) / np.dot(command, command))
        transverse = float(np.linalg.norm(actual - gain * command))
        self.edge = edge
        if gain <= .10:
            self.no_response += 1
            return 'change_contact_or_stop'
        if gain > 2.5 or transverse > max(.003, norm * .35):
            return 'reobserve'
        self.gain = float(np.clip(gain, .5, 1.8))
        self.updates += 1
        self.no_response = 0
        self.valid = True
        return 'updated'

    def predict_command(self, desired_delta, *, edge, max_command=.018):
        desired = np.asarray(desired_delta, float)
        if (desired.shape != (3,) or not np.isfinite(desired).all() or not self.valid
                or self.edge != edge or not 0 < max_command <= .05):
            return None
        command = desired / self.gain
        norm = float(np.linalg.norm(command))
        if norm > max_command:
            command *= max_command / norm
        return command.tolist()
