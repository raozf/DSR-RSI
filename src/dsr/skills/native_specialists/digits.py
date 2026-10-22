"""Preserve PhysicalRSI glyph/grasp/path skills; require observed completion."""
import os
from pathlib import Path

from arrange_rgb.model import Model as ParentBatch
from arrange_rgb.single_env import Model as ParentEpisode, observe_relay
from .boundary import FreshNativeMixin


class DigitEpisode(ParentEpisode):
    def _memory_Model_finish_move_2(self, packet):
        if packet.get('returned'):
            return packet
        frame = packet['locals']
        if frame['placed'] is not None:
            return super()._memory_Model_finish_move_2(packet)
        # Retain the original source-local identity and full grasp/path gate.
        if self.attempts.get(self.rank, 0) < 2 and self.step < 900:
            source, evidence = observe_relay(self.reader, frame['rgb'], frame['digit'],
                frame['digit']['xy'], self.output / self.trace.stem / f'retry_{self.rank}')
            self.log(dict(event='source_after_attempt', rank=self.rank, **evidence))
            if source is not None:
                self.digits[self.rank] = source
                self.pending_relay = None
                self.choose_next()
                return dict(locals=frame, returned=True, result=None)
        self.stop('unconfirmed_digit_placement')
        return dict(locals=frame, returned=True, result=None)

    def skip_rank(self, reason):
        # A failed motion is not a completed rank. Stop before contaminating
        # the parent's occupancy/collision ledger with a false placement.
        self.stop('unconfirmed_digit_motion:' + reason)


class Model(FreshNativeMixin, ParentBatch):
    def __init__(self, cfg):
        if cfg['task_name'] not in ('arrange_largest_number', 'arrange_largest_number_random'):
            raise ValueError('digit task required')
        from .integrity import verify
        verify(cfg)
        super().__init__(cfg)

    def reset(self):
        super().reset()
        self._reset_fresh()

    def _memory_Model_update_obs_batch_0(self, packet):
        if packet.get('returned'):
            return packet
        self.latest_indices = []
        for obs in packet['locals']['observations']:
            index = obs['env_idx']
            if index not in self.models:
                model = DigitEpisode(self.model_cfg)
                model.skill_choice_pool = self.skill_choice_pool
                model.output = Path(os.environ['ARRANGE_TRACE_DIR']) / f'env_{index}'
                model.output.mkdir(parents=True, exist_ok=True)
                model.reset()
                self.models[index] = model
            self.models[index].update_obs(obs)
            self.latest_indices.append(index)
        return packet
