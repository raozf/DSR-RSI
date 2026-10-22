"""Random profiles around the preserved PhysicalRSI doll controller."""
import json
import os
from pathlib import Path
import numpy as np
from sort_nesting_dolls.model import PACKAGE
from sort_nesting_dolls.nesting_rgb.model import Model as ParentBatch, EpisodePolicy
from sort_nesting_dolls.nesting_rgb.perception import Reader as ParentReader
from sort_nesting_dolls.nesting_rgb.head_size import HeadSize
from .boundary import FreshNativeMixin


class RandomHeadSize(HeadSize):
    def __init__(self, families):
        super().__init__()
        self.families=[[np.array(p,float) for p in family['points']] for family in families]
        self.locked=False

    def assign(self, dolls):
        if self.locked:return super().assign(dolls)
        candidates=[]
        for points in self.families:
            self.geometry.points=points
            ordered,diag=super().assign(dolls)
            if ordered is not None:candidates.append((diag['mean_residual'],points,ordered,diag))
        if not candidates:return None,dict(valid=False,reason='no_random_profile_fit')
        candidates.sort(key=lambda x:x[0]);best=candidates[0]
        if any(c[3]['indices']!=best[3]['indices'] and c[0]-best[0]<.04 for c in candidates[1:]):
            return None,dict(valid=False,reason='ambiguous_random_size_order')
        self.geometry.points=best[1];self.locked=True
        return best[2],dict(best[3],profile_source='current_RGB_random_envelope',locked=True)


class RandomReader(ParentReader):
    def __init__(self, shared, families):
        self.__dict__.update(shared.__dict__)
        self.head_size=RandomHeadSize(families)

    def decode(self, results):
        result=super().decode(results)
        dolls=result['dolls']
        if len(dolls)==5:
            ordered,diag=self.head_size.assign(dolls)
            result.update(valid=ordered is not None,rank_uncertain=ordered is None,shape_assignment=diag,
                          failure=None if ordered is not None else 'random_metric_order_unconfirmed')
            if ordered is not None:
                for rank,doll in enumerate(ordered):
                    doll.update(rank=rank,calibrated_height_m=float(self.head_size.geometry.points[rank][:,2].max()))
                result['dolls']=ordered
        return result


class DollEpisode(EpisodePolicy):
    def __init__(self,cfg,reader):
        super().__init__(cfg,reader)
        self.geometry=reader.head_size.geometry

    def observe_target(self,xy,rank,gates=None):
        self.last_qualified.pop(rank,None)
        return super().observe_target(xy,rank,gates)


class IsolatedBatch(ParentBatch):
    def update_obs_batch(self,observations):
        for obs in observations:
            index=obs['env_idx']
            if index not in self.envs:self.envs[index]=self._new_episode(index)
            self.envs[index].update_obs(obs)


class DeferredReader:
    """Construction sentinel; the first real episode creates the shared GPU reader."""


class Model(FreshNativeMixin,IsolatedBatch):
    def __init__(self,cfg):
        from .integrity import verify
        root=verify(cfg)
        if cfg['task_name']!='sort_nesting_dolls_by_size_random':raise ValueError('random dolls required')
        cfg=dict(cfg);cfg['trace_dir']=str(Path(os.environ['RDJ_OUTPUT_DIR'])/'policy')
        Path(cfg['trace_dir']).mkdir(parents=True,exist_ok=True)
        os.environ['NESTING_TRACE_DIR']=cfg['trace_dir']
        os.environ['NESTING_DINO_MODEL']=str(PACKAGE/'assets/grounding-dino-tiny')
        self.families=json.loads((root/'public_bank/dolls.json').read_text())['families']
        # The parent eagerly moves DINO to CUDA. Keep descriptor loading CPU-only
        # and allocate that unchanged reader only after this policy receives an
        # observation. One parent reader remains shared across episode adapters.
        self._shared_reader=None
        super().__init__(cfg,reader=DeferredReader())

    def reset(self):
        super().reset();self._reset_fresh()

    def _new_episode(self,index):
        if self._shared_reader is None:self._shared_reader=ParentReader()
        return DollEpisode(self.cfg,RandomReader(self._shared_reader,self.families))
