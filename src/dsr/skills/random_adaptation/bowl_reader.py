"""RGB-only public bowl registration and conservative episode-local identity."""
from copy import deepcopy
from functools import partial
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .support.control import PolicyFailure, rotation
from .support.registration import registration, matrix, pose, project, CAM_R, CAM_T


class ProjectedOcclusion:
    """Approximate depth gate from registered bowls; never evaluator depth."""
    def __init__(self, shapes, tracked, excluded):
        self.depth = np.full((480, 640), np.inf)
        for identity, item in tracked.items():
            if identity == excluded: continue
            t = matrix([*item.get('predicted_position', item['position']), *item['quaternion']])
            points = shapes[item['features']['profile_key']] @ t[:3,:3].T + t[:3,3]
            uv = np.rint(project(points)).astype(int)
            depth = -((points - CAM_T) @ CAM_R)[:,2]
            valid = (uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)&(depth>0)
            np.minimum.at(self.depth, (uv[valid,1], uv[valid,0]), depth[valid])
        # Dilation closes holes in the sampled public mesh projection.
        self.depth = cv2.erode(self.depth.astype(np.float32), np.ones((5,5),np.uint8))

    def hidden(self, world):
        uv=np.rint(project(world)).astype(int)
        valid=(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)
        depth=-((world-CAM_T)@CAM_R)[:,2]
        hidden=np.zeros(len(world),bool)
        hidden[valid]=depth[valid]>self.depth[uv[valid,1],uv[valid,0]]+.003
        return hidden


class RGBBowlScene:
    def __init__(self, bank, *, segment=None, fitter=None):
        root=Path(bank)
        self.bank=json.loads((root/'bank.json').read_text())['entries']
        self.shapes=dict(np.load(root/'bowl_shapes.npz'))
        if set(self.bank)!=set(self.shapes) or set(self.bank)!={f'bowl{i}' for i in range(11,16)}:
            raise ValueError('expected five audited public bowl shapes')
        self.meta=self.bank
        # Four small public bowls are intentionally treated as one RGB
        # equivalence class. Their silhouettes are below the registration
        # resolution, so choosing one by a sub-pixel residual would invent an
        # identity. The merged profile is conservative for every member.
        self.profile_groups={k:('bowl11' if float(v.get('radius',0))>.085 else 'small_bowl_12_15')
                             for k,v in self.bank.items()}
        self.group_profiles={}
        for group in set(self.profile_groups.values()):
            keys=[k for k,v in self.profile_groups.items() if v==group]
            rows=[self.bank[k]['profile'] for k in keys]
            # A union of all section breakpoints preserves the conservative
            # envelope under linear interpolation between section samples.
            grid=np.unique(np.concatenate([r['z'] for r in rows]))
            grid=grid[(grid>=max(r['z'][0] for r in rows))&(grid<=min(r['z'][-1] for r in rows))]
            self.group_profiles[group]=dict(
                z=grid.tolist(),
                inner_lower=np.min([np.interp(grid,r['z'],r['inner_lower']) for r in rows],axis=0).tolist(),
                outer_upper=np.max([np.interp(grid,r['z'],r['outer_upper']) for r in rows],axis=0).tolist(),
                floor_z=max(r['floor_z'] for r in rows), bottom_z=min(r['bottom_z'] for r in rows),
                rim_z=min(r['rim_z'] for r in rows),
                model_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())
        if segment is None:
            from rdj_pour.sam.sam3_http import segment_many_http
            segment=lambda rgb: segment_many_http(rgb,[('bowl',.5)],timeout_s=600)[0].masks
        self.segment=segment
        self.fit=fitter or partial(registration,shapes=self.shapes,meta=self.meta)

    @staticmethod
    def accepted(fit):
        t=np.asarray(fit['pose'])
        return (t.shape==(4,4) and np.isfinite(t).all()
                and np.isfinite([fit['error_px'],fit['iou']]).all()
                and 0<=fit['error_px']<=4 and .45<=fit['iou']<=1
                and fit.get('visible_model_points',100)>=80)

    def item(self, identity, fit):
        key=fit['key'];entry=self.bank[key];t=fit['pose']
        points=self.shapes[key];rim=points[points[:,2]>=entry['profile']['rim_z']-.003]
        if len(rim)<8:raise PolicyFailure('unresolved_rim')
        # Actual public-mesh rim point, on the nearer x side of the grasp arm.
        p=pose(t); side=-1 if p[0]<0 else 1
        contact=rim[np.argmax(side*rim[:,0])]
        grasp=t[:3,3]+t[:3,:3]@contact
        bbox=np.asarray(entry['bbox']);world=bbox@t[:3,:3].T+t[:3,3]
        group=self.profile_groups[key]
        return dict(id=identity,kind='bowl',position=p[:3].tolist(),quaternion=p[3:].tolist(),
                    size=np.ptp(world,axis=0).tolist(),confidence=float(np.clip(1-fit['error_px']/12,.3,1)),
                    features=dict(profile_key=key,profile_group=group,
                                  profile=deepcopy(self.group_profiles[group]),
                                  rim_grasp=grasp.tolist(),yaw_symmetric=True,
                                  fit_error_px=fit['error_px'],fit_iou=fit['iou']))

    def detect(self, observation, previous):
        rgb=np.asarray(observation['vision']['cam_head']['color'],dtype=np.uint8)
        if rgb.shape not in ((480,640,3),(480,640,4)):raise PolicyFailure('camera_calibration_shape')
        rgb=rgb[:,:,:3]
        try:segmented=self.segment(rgb)
        except (RuntimeError,OSError,TimeoutError) as error:
            raise PolicyFailure('bowl_segmentation_unavailable',str(error)) from error
        masks=[]
        for raw in segmented:
            m=np.asarray(raw)
            if m.shape!=rgb.shape[:2]:raise PolicyFailure('invalid_bowl_mask_shape')
            if not np.isfinite(m).all():raise PolicyFailure('invalid_bowl_mask_values')
            m=m.astype(bool)
            if np.count_nonzero(m)>=80:masks.append(m)
        # Remove nearly duplicated SAM masks; retain separate nearby bowls.
        unique=[]
        for m in masks:
            if not any(np.count_nonzero(m&n)/max(1,np.count_nonzero(m|n))>.9 for n in unique):unique.append(m)
        masks=unique
        if not previous:
            if len(masks)!=3:raise PolicyFailure('initial_bowl_count')
            alternatives=[]
            for key in sorted(self.bank):
                try:fits=[self.fit(mask,key) for mask in masks]
                except (ValueError,KeyError,np.linalg.LinAlgError):continue
                if all(self.accepted(f) for f in fits):
                    positions=np.array([f['pose'][:3,3] for f in fits])
                    if min(np.linalg.norm(positions[a]-positions[b]) for a,b in ((0,1),(0,2),(1,2)))>.055:
                        alternatives.append((sum(f['error_px'] for f in fits),key,fits))
            alternatives.sort(key=lambda x:x[0])
            if not alternatives:raise PolicyFailure('public_bowl_fit_rejected')
            if len(alternatives)>1 and alternatives[1][0]-alternatives[0][0]<.30:
                groups={self.profile_groups[a[1]] for a in alternatives if a[0]-alternatives[0][0]<.30}
                if len(groups)!=1:raise PolicyFailure('ambiguous_bowl_profile')
            fits=sorted(alternatives[0][2],key=lambda f:tuple(f['pose'][:2,3]))
            return {f'bowl:{i}':self.item(f'bowl:{i}',f) for i,f in enumerate(fits)}
        if len(previous)!=3:raise PolicyFailure('identity_ledger_count')
        if len(masks)!=3:raise PolicyFailure('occluded_bowl_identity_unknown')
        identities=sorted(previous);fits={};cost=np.full((3,len(masks)),1e6)
        for i,identity in enumerate(identities):
            item=previous[identity];key=item['features']['profile_key']
            prior=matrix([*item.get('predicted_position',item['position']),*item['quaternion']])
            visibility=ProjectedOcclusion(self.shapes,previous,identity)
            group=item['features'].get('profile_group',self.profile_groups[key])
            for j,mask in enumerate(masks):
                candidates=[]
                for candidate_key,candidate_group in self.profile_groups.items():
                    if candidate_group!=group:continue
                    try:fit=self.fit(mask,candidate_key,prior=prior,visibility=visibility)
                    except (ValueError,KeyError,np.linalg.LinAlgError):continue
                    distance=np.linalg.norm(fit['pose'][:3,3]-prior[:3,3])
                    if self.accepted(fit) and distance<=.045:
                        candidates.append((fit['error_px']+40*distance,fit))
                if candidates:
                    candidates.sort(key=lambda row:row[0])
                    cost[i,j],fits[i,j]=candidates[0]
        rows,cols=linear_sum_assignment(cost)
        if len(rows)!=3 or np.any(cost[rows,cols]>=1e6):raise PolicyFailure('bowl_tracking_rejected')
        best=float(cost[rows,cols].sum())
        # Reject ambiguous global association rather than silently swapping bowls.
        for i,j in zip(rows,cols):
            alternate=cost.copy();alternate[i,j]=1e6
            ar,ac=linear_sum_assignment(alternate)
            if float(alternate[ar,ac].sum())-best<.25:raise PolicyFailure('ambiguous_bowl_identity')
        return {identities[i]:self.item(identities[i],fits[i,j]) for i,j in zip(rows,cols)}
