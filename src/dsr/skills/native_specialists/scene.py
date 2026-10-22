"""Fresh semantic RGB measurements for the new task-specific controllers.

Object/part DINO proposals and public calibrated monocular depth are estimates,
not simulator measurements. Unresolved parts cause abstention. Every feature
is re-observed; no stale rigid propagation for cloth or articulated objects.
"""
from copy import deepcopy
from itertools import combinations
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from .catalog import QUERIES
from .rgb import RGBBackend, body_geometry, associate, cloud_frame, normalized
from .control import PolicyFailure, rotation, quaternion
from .instances import within_parent


def frame(axis, normal):
    z=normalized(normal);x=np.asarray(axis)-z*np.dot(axis,z);x=normalized(x)
    return Rotation.from_matrix(np.stack((x,np.cross(z,x),z),axis=1))


class SemanticScene(RGBBackend):
    def __init__(self,kind):
        super().__init__(kind)
        self.kind=kind
        self.perception_profile='default'
        self.supported_profiles=('default','independent_parts') if kind in ('laptop','toast') else ('default',)

    def detect(self,observation,previous):
        self.last_detection=dict(depth_calibration=None,objects={},profile=getattr(self,'perception_profile','default'))
        self.last_part_rejections=[]
        self.last_detection['part_rejections']=self.last_part_rejections
        rgb=np.asarray(observation['vision']['cam_head']['color'])[:,:,:3]
        try:world=self.metric_image(rgb)
        finally:self.last_detection['depth_calibration']=getattr(self,'last_depth_calibration',None)
        queries=QUERIES[self.kind];rows=self.proposals(rgb,queries);result={}
        if self.kind=='eggs':
            from .carton import select_base, well_geometry
            def summaries(proposals):
                return [dict(box=np.asarray(p['box']).tolist(),confidence=float(p['confidence']),
                             source=p.get('source','dino_box_grabcut'),
                             mask_pixels=int(np.count_nonzero(p['mask'])) if p.get('mask') is not None else None)
                        for p in proposals]
            lid_required=getattr(self,'require_carton_lid',False)
            diagnostic=self.last_detection['carton']=dict(lid_required=lid_required,base_queries=[])
            previous_base=next((o for o in previous.values() if o['kind']=='egg_holder'),None)
            retained_wells=(previous_base or {}).get('features',{}).get('wells',[])
            # Use a bounded prompt ensemble for the same visible objects. A
            # failed/overlapping prompt is recorded and does not cancel a
            # valid independently segmented prompt. No expected-count split
            # is performed here.
            egg_candidates=[];egg_diagnostics=[]
            for prompt in ('egg','eggs','white egg','oval egg'):
                try:
                    found=self.instances(rgb,[prompt],separate_components=True)[0]
                except PolicyFailure as error:
                    egg_diagnostics.append(dict(query=prompt,failure=error.reason,detail=str(error)))
                    continue
                egg_diagnostics.append(dict(query=prompt,accepted=len(found)))
                if len(found)<=4 and (previous or lid_required or len(found)==4):
                    egg_candidates.append((prompt,found))
            diagnostic['egg_instance_queries']=egg_diagnostics
            if not egg_candidates:
                raise PolicyFailure('egg_instances_unresolved',json.dumps(egg_diagnostics,sort_keys=True))
            # Independent prompts must agree on the four image regions. This
            # prevents selecting a different object set just because it has a
            # higher score.
            rows[0]=self._agree_prompt_sets(egg_candidates,'egg_prompt_identity_unresolved')
            diagnostic['egg_source']='sam3_prompt_ensemble'
            if lid_required:
                rows[2]=self.instances(rgb,['egg carton lid'])[0]
                # The completed fill stage owns its occupancy ledger. Closing
                # the lid naturally hides the wells; do not require empty holes.
                rows[1]=[]
                diagnostic['well_measurement']='not_requested_in_lid_stage'
            else:
                well_candidates=[];well_diagnostics=[]
                for prompt in ('empty round compartment in egg carton',
                               'round hole in egg carton','circular recess in egg carton',
                               'empty compartment'):
                    try:
                        found=self.instances(rgb,[prompt],separate_components=True)[0]
                    except PolicyFailure as error:
                        well_diagnostics.append(dict(query=prompt,failure=error.reason,detail=str(error)))
                        continue
                    selected=self._select_well_set(rgb,world,found,allow_partial=len(retained_wells)==4)
                    well_diagnostics.append(dict(query=prompt,accepted=len(found),
                        selected=len(selected) if selected else 0,
                        geometry=getattr(self,'last_well_selection',None)))
                    if len(selected)==4 or (len(retained_wells)==4 and (selected or not found)):
                        well_candidates.append((prompt,selected))
                diagnostic['well_queries']=well_diagnostics
                if not well_candidates:
                    # Keep the full measurement in durable diagnostics, not in
                    # the short failure repeated into every planner request.
                    summary=[{k:v for k,v in row.items() if k!='geometry'} for row in well_diagnostics]
                    raise PolicyFailure('egg_wells_unresolved',json.dumps(summary,sort_keys=True))
                prompt,_=max(well_candidates,key=lambda row:len(row[1]))
                proposals=self._agree_prompt_sets(well_candidates,'well_prompt_identity_unresolved')
                diagnostic['well_source']=prompt
                diagnostic.update(independent_wells=len(proposals),wells=summaries(proposals))
                # On recorded layout 13, the base noun gives a whole carton or
                # no instance; the tray noun gives its observed lower surface.
                # All three bounded paths use the SAME geometry/identity gate.
                # A conflicting identity or invalid mask is never bypassed.
                for query in (None,'egg carton base','egg tray'):
                    record=dict(query=query or 'dino:egg carton base')
                    diagnostic['base_queries'].append(record)
                    try:
                        if query is None:
                            bases=rows[1]
                        else:
                            # Both nouns query the same base. Apply current,
                            # independent well evidence before mask overlap
                            # rejection; never skip a surviving identity conflict.
                            self._carton_base_evidence=(proposals,previous_base)
                            try:bases=self.instances(rgb,[query])[0]
                            finally:del self._carton_base_evidence
                        record['proposals']=summaries(bases)
                        base=select_base(bases,proposals,previous_base)
                    except PolicyFailure as error:
                        record['failure']=error.reason
                        if error.reason!='carton_base_evidence_unresolved' or query=='egg tray':raise
                    else:
                        diagnostic.update(base_box=np.asarray(base['box']).tolist(),selected_query=record['query'])
                        break
                self.carton_wells=well_geometry(self,rgb,world,proposals)
                rows[1]=[base];rows[2]=[]
        for (kind,_,count),proposals in zip(queries,rows):
            detections=[];clouds={};rejected=[]
            for proposal in proposals:
                if kind=='lever' and self.kind=='toast':
                    parents=[o for o in result.values() if o['kind']=='toaster']
                    if len(parents)!=1 or not within_parent(proposal['box'],parents[0]['box']):continue
                try:
                    points,hist=self.cloud(rgb,world,proposal['box'],proposal.get('mask'))
                    centre,size,r=body_geometry(points,kind)
                except PolicyFailure as error:
                    # A bad detector proposal is not a failure of every other
                    # object in the image. Keep its rejection and continue to
                    # independently qualified proposals; count/identity gates
                    # below still reject missing or ambiguous required objects.
                    rejected.append(dict(reason=error.reason,detail=str(error),
                                         box=np.asarray(proposal['box']).tolist()))
                    continue
                obj=dict(kind=kind,position=centre.tolist(),size=size.tolist(),quaternion=quaternion(r),
                         confidence=proposal['confidence'],features={},appearance=hist.tolist(),box=proposal['box'].tolist())
                if 'mask' in proposal:obj['_instance_mask']=proposal['mask']
                obj['perception_source']=proposal.get('source','dino_box_grabcut')
                clouds[id(obj)]=points;detections.append(obj)
            try:pairs=associate(detections,previous,kind)
            except PolicyFailure as error:
                self.last_detection['association']=getattr(error,'association_evidence',
                    dict(kind=kind,reason=error.reason))
                raise
            for identity,obj,old in pairs:
                points=clouds[id(obj)]
                if old is not None and kind not in ('garment','laptop','lever'):
                    centre,size,r=body_geometry(points,kind,old)
                    obj.update(position=centre.tolist(),size=size.tolist(),quaternion=quaternion(r))
                obj['id']=identity
                # Occluding a slot with bread must not erase the toaster pose.
                # Missing features stay missing and are rejected at use time;
                # they are never filled from a previous frame.
                try:self.features(rgb,world,obj,points,old)
                except PolicyFailure as error:
                    obj['feature_error']=dict(reason=error.reason,detail=str(error))
                obj.pop('_instance_mask',None)
                result[identity]=obj
            actual=sum(o['kind']==kind for o in result.values())
            self.last_detection['objects'][kind]=dict(proposals=len(proposals),accepted=actual,expected_initial=count,
                rejected_proposals=rejected,
                feature_errors=[o['feature_error'] for o in result.values() if o['kind']==kind and 'feature_error' in o])
            required=getattr(self,'required_objects',None)
            initial_required=required is None or kind in required
            if not previous and initial_required and count is not None and actual!=count and not (self.kind=='eggs' and kind=='lid'):
                raise PolicyFailure('initial_semantic_count',f'{kind}:{actual}, expected {count}')
        return result

    @staticmethod
    def _agree_prompt_sets(candidates,reason):
        """Prompts may miss occluded objects; visible identities must agree."""
        prompt,selected=max(candidates,key=lambda row:len(row[1]))
        centres=lambda group:np.asarray([(np.asarray(p['box'][:2])+p['box'][2:])/2 for p in group])
        for other_prompt,other in candidates:
            if not other:continue
            distances=np.linalg.norm(centres(other)[:,None]-centres(selected)[None,:],axis=-1)
            left,right=linear_sum_assignment(distances)
            if np.max(distances[left,right])>12:
                raise PolicyFailure(reason,prompt+' vs '+other_prompt)
        return selected

    def _select_well_set(self,rgb,world,candidates,allow_partial=False):
        """Select four independently measured wells from a bounded proposal set.

        Candidates are accepted only when their own metric clouds are small,
        separated and coplanar. The expected four count chooses a geometric
        arrangement among already independent components; it never splits a
        connected mask or turns a base mask into wells.
        """
        diagnostic=self.last_well_selection=dict(candidates=len(candidates),
            measurements=[],groups_rejected={},valid_groups=0,selected=0)
        def reject(reason):
            counts=diagnostic['groups_rejected']
            counts[reason]=counts.get(reason,0)+1
        # Keep prompt search and geometric subset selection bounded.
        if len(candidates)>12:
            reject('candidate_limit');return []
        measured=[]
        for proposal in candidates:
            row=dict(box=np.asarray(proposal['box']).tolist())
            diagnostic['measurements'].append(row)
            try:points,_=self.cloud(rgb,world,proposal['box'],proposal.get('mask'))
            except PolicyFailure as error:
                row['rejected']=error.reason;continue
            extent=np.ptp(points,axis=0)
            centre=np.median(points,axis=0)
            row.update(extent=extent.tolist(),centre=centre.tolist())
            if min(extent[:2])<=.012 or max(extent)>.10:
                row['rejected']='well_extent';continue
            measured.append((proposal,centre))
        if len(measured)<4:
            # Partial visible wells are only useful with an existing four-well
            # rigid-frame ledger. features() still checks each measured identity.
            selected=[row[0] for row in measured] if allow_partial and len(measured)==len(candidates) else []
            diagnostic['selected']=len(selected)
            return selected
        valid=[]
        for group in combinations(measured,4):
            centres=np.asarray([row[1] for row in group])
            distances=np.linalg.norm(centres[:,None,:2]-centres[None,:,:2],axis=-1)
            if np.min(distances[np.triu_indices(4,1)])<.025:
                reject('well_separation');continue
            spread=np.linalg.svd(centres[:,:2]-centres[:,:2].mean(axis=0),compute_uv=False)
            if spread[1]<=.008:
                reject('well_planar_spread');continue
            if np.ptp(centres[:,2])>.025:
                reject('well_height_spread');continue
            footprint=np.prod(centres[:,:2].max(axis=0)-centres[:,:2].min(axis=0))
            score=(footprint,-sum(float(row[0].get('confidence',0)) for row in group))
            valid.append((score,[row[0] for row in group]))
        diagnostic['valid_groups']=len(valid)
        if not valid:return []
        valid.sort(key=lambda row:row[0]);best=valid[0]
        if len(valid)>1 and np.isclose(valid[1][0][0],best[0][0],rtol=.02,atol=.0005):
            reject('well_subset_identity');return []
        diagnostic['selected']=4
        return best[1]

    def features(self,rgb,world,obj,points,previous=None):
        kind=obj['kind'];f=obj['features'];r=rotation(obj['quaternion']);c=np.array(obj['position'])
        def parts(query,count=1):return self.part(rgb,world,obj,query,count)
        def point(query):return parts(query)[0]['point']
        def local(p):return r.inv().apply(p-c).tolist()
        def axis(query):
            p=parts(query)[0];v=cloud_frame(p['points']).apply([1,0,0])
            return p['point'],v,p['points']
        def grasp(query=None,contact=None):
            cloud=contact if contact is not None else points if query is None else parts(query)[0]['points']
            centre=np.median(cloud,axis=0)
            rot=cloud_frame(cloud);direction=rot.apply([1,0,0]);direction[2]=0
            direction=normalized(direction);yaw=np.arctan2(direction[1],direction[0])
            q=quaternion(Rotation.from_euler('z',yaw)*rotation([.5,-.5,.5,.5]))
            across=np.cross([0,0,1],direction);width=float(np.ptp(cloud@across))
            f.update(grasp_point=centre.tolist(),grasp=dict(quaternion=q,width_m=width,source='current_visible_contact_patch'))
        if kind in ('egg','block','bread','mug','headset','broom','car','brush','hammer','shoe','laptop','box'):
            query={'mug':'mug body','headset':'headphone headband','broom':'broom handle','hammer':'hammer handle',
                   'brush':'toothbrush handle','shoe':'shoe heel','laptop':'closed laptop edge','box':'cardboard box rim'}.get(kind)
            # Laptop must be closed before this candidate attempts its grasp.
            if kind!='laptop' and not (kind=='headset' and self.perception_profile=='independent_parts'):grasp(query)
        if kind=='egg_holder':
            try:wells=self.carton_wells if self.kind=='eggs' else parts('egg well',None)
            except PolicyFailure:
                if previous is None or not previous.get('features',{}).get('wells'):raise
                wells=[]
            ordered=sorted(wells,key=lambda p:tuple(np.round(r.inv().apply(p['point']-c),3)))
            previous_wells=(previous or {}).get('features',{}).get('wells',[])
            if len(previous_wells)!=4 and len(ordered)!=4:
                raise PolicyFailure('four_independent_wells_required')
            f['wells']=[p['point'].tolist() for p in ordered]
            if len(previous_wells)==4:
                # Well indices are geometry memory in the freshly observed
                # rigid carton frame. Filled/occluded wells are not new detections.
                old_r=rotation(previous['quaternion']);old_c=np.array(previous['position'])
                predicted=[c+r.apply(old_r.inv().apply(np.array(p)-old_c)) for p in previous_wells]
                used=set()
                for part in ordered:
                    distances=[np.linalg.norm(part['point']-p) for p in predicted]
                    i=int(np.argmin(distances))
                    if distances[i]>.018 or i in used:raise PolicyFailure('well_identity_unresolved')
                    used.add(i);predicted[i]=part['point']
                f['wells']=[p.tolist() for p in predicted]
                f['well_visibility']=['current_rgb' if i in used else 'rigid_frame_memory' for i in range(len(predicted))]
        elif kind=='lid':
            hinge,v,_=axis('egg carton hinge');edge=point('free edge of egg carton lid')
            targets=self.boxes(rgb,['front rim of egg carton base'])[0]
            if len(targets)!=1:raise PolicyFailure('carton_front_rim_unresolved')
            cloud,_=self.cloud(rgb,world,targets[0]['box']);target=np.median(cloud,axis=0)
            radial=edge-hinge;radial-=v*np.dot(v,radial)
            closing=target-hinge;closing-=v*np.dot(v,closing)
            angle=float(np.arctan2(np.dot(v,np.cross(radial,closing)),np.dot(radial,closing)))
            f.update(hinge=hinge.tolist(),hinge_axis=v.tolist(),free_edge=edge.tolist(),closing_angle_rad=angle)
        elif kind=='block':
            top=points[points[:,2]>np.quantile(points[:,2],.85)]
            f.update(support_point=np.median(top,axis=0).tolist(),support_normal=cloud_frame(top).apply([0,0,1]).tolist())
        elif kind=='garment':
            f['landmarks']={key:point(query).tolist() for key,query in {
                'left_sleeve':'left sleeve cuff','right_sleeve':'right sleeve cuff',
                'left_chest':'left chest of shirt','right_chest':'right chest of shirt',
                'left_hem':'left bottom hem corner','right_hem':'right bottom hem corner',
                'left_shoulder':'left shoulder seam','right_shoulder':'right shoulder seam'}.items()}
        elif kind=='box':
            opening=parts('inside opening')[0];floor=point('inside bottom surface')
            rot=cloud_frame(opening['points']);size=np.ptp(rot.inv().apply(opening['points']-opening['point']),axis=0)[:2]
            if min(size)<.04 or floor[2]>=opening['point'][2]:raise PolicyFailure('box_interior_unresolved')
            f.update(inner_floor=floor.tolist(),inner_size=size.tolist(),rim_height=float(opening['point'][2]),interior_quaternion=quaternion(rot))
        elif kind in ('car','brush','hammer','shoe'):
            front=point({'car':'front of car','brush':'brush head','hammer':'hammer head','shoe':'shoe toe'}[kind])
            back=point({'car':'rear of car','brush':'bottom end of toothbrush','hammer':'end of hammer handle','shoe':'shoe heel'}[kind])
            top=parts({'car':'roof of car','brush':'front face of toothbrush handle',
                       'hammer':'flat face of hammer head','shoe':'top of shoe'}[kind])[0]
            normal=cloud_frame(top['points']).apply([0,0,1])
            f.update(front_axis_local=r.inv().apply(normalized(front-back)).tolist(),
                     up_axis_local=r.inv().apply(normal).tolist())
        elif kind=='bread':
            # The thin face normal is geometric; no native asset-axis claim.
            face=parts('broad face of bread slice')[0]
            face_r=cloud_frame(face['points'])
            f.update(face_normal_local=r.inv().apply(face_r.apply([0,0,1])).tolist(),
                     face_long_local=r.inv().apply(face_r.apply([1,0,0])).tolist())
        elif kind=='mug':
            opening,normal,aperture=self.handle_aperture(rgb,world,obj)
            if np.dot(normal,opening-c)<0:normal=-normal
            rim=parts('mug rim')[0];up=cloud_frame(rim['points']).apply([0,0,1])
            f.update(handle_local=local(opening),handle_axis_local=r.inv().apply(normal).tolist(),
                     up_axis_local=r.inv().apply(up).tolist(),aperture_m=aperture)
        elif kind=='rack':
            supports=parts('cup holder supporting arm',None);f['supports']=[]
            for part in supports:
                cloud=part['points'];v=cloud_frame(cloud).apply([1,0,0]);v[2]=0;v=normalized(v)
                if np.dot(v,part['point']-c)<0:v=-v
                projections=cloud@v;lo,hi=np.quantile(projections,[.08,.92]);length=float(hi-lo)
                if length<.025:continue
                f['supports'].append(dict(point=part['point'].tolist(),axis=v.tolist(),length_m=length))
            if not f['supports']:raise PolicyFailure('no_visible_holder_support')
        elif kind=='toaster':
            f['slots']=[]
            for slot in sorted(parts('toaster slot',2),key=lambda p:tuple(r.inv().apply(p['point']-c))):
                axis=cloud_frame(slot['points']).apply([1,0,0]);axis[2]=0;axis=normalized(axis)
                q=quaternion(Rotation.from_matrix(np.stack((axis,[0,0,1],np.cross(axis,[0,0,1])),axis=1)))
                f['slots'].append(dict(point=slot['point'].tolist(),quaternion=q,reference='visible_opening_only'))
        elif kind=='lever':
            contact=point('lever handle');bottom=point('bottom of lever track');delta=bottom-contact
            length=float(np.linalg.norm(delta))
            if length>.10:raise PolicyFailure('lever_travel_unresolved')
            # At the latched end there can be almost no remaining travel.
            f.update(contact=contact.tolist(),bottom=bottom.tolist(),travel_m=length,
                     press_axis=(delta/max(length,1e-6)).tolist())
        elif kind=='headset':
            if self.perception_profile=='independent_parts':
                from .semantic_parts import headset_bridge
                bridge,bridge_axis,contact=headset_bridge(parts('headphone headband')[0],
                    parts('headphone ear cushions',2))
                grasp(contact=contact)
                f.update(bridge_local=local(bridge),bridge_axis_local=r.inv().apply(bridge_axis).tolist(),
                         bridge_source='independent_band_and_two_cushions')
            else:
                bridge=point('center of headphone headband bridge');left=point('left headphone ear cup');right=point('right headphone ear cup')
                f.update(bridge_local=local(bridge),bridge_axis_local=r.inv().apply(normalized(right-left)).tolist())
        elif kind=='stand':
            p,v,_=axis('headphone support cradle');f.update(cradle=p.tolist(),cradle_axis=v.tolist())
        elif kind=='laptop':
            hinge,axis_v,_=axis('laptop hinge')
            try:
                edge=point('top edge of laptop screen');base_edge=point('front edge of laptop keyboard base')
            except PolicyFailure:
                # Closing hides the keyboard. Require two separately visible
                # physical edges, rather than carrying a previous hinge angle.
                edge=point('front edge of closed laptop lid');base_edge=point('front edge of laptop base')
                if np.linalg.norm(edge-base_edge)>.025:raise PolicyFailure('closed_laptop_edges_separated')
            radial=edge-hinge;radial-=axis_v*np.dot(axis_v,radial)
            target=base_edge-hinge;target-=axis_v*np.dot(axis_v,target)
            angle=float(np.arctan2(np.dot(axis_v,np.cross(radial,target)),np.dot(radial,target)))
            f.update(hinge=hinge.tolist(),hinge_axis=axis_v.tolist(),free_edge=edge.tolist(),closing_angle_rad=angle,
                     bottom_local=local(hinge),hinge_axis_local=r.inv().apply(axis_v).tolist(),
                     storage_axis_local=r.inv().apply(normalized(base_edge-hinge)).tolist())
            if abs(angle)<.15:grasp('closed laptop edge')
        elif kind=='laptop_rack':
            p,v,cloud=axis('laptop rack slot opening')
            normal=cloud_frame(cloud).apply([0,0,1])
            if normal[2]<np.cos(np.deg2rad(10)):raise PolicyFailure('laptop_rack_opening_tilt')
            v[2]=0;v=normalized(v);across=np.cross([0,0,1],v)
            f.update(opening=p.tolist(),slot_axis=v.tolist(),slot_across=across.tolist(),
                     slot_width_m=float(np.ptp((cloud-p)@across)))
        elif kind=='broom':
            tip=point('bottom of broom brush');brush=parts('broom brush head')[0];handle=parts('broom handle')[0]
            v=cloud_frame(handle['points']).apply([1,0,0]);cloud=handle['points'];proj=(cloud-handle['point'])@v
            # Resolve the far handle end from the currently visible brush.
            if np.dot(v,handle['point']-tip)<0:v=-v;proj=-proj
            other=handle['point']+v*np.quantile(proj,.85)
            brush_axis=cloud_frame(brush['points']).apply([1,0,0])
            brush_up=normalized(handle['point']-tip)
            f.update(tip_local=local(tip),handover_local=local(other),
                     brush_axis_local=r.inv().apply(brush_axis).tolist(),
                     brush_up_local=r.inv().apply(brush_up).tolist())
        elif kind=='pan':
            mouth,v,cloud=axis('dustpan front lip');floor=parts('dustpan inside bottom surface')[0]
            inward=normalized(floor['point']-mouth);inward[2]=0;inward=normalized(inward)
            f.update(mouth=mouth.tolist(),floor=floor['point'].tolist(),inward=inward.tolist(),
                     width_m=float(np.ptp(cloud@v)),normal=cloud_frame(floor['points']).apply([0,0,1]).tolist())
