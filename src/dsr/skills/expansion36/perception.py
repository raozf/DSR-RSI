"""RGB-only development perception for the expansion controllers.

Grounding DINO supplies object/part proposals; monocular metric depth supplies
geometry, with the existing public head-camera calibration. No simulator pose,
segmentation, functional-point or reward channel is read. Semantic landmarks
must be independently detected: a PCA axis is never called a semantic front.
These estimates are unqualified until native validation, especially small parts.
"""
from copy import deepcopy
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .catalog import POLICIES
from .control import PolicyFailure, quaternion, rotation

# Public object vocabularies, not layout instance labels or hidden category IDs.
QUERIES = {
    "arrange_digits": [("digit", "number shaped object"), ("pad", "round wooden mat")],
    "build_tower": [("tower_support", "wooden cylindrical block"),
                    ("tower_board", "wooden rectangular board"),
                    ("tower_crown", "wooden arch block"), ("tower_roof", "wooden triangular block")],
    "classify_category": [("basket", "basket"), ("item", "toy car . action camera . pen . watch . tool")],
    "classify_language": [("basket", "basket"), ("item", "watch . toy car . wooden toy . pepper . chocolate bar")],
    "stack_blocks": [("block", "cube block")],
    "stack_bowls": [("bowl", "bowl")],
    "stack_language": [("block", "colored cube block")],
    "swap_blocks": [("block", "cube block"), ("pad", "round mat"), ("button", "button")],
    "pack_box": [("box", "open box"), ("pack_item", "toy car . electric toothbrush . hammer . shoe")],
    "fill_eggs": [("egg", "egg"), ("egg_holder", "egg carton"), ("lid", "egg carton lid")],
    "fill_pens": [("pen", "pen"), ("pen_holder", "pen holder cup")],
    "hang_mugs": [("mug", "mug"), ("rack", "mug rack")],
    "make_toast": [("bread", "slice of bread"), ("toaster", "toaster"),
                   ("bread_shelf", "bread shelf"), ("lever", "toaster lever")],
    "play_xylophone": [("mallet", "xylophone mallet"), ("xylophone", "xylophone")],
}
CATEGORIES = {"toy_car":"toy car", "action_camera":"action camera", "pen":"pen",
              "watch":"watch", "garage":"tool", "car":"toy car", "wooden_toy":"wooden toy",
              "pepper":"pepper", "chocolate_bar":"chocolate bar"}


def normalized(v):
    v = np.asarray(v, float)
    if not np.isfinite(v).all() or np.linalg.norm(v) < .005:
        raise PolicyFailure("unresolved_axis")
    return v / np.linalg.norm(v)


def cloud_frame(points, previous=None):
    """A right-handed geometric frame, with sign continuity when visible."""
    points = np.asarray(points, float)
    if len(points) < 30 or not np.isfinite(points).all():
        raise PolicyFailure("insufficient_depth_support")
    values, vectors = np.linalg.eigh(np.cov(points.T))
    if values[1] < 1e-6:
        raise PolicyFailure("degenerate_geometry")
    normal = vectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    axis = vectors[:, 2]
    if previous is not None:
        ref = rotation(previous["quaternion"]).apply([1,0,0])
    else:
        ref = np.array([1.,0,0])
    if np.dot(axis, ref) < 0:
        axis = -axis
    # Near-square planar faces have no reliable in-plane PCA angle. Preserve
    # the previous axis projected onto the *observed* face rather than invent
    # an observable yaw; square/circular controllers tolerate this symmetry.
    if values[2] / values[1] < 1.25:
        axis = ref - np.dot(ref, normal) * normal
        if np.linalg.norm(axis) < .1:
            axis = np.cross([0,1,0], normal)
    axis = normalized(axis)
    across = normalized(np.cross(normal, axis))
    axis = normalized(np.cross(across, normal))
    return Rotation.from_matrix(np.stack((axis, across, normal), axis=1))


def associate(detections, previous, kind):
    """Appearance + metric association; reject equally plausible identities."""
    old = {k:v for k,v in previous.items() if v["kind"] == kind}
    if not old:
        return [(f"{kind}:{i}", x, None) for i,x in enumerate(sorted(detections,key=lambda x:tuple(x["position"]))) ]
    if not detections:
        return []
    names = sorted(old)
    cost = np.full((len(names),len(detections)), 10.)
    for i,name in enumerate(names):
        for j,new in enumerate(detections):
            before = old[name]
            if "digit" in new.get("features",{}) and before.get("features",{}).get("digit") != new["features"]["digit"]:
                continue
            displacement = np.linalg.norm(np.asarray(before.get("predicted_position",before["position"]))-new["position"])
            appearance = np.linalg.norm(np.asarray(before["appearance"])-new["appearance"])
            cost[i,j] = displacement + .20 * appearance
    rows, cols = linear_sum_assignment(cost)
    pairs=[]
    for i,j in zip(rows,cols):
        if cost[i,j] > .35:
            continue
        competitors = [cost[k,j] for k in range(len(names)) if k != i]
        if competitors and min(competitors)-cost[i,j] < .015:
            raise PolicyFailure("ambiguous_identity",names[i])
        alternatives = [cost[i,k] for k in range(len(detections)) if k != j]
        if alternatives and min(alternatives)-cost[i,j] < .015:
            raise PolicyFailure("ambiguous_detection",names[i])
        pairs.append((names[i],detections[j],old[names[i]]))
    return pairs


def retained_features(old, obj):
    """Transform previously observed rigid landmarks; never invent visibility."""
    f=deepcopy(old["features"])
    before,after=rotation(old["quaternion"]),rotation(obj["quaternion"])
    def point(value):
        return (np.asarray(obj["position"])+after.apply(before.inv().apply(np.asarray(value)-old["position"]))).tolist()
    def direction(value):return after.apply(before.inv().apply(value)).tolist()
    for key in ("inner_floor","opening","rim_grasp","grasp_point"):
        if key in f:f[key]=point(f[key])
    if 'rim_height' in f:
        f['rim_height'] = point([*old['position'][:2], f['rim_height']])[2]
    for key in ("wells","keys"):
        if key in f:f[key]=[point(v) for v in f[key]]
    for key in ("hooks","slots"):
        if key in f:
            for v in f[key]:
                v["point"]=point(v["point"])
                if "axis" in v:v["axis"]=direction(v["axis"])
                if "quaternion" in v:v["quaternion"]=quaternion(after*before.inv()*rotation(v["quaternion"]))
    return f


def body_geometry(points, kind, old=None):
    # Visible top-plane fit for table-supported objects. The initial table
    # contact prior is public calibration; moving frames retain metric size.
    top=float(np.quantile(points[:,2],.95))
    planar=kind in {"block","tower_board","tower_support","bowl","basket","bin","box","pad","digit","pen_holder"}
    if planar:
        upper=points[points[:,2]>top-.012]
        r=cloud_frame(upper,old)
        local=r.inv().apply(points)
        low,high=np.quantile(local,[.03,.97],axis=0)
        size=np.maximum(high-low,.008)
        size[2]=max(.008,top-.765) if old is None else old["size"][2]
        centre=np.median(upper,axis=0)-r.apply([0,0,size[2]/2])
    else:
        r=cloud_frame(points,old)
        low,high=np.quantile(r.inv().apply(points),[.03,.97],axis=0)
        size=np.maximum(high-low,.008);centre=r.apply((high+low)/2)
    if old is not None:size=np.asarray(old["size"])
    if min(size)<=0 or max(size)>.60:raise PolicyFailure("implausible_body_extent")
    return centre,size,r


class RGBScene:
    def __init__(self, name):
        self.name, self.module = name, POLICIES[name]["module"]
        self.backend = None
        self.reader = None

    def _load(self):
        if self.backend is None:
            from rdj_master.tasks.general_pickup.rgb_perception import GroundedClipPerceptor
            self.backend = GroundedClipPerceptor(os.environ["RDJ_GENERAL_PICKUP_DINO_MODEL"],
                os.environ["RDJ_GENERAL_PICKUP_CLIP_MODEL"],os.environ["RDJ_GENERAL_PICKUP_DEPTH_MODEL"])
        return self.backend

    def boxes(self, rgb, queries):
        from PIL import Image
        from torchvision.ops import nms
        p=self._load(); image=Image.fromarray(rgb)
        inputs=p._dino_processor(images=[image]*len(queries),text=[q+" ." for q in queries],
                                 padding=True,return_tensors="pt").to(p._device)
        with p._torch.inference_mode():
            output=p._dino(**inputs)
        rows=p._dino_processor.post_process_grounded_object_detection(output,inputs.input_ids,
            threshold=.30,text_threshold=.25,target_sizes=[rgb.shape[:2]]*len(queries))
        return [[dict(box=row["boxes"][i].detach().cpu().numpy(),
                      confidence=float(row["scores"][i]))
                 for i in nms(row["boxes"],row["scores"],.35).detach().cpu().tolist()] for row in rows]

    def metric_image(self, rgb):
        from PIL import Image
        try:
            from .depth_calibration import calibrate,DepthCalibrationError,visible_table_support
        except ModuleNotFoundError as error:
            if error.name != __package__+'.depth_calibration':raise
            from dsr.skills.native_specialists.depth_calibration import calibrate,DepthCalibrationError,visible_table_support
        from rdj_master.tasks.general_pickup import rgb_perception as c
        if rgb.shape != (480,640,3):
            raise PolicyFailure("camera_calibration_size")
        p=self._load()
        with p._torch.inference_mode():
            inputs=p._depth_processor(images=[Image.fromarray(rgb)],return_tensors="pt").to(p._device)
            output=p._depth(**inputs).predicted_depth
            depth=p._torch.nn.functional.interpolate(output.unsqueeze(1),size=(480,640),
                mode="bicubic",align_corners=False)[0,0].detach().cpu().numpy()
        try:
            distance,self.last_depth_calibration=calibrate(depth,c._TABLE_FORWARD_DISTANCE,c._TABLE_FIT_PIXELS)
        except DepthCalibrationError as error:
            # Re-measure table support from the current RGB frame. This is a
            # bounded diagnostic recovery, not a relaxed residual threshold or
            # a simulator pose substitute.
            diagnostic=dict(initial_failure=str(error));masks=[];proposals=[]
            for query in ('wooden tabletop','wooden table','tabletop'):
                for row in self.boxes(rgb,[query])[0]:
                    box=np.asarray(row['box'],float);area=float(np.prod(np.maximum(box[2:]-box[:2],0)))
                    if not .15*rgb.shape[0]*rgb.shape[1]<=area<=.95*rgb.shape[0]*rgb.shape[1]:continue
                    try:mask=self.rgb_mask(rgb,box)
                    except PolicyFailure:continue
                    masks.append(mask);proposals.append(dict(query=query,box=box.tolist(),
                        confidence=row['confidence'],pixels=int(mask.sum())))
                if masks:break
            try:
                support=visible_table_support(np.asarray(masks),depth.shape)
                distance,receipt=calibrate(depth,c._TABLE_FORWARD_DISTANCE,c._TABLE_FIT_PIXELS&support)
            except DepthCalibrationError as masked_error:
                diagnostic['rgb_support_failure']=str(masked_error);self.last_depth_calibration=diagnostic
                raise PolicyFailure(str(masked_error)) from masked_error
            receipt.update(initial_failure=str(error),method='independent_rgb_table_support',
                           table_rgb_pixels=int(support.sum()),table_proposals=proposals)
            self.last_depth_calibration=receipt
        yy,xx=np.indices((480,640))
        return np.stack((distance*(xx-320)/c._FOCAL_PIXELS,
            -.41+distance*(.8660254037844386*(-(yy-240)/c._FOCAL_PIXELS)+.5),
            1.308+distance*c._RAY_Z),axis=-1)

    @staticmethod
    def rgb_mask(rgb,box):
        import cv2
        x0,y0,x1,y1=np.asarray(box,int);x0,y0=max(0,x0),max(0,y0);x1,y1=min(rgb.shape[1],x1),min(rgb.shape[0],y1)
        if x1-x0<4 or y1-y0<4:raise PolicyFailure('landmark_too_small')
        mask=np.zeros(rgb.shape[:2],np.uint8)
        cv2.grabCut(rgb,mask,(x0,y0,x1-x0,y1-y0),np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_RECT)
        return (mask==1)|(mask==3)

    @staticmethod
    def cloud(rgb, world, box):
        import cv2
        x0,y0,x1,y1=np.asarray(box,int)
        x0,y0=max(0,x0),max(0,y0);x1,y1=min(rgb.shape[1],x1),min(rgb.shape[0],y1)
        if x1-x0<4 or y1-y0<4:
            raise PolicyFailure("landmark_too_small")
        mask=np.zeros(rgb.shape[:2],np.uint8)
        cv2.grabCut(rgb,mask,(x0,y0,x1-x0,y1-y0),np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_RECT)
        mask=((mask==1)|(mask==3))
        points=world[mask]
        points=points[(points[:,2]>.77)&(points[:,2]<1.45)]
        if len(points)<30:
            raise PolicyFailure("empty_visible_geometry")
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        hist=np.histogram(hsv[:,:,0][mask],bins=18,range=(0,180))[0].astype(float)
        hist/=max(1,hist.sum())
        return points,hist

    def part(self, rgb, world, obj, query, count=1):
        """Detect part proposals inside the current object's RGB crop only."""
        x0,y0,x1,y1=np.asarray(obj["box"],int)
        x0,y0=max(x0-4,0),max(y0-4,0);x1,y1=min(x1+4,640),min(y1+4,480)
        rows=self.boxes(rgb[y0:y1,x0:x1], [query])[0]
        if not rows or count is not None and len(rows)!=count:
            raise PolicyFailure("semantic_part_count",f"{obj['kind']}/{query}: {len(rows)} != {count}")
        result=[]
        for row in rows:
            box=row["box"]+np.array([x0,y0,x0,y0])
            points,_=self.cloud(rgb,world,box)
            extent=np.ptp(points,axis=0)
            if max(extent)>.35:
                raise PolicyFailure("semantic_part_extent",query)
            result.append(dict(point=np.median(points,axis=0),points=points,
                               confidence=row["confidence"]))
        return result

    def category(self, rgb, obj, vocabulary):
        from PIL import Image
        p=self._load();box=np.asarray(obj["box"],int)
        crop=Image.fromarray(rgb).crop(tuple(box))
        keys=list(vocabulary)
        inputs=p._clip_processor(text=["a photo of a "+vocabulary[k] for k in keys],images=crop,
                                return_tensors="pt",padding=True).to(p._device)
        with p._torch.inference_mode():
            scores=p._clip(**inputs).logits_per_image[0].softmax(-1).detach().cpu().numpy()
        order=np.argsort(scores)
        if scores[order[-1]]<.55 or scores[order[-1]]-scores[order[-2]]<.20:
            raise PolicyFailure("ambiguous_visual_category")
        return keys[order[-1]]

    def features(self, rgb, world, obj, points, instruction):
        kind=obj["kind"];f=obj["features"];r=rotation(obj["quaternion"]);centre=np.asarray(obj["position"])
        def parts(query,count=1):return self.part(rgb,world,obj,query,count)
        def point(query):return parts(query)[0]["point"]
        def local(p):return r.inv().apply(np.asarray(p)-centre).tolist()
        if kind in ("basket","bin","box"):
            opening=parts("inside opening")[0];floor=point("inside bottom surface")
            local_cloud=r.inv().apply(opening["points"]-opening["point"])
            size=np.quantile(local_cloud,.95,axis=0)-np.quantile(local_cloud,.05,axis=0)
            if min(size[:2])<.03 or floor[2]>=opening["point"][2]-.01:
                raise PolicyFailure("unresolved_container_interior")
            f.update(inner_size=size[:2].tolist(),inner_floor=floor.tolist(),rim_height=float(opening["point"][2]))
        if kind=="block" and self.module=="stack_language":
            from .policies.stack_language import color_order
            vocabulary={color:color+" block" for color in color_order(instruction)}
            f["color"]=self.category(rgb,obj,vocabulary)
        if kind=="item":
            if self.module=="classify_language":
                from .policies.classify_language import destinations
                names=set(destinations(instruction).values())
                vocabulary={k:CATEGORIES.get(k,k.replace('_',' ')) for k in names}
            else:
                vocabulary={k:CATEGORIES[k] for k in ("toy_car","action_camera","pen","watch","garage")}
            f["category"]=self.category(rgb,obj,vocabulary)
        if kind in ("bowl","pen_holder"):
            rim=parts("bowl rim" if kind=="bowl" else "cup rim")[0];cloud=rim["points"]
            f["rim_diameter"]=float(max(np.ptp(cloud,axis=0)[:2]))
            f["rim_grasp"]=cloud[np.argmin(cloud[:,0]) if centre[0]<0 else np.argmax(cloud[:,0])].tolist()
        if kind in ("tower_roof","tower_crown"):
            # Desired orientation uses the observed visible top/up feature,
            # rather than assuming the asset's root Z is its physical up axis.
            up=normalized(point("top ridge" if kind=="tower_roof" else "top surface")-centre)
            axis_local=r.inv().apply(up)
            desired=Rotation.align_vectors([[0,0,1],[1,0,0]],[axis_local,[1,0,0]])[0]
            f.update(assembly_quaternion=quaternion(desired),assembly_height=float((abs(desired.as_matrix())@obj["size"])[2]))
        if kind=="pack_item":
            name=self.category(rgb,obj,{"car":"toy car","brush":"electric toothbrush","hammer":"hammer","shoe":"shoe"})
            landmark={"car":"front of car","brush":"brush head","hammer":"hammer head","shoe":"shoe toe"}[name]
            f["front_axis_local"]=r.inv().apply(normalized(point(landmark)-centre)).tolist()
        if kind=="egg_holder":
            # The task contains four eggs; it does not require a four-well
            # carton. Keep every observed empty well, then assign four slots.
            f["wells"]=[v["point"].tolist() for v in parts("empty egg well",None)]
        if kind=="lid":
            hinge=parts("hinge edge")[0];edge=point("outer free edge of lid")
            axis=cloud_frame(hinge["points"]).apply([1,0,0]);radial=edge-hinge["point"]
            radial-=axis*np.dot(axis,radial)
            target=radial.copy();target[2]=0;target=normalized(target)*np.linalg.norm(radial)
            angle=float(np.arctan2(np.dot(axis,np.cross(radial,target)),np.dot(radial,target)))
            f.update(hinge=hinge["point"].tolist(),hinge_axis=axis.tolist(),free_edge=edge.tolist(),closing_angle_rad=angle)
        if kind=="pen_holder":
            from scipy.spatial import ConvexHull, QhullError
            opening=parts("cup opening")[0]
            rim=parts("cup rim")[0]
            floor=point("inside bottom of cup")
            # Opening crops see the bottom; use their XY footprint but obtain
            # mouth height from the separately visible rim.
            mouth=np.asarray(opening["point"]).copy()
            mouth+=r.apply([0,0,np.median(r.inv().apply(rim["points"]-mouth)[:,2])])
            xy=r.inv().apply(opening["points"]-mouth)[:,:2]
            try:hull=ConvexHull(xy)
            except QhullError as error:raise PolicyFailure("cup_aperture_unresolved") from error
            radius=float(np.min(-hull.equations[:,2]/np.linalg.norm(hull.equations[:,:2],axis=1))-.004)
            depth=float(r.inv().apply(mouth-floor)[2])
            if not .008<radius<.10 or not .04<depth<.25:
                raise PolicyFailure("cup_aperture_unresolved")
            f.update(opening=mouth.tolist(),opening_radius_m=radius,opening_depth_m=depth)
        if kind=="pen":
            tip=point("writing tip of pen");f["tip_local"]=local(tip)
            f["tip_axis_local"]=normalized(f["tip_local"]).tolist()
        if kind=="rack":
            hooks=parts("mug hook",3);f["hooks"]=[]
            for hook in hooks:
                axis=cloud_frame(hook["points"]).apply([1,0,0]);axis[2]=0;axis=normalized(axis)
                # Opening direction points from rack centre towards hook tip.
                if np.dot(axis,hook["point"]-centre)<0:axis=-axis
                f["hooks"].append(dict(point=hook["point"].tolist(),axis=axis.tolist()))
        if kind=="mug":
            handle=parts("mug handle opening")[0]
            normal=cloud_frame(handle["points"]).apply([0,0,1])
            f.update(handle_local=local(handle["point"]),handle_axis_local=r.inv().apply(normal).tolist())
        if kind=="toaster":
            f["slots"]=[]
            for slot in parts("toaster slot",2):
                axis=cloud_frame(slot["points"]).apply([1,0,0]);axis[2]=0;axis=normalized(axis)
                # Bread's thin plane (local XY) becomes vertical, X along slot.
                desired=Rotation.from_matrix(np.stack((axis,[0,0,1],np.cross(axis,[0,0,1])),axis=1))
                f["slots"].append(dict(point=slot["point"].tolist(),quaternion=quaternion(desired)))
        if kind in ("button","lever"):
            contact=point("button top" if kind=="button" else "lever handle")
            bottom=point("bottom of button travel" if kind=="button" else "bottom of lever track")
            delta=bottom-contact;length=float(np.linalg.norm(delta))
            if not .003<=length<=.08:raise PolicyFailure("unresolved_press_travel")
            f.update(contact=contact.tolist(),press_axis=normalized(delta).tolist(),travel_m=length)
        if kind=="xylophone":
            keys=parts("colored xylophone key",8)
            first=point("longest xylophone key at the low note end")
            centres=np.asarray([x["point"] for x in keys])
            _,_,vh=np.linalg.svd(centres-centres.mean(axis=0),full_matrices=False)
            axis=vh[0]
            if np.dot(first-centres.mean(axis=0),axis)>0:axis=-axis
            order=np.argsort(centres@axis)
            if np.linalg.norm(centres[order[0]]-first)>.018 or np.min(np.diff(centres[order]@axis))<.012:
                raise PolicyFailure("xylophone_order_unresolved")
            f["keys"]=[p.tolist() for p in centres[order]]
            f["key_order_source"]="visible_low_note_endpoint"
        if kind=="mallet":
            tip=point("mallet striking head");f["tip_local"]=local(tip)
            f["grasp_point"]=point("mallet handle").tolist()

    def detect(self, observation, previous):
        from rdj_master.tasks.general_pickup.rgb_perception import head_rgb, world_to_pixel_on_height
        rgb=head_rgb({"vision":observation["vision"]})
        world=self.metric_image(rgb)
        queries=QUERIES[self.module]
        rows=self.boxes(rgb,[q for _,q in queries])
        readings=None
        if self.module=="arrange_digits" and not previous:
            from arrange_rgb.perception import Reader
            if self.reader is None:self.reader=Reader()
            readings,_,_=self.reader.read_color(rgb)
            if not readings["valid"]:
                readings,_,_=self.reader.read_color(rgb,adaptive=True)
        result={}
        for (kind,_),row in zip(queries,rows):
            detections=[]
            for proposal in row:
                try:points,hist=self.cloud(rgb,world,proposal["box"])
                except PolicyFailure:continue
                centre,size,r=body_geometry(points,kind)
                # Initial visible bodies on the public table have an occluded
                # lower side; this is a geometric estimate, never ground truth.
                size=np.maximum(size,.008)
                obj=dict(kind=kind,position=centre.tolist(),size=size.tolist(),quaternion=quaternion(r),
                         confidence=proposal["confidence"],features={},appearance=hist.tolist(),geometric_quaternion=quaternion(r),
                         box=proposal["box"].tolist(),geometry_method="rgb_monocular_visible_obb")
                if kind in ("digit","pad") and readings is not None:
                    values=readings["digits" if kind=="digit" else "pads"]
                    matched=[]
                    for value in values:
                        uv=world_to_pixel_on_height(value["xy"],world_z=.78)
                        b=proposal["box"]
                        if b[0]<=uv[0]<=b[2] and b[1]<=uv[1]<=b[3]:matched.append(value)
                    if len(matched)!=1:continue
                    value=matched[0]
                    if kind=="digit":
                        obj["features"].update(digit=int(value["digit"]),upright_quaternion=[1,0,0,0])
                        # Reader determines semantic glyph yaw on the table;
                        # depth continues to measure height after lifting.
                        obj["quaternion"]=quaternion(Rotation.from_euler("z",value["yaw_deg"],degrees=True))
                        if value.get("grasp_support",{}).get("valid") and "grasp_offset_object_xy" in value:
                            obj["features"]["grasp_point"]=(centre+rotation(obj["quaternion"]).apply([*value["grasp_offset_object_xy"],0])).tolist()
                detections.append((obj,points))
            pairs=associate([o for o,_ in detections],previous,kind)
            for identity,obj,old in pairs:
                points=next(p for o,p in detections if o is obj)
                if old is not None:
                    # Retain dimensions after lifting/stacking. Do not infer
                    # object height from its height above the table each frame.
                    centre,size,r=body_geometry(points,kind,old)
                    obj["position"],obj["size"]=centre.tolist(),size.tolist()
                    if kind=="digit":
                        prior_geo=rotation(old.get("geometric_quaternion",old["quaternion"]))
                        obj["quaternion"]=quaternion(r*prior_geo.inv()*rotation(old["quaternion"]))
                    else:obj["quaternion"]=quaternion(r)
                    obj["geometric_quaternion"]=quaternion(r)
                obj["id"]=identity
                if old is not None and kind not in ("lid","button","lever"):
                    current=deepcopy(obj["features"])
                    obj["features"]=retained_features(old,obj)
                    obj["features"].update(current)
                    if kind=="pen_holder":
                        # Drift checks require fresh rim geometry, not two
                        # predictions of the same retained landmark.
                        fresh=self.part(rgb,world,obj,"cup rim")[0]["points"]
                        normal=rotation(obj["quaternion"]).apply([0,0,1])
                        mouth=np.asarray(obj["features"]["opening"])
                        delta=np.median(fresh,axis=0)-mouth
                        mouth+=delta-normal*np.dot(delta,normal)
                        mouth+=normal*np.median((fresh-mouth)@normal)
                        obj["features"]["opening"]=mouth.tolist()
                elif old is not None and kind in ("button","lever"):
                    obj["features"]=deepcopy(old["features"])
                    obj["features"]["contact"]=self.part(rgb,world,obj,"button top" if kind=="button" else "lever handle")[0]["point"].tolist()
                else:
                    self.features(rgb,world,obj,points,observation["instruction"])
                if kind in ("bowl","bottle","egg","pen_holder","tower_support"):
                    obj["features"]["yaw_symmetric"]=True
                result[identity]=obj
        expected=POLICIES[self.name]["expected_objects"]
        if not previous:
            for kind,count in expected.items():
                actual=sum(x["kind"]==kind for x in result.values())
                if actual not in (count if isinstance(count,list) else [count]):
                    raise PolicyFailure("initial_rgb_count",f"{kind}: {actual}, expected {count}")
        return result
