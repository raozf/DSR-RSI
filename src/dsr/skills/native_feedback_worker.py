"""Native batch worker for feedback policies using existing RGB model assets.

This candidate adapter is deliberately independent of task/checker/reset code.
It estimates cuboids from DINO + monocular depth using the installed
head-camera calibration. Occlusion, unresolved identity and uncertain progress
stop execution; the unchanged simulator decides success. It is unqualified.
"""
from __future__ import annotations

import ast
from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


def estimated_axis_quaternion(points, *, planar):
    """Estimate an axis from visible depth geometry; never assert uprightness."""
    from scipy.spatial.transform import Rotation
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30 or not np.isfinite(points).all():
        raise ValueError('Insufficient visible depth geometry for orientation')
    values, vectors = np.linalg.eigh(np.cov(points.T))
    if planar:
        if values[1] <= 1e-6 or values[0] / values[1] > .4:
            raise ValueError('Bowl rim plane is unresolved')
        axis = vectors[:, 0]
    else:
        if values[2] <= 1e-6 or values[2] / max(values[1],1e-12) < 1.5:
            raise ValueError('Bottle principal axis is unresolved')
        axis = vectors[:, 2]
    if axis[2] < 0:
        axis = -axis
    # Rotationally symmetric objects expose tilt but not yaw around their axis.
    rotation, _ = Rotation.align_vectors([axis], [[0,0,1]])
    return rotation.as_quat()[[3,0,1,2]].tolist()


class _PrimitiveCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "robot":
            if node.func.attr not in {"scene", "approach", "gripper", "reobserve"} or node.args:
                raise ValueError("Only declared keyword primitive calls are supported")
            if any(keyword.arg is None for keyword in node.keywords):
                raise ValueError("Expanded keyword capabilities are forbidden")
            return ast.copy_location(ast.Yield(ast.Tuple(elts=[ast.Constant(node.func.attr),
                ast.Dict(keys=[ast.Constant(item.arg) for item in node.keywords],
                         values=[item.value for item in node.keywords])], ctx=ast.Load())), node)
        return node


def controller(source):
    tree = ast.parse(source)
    tree = ast.fix_missing_locations(_PrimitiveCalls().visit(tree))
    namespace = {}
    exec(compile(tree, "<frozen-feedback-policy>", "exec"), namespace)
    return namespace["policy"]


class HeadScene:
    """Reuse the registered pickup worker's pretrained RGB/depth models."""
    def __init__(self, task):
        self.queries = feedback_queries(task)
        from rdj_master.tasks.general_pickup.rgb_perception import GroundedClipPerceptor
        self.perceptor = GroundedClipPerceptor(os.environ["RDJ_GENERAL_PICKUP_DINO_MODEL"],
            os.environ["RDJ_GENERAL_PICKUP_CLIP_MODEL"], os.environ["RDJ_GENERAL_PICKUP_DEPTH_MODEL"])
        self.task = task

    def detect(self, observation, previous):
        from PIL import Image
        from sklearn.linear_model import LinearRegression, RANSACRegressor
        from scipy.optimize import linear_sum_assignment
        from rdj_master.tasks.general_pickup import rgb_perception as calibration
        p = self.perceptor
        rgb = calibration.head_rgb(observation)
        image = Image.fromarray(rgb)
        queries = self.queries
        inputs = p._dino_processor(images=[image] * len(queries), text=[q + " ." for q, _ in queries],
                                  padding=True, return_tensors="pt").to(p._device)
        with p._torch.inference_mode():
            output = p._dino(**inputs)
            depth_inputs = p._depth_processor(images=[image], return_tensors="pt").to(p._device)
            depth = p._depth(**depth_inputs).predicted_depth
            depth = p._torch.nn.functional.interpolate(depth.unsqueeze(1), size=(480,640),
                mode="bicubic", align_corners=False)[0,0].detach().cpu().numpy()
        rows = p._dino_processor.post_process_grounded_object_detection(output, inputs.input_ids,
            threshold=0.20, text_threshold=0.15, target_sizes=[(480,640)] * len(queries))
        fit = RANSACRegressor(estimator=LinearRegression(), min_samples=0.25,
            residual_threshold=0.015, max_trials=100, random_state=0).fit(
                depth[calibration._TABLE_FIT_PIXELS, None], calibration._TABLE_FORWARD_DISTANCE[calibration._TABLE_FIT_PIXELS])
        distance = fit.predict(depth.reshape(-1,1)).reshape(480,640)
        heights = 1.308 + distance * calibration._RAY_Z
        result = {}
        for (query, expected), row in zip(queries, rows):
            from torchvision.ops import nms
            keep = nms(row["boxes"], row["scores"], 0.35).detach().cpu().tolist()
            detections = []
            for index in keep:
                box = row["boxes"][index].detach().cpu().numpy()
                x0,y0,x1,y1 = box
                if x1-x0 < 10 or y1-y0 < 10 or (query != "trash can" and (x1-x0 > 220 or y1-y0 > 210)):
                    continue
                centre = ((x0+x1)/2, (y0+y1)/2)
                crop = heights[max(0,int(y0)):min(480,int(y1)),max(0,int(x0)):min(640,int(x1))]
                if crop.size < 30:
                    continue
                top = float(np.quantile(crop, .80))
                if not 0.775 < top < 1.30:
                    continue
                xy = calibration.pixel_to_world_on_height(centre, world_z=top)
                if not -.6 < xy[0] < .6 or not -.45 < xy[1] < .35:
                    continue
                left = calibration.pixel_to_world_on_height((x0,centre[1]), world_z=top)
                right = calibration.pixel_to_world_on_height((x1,centre[1]), world_z=top)
                front = calibration.pixel_to_world_on_height((centre[0],y0), world_z=top)
                back = calibration.pixel_to_world_on_height((centre[0],y1), world_z=top)
                width = max(.015, min(.35, abs(right[0]-left[0])))
                length = max(.015, min(.35, abs(back[1]-front[1])))
                height = max(.02, min(.30, top-.765))
                yy, xx = np.mgrid[max(0,int(y0)):min(480,int(y1)),max(0,int(x0)):min(640,int(x1))]
                depths = distance[yy, xx]
                world = np.stack((depths*(xx-320)/calibration._FOCAL_PIXELS,
                    -.41+depths*(.8660254037844386*(-(yy-240)/calibration._FOCAL_PIXELS)+.5),crop),axis=-1)
                valid = (crop > max(.785, top-(.035 if query in ('bowl', 'block') else .20))) & (crop < top+.02)
                try:
                    quaternion = estimated_axis_quaternion(world[valid], planar=query in ('bowl','block','trash can'))
                except ValueError:
                    continue
                detections.append(dict(pixel=list(centre), top=top, xy=xy, dims=[width,length,height],quaternion=quaternion))
            prefix = "container" if query == "trash can" else ("block_" if query == "block" else query)
            old = {name:value for name,value in previous.items() if name.startswith(prefix)}
            if not old:
                if len(detections) != expected:
                    raise ValueError("Initial RGB instance count unresolved: " + query)
                assignments = [(prefix+str(i), detection) for i,detection in enumerate(sorted(detections, key=lambda item:item["xy"][0]))]
            else:
                names = sorted(old)
                if not detections:
                    continue  # Unresolved targets stay absent; policy cannot count them complete.
                cost = np.asarray([[np.linalg.norm(np.asarray(old[name]["pixel"])-item["pixel"]) for item in detections] for name in names])
                ri, ci = linear_sum_assignment(cost)
                assignments = [(names[r],detections[c]) for r,c in zip(ri,ci) if cost[r,c] < 160]
            for name,item in assignments:
                dims = old[name]["dims"] if name in old else item["dims"]
                result[name] = {"pose": [*item["xy"],item["top"]-dims[2]/2,*item['quaternion']],
                                "dims": list(dims), "pixel": item["pixel"]}
        return result


def feedback_queries(task):
    """Only task-declared observed object classes may enter a candidate scene."""
    if task in ("stack_bowls", "stack_bowls_random"):
        return [("bowl", 3)]
    if task in ("stack_blocks", "stack_blocks_random"):
        return [("block", 3)]
    if task == "put_bottles_into_dustbin":
        return [("bottle", 4), ("trash can", 1)]
    raise ValueError("No declared feedback perception contract for task: " + str(task))


class Episode:
    def __init__(self, source, memory, scene, output):
        self.source, self.memory, self.perception = source, deepcopy(memory), scene
        self.channel = uuid.uuid4().hex
        self.sequence = 0
        self.observation = None
        self.previous_objects = {}
        self.generator = None
        self.actions = deque()
        self.pending = None
        self.result = None
        self.failed = None
        self.output = output
        self.settles = 0

    def observe(self, observation):
        self.observation = {key:observation[key] for key in ("vision","state","instruction")}
        self.sequence += 1

    def event(self, **record):
        with self.output.open("a") as stream:
            stream.write(json.dumps(dict(sequence=self.sequence,**record),allow_nan=False)+"\n")

    def hold(self):
        state = self.observation["state"]
        return {arm+suffix:list(state[arm+suffix]) for arm in ("left","right")
                for suffix in ("_ee_pose","_ee_joint_state")}

    def action(self):
        from scipy.spatial.transform import Rotation, Slerp
        if self.failed or self.result is not None:
            return self.hold()
        if self.actions:
            return self.actions.popleft()
        try:
            response = {"state":deepcopy(self.observation["state"])}
            if self.pending is not None:
                arm, target = self.pending
                actual = np.asarray(self.observation["state"][arm+"_ee_pose"])
                if (np.linalg.norm(actual[:3]-target[:3]) > .02 or
                        abs(np.dot(actual[3:], target[3:])) < .995):
                    if self.settles >= 8:
                        raise ValueError("Observed end effector failed to reach the commanded target")
                    self.settles += 1
                    hold = self.hold();hold[arm+"_ee_pose"]=target
                    return hold
                self.pending = None;self.settles=0
            if self.generator is None:
                home = {arm:[float(value) for value in self.observation["state"][arm+"_ee_pose"]]
                        for arm in ("left","right")}
                self.memory["manipulation"]["home_poses"] = home
                self.generator = controller(self.source)(None, self.memory)
                request = next(self.generator)
            else:
                request = self.generator.send(response)
            for _ in range(16):
                method, parameters = request
                self.event(primitive=method,parameters=parameters)
                if method == "scene":
                    current = self.perception.detect(self.observation,self.previous_objects)
                    self.previous_objects.update(current)
                    response = dict(frame="world",unit="m",objects={name:{"pose":item["pose"],"dims":item["dims"]} for name,item in current.items()},
                        observation=dict(channel_id=self.channel,sequence=self.sequence))
                    request = self.generator.send(response)
                    continue
                if method == "reobserve":
                    request = self.generator.send(response)
                    continue
                arm = parameters["arm"]
                if arm not in ("left","right"):
                    raise ValueError("Invalid requested arm")
                current = self.hold()
                if method == "gripper":
                    value = float(parameters["command"])
                    if not 0 <= value <= 1:
                        raise ValueError("Invalid normalized gripper command")
                    current[arm+"_ee_joint_state"]=[value]
                    self.actions.extend(deepcopy(current) for _ in range(5))
                elif method == "approach":
                    target = np.asarray([*parameters["position_m"],*parameters["quaternion_wxyz"]],float)
                    if target.shape != (7,) or not np.isfinite(target).all() or abs(np.linalg.norm(target[3:])-1) > 1e-4:
                        raise ValueError("Invalid end effector target")
                    if not (-.6 < target[0] < .6 and -.6 < target[1] < .4 and .82 < target[2] < 1.5):
                        raise ValueError("Target exceeds the bounded robot workspace")
                    start=np.asarray(current[arm+"_ee_pose"],float)
                    rotations=Rotation.from_quat(np.asarray([start[3:],target[3:]])[:,[1,2,3,0]])
                    angle=(rotations[0].inv()*rotations[1]).magnitude()
                    count=max(4,int(np.ceil(np.linalg.norm(target[:3]-start[:3])/.008)),int(np.ceil(angle/.06)))
                    if count > 160:
                        raise ValueError("Requested motion exceeds the action budget")
                    orientations=Slerp([0,1],rotations)
                    for fraction in np.linspace(0,1,count+1)[1:]:
                        action=deepcopy(current)
                        action[arm+"_ee_pose"]=[*(start[:3]*(1-fraction)+target[:3]*fraction),*orientations(fraction).as_quat()[[3,0,1,2]]]
                        self.actions.append(action)
                    self.pending=(arm,target.tolist())
                else:
                    raise ValueError("Unknown primitive requested")
                return self.actions.popleft()
            raise ValueError("Policy exceeded the observation-call budget")
        except StopIteration as stopped:
            self.result = stopped.value
            self.event(controller_result=self.result,scope="Not a benchmark outcome")
        except (ValueError, KeyError) as error:
            self.failed = str(error)
            self.event(controller_failure=self.failed,scope="Candidate rejected the current execution; hold until native timeout")
        return self.hold()


class Model:
    def __init__(self, model_cfg):
        root=Path(__file__).parent
        self.source=(root/"controller.py").read_text()
        self.memory=json.loads((root/"memory.json").read_text())
        if hashlib.sha256(self.source.encode()).hexdigest() != model_cfg["candidate_source_sha256"]:
            raise ValueError("Candidate policy source changed")
        if hashlib.sha256(json.dumps(self.memory,sort_keys=True).encode()).hexdigest() != model_cfg['candidate_memory_sha256']:
            raise ValueError('Candidate memory changed')
        self.perception=HeadScene(model_cfg["task_name"])
        self.output=Path(os.environ["RDJ_OUTPUT_DIR"])
        self.reset()

    def reset(self):
        self.episodes={};self.indices=[]

    def update_obs_batch(self, observations):
        self.indices=[]
        for observation in observations:
            index=int(observation["env_idx"])
            if index not in self.episodes:
                self.episodes[index]=Episode(self.source,self.memory,self.perception,self.output/(str(index)+"-feedback.jsonl"))
            self.episodes[index].observe(observation)
            self.indices.append(index)

    def update_obs(self, observation):
        return self.update_obs_batch([dict(observation,env_idx=0)])

    def get_action_batch(self, indices=None):
        selected = self.indices if indices is None else list(indices)
        if any(index not in self.indices for index in selected):
            raise ValueError('Requested environment has no fresh observation')
        return [[self.episodes[index].action()] for index in selected]

    def get_action(self):
        return self.get_action_batch([0])[0]
