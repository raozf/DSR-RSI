"""SAM3 and public shape registration; no scene object/pose/depth inputs."""
from pathlib import Path
import json,cv2,numpy as np
import os,time
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
CAM_R=Rot.from_euler('x',30,degrees=True).as_matrix();CAM_T=np.array([0,-.41,1.308]);F=6400/22.212

def matrix(p):
 t=np.eye(4);t[:3,3]=p[:3];t[:3,:3]=Rot.from_quat(np.asarray(p)[[4,5,6,3]]).as_matrix();return t

def pose(t):
 q=Rot.from_matrix(t[:3,:3]).as_quat();return np.r_[t[:3,3],q[[3,0,1,2]]]

def project(p):
 p=(p-CAM_T)@CAM_R;return np.c_[320+F*p[:,0]/-p[:,2],240-F*p[:,1]/-p[:,2]]

def registration(mask,key,prior=None,visibility=None,*,shapes,meta):
 points=np.asarray(shapes[key],float);sample=points[np.linspace(0,len(points)-1,1200).astype(int)]
 y,x=np.where(mask);visible=np.c_[x,y][np.linspace(0,len(x)-1,min(400,len(x))).astype(int)]
 outside=cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,5)
 rest=matrix(np.r_[0,0,0,meta[key]['rest_quat']])[:3,:3]
 def decode(v):
  t=np.eye(4)
  if prior is None:
   t[:3,:3]=Rot.from_euler('z',v[2]).as_matrix()@rest;t[:2,3]=v[:2];t[2,3]=v[3] if key=='bin' else .765-(points@t[:3,:3].T)[:,2].min()
  else:t[:3,3]=prior[:3,3]+v[:3];t[:3,:3]=Rot.from_rotvec(v[3:]).as_matrix()@prior[:3,:3]
  return t
 def residual(v):
  t=decode(v);world=sample@t[:3,:3].T+t[:3,3];uv=project(world);d=map_coordinates(outside,[uv[:,1],uv[:,0]],order=1,mode='constant',cval=100)
  inside=(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)
  kept=inside.copy()
  if visibility is not None:kept &= ~visibility.hidden(world)
  if visibility is not None:d=np.where(kept,d,0.)
  reverse=cKDTree(uv[kept] if kept.any() else uv).query(visible)[0]
  values=np.r_[d,reverse]
  if visibility is not None:values*=np.sqrt(len(values)/max(1,int(kept.sum())+len(reverse)))
  return values
 if prior is None:
  uv=visible.mean(0);ray=CAM_R@np.array([(uv[0]-320)/F,-(uv[1]-240)/F,-1]);guess=CAM_T+ray*((.35 if key=='bin' else .88)-CAM_T[2])/ray[2]
  yaws=[0.,np.pi/2] if key.startswith('bottle') else [0.]
  starts=[np.r_[guess[:2],a,.35] if key=='bin' else np.r_[guess[:2],a] for a in yaws];bounds=([-1.1,-.55,-4,.20],[.65,.45,4,.50]) if key=='bin' else ([-.65,-.55,-4],[.65,.45,4])
 else:starts=[np.zeros(6)];bounds=(np.r_[[-.12]*3,[-.7]*3],np.r_[[.12]*3,[.7]*3])
 best=None
 for start in starts:
  opt=least_squares(residual,np.clip(start,np.asarray(bounds[0])+1e-5,np.asarray(bounds[1])-1e-5),bounds=bounds,max_nfev=45,diff_step=1e-4,ftol=1e-4,xtol=1e-4,gtol=1e-4)
  error=float(np.sqrt(np.mean(opt.fun**2)))
  if best is None or error<best[0]:best=(error,decode(opt.x))
 error,t=best;world=points@t[:3,:3].T+t[:3,3];uv=np.rint(project(world)).astype(int);im=np.zeros(mask.shape,'uint8');valid=(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480);hidden=visibility.hidden(world) if visibility is not None else np.zeros(len(points),bool);valid &= ~hidden;uv=uv[valid];im[uv[:,1],uv[:,0]]=1
 im=cv2.morphologyEx(im,cv2.MORPH_CLOSE,np.ones((5,5),'uint8'));pred=im>0;iou=float((pred&mask).sum()/max(1,(pred|mask).sum()))
 return {'key':key,'pose':t,'error_px':error,'iou':iou,'visible_model_points':int(valid.sum()),'occluded_model_points':int(hidden.sum())}
