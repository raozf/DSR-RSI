import numpy as np
from scipy.spatial.transform import Rotation
from ..control import PolicyFailure, feature, rotation, quaternion, vector


def align(primary,up,target_primary,target_up=(0,0,1)):
    def frame(x,z):
        z=vector(z,3);z/=np.linalg.norm(z)
        x=vector(x,3);x=x-z*np.dot(x,z)
        if np.linalg.norm(x)<.001:raise PolicyFailure('degenerate_functional_frame')
        x/=np.linalg.norm(x)
        return np.stack((x,np.cross(z,x),z),axis=1)
    return quaternion(Rotation.from_matrix(frame(target_primary,target_up)@frame(primary,up).T))


def footprint_inside(item,container,margin=.005):
    f=container['features'];r=rotation(f.get('interior_quaternion',container['quaternion']))
    local=r.inv().apply(np.asarray(item['position'])-f['inner_floor'])
    extent=abs(r.as_matrix().T@rotation(item['quaternion']).as_matrix())@np.asarray(item['size'])/2
    return bool(np.all(abs(local[:2])+extent[:2]+margin<=np.asarray(f['inner_size'])/2))


def floor_slots(box,items,margin=.006):
    """2D observed floor packing; no invented below-rim height requirement."""
    f=box['features'];size=vector(f['inner_size'],2);r=rotation(f.get('interior_quaternion',box['quaternion']))
    ordered=sorted(items,key=lambda o:(-o['size'][1],-o['size'][0],o['id']))
    x=y=margin;row=0.;slots={}
    for item in ordered:
        w,h,z=item['size']
        if w+2*margin>size[0] or h+2*margin>size[1]:raise PolicyFailure('footprint_does_not_fit',item['id'])
        if x+w+margin>size[0]:x=margin;y+=row+margin;row=0.
        if y+h+margin>size[1]:raise PolicyFailure('floor_capacity')
        point=np.asarray(f['inner_floor'])+r.apply([x+w/2-size[0]/2,y+h/2-size[1]/2,0])
        point[2]+=z/2;slots[item['id']]=point
        x+=w+margin;row=max(row,h)
    return slots


def verify_direction(item,local_axis,target,tolerance_deg=15):
    observed=rotation(item['quaternion']).apply(local_axis)
    cosine=np.dot(observed,target)/np.linalg.norm(target)
    if cosine<np.cos(np.deg2rad(tolerance_deg)):raise PolicyFailure('functional_direction_unconfirmed',item['id'])


def observe_stable_all(robot,identities):
    for identity in identities:yield from robot.stable(identity,tolerance=.012)
