"""Offline public shape extraction; never inspect episode/category metadata."""
import hashlib
import json
from pathlib import Path
import numpy as np


def build(asset_root, output):
    from dsr.skills.random_adaptation.assets import mesh
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    families=[]
    for first in (15,20):
        shapes=[];sources=[]
        for index in reversed(range(first,first+5)):
            path=Path(asset_root)/'Rigid/matryoshka_dolls'/f'{index:05d}'/'object.usdz'
            points,_=mesh(path)
            points[:,2]-=points[:,2].min()
            z=np.linspace(.0005,points[:,2].max()-.0005,48)
            radii=[]
            for height in z:
                nearby=points[abs(points[:,2]-height)<.003]
                if len(nearby)<3:nearby=points[np.argsort(abs(points[:,2]-height))[:20]]
                radii.append(float(np.max(np.linalg.norm(nearby[:,:2],axis=1))))
            angles=np.linspace(0,2*np.pi,64,endpoint=False)
            xyz=np.stack((np.array(radii)[:,None]*np.cos(angles),np.array(radii)[:,None]*np.sin(angles),np.broadcast_to(z[:,None],(len(z),len(angles)))),axis=-1).reshape(-1,3)
            shapes.append(xyz.tolist());sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        families.append(dict(points=shapes,sources=sources))
    report=dict(schema='dsr.public-doll-profiles/v1',families=families,runtime_selection='current RGB contour fit, never asset ID',qualification='radial envelope proxy; native comparison required')
    (out/'dolls.json').write_text(json.dumps(report,separators=(',',':'))+chr(10))
    return dict(families=len(families),shapes=10,sha256=hashlib.sha256((out/'dolls.json').read_bytes()).hexdigest())

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(a.assets,a.output)))
