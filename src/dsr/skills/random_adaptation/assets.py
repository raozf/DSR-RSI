"""Offline public USD mesh audit/bank builder; never called with scene IDs."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh(path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(Path(path).resolve()))
    candidates = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if not candidates:
        raise ValueError('public mesh missing: ' + str(path))
    visual = [p for p in candidates if '/visual/' in str(p.GetPath())]
    prim = (visual or candidates)[0]
    obj = UsdGeom.Mesh(prim)
    points = np.asarray(obj.GetPointsAttr().Get(), float)
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), float)
    points = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
    # RoboDojo assets are authored in metres; reject unknown unit conversion.
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError('public mesh units differ from native metres')
    indices = np.asarray(obj.GetFaceVertexIndicesAttr().Get(), int)
    counts = np.asarray(obj.GetFaceVertexCountsAttr().Get(), int)
    if np.all(counts == 3):
        triangles = indices.reshape(-1, 3)
    else:
        triangles, start = [], 0
        for count in counts:
            polygon = indices[start:start + count]
            triangles.extend([polygon[0], polygon[j], polygon[j + 1]] for j in range(1, count - 1))
            start += count
        triangles = np.asarray(triangles, int)
    return points, triangles


def shell_profile(points, triangles, identity):
    lo, hi = points[:, 2].min(), points[:, 2].max()
    centre = (points[:, :2].min(0) + points[:, :2].max(0)) / 2
    radial = np.linalg.norm(points[:, :2] - centre, axis=1)
    central = radial < min(.010, radial.max() * .12)
    if central.sum() < 10:
        raise ValueError('public bowl central floor unresolved')
    floor = float(points[central, 2].max())
    vertices = points[triangles]
    zs, inners, outers = [], [], []
    for z in np.linspace(lo + .0002, hi - .0002, 100):
        crossing = (vertices[:, :, 2].min(1) <= z) & (vertices[:, :, 2].max(1) >= z)
        tri = vertices[crossing]
        cuts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            dz = tri[:, b, 2] - tri[:, a, 2]
            valid = (abs(dz) > 1e-10) & ((tri[:, a, 2] - z) * (tri[:, b, 2] - z) <= 0)
            start, end = tri[valid, a], tri[valid, b]
            t = (z - start[:, 2]) / (end[:, 2] - start[:, 2])
            cuts.extend(start + t[:, None] * (end - start))
        if len(cuts) < 20:
            raise ValueError('public bowl shell section unresolved')
        radius = np.linalg.norm(np.asarray(cuts)[:, :2] - centre, axis=1)
        zs.append(float(z)); outers.append(float(radius.max() + .0005))
        inners.append(float(max(0., radius.min() - .0005)) if z > floor + .0005 else 0.)
    return dict(z=zs, inner_lower=inners, outer_upper=outers,
                floor_z=floor, bottom_z=float(lo), rim_z=float(hi), model_sha256=identity)


def build_bank(asset_root, output):
    """Freeze 5 public bowl meshes and verify all bottle geometry variants."""
    root, out = Path(asset_root), Path(output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    shapes, entries, bottles = {}, {}, []
    for index in range(4):
        base = root / 'Rigid/wuliangye' / f'{index:05d}'
        points, faces = mesh(base / 'object.usdz')
        bottles.append(dict(index=index, mesh_sha256=sha(base / 'object.usdz'),
                            transformed_points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
                            faces_sha256=hashlib.sha256(faces.tobytes()).hexdigest()))
    for index in range(11, 16):
        base = root / 'Rigid/bowl' / f'{index:05d}'
        points, triangles = mesh(base / 'object.usdz')
        metadata = json.loads((base / 'metadata.json').read_text())
        digest = sha(base / 'object.usdz')
        key = f'bowl{index}'
        # Deterministic shuffle avoids triangle/vertex authoring-order bias.
        rng = np.random.default_rng(1000 + index)
        shapes[key] = points[rng.permutation(len(points))[:6000]]
        entries[key] = dict(profile=shell_profile(points, triangles, digest),
                            bbox=metadata['geometry']['aligned_bbox']['vertices'],
                            rest_quat=metadata['active']['place']['up']['orientation'],
                            contact_center=metadata['active']['place']['up']['contact_circle']['center'][:3],
                            radius=metadata['geometry']['radius'],
                            asset=str(base), mesh_sha256=digest,
                            metadata_sha256=sha(base / 'metadata.json'))
    np.savez_compressed(out / 'bowl_shapes.npz', **shapes)
    report = dict(schema='dsr.random-public-mesh-bank/v1', entries=entries,
                  bottles=bottles,
                  bottle_geometry_identical=len({(x['transformed_points_sha256'], x['faces_sha256']) for x in bottles}) == 1,
                  runtime_identity_source='RGB registration only; no evaluator asset ID',
                  units='metres', profiles='horizontal triangle shell sections; geometry proxy')
    (out / 'bank.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--asset-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(build_bank(args.asset_root, args.output), indent=2))
