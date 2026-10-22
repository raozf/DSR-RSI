"""Egg-stage geometry from independent visible compartments, without lid gates."""
import numpy as np
from .control import PolicyFailure
from .instances import within_parent


def deduplicate_bases(candidates):
    """Collapse only near-identical boxes, independently of holes or lid parts."""
    groups = []
    for row in sorted(candidates, key=lambda p: -p['confidence']):
        box = np.asarray(row['box'], float)
        if not within_parent(box, box):
            continue
        for group in groups:
            if all(within_parent(box, old['box'], .9)
                   and within_parent(old['box'], box, .9) for old in group):
                group.append(row)
                break
        else:
            groups.append([row])
    return [group[0] for group in groups]


def select_base(candidates, wells, previous=None):
    """Remove lid/whole-carton boxes using the observed compartment footprint."""
    candidates = deduplicate_bases(candidates)
    if previous is None and len(wells) != 4:
        # A separate SAM3 base query can resolve identity before all holes are
        # visible. It does not qualify insertion targets or invent missing holes.
        segmented = [p for p in candidates if p.get('source', '').startswith('sam3_')
                     and p.get('mask') is not None]
        if (wells and len(candidates) == len(segmented) == 1
                and all(within_parent(w['box'], segmented[0]['box']) for w in wells)):
            return segmented[0]
        raise PolicyFailure('carton_base_evidence_unresolved')
    if previous is not None:
        witness = np.asarray(previous['box'], float)
    else:
        boxes = np.array([w['box'] for w in wells])
        witness = np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]
    valid = []
    for row in candidates:
        box = np.asarray(row['box'], float)
        if (not within_parent(witness, box, .85)
                or not all(within_parent(w['box'], box) for w in wells)):
            continue
        ratio = np.prod(box[2:]-box[:2]) / np.prod(witness[2:]-witness[:2])
        # A narrow front lip cannot contain the wells; a box including the
        # upright lid has much more area than their independent footprint.
        if .8 <= ratio <= (1.3 if previous is not None else 1.8):
            valid.append(row)
    if not valid:
        raise PolicyFailure('carton_base_evidence_unresolved')
    selected = max(valid, key=lambda p: p['confidence'])
    for row in valid:
        if not (within_parent(row['box'], selected['box'], .9)
                and within_parent(selected['box'], row['box'], .9)):
            raise PolicyFailure('carton_base_identity_unresolved')
    return selected


def well_geometry(reader, rgb, world, proposals):
    parts = []
    for row in proposals:
        points, _ = reader.cloud(rgb, world, row['box'], row['mask'])
        centre = np.median(points, axis=0)
        extent = np.ptp(points, axis=0)
        if not .012 < min(extent[:2]) or max(extent) > .10:
            raise PolicyFailure('carton_well_extent_unresolved')
        parts.append(dict(point=centre, points=points, box=row['box']))
    if len(parts) == 4:
        centres = np.array([p['point'] for p in parts])
        spread = np.linalg.svd(centres[:, :2]-centres[:, :2].mean(axis=0), compute_uv=False)
        if np.ptp(centres[:, 2]) > .025 or spread[1] <= .008:
            raise PolicyFailure('carton_well_geometry_unresolved')
        distances = np.linalg.norm(centres[:, None, :2]-centres[None, :, :2], axis=-1)
        if min(distances[np.triu_indices(4, 1)]) < .025:
            raise PolicyFailure('carton_well_geometry_unresolved')
    return parts
