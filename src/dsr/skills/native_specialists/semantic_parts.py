"""Independent RGB part evidence for a bounded, agent-selected reader.

Parent membership is checked before overlap between competing part hypotheses.
A whole-object response is not a part, even if it has the highest confidence.
Neither expected counts nor task/checker coordinates create missing geometry.
"""
from types import SimpleNamespace

import numpy as np

from .control import PolicyFailure
from .instances import instance_proposals, within_parent


PROFILE = 'independent_parts'
# Only these independently diagnosed nouns use SAM. Other features retain
# their existing reader and may still fail; the agent cannot invent a prompt.
PART_QUERIES = {
    ('headset', 'headphone headband'): ('headband', .65),
    ('headset', 'headphone ear cushions'): ('ear cushions', .65),
    ('stand', 'headphone support cradle'): ('top of headphone stand', .65),
    ('toaster', 'toaster slot'): ('toaster slot', .40),
}
BODY_QUERIES = {'headset': 'headphones', 'stand': 'wooden headphone stand',
                'toaster': 'toaster'}


def part_instances(result, shape, parent, *, max_parent_fraction, diagnostic):
    """Filter independent parent evidence, then apply unchanged instance gates.

An independently measured parent mask is required. A small part must lie in
the parent's pixels and occupy less than the fixed maximum parent fraction.
Near-duplicate, competing partial, or merged masks still pass through the
original overlap gate as a set. We never select the requested top-N scores.
"""
    masks = np.asarray(result.masks)
    boxes = np.asarray(result.boxes_xyxy_px, float)
    scores = np.asarray(result.scores, float)
    count = len(scores) if scores.ndim == 1 else -1
    if (tuple(result.image_size_wh) != (shape[1], shape[0]) or
            masks.shape != (count, *shape) or masks.dtype != np.bool_ or
            boxes.shape != (count, 4) or not np.isfinite(boxes).all() or
            not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise PolicyFailure('instance_response_contract')
    support = np.asarray(parent.get('_instance_mask'))
    if (support.shape != tuple(shape) or support.dtype != np.bool_
            or support.sum() < 30 or not 0 < max_parent_fraction < 1):
        raise PolicyFailure('semantic_parent_mask_required')
    selected = []
    for i, (mask, score) in enumerate(zip(masks, scores, strict=True)):
        area = int(mask.sum())
        record = dict(index=i, confidence=float(score), pixels=area)
        diagnostic.append(record)
        if score < .3:
            record['rejected'] = 'confidence'; continue
        if area < 30:
            record['rejected'] = 'instance_mask_support'; continue
        yy, xx = np.nonzero(mask)
        box = np.array([xx.min(), yy.min(), xx.max()+1, yy.max()+1], float)
        membership = float(np.count_nonzero(mask & support) / area)
        fraction = float(area / support.sum())
        record.update(box=box.tolist(), parent_membership=membership,
                      parent_fraction=fraction)
        if not within_parent(box, parent['box']) or membership < .80:
            record['rejected'] = 'part_outside_parent'; continue
        if fraction >= max_parent_fraction:
            record['rejected'] = 'whole_parent_as_part'; continue
        selected.append(i)
    view = SimpleNamespace(masks=masks[selected], boxes_xyxy_px=boxes[selected],
                           scores=scores[selected], image_size_wh=result.image_size_wh)
    return instance_proposals(view, shape)


def distinct_body_rows(rows, queries):
    """Reject shared independently segmented pixels across semantic categories.

    Detection boxes can enclose different objects at different depths. Their
    overlap alone cannot establish a shared identity. Those objects still
    require the ordinary count, feature and physical geometry contracts.
    """
    for i, group in enumerate(rows):
        for j in range(i):
            if queries[i][0] == queries[j][0]:
                continue
            for one in group:
                for other in rows[j]:
                    a, b = one.get('mask'), other.get('mask')
                    if a is None or b is None:
                        continue
                    a,b=np.asarray(a),np.asarray(b)
                    if a.dtype!=np.bool_ or b.dtype!=np.bool_ or a.ndim!=2 or a.shape!=b.shape:
                        raise PolicyFailure('instance_response_contract')
                    overlap = np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))
                    if overlap >= .70:
                        raise PolicyFailure('cross_class_parent_identity', queries[i][0] + '/' + queries[j][0])


def headset_bridge(band, cups):
    """Resolve a visible bridge patch from the band and two distinct ear pads.

The two cup centres define an unsigned transverse axis. The far arc of the
independently segmented band supplies a real contact patch. No image left/
right label, body PCA axis or unobserved point is used as a semantic landmark.
"""
    cloud = np.asarray(band['points'], float)
    centres = np.asarray([p['point'] for p in cups], float)
    if (cloud.ndim != 2 or cloud.shape[1:] != (3,) or len(cloud) < 30
            or centres.shape != (2, 3) or not np.isfinite(cloud).all()
            or not np.isfinite(centres).all()):
        raise PolicyFailure('headset_part_geometry')
    delta = centres[1] - centres[0]
    separation = float(np.linalg.norm(delta))
    if not .04 <= separation <= .30:
        raise PolicyFailure('ear_cup_separation')
    axis = delta / separation
    middle = centres.mean(axis=0)
    offset = cloud - middle
    along = offset @ axis
    across = offset - along[:, None] * axis
    distances = np.linalg.norm(across, axis=1)
    # A visible far arc must exist near the centre of the two cups, on one
    # side of their axis. A cup mask mislabeled as the band fails this gate.
    eligible = (abs(along) <= separation * .30) & (distances >= np.quantile(distances, .80))
    patch = cloud[eligible]
    if len(patch) < 30:
        raise PolicyFailure('headband_bridge_support')
    point = np.median(patch, axis=0)
    radial = point - middle; radial -= axis * np.dot(radial, axis)
    radius = np.linalg.norm(radial)
    if not .02 < radius < .25 or np.max(np.linalg.norm(patch-point, axis=1)) > .06:
        raise PolicyFailure('headband_bridge_geometry')
    if np.min((patch-middle) @ (radial/radius)) < .5 * radius:
        raise PolicyFailure('headband_bridge_side_ambiguous')
    return point, axis, patch
