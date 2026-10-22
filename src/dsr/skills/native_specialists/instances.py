"""Validate instance masks without merging adjacent objects into one cloud."""
import numpy as np

from .control import PolicyFailure


def instance_proposals(result, shape, threshold=.3, *, separate_components=False):
    masks = np.asarray(result.masks)
    boxes = np.asarray(result.boxes_xyxy_px, float)
    scores = np.asarray(result.scores, float)
    count = len(scores) if scores.ndim == 1 else -1
    if (tuple(result.image_size_wh) != (shape[1], shape[0]) or
            masks.shape != (count, *shape) or masks.dtype != np.bool_ or
            boxes.shape != (count, 4) or not np.isfinite(boxes).all() or
            not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise PolicyFailure('instance_response_contract')
    masks = masks[scores >= threshold]
    scores = scores[scores >= threshold]
    if not len(masks):
        return []
    # SAM can return the same visible object more than once. Suppress only
    # near-identical full masks, before computing shared pixels. A merged
    # pair or competing partial mask must still fail the overlap gate.
    keep=[]
    for i in np.argsort(-scores,kind='stable'):
        if any(np.count_nonzero(masks[i]&masks[j]) /
               max(1,np.count_nonzero(masks[i]|masks[j])) >= .90 for j in keep):
            continue
        keep.append(i)
    masks,scores=masks[keep],scores[keep]
    if separate_components:
        # A text query can return a disconnected pair of holes as well as each
        # hole separately. Split only observed connected pixels; never complete
        # a grid or cut a connected mask to reach the desired object count.
        import cv2
        components = []
        for mask, score in zip(masks, scores, strict=True):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            for index in range(1, count):
                if stats[index, cv2.CC_STAT_AREA] >= 30:
                    components.append((labels == index, float(score)))
        unique = []
        for mask, score in sorted(components, key=lambda x: -x[1]):
            # Near-identical masks are alternate measurements of one hole.
            # Partly overlapping hypotheses still fail the ambiguity gate below.
            if any(np.count_nonzero(mask & old) / np.count_nonzero(mask | old) >= .8
                   for old, _ in unique):
                continue
            unique.append((mask, score))
        if not unique:
            return []
        masks = np.stack([m for m, _ in unique])
        scores = np.array([s for _, s in unique])
    # Ambiguous pixels do not belong to either cloud. Large overlaps suggest
    # duplicate/merged instances; rejecting them preserves the count gate.
    shared = masks.sum(axis=0) > 1
    proposals = []
    for mask, score in zip(masks, scores, strict=True):
        if mask.sum() < 30 or np.count_nonzero(mask & shared) > .25 * mask.sum():
            raise PolicyFailure('ambiguous_instance_mask')
        mask = mask & ~shared
        yy, xx = np.nonzero(mask)
        if len(xx) < 30:
            raise PolicyFailure('instance_mask_support')
        # Bounds come from actual accepted pixels, not a possibly loose box.
        box = np.array([xx.min(), yy.min(), xx.max()+1, yy.max()+1], float)
        proposals.append(dict(box=box, confidence=float(score), mask=mask,
                              source='sam3_connected_component' if separate_components else 'sam3_instance_mask'))
    return proposals


def _instance_proposals_with_base_evidence(result, shape, wells, previous=None):
    '''Qualify tray hypotheses against four fresh wells before the normal gate.'''
    from .carton import select_base
    masks = np.asarray(result.masks)
    boxes = np.asarray(result.boxes_xyxy_px, float)
    scores = np.asarray(result.scores, float)
    count = len(scores) if scores.ndim == 1 else -1
    if (tuple(result.image_size_wh) != (shape[1], shape[0]) or
            masks.shape != (count, *shape) or masks.dtype != np.bool_ or
            boxes.shape != (count, 4) or not np.isfinite(boxes).all() or
            not np.isfinite(scores).all() or
            np.any((scores < 0) | (scores > 1))):
        raise PolicyFailure('instance_response_contract')
    if len(wells) != 4:
        return instance_proposals(result, shape)

    well_masks = []
    for well in wells:
        mask = np.asarray(well.get('mask'))
        if (mask.dtype != np.bool_ or mask.shape != tuple(shape)
                or mask.sum() < 30):
            raise PolicyFailure('carton_base_evidence_unresolved')
        yy, xx = np.nonzero(mask)
        bounds = np.array([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1], float)
        if not within_parent(bounds, well.get('box'), 1.):
            raise PolicyFailure('carton_base_evidence_unresolved')
        well_masks.append(mask)
    well_union = np.stack(well_masks).sum(axis=0)
    if np.any(well_union > 1):
        raise PolicyFailure('carton_base_evidence_unresolved')

    import cv2
    kernel = np.ones((7, 7), np.uint8)
    accepted = []
    for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
        if score < .3:
            continue
        view = type('_InstanceResponseView', (), {})()
        view.masks = mask[None, ...]
        view.boxes_xyxy_px = boxes[index:index + 1]
        view.scores = scores[index:index + 1]
        view.image_size_wh = result.image_size_wh
        try:
            rows = instance_proposals(view, shape)
        except PolicyFailure as error:
            if error.reason == 'ambiguous_instance_mask':
                continue
            raise
        if len(rows) != 1:
            continue
        row = rows[0]
        try:
            select_base([row], wells, previous)
        except PolicyFailure as error:
            if error.reason == 'carton_base_evidence_unresolved':
                continue
            raise
        supported = True
        for part in well_masks:
            dilated = cv2.dilate(part.astype(np.uint8), kernel) > 0
            ring = dilated & ~well_union.astype(bool)
            area = int(ring.sum())
            pixels = int(np.count_nonzero(row['mask'] & ring))
            if area < 30 or pixels < max(12, int(np.ceil(.2 * area))):
                supported = False
                break
        if supported:
            accepted.append(index)
    if not accepted:
        return []
    view = type('_InstanceResponseView', (), {})()
    view.masks = masks[accepted]
    view.boxes_xyxy_px = boxes[accepted]
    view.scores = scores[accepted]
    view.image_size_wh = result.image_size_wh
    return instance_proposals(view, shape)


def within_parent(box, parent_box, fraction=.8):
    box = np.asarray(box, float)
    parent = np.asarray(parent_box, float)
    if (box.shape != (4,) or parent.shape != (4,) or
            not np.isfinite(box).all() or not np.isfinite(parent).all() or
            np.any(box[2:] <= box[:2]) or np.any(parent[2:] <= parent[:2])):
        return False
    overlap = np.maximum(0, np.minimum(box[2:], parent[2:]) - np.maximum(box[:2], parent[:2]))
    return bool(np.prod(overlap) >= fraction * np.prod(box[2:] - box[:2]))
