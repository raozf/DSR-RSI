"""Causal, immutable episode ledgers. Missing visual evidence remains unknown."""
from copy import deepcopy
import numpy as np
from .geometry import SkillError, scalar


def identities(values, name):
    if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise SkillError(name + ': unique visual identities required')
    return list(values)


def stamp(value, previous=None):
    if set(value) != {'episode', 'frame', 'time_s', 'clock'} or not isinstance(value['episode'], str) or not value['episode']:
        raise SkillError('episode/frame/time stamp required')
    if value['clock'] not in ('simulation', 'real_sensor'):
        raise SkillError('explicit physical clock required; wall inference time is not motion time')
    if type(value['frame']) is not int or value['frame'] < 0:
        raise SkillError('invalid frame')
    scalar(value['time_s'], 'time_s')
    if previous is not None and (value['episode'] != previous['episode'] or value['clock'] != previous['clock'] or value['frame'] <= previous['frame'] or value['time_s'] <= previous['time_s']):
        raise SkillError('stale observation or different episode')
    return deepcopy(value)


def sequence_update(memory, *, observation, contained_ids, expected_ids, fully_visible, max_gap_s):
    """Remember a demonstrated insertion order without reconstructing hidden time.

    Start with an observed empty destination. Simultaneous arrivals, missed
    frames or disappearing reference objects invalidate the order permanently.
    """
    allowed = identities(expected_ids, 'expected_ids')
    if not allowed:
        raise SkillError('empty demonstration')
    current = identities(contained_ids, 'contained_ids')
    if not set(current) <= set(allowed):
        raise SkillError('unexpected demonstrated object')
    gap = scalar(max_gap_s, 'max_gap_s', positive=True)
    old = deepcopy(memory) if memory else None
    mark = stamp(observation, old['stamp'] if old else None)
    if type(fully_visible) is not bool:
        raise SkillError('explicit visibility required')
    if old and old['expected_ids'] != allowed:
        raise SkillError('demonstration identity set changed')
    order = list(old['order']) if old else []
    reason = old.get('invalid_reason') if old else None
    previous = set(old['contained_ids']) if old else set()
    if not fully_visible:
        reason = reason or 'demonstration_occluded'
    elif old is None and current:
        reason = 'initial_empty_destination_not_observed'
    elif old and mark['time_s']-old['stamp']['time_s'] > gap:
        reason = reason or 'demonstration_observation_gap'
    elif not previous <= set(current):
        reason = reason or 'reference_object_removed'
    elif len(set(current)-previous) > 1:
        reason = reason or 'simultaneous_arrivals_order_unknown'
    elif not reason:
        order.extend(sorted(set(current)-previous))
    return dict(stamp=mark, expected_ids=allowed, contained_ids=current, order=order,
                invalid_reason=reason, complete=not reason and len(order) == len(allowed))


def occupancy_update(memory, *, observation, assignments, released_ids, min_frames=3):
    """Reconcile released item -> slot observations without erasing occlusions.

    A confirmed placement loses its certificate on a contradictory observation.
    Occlusion preserves the historical record but never yields a fresh permit
    to insert into, or declare completion of, that region.
    """
    if type(min_frames) is not int or min_frames < 3:
        raise SkillError('at least three fresh frames required')
    if not isinstance(assignments, dict) or not assignments:
        raise SkillError('explicit item observations required')
    identities(list(assignments), 'objects')
    slots = [v for v in assignments.values() if v is not None]
    identities(slots, 'observed slots')
    released = set(identities(released_ids, 'released_ids'))
    if not released <= assignments.keys():
        raise SkillError('release identity not observed')
    old = deepcopy(memory) if memory else {'stamp': None, 'items': {}}
    if old['items'] and set(old['items']) != set(assignments):
        raise SkillError('item identity set changed')
    mark = stamp(observation, old['stamp'])
    result = {}
    invalidated = []
    for item, slot in assignments.items():
        prior = old['items'].get(item, {})
        certified = prior.get('certified_slot')
        if slot is not None and certified is not None and (slot != certified or item not in released):
            invalidated.append(item)
            certified = None
        if slot is None:
            count = 0
        else:
            count = (prior.get('consecutive_frames', 0)+1 if prior.get('last_slot') == slot else 1) if item in released else 0
            if count >= min_frames:
                certified = slot
        result[item] = dict(last_slot=slot, consecutive_frames=count,
                            certified_slot=certified,
                            fresh_certificate=slot is not None and item in released and count >= min_frames)
    # A newly observed item occupying an occluded item's historical slot
    # invalidates that historical certificate too; never double-book a slot.
    visible_owners = {s: i for i, s in assignments.items() if s is not None}
    for item, record in result.items():
        certified = record['certified_slot']
        if certified in visible_owners and visible_owners[certified] != item:
            record['certified_slot'] = None
            if item not in invalidated:
                invalidated.append(item)
    return dict(stamp=mark, items=result, invalidated=sorted(invalidated))


def cycle_update(memory, *, observation, observed_state, expected_cycles, max_gap_s):
    """Exactly observed released -> pressed -> released cycles, with unknowns.

    The same mechanism can represent an upright/pour/upright cycle after
    task-specific visual classification. It does not inspect reward events.
    """
    if observed_state not in {'released', 'pressed', 'unknown'}:
        raise SkillError('unknown event vocabulary')
    if type(expected_cycles) is not int or expected_cycles < 1:
        raise SkillError('positive cycle count required')
    gap = scalar(max_gap_s, 'max_gap_s', positive=True)
    old = deepcopy(memory) if memory else None
    mark = stamp(observation, old['stamp'] if old else None)
    if old and old['expected_cycles'] != expected_cycles:
        raise SkillError('cycle contract changed')
    count = old['completed_cycles'] if old else 0
    reason = old.get('invalid_reason') if old else None
    if observed_state == 'unknown':
        reason = reason or 'event_occluded_do_not_repeat'
    elif old is None and observed_state != 'released':
        reason = 'initial_released_state_unobserved'
    elif old and mark['time_s']-old['stamp']['time_s'] > gap:
        reason = reason or 'event_gap_do_not_repeat'
    elif old and not reason and old['state'] == 'pressed' and observed_state == 'released':
        count += 1
    if count > expected_cycles:
        reason = 'extra_cycle'
    return dict(stamp=mark, state=observed_state, completed_cycles=count,
                expected_cycles=expected_cycles, invalid_reason=reason,
                complete=not reason and count == expected_cycles and observed_state == 'released')


def reserve_budget(*, limit, used, proposed, finish_reserve, observation_reserve):
    """Allocate physical steps including settling/observation and terminal work."""
    values = (limit, used, proposed, finish_reserve, observation_reserve)
    if any(type(v) is not int or v < 0 for v in values) or limit < 1:
        raise SkillError('nonnegative integer physical-step budgets required')
    needed = proposed+finish_reserve+observation_reserve
    if used+needed > limit:
        raise SkillError('insufficient steps including terminal reserve')
    return dict(remaining_after_proposal=limit-used-proposed,
                reserved_steps=finish_reserve+observation_reserve,
                unallocated_steps=limit-used-needed)


def first_return_update(memory, *, observation, candidates, fully_visible,
                        max_match_cost, min_match_gap, departure_frames=2):
    """Remember first RGB descriptor; require departure before a later match.

    `fully_visible` covers the reader's conveyor region, not just one box.
    Candidate IDs are visual tracks. Appearance thresholds must be calibrated
    outside this function; a cosine distance is not a success probability.
    Departure is a conservative disambiguation gate, not a native task rule.
    A distinct later arrival may overlap the first in the camera view and
    remain eligible after this gate. A native reader must evaluate missed
    intercept windows as well as wrong-object picks before using this gate.
    """
    from .geometry import unique_assignment
    if type(fully_visible) is not bool or type(departure_frames) is not int or departure_frames < 2:
        raise SkillError('explicit visibility and two departure frames required')
    identities([c['id'] for c in candidates], 'candidate tracks')
    vectors = {}
    for c in candidates:
        v = np.asarray(c['appearance'], float)
        if v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-9:
            raise SkillError('finite nonzero RGB descriptor required')
        vectors[c['id']] = v/np.linalg.norm(v)
    old = deepcopy(memory) if memory else {}
    mark = stamp(observation, old.get('stamp'))
    state = dict(old, stamp=mark)
    state.setdefault('phase', 'await_first')
    state['visible_match'] = None
    if not fully_visible:
        state['absent_frames'] = 0
        state['reason'] = 'conveyor_occluded'
        return state
    state.pop('reason', None)
    if state['phase'] == 'await_first':
        if len(candidates) > 1:
            state.update(phase='invalid', reason='first_arrival_order_unknown')
        elif candidates:
            first = candidates[0]['id']
            state.update(phase='await_departure', first_track=first,
                         first_appearance=vectors[first].tolist(), absent_frames=0)
        return state
    if state['phase'] == 'invalid':
        state['reason'] = old.get('reason', 'first_arrival_order_unknown')
        return state
    first = state['first_track']
    if state['phase'] == 'await_departure':
        state['absent_frames'] = 0 if first in vectors else state.get('absent_frames', 0)+1
        if state['absent_frames'] >= departure_frames:
            state.update(phase='await_return', departed_at_s=mark['time_s'])
        # Confirm departure before proposing a match on the next observation.
        # Do not permanently blacklist other tracks already in view: a valid
        # later arrival can overlap the first in the reader's camera region.
        return state
    options = [c['id'] for c in candidates if c['id'] != first]
    if not options:
        return state
    reference = np.asarray(state['first_appearance'])
    if any(vectors[k].shape != reference.shape for k in options):
        raise SkillError('appearance descriptor dimension changed')
    costs = [[float(np.clip(1-np.dot(reference, vectors[k]), 0, 2)) for k in options]]
    # No match or ambiguous match is a wait/reobserve condition, never an
    # arbitrary candidate choice. Other malformed input errors still raise.
    ceiling = scalar(max_match_cost, 'max_match_cost')
    gap = scalar(min_match_gap, 'min_match_gap', positive=True)
    ordered = sorted(costs[0])
    if ordered[0] > ceiling or len(ordered) > 1 and ordered[1]-ordered[0] < gap:
        state['reason'] = 'no_unique_return_match'
        return state
    selected = options[unique_assignment(costs, max_cost=ceiling, min_gap=gap)['columns'][0]]
    if state.get('matched_track') not in (None, selected):
        state['reason'] = 'matched_track_changed'
        return state
    state.update(matched_track=selected, visible_match=selected)
    return state
