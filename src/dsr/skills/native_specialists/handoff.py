"""One observable Code-stage handoff, without resetting the physical episode."""
from copy import deepcopy
from .control import PolicyFailure


PAIRS = {
    'eggs': ('egg-fill-v1', 'egg-lid-finish-v1', 230),
    'toast': ('toast-insert-v1', 'toast-lever-finish-v1', 180),
    'laptop': ('headset-place-v1', 'laptop-store-finish-v1', 330),
}


def boundary(robot, context, predicate, protected, reserve):
    if robot.holding or robot.hypotheses:
        raise PolicyFailure('handoff_ownership_not_empty')
    # Opening and short holds are real actions charged to the same native budget.
    for arm in ('left','right'):
        yield from robot.grip(arm,True)
    baseline=None
    for _ in range(3):
        yield from robot.wait(2)
        yield from robot.observe()
        predicate(robot,context)
        positions={identity:robot.object(identity)['position'] for identity in protected}
        if baseline is None:baseline=deepcopy(positions)
        else:
            import numpy as np
            if any(np.linalg.norm(np.array(positions[i])-baseline[i])>.006 for i in protected):
                raise PolicyFailure('handoff_subgoal_not_stable')
    state=robot.scene.get('robot_state',{})
    if robot.closed_arms or any(state.get(arm+'_ee_joint_state',[0])[0]<.95 for arm in ('left','right')):
        raise PolicyFailure('handoff_measured_gripper_not_open')
    remaining=robot.memory['max_steps']-robot.native_steps
    if remaining<reserve:raise PolicyFailure('handoff_insufficient_remaining_budget')
    for identity in protected:
        robot.protect(identity,.008)
    return dict(channel=robot.channel,sequence=robot.sequence,consumed_steps=robot.native_steps,
                total_step_limit=robot.memory['max_steps'],remaining_steps=remaining,
                holding={},protected=sorted(robot.protected),context=deepcopy(context),
                observations='three_fresh_rgb_and_measured_robot_states',native_success=None)


def composed(kind, module):
    first,second,reserve=PAIRS[kind]
    if kind=='toast': a,b,check=module.insert_slices,module.finish_lever,module.check_slots
    elif kind=='laptop': a,b,check=module.place_headset,module.finish_laptop,module.check_headset
    else: a,b,check=module.fill_wells,module.finish_lid,module.check_wells

    def policy(robot,memory):
        robot.controller=first
        context=yield from a(robot,memory)
        protected=(context['bread_ids'] if kind=='toast' else
                   [context['headset']] if kind=='laptop' else context['egg_ids'])
        receipt=yield from boundary(robot,context,check,protected,reserve)
        yield from robot.command('handoff',from_controller=first,to_controller=second,receipt=receipt)
        robot.controller=second
        yield from robot.observe()
        check(robot,context)
        return (yield from b(robot,memory,context))
    return policy
