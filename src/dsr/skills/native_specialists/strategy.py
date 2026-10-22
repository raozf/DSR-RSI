"""Capability adapters; strategy/order is selected by the agent, never task IDs."""
from .control import PolicyFailure
from .handoff import boundary
from .resumption import resume_placement

if __package__ == 'dsr.skills.native_specialists':
    from dsr.harness.strategy import Capability, Strategy, StrategyLimits
else:
    from .strategy_runtime import Capability, Strategy, StrategyLimits


def capabilities(module, revision):
    def whole(robot, memory, value):
        return (yield from module.policy(robot, memory))

    def inspect_state(robot, memory, value):
        yield from robot.observe()

    def clear_view(robot, memory, value):
        yield from robot.recover(PolicyFailure('agent_requested_view_recovery'))
        yield from robot.observe()

    def independent_parts(robot, memory, value):
        # A code-owned reader profile, not agent-supplied prompts/geometry.
        # It persists for subsequent subgoals under the same tracking ledger.
        required=getattr(robot,'perception_status',{}).get('required')
        yield from robot.command('select_perception', profile='independent_parts')
        yield from robot.observe(required=required)

    def withdraw_contact(robot, memory, value):
        # Known fist contact is distinct from a grasp. Unknown ownership is
        # never opened or discarded to make a new plan appear executable.
        if (robot.holding or robot.hypotheses or not robot.closed_arms
                or not robot.stage.startswith(('close_lid:', 'press_latch:'))):
            raise PolicyFailure('contact_withdrawal_precondition')
        for arm in sorted(robot.closed_arms):
            measured = list(robot.ee[arm])
            target = [*measured[:2], measured[2] + .08]
            yield from robot.move(arm, target, measured[3:], slow=True)
            yield from robot.grip(arm, True)
        yield from robot.observe()

    rows = [Capability('complete_policy', 'Original complete Code Policy, initial physical state only.',
                       'whole_policy', whole, terminal=True, revision=revision, accepts_ownership=True,
                       precondition=lambda r,v:not (r.holding or r.hypotheses)),
            Capability('inspect_state', 'Obtain fresh RGB and measured state; retain all ownership and progress.',
                       'recovery', inspect_state, accepts_ownership=True, revision=revision),
            Capability('clear_view', 'Existing bounded view recovery; only move open unoccupied arms, then observe.',
                       'recovery', clear_view, accepts_ownership=True, revision=revision),
            Capability('withdraw_contact', 'Withdraw a known lid/latch fist contact before opening it. '
                       'Requires that contact stage and no held or uncertain object.',
                       'recovery', withdraw_contact, accepts_ownership=True, revision=revision),
            Capability('resume_placement', 'Complete the code-owned pending placement only with a fresh '
                       'verified attachment or measured open-gripper stable release, and unchanged support. '
                       'Never adopts an uncertain grasp. Then replan the original stage to verify its subgoal.',
                       'recovery', resume_placement, accepts_ownership=True, revision=revision)]
    if 'independent_parts' in getattr(module,'PERCEPTION_PROFILES',()):
        rows.append(Capability('inspect_independent_parts',
            'Switch to independent SAM body/part masks for diagnosed semantic ambiguity, then observe. '
            'Reject whole objects as parts, shared parent identities and overlapping hypotheses. '
            'Retain all ownership, tracking, progress and thresholds. Missing hinge/floor evidence may remain unresolved.',
            'recovery', independent_parts, accepts_ownership=True, revision=revision))
    # Modules declare executable subgoal contracts, not a strategy or fixed
    # plan. Any new module may expose this same interface. No benchmark name,
    # success-rate table, difficulty threshold or task router appears here.
    for spec in getattr(module, 'STRATEGY_STAGES', ()):
        def run(robot, memory, value, spec=spec):
            if spec.get('input'):
                yield from robot.observe()
                spec['check_input'](robot, value)
                return (yield from spec['run'](robot, memory, value))
            return (yield from spec['run'](robot, memory))

        def verify(robot, value, spec=spec):
            if spec.get('terminal'):
                # The registered continuation itself checks its task-specific
                # physical postconditions and returns home. This is still a
                # controller claim, never the native checker's success result.
                yield from robot.observe()
                if robot.holding or robot.hypotheses or robot.closed_arms:
                    raise PolicyFailure('strategy_terminal_ownership')
            else:
                yield from boundary(robot, value, spec['check'], spec['protected'](value), spec['reserve'])
            return True

        rows.append(Capability(spec['name'], spec['description'], 'verified_stage', run,
                    input_contract=spec.get('input', ''), output_contract=spec['output'],
                    verify=verify, terminal=spec.get('terminal', False), revision=revision,
                    resume_contract=spec['resume']))
    return rows


def supervisor(module, revision, settings):
    if set(settings) != {'protocol', 'limits'} or settings['protocol'] != 'dsr.execution-strategy/v1':
        raise ValueError('Frozen agent-selected strategy settings required')
    return Strategy(capabilities(module, revision), StrategyLimits(**settings['limits']))
