"""Bind startup and each action tick with PhysicalRSI's memory compiler.

This is composition of operations *inside one controller*. It neither changes
RoboDojo's selector nor chains separately registered complete code policies.
"""
import hashlib
import importlib
import json
from pathlib import Path

from .catalog import POLICIES
from .control import PolicyFailure


def specification(name, revision):
    return dict(schema='physicalrsi.skill-program-memory/v1',revision=revision,
        programs={'controller.start':{'steps':[
            {'operation':name+'.check','revision':revision},
            {'operation':name+'.instantiate','revision':revision}]},
            'controller.tick':{'steps':[
                {'operation':name+'.observation.check','revision':revision},
                {'operation':name+'.scheduler.advance','revision':revision},
                {'operation':name+'.action.check','revision':revision}]}},
        policy=name,parameters_file=name+'.memory.json',
        permitted_primitives=['observe','move','grip','wait'],
        effects=['left-arm','right-arm','scene-objects'],
        native_success_authority='unchanged native RoboDojo checker')


def compile_start(name, parameters, spec, workspace, expected_revision):
    from PhysicalRSI_core.contracts import Contract, Operation
    from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
    from PhysicalRSI.Embodied_Harness.skills.memory_program import compile_program
    if spec != specification(name,expected_revision):
        raise ValueError('Unexpected operation memory or controller revision')
    def check(robot, context):
        if not robot.instruction.strip():raise PolicyFailure('missing_instruction')
        if robot.memory != parameters:raise PolicyFailure('parameter_revision_mismatch')
        return robot
    def instantiate(robot, context):
        module=importlib.import_module(__package__+'.policies.'+POLICIES[name]['module'])
        return module.policy(robot,parameters.copy())
    operations={name+'.check':Operation(name+'.check',expected_revision,
        Contract(name+'.request'),Contract(name+'.checked-request'),check),
        name+'.instantiate':Operation(name+'.instantiate',expected_revision,
        Contract(name+'.checked-request'),Contract(name+'.command-generator'),instantiate,
        frozenset({'episode-state'}))}
    snapshot=MemoryStore(workspace).snapshot(spec)
    return compile_program(snapshot,'controller.start',operations)


def compile_tick(name, spec, workspace, expected_revision, episode):
    """Freeze executable feedback/advance/emission stages, not just startup.

    Task-specific generators still own their coupled physical state. These
    stages do not expose a cross-controller pause/resume or holding API.
    """
    from PhysicalRSI_core.contracts import Contract, Operation
    from PhysicalRSI.Embodied_Harness.memory.store import MemoryStore
    from PhysicalRSI.Embodied_Harness.skills.memory_program import compile_program
    if spec != specification(name, expected_revision):
        raise ValueError('Unexpected tick operation memory or controller revision')
    request = Contract(name+'.observed-episode')
    checked = Contract(name+'.checked-episode')
    proposed = Contract(name+'.proposed-action', frame='world', embodiment='arx_x5')
    action = Contract(name+'.validated-action', frame='world', embodiment='arx_x5')
    operations = {}
    for suffix, before, after, callback in (
        ('observation.check', request, checked, episode.check_observation),
        ('scheduler.advance', checked, proposed, lambda value: value._advance()),
        ('action.check', proposed, action, episode.check_action),
    ):
        key = name+'.'+suffix
        def invoke(value, context, callback=callback, key=key):
            episode.event(operation=key, operation_revision=expected_revision)
            return callback(value)
        operations[key] = Operation(key, expected_revision, before, after, invoke,
                                    frozenset({'episode-state'}))
    snapshot = MemoryStore(workspace).snapshot(spec)
    return compile_program(snapshot, 'controller.tick', operations)
