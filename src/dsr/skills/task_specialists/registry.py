"""Isolated registry of executable analytical/temporal PhysicalRSI skills.

They have no physical effects. A future motion provider must explicitly bind
their feature outputs to qualified robot actions. They are not Code Policies
in RoboDojo's observation-batch -> action-chunks controller registry.
"""
from . import geometry, intercept, language, memory


FUNCTIONS = {
    'identity.global_assignment': geometry.unique_assignment,
    'geometry.functional_frame': geometry.fit_functional_frame,
    'geometry.precision_envelope': geometry.precision_envelope,
    'geometry.support_margin': geometry.support_margin,
    'contact.hinge_increment': geometry.hinge_increment,
    'geometry.pivot_end_effector': geometry.pivot_end_effector,
    'memory.demonstration_sequence': memory.sequence_update,
    'memory.released_occupancy': memory.occupancy_update,
    'memory.observed_cycle': memory.cycle_update,
    'memory.first_return': memory.first_return_update,
    'budget.terminal_reserve': memory.reserve_budget,
    'motion.fit_visible_track': intercept.fit_track,
    'motion.predict_intercept': intercept.predict_intercept,
    'language.pour_order': language.pour_order,
    'identity.color_pairing': language.color_pairing,
}


def operations():
    from PhysicalRSI_core.contracts import Contract, Operation
    from PhysicalRSI_core.infra.storage import digest
    from .programs import source_identity
    revision = digest(source_identity())
    registry = {}
    for name, function in FUNCTIONS.items():
        def invoke(request, context, function=function):
            if not isinstance(request, dict):
                raise geometry.SkillError('named skill arguments required')
            return function(**request)
        registry[name] = Operation(name, revision, Contract(name+'.request/v1'),
                                   Contract(name+'.result/v1'), invoke)
    return registry
