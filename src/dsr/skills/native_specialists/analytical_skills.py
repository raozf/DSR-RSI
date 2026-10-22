"""Use the same reviewed skill functions in development and frozen workers."""
try:
    from .analytical.geometry import hinge_increment, pivot_end_effector
    from .analytical.language import pour_order
    from .analytical.memory import reserve_budget
except ModuleNotFoundError as error:
    if error.name != __package__+'.analytical':
        raise
    from dsr.skills.task_specialists.geometry import hinge_increment, pivot_end_effector
    from dsr.skills.task_specialists.language import pour_order
    from dsr.skills.task_specialists.memory import reserve_budget
