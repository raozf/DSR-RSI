"""Fast online System 1 actor around a DM0.5-compatible adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from dsr.harness.runtime import HarnessController


class DM05Policy(Protocol):
    def predict(self, observation: Mapping[str, Any], instruction: str, **kwargs: Any) -> Any: ...


@dataclass
class System1State:
    task: str = ""
    step: int = 0
    history: list[Mapping[str, Any]] = field(default_factory=list)
    selected_skill: str | None = None
    memory_ids: list[str] = field(default_factory=list)


class System1Actor:
    """Keep per-episode state out of the VLA adapter and provide one call site."""

    def __init__(self, policy: DM05Policy, *, harness: Any = None, memory: Any = None, skills: Any = None):
        self.policy = policy
        self.harness = harness
        self.controller = HarnessController(harness) if isinstance(harness, Mapping) else None
        self.memory = memory
        self.skills = skills
        self.bound_skill = None
        self.state = System1State()
        self.current_instruction = ""

    def reset(self, task: str) -> None:
        self.state = System1State(task=task)
        self.current_instruction = task
        self.bound_skill = None
        if self.controller is not None:
            self.controller.reset(task)
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(task)

    def act(self, observation: Mapping[str, Any]) -> Any:
        self.state.history.append(dict(observation))
        self.state.history = self.state.history[-4:]
        policy_instruction = self.state.task
        if self.controller is not None:
            policy_instruction = self.controller.instruction(self.state.task, observation)
            self.state.selected_skill = self.controller.selected_skill
        self.current_instruction = policy_instruction
        kwargs: dict[str, Any] = {"history": tuple(self.state.history[-4:])}
        if self.harness is not None:
            kwargs["harness"] = self.harness
        if self.state.selected_skill:
            kwargs["skill"] = self.state.selected_skill
            # DM05 sends this value as request metadata as well as in the
            # instruction.  OpenDM currently uses the latter for inference,
            # while a PhysicalRSI-compatible policy service can bind the
            # registered skill by this stable episode choice.
            kwargs["skill_choice"] = self.state.selected_skill
        if self.controller is not None:
            kwargs["harness_state"] = self.controller.audit()
        if self.memory is not None and hasattr(self.memory, "retrieve"):
            failure_type = observation.get("failure_type")
            kwargs["retrieved_experience"] = self.memory.retrieve(
                task=self.state.task, failure_type=str(failure_type) if failure_type else None, top_k=4
            )
            self.state.memory_ids = [getattr(item, "record_id", "") for item in kwargs["retrieved_experience"]]
        if self.skills is not None and hasattr(self.skills, "search"):
            kwargs["available_skills"] = self.skills.search(self.state.task, top_k=4)
        executor = self.policy
        binder = getattr(self.skills, "bind", None) if self.skills is not None else None
        if self.state.selected_skill and callable(binder):
            bound_name = getattr(self.bound_skill, "name", None)
            if self.bound_skill is None or bound_name != self.state.selected_skill:
                # Binding is deliberately explicit.  An unknown skill is a
                # configuration error, not a request to silently fall back to
                # a different controller.
                self.bound_skill = binder(self.state.selected_skill)
            executor = self.bound_skill
        output = executor.predict(observation, policy_instruction, **kwargs)
        self.state.step += 1
        return output

    def should_terminate(self, observation: Mapping[str, Any]) -> bool:
        return self.controller.should_terminate(observation) if self.controller is not None else False

    def after_action(self, observation: Mapping[str, Any]) -> None:
        if self.controller is not None:
            self.controller.after_action(observation)

    def audit(self) -> dict[str, Any]:
        result = self.controller.audit() if self.controller is not None else {"parsed_counts": {}}
        describe = getattr(self.bound_skill, "describe", None)
        if callable(describe):
            result["execution_skill"] = describe()
        return result

    def select_skill(self, name: str | None) -> None:
        self.state.selected_skill = name
