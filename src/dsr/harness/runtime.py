"""Episode-scoped execution harness for the native DSR-RSI adapter.

The public PhysicalRSI implementation separates a low-frequency System-2
development loop from System-1 execution.  System 1 binds one registered skill
for an episode, reads an observation and an immutable memory snapshot, and
records the resulting state.  This module is the small native adapter for that
boundary: a validated ``harness_patch`` is materialised before a rollout and
the eight fields below are interpreted on every System-1 control tick.

The strings in the list fields are intentionally a bounded, auditable rule
language.  Plain task/skill labels remain useful data; state, routing,
recovery, and termination behaviour must use the explicit ``key=value`` form.
No string is executed as Python and no field silently changes model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping

from dsr.schema import FailureType


HARNESS_LIST_FIELDS = (
    "task_plan",
    "state_memory",
    "skill_sequence",
    "recovery_rules",
    "termination_checks",
    "routing_rules",
)
HARNESS_FIELDS = ("instruction_suffix", "action_steps", *HARNESS_LIST_FIELDS)
_CLAUSE_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_-]*)\s*=\s*([^;]+)\s*")
_COMPARISON_RE = re.compile(
    r"^(?P<path>(?:state|memory)\.[A-Za-z0-9_.-]+)\s*"
    r"(?P<op>==|!=|>=|<=|>|<)\s*(?P<expected>.+)$",
    re.IGNORECASE,
)
_SAFE_TERMINATION = {"success", "goal_reached", "episode_end", "unstable"}
_MISSING = object()
_RULE_KEYS = {
    "task_plan": {"stage", "name", "when", "done", "description", "instruction"},
    "state_memory": {"remember", "source", "value", "when", "on", "mode"},
    "skill_sequence": {"skill", "stage", "when", "description"},
    "recovery_rules": {"on", "when", "failure", "stage", "instruction"},
    "routing_rules": {"when", "on", "failure", "stage", "use", "skill"},
    "termination_checks": {"when", "on", "action", "reason"},
}


def default_native_harness() -> dict[str, Any]:
    return {
        "instruction_suffix": "",
        "action_steps": 25,
        **{name: [] for name in HARNESS_LIST_FIELDS},
    }


def normalize_native_harness(value: Any, *, check_references: bool = True) -> dict[str, Any]:
    """Validate a bounded harness and return all eight fields.

    Normalisation is deliberately strict for types and bounds.  Unknown keys
    are rejected by :func:`merge_native_harness`; this prevents an agent from
    creating an unreviewed control surface.
    """

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("harness_patch must be an object")
    unknown = set(value) - set(HARNESS_FIELDS)
    if unknown:
        raise ValueError(f"unknown harness fields: {sorted(unknown)}")
    result = default_native_harness()
    suffix = value.get("instruction_suffix", "")
    steps = value.get("action_steps", 25)
    if not isinstance(suffix, str) or len(suffix) > 1200:
        raise ValueError("instruction_suffix must be a string of at most 1200 characters")
    if type(steps) is not int or not 1 <= steps <= 25:
        raise ValueError("action_steps must be an integer in [1, 25]")
    result["instruction_suffix"], result["action_steps"] = suffix, steps
    for name in HARNESS_LIST_FIELDS:
        items = value.get(name, [])
        if not isinstance(items, list) or len(items) > 16:
            raise ValueError(f"{name} must be a list with at most 16 entries")
        if not all(isinstance(item, str) and item.strip() and len(item) <= 300 for item in items):
            raise ValueError(f"{name} entries must be non-empty strings of at most 300 characters")
        # Validate known rule keys now, before a candidate can reach a native
        # simulator.  Plain labels are allowed only for plan/skill lists.
        for item in items:
            parsed = parse_rule(item)
            if parsed is None:
                if "=" in item or ";" in item:
                    raise ValueError(f"malformed {name} rule: {item!r}")
                if name not in {"task_plan", "skill_sequence"}:
                    raise ValueError(
                        f"{name} entries must use key=value clauses; got {item!r}"
                    )
                continue
            invalid = set(parsed) - _RULE_KEYS[name]
            if invalid:
                raise ValueError(f"unknown {name} rule keys: {sorted(invalid)}")
            _validate_rule(name, parsed)
        result[name] = list(items)
    stages = [_stage_entry(item)[0] for item in result["task_plan"]]
    if len(stages) != len(set(stages)):
        raise ValueError("task_plan stage names must be unique")
    for name in ("skill_sequence", "recovery_rules", "routing_rules") if check_references else ():
        for item in result[name]:
            rule = parse_rule(item)
            if rule and rule.get("stage") and rule["stage"] not in stages:
                raise ValueError(f"{name} references undefined stage: {rule['stage']}")
    return result


def merge_native_harness(parent: Any, patch: Any) -> dict[str, Any]:
    """Materialise a candidate patch on top of an immutable parent."""

    if not isinstance(patch, Mapping):
        raise ValueError("harness_patch must be an object")
    base = normalize_native_harness(parent)
    unknown = set(patch) - set(HARNESS_FIELDS)
    if unknown:
        raise ValueError(f"unknown harness fields: {sorted(unknown)}")
    candidate = dict(base)
    candidate.update(patch)
    return normalize_native_harness(candidate)


def parse_rule(text: str) -> dict[str, str] | None:
    """Parse semicolon-separated ``key=value`` clauses.

    A rule returns ``None`` for a plain label.  The caller decides whether a
    plain label is a task stage/skill name or should be rejected.  Values may
    contain ``=`` after the first one, which is useful for textual predicates.
    """

    if not isinstance(text, str) or "=" not in text:
        return None
    parsed: dict[str, str] = {}
    for part in (part.strip() for part in text.split(";") if part.strip()):
        match = _CLAUSE_RE.fullmatch(part)
        if not match:
            return None
        key, value = match.group(1).lower(), match.group(2).strip()
        if not value:
            return None
        if key in parsed:
            raise ValueError(f"duplicate rule key: {key}")
        parsed[key] = value
    return parsed or None


def _validate_rule(name: str, rule: Mapping[str, str]) -> None:
    """Reject syntactically valid rules with no executable target."""

    if name == "task_plan" and not (rule.get("stage") or rule.get("name")):
        raise ValueError("task_plan rule requires stage=name")
    if name == "state_memory":
        if not rule.get("remember") or ("source" in rule) == ("value" in rule):
            raise ValueError("state_memory requires remember and exactly one source or value")
        if rule.get("mode", "latest") not in {"first", "latest"}:
            raise ValueError("state_memory mode must be first or latest")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", rule["remember"]):
            raise ValueError("state_memory remember must be a simple key")
        if "source" in rule and not rule["source"].startswith(("state.", "memory.")):
            raise ValueError("state_memory source must be an explicit state or memory path")
    if name == "skill_sequence" and not rule.get("skill"):
        raise ValueError("skill_sequence rule requires skill=name")
    if name in {"routing_rules", "recovery_rules", "termination_checks"}:
        if not any(rule.get(key) for key in ("when", "on", "failure")):
            raise ValueError(f"{name} requires a condition")
    if name == "routing_rules" and not (rule.get("use") or rule.get("skill")):
        raise ValueError("routing_rules requires a registered use or skill target")
    if name == "recovery_rules" and not rule.get("instruction"):
        raise ValueError("recovery_rules requires instruction=text")
    if name == "termination_checks" and rule.get("action", "stop") != "stop":
        raise ValueError("termination_checks can only stop execution, never declare success")

    for key in ("when", "on", "failure", "done"):
        if key in rule:
            _validate_predicate(rule[key])


def _validate_predicate(predicate: str) -> None:
    value = predicate.strip()
    if value.lower() in _SAFE_TERMINATION | {item.value.lower() for item in FailureType}:
        return
    if re.fullmatch(r"step>=\d+", value) and 0 <= int(value[6:]) <= 10000:
        return
    if re.fullmatch(r"stage:[A-Za-z0-9_. -]+", value):
        return
    if re.fullmatch(r"(?:state|memory)\.[A-Za-z0-9_.-]+", value):
        return
    if _COMPARISON_RE.fullmatch(value):
        return
    raise ValueError(f"unsupported harness predicate: {predicate!r}")


def content_revision(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _path_get(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for component in path.split("."):
        if hasattr(current, "tolist"):
            current = current.tolist()
        if isinstance(current, Mapping):
            if component not in current:
                return default
            current = current[component]
        elif isinstance(current, (list, tuple)) and component.isdigit():
            index = int(component)
            if index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        return math.isfinite(value) and value != 0
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on", "success"}


def _snapshot_value(value: Any) -> Any:
    """Detach episode memory from simulator-owned mutable buffers."""

    if isinstance(value, Mapping):
        return {str(key): _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _snapshot_value(value.tolist())
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _lookup(observation: Mapping[str, Any], memory: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Resolve explicit ``state.*``/``memory.*`` paths without flattening data."""

    if path.startswith("memory."):
        return _path_get(memory, path[7:], default)
    if path.startswith("state."):
        return _path_get(observation.get("state", {}), path[6:], default)
    return _path_get(observation, path, default)


def _literal(text: str) -> Any:
    value = text.strip()
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return float(value) if any(ch in value for ch in ".eE") else int(value)
    except ValueError:
        return value.strip("\"'")


def _predicate_value(observation: Mapping[str, Any], memory: Mapping[str, Any], predicate: str) -> bool:
    match = _COMPARISON_RE.fullmatch(predicate.strip())
    if not match:
        return _truth(_lookup(observation, memory, predicate.strip()))
    actual = _lookup(observation, memory, match.group("path"), _MISSING)
    expected = _literal(match.group("expected"))
    op = match.group("op")
    if actual is _MISSING:
        return False
    if op in {"==", "!="}:
        if isinstance(actual, str) or isinstance(expected, str):
            equal = str(actual).lower() == str(expected).lower()
        else:
            equal = actual == expected
        if not isinstance(equal, (bool, Real)) and hasattr(equal, "shape") and equal.shape != ():
            return False
        return bool(equal) if op == "==" else not bool(equal)
    if not isinstance(actual, Real) or isinstance(actual, bool) or not isinstance(expected, Real):
        return False
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    if op == ">=":
        return actual >= expected
    if op == "<=":
        return actual <= expected
    if op == ">":
        return actual > expected
    return actual < expected


def _matches(
    rule: Mapping[str, str],
    *,
    event: str | None,
    failure_type: str | None,
    stage: str | None,
    step: int,
    observation: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> bool:
    required_stage = rule.get("stage")
    if required_stage and not (
        stage and required_stage.lower() in {stage.lower(), str(stage)}
    ):
        return False
    predicates = [rule[key].strip() for key in ("when", "on", "failure") if rule.get(key)]
    if len(predicates) > 1:
        return all(_matches(
            {"when": predicate}, event=event, failure_type=failure_type, stage=stage,
            step=step, observation=observation, memory=memory,
        ) for predicate in predicates)
    predicate = predicates[0] if predicates else ""
    if not predicate:
        return False
    lower = predicate.lower()
    if lower.startswith("step>="):
        try:
            return step >= int(predicate[6:])
        except (TypeError, ValueError):
            return False
    if lower.startswith("state.") or lower.startswith("memory."):
        return _predicate_value(observation, memory, predicate)
    if lower.startswith("stage:"):
        return bool(stage and lower[6:] == stage.lower())
    if lower in _SAFE_TERMINATION:
        return _truth(observation.get(lower))
    # An explicit failure predicate is valid during a candidate retry, where
    # the terminal diagnosis is an immutable System-2 context rather than a
    # mutable observation field.
    if event and lower == event.lower():
        return True
    if failure_type and lower == failure_type.lower():
        return True
    return bool(stage and lower == stage.lower())


def _stage_entry(item: str) -> tuple[str, dict[str, str] | None]:
    rule = parse_rule(item)
    if rule is None:
        return item.strip(), None
    return rule.get("stage", rule.get("name", "")).strip(), rule


class HarnessController:
    """Materialized System-1 state for one episode.

    This is intentionally an episode object.  It cannot mutate the parent
    harness and it exposes an audit trail that the native adapter writes into
    ``result.json``.  The same controller drives task-stage display, state
    memory, registered skill binding, recovery guidance and safe termination.
    """

    def __init__(self, harness: Mapping[str, Any], *, failure_type: str | None = None):
        self.harness = normalize_native_harness(harness)
        self.failure_type = failure_type
        self.step = 0
        self.stage_index = 0
        self.memory: dict[str, Any] = {}
        self.selected_skill: str | None = None
        self.bound_skills: tuple[str, ...] = ()
        self.binding_frozen = False
        self.routed_skill: str | None = None
        self.bound_sequence: tuple[dict[str, str], ...] = ()
        self.termination_reason: str | None = None
        self.events: list[dict[str, Any]] = []
        self.unresolved_state_memory: list[dict[str, str]] = []
        self.current_event: str | None = None
        self.parsed = {
            name: [parse_rule(item) for item in self.harness[name]]
            for name in HARNESS_LIST_FIELDS
        }
        self.unparsed = {
            name: [
                item for item, parsed in zip(self.harness[name], self.parsed[name])
                if parsed is None
            ]
            for name in HARNESS_LIST_FIELDS
        }

    def set_failure_context(self, failure_type: str | None) -> None:
        self.failure_type = failure_type

    def reset(self, _task: str = "") -> None:
        self.step = 0
        self.stage_index = 0
        self.memory.clear()
        self.selected_skill = None
        self.bound_skills = ()
        self.binding_frozen = False
        self.routed_skill = None
        self.bound_sequence = ()
        self.termination_reason = None
        self.current_event = None
        self.events.clear()
        self.unresolved_state_memory.clear()

    @property
    def stage(self) -> str | None:
        plan = self.harness["task_plan"]
        if not plan:
            return None
        return _stage_entry(plan[min(self.stage_index, len(plan) - 1)])[0] or None

    def initial_skill(self) -> str | None:
        """Return the episode-start registered choice before observations arrive."""

        return self._route({})

    def _advance_stage(self, observation: Mapping[str, Any]) -> None:
        plan = self.harness["task_plan"]
        # PhysicalRSI binds a selected skill for the episode; a task plan is
        # advanced by an explicit observed condition, never by an arbitrary
        # action-count boundary.
        while plan and self.stage_index < len(plan) - 1:
            _, rule = _stage_entry(plan[self.stage_index])
            if not rule:
                break
            predicate = rule.get("done") or rule.get("when")
            if not predicate:
                break
            if not _matches(
                {"when": predicate}, event=self.current_event,
                failure_type=self.failure_type, stage=self.stage, step=self.step,
                observation=observation, memory=self.memory,
            ):
                break
            self.stage_index += 1

    def observe(self, observation: Mapping[str, Any], *, event: str | None = None) -> None:
        """Consume one native observation and update the bounded episode state."""

        if not isinstance(observation, Mapping):
            raise ValueError("Harness observation must be a mapping")
        self.current_event = event
        if event:
            self.failure_type = event
        for rule in self.parsed["state_memory"]:
            if not rule or "remember" not in rule:
                continue
            if rule.get("when") or rule.get("on"):
                if not _matches(
                    rule, event=event, failure_type=self.failure_type, stage=self.stage,
                    step=self.step, observation=observation, memory=self.memory,
                ):
                    continue
            source = rule.get("source")
            value = _lookup(observation, self.memory, source, _MISSING) if source else rule.get("value")
            if value is _MISSING:
                unresolved = {"remember": rule["remember"], "source": source}
                if unresolved not in self.unresolved_state_memory:
                    self.unresolved_state_memory.append(unresolved)
                continue
            if rule.get("mode", "latest") == "first" and rule["remember"] in self.memory:
                continue
            self.memory[rule["remember"]] = _snapshot_value(value)
        self._advance_stage(observation)
        self.selected_skill = self._route(observation)
        self.events.append({
            "step": self.step,
            "event": event,
            "stage": self.stage,
            "skill": self.selected_skill,
            "memory_keys": sorted(self.memory),
        })

    def _route(self, observation: Mapping[str, Any]) -> str | None:
        # PhysicalRSI binds an operation or operation composition once at the
        # start of an episode.  A later observation can advance the active
        # operation inside that composition, but cannot silently replace the
        # registered binding.
        if not self.binding_frozen:
            self.binding_frozen = True
            for rule in self.parsed["routing_rules"]:
                if rule and _matches(
                    rule, event=self.current_event, failure_type=self.failure_type,
                    stage=self.stage, step=self.step, observation=observation, memory=self.memory,
                ):
                    selected = rule.get("use", rule.get("skill"))
                    if selected:
                        self.bound_skills = (selected,)
                        self.routed_skill = selected
                        return selected
            sequence = self.harness["skill_sequence"]
            self.bound_sequence = tuple(parse_rule(item) or {"skill": item} for item in sequence)
            self.bound_skills = tuple(rule["skill"] for rule in self.bound_sequence)
        if self.routed_skill:
            return self.routed_skill
        if not self.bound_sequence:
            return None
        # Explicit stage labels take precedence over list positions. The
        # composition is frozen, while its activation predicates are evaluated
        # on current observations (including stages not active at binding).
        positional = []
        for rule in self.bound_sequence:
            if rule.get("stage"):
                if rule["stage"] != self.stage:
                    continue
            else:
                positional.append(rule)
                continue
            if not rule.get("when") or _matches(
                rule, event=self.current_event, failure_type=self.failure_type,
                stage=self.stage, step=self.step, observation=observation, memory=self.memory,
            ):
                return rule["skill"]
        if positional:
            rule = positional[min(self.stage_index, len(positional) - 1)]
            if not rule.get("when") or _matches(
                rule, event=self.current_event, failure_type=self.failure_type,
                stage=self.stage, step=self.step, observation=observation, memory=self.memory,
            ):
                return rule["skill"]
        return None

    def should_terminate(self, observation: Mapping[str, Any]) -> bool:
        for rule in self.parsed["termination_checks"]:
            if not rule or rule.get("action", "stop").lower() != "stop":
                continue
            if _matches(
                rule, event=self.current_event, failure_type=self.failure_type,
                stage=self.stage, step=self.step, observation=observation, memory=self.memory,
            ):
                self.termination_reason = rule.get("reason") or rule.get("when") or rule.get("on")
                return True
        return False

    def instruction(self, task: str, observation: Mapping[str, Any]) -> str:
        # Current event is intentionally absent here: failure context for a
        # retry remains available through ``failure_type`` without relabelling
        # every normal observation as a new failure event.
        self.observe(observation)
        parts = [task]
        if self.stage:
            parts.append(f"Current harness stage: {self.stage}")
            stage_rule = parse_rule(self.harness["task_plan"][min(self.stage_index, len(self.harness["task_plan"]) - 1)])
            if stage_rule and (stage_rule.get("description") or stage_rule.get("instruction")):
                parts.append("Stage guidance: " + (stage_rule.get("instruction") or stage_rule.get("description")))
        if self.selected_skill:
            parts.append(f"Bound PhysicalRSI skill: {self.selected_skill}")
        if self.memory:
            parts.append("Episode state memory: " + ", ".join(
                f"{key}={value}" for key, value in self.memory.items()
            ))
        for item, rule in zip(self.harness["recovery_rules"], self.parsed["recovery_rules"]):
            if rule is not None:
                if not _matches(
                    rule, event=self.current_event, failure_type=self.failure_type,
                    stage=self.stage, step=self.step, observation=observation, memory=self.memory,
                ):
                    continue
                text = rule.get("instruction")
            else:
                # A plain recovery entry is advisory text.  It is included
                # only for a diagnosed retry, never on the clean first pass.
                text = item if self.failure_type else None
            if text:
                parts.append("Recovery rule: " + text)
        if self.harness["instruction_suffix"]:
            parts.append(self.harness["instruction_suffix"])
        return "\n".join(parts)

    def after_action(self, observation: Mapping[str, Any]) -> None:
        self.step += 1
        self.observe(observation)

    def audit(self) -> dict[str, Any]:
        return {
            "protocol": "physicalrsi_system1_episode_binding/v1",
            "harness_revision": content_revision(self.harness),
            "memory_revision": content_revision(self.memory),
            "parsed_counts": {
                name: sum(item is not None for item in self.parsed[name])
                for name in HARNESS_LIST_FIELDS
            },
            "unparsed": self.unparsed,
            "unresolved_state_memory": self.unresolved_state_memory[-16:],
            "selected_skill": self.selected_skill,
            "bound_skills": list(self.bound_skills),
            "binding_frozen": self.binding_frozen,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "steps": self.step,
            "memory": self.memory,
            "termination_reason": self.termination_reason,
            "events": self.events[-16:],
        }


def native_harness_prompt(harness: Any) -> str:
    """Render only declarative context; execution is performed by the controller."""

    value = normalize_native_harness(harness)
    sections: list[str] = []
    labels = (
        ("task_plan", "Task plan"),
        ("state_memory", "State memory rules"),
        ("skill_sequence", "Registered skill sequence"),
        ("recovery_rules", "Recovery rules"),
        ("routing_rules", "Routing rules"),
        ("termination_checks", "Termination checks"),
    )
    for key, label in labels:
        if value[key]:
            sections.append(f"{label}: " + " | ".join(value[key]))
    if value["instruction_suffix"]:
        sections.append("Supervisor instruction: " + value["instruction_suffix"])
    if value["action_steps"] != 25:
        sections.append(f"Execute at most {value['action_steps']} actions before requesting a fresh policy chunk.")
    return "\n".join(sections)


@dataclass
class RuntimeHarness:
    """Legacy mock harness retained for the explicit mock backend only."""

    max_steps: int = 128
    failure_patience: int = 2
    recovery_budget: int = 2
    termination_keys: tuple[str, ...] = ("success", "goal_reached")
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, changes: dict[str, Any]) -> None:
        for key in ("max_steps", "failure_patience", "recovery_budget"):
            if key in changes:
                value = int(changes[key])
                if value <= 0:
                    raise ValueError(f"{key} must be positive")
                setattr(self, key, value)
        self.metadata.update({
            key: value for key, value in changes.items()
            if key not in {"max_steps", "failure_patience", "recovery_budget"}
        })

    def is_terminal(self, observation: dict[str, Any]) -> bool:
        return any(observation.get(key) is True for key in self.termination_keys)
