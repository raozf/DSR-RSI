"""Real System 1 transport: all motor actions come from the OpenDM server."""

from collections import deque
import hashlib
from pathlib import Path

from dsr.harness.runtime import parse_rule


REGISTERED_DM05_SKILLS = (
    "dm05",
    "dm05-mem",
    "dm05_mem",
)


class DM05Policy:
    def __init__(self, client, evaluation_id, *, action_steps=25, skill_choice=None):
        if isinstance(action_steps, bool) or not isinstance(action_steps, int) or not 1 <= action_steps <= 25:
            raise ValueError("action_steps must be in [1, 25]")
        if not isinstance(evaluation_id, str) or not evaluation_id.strip():
            raise ValueError("evaluation_id must be a non-empty string")
        self.client = client
        self.scope = {"evaluation_id": evaluation_id}
        self.default_skill_choice = str(skill_choice) if skill_choice else None
        if skill_choice:
            self.scope["skill_choice"] = str(skill_choice)
        self.action_steps = action_steps
        self.queue = deque()
        self.calls = 0
        self.execution_context = None

    def reset(self, _task=None):
        self.queue.clear()
        self.execution_context = None
        self.client.call(func_name="reset_evaluation", obs=self.scope)

    def predict(self, observation, instruction, **kwargs):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("DM0.5 requires a non-empty task instruction")
        # Every executed control tick updates DM0.5-Mem's 25Hz history.
        obs = {**observation, **self.scope, "instruction": instruction}
        skill = kwargs.get("skill_choice", kwargs.get("skill", self.default_skill_choice))
        if skill:
            self.scope["skill_choice"] = str(skill)
            obs["skill_choice"] = str(skill)
        else:
            self.scope.pop("skill_choice", None)
            obs.pop("skill_choice", None)
        harness_state = kwargs.get("harness_state")
        if isinstance(harness_state, dict):
            # Keep metadata bounded.  The policy input remains observation
            # grounded; this is an auditable binding record, not a second
            # hidden control channel.
            obs["harness_state"] = {
                key: harness_state[key]
                for key in ("stage", "stage_index", "selected_skill", "memory")
                if key in harness_state
            }
        execution_context = (
            harness_state.get("stage_index") if isinstance(harness_state, dict) else None,
            str(skill) if skill else None,
            instruction,
        )
        if self.execution_context is not None and execution_context != self.execution_context:
            self.queue.clear()
        self.execution_context = execution_context
        self.client.call(func_name="update_obs", obs=obs)
        if not self.queue:
            actions = self.client.call(func_name="get_action", obs=self.scope)
            if not isinstance(actions, (list, tuple)) or not actions:
                raise ValueError("DM0.5 returned no action chunk")
            self.queue.extend(actions[:self.action_steps])
            self.calls += 1
        return self.queue.popleft()


class DM05SkillBinding:
    """A registered PhysicalRSI-style operation backed by the DM0.5 service.

    The current project has one low-level DM0.5 server, so aliases select a
    stable operation role and request metadata rather than a different neural
    checkpoint.  The explicit object is still important: the actor calls the
    bound operation, and unknown names are rejected before motor execution.
    """

    def __init__(self, policy, name):
        self.policy = policy
        self.name = str(name)
        # This is the execution-adapter revision, not a claim that model
        # weights were hashed or changed by selecting a role.
        self.revision = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.input_contract = "robodojo.observation/v1"
        self.output_contract = "robodojo.action/v1"

    def describe(self):
        return {
            "name": self.name,
            "revision": self.revision,
            "revision_scope": "execution_adapter_source",
            "input": self.input_contract,
            "output": self.output_contract,
            "state": "ready",
        }

    def predict(self, observation, instruction, **kwargs):
        kwargs["skill_choice"] = self.name
        return self.policy.predict(observation, instruction, **kwargs)


class DM05SkillRegistry:
    """Small executable registry for the native DM0.5 deployment."""

    def __init__(self, policy, names=REGISTERED_DM05_SKILLS):
        self.policy = policy
        self.names = frozenset(str(name) for name in names)

    def bind(self, name):
        name = str(name)
        if name not in self.names:
            raise ValueError(f"unregistered DM0.5 skill: {name}")
        return DM05SkillBinding(self.policy, name)

    def validate_harness(self, harness):
        """Reject non-executable role names before a native rollout starts."""

        for field in ("skill_sequence", "routing_rules"):
            for item in harness.get(field, ()):
                rule = parse_rule(item)
                if field == "skill_sequence":
                    target = rule.get("skill") if rule else item
                else:
                    target = (rule or {}).get("use", (rule or {}).get("skill"))
                if target and target not in self.names:
                    raise ValueError(f"unregistered DM0.5 skill in {field}: {target}")

    def search(self, _task, top_k=4):
        return sorted(self.names)[:max(0, int(top_k))]
