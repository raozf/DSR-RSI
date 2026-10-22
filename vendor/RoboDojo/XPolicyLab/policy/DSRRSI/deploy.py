"""Native RoboDojo calls this once per episode, with one simulator per worker."""
import os
from pathlib import Path
from dsr.paths import repository_root

from dsr.agent.robodojo import DSRRSIRuntime
from dsr.robodojo.adapter import RoboDojoAdapter


def eval_one_episode(TASK_ENV, model_client):
    if os.environ.get('DSR_AGENT_STRATEGY') == 'physicalrsi':
        from dsr.robodojo.physicalrsi_eval import eval_one_episode as physicalrsi_eval
        return physicalrsi_eval(TASK_ENV, model_client)
    root = repository_root()
    output = Path(os.environ["DSR_AGENT_OUTPUT"])
    if not hasattr(TASK_ENV, "_dsr_runtime"):
        TASK_ENV._dsr_runtime = DSRRSIRuntime(
            output, codex_home=os.environ["DSR_CODEX_HOME"],
            mode=os.environ.get("DSR_AGENT_MODE", "frozen"),
            memory_path=os.environ.get("DSR_MEMORY"),
        )
    report = TASK_ENV._dsr_runtime.run_episode(
        RoboDojoAdapter(root / "vendor/RoboDojo", env=TASK_ENV, client=model_client)
    )
    # Native aggregation is strictly first-pass. Candidate replays remain in
    # report.json and never overwrite the reported benchmark score.
    TASK_ENV._dsr_first_result = report["first_attempt"]


def eval_one_episode_batch(TASK_ENV, model_client):
    raise ValueError("Use one simulator per DSR GPU worker")
