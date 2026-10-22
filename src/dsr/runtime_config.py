"""Explicit process and asset configuration for independent embodied execution."""

import os
from pathlib import Path

from dsr.paths import repository_root


def independent_path(path, *, purpose):
    """Resolve a configured path without imposing a repository-name convention."""
    supplied = Path(path).expanduser().absolute()
    value = supplied.resolve()
    return value


def interpreter(role, *, required=True):
    if role not in {"policy", "sim", "code", "train"}:
        raise ValueError(f"Unknown interpreter role: {role}")
    name = f"DSR_{role.upper()}_PYTHON"
    default = repository_root()/"runtime"/f"{role}_env"/"bin"/"python"
    # A virtualenv executable can be a symlink to its base interpreter. Validate
    # the resolved target, but execute the original virtualenv path so its
    # pyvenv.cfg and installed dependencies remain active.
    value = Path(os.environ.get(name, default)).expanduser().absolute()
    independent_path(value, purpose=name)
    if required and (not value.is_file() or not os.access(value, os.X_OK)):
        raise FileNotFoundError(f"Set {name} to an independently installed executable Python; missing {value}")
    return value


def checkpoint_root():
    # Released checkpoints are read-only external assets, not implementation code.
    return Path(os.environ.get("DSR_CHECKPOINT_ROOT", repository_root()/"assets"/"checkpoints")).expanduser().resolve()


def code_asset_root(*, required=True):
    value = independent_path(os.environ.get("DSR_CODE_ASSET_ROOT", repository_root()/"assets"/"physicalrsi"), purpose="DSR_CODE_ASSET_ROOT")
    if required and not (value/"implementations"/"code-skills").is_dir():
        raise FileNotFoundError(f"Install the independently licensed code-skill bundle at {value}; set DSR_CODE_ASSET_ROOT")
    return value


def local_state_root():
    """Explicit writable state, separate from source and immutable model assets.

    Shared project filesystems may not implement POSIX locks. Callers choose a
    lock-capable DSR_LOCAL_STATE; there is no silent fallback or lock bypass.
    """
    return independent_path(os.environ.get("DSR_LOCAL_STATE", repository_root()/"runtime"),
                            purpose="DSR_LOCAL_STATE")


def verify_state_lock(directory):
    """Probe the actual filesystem before launching expensive workers."""
    import fcntl
    import tempfile
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryFile(dir=directory) as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(stream, fcntl.LOCK_UN)
    except OSError as exc:
        raise RuntimeError(f"DSR state requires POSIX file locks at {directory}; "
                           "set DSR_LOCAL_STATE and the experiment workspace to a lock-capable filesystem") from exc


def simulator_command(argv, *, required=True):
    """Build a direct process invocation; never source an inherited environment script."""
    argv = list(argv)
    if len(argv) < 3 or argv[1] != "-u":
        raise ValueError("Expected Python -u simulator script arguments")
    argv[0] = str(interpreter("sim", required=required))
    script = Path(argv[2])
    if not script.is_absolute():
        script = repository_root()/"vendor"/"RoboDojo"/script
    script = independent_path(script, purpose="simulator script")
    argv[2] = str(script)
    kit_args = os.environ.get("DSR_KIT_ARGS", "")
    if kit_args:
        # RoboDojo accepts the entire Kit option string as one argument.
        argv.extend(["--kit_args", kit_args])
    return argv


def process_environment(gpu=None, *, role="policy"):
    if role not in {"policy", "sim"}:
        raise ValueError(f"Unknown process environment role: {role}")
    root = repository_root()
    state = local_state_root()
    env = os.environ.copy()
    for key in ("PYTHONHOME", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    if role == "sim":
        paths = [root/"src", root/"vendor"/"RoboDojo",
                 root/"vendor"/"RoboDojo"/"XPolicyLab", root/"vendor",
                 root/"vendor"/"PhysicalRSI", root/"vendor"/"XPolicyLab",
                 root/"vendor"/"XPolicyLab"/"policy"/"OpenDM"/"opendm"]
    else:
        paths = [root/"src", root/"vendor", root/"vendor"/"XPolicyLab",
                 root/"vendor"/"PhysicalRSI", root/"vendor"/"RoboDojo",
                 root/"vendor"/"XPolicyLab"/"policy"/"OpenDM"/"opendm"]
    # SAM3's pinned source imports iopath before any policy code executes.
    # Keep this vendored dependency on the policy process path; do not mutate
    # either virtualenv or remove an existing installed package.
    if role == "policy":
        from dsr.robodojo.code_dependency_preflight import vendored_import_paths
        program_root = code_asset_root(required=False) / "implementations/code-skills"
        if program_root.is_dir():
            paths.extend(vendored_import_paths(program_root))
    env.update(DSR_ROOT=str(root), PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost", TOKENIZERS_PARALLELISM="false",
        NO_ALBUMENTATIONS_UPDATE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled", OMP_NUM_THREADS=env.get("DSR_OMP_THREADS", "4"))
    for key, suffix in (("TMPDIR", "tmp"), ("HF_HOME", "cache/huggingface"),
                        ("XDG_CACHE_HOME", "cache"), ("TORCH_EXTENSIONS_DIR", "cache/torch_extensions")):
        # Keep explicitly supplied cache paths and use DSR's configured local
        # state for defaults. In particular, never move TMPDIR back onto HPVS.
        value = independent_path(env.get(key, state/suffix), purpose=key)
        value.mkdir(parents=True, exist_ok=True)
        env[key] = str(value)
    env["DSR_LOCAL_STATE"] = str(state)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def prepare_simulator_assets():
    """Bind the vendor's fixed Assets convention only inside the DSR checkout."""
    root = repository_root()
    target = Path(os.environ.get("DSR_ROBODOJO_ASSETS", root/"assets"/"RoboDojo"/"Assets")).expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Set DSR_ROBODOJO_ASSETS to the RoboDojo evaluation assets; missing {target}")
    link = root/"vendor"/"RoboDojo"/"Assets"
    if link.exists() or link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"Existing {link} does not match configured assets {target}; preserve it and configure the matching root")
    else:
        link.symlink_to(target, target_is_directory=True)
    return target


def preflight(*, agent=False):
    result = {role: str(interpreter(role)) for role in ("policy", "sim")}
    verify_state_lock(local_state_root())
    result["local_state"] = str(local_state_root())
    result["assets"] = str(prepare_simulator_assets())
    if agent:
        result["code"] = str(interpreter("code"))
        result["code_assets"] = str(code_asset_root())
        home = os.environ.get("DSR_CODEX_HOME")
        if not home or not (Path(home)/"config.toml").is_file():
            raise FileNotFoundError("Set DSR_CODEX_HOME to an authorized external Codex configuration directory")
    return result
