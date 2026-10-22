"""Combine released RoboDojo trajectories with only validated DSR-RSI repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsr.schema import SINGLE_RETRY_PROTOCOL


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build(base, agent_root, output):
    base, agent_root, output = map(lambda p: Path(p).resolve(), (base, agent_root, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite training dataset: {output}")
    manifest = json.loads((base / "dataset.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    jsonl_root = output / "jsonl"
    jsonl_root.mkdir()
    release_index, release_episodes = {}, []
    for item in rows(base / "episodes.jsonl"):
        source = (base / "jsonl" / item["jsonl"]).resolve()
        target = jsonl_root / "release" / item["jsonl"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        release_index[str(target)] = int(item["frames"])
        release_episodes.append({**item, "jsonl": str(Path("release") / item["jsonl"])})

    verified, seen = [], set()
    for report_path in sorted(agent_root.glob("**/report.json")):
        report = json.loads(report_path.read_text())
        episode_id = report.get("episode_id")
        if not episode_id or episode_id in seen:
            continue
        first = report.get("first_attempt", {})
        if (report.get("status") != "COMPLETED" or report.get("mode") != "adapt"
                or first.get("backend") != "native_robodojo_dm05"):
            continue
        selected = []
        if first.get("success") and first.get("training_jsonl"):
            selected.append(("first", first["training_jsonl"]))
        if report.get("accepted"):
            attempts = report.get("attempts", [])
            single_retry = report.get("protocol") == SINGLE_RETRY_PROTOCOL
            expected = 2 if single_retry else 3
            if (len(attempts) != expected or any(r.get("backend") != "native_robodojo_dm05" for r in attempts)
                    or attempts[0].get("success") or not attempts[-1].get("success")
                    or (not single_retry and attempts[1].get("success"))
                    or (single_retry and not (report.get("retry_attempted")
                                              and report.get("retry_success") and report.get("final_success")))
                    or len({(r["task"], r["seed"], r["layout_id"]) for r in attempts}) != 1
                    or not report.get("verification", {}).get("accepted")):
                kind = "single retry" if single_retry else "paired repair"
                raise ValueError(f"Invalid {kind} evidence: {report_path}")
            path = attempts[-1].get("training_jsonl") or report.get("verified_training_episode")
            if path:
                selected.append(("verified_repair", path))
        for kind, source_value in selected:
            source = Path(source_value).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            episode_rows = rows(source)
            if len(episode_rows) < 2:
                raise ValueError(f"Agent trajectory has too few frames: {source}")
            for sample in episode_rows:
                for key in ("images_1", "images_2", "images_3"):
                    image = Path(sample[key]["url"])
                    if not image.is_absolute():
                        image = (source.parent / image).resolve()
                    if not image.is_file():
                        raise FileNotFoundError(image)
                    sample[key]["url"] = str(image)
            relative = Path("agent") / kind / f"{episode_id}.jsonl"
            target = jsonl_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in episode_rows))
            release_index[str(target)] = len(episode_rows)
            verified.append({
                "episode_id": episode_id, "kind": kind, "jsonl": str(relative),
                "frames": len(episode_rows), "source_report": str(report_path),
                "repair_evidence_protocol": report.get("protocol", "paired_control_candidate/v1"),
            })
        seen.add(episode_id)

    repaired = [row for row in verified if row["kind"] == "verified_repair"]
    if not repaired:
        raise RuntimeError("No verified repair trajectories were collected; refusing to train an agent adapter")
    image_dir = Path(manifest["image_dir"])
    output_media = output / "media"
    output_media.symlink_to(image_dir.resolve(), target_is_directory=True)
    manifest.update(
        source_dataset=str(base),
        agent_experience_root=str(agent_root),
        image_dir=str(output_media),
        train_episodes=int(manifest["train_episodes"]) + len(verified),
        release_episodes=int(manifest["train_episodes"]),
        verified_agent_episodes=len(verified),
        verified_repair_episodes=len(repaired),
        single_retry_recovery_episodes=sum(row["repair_evidence_protocol"] == SINGLE_RETRY_PROTOCOL
                                         for row in repaired),
        agent_frames=sum(row["frames"] for row in verified),
        frames=int(manifest["frames"]) + sum(row["frames"] for row in verified),
        resampling=False,
    )
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "agent_episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in verified)
    )
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in release_episodes)
        + "".join(json.dumps(row) + "\n" for row in verified)
    )
    (jsonl_root / "index_cache.json").write_text(json.dumps({"data": release_index}, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.base, args.agent_root, args.output), indent=2))


if __name__ == "__main__":
    main()
