#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EP_RE = re.compile(r"^ep(\d{4})_(succ|fail)\.pkl$")
TEST_SCORE_RE = re.compile(r"^test/sim_max_reward_(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_eval_log(
    path: Path, test_start_seed: int, expected_episodes: int
) -> tuple[dict[int, float], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "eval log must contain a JSON object"
    scores = {}
    for key, value in payload.items():
        match = TEST_SCORE_RE.fullmatch(str(key))
        if match:
            seed = int(match.group(1))
            score = float(value)
            assert np.isfinite(score), (seed, score)
            assert score in (0.0, 1.0), (seed, score)
            assert seed not in scores, seed
            scores[seed] = score

    expected_seeds = set(range(
        test_start_seed, test_start_seed + expected_episodes
    ))
    assert set(scores) == expected_seeds, {
        "missing": sorted(expected_seeds - set(scores)),
        "unexpected": sorted(set(scores) - expected_seeds),
    }
    mean_score = float(payload["test/mean_score"])
    assert np.isfinite(mean_score)
    assert np.isclose(mean_score, np.mean(list(scores.values()))), (
        mean_score,
        np.mean(list(scores.values())),
    )
    return scores, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--test-start-seed", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    episode_dir = args.eval_dir / "episodes"
    media_dir = args.eval_dir / "media"
    metadata_path = episode_dir / "metadata.yaml"
    eval_log_path = args.eval_dir / "eval_log.json"

    assert episode_dir.is_dir(), episode_dir
    assert media_dir.is_dir(), media_dir
    assert metadata_path.exists(), metadata_path
    assert eval_log_path.is_file(), eval_log_path

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = yaml.safe_load(metadata_path.read_text())
    lengths = list(metadata["episode_lengths"])
    successes = list(metadata["episode_successes"])

    assert len(lengths) == args.expected_episodes
    assert len(successes) == args.expected_episodes
    eval_scores, eval_log = audit_eval_log(
        eval_log_path, args.test_start_seed, args.expected_episodes
    )

    pkl_paths = sorted(episode_dir.glob("ep*.pkl"))
    assert len(pkl_paths) == args.expected_episodes

    rows = []
    initial_obs = {}
    global_indices = []
    total_rows = 0

    for expected_idx, path in enumerate(pkl_paths):
        match = EP_RE.match(path.name)
        assert match, path.name

        episode_idx = int(match.group(1))
        filename_status = match.group(2)
        assert episode_idx == expected_idx

        with path.open("rb") as f:
            frame = pickle.load(f)

        assert path.stat().st_size > 0, path
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == int(lengths[episode_idx])
        assert len(frame) > 0

        required_columns = {
            "idx", "episode", "timestep", "obs",
            "action", "reward", "success"
        }
        missing = required_columns - set(frame.columns)
        assert not missing, f"{path}: missing {missing}"

        assert frame["episode"].nunique() == 1
        assert int(frame["episode"].iloc[0]) == episode_idx

        file_success = filename_status == "succ"
        metadata_success = bool(successes[episode_idx])
        dataframe_success = bool(frame["success"].iloc[0])
        seed = args.test_start_seed + episode_idx
        eval_success = bool(eval_scores[seed])

        assert file_success == metadata_success == dataframe_success == eval_success
        assert frame["success"].astype(bool).nunique() == 1

        obs_arrays = [np.asarray(x) for x in frame["obs"]]
        action_arrays = [np.asarray(x) for x in frame["action"]]

        assert all(np.all(np.isfinite(x)) for x in obs_arrays)
        assert all(np.all(np.isfinite(x)) for x in action_arrays)
        assert all(x.shape == obs_arrays[0].shape for x in obs_arrays)
        assert all(x.shape == action_arrays[0].shape for x in action_arrays)
        rewards = pd.to_numeric(frame["reward"], errors="raise").to_numpy(
            dtype=float
        )
        assert np.all(np.isfinite(rewards))
        timesteps = pd.to_numeric(
            frame["timestep"], errors="raise"
        ).to_numpy(dtype=int)
        assert timesteps[0] == 0
        assert len(timesteps) == 1 or np.all(np.diff(timesteps) > 0)

        ids = frame["idx"].astype(int).tolist()
        global_indices.extend(ids)
        total_rows += len(frame)

        initial_obs[f"episode_{episode_idx:04d}"] = obs_arrays[0]

        video_matches = list(
            media_dir.glob(f"ep{episode_idx:04d}_{filename_status}.*")
        )
        assert len(video_matches) == 1, (
            episode_idx, filename_status, video_matches
        )
        video_path = video_matches[0]
        assert video_path.stat().st_size > 0, video_path

        rows.append({
            "episode": episode_idx,
            "seed": seed,
            "success": metadata_success,
            "decision_points": len(frame),
            "first_timestep": int(frame["timestep"].iloc[0]),
            "last_timestep": int(frame["timestep"].iloc[-1]),
            "episode_file": str(path.resolve()),
            "video_file": str(video_path.resolve()),
            "episode_sha256": sha256(path),
            "video_sha256": sha256(video_path),
            "obs_shape": str(tuple(obs_arrays[0].shape)),
            "action_shape": str(tuple(action_arrays[0].shape)),
        })

    assert total_rows == int(metadata["length"])
    assert sorted(global_indices) == list(range(total_rows))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "episode_manifest.csv", index=False)

    np.savez_compressed(
        args.output_dir / "initial_observations.npz",
        **initial_obs,
    )

    summary = {
        "eval_dir": str(args.eval_dir.resolve()),
        "test_start_seed": args.test_start_seed,
        "episode_count": len(rows),
        "success_count": int(manifest["success"].sum()),
        "failure_count": int((~manifest["success"]).sum()),
        "decision_point_count": total_rows,
        "metadata_sha256": sha256(metadata_path),
        "eval_log_sha256": sha256(eval_log_path),
        "eval_mean_score": float(eval_log["test/mean_score"]),
    }
    (args.output_dir / "rollout_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("ROLLOUT AUDIT PASS")


if __name__ == "__main__":
    main()
