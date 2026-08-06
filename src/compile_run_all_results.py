"""Compile the Procgen evaluation matrix launched by run_all.sh from W&B."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import wandb


ENTITY = "labeebah-islaam"
PROJECT = "world_models"
ENV_TYPE = "CoinRun"
CHECKPOINT_BASENAME = "agent_coinrun_300k.pt"

PER_RUN_CSV = "procgen_run_all_per_run.csv"
SUMMARY_CSV = "procgen_run_all_summary.csv"
BASELINE_PER_RUN_CSV = "procgen_run_all_baseline_per_run.csv"
BASELINE_SUMMARY_CSV = "procgen_run_all_baseline_summary.csv"

PLANNED_RETURN_KEYS = (
    "actor_critic/eval/planned_return_mean",
    "actor_critic/eval/planned_cumulative_reward",
)
BASELINE_RETURN_KEYS = (
    "actor_critic/eval/return_mean",
    "actor_critic/eval/cumulative_reward",
)
EPISODE_LENGTH_KEYS = ("actor_critic/eval/episode_length",)
NUM_PLANNING_STEPS_KEYS = ("actor_critic/eval/num_planning_steps",)

NAME_RE = re.compile(
    r"^(?P<env>.+?)_"
    r"(?P<steps>\d+)_roll_"
    r"(?P<inner>\d+)_inner_"
    r"(?P<percentage>[0-9]*\.?[0-9]+)_pct_"
    r"(?P<depth>\d+)_max_"
    r"(?P<mode>[^_]+)_seed_"
    r"(?P<seed>\d+)_time_"
)


@dataclass(frozen=True)
class Setting:
    planning_steps: int
    inner_planning_steps: int
    planning_percentage: float
    planning_depth: int
    planning_mode: str
    seeds: tuple[int, ...]

    @property
    def key(self) -> tuple:
        return (
            self.planning_steps,
            self.inner_planning_steps,
            self.planning_percentage,
            self.planning_depth,
            self.planning_mode,
        )


def expected_settings() -> list[Setting]:
    settings = [
        Setting(0, 0, 0.0, 1, "value", tuple(range(10))),
        Setting(5, 0, 1.0, 1, "random", tuple(range(10))),
    ]
    for planning_steps in (5, 10):
        for planning_percentage in (0.05, 0.1, 0.2, 0.5):
            for planning_mode in ("value", "reward"):
                settings.append(
                    Setting(
                        planning_steps,
                        5,
                        planning_percentage,
                        2,
                        planning_mode,
                        (0, 1, 2),
                    )
                )
    return settings


def parse_run_name(name: str) -> Optional[dict]:
    match = NAME_RE.match(name or "")
    if match is None:
        return None
    values = match.groupdict()
    return {
        "env_type": values["env"],
        "planning_steps": int(values["steps"]),
        "inner_planning_steps": int(values["inner"]),
        "planning_percentage": float(values["percentage"]),
        "planning_depth": int(values["depth"]),
        "planning_mode": values["mode"],
        "seed": int(values["seed"]),
    }


def nested_get(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def final_metric(run, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = run.summary.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                pass
            else:
                if np.isfinite(value):
                    return value

        try:
            last_value = None
            for row in run.scan_history(keys=[key]):
                value = row.get(key)
                if value is not None:
                    last_value = value
            if last_value is not None and np.isfinite(float(last_value)):
                return float(last_value)
        except Exception:
            continue
    return None


def setting_key(info: dict) -> tuple:
    return (
        info["planning_steps"],
        info["inner_planning_steps"],
        info["planning_percentage"],
        info["planning_depth"],
        info["planning_mode"],
    )


def checkpoint_matches(run) -> bool:
    path = nested_get(run.config, "initialization", "path_to_ckpt")
    return path is not None and Path(str(path)).name == CHECKPOINT_BASENAME


def main() -> None:
    settings = expected_settings()
    expected_by_key = {setting.key: setting for setting in settings}
    selected = {}
    duplicate_count = 0
    rejected_checkpoint = 0

    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    for run in runs:
        if run.state != "finished":
            continue
        info = parse_run_name(run.name)
        if info is None or info["env_type"].lower() != ENV_TYPE.lower():
            continue
        setting = expected_by_key.get(setting_key(info))
        if setting is None or info["seed"] not in setting.seeds:
            continue
        if not checkpoint_matches(run):
            rejected_checkpoint += 1
            continue

        key = (*setting.key, info["seed"])
        previous = selected.get(key)
        if previous is not None:
            duplicate_count += 1
        if previous is None or (run.created_at or "") > (previous.created_at or ""):
            selected[key] = run

    rows = []
    missing = []
    for setting in settings:
        for seed in setting.seeds:
            run = selected.get((*setting.key, seed))
            if run is None:
                missing.append((*setting.key, seed))
                continue

            return_keys = BASELINE_RETURN_KEYS if setting.planning_steps == 0 else PLANNED_RETURN_KEYS
            score = final_metric(run, return_keys)
            episode_length = final_metric(run, EPISODE_LENGTH_KEYS)
            num_planning_steps = final_metric(run, NUM_PLANNING_STEPS_KEYS)
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "created_at": run.created_at,
                    "env_type": ENV_TYPE,
                    "planning_steps": setting.planning_steps,
                    "inner_planning_steps": setting.inner_planning_steps,
                    "planning_percentage": setting.planning_percentage,
                    "planning_depth": setting.planning_depth,
                    "planning_mode": setting.planning_mode,
                    "seed": seed,
                    "score": score,
                    "episode_length": episode_length,
                    "num_planning_steps": num_planning_steps,
                }
            )

    columns = [
        "run_id",
        "run_name",
        "created_at",
        "env_type",
        "planning_steps",
        "inner_planning_steps",
        "planning_percentage",
        "planning_depth",
        "planning_mode",
        "seed",
        "score",
        "episode_length",
        "num_planning_steps",
    ]
    all_per_run = pd.DataFrame(rows, columns=columns).sort_values(
        [
            "planning_mode",
            "planning_steps",
            "inner_planning_steps",
            "planning_percentage",
            "seed",
        ]
    )

    per_run = all_per_run[all_per_run["planning_steps"] != 0].copy()
    per_run.to_csv(PER_RUN_CSV, index=False)

    group_columns = [
        "env_type",
        "planning_steps",
        "inner_planning_steps",
        "planning_percentage",
        "planning_depth",
        "planning_mode",
    ]
    if per_run.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            per_run.groupby(group_columns, dropna=False)
            .agg(
                n_runs=("run_id", "count"),
                n_scores=("score", "count"),
                score_mean=("score", "mean"),
                score_std=("score", lambda values: values.std(ddof=0)),
                episode_length_mean=("episode_length", "mean"),
                episode_length_std=("episode_length", lambda values: values.std(ddof=0)),
                num_planning_steps_mean=("num_planning_steps", "mean"),
                num_planning_steps_std=("num_planning_steps", lambda values: values.std(ddof=0)),
                seeds=("seed", lambda values: ",".join(map(str, sorted(values)))),
            )
            .reset_index()
        )
    summary.to_csv(SUMMARY_CSV, index=False)

    baseline_per_run = all_per_run[all_per_run["planning_steps"] == 0].copy()
    baseline_per_run.to_csv(BASELINE_PER_RUN_CSV, index=False)
    if baseline_per_run.empty:
        baseline_summary = pd.DataFrame()
    else:
        baseline_summary = (
            baseline_per_run.groupby(group_columns, dropna=False)
            .agg(
                n_runs=("run_id", "count"),
                n_scores=("score", "count"),
                score_mean=("score", "mean"),
                score_std=("score", lambda values: values.std(ddof=0)),
                episode_length_mean=("episode_length", "mean"),
                episode_length_std=("episode_length", lambda values: values.std(ddof=0)),
                num_planning_steps_mean=("num_planning_steps", "mean"),
                num_planning_steps_std=("num_planning_steps", lambda values: values.std(ddof=0)),
                seeds=("seed", lambda values: ",".join(map(str, sorted(values)))),
            )
            .reset_index()
        )
    baseline_summary.to_csv(BASELINE_SUMMARY_CSV, index=False)

    expected_run_count = sum(len(setting.seeds) for setting in settings)
    expected_non_baseline_count = sum(
        len(setting.seeds) for setting in settings if setting.planning_steps != 0
    )
    expected_baseline_count = sum(
        len(setting.seeds) for setting in settings if setting.planning_steps == 0
    )
    print(f"Loaded {len(runs)} project runs")
    print(f"Matched {len(all_per_run)} / {expected_run_count} expected run_all runs")
    print(f"Saved {len(per_run)} / {expected_non_baseline_count} non-baseline rows")
    print(f"Saved {len(baseline_per_run)} / {expected_baseline_count} baseline rows")
    print(f"Ignored {rejected_checkpoint} matching old-checkpoint runs")
    print(f"Resolved {duplicate_count} duplicate setting/seed runs by keeping the newest")
    if missing:
        print("Missing setting/seed combinations:")
        for item in missing:
            print(f"  {item}")
    print(f"Saved {PER_RUN_CSV}")
    print(f"Saved {SUMMARY_CSV}")
    print(f"Saved {BASELINE_PER_RUN_CSV}")
    print(f"Saved {BASELINE_SUMMARY_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
