#!/bin/bash -l

set -euo pipefail
trap "echo 'Script interrupted'; exit 1" INT

# source ~/miniconda3/etc/profile.d/conda.sh # Must already have activated env
# conda activate diffusion
cd ~/wm_inference_diffusion

export WANDB_MODE=online
export WANDB_START_METHOD=fork
OUTPUT_DIR="./outputs"
WANDB_KEY="8e782a594dad15c64868ccff129984a8a344af28"

# -------------------------------------------------------
# Function to safely run experiment ONCE (no retries)
# -------------------------------------------------------
run_experiment() {
  local args=("$@")
  echo "🚀 Running experiment: ${args[*]}"

  set +e
  python src/main.py "${args[@]}"
  exit_code=$?
  set -e

  if [[ $exit_code -ne 0 ]]; then
    echo "❌ Run failed (exit code $exit_code)"
    echo "⚠️ Syncing offline W&B runs..."
    WANDB_API_KEY=$WANDB_KEY wandb sync "$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'offline-run-*' 2>/dev/null | sort -V)" || true
    echo "Killing stray wandb processes..."
    pkill -9 wandb || true
    return 0   # <-- IMPORTANT: continue to next run
  fi

  # Find the most recent Hydra output directory
  LAST_RUN="$(ls -td outputs/*/* 2>/dev/null | head -n 1 || true)"
  [[ -z "$LAST_RUN" ]] && { echo "⚠️ No outputs/*/* run dir found"; return 0; }

  DATASET_DIR="$LAST_RUN/dataset"
  [[ -d "$DATASET_DIR/train" ]] || { echo "⚠️ No dataset produced in $DATASET_DIR"; return 0; }

  DEST="$HOME/wm_inference_diffusion/wm_atari_2hrs/trained_policy_${env_type}"
  mkdir -p "$DEST"

  # Don’t crash if folders already exist; overwrite cleanly
  rm -rf "$DEST/train" "$DEST/test" || true
  mv "$DATASET_DIR/train" "$DEST/train" || true
  mv "$DATASET_DIR/test"  "$DEST/test"  || true

  echo "📦 Moved dataset from $DATASET_DIR → $DEST"
  return 0
}

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------

# for env_type in CoinRun StarPilot CaveFlyer Dodgeball FruitBot Chaser Miner Jumper Leaper Maze BigFish Heist Climber Plunder Ninja BossFight; do
#   python src/main.py env.env_type="$env_type" common.seed=0
# done

# for env_type in CoinRun; do
#   for planning_steps in 0; do # No planning, baselines
#     for inner_planning_steps in 0; do # No inner planning
#       for planning_percentage in 0; do
#         for planning_mode in value; do
#           for seed in 0 1 2 3 4 5 6 7 8 9; do

#             run_experiment \
#               env.env_type="$env_type" \
#               evaluation.planning_steps="$planning_steps" \
#               evaluation.inner_planning_steps="$inner_planning_steps" \
#               evaluation.planning_percentage="$planning_percentage" \
#               evaluation.planning_mode="$planning_mode" \
#               evaluation.planning_depth=1 \
#               common.seed="$seed" \
#               wandb.mode=online

#           done
#         done
#       done
#     done
#   done
# done


for env_type in CoinRun StarPilot CaveFlyer Dodgeball FruitBot Chaser Miner Jumper Leaper Maze BigFish Heist Climber Plunder Ninja BossFight; do
  for planning_steps in 5; do
    for inner_planning_steps in 0; do # No inner planning
      for planning_percentage in 1; do
        for planning_mode in value reward; do
          for seed in 0 1 2 3 4 5 6 7 8 9; do

            run_experiment \
              env.env_type="$env_type" \
              evaluation.planning_steps="$planning_steps" \
              evaluation.inner_planning_steps="$inner_planning_steps" \
              evaluation.planning_percentage="$planning_percentage" \
              evaluation.planning_mode="$planning_mode" \
              evaluation.planning_depth=1 \
              common.seed="$seed" \
              wandb.mode=online

          done
        done
      done
    done
  done
done

# for env_type in Coinrun; do
#   for planning_steps in 5 10; do
#     for inner_planning_steps in 5; do
#       for planning_percentage in 0.05 0.1 0.2 0.5; do
#         for planning_mode in value reward; do
#           for seed in 0 1 2; do

#             run_experiment \
#               env.env_type="$env_type" \
#               evaluation.planning_steps="$planning_steps" \
#               evaluation.inner_planning_steps="$inner_planning_steps" \
#               evaluation.planning_percentage="$planning_percentage" \
#               evaluation.planning_mode="$planning_mode" \
#               evaluation.planning_depth=2 \
#               common.seed="$seed" \
#               wandb.mode=online

#           done
#         done
#       done
#     done
#   done
# done
