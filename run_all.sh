#!/bin/bash -l

set -euo pipefail
trap "echo 'Script interrupted'; exit 1" INT

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffusion
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
        WANDB_API_KEY=$WANDB_KEY wandb sync "$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'offline-run-*' | sort -V)" || true

        echo "Killing stray wandb processes..."
        pkill -9 wandb || true
    fi
    DEST="wm_atari_2hrs/trained_policy_${env_type}"

    mkdir -p "$DEST"

    # Move train and test into your final location
    mv dataset/train "$DEST/train"
    mv dataset/test  "$DEST/test"

    echo "📦 Moved dataset to $DEST"
}

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
for env_type in Alien; do # Amidar Assault Asterix BankHeist BattleZone Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown; do #
  for planning_steps in 0; do
    for inner_planning_steps in 0; do
      for entropy_threshold in 2; do
        for planning_mode in reward; do
          for seed in 0; do

            run_experiment \
              env.env_type="$env_type" \
              evaluation.planning_steps="$planning_steps" \
              evaluation.inner_planning_steps="$inner_planning_steps" \
              evaluation.entropy_threshold="$entropy_threshold" \
              evaluation.planning_mode="$planning_mode" \
              evaluation.planning_depth=2 \
              common.seed="$seed" \
              wandb.mode=online

          done
        done
      done
    done
  done
done