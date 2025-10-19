#!/bin/bash -l

#$ -l gpu=1
#$ -ac allow=L
#$ -l h_rt=72:0:0
#$ -l mem=48G

set -euo pipefail
trap "echo 'Script interrupted'; exit 1" INT

source ~/miniconda/etc/profile.d/conda.sh
conda activate .venv
cd wm_inference_diffusion

export WANDB_MODE=offline
export WANDB_START_METHOD=fork
OUTPUT_DIR="/myriadfs/home/ucabahg/wm_inference_diffusion/outputs"
WANDB_KEY="8e782a594dad15c64868ccff129984a8a344af28"

# Function to handle failure
handle_failure() {
    echo "⚠️ Run failed — syncing offline runs..."
    WANDB_API_KEY=$WANDB_KEY wandb sync "$(find "$OUTPUT_DIR" -type d -name 'offline-run-*' | sort -V)" || echo "Sync failed"

    echo "Killing leftover W&B processes..."
    pkill -9 -u "$USER" wandb || true

    echo "Clearing output directory..."
    rm -rf "$OUTPUT_DIR"
}

# Wrapper to run Python safely (retry max 3 times)
run_experiment() {
    local args=("$@")
    local retries=0
    local max_retries=3

    while (( retries < max_retries )); do
        echo "🚀 Running experiment (attempt $((retries+1))/$max_retries): ${args[*]}"

        # Ensure clean environment
        pkill -9 -u "$USER" wandb || true
        rm -rf "$OUTPUT_DIR"

        # Run Python and capture exit code
        set +e
        python src/main.py "${args[@]}"
        exit_code=$?
        set -e

        if [[ $exit_code -eq 0 ]]; then
            echo "✅ Run completed successfully — syncing latest run..."
            WANDB_API_KEY=$WANDB_KEY wandb sync "$(find "$OUTPUT_DIR" -type d -name 'offline-run-*' | sort -V | tail -n 1)" || echo "⚠️ Sync failed"
            return 0
        else
            echo "❌ Run failed (exit code $exit_code). Retrying..."
            handle_failure
            ((retries++))
        fi
    done

    echo "⚠️ Experiment failed after $max_retries attempts. Skipping..."
}

# ---------------------------
# First loop
# ---------------------------
for env_type in Alien Amidar Assault Asterix BankHeist BattleZone Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown; do
  for planning_steps in 5; do
    for inner_planning_steps in 1; do
      for entropy_threshold in 1.5; do
        for planning_mode in reward value; do
          for seed in 0 1 2; do
              run_experiment \
                  initialization.env_type=$env_type \
                  evaluation.planning_steps=$planning_steps \
                  evaluation.inner_planning_steps=$inner_planning_steps \
                  evaluation.entropy_threshold=$entropy_threshold \
                  evaluation.planning_mode=$planning_mode \
                  evaluation.planning_depth=2 \
                  common.seed=$seed \
                  wandb.mode=offline
          done
        done
      done
    done
  done
done