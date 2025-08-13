#!/bin/bash -l

#$ -l gpu=1
#$ -ac allow=L
#$ -l h_rt=48:0:0
#$ -l mem=48G

set -euo pipefail
trap "echo 'Script interrupted'; exit 1" INT

source ~/miniconda/etc/profile.d/conda.sh
conda activate .venv
cd wm_inference_diffusion

export WANDB_MODE=offline
OUTPUT_DIR="/myriadfs/home/ucabahg/wm_inference_diffusion/outputs"
WANDB_KEY="8e782a594dad15c64868ccff129984a8a344af28"

# Function to handle failure
handle_failure() {
    echo "⚠️ Run failed — syncing offline runs..."
    WANDB_API_KEY=$WANDB_KEY wandb sync "$(find "$OUTPUT_DIR" -type d -name 'offline-run-*' | sort -V)" || echo "Sync failed"

    echo "Killing leftover W&B processes..."
    ps -u "$USER" -o pid,cmd | grep "[w]andb" | awk '{print $1}' | xargs -r kill -9 || true

    echo "Clearing output directory..."
    rm -rf "$OUTPUT_DIR"

    echo "Retrying run..."
}

# Main loop
for planning_steps in 1 2 5 10 15 20; do
  for inner_planning_steps in 0 1 2 5; do
    for entropy_threshold in 1.5 1; do
      for seed in 0 1 2; do
        for planning_mode in value reward; do

          while true; do
            echo "Running with steps=$planning_steps, inner_steps=$inner_planning_steps, ent=$entropy_threshold, mode=$planning_mode, seed=$seed"

            if python src/main.py \
              evaluation.planning_steps=$planning_steps \
              evaluation.inner_planning_steps=$inner_planning_steps \
              evaluation.entropy_threshold=$entropy_threshold \
              evaluation.planning_mode=$planning_mode \
              evaluation.planning_depth=10 \
              common.seed=$seed \
              wandb.mode=offline; then

              echo "✅ Run completed, syncing latest run..."
              if ! WANDB_API_KEY=$WANDB_KEY wandb sync "$(find "$OUTPUT_DIR" -type d -name 'offline-run-*' | sort -V | tail -n 1)"; then
                handle_failure
                continue
              fi

              break  # move to next run
            else
              handle_failure
            fi
          done
        done
      done
    done
  done
done

# Final sync of all runs at the end
# WANDB_API_KEY=$WANDB_KEY wandb sync $(find "$OUTPUT_DIR" -type d -name 'offline-run-*' | sort -V)