#!/bin/bash -l

set -euo pipefail
trap "echo 'Script interrupted'; exit 1" INT

# source ~/miniconda3/etc/profile.d/conda.sh # Must already have activated env
# conda activate diffusion
cd ~/wm_inference_diffusion

export WANDB_MODE=online
# Avoid forking a process after CUDA/native libraries have initialized.
export WANDB_START_METHOD=thread
export PYTHONFAULTHANDLER=1
OUTPUT_DIR="./outputs"

# -------------------------------------------------------
# Function to safely run experiment ONCE (no retries)
# -------------------------------------------------------
run_experiment() {
  local args=("$@")
  echo "🚀 Running experiment: ${args[*]}"

  set +e
  python -X faulthandler src/main.py "${args[@]}" \
    training.agent_in_wm=true \
    training.should=false \
    initialization.load_denoiser=true \
    initialization.load_rew_end_model=true \
    initialization.load_actor_critic=true
  exit_code=$?
  set -e

  if [[ $exit_code -ne 0 ]]; then
    echo "❌ Run failed (exit code $exit_code)"
    mapfile -t offline_runs < <(find "$OUTPUT_DIR" -type d -name 'offline-run-*' 2>/dev/null | sort -V)
    if (( ${#offline_runs[@]} > 0 )); then
      echo "⚠️ Syncing ${#offline_runs[@]} offline W&B run(s)..."
      wandb sync "${offline_runs[@]}" || true
    else
      echo "ℹ️ No offline W&B runs to sync."
    fi
    echo "Killing stray wandb processes..."
    pkill -9 wandb || true
    if [[ $exit_code -eq 139 ]]; then
      echo "🛑 Native-code segmentation fault; stopping the batch to avoid repeated crashes."
      return "$exit_code"
    fi
    return 0   # <-- IMPORTANT: continue to next run
  fi

  echo "✅ Evaluation completed."
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


for env_type in CoinRun; do #StarPilot CaveFlyer Dodgeball FruitBot Chaser Miner Jumper Leaper Maze BigFish Heist Climber Plunder Ninja BossFight; do
  for planning_steps in 5; do
    for inner_planning_steps in 0; do # No inner planning
      for planning_percentage in 1; do
        for planning_mode in random; do
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

for env_type in CoinRun; do
  for planning_steps in 5 10; do
    for inner_planning_steps in 5; do
      for planning_percentage in 0.05 0.1 0.2 0.5; do
        for planning_mode in value reward; do
          for seed in 0 1 2; do

            run_experiment \
              env.env_type="$env_type" \
              evaluation.planning_steps="$planning_steps" \
              evaluation.inner_planning_steps="$inner_planning_steps" \
              evaluation.planning_percentage="$planning_percentage" \
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
