#!/bin/bash

for planning_steps in 2 5 10 15 20; do
  for inner_planning_steps in 1 2 5; do
    for entropy_threshold in 2 1.5; do
      for seed in 0 1 2; do
        for planning_mode in value reward; do
          echo "Running with steps=$planning_steps, ent=$entropy_threshold, mode=$planning_mode"
          CUDA_VISIBLE_DEVICES=0 python src/main.py \
            evaluation.planning_steps=$planning_steps \
            evaluation.inner_planning_steps=$inner_planning_steps \
            evaluation.entropy_threshold=$entropy_threshold \
            evaluation.planning_mode=$planning_mode \
            common.seed=$seed
        done
      done
    done
  done
done