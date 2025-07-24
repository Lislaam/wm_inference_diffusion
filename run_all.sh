#!/bin/bash

for planning_steps in 1 2 5 10 15 20; do
  for entropy_threshold in 2 1.5 1; do
    for planning_mode in value td; do
      echo "Running with steps=$planning_steps, ent=$entropy_threshold, mode=$planning_mode"
      CUDA_VISIBLE_DEVICES=2 python src/main.py \
        evaluation.planning_steps=$planning_steps \
        evaluation.entropy_threshold=$entropy_threshold \
        evaluation.planning_mode=$planning_mode
    done
  done
done