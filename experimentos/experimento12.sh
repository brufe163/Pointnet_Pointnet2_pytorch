#!/bin/bash

# Base directory for data
data_dir="/home/nicolas/repos/custom_pointnet2_pytorch/data/experimentos/sintetico6/"
voxel_size=0.5
epoch=100
dropout=0.7
sets="[[1],[2],[3]]"

# Feature lists
feature_lists=("coord" "coord intensity")

# Function to run experiments
run_experiment() {
  local model=$1
  local learning_rate=$2
  local log_dir=$3
  local extra_notes=$4

  for feat in "${feature_lists[@]}"; do
    # Convert feature list into array for nargs='+'
    IFS=' ' read -r -a feat_array <<< "$feat"

    echo "Running experiment with feat_list: $feat"
    python train_amtc_v3.py \
      --data_dir "$data_dir" \
      --voxel_size "$voxel_size" \
      --epoch "$epoch" \
      --dropout "$dropout" \
      --model "$model" \
      --sets "$sets" \
      --learning_rate "$learning_rate" \
      --log "$log_dir" \
      --feat_list "${feat_array[@]}" \
      --notes "Entrenamiento (v3) con datos sinteticos (12. Modelo: $model, learning_rate: $learning_rate, características: $feat. $extra_notes"
    # Check for errors
    if [ $? -ne 0 ]; then
      echo "Error detected with feat_list: $feat"
      exit 1
    fi
  done
}

# Experiments
run_experiment "pointnet2_sem_seg_amtc" 0.01 "experimento12/0.01" ""
run_experiment "pointnet2_sem_seg_amtc" 0.008 "experimento12/0.008" ""
run_experiment "pointnet2_sem_seg_amtc" 0.005 "experimento12/0.005" ""