#!/bin/bash

# Ruta base de datos
data_dir="/home/nicolas/repos/custom_pointnet2_pytorch/data/experimentos/exterior1"

# Tamaño del voxel
voxel_size=0.5

# Número de épocas
epoch=100

# Dropout
dropout=0.7

# Modelo
model="pointnet2_sem_seg_amtc"

# Conjuntos de datos (entrenamiento, validación, prueba)
sets="[[1,2,3,4],[5,6,7],[8,9,10,11,12,13]]"

# Tasa de aprendizaje
learning_rate=0.01

log=experimento8

freeze_layers=1
# Lista de características y checkpoints correspondientes
feature_lists=("coord intensity" "coord diff" "coord diff_vectors" "coord interp" "coord intensity diff" "coord intensity diff_vectors" "coord intensity interp")
checkpoints=(
    "/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/experimento4/2024-12-08_20-53/checkpoints/best_model_acc.pth"
    "/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/experimento4/1/2024-12-08_02-14/checkpoints/best_model_acc.pth"
    "/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/experimento4/1/2024-12-08_02-43/checkpoints/best_model_acc.pth"
    "/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/experimento4/1/2024-12-08_03-34/checkpoints/best_model_acc.pth"
    "/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/experimento4/2024-12-08_22-00/checkpoints/best_model_acc.pth"
    "/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/experimento4/2024-12-08_22-38/checkpoints/best_model_acc.pth"
    "/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/experimento4/2024-12-09_00-24/checkpoints/best_model_acc.pth"
)

# Iteramos por características y checkpoints correspondientes
for i in "${!feature_lists[@]}"; do
  feat="${feature_lists[i]}"
  checkpoint="${checkpoints[i]}"
  python train_amtc_v3.py \
    --data_dir $data_dir \
    --voxel_size $voxel_size \
    --epoch $epoch \
    --dropout $dropout \
    --model $model \
    --sets $sets \
    --learning_rate $learning_rate \
    --feat_list $feat \
    --pretrained_model $checkpoint \
    --freeze_layers $freeze_layers \
    --log $log \
    --notes "Fine-tuning a modelo preentrenado usando datos de exterior2, con datos de exterior1, características $feat y checkpoint en $checkpoint, congelando capas en modo $freeze_layers."
done
