import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, jaccard_score, roc_curve, auc, precision_recall_curve
from datetime import datetime
from models.pointnet2_sem_seg_amtc import get_model
from data_utils.AMTCDataLoader import AMTCDataset
from utils.test_sequence_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import ast 

# Evaluar modelo y generar gráficos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def model_eval_sequence(model, root='', feat=['coord', 'intensity'], data_set='test', num_classes=2, labels_available=True, timestamp_dir='', test_areas = [1,2,3]):
    #DATASET = AMTCDataset(split=data_set, data_root=root, num_point=4096, voxel_size=0.1, val_test_area=sets , feats=feat, num_classes=num_classes, labels_available=labels_available)
    DATASET = AMTCDataset(
        areas=test_areas,
        data_root=root,
        num_point=4096,
        voxel_size=args.voxel_size,
        feats=feat,
        num_classes=num_classes,
        labels_available=True
    )
    dataloader = DataLoader(DATASET, batch_size=1, shuffle=False, num_workers=4)

    all_predicted_labels = np.array([], dtype=np.int32)
    all_probabilities = np.array([], dtype=np.float32)
    all_true_labels = np.array([], dtype=np.int32) if labels_available else None

    acc_list, iou_list = [], []

    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Evaluando frames"):
            predicted_classes, dust_probabilities, total_acc, iou, labels_np = evaluate_frame(model, data, labels, labels_available)

            # Concatenar resultados
            all_predicted_labels = np.concatenate((all_predicted_labels, predicted_classes))
            all_probabilities = np.concatenate((all_probabilities, dust_probabilities))

            if labels_available:
                all_true_labels = np.concatenate((all_true_labels, labels_np))
                acc_list.append(total_acc)
                iou_list.append(iou)
    print('labels:')
    print(np.unique(all_predicted_labels, return_counts = True ))

    print('true_labels:')
    print(np.unique(all_true_labels, return_counts = True ))

    # Filtrar los valores de padding (-1)
    mask = all_true_labels != -1
    all_true_labels = all_true_labels[mask]
    all_predicted_labels = all_predicted_labels[mask]
    all_probabilities = all_probabilities[mask]
    

    # Métricas finales, sólo si hay datos etiquetados
    if labels_available:
        metrics = calculate_metrics(all_true_labels, all_predicted_labels, all_probabilities)

        save_results(
            metrics['confusion_matrix'],
            metrics['confusion_matrix_normalized'],
            acc_list,
            iou_list,
            metrics['classification_report'],
            metrics['fpr'],
            metrics['tpr'],
            metrics['roc_auc'],
            metrics['precision'],
            metrics['recall'],
            timestamp_dir
        )
    else:
        print("No labels available, cannot compute accuracy or save results.")


# Función principal
def main(args):

    # Convertir el argumento `--test_areas` en una lista
    try:
        test_areas = ast.literal_eval(args.test_areas)  # Convierte la cadena a una lista
        if not isinstance(test_areas, list) or not all(isinstance(i, int) for i in test_areas):
            raise ValueError("El formato debe ser una lista de enteros, por ejemplo: [1,2,3,4].")
    except Exception as e:
        raise ValueError(f"Error procesando test_areas: {e}")

    model, NUM_CLASSES, NUM_FEAT = ptnt2_loader(args.checkpoint_dir, args.model)
    labels_available = not args.unlabeled
    # Identificamos carpeta donde queremos guardar las cosas
    timestamp_dir = path_to_save(args.checkpoint_dir)
    # Guardamos los argumentos del experimento
    save_info(args, timestamp_dir)
    # Evaluamos
    model_eval_sequence(model, root=args.root_dir, feat=args.feat_list, num_classes=NUM_CLASSES, labels_available=labels_available, timestamp_dir=timestamp_dir, test_areas = test_areas)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on datasets and generate graphics.')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_amtc', help='Nombre del modelo [default: pointnet2_sem_seg_amtc]')
    parser.add_argument('--root_dir', type=str, default='/home/nicolas/repos/dust-filtering/data/blender_areas', help='Data root path')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/checkpoints/best_model_acc.pth', help='Checkpoint file dir path')
    parser.add_argument('--feat_list', nargs='+', default=["coord", "intensity"], help='List of features to consider')
    parser.add_argument('--eval_set', type=str, choices=['test', 'train'], default='test', help='Dataset to evaluate')
    parser.add_argument('--test_areas', type=str, required=True,
                    help="Define las áreas para test en formato [1,2,3,4]")
    parser.add_argument('--unlabeled', action='store_true', help='Indicate if data is unlabeled')
    parser.add_argument('--voxel_size', type=float, default=0.1, help='Tamaño del voxel [default: 0.1]')
    args = parser.parse_args()
    main(args)
