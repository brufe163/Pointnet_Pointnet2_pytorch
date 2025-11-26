import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve,
                             f1_score, classification_report)
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.pointnet2_sem_seg_amtc import get_model
from data_utils.AMTCDataLoader import AMTCDataset
from utils.test_sequence_utils import evaluate_frame, ptnt2_loader
import os
import argparse
import json
import datetime

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def analyze_confidence_threshold(model, 
                                 root='',
                                 feat=['coord', 'intensity'], 
                                 num_classes=2,
                                 val_areas=[1, 2, 3],
                                 test_areas=[4, 5],  # Por ejemplo, si test son áreas distintas
                                 corrected=False, 
                                 thresholds=np.linspace(0, 1, 50),
                                 output_dir='results', 
                                 hyperset=False):
    """
    1) Analiza el modelo en el conjunto de validación para diferentes umbrales 
       de confianza, considerando la clase dust=0 como "positiva".
    2) Selecciona el umbral óptimo (máximo F1 para dust=0).
    3) Aplica ese umbral en el conjunto de prueba, genera la matriz de confusión 
       (normalizada y no normalizada) y guarda métricas finales.
    4) Crea un archivo info.txt con la información relevante.
    """

    # -------------------------------------------------------------------------
    #  Preparar carpetas de salida
    # -------------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join(output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    
    # Guardaremos toda la info en un archivo .txt adicional
    info_path = os.path.join(output_path, "info.txt")

    # -------------------------------------------------------------------------
    #  Imprimir información inicial
    # -------------------------------------------------------------------------
    print("Iniciando análisis de umbral de confianza...")
    print(f"Root: {root}")
    print(f"Features: {feat}")
    print(f"Num Classes: {num_classes}")
    print(f"Val Areas: {val_areas}")
    print(f"Test Areas: {test_areas}")
    print(f"Corrected: {corrected}")
    
    # -------------------------------------------------------------------------
    #  Cargar val_areas y test_areas desde hypersets.json si corresponde
    # -------------------------------------------------------------------------
    if hyperset:
        with open('data/experimentos/hypersets.json', 'r') as f:  
            sets_dict = json.load(f)
        val_areas = sets_dict["val_set"]
        test_areas = sets_dict["test_set"]
    
    # -------------------------------------------------------------------------
    #  Crear datasets y dataloaders
    # -------------------------------------------------------------------------
    val_dataset = AMTCDataset(
        areas=val_areas,
        data_root=root,
        num_point=4096,
        voxel_size=0.5,  # Ajustar si es necesario
        feats=feat,
        num_classes=num_classes,
        labels_available=True,
        hyperset=hyperset
    )
    test_dataset = AMTCDataset(
        areas=test_areas,
        data_root=root,
        num_point=4096,
        voxel_size=0.5,  # Ajustar si es necesario
        feats=feat,
        num_classes=num_classes,
        labels_available=True,
        hyperset=hyperset
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # -------------------------------------------------------------------------
    #  1) Recolectar probabilidades/etiquetas en VALIDACIÓN
    # -------------------------------------------------------------------------
    val_probabilities, val_labels = [], []
    with torch.no_grad():
        for data, labels in tqdm(val_dataloader, desc="Evaluando VAL frames"):
            labels = labels.squeeze()
            # evaluate_frame retorna dust_probabilities = P(clase=0)
            _, dust_probabilities, _, _, labels_np = evaluate_frame(model, data, labels, labels_available=True)
            val_probabilities.append(dust_probabilities)
            val_labels.append(labels_np)
    
    val_probabilities = np.concatenate(val_probabilities)
    val_labels = np.concatenate(val_labels)
    
    # Filtrar etiquetas = -1 (puntos no etiquetados)
    mask_val = (val_labels != -1)
    val_probabilities = val_probabilities[mask_val]
    val_labels = val_labels[mask_val]
    
    # -------------------------------------------------------------------------
    #  Imprimir info de validación
    # -------------------------------------------------------------------------
    n_dust_val = np.sum(val_labels == 0)
    n_nondust_val = np.sum(val_labels == 1)
    print(f"\n[VAL] #dust=0 -> {n_dust_val}, #non-dust=1 -> {n_nondust_val}")
    
    # -------------------------------------------------------------------------
    #  2) Curvas ROC / PR en validación y selección de umbral óptimo
    # -------------------------------------------------------------------------
    fpr_val, tpr_val, thresholds_roc_val = roc_curve(val_labels, val_probabilities, pos_label=0)
    roc_auc_val = auc(fpr_val, tpr_val)
    
    precision_val, recall_val, thresholds_pr_val = precision_recall_curve(val_labels, val_probabilities, pos_label=0)
    
    # Buscar umbral que maximice F1(dust=0) en validación
    f1_scores_val = []
    for t in thresholds:
        preds = np.where(val_probabilities >= t, 0, 1)
        f1_val = f1_score(val_labels, preds, pos_label=0)
        f1_scores_val.append(f1_val)
    
    best_threshold_index = np.argmax(f1_scores_val)
    best_threshold = thresholds[best_threshold_index]
    best_f1 = f1_scores_val[best_threshold_index]
    
    print(f"[VAL] Mejor threshold={best_threshold:.3f}, F1(dust=0)={best_f1:.4f}, AUC(ROC)={roc_auc_val:.4f}")
    
    # -------------------------------------------------------------------------
    #  Guardar curva ROC de validación
    # -------------------------------------------------------------------------
    plt.figure()
    plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.annotate(f"Threshold: {best_threshold:.2f}", (0.6, 0.2), 
                 fontsize=12, color='black',
                 bbox=dict(facecolor='white', alpha=0.7))
    plt.xlabel('False Positive Rate (dust=0)')
    plt.ylabel('True Positive Rate (dust=0)')
    plt.title('ROC (Validación, dust=0)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_path, 'val_roc_curve.png'))
    plt.close()
    
    # -------------------------------------------------------------------------
    #  3) Aplicar el umbral al conjunto de PRUEBA
    # -------------------------------------------------------------------------
    test_probabilities, test_labels = [], []
    with torch.no_grad():
        for data, labels in tqdm(test_dataloader, desc="Evaluando TEST frames"):
            _, dust_probabilities, _, _, labels_np = evaluate_frame(model, data, labels, labels_available=True)
            test_probabilities.append(dust_probabilities)
            test_labels.append(labels_np)
    
    test_probabilities = np.concatenate(test_probabilities)
    test_labels = np.concatenate(test_labels)
    
    mask_test = (test_labels != -1)
    test_probabilities = test_probabilities[mask_test]
    test_labels = test_labels[mask_test]
    
    n_dust_test = np.sum(test_labels == 0)
    n_nondust_test = np.sum(test_labels == 1)
    print(f"\n[TEST] #dust=0 -> {n_dust_test}, #non-dust=1 -> {n_nondust_test}")
    
    # Predicciones con el umbral óptimo hallado en validación
    test_preds = np.where(test_probabilities >= best_threshold, 0, 1)
    
    # -------------------------------------------------------------------------
    #  4) Matrices de confusión en TEST (normalizada y no normalizada)
    # -------------------------------------------------------------------------
    cm_test = confusion_matrix(test_labels, test_preds)
    cm_test_norm = confusion_matrix(test_labels, test_preds, normalize='true')
    
    # Matriz NO normalizada
    cm_figure, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Reds',
                xticklabels=['dust', 'non-dust'], yticklabels=['dust', 'non-dust'], ax=ax)
    plt.title(f'Matriz de Confusión (TEST) - Threshold={best_threshold:.2f}')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.savefig(os.path.join(output_path, 'test_conf_matrix.png'), bbox_inches='tight')
    plt.close()
    
    # Matriz normalizada
    cm_figure_norm, ax_norm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_test_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['dust', 'non-dust'], yticklabels=['dust', 'non-dust'], ax=ax_norm)
    plt.title(f'Matriz de Confusión Normalizada (TEST) - Threshold={best_threshold:.2f}')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.savefig(os.path.join(output_path, 'test_conf_matrix_normalized.png'), bbox_inches='tight')
    plt.close()
    
    print(f"\n[TEST] Matrices de confusión guardadas en: {output_path}")
    
    # -------------------------------------------------------------------------
    #  5) Métricas finales en TEST
    # -------------------------------------------------------------------------
    f1_test = f1_score(test_labels, test_preds, pos_label=0)
    report_test = classification_report(test_labels, test_preds, target_names=['dust','non-dust'])
    
    # -------------------------------------------------------------------------
    #  6) Guardar info del experimento en un archivo info.txt
    # -------------------------------------------------------------------------
    with open(info_path, 'w') as f:
        f.write("=== INFORMACIÓN DEL EXPERIMENTO ===\n")
        f.write(f"Fecha/Hora: {timestamp}\n")
        f.write(f"Root: {root}\n")
        f.write(f"Features: {feat}\n")
        f.write(f"Num Classes: {num_classes}\n")
        f.write(f"Val Areas: {val_areas}\n")
        f.write(f"Test Areas: {test_areas}\n")
        f.write(f"Corrected: {corrected}\n")
        f.write(f"\n--- Resultados en VALIDACIÓN ---\n")
        f.write(f"ROC AUC (val) = {roc_auc_val:.4f}\n")
        f.write("Thresholds vs. F1(dust=0):\n")
        for t, f1_val in zip(thresholds, f1_scores_val):
            f.write(f"  t={t:.3f}, F1={f1_val:.4f}\n")
        f.write(f"\nMejor threshold (val) = {best_threshold:.3f}, F1={best_f1:.4f}\n")
        
        f.write("\n--- Resultados en TEST ---\n")
        f.write(f"Matriz de confusión (no normalizada):\n{cm_test}\n")
        f.write(f"Matriz de confusión (normalizada, por fila):\n{cm_test_norm}\n")
        f.write(f"F1(dust=0) (test) = {f1_test:.4f}\n\n")
        f.write("=== Classification Report (test) ===\n")
        f.write(report_test)
    
    print(f"[TEST] F1(dust=0)={f1_test:.4f}")
    print(f"Archivo info.txt guardado en: {info_path}")
    
    return best_threshold


def main():
    parser = argparse.ArgumentParser(description='Analiza el umbral de confianza para el modelo PointNet++ (dust=0).')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directorio del checkpoint del modelo')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_amtc', help='Nombre del modelo')
    parser.add_argument('--data_root', type=str, required=True, help='Directorio raíz del dataset.')
    parser.add_argument('--output_dir', type=str, default='threshold_results', help='Directorio para guardar los resultados.')
    parser.add_argument('--feat', type=str, nargs='+', default=['coord', 'intensity'], help='Características a usar.')
    parser.add_argument('--num_classes', type=int, default=2, help='Número de clases (dust=0, non-dust=1).')
    parser.add_argument('--val_areas', type=str, nargs='+', default=['1', '2', '3'], help='Áreas de validación.')
    parser.add_argument('--test_areas', type=str, nargs='+', default=['4', '5'], help='Áreas de prueba.')
    parser.add_argument('--corrected', action='store_true', help='Usar etiquetas corregidas si están disponibles.')
    parser.add_argument('--hyperset', action='store_true', help="Activa el modo con todos los conjuntos de datos")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Cargando modelo...")
    model, NUM_CLASSES, NUM_FEAT = ptnt2_loader(args.checkpoint_dir, args.model)
    
    best_threshold = analyze_confidence_threshold(
        model,
        root=args.data_root,
        feat=args.feat,
        num_classes=args.num_classes,
        val_areas=args.val_areas,
        test_areas=args.test_areas,
        corrected=args.corrected,
        output_dir=args.output_dir,
        hyperset=args.hyperset
    )
    
    print(f'\nUmbral óptimo para la clasificación (dust=0), encontrado en validación: {best_threshold:.2f}')


if __name__ == "__main__":
    main()
