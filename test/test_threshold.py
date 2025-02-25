import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve,
                             f1_score, classification_report,
                             precision_recall_fscore_support)
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
                                 test_areas=[1, 2, 3], 
                                 corrected=False, 
                                 thresholds=np.linspace(0, 1, 50),
                                 output_dir='results', 
                                 hyperset=False):
    """
    Evalúa el modelo para diferentes umbrales de confianza, 
    considerando la clase dust=0 como "positiva".
    Genera ROC, matrices de confusión y guarda métricas en .txt.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = os.path.join(output_dir, timestamp)
    os.makedirs(output_path, exist_ok=True)
    
    print("Iniciando análisis de umbral de confianza...")
    print(f"Root: {root}, Features: {feat}, Num Classes: {num_classes}, Test Areas: {test_areas}, Corrected: {corrected}")

    # Cargar test_areas desde hypersets.json si corresponde
    if hyperset:
        with open('data/experimentos/hypersets.json', 'r') as f:  
            sets_dict = json.load(f)
        test_areas = sets_dict["test_set"]
    
    print("test_areas:", test_areas)
    
    # Crear dataset y dataloader
    dataset = AMTCDataset(
        areas=test_areas,
        data_root=root,
        num_point=4096,
        voxel_size=0.1,  # Ajustar si es necesario
        feats=feat,
        num_classes=num_classes,
        labels_available=True,
        hyperset=hyperset
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    all_probabilities, all_true_labels = [], []
    
    # Recolección de probabilidades y etiquetas
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(dataloader, desc="Evaluando frames")):
            # evaluate_frame retorna dust_probabilities = P(clase=0)
            _, dust_probabilities, _, _, labels_np = evaluate_frame(model, data, labels, labels_available=True)
            all_probabilities.append(dust_probabilities)
            all_true_labels.append(labels_np)
            # print(f"Frame {i}: Probabilidades - {dust_probabilities[:5]}... Labels - {labels_np[:5]}...")
    
    all_probabilities = np.concatenate(all_probabilities)
    all_true_labels = np.concatenate(all_true_labels)
    
    # Filtrar etiquetas = -1 (puntos no etiquetados)
    mask = all_true_labels != -1
    all_probabilities = all_probabilities[mask]
    all_true_labels = all_true_labels[mask]
    
    print("Distribución de probabilidades (para dust=0):")
    print(np.histogram(all_probabilities, bins=10))
    
    # Verificar clases presentes
    unique_labels = np.unique(all_true_labels)
    print("Clases presentes en los datos:", unique_labels)
    if 0 not in unique_labels or 1 not in unique_labels:
        raise ValueError("Las etiquetas deben ser 0 (dust) y 1 (non-dust). Verifica la asignación de clases.")
    
    # -------------------------------------------------------------------------
    # 1) Calcular curva ROC/PR asumiendo que all_probabilities = P(clase=0).
    #    Para que la ROC interprete "1" como la clase positiva, 
    #    tendríamos que usar pos_label=0 (o invertir la probabilidad).
    # -------------------------------------------------------------------------
    fpr, tpr, thresholds_roc = roc_curve(all_true_labels, all_probabilities, pos_label=0)
    roc_auc = auc(fpr, tpr)
    
    # Ojo: 'precision_recall_curve' con pos_label=0
    precision, recall, thresholds_pr = precision_recall_curve(all_true_labels, all_probabilities, pos_label=0)
    
    print("Calculando curvas ROC y Precision-Recall (clase dust=0 como positiva)...")
    
    # -------------------------------------------------------------------------
    # 2) Seleccionar el umbral que maximice F1 para la clase dust=0
    # -------------------------------------------------------------------------
    f1_scores = []
    for t in thresholds:
        preds = (all_probabilities >= t).astype(int)  # 1 si prob >= t, 0 en caso contrario
        # Como la clase dust=0 es la "positiva", medimos F1 con pos_label=0
        f1 = f1_score(all_true_labels, preds, pos_label=0)
        f1_scores.append(f1)
    
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1 = f1_scores[best_threshold_index]
    print(f"F1-score máximo (dust=0): {best_f1:.3f} con umbral: {best_threshold:.2f}")
    
    # -------------------------------------------------------------------------
    # 3) Graficar la curva ROC con el umbral "óptimo" anotado
    # -------------------------------------------------------------------------
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.annotate(f"Threshold: {best_threshold:.2f}", (0.6, 0.2), 
                 fontsize=12, color='black',
                 bbox=dict(facecolor='white', alpha=0.7))
    plt.xlabel('False Positive Rate (para dust=0)')
    plt.ylabel('True Positive Rate (para dust=0)')
    plt.title('Receiver Operating Characteristic (dust=0)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_path, 'roc_curve.png'))
    plt.close()
    
    print("Curva ROC guardada en:", output_path)
    
    # -------------------------------------------------------------------------
    # 4) Matriz de confusión para el umbral óptimo
    # -------------------------------------------------------------------------
    predicted_labels = np.where(all_probabilities >= best_threshold, 0, 1)

    cm = confusion_matrix(all_true_labels, predicted_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dust', 'non-dust'])
    disp.plot()
    plt.title(f'Matriz de Confusión (Threshold={best_threshold:.2f})')
    plt.savefig(os.path.join(output_path, 'conf_matrix.png'))
    plt.close()
    
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['dust', 'non-dust'])
    disp_norm.plot()
    plt.title(f'Matriz de Confusión Normalizada (Threshold={best_threshold:.2f})')
    plt.savefig(os.path.join(output_path, 'conf_matrix_normalized.png'))
    plt.close()
    
    print("Matrices de confusión guardadas en:", output_path)
    
    # -------------------------------------------------------------------------
    # 5) Guardar métricas en un archivo de texto
    # -------------------------------------------------------------------------
    # Por ejemplo, guardar F1 de cada threshold, y también un classification_report
    with open(os.path.join(output_path, "metrics.txt"), 'w') as f:
        f.write("=== Métricas de F1 (clase dust=0) para cada threshold ===\n")
        for t, val in zip(thresholds, f1_scores):
            f.write(f"Threshold={t:.4f}, F1(dust=0)={val:.4f}\n")
        f.write("\n")
        f.write(f"Mejor threshold={best_threshold:.4f}, F1(dust=0)={best_f1:.4f}\n\n")
        
        # classification_report con dust=0 como "positiva" (pos_label=0)
        report = classification_report(all_true_labels, predicted_labels, target_names=['dust','non-dust'])
        f.write("=== Classification Report ===\n")
        f.write(report)
    
    print("Archivo metrics.txt guardado en:", output_path)
    
    return best_threshold

def main():
    parser = argparse.ArgumentParser(description='Analiza el umbral de confianza para el modelo PointNet++ (dust=0).')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directorio del checkpoint del modelo')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_amtc', help='Nombre del modelo')
    parser.add_argument('--data_root', type=str, required=True, help='Directorio raíz del dataset.')
    parser.add_argument('--output_dir', type=str, default='threshold_results', help='Directorio para guardar los resultados.')
    parser.add_argument('--feat', type=str, nargs='+', default=['coord', 'intensity'], help='Características a usar.')
    parser.add_argument('--num_classes', type=int, default=2, help='Número de clases (dust=0, non-dust=1).')
    parser.add_argument('--test_areas', type=str, nargs='+', default=['1', '2', '3'], help='Áreas de prueba a evaluar.')
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
        test_areas=args.test_areas,
        corrected=args.corrected,
        output_dir=args.output_dir,
        hyperset=args.hyperset
    )
    
    print(f'Umbral óptimo para la clasificación (dust=0): {best_threshold:.2f}')

if __name__ == "__main__":
    main()
