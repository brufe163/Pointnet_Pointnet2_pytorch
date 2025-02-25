import argparse
import os
import ast
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
import tkinter as tk

from sklearn.metrics import classification_report, confusion_matrix, jaccard_score, ConfusionMatrixDisplay

# Importar funciones y módulos del proyecto
from data_utils.AMTCDataLoader import AMTCDataset
from models.pointnet2_sem_seg_amtc import get_model
from utils.test_sequence_utils import evaluate_frame  # Se asume que esta función está definida en este módulo

# Configuración global
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['dust', 'non-dust']


# --------------------- Funciones de Visualización ---------------------
def setup_tk_window():
    """Configura la ventana de Tkinter para mostrar métricas en tiempo real."""
    root = tk.Tk()
    root.title("Métricas en Tiempo Real")
    label = tk.Label(root, text="", font=("Helvetica", 16))
    label.pack(pady=10, padx=10)
    return root, label

def update_tk_window(label, accuracy, acc_per_class, iou, frame_idx):
    """Actualiza la ventana de Tkinter con las métricas del frame actual."""
    if accuracy is not None and iou is not None:
        label_text = (f"Frame {frame_idx}\n"
                      f"Avg Accuracy: {accuracy:.4f}\n"
                      f"Accuracy dust: {acc_per_class.get(0, 0.0):.4f}\n"
                      f"Accuracy non-dust: {acc_per_class.get(1, 0.0):.4f}\n"
                      f"IoU: {iou:.4f}")
    else:
        label_text = f"Frame {frame_idx}\nDatos sin etiquetas.\nMétricas no disponibles."
    label.config(text=label_text)
    label.update()


# --------------------- Funciones para Guardar Resultados ---------------------
def save_info(args, timestamp_dir):
    """Guarda en un archivo la configuración del experimento."""
    info_file = timestamp_dir / "info.txt"
    with open(info_file, 'w') as f:
        f.write(f"root_dir: {args.root_dir}\n")
        f.write(f"checkpoint_dir: {args.checkpoint_dir}\n")
        f.write(f"feat_list: {args.feat_list}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"visualization_delay: {args.visualization_delay}\n")
        f.write(f"test_areas: {args.test_areas}\n")
        f.write(f"real_mode: {args.real}\n")
        f.write(f"voxel_size: {args.voxel_size}\n")

def save_results(cm, cm_n, acc_list, iou_list, report_text, timestamp_dir):
    """Guarda gráficos y reporte en la carpeta de resultados."""
    timestamp_dir.mkdir(exist_ok=True)
    
    # Guardar reporte de clasificación
    report_file = timestamp_dir / "classification_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"Reporte de clasificación guardado en {report_file}")
    
    # Guardar matriz de confusión
    cm_file = timestamp_dir / "confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='7d')
    plt.title('Confusion Matrix')
    plt.savefig(cm_file)
    plt.close()
    
    # Guardar matriz de confusión normalizada
    cm_n_file = timestamp_dir / "confusion_matrix_normalized.png"
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_n, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='.3f')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(cm_n_file)
    plt.close()
    
    # Guardar evolución de métricas
    metrics_file = timestamp_dir / "metrics_evolution.png"
    plt.figure()
    plt.plot(acc_list, label='Accuracy')
    plt.plot(iou_list, label='IoU')
    plt.xlabel('Frames')
    plt.ylabel('Valor de Métrica')
    plt.legend()
    plt.title('Evolución de Métricas')
    plt.savefig(metrics_file)
    plt.close()

def path_to_save(checkpoint_dir):
    """Crea el directorio para guardar los resultados basado en la fecha y hora de ejecución."""
    parent_dir = Path(checkpoint_dir).parent.parent
    results_dir = parent_dir / "results"
    results_dir.mkdir(exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_dir = results_dir / current_time
    timestamp_dir.mkdir(exist_ok=True)
    return timestamp_dir


# --------------------- Funciones del Modelo ---------------------
def ptnt2_loader(checkpoint_dir, model_name):
    """Carga el modelo a partir del checkpoint y retorna el modelo, número de clases y de features."""
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    NUM_CLASSES = checkpoint['model_state_dict']['conv2.weight'].shape[0]
    NUM_FEAT = checkpoint['model_state_dict']['sa1.mlp_convs.0.weight'].shape[1] - 3
    model = get_model(NUM_CLASSES, NUM_FEAT).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modelo cargado. NUM_FEAT: {NUM_FEAT}, NUM_CLASSES: {NUM_CLASSES}")
    return model, NUM_CLASSES, NUM_FEAT


# --------------------- Funciones de Visualización de la Nube ---------------------
def load_view_parameters(root_dir, area_folder, vis, pcd):
    """
    Carga parámetros de cámara desde un archivo JSON ubicado en la carpeta de área y
    configura el visualizador Open3D.
    """
    json_file = os.path.join(root_dir, str(area_folder), 'camera.json')
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        front = data["trajectory"][0]["front"]
        lookat = data["trajectory"][0]["lookat"]
        up = data["trajectory"][0]["up"]
        zoom = data["trajectory"][0]["zoom"]
    else:
        # Parámetros por defecto si no se encuentra el archivo
        front = [-0.0286, -0.9598, -0.2793]
        lookat = [-0.5015, -0.3130, 0.4484]
        up = [-0.3230, 0.2733, -0.9061]
        zoom = 0.2

    vis.clear_geometries()
    vis.get_render_option().point_size = 5.0
    vis.get_render_option().background_color = np.array([0.95, 0.95, 0.95])
    vis.add_geometry(pcd)
    
    view_ctl = vis.get_view_control()
    view_ctl.set_front(front)
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(up)
    view_ctl.set_zoom(zoom)
    return view_ctl

def vis_result_with_metrics(coord, pred_classes, true_classes, vis, root_dir, area_folder):
    """
    Visualiza la nube de puntos coloreada de acuerdo a las predicciones y etiquetas.
    Se usa para actualizar el visualizador Open3D en tiempo real.
    """
    colors = np.zeros((len(pred_classes), 3))
    lidar_position = (0, 0, 0)
    pred_classes = np.squeeze(pred_classes)
    
    # Normalización de distancias para colormap
    distances = np.linalg.norm(coord - np.array(lidar_position), axis=1)
    distance_range = np.max(distances) - np.min(distances)
    distances_normalized = (distances - np.min(distances)) / distance_range if distance_range != 0 else np.zeros_like(distances)
    grayscale_colormap = plt.get_cmap("gray")

    if true_classes is None:
        # Sin etiquetas: usar escala de grises
        for i in range(len(pred_classes)):
            colors[i] = grayscale_colormap(distances_normalized[i])[:3]
    else:
        true_classes = np.squeeze(true_classes)
        for i in range(len(pred_classes)):
            if true_classes[i] == 0 and pred_classes[i] == 0:
                colors[i] = [0, 1, 0]  # TP: verde
            elif true_classes[i] == 0 and pred_classes[i] == 1:
                colors[i] = [1, 0, 0]  # FN: rojo
            elif true_classes[i] == 1 and pred_classes[i] == 0:
                colors[i] = [0, 0, 1]  # FP: azul
            else:
                colors[i] = grayscale_colormap(distances_normalized[i])[:3]  # TN en escala de grises

    # Crear la nube de puntos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Configurar parámetros de visualización
    load_view_parameters(root_dir, area_folder, vis, pcd)
    vis.poll_events()
    vis.update_renderer()


# --------------------- Funciones de Métricas ---------------------
def class_acc(true_classes, predicted_classes):
    """Calcula la accuracy total, por clase y promedio."""
    assert len(true_classes) == len(predicted_classes), "Longitudes diferentes entre etiquetas y predicciones."
    unique_classes = np.unique(true_classes)
    accuracy_per_class = {}
    for cls in unique_classes:
        indices = (true_classes == cls)
        correct = np.sum(predicted_classes[indices] == cls)
        total = np.sum(indices)
        accuracy_per_class[cls] = correct / total if total > 0 else 0.0
    total_accuracy = np.sum(true_classes == predicted_classes) / len(true_classes)
    avg_accuracy = np.mean(list(accuracy_per_class.values()))
    return total_accuracy, accuracy_per_class, avg_accuracy


# --------------------- Función de Evaluación y Visualización en Tiempo Real ---------------------
def model_eval_sequence(model, root, feat, num_classes, visualization_delay, labels_available, test_areas, timestamp_dir, voxel_size):
    """
    Evalúa una secuencia de frames usando DataLoader, actualiza en tiempo real la visualización
    y acumula métricas para generar reportes.
    """
    # Crear dataset y dataloader (batch_size=1 para procesamiento frame a frame)
    DATASET = AMTCDataset(
        areas=test_areas,
        data_root=root,
        num_point=4096,
        voxel_size=voxel_size,
        feats=feat,
        num_classes=num_classes,
        labels_available=labels_available
    )
    dataloader = DataLoader(DATASET, batch_size=1, shuffle=False, num_workers=4)

    all_predicted_labels = np.array([], dtype=np.int32)
    all_true_labels = np.array([], dtype=np.int32) if labels_available else None
    acc_list, iou_list = [], []

    # Inicializar visualizador Open3D y ventana Tkinter
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualización de PointClouds")
    vis.get_render_option().background_color = np.array([0, 0, 0])
    vis.get_render_option().point_size = 2.0
    root_tk, tk_label = setup_tk_window()

    frame_idx = 0
    for data_batch, labels_batch in tqdm(dataloader, desc="Evaluando frames"):
        # data_batch: [1, N, num_features]
        # Convertir a tensor y mover a device si fuera necesario (evaluate_frame se encarga de procesar)
        data_batch = data_batch.to(device)
        if labels_available:
            labels_batch = labels_batch.to(device)
        # Se asume que evaluate_frame devuelve:
        # predicted_classes, dust_probabilities, total_acc, iou, labels_np
        predicted_classes, dust_probabilities, total_acc, iou, labels_np = evaluate_frame(model, data_batch, labels_batch, labels_available)
        
        # Acumular resultados
        predicted_classes = np.squeeze(predicted_classes)
        all_predicted_labels = np.concatenate((all_predicted_labels, predicted_classes))
        if labels_available:
            labels_np = np.squeeze(labels_np)
            all_true_labels = np.concatenate((all_true_labels, labels_np))
            acc_list.append(total_acc)
            iou_list.append(iou)
            # Calcular accuracy por clase para visualización
            _, acc_class, avg_acc = class_acc(labels_np, predicted_classes)
            update_tk_window(tk_label, avg_acc, acc_class, iou, frame_idx)
        else:
            update_tk_window(tk_label, None, None, None, frame_idx)
        
        # Extraer coordenadas (se asume que las primeras 3 features son coordenadas)
        coord = data_batch[0, :, :3].cpu().numpy()
        # Actualizar visualización Open3D; se usa el primer área (asumido folder) de test_areas
        vis_result_with_metrics(coord, predicted_classes, labels_np if labels_available else None, vis, root, str(test_areas[0]))
        
        root_tk.update_idletasks()
        root_tk.update()
        time.sleep(visualization_delay)
        frame_idx += 1

    vis.destroy_window()
    root_tk.destroy()

    # Calcular métricas globales si hay etiquetas
    if labels_available:
        # Filtrar posibles valores de padding (si existen)
        mask = all_true_labels != -1
        all_true_labels = all_true_labels[mask]
        all_predicted_labels = all_predicted_labels[mask]
        overall_acc, _, _ = class_acc(all_true_labels, all_predicted_labels)
        overall_iou = jaccard_score(all_true_labels, all_predicted_labels, average='macro')
        report_text = "Reporte de clasificación (cada punto es una muestra):\n\n" + classification_report(all_true_labels, all_predicted_labels, target_names=classes)
        print("\n=== Clasificación Report ===")
        print(report_text)
        print("=== Matriz de Confusión ===")
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        cm_n = confusion_matrix(all_true_labels, all_predicted_labels, normalize='true')
        print(cm)
        print(f"Mean IoU: {overall_iou:.4f}")
        save_results(cm, cm_n, acc_list, iou_list, report_text, timestamp_dir)
    else:
        print("No hay etiquetas; no se pueden calcular métricas globales.")


# --------------------- Función Principal ---------------------
def main(args):
    # Convertir test_areas de cadena a lista (por ejemplo "[1,2,3]")
    try:
        test_areas = ast.literal_eval(args.test_areas)
        if not isinstance(test_areas, list) or not all(isinstance(i, int) for i in test_areas):
            raise ValueError("El formato debe ser una lista de enteros, por ejemplo: [1,2,3].")
    except Exception as e:
        raise ValueError(f"Error procesando test_areas: {e}")
    
    # Cargar modelo
    model, NUM_CLASSES, NUM_FEAT = ptnt2_loader(args.checkpoint_dir, args.model)
    # Directorio para guardar resultados
    timestamp_dir = path_to_save(args.checkpoint_dir)
    # Guardar información del experimento
    save_info(args, timestamp_dir)
    # Evaluar la secuencia con visualización en tiempo real
    model_eval_sequence(
        model=model,
        root=args.root_dir,
        feat=args.feat_list,
        num_classes=NUM_CLASSES,
        visualization_delay=args.visualization_delay,
        labels_available=not args.unlabeled,
        test_areas=test_areas,
        timestamp_dir=timestamp_dir,
        voxel_size=args.voxel_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualización en tiempo real de inferencia con PointNet++.')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_amtc', help='Nombre del modelo [default: pointnet2_sem_seg_amtc]')
    parser.add_argument('--root_dir', type=str, default='/home/nicolas/repos/dust-filtering/data/blender_areas', help='Ruta raíz de los datos')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/checkpoints/best_model_acc.pth', help='Ruta del checkpoint del modelo')
    parser.add_argument('--feat_list', nargs='+', default=["coord", "intensity"], help='Lista de features a considerar')
    parser.add_argument('--visualization_delay', type=float, default=0.01, help='Retraso entre frames en segundos [default: 0.01]')
    parser.add_argument('--test_areas', type=str, required=True, help='Define las áreas para test en formato, por ejemplo: [1,2,3]')
    parser.add_argument('--unlabeled', action='store_true', help='Indica si los datos no tienen etiquetas')
    parser.add_argument('--voxel_size', type=float, default=0.1, help='Tamaño del voxel [default: 0.1]')
    parser.add_argument('--real', action='store_true', help='Activa modo de test con datos reales')
    args = parser.parse_args()
    main(args)
