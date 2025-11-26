import argparse
import os
from data_utils.AMTCDataLoader import AMTCDataset
import random
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from models.pointnet2_sem_seg_amtc import get_model
from utils.test_sequence_utils import *
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
from scipy.spatial.transform import Rotation as R
import tkinter as tk
from datetime import datetime

# Configurar la ventana de Tkinter para métricas
def setup_tk_window():
    root = tk.Tk()
    root.title("Real-Time Metrics")
    label = tk.Label(root, text="", font=("Helvetica", 16))
    label.pack(pady=10, padx=10)
    return root, label

# Actualizar los valores de métricas en la ventana de Tkinter
def update_tk_window(label, accuracy, acc_per_class, iou, idx):
    if accuracy is not None and iou is not None:
        label_text = f"Frame {idx}\n"\
                f"Avg Accuracy: {accuracy:.4f}\n" \
                f"Accuracy dust: {acc_per_class.get(0, 0.0):.4f}\n" \
                f"Accuracy non-dust: {acc_per_class.get(1, 0.0):.4f}\n" \
                f"IoU: {iou:.4f}"
    else:
        label_text = f"Frame {idx}\n"\
                f"Data without labels.\n" \
                f"Metrics not available."
    label.config(text=label_text)
    label.update()

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score , ConfusionMatrixDisplay

def save_info(args, timestamp_dir):
    # Crear la ruta para el archivo de información
    info_file = timestamp_dir / "info.txt"

    # Abrir el archivo en modo escritura y almacenar los argumentos
    with open(info_file, 'w') as f:
        f.write(f"root_dir: {args.root_dir}\n")
        f.write(f"checkpoint_dir: {args.checkpoint_dir}\n")
        f.write(f"feat_list: {args.feat_list}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"visualization_delay: {args.visualization_delay}\n")
        f.write(f"test_areas: {args.test_areas}\n")
        f.write(f"real_mode: {args.real}\n")


def save_results(cm, cm_n, acc_list, iou_list, report_text, checkpoint_dir):
    # Crear carpeta "results" en el mismo directorio padre del checkpoint
    parent_dir = Path(checkpoint_dir).parent.parent
    results_dir = parent_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Obtener la fecha de ejecución
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    timestamp_dir = results_dir / current_time
    timestamp_dir.mkdir(exist_ok=True)

    # Guardar la matriz de confusión normalizada
    cm_file = timestamp_dir /f"confusion_matrix.png"
    plt.figure(figsize=(8, 6))

    # Guardar reporte en un archivo de texto
    report_file = os.path.join(timestamp_dir, f'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"Reporte de clasificación guardado en {report_file}")

    
    # Mostrar la matriz de confusión con etiquetas y valores normalizados
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dust','non-dust'])
    disp.plot(cmap=plt.cm.Blues, values_format ='7d')
    plt.title('Confusion Matrix')
    plt.savefig(cm_file)
    plt.close()


    cm_n_file = timestamp_dir / f"confusion_matrix_normalized.png"
    plt.figure(figsize=(8, 6))
    # Mostrar la matriz de confusión con etiquetas y valores normalizados
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_n, display_labels=['dust','non-dust'])
    disp.plot(cmap=plt.cm.Blues,values_format='.3f')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(cm_n_file)
    plt.close()

    # Guardar el gráfico de evolución temporal de métricas
    metrics_file = timestamp_dir / f"metrics_evolution.png"
    plt.figure()
    plt.plot(acc_list, label='Accuracy')
    plt.plot(iou_list, label='IoU')
    plt.xlabel('Frames')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Metrics Evolution Over Frames')
    plt.savefig(metrics_file)
    plt.close()

    save_info(args, timestamp_dir)
# Código creado para visualizar inferencia y realizar pruebas en PointNet++ con los datos simulados.

classes = ['dust', 'non-dust']

def class_acc(segment_classes, predicted_classes):  # Calcular accuracy total, por clase y promedio.
    assert len(segment_classes) == len(predicted_classes), "Las longitudes de las clases segmentadas y predichas deben ser iguales."
    unique_classes = np.unique(segment_classes)
    accuracy_per_class = {}

    for cls in unique_classes:
        class_indices = (segment_classes == cls)
        correct_predictions = np.sum(predicted_classes[class_indices] == cls)
        total_points = np.sum(class_indices)
        if total_points > 0:
            accuracy_per_class[cls] = correct_predictions / total_points
        else:
            accuracy_per_class[cls] = 0.0

    # Calcular accuracy total
    total_accuracy = np.sum(segment_classes == predicted_classes) / len(segment_classes)

    # Calcular accuracy promedio
    average_accuracy = np.mean(list(accuracy_per_class.values()))

    return total_accuracy, accuracy_per_class, average_accuracy


# def ptnt2_loader(checkpoint_dir):
#     checkpoint = torch.load(checkpoint_dir)
#     NUM_CLASSES = checkpoint['model_state_dict']['conv2.weight'].shape[0]
#     NUM_FEAT = checkpoint['model_state_dict']['sa1.mlp_convs.0.weight'].shape[1] - 3

#     model = get_model(NUM_CLASSES, NUM_FEAT).cuda()
#     load_state_info = model.load_state_dict(checkpoint['model_state_dict'])
#     print(load_state_info)
#     return model, NUM_CLASSES, NUM_FEAT

import json

def load_view_parameters(data, vis, pcd):
    # Acceder a los parámetros desde el JSON
    front = data["trajectory"][0]["front"]
    lookat = data["trajectory"][0]["lookat"]
    up = data["trajectory"][0]["up"]
    zoom = data["trajectory"][0]["zoom"]
    
    # Limpiar la geometría anterior y agregar la nueva
    vis.clear_geometries()
    vis.get_render_option().point_size = 5.0
    vis.get_render_option().background_color = np.array([0.95, 0.95, 0.95])  # Color #f3f3f3 en RGB
    vis.add_geometry(pcd)
    
    # Configurar el control de la vista
    view_ctl = vis.get_view_control()
    view_ctl.set_front(front)
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(up)
    view_ctl.set_zoom(zoom)

    return view_ctl


def vis_result_with_metrics(coord, pred_classes, true_classes, vis, area_name,filter=False):
    colors = np.zeros((len(pred_classes), 3))

    lidar_position = (0, 0, 0)

    # Asegurarse de que 'pred_classes' y 'true_classes' son arrays unidimensionales
    pred_classes = np.squeeze(pred_classes)
    distances = np.linalg.norm(coord - lidar_position, axis=1)
    # Normalizar las distancias entre 0 y 1
    distance_range = np.max(distances) - np.min(distances)
    distances_normalized = (distances - np.min(distances)) / distance_range if distance_range != 0 else np.zeros_like(distances)

    # Crear un colormap (viridis) para los puntos
    viridis_colormap = plt.get_cmap("viridis")

    if filter:  # Verificar si el filtro está activado
        # Filtrar los puntos que no son de clase 0
        valid_indices = pred_classes != 0
        filtered_coord = coord[valid_indices]
        filtered_distances_normalized = distances_normalized[valid_indices]

        # Asignar colores basados en la distancia para los puntos válidos
        colors = viridis_colormap(filtered_distances_normalized)[:, :3]  # Eliminar el canal alfa

        # Crear la nube de puntos
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_coord)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Calcular el centro de los puntos filtrados
        center = np.mean(filtered_coord, axis=0)
    else:
        # Asignar colores basados en la distancia para todos los puntos
        colors = viridis_colormap(distances_normalized)[:, :3]  # Eliminar el canal alfa

        # Crear la nube de puntos
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Calcular el centro de todos los puntos
        center = np.mean(coord, axis=0)

    # Limpiar la geometría anterior y agregar la nueva
    vis.clear_geometries()
    vis.get_render_option().point_size = 5.0
    vis.get_render_option().background_color = np.array([0.95, 0.95, 0.95])
    vis.add_geometry(pcd)

    # Configurar el control de la vista para centrar en los puntos
    view_ctl = vis.get_view_control()
    view_ctl.set_lookat(center)
    view_ctl.set_front([0, -1, 0])  # Mirar desde el eje Y negativo
    view_ctl.set_up([0, 0, 1])  # Eje Z como "arriba"
    view_ctl.set_zoom(0.5)  # Ajustar el zoom para una vista cercana

    # Actualizar el visualizador
    vis.poll_events()
    vis.update_renderer()


def vis_result(coord, classes, vis):
    num_classes = len(np.unique(classes))
    colors = np.zeros((len(classes), 3))

    # Asegurarse de que 'classes' es un array unidimensional
    classes = np.squeeze(classes)

    # Calcular el centro de la pointcloud
    center = np.mean(coord, axis=0)

    # Calcular la distancia de cada punto al centro de la pointcloud
    distances = np.linalg.norm(coord - center, axis=1)

    # Normalizar las distancias entre 0 y 1, manejando división por cero
    distance_range = np.max(distances) - np.min(distances)
    if distance_range == 0:
        distances_normalized = np.zeros_like(distances)
    else:
        distances_normalized = (distances - np.min(distances)) / distance_range

    # Verificar que 'classes' y 'distances_normalized' tienen la misma longitud
    assert len(classes) == len(distances_normalized), "Las longitudes de 'classes' y 'distances_normalized' no coinciden."

    # Crear un colormap en escala de grises para los puntos no polvo
    grayscale_colormap = plt.get_cmap("gray")

    for i in range(len(classes)):
        class_value = classes[i]
        if class_value == 0:
            colors[i] = [1, 0, 0]  # Rojo para la clase 0 (polvo)
        else:
            # Asignar un color en escala de grises basado en la distancia normalizada
            colors[i] = grayscale_colormap(distances_normalized[i])[:3]  # Eliminar el canal alfa ([:3])

    # Crear la nube de puntos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Limpiar la geometría anterior y agregar las nuevas
    vis.clear_geometries()
    vis.get_render_option().point_size = 5.0
    # Cambiar el color de fondo de la visualización
    vis.get_render_option().background_color = np.array([0.95, 0.95, 0.95])  # Color #f3f3f3 en RGB
    vis.add_geometry(pcd)

    # Actualizar el visualizador
    vis.poll_events()
    vis.update_renderer()

def model_eval_sequence(model, root='', feat=['coord','intensity'], num_classes=2, visualization_delay=0.1, labels_available=True, test_areas = [1,2,3]):
    """
    Procesa una secuencia de frames accediendo directamente al Dataset y visualiza los resultados.
    """
    DATASET = AMTCDataset(
        areas=test_areas,
        data_root=root,
        num_point=4096,
        voxel_size=args.voxel_size,
        feats=feat,
        num_classes=num_classes,
        labels_available=True
    )
    area_num = DATASET.areas[0]
    room_names = DATASET.room_names  # Lista de nombres de salas
    sorted_indices = sorted(range(len(room_names)), key=lambda idx: room_names[idx])

    if labels_available:
        all_true_labels = []
        acc_list = []
        iou_list = []
    all_predicted_labels = []

    # Visualizador
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualización de PointClouds")
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0]) 
    render_option.point_size = 2.0 
    root_tk, label = setup_tk_window()
    model.eval()

    with torch.no_grad():
        for count, idx in enumerate(tqdm(sorted_indices, desc='Procesando frames')):
            # Obtener data y labels directamente del Dataset
            data, labels, _ = DATASET.__getitem__(idx, return_index=True)

            # Procesamiento de los datos
            points = torch.Tensor(data).unsqueeze(0).float().cuda().transpose(2, 1)
            output = model(points)
            predicted_classes = torch.argmax(output[0], dim=2).reshape(-1).cpu().numpy()

            all_predicted_labels.append(predicted_classes)
            if labels_available:
                all_true_labels.append(labels)
                total_acc, acc_class, avg_acc = class_acc(labels, predicted_classes)
                iou = jaccard_score(labels, predicted_classes, average='macro')
                acc_list.append(total_acc)
                iou_list.append(iou)
                update_tk_window(label, avg_acc, acc_class, iou, count)
            else:
                update_tk_window(label, None, None, None, count)

            # Actualizar Tkinter
            root_tk.update_idletasks()
            root_tk.update()

            # Visualizar los resultados con enfoque en puntos de polvo
            vis_result_with_metrics(data[:, :3], predicted_classes, labels, vis, area_num)

        vis.destroy_window()
        root_tk.destroy()

    if labels_available:
        all_true_labels = np.concatenate(all_true_labels)
        all_predicted_labels = np.concatenate(all_predicted_labels)

        total_accuracy, accuracy_per_class, avg_class = class_acc(all_true_labels, all_predicted_labels)
        print("\n=== Clasificación Report ===")
        target_names = classes
        report_text = classification_report(all_true_labels, all_predicted_labels, target_names=target_names)
        print(report_text)

        print("=== Matriz de Confusión ===")
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        cm_n = confusion_matrix(all_true_labels, all_predicted_labels, normalize='true')
        print(cm)

        iou = jaccard_score(all_true_labels, all_predicted_labels, average='macro')
        print(f"Mean IoU: {iou:.4f}")

        report_header = "Reporte de clasificación basado en puntos totales (cada punto es una muestra)\n\n"
        report_text = report_header + report_text

        save_results(cm, cm_n, acc_list, iou_list, report_text,  args.checkpoint_dir)

    else:
        print("No labels available, cannot compute accuracy or save results.")


def model_eval_single_frame(model, root='', feat=['coord', 'intensity'], num_classes=2, labels_available=True, test_areas=[1, 2, 3], filter=False, threshold=None):
    """
    Procesa un único frame accediendo directamente al Dataset y visualiza los resultados.
    """
    DATASET = AMTCDataset(
        areas=test_areas,
        data_root=root,
        num_point=4096,
        voxel_size=args.voxel_size,
        feats=feat,
        num_classes=num_classes,
        labels_available=True
    )

    room_names = DATASET.room_names  # Lista de nombres de salas

    area_num = DATASET.areas[0]

    # Visualizador
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualización de PointClouds")
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.95, 0.95, 0.95])  # Color de fondo claro
    render_option.point_size = 5.0  # Tamaño de los puntos

    model.eval()

    with torch.no_grad():
        # Obtener data y labels directamente del Dataset
        data, labels, r_idx = DATASET.__getitem__(0, return_index=True)  # Usar el primer frame por defecto
        print(f"Área: {area_num}")

        # Procesamiento de los datos
        points = torch.Tensor(data).unsqueeze(0).float().cuda().transpose(2, 1)
        output = model(points)
        probabilities = torch.softmax(output[0], dim=2).cpu().numpy()
        
        if threshold is not None:
            # Aplicar el umbral para la clase 0
            predicted_classes = np.where(probabilities[:, :, 0].reshape(-1) > threshold, 0, np.argmax(probabilities, axis=2).reshape(-1))
        else:
            predicted_classes = np.argmax(probabilities, axis=2).reshape(-1)

        if labels_available:
            total_acc, acc_class, avg_acc = class_acc(labels, predicted_classes)
            iou = jaccard_score(labels, predicted_classes, average='macro')
            print(f"Accuracy promedio: {avg_acc:.4f}")
            print(f"IoU: {iou:.4f}")
        else:
            print("No hay etiquetas disponibles para este frame.")

        # Visualizar los resultados con enfoque en puntos de polvo
        vis_result_with_metrics(data[:, :3], predicted_classes, labels if labels_available else None, vis, area_num, filter)

        # Mantener la ventana abierta para interacción con el mouse
        print("Use el mouse para interactuar con la visualización. Cierre la ventana para finalizar.")
        vis.run()

    vis.destroy_window()


def main(args):
    model, NUM_CLASSES, NUM_FEAT = ptnt2_loader(args.checkpoint_dir, args.model)
    print(f"NUM_FEAT: {NUM_FEAT}")
    labels_available = not args.unlabeled

    if args.real:
        print("El modo real no es compatible con la evaluación de un único frame.")
    else:
        model_eval_single_frame(
            model,
            root=args.root_dir,
            feat=args.feat_list,
            num_classes=NUM_CLASSES,
            labels_available=labels_available,
            test_areas=[args.test_areas],
            filter=args.filter
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a single processed pointcloud frame with classes.')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_amtc_red', help='Nombre del modelo [default: pointnet2_sem_seg_amtc]')
    parser.add_argument('--root_dir', type=str, default='/home/nicolas/repos/dust-filtering/data/blender_areas', 
                        help='Data root path [default: None]')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/2024-09-18_00-07/checkpoints/best_model_acc.pth', 
                        help='Checkpoint file dir path')
    parser.add_argument('--feat_list', nargs='+', default=["coord", "intensity"], help='list of the desired features to consider [default: ["coord", "color"]]')
    parser.add_argument('--test_areas', type=int, required=True,
                        help="Define las áreas para test en formato entero.")
    parser.add_argument('--real', action='store_true', help='Activate real data testing mode.')
    parser.add_argument('--unlabeled', action='store_true', help='Indicate if data is unlabeled.')
    parser.add_argument('--filter', action='store_true', help='Filter points based on class 0.')
    parser.add_argument('--threshold', type=float, default=None, help='Threshold for class 0 probability.')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Tamaño del voxel [default: 0.1]')
    args = parser.parse_args()
    main(args)
