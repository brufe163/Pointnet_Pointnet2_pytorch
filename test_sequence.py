import argparse
import os
from data_utils.AMTCDataLoader import AMTCDataset, AMTCRealDataset
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
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
from scipy.spatial.transform import Rotation as R
import tkinter as tk

# Configurar la ventana de Tkinter para métricas
def setup_tk_window():
    root = tk.Tk()
    root.title("Real-Time Metrics")
    label = tk.Label(root, text="", font=("Helvetica", 16))
    label.pack(pady=10, padx=10)
    return root, label

# Actualizar los valores de métricas en la ventana de Tkinter
# def update_tk_window(label, accuracy, iou):
#     label_text = f"Accuracy: {accuracy:.4f}\nIoU: {iou:.4f}"
#     label.config(text=label_text)
#     label.update()

def update_tk_window(label,accuracy, acc_per_class, iou):
    label_text = f"Accuracy: {accuracy:.4f}\n" \
            f"Accuracy dust: {acc_per_class.get(0, 0.0):.4f}\n" \
            f"Accuracy non-dust: {acc_per_class.get(1, 0.0):.4f}\n" \
            f"IoU: {iou:.4f}"
    label.config(text=label_text)
    label.update()

# Código creado para visualizar inferencia y realizar pruebas en PointNet++ con los datos simulados.

classes = ['dust', 'non-dust']

def class_acc(segment_classes, predicted_classes):  # Calcular accuracy total y por clase.
    assert len(segment_classes) == len(predicted_classes), "Las longitudes de las clases segmentadas y predichas deben ser iguales."
    unique_classes = np.unique(segment_classes)
    accuracy_per_class = {}

    #print(f'Clases únicas en segment_classes: {unique_classes}')  # Verifica las clases únicas

    for cls in unique_classes:
        class_indices = (segment_classes == cls)
        correct_predictions = np.sum(predicted_classes[class_indices] == cls)
        total_points = np.sum(class_indices)
        if total_points > 0:
            accuracy_per_class[cls] = correct_predictions / total_points
        else:
            accuracy_per_class[cls] = 0.0
    total_accuracy = np.sum(segment_classes == predicted_classes) / len(segment_classes)
    #print(f'Accuracy total: {total_accuracy:.4f}')

    # for cls, acc in accuracy_per_class.items():
    #     print(f'Accuracy for class {cls}: {acc:.4f}')

    return total_accuracy, accuracy_per_class

def ptnt2_loader(checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    NUM_CLASSES = checkpoint['model_state_dict']['conv2.weight'].shape[0]
    NUM_FEAT = checkpoint['model_state_dict']['sa1.mlp_convs.0.weight'].shape[1] - 3

    model = get_model(NUM_CLASSES, NUM_FEAT).cuda()
    load_state_info = model.load_state_dict(checkpoint['model_state_dict'])
    print(load_state_info)
    return model, NUM_CLASSES, NUM_FEAT

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

    # Generar un colormap para las distancias
    colormap = plt.get_cmap("viridis")

    for i in range(len(classes)):
        class_value = classes[i]
        if class_value == 0:
            colors[i] = [1, 0, 0]  # Rojo para la clase 0
        else:
            # Asignar color basado en la distancia normalizada
            colors[i] = colormap(distances_normalized[i])[:3]  # Eliminar el canal alfa ([:3])

    # Crear la nube de puntos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Posicionar el texto delante de la cámara
    front = [ -0.99257381817347978, 0.0089755357527696415, 0.12131222211496497 ]
    lookat = [ 0.004787915371206871, 0.13689065214758228, 0.7614587023843854 ]
    up = [ 0.12148441682944051, 0.022106110001100548, 0.99234714508997812 ]
    zoom = 0.1

    
    # Limpiar la geometría anterior y agregar las nuevas
    vis.clear_geometries()
    vis.add_geometry(pcd)

    # Configurar el control de la vista
    view_ctl = vis.get_view_control()
    view_ctl.set_front(front)
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(up)
    view_ctl.set_zoom(zoom)

    # Actualizar el visualizador
    vis.poll_events()
    vis.update_renderer()

def model_eval_sequence(model, root='', feat=['coord','intensity'], data_set='test', num_classes=2, visualization_delay=0.1):
    """
    Procesa una secuencia de frames accediendo directamente al Dataset y visualiza los resultados.
    """
    DATASET = AMTCDataset(split=data_set, data_root=root, num_point=5000, voxel_size=0.05, test_area=args.test_area, feats=feat, num_classes=num_classes)

    # Obtener los nombres de las salas desde el dataset
    room_names = DATASET.room_names  # Lista de nombres de salas
    # Crear una lista de índices ordenados basados en los nombres de las salas
    sorted_indices = sorted(range(len(room_names)), key=lambda idx: room_names[idx])

    all_true_labels = []
    all_predicted_labels = []

    # Crear el visualizador
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Visualización de PointClouds")
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # Fondo negro
    render_option.point_size = 2.0  # Tamaño de los puntos
    root, label = setup_tk_window()
    model.eval()
    with torch.no_grad():
        for count, idx in enumerate(tqdm(sorted_indices, desc='Procesando frames')):
            #print(f'Viendo escena {room_names[idx]}...')

            # Obtener data y labels directamente del Dataset
            data, labels, _ = DATASET.__getitem__(idx, return_index=True)

            #print('labels: ', np.unique(labels))
            #print('data_shape: ', np.shape(data))

            # Mantener la parte del código especificada intacta
            #fix
            points = data
            points = torch.Tensor(points).unsqueeze(0)
            points = points.float().cuda()
            points = points.transpose(2, 1)
            #end fix

            output = model(points)
            predicted_classes = torch.argmax(output[0],dim=2).reshape(-1)

            predicted_classes = predicted_classes.cpu().numpy()

            #print('pred: ', np.unique(predicted_classes, return_counts=True))
            #print('real: ', np.unique(labels, return_counts=True))

            all_true_labels.append(labels)
            all_predicted_labels.append(predicted_classes)

            # Visualizar los resultados
            vis_result(data[:, :3], predicted_classes, vis)

            # Calcular métricas parciales para visualizar
            total_acc, acc_class = class_acc(labels, predicted_classes)
            
            total_accuracy = np.sum(labels == predicted_classes) / len(labels)
            iou = jaccard_score(labels, predicted_classes, average='macro')
            update_tk_window(label,total_acc, acc_class, iou)

            # Actualizar Tkinter
            root.update_idletasks()
            root.update()

            

    vis.destroy_window()
    root.destroy()

    # Calcular métricas
    all_true_labels = np.concatenate(all_true_labels)
    all_predicted_labels = np.concatenate(all_predicted_labels)

    total_accuracy, accuracy_per_class = class_acc(all_true_labels, all_predicted_labels)
    print("\n=== Clasificación Report ===")
    target_names = classes
    print(classification_report(all_true_labels, all_predicted_labels, target_names=target_names))

    print("=== Matriz de Confusión ===")
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    print(cm)

    # Cálculo de IoU (Jaccard Score)
    iou = jaccard_score(all_true_labels, all_predicted_labels, average='macro')
    print(f"Mean IoU: {iou:.4f}")

def main(args):
    model, NUM_CLASSES, NUM_FEAT = ptnt2_loader(args.checkpoint_dir)
    print(f"NUM_FEAT: {NUM_FEAT}")
    if args.real:
        model_eval_real(model, root=args.root_dir, feat=args.feat_list, data_set=args.eval_set)
    else:
        model_eval_sequence(model, root=args.root_dir, feat=args.feat_list, num_classes=NUM_CLASSES, data_set=args.eval_set, visualization_delay=args.visualization_delay)
    #vis_data(args.root_dir, args.eval_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize processed pointclouds with classes.')
    parser.add_argument('--root_dir', type=str, default='/home/nicolas/repos/dust-filtering/data/blender_areas', 
                        help='Data root path [default: None]')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/2024-09-18_00-07/checkpoints/best_model_acc.pth', 
                        help='Checkpoint file dir path')
    parser.add_argument('--feat_list', nargs='+', default=["coord", "intensity"], help='list of the desired features to consider [default: ["coord", "color"]]')
    parser.add_argument('--eval_set', type=str, choices=['test', 'train'], default='test',
                        help='Set to evaluate [default: test]')
    parser.add_argument('--visualization_delay', type=float, default=0.1, help='Retraso entre frames en segundos [default: 0.1]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-5 [default: 5]')
    parser.add_argument('--real', action='store_true', help='Activate real data testing mode.')
    args = parser.parse_args()
    main(args)
