import argparse
import os
from data_utils.AMTCDataLoader import *
from data_utils.indoor3d_util import g_label2color
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
from models.pointnet2_sem_seg_amtc import *

# Código creado para visualizar inferencia y realizar pruebas en PointNet++ con los datos simulados.

classes = ['dust', 'non-dust']

def class_acc(segment_classes, predicted_classes):  # Calcular accuracy total y por clase.
    assert len(segment_classes) == len(predicted_classes), "Las longitudes de las clases segmentadas y predichas deben ser iguales."
    unique_classes = np.unique(segment_classes)
    accuracy_per_class = {}

    print(f'Clases únicas en segment_classes: {unique_classes}')  # Verifica las clases únicas
    

    for cls in unique_classes:
        class_indices = (segment_classes == cls)
        correct_predictions = np.sum(predicted_classes[class_indices] == cls)
        total_points = np.sum(class_indices)
        if total_points > 0:
            accuracy_per_class[cls] = correct_predictions / total_points
        else:
            accuracy_per_class[cls] = 0.0
    total_accuracy = np.sum(segment_classes == predicted_classes) / len(segment_classes)
    print(f'Accuracy total: {total_accuracy:.4f}')
    
    for cls, acc in accuracy_per_class.items():
        print(f'Accuracy for class {cls}: {acc:.4f}')

    return accuracy_per_class


def ptnt2_loader(checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    NUM_CLASSES = checkpoint['model_state_dict']['conv2.weight'].shape[0]
    NUM_FEAT = checkpoint['model_state_dict']['sa1.mlp_convs.0.weight'].shape[1] - 3

    model = get_model(NUM_CLASSES, NUM_FEAT).cuda()
    load_state_info = model.load_state_dict(checkpoint['model_state_dict'])
    print(load_state_info)
    return model, NUM_CLASSES, NUM_FEAT


import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def vis_result(coord, classes):
    num_classes = len(np.unique(classes))
    colors = np.zeros((len(classes), 3))
    
    # Calcular el centro de la pointcloud
    center = [-0.02121615, -0.17857136,  0.53300786]
    
    # Calcular la distancia de cada punto al centro de la pointcloud
    distances = np.linalg.norm(coord - center, axis=1)
    
    # Normalizar las distancias entre 0 y 1
    distances_normalized = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
    
    # Generar un colormap para las distancias
    colormap = plt.get_cmap("viridis")
    
    for i in range(len(classes)):
        if classes[i] == 0:
            colors[i] = [1, 0, 0]  # Rojo para la clase 0
        else:
            # Asignar color basado en la distancia normalizada
            colors[i] = colormap(distances_normalized[i])[:3]  # Eliminar el canal alfa ([:3])

    # Crear la nube de puntos
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Crear el visualizador
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Nube de puntos por distancia")
    vis.add_geometry(pcd)

    # Obtener el view_control para establecer los parámetros de la cámara
    view_ctl = vis.get_view_control()

    # Parámetros de la visualización
    front = [-0.7898722749711754, -0.55708060133549198, 0.25644296217199342]
    lookat = [0.18329496841419204, -0.35543351529051487, 0.44828765533995957]
    up = [0.18403812731142727, 0.183566510259043, 0.96562586129774952]
    zoom = 0.1

    # Establecer los parámetros de la cámara
    view_ctl.set_front(front)
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(up)
    view_ctl.set_zoom(zoom)
    
    vis.run()  # Iniciar el visualizador
    vis.destroy_window()


def model_eval(model, root = '', feat = ['coord','intensity'], data_set = 'test', num_classes=2):    # Para PointNet++

    DATASET = AMTCDataset(split=data_set, data_root=root, num_point=5000, voxel_size = 0.1, test_area=2, feats = feat, num_classes = num_classes)
    idx = random.randint(0, DATASET.__len__())
    # idx = 100

    data, labels, r_idx = DATASET.__getitem__(idx, return_index = True)
    print(f'Viendo escena {r_idx}...')
    print('labels: ', np.unique(labels))
    #data_t = torch.Tensor(data).reshape(1, data.shape[1], data.shape[0])
    print('data_shape: ', np.shape(data))

    #fix
    points = data
    points = torch.Tensor(points).unsqueeze(0)
    points = points.float().cuda()
    points = points.transpose(2, 1)
    #end fix


    model.eval()
    with torch.no_grad():
        output = model(points.cuda())


    predicted_classes = torch.argmax(output[0],dim=2).reshape(-1)

    print('pred: ', np.unique(predicted_classes.cpu().numpy(), return_counts=True))
    print('real: ', np.unique(labels, return_counts=True))

    class_acc(labels, predicted_classes.cpu().numpy())
    print(np.shape(data[:,:3]))
    vis_result(data[:,:3],predicted_classes.cpu().numpy())  # pred
    vis_result(data[:,:3],labels)   # real

def model_eval_real(model, root = '', feat = ['coord','intensity'], data_set = 'test'):    # Para PointNet++

    DATASET = AMTCRealDataset(split=data_set, data_root=root, num_point=None, test_area=1, feats = feat, angle=90)
    idx = random.randint(0, DATASET.__len__())
    idx = 50

    data, r_idx = DATASET.__getitem__(idx, return_index = True)
    print(f'Viendo escena {r_idx}...')
    #data_t = torch.Tensor(data).reshape(1, data.shape[1], data.shape[0])
    print('data_shape: ', np.shape(data))

    #fix
    points = data
    points = torch.Tensor(points).unsqueeze(0)
    points = points.float().cuda()
    points = points.transpose(2, 1)
    #end fix


    model.eval()
    with torch.no_grad():
        output = model(points.cuda())


    predicted_classes = torch.argmax(output[0],dim=2).reshape(-1)

    print('pred: ', np.unique(predicted_classes.cpu().numpy(), return_counts=True))
    vis_result(data[:,:3],predicted_classes.cpu().numpy())  # pred

def vis_data(root, data_set):
    DATASET = AMTCDataset(split='test', data_root=root, num_point=None, test_area=5, block_size=1, sample_rate=1, feats = args.feat_list, transform=None, num_classes = 2)
    idx = random.randint(0, DATASET.__len__())
    # idx = 0
    print(idx)
    data, labels, r_idx= DATASET.room_points[idx], DATASET.room_labels[idx]

    vis_result(data[:,:3], labels)


def main(args):
    model, NUM_CLASSES, NUM_FEAT = ptnt2_loader(args.checkpoint_dir)
    print(NUM_FEAT)
    if args.real:
        model_eval_real(model, root = args.root_dir,feat = args.feat_list, data_set = args.eval_set)
    else:
        model_eval(model, root = args.root_dir,feat = args.feat_list, num_classes = NUM_CLASSES, data_set = args.eval_set)
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
    parser.add_argument('--real', action='store_true', help='Activate real data testing mode.')
    args = parser.parse_args()
    main(args)
