import os
import re
import argparse
import glob
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]  # maña, para ordenar los archivos de salida.

# Generador de intensidad
def assign_intensity(labels):
    """
    Asigna valores de intensidad a los puntos basándose en las distribuciones observadas en la imagen.
    - Para polvo, sigue una distribución concentrada en valores bajos (1-2).
    - Para no polvo, sigue una distribución más dispersa (1-16 con valores mayores presentes).
    """
    intensity_values = np.zeros(labels.shape[0])
    
    # Generación de intensidades para puntos de polvo
    dust_probabilities = [0.01, 0.7, 0.15, 0.03, 0.02, 0.01, 0.01, 0.02, 0.01]  # Probabilidades basadas en paper
    dust_probabilities = np.array(dust_probabilities) / np.sum(dust_probabilities)  # Asegurar que sumen 1
    dust_intensity_range = np.arange(1, 10)
    dust_indices = np.where(labels == args.dust_cls)[0]
    intensity_values[dust_indices] = np.random.choice(dust_intensity_range, size=len(dust_indices), p=dust_probabilities)
    
    # Generación de intensidades para puntos de no polvo
    no_dust_probabilities = [0.13, 0.5, 0.17, 0.034, 0.03, 0.032, 0.03, 0.025, 0.02, 0.01, 0.007, 0.01, 0.008, 0.001, 0.0001, 0.001, 0.03]  # Probabilidades basadas en paper
    no_dust_probabilities = np.array(no_dust_probabilities) / np.sum(no_dust_probabilities)  # Asegurar que sumen 1
    no_dust_intensity_range = np.array([8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136])
    no_dust_indices = np.where(labels == args.else_cls)[0]
    intensity_values[no_dust_indices] = np.random.choice(no_dust_intensity_range, size=len(no_dust_indices), p=no_dust_probabilities)
    
    return intensity_values

# Generadores de color
def assign_zero_rgb(labels): # Deja los valores RGB en cero
    return np.zeros((labels.shape[0], 3), dtype=np.uint8)

# def assign_class_based_rgb(labels):   # Mismos valores en las componentes da error (divide by zero)
#     rgb_values = np.zeros((labels.shape[0], 3))
#     rgb_values[labels == 1] = [1, 1, 1]  # Polvo
#     rgb_values[labels == 0] = [250,250,250]  # No polvo
#     return rgb_values

# def assign_class_based_rgb(labels):
#     rgb_values = np.zeros((labels.shape[0], 3))
#     base_dust = np.array([1, 1, 1])  # Polvo
#     base_no_dust = np.array([250, 250, 250])  # No polvo
#     perturbation_range_dust = np.array([10, 10, 10])  # Variación permitida para polvo
#     perturbation_range_no_dust = np.array([5, 5, 5])  # Variación permitida para no polvo
#     rgb_values[labels == 1] = base_dust + np.random.randint(-perturbation_range_dust, perturbation_range_dust + 1)
#     rgb_values[labels == 0] = base_no_dust + np.random.randint(-perturbation_range_no_dust, perturbation_range_no_dust + 1)
#     rgb_values = np.clip(rgb_values, 0, 255)        # Valores dentro de los rangos permitidos
    
#     return rgb_values

def assign_class_based_rgb(labels):
    # Inicializar los valores RGB con ceros
    rgb_values = np.zeros((labels.shape[0], 3))
    
    # Definir los valores base para las clases
    base_dust = np.array([1, 1, 1])  # Polvo
    base_no_dust = np.array([250, 250, 250])  # No polvo
    
    # Definir el rango de variación para cada componente de color
    perturbation_range_dust = np.array([10, 10, 10])  # Variación permitida para polvo
    perturbation_range_no_dust = np.array([5, 5, 5])  # Variación permitida para no polvo
    
    # Generar perturbaciones aleatorias para polvo
    perturbations_dust = np.random.randint(-perturbation_range_dust, perturbation_range_dust + 1, size=(np.sum(labels == args.dust_cls), 3))
    rgb_values[labels == args.dust_cls] = base_dust + perturbations_dust
    
    # Generar perturbaciones aleatorias para no polvo
    perturbations_no_dust = np.random.randint(-perturbation_range_no_dust, perturbation_range_no_dust + 1, size=(np.sum(labels == args.else_cls), 3))
    rgb_values[labels == args.else_cls] = base_no_dust + perturbations_no_dust
    
    # Asegurarse de que los valores estén dentro del rango [0, 255]
    rgb_values = np.clip(rgb_values, 0, 255)
    
    return rgb_values

def normalize_xyz(points):
    centroid = np.mean(points, axis=0)
    points -= centroid  # Centro en el origen
    max_distance = np.max(np.linalg.norm(points, axis=1))
    points /= max_distance  # Normaliza por la distancia máxima
    return points

def toSegPointCloud(depth_dir,mask_dir, args):
    depth = cv2.imread(depth_dir,0)
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    # Asumiendo que máscara y depth tienen las mismas dimensiones
    assert depth.shape == mask.shape

    filename = os.path.basename(depth_dir)
    match = re.search(r'img_(\d+)', filename)
    num = int(match.group(1))

    # Parámetros del lidar:
    fx = args.focal_length_x
    fy = args.focal_length_y
    camera_fov = args.camera_fov
    lidar_horizontal_points = args.lidar_horizontal_points
    lidar_vertical_channels = args.lidar_vertical_channels

    original_height, original_width = depth.shape
    x, y = np.meshgrid(np.arange(original_width), np.arange(original_height))
    x = (x - original_width / 2) / fx
    y = (y - original_height / 2) / fy
    z = np.array(depth) / 10.0
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

    temp_mask = mask.copy() # Temporal para no sobreescribir en la misma máscara
    
    mask[temp_mask != 0] = args.dust_cls # Asignar la clase para el polvo
    mask[temp_mask == 0] = args.else_cls # Asignar la clase para el resto

    classes = mask.reshape(-1)
    #points_with_classes = np.hstack((points, classes[:, np.newaxis]))


    
    if not args.no_subsampling:

        h_fov_rad = math.radians(camera_fov)
        v_fov_rad = math.radians(camera_fov)
        h_angle_step = h_fov_rad / lidar_horizontal_points
        v_angle_step = v_fov_rad / lidar_vertical_channels

        r = np.linalg.norm(points, axis=1)
        azimuth = np.arctan2(points[:, 0], points[:, 2])
        elevation = np.arcsin(points[:, 1] / r)

        h_bins = np.round(azimuth / h_angle_step).astype(int)
        v_bins = np.round(elevation / v_angle_step).astype(int)

        unique_bins = np.unique(np.stack((h_bins, v_bins), axis=1), axis=0)

        subsampled_points = []
        subsampled_classes = []
        for ub in unique_bins:
            indices = np.where((h_bins == ub[0]) & (v_bins == ub[1]))[0]
            if len(indices) > 0:
                subsampled_points.append(points[indices[0]])
                subsampled_classes.append(classes[indices[0]])

        points = np.array(subsampled_points, dtype=np.float32)
        classes = np.array(subsampled_classes, dtype=np.int16)
    

    if args.out_format == 'npy':    # Necesitamos normalizar y "crear datos" de RGB
        if args.train_test:
            if num <=120:
                out_dir = f'{args.outdir}/Area_7/amtc_{num}/'
            else:
                out_dir = f'{args.outdir}/Area_8/amtc_{num}/'
        else:
            out_dir = f'{args.outdir}/amtc_{num}/'
        if args.normalize:
            points = normalize_xyz(points)

        os.makedirs(out_dir, exist_ok=True)

        np.save(f'{out_dir}/coord.npy', points)
        np.save(f'{out_dir}/segment.npy', classes.reshape(-1, 1))

        # Generar las características solicitadas en args.feats
        if 'color0' in args.feats:
            rgb_values = assign_zero_rgb(classes)
            np.save(f'{out_dir}/color.npy', rgb_values)

        if 'color1' in args.feats:
            rgb_values = assign_class_based_rgb(classes)
            np.save(f'{out_dir}/color.npy', rgb_values)
        
        if 'intensity' in args.feats:
            intensity_values = assign_intensity(classes)
            np.save(f'{out_dir}/intensity.npy', intensity_values)
 
        
    else:
        colors = np.zeros((len(classes), 3))
        for i in range(len(classes)):
            if classes[i] == 0:
                colors[i] = np.array([0, 0, 1])
            else:
                colors[i] = np.array([1, 0, 0])  # Color rojo para clases distintas de 0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        out_dir = f'{args.outdir}/PointCloud/'
        os.makedirs(out_dir, exist_ok=True)
        o3d.io.write_point_cloud(f"{out_dir}img_{num}.{args.out_format}", pcd)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from images.')
    parser.add_argument('--max-depth', default=20, type=float,
                        help='Maximum depth value for the depth map.')
    parser.add_argument('--depth-path', type=str, default = '/home/nicolas/repos/dust-filtering/data/depth_seg_2/Depth/',
                        help='Path to the input image or directory containing images.')
    parser.add_argument('--mask-path', type=str, default = '/home/nicolas/repos/dust-filtering/data/depth_seg_2/Seg/',
                        help='Path to the mask image or directory containing masks.')           
    parser.add_argument('--outdir', type=str, default='/home/nicolas/repos/dust-filtering/data/s3dis/Area_7',
                        help='Directory to save the output point clouds.')
    parser.add_argument('--out-format', type=str, choices = ['ply', 'npy', 'pcd'],default='npy',
                        help='Output format for points and labels.')
    parser.add_argument('--focal-length-x', default=927.06, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=927.06, type=float,
                        help='Focal length along the y-axis.')
    parser.add_argument('--lidar-horizontal-points', default=256, type=int,
                        help='Number of horizontal points (resolution).')
    parser.add_argument('--lidar-vertical-channels', default=64, type=int,
                        help='Number of vertical channels.')
    parser.add_argument('--camera-fov', default=90, type=float,
                        help='Field of view of the camera in degrees (assumed square FOV).')
    parser.add_argument('--dust-cls', type = int, default = 12,
                        help='Class value asigned to dust.')
    parser.add_argument('--else-cls', type = int, default = 13 ,
                        help='Class value asigned to anything else on the dataset.')
    parser.add_argument('--no-subsampling', action='store_true', help='Deactivate pointcloud subsampling.')
    parser.add_argument('--normalize', action='store_true', help='Return normalized coords.')
    parser.add_argument('--feats', type=str, nargs='+', choices=['color0', 'color1', 'intensity'], help='List of features to generate (color, intensity).')
    parser.add_argument('--train-test', action='store_true', help='Separates dataset in training and testing subsets.')
    
    #parser.add_argument('--color_format', type=str, choices = [None, 'zeros', 'class'],default=None,
    #                    help='Add color.npy file to output data. Could it be None, zeros (if its needed to create a file but only with zeros) or class (assign different colors for each class).')

    args = parser.parse_args()

    depth_files = sorted(glob.glob(os.path.join(args.depth_path, '*.png')), key=natural_sort_key)
    mask_files = sorted(glob.glob(os.path.join(args.mask_path, '*.png')), key=natural_sort_key)

    os.makedirs(args.outdir, exist_ok=True)
    file_path = os.path.join(args.outdir, "info.txt")
    with open(file_path, "w") as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")

    for i, (depth_path, mask_path) in enumerate(zip(depth_files, mask_files), start=1):
        progress_message = f'Processing {i}/{len(depth_files)}: {depth_path}'
        print(f'\r{progress_message}', end='', flush=True)
        toSegPointCloud(depth_path, mask_path, args)
    
