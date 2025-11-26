import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
import open3d as o3d
import torch 
from data_utils.AMTCTransforms import *

def analyze_voxelization_impact(points, voxel_size):
    """
    Analiza el impacto de la voxelización en términos de reducción de puntos
    """
    original_count = points.shape[0]
    coord_range = np.ptp(points[:, :3], axis=0)  # Rango en cada dimensión (max - min)
    
    # Crear la nube de puntos para voxelizar
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Voxelizar
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    voxelized_count = len(pcd_down.points)
    
    # Cálculos
    reduction_factor = original_count / voxelized_count if voxelized_count > 0 else 0
    reduction_percentage = (1 - voxelized_count / original_count) * 100
    
    # Estimación teórica de voxels máximos
    theoretical_max_voxels = np.prod(np.ceil(coord_range / voxel_size))
    
    # Densidad de ocupación de voxels
    occupancy_rate = voxelized_count / theoretical_max_voxels * 100 if theoretical_max_voxels > 0 else 0
    
    return {
        'original_points': original_count,
        'voxelized_points': voxelized_count,
        'reduction_factor': reduction_factor,
        'reduction_percentage': reduction_percentage,
        'coord_range': coord_range,
        'voxel_size': voxel_size,
        'theoretical_max_voxels': int(theoretical_max_voxels),
        'occupancy_rate': occupancy_rate,
        'voxel_volume': voxel_size ** 3,
        'scene_volume': np.prod(coord_range)
    }

class AMTCDataset(Dataset):
    def __init__(self, areas, data_root='trainval_fullarea', num_point=4096, voxel_size=0.1, 
                feats=['coord', 'color', 'intensity'], num_classes=2, labels_available=True, 
                transform=None, hyperset=False, corrected = False, frame = None):
        super().__init__()

        self.feats = feats
        self.voxel_size = voxel_size
        self.labels_available = labels_available
        self.transform = transform
        self.feature_positions = {}
        self.corrected = corrected

        # Asegurarnos de que se pasen las áreas necesarias
        assert areas, "Debes proporcionar al menos una lista de áreas para este conjunto."
        
        if hyperset:
            selected_areas = []
            for area_path in areas:
                # area_path es algo como "experimento1/Area_1"
                full_path = os.path.join(data_root, area_path)  # data_root/experimento1/Area_1
                if os.path.isdir(full_path):  # check if it is a directory
                    selected_areas.append(area_path)

            assert selected_areas, "No se encontraron áreas válidas con las rutas especificadas."

            self.areas = selected_areas
        else:


            # Cargamos las carpetas del dataset
            all_areas = sorted([area for area in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, area)) and 'Area_' in area])
            
            # Convertimos los números de áreas en nombres
            areas = [f"Area_{i}" for i in areas]
            
            # Filtramos las áreas disponibles
            self.areas = [area for area in all_areas if area in areas]
            assert self.areas, "No se encontraron áreas válidas con los números especificados."

        print("Áreas seleccionadas:", self.areas)

        self.room_points = []
        self.room_labels = [] if self.labels_available else None
        self.room_coord_min, self.room_coord_max = [], []
        self.room_names = []
        labelweights = np.zeros(num_classes)

        # Procesamos las áreas seleccionadas
        for area in tqdm(self.areas, total=len(self.areas)):
            area_path = os.path.join(data_root, area)
            if frame is not None:
                room_list = [room for room in os.listdir(area_path) if f'amtc_{frame}' in room and os.path.isdir(os.path.join(area_path, room))]
            else:
                room_list = [room for room in os.listdir(area_path) if 'amtc_' in room and os.path.isdir(os.path.join(area_path, room))]
            for room_name in room_list:
                room_path = os.path.join(area_path, room_name)
                if self.labels_available:
                    labels = np.load(os.path.join(room_path, 'segment.npy')).reshape(-1)
                else:
                    labels = None
                self.room_names.append(room_name)

                points, coord = self.load_features(self.feats, room_path, self.corrected)

                # Calculamos los valores mínimos y máximos para normalización
                if coord is not None:
                    coord_min, coord_max = np.amin(coord, axis=0), np.amax(coord, axis=0)
                    self.room_coord_min.append(coord_min)
                    self.room_coord_max.append(coord_max)

                # Verificamos si es necesario voxelizar
                if points.shape[0] > num_point:
                    if self.labels_available:
                        voxelized_points, voxelized_labels = self.voxelize(points, labels)
                    else:
                        voxelized_points = self.voxelize(points)
                        voxelized_labels = None
                    self.room_points.append(voxelized_points)
                    if self.labels_available:
                        self.room_labels.append(voxelized_labels)
                else:
                    # Si no se requiere voxelización, almacenamos los puntos originales
                    self.room_points.append(points)
                    if self.labels_available:
                        self.room_labels.append(labels)

                if self.labels_available:
                    # Acumulamos los pesos de las etiquetas
                    tmp, _ = np.histogram(labels, range(num_classes + 1))
                    labelweights += tmp

        if self.labels_available:
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.amax(labelweights) / labelweights
            self.labelweights = self.labelweights / np.sum(self.labelweights)
            print('Labelweights: ', self.labelweights)
        else:
            self.labelweights = None

        self.num_feats = self.room_points[0].shape[1]  # coord = 3; color = 3; intensity = 1
        print('Number of feats in dataset: ', self.num_feats)

        if num_point is None or num_point < points.shape[0]:
            self.num_point = points.shape[0]
        else:
            self.num_point = num_point

    def _calculate_feature_positions(self):
        feature_map = {'coord': 3, 'color': 3, 'normal': 3, 'intensity': 1, 'flow': 3, 'diff':1, 'diff_vectors':3, 'interp':1}  # Define dimensiones
        positions = {}
        pos = 0
        for feat in self.feats:
            positions[feat] = (pos, pos + feature_map[feat])
            pos += feature_map[feat]
        return positions

    def voxelize(self, points, labels=None):
        # Aplicamos la voxelización con el tamaño de voxel definido
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Solo usamos las coordenadas

        # Voxelización
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Obtenemos las coordenadas voxelizadas
        voxelized_points = np.asarray(pcd_down.points)

        # Mapeamos las características adicionales (color, intensidad) a los puntos voxelizados
        voxelized_features = self.map_voxel_features(points[:, 3:], points[:, :3], voxelized_points)

        # Concatenamos las coordenadas voxelizadas con las características adicionales
        voxelized_features = np.concatenate([voxelized_points, voxelized_features], axis=1)

        if self.labels_available and labels is not None:
            # Mapeamos las etiquetas de los puntos originales a los puntos voxelizados
            voxelized_labels = self.map_voxel_labels(points[:, :3], voxelized_points, labels)
            return voxelized_features, voxelized_labels
        else:
            return voxelized_features

    def map_voxel_labels(self, original_points, voxelized_points, labels):
        # Esta función asigna etiquetas a los puntos voxelizados
        tree = cKDTree(original_points)
        _, idx = tree.query(voxelized_points)
        voxelized_labels = labels[idx]
        return voxelized_labels

    def map_voxel_features(self, original_features, original_points, voxelized_points):
        # Creamos un árbol KD para buscar los vecinos más cercanos
        tree = cKDTree(original_points)
        _, idx = tree.query(voxelized_points)
        
        # Lista para almacenar las características mapeadas
        mapped_features = []

        # Iteramos por cada característica usando `feature_positions`
        for feat, (start, end) in self.feature_positions.items():
            # Extraemos la porción correspondiente de `original_features`
            feature_slice = original_features[:, start:end]
            # Mapeamos las características utilizando el índice del KDTree
            mapped_feature = feature_slice[idx]
            mapped_features.append(mapped_feature)
        
        # Concatenamos todas las características mapeadas
        voxelized_features = np.concatenate(mapped_features, axis=1)
        return voxelized_features


    def load_features(self, feats, room_path, corrected):

        # Mapas de archivos base y corregidos
        base_feature_map = {
            'coord': 'coord.npy',       # Coordenadas (N, 3)
            'color': 'color.npy',       # Colores (N, 3)
            'normal': 'normal.npy',     # Normales (N, 3)
            'intensity': 'intensity.npy',  # Intensidad (N, 1)
            'flow': 'flow.npy',          # Scene flow (N, 3)
            'diff': 'diff.npy',          # Diferencia  (N, 1)
            'diff_vectors': 'diff_vectors.npy', # Diferencia  (N, 3)
            'interp': 'interp.npy'
        }
        
        corrected_feature_map = {
            'coord': 'coord.npy',       # Coordenadas (N, 3)
            'color': 'color.npy',       # Colores (N, 3)
            'normal': 'normal.npy',     # Normales (N, 3)
            'intensity': 'intensity.npy',  # Intensidad (N, 1)
            'flow': 'flow.npy',          # Scene flow (N, 3)
            'diff': 'diff_corrected.npy',          # Diferencia corregida (N, 1)
            'diff_vectors': 'diff_vectors_corrected.npy', # Diferencia corregida (N, 3)
            'interp': 'interp_corrected.npy'
        }

        loaded_features = []
        coord = None
        position = 0  # Posición actual en la concatenación de características

        for feat in feats:
            if feat in base_feature_map:
                # Si corrected está activado, intentar cargar la versión corregida primero
                if corrected and feat in corrected_feature_map:
                    corrected_file_path = os.path.join(room_path, corrected_feature_map[feat])
                    base_file_path = os.path.join(room_path, base_feature_map[feat])
                    
                    # Intentar cargar el archivo corregido, si no existe usar el base
                    if os.path.exists(corrected_file_path):
                        file_path = corrected_file_path
                    else:
                        file_path = base_file_path
                else:
                    # Usar la versión base
                    file_path = os.path.join(room_path, base_feature_map[feat])
                
                data = np.load(file_path)
                
                # Ajuste de dimensión si es necesario
                if feat == 'intensity' or feat == 'diff':
                    data = data.reshape(-1, 1)
                
                # Agregar data a loaded_features
                loaded_features.append(data)
                
                # Guardar la posición en el diccionario
                self.feature_positions[feat] = (position, position + data.shape[1])
                
                # Actualizar posición actual
                position += data.shape[1]

                # Guardar coordenadas si es `coord`
                if feat == 'coord':
                    coord = data

        # Concatenar todas las características cargadas
        points = np.concatenate(loaded_features, axis=1)
        return points, coord

    def __getitem__(self, idx, return_index=False):
        # Selecciona la pointcloud completa del índice proporcionado
        points = self.room_points[idx]
        if self.labels_available:
            labels = self.room_labels[idx]
        else:
            labels = None

        # Recuperamos los mínimos y máximos de las coordenadas para la normalización
        room_idx = idx
        coord_min = self.room_coord_min[room_idx]
        coord_max = self.room_coord_max[room_idx]

        # Inicializamos un array para almacenar los puntos normalizados (num_feats + 3 incluye las coordenadas normalizadas)
        current_points = np.zeros((points.shape[0], self.num_feats + 3))  # N * (num_feats + 3)

        # Normalizamos las coordenadas con respecto a las coordenadas máximas de la room actual
        current_points[:, -3] = points[:, 0] / self.room_coord_max[room_idx][0]  # Normalización de X
        current_points[:, -2] = points[:, 1] / self.room_coord_max[room_idx][1]  # Normalización de Y
        current_points[:, -1] = points[:, 2] / self.room_coord_max[room_idx][2]  # Normalización de Z

        # Copiamos las características originales (como color, intensidad, etc.) en las primeras columnas
        current_points[:, 0:self.num_feats] = points

        # Aplicamos transformaciones
        if self.transform is not None:
            current_points, labels = self.transform(current_points, labels, self.feature_positions)

        if return_index:
            if self.labels_available:
                return current_points, labels, self.room_names[room_idx]
            else:
                return current_points, None, self.room_names[room_idx]
        else:
            if self.labels_available:
                return current_points, labels
            else:
                return current_points

    def __len__(self):
        return len(self.room_points)


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

# if __name__ == '__main__':
#     curr_dir = os.getcwd()
#     parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))

#     data_root = os.path.join(parent_dir, 'data', 'processed_ouster_data/grabaciones_08_11')
#     num_point, voxel_size= 5000, 0.1

#     point_data = AMTCDataset(split='train', data_root=data_root, num_point=num_point, val_test_area=[51, 61], voxel_size = voxel_size, feats = ['coord', 'intensity', 'diff'], num_classes = 2)
#     data, labels, r_idx = point_data.__getitem__(0, return_index = True)
#     print(f'Viendo escena {r_idx}...')
#     print('data shape: ', np.shape(data))
#     print('labels: ', np.unique(labels, return_counts=True))
#     vis_result(data[:,:3],labels)   # real

# if __name__ == '__main__':
#     import json
#     with open('/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/hypersets.json', 'r') as f:  
#             sets_dict = json.load(f) 
#         test_areas = sets_dict["test_set"]
#     ROOT = '/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/experimentos
#     TEST_DATASET = AMTCDataset(
#             areas=test_areas,
#             data_root=ROOT,
#             num_point=args.npoint,
#             voxel_size=args.voxel_size,
#             feats=FEATS,
#             num_classes=NUM_CLASSES,
#             labels_available=True,
#             hyperset=True,
#             corrected=args.corrected
#         )

# Código de análisis de voxelización
# def test_voxelization_analysis():
#     """
#     Función para probar el análisis de voxelización
#     """
#     print("=== ANÁLISIS DE VOXELIZACIÓN ===")
    
#     # Simular algunos puntos para demostrar
#     ROOT = '/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos'
    
#     try:
#         # Crear dataset de prueba
#         test_dataset = AMTCDataset(
#             areas=[1],  # Solo un área para prueba
#             data_root=os.path.join(ROOT, 'interior1'),
#             num_point=4096,
#             voxel_size=0.000001,  # Voxel pequeño para comparar
#             feats=['coord', 'intensity'],
#             num_classes=2,
#             labels_available=True
#         )
        
#         # Obtener un sample
#         points, labels = test_dataset[100]
#         coords = points[:, :3]  # Solo coordenadas
        
#         print(f"\nEjemplo con un frame:")
#         print(f"Coordenadas van de {coords.min(axis=0)} a {coords.max(axis=0)}")
        
#         # Probar diferentes tamaños de voxel
#         voxel_sizes = [0.1, 0.2, 0.5, 1.0]
        
#         for vs in voxel_sizes:
#             analysis = analyze_voxelization_impact(points, vs)
#             print(f"\n--- Voxel Size: {vs} ---")
#             print(f"Puntos originales: {analysis['original_points']:,}")
#             print(f"Puntos voxelizados: {analysis['voxelized_points']:,}")
#             print(f"Factor de reducción: {analysis['reduction_factor']:.2f}x")
#             print(f"Reducción porcentual: {analysis['reduction_percentage']:.1f}%")
#             print(f"Rango de coordenadas: {analysis['coord_range']}")
#             print(f"Voxels teóricos máximos: {analysis['theoretical_max_voxels']:,}")
#             print(f"Tasa de ocupación: {analysis['occupancy_rate']:.1f}%")
#             print(f"Volumen de voxel: {analysis['voxel_volume']:.3f}")
#             print(f"Volumen de escena: {analysis['scene_volume']:.3f}")
            
#     except Exception as e:
#         print(f"Error en el análisis: {e}")
#         print("Para ejecutar el análisis, asegúrate de tener datos en la ruta especificada")

# # Descomenta la siguiente línea para ejecutar el análisis:
# test_voxelization_analysis()


class AMTCDatasetCNN(Dataset):
    def __init__(self, areas, data_root, num_classes=2, labels_available=True, corrected=False, frame=None, hyperset=False):
        super().__init__()
        self.labels_available = labels_available
        self.corrected = corrected
        self.num_classes = num_classes
        self.data_root = data_root
        
        if hyperset:
            selected_areas = []
            for area_path in areas:
                # area_path es algo como "experimento1/Area_1"
                full_path = os.path.join(data_root, area_path)  # data_root/experimento1/Area_1
                if os.path.isdir(full_path):  # check if it is a directory
                    selected_areas.append(area_path)

            assert selected_areas, "No se encontraron áreas válidas con las rutas especificadas."

            self.areas = selected_areas
        else:


            # Cargamos las carpetas del dataset
            all_areas = sorted([area for area in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, area)) and 'Area_' in area])
            
            # Convertimos los números de áreas en nombres
            areas = [f"Area_{i}" for i in areas]
            
            # Filtramos las áreas disponibles
            self.areas = [area for area in all_areas if area in areas]
            assert self.areas, "No se encontraron áreas válidas con los números especificados."

        print("Áreas seleccionadas:", self.areas)

        # sufijo para archivos
        self.suffix = "_corrected" if corrected else ""

        self.samples = []
        for area in self.areas:
            area_path = os.path.join(data_root, area)
            room_list = [r for r in os.listdir(area_path) if 'amtc_' in r]
            if frame is not None:
                room_list = [r for r in room_list if f"amtc_{frame}" in r]

            for room_name in room_list:
                room_path = os.path.join(area_path, room_name)
                feat_path = os.path.join(room_path, f"voxel_feats{self.suffix}.npy")
                label_path = os.path.join(room_path, f"voxel_labels{self.suffix}.npy")
                if os.path.exists(feat_path):
                    self.samples.append((feat_path, label_path if labels_available else None))

        print(f"Dataset CNN: {len(self.samples)} samples loaded from {len(self.areas)} areas")

        # número de features por voxel
        self.num_feats = 5

        # calcular el número máximo de voxels en el dataset (para padding)
        if self.samples:
            self.num_point = max(np.load(f).shape[0] for f, _ in self.samples)
        else:
            self.num_point = 0

        # calcular labelweights
        if self.labels_available:
            all_labels = []
            for _, label_path in self.samples:
                if label_path and os.path.exists(label_path):
                    all_labels.append(np.load(label_path))
            if all_labels:
                hist, _ = np.histogram(np.concatenate(all_labels), bins=num_classes, range=(0, num_classes))
                hist = hist.astype(np.float32)
                hist /= np.sum(hist)
                self.labelweights = np.amax(hist) / hist
                self.labelweights = self.labelweights / np.sum(self.labelweights)
                print("Labelweights:", self.labelweights)
            else:
                self.labelweights = None
        else:
            self.labelweights = None

    def __getitem__(self, idx):
        feat_path, label_path = self.samples[idx]
        feats = np.load(feat_path)  # (N, 5)
        feats = torch.from_numpy(feats).float()

        if self.labels_available and label_path is not None and os.path.exists(label_path):
            labels = np.load(label_path).astype(np.int64)
            labels = torch.from_numpy(labels)
        else:
            labels = torch.full((feats.shape[0],), -1, dtype=torch.long)  # padding con -1

        return feats, labels

    def __len__(self):
        return len(self.samples)