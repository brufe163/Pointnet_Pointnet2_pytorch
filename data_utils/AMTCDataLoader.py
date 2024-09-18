import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
import open3d as o3d


class AMTCDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, voxel_size=0.1,test_area=5, feats = ['coord', 'color', 'intensity'], num_classes = 2):
        super().__init__()

        self.feats = feats
        self.voxel_size = voxel_size

        # Cargamos las carpetas del amtc como áreas
        areas = sorted(os.listdir(data_root))
        areas = [area for area in areas if 'Area_' in area]

        assert split in ['train', 'test']

        if split == 'train':
            areas_split = [area for area in areas if not 'Area_{}'.format(test_area) in area]
        else:
            areas_split = [area for area in areas if 'Area_{}'.format(test_area) in area]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        self.room_names = []
        labelweights = np.zeros(num_classes)

        # Iteramos por cada área del dataset
        for area in tqdm(areas_split, total=len(areas_split)):
            area_path = os.path.join(data_root, area)
            room_list = os.listdir(area_path)
            room_list = [room for room in room_list if 'amtc_' in room]
            for room_name in room_list:
                room_path = os.path.join(area_path, room_name)
                labels = np.load(os.path.join(room_path, 'segment.npy'))  # Etiquetas (N,)
                labels = labels.reshape(-1)
                self.room_names.append(room_name)

                points, coord = self.load_features(self.feats, room_path)

                # Calculamos los valores mínimos y máximos para normalización
                if coord is not None:
                    coord_min, coord_max = np.amin(coord, axis=0), np.amax(coord, axis=0)
                    self.room_coord_min.append(coord_min)
                    self.room_coord_max.append(coord_max)

                # Verificamos si es necesario voxelizar
                if points.shape[0] > num_point:
                    voxelized_points, voxelized_labels = self.voxelize(points, labels)
                    self.room_points.append(voxelized_points)
                    self.room_labels.append(voxelized_labels)
                else:
                    # Si no se requiere voxelización, almacenamos los puntos originales
                    self.room_points.append(points)
                    self.room_labels.append(labels)

                # Acumulamos los pesos de las etiquetas
                tmp, _ = np.histogram(labels, range(num_classes + 1))
                labelweights += tmp


        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = [10, 1] #np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print('Labelweights: ', self.labelweights)
        self.num_feats = self.room_points[0].shape[1] # coord = 3; color = 3; intensity = 1
        print('Number of feats in dataset: ', self.num_feats)

        if num_point is None or num_point < points.shape[0]:
            self.num_point = points.shape[0]
        else:        
            self.num_point = num_point

    def voxelize(self, points, labels):
        # Aplicamos la voxelización con el tamaño de voxel definido
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Solo usamos las coordenadas

        # Voxelización
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Obtenemos las coordenadas voxelizadas
        voxelized_points = np.asarray(pcd_down.points)

        # Mapeamos las etiquetas de los puntos originales a los puntos voxelizados
        voxelized_labels = self.map_voxel_labels(points[:, :3], voxelized_points, labels)

        # Mapeamos las características adicionales (color, intensidad) a los puntos voxelizados
        voxelized_features = self.map_voxel_features(points[:, 3:], points[:, :3], voxelized_points)

        # Concatenamos las coordenadas voxelizadas con las características adicionales
        voxelized_features = np.concatenate([voxelized_points, voxelized_features], axis=1)

        return voxelized_features, voxelized_labels

    def map_voxel_labels(self, original_points, voxelized_points, labels):
        # Esta función asigna etiquetas a los puntos voxelizados
        tree = cKDTree(original_points)
        _, idx = tree.query(voxelized_points)
        voxelized_labels = labels[idx]
        return voxelized_labels

    def map_voxel_features(self, original_features, original_points, voxelized_points):
        # Esta función asigna características adicionales (color, intensidad) a los puntos voxelizados
        tree = cKDTree(original_points)
        _, idx = tree.query(voxelized_points)
        voxelized_features = original_features[idx]
        return voxelized_features

    def load_features(self, feats, room_path):
        feature_map = {
            'coord': 'coord.npy',       # Coordenadas (N, 3)
            'color': 'color.npy',       # Colores (N, 3)
            'normal': 'normal.npy',     # Normales (N, 3) (Aquí solo como ejemplo, puedes agregar la carga si tienes estos datos)
            'intensity': 'intensity.npy'  # Intensidad (N, 1)
        }   # TODO: agregar doble rebote (considerar si son las mismas dimensiones que en el primero)
        
        loaded_features = []
        coord = None
        for feat in feats:
            if feat in feature_map:
                file_path = os.path.join(room_path, feature_map[feat])
                data = np.load(file_path)
                if feat == 'intensity': 
                    loaded_features.append(data.reshape(-1, 1))
                else: 
                    loaded_features.append(data)
                if feat == 'coord':
                    coord = data
    
        # Concatenar todas las características cargadas a lo largo de la segunda dimensión (axis=1)
        points = np.concatenate(loaded_features, axis=1)
        return points, coord


    def __getitem__(self, idx,return_index = False):
        # Selecciona la pointcloud completa del índice proporcionado
        points = self.room_points[idx]
        labels = self.room_labels[idx]

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

        # Devuelve todos los puntos normalizados y sus etiquetas
        if return_index:
            return current_points, labels, self.room_names[room_idx]
        else:
            return current_points, labels


    def __len__(self):
        return len(self.room_points)

class AMTCRealDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=1, feats=['coord', 'intensity'], angle = None):
        super().__init__()
        self.feats = feats
        self.angle = angle

        # Cargamos las carpetas del amtc como áreas
        areas = sorted(os.listdir(data_root))
        areas = [area for area in areas if 'Area_' in area]

        assert split in ['train', 'test']

        if split == 'train':
            areas_split = [area for area in areas if not 'Area_{}'.format(test_area) in area]
        else:
            areas_split = [area for area in areas if 'Area_{}'.format(test_area) in area]

        self.room_points = []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        self.room_names = []

        # Iteramos por cada área del dataset
        for area in tqdm(areas_split, total=len(areas_split)):
            area_path = os.path.join(data_root, area)
            room_list = os.listdir(area_path)
            room_list = [room for room in room_list if 'amtc_' in room]
            for room_name in room_list:
                room_path = os.path.join(area_path, room_name)
                self.room_names.append(room_name)

                points, coord = self.load_features(self.feats, room_path)  # Cargamos las características

                if coord is not None:
                    coord_min, coord_max = np.amin(coord, axis=0), np.amax(coord, axis=0)
                    self.room_coord_min.append(coord_min)
                    self.room_coord_max.append(coord_max)
            
                self.room_points.append(points)
                num_point_all.append(points.shape[0])

        self.num_feats = self.room_points[0].shape[1]  # coord = 3; color = 3; intensity = 1
        print('Number of feats in dataset: ', self.num_feats)

        if num_point is None:
            self.num_point = points.shape[0]
        else:        
            self.num_point = num_point

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) / self.num_point)

        room_idxs = []
        for index in range(len(self.room_points)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))

        self.room_idxs = np.array(room_idxs)

    def filter_points_within_angle(self, data):
        half_angle = np.radians(self.angle/ 2) 
        # data contiene puntos y características: x, y, z, i, ...
        x = data[:, 0]  # Extrae todas las x
        y = data[:, 1]  # Extrae todas las y
        
        # Calcula el ángulo en el plano XY para todos los puntos
        theta = np.arctan2(y, x)
        
        # Filtra los puntos que están dentro del rango de ángulo
        mask = (-half_angle <= theta) & (theta <= half_angle)
        
        # Aplica la máscara para seleccionar las filas filtradas
        filtered_data = data[mask]
        
        return filtered_data

    def load_features(self, feats, room_path):
        feature_map = {
            'coord_1': 'coord_1.npy',       # Coordenadas (N, 3)
            'coord_2': 'coord_2.npy',       # Coordenadas (N, 3)
            'coord_12': 'coord_12.npy',       # Coordenadas (2*N, 3)
            'intensity_1': 'intensity_1.npy',  # Intensidad (N, 1)
            'intensity_2': 'intensity_2.npy',  # Intensidad (N, 1)
            'intensity_12': 'intensity_12.npy',  # Intensidad (2*N, 1)
            'reflectivity_1': 'reflectivity_1.npy',  # Reflectividad (N, 1)
            'reflectivity_2': 'reflectivity_2.npy',  # Reflectividad (N, 1)
            'reflectivity_12': 'reflectivity_12.npy',  # Reflectividad (2*N, 1)
        }
        
        loaded_features = []
        coord = None
        for feat in feats:
            if feat in feature_map:
                file_path = os.path.join(room_path, feature_map[feat])
                data = np.load(file_path)
                if 'intensity' in feat or 'reflectivity' in feat:
                    loaded_features.append(data.reshape(-1, 1))
                else: 
                    loaded_features.append(data)
                if 'coord' in feat:
                    coord = data

        points = np.concatenate(loaded_features, axis=1)
        return points, coord

    def __getitem__(self, idx, return_index=False):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * num_feats

        if self.angle is not None:
            points = self.filter_points_within_angle(points)

        current_points = np.zeros((points.shape[0], self.num_feats + 3))  # N * (num_feats + 3)

        current_points[:, -3] = points[:, 0] / self.room_coord_max[room_idx][0]  # Normalización de X
        current_points[:, -2] = points[:, 1] / self.room_coord_max[room_idx][1]  # Normalización de Y
        current_points[:, -1] = points[:, 2] / self.room_coord_max[room_idx][2]  # Normalización de Z

        current_points[:, 0:self.num_feats] = points
        if return_index:
            return current_points, self.room_names[room_idx]
        else:
            return current_points

    def __len__(self):
        return len(self.room_idxs)

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

if __name__ == '__main__':
    curr_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))

    data_root = os.path.join(parent_dir, 'data', 'blender_outside_ns_md2')
    num_point, test_area, voxel_size= 5000, 2, 0.1

    point_data = AMTCDataset(split='train', data_root=data_root, num_point=num_point, test_area=2, voxel_size = voxel_size, feats = ['coord', 'intensity'], num_classes = 2)
    data, labels, r_idx = point_data.__getitem__(0, return_index = True)
    print(f'Viendo escena {r_idx}...')
    print('data shape: ', np.shape(data))
    print('labels: ', np.unique(labels, return_counts=True))
    vis_result(data[:,:3],labels)   # real