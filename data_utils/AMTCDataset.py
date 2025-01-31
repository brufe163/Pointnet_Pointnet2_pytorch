import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm
from torch.utils.data import Dataset

class AMTCDataset:
    def __init__(self, data_root='trainval_fullarea', num_point=4096, voxel_size=0.1,
                 feats=['coord', 'color', 'intensity'], num_classes=2, labels_available=True,
                 split_ratios=(0.6, 0.2, 0.2), random_seed=None):  # Añadimos random_seed
        super().__init__()

        self.feats = feats
        self.voxel_size = voxel_size
        self.labels_available = labels_available
        self.split_ratios = split_ratios
        assert sum(split_ratios) == 1.0, "Los porcentajes de división deben sumar 1"

        # Si se proporciona una semilla, se establece para la reproducibilidad
        if random_seed is not None:
            np.random.seed(random_seed)

        # Cargamos las carpetas del AMTC como áreas
        areas = sorted(os.listdir(data_root))
        areas = [area for area in areas if 'Area_' in area]

        # Lista para almacenar información de todas las escenas
        all_rooms = []

        # Iteramos por cada área del dataset
        for area in tqdm(areas, total=len(areas)):
            area_path = os.path.join(data_root, area)
            room_list = os.listdir(area_path)
            room_list = [room for room in room_list if 'amtc_' in room]
            for room_name in room_list:
                room_path = os.path.join(area_path, room_name)
                all_rooms.append((area, room_name, room_path))

        # Barajamos las escenas
        np.random.shuffle(all_rooms)

        # Calculamos los índices de corte para los splits
        total_rooms = len(all_rooms)
        train_end = int(self.split_ratios[0] * total_rooms)
        val_end = train_end + int(self.split_ratios[1] * total_rooms)

        # Dividimos las escenas
        train_rooms = all_rooms[:train_end]
        val_rooms = all_rooms[train_end:val_end]
        test_rooms = all_rooms[val_end:]

        # Inicializamos listas para cada conjunto
        self.train_points = []
        self.train_labels = []
        self.train_coord_min = []
        self.train_coord_max = []
        self.train_room_names = []
        self.train_area_names = []

        self.val_points = []
        self.val_labels = []
        self.val_coord_min = []
        self.val_coord_max = []
        self.val_room_names = []
        self.val_area_names = []

        self.test_points = []
        self.test_labels = []
        self.test_coord_min = []
        self.test_coord_max = []
        self.test_room_names = []
        self.test_area_names = []

        # Acumulador de labelweights
        labelweights = np.zeros(num_classes)

        # Función para procesar las escenas
        def process_rooms(room_list, points_list, labels_list, coord_min_list, coord_max_list, room_names_list, area_names_list):
            for area_name, room_name, room_path in room_list:
                if self.labels_available:
                    labels = np.load(os.path.join(room_path, 'segment.npy')).reshape(-1)
                else:
                    labels = None

                points, coord = self.load_features(self.feats, room_path)

                # Calculamos los valores mínimos y máximos para normalización
                if coord is not None:
                    coord_min, coord_max = np.amin(coord, axis=0), np.amax(coord, axis=0)
                else:
                    coord_min, coord_max = np.zeros(3), np.ones(3)

                # Verificamos si es necesario voxelizar
                if points.shape[0] > num_point:
                    if self.labels_available:
                        points, labels = self.voxelize(points, labels)
                    else:
                        points = self.voxelize(points)
                        labels = None

                points_list.append(points)
                coord_min_list.append(coord_min)
                coord_max_list.append(coord_max)
                room_names_list.append(room_name)  # Almacenamos el nombre de la habitación
                area_names_list.append(area_name)  # Almacenamos el nombre del área
                if self.labels_available:
                    labels_list.append(labels)

                    # Acumulamos los pesos de las etiquetas
                    tmp, _ = np.histogram(labels, range(num_classes + 1))
                    nonlocal labelweights
                    labelweights += tmp

        # Procesamos los conjuntos
        process_rooms(train_rooms, self.train_points, self.train_labels, self.train_coord_min, self.train_coord_max, self.train_room_names, self.train_area_names)
        process_rooms(val_rooms, self.val_points, self.val_labels, self.val_coord_min, self.val_coord_max, self.val_room_names, self.val_area_names)
        process_rooms(test_rooms, self.test_points, self.test_labels, self.test_coord_min, self.test_coord_max, self.test_room_names, self.test_area_names)

        if self.labels_available:
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = [10, 1]  # Puedes ajustar estos valores según tus necesidades
            print('Labelweights: ', self.labelweights)
        else:
            self.labelweights = None

        self.num_feats = self.train_points[0].shape[1]  # coord = 3; color = 3; intensidad = 1
        print('Number of feats in dataset: ', self.num_feats)

        # Creamos las instancias de los datasets para cada conjunto
        self.train = self.AMTCDatasetSplit(
            self.train_points,
            self.train_labels,
            self.train_coord_min,
            self.train_coord_max,
            self.train_room_names,
            self.train_area_names,
            self.num_feats,
            self.labels_available
        )
        self.val = self.AMTCDatasetSplit(
            self.val_points,
            self.val_labels,
            self.val_coord_min,
            self.val_coord_max,
            self.val_room_names,
            self.val_area_names,
            self.num_feats,
            self.labels_available
        )
        self.test = self.AMTCDatasetSplit(
            self.test_points,
            self.test_labels,
            self.test_coord_min,
            self.test_coord_max,
            self.test_room_names,
            self.test_area_names,
            self.num_feats,
            self.labels_available
        )

    def voxelize(self, points, labels=None):
        # Aplicamos la voxelización con el tamaño de voxel definido
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Solo usamos las coordenadas

        # Voxelización
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Obtenemos las coordenadas voxelizadas
        voxelized_points = np.asarray(pcd_down.points)

        # Mapeamos las características adicionales (color, intensidad) a los puntos voxelizados
        if points.shape[1] > 3:
            voxelized_features = self.map_voxel_features(points[:, 3:], points[:, :3], voxelized_points)
            # Concatenamos las coordenadas voxelizadas con las características adicionales
            voxelized_features = np.concatenate([voxelized_points, voxelized_features], axis=1)
        else:
            voxelized_features = voxelized_points

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
        # Esta función asigna características adicionales (color, intensidad) a los puntos voxelizados
        tree = cKDTree(original_points)
        _, idx = tree.query(voxelized_points)
        voxelized_features = original_features[idx]
        return voxelized_features

    def load_features(self, feats, room_path):
        feature_map = {
            'coord': 'coord.npy',       # Coordenadas (N, 3)
            'color': 'color.npy',       # Colores (N, 3)
            'normal': 'normal.npy',     # Normales (N, 3)
            'intensity': 'intensity.npy',  # Intensidad (N, 1)
            'flow': 'flow.npy' # Scene Flow (N, 3)
        }
        # Puedes agregar más características si están disponibles

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

    class AMTCDatasetSplit(Dataset):
        def __init__(self, points_list, labels_list, coord_min_list, coord_max_list, room_names_list, area_names_list, num_feats, labels_available):
            self.points_list = points_list  # Lista de pointclouds
            self.labels_list = labels_list  # Lista de etiquetas
            self.coord_min_list = coord_min_list
            self.coord_max_list = coord_max_list
            self.room_names_list = room_names_list  # Lista de nombres de habitaciones
            self.area_names_list = area_names_list  # Lista de nombres de áreas
            self.num_feats = num_feats
            self.labels_available = labels_available

        def __len__(self):
            return len(self.points_list)

        def __getitem__(self, idx):
            points = self.points_list[idx]
            coord_min = self.coord_min_list[idx]
            coord_max = self.coord_max_list[idx]
            room_name = self.room_names_list[idx]
            area_name = self.area_names_list[idx]

            # Normalización de las coordenadas
            current_points = np.zeros((points.shape[0], self.num_feats + 3))
            # Evitar divisiones por cero en caso de que coord_max sea cero
            coord_max = np.where(coord_max == 0, 1e-6, coord_max)
            current_points[:, -3] = points[:, 0] / coord_max[0]  # Normalización de X
            current_points[:, -2] = points[:, 1] / coord_max[1]  # Normalización de Y
            current_points[:, -1] = points[:, 2] / coord_max[2]  # Normalización de Z

            # Copiamos las características originales en las primeras columnas
            current_points[:, 0:self.num_feats] = points

            if self.labels_available:
                labels = self.labels_list[idx]
                return current_points, labels, room_name, area_name
            else:
                return current_points, room_name, area_name

import matplotlib.pyplot as plt

def vis_result(coord, classes):
    num_classes = len(np.unique(classes))
    colors = np.zeros((len(classes), 3))
    
    # Calcular el centro de la pointcloud
    center = np.mean(coord, axis=0)
    
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

    # Establecer los parámetros de la cámara (ajusta estos valores si es necesario)
    front = [-0.5, -0.5, 0.5]
    lookat = [0.0, 0.0, 0.0]
    up = [0.0, 0.0, 1.0]
    zoom = 0.5

    view_ctl.set_front(front)
    view_ctl.set_lookat(lookat)
    view_ctl.set_up(up)
    view_ctl.set_zoom(zoom)
    
    vis.run()  # Iniciar el visualizador
    vis.destroy_window()

# Código de prueba
if __name__ == '__main__':
    import os
    import numpy as np

    curr_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))

    data_root = os.path.join(parent_dir, 'data', 'blender_areas')  # Ajusta esta ruta a tu dataset
    num_point, voxel_size = 5000, 0.1

    # Instanciamos el dataset
    dataset = AMTCDataset(
        data_root=data_root,
        num_point=num_point,
        voxel_size=voxel_size,
        feats=['coord', 'intensity'],
        num_classes=2,
        labels_available=True,
        split_ratios=(0.6, 0.2, 0.2)
    )

    print("The number of training data is:", len(dataset.train))
    print("The number of validation data is:", len(dataset.val))
    print("The number of test data is:", len(dataset.test))

    # Obtenemos un ejemplo del conjunto de entrenamiento
    data, labels, room_name, area_name = dataset.train[0]

    print(f'Visualizando la habitación: {room_name} del área: {area_name}')
    print('Data shape:', np.shape(data))
    print('Labels:', np.unique(labels, return_counts=True))
    
    # Visualizamos la pointcloud
    vis_result(data[:, :3], labels)
