import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm
from torch.utils.data import Dataset

class AMTCDataset:
    def __init__(self, data_root='trainval_fullarea', num_point=4096, voxel_size=0.1,
                 feats=['coord', 'color', 'intensity'], num_classes=2, labels_available=True,
                 split_ratios=(0.6, 0.2, 0.2)):
        super().__init__()

        self.feats = feats
        self.voxel_size = voxel_size
        self.labels_available = labels_available
        self.split_ratios = split_ratios
        assert sum(split_ratios) == 1.0, "Los porcentajes de división deben sumar 1"

        # Cargamos las carpetas del AMTC como áreas
        areas = sorted(os.listdir(data_root))
        areas = [area for area in areas if 'Area_' in area]

        self.room_coord_min = []
        self.room_coord_max = []
        self.room_names = []
        labelweights = np.zeros(num_classes)

        # Inicializamos listas para cada conjunto
        self.train_points = []
        self.train_labels = []
        self.val_points = []
        self.val_labels = []
        self.test_points = []
        self.test_labels = []

        # Iteramos por cada área del dataset
        for area in tqdm(areas, total=len(areas)):
            area_path = os.path.join(data_root, area)
            room_list = os.listdir(area_path)
            room_list = [room for room in room_list if 'amtc_' in room]
            for room_name in room_list:
                room_path = os.path.join(area_path, room_name)
                if self.labels_available:
                    labels = np.load(os.path.join(room_path, 'segment.npy'))  # Etiquetas (N,)
                    labels = labels.reshape(-1)
                else:
                    labels = None  # No cargamos etiquetas
                room_idx = len(self.room_names)
                self.room_names.append(room_name)

                points, coord = self.load_features(self.feats, room_path)

                # Calculamos los valores mínimos y máximos para normalización
                if coord is not None:
                    coord_min, coord_max = np.amin(coord, axis=0), np.amax(coord, axis=0)
                    self.room_coord_min.append(coord_min)
                    self.room_coord_max.append(coord_max)

                # Verificamos si es necesario voxelizar
                if points.shape[0] > num_point:
                    if self.labels_available:
                        points, labels = self.voxelize(points, labels)
                    else:
                        points = self.voxelize(points)
                        labels = None

                # División de datos en train, val y test
                total_points = points.shape[0]
                indices = np.arange(total_points)
                np.random.shuffle(indices)

                train_end = int(self.split_ratios[0] * total_points)
                val_end = train_end + int(self.split_ratios[1] * total_points)

                train_indices = indices[:train_end]
                val_indices = indices[train_end:val_end]
                test_indices = indices[val_end:]

                # Almacenamos los datos por escena
                self.train_points.append(points[train_indices])
                if self.labels_available:
                    self.train_labels.append(labels[train_indices])

                self.val_points.append(points[val_indices])
                if self.labels_available:
                    self.val_labels.append(labels[val_indices])

                self.test_points.append(points[test_indices])
                if self.labels_available:
                    self.test_labels.append(labels[test_indices])

                if self.labels_available:
                    # Acumulamos los pesos de las etiquetas
                    tmp, _ = np.histogram(labels, range(num_classes + 1))
                    labelweights += tmp

        if self.labels_available:
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = [10, 1]  # np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
            print('Labelweights: ', self.labelweights)
        else:
            self.labelweights = None

        self.num_feats = self.train_points[0].shape[1]  # coord = 3; color = 3; intensidad = 1
        print('Number of feats in dataset: ', self.num_feats)

        # Creamos las instancias de los datasets para cada conjunto
        self.train = self.AMTCDatasetSplit(
            self.train_points,
            self.train_labels,
            self.room_coord_min,
            self.room_coord_max,
            self.num_feats,
            self.labels_available
        )
        self.val = self.AMTCDatasetSplit(
            self.val_points,
            self.val_labels,
            self.room_coord_min,
            self.room_coord_max,
            self.num_feats,
            self.labels_available
        )
        self.test = self.AMTCDatasetSplit(
            self.test_points,
            self.test_labels,
            self.room_coord_min,
            self.room_coord_max,
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
            'intensity': 'intensity.npy'  # Intensidad (N, 1)
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
        def __init__(self, points_list, labels_list, room_coord_min, room_coord_max, num_feats, labels_available):
            self.points_list = points_list  # Lista de arrays de puntos por escena
            self.labels_list = labels_list  # Lista de arrays de etiquetas por escena
            self.room_coord_min = room_coord_min
            self.room_coord_max = room_coord_max
            self.num_feats = num_feats
            self.labels_available = labels_available

        def __len__(self):
            return len(self.points_list)

        def __getitem__(self, idx):
            points = self.points_list[idx]
            room_idx = idx
            coord_min = self.room_coord_min[room_idx]
            coord_max = self.room_coord_max[room_idx]

            # Normalización de las coordenadas
            current_points = np.zeros((points.shape[0], self.num_feats + 3))
            current_points[:, -3] = points[:, 0] / coord_max[0]  # Normalización de X
            current_points[:, -2] = points[:, 1] / coord_max[1]  # Normalización de Y
            current_points[:, -1] = points[:, 2] / coord_max[2]  # Normalización de Z

            # Copiamos las características originales en las primeras columnas
            current_points[:, 0:self.num_feats] = points

            if self.labels_available:
                labels = self.labels_list[idx]
                return current_points, labels
            else:
                return current_points


import matplotlib.pyplot as plt

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

# Código de prueba
if __name__ == '__main__':
    import os
    import numpy as np

    curr_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))

    data_root = os.path.join(parent_dir, 'data', 'blender_outside_ns_md2')
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

    # Obtenemos un ejemplo del conjunto de entrenamiento
    data, labels = dataset.train[0]
    room_name = dataset.room_names[0]

    print(f'Viendo escena {room_name}...')
    print('data shape: ', np.shape(data))
    print('labels: ', np.unique(labels, return_counts=True))
    vis_result(data[:, :3], labels)
