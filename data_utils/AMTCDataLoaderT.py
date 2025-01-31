import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
import open3d as o3d

from data_utils.AMTCTransforms import *

class AMTCDatasetT(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, voxel_size=0.1, val_test_area=[0,1], feats=['coord', 'color', 'intensity'], num_classes=2, labels_available=True, transform=None, sequence_length=4):
        super().__init__()

        self.feats = feats
        self.voxel_size = voxel_size
        self.labels_available = labels_available
        self.transform = transform
        self.sequence_length = sequence_length  # Nuevo parámetro
        self.feature_positions = {}

        # Cargamos las carpetas del AMTC como áreas
        areas = sorted([area for area in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, area)) and 'Area_' in area])
        print("Áreas encontradas:", areas)

        assert split in ['train', 'test', 'val']

        if split == 'train':
            areas_split = [area for area in areas if all('Area_{}'.format(v) not in area for v in val_test_area)]
        elif split == 'val':
            areas_split = [area for area in areas if 'Area_{}'.format(val_test_area[0]) in area]
        elif split == 'test':
            areas_split = [area for area in areas if 'Area_{}'.format(val_test_area[1]) in area]

        self.sequences = []
        self.sequence_labels = [] if self.labels_available else None
        self.sequence_coord_min, self.sequence_coord_max = [], []
        self.sequence_names = []
        labelweights = np.zeros(num_classes)

        # Iteramos por cada área del dataset
        for area in tqdm(areas_split, total=len(areas_split)):
            area_path = os.path.join(data_root, area)
            room_list = os.listdir(area_path)
            room_list = sorted([room for room in room_list if 'amtc_' in room])
            num_rooms = len(room_list)

            # Crear secuencias a partir de las habitaciones (o frames)
            for i in range(0, num_rooms - self.sequence_length + 1):
                sequence_points = []
                sequence_labels = [] if self.labels_available else None
                sequence_coord_min = []
                sequence_coord_max = []
                sequence_names = []

                for j in range(self.sequence_length):
                    room_name = room_list[i + j]
                    room_path = os.path.join(area_path, room_name)

                    if self.labels_available:
                        labels = np.load(os.path.join(room_path, 'segment.npy'))
                        labels = labels.reshape(-1)
                    else:
                        labels = None

                    points, coord = self.load_features(self.feats, room_path)

                    # Calculamos los valores mínimos y máximos para normalización
                    if coord is not None:
                        coord_min, coord_max = np.amin(coord, axis=0), np.amax(coord, axis=0)
                        sequence_coord_min.append(coord_min)
                        sequence_coord_max.append(coord_max)

                    # Voxelización si es necesario
                    if points.shape[0] > num_point:
                        if self.labels_available:
                            voxelized_points, voxelized_labels = self.voxelize(points, labels)
                        else:
                            voxelized_points = self.voxelize(points)
                            voxelized_labels = None
                    else:
                        voxelized_points = points
                        voxelized_labels = labels

                    sequence_points.append(voxelized_points)
                    if self.labels_available:
                        sequence_labels.append(voxelized_labels)

                    sequence_names.append(room_name)

                    if self.labels_available:
                        tmp, _ = np.histogram(labels, range(num_classes + 1))
                        labelweights += tmp

                # Guardamos la secuencia completa
                self.sequences.append(sequence_points)
                if self.labels_available:
                    self.sequence_labels.append(sequence_labels)
                self.sequence_coord_min.append(sequence_coord_min)
                self.sequence_coord_max.append(sequence_coord_max)
                self.sequence_names.append(sequence_names)

        if self.labels_available:
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = np.amax(labelweights) / labelweights
            self.labelweights = self.labelweights / np.sum(self.labelweights)
            print('Labelweights: ', self.labelweights)
        else:
            self.labelweights = None

        # Determinar el número de características
        self.num_feats = self.sequences[0][0].shape[1]
        print('Number of feats in dataset: ', self.num_feats)

        if num_point is None or num_point < points.shape[0]:
            self.num_point = points.shape[0]
        else:
            self.num_point = num_point

    def _calculate_feature_positions(self):
        feature_map = {'coord': 3, 'color': 3, 'normal': 3, 'intensity': 1, 'flow': 3}  # Define dimensiones
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
            'flow': 'flow.npy'          # Scene flow (N, 3)
        }

        loaded_features = []
        coord = None
        position = 0  # Posición actual en la concatenación de características

        for feat in feats:
            if feat in feature_map:
                file_path = os.path.join(room_path, feature_map[feat])
                data = np.load(file_path)
                
                # Ajuste de dimensión si es necesario
                if feat == 'intensity':
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
        # Seleccionamos la secuencia completa
        sequence_points = self.sequences[idx]  # Lista de length T, cada uno con forma [N_i, C]
        if self.labels_available:
            sequence_labels = self.sequence_labels[idx]
        else:
            sequence_labels = None

        sequence_coord_min = self.sequence_coord_min[idx]
        sequence_coord_max = self.sequence_coord_max[idx]

        sequence_names = self.sequence_names[idx]

        T = len(sequence_points)
        processed_sequence = []
        processed_labels = []

        for t in range(T):
            points = sequence_points[t]
            if self.labels_available:
                labels = sequence_labels[t]
            else:
                labels = None

            coord_min = sequence_coord_min[t]
            coord_max = sequence_coord_max[t]

            # Muestreamos num_point puntos de la nube actual
            if points.shape[0] >= self.num_point:
                choice = np.random.choice(points.shape[0], self.num_point, replace=False)
            else:
                # Si hay menos puntos que num_point, hacemos muestreo con reemplazo
                choice = np.random.choice(points.shape[0], self.num_point, replace=True)
            points = points[choice, :]
            if labels is not None:
                labels = labels[choice]

            # Normalizamos las coordenadas
            current_points = np.zeros((self.num_point, self.num_feats + 3))  # N * (num_feats + 3)
            current_points[:, -3] = points[:, 0] / coord_max[0]  # X
            current_points[:, -2] = points[:, 1] / coord_max[1]  # Y
            current_points[:, -1] = points[:, 2] / coord_max[2]  # Z

            # Copiamos las características originales
            current_points[:, 0:self.num_feats] = points

            # Aplicamos transformaciones
            if self.transform is not None:
                current_points, labels = self.transform(current_points, labels, self.feature_positions)

            processed_sequence.append(current_points)
            if self.labels_available:
                processed_labels.append(labels)

        # Apilamos las secuencias
        processed_sequence = np.stack(processed_sequence)  # [T, num_point, num_feats + 3]
        if self.labels_available:
            processed_labels = np.stack(processed_labels)  # [T, num_point]
        else:
            processed_labels = None

        if return_index:
            if self.labels_available:
                return processed_sequence, processed_labels, sequence_names
            else:
                return processed_sequence, None, sequence_names
        else:
            if self.labels_available:
                return processed_sequence, processed_labels
            else:
                return processed_sequence



    def __len__(self):
        return len(self.sequences)

if __name__ == '__main__':
    curr_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))

    data_root = os.path.join(parent_dir, 'data', 'blender_outside_ns_md2')
    num_point, val_test_area, voxel_size= 5000, [0,1], 0.1

    dataset = AMTCDatasetT(
        split='train',
        data_root=data_root,
        num_point=num_point,
        voxel_size=voxel_size,
        val_test_area=val_test_area,
        feats=['coord', 'color', 'intensity'],
        num_classes=2,
        labels_available=True,
        transform=None,
        sequence_length=4  # Número de frames por secuencia
    )

    sequence_data, sequence_labels = dataset[0]
    print(sequence_data.shape)  # Debería ser [T, N, num_feats + 3], donde T = 4

