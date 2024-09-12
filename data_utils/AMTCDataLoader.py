import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class AMTCDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, feats = ['coord', 'color', 'intensity'], num_classes = 2, transform=None):
        super().__init__()

        self.block_size = block_size
        self.transform = transform
        self.feats = feats

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
        num_point_all = []
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
                labels = labels.reshape(-1)  # Etiquetas (N,1)
                self.room_names.append(room_name)

                points, coord = self.load_features(self.feats, room_path)   # points debería llamarse "feats" pero así estaba en el código original

                tmp, _ = np.histogram(labels, range(num_classes+1))
                labelweights += tmp
                if coord is not None:
                    coord_min, coord_max = np.amin(coord, axis=0), np.amax(coord, axis=0)
                    self.room_coord_min.append(coord_min)
                    self.room_coord_max.append(coord_max)
            
                self.room_points.append(points)
                self.room_labels.append(labels)
                num_point_all.append(labels.size)


        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print('Labelweights: ', self.labelweights)
        self.num_feats = self.room_points[0].shape[1] # coord = 3; color = 3; intensity = 1
        print('Number of feats in dataset: ', self.num_feats)

        if num_point is None:
            self.num_point = points.shape[0]
        else:        
            self.num_point = num_point

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / self.num_point)



        room_idxs = []
        for index in range(len(self.room_points)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))

        self.room_idxs = np.array(room_idxs)


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
        idx = 0
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * num_feats
        labels = self.room_labels[room_idx]   # N

        # Inicializamos un array para almacenar los puntos normalizados (num_feats + 3 incluye las coordenadas normalizadas)
        current_points = np.zeros((points.shape[0], self.num_feats + 3))  # N * (num_feats + 3)

        # Normalizamos las coordenadas con respecto a las coordenadas máximas de la room actual
        current_points[:, -3] = points[:, 0] / self.room_coord_max[room_idx][0]  # Normalización de X
        current_points[:, -2] = points[:, 1] / self.room_coord_max[room_idx][1]  # Normalización de Y
        current_points[:, -1] = points[:, 2] / self.room_coord_max[room_idx][2]  # Normalización de Z

        # Copiamos las características originales (como color, intensidad, etc.) en las primeras columnas
        current_points[:, 0:self.num_feats] = points

        # Si hay algún proceso de transformación, lo aplicamos
        if self.transform is not None:
            current_points, labels = self.transform(current_points, labels)

        # Devuelve todos los puntos normalizados y sus etiquetas
        if return_index:
            return current_points, labels, self.room_names[room_idx]
        else:
            return current_points, labels


    def __len__(self):
        return len(self.room_idxs)
    

if __name__ == '__main__':
    data_root = '/home/nicolas/repos/dust-filtering/data/blender_areas'
    num_point, test_area, block_size, sample_rate = None, 5, 1.0, 1.0

    point_data = AMTCDataset(split='train', data_root=data_root, num_point=None, test_area=2, block_size=block_size, sample_rate=sample_rate, feats = ['coord', 'intensity'], transform=None, num_classes = 2)
    data, labels, r_idx = point_data.__getitem__(10, return_index = True)
    print(f'Viendo escena {r_idx}...')
    print('labels: ', np.unique(labels, return_counts=True))