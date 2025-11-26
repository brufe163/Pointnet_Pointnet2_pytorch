import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import open3d as o3d

###########################################
# LÓGICA PARA DIFF
###########################################

def compute_magnitude(current_points, prev_points, device, batch_size=1000, min_distance_threshold=0.01):
    prev_coords = torch.tensor(prev_points, device=device, dtype=torch.float32)
    current_coords = torch.tensor(current_points, device=device, dtype=torch.float32)

    num_points = current_coords.shape[0]
    magnitudes = torch.zeros((num_points,), device=device)

    for start in range(0, num_points, batch_size):
        end = start + batch_size
        batch_current = current_coords[start:end]
        distances = torch.cdist(batch_current.unsqueeze(0), prev_coords.unsqueeze(0)).squeeze(0)
        min_distances, indices = distances.min(dim=1)

        valid_mask = min_distances > min_distance_threshold
        if valid_mask.sum() == 0:
            continue

        closest_prev = prev_coords[indices[valid_mask]]
        batch_diff = batch_current[valid_mask] - closest_prev
        batch_mag = torch.norm(batch_diff, dim=1)

        magnitudes[start:end][valid_mask] = batch_mag

    return magnitudes.cpu().numpy()

def compute_differences(current_points, prev_points, device, batch_size=1000, min_distance_threshold=0.01):
    prev_coords = torch.tensor(prev_points, device=device, dtype=torch.float32)
    current_coords = torch.tensor(current_points, device=device, dtype=torch.float32)

    num_points = current_coords.shape[0]
    magnitudes = torch.zeros((num_points,), device=device)
    differences = torch.zeros((num_points, 3), device=device)

    for start in range(0, num_points, batch_size):
        end = start + batch_size
        batch_current = current_coords[start:end]
        distances = torch.cdist(batch_current.unsqueeze(0), prev_coords.unsqueeze(0)).squeeze(0)
        min_distances, indices = distances.min(dim=1)

        valid_mask = min_distances > min_distance_threshold
        if valid_mask.sum() == 0:
            continue

        closest_prev = prev_coords[indices[valid_mask]]
        batch_diff = batch_current[valid_mask] - closest_prev
        differences[start:end][valid_mask] = batch_diff

    return differences.cpu().numpy()


###########################################
# LÓGICA PARA TVAI
###########################################

class TemporalVariationInterpolation(nn.Module):
    def __init__(self, alpha=0.5, beta=2.0, feature_dim=1, hidden_dim=128):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def compute_weights(self, distances):
        weights = self.alpha - torch.clamp(distances, max=self.alpha)
        weights *= self.beta
        weights = F.softmax(weights, dim=-1)
        return weights

    def interpolate(self, current_features, previous_features, distances, k_indices, device):
        with torch.no_grad():
            current_features = torch.tensor(current_features, dtype=torch.float32, device=device)
            previous_features = torch.tensor(previous_features, dtype=torch.float32, device=device)
            distances = torch.tensor(distances, dtype=torch.float32, device=device)
            k_indices = torch.tensor(k_indices, dtype=torch.long, device=device)

            neighbors_features = previous_features[k_indices]  # (N, k, F)
            target_features = current_features.unsqueeze(1).expand_as(neighbors_features)  # (N, k, F)

            variation_features = torch.cat([neighbors_features, target_features - neighbors_features], dim=-1)
            variation_features = self.mlp(variation_features)

            weights = self.compute_weights(distances)
            interpolated = torch.sum(weights.unsqueeze(-1) * variation_features, dim=1)

        return interpolated.cpu().numpy()

def compute_distances_and_knn(current_points, prev_points, device, batch_size=1000, k=8):
    prev_coords = torch.tensor(prev_points, device=device, dtype=torch.float32)
    current_coords = torch.tensor(current_points, device=device, dtype=torch.float32)

    knn_distances = []
    knn_indices = []
    num_points = current_coords.shape[0]

    for start in range(0, num_points, batch_size):
        end = start + batch_size
        batch_current = current_coords[start:end]
        distances = torch.cdist(batch_current.unsqueeze(0), prev_coords.unsqueeze(0)).squeeze(0)
        batch_knn_dist, batch_knn_idx = torch.topk(distances, k, largest=False, dim=-1)

        knn_distances.append(batch_knn_dist)
        knn_indices.append(batch_knn_idx)

    knn_distances = torch.cat(knn_distances, dim=0)
    knn_indices = torch.cat(knn_indices, dim=0)
    return knn_distances, knn_indices


###########################################
# DATASET PARA CÁLCULO ON-THE-FLY
# QUE GUARDA EN MEMORIA EL FRAME PREVIO
###########################################

class InferenceDataset(Dataset):
    def __init__(self,
                 data_path,
                 feats=['coord', 'intensity', 'diff', 'diff_vectors', 'interp'],
                 voxel_size=None,
                 transform=None,
                 device='cpu',
                 alpha=0.5,
                 beta=2.0,
                 k=8,
                 hidden_dim=128,
                 min_dist_diff=0.01):
        super().__init__()
        self.data_path = data_path
        self.feats = feats
        self.voxel_size = voxel_size
        self.transform = transform
        self.device = torch.device(device)
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.hidden_dim = hidden_dim
        self.min_dist_diff = min_dist_diff

        # Ver si necesitamos almacenar el frame previo (solo si diff, diff_vectors o interp están en feats)
        self.store_prev_frame = any(x in feats for x in ['diff', 'diff_vectors', 'interp'])
        # Variables para almacenar el frame anterior en RAM
        self.prev_coords = None
        self.prev_intensity = None

        # MLP para TVAI si 'interp' está en feats
        self.tvai_model = None
        if 'interp' in self.feats:
            feature_dim = 1  # Asumiendo que solo se interpola intensidad
            self.tvai_model = TemporalVariationInterpolation(
                alpha=self.alpha,
                beta=self.beta,
                feature_dim=feature_dim,
                hidden_dim=self.hidden_dim
            ).to(self.device)

        # Listar subcarpetas
        self.subfolders = sorted([
            os.path.join(self.data_path, d) for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d)) and d.startswith('amtc_')
        ])

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, idx):
        current_folder = self.subfolders[idx] 

        # 1) Cargar coord
        coord_path = os.path.join(current_folder, "coord.npy")
        if not os.path.exists(coord_path):
            raise FileNotFoundError(f"No existe coord.npy en {current_folder}.")
        coords = np.load(coord_path)

        # 2) Cargar intensity si lo pides en feats
        intensity = None
        if 'intensity' in self.feats or 'interp' in self.feats:
            intensity_path = os.path.join(current_folder, "intensity.npy")
            if os.path.exists(intensity_path):
                intensity = np.load(intensity_path).reshape(-1, 1)

        # 3) Otras features (segment, odom, etc.) si las tuvieras
        # segment = ...
        # odom = ...

        # 4) Construir array base
        loaded_features = []
        if 'coord' in self.feats:
            loaded_features.append(coords)
        if ('intensity' in self.feats and intensity is not None) or 'interp' in self.feats:
            loaded_features.append(intensity)

        # 5) diff y diff_vectors
        diff = None
        diff_vectors = None
        if 'diff' in self.feats:
            if idx == 0:
                # Primer frame => 0
                diff = np.zeros((coords.shape[0], 1))
            else:
                # Si tenemos self.prev_coords en RAM, la usamos
                if self.prev_coords is not None:
                    mag= compute_magnitude(
                        current_points=coords,
                        prev_points=self.prev_coords,
                        device=self.device,
                        min_distance_threshold=self.min_dist_diff
                    )
                    diff = mag.reshape(-1, 1)
                else:
                    # Si por algún motivo es None, fallback a 0
                    diff = np.zeros((coords.shape[0], 1))
            loaded_features.append(diff)
        if 'diff_vectors' in self.feats:
            if idx == 0:
                # Primer frame => 0
                diff_vectors = np.zeros((coords.shape[0], 3))
            else:
                # Si tenemos self.prev_coords en RAM, la usamos
                if self.prev_coords is not None:
                    vec= compute_differences(
                        current_points=coords,
                        prev_points=self.prev_coords,
                        device=self.device,
                        min_distance_threshold=self.min_dist_diff
                    )
                    diff_vectors = vec
                else:
                    # Si por algún motivo es None, fallback a 0
                    diff_vectors = np.zeros((coords.shape[0], 1))
            loaded_features.append(diff_vectors)
        # 6) interp
        interp = None
        if 'interp' in self.feats and self.tvai_model is not None:
            #print(intensity)
            if idx == 0:
                # Primer frame => 0
                interp = np.zeros((coords.shape[0], 1))
                
            else:
                # print(f"self.prev_coords: {self.prev_coords}")
                # print(f"intensity: {intensity}")
                # print(f"self.prev_intensity: {self.prev_intensity}")
                # print(self.store_prev_frame)
                # if idx == 10:
                #     hola
                if (self.prev_coords is not None and
                    intensity is not None and
                    self.prev_intensity is not None):

                    # KNN y TVAI
                    knn_dist, knn_idx = compute_distances_and_knn(
                        current_points=coords,
                        prev_points=self.prev_coords,
                        device=self.device,
                        k=self.k
                    )
                    interp_np = self.tvai_model.interpolate(
                        current_features=intensity,
                        previous_features=self.prev_intensity,
                        distances=knn_dist.cpu().numpy(),
                        k_indices=knn_idx.cpu().numpy(),
                        device=self.device
                    )
                    interp = interp_np
                else:
                    # No hay con qué interpolar => 0
                    interp = np.zeros((coords.shape[0], 1))
            loaded_features.append(interp)

        # Concatenamos todo
        if loaded_features:
            current_points = np.concatenate(loaded_features, axis=1)
        else:
            current_points = coords

        # 7) voxel downsample si procede
        if self.voxel_size is not None and self.voxel_size > 0:
            current_points = self._voxel_downsample(current_points)

        # 8) transform (opcional)
        # if self.transform: ...

        # **Almacenar en memoria** coord e intensidad para el siguiente frame
        # sólo si store_prev_frame es True
        if self.store_prev_frame:
            self.prev_coords = coords  # guardamos el frame actual
            if 'intensity' in self.feats or 'interp' in self.feats:
                self.prev_intensity = intensity
            else:
                self.prev_intensity = None

        # Retornamos (current_points, None) o lo que quieras
        return current_points, None

    def _voxel_downsample(self, points):
        coords = points[:, :3]
        extra_feats = points[:, 3:]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        voxel_coords = np.asarray(pcd_down.points)

        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        _, idx = tree.query(voxel_coords)
        voxel_feats = extra_feats[idx]
        voxel_points = np.concatenate([voxel_coords, voxel_feats], axis=1)
        return voxel_points

def process_inference(data_path, feats, voxel_size=0.5, device='cuda', alpha=0.5, beta=2.0, k=8, hidden_dim=128, min_dist_diff=0.01):
        dataset = InferenceDataset(
            data_path=data_path,
            feats=feats,
            voxel_size=voxel_size,
            device=device,
            alpha=alpha,
            beta=beta,
            k=k,
            hidden_dim=hidden_dim,
            min_dist_diff=min_dist_diff
        )

        total_start = time.time()
        frame_times = []

        for i in tqdm(range(len(dataset)), desc="Procesando frames"):
            start_frame = time.time()
            points, _ = dataset[i]
            # Aquí tu lógica de inferencia con points
            end_frame = time.time()
            frame_times.append(end_frame - start_frame)

        total_end = time.time()
        total_time = total_end - total_start
        avg_time = sum(frame_times) / len(frame_times) if frame_times else 0

        print(f"\nTiempo total: {total_time:.4f} s")
        print(f"Tiempo promedio por frame: {avg_time:.4f} s")
        print(f"Frames procesados: {len(dataset)}")

#########################
# EJEMPLO DE USO
#########################
if __name__ == "__main__":
    import time
    from tqdm import tqdm

    # Lista de áreas y sus rutas
    areas = {
        "Peldehue_Area_1": "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/peldehue/Area_1",
        "Caren_Area_3": "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/caren/Area_3",
        "Interior1_Area_1": "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/interior1/Area_1",
        "Interior2_Area_1": "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/interior2/Area_1",
        "Exterior1_Area_1": "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/exterior1/Area_1",
        "Exterior2_Area_1": "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/exterior2/Area_1",
    }

    # Lista de combinaciones de características
    feature_combinations = [
        # ['coord', 'intensity'],
        # ['coord', 'diff'],
        # ['coord', 'diff_vectors'],
        ['coord', 'interp'],
        # ['coord', 'intensity', 'diff'],
        # ['coord', 'intensity', 'diff_vectors'],
        # ['coord', 'intensity', 'interp']
    ]

    # Diccionario para almacenar los tiempos
    report = {}

    # Procesar cada área y combinación de características
    for area_name, data_path in areas.items():
        report[area_name] = {}
        for features in feature_combinations:
            print(f"Procesando área: {area_name}, características: {features}")
            start_time = time.time()
            process_inference(data_path, features)
            end_time = time.time()
            elapsed_time = end_time - start_time
            report[area_name][tuple(features)] = elapsed_time

    # Mostrar informe final
    print("\n==================== INFORME FINAL ====================")
    for area_name, feature_times in report.items():
        print(f"\nÁrea: {area_name}")
        for features, elapsed_time in feature_times.items():
            print(f"  Características: {features} -> Tiempo: {elapsed_time:.4f} s")


