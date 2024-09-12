import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import argparse

def compute_metrics(pcd):
    # 1. Bounding Box
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    print(f"Bounding Box Extent (Escala): {bbox_extent}")
    
    # 2. Densidad: Número de puntos por unidad de volumen
    num_points = np.asarray(pcd.points).shape[0]
    volume = np.prod(bbox_extent)
    density = num_points / volume
    print(f"Densidad (puntos por unidad de volumen): {density}")
    
    # 3. Distancia promedio entre puntos vecinos
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    avg_distances = []
    for i in range(num_points):
        [_, idx, dists] = pcd_tree.search_knn_vector_3d(pcd.points[i], 2)
        avg_distances.append(np.sqrt(dists[1]))
    avg_distance = np.mean(avg_distances)
    print(f"Distancia promedio entre puntos vecinos: {avg_distance}")
    
    # 4. Análisis de Componentes Principales (PCA)
    pcd_mean = np.mean(np.asarray(pcd.points), axis=0)
    pcd_centered = np.asarray(pcd.points) - pcd_mean
    H = np.dot(pcd_centered.T, pcd_centered)
    eigenvalues, eigenvectors = np.linalg.eig(H)
    eigenvalues = np.sort(eigenvalues)[::-1]
    print(f"Componentes Principales (PCA): {eigenvalues}")
    
    # 5. Histograma de Distancias al Centroide
    distances_to_centroid = np.linalg.norm(pcd_centered, axis=1)
    plt.hist(distances_to_centroid, bins=30, color='c', alpha=0.75)
    plt.title("Histograma de Distancias al Centroide")
    plt.xlabel("Distancia")
    plt.ylabel("Frecuencia")
    plt.show()

def main(args):
    # Construir las rutas completas a los archivos .npy
    coord_path = os.path.join(args.d, 'coord.npy')
    segment_path = os.path.join(args.d, 'segment.npy')

    

    # Cargar los datos desde los archivos .npy
    coord = np.load(coord_path)  # Formato esperado: (N, 3)
    segment = np.load(segment_path).reshape(-1)  # Formato esperado: (N,)

    num_segment = len(np.unique(segment))
    print(np.unique(segment,return_counts=True))
    colors = np.zeros((len(segment), 3))
    
    # Generar una paleta de colores con tantos colores como clases haya
    colormap = plt.get_cmap("tab10", num_segment)
    
    for i in range(len(segment)):
        colors[i] = colormap(segment[i])[:3]  # Asignar color a cada clase

    # Verificar que coord y segment tengan el mismo número de puntos
    assert coord.shape[0] == segment.shape[0], "El número de coordenadas y segmentos no coincide."

    # Crear una point cloud de Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.points = o3d.utility.Vector3dVector(coord)



    # Visualizar la point cloud
    o3d.visualization.draw_geometries([pcd])


    # Calcular las métricas
    compute_metrics(pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize dataset pointclouds and compute metrics.')
    parser.add_argument('-d', type=str, default='', 
                        help='Data root path [default: None]')
    args = parser.parse_args()
    main(args)

