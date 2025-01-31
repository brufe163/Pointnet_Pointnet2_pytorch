import open3d as o3d
import numpy as np
import argparse
import os

def viz(args):

    coords_path = os.path.join(args.data_path, 'coord.npy')
    point_cloud = np.load(coords_path)
    # Crear la nube de puntos en Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    try:
        segment_path = os.path.join(args.data_path, 'segment.npy')
        classes = np.load(segment_path)
        # Mapa de colores (colores RGB) para cada clase
        color_map = {
            0: [1, 0, 0],  # Rojo
            1: [0, 1, 0],  # Verde
        }
        # Verificar que las clases sean enteros
        if not np.issubdtype(classes.dtype, np.integer):
            raise ValueError("Las clases deben ser enteros.")

        colors = np.array([color_map[int(class_label)] for class_label in classes])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    except:
        pass

    # Visualizar la nube de puntos
    o3d.visualization.draw_geometries([pcd], window_name="Nube de puntos por clase")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from images.')
    parser.add_argument('--data_path', type=str,
                        help='Path to the coord and segmentation .npy file.')
    args = parser.parse_args()
    viz(args)