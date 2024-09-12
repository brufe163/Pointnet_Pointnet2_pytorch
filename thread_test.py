import open3d as o3d
import numpy as np
import cv2

# Función para generar una nube de puntos aleatoria
def generate_pointcloud():
    num_points = 1000
    points = np.random.rand(num_points, 3) * 100  # Puntos aleatorios en el espacio [0, 100]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

# Función para visualizar la nube de puntos en Open3D
def show_pointcloud():
    pcd = generate_pointcloud()  # Generar la nube de puntos
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='PointCloud Window', width=800, height=600)
    vis.add_geometry(pcd)
    
    # Función para mantener la ventana de Open3D actualizada
    def update_vis():
        while True:
            vis.poll_events()
            vis.update_renderer()

    return vis, update_vis

# Función para mostrar la imagen usando OpenCV
def show_image():
    # Crear una imagen simple con un gradiente
    width, height = 256, 256
    image_data = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generar un gradiente de colores
    for i in range(height):
        for j in range(width):
            image_data[i, j] = [i % 256, j % 256, (i + j) % 256]
    
    # Mostrar la imagen en modo no bloqueante
    cv2.imshow('Image Window', image_data)
    return image_data

# Inicializar ambas visualizaciones
vis, update_vis = show_pointcloud()

# Bucle principal para las visualizaciones
while True:
    # Mostrar la imagen en la ventana de OpenCV
    image_data = show_image()

    # Mantener la ventana de la nube de puntos interactiva
    update_vis()

    # Cerrar cuando se detecte una tecla en OpenCV ('q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerrar las ventanas de manera segura
cv2.destroyAllWindows()
vis.destroy_window()
