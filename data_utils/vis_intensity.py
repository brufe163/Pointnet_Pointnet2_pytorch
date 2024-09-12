import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_pointcloud_by_intensity(input_dir):
    # Verificar si los archivos existen
    coord_file = os.path.join(input_dir, 'coord.npy')
    intensity_file = os.path.join(input_dir, 'intensity.npy')
    segment_file = os.path.join(input_dir, 'segment.npy')
    
    if not all([os.path.exists(coord_file), os.path.exists(intensity_file), os.path.exists(segment_file)]):
        raise FileNotFoundError("No se encontraron todos los archivos necesarios (coord.npy, intensity.npy, segment.npy) en la carpeta proporcionada.")
    
    # Cargar los archivos
    coords = np.load(coord_file)  # Cargar coordenadas
    intensities = np.load(intensity_file)
    print(np.shape(intensities))
    intensities = np.load(intensity_file).reshape(-1)  # Cargar intensidades y convertir a 1D si es necesario
    print(np.shape(intensities))
    segments = np.load(segment_file).reshape(-1)  # Cargar segmentaciones y convertir a 1D si es necesario

    # Normalizar intensidades
    intensities_normalized = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))

    # Aplicar un colormap (jet o viridis, por ejemplo)
    cmap = plt.get_cmap('jet')  # Puedes cambiar el mapa de colores a 'viridis' u otro si prefieres
    colors = cmap(intensities_normalized)[:, :3]  # Seleccionar solo los tres primeros canales RGB

    # Crear la nube de puntos con Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualizar la nube de puntos
    o3d.visualization.draw_geometries([pcd], window_name="PointCloud Colored by Intensity",
                                      width=800, height=600)

def plot_intensity_distributions(input_dir):
    # Verificar si los archivos existen
    intensity_file = os.path.join(input_dir, 'intensity.npy')
    segment_file = os.path.join(input_dir, 'segment.npy')
    
    if not all([os.path.exists(intensity_file), os.path.exists(segment_file)]):
        raise FileNotFoundError("No se encontraron los archivos necesarios (intensity.npy, segment.npy) en la carpeta proporcionada.")

    # Cargar los archivos
    intensities = np.load(intensity_file).reshape(-1)  # Cargar intensidades y convertir a 1D si es necesario
    segments = np.load(segment_file).reshape(-1)  # Cargar segmentaciones y convertir a 1D si es necesario

    # Dividir las intensidades en polvo (1) y no polvo (0)
    dust_intensities = intensities[segments == 1]
    no_dust_intensities = intensities[segments == 0]

    # Crear histogramas
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Histograma para los puntos de polvo
    ax[0].hist(dust_intensities, bins=20, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax[0].set_title('Dust points')
    ax[0].set_xlabel('Intensity value')
    ax[0].set_ylabel('Percentage of dust data')

    # Histograma para los puntos de no polvo
    ax[1].hist(no_dust_intensities, bins=20, density=True, alpha=0.7, color='green', edgecolor='black')
    ax[1].set_title('Non-dust points')
    ax[1].set_xlabel('Intensity value')
    ax[1].set_ylabel('Percentage of non-dust data')

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize point clouds colored by intensity and plot intensity distributions.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing coord.npy, intensity.npy, and segment.npy files.')
    parser.add_argument('--plot-distributions', action='store_true', help='Plot intensity distributions.')

    args = parser.parse_args()

    # Visualizar la nube de puntos coloreada por intensidad
    visualize_pointcloud_by_intensity(args.input_dir)

    # Si se especifica el argumento, graficar las distribuciones de intensidad
    if args.plot_distributions:
        plot_intensity_distributions(args.input_dir)
