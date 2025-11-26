import os
import numpy as np
import json

def count_points_in_area(Area_folder):
    total_points = 0
    # Recorremos de manera recursiva la carpeta
    for root, dirs, files in os.walk(Area_folder):
        if "coord.npy" in files:
            file_path = os.path.join(root, "coord.npy")
            try:
                coords = np.load(file_path)
                # Asumimos que los puntos están en un arreglo de forma (N, 3)
                if coords.ndim == 2 and coords.shape[1] == 3:
                    num_points = coords.shape[0]
                # Caso alternativo: el archivo contiene un arreglo unidimensional (o estructura distinta)
                else:
                    num_points = len(coords)
                total_points += num_points
                #print(f"Procesado {file_path}: {num_points} puntos")
            except Exception as e:
                print(f"Error al cargar {file_path}: {e}")
    return total_points


def count_classes_in_area(Area_folder):
    # Inicializamos un diccionario para acumular los conteos de las clases 0 y 1
    class_counts = {0: 0, 1: 0}
    frame_point_counts = []  # Lista para almacenar el número de puntos por frame
    
    # Recorremos recursivamente la carpeta
    for root, dirs, files in os.walk(Area_folder):
        if "segment.npy" in files:
            file_path = os.path.join(root, "segment.npy")
            try:
                segments = np.load(file_path)
                frame_point_counts.append(len(segments))  # Agregar el número de puntos de este frame
                
                # Se cuentan las ocurrencias de cada clase usando np.unique
                unique, counts = np.unique(segments, return_counts=True)
                counts_dict = dict(zip(unique, counts))
                # Se actualiza el conteo solo para las clases 0 y 1
                for cls in [0, 1]:
                    class_counts[cls] += counts_dict.get(cls, 0)
                #print(f"Procesado {file_path}: {counts_dict}")
            except Exception as e:
                print(f"Error al cargar {file_path}: {e}")
    
    return class_counts, frame_point_counts

def calculate_proportions(dust_counts, nondust_counts):
    total_dust = sum(dust_counts)
    total_nondust = sum(nondust_counts)
    total = total_dust + total_nondust
    if total > 0:
        return (total_dust / total, total_nondust / total)
    else:
        return (0, 0)

def calculate_frame_statistics(frame_point_counts_list):
    """Calcula estadísticas de puntos por frame para todas las áreas de un dataset"""
    all_frame_counts = []
    for frame_counts in frame_point_counts_list:
        all_frame_counts.extend(frame_counts)
    
    if all_frame_counts:
        return {
            'min': min(all_frame_counts),
            'max': max(all_frame_counts),
            'mean': np.mean(all_frame_counts),
            'total_frames': len(all_frame_counts)
        }
    else:
        return {'min': 0, 'max': 0, 'mean': 0, 'total_frames': 0}

def print_dataset_info(dataset_name, dust_counts, nondust_counts, frame_point_counts_list):
    """Imprime información completa de un dataset"""
    total_dust = sum(dust_counts)
    total_nondust = sum(nondust_counts)
    total_points = total_dust + total_nondust
    proportions = calculate_proportions(dust_counts, nondust_counts)
    frame_stats = calculate_frame_statistics(frame_point_counts_list)
    
    print(f"\n=== {dataset_name.upper()} ===")
    print(f"Puntos por clase:")
    print(f"  - Dust (clase 0): {total_dust:,} puntos")
    print(f"  - Non-dust (clase 1): {total_nondust:,} puntos")
    print(f"  - Total: {total_points:,} puntos")
    print(f"Proporciones: Dust={proportions[0]:.3f}, Non-dust={proportions[1]:.3f}")
    print(f"Estadísticas de frames:")
    print(f"  - Total frames: {frame_stats['total_frames']}")
    print(f"  - Puntos por frame - Min: {frame_stats['min']}, Max: {frame_stats['max']}, Promedio: {frame_stats['mean']:.1f}")
    print(f"  - Áreas procesadas: {len(dust_counts)}")

def load_hypersets(json_path):
    """Carga las particiones desde el archivo hypersets.json"""
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_partition_stats(areas_list, base_path):
    """Calcula estadísticas para una partición específica (train, val, test)"""
    total_dust = 0
    total_nondust = 0
    total_frames = 0
    all_frame_counts = []
    
    for area_path in areas_list:
        full_path = os.path.join(base_path, area_path)
        if os.path.exists(full_path):
            class_counts, frame_counts = count_classes_in_area(full_path)
            total_dust += class_counts[0]
            total_nondust += class_counts[1]
            all_frame_counts.extend(frame_counts)
            total_frames += len(frame_counts)
    
    total_points = total_dust + total_nondust
    proportions = calculate_proportions([total_dust], [total_nondust])
    
    frame_stats = {
        'min': min(all_frame_counts) if all_frame_counts else 0,
        'max': max(all_frame_counts) if all_frame_counts else 0,
        'mean': np.mean(all_frame_counts) if all_frame_counts else 0,
        'total_frames': total_frames
    }
    
    return {
        'dust': total_dust,
        'nondust': total_nondust,
        'total': total_points,
        'proportions': proportions,
        'frame_stats': frame_stats,
        'areas_count': len(areas_list)
    }

def print_partition_info(partition_name, stats):
    """Imprime información de una partición específica"""
    print(f"\n=== PARTICIÓN {partition_name.upper()} ===")
    print(f"Puntos por clase:")
    print(f"  - Dust (clase 0): {stats['dust']:,} puntos")
    print(f"  - Non-dust (clase 1): {stats['nondust']:,} puntos")
    print(f"  - Total: {stats['total']:,} puntos")
    print(f"Proporciones: Dust={stats['proportions'][0]:.3f}, Non-dust={stats['proportions'][1]:.3f}")
    print(f"Estadísticas de frames:")
    print(f"  - Total frames: {stats['frame_stats']['total_frames']}")
    print(f"  - Puntos por frame - Min: {stats['frame_stats']['min']}, Max: {stats['frame_stats']['max']}, Promedio: {stats['frame_stats']['mean']:.1f}")
    print(f"  - Áreas incluidas: {stats['areas_count']}")

def calculate_dataset_partition_stats(dataset_name, areas_in_partition, base_data_path):
    """Calcula estadísticas para las áreas de un dataset específico en una partición"""
    total_dust = 0
    total_nondust = 0
    areas_found = 0
    
    for area_path in areas_in_partition:
        if area_path.startswith(f"{dataset_name}/"):
            full_path = os.path.join(base_data_path, area_path)
            if os.path.exists(full_path):
                class_counts, _ = count_classes_in_area(full_path)
                total_dust += class_counts[0]
                total_nondust += class_counts[1]
                areas_found += 1
    
    total_points = total_dust + total_nondust
    if total_points > 0:
        dust_prop = total_dust / total_points
        nondust_prop = total_nondust / total_points
    else:
        dust_prop = nondust_prop = 0
    
    return {
        'dust': total_dust,
        'nondust': total_nondust,
        'total': total_points,
        'dust_prop': dust_prop,
        'nondust_prop': nondust_prop,
        'areas_count': areas_found
    }

def analyze_datasets_by_partitions(hypersets_path, base_data_path):
    """Analiza las proporciones de clases por dataset y partición"""
    hypersets = load_hypersets(hypersets_path)
    
    print("\n" + "="*80)
    print("ANÁLISIS POR CONJUNTO DE DATOS Y PARTICIÓN")
    print("="*80)
    
    # Definir los nombres de los datasets
    datasets = ['interior1', 'interior2', 'exterior1', 'exterior2', 'peldehue', 'caren']
    partitions = ['train_set', 'val_set', 'test_set']
    partition_names = ['TRAIN', 'VAL', 'TEST']
    
    for dataset in datasets:
        print(f"\n=== {dataset.upper()} ===")
        dataset_totals = {'dust': 0, 'nondust': 0, 'total': 0}
        
        for i, partition in enumerate(partitions):
            stats = calculate_dataset_partition_stats(dataset, hypersets[partition], base_data_path)
            
            # Acumular totales del dataset
            dataset_totals['dust'] += stats['dust']
            dataset_totals['nondust'] += stats['nondust']
            dataset_totals['total'] += stats['total']
            
            if stats['areas_count'] > 0:
                print(f"  {partition_names[i]}: Dust={stats['dust']:,} ({stats['dust_prop']:.3f}), "
                      f"Non-dust={stats['nondust']:,} ({stats['nondust_prop']:.3f}), "
                      f"Total={stats['total']:,}, Áreas={stats['areas_count']}")
            else:
                print(f"  {partition_names[i]}: Sin áreas en esta partición")
        
        # Mostrar totales del dataset
        if dataset_totals['total'] > 0:
            dust_prop_total = dataset_totals['dust'] / dataset_totals['total']
            nondust_prop_total = dataset_totals['nondust'] / dataset_totals['total']
            print(f"  TOTAL: Dust={dataset_totals['dust']:,} ({dust_prop_total:.3f}), "
                  f"Non-dust={dataset_totals['nondust']:,} ({nondust_prop_total:.3f}), "
                  f"Total={dataset_totals['total']:,}")

def analyze_partitions(hypersets_path, base_data_path):
    """Analiza las proporciones de clases para cada partición"""
    hypersets = load_hypersets(hypersets_path)
    
    print("\n" + "="*60)
    print("ANÁLISIS DE PARTICIONES (TRAIN/VAL/TEST)")
    print("="*60)
    
    # Calcular estadísticas para cada partición
    train_stats = calculate_partition_stats(hypersets['train_set'], base_data_path)
    val_stats = calculate_partition_stats(hypersets['val_set'], base_data_path)
    test_stats = calculate_partition_stats(hypersets['test_set'], base_data_path)
    
    # Imprimir resultados
    print_partition_info("TRAIN", train_stats)
    print_partition_info("VAL", val_stats)
    print_partition_info("TEST", test_stats)
    
    # Resumen de particiones
    total_all_partitions = train_stats['total'] + val_stats['total'] + test_stats['total']
    print(f"\n=== RESUMEN DE PARTICIONES ===")
    print(f"Train: {train_stats['total']:,} puntos ({train_stats['total']/total_all_partitions:.3f})")
    print(f"Val: {val_stats['total']:,} puntos ({val_stats['total']/total_all_partitions:.3f})")
    print(f"Test: {test_stats['total']:,} puntos ({test_stats['total']/total_all_partitions:.3f})")
    print(f"Total: {total_all_partitions:,} puntos")
    
    return train_stats, val_stats, test_stats

if __name__ == "__main__":
    # Definición de las rutas de cada área
    interior1_dust, interior1_nondust, interior1_frame_counts = [], [], []
    for i in range(10):
        area = f"/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/interior1/Area_{i+1}"
        class_counts, frame_counts = count_classes_in_area(area)
        interior1_dust.append(class_counts[0])
        interior1_nondust.append(class_counts[1])
        interior1_frame_counts.append(frame_counts)

    interior2_dust, interior2_nondust, interior2_frame_counts = [], [], []
    for i in range(12):
        area = f"/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/interior2/Area_{i+1}"
        class_counts, frame_counts = count_classes_in_area(area)
        interior2_dust.append(class_counts[0])
        interior2_nondust.append(class_counts[1])
        interior2_frame_counts.append(frame_counts)
    
    exterior1_dust, exterior1_nondust, exterior1_frame_counts = [], [], []
    for i in range(10):
        area = f"/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/exterior1/Area_{i+1}"
        class_counts, frame_counts = count_classes_in_area(area)
        exterior1_dust.append(class_counts[0])
        exterior1_nondust.append(class_counts[1])
        exterior1_frame_counts.append(frame_counts)

    exterior2_dust, exterior2_nondust, exterior2_frame_counts = [], [], []
    for i in range(13):
        area = f"/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/exterior2/Area_{i+1}"
        class_counts, frame_counts = count_classes_in_area(area)
        exterior2_dust.append(class_counts[0])
        exterior2_nondust.append(class_counts[1])
        exterior2_frame_counts.append(frame_counts)

    peldehue_dust, peldehue_nondust, peldehue_frame_counts = [], [], []
    for i in range(2):
        area = f"/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/peldehue/Area_{i+1}"
        class_counts, frame_counts = count_classes_in_area(area)
        peldehue_dust.append(class_counts[0])
        peldehue_nondust.append(class_counts[1])
        peldehue_frame_counts.append(frame_counts)
    
    caren_dust, caren_nondust, caren_frame_counts = [], [], []
    for i in range(2,16):
        area = f"/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/caren/Area_{i+1}"
        class_counts, frame_counts = count_classes_in_area(area)
        caren_dust.append(class_counts[0])
        caren_nondust.append(class_counts[1])
        caren_frame_counts.append(frame_counts)
    
    # Imprimir información detallada de cada dataset
    print_dataset_info("Interior1", interior1_dust, interior1_nondust, interior1_frame_counts)
    print_dataset_info("Interior2", interior2_dust, interior2_nondust, interior2_frame_counts)
    print_dataset_info("Exterior1", exterior1_dust, exterior1_nondust, exterior1_frame_counts)
    print_dataset_info("Exterior2", exterior2_dust, exterior2_nondust, exterior2_frame_counts)
    print_dataset_info("Peldehue", peldehue_dust, peldehue_nondust, peldehue_frame_counts)
    print_dataset_info("Caren", caren_dust, caren_nondust, caren_frame_counts)
    
    # Resumen general
    all_datasets = [
        ("Interior1", interior1_dust, interior1_nondust, interior1_frame_counts),
        ("Interior2", interior2_dust, interior2_nondust, interior2_frame_counts),
        ("Exterior1", exterior1_dust, exterior1_nondust, exterior1_frame_counts),
        ("Exterior2", exterior2_dust, exterior2_nondust, exterior2_frame_counts),
        ("Peldehue", peldehue_dust, peldehue_nondust, peldehue_frame_counts),
        ("Caren", caren_dust, caren_nondust, caren_frame_counts)
    ]
    
    total_dust_all = sum([sum(dust) for _, dust, _, _ in all_datasets])
    total_nondust_all = sum([sum(nondust) for _, _, nondust, _ in all_datasets])
    total_points_all = total_dust_all + total_nondust_all
    total_frames_all = sum([calculate_frame_statistics(frames)['total_frames'] for _, _, _, frames in all_datasets])
    
    print(f"\n=== RESUMEN GENERAL ===")
    print(f"Total de puntos: {total_points_all:,}")
    print(f"Total dust: {total_dust_all:,} ({total_dust_all/total_points_all:.3f})")
    print(f"Total non-dust: {total_nondust_all:,} ({total_nondust_all/total_points_all:.3f})")
    print(f"Total frames: {total_frames_all}")
    print(f"Total datasets: {len(all_datasets)}")
    
    # Análisis adicional de particiones basado en hypersets.json
    hypersets_path = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/hypersets.json"
    base_data_path = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos"
    
    if os.path.exists(hypersets_path):
        # Análisis general de particiones
        analyze_partitions(hypersets_path, base_data_path)
        
        # Análisis detallado por conjunto de datos y partición
        analyze_datasets_by_partitions(hypersets_path, base_data_path)
    else:
        print(f"\nAdvertencia: No se encontró el archivo {hypersets_path}")
        print("No se puede realizar el análisis de particiones.")


    
    



    
    # # Se acumulan los resultados de todas las áreas
    # total_counts = {0: 0, 1: 0}
    # areas = [area1, area2, area3, area4, area5, area6, area7, area8, area9, area10, area11]
    # for area in areas:
    #     counts = count_classes_in_area(area)
    #     total_counts[0] += counts[0]
    #     total_counts[1] += counts[1]
    
    # print("\nConteo total de clases:")
    # print(f"Clase 0: {total_counts[0]}")
    # print(f"Clase 1: {total_counts[1]}")


# if __name__ == "__main__":
#     area1 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/caren/Area_6"  # Ruta a la carpeta principal
#     area2 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/caren/Area_7"  # Ruta a la carpeta principal
#     area3 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/caren/Area_12"  # Ruta a la carpeta principal
#     area4 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/caren/Area_14"  # Ruta a la carpeta principal
#     area5 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/exterior1/Area_10"  # Ruta a la carpeta principal
#     area6 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/exterior2/Area_12"  # Ruta a la carpeta principal
#     area7 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/exterior2/Area_13"  # Ruta a la carpeta principal
#     area8 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/interior1/Area_10"  # Ruta a la carpeta principal
#     area9 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/interior2/Area_11"  # Ruta a la carpeta principal
#     area10 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/interior2/Area_12"  # Ruta a la carpeta principal
#     area11 = "/home/bruno/repos/tesis/Pointnet_Pointnet2_pytorch/data/experimentos/peldehue/Area_4"  # Ruta a la carpeta principal

#     total = count_points_in_area(area1) + count_points_in_area(area2) + count_points_in_area(area3) + count_points_in_area(area4) + count_points_in_area(area5) + count_points_in_area(area6) + count_points_in_area(area7) + count_points_in_area(area8) + count_points_in_area(area9) + count_points_in_area(area10) + count_points_in_area(area11)
#     print(f"\nCantidad total de puntos: {total}")
