import os
import argparse
import random
import re
import numpy as np 
def process_npy_files(area_dir):
    for npy_file in os.listdir(area_dir):
        npy_path = os.path.join(area_dir, npy_file)
        
        # Carga el archivo .npy
        data = np.load(npy_path)
        
        # Aquí puedes procesar los datos cargados
        print(f"Procesando archivo: {npy_path}")
        print(f"Datos cargados: {data.shape}")
        print(f"Ejemplo de contenidos de {npy_file}:")
        random_integers = [random.randint(1, 4000) for _ in range(5)]
        for i in random_integers:

            print(data[i])
        if npy_file == 'segment.npy':
            print(' Cantidad de elementos por cada clase:')
            labels, counts = np.unique(data, return_counts = True)
            for label, i in enumerate(labels):
                print(f'    Class {label}: {counts[i]}')
            ratio = counts[0]/counts[1]
            print(f'    Ratio dust/otros: {ratio}')


def main(args):
    areas_list = []
    pattern = re.compile(r'^Area_\d+$')
    
    # Recorre todos los elementos en el directorio base
    for item in os.listdir(args.data_dir):
        item_path = os.path.join(args.data_dir, item)
        
        if os.path.isdir(item_path) and pattern.match(item):
            areas_list.append(item_path)

    random_area = random.choice(areas_list)
    #random_area = 'Area_8'
    print(random_area)
    rooms_list = os.listdir(os.path.join(args.data_dir,random_area))
    random_room = random.choice(rooms_list)
    #random_room = 'amtc_166'
    print(random_room)
    full_dir = os.path.join(args.data_dir,random_area, random_room)
    print(full_dir)
    process_npy_files(full_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data tester for datasets.')
    parser.add_argument('--data-dir', type=str, default='/home/nicolas/repos/dust-filtering/data/s3dis',
                        help='Directory where the dataset is located.')
    args = parser.parse_args()
    main(args)

