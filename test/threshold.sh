#!/bin/bash
# Uso: ./test/threshold.sh <global_checkpoint_folder>
# Ejemplo: ./test/threshold.sh log/sem_seg_amtc/preentrega/experimento10/0.005/

if [ "$#" -ne 1 ]; then
    echo "Uso: $0 <global_checkpoint_folder>"
    exit 1
fi

global_folder="$1"

# Determinar el directorio donde se encuentra este script (por ejemplo, test/)
script_dir="$(dirname "$0")"
# Definir el nombre del script de Python ubicado en el mismo directorio (test/)
python_script="${script_dir}/test_threshold.py"

# Recorre todos los archivos .pth en el directorio global (incluyendo subdirectorios)
find "$global_folder" -type f -name "*.pth" | while read ckpt_file; do
    # Directorio del checkpoint (donde se encuentra el archivo .pth)
    ckpt_dir=$(dirname "$ckpt_file")
    
    # Se asume que info.txt se encuentra en training_results/ relativo al directorio del checkpoint
    info_file="${ckpt_dir}/training_results/info.txt"

    # Valores por defecto
    model="pointnet2_sem_seg_amtc"
    data_root="default_data_root"
    num_classes=2
    feat_list="coord intensity"
    corrected_flag=""
    hyperset_flag=""
    val_areas="1 2 3"
    test_areas="4 5"

    if [ -f "$info_file" ]; then
        # Extraer parámetros del archivo info.txt
        model=$(grep "^model:" "$info_file" | cut -d':' -f2- | xargs)
        data_root=$(grep "^data_dir:" "$info_file" | cut -d':' -f2- | xargs)
        num_classes=$(grep "^nclasses:" "$info_file" | cut -d':' -f2- | xargs)
        feat_list=$(grep "^feat_list:" "$info_file" | cut -d':' -f2- | xargs)
        # Remover corchetes, comillas y reemplazar comas por espacios en feat_list
        feat_list=$(echo "$feat_list" | sed "s/[][]//g" | sed "s/'//g" | sed "s/,/ /g")

        corrected=$(grep "^corrected:" "$info_file" | cut -d':' -f2- | xargs)
        if [ "$corrected" = "True" ]; then
            corrected_flag="--corrected"
        else
            corrected_flag=""
        fi

        # Si data_root termina en "data/experimentos", se activa el flag hyperset
        if [[ "$data_root" == */data/experimentos ]]; then
            hyperset_flag="--hyperset"
        else
            hyperset_flag=""
        fi

        # Extraer sets (ej: [[1,2,3],[4,5,6],[7,8]])
        sets_line=$(grep "^sets:" "$info_file" | cut -d':' -f2- | xargs)
        # val_areas = segunda sublista
        val_areas=$(echo "$sets_line" | sed -E 's/^\[\[[^]]+\],\[([^]]+)\].*$/\1/')
        val_areas=$(echo "$val_areas" | sed "s/,/ /g")
        # test_areas = tercera sublista
        test_areas=$(echo "$sets_line" | sed -E 's/^\[\[[^]]+\],\[[^]]+\],\[([^]]+)\]\]$/\1/')
        test_areas=$(echo "$test_areas" | sed "s/,/ /g")
    else
        echo "No se encontró $info_file, usando valores por defecto."
    fi

    # Definir output_dir como subcarpeta 'threshold_results' dentro del directorio del checkpoint
    output_dir="${ckpt_dir}/threshold_results"
    mkdir -p "$output_dir"

    echo "Procesando checkpoint: $ckpt_file"
    echo "Parámetros:"
    echo "  model=$model"
    echo "  data_root=$data_root"
    echo "  num_classes=$num_classes"
    echo "  feat_list=$feat_list"
    echo "  corrected_flag=$corrected_flag"
    echo "  val_areas=$val_areas"
    echo "  test_areas=$test_areas"
    echo "  hyperset_flag=$hyperset_flag"

    # Ejecutar el script de Python pasando el archivo de checkpoint
    python "$python_script" \
        --checkpoint_dir "$ckpt_file" \
        --model "$model" \
        --data_root "$data_root" \
        --output_dir "$output_dir" \
        --feat $feat_list \
        --num_classes "$num_classes" \
        --val_areas $val_areas \
        --test_areas $test_areas \
        $corrected_flag \
        $hyperset_flag

done
