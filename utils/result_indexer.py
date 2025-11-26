import os
import re
import csv
import sys
import argparse

import re

import re

def extract_classification_report(filepath):
    """
    Extrae precision, recall y f1-score de la línea correspondiente a "dust"
    del archivo best_acc_test_classification_report.txt.
    """
    precision = recall = f1 = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # print(content)
        # Para depuración: descomenta la siguiente línea para ver el contenido leído
        # print("Contenido classification report:\n", content)
        # Se utiliza re.MULTILINE para que ^ coincida con el inicio de cada línea.
        # Se busca la línea que comience con "dust" (ignorando espacios al inicio)
        m = re.search(r"^\s*dust\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)", content, re.MULTILINE)
        if m:
            precision, recall, f1 = m.group(1), m.group(2), m.group(3)
            # print("Valores encontrados: precision={}, recall={}, f1-score={}".format(precision, recall, f1))
        else:
            # print("No se encontró coincidencia para 'dust' en el classification report.")
    except Exception as e:
        # print(f"Error leyendo {filepath}: {e}")
    return precision, recall, f1

def extract_metrics(filepath):
    """
    Extrae Average Class Accuracy, y las accuracy de dust y non-dust
    del archivo best_acc_test_metrics.txt.
    """
    avg_class_acc = dust_acc = non_dust_acc = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # print("Contenido metrics:\n", content)
        m_avg = re.search(r"Average Class Accuracy:\s*([\d\.]+)", content)
        if m_avg:
            avg_class_acc = m_avg.group(1)
            # print("Average Class Accuracy:", avg_class_acc)
        else:
            # print("No se encontró 'Average Class Accuracy'.")
        
        m_dust = re.search(r"Class\s+dust\s*-\s*IoU:\s*([\d\.]+),\s*Accuracy:\s*([\d\.]+)", content)
        if m_dust:
            dust_acc = m_dust.group(2)
            # print("Dust Accuracy:", dust_acc)
        else:
            # print("No se encontró información para 'Class dust'.")
        
        m_nondust = re.search(r"Class\s+non-dust\s*-\s*IoU:\s*([\d\.]+),\s*Accuracy:\s*([\d\.]+)", content)
        if m_nondust:
            non_dust_acc = m_nondust.group(2)
            # print("Non-dust Accuracy:", non_dust_acc)
        else:
            # print("No se encontró información para 'Class non-dust'.")
    except Exception as e:
        # print(f"Error leyendo {filepath}: {e}")
    return avg_class_acc, dust_acc, non_dust_acc


def is_date_folder(foldername):
    # Se asume que el nombre de la carpeta es una fecha en formato "AAAA-MM-DD_HH-MM"
    return re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$", foldername) is not None


def main(mother_folder):
    data_rows = []
    # Recorre todas las entradas de la carpeta madre
    for entry in os.listdir(mother_folder):
        entry_path = os.path.join(mother_folder, entry)
        if os.path.isdir(entry_path) and is_date_folder(entry):
            timestamp = entry
            # Define la ruta base dentro de cada carpeta de fecha
            base_test_folder = os.path.join(entry_path, "checkpoints", "training_results", "test")
            cls_report_path = os.path.join(base_test_folder, "best_acc_test_classification_report.txt")
            metrics_path = os.path.join(base_test_folder, "best_acc_test_metrics.txt")
            
            # Verifica que ambos archivos existan
            if not os.path.exists(cls_report_path):
                # print(f"No se encontró {cls_report_path}. Se omite {timestamp}.")
                continue
            if not os.path.exists(metrics_path):
                # print(f"No se encontró {metrics_path}. Se omite {timestamp}.")
                continue

            # Extraer los datos de cada archivo
            precision, recall, f1 = extract_classification_report(cls_report_path)
            avg_class_acc, dust_acc, non_dust_acc = extract_metrics(metrics_path)
            
            # Solo se agrega la fila si se extrajeron todos los datos necesarios
            if None not in (precision, recall, f1, avg_class_acc, dust_acc, non_dust_acc):
                data_rows.append({
                    "timestamp": timestamp,
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1,
                    "dust accuracy": dust_acc,
                    "non-dust accuracy": non_dust_acc,
                    "Average Class Accuracy": avg_class_acc
                })
            else:
                # print(f"Información incompleta en {timestamp}. Se omite esta entrada.")

    # Genera el archivo CSV con las columnas solicitadas
    output_csv = os.path.join(mother_folder, "resultados.csv")
    try:
        with open(output_csv, "w", newline="") as csvfile:
            fieldnames = ["timestamp", "precision", "recall", "f1-score", "dust accuracy", "non-dust accuracy", "Average Class Accuracy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data_rows:
                writer.writerow(row)
        print(f"Archivo CSV generado en: {output_csv}")
    except Exception as e:
        print(f"Error al escribir el CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extrae datos de archivos de métricas y genera un CSV.')
    parser.add_argument('mother_folder', type=str, help='Directorio de la carpeta madre que contiene las carpetas de fecha.')
    args = parser.parse_args()
    
    main(args.mother_folder)
