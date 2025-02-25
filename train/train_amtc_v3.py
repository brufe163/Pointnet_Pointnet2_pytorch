import argparse
import os
from data_utils.AMTCDataLoader import *
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import seaborn as sns  # Importamos seaborn para mejorar las gráficas
import ast
import json
import csv
"""
v3: usa grabaciones como conjuntos de validación y prueba. Además, añade más métricas.
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['dust', 'non-dust']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

beginning_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

logger = None  # Global logger variable

def log_string(str):
    global logger
    logger.info(str)
    print(str)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def save_info(args, outdir):    
    # Guardamos los argumentos del entrenamiento en la misma carpeta donde se guarda el checkpoint
    info_file = outdir / "info.txt"
    with open(info_file, 'w') as f:
        f.write(f"model: {args.model}\n")
        f.write(f"data_dir: {args.data_dir}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"nclasses: {args.nclasses}\n")
        f.write(f"epoch: {args.epoch}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")
        f.write(f"gpu: {args.gpu}\n")
        f.write(f"optimizer: {args.optimizer}\n")
        f.write(f"log_dir: {args.log_dir}\n")
        f.write(f"decay_rate: {args.decay_rate}\n")
        f.write(f"npoint: {args.npoint}\n")
        f.write(f"voxel_size: {args.voxel_size}\n")
        f.write(f"step_size: {args.step_size}\n")
        f.write(f"lr_decay: {args.lr_decay}\n")
        f.write(f"dropout: {args.dropout}\n")
        # f.write(f"val_test_set: {args.val_test_set}\n")
        f.write(f"sets: {args.sets}\n")
        f.write(f"random_seed: {args.random_seed}\n")
        f.write(f"feat_list: {args.feat_list}\n")
        f.write(f"pretrained_model: {args.pretrained_model}\n")
        f.write(f"freeze_layers: {args.freeze_layers}\n")
        f.write(f"notes: {args.notes}\n")
        f.write(f"patience: {args.patience}\n")

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_amtc', help='Nombre del modelo [default: pointnet2_sem_seg_amtc]')
    parser.add_argument('--data_dir', type=str, default='/ruta/a/tus/datos', help='Ruta a los datos [default: None]')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del batch durante el entrenamiento [default: 16]')
    parser.add_argument('--nclasses', type=int, default=2, help='Número de clases de los datos [default: 2]')
    parser.add_argument('--epoch', default=20, type=int, help='Número de épocas a ejecutar [default: 20]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Tasa de aprendizaje inicial [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU a usar [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam o SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Ruta para los logs [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Tasa de decaimiento de pesos [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Número de puntos [default: 4096]')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Tamaño del voxel [default: 0.05]')
    parser.add_argument('--step_size', type=int, default=10, help='Paso de decaimiento para lr decay [default: cada 10 épocas]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Tasa de decaimiento para lr decay [default: 0.7]')
    parser.add_argument('--dropout', type=float, default=0.7, help='Probabilidad de dropout [default: 0.7]')
    parser.add_argument('--feat_list', nargs='+', default=["coord", "intensity"], help='Lista de características a considerar [default: ["coord", "intensity"]]')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Ruta al modelo preentrenado para fine-tuning [default: None]')
    #parser.add_argument('--freeze_layers', action='store_true', help='Congelar capas excepto las últimas capas totalmente conectadas [default: False]')
    parser.add_argument('--freeze_layers', type=int, default=None, help='Modo de congelado de capas: 1. se congelan hasta FP; 2. se congelan hasta conv1; 3. se congelan hasta conv2.  [default: 1]')
    parser.add_argument('--notes', type=str, default='notes', help='Notas para recordar el propósito de este entrenamiento [default: notes]')
    parser.add_argument('--patience', type=int, default=10, help='Paciencia para EarlyStopping [default: 10 épocas]')
    # parser.add_argument('--val_test_set', nargs=2, type=int, default=[0,1],
    #                     help='Conjuntos para usar como validación y prueba (default: 0 1)')
    parser.add_argument('--sets', type=str, required=True,
                    help="Define las áreas para train, val, test en formato [[1,2,3],[4],[5]]")
    parser.add_argument('--hyperset', action='store_true', help="Activa el modo de entrenamiento con todos los conjuntos de datos")
    parser.add_argument('--random_seed', type=int, default=42, help='Semilla para la creación de datasets [default: 42]')

    args = parser.parse_args()

    # # Validar que los sets no sean los mismos
    # if args.val_test_set[0] == args.val_test_set[1]:
    #     raise ValueError("Los conjuntos de valdiación y prueba son los mismos.")

    return args

def save_metrics_to_csv(metrics, csv_path='log/sem_seg_amtc/training_results.csv'):
    """
    Save training metrics to a CSV file.

    Args:
    - metrics (dict): Metrics to save.
    - csv_path (str): Path to the CSV file.
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()  # Write header only if file is new
        writer.writerow(metrics)

def generate_metrics_and_plots(true_labels, pred_labels, dataset_name, results_dir, model_name):
    sns.set(style="whitegrid", font_scale=1.2)

    # Filtrar los valores de padding (-1)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    mask = true_labels != -1
    true_labels = true_labels[mask]
    pred_labels = pred_labels[mask]

    # Generar reporte de clasificación
    report = classification_report(true_labels, pred_labels, target_names=classes, output_dict=True)
    report_text = classification_report(true_labels, pred_labels, target_names=classes)

    # Añadir aclaración
    report_header = "Reporte de clasificación basado en puntos totales (cada punto es una muestra)\n\n"
    report_text = report_header + report_text

    # Guardar reporte en un archivo de texto
    report_file = os.path.join(results_dir, f'{model_name}_{dataset_name}_classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"Reporte de clasificación guardado en {report_file}")

    # Matriz de confusión
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = confusion_matrix(true_labels, pred_labels, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    plt.title(f'Matriz de Confusión - Conjunto {dataset_name.capitalize()}', fontsize=16)
    plt.xlabel('Predicción', fontsize=14)
    plt.ylabel('Verdadero', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cm_image_path = os.path.join(results_dir, f'{model_name}_{dataset_name}_confusion_matrix_normalized.png')
    plt.savefig(cm_image_path, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusión normalizada guardada en {cm_image_path}")

    if dataset_name=='test':
        # Extraer métricas clave para el CSV
        tp, fp = cm_normalized[0]
        fn, tn = cm_normalized[1]
        print(tp, tn)
        #tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        avg_accuracy = (tp + tn) / 2 
        

        # Consolidar métricas en un diccionario
        metrics = {
            "timestamp": beginning_time, # modificado para que sea el tiempo de inicio del entrenamiento
            "model": args.model,
            "dataset_name": dataset_name,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epoch": args.epoch,
            "voxel_size": args.voxel_size,
            "TN": tn,
            "TP": tp,
            "AVG_Accuracy": avg_accuracy,
            "notes": args.notes
        }

        # Guardar métricas en el archivo CSV
        save_metrics_to_csv(metrics)

    # Curva ROC
    # Binarizar las etiquetas
    true_labels_binarized = label_binarize(true_labels, classes=[0, 1])
    pred_labels_binarized = label_binarize(pred_labels, classes=[0, 1])

    fpr, tpr, _ = roc_curve(true_labels_binarized.ravel(), pred_labels_binarized.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=14)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=14)
    plt.title(f'Curva ROC - Conjunto {dataset_name.capitalize()}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    roc_image_path = os.path.join(results_dir, f'{model_name}_{dataset_name}_roc_curve.png')
    plt.savefig(roc_image_path, bbox_inches='tight')
    plt.close()
    print(f"Curva ROC guardada en {roc_image_path}")

def plot_and_save_metrics(train_loss_history, val_loss_history, iou_history, train_acc_history, val_acc_history, train_acc_avg_history, val_acc_avg_history, results_dir):
    sns.set(style="whitegrid", font_scale=1.2)

    epochs = range(1, len(train_loss_history) + 1)

    # Graficar pérdidas (loss)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_history, label='Pérdida de Entrenamiento')
    plt.plot(epochs, val_loss_history, label='Pérdida de Validación')
    plt.title('Pérdida vs Épocas', fontsize=16)
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Pérdida', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    loss_image_path = os.path.join(results_dir, 'loss.png')
    plt.savefig(loss_image_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de pérdidas guardado en {loss_image_path}")

    # Graficar mIoU
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, iou_history, label='mIoU de Validación', color='green')
    plt.title('mIoU de Validación vs Épocas', fontsize=16)
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('mIoU', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    miou_image_path = os.path.join(results_dir, 'miou.png')
    plt.savefig(miou_image_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de mIoU guardado en {miou_image_path}")

    # Graficar accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_history, label='Accuracy global de Entrenamiento', color='blue')
    plt.plot(epochs, val_acc_history, label='Accuracy global de Validación', color='orange')
    plt.title('Accuracy global vs Épocas', fontsize=16)
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Accuracy global', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    acc_image_path = os.path.join(results_dir, 'accuracy.png')
    plt.savefig(acc_image_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de accuracy global guardado en {acc_image_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_avg_history, label='Accuracy promedio de Entrenamiento', color='blue')
    plt.plot(epochs, val_acc_avg_history, label='Accuracy promedio de Validación', color='orange')
    plt.title('Accuracy promedio vs Épocas', fontsize=16)
    plt.xlabel('Épocas', fontsize=14)
    plt.ylabel('Accuracy promedio', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    acc_avg_image_path = os.path.join(results_dir, 'accuracy_avg.png')
    plt.savefig(acc_avg_image_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de accuracy promedio guardado en {acc_avg_image_path}")


    # Guardar todas las métricas en un archivo de texto
    metrics_file = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write('Época\tPérdida Entrenamiento\tPérdida Validación\tmIoU Validación\tPrecisión Entrenamiento\tPrecisión Validación\tPrecisión Media Validación\n')
        for epoch in range(len(train_loss_history)):
            f.write(f"{epoch + 1}\t{train_loss_history[epoch]:.4f}\t{val_loss_history[epoch]:.4f}\t{iou_history[epoch]:.4f}\t{train_acc_history[epoch]:.4f}\t{val_acc_history[epoch]:.4f}\t{val_acc_avg_history[epoch]:.4f}\n")
    print(f"Métricas guardadas en {metrics_file}")

def collate_fn(batch):
    max_points = max(points.shape[0] for points, _ in batch) # Encuentra la cantidad máxima de puntos en el batch
    points_padded = [] # Lista para almacenar las pointclouds y las etiquetas
    labels_padded = []

    for points, labels in batch:
        padded_points = np.pad(points, ((0, max_points - points.shape[0]), (0, 0)), mode='constant', constant_values=-1)
        points_padded.append(padded_points)
        padded_labels = np.pad(labels, (0, max_points - labels.shape[0]), mode='constant', constant_values=-1)
        labels_padded.append(padded_labels)
    return torch.tensor(points_padded), torch.tensor(labels_padded)


def evaluate_and_generate_metrics(model_name, classifier, data_loader, dataset_name, results_dir):
    classifier.eval()
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        num_batches = len(data_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        for i, (points, target) in tqdm(enumerate(data_loader), total=len(data_loader), smoothing=0.9):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss_sum += loss

            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += target.shape[0]

            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp
            all_pred_labels.append(pred_val.flatten())
            all_true_labels.append(batch_label.flatten())

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        overall_acc = total_correct / float(total_seen)
        test_acc_avg = np.mean(np.array(total_correct_class) / (np.array(total_seen_class) + 1e-6))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class) + 1e-6))

        log_string(f'{model_name} - {dataset_name} - Mean Loss: %f' % (loss_sum / float(num_batches)))
        log_string(f'{model_name} - {dataset_name} - Point Avg Class IoU: %f' % (mIoU))
        log_string(f'{model_name} - {dataset_name} - Point Accuracy: %f' % (overall_acc))
        log_string(f'{model_name} - {dataset_name} - Point Avg Class Acc: %f' % (test_acc_avg))

        # Guardar métricas por clase
        per_class_metrics = {}
        for l in range(NUM_CLASSES):
            iou = total_correct_class[l] / (total_iou_deno_class[l] + 1e-6)
            acc = total_correct_class[l] / (total_seen_class[l] + 1e-6)
            per_class_metrics[seg_label_to_cat[l]] = {
                'IoU': iou,
                'Accuracy': acc
            }
            log_string(f'Class {seg_label_to_cat[l]} - IoU: {iou:.4f}, Accuracy: {acc:.4f}')

        # Guardar reporte en un archivo de texto
        report_file = os.path.join(results_dir, f'{model_name}_{dataset_name}_metrics.txt')
        with open(report_file, 'w') as f:
            f.write(f'Mean Loss: {loss_sum / float(num_batches):.4f}\n')
            f.write(f'Overall Accuracy: {overall_acc:.4f}\n')
            f.write(f'Average Class Accuracy: {test_acc_avg:.4f}\n')
            f.write(f'mIoU: {mIoU:.4f}\n')
            f.write('\nPer Class Metrics:\n')
            for cls_name, metrics in per_class_metrics.items():
                f.write(f'Class {cls_name} - IoU: {metrics["IoU"]:.4f}, Accuracy: {metrics["Accuracy"]:.4f}\n')

        all_pred_labels = np.concatenate(all_pred_labels)
        all_true_labels = np.concatenate(all_true_labels)

        # Generar métricas y gráficos para el conjunto de datos
        generate_metrics_and_plots(
            all_true_labels,
            all_pred_labels,
            dataset_name,
            results_dir,
            model_name
        )

class EarlyStoppingWithCheckpoints:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience  
        self.min_delta = min_delta  
        self.verbose = verbose
        self.counter = 0  
        self.best_acc = None  # Mejor val_acc_avg registrado
        self.best_iou = None  # Mejor mIoU registrado
        self.early_stop = False
        self.best_epoch_acc = None 
        self.best_epoch_iou = None 

    def __call__(self, val_acc_avg, epoch, model, optimizer, checkpoints_dir):
        improved = False

        # VAL_ACC_AVG
        if self.best_acc is None or val_acc_avg > self.best_acc + self.min_delta:
            self.best_acc = val_acc_avg
            self.best_epoch_acc = epoch
            self.save_checkpoint(model, optimizer, checkpoints_dir, "best_model_acc.pth", val_acc_avg, self.best_epoch_acc)
            improved = True

        # Quitamos iou porque si

        if not improved:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} epochs without improvement")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0  # Reiniciar contador si hubo mejora

        return self.best_epoch_acc


    def save_checkpoint(self, model, optimizer, checkpoints_dir, filename, val_acc_avg, epoch):
        savepath = f"{checkpoints_dir}/{filename}"
        print(f"Saving model at {savepath} (val_acc_avg: {val_acc_avg:.4f})")
        state = {
            'epoch': epoch,
            #'class_avg_iou': mIoU,
            'class_avg_acc': val_acc_avg,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
        }
        torch.save(state, savepath)


def main(args):
    global logger  # Declare logger as global
    global NUM_CLASSES  # Añadimos esta línea para usar NUM_CLASSES en evaluate_and_generate_metrics
    global weights      # Añadimos esta línea para usar weights en evaluate_and_generate_metrics
    global criterion    # Añadimos esta línea para usar criterion en evaluate_and_generate_metrics

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg_amtc')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        log_base_dir = Path(args.log_dir)  # Usamos la ruta especificada en args.log_dir
        experiment_dir = experiment_dir.joinpath(log_base_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)  # Crear directorios anidados
        experiment_dir = experiment_dir.joinpath(timestr)  # Subcarpeta con el timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    results_dir = checkpoints_dir.joinpath('training_results/')  # Changed to 'training_results'
    results_dir.mkdir(exist_ok=True)
    # Create subfolders
    train_results_dir = results_dir.joinpath('train/')
    train_results_dir.mkdir(exist_ok=True)
    val_results_dir = results_dir.joinpath('val/')
    val_results_dir.mkdir(exist_ok=True)
    test_results_dir = results_dir.joinpath('test/')
    test_results_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = args.nclasses

    BATCH_SIZE = args.batch_size
    ROOT = args.data_dir
    FEATS = args.feat_list
    VOXEL_SIZE = args.voxel_size
    DROPOUT = args.dropout

    if args.hyperset:
        with open('hypersets.json', 'r') as f:  
            sets_dict = json.load(f) 

        train_areas = sets_dict['train_set'] 
        val_areas = sets_dict['val_set'] 
        test_areas = sets_dict["test_set"]
    else:
        try:
            splits = ast.literal_eval(args.sets)  # Convierte la cadena a una lista de listas
            if not isinstance(splits, list) or len(splits) != 3 or not all(isinstance(lst, list) for lst in splits):
                raise ValueError("El formato debe ser [[1,2,3],[4],[5]] con tres listas para train, val y test.")
        except Exception as e:
            raise ValueError(f"Error procesando los sets: {e}")

        train_areas, val_areas, test_areas = splits

    TRANSFORMS = ComposeTransforms([
        #IntensityJitter(jitter_range_dust=69, jitter_range_no_dust=408),
        DustDispersion(dispersion_level=0.25),                                   
        # FlowVariation(flow_variation_dust=0.005, flow_variation_no_dust=0.004),   
        LocalScaling(scaling_range=(0.9, 1.1)),                                 
        RandomRotation(angle_range=(0, np.pi)),
        PositionJitter(coord_jitter=0.1, uniform=False)
    ])


    RANDOM_SEED = args.random_seed

    def set_seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Si usas múltiples GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def worker_init_fn(worker_id):
        np.random.seed(RANDOM_SEED + worker_id)
        random.seed(RANDOM_SEED + worker_id)
        torch.manual_seed(RANDOM_SEED + worker_id)

    set_seed(RANDOM_SEED)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)



    print("Start loading data...")

    # Crear datasets
    
    if args.hyperset:
        print("Train Set:")
        TRAIN_DATASET = AMTCDataset(
            areas=train_areas,
            data_root=ROOT,
            num_point=args.npoint,
            voxel_size=args.voxel_size,
            feats=FEATS,
            num_classes=NUM_CLASSES,
            labels_available=True,
            transform=TRANSFORMS,
            hyperset=True
        )

        print("Validation Set:")
        VAL_DATASET = AMTCDataset(
            areas=val_areas,
            data_root=ROOT,
            num_point=args.npoint,
            voxel_size=args.voxel_size,
            feats=FEATS,
            num_classes=NUM_CLASSES,
            labels_available=True,
            hyperset=True
        )

        print("Testing Set:")
        TEST_DATASET = AMTCDataset(
            areas=test_areas,
            data_root=ROOT,
            num_point=args.npoint,
            voxel_size=args.voxel_size,
            feats=FEATS,
            num_classes=NUM_CLASSES,
            labels_available=True,
            hyperset=True
        )
    else:
        print("Train Set:")
        TRAIN_DATASET = AMTCDataset(
        areas=train_areas,
        data_root=ROOT,
        num_point=args.npoint,
        voxel_size=args.voxel_size,
        feats=FEATS,
        num_classes=NUM_CLASSES,
        labels_available=True,
        transform=TRANSFORMS
        )

        print("Validation Set:")
        VAL_DATASET = AMTCDataset(
            areas=val_areas,
            data_root=ROOT,
            num_point=args.npoint,
            voxel_size=args.voxel_size,
            feats=FEATS,
            num_classes=NUM_CLASSES,
            labels_available=True
        )

        print("Testing Set:")
        TEST_DATASET = AMTCDataset(
            areas=test_areas,
            data_root=ROOT,
            num_point=args.npoint,
            voxel_size=args.voxel_size,
            feats=FEATS,
            num_classes=NUM_CLASSES,
            labels_available=True
        )
        
    


    NUM_FEATS = TRAIN_DATASET.num_feats
    NUM_POINT = TRAIN_DATASET.num_point


    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    valDataLoader = torch.utils.data.DataLoader(
        VAL_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )

    # unique_labels = np.unique(np.concatenate([dataset.train.labels_list[i] for i in range(len(dataset.train.labels_list))]))
    # print("Valores únicos de las etiquetas en el conjunto de entrenamiento:", unique_labels)

    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()
    

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of validation data is: %d" % len(VAL_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    global_epoch = 0
    best_iou = 0
    best_acc = 0

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    iou_history = []
    train_acc_avg_history = []
    val_acc_avg_history = []    

    classifier = MODEL.get_model(NUM_CLASSES, NUM_FEATS + 3, dropout = DROPOUT).cuda()  # +3 por las coordenadas normalizadas
    criterion = MODEL.get_loss(ignore_index=-1).cuda()

    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # Cargar el checkpoint del modelo preentrenado
    if args.pretrained_model is not None:
        try:
            checkpoint = torch.load(args.pretrained_model)
            pretrained_dict = checkpoint['model_state_dict']
            
            # Obtener el número de clases en el modelo preentrenado
            output_layer_name = 'conv2.weight'  # Nombre de la capa de salida, ajusta si es necesario
            pretrained_num_classes = pretrained_dict[output_layer_name].size(0)

            if pretrained_num_classes != 2 and NUM_CLASSES == 2:
                log_string(f'Pretrained model has {pretrained_num_classes} classes; adapting for 2 classes.')

                # Cargar solo los pesos compatibles, excluyendo la capa de clasificación
                model_dict = classifier.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                
                # Actualizar los pesos del modelo
                model_dict.update(pretrained_dict)
                classifier.load_state_dict(model_dict)

                # Reemplazar la capa de clasificación para el nuevo número de clases
                classifier.conv2 = nn.Conv1d(128, NUM_CLASSES, 1).cuda()
                classifier.conv2.apply(weights_init)  
                start_epoch = 0

            elif pretrained_num_classes == NUM_CLASSES:
                classifier.load_state_dict(pretrained_dict)
                log_string('Loaded pretrained model with matching number of classes.')
                start_epoch = 0

            else:
                log_string(f"Mismatch in number of classes: pretrained model has {pretrained_num_classes}, expected {NUM_CLASSES}.")
                log_string("Starting training from scratch...")
                start_epoch = 0
                classifier.apply(weights_init)

        except Exception as e:
            log_string('Error loading pretrained model: %s' % str(e))
            log_string('Starting training from scratch...')
            start_epoch = 0
            classifier.apply(weights_init)
    else:
        # Si no se proporciona un modelo preentrenado, proceder como antes
        try:
            checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model from checkpoints')
        except:
            log_string('No existing model, starting training from scratch...')
            start_epoch = 0
            classifier.apply(weights_init)

    # Congelar capas si está configurado
    if args.freeze_layers is not None:
        log_string('Freezing layers except the last fully connected layers')
        for name, param in classifier.named_parameters():
            param.requires_grad = False  # Congelamos todas las capas inicialmente
        # Cuáles capas descongelamos
        if args.freeze_layers == 1:
            layers_to_train = ['fp1', 'fp2', 'fp3', 'fp4', 'conv1', 'conv2', 'bn1']
        elif args.freeze_layers == 2:
            layers_to_train = ['conv1', 'conv2', 'bn1']
        else:
            layers_to_train = ['conv2'] # Sólo la capa conv2
        
        for name, param in classifier.named_parameters():
            if any(layer_name in name for layer_name in layers_to_train):
                param.requires_grad = True
                print(f"Descongelando capa: {name}")
            else:
                print(f"Congelando capa: {name}")



    # Verificamos que hay parámetros entrenables
    trainable_params = list(filter(lambda p: p.requires_grad, classifier.parameters()))
    print(f"Número de parámetros entrenables: {len(trainable_params)}")
    if len(trainable_params) == 0:
        raise ValueError("No hay parámetros entrenables. Verifica la lógica para congelar/descongelar capas.")

    # Configuramos el optimizador para que solo actualice los parámetros entrenables
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, classifier.parameters()),
            lr=args.learning_rate,
            momentum=0.9
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    try:
        early_stopping = EarlyStoppingWithCheckpoints(patience=args.patience, min_delta=0, verbose=True)

        for epoch in range(start_epoch, args.epoch):
            '''Train on chopped scenes'''
            log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
            lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
            log_string('Learning rate:%f' % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
            if momentum < 0.01:
                momentum = 0.01
            print('BN momentum updated to: %f' % momentum)
            classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
            num_batches = len(trainDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.train()

            for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
                optimizer.zero_grad()

                points = points.data.numpy()
                # Si tienes la función rotate_point_cloud_z en provider, puedes usarla. Si no, comenta la siguiente línea.
                # points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss.backward()
                optimizer.step()

                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += target.shape[0]
                loss_sum += loss

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))
        
            log_string('Training mean loss: %f' % (loss_sum / num_batches))
            train_loss_history.append(loss_sum.detach().cpu().numpy() / num_batches)
            log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
            train_acc_history.append((total_correct / float(total_seen)))
            train_acc_avg = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
            log_string('Training avg class accuracy: %f' % train_acc_avg)
            train_acc_avg_history.append(train_acc_avg)
            # No generamos métricas y gráficas aquí para reducir el tiempo de entrenamiento

            # if epoch % 5 == 0:
            #     logger.info('Save model...')
            #     savepath = str(checkpoints_dir) + '/model.pth'
            #     log_string('Saving at %s' % savepath)
            #     state = {
            #         'epoch': epoch,
            #         'model_state_dict': classifier.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }
            #     torch.save(state, savepath)
            #     log_string('Saving model....')

            '''Evaluate on validation set'''
            all_pred_labels = []
            all_true_labels = []

            with torch.no_grad():
                num_batches = len(valDataLoader)
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                labelweights = np.zeros(NUM_CLASSES)
                total_seen_class = [0 for _ in range(NUM_CLASSES)]
                total_correct_class = [0 for _ in range(NUM_CLASSES)]
                total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
                classifier = classifier.eval()

                log_string('---- EPOCH %03d VALIDATION ----' % (global_epoch + 1))
                for i, (points, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                    points = points.data.numpy()
                    points = torch.Tensor(points)
                    points, target = points.float().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)

                    seg_pred, trans_feat = classifier(points)
                    pred_val = seg_pred.contiguous().cpu().data.numpy()
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                    batch_label = target.cpu().data.numpy()
                    target = target.view(-1, 1)[:, 0]
                    loss = criterion(seg_pred, target, trans_feat, weights)
                    loss_sum += loss

                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum((pred_val == batch_label))
                    total_correct += correct
                    total_seen += target.shape[0]

                    tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))

                    labelweights += tmp
                    all_pred_labels.append(pred_val.flatten())
                    all_true_labels.append(batch_label.flatten())

                    for l in range(NUM_CLASSES):
                        total_seen_class[l] += np.sum((batch_label == l))
                        total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                        total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

                    total_iou_deno_class = np.maximum(total_iou_deno_class, 1e-6)

                labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
                print('total_correct_class: ', np.array(total_correct_class))
                print('total_iou_deno_class: ', total_iou_deno_class)

                mIoU = np.mean(np.array(total_correct_class) / total_iou_deno_class)
                log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
                log_string('eval point avg class IoU: %f' % (mIoU))
                val_loss_history.append(loss_sum.detach().cpu().numpy() / float(num_batches))
                iou_history.append(mIoU)
                log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))

                val_acc_history.append((total_correct / float(total_seen)))
                val_acc_avg = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
                log_string('eval point avg class acc: %f' % (val_acc_avg))
                val_acc_avg_history.append(val_acc_avg)

                iou_per_class_str = '------- IoU --------\n'
                for l in range(NUM_CLASSES):
                    iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f, acc: %.3f \n' % (
                        seg_label_to_cat[l] + ' ' * (NUM_CLASSES + 1 - len(seg_label_to_cat[l])), labelweights[l],
                        total_correct_class[l] / float(total_iou_deno_class[l]),
                        total_correct_class[l] / (total_seen_class[l] + 1e-6))

                log_string(iou_per_class_str)
                log_string('Eval mean loss: %f' % (loss_sum / num_batches))
                log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

                # Llamar al early stopping
                best_epoch_acc = early_stopping(
                    val_acc_avg=val_acc_avg,
                    epoch=epoch + 1,  # Sumamos 1 para que coincida con la época real
                    model=classifier,
                    optimizer=optimizer,
                    checkpoints_dir=checkpoints_dir
                )

                # Verificar si debemos detener el entrenamiento
                if early_stopping.early_stop:
                    log_string(f'Early stopping triggered after {early_stopping.patience} epochs without improvement.')
                    break  # Salimos del bucle de entrenamiento

            global_epoch += 1

        # Después del entrenamiento, guardamos la información del mejor modelo
        best_model_info_file = os.path.join(results_dir, 'best_model_info.txt')
        with open(best_model_info_file, 'w') as f:
            #f.write(f'Best mIoU Model at Epoch: {early_stopping.best_epoch_iou}\n')
            #f.write(f'Best mIoU: {early_stopping.best_iou:.4f}\n')
            f.write(f'Best Average Class Accuracy Model at Epoch: {early_stopping.best_epoch_acc}\n')
            f.write(f'Best Average Class Accuracy: {early_stopping.best_acc:.4f}\n')

        # Evaluación final en los conjuntos de datos con el mejor modelo
        def evaluate_on_all_datasets():
            dataset_dirs = {'train': str(train_results_dir), 'val': str(val_results_dir), 'test': str(test_results_dir)}
            for model_name, model_path in [('best_acc', str(checkpoints_dir) + '/best_model_acc.pth')]:
                log_string(f'---- EVALUATING {model_name.upper()} MODEL ON ALL DATASETS ----')
                checkpoint = torch.load(model_path)
                classifier.load_state_dict(checkpoint['model_state_dict'])
                classifier.eval()
                for dataset_name, data_loader in [('train', trainDataLoader), ('val', valDataLoader), ('test', testDataLoader)]:
                    log_string(f'Evaluating on {dataset_name} dataset')
                    dataset_results_dir = dataset_dirs[dataset_name]
                    evaluate_and_generate_metrics(model_name, classifier, data_loader, dataset_name, dataset_results_dir)

        evaluate_on_all_datasets()

        # Generar y guardar las métricas y gráficos finales
        plot_and_save_metrics(
            train_loss_history,
            val_loss_history,
            iou_history,
            train_acc_history,
            val_acc_history,
            train_acc_avg_history,
            val_acc_avg_history,
            str(results_dir)  # Guardamos en training_results
        )

        # Retornamos results_dir para usarlo fuera de main()
        return results_dir

    except KeyboardInterrupt:
        print("Entrenamiento interrumpido por el usuario. Generando gráfico...")
        # Guardamos los gráficos con los resultados hasta el punto de interrupción
        plot_and_save_metrics(
            train_loss_history,
            val_loss_history,
            iou_history,
            train_acc_history,
            val_acc_history,
            val_acc_avg_history,
            str(results_dir)  # Guardamos en training_results
        )
        return results_dir  # Retornamos results_dir

if __name__ == '__main__':
    args = parse_args()
    results_dir = main(args)
    save_info(args, results_dir)
