import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, jaccard_score, roc_curve, auc, precision_recall_curve
from datetime import datetime
from tqdm import tqdm
from data_utils.AMTCDataLoader import *
from utils.training_utils import *
#from models.pointnet2_sem_seg_amtc import get_model
from utils.logger_utils import logger, log_string


# Guardar todos los argumentos del entrenamiento automáticamente en la misma carpeta donde se guarda el checkpoint
def save_info(args, outdir):
    info_file = outdir / "info.txt"
    with open(info_file, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

# En base al checkpoint, buscamos una carpeta donde guardar las cosas
def path_to_save(checkpoint_dir):
    # Creamos carpeta "results" en el directorio del checkpoint
    parent_dir = Path(checkpoint_dir).parent.parent
    results_dir = parent_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Usamos la fecha como id de cada uso
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_dir = results_dir / current_time
    timestamp_dir.mkdir(exist_ok=True)

    return timestamp_dir

def setup_dataloaders(args, transform = None, test_only = False):
    print("Start loading data...")
    trainDataLoader = None
    valDataLoader = None
    TRAIN_DATASET = None
    VAL_DATASET = None

    if test_only == False:
        print("Train Set:")
        TRAIN_DATASET = AMTCDataset(
            split='train',
            data_root=args.data_dir,
            num_point=args.npoint,
            voxel_size=args.voxel_size,
            val_test_area=args.val_test_set,
            feats=args.feat_list,
            num_classes=args.nclasses,
            labels_available=True,
            transform=transform
        )
        trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                                  pin_memory=True, drop_last=True, collate_fn=collate_fn,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
                                                  
        print("Validation Set:")
        VAL_DATASET = AMTCDataset(split='val', 
            data_root=args.data_dir, 
            num_point=args.npoint, 
            voxel_size=args.voxel_size, 
            val_test_area=args.val_test_set,
            feats=args.feat_list, 
            num_classes=args.nclasses, 
            labels_available=True)
        
        valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                 pin_memory=True, drop_last=True, collate_fn=collate_fn)

    print("Testing Set:")
    TEST_DATASET = AMTCDataset(split='test', 
            data_root=args.data_dir, 
            num_point=args.npoint, 
            voxel_size=args.voxel_size, 
            val_test_area=args.val_test_set,
            feats=args.feat_list, 
            num_classes=args.nclasses, 
            labels_available=True)
    

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=8,
                                                 pin_memory=True, drop_last=True, collate_fn=collate_fn)

    return TRAIN_DATASET, VAL_DATASET, TEST_DATASET, trainDataLoader, valDataLoader, testDataLoader


# Guardamos resultados y métricas en gráficos
def save_results(cm, cm_n, acc_list, iou_list, report_text, fpr, tpr, roc_auc, precision, recall, avg_inference_time, timestamp_dir):
    # Guardamos la matriz de confusión 
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dust', 'non-dust'])
    disp.plot(cmap=plt.cm.Blues, values_format='7d')
    plt.title('Confusion Matrix')
    plt.savefig(Path(timestamp_dir) / "confusion_matrix.png")
    plt.close()

    # Guardamos la matriz de confusión normalizada
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_n, display_labels=['dust', 'non-dust'])
    disp.plot(cmap=plt.cm.Blues, values_format='.3f')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(Path(timestamp_dir) / "confusion_matrix_normalized.png")
    plt.close()

    # Gráfico de evolución de métricas
    plt.figure()
    plt.plot(acc_list, label='Accuracy')
    plt.plot(iou_list, label='IoU')
    plt.xlabel('Frames')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Metrics Evolution Over Frames')
    plt.savefig(Path(timestamp_dir) / "metrics_evolution.png")
    plt.close()

    # Curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(Path(timestamp_dir) / "roc_curve.png")
    plt.close()

    # Curva PR
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.savefig(Path(timestamp_dir) / "pr_curve.png")
    plt.close()

    # Guardamos reporte de clasificación
    report_text += f"\nAverage Inference Time: {avg_inference_time:.4f} seconds"
    with open(Path(timestamp_dir) / 'classification_report.txt', 'w') as f:
        f.write(report_text)


# Calculamos las métricas finales para toda la secuencia de pointclouds
def calculate_metrics(all_true_labels, all_predicted_labels, all_probabilities, class_names=['dust', 'non-dust']):
    assert len(all_true_labels) == len(all_probabilities), "Las longitudes de las etiquetas y probabilidades no coinciden."
    # Calculamos la precisión total y por clase usando la función `class_acc`
    total_accuracy, accuracy_per_class = class_acc(all_true_labels, all_predicted_labels)

    # Generamos el reporte de clasificación, que incluye precisión, recall y F1-score para cada clase
    report_text = classification_report(all_true_labels, all_predicted_labels, target_names=class_names)

    # Calculamos la matriz de confusión y su versión normalizada
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    cm_n = confusion_matrix(all_true_labels, all_predicted_labels, normalize='true')

    # Calculamos la curva ROC para la clase "dust", que consideramos como la clase positiva (0)
    fpr, tpr, _ = roc_curve(all_true_labels, all_probabilities, pos_label=0)
    roc_auc = auc(fpr, tpr)

    # Calculamos la curva de precisión-recall para la clase "dust"
    precision, recall, _ = precision_recall_curve(all_true_labels, all_probabilities, pos_label=0)

    # Creamos un diccionario con todas las métricas calculadas
    metrics = {
        'total_accuracy': total_accuracy,
        'accuracy_per_class': accuracy_per_class,
        'classification_report': report_text,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_n,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall
    }

    return metrics

# Calcularmos accuracy total y por clase
def class_acc(segment_classes, predicted_classes):
    assert len(segment_classes) == len(predicted_classes), "Las longitudes de las clases segmentadas y predichas deben ser iguales."
    unique_classes = np.unique(segment_classes)
    accuracy_per_class = {}
    
    for cls in unique_classes:
        class_indices = (segment_classes == cls)
        correct_predictions = np.sum(predicted_classes[class_indices] == cls)
        total_points = np.sum(class_indices)
        accuracy_per_class[cls] = correct_predictions / total_points if total_points > 0 else 0.0
    total_accuracy = np.sum(segment_classes == predicted_classes) / len(segment_classes)
    
    return total_accuracy, accuracy_per_class

# Evaluamos un solo frame para devolver las predicciones, probabilidades y métricas
def evaluate_frame(model, data, labels, labels_available):
    points = data.float().to('cuda').transpose(2, 1)
    output = model(points)

    probabilities = torch.softmax(output[0], dim=2)
    predicted_classes = torch.argmax(output[0], dim=2).cpu().numpy().reshape(-1)
    dust_probabilities = probabilities[0, :, 0].detach().cpu().numpy().reshape(-1)


    if labels_available:
        labels_np = labels.cpu().numpy().reshape(-1)
        total_acc, _ = class_acc(labels_np, predicted_classes)
        iou = jaccard_score(labels_np, predicted_classes, average='macro')
        return predicted_classes, dust_probabilities, total_acc, iou, labels_np
    else:
        return predicted_classes, dust_probabilities, None, None, None



# Cargamos el modelo y sus pesos desde el checkpoint
def ptnt2_loader(checkpoint_dir, model):
    if model == 'pointnet2_sem_seg_amtc':
        from models.pointnet2_sem_seg_amtc import get_model
    elif model == 'pointnet2_sem_seg_amtc_red':
        from models.pointnet2_sem_seg_amtc_red import get_model
    else:
        print('No se ha especificado el modelo a probar.')

    checkpoint = torch.load(checkpoint_dir)
    NUM_CLASSES = checkpoint['model_state_dict']['conv2.weight'].shape[0]
    NUM_FEAT = checkpoint['model_state_dict']['sa1.mlp_convs.0.weight'].shape[1] - 3
    
    model = get_model(NUM_CLASSES, NUM_FEAT).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, NUM_CLASSES, NUM_FEAT
