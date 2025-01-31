import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, jaccard_score, roc_curve, auc, precision_recall_curve
from datetime import datetime
from models.pointnet2_sem_seg_amtc import get_model
from data_utils.AMTCDataLoader import AMTCDataset
from tqdm import tqdm

def save_info(args, outdir):    
    # Guardamos los argumentos del entrenamiento en la misma carpeta donde se guarda el checkpoint
    info_file = outdir / "info.txt"
    with open(info_file, 'w') as f:
        f.write(f"root_dir: {args.root_dir}\n")
        f.write(f"checkpoint_dir: {args.checkpoint_dir}\n")
        f.write(f"feat_list: {args.feat_list}\n")
        f.write(f"eval_set: {args.eval_set}\n")
        f.write(f"val_test_set: {args.val_test_set}\n")
        f.write(f"unlabeled: {args.unlabeled}\n")

# Calcular accuracy total y por clase
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

# Cargar el modelo y sus pesos desde el checkpoint
def ptnt2_loader(checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    NUM_CLASSES = checkpoint['model_state_dict']['conv2.weight'].shape[0]
    NUM_FEAT = checkpoint['model_state_dict']['sa1.mlp_convs.0.weight'].shape[1] - 3
    
    model = get_model(NUM_CLASSES, NUM_FEAT).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, NUM_CLASSES, NUM_FEAT

# Guardar resultados y métricas en gráficos
def save_results(cm, cm_n, acc_list, iou_list, report_text, fpr, tpr, roc_auc, precision, recall, checkpoint_dir):
    # Crear carpeta "results" en el directorio del checkpoint
    parent_dir = Path(checkpoint_dir).parent.parent
    results_dir = parent_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Crear un subdirectorio con la fecha y hora actuales
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp_dir = results_dir / current_time
    timestamp_dir.mkdir(exist_ok=True)

    # Guardar la matriz de confusión normalizada
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dust', 'non-dust'])
    disp.plot(cmap=plt.cm.Blues, values_format='7d')
    plt.title('Confusion Matrix')
    plt.savefig(timestamp_dir / "confusion_matrix.png")
    plt.close()

    # Guardar la matriz de confusión normalizada
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_n, display_labels=['dust', 'non-dust'])
    disp.plot(cmap=plt.cm.Blues, values_format='.3f')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(timestamp_dir / "confusion_matrix_normalized.png")
    plt.close()

    # Guardar gráfico de evolución de métricas
    plt.figure()
    plt.plot(acc_list, label='Accuracy')
    plt.plot(iou_list, label='IoU')
    plt.xlabel('Frames')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Metrics Evolution Over Frames')
    plt.savefig(timestamp_dir / "metrics_evolution.png")
    plt.close()

    # Graficar curva ROC
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
    roc_curve_path = f"{timestamp_dir}/roc_curve.png"
    plt.savefig(roc_curve_path)
    plt.close()

    # Curva Precision-Recall
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid()
    pr_curve_path = f"{timestamp_dir}/pr_curve.png"
    plt.savefig(pr_curve_path)
    plt.close()

    # Guardar el reporte de clasificación
    with open(timestamp_dir / 'classification_report.txt', 'w') as f:
        f.write(report_text)
    return timestamp_dir 

# Evaluar modelo y generar gráficos
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, jaccard_score
import torch
from tqdm import tqdm

def model_eval_sequence(model, root='', feat=['coord', 'intensity'], data_set='test', num_classes=2, labels_available=True):
    DATASET = AMTCDataset(split=data_set, data_root=root, num_point=4096, voxel_size=0.1, val_test_area=args.val_test_set, feats=feat, num_classes=num_classes, labels_available=labels_available)
    sorted_indices = sorted(range(len(DATASET)), key=lambda idx: DATASET.room_names[idx])
    
    all_predicted_labels, all_true_labels, all_probabilities = [], [], []
    acc_list, iou_list = [], []
    with torch.no_grad():
        for idx in tqdm(sorted_indices, desc="Evaluando frames"):
            data, labels, _ = DATASET.__getitem__(idx, return_index=True)
            points = torch.Tensor(data).unsqueeze(0).float().cuda().transpose(2, 1)
            output = model(points)
            
            # Obtener probabilidades y predicciones
            probabilities = torch.softmax(output[0], dim=2)
            predicted_classes = torch.argmax(output[0], dim=2).reshape(-1).cpu().numpy()
            dust_probabilities = probabilities[0, :, 0].cpu().numpy().reshape(-1)  # Aplanar las probabilidades

            # Concatenar resultados directamente
            if len(all_predicted_labels) == 0:
                all_predicted_labels = predicted_classes
                all_probabilities = dust_probabilities
            else:
                all_predicted_labels = np.concatenate((all_predicted_labels, predicted_classes))
                all_probabilities = np.concatenate((all_probabilities, dust_probabilities))

            if labels_available:
                if len(all_true_labels) == 0:
                    all_true_labels = labels.reshape(-1)
                else:
                    all_true_labels = np.concatenate((all_true_labels, labels.reshape(-1)))

                # Calcular accuracy e IoU
                total_acc, acc_class = class_acc(labels, predicted_classes)
                iou = jaccard_score(labels, predicted_classes, average='macro')
                acc_list.append(total_acc)
                iou_list.append(iou)

    # Procesar resultados finales
    if labels_available:
        total_accuracy, accuracy_per_class = class_acc(all_true_labels, all_predicted_labels)
        report_text = classification_report(all_true_labels, all_predicted_labels, target_names=['dust', 'non-dust'])
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        cm_n = confusion_matrix(all_true_labels, all_predicted_labels, normalize='true')

        # Generar curva ROC
        fpr, tpr, _ = roc_curve(all_true_labels, all_probabilities, pos_label=0)  # Clase "polvo" es la positiva (0)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(all_true_labels, all_probabilities, pos_label=0)

        # Guardar resultados
        timestamp_dir = save_results(cm, cm_n, acc_list, iou_list, report_text, fpr, tpr, roc_auc, precision, recall, args.checkpoint_dir)
    else:
        print("No labels available, cannot compute accuracy or save results.")
    
    return timestamp_dir


# Función principal
def main(args):
    model, NUM_CLASSES, NUM_FEAT = ptnt2_loader(args.checkpoint_dir)
    labels_available = not args.unlabeled
    timestamp_dir = model_eval_sequence(model, root=args.root_dir, feat=args.feat_list, num_classes=NUM_CLASSES, data_set=args.eval_set, labels_available=labels_available)
    save_info(args, timestamp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate models on datasets and generate graphics.')
    parser.add_argument('--root_dir', type=str, default='/home/nicolas/repos/dust-filtering/data/blender_areas', help='Data root path')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/nicolas/repos/custom_pointnet2_pytorch/log/sem_seg_amtc/checkpoints/best_model_acc.pth', help='Checkpoint file dir path')
    parser.add_argument('--feat_list', nargs='+', default=["coord", "intensity"], help='List of features to consider')
    parser.add_argument('--eval_set', type=str, choices=['test', 'train'], default='test', help='Dataset to evaluate')
    parser.add_argument('--val_test_set', nargs=2, type=int, default=[0,1],
                        help='Conjuntos para usar como validación y prueba (default: 0 1)')
    parser.add_argument('--unlabeled', action='store_true', help='Indicate if data is unlabeled')
    args = parser.parse_args()
    main(args)
