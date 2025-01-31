import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from data_utils.AMTCDataLoader import *
from tqdm import tqdm
from utils.test_sequence_utils import *
from utils.training_utils import *
from torch.utils.data import DataLoader

# Variables globales y definiciones
NUM_CLASSES = 2  # Modifica según tu caso
seg_label_to_cat = {0: "Polvo", 1: " No Polvo"}
classes = ["Polvo", "NO Polvo"]

# Define la función para evaluación
def evaluate_and_generate_metrics(model_name, classifier, data_loader, dataset_name, results_dir):
    classifier.eval()
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        num_batches = len(data_loader)
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        for i, (points, target) in tqdm(enumerate(data_loader), total=len(data_loader), smoothing=0.9):
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]

            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += target.shape[0]

            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            all_pred_labels.append(pred_val.flatten())
            all_true_labels.append(batch_label.flatten())

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        overall_acc = total_correct / float(total_seen)
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class) + 1e-6))

        print(f'Point Accuracy: {overall_acc}')
        print(f'mIoU: {mIoU}')

        all_pred_labels = np.concatenate(all_pred_labels)
        all_true_labels = np.concatenate(all_true_labels)

        generate_metrics_and_plots(all_true_labels, all_pred_labels, dataset_name, results_dir, model_name)

def collate_fn(batch):
    max_points = max(points.shape[0] for points, _ in batch) # Encuentra la cantidad máxima de puntos en el batch
    points_padded = [] # Lista para almacenar las pointclouds y las etiquetas
    labels_padded = []

    for points, labels in batch:
        padded_points = np.pad(points, ((0, max_points - points.shape[0]), (0, 0)), mode='constant')
        points_padded.append(padded_points)
        padded_labels = np.pad(labels, (0, max_points - labels.shape[0]), mode='constant')
        labels_padded.append(padded_labels)
    return torch.tensor(points_padded), torch.tensor(labels_padded)

def generate_metrics_and_plots(true_labels, pred_labels, dataset_name, results_dir, model_name):
    sns.set(style="whitegrid", font_scale=1.2)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    mask = true_labels != -1
    true_labels = true_labels[mask]
    pred_labels = pred_labels[mask]
    report_text = classification_report(true_labels, pred_labels, target_names=classes)
    report_file = os.path.join(results_dir, f'{model_name}_{dataset_name}_classification_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"Reporte de clasificación guardado en {report_file}")

    cm_normalized = confusion_matrix(true_labels, pred_labels, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    cm_image_path = os.path.join(results_dir, f'{model_name}_{dataset_name}_confusion_matrix.png')
    plt.savefig(cm_image_path)
    plt.close()
    print(f"Matriz de confusión guardada en {cm_image_path}")

    true_labels_binarized = label_binarize(true_labels, classes=[0, 1])
    pred_labels_binarized = label_binarize(pred_labels, classes=[0, 1])
    fpr, tpr, _ = roc_curve(true_labels_binarized.ravel(), pred_labels_binarized.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Curva ROC (área = {roc_auc:.2f})')
    roc_image_path = os.path.join(results_dir, f'{model_name}_{dataset_name}_roc_curve.png')
    plt.savefig(roc_image_path)
    plt.close()
    print(f"Curva ROC guardada en {roc_image_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluación del modelo de segmentación')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Ruta al modelo preentrenado')
    parser.add_argument('--data_path', type=str, required=True, help='Ruta al DataLoader de evaluación')
    parser.add_argument('--val_test_set', nargs=2, type=int, default=[0,1])
    args = parser.parse_args()

    # Cargar el modelo y DataLoader
    timestamp_dir = path_to_save(args.checkpoint_dir)
    model, NUM_CLASSES, NUM_FEAT = ptnt2_loader(args.checkpoint_dir)
    DATASET = AMTCDataset(split='test', data_root=args.data_path, num_point=4096, voxel_size=0.1, val_test_area=args.val_test_set, feats=['coord', 'intensity'], num_classes=NUM_CLASSES, labels_available=True)
    dataloader = DataLoader(DATASET, batch_size=1, shuffle=False, num_workers=8,
                                                 pin_memory=True, drop_last=True)

    evaluate_and_generate_metrics('model', model, dataloader, 'test', timestamp_dir)

if __name__ == '__main__':
    main()
