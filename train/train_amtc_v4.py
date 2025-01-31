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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from utils.test_sequence_utils import *
from utils.training_utils import *
import seaborn as sns  # Importamos seaborn para mejorar las gráficas

"""
v4: usa grabaciones como conjuntos de validación y prueba, añade métricas, modularidad del código. Aun no terminado!
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

logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def log_string(str):
    logger.info(str)
    print(str)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


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
    parser.add_argument('--val_test_set', nargs=2, type=int, default=[0,1],
                        help='Conjuntos para usar como validación y prueba (default: 0 1)')
    parser.add_argument('--random_seed', type=int, default=42, help='Semilla para la creación de datasets [default: 42]')

    args = parser.parse_args()

    # Validar que los sets no sean los mismos
    if args.val_test_set[0] == args.val_test_set[1]:
        raise ValueError("Los conjuntos de valdiación y prueba son los mismos.")

    return args


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

def main(args, experiment_dir, checkpoints_dir, results_dir, train_results_dir, val_results_dir, test_results_dir, log_dir):
    #global logger  # Declare logger as global
    global NUM_CLASSES  # Añadimos esta línea para usar NUM_CLASSES en evaluate_and_generate_metrics
    global weights      # Añadimos esta línea para usar weights en evaluate_and_generate_metrics
    global criterion    # Añadimos esta línea para usar criterion en evaluate_and_generate_metrics

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    

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
    val_test_set = args.val_test_set
    DROPOUT = args.dropout

    TRANSFORMS = ComposeTransforms([
        IntensityJitter(jitter_range_dust=69, jitter_range_no_dust=408),
        DustDispersion(dispersion_level=0.25),                                   
        # FlowVariation(flow_variation_dust=0.005, flow_variation_no_dust=0.004),   
        LocalScaling(scaling_range=(0.9, 1.1)),                                 
        RandomRotation(angle_range=(0, np.pi)),
        PositionJitter(coord_jitter=0.1, uniform=False)
    ])


    RANDOM_SEED = args.random_seed
    
    # Carga de datos
    TRAIN_DATASET, VAL_DATASET, TEST_DATASET, trainDataLoader, valDataLoader, testDataLoader = setup_dataloaders(args, transform=TRANSFORMS, test_only=False)
    NUM_FEATS = TRAIN_DATASET.num_feats
    NUM_POINT = TRAIN_DATASET.num_point
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
    val_acc_avg_history = []    


    # Carga del modelo y optimizador
    classifier = MODEL.get_model(NUM_CLASSES, NUM_FEATS + 3, dropout = DROPOUT).cuda()  # +3 por las coordenadas normalizadas
    criterion = MODEL.get_loss(ignore_index=-1).cuda()

    classifier.apply(inplace_relu)

    classifier, start_epoch = load_model_and_checkpoint(args, classifier, log_string)
    optimizer = configure_optimizer(args, classifier)
    

    # Verificamos que hay parámetros entrenables
    trainable_params = list(filter(lambda p: p.requires_grad, classifier.parameters()))
    print(f"Número de parámetros entrenables: {len(trainable_params)}")
    if len(trainable_params) == 0:
        raise ValueError("No hay parámetros entrenables. Verifica la lógica para congelar/descongelar capas.")

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    try:
        for epoch in range(start_epoch, args.epoch):
            '''Train on chopped scenes'''
            log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        
            # Ajustamos learning rate y momentum
            adjust_learning_rate_and_momentum(optimizer, epoch, args, LEARNING_RATE_CLIP, MOMENTUM_ORIGINAL, MOMENTUM_DECCAY, MOMENTUM_DECCAY_STEP, classifier, log_string)
            
            # Entrenamos una época
            train_loss, train_accuracy = train_one_epoch(classifier, optimizer, trainDataLoader, criterion, epoch, NUM_CLASSES, weights)
            log_string('Training mean loss: %f' % train_loss)
            train_loss_history.append(train_loss)
            log_string('Training accuracy: %f' % train_accuracy)
            train_acc_history.append(train_accuracy)

            # Guardamos el modelo cada 5 épocas
            if epoch % 5 == 0:
                logger.info('Save model...')
                savepath = os.path.join(checkpoints_dir, 'model.pth')
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Model saved.')

            # Validamos una época
            val_loss, val_accuracy, mIoU, val_acc_avg, iou_per_class_str, all_pred_labels, all_true_labels = validate_one_epoch(
                classifier, valDataLoader, criterion, NUM_CLASSES, weights, seg_label_to_cat)

            log_string('eval mean loss: %f' % val_loss)
            val_loss_history.append(val_loss)
            iou_history.append(mIoU)
            log_string('eval point avg class IoU: %f' % mIoU)
            log_string('eval point accuracy: %f' % val_accuracy)
            val_acc_history.append(val_accuracy)
            val_acc_avg_history.append(val_acc_avg)
            log_string('eval point avg class acc: %f' % val_acc_avg)
            log_string(iou_per_class_str)

            # Guardamos el mejor modelo
            best_iou, best_acc, best_epoch_iou, best_epoch_acc, saved = save_best_model(checkpoints_dir, epoch, mIoU, val_acc_avg, classifier, optimizer, best_iou, best_acc, log_string)
            if saved:
                log_string('Best model saved.')

            log_string('Best mIoU so far: %f' % best_iou)
            global_epoch += 1

        # Después del entrenamiento, guardamos la información del mejor modelo
        best_model_info_file = os.path.join(results_dir, 'best_model_info.txt')
        with open(best_model_info_file, 'w') as f:
            f.write(f'Best mIoU Model at Epoch: {best_epoch_iou}\n')
            f.write(f'Best mIoU: {best_iou:.4f}\n')
            f.write(f'Best Average Class Accuracy Model at Epoch: {best_epoch_acc}\n')
            f.write(f'Best Average Class Accuracy: {best_acc:.4f}\n')

        # Evaluación final en los conjuntos de datos con el mejor modelo
        evaluate_on_all_datasets(classifier, checkpoints_dir, trainDataLoader, valDataLoader, testDataLoader, train_results_dir, val_results_dir, test_results_dir, log_string)
        # Generar y guardar las métricas y gráficos finales
        save_training_metrics(
            train_loss_history,
            val_loss_history,
            iou_history,
            train_acc_history,
            val_acc_history,
            val_acc_avg_history,
            str(results_dir)  # Guardamos en training_results
        )

    except KeyboardInterrupt:
        print("Entrenamiento interrumpido por el usuario. Generando gráfico...")
        # Guardamos los gráficos con los resultados hasta el punto de interrupción
        save_training_metrics(
            train_loss_history,
            val_loss_history,
            iou_history,
            train_acc_history,
            val_acc_history,
            val_acc_avg_history,
            str(results_dir)  # Guardamos en training_results
        )


if __name__ == '__main__':
    args = parse_args()
    experiment_dir, checkpoints_dir, results_dir, train_results_dir, val_results_dir, test_results_dir, log_dir = create_experiment_dirs(args)
    main(args, experiment_dir, checkpoints_dir, results_dir, train_results_dir, val_results_dir, test_results_dir, log_dir)

    save_info(args, results_dir)
