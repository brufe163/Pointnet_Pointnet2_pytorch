import argparse
import os  
from pathlib import Path  
from datetime import datetime  
import numpy as np 
import torch  
import torch.nn as nn  
import torch.optim as optim  
from tqdm import tqdm 
import matplotlib.pyplot as plt  
from sklearn.metrics import jaccard_score
import logging
import time
from utils.test_sequence_utils import *

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


def create_experiment_dirs(args):
    # Directorio base
    base_dir = Path('./log/sem_seg_amtc')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre del experimento
    if args.log_dir is None:
        timestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        experiment_dir = base_dir / timestr
    else:
        experiment_dir = base_dir / args.log_dir
    experiment_dir.mkdir(exist_ok=True)
    
    # Directorio de checkpoints
    checkpoints_dir = experiment_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Directorio de resultados
    results_dir = checkpoints_dir / 'training_results'
    results_dir.mkdir(exist_ok=True)
    
    # Subdirectorios de resultados
    train_results_dir = results_dir / 'train'
    val_results_dir = results_dir / 'val'
    test_results_dir = results_dir / 'test'
    for dir in [train_results_dir, val_results_dir, test_results_dir]:
        dir.mkdir(exist_ok=True)
    
    # Directorio de logs
    log_dir = experiment_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    return experiment_dir, checkpoints_dir, results_dir, train_results_dir, val_results_dir, test_results_dir, log_dir


# Necesario para cuando cargamos un checkpoint de un modelo con más clases
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

def load_model_and_checkpoint(args, classifier, log_string = None):
    if args.pretrained_model is not None:
        try:
            checkpoint = torch.load(args.pretrained_model)
            pretrained_dict = checkpoint['model_state_dict']
            
            # Obtener el número de clases en el modelo preentrenado
            output_layer_name = 'conv2.weight'  # Ajusta si es necesario
            pretrained_num_classes = pretrained_dict[output_layer_name].size(0)

            if pretrained_num_classes != args.nclasses:
                log_string(f'El modelo preentrenado tiene {pretrained_num_classes} clases; adaptando a {args.nclasses} clases.')

                # Cargar los pesos compatibles, excluyendo la capa de clasificación
                model_dict = classifier.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                
                # Actualizar los pesos del modelo
                model_dict.update(pretrained_dict)
                classifier.load_state_dict(model_dict)

                # Reemplazar la capa de clasificación para el nuevo número de clases
                classifier.conv2 = nn.Conv1d(128, args.nclasses, 1).cuda()
                classifier.conv2.apply(weights_init)
                start_epoch = 0

            else:
                classifier.load_state_dict(pretrained_dict)
                log_string('Modelo preentrenado cargado con el mismo número de clases.')
                start_epoch = 0

        except Exception as e:
            log_string(f'Error al cargar el modelo preentrenado: {str(e)}')
            log_string('Iniciando entrenamiento desde cero...')
            classifier.apply(weights_init)
            start_epoch = 0

    else:
        # Si no se proporciona un modelo preentrenado
        log_string('No se proporcionó modelo preentrenado, iniciando desde cero.')
        classifier.apply(weights_init)
        start_epoch = 0

    # Congelar capas si está configurado
    if args.freeze_layers is not None:
        log_string('Congelando capas según la configuración.')
        for name, param in classifier.named_parameters():
            param.requires_grad = False  # Congelar todas las capas inicialmente

        # Definir qué capas descongelar
        if args.freeze_layers == 1:
            layers_to_train = ['fp1', 'fp2', 'fp3', 'fp4', 'conv1', 'conv2', 'bn1']
        elif args.freeze_layers == 2:
            layers_to_train = ['conv1', 'conv2', 'bn1']
        else:
            layers_to_train = ['conv2']  # Solo la capa conv2

        for name, param in classifier.named_parameters():
            if any(layer_name in name for layer_name in layers_to_train):
                param.requires_grad = True
                log_string(f"Descongelando capa: {name}")
            else:
                log_string(f"Congelando capa: {name}")

    # Verificar que hay parámetros entrenables
    trainable_params = list(filter(lambda p: p.requires_grad, classifier.parameters()))
    log_string(f"Número de parámetros entrenables: {len(trainable_params)}")
    if len(trainable_params) == 0:
        raise ValueError("No hay parámetros entrenables. Verifica la lógica para congelar/descongelar capas.")

    return classifier, start_epoch

def configure_optimizer(args, classifier):
    trainable_params = filter(lambda p: p.requires_grad, classifier.parameters())

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.learning_rate,
            momentum=0.9
        )
    return optimizer

def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum

def adjust_learning_rate_and_momentum(optimizer, epoch, args, LEARNING_RATE_CLIP, MOMENTUM_ORIGINAL, MOMENTUM_DECCAY, MOMENTUM_DECCAY_STEP, classifier, log_string = None):
    lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    log_string('Learning rate:%f' % lr)
    
    momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
    if momentum < 0.01:
        momentum = 0.01
    print('BN momentum updated to: %f' % momentum)
    classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

def train_one_epoch(classifier, optimizer, trainDataLoader, criterion, epoch, NUM_CLASSES, weights):
    classifier.train()
    num_batches = len(trainDataLoader)
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        optimizer.zero_grad()

        points = points.float().cuda()
        target = target.long().cuda()
        points = points.transpose(2, 1)

        seg_pred, trans_feat = classifier(points)
        seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

        target = target.view(-1)
        loss = criterion(seg_pred, target, trans_feat, weights)
        loss.backward()
        optimizer.step()

        pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
        correct = np.sum(pred_choice == target.cpu().numpy())
        total_correct += correct
        total_seen += target.shape[0]
        loss_sum += loss.item()

    train_loss = loss_sum / num_batches
    train_accuracy = total_correct / float(total_seen)
    return train_loss, train_accuracy

def validate_one_epoch(classifier, valDataLoader, criterion, NUM_CLASSES, weights, seg_label_to_cat):
    classifier.eval()
    num_batches = len(valDataLoader)
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    labelweights = np.zeros(NUM_CLASSES)
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        for i, (points, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
            points = points.float().cuda()
            target = target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1)
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss_sum += loss.item()

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
    total_iou_deno_class = np.maximum(total_iou_deno_class, 1e-6)

    mIoU = np.mean(np.array(total_correct_class) / total_iou_deno_class)
    val_loss = loss_sum / num_batches
    val_accuracy = total_correct / float(total_seen)
    val_acc_avg = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))

    # Preparar resultados por clase para imprimir
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        iou = total_correct_class[l] / float(total_iou_deno_class[l])
        acc = total_correct_class[l] / (total_seen_class[l] + 1e-6)
        iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f, acc: %.3f \n' % (
            seg_label_to_cat[l] + ' ' * (NUM_CLASSES + 1 - len(seg_label_to_cat[l])),
            labelweights[l], iou, acc)

    return val_loss, val_accuracy, mIoU, val_acc_avg, iou_per_class_str, all_pred_labels, all_true_labels

def evaluate_on_all_datasets(classifier, checkpoints_dir, trainDataLoader, valDataLoader, testDataLoader, train_results_dir, val_results_dir, test_results_dir, log_string = None):
    dataset_dirs = {'train': str(train_results_dir), 'val': str(val_results_dir), 'test': str(test_results_dir)}
    for model_name, model_path in [('best_iou', os.path.join(checkpoints_dir, 'best_model_iou.pth')), ('best_acc', os.path.join(checkpoints_dir, 'best_model_acc.pth'))]:
        log_string(f'---- EVALUATING {model_name.upper()} MODEL ON ALL DATASETS ----')
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.eval()
        for dataset_name, data_loader in [('train', trainDataLoader), ('val', valDataLoader), ('test', testDataLoader)]:
            log_string(f'Evaluating on {dataset_name} dataset')
            dataset_results_dir = dataset_dirs[dataset_name]
            evaluate_and_generate_metrics(model_name, classifier, data_loader, dataset_name, dataset_results_dir)


def save_best_model(checkpoints_dir, epoch, mIoU, val_acc_avg, classifier, optimizer, best_iou, best_acc, log_string = None):
    saved = False
    best_epoch_acc = None
    best_epoch_iou = None
    if mIoU >= best_iou:
        best_iou = mIoU
        best_epoch_iou = epoch + 1
        log_string('Save best mIoU model...')
        savepath = os.path.join(checkpoints_dir, 'best_model_iou.pth')
        log_string('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'class_avg_iou': mIoU,
            'class_avg_acc': val_acc_avg,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        saved = True

    if val_acc_avg >= best_acc:
        best_acc = val_acc_avg
        best_epoch_acc = epoch + 1
        log_string('Save best accuracy model...')
        savepath = os.path.join(checkpoints_dir, 'best_model_acc.pth')
        log_string('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'class_avg_iou': mIoU,
            'class_avg_acc': val_acc_avg,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        saved = True

    return best_iou, best_acc, best_epoch_iou, best_epoch_acc,  saved

# Guardamos los gráficos de entrenamiento y las métricas en un archivo de texto
def save_training_metrics(train_loss_history, val_loss_history, iou_history, train_acc_history, val_acc_history, val_acc_avg_history, results_dir):
    epochs = range(1, len(train_loss_history) + 1)

    # Graficamos la pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_history, label='Pérdida de Entrenamiento')
    plt.plot(epochs, val_loss_history, label='Pérdida de Validación')
    plt.title('Pérdida vs Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss.png'))
    plt.close()

    # Graficamos el mIoU
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, iou_history, label='mIoU de Validación', color='green')
    plt.title('mIoU de Validación vs Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'miou.png'))
    plt.close()

    # Graficamos la precisión
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc_history, label='Precisión de Entrenamiento', color='blue')
    plt.plot(epochs, val_acc_history, label='Precisión de Validación', color='orange')
    plt.title('Precisión vs Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'accuracy.png'))
    plt.close()

    # Guardamos todas las métricas en un archivo de texto
    metrics_file = os.path.join(results_dir, 'training_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write('Época\tPérdida Entrenamiento\tPérdida Validación\tmIoU\tPrecisión Entrenamiento\tPrecisión Validación\tPrecisión Media Validación\n')
        for epoch in range(len(train_loss_history)):
            f.write(f"{epoch + 1}\t{train_loss_history[epoch]:.4f}\t{val_loss_history[epoch]:.4f}\t{iou_history[epoch]:.4f}\t{train_acc_history[epoch]:.4f}\t{val_acc_history[epoch]:.4f}\t{val_acc_avg_history[epoch]:.4f}\n")
    print(f"Métricas de entrenamiento guardadas en {metrics_file}")
