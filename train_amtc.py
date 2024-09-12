"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.AMTCDataLoader import AMTCDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['dust', 'non-dust']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_amtc', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--data_dir', type=str, default='/home/nicolas/repos/dust-filtering/data/blender_areas', help='Data path [default: None]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--nclasses', type=int, default=2, help='number of classes of the data [default: 13]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-5 [default: 5]')
    parser.add_argument('--feat_list', nargs='+', default=["coord", "intensity"], help='list of the desired features to consider [default: ["coord", "intensity"]]')

    return parser.parse_args()

import matplotlib.pyplot as plt
import os

def plot_loss_and_iou(train_loss_history, eval_loss_history, iou_history, train_acc_history, eval_acc_history, eval_acc_avg_history, checkpoints_dir):
    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(12, 8))

    # Graficar el Training Loss y el Eval Loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss_history, label='Training Loss')
    plt.plot(epochs, eval_loss_history, label='Eval Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Graficar el mIoU
    plt.subplot(3, 1, 2)
    plt.plot(epochs, iou_history, label='mIoU', color='green')
    plt.title('mIoU vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()

    # Graficar los accuracies
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_acc_history, label='Training Accuracy', color='blue')
    plt.plot(epochs, eval_acc_history, label='Eval Accuracy', color='orange')
    plt.plot(epochs, eval_acc_avg_history, label='Eval Avg Class Accuracy', color='red')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Guardar la imagen en la carpeta de checkpoints
    image_path = os.path.join(checkpoints_dir, 'training_metrics.png')
    plt.savefig(image_path)
    print(f"Gráfico guardado en {image_path}")

    plt.show()  # Opcional: Puedes eliminar esto si solo quieres guardar la imagen


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

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
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
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
    TEST_AREA = args.test_area
    FEATS = args.feat_list

    print("start loading training data ...")
    TRAIN_DATASET = AMTCDataset(split='train', data_root=ROOT, num_point=None, test_area=TEST_AREA, block_size=1.0, sample_rate=1.0, feats = FEATS, num_classes = NUM_CLASSES, transform=None)
    print("start loading test data ...")
    TEST_DATASET = AMTCDataset(split='test', data_root=ROOT, num_point=None, test_area=TEST_AREA, block_size=1.0, sample_rate=1.0,feats = FEATS, num_classes = NUM_CLASSES, transform=None)

    NUM_FEATS = TRAIN_DATASET.num_feats
    NUM_POINT = TRAIN_DATASET.num_point

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    train_loss_history = []
    train_acc_history = []
    eval_loss_history = []
    eval_acc_history = []
    iou_history = []
    eval_acc_avg_history = []


    classifier = MODEL.get_model(NUM_CLASSES, NUM_FEATS+3).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0
    best_acc = 0
    try:
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
            classifier = classifier.train()

            for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
                optimizer.zero_grad()

                #  print('target al inicio: ', target.cpu().numpy(), target.shape)

                points = points.data.numpy()
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                #  print('target despues de long: ', target.cpu().numpy(), target.shape)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                # print('batch_label: ', batch_label, np.shape(batch_label))
                target = target.view(-1, 1)[:, 0]
                #  print('target despues de view: ', target.cpu().numpy(),target.shape)
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss.backward()
                optimizer.step()

                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                # print('pred_choice: ', pred_choice, np.shape(pred_choice))
                # print(f"Predictions unique values: {np.unique(pred_choice)}")
                # print(f"Labels unique values: {np.unique(batch_label)}")

                # print(f"Batch label shape: {batch_label.shape}")
                # print(f"Prediction shape: {pred_choice.shape}")


                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                loss_sum += loss
            log_string('Training mean loss: %f' % (loss_sum / num_batches))
            train_loss_history.append(loss_sum.detach().cpu().numpy() / num_batches)
            log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
            train_acc_history.append((total_correct / float(total_seen)))

            if epoch % 5 == 0:
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            '''Evaluate on chopped scenes'''
            with torch.no_grad():
                num_batches = len(testDataLoader)
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                labelweights = np.zeros(NUM_CLASSES)
                total_seen_class = [0 for _ in range(NUM_CLASSES)]
                total_correct_class = [0 for _ in range(NUM_CLASSES)]
                total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
                classifier = classifier.eval()

                log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
                for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
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
                    total_seen += (BATCH_SIZE * NUM_POINT)
                    tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                    labelweights += tmp

                    for l in range(NUM_CLASSES):
                        total_seen_class[l] += np.sum((batch_label == l))  # Total de puntos en la clase l
                        total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))  # Correctamente predichos en clase l
                        total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))  # Unión de predicción y verdad

                    # Añadir límite mínimo para evitar divisiones pequeñas
                    total_iou_deno_class = np.maximum(total_iou_deno_class, 1e-6)

                labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
                mIoU = np.mean(np.array(total_correct_class) / total_iou_deno_class)
                log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
                log_string('eval point avg class IoU: %f' % (mIoU))
                eval_loss_history.append(loss_sum.detach().cpu().numpy() / float(num_batches)) 
                iou_history.append(mIoU)
                log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
                
                eval_acc_history.append((total_correct / float(total_seen)))
                eval_acc_avg = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
                log_string('eval point avg class acc: %f' % (eval_acc_avg))
                eval_acc_avg_history.append(eval_acc_avg)

                iou_per_class_str = '------- IoU --------\n'
                for l in range(NUM_CLASSES):
                    iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                        seg_label_to_cat[l] + ' ' * (NUM_CLASSES + 1 - len(seg_label_to_cat[l])), labelweights[l - 1],
                        total_correct_class[l] / float(total_iou_deno_class[l]))

                log_string(iou_per_class_str)
                log_string('Eval mean loss: %f' % (loss_sum / num_batches))
                log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

                if mIoU >= best_iou:
                    best_iou = mIoU
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model_iou.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU,
                        'class_avg_acc': eval_acc_avg,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    log_string('Saving model....')
                if eval_acc_avg >= best_acc:
                    best_acc = eval_acc_avg
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model_acc.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU,
                        'class_avg_acc': eval_acc_avg,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    log_string('Saving model....')
                log_string('Best mIoU: %f' % best_iou)
            global_epoch += 1
    except KeyboardInterrupt:
        print("Entrenamiento interrumpido por el usuario. Generando gráfico...")
        return train_loss_history, eval_loss_history, iou_history, train_acc_history, eval_acc_history, eval_acc_avg_history, checkpoints_dir
            
    return train_loss_history, eval_loss_history, iou_history, train_acc_history, eval_acc_history, eval_acc_avg_history, checkpoints_dir

if __name__ == '__main__':
    args = parse_args()
    train_loss_history, eval_loss_history, iou_history, train_acc_history, eval_acc_history, eval_acc_avg_history, checkpoints_dir = main(args)

    plot_loss_and_iou(train_loss_history, eval_loss_history, iou_history, train_acc_history, eval_acc_history, eval_acc_avg_history, checkpoints_dir)
