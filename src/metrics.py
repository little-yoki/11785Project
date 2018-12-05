import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import math
import argparse
import pandas as pd

from model import UNet
from weighted_multi_class_IoU import Weighted_IoU as mean_IoU
from train_small import Dataset, preprocess
from train import preprocess as preprocess_large


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WORKERS = 4
BATCH_SIZE = 1
AVERAGE_RGB = True
SAMPLE_RATE = 0.2
TEST_SIZE = 0.2


def validate(model, val_loader, n_classes, channel='RGB'):
    print('Validating...')
    model.eval()

    iou_list = []
    f1_scores = []
    outputs = []
    preds = []
    targets = []

    with torch.no_grad():
        for inputs, target in val_loader:  # list of cuda tensor, list of cpu tensor
            torch.cuda.empty_cache()
            inputs, target = inputs.float().to(DEVICE), target

            output = model(inputs)   # N x n_classes x H x W
            pred = output.max(1)[1]  # N x H x W
            # score = output.max(1)[0]  # N x H x W

            # scores.append(score.cpu())  # n_batches of [N x H x W] tensor
            outputs.append(output.cpu())
            targets.append(target.cpu())  # n_batches of [N x H x W] tensor
            preds.append(pred.cpu())

            iou_list.extend(mean_IoU(output.cpu().numpy(), target.cpu().numpy()))

            for idx, sample_pred in enumerate(pred):
                pred_flatten = sample_pred.view(-1).cpu().numpy()
                target_flatten = target[idx].view(-1).cpu().numpy()
                f1_src = f1_score(target_flatten, pred_flatten, average='micro')
                f1_scores.append(f1_src)

    # scores = torch.cat(scores, 0).view(-1).numpy()      # (n_samples * H * W,)
    outputs = torch.cat(outputs, 0).permute(0, 2, 3, 1).contiguous().view(-1, n_classes).numpy()
    # (n_samples * H * W, n_classes)
    targets = torch.cat(targets, 0).numpy()    # (n_samples * H * W,)
    preds = torch.cat(preds, 0).numpy()   # n_samples x H x W
    np.save('targets.npy', targets)
    np.save('preds.npy', preds)
    plt.figure()
    plt.matshow(preds[5])
    plt.colorbar()
    plt.savefig('prediction.png')
    plt.close()
    plt.figure()
    plt.matshow(targets[5])
    plt.colorbar()
    plt.savefig('true.png')
    plt.close()
    print('save figures')

    targets = targets.reshape(-1)   # (n_samples * H * W,)
    # fpr, tpr, thresholds = roc_curve(targets, scores)
    # roc_auc = auc(y_true, y_score)
    # plot_roc(fpr, tpr, roc_auc, nclasses, channel)
    targets = label_binarize(targets, classes=np.arange(n_classes))  # (n_samples * H * W, n_classes)
    roc_auc = roc_auc_score(targets, outputs, average='micro')
    #roc_auc=0.0
    iou_mean, iou_std = np.mean(iou_list), np.std(iou_list)
    f1_mean, f1_std = np.mean(f1_scores), np.std(f1_scores)

    return roc_auc, (f1_mean, f1_std), (iou_mean, iou_std)


def plot_roc(fpr, tpr, roc_auc, nclasses=5, channel='RGB'):
    filename = 'roc_cur_{}_{}.png'.format(dataset, channel)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predicions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for {} classes using {} data'.format(nclasses, channel))
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


def get_in_channels(channel):
    if AVERAGE_RGB and channel == 'rgbd':
        in_channels = 2
    elif (AVERAGE_RGB and channel == 'rgb') or channel == 'd':
        in_channels = 1
    elif not AVERAGE_RGB and channel == 'rgbd':
        in_channels = 4
    elif not AVERAGE_RGB and channel == 'rgb':
        in_channels = 3
    else:
        raise ValueError('Invalid combination of AVERAGE_RGB({})  and CHANNELS({})!'.format(AVERAGE_RGB, channel))

    return in_channels


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CNN help')
    parser.add_argument('-m', '--model', dest='model', type=str, default='')
    parser.add_argument('-d', '--data', dest='data', type=str, default='small')
    parser.add_argument('-c', '--channel', dest='channel', type=str, default='rgb')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true', default=False)
    args = parser.parse_args()

    if args.data == 'mini' or args.data == 'small':
        filename = 'dataset_{}.npy'.format(args.data)
        print('Loading data from {}...'.format(filename))
        inputs = np.load(filename)
        depths = inputs[()]['depths']
        labels = inputs[()]['labels']
        images = inputs[()]['images']
        classes = inputs[()]['classes']
        print('Data loaded succeeded!')

        inputs, labels = preprocess(images, depths, labels, classes, channels=args.channel,
                                    average_rgb=AVERAGE_RGB, sample_rate=SAMPLE_RATE)
        n_classes = len(classes)

    else:
        filename = 'nyu_data.npy'
        print('Loading data from {}...'.format(filename))
        inputs = np.load(filename)
        depths = inputs[()]['depths']
        labels = inputs[()]['labels']
        images = inputs[()]['images']
        print('Data loaded succeeded!')

        classes = None
        inputs, labels = preprocess_large(images, depths, labels, classes, channels=args.channel,
                                          average_rgb=AVERAGE_RGB, sample_rate=SAMPLE_RATE)
        n_classes = 895

    #X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=TEST_SIZE)

    in_channels = get_in_channels(channel=args.channel)
    model = UNet(in_channels, n_classes).to(DEVICE)

    model_file = args.model
    model.load_state_dict(torch.load(model_file))
    print('using pretrained model from {}'.format(model_file))

    val_set = Dataset(inputs, labels)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)
    roc_auc, f1, iou = validate(model, val_loader, n_classes=n_classes)

    print('Auc: {}\tF1 score: {}\tIoU: {}'.format(roc_auc, f1, iou))


if __name__ == '__main__':
    main()
