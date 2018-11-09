import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

from model import UNet

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cpu'
N_WORKERS = 4
N_EPOCHS = 10
BATCH_SIZE = 4
LR = 0.001


class Dataset(data.Dataset):
    def __init__(self, inputs, labels=None):
        """
        Initialization
        :param inputs: N x 3 x H x W
        :param labels: N x H x W
        """
        self.inputs = inputs
        self.labels = labels
        self.n_pixels = self.inputs.shape[2] * self.inputs.shape[3]  # H * W

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inputs)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        if self.labels is not None:
            return self.inputs[idx].astype('float'), self.labels[idx].astype('int32')
        else:
            return self.inputs[idx].astype('float')


def train(model, optimizer, train_loader, epoch):
    log_interval = len(train_loader) // 10
    criterion = nn.CrossEntropyLoss()
    model.train()
    epoch_loss = 0
    correct = 0

    for batch_id, (inputs, target) in enumerate(train_loader):
        torch.cuda.empty_cache()
        inputs, target = inputs.float().to(DEVICE), target.long().to(DEVICE)

        optimizer.zero_grad()
        output = model(inputs)  # N x n_classes x H x W

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        pred = output.max(1, keepdim=True)[1]     # N x 1 x H x W
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(inputs), len(train_loader.dataset),
                       100. * batch_id / len(train_loader), epoch_loss / (batch_id + 1) / len(inputs)))

    train_loss = epoch_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset) / train_loader.dataset.n_pixels
    print("Training loss per sample: {}".format(train_loss))
    print('Accuracy: {}'.format(train_acc))

    return train_loss, train_acc


def validation(model, val_loader):
    print('Validating...')
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for inputs, target in val_loader:  # list of cuda tensor, list of cpu tensor
            torch.cuda.empty_cache()
            inputs, target = inputs.float().to(DEVICE), target.to(DEVICE)

            output = model(inputs)   # N x n_classes x H x W
            val_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]    # N x 1 x H x W
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset) / val_loader.dataset.n_pixels
    print("Validation loss per sample: {}".format(val_loss))
    print('Accuracy: {}'.format(acc))

    return val_loss, val_acc


def plot(train_losses, train_accs, val_losses, val_accs, n_epochs):
    epochs = range(1, n_epochs + 1)

    plt.figure()
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.savefig('Loss_vs_epochs.png')

    plt.figure()
    plt.plot(epochs, train_accs)
    plt.plot(epochs, val_accs)
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper right')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.savefig('Acc_vs_epochs.png')


def preprocess(images, depths, labels, sample_size=0.2):
    """

    :param images: H x W x 3 x N
    :param depths: H x W x N
    :param labels: H x W x N
    :return: [N x 4 x H x W] data, [N x H x W] labels
    """
    images = np.transpose(images, (3, 2, 0, 1))  # N x 3 x H x W
    depths = np.transpose(depths, (2, 0, 1))  # N x H x W
    depths = np.expand_dims(depths, 1)  # N x H x W
    inputs = np.concatenate((images, depths), axis=1)  # N x 4 x H x W
    labels = np.transpose(labels, (2, 0, 1))  # N x H x W

    n = len(inputs)
    idx = math.floor(sample_size * n)
    return inputs[:idx], labels[:idx]


if __name__ == '__main__':
    filename = 'nyu_data.npy'
    print('Loading data...')
    inputs = np.load(filename)
    depths = inputs[()]['depths']
    labels = inputs[()]['labels']
    images = inputs[()]['images']
    print('Data loaded succeeded!')

    inputs, labels = preprocess(images, depths, labels)
    X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.1)

    in_channels, classes = 4, 895
    model = UNet(in_channels, classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)

    train_set = Dataset(X_train, y_train)
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
    val_set = Dataset(X_val, y_val)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=N_WORKERS)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train(model, optimizer, train_loader, epoch)
        torch.save(model.state_dict(), '{}epoch{}.pt'.format('1st_version', epoch))

        val_loss, val_acc = validate(model, validation_loader)
        scheduler.step(val_acc)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    plot(train_losses, train_accs, val_losses, val_accs, N_EPOCHS)
