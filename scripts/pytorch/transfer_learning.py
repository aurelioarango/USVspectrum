"""
Source Pytorch
License: BSD
Author: Sasank Chilamkurthy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# from optparse import OptionParser
from argparse import ArgumentParser

from PyQt5.QtWidgets import QMessageBox

def train_model( model, criterion, optimizer, scheduler, device_, data_loaders, data_set_sizes,num_epochs=25 ):
    #modelft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs = epochs
    start = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device_)
                labels = labels.to(device_)

                # zero the parameter gradients ??
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # keep statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / data_set_sizes[phase]
            epoch_acc = running_corrects.double() / data_set_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




def main(data_dir, model_name, epochs, learning_rate, save_path ):

    #####################################################
    # Load data

    data_transforms = {
        'train': transforms.Compose(
            [transforms.RandomAffine(0, scale=(1, 1.2)),
             transforms.Resize(224),
             transforms.ToTensor()
             ]
        ),
        'val': transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor()
             ]
        )
    }

    """Load Data"""
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in
                      ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Load pretrained model
    print(model_name)
    modelft = models.alexnet()
    #model = torch.load('resnet18.pth')
    if model_name =='resnet18':
        modelft=models.resnet18(pretrained=True)
    elif model_name == 'vgg19':
        models.vgg19(pretrained=True)
    else:
        modelft = models.resnet18(pretrained=True)

    modelft = modelft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(modelft.parameters(), lr=learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    # Eval the initial model on the training and val data sets
    modelft.eval()
    for phase in ['train', 'val']:
        # Iterate over data.
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = modelft(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        print('Original Model: {} Loss: {:.4f} {:.4f}'.format(phase, running_loss / dataset_sizes[phase],
                                                              running_corrects.double() / dataset_sizes[phase]))
    """Train and Evaluate"""
    epochs = int(epochs)
    print("Epochs ",epochs)
    model = train_model(modelft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=epochs)

    model_name = save_path + model_name
    print(model_name)
    torch.save(model.state_dict(),model_name)


    plt.ion()   # interactive mode


