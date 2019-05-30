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

#data_dir = 'E:\\data\\USV_DATA\\data_512\\full'
data_dir = '/media/lio/ZEUS/data/USV_DATA/data_512/full'

#arc_data_dir = 'E:\\data\\USV_DATA\\archived_data'
arc_data_dir = '/media/lio/ZEUS/data/USV_DATA/archived_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x]) for x in
                  ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

print(dataset_sizes)
print('{} classes: {}'.format(num_classes, class_names))


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    # Imshow for Tensor.
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
print(inputs.shape, classes.shape)
inputs = inputs.narrow(0, 0, 4)
classes = classes.narrow(0, 0, 4)
print(inputs[0].shape)

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


######################################################################
#
# get a pretrained torchvision model by name
#
def get_tv_model(name):
    model = None
    if name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        print("{} fc layer # input features: {}".format(name, num_ftrs))
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif name == 'resnet34':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features
        print("{} fc layer # input features: {}".format(name, num_ftrs))
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        print("{} fc layer # input features: {}".format(name, num_ftrs))
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif name == 'resnet101':
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        print("{} fc layer # input features: {}".format(name, num_ftrs))
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif name == 'resnet152':
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        print("{} fc layer # input features: {}".format(name, num_ftrs))
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    elif name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Linear(512 * 7 * 7, num_classes)
        return model
    elif name == 'vgg19':
        model = models.vgg19(pretrained=True)
        model.classifier = nn.Linear(512 * 7 * 7, num_classes)
        return model
    elif name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        model.classifier = nn.Linear(512 * 7 * 7, num_classes)
        return model
    elif name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
        model.classifier = nn.Linear(512 * 7 * 7, num_classes)
        return model
    elif name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        print("{} fc layer # input features: {}".format(name, num_ftrs))
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    return model


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#
# model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn', 'inception_v3']
model_names = ['resnet18']

for m in model_names:
    model_ft = get_tv_model(m)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # minute.
    #

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)

    # visualize_model(model_ft)

    # Save the model to a file

    path = '/media/lio/ZEUS/USV_DATA/data_512/models/old_data_96_v1_resnet18.h5'
    #path = 'E:\\data\\USV_DATA\\data_512\\models\\old_data_96_v1_resnet18.h5'

    #path = '../model/' + m + '_4_classes.h5'
    torch.save(model_ft, path)

'''
for x in ['train', 'val']:
    pop_mean = []
    pop_std = []
    pop_min = []
    pop_max = []
    for i, data in enumerate(dataloaders[x], 0):
        images, labels = data
        numpy_image = images.numpy()
        #print(numpy_image.shape)

        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std = np.std(numpy_image, axis=(0, 2, 3))
        batch_max = np.max(numpy_image, axis=(0, 2, 3))
        batch_min = np.min(numpy_image, axis=(0, 2, 3))

        #print(x, i, 'batch mean: {}'.format(batch_mean))
        #print(x, i, 'batch std: {}'.format(batch_std))
        #print(x, i, 'batch max: {}'.format(batch_max))
        #print('{} {}th batch min {}'.format(x, i, batch_min))

        pop_mean.append(batch_mean)
        pop_std.append(batch_std)
        pop_max.append(batch_max)
        pop_min.append(batch_min)

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)
    pop_min = np.array(pop_min).mean(axis=0)
    pop_max = np.array(pop_max).mean(axis=0)

    print(x, 'pop mean: {}'.format(pop_mean))
    print(x, 'pop std: {}'.format(pop_std))
    print('{} pop max: {}'.format(x, pop_max))
    print('{} pop min: {}'.format(x, pop_min))
'''
