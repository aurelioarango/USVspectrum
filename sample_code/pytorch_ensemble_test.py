import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn import metrics
import time
import os
import copy
import shutil

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

data_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

data_dir = '../../data/'
batch_size = 6
num_classes = 4
test_images = ImageFolderWithPaths(os.path.join(data_dir, 'test'), transform=data_transform)
dataloader = torch.utils.data.DataLoader(test_images, batch_size=batch_size)

class_names = test_images.classes
print(class_names)
print("test images #: ", len(test_images))
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

#model_names = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn', 'inception_v3']
#model_names = ['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19']
model_names = ['resnet18', 'vgg16']
model_weights = [0.65, 0.35]
models = []

##########################################################
# load pretrained models

for name in model_names:
    path = '../model/' + name + '_4_classes.h5'
    model = torch.load(path)
    model.eval()
    model.to(device)
    models.append(model)

############################################################
# Create output folders
def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        print(os.path.join(*path))
        os.makedirs(os.path.join(*path))

output_dir = "/home/xiaoyu/thesis_workspace/output" #"../../output"
mkdir_if_not_exist([output_dir])
for cls in class_names:
    mkdir_if_not_exist([output_dir, cls])
    for c2 in class_names:
        mkdir_if_not_exist([output_dir, cls, c2])


############################################################
# Iterate through the test dataset
start = time.time()

all_labels = np.array([]) 
all_preds = np.array([])
for inputs, labels, paths in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    print(paths)
    in_size = [*inputs.size()]
    #print(in_size[0])
    probs = np.zeros((in_size[0], num_classes))
    for model, weight in zip(models, model_weights):
        outputs = model(inputs)
        #print(outputs)
        outputs = softmax(outputs.detach().numpy(), axis=1)	
        probs += outputs * weight
        preds = np.argmax(outputs, 1)
        print(preds, labels)
    ensemble_preds = np.argmax(probs, 1)
    for c1, c2, path in zip(labels, ensemble_preds, paths):
        print(path, class_names[c1], class_names[c2])
        shutil.copy(path, os.path.join(output_dir, class_names[c1], class_names[c2]))

    print('overall preds: ', ensemble_preds)
    print('----------------------------------------------')
    print()
    all_preds = np.append(all_preds, ensemble_preds)
    all_labels = np.append(all_labels, labels.numpy())

time_elapsed = time.time() - start
print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

confusion = metrics.confusion_matrix(all_labels, all_preds)
print(confusion)
print("testing accuracy = ", metrics.accuracy_score(all_labels, all_preds))
