
from torchvision import datasets, models

import torch

import torchvision
from torchvision import transforms

import csv
import numpy
import os


def load_model( PATH):
    model = models.resnet18()
    print("LoadModel:PATH: ",PATH)
    model.load_state_dict(torch.load(PATH))
    return model

def load_test(path):
    data_transforms = {
        'test': transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor()
             ]
        )
    }
    print("path: ", path)
    try:

        test_data = datasets.ImageFolder(path, transform=data_transforms )

    except FileNotFoundError:
        print("Could not find 'Test' directory")
    #test_data = ImagePathFolder(path, transform = transforms.Compose([transforms.ToTensor()]) )

    return test_data


def evaluate(test_data, model):
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    classes  = ('FF', 'FM', 'Noise', 'Trills')
    # to save to file
    to_file = []
    to_file.append('Image Path, Predicted')
    for data in testloader:
    # print(len(data))
        images, labels, path = data
        
        if len(labels) >= 1 :
                #print('groundTruth: ', ' '.join('%6s' % classes[labels[j]] for j in range(3)))
                # to_file.append('groundTruth: ', ' '.join('%6s' % classes[labels[j]] for j in range(3)))
            outputs = model(images)
                # print('images ',images) # actual images
            _, predicted = torch.max(outputs, 1)
                # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(3)))
            out = path[0]+', ' + classes[labels[0]] +',' + classes[predicted[0]]
            to_file.append(out)
            print(out)
           
        #else:
           # print(path)
            #print(classes[labels[0]])
    # print(to_file[0])
    numpy.savetxt('predictions.txt',to_file,fmt='%6s', delimiter=',') 


