
from torchvision import datasets, models

import torch

import torchvision
from torchvision import transforms

import csv
import numpy
import os



def load_model( PATH):
    model = models.resnet18()
    #print("LoadModel:PATH: ",PATH)
    #model.load_state_dict(torch.load(PATH))
    #torch.device("cpu")
    model.load_state_dict(torch.load(PATH, map_location='cpu'), strict=False)
    #model = torch.load(PATH, map_location='cpu')
    #model.cpu()

    return model

def load_test(path):
    data_transforms = transforms.Compose([transforms.ToTensor()])
    #print("path: ", path)
    try:

        test_data = datasets.ImageFolder(os.path.join( path, 'test'), transform=data_transforms)
        #test_data = datasets.ImageFolder(path, transform = transforms.Compose([transforms.ToTensor()]))
    except FileNotFoundError:
        print("Could not find 'Test' directory")
    #test_data = ImagePathFolder(path, transform = transforms.Compose([transforms.ToTensor()]) )

    return test_data


def evaluate(test_data, model):
    print(test_data)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False )
    classes  = ('FF', 'FM', 'Noise', 'Trills')
    # to save to file
    #to_file = []
    #to_file.append('Image Path, Predicted')
    device = torch.device("cpu")

    model.eval()

    #print(testloader)
    #print(testloader[0])
    for data in testloader:
    #    print(len(data))
        images, labels = data

        #print(len(images))
        #print(labels)
        if len(labels) >= 1 :
                #print('groundTruth: ', ' '.join('%6s' % classes[labels[j]] for j in range(3)))
                # to_file.append('groundTruth: ', ' '.join('%6s' % classes[labels[j]] for j in range(3)))
            outputs = model(images)
                # print('images ',images) # actual images
            _, predicted = torch.max(outputs, 1)
                # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(3)))
            #out = 'predicted: {} '.format(classes[predicted[0]])
            #print (predicted[0].argmax() )
            #out = classes[labels[0]] +',' + classes[predicted[0]]
            #to_file.append(out)
            #print(out)
            print(predicted)

        #else:
           # print(path)
            #print(classes[labels[0]])
    # print(to_file[0])
    #numpy.savetxt('predictions.txt',to_file,fmt='%6s', delimiter=',')"""



