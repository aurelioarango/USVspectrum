
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
    model = torch.load(PATH, map_location='cpu')
    return model

def load_test(path):
    """data_transforms = {
        'test': transforms.Compose(
            [transforms.Resize(224),
             transforms.ToTensor()
             ]
        )
    }"""
    #print("path: ", path)
    try:

        #test_data = {x: datasets.ImageFolder(path, transform=data_transforms[x])for x in ['test']}
        test_data = datasets.ImageFolder(path, transform = transforms.Compose([transforms.ToTensor()]))
    except FileNotFoundError:
        print("Could not find 'Test' directory")
    #test_data = ImagePathFolder(path, transform = transforms.Compose([transforms.ToTensor()]) )

    return test_data


def evaluate(test_data, model):
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False )
    classes  = ('FF', 'FM', 'Noise', 'Trills')
    # to save to file
    #to_file = []
    #to_file.append('Image Path, Predicted')
    device = torch.device("cpu")

    model.eval()

    """with torch.no_grad():
        for i, (inputs, labels) in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            out =  classes[labels[0]] + ',' + classes[predicted[0]]

            print(out)
            #for j in range(inputs.size()[0]):
                #images_so_far += 1
                #ax = plt.subplot(num_images // 2, 2, images_so_far)
                #ax.axis('off')
                #ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                #imshow(inputs.cpu().data[j])"""

    #print(testloader)
    #print(testloader[0])
    for data in testloader:
    #    print(len(data))
        images, labels = data

        print(len(images))
        print(labels)
        if len(labels) >= 1 :
                #print('groundTruth: ', ' '.join('%6s' % classes[labels[j]] for j in range(3)))
                # to_file.append('groundTruth: ', ' '.join('%6s' % classes[labels[j]] for j in range(3)))
            outputs = model(images)
                # print('images ',images) # actual images
            _, predicted = torch.max(outputs, 1)
                # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(3)))
            out = classes[labels[0]] +',' + classes[predicted[0]]
            #to_file.append(out)
            print(out)
           
        #else:
           # print(path)
            #print(classes[labels[0]])"""
    # print(to_file[0])
    #numpy.savetxt('predictions.txt',to_file,fmt='%6s', delimiter=',')


