import torch
import matplotlib.pyplot as plt
from model import model as m

from torchsummary import summary
import yaml
from pprint import pprint
import random
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms


def calculate_mean_std(dataset):
    if dataset == 'CIFAR10':
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return mean, std

def set_seed(seed,cuda_available):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed(seed)
    
    
def process_config(file_name):
    with open(file_name, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            print(" loading Configuration of your experiment ..")
            return config
        except ValueError:
            print("INVALID yaml file format.. Please provide a good yaml file")
            exit(-1)


def model_summary(model, input_size):
    result = summary(model, input_size=input_size)
    print(result)
    
def class_level_accuracy(model, loader, device, classes):

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
