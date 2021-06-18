import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as utils
from dataloader import albumentation


class Cifar10DataLoader:
    def __init__(self, config):
        self.config = config
        
        horizontalflip_prob = self.config['data_augmentation']['args']['horizontalflip_prob']
        rotate_limit = self.config['data_augmentation']['args']['rotate_limit']
        shiftscalerotate_prob = self.config['data_augmentation']['args']['shiftscalerotate_prob']
        num_holes = self.config['data_augmentation']['args']['num_holes']
        cutout_prob = self.config['data_augmentation']['args']['cutout_prob']
        
        train_transforms, test_transforms = albumentation.data_albumentations(horizontalflip_prob,rotate_limit,
                                                                              shiftscalerotate_prob,num_holes,cutout_prob)

                         
        trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transforms)  
        
        testset  = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(trainset, 
                                                  batch_size=self.config['data_loader']['args']['batch_size'], 
                                                  shuffle=True,
                                                  num_workers=self.config['data_loader']['args']['num_workers'], 
                                                  pin_memory=self.config['data_loader']['args']['pin_memory'])
        self.test_loader = torch.utils.data.DataLoader(testset, 
                                                 batch_size=self.config['data_loader']['args']['batch_size'],  
                                                 shuffle=False,
                                                 num_workers=self.config['data_loader']['args']['num_workers'], 
                                                 pin_memory=self.config['data_loader']['args']['pin_memory'])
             
