import torch
from torchvision import datasets, transforms


def data_augmentation():
    # Train Phase transformations
    train_transforms = transforms.Compose([
                                       transforms.RandomRotation((-6.0, 6.0), fill=(1,)),                
                                       transforms.RandomAffine(degrees=7, shear=7, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                                       transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.40, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

    return train_transforms, test_transforms
    

def download_mnist_data(train_transforms, test_transforms):
    
    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)
    return train, test
    
def dataloader(train_data, test_data,data_loader_args):
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, **data_loader_args)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, **data_loader_args)

    return train_loader, test_loader
