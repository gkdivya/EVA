import torchvision
from dataloader.load_data import Cifar10DataLoader
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from model import model as m
from utils import train as trn
from utils import test as tst
from torchsummary import summary
import yaml
from pprint import pprint
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


class TriggerEngine:
    def __init__(self, config):
        self.config = config
        self.dataset=Cifar10DataLoader(self.config)
        self.device = self.set_device()
        
        
    def set_device(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device
        
    def run_experiment(self):
            
        dropout=self.config['model_params']['dropout']
        epochs=self.config['training_params']['epochs']
        l2_factor = self.config['training_params']['l2_factor']
        l1_factor = self.config['training_params']['l1_factor']
        
        criterion = nn.CrossEntropyLoss() if self.config['criterion'] == 'CrossEntropyLoss' else F.nll_loss()
        opt_func = optim.Adam if self.config['optimizer']['type'] == 'optim.Adam' else optim.SGD
        lr = self.config['optimizer']['args']['lr']
        
        grad_clip = 0.1
            
        train_losses = []
        test_losses = []
        train_accuracy = []
        test_accuracy = []
        lrs=[]
            
        #device = self.set_device()
            
        model = m.Net(dropout).to(self.device)
        # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.7,weight_decay=l2_factor)
        optimizer = opt_func(model.parameters(), lr=lr, weight_decay=l2_factor)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True,mode='max')
        scheduler = OneCycleLR(optimizer, max_lr=lr,epochs=epochs,steps_per_epoch=len(self.dataset.train_loader))

        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}:')
            trn.train(model, self.device, self.dataset.train_loader, optimizer,epoch, train_accuracy, train_losses, l1_factor,scheduler,criterion,lrs,grad_clip)

            tst.test(model, self.device, self.dataset.test_loader,test_accuracy,test_losses,criterion)
            # if epoch > 20:
            #     scheduler.step(test_accuracy[-1])

        
        return (train_accuracy,train_losses,test_accuracy,test_losses),model
        
    def save_experiment(self,model, experiment_name):
        print(f"Saving the model for {experiment_name}")
        torch.save(model, './saved_models/{}.pt'.format(experiment_name))
    
    def model_summary(self,model, input_size):
        result = summary(model, input_size=input_size)
        print(result)    
        
    def wrong_predictions(self,model):
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        wrong_images=[]
        wrong_label=[]
        correct_label=[]
        with torch.no_grad():
            for data, target in self.dataset.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)        
                pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])  
      
                wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
            print(f'Total wrong predictions are {len(wrong_predictions)}')
      
      
            fig = plt.figure(figsize=(10,12))
            fig.tight_layout()
            # mean,std = helper.calculate_mean_std("CIFAR10")
            for i, (img, pred, correct) in enumerate(wrong_predictions[:10]):
                  img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
        
                  #mean = torch.FloatTensor(mean).view( 3, 1, 1).expand_as(img).cpu()
                  #std = torch.FloatTensor(std).view( 3, 1, 1).expand_as(img).cpu()
                  #img = img.mul(std).add(mean)
                  #img=img.numpy()
                  
                  img = np.transpose(img, (1, 2, 0)) / 2 + 0.5
                  ax = fig.add_subplot(5, 5, i+1)
                  ax.axis('off')
                  ax.set_title(f'\nactual : {class_names[target.item()]}\npredicted : {class_names[pred.item()]}',fontsize=10)  
                  ax.imshow(img)  
          
            plt.show()
      
        return 
