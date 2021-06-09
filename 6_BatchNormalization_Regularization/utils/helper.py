import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR,OneCycleLR

def experiments(train_loader, test_loader, norm_type, l1_factor, l2_factor, dropout, epochs):
    
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net(norm_type, dropout).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.7)
    scheduler = OneCycleLR(optimizer, max_lr=0.015,epochs=epochs,steps_per_epoch=len(train_loader))
    epochs = epochs

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')
        train(model, device, train_loader, optimizer, epoch, train_accuracy, train_losses, l1_factor,scheduler)
        test(model, device, test_loader,test_accuracy,test_losses)

    return train_accuracy,train_losses,test_accuracy,test_losses

def wrong_predictions(test_loader,model,device):
  wrong_images=[]
  wrong_label=[]
  correct_label=[]
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)        
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

      wrong_pred = (pred.eq(target.view_as(pred)) == False)
      wrong_images.append(data[wrong_pred])
      wrong_label.append(pred[wrong_pred])
      correct_label.append(target.view_as(pred)[wrong_pred])  
      
      wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
      print(f'Total wrong predictions are {len(wrong_predictions)}')
      
      
      fig = plt.figure(figsize=(20,25))
      for i, (img, pred, correct) in enumerate(wrong_predictions[:20]):
          img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
          ax = fig.add_subplot(10, 10, i+1)
          ax.axis('off')
          ax.set_title(f'actual {target.item()}\npredicted {pred.item()}',fontsize=15)
          ax.imshow(img.squeeze())
      plt.show()
      
  return 


