import torch

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


