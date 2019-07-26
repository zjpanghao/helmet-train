import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import copy
import sys
#sys.path.append("../td")
data_transforms = {
  'train': transforms.Compose([
      transforms.Resize(299),
      transforms.CenterCrop(299),
#transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
      #transforms.RandomRotation(30),
      transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  'val': transforms.Compose([
      transforms.Resize(299),
      transforms.CenterCrop(299),
      #transforms.RandomRotation(30),
      transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
}

data_dir = 'db'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=2,
  shuffle=True, num_workers=1)
  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    if best_acc > 0.999:
      break;
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        scheduler.step()
        model.train()  # Set model to training mode
      else:
        model.eval()   # Set model to evaluate mode

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        
        with torch.set_grad_enabled(phase == 'train'):
          try:
            outputs = model(inputs)
          except:
            continue
          #print (outputs)
          if phase == 'train':
            _, preds = torch.max(outputs[0], 1)
            loss = criterion(outputs[0], labels)
            loss += 0.4 * criterion(outputs[1], labels)
          else:
            _, preds = torch.max(outputs, 1)
          # backward + optimize only if in training phase
          if phase == 'train':
            loss.backward()
            optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.double() / dataset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))

  # load best model weights
  model.load_state_dict(best_model_wts)
  return model
print (device)
# init model
model_ft = models.inception_v3(pretrained=False)
num_ftrs = model_ft.fc.in_features
aux_ftrs = model_ft.AuxLogits.fc.in_features
print(num_ftrs, aux_ftrs)
model_ft.fc = nn.Linear(num_ftrs, 4)
model_ft.AuxLogits.fc = nn.Linear(aux_ftrs, 4)
print  (image_datasets['train'].classes)
#print (model_ft)
model_ft = model_ft.to(device)
if os.path.exists("helmet_inception_v3_stat.ft"):
  model_ft.load_state_dict(torch.load("helmet_inception_v3_stat.ft", map_location=device))
#train
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=5)
torch.save(model_ft.state_dict(), "helmet_inception_v3_stat.ft")
