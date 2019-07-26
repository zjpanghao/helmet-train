import torch
from PIL import Image
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.nn as nn
import os
import shutil
import time
import sys
imsize=299
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loader = transforms.Compose([
    transforms.Resize((int)(imsize * 1.3)),  # scale imported image
    transforms.CenterCrop(299),
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
  image = Image.open(image_name)
  if image.mode == 'RGBA':
    r,g,b,a = image.split()
    image = Image.merge("RGB", (r, g, b))
  elif image.mode != 'RGB':
    print (image.mode)
    image = image.convert("RGB") 
  # fake batch dimension required to fit network's input dimensions
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)
filename=sys.argv[1]
model_ft = models.inception_v3(pretrained=False)
num_ftrs = model_ft.fc.in_features
aux_ftrs = model_ft.AuxLogits.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 3)
model_ft.AuxLogits.fc = nn.Linear(aux_ftrs, 3)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load("helmet_inception_v3_stat.ft", map_location=device))
model_ft.eval()   # Set model to evaluate mode
chars = ["land", "none", "wich"]
data=image_loader(filename).to(device)
pred=model_ft(data)
print (pred)
p= F.softmax(pred, dim=1)
va,inx = torch.max(p, 1)
print (inx.item())
print (chars[inx.item()], va.item())
