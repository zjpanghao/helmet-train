import tornado.web 
import tornado.ioloop 
import torch
from PIL import Image
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.nn as nn
import os
import shutil
import time
import sys
import base64
from io import BytesIO
import json
import io
from concurrent.futures import ThreadPoolExecutor
import logging
import LogFactory
logger = LogFactory.Logger("helmet").getLogger()
imsize=299
chars = ["none", "with"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.inception_v3(pretrained=True)
num_ftrs = model_ft.fc.in_features
aux_ftrs = model_ft.AuxLogits.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(chars))
model_ft.AuxLogits.fc = nn.Linear(aux_ftrs, len(chars))
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load("helmet/helmet_inception_v3_stat.ft", map_location=device))
model_ft.eval()   # Set model to evaluate mode
executor = ThreadPoolExecutor(max_workers=10)
loader = transforms.Compose([
    transforms.Resize(int(imsize*1)),  # scale imported image
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # transform it into a torch tensor

def image_toRGB(image):
  if image.mode == 'RGBA':
    r,g,b,a = image.split()
    image = Image.merge("RGB", (r, g, b))
  elif image.mode != 'RGB':
    image = image.convert("RGB") 
  return image

def image_loader(bufferimage):
  image = loader(bufferimage).unsqueeze(0)
  return image.to(device, torch.float)

class VersionHandler(tornado.web.RequestHandler):
  def get(self):
    self.write(ver.getVersion())


class IndexHandler(tornado.web.RequestHandler): 

  def saveFile(self, fileName, image, score):
      low = servDb.getScore()
      if score > low:
        return
      logger.info(fileName)
      image.save(fileName, "jpeg")

  def post(self): 
    body = self.request.body
    obj=json.loads(body)
    image64 = obj["image"]
    bufferImage = base64.b64decode(image64)
    image = Image.open(io.BytesIO(bufferImage))
    image = image_toRGB(image)
    data=image_loader(image)
    error_code=0
    index=0
    park="unknown"
    score=0
    res = {"error_code" : -1, "results" : [{"name":"unknown", "score": 0}]}
    try:
      pred=model_ft(data)
      p= F.softmax(pred, dim=1)
      va,inx = torch.max(p, 1)
      index = inx.item()
      score = round(va.item(), 2)
      park = chars[index]
      res = {"error_code" : 0, "results" : [{"name": park, "score": score}]}
      #fileName = '/home/panghao/ty/low/' + park + "/" + str(time.time()) + "_" + str(score) + "_.jpg"
      #executor.submit(self.saveFile, fileName, image, (int)(score*100))
    except Exception as  e:
      logger.error('exeption' + str(e))
    finally:
      result = json.dumps(res)
      logger.info(result)
      self.write(result) 
if __name__ == '__main__': 
	app = tornado.web.Application([
      (r'/helmet/detect',IndexHandler)
      ]
      ) 
	app.listen(9098)
	tornado.ioloop.IOLoop.current().start()
