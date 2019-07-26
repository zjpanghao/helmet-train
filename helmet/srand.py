import random
import shutil
import os
import time
def checkVal():
	val = random.random()
	if val > 0.7:
		return True
	return False
ssrc="../data"
strain="db/train"
sval="db/val"

def removeFile(destdir):
  for name in os.listdir(destdir):
      destname = os.path.join(destdir, name)
      for fname in os.listdir(destname):
        destfile = os.path.join(destname, fname)
        os.remove(destfile)
def cprand(src, train, val):
  for name in os.listdir(src):
    srcname = os.path.join(src, name)
    for fname in os.listdir(srcname):
      if checkVal():
        destname = os.path.join(val, name)
      else:
          destname = os.path.join(train, name)
      destfile = os.path.join(destname, fname)
      srcfile = os.path.join(srcname, fname)
      shutil.copyfile(srcfile, destfile)
random.seed(time.time())
removeFile(strain)
removeFile(sval)
cprand(ssrc, strain, sval)
