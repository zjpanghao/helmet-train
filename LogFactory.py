import logging

class Logger:
  def __init__(self, name):
    self.logger = logging.getLogger(name)
    self.logger.setLevel(logging.INFO)
    fh = logging.FileHandler(name + '.txt')
    self.logger.addHandler(fh)
  def getLogger(self):
    return self.logger
#logger = Logger("abc").getLogger()
#logger.info("aaaa")
