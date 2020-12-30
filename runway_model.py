import runway
import developset
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch, torchvision
import numpy as np
import os, json, cv2, random

  
@runway.command('visualize', inputs={'input': runway.image}, outputs={'output': runway.image})
def visualize(model, inputs):
  im = np.array(inputs['input'])
  
  outputs = developset.predictor(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(developset.cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  return out.get_image()[:, :, ::-1]

@runway.command('visualize2', inputs2={'input2': runway.image}, outputs={'output2': runway.image})
def visualize2(model, inputs2):
  im = np.array(inputs2['input2'])
  outputs = developset.predictorkey(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(developset.cfgkey.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  return out.get_image()[:, :, ::-1]

if __name__ == '__main__':
  runway.run()