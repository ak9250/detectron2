import runway
import developset
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch, torchvision
import numpy as np
import os, json, cv2, random

  
@runway.command('InstanceSeg', inputs={'input': runway.image(description='input image')}, outputs={'output': runway.image(description='output image')})
def InstanceSeg(model, inputs):
  im = np.array(inputs['input'])
  
  outputs = developset.predictor(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(developset.cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  return out.get_image()[:, :, ::-1]

@runway.command('Keypoints', inputs={'input2': runway.image(description='input image')}, outputs={'output2': runway.image(description='output image')})
def Keypoints(model, inputs):
  im = np.array(inputs['input2'])
  outputs = developset.predictorkey(im)
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(developset.cfgkey.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  return out.get_image()[:, :, ::-1]

@runway.command('PanopticSeg', inputs={'input3': runway.image(description='input image')}, outputs={'output3': runway.image(description='output image')})
def PanopticSeg(model, inputs):
  im = np.array(inputs['input3'])
  panoptic_seg, segments_info = developset.predictorpan(im)["panoptic_seg"]
  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(developset.cfgpan.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
  return out.get_image()[:, :, ::-1]

if __name__ == '__main__':
  runway.run()