import runway
import developset
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch, torchvision
import numpy as np
import os, json, cv2, random

 
def predict(img):
    outputs = predictor(img)['instances']
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    data = extractor(outputs)
    image_vis = visualizer.visualize(image, data)
    return image_vis

  
@runway.command('Densepose', inputs={'input5': runway.image}, outputs={'output5': runway.image})
def visualize(model, inputs):
  os.chdir('/model/projects/DensePose/')
  from typing import ClassVar, Dict

  from detectron2.config import get_cfg
  from detectron2.structures.instances import Instances
  from detectron2.engine.defaults import DefaultPredictor

  from densepose import add_densepose_config
  from densepose.vis.base import CompoundVisualizer
  from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
  from densepose.vis.extractor import CompoundExtractor, create_extractor

  from densepose.vis.densepose_results import (
      DensePoseResultsContourVisualizer,
      DensePoseResultsFineSegmentationVisualizer,
      DensePoseResultsUVisualizer,
      DensePoseResultsVVisualizer,
  )

  cfg = get_cfg()
  add_densepose_config(cfg)

  cfg.merge_from_file("configs/densepose_rcnn_R_50_FPN_s1x.yaml")
  cfg.MODEL.DEVICE = "cuda"
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 

  cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
  predictor = DefaultPredictor(cfg)

  VISUALIZERS: ClassVar[Dict[str, object]] = {
      "dp_contour": DensePoseResultsContourVisualizer,
      "dp_segm": DensePoseResultsFineSegmentationVisualizer,
      "dp_u": DensePoseResultsUVisualizer,
      "dp_v": DensePoseResultsVVisualizer,
      "bbox": ScoredBoundingBoxVisualizer,
  }

  vis_specs = ['dp_contour', 'bbox']
  visualizers = []
  extractors = []
  for vis_spec in vis_specs:
      vis = VISUALIZERS[vis_spec]()
      visualizers.append(vis)
      extractor = create_extractor(vis)
      extractors.append(extractor)
  visualizer = CompoundVisualizer(visualizers)
  extractor = CompoundExtractor(extractors)

  context = {
      "extractor": extractor,
      "visualizer": visualizer
  }

  visualizer = context["visualizer"]
  extractor = context["extractor"]
  im = np.array(inputs['input5'])
  out = predict(im[:, :, ::-1])
  return out
  
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