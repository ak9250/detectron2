import cv2
import numpy as np
import runway
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

 
def predict(img):
    outputs = predictor(img)['instances']
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    data = extractor(outputs)
    image_vis = visualizer.visualize(image, data)
    return image_vis

  
@runway.command('visualize', inputs={'input': runway.image}, outputs={'output': runway.image})
def visualize(model, inputs):
  im = np.array(inputs['input'])
  out = predict(im[:, :, ::-1])
  return out


if __name__ == '__main__':
  runway.run()