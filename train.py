import copy
import os
from os.path import join
from pathlib import Path

import albumentations as A
import detectron2.data.transforms as T
import numpy as np
import torch
# import some common detectron2 utilities
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import BestCheckpointer
from detectron2.solver.lr_scheduler import WarmupParamScheduler
from detectron2.utils.logger import setup_logger
from fvcore.common.param_scheduler import (CompositeParamScheduler,
                                           CosineParamScheduler,
                                           LinearParamScheduler)
from PIL import Image

from src.copypaste import CopyPasteAugmentator
from src.evaluator import MAPIOUEvaluator
from src.swin.swint.config import add_swinb_config

setup_logger()

class CustomDatasetMapper:
  def __init__(self, cfg):
    self.copypaste_augmentator = CopyPasteAugmentator(
        DatasetCatalog.get(cfg.DATASETS.TRAIN[0]),
        paste_same_class=True,
        paste_density=cfg.CUSTOM_MAPPER.PASTE_DENSITY,
        filter_area_thresh=0.1,
        p=cfg.CUSTOM_MAPPER.COPYPASTE_PROB,
    )

    self.min_size_train = cfg.INPUT.MIN_SIZE_TRAIN
    self.max_size_train = cfg.INPUT.MAX_SIZE_TRAIN

    # See "Data Augmentation" tutorial for details usage
    self.augs = T.AugmentationList([
      T.ResizeShortestEdge(
          short_edge_length=self.min_size_train,
          max_size=self.max_size_train,
          sample_style='choice', interp=Image.BICUBIC),
      T.RandomFlip(prob=0.5, vertical=True, horizontal=False),
      T.RandomFlip(prob=0.5, vertical=False, horizontal=True),
    ])

    # Non-geometric transformations
    self.albu_transform = A.Compose([
      A.CLAHE(p=0.1),
      A.Blur(p=0.1),
      A.MedianBlur(p=0.1),
      A.MotionBlur(p=0.1),
      A.RandomBrightnessContrast(p=0.1),
    ])

  # Show how to implement a minimal mapper, similar to the default DatasetMapper
  def __call__(self, dataset_dict):
      dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
      
      # can use other ways to read image
      image, dataset_dict = self.copypaste_augmentator(dataset_dict) # already have deepcopy inside

      auginput = T.AugInput(image)
      transform = self.augs(auginput)
      image = auginput.image
      image = self.albu_transform(image=image)['image']

      image_shape = image.shape[:2]  # h, w
      annos = [
          utils.transform_instance_annotations(annotation, [transform], image_shape)
          for annotation in dataset_dict.pop("annotations")
      ]

      dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype(np.float32))
      dataset_dict['instances'] = utils.annotations_to_instances(
          annos, image_shape, mask_format="bitmask")
      return dataset_dict


def run(root:str):
    dataDir=Path(join(root, "data"))
    cfg = get_cfg()
    add_swinb_config(cfg)
    config_name = join(root, "detectron_sartorius", "lib", "swin", "configs", "SwinT", "mask_rcnn_swint_T_FPN_3x.yaml")
    cfg.merge_from_file(config_name)

    class Trainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg):
            return build_detection_train_loader(cfg, 
                mapper = CustomDatasetMapper(cfg))

        @classmethod
        def build_lr_scheduler(cls, cfg, optimizer):
            step_iter = 2000
            warmup_iter = cfg.SOLVER.MAX_ITER - step_iter
            steps = warmup_iter // 5000
            step_scheduler = LinearParamScheduler(start_value = 1.5, end_value=0.75) #MultiStepParamScheduler(values=[1, 0.1, 0.01], num_updates=500)
            warmup_scheduler = CompositeParamScheduler(
                        schedulers = [WarmupParamScheduler(
                                        scheduler = CosineParamScheduler((0.75**x),
                                                                         0.05*(0.75**x)),
                                        warmup_length = 10 / cfg.SOLVER.MAX_ITER,
                                        warmup_method = "linear",
                                        warmup_factor = 0.05)
                                        for x in range(steps)],
                        lengths = [1/steps for _ in range(steps)],
                        interval_scaling = ['rescaled' for _ in range(steps)],
                        )

            return CompositeParamScheduler(schedulers = [step_scheduler, warmup_scheduler],
                                            lengths = [step_iter/cfg.SOLVER.MAX_ITER, warmup_iter/cfg.SOLVER.MAX_ITER],
                                            interval_scaling=['rescaled', 'rescaled'])

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return MAPIOUEvaluator(dataset_name)


    
    cfg.INPUT.MASK_FORMAT='bitmask'
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = join(root, "detectron_sartorius", "output", "cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv", "pretrain", "cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv_best.pth") #model_zoo.get_checkpoint_url(config_name)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.01 
    cfg.SOLVER.MAX_ITER = 12000 #Maximum of iterations 1     
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    cfg.SOLVER.AMP.ENABLED = False
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "ciou"
    cfg.CUSTOM_MAPPER = CN()
    cfg.CUSTOM_MAPPER.PASTE_DENSITY = [0.3, 0.6]
    cfg.CUSTOM_MAPPER.COPYPASTE_PROB = .5

    cfg.TEST.EVAL_PERIOD = 250  # Once per epoch

    cfg.TEST.CHECKPOINT_PERIOD = 250
    
    for fold in range(4, 5): 

        class FoldTrainer(Trainer):
            def build_hooks(self):
                ret = super().build_hooks()
                ret.append(BestCheckpointer(cfg.TEST.EVAL_PERIOD,
                                            self.checkpointer,
                                            'MaP IoU',
                                            file_prefix=f'{os.path.basename(config_name).rstrip(".yaml")}_best_{fold}'))
                return ret
        
        register_coco_instances(f'sartorius_train_{fold}',{}, join(root, "data", "annotations", f'/coco_cell_train_fold{fold}.json', dataDir/"train")
        register_coco_instances(f'sartorius_val_{fold}',{}, join(root, "data", "annotations", f'/coco_cell_valid_fold{fold}.json', dataDir/"train")
        cfg.DATASETS.TRAIN = (f"sartorius_train_{fold}",)
        cfg.DATASETS.TEST = (f"sartorius_val_{fold}",)
        
        cfg.OUTPUT_DIR = f'./output/{os.path.basename(config_name).rstrip(".yaml")}_v2/{fold}/'
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        trainer = FoldTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()

        torch.cuda.empty_cache()

if __name__ == '__main__':
    root = "/home/nmark/kaggle/sartorius/"
    run(root)