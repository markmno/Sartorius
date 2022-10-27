import cv2
import copy
import pycocotools.mask as mask_util
import numpy as np


class CopyPasteAugmentator:
  """Copy-paste cells from another image in the dataset
  """
  def __init__(self, d2_dataset,
               paste_same_class=True,
               paste_density=[0.3, 0.6],
               filter_area_thresh=0.1,
               p=1.0):
    self.data = d2_dataset
    self.n_samples = len(d2_dataset)
    self.paste_same_class = paste_same_class
    if paste_same_class:
      self.cls_indices = [
        [
          i for i, item in enumerate(d2_dataset)
          if item['annotations'][0]['category_id'] == cls_index
        ]
        for cls_index in range(3)
      ]
    self.filter_area_thresh = filter_area_thresh
    self.paste_density = paste_density
    self.p = p
  
  def __call__(self, dataset_dict):
    # print(dataset_dict)
    orig_img = cv2.imread(dataset_dict["file_name"])
    if 'LIVECell_dataset_2021' in dataset_dict["file_name"]:
      return orig_img, dataset_dict

    if np.random.uniform() < self.p:

      # Choose a sample to copy-paste from
      if self.paste_same_class:
        cls_id = dataset_dict['annotations'][0]['category_id']
        random_idx = np.random.randint(0, len(self.cls_indices[cls_id]))
        random_ds_dict = self.data[self.cls_indices[cls_id][random_idx]]
      else:
        random_idx = np.random.randint(0, self.n_samples)
        random_ds_dict = self.data[random_idx]

      # Load chosen sample
      random_img = cv2.imread(random_ds_dict['file_name'])
      if isinstance(self.paste_density, list):
        paste_density = np.random.uniform(self.paste_density[0], self.paste_density[1])
      else:
        paste_density = self.paste_density
      
      # Selection indices
      selected_cell_ids = np.random.choice(
        len(random_ds_dict['annotations']),
        size=round(paste_density * len(random_ds_dict['annotations'])),
        replace=False)

      # Select annotations (we deepcopy only selected ones, not the whole dict)
      selected_annos = [copy.deepcopy(random_ds_dict['annotations'][i])
                        for i in selected_cell_ids]
      copypaste_mask = mask_util.decode(selected_annos[0]['segmentation']).astype(np.bool)
      for anno in selected_annos[1:]:
        copypaste_mask |= mask_util.decode(anno['segmentation']).astype(np.bool)
      
      # Copy cells over
      neg_mask = ~copypaste_mask
      filtered_annos = []
      for anno in dataset_dict['annotations']:
        mask = mask_util.decode(anno['segmentation']).astype(np.bool)
        ocluded_mask = (mask & neg_mask)
        if (round(self.filter_area_thresh * mask.sum()) < ocluded_mask.sum()):
          anno['segmentation'] = mask_util.encode(np.asfortranarray(ocluded_mask))
          filtered_annos.append(anno)

      # Form output
      orig_img[copypaste_mask] = random_img[copypaste_mask]
      dataset_dict['annotations'] = filtered_annos + selected_annos

    return orig_img, dataset_dict