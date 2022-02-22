# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:33:54 2022

@author: Anders Emil Pedersen
@email: ande7941@gmail.com
"""

import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
import os

#dataset = foz.load_zoo_dataset("quickstart")
#session = fo.launch_app(dataset)
#%%
dataset = foz.load_zoo_dataset("coco-2017", split="train",max_samples=50,label_types=["segmentations"],classes=["cat", "dog"])

session = fo.launch_app(dataset)


#%%
# Give the dataset a classes list so it can be exported + imported
dataset.default_classes = dataset.distinct("ground_truth.detections.label")

# The directory in which the dataset's images are stored
IMAGES_DIR = os.path.dirname(dataset.first().filepath)

# Export some labels in COCO format
dataset.take(5).export(
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
    labels_path="/tmp/coco.json",
)

#%%

# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=IMAGES_DIR,
    labels_path="/tmp/coco.json",
    include_id=True,
    label_field="",
)

# Verify that the class list for our dataset was imported
print(coco_dataset.default_classes)  # ['airplane', 'apple', ...]

print(coco_dataset)