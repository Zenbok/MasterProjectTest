# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:38:57 2022

@author: ande7
"""

import os

import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")

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
1#%%
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