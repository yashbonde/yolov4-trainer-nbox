#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MIT License, packages used under respective licenses.
# __author__ = "Yash Bonde" yash@nimblebox.ai
#
# Script to train a yolo-v4 model on NBX-Jobs using nbox-sdk

import os
from types import SimpleNamespace
from torchvision.datasets import VOCDetection

from nbox import Operator
from nbox.utils import folder, join

# https://pytorch.org/vision/stable/_modules/torchvision/datasets/voc.html#VOCDetection
year_wise_fpaths = {
  "2012": join('VOCdevkit', 'VOC2012'),
  "2011": join('TrainVal', 'VOCdevkit', 'VOC2011'),
  "2010": join('VOCdevkit', 'VOC2010'),
  "2009": join('VOCdevkit', 'VOC2009'),
  "2008": join('VOCdevkit', 'VOC2008'),
  "2007": join('VOCdevkit', 'VOC2007'),
}

# you can define a custom operator that takes in the root folder and target folder,
# the files are downloaded using torchvision, converted to yolo format in the
# target folder. you start by writing config in __init__ and logic in forward

class CreateDataset(Operator):
  def __init__(self, root_folder, target_folder) -> None:
    os.makedirs(root_folder, exist_ok=False)
    os.makedirs(target_folder, exist_ok=False)

    self.root_folder = root_folder  # where the dataset is stored
    self.target_folder = target_folder  # where the processed is stored

  def _convert_tv_to_yolo_format(self, ds, save_folder) -> tuple:
    # create the target format that Yolo expects
    all_paths = []
    labels = {}
    for i in range(len(ds)):
      image, annot = ds[i] # (PIL, dict)
      image_path = join(save_folder, f"{i}.jpg")
      image.save(image_path)
      img_size = image.size
      f = open(image_path.replace(".jpg", ".txt"), "w")
      for i in annot["object"]:
        if i["name"] not in labels:
          labels[i["name"]] = len(labels)
        l = labels[i["name"]]
        xmin = i["bndbox"]["xmin"]
        ymin = i["bndbox"]["ymin"]
        xmax = i["bndbox"]["xmax"]
        ymax = i["bndbox"]["ymax"]
        w = xmax - xmin
        h = ymax - ymin
        x = (xmin + xmax) / 2 / img_size[0]
        y = (ymin + ymax) / 2 / img_size[1]
        w = w / img_size[0]
        h = h / img_size[1]
        f.write(f"{l} {x} {y} {w} {h}\n")
      f.close()
      all_paths.append(image_path)
    return all_paths, labels

  def forward(self, year = "2007") -> SimpleNamespace:
    if year not in year_wise_fpaths:
      raise ValueError(f"{year} is not a valid year, chose one of ({list(year_wise_fpaths.keys())})")

    if self.dry_run:
      # You can check if this is a dry run just like .training
      # here we will simply create the target structure of the dataset
      return

    # first download the dataset using torchvision
    _train = VOCDetection(
      root = self.root_folder,
      year = year,
      image_set = "train",
      download = True
    )
    _test = VOCDetection(
      root = self.root_folder,
      year = year,
      image_set = "val", # val is smaller than trainval
      download = True
    )

    # create image files and labels
    all_paths_train, labels = self._convert_tv_to_yolo_format(
      ds = _train,
      save_folder = join(self.target_folder, "train")
    )
    all_paths_test, _ = self._convert_tv_to_yolo_format(
      ds = _test,
      save_folder = join(self.target_folder, "test")
    )

    del _train, _test # free memory

    _train_path = join(self.target_folder, "train.txt")
    with open(_train_path, "w") as f:
      for path in all_paths_train:
        f.write(f"{path}\n")
    
    _test_path = join(self.target_folder, "test.txt")
    with open(_test_path, "w") as f:
      for path in all_paths_test:
        f.write(f"{path}\n")

    # create obj.names file
    _obj_names_path = join(self.target_folder, "obj.names")
    with open(_obj_names_path, "w") as f:
      for key in labels:
        f.write(f"{key}\n")

    # create obj.data file:
    _obj_data_path = join(self.target_folder, "obj.data")
    with open(_obj_data_path, "w") as f:
      f.write(f"classes={len(labels)}\n")
      f.write(f"train={_train_path}\n")
      f.write(f"valid={_test_path}\n")
      f.write(f"names={_obj_names_path}\n")
      f.write(f"backup={self.target_folder}/backup\n")

    return SimpleNamespace(
      train_path = _train_path,
      test_path = _test_path,
      obj_data_path = _obj_data_path,
      obj_names_path = _obj_names_path,
    )

# Operators are very versatile and we have already created a library of them
# nbox is open source and you can contribute if you build something cool
# goodies on the way!

from nbox.operators.lib import (
  GitClone,
  ShellCommand,
  Notify,
  NboxModelDeployOperator,
)


# Your entire job can be summed up in a single operator
# just like everything else, this is also just an Operator
# this op will run all the instructions and you can run
# this job

class Yolo(Operator):
  def __init__(self):
    """
    This module will do the following:

    1. Create datasets from the original VOC dataset
    2. Clone the YOLO repository
    3. Build the YOLO model and preload weights
    4. Train & Deploy the model
    """

    # this operation will create the following files:
    # ./data/yolo/train.txt
    # ./data/yolo/test.txt
    # ./data/yolo/obj.names
    # ./data/yolo/obj.data
    # ./data/yolo/train/
    # ./data/yolo/test/
    root_folder = join(folder(__file__), "data")
    target_folder = join(root_folder, "yolo")
    os.makedirs(root_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)
    self.prepare_dataset = CreateDataset(root_folder, target_folder)

    # clone the YOLO repository
    self.git_clone = GitClone("https://github.com/AlexeyAB/darknet")

    # execution on shell is most natural
    self.setup = ShellCommand(
      "sed -i 's/OPENCV=0/OPENCV=1/' ./darknet/Makefile"
      "sed -i 's/GPU=0/GPU=1/' ./darknet/Makefile"
      "sed -i 's/CUDNN=0/CUDNN=1/' ./darknet/Makefile"
      "sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' ./darknet/Makefile"
      "sed -i 's/LIBSO=0/LIBSO=1/' ./darknet/Makefile'"
    )
    self.make = ShellCommand('cd darknet/ & make')
    self.download_weights = ShellCommand(
      'wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 ' \
      '-O ./darknet/yolov4.conv.137'
    )

    # You can also templatise your code like this:
    self.train_shell = ShellCommand('''
./darknet/darknet detector train \
  {obj_data} \
  {cfg} \
  {backup} \
  -dont_show -map
'''.strip())

    # now deploy your model on managed k8s clusters (NBX-Deploy)
    self.deploy_model = NboxModelDeployOperator(
      model_name = "yolo",
      model_path = "darknet/darknet/cfg/yolo.cfg",
      model_weights = "darknet/darknet/backup/yolo.backup",
      model_labels = "darknet/darknet/data/obj.names",
    )

  def forward(self, model_name, slack_secret):
    # Now we execute the operations in order

    paths = self.prepare_dataset()
    self.git_cloner()
    self.setup()
    self.make()
    self.download_weights()
    self.train_shell(
      obj_data = join(folder(__file__), paths.obj_data_path),
      cfg = "cfg/yolov3.cfg",
      backup = "backup/yolov3.backup"
    )
    url, key = self.deploy_model(name = model_name)

    # you can call operators right in the middle of execution
    Notify(slack_secret = slack_secret)(
      f"Model '{model_name}' deployed at {url}, key: {key}"
    )

# ---------------
