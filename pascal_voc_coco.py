# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
import pickle
import itertools
from collections import defaultdict
__all__ = ["load_voc_instances", "register_pascal_voc"]

VOC_CLASS_NAMES_COCOFIED = [
    "airplane",  "dining table", "motorcycle",
    "potted plant", "couch", "tv"
]

BASE_VOC_CLASS_NAMES = [
    "aeroplane", "diningtable", "motorbike",
    "pottedplant",  "sofa", "tvmonitor"
]


def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    
    UNKNOWN = True
    PREV_KNOWN = 40
    EXEMPLAR = False
    PSEUDO = False
    NUM_CLASSES = PREV_KNOWN + 20

    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
    # fileids is a string
    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    exemplar_set = set()
    if EXEMPLAR:
        with open("../PROB/data/VOC2007/ImageSets/Main/owod_t3_ft.txt") as fp:
        #with open("../PROB/data/VOC2007/ImageSets/Main/owod_t2_ft.txt") as fp:
            exemplar_files = fp.readlines()
        for ef in exemplar_files:
            exemplar_set.add(ef.rstrip())
    pseudo_file_set = set()
    if PSEUDO:
        with open("pseudo_files_set.pickle", "rb") as fp:
            pseudo_file_set = pickle.load(fp)
    exemplar_class_counts = defaultdict(int)
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # In the towod dataset creation, voc labels were converted to coco
            # In the class names tuple, the first 20 are the voc names
            if cls in VOC_CLASS_NAMES_COCOFIED:
                cls = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(cls)]
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            cid = class_names.index(cls)
            # 1 for unknown
            if UNKNOWN:
                if NUM_CLASSES > cid >= PREV_KNOWN:
                    instances.append(
                        {"category_id": 0, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                    )
            else:
                if cid < NUM_CLASSES:
                    if cid >= PREV_KNOWN or (cid < PREV_KNOWN and fileid in pseudo_file_set):
                        instances.append(
                            {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                        )
                    elif EXEMPLAR:
                        if cid < PREV_KNOWN and fileid in exemplar_set:
                            #print("fileid")
                            # exemplar_class_counts[cid] += 1
                            # if exemplar_class_counts[cid] <= 50:
                            #     #print("hit max for class", cid)
                            #     instances.append(
                            #         {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                            #     )
                            # else:
                            #     continue
                            instances.append(
                                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
                            )
        r["annotations"] = instances
        dicts.append(r)
        # returns filename which is the full filepath, image_id which is just a string,
        # image height, width, and bounding box annotations with category id, 4 coordinate bbox, and mode
    return dicts


def register_pascal_voc(name, dirname, split, year, class_names):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )


if __name__ == "__main__":
    dir = "~/hpc-share/omorim/projects/PROB/data/VOC2007"
    register_pascal_voc("towod_t1", dir, "train", 2007, VOC_COCO_CLASS_NAMES["TOWOD"])

